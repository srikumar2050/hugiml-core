# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Downstream feature assembly and feature-budget selection."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.preprocessing import StandardScaler

from hugiml._classifier_runtime import _CORE_AVAILABLE, _core
from hugiml._classifier_support import (
    NativeAugmentedPairTransformBlock,
    _continuous_to_quantile_codes,
    _information_gain_from_codes,
)
from hugiml.exceptions import HUGIMLParamError, HUGIMLSchemaError, HUGIMLWarning


class _FeatureAssemblyMixin:
    """Implement downstream representation construction."""

    def _copy_input_for_downstream(self, X: Any) -> Any:
        """Preserve raw input before adaptive/pre-binning for hybrid modes.

        Downstream original-feature preparation is read-only.  For ndarray inputs
        the mining/pre-binning stages either consume X without mutation or bind a
        new pre-binned object to the local X_train variable, so a full eager copy
        here only adds O(n*p) time and memory.  DataFrames still get a shallow
        schema-stable copy because later preprocessing may add/reorder columns.
        """
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return np.asarray(X)

    def _pattern_order_from_label(self, label: str) -> int:
        """Infer pattern order from a human-readable HUG pattern label.

        .. deprecated::
            This method is retained for backward compatibility only.
            ``_setup_feature_mode_metadata`` derives pattern order directly
            from ``PatternEntry.items`` (the C++ item-ID list), which is the
            authoritative source of pattern length and is not affected by
            comma characters inside numeric interval notation such as
            ``age=[29.2, 38.4)``.

        The fallback parser intentionally counts feature assignments, not
        comma-separated chunks.  Numeric intervals contain commas, so a label
        like ``age=[29.2, 38.4)`` must remain order-1.  A conjunction such as
        ``age=[29,50), income=[50k,80k)`` is order-2 because it contains two
        top-level ``feature=...`` assignments.
        """
        import re

        s = str(label or "").strip()
        if not s:
            return 1

        # Fast path for native / ndarray labels such as
        # ``col0=[29.2, 38.4), col1=A``.  Count distinct column tokens rather
        # than commas so interval bounds do not inflate the order.
        col_matches = re.findall(r"\b(col\d+)\s*=", s)
        if col_matches:
            return max(1, len(set(col_matches)))

        # Human-readable labels are emitted as ``name=value`` assignments,
        # with conjunctions separated by either commas or explicit boolean
        # markers (``AND``, ``and`` or ``&``).  Count assignment starts that
        # occur at the beginning of the string or after one of those top-level
        # separators.  The feature-name pattern must start with a
        # letter/underscore, so commas inside numeric intervals, e.g.
        # ``[29.2, 38.4)``, are not mistaken for new assignments.
        assignment_matches = re.findall(
            r"(?:^|,|\s+(?:AND|and|&)\s+)\s*([A-Za-z_][A-Za-z0-9_ .:/\-]*)\s*=",
            s,
        )
        if assignment_matches:
            return max(1, len({name.strip() for name in assignment_matches if name.strip()}))

        # Last-resort fallback for unknown legacy formats: a label with an
        # explicit boolean conjunction marker is treated as an interaction;
        # otherwise keep the conservative singleton default.  Do not split on
        # commas here because interval labels contain commas.
        if re.search(r"\s+(?:AND|and|&)\s+", s):
            return 2
        return 1

    def _setup_feature_mode_metadata(self) -> None:
        """Cache pattern-order masks used by hybrid feature modes.

        Pattern order (number of features in a pattern) is read directly from
        ``PatternEntry.items`` — the C++ item-ID list — rather than inferred
        from the human-readable label string.  Label-string parsing
        (``_pattern_order_from_label``) mis-counts numeric singletons such as
        ``age=[29.2, 38.4)`` as order-2 because of the comma inside the
        interval notation, causing ``original_plus_interactions`` to
        incorrectly include numeric singletons in the downstream feature
        matrix. Using ``len(pe.items)`` gives the retained item count:
        1 for singletons, 2 for pair conjunctions, regardless of feature type
        or label format.
        """
        patterns = getattr(self, "patterns_", None)
        if hasattr(self, "x_train_hup_"):
            n_hup_cols = int(self.x_train_hup_.shape[1])
        elif getattr(self, "_training_pattern_matrix_shape_", None) is not None:
            n_hup_cols = int(self._training_pattern_matrix_shape_[1])
        elif patterns is not None:
            n_hup_cols = int(len(patterns))
        else:
            n_hup_cols = 0
        if patterns is not None and len(patterns) == n_hup_cols:
            # Primary path: read order from C++ PatternEntry.items directly.
            orders = np.asarray([len(pe.items) for pe in patterns], dtype=int)
        else:
            # Fallback: patterns_ unavailable or length mismatch — should not
            # occur after a completed fit, but guard defensively.
            features = self.get_hug_features()
            orders = np.asarray([self._pattern_order_from_label(f) for f in features], dtype=int)
        if len(orders) != n_hup_cols:
            orders = np.ones(n_hup_cols, dtype=int)
        self._pattern_orders_ = orders
        self._interaction_pattern_mask_ = orders > 1

    def _prepare_selected_original_features_for_downstream_transform(
        self, X: Any, selected_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Materialize only persisted selected original downstream columns at predict time.

        Fit still prepares the full original block once so scoring/serialization stay
        unchanged.  Prediction should not rebuild all original columns and then apply
        the fitted TopK mask: in original_plus_* modes the retained original columns
        are already known from ``_original_selected_feature_names_downstream_``.
        This helper constructs just those columns, preserving the exact fitted
        StandardScaler/median-imputation/dummy-column contract.
        """
        selected_names = list(selected_names or [])
        if not selected_names:
            n_rows = (
                len(X)
                if not isinstance(X, np.ndarray)
                else (1 if np.asarray(X).ndim == 1 else np.asarray(X).shape[0])
            )
            return self._empty_dense_block(n_rows), []

        selected_raw = [
            str(name)[5:] if str(name).startswith("orig:") else str(name) for name in selected_names
        ]
        num_cols = list(getattr(self, "_original_numeric_cols_", []))
        cat_cols = list(getattr(self, "_original_cat_cols_", []))
        dummy_cols = list(getattr(self, "_original_dummy_columns_", []))
        num_pos = {str(c): i for i, c in enumerate(num_cols)}
        dummy_set = {str(c) for c in dummy_cols}
        selected_numeric = [name for name in selected_raw if name in num_pos]
        selected_dummy = [name for name in selected_raw if name in dummy_set]

        train_names = list(getattr(self, "feature_names_in_", []) or [])
        train_pos = {str(c): i for i, c in enumerate(train_names)}
        is_df = isinstance(X, pd.DataFrame)
        arr = None if is_df else np.asarray(X)
        if arr is not None and arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n_rows = len(X) if is_df else int(arr.shape[0])

        blocks: list[np.ndarray] = []
        block_names: list[str] = []

        if selected_numeric:
            idx_in_num = np.asarray([num_pos[name] for name in selected_numeric], dtype=np.int64)
            if is_df:
                X_num_sel = pd.DataFrame(index=X.index)
                for name in selected_numeric:
                    X_num_sel[name] = X[name] if name in X.columns else np.nan
                raw_num = (
                    X_num_sel.apply(pd.to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .to_numpy(dtype=np.float64, copy=True)
                )
            else:
                assert arr is not None
                raw_num = np.empty((n_rows, len(selected_numeric)), dtype=np.float64)
                for out_j, name in enumerate(selected_numeric):
                    src_j = train_pos.get(name, out_j)
                    if 0 <= src_j < arr.shape[1]:
                        raw_num[:, out_j] = np.asarray(arr[:, src_j], dtype=np.float64)
                    else:
                        raw_num[:, out_j] = np.nan
                raw_num[~np.isfinite(raw_num)] = np.nan

            med_arr = getattr(self, "_original_numeric_medians_array_", None)
            if med_arr is None or len(med_arr) != len(num_cols):
                med = getattr(self, "_original_numeric_medians_", pd.Series(dtype=float))
                med_arr = med.reindex(num_cols).fillna(0.0).to_numpy(dtype=np.float64, copy=True)
            med_sel = np.asarray(med_arr, dtype=np.float64)[idx_in_num]
            bad = ~np.isfinite(raw_num)
            if bad.any():
                raw_num[bad] = np.take(med_sel, np.nonzero(bad)[1])

            scaler = self._original_scaler_
            mean = np.asarray(getattr(scaler, "mean_", np.zeros(len(num_cols))), dtype=np.float64)[
                idx_in_num
            ]
            scale = np.asarray(getattr(scaler, "scale_", np.ones(len(num_cols))), dtype=np.float64)[
                idx_in_num
            ]
            scale = np.where(scale == 0.0, 1.0, scale)
            blocks.append(((raw_num - mean) / scale).astype(np.float32, copy=False))
            block_names.extend([f"orig:{name}" for name in selected_numeric])

        if selected_dummy:
            if is_df:
                X_cat = X.reindex(columns=cat_cols)
            else:
                assert arr is not None
                data = {}
                for name in cat_cols:
                    src_j = train_pos.get(str(name), None)
                    if src_j is not None and 0 <= src_j < arr.shape[1]:
                        data[name] = arr[:, src_j]
                    else:
                        data[name] = np.full(n_rows, np.nan, dtype=object)
                X_cat = pd.DataFrame(data)
            X_cat_str = X_cat.astype("string") if len(cat_cols) else X_cat
            if len(cat_cols):
                # Collapse category values outside selected_dummy before one-hot
                # encoding, so get_dummies expands only the downstream levels
                # retained during fitting rather than every distinct value in
                # the prediction batch.
                #
                # Collapse to a sentinel value, not to missing/NaN. A valid but
                # unselected category is semantically different from a missing
                # cell, especially when a column's "_<NA>" dummy survived feature
                # selection. The sentinel-generated dummy is removed by the
                # reindex below, leaving all selected dummy columns at zero for
                # that row.
                #
                # This is vectorized per column: construct candidate dummy names
                # once and test membership against the fitted selected set.
                dummy_set_local = set(selected_dummy)
                _COLLAPSE_SENTINEL = "\ue000__hugiml_other__"
                for col in cat_cols:
                    candidate_names = col + "_" + X_cat_str[col]
                    not_keep = X_cat_str[col].notna() & ~candidate_names.isin(dummy_set_local)
                    if not_keep.any():
                        X_cat_str.loc[not_keep, col] = _COLLAPSE_SENTINEL
            X_cat_dum = (
                pd.get_dummies(X_cat_str, dummy_na=True)
                if len(cat_cols)
                else pd.DataFrame(index=range(n_rows))
            )
            X_cat_dum = X_cat_dum.reindex(columns=selected_dummy, fill_value=0)
            blocks.append(X_cat_dum.to_numpy(dtype=np.float32, copy=False))
            block_names.extend([f"orig:{name}" for name in selected_dummy])

        # Preserve fitted selected_names order even when numeric and dummy columns
        # are interleaved.  The two blocks above are built by type for speed; this
        # final gather restores the exact downstream coefficient alignment.
        if not blocks:
            return self._empty_dense_block(n_rows), []
        by_name = {}
        dense_concat = (
            np.hstack(blocks).astype(np.float32, copy=False) if len(blocks) > 1 else blocks[0]
        )
        for j, name in enumerate(block_names):
            by_name[name] = dense_concat[:, j]
        missing_selected = [name for name in selected_names if name not in by_name]
        if missing_selected:
            raise HUGIMLSchemaError(
                "Selected original downstream feature(s) are unavailable during transform: "
                f"{missing_selected[:10]!r}. This usually indicates schema drift or a "
                "model/metadata mismatch. Refit the model or provide input columns "
                "matching the training schema."
            )
        out = np.empty((n_rows, len(selected_names)), dtype=np.float32)
        for j, name in enumerate(selected_names):
            out[:, j] = by_name[name]
        return out, list(selected_names)

    def _prepare_original_features_for_downstream(self, X: Any, fit: bool = False):
        """Prepare original input features for hybrid downstream estimators.

        This intentionally does not affect transform(), get_hug_features(), or
        any pattern diagnostics.  It is used only by predict/fit when
        feature_mode includes original features.
        """
        # Fast all-numeric ndarray fit/transform path.  Avoid constructing a
        # DataFrame and running pandas apply/to_numeric over every column for the
        # common large-n benchmark path.  This preserves the exact fitted
        # StandardScaler/median-imputation/original feature-name contract.
        if not isinstance(X, pd.DataFrame):
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            names = getattr(self, "feature_names_in_", None)
            if names is None or len(names) != arr.shape[1]:
                names = [f"col{j}" for j in range(arr.shape[1])]
            cat_mask = getattr(self, "cat_cols_mask_", None)
            if cat_mask is not None and not bool(np.any(cat_mask)):
                raw_dtype = np.float32 if getattr(arr, "dtype", None) == np.float32 else np.float64
                raw = np.array(arr, dtype=raw_dtype, copy=True, order="C")
                raw[~np.isfinite(raw)] = np.nan
                if fit:
                    self._original_numeric_cols_ = list(names)
                    self._original_cat_cols_ = []
                    med_arr = (
                        np.nanmedian(raw, axis=0) if raw.shape[1] else np.empty(0, dtype=raw.dtype)
                    )
                    med_arr = np.where(np.isfinite(med_arr), med_arr, 0.0).astype(
                        raw.dtype, copy=False
                    )
                    self._original_numeric_medians_array_ = med_arr.astype(np.float64, copy=True)
                    self._original_numeric_medians_ = pd.Series(med_arr, index=list(names))
                    bad = ~np.isfinite(raw)
                    if bad.any():
                        raw[bad] = np.take(med_arr, np.nonzero(bad)[1])
                    self._original_scaler_ = StandardScaler()
                    X_num_arr = (
                        self._original_scaler_.fit_transform(raw)
                        if raw.shape[1]
                        else np.empty((raw.shape[0], 0))
                    )
                    self._original_dummy_columns_ = []
                    self._original_feature_names_downstream_ = list(names)
                    return X_num_arr.astype(np.float32, copy=False)
                num_cols = list(getattr(self, "_original_numeric_cols_", []))
                if (
                    num_cols
                    and list(names) == num_cols
                    and not getattr(self, "_original_cat_cols_", [])
                ):
                    med_arr = getattr(self, "_original_numeric_medians_array_", None)
                    if med_arr is None or len(med_arr) != raw.shape[1]:
                        med = getattr(self, "_original_numeric_medians_", pd.Series(dtype=float))
                        med_arr = (
                            med.reindex(num_cols).fillna(0.0).to_numpy(dtype=raw.dtype, copy=True)
                        )
                    else:
                        med_arr = np.asarray(med_arr, dtype=raw.dtype)
                    bad = ~np.isfinite(raw)
                    if bad.any():
                        raw[bad] = np.take(med_arr, np.nonzero(bad)[1])
                    X_num_arr = self._original_scaler_.transform(raw)
                    return X_num_arr.astype(np.float32, copy=False)

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            names = getattr(self, "feature_names_in_", None)
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if names is None or len(names) != arr.shape[1]:
                names = [f"col{j}" for j in range(arr.shape[1])]
            X_df = pd.DataFrame(arr, columns=list(names))

        # Stabilize column order against training schema when available.
        train_names = getattr(self, "feature_names_in_", None)
        if train_names is not None:
            for col in train_names:
                if col not in X_df.columns:
                    X_df[col] = np.nan
            X_df = X_df[list(train_names)]

        # Hot predict path: fitted all-numeric original_plus_* models can avoid
        # pandas apply/fillna/reindex/get_dummies.  This preserves the fitted
        # StandardScaler and median-imputation contract exactly.
        if not fit:
            num_cols = list(getattr(self, "_original_numeric_cols_", []))
            cat_cols = list(getattr(self, "_original_cat_cols_", []))
            dummy_cols = list(getattr(self, "_original_dummy_columns_", []))
            if num_cols and not cat_cols and not dummy_cols and list(X_df.columns) == num_cols:
                try:
                    X_num_arr_raw = X_df.to_numpy(dtype=np.float64, copy=True)
                    med_arr = getattr(self, "_original_numeric_medians_array_", None)
                    if med_arr is None or len(med_arr) != X_num_arr_raw.shape[1]:
                        med = getattr(self, "_original_numeric_medians_", pd.Series(dtype=float))
                        med_arr = (
                            med.reindex(num_cols).fillna(0.0).to_numpy(dtype=np.float64, copy=True)
                        )
                    bad = ~np.isfinite(X_num_arr_raw)
                    if bad.any():
                        X_num_arr_raw[bad] = np.take(med_arr, np.nonzero(bad)[1])
                    X_num_arr = self._original_scaler_.transform(X_num_arr_raw)
                    return X_num_arr.astype(np.float32, copy=False)
                except Exception:
                    pass

        # Numeric columns are scaled; categorical columns, including
        # numeric-looking binary features inferred during fit, are one-hot encoded.
        numeric = X_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        cat_mask = getattr(self, "cat_cols_mask_", None)
        cat_by_mask = set()
        if cat_mask is not None:
            cat_by_mask = {
                str(c) for c, is_cat in zip(X_df.columns.astype(str), cat_mask) if bool(is_cat)
            }
        numeric_cols = [
            c for c in X_df.columns if str(c) not in cat_by_mask and not numeric[c].isna().all()
        ]
        X_num = numeric[numeric_cols] if numeric_cols else pd.DataFrame(index=X_df.index)
        X_cat = X_df.drop(columns=numeric_cols, errors="ignore")

        if fit:
            self._original_numeric_cols_ = list(X_num.columns)
            self._original_cat_cols_ = list(X_cat.columns)
            self._original_numeric_medians_ = X_num.median(numeric_only=True).fillna(0.0)
            self._original_numeric_medians_array_ = self._original_numeric_medians_.reindex(
                self._original_numeric_cols_
            ).to_numpy(dtype=np.float64, copy=True)
            X_num_filled = X_num.fillna(self._original_numeric_medians_)
            self._original_scaler_ = StandardScaler()
            X_num_arr = (
                self._original_scaler_.fit_transform(
                    X_num_filled.to_numpy(dtype=np.float64, copy=False)
                )
                if len(self._original_numeric_cols_)
                else np.empty((len(X_df), 0))
            )
            X_cat_dum = (
                pd.get_dummies(X_cat.astype("string"), dummy_na=True)
                if len(self._original_cat_cols_)
                else pd.DataFrame(index=X_df.index)
            )
            self._original_dummy_columns_ = list(X_cat_dum.columns)
        else:
            num_cols = getattr(self, "_original_numeric_cols_", [])
            med = getattr(self, "_original_numeric_medians_", pd.Series(dtype=float))
            X_num = numeric.reindex(columns=num_cols)
            X_num_filled = X_num.fillna(med).fillna(0.0)
            if len(num_cols):
                X_num_arr = self._original_scaler_.transform(
                    X_num_filled.to_numpy(dtype=np.float64, copy=False)
                )
            else:
                X_num_arr = np.empty((len(X_df), 0))
            cat_cols = getattr(self, "_original_cat_cols_", [])
            X_cat = X_df.reindex(columns=cat_cols)
            dummy_cols = getattr(self, "_original_dummy_columns_", [])
            X_cat_str = X_cat.astype("string") if len(cat_cols) else X_cat
            if len(cat_cols):
                # Same sentinel approach as the selected-original transform.
                dummy_set_local = set(dummy_cols)
                _COLLAPSE_SENTINEL = "\ue000__hugiml_other__"
                for col in cat_cols:
                    candidate_names = col + "_" + X_cat_str[col]
                    not_keep = X_cat_str[col].notna() & ~candidate_names.isin(dummy_set_local)
                    if not_keep.any():
                        X_cat_str.loc[not_keep, col] = _COLLAPSE_SENTINEL
            X_cat_dum = (
                pd.get_dummies(X_cat_str, dummy_na=True)
                if len(cat_cols)
                else pd.DataFrame(index=X_df.index)
            )
            X_cat_dum = X_cat_dum.reindex(columns=dummy_cols, fill_value=0)

        X_cat_arr = (
            X_cat_dum.to_numpy(dtype=np.float64, copy=False)
            if X_cat_dum.shape[1]
            else np.empty((len(X_df), 0))
        )
        X_base = np.hstack([X_num_arr, X_cat_arr]) if X_cat_arr.shape[1] else X_num_arr
        if fit:
            self._original_feature_names_downstream_ = list(
                getattr(self, "_original_numeric_cols_", [])
            ) + list(getattr(self, "_original_dummy_columns_", []))
        return X_base.astype(np.float32, copy=False)

    def _as_dense_float32(self, X: Any) -> np.ndarray:
        """Return a dense float32 2-D array without changing estimator semantics."""
        arr = X.toarray() if issparse(X) else np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr.astype(np.float32, copy=False)

    def _empty_dense_block(self, n_rows: int) -> np.ndarray:
        return np.empty((int(n_rows), 0), dtype=np.float32)

    def _original_topk_budget_enabled(self) -> bool:
        """Whether original features should be pre-budgeted before downstream fit."""
        return self.topK is not None and int(self.topK) >= 0

    def _select_original_topk_fit(
        self, X_base: Any, y: Any, names: list[str]
    ) -> tuple[Any, list[str]]:
        """Select at most topK original downstream columns and persist the mask.

        The mask is applied before concatenating originals with pattern blocks.
        This makes original_plus_* non-strict mode symmetric with mined and
        augmented features: originals contribute at most topK columns when a
        finite topK budget is configured.  Strict mode then performs the global
        topK pass over this already-budgeted candidate set.
        """
        n_cols = len(names)
        self._original_feature_names_downstream_full_ = list(names)
        self._original_feature_scores_downstream_ = np.zeros(n_cols, dtype=np.float64)
        self._original_feature_mask_downstream_ = np.ones(n_cols, dtype=bool)
        self._original_selected_feature_names_downstream_ = list(names)
        if n_cols == 0 or not self._original_topk_budget_enabled():
            return X_base, list(names)
        budget = min(max(1, int(self.topK)), n_cols)
        if budget >= n_cols:
            native_scores = getattr(self, "_native_original_feature_scores_downstream_", None)
            if native_scores is not None and len(native_scores) == n_cols:
                self._original_feature_scores_downstream_ = np.asarray(
                    native_scores, dtype=np.float64
                )
            return X_base, list(names)

        native_names = list(getattr(self, "_native_original_feature_names_downstream_", []) or [])
        native_scores = getattr(self, "_native_original_feature_scores_downstream_", None)
        if (
            native_names == list(names)
            and native_scores is not None
            and len(native_scores) == n_cols
        ):
            scores = np.asarray(native_scores, dtype=np.float64)
            order = np.lexsort((np.arange(n_cols), -scores))
            keep_idx = np.sort(order[:budget])
            mask = np.zeros(n_cols, dtype=bool)
            mask[keep_idx] = True
        else:
            # Non-fused or schema-mismatch path: still native, but necessarily
            # uses the dense downstream block because no preparation-stage score
            # metadata is available for this fit.
            scores, mask = self._strict_topk_dense_column_scores(X_base, y, names, top_k=budget)

        selected_names = [name for name, keep in zip(names, mask) if keep]
        self._original_feature_scores_downstream_ = scores
        self._original_feature_mask_downstream_ = mask
        self._original_selected_feature_names_downstream_ = list(selected_names)
        return X_base[:, mask], selected_names

    def _select_original_topk_transform(
        self, X_base: Any, names: list[str]
    ) -> tuple[Any, list[str]]:
        """Apply the persisted original-feature prefilter at transform time."""
        mask = getattr(self, "_original_feature_mask_downstream_", None)
        selected_names = getattr(self, "_original_selected_feature_names_downstream_", None)
        if mask is None:
            return X_base, list(names)
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.size != len(names):
            return X_base, list(names)
        if bool(np.all(mask_arr)):
            return X_base, list(names if selected_names is None else selected_names)
        return X_base[:, mask_arr], list(names if selected_names is None else selected_names)

    def _selected_original_downstream_names(self) -> list[str]:
        names = getattr(self, "_original_selected_feature_names_downstream_", None)
        if names is not None:
            return list(names)
        return [f"orig:{name}" for name in getattr(self, "_original_feature_names_downstream_", [])]

    def _strict_topk_dense_column_scores(
        self, X: Any, y: Any, names: list[str], top_k: int = -1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score dense downstream columns and return ``(scores, mask)``.

        Prefer the native dense TopK helper when the extension exposes it,
        but fall back to the same Python IG scoring used by the sparse path.
        This preserves correctness for hybrid feature modes with
        ``topk_budget_strict=True``.
        """
        X_arr = np.ascontiguousarray(self._as_dense_float32(X), dtype=np.float32)
        n_cols = int(X_arr.shape[1])
        if n_cols == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool)
        y_codes, _ = pd.factorize(np.asarray(y), sort=True)
        y_codes = np.ascontiguousarray(y_codes.astype(np.int64, copy=False))
        discrete_mask_bool = self._strict_topk_discrete_mask(names).astype(bool, copy=False)
        max_bins = max(
            8,
            int(
                getattr(self, "B", 8) if getattr(self, "B", 8) and getattr(self, "B", 8) > 0 else 16
            ),
        )
        if _CORE_AVAILABLE and hasattr(_core, "strict_topk_filter_dense"):
            discrete_mask = np.ascontiguousarray(discrete_mask_bool.astype(np.uint8, copy=False))
            scores, mask = _core.strict_topk_filter_dense(
                X_arr, y_codes, discrete_mask, int(top_k), int(max_bins)
            )
            return np.asarray(scores, dtype=np.float64), np.asarray(mask, dtype=bool)

        n_classes = int(np.max(y_codes)) + 1 if y_codes.size else 0
        scores = np.zeros(n_cols, dtype=np.float64)
        if n_classes > 1:
            for j in range(n_cols):
                col = np.asarray(X_arr[:, j], dtype=np.float64)
                if bool(discrete_mask_bool[j]):
                    vals = np.where(col > 0.5, 1, 0).astype(np.int64, copy=False)
                else:
                    vals = _continuous_to_quantile_codes(col, max_bins=max_bins)
                scores[j] = _information_gain_from_codes(vals, y_codes, n_classes)

        budget = int(top_k)
        mask = np.ones(n_cols, dtype=bool)
        if budget >= 0 and budget < n_cols:
            budget = max(1, budget)
            order = np.lexsort((np.arange(n_cols), -scores))
            keep_idx = np.sort(order[:budget])
            mask = np.zeros(n_cols, dtype=bool)
            mask[keep_idx] = True
        return scores, mask

    def _combine_dense_blocks(self, blocks: list[Any], n_rows: int) -> np.ndarray:
        """Combine selected downstream blocks as one dense float32 matrix."""
        dense_blocks = []
        for block in blocks:
            if block is None or int(getattr(block, "shape", (n_rows, 0))[1]) == 0:
                continue
            dense_blocks.append(self._as_dense_float32(block))
        if not dense_blocks:
            return self._empty_dense_block(n_rows)
        if len(dense_blocks) == 1:
            return dense_blocks[0]
        return np.hstack(dense_blocks).astype(np.float32, copy=False)

    def _dense_downstream_width_threshold(self) -> int:
        """Width threshold for materializing hybrid downstream matrices as dense.

        Hybrid feature modes contain dense original columns and sparse pattern
        columns.  Dense output is fastest and smallest for small selected widths
        because LR sees a compact float32 ndarray.  For large selected widths,
        densifying sparse pattern columns wastes memory; keep the output CSR.

        The threshold is user-configurable via the sklearn parameter
        ``dense_downstream_max_width``.  Set it to 0 to keep hybrid matrices CSR
        except for the empty-width degenerate case.  Older loaded models that
        lack the public parameter can still fall back to the private compatibility
        attribute or the historical default of 200.
        """
        try:
            value = getattr(
                self,
                "dense_downstream_max_width",
                getattr(self, "_dense_downstream_max_width_", 200),
            )
            if isinstance(value, bool):
                return 200
            return max(0, int(value))
        except Exception:
            return 200

    def _should_use_dense_downstream(self, total_width: int) -> bool:
        mode = getattr(self, "feature_mode", "patterns_only")
        if mode == "patterns_only":
            return False
        threshold = self._dense_downstream_width_threshold()
        return int(total_width) <= int(threshold)

    def _combine_downstream_blocks(self, blocks: list[Any], n_rows: int):
        """Combine downstream blocks using a memory-aware output format.

        * patterns_only is handled earlier and remains CSR.
        * hybrid modes use dense float32 for small/moderate widths, avoiding
          sparse->dense churn before LR.
        * hybrid modes use CSR once width is large enough that densifying sparse
          pattern/augmented blocks would dominate memory.
        """
        live_blocks = [
            block
            for block in blocks
            if block is not None and int(getattr(block, "shape", (n_rows, 0))[1]) > 0
        ]
        total_width = sum(int(block.shape[1]) for block in live_blocks)
        if total_width == 0:
            return (
                self._empty_dense_block(n_rows)
                if self._should_use_dense_downstream(0)
                else csr_matrix((int(n_rows), 0), dtype=np.float32)
            )
        if self._should_use_dense_downstream(total_width):
            return self._combine_dense_blocks(live_blocks, n_rows)
        sparse_blocks = []
        for block in live_blocks:
            if issparse(block):
                sparse_blocks.append(block.astype(np.float32, copy=False).tocsr())
            else:
                sparse_blocks.append(csr_matrix(np.asarray(block, dtype=np.float32)))
        if len(sparse_blocks) == 1:
            return sparse_blocks[0]
        return hstack(sparse_blocks, format="csr", dtype=np.float32)

    def _select_strict_topk_from_blocks_fit(
        self,
        blocks: list[tuple[str, Any, list[str]]],
        y: Any,
    ) -> tuple[list[Any], list[str]]:
        """Apply strict global TopK before concatenating downstream feature blocks.

        This keeps strict mode as a compute budget: dense original columns are
        scored directly, sparse pattern columns are scored in sparse form, and
        only selected columns are materialized for the downstream estimator.
        The persisted full mask and scores retain the same public/serialization
        contract as the prior post-concatenation implementation.
        """
        full_names: list[str] = []
        full_scores_parts: list[np.ndarray] = []
        selected_blocks: list[Any] = []

        for _, block, names in blocks:
            full_names.extend(list(names))
        n_cols = len(full_names)
        self._downstream_feature_names_full_ = list(full_names)
        self._strict_topk_applied_during_construction_ = False
        self._strict_topk_feature_scores_ = np.zeros(n_cols, dtype=np.float64)
        self._strict_topk_feature_mask_ = np.ones(n_cols, dtype=bool)
        self._strict_topk_selected_feature_names_ = list(full_names)

        if (
            not bool(getattr(self, "topk_budget_strict", False))
            or self.topK is None
            or int(self.topK) < 0
            or n_cols == 0
            or int(self.topK) >= n_cols
        ):
            return [block for _, block, _ in blocks], list(full_names)

        for kind, block, names in blocks:
            if len(names) == 0:
                full_scores_parts.append(np.zeros(0, dtype=np.float64))
            elif kind == "dense":
                full_scores_parts.append(
                    self._strict_topk_dense_column_scores(block, y, names, top_k=-1)[0]
                )
            else:
                full_scores_parts.append(self._strict_topk_column_scores(block, y, names))
        scores = (
            np.concatenate(full_scores_parts).astype(np.float64, copy=False)
            if full_scores_parts
            else np.zeros(0, dtype=np.float64)
        )
        budget = min(max(1, int(self.topK)), n_cols)
        order = np.lexsort((np.arange(n_cols), -scores))
        keep_idx = np.sort(order[:budget])
        mask = np.zeros(n_cols, dtype=bool)
        mask[keep_idx] = True

        offset = 0
        selected_names: list[str] = []
        for kind, block, names in blocks:
            width = len(names)
            block_mask = mask[offset : offset + width]
            if width and bool(np.any(block_mask)):
                selected_names.extend([name for name, keep in zip(names, block_mask) if keep])
                selected_blocks.append(block[:, block_mask])
            offset += width

        self._strict_topk_feature_scores_ = scores
        self._strict_topk_feature_mask_ = mask
        self._strict_topk_selected_feature_names_ = list(selected_names)
        self._strict_topk_applied_during_construction_ = True
        return selected_blocks, selected_names

    def _select_strict_topk_from_blocks_transform(
        self, blocks: list[tuple[str, Any, list[str]]]
    ) -> list[Any]:
        """Select persisted strict TopK columns from logical blocks before concat."""
        mask = getattr(self, "_strict_topk_feature_mask_", None)
        if mask is None:
            return [block for _, block, _ in blocks]
        mask_arr = np.asarray(mask, dtype=bool)
        full_width = sum(len(names) for _, _, names in blocks)
        if mask_arr.size != full_width or bool(np.all(mask_arr)):
            return [block for _, block, _ in blocks]
        selected_blocks: list[Any] = []
        offset = 0
        for _, block, names in blocks:
            width = len(names)
            block_mask = mask_arr[offset : offset + width]
            if width and bool(np.any(block_mask)):
                selected_blocks.append(block[:, block_mask])
            offset += width
        return selected_blocks

    def _make_downstream_features(self, X_original: Any, Z_patterns: csr_matrix, fit: bool = False):
        """Build the estimator input matrix for the configured feature_mode."""
        mode = getattr(self, "feature_mode", "patterns_only")
        Z = Z_patterns if issparse(Z_patterns) else csr_matrix(Z_patterns)
        Z_aug = self._make_augmented_pair_features(X_original, fit=fit)
        n_rows = int(Z.shape[0])
        if mode == "patterns_only":
            return hstack([Z, Z_aug], format="csr") if Z_aug.shape[1] else Z

        if not fit:
            selected_original_names = getattr(
                self, "_original_selected_feature_names_downstream_", None
            )
            full_original_names = [
                f"orig:{name}" for name in getattr(self, "_original_feature_names_downstream_", [])
            ]
            mask = getattr(self, "_original_feature_mask_downstream_", None)
            use_selected_originals = (
                selected_original_names is not None
                and mask is not None
                and len(selected_original_names) <= len(full_original_names)
            )
            if use_selected_originals:
                X_base, original_names = (
                    self._prepare_selected_original_features_for_downstream_transform(
                        X_original, list(selected_original_names)
                    )
                )
            else:
                X_base = self._prepare_original_features_for_downstream(X_original, fit=False)
                X_base, original_names = self._select_original_topk_transform(
                    X_base, full_original_names
                )
        else:
            X_base = self._prepare_original_features_for_downstream(X_original, fit=True)
            original_names_full = [
                f"orig:{name}" for name in getattr(self, "_original_feature_names_downstream_", [])
            ]
            X_base, original_names = self._select_original_topk_fit(
                X_base, self._current_y_for_downstream_topk_, original_names_full
            )
        pattern_names = [f"pattern:{name}" for name in self.get_hug_features()]
        aug_names = [
            f"augmented_pair:{t['name']}" for t in getattr(self, "augmented_pair_transforms_", [])
        ]

        if mode == "original_plus_patterns":
            blocks = [("dense", X_base, original_names), ("sparse", Z, pattern_names)]
            if Z_aug.shape[1]:
                blocks.append(("sparse", Z_aug, aug_names))
            if fit and bool(getattr(self, "topk_budget_strict", False)):
                selected_blocks, _ = self._select_strict_topk_from_blocks_fit(
                    blocks, self._current_y_for_downstream_topk_
                )
                return self._combine_downstream_blocks(selected_blocks, n_rows)
            if bool(getattr(self, "topk_budget_strict", False)):
                selected_blocks = self._select_strict_topk_from_blocks_transform(blocks)
                return self._combine_downstream_blocks(selected_blocks, n_rows)
            return self._combine_downstream_blocks([X_base, Z, Z_aug], n_rows)

        if mode == "original_plus_interactions":
            mask = getattr(self, "_interaction_pattern_mask_", None)
            if mask is None:
                self._setup_feature_mode_metadata()
                mask = self._interaction_pattern_mask_
            Z_sel = Z[:, mask]
            selected_pattern_names = [name for name, keep in zip(pattern_names, mask) if keep]
            blocks = [("dense", X_base, original_names), ("sparse", Z_sel, selected_pattern_names)]
            if Z_aug.shape[1]:
                blocks.append(("sparse", Z_aug, aug_names))
            if fit and bool(getattr(self, "topk_budget_strict", False)):
                selected_blocks, _ = self._select_strict_topk_from_blocks_fit(
                    blocks, self._current_y_for_downstream_topk_
                )
                return self._combine_downstream_blocks(selected_blocks, n_rows)
            if bool(getattr(self, "topk_budget_strict", False)):
                selected_blocks = self._select_strict_topk_from_blocks_transform(blocks)
                return self._combine_downstream_blocks(selected_blocks, n_rows)
            return self._combine_downstream_blocks([X_base, Z_sel, Z_aug], n_rows)

        raise HUGIMLParamError(f"Unknown feature_mode={mode!r}.")

    def _get_downstream_feature_names_full(self) -> list[str]:
        """Names for the unfiltered downstream feature matrix."""
        mode = getattr(self, "feature_mode", "patterns_only")
        pattern_names = list(self.get_hug_features())
        aug_names = [
            f"augmented_pair:{t['name']}" for t in getattr(self, "augmented_pair_transforms_", [])
        ]
        if mode == "patterns_only":
            return [f"pattern:{name}" for name in pattern_names] + aug_names
        original_names = self._selected_original_downstream_names()
        if mode == "original_plus_patterns":
            return original_names + [f"pattern:{name}" for name in pattern_names] + aug_names
        if mode == "original_plus_interactions":
            mask = getattr(
                self, "_interaction_pattern_mask_", np.ones(len(pattern_names), dtype=bool)
            )
            selected = [name for name, keep in zip(pattern_names, mask) if keep]
            return original_names + [f"pattern:{name}" for name in selected] + aug_names
        return [f"pattern:{name}" for name in pattern_names]

    def _get_downstream_feature_names(self) -> list[str]:
        """Names aligned with coefficients of the downstream estimator."""
        names = self._get_downstream_feature_names_full()
        mask = getattr(self, "_strict_topk_feature_mask_", None)
        if mask is None:
            return names
        return [name for name, keep in zip(names, np.asarray(mask, dtype=bool)) if keep]

    def _is_discrete_downstream_feature(self, name: str) -> bool:
        return name.startswith("pattern:") or (
            name.startswith("orig:") and name in getattr(self, "_strict_topk_dummy_names_", set())
        )

    def _strict_topk_discrete_mask(self, names: list[str]) -> np.ndarray:
        """Boolean mask of downstream columns that should be IG-scored as discrete."""
        dummy_names = {f"orig:{c}" for c in getattr(self, "_original_dummy_columns_", [])}
        self._strict_topk_dummy_names_ = dummy_names
        return np.asarray(
            [name.startswith("pattern:") or name in dummy_names for name in names],
            dtype=np.uint8,
        )

    def _strict_topk_column_scores(self, X: csr_matrix, y: Any, names: list[str]) -> np.ndarray:
        """Compute comparable IG scores for strict global topK filtering.

        The native path scores CSC columns directly. The Python fallback is kept
        only for source-tree development before the extension has been rebuilt.
        """
        y_codes, _ = pd.factorize(np.asarray(y), sort=True)
        n_classes = int(np.max(y_codes)) + 1 if y_codes.size else 0
        if n_classes <= 1:
            return np.zeros(X.shape[1], dtype=np.float64)
        X_csc = X.tocsc() if issparse(X) else csr_matrix(X).tocsc()
        discrete_mask = self._strict_topk_discrete_mask(names)
        max_bins = max(
            8,
            int(
                getattr(self, "B", 8) if getattr(self, "B", 8) and getattr(self, "B", 8) > 0 else 16
            ),
        )
        if _CORE_AVAILABLE and hasattr(_core, "strict_topk_filter_csc"):
            scores, _ = _core.strict_topk_filter_csc(
                np.asarray(X_csc.data, dtype=np.float32),
                np.asarray(X_csc.indices, dtype=np.int32),
                np.asarray(X_csc.indptr, dtype=np.int32),
                int(X_csc.shape[0]),
                int(X_csc.shape[1]),
                np.asarray(y_codes, dtype=np.int64),
                discrete_mask,
                -1,
                int(max_bins),
            )
            return np.asarray(scores, dtype=np.float64)
        scores = np.zeros(X.shape[1], dtype=np.float64)
        for j in range(X.shape[1]):
            col = np.asarray(X_csc[:, j].toarray()).ravel()
            if bool(discrete_mask[j]):
                vals = np.where(col > 0.5, 1, 0).astype(np.int64, copy=False)
            else:
                vals = _continuous_to_quantile_codes(col, max_bins=max_bins)
            scores[j] = _information_gain_from_codes(vals, y_codes, n_classes)
        return scores

    def _apply_strict_topk_budget_fit(self, X: csr_matrix, y: Any) -> csr_matrix:
        """Optionally apply a single native global IG topK budget over all downstream features."""
        n_cols = int(X.shape[1])
        names = self._get_downstream_feature_names_full()
        existing_mask = getattr(self, "_strict_topk_feature_mask_", None)
        if bool(getattr(self, "_strict_topk_applied_during_construction_", False)):
            return X
        if (
            bool(getattr(self, "topk_budget_strict", False))
            and existing_mask is not None
            and len(existing_mask) != n_cols
        ):
            # Strict TopK was already applied during block-wise downstream
            # construction.  Preserve the full-length persisted mask/scores and
            # return the selected estimator matrix unchanged.
            return X
        self._downstream_feature_names_full_ = list(names)
        self._strict_topk_feature_scores_ = np.zeros(n_cols, dtype=np.float64)
        self._strict_topk_feature_mask_ = np.ones(n_cols, dtype=bool)
        self._strict_topk_selected_feature_names_ = list(names)
        if (
            not bool(getattr(self, "topk_budget_strict", False))
            or self.topK is None
            or int(self.topK) < 0
            or n_cols == 0
            or int(self.topK) >= n_cols
        ):
            return X
        budget = min(max(1, int(self.topK)), n_cols)
        y_codes, _ = pd.factorize(np.asarray(y), sort=True)
        X_csc = X.tocsc() if issparse(X) else csr_matrix(X).tocsc()
        discrete_mask = self._strict_topk_discrete_mask(names)
        max_bins = max(
            8,
            int(
                getattr(self, "B", 8) if getattr(self, "B", 8) and getattr(self, "B", 8) > 0 else 16
            ),
        )
        if _CORE_AVAILABLE and hasattr(_core, "strict_topk_filter_csc"):
            scores, mask_native = _core.strict_topk_filter_csc(
                np.asarray(X_csc.data, dtype=np.float32),
                np.asarray(X_csc.indices, dtype=np.int32),
                np.asarray(X_csc.indptr, dtype=np.int32),
                int(X_csc.shape[0]),
                int(X_csc.shape[1]),
                np.asarray(y_codes, dtype=np.int64),
                discrete_mask,
                int(budget),
                int(max_bins),
            )
            scores = np.asarray(scores, dtype=np.float64)
            mask = np.asarray(mask_native, dtype=bool)
        else:
            scores = self._strict_topk_column_scores(X, y, names)
            order = np.lexsort((np.arange(n_cols), -scores))
            keep_idx = np.sort(order[:budget])
            mask = np.zeros(n_cols, dtype=bool)
            mask[keep_idx] = True
        self._strict_topk_feature_scores_ = scores
        self._strict_topk_feature_mask_ = mask
        self._strict_topk_selected_feature_names_ = [
            name for name, keep in zip(names, mask) if keep
        ]
        return X[:, mask]

    def _apply_strict_topk_budget_transform(self, X: csr_matrix) -> csr_matrix:
        if bool(getattr(self, "_strict_topk_applied_during_construction_", False)):
            return X
        mask = getattr(self, "_strict_topk_feature_mask_", None)
        if mask is None:
            return X
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.size != X.shape[1] or bool(np.all(mask_arr)):
            return X
        return X[:, mask_arr]

    def _numeric_feature_names_for_augmented_pairs(self) -> list[str]:
        names = list(getattr(self, "feature_names_in_", []) or [])
        cat_mask = getattr(self, "cat_cols_mask_", np.zeros(len(names), dtype=bool))
        return [name for name, is_cat in zip(names, cat_mask) if not bool(is_cat)]

    def _setup_augmented_pair_transforms(
        self, X_original: Any, y: Any | None = None, fit: bool = False
    ) -> None:
        """Create internal augmented_pair_transforms when L>1.

        This reuses native adaptive-binning IG metadata and does not expose a
        public hyperparameter. Augmented pair transforms are not fed into HUGIML
        mining; they are appended only before the downstream estimator.
        """
        enabled = bool(self.augmented_pair_transforms)
        if not fit or self.L <= 1 or not enabled or X_original is None:
            self._augmented_pair_block_ = None
            self.augmented_pair_transforms_ = []
            self.augmented_pair_selected_features_ = []
            self.augmented_pair_transforms_enabled_ = False
            return
        if not getattr(self, "adaptive_binning", False):
            warnings.warn(
                "augmented_pair_transforms require adaptive_binning=True because they are selected from adaptive-binning IG metadata; no augmented pair features will be added.",
                HUGIMLWarning,
                stacklevel=2,
            )
            self._augmented_pair_block_ = None
            self.augmented_pair_transforms_ = []
            self.augmented_pair_selected_features_ = []
            self.augmented_pair_transforms_enabled_ = False
            self.augmented_pair_config_ = {
                "enabled": False,
                "reason": "adaptive_binning_required",
                "augmented_pair_mode": str(self.augmented_pair_mode),
                "aug_feature_size": int(self.aug_feature_size),
                "ii_partner_size": self.ii_partner_size,
                "max_pair_features": int(self.max_pair_features),
                "budget": int(self.topK) if self.topK is not None and self.topK >= 0 else None,
                "num_candidates": 0,
                "num_retained": 0,
            }
            return
        if not getattr(self, "ig_scores_", None):
            warnings.warn(
                "augmented_pair_transforms were requested but no adaptive-binning IG scores are available; no augmented pair features will be added.",
                HUGIMLWarning,
                stacklevel=2,
            )
            self._augmented_pair_block_ = None
            self.augmented_pair_transforms_ = []
            self.augmented_pair_selected_features_ = []
            self.augmented_pair_transforms_enabled_ = False
            self.augmented_pair_config_ = {
                "enabled": False,
                "reason": "missing_ig_scores",
                "augmented_pair_mode": str(self.augmented_pair_mode),
                "aug_feature_size": int(self.aug_feature_size),
                "ii_partner_size": self.ii_partner_size,
                "max_pair_features": int(self.max_pair_features),
                "budget": int(self.topK) if self.topK is not None and self.topK >= 0 else None,
                "num_candidates": 0,
                "num_retained": 0,
            }
            return
        pair_budget = None if bool(getattr(self, "topk_budget_strict", False)) else self.topK
        block = NativeAugmentedPairTransformBlock(
            augmented_pair_mode=self.augmented_pair_mode,
            aug_feature_size=self.aug_feature_size,
            ii_partner_size=self.ii_partner_size,
            max_pair_features=self.max_pair_features,
            budget_topK=pair_budget,
            min_source_ig=self.G,
            native_cache=getattr(self, "_native_augmented_pair_cache_", None),
            score_topK=getattr(self, "_native_augmented_pair_score_topK_", None),
        )
        try:
            block.fit(
                X_original,
                y,
                getattr(self, "ig_scores_", {}) or {},
                getattr(self, "_bin_edges_", {}) or {},
                self._numeric_feature_names_for_augmented_pairs(),
                budget_topK=pair_budget,
                min_source_ig=self.G,
                full_feature_names=list(getattr(self, "feature_names_in_", []) or []),
            )
        finally:
            block.native_cache = None
        self._augmented_pair_block_ = block
        self.augmented_pair_transforms_ = list(block.augmented_pair_transforms_)
        self.augmented_pair_selected_features_ = list(block.selected_aug_features_)
        self.augmented_pair_transforms_enabled_ = bool(self.augmented_pair_transforms_)
        self.augmented_pair_config_ = {
            "enabled": self.augmented_pair_transforms_enabled_,
            "augmented_pair_mode": str(self.augmented_pair_mode),
            "aug_feature_size": int(self.aug_feature_size),
            "ii_partner_size": self.ii_partner_size,
            "max_pair_features": int(self.max_pair_features),
            "budget": int(self.topK) if self.topK is not None and self.topK >= 0 else None,
            "budget_source": "global_strict_topK"
            if bool(getattr(self, "topk_budget_strict", False))
            else "topK",
            "ops": ["product", "absolute_difference", "sum", "signed_difference"],
            "score": str(self.augmented_pair_mode),
            "min_source_ig": float(
                getattr(block, "min_source_ig_", max(1e-12, float(self.G or 0.0)))
            ),
            "num_candidates": int(getattr(block, "candidate_count_", 0)),
            "num_retained": len(self.augmented_pair_transforms_),
        }

    def _make_augmented_pair_features(self, X_original: Any, fit: bool = False):
        if self.L <= 1 or not bool(self.augmented_pair_transforms) or X_original is None:
            n_rows = 0 if X_original is None else len(X_original)
            return csr_matrix((n_rows, 0), dtype=np.float32)
        block = getattr(self, "_augmented_pair_block_", None)
        if block is None:
            return csr_matrix((len(X_original), 0), dtype=np.float32)
        if fit and hasattr(block, "_fit_Z_cache_"):
            block._consume_fit_cache_ = True
        return block.transform(X_original)

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

"""Input preparation, validation, adaptive binning, and missing-value handling."""

from __future__ import annotations

import copy
import math
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

from hugiml._binning import _apply_edges as _adap_apply_edges
from hugiml._binning import _quantile_edges as _adap_quantile_edges
from hugiml._binning import _select_b as _adap_select_b
from hugiml._classifier_runtime import _CORE_AVAILABLE, _core
from hugiml._classifier_support import (
    _NUMERIC_DTYPE_KINDS,
    AUGMENTED_PAIR_MODES,
    _edge_information_gain,
    _is_binary_feature_series,
    _is_zero_variance_numeric_column,
    _joint_information_gain_from_binned_columns,
    logger,
)
from hugiml._compat import liblinear_penalty_kwargs, logistic_penalty_kwargs
from hugiml.exceptions import (
    HUGIMLConvergenceWarning,
    HUGIMLMemoryError,
    HUGIMLParamError,
    HUGIMLValidationError,
    HUGIMLWarning,
)


class _BinningMixin:
    """Implement input preparation and binning operations."""

    def prepareXy(self, X: pd.DataFrame, y: Any) -> tuple[pd.DataFrame, np.ndarray]:
        """Detect column types and encode the target variable.

        Call on the full dataset **before** any train/test split.  Records
        which columns are integer, float, or categorical, and performs
        basic label validation.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series or array-like

        Returns
        -------
        X : pd.DataFrame (copy with string column names)
        y : np.ndarray of int64
        """
        if not isinstance(X, pd.DataFrame):
            raise HUGIMLParamError(f"X must be a pandas DataFrame, got {type(X).__name__}")

        X = X.copy()
        X.columns = [str(c) for c in X.columns]

        if len(set(X.columns)) < len(X.columns):
            dups = {c for c in X.columns if list(X.columns).count(c) > 1}
            warnings.warn(
                f"Duplicate column names detected: {dups}.  Results may be unreliable.",
                HUGIMLWarning,
                stacklevel=2,
            )

        catCols: list[str] = []
        binaryCatCols: list[str] = []
        for idx, c in enumerate(X.columns):
            col = X.iloc[:, idx]
            is_explicit_cat = (
                pd.api.types.is_object_dtype(col)
                or pd.api.types.is_string_dtype(col)
                or isinstance(col.dtype, pd.CategoricalDtype)
            )
            # Estimators serialized before this parameter existed used binary
            # numeric columns as categorical indicators. Preserve that legacy
            # behavior only when the attribute is absent; newly constructed
            # estimators use the explicit constructor default (False).
            is_binary_numeric = (
                getattr(self, "convert_binary_to_categorical", True)
                and not is_explicit_cat
                and (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col))
                and _is_binary_feature_series(col)
            )
            if is_explicit_cat or is_binary_numeric:
                catCols.append(c)
            if is_binary_numeric:
                binaryCatCols.append(c)
                X[c] = col.astype("category")

        intCols = [
            c
            for idx, c in enumerate(X.columns)
            if c not in set(catCols) and pd.api.types.is_integer_dtype(X.iloc[:, idx])
        ]

        for idx, c in enumerate(X.columns):
            if c not in catCols and X.iloc[:, idx].nunique() <= 1:
                warnings.warn(
                    f"Column '{c}' is constant and will produce zero utility.",
                    HUGIMLConvergenceWarning,
                    stacklevel=2,
                )

        X = X.reset_index(drop=True)
        self.feature_names_in_ = X.columns.tolist()
        self.cat_cols_mask_ = np.array([c in set(catCols) for c in X.columns], dtype=bool)
        self.is_int_mask_ = np.array([c in set(intCols) for c in X.columns], dtype=bool)
        self.binary_categorical_cols_ = list(binaryCatCols)

        y = np.asarray(y)
        try:
            y_float = y.astype(float)
            if np.isnan(y_float).any():
                raise HUGIMLValidationError("y contains NaN values.")
        except (ValueError, TypeError) as e:
            if "NaN" in str(e):
                raise

        if np.issubdtype(y.dtype, np.floating):
            if np.allclose(y, y.astype(int)):
                y = y.astype(np.int64)
            else:
                raise HUGIMLValidationError(
                    "y contains non-integer float values.  HUG-IML requires integer class labels."
                )

        return X, y

    @staticmethod
    def _safe_cast_y(y: Any) -> np.ndarray:
        """Cast label array to int64, raising a clear error if y is non-finite.

        ``np.asarray(y, dtype=np.int64)`` silently produces undefined values
        (typically INT64_MIN = -9223372036854775808) when y contains NaN or
        infinity; the C++ mining kernel then processes garbage class codes.
        This helper converts via float first and raises ``HUGIMLValidationError``
        if any non-finite values are present, which is the correct contract
        for a classifier that requires integer class labels.
        """
        y_arr = np.asarray(y)
        if np.issubdtype(y_arr.dtype, np.floating) or y_arr.dtype == object:
            y_float = y_arr.astype(float, copy=False)
            if not np.all(np.isfinite(y_float)):
                raise HUGIMLValidationError(
                    "y contains NaN or infinite values. HUG-IML requires integer class labels."
                )
            return y_float.astype(np.int64)
        return y_arr.astype(np.int64)

    def _adaptive_binning_sample_fraction_for_native(self) -> float:
        frac = getattr(self, "adaptive_binning_sample_frac", False)
        if frac is False or frac is None:
            return 1.0
        return float(frac)

    @staticmethod
    def _to_float_array(arr: Any, cat_mask: np.ndarray | None = None) -> tuple:
        """Split input into a numeric array and raw categorical arrays.

        All-numeric float32 inputs remain float32 for native L1 preparation;
        other numeric inputs use float64.

        Adversarial-input hardening:
        - Forces writable copies of read-only column views.
        - Non-finite cells (NaN/Inf) in numerical columns are pre-converted
          to np.nan string-label bins by _prebin_nan_cols (fit) or
          _handle_test_nan (predict), so they arrive here as categorical.
          No median imputation is performed.
        """
        is_df = isinstance(arr, pd.DataFrame)
        n = len(arr)
        if is_df:
            p = len(arr.columns)
            arr_np: np.ndarray | None = None
        else:
            arr_np = np.asarray(arr)
            p = arr_np.shape[1]

        if cat_mask is None:
            cat_mask = np.zeros(p, dtype=bool)

        # Hot all-numeric path: keep float32 inputs as float32.  The native L1
        # bindings accept float32 directly, so avoiding the unconditional
        # float64 upcast removes a full n×p copy in scalability/production runs.
        # Other numeric dtypes use float64 to preserve historical precision.
        if not np.any(cat_mask):
            try:
                target_dtype = np.float64
                if is_df:
                    try:
                        dtypes = [getattr(dt, "type", None) for dt in arr.dtypes]
                        if dtypes and all(dt is np.float32 for dt in dtypes):
                            target_dtype = np.float32
                    except Exception:
                        target_dtype = np.float64
                    out = arr.to_numpy(dtype=target_dtype, copy=True)
                    return np.ascontiguousarray(out, dtype=target_dtype), [None] * p
                assert arr_np is not None
                if arr_np.dtype == np.float32:
                    return np.ascontiguousarray(arr_np, dtype=np.float32), [None] * p
                return np.ascontiguousarray(arr_np, dtype=np.float64), [None] * p
            except Exception:
                pass

        X_num = np.zeros(
            (n, p), dtype=np.float32 if getattr(arr_np, "dtype", None) == np.float32 else np.float64
        )
        X_cat_raw = [None] * p

        for j in range(p):
            if is_df:
                raw = arr.iloc[:, j]
            else:
                assert arr_np is not None  # nosec B101 – guaranteed by control flow
                raw = arr_np[:, j]
            if cat_mask[j]:
                col_obj = np.asarray(raw, dtype=object).copy()
                for i, v in enumerate(col_obj):
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        col_obj[i] = np.nan
                X_cat_raw[j] = col_obj
                X_num[:, j] = 0.0
            else:
                col = np.array(raw, dtype=X_num.dtype, copy=True)
                # Non-finite cells (NaN/Inf) are pre-handled by
                # _prebin_nan_cols (fit) and _handle_test_nan (predict)
                # before reaching here.  No median imputation.
                X_num[:, j] = col

        return X_num, X_cat_raw

    def _effective_topK_total(self, n_items: int | None = None) -> int:
        """Return the user-facing total topK pattern budget."""
        if self.topK != -1:
            return int(self.topK)
        nitems = int(n_items) if n_items is not None else 100
        nitems = max(nitems, 1)
        cap = 20000
        if self.L == 1:
            return min(nitems, cap)
        if isinstance(self.L, int) and self.L >= 2:
            return min(math.comb(nitems, min(self.L, nitems)), cap)
        return min(sum(math.comb(nitems, r) for r in range(1, min(nitems, 6) + 1)), cap)

    def _effective_topK(self, n_items: int | None = None) -> int:
        """Return the effective topK budget used by the single-pass miner."""
        return max(1, int(self._effective_topK_total(n_items)))

    def _effective_mining_topK(self, n_items: int | None = None) -> int:
        """Return the topK passed to the native miner.

        Mining uses exactly the requested topK budget. No expanded
        information-gain candidate pool and no round-wise budget splitting are
        used.
        """
        return self._effective_topK(n_items)

    def _select_patterns_for_budget(self, patterns: list, n_items: int | None = None) -> list:
        """Return mined patterns without post-mining IG oversampling/filtering."""
        return list(patterns)

    def _deduplicate_patterns_by_coverage(
        self, patterns: list, n_rows: int
    ) -> tuple[list, tuple | None]:
        """Remove duplicate HUG patterns with identical training-row coverage.

        This is intentionally a post-mining HUGIML optimization.  Raw top-k
        mining output remains non-deduplicated and can be compared directly
        with THUI/HMiner/brute-force utility oracles.

        Among patterns that cover exactly the same set of training rows, the
        one with the highest information gain (and highest utility on ties) is
        retained; all lower-ranked duplicates are discarded.  Scanning forward
        through the descending-IG-sorted list achieves this: the first time a
        coverage key is encountered it belongs to the best representative, so
        every subsequent occurrence of that key is marked for removal.
        fsK is intentionally not applied.

        COO cache: the matrix built here for coverage-key computation
        is a superset of the Stage-4 matrix (all raw patterns vs. survivors).
        We filter the COO in Python to keep only survivor columns and return
        the result alongside the pattern list, so Stage 4 can skip the second
        C++ build_train_matrix call entirely.

        Returns
        -------
        survivors : list of PatternEntry
        cached_coo : (rows_array, cols_array) filtered to survivors, or None
        """
        if not patterns:
            return [], None
        ordered = sorted(patterns, key=lambda pe: (-pe.ig, -pe.utility))
        if hasattr(_core, "build_train_matrix_csr"):
            indptr_raw, indices_raw = _core.build_train_matrix_csr(self.td_, ordered)
            indptr_np = np.asarray(indptr_raw, dtype=np.int64)
            cols_np = np.asarray(indices_raw, dtype=np.int64)
            rows_np = np.repeat(
                np.arange(max(len(indptr_np) - 1, 0), dtype=np.int64), np.diff(indptr_np)
            )
        else:
            rows_raw, cols_raw = _core.build_train_matrix(self.td_, ordered)
            rows_np = np.asarray(rows_raw, dtype=np.int64)
            cols_np = np.asarray(cols_raw, dtype=np.int64)

        # build coverage keys without Python-level int() conversions.
        # Sort COO by column index, then use searchsorted to split rows into
        # per-pattern arrays.  Hash each array via .tobytes() rather than
        # materializing a Python tuple, which avoids O(n) object allocation
        # per pattern.
        n_ordered = len(ordered)
        seen: set[bytes] = set()
        remove: set[int] = set()
        if len(rows_np) > 0:
            order = np.argsort(cols_np, kind="stable")
            rows_sorted = rows_np[order].astype(np.int64)
            cols_sorted = cols_np[order]
            # split_points[i] = start index of column i in sorted arrays
            split_points = np.searchsorted(cols_sorted, np.arange(n_ordered))
            split_points_end = np.append(split_points[1:], len(rows_sorted))
            for i in range(n_ordered):
                seg = rows_sorted[split_points[i] : split_points_end[i]]
                # Sort within segment for a canonical key regardless of COO order
                key = np.sort(seg).tobytes()
                if key in seen:
                    remove.add(i)
                else:
                    seen.add(key)
        survivors = [pe for i, pe in enumerate(ordered) if i not in remove]

        # Build the column-index remap: old ordered-index → new survivor index.
        # Then filter the raw COO to keep only survivor entries, remapping cols.
        survivor_old_indices = [i for i in range(len(ordered)) if i not in remove]
        old_to_new = {old: new for new, old in enumerate(survivor_old_indices)}
        keep_mask = np.isin(cols_np, survivor_old_indices)
        rows_filtered = rows_np[keep_mask].astype(np.int32)
        cols_filtered = np.array([old_to_new[int(c)] for c in cols_np[keep_mask]], dtype=np.int32)
        return survivors, (rows_filtered, cols_filtered)

    def _make_estimator(self, n_cls: int) -> Any:
        if self.base_estimator is not None:
            estimator = copy.deepcopy(self.base_estimator)
            if (
                n_cls > 2
                and isinstance(estimator, LogisticRegression)
                and str(getattr(estimator, "solver", "")).lower() == "liblinear"
            ):
                return OneVsRestClassifier(estimator, n_jobs=1)
            return estimator

        lr_solver = str(getattr(self, "lr_solver", "auto")).lower()
        lr_C = float(getattr(self, "_benchmark_lr_C", 1.0))
        if lr_solver == "adaptive_l1":
            if n_cls == 2:
                return LogisticRegression(
                    solver="liblinear",
                    C=lr_C,
                    random_state=0,
                    max_iter=300,
                    **liblinear_penalty_kwargs("l1"),
                )
            return LogisticRegression(
                solver="saga",
                C=lr_C,
                random_state=0,
                max_iter=500,
                **logistic_penalty_kwargs("l1"),
            )
        if lr_solver == "auto":
            if n_cls == 2:
                return LogisticRegression(
                    solver="liblinear",
                    C=lr_C,
                    random_state=0,
                    max_iter=500,
                    **liblinear_penalty_kwargs("l1"),
                )
            return LogisticRegression(
                solver="saga",
                C=lr_C,
                random_state=0,
                max_iter=500,
                **logistic_penalty_kwargs("l1"),
            )
        if lr_solver == "saga":
            return LogisticRegression(
                solver="saga",
                C=lr_C,
                random_state=0,
                max_iter=500,
                **logistic_penalty_kwargs("l1"),
            )
        if lr_solver == "sgd":
            return SGDClassifier(loss="log_loss", penalty="l1", random_state=0, max_iter=500)
        raise HUGIMLParamError(
            "lr_solver must be one of {'auto', 'adaptive_l1', 'saga', 'sgd'}, "
            f"got {getattr(self, 'lr_solver', None)!r}."
        )

    def _validate_params(self) -> None:
        if not isinstance(self.B, int):
            raise HUGIMLParamError(f"B must be int, got {type(self.B).__name__}")
        if self.B != -1 and self.B < 2:
            raise HUGIMLParamError(f"B must be -1 (auto) or >= 2, got {self.B}")
        if not isinstance(self.L, int):
            raise HUGIMLParamError(f"L must be int, got {type(self.L).__name__}")
        if not isinstance(self.G, (float, int)):
            raise HUGIMLParamError(f"G must be numeric, got {type(self.G).__name__}")
        if self.G < 0:
            raise HUGIMLParamError(f"G must be >= 0, got {self.G}")
        lr_solver = str(getattr(self, "lr_solver", "auto")).lower()
        if lr_solver not in {"auto", "adaptive_l1", "saga", "sgd"}:
            raise HUGIMLParamError(
                "lr_solver must be one of {'auto', 'adaptive_l1', 'saga', 'sgd'}, "
                f"got {getattr(self, 'lr_solver', None)!r}."
            )
        dense_width = getattr(self, "dense_downstream_max_width", 200)
        if isinstance(dense_width, bool) or not isinstance(dense_width, int):
            raise HUGIMLParamError(
                f"dense_downstream_max_width must be an int >= 0, got {type(dense_width).__name__}"
            )
        if int(dense_width) < 0:
            raise HUGIMLParamError(f"dense_downstream_max_width must be >= 0, got {dense_width}")
        if getattr(self, "execution_mode", "audit") not in {"audit", "production"}:
            raise HUGIMLParamError(
                "execution_mode must be either 'audit' or 'production'. "
                "Use 'audit' for complete traceability/governance artifacts, "
                "or 'production' to retain only prediction-critical state."
            )
        if self.allCols is not None or self.origColumns is not None:
            if self.allCols is None or self.origColumns is None:
                raise HUGIMLParamError("allCols and origColumns must both be supplied together.")
            if not (isinstance(self.allCols, list) and len(self.allCols) == 3):
                raise HUGIMLParamError("allCols must be [int_cols, float_cols, cat_cols].")
        # ── Adaptive binning params ──────────────────────────────────────
        if not isinstance(self.adaptive_binning, bool):
            raise HUGIMLParamError("adaptive_binning must be bool.")
        if self.b_candidates is not None:
            if (
                not isinstance(self.b_candidates, list)
                or len(self.b_candidates) == 0
                or not all(isinstance(b, int) and b >= 2 for b in self.b_candidates)
            ):
                raise HUGIMLParamError("b_candidates must be a non-empty list of int >= 2.")
        if not isinstance(self.min_marginal_gain_ratio, (float, int)):
            raise HUGIMLParamError("min_marginal_gain_ratio must be numeric.")
        if not 0 < float(self.min_marginal_gain_ratio) < 1:
            raise HUGIMLParamError(
                f"min_marginal_gain_ratio must be in (0, 1), got {self.min_marginal_gain_ratio}."
            )
        frac = self.adaptive_binning_sample_frac
        if isinstance(frac, bool):
            if frac is True:
                raise HUGIMLParamError(
                    "adaptive_binning_sample_frac must be False or a float in (0, 1]."
                )
        elif frac is not None:
            if not isinstance(frac, (float, int)):
                raise HUGIMLParamError(
                    "adaptive_binning_sample_frac must be False or a float in (0, 1]."
                )
            if not 0 < float(frac) <= 1:
                raise HUGIMLParamError(
                    f"adaptive_binning_sample_frac must be in (0, 1], got {frac}."
                )
        if not isinstance(self.adaptive_binning_sample_random_state, int) or isinstance(
            self.adaptive_binning_sample_random_state, bool
        ):
            raise HUGIMLParamError("adaptive_binning_sample_random_state must be an int.")
        allowed_feature_modes = {
            "patterns_only",
            "original_plus_patterns",
            "original_plus_interactions",
        }
        if self.feature_mode not in allowed_feature_modes:
            raise HUGIMLParamError(
                f"feature_mode must be one of {sorted(allowed_feature_modes)}, "
                f"got {self.feature_mode!r}."
            )
        if not isinstance(self.augmented_pair_transforms, bool):
            raise HUGIMLParamError(
                "augmented_pair_transforms must be bool, "
                f"got {type(self.augmented_pair_transforms).__name__}."
            )
        if not isinstance(self.topk_budget_strict, bool):
            raise HUGIMLParamError(
                f"topk_budget_strict must be bool, got {type(self.topk_budget_strict).__name__}."
            )
        if self.augmented_pair_mode not in AUGMENTED_PAIR_MODES:
            raise HUGIMLParamError(
                f"augmented_pair_mode must be one of {list(AUGMENTED_PAIR_MODES)}, "
                f"got {self.augmented_pair_mode!r}."
            )
        if self.augmented_pair_max_features is not None:
            if not isinstance(self.augmented_pair_max_features, int):
                raise HUGIMLParamError(
                    "augmented_pair_max_features must be int or None, "
                    f"got {type(self.augmented_pair_max_features).__name__}."
                )
            if self.augmented_pair_max_features < 2:
                raise HUGIMLParamError(
                    "augmented_pair_max_features must be >= 2 when provided, "
                    f"got {self.augmented_pair_max_features}."
                )
        if not isinstance(self.aug_feature_size, int):
            raise HUGIMLParamError(
                f"aug_feature_size must be int, got {type(self.aug_feature_size).__name__}."
            )
        if self.aug_feature_size < 2:
            raise HUGIMLParamError(f"aug_feature_size must be >= 2, got {self.aug_feature_size}.")
        if self.ii_partner_size is not None:
            if not isinstance(self.ii_partner_size, int):
                raise HUGIMLParamError(
                    f"ii_partner_size must be int or None, got {type(self.ii_partner_size).__name__}."
                )
            if self.ii_partner_size < 2:
                raise HUGIMLParamError(f"ii_partner_size must be >= 2, got {self.ii_partner_size}.")
        if self.augmented_pair_mode == "marginal_ig":
            if not isinstance(self.max_pair_features, int):
                raise HUGIMLParamError(
                    f"max_pair_features must be int, got {type(self.max_pair_features).__name__}."
                )
            if self.max_pair_features < 2:
                raise HUGIMLParamError(
                    f"max_pair_features must be >= 2, got {self.max_pair_features}."
                )
        if not isinstance(self.interaction_relaxed_mining, bool):
            raise HUGIMLParamError(
                "interaction_relaxed_mining must be bool, "
                f"got {type(self.interaction_relaxed_mining).__name__}."
            )
        if self.interaction_relaxed_mining:
            # Mutually exclusive with the augmented-pair operator-feature
            # path at L >= 2.
            if bool(self.augmented_pair_transforms) and isinstance(self.L, int) and self.L >= 2:
                raise HUGIMLParamError(
                    "interaction_relaxed_mining=True is mutually exclusive with "
                    "augmented_pair_transforms=True at L >= 2. Set "
                    "augmented_pair_transforms=False to use interaction_relaxed_mining, "
                    "or set interaction_relaxed_mining=False to use the augmented-pair "
                    "operator-feature path instead."
                )
            # L=1 always uses the fused hotpath, which has no relaxed
            # variant; relaxation is a no-op there. L>=2 and L=-1 route
            # through mine_patterns_relaxed_cpp's dispatcher.  Do not cap this
            # at L<=3: bounded higher-order mining (for example L=4 or L=5 for
            # multiplexer-style rules) uses the generic relaxed dispatcher.
            if isinstance(self.L, int) and self.L != -1 and self.L < 1:
                raise HUGIMLParamError(
                    f"interaction_relaxed_mining=True requires L=-1 or L>=1, got L={self.L}."
                )
        if not isinstance(self.interaction_relaxed_feature_size, int):
            raise HUGIMLParamError(
                "interaction_relaxed_feature_size must be int, "
                f"got {type(self.interaction_relaxed_feature_size).__name__}."
            )
        if self.interaction_relaxed_feature_size < 2:
            raise HUGIMLParamError(
                "interaction_relaxed_feature_size must be >= 2, "
                f"got {self.interaction_relaxed_feature_size}."
            )

    def _resolve_col_meta(self, X_train: Any) -> np.ndarray:
        """Determine column names and type masks from whichever setup path was used."""
        if hasattr(self, "cat_cols_mask_"):
            return self.cat_cols_mask_

        if self.allCols is not None and self.origColumns is not None:
            cat_set = set(self.allCols[2])
            int_set = set(self.allCols[0])
            col_list = list(self.origColumns)
            self.cat_cols_mask_ = np.array([c in cat_set for c in col_list], dtype=bool)
            self.is_int_mask_ = np.array([c in int_set for c in col_list], dtype=bool)
            self.binary_categorical_cols_ = []
            self.feature_names_in_ = col_list
            return self.cat_cols_mask_

        if isinstance(X_train, pd.DataFrame):
            col_list = X_train.columns.astype(str).tolist()
            cat_mask_values: list[bool] = []
            int_mask_values: list[bool] = []
            binary_cat_cols: list[str] = []
            # Attribute absence identifies a legacy estimator whose binary
            # numeric columns were inferred categorically before this option
            # became explicit. New estimators always carry the False default.
            convert_binary = getattr(self, "convert_binary_to_categorical", True)
            for c in X_train.columns:
                col = X_train[c]
                is_explicit_cat = (
                    pd.api.types.is_object_dtype(col)
                    or pd.api.types.is_string_dtype(col)
                    or isinstance(col.dtype, pd.CategoricalDtype)
                )
                is_binary_numeric = (
                    convert_binary
                    and not is_explicit_cat
                    and (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col))
                    and _is_binary_feature_series(col)
                )
                cat_mask_values.append(bool(is_explicit_cat or is_binary_numeric))
                int_mask_values.append(
                    bool((not is_binary_numeric) and pd.api.types.is_integer_dtype(col))
                )
                if is_binary_numeric:
                    binary_cat_cols.append(str(c))
            self.cat_cols_mask_ = np.array(cat_mask_values, dtype=bool)
            self.is_int_mask_ = np.array(int_mask_values, dtype=bool)
            self.binary_categorical_cols_ = list(binary_cat_cols)
            self.feature_names_in_ = col_list
            return self.cat_cols_mask_

        arr = np.asarray(X_train)
        if arr.ndim < 2:
            raise ValueError(
                f"HUGIMLClassifier expects a 2D array, got array of shape {arr.shape}."
            )
        p = arr.shape[1]
        self.cat_cols_mask_ = np.zeros(p, dtype=bool)
        self.is_int_mask_ = np.zeros(p, dtype=bool)
        self.binary_categorical_cols_ = []
        # Array inputs have no native column labels, but downstream components
        # (notably augmented-pair transforms) require stable feature names to
        # align IG scores, selected source columns, and transform-time matrices.
        # Use deterministic synthetic names for ndarray inputs.
        self.feature_names_in_ = [f"col{j}" for j in range(p)]
        return self.cat_cols_mask_

    def _identify_zero_variance_columns(self, X_train: Any) -> list[str]:
        """Return the names of columns with at most one distinct observed value.

        A column where every non-missing cell holds the same value (or every
        cell is missing) carries no mutual information with the target by
        construction, so it can never contribute an eligible item to the
        registry built during transaction preparation -- the existing
        information-gain computation already assigns such a column zero
        utility for every candidate bin or category. Recognising this here,
        before that computation runs, lets the per-row work building the
        item registry (categorical label extraction in particular, which is
        comparatively the most expensive part of it) skip these columns
        entirely rather than touching every row only to confirm what is
        already known from one cheap, vectorised distinct-value count.

        Called once at fit time; the resulting column names are stored on
        the fitted estimator (``_zero_variance_cols_``) so predict-time
        processing excludes exactly the same columns regardless of what
        values they happen to hold in new data -- the model never learned
        anything from this column during training, so it consistently
        ignores it afterward rather than re-deciding per call.
        """
        names = list(getattr(self, "feature_names_in_", []) or [])
        if not names:
            return []

        if isinstance(X_train, pd.DataFrame):
            return [
                name
                for name in names
                if name in X_train.columns and X_train[name].nunique(dropna=True) <= 1
            ]

        arr = np.asarray(X_train)
        if arr.dtype.kind in _NUMERIC_DTYPE_KINDS:
            return [
                name
                for j, name in enumerate(names)
                if j < arr.shape[1] and _is_zero_variance_numeric_column(arr[:, j])
            ]

        return self._identify_zero_variance_columns_fallback(names, X_train)

    def _identify_zero_variance_columns_fallback(self, names: list[str], X_train: Any) -> list[str]:
        """Object-safe zero-variance check for non-numeric input.

        Handles arbitrary Python objects (strings, mixed types, unhashable
        values) cell by cell. Only reached for non-numeric ndarrays/lists;
        numeric input and DataFrames are handled by the vectorised branches
        above, which is what matters for performance since this is the one
        branch that can't be vectorised in general.
        """
        arr = np.asarray(X_train, dtype=object)
        zero_variance = []
        for j, name in enumerate(names):
            if j >= arr.shape[1]:
                continue
            observed: set[Any] = set()
            for value in arr[:, j]:
                try:
                    missing = pd.isna(value)
                except (TypeError, ValueError):
                    missing = False
                if isinstance(missing, (list, tuple, np.ndarray)):
                    missing = False
                if bool(missing):
                    continue
                try:
                    observed.add(value)
                except TypeError:
                    observed.add(repr(value))
                if len(observed) > 1:
                    break
            if len(observed) <= 1:
                zero_variance.append(name)
        return zero_variance

    def _exclude_zero_variance_columns(
        self,
        X_num: np.ndarray,
        X_cat_raw: list[Any],
        cat_mask: np.ndarray,
        zero_variance_names: list[str],
    ) -> None:
        """Neutralise the listed columns in-place so native code skips them cheaply.

        Categorical columns are excluded by setting their ``X_cat_raw`` entry
        to ``None``, which the native categorical-extraction step (shared by
        the fit-time and predict-time code paths) already treats as "no data
        for this column" and skips outright -- no native-side change needed,
        since that behaviour already exists for the ordinary case where a
        caller has no categorical columns at all. Numeric columns are
        neutralised by writing a single constant value through the column;
        since the column is already constant (or entirely missing) by the
        definition used to select it, this changes nothing about what the
        column represents while letting every downstream numeric pass over
        it (range computation, binning, correlation) settle on its answer
        immediately rather than scanning genuinely varying values.
        """
        if not zero_variance_names:
            return
        names = list(getattr(self, "feature_names_in_", []) or [])
        name_to_index = {name: j for j, name in enumerate(names)}
        for name in zero_variance_names:
            j = name_to_index.get(name)
            if j is None or j >= len(cat_mask):
                continue
            if cat_mask[j]:
                if j < len(X_cat_raw):
                    X_cat_raw[j] = None
            else:
                if X_num.shape[1] > j:
                    X_num[:, j] = 0.0

    @staticmethod
    def _timer() -> Any:
        """Return a lightweight timer object."""

        class _T:
            def __init__(self) -> None:
                self.start = time.perf_counter()

            @property
            def ms(self) -> float:
                return (time.perf_counter() - self.start) * 1000

        return _T()

    def _rebuild_adaptive_code_label_map(self) -> None:
        """Reconstruct ``_adaptive_code_label_map_`` from stored ``_bin_edges_``.

        Called automatically by ``__setstate__`` after deserialization (both
        pickle and the custom .hugiml format) when the map is absent or empty.
        The map is not stored explicitly in the .hugiml format; it is always
        derived from ``_bin_edges_``, which IS persisted.

        The reconstruction exactly mirrors the map built in
        ``_apply_adaptive_binning``: for each column whose edges are stored, and
        for each bin k, the key is the C++ item-map label
        ``"name=[k.000,(k+1).000]"`` and the value is the original-scale label
        ``"name=[edges[k]:.4g, edges[k+1]:.4g)"``.
        """
        bin_edges = getattr(self, "_bin_edges_", {})
        if not bin_edges:
            return
        # We only build entries for columns that were encoded as is_precoded
        # (i.e., had no NaN at training time).  After deserialization we cannot
        # distinguish which columns were NaN-fallback vs is_precoded.  Safe
        # approach: rebuild for all columns in _bin_edges_.  The entry is only
        # consulted when the C++ item_map actually contains the key, so spurious
        # entries for NaN-fallback columns are harmless.
        new_map: dict[str, str] = {}
        for name, edges in bin_edges.items():
            n_bins = len(edges) - 1
            for k in range(n_bins):
                cpp_label = f"{name}=[{float(k):.3f},{float(k + 1):.3f}]"
                orig_label = f"{name}=[{edges[k]:.4g},{edges[k + 1]:.4g})"
                new_map[cpp_label] = orig_label
        self._adaptive_code_label_map_ = new_map
        self._adaptive_precoded_features_ = set(bin_edges)

    def _apply_adaptive_binning_cpp(self, X_train: Any, y_arr: np.ndarray) -> Any:
        """C++ replacement for _apply_adaptive_binning.

        Calls _core.select_adaptive_bins (C++ elbow_stop_nb_cpp) instead of
        the Python _adap_select_b loop.  Produces identical _bin_edges_,
        per_feature_b_, ig_scores_, _adaptive_code_label_map_, cat_cols_mask_,
        and X_pre outputs.  Falls back to the Python path on any error.
        """
        try:
            return self._apply_adaptive_binning_cpp_impl(X_train, y_arr)
        except (MemoryError, HUGIMLMemoryError):
            # A native allocation failure means the Python fallback is very
            # likely to allocate even more memory.  Surface a clean OOM instead
            # of cascading into an OS-level kill.
            raise
        except RuntimeError as exc:
            if "hugiml_timeout" in str(exc):
                raise
            logger.warning("C++ adaptive binning failed (%s); falling back to Python path.", exc)
            return self._apply_adaptive_binning(X_train, y_arr)
        except Exception as exc:
            logger.warning("C++ adaptive binning failed (%s); falling back to Python path.", exc)
            return self._apply_adaptive_binning(X_train, y_arr)

    def _apply_adaptive_binning_cpp_impl(self, X_train: Any, y_arr: np.ndarray) -> Any:
        """Implementation of the C++ adaptive B-selection path.

        Works for numeric-only and mixed (numeric + string categorical) DataFrames.
        Extracts only the numeric columns into a float64 array before calling
        select_adaptive_bins, avoiding the ValueError that X_df.to_numpy(float64)
        raises when string-categorical columns are present.
        """
        is_df = isinstance(X_train, pd.DataFrame)
        X_df = (
            X_train
            if is_df
            else pd.DataFrame(
                X_train,
                columns=(
                    list(getattr(self, "feature_names_in_", []) or [])
                    if getattr(self, "feature_names_in_", None) is not None
                    and len(getattr(self, "feature_names_in_", [])) == np.asarray(X_train).shape[1]
                    else None
                ),
            )
        )
        candidates = sorted(set(self.b_candidates or [2, 3, 5, 7, 10, 15]))
        ratio = self.min_marginal_gain_ratio
        cat_mask = self.cat_cols_mask_
        col_names = list(X_df.columns)
        n_cls = len(np.unique(y_arr))

        self._bin_edges_: dict = {}
        self.per_feature_b_: dict = {}
        self.ig_scores_: dict = {}

        # ── Extract only the numeric (non-cat) columns ────────────────────────
        # Passing the full mixed DataFrame to to_numpy(float64) raises ValueError
        # when string-categorical columns are present.  select_adaptive_bins
        # already skips is_cat columns, so we can pass only the numeric subset
        # and map indices back afterwards via num_col_map.
        num_col_map = [j for j, v in enumerate(cat_mask) if not v]
        col_names_num = [col_names[j] for j in num_col_map]
        # C++ binding expects std::string column names.  ndarray inputs create
        # integer DataFrame column names, so pass string labels to C++ while
        # retaining the original names for pandas indexing below.
        col_names_num_cpp = [str(name) for name in col_names_num]

        if not num_col_map:
            # All columns are categorical — nothing to adapt
            self._adaptive_code_label_map_: dict[str, str] = {}
            return X_train

        # Build X_num: shape (n, len(num_col_map)), float64
        X_num = X_df.iloc[:, num_col_map].to_numpy(dtype=np.float64, na_value=np.nan)
        y_int = y_arr.astype(np.int64)
        # Pass all-zeros is_cat so C++ processes every column in X_num
        is_cat_zeros = np.zeros(len(num_col_map), dtype=np.uint8)

        adap_result = _core.select_adaptive_bins(
            X_num,
            y_int,
            n_cls,
            col_names_num_cpp,
            is_cat_zeros,
            candidates,
            ratio,
            adaptive_binning_sample_frac=self._adaptive_binning_sample_fraction_for_native(),
            adaptive_binning_sample_random_state=int(self.adaptive_binning_sample_random_state),
        )

        # Pack C++ results into Python model attributes.
        # adap_result.num_col_indices[ci] is the index within X_num (= within
        # col_names_num).  Map back to the original X_df column index via num_col_map.
        for ci, col_res in enumerate(adap_result.cols):
            j_num = adap_result.num_col_indices[ci]
            name = col_names_num[j_num]
            edges = np.array(col_res.edges)
            self._bin_edges_[name] = edges
            # Match the Python adaptive path, which records the effective number
            # of stored bins after duplicate quantile edges have collapsed.
            self.per_feature_b_[name] = len(edges) - 1
            # Pad missing candidates (early elbow-stop) with 0.0 for diagnostics
            scores: dict[int, float] = {}
            for k, b in enumerate(candidates):
                scores[b] = col_res.ig_scores[k] if k < len(col_res.ig_scores) else 0.0
            self.ig_scores_[name] = scores

        # Build _adaptive_code_label_map_ and update column-type masks
        self._adaptive_code_label_map_: dict[str, str] = {}
        self._adaptive_precoded_features_ = set(self._bin_edges_)
        new_cat = cat_mask.copy()
        new_int = getattr(self, "is_int_mask_", np.zeros(len(col_names), dtype=bool)).copy()

        for name, edges in self._bin_edges_.items():
            if name not in X_df.columns:
                continue
            j = col_names.index(name) if name in col_names else -1
            if j >= 0:
                new_cat[j] = False
                new_int[j] = True
            n_bins = len(edges) - 1
            for k in range(n_bins):
                cpp_label = f"{name}=[{float(k):.3f},{float(k + 1):.3f}]"
                orig_label = f"{name}=[{edges[k]:.4g},{edges[k + 1]:.4g})"
                self._adaptive_code_label_map_[cpp_label] = orig_label

        self.cat_cols_mask_ = new_cat
        self.is_int_mask_ = new_int

        # Apply integer codes to the pre-binned numeric columns in X_pre.
        # Native storage is int32 with -1 as the missing sentinel; fetch one
        # column at a time to avoid materialising a second full float64 code
        # matrix.  Cast only the one column that pandas needs so NaN can be
        # represented for the legacy pre-binned path.
        X_pre = X_df.copy()
        for ci in range(adap_result.n_num_cols):
            j_num = adap_result.num_col_indices[ci]
            name = col_names_num[j_num]
            col_raw = pd.to_numeric(X_df[name], errors="coerce").values
            if hasattr(adap_result, "get_X_codes_col"):
                codes_i32 = np.asarray(adap_result.get_X_codes_col(ci), dtype=np.int32)
                missing_codes = codes_i32 < 0
            else:  # compatibility with older native wheels
                # Older native wheels expose get_X_codes() as float64 with
                # np.nan as the missing sentinel.  Casting that matrix directly
                # to int32 can platform-dependently produce either INT32_MIN or
                # 0; the latter silently aliases a valid bin.  Detect missing
                # sentinels before the integer cast.
                codes_raw = np.asarray(adap_result.get_X_codes()[:, ci])
                missing_codes = ~np.isfinite(codes_raw)
                codes_i32 = np.zeros(codes_raw.shape, dtype=np.int32)
                finite_codes = ~missing_codes
                if np.any(finite_codes):
                    codes_i32[finite_codes] = codes_raw[finite_codes].astype(np.int32)
            codes = codes_i32.astype(np.float32, copy=False)
            codes[missing_codes | (codes_i32 < 0) | (~np.isfinite(col_raw))] = np.nan
            X_pre[name] = codes

        return X_pre if is_df else X_pre

    def _apply_adaptive_binning(self, X_train: Any, y_arr: np.ndarray) -> Any:
        """Pre-discretise numerical features using per-feature IG-selected B_j.

        Called by _fit_impl when adaptive_binning=True.  The method:

        1. Iterates over numerical columns, runs _adap_select_b to choose B_j.
        2. Computes quantile edges on the training column and stores them in
           _bin_edges_ so they can be reapplied at predict time.
        3. Replaces each numerical column with string bin labels ([lo,hi)).
        4. Updates cat_cols_mask_ and is_int_mask_ to mark pre-binned features
           as categorical, so Stage 1 of _fit_impl routes them through the
           C++ categorical path.

        Returns the pre-binned DataFrame (or the input unchanged if no
        numerical features are found).
        """
        is_df = isinstance(X_train, pd.DataFrame)
        X_df = (
            X_train
            if is_df
            else pd.DataFrame(
                X_train,
                columns=(
                    list(getattr(self, "feature_names_in_", []) or [])
                    if getattr(self, "feature_names_in_", None) is not None
                    and len(getattr(self, "feature_names_in_", [])) == np.asarray(X_train).shape[1]
                    else None
                ),
            )
        )
        candidates = sorted(set(self.b_candidates or [2, 3, 5, 7, 10, 15]))
        ratio = self.min_marginal_gain_ratio
        cat_mask = self.cat_cols_mask_
        col_names = list(X_df.columns)

        self._bin_edges_: dict = {}
        self.per_feature_b_: dict = {}
        self.ig_scores_: dict = {}
        pre_binned: set = set()

        # Identify which columns need adaptive B-selection.
        num_cols = [
            (j, name) for j, name in enumerate(col_names) if not (j < len(cat_mask) and cat_mask[j])
        ]

        # parallel B-selection:
        def _select_one(j: int, name: str) -> tuple:
            col = pd.to_numeric(X_df.iloc[:, j], errors="coerce").values
            finite_mask = np.isfinite(col)
            if finite_mask.sum() < 10:
                chosen = candidates[len(candidates) // 2]
                scores = {b: 0.0 for b in candidates}
            else:
                chosen, scores = _adap_select_b(
                    col[finite_mask], y_arr[finite_mask], candidates, ratio
                )
            edges = _adap_quantile_edges(col, chosen)
            return name, chosen, scores, edges

        try:
            from joblib import Parallel
            from joblib import delayed as _delayed

            _n_jobs = self.n_jobs if hasattr(self, "n_jobs") else 1
            _results = Parallel(n_jobs=_n_jobs, prefer="threads")(
                _delayed(_select_one)(j, name) for j, name in num_cols
            )
        except Exception:
            _results = [_select_one(j, name) for j, name in num_cols]

        for name, chosen, scores, edges in _results:
            self.ig_scores_[name] = scores
            self.per_feature_b_[name] = len(edges) - 1
            self._bin_edges_[name] = edges
            pre_binned.add(name)

        # ── Integer-code path ─────────────────────────────────────
        # Encode pre-binned columns as integer codes (0..B_j-1) and route them
        # through the C++ is_int path, which is 8–20x faster than the old
        # string-categorical std::string path.
        #
        # NaN handling (hybrid):
        #   Columns WITHOUT any NaN/Inf → is_int=True (fast integer path).
        #   Columns WITH NaN/Inf        → cat=True  (existing string path,
        #       correct because C++ categorical skips np.nan cells).
        # In practice most adaptive-mode datasets have no NaN; the string
        # fallback only fires when needed.
        #
        # Label translation: the C++ integer path stores item labels as
        # "feat=[k,k+1]" (integer-range format).  get_hug_features() remaps
        # these to original-scale "[lo,hi)" labels via _adaptive_code_label_map_.
        self._adaptive_code_label_map_: dict[str, str] = {}
        self._adaptive_precoded_features_ = set(self._bin_edges_)

        new_cat = cat_mask.copy()
        new_int = getattr(self, "is_int_mask_", np.zeros(len(col_names), dtype=bool)).copy()

        X_pre = X_df.copy()
        for name, edges in self._bin_edges_.items():
            if name not in X_df.columns:
                continue
            col = pd.to_numeric(X_df[name], errors="coerce").values
            n_bins = len(edges) - 1
            has_nan = not np.isfinite(col).all()

            j = col_names.index(name) if name in col_names else -1

            # Integer-code path for all adaptive columns, with or without NaN.
            # NaN/Inf cells are encoded as np.nan (float64), which the C++
            # is_precoded handler reads as non-finite and maps to -1 (skipped
            # — no item generated for that row/feature pair).
            # This keeps the column numeric rather than converting the whole
            # column to object/string dtype for a small number of NaNs, which
            # would force the slower C++ categorical path.
            codes = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1).astype(np.float64)
            if has_nan:
                codes[~np.isfinite(col)] = np.nan  # sentinel: C++ skips
            X_pre[name] = codes
            if j >= 0:
                new_cat[j] = False
                new_int[j] = True
            # Build C++ label -> original-scale label translation.
            # Key format matches the C++ is_precoded label exactly:
            # std::setprecision(3) -> "name=[0.000,1.000]"
            for k in range(n_bins):
                cpp_label = f"{name}=[{float(k):.3f},{float(k + 1):.3f}]"
                orig_label = f"{name}=[{edges[k]:.4g},{edges[k + 1]:.4g})"
                self._adaptive_code_label_map_[cpp_label] = orig_label

        self.cat_cols_mask_ = new_cat
        self.is_int_mask_ = new_int
        return X_pre if is_df else X_pre

    def _prebin_for_predict(self, X: Any) -> Any:
        """Apply stored adaptive bin edges before C++ inference.

        The common ndarray path is kept entirely in NumPy to avoid constructing
        and copying a pandas DataFrame for every predict()/transform() call.
        DataFrame input still preserves labels and mixed categorical columns.
        """
        bin_edges = getattr(self, "_bin_edges_", {})
        if not bin_edges:
            return X

        feat_names = getattr(self, "feature_names_in_", None)
        code_label_map = getattr(self, "_adaptive_code_label_map_", {})
        precoded_features = getattr(self, "_adaptive_precoded_features_", None)
        if precoded_features is None:
            # Backward-compatible fallback for models saved before this attribute.
            precoded_features = set(bin_edges) if code_label_map else set()

        # Fast path: numeric ndarray input.  Adaptive fused L1 stores ndarray
        # feature names as col0, col1, ...; keep the output as ndarray so
        # _to_float_array can consume it without pandas overhead.
        if not isinstance(X, pd.DataFrame):
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            names = (
                list(feat_names)
                if feat_names is not None and len(feat_names) == arr.shape[1]
                else [f"col{j}" for j in range(arr.shape[1])]
            )
            name_to_idx = {name: j for j, name in enumerate(names)}
            # If a name mismatch occurs, fall back to the labelled path rather
            # than silently applying edges to the wrong column.
            if all(name in name_to_idx for name in bin_edges):
                X_out = np.array(arr, dtype=np.float64, copy=True)
                for name, edges in bin_edges.items():
                    j = name_to_idx[name]
                    n_bins = len(edges) - 1
                    col = X_out[:, j]
                    if name in precoded_features:
                        codes = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1).astype(
                            np.float64
                        )
                        nan_mask = ~np.isfinite(col)
                        if nan_mask.any():
                            codes[nan_mask] = np.nan
                        X_out[:, j] = codes
                    else:
                        # Legacy string categorical fallback requires labels.
                        # It is rare in current adaptive models, but preserve
                        # correctness by using the DataFrame path below.
                        break
                else:
                    return X_out

        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            X_df = X
        else:
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            cols = (
                list(feat_names)
                if feat_names is not None and len(feat_names) == arr.shape[1]
                else [f"col{j}" for j in range(arr.shape[1])]
            )
            X_df = pd.DataFrame(arr, columns=cols)

        # Fast labelled path for the common adaptive case: all stored adaptive
        # columns are numeric pre-coded features.  Convert once to NumPy, edit
        # columns in-place, and rebuild one DataFrame instead of assigning one
        # pandas Series per feature.
        #
        # This path is only taken when every column outside of bin_edges is
        # non-categorical per the fitted schema. A blanket float64 cast across
        # the whole DataFrame would otherwise destroy the categorical-ness of
        # columns that are not in bin_edges at all (e.g. binary-numeric
        # auto-categorical columns admitted via
        # interaction_relaxed_mining/select_interaction_information_features)
        # before _build_test_hup ever saw them, which would prevent the native
        # build_test_matrix_csr categorical-matching path from recognizing
        # them -- any mined pattern touching such a column would then evaluate
        # to 0 for every row at predict/transform time, even though the
        # training-time pattern matrix (built on correctly-typed training
        # data) is unaffected.
        #
        # The raw input's pandas dtype is NOT a reliable signal here: at
        # predict time callers normally pass the original int64/object
        # columns (not a pre-cast 'category' dtype), so checking the input
        # dtype directly under-detects this case. The source of truth for
        # which columns are categorical is the fitted model's own
        # cat_cols_mask_/feature_names_in_ (set during fit's schema
        # detection). The fast path is therefore only taken when EVERY
        # column outside of bin_edges is, per the FITTED schema, not
        # categorical -- i.e. the blanket float64 cast cannot lose
        # information the native layer needs. Otherwise we always fall
        # through to the slower column-preserving path below, which only
        # mutates the specific bin_edges columns and leaves every other
        # column untouched (so its dtype, and the categorical match against
        # X_cat_raw downstream, stays correct).
        _fit_cat_mask = getattr(self, "cat_cols_mask_", None)
        _fit_feat_names = getattr(self, "feature_names_in_", None)
        if (
            _fit_cat_mask is not None
            and _fit_feat_names is not None
            and len(_fit_feat_names) == len(_fit_cat_mask)
        ):
            _fit_cat_names = {
                str(name) for name, is_cat in zip(_fit_feat_names, _fit_cat_mask) if is_cat
            }
        else:
            # Unknown schema: be conservative and assume any column could be
            # categorical, which simply routes to the always-correct slow path.
            _fit_cat_names = {str(c) for c in X_df.columns}
        non_bin_edge_cols_all_noncat = not any(
            str(c) in _fit_cat_names for c in X_df.columns if c not in bin_edges
        )
        if (
            is_df
            and non_bin_edge_cols_all_noncat
            and all(name in X_df.columns and name in precoded_features for name in bin_edges)
        ):
            try:
                cols = list(X_df.columns)
                name_to_idx = {str(c): j for j, c in enumerate(cols)}
                X_mat = X_df.to_numpy(dtype=np.float64, copy=True)
                for name, edges in bin_edges.items():
                    j = name_to_idx[name]
                    n_bins = len(edges) - 1
                    col = X_mat[:, j]
                    codes = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1).astype(np.float64)
                    nan_mask = ~np.isfinite(col)
                    if nan_mask.any():
                        codes[nan_mask] = np.nan
                    X_mat[:, j] = codes
                return pd.DataFrame(X_mat, columns=X_df.columns, index=X_df.index)
            except Exception:
                pass

        X_out = X_df.copy()
        for name, edges in bin_edges.items():
            if name not in X_df.columns:
                continue
            col = pd.to_numeric(X_df[name], errors="coerce").values
            n_bins = len(edges) - 1
            if name in precoded_features:
                codes = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1).astype(np.float64)
                nan_mask = ~np.isfinite(col)
                if nan_mask.any():
                    codes[nan_mask] = np.nan
                X_out[name] = codes
            else:
                X_out[name] = _adap_apply_edges(col, edges)
        return X_out if is_df else X_out

    def _prebin_nan_cols(self, X_train: Any) -> Any:
        """Pre-bin constant-B numeric columns that must follow the string path.

        Contract for every L value:
        - Finite numeric columns stay numeric and use the native constant-B numeric
          path, including L > 1.
        - Numeric columns with NaN/Inf during training are pre-binned to the
          string/categorical path so the fitted transaction data and later
          predictions share the same missing-value representation.

        A new NaN/Inf at prediction time in a column that was clean during
        training is handled by the native numeric transaction builder, which
        skips item generation for that cell.  This fit-only helper resets
        ``_missing_col_edges_`` at the beginning of each call; callers outside
        the normal fit path should not invoke it incrementally.  For ndarray
        inputs, the returned object preserves ndarray type when a conversion is
        needed, avoiding accidental DataFrame type coercion.
        """
        is_df = isinstance(X_train, pd.DataFrame)
        if is_df:
            X_df = X_train
        else:
            arr = np.asarray(X_train)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            feat_names = list(getattr(self, "feature_names_in_", []) or [])
            cols = (
                feat_names
                if len(feat_names) == arr.shape[1]
                else [f"col{j}" for j in range(arr.shape[1])]
            )
            X_df = pd.DataFrame(arr, columns=cols)
        cat_mask = self.cat_cols_mask_
        col_names = list(X_df.columns)
        n_cols = len(col_names)

        self._missing_col_edges_: dict = {}
        new_cat = cat_mask.copy()
        new_int = getattr(self, "is_int_mask_", np.zeros(n_cols, dtype=bool)).copy()
        modified = False

        for j, name in enumerate(col_names):
            if j >= len(cat_mask) or cat_mask[j]:
                continue  # already categorical — C++ handles np.nan natively

            col = pd.to_numeric(X_df.iloc[:, j], errors="coerce").values
            finite_mask = np.isfinite(col)
            if bool(np.all(finite_mask)):
                # Consistent fast path for every L: finite numeric columns
                # remain numeric and are binned by native constant-B code.
                continue
            finite = col[finite_mask]
            edges = (
                _adap_quantile_edges(finite, self.B) if finite.size > 0 else np.array([0.0, 1.0])
            )
            self._missing_col_edges_[name] = edges
            new_cat[j] = True
            new_int[j] = False
            modified = True

        if not modified:
            return X_train

        self.cat_cols_mask_ = new_cat
        self.is_int_mask_ = new_int

        X_pre = X_df.copy()
        for name, edges in self._missing_col_edges_.items():
            col = pd.to_numeric(X_df[name], errors="coerce").values
            X_pre[name] = _adap_apply_edges(col, edges)
        return X_pre if is_df else X_pre.to_numpy()

    def _frame_from_input(self, X: Any, names: list[str] | None = None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        cols = list(names or [])
        if len(cols) != arr.shape[1]:
            cols = [f"col{j}" for j in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)

    def _select_interaction_relaxed_survivors_from_frame(
        self,
        X_train: Any,
        y_arr: np.ndarray,
        col_names: list[str] | None = None,
        cat_mask: np.ndarray | None = None,
    ) -> tuple[list[dict], list[int]]:
        if not (
            _CORE_AVAILABLE
            and hasattr(_core, "select_interaction_information_features")
            and hasattr(_core, "mine_patterns_relaxed")
        ):
            raise HUGIMLParamError(
                "interaction_relaxed_mining=True requires "
                "_hugiml_core.select_interaction_information_features and "
                "_hugiml_core.mine_patterns_relaxed."
            )
        X_df = self._frame_from_input(
            X_train, col_names or getattr(self, "feature_names_in_", None)
        )
        names = [str(name) for name in list(X_df.columns)]
        cat = (
            cat_mask
            if cat_mask is not None
            else getattr(self, "cat_cols_mask_", np.zeros(len(names), dtype=bool))
        )
        binary_cat_names = set(getattr(self, "binary_categorical_cols_", []) or [])
        score_idx: list[int] = []
        score_cols: list[np.ndarray] = []
        score_names: list[str] = []
        for j, name in enumerate(names):
            is_cat = bool(j < len(cat) and cat[j])
            if is_cat and name not in binary_cat_names:
                continue
            col = pd.to_numeric(X_df.iloc[:, j], errors="coerce").to_numpy(dtype=np.float64)
            if not np.isfinite(col).any():
                continue
            score_idx.append(j)
            score_names.append(name)
            score_cols.append(col)
        if not score_cols:
            return [], []
        X_scored = np.ascontiguousarray(np.column_stack(score_cols), dtype=np.float64)
        selected = _core.select_interaction_information_features(
            X_scored,
            np.asarray(y_arr, dtype=np.int64),
            score_names,
            int(self.interaction_relaxed_feature_size),
            self.ii_partner_size,
        )
        name_to_idx = {name: score_idx[k] for k, name in enumerate(score_names)}
        relaxed_cols = [
            name_to_idx[str(row["name"])] for row in selected if str(row.get("name")) in name_to_idx
        ]
        return [dict(row) for row in selected], relaxed_cols

    @staticmethod
    def _interaction_relaxed_pairs_from_survivors(
        survivor_rows: list[dict],
        numeric_names: list[str],
    ) -> list[tuple[str, str]]:
        numeric = {str(name) for name in numeric_names}
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add_pair(left: str, right: str) -> None:
            if left == right or left not in numeric or right not in numeric:
                return
            key = tuple(sorted((left, right)))
            if key in seen:
                return
            seen.add(key)
            pairs.append((left, right))

        for row in survivor_rows:
            left = str(row.get("name")) if row.get("name") is not None else ""
            right = str(row.get("best_partner")) if row.get("best_partner") is not None else ""
            add_pair(left, right)
        return pairs

    def _select_pair_aware_adaptive_b(
        self,
        feature_name: str,
        feature_values: np.ndarray,
        partner_values: list[tuple[str, np.ndarray]],
        y_codes: np.ndarray,
        n_classes: int,
        candidates: list[int],
        ratio: float,
    ) -> tuple[int, dict[int, float], dict]:
        finite = np.isfinite(feature_values)
        if int(finite.sum()) < 10:
            chosen = candidates[len(candidates) // 2]
            scores = {int(b): 0.0 for b in candidates}
            return int(chosen), scores, {"mode": "insufficient_finite_values", "best_partner": None}

        scores: dict[int, float] = {}
        marginal_scores: dict[int, float] = {}
        best_partner_by_b: dict[int, dict] = {}
        scan_candidates = sorted({int(v) for v in candidates if int(v) >= 2})
        if partner_values:
            scan_candidates = [b for b in scan_candidates if b <= 8] or scan_candidates
        for b in scan_candidates:
            partner_bins = [int(b)]
            marginal, _edges, _codes = _edge_information_gain(feature_values, y_codes, n_classes, b)
            marginal_scores[b] = float(marginal)
            best_pair_score = 0.0
            best_partner_name: str | None = None
            best_partner_bins = None
            for partner_name, partner_col in partner_values:
                if int(np.isfinite(partner_col).sum()) < 10:
                    continue
                for pb in partner_bins:
                    joint, left_score, right_score = _joint_information_gain_from_binned_columns(
                        feature_values,
                        partner_col,
                        y_codes,
                        n_classes,
                        b,
                        pb,
                    )
                    conditional = max(0.0, float(joint) - float(right_score))
                    pair_score = max(
                        conditional, min(float(joint), max(float(left_score), float(right_score)))
                    )
                    if pair_score > best_pair_score:
                        best_pair_score = float(pair_score)
                        best_partner_name = partner_name
                        best_partner_bins = int(pb)
            scores[b] = float(max(marginal, best_pair_score))
            best_partner_by_b[b] = {
                "partner": best_partner_name,
                "partner_bins": best_partner_bins,
                "pair_score": float(best_pair_score),
                "marginal_score": float(marginal),
            }

        if not scores:
            chosen = candidates[len(candidates) // 2]
            return (
                int(chosen),
                {int(b): 0.0 for b in candidates},
                {"mode": "no_scores", "best_partner": None},
            )
        best_score = max(scores.values())
        if best_score <= 0.0:
            chosen = min(scores)
        elif partner_values:
            threshold = best_score * 0.85
            eligible = [b for b in sorted(scores) if scores[b] >= threshold]
            chosen = eligible[0] if eligible else max(scores, key=scores.get)
        else:
            threshold = best_score * max(0.0, 1.0 - float(ratio))
            eligible = [b for b in sorted(scores) if scores[b] >= threshold]
            chosen = eligible[0] if eligible else max(scores, key=scores.get)
        evidence = dict(best_partner_by_b.get(int(chosen), {}))
        evidence.update(
            {
                "mode": "pair_aware" if partner_values else "marginal",
                "feature": feature_name,
                "chosen_b": int(chosen),
                "score": float(scores.get(int(chosen), 0.0)),
                "best_score": float(best_score),
                "marginal_scores": {int(k): float(v) for k, v in marginal_scores.items()},
            }
        )
        return int(chosen), {int(k): float(v) for k, v in scores.items()}, evidence

    def _apply_pair_aware_adaptive_binning(self, X_train: Any, y_arr: np.ndarray) -> Any:
        is_df = isinstance(X_train, pd.DataFrame)
        X_df = self._frame_from_input(X_train, getattr(self, "feature_names_in_", None))
        base_candidates = sorted(
            {int(v) for v in (self.b_candidates or [2, 3, 5, 7, 10, 15]) if int(v) >= 2}
        )
        candidates = sorted(set(base_candidates + [4, 6, 8]))
        ratio = float(self.min_marginal_gain_ratio)
        cat_mask = self.cat_cols_mask_
        col_names = list(X_df.columns)
        y_codes = np.asarray(pd.factorize(np.asarray(y_arr), sort=True)[0], dtype=np.int64)
        n_classes = int(len(np.unique(y_codes)))

        survivors, _ = self._select_interaction_relaxed_survivors_from_frame(
            X_df,
            y_codes,
            [str(name) for name in col_names],
            cat_mask,
        )
        self.interaction_relaxed_mining_survivors_ = [dict(row) for row in survivors]

        num_cols = [
            (j, name) for j, name in enumerate(col_names) if not (j < len(cat_mask) and cat_mask[j])
        ]
        num_names = [str(name) for _, name in num_cols]
        pair_names = self._interaction_relaxed_pairs_from_survivors(survivors, num_names)
        partners_by_feature: dict[str, list[str]] = {str(name): [] for name in num_names}
        for left, right in pair_names:
            partners_by_feature.setdefault(left, []).append(right)
            partners_by_feature.setdefault(right, []).append(left)

        raw_numeric: dict[str, np.ndarray] = {
            str(name): pd.to_numeric(X_df.iloc[:, j], errors="coerce").to_numpy(dtype=np.float64)
            for j, name in num_cols
        }
        self._bin_edges_: dict = {}
        self.per_feature_b_: dict = {}
        self.ig_scores_: dict = {}
        self._interaction_relaxed_adaptive_pairs_ = [
            {"left": left, "right": right} for left, right in pair_names
        ]
        self._interaction_relaxed_adaptive_evidence_ = {}

        # One native call computes chosen bin counts for every numeric
        # column at once: pair-aware scoring for survivor columns with a
        # known partner, ordinary marginal-IG elbow selection for everything
        # else. See native/augmented_pair.cpp::select_pair_aware_adaptive_bins
        # for the full algorithm. A pure-Python column-by-column path
        # (_select_pair_aware_adaptive_b below) is used when the native
        # function is unavailable or raises, matching the fallback pattern
        # used elsewhere in this module for other native/Python pairs.
        use_native_pair_aware = bool(
            _CORE_AVAILABLE and hasattr(_core, "select_pair_aware_adaptive_bins")
        )
        chosen_b_by_name: dict[str, int] = {}
        scores_by_name: dict[str, dict[int, float]] = {}
        evidence_by_name: dict[str, dict] = {}
        if use_native_pair_aware and num_names:
            name_to_idx = {name: k for k, name in enumerate(num_names)}
            pair_idx_list = [
                (name_to_idx[left], name_to_idx[right])
                for left, right in pair_names
                if left in name_to_idx and right in name_to_idx
            ]
            X_numeric_mat = np.ascontiguousarray(
                np.column_stack([raw_numeric[name] for name in num_names])
                if num_names
                else np.zeros((len(y_codes), 0)),
                dtype=np.float64,
            )
            try:
                native_result = _core.select_pair_aware_adaptive_bins(
                    X_numeric_mat,
                    y_codes,
                    num_names,
                    [int(c) for c in candidates],
                    pair_idx_list,
                    float(ratio),
                    0.85,
                    8,
                )
                chosen_b_by_name = {
                    str(k): int(v) for k, v in dict(native_result["chosen_b"]).items()
                }
                scores_by_name = {
                    str(k): {int(kk): float(vv) for kk, vv in dict(v).items()}
                    for k, v in dict(native_result["scores"]).items()
                }
                evidence_by_name = {
                    str(k): dict(v) for k, v in dict(native_result["evidence"]).items()
                }
            except Exception:
                use_native_pair_aware = False

        for _j, name_obj in num_cols:
            name = str(name_obj)
            col = raw_numeric[name]
            if use_native_pair_aware and name in chosen_b_by_name:
                chosen = chosen_b_by_name[name]
                scores = scores_by_name.get(name, {int(b): 0.0 for b in candidates})
                evidence = evidence_by_name.get(
                    name,
                    {
                        "mode": "marginal",
                        "feature": name,
                        "chosen_b": int(chosen),
                        "best_partner": None,
                    },
                )
            else:
                partner_values = [
                    (partner_name, raw_numeric[partner_name])
                    for partner_name in partners_by_feature.get(name, [])
                    if partner_name in raw_numeric
                ]
                if partner_values:
                    chosen, scores, evidence = self._select_pair_aware_adaptive_b(
                        name,
                        col,
                        partner_values,
                        y_codes,
                        n_classes,
                        candidates,
                        ratio,
                    )
                else:
                    finite_mask = np.isfinite(col)
                    if int(finite_mask.sum()) < 10:
                        chosen = candidates[len(candidates) // 2]
                        scores = {int(b): 0.0 for b in candidates}
                    else:
                        chosen, scores = _adap_select_b(
                            col[finite_mask], y_codes[finite_mask], candidates, ratio
                        )
                    evidence = {
                        "mode": "marginal",
                        "feature": name,
                        "chosen_b": int(chosen),
                        "best_partner": None,
                    }
            edges = _adap_quantile_edges(col, int(chosen))
            self._bin_edges_[name_obj] = edges
            self.per_feature_b_[name_obj] = len(edges) - 1
            self.ig_scores_[name_obj] = {int(k): float(v) for k, v in scores.items()}
            self._interaction_relaxed_adaptive_evidence_[name] = evidence

        self._adaptive_code_label_map_: dict[str, str] = {}
        self._adaptive_precoded_features_ = set(self._bin_edges_)
        new_cat = cat_mask.copy()
        new_int = getattr(self, "is_int_mask_", np.zeros(len(col_names), dtype=bool)).copy()
        X_pre = X_df.copy()
        for name_obj, edges in self._bin_edges_.items():
            if name_obj not in X_df.columns:
                continue
            name = str(name_obj)
            col = pd.to_numeric(X_df[name_obj], errors="coerce").values
            n_bins = len(edges) - 1
            has_nan = not np.isfinite(col).all()
            j = col_names.index(name_obj) if name_obj in col_names else -1
            codes = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1).astype(np.float64)
            if has_nan:
                codes[~np.isfinite(col)] = np.nan
            X_pre[name_obj] = codes
            if j >= 0:
                new_cat[j] = False
                new_int[j] = True
            for k in range(n_bins):
                cpp_label = f"{name_obj}=[{float(k):.3f},{float(k + 1):.3f}]"
                orig_label = f"{name_obj}=[{edges[k]:.4g},{edges[k + 1]:.4g})"
                self._adaptive_code_label_map_[cpp_label] = orig_label
        self.cat_cols_mask_ = new_cat
        self.is_int_mask_ = new_int
        return X_pre if is_df else X_pre

    def _handle_test_nan(self, X_test: Any) -> tuple:
        """Apply training-time missing-column bin edges at test time.

        In constant-B non-adaptive models, this converts only the columns recorded
        in ``_missing_col_edges_`` back to the exact string-label representation
        used at fit time.  The rule is identical for L == 1 and L > 1: only
        numeric columns that had NaN/Inf during training are recorded here.
        Numeric columns not recorded in ``_missing_col_edges_`` remain numeric
        and are binned by C++ directly; if they contain new non-finite values at
        prediction time, the native numeric transaction builder skips them.

        Returns ``(X_modified, local_cat_mask)``.  ``self.cat_cols_mask_``
        is never mutated; ``local_cat_mask`` is a per-call copy.
        """
        base_cat = getattr(self, "cat_cols_mask_", None)
        if base_cat is None:
            return X_test, base_cat

        is_df = isinstance(X_test, pd.DataFrame)
        feat_names = getattr(self, "feature_names_in_", None)
        if is_df:
            X_df = X_test
        else:
            arr = np.asarray(X_test)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            cols = (
                list(feat_names)
                if feat_names is not None and len(feat_names) == arr.shape[1]
                else [f"col{j}" for j in range(arr.shape[1])]
            )
            X_df = pd.DataFrame(arr, columns=cols)

        col_names = list(X_df.columns)
        missing_edges = getattr(self, "_missing_col_edges_", {})
        local_cat = base_cat.copy()
        modified = False

        for j, name in enumerate(col_names):
            if j >= len(base_cat):
                continue

            # Column was pre-binned at training time because it contained
            # NaN/Inf during fit: convert raw float values to the same string
            # bin labels so C++ item lookups match.
            if name in missing_edges:
                col = pd.to_numeric(X_df.iloc[:, j], errors="coerce").values
                if not modified:
                    X_df = X_df.copy()
                    modified = True
                X_df[name] = _adap_apply_edges(col, missing_edges[name])
                local_cat[j] = True
                continue

            # Genuinely categorical column or column not in model:
            # C++ already handles np.nan natively — no action needed.

        if not modified:
            return X_test, base_cat
        return (X_df if is_df else X_df.to_numpy()), local_cat

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

"""Transformation, schema checks, monitoring, and drift APIs."""

from __future__ import annotations

import math
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted

from hugiml._classifier_runtime import PredictionMonitor, _core
from hugiml._classifier_support import HUGIMLClassifier, logger
from hugiml._compat import check_array
from hugiml.exceptions import (
    HUGIMLDtypeDriftWarning,
    HUGIMLPredictionError,
    HUGIMLRangeWarning,
    HUGIMLSchemaError,
)


class _PredictionMixin:
    """Implement transform-time validation and monitoring operations."""

    def transform_patterns(self, X: Any) -> csr_matrix:
        """Return the binary HUG pattern matrix for X.

        Each column corresponds to one mined pattern. Entry ``(i, j)`` is 1
        when all items of pattern ``j`` appear in row ``i``. This is the
        pattern-space transform used by pattern inspection, pruning, and
        pattern-level explanation utilities.
        """
        check_is_fitted(self)
        if self._is_constant_prior_fallback_active():
            self._validate_test_input(X)
            return csr_matrix((self._n_rows_for_prediction(X), 0), dtype=np.float32)
        # ── Adaptive pre-binning ─────────────────────────────────────────
        if getattr(self, "adaptive_binning", False) and getattr(self, "_bin_edges_", None):
            X = self._prebin_for_predict(X)
        # ─────────────────────────────────────────────────────────────────
        return self._build_test_hup(X)

    def transform(self, X: Any) -> Any:
        """Return the fitted downstream feature representation for X.

        The returned columns are exactly the representation consumed by the
        fitted downstream estimator after feature-mode assembly, strict TopK
        selection, and logistic-regression canonicalization. Use
        :meth:`transform_patterns` when the binary HUG pattern matrix itself is
        required.

        When a patterns-only fit uses the zero-pattern constant-prior fallback,
        an empty sparse matrix of shape ``(n_samples, 0)`` is returned, matching
        :meth:`transform_patterns` and the fitted ``x_train_downstream_`` in that
        degenerate case.
        """
        check_is_fitted(self)
        if self._is_constant_prior_fallback_active():
            # Delegate to transform_patterns so that input validation and
            # row-count resolution run exactly once through the shared path.
            return self.transform_patterns(X)
        Z_patterns = self.transform_patterns(X)
        X_downstream = self._make_downstream_features(X, Z_patterns, fit=False)
        X_downstream = self._apply_strict_topk_budget_transform(X_downstream)
        return self._apply_lr_downstream_canonical_transform(X_downstream)

    def _build_test_hup(self, X_test: Any) -> csr_matrix:
        """Build the sparse binary pattern matrix for test data.

        This follows the single-pass path.
        """
        self._check_health()
        # In original_plus_patterns mode, a fitted model may legitimately have
        # zero mined patterns.  Return an empty pattern matrix and let
        # _make_downstream_features use the original feature block.
        if (
            len(getattr(self, "patterns_", [])) == 0
            and getattr(self, "feature_mode", "patterns_only") != "patterns_only"
        ):
            return csr_matrix((len(X_test), 0), dtype=np.float32)
        # ── Non-finite handling ──────────────────────────────────────────
        if not getattr(self, "adaptive_binning", False):
            X_test, _cat_mask = self._handle_test_nan(X_test)
        else:
            _cat_mask = getattr(self, "cat_cols_mask_", None)
        # ─────────────────────────────────────────────────────────────────
        self._validate_test_input(X_test)
        X_num, X_cat_raw = self._to_float_array(X_test, _cat_mask)
        try:
            X_num = check_array(X_num, dtype=None, ensure_all_finite=False)
        except TypeError:
            X_num = check_array(X_num, dtype=None, force_all_finite=False)

        # Exclude exactly the columns identified as zero-variance during
        # fit(), regardless of what they hold in this batch -- the fitted
        # model never learned anything from them, so prediction ignores them
        # the same way every time rather than re-deciding per call.
        zero_variance_cols = getattr(self, "_zero_variance_cols_", None)
        if zero_variance_cols and _cat_mask is not None:
            self._exclude_zero_variance_columns(X_num, X_cat_raw, _cat_mask, zero_variance_cols)

        n = X_num.shape[0]
        X_cat_arg = X_cat_raw if any(v is not None for v in X_cat_raw) else None

        # Single-pass path. Prefer native CSR output to avoid copying COO
        # row/column arrays into Python and then asking scipy to sort/compress
        # them again. build_test_matrix remains a compatibility fallback.
        n_pats = len(self.patterns_)
        if getattr(self, "_native_available_", True):
            try:
                if hasattr(_core, "build_test_matrix_csr"):
                    indptr, indices = _core.build_test_matrix_csr(
                        X_num,
                        self.td_,
                        X_cat_arg,
                        self.patterns_,
                    )
                    data = np.ones(len(indices), dtype=np.float32)
                    return csr_matrix((data, indices, indptr), shape=(n, n_pats), dtype=np.float32)
                rows, cols = _core.build_test_matrix(
                    X_num,
                    self.td_,
                    X_cat_arg,
                    self.patterns_,
                )
                data = np.ones(len(rows), dtype=np.float32)
                return csr_matrix((data, (rows, cols)), shape=(n, n_pats), dtype=np.float32)
            except Exception:
                logger.debug(
                    "Native build_test_matrix failed; falling back to Python path.",
                    exc_info=True,
                )

        return self._build_test_hup_fallback(X_num, X_cat_raw, n, n_pats)

    def _build_test_hup_fallback(
        self,
        X_num: np.ndarray,
        X_cat_raw: list,
        n: int,
        n_pats: int,
    ) -> csr_matrix:
        """Pure-Python fallback for deserialized models without C++ extension."""
        td = self.td_
        p = X_num.shape[1]

        cpp_bn2id = td._cpp_bn2id
        cpp_stride = td._cpp_bkey_stride
        cpp_all_edges = td._cpp_all_edges
        cpp_nb_col = td._cpp_nb_col
        cpp_col_min = td._cpp_col_min
        cpp_col_range = td._cpp_col_range
        cpp_is_cat = td._cpp_is_cat
        cpp_is_int = td._cpp_is_int
        cpp_is_precoded = getattr(td, "_cpp_is_precoded", [])
        # If the wrapper was deserialized without _cpp_is_precoded, reconstruct it
        # from the classifier's compact adaptive precoded feature set.
        if not cpp_is_precoded:
            precoded_features = getattr(self, "_adaptive_precoded_features_", None)
            if precoded_features is None and getattr(self, "_adaptive_code_label_map_", {}):
                precoded_features = set(getattr(self, "_bin_edges_", {}))
            if precoded_features:
                feat_names = getattr(self, "feature_names_in_", None) or []
                cpp_is_precoded = [name in precoded_features for name in feat_names]
        cpp_cat_cats = td._cpp_cat_categories

        label2code: list[dict[object, int] | None] = [None] * p
        for j in range(p):
            if j < len(cpp_is_cat) and cpp_is_cat[j]:
                if j < len(cpp_cat_cats) and cpp_cat_cats[j]:
                    label2code[j] = {v: i for i, v in enumerate(cpp_cat_cats[j])}

        def bkey(bi: int, j: int) -> int:
            return int(bi * cpp_stride + j)

        test_trans_sets = []
        for r in range(n):
            items: set = set()
            for j in range(p):
                if j < len(cpp_is_cat) and cpp_is_cat[j]:
                    if X_cat_raw[j] is None:
                        continue
                    v = X_cat_raw[j][r]
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    _lc = label2code[j]
                    if _lc is None:
                        continue
                    lc: dict[object, int] = _lc
                    # cpp_cat_cats stores category labels as strings (the
                    # native side stringifies every category value), while
                    # X_cat_raw preserves the original typed value. Match the
                    # native convention here, or every categorical lookup
                    # silently misses and that column is dropped from every
                    # test-time transaction.
                    code = lc.get(str(v))
                    if code is None:
                        continue
                    bi = code + 1
                elif j < len(cpp_is_precoded) and cpp_is_precoded[j]:
                    # Pre-coded column: X_num[r, j] IS the 0-indexed bin code.
                    # bi = code + 1 directly — no scaling, no upper_bound.
                    code_val = X_num[r, j]
                    if not math.isfinite(code_val):
                        continue
                    nb = cpp_nb_col[j] if j < len(cpp_nb_col) else 0
                    bi = max(1, min(int(code_val) + 1, nb))
                else:
                    edges = cpp_all_edges[j]
                    if edges is None or len(edges) < 2:
                        continue
                    nb = cpp_nb_col[j]
                    raw = X_num[r, j]
                    # Non-finite values must generate no item — same contract
                    # as every other column type and the C++ build_test_matrix.
                    if not math.isfinite(raw):
                        continue
                    if j < len(cpp_is_int) and cpp_is_int[j]:
                        val = raw
                    else:
                        val = (raw - cpp_col_min[j]) / cpp_col_range[j]
                    inner = edges[1:-1] if isinstance(edges, np.ndarray) else np.array(edges[1:-1])
                    bi = int(np.searchsorted(inner, val, side="right")) + 1
                    bi = max(1, min(bi, nb))

                bk = bkey(bi, j)
                iid = cpp_bn2id.get(bk)
                if iid is not None:
                    items.add(iid)
            test_trans_sets.append(frozenset(items))

        rows_v, cols_v = [], []
        for pi, pe in enumerate(self.patterns_):
            pat_items = frozenset(pe.items)
            for tid, ts in enumerate(test_trans_sets):
                if pat_items.issubset(ts):
                    rows_v.append(tid)
                    cols_v.append(pi)

        data = np.ones(len(rows_v), dtype=np.float32)
        return csr_matrix((data, (rows_v, cols_v)), shape=(n, n_pats), dtype=np.float32)

    def _check_health(self) -> None:
        check_is_fitted(self)
        if self._is_constant_prior_fallback_active():
            return
        if not hasattr(self, "patterns_"):
            raise HUGIMLPredictionError("Pattern state missing — fit() may have failed.")
        if (
            len(self.patterns_) == 0
            and getattr(self, "feature_mode", "patterns_only") == "patterns_only"
        ):
            raise HUGIMLPredictionError("Model has no patterns — fit() may have failed.")
        if not hasattr(self, "model_"):
            raise HUGIMLPredictionError("Downstream model missing — fit() incomplete.")
        if not hasattr(self, "td_") or self.td_ is None:
            raise HUGIMLPredictionError("Transaction data missing — model state corrupt.")

    def _validate_test_input(self, X_test: Any) -> None:
        """Validate test-time input against training schema."""
        from scipy.sparse import issparse as _issparse

        if _issparse(X_test):
            raise ValueError(
                "HUGIMLClassifier does not support sparse input.  "
                "Convert to a dense array via X.toarray() first."
            )
        is_df = isinstance(X_test, pd.DataFrame)
        arr = None
        if not is_df:
            arr = np.asarray(X_test)
            if arr.ndim == 1:
                raise ValueError(
                    f"HUGIMLClassifier expects a 2D array, got 1D array of shape {arr.shape}."
                )
        n_test_features = (
            len(X_test.columns) if is_df else arr.shape[1]  # type: ignore[union-attr]
        )

        expected = getattr(self, "n_features_in_", None)
        if expected is not None and n_test_features != expected:
            raise HUGIMLSchemaError(
                f"X has {n_test_features} features, but the model was fitted "
                f"with {expected} features."
            )

        expected_names = getattr(self, "feature_names_in_", None)
        if is_df and expected_names is not None:
            test_names = [str(c) for c in X_test.columns]
            if test_names != expected_names:
                missing = set(expected_names) - set(test_names)
                extra = set(test_names) - set(expected_names)
                parts = []
                if missing:
                    parts.append(f"missing: {sorted(missing)}")
                if extra:
                    parts.append(f"unexpected: {sorted(extra)}")
                if not missing and not extra:
                    parts.append("columns in different order")
                raise HUGIMLSchemaError(
                    "Column mismatch between training and test data.  " + "; ".join(parts)
                )

        cat_mask = getattr(self, "cat_cols_mask_", None)
        if is_df and cat_mask is not None and np.any(cat_mask):
            for j, is_cat in enumerate(cat_mask):
                if j >= n_test_features:
                    break
                col = X_test.iloc[:, j]
                if is_cat and pd.api.types.is_numeric_dtype(col):
                    warnings.warn(
                        f"Column '{X_test.columns[j]}' was categorical during "
                        f"training but has numeric dtype ({col.dtype}) in test data.",
                        HUGIMLDtypeDriftWarning,
                        stacklevel=4,
                    )

        if is_df and cat_mask is not None:
            td = self.td_
            cpp_all_edges = getattr(td, "_cpp_all_edges", None)
            if cpp_all_edges is not None:
                try:
                    numeric_idx = [
                        j
                        for j in range(min(n_test_features, len(cat_mask), len(cpp_all_edges)))
                        if not cat_mask[j]
                        and cpp_all_edges[j] is not None
                        and len(cpp_all_edges[j]) >= 2
                    ]
                    if numeric_idx:
                        train_min = np.asarray(
                            [float(cpp_all_edges[j][0]) for j in numeric_idx], dtype=float
                        )
                        train_max = np.asarray(
                            [float(cpp_all_edges[j][-1]) for j in numeric_idx], dtype=float
                        )
                        cpp_col_min = getattr(td, "_cpp_col_min", None)
                        cpp_col_range = getattr(td, "_cpp_col_range", None)
                        if cpp_col_min is not None and cpp_col_range is not None:
                            cm = np.asarray(
                                [float(cpp_col_min[j]) for j in numeric_idx], dtype=float
                            )
                            cr = np.asarray(
                                [float(cpp_col_range[j]) for j in numeric_idx], dtype=float
                            )
                            ok = np.isfinite(cr) & (cr > 0.0) & np.isfinite(cm)
                            train_min[ok] = train_min[ok] * cr[ok] + cm[ok]
                            train_max[ok] = train_max[ok] * cr[ok] + cm[ok]
                        train_span = train_max - train_min
                        valid = train_span > 0
                        if np.any(valid):
                            vals = X_test.iloc[:, numeric_idx].to_numpy(
                                dtype=np.float64, copy=False
                            )
                            finite_vals = np.isfinite(vals)
                            test_min = np.full(len(numeric_idx), np.nan, dtype=float)
                            test_max = np.full(len(numeric_idx), np.nan, dtype=float)
                            observed = finite_vals.any(axis=0)
                            if np.any(observed):
                                observed_vals = np.where(
                                    finite_vals[:, observed], vals[:, observed], np.nan
                                )
                                test_min[observed] = np.nanmin(observed_vals, axis=0)
                                test_max[observed] = np.nanmax(observed_vals, axis=0)
                            drift = (
                                valid
                                & np.isfinite(test_min)
                                & np.isfinite(test_max)
                                & (
                                    (test_min < train_min - train_span * 0.5)
                                    | (test_max > train_max + train_span * 0.5)
                                )
                            )
                            for pos in np.flatnonzero(drift):
                                j = numeric_idx[int(pos)]
                                warnings.warn(
                                    f"Column '{X_test.columns[j]}' has values "
                                    f"[{float(test_min[pos]):.4g}, {float(test_max[pos]):.4g}] outside training "
                                    f"range [{float(train_min[pos]):.4g}, {float(train_max[pos]):.4g}].",
                                    HUGIMLRangeWarning,
                                    stacklevel=4,
                                )
                except Exception:
                    # Preserve prediction behaviour if warning-only drift checks
                    # cannot be vectorized for mixed/object inputs.
                    pass

    def enable_monitoring(self, window_size: int = 1000) -> HUGIMLClassifier:
        """Enable prediction monitoring.  Access via ``self.monitor``."""
        self.monitor = PredictionMonitor(window_size=window_size)
        return self

    def disable_monitoring(self) -> HUGIMLClassifier:
        """Disable prediction monitoring."""
        self.monitor = None
        return self

    def detect_drift(
        self,
        X_test: Any,
        y_test: np.ndarray | None = None,
        threshold: float = 0.1,
    ) -> str:
        """Run multi-method drift detection and return a human-readable report.

        Uses PSI + KL divergence.  When ``y_test`` is provided, also checks
        label distribution drift.

        Notes
        -----
        Drift metrics are computed on the numeric array retained by the mining
        path.  Constant-B numeric columns that contained NaN/Inf during training are
        converted to the categorical bin-label path so missingness is handled
        consistently at fit/predict time; those columns are therefore not
        represented as continuous numeric drift baselines.  PSI/KL alerts for
        such columns should be interpreted through pattern/feature-importance
        diagnostics rather than through ``detect_drift()``.

        Parameters
        ----------
        X_test : array-like or DataFrame
        y_test : array-like, optional
        threshold : float

        Returns
        -------
        str
        """
        check_is_fitted(self)
        if getattr(self, "_drift_det", None) is None:
            if self._is_production_mode():
                raise RuntimeError(self._audit_artifact_message("Drift-detection baseline"))
            return "Drift detection unavailable (no baseline stored)."
        cat_mask = getattr(self, "cat_cols_mask_", np.zeros(0, dtype=bool))
        X_num, _ = self._to_float_array(X_test, cat_mask)
        y_arr = np.asarray(y_test) if y_test is not None else None
        report = self._drift_det.detect(X_num, y_test=y_arr, threshold=threshold)
        return str(report)

    def get_drift_psi(self, X_test: Any) -> dict:
        """Return per-feature PSI values as a dict.

        See ``detect_drift()`` for the constant-B missing-numeric limitation: columns
        that were routed to categorical bin labels because they contained
        NaN/Inf during training do not have meaningful continuous PSI baselines.
        """
        check_is_fitted(self)
        if getattr(self, "_drift_det", None) is None:
            if self._is_production_mode():
                raise RuntimeError(self._audit_artifact_message("Drift PSI baseline"))
            return {}
        cat_mask = getattr(self, "cat_cols_mask_", np.zeros(0, dtype=bool))
        X_num, _ = self._to_float_array(X_test, cat_mask)
        return self._drift_det.compute_psi(X_num)

    def cross_validate_monitored(
        self,
        X: Any,
        y: Any,
        cv: Any = None,
        scoring: str = "roc_auc",
    ) -> dict:
        """Cross-validation with per-fold monitoring and drift detection.

        Parameters
        ----------
        X : pd.DataFrame or ndarray
        y : array-like
        cv : int or CV splitter (default: StratifiedKFold(5))
        scoring : str

        Returns
        -------
        dict with keys: test_scores, fit_times_ms, fold_monitors, fold_drift, fold_metadata
        """
        from sklearn.metrics import get_scorer
        from sklearn.model_selection import StratifiedKFold

        y = np.asarray(y)
        if cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        elif isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        scorer = get_scorer(scoring)
        results: dict = {
            "test_scores": [],
            "fit_times_ms": [],
            "fold_monitors": [],
            "fold_drift": [],
            "fold_metadata": [],
        }

        # parallelise CV folds.
        # Each fold is fully independent (separate clone, separate fit).
        # prefer="threads" avoids pickling the classifier and its C++ state;
        # the GIL is released during C++ mining so threads still scale.
        # Falls back to serial when n_jobs=1 (the default).
        base_params = {
            k: v for k, v in self.get_params().items() if k not in ("allCols", "origColumns")
        }
        clf_cls = self.__class__

        def _fit_fold(train_idx, test_idx):
            if isinstance(X, pd.DataFrame):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            fold_clf = clf_cls(**base_params)
            t0 = time.perf_counter()
            fold_clf.fit(X_tr, y_tr)
            fit_ms = (time.perf_counter() - t0) * 1000
            score = scorer(fold_clf, X_te, y_te)

            fold_clf.enable_monitoring()
            fold_clf.predict_proba(X_te)
            fold_mon = fold_clf.monitor
            mon_stats = fold_mon.stats if fold_mon is not None else {}

            psi: dict = {}
            if getattr(fold_clf, "_drift_det", None) is not None:
                cat_mask = getattr(fold_clf, "cat_cols_mask_", np.zeros(0, dtype=bool))
                X_te_num, _ = fold_clf._to_float_array(X_te, cat_mask)
                psi = fold_clf._drift_det.compute_psi(X_te_num)

            return score, fit_ms, fold_clf.fit_metadata_, mon_stats, psi

        try:
            from joblib import Parallel
            from joblib import delayed as _delayed

            _n_jobs = self.n_jobs if hasattr(self, "n_jobs") else 1
            fold_outputs = Parallel(n_jobs=_n_jobs, prefer="threads")(
                _delayed(_fit_fold)(tr, te) for tr, te in cv.split(X, y)
            )
        except Exception:
            fold_outputs = [_fit_fold(tr, te) for tr, te in cv.split(X, y)]

        for score, fit_ms, metadata, mon_stats, psi in fold_outputs:
            results["test_scores"].append(score)
            results["fit_times_ms"].append(fit_ms)
            results["fold_metadata"].append(metadata)
            results["fold_monitors"].append(mon_stats)
            results["fold_drift"].append(psi)

        return results

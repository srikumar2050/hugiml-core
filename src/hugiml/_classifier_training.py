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

"""Fit orchestration, native mining, fallback models, and prediction."""

from __future__ import annotations

import os
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from hugiml._classifier_runtime import _CORE_AVAILABLE, _CORE_IMPORT_ERROR, DriftDetector, _core
from hugiml._classifier_support import (
    FitMetadata,
    HUGIMLClassifier,
    _get_peak_rss_kb,
    _MemoryTracker,
    _TransactionDataWrapper,
    _wire_hugiml_feature_metadata,
    logger,
)
from hugiml._compat import check_X_y
from hugiml.exceptions import (
    HUGIMLMemoryError,
    HUGIMLTimeoutError,
    HUGIMLValidationError,
    HUGIMLWarning,
)


class _TrainingMixin:
    """Implement fit orchestration and fitted-model prediction."""

    def _require_core(self) -> None:
        """Raise a diagnostic ImportError if the native extension is absent."""
        if not _CORE_AVAILABLE:
            import platform
            import sys

            cause = f"\n  Original error: {_CORE_IMPORT_ERROR}" if _CORE_IMPORT_ERROR else ""
            raise ImportError(
                "HUGIMLClassifier requires the compiled C++ extension '_hugiml_core', "
                "which was not importable at package load time."
                f"{cause}\n\n"
                f"  Platform : {platform.system()} {platform.machine()}\n"
                f"  Python   : {sys.version.split()[0]}\n"
                f"  sys.path : {sys.path}\n\n"
                "Likely causes and remedies:\n"
                "  1. No pre-built wheel for your platform (most common).\n"
                "     Check https://pypi.org/project/hugiml-core/#files for available wheels.\n"
                "     Linux x86_64 wheels require glibc >= 2.17. "
                "Run 'ldd --version' to check yours.\n"
                "  2. pip fell back to the sdist and the C++ build failed.\n"
                "     Look for gcc/g++ errors above in the pip install output.\n"
                "  3. The .so was installed but is not on sys.path (rare; "
                "check for a broken venv).\n\n"
                "To build from source (requires gcc/g++ and pybind11):\n"
                "  pip install . --no-build-isolation\n"
                "Fast development build:\n"
                "  HUGIML_FAST_BUILD=1 python setup.py build_ext --inplace\n"
                "Verify after install:\n"
                '  python -c "import hugiml; assert hugiml.check_native(), '
                "'native missing'\""
            )

    def fit(self, X: Any, y: Any) -> HUGIMLClassifier:
        """Fit the HUG-IML model on training data.

        Parameters
        ----------
        X : pd.DataFrame or ndarray, shape (n_samples, n_features)
        y : array-like of int, shape (n_samples,)

        Returns
        -------
        self

        Thread safety
        -------------
        fit() acquires an exclusive lock.  Concurrent fit() calls on the same
        instance are serialized.  predict/predict_proba/transform are read-only
        on fitted state and safe for concurrent use after fit() returns.
        """
        self._require_core()
        with self._fit_lock:
            try:
                return self._fit_impl(X, y)
            except MemoryError as exc:
                raise HUGIMLMemoryError(
                    "HUGIML fit failed cleanly because required memory could not be allocated. "
                    "Reduce n/p/topK/B, keep adaptive_binning=True/use_hotpath=True, or increase "
                    "the process memory limit. Original error: " + str(exc)
                ) from exc
            except RuntimeError as exc:
                if "hugiml_timeout" in str(exc):
                    raise HUGIMLTimeoutError(str(exc)) from exc
                raise

    def _fit_impl(self, X_train: Any, y_train: Any) -> HUGIMLClassifier:
        # Clear all fitted state so that re-fitting the same instance is
        # idempotent.  Without this, _resolve_col_meta() short-circuits on
        # the cached cat_cols_mask_ from the previous fit, causing the column
        # type masks to carry over and producing non-reproducible results.
        for _attr in (
            "cat_cols_mask_",
            "is_int_mask_",
            "feature_names_in_",
            "binary_categorical_cols_",
            "_bin_edges_",
            "_missing_col_edges_",
            "_adaptive_code_label_map_",
            "per_feature_b_",
            "ig_scores_",
            "patterns_",
            "model_",
            "classes_",
            "x_train_hup_",
            "fit_metadata_",
            "_original_scaler_",
            "_original_numeric_medians_",
            "_original_numeric_medians_array_",
            "_original_feature_names_downstream_",
            "_pattern_orders_",
            "_interaction_pattern_mask_",
            "x_train_downstream_",
            "_augmented_pair_block_",
            "augmented_pair_transforms_",
            "augmented_pair_selected_features_",
            "_native_original_feature_names_downstream_",
            "_native_original_feature_scores_downstream_",
            "_strict_topk_applied_during_construction_",
            "_strict_topk_feature_mask_",
            "_strict_topk_feature_scores_",
            "_strict_topk_selected_feature_names_",
            "_downstream_feature_names_full_",
            "_training_pattern_matrix_shape_",
            "_training_pattern_matrix_nnz_",
            "_training_downstream_matrix_shape_",
            "_training_downstream_matrix_nnz_",
            "_drift_det",
            "fallback_active_",
            "fallback_strategy_",
            "fallback_reason_",
            "fallback_class_prior_",
            "fallback_majority_class_",
            "fallback_n_samples_",
        ):
            self.__dict__.pop(_attr, None)

        self.fallback_active_ = False

        t_total = self._timer()
        stage_times: dict[str, float] = {}

        # Reject sparse matrices with an informative message
        from scipy.sparse import issparse as _issparse

        if _issparse(X_train):
            raise ValueError(
                "HUGIMLClassifier does not support sparse input.  "
                "Convert to a dense array via X.toarray() first."
            )

        # Reject complex-valued arrays
        if hasattr(X_train, "dtype") and np.iscomplexobj(X_train):
            raise ValueError("Complex data not supported by HUGIMLClassifier.")

        self._validate_params()

        # Configure OpenMP before adaptive binning.  Adaptive B-selection is
        # column-parallel in the native path, so applying n_jobs after adaptive
        # preprocessing is too late.
        n_threads = _core.openmp_get_max_threads() if self.n_jobs == -1 else self.n_jobs
        if n_threads > 0:
            _core.openmp_set_num_threads(n_threads)
        actual_threads = _core.openmp_get_max_threads()

        # fast_grid_tune can pass a precomputed adaptive-binning context for
        # cache-building fits.  This is strictly fit-local and never becomes
        # persisted estimator state.  It is only used when the caller has
        # already selected bins on the exact same training fold with the same
        # adaptive parameters; mining still runs independently for each exact
        # (G, L, topK) tuple, so pattern budgets and rankings are unchanged.
        _fast_tune_adaptive_context = getattr(self, "_fast_tune_adaptive_context", None)
        _fast_tune_has_adaptive_context = isinstance(_fast_tune_adaptive_context, dict)

        # Preserve raw input if the downstream mode needs original features,
        # or if L>1 will create internal augmented_pair_transforms. This remains
        # an internal operation; no public hyperparameter is added.
        _needs_augmented_pairs = bool(self.L > 1 and bool(self.augmented_pair_transforms))
        X_train_original_for_downstream = (
            self._copy_input_for_downstream(X_train)
            if (self.feature_mode != "patterns_only" or _needs_augmented_pairs)
            else None
        )

        # Fused adaptive+L1 hot path can consume raw X directly and must not
        # materialise the intermediate X_codes matrix/DataFrame.
        _use_fused_adaptive_l1 = (
            self.adaptive_binning
            and self.use_hotpath
            and _CORE_AVAILABLE
            and self.L == 1
            and not _fast_tune_has_adaptive_context
            and hasattr(_core, "prepare_and_mine_l1_adaptive")
        )

        # ── Adaptive pre-binning (C++ hot path or Python fallback) ────────
        # ── Adaptive B-selection always uses C++ ──────────────────────────
        # _apply_adaptive_binning_cpp calls _core.select_adaptive_bins
        # (elbow_stop_nb_cpp) whenever the C++ extension is available.
        # use_hotpath does NOT gate this: C++ adaptive selection is always
        # preferred because it produces identical outputs with no conflicts.
        # Python _apply_adaptive_binning is kept as a fallback for
        # environments where the C++ extension is absent.
        if (
            self.adaptive_binning
            and _fast_tune_has_adaptive_context
            and "X_pre" in _fast_tune_adaptive_context
        ):
            _ctx = _fast_tune_adaptive_context
            _ctx_attrs = dict(_ctx.get("attrs", {}))
            for _name, _value in _ctx_attrs.items():
                # Shallow copies are enough for dict/list/set/ndarray metadata and
                # prevent accidental mutation from one candidate leaking into the
                # next.  The pre-binned X object itself is immutable-by-contract
                # inside fast_grid_tune cache fits.
                if isinstance(_value, dict):
                    _value = dict(_value)
                elif isinstance(_value, set):
                    _value = set(_value)
                elif isinstance(_value, list):
                    _value = list(_value)
                elif isinstance(_value, np.ndarray):
                    _value = _value.copy()
                setattr(self, _name, _value)
            X_train = _ctx["X_pre"]
            stage_times["adaptive_binning"] = 0.0
            stage_times["adaptive_binning_cache_hit"] = 1.0
            if self.verbose:
                logger.info(
                    "  adaptive binning: reused fast_grid_tune context, %d features pre-binned",
                    len(getattr(self, "_bin_edges_", {})),
                )
        elif self.adaptive_binning and not _use_fused_adaptive_l1:
            self._resolve_col_meta(X_train)  # prime cat_cols_mask_ first
            _y_for_ig = self._safe_cast_y(y_train)
            _use_pair_aware_adaptive = bool(
                self.L != 1 and bool(getattr(self, "interaction_relaxed_mining", False))
            )
            if _use_pair_aware_adaptive:
                X_train = self._apply_pair_aware_adaptive_binning(X_train, _y_for_ig)
            elif _CORE_AVAILABLE and hasattr(_core, "select_adaptive_bins"):
                X_train = self._apply_adaptive_binning_cpp(X_train, _y_for_ig)
            else:
                X_train = self._apply_adaptive_binning(X_train, _y_for_ig)
            if self.verbose:
                logger.info(
                    "  adaptive binning: %d features pre-binned, B_j in [%d, %d]",
                    len(self._bin_edges_),
                    min(self.per_feature_b_.values(), default=0),
                    max(self.per_feature_b_.values(), default=0),
                )
            if _fast_tune_has_adaptive_context and "X_pre" not in _fast_tune_adaptive_context:
                _attrs: dict[str, Any] = {}
                for _name in (
                    "cat_cols_mask_",
                    "is_int_mask_",
                    "feature_names_in_",
                    "binary_categorical_cols_",
                    "_bin_edges_",
                    "_missing_col_edges_",
                    "_adaptive_code_label_map_",
                    "_adaptive_precoded_features_",
                    "per_feature_b_",
                    "ig_scores_",
                ):
                    if hasattr(self, _name):
                        _value = getattr(self, _name)
                        if isinstance(_value, dict):
                            _value = dict(_value)
                        elif isinstance(_value, set):
                            _value = set(_value)
                        elif isinstance(_value, list):
                            _value = list(_value)
                        elif isinstance(_value, np.ndarray):
                            _value = _value.copy()
                        _attrs[_name] = _value
                _fast_tune_adaptive_context["X_pre"] = X_train
                _fast_tune_adaptive_context["attrs"] = _attrs
                _fast_tune_adaptive_context["misses"] = (
                    int(_fast_tune_adaptive_context.get("misses", 0)) + 1
                )
        # ─────────────────────────────────────────────────────────────────

        # ── Constant-B non-finite handling (non-adaptive path) ───────────────
        # Use one consistent scheme for every L: numeric columns stay numeric
        # unless they contain NaN/Inf during training.  Columns with training
        # non-finite cells are pre-binned to the string/categorical path so
        # fit and predict use the same missing-value representation.  Clean
        # numeric columns, including L>1 columns, use the native numeric path;
        # new test-time NaN/Inf values are skipped by native item generation.
        if not self.adaptive_binning:
            self._resolve_col_meta(X_train)
            X_train = self._prebin_nan_cols(X_train)
        # ─────────────────────────────────────────────────────────────────

        # Line-level allocation tracing (tracemalloc, inside _MemoryTracker)
        # is independent of the audit/production retention distinction below.
        # execution_mode='production' skips it, since it can dominate fit()
        # wall time on categorical-heavy inputs and the line-level breakdown
        # it provides is diagnostic rather than prediction-relevant.
        # execution_mode='audit' (the default) keeps it active, tracking the
        # same traced_peak_mb granularity as every other stage of fit().
        #
        # Internal cache-building fits performed by fast_grid_tune() set
        # _fast_tune_cache_only and are tagged execution_mode='audit' purely
        # so x_train_hup_ / x_train_downstream_ survive for reuse across the
        # feature_mode variants built from the same cached fit -- that need
        # is about which arrays are retained afterward, not about whether
        # allocation tracing runs during the fit itself, so those fits skip
        # tracing too regardless of the execution_mode tag they carry.
        mem = _MemoryTracker(
            enable_tracing=(
                getattr(self, "execution_mode", "audit") != "production"
                and not getattr(self, "_fast_tune_cache_only", False)
            )
        )
        with mem:
            # Stage 1: resolve column metadata
            t = self._timer()
            cat_mask = self._resolve_col_meta(X_train)
            int_mask = getattr(self, "is_int_mask_", None)

            X_num, X_cat_raw = self._to_float_array(X_train, cat_mask)
            y_train = self._safe_cast_y(y_train)
            # Native numeric paths treat non-finite feature cells as missing
            # observations and skip item generation.  Let NaN/Inf through
            # sklearn validation; y is already checked separately by _safe_cast_y.
            try:
                X_num, y_train = check_X_y(X_num, y_train, dtype=None, ensure_all_finite=False)
            except TypeError:
                X_num, y_train = check_X_y(X_num, y_train, dtype=None, force_all_finite=False)

            # Columns with at most one distinct observed value carry no
            # information for either path below to mine, so they're excluded
            # from the per-row work that would otherwise touch every row only
            # to confirm what a single distinct-value count already shows.
            # n_features_in_ / feature_names_in_ still reflect every column
            # exactly as given, so predict()'s schema check is unaffected --
            # only the per-column native processing is skipped.
            self._zero_variance_cols_ = self._identify_zero_variance_columns(X_train)
            self._exclude_zero_variance_columns(
                X_num, X_cat_raw, cat_mask, self._zero_variance_cols_
            )

            self.n_features_in_ = X_num.shape[1]
            self.classes_ = np.unique(y_train)
            n_cls = len(self.classes_)
            stage_times["resolve_meta"] = t.ms

            if n_cls < 2:
                raise HUGIMLValidationError(
                    f"y contains only {n_cls} class(es).  At least 2 are required."
                )
            if X_num.shape[0] < n_cls:
                raise HUGIMLValidationError(
                    f"Fewer samples ({X_num.shape[0]}) than classes ({n_cls})."
                )

            est_mb = _MemoryTracker.estimate_fit_mb(
                X_num.shape[0], X_num.shape[1], X_num.shape[1] * 10, self._effective_topK()
            )
            if est_mb > 4000:
                warnings.warn(
                    f"Estimated peak memory ~{est_mb:.0f} MB.  "
                    "Consider reducing topK or dataset size.",
                    HUGIMLWarning,
                    stacklevel=4,
                )

            if self.verbose:
                logger.info(
                    "HUGIMLClassifier.fit — %dx%d, %d classes",
                    X_num.shape[0],
                    X_num.shape[1],
                    n_cls,
                )

            # Stage 2+3+4: prepare / mine / build matrix
            # ── Fused L=1 hot path ─────────────────────────────────────────
            # When use_hotpath=True and L=1: a single C++ call replaces
            # prepare_transactions + mine_patterns + build_train_matrix.
            # No TransList, no hash-map lookups, direct COO from TID index.
            # Falls back to the original three-step path for L>1, adaptive
            # binning with Python fallback, or when explicitly disabled.
            t = self._timer()
            rss_before = _get_peak_rss_kb()
            col_names = getattr(self, "feature_names_in_", None)
            is_cat_np = cat_mask.astype(np.uint8)
            is_int_np = (
                int_mask if int_mask is not None else np.zeros(X_num.shape[1], dtype=bool)
            ).astype(np.uint8)

            # Build is_precoded mask without scanning every adaptive label key.
            is_precoded_np: np.ndarray | None = None
            if self.adaptive_binning:
                precoded_features = getattr(self, "_adaptive_precoded_features_", set())
                p_cols = X_num.shape[1]
                feat_names_list = (
                    col_names if col_names is not None else [f"col{j}" for j in range(p_cols)]
                )
                is_precoded_np = np.fromiter(
                    (name in precoded_features for name in feat_names_list),
                    dtype=np.uint8,
                    count=p_cols,
                )

            _use_fused = (
                self.use_hotpath
                and _CORE_AVAILABLE
                and self.L == 1
                and not _fast_tune_has_adaptive_context
                and hasattr(_core, "prepare_and_mine_l1")
            )

            if _use_fused:
                # ── Fused path ────────────────────────────────────────────────
                fit_deadline = self._mining_deadline_from_now()
                remaining_s = max(fit_deadline - time.perf_counter(), 0.0) if fit_deadline else 0.0
                K_eff = self._effective_mining_topK()  # rough pre-estimate (no n_items yet)

                if (
                    os.environ.get("HUGIML_DISABLE_NUMERIC_L1_FASTPATH", "0") != "1"
                    and (not self.adaptive_binning)
                    and hasattr(_core, "prepare_and_mine_l1_fixed_numeric")
                    and not bool(np.any(is_cat_np))
                ):
                    _l1_result = _core.prepare_and_mine_l1_fixed_numeric(
                        X_num,
                        y_train,
                        self.B,
                        col_names,
                        is_int_np,
                        K_eff,
                        self.G,
                        remaining_s,
                        compute_original_scores=(self.feature_mode != "patterns_only"),
                    )
                elif self.adaptive_binning and hasattr(_core, "prepare_and_mine_l1_adaptive"):
                    candidates = sorted(set(self.b_candidates or [2, 3, 5, 7, 10, 15]))
                    _l1_result = _core.prepare_and_mine_l1_adaptive(
                        X_num,
                        y_train,
                        col_names,
                        is_cat_np,
                        is_int_np,
                        X_cat_raw if any(v is not None for v in X_cat_raw) else None,
                        candidates,
                        self.min_marginal_gain_ratio,
                        K_eff,
                        self.G,
                        remaining_s,
                        compute_original_scores=(self.feature_mode != "patterns_only"),
                        adaptive_binning_sample_frac=self._adaptive_binning_sample_fraction_for_native(),
                        adaptive_binning_sample_random_state=int(
                            self.adaptive_binning_sample_random_state
                        ),
                    )

                    # Install adaptive metadata for predict()/transform() so test
                    # data is pre-binned to the same integer-code representation
                    # used by the fitted td.
                    feat_names_list = (
                        list(col_names)
                        if col_names is not None
                        else [f"col{j}" for j in range(X_num.shape[1])]
                    )
                    self._bin_edges_ = {}
                    self.per_feature_b_ = {}
                    self.ig_scores_ = {}
                    self._adaptive_code_label_map_ = {}
                    self._adaptive_precoded_features_ = set()
                    new_cat = cat_mask.copy()
                    new_int = (
                        int_mask.copy()
                        if int_mask is not None
                        else np.zeros(X_num.shape[1], dtype=bool)
                    )
                    for ci, col_res in enumerate(getattr(_l1_result, "adaptive_cols", [])):
                        j = int(_l1_result.adaptive_num_col_indices[ci])
                        name = feat_names_list[j]
                        edges = np.array(col_res.edges)
                        self._bin_edges_[name] = edges
                        self._adaptive_precoded_features_.add(name)
                        self.per_feature_b_[name] = len(edges) - 1
                        scores: dict[int, float] = {}
                        for k, b in enumerate(candidates):
                            scores[b] = col_res.ig_scores[k] if k < len(col_res.ig_scores) else 0.0
                        self.ig_scores_[name] = scores
                        new_cat[j] = False
                        new_int[j] = True
                        for k in range(len(edges) - 1):
                            cpp_label = f"{name}=[{float(k):.3f},{float(k + 1):.3f}]"
                            orig_label = f"{name}=[{edges[k]:.4g},{edges[k + 1]:.4g})"
                            self._adaptive_code_label_map_[cpp_label] = orig_label
                    self.cat_cols_mask_ = new_cat
                    self.is_int_mask_ = new_int
                else:
                    _l1_result = _core.prepare_and_mine_l1(
                        X_num,
                        y_train,
                        2 if self.adaptive_binning else self.B,
                        col_names,
                        is_cat_np,
                        is_int_np,
                        X_cat_raw if any(v is not None for v in X_cat_raw) else None,
                        is_precoded_np,
                        K_eff,
                        self.G,
                        remaining_s,
                        compute_original_scores=(self.feature_mode != "patterns_only"),
                    )
                self.td_ = _l1_result.td
                stage_times["l1_fused_hotpath"] = 1.0
                stage_times[
                    "l1_fused_adaptive_hotpath"
                    if self.adaptive_binning
                    else "l1_fused_fixed_hotpath"
                ] = 1.0
                if self.feature_mode != "patterns_only":
                    native_orig_names = [
                        f"orig:{name}"
                        for name in list(getattr(_l1_result, "original_feature_names", []) or [])
                    ]
                    native_orig_scores = np.asarray(
                        list(getattr(_l1_result, "original_feature_scores", []) or []),
                        dtype=np.float64,
                    )
                    if native_orig_names and len(native_orig_names) == len(native_orig_scores):
                        self._native_original_feature_names_downstream_ = native_orig_names
                        self._native_original_feature_scores_downstream_ = native_orig_scores
                cpp_mem_bytes = self.td_.memory_usage_bytes()
                n_items = len(self.td_.item_twu)
                K = self._effective_topK(n_items)
                stage_times["prepare_transactions"] = t.ms

                if self.verbose:
                    logger.info("  items=%d, K=%d [fused L=1 path]", n_items, K)

                t = self._timer()
                raw_patterns_list = list(_l1_result.patterns)
                self.raw_patterns_ = sorted(
                    raw_patterns_list, key=lambda pe: (-pe.utility, tuple(pe.items))
                )
                # L=1 singletons are unique by definition (each pattern is one
                # distinct item), so deduplication by coverage is a no-op.
                # Skip calling _deduplicate_patterns_by_coverage (which would
                # call build_train_matrix on the empty td.transactions) and use
                # the COO returned directly by the fused path.
                self.patterns_ = self.raw_patterns_
                stage_times["mine_patterns"] = t.ms

                if len(self.patterns_) == 0:
                    # Zero mined HUG patterns is not fatal.  In hybrid modes the
                    # downstream estimator can still use the original feature
                    # block.  In patterns_only mode, a common no-fail fallback
                    # below returns a constant-prior classifier.
                    n_train = len(y_train)
                    self.x_train_hup_ = csr_matrix((n_train, 0), dtype=np.float32)
                    stage_times["build_matrix"] = 0.0

                if self.verbose:
                    logger.info(
                        "  %d patterns in %.0f ms [fused]",
                        len(self.patterns_),
                        stage_times["mine_patterns"],
                    )

                t = self._timer()
                n_train = len(y_train)
                n_pats = len(self.patterns_)
                if n_pats == 0:
                    # original_plus_* modes are allowed to continue with an
                    # empty pattern block; the downstream matrix will contain
                    # the original feature block and any enabled augmented block.
                    # Do not call native build/get_coo paths with an empty
                    # pattern list because native code rejects that as
                    # "patterns list is empty — nothing to build".
                    self.x_train_hup_ = csr_matrix((n_train, 0), dtype=np.float32)
                else:
                    # Build train matrix from fused native CSR when available.  This
                    # avoids copying COO rows/cols into Python and lets scipy consume the
                    # compact CSR structure directly.  get_coo remains as a compatibility
                    # fallback for older native wheels.
                    if hasattr(_l1_result, "get_csr"):
                        indptr, indices = _l1_result.get_csr(n_train, n_pats)
                        data = np.ones(len(indices), dtype=np.float32)
                        self.x_train_hup_ = csr_matrix(
                            (data, indices, indptr), shape=(n_train, n_pats), dtype=np.float32
                        )
                    else:
                        rows, cols = _l1_result.get_coo()
                        # The fused COO is ordered by pattern index matching raw_patterns_
                        # (both sorted by descending utility).  If patterns_ was reordered
                        # by dedup we would need to remap cols — but dedup is skipped here
                        # so the order is identical.
                        data = np.ones(len(rows), dtype=np.float32)
                        self.x_train_hup_ = csr_matrix(
                            (data, (rows, cols)), shape=(n_train, n_pats), dtype=np.float32
                        )
                stage_times["build_matrix"] = t.ms

            else:
                # ── Original three-step path (L>1, or hotpath disabled) ───────
                # interaction_relaxed_mining identifies survivor pairs via a
                # joint/interaction-aware score upstream; here we only
                # translate those pairs into column indices.  The actual
                # joint-correlation computation and the override of the
                # per-column marginal-correlation admission gate both happen
                # natively inside prepare_transactions_cpp (see
                # native/transaction.cpp), not in Python.
                _eu_pair_left = None
                _eu_pair_right = None
                if bool(getattr(self, "interaction_relaxed_mining", False)):
                    _pairs = getattr(self, "_interaction_relaxed_adaptive_pairs_", None) or []
                    if _pairs:
                        _name_to_idx = {str(name): jx for jx, name in enumerate(list(col_names))}
                        _left_idx, _right_idx = [], []
                        for _pair in _pairs:
                            _left = str(_pair.get("left"))
                            _right = str(_pair.get("right"))
                            if _left in _name_to_idx and _right in _name_to_idx:
                                _left_idx.append(_name_to_idx[_left])
                                _right_idx.append(_name_to_idx[_right])
                        if _left_idx:
                            _eu_pair_left = np.asarray(_left_idx, dtype=np.int32)
                            _eu_pair_right = np.asarray(_right_idx, dtype=np.int32)

                _tx_args = [
                    X_num,
                    y_train,
                    2 if self.adaptive_binning else self.B,
                    col_names,
                    is_cat_np,
                    is_int_np,
                    X_cat_raw if any(v is not None for v in X_cat_raw) else None,
                    is_precoded_np,
                ]
                _tx_kwargs: dict[str, Any] = {}
                if _eu_pair_left is not None and _eu_pair_right is not None:
                    _tx_kwargs["eu_pair_left"] = _eu_pair_left
                    _tx_kwargs["eu_pair_right"] = _eu_pair_right
                _tx_cache = getattr(self, "_fast_tune_transaction_cache", None)
                _tx_cache_key = None
                if isinstance(_tx_cache, dict):
                    # Transaction preparation depends on the pre-binned X/y/schema
                    # and the relaxed-pair override arrays, but not on G or topK.
                    # The cache is created inside one fast_grid_tune call/fold and
                    # receives the same immutable pre-binned X object, so using the
                    # adaptive context id avoids hashing or copying n×p data.
                    _tx_cache_key = (
                        id(getattr(self, "_fast_tune_adaptive_context", None)),
                        tuple(X_num.shape),
                        str(X_num.dtype),
                        tuple(map(str, col_names)),
                        bytes(np.asarray(is_cat_np, dtype=np.uint8)),
                        bytes(np.asarray(is_int_np, dtype=np.uint8)),
                        bytes(np.asarray(is_precoded_np, dtype=np.uint8))
                        if is_precoded_np is not None
                        else b"",
                        tuple(np.asarray(_eu_pair_left, dtype=np.int32).tolist())
                        if _eu_pair_left is not None
                        else (),
                        tuple(np.asarray(_eu_pair_right, dtype=np.int32).tolist())
                        if _eu_pair_right is not None
                        else (),
                    )
                    _cached_td = _tx_cache.get(_tx_cache_key)
                else:
                    _cached_td = None
                if _cached_td is not None:
                    self.td_ = _cached_td
                    stage_times["prepare_transactions"] = 0.0
                    stage_times["prepare_transactions_cache_hit"] = 1.0
                else:
                    try:
                        self.td_ = _core.prepare_transactions(*_tx_args, **_tx_kwargs)
                    except TypeError:
                        # Retry without pair kwargs when the active native backend
                        # does not expose eu_pair inputs; the marginal-correlation
                        # path remains the compatible behavior.
                        if _tx_kwargs:
                            self.td_ = _core.prepare_transactions(*_tx_args)
                        else:
                            raise
                    if isinstance(_tx_cache, dict) and _tx_cache_key is not None:
                        _tx_cache[_tx_cache_key] = self.td_
                    stage_times["prepare_transactions"] = t.ms
                    stage_times["prepare_transactions_cache_hit"] = 0.0
                cpp_mem_bytes = self.td_.memory_usage_bytes()

                n_items = len(self.td_.item_twu)
                K = self._effective_topK(n_items)
                K_mine = self._effective_mining_topK(n_items)

                if self.verbose:
                    logger.info(
                        "  items=%d, K=%d, K_mine=%d, td_mem=%.1fMB",
                        n_items,
                        K,
                        K_mine,
                        cpp_mem_bytes / 1e6,
                    )

                t = self._timer()
                fit_deadline = self._mining_deadline_from_now()
                relaxed_cols: list[int] | None = None
                if bool(getattr(self, "interaction_relaxed_mining", False)):
                    survivors = list(
                        getattr(self, "interaction_relaxed_mining_survivors_", []) or []
                    )
                    name_to_idx = {str(name): j for j, name in enumerate(list(col_names))}
                    if survivors:
                        relaxed_cols = [
                            name_to_idx[str(row["name"])]
                            for row in survivors
                            if str(row.get("name")) in name_to_idx
                        ]
                    else:
                        survivors, relaxed_cols = (
                            self._select_interaction_relaxed_survivors_from_frame(
                                X_train,
                                y_train,
                                [str(name) for name in list(col_names)],
                                is_cat_np.astype(bool, copy=False),
                            )
                        )
                        self.interaction_relaxed_mining_survivors_ = [
                            dict(row) for row in survivors
                        ]
                raw_patterns = self._mine_with_fallback(
                    y_train, n_cls, K_mine, fit_deadline, relaxed_cols=relaxed_cols
                )
                if self.L == 1 and _CORE_AVAILABLE and not bool(relaxed_cols):
                    stage_times["l1_mining_dispatch_hotpath"] = 1.0
                self.raw_patterns_ = sorted(
                    raw_patterns, key=lambda pe: (-pe.utility, tuple(pe.items))
                )
                selected_patterns = self._select_patterns_for_budget(self.raw_patterns_, n_items)
                self.patterns_, _cached_coo = self._deduplicate_patterns_by_coverage(
                    selected_patterns, len(y_train)
                )
                stage_times["mine_patterns"] = t.ms

                if len(self.patterns_) == 0:
                    # Zero mined HUG patterns is not fatal.  In hybrid modes the
                    # downstream estimator can still use the original feature
                    # block.  In patterns_only mode, a common no-fail fallback
                    # below returns a constant-prior classifier.
                    n_train = len(y_train)
                    self.x_train_hup_ = csr_matrix((n_train, 0), dtype=np.float32)
                    stage_times["build_matrix"] = 0.0

                degraded = hasattr(self, "_degraded_reason")
                if degraded and self.verbose:
                    logger.warning("  DEGRADED: %s", self._degraded_reason)

                if self.verbose:
                    logger.info(
                        "  %d patterns in %.0f ms",
                        len(self.patterns_),
                        stage_times["mine_patterns"],
                    )

                t = self._timer()
                n_train = len(y_train)
                n_pats = len(self.patterns_)
                if n_pats == 0:
                    # original_plus_* modes are allowed to continue with an
                    # empty pattern block.  Avoid calling native
                    # build_train_matrix with an empty pattern list.
                    self.x_train_hup_ = csr_matrix((n_train, 0), dtype=np.float32)
                else:
                    if _cached_coo is not None:
                        rows, cols = _cached_coo
                        data = np.ones(len(rows), dtype=np.float32)
                        self.x_train_hup_ = csr_matrix(
                            (data, (rows, cols)), shape=(n_train, n_pats), dtype=np.float32
                        )
                    elif hasattr(_core, "build_train_matrix_csr"):
                        indptr, indices = _core.build_train_matrix_csr(self.td_, self.patterns_)
                        data = np.ones(len(indices), dtype=np.float32)
                        self.x_train_hup_ = csr_matrix(
                            (data, indices, indptr), shape=(n_train, n_pats), dtype=np.float32
                        )
                    else:
                        rows, cols = _core.build_train_matrix(self.td_, self.patterns_)
                        data = np.ones(len(rows), dtype=np.float32)
                        self.x_train_hup_ = csr_matrix(
                            (data, (rows, cols)), shape=(n_train, n_pats), dtype=np.float32
                        )
                stage_times["build_matrix"] = t.ms

            # Optional internal cache-only path used by fast_grid_tune().
            # At this point adaptive metadata, transaction data, mined patterns,
            # and the training HUG matrix are available.  Skipping downstream
            # fitting, rich feature metadata, and drift-baseline construction is
            # correctness-preserving for tuning because each evaluated candidate
            # rebuilds its own downstream matrix/model from these cached mining
            # artefacts.
            if bool(getattr(self, "_fast_tune_cache_only", False)):
                self.td_ = _TransactionDataWrapper(self.td_, self)
                self._native_available_ = True
                self._fast_tune_stage_times_ = dict(stage_times)
                return self

            if len(getattr(self, "patterns_", [])) == 0 and self.feature_mode == "patterns_only":
                # No mined patterns in a patterns-only model is a legitimate
                # degenerate outcome, especially on null/tiny/strictly-filtered
                # data.  Keep fit() total by installing a constant-prior
                # majority fallback instead of raising HUGIMLMiningError.
                t = self._timer()
                self._setup_constant_prior_fallback(
                    y_train,
                    reason=(
                        "No HUG patterns found. Try reducing G, increasing topK, "
                        "or adjusting B / L."
                    ),
                )
                stage_times["fit_downstream"] = t.ms

                # Stage 6: wrap C++ td_ for Python compatibility.  The fallback
                # prediction path does not need td_, but retaining it preserves
                # audit/serialization state and keeps transform() schema-stable.
                t = self._timer()
                self.td_ = _TransactionDataWrapper(self.td_, self)
                self._native_available_ = True
                stage_times["compat"] = t.ms

                t = self._timer()
                if self._is_production_mode():
                    self.__dict__.pop("_drift_det", None)
                    stage_times["drift_baseline"] = 0.0
                else:
                    self._drift_det = DriftDetector()
                    self._drift_det.fit_baseline(
                        X_num,
                        cat_mask,
                        getattr(self, "feature_names_in_", None)
                        or [f"col{j}" for j in range(X_num.shape[1])],
                        y=y_train,
                    )
                    stage_times["drift_baseline"] = t.ms
            else:
                # Stage 5: fit downstream classifier
                t = self._timer()
                self._setup_feature_mode_metadata()
                self._setup_augmented_pair_transforms(
                    X_train_original_for_downstream, y_train, fit=True
                )
                self._current_y_for_downstream_topk_ = y_train
                try:
                    self.x_train_downstream_ = self._make_downstream_features(
                        X_train_original_for_downstream, self.x_train_hup_, fit=True
                    )
                finally:
                    if hasattr(self, "_current_y_for_downstream_topk_"):
                        delattr(self, "_current_y_for_downstream_topk_")
                self.x_train_downstream_ = self._apply_strict_topk_budget_fit(
                    self.x_train_downstream_, y_train
                )
                self._cache_downstream_feature_metadata()
                self.model_ = Pipeline([("clf", self._make_estimator(n_cls))])
                _downstream_names_for_wiring = self._get_downstream_feature_names()
                if len(_downstream_names_for_wiring) != self.x_train_downstream_.shape[1]:
                    raise RuntimeError(
                        f"Internal error: downstream feature name count "
                        f"({len(_downstream_names_for_wiring)}) does not match the "
                        f"downstream matrix width ({self.x_train_downstream_.shape[1]}). "
                        f"Refusing to wire HUGIML feature metadata into the downstream "
                        f"estimator with mismatched names -- every later column would be "
                        f"misattributed."
                    )
                _wire_hugiml_feature_metadata(
                    self.model_.named_steps["clf"],
                    _downstream_names_for_wiring,
                    self.get_augmented_pair_transforms(),
                    self.get_pattern_provenance(),
                    self.get_original_feature_standardization(),
                )
                self.model_.fit(self.x_train_downstream_, y_train)
                stage_times["fit_downstream"] = t.ms

                # Stage 6: wrap C++ td_ for Python compatibility
                t = self._timer()
                self.td_ = _TransactionDataWrapper(self.td_, self)
                self._native_available_ = True
                stage_times["compat"] = t.ms

                t = self._timer()
                if self._is_production_mode():
                    self.__dict__.pop("_drift_det", None)
                    stage_times["drift_baseline"] = 0.0
                else:
                    self._drift_det = DriftDetector()
                    self._drift_det.fit_baseline(
                        X_num,
                        cat_mask,
                        getattr(self, "feature_names_in_", None)
                        or [f"col{j}" for j in range(X_num.shape[1])],
                        y=y_train,
                    )
                    stage_times["drift_baseline"] = t.ms

        rss_delta_mb = (_get_peak_rss_kb() - rss_before) / 1024
        n_compound = sum(1 for pe in self.patterns_ if len(pe.items) > 1)
        n_pats_final = len(self.patterns_)
        x_hup_for_metadata = getattr(self, "x_train_hup_", None)
        if x_hup_for_metadata is not None:
            n_train_final = int(x_hup_for_metadata.shape[0])
            nnz = int(x_hup_for_metadata.nnz)
        else:
            shape = tuple(
                getattr(self, "_training_pattern_matrix_shape_", (len(y_train), n_pats_final))
            )
            n_train_final = int(shape[0]) if shape else int(len(y_train))
            nnz = int(getattr(self, "_training_pattern_matrix_nnz_", 0))
        density = (
            nnz / (n_train_final * n_pats_final) if (n_train_final * n_pats_final) > 0 else 0.0
        )
        downstream_names_for_metadata = list(
            getattr(self, "_downstream_feature_names_", []) or self._get_downstream_feature_names()
        )
        downstream_feature_counts = {
            "original": sum(
                1 for name in downstream_names_for_metadata if str(name).startswith("orig:")
            ),
            "pattern": sum(
                1 for name in downstream_names_for_metadata if str(name).startswith("pattern:")
            ),
            "augmented_pair": sum(
                1
                for name in downstream_names_for_metadata
                if str(name).startswith("augmented_pair:")
            ),
        }
        downstream_feature_counts["total"] = len(downstream_names_for_metadata)

        self.fit_metadata_ = FitMetadata(
            n_samples=n_train_final,
            n_features=X_num.shape[1],
            n_classes=n_cls,
            n_items=len(getattr(self.td_, "item_twu", [])),
            n_patterns=n_pats_final,
            n_compound=n_compound,
            topK_used=self._effective_topK(len(getattr(self.td_, "item_twu", [])) or None),
            n_augmented_pairs=downstream_feature_counts.get("augmented_pair", 0),
            n_downstream_features=downstream_feature_counts.get("total", 0),
            downstream_feature_counts=downstream_feature_counts,
            stage_times_ms=stage_times,
            total_fit_ms=t_total.ms,
            matrix_density=density,
            config=dict(
                B=self.B,
                L=self.L,
                G=self.G,
                topK=self.topK,
                adaptive_binning=self.adaptive_binning,
                feature_mode=self.feature_mode,
                execution_mode=self.execution_mode,
                max_fit_seconds=self.max_fit_seconds,
                max_mining_seconds=getattr(self, "max_mining_seconds", None),
                effective_mining_timeout_seconds=self._effective_mining_timeout_seconds(),
                mining_audit_log_entries=len(getattr(self, "mining_audit_log_", []) or []),
            ),
            memory_peak_mb=round(mem.traced_peak_mb, 1),
            memory_rss_mb=round(rss_delta_mb, 1),
            memory_cpp_mb=round(cpp_mem_bytes / 1e6, 2),
            openmp_threads=actual_threads,
            degraded=hasattr(self, "_degraded_reason"),
        )

        self._apply_execution_mode_retention()

        if self.verbose:
            logger.info("  fit complete: %s", self.fit_metadata_.summary())

        return self

    def _effective_mining_timeout_seconds(self) -> float | None:
        """Return the configured mining-stage timeout in seconds.

        ``max_mining_seconds`` is the explicit new API. ``max_fit_seconds`` is
        kept as a backward-compatible alias because earlier releases already
        routed it to the native miner rather than to the whole fit pipeline.
        """
        timeout = getattr(self, "max_mining_seconds", None)
        if timeout is None:
            timeout = getattr(self, "max_fit_seconds", None)
        if timeout is None:
            return None
        return float(timeout)

    def _mining_deadline_from_now(self) -> float | None:
        timeout = self._effective_mining_timeout_seconds()
        return time.perf_counter() + timeout if timeout else None

    def get_mining_audit_log(self) -> pd.DataFrame:
        """Return the serialized mining audit log as a DataFrame.

        The log is populated during ``fit()`` and intentionally contains only
        scalar/list metadata: requested bounded depth, K/G, relaxed-mining
        status, per-attempt timeout budget, elapsed time, status, and number of
        patterns returned. It is safe to pickle/joblib with the estimator.
        """
        return pd.DataFrame(list(getattr(self, "mining_audit_log_", []) or []))

    def _mine_with_fallback(
        self,
        y_train: np.ndarray,
        n_cls: int,
        K: int,
        deadline: float | None,
        relaxed_cols: list[int] | None = None,
    ) -> list:
        """Mine patterns with graceful degradation on OOM or timeout.

        The ``deadline`` is forwarded into the C++ mining engine as a
        wall-clock ``timeout_s`` budget so the native layer can abort
        mid-run rather than only being checked between attempts.

        ``relaxed_cols``, when not None/empty, routes every attempt through
        ``mine_patterns_relaxed`` instead of ``mine_patterns`` (interaction_
        relaxed_mining). The minimal final fallback attempt drops G to 0.0
        for ordinary mining already; relaxation does not change that
        behavior, and the result remains bounded by the requested K.

        Every attempt appends a JSON/pickle-safe row to ``mining_audit_log_``.
        Native timeout returns are graceful: the C++ layer returns whatever
        patterns were mined before the deadline, and the audit row marks that
        the deadline was reached.
        """
        use_relaxed = bool(relaxed_cols)
        mine_fn = _core.mine_patterns_relaxed if use_relaxed else _core.mine_patterns
        timeout_budget_s = self._effective_mining_timeout_seconds()
        self.mining_audit_log_ = []
        self.mining_audit_config_ = {
            "requested_L": int(self.L) if isinstance(self.L, int) else self.L,
            "requested_K": int(K),
            "requested_G": float(self.G),
            "timeout_budget_s": timeout_budget_s,
            "uses_max_mining_seconds": getattr(self, "max_mining_seconds", None) is not None,
            "legacy_max_fit_seconds": getattr(self, "max_fit_seconds", None),
            "interaction_relaxed_mining": bool(use_relaxed),
            "relaxed_cols_count": len(relaxed_cols or []),
            "execution_mode": getattr(self, "execution_mode", "audit"),
        }

        def _call(K_arg, L_arg, G_arg, timeout_arg):
            if use_relaxed:
                return mine_fn(
                    self.td_, y_train, n_cls, K_arg, L_arg, G_arg, relaxed_cols, timeout_arg
                )
            return mine_fn(self.td_, y_train, n_cls, K_arg, L_arg, G_arg, timeout_arg)

        def _audit_entry(
            label: str, K_arg: int, L_arg: int, G_arg: float, timeout_arg: float
        ) -> dict:
            return {
                "attempt_index": len(getattr(self, "mining_audit_log_", []) or []) + 1,
                "label": str(label),
                "K": int(K_arg),
                "L": int(L_arg) if isinstance(L_arg, int) else L_arg,
                "G": float(G_arg),
                "timeout_s": float(timeout_arg or 0.0),
                "deadline_enabled": bool(deadline),
                "interaction_relaxed_mining": bool(use_relaxed),
                "relaxed_cols_count": len(relaxed_cols or []),
                "status": "started",
                "elapsed_ms": 0.0,
                "n_patterns_returned": 0,
                "deadline_reached_after_attempt": False,
                "exception_type": "",
                "exception_message": "",
            }

        attempts = [
            (K, self.L, self.G, "full"),
            (max(K // 2, 10), self.L, self.G, "K//2"),
            (max(K // 4, 10), 1, self.G, "K//4,L=1"),
            (50, 1, 0.0, "minimal"),
        ]
        for attempt_K, attempt_L, attempt_G, label in attempts:
            if deadline and time.perf_counter() > deadline:
                # Time budget exhausted — skip to minimal attempt immediately.
                minimal_K, minimal_L, minimal_G = 50, 1, 0.0
                preempt = _audit_entry(label, attempt_K, attempt_L, attempt_G, 0.0)
                preempt.update(
                    status="deadline_exhausted_before_attempt",
                    deadline_reached_after_attempt=True,
                )
                self.mining_audit_log_.append(preempt)
                self._degraded_reason = (
                    f"Time budget exceeded at '{label}'; "
                    f"falling back to minimal (K={minimal_K}, L={minimal_L})."
                )
                logger.warning("  fit timeout: %s", self._degraded_reason)
                # Give the minimal attempt a constant 5-second window; it is
                # cheap and must not run indefinitely on degenerate data.
                entry = _audit_entry("minimal_after_timeout", minimal_K, minimal_L, minimal_G, 5.0)
                t0 = time.perf_counter()
                try:
                    patterns = list(_call(minimal_K, minimal_L, minimal_G, 5.0))
                    entry.update(
                        status="ok",
                        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                        n_patterns_returned=len(patterns),
                    )
                    self.mining_audit_log_.append(entry)
                    return patterns
                except Exception as exc:
                    entry.update(
                        status="failed",
                        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                        exception_type=type(exc).__name__,
                        exception_message=str(exc)[:500],
                    )
                    self.mining_audit_log_.append(entry)
                    raise HUGIMLTimeoutError(
                        f"fit() exceeded max_mining_seconds/max_fit_seconds and the minimal "
                        f"fallback also failed: {exc}"
                    ) from exc
            # Compute remaining budget and pass it to the C++ engine so it
            # can abort mid-run rather than running past the wall-clock limit.
            remaining_s = max(deadline - time.perf_counter(), 0.0) if deadline else 0.0
            entry = _audit_entry(label, attempt_K, attempt_L, attempt_G, remaining_s)
            t0 = time.perf_counter()
            try:
                patterns: list = list(_call(attempt_K, attempt_L, attempt_G, remaining_s))
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                deadline_reached = bool(deadline and time.perf_counter() >= deadline)
                entry.update(
                    status="ok_timeout_partial" if deadline_reached and remaining_s > 0.0 else "ok",
                    elapsed_ms=elapsed_ms,
                    n_patterns_returned=len(patterns),
                    deadline_reached_after_attempt=deadline_reached,
                )
                self.mining_audit_log_.append(entry)
                if deadline_reached and remaining_s > 0.0:
                    self._degraded_reason = (
                        f"Mining deadline reached during {label}; using "
                        f"{len(patterns)} partial pattern(s) returned by native miner."
                    )
                if label != "full" and len(patterns) > 0:
                    self._degraded_reason = (
                        f"Recovered with {label}: K={attempt_K}, L={attempt_L}, G={attempt_G}"
                    )
                return patterns
            except MemoryError as exc:
                entry.update(
                    status="memory_error",
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    exception_type=type(exc).__name__,
                    exception_message=str(exc)[:500],
                )
                self.mining_audit_log_.append(entry)
                logger.warning("MemoryError during mining (%s), retrying…", label)
                continue
            except Exception as e:
                msg = str(e)
                if "bad_alloc" in msg.lower() or "memory" in msg.lower():
                    entry.update(
                        status="cpp_memory_error",
                        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                        exception_type=type(e).__name__,
                        exception_message=msg[:500],
                    )
                    self.mining_audit_log_.append(entry)
                    logger.warning("C++ memory error during mining (%s), retrying…", label)
                    continue
                entry.update(
                    status="exception",
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    exception_type=type(e).__name__,
                    exception_message=msg[:500],
                )
                self.mining_audit_log_.append(entry)
                raise
        self.mining_audit_log_.append(
            {
                "attempt_index": len(getattr(self, "mining_audit_log_", []) or []) + 1,
                "label": "all_attempts_exhausted",
                "K": int(K),
                "L": int(self.L) if isinstance(self.L, int) else self.L,
                "G": float(self.G),
                "timeout_s": 0.0,
                "deadline_enabled": bool(deadline),
                "interaction_relaxed_mining": bool(use_relaxed),
                "relaxed_cols_count": len(relaxed_cols or []),
                "status": "no_patterns",
                "elapsed_ms": 0.0,
                "n_patterns_returned": 0,
                "deadline_reached_after_attempt": bool(
                    deadline and time.perf_counter() >= deadline
                ),
                "exception_type": "",
                "exception_message": "",
            }
        )
        return []

    def _setup_constant_prior_fallback(self, y_train: np.ndarray, reason: str) -> None:
        """Install a no-fail constant-prior classifier for zero-pattern fits.

        The fallback is intentionally simple and deterministic: probabilities
        are the empirical class priors from the training target, and hard
        predictions are the majority class.  It is used only after the normal
        HUG mining fallback attempts have completed and produced no patterns in
        ``feature_mode='patterns_only'``.
        """
        y_arr = np.asarray(y_train)
        classes = np.asarray(getattr(self, "classes_", np.unique(y_arr)))
        counts = np.asarray([np.sum(y_arr == cls) for cls in classes], dtype=np.float64)
        total = float(np.sum(counts))
        if total <= 0.0:
            prior = np.full(len(classes), 1.0 / max(len(classes), 1), dtype=np.float64)
        else:
            prior = counts / total

        majority_idx = int(np.argmax(counts if counts.size else prior)) if len(classes) else 0
        self.fallback_active_ = True
        self.fallback_strategy_ = "constant_prior"
        self.fallback_reason_ = str(reason)
        self.fallback_class_prior_ = np.asarray(prior, dtype=np.float64)
        self.fallback_majority_class_ = classes[majority_idx] if len(classes) else None
        self.fallback_n_samples_ = int(y_arr.shape[0])

        self.raw_patterns_ = []
        self.patterns_ = []
        self.x_train_hup_ = csr_matrix((len(y_arr), 0), dtype=np.float32)
        self.x_train_downstream_ = csr_matrix((len(y_arr), 0), dtype=np.float32)
        self._pattern_orders_ = np.zeros(0, dtype=int)
        self._interaction_pattern_mask_ = np.zeros(0, dtype=bool)
        self._downstream_feature_names_ = []
        self._downstream_feature_names_full_ = []
        self._strict_topk_selected_feature_names_ = []
        self._strict_topk_feature_mask_ = np.zeros(0, dtype=bool)
        self._strict_topk_feature_scores_ = np.zeros(0, dtype=np.float64)
        self._original_feature_names_downstream_ = []
        self._original_selected_feature_names_downstream_ = []
        self._original_feature_names_downstream_full_ = []
        self._original_feature_mask_downstream_ = None
        self._original_feature_scores_downstream_ = np.zeros(0, dtype=np.float64)
        self.augmented_pair_transforms_enabled_ = False
        self.augmented_pair_config_ = {"enabled": False, "reason": "constant_prior_fallback"}
        self.augmented_pair_transforms_ = []
        self.augmented_pair_selected_features_ = []
        self._augmented_pair_block_ = None
        self._degraded_reason = str(reason)

        # Keep a fitted sklearn estimator wrapped in a Pipeline for
        # compatibility with code that inspects model_.named_steps["clf"],
        # serialization (which dispatches on Pipeline vs bare estimator), and
        # feature_importances() (which expects named_steps).  The public
        # prediction path uses fallback_class_prior_ directly and never
        # reaches model_.
        from sklearn.dummy import DummyClassifier

        dummy_X = np.zeros((len(y_arr), 1), dtype=np.float64)
        _dummy = DummyClassifier(strategy="prior")
        _dummy.fit(dummy_X, y_arr)
        self.model_ = Pipeline([("clf", _dummy)])

    def _is_constant_prior_fallback_active(self) -> bool:
        return bool(
            getattr(self, "fallback_active_", False)
            and getattr(self, "fallback_strategy_", None) == "constant_prior"
        )

    def _n_rows_for_prediction(self, X: Any) -> int:
        if isinstance(X, pd.DataFrame):
            return int(len(X))
        arr = np.asarray(X)
        if arr.ndim == 1:
            return 1
        return int(arr.shape[0])

    def _constant_prior_predict_proba(self, X: Any) -> np.ndarray:
        self._validate_test_input(X)
        n = self._n_rows_for_prediction(X)
        prior = np.asarray(getattr(self, "fallback_class_prior_", []), dtype=np.float64)
        if prior.size != len(getattr(self, "classes_", [])) or prior.size == 0:
            n_cls = max(1, len(getattr(self, "classes_", [])))
            prior = np.full(n_cls, 1.0 / n_cls, dtype=np.float64)
        return np.tile(prior.reshape(1, -1), (n, 1))

    def _constant_prior_predict(self, X: Any) -> np.ndarray:
        self._validate_test_input(X)
        n = self._n_rows_for_prediction(X)
        majority = getattr(self, "fallback_majority_class_", None)
        if majority is None:
            classes = np.asarray(getattr(self, "classes_", []))
            majority = classes[0] if classes.size else 0
        return np.full(n, majority, dtype=np.asarray(getattr(self, "classes_", [majority])).dtype)

    def predict_proba(self, X_test: Any) -> np.ndarray:
        """Predict class probabilities for X_test.

        When ``max_predict_ms`` is set large batches are processed in chunks.
        Rows exceeding the time budget receive uniform probabilities and a
        warning is emitted.

        Parameters
        ----------
        X_test : array-like or DataFrame

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        check_is_fitted(self)
        if self._is_constant_prior_fallback_active():
            t0 = time.perf_counter()
            proba = self._constant_prior_predict_proba(X_test)
            _mon = getattr(self, "monitor", None)
            if _mon is not None:
                _mon.record(proba, (time.perf_counter() - t0) * 1000)
            return proba
        # Keep the same representation used to fit the downstream original
        # feature block: raw user input before adaptive/constant-B pre-binning.
        # _build_test_hup applies _handle_test_nan() internally for the HUG
        # pattern matrix only; original_plus_* downstream columns are fitted
        # from raw X_train_original_for_downstream and therefore must transform
        # the raw test input as well.
        X_test_original_for_downstream = X_test
        # ── Adaptive pre-binning ─────────────────────────────────────────
        if getattr(self, "adaptive_binning", False) and getattr(self, "_bin_edges_", None):
            X_test = self._prebin_for_predict(X_test)
        # ─────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()

        budget_ms = self.max_predict_ms
        if budget_ms is None or not isinstance(X_test, (pd.DataFrame, np.ndarray)):
            Z_test = self._build_test_hup(X_test)
            X_downstream = self._make_downstream_features(
                X_test_original_for_downstream, Z_test, fit=False
            )
            X_downstream = self._apply_strict_topk_budget_transform(X_downstream)
            proba = np.asarray(self.model_.predict_proba(X_downstream))
            _mon = getattr(self, "monitor", None)
            if _mon is not None:
                _mon.record(proba, (time.perf_counter() - t0) * 1000)
            return proba

        n = len(X_test)
        n_cls = len(self.classes_)
        chunk_size = max(100, n // 10)
        result = np.full((n, n_cls), 1.0 / n_cls, dtype=np.float64)
        completed = 0
        is_df = isinstance(X_test, pd.DataFrame)

        for start in range(0, n, chunk_size):
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms > budget_ms:
                warnings.warn(
                    f"Prediction SLA exceeded ({elapsed_ms:.0f}ms > {budget_ms}ms) "
                    f"after {completed}/{n} rows.  Remaining rows filled with uniform.",
                    HUGIMLWarning,
                    stacklevel=2,
                )
                break
            end = min(start + chunk_size, n)
            chunk = X_test.iloc[start:end] if is_df else X_test[start:end]  # type: ignore[union-attr]
            orig_chunk = (
                X_test_original_for_downstream.iloc[start:end]
                if isinstance(X_test_original_for_downstream, pd.DataFrame)
                else X_test_original_for_downstream[start:end]
            )
            Z_chunk = self._build_test_hup(chunk)
            X_downstream_chunk = self._make_downstream_features(orig_chunk, Z_chunk, fit=False)
            X_downstream_chunk = self._apply_strict_topk_budget_transform(X_downstream_chunk)
            result[start:end] = self.model_.predict_proba(X_downstream_chunk)
            completed = end

        _mon = getattr(self, "monitor", None)
        if _mon is not None:
            _mon.record(result[:completed], (time.perf_counter() - t0) * 1000)
        return result

    def predict(self, X_test: Any) -> np.ndarray:
        """Predict class labels for X_test.

        Parameters
        ----------
        X_test : array-like or DataFrame

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        check_is_fitted(self)
        if self._is_constant_prior_fallback_active():
            return self._constant_prior_predict(X_test)
        # Keep the same representation used to fit the downstream original
        # feature block: raw user input before adaptive/constant-B pre-binning.
        # _build_test_hup applies _handle_test_nan() internally for the HUG
        # pattern matrix only; original_plus_* downstream columns are fitted
        # from raw X_train_original_for_downstream and therefore must transform
        # the raw test input as well.
        X_test_original_for_downstream = X_test
        # ── Adaptive pre-binning ─────────────────────────────────────────
        if getattr(self, "adaptive_binning", False) and getattr(self, "_bin_edges_", None):
            X_test = self._prebin_for_predict(X_test)
        # ─────────────────────────────────────────────────────────────────
        Z_test = self._build_test_hup(X_test)
        X_downstream = self._make_downstream_features(
            X_test_original_for_downstream, Z_test, fit=False
        )
        X_downstream = self._apply_strict_topk_budget_transform(X_downstream)
        return np.asarray(self.model_.predict(X_downstream))

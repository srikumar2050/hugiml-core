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

"""HUGIMLClassifierNative — C++ accelerated, scikit-learn compatible classifier.

Implements the High Utility Gain Interpretable Machine Learning (HUG-IML)
algorithm from:

    Krishnamoorthy, S. (2024). Interpretable Classifier Models for Decision
    Support Using High Utility Gain Patterns. IEEE Access, 12, 126088–126107.
    DOI: 10.1109/ACCESS.2024.3455563

Computationally intensive stages (discretisation, transaction construction,
pattern mining, matrix assembly) run at native speed via a compiled C++
extension with optional OpenMP parallelism.  The Python layer handles
DataFrame ingestion, column-type detection, downstream estimation,
explanation methods, monitoring, and drift detection.

Architecture
------------
C++ extension (_hugiml_core):
    Discretisation, transaction construction, top-K HUI pattern mining with
    information-gain filtering, bitmap-accelerated matrix assembly, OpenMP
    parallel pattern matching.

Python layer:
    Column-type detection (prepareXy), NaN/Inf imputation, downstream sklearn
    estimator (LogisticRegression default), explanation methods
    (get_hug_features, get_pattern_info, feature_importances), versioned
    model serialisation, prediction monitoring, multi-method drift detection,
    latency SLA enforcement, and graceful degradation under memory pressure.

Quick start
-----------
Two usage paths are supported:

**Path A — prepareXy** (recommended when the full dataset is available upfront)::

    from hugiml import HUGIMLClassifierNative

    clf = HUGIMLClassifierNative()
    X, y = clf.prepareXy(X_df, y_series)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)

    print(clf.model_summary())
    print(clf.feature_importances())

**Path B — allCols + origColumns** (cross-validation loops)::

    clf = HUGIMLClassifierNative(
        allCols=[int_cols, float_cols, cat_cols],
        origColumns=X_df.columns.tolist(),
    )
    clf.fit(X_train, y_train)

Monitoring and drift detection::

    clf.enable_monitoring()
    clf.predict_proba(X_new)
    print(clf.monitor.report())

    drift = clf.detect_drift(X_new)
    print(drift)

Versioned serialisation::

    clf.save_model("model.hugiml")
    clf2 = HUGIMLClassifierNative.load_model("model.hugiml")
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
import os
import threading
import time
import tracemalloc
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from hugiml._binning import (
    _apply_edges as _adap_apply_edges,
)
from hugiml._binning import (
    _quantile_edges as _adap_quantile_edges,
)
from hugiml._binning import (
    _select_b as _adap_select_b,
)
from hugiml._compat import check_array, check_X_y

__all__ = [
    "HUGIMLClassifierNative",
    "FitMetadata",
]
from hugiml.exceptions import (
    HUGIMLConvergenceWarning,
    HUGIMLDtypeDriftWarning,
    HUGIMLMiningError,
    HUGIMLParamError,
    HUGIMLPredictionError,
    HUGIMLRangeWarning,
    HUGIMLSchemaError,
    HUGIMLTimeoutError,
    HUGIMLValidationError,
    HUGIMLVersionError,
    HUGIMLWarning,
)
from hugiml.monitoring import DriftDetector, PredictionMonitor
from hugiml.serialization import MIN_SCHEMA_VERSION, MODEL_SCHEMA_VERSION
from hugiml.serialization import load_model as _load_model
from hugiml.serialization import save_model as _save_model

try:
    import _hugiml_core as _core

    _CORE_AVAILABLE: bool = True
except ImportError:
    _core = None
    _CORE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helpers: RSS memory (Unix) with Windows fallback
# ---------------------------------------------------------------------------
try:
    import resource as _resource

    def _get_peak_rss_kb() -> int:
        return int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)

except ImportError:

    def _get_peak_rss_kb() -> int:
        try:
            import psutil

            return int(psutil.Process().memory_info().peak_wset) // 1024
        except ImportError:
            return 0


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration presets
# =============================================================================

_PRESETS: dict[str, dict] = {
    "quick": dict(B=5, L=1, G=1e-2, topK=50),
    "balanced": dict(B=7, L=1, G=5e-3, topK=-1),
    "thorough": dict(B=-1, L=2, G=1e-4, topK=-1),
}


# =============================================================================
# Fit metadata
# =============================================================================


@dataclasses.dataclass(frozen=True)
class FitMetadata:
    """Immutable record of everything that happened during fit().

    Attributes
    ----------
    n_samples, n_features : int
        Training set dimensions.
    n_classes : int
        Number of distinct target classes.
    n_items : int
        Number of utility-annotated items (bins + categories).
    n_patterns : int
        Number of HUG patterns mined and retained.
    n_compound : int
        Compound patterns (length > 1).
    topK_used : int
        Effective topK budget used during mining.
    stage_times_ms : dict[str, float]
        Wall-clock milliseconds per fit stage.
    total_fit_ms : float
        Total fit wall-clock milliseconds.
    matrix_density : float
        Fraction of non-zero entries in the training pattern matrix.
    config : dict
        Snapshot of (B, L, G, topK) as used.
    memory_peak_mb : float
        Python-traced peak memory during fit.
    memory_rss_mb : float
        RSS delta during fit (Unix only).
    memory_cpp_mb : float
        Estimated C++ extension memory usage.
    openmp_threads : int
        Number of OpenMP threads used.
    degraded : bool
        True when fit fell back to reduced parameters.
    """

    n_samples: int
    n_features: int
    n_classes: int
    n_items: int
    n_patterns: int
    n_compound: int
    topK_used: int
    stage_times_ms: dict
    total_fit_ms: float
    matrix_density: float
    config: dict
    memory_peak_mb: float = 0.0
    memory_rss_mb: float = 0.0
    memory_cpp_mb: float = 0.0
    openmp_threads: int = 1
    degraded: bool = False

    def summary(self) -> str:
        """Return a single-line human-readable summary of the fit outcome."""
        return (
            f"{self.n_patterns} patterns "
            f"({self.n_compound} compound) from "
            f"{self.n_samples}×{self.n_features} in "
            f"{self.total_fit_ms:.0f} ms  "
            f"[density={self.matrix_density:.4f}]"
        )


# =============================================================================
# Memory profiling context manager
# =============================================================================

# tracemalloc is a process-global resource.  Concurrent fits on separate
# instances would race on is_tracing / start / stop without this lock.
_tracemalloc_lock = threading.Lock()


class _MemoryTracker:
    """Track peak memory during a code block via tracemalloc + RSS.

    Thread-safe: a module-level lock ensures that only one fit() at a time
    owns the tracemalloc session.  Other concurrent fits skip tracing and
    report traced_peak_mb = 0.0, which is clearly distinguished from a
    real measurement rather than a corrupted one.
    """

    def __enter__(self) -> _MemoryTracker:
        self._rss_before = _get_peak_rss_kb()
        self._lock_acquired = _tracemalloc_lock.acquire(blocking=False)
        self._snap_before: tracemalloc.Snapshot | None = None
        if self._lock_acquired:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                self._started = True
            else:
                self._started = False
            self._snap_before = tracemalloc.take_snapshot()
        else:
            self._started = False
            self._snap_before = None
        return self

    def __exit__(self, *exc: object) -> None:
        if self._lock_acquired:
            try:
                if self._snap_before is not None:
                    snap_after = tracemalloc.take_snapshot()
                    stats = snap_after.compare_to(self._snap_before, "lineno")
                    self.traced_peak_mb = sum(s.size for s in stats if s.size > 0) / 1e6
                else:
                    self.traced_peak_mb = 0.0
            finally:
                if self._started:
                    tracemalloc.stop()
                _tracemalloc_lock.release()
        else:
            self.traced_peak_mb = 0.0
        self.rss_mb = (_get_peak_rss_kb() - self._rss_before) / 1024

    @staticmethod
    def estimate_fit_mb(n: int, p: int, n_items: int, K: int) -> float:
        """Rough peak-memory estimate in MB for a fit() call."""
        disc_mb = n * p * 4 / 1e6
        trans_mb = n * p * 16 / 1e6
        ul_mb = n_items * n * 24 / 1e6
        matrix_mb = n * min(K, n_items) * 4 / 1e6
        overhead = 50
        return disc_mb + trans_mb + ul_mb + matrix_mb + overhead


# =============================================================================
# Transaction data wrapper (C++ ↔ Python bridge)
# =============================================================================


class _TransactionDataWrapper:
    """Augments native TransactionDataCpp with Python-compatible attributes.

    Stores exact C++ state (prefixed _cpp_) so that deserialized models can
    still run predict() via the pure-Python fallback transform.
    """

    def __init__(self, td_native: Any, classifier: HUGIMLClassifierNative) -> None:
        self._td = td_native
        self._clf = classifier

        self._cpp_bn2id = dict(td_native.bn2id)
        self._cpp_bkey_stride = int(td_native.bkey_stride)
        self._cpp_col_min = np.array(td_native.col_min, dtype=np.float64)
        self._cpp_col_range = np.array(td_native.col_range, dtype=np.float64)
        self._cpp_all_edges = [np.array(e, dtype=np.float64) for e in td_native.all_edges]
        self._cpp_nb_col = list(td_native.nb_col)
        self._cpp_is_cat = list(td_native.is_cat_v)
        self._cpp_is_int = list(td_native.is_int_v)
        self._cpp_cat_categories = [list(c) for c in td_native.cat_categories]

        self.bn2id = self._build_compat_bn2id()
        self.all_edges = self._cpp_all_edges
        self.col_range = self._cpp_col_range
        self.col_min = self._cpp_col_min
        self.is_cat = classifier.cat_cols_mask_
        self.is_int = classifier.is_int_mask_

    def __getattr__(self, name: str) -> Any:
        return getattr(self._td, name)

    def __getstate__(self) -> dict:
        state = {k: v for k, v in self.__dict__.items() if k not in ("_td", "_clf")}
        if self._td is not None:
            state["item_map"] = dict(self._td.item_map)
            state["item_twu"] = list(self._td.item_twu)
            state["nb_col"] = list(self._td.nb_col)
        return state

    def __setstate__(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)
        self._td = None

    def _build_compat_bn2id(self) -> dict:
        bn2id: dict = {}
        item_map = self._td.item_map
        feature_items: dict[str, list] = {}
        for item_id, label in item_map.items():
            if "=" in label:
                feat_name = label.split("=")[0]
                feature_items.setdefault(feat_name, []).append(item_id)
        feature_names = getattr(self._clf, "feature_names_in_", None) or self._clf.origColumns
        if feature_names is None:
            return bn2id
        stride = self._cpp_bkey_stride
        for col_idx, feat_name in enumerate(feature_names):
            if feat_name in feature_items:
                for bin_idx, item_id in enumerate(sorted(feature_items[feat_name])):
                    bn2id[(col_idx * stride) + bin_idx] = item_id
        return bn2id


# =============================================================================
# HUGIMLClassifierNative
# =============================================================================


# =============================================================================
# ── v1.1.0  Per-feature adaptive binning — module-level helpers ──────────────
#
# Imported from hugiml._binning — the single source of truth for all
# adaptive-binning maths.  Local aliases preserve every existing call-site
# inside this file without modification.
#
# =============================================================================


class HUGIMLClassifierNative(TransformerMixin, ClassifierMixin, BaseEstimator):
    """HUG-IML interpretable classifier — C++ accelerated, scikit-learn compatible.

    Extracts High Utility Gain (HUG) patterns from labelled tabular data,
    transforms the input into a binary pattern-presence matrix, and fits an
    interpretable downstream classifier.  The mined patterns are human-readable
    and serve as the primary source of model explanations.

    Parameters
    ----------
    allCols : list of 3 lists, optional
        ``[int_col_names, float_col_names, cat_col_names]``.
        Must be paired with ``origColumns``.
    origColumns : list of str, optional
        Ordered column names matching the columns of X passed to fit/predict.
    B : int, default 8
        Number of quantile bins per numerical feature.
        Use -1 for supervised auto-selection (maximises IG over [2, 20]).
    L : int, default 2
        Maximum HUG pattern length.  1 = singletons; 2 = pairs; -1 = unlimited.
    G : float, default 1e-4
        Minimum information-gain threshold.
    topK : int, default 200
        Maximum number of patterns to retain.  -1 computes automatically.
    base_estimator : sklearn estimator, optional
        Downstream classifier trained on the binary pattern matrix.
        Defaults to LogisticRegression.
    n_jobs : int, default 1
        Number of OpenMP threads.  -1 uses all available cores.
    max_predict_ms : float or None
        Prediction latency budget in milliseconds.
    max_fit_seconds : float or None
        Wall-clock budget for the pattern-mining stage of fit().  Transaction
        preparation and downstream model fitting (e.g. LogisticRegression) are
        not bounded — total fit() time may exceed this value.  When the budget
        is exhausted mid-mine, graceful degradation produces a smaller pattern
        set; if even the minimal fallback cannot finish in time,
        ``HUGIMLTimeoutError`` is raised.
    verbose : bool, default False
        Emit INFO-level log messages during fit.

    Attributes (available after fit)
    ----------------------------------
    classes_           : ndarray — unique class labels.
    n_features_in_     : int — number of input features.
    feature_names_in_  : list or None — column names from training data.
    cat_cols_mask_     : ndarray[bool] — True for categorical columns.
    is_int_mask_       : ndarray[bool] — True for integer columns.
    td_                : _TransactionDataWrapper — discretisation artefacts.
    patterns_          : list — mined HUG patterns.
    x_train_hup_       : csr_matrix — binary training pattern matrix.
    model_             : Pipeline — fitted downstream estimator.
    fit_metadata_      : FitMetadata — timings, memory, pattern stats.
    monitor            : PredictionMonitor or None — prediction statistics.
    """

    _fit_lock: threading.RLock  # per-instance, created in __init__
    monitor: PredictionMonitor | None  # set by enable_monitoring() / disable_monitoring()
    feature_names_in_: list[str] | None  # set by prepareXy / _resolve_col_meta after fit

    def __init__(
        self,
        allCols: list | None = None,
        origColumns: list | None = None,
        B: int = 8,
        L: int = 2,
        G: float = 1e-4,
        topK: int = 200,
        base_estimator: Any = None,
        n_jobs: int = 1,
        max_predict_ms: float | None = None,
        max_fit_seconds: float | None = None,
        verbose: bool = False,
        # ── v1.1.0 adaptive binning ───────────────────────────────────────
        # When adaptive_binning=True each numerical feature is pre-discretised
        # to B_j quantile bins chosen by elbow-stopping IG search.  The
        # pre-binned columns are declared categorical before the C++ layer
        # (global B is overridden to sentinel 2).  Bin edges are stored in
        # _bin_edges_ and reapplied identically at predict/transform time.
        # ─────────────────────────────────────────────────────────────────
        adaptive_binning: bool = False,
        b_candidates: list | None = None,
        min_marginal_gain_ratio: float = 0.02,
        feature_mode: str = "patterns_only",
    ) -> None:
        self.allCols = allCols
        self.origColumns = origColumns
        self.B = B
        self.L = L
        self.G = G
        self.topK = topK
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.max_predict_ms = max_predict_ms
        self.max_fit_seconds = max_fit_seconds
        self.verbose = verbose
        self.adaptive_binning = adaptive_binning
        self.b_candidates = b_candidates
        self.min_marginal_gain_ratio = min_marginal_gain_ratio
        self.feature_mode = feature_mode
        self._fit_lock = threading.RLock()

    # ── Class methods ─────────────────────────────────────────────────────────

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> HUGIMLClassifierNative:
        """Create a classifier from a named configuration preset.

        Parameters
        ----------
        name : {'quick', 'balanced', 'thorough'}
            quick     — B=5, L=1, G=1e-2, topK=50
            balanced  — B=7, L=1, G=5e-3, topK=-1
            thorough  — B=-1, L=2, G=1e-4, topK=-1

        Returns
        -------
        HUGIMLClassifierNative
        """
        if name not in _PRESETS:
            raise HUGIMLParamError(f"Unknown preset '{name}'.  Available: {list(_PRESETS)}")
        params = {**_PRESETS[name], **overrides}
        return cls(**params)

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        fitted = hasattr(self, "patterns_")
        status = f", {len(self.patterns_)} patterns" if fitted else ", not fitted"
        adap = ", adaptive" if self.adaptive_binning else ""
        mode = f", feature_mode={self.feature_mode}"
        return f"HUGIMLClassifierNative(B={self.B}, L={self.L}, G={self.G}{adap}{mode}{status})"

    # ── sklearn protocol ──────────────────────────────────────────────────────

    def get_params(self, deep: bool = True) -> dict:
        """Return constructor parameters as a dict (sklearn protocol)."""
        return dict(
            allCols=self.allCols,
            origColumns=self.origColumns,
            B=self.B,
            L=self.L,
            G=self.G,
            topK=self.topK,
            base_estimator=(copy.deepcopy(self.base_estimator) if deep else self.base_estimator),
            n_jobs=self.n_jobs,
            max_predict_ms=self.max_predict_ms,
            max_fit_seconds=self.max_fit_seconds,
            verbose=self.verbose,
            adaptive_binning=self.adaptive_binning,
            b_candidates=self.b_candidates,
            min_marginal_gain_ratio=self.min_marginal_gain_ratio,
            feature_mode=self.feature_mode,
        )

    def set_params(self, **params: Any) -> HUGIMLClassifierNative:
        """Set constructor parameters in-place and return self (sklearn protocol)."""
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _more_tags(self) -> dict:
        return {
            "requires_y": True,
            "binary_only": False,
            "poor_score": False,
            "X_types": ["2darray", "dataframe"],
            "allow_nan": False,
        }

    def __sklearn_tags__(self) -> Any:
        """Declare sklearn 1.6+ Tags, including TransformerTags for transform().

        ``__sklearn_tags__`` was introduced in sklearn 1.6.  Base classes on
        older installations do not implement it, so ``super().__sklearn_tags__()``
        raises ``AttributeError``.  Guard that call and return ``None`` when the
        parent chain does not support the protocol — callers must handle ``None``.
        """
        try:
            tags = super().__sklearn_tags__()
        except AttributeError:
            logger.debug(
                "super().__sklearn_tags__() raised AttributeError; "
                "sklearn base classes do not implement the tag protocol "
                "(expected sklearn >= 1.6).",
            )
            return None
        try:
            from sklearn.utils._tags import TransformerTags

            tags.transformer_tags = TransformerTags()
        except ImportError:
            logger.debug(
                "sklearn.utils._tags.TransformerTags not available; "
                "TransformerTags will not be declared.",
                exc_info=True,
            )
        return tags

    # ── Pickle protocol ───────────────────────────────────────────────────────

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_fit_lock", None)
        # Strip instance-level method patches set by instrument_classifier():
        # these are closures and are not picklable.
        state.pop("predict_proba", None)
        state.pop("predict", None)
        state["_schema_version_"] = MODEL_SCHEMA_VERSION
        if "patterns_" in state and state["patterns_"]:
            state["patterns_"] = [
                {"utility": pe.utility, "items": list(pe.items), "ig": pe.ig}
                for pe in state["patterns_"]
            ]
            state["_patterns_pickled_"] = True
        return state

    def __setstate__(self, state: dict) -> None:
        schema_ver = state.pop("_schema_version_", 1)
        if schema_ver < MIN_SCHEMA_VERSION:
            raise HUGIMLVersionError(
                f"Model schema version {schema_ver} is too old.  "
                f"Minimum supported: {MIN_SCHEMA_VERSION}.  Re-fit the model."
            )

        if state.pop("_patterns_pickled_", False):

            class _PE:
                __slots__ = ("utility", "items", "ig")

                def __init__(self, d: dict) -> None:
                    self.utility = d["utility"]
                    self.items = d["items"]
                    self.ig = d["ig"]

            state["patterns_"] = [_PE(d) for d in state["patterns_"]]

        self.__dict__.update(state)
        self._fit_lock = threading.RLock()

        # ── v1.1.0 backward compatibility ─────────────────────────────────
        # Models saved with v1.0.0 have no adaptive_binning in their pickle
        # state.  Initialise all adaptive attrs to their off-state defaults
        # so the model behaves identically to a v1.0.0 model after restore.
        if not hasattr(self, "adaptive_binning"):
            self.adaptive_binning = False
            self.b_candidates = None
            self.min_marginal_gain_ratio = 0.02
        # v1.1.0 missing value handling — absent in models saved before this version
        if not hasattr(self, "_missing_col_edges_"):
            self._missing_col_edges_ = {}
        # ──────────────────────────────────────────────────────────────────

        if hasattr(self, "td_") and self.td_ is not None:
            td = self.td_
            self._native_available_ = not (hasattr(td, "_td") and td._td is None)
        else:
            self._native_available_ = False

    # ── Versioned save / load ─────────────────────────────────────────────────

    def save_model(self, path: str | os.PathLike) -> None:
        """Persist the fitted model to a binary file with schema versioning.

        Parameters
        ----------
        path : str or Path

        Raises
        ------
        HUGIMLSerializationError
        """
        _save_model(self, path)

    @classmethod
    def load_model(cls, path: str | os.PathLike) -> HUGIMLClassifierNative:
        """Load a model previously saved with :meth:`save_model`.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        HUGIMLClassifierNative

        Raises
        ------
        HUGIMLVersionError, HUGIMLSerializationError
        """
        return _load_model(path, expected_type=cls)  # type: ignore[no-any-return]

    # ── Data preparation ──────────────────────────────────────────────────────

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

        catCols = [
            c
            for idx, c in enumerate(X.columns)
            if pd.api.types.is_object_dtype(X.iloc[:, idx])
            or pd.api.types.is_string_dtype(X.iloc[:, idx])
            or isinstance(X.iloc[:, idx].dtype, pd.CategoricalDtype)
        ]
        intCols = [
            c for idx, c in enumerate(X.columns) if pd.api.types.is_integer_dtype(X.iloc[:, idx])
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

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_float_array(arr: Any, cat_mask: np.ndarray | None = None) -> tuple:
        """Split input into a float64 numeric array and raw categorical arrays.

        Adversarial-input hardening:
        - Forces writable copies of read-only column views.
        - Non-finite cells (NaN/Inf) in numerical columns are pre-converted
          to np.nan string-label bins by _prebin_nan_cols (fit) or
          _handle_test_nan (predict), so they arrive here as categorical.
          No median imputation is performed (removed in v1.1.0).
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

        X_num = np.zeros((n, p), dtype=np.float64)
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
                col = np.array(raw, dtype=np.float64, copy=True)
                # v1.1.0: non-finite cells (NaN/Inf) are pre-handled by
                # _prebin_nan_cols (fit) and _handle_test_nan (predict)
                # before reaching here.  No median imputation.
                X_num[:, j] = col

        return X_num, X_cat_raw

    def _effective_topK(self) -> int:
        if self.topK != -1:
            return self.topK
        nitems = 100
        lsize: dict[int, int] = {i: math.comb(nitems, i) for i in range(1, 7)}
        if self.L in (-1, 1):
            return lsize[1]
        elif isinstance(self.L, int) and 2 <= self.L <= 6:
            return lsize[self.L]
        else:
            return lsize[2]

    def _make_estimator(self, n_cls: int) -> Any:
        if self.base_estimator is not None:
            return copy.deepcopy(self.base_estimator)
        solver = "liblinear" if n_cls == 2 else "lbfgs"
        return LogisticRegression(solver=solver, random_state=0, max_iter=500)

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
        if self.allCols is not None or self.origColumns is not None:
            if self.allCols is None or self.origColumns is None:
                raise HUGIMLParamError("allCols and origColumns must both be supplied together.")
            if not (isinstance(self.allCols, list) and len(self.allCols) == 3):
                raise HUGIMLParamError("allCols must be [int_cols, float_cols, cat_cols].")
        # ── v1.1.0 adaptive binning params ────────────────────────────────
        if not isinstance(self.adaptive_binning, bool):
            raise HUGIMLParamError("adaptive_binning must be bool.")
        if self.b_candidates is not None:
            if (
                not isinstance(self.b_candidates, list)
                or len(self.b_candidates) == 0
                or not all(isinstance(b, int) and b >= 2 for b in self.b_candidates)
            ):
                raise HUGIMLParamError(
                    "b_candidates must be a non-empty list of int >= 2."
                )
        if not isinstance(self.min_marginal_gain_ratio, (float, int)):
            raise HUGIMLParamError("min_marginal_gain_ratio must be numeric.")
        if not 0 < float(self.min_marginal_gain_ratio) < 1:
            raise HUGIMLParamError(
                f"min_marginal_gain_ratio must be in (0, 1), "
                f"got {self.min_marginal_gain_ratio}."
            )
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
            self.feature_names_in_ = col_list
            return self.cat_cols_mask_

        if isinstance(X_train, pd.DataFrame):
            col_list = X_train.columns.astype(str).tolist()
            self.cat_cols_mask_ = np.array(
                [
                    pd.api.types.is_object_dtype(X_train[c])
                    or pd.api.types.is_string_dtype(X_train[c])
                    or isinstance(X_train[c].dtype, pd.CategoricalDtype)
                    for c in X_train.columns
                ],
                dtype=bool,
            )
            self.is_int_mask_ = np.array(
                [pd.api.types.is_integer_dtype(X_train[c]) for c in X_train.columns],
                dtype=bool,
            )
            self.feature_names_in_ = col_list
            return self.cat_cols_mask_

        arr = np.asarray(X_train)
        if arr.ndim < 2:
            raise ValueError(
                f"HUGIMLClassifierNative expects a 2D array, got array of shape {arr.shape}."
            )
        p = arr.shape[1]
        self.cat_cols_mask_ = np.zeros(p, dtype=bool)
        self.is_int_mask_ = np.zeros(p, dtype=bool)
        if not hasattr(self, "feature_names_in_"):
            self.feature_names_in_ = None
        return self.cat_cols_mask_

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

    # ── Core fit ──────────────────────────────────────────────────────────────

    # ── v1.1.0  Adaptive binning methods ─────────────────────────────────────

    def _apply_adaptive_binning(
        self, X_train: Any, y_arr: np.ndarray
    ) -> Any:
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
        X_df = X_train if is_df else pd.DataFrame(X_train)
        candidates = sorted(set(self.b_candidates or [2, 3, 5, 7, 10, 15]))
        ratio = self.min_marginal_gain_ratio
        cat_mask = self.cat_cols_mask_
        col_names = list(X_df.columns)

        self._bin_edges_: dict = {}
        self.per_feature_b_: dict = {}
        self.ig_scores_: dict = {}
        pre_binned: set = set()

        for j, name in enumerate(col_names):
            if j < len(cat_mask) and cat_mask[j]:
                continue  # leave existing categorical columns untouched
            col = pd.to_numeric(X_df.iloc[:, j], errors="coerce").values
            finite = np.isfinite(col)
            if finite.sum() < 10:
                chosen = candidates[len(candidates) // 2]
                self.ig_scores_[name] = {b: 0.0 for b in candidates}
            else:
                chosen, scores = _adap_select_b(
                    col[finite], y_arr[finite], candidates, ratio
                )
                self.ig_scores_[name] = scores
            edges = _adap_quantile_edges(col, chosen)
            self.per_feature_b_[name] = len(edges) - 1
            self._bin_edges_[name] = edges
            pre_binned.add(name)

        # Update masks so Stage 1 treats pre-binned features as categorical
        new_cat = cat_mask.copy()
        new_int = getattr(
            self, "is_int_mask_", np.zeros(len(col_names), dtype=bool)
        ).copy()
        for j, name in enumerate(col_names):
            if name in pre_binned:
                new_cat[j] = True
                new_int[j] = False
        self.cat_cols_mask_ = new_cat
        self.is_int_mask_ = new_int

        X_pre = X_df.copy()
        for name, edges in self._bin_edges_.items():
            if name in X_df.columns:
                col = pd.to_numeric(X_df[name], errors="coerce").values
                X_pre[name] = _adap_apply_edges(col, edges)
        return X_pre if is_df else X_pre

    def _prebin_for_predict(self, X: Any) -> Any:
        """Apply stored bin edges to X before the C++ inference path.

        Called at the top of predict_proba / predict / transform when
        adaptive_binning=True.  Accepts raw (un-binned) input in any form
        that _build_test_hup accepts, and returns a DataFrame with string
        bin labels for the features that were pre-binned during fit().
        """
        bin_edges = getattr(self, "_bin_edges_", {})
        if not bin_edges:
            return X
        is_df = isinstance(X, pd.DataFrame)
        feat_names = getattr(self, "feature_names_in_", None)
        if is_df:
            X_df = X
        else:
            arr = np.asarray(X)
            cols = feat_names or [f"col{j}" for j in range(arr.shape[1])]
            X_df = pd.DataFrame(arr, columns=cols)
        X_out = X_df.copy()
        for name, edges in bin_edges.items():
            if name in X_df.columns:
                col = pd.to_numeric(X_df[name], errors="coerce").values
                X_out[name] = _adap_apply_edges(col, edges)
        return X_out if is_df else X_out

    # ── v1.1.0  Missing value handling methods ──────────────────────────────────
    #
    # NaN (and Inf) in a numerical feature is treated as "not observed" —
    # no item is generated in the transaction for that (row, feature) pair.
    # This matches the categorical path, where np.nan in X_cat_raw is already
    # silently skipped by the C++ transaction builder.
    #
    # Numerical columns that contain non-finite values in training are
    # pre-binned to string labels (same mechanism as adaptive_binning) so the
    # C++ sees them as categorical.  Non-finite cells become np.nan in the
    # label array → C++ skips → no item.  Edges are stored in
    # _missing_col_edges_ and reused at predict time.
    #
    # At predict time, columns that were non-finite-free in training but
    # receive non-finite test values are handled dynamically using the C++
    # edge arrays stored in td_._cpp_all_edges[j].
    #
    # The old median imputation in _to_float_array is removed entirely.
    # ────────────────────────────────────────────────────────────────────────────

    def _prebin_nan_cols(self, X_train: Any) -> Any:
        """Pre-bin ALL numerical columns to string quantile labels.

        Called in _fit_impl (non-adaptive path) after _resolve_col_meta.

        Every numerical column is discretised into B equal-frequency bins
        using the finite training values.  Non-finite cells (NaN, Inf) become
        ``np.nan`` in the label array; the C++ transaction builder skips those
        cells, generating no item for that (row, feature) pair.

        By pre-binning all numerical columns unconditionally, NaN at *any*
        point — training or test time, whatever columns — is handled correctly
        and identically: the item is simply absent from the transaction.

        Stores edges in ``self._missing_col_edges_[feature_name]``.
        Updates ``self.cat_cols_mask_`` to mark all pre-binned columns as
        categorical so the C++ routes them through the string-label path.
        """
        is_df = isinstance(X_train, pd.DataFrame)
        X_df = X_train if is_df else pd.DataFrame(X_train)
        cat_mask = self.cat_cols_mask_
        col_names = list(X_df.columns)
        n_cols = len(col_names)

        self._missing_col_edges_: dict = {}
        new_cat = cat_mask.copy()
        new_int = getattr(
            self, "is_int_mask_", np.zeros(n_cols, dtype=bool)
        ).copy()
        modified = False

        for j, name in enumerate(col_names):
            if j >= len(cat_mask) or cat_mask[j]:
                continue  # already categorical — C++ handles np.nan natively

            col = pd.to_numeric(X_df.iloc[:, j], errors="coerce").values
            finite = col[np.isfinite(col)]
            edges = _adap_quantile_edges(finite, self.B) if finite.size > 0                 else np.array([0.0, 1.0])
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
        return X_pre if is_df else X_pre

    def _handle_test_nan(self, X_test: Any) -> tuple:
        """Apply training-time bin edges to all pre-binned columns at test time.

        All numerical columns are pre-binned during _fit_impl (non-adaptive path),
        so ``_missing_col_edges_`` contains edges for every numerical column.

        At test time the input still carries raw float values.  This method
        converts every pre-binned column to its string bin labels using the
        stored training edges.  Non-finite values (NaN, Inf) become ``np.nan``
        → C++ skips → no item for that (row, feature) pair.

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
            cols = feat_names or [f"col{j}" for j in range(arr.shape[1])]
            X_df = pd.DataFrame(arr, columns=cols)

        col_names = list(X_df.columns)
        missing_edges = getattr(self, "_missing_col_edges_", {})
        local_cat = base_cat.copy()
        modified = False

        for j, name in enumerate(col_names):
            if j >= len(base_cat):
                continue

            # Column was pre-binned at training time: convert ALL raw float
            # values to string bin labels so C++ item lookups match.
            # base_cat[j] is already True (set by _prebin_nan_cols); we
            # still need to apply edges because test data arrives as raw floats.
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
        return (X_df if is_df else X_df), local_cat

    # ── End v1.1.0 missing value handling methods ─────────────────────────────

    # ── End v1.1.0 adaptive binning methods ──────────────────────────────────

    def _require_core(self) -> None:
        """Raise a clear ImportError if the native extension is absent."""
        if not _CORE_AVAILABLE:
            raise ImportError(
                "HUGIMLClassifierNative requires the compiled C++ extension "
                "'_hugiml_core'.\n"
                "Build it with:  pip install . --no-build-isolation\n"
                "Or for development:  HUGIML_FAST_BUILD=1 python setup.py "
                "build_ext --inplace"
            )

    def fit(self, X: Any, y: Any) -> HUGIMLClassifierNative:
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
            return self._fit_impl(X, y)

    def _fit_impl(self, X_train: Any, y_train: Any) -> HUGIMLClassifierNative:
        # Clear all fitted state so that re-fitting the same instance is
        # idempotent.  Without this, _resolve_col_meta() short-circuits on
        # the cached cat_cols_mask_ from the previous fit, causing the column
        # type masks to carry over and producing non-reproducible results.
        for _attr in (
            "cat_cols_mask_",
            "is_int_mask_",
            "feature_names_in_",
            "_bin_edges_",
            "_missing_col_edges_",
            "per_feature_b_",
            "ig_scores_",
            "patterns_",
            "model_",
            "classes_",
            "x_train_hup_",
            "fit_metadata_",
            "_original_scaler_",
            "_original_feature_names_downstream_",
            "_pattern_orders_",
            "_interaction_pattern_mask_",
            "x_train_downstream_",
        ):
            self.__dict__.pop(_attr, None)

        t_total = self._timer()
        stage_times: dict[str, float] = {}

        # Reject sparse matrices with an informative message
        from scipy.sparse import issparse as _issparse

        if _issparse(X_train):
            raise ValueError(
                "HUGIMLClassifierNative does not support sparse input.  "
                "Convert to a dense array via X.toarray() first."
            )

        # Reject complex-valued arrays
        if hasattr(X_train, "dtype") and np.iscomplexobj(X_train):
            raise ValueError("Complex data not supported by HUGIMLClassifierNative.")

        self._validate_params()
        X_train_original_for_downstream = self._copy_input_for_downstream(X_train)

        # ── v1.1.0 adaptive pre-binning ───────────────────────────────────
        # Must run before Stage 1 (_resolve_col_meta) so that the updated
        # cat_cols_mask_ (pre-binned features → categorical) is already in
        # place when Stage 1 short-circuits on the existing mask.
        if self.adaptive_binning:
            self._resolve_col_meta(X_train)   # prime cat_cols_mask_ first
            _y_for_ig = np.asarray(y_train, dtype=np.int64)
            X_train = self._apply_adaptive_binning(X_train, _y_for_ig)
            if self.verbose:
                logger.info(
                    "  adaptive binning: %d features pre-binned, B_j in [%d, %d]",
                    len(self._bin_edges_),
                    min(self.per_feature_b_.values(), default=0),
                    max(self.per_feature_b_.values(), default=0),
                )
        # ─────────────────────────────────────────────────────────────────

        # ── v1.1.0 missing value pre-binning (non-adaptive path) ──────────
        # Adaptive path handles non-finite values in _apply_adaptive_binning
        # (via _adap_apply_edges which now maps non-finite → np.nan).
        # Non-adaptive path: pre-bin NaN-containing columns here so the C++
        # receives them as categorical with np.nan → no item generated.
        if not self.adaptive_binning:
            self._resolve_col_meta(X_train)   # ensure cat_cols_mask_ exists
            X_train = self._prebin_nan_cols(X_train)
        # ─────────────────────────────────────────────────────────────────

        n_threads = _core.openmp_get_max_threads() if self.n_jobs == -1 else self.n_jobs
        if n_threads > 0:
            _core.openmp_set_num_threads(n_threads)
        actual_threads = _core.openmp_get_max_threads()

        mem = _MemoryTracker()
        with mem:
            # Stage 1: resolve column metadata
            t = self._timer()
            cat_mask = self._resolve_col_meta(X_train)
            int_mask = getattr(self, "is_int_mask_", None)

            X_num, X_cat_raw = self._to_float_array(X_train, cat_mask)
            y_train = np.asarray(y_train, dtype=np.int64)
            X_num, y_train = check_X_y(X_num, y_train, dtype=None)

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
                    "HUGIMLClassifierNative.fit — %dx%d, %d classes",
                    X_num.shape[0],
                    X_num.shape[1],
                    n_cls,
                )

            # Stage 2: prepare transactions (C++)
            t = self._timer()
            rss_before = _get_peak_rss_kb()
            col_names = getattr(self, "feature_names_in_", None)
            is_cat_np = cat_mask.astype(np.uint8)
            is_int_np = (
                int_mask if int_mask is not None else np.zeros(X_num.shape[1], dtype=bool)
            ).astype(np.uint8)

            self.td_ = _core.prepare_transactions(
                X_num,
                y_train,
                # v1.1.0: when adaptive_binning=True all numerical features
                # are already pre-binned as categorical; B=2 is the sentinel
                # value that satisfies the C++ >=2 constraint while having
                # no effect on already-categorical columns.
                2 if self.adaptive_binning else self.B,
                col_names,
                is_cat_np,
                is_int_np,
                X_cat_raw if any(v is not None for v in X_cat_raw) else None,
            )
            stage_times["prepare_transactions"] = t.ms
            cpp_mem_bytes = self.td_.memory_usage_bytes()

            K = self._effective_topK()
            n_items = len(self.td_.item_twu)

            if self.verbose:
                logger.info("  items=%d, K=%d, td_mem=%.1fMB", n_items, K, cpp_mem_bytes / 1e6)

            # Stage 3: mine patterns (C++) with graceful degradation
            t = self._timer()
            fit_deadline = (
                time.perf_counter() + self.max_fit_seconds if self.max_fit_seconds else None
            )
            raw_patterns = self._mine_with_fallback(y_train, n_cls, K, fit_deadline)
            self.patterns_ = sorted(raw_patterns, key=lambda pe: (pe.utility, pe.items))
            stage_times["mine_patterns"] = t.ms

            if len(self.patterns_) == 0:
                raise HUGIMLMiningError(
                    "No HUG patterns found.  Try reducing G, increasing topK, or adjusting B / L."
                )

            degraded = hasattr(self, "_degraded_reason")
            if degraded and self.verbose:
                logger.warning("  DEGRADED: %s", self._degraded_reason)

            if self.verbose:
                logger.info(
                    "  %d patterns in %.0f ms",
                    len(self.patterns_),
                    stage_times["mine_patterns"],
                )

            # Stage 4: build binary pattern matrix (C++)
            t = self._timer()
            n_train = len(y_train)
            n_pats = len(self.patterns_)
            rows, cols = _core.build_train_matrix(self.td_, self.patterns_)
            data = np.ones(len(rows), dtype=np.float32)
            self.x_train_hup_ = csr_matrix(
                (data, (rows, cols)), shape=(n_train, n_pats), dtype=np.float32
            )
            stage_times["build_matrix"] = t.ms

            # Stage 5: fit downstream classifier
            t = self._timer()
            self._setup_feature_mode_metadata()
            self.x_train_downstream_ = self._make_downstream_features(
                X_train_original_for_downstream, self.x_train_hup_, fit=True
            )
            self.model_ = Pipeline([("clf", self._make_estimator(n_cls))])
            self.model_.fit(self.x_train_downstream_, y_train)
            stage_times["fit_downstream"] = t.ms

            # Stage 6: wrap C++ td_ for Python compatibility
            t = self._timer()
            self.td_ = _TransactionDataWrapper(self.td_, self)
            self._native_available_ = True
            stage_times["compat"] = t.ms

            # Drift baseline
            self._drift_det = DriftDetector()
            self._drift_det.fit_baseline(
                X_num,
                cat_mask,
                getattr(self, "feature_names_in_", None)
                or [f"col{j}" for j in range(X_num.shape[1])],
                y=y_train,
            )

        rss_delta_mb = (_get_peak_rss_kb() - rss_before) / 1024

        n_compound = sum(1 for pe in self.patterns_ if len(pe.items) > 1)
        nnz = self.x_train_hup_.nnz
        density = nnz / (n_train * n_pats) if (n_train * n_pats) > 0 else 0.0

        self.fit_metadata_ = FitMetadata(
            n_samples=n_train,
            n_features=X_num.shape[1],
            n_classes=n_cls,
            n_items=n_items,
            n_patterns=n_pats,
            n_compound=n_compound,
            topK_used=K,
            stage_times_ms=stage_times,
            total_fit_ms=t_total.ms,
            matrix_density=density,
            config=dict(
                B=self.B, L=self.L, G=self.G, topK=self.topK,
                adaptive_binning=self.adaptive_binning,
                feature_mode=self.feature_mode,
            ),
            memory_peak_mb=round(mem.traced_peak_mb, 1),
            memory_rss_mb=round(rss_delta_mb, 1),
            memory_cpp_mb=round(cpp_mem_bytes / 1e6, 2),
            openmp_threads=actual_threads,
            degraded=degraded,
        )

        if self.verbose:
            logger.info("  fit complete: %s", self.fit_metadata_.summary())

        return self

    def _mine_with_fallback(
        self, y_train: np.ndarray, n_cls: int, K: int, deadline: float | None
    ) -> list:
        """Mine patterns with graceful degradation on OOM or timeout.

        The ``deadline`` is forwarded into the C++ mining engine as a
        wall-clock ``timeout_s`` budget so the native layer can abort
        mid-run rather than only being checked between attempts.
        """
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
                self._degraded_reason = (
                    f"Time budget exceeded at '{label}'; "
                    f"falling back to minimal (K={minimal_K}, L={minimal_L})."
                )
                logger.warning("  fit timeout: %s", self._degraded_reason)
                # Give the minimal attempt a fixed 5-second window; it is
                # cheap and must not run indefinitely on degenerate data.
                try:
                    return list(
                        _core.mine_patterns(
                            self.td_,
                            y_train,
                            n_cls,
                            minimal_K,
                            minimal_L,
                            minimal_G,
                            5.0,
                        )
                    )
                except Exception as exc:
                    raise HUGIMLTimeoutError(
                        f"fit() exceeded max_fit_seconds and the minimal fallback "
                        f"also failed: {exc}"
                    ) from exc
            # Compute remaining budget and pass it to the C++ engine so it
            # can abort mid-run rather than running past the wall-clock limit.
            remaining_s = max(deadline - time.perf_counter(), 0.0) if deadline else 0.0
            try:
                patterns: list = list(
                    _core.mine_patterns(
                        self.td_,
                        y_train,
                        n_cls,
                        attempt_K,
                        attempt_L,
                        attempt_G,
                        remaining_s,
                    )
                )
                if label != "full" and len(patterns) > 0:
                    self._degraded_reason = (
                        f"Recovered with {label}: K={attempt_K}, L={attempt_L}, G={attempt_G}"
                    )
                return patterns
            except MemoryError:
                logger.warning("MemoryError during mining (%s), retrying…", label)
                continue
            except Exception as e:
                if "bad_alloc" in str(e).lower() or "memory" in str(e).lower():
                    logger.warning("C++ memory error during mining (%s), retrying…", label)
                    continue
                raise
        return []

    # ── Prediction ────────────────────────────────────────────────────────────

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
        X_test_original_for_downstream = self._copy_input_for_downstream(X_test)
        # ── v1.1.0 adaptive pre-binning ───────────────────────────────────
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
        X_test_original_for_downstream = self._copy_input_for_downstream(X_test)
        # ── v1.1.0 adaptive pre-binning ───────────────────────────────────
        if getattr(self, "adaptive_binning", False) and getattr(self, "_bin_edges_", None):
            X_test = self._prebin_for_predict(X_test)
        # ─────────────────────────────────────────────────────────────────
        Z_test = self._build_test_hup(X_test)
        X_downstream = self._make_downstream_features(
            X_test_original_for_downstream, Z_test, fit=False
        )
        return np.asarray(self.model_.predict(X_downstream))

    # ── Downstream feature modes ─────────────────────────────────────────────

    def _copy_input_for_downstream(self, X: Any) -> Any:
        """Preserve raw input before adaptive/pre-binning for hybrid modes."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return np.array(X, copy=True)

    def _pattern_order_from_label(self, label: str) -> int:
        """Infer pattern order from a human-readable HUG pattern label."""
        import re

        matches = re.findall(r"\bcol\d+\s*=", str(label))
        if matches:
            return len(matches)
        if "," in str(label):
            return len([part for part in str(label).split(",") if part.strip()])
        return 1

    def _setup_feature_mode_metadata(self) -> None:
        """Cache pattern-order masks used by hybrid feature modes."""
        features = self.get_hug_features()
        orders = np.asarray([self._pattern_order_from_label(f) for f in features], dtype=int)
        if len(orders) != self.x_train_hup_.shape[1]:
            orders = np.ones(self.x_train_hup_.shape[1], dtype=int)
        self._pattern_orders_ = orders
        self._interaction_pattern_mask_ = orders > 1

    def _prepare_original_features_for_downstream(self, X: Any, fit: bool = False):
        """Prepare original input features for hybrid downstream estimators.

        This intentionally does not affect transform(), get_hug_features(), or
        any pattern diagnostics.  It is used only by predict/fit when
        feature_mode includes original features.
        """
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

        # Numeric columns are scaled; non-numeric columns are one-hot encoded.
        numeric = X_df.apply(pd.to_numeric, errors="coerce")
        numeric_cols = [c for c in X_df.columns if not numeric[c].isna().all()]
        X_num = numeric[numeric_cols] if numeric_cols else pd.DataFrame(index=X_df.index)
        X_cat = X_df.drop(columns=numeric_cols, errors="ignore")

        if fit:
            self._original_numeric_cols_ = list(X_num.columns)
            self._original_cat_cols_ = list(X_cat.columns)
            self._original_numeric_medians_ = X_num.median(numeric_only=True).fillna(0.0)
            X_num_filled = X_num.fillna(self._original_numeric_medians_)
            self._original_scaler_ = StandardScaler()
            X_num_arr = self._original_scaler_.fit_transform(X_num_filled) if len(self._original_numeric_cols_) else np.empty((len(X_df), 0))
            X_cat_dum = pd.get_dummies(X_cat.astype("string"), dummy_na=True) if len(self._original_cat_cols_) else pd.DataFrame(index=X_df.index)
            self._original_dummy_columns_ = list(X_cat_dum.columns)
        else:
            num_cols = getattr(self, "_original_numeric_cols_", [])
            med = getattr(self, "_original_numeric_medians_", pd.Series(dtype=float))
            X_num = numeric.reindex(columns=num_cols)
            X_num_filled = X_num.fillna(med).fillna(0.0)
            if len(num_cols):
                X_num_arr = self._original_scaler_.transform(X_num_filled)
            else:
                X_num_arr = np.empty((len(X_df), 0))
            cat_cols = getattr(self, "_original_cat_cols_", [])
            X_cat = X_df.reindex(columns=cat_cols)
            X_cat_dum = pd.get_dummies(X_cat.astype("string"), dummy_na=True) if len(cat_cols) else pd.DataFrame(index=X_df.index)
            dummy_cols = getattr(self, "_original_dummy_columns_", [])
            X_cat_dum = X_cat_dum.reindex(columns=dummy_cols, fill_value=0)

        X_cat_arr = X_cat_dum.to_numpy(dtype=np.float64, copy=False) if X_cat_dum.shape[1] else np.empty((len(X_df), 0))
        X_base = np.hstack([X_num_arr, X_cat_arr]) if X_cat_arr.shape[1] else X_num_arr
        if fit:
            self._original_feature_names_downstream_ = list(getattr(self, "_original_numeric_cols_", [])) + list(getattr(self, "_original_dummy_columns_", []))
        return csr_matrix(X_base.astype(np.float32, copy=False))

    def _make_downstream_features(self, X_original: Any, Z_patterns: csr_matrix, fit: bool = False):
        """Build the estimator input matrix for the configured feature_mode."""
        mode = getattr(self, "feature_mode", "patterns_only")
        if mode == "patterns_only":
            return Z_patterns

        X_base = self._prepare_original_features_for_downstream(X_original, fit=fit)
        Z = Z_patterns if issparse(Z_patterns) else csr_matrix(Z_patterns)

        if mode == "original_plus_patterns":
            return hstack([X_base, Z], format="csr")

        if mode == "original_plus_interactions":
            mask = getattr(self, "_interaction_pattern_mask_", None)
            if mask is None:
                self._setup_feature_mode_metadata()
                mask = self._interaction_pattern_mask_
            return hstack([X_base, Z[:, mask]], format="csr")

        raise HUGIMLParamError(f"Unknown feature_mode={mode!r}.")

    def _get_downstream_feature_names(self) -> list[str]:
        """Names aligned with coefficients of the downstream estimator."""
        mode = getattr(self, "feature_mode", "patterns_only")
        pattern_names = list(self.get_hug_features())
        if mode == "patterns_only":
            return pattern_names
        original_names = [f"orig:{name}" for name in getattr(self, "_original_feature_names_downstream_", [])]
        if mode == "original_plus_patterns":
            return original_names + [f"pattern:{name}" for name in pattern_names]
        if mode == "original_plus_interactions":
            mask = getattr(self, "_interaction_pattern_mask_", np.ones(len(pattern_names), dtype=bool))
            selected = [name for name, keep in zip(pattern_names, mask) if keep]
            return original_names + [f"pattern:{name}" for name in selected]
        return pattern_names

    def transform(self, X: Any) -> csr_matrix:
        """Return the binary HUG pattern matrix for X.

        Each column corresponds to one mined pattern.  Entry (i, j) is 1 when
        all items of pattern j appear in row i.

        Parameters
        ----------
        X : array-like or DataFrame

        Returns
        -------
        csr_matrix, shape (n_samples, n_patterns)
        """
        check_is_fitted(self)
        # ── v1.1.0 adaptive pre-binning ───────────────────────────────────
        if getattr(self, "adaptive_binning", False) and getattr(self, "_bin_edges_", None):
            X = self._prebin_for_predict(X)
        # ─────────────────────────────────────────────────────────────────
        return self._build_test_hup(X)

    def _build_test_hup(self, X_test: Any) -> csr_matrix:
        """Build the sparse binary pattern matrix for test data."""
        self._check_health()
        # ── v1.1.0 non-finite handling ────────────────────────────────────
        # Adaptive: X_test already pre-binned by _prebin_for_predict
        #           (called in predict_proba/predict/transform before here).
        # Non-adaptive: apply stored or reconstructed edges; non-finite
        #               cells become np.nan → C++ generates no item.
        if not getattr(self, "adaptive_binning", False):
            X_test, _cat_mask = self._handle_test_nan(X_test)
        else:
            _cat_mask = getattr(self, "cat_cols_mask_", None)
        # ─────────────────────────────────────────────────────────────────
        self._validate_test_input(X_test)
        X_num, X_cat_raw = self._to_float_array(X_test, _cat_mask)
        X_num = check_array(X_num, dtype=None)

        n = X_num.shape[0]
        n_pats = len(self.patterns_)

        if getattr(self, "_native_available_", True):
            try:
                rows, cols = _core.build_test_matrix(
                    X_num,
                    self.td_,
                    X_cat_raw if any(v is not None for v in X_cat_raw) else None,
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
                    code = lc.get(v)
                    if code is None:
                        continue
                    bi = code + 1
                else:
                    edges = cpp_all_edges[j]
                    if edges is None or len(edges) < 2:
                        continue
                    nb = cpp_nb_col[j]
                    if j < len(cpp_is_int) and cpp_is_int[j]:
                        val = X_num[r, j]
                    else:
                        val = (X_num[r, j] - cpp_col_min[j]) / cpp_col_range[j]
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
        if not hasattr(self, "patterns_") or len(self.patterns_) == 0:
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
                "HUGIMLClassifierNative does not support sparse input.  "
                "Convert to a dense array via X.toarray() first."
            )
        is_df = isinstance(X_test, pd.DataFrame)
        arr = None
        if not is_df:
            arr = np.asarray(X_test)
            if arr.ndim == 1:
                raise ValueError(
                    f"HUGIMLClassifierNative expects a 2D array, got 1D array of shape {arr.shape}."
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
        if is_df and cat_mask is not None:
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
                for j in range(min(n_test_features, len(cat_mask))):
                    if cat_mask[j]:
                        continue
                    edges = cpp_all_edges[j] if j < len(cpp_all_edges) else None
                    if edges is None or len(edges) < 2:
                        continue
                    train_min = float(edges[0])
                    train_max = float(edges[-1])
                    train_span = train_max - train_min
                    if train_span <= 0:
                        continue
                    col = pd.to_numeric(X_test.iloc[:, j], errors="coerce")
                    finite = col[np.isfinite(col)]
                    if finite.empty:
                        continue
                    test_min, test_max = float(finite.min()), float(finite.max())
                    if (
                        test_min < train_min - train_span * 0.5
                        or test_max > train_max + train_span * 0.5
                    ):
                        warnings.warn(
                            f"Column '{X_test.columns[j]}' has values "
                            f"[{test_min:.4g}, {test_max:.4g}] outside training "
                            f"range [{train_min:.4g}, {train_max:.4g}].",
                            HUGIMLRangeWarning,
                            stacklevel=4,
                        )

    # ── Monitoring and drift ──────────────────────────────────────────────────

    def enable_monitoring(self, window_size: int = 1000) -> HUGIMLClassifierNative:
        """Enable prediction monitoring.  Access via ``self.monitor``."""
        self.monitor = PredictionMonitor(window_size=window_size)
        return self

    def disable_monitoring(self) -> HUGIMLClassifierNative:
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
            return "Drift detection unavailable (no baseline stored)."
        cat_mask = getattr(self, "cat_cols_mask_", np.zeros(0, dtype=bool))
        X_num, _ = self._to_float_array(X_test, cat_mask)
        y_arr = np.asarray(y_test) if y_test is not None else None
        report = self._drift_det.detect(X_num, y_test=y_arr, threshold=threshold)
        return str(report)

    def get_drift_psi(self, X_test: Any) -> dict:
        """Return per-feature PSI values as a dict."""
        check_is_fitted(self)
        if getattr(self, "_drift_det", None) is None:
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

        for train_idx, test_idx in cv.split(X, y):
            if isinstance(X, pd.DataFrame):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            fold_clf = self.__class__(
                **{
                    k: v
                    for k, v in self.get_params().items()
                    if k not in ("allCols", "origColumns")
                }
            )
            t0 = time.perf_counter()
            fold_clf.fit(X_tr, y_tr)
            results["fit_times_ms"].append((time.perf_counter() - t0) * 1000)
            results["test_scores"].append(scorer(fold_clf, X_te, y_te))
            results["fold_metadata"].append(fold_clf.fit_metadata_)

            fold_clf.enable_monitoring()
            fold_clf.predict_proba(X_te)
            fold_mon = fold_clf.monitor
            results["fold_monitors"].append(fold_mon.stats if fold_mon is not None else {})

            if fold_clf._drift_det is not None:
                cat_mask = getattr(fold_clf, "cat_cols_mask_", np.zeros(0, dtype=bool))
                X_te_num, _ = fold_clf._to_float_array(X_te, cat_mask)
                results["fold_drift"].append(fold_clf._drift_det.compute_psi(X_te_num))
            else:
                results["fold_drift"].append({})

        return results

    # ── Explanation methods ───────────────────────────────────────────────────

    def get_hug_features(self) -> list[str]:
        """Return a human-readable label for each mined HUG pattern.

        Singleton patterns use the format ``feature=[lo,hi]`` for numerical
        columns (e.g. ``age=[35,50]``) and ``feature=value`` for categorical
        columns (e.g. ``gender=F``).  Compound patterns (L > 1) are
        comma-separated, e.g. ``age=[35,50], gender=F``.

        Returns
        -------
        list of str
        """
        check_is_fitted(self)
        item_map = self.td_.item_map
        return [", ".join(item_map.get(it, str(it)) for it in pe.items) for pe in self.patterns_]

    def get_transformed_shape(self) -> tuple[int, int]:
        """Return (n_samples, n_patterns) of the training pattern matrix."""
        check_is_fitted(self)
        shape = self.x_train_hup_.shape
        return int(shape[0]), int(shape[1])

    def get_pattern_info(self) -> pd.DataFrame:
        """Summary DataFrame with one row per mined HUG pattern.

        Columns: pattern, utility, information_gain, support.
        """
        check_is_fitted(self)
        n_train = self.x_train_hup_.shape[0]
        features = self.get_hug_features()
        records: list[dict[str, object]] = []
        for i, pe in enumerate(self.patterns_):
            support = float(self.x_train_hup_[:, i].sum()) / n_train
            records.append(
                {
                    "pattern": features[i],
                    "utility": round(pe.utility, 6),
                    "information_gain": round(pe.ig, 6),
                    "support": round(support, 4),
                }
            )
        return pd.DataFrame(records)

    def feature_importances(self) -> pd.DataFrame:
        """Map downstream LR coefficients back to HUG pattern labels.

        Returns a DataFrame sorted by absolute coefficient magnitude with
        columns: pattern, coefficient, abs_coefficient, support.

        Raises
        ------
        AttributeError
            When the downstream estimator does not expose ``coef_``
            (e.g. non-linear models).
        """
        check_is_fitted(self)
        clf_step = self.model_.named_steps.get("clf")
        if not hasattr(clf_step, "coef_"):
            raise AttributeError(
                "feature_importances requires the downstream estimator "
                "to expose coef_ (e.g. LogisticRegression)."
            )

        raw_coef = clf_step.coef_
        coef = (
            raw_coef.mean(axis=0)
            if raw_coef.ndim == 2 and raw_coef.shape[0] > 1
            else raw_coef.ravel()
        )
        features = self._get_downstream_feature_names()
        n_train = self.x_train_hup_.shape[0]
        mode = getattr(self, "feature_mode", "patterns_only")
        original_count = len(getattr(self, "_original_feature_names_downstream_", [])) if mode != "patterns_only" else 0

        rows: list[dict[str, object]] = []
        for i, (feat, c) in enumerate(zip(features, coef)):
            if i < original_count:
                support = 1.0
                ftype = "original"
            else:
                pat_idx = i - original_count
                if mode == "original_plus_interactions":
                    interaction_indices = np.where(getattr(self, "_interaction_pattern_mask_", []))[0]
                    pat_idx = int(interaction_indices[pat_idx]) if pat_idx < len(interaction_indices) else pat_idx
                support = (
                    float(self.x_train_hup_[:, pat_idx].sum()) / n_train
                    if 0 <= pat_idx < self.x_train_hup_.shape[1]
                    else 1.0
                )
                ftype = "pattern"
            rows.append(
                {
                    "pattern": feat,
                    "feature_type": ftype,
                    "coefficient": round(float(c), 6),
                    "abs_coefficient": round(abs(float(c)), 6),
                    "support": round(support, 4),
                }
            )
        result: pd.DataFrame = pd.DataFrame(rows)
        result = result.sort_values("abs_coefficient", ascending=False)
        return pd.DataFrame(result.reset_index(drop=True))

    # ── v1.1.0  Adaptive-binning diagnostic plots ─────────────────────────────
    # These methods are available on any fitted HUGIMLClassifierNative instance
    # when adaptive_binning=True.  HUGIMLAdaptive inherits them automatically
    # as a subclass.  Both require matplotlib (optional dependency).

    def plot_bin_profiles(self, figsize: tuple | None = None):
        """Bar chart of the chosen B per numerical feature (adaptive binning only).

        Colour encodes position in the candidate range:
        blue = coarse end, green = mid, amber/red = fine end.

        Returns
        -------
        (fig, ax)

        Raises
        ------
        RuntimeError
            When called on a non-adaptive or unfitted model.
        ImportError
            When matplotlib is not installed.
        """
        self._check_adaptive_fitted("plot_bin_profiles")
        self._require_mpl()
        import matplotlib.pyplot as plt

        feats = list(self.per_feature_b_.keys())
        bvals = [self.per_feature_b_[f] for f in feats]
        cands = self.b_candidates or [2, 15]
        lo, hi = min(cands), max(cands)

        colors = [
            "#2166ac" if b <= lo + (hi - lo) / 3
            else "#1a9641" if b <= lo + 2 * (hi - lo) / 3
            else "#d7191c"
            for b in bvals
        ]

        fig, ax = plt.subplots(figsize=figsize or (max(7, len(feats) * 0.5 + 2), 4))
        ax.bar(range(len(feats)), bvals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Chosen B_j", fontsize=10)
        ax.set_title(
            f"Adaptive binning — chosen B per feature  "
            f"(threshold={self.min_marginal_gain_ratio:.0%})",
            fontsize=11,
        )
        for i, b in enumerate(bvals):
            ax.text(i, b + 0.05, str(b), ha="center", fontsize=8)
        fig.tight_layout()
        return fig, ax

    def ig_heatmap(self, figsize: tuple | None = None):
        """Heatmap of IG score at every (feature, B) grid point (adaptive binning only).

        The chosen B per feature is highlighted with a bounding box.

        Returns
        -------
        (fig, ax)

        Raises
        ------
        RuntimeError
            When called on a non-adaptive or unfitted model, or when
            ``ig_scores_`` is empty.
        ImportError
            When matplotlib is not installed.
        """
        self._check_adaptive_fitted("ig_heatmap")
        if not getattr(self, "ig_scores_", None):
            raise RuntimeError("ig_scores_ is empty — call fit() first.")
        self._require_mpl()
        import matplotlib.pyplot as plt

        feats = sorted(self.ig_scores_)
        bs = sorted({b for sc in self.ig_scores_.values() for b in sc})
        grid = np.array([[self.ig_scores_[f].get(b, 0.0) for b in bs] for f in feats])

        fig, ax = plt.subplots(
            figsize=figsize or (max(6, len(bs) * 0.9), max(4, len(feats) * 0.45))
        )
        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(bs)))
        ax.set_xticklabels([str(b) for b in bs], fontsize=9)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=8)
        ax.set_xlabel("B candidates", fontsize=10)
        ax.set_title("IG score per (feature, B)  — box = chosen B", fontsize=11)
        for i, f in enumerate(feats):
            chosen = self.per_feature_b_.get(f)
            if chosen and chosen in bs:
                j = bs.index(chosen)
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="black", linewidth=2,
                ))
        plt.colorbar(im, ax=ax, label="Information gain")
        fig.tight_layout()
        return fig, ax

    def _check_adaptive_fitted(self, method_name: str) -> None:
        """Raise a clear error when an adaptive-only method is called incorrectly."""
        check_is_fitted(self)
        if not getattr(self, "adaptive_binning", False):
            raise RuntimeError(
                f"{method_name}() is only available when adaptive_binning=True.  "
                f"Re-fit with HUGIMLClassifierNative(adaptive_binning=True, ...) "
                f"or use HUGIMLAdaptive."
            )
        if not getattr(self, "per_feature_b_", None):
            raise RuntimeError(
                f"{method_name}() requires per_feature_b_ — call fit() first."
            )

    @staticmethod
    def _require_mpl() -> None:
        """Raise ImportError when matplotlib is not installed."""
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise ImportError(
                "matplotlib is required for diagnostic plots. "
                "Install with:  pip install matplotlib  "
                "or:  pip install 'hugiml-core[plots]'"
            )

    # ── End v1.1.0 adaptive-binning diagnostic plots ──────────────────────────

    def model_summary(self) -> str:
        """Human-readable model summary including top patterns."""
        check_is_fitted(self)
        lines = [
            "HUGIMLClassifierNative — Model Summary",
            "=" * 50,
            f"Config:       B={self.B}, L={self.L}, G={self.G}",
            f"Feature mode: {getattr(self, 'feature_mode', 'patterns_only')}",
            f"Training:     {self.fit_metadata_.n_samples} samples, "
            f"{self.fit_metadata_.n_features} features, "
            f"{self.fit_metadata_.n_classes} classes",
            f"Patterns:     {self.fit_metadata_.n_patterns} "
            f"({self.fit_metadata_.n_compound} compound)",
            f"Matrix:       {self.x_train_hup_.shape} "
            f"(density={self.fit_metadata_.matrix_density:.4f})",
            f"Downstream:   {getattr(self, 'x_train_downstream_', self.x_train_hup_).shape}",
            f"Fit time:     {self.fit_metadata_.total_fit_ms:.0f} ms",
            "",
            "Stage breakdown (ms):",
        ]
        for stage, ms in self.fit_metadata_.stage_times_ms.items():
            lines.append(f"  {stage:<25} {ms:>8.1f}")
        lines += ["", "Top 10 patterns by importance:"]

        try:
            imp = self.feature_importances().head(10)
            for _, row in imp.iterrows():
                lines.append(
                    f"  {row['pattern']:<40} "
                    f"coef={row['coefficient']:>+8.4f}  "
                    f"sup={row['support']:.3f}"
                )
        except AttributeError:
            lines.append("  (not available — non-LR downstream estimator)")

        # ── v1.1.0 adaptive binning section ──────────────────────────────
        if getattr(self, "_missing_col_edges_", None):
            lines += [
                "",
                f"NaN handling: {len(self._missing_col_edges_)} numerical column(s) "
                f"pre-binned (NaN/Inf generates no transaction item at train or test time).",
            ]
        if self.adaptive_binning and getattr(self, "per_feature_b_", None):
            lines += ["", "Adaptive binning — chosen B per feature:"]
            for feat, b in sorted(
                self.per_feature_b_.items(), key=lambda kv: -kv[1]
            ):
                edges = self._bin_edges_.get(feat, [])
                rng = (
                    f"  [{float(edges[0]):.4g}…{float(edges[-1]):.4g}]"
                    if len(edges) >= 2 else ""
                )
                lines.append(f"  {feat:<35} B={b:<3}{rng}")
        # ─────────────────────────────────────────────────────────────────

        return "\n".join(lines)

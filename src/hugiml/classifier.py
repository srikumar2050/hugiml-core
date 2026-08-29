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

"""HUGIMLClassifier — C++ accelerated, scikit-learn compatible classifier.

``HUGIMLClassifier`` is the primary public class name.
``HUGIMLClassifierNative`` remains as a backward-compatible alias.

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
    estimator (LogisticRegression default, with optional saga/SGD downstream solvers), explanation methods
    (get_hug_features, get_pattern_info, feature_importances), versioned
    model serialisation, prediction monitoring, multi-method drift detection,
    latency SLA enforcement, and graceful degradation under memory pressure.

Quick start
-----------
Two usage paths are supported:

**Path A — prepareXy** (recommended when the full dataset is available upfront)::

    from hugiml import HUGIMLClassifier

    clf = HUGIMLClassifier()
    X, y = clf.prepareXy(X_df, y_series)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)

    print(clf.model_summary())
    print(clf.feature_importances())

**Path B — allCols + origColumns** (cross-validation loops)::

    clf = HUGIMLClassifier(
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
    clf2 = HUGIMLClassifier.load_model("model.hugiml")
"""

from __future__ import annotations

import threading
import warnings
from typing import Any

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from hugiml import _binning as _binning
from hugiml import _compat
from hugiml import monitoring as _monitoring
from hugiml import serialization as _serialization

try:
    import _hugiml_core as _core

    _CORE_AVAILABLE: bool = True
    _CORE_IMPORT_ERROR: ImportError | None = None
except ImportError as _e:
    _core = None
    _CORE_AVAILABLE = False
    _CORE_IMPORT_ERROR: ImportError | None = _e
    warnings.warn(
        f"hugiml-core: the compiled C++ extension '_hugiml_core' could not be imported "
        f"({_e}). "
        f"All classifier fit/transform operations will raise ImportError. "
        f"On Linux x86_64 with glibc >= 2.17, a pre-built wheel should have been installed "
        f"automatically — verify your platform matches a published wheel at "
        f"https://pypi.org/project/hugiml-core/#files. "
        f"To build from source: pip install . --no-build-isolation",
        RuntimeWarning,
        stacklevel=2,
    )

from hugiml import _classifier_support as _support
from hugiml import _classifier_tuning as _tuning
from hugiml import exceptions as _exceptions
from hugiml._classifier_binning import _BinningMixin
from hugiml._classifier_estimator import _EstimatorMixin
from hugiml._classifier_features import _FeatureAssemblyMixin
from hugiml._classifier_inspection import _InspectionMixin
from hugiml._classifier_interpretation import _InterpretationMixin
from hugiml._classifier_prediction import _PredictionMixin
from hugiml._classifier_training import _TrainingMixin
from hugiml.hyperparameter_configs import DEFAULT_HUGIML_GRID_NAME, get_hugiml_grid

_adap_apply_edges = _binning._apply_edges
_adap_quantile_edges = _binning._quantile_edges
_adap_select_b = _binning._select_b
check_array = _compat.check_array
check_X_y = _compat.check_X_y
liblinear_penalty_kwargs = _compat.liblinear_penalty_kwargs
MIN_SCHEMA_VERSION = _serialization.MIN_SCHEMA_VERSION
MODEL_SCHEMA_VERSION = _serialization.MODEL_SCHEMA_VERSION
_load_model = _serialization.load_model
_save_model = _serialization.save_model

DriftDetector = _monitoring.DriftDetector
PredictionMonitor = _monitoring.PredictionMonitor
HUGIMLConvergenceWarning = _exceptions.HUGIMLConvergenceWarning
HUGIMLDtypeDriftWarning = _exceptions.HUGIMLDtypeDriftWarning
HUGIMLMemoryError = _exceptions.HUGIMLMemoryError
HUGIMLParamError = _exceptions.HUGIMLParamError
HUGIMLPredictionError = _exceptions.HUGIMLPredictionError
HUGIMLRangeWarning = _exceptions.HUGIMLRangeWarning
HUGIMLSchemaError = _exceptions.HUGIMLSchemaError
HUGIMLTimeoutError = _exceptions.HUGIMLTimeoutError
HUGIMLValidationError = _exceptions.HUGIMLValidationError
HUGIMLVersionError = _exceptions.HUGIMLVersionError
HUGIMLWarning = _exceptions.HUGIMLWarning

DEFAULT_AUGMENTED_PAIR_MAX_FEATURES = _support.DEFAULT_AUGMENTED_PAIR_MAX_FEATURES
DEFAULT_AUGMENTED_PAIR_UNBOUNDED_CAP = _support.DEFAULT_AUGMENTED_PAIR_UNBOUNDED_CAP
AUGMENTED_PAIR_OPS = _support.AUGMENTED_PAIR_OPS
AUGMENTED_PAIR_MODES = _support.AUGMENTED_PAIR_MODES
_II_N_BINS = _support._II_N_BINS
_NUMERIC_DTYPE_KINDS = _support._NUMERIC_DTYPE_KINDS
_PRESETS = _support._PRESETS
_tracemalloc_lock = _support._tracemalloc_lock
_resource = getattr(_support, "_resource", None)
_is_zero_variance_numeric_column = _support._is_zero_variance_numeric_column
FitMetadata = _support.FitMetadata
NativeAugmentedPairTransformBlock = _support.NativeAugmentedPairTransformBlock
_MemoryTracker = _support._MemoryTracker
_TransactionDataWrapper = _support._TransactionDataWrapper
_best_ig_score = _support._best_ig_score
_codes_from_edges = _support._codes_from_edges
_continuous_to_quantile_codes = _support._continuous_to_quantile_codes
_dense_full_csr = _support._dense_full_csr
_edge_information_gain = _support._edge_information_gain
_entropy_from_counts = _support._entropy_from_counts
_get_peak_rss_kb = _support._get_peak_rss_kb
_information_gain_from_codes = _support._information_gain_from_codes
_is_binary_feature_series = _support._is_binary_feature_series
_joint_information_gain_from_binned_columns = _support._joint_information_gain_from_binned_columns
_wire_hugiml_feature_metadata = _support._wire_hugiml_feature_metadata

HUGIMLTuneResult = _tuning.HUGIMLTuneResult
_hugiml_auc_score_for_fast_grid = _tuning._hugiml_auc_score_for_fast_grid
_hugiml_build_fast_tune_adaptive_context = _tuning._hugiml_build_fast_tune_adaptive_context
_hugiml_expand_grid_for_fast_tune = _tuning._hugiml_expand_grid_for_fast_tune
_hugiml_fast_grid_tune = _tuning._hugiml_fast_grid_tune
_hugiml_fit_downstream_estimator_from_template = (
    _tuning._hugiml_fit_downstream_estimator_from_template
)
_hugiml_params_key = _tuning._hugiml_params_key
_hugiml_prepare_candidate_from_cached_base = _tuning._hugiml_prepare_candidate_from_cached_base
_hugiml_prepare_downstream_template_from_cached_base = (
    _tuning._hugiml_prepare_downstream_template_from_cached_base
)
_hugiml_score_model_for_tune = _tuning._hugiml_score_model_for_tune
_hugiml_shallow_candidate_from_base = _tuning._hugiml_shallow_candidate_from_base
_hugiml_standard_grid_tune_one_split = _tuning._hugiml_standard_grid_tune_one_split
_hugiml_tune = _tuning._hugiml_tune
_hugiml_validate_fast_tune_grid = _tuning._hugiml_validate_fast_tune_grid


class HUGIMLClassifier(
    _EstimatorMixin,
    _BinningMixin,
    _TrainingMixin,
    _FeatureAssemblyMixin,
    _InterpretationMixin,
    _PredictionMixin,
    _InspectionMixin,
    TransformerMixin,
    ClassifierMixin,
    BaseEstimator,
):
    """HUG-IML interpretable classifier — C++ accelerated, scikit-learn compatible.

    Extracts High Utility Gain (HUG) patterns from labelled tabular data,
    assembles the configured downstream representation, and fits an
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
        Number of quantile bins per numerical feature when
        ``adaptive_binning=False``. With adaptive binning enabled, per-feature
        bin counts are selected from ``b_candidates``.
    L : int, default 1
        Maximum HUG pattern length. 1 = singletons; 2 = pairs; -1 = unlimited.
    G : float, default 1e-3
        Minimum information-gain threshold.
    topK : int, default 30
        Maximum number of patterns to retain. -1 computes automatically.
    base_estimator : sklearn estimator, optional
        Downstream classifier trained on the selected representation.
        Defaults to LogisticRegression. An explicit LogisticRegression using
        the ``liblinear`` solver is fitted directly for binary targets and
        through one-vs-rest classification for targets with three or more
        classes.
    lr_solver : {"auto", "saga", "sgd"}, default "auto"
        Downstream linear classifier used when ``base_estimator`` is not supplied.
        ``"auto"`` uses L1-regularized logistic regression: binary classifiers use
        the ``liblinear`` solver and multiclass classifiers use the ``saga`` solver. ``"saga"`` uses
        ``LogisticRegression(solver="saga")``. ``"sgd"`` uses
        ``SGDClassifier(loss="log_loss")`` so large sparse downstream matrices can
        be trained with stochastic gradient descent. All built-in choices keep the
        existing deterministic ``random_state=0`` and ``max_iter=500`` defaults.
    n_jobs : int, default 1
        Number of OpenMP threads. -1 uses all available cores.
    max_predict_ms : float or None
        Prediction latency budget in milliseconds.
    max_fit_seconds : float or None
        Backward-compatible alias for the mining-stage wall-clock budget.
        Transaction preparation and downstream model fitting are not bounded
        by this value. Prefer ``max_mining_seconds`` for new code.
    max_mining_seconds : float or None
        Wall-clock budget, in seconds, for native pattern mining. This is
        especially useful for explicit high-order bounded mining such as
        ``L=4``/``L=5``/larger values. Use ``1800`` for a 30-minute mining
        cap. When unset, ``max_fit_seconds`` is used for backward
        compatibility. Partial patterns mined before timeout can be retained
        when ``mining_degradation_policy="allow"``.
    mining_degradation_policy : {"allow", "raise"}, default "allow"
        Policy for resource-driven mining degradation. ``"allow"`` preserves
        staged memory recovery and partial timeout results while recording the
        outcome in ``mining_audit_log_`` and ``fit_metadata_``. ``"raise"``
        rejects those degraded outcomes with ``HUGIMLMemoryError`` or
        ``HUGIMLTimeoutError``.
    adaptive_binning : bool, default True
        Select per-feature numeric bin counts using supervised information gain.
    b_candidates : list of int or None
        Candidate bin counts evaluated when adaptive binning is enabled.
    min_marginal_gain_ratio : float, default 0.02
        Elbow threshold for adaptive-binning marginal gain.
    adaptive_binning_sample_frac : float or bool, default False
        Fraction of training rows used for adaptive-bin selection. ``False``
        uses all rows; a float in ``(0, 1]`` uses a deterministic stratified
        sample for selecting edges before applying those edges to all rows.
    adaptive_binning_sample_random_state : int, default 42
        Random seed used when ``adaptive_binning_sample_frac`` requests a
        stratified sample.
    convert_binary_to_categorical : bool, default False
        When enabled, numeric columns with exactly two observed values are
        inferred as categorical indicators during automatic column detection.
        The default keeps them numeric so they remain eligible for numeric
        interaction and augmented-pair paths. The named performance grids
        explicitly keep this disabled, while the named interpretability grids
        enable it for the categorical pattern surface. Explicit ``allCols``
        metadata takes precedence over this inference option.
    feature_mode : {"patterns_only", "original_plus_patterns",
        "original_plus_interactions"}, default "patterns_only"
        Downstream representation used by fit/predict/transform APIs.
        ``transform_patterns(X)`` returns the binary HUG pattern matrix.
    lr_source_policy : {"standard", "main_effect", "strict"}, default "standard"
        Raw-source reuse policy applied immediately before downstream logistic
        fitting. ``main_effect`` preserves surviving ``orig:`` main-effect
        columns while source-disjointing generated contextual terms; in
        ``patterns_only`` it is equivalent to ``strict``.
    use_hotpath : bool, default True
        Use the fused native ``L=1`` preparation/mining/matrix path when
        eligible. Disable only for diagnostic equivalence checks against the
        staged path.
    augmented_pair_transforms : bool, default True
        Enable downstream augmented-pair operator features for eligible
        ``L >= 2`` adaptive-binning models.
    augmented_pair_mode : {"interaction_information", "marginal_ig"},
        default "interaction_information"
        Source-column scorer for augmented-pair features.
    ii_partner_size : int or None
        Optional partner-search bound for interaction-information scoring.
    aug_feature_size : int, default 10
        Number of source columns retained for augmented-pair candidate
        generation in interaction-information mode.
    max_pair_features : int, default 10
        Source-column budget used by the marginal-IG augmented-pair mode.
    augmented_pair_max_features : int or None
        v1.1.11-compatible alias for the augmented-pair source budget. When
        provided with default new budgets, it maps to both ``aug_feature_size``
        and ``max_pair_features``.
    topk_budget_strict : bool, default False
        Apply one global ``topK`` cap across the constructed downstream
        representation.
    dense_downstream_max_width : int, default 200
        Width threshold below which downstream matrices may stay dense.
    execution_mode : {"audit", "production"}, default "audit"
        Artifact-retention mode.
    interaction_relaxed_mining : bool, default False
        Allow interaction-information survivors to participate in native mining
        as original-feature bins without creating augmented-pair operator
        columns. Relaxed admission covers the root and its immediate first-child
        pairing partner; deeper positions receive no new admission exemption.
        The generic miner still requires the constructed child pattern to clear
        its joint information-gain gate. Mutually exclusive with augmented-pair
        transforms at ``L >= 2``.
    interaction_relaxed_feature_size : int, default 10
        Survivor-source budget for interaction-relaxed mining.
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

    # The grid values themselves live in hugiml.hyperparameter_configs, shared
    # with the benchmark runner and dashboard rather than declared here only.
    # See default_param_grid() below for the public, name-aware accessor.
    DEFAULT_PARAM_GRID: dict[str, list] = get_hugiml_grid(DEFAULT_HUGIML_GRID_NAME)

    def __init__(
        self,
        allCols: list | None = None,
        origColumns: list | None = None,
        B: int = 8,
        L: int = 1,
        G: float = 1e-3,
        topK: int = 30,
        base_estimator: Any = None,
        lr_solver: str = "auto",
        n_jobs: int = 1,
        max_predict_ms: float | None = None,
        max_fit_seconds: float | None = None,
        max_mining_seconds: float | None = None,
        mining_degradation_policy: str = "allow",
        verbose: bool = False,
        # ── Adaptive binning ──────────────────────────────────────────────
        # When adaptive_binning=True each numerical feature is pre-discretised
        # to B_j quantile bins chosen by elbow-stopping IG search.  The
        # pre-binned columns are declared categorical before the C++ layer
        # (global B is overridden to sentinel 2).  Bin edges are stored in
        # _bin_edges_ and reapplied identically at predict/transform time.
        #
        # adaptive_binning=True avoids requiring the user to guess a single
        # global B: on internal benchmarks, accuracy with a constant B varied
        # by several points of ROC-AUC depending on the value chosen (e.g.
        # B=3 vs B=20), with no single constant value best across datasets.
        # Adaptive binning lands at parity with a well-chosen constant B
        # without requiring a search for one; head-to-head testing shows it
        # roughly tied with, not uniformly ahead of, a well-chosen constant B,
        # so the benefit is removing a sensitive hyperparameter from the
        # quick-start path rather than a raw accuracy gain. Pass
        # adaptive_binning=False to use a single constant B instead.
        # ─────────────────────────────────────────────────────────────────
        adaptive_binning: bool = True,
        b_candidates: list | None = None,
        min_marginal_gain_ratio: float = 0.02,
        adaptive_binning_sample_frac: float | bool = False,
        adaptive_binning_sample_random_state: int = 42,
        # When True, any
        # *numeric* column with exactly two distinct values is automatically
        # treated as categorical during column-type detection (see
        # _resolve_col_meta), the same as an explicitly categorical
        # (object/string/pandas Categorical) column. This is sometimes
        # desirable for genuinely nominal binary columns (e.g. a yes/no
        # flag encoded as 0/1) -- set convert_binary_to_categorical=True to
        # restore that behavior -- but the default (False) treats binary-
        # valued numeric columns as numeric, matching their pandas dtype
        # rather than their observed cardinality, because the previous
        # default (True) silently excluded every such column from
        # augmented-pair transforms: _numeric_feature_names_for_augmented_pairs()
        # only considers columns NOT marked categorical, so a dataset made up
        # entirely of 0/1-valued *measurements* (not categories) used to end
        # up with augmented_pair_transforms constructing zero pairs
        # regardless of augmented_pair_transforms=True, with no warning.
        # Has no effect when column types are supplied explicitly via
        # allCols/origColumns, since that path never auto-detects types.
        convert_binary_to_categorical: bool = False,
        feature_mode: str = "patterns_only",
        use_hotpath: bool = True,
        augmented_pair_transforms: bool = True,
        augmented_pair_mode: str = "interaction_information",
        ii_partner_size: int | None = None,
        aug_feature_size: int = 10,
        max_pair_features: int = 10,
        augmented_pair_max_features: int | None = None,
        topk_budget_strict: bool = False,
        dense_downstream_max_width: int = 200,
        execution_mode: str = "audit",
        # When True, interaction-information survivors can enter native mining
        # as ordinary source columns even when their marginal IG is weak. Relaxed
        # admission covers the first two positions of the initial branch: the root
        # and its immediate first-child pairing partner. In the generic miner, a
        # survivor at either position can bypass its own singleton admission gates
        # long enough for the joint child pattern to be scored; no new relaxed
        # admission is introduced at item positions 2+, where survivor items must
        # pass the ordinary gates. The specialized L=2 path likewise treats a pair
        # as relaxed when either member is a survivor. See native/mining.hpp for
        # the exact pruning and heap-routing scope.
        # No augmented-pair operator features (sum/product/etc.) are generated
        # by this path.  The resulting HUG patterns remain conjunctions of
        # original feature bins and are annotated for audit APIs as
        # survivor-led patterns when they include a relaxed survivor column.
        # Mutually exclusive with augmented_pair_transforms at L >= 2
        # (validated in _validate_params). At L=1 this is a no-op because L=1
        # uses the fused hotpath, which has no relaxed variant.  Effective for
        # L in {2, 3, -1}.
        interaction_relaxed_mining: bool = False,
        # Number of columns select_interaction_information_features returns
        # for interaction_relaxed_mining's own column selection. Separate
        # from aug_feature_size, which controls the unrelated
        # augmented_pair_mode source-column count.
        interaction_relaxed_feature_size: int = 10,
        lr_source_policy: str = "standard",
    ) -> None:
        self.allCols = allCols
        self.origColumns = origColumns
        self.B = B
        self.L = L
        self.G = G
        self.topK = topK
        self.base_estimator = base_estimator
        self.lr_solver = lr_solver
        self.n_jobs = n_jobs
        self.max_predict_ms = max_predict_ms
        self.max_fit_seconds = max_fit_seconds
        self.max_mining_seconds = max_mining_seconds
        self.mining_degradation_policy = mining_degradation_policy
        self.verbose = verbose
        self.adaptive_binning = adaptive_binning
        self.b_candidates = b_candidates
        self.min_marginal_gain_ratio = min_marginal_gain_ratio
        self.adaptive_binning_sample_frac = adaptive_binning_sample_frac
        self.adaptive_binning_sample_random_state = adaptive_binning_sample_random_state
        self.convert_binary_to_categorical = convert_binary_to_categorical
        self.feature_mode = feature_mode
        self.lr_source_policy = lr_source_policy
        self.use_hotpath = use_hotpath
        self.augmented_pair_transforms = augmented_pair_transforms
        self.augmented_pair_mode = augmented_pair_mode
        self.ii_partner_size = ii_partner_size
        if isinstance(augmented_pair_max_features, int) and not isinstance(
            augmented_pair_max_features, bool
        ):
            if (
                isinstance(aug_feature_size, int)
                and not isinstance(aug_feature_size, bool)
                and aug_feature_size == DEFAULT_AUGMENTED_PAIR_MAX_FEATURES
            ):
                aug_feature_size = augmented_pair_max_features
            if (
                isinstance(max_pair_features, int)
                and not isinstance(max_pair_features, bool)
                and max_pair_features == DEFAULT_AUGMENTED_PAIR_MAX_FEATURES
            ):
                max_pair_features = augmented_pair_max_features
        self.aug_feature_size = aug_feature_size
        self.max_pair_features = max_pair_features
        self.augmented_pair_max_features = augmented_pair_max_features
        self.topk_budget_strict = topk_budget_strict
        self.dense_downstream_max_width = dense_downstream_max_width
        # sklearn estimator compatibility: constructor and set_params must not
        # validate parameter values.  execution_mode is validated in fit/load.
        self.execution_mode = execution_mode
        self.interaction_relaxed_mining = interaction_relaxed_mining
        self.interaction_relaxed_feature_size = interaction_relaxed_feature_size
        self._fit_lock = threading.RLock()


NativeAugmentedPairTransformBlock.__module__ = __name__
FitMetadata.__module__ = __name__
_MemoryTracker.__module__ = __name__
_TransactionDataWrapper.__module__ = __name__
HUGIMLTuneResult.__module__ = __name__

HUGIMLClassifier.fast_grid_tune = classmethod(_hugiml_fast_grid_tune)
HUGIMLClassifier.tune = classmethod(_hugiml_tune)
HUGIMLClassifierNative = HUGIMLClassifier

__all__ = [
    "HUGIMLClassifier",
    "HUGIMLClassifierNative",
    "FitMetadata",
    "HUGIMLTuneResult",
]

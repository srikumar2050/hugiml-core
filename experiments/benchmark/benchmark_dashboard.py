from __future__ import annotations

# ruff: noqa: E402
import argparse
import copy
import hashlib
import html
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR_NAME = "results"
DEFAULT_TEMPLATE_NAME = "hugiml_benchmark_analysis_dashboard.html"
DATASET_FEATURE_POLICY = "statsmodels_data_columns_only_no_source_index_v1"


def _find_repo_root(start: Path) -> Path | None:
    """Find the nearest parent containing src/hugiml, or return None."""
    for base in [start, *start.parents]:
        if (base / "src" / "hugiml").exists():
            return base.resolve()
    return None


# Prefer an explicitly supplied repo root; otherwise discover one by walking
# upward from this script.  If neither is available, use the installed package.
SOURCE_ROOT_ENV = os.environ.get("HUGIML_SOURCE_ROOT")
SOURCE_ROOT = (
    Path(SOURCE_ROOT_ENV).expanduser().resolve() if SOURCE_ROOT_ENV else _find_repo_root(SCRIPT_DIR)
)
if SOURCE_ROOT is not None and (SOURCE_ROOT / "src").exists():
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

# is_categorical_dtype was removed in pandas 2.2+; provide a safe fallback.
try:
    from pandas.api.types import is_categorical_dtype as _is_categorical_dtype
except ImportError:
    def _is_categorical_dtype(arr_or_dtype) -> bool:
        try:
            return isinstance(getattr(arr_or_dtype, "dtype", arr_or_dtype), pd.CategoricalDtype)
        except Exception:
            return False
is_categorical_dtype = _is_categorical_dtype

try:
    import statsmodels.api as sm
except Exception as exc:
    sm = None
    STATSMODELS_IMPORT_ERROR = exc
else:
    STATSMODELS_IMPORT_ERROR = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from scipy.stats import t as student_t
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_linnerud,
    load_wine,
    make_circles,
    make_classification,
    make_moons,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except Exception:  # pragma: no cover - optional benchmark dependency
    ExplainableBoostingClassifier = None

try:
    from imodels import RuleFitClassifier
except Exception:  # pragma: no cover - optional benchmark dependency
    RuleFitClassifier = None

import hugiml as _hugiml_pkg
from hugiml import HUGIMLClassifierNative, get_complexity, get_complexity_report

try:
    from hugiml.benchmarks.runner import (
        _prepare_features_for_hugiml,
        _wrap_non_hugiml_pipeline,
    )
except Exception:

    def _prepare_features_for_hugiml(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy().replace([np.inf, -np.inf], np.nan)
        for col in list(X.columns):
            if X[col].isna().all():
                X = X.drop(columns=[col])
                continue
            if (
                is_bool_dtype(X[col])
                or is_object_dtype(X[col])
                or is_string_dtype(X[col])
                or is_categorical_dtype(X[col])
            ):
                X[col] = X[col].astype("category")
            elif is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                X[col] = X[col].astype("category")
        return X

    def _make_ohe():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)

    def _wrap_non_hugiml_pipeline(model):
        prep = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                    make_column_selector(dtype_include=np.number),
                ),
                (
                    "cat",
                    Pipeline(
                        [("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", _make_ohe())]
                    ),
                    make_column_selector(dtype_exclude=np.number),
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return Pipeline([("prep", prep), ("model", model)])


def _dense_numeric_array(X: Any) -> np.ndarray:
    """Return a writable dense numeric matrix without dataframe feature labels."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    array = np.array(X, dtype=np.float32, order="C", copy=True)
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional feature matrix, received shape {array.shape}")
    return array


class DenseArrayTransformer(BaseEstimator, TransformerMixin):
    """Convert preprocessed baseline features to an unlabeled numeric matrix."""

    def fit(self, X: Any, y: Any = None) -> DenseArrayTransformer:
        return self

    def transform(self, X: Any) -> np.ndarray:
        return _dense_numeric_array(X)


def baseline_pipeline(estimator: Any) -> Pipeline:
    """Build a baseline pipeline whose estimator receives only a numeric array."""
    wrapped = _wrap_non_hugiml_pipeline(estimator)
    steps = list(getattr(wrapped, "steps", []))
    if not steps or steps[-1][0] != "model":
        raise TypeError("Baseline preprocessing must return a Pipeline ending in a 'model' step")
    if any(name == "array" for name, _ in steps[:-1]):
        return Pipeline(steps)
    return Pipeline([*steps[:-1], ("array", DenseArrayTransformer()), steps[-1]])


def _make_ohe_encoder():
    """Dense OHE used for non-HUGIML baselines."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def _make_non_hugiml_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", _make_ohe_encoder()),
                    ]
                ),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _fit_transform_non_hugiml(
    X_fit_src: pd.DataFrame,
    X_other_src: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None, ColumnTransformer, float]:
    """Fit OHE once and transform matrices for non-HUGIML models.

    The elapsed time is returned so final fit timings can include the fair OHE
    cost without forcing every hyperparameter candidate to refit the same
    deterministic encoder.
    """
    prep = _make_non_hugiml_preprocessor()
    t0 = time.perf_counter()
    X_fit = prep.fit_transform(X_fit_src)
    X_other = prep.transform(X_other_src) if X_other_src is not None else None
    elapsed = time.perf_counter() - t0
    return (
        _dense_numeric_array(X_fit),
        (None if X_other is None else _dense_numeric_array(X_other)),
        prep,
        elapsed,
    )


def _transform_with_preprocessor(prep: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    return _dense_numeric_array(prep.transform(X))


def _preprocessor_feature_count(prep: ColumnTransformer, fallback: int) -> int:
    try:
        return int(len(prep.get_feature_names_out()))
    except Exception:
        return int(fallback)



try:
    from hugiml.hyperparameter_configs import (
        DEFAULT_HUGIML_GRID_NAME,
        get_baseline_grid,
        get_budgeted_baseline_grid,
        get_hugiml_grid,
        make_l1_logistic_base_estimator,
    )
except Exception:
    # PyPI wheels may not ship the benchmark helper grids.  Keep the dashboard
    # benchmark reproducible by embedding the same default grids here.
    from sklearn.multiclass import OneVsRestClassifier

    from hugiml._compat import logistic_penalty_kwargs
    from hugiml.rpte_bounded_lookahead_leafwise import (
        LeafWiseBoundedLookaheadRPTEFeatureLR,
    )

    DEFAULT_HUGIML_GRID_NAME = "performance_ho"

    def make_l1_logistic_base_estimator() -> LogisticRegression:
        return LogisticRegression(
            solver="liblinear",
            C=0.5,
            random_state=0,
            max_iter=300,
            **logistic_penalty_kwargs("l1"),
        )

    def get_hugiml_grid(name: str = DEFAULT_HUGIML_GRID_NAME) -> dict[str, list[Any]]:
        grids = {
            "performance": {
                "B": [-1],
                "adaptive_binning": [True],
                "L": [1, 2],
                "topK": [50, 100],
                "feature_mode": ["original_plus_patterns"],
                "G": [0.01, 0.001],
                "convert_binary_to_categorical": [False],
                "base_estimator": [make_l1_logistic_base_estimator()],
            },
            "interpretability": {
                "B": [-1],
                "adaptive_binning": [True],
                "L": [1, 2],
                "topK": [50, 100],
                "feature_mode": ["patterns_only"],
                "G": [0.01, 0.001],
                "interaction_relaxed_mining": [True],
                "convert_binary_to_categorical": [True],
                "augmented_pair_transforms": [False],
                "base_estimator": [make_l1_logistic_base_estimator()],
            },
            "interpretability_ho": {
                "B": [-1],
                "adaptive_binning": [True],
                "L": [1, 2],
                "topK": [50, 100],
                "feature_mode": ["patterns_only"],
                "G": [0.01, 0.001],
                "interaction_relaxed_mining": [True],
                "augmented_pair_transforms": [False],
                "convert_binary_to_categorical": [True],
                "topk_budget_strict": [False],
                "base_estimator": [
                    None,
                    OneVsRestClassifier(
                        LeafWiseBoundedLookaheadRPTEFeatureLR(
                            leaf_config="3xD",
                            depth=4,
                            enable_lookahead=False,
                        ),
                        n_jobs=1,
                    ),
                ],
            },
            "performance_ho": {
                "B": [-1],
                "adaptive_binning": [True],
                "L": [1, 2],
                "topK": [50, 100],
                "feature_mode": ["original_plus_patterns"],
                "G": [0.01, 0.001],
                "convert_binary_to_categorical": [False],
                "augmented_pair_transforms": [True],
                "topk_budget_strict": [False],
                "lr_solver": ["adaptive_l1"],
                "lr_C": [0.5],
                "base_estimator": [
                    None,
                    OneVsRestClassifier(
                        LeafWiseBoundedLookaheadRPTEFeatureLR(
                            leaf_config="3xD",
                            depth=4,
                            enable_lookahead="adaptive",
                            lr_C=0.5,
                            lr_penalty="l1",
                        ),
                        n_jobs=1,
                    ),
                ],
            },
        }
        interaction_l1 = copy.deepcopy(grids["interpretability_ho"])
        interaction_l1["lr_solver"] = ["adaptive_l1"]
        interaction_l1["lr_C"] = [0.5]
        interaction_l1["base_estimator"] = [
            None,
            OneVsRestClassifier(
                LeafWiseBoundedLookaheadRPTEFeatureLR(
                    leaf_config="3xD",
                    depth=4,
                    enable_lookahead=False,
                    lr_C=0.5,
                    lr_penalty="l1",
                ),
                n_jobs=1,
            ),
        ]
        grids["interpretability_ho"] = interaction_l1

        resolved = name or DEFAULT_HUGIML_GRID_NAME
        if resolved not in grids:
            raise KeyError(resolved)
        return {k: list(v) for k, v in grids[resolved].items()}

    def get_baseline_grid(model: str) -> dict[str, list[Any]]:
        grids = {
            "XGBoost": {
                "n_estimators": [100, 200],
                "max_depth": [3, 4],
                "learning_rate": [0.03, 0.1],
                "min_child_weight": [1, 5],
            },
            "LightGBM": {
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.1],
                "num_leaves": [15, 31],
                "min_child_samples": [10, 20],
            },
            "RandomForest": {
                "n_estimators": [200, 400],
                "max_depth": [4, 8],
                "min_samples_leaf": [1, 5],
                "max_features": ["sqrt", 0.5],
            },
            "LogisticRegression": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l1", "l2"],
            "class_weight": [None, "balanced"],
            },
            "EBM": {
                "learning_rate": [0.01, 0.05],
                "max_bins": [32, 64],
                "interactions": [0, 5],
                "max_rounds": [500],
            },
            "RuleFit": {
                "n_estimators": [50, 100],
                "max_rules": [50, 100],
                "tree_size": [5, 10],
            },
        }
        return copy.deepcopy(grids[model])

    def get_budgeted_baseline_grid(model: str) -> dict[str, list[Any]]:
        grids = {
            "XGBoost": {
                "n_estimators": [25, 50],
                "max_depth": [1, 2],
                "learning_rate": [0.03, 0.1],
                "min_child_weight": [1, 5],
            },
            "LightGBM": {
                "n_estimators": [25, 50],
                "num_leaves": [2, 4],
                "learning_rate": [0.03, 0.1],
                "min_child_samples": [10, 20],
            },
            "RandomForest": {
                "n_estimators": [25, 50],
                "max_leaf_nodes": [2, 4],
                "min_samples_leaf": [1, 5],
                "max_features": ["sqrt", 0.5],
                "max_depth": [None],
            },
        }
        return copy.deepcopy(grids[model])


_BASELINE_GRID_PROVIDER = get_baseline_grid


def get_baseline_grid(model: str) -> dict[str, list[Any]]:
    """Return the benchmark grid for a supported baseline family."""
    if model == "LogisticRegression":
        return {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "class_weight": [None, "balanced"],
        }
    return copy.deepcopy(_BASELINE_GRID_PROVIDER(model))


def baseline_constant_parameters(model: str) -> dict[str, Any]:
    settings = {
        "XGBoost": {
            "eval_metric": "logloss",
            "verbosity": 0,
            "n_jobs": 1,
            "random_state": RANDOM_STATE,
        },
        "LightGBM": {
            "verbose": -1,
            "n_jobs": 1,
            "random_state": RANDOM_STATE,
        },
        "RandomForest": {
            "n_jobs": 1,
            "random_state": RANDOM_STATE,
        },
        "LogisticRegression": {
            "solver": "liblinear",
            "max_iter": 2000,
            "random_state": RANDOM_STATE,
        },
        "EBM": {
            "outer_bags": 4,
            "max_rounds": 500,
            "n_jobs": 1,
            "random_state": RANDOM_STATE,
        },
        "RuleFit": {
            "alpha": None,
            "random_state": RANDOM_STATE,
        },
    }
    return dict(settings[model])


# This benchmark intentionally uses the higher-order grid, irrespective of
# the package-wide default used by other APIs.
DEFAULT_HUGIML_GRID_NAME = "performance_ho"

RANDOM_STATE = 42
BUDGET = 200.0
MODEL_ORDER = [
    "HUGIML",
    "XGB standard",
    "XGB complexity-budgeted",
    "LightGBM standard",
    "LightGBM complexity-budgeted",
    "RandomForest standard",
    "RandomForest complexity-budgeted",
    "Logistic Regression",
    "EBM",
    "RuleFit",
]

# The dashboard exposes two HUGIML paths while retaining the plain HUGIML model label.
HUGIML_SCENARIOS: dict[str, dict[str, Any]] = {
    "augmented_pair": {
        "label": "Augmented pair path",
        "description": (
            "The performance_ho grid with original features, mined patterns, "
            "augmented pairs, numeric 0/1 columns retained as numeric sources, "
            "and inner-CV selection between the HUGIML logistic branch "
            "and its corresponding adaptive RPTE branch"
        ),
        "grid_name": "performance_ho",
        "overrides": {},
    },
    "interaction_relaxed": {
        "label": "Interaction-relaxed path",
        "description": (
            "The interpretability_ho grid with pattern-only interaction-relaxed "
            "mining, numeric 0/1 indicators treated categorically, no augmented "
            "pairs, and inner-CV selection between the HUGIML logistic branch "
            "and its corresponding sequential RPTE branch"
        ),
        "grid_name": "interpretability_ho",
        "overrides": {},
    },
}
DEFAULT_DASHBOARD_HUGIML_SCENARIO = "augmented_pair"
# A comprehensive benchmark panel with 50 public real-world classification
# tasks and 50 deterministic synthetic stress tests. Some real-world tasks
# share a source dataset but use distinct, explicitly defined binary targets;
# this mirrors the existing Digits and Fertility task construction.
# Public real-world tasks from scikit-learn and statsmodels
REAL_DATASET_NAMES = [
    'BreastCancerOriginal',
    'DiabetesHighTarget',
    'DigitsOddVsEven64',
    'DigitsHighVsLow64',
    'WineClass1',
    'IrisVersicolor',
    'ANES96_PIDHigh',
    'CCard_HighAvgExp',
    'Cancer_HighRate',
    'CopperHighWorldConsumption',
    'DanishHighMoneyDemand',
    'ElNinoHighAnnualTemp',
    'Engel_HighIncome',
    'FairAffairs',
    'FertilityHigh2010',
    'FertilityHighDecline1960_2010',
    'Grunfeld_HighInvest',
    'HeartTransplantCensorStatus',
    'InterestInflationHighRate',
    'LongleyHighEmployment',
    'MacroData_HighInflation',
    'ModeChoice',
    'Nile_HighVolume6Lag',
    'RandHIE_HighMDVisits',
    'ScotlandHighYesVote',
    'SpectorGrade',
    'StacklossHigh',
    'Star98_HighAboveRate',
    'StateCrimeHighViolent',
    'Sunspots_HighActivity12Lag',
    'IrisSetosa',
    'IrisVirginica',
    'WineClass0',
    'WineClass2',
    'Digits0Vs1Original64',
    'Digits3Vs8Original64',
    'Digits4Vs9Original64',
    'Digits5Vs6Original64',
    'DigitsPrimeVsComposite64',
    'DigitsZeroVsNonzero64',
    'ANES96_Vote',
    'CommitteeHighBills104',
    'CPunishHighExecutions',
    'StrikesHighDuration',
    'CO2HighWeekly12Lag',
    'LinnerudHighWeight',
    'LinnerudHighWaist',
    'LinnerudHighPulse',
    'MacroData_HighUnemployment',
    'StateCrimeHighMurder',
]

# Higher-order binary interaction benchmarks (seed=2026, n=2000, p=32, 5% label noise)
HIGHER_ORDER_SYNTHETIC_NAMES = [
    'xor3_redundant',
    'xor4_redundant',
    'xor5_redundant',
    'xor6_redundant',
    'xor8_redundant',
    'exactly2_of_5',
    'exactly3_of_7',
    'exactly4_of_9',
    'atleast5_of_9',
    'window_3_of_6_or_4_of_8',
    'dnf_6clauses_3way',
    'dnf_4clauses_4way',
    'cnf_5clauses_3way',
    'hierarchical_xor',
    'gated_parity_2of4',
    'mux_2addr_4data',
    'mux_3addr_8data',
    'mux_4addr_16data',
    'two_block_xor_threshold',
    'three_block_majority',
    'nested_exact_threshold',
    'cyclic_4way_dnf',
    'sparse_5way_conjunctions',
    'parity_of_majorities',
    'mixed_order_logic',
]

# All deterministic synthetic tasks
SYNTHETIC_DATASET_NAMES = [
    'SynthLinearLowDim',
    'SynthSparseWide',
    'SynthMoonsNonlinear',
    'SynthCirclesNonlinear',
    'SynthXORInteractions',
    'SynthAdditiveNonlinear',
    'SynthCategoricalRules',
    'SynthCategoricalNumericInteraction',
    'SynthHighCardinalityCategorical',
    'SynthOrdinalCategorical',
    'SynthMixedMissing',
    'SynthMissingNotAtRandom',
    'SynthImbalancedRare',
    'SynthNoisyHighDimensional',
    'SynthSmallNWide',
    'SynthCorrelatedBlocks',
    'SynthThresholdRules',
    'SynthPiecewiseLinear',
    'SynthHeteroskedasticNoise',
    'SynthMostlyCategorical',
    'xor3_redundant',
    'xor4_redundant',
    'xor5_redundant',
    'xor6_redundant',
    'xor8_redundant',
    'exactly2_of_5',
    'exactly3_of_7',
    'exactly4_of_9',
    'atleast5_of_9',
    'window_3_of_6_or_4_of_8',
    'dnf_6clauses_3way',
    'dnf_4clauses_4way',
    'cnf_5clauses_3way',
    'hierarchical_xor',
    'gated_parity_2of4',
    'mux_2addr_4data',
    'mux_3addr_8data',
    'mux_4addr_16data',
    'two_block_xor_threshold',
    'three_block_majority',
    'nested_exact_threshold',
    'cyclic_4way_dnf',
    'sparse_5way_conjunctions',
    'parity_of_majorities',
    'mixed_order_logic',
    'SynthCheckerboard2D',
    'SynthFriedmanThreshold',
    'SynthRotatedHyperplane',
    'SynthRegimeSwitch',
    'SynthRareHighOrderRule',
]

DATASET_NAMES = REAL_DATASET_NAMES + SYNTHETIC_DATASET_NAMES

if len(REAL_DATASET_NAMES) != 50 or len(SYNTHETIC_DATASET_NAMES) != 50:
    raise RuntimeError("Benchmark panel must contain exactly 50 real and 50 synthetic tasks.")


def _binary_from_median(values: pd.Series | np.ndarray) -> np.ndarray:
    y = pd.Series(values).astype(float)
    med = float(np.nanmedian(y))
    out = (y > med).astype(int).to_numpy()
    if len(np.unique(out)) < 2:
        q = float(np.nanquantile(y, 0.65))
        out = (y > q).astype(int).to_numpy()
    return out


def _force_writable_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Return a deep, writable dataframe while preserving HUGIML-native categoricals.

    Some sklearn/statsmodels/pandas paths can produce arrays or categorical codes
    backed by read-only ndarray views.  HUGIML may materialize/clean columns during
    fit, so make every column independently writable before model code sees it.
    """
    X = pd.DataFrame(X).copy(deep=True)
    out = pd.DataFrame(index=X.index.copy())
    for col in X.columns:
        s = X[col]
        if is_categorical_dtype(s):
            vals = s.astype("object").to_numpy(copy=True)
            out[col] = pd.Series(vals, index=X.index, dtype="category")
        elif is_bool_dtype(s) or is_object_dtype(s) or is_string_dtype(s):
            vals = s.astype("object").to_numpy(copy=True)
            out[col] = pd.Series(vals, index=X.index, dtype="category")
        elif is_numeric_dtype(s):
            vals = pd.to_numeric(s, errors="coerce").to_numpy(copy=True)
            out[col] = pd.Series(vals, index=X.index)
        else:
            vals = s.astype("object").to_numpy(copy=True)
            out[col] = pd.Series(vals, index=X.index, dtype="category")
    return out.reset_index(drop=True).copy(deep=True)


def _clean_xy(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Clean X/y while preserving native categoricals for HUGIML.

    This intentionally delegates feature preparation to the package-level
    HUGIML benchmark helper.  Do not call _prepare_features_for_benchmarks here:
    that helper is baseline/OHE-oriented and must not run before HUGIML sees
    the dataframe.
    """
    y = np.array(y, dtype=int, copy=True)
    keep = pd.Series(y).notna().to_numpy()
    X = pd.DataFrame(X).loc[keep].reset_index(drop=True).copy(deep=True)
    y = np.array(y[keep], dtype=int, copy=True)

    X = _prepare_features_for_hugiml(X)
    X = _force_writable_frame(X)
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1].reset_index(drop=True).copy(deep=True)
    return _force_writable_frame(X), y


def _is_categorical_like(s: pd.Series) -> bool:
    return bool(
        is_categorical_dtype(s) or is_object_dtype(s) or is_string_dtype(s) or is_bool_dtype(s)
    )


def _categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if _is_categorical_like(X[c])]


def _unwrap_model(model):
    if hasattr(model, "named_steps") and "model" in getattr(model, "named_steps", {}):
        return model.named_steps["model"]
    return model


def _model_feature_count(model, fallback: int) -> int:
    if hasattr(model, "named_steps") and "prep" in getattr(model, "named_steps", {}):
        prep = model.named_steps["prep"]
        try:
            return int(len(prep.get_feature_names_out()))
        except Exception:
            try:
                return int(len(getattr(prep, "feature_names_out_")))
            except Exception:
                return int(fallback)
    return int(fallback)


def _statsmodels_df(name: str) -> pd.DataFrame:
    """Load only declared statsmodels data columns, never the row index.

    ``load_pandas().data`` may use a RangeIndex, a named entity index (for
    example, state names), or a time index.  Materialising that index as a
    feature leaks row identity/order into cross-validation and can materially
    bias benchmark results.  The benchmark therefore discards the source index
    unconditionally and keeps only columns explicitly supplied as data.
    """
    if sm is None:
        raise RuntimeError(
            "statsmodels is required for this dataset. Install statsmodels or exclude "
            "statsmodels-backed datasets from --datasets."
        ) from STATSMODELS_IMPORT_ERROR
    try:
        dataset_obj = getattr(sm.datasets, name)
    except AttributeError as exc:
        raise RuntimeError(
            f"statsmodels dataset {name!r} is not available in this statsmodels version."
        ) from exc
    data = dataset_obj.load_pandas().data
    return pd.DataFrame(data).reset_index(drop=True).copy(deep=True)


def _synthetic_frame(
    X: np.ndarray,
    y: np.ndarray,
    *,
    stem: str = "x",
    group: str = "Synthetic",
) -> tuple[pd.DataFrame, np.ndarray, str]:
    cols = [f"{stem}{i:03d}" for i in range(X.shape[1])]
    return *_clean_xy(pd.DataFrame(X, columns=cols), np.asarray(y).astype(int)), group


def _cat_series(values, index=None) -> pd.Series:
    vals = np.array(values, dtype=object, copy=True)
    return pd.Series(vals, index=index, dtype="category")



def _ho_clause(X: np.ndarray, terms: list[tuple[int, int]]) -> np.ndarray:
    out = np.ones(len(X), dtype=bool)
    for column, value in terms:
        out &= X[:, column] == value
    return out


def _ho_flip_labels(y: np.ndarray, rng: np.random.Generator, rate: float = 0.05) -> np.ndarray:
    result = np.asarray(y, dtype=np.int8).copy()
    mask = rng.random(len(result)) < rate
    result[mask] = 1 - result[mask]
    return result


def _higher_order_target(name: str, X: np.ndarray) -> np.ndarray:
    def s(a: int, b: int) -> np.ndarray:
        return X[:, a:b].sum(axis=1)

    def parity(a: int, b: int) -> np.ndarray:
        return np.bitwise_xor.reduce(X[:, a:b], axis=1)
    if name.startswith("xor") and name.endswith("_redundant"):
        order = int(name[3:name.index("_")])
        return parity(0, order)
    if name == "exactly2_of_5":
        return s(0, 5) == 2
    if name == "exactly3_of_7":
        return s(0, 7) == 3
    if name == "exactly4_of_9":
        return s(0, 9) == 4
    if name == "atleast5_of_9":
        return s(0, 9) >= 5
    if name == "window_3_of_6_or_4_of_8":
        return (s(0, 6) == 3) | (s(6, 14) == 4)
    if name == "dnf_6clauses_3way":
        specs = [[(3 * i, 1), (3 * i + 1, i % 2), (3 * i + 2, 1 - (i % 2))] for i in range(6)]
        return np.logical_or.reduce([_ho_clause(X, clause) for clause in specs])
    if name == "dnf_4clauses_4way":
        specs = [[(4 * i, 1), (4 * i + 1, 0), (4 * i + 2, i % 2), (4 * i + 3, 1)] for i in range(4)]
        return np.logical_or.reduce([_ho_clause(X, clause) for clause in specs])
    if name == "cnf_5clauses_3way":
        clauses = [(X[:, 3 * i] | X[:, 3 * i + 1] | (1 - X[:, 3 * i + 2])).astype(bool) for i in range(5)]
        return np.logical_and.reduce(clauses)
    if name == "hierarchical_xor":
        a = X[:, 0] & X[:, 1]
        b = X[:, 2] & (1 - X[:, 3])
        c = X[:, 4] & X[:, 5] & X[:, 6]
        return np.bitwise_xor(np.bitwise_xor(a, b), c)
    if name == "gated_parity_2of4":
        gate = X[:, 0] & (1 - X[:, 1])
        return np.where(gate, parity(2, 6), X[:, 6] & X[:, 7])
    if name.startswith("mux_"):
        addr = {"mux_2addr_4data": 2, "mux_3addr_8data": 3, "mux_4addr_16data": 4}[name]
        weights = 2 ** np.arange(addr - 1, -1, -1)
        idx = X[:, :addr] @ weights
        return X[np.arange(len(X)), addr + idx]
    if name == "two_block_xor_threshold":
        a, b, c = s(0, 5) >= 3, s(5, 10) >= 3, s(10, 14) == 2
        return np.logical_xor(np.logical_xor(a, b), c)
    if name == "three_block_majority":
        blocks = np.column_stack([s(0, 5) >= 3, s(5, 10) >= 3, s(10, 15) >= 3])
        return blocks.sum(axis=1) >= 2
    if name == "nested_exact_threshold":
        return ((s(0, 6) == 3) & (s(6, 12) >= 4)) | ((s(12, 18) == 2) & (s(18, 24) >= 3))
    if name == "cyclic_4way_dnf":
        terms = [_ho_clause(X, [(i, 1), ((i + 1) % 12, 0), ((i + 4) % 12, 1), ((i + 7) % 12, 1)]) for i in range(12)]
        return np.logical_or.reduce(terms)
    if name == "sparse_5way_conjunctions":
        terms = [_ho_clause(X, [(5 * i + j, (i + j) % 2) for j in range(5)]) for i in range(5)]
        return np.logical_or.reduce(terms)
    if name == "parity_of_majorities":
        a, b, c = s(0, 7) >= 4, s(7, 14) >= 4, s(14, 21) >= 4
        return np.logical_xor(np.logical_xor(a, b), c)
    if name == "mixed_order_logic":
        a = _ho_clause(X, [(0, 1), (1, 0), (2, 1)])
        b, c, d = s(3, 9) == 3, parity(9, 14), s(14, 22) >= 5
        return (a | b) ^ (c & d)
    raise KeyError(name)


def _load_higher_order_synthetic(name: str) -> tuple[pd.DataFrame, np.ndarray, str] | None:
    if name not in HIGHER_ORDER_SYNTHETIC_NAMES:
        return None
    dataset_index = HIGHER_ORDER_SYNTHETIC_NAMES.index(name)
    rng = np.random.default_rng(np.random.SeedSequence([2026, dataset_index]))
    X = rng.integers(0, 2, size=(2000, 32), dtype=np.int8)
    y = _ho_flip_labels(_higher_order_target(name, X), rng, rate=0.05)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    return *_clean_xy(frame, y), "Synthetic"

def _load_synthetic_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray, str] | None:
    higher_order = _load_higher_order_synthetic(name)
    if higher_order is not None:
        return higher_order

    rng = np.random.default_rng(RANDOM_STATE + 101)

    if name == "SynthLinearLowDim":
        X, y = make_classification(
            n_samples=1200,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            n_repeated=0,
            n_clusters_per_class=2,
            class_sep=1.0,
            flip_y=0.02,
            random_state=RANDOM_STATE + 11,
        )
        return _synthetic_frame(X, y, stem="lin")

    if name == "SynthSparseWide":
        X, y = make_classification(
            n_samples=1200,
            n_features=120,
            n_informative=8,
            n_redundant=6,
            n_repeated=0,
            n_clusters_per_class=2,
            class_sep=1.15,
            flip_y=0.03,
            random_state=RANDOM_STATE + 12,
        )
        mask = rng.random(X.shape) < 0.72
        X = np.array(X, dtype=float, copy=True)
        X[mask] = 0.0
        return _synthetic_frame(X, y, stem="wide")

    if name == "SynthMoonsNonlinear":
        X, y = make_moons(n_samples=1000, noise=0.25, random_state=RANDOM_STATE + 13)
        X = np.column_stack([X, X[:, 0] ** 2, X[:, 1] ** 2, X[:, 0] * X[:, 1]])
        return _synthetic_frame(X, y, stem="moon")

    if name == "SynthCirclesNonlinear":
        X, y = make_circles(n_samples=1000, noise=0.12, factor=0.38, random_state=RANDOM_STATE + 17)
        radius = np.sqrt((X**2).sum(axis=1))
        angle = np.arctan2(X[:, 1], X[:, 0])
        X = np.column_stack([X, radius, angle, X[:, 0] * X[:, 1]])
        return _synthetic_frame(X, y, stem="circ")

    if name == "SynthXORInteractions":
        n = 1200
        X = rng.normal(size=(n, 12))
        score = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
        score += 0.7 * ((X[:, 2] > 0.5) & (X[:, 3] < -0.25)).astype(float)
        score += 0.35 * X[:, 4] - 0.25 * X[:, 5] + rng.normal(0, 0.18, n)
        y = (score > np.median(score)).astype(int)
        return _synthetic_frame(X, y, stem="xor")

    if name == "SynthAdditiveNonlinear":
        n = 1200
        X = rng.normal(size=(n, 16))
        score = np.sin(1.7 * X[:, 0]) + 0.8 * np.maximum(X[:, 1], 0) ** 2
        score -= 0.55 * np.abs(X[:, 2])
        score += 0.45 * X[:, 3] + 0.35 * np.cos(X[:, 4] * X[:, 5])
        score += rng.normal(0, 0.25, n)
        y = (score > np.median(score)).astype(int)
        return _synthetic_frame(X, y, stem="add")

    if name == "SynthCategoricalRules":
        n = 1200
        color = rng.choice(["red", "blue", "green", "yellow"], size=n, p=[0.25, 0.25, 0.35, 0.15])
        region = rng.choice(["north", "south", "east", "west", "central"], size=n)
        channel = rng.choice(["web", "store", "partner"], size=n, p=[0.45, 0.35, 0.20])
        spend = rng.gamma(shape=2.0, scale=1.3, size=n)
        age = rng.normal(42, 12, n)
        score = (
            ((color == "red") & (spend > 2.3)).astype(float)
            + ((region == "west") & (channel == "web")).astype(float)
            + 0.35 * (age > 50).astype(float)
            + rng.normal(0, 0.15, n)
        )
        y = (score > np.quantile(score, 0.58)).astype(int)
        X = pd.DataFrame(
            {"color": color, "region": region, "channel": channel, "spend": spend, "age": age}
        )
        for c in ["color", "region", "channel"]:
            X[c] = _cat_series(X[c], index=X.index)
        return *_clean_xy(X, y), "Synthetic"

    if name == "SynthCategoricalNumericInteraction":
        n = 1300
        segment = rng.choice(["A", "B", "C", "D"], size=n, p=[0.35, 0.25, 0.25, 0.15])
        device = rng.choice(["mobile", "desktop", "tablet"], size=n)
        x0 = rng.normal(size=n)
        x1 = rng.normal(size=n)
        x2 = rng.gamma(2.0, 1.0, n)
        score = 0.2 * x1 + 0.3 * x2
        score += ((segment == "A") & (x0 > 0.4)).astype(float)
        score += ((segment == "C") & (device == "mobile") & (x1 < -0.2)).astype(float)
        score -= 0.7 * ((segment == "D") & (x2 < 1.2)).astype(float)
        score += rng.normal(0, 0.18, n)
        y = (score > np.quantile(score, 0.55)).astype(int)
        X = pd.DataFrame({"segment": segment, "device": device, "x0": x0, "x1": x1, "x2": x2})
        for c in ["segment", "device"]:
            X[c] = _cat_series(X[c], index=X.index)
        return *_clean_xy(X, y), "Synthetic"

    if name == "SynthHighCardinalityCategorical":
        n = 1300
        levels = np.array([f"L{i:02d}" for i in range(60)], dtype=object)
        account = rng.choice(
            levels,
            size=n,
            p=np.linspace(1.8, 0.4, len(levels)) / np.linspace(1.8, 0.4, len(levels)).sum(),
        )
        region = rng.choice(["r0", "r1", "r2", "r3", "r4", "r5"], size=n)
        x0 = rng.normal(size=n)
        active = np.isin(account, levels[[2, 5, 9, 17, 31, 42]]).astype(float)
        score = 1.2 * active + 0.55 * ((region == "r3") & (x0 > -0.15)).astype(float) + 0.25 * x0
        score += rng.normal(0, 0.25, n)
        y = (score > np.quantile(score, 0.62)).astype(int)
        X = pd.DataFrame({"account": account, "region": region, "x0": x0})
        for c in ["account", "region"]:
            X[c] = _cat_series(X[c], index=X.index)
        return *_clean_xy(X, y), "Synthetic"

    if name == "SynthOrdinalCategorical":
        n = 1100
        tier = rng.choice(
            ["bronze", "silver", "gold", "platinum"], size=n, p=[0.35, 0.30, 0.22, 0.13]
        )
        risk = rng.choice(["low", "medium", "high"], size=n, p=[0.5, 0.35, 0.15])
        x0 = rng.normal(size=n)
        tier_score = (
            pd.Series(tier)
            .map({"bronze": 0, "silver": 0.4, "gold": 0.9, "platinum": 1.4})
            .to_numpy(float)
        )
        risk_score = pd.Series(risk).map({"low": -0.2, "medium": 0.25, "high": 0.9}).to_numpy(float)
        score = tier_score + risk_score + 0.35 * x0 + rng.normal(0, 0.22, n)
        y = (score > np.quantile(score, 0.56)).astype(int)
        X = pd.DataFrame({"tier": tier, "risk": risk, "x0": x0})
        for c in ["tier", "risk"]:
            X[c] = _cat_series(X[c], index=X.index)
        return *_clean_xy(X, y), "Synthetic"

    if name == "SynthMixedMissing":
        X, y = make_classification(
            n_samples=1300,
            n_features=14,
            n_informative=7,
            n_redundant=3,
            class_sep=0.9,
            flip_y=0.05,
            random_state=RANDOM_STATE + 14,
        )
        X = np.array(X, dtype=float, copy=True)
        df = pd.DataFrame(X, columns=[f"mix{i:02d}" for i in range(X.shape[1])]).copy(deep=True)
        for c in ["mix01", "mix03", "mix07"]:
            vals = df[c].to_numpy(copy=True)
            miss = rng.random(len(df)) < 0.10
            vals[miss] = np.nan
            df[c] = vals
        band = pd.cut(
            df["mix00"].to_numpy(copy=True),
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=["low", "mid", "high"],
        ).astype("object")
        band = np.array(band, dtype=object, copy=True)
        miss_cat = rng.random(len(df)) < 0.15
        band[miss_cat] = None
        df["band"] = _cat_series(band, index=df.index)
        return *_clean_xy(df, np.array(y, dtype=int, copy=True)), "Synthetic"

    if name == "SynthMissingNotAtRandom":
        n = 1200
        X = rng.normal(size=(n, 10))
        score = 0.9 * X[:, 0] - 0.7 * X[:, 1] + 0.45 * (X[:, 2] > 0.4).astype(float)
        y = (score + rng.normal(0, 0.35, n) > np.median(score)).astype(int)
        df = pd.DataFrame(X, columns=[f"mnar{i:02d}" for i in range(10)])
        miss0 = (y == 1) & (rng.random(n) < 0.28)
        miss1 = (y == 0) & (rng.random(n) < 0.22)
        v0 = df["mnar00"].to_numpy(copy=True)
        v0[miss0] = np.nan
        df["mnar00"] = v0
        v1 = df["mnar01"].to_numpy(copy=True)
        v1[miss1] = np.nan
        df["mnar01"] = v1
        flag = np.where(miss0, "target_like", np.where(miss1, "control_like", "observed"))
        df["missing_pattern"] = _cat_series(flag, index=df.index)
        return *_clean_xy(df, y), "Synthetic"

    if name == "SynthImbalancedRare":
        X, y = make_classification(
            n_samples=1500,
            n_features=18,
            n_informative=6,
            n_redundant=4,
            weights=[0.86, 0.14],
            class_sep=1.25,
            flip_y=0.015,
            random_state=RANDOM_STATE + 15,
        )
        return _synthetic_frame(X, y, stem="imb")

    if name == "SynthNoisyHighDimensional":
        X, y = make_classification(
            n_samples=1400,
            n_features=220,
            n_informative=10,
            n_redundant=10,
            n_clusters_per_class=3,
            class_sep=0.75,
            flip_y=0.12,
            random_state=RANDOM_STATE + 16,
        )
        return _synthetic_frame(X, y, stem="noisy")

    if name == "SynthSmallNWide":
        X, y = make_classification(
            n_samples=180,
            n_features=320,
            n_informative=8,
            n_redundant=8,
            n_clusters_per_class=1,
            class_sep=1.05,
            flip_y=0.04,
            random_state=RANDOM_STATE + 18,
        )
        return _synthetic_frame(X, y, stem="snw")

    if name == "SynthCorrelatedBlocks":
        n = 1200
        base = rng.normal(size=(n, 8))
        blocks = []
        for j in range(base.shape[1]):
            blocks.extend([base[:, j] + rng.normal(0, 0.08, n) for _ in range(4)])
        X = np.column_stack(blocks)
        score = (
            0.9 * base[:, 0]
            - 0.75 * base[:, 1]
            + 0.55 * (base[:, 2] * base[:, 3] > 0).astype(float)
        )
        score += rng.normal(0, 0.3, n)
        y = (score > np.median(score)).astype(int)
        return _synthetic_frame(X, y, stem="corr")

    if name == "SynthThresholdRules":
        n = 1200
        X = rng.normal(size=(n, 14))
        score = ((X[:, 0] > 0.8) | (X[:, 1] < -0.9)).astype(float)
        score += ((X[:, 2] > -0.1) & (X[:, 3] < 0.6)).astype(float)
        score += 0.2 * X[:, 4] + rng.normal(0, 0.12, n)
        y = (score > np.quantile(score, 0.57)).astype(int)
        return _synthetic_frame(X, y, stem="thr")

    if name == "SynthPiecewiseLinear":
        n = 1250
        X = rng.normal(size=(n, 12))
        score = np.where(
            X[:, 0] < -0.3, 1.2 * X[:, 1] - 0.4 * X[:, 2], -0.9 * X[:, 1] + 0.8 * X[:, 3]
        )
        score += np.where(X[:, 4] > 0.5, 0.7, -0.2)
        score += rng.normal(0, 0.28, n)
        y = (score > np.median(score)).astype(int)
        return _synthetic_frame(X, y, stem="pwl")

    if name == "SynthHeteroskedasticNoise":
        n = 1300
        X = rng.normal(size=(n, 14))
        noise_scale = np.where(X[:, 5] > 0.0, 0.85, 0.18)
        score = 0.95 * X[:, 0] - 0.65 * X[:, 1] + 0.45 * (X[:, 2] > 0.2).astype(float)
        y = (score + rng.normal(0, noise_scale, n) > np.median(score)).astype(int)
        return _synthetic_frame(X, y, stem="het")

    if name == "SynthMostlyCategorical":
        n = 1300
        data = {}
        for j in range(8):
            levels = [f"c{j}_{k}" for k in range(4 + (j % 3))]
            data[f"cat{j}"] = rng.choice(levels, size=n)
        x0 = rng.normal(size=n)
        x1 = rng.gamma(2.0, 1.0, n)
        score = ((data["cat0"] == "c0_1") & (data["cat3"] == "c3_2")).astype(float)
        score += ((data["cat5"] == "c5_4") | (data["cat7"] == "c7_1")).astype(float)
        score += 0.35 * x0 - 0.2 * x1 + rng.normal(0, 0.18, n)
        y = (score > np.quantile(score, 0.60)).astype(int)
        df = pd.DataFrame(data)
        for c in df.columns:
            df[c] = _cat_series(df[c], index=df.index)
        df["num0"] = x0
        df["num1"] = x1
        return *_clean_xy(df, y), "Synthetic"


    if name == "SynthCheckerboard2D":
        n = 1400
        xy = rng.uniform(-3.0, 3.0, size=(n, 2))
        cells = np.floor((xy + 3.0) / 0.75).astype(int)
        y = ((cells[:, 0] + cells[:, 1]) % 2).astype(int)
        flip = rng.random(n) < 0.05
        y = np.where(flip, 1 - y, y)
        noise = rng.normal(size=(n, 8))
        X = np.column_stack([xy, noise])
        return _synthetic_frame(X, y, stem="chk")

    if name == "SynthFriedmanThreshold":
        n = 1500
        X = rng.uniform(0.0, 1.0, size=(n, 12))
        score = 10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        score += 20.0 * (X[:, 2] - 0.5) ** 2 + 10.0 * X[:, 3] + 5.0 * X[:, 4]
        score += rng.normal(0.0, 1.5, size=n)
        y = (score > np.median(score)).astype(int)
        return _synthetic_frame(X, y, stem="fr")

    if name == "SynthRotatedHyperplane":
        n, p = 1400, 18
        latent = rng.normal(size=(n, p))
        rotation, _ = np.linalg.qr(rng.normal(size=(p, p)))
        X = latent @ rotation
        score = 1.6 * latent[:, 0] - 1.2 * latent[:, 1] + 0.8 * latent[:, 2]
        score += 0.5 * latent[:, 3] + rng.normal(0.0, 0.7, size=n)
        y = (score > np.median(score)).astype(int)
        return _synthetic_frame(X, y, stem="rot")

    if name == "SynthRegimeSwitch":
        n = 1500
        X = rng.normal(size=(n, 14))
        regime = rng.integers(0, 2, size=n)
        linear_score = 1.1 * X[:, 0] - 0.9 * X[:, 1] + 0.5 * X[:, 2]
        interaction_score = 1.8 * ((X[:, 3] > 0) ^ (X[:, 4] > 0)).astype(float)
        interaction_score += 0.8 * ((X[:, 5] > 0.5) & (X[:, 6] < -0.25)).astype(float) - 0.9
        score = np.where(regime == 0, linear_score, interaction_score)
        score += rng.normal(0.0, 0.35, size=n)
        y = (score > np.median(score)).astype(int)
        frame = pd.DataFrame(X, columns=[f"reg{i:02d}" for i in range(X.shape[1])])
        frame["regime"] = _cat_series(np.where(regime == 1, "B", "A"), index=frame.index)
        return *_clean_xy(frame, y), "Synthetic"

    if name == "SynthRareHighOrderRule":
        n = 1800
        X = rng.integers(0, 2, size=(n, 24), dtype=np.int8)
        rare = np.all(X[:, :5] == np.array([1, 0, 1, 1, 0], dtype=np.int8), axis=1)
        exact = X[:, 5:10].sum(axis=1) == 2
        signed_three_way = (X[:, 10] == 1) & (X[:, 11] == 1) & (X[:, 12] == 0)
        y = (rare | exact | signed_three_way).astype(int)
        flip = rng.random(n) < 0.04
        y = np.where(flip, 1 - y, y)
        return _synthetic_frame(X, y, stem="rare")

    return None


def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray, str]:
    synth = _load_synthetic_dataset(name)
    if synth is not None:
        return synth
    if name.startswith("Iris"):
        d = load_iris(as_frame=True)
        cls = {"IrisSetosa": 0, "IrisVersicolor": 1, "IrisVirginica": 2}[name]
        X, y = _clean_xy(d.data, (d.target == cls).astype(int))
        return X, y, "Real-world"
    if name.startswith("WineClass"):
        d = load_wine(as_frame=True)
        cls = int(name.replace("WineClass", ""))
        X, y = _clean_xy(d.data, (d.target == cls).astype(int))
        return X, y, "Real-world"
    if name == "BreastCancerOriginal":
        d = load_breast_cancer(as_frame=True)
        X, y = _clean_xy(d.data, d.target.astype(int))
        return X, y, "Real-world"
    if name == "DiabetesHighTarget":
        d = load_diabetes(as_frame=True)
        X, y = _clean_xy(d.data, _binary_from_median(d.target))
        return X, y, "Real-world"
    if name.startswith("Digits"):
        d = load_digits(as_frame=True)
        X = d.data.copy()
        y0 = d.target.to_numpy()
        if name == "DigitsOddVsEven64":
            y = (y0 % 2 == 1).astype(int)
        elif name == "DigitsHighVsLow64":
            y = (y0 >= 5).astype(int)
        elif name == "DigitsPrimeVsComposite64":
            mask = np.isin(y0, [2, 3, 4, 5, 6, 7, 8, 9])
            X = X.loc[mask]
            y = np.isin(y0[mask], [2, 3, 5, 7]).astype(int)
        elif name == "DigitsZeroVsNonzero64":
            y = (y0 == 0).astype(int)
        else:
            m = re.match(r"Digits(\d)Vs(\d)Original64", name)
            if not m:
                raise KeyError(name)
            a, b = int(m.group(1)), int(m.group(2))
            mask = np.isin(y0, [a, b])
            X = X.loc[mask]
            y = (y0[mask] == b).astype(int)
        X, y = _clean_xy(X, y)
        return X, y, "Real-world"


    if name == "ANES96_Vote":
        df = _statsmodels_df("anes96")
        y = df["vote"].astype(int).to_numpy()
        X = df.drop(columns=["vote"])
        return *_clean_xy(X, y), "Real-world"
    if name == "CommitteeHighBills104":
        df = _statsmodels_df("committee")
        y = _binary_from_median(df["BILLS104"])
        X = df.drop(columns=["BILLS104"])
        return *_clean_xy(X, y), "Real-world"
    if name == "CPunishHighExecutions":
        df = _statsmodels_df("cpunish")
        y = _binary_from_median(df["EXECUTIONS"])
        X = df.drop(columns=["EXECUTIONS"])
        return *_clean_xy(X, y), "Real-world"
    if name == "StrikesHighDuration":
        df = _statsmodels_df("strikes")
        y = _binary_from_median(df["duration"])
        X = df.drop(columns=["duration"])
        return *_clean_xy(X, y), "Real-world"
    if name == "CO2HighWeekly12Lag":
        if sm is None:
            raise RuntimeError("statsmodels is required for CO2HighWeekly12Lag") from STATSMODELS_IMPORT_ERROR
        co2 = sm.datasets.co2.load_pandas().data["co2"].astype(float).interpolate(limit_direction="both")
        frame = pd.DataFrame(index=np.arange(len(co2)))
        for lag in range(1, 13):
            frame[f"co2_lag{lag}"] = co2.shift(lag).to_numpy()
        week = np.arange(len(co2), dtype=float)
        frame["season_sin"] = np.sin(2.0 * np.pi * week / 52.0)
        frame["season_cos"] = np.cos(2.0 * np.pi * week / 52.0)
        frame["trend"] = week / max(len(co2) - 1, 1)
        frame["target_value"] = co2.to_numpy()
        frame = frame.dropna().reset_index(drop=True)
        y = _binary_from_median(frame.pop("target_value"))
        return *_clean_xy(frame, y), "Real-world"
    if name in {"LinnerudHighWeight", "LinnerudHighWaist", "LinnerudHighPulse"}:
        d = load_linnerud(as_frame=True)
        target_col = {
            "LinnerudHighWeight": "Weight",
            "LinnerudHighWaist": "Waist",
            "LinnerudHighPulse": "Pulse",
        }[name]
        y = _binary_from_median(d.target[target_col])
        return *_clean_xy(d.data, y), "Real-world"
    if name == "MacroData_HighUnemployment":
        df = _statsmodels_df("macrodata")
        y = _binary_from_median(df["unemp"])
        X = df.drop(columns=["unemp"])
        return *_clean_xy(X, y), "Real-world"
    if name == "StateCrimeHighMurder":
        df = _statsmodels_df("statecrime")
        y = _binary_from_median(df["murder"])
        X = df.drop(columns=["murder"])
        return *_clean_xy(X, y), "Real-world"
    if name == "ANES96_PIDHigh":
        df = _statsmodels_df("anes96")
        y = _binary_from_median(df["PID"])
        X = df.drop(columns=["PID"])
        return *_clean_xy(X, y), "Real-world"
    if name == "CCard_HighAvgExp":
        df = _statsmodels_df("ccard")
        y = _binary_from_median(df["AVGEXP"])
        X = df.drop(columns=["AVGEXP"])
        return *_clean_xy(X, y), "Real-world"
    if name == "Cancer_HighRate":
        df = _statsmodels_df("cancer")
        y = _binary_from_median(df["cancer"])
        X = df.drop(columns=["cancer"])
        return *_clean_xy(X, y), "Real-world"
    if name == "CopperHighWorldConsumption":
        df = _statsmodels_df("copper")
        y = _binary_from_median(df["WORLDCONSUMPTION"])
        X = df.drop(columns=["WORLDCONSUMPTION"])
        return *_clean_xy(X, y), "Real-world"
    if name == "DanishHighMoneyDemand":
        df = _statsmodels_df("danish_data")
        y = _binary_from_median(df["lrm"])
        X = df.drop(columns=["lrm"])
        return *_clean_xy(X, y), "Real-world"
    if name == "ElNinoHighAnnualTemp":
        df = _statsmodels_df("elnino")
        months = [c for c in df.columns if c != "YEAR"]
        y = _binary_from_median(df[months].mean(axis=1))
        X = df.copy()
        return *_clean_xy(X, y), "Real-world"
    if name == "Engel_HighIncome":
        df = _statsmodels_df("engel")
        y = _binary_from_median(df["income"])
        X = df.drop(columns=["income"])
        return *_clean_xy(X, y), "Real-world"
    if name == "FairAffairs":
        df = _statsmodels_df("fair")
        y = (df["affairs"] > 0).astype(int).to_numpy()
        X = df.drop(columns=["affairs"])
        return *_clean_xy(X, y), "Real-world"
    if name == "FertilityHigh2010":
        df = _statsmodels_df("fertility")
        y = _binary_from_median(df["2010"])
        X = df.drop(
            columns=[
                "Country Name",
                "Country Code",
                "Indicator Name",
                "Indicator Code",
                "2010",
                "2011",
                "2012",
                "2013",
            ],
            errors="ignore",
        )
        return *_clean_xy(X, y), "Real-world"
    if name == "FertilityHighDecline1960_2010":
        df = _statsmodels_df("fertility")
        decline = pd.to_numeric(df["1960"], errors="coerce") - pd.to_numeric(
            df["2010"], errors="coerce"
        )
        y = _binary_from_median(decline)
        X = df.drop(
            columns=[
                "Country Name",
                "Country Code",
                "Indicator Name",
                "Indicator Code",
                "2010",
                "2011",
                "2012",
                "2013",
            ],
            errors="ignore",
        )
        return *_clean_xy(X, y), "Real-world"
    if name == "Grunfeld_HighInvest":
        df = _statsmodels_df("grunfeld")
        y = _binary_from_median(df["invest"])
        X = df.drop(columns=["invest"])
        return *_clean_xy(X, y), "Real-world"
    if name == "HeartTransplantCensorStatus":
        df = _statsmodels_df("heart")
        y = df["censors"].astype(int).to_numpy()
        X = df.drop(columns=["censors"])
        return *_clean_xy(X, y), "Real-world"
    if name == "InterestInflationHighRate":
        df = _statsmodels_df("interest_inflation")
        y = _binary_from_median(df["R"])
        X = df.drop(columns=["R"])
        X["year"] = "Y_" + X["year"].round().astype("Int64").astype(str)
        X["year"] = X["year"].astype("category")
        return *_clean_xy(X, y), "Real-world"
    if name == "LongleyHighEmployment":
        df = _statsmodels_df("longley")
        y = _binary_from_median(df["TOTEMP"])
        X = df.drop(columns=["TOTEMP"])
        return *_clean_xy(X, y), "Real-world"
    if name == "MacroData_HighInflation":
        df = _statsmodels_df("macrodata")
        y = _binary_from_median(df["infl"])
        X = df.drop(columns=["infl"])
        return *_clean_xy(X, y), "Real-world"
    if name == "ModeChoice":
        df = _statsmodels_df("modechoice")
        y = df["choice"].astype(int).to_numpy()
        X = df.drop(columns=["choice"])
        return *_clean_xy(X, y), "Real-world"
    if name == "Nile_HighVolume6Lag":
        df = _statsmodels_df("nile")
        for lag in range(1, 7):
            df[f"volume_lag{lag}"] = df["volume"].shift(lag)
        df = df.dropna().reset_index(drop=True)
        y = _binary_from_median(df["volume"])
        X = df.drop(columns=["volume"])
        return *_clean_xy(X, y), "Real-world"
    if name == "RandHIE_HighMDVisits":
        df = _statsmodels_df("randhie")
        y = (df["mdvis"] > 0).astype(int).to_numpy()
        X = df.drop(columns=["mdvis"])
        return *_clean_xy(X, y), "Real-world"
    if name == "ScotlandHighYesVote":
        df = _statsmodels_df("scotland")
        y = _binary_from_median(df["YES"])
        X = df.drop(columns=["YES"])
        return *_clean_xy(X, y), "Real-world"
    if name == "SpectorGrade":
        df = _statsmodels_df("spector")
        y = df["GRADE"].astype(int).to_numpy()
        X = df.drop(columns=["GRADE"])
        return *_clean_xy(X, y), "Real-world"
    if name == "StacklossHigh":
        df = _statsmodels_df("stackloss")
        y = _binary_from_median(df["STACKLOSS"])
        X = df.drop(columns=["STACKLOSS"])
        return *_clean_xy(X, y), "Real-world"
    if name == "Star98_HighAboveRate":
        df = _statsmodels_df("star98")
        rate = df["NABOVE"] / (df["NABOVE"] + df["NBELOW"])
        y = _binary_from_median(rate)
        X = df.drop(columns=["NABOVE", "NBELOW"])
        return *_clean_xy(X, y), "Real-world"
    if name == "StateCrimeHighViolent":
        df = _statsmodels_df("statecrime")
        y = _binary_from_median(df["violent"])
        X = df.drop(columns=["violent"])
        return *_clean_xy(X, y), "Real-world"
    if name == "Sunspots_HighActivity12Lag":
        df = _statsmodels_df("sunspots")
        for lag in range(1, 13):
            df[f"sun_lag{lag}"] = df["SUNACTIVITY"].shift(lag)
        df = df.dropna().reset_index(drop=True)
        y = _binary_from_median(df["SUNACTIVITY"])
        X = df.drop(columns=["SUNACTIVITY"])
        return *_clean_xy(X, y), "Real-world"
    raise KeyError(name)


def _series_summary(s: pd.Series) -> str:
    missing = int(s.isna().sum())
    unique = int(s.nunique(dropna=True))
    if is_numeric_dtype(s):
        numeric = pd.to_numeric(s, errors="coerce")
        non_missing = numeric.dropna()
        if non_missing.empty:
            return "all missing"
        parts = [
            f"min={non_missing.min():.4g}",
            f"p25={non_missing.quantile(0.25):.4g}",
            f"median={non_missing.median():.4g}",
            f"mean={non_missing.mean():.4g}",
            f"p75={non_missing.quantile(0.75):.4g}",
            f"max={non_missing.max():.4g}",
            f"std={non_missing.std(ddof=1):.4g}" if len(non_missing) > 1 else "std=NA",
        ]
        return "; ".join(parts)
    vc = s.astype("object").where(~s.isna(), "<missing>").value_counts(dropna=False).head(5)
    top = ", ".join(f"{k}: {int(v)}" for k, v in vc.items())
    return f"unique={unique}; top={top}; missing={missing}"


def _profile_column(col: str, s: pd.Series, n_rows: int) -> dict[str, Any]:
    missing = int(s.isna().sum())
    return {
        "column": str(col),
        "dtype": str(s.dtype),
        "missing": missing,
        "missing_pct": float(missing / max(n_rows, 1)),
        "unique": int(s.nunique(dropna=True)),
        "summary": _series_summary(s),
    }


def dataset_profile(name: str) -> dict[str, Any]:
    X, y, group = load_dataset(name)
    n_rows = int(len(y))
    rows = [_profile_column(c, X[c], n_rows) for c in X.columns]
    rows.append(_profile_column("__target__", pd.Series(y, name="__target__"), n_rows))
    return {
        "dataset": name,
        "dataset_group": group,
        "n_rows": n_rows,
        "n_features": int(X.shape[1]),
        "class_balance": float(np.mean(y)),
        "columns": rows,
    }


def build_dataset_profiles(names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in names:
        try:
            out[name] = dataset_profile(name)
        except Exception as exc:
            out[name] = {"dataset": name, "error": repr(exc), "columns": []}
    return out


def split_data(X: pd.DataFrame, y: np.ndarray, *, row_cap: int | None = -1):
    if row_cap is not None and row_cap > 0 and len(y) > row_cap:
        idx = np.arange(len(y))
        keep, _ = train_test_split(idx, train_size=row_cap, stratify=y, random_state=RANDOM_STATE)
        keep = np.sort(keep)
        X = X.iloc[keep].reset_index(drop=True)
        y = y[keep]
    test_size = 0.2 if len(y) >= 60 else 0.25
    val_size = 0.25
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_size, stratify=y_tv, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def _n_rows(X) -> int:
    return int(X.shape[0]) if hasattr(X, "shape") else len(X)


def probas(clf, X):
    try:
        return np.asarray(clf.predict_proba(X))[:, 1]
    except Exception:
        try:
            s = clf.decision_function(X)
            return 1 / (1 + np.exp(-s))
        except Exception:
            return np.full(_n_rows(X), 0.5)


def metric_row(clf, X_test, y_test, fit_seconds, valid_auc, params, complexity, budget=None):
    p = probas(clf, X_test)
    pred = (p >= 0.5).astype(int)
    return {
        "auc": safe_auc(y_test, p),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "valid_auc": None
        if valid_auc is None or math.isnan(float(valid_auc))
        else float(valid_auc),
        "complexity": None if complexity is None else float(complexity),
        "complexity_budget": None if budget is None else float(budget),
        "fit_seconds": float(fit_seconds),
        "best_params_json": json.dumps(params, sort_keys=True, default=str),
    }


def _metric_value(fn, *args, **kwargs) -> float:
    try:
        value = float(fn(*args, **kwargs))
        return value if math.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def _runner_style_metrics(y_true, proba) -> dict[str, float]:
    pred = (np.asarray(proba) >= 0.5).astype(int)
    return {
        "accuracy": _metric_value(accuracy_score, y_true, pred),
        "balanced_accuracy": _metric_value(balanced_accuracy_score, y_true, pred),
        "roc_auc": _metric_value(roc_auc_score, y_true, proba),
        "avg_precision": _metric_value(average_precision_score, y_true, proba),
        "brier": _metric_value(brier_score_loss, y_true, proba),
        "f1": _metric_value(f1_score, y_true, pred, zero_division=0),
    }


def _validated_stratified_splits(y: np.ndarray, requested: int, *, label: str) -> int:
    """Return a safe StratifiedKFold split count for binary benchmark data."""
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    if classes.size < 2:
        raise ValueError(f"{label} requires at least two target classes.")
    min_count = int(np.min(counts))
    if min_count < 2:
        raise ValueError(
            f"{label} needs at least two samples in every class; smallest class has {min_count}."
        )
    return max(2, min(int(requested), min_count))


def _prefix_grid_for_wrapped_model(grid: dict[str, list[Any]] | None) -> dict[str, list[Any]] | None:
    if not grid:
        return grid
    return {f"model__{k}": list(v) for k, v in grid.items()}


def _strip_model_prefix(params: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(params or {}).items():
        out[str(key).replace("model__", "", 1)] = value
    return out


def _finite_positive_ms(value: Any) -> float | None:
    """Return a finite positive millisecond value, or None when unavailable."""
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out) or out <= 0.0:
        return None
    return out


def _finite_or_none(value: Any) -> float | None:
    """Return a finite numeric value, or None when unavailable."""
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _model_fit_ms_from_metadata(model: Any) -> float | None:
    """Read a fitted estimator's own final-fit timing metadata when present."""
    metadata = getattr(model, "fit_metadata_", None)
    if isinstance(metadata, dict):
        for key in ("total_fit_ms", "fit_ms", "fit_time_ms"):
            val = _finite_positive_ms(metadata.get(key))
            if val is not None:
                return val
    for key in ("total_fit_ms", "fit_ms", "fit_time_ms"):
        val = _finite_positive_ms(getattr(metadata, key, None))
        if val is not None:
            return val
    return None


def _final_refit_ms_from_info(model: Any, info: dict[str, Any] | None = None) -> float:
    """Final estimator refit time for tuned runs, never tune/run elapsed time.

    The tuning helpers return already-refit estimators. Older benchmark code left
    fit_ms at 0.0 in that path. This helper populates fit_ms only when the
    final refit itself is timed or exposed by estimator metadata; it deliberately
    does not fall back to tune_ms or pair_seconds.
    """
    info = info or {}
    val = _finite_positive_ms(info.pop("_final_refit_ms", None))
    if val is not None:
        return float(val)
    val = _model_fit_ms_from_metadata(model)
    return float("nan") if val is None else float(val)


def _apply_row_cap(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    row_cap: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if row_cap is not None and row_cap > 0 and len(y) > row_cap:
        idx = np.arange(len(y))
        keep, _ = train_test_split(idx, train_size=row_cap, stratify=y, random_state=random_state)
        keep = np.sort(keep)
        return X.iloc[keep].reset_index(drop=True), np.asarray(y, dtype=int)[keep]
    return X.reset_index(drop=True), np.asarray(y, dtype=int, copy=True)


def _hugiml_base_params_from_grid(
    grid_dict: dict[str, list[Any]],
    *,
    hugiml_max_fit_seconds: float | None,
) -> dict[str, Any]:
    base_params: dict[str, Any] = {}
    for key, values in grid_dict.items():
        if len(values) == 1:
            base_params[key] = copy.deepcopy(values[0])
    base_params.setdefault("execution_mode", "production")
    base_params.setdefault("n_jobs", 1)
    if hugiml_max_fit_seconds is not None:
        base_params.setdefault("max_fit_seconds", float(hugiml_max_fit_seconds))
    return base_params


def _tune_hugiml_inner_cv(
    candidates: list[dict[str, Any]],
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    *,
    inner_splits: int,
    random_state: int,
    hugiml_max_fit_seconds: float | None,
) -> tuple[Any, dict[str, Any], float, float, dict[str, Any]]:
    inner_splits = _validated_stratified_splits(y_tr, inner_splits, label="Inner HUGIML tuning CV")
    if not hasattr(HUGIMLClassifierNative, "tune"):
        raise RuntimeError("HUGIMLClassifierNative.tune is required for inner-CV tuning.")
    grid_dict = _candidate_grid_dict(candidates)
    base_params = _hugiml_base_params_from_grid(
        grid_dict, hugiml_max_fit_seconds=hugiml_max_fit_seconds
    )
    t0 = time.perf_counter()
    result = HUGIMLClassifierNative.tune(
        X_tr,
        y_tr,
        cv=inner_splits,
        shuffle=True,
        random_state=random_state,
        scoring="roc_auc",
        param_grid=grid_dict,
        base_params=base_params,
        refit=True,
        use_fast_path=True,
    )
    tune_ms = (time.perf_counter() - t0) * 1000.0
    refit_seconds = _safe_number_or_none(getattr(result, "refit_time_", None))
    final_refit_ms = (refit_seconds * 1000.0) if refit_seconds is not None else None
    if final_refit_ms is None:
        final_refit_ms = _model_fit_ms_from_metadata(getattr(result, "best_estimator_", None))
    info = {
        "_final_refit_ms": final_refit_ms,
        "hugiml_fast_path_requested": True,
        "hugiml_fast_path_used": bool(getattr(result, "fast_path_used_", True)),
        "hugiml_tune_elapsed_seconds": getattr(result, "elapsed_seconds_", None),
        "hugiml_tune_n_splits": getattr(result, "n_splits_", inner_splits),
    }
    return (
        result.best_estimator_,
        dict(getattr(result, "best_params_", {}) or {}),
        float(getattr(result, "best_score_", float("nan"))),
        tune_ms,
        info,
    )


def _tune_pipeline_gridsearch(
    candidates: list[dict[str, Any]],
    builder,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    *,
    inner_splits: int,
    random_state: int,
) -> tuple[Any, dict[str, Any], float, float, dict[str, Any]]:
    inner_splits = _validated_stratified_splits(y_tr, inner_splits, label="Inner tuning CV")
    grid_dict = _candidate_grid_dict(candidates)
    clf0 = baseline_pipeline(builder({}))
    if not grid_dict:
        t0 = time.perf_counter()
        clf0.fit(X_tr, y_tr)
        fit_ms = (time.perf_counter() - t0) * 1000.0
        return clf0, {"static": True}, float("nan"), fit_ms, {"_final_refit_ms": fit_ms}
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        clf0,
        _prefix_grid_for_wrapped_model(grid_dict),
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score=np.nan,
    )
    t0 = time.perf_counter()
    search.fit(X_tr, y_tr)
    tune_ms = (time.perf_counter() - t0) * 1000.0
    final_refit_ms = None
    if hasattr(search, "refit_time_"):
        final_refit_ms = float(search.refit_time_) * 1000.0
    info = {"_final_refit_ms": final_refit_ms}
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_), tune_ms, info


def _tune_budgeted_pipeline_inner_cv(
    candidates: list[dict[str, Any]],
    builder,
    complexity_fn,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    *,
    budget: float,
    inner_splits: int,
    random_state: int,
) -> tuple[Any, dict[str, Any], float, float, dict[str, Any]]:
    inner_splits = _validated_stratified_splits(y_tr, inner_splits, label="Inner budgeted tuning CV")
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    scored: list[tuple[float, float, dict[str, Any]]] = []
    errors: list[str] = []
    t0 = time.perf_counter()
    for params in candidates:
        fold_scores: list[float] = []
        fold_complexities: list[float] = []
        for tr_idx, va_idx in cv.split(X_tr, y_tr):
            X_i_tr = X_tr.iloc[tr_idx] if isinstance(X_tr, pd.DataFrame) else X_tr[tr_idx]
            X_i_va = X_tr.iloc[va_idx] if isinstance(X_tr, pd.DataFrame) else X_tr[va_idx]
            y_i_tr, y_i_va = y_tr[tr_idx], y_tr[va_idx]
            try:
                clf = baseline_pipeline(builder(params))
                clf.fit(X_i_tr, y_i_tr)
                fold_scores.append(safe_auc(y_i_va, probas(clf, X_i_va)))
                comp = complexity_fn(clf) if complexity_fn else None
                if comp is not None:
                    fold_complexities.append(float(comp))
            except Exception as exc:
                errors.append(repr(exc))
                fold_scores.append(float("nan"))
        score = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
        mean_comp = float(np.nanmean(fold_complexities)) if fold_complexities else float("inf")
        scored.append((score, mean_comp, copy.deepcopy(params)))
    scored.sort(key=lambda item: ((-1.0 if math.isnan(item[0]) else item[0]), -item[1]), reverse=True)
    viable = [item for item in scored if math.isfinite(item[1]) and item[1] <= float(budget)]
    ordered = viable or scored
    last_error = None
    for score, _, params in ordered:
        try:
            clf = baseline_pipeline(builder(params))
            t_fit = time.perf_counter()
            clf.fit(X_tr, y_tr)
            final_refit_ms = (time.perf_counter() - t_fit) * 1000.0
            comp = complexity_fn(clf) if complexity_fn else None
            if viable and comp is not None and comp > budget:
                continue
            tune_ms = (time.perf_counter() - t0) * 1000.0
            info = {"budgeted_inner_cv_errors": len(errors), "_final_refit_ms": final_refit_ms}
            return clf, copy.deepcopy(params), float(score), tune_ms, info
        except Exception as exc:
            last_error = repr(exc)
    raise RuntimeError(last_error or "No budgeted candidate completed inner-CV tuning.")


def _fit_default_model(
    model: str,
    builder,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
) -> tuple[Any, float, float]:
    clf = builder({}) if is_hugiml_model(model) else baseline_pipeline(builder({}))
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    fit_ms = (time.perf_counter() - t0) * 1000.0
    return clf, fit_ms, 0.0


def _evaluate_outer_fold(
    clf,
    X_te,
    y_te,
    *,
    fold: int,
    fit_ms: float,
    tune_ms: float,
    best_params: dict[str, Any],
    best_inner_score: float,
    complexity,
    complexity_budget,
    tuned: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    p = probas(clf, X_te)
    predict_ms = (time.perf_counter() - t0) * 1000.0
    row = _runner_style_metrics(y_te, p)
    row.update(
        {
            "fold": int(fold),
            "auc": row["roc_auc"],
            "valid_auc": None
            if best_inner_score is None or math.isnan(float(best_inner_score))
            else float(best_inner_score),
            "best_inner_score": None
            if best_inner_score is None or math.isnan(float(best_inner_score))
            else float(best_inner_score),
            "best_params": json.dumps(best_params or {}, sort_keys=True, default=str),
            "best_params_json": json.dumps(best_params or {}, sort_keys=True, default=str),
            "complexity": None if complexity is None else float(complexity),
            "complexity_budget": None if complexity_budget is None else float(complexity_budget),
            "fit_ms": float(fit_ms),
            "predict_ms": float(predict_ms),
            "tune_ms": float(tune_ms),
            "fit_seconds": float(fit_ms) / 1000.0,
            "predict_seconds": float(predict_ms) / 1000.0,
            "tune_seconds": float(tune_ms) / 1000.0,
            "tuned": bool(tuned),
            "fit_time_included_in_tuning": bool(tuned and float(tune_ms) > 0.0),
            "fit_time_accounting": (
                "included_in_tuning"
                if tuned and float(tune_ms) > 0.0
                else "separate_from_tuning"
            ),
        }
    )
    if extra:
        row.update(extra)
    return row


def _mean_or_none(values: pd.Series) -> float | None:
    nums = pd.to_numeric(values, errors="coerce").dropna()
    return None if nums.empty else float(nums.mean())


def _std_or_none(values: pd.Series) -> float | None:
    nums = pd.to_numeric(values, errors="coerce").dropna()
    return None if len(nums) < 2 else float(nums.std(ddof=1))


RPTE_BACKEND_TO_PATH = {
    "sequential_default": "rpte_sequential",
    "bounded_lookahead": "rpte_lookahead",
}
RPTE_DISTRIBUTION_METRICS = (
    "rpte_inputs_passed",
    "rpte_tree_count",
    "rpte_active_tree_count",
    "rpte_leaf_count",
    "rpte_active_leaf_count",
    "rpte_direct_term_count",
    "rpte_average_leaf_path_length",
    "rpte_active_average_leaf_path_length",
)


def _fitted_rpte_estimators(estimator: Any) -> list[Any]:
    """Find fitted binary RPTE estimators through HUGIML and sklearn wrappers."""
    found: list[Any] = []
    seen: set[int] = set()
    stack = [estimator]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if (
            current.__class__.__name__ == "LeafWiseBoundedLookaheadRPTEFeatureLR"
            and hasattr(current, "fe_")
        ):
            found.append(current)
            continue
        model = getattr(current, "model_", None)
        if model is not None:
            stack.append(model)
        for attr in ("best_estimator_", "estimator_", "estimator"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                stack.append(child)
        for child in getattr(current, "estimators_", []) or []:
            stack.append(child)
        for _, child in getattr(current, "steps", []) or []:
            stack.append(child)
        named_steps = getattr(current, "named_steps", None)
        if named_steps:
            stack.extend(named_steps.values())
    return found


def _rpte_rule_rows(estimator: Any, fitted_rpte: list[Any]) -> list[dict[str, Any]]:
    for candidate in (estimator, getattr(estimator, "best_estimator_", None)):
        method = getattr(candidate, "rpte_rule_table", None)
        if callable(method):
            return [dict(row) for row in (method() or [])]
    rows: list[dict[str, Any]] = []
    for class_index, candidate in enumerate(fitted_rpte):
        method = getattr(candidate, "unified_rule_table", None)
        if not callable(method):
            continue
        for raw in method() or []:
            row = dict(raw)
            row.setdefault("class", class_index)
            rows.append(row)
    return rows


def hugiml_run_fields(estimator: Any) -> dict[str, Any]:
    """Describe the selected HUGIML route and fitted RPTE structure for one fold."""
    rpte_estimators = _fitted_rpte_estimators(estimator)
    rows = _rpte_rule_rows(estimator, rpte_estimators)
    tree_rows = [
        row
        for row in rows
        if row.get("tree_index") is not None
        and str(row.get("backend")) in RPTE_BACKEND_TO_PATH
    ]
    direct_rows = [
        row
        for row in rows
        if row.get("tree_index") is None
        and str(row.get("backend")) in {"direct_hugiml_feature", "raw_hugiml_features"}
        and row.get("source_selection_status") in {None, "not_selected_in_tree_split"}
    ]
    active_tree_rows = [
        row
        for row in tree_rows
        if abs(float(row.get("final_logistic_coefficient") or 0.0)) > 1e-12
    ]
    active_direct_rows = [
        row
        for row in direct_rows
        if abs(float(row.get("final_logistic_coefficient") or 0.0)) > 1e-12
    ]

    backend_counts: dict[str, int] = {}
    for rpte in rpte_estimators:
        fe = getattr(rpte, "fe_", None)
        if getattr(fe, "_default_fe", None) is not None:
            backend = "sequential_default"
        elif getattr(fe, "trees_", None):
            backend = "bounded_lookahead"
        else:
            continue
        path = RPTE_BACKEND_TO_PATH[backend]
        backend_counts[path] = backend_counts.get(path, 0) + 1

    if not rpte_estimators and not tree_rows and not direct_rows:
        path = "lr"
    elif len(backend_counts) == 1:
        path = next(iter(backend_counts))
    else:
        path = None
    paths = ["lr"] if path == "lr" else sorted(backend_counts)
    input_counts = [
        int(rpte.n_input_features_)
        for rpte in rpte_estimators
        if getattr(rpte, "n_input_features_", None) is not None
    ]
    tree_keys = {
        (str(row.get("class")), int(row["tree_index"]), str(row.get("backend")))
        for row in tree_rows
    }
    active_tree_keys = {
        (str(row.get("class")), int(row["tree_index"]), str(row.get("backend")))
        for row in active_tree_rows
    }
    path_lengths = [len(row.get("conditions") or []) for row in tree_rows]
    active_path_lengths = [len(row.get("conditions") or []) for row in active_tree_rows]
    selected_tree_counts = [
        int(rpte.n_estimators_)
        for rpte in rpte_estimators
        if getattr(rpte, "n_estimators_", None) is not None
    ]
    canonical = dict(
        getattr(estimator, "_downstream_lr_canonicalization_", None) or {}
    )
    return {
        "hugiml_path": path,
        "hugiml_benchmark_lr_C": _finite_or_none(
            getattr(estimator, "_benchmark_lr_C", None)
        ),
        "hugiml_lr_solver": str(getattr(estimator, "lr_solver", "auto")),
        "downstream_columns_before_reduction": canonical.get("input_columns"),
        "downstream_columns_after_reduction": canonical.get("retained_columns"),
        "downstream_vif_columns_above_threshold": canonical.get(
            "vif_columns_above_threshold"
        ),
        "downstream_vif_removed_patterns": canonical.get(
            "removed_high_vif_pattern_columns"
        ),
        "downstream_vif_removed_augmented_pairs": canonical.get(
            "removed_high_vif_augmented_pair_columns"
        ),
        "downstream_vif_maximum": canonical.get("maximum_vif"),
        "downstream_vif_median": canonical.get("median_vif"),
        "downstream_vif_analysis_seconds": canonical.get(
            "vif_analysis_seconds"
        ),
        "hugiml_paths_json": json.dumps(paths),
        "hugiml_path_counts_json": json.dumps(
            {"lr": 1} if path == "lr" else backend_counts, sort_keys=True
        ),
        "rpte_binary_estimator_count": len(rpte_estimators) or None,
        "rpte_sequential_estimator_count": backend_counts.get("rpte_sequential", 0),
        "rpte_lookahead_estimator_count": backend_counts.get("rpte_lookahead", 0),
        "rpte_inputs_passed": (
            int(input_counts[0])
            if input_counts and len(set(input_counts)) == 1
            else (int(max(input_counts)) if input_counts else None)
        ),
        "rpte_inputs_passed_per_estimator_json": json.dumps(input_counts),
        "rpte_tree_count": len(tree_keys) if path != "lr" else None,
        "rpte_active_tree_count": len(active_tree_keys) if path != "lr" else None,
        "rpte_selected_tree_counts_json": json.dumps(selected_tree_counts),
        "rpte_selected_tree_count": (
            int(max(selected_tree_counts)) if selected_tree_counts else None
        ),
        "rpte_leaf_count": len(tree_rows) if path != "lr" else None,
        "rpte_active_leaf_count": len(active_tree_rows) if path != "lr" else None,
        "rpte_direct_term_count": len(active_direct_rows) if path != "lr" else None,
        "rpte_candidate_direct_term_count": len(direct_rows) if path != "lr" else None,
        "rpte_average_leaf_path_length": (
            float(np.mean(path_lengths)) if path_lengths else (0.0 if path != "lr" else None)
        ),
        "rpte_active_average_leaf_path_length": (
            float(np.mean(active_path_lengths))
            if active_path_lengths
            else (0.0 if path != "lr" else None)
        ),
        "rpte_max_leaf_path_length": (
            int(max(path_lengths)) if path_lengths else (0 if path != "lr" else None)
        ),
    }


def _distribution_summary(values: Any) -> dict[str, Any]:
    clean: list[float] = []
    for raw in values:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            clean.append(value)
    if not clean:
        return {
            "n": 0, "mean": None, "std": None, "min": None, "q25": None,
            "median": None, "q75": None, "max": None,
        }
    array = np.asarray(clean, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def _summary_from_moments(
    n_samples: int,
    total: float,
    sum_squares: float,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    n = int(n_samples)
    if n <= 0:
        raise ValueError("n_samples must be positive")
    mean = float(total) / n
    if n == 1:
        std = 0.0
        standard_error = 0.0
        lower = mean
        upper = mean
    else:
        variance = max(0.0, (float(sum_squares) - n * mean * mean) / (n - 1))
        std = math.sqrt(variance)
        standard_error = std / math.sqrt(n)
        critical = float(student_t.ppf((1.0 + confidence_level) / 2.0, n - 1))
        margin = critical * standard_error
        lower = mean - margin
        upper = mean + margin
    return {
        "mean": mean,
        "std": std,
        "standard_error": standard_error,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": float(confidence_level),
        "n_samples": n,
        "sum": float(total),
        "sum_squares": float(sum_squares),
    }


def _series_mean_ci(
    values: pd.Series,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float | int | None]:
    nums = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if nums.empty:
        return {
            "mean": None,
            "ci_lower": None,
            "ci_upper": None,
            "confidence_level": float(confidence_level),
            "n_samples": 0,
        }
    return _summary_from_moments(
        len(nums),
        float(nums.sum()),
        float(np.square(nums.to_numpy()).sum()),
        confidence_level=confidence_level,
    )


def _timing_inclusion_marker(row: dict[str, Any]) -> bool | None:
    explicit = row.get("fit_time_included_in_tuning")
    if isinstance(explicit, bool):
        return explicit
    tune_seconds = _safe_number_or_none(row.get("tune_seconds"))
    if tune_seconds is None:
        return None
    return bool(row.get("tuned", True) and tune_seconds > 0.0)


def _aggregate_fold_rows(fold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_fold_rows: list[dict[str, Any]] = []
    for source in fold_rows:
        item = dict(source)
        marker = _timing_inclusion_marker(item)
        item["fit_time_included_in_tuning"] = marker
        item["fit_time_accounting"] = (
            "included_in_tuning"
            if marker is True
            else "separate_from_tuning"
            if marker is False
            else "unavailable"
        )
        normalized_fold_rows.append(item)
    fold_rows = normalized_fold_rows
    df = pd.DataFrame(fold_rows)
    row: dict[str, Any] = {}
    mean_cols = [
        "roc_auc",
        "auc",
        "accuracy",
        "balanced_accuracy",
        "avg_precision",
        "brier",
        "f1",
        "valid_auc",
        "best_inner_score",
        "complexity",
        "complexity_model_inspection_units",
        "complexity_model_units",
        "complexity_budget",
        "fit_ms",
        "predict_ms",
        "tune_ms",
        "fit_seconds",
        "predict_seconds",
        "tune_seconds",
        "downstream_columns_before_reduction",
        "downstream_columns_after_reduction",
        "downstream_vif_columns_above_threshold",
        "downstream_vif_removed_patterns",
        "downstream_vif_removed_augmented_pairs",
        "downstream_vif_maximum",
        "downstream_vif_median",
        "downstream_vif_analysis_seconds",
    ]
    for col in mean_cols:
        if col in df:
            row[col] = _mean_or_none(df[col])
    for col in ["roc_auc", "accuracy", "balanced_accuracy", "avg_precision", "brier", "f1"]:
        if col in df:
            row[f"std_{col}"] = _std_or_none(df[col])

    instance_n = pd.to_numeric(
        df.get("complexity_instance_inspection_units_n_samples", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    instance_sum = pd.to_numeric(
        df.get("complexity_instance_inspection_units_sum", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    instance_sum_squares = pd.to_numeric(
        df.get("complexity_instance_inspection_units_sum_squares", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    pooled_n = int(instance_n.sum())
    if pooled_n > 0:
        confidence_values = pd.to_numeric(
            df.get(
                "complexity_instance_inspection_units_confidence_level",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).dropna()
        confidence_level = (
            float(confidence_values.iloc[0]) if not confidence_values.empty else 0.95
        )
        pooled = _summary_from_moments(
            pooled_n,
            float(instance_sum.sum()),
            float(instance_sum_squares.sum()),
            confidence_level=confidence_level,
        )
        for key, value in pooled.items():
            row[f"complexity_instance_inspection_units_{key}"] = value
        min_values = pd.to_numeric(
            df.get("complexity_instance_inspection_units_min", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        max_values = pd.to_numeric(
            df.get("complexity_instance_inspection_units_max", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        row["complexity_instance_inspection_units_min"] = (
            None if min_values.empty else int(min_values.min())
        )
        row["complexity_instance_inspection_units_max"] = (
            None if max_values.empty else int(max_values.max())
        )
    if "best_params_json" in df and not df["best_params_json"].dropna().empty:
        row["best_params_json"] = str(df["best_params_json"].dropna().iloc[0])
    else:
        row["best_params_json"] = "{}"
    row["best_params_by_fold_json"] = json.dumps(
        [r.get("best_params_json", "{}") for r in fold_rows], default=str
    )
    row["fold_rows_json"] = json.dumps(_safe_jsonable(fold_rows), default=str)
    row["complexity_report_by_fold_json"] = json.dumps(
        [r.get("complexity_report_json", "{}") for r in fold_rows], default=str
    )
    markers = [_timing_inclusion_marker(item) for item in fold_rows]
    known = [marker for marker in markers if marker is not None]
    included_count = sum(marker is True for marker in known)
    separate_count = sum(marker is False for marker in known)
    row["fit_time_included_in_tuning_fold_count"] = int(included_count)
    row["fit_time_separate_from_tuning_fold_count"] = int(separate_count)
    row["fit_time_accounting_unknown_fold_count"] = int(len(markers) - len(known))
    if known and included_count == len(known) and len(known) == len(markers):
        row["fit_time_included_in_tuning"] = True
        row["fit_time_accounting"] = "included_in_tuning"
    elif known and separate_count == len(known) and len(known) == len(markers):
        row["fit_time_included_in_tuning"] = False
        row["fit_time_accounting"] = "separate_from_tuning"
    elif known:
        row["fit_time_included_in_tuning"] = None
        row["fit_time_accounting"] = "mixed"
    else:
        row["fit_time_included_in_tuning"] = None
        row["fit_time_accounting"] = "unavailable"
    path_counts: dict[str, int] = {}
    for fold_row in fold_rows:
        if fold_row.get("hugiml_path") == "lr":
            path_counts["lr"] = path_counts.get("lr", 0) + 1
        for path, field in (
            ("rpte_sequential", "rpte_sequential_estimator_count"),
            ("rpte_lookahead", "rpte_lookahead_estimator_count"),
        ):
            count = int(fold_row.get(field, 0) or 0)
            if count:
                path_counts[path] = path_counts.get(path, 0) + count
    if path_counts:
        row["hugiml_path_counts_json"] = json.dumps(path_counts, sort_keys=True)
        row["hugiml_paths_json"] = json.dumps(sorted(path_counts))
        row["hugiml_dominant_path"] = sorted(
            path_counts, key=lambda value: (-path_counts[value], value)
        )[0]
        for metric in RPTE_DISTRIBUTION_METRICS:
            summary = _distribution_summary(r.get(metric) for r in fold_rows)
            for statistic, value in summary.items():
                row[f"{metric}_{statistic}"] = value
    row["outer_folds_completed"] = int(len(fold_rows))
    row["error_count"] = int(sum(int(r.get("error_count", 0) or 0) for r in fold_rows))
    errors = [str(r.get("last_error")) for r in fold_rows if r.get("last_error")]
    row["last_error"] = errors[-1] if errors else None
    return row


def fit_select(
    candidates, builder, X_train, y_train, X_val, y_val, complexity_fn=None, budget=None
):
    best = None
    fallback = None
    errors = []
    for params in candidates:
        try:
            clf = builder(params)
            t0 = time.perf_counter()
            clf.fit(X_train, y_train)
            fit_s = time.perf_counter() - t0
            vauc = safe_auc(y_val, probas(clf, X_val))
            comp = complexity_fn(clf) if complexity_fn else None
            item = (vauc, -float(comp or 0), fit_s, clf, copy.deepcopy(params), comp)
            if fallback is None or (
                comp is not None and comp < (fallback[-1] if fallback[-1] is not None else 1e18)
            ):
                fallback = item
            if budget is not None and comp is not None and comp > budget:
                continue
            if best is None or item[:2] > best[:2]:
                best = item
        except Exception as exc:
            errors.append(repr(exc))
    return best or fallback, errors


def _candidate_grid_dict(candidates: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Reconstruct the compact sklearn-style grid used to create candidates.

    The benchmark stores model specs as an expanded ``ParameterGrid`` list.  The
    HUGIML cached tuner accepts the original dict-of-lists representation, so
    this helper rebuilds it while preserving first-seen value order.
    """
    keys: list[str] = []
    seen_keys: set[str] = set()
    for cand in candidates:
        for key in cand:
            if key not in seen_keys:
                seen_keys.add(key)
                keys.append(key)
    grid: dict[str, list[Any]] = {}
    for key in keys:
        vals: list[Any] = []
        seen_vals: set[str] = set()
        for cand in candidates:
            if key not in cand:
                continue
            val = cand[key]
            marker = repr(val)
            if marker not in seen_vals:
                seen_vals.add(marker)
                vals.append(copy.deepcopy(val))
        grid[key] = vals
    return grid


def fit_select_hugiml_fast(
    candidates, builder, X_train, y_train, X_val, y_val, complexity_fn=None, budget=None
):
    """Select HUGIML params through ``fast_grid_tune`` when the grid is eligible.

    This keeps benchmark semantics aligned with the ordinary validation split:
    select on train/validation, then let ``run_pair`` perform the same final
    train+validation refit used for all models.  If the grid is not eligible for
    the exact cached path, callers can fall back to the ordinary ``fit_select``.
    """
    errors: list[str] = []
    info: dict[str, Any] = {"hugiml_fast_tune_used": False}
    try:
        if budget is not None:
            raise RuntimeError("HUGIML fast tune path is not used with a complexity budget.")
        if not hasattr(HUGIMLClassifierNative, "fast_grid_tune"):
            raise RuntimeError("Installed HUGIML does not expose fast_grid_tune.")

        param_grid = _candidate_grid_dict(list(candidates))
        base_params: dict[str, Any] = {}
        for key, vals in param_grid.items():
            if len(vals) == 1:
                base_params[key] = copy.deepcopy(vals[0])
        base_params.setdefault("execution_mode", "production")
        base_params.setdefault("n_jobs", 1)

        t0 = time.perf_counter()
        result = HUGIMLClassifierNative.fast_grid_tune(
            X_train,
            y_train,
            X_val,
            y_val,
            param_grid=param_grid,
            base_params=base_params,
            scoring="roc_auc",
            refit_full=False,
            return_results=True,
        )
        elapsed = time.perf_counter() - t0

        best_model = result["best_model"]
        best_params = dict(base_params)
        best_params.update(result.get("best_params") or {})
        # ``builder`` owns benchmark-level defaults such as n_jobs and possible
        # runtime caps.  Round-trip the selected params through it so the final
        # refit receives the same normalized parameter dictionary as the ordinary
        # path.
        normalized = builder(best_params).get_params(deep=False)
        for key, val in best_params.items():
            normalized[key] = val
        comp = complexity_fn(best_model) if complexity_fn else None
        selected = (
            float(result["best_score"]),
            -float(comp or 0),
            float(elapsed),
            best_model,
            copy.deepcopy(best_params),
            comp,
        )
        info.update(
            {
                "hugiml_fast_tune_used": True,
                "hugiml_fast_tune_method": result.get("method"),
                "hugiml_fast_tune_elapsed_seconds": float(result.get("elapsed_seconds", elapsed)),
                "hugiml_fast_tune_cache_fit_seconds_json": json.dumps(
                    result.get("cache_fit_seconds_by_G_L_topK", {}),
                    sort_keys=True,
                    default=str,
                ),
                "hugiml_fast_tune_augmented_pair_cache_stats_json": json.dumps(
                    result.get("native_augmented_pair_cache_stats"),
                    sort_keys=True,
                    default=str,
                ),
                "hugiml_fast_tune_validation_cache_entries": result.get(
                    "validation_cache_entries"
                ),
                "hugiml_fast_tune_transaction_cache_entries": result.get(
                    "transaction_cache_entries"
                ),
                "hugiml_fast_tune_template_prepare_seconds_json": json.dumps(
                    result.get("template_prepare_seconds", {}),
                    sort_keys=True,
                    default=str,
                ),
                "hugiml_fast_tune_validation_prepare_seconds_json": json.dumps(
                    result.get("validation_prepare_seconds", {}),
                    sort_keys=True,
                    default=str,
                ),
                "hugiml_fast_tune_fingerprint_seconds_json": json.dumps(
                    result.get("fingerprint_seconds", {}),
                    sort_keys=True,
                    default=str,
                ),
                "hugiml_fast_tune_downstream_fit_seconds": float(
                    result.get("downstream_fit_seconds", 0.0)
                ),
                "hugiml_fast_tune_downstream_score_seconds": float(
                    result.get("downstream_score_seconds", 0.0)
                ),
                "hugiml_fast_tune_downstream_template_count": int(
                    result.get("downstream_template_count", 0)
                ),
                "hugiml_fast_tune_validation_structure_reuses": int(
                    result.get("validation_structure_reuses", 0)
                ),
                "hugiml_fast_tune_original_feature_cache_stats_json": json.dumps(
                    result.get("original_feature_cache_stats"),
                    sort_keys=True,
                    default=str,
                ),
                "hugiml_fast_tune_equivalent_downstream_fit_reuses": int(
                    result.get("equivalent_downstream_fit_reuses", 0)
                ),
            }
        )
        return selected, errors, info
    except Exception as exc:
        errors.append(f"fast_grid_tune fallback: {type(exc).__name__}: {exc}")
        selected, ordinary_errors = fit_select(
            candidates, builder, X_train, y_train, X_val, y_val, complexity_fn, budget
        )
        errors.extend(ordinary_errors)
        return selected, errors, info


def fit_final(builder, params, X_fit, y_fit):
    clf = builder(params)
    t0 = time.perf_counter()
    clf.fit(X_fit, y_fit)
    return clf, time.perf_counter() - t0


def _hugiml_grid_for_scenario(hugiml_scenario: str | None) -> tuple[str, list[dict[str, Any]]]:
    scenario = hugiml_scenario or DEFAULT_DASHBOARD_HUGIML_SCENARIO
    if scenario not in HUGIML_SCENARIOS:
        raise ValueError(
            f"Unknown HUGIML scenario {scenario!r}. Allowed: {list(HUGIML_SCENARIOS)}"
        )
    spec = HUGIML_SCENARIOS[scenario]
    grid_dict = get_hugiml_grid(str(spec["grid_name"]))
    for key, values in dict(spec.get("overrides", {})).items():
        grid_dict[key] = list(values)
    return str(spec["grid_name"]), list(ParameterGrid(grid_dict))


HUGIML_MODEL_GRIDS = {
    "HUGIML": "performance_ho",
}

HUGIML_INTERACTION_GRIDS = {
    "HUGIML": "interpretability_ho",
}


def is_hugiml_model(model: str) -> bool:
    return str(model) in HUGIML_MODEL_GRIDS


def _hugiml_grid_for_model(
    model: str, hugiml_scenario: str | None
) -> tuple[str, list[dict[str, Any]]]:
    scenario = hugiml_scenario or DEFAULT_DASHBOARD_HUGIML_SCENARIO
    if scenario not in HUGIML_SCENARIOS:
        raise ValueError(
            f"Unknown HUGIML scenario {scenario!r}. Allowed: {list(HUGIML_SCENARIOS)}"
        )
    spec = HUGIML_SCENARIOS[scenario]
    configured_grid = str(spec["grid_name"])
    if configured_grid == "performance_ho":
        grid_name = HUGIML_MODEL_GRIDS[model]
    elif configured_grid == "interpretability_ho":
        grid_name = HUGIML_INTERACTION_GRIDS[model]
    else:
        grid_name = configured_grid
    grid_dict = get_hugiml_grid(grid_name)
    for key, values in dict(spec.get("overrides", {})).items():
        grid_dict[key] = list(values)
    return grid_name, list(ParameterGrid(grid_dict))


def _hugiml_scenario_label(scenario: str | None) -> str | None:
    if scenario is None:
        return None
    return str(HUGIML_SCENARIOS.get(scenario, {}).get("label", scenario))


def _model_inspection_complexity(model: Any) -> int | float | None:
    """Return the complete-model inspection measure used by this benchmark."""
    return get_complexity(model, "model inspection units")


class AdaptiveSolverLogisticRegression(LogisticRegression):
    """Use liblinear for binary targets and saga for multiclass targets."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None):
        self.solver = "liblinear" if np.unique(np.asarray(y)).size <= 2 else "saga"
        return super().fit(X, y, sample_weight=sample_weight)

def get_model_spec(
    model: str,
    *,
    hugiml_scenario: str | None = None,
    hugiml_max_fit_seconds: float | None = None,
):
    if is_hugiml_model(model):
        _, grid = _hugiml_grid_for_model(model, hugiml_scenario)

        def builder(params):
            pp = dict(params)
            lr_C = float(pp.pop("lr_C", 1.0))
            pp.setdefault("execution_mode", "production")
            if hugiml_max_fit_seconds is not None:
                pp.setdefault("max_fit_seconds", float(hugiml_max_fit_seconds))
            pp.setdefault("n_jobs", 1)
            estimator = HUGIMLClassifierNative(**pp)
            estimator._benchmark_lr_C = lr_C
            return estimator

        return grid, builder, _model_inspection_complexity, None

    xgb_grid = list(ParameterGrid(get_baseline_grid("XGBoost")))
    lgb_grid = list(ParameterGrid(get_baseline_grid("LightGBM")))
    rf_grid = list(ParameterGrid(get_baseline_grid("RandomForest")))
    lr_grid = list(ParameterGrid(get_baseline_grid("LogisticRegression") or {})) or [{}]
    ebm_grid = list(ParameterGrid(get_baseline_grid("EBM") or {})) or [{}]
    raw_rulefit_grid = list(ParameterGrid(get_baseline_grid("RuleFit") or {})) or [{}]
    rulefit_grid = [
        {f"estimator__{key}": value for key, value in candidate.items()}
        for candidate in raw_rulefit_grid
    ]
    xgb_budget_grid = list(ParameterGrid(get_budgeted_baseline_grid("XGBoost")))
    lgb_budget_grid = list(ParameterGrid(get_budgeted_baseline_grid("LightGBM")))
    rf_budget_grid = list(ParameterGrid(get_budgeted_baseline_grid("RandomForest")))

    def xgb_builder(params):
        return XGBClassifier(**baseline_constant_parameters("XGBoost"), **params)

    def lgb_builder(params):
        return LGBMClassifier(**baseline_constant_parameters("LightGBM"), **params)

    def rf_builder(params):
        return RandomForestClassifier(**baseline_constant_parameters("RandomForest"), **params)

    def lr_builder(params):
        pp = baseline_constant_parameters("LogisticRegression")
        pp.update(params)
        return AdaptiveSolverLogisticRegression(**pp)

    def ebm_builder(params):
        if ExplainableBoostingClassifier is None:
            raise ImportError("interpret.glassbox.ExplainableBoostingClassifier is required for EBM")
        pp = baseline_constant_parameters("EBM")
        pp.update(params)
        return ExplainableBoostingClassifier(**pp)

    def rulefit_builder(params):
        if RuleFitClassifier is None:
            raise ImportError("imodels.RuleFitClassifier is required for RuleFit")
        pp = baseline_constant_parameters("RuleFit")
        pp.update(
            {
                str(key).replace("estimator__", "", 1): value
                for key, value in dict(params or {}).items()
            }
        )
        # Keep max_rules authoritative. imodels ignores max_rules when alpha
        # is explicitly numeric, so alpha remains None for every candidate.
        pp["alpha"] = None
        return OneVsRestClassifier(RuleFitClassifier(**pp), n_jobs=1)

    specs = {
        "XGB standard": (xgb_grid, xgb_builder, _model_inspection_complexity, None),
        "LightGBM standard": (lgb_grid, lgb_builder, _model_inspection_complexity, None),
        "RandomForest standard": (rf_grid, rf_builder, _model_inspection_complexity, None),
        "XGB complexity-budgeted": (xgb_budget_grid, xgb_builder, _model_inspection_complexity, BUDGET),
        "LightGBM complexity-budgeted": (lgb_budget_grid, lgb_builder, _model_inspection_complexity, BUDGET),
        "RandomForest complexity-budgeted": (rf_budget_grid, rf_builder, _model_inspection_complexity, BUDGET),
        "Logistic Regression": (lr_grid, lr_builder, _model_inspection_complexity, None),
        "EBM": (ebm_grid, ebm_builder, _model_inspection_complexity, None),
        "RuleFit": (rulefit_grid, rulefit_builder, _model_inspection_complexity, None),
    }
    return specs[model]


def run_pair(
    dataset: str,
    model: str,
    *,
    hugiml_scenario: str | None = None,
    row_cap: int | None = -1,
    hugiml_max_fit_seconds: float | None = None,
    n_splits: int = 5,
    inner_splits: int = 3,
    tune: bool = True,
    random_state: int | None = None,
    fold_checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    random_state = RANDOM_STATE if random_state is None else int(random_state)
    X, y, group = load_dataset(dataset)
    X, y = _apply_row_cap(X, y, row_cap=row_cap, random_state=random_state)
    raw_features = int(X.shape[1])
    native_categorical_features = _categorical_columns(X)

    if dataset == "InterestInflationHighRate" and is_hugiml_model(model):
        if "year" not in X.columns or not _is_categorical_like(X["year"]):
            raise RuntimeError(
                "InterestInflationHighRate expected HUGIML to receive native categorical year. "
                f"Saw columns={list(X.columns)} dtypes={X.dtypes.astype(str).to_dict()}"
            )

    grid, builder, complexity_fn, budget = get_model_spec(
        model,
        hugiml_scenario=hugiml_scenario,
        hugiml_max_fit_seconds=hugiml_max_fit_seconds,
    )
    n_splits = _validated_stratified_splits(y, n_splits, label="Outer benchmark CV")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows_by_fold: dict[int, dict[str, Any]] = {}
    feature_counts_by_fold: dict[int, int] = {}
    fold_checkpoint_path: Path | None = None
    if fold_checkpoint_dir is not None:
        fold_checkpoint_dir = Path(fold_checkpoint_dir)
        fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "dataset": dataset, "model": model, "hugiml_scenario": hugiml_scenario,
            "row_cap": row_cap, "n_splits": n_splits, "inner_splits": inner_splits,
            "tune": tune, "random_state": random_state,
            "dataset_feature_policy": DATASET_FEATURE_POLICY,
        }
        token = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:20]
        fold_checkpoint_path = fold_checkpoint_dir / f"{token}.json"
        if fold_checkpoint_path.exists():
            try:
                saved = json.loads(fold_checkpoint_path.read_text())
                if saved.get("identity") == identity:
                    for row in saved.get("fold_rows", []):
                        if row.get("fold") is not None:
                            rows_by_fold[int(row["fold"])] = dict(row)
                    raw_counts = saved.get("model_feature_counts", {})
                    if isinstance(raw_counts, dict):
                        feature_counts_by_fold = {
                            int(key): int(value) for key, value in raw_counts.items()
                        }
                    elif isinstance(raw_counts, list):
                        successful_folds = sorted(
                            fold_id
                            for fold_id, row in rows_by_fold.items()
                            if int(row.get("error_count", 0) or 0) == 0
                            and row.get("roc_auc") is not None
                        )
                        feature_counts_by_fold = {
                            fold_id: int(value)
                            for fold_id, value in zip(successful_folds, raw_counts)
                        }
            except Exception:
                rows_by_fold = {}
                feature_counts_by_fold = {}
    successful_fold_ids = {
        fold_id
        for fold_id, row in rows_by_fold.items()
        if (
            row.get("status") == "ok"
            or (
                row.get("status") is None
                and int(row.get("error_count", 0) or 0) == 0
                and row.get("roc_auc") is not None
            )
        )
    }

    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
        if fold_idx in successful_fold_ids:
            continue
        X_train_native = _force_writable_frame(X.iloc[tr_idx].reset_index(drop=True))
        X_test_native = _force_writable_frame(X.iloc[te_idx].reset_index(drop=True))
        y_train = np.asarray(y[tr_idx], dtype=int, copy=True)
        y_test = np.asarray(y[te_idx], dtype=int, copy=True)
        params: dict[str, Any] = {}
        best_inner_score = float("nan")
        tune_ms = 0.0
        fit_ms = 0.0
        selection_info: dict[str, Any] = {}
        error_count = 0
        last_error = None

        try:
            if is_hugiml_model(model):
                X_train_model, X_test_model = X_train_native, X_test_native
                if tune:
                    clf, params, best_inner_score, tune_ms, selection_info = _tune_hugiml_inner_cv(
                        grid,
                        X_train_model,
                        y_train,
                        inner_splits=inner_splits,
                        random_state=random_state,
                        hugiml_max_fit_seconds=hugiml_max_fit_seconds,
                    )
                    fit_ms = _final_refit_ms_from_info(clf, selection_info)
                else:
                    clf, fit_ms, tune_ms = _fit_default_model(model, builder, X_train_model, y_train)
                    selection_info = {"hugiml_fast_path_requested": False, "hugiml_fast_path_used": False}
            else:
                X_train_model, X_test_model = X_train_native, X_test_native
                if tune:
                    if budget is None:
                        clf, params, best_inner_score, tune_ms, selection_info = _tune_pipeline_gridsearch(
                            grid,
                            builder,
                            X_train_model,
                            y_train,
                            inner_splits=inner_splits,
                            random_state=random_state,
                        )
                        fit_ms = _final_refit_ms_from_info(clf, selection_info)
                    else:
                        clf, params, best_inner_score, tune_ms, selection_info = (
                            _tune_budgeted_pipeline_inner_cv(
                                grid,
                                builder,
                                complexity_fn,
                                X_train_model,
                                y_train,
                                budget=float(budget),
                                inner_splits=inner_splits,
                                random_state=random_state,
                            )
                        )
                        fit_ms = _final_refit_ms_from_info(clf, selection_info)
                else:
                    clf, fit_ms, tune_ms = _fit_default_model(model, builder, X_train_model, y_train)

            selection_info = dict(selection_info or {})
            if is_hugiml_model(model):
                selection_info.update(hugiml_run_fields(clf))
            complexity_report = get_complexity_report(clf, X=X_test_model)
            model_units = (complexity_report or {}).get("model_units", {}).get("value")
            model_inspection_units = (complexity_report or {}).get(
                "model_inspection_units", {}
            ).get("value")
            instance_summary = (complexity_report or {}).get(
                "instance_inspection_units", {}
            )
            complexity = (
                float(model_inspection_units)
                if model_inspection_units is not None
                else (complexity_fn(clf) if complexity_fn else None)
            )
            selection_info["complexity_model_units"] = (
                None if model_units is None else float(model_units)
            )
            selection_info["complexity_model_inspection_units"] = (
                None if model_inspection_units is None else float(model_inspection_units)
            )
            for key in (
                "mean",
                "std",
                "standard_error",
                "ci_lower",
                "ci_upper",
                "confidence_level",
                "n_samples",
                "sum",
                "sum_squares",
                "min",
                "max",
            ):
                selection_info[f"complexity_instance_inspection_units_{key}"] = (
                    instance_summary.get(key) if instance_summary.get("available") else None
                )
            selection_info["complexity_report_json"] = json.dumps(
                complexity_report or {}, sort_keys=True, default=str
            )
            feature_count = _model_feature_count(clf, raw_features)
            effective_budget = budget
            fold_row = _evaluate_outer_fold(
                clf,
                X_test_model,
                y_test,
                fold=fold_idx,
                fit_ms=fit_ms,
                tune_ms=tune_ms,
                best_params=params,
                best_inner_score=best_inner_score,
                complexity=complexity,
                complexity_budget=effective_budget,
                tuned=tune,
                extra=selection_info,
            )
        except Exception as exc:
            error_count = 1
            last_error = f"{type(exc).__name__}: {exc}"
            fold_row = {
                "fold": int(fold_idx),
                "accuracy": None,
                "balanced_accuracy": None,
                "roc_auc": None,
                "auc": None,
                "avg_precision": None,
                "brier": None,
                "f1": None,
                "valid_auc": None,
                "best_inner_score": None,
                "complexity": None,
                "complexity_model_inspection_units": None,
                "complexity_model_units": None,
                "complexity_instance_inspection_units_mean": None,
                "complexity_instance_inspection_units_std": None,
                "complexity_instance_inspection_units_standard_error": None,
                "complexity_instance_inspection_units_ci_lower": None,
                "complexity_instance_inspection_units_ci_upper": None,
                "complexity_instance_inspection_units_confidence_level": None,
                "complexity_instance_inspection_units_n_samples": None,
                "complexity_instance_inspection_units_sum": None,
                "complexity_instance_inspection_units_sum_squares": None,
                "complexity_instance_inspection_units_min": None,
                "complexity_instance_inspection_units_max": None,
                "complexity_report_json": "{}",
                "hugiml_path": None,
                "hugiml_paths_json": "[]",
                "hugiml_path_counts_json": "{}",
                "rpte_binary_estimator_count": None,
                "rpte_sequential_estimator_count": 0,
                "rpte_lookahead_estimator_count": 0,
                "rpte_inputs_passed": None,
                "rpte_inputs_passed_per_estimator_json": "[]",
                "rpte_tree_count": None,
                "rpte_active_tree_count": None,
                "rpte_selected_tree_counts_json": "[]",
                "rpte_selected_tree_count": None,
                "rpte_leaf_count": None,
                "rpte_active_leaf_count": None,
                "rpte_direct_term_count": None,
                "rpte_candidate_direct_term_count": None,
                "rpte_average_leaf_path_length": None,
                "rpte_active_average_leaf_path_length": None,
                "rpte_max_leaf_path_length": None,
                "complexity_budget": None if budget is None else float(budget),
                "fit_ms": None,
                "predict_ms": None,
                "tune_ms": None,
                "fit_seconds": None,
                "predict_seconds": None,
                "tune_seconds": None,
                "best_params": "{}",
                "best_params_json": "{}",
                "tuned": bool(tune),
            }

        fold_row.update(
            {
                "dataset": dataset,
                "model": model,
                "hugiml_scenario": hugiml_scenario if is_hugiml_model(model) else None,
                "outer_n_splits": int(n_splits),
                "inner_n_splits": int(inner_splits) if tune else None,
                "random_state": int(random_state),
                "status": "ok" if error_count == 0 else "error",
                "error_count": error_count,
                "last_error": last_error,
            }
        )
        rows_by_fold[int(fold_idx)] = fold_row
        if error_count == 0:
            feature_counts_by_fold[int(fold_idx)] = int(feature_count)
        else:
            feature_counts_by_fold.pop(int(fold_idx), None)
        fold_rows = [rows_by_fold[key] for key in sorted(rows_by_fold)]
        if fold_checkpoint_path is not None:
            checkpoint_payload = {
                "identity": identity,
                "fold_rows": fold_rows,
                "model_feature_counts": {
                    str(key): feature_counts_by_fold[key]
                    for key in sorted(feature_counts_by_fold)
                },
            }
            tmp = fold_checkpoint_path.with_suffix(fold_checkpoint_path.suffix + ".tmp")
            tmp.write_text(json.dumps(checkpoint_payload, indent=2, default=_json_default))
            tmp.replace(fold_checkpoint_path)

    fold_rows = [rows_by_fold[key] for key in sorted(rows_by_fold)]
    row = _aggregate_fold_rows(fold_rows)
    model_feature_counts = [
        feature_counts_by_fold[key] for key in sorted(feature_counts_by_fold)
    ]
    model_features = int(round(float(np.nanmean(model_feature_counts)))) if model_feature_counts else raw_features
    preprocessing_policy = (
        "HUGIML native categorical"
        if is_hugiml_model(model)
        else "runner-compatible Pipeline preprocessing fitted inside CV"
    )
    protocol = "outer_cv_inner_cv_tuning" if tune else "outer_cv_no_inner_tuning"
    row.update(
        {
            "dataset": dataset,
            "dataset_group": group,
            "model": model,
            "hugiml_scenario": hugiml_scenario if is_hugiml_model(model) else None,
            "hugiml_scenario_label": _hugiml_scenario_label(hugiml_scenario) if is_hugiml_model(model) else None,
            "hugiml_grid_name": _hugiml_grid_for_model(model, hugiml_scenario)[0] if is_hugiml_model(model) else None,
            "raw_features": raw_features,
            "model_features": model_features,
            "categorical_features": native_categorical_features,
            "n_rows": int(len(y)),
            "class_balance": float(np.mean(y)),
            "preprocessing_policy": preprocessing_policy,
            "evaluation_protocol": protocol,
            "outer_n_splits": int(n_splits),
            "inner_n_splits": int(inner_splits) if tune else None,
            "random_state": int(random_state),
            "scoring": "roc_auc",
            "tuned": bool(tune),
        }
    )
    if is_hugiml_model(model):
        row.setdefault("hugiml_fast_path_requested", bool(tune))
        if tune and "hugiml_fast_path_used" not in row:
            used = [r.get("hugiml_fast_path_used") for r in fold_rows if r.get("hugiml_fast_path_used") is not None]
            row["hugiml_fast_path_used"] = bool(used and all(bool(v) for v in used))
    return row

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if math.isnan(float(obj)):
            return None
        return float(obj)
    if pd.isna(obj):
        return None
    return str(obj)


def _safe_jsonable(obj):
    """Return an object containing only standards-compliant JSON values."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, dict):
        return {str(k): _safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_safe_jsonable(v) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def grid_snapshot() -> dict[str, Any]:
    return {
        "hugiml_dashboard_scenarios": {
            key: {
                "label": spec["label"],
                "description": spec["description"],
                "model_grids": {
                    model: {
                        "grid_name": _hugiml_grid_for_model(model, key)[0],
                        "grid": get_hugiml_grid(_hugiml_grid_for_model(model, key)[0]),
                    }
                    for model in ("HUGIML",)
                },
            }
            for key, spec in HUGIML_SCENARIOS.items()
        },
        "default_hugiml_dashboard_scenario": DEFAULT_DASHBOARD_HUGIML_SCENARIO,
        "baseline_grids": {
            "XGBoost": get_baseline_grid("XGBoost"),
            "LightGBM": get_baseline_grid("LightGBM"),
            "RandomForest": get_baseline_grid("RandomForest"),
            "LogisticRegression": get_baseline_grid("LogisticRegression"),
            "EBM": get_baseline_grid("EBM"),
            "RuleFit": get_baseline_grid("RuleFit"),
        },
        "baseline_budgeted_grids": {
            "XGBoost": get_budgeted_baseline_grid("XGBoost"),
            "LightGBM": get_budgeted_baseline_grid("LightGBM"),
            "RandomForest": get_budgeted_baseline_grid("RandomForest"),
        },
        "baseline_constant_parameters": {
            model: baseline_constant_parameters(model)
            for model in ["XGBoost", "LightGBM", "RandomForest", "LogisticRegression", "EBM", "RuleFit"]
        },
        "execution_mode_base_setting": "production",
    }


def _methodology_display_value(value: Any) -> str:
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    text = str(value)
    if "LeafWiseBoundedLookaheadRPTEFeatureLR" in text:
        params: dict[str, Any] = {}
        try:
            params = value.get_params(deep=True)
        except Exception:
            params = {}
        lookahead = params.get(
            "estimator__enable_lookahead",
            params.get("enable_lookahead", "adaptive"),
        )
        backend = "sequential; lookahead inactive" if lookahead is False else "adaptive lookahead"
        return f"OneVsRestClassifier(RPTE; leaf_config=3xD, depth=4, {backend})"
    return text


def _methodology_parameter_rows(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    rows = []
    automatic_lr_C = float((grid.get("lr_C") or [1.0])[0])
    for parameter, values in grid.items():
        display_values = [_methodology_display_value(value) for value in values]
        if parameter == "base_estimator":
            normalized = []
            for value in values:
                text = str(value)
                if value is None:
                    normalized.append(
                        "Automatic L2 logistic regression "
                        f"(binary: lbfgs; multiclass: lbfgs-OvR; C={automatic_lr_C:g})"
                    )
                elif isinstance(value, LogisticRegression) or text.startswith("LogisticRegression("):
                    normalized.append(
                        "Logistic regression (L1 penalty, liblinear solver, C=0.5)"
                    )
                else:
                    normalized.append(_methodology_display_value(value))
            display_values = normalized
        rows.append({"parameter": str(parameter), "values": display_values})
    return rows


def _methodology_candidate_count(grid: dict[str, list[Any]]) -> int:
    count = 1
    for values in grid.values():
        count *= max(1, len(values))
    return int(count)


def methodology_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(payload.get("metadata", {}))
    snapshot = dict(metadata.get("grid_snapshot", {}) or {})
    scenario_snapshot = dict(snapshot.get("hugiml_dashboard_scenarios", {}) or {})
    standard_grids = dict(snapshot.get("baseline_grids", {}) or {})
    budgeted_grids = dict(snapshot.get("baseline_budgeted_grids", {}) or {})
    constant_parameters = dict(snapshot.get("baseline_constant_parameters", {}) or {})

    for name in ["XGBoost", "LightGBM", "RandomForest", "LogisticRegression", "EBM", "RuleFit"]:
        standard_grids.setdefault(name, get_baseline_grid(name))
        constant_parameters.setdefault(name, baseline_constant_parameters(name))
    for name in ["XGBoost", "LightGBM", "RandomForest"]:
        budgeted_grids.setdefault(name, get_budgeted_baseline_grid(name))

    hugiml_models = []
    for scenario_id, spec in HUGIML_SCENARIOS.items():
        stored = dict(scenario_snapshot.get(scenario_id, {}) or {})
        grid = dict(stored.get("grid", {}) or {})
        if not grid:
            grid = get_hugiml_grid(str(spec["grid_name"]))
            grid.update(dict(spec.get("overrides", {})))
        automatic_lr_C = float((grid.get("lr_C") or [1.0])[0])
        notes = [
            "execution_mode=production",
            "n_jobs=1",
            "Automatic linear base estimator: logistic regression with L2 penalty, "
            f"binary lbfgs / multiclass lbfgs-OvR, C={automatic_lr_C:g}, "
            "random_state=0, and max_iter=300 for every binary fit.",
        ]
        if scenario_id == "augmented_pair":
            notes.extend(
                [
                    "Inner cross-validation selects between the stated logistic-regression base estimator and one-vs-rest RPTE.",
                    "Numeric 0/1 columns remain numeric and eligible for augmented-pair transforms.",
                    "RPTE uses leaf indicators together with downstream inputs not selected in accepted tree splits.",
                    "RPTE lookahead is adaptive for this path.",
                ]
            )
        else:
            notes.extend(
                [
                    "Inner cross-validation selects between the stated logistic-regression base estimator and one-vs-rest RPTE.",
                    "Numeric 0/1 indicators are treated categorically for the pattern-mining surface.",
                    "Interaction relaxation is performed by the pattern miner; augmented pairs are disabled.",
                    "RPTE uses the sequential backend with lookahead inactive for this path.",
                ]
            )
        hugiml_models.append(
            {
                "model": f'HUGIML — {spec["label"]}',
                "grid_name": str(spec["grid_name"]),
                "candidate_count": _methodology_candidate_count(grid),
                "parameters": _methodology_parameter_rows(grid),
                "constant_settings": notes,
                "complexity": (
                    "Model inspection units represent the complete fitted HUGIML model that a reviewer "
                    "must inspect. Linear branches count active source contributions; RPTE branches count "
                    "conditions across active terminal paths plus direct terms. Intercepts are excluded, "
                    "and fitted numeric components are active when their absolute value exceeds 1e-12."
                ),
            }
        )

    # Present each regularization variant separately for both HUGIML scenarios.
    hugiml_models = []
    for scenario_id, spec in HUGIML_SCENARIOS.items():
        stored = dict(scenario_snapshot.get(scenario_id, {}) or {})
        stored_model_grids = dict(stored.get("model_grids", {}) or {})
        for model in ("HUGIML",):
            grid_name, _ = _hugiml_grid_for_model(model, scenario_id)
            stored_model = dict(stored_model_grids.get(model, {}) or {})
            grid = dict(stored_model.get("grid", {}) or {})
            if not grid:
                grid = get_hugiml_grid(grid_name)
                grid.update(dict(spec.get("overrides", {})))
            lr_C = float((grid.get("lr_C") or [0.5 if model == "HUGIML" else 0.1])[0])
            solver_note = (
                "Direct logistic branch: L1 penalty with binary liblinear "
                f"(max_iter=300) or multiclass SAGA (max_iter=500), C={lr_C:g}. "
                "RPTE downstream fits use L1/liblinear with max_iter=300."
            )
            notes = [
                "execution_mode=production",
                "n_jobs=1",
                solver_note,
                "Downstream redundancy uses one training-only VIF analysis; generated terms with VIF greater than 5 are reduced when an earlier preferred term explains at least 80 percent of their variance, while originals are preserved and patterns precede augmented pairs.",
            ]
            if scenario_id == "augmented_pair":
                notes.extend(
                    [
                        "Inner cross-validation selects between the stated logistic-regression base estimator and one-vs-rest RPTE.",
                        "Numeric 0/1 columns remain numeric and eligible for augmented-pair transforms.",
                        "RPTE uses leaf indicators together with downstream inputs not selected in accepted tree splits.",
                        "RPTE lookahead is adaptive for this path.",
                    ]
                )
            else:
                notes.extend(
                    [
                        "Inner cross-validation selects between the stated logistic-regression base estimator and one-vs-rest RPTE.",
                        "Numeric 0/1 indicators are treated categorically for the pattern-mining surface.",
                        "Interaction relaxation is performed by the pattern miner; augmented pairs are disabled.",
                        "RPTE uses the sequential backend with lookahead inactive for this path.",
                    ]
                )
            hugiml_models.append(
                {
                    "model": f'{model} - {spec["label"]}',
                    "grid_name": grid_name,
                    "candidate_count": _methodology_candidate_count(grid),
                    "parameters": _methodology_parameter_rows(grid),
                    "constant_settings": notes,
                    "complexity": (
                        "Model inspection units represent the complete fitted HUGIML model that a reviewer "
                        "must inspect. Linear branches count active source contributions; RPTE branches count "
                        "conditions across active terminal paths plus direct terms. Intercepts are excluded, "
                        "and fitted numeric components are active when their absolute value exceeds 1e-12."
                    ),
                }
            )

    baseline_models = []
    # Methodology-card display order only: the two HUGIML variants are
    # followed by the three standard ensembles; the second methodology row
    # contains the corresponding budgeted ensembles and interpretable models.
    baseline_labels = [
        ("XGB standard", "XGBoost", False),
        ("LightGBM standard", "LightGBM", False),
        ("RandomForest standard", "RandomForest", False),
        ("XGB complexity-budgeted", "XGBoost", True),
        ("LightGBM complexity-budgeted", "LightGBM", True),
        ("RandomForest complexity-budgeted", "RandomForest", True),
        ("Logistic Regression", "LogisticRegression", False),
        ("EBM", "EBM", False),
        ("RuleFit", "RuleFit", False),
    ]
    complexity_text = {
        "XGBoost": (
            "Model inspection units sum the conditions across complete root-to-leaf paths "
            "with active terminal outputs."
        ),
        "LightGBM": (
            "Model inspection units sum the conditions across complete root-to-leaf paths "
            "with active terminal outputs."
        ),
        "RandomForest": (
            "Model inspection units sum the conditions across every complete root-to-leaf "
            "path in the fitted forest."
        ),
        "LogisticRegression": (
            "Model inspection units count nonzero coefficients across fitted class models."
        ),
        "EBM": (
            "Model inspection units count finite nonzero term-score cells expanded by "
            "source-feature arity, so pairwise interaction cells contribute two units."
        ),
        "RuleFit": (
            "Model inspection units count active linear terms and every condition in each "
            "active rule."
        ),
    }
    for label, family, budgeted in baseline_labels:
        grid = budgeted_grids[family] if budgeted else standard_grids[family]
        notes = [
            f"{key}={_methodology_display_value(value)}"
            for key, value in constant_parameters[family].items()
            if not (family == "LogisticRegression" and key == "solver")
        ]
        if family == "LogisticRegression":
            notes.append("Adaptive solver selection: liblinear for binary targets; saga for multiclass targets.")
        if budgeted:
            notes.append(
                f"Complexity budget={float(metadata.get('budget', BUDGET)):g}; candidates within the budget are preferred during inner-CV selection."
            )
        notes.append("Unspecified estimator settings use the library defaults.")
        baseline_models.append(
            {
                "model": label,
                "grid_name": family,
                "candidate_count": _methodology_candidate_count(grid),
                "parameters": _methodology_parameter_rows(grid),
                "constant_settings": notes,
                "complexity": complexity_text[family],
            }
        )

    dataset_names = list(metadata.get("dataset_names", DATASET_NAMES))
    real_count = len([name for name in dataset_names if name in REAL_DATASET_NAMES])
    synthetic_count = len(dataset_names) - real_count
    outer_splits = int(metadata.get("n_splits", 5))
    inner_splits = int(metadata.get("inner_splits", 3))
    random_state = int(metadata.get("random_state", RANDOM_STATE))
    row_cap = int(metadata.get("row_cap", -1))
    protocol = [
        f"Dataset panel: {len(dataset_names)} binary-classification tasks ({real_count} real-world and {synthetic_count} synthetic).",
        f"Rows per dataset: {'all available rows' if row_cap < 0 else row_cap}.",
        f"Outer evaluation: {outer_splits}-fold StratifiedKFold with shuffle=True and random_state={random_state}.",
        f"Inner selection: {inner_splits}-fold StratifiedKFold within every outer-training partition, with shuffle=True and random_state={random_state}.",
        "Selection metric: mean inner-fold ROC-AUC. The selected candidate is refitted on the complete outer-training partition and evaluated once on the untouched outer-test partition.",
        f"Reported dataset AUC: arithmetic mean of the {outer_splits} outer-fold ROC-AUC values.",
        "Statistical comparisons use dataset-level outer-CV aggregates, Friedman ranking, paired Wilcoxon tests, and Holm adjustment where shown.",
    ]
    preprocessing = [
        "HUGIML receives the native pandas table, including categorical columns, and performs its own binning and pattern construction inside each training fold.",
        "HUGIML applies exact downstream canonicalization followed by one training-only VIF analysis. Original terms are preserved; patterns and augmented pairs with VIF greater than 5 are reduced in that order when an earlier preferred term explains at least 80 percent of their variance.",
        "Non-HUGIML models use a fold-local pipeline: numeric median imputation; categorical most-frequent imputation followed by dense one-hot encoding with unknown categories ignored.",
        "Source dataframe indexes are discarded before modelling; statsmodels-backed tasks use only columns declared in load_pandas().data.",
        "Preprocessing is fitted only on the relevant training partition. No test-fold information is used for preprocessing or parameter selection.",
        "fit_seconds measures the selected estimator's final outer-fold refit; tune_seconds measures inner-CV search; predict_seconds measures outer-test inference.",
    ]

    complexity_overview = [
        "The dashboard reports model inspection units as the common complexity measure across all supported models.",
        "Model inspection units represent the complete fitted model a reviewer must inspect, expressed through reviewed conditions, source elements, rule literals, or active score cells according to model family.",
        "A fitted numeric component is active when its absolute value exceeds 1e-12. Intercepts are not counted.",
        "The measure is computed after the selected hyperparameter configuration is refitted on the complete outer-training partition.",
    ]

    return {
        "title": "Methodology and parameter search space",
        "protocol": protocol,
        "preprocessing": preprocessing,
        "complexity_overview": complexity_overview,
        "models": hugiml_models + baseline_models,
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "metadata": {
            "random_state": RANDOM_STATE,
            "row_cap": -1,
            "budget": BUDGET,
            "dataset_names": DATASET_NAMES,
            "model_order": MODEL_ORDER,
            "grid_snapshot": grid_snapshot(),
            "dataset_feature_policy": DATASET_FEATURE_POLICY,
        },
        "results": [],
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default))
    tmp.replace(path)


def _canonical_model_label(model: Any) -> str:
    return str(model)


def _hugiml_scenario_for_row(row: dict[str, Any]) -> str | None:
    if not is_hugiml_model(str(row.get("model"))):
        return None
    return str(row.get("hugiml_scenario") or DEFAULT_DASHBOARD_HUGIML_SCENARIO)


def _pair_key(dataset: str, model: str, hugiml_scenario: str | None = None) -> tuple[str, str, str | None]:
    canonical_model = _canonical_model_label(model)
    return (
        dataset,
        canonical_model,
        hugiml_scenario if is_hugiml_model(canonical_model) else None,
    )


def pair_plan(datasets: list[str], models: list[str]) -> list[tuple[str, str, str | None]]:
    pairs: list[tuple[str, str, str | None]] = []
    for d in datasets:
        for m in models:
            if is_hugiml_model(m):
                for scenario in HUGIML_SCENARIOS:
                    pairs.append((d, m, scenario))
            else:
                pairs.append((d, m, None))
    return pairs


def completed_keys(payload: dict[str, Any]) -> set[tuple[str, str, str | None]]:
    return {
        _pair_key(str(r.get("dataset")), str(r.get("model")), _hugiml_scenario_for_row(r))
        for r in payload.get("results", [])
    }


def _validate_resume_feature_policy(payload: dict[str, Any]) -> None:
    """Reject checkpoints whose rows may have used a source index as a feature."""
    if not payload.get("results"):
        return
    observed = payload.get("metadata", {}).get("dataset_feature_policy")
    if observed != DATASET_FEATURE_POLICY:
        raise RuntimeError(
            "The existing checkpoint predates source-index exclusion and may contain "
            "index-biased statsmodels results. Start a fresh run with --fresh or use a "
            "new --out-dir/--checkpoint before enabling --resume."
        )


def _scope_label_for_dataset(dataset: str, dataset_group: str | None = None) -> str:
    """Return the dashboard summary scope for a dataset."""
    if str(dataset).startswith("Synth") or str(dataset_group) == "Synthetic":
        return "Synthetic"
    return "Real-world"


def _scope_side_rows(df_scope: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Build per-dataset AUC side table and ranks for one dataset subset."""
    side_rows: list[dict[str, Any]] = []
    if df_scope.empty:
        return side_rows, pd.DataFrame()
    for ds, g in df_scope.groupby("dataset", sort=False):
        row = {
            "dataset": ds,
            "raw_features": int(g["raw_features"].iloc[0]) if "raw_features" in g else None,
            "dataset_group": str(g["dataset_group"].iloc[0]) if "dataset_group" in g else "",
        }
        aucs = []
        for m in MODEL_ORDER:
            vals = g.loc[g["model"] == m, "auc"]
            val = float(vals.iloc[0]) if len(vals) and pd.notnull(vals.iloc[0]) else float("nan")
            row[m] = None if math.isnan(val) else val
            aucs.append(val)
        ranks_input = np.array([(-a if not math.isnan(a) else np.inf) for a in aucs])
        ranks = rankdata(ranks_input, method="average")
        finite_aucs = [a for a in aucs if not math.isnan(a)]
        if finite_aucs:
            max_auc = float(np.nanmax(aucs))
            tied = [
                m
                for m, a in zip(MODEL_ORDER, aucs)
                if not math.isnan(a) and abs(a - max_auc) <= 1e-12
            ]
        else:
            max_auc = float("nan")
            tied = []
        row["winner_auc"] = None if math.isnan(max_auc) else float(max_auc)
        row["tied_winners"] = "; ".join(tied)
        row["winner_model"] = tied[0] if len(tied) == 1 else ("Tie" if tied else None)
        for m, r in zip(MODEL_ORDER, ranks):
            row[m + "_rank"] = float(r)
        side_rows.append(row)
    return side_rows, pd.DataFrame(side_rows)


def _model_inspection_complexity_series(frame: pd.DataFrame) -> pd.Series:
    """Use the model-inspection field when present and complexity otherwise."""
    fallback = pd.to_numeric(
        frame.get("complexity", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    if "complexity_model_inspection_units" not in frame:
        return fallback
    primary = pd.to_numeric(
        frame["complexity_model_inspection_units"], errors="coerce"
    )
    return primary.combine_first(fallback)


def _model_inspection_complexity_value(row: pd.Series) -> float | None:
    """Return one row's model-inspection value, using complexity when needed."""
    primary = row.get("complexity_model_inspection_units")
    if primary is not None and not pd.isna(primary):
        return float(primary)
    fallback = row.get("complexity")
    return None if fallback is None or pd.isna(fallback) else float(fallback)


def _summary_for_scope(df_scope: pd.DataFrame, scope: str) -> dict[str, Any]:
    """Compute model-level summary rows for one scope: overall, real-world, or synthetic."""
    df_scope = df_scope[df_scope["model"].isin(MODEL_ORDER)].copy()
    side_rows, side = _scope_side_rows(df_scope)
    n_datasets = int(side.shape[0]) if not side.empty else 0
    rows: list[dict[str, Any]] = []
    if df_scope.empty or side.empty:
        return {
            "scope": scope,
            "n_datasets": 0,
            "rows": rows,
            "friedman_statistic": None,
            "friedman_p": None,
            "best_mean_auc_model": None,
            "best_mean_auc": None,
            "hugiml_mean_auc": None,
            "hugiml_mean_rank": None,
        }

    for m in MODEL_ORDER:
        sub = df_scope[df_scope.model == m]
        auc = pd.to_numeric(sub["auc"], errors="coerce")
        f1 = pd.to_numeric(sub.get("f1", pd.Series(dtype=float)), errors="coerce")
        acc = pd.to_numeric(sub.get("accuracy", pd.Series(dtype=float)), errors="coerce")
        fit = pd.to_numeric(sub.get("fit_seconds", pd.Series(dtype=float)), errors="coerce")
        pair = pd.to_numeric(sub.get("pair_seconds", pd.Series(dtype=float)), errors="coerce")
        comp = _model_inspection_complexity_series(sub).dropna()
        model_units = pd.to_numeric(
            sub.get("complexity_model_units", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        instance_ci = _series_mean_ci(
            sub.get(
                "complexity_instance_inspection_units_mean", pd.Series(dtype=float)
            )
        )
        ranks = side[m + "_rank"].astype(float) if m + "_rank" in side else pd.Series(dtype=float)
        strict_wins = int((side["winner_model"] == m).sum()) if "winner_model" in side else 0
        tied_best = (
            int(side["tied_winners"].fillna("").str.split("; ").apply(lambda xs: m in xs).sum())
            if "tied_winners" in side
            else 0
        )
        rows.append(
            {
                "scope": scope,
                "model": m,
                "n_datasets": int(sub.shape[0]),
                "mean_auc": None if auc.dropna().empty else float(auc.mean()),
                "median_auc": None if auc.dropna().empty else float(auc.median()),
                "std_auc": None
                if auc.dropna().empty or len(auc.dropna()) < 2
                else float(auc.std(ddof=1)),
                "mean_f1": None if f1.dropna().empty else float(f1.mean()),
                "mean_accuracy": None if acc.dropna().empty else float(acc.mean()),
                "mean_rank": None if ranks.dropna().empty else float(ranks.mean()),
                "median_rank": None if ranks.dropna().empty else float(ranks.median()),
                "strict_wins": strict_wins,
                "tied_best_count": tied_best,
                "mean_complexity": None if comp.empty else float(comp.mean()),
                "median_complexity": None if comp.empty else float(comp.median()),
                "mean_model_units": None if model_units.empty else float(model_units.mean()),
                "mean_model_inspection_units": None if comp.empty else float(comp.mean()),
                "median_model_inspection_units": None if comp.empty else float(comp.median()),
                "mean_instance_inspection_units": instance_ci["mean"],
                "instance_inspection_ci_lower": instance_ci["ci_lower"],
                "instance_inspection_ci_upper": instance_ci["ci_upper"],
                "instance_inspection_confidence_level": instance_ci["confidence_level"],
                "instance_inspection_n_datasets": instance_ci["n_samples"],
                "mean_fit_seconds": None if fit.dropna().empty else float(fit.mean()),
                "mean_pair_seconds": None if pair.dropna().empty else float(pair.mean()),
            }
        )

    arrays = [side[m].astype(float).to_numpy() for m in MODEL_ORDER]
    try:
        stat, p = friedmanchisquare(*arrays) if n_datasets >= 2 else (float("nan"), float("nan"))
    except Exception:
        stat, p = float("nan"), float("nan")

    valid_rows = [r for r in rows if r["mean_auc"] is not None]
    best = (
        sorted(
            valid_rows,
            key=lambda r: (
                -(r["mean_auc"] or -1),
                r["mean_rank"] if r["mean_rank"] is not None else 999,
            ),
        )[0]
        if valid_rows
        else None
    )
    hug = next((r for r in rows if r["model"] == "HUGIML"), None)
    return {
        "scope": scope,
        "n_datasets": n_datasets,
        "rows": rows,
        "friedman_statistic": None if math.isnan(float(stat)) else float(stat),
        "friedman_p": None if math.isnan(float(p)) else float(p),
        "best_mean_auc_model": None if best is None else best["model"],
        "best_mean_auc": None if best is None else best["mean_auc"],
        "hugiml_mean_auc": None if hug is None else hug["mean_auc"],
        "hugiml_mean_rank": None if hug is None else hug["mean_rank"],
        "hugiml_mean_model_inspection_units": None
        if hug is None
        else hug["mean_model_inspection_units"],
        "hugiml_mean_instance_inspection_units": None
        if hug is None
        else hug["mean_instance_inspection_units"],
        "hugiml_instance_inspection_ci_lower": None
        if hug is None
        else hug["instance_inspection_ci_lower"],
        "hugiml_instance_inspection_ci_upper": None
        if hug is None
        else hug["instance_inspection_ci_upper"],
    }


def _build_scope_summaries(df: pd.DataFrame) -> dict[str, Any]:
    """Build dashboard summary results for Overall, Real-world, and Synthetic."""
    if df.empty:
        scopes = [
            _summary_for_scope(df, "Overall"),
            _summary_for_scope(df, "Real-world"),
            _summary_for_scope(df, "Synthetic"),
        ]
    else:
        tmp = df.copy()
        tmp["summary_scope"] = [
            _scope_label_for_dataset(d, g)
            for d, g in zip(tmp["dataset"], tmp.get("dataset_group", pd.Series([None] * len(tmp))))
        ]
        scopes = [
            _summary_for_scope(tmp, "Overall"),
            _summary_for_scope(tmp[tmp["summary_scope"] == "Real-world"], "Real-world"),
            _summary_for_scope(tmp[tmp["summary_scope"] == "Synthetic"], "Synthetic"),
        ]
    flat_rows = [row for scope in scopes for row in scope["rows"]]
    scope_tests = [{k: v for k, v in scope.items() if k != "rows"} for scope in scopes]
    return {"summary_by_scope": flat_rows, "scope_tests": scope_tests}



def _safe_number_or_none(value: Any) -> float | None:
    """Return a finite float for numeric values, otherwise None."""
    try:
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _infer_complexity_budget(row: dict[str, Any]) -> float | None:
    """Infer the aggregate complexity budget for older checkpoints.

    Older checkpoint files stored ``complexity_budget`` only inside
    ``fold_rows_json``.  Assembly builds a dataframe from top-level aggregate
    rows, so normalize the schema before dataframe construction.
    """
    direct = _safe_number_or_none(row.get("complexity_budget"))
    if direct is not None:
        return direct

    fold_rows_raw = row.get("fold_rows_json")
    if fold_rows_raw:
        try:
            fold_rows = json.loads(fold_rows_raw) if isinstance(fold_rows_raw, str) else fold_rows_raw
        except Exception:
            fold_rows = []
        budgets = [
            value
            for value in (_safe_number_or_none(fr.get("complexity_budget")) for fr in fold_rows)
            if value is not None
        ]
        if budgets:
            return float(np.mean(budgets))

    # Last-resort fallbacks for very old rows without fold-level details.
    # HUGIML's effective budget is the selected topK; budgeted baselines use
    # the global benchmark budget.
    try:
        params = json.loads(row.get("best_params_json") or "{}")
    except Exception:
        params = {}
    if is_hugiml_model(str(row.get("model"))):
        return _safe_number_or_none(params.get("topK"))
    if "complexity-budgeted" in str(row.get("model", "")):
        return float(BUDGET)
    return None


_DASHBOARD_UNUSED_FOLD_FIELDS = frozenset(
    {
        "best_params_by_fold_json",
        "fold_rows_json",
        "complexity_report_by_fold_json",
    }
)


def _normalize_detail_rows_for_assembly(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize checkpoint rows and omit fold payloads unused by dashboard assembly."""
    normalized: list[dict[str, Any]] = []
    for row in details:
        item = dict(row)
        item["model"] = _canonical_model_label(item.get("model"))
        marker = _timing_inclusion_marker(item)
        item["fit_time_included_in_tuning"] = marker
        item["fit_time_accounting"] = (
            "included_in_tuning"
            if marker is True
            else "separate_from_tuning"
            if marker is False
            else "unavailable"
        )
        if "complexity_budget" not in item or _safe_number_or_none(item.get("complexity_budget")) is None:
            item["complexity_budget"] = _infer_complexity_budget(item)
        for field in _DASHBOARD_UNUSED_FOLD_FIELDS:
            item.pop(field, None)
        normalized.append(item)
    return normalized

def _make_data_single(details: list[dict[str, Any]]) -> dict[str, Any]:
    details = _normalize_detail_rows_for_assembly(details)
    df = pd.DataFrame(details)
    df = df[df["model"].isin(MODEL_ORDER) & df["dataset"].isin(DATASET_NAMES)].copy()
    order = {d: i for i, d in enumerate(DATASET_NAMES)}
    mo = {m: i for i, m in enumerate(MODEL_ORDER)}
    df["dataset_order"] = df["dataset"].map(order)
    df["model_order"] = df["model"].map(mo)
    df = df.sort_values(["dataset_order", "model_order"]).drop(
        columns=["dataset_order", "model_order"]
    )

    side_rows = []
    for ds, g in df.groupby("dataset", sort=False):
        row = {
            "dataset": ds,
            "raw_features": int(g["raw_features"].iloc[0]),
            "dataset_group": str(g["dataset_group"].iloc[0]),
        }
        aucs = []
        for m in MODEL_ORDER:
            vals = g.loc[g["model"] == m, "auc"]
            val = float(vals.iloc[0]) if len(vals) and pd.notnull(vals.iloc[0]) else float("nan")
            row[m] = None if math.isnan(val) else val
            aucs.append(val)
        ranks_input = np.array([(-a if not math.isnan(a) else np.inf) for a in aucs])
        ranks = rankdata(ranks_input, method="average")
        max_auc = np.nanmax(aucs)
        tied = [
            m for m, a in zip(MODEL_ORDER, aucs) if not math.isnan(a) and abs(a - max_auc) <= 1e-12
        ]
        row["winner_auc"] = float(max_auc)
        row["tied_winners"] = "; ".join(tied)
        row["winner_model"] = tied[0] if len(tied) == 1 else "Tie"
        for m, r in zip(MODEL_ORDER, ranks):
            row[m + "_rank"] = float(r)
        side_rows.append(row)
    side = pd.DataFrame(side_rows)

    overall = []
    for m in MODEL_ORDER:
        sub = df[df.model == m]
        vals = pd.to_numeric(sub["auc"], errors="coerce")
        comp = _model_inspection_complexity_series(sub).dropna()
        model_units = pd.to_numeric(
            sub.get("complexity_model_units", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        instance_ci = _series_mean_ci(
            sub.get(
                "complexity_instance_inspection_units_mean", pd.Series(dtype=float)
            )
        )
        ranks = side[m + "_rank"].astype(float)
        strict_wins = int((side["winner_model"] == m).sum())
        tied_best = int(
            side["tied_winners"].fillna("").str.split("; ").apply(lambda xs: m in xs).sum()
        )
        overall.append(
            {
                "model": m,
                "mean_auc": float(vals.mean()),
                "std_auc": float(vals.std(ddof=1)),
                "median_auc": float(vals.median()),
                "min_auc": float(vals.min()),
                "max_auc": float(vals.max()),
                "strict_wins": strict_wins,
                "tied_best_count": tied_best,
                "mean_rank": float(ranks.mean()),
                "median_rank": float(ranks.median()),
                "mean_complexity": None if comp.empty else float(comp.mean()),
                "median_complexity": None if comp.empty else float(comp.median()),
                "min_complexity": None if comp.empty else float(comp.min()),
                "max_complexity": None if comp.empty else float(comp.max()),
                "mean_model_units": None if model_units.empty else float(model_units.mean()),
                "mean_model_inspection_units": None if comp.empty else float(comp.mean()),
                "median_model_inspection_units": None if comp.empty else float(comp.median()),
                "mean_instance_inspection_units": instance_ci["mean"],
                "instance_inspection_ci_lower": instance_ci["ci_lower"],
                "instance_inspection_ci_upper": instance_ci["ci_upper"],
                "instance_inspection_confidence_level": instance_ci["confidence_level"],
                "instance_inspection_n_datasets": instance_ci["n_samples"],
            }
        )
    overall = sorted(overall, key=lambda r: (-r["mean_auc"], r["mean_rank"]))

    arrays = [side[m].astype(float).to_numpy() for m in MODEL_ORDER]
    try:
        stat, p = friedmanchisquare(*arrays)
    except Exception:
        stat, p = float("nan"), float("nan")
    global_rows = [
        {
            "friedman_statistic": float(stat),
            "p_value": float(p),
            "n_datasets": int(side.shape[0]),
            "n_models": len(MODEL_ORDER),
            "interpretation_alpha_0_05": "significant_global_difference"
            if p < 0.05
            else "not_significant",
        }
    ]

    pairs = []
    hug = side["HUGIML"].astype(float).to_numpy()
    for m in MODEL_ORDER:
        if m == "HUGIML":
            continue
        arr = side[m].astype(float).to_numpy()
        diff = arr - hug
        try:
            w, pv = wilcoxon(
                diff, zero_method="wilcox", correction=False, alternative="two-sided", method="auto"
            )
        except Exception:
            w, pv = float("nan"), 1.0
        pairs.append(
            {
                "comparison": f"{m} - HUGIML",
                "baseline_model": m,
                "mean_auc_diff_vs_hugiml": float(np.nanmean(diff)),
                "median_auc_diff_vs_hugiml": float(np.nanmedian(diff)),
                "wins_vs_hugiml": int(np.sum(arr > hug)),
                "losses_vs_hugiml": int(np.sum(arr < hug)),
                "ties": int(np.sum(np.isclose(arr, hug, rtol=1e-12, atol=1e-12))),
                "wilcoxon_statistic": float(w),
                "wilcoxon_p": float(pv),
                "bootstrap_ci_low": None,
                "bootstrap_ci_high": None,
            }
        )
    pairs_sorted = sorted(
        pairs, key=lambda r: r["wilcoxon_p"] if r["wilcoxon_p"] is not None else 1.0
    )
    prev = 0.0
    for i, r in enumerate(pairs_sorted, start=1):
        pval = (
            1.0
            if r["wilcoxon_p"] is None or not math.isfinite(float(r["wilcoxon_p"]))
            else float(r["wilcoxon_p"])
        )
        adj = min(1.0, max(prev, pval * (len(pairs_sorted) - i + 1)))
        r["holm_p_vs_hugiml_family"] = float(adj)
        r["significant_holm_0_05"] = bool(adj < 0.05)
        prev = adj

    pairwise = []
    for i, a in enumerate(MODEL_ORDER):
        for b in MODEL_ORDER[i + 1 :]:
            arr_a = side[a].astype(float).to_numpy()
            arr_b = side[b].astype(float).to_numpy()
            diff = arr_a - arr_b
            try:
                w, pv = wilcoxon(
                    diff,
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    method="auto",
                )
            except Exception:
                w, pv = float("nan"), 1.0
            pairwise.append(
                {
                    "model_1": a,
                    "model_2": b,
                    "comparison": f"{a} - {b}",
                    "mean_auc_diff": float(np.nanmean(diff)),
                    "median_auc_diff": float(np.nanmedian(diff)),
                    "wins_model_1": int(np.sum(arr_a > arr_b)),
                    "wins_model_2": int(np.sum(arr_a < arr_b)),
                    "ties": int(np.sum(np.isclose(arr_a, arr_b, rtol=1e-12, atol=1e-12))),
                    "wilcoxon_statistic": float(w),
                    "wilcoxon_p": float(pv),
                    "bootstrap_ci_low": None,
                    "bootstrap_ci_high": None,
                }
            )
    ps = sorted(pairwise, key=lambda r: r["wilcoxon_p"] if r["wilcoxon_p"] is not None else 1.0)
    prev = 0.0
    for i, r in enumerate(ps, start=1):
        pval = (
            1.0
            if r["wilcoxon_p"] is None or not math.isfinite(float(r["wilcoxon_p"]))
            else float(r["wilcoxon_p"])
        )
        adj = min(1.0, max(prev, pval * (len(ps) - i + 1)))
        r["holm_p_all_pairs"] = float(adj)
        r["significant_holm_0_05"] = bool(adj < 0.05)
        prev = adj

    heat = []
    for _, r in df.iterrows():
        heat.append(
            {
                "dataset": r["dataset"],
                "model": r["model"],
                "auc": None if pd.isna(r["auc"]) else float(r["auc"]),
                "f1": None if pd.isna(r["f1"]) else float(r["f1"]),
                "accuracy": None if pd.isna(r["accuracy"]) else float(r["accuracy"]),
            }
        )

    budget = []
    for m in [x for x in MODEL_ORDER if "complexity-budgeted" in x]:
        sub = df[df.model == m]
        c = pd.to_numeric(sub["complexity"], errors="coerce").dropna()
        budget.append(
            {
                "model": m,
                "budget": BUDGET,
                "n_datasets": int(sub.shape[0]),
                "mean_complexity": float(c.mean()),
                "median_complexity": float(c.median()),
                "max_complexity": float(c.max()),
                "n_over_budget": int((c > BUDGET).sum()),
            }
        )

    hugrows = []
    for _, r in df[df.model.map(is_hugiml_model)].iterrows():
        try:
            params = json.loads(r["best_params_json"])
        except Exception:
            params = {}
        hugrows.append(
            {
                "dataset": r["dataset"],
                "test_auc": None if pd.isna(r["auc"]) else float(r["auc"]),
                "test_f1": None if pd.isna(r["f1"]) else float(r["f1"]),
                "test_accuracy": None if pd.isna(r["accuracy"]) else float(r["accuracy"]),
                "model_units": None
                if pd.isna(r.get("complexity_model_units"))
                else float(r.get("complexity_model_units")),
                "model_inspection_units": _model_inspection_complexity_value(r),
                "instance_inspection_units_mean": None
                if pd.isna(r.get("complexity_instance_inspection_units_mean"))
                else float(r.get("complexity_instance_inspection_units_mean")),
                "instance_inspection_units_ci_lower": None
                if pd.isna(r.get("complexity_instance_inspection_units_ci_lower"))
                else float(r.get("complexity_instance_inspection_units_ci_lower")),
                "instance_inspection_units_ci_upper": None
                if pd.isna(r.get("complexity_instance_inspection_units_ci_upper"))
                else float(r.get("complexity_instance_inspection_units_ci_upper")),
                "complexity_budget": None
                if pd.isna(r.get("complexity_budget"))
                else float(r.get("complexity_budget")),
                "feature_mode": params.get("feature_mode"),
                "topK": params.get("topK"),
                "L": params.get("L"),
                "best_params_json": r["best_params_json"],
            }
        )

    scope_summary = _build_scope_summaries(df)
    return {
        "side": side_rows,
        "overall": overall,
        "details": df.drop(
            columns=[c for c in df.columns if _is_local_provenance_key(c)]
        ).to_dict(orient="records"),
        "heat": heat,
        "global": global_rows,
        "vs_hugiml": pairs,
        "pairwise_all": pairwise,
        "budget": budget,
        "hug": hugrows,
        "summary_by_scope": scope_summary["summary_by_scope"],
        "scope_tests": scope_summary["scope_tests"],
        "dataset_catalog": [
            {
                "dataset": name,
                "kind": (
                    "Synthetic" if name in SYNTHETIC_DATASET_NAMES else "Public real-world / package"
                ),
                "source": (
                    "generated in script"
                    if name in SYNTHETIC_DATASET_NAMES
                    else "scikit-learn or statsmodels"
                ),
            }
            for name in DATASET_NAMES
        ],
    }


def _scenario_details_for_dashboard(df: pd.DataFrame, scenario: str) -> list[dict[str, Any]]:
    hug_mask = df["model"].map(is_hugiml_model)
    baseline = df[~hug_mask].copy()
    hug = df[hug_mask].copy()
    if "hugiml_scenario" not in hug.columns:
        hug["hugiml_scenario"] = DEFAULT_DASHBOARD_HUGIML_SCENARIO
    hug["hugiml_scenario"] = hug["hugiml_scenario"].fillna(DEFAULT_DASHBOARD_HUGIML_SCENARIO)
    hug = hug[hug["hugiml_scenario"].astype(str) == scenario].copy()
    details = pd.concat([hug, baseline], axis=0, ignore_index=True)
    return details.to_dict(orient="records")


def _rpte_dashboard_rows(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for detail in details:
        if not is_hugiml_model(str(detail.get("model"))):
            continue
        try:
            path_counts = json.loads(detail.get("hugiml_path_counts_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            path_counts = {}
        output = {
            "dataset": detail.get("dataset"),
            "split_count": int(detail.get("outer_folds_completed", 0) or 0),
            "lr_split_count": int(path_counts.get("lr", 0) or 0),
            "rpte_sequential_estimator_count": int(
                path_counts.get("rpte_sequential", 0) or 0
            ),
            "rpte_lookahead_estimator_count": int(
                path_counts.get("rpte_lookahead", 0) or 0
            ),
        }
        output["rpte_split_count"] = max(
            0, output["split_count"] - output["lr_split_count"]
        )
        for metric in RPTE_DISTRIBUTION_METRICS:
            for statistic in ("n", "mean", "std", "min", "q25", "median", "q75", "max"):
                output[f"{metric}_{statistic}"] = detail.get(f"{metric}_{statistic}")
        rows.append(output)
    return rows


def make_data(details: list[dict[str, Any]]) -> dict[str, Any]:
    """Build dashboard data for selectable HUGIML configuration options.

    Each option keeps the same chart and table schema. Baseline rows are shared,
    while the HUGIML rows are selected from the requested configuration.
    """
    details = _normalize_detail_rows_for_assembly(details)
    df = pd.DataFrame(details)
    if df.empty or "hugiml_scenario" not in df.columns:
        return _make_data_single(details)

    available: list[dict[str, Any]] = []
    shared_catalog: list[dict[str, Any]] | None = None
    for scenario, spec in HUGIML_SCENARIOS.items():
        scenario_details = _scenario_details_for_dashboard(df, scenario)
        if not any(is_hugiml_model(str(r.get("model"))) for r in scenario_details):
            continue
        scenario_data = _make_data_single(scenario_details)
        scenario_data["hugiml_rpte_distributions"] = _rpte_dashboard_rows(scenario_details)
        scenario_data["active_hugiml_scenario"] = scenario
        scenario_data["active_hugiml_scenario_label"] = spec["label"]
        scenario_data["active_hugiml_grid_name"] = spec["grid_name"]
        if shared_catalog is None:
            shared_catalog = scenario_data.get("dataset_catalog", [])
        else:
            scenario_data["dataset_catalog"] = shared_catalog
        available.append(
            {
                "id": scenario,
                "label": spec["label"],
                "description": spec["description"],
                "grid_name": spec["grid_name"],
                "data": scenario_data,
            }
        )

    if not available:
        return _make_data_single(details)

    default_entry = next(
        (s for s in available if s["id"] == DEFAULT_DASHBOARD_HUGIML_SCENARIO), available[0]
    )
    data = copy.deepcopy(default_entry["data"])
    data["dashboard_scenarios"] = available
    data["default_hugiml_scenario"] = default_entry["id"]
    data["active_hugiml_scenario"] = default_entry["id"]
    data["active_hugiml_scenario_label"] = default_entry["label"]
    data["active_hugiml_grid_name"] = default_entry["grid_name"]
    return data


def extract_original_data(html_path: Path) -> dict[str, Any]:
    text = html_path.read_text(errors="ignore")
    model_order_pos = text.index(";\nconst MODEL_ORDER")
    data_pos = text.index("const DATA=")
    scenarios_pos = text.find("const DASHBOARD_SCENARIOS=")

    # Compact dashboards serialize the scenario payload once, then construct
    # the active DATA object in JavaScript from the selected default scenario.
    if 0 <= scenarios_pos < data_pos:
        scenario_start = scenarios_pos + len("const DASHBOARD_SCENARIOS=")
        scenario_end = text.index(";\nconst DEFAULT_DASHBOARD_SCENARIO", scenario_start)
        scenarios = json.loads(text[scenario_start:scenario_end])
        default_match = re.search(
            r"const DEFAULT_DASHBOARD_SCENARIO=(.*?);\nconst DATA=",
            text[scenario_end:model_order_pos],
            flags=re.S,
        )
        default_id = json.loads(default_match.group(1)) if default_match else scenarios[0]["id"]
        entry = next((item for item in scenarios if item.get("id") == default_id), scenarios[0])
        data = copy.deepcopy(entry["data"])
        data["dashboard_scenarios"] = scenarios
        data["default_hugiml_scenario"] = default_id
        return data

    start = data_pos + len("const DATA=")
    if scenarios_pos > data_pos:
        end = text.rfind(";\n", start, scenarios_pos)
    else:
        end = model_order_pos
    return json.loads(text[start:end])


def summary_comparison(
    original_html: Path, new_data: dict[str, Any], out_csv: Path
) -> pd.DataFrame:
    comparison_columns = [
        "model",
        "mean_auc",
        "mean_rank",
        "strict_wins",
        "tied_best_count",
        "mean_model_units",
        "mean_model_inspection_units",
        "mean_instance_inspection_units",
        "instance_inspection_ci_lower",
        "instance_inspection_ci_upper",
    ]
    new_frame = pd.DataFrame(new_data["overall"])
    new = new_frame.reindex(columns=comparison_columns)
    try:
        original = extract_original_data(original_html)
        old_frame = pd.DataFrame(original["overall"])
        old = old_frame.reindex(columns=comparison_columns)
        old = old.rename(
            columns={
                "mean_auc": "mean_auc_earlier",
                "mean_rank": "mean_rank_earlier",
                "strict_wins": "strict_wins_earlier",
                "tied_best_count": "tied_best_count_earlier",
                "mean_model_units": "mean_model_units_earlier",
                "mean_model_inspection_units": "mean_model_inspection_units_earlier",
                "mean_instance_inspection_units": "mean_instance_inspection_units_earlier",
                "instance_inspection_ci_lower": "instance_inspection_ci_lower_earlier",
                "instance_inspection_ci_upper": "instance_inspection_ci_upper_earlier",
            }
        )
        new = new.rename(
            columns={
                "mean_auc": "mean_auc_new",
                "mean_rank": "mean_rank_new",
                "strict_wins": "strict_wins_new",
                "tied_best_count": "tied_best_count_new",
                "mean_model_units": "mean_model_units_new",
                "mean_model_inspection_units": "mean_model_inspection_units_new",
                "mean_instance_inspection_units": "mean_instance_inspection_units_new",
                "instance_inspection_ci_lower": "instance_inspection_ci_lower_new",
                "instance_inspection_ci_upper": "instance_inspection_ci_upper_new",
            }
        )
        comp = old.merge(new, on="model")
        comp["delta_mean_auc"] = comp["mean_auc_new"] - comp["mean_auc_earlier"]
        comp["delta_mean_rank"] = comp["mean_rank_new"] - comp["mean_rank_earlier"]
        comp = comp.sort_values(["mean_auc_new", "mean_rank_new"], ascending=[False, True])
    except Exception:
        comp = new.rename(
            columns={
                "mean_auc": "mean_auc_new",
                "mean_rank": "mean_rank_new",
                "strict_wins": "strict_wins_new",
                "tied_best_count": "tied_best_count_new",
                "mean_model_units": "mean_model_units_new",
                "mean_model_inspection_units": "mean_model_inspection_units_new",
                "mean_instance_inspection_units": "mean_instance_inspection_units_new",
                "instance_inspection_ci_lower": "instance_inspection_ci_lower_new",
                "instance_inspection_ci_upper": "instance_inspection_ci_upper_new",
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_csv, index=False)
    return comp


def _default_dashboard_template() -> str:
    return """<!doctype html>
<html><head><meta charset=\"utf-8\"><title>HUGIML Benchmark Dashboard</title>
<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827}.card{background:white;border:1px solid #e5e7eb;border-radius:14px;padding:18px;margin:16px 0;box-shadow:0 1px 2px #0001}.plot{height:430px;overflow:visible}.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e5e7eb;padding:8px;text-align:left;vertical-align:top}th{background:#f3f4f6}.meta{color:#6b7280;font-size:13px}</style>
</head><body><h1>HUGIML Benchmark Dashboard</h1><div class=\"meta\">Friedman p-value: pending</div>
<section class=\"grid-2\"><div class=\"card\"><h2>Mean AUC by model</h2><div id=\"meanAuc\" class=\"plot\"></div></div><div class=\"card\"><h2>Model inspection units vs performance</h2><div id=\"complexityPerf\" class=\"plot\"></div></div></section>
<section class=\"card\"><h2>Dataset × model heatmap</h2><div id=\"aucHeat\" class=\"plot\"></div></section>
<script>
const DATA={};
const MODEL_ORDER=[];
function theme(){return {colors:['#2563eb','#dc2626','#16a34a','#ca8a04','#9333ea','#0891b2','#4b5563'],grid:'#e5e7eb'}}
function renderBase(){
  if(!window.Plotly || !DATA.overall) return;
  const models=DATA.overall.map(r=>r.model), auc=DATA.overall.map(r=>r.mean_auc);
  Plotly.react('meanAuc',[{type:'bar',x:models,y:auc}],{margin:{l:55,r:30,t:20,b:100},yaxis:{title:'Mean AUC',range:[Math.max(0,Math.min(...auc)-0.03),1.01]},xaxis:{automargin:true}}, {responsive:true});
  const details=DATA.details||[]; const rows=details.filter(r=>r.complexity!=null && r.auc!=null);
  const traces=[...new Set(rows.map(r=>r.model))].map((m,i)=>{const sub=rows.filter(r=>r.model===m);return {type:'scatter',mode:'markers',name:m,x:sub.map(r=>r.complexity),y:sub.map(r=>r.auc),text:sub.map(r=>r.dataset),marker:{size:12,line:{width:1}},cliponaxis:false};});
  Plotly.react('complexityPerf',traces,{margin:{l:70,r:80,t:20,b:70},xaxis:{title:'Model inspection units',type:'log',automargin:true},yaxis:{title:'AUC',automargin:true,range:[0,1.03]},legend:{orientation:'h',y:-0.22}}, {responsive:true});
  const ds=[...new Set((DATA.heat||[]).map(r=>r.dataset))]; const ms=[...new Set((DATA.heat||[]).map(r=>r.model))];
  const z=ds.map(d=>ms.map(m=>{const r=(DATA.heat||[]).find(x=>x.dataset===d&&x.model===m);return r?r.auc:null;}));
  Plotly.react('aucHeat',[{type:'heatmap',x:ms,y:ds,z:z}],{margin:{l:220,r:30,t:20,b:120},xaxis:{automargin:true},yaxis:{automargin:true}}, {responsive:true});
}
window.addEventListener('load', renderBase);
</script></body></html>"""


def _inject_before_body_end(text: str, block: str) -> str:
    if "</body>" in text:
        return text.replace("</body>", block + "\n</body>")
    return text + "\n" + block


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    """Return a compact SHA-256 fingerprint for a file if it is readable."""
    p = Path(path)
    out: dict[str, Any] = {
        "path": str(p),
        "exists": bool(p.exists()),
        "type": "file" if p.is_file() else ("directory" if p.is_dir() else "missing"),
    }
    if not p.exists() or not p.is_file():
        return out
    try:
        st = p.stat()
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        out.update(
            {
                "size_bytes": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
                "sha256": h.hexdigest(),
            }
        )
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def _fingerprint_files(paths: list[Path]) -> list[dict[str, Any]]:
    """Fingerprint a de-duplicated ordered list of files."""
    seen: set[Path] = set()
    out: list[dict[str, Any]] = []
    for raw in paths:
        try:
            p = Path(raw).expanduser().resolve()
        except Exception:
            p = Path(raw)
        if p in seen:
            continue
        seen.add(p)
        out.append(_sha256_file(p))
    return out


def _source_tree_fingerprint(
    root: Path | None,
    *,
    max_files: int = 2000,
    max_file_size_bytes: int = 2 * 1024 * 1024,
) -> dict[str, Any]:
    """Hash source/build inputs that commonly affect the Python and native extension build."""
    if root is None:
        return {"available": False, "reason": "SOURCE_ROOT was not discovered; using installed package."}
    base = Path(root).resolve()
    if not base.exists():
        return {"available": False, "path": str(base), "reason": "SOURCE_ROOT does not exist."}

    rel_roots = [
        "src",
        "native",
        "include",
        "cmake",
        "CMakeLists.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "MANIFEST.in",
        "requirements.txt",
        "requirements-dev.txt",
        "experiments/benchmark/benchmark_dashboard.py",
    ]
    allowed_suffixes = {
        "",
        ".c",
        ".cc",
        ".cmake",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".hxx",
        ".in",
        ".lock",
        ".md",
        ".py",
        ".pyi",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
        ".cfg",
    }
    skip_parts = {".git", ".mypy_cache", ".pytest_cache", "__pycache__", "build", "dist"}
    files: list[Path] = []
    for rel in rel_roots:
        candidate = base / rel
        if not candidate.exists():
            continue
        if candidate.is_file():
            files.append(candidate)
            continue
        for f in candidate.rglob("*"):
            if not f.is_file():
                continue
            if any(part in skip_parts for part in f.relative_to(base).parts):
                continue
            if f.suffix.lower() not in allowed_suffixes:
                continue
            try:
                if f.stat().st_size > max_file_size_bytes:
                    continue
            except Exception:
                continue
            files.append(f)

    files = sorted(set(files), key=lambda p: p.relative_to(base).as_posix())[:max_files]
    entries: list[dict[str, Any]] = []
    tree_hash = hashlib.sha256()
    for f in files:
        rel = f.relative_to(base).as_posix()
        fp = _sha256_file(f)
        entry = {
            "path": rel,
            "size_bytes": fp.get("size_bytes"),
            "sha256": fp.get("sha256"),
        }
        if "error" in fp:
            entry["error"] = fp["error"]
        entries.append(entry)
        tree_hash.update(rel.encode("utf-8", errors="replace"))
        tree_hash.update(b"\0")
        tree_hash.update(str(entry.get("size_bytes")).encode("ascii", errors="replace"))
        tree_hash.update(b"\0")
        tree_hash.update(str(entry.get("sha256")).encode("ascii", errors="replace"))
        tree_hash.update(b"\0")

    return {
        "available": True,
        "root": str(base),
        "file_count_recorded": len(entries),
        "file_count_limit": int(max_files),
        "max_file_size_bytes": int(max_file_size_bytes),
        "tree_sha256": tree_hash.hexdigest(),
        "files": entries,
    }


def _run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    executable = shutil.which(cmd[0]) if cmd else None
    out: dict[str, Any] = {"cmd": cmd, "available": bool(executable)}
    if not executable:
        return out
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        out.update(
            {
                "returncode": int(p.returncode),
                "stdout": (p.stdout or "")[:12000],
                "stderr": (p.stderr or "")[:12000],
            }
        )
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def _redact_remote_line(line: str) -> str:
    # Redact tokens in https://token@host/path remotes while leaving host/path visible.
    return re.sub(r"(https?://)([^/@\s]+)@", r"\1<redacted>@", line)


def _git_metadata(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"available": False, "reason": "SOURCE_ROOT was not discovered."}
    base = Path(root).resolve()
    if not (base / ".git").exists() and _run_command(["git", "rev-parse", "--is-inside-work-tree"], cwd=base).get("returncode") != 0:
        return {"available": False, "root": str(base), "reason": "not a git checkout"}

    def git(*args: str) -> dict[str, Any]:
        return _run_command(["git", *args], cwd=base, timeout_s=8.0)

    remote = git("remote", "-v")
    if isinstance(remote.get("stdout"), str):
        remote["stdout"] = "\n".join(_redact_remote_line(x) for x in remote["stdout"].splitlines())
    return {
        "available": True,
        "root": str(base),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "status_short": git("status", "--short"),
        "diff_stat": git("diff", "--stat"),
        "diff_name_status": git("diff", "--name-status"),
        "tag_describe": git("describe", "--tags", "--dirty", "--always"),
        "remotes_redacted": remote,
    }


def _distribution_record(dist_name: str) -> dict[str, Any] | None:
    try:
        dist = importlib_metadata.distribution(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception as exc:
        return {"name": dist_name, "error": repr(exc)}
    files = []
    try:
        for f in list(dist.files or [])[:500]:
            files.append(str(f))
    except Exception:
        files = []
    metadata = {}
    try:
        for key in ["Name", "Version", "Summary", "Home-page", "License", "Installer"]:
            val = dist.metadata.get(key)
            if val:
                metadata[key] = val
    except Exception:
        pass
    return {
        "name": metadata.get("Name", dist_name),
        "version": getattr(dist, "version", None),
        "location": str(getattr(dist, "_path", "")),
        "metadata": metadata,
        "recorded_files_sample": files,
        "recorded_files_sample_limit": 500,
    }


def _installed_python_distributions() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dist in importlib_metadata.distributions():
        try:
            name = dist.metadata.get("Name") or "<unknown>"
            out.append(
                {
                    "name": str(name),
                    "version": str(getattr(dist, "version", "")),
                    "location": str(getattr(dist, "_path", "")),
                    "installer": str(dist.metadata.get("Installer") or ""),
                }
            )
        except Exception as exc:
            out.append({"error": repr(exc)})
    return sorted(out, key=lambda r: (str(r.get("name", "")).lower(), str(r.get("version", ""))))


def _pip_freeze() -> dict[str, Any]:
    return _run_command([sys.executable, "-m", "pip", "freeze", "--all"], timeout_s=15.0)


def _compiler_probe(command: str | None) -> dict[str, Any]:
    if not command:
        return {"command": command, "available": False}
    try:
        parts = shlex.split(str(command))
    except Exception:
        parts = str(command).split()
    if not parts:
        return {"command": command, "available": False}
    return _run_command([parts[0], "--version"], timeout_s=5.0)


def _hugiml_module_metadata() -> dict[str, Any]:
    out: dict[str, Any] = {
        "package_version_attr": getattr(_hugiml_pkg, "__version__", None),
        "distributions": {},
    }
    for dist_name in ["hugiml", "hugiml-core", "hugiml_core"]:
        rec = _distribution_record(dist_name)
        if rec is not None:
            out["distributions"][dist_name] = rec
    try:
        import _hugiml_core as core_ext  # type: ignore

        core_path = Path(getattr(core_ext, "__file__", "")).resolve()
        out["native_extension"] = {
            "module": "_hugiml_core",
            "path": str(core_path),
            "fingerprint": _sha256_file(core_path),
            "module_attributes": {
                k: str(getattr(core_ext, k))
                for k in ["__doc__", "__file__", "__name__", "__package__"]
                if hasattr(core_ext, k)
            },
        }
    except Exception as exc:
        out["native_extension"] = {"module": "_hugiml_core", "error": repr(exc)}
    return out


def _native_build_metadata() -> dict[str, Any]:
    """Capture native-extension build/rebuild evidence relevant to performance reproducibility."""
    config_keys = [
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "CPPFLAGS",
        "LDFLAGS",
        "LDSHARED",
        "CCSHARED",
        "LINKFORSHARED",
        "AR",
        "ARFLAGS",
        "SOABI",
        "EXT_SUFFIX",
        "CONFIG_ARGS",
        "Py_DEBUG",
        "Py_ENABLE_SHARED",
        "MULTIARCH",
        "LDLIBRARY",
        "LIBDIR",
        "INCLUDEPY",
    ]
    env_keys = [
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "CPPFLAGS",
        "LDFLAGS",
        "ARCHFLAGS",
        "MACOSX_DEPLOYMENT_TARGET",
        "CMAKE_ARGS",
        "CMAKE_BUILD_PARALLEL_LEVEL",
        "CMAKE_GENERATOR",
        "CMAKE_PREFIX_PATH",
        "SKBUILD_CONFIGURE_OPTIONS",
        "PYBIND11_GLOBAL_SDIST",
        "PYBIND11_PYTHON_VERSION",
        "NPY_NUM_BUILD_JOBS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    sysconfig_vars = {k: sysconfig.get_config_var(k) for k in config_keys}
    env_vars = {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None}
    compiler_commands = {
        "env_CC": env_vars.get("CC"),
        "env_CXX": env_vars.get("CXX"),
        "sysconfig_CC": sysconfig_vars.get("CC"),
        "sysconfig_CXX": sysconfig_vars.get("CXX"),
    }
    compilers = {name: _compiler_probe(cmd) for name, cmd in compiler_commands.items() if cmd}

    native_extension = _hugiml_module_metadata().get("native_extension", {})
    native_path_raw = native_extension.get("path") if isinstance(native_extension, dict) else None
    native_path = Path(native_path_raw) if native_path_raw else None
    linkage: dict[str, Any] = {}
    if native_path is not None and native_path.exists():
        linkage["ldd"] = _run_command(["ldd", str(native_path)], timeout_s=8.0)
        linkage["readelf_dynamic"] = _run_command(["readelf", "-d", str(native_path)], timeout_s=8.0)
        linkage["otool_L"] = _run_command(["otool", "-L", str(native_path)], timeout_s=8.0)

    return {
        "note": (
            "Exact compiler flags used to build an already-installed wheel are not always "
            "recoverable. This section records the active Python build configuration, "
            "build-related environment variables, compiler probes, native-extension binary "
            "fingerprint, and dynamic linkage so others can rebuild or compare environments."
        ),
        "sysconfig": sysconfig_vars,
        "environment": env_vars,
        "compiler_version_probes": compilers,
        "native_extension_linkage": linkage,
    }


def _sanitize_sbom_local_paths(sbom: dict[str, Any], *, out_dir: Path) -> dict[str, Any]:
    """Replace the local output-directory path recorded anywhere in the SBOM
    (the top-level "benchmark.out_dir" field, and every per-artifact "path"
    under "artifacts" -- each artifact path is out_dir/<filename>, so it
    carries the same prefix) with a stable placeholder. Mirrors
    experiments/scalability/scalability_dashboard.py's
    ``_sanitize_local_paths``: the "Reproducibility / SBOM manifest" is
    documented as shareable output, so the one local path this script
    chooses (where to write results) should not leak the machine/user's
    directory structure. Broader scrubbing of every possibly-local path
    recorded elsewhere in the manifest (installed-package locations,
    sys.path entries, working directory) is intentionally out of scope
    here, same as in the scalability dashboard's version of this function.

    Operates on every string value in the (nested dict/list) structure
    rather than only known field names, so it also catches the artifact
    paths without having to enumerate each one.
    """
    raw = str(out_dir)
    resolved = str(Path(out_dir).resolve())
    needles = [n for n in {raw, resolved} if n]

    def _walk(value: Any) -> Any:
        if isinstance(value, str):
            for needle in needles:
                value = value.replace(needle, "<output-dir>")
            return value
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v) for v in value]
        return value

    return _walk(copy.deepcopy(sbom))


def build_reproducibility_sbom(
    *,
    checkpoint: Path,
    out_dir: Path,
    template_html: Path,
    payload: dict[str, Any],
    data: dict[str, Any],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    """Build an SBOM-like reproducibility manifest for benchmark assembly outputs."""
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    existing_outputs: dict[str, Any] = {}
    for name, p in output_paths.items():
        if name == "html":
            existing_outputs[name] = {
                "path": str(p),
                "sha256": None,
                "note": "The HTML embeds this SBOM section, so its own hash is intentionally not included to avoid self-reference.",
            }
        else:
            existing_outputs[name] = _sha256_file(p)

    return _safe_jsonable(
        {
            "bom_format": "HUGIML benchmark reproducibility manifest",
            "bom_version": "1.0",
            "schema_hint": "SBOM-like JSON; intentionally lightweight rather than SPDX/CycloneDX-complete.",
            "command": {
                "argv": list(sys.argv),
                "executable": sys.executable,
                "working_directory": str(Path.cwd().resolve()),
            },
            "benchmark": {
                "script": _sha256_file(Path(__file__).resolve()),
                "script_dir": str(SCRIPT_DIR),
                "source_root": str(SOURCE_ROOT) if SOURCE_ROOT is not None else None,
                "checkpoint_input": _sha256_file(checkpoint),
                "template_html_input": _sha256_file(template_html),
                "out_dir": str(out_dir),
                "metadata": metadata,
                "grid_snapshot": grid_snapshot(),
                "dataset_names": list(DATASET_NAMES),
                "model_order": list(MODEL_ORDER),
                "hugiml_scenarios": HUGIML_SCENARIOS,
                "data_object_counts": {
                    "details": len(data.get("details", [])),
                    "overall": len(data.get("overall", [])),
                    "summary_by_scope": len(data.get("summary_by_scope", [])),
                    "scope_tests": len(data.get("scope_tests", [])),
                },
            },
            "artifacts": existing_outputs,
            "source_fingerprints": _source_tree_fingerprint(SOURCE_ROOT),
            "git": _git_metadata(SOURCE_ROOT),
            "python_runtime": {
                "version": sys.version,
                "version_info": list(sys.version_info),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "system": platform.system(),
                "release": platform.release(),
                "python_build": platform.python_build(),
                "python_compiler": platform.python_compiler(),
                "sys_path_sample": list(sys.path[:20]),
            },
            "environment": {
                "selected_variables": {
                    k: os.environ.get(k)
                    for k in sorted(
                        {
                            "HUGIML_SOURCE_ROOT",
                            "PYTHONPATH",
                            "PYTHONHASHSEED",
                            "OMP_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "NUMEXPR_NUM_THREADS",
                            "JOBLIB_TEMP_FOLDER",
                            "SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL",
                            "CC",
                            "CXX",
                            "CFLAGS",
                            "CXXFLAGS",
                            "CPPFLAGS",
                            "LDFLAGS",
                            "CMAKE_ARGS",
                            "CMAKE_BUILD_PARALLEL_LEVEL",
                        }
                    )
                    if os.environ.get(k) is not None
                }
            },
            "packages": {
                "hugiml": _hugiml_module_metadata(),
                "installed_distributions": _installed_python_distributions(),
                "pip_freeze_all": _pip_freeze(),
            },
            "native_build": _native_build_metadata(),
        }
    )


def _ensure_dashboard_scope_summary_ui(text: str) -> str:
    if "scopeSummaryTables" in text:
        return text
    html = """
<section class="card" id="scopeSummaryCard" style="margin-top:16px">
  <div class="section-title"><h2>Summary results by dataset scope</h2><div class="meta">Overall, public real-world/package datasets, and reproducible synthetic datasets.</div></div>
  <div id="scopeSummaryTables"></div>
</section>
"""
    js = r"""
<script>
(function(){
  function esc(v){return String(v ?? '').replace(/[&<>"']/g, ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));}
  function fmt(v, digits=4){if(v==null || !Number.isFinite(Number(v))) return ''; return new Intl.NumberFormat('en-US',{maximumFractionDigits:digits}).format(Number(v));}
  function fmt2(v){return fmt(v,2);}
  function renderScopeSummary(){
    const root=document.getElementById('scopeSummaryTables');
    if(!root) return;
    const data=(window.DATA||DATA||{});
    const rows=data.summary_by_scope || [];
    const tests=data.scope_tests || [];
    const scopes=['Overall','Real-world','Synthetic'].filter(s=>rows.some(r=>r.scope===s));
    root.innerHTML=scopes.map(scope=>{
      const scopeRows=rows.filter(r=>r.scope===scope).slice().sort((a,b)=>(Number(b.mean_auc||-1)-Number(a.mean_auc||-1)) || (Number(a.mean_rank||999)-Number(b.mean_rank||999)));
      const t=tests.find(x=>x.scope===scope)||{};
      const meta=`datasets=${t.n_datasets ?? ''} · best mean AUC=${esc(t.best_mean_auc_model||'')}${t.best_mean_auc==null?'':' ('+fmt(t.best_mean_auc)+')'} · HUGIML mean AUC=${fmt(t.hugiml_mean_auc)} · Friedman p=${fmt(t.friedman_p)}`;
      return `<div style="margin-top:14px"><h3 style="margin:8px 0 4px 0">${esc(scope)}</h3><div class="meta">${meta}</div><div style="overflow:auto;margin-top:8px"><table><thead><tr><th>Model</th><th>n</th><th>Mean AUC</th><th>Median AUC</th><th>Mean F1</th><th>Mean accuracy</th><th>Mean rank</th><th>Strict wins</th><th>Tied best</th><th>Mean model inspection units</th><th>Mean instance inspection units (95% CI)</th><th>Mean fit s</th></tr></thead><tbody>` +
        scopeRows.map(r=>`<tr><td>${esc(r.model)}</td><td>${esc(r.n_datasets)}</td><td>${fmt(r.mean_auc)}</td><td>${fmt(r.median_auc)}</td><td>${fmt(r.mean_f1)}</td><td>${fmt(r.mean_accuracy)}</td><td>${fmt2(r.mean_rank)}</td><td>${esc(r.strict_wins)}</td><td>${esc(r.tied_best_count)}</td><td>${fmt2(r.mean_model_inspection_units ?? r.mean_complexity)}</td><td>${r.mean_instance_inspection_units==null?'':fmt2(r.mean_instance_inspection_units)+' ('+fmt2(r.instance_inspection_ci_lower)+'–'+fmt2(r.instance_inspection_ci_upper)+')'}</td><td>${fmt(r.mean_fit_seconds)}</td></tr>`).join('') +
        `</tbody></table></div></div>`;
    }).join('');
  }
  window.renderScopeSummary=renderScopeSummary;
  window.addEventListener('load', renderScopeSummary);
  setTimeout(renderScopeSummary,0);
})();
</script>
"""
    return _inject_before_body_end(text, html + js)


def _ensure_dashboard_profile_ui(text: str) -> str:
    if "datasetProfileTable" in text:
        return text
    html = """
<section class="card" id="datasetProfileCard" style="margin-top:16px">
  <div class="section-title"><h2>Dataset column profile</h2><div class="meta">Select a dataset to inspect feature dtypes, missingness, uniqueness, and summary statistics.</div></div>
  <label for="datasetProfileSelect" style="font-weight:600;margin-right:8px">Dataset</label>
  <select id="datasetProfileSelect" style="max-width:420px;padding:6px 8px;margin:8px 0 12px 0"></select>
  <div id="datasetProfileMeta" class="meta"></div>
  <div style="overflow:auto;max-height:560px;margin-top:10px"><table id="datasetProfileTable"></table></div>
</section>
"""
    js = """
<script>
(function(){
  function esc(v){return String(v ?? '').replace(/[&<>"']/g, ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));}
  function fmt(v, digits=2){if(v==null || !Number.isFinite(Number(v))) return ''; return new Intl.NumberFormat('en-US',{maximumFractionDigits:digits}).format(Number(v));}
  function renderDatasetProfile(){
    const profiles=(window.DATA||DATA||{}).dataset_profiles || {};
    const sel=document.getElementById('datasetProfileSelect');
    const tbl=document.getElementById('datasetProfileTable');
    const meta=document.getElementById('datasetProfileMeta');
    if(!sel || !tbl) return;
    const names=Object.keys(profiles);
    if(!sel.dataset.loaded){
      sel.innerHTML=names.map(n=>`<option value="${esc(n)}">${esc(n)}</option>`).join('');
      sel.dataset.loaded='1';
      sel.addEventListener('change', renderDatasetProfile);
    }
    const name=sel.value || names[0];
    const p=profiles[name] || {};
    if(meta){meta.textContent=p.error ? p.error : `${p.dataset_group||''} · rows=${p.n_rows ?? ''} · features=${p.n_features ?? ''} · positive rate=${p.class_balance==null?'':fmt(p.class_balance,3)}`;}
    const rows=p.columns || [];
    tbl.innerHTML='<thead><tr><th>Column</th><th>Data type</th><th>Missing</th><th>Missing %</th><th>Unique</th><th>Summary stats / top values</th></tr></thead><tbody>' +
      rows.map(r=>`<tr><td>${esc(r.column)}</td><td>${esc(r.dtype)}</td><td>${esc(r.missing)}</td><td>${r.missing_pct==null?'':fmt(100*Number(r.missing_pct),2)+'%'}</td><td>${esc(r.unique)}</td><td>${esc(r.summary)}</td></tr>`).join('') + '</tbody>';
  }
  window.renderDatasetProfile=renderDatasetProfile;
  window.addEventListener('load', renderDatasetProfile);
  setTimeout(renderDatasetProfile, 0);
})();
</script>
"""
    return _inject_before_body_end(text, html + js)


def _remove_dashboard_profile_ui(text: str) -> str:
    """Remove the optional dataset column-profile card and its runtime code."""
    marker = 'id="datasetProfileCard"'
    marker_pos = text.find(marker)
    if marker_pos >= 0:
        section_start = text.rfind("<section", 0, marker_pos)
        section_end = text.find("</section>", marker_pos)
        if section_start >= 0 and section_end >= 0:
            text = text[:section_start] + text[section_end + len("</section>") :]

    marker = "function renderDatasetProfile(){"
    marker_pos = text.find(marker)
    if marker_pos >= 0:
        script_start = text.rfind("<script>", 0, marker_pos)
        script_end = text.find("</script>", marker_pos)
        if script_start >= 0 and script_end >= 0:
            text = text[:script_start] + text[script_end + len("</script>") :]
    return text


def _ensure_dashboard_complexity_rendering(text: str) -> str:
    js = """
<script>
(function(){
  function numericValues(arr){return (arr||[]).map(Number).filter(v=>Number.isFinite(v)&&v>0);}
  function adjustOne(gd){
    if(!gd || !gd.data || !window.Plotly) return;
    const id=(gd.id||'').toLowerCase();
    const title=((gd.layout&&gd.layout.title&&gd.layout.title.text)||'').toLowerCase();
    if(!(id.includes('complex') || title.includes('complex'))) return;
    let xs=[]; gd.data.forEach(t=>{xs=xs.concat(numericValues(t.x));});
    const update={};
    if(xs.length){
      const min=Math.min(...xs), max=Math.max(...xs);
      if(max>min){
        update['xaxis.range']=[Math.log10(min)-0.18, Math.log10(max)+0.18];
        update['xaxis.type']='log';
      }
    }
    update['margin.l']=Math.max(75, (gd.layout&&gd.layout.margin&&gd.layout.margin.l)||0);
    update['margin.r']=Math.max(90, (gd.layout&&gd.layout.margin&&gd.layout.margin.r)||0);
    update['margin.b']=Math.max(80, (gd.layout&&gd.layout.margin&&gd.layout.margin.b)||0);
    update['xaxis.automargin']=true; update['yaxis.automargin']=true; update['yaxis.range']=[0,1.03];
    Plotly.restyle(gd, {'cliponaxis': false, 'marker.size': 12, 'marker.line.width': 1});
    Plotly.relayout(gd, update);
    Plotly.Plots.resize(gd);
  }
  function adjustComplexityCharts(){document.querySelectorAll('.js-plotly-plot,[id*=complex i]').forEach(adjustOne);}
  window.adjustComplexityCharts=adjustComplexityCharts;
  window.addEventListener('load', function(){setTimeout(adjustComplexityCharts, 50); setTimeout(adjustComplexityCharts, 500);});
  if(window.ResizeObserver){new ResizeObserver(()=>setTimeout(adjustComplexityCharts,50)).observe(document.body);}
})();
</script>
"""
    if "adjustComplexityCharts" in text:
        return text
    # Also make clipping less likely in templates that set plot containers too tightly.
    if "</style>" in text and ".plot{overflow:visible" not in text:
        text = text.replace(
            "</style>",
            ".plot{overflow:visible}.js-plotly-plot .scatterlayer .trace path{vector-effect:non-scaling-stroke}\n</style>",
        )
    return _inject_before_body_end(text, js)


def _ensure_dashboard_scenario_ui(text: str, data: dict[str, Any]) -> str:
    scenarios = data.get("dashboard_scenarios") or []
    if len(scenarios) < 2:
        return text

    options = "".join(
        f'<option value="{html.escape(str(item.get("id", "")))}">'
        f'{html.escape(str(item.get("label", item.get("id", ""))))}</option>'
        for item in scenarios
    )
    select_markup = (
        '<div class="control scenario-control"><label>HUGIML path</label>'
        f'<select id="hugimlScenarioSelect">{options}</select></div>'
    )

    text = re.sub(
        r'<div class="scenario-panes" id="hugimlScenarioPanes".*?</div>',
        '',
        text,
        flags=re.S,
    )
    text = re.sub(
        r'<div class="control scenario-control"><label>HUGIML path</label>'
        r'<select id="hugimlScenarioSelect">.*?</select></div>',
        select_markup,
        text,
        count=1,
        flags=re.S,
    )
    if 'id="hugimlScenarioSelect"' not in text:
        if '<div class="theme-switcher">' in text:
            text = text.replace(
                '<div class="theme-switcher">',
                '<div class="theme-switcher">' + select_markup,
                1,
            )
        elif '<section class="hero">' in text:
            text = text.replace('<section class="hero">', '<section class="hero">' + select_markup, 1)

    text = re.sub(
        r'\.scenario-panes\{.*?@media\(max-width:760px\)\{\.theme-switcher.*?\}\}',
        '',
        text,
        flags=re.S,
    )

    runtime = r"""
<script>
(function(){
  if (typeof DASHBOARD_SCENARIOS === 'undefined' || !Array.isArray(DASHBOARD_SCENARIOS) || DASHBOARD_SCENARIOS.length < 2) return;
  window.DATA = DATA;
  function clone(obj){ return JSON.parse(JSON.stringify(obj)); }
  function fmt(v,d=4){
    const n=Number(v);
    if(v==null || !Number.isFinite(n)) return '';
    if(Math.abs(n)>0 && Math.abs(n)<Math.pow(10,-d)) return n.toExponential(2);
    return n.toFixed(d).replace(/0+$/,'').replace(/\.$/,'');
  }
  function setStat(label,value,sub){
    document.querySelectorAll('.card.stat').forEach(card=>{
      const lab=card.querySelector('.label');
      if(lab && lab.textContent.trim()===label){
        const val=card.querySelector('.value');
        const detail=card.querySelector('.sub');
        if(val) val.textContent=value;
        if(detail) detail.textContent=sub;
      }
    });
  }
  function significantCount(rows){return (rows||[]).filter(row=>row.significant_holm_0_05).length;}
  function setSignificantPairsMeta(){
    const rows=DATA.pairwise_all||[];
    const sig=significantCount(rows);
    document.querySelectorAll('.section-title').forEach(section=>{
      const h2=section.querySelector('h2');
      const meta=section.querySelector('.meta');
      if(h2 && meta && h2.textContent.trim()==='Significant all-pair comparisons'){
        meta.textContent=`${sig} significant pair(s) after Holm correction across all ${rows.length} pairwise tests`;
      }
    });
  }
  function updateNarrative(){
    const overall=(DATA.overall||[]).slice();
    if(!overall.length) return;
    const best=overall.slice().sort((a,b)=>(Number(b.mean_auc||-1)-Number(a.mean_auc||-1))||(Number(a.mean_rank||999)-Number(b.mean_rank||999)))[0];
    const hug=overall.find(row=>row.model==='HUGIML')||{};
    const bestRank=overall.slice().sort((a,b)=>(Number(a.mean_rank||999)-Number(b.mean_rank||999))||(Number(b.mean_auc||-1)-Number(a.mean_auc||-1)))[0];
    const p=Number((((DATA.global||[])[0]||{}).p_value));
    const budgetOver=(DATA.budget||[]).reduce((sum,row)=>sum+Number(row.n_over_budget||0),0);
    const vsRows=DATA.vs_hugiml||[];
    const sig=significantCount(vsRows);
    setStat('Best mean AUC',fmt(best.mean_auc),best.model||'');
    const modelInspection=fmt(hug.mean_model_inspection_units ?? hug.mean_complexity,1);
    const instanceInspection=fmt(hug.mean_instance_inspection_units,1);
    const inspectionSummary=instanceInspection
      ? `mean model inspection units: ${modelInspection} · mean instance inspection units: ${instanceInspection}`
      : `mean model inspection units: ${modelInspection}`;
    setStat('HUGIML mean AUC',fmt(hug.mean_auc),inspectionSummary);
    setStat('Best mean rank',fmt(bestRank.mean_rank,2),bestRank.model||'');
    setStat('Global rank test',fmt(p),Number.isFinite(p)&&p<0.05?'Significant global rank differences among models':'No significant global rank difference detected');
    setStat('Pairwise vs HUGIML',sig ? `${sig} significant` : 'None significant',`HUGIML vs ${vsRows.length} baselines, Holm-adjusted`);
    setStat('Budget violations',String(budgetOver),'budgeted fits exceeded __TREE_BUDGET__ model inspection units');
    document.querySelectorAll('.chip').forEach(chip=>{
      if(chip.textContent.includes('Friedman p-value:')) chip.textContent='Friedman p-value: '+fmt(p);
    });
    document.querySelectorAll('.callout').forEach(callout=>{
      if(callout.textContent.includes('Global comparison:')) callout.innerHTML='<strong>Global comparison:</strong> The Friedman test shows '+(Number.isFinite(p)&&p<0.05?'significant global rank differences among models':'no significant global rank difference at alpha 0.05')+'. It is a global rank test and does not identify which model pairs differ.';
      if(callout.textContent.includes('HUGIML-specific comparison:')) callout.innerHTML='<strong>HUGIML-specific comparison:</strong> '+(sig ? `${sig} of ${vsRows.length} HUGIML-vs-baseline comparisons are significant after Holm correction.` : `No HUGIML-vs-baseline comparison is significant after Holm correction across the ${vsRows.length} focused tests.`);
    });
    setSignificantPairsMeta();
  }
  function rerender(){
    updateNarrative();
    if(typeof refreshAll==='function') refreshAll();
    if(typeof renderScopeSummary==='function') renderScopeSummary();
    if(typeof renderProfiles==='function') renderProfiles();
    if(typeof renderDatasetProfile==='function') renderDatasetProfile();
    if(typeof renderInternalRpte==='function') renderInternalRpte();
    if(typeof adjustComplexityCharts==='function') setTimeout(adjustComplexityCharts,0);
  }
  function applyScenario(id){
    const entry=DASHBOARD_SCENARIOS.find(item=>item.id===id) || DASHBOARD_SCENARIOS[0];
    Object.keys(DATA).forEach(key=>delete DATA[key]);
    Object.assign(DATA, clone(entry.data));
    DATA.dashboard_scenarios = DASHBOARD_SCENARIOS;
    DATA.default_hugiml_scenario = entry.id;
    DATA.active_hugiml_scenario = entry.id;
    DATA.active_hugiml_scenario_label = entry.label;
    DATA.active_hugiml_grid_name = entry.grid_name;
    window.DATA = DATA;
    const select=document.getElementById('hugimlScenarioSelect');
    if(select) select.value=entry.id;
    rerender();
  }
  const select=document.getElementById('hugimlScenarioSelect');
  if(select){
    const desired=DATA.default_hugiml_scenario || DATA.active_hugiml_scenario || DASHBOARD_SCENARIOS[0].id;
    select.value=desired;
    select.addEventListener('change',()=>applyScenario(select.value));
    if(DATA.active_hugiml_scenario !== desired) applyScenario(desired); else updateNarrative();
  } else {
    updateNarrative();
  }
})();
</script>
"""
    runtime = runtime.replace("__TREE_BUDGET__", f"{BUDGET:g}")
    marker = "if (typeof DASHBOARD_SCENARIOS === 'undefined'"
    marker_pos = text.rfind(marker)
    if marker_pos >= 0:
        script_start = text.rfind('<script>', 0, marker_pos)
        script_end = text.find('</script>', marker_pos)
        if script_start >= 0 and script_end >= 0:
            text = text[:script_start] + runtime + text[script_end + len('</script>'):]
    else:
        text = _inject_before_body_end(text, runtime)
    return text


def _ensure_dashboard_rpte_ui(text: str, data: dict[str, Any]) -> str:
    scenarios = data.get("dashboard_scenarios") or []
    available = any(
        (item.get("data", {}) or {}).get("hugiml_rpte_distributions")
        for item in scenarios
        if isinstance(item, dict)
    ) or bool(data.get("hugiml_rpte_distributions"))
    if not available:
        return text
    text = re.sub(
        r'<!-- BEGIN INTERNAL_RPTE_DASHBOARD -->.*?<!-- END INTERNAL_RPTE_DASHBOARD -->',
        '', text, flags=re.S,
    )
    section = r"""
<!-- BEGIN INTERNAL_RPTE_DASHBOARD -->
<style>
.rpte-dashboard{margin-top:16px;color:var(--ink,#edf2ff)}
.rpte-dashboard h2,.rpte-dashboard strong,.rpte-dashboard td{color:var(--ink,#edf2ff)}
.rpte-dashboard .meta,.rpte-dashboard .rpte-note,.rpte-dashboard th{color:var(--muted,#aeb9d4)}
.rpte-dashboard .rpte-summary-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:14px 0}
.rpte-dashboard .rpte-summary-card{background:var(--panel2,#18223d);border:1px solid var(--border,#2a3658);border-radius:12px;padding:14px}
.rpte-dashboard .rpte-summary-card .value{color:var(--accent2,#86efcf);font-size:1.45rem;font-weight:750;margin-top:4px}
.rpte-dashboard .rpte-controls{display:flex;align-items:flex-end;gap:12px;flex-wrap:wrap;margin:12px 0}
.rpte-dashboard .rpte-control{display:flex;flex-direction:column;gap:5px;min-width:240px}
.rpte-dashboard .rpte-control label{color:var(--muted,#aeb9d4);font-size:.78rem;font-weight:650}
.rpte-dashboard .rpte-control select{color:var(--ink,#edf2ff);background:var(--panel2,#18223d);border:1px solid var(--border,#2a3658);border-radius:8px;padding:8px 10px;font:inherit}
.rpte-dashboard .rpte-control select option{color:var(--ink,#edf2ff);background:var(--panel2,#18223d)}
.rpte-dashboard .rpte-table-scroll{max-height:min(54vh,560px);overflow:auto;border:1px solid var(--border,#2a3658);border-radius:12px;background:var(--panel,#121a31);scrollbar-gutter:stable}
.rpte-dashboard table{width:100%;border-collapse:collapse;font-size:.82rem;background:transparent;color:var(--ink,#edf2ff)}
.rpte-dashboard th,.rpte-dashboard td{padding:8px 10px;border-bottom:1px solid var(--border,#2a3658);text-align:right;white-space:nowrap;background:transparent}
.rpte-dashboard thead th{position:sticky;top:0;z-index:2;background:var(--panel2,#18223d);box-shadow:0 1px 0 var(--border,#2a3658)}
.rpte-dashboard th:first-child,.rpte-dashboard td:first-child{text-align:left}
.rpte-dashboard code{color:var(--accent,#82aaff);background:var(--panel2,#18223d);padding:1px 4px;border-radius:4px}
@media(max-width:760px){.rpte-dashboard .rpte-summary-grid{grid-template-columns:1fr}}
</style>
<section class="card rpte-dashboard" id="internalRpteDashboard">
  <div class="section-title"><h2>RPTE path and structural distributions</h2><div class="meta" id="rpteScenarioLabel"></div></div>
  <div class="rpte-controls"><div class="rpte-control"><label for="rpteScenarioSelect">HUGIML path shown in this section</label><select id="rpteScenarioSelect" aria-label="Select HUGIML path for RPTE statistics"></select></div></div>
  <div class="rpte-summary-grid">
    <article class="rpte-summary-card"><div class="meta">RPTE-selected folds</div><div class="value" id="rpteSelectedFolds">—</div></article>
    <article class="rpte-summary-card"><div class="meta">Direct LR folds</div><div class="value" id="rpteLrFolds">—</div></article>
    <article class="rpte-summary-card"><div class="meta">Datasets activating RPTE</div><div class="value" id="rpteDatasetCount">—</div></article>
  </div>
  <div class="rpte-table-scroll"><table id="internalRpteTable"></table></div>
  <p class="rpte-note">Values are recorded during each fitted outer fold. Distributions show median [Q25, Q75] for inputs, generated trees, active trees, leaves, active leaves, active direct terms and leaf-path length.</p>
</section>
<script>
(function(){
  function num(v){const n=Number(v);return Number.isFinite(n)?n:null}
  function fmt(v,d=1){const n=num(v);return n==null?'—':n.toFixed(d).replace(/\.0+$/,'')}
  function range(row,p,d=1){return num(row[p+'_n'])?`${fmt(row[p+'_median'],d)} [${fmt(row[p+'_q25'],d)}, ${fmt(row[p+'_q75'],d)}]`:'—'}
  window.renderInternalRpte=function(){
    const rows=DATA.hugiml_rpte_distributions||[];
    const scenarioSelect=document.getElementById('rpteScenarioSelect');
    if(scenarioSelect && typeof DASHBOARD_SCENARIOS!=='undefined'){
      if(!scenarioSelect.options.length){
        DASHBOARD_SCENARIOS.forEach(item=>scenarioSelect.add(new Option(item.label||item.id,item.id)));
        scenarioSelect.addEventListener('change',()=>{
          const mainSelect=document.getElementById('hugimlScenarioSelect');
          if(mainSelect){mainSelect.value=scenarioSelect.value;mainSelect.dispatchEvent(new Event('change',{bubbles:true}));}
        });
      }
      scenarioSelect.value=DATA.active_hugiml_scenario||DATA.default_hugiml_scenario||'';
    }
    const total=rows.reduce((s,r)=>s+Number(r.split_count||0),0);
    const lr=rows.reduce((s,r)=>s+Number(r.lr_split_count||0),0);
    const rpte=rows.reduce((s,r)=>s+Number(r.rpte_split_count||0),0);
    const active=rows.filter(r=>Number(r.rpte_split_count||0)>0).length;
    const label=document.getElementById('rpteScenarioLabel');if(label)label.textContent=DATA.active_hugiml_scenario_label||'';
    const a=document.getElementById('rpteSelectedFolds');if(a)a.textContent=`${rpte} / ${total}`;
    const b=document.getElementById('rpteLrFolds');if(b)b.textContent=String(lr);
    const c=document.getElementById('rpteDatasetCount');if(c)c.textContent=String(active);
    const headers=['Dataset','Splits','LR','RPTE','Sequential estimators','Lookahead estimators','Inputs','Trees','Active trees','Leaves','Active leaves','Direct terms','Average active leaf path'];
    const body=rows.map(r=>`<tr><td>${String(r.dataset??'')}</td><td>${r.split_count}</td><td>${r.lr_split_count}</td><td>${r.rpte_split_count}</td><td>${r.rpte_sequential_estimator_count}</td><td>${r.rpte_lookahead_estimator_count}</td><td>${range(r,'rpte_inputs_passed')}</td><td>${range(r,'rpte_tree_count')}</td><td>${range(r,'rpte_active_tree_count')}</td><td>${range(r,'rpte_leaf_count')}</td><td>${range(r,'rpte_active_leaf_count')}</td><td>${range(r,'rpte_direct_term_count')}</td><td>${range(r,'rpte_active_average_leaf_path_length',2)}</td></tr>`).join('');
    const table=document.getElementById('internalRpteTable');if(table)table.innerHTML=`<thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${body}</tbody>`;
  };
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',window.renderInternalRpte);else window.renderInternalRpte();
})();
</script>
<!-- END INTERNAL_RPTE_DASHBOARD -->
"""
    return _inject_before_body_end(text, section)


def _remove_embedded_reproducibility_section(text: str) -> str:
    return re.sub(
        r'<!-- BEGIN HUGIML_REPRODUCIBILITY_SBOM -->.*?'
        r'<!-- END HUGIML_REPRODUCIBILITY_SBOM -->',
        '',
        text,
        flags=re.S,
    )


def _remove_embedded_methodology_section(text: str) -> str:
    return re.sub(
        r'<!-- BEGIN HUGIML_METHODOLOGY -->.*?'
        r'<!-- END HUGIML_METHODOLOGY -->',
        '',
        text,
        flags=re.S,
    )


def _methodology_section_html(methodology: dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value), quote=True)

    def bullet_list(items: list[Any]) -> str:
        return '<ul>' + ''.join(f'<li>{esc(item)}</li>' for item in items) + '</ul>'

    model_cards = []
    for model in methodology.get('models', []):
        parameter_rows = ''.join(
            '<tr>'
            f'<th scope="row">{esc(row.get("parameter", ""))}</th>'
            f'<td>{esc(", ".join(str(value) for value in row.get("values", [])))}</td>'
            '</tr>'
            for row in model.get('parameters', [])
        )
        model_cards.append(
            '<article class="methodology-model">'
            f'<h4>{esc(model.get("model", ""))}</h4>'
            f'<div class="methodology-meta">Search space: {esc(model.get("grid_name", ""))} · '
            f'{esc(model.get("candidate_count", 0))} candidate configurations</div>'
            '<div class="methodology-table-wrap"><table class="methodology-table">'
            '<thead><tr><th>Parameter</th><th>Values considered</th></tr></thead>'
            f'<tbody>{parameter_rows}</tbody></table></div>'
            f'<p><strong>Constant settings:</strong> {esc("; ".join(model.get("constant_settings", [])))}</p>'
            f'<p><strong>Complexity definition:</strong> {esc(model.get("complexity", ""))}</p>'
            '</article>'
        )

    # The methodology cards are intentionally arranged as two five-model rows.
    # This ordering is local to the methodology section and does not alter the
    # model ordering used by charts, tables, rankings, or exported results.
    methodology_rows = [model_cards[:5], model_cards[5:10]]
    model_rows_html = ''.join(
        '<div class="methodology-row-scroll"><div class="methodology-model-grid">'
        + ''.join(row)
        + '</div></div>'
        for row in methodology_rows
        if row
    )

    return f"""<!-- BEGIN HUGIML_METHODOLOGY -->
<style>
.methodology-card{{margin-top:18px}}
.methodology-card details{{border:0}}
.methodology-card summary{{cursor:pointer;font-weight:750;font-size:1.08rem;list-style-position:outside;padding:4px 0}}
.methodology-card summary::marker{{color:var(--accent,#2563eb)}}
.methodology-content{{padding-top:14px}}
.methodology-content h3{{margin:20px 0 8px;font-size:1rem}}
.methodology-content h4{{margin:0 0 4px;font-size:.98rem}}
.methodology-content ul{{margin:8px 0 0;padding-left:22px}}
.methodology-content li{{margin:5px 0;line-height:1.45}}
.methodology-row-scroll{{overflow-x:auto;margin-top:12px;padding-bottom:4px}}
.methodology-model-grid{{display:grid;grid-template-columns:repeat(5,minmax(280px,1fr));gap:14px;min-width:1456px}}
.methodology-model{{border:1px solid var(--border,#d9e1ec);border-radius:12px;padding:14px;background:var(--panel,#fff)}}
.methodology-meta{{font-size:.82rem;color:var(--muted,#64748b);margin-bottom:10px}}
.methodology-table-wrap{{overflow-x:auto}}
.methodology-table{{width:100%;border-collapse:collapse;font-size:.82rem}}
.methodology-table th,.methodology-table td{{padding:6px 8px;border-bottom:1px solid var(--border,#e2e8f0);text-align:left;vertical-align:top}}
.methodology-table th[scope=row]{{width:36%;font-weight:650}}
.methodology-model p{{font-size:.82rem;line-height:1.45;margin:10px 0 0}}
@media(max-width:720px){{.methodology-model-grid{{grid-template-columns:repeat(5,minmax(250px,1fr));min-width:1306px}}}}
</style>
<section class="card methodology-card">
  <details id="benchmarkMethodology">
    <summary>{esc(methodology.get('title', 'Methodology and parameter search space'))}</summary>
    <div class="methodology-content">
      <h3>Cross-validation protocol</h3>
      {bullet_list(list(methodology.get('protocol', [])))}
      <h3>Preprocessing and timing</h3>
      {bullet_list(list(methodology.get('preprocessing', [])))}
      <h3>Model search spaces</h3>
      {model_rows_html}
    </div>
  </details>
</section>
<!-- END HUGIML_METHODOLOGY -->"""

def render_html(
    template_html: Path,
    data: dict[str, Any],
    out_html: Path,
) -> None:
    text = (
        template_html.read_text(errors="ignore")
        if template_html.exists()
        else _default_dashboard_template()
    )
    text = _remove_embedded_reproducibility_section(text)
    text = _remove_embedded_methodology_section(text)
    if "const DATA=" not in text or ";\nconst MODEL_ORDER" not in text:
        text = _default_dashboard_template()
    data_start = text.index("const DATA=")
    scenario_start = text.rfind("const DASHBOARD_SCENARIOS=", 0, data_start)
    start = scenario_start if scenario_start >= 0 else data_start
    end = text.index(";\nconst MODEL_ORDER", data_start)
    safe_data = _safe_jsonable(data)
    scenario_payload = safe_data.get("dashboard_scenarios", []) if isinstance(safe_data, dict) else []
    if isinstance(safe_data, dict):
        safe_data.pop("dataset_profiles", None)
    for scenario in scenario_payload:
        scenario_data = scenario.get("data", {}) if isinstance(scenario, dict) else {}
        if isinstance(scenario_data, dict):
            scenario_data.pop("dataset_profiles", None)

    if scenario_payload:
        default_scenario = str(
            safe_data.get("default_hugiml_scenario")
            or safe_data.get("active_hugiml_scenario")
            or scenario_payload[0].get("id", "")
        )
        scenarios_json = json.dumps(scenario_payload, separators=(",", ":"), allow_nan=False)
        default_json = json.dumps(default_scenario, separators=(",", ":"), allow_nan=False)
        new_blob = (
            "const DASHBOARD_SCENARIOS="
            + scenarios_json
            + ";\nconst DEFAULT_DASHBOARD_SCENARIO="
            + default_json
            + ";\nconst DATA=JSON.parse(JSON.stringify((DASHBOARD_SCENARIOS.find("
            "item=>item.id===DEFAULT_DASHBOARD_SCENARIO)||DASHBOARD_SCENARIOS[0]).data));\n"
            "DATA.dashboard_scenarios=DASHBOARD_SCENARIOS;\n"
            "DATA.default_hugiml_scenario=DEFAULT_DASHBOARD_SCENARIO"
        )
    else:
        new_blob = "const DATA=" + json.dumps(
            safe_data, separators=(",", ":"), allow_nan=False
        )
    text = text[:start] + new_blob + text[end:]

    text = text.replace(
        "function fmt(v,d=4){return v==null||Number.isNaN(v)?'':Number(v).toFixed(d)}",
        "function fmt(v,d=4){const n=Number(v);if(v==null||!Number.isFinite(n))return '';if(Math.abs(n)>0&&Math.abs(n)<Math.pow(10,-d))return n.toExponential(2);return n.toFixed(d).replace(/0+$/,'').replace(/\\.$/,'')}",
    )

    # Keep the JavaScript model list aligned with the Python benchmark model order.
    mo_start = text.find("const MODEL_ORDER=")
    if mo_start >= 0:
        mo_end = text.find(";", mo_start)
        if mo_end >= 0:
            text = (
                text[:mo_start]
                + "const MODEL_ORDER="
                + json.dumps(MODEL_ORDER, separators=(",", ":"))
                + text[mo_end:]
            )

    overall = pd.DataFrame(data["overall"])
    best = overall.sort_values(["mean_auc", "mean_rank"], ascending=[False, True]).iloc[0]
    hug = overall.loc[overall["model"] == "HUGIML"].iloc[0]
    best_rank = overall.sort_values(["mean_rank", "mean_auc"], ascending=[True, False]).iloc[0]
    p = float(data["global"][0]["p_value"])
    p_text = f"{p:.4g}"
    budget_over = int(sum(int(r.get("n_over_budget", 0)) for r in data["budget"]))
    vs_rows = list(data.get("vs_hugiml", []))
    pairwise_rows = list(data.get("pairwise_all", []))
    hug_sig_rows = [r for r in vs_rows if bool(r.get("significant_holm_0_05"))]
    sig_pair_rows = [r for r in pairwise_rows if bool(r.get("significant_holm_0_05"))]
    sig_vs = bool(hug_sig_rows)
    vs_total = len(vs_rows)
    pairwise_total = len(pairwise_rows)
    sig_pair_count = len(sig_pair_rows)
    hug_sig_count = len(hug_sig_rows)
    vs_text = f"{hug_sig_count} significant" if sig_vs else "None significant"
    vs_sub = f"HUGIML vs {vs_total} baselines, Holm-adjusted"
    global_text = (
        "Significant global rank differences among models"
        if p < 0.05
        else "No significant global rank difference detected"
    )
    interp = (
        "significant global rank differences among models"
        if p < 0.05
        else "no significant global rank difference at alpha 0.05"
    )
    hug_interp = (
        f"{hug_sig_count} of {vs_total} HUGIML-vs-baseline comparisons are significant after Holm correction."
        if sig_vs
        else f"No HUGIML-vs-baseline comparison is significant after Holm correction across the {vs_total} focused tests."
    )
    all_pair_meta = f"{sig_pair_count} significant pair(s) after Holm correction across all {pairwise_total} pairwise tests"
    all_pair_note = (
        "Rows below are the statistically significant differences among all model pairs after Holm correction. "
        "This table can include HUGIML pairs if they are significant; use the HUGIML-focused comparison above for only HUGIML-vs-baseline tests."
    )

    model_inspection_text = f"mean model inspection units: {float(hug.mean_model_inspection_units):.1f}"
    if pd.notna(hug.mean_instance_inspection_units):
        inspection_summary = (
            f"{model_inspection_text}; mean instance inspection units: "
            f"{float(hug.mean_instance_inspection_units):.1f}"
        )
    else:
        inspection_summary = model_inspection_text

    replacements = {
        r"Friedman p-value: [^<]+": f"Friedman p-value: {p_text}",
        r'<div class="label">Best mean AUC</div><div class="value">[^<]+</div><div class="sub">[^<]+</div>': f'<div class="label">Best mean AUC</div><div class="value">{best.mean_auc:.4f}</div><div class="sub">{best.model}</div>',
        r'<div class="label">HUGIML mean AUC</div><div class="value">[^<]+</div><div class="sub">[^<]+</div>': f'<div class="label">HUGIML mean AUC</div><div class="value">{hug.mean_auc:.4f}</div><div class="sub">{inspection_summary}</div>',
        r'<div class="label">Best mean rank</div><div class="value">[^<]+</div><div class="sub">[^<]+</div>': f'<div class="label">Best mean rank</div><div class="value">{best_rank.mean_rank:.2f}</div><div class="sub">{best_rank.model}</div>',
        r'<div class="label">Global rank test</div><div class="value">[^<]+</div><div class="sub">[^<]+</div>': f'<div class="label">Global rank test</div><div class="value">{p_text}</div><div class="sub">{global_text}</div>',
        r'<div class="label">Pairwise vs HUGIML</div><div class="value small">[^<]+</div><div class="sub">[^<]+</div>': f'<div class="label">Pairwise vs HUGIML</div><div class="value small">{vs_text}</div><div class="sub">{vs_sub}</div>',
        r'<div class="label">Budget violations</div><div class="value">[^<]+</div><div class="sub">[^<]+</div>': f'<div class="label">Budget violations</div><div class="value">{budget_over}</div><div class="sub">budgeted fits exceeded {BUDGET:g} model inspection units</div>',
        r"<strong>Global comparison:</strong>[^<]+</div>": f"<strong>Global comparison:</strong> The Friedman test shows {interp}. It is a global rank test and does not identify which model pairs differ.</div>",
        r"<strong>HUGIML-specific comparison:</strong>[^<]+</div>": f"<strong>HUGIML-specific comparison:</strong> {hug_interp}</div>",
    }
    # Refresh static dashboard copy and keep panel size aligned with DATASET_NAMES.
    dataset_count = len(DATASET_NAMES)
    real_count = len(REAL_DATASET_NAMES)
    synthetic_count = len(SYNTHETIC_DATASET_NAMES)
    text = re.sub(
        r"A \d+-dataset tabular classification benchmark comparing HUGIML with tuned XGBoost,\s*LightGBM, RandomForest, logistic regression, EBM, and RuleFit baselines\.",
        f"A {dataset_count}-dataset tabular classification benchmark ({real_count} real-world + {synthetic_count} synthetic) comparing HUGIML with tuned XGBoost, LightGBM, RandomForest, logistic regression, EBM, and RuleFit baselines.",
        text,
        flags=re.S,
    )
    text = re.sub(r"Datasets(?: configured)?: \d+(?:; embedded completed: \d+)?", f"Datasets: {dataset_count}", text)
    text = re.sub(r'<div class="callout info" id="templateStatus">.*?</div>', '', text, flags=re.S)
    static_replacements = {
        "A 40-dataset tabular classification benchmark comparing HUGIML with tuned XGBoost,\n        LightGBM, and RandomForest baselines.":
            f"A {dataset_count}-dataset tabular classification benchmark ({real_count} real-world + {synthetic_count} synthetic) comparing HUGIML with tuned XGBoost,\n        LightGBM, RandomForest, logistic regression, EBM, and RuleFit baselines.",
        "HUGIML topK: 30 / 50 / 100": "HUGIML topK: 50 / 100",
        "Models: 7": f"Models: {len(MODEL_ORDER)}",
        "Leaves for tree ensembles; selected patterns for HUGIML":
            "Benchmark complexity uses model inspection units for every supported model",
        "Mean model complexity (leaves or patterns)":
            "Mean model inspection units",
    }
    for old, new in static_replacements.items():
        text = text.replace(old, new)

    for pat, repl in replacements.items():
        text = re.sub(pat, repl, text, count=1)

    # Keep the significant-all-pairs section description consistent with the actual run.
    # Some templates contain hard-coded copy such as "all 21 tests" or an assertion
    # that no HUGIML-specific comparison is significant; both can become stale.
    text = re.sub(
        r'(<h2>Significant all-pair comparisons</h2>\s*<div class="meta">)[^<]*(</div>)',
        lambda m: m.group(1) + all_pair_meta + m.group(2),
        text,
        flags=re.S,
    )
    text = re.sub(
        r"These rows are statistically significant pairwise differences[^<]*",
        all_pair_note,
        text,
        flags=re.S,
    )
    text = re.sub(
        r"Real model-pair differences after Holm correction across all \d+ tests",
        all_pair_meta,
        text,
    )

    text = text.replace(
        "function renderSigPairs(){\n  const rows=DATA.pairwise_all.filter(r=>r.significant_holm_0_05);",
        "function renderSigPairs(){\n  if(!document.getElementById('significantPairsTable')) return;\n  const rows=DATA.pairwise_all.filter(r=>r.significant_holm_0_05);",
    )
    if not any(bool(r.get("significant_holm_0_05")) for r in data.get("pairwise_all", [])):
        text = re.sub(
            r'\n<section class="card" style="margin-top:16px">\s*<div class="section-title"><h2>Significant all-pair comparisons</h2>.*?<table id="significantPairsTable"></table></div>\s*</section>\s*',
            "\n",
            text,
            flags=re.S,
        )
    # The default HUGIML grid uses one feature mode, so omit the feature-mode plot.
    text = text.replace(
        '<section class="grid-3">\n  <div class="card"><div class="section-title"><h2>HUGIML feature mode</h2><div class="meta">Best selected mode</div></div><div id="hugFeatureMode" class="plot small"></div></div>\n  <div class="card"><div class="section-title"><h2>HUGIML depth and topK</h2><div class="meta">Selected configuration frequencies</div></div><div id="hugTopK" class="plot small"></div></div>\n  <div class="card"><div class="section-title"><h2>HUGIML AUC gap</h2><div class="meta">HUGIML minus best non-HUGIML baseline</div></div><div id="hugGap" class="plot small"></div></div>\n</section>',
        '<section class="grid-2">\n  <div class="card"><div class="section-title"><h2>HUGIML depth and topK</h2><div class="meta">Selected configuration frequencies</div></div><div id="hugTopK" class="plot small"></div></div>\n  <div class="card"><div class="section-title"><h2>HUGIML AUC gap</h2><div class="meta">HUGIML minus best non-HUGIML baseline</div></div><div id="hugGap" class="plot small"></div></div>\n</section>',
    )
    text = text.replace(
        "  Plotly.react('hugFeatureMode',[{type:'bar',x:Object.keys(fm),y:Object.values(fm),marker:{color:theme().colors}}],layout({margin:{l:55,r:15,t:20,b:80},yaxis:{title:'Datasets',gridcolor:theme().grid}}),{displayModeBar:false,responsive:true});\n",
        "",
    )
    text = text.replace(
        "range:[.86,1.0]",
        "range:[Math.max(0,Math.min(...d.map(r=>Number(r.mean_auc)).filter(Number.isFinite))-.02),Math.min(1,Math.max(...d.map(r=>Number(r.mean_auc)).filter(Number.isFinite))+.02)]",
        1,
    )
    text = text.replace(
        "range:[.88,.95]",
        "range:[Math.max(0,Math.min(...d.map(r=>Number(r.mean_auc)).filter(Number.isFinite))-.02),Math.min(1,Math.max(...d.map(r=>Number(r.mean_auc)).filter(Number.isFinite))+.02)]",
        1,
    )
    text = _ensure_dashboard_scope_summary_ui(text)
    text = _remove_dashboard_profile_ui(text)
    text = _ensure_dashboard_complexity_rendering(text)
    text = _ensure_dashboard_scenario_ui(text, data)
    text = _ensure_dashboard_rpte_ui(text, data)
    methodology = data.get("methodology", {}) if isinstance(data, dict) else {}
    if methodology:
        text = _inject_before_body_end(text, _methodology_section_html(methodology))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(text)




_LOCAL_PROVENANCE_KEYS = {"checkpoint", "out_dir", "template_html"}


def _is_local_provenance_key(key: Any) -> bool:
    name = str(key).lower()
    return (
        name in _LOCAL_PROVENANCE_KEYS
        or name in {"baseline_source", "source_root"}
        or name.endswith("_imported_from")
    )


_LOCAL_FILE_REFERENCE_RE = re.compile(
    r'File\s+(["\'])(?:[A-Za-z]:[\\/]|/).*?\1'
)
_LOCAL_ABSOLUTE_PATH_RE = re.compile(
    r'''(?<![A-Za-z0-9_:/])(?:[A-Za-z]:[\\/]|/)(?:[^\s"'<>:,]+[\\/]?)+'''
)


def _sanitize_local_text(value: str) -> str:
    text = _LOCAL_FILE_REFERENCE_RE.sub('File "<local-path>"', value)
    return _LOCAL_ABSOLUTE_PATH_RE.sub("<local-path>", text)


def _remove_local_provenance(value: Any) -> Any:
    """Remove machine-specific locations from data written by dashboard assembly."""
    if isinstance(value, dict):
        return {
            key: _remove_local_provenance(item)
            for key, item in value.items()
            if not _is_local_provenance_key(key)
        }
    if isinstance(value, list):
        return [_remove_local_provenance(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_remove_local_provenance(item) for item in value)
    if isinstance(value, str):
        return _sanitize_local_text(value)
    return value


def assemble_outputs(
    checkpoint: Path,
    out_dir: Path,
    template_html: Path,
    *,
    include_sbom: bool = False,
) -> dict[str, Path]:
    global MODEL_ORDER
    payload = _remove_local_provenance(load_checkpoint(checkpoint))
    details = payload.get("results", [])
    observed_models = {
        _canonical_model_label(row.get("model")) for row in details
    }
    MODEL_ORDER = [model for model in MODEL_ORDER if model in observed_models]
    order = {d: i for i, d in enumerate(DATASET_NAMES)}
    mo = {m: i for i, m in enumerate(MODEL_ORDER)}
    so = {s: i for i, s in enumerate(HUGIML_SCENARIOS)}
    details = sorted(
        details,
        key=lambda r: (
            order.get(r.get("dataset"), 999),
            mo.get(r.get("model"), 999),
            so.get(_hugiml_scenario_for_row(r), 999),
        ),
    )
    expected_keys = set(pair_plan(DATASET_NAMES, MODEL_ORDER))
    actual_keys = {
        _pair_key(str(r.get("dataset")), str(r.get("model")), _hugiml_scenario_for_row(r))
        for r in details
    }
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=lambda x: (order.get(x[0], 999), mo.get(x[1], 999), so.get(x[2], 999)))
        extra = sorted(actual_keys - expected_keys, key=lambda x: (order.get(x[0], 999), mo.get(x[1], 999), so.get(x[2], 999)))
        raise RuntimeError(
            f"checkpoint is incomplete or contains unexpected rows: {len(details)} rows, "
            f"missing {len(missing)} pairs, extra {len(extra)} pairs, "
            f"first missing={missing[:5]}, first extra={extra[:5]}"
        )
    data = _remove_local_provenance(make_data(details))
    methodology = methodology_snapshot(payload)
    data["methodology"] = methodology
    for scenario in data.get("dashboard_scenarios", []):
        if isinstance(scenario, dict) and isinstance(scenario.get("data"), dict):
            scenario["data"]["methodology"] = methodology
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {
        "checkpoint": out_dir / "benchmark_checkpoint.json",
        "data": out_dir / "dashboard_data.json",
        "details_csv": out_dir / "details.csv",
        "overall_csv": out_dir / "overall.csv",
        "summary_by_scope_csv": out_dir / "summary_by_scope.csv",
        "scope_tests_csv": out_dir / "scope_tests.csv",
        "summary_csv": out_dir / "summary_comparison.csv",
        "html": out_dir / "hugiml_benchmark_analysis_dashboard.html",
    }

    paths["checkpoint"].write_text(json.dumps(_safe_jsonable(payload), indent=2, allow_nan=False))
    paths["data"].write_text(json.dumps(_safe_jsonable(data), separators=(",", ":"), allow_nan=False))
    pd.DataFrame(data["details"]).to_csv(paths["details_csv"], index=False)
    pd.DataFrame(data["overall"]).to_csv(paths["overall_csv"], index=False)
    pd.DataFrame(data.get("summary_by_scope", [])).to_csv(paths["summary_by_scope_csv"], index=False)
    pd.DataFrame(data.get("scope_tests", [])).to_csv(paths["scope_tests_csv"], index=False)
    summary_comparison(template_html, data, paths["summary_csv"])

    sbom: dict[str, Any] | None = None
    if include_sbom:
        paths["sbom"] = out_dir / "benchmark_reproducibility_sbom.json"
        sbom = build_reproducibility_sbom(
            checkpoint=checkpoint,
            out_dir=out_dir,
            template_html=template_html,
            payload=payload,
            data=data,
            output_paths=paths,
        )
        sbom = _sanitize_sbom_local_paths(sbom, out_dir=out_dir)
        paths["sbom"].write_text(json.dumps(sbom, indent=2, sort_keys=True, allow_nan=False))

    render_html(template_html, data, paths["html"])
    return paths


def parse_list(raw: str | None, allowed: list[str]) -> list[str]:
    if raw in (None, "", "all"):
        return allowed
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in vals if x not in allowed]
    if bad:
        raise ValueError(f"Unknown values: {bad}. Allowed: {allowed}")
    return vals


def _resolve_output_dir(raw: str | None) -> Path:
    return (Path(raw).expanduser() if raw else (SCRIPT_DIR / RESULTS_DIR_NAME)).resolve()


def _resolve_checkpoint(raw: str | None, out_dir: Path) -> Path:
    return (Path(raw).expanduser() if raw else (out_dir / "benchmark_checkpoint.json")).resolve()


def _resolve_template_html(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()

    # Scripts may live under repo/experiments/benchmark/ or
    # repo/experiments/scalability/.  The dashboard template usually lives
    # at the repo root, so search upward from this script before falling back.
    # Order matters: prefer the nearest template to allow subproject-specific
    # templates, but still find the repo-level base dashboard two or more
    # levels above the script.
    candidates: list[Path] = []
    for base in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        candidates.append(base / DEFAULT_TEMPLATE_NAME)

    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        candidates.append(base / DEFAULT_TEMPLATE_NAME)

    seen: set[Path] = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if rp.exists():
            return rp

    # render_html has an internal fallback template if this path does not exist.
    # Return the repo-level expected location when discoverable; otherwise use
    # the script directory.
    if SOURCE_ROOT is not None:
        return (SOURCE_ROOT / DEFAULT_TEMPLATE_NAME).resolve()
    return (SCRIPT_DIR / DEFAULT_TEMPLATE_NAME).resolve()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Batch benchmark runner for the HUGIML dashboard")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to <out-dir>/benchmark_checkpoint.json",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Results directory. Defaults to ./results next to this script",
    )
    ap.add_argument(
        "--template-html",
        default=None,
        help="Dashboard template. Defaults to the nearest template found by walking upward from this script, then cwd, then built-in fallback",
    )
    ap.add_argument("--datasets", default="all", help="Comma-separated dataset names, or all")
    ap.add_argument("--models", default="all", help="Comma-separated model labels, or all")
    ap.add_argument("--start-pair", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Delete and recreate the output folder/checkpoint before a fresh run",
    )
    ap.add_argument(
        "--assemble", action="store_true", help="Build CSV/JSON/HTML from the checkpoint"
    )
    ap.add_argument(
        "--include-sbom",
        action="store_true",
        help="When assembling, emit benchmark_reproducibility_sbom.json alongside the dashboard outputs",
    )
    ap.add_argument(
        "--row-cap",
        type=int,
        default=-1,
        help="Cap rows per dataset. Use -1 for the full dataset, which is the default",
    )
    ap.add_argument("--n-splits", type=int, default=5, help="Outer StratifiedKFold split count")
    ap.add_argument("--inner-splits", type=int, default=3, help="Inner tuning StratifiedKFold split count")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed for data generation and CV")
    ap.add_argument("--no-tune", action="store_true", help="Disable inner-CV hyperparameter tuning")
    ap.add_argument(
        "--hugiml-max-fit-seconds",
        type=float,
        default=None,
        help="Optional base parameter for constrained environments",
    )
    args = ap.parse_args(argv)

    global RANDOM_STATE
    RANDOM_STATE = int(args.random_state)

    out_dir = _resolve_output_dir(args.out_dir)
    checkpoint = _resolve_checkpoint(args.checkpoint, out_dir)
    template_html = _resolve_template_html(args.template_html)
    if args.fresh and not args.assemble:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint.exists():
            try:
                checkpoint.unlink()
            except FileNotFoundError:
                pass
    if args.assemble:
        paths = assemble_outputs(checkpoint, out_dir, template_html, include_sbom=args.include_sbom)
        for k, v in paths.items():
            print(f"{k}: {v}")
        return 0

    datasets = parse_list(args.datasets, DATASET_NAMES)
    models = parse_list(args.models, MODEL_ORDER)
    plan = pair_plan(datasets, models)
    if args.start_pair:
        plan = plan[args.start_pair :]
    if args.max_pairs is not None:
        plan = plan[: args.max_pairs]

    for model_name, scenario, expected_grid_name, expected_C, expected_solver in (
        ("HUGIML", "augmented_pair", "performance_ho", 0.5, "adaptive_l1"),
        ("HUGIML", "interaction_relaxed", "interpretability_ho", 0.5, "adaptive_l1"),
    ):
        grid_name, candidates = _hugiml_grid_for_model(model_name, scenario)
        assert grid_name == expected_grid_name
        grid = get_hugiml_grid(grid_name)
        assert grid.get("topK") == [50, 100]
        assert grid.get("L") == [1, 2]
        assert grid.get("G") == [0.01, 0.001]
        assert grid.get("topk_budget_strict") == [False]
        assert grid.get("lr_C") == [expected_C]
        assert grid.get("lr_solver", ["auto"]) == [expected_solver]
        assert len(candidates) == 16
        base_estimators = list(grid.get("base_estimator", []))
        assert len(base_estimators) == 2
        assert sum(estimator is None for estimator in base_estimators) == 1
        rpte_candidates = [estimator for estimator in base_estimators if estimator is not None]
        assert len(rpte_candidates) == 1
        rpte_inner = getattr(rpte_candidates[0], "estimator", rpte_candidates[0])
        assert rpte_inner.get_params().get("leaf_config") == "3xD"
        if scenario == "augmented_pair":
            assert grid.get("feature_mode") == ["original_plus_patterns"]
            assert grid.get("augmented_pair_transforms") == [True]
            assert grid.get("convert_binary_to_categorical") == [False]
        else:
            assert grid.get("feature_mode") == ["patterns_only"]
            assert grid.get("interaction_relaxed_mining") == [True]
            assert grid.get("augmented_pair_transforms") == [False]
            assert grid.get("convert_binary_to_categorical") == [True]
    lr_grid = get_baseline_grid("LogisticRegression")
    assert set(lr_grid) == {"C", "penalty", "class_weight"}
    assert len(list(ParameterGrid(lr_grid))) == 16
    ebm_grid = get_baseline_grid("EBM")
    assert ebm_grid.get("learning_rate") == [0.01, 0.05]
    assert ebm_grid.get("max_bins") == [32, 64]
    assert ebm_grid.get("interactions") == [0, 5]
    assert len(list(ParameterGrid(ebm_grid))) == 8
    xgb_grid = get_baseline_grid("XGBoost")
    assert xgb_grid.get("n_estimators") == [100, 200]
    assert xgb_grid.get("max_depth") == [3, 4]
    assert xgb_grid.get("learning_rate") == [0.03, 0.1]
    assert xgb_grid.get("min_child_weight") == [1, 5]
    assert len(list(ParameterGrid(xgb_grid))) == 16
    assert "subsample" not in xgb_grid
    lgb_grid = get_baseline_grid("LightGBM")
    assert lgb_grid.get("n_estimators") == [100, 200]
    assert lgb_grid.get("learning_rate") == [0.03, 0.1]
    assert lgb_grid.get("num_leaves") == [15, 31]
    assert lgb_grid.get("min_child_samples") == [10, 20]
    assert len(list(ParameterGrid(lgb_grid))) == 16
    assert "subsample" not in lgb_grid
    rf_grid = get_baseline_grid("RandomForest")
    assert rf_grid.get("max_features") == ["sqrt", 0.5]
    assert len(list(ParameterGrid(rf_grid))) == 16
    for model_name in ["XGBoost", "LightGBM", "RandomForest"]:
        assert len(list(ParameterGrid(get_budgeted_baseline_grid(model_name)))) == 16
    rulefit_grid = get_baseline_grid("RuleFit")
    assert rulefit_grid.get("n_estimators") == [50, 100]
    assert rulefit_grid.get("max_rules") == [50, 100]
    assert rulefit_grid.get("tree_size") == [5, 10]
    assert len(list(ParameterGrid(rulefit_grid))) == 8
    assert BUDGET == 200.0

    payload = load_checkpoint(checkpoint)
    if args.resume:
        _validate_resume_feature_policy(payload)
    payload["metadata"].update(
        {
            "random_state": RANDOM_STATE,
            "row_cap": args.row_cap,
            "n_splits": args.n_splits,
            "inner_splits": args.inner_splits,
            "tune": not args.no_tune,
            "evaluation_protocol": "outer_cv_inner_cv_tuning" if not args.no_tune else "outer_cv_no_inner_tuning",
            "hugiml_max_fit_seconds": args.hugiml_max_fit_seconds,
            "grid_snapshot": grid_snapshot(),
            "hugiml_dashboard_scenarios": HUGIML_SCENARIOS,
            "default_hugiml_dashboard_scenario": DEFAULT_DASHBOARD_HUGIML_SCENARIO,
            "dataset_feature_policy": DATASET_FEATURE_POLICY,
        }
    )
    done = completed_keys(payload) if args.resume else set()

    for dataset, model, hugiml_scenario in plan:
        key = _pair_key(dataset, model, hugiml_scenario)
        scenario_suffix = f" :: {hugiml_scenario}" if is_hugiml_model(model) else ""
        if key in done:
            print(f"skip {dataset} :: {model}{scenario_suffix}", flush=True)
            continue
        print(f"run {dataset} :: {model}{scenario_suffix}", flush=True)
        started = time.perf_counter()
        row = run_pair(
            dataset,
            model,
            hugiml_scenario=hugiml_scenario,
            row_cap=args.row_cap,
            hugiml_max_fit_seconds=args.hugiml_max_fit_seconds,
            n_splits=args.n_splits,
            inner_splits=args.inner_splits,
            tune=not args.no_tune,
            random_state=args.random_state,
            fold_checkpoint_dir=out_dir / "fold_checkpoints",
        )
        row["pair_seconds"] = float(time.perf_counter() - started)
        payload["results"] = [
            r
            for r in payload["results"]
            if _pair_key(str(r.get("dataset")), str(r.get("model")), _hugiml_scenario_for_row(r)) != key
        ]
        payload["results"].append(row)
        save_checkpoint(checkpoint, payload)
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "model": model,
                    "hugiml_scenario": hugiml_scenario,
                    "auc": row.get("auc"),
                    "roc_auc": row.get("roc_auc"),
                    "best_inner_score": row.get("best_inner_score"),
                    "complexity": row.get("complexity"),
                    "seconds": row.get("pair_seconds"),
                },
                default=_json_default,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

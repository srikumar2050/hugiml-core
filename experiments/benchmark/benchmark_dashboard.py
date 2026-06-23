from __future__ import annotations

# ruff: noqa: E402
import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
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
import statsmodels.api as sm
from lightgbm import LGBMClassifier
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_circles,
    make_classification,
    make_moons,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

import hugiml as _hugiml_pkg
from hugiml import HUGIMLClassifierNative

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
        np.asarray(X_fit, dtype=np.float32),
        (None if X_other is None else np.asarray(X_other, dtype=np.float32)),
        prep,
        elapsed,
    )


def _transform_with_preprocessor(prep: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(prep.transform(X), dtype=np.float32)


def _preprocessor_feature_count(prep: ColumnTransformer, fallback: int) -> int:
    try:
        return int(len(prep.get_feature_names_out()))
    except Exception:
        return int(fallback)


HUGIML_IMPORTED_FROM = str(Path(getattr(_hugiml_pkg, "__file__", "")).resolve())

try:
    from hugiml.hyperparameter_configs import (
        DEFAULT_HUGIML_GRID_NAME,
        get_baseline_grid,
        get_hugiml_grid,
    )
except Exception:
    # PyPI wheels may not ship the benchmark helper grids.  Keep the dashboard
    # benchmark reproducible by embedding the same default grids here.
    DEFAULT_HUGIML_GRID_NAME = "performance"

    def get_hugiml_grid(name: str = DEFAULT_HUGIML_GRID_NAME) -> dict[str, list[Any]]:
        if name != DEFAULT_HUGIML_GRID_NAME:
            raise KeyError(name)
        return {
            "B": [-1],
            "adaptive_binning": [True],
            "L": [1, 2],
            "topK": [50, 100],
            "feature_mode": ["original_plus_patterns"],
            "G": [0.01, 0.001],
        }

    def get_baseline_grid(model: str) -> dict[str, list[Any]]:
        grids = {
            "XGBoost": {
                "n_estimators": [100, 200],
                "max_depth": [3, 4],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8, 1.0],
            },
            "LightGBM": {
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.1],
                "num_leaves": [15, 31],
                "subsample": [0.8, 1.0],
            },
            "RandomForest": {
                "n_estimators": [200, 400],
                "max_depth": [4, 8, None],
                "min_samples_leaf": [1, 5],
            },
        }
        return grids[model]


RANDOM_STATE = 2026
BUDGET = 300.0
MODEL_ORDER = [
    "HUGIML",
    "XGB standard",
    "XGB complexity-budgeted",
    "LightGBM standard",
    "LightGBM complexity-budgeted",
    "RandomForest standard",
    "RandomForest complexity-budgeted",
]
# A comprehensive benchmark panel with 30 public real-world datasets and
# 20 deterministic synthetic cases that stress different inductive biases.
DATASET_NAMES = [
    # Public real-world datasets from scikit-learn and statsmodels
    "BreastCancerOriginal",
    "DiabetesHighTarget",
    "DigitsOddVsEven64",
    "DigitsHighVsLow64",
    "WineClass1",
    "IrisVersicolor",
    "ANES96_PIDHigh",
    "CCard_HighAvgExp",
    "Cancer_HighRate",
    "CopperHighWorldConsumption",
    "DanishHighMoneyDemand",
    "ElNinoHighAnnualTemp",
    "Engel_HighIncome",
    "FairAffairs",
    "FertilityHigh2010",
    "FertilityHighDecline1960_2010",
    "Grunfeld_HighInvest",
    "HeartTransplantCensorStatus",
    "InterestInflationHighRate",
    "LongleyHighEmployment",
    "MacroData_HighInflation",
    "ModeChoice",
    "Nile_HighVolume6Lag",
    "RandHIE_HighMDVisits",
    "ScotlandHighYesVote",
    "SpectorGrade",
    "StacklossHigh",
    "Star98_HighAboveRate",
    "StateCrimeHighViolent",
    "Sunspots_HighActivity12Lag",
    # Reproducible synthetic stress tests
    "SynthLinearLowDim",
    "SynthSparseWide",
    "SynthMoonsNonlinear",
    "SynthCirclesNonlinear",
    "SynthXORInteractions",
    "SynthAdditiveNonlinear",
    "SynthCategoricalRules",
    "SynthCategoricalNumericInteraction",
    "SynthHighCardinalityCategorical",
    "SynthOrdinalCategorical",
    "SynthMixedMissing",
    "SynthMissingNotAtRandom",
    "SynthImbalancedRare",
    "SynthNoisyHighDimensional",
    "SynthSmallNWide",
    "SynthCorrelatedBlocks",
    "SynthThresholdRules",
    "SynthPiecewiseLinear",
    "SynthHeteroskedasticNoise",
    "SynthMostlyCategorical",
]


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
    return getattr(sm.datasets, name).load_pandas().data.reset_index(drop=False)


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


def _load_synthetic_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray, str] | None:
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
        months = [c for c in df.columns if c not in ["YEAR", "index"]]
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


def xgb_complexity(model):
    model = _unwrap_model(model)
    try:
        df = model.get_booster().trees_to_dataframe()
        return int((df["Feature"] == "Leaf").sum())
    except Exception:
        return None


def lgb_complexity(model):
    model = _unwrap_model(model)

    def count(node):
        if "leaf_index" in node:
            return 1
        return count(node["left_child"]) + count(node["right_child"])

    try:
        dump = model.booster_.dump_model()
        return int(sum(count(t["tree_structure"]) for t in dump["tree_info"]))
    except Exception:
        return None


def rf_complexity(model):
    model = _unwrap_model(model)
    try:
        return int(sum(est.tree_.n_leaves for est in model.estimators_))
    except Exception:
        return None


def hug_complexity(model):
    for method in ["get_pattern_info"]:
        try:
            return int(len(getattr(model, method)()))
        except Exception:
            pass
    for attr in ["selected_patterns_", "raw_patterns_"]:
        try:
            val = getattr(model, attr, None)
            if val is not None:
                return int(len(val))
        except Exception:
            pass
    return None


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


def fit_final(builder, params, X_fit, y_fit):
    clf = builder(params)
    t0 = time.perf_counter()
    clf.fit(X_fit, y_fit)
    return clf, time.perf_counter() - t0


def get_model_spec(model: str, *, hugiml_max_fit_seconds: float | None = None):
    if model == "HUGIML":
        grid = list(ParameterGrid(get_hugiml_grid(DEFAULT_HUGIML_GRID_NAME)))

        def builder(params):
            pp = dict(params)
            pp.setdefault("execution_mode", "production")
            if hugiml_max_fit_seconds is not None:
                pp.setdefault("max_fit_seconds", float(hugiml_max_fit_seconds))
            pp.setdefault("n_jobs", 1)
            return HUGIMLClassifierNative(**pp)

        return grid, builder, hug_complexity, None

    xgb_grid = list(ParameterGrid(get_baseline_grid("XGBoost")))
    lgb_grid = list(ParameterGrid(get_baseline_grid("LightGBM")))
    rf_grid = list(ParameterGrid(get_baseline_grid("RandomForest")))
    xgb_budget_grid = list(
        ParameterGrid(
            {
                "n_estimators": [25, 50, 75],
                "max_depth": [1, 2, 3],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }
        )
    )
    lgb_budget_grid = list(
        ParameterGrid(
            {
                "n_estimators": [25, 50, 75],
                "num_leaves": [2, 4, 8],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8, 1.0],
                "min_child_samples": [5],
            }
        )
    )
    rf_budget_grid = list(
        ParameterGrid(
            {
                "n_estimators": [20, 50, 75],
                "max_leaf_nodes": [2, 4, 8],
                "min_samples_leaf": [1, 5],
                "max_depth": [None],
            }
        )
    )

    def xgb_builder(params):
        return XGBClassifier(
            eval_metric="logloss", verbosity=0, n_jobs=1, random_state=RANDOM_STATE, **params
        )

    def lgb_builder(params):
        return LGBMClassifier(verbose=-1, n_jobs=1, random_state=RANDOM_STATE, **params)

    def rf_builder(params):
        return RandomForestClassifier(n_jobs=1, random_state=RANDOM_STATE, **params)

    specs = {
        "XGB standard": (xgb_grid, xgb_builder, xgb_complexity, None),
        "LightGBM standard": (lgb_grid, lgb_builder, lgb_complexity, None),
        "RandomForest standard": (rf_grid, rf_builder, rf_complexity, None),
        "XGB complexity-budgeted": (xgb_budget_grid, xgb_builder, xgb_complexity, BUDGET),
        "LightGBM complexity-budgeted": (lgb_budget_grid, lgb_builder, lgb_complexity, BUDGET),
        "RandomForest complexity-budgeted": (rf_budget_grid, rf_builder, rf_complexity, BUDGET),
    }
    return specs[model]


def run_pair(
    dataset: str,
    model: str,
    *,
    row_cap: int | None = -1,
    hugiml_max_fit_seconds: float | None = None,
) -> dict[str, Any]:
    X, y, group = load_dataset(dataset)
    raw_features = int(X.shape[1])
    native_categorical_features = _categorical_columns(X)

    if dataset == "InterestInflationHighRate" and model == "HUGIML":
        if "year" not in X.columns or not _is_categorical_like(X["year"]):
            raise RuntimeError(
                "InterestInflationHighRate expected HUGIML to receive native categorical year. "
                f"Saw columns={list(X.columns)} dtypes={X.dtypes.astype(str).to_dict()}"
            )

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, row_cap=row_cap)
    # Preserve native categoricals, but force independent writable column storage
    # for every model.  This prevents read-only ndarray/categorical-code views
    # from leaking out of data generators or train_test_split slices.
    X_train_native = _force_writable_frame(X_train.reset_index(drop=True))
    X_val_native = _force_writable_frame(X_val.reset_index(drop=True))
    X_test_native = _force_writable_frame(X_test.reset_index(drop=True))
    X_fit_native = _force_writable_frame(
        pd.concat([X_train_native, X_val_native], axis=0).reset_index(drop=True)
    )
    y_train = np.array(y_train, dtype=int, copy=True)
    y_val = np.array(y_val, dtype=int, copy=True)
    y_test = np.array(y_test, dtype=int, copy=True)
    y_fit = np.array(np.concatenate([y_train, y_val]), dtype=int, copy=True)

    grid, builder, complexity_fn, budget = get_model_spec(
        model, hugiml_max_fit_seconds=hugiml_max_fit_seconds
    )

    preprocess_select_seconds = 0.0
    preprocess_final_seconds = 0.0
    preprocessing_policy = "HUGIML native categorical"
    model_features = raw_features

    if model == "HUGIML":
        X_train_model, X_val_model = X_train_native, X_val_native
        X_fit_model, X_test_model = X_fit_native, X_test_native
        final_prep = None
    else:
        # Fair + optimized baseline path:
        #   * HUGIML's fit_seconds includes its internal preprocessing.
        #   * Non-HUGIML fit_seconds includes one final OHE fit/transform plus
        #     the selected estimator fit.
        #   * Pair/grid time avoids unfair repeated OHE fitting by fitting OHE
        #     once for validation selection and once for the final train+val fit.
        X_train_model, X_val_model, select_prep, preprocess_select_seconds = (
            _fit_transform_non_hugiml(X_train_native, X_val_native)
        )
        X_fit_model, X_test_model, final_prep, preprocess_final_seconds = _fit_transform_non_hugiml(
            X_fit_native, X_test_native
        )
        model_features = _preprocessor_feature_count(final_prep, raw_features)
        preprocessing_policy = "non-HUGIML train-fitted OHE, fitted once per split"

    selected, errors = fit_select(
        grid, builder, X_train_model, y_train, X_val_model, y_val, complexity_fn, budget
    )
    if selected is None:
        row = {
            "auc": None,
            "f1": None,
            "accuracy": None,
            "valid_auc": None,
            "complexity": None,
            "complexity_budget": None if budget is None else float(budget),
            "fit_seconds": None,
            "model_fit_seconds": None,
            "preprocess_fit_transform_seconds": preprocess_final_seconds,
            "selection_preprocess_seconds": preprocess_select_seconds,
            "best_params_json": "{}",
            "error_count": len(errors),
            "last_error": errors[-1] if errors else None,
        }
    else:
        vauc, _, _, _, params, _ = selected
        clf, model_fit_s = fit_final(builder, params, X_fit_model, y_fit)
        fit_s = float(model_fit_s + (preprocess_final_seconds if model != "HUGIML" else 0.0))
        comp = complexity_fn(clf)
        if budget is not None and comp is not None and comp > budget:
            viable = []
            for params2 in grid:
                try:
                    # Rank by the original train/validation selection split, but
                    # keep the final model trained on train+validation.
                    clf_sel, _ = fit_final(builder, params2, X_train_model, y_train)
                    vauc2 = safe_auc(y_val, probas(clf_sel, X_val_model))
                    clf2, fs2 = fit_final(builder, params2, X_fit_model, y_fit)
                    comp2 = complexity_fn(clf2)
                    if comp2 is not None and comp2 <= budget:
                        viable.append(
                            (
                                vauc2,
                                comp2,
                                float(
                                    fs2 + (preprocess_final_seconds if model != "HUGIML" else 0.0)
                                ),
                                clf2,
                                copy.deepcopy(params2),
                            )
                        )
                except Exception:
                    pass
            if viable:
                viable.sort(key=lambda z: (z[0], -z[1]), reverse=True)
                vauc, comp, fit_s, clf, params = viable[0]
                model_fit_s = max(
                    0.0, float(fit_s - (preprocess_final_seconds if model != "HUGIML" else 0.0))
                )
        effective_budget = params.get("topK") if model == "HUGIML" else budget
        row = metric_row(
            clf, X_test_model, y_test, fit_s, vauc, params, complexity_fn(clf), effective_budget
        )
        row["model_fit_seconds"] = float(model_fit_s)
        row["preprocess_fit_transform_seconds"] = float(preprocess_final_seconds)
        row["selection_preprocess_seconds"] = float(preprocess_select_seconds)
        row["error_count"] = len(errors)
        row["last_error"] = errors[-1] if errors else None
    row.update(
        {
            "dataset": dataset,
            "dataset_group": group,
            "model": model,
            "raw_features": raw_features,
            "model_features": model_features,
            "categorical_features": native_categorical_features,
            "n_rows": int(len(y)),
            "class_balance": float(np.mean(y)),
            "source_root": str(SOURCE_ROOT) if SOURCE_ROOT is not None else "installed_package",
            "hugiml_imported_from": HUGIML_IMPORTED_FROM,
            "preprocessing_policy": preprocessing_policy,
        }
    )
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
        "hugiml_grid_name": DEFAULT_HUGIML_GRID_NAME,
        "hugiml_grid": get_hugiml_grid(DEFAULT_HUGIML_GRID_NAME),
        "baseline_grids": {
            "XGBoost": get_baseline_grid("XGBoost"),
            "LightGBM": get_baseline_grid("LightGBM"),
            "RandomForest": get_baseline_grid("RandomForest"),
        },
        "execution_mode_base_setting": "production",
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
        },
        "results": [],
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default))
    tmp.replace(path)


def pair_plan(datasets: list[str], models: list[str]) -> list[tuple[str, str]]:
    return [(d, m) for d in datasets for m in models]


def completed_keys(payload: dict[str, Any]) -> set[tuple[str, str]]:
    return {(r.get("dataset"), r.get("model")) for r in payload.get("results", [])}


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
        comp = pd.to_numeric(
            sub.get("complexity", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
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


def make_data(details: list[dict[str, Any]]) -> dict[str, Any]:
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
        comp = pd.to_numeric(sub["complexity"], errors="coerce").dropna()
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
    for _, r in df[df.model == "HUGIML"].iterrows():
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
                "model_complexity_leaves_or_patterns": None
                if pd.isna(r["complexity"])
                else int(r["complexity"]),
                "complexity_budget": None
                if pd.isna(r["complexity_budget"])
                else float(r["complexity_budget"]),
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
        "details": df.drop(columns=[c for c in ["source_root"] if c in df.columns]).to_dict(
            orient="records"
        ),
        "heat": heat,
        "global": global_rows,
        "vs_hugiml": pairs,
        "pairwise_all": pairwise,
        "budget": budget,
        "hug": hugrows,
        "summary_by_scope": scope_summary["summary_by_scope"],
        "scope_tests": scope_summary["scope_tests"],
        "dataset_profiles": build_dataset_profiles(DATASET_NAMES),
        "dataset_catalog": [
            {
                "dataset": name,
                "kind": (
                    "Synthetic" if name.startswith("Synth") else "Public real-world / package"
                ),
                "source": (
                    "generated in script"
                    if name.startswith("Synth")
                    else "scikit-learn or statsmodels"
                ),
            }
            for name in DATASET_NAMES
        ],
    }


def extract_original_data(html_path: Path) -> dict[str, Any]:
    text = html_path.read_text(errors="ignore")
    start = text.index("const DATA=") + len("const DATA=")
    end = text.index(";\nconst MODEL_ORDER", start)
    return json.loads(text[start:end])


def summary_comparison(
    original_html: Path, new_data: dict[str, Any], out_csv: Path
) -> pd.DataFrame:
    new = pd.DataFrame(new_data["overall"])[
        ["model", "mean_auc", "mean_rank", "strict_wins", "tied_best_count", "mean_complexity"]
    ]
    try:
        original = extract_original_data(original_html)
        old = pd.DataFrame(original["overall"])[
            ["model", "mean_auc", "mean_rank", "strict_wins", "tied_best_count", "mean_complexity"]
        ]
        old = old.rename(
            columns={
                "mean_auc": "mean_auc_earlier",
                "mean_rank": "mean_rank_earlier",
                "strict_wins": "strict_wins_earlier",
                "tied_best_count": "tied_best_count_earlier",
                "mean_complexity": "mean_complexity_earlier",
            }
        )
        new = new.rename(
            columns={
                "mean_auc": "mean_auc_new",
                "mean_rank": "mean_rank_new",
                "strict_wins": "strict_wins_new",
                "tied_best_count": "tied_best_count_new",
                "mean_complexity": "mean_complexity_new",
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
                "mean_complexity": "mean_complexity_new",
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
<section class=\"grid-2\"><div class=\"card\"><h2>Mean AUC by model</h2><div id=\"meanAuc\" class=\"plot\"></div></div><div class=\"card\"><h2>Complexity vs performance</h2><div id=\"complexityPerf\" class=\"plot\"></div></div></section>
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
  Plotly.react('complexityPerf',traces,{margin:{l:70,r:80,t:20,b:70},xaxis:{title:'Complexity',type:'log',automargin:true},yaxis:{title:'AUC',automargin:true,range:[0,1.03]},legend:{orientation:'h',y:-0.22}}, {responsive:true});
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


def _ensure_dashboard_scope_summary_ui(text: str) -> str:
    if "scopeSummaryTables" in text:
        return text
    html = """
<section class="card" id="scopeSummaryCard" style="margin-top:16px">
  <div class="section-title"><h2>Summary results by dataset scope</h2><div class="meta">Overall, public real-world/package datasets, and reproducible synthetic datasets.</div></div>
  <div id="scopeSummaryTables"></div>
</section>
"""
    js = """
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
      return `<div style="margin-top:14px"><h3 style="margin:8px 0 4px 0">${esc(scope)}</h3><div class="meta">${meta}</div><div style="overflow:auto;margin-top:8px"><table><thead><tr><th>Model</th><th>n</th><th>Mean AUC</th><th>Median AUC</th><th>Mean F1</th><th>Mean accuracy</th><th>Mean rank</th><th>Strict wins</th><th>Tied best</th><th>Mean complexity</th><th>Mean fit s</th></tr></thead><tbody>` +
        scopeRows.map(r=>`<tr><td>${esc(r.model)}</td><td>${esc(r.n_datasets)}</td><td>${fmt(r.mean_auc)}</td><td>${fmt(r.median_auc)}</td><td>${fmt(r.mean_f1)}</td><td>${fmt(r.mean_accuracy)}</td><td>${fmt2(r.mean_rank)}</td><td>${esc(r.strict_wins)}</td><td>${esc(r.tied_best_count)}</td><td>${fmt2(r.mean_complexity)}</td><td>${fmt(r.mean_fit_seconds)}</td></tr>`).join('') +
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


def render_html(template_html: Path, data: dict[str, Any], out_html: Path) -> None:
    text = (
        template_html.read_text(errors="ignore")
        if template_html.exists()
        else _default_dashboard_template()
    )
    if "const DATA=" not in text or ";\nconst MODEL_ORDER" not in text:
        text = _default_dashboard_template()
    start = text.index("const DATA=")
    end = text.index(";\nconst MODEL_ORDER", start)
    safe_data = _safe_jsonable(data)
    new_blob = "const DATA=" + json.dumps(safe_data, separators=(",", ":"), allow_nan=False)
    text = text[:start] + new_blob + text[end:]

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

    replacements = {
        r"Friedman p-value: [^<]+": f"Friedman p-value: {p_text}",
        r'<div class="label">Best mean AUC</div><div class="value">[^<]+</div><div class="sub">[^<]+</div>': f'<div class="label">Best mean AUC</div><div class="value">{best.mean_auc:.4f}</div><div class="sub">{best.model}</div>',
        r'<div class="label">HUGIML mean AUC</div><div class="value">[^<]+</div><div class="sub">[^<]+</div>': f'<div class="label">HUGIML mean AUC</div><div class="value">{hug.mean_auc:.4f}</div><div class="sub">mean complexity: {hug.mean_complexity:.1f} patterns</div>',
        r'<div class="label">Best mean rank</div><div class="value">[^<]+</div><div class="sub">[^<]+</div>': f'<div class="label">Best mean rank</div><div class="value">{best_rank.mean_rank:.2f}</div><div class="sub">{best_rank.model}</div>',
        r'<div class="label">Global rank test</div><div class="value">[^<]+</div><div class="sub">[^<]+</div>': f'<div class="label">Global rank test</div><div class="value">{p_text}</div><div class="sub">{global_text}</div>',
        r'<div class="label">Pairwise vs HUGIML</div><div class="value small">[^<]+</div><div class="sub">[^<]+</div>': f'<div class="label">Pairwise vs HUGIML</div><div class="value small">{vs_text}</div><div class="sub">{vs_sub}</div>',
        r'<div class="label">Budget violations</div><div class="value">[^<]+</div><div class="sub">[^<]+</div>': f'<div class="label">Budget violations</div><div class="value">{budget_over}</div><div class="sub">budgeted tree fits exceeded 300 leaves</div>',
        r"<strong>Global comparison:</strong>[^<]+</div>": f"<strong>Global comparison:</strong> The Friedman test shows {interp}. It is a global rank test and does not identify which model pairs differ.</div>",
        r"<strong>HUGIML-specific comparison:</strong>[^<]+</div>": f"<strong>HUGIML-specific comparison:</strong> {hug_interp}</div>",
    }
    for pat, repl in replacements.items():
        text = re.sub(pat, repl, text)

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
    text = _ensure_dashboard_scope_summary_ui(text)
    text = _ensure_dashboard_profile_ui(text)
    text = _ensure_dashboard_complexity_rendering(text)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(text)


def assemble_outputs(checkpoint: Path, out_dir: Path, template_html: Path) -> dict[str, Path]:
    payload = load_checkpoint(checkpoint)
    details = payload.get("results", [])
    order = {d: i for i, d in enumerate(DATASET_NAMES)}
    mo = {m: i for i, m in enumerate(MODEL_ORDER)}
    details = sorted(
        details, key=lambda r: (order.get(r.get("dataset"), 999), mo.get(r.get("model"), 999))
    )
    expected = len(DATASET_NAMES) * len(MODEL_ORDER)
    if len({(r.get("dataset"), r.get("model")) for r in details}) != expected:
        missing = [
            (d, m)
            for d, m in pair_plan(DATASET_NAMES, MODEL_ORDER)
            if (d, m) not in {(r.get("dataset"), r.get("model")) for r in details}
        ]
        raise RuntimeError(
            f"checkpoint is incomplete: {len(details)} rows, missing {len(missing)} pairs, first missing={missing[:5]}"
        )
    data = make_data(details)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_checkpoint.json").write_text(
        json.dumps(_safe_jsonable(payload), indent=2, allow_nan=False)
    )
    (out_dir / "dashboard_data.json").write_text(
        json.dumps(_safe_jsonable(data), separators=(",", ":"), allow_nan=False)
    )
    pd.DataFrame(data["details"]).to_csv(out_dir / "details.csv", index=False)
    pd.DataFrame(data["overall"]).to_csv(out_dir / "overall.csv", index=False)
    pd.DataFrame(data.get("summary_by_scope", [])).to_csv(
        out_dir / "summary_by_scope.csv", index=False
    )
    pd.DataFrame(data.get("scope_tests", [])).to_csv(out_dir / "scope_tests.csv", index=False)
    summary_comparison(template_html, data, out_dir / "summary_comparison.csv")
    render_html(template_html, data, out_dir / "hugiml_benchmark_analysis_dashboard_revised.html")
    return {
        "checkpoint": out_dir / "benchmark_checkpoint.json",
        "data": out_dir / "dashboard_data.json",
        "details_csv": out_dir / "details.csv",
        "overall_csv": out_dir / "overall.csv",
        "summary_by_scope_csv": out_dir / "summary_by_scope.csv",
        "scope_tests_csv": out_dir / "scope_tests.csv",
        "summary_csv": out_dir / "summary_comparison.csv",
        "html": out_dir / "hugiml_benchmark_analysis_dashboard_revised.html",
    }


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
        "--row-cap",
        type=int,
        default=-1,
        help="Cap rows per dataset. Use -1 for the full dataset, which is the default",
    )
    ap.add_argument(
        "--hugiml-max-fit-seconds",
        type=float,
        default=None,
        help="Optional base parameter for constrained environments",
    )
    args = ap.parse_args(argv)

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
        paths = assemble_outputs(checkpoint, out_dir, template_html)
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

    grid = get_hugiml_grid(DEFAULT_HUGIML_GRID_NAME)
    assert grid.get("feature_mode") == ["original_plus_patterns"]
    assert grid.get("topK") == [50, 100]
    assert grid.get("L") == [1, 2]
    assert grid.get("G") == [0.01, 0.001]

    payload = load_checkpoint(checkpoint)
    payload["metadata"].update(
        {
            "random_state": RANDOM_STATE,
            "row_cap": args.row_cap,
            "hugiml_max_fit_seconds": args.hugiml_max_fit_seconds,
            "grid_snapshot": grid_snapshot(),
            "out_dir": str(out_dir),
            "checkpoint": str(checkpoint),
            "template_html": str(template_html),
        }
    )
    done = completed_keys(payload) if args.resume else set()

    for dataset, model in plan:
        if (dataset, model) in done:
            print(f"skip {dataset} :: {model}", flush=True)
            continue
        print(f"run {dataset} :: {model}", flush=True)
        started = time.perf_counter()
        row = run_pair(
            dataset, model, row_cap=args.row_cap, hugiml_max_fit_seconds=args.hugiml_max_fit_seconds
        )
        row["pair_seconds"] = float(time.perf_counter() - started)
        payload["results"] = [
            r
            for r in payload["results"]
            if not (r.get("dataset") == dataset and r.get("model") == model)
        ]
        payload["results"].append(row)
        save_checkpoint(checkpoint, payload)
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "model": model,
                    "auc": row.get("auc"),
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

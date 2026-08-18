"""Reusable two-model baselines for binary-treatment causal analysis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from packaging.version import Version
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hugiml.hyperparameter_configs import get_baseline_grid

_PROB_EPS = 1e-6
_STANDARD_HUG_GRIDS = {"performance", "interpretability"}


def baseline_grid(name: str, hug_grid: str | dict[str, list]) -> dict[str, list]:
    """Return a baseline search grid matched to a HUG grid family."""
    registry_name = {"T-LR": "LogisticRegression", "T-XGB": "XGBoost", "T-LGBM": "LightGBM"}.get(
        name
    )
    grid = get_baseline_grid(registry_name) if registry_name else None
    if grid is None:
        raise ValueError(f"No registered baseline grid for {name}.")
    if hug_grid == "quick":
        return {key: [values[0]] for key, values in grid.items()}
    if isinstance(hug_grid, str) and hug_grid in _STANDARD_HUG_GRIDS:
        if name == "T-LR":
            grid["C"] = [0.1, 1.0]
        elif name == "T-XGB":
            grid["min_child_weight"] = [1]
        elif name == "T-LGBM":
            grid["min_child_samples"] = [20]
    return grid


def clip_probabilities(values: np.ndarray) -> np.ndarray:
    """Clip probability estimates away from numerical endpoints."""
    return np.clip(np.asarray(values, dtype=float), _PROB_EPS, 1.0 - _PROB_EPS)


def build_baseline_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    """Build preprocessing and estimation steps for mixed tabular data."""
    categorical = [column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])]
    numeric = [column for column in X.columns if column not in categorical]
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    preprocessing = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("pre", preprocessing), ("model", estimator)])


def baseline_estimator_factory(name: str, random_state: int, params: dict[str, Any] | None = None):
    """Create a configured estimator factory for a supported baseline."""
    params = dict(params or {})
    if name == "T-LR":
        if "penalty" in params and Version(sklearn.__version__) >= Version("1.8"):
            penalty = params.pop("penalty")
            params["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
        return lambda: LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=random_state, **params
        )
    if name == "T-XGB":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise RuntimeError("T-XGB requires the causal optional dependencies.") from exc
        defaults = {
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "eval_metric": "logloss",
            "n_jobs": 2,
            "random_state": random_state,
        }
        defaults.update(params)
        return lambda: XGBClassifier(**defaults)
    if name == "T-LGBM":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:
            raise RuntimeError("T-LGBM requires the causal optional dependencies.") from exc
        defaults = {
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "verbosity": -1,
            "n_jobs": 2,
            "random_state": random_state,
        }
        defaults.update(params)
        return lambda: LGBMClassifier(**defaults)
    raise ValueError(f"Unknown baseline: {name}")


class TwoModelBaseline:
    """Fit separate potential-outcome models using a common estimator family."""

    def __init__(self, estimator_factory, covariates: Sequence[str]):
        self.estimator_factory = estimator_factory
        self.covariates = list(covariates)
        self.models_: dict[int, Pipeline] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray, treatment: str):
        treatment_values = pd.to_numeric(X[treatment], errors="raise").astype(int).to_numpy()
        for group in (0, 1):
            mask = treatment_values == group
            if mask.sum() < 10 or np.unique(y[mask]).size < 2:
                label = "control" if group == 0 else "treatment"
                raise ValueError(
                    f"The {label} group has insufficient outcome variation for baseline fitting."
                )
            group_data = X.loc[mask, self.covariates]
            self.models_[group] = build_baseline_pipeline(group_data, self.estimator_factory()).fit(
                group_data, y[mask]
            )
        return self

    def potential_outcomes(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        data = X.loc[:, self.covariates]
        p0 = np.asarray(self.models_[0].predict_proba(data)[:, 1], dtype=float)
        p1 = np.asarray(self.models_[1].predict_proba(data)[:, 1], dtype=float)
        return clip_probabilities(p0), clip_probabilities(p1)


def _joint_split(treatment: np.ndarray, outcome: np.ndarray, test_size: float, random_state: int):
    indices = np.arange(len(outcome))
    labels = 2 * treatment + outcome
    try:
        train, test = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=labels
        )
    except ValueError:
        train, test = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=treatment
        )
    return np.asarray(train), np.asarray(test)


def _candidate_score(y_true: np.ndarray, probability: np.ndarray, scoring: str) -> float:
    probability = clip_probabilities(probability)
    if scoring == "roc_auc":
        return (
            float(roc_auc_score(y_true, probability)) if np.unique(y_true).size == 2 else -math.inf
        )
    return -float(log_loss(y_true, probability, labels=[0, 1]))


def fit_tuned_baseline(
    frame: pd.DataFrame,
    y: np.ndarray,
    treatment_values: np.ndarray,
    train_idx: np.ndarray,
    *,
    treatment: str,
    covariates: Sequence[str],
    name: str,
    random_state: int,
    scoring: str,
    hug_grid: str | dict[str, list],
    tuning_fraction: float = 0.20,
) -> tuple[TwoModelBaseline, dict[str, Any]]:
    """Tune and refit a supported two-model baseline."""
    relative_fit, relative_validation = _joint_split(
        treatment_values[train_idx], y[train_idx], tuning_fraction, random_state + 1701
    )
    tune_fit = train_idx[relative_fit]
    tune_validation = train_idx[relative_validation]
    candidates = list(ParameterGrid(baseline_grid(name, hug_grid)))
    baseline = TwoModelBaseline(baseline_estimator_factory(name, random_state), covariates)
    details = {"selection_scoring": scoring, "candidate_count": len(candidates), "groups": []}
    for group in (0, 1):
        fit_rows = tune_fit[treatment_values[tune_fit] == group]
        validation_rows = tune_validation[treatment_values[tune_validation] == group]
        best_score = -math.inf
        best_params: dict[str, Any] | None = None
        for params in candidates:
            estimator = build_baseline_pipeline(
                frame.loc[fit_rows, covariates],
                baseline_estimator_factory(name, random_state, params)(),
            )
            estimator.fit(frame.loc[fit_rows, covariates], y[fit_rows])
            probability = np.asarray(
                estimator.predict_proba(frame.loc[validation_rows, covariates])[:, 1],
                dtype=float,
            )
            score = _candidate_score(y[validation_rows], probability, scoring)
            if score > best_score:
                best_score, best_params = score, dict(params)
        if best_params is None:
            raise ValueError(f"No valid {name} candidate for group {group}.")
        group_rows = train_idx[treatment_values[train_idx] == group]
        final_model = build_baseline_pipeline(
            frame.loc[group_rows, covariates],
            baseline_estimator_factory(name, random_state, best_params)(),
        )
        baseline.models_[group] = final_model.fit(frame.loc[group_rows, covariates], y[group_rows])
        details["groups"].append(
            {"group": group, "best_score": best_score, "best_params": best_params}
        )
    return baseline, details


__all__ = [
    "TwoModelBaseline",
    "baseline_estimator_factory",
    "baseline_grid",
    "build_baseline_pipeline",
    "clip_probabilities",
    "fit_tuned_baseline",
]

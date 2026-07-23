"""Dash Workbench model modes and guided parameter selection."""

from __future__ import annotations

import copy
import itertools
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from hugiml.dashboard.workbench import (
    _build_estimator,
    _cv_predictions,
    _default_model_params,
    _feature_importance,
    _fresh_estimator,
    _metric_row,
    _run_single_model,
    _safe_numeric_frame,
    _scalar_metric,
)

COMPARISON_MODEL_KEYS = {
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "ebm": "EBM",
    "rulefit": "RuleFit",
}

DASH_GUIDED_GRIDS: dict[str, dict[str, dict[str, list[Any]]]] = {
    "Logistic Regression": {
        "standard": {"C": [0.1, 1.0, 10.0], "max_iter": [1000]},
    },
    "Decision Tree": {
        "standard": {"max_depth": [3, 4, 6, None], "min_samples_leaf": [1, 2, 5]},
    },
    "Random Forest": {
        "standard": {
            "n_estimators": [200, 400],
            "max_depth": [4, 8],
            "min_samples_leaf": [1, 5],
        },
        "budgeted": {
            "n_estimators": [25, 50],
            "max_depth": [None],
            "min_samples_leaf": [1, 5],
        },
    },
    "XGBoost": {
        "standard": {
            "n_estimators": [100, 200],
            "max_depth": [3, 4],
            "learning_rate": [0.03, 0.1],
        },
        "budgeted": {
            "n_estimators": [25, 50],
            "max_depth": [1, 2],
            "learning_rate": [0.03, 0.1],
        },
    },
    "LightGBM": {
        "standard": {
            "n_estimators": [100, 200],
            "max_depth": [-1],
            "learning_rate": [0.03, 0.1],
            "num_leaves": [15, 31],
        },
        "budgeted": {
            "n_estimators": [25, 50],
            "max_depth": [-1],
            "learning_rate": [0.03, 0.1],
            "num_leaves": [2, 4],
        },
    },
    "EBM": {
        "standard": {"max_bins": [32, 64], "interactions": [0, 5]},
    },
    "RuleFit": {
        "standard": {"tree_size": [5, 10], "max_rules": [50, 100]},
    },
}


def comparison_guided_grid(
    model_name: str,
    grid_name: str | None = None,
) -> dict[str, list[Any]]:
    """Return a copy of one comparison-model named grid."""
    grids = DASH_GUIDED_GRIDS.get(model_name, {})
    if not grids:
        return {}
    resolved = str(grid_name or next(iter(grids)))
    if resolved not in grids:
        resolved = next(iter(grids))
    return copy.deepcopy(grids[resolved])


def comparison_guided_grid_names(model_name: str) -> list[str]:
    """Return the named grids available for one comparison model."""
    return list(DASH_GUIDED_GRIDS.get(model_name, {}))


def parameter_grid_count(grid: dict[str, list[Any]]) -> int:
    """Return the Cartesian candidate count for a parameter grid."""
    count = 1
    for values in grid.values():
        count *= max(1, len(list(values)))
    return int(count)


def guided_model_tune_params(
    model_name: str,
    grid_name: str,
    param_grid: dict[str, list[Any]],
) -> dict[str, Any]:
    """Return one outer configuration for guided parameter selection."""
    return {
        "__guided_tune__": True,
        "__label__": f"guided_{str(grid_name).replace(' ', '_').lower()}",
        "__display_name__": f"Guided ({grid_name})",
        "grid_name": str(grid_name),
        "param_grid": copy.deepcopy(param_grid),
        "base_params": _default_model_params(model_name),
    }


def _parameter_candidates(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    active: dict[str, list[Any]] = {}
    for key, values in grid.items():
        candidates = list(values)
        if candidates:
            active[key] = candidates
    if not active:
        return [{}]
    keys = list(active)
    return [dict(zip(keys, values)) for values in itertools.product(*(active[key] for key in keys))]


def _guided_comparison_run(
    model_name: str,
    category: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int,
    random_state: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw_params = dict(params)
    raw_params.pop("__guided_tune__", None)
    raw_params.pop("__label__", None)
    display_name = str(raw_params.pop("__display_name__", "Guided"))
    grid_name = str(raw_params.pop("grid_name", "standard"))
    param_grid = copy.deepcopy(raw_params.pop("param_grid", {}) or {})
    base_params = dict(raw_params.pop("base_params", {}) or {})
    if not param_grid:
        raise ValueError(f"Guided grid '{grid_name}' for {model_name} is empty.")

    X_model = _safe_numeric_frame(X)
    candidate_rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_score: float | None = None
    best_sort_score = float("-inf")
    best_predictions: tuple[np.ndarray, np.ndarray] | None = None

    for candidate_index, candidate in enumerate(_parameter_candidates(param_grid), start=1):
        candidate_params = dict(base_params)
        candidate_params.update(candidate)
        try:
            estimator = _build_estimator(model_name, candidate_params, random_state)
            y_pred, y_proba, score = _cv_predictions(
                estimator,
                X_model,
                y,
                cv,
                random_state,
            )
            scalar_score = _scalar_metric(score)
            sort_score = scalar_score if scalar_score is not None else float("-inf")
            candidate_rows.append(
                {
                    "candidate_index": candidate_index,
                    "status": "ok",
                    "params": copy.deepcopy(candidate),
                    "score": scalar_score,
                    "diagnostic": None,
                }
            )
            if best_params is None or sort_score > best_sort_score:
                best_params = copy.deepcopy(candidate)
                best_score = scalar_score
                best_sort_score = sort_score
                best_predictions = (y_pred, y_proba)
        except Exception as exc:
            candidate_rows.append(
                {
                    "candidate_index": candidate_index,
                    "status": "failed",
                    "params": copy.deepcopy(candidate),
                    "score": None,
                    "diagnostic": repr(exc),
                }
            )

    if best_params is None or best_predictions is None:
        diagnostics = [
            str(row.get("diagnostic")) for row in candidate_rows if row.get("diagnostic")
        ]
        detail = diagnostics[0] if diagnostics else "No candidate completed successfully."
        raise ValueError(f"Guided selection failed for {model_name}: {detail}")

    selected_params = dict(base_params)
    selected_params.update(best_params)
    model = _fresh_estimator(_build_estimator(model_name, selected_params, random_state))
    model.fit(X_model, y)
    y_pred, y_proba = best_predictions
    metrics = _metric_row(y, y_pred, y_proba)
    metrics["cv_roc_auc"] = best_score
    run_params = {
        "mode": "guided_tune",
        "grid_name": grid_name,
        "selected_params": copy.deepcopy(best_params),
        "candidate_count": parameter_grid_count(param_grid),
        "param_grid": copy.deepcopy(param_grid),
        "display_name": display_name,
    }
    tuning_result = SimpleNamespace(
        best_estimator_=model,
        best_params_=copy.deepcopy(best_params),
        best_score_=best_score,
        results_=candidate_rows,
        status_="ok",
        error_=None,
    )
    artifact = {
        "model": model,
        "params": run_params,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_importance": _feature_importance(model, X_model),
        "feature_frame": X_model,
        "confusion_matrix": confusion_matrix(y, y_pred),
        "evaluation_scope": "out_of_fold",
        "evaluation_error": None,
        "evaluation_cv": int(cv),
        "evaluation_random_state": int(random_state),
        "implementation": getattr(model, "backend_", "native"),
        "tuning_result": tuning_result,
    }
    return {
        "model": model_name,
        "category": category,
        "status": "ok",
        "params": run_params,
        **metrics,
        "fit_time_sec": round(time.perf_counter() - started, 4),
        "artifact": artifact,
    }


def run_dash_experiments(
    selected: dict[str, list[str]],
    param_map: dict[str, list[dict[str, Any]]],
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """Run Dash Workbench configurations with guided selection as one outer run."""
    runs: list[dict[str, Any]] = []
    for category, names in selected.items():
        for model_name in names:
            configs = param_map.get(model_name, [_default_model_params(model_name)])
            multi = len(configs) > 1
            for index, params in enumerate(configs, start=1):
                label_params = dict(params)
                try:
                    if model_name != "HUGIML" and label_params.get("__guided_tune__"):
                        row = _guided_comparison_run(
                            model_name,
                            category,
                            label_params,
                            X,
                            y,
                            cv,
                            random_state,
                        )
                    else:
                        row = _run_single_model(
                            model_name,
                            category,
                            label_params,
                            X,
                            y,
                            cv,
                            random_state,
                        )
                except Exception as exc:
                    row = {
                        "model": model_name,
                        "category": category,
                        "status": "failed",
                        "params": label_params,
                        "diagnostic": repr(exc),
                        "fit_time_sec": None,
                        "artifact": None,
                    }
                base_id = model_name.replace(" ", "_").lower()
                if (
                    model_name == "HUGIML"
                    and isinstance(label_params, dict)
                    and label_params.get("__label__")
                ):
                    base_id = f"{base_id}_{label_params.get('__label__')}"
                row["run_id"] = f"{base_id}_{index}" if multi else base_id
                row["config_index"] = index if multi else None
                runs.append(row)
    return runs

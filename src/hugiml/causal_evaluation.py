"""Reusable causal model evaluation and comparison workflows."""

from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from hugiml import HUGIMLClassifier
from hugiml.causal import CausalHUGClassifier as T_HUGClassifier
from hugiml.causal import summarize_cross_fitted_effects
from hugiml.causal_baselines import (
    build_baseline_pipeline,
    clip_probabilities,
    fit_tuned_baseline,
)

_ALLOWED_SELECTION_SCORING = {"roc_auc", "neg_log_loss"}
_ALLOWED_HUG_GRIDS = {
    "quick",
    "performance",
    "interpretability",
    "performance_ho",
    "interpretability_ho",
}


def hug_grid_config(hug_grid: str | dict[str, list]) -> str | dict[str, list]:
    if hug_grid != "quick":
        return hug_grid
    grid = HUGIMLClassifier.default_param_grid("performance")
    return {key: [values[0]] for key, values in grid.items()}


@dataclasses.dataclass
class ModelResult:
    name: str
    model: Any
    p0: np.ndarray
    p1: np.ndarray
    tau: np.ndarray
    test_metrics: dict[str, float]
    fit_seconds: float
    details: dict[str, Any]


def _joint_split(t: np.ndarray, y: np.ndarray, test_size: float, random_state: int):
    idx = np.arange(len(y))
    labels = 2 * t + y
    try:
        tr, te = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=labels
        )
    except ValueError:
        tr, te = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=t)
    return np.asarray(tr), np.asarray(te)


def _factual_metrics(
    y: np.ndarray, t: np.ndarray, p0: np.ndarray, p1: np.ndarray, idx: np.ndarray
) -> dict[str, float]:
    p = clip_probabilities(np.where(t[idx] == 1, p1[idx], p0[idx]))
    out = {"brier": float(brier_score_loss(y[idx], p))}
    out["auc"] = float(roc_auc_score(y[idx], p)) if np.unique(y[idx]).size == 2 else math.nan
    return out


def _validate_analysis_input(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    baselines: Sequence[str],
    test_size: float,
    selection_scoring: str,
    hug_grid: str | dict[str, list],
) -> list[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Analysis requires a non-empty pandas DataFrame.")
    if treatment == outcome:
        raise ValueError("Treatment and outcome must be different columns.")
    cov = [str(c) for c in covariates]
    if not cov:
        raise ValueError("Select at least one adjustment or pre-treatment covariate.")
    if len(cov) != len(set(cov)):
        raise ValueError("Adjustment covariates must be unique.")
    missing = [c for c in [treatment, outcome, *cov] if c not in df.columns]
    if missing:
        raise ValueError("Missing analysis columns: " + ", ".join(dict.fromkeys(missing)))
    if treatment in cov or outcome in cov:
        raise ValueError("Treatment/outcome cannot appear in the adjustment covariate set.")
    empty = [c for c in cov if df[c].notna().sum() == 0]
    if empty:
        raise ValueError("Adjustment covariates contain no observed values: " + ", ".join(empty))
    allowed_baselines = {"T-LR", "T-XGB", "T-LGBM"}
    unknown = sorted(set(baselines) - allowed_baselines)
    if unknown:
        raise ValueError("Unknown baselines: " + ", ".join(unknown))
    if not 0.05 <= float(test_size) <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")
    if selection_scoring not in _ALLOWED_SELECTION_SCORING:
        raise ValueError("T-HUG selection scoring must be 'roc_auc' or 'neg_log_loss'.")
    if isinstance(hug_grid, str) and hug_grid not in _ALLOWED_HUG_GRIDS:
        raise ValueError(
            "Dashboard HUG grid must be quick, performance, interpretability, performance_ho, or interpretability_ho."
        )
    return cov


def _fit_models_on_training(
    frame: pd.DataFrame,
    y: np.ndarray,
    t: np.ndarray,
    train_idx: np.ndarray,
    *,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    baselines: Sequence[str],
    random_state: int,
    hug_grid: str | dict[str, list],
    selection_scoring: str,
    evaluation_idx: np.ndarray | None = None,
) -> dict[str, ModelResult]:
    evaluation_idx = np.asarray(train_idx if evaluation_idx is None else evaluation_idx)
    results: dict[str, ModelResult] = {}
    start = time.perf_counter()
    t_hug = T_HUGClassifier(
        treatment=treatment,
        covariates=covariates,
        param_grid=hug_grid_config(hug_grid),
        scoring=selection_scoring,
        random_state=random_state,
        min_arm_rows=50,
        min_arm_events=4,
        hug_base_params={"n_jobs": 1},
    ).fit(frame.iloc[train_idx].drop(columns=[outcome]), y[train_idx])
    p0, p1 = t_hug.predict_potential_outcomes(frame.drop(columns=[outcome]))
    p0, p1 = clip_probabilities(p0), clip_probabilities(p1)
    results["T-HUG"] = ModelResult(
        "T-HUG",
        t_hug,
        p0,
        p1,
        p1 - p0,
        _factual_metrics(y, t, p0, p1, evaluation_idx),
        time.perf_counter() - start,
        {
            "group_models": t_hug.summary().to_dict("records"),
            "selection_scoring": selection_scoring,
        },
    )
    for name in baselines:
        start = time.perf_counter()
        baseline, tuning_details = fit_tuned_baseline(
            frame,
            y,
            t,
            train_idx,
            treatment=treatment,
            covariates=covariates,
            name=name,
            random_state=random_state,
            scoring=selection_scoring,
            hug_grid=hug_grid,
        )
        p0b, p1b = baseline.potential_outcomes(frame)
        results[name] = ModelResult(
            name,
            baseline,
            p0b,
            p1b,
            p1b - p0b,
            _factual_metrics(y, t, p0b, p1b, evaluation_idx),
            time.perf_counter() - start,
            tuning_details,
        )
    return results


def _cross_fitted_robustness(
    frame: pd.DataFrame,
    y: np.ndarray,
    t: np.ndarray,
    *,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    baselines: Sequence[str],
    random_state: int,
    hug_grid: str | dict[str, list],
    selection_scoring: str,
    study_type: str,
    n_splits: int,
    seeds: Sequence[int],
) -> dict[str, Any]:
    labels = 2 * t + y.astype(int)
    if min(np.bincount(labels, minlength=4)) < n_splits:
        raise ValueError(
            "Each treatment/outcome combination needs at least one row per cross-fitting fold."
        )
    names = ["T-HUG", *baselines]
    p0 = {name: np.full((len(seeds), len(frame)), np.nan) for name in names}
    p1 = {name: np.full((len(seeds), len(frame)), np.nan) for name in names}
    propensity = np.full((len(seeds), len(frame)), np.nan)
    for repeat, repeat_seed in enumerate(seeds):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(repeat_seed))
        for fold, (train_idx, validation_idx) in enumerate(splitter.split(frame, labels)):
            model_seed = int(repeat_seed) + 1009 * (fold + 1)
            fold_results = _fit_models_on_training(
                frame,
                y,
                t,
                np.asarray(train_idx),
                treatment=treatment,
                outcome=outcome,
                covariates=covariates,
                baselines=baselines,
                random_state=model_seed,
                hug_grid=hug_grid,
                selection_scoring=selection_scoring,
                evaluation_idx=np.asarray(validation_idx),
            )
            for name, result in fold_results.items():
                p0[name][repeat, validation_idx] = result.p0[validation_idx]
                p1[name][repeat, validation_idx] = result.p1[validation_idx]
            if str(study_type).lower() == "randomized":
                fold_propensity = np.repeat(float(np.mean(t[train_idx])), len(validation_idx))
            else:
                propensity_model = build_baseline_pipeline(
                    frame.loc[train_idx, covariates],
                    LogisticRegression(max_iter=1000, solver="liblinear", random_state=model_seed),
                ).fit(frame.loc[train_idx, covariates], t[train_idx])
                fold_propensity = propensity_model.predict_proba(
                    frame.loc[validation_idx, covariates]
                )[:, 1]
            propensity[repeat, validation_idx] = clip_probabilities(fold_propensity)
    if not np.isfinite(propensity).all() or any(
        not np.isfinite(values).all() for name in names for values in (p0[name], p1[name])
    ):
        raise RuntimeError(
            "Cross-fitting did not produce one out-of-fold prediction per row and repetition."
        )
    summary = summarize_cross_fitted_effects(
        y, t, {name: (p0[name], p1[name]) for name in names}, propensity
    )
    return {
        "estimates": summary.estimates,
        "overlap_sensitivity": summary.overlap_sensitivity,
        "folds": int(n_splits),
        "repeats": int(len(seeds)),
        "seeds": [int(seed) for seed in seeds],
        "propensity_model": "Known assignment probability"
        if str(study_type).lower() == "randomized"
        else "Cross-fitted logistic regression",
    }


def fit_analysis(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    baselines: Sequence[str] = ("T-LR", "T-XGB"),
    random_state: int = 2026,
    test_size: float = 0.25,
    hug_grid: str | dict[str, list] = "performance_ho",
    selection_scoring: str = "roc_auc",
    robust_seeds: Sequence[int] | None = None,
    robust_folds: int = 2,
    study_type: str = "observational",
) -> dict[str, Any]:
    cov = _validate_analysis_input(
        df, treatment, outcome, covariates, baselines, test_size, selection_scoring, hug_grid
    )
    valid = df[treatment].notna() & df[outcome].notna()
    source_positions = np.flatnonzero(np.asarray(valid))
    row_index = np.asarray(df.index)[source_positions]
    frame = df.loc[valid, [*cov, treatment, outcome]].copy().reset_index(drop=True)
    if len(frame) < 100:
        raise ValueError("At least 100 complete treatment/outcome rows are required.")
    try:
        t = pd.to_numeric(frame[treatment], errors="raise").astype(int).to_numpy()
        y = pd.to_numeric(frame[outcome], errors="raise").astype(int).to_numpy()
    except Exception as exc:
        raise ValueError("Treatment and outcome must be binary and encoded 0/1.") from exc
    if set(np.unique(t)) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1 groups.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Outcome must contain both 0 and 1 classes.")
    for group in (0, 1):
        mask = t == group
        if mask.sum() < 50:
            label = "Control" if group == 0 else "Treatment"
            raise ValueError(f"{label} group requires at least 50 rows.")
        if min(np.sum(y[mask] == 0), np.sum(y[mask] == 1)) < 4:
            label = "Control" if group == 0 else "Treatment"
            raise ValueError(f"{label} group requires at least four events and four non-events.")

    tr, te = _joint_split(t, y, test_size, int(random_state))
    results = _fit_models_on_training(
        frame,
        y,
        t,
        tr,
        treatment=treatment,
        outcome=outcome,
        covariates=cov,
        baselines=baselines,
        random_state=int(random_state),
        hug_grid=hug_grid,
        selection_scoring=selection_scoring,
        evaluation_idx=te,
    )

    robustness = None
    robustness_error = None
    if robust_seeds is not None:
        seeds = [int(seed) for seed in robust_seeds]
        if len(seeds) < 2 or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
            raise ValueError(
                "robust_seeds must contain at least two distinct non-negative integers."
            )
        if int(robust_folds) < 2:
            raise ValueError("robust_folds must be at least two.")
        try:
            robustness = _cross_fitted_robustness(
                frame,
                y,
                t,
                treatment=treatment,
                outcome=outcome,
                covariates=cov,
                baselines=baselines,
                random_state=int(random_state),
                hug_grid=hug_grid,
                selection_scoring=selection_scoring,
                study_type=study_type,
                n_splits=int(robust_folds),
                seeds=seeds,
            )
        except Exception as exc:
            robustness_error = str(exc)

    return {
        "frame": frame,
        "row_index": row_index,
        "source_positions": source_positions,
        "t": t,
        "y": y,
        "train_idx": tr,
        "test_idx": te,
        "results": results,
        "covariates": cov,
        "treatment": treatment,
        "outcome": outcome,
        "selection_scoring": selection_scoring,
        "hug_grid": hug_grid if isinstance(hug_grid, str) else "custom",
        "robustness": robustness,
        "robustness_error": robustness_error,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return math.nan
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def comparison_frame(
    analysis: dict[str, Any], oracle_tau: np.ndarray | None = None
) -> pd.DataFrame:
    rows = []
    test_idx = np.asarray(analysis["test_idx"])
    for name, result in analysis["results"].items():
        row = {
            "Model": name,
            "ATE": float(np.mean(result.tau)),
            "Mean P(Y|do(T=0))": float(np.mean(result.p0)),
            "Mean P(Y|do(T=1))": float(np.mean(result.p1)),
            "CATE SD": float(np.std(result.tau)),
            "Held-out AUC": result.test_metrics.get("auc", math.nan),
            "Held-out Brier": result.test_metrics.get("brier", math.nan),
            "Fit seconds": result.fit_seconds,
        }
        if oracle_tau is not None and len(oracle_tau) == len(result.tau):
            truth = np.asarray(oracle_tau, dtype=float)
            row["Oracle CATE RMSE"] = float(
                np.sqrt(np.mean((result.tau[test_idx] - truth[test_idx]) ** 2))
            )
            row["Oracle CATE corr"] = _safe_corr(result.tau[test_idx], truth[test_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def _selection_metric_label(scoring: str | None) -> str:
    return "ROC AUC" if scoring == "roc_auc" else "Negative log loss"


def _compact_params(params: dict[str, Any], *, t_hug: bool = False) -> str:
    values = dict(params or {})
    if t_hug:
        base_estimator = values.get("base_estimator")
        branch = (
            "LR"
            if base_estimator is None or isinstance(base_estimator, LogisticRegression)
            else "RPTE"
        )
        values = {key: values[key] for key in ("L", "topK", "G", "feature_mode") if key in values}
        values["downstream"] = branch
    return json.dumps(values, sort_keys=True, default=str, separators=(", ", ": "))


def selected_configurations_frame(analysis: dict[str, Any]) -> pd.DataFrame:
    """Summarize the winning group-level tuning configuration for every model."""
    rows = []
    for name, result in analysis["results"].items():
        details = result.details or {}
        scoring = details.get("selection_scoring", analysis.get("selection_scoring"))
        if name == "T-HUG":
            model = result.model
            params_by_group = getattr(model, "best_params_", {})
            scores_by_group = getattr(model, "best_scores_", {})
            tuning_by_group = getattr(model, "tuning_results_", {})
            for group in (0, 1):
                tuning = tuning_by_group.get(group, {}) or {}
                candidates = tuning.get("cv_results", [])
                rows.append(
                    {
                        "Model": name,
                        "Group": "Control" if group == 0 else "Treatment",
                        "Selection metric": _selection_metric_label(scoring),
                        "Best score": scores_by_group.get(group, math.nan),
                        "Candidates evaluated": len(candidates),
                        "Selected parameters": _compact_params(
                            params_by_group.get(group, {}), t_hug=True
                        ),
                    }
                )
            continue
        for group_record in details.get("groups", []):
            group = int(group_record.get("group", 0))
            rows.append(
                {
                    "Model": name,
                    "Group": "Control" if group == 0 else "Treatment",
                    "Selection metric": _selection_metric_label(scoring),
                    "Best score": group_record.get("best_score", math.nan),
                    "Candidates evaluated": details.get("candidate_count", 0),
                    "Selected parameters": _compact_params(group_record.get("best_params", {})),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Group",
            "Selection metric",
            "Best score",
            "Candidates evaluated",
            "Selected parameters",
        ],
    )


def t_hug_selection_frame(analysis: dict[str, Any]) -> pd.DataFrame:
    result = analysis["results"]["T-HUG"]
    rows = []
    metric = (
        "ROC AUC" if result.details.get("selection_scoring") == "roc_auc" else "Negative log loss"
    )
    for group, record in enumerate(result.details.get("group_models", [])[:2]):
        rows.append(
            {
                "Group": "Control" if group == 0 else "Treatment",
                "Downstream model": record.get("grid_branch", ""),
                "Selection metric": metric,
                "Selection score": record.get("grid_auc", math.nan),
                "Fit rows": record.get("n_fit", 0),
                "Event rate": record.get("event_rate_fit", math.nan),
                "HUG patterns": record.get("n_patterns", 0),
            }
        )
    return pd.DataFrame(rows)


def t_hug_region_frame(analysis: dict[str, Any], max_patterns_per_group: int = 12) -> pd.DataFrame:
    t_hug: T_HUGClassifier = analysis["results"]["T-HUG"].model
    Z = analysis["frame"].loc[:, analysis["covariates"]]
    result = analysis["results"]["T-HUG"]
    rows: dict[str, dict[str, Any]] = {}
    for group in (0, 1):
        model = t_hug.models_[group]
        try:
            info = (
                model.get_pattern_info()
                .sort_values("utility", ascending=False)
                .head(max_patterns_per_group)
            )
            matrix = model.transform_patterns(Z)
            names = list(model.get_hug_features())
            name_to_idx = {str(n): i for i, n in enumerate(names)}
        except Exception:
            continue
        prefix = "Control" if group == 0 else "Treatment"
        for record in info.to_dict("records"):
            pattern = str(record.get("pattern", ""))
            idx = name_to_idx.get(pattern)
            if idx is None:
                continue
            mask = np.asarray(matrix[:, idx].toarray()).ravel() > 0
            if mask.sum() == 0:
                continue
            row = rows.setdefault(
                pattern,
                {
                    "Region / HUG pattern": pattern,
                    "Control utility": np.nan,
                    "Treatment utility": np.nan,
                    "Control support": np.nan,
                    "Treatment support": np.nan,
                },
            )
            row[f"{prefix} utility"] = record.get("utility", np.nan)
            row[f"{prefix} support"] = record.get("support", np.nan)
            row.update(
                {
                    "Population N": int(mask.sum()),
                    "Population %": float(mask.mean()),
                    "Mean P0": float(np.mean(result.p0[mask])),
                    "Mean P1": float(np.mean(result.p1[mask])),
                    "Mean CATE": float(np.mean(result.tau[mask])),
                }
            )
    out = pd.DataFrame(rows.values())
    if out.empty:
        return out
    out["Structure"] = np.where(
        out["Control utility"].notna() & out["Treatment utility"].notna(),
        "shared",
        np.where(out["Treatment utility"].notna(), "treatment-only", "control-only"),
    )
    return out.sort_values(["Population %", "Mean CATE"], ascending=[False, True]).reset_index(
        drop=True
    )


def t_hug_rpte_frame(analysis: dict[str, Any]) -> pd.DataFrame:
    """Return RPTE leaf and direct-rule evidence by control or treatment group."""
    t_hug: T_HUGClassifier = analysis["results"]["T-HUG"].model
    parts = []
    for group in (0, 1):
        model = t_hug.models_[group]
        try:
            rows = model.rpte_rule_table()
        except Exception:
            rows = []
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        frame.insert(0, "Group", "Control" if group == 0 else "Treatment")
        for col in ("raw_conditions", "raw_sources"):
            if col in frame.columns:
                frame[col] = frame[col].map(
                    lambda value: (
                        " AND ".join(map(str, value))
                        if isinstance(value, list) and col == "raw_conditions"
                        else (", ".join(map(str, value)) if isinstance(value, list) else value)
                    )
                )
        keep = [
            c
            for c in [
                "Group",
                "tree_index",
                "leaf_index",
                "term_role",
                "raw_conditions",
                "raw_sources",
                "support_rate",
                "final_logistic_coefficient",
                "source_selection_status",
            ]
            if c in frame.columns
        ]
        parts.append(frame.loc[:, keep])
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()

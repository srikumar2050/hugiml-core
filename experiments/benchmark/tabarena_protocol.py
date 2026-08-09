from __future__ import annotations

import copy
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

TABARENA_INNER_FOLDS = 8
TABARENA_ELO_BOOTSTRAPS = 200


def _complete_hugiml_split_errors(
    rows: Sequence[dict[str, Any]],
    *,
    problem_scope: str = "all",
) -> tuple[dict[str, dict[int, float]], list[dict[str, Any]]]:
    """Return outer-split errors for fully completed HUGIML datasets."""
    if problem_scope not in {"all", "binary", "multiclass"}:
        raise ValueError(f"Unknown TabArena problem scope: {problem_scope}")
    by_dataset: dict[str, dict[int, float]] = {}
    excluded: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("model")) != "HUGIML":
            continue
        n_classes = int(row.get("n_classes") or 2)
        if problem_scope == "binary" and n_classes != 2:
            continue
        if problem_scope == "multiclass" and n_classes <= 2:
            continue
        expected = int(row.get("official_splits_expected") or 0)
        successful = int(row.get("official_splits_successful") or 0)
        split_errors: dict[int, float] = {}
        for split in row.get("split_rows", []):
            if split.get("status") != "ok":
                continue
            value = split.get("roc_auc") if n_classes == 2 else split.get("log_loss")
            if value is None or not math.isfinite(float(value)):
                continue
            split_id = int(split.get("split_id", len(split_errors)))
            split_errors[split_id] = 1.0 - float(value) if n_classes == 2 else float(value)
        complete = (
            row.get("status") == "ok"
            and expected > 0
            and successful == expected
            and len(split_errors) == expected
        )
        if complete:
            by_dataset[str(row.get("dataset"))] = split_errors
        else:
            excluded.append(
                {
                    "dataset": str(row.get("dataset")),
                    "status": str(row.get("status")),
                    "successful_splits": successful,
                    "expected_splits": expected,
                }
            )
    return by_dataset, excluded


def official_reference_comparison(
    rows: Sequence[dict[str, Any]],
    official_results_path: str | Any,
    *,
    regime: str = "all",
    model_scope: str = "all",
    problem_scope: str = "all",
    bootstrap_rounds: int = TABARENA_ELO_BOOTSTRAPS,
    random_state: int = 0,
) -> dict[str, Any]:
    """Compare completed HUGIML datasets with official TabArena split results.

    Outer-split errors are matched by dataset name and split number. They are
    averaged within each dataset before leaderboard aggregation so datasets with
    30 outer splits receive the same weight as datasets with 9 outer splits.
    """
    official = pd.read_parquet(official_results_path)
    required = {"dataset", "fold", "method", "metric_error"}
    missing = sorted(required.difference(official.columns))
    if missing:
        raise ValueError(f"Official TabArena artifact is missing columns: {missing}")
    regime_suffixes = {
        "default": "(default)",
        "tuned": "(tuned)",
        "tuned_ensemble": "(tuned + ensemble)",
    }
    if regime not in {"all", *regime_suffixes}:
        raise ValueError(f"Unknown official TabArena comparison regime: {regime}")
    if regime != "all":
        suffix = regime_suffixes[regime]
        official = official[official["method"].astype(str).str.endswith(suffix)].copy()
    if model_scope not in {"all", "non_neural", "interpretable"}:
        raise ValueError(f"Unknown official TabArena model scope: {model_scope}")
    if model_scope == "non_neural":
        non_neural_sources = {
            "CatBoost", "ExplainableBM", "ExtraTrees", "KNeighbors",
            "LightGBM", "LinearModel", "RandomForest", "XGBoost",
        }
        official = official[official["source_model"].astype(str).isin(non_neural_sources)].copy()
    elif model_scope == "interpretable":
        official = official[
            official["source_model"].astype(str).isin({"ExplainableBM", "LinearModel"})
        ].copy()
    local, excluded = _complete_hugiml_split_errors(rows, problem_scope=problem_scope)
    matched_datasets: list[str] = []
    hugiml_dataset_error: dict[str, float] = {}
    official_dataset_errors: dict[tuple[str, str], float] = {}
    official_methods = sorted(str(value) for value in official["method"].dropna().unique())
    for dataset, split_errors in local.items():
        subset = official[official["dataset"].astype(str) == dataset].copy()
        if subset.empty:
            excluded.append({"dataset": dataset, "reason": "absent_from_official_artifact"})
            continue
        expected_folds = set(split_errors)
        method_errors: dict[str, float] = {}
        valid = True
        for method in official_methods:
            method_rows = subset[subset["method"].astype(str) == method]
            values = {
                int(record.fold): float(record.metric_error)
                for record in method_rows[["fold", "metric_error"]].itertuples(index=False)
                if pd.notna(record.metric_error)
            }
            if set(values) != expected_folds:
                valid = False
                break
            method_errors[method] = float(np.mean([values[index] for index in sorted(values)]))
        if not valid:
            excluded.append({"dataset": dataset, "reason": "outer_split_mismatch"})
            continue
        matched_datasets.append(dataset)
        hugiml_dataset_error[dataset] = float(
            np.mean([split_errors[index] for index in sorted(split_errors)])
        )
        for method, value in method_errors.items():
            official_dataset_errors[(dataset, method)] = value

    methods = ["HUGIML", *official_methods]
    errors = np.asarray(
        [
            [
                hugiml_dataset_error[dataset]
                if method == "HUGIML"
                else official_dataset_errors[(dataset, method)]
                for method in methods
            ]
            for dataset in matched_datasets
        ],
        dtype=float,
    )
    if not len(errors):
        return {
            "available": False,
            "regime": regime,
            "model_scope": model_scope,
            "problem_scope": problem_scope,
            "matched_dataset_count": 0,
            "excluded_datasets": excluded,
            "leaderboard": [],
            "hugiml_pairwise": [],
        }

    ranks = np.vstack([pd.Series(row).rank(method="average").to_numpy() for row in errors])
    best = np.min(errors, axis=1)
    median = np.median(errors, axis=1)
    denominator = median - best
    normalized = np.where(
        denominator[:, None] > 0,
        np.clip((median[:, None] - errors) / denominator[:, None], 0.0, 1.0),
        (errors == best[:, None]).astype(float),
    )
    improvability = np.where(
        errors > 0,
        np.clip((errors - best[:, None]) / errors * 100.0, 0.0, 100.0),
        0.0,
    )
    elo_mean = _elo_mean_centered(errors)
    preferred_references = {
        "default": "RF (default)",
        "tuned": "RF (tuned)",
        "tuned_ensemble": "RF (tuned + ensemble)",
        "all": "RF (tuned)",
    }
    if model_scope == "interpretable":
        preferred_reference = {
            "default": "LR (default)",
            "tuned": "LR (tuned)",
            "tuned_ensemble": "LR (tuned + ensemble)",
            "all": "LR (tuned)",
        }[regime]
    else:
        preferred_reference = preferred_references[regime]
    reference = preferred_reference if preferred_reference in methods else methods[1]
    shift = 1000.0 - elo_mean[methods.index(reference)]
    rng = np.random.default_rng(random_state)
    samples = np.empty((bootstrap_rounds, len(methods)), dtype=float)
    for iteration in range(bootstrap_rounds):
        selection = rng.integers(0, len(errors), size=len(errors))
        sample_rating = _elo_mean_centered(errors[selection])
        samples[iteration] = sample_rating - sample_rating[methods.index(reference)] + 1000.0
    elo = elo_mean + shift
    lower = np.quantile(samples, 0.025, axis=0)
    upper = np.quantile(samples, 0.975, axis=0)
    leaderboard: list[dict[str, Any]] = []
    for index, method in enumerate(methods):
        leaderboard.append(
            {
                "method": method,
                "dataset_count": len(matched_datasets),
                "mean_error": float(np.mean(errors[:, index])),
                "median_error": float(np.median(errors[:, index])),
                "average_rank": float(np.mean(ranks[:, index])),
                "normalized_score": float(np.mean(normalized[:, index])),
                "mean_improvability_percent": float(np.mean(improvability[:, index])),
                "median_improvability_percent": float(np.median(improvability[:, index])),
                "elo": float(elo[index]),
                "elo_ci_lower": float(lower[index]),
                "elo_ci_upper": float(upper[index]),
            }
        )
    leaderboard.sort(key=lambda row: (-row["elo"], row["average_rank"], row["method"]))

    hug_index = methods.index("HUGIML")
    pairwise: list[dict[str, Any]] = []
    for index, method in enumerate(methods):
        if method == "HUGIML":
            continue
        delta = errors[:, index] - errors[:, hug_index]
        tolerance = 1e-12
        pairwise.append(
            {
                "method": method,
                "wins": int(np.sum(delta > tolerance)),
                "ties": int(np.sum(np.abs(delta) <= tolerance)),
                "losses": int(np.sum(delta < -tolerance)),
                "mean_hugiml_error_reduction_percent": float(
                    np.mean(np.divide(delta, errors[:, index], out=np.zeros_like(delta), where=errors[:, index] > 0))
                    * 100.0
                ),
                "median_hugiml_error_reduction_percent": float(
                    np.median(np.divide(delta, errors[:, index], out=np.zeros_like(delta), where=errors[:, index] > 0))
                    * 100.0
                ),
            }
        )
    pairwise.sort(key=lambda row: (-row["mean_hugiml_error_reduction_percent"], row["method"]))
    return {
        "available": True,
        "regime": regime,
        "model_scope": model_scope,
        "problem_scope": problem_scope,
        "matched_dataset_count": len(matched_datasets),
        "matched_datasets": matched_datasets,
        "excluded_datasets": excluded,
        "official_method_count": len(official_methods),
        "elo_reference_method": reference,
        "elo_bootstrap_rounds": bootstrap_rounds,
        "leaderboard": leaderboard,
        "hugiml_pairwise": pairwise,
        "methodology": {
            "matching": "dataset name and exact outer split number",
            "dataset_weighting": "equal weight after averaging outer-split error within each dataset",
            "binary_error": "1 - ROC AUC",
            "multiclass_error": "log loss",
            "scope": "fully completed local HUGIML datasets with complete official results for every reference method",
            "comparability_note": "Outer test splits and metrics match. Inner model selection, search spaces, compute budgets, and retained ensembles remain method-specific.",
        },
    }


def detailed_metric_comparison(
    rows: Sequence[dict[str, Any]],
    detailed_results_path: str | Any,
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate aligned HUGIML and official predictive metrics."""
    official = pd.read_parquet(detailed_results_path)
    required = {
        "dataset", "fold", "source_model", "regime", "problem_type",
        "roc_auc", "balanced_accuracy", "f1", "brier",
    }
    missing = sorted(required.difference(official.columns))
    if missing:
        raise ValueError(f"Detailed TabArena artifact is missing columns: {missing}")
    local_records: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("model")) != "HUGIML" or row.get("status") != "ok":
            continue
        expected = int(row.get("official_splits_expected") or 0)
        successful = int(row.get("official_splits_successful") or 0)
        if expected <= 0 or successful != expected:
            continue
        problem_type = "binary" if int(row.get("n_classes") or 2) == 2 else "multiclass"
        for split in row.get("split_rows", []):
            if split.get("status") != "ok":
                continue
            local_records.append(
                {
                    "dataset": str(row.get("dataset")),
                    "fold": int(split.get("split_id", 0)),
                    "source_model": "HUGIML",
                    "regime": "hugiml",
                    "problem_type": problem_type,
                    "roc_auc": split.get("roc_auc"),
                    "balanced_accuracy": split.get("balanced_accuracy"),
                    "f1": split.get("f1"),
                    "brier": split.get("brier"),
                }
            )
    local = pd.DataFrame(local_records)
    if local.empty:
        return {}
    datasets = set(local["dataset"])
    official = official[official["dataset"].astype(str).isin(datasets)].copy()
    official["dataset"] = official["dataset"].astype(str)
    metrics = ("roc_auc", "balanced_accuracy", "f1", "brier")
    output: dict[str, list[dict[str, Any]]] = {}
    for regime in ("default", "tuned"):
        combined = pd.concat(
            [local, official[official["regime"].astype(str) == regime]],
            ignore_index=True,
        )
        combined["method"] = np.where(
            combined["source_model"].astype(str) == "HUGIML",
            "HUGIML",
            combined["source_model"].astype(str),
        )
        for scope in ("overall", "binary", "multiclass"):
            scoped = combined if scope == "overall" else combined[combined["problem_type"] == scope]
            per_dataset = scoped.groupby(["dataset", "method"], as_index=False)[list(metrics)].mean()
            records: list[dict[str, Any]] = []
            hug = per_dataset[per_dataset["method"] == "HUGIML"].set_index("dataset")
            for method, group in per_dataset.groupby("method"):
                aligned = group.set_index("dataset").join(
                    hug[list(metrics)], how="inner", rsuffix="_hugiml"
                )
                if aligned.empty:
                    continue
                record: dict[str, Any] = {"method": str(method), "dataset_count": len(aligned)}
                for metric in metrics:
                    values = aligned[metric].dropna().to_numpy(float)
                    record[f"{metric}_mean"] = float(np.mean(values)) if len(values) else None
                    record[f"{metric}_median"] = float(np.median(values)) if len(values) else None
                    paired = aligned[[metric, f"{metric}_hugiml"]].dropna()
                    if metric == "brier":
                        delta = paired[metric] - paired[f"{metric}_hugiml"]
                    else:
                        delta = paired[f"{metric}_hugiml"] - paired[metric]
                    record[f"{metric}_hugiml_delta"] = (
                        float(delta.mean()) if len(delta) else None
                    )
                records.append(record)
            records.sort(
                key=lambda item: (
                    -(item.get("roc_auc_mean") or float("-inf")),
                    item["method"],
                )
            )
            output[f"{regime}_{scope}"] = records
    return output


@dataclass
class CrossValidatedEnsemble:
    """Arithmetic-probability ensemble retained from inner CV."""

    estimators_: list[Any]
    classes_: np.ndarray
    probability_fn: Callable[..., np.ndarray]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        matrices = [
            self.probability_fn(
                estimator,
                X,
                len(self.classes_),
                trained_class_labels=self.classes_,
            )
            for estimator in self.estimators_
        ]
        return np.mean(np.stack(matrices, axis=0), axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def selection_score(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, str]:
    """Return the TabArena classification selection score (larger is better)."""
    if proba.shape[1] == 2:
        return float(roc_auc_score(y_true, proba[:, 1])), "roc_auc"
    labels = np.arange(proba.shape[1], dtype=int)
    return -float(log_loss(y_true, proba, labels=labels)), "negative_log_loss"


def fit_cross_validated_ensemble(
    *,
    candidates: Sequence[dict[str, Any]],
    X: pd.DataFrame,
    y: np.ndarray,
    random_state: int,
    fit_child: Callable[[dict[str, Any], pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray], Any],
    probability_fn: Callable[..., np.ndarray],
    n_splits: int = TABARENA_INNER_FOLDS,
    prepare_fold: Callable[[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray], Any]
    | None = None,
) -> tuple[CrossValidatedEnsemble, dict[str, Any], float, float, dict[str, Any]]:
    """Select a configuration by OOF performance and retain all fold models."""
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("TabArena inner CV requires at least two classes.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(X, y))
    for fold_index, (train_idx, _) in enumerate(folds):
        training_classes = np.unique(y[train_idx])
        if not np.array_equal(training_classes, classes):
            missing = np.setdiff1d(classes, training_classes).tolist()
            raise ValueError(
                "TabArena inner CV produced a child-training fold without all "
                f"classes; fold={fold_index}, missing_classes={missing}."
            )
    n_classes = int(np.unique(y).size)
    started = time.perf_counter()
    prepared_folds: list[
        tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, Any]
    ] = []
    for train_idx, validation_idx in folds:
        X_fit = X.iloc[train_idx].reset_index(drop=True).copy()
        X_validation = X.iloc[validation_idx].reset_index(drop=True).copy()
        y_fit = np.asarray(y[train_idx], dtype=int)
        y_validation = np.asarray(y[validation_idx], dtype=int)
        prepared = (
            None
            if prepare_fold is None
            else prepare_fold(X_fit, y_fit, X_validation, y_validation)
        )
        prepared_folds.append(
            (validation_idx, X_fit, y_fit, X_validation, y_validation, prepared)
        )
    best: tuple[float, int, dict[str, Any], list[Any]] | None = None
    candidate_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for candidate_index, raw_params in enumerate(candidates):
        params = copy.deepcopy(raw_params)
        oof = np.zeros((len(y), n_classes), dtype=float)
        children: list[Any] = []
        candidate_started = time.perf_counter()
        try:
            for (
                validation_idx,
                X_fit,
                y_fit,
                X_validation,
                y_validation,
                prepared,
            ) in prepared_folds:
                if prepare_fold is None:
                    child = fit_child(params, X_fit, y_fit, X_validation, y_validation)
                else:
                    child = fit_child(
                        params,
                        X_fit,
                        y_fit,
                        X_validation,
                        y_validation,
                        prepared,
                    )
                oof[validation_idx] = probability_fn(
                    child,
                    X_validation,
                    n_classes,
                    trained_class_labels=np.arange(n_classes, dtype=int),
                )
                children.append(child)
            score, metric = selection_score(y, oof)
            candidate_rows.append(
                {
                    "candidate_index": candidate_index,
                    "score": score,
                    "metric": metric,
                    "fit_count": len(children),
                    "elapsed_seconds": time.perf_counter() - candidate_started,
                    "status": "ok",
                }
            )
            item = (score, -candidate_index, params, children)
            if best is None or item[:2] > best[:2]:
                best = item
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            candidate_rows.append(
                {
                    "candidate_index": candidate_index,
                    "score": None,
                    "metric": None,
                    "fit_count": len(children),
                    "elapsed_seconds": time.perf_counter() - candidate_started,
                    "status": "error",
                    "error": errors[-1],
                }
            )
    if best is None:
        raise RuntimeError("All TabArena CV-ensemble candidates failed: " + " | ".join(errors[-3:]))
    ensemble = CrossValidatedEnsemble(
        estimators_=best[3],
        classes_=np.arange(n_classes, dtype=int),
        probability_fn=probability_fn,
    )
    metric = "roc_auc" if n_classes == 2 else "negative_log_loss"
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return ensemble, best[2], best[0], elapsed_ms, {
        "_final_refit_ms": 0.0,
        "inner_scoring": metric,
        "selection_refit_performed": False,
        "retained_cv_ensemble": True,
        "cv_ensemble_child_count": int(n_splits),
        "candidate_count": len(candidates),
        "candidate_fit_count": sum(int(row["fit_count"]) for row in candidate_rows),
        "candidate_error_count": len(errors),
        "preprocessing_fit_count": (
            int(n_splits) if prepare_fold is not None else None
        ),
        "preprocessing_reuse_count": (
            int(n_splits) * max(0, len(candidates) - 1)
            if prepare_fold is not None
            else None
        ),
        "validation_selection_metrics": [metric],
        "candidate_rows": candidate_rows,
    }


def _dataset_errors(rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[int], np.ndarray]:
    models = sorted({str(row["model"]) for row in rows})
    datasets = sorted({int(row["task_id"]) for row in rows})
    table = np.full((len(datasets), len(models)), np.nan, dtype=float)
    model_index = {model: i for i, model in enumerate(models)}
    dataset_index = {dataset: i for i, dataset in enumerate(datasets)}
    for row in rows:
        n_classes = int(row.get("n_classes", 2))
        split_errors: list[float] = []
        for split in row.get("split_rows", []):
            metric = split.get("roc_auc") if n_classes == 2 else split.get("log_loss")
            if metric is None or not math.isfinite(float(metric)):
                continue
            split_errors.append(1.0 - float(metric) if n_classes == 2 else float(metric))
        if not split_errors:
            continue
        error = float(np.mean(split_errors))
        table[dataset_index[int(row["task_id"])], model_index[str(row["model"])]] = error
    return models, datasets, table


def tabarena_aggregate_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute dataset-balanced TabArena alternative leaderboard metrics."""
    models, _, errors = _dataset_errors(rows)
    complete = np.isfinite(errors).all(axis=1)
    errors = errors[complete]
    if not len(errors):
        return []
    ranks = np.vstack([pd.Series(row).rank(method="average").to_numpy() for row in errors])
    best = np.min(errors, axis=1)
    median = np.median(errors, axis=1)
    denom = median - best
    normalized = np.where(
        denom[:, None] > 0,
        np.clip((median[:, None] - errors) / denom[:, None], 0.0, None),
        (errors == best[:, None]).astype(float),
    )
    improvability = np.where(
        errors > 0,
        np.clip((errors - best[:, None]) / errors * 100.0, 0.0, 100.0),
        0.0,
    )
    output: list[dict[str, Any]] = []
    for index, model in enumerate(models):
        output.append(
            {
                "model": model,
                "dataset_count": int(len(errors)),
                "normalized_score": float(np.mean(normalized[:, index])),
                "average_rank": float(np.mean(ranks[:, index])),
                "harmonic_mean_rank": float(1.0 / np.mean(1.0 / ranks[:, index])),
                "improvability_percent": float(np.mean(improvability[:, index])),
                "pairwise_win_rate": float(
                    np.mean(
                        [
                            np.mean(errors[:, index] < errors[:, other])
                            + 0.5 * np.mean(errors[:, index] == errors[:, other])
                            for other in range(len(models))
                            if other != index
                        ]
                    )
                ),
            }
        )
    return output


def _elo_mean_centered(errors: np.ndarray) -> np.ndarray:
    n_models = errors.shape[1]
    battles: list[tuple[int, int, float]] = []
    for row in errors:
        for left in range(n_models):
            for right in range(left + 1, n_models):
                outcome = 1.0 if row[left] < row[right] else (0.0 if row[left] > row[right] else 0.5)
                battles.append((left, right, outcome))
    left_indices = np.asarray([battle[0] for battle in battles], dtype=int)
    right_indices = np.asarray([battle[1] for battle in battles], dtype=int)
    outcomes = np.asarray([battle[2] for battle in battles], dtype=float)

    def objective(free: np.ndarray) -> tuple[float, np.ndarray]:
        rating = np.r_[free, -np.sum(free)]
        delta = np.clip(rating[left_indices] - rating[right_indices], -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-delta))
        loss = -float(
            np.sum(
                outcomes * np.log(np.maximum(probability, 1e-15))
                + (1.0 - outcomes) * np.log(np.maximum(1.0 - probability, 1e-15))
            )
        )
        gradient = np.zeros(n_models, dtype=float)
        residual = probability - outcomes
        np.add.at(gradient, left_indices, residual)
        np.add.at(gradient, right_indices, -residual)
        return loss, gradient[:-1] - gradient[-1]

    result = minimize(
        lambda value: objective(value)[0],
        np.zeros(n_models - 1, dtype=float),
        jac=lambda value: objective(value)[1],
        method="BFGS",
    )
    logits = np.r_[result.x, -np.sum(result.x)]
    return logits * (400.0 / math.log(10.0))


def tabarena_elo_metrics(
    rows: Sequence[dict[str, Any]],
    *, reference_model: str = "RandomForest standard",
    bootstrap_rounds: int = TABARENA_ELO_BOOTSTRAPS,
    random_state: int = 0,
) -> list[dict[str, Any]]:
    models, _, errors = _dataset_errors(rows)
    errors = errors[np.isfinite(errors).all(axis=1)]
    if not len(errors) or reference_model not in models:
        return []
    point_mean = _elo_mean_centered(errors)
    shift = 1000.0 - point_mean[models.index(reference_model)]
    rng = np.random.default_rng(random_state)
    samples = np.empty((bootstrap_rounds, len(models)), dtype=float)
    for index in range(bootstrap_rounds):
        selected = rng.integers(0, len(errors), size=len(errors))
        samples[index] = _elo_mean_centered(errors[selected])
    lower = np.quantile(samples, 0.025, axis=0) + shift
    upper = np.quantile(samples, 0.975, axis=0) + shift
    point = point_mean + shift
    return [
        {
            "model": model,
            "dataset_count": int(len(errors)),
            "elo": float(point[index]),
            "elo_ci_lower": float(lower[index]),
            "elo_ci_upper": float(upper[index]),
            "elo_reference_model": reference_model,
            "elo_bootstrap_rounds": int(bootstrap_rounds),
        }
        for index, model in enumerate(models)
    ]

"""Model training/scoring helpers for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class SimpleTuneResult:
    best_estimator_: Any | None
    best_params_: dict
    best_score_: float | None
    results_: list[dict]
    cv_splits_: list | None = None
    fast_path_used_: bool = False
    status_: str = "ok"
    error_: str | None = None


@dataclass
class PrunedRepresentationResult:
    label: str
    estimator: Any
    score: float | None
    rows: list[dict]
    kept_columns: list[str]
    removed_columns: list[str]
    family: str


def _make_classifier(**params):
    from hugiml import HUGIMLClassifierNative

    return HUGIMLClassifierNative(**params)


_HUGIML_PARAM_KEYS = {
    "adaptive_binning",
    "B",
    "L",
    "topK",
    "G",
    "feature_mode",
    "topk_budget_strict",
    "augmented_pair_transforms",
    "augmented_pair_mode",
    "aug_feature_size",
    "max_pair_features",
    "ii_partner_size",
    "interaction_relaxed_mining",
    "interaction_relaxed_feature_size",
}


def _default_params() -> dict:
    return {
        "adaptive_binning": True,
        "B": -1,
        "L": 2,
        "topK": 100,
        "G": 1e-2,
        "feature_mode": "original_plus_patterns",
        "topk_budget_strict": False,
        "augmented_pair_transforms": True,
        "augmented_pair_mode": "interaction_information",
        "aug_feature_size": 10,
        "max_pair_features": 10,
        "ii_partner_size": None,
        "interaction_relaxed_mining": False,
        "interaction_relaxed_feature_size": 10,
    }


def _safe_params(model: Any) -> dict:
    out = {}
    for k in _HUGIML_PARAM_KEYS:
        if hasattr(model, k):
            out[k] = getattr(model, k)
    if not out:
        out = _default_params()
    return {**_default_params(), **out}


def _is_empty_pattern_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "patterns list is empty" in text or ("pattern" in text and "empty" in text and "build" in text)


def _error_result(params: dict, message: str, status: str = "failed") -> SimpleTuneResult:
    return SimpleTuneResult(
        best_estimator_=None,
        best_params_=params,
        best_score_=None,
        results_=[{"status": status, "message": message, **params}],
        cv_splits_=None,
        fast_path_used_=False,
        status_=status,
        error_=message,
    )


def train_hugiml(X: pd.DataFrame, y, cv: int = 5, scoring: str = "roc_auc", random_state: int = 2026):
    from hugiml import HUGIMLClassifierNative

    if hasattr(HUGIMLClassifierNative, "tune"):
        return HUGIMLClassifierNative.tune(
            X,
            y,
            cv=cv,
            shuffle=True,
            random_state=random_state,
            scoring=scoring,
            param_grid=None,
            refit=True,
            use_fast_path=True,
        )

    return fit_hugiml_config(X, y, params=_default_params(), cv=cv, scoring=scoring, random_state=random_state)


def fit_hugiml_config(
    X: pd.DataFrame,
    y,
    params: dict | None = None,
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 2026,
    *,
    raise_on_error: bool = False,
):
    """Fit one explicit HUGIML configuration and return a tune-like result.

    Candidate configurations can legitimately mine zero patterns for a given
    dataset/G/L/topK combination. In that case HUGIML may raise
    "patterns list is empty — nothing to build". The dashboard should display
    that as a failed candidate run, not crash.
    """
    params = {**_default_params(), **(params or {})}
    params["topk_budget_strict"] = bool(params.get("topk_budget_strict", False))

    score = None
    rows = []

    try:
        model_for_cv = _make_classifier(**params)
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = cross_val_score(model_for_cv, X, y, cv=folds, scoring=scoring, error_score="raise")
        score = float(np.mean(scores))
        for i, s in enumerate(scores, start=1):
            rows.append({"fold": i, "score": float(s), "status": "ok", **params})
    except Exception as exc:
        message = repr(exc)
        rows = [{"status": "cv_failed", "message": message, **params}]
        if raise_on_error and not _is_empty_pattern_error(exc):
            raise

    try:
        model = _make_classifier(**params)
        model.fit(X, y)
    except Exception as exc:
        message = repr(exc)
        if raise_on_error:
            raise
        status = "no_patterns" if _is_empty_pattern_error(exc) else "fit_failed"
        return _error_result(params, message, status=status)

    return SimpleTuneResult(
        best_estimator_=model,
        best_params_=params,
        best_score_=score,
        results_=rows,
        cv_splits_=None,
        fast_path_used_=False,
        status_="ok",
        error_=None,
    )


def fit_feature_pruned_hugiml(
    X: pd.DataFrame,
    y,
    base_model: Any | None = None,
    remove_features: list[str] | None = None,
    params: dict | None = None,
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 2026,
):
    """Remove selected original input features, rerun HUGIML, and return result + pruned frame."""
    remove_features = [str(c) for c in (remove_features or [])]
    remove_set = set(remove_features)
    keep_cols = [c for c in X.columns if str(c) not in remove_set]
    if not keep_cols:
        raise ValueError("Pruning removed all model input features; keep at least one feature.")

    X_pruned = X.loc[:, keep_cols].copy()

    if params is None:
        params = _safe_params(base_model) if base_model is not None else _default_params()

    params = {k: v for k, v in params.items() if k in _HUGIML_PARAM_KEYS}
    result = fit_hugiml_config(X_pruned, y, params=params, cv=cv, scoring=scoring, random_state=random_state)
    return result, X_pruned


def _transform_downstream(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Best-effort downstream representation transform for pattern/augmented pruning."""
    candidates = [
        "transform_downstream",
        "transform_features",
        "transform",
        "_transform_downstream",
        "_transform_features",
    ]
    for name in candidates:
        fn = getattr(model, name, None)
        if fn is None:
            continue
        try:
            Z = fn(X)
            if hasattr(Z, "toarray"):
                Z = Z.toarray()
            if isinstance(Z, pd.DataFrame):
                return Z.copy()
            arr = np.asarray(Z)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            cols = [f"repr_{i+1}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols, index=X.index)
        except Exception:
            continue

    raise RuntimeError(
        "This HUGIML version does not expose a downstream representation transform. "
        "Original-feature pruning is available; pattern/augmented representation pruning "
        "requires transform/downstream columns from the fitted model."
    )


def _fit_downstream_estimator_on_representation(Z: pd.DataFrame, y, cv: int, scoring: str, random_state: int):
    """Fit a simple auditable downstream estimator on selected representation columns."""
    estimator = LogisticRegression(max_iter=1000, solver="lbfgs")

    score = None
    rows = []
    try:
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = cross_val_score(estimator, Z, y, cv=folds, scoring=scoring)
        score = float(np.mean(scores))
        for i, s in enumerate(scores, start=1):
            rows.append({"fold": i, "score": float(s)})
    except Exception as exc:
        rows = [{"status": "cv_failed", "message": repr(exc)}]

    fitted = clone(estimator)
    fitted.fit(Z, y)
    return fitted, score, rows


def fit_representation_pruned_downstream(
    base_model: Any,
    X: pd.DataFrame,
    y,
    remove_columns: list[str],
    family: str,
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 2026,
) -> tuple[PrunedRepresentationResult, pd.DataFrame]:
    """Remove selected downstream representation columns and refit downstream estimator."""
    remove_columns = [str(c) for c in (remove_columns or [])]
    Z = _transform_downstream(base_model, X)
    Z.columns = [str(c) for c in Z.columns]
    remove_set = set(remove_columns)
    keep_cols = [c for c in Z.columns if c not in remove_set]
    if not keep_cols:
        raise ValueError("Representation pruning removed all downstream columns; keep at least one column.")

    Z_pruned = Z.loc[:, keep_cols].copy()
    estimator, score, rows = _fit_downstream_estimator_on_representation(
        Z_pruned,
        y,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
    )
    return (
        PrunedRepresentationResult(
            label=f"{family} representation-pruned downstream",
            estimator=estimator,
            score=score,
            rows=rows,
            kept_columns=keep_cols,
            removed_columns=remove_columns,
            family=family,
        ),
        Z_pruned,
    )


def score_cases(model, X: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return pd.DataFrame({"prediction": pred, "probability": proba})

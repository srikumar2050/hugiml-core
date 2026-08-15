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
    "adaptive_binning_sample_frac",
    "adaptive_binning_sample_random_state",
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
    # Preserved so a promoted RPTE-based model keeps using RPTE when
    # feature-removal pruning re-fits HUGIML on a reduced input set --
    # without this, _safe_params()/fit_feature_pruned_hugiml() would
    # silently drop it back to the built-in logistic-regression branch.
    "base_estimator",
}


def _default_params() -> dict:
    return {
        "adaptive_binning": True,
        "adaptive_binning_sample_frac": False,
        "adaptive_binning_sample_random_state": 42,
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
            # tune() evaluates every search candidate under
            # execution_mode='production' for speed and refits the returned
            # best_estimator_ under 'audit' by default -- this dashboard's
            # own components (patterns, feature_family, pruning, prediction,
            # governance_evidence, overview) call get_pattern_info() on the
            # trained model, which needs that 'audit' state; an explicit
            # override here would silently degrade those views to a sparser
            # fallback instead of erroring, since each call site wraps
            # get_pattern_info() in a try/except.
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


def _rpte_final_representation(model: Any) -> str | None:
    try:
        from hugiml.dashboard.components.patterns import _get_rpte_feature_flow_audit

        audit = _get_rpte_feature_flow_audit(model)
    except Exception:
        audit = {}
    return str(audit.get("final_representation")) if audit else None


def _transform_downstream(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Return the fitted HUGIML source matrix with aligned column names.

    HUGIML's public ``transform()`` returns the fitted downstream source
    representation. Prefer it together with the fitted downstream names so
    representation-column pruning cannot silently target synthetic ``repr_*``
    names or the wrong matrix.
    """
    transform = getattr(model, "transform", None)
    get_names = getattr(model, "get_downstream_features", None)
    if callable(transform) and callable(get_names):
        try:
            Z = transform(X)
            names = [str(v) for v in list(get_names() or [])]
            if hasattr(Z, "toarray"):
                arr = Z.toarray()
            else:
                arr = np.asarray(Z)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if len(names) == arr.shape[1]:
                return pd.DataFrame(arr, columns=names, index=X.index)
        except Exception:
            pass

    candidates = [
        "transform_downstream",
        "transform_features",
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
        "This HUGIML version does not expose an aligned downstream source representation. "
        "Use raw-input exclusion with a full HUGIML rebuild instead."
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
    """Remove selected direct LR representation columns for non-RPTE models.

    RPTE models containing leaf indicators are rejected because a plain LR
    refit on the HUGIML source matrix would discard the fitted leaf block. The
    RPTE final LR may also contain direct source terms, but those terms
    cannot be isolated with this generic source-matrix refit.
    """
    representation = _rpte_final_representation(base_model)
    if representation in {"rpte_leaf_rules", "rpte_leaves_plus_direct_terms"}:
        raise ValueError(
            "The fitted RPTE final LR contains leaf indicators. Direct original, pattern, "
            "and augmented-pair terms may also be appended directly, but a plain LR refit on the "
            "HUGIML source matrix would discard the leaf block and change the model class. Exclude "
            "raw inputs and rebuild the full pipeline, or refit a smaller RPTE configuration."
        )
    requested_columns = [str(c) for c in (remove_columns or [])]
    Z = _transform_downstream(base_model, X)
    Z.columns = [str(c) for c in Z.columns]

    family_prefix = {
        "HUG patterns": "pattern:",
        "Augmented features": "augmented_pair:",
    }.get(str(family), "")
    column_set = set(Z.columns)
    matched: list[str] = []
    for requested in requested_columns:
        candidates = [requested]
        if family_prefix and not requested.startswith(family_prefix):
            candidates.append(f"{family_prefix}{requested}")
        match = next((candidate for candidate in candidates if candidate in column_set), None)
        if match is not None and match not in matched:
            matched.append(match)

    if not matched:
        raise ValueError(
            "None of the requested representation columns exist in the fitted downstream matrix. "
            "No pruning was performed."
        )
    remove_set = set(matched)
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
            removed_columns=matched,
            family=family,
        ),
        Z_pruned,
    )


def score_cases(model, X: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return pd.DataFrame({"prediction": pred, "probability": proba})

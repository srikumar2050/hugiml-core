"""Compact causal diagnostics for dashboard investigation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _validate_inputs(
    df: pd.DataFrame, treatment: str, outcome: str, covariates: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Diagnostics require a non-empty dataset.")
    missing = [c for c in [treatment, outcome, *covariates] if c not in df.columns]
    if missing:
        raise ValueError("Missing diagnostic columns: " + ", ".join(dict.fromkeys(missing)))
    if treatment == outcome:
        raise ValueError("Treatment and outcome must be different columns.")
    if treatment in covariates or outcome in covariates:
        raise ValueError("Treatment/outcome cannot be adjustment variables.")
    if not covariates:
        raise ValueError("Select at least one adjustment or pre-treatment covariate.")
    try:
        t = pd.to_numeric(df[treatment], errors="raise").astype(int).to_numpy()
        y = pd.to_numeric(df[outcome], errors="raise").astype(int).to_numpy()
    except Exception as exc:
        raise ValueError("Treatment and outcome must be binary 0/1 columns.") from exc
    if set(np.unique(t)) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1 groups.")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("Outcome must be encoded 0/1.")
    return t, y


def _numeric_smd(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    bb = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    if len(aa) < 2 or len(bb) < 2:
        return math.nan
    den = np.sqrt((np.var(aa, ddof=1) + np.var(bb, ddof=1)) / 2)
    return 0.0 if den == 0 else float((np.mean(aa) - np.mean(bb)) / den)


def balance_frame(df: pd.DataFrame, treatment: str, covariates: Sequence[str]) -> pd.DataFrame:
    if treatment not in df.columns:
        raise ValueError(f"Treatment column {treatment!r} is missing.")
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise ValueError("Missing adjustment variables: " + ", ".join(missing))
    t = pd.to_numeric(df[treatment], errors="raise").astype(int).to_numpy()
    if set(np.unique(t)) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1 groups.")
    rows = []
    for c in covariates:
        if pd.api.types.is_numeric_dtype(df[c]):
            smd = _numeric_smd(df.loc[t == 1, c], df.loc[t == 0, c])
            rows.append(
                {
                    "Covariate": c,
                    "Type": "numeric",
                    "SMD": smd,
                    "Flag": "Review" if np.isfinite(smd) and abs(smd) > 0.1 else "OK",
                }
            )
        else:
            tab = pd.crosstab(df[c].fillna("<missing>").astype(str), t, normalize="columns")
            gap = float((tab.get(1, 0) - tab.get(0, 0)).abs().max()) if not tab.empty else math.nan
            rows.append(
                {
                    "Covariate": c,
                    "Type": "categorical",
                    "SMD": np.nan,
                    "Max proportion gap": gap,
                    "Flag": "Review" if np.isfinite(gap) and gap > 0.1 else "OK",
                }
            )
    return pd.DataFrame(rows)


def propensity_scores(
    df: pd.DataFrame, treatment: str, covariates: Sequence[str], study_type: str = "observational"
) -> np.ndarray:
    if treatment not in df.columns:
        raise ValueError(f"Treatment column {treatment!r} is missing.")
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise ValueError("Missing adjustment variables: " + ", ".join(missing))
    if not covariates:
        raise ValueError(
            "At least one pre-treatment covariate is required for propensity diagnostics."
        )
    t = pd.to_numeric(df[treatment], errors="raise").astype(int).to_numpy()
    if set(np.unique(t)) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1 groups.")
    if str(study_type).lower() == "randomized":
        return np.repeat(float(np.mean(t)), len(t))
    X = df.loc[:, covariates]
    cat = [c for c in covariates if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in covariates if c not in cat]
    trans = []
    if num:
        trans.append(
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                num,
            )
        )
    if cat:
        trans.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            )
        )
    pipe = Pipeline(
        [
            ("pre", ColumnTransformer(trans)),
            ("lr", LogisticRegression(max_iter=1000, solver="liblinear")),
        ]
    )
    e = np.asarray(pipe.fit(X, t).predict_proba(X)[:, 1], dtype=float)
    return np.clip(e, 1e-6, 1 - 1e-6)


def diagnostic_summary(
    df: pd.DataFrame, treatment: str, outcome: str, covariates: Sequence[str], study_type: str
) -> dict:
    t, y = _validate_inputs(df, treatment, outcome, covariates)
    e = propensity_scores(df, treatment, covariates, study_type)
    return {
        "n": len(df),
        "n0": int((t == 0).sum()),
        "n1": int((t == 1).sum()),
        "y0": float(np.mean(y[t == 0])),
        "y1": float(np.mean(y[t == 1])),
        "raw_diff": float(np.mean(y[t == 1]) - np.mean(y[t == 0])),
        "propensity": e,
        "overlap_rate": float(np.mean((e >= 0.05) & (e <= 0.95))),
        "extreme_low": float(np.mean(e < 0.05)),
        "extreme_high": float(np.mean(e > 0.95)),
        "balance": balance_frame(df, treatment, covariates),
    }

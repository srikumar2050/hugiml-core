"""Data profiling statistics for the Dash Workbench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from scipy.stats import chi2_contingency

PROFILE_ROW_LIMIT = 10_000
CHART_ROW_LIMIT = 10_000
SCATTER_ROW_LIMIT = 5_000
MISSINGNESS_MAP_ROW_LIMIT = 1_000
CORRELATION_COLUMN_LIMIT = 30
CATEGORY_DISPLAY_LIMIT = 20


@dataclass(frozen=True)
class ProfileScope:
    """Rows used for profiling and the sampling description shown in the UI."""

    frame: pd.DataFrame
    sampled: bool
    total_rows: int
    profile_rows: int
    label: str


def _normalise_roles(roles: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(roles or {})
    return {
        "target": values.get("target"),
        "id_column": values.get("id_column"),
        "excluded_columns": list(values.get("excluded_columns") or []),
        "sensitive_columns": list(values.get("sensitive_columns") or []),
    }


def _valid_target(df: pd.DataFrame, target: Any) -> str | None:
    return str(target) if target is not None and str(target) in df.columns else None


def _safe_unique_count(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.dropna().map(repr).nunique(dropna=True))


def _duplicate_row_count(df: pd.DataFrame) -> int:
    try:
        return int(df.duplicated().sum())
    except TypeError:
        safe = df.copy()
        for column in safe.columns:
            if safe[column].dtype == object:
                safe[column] = safe[column].map(
                    lambda value: repr(value) if isinstance(value, (list, dict, set)) else value
                )
        return int(safe.duplicated().sum())


def profile_scope(
    df: pd.DataFrame,
    *,
    mode: str = "sample",
    target: str | None = None,
    limit: int = PROFILE_ROW_LIMIT,
    random_state: int = 2026,
) -> ProfileScope:
    """Return the full frame or a deterministic, target-aware sample."""
    total = int(len(df))
    if total <= limit or str(mode) == "full":
        return ProfileScope(
            frame=df.copy(deep=False),
            sampled=False,
            total_rows=total,
            profile_rows=total,
            label=f"Full dataset · {total:,} rows",
        )

    target_name = _valid_target(df, target)
    if target_name is None:
        sampled = df.sample(n=limit, random_state=random_state, replace=False).sort_index()
    else:
        y = df[target_name]
        groups = y.astype("object").where(y.notna(), "<missing>").astype(str).reset_index(drop=True)
        counts = groups.value_counts(dropna=False)
        if len(counts) > limit:
            sampled = df.sample(n=limit, random_state=random_state, replace=False).sort_index()
            return ProfileScope(
                frame=sampled,
                sampled=True,
                total_rows=total,
                profile_rows=int(len(sampled)),
                label=f"Deterministic sample · {len(sampled):,} of {total:,} rows",
            )
        allocations = (counts / counts.sum() * limit).astype(int).clip(lower=1)
        while int(allocations.sum()) > limit:
            key = allocations.idxmax()
            if allocations.loc[key] <= 1:
                break
            allocations.loc[key] -= 1
        while int(allocations.sum()) < limit:
            candidates = (counts - allocations).sort_values(ascending=False)
            key = candidates.index[0]
            allocations.loc[key] += 1
        chosen_positions: list[int] = []
        rng = np.random.default_rng(random_state)
        for level, amount in allocations.items():
            positions = np.flatnonzero(groups.to_numpy() == level)
            amount = min(int(amount), int(len(positions)))
            if amount:
                chosen_positions.extend(rng.choice(positions, size=amount, replace=False).tolist())
        chosen_set = set(chosen_positions)
        if len(chosen_positions) < limit:
            remaining = np.array([position for position in range(total) if position not in chosen_set])
            amount = min(limit - len(chosen_positions), len(remaining))
            if amount:
                chosen_positions.extend(rng.choice(remaining, size=amount, replace=False).tolist())
        sampled = df.iloc[sorted(chosen_positions[:limit])]

    return ProfileScope(
        frame=sampled,
        sampled=True,
        total_rows=total,
        profile_rows=int(len(sampled)),
        label=f"Deterministic sample · {len(sampled):,} of {total:,} rows",
    )


def semantic_type(series: pd.Series) -> str:
    """Infer a presentation-oriented variable type without mutating data."""
    non_missing = series.dropna()
    if non_missing.empty:
        return "all missing"
    if _safe_unique_count(non_missing) <= 1:
        return "constant"
    if ptypes.is_bool_dtype(series.dtype):
        return "binary"
    if ptypes.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if ptypes.is_numeric_dtype(series.dtype):
        unique = _safe_unique_count(non_missing)
        if unique <= 2:
            return "binary"
        if ptypes.is_integer_dtype(series.dtype) and unique <= min(30, max(8, len(non_missing) // 50)):
            return "discrete numeric"
        return "continuous numeric"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"

    as_text = non_missing.astype(str).str.strip()
    unique = _safe_unique_count(as_text)
    uniqueness = unique / max(len(non_missing), 1)
    if unique <= 2:
        return "binary"

    numeric = pd.to_numeric(as_text.str.replace(",", "", regex=False), errors="coerce")
    if float(numeric.notna().mean()) >= 0.95:
        return "numeric text"

    date_like = as_text.str.contains(r"(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4})|(?:\d{1,2}:\d{2})", regex=True, na=False)
    if float(date_like.mean()) >= 0.8:
        datetime_values = pd.to_datetime(as_text, errors="coerce", utc=True)
        if float(datetime_values.notna().mean()) >= 0.9:
            return "datetime text"

    mean_length = float(as_text.str.len().mean()) if len(as_text) else 0.0
    if uniqueness >= 0.98 and unique >= min(50, max(10, len(non_missing) // 4)):
        return "identifier-like"
    if unique <= min(100, max(20, len(non_missing) // 5)) and mean_length <= 80:
        return "categorical"
    return "text"


def _role_for(name: str, roles: dict[str, Any]) -> str:
    if name == roles.get("target"):
        return "Target"
    if name == roles.get("id_column"):
        return "ID"
    if name in roles.get("excluded_columns", []):
        return "Excluded"
    if name in roles.get("sensitive_columns", []):
        return "Sensitive"
    return "Predictor"


def _safe_mode(series: pd.Series) -> tuple[Any, int]:
    try:
        counts = series.value_counts(dropna=True)
    except TypeError:
        counts = series.dropna().map(repr).value_counts(dropna=True)
    if counts.empty:
        return None, 0
    return counts.index[0], int(counts.iloc[0])


def variable_inventory(df: pd.DataFrame, roles: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return one compact profiling row per variable."""
    role_map = _normalise_roles(roles)
    rows: list[dict[str, Any]] = []
    for position, raw_name in enumerate(df.columns):
        name = str(raw_name)
        series = df.iloc[:, position]
        non_missing = int(series.notna().sum())
        unique = _safe_unique_count(series)
        mode, mode_count = _safe_mode(series)
        inferred = semantic_type(series)
        warning_count = 0
        if inferred in {"all missing", "constant", "identifier-like", "numeric text", "datetime text"}:
            warning_count += 1
        missing_pct = float(series.isna().mean() * 100.0)
        if missing_pct >= 40:
            warning_count += 1
        uniqueness = unique / max(non_missing, 1)
        if inferred == "categorical" and unique > 100:
            warning_count += 1
        rows.append(
            {
                "variable": name,
                "role": _role_for(name, role_map),
                "storage_dtype": str(series.dtype),
                "semantic_type": inferred,
                "non_missing": non_missing,
                "missing_pct": round(missing_pct, 2),
                "unique": unique,
                "unique_pct": round(100.0 * uniqueness, 2),
                "most_frequent": "" if mode is None else str(mode)[:80],
                "mode_count": mode_count,
                "warning_count": warning_count,
                "position": position,
            }
        )
    return pd.DataFrame(rows)


def dataset_summary(df: pd.DataFrame, inventory: pd.DataFrame) -> dict[str, Any]:
    """Dataset-level summary values for cards and findings."""
    total_cells = max(int(df.shape[0] * df.shape[1]), 1)
    missing_cells = int(df.isna().sum().sum())
    type_counts = inventory["semantic_type"].value_counts().to_dict() if not inventory.empty else {}
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": missing_cells,
        "missing_pct": 100.0 * missing_cells / total_cells,
        "duplicate_rows": _duplicate_row_count(df) if len(df) else 0,
        "numeric": int(
            sum(type_counts.get(key, 0) for key in ("continuous numeric", "discrete numeric", "numeric text"))
        ),
        "categorical": int(sum(type_counts.get(key, 0) for key in ("categorical", "binary"))),
        "datetime": int(sum(type_counts.get(key, 0) for key in ("datetime", "datetime text"))),
        "text": int(sum(type_counts.get(key, 0) for key in ("text", "identifier-like"))),
        "constant": int(type_counts.get("constant", 0)),
        "all_missing": int(type_counts.get("all missing", 0)),
    }


def profile_findings(df: pd.DataFrame, inventory: pd.DataFrame, roles: dict[str, Any] | None) -> list[dict[str, str]]:
    """Return concise, actionable profiling observations."""
    role_map = _normalise_roles(roles)
    findings: list[dict[str, str]] = []
    if df.empty:
        return [{"level": "warning", "title": "Empty dataset", "detail": "No rows are available for profiling."}]
    duplicate_rows = _duplicate_row_count(df)
    if duplicate_rows:
        findings.append(
            {
                "level": "warning",
                "title": "Duplicate rows",
                "detail": f"{duplicate_rows:,} rows duplicate another profiled row.",
            }
        )
    for row in inventory.to_dict("records"):
        name = row["variable"]
        inferred = row["semantic_type"]
        if inferred == "all missing":
            findings.append({"level": "warning", "title": name, "detail": "All values are missing."})
        elif inferred == "constant":
            findings.append({"level": "warning", "title": name, "detail": "The variable is constant."})
        elif inferred == "identifier-like" and name != role_map.get("id_column"):
            findings.append(
                {
                    "level": "attention",
                    "title": name,
                    "detail": "High uniqueness suggests an identifier; consider assigning the ID role or excluding it.",
                }
            )
        elif inferred == "numeric text":
            findings.append(
                {
                    "level": "attention",
                    "title": name,
                    "detail": "Values appear numeric but are stored as text.",
                }
            )
        elif inferred == "datetime text":
            findings.append(
                {
                    "level": "attention",
                    "title": name,
                    "detail": "Values appear to be dates but are stored as text.",
                }
            )
        if float(row["missing_pct"]) >= 40:
            findings.append(
                {
                    "level": "warning",
                    "title": name,
                    "detail": f"{row['missing_pct']:.1f}% of values are missing.",
                }
            )
        if inferred == "categorical" and int(row["unique"]) > 100:
            findings.append(
                {
                    "level": "attention",
                    "title": name,
                    "detail": f"High cardinality: {int(row['unique']):,} observed levels.",
                }
            )
    target = _valid_target(df, role_map.get("target"))
    if target is not None:
        counts = df[target].value_counts(dropna=False)
        if len(counts) >= 2 and counts.sum() > 0:
            smallest = float(counts.min() / counts.sum())
            if smallest < 0.1:
                findings.append(
                    {
                        "level": "warning",
                        "title": "Target imbalance",
                        "detail": f"The smallest target class represents {smallest * 100:.1f}% of profiled rows.",
                    }
                )
    if not findings:
        findings.append(
            {
                "level": "ok",
                "title": "No major profile warnings",
                "detail": "The selected profiling scope has no constant, all-missing, or severely incomplete variables.",
            }
        )
    return findings[:30]


def numeric_profile(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = values.dropna()
    if finite.empty:
        return {"count": 0, "missing": int(series.isna().sum())}
    q1, median, q3 = finite.quantile([0.25, 0.5, 0.75]).tolist()
    iqr = q3 - q1
    outliers = int(((finite < q1 - 1.5 * iqr) | (finite > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
    return {
        "count": int(finite.size),
        "missing": int(values.isna().sum()),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "min": float(finite.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(finite.max()),
        "iqr": float(iqr),
        "skewness": float(finite.skew()) if len(finite) > 2 else 0.0,
        "zeros": int((finite == 0).sum()),
        "negative": int((finite < 0).sum()),
        "outliers": outliers,
    }


def categorical_profile(series: pd.Series) -> dict[str, Any]:
    values = series.dropna().astype(str)
    counts = values.value_counts()
    total = max(int(counts.sum()), 1)
    probabilities = counts / total
    entropy = float(-(probabilities * np.log2(probabilities)).sum()) if len(counts) else 0.0
    return {
        "count": int(values.size),
        "missing": int(series.isna().sum()),
        "unique": int(counts.size),
        "mode": None if counts.empty else str(counts.index[0]),
        "mode_count": 0 if counts.empty else int(counts.iloc[0]),
        "mode_pct": 0.0 if counts.empty else float(counts.iloc[0] / total * 100.0),
        "rare_levels": int((counts < max(2, int(total * 0.01))).sum()),
        "entropy": entropy,
    }


def datetime_profile(series: pd.Series) -> dict[str, Any]:
    values = pd.to_datetime(series, errors="coerce", utc=True).dropna()
    if values.empty:
        return {"count": 0, "missing": int(series.isna().sum())}
    ordered = values.sort_values()
    diffs = ordered.diff().dropna()
    common_interval = None
    if not diffs.empty:
        common_interval = str(diffs.value_counts().index[0])
    return {
        "count": int(values.size),
        "missing": int(series.size - values.size),
        "unique": int(values.nunique()),
        "minimum": str(values.min()),
        "maximum": str(values.max()),
        "span": str(values.max() - values.min()),
        "common_interval": common_interval,
    }


def text_profile(series: pd.Series) -> dict[str, Any]:
    values = series.dropna().astype(str)
    lengths = values.str.len()
    return {
        "count": int(values.size),
        "missing": int(series.isna().sum()),
        "unique": int(values.nunique()),
        "unique_pct": float(values.nunique() / max(len(values), 1) * 100.0),
        "average_length": float(lengths.mean()) if len(lengths) else 0.0,
        "maximum_length": int(lengths.max()) if len(lengths) else 0,
        "blank": int(values.str.strip().eq("").sum()),
    }


def missingness_patterns(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    missing_rates = df.isna().mean().sort_values(ascending=False)
    missing_cols = [str(column) for column in missing_rates.index[missing_rates > 0][:50]]
    if not missing_cols:
        return pd.DataFrame(columns=["pattern", "rows", "percentage"])
    mask = df[missing_cols].isna()
    labels = mask.apply(
        lambda row: ", ".join(name for name, missing in zip(missing_cols, row.to_numpy()) if missing)
        or "Complete",
        axis=1,
    )
    counts = labels.value_counts().head(limit)
    return pd.DataFrame(
        {
            "pattern": counts.index,
            "rows": counts.values.astype(int),
            "percentage": (counts.values / max(len(df), 1) * 100.0).round(2),
        }
    )


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x.fillna("<missing>").astype(str), y.fillna("<missing>").astype(str))
    if min(table.shape) < 2 or table.values.sum() == 0:
        return 0.0
    chi2 = float(chi2_contingency(table, correction=False)[0])
    n = float(table.values.sum())
    phi2 = chi2 / n
    r, k = table.shape
    correction = ((k - 1) * (r - 1)) / max(n - 1, 1)
    phi2_corrected = max(0.0, phi2 - correction)
    r_corrected = r - ((r - 1) ** 2) / max(n - 1, 1)
    k_corrected = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(k_corrected - 1, r_corrected - 1)
    return float(np.sqrt(phi2_corrected / denom)) if denom > 0 else 0.0


def _correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    valid = categories.notna() & measurements.notna()
    if valid.sum() < 3:
        return 0.0
    cats = categories[valid].astype(str)
    values = measurements[valid].astype(float)
    grand_mean = float(values.mean())
    denominator = float(((values - grand_mean) ** 2).sum())
    if denominator <= 0:
        return 0.0
    numerator = 0.0
    for level in cats.unique():
        group = values[cats == level]
        numerator += float(len(group)) * (float(group.mean()) - grand_mean) ** 2
    return float(np.sqrt(max(numerator, 0.0) / denominator))


def target_associations(
    df: pd.DataFrame,
    inventory: pd.DataFrame,
    target: str | None,
    limit: int = 20,
) -> pd.DataFrame:
    target_name = _valid_target(df, target)
    if target_name is None or df[target_name].nunique(dropna=True) < 2:
        return pd.DataFrame(columns=["variable", "association", "method"])
    y = df[target_name]
    rows = []
    type_by_name = dict(zip(inventory["variable"], inventory["semantic_type"]))
    role_by_name = dict(zip(inventory["variable"], inventory["role"]))
    for name in map(str, df.columns):
        if name == target_name or role_by_name.get(name) in {"ID", "Excluded"}:
            continue
        inferred = type_by_name.get(name, semantic_type(df[name]))
        if inferred in {"constant", "all missing", "identifier-like"}:
            continue
        if inferred in {"continuous numeric", "discrete numeric", "numeric text"}:
            x = pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
            value = _correlation_ratio(y, x)
            method = "Correlation ratio"
        else:
            value = _cramers_v(df[name], y)
            method = "Cramér's V"
        if np.isfinite(value):
            rows.append({"variable": name, "association": float(value), "method": method})
    return pd.DataFrame(rows).sort_values("association", ascending=False).head(limit).reset_index(drop=True)


def predictor_exclusions(
    df: pd.DataFrame, roles: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Describe source columns omitted before model training."""
    role_map = _normalise_roles(roles)
    target = role_map.get("target")
    id_column = role_map.get("id_column")
    user_excluded = set(role_map["excluded_columns"])
    sensitive = set(role_map["sensitive_columns"])
    omitted: list[dict[str, Any]] = []
    for raw_name in df.columns:
        name = str(raw_name)
        if name == target:
            reason = "Target"
        elif name == id_column:
            reason = "ID"
        elif name in user_excluded:
            reason = "User excluded"
        else:
            continue
        omitted.append(
            {
                "variable": name,
                "reason": reason,
                "sensitive": name in sensitive,
            }
        )
    return omitted


def model_input_columns(df: pd.DataFrame, roles: dict[str, Any] | None) -> list[str]:
    omitted = {item["variable"] for item in predictor_exclusions(df, roles)}
    return [str(column) for column in df.columns if str(column) not in omitted]

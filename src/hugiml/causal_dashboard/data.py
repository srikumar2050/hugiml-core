"""Demo data and upload helpers for the causal dashboard."""

from __future__ import annotations

import base64
import binascii
import io
import json
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

_MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def binary_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric or Boolean columns whose observed values are 0 and 1."""
    out: list[str] = []
    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue
        if pd.api.types.is_bool_dtype(series):
            values = set(series.astype(int).unique().tolist())
        elif pd.api.types.is_numeric_dtype(series):
            values = set(
                pd.to_numeric(series, errors="coerce").dropna().astype(float).unique().tolist()
            )
        else:
            continue
        if values == {0, 1} or values == {0.0, 1.0}:
            out.append(str(column))
    return out


def dataset_roles(
    df: pd.DataFrame,
    metadata: dict[str, Any] | None,
    *,
    metadata_backed: bool,
) -> dict[str, Any]:
    """Resolve visible columns and causal role controls for a dataset."""
    meta = metadata or {}
    hidden = set(meta.get("hidden_columns", []))
    visible = [str(column) for column in df.columns if str(column) not in hidden]
    binaries = [column for column in binary_columns(df) if column in visible]

    if metadata_backed:
        treatment = str(meta["treatment"])
        outcome = str(meta["outcome"])
        treatment_options = [treatment]
        outcome_options = [outcome]
    else:
        treatment = binaries[0] if binaries else None
        outcome = next((column for column in binaries if column != treatment), None)
        treatment_options = list(binaries)
        outcome_options = list(binaries)

    adjustment = [
        column
        for column in meta.get("adjustment_set", [])
        if column in visible and column not in {treatment, outcome}
    ]
    if not adjustment:
        adjustment = [column for column in visible if column not in {treatment, outcome}][:12]
    return {
        "columns": visible,
        "binary": binaries,
        "treatment": treatment,
        "outcome": outcome,
        "treatment_options": treatment_options,
        "outcome_options": outcome_options,
        "adjustment_options": [column for column in visible if column not in {treatment, outcome}],
        "adjustment": adjustment,
        "metadata_backed": metadata_backed,
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25, 25)))


def _credit(seed: int = 2026, n: int = 3200) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    score = np.clip(r.normal(665, 82, n), 420, 850)
    dti = np.clip(r.beta(2.3, 4.8, n), 0.03, 0.82)
    util = np.clip(r.beta(2.7, 3.0, n), 0.02, 0.99)
    delin = r.poisson(0.55, n)
    income = np.clip(r.lognormal(np.log(65000), 0.48, n), 15000, 250000)
    age = np.clip(r.normal(42, 12, n), 21, 78).round().astype(int)
    region = r.choice(["North", "South", "East", "West"], n, p=[0.25, 0.29, 0.24, 0.22])
    homeowner = r.choice(["rent", "mortgage", "own"], n, p=[0.42, 0.44, 0.14])
    lp_t = (
        -0.15
        + 1.1 * (score < 620)
        + 0.85 * (dti > 0.38)
        + 0.65 * (delin >= 2)
        + 0.55 * (util > 0.72)
        - 0.35 * (age > 55)
    )
    e = _sigmoid(lp_t)
    t = r.binomial(1, e)
    base = (
        -2.05
        + 1.2 * (score < 620)
        + 0.9 * (dti > 0.4)
        + 0.95 * (util > 0.75)
        + 0.8 * (delin >= 2)
        - 0.38 * (income > 90000)
        + 0.25 * (homeowner == "rent")
    )
    tau_logit = -0.95 * ((util > 0.72) & (score < 640) & (delin >= 1)) + 0.42 * (
        (age < 30) & (dti > 0.45) & (income < 55000)
    )
    p0 = _sigmoid(base)
    p1 = _sigmoid(base + tau_logit)
    y = r.binomial(1, np.where(t == 1, p1, p0))
    return pd.DataFrame(
        {
            "credit_score": score.round().astype(int),
            "dti": dti.round(3),
            "credit_util": util.round(3),
            "prior_delinquency": delin.astype(int),
            "annual_income": income.round().astype(int),
            "age": age,
            "region": region,
            "home_ownership": homeowner,
            "enhanced_review": t.astype(int),
            "default_12m": y.astype(int),
            "_oracle_p0": p0,
            "_oracle_p1": p1,
            "_oracle_tau": p1 - p0,
        }
    )


def _claims(seed: int = 2027, n: int = 7000) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    a = r.normal(size=(n, 8))
    amount = np.clip(np.exp(np.log(6000) + 0.62 * a[:, 0]), 500, 45000)
    ps = 1.0 / (1.0 + np.exp(-a[:, 1]))
    complexity = np.where(a[:, 2] > 0, "complex", np.where(a[:, 2] > -0.8, "moderate", "routine"))
    prior = np.clip(np.round(2.2 + 1.35 * a[:, 3]), 0, 7).astype(int)
    age = np.clip(np.round(48 + 14 * a[:, 4]), 18, 90).astype(int)
    region = np.where(a[:, 5] > 0.55, "Rural", np.where(a[:, 5] < -0.45, "Urban", "Suburban"))
    days = np.clip(np.round(8 + 5 * a[:, 6]), 0, 35).astype(int)
    t = r.binomial(1, 0.5, n)
    base = -1.5 + 0.65 * a[:, 4] - 0.4 * a[:, 5] + 0.25 * (a[:, 6] > 0)
    tau_logit = (
        0.05
        + 1.6 * ((a[:, 0] > 0) & (a[:, 1] > 0) & (a[:, 2] > 0))
        - 0.9 * ((a[:, 0] < -0.7) & (a[:, 3] > 0) & (a[:, 4] > 0))
    )
    p0 = _sigmoid(base)
    p1 = _sigmoid(base + tau_logit)
    y = r.binomial(1, np.where(t == 1, p1, p0))
    return pd.DataFrame(
        {
            "claim_amount": amount.round(2),
            "provider_score": ps.round(3),
            "prior_claims": prior,
            "complexity": complexity,
            "region": region,
            "claimant_age": age,
            "days_to_file": days,
            "proactive_outreach": t.astype(int),
            "escalation_30d": y.astype(int),
            "_oracle_p0": p0,
            "_oracle_p1": p1,
            "_oracle_tau": p1 - p0,
        }
    )


def _collections(seed: int = 2028, n: int = 3600) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    bal = np.clip(r.lognormal(np.log(4200), 0.8, n), 100, 60000)
    dpd = np.clip(r.gamma(2.1, 18, n), 1, 150).round().astype(int)
    contacts = r.poisson(2.0, n)
    band = pd.cut(
        dpd + 18 * np.log1p(bal / 5000), [-1, 35, 60, 90, 999], labels=["A", "B", "C", "D"]
    ).astype(str)
    tenure = r.integers(3, 121, n)
    age = np.clip(r.normal(43, 13, n), 19, 82).round().astype(int)
    lp_t = (
        -1.0 + 1.2 * (dpd > 45) + 2.6 * (band == "D") + 1.1 * (bal > 8000) + 0.6 * (contacts >= 3)
    )
    e = _sigmoid(lp_t)
    t = r.binomial(1, e)
    base = (
        0.35 - 0.9 * (dpd > 60) - 0.55 * (bal > 7000) - 0.5 * (band == "D") + 0.35 * (tenure > 36)
    )
    tau_logit = 0.85 * ((dpd > 45) & (bal < 9000) & (contacts <= 2)) - 0.6 * (
        (band == "D") & (bal > 12000)
    )
    p0 = _sigmoid(base)
    p1 = _sigmoid(base + tau_logit)
    y = r.binomial(1, np.where(t == 1, p1, p0))
    return pd.DataFrame(
        {
            "balance": bal.round(2),
            "days_past_due": dpd,
            "prior_contacts": contacts.astype(int),
            "risk_band": band,
            "tenure_months": tenure,
            "age": age,
            "intensive_contact": t.astype(int),
            "cure_60d": y.astype(int),
            "_oracle_p0": p0,
            "_oracle_p1": p1,
            "_oracle_tau": p1 - p0,
        }
    )


DEMO_GENERATORS = {
    "credit_intervention": _credit,
    "claims_outreach": _claims,
    "collections_overlap": _collections,
}


def load_demo_metadata(key: str) -> dict[str, Any]:
    path = files("hugiml.causal_dashboard").joinpath("assets", "demos", f"{key}.json")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def demo_catalog() -> dict[str, dict[str, Any]]:
    return {key: load_demo_metadata(key) for key in DEMO_GENERATORS}


def load_demo(key: str = "credit_intervention") -> tuple[pd.DataFrame, dict[str, Any]]:
    if key not in DEMO_GENERATORS:
        raise KeyError(f"Unknown causal demo: {key}")
    return DEMO_GENERATORS[key](), load_demo_metadata(key)


def load_metadata_schema() -> dict[str, Any]:
    path = files("hugiml.causal_dashboard").joinpath("assets", "causal_metadata.schema.json")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_metadata_template() -> dict[str, Any]:
    path = files("hugiml.causal_dashboard").joinpath("assets", "causal_metadata.template.json")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _decode_upload(content: str) -> bytes:
    if not isinstance(content, str) or not content.startswith("data:") or ";base64," not in content:
        raise ValueError("Upload payload must be a base64 data URI.")
    encoded = content.split(",", 1)[1].strip()
    if not encoded:
        raise ValueError("Upload payload is empty.")
    estimated = (len(encoded) * 3) // 4
    if estimated > _MAX_UPLOAD_BYTES + 3:
        raise ValueError("Upload exceeds the 100 MB limit.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Upload payload is not valid base64.") from exc
    if not data:
        raise ValueError("Upload payload is empty.")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise ValueError("Upload exceeds the 100 MB limit.")
    return data


def _check_columns(columns: list[Any]) -> None:
    names = [str(c).strip() for c in columns]
    if not names or any(not n for n in names):
        raise ValueError("Dataset must have non-empty column names.")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError("Dataset contains duplicate column names: " + ", ".join(duplicates))


def _validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Uploaded dataset is empty.")
    _check_columns(list(df.columns))
    if len(df.columns) < 2:
        raise ValueError("Dataset must contain at least two columns.")
    return df


def _read_text_frame(data: bytes, sep: str) -> pd.DataFrame:
    header = pd.read_csv(io.BytesIO(data), sep=sep, header=None, nrows=1, dtype=str)
    if header.empty:
        raise ValueError("Uploaded dataset is empty.")
    _check_columns(header.iloc[0].tolist())
    return pd.read_csv(io.BytesIO(data), sep=sep)


def _read_excel_frame(data: bytes) -> pd.DataFrame:
    header = pd.read_excel(io.BytesIO(data), sheet_name=0, header=None, nrows=1, dtype=str)
    if header.empty:
        raise ValueError("Uploaded dataset is empty.")
    _check_columns(header.iloc[0].tolist())
    return pd.read_excel(io.BytesIO(data), sheet_name=0)


def read_data_upload(content: str, filename: str) -> pd.DataFrame:
    data = _decode_upload(content)
    name = (filename or "").lower().strip()
    if name.endswith(".csv"):
        frame = _read_text_frame(data, ",")
    elif name.endswith(".tsv"):
        frame = _read_text_frame(data, "\t")
    elif name.endswith((".parquet", ".pq")):
        frame = pd.read_parquet(io.BytesIO(data))
    elif name.endswith((".xlsx", ".xls")):
        frame = _read_excel_frame(data)
    else:
        raise ValueError("Supported data uploads: CSV, TSV, Parquet, XLSX.")
    return _validate_frame(frame)


def read_metadata_upload(content: str, filename: str = "causal_metadata.json") -> dict[str, Any]:
    if not (filename or "").lower().endswith(".json"):
        raise ValueError("Causal metadata must be a JSON file.")
    data = _decode_upload(content)
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Causal metadata is not valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError("Causal metadata must be a JSON object.")
    return value

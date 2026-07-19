"""Data helpers: demo generators, upload parsing, cache keys (no Streamlit dependency)."""

from __future__ import annotations

import base64
import hashlib
import io
import json

import numpy as np
import pandas as pd

from hugiml.dashboard.data import prepare_model_frame

DEMO_DATASETS = {
    "credit": {
        "label": "Credit risk",
        "target": "default_risk",
        "id_column": "customer_id",
        "excluded_columns": ["date_issued"],
        "sensitive_columns": ["age", "gender", "region"],
    },
    "churn": {
        "label": "Customer churn",
        "target": "churned",
        "id_column": "account_id",
        "excluded_columns": ["signup_date"],
        "sensitive_columns": ["region", "age_band", "service_tier"],
    },
    "claims": {
        "label": "Claims review",
        "target": "manual_review",
        "id_column": "claim_id",
        "excluded_columns": ["received_date"],
        "sensitive_columns": ["region", "provider_type", "claimant_age"],
    },
}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _make_credit(seed=2026, n=800):
    r = np.random.default_rng(seed)
    cs = np.clip(r.normal(670, 85, n), 420, 850)
    dti = np.clip(r.beta(2.4, 5.2, n), 0.02, 0.78)
    inc = np.clip(r.lognormal(np.log(62000), 0.45, n), 18000, 210000)
    la = np.clip(r.normal(15500, 6500, n), 1000, 42000)
    d2 = r.poisson(0.45, n)
    cu = np.clip(r.beta(2.8, 3.3, n), 0.02, 0.98)
    t60 = r.binomial(1, 0.38, n)
    ho = r.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.42, 0.44, 0.14])
    gen = r.choice(["F", "M", "X"], n, p=[0.49, 0.49, 0.02])
    reg = r.choice(["R1", "R2", "R3", "R4"], n, p=[0.3, 0.25, 0.25, 0.2])
    age = np.clip(r.normal(41, 12, n).round(), 21, 74).astype(int)
    s = (
        -2
        + 2.8 * (cs < 620)
        + 2 * (dti > 0.35)
        + 0.75 * (la > 22000)
        + 0.7 * t60
        + 0.65 * (d2 >= 2)
        + 0.7 * (cu > 0.7)
        + 0.45 * (ho == "RENT")
        - 0.55 * (inc > 85000)
    )
    y = r.binomial(1, _sig(s))
    raw = pd.DataFrame(
        {
            "customer_id": [f"C-{i:05d}" for i in range(1, n + 1)],
            "date_issued": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "credit_score": cs.round().astype(int),
            "dti": dti.round(3),
            "annual_inc": inc.round().astype(int),
            "loan_amnt": la.round().astype(int),
            "delinq_2yr": d2.astype(int),
            "credit_util": cu.round(3),
            "loan_term": np.where(t60 == 1, "60", "36"),
            "home_ownership": ho,
            "age": age,
            "gender": gen,
            "region": reg,
            "default_risk": y.astype(int),
        }
    )
    return raw, y.astype(int)


def _make_churn(seed=2027, n=900):
    r = np.random.default_rng(seed)
    ten = r.integers(1, 72, n)
    tix = r.poisson(1.2, n)
    ch = np.clip(r.normal(72, 24, n), 20, 150)
    ud = np.clip(r.normal(12, 18, n), 0, 80)
    lp = r.binomial(1, 0.2, n)
    ct = r.choice(["month-to-month", "annual", "two-year"], n, p=[0.55, 0.32, 0.13])
    st = r.choice(["basic", "plus", "premium"], n, p=[0.42, 0.38, 0.2])
    na = r.binomial(1, 0.48, n)
    reg = r.choice(["North", "South", "East", "West"], n)
    ab = r.choice(["18-29", "30-44", "45-64", "65+"], n, p=[0.22, 0.38, 0.3, 0.1])
    s = (
        -2.25
        + 1.15 * (ten < 6)
        + 0.55 * (tix >= 3)
        + 0.8 * (ch > 90)
        + 1 * (ud > 30)
        + 0.7 * lp
        + 0.95 * (ct == "month-to-month")
        + 0.5 * ((st == "basic") & (na == 1))
    )
    y = r.binomial(1, _sig(s))
    raw = pd.DataFrame(
        {
            "account_id": [f"A-{i:05d}" for i in range(1, n + 1)],
            "signup_date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "tenure_months": ten,
            "support_tickets": tix,
            "monthly_charge": ch.round(2),
            "usage_drop_pct": ud.round(1),
            "late_payment": np.where(lp == 1, "yes", "no"),
            "contract_type": ct,
            "service_tier": st,
            "no_addon": np.where(na == 1, "true", "false"),
            "region": reg,
            "age_band": ab,
            "churned": y.astype(int),
        }
    )
    return raw, y.astype(int)


def _make_claims(seed=2028, n=720):
    r = np.random.default_rng(seed)
    amt = np.clip(r.lognormal(np.log(5200), 0.65, n), 250, 35000)
    pc = r.poisson(1.1, n)
    ps = np.clip(r.beta(4, 2.2, n), 0.05, 0.98)
    df_ = r.integers(0, 45, n)
    ma = np.clip(r.normal(0.08, 0.1, n), 0, 0.55)
    dg = r.choice(["routine", "complex", "injury", "chronic"], n, p=[0.42, 0.2, 0.22, 0.16])
    pt = r.choice(["hospital", "clinic", "specialist", "new_provider"], n, p=[0.36, 0.34, 0.2, 0.1])
    reg = r.choice(["Urban", "Suburban", "Rural"], n, p=[0.48, 0.34, 0.18])
    age = np.clip(r.normal(48, 17, n).round(), 1, 92).astype(int)
    s = (
        -2.1
        + 1.45 * (amt > 12000)
        + 0.75 * (pc >= 3)
        + 1.25 * (ps < 0.4)
        + 0.7 * (df_ < 2)
        + 0.8 * ((dg == "complex") & (amt > 8000))
        + 0.65 * (pt == "new_provider")
        + 0.8 * (ma > 0.2)
    )
    y = r.binomial(1, _sig(s))
    raw = pd.DataFrame(
        {
            "claim_id": [f"CL-{i:05d}" for i in range(1, n + 1)],
            "received_date": pd.date_range("2025-06-01", periods=n, freq="D").astype(str),
            "claim_amount": amt.round(2),
            "prior_claims": pc,
            "provider_score": ps.round(3),
            "diagnosis_group": dg,
            "days_to_file": df_,
            "manual_adjustment": ma.round(3),
            "provider_type": pt,
            "region": reg,
            "claimant_age": age,
            "manual_review": y.astype(int),
        }
    )
    return raw, y.astype(int)


_MAKERS = {"credit": _make_credit, "churn": _make_churn, "claims": _make_claims}


def load_demo(key="credit"):
    spec = DEMO_DATASETS.get(key, DEMO_DATASETS["credit"])
    maker = _MAKERS.get(key, _make_credit)
    raw, _ = maker()
    roles = dict(
        target=spec["target"],
        id_column=spec["id_column"],
        excluded_columns=list(spec["excluded_columns"]),
        sensitive_columns=list(spec["sensitive_columns"]),
    )
    X, y, cids, meta = prepare_model_frame(
        raw,
        target=roles["target"],
        id_column=roles["id_column"],
        excluded_columns=roles["excluded_columns"],
    )
    return {
        "mode": f"Demo: {spec['label']}",
        "X": X,
        "y": np.asarray(y, dtype=int),
        "case_ids": cids,
        "roles": roles,
        "meta": meta,
    }


def read_upload(content, filename):
    _, b64 = content.split(",", 1)
    data = base64.b64decode(b64)
    nm = filename.lower()
    if nm.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    if nm.endswith(".tsv"):
        return pd.read_csv(io.BytesIO(data), sep="\t")
    if nm.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data), sheet_name=0)
    if nm.endswith((".parquet", ".pq")):
        return pd.read_parquet(io.BytesIO(data))
    raise ValueError(f"Unsupported: {filename}")


def fingerprint(df):
    p = {
        "shape": df.shape,
        "columns": list(map(str, df.columns)),
        "dtypes": [str(t) for t in df.dtypes],
    }
    try:
        p["sd"] = hashlib.sha256(
            pd.util.hash_pandas_object(df.head(1000), index=True).values.tobytes()
        ).hexdigest()
    except Exception:
        p["sd"] = str(df.head(20).to_json(default_handler=str))
    return hashlib.sha256(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()


def cache_key(src, cv, rs, ufp=None, rcfg=None):
    return hashlib.sha256(
        json.dumps(
            dict(source=src, cv=int(cv), random_state=int(rs), ufp=ufp, rc=rcfg or {}),
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()

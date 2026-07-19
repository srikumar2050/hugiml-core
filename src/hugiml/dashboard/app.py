"""Streamlit app for HUGIML Governance Studio.

Run directly:
    python -m streamlit run src/hugiml/dashboard/app.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Compatibility for older dependencies under NumPy 2.x.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
import pandas as pd

if __package__ is None or __package__ == "":
    _SRC_ROOT = Path(__file__).resolve().parents[2]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

import streamlit as st

from hugiml.dashboard.components.complexity import render_complexity
from hugiml.dashboard.components.config_compare import render_config_comparison
from hugiml.dashboard.components.drift import render_drift
from hugiml.dashboard.components.fairness import render_fairness
from hugiml.dashboard.components.feature_family import render_feature_family_audit
from hugiml.dashboard.components.governance_evidence import (
    render_adaptive_binning_evidence,
    render_augmented_pair_traceability,
    render_rpte_rule_evidence,
    render_survivor_led_pattern_audit,
)
from hugiml.dashboard.components.missingness import render_missingness
from hugiml.dashboard.components.overview import render_overview
from hugiml.dashboard.components.patterns import render_patterns
from hugiml.dashboard.components.performance import render_performance
from hugiml.dashboard.components.prediction import render_prediction
from hugiml.dashboard.components.pruning import render_pruning_analysis
from hugiml.dashboard.components.rpte_governance import rpte_has_tree_representation, rpte_is_active
from hugiml.dashboard.data import prepare_model_frame
from hugiml.dashboard.display import dataframe_for_display
from hugiml.dashboard.runner import score_cases, train_hugiml

try:
    # Prefer the local workbench.py sitting alongside this file (user-provided).
    import importlib.util as _ilu
    _wb_path = Path(__file__).resolve().parent / "workbench.py"
    if _wb_path.exists():
        _wb_spec = _ilu.spec_from_file_location("workbench", _wb_path)
        _wb_mod = _ilu.module_from_spec(_wb_spec)  # type: ignore[arg-type]
        _wb_spec.loader.exec_module(_wb_mod)  # type: ignore[union-attr]
        render_workbench = _wb_mod.render_workbench
    else:
        from hugiml.dashboard.workbench import render_workbench  # type: ignore[assignment]
except Exception:
    from hugiml.dashboard.workbench import render_workbench  # type: ignore[assignment]

SECTION_LABELS = [
    "Overview",
    "Validation",
    "Representation Audit",
    "Pattern Inventory",
    "Case Review",
    "Data Quality & Policy",
    "Configuration Comparison",
    "Representation Pruning",
    "Monitoring",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=2026)
    args, _ = parser.parse_known_args()
    return args


def _theme_tokens(theme: str) -> dict[str, str]:
    if theme == "Forest":
        return {
            "accent": "#10b981",
            "accent_2": "#84cc16",
            "hero_a": "rgba(16, 185, 129, 0.20)",
            "hero_b": "rgba(132, 204, 22, 0.14)",
            "chip_bg": "rgba(16, 185, 129, 0.10)",
            "border": "rgba(16, 185, 129, 0.30)",
            "shadow": "0 8px 24px rgba(5, 122, 85, 0.08)",
            "dark": False,
        }
    if theme == "Dark":
        # Dark-accent theme: custom dashboard surfaces become dark, but native
        # Streamlit widgets/tables/selectboxes are not globally overridden.
        return {
            "accent": "#60a5fa",
            "accent_2": "#a78bfa",
            "hero_a": "rgba(15, 23, 42, 0.94)",
            "hero_b": "rgba(30, 41, 59, 0.86)",
            "chip_bg": "rgba(30, 41, 59, 0.90)",
            "border": "rgba(96, 165, 250, 0.44)",
            "shadow": "0 12px 32px rgba(0, 0, 0, 0.22)",
            "dark": True,
        }
    return {
        "accent": "#2563eb",
        "accent_2": "#06b6d4",
        "hero_a": "rgba(37, 99, 235, 0.20)",
        "hero_b": "rgba(6, 182, 212, 0.14)",
        "chip_bg": "rgba(37, 99, 235, 0.10)",
        "border": "rgba(37, 99, 235, 0.30)",
        "shadow": "0 8px 24px rgba(14, 116, 144, 0.08)",
        "dark": False,
    }


def _apply_css(theme: str) -> None:
    """Apply dashboard CSS. Hides deploy chrome only; does not override Streamlit widget colors."""
    t = _theme_tokens(theme)

    dark_extra = ""
    if t["dark"]:
        dark_extra = """
        .hugiml-hero, .hugiml-workbench-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f172a 100%) !important;
            border-color: rgba(96,165,250,0.25) !important;
        }
        .hugiml-hero::after {
            background: linear-gradient(90deg, #60a5fa, #a78bfa, #60a5fa) !important;
        }
        .hugiml-hero h1, .hugiml-hero p { color: #f1f5f9 !important; }
        .hugiml-eyebrow { color: #93c5fd !important; }
        .hugiml-chip {
            background: rgba(15,23,42,0.85) !important;
            border-color: rgba(96,165,250,0.30) !important;
            color: #cbd5e1 !important;
        }
        .hugiml-section-note {
            background: rgba(15,23,42,0.55) !important;
            border-left-color: #60a5fa !important;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(160deg, rgba(30,41,59,0.92), rgba(15,23,42,0.88)) !important;
            border-top-color: #60a5fa !important;
        }
        div[data-testid="stMetric"] * { color: #f1f5f9 !important; }
        .hugiml-fit-complete {
            background: rgba(15,23,42,0.80) !important;
            border-color: rgba(96,165,250,0.28) !important;
            color: #cbd5e1 !important;
        }
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

        /* ── Chrome ───────────────────────────────────── */
        #MainMenu, div[data-testid="stDeployButton"],
        div[data-testid="stStatusWidget"], .stDeployButton {{ display: none !important; }}

        /* ── Base ─────────────────────────────────────── */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}
        .block-container {{
            padding-top: 0.75rem;
            padding-bottom: 3rem;
            max-width: 1540px;
        }}

        /* ── Hero ─────────────────────────────────────── */
        .hugiml-hero {{
            position: relative;
            background:
                radial-gradient(ellipse 55% 45% at 0% 0%, {t["hero_a"]}, transparent),
                radial-gradient(ellipse 38% 30% at 102% 0%, {t["hero_b"]}, transparent),
                var(--secondary-background-color);
            border: 1px solid {t["border"]};
            border-radius: 14px;
            padding: 20px 28px 18px;
            margin-bottom: 14px;
            box-shadow: {t["shadow"]};
            overflow: hidden;
        }}
        .hugiml-hero::after {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, {t["accent"]}, {t["accent_2"]});
            border-radius: 14px 14px 0 0;
        }}
        .hugiml-hero::before {{
            content: "";
            position: absolute;
            top: -55px; right: -55px;
            width: 190px; height: 190px;
            border-radius: 50%;
            background: radial-gradient(circle, {t["hero_b"]}, transparent 68%);
            pointer-events: none;
        }}
        .hugiml-eyebrow {{
            display: inline-block;
            font-size: 0.64rem;
            font-weight: 700;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: {t["accent"]};
            margin-bottom: 8px;
        }}
        .hugiml-hero h1 {{
            font-family: 'Inter', sans-serif !important;
            margin: 0 0 10px 0;
            font-size: clamp(1.55rem, 2.3vw, 2.15rem);
            font-weight: 900;
            letter-spacing: -0.045em;
            line-height: 1.04;
        }}
        .hugiml-hero p {{
            margin: 0;
            opacity: 0.72;
            line-height: 1.62;
            font-size: 0.91rem;
            max-width: 820px;
        }}

        /* ── Chips ────────────────────────────────────── */
        .hugiml-chip-row {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 18px; }}
        .hugiml-chip {{
            display: inline-flex;
            align-items: center;
            background: {t["chip_bg"]};
            border: 1px solid {t["border"]};
            border-radius: 4px;
            padding: 3px 8px;
            font-size: 0.67rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
        }}

        /* ── Metric cards ─────────────────────────────── */
        div[data-testid="stMetric"] {{
            background: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128,0.14);
            border-top: 3px solid {t["accent"]};
            border-radius: 10px;
            padding: 16px 18px 14px;
            min-height: 90px;
        }}
        div[data-testid="stMetricValue"] {{
            font-family: 'Inter', sans-serif !important;
            font-size: clamp(1.2rem, 1.65vw, 1.80rem) !important;
            font-weight: 900 !important;
            letter-spacing: -0.03em !important;
            line-height: 1.07 !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
        }}
        div[data-testid="stMetricLabel"] > div {{
            font-size: 0.68rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.11em !important;
            text-transform: uppercase !important;
            opacity: 0.52 !important;
        }}

        /* ── Section callout ──────────────────────────── */
        .hugiml-section-note {{
            border: 1px solid rgba(128,128,128,0.12);
            border-left: 3px solid {t["accent"]};
            background: {t["chip_bg"]};
            border-radius: 0 8px 8px 0;
            padding: 10px 16px;
            margin: 4px 0 18px 0;
        }}
        .hugiml-section-note p {{
            margin: 0;
            font-size: 0.87rem;
            line-height: 1.54;
            opacity: 0.80;
        }}

        /* ── Tabs ─────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 1px;
            border-bottom: 1px solid rgba(128,128,128,0.13);
        }}
        .stTabs [data-baseweb="tab"] {{
            font-family: 'Inter', sans-serif !important;
            font-size: 0.81rem !important;
            font-weight: 600 !important;
            padding: 8px 16px !important;
            border-radius: 7px 7px 0 0;
            border-bottom: 2px solid transparent;
        }}
        .stTabs [aria-selected="true"] {{
            border-bottom-color: {t["accent"]} !important;
            color: {t["accent"]} !important;
        }}

        /* ── DataFrames ───────────────────────────────── */
        div[data-testid="stDataFrame"] {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(128,128,128,0.11);
        }}

        /* ── Sidebar ──────────────────────────────────── */
        div[data-testid="stSidebarContent"] {{
            padding-top: 0.75rem !important;
        }}
        /* Brand strip */
        .hugiml-brand {{
            padding-bottom: 8px;
            margin-bottom: 6px;
            border-bottom: 1px solid rgba(128,128,128,0.15);
        }}
        .hugiml-brand-name {{
            font-size: 0.88rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }}
        .hugiml-brand-tag {{
            font-size: 0.59rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.42;
            margin-top: 2px;
        }}
        /* Workspace button pair — secondary buttons styled as boxed tabs */
        div[data-testid="stSidebarContent"] div[data-testid="stHorizontalBlock"] {{
            gap: 4px !important;
            margin-bottom: 0 !important;
        }}
        div[data-testid="stSidebarContent"] div[data-testid="stHorizontalBlock"] button[kind="secondary"] {{
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            padding: 4px 6px !important;
            border-radius: 6px !important;
            border: 1px solid rgba(128,128,128,0.22) !important;
            background: rgba(128,128,128,0.05) !important;
            color: inherit !important;
            transition: background 0.12s !important;
        }}
        div[data-testid="stSidebarContent"] div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {{
            background: rgba(128,128,128,0.12) !important;
        }}
        /* Section label */
        .hugiml-sidebar-section {{
            font-size: 0.62rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            opacity: 0.45;
            margin-top: 8px;
            margin-bottom: 2px;
        }}
        .hugiml-fit-complete {{
            background: {t["chip_bg"]};
            border: 1px solid {t["border"]};
            border-radius: 7px;
            padding: 8px 11px;
            margin-top: 8px;
            font-size: 0.80rem;
            line-height: 1.42;
        }}

        /* ── Buttons ──────────────────────────────────── */
        div[data-testid="stButton"] > button[kind="primary"] {{
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
        }}

        /* ── Form labels ──────────────────────────────── */
        div[data-testid="stMultiSelect"] label p,
        div[data-testid="stRadio"] label p,
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stNumberInput"] label p {{
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.83rem !important;
        }}

        /* ── Headings ─────────────────────────────────── */
        h3 {{ font-family: 'Inter', sans-serif !important; font-weight: 800 !important; letter-spacing: -0.025em !important; }}
        h4 {{ font-family: 'Inter', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.015em !important; }}

        /* ── Workbench hero variant ───────────────────── */
        .hugiml-workbench-hero .hugiml-eyebrow {{ color: {t["accent_2"]} !important; }}
        .hugiml-workbench-hero::after {{
            background: linear-gradient(90deg, {t["accent_2"]}, {t["accent"]}) !important;
        }}


        /* ── Workflow guide cards ─────────────────────── */
        .hugiml-workflow {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 10px;
            margin: 0 0 18px 0;
        }}
        .hugiml-step {{
            background: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128,0.13);
            border-radius: 10px;
            padding: 12px 14px;
        }}
        .hugiml-step-num {{
            display: inline-flex;
            width: 22px;
            height: 22px;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: {t["chip_bg"]};
            border: 1px solid {t["border"]};
            color: {t["accent"]};
            font-weight: 900;
            font-size: 0.72rem;
            margin-bottom: 7px;
        }}
        .hugiml-step b {{
            display: block;
            margin-bottom: 4px;
            font-size: 0.86rem;
        }}
        .hugiml-step span:not(.hugiml-step-num) {{
            display: block;
            font-size: 0.76rem;
            opacity: 0.66;
            line-height: 1.45;
        }}

        /* ── Misc ─────────────────────────────────────── */
        /* Don't override hr globally — specific margins set inline where needed */
        .hugiml-hero hr, .hugiml-section hr {{ border-color: rgba(128,128,128,0.11) !important; }}
        .hugiml-compact-note {{ font-size: 0.78rem; opacity: 0.62; margin-top: -0.3rem; }}

        {dark_extra}

        @media (max-width: 900px) {{
            .hugiml-hero {{ padding: 20px 18px 18px; }}
            .hugiml-hero h1 {{ font-size: 1.42rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _read_uploaded_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    if name.endswith(".tsv"):
        return pd.read_csv(io.BytesIO(data), sep="\t")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data), sheet_name=0)
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(io.BytesIO(data))

    raise ValueError("Unsupported upload type. Use CSV, TSV, XLSX, XLS, or Parquet.")


def _dataframe_fingerprint(df: pd.DataFrame) -> str:
    payload = {
        "shape": df.shape,
        "columns": list(map(str, df.columns)),
        "dtypes": [str(t) for t in df.dtypes],
    }
    try:
        sample_hash = pd.util.hash_pandas_object(df.head(1000), index=True).values.tobytes()
        sample_digest = hashlib.sha256(sample_hash).hexdigest()
    except Exception:
        sample_digest = str(df.head(20).to_json(default_handler=str))
    payload["sample_digest"] = sample_digest
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _config_key(
    source: str,
    cv: int,
    random_state: int,
    upload_fingerprint: str | None = None,
    role_config: dict[str, Any] | None = None,
) -> str:
    payload = {
        "source": source,
        "cv": int(cv),
        "random_state": int(random_state),
        "upload_fingerprint": upload_fingerprint,
        "role_config": role_config or {},
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


DEMO_DATASETS: dict[str, dict[str, Any]] = {
    "credit": {
        "label": "Credit risk",
        "mode": "Demo: credit risk",
        "target": "default_risk",
        "id_column": "customer_id",
        "excluded_columns": ["date_issued"],
        "sensitive_columns": ["age", "gender", "region"],
    },
    "churn": {
        "label": "Customer churn",
        "mode": "Demo: customer churn",
        "target": "churned",
        "id_column": "account_id",
        "excluded_columns": ["signup_date"],
        "sensitive_columns": ["region", "age_band", "service_tier"],
    },
    "claims": {
        "label": "Claims review",
        "mode": "Demo: claims review",
        "target": "manual_review",
        "id_column": "claim_id",
        "excluded_columns": ["received_date"],
        "sensitive_columns": ["region", "provider_type", "claimant_age"],
    },
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _make_credit_demo(seed: int = 2026, n: int = 800) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    credit_score = np.clip(rng.normal(670, 85, n), 420, 850)
    dti = np.clip(rng.beta(2.4, 5.2, n), 0.02, 0.78)
    annual_inc = np.clip(rng.lognormal(np.log(62000), 0.45, n), 18000, 210000)
    loan_amnt = np.clip(rng.normal(15500, 6500, n), 1000, 42000)
    delinq_2yr = rng.poisson(0.45, n)
    credit_util = np.clip(rng.beta(2.8, 3.3, n), 0.02, 0.98)
    term_60 = rng.binomial(1, 0.38, n)
    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.42, 0.44, 0.14])
    gender = rng.choice(["F", "M", "X"], n, p=[0.49, 0.49, 0.02])
    region = rng.choice(["R1", "R2", "R3", "R4"], n, p=[0.30, 0.25, 0.25, 0.20])
    age = np.clip(rng.normal(41, 12, n).round(), 21, 74).astype(int)
    score = (
        -2.0
        + 2.8 * (credit_score < 620)
        + 2.0 * (dti > 0.35)
        + 0.75 * (loan_amnt > 22000)
        + 0.70 * term_60
        + 0.65 * (delinq_2yr >= 2)
        + 0.70 * (credit_util > 0.70)
        + 0.45 * (home_ownership == "RENT")
        - 0.55 * (annual_inc > 85000)
    )
    y = rng.binomial(1, _sigmoid(score))
    raw = pd.DataFrame(
        {
            "customer_id": [f"C-{i:05d}" for i in range(1, n + 1)],
            "date_issued": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "credit_score": credit_score.round(0).astype(int),
            "dti": dti.round(3),
            "annual_inc": annual_inc.round(0).astype(int),
            "loan_amnt": loan_amnt.round(0).astype(int),
            "delinq_2yr": delinq_2yr.astype(int),
            "credit_util": credit_util.round(3),
            "loan_term": np.where(term_60 == 1, "60", "36"),
            "home_ownership": home_ownership,
            "age": age,
            "gender": gender,
            "region": region,
            "default_risk": y.astype(int),
        }
    )
    return raw, y.astype(int)


def _make_churn_demo(seed: int = 2027, n: int = 900) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    tenure_months = rng.integers(1, 72, n)
    support_tickets = rng.poisson(1.2, n)
    monthly_charge = np.clip(rng.normal(72, 24, n), 20, 150)
    usage_drop_pct = np.clip(rng.normal(12, 18, n), 0, 80)
    late_payment = rng.binomial(1, 0.20, n)
    contract_type = rng.choice(["month-to-month", "annual", "two-year"], n, p=[0.55, 0.32, 0.13])
    service_tier = rng.choice(["basic", "plus", "premium"], n, p=[0.42, 0.38, 0.20])
    no_addon = rng.binomial(1, 0.48, n)
    region = rng.choice(["North", "South", "East", "West"], n)
    age_band = rng.choice(["18-29", "30-44", "45-64", "65+"], n, p=[0.22, 0.38, 0.30, 0.10])
    score = (
        -2.25
        + 1.15 * (tenure_months < 6)
        + 0.55 * (support_tickets >= 3)
        + 0.80 * (monthly_charge > 90)
        + 1.00 * (usage_drop_pct > 30)
        + 0.70 * late_payment
        + 0.95 * (contract_type == "month-to-month")
        + 0.50 * ((service_tier == "basic") & (no_addon == 1))
    )
    y = rng.binomial(1, _sigmoid(score))
    raw = pd.DataFrame(
        {
            "account_id": [f"A-{i:05d}" for i in range(1, n + 1)],
            "signup_date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "tenure_months": tenure_months,
            "support_tickets": support_tickets,
            "monthly_charge": monthly_charge.round(2),
            "usage_drop_pct": usage_drop_pct.round(1),
            "late_payment": np.where(late_payment == 1, "yes", "no"),
            "contract_type": contract_type,
            "service_tier": service_tier,
            "no_addon": np.where(no_addon == 1, "true", "false"),
            "region": region,
            "age_band": age_band,
            "churned": y.astype(int),
        }
    )
    return raw, y.astype(int)


def _make_claims_demo(seed: int = 2028, n: int = 720) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    claim_amount = np.clip(rng.lognormal(np.log(5200), 0.65, n), 250, 35000)
    prior_claims = rng.poisson(1.1, n)
    provider_score = np.clip(rng.beta(4.0, 2.2, n), 0.05, 0.98)
    days_to_file = rng.integers(0, 45, n)
    manual_adjustment = np.clip(rng.normal(0.08, 0.10, n), 0, 0.55)
    diagnosis_group = rng.choice(["routine", "complex", "injury", "chronic"], n, p=[0.42, 0.20, 0.22, 0.16])
    provider_type = rng.choice(["hospital", "clinic", "specialist", "new_provider"], n, p=[0.36, 0.34, 0.20, 0.10])
    region = rng.choice(["Urban", "Suburban", "Rural"], n, p=[0.48, 0.34, 0.18])
    claimant_age = np.clip(rng.normal(48, 17, n).round(), 1, 92).astype(int)
    score = (
        -2.10
        + 1.45 * (claim_amount > 12000)
        + 0.75 * (prior_claims >= 3)
        + 1.25 * (provider_score < 0.40)
        + 0.70 * (days_to_file < 2)
        + 0.80 * ((diagnosis_group == "complex") & (claim_amount > 8000))
        + 0.65 * (provider_type == "new_provider")
        + 0.80 * (manual_adjustment > 0.20)
    )
    y = rng.binomial(1, _sigmoid(score))
    raw = pd.DataFrame(
        {
            "claim_id": [f"CL-{i:05d}" for i in range(1, n + 1)],
            "received_date": pd.date_range("2025-06-01", periods=n, freq="D").astype(str),
            "claim_amount": claim_amount.round(2),
            "prior_claims": prior_claims,
            "provider_score": provider_score.round(3),
            "diagnosis_group": diagnosis_group,
            "days_to_file": days_to_file,
            "manual_adjustment": manual_adjustment.round(3),
            "provider_type": provider_type,
            "region": region,
            "claimant_age": claimant_age,
            "manual_review": y.astype(int),
        }
    )
    return raw, y.astype(int)


def _demo_raw(dataset_key: str) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    key = dataset_key if dataset_key in DEMO_DATASETS else "credit"
    if key == "churn":
        raw, y = _make_churn_demo()
    elif key == "claims":
        raw, y = _make_claims_demo()
    else:
        raw, y = _make_credit_demo()
    return raw, y, DEMO_DATASETS[key]


@st.cache_data(show_spinner=False)
def _load_demo_context_cached(dataset_key: str = "credit") -> dict[str, Any]:
    raw, y, spec = _demo_raw(dataset_key)
    roles = {
        "target": spec["target"],
        "id_column": spec["id_column"],
        "excluded_columns": list(spec["excluded_columns"]),
        "sensitive_columns": list(spec["sensitive_columns"]),
    }
    X, y, case_ids, meta = prepare_model_frame(
        raw,
        target=roles["target"],
        id_column=roles["id_column"],
        excluded_columns=roles["excluded_columns"],
    )
    return {
        "mode": spec["mode"],
        "demo_dataset": spec["label"],
        "X": X,
        "y": np.asarray(y, dtype=int),
        "case_ids": case_ids,
        "roles": roles,
        "meta": meta,
    }


def _prepare_uploaded_context(
    raw: pd.DataFrame,
    target: str,
    id_column: str | None,
    excluded_columns: list[str],
    sensitive_columns: list[str],
    positive_label: str | None,
) -> dict[str, Any]:
    roles = {
        "target": target,
        "id_column": id_column,
        "excluded_columns": excluded_columns,
        "sensitive_columns": sensitive_columns,
    }
    X, y, case_ids, meta = prepare_model_frame(
        raw,
        target=target,
        id_column=id_column,
        excluded_columns=excluded_columns,
        positive_label=positive_label,
    )
    return {
        "mode": "Uploaded dataset",
        "X": X,
        "y": np.asarray(y, dtype=int),
        "case_ids": case_ids,
        "roles": roles,
        "meta": meta,
    }


@st.cache_resource(show_spinner=True)
def _train_model_cached(
    cache_key: str,
    _X: pd.DataFrame,
    y_tuple: tuple[int, ...],
    cv: int,
    random_state: int,
):
    y = np.asarray(y_tuple, dtype=int)
    return train_hugiml(_X, y, cv=cv, scoring="roc_auc", random_state=random_state)


def _persist_uploaded_file(uploaded_file) -> None:
    if uploaded_file is not None:
        st.session_state["hugiml_uploaded_file_name"] = uploaded_file.name
        st.session_state["hugiml_uploaded_file_bytes"] = uploaded_file.getvalue()


def _read_persisted_upload() -> tuple[pd.DataFrame, str] | None:
    if "hugiml_uploaded_file_bytes" not in st.session_state:
        return None

    class _StoredUpload:
        def __init__(self, name: str, data: bytes):
            self.name = name
            self._data = data

        def getvalue(self) -> bytes:
            return self._data

    stored = _StoredUpload(
        st.session_state.get("hugiml_uploaded_file_name", "uploaded.csv"),
        st.session_state["hugiml_uploaded_file_bytes"],
    )
    return _read_uploaded_table(stored), stored.name


def _reset_run_state_for_dataset_change(new_dataset_key: str) -> bool:
    """Clear stale run/governance state when the selected demo dataset changes.

    Returns ``True`` only when the dataset actually changed. The caller should
    immediately rerun after a reset so the already-computed top-level workspace
    state cannot continue rendering stale Governance/Results panes during the
    same Streamlit pass.
    """
    previous_dataset_key = st.session_state.get("hugiml_active_demo_dataset_key")
    st.session_state["hugiml_active_demo_dataset_key"] = new_dataset_key
    if previous_dataset_key is None or previous_dataset_key == new_dataset_key:
        return False

    keys_to_clear = [
        "hugiml_workbench_runs",
        "hugiml_workbench_context_key",
        "hugiml_workbench_last_message",
        "hugiml_promoted_governance_ctx",
        "hugiml_cv_monitoring_cache",
        "hugiml_cv_monitoring_cache_key",
        "hugiml_jump_to_results_tab",
        "hugiml_workbench_open_results",
        "hugiml_governance_requested",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)

    st.session_state["hugiml_nav_section"] = "Setup"
    st.session_state["hugiml_workbench_section"] = "Setup"
    st.session_state["hugiml_workbench_section_token"] = int(st.session_state.get("hugiml_workbench_section_token", 0)) + 1
    return True


def _pop_session_flag(key: str) -> bool:
    """Pop a navigation flag without evaluating arbitrary objects as bool.

    Streamlit session state can contain DataFrames and other array-like objects.
    Accidentally using one in an ``if`` condition raises Pandas' ambiguous truth
    value error. Only explicit scalar truthy values count as flags here.
    """
    value = st.session_state.pop(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _sidebar_dataset_controls(args: argparse.Namespace, *, require_fit_action: bool = True) -> tuple[dict[str, Any] | None, str | None, int, int, str]:
    theme = st.session_state.get("hugiml_theme", "Ocean")

    st.sidebar.markdown(
        '<hr style="margin:2px 0 4px 0;border:none;border-top:1px solid rgba(128,128,128,0.12)"/>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="hugiml-sidebar-section">Data &amp; Run Setup</div>', unsafe_allow_html=True)

    source = st.sidebar.radio(
        "Data source",
        ["Demo dataset", "Upload file"],
        index=0,
        key="hugiml_data_source",
    )

    with st.sidebar.expander("Run settings", expanded=True):
        cv = st.number_input("CV folds", min_value=2, max_value=10, value=int(args.cv), step=1)
        random_state = st.number_input("Random seed", min_value=0, max_value=999999, value=int(args.random_state), step=1)

    if source == "Demo dataset":
        demo_labels = [spec["label"] for spec in DEMO_DATASETS.values()]
        demo_keys = list(DEMO_DATASETS.keys())
        selected_label = st.sidebar.selectbox(
            "Demo dataset",
            demo_labels,
            index=0,
            help="Use the same demo scenarios shown in the static HTML preview.",
            key="hugiml_demo_dataset",
        )
        demo_key = demo_keys[demo_labels.index(selected_label)]
        if _reset_run_state_for_dataset_change(demo_key):
            st.rerun()
        return (
            _load_demo_context_cached(demo_key),
            _config_key(f"demo:{demo_key}", int(cv), int(random_state)),
            int(cv),
            int(random_state),
            theme,
        )

    st.sidebar.markdown("### Upload dataset")
    uploaded = st.sidebar.file_uploader(
        "Dataset file",
        type=["csv", "tsv", "xlsx", "xls", "parquet", "pq"],
        help="Upload a binary-classification table. Select target and column roles below.",
        key="hugiml_uploaded_file",
    )
    _persist_uploaded_file(uploaded)

    try:
        if uploaded is not None:
            raw = _read_uploaded_table(uploaded)
            upload_name = uploaded.name
        else:
            persisted = _read_persisted_upload()
            if persisted is None:
                st.markdown(
                    """
                    <div class="hugiml-hero">
                      <span class="hugiml-eyebrow">Upload Dataset</span>
                      <h1>Choose a file to begin</h1>
                      <p>Upload a CSV, TSV, Excel, or Parquet file, define your target column and review column roles, then fit the model.</p>
                      <div class="hugiml-chip-row">
                        <span class="hugiml-chip">CSV</span>
                        <span class="hugiml-chip">TSV</span>
                        <span class="hugiml-chip">Excel</span>
                        <span class="hugiml-chip">Parquet</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                return None, None, int(cv), int(random_state), theme
            raw, upload_name = persisted
    except Exception as exc:
        st.error(f"Could not read uploaded file: {exc}")
        return None, None, int(cv), int(random_state), theme

    columns = list(map(str, raw.columns))
    if not columns:
        st.error("Uploaded file has no columns.")
        return None, None, int(cv), int(random_state), theme

    st.sidebar.caption(f"Loaded {raw.shape[0]:,} rows × {raw.shape[1]:,} columns.")

    with st.sidebar.expander("Column roles", expanded=True):
        target_default_idx = next((i for i, c in enumerate(columns) if str(c).strip().lower() == "target"), 0)
        target = st.selectbox("Target column", columns, index=target_default_idx, key="hugiml_upload_target")

        id_options = ["<none>"] + columns
        id_selected = st.selectbox("ID column", id_options, index=0, key="hugiml_upload_id")
        id_column = None if id_selected == "<none>" else id_selected

        excluded_candidates = [c for c in columns if c not in {target, id_column}]
        excluded_columns = st.multiselect(
            "Exclude from modeling",
            excluded_candidates,
            default=[],
            key="hugiml_upload_excluded",
        )

        sensitive_candidates = [c for c in excluded_candidates if c not in set(excluded_columns)]
        sensitive_columns = st.multiselect(
            "Sensitive/proxy review",
            sensitive_candidates,
            default=[],
            key="hugiml_upload_sensitive",
        )

        positive_label = None
        try:
            unique_target = sorted([str(x) for x in raw[target].dropna().unique().tolist()])
            if 1 < len(unique_target) <= 10:
                positive_options = ["<auto>"] + unique_target
                selected_positive = st.selectbox("Positive label", positive_options, index=0)
                positive_label = None if selected_positive == "<auto>" else selected_positive
        except Exception:
            pass

    role_config = {
        "target": target,
        "id_column": id_column,
        "excluded_columns": excluded_columns,
        "sensitive_columns": sensitive_columns,
        "positive_label": positive_label,
    }
    role_key = hashlib.sha256(json.dumps(role_config, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    upload_key = _dataframe_fingerprint(raw)
    fit_key = f"{upload_key}:{role_key}:{int(cv)}:{int(random_state)}"

    if not require_fit_action:
        # Gate preparation on explicit role confirmation for the current upload.
        upload_fingerprint_now = _dataframe_fingerprint(raw)
        roles_confirmed_key = f"hugiml_roles_confirmed:{upload_fingerprint_now}"
        if not st.session_state.get(roles_confirmed_key, False):
            # Show a data preview until the user confirms column roles.
            st.markdown(
                f"""
                <div class="hugiml-hero">
                  <span class="hugiml-eyebrow">Dataset Uploaded</span>
                  <h1>Confirm column roles to continue</h1>
                  <p><b>{upload_name}</b> loaded. Select the target column and any other roles in the
                  sidebar, then click <b>Confirm roles</b> to proceed.</p>
                  <div class="hugiml-chip-row">
                    <span class="hugiml-chip">{raw.shape[0]:,} rows</span>
                    <span class="hugiml-chip">{raw.shape[1]:,} columns</span>
                    <span class="hugiml-chip">Target (auto): {target}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                st.markdown("#### Data preview")
                st.dataframe(dataframe_for_display(raw.head(30)), width="stretch", hide_index=True)
            if st.sidebar.button("Confirm roles and continue", type="primary", width="stretch"):
                st.session_state[roles_confirmed_key] = True
                st.rerun()
            return None, None, int(cv), int(random_state), theme

        try:
            data_ctx = _prepare_uploaded_context(
                raw,
                target=target,
                id_column=id_column,
                excluded_columns=excluded_columns,
                sensitive_columns=sensitive_columns,
                positive_label=positive_label,
            )
            return data_ctx, _config_key("upload", int(cv), int(random_state), upload_key, role_config), int(cv), int(random_state), theme
        except Exception as exc:
            st.error(f"Could not prepare uploaded dataset: {exc}")
            with st.expander("Dataset preview", expanded=True):
                st.dataframe(dataframe_for_display(raw.head(30)), width="stretch", hide_index=True)
            # Clear confirmation so roles can be adjusted before retrying.
            st.session_state[roles_confirmed_key] = False
            return None, None, int(cv), int(random_state), theme

    if st.session_state.get("hugiml_last_fit_key") != fit_key:
        st.session_state["hugiml_upload_fit_requested"] = False

    already_fit = bool(st.session_state.get("hugiml_upload_fit_requested", False)) and st.session_state.get("hugiml_last_fit_key") == fit_key

    fit_clicked = st.sidebar.button(
        "Model fitted" if already_fit else "Fit model and open evidence views",
        type="primary",
        width="stretch",
        disabled=already_fit,
        help=(
            "Model has already been fitted for this uploaded file and selected roles. "
            "Upload a different file or change target/roles/settings to re-enable."
            if already_fit
            else "Fit HUGIML and populate the governance evidence views."
        ),
    )
    if fit_clicked:
        st.session_state["hugiml_upload_fit_requested"] = True
        st.session_state["hugiml_last_fit_key"] = fit_key

    if already_fit:
        st.sidebar.markdown(
            '<div class="hugiml-fit-complete">Model fitted for the current upload and role configuration. '
            'Change file, target, roles, CV, or seed to fit again.</div>',
            unsafe_allow_html=True,
        )

    if not st.session_state.get("hugiml_upload_fit_requested", False):
        st.markdown(
            f"""
            <div class="hugiml-hero">
              <span class="hugiml-eyebrow">Dataset Preview</span>
              <h1>Review uploaded dataset</h1>
              <p><b>{upload_name}</b> loaded successfully. Confirm the target and column roles in the sidebar,
              then click <b>Fit model and open evidence views</b>.</p>
              <div class="hugiml-chip-row">
                <span class="hugiml-chip">{raw.shape[0]:,} rows</span>
                <span class="hugiml-chip">{raw.shape[1]:,} columns</span>
                <span class="hugiml-chip">Target: {target}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown("#### Data preview")
            st.dataframe(dataframe_for_display(raw.head(30)), width="stretch", hide_index=True)
        return None, None, int(cv), int(random_state), theme

    try:
        data_ctx = _prepare_uploaded_context(
            raw,
            target=target,
            id_column=id_column,
            excluded_columns=excluded_columns,
            sensitive_columns=sensitive_columns,
            positive_label=positive_label,
        )
        return data_ctx, _config_key("upload", int(cv), int(random_state), upload_key, role_config), int(cv), int(random_state), theme
    except Exception as exc:
        st.error(f"Could not prepare uploaded dataset: {exc}")
        with st.expander("Dataset preview", expanded=True):
            st.dataframe(dataframe_for_display(raw.head(30)), width="stretch", hide_index=True)
        return None, None, int(cv), int(random_state), theme


def _load_context(args: argparse.Namespace, workspace: str) -> tuple[dict[str, Any] | None, str]:
    has_promoted_context = isinstance(st.session_state.get("hugiml_promoted_governance_ctx"), dict)
    data_ctx, cache_key, cv, random_state, theme = _sidebar_dataset_controls(
        args,
        require_fit_action=(workspace == "Governance" and not has_promoted_context),
    )
    if data_ctx is None:
        return None, theme

    if workspace == "Workbench":
        return {
            **data_ctx,
            "cache_key": cache_key,
            "cv": int(cv),
            "random_state": int(random_state),
        }, theme

    promoted = st.session_state.get("hugiml_promoted_governance_ctx")
    if isinstance(promoted, dict):
        return promoted, theme

    result = _train_model_cached(
        cache_key or "default",
        data_ctx["X"],
        tuple(int(v) for v in data_ctx["y"]),
        int(cv),
        int(random_state),
    )
    model = result.best_estimator_

    try:
        predictions = score_cases(model, data_ctx["X"])
    except Exception:
        predictions = pd.DataFrame()

    return {
        **data_ctx,
        "cache_key": cache_key,
        "result": result,
        "model": model,
        "predictions": predictions,
        "cv": int(cv),
        "random_state": int(random_state),
    }, theme


def _render_hero(ctx: dict[str, Any]) -> None:
    best_score = getattr(ctx["result"], "best_score_", None)
    model = ctx["model"]
    score_text = f"{best_score:.4f}" if best_score is not None else "N/A"
    promoted_run = ctx.get("promoted_run") if isinstance(ctx.get("promoted_run"), dict) else {}
    promoted_chip = (
        f'<span class="hugiml-chip">Promoted run: {promoted_run.get("run_id")}</span>'
        if promoted_run.get("run_id")
        else ""
    )

    st.markdown(
        f"""
        <div class="hugiml-hero">
          <span class="hugiml-eyebrow">HUGIML Governance Studio</span>
          <h1>Audit &amp; Evidence Dashboard</h1>
          <p>Validation performance, model complexity, feature-family provenance, HUG pattern inventory,
          case explanations, data quality, representation-valid rebuilds, configuration comparison, and monitoring signals.</p>
          <div class="hugiml-chip-row">
            <span class="hugiml-chip">Data: {ctx['mode']}</span>
            <span class="hugiml-chip">Rows: {ctx['meta']['n_rows']:,}</span>
            <span class="hugiml-chip">Features: {ctx['meta']['n_features']:,}</span>
            <span class="hugiml-chip">Best CV ROC-AUC: {score_text}</span>
            {promoted_chip}
            <span class="hugiml-chip">Feature mode: {getattr(model, 'feature_mode', 'N/A')}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_governance_workflow() -> None:
    st.markdown(
        """
        <div class="hugiml-workflow">
          <div class="hugiml-step">
            <span class="hugiml-step-num">1</span>
            <b>Model overview</b>
            <span>Confirm roles, selected parameters, final score, and data shape.</span>
          </div>
          <div class="hugiml-step">
            <span class="hugiml-step-num">2</span>
            <b>Evidence inspection</b>
            <span>Open validation, representation, patterns, cases, and quality views.</span>
          </div>
          <div class="hugiml-step">
            <span class="hugiml-step-num">3</span>
            <b>Governance actions</b>
            <span>Review sensitive/proxy lineage, apply representation-valid rebuilds or simplification tests, compare configs, and monitor drift.</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_representation_page(ctx: dict[str, Any]) -> None:
    is_rpte = rpte_is_active(ctx["model"])
    is_rpte_leaf = rpte_has_tree_representation(ctx["model"])
    is_rpte_fallback = is_rpte and not is_rpte_leaf
    title = (
        "RPTE Representation Audit"
        if is_rpte_leaf
        else "RPTE Fallback Representation Audit"
        if is_rpte_fallback
        else "Representation Audit"
    )
    st.subheader(title)
    if is_rpte_leaf:
        st.markdown(
            """
            <div class="hugiml-section-note">
              <p>Audits the fitted path from raw inputs to HUGIML source columns, accepted RPTE splits,
              root-to-leaf indicators, direct source columns, and final LR coefficients. Source
              columns selected in tree splits are represented through leaves; direct original features, HUG
              patterns, and augmented pairs are carried directly into the final LR.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif is_rpte_fallback:
        st.markdown(
            """
            <div class="hugiml-section-note">
              <p>RPTE could not form a valid tree in this fitted run. The emergency fallback therefore
              uses HUGIML source columns directly in LR. This page audits those direct fallback terms
              separately and does not describe them as RPTE leaf rules.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="hugiml-section-note">
              <p>Combines complexity and feature-family provenance using explicit model evidence:
              selected parameters, feature-family counts, pattern inventory, and generated-feature provenance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        render_complexity(ctx["model"])

    st.divider()

    if is_rpte_leaf:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Feature-flow audit",
            "Final RPTE LR evidence",
            "Raw-input preprocessing",
            "HUG pattern metadata",
            "Augmented-pair metadata",
        ])
        with tab1:
            render_feature_family_audit(
                ctx["model"],
                ctx["X"],
                sensitive_columns=ctx["roles"]["sensitive_columns"],
                excluded_columns=ctx["roles"]["excluded_columns"],
                id_column=ctx["roles"]["id_column"],
            )
        with tab2:
            render_rpte_rule_evidence(ctx["model"])
        with tab3:
            render_adaptive_binning_evidence(ctx["model"], ctx.get("X"))
        with tab4:
            render_survivor_led_pattern_audit(ctx["model"])
        with tab5:
            render_augmented_pair_traceability(ctx["model"])
        return

    if is_rpte_fallback:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Final fallback LR source terms",
            "Raw-input preprocessing",
            "HUG pattern fallback inputs",
            "Augmented-pair fallback inputs",
            "RPTE fallback evidence",
        ])
        with tab1:
            render_feature_family_audit(
                ctx["model"],
                ctx["X"],
                sensitive_columns=ctx["roles"]["sensitive_columns"],
                excluded_columns=ctx["roles"]["excluded_columns"],
                id_column=ctx["roles"]["id_column"],
            )
        with tab2:
            render_adaptive_binning_evidence(ctx["model"], ctx.get("X"))
        with tab3:
            render_survivor_led_pattern_audit(ctx["model"])
        with tab4:
            render_augmented_pair_traceability(ctx["model"])
        with tab5:
            render_rpte_rule_evidence(ctx["model"])
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Feature families",
        "Adaptive binning evidence",
        "Augmented pair traceability",
        "Survivor-led pattern audit",
        "RPTE final LR evidence",
    ])
    with tab1:
        render_feature_family_audit(
            ctx["model"],
            ctx["X"],
            sensitive_columns=ctx["roles"]["sensitive_columns"],
            excluded_columns=ctx["roles"]["excluded_columns"],
            id_column=ctx["roles"]["id_column"],
        )
    with tab2:
        render_adaptive_binning_evidence(ctx["model"], ctx.get("X"))
    with tab3:
        render_augmented_pair_traceability(ctx["model"])
    with tab4:
        render_survivor_led_pattern_audit(ctx["model"])
    with tab5:
        render_rpte_rule_evidence(ctx["model"])


def _render_data_quality_policy_page(ctx: dict[str, Any]) -> None:
    st.subheader("Data Quality & Policy")
    st.markdown(
        """
        <div class="hugiml-section-note">
          <p>Feature-level missingness and sensitive/proxy column review are grouped together.
          Sensitive columns are surfaced for governance review; they are not automatically removed.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Missingness", "Sensitive / Proxy Review"])
    with tab1:
        render_missingness(ctx["X"], model=ctx["model"])
    with tab2:
        render_fairness(
            ctx["roles"]["sensitive_columns"],
            model=ctx["model"],
            X=ctx["X"],
            y=ctx.get("y"),
        )


def _render_page(page: str, ctx: dict[str, Any]) -> None:
    if page == "Overview":
        render_overview(
            model=ctx["model"],
            result=ctx["result"],
            roles=ctx["roles"],
            meta=ctx["meta"],
            X=ctx.get("X"),
            y=ctx.get("y"),
        )
    elif page == "Validation":
        cv_results = pd.DataFrame(getattr(ctx["result"], "results_", []))
        render_performance(
            cv_results,
            model=ctx["model"],
            X=ctx["X"],
            y=ctx.get("y"),
            evaluation=ctx.get("evaluation"),
        )
    elif page == "Representation Audit":
        _render_representation_page(ctx)
    elif page == "Pattern Inventory":
        render_patterns(ctx["model"])
    elif page == "Case Review":
        render_prediction(ctx["model"], ctx["X"], ctx["case_ids"])
    elif page == "Data Quality & Policy":
        _render_data_quality_policy_page(ctx)
    elif page == "Configuration Comparison":
        render_config_comparison(ctx)
    elif page == "Representation Pruning":
        render_pruning_analysis(ctx)
    elif page == "Monitoring":
        render_drift(ctx["model"], ctx["X"], ctx.get("y"))
    else:
        st.warning(f"Unknown page: {page}")


def _render_top_nav() -> tuple[str, str]:
    """Header row (title + theme selectbox) and sidebar workspace buttons.

    Returns (section, workspace). Navigation state lives in
    ``hugiml_nav_section`` (never a widget key) to avoid StreamlitAPIException.
    """
    # Consume deferred navigation flags before any widget is created.
    if _pop_session_flag("hugiml_governance_requested"):
        st.session_state["hugiml_nav_section"] = "Governance"
    if _pop_session_flag("hugiml_workbench_open_results"):
        st.session_state["hugiml_nav_section"] = "Results"

    current_theme = st.session_state.get("hugiml_theme", "Ocean")
    _THEMES = ["Ocean", "Forest", "Dark"]

    # ── Main panel: title left, theme right ──────────────────────────────
    t = _theme_tokens(current_theme)
    hdr_left, hdr_right = st.columns([3, 1], vertical_alignment="center")

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    hdr_left, hdr_right = st.columns([0.78, 0.22], vertical_alignment="center")
    with hdr_left:
        st.markdown(
            f'<div style="font-size:1.05rem;font-weight:900;letter-spacing:-0.03em;'
            f'color:{t["accent"]};line-height:1.2;margin:14px 0 0 0;">'
            f'HUGIML Governance Studio</div>',
            unsafe_allow_html=True,
        )

    with hdr_right:
        st.selectbox(
            "Theme",
            _THEMES,
            index=_THEMES.index(current_theme),
            key="hugiml_theme",
            label_visibility="collapsed",
        )

    st.markdown(
        '<hr style="margin:12px 0 8px 0;border:none;border-top:1px solid rgba(128,128,128,0.15);"/>',
        unsafe_allow_html=True,
    )

    # ── Sidebar: brand strip + two boxed workspace buttons ───────────────
    current_section = st.session_state.get("hugiml_nav_section", "Setup")
    current_ws = "Governance" if current_section == "Governance" else "Workbench"

    st.sidebar.markdown(
        '''<div class="hugiml-brand">
          <div class="hugiml-brand-name">HUGIML Studio</div>
          <div class="hugiml-brand-tag">Governance &amp; Workbench</div>
        </div>''',
        unsafe_allow_html=True,
    )

    wb_is_active = (current_ws == "Workbench")
    gov_is_active = (current_ws == "Governance")

    sb_col1, sb_col2 = st.sidebar.columns(2, gap="small")
    with sb_col1:
        wb_clicked = st.button(
            "Workbench",
            key="hugiml_ws_wb",
            width="stretch",
            type="primary" if wb_is_active else "secondary",
        )
    with sb_col2:
        gov_clicked = st.button(
            "Governance",
            key="hugiml_ws_gov",
            width="stretch",
            type="primary" if gov_is_active else "secondary",
        )

    if wb_clicked:
        st.session_state["hugiml_nav_section"] = (
            current_section if current_section in ("Setup", "Results") else "Setup"
        )
        st.rerun()
    if gov_clicked:
        st.session_state["hugiml_nav_section"] = "Governance"
        st.rerun()

    workspace = current_ws
    if workspace == "Governance":
        chosen_section = "Governance"
    else:
        chosen_section = current_section if current_section in ("Setup", "Results") else "Setup"
    st.session_state["hugiml_nav_section"] = chosen_section
    return chosen_section, workspace


def run_app() -> None:
    st.set_page_config(page_title="HUGIML Governance Studio", page_icon="H", layout="wide")

    args = _parse_args()

    theme = st.session_state.get("hugiml_theme", "Ocean")
    _apply_css(theme)

    section, workspace = _render_top_nav()

    if workspace == "Governance" and not isinstance(
        st.session_state.get("hugiml_promoted_governance_ctx"), dict
    ):
        st.info(
            "No promoted HUGIML run yet. Go to Workbench \u2192 Results \u2192 "
            "select a HUGIML run \u2192 Promote to Governance."
        )
        return

    ctx, theme = _load_context(args, workspace)

    if ctx is None:
        return

    st.sidebar.markdown('<div class="hugiml-sidebar-section">Column Roles</div>', unsafe_allow_html=True)
    with st.sidebar.expander("View roles", expanded=False):
        st.json(ctx["roles"])

    if workspace == "Workbench":
        render_workbench(ctx, section=section)
        return

    st.sidebar.markdown('<div class="hugiml-sidebar-section">Views</div>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Governance views",
        SECTION_LABELS,
        index=0,
        key="hugiml_dashboard_page",
        label_visibility="collapsed",
    )

    _render_hero(ctx)
    _render_governance_workflow()
    _render_page(page, ctx)


def _is_running_under_streamlit() -> bool:
    """Return True when this file is being executed by ``streamlit run``.

    Console-script entry points execute Python modules directly. Calling
    Streamlit APIs in that context only emits "missing ScriptRunContext"
    warnings and exits, so the installed ``hugiml-dashboard`` command must
    delegate to Streamlit unless Streamlit is already running this script.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def main() -> None:
    if _is_running_under_streamlit() or os.environ.get("HUGIML_DASHBOARD_BARE") == "1":
        run_app()
        return

    # Installed console-script path: launch the real Streamlit runner while
    # preserving dashboard args such as ``--cv`` and ``--random-state`` after
    # Streamlit's ``--`` separator.
    from streamlit.web import cli as streamlit_cli

    app_path = str(Path(__file__).resolve())
    dashboard_args = list(sys.argv[1:])
    sys.argv = ["streamlit", "run", app_path]
    if dashboard_args:
        sys.argv.extend(["--", *dashboard_args])
    streamlit_cli.main()


if __name__ == "__main__":
    main()

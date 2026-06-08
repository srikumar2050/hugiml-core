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
from hugiml.dashboard.components.missingness import render_missingness
from hugiml.dashboard.components.overview import render_overview
from hugiml.dashboard.components.patterns import render_patterns
from hugiml.dashboard.components.performance import render_performance
from hugiml.dashboard.components.prediction import render_prediction
from hugiml.dashboard.components.pruning import render_pruning_analysis
from hugiml.dashboard.data import prepare_model_frame
from hugiml.dashboard.demo import demo_roles, load_demo_credit_risk
from hugiml.dashboard.display import dataframe_for_display
from hugiml.dashboard.runner import score_cases, train_hugiml

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
    """Apply only dashboard-owned CSS.

    Important:
    - hide deploy/menu controls only;
    - do not hide the sidebar collapse/expand control;
    - do not globally override Streamlit widget/dataframe text colors.
    """
    t = _theme_tokens(theme)
    dark_extra = ""
    if t["dark"]:
        dark_extra = """
        .hugiml-hero,
        .hugiml-section-note {
            background:
                radial-gradient(circle at 6% 5%, rgba(96, 165, 250, 0.18), transparent 34%),
                radial-gradient(circle at 92% 12%, rgba(167, 139, 250, 0.15), transparent 30%),
                #111827 !important;
            color: #f8fafc !important;
            border-color: rgba(147, 197, 253, 0.42) !important;
        }

        .hugiml-hero h1,
        .hugiml-hero p,
        .hugiml-section-note p,
        .hugiml-chip {
            color: #f8fafc !important;
        }

        .hugiml-chip {
            background: rgba(30, 41, 59, 0.92) !important;
            border-color: rgba(147, 197, 253, 0.44) !important;
        }

        div[data-testid="stMetric"] {
            background:
                linear-gradient(180deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.88)),
                #111827 !important;
            color: #f8fafc !important;
            border-color: rgba(147, 197, 253, 0.34) !important;
            border-top-color: #60a5fa !important;
        }

        div[data-testid="stMetric"] * {
            color: #f8fafc !important;
        }

        .hugiml-fit-complete {
            color: #f8fafc !important;
            background: rgba(30, 41, 59, 0.92) !important;
            border-color: rgba(147, 197, 253, 0.44) !important;
        }
        """

    st.markdown(
        f"""
        <style>
        /* Hide only the app menu/deploy/status affordances. Do not hide the
           header itself or header buttons, because that can break sidebar
           collapse/expand behavior. */
        #MainMenu,
        div[data-testid="stDeployButton"],
        div[data-testid="stStatusWidget"],
        .stDeployButton {{
            display: none !important;
            visibility: hidden !important;
        }}

        .block-container {{
            padding-top: 1.1rem;
            padding-bottom: 2.25rem;
            max-width: 1480px;
        }}

        .hugiml-hero {{
            background:
                radial-gradient(circle at 5% 5%, {t["hero_a"]}, transparent 34%),
                radial-gradient(circle at 92% 12%, {t["hero_b"]}, transparent 30%),
                var(--secondary-background-color);
            border: 1px solid {t["border"]};
            border-radius: 20px;
            padding: 24px 26px;
            margin-bottom: 18px;
            box-shadow: {t["shadow"]};
        }}

        .hugiml-hero h1 {{
            margin: 0 0 8px 0;
            font-size: clamp(1.7rem, 2.8vw, 2.35rem);
            letter-spacing: -0.035em;
            line-height: 1.06;
            font-weight: 850;
        }}

        .hugiml-hero p {{
            margin: 0;
            opacity: 0.82;
            line-height: 1.52;
            max-width: 1080px;
        }}

        .hugiml-chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }}

        .hugiml-chip {{
            display: inline-flex;
            align-items: center;
            border: 1px solid {t["border"]};
            background: {t["chip_bg"]};
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1.2;
        }}

        .hugiml-section-note {{
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-left: 5px solid {t["accent"]};
            background:
                linear-gradient(90deg, {t["hero_a"]}, transparent),
                var(--secondary-background-color);
            border-radius: 12px;
            padding: 12px 14px;
            margin: 8px 0 16px 0;
        }}

        .hugiml-section-note p {{
            margin: 0;
            opacity: 0.84;
            line-height: 1.45;
        }}

        div[data-testid="stMetric"] {{
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-top: 3px solid {t["accent"]};
            border-radius: 14px;
            padding: 14px 14px 12px 14px;
            background: var(--secondary-background-color);
            min-height: 104px;
        }}

        div[data-testid="stMetricValue"] {{
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            line-height: 1.12 !important;
            font-size: clamp(1.05rem, 1.45vw, 1.65rem) !important;
        }}

        .hugiml-sidebar-title {{
            font-size: 1.08rem;
            font-weight: 850;
            line-height: 1.15;
            letter-spacing: -0.01em;
            margin-bottom: 0.15rem;
        }}

        .hugiml-sidebar-subtitle {{
            opacity: 0.72;
            font-size: 0.82rem;
            margin-bottom: 0.75rem;
        }}

        .hugiml-fit-complete {{
            border: 1px solid {t["border"]};
            background: {t["chip_bg"]};
            border-radius: 12px;
            padding: 9px 11px;
            margin-top: 8px;
            font-size: 0.86rem;
            line-height: 1.35;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            flex-wrap: wrap;
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 999px;
            padding: 8px 14px;
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
        }}

        {dark_extra}

        @media (max-width: 900px) {{
            .hugiml-hero {{
                padding: 18px;
            }}
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


@st.cache_data(show_spinner=False)
def _load_demo_context_cached() -> dict[str, Any]:
    X_raw, y = load_demo_credit_risk()
    roles = demo_roles()
    X, y, case_ids, meta = prepare_model_frame(
        X_raw.assign(default_risk=y),
        target="default_risk",
        id_column=roles["id_column"],
        excluded_columns=roles["excluded_columns"],
    )
    return {
        "mode": "Demo dataset",
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


def _sidebar_dataset_controls(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None, int, int, str]:
    st.sidebar.markdown('<div class="hugiml-sidebar-title">HUGIML Governance Studio</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="hugiml-sidebar-subtitle">Audit evidence workspace</div>', unsafe_allow_html=True)

    theme = st.sidebar.radio(
        "Theme",
        ["Ocean", "Forest", "Dark"],
        index=["Ocean", "Forest", "Dark"].index(st.session_state.get("hugiml_theme", "Ocean")),
        horizontal=True,
        key="hugiml_theme",
    )

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
        return _load_demo_context_cached(), _config_key("demo", int(cv), int(random_state)), int(cv), int(random_state), theme

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
                      <h1>Upload a dataset to start</h1>
                      <p>Choose a CSV/TSV/Excel/Parquet file, then define target and review columns.</p>
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


def _load_context(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    data_ctx, cache_key, cv, random_state, theme = _sidebar_dataset_controls(args)
    if data_ctx is None:
        return None, theme

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

    st.markdown(
        f"""
        <div class="hugiml-hero">
          <h1>HUGIML Governance Studio</h1>
          <p>Audit-ready evidence view for validation performance, model complexity,
          feature-family provenance, HUG pattern inventory, case explanations, data quality,
          representation pruning, configuration comparison, and monitoring signals.</p>
          <div class="hugiml-chip-row">
            <span class="hugiml-chip">Data: {ctx['mode']}</span>
            <span class="hugiml-chip">Rows: {ctx['meta']['n_rows']:,}</span>
            <span class="hugiml-chip">Features: {ctx['meta']['n_features']:,}</span>
            <span class="hugiml-chip">Best CV: {score_text}</span>
            <span class="hugiml-chip">Mode: {getattr(model, 'feature_mode', 'N/A')}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_representation_page(ctx: dict[str, Any]) -> None:
    st.subheader("Representation Audit")
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

    render_feature_family_audit(
        ctx["model"],
        ctx["X"],
        sensitive_columns=ctx["roles"]["sensitive_columns"],
        excluded_columns=ctx["roles"]["excluded_columns"],
        id_column=ctx["roles"]["id_column"],
    )


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
        render_missingness(ctx["X"])
    with tab2:
        render_fairness(ctx["roles"]["sensitive_columns"])


def _render_page(page: str, ctx: dict[str, Any]) -> None:
    if page == "Overview":
        render_overview(
            model=ctx["model"],
            result=ctx["result"],
            roles=ctx["roles"],
            meta=ctx["meta"],
        )
    elif page == "Validation":
        cv_results = pd.DataFrame(getattr(ctx["result"], "results_", []))
        render_performance(cv_results)
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
        render_drift(ctx["model"], ctx["X"])
    else:
        st.warning(f"Unknown page: {page}")


def run_app() -> None:
    st.set_page_config(page_title="HUGIML Governance Studio", page_icon="H", layout="wide")

    args = _parse_args()

    ctx, theme = _load_context(args)
    _apply_css(theme)

    if ctx is None:
        return

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Evidence views",
        SECTION_LABELS,
        index=0,
        key="hugiml_dashboard_page",
    )

    with st.sidebar.expander("Column roles", expanded=False):
        st.json(ctx["roles"])

    _render_hero(ctx)
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

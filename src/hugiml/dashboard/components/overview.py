"""Executive overview component."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from hugiml.dashboard.display import dataframe_for_display


def _metric_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_overview(model: Any = None, result: Any = None, roles: dict | None = None, meta: dict | None = None) -> None:
    roles = roles or {}
    meta = meta or {}

    st.subheader("Executive Overview")
    st.markdown(
        """
        <div class="hugiml-section-note">
          <p>Concise governance summary of model configuration, column roles, and evidence available for review.
          No opaque interpretability score is used; auditability is represented through explicit artifacts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_score = getattr(result, "best_score_", None) if result is not None else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows reviewed", _metric_value(meta.get("n_rows")))
    c2.metric("Model features", _metric_value(meta.get("n_features")))
    c3.metric("Best CV score", _metric_value(best_score))
    positive_rate = meta.get("positive_rate")
    c4.metric("Positive rate", f"{positive_rate:.2%}" if isinstance(positive_rate, (int, float)) else "N/A")

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("#### Evidence status")
        status_rows = [
            {"area": "Validation evidence", "status": "Available", "evidence": "CV/tuning result table"},
            {"area": "Representation audit", "status": "Available", "evidence": "Original, pattern, and augmented/generated feature families"},
            {"area": "Pattern inventory", "status": "Model-dependent", "evidence": "Human-readable HUG pattern table when exposed by model"},
            {"area": "Case review", "status": "Available", "evidence": "Prediction and case input traceability"},
            {"area": "Data quality", "status": "Available", "evidence": "Feature-level missingness summary"},
            {"area": "Monitoring", "status": "Model/API-dependent", "evidence": "Prediction summary, drift output, monitor report"},
        ]
        st.dataframe(dataframe_for_display(pd.DataFrame(status_rows)), width="stretch", hide_index=True)

    with right:
        st.markdown("#### Column roles")
        role_rows = [
            {"role": "Target", "columns": roles.get("target") or "N/A"},
            {"role": "ID / case lookup", "columns": roles.get("id_column") or "Generated case IDs"},
            {"role": "Excluded from modeling", "columns": ", ".join(roles.get("excluded_columns", [])) or "None"},
            {"role": "Sensitive/proxy review", "columns": ", ".join(roles.get("sensitive_columns", [])) or "None"},
        ]
        st.dataframe(dataframe_for_display(pd.DataFrame(role_rows)), width="stretch", hide_index=True)

        st.markdown("#### Selected HUGIML configuration")
        config_rows = [
            {"parameter": "L", "value": getattr(model, "L", "N/A"), "meaning": "Maximum pattern order used by the selected model."},
            {"parameter": "topK", "value": getattr(model, "topK", "N/A"), "meaning": "Selected pattern budget."},
            {"parameter": "G", "value": getattr(model, "G", "N/A"), "meaning": "Mining threshold used by the selected model."},
            {"parameter": "feature_mode", "value": str(getattr(model, "feature_mode", "N/A")), "meaning": "Final representation regime. Shown as text to avoid truncation."},
        ]
        st.dataframe(
            dataframe_for_display(pd.DataFrame(config_rows)),
            width="stretch",
            hide_index=True,
            column_config={
                "parameter": st.column_config.TextColumn("Parameter", width="small"),
                "value": st.column_config.TextColumn("Value", width="medium"),
                "meaning": st.column_config.TextColumn("Meaning", width="large"),
            },
        )

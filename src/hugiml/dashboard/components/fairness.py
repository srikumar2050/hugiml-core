"""Sensitive/proxy feature review component."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from hugiml.dashboard.display import dataframe_for_display


def render_fairness(sensitive_columns: list[str] | None = None, *args, **kwargs) -> None:
    st.markdown("### Sensitive / Proxy Feature Review")
    sensitive_columns = sensitive_columns or []

    if not sensitive_columns:
        st.info("No sensitive/proxy-review columns were provided.")
        return

    df = pd.DataFrame({
        "column": sensitive_columns,
        "role": ["Sensitive / proxy review"] * len(sensitive_columns),
        "required_action": ["Review usage in originals, HUG patterns, and augmented/generated features"] * len(sensitive_columns),
    })
    st.dataframe(dataframe_for_display(df), width="stretch", hide_index=True)

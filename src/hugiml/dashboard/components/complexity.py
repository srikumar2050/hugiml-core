"""Complexity and budget component."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from hugiml.dashboard.display import dataframe_for_display


def render_complexity(model: Any = None, *args, **kwargs) -> None:
    st.markdown("### Complexity and Budget Evidence")
    if model is None:
        st.info("No model loaded.")
        return

    config = {
        "L": getattr(model, "L", None),
        "topK": getattr(model, "topK", None),
        "G": getattr(model, "G", None),
        "feature_mode": getattr(model, "feature_mode", None),
        "topk_budget_strict": getattr(model, "topk_budget_strict", None),
    }
    config = {k: v for k, v in config.items() if v is not None}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("L", config.get("L", "N/A"))
    c2.metric("topK", config.get("topK", "N/A"))
    c3.metric("G", config.get("G", "N/A"))
    c4.metric("Feature mode", str(config.get("feature_mode", "N/A")))

    st.markdown("#### Explicit configuration")
    st.json(config)

    rows = [
        {"regime": "patterns_only", "audit_meaning": "Representation is limited to selected HUG pattern features."},
        {"regime": "original_plus_patterns", "audit_meaning": "Original business features are retained alongside HUG pattern features."},
        {"regime": "original_plus_interactions", "audit_meaning": "Original features plus multi-feature HUG interactions."},
        {"regime": "strict budget", "audit_meaning": "Optional mode that caps the final representation for compactness-sensitive deployments."},
    ]
    st.dataframe(dataframe_for_display(pd.DataFrame(rows)), width="stretch", hide_index=True)

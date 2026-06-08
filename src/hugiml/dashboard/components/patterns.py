"""Pattern inventory component."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def _patterns_to_frame(model: Any) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame()

    try:
        if hasattr(model, "get_pattern_info"):
            value = model.get_pattern_info()
            if isinstance(value, pd.DataFrame):
                return value
            if value is not None:
                return pd.DataFrame(value)
    except Exception:
        pass

    labels = None
    for attr in ("pattern_labels_", "_pattern_labels_", "raw_patterns_", "patterns_"):
        try:
            labels = getattr(model, attr, None)
            if labels is not None:
                break
        except Exception:
            labels = None

    if labels is None:
        return pd.DataFrame()

    try:
        return pd.DataFrame({"pattern": list(labels)})
    except Exception:
        return pd.DataFrame({"pattern": [str(labels)]})


def render_patterns(model: Any = None, *args, **kwargs) -> None:
    st.subheader("Pattern Inventory")
    st.markdown(
        """
        <div class="hugiml-section-note">
          <p>Human-readable HUG patterns are the primary explanation artifact. Search by feature name,
          category, interval, or any text contained in the pattern label.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = _patterns_to_frame(model)
    if df.empty:
        st.info("Pattern information is not available for this model/version.")
        return

    c1, c2 = st.columns([0.7, 0.3], gap="large")
    with c1:
        query = st.text_input("Search patterns", value="", placeholder="feature name, interval, category...")
    with c2:
        st.metric("Patterns shown", f"{len(df):,}")

    show_df = df
    if query:
        mask = df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        show_df = df.loc[mask]

    st.dataframe(show_df, width="stretch", hide_index=True)

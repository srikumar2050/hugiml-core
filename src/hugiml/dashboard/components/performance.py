"""Performance comparison component."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from hugiml.dashboard.display import dataframe_for_display


def render_performance(results: Any = None, *args, **kwargs) -> None:
    st.subheader("Validation Evidence")
    st.markdown(
        """
        <div class="hugiml-section-note">
          <p>Shows HUGIML tuning/CV evidence when available. Baseline comparisons can be generated
          by benchmark workflows and shown here when supplied by the calling pipeline.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if results is None:
        st.info("No performance result table is available.")
        return

    try:
        df = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    except Exception as exc:
        st.warning(f"Could not convert results to a table: {exc}")
        st.write(results)
        return

    if df.empty:
        st.info("No performance rows were returned.")
        return

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    score_cols = [c for c in numeric_cols if "score" in str(c).lower() or "auc" in str(c).lower()]
    score_col = score_cols[0] if score_cols else None

    if score_col:
        best_idx = df[score_col].idxmax()
        best_row = df.loc[best_idx]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best displayed score", f"{float(best_row[score_col]):.4f}")
        c2.metric("Rows compared", f"{len(df):,}")
        c3.metric("Score column", str(score_col))

    with st.container(border=True):
        st.markdown("#### CV / tuning result table")
        st.dataframe(dataframe_for_display(df), width="stretch", hide_index=True)

    if score_col:
        with st.container(border=True):
            st.markdown("#### Best row by displayed score")
            st.dataframe(df.loc[[best_idx]], width="stretch", hide_index=True)

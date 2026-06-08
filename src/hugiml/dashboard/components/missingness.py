"""Missing-value governance component."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from hugiml.dashboard.display import dataframe_for_display


def render_missingness(X: pd.DataFrame | None = None, *args, **kwargs) -> None:
    st.markdown("### Missingness Evidence")
    if X is None:
        st.info("No dataset loaded.")
        return

    rows = []
    for c in X.columns:
        rows.append({
            "feature": str(c),
            "dtype": str(X[c].dtype),
            "missing_pct": float(X[c].isna().mean() * 100.0),
            "n_missing": int(X[c].isna().sum()),
        })
    df = pd.DataFrame(rows).sort_values("missing_pct", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Features reviewed", f"{len(df):,}")
    c2.metric("Mean missingness", f"{df['missing_pct'].mean():.2f}%")
    c3.metric("Max missingness", f"{df['missing_pct'].max():.2f}%")

    st.dataframe(dataframe_for_display(df), width="stretch", hide_index=True)
    st.caption("Derived metrics are computed directly from the currently loaded dashboard dataframe.")

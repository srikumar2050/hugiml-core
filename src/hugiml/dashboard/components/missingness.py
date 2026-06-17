"""Missing-value governance component."""

from __future__ import annotations

from typing import Any

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hugiml.dashboard.display import dataframe_for_display


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, dict):
        return {str(k) for k in value.keys()}
    try:
        return {str(v) for v in value}
    except Exception:
        return {str(value)}


def missingness_frame(X: pd.DataFrame | None = None, model: Any = None) -> pd.DataFrame:
    if X is None:
        return pd.DataFrame()
    missing_edges = _as_set(getattr(model, "_missing_col_edges_", None))
    binary_cats = _as_set(getattr(model, "binary_categorical_cols_", None))
    rows = []
    for c in X.columns:
        name = str(c)
        rows.append({
            "feature": name,
            "dtype": str(X[c].dtype),
            "missing_pct": float(X[c].isna().mean() * 100.0),
            "n_missing": int(X[c].isna().sum()),
            "model_missing_edge": name in missing_edges,
            "binary_inferred_categorical": name in binary_cats,
        })
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


def render_missingness(X: pd.DataFrame | None = None, model: Any = None, *args, **kwargs) -> None:
    st.markdown("### Missingness Evidence")
    df = missingness_frame(X, model=model)
    if df.empty:
        st.info("No dataset loaded.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features reviewed", f"{len(df):,}")
    c2.metric("Mean missingness", f"{df['missing_pct'].mean():.2f}%")
    c3.metric("Max missingness", f"{df['missing_pct'].max():.2f}%")
    c4.metric("Model missing edges", f"{int(df['model_missing_edge'].sum()):,}")

    st.dataframe(dataframe_for_display(df), width="stretch", hide_index=True)
    st.caption(
        "Raw missingness is computed from the dashboard dataframe. model_missing_edge and "
        "binary_inferred_categorical come from fitted HUGIML metadata when available."
    )

    # Co-missingness heatmap: features missing together share an upstream data source
    _miss_cols = [c for c in X.columns if X[c].isna().any()]
    if len(_miss_cols) >= 2:
        try:
            _corr = X[_miss_cols].isna().corr()
            _pairs = []
            for _i, _c1 in enumerate(_miss_cols):
                for _c2 in _miss_cols[_i + 1:]:
                    _r = float(_corr.loc[_c1, _c2])
                    if abs(_r) >= 0.30:
                        _pairs.append({"feature_A": _c1, "feature_B": _c2, "co_missing_r": round(_r, 3)})
            if _pairs:
                _pairs_df = pd.DataFrame(_pairs).sort_values("co_missing_r", ascending=False)
                with st.expander(f"Co-missingness pairs (|r| ≥ 0.30) — {len(_pairs_df)} found", expanded=True):
                    st.caption(
                        "High correlation means features are missing together — they likely share an upstream "
                        "data source. This is a data pipeline dependency auditors must document."
                    )
                    _fig_cm, _ax_cm = plt.subplots(figsize=(7, max(2.0, len(_pairs_df) * 0.40)))
                    _ax_cm.barh(
                        [f"{r['feature_A']} ↔ {r['feature_B']}" for _, r in _pairs_df.iterrows()],
                        _pairs_df["co_missing_r"],
                        color=["#E24B4A" if r > 0.5 else "#EF9F27" for r in _pairs_df["co_missing_r"]],
                    )
                    _ax_cm.axvline(0.30, linestyle="--", linewidth=0.8, color="#888780", alpha=0.7)
                    _ax_cm.set_xlabel("Co-missingness correlation (r)")
                    _ax_cm.set_title("Feature pairs with correlated missingness")
                    _ax_cm.set_xlim(0, 1.05)
                    _ax_cm.invert_yaxis()
                    _fig_cm.tight_layout()
                    st.pyplot(_fig_cm)
                    plt.close(_fig_cm)
                    st.dataframe(dataframe_for_display(_pairs_df), width="stretch", hide_index=True)
        except Exception:
            pass

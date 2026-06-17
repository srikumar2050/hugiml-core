"""Pattern inventory component."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hugiml.dashboard.display import dataframe_for_display


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

    st.divider()
    render_pattern_support_coverage_audit(model, df)


def _pattern_support_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    support_col = next((c for c in df.columns if str(c).lower() == "support"), None)
    if support_col is None:
        return pd.DataFrame()
    out = df.copy()
    out["support"] = pd.to_numeric(out[support_col], errors="coerce")
    return out.loc[out["support"].notna()].copy()


def _row_pattern_counts_from_model(model: Any) -> pd.Series | None:
    matrix = getattr(model, "x_train_hup_", None)
    if matrix is None:
        return None
    try:
        row_counts = matrix.sum(axis=1)
        if hasattr(row_counts, "A1"):
            values = row_counts.A1
        else:
            values = pd.Series(row_counts).to_numpy().ravel()
        return pd.Series(values, name="n_patterns_fired").astype(float)
    except Exception:
        return None


def population_coverage_frame(model: Any) -> pd.DataFrame:
    row_counts = _row_pattern_counts_from_model(model)
    if row_counts is None or row_counts.empty:
        return pd.DataFrame()
    thresholds = [1, 5, 10]
    rows = [
        {
            "coverage_level": "rows with zero pattern coverage",
            "threshold": 0,
            "row_count": int((row_counts == 0).sum()),
            "fraction_of_training_rows": float((row_counts == 0).mean()),
        }
    ]
    for threshold in thresholds:
        mask = row_counts >= threshold
        rows.append({
            "coverage_level": f"rows covered by at least {threshold} pattern" + ("s" if threshold != 1 else ""),
            "threshold": threshold,
            "row_count": int(mask.sum()),
            "fraction_of_training_rows": float(mask.mean()),
        })
    return pd.DataFrame(rows)


def render_pattern_support_coverage_audit(model: Any, pattern_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Render model-level pattern support and population-coverage evidence."""
    st.markdown("### Pattern Support Distribution & Population Coverage")
    st.caption(
        "Audits whether mined patterns are broadly grounded in the training population or concentrated in sparse subgroups."
    )

    support_df = _pattern_support_distribution(pattern_df)
    coverage_df = population_coverage_frame(model)

    c1, c2, c3 = st.columns(3)
    if support_df.empty:
        c1.metric("Median pattern support", "N/A")
        c2.metric("Patterns with support < 0.05", "N/A")
    else:
        c1.metric("Median pattern support", f"{support_df['support'].median():.4f}")
        c2.metric("Patterns with support < 0.05", f"{(support_df['support'] < 0.05).mean():.2%}")
    if coverage_df.empty:
        c3.metric("Zero-coverage rows", "N/A")
    else:
        zero = coverage_df.loc[coverage_df["threshold"].eq(0), "fraction_of_training_rows"]
        c3.metric("Zero-coverage rows", f"{zero.iloc[0]:.2%}" if len(zero) else "N/A")

    tab1, tab2 = st.tabs(["Support distribution", "Population coverage"])
    with tab1:
        if support_df.empty:
            st.info("Pattern support values are not available in get_pattern_info().")
        else:
            bins = np.linspace(0.0, 1.0, 21)
            hist_counts, hist_edges = np.histogram(support_df["support"].clip(0, 1), bins=bins)
            hist_df = pd.DataFrame({
                "support_bin": [f"[{hist_edges[i]:.2f}, {hist_edges[i + 1]:.2f})" for i in range(len(hist_counts))],
                "pattern_count": hist_counts.astype(int),
            })
            st.bar_chart(hist_df.set_index("support_bin")["pattern_count"])
            st.dataframe(dataframe_for_display(support_df), width="stretch", hide_index=True)
    with tab2:
        if coverage_df.empty:
            st.info(
                "Population coverage requires audit-mode training pattern activations on model.x_train_hup_. "
                "This matrix is usually not retained in production-mode model objects."
            )
        else:
            st.dataframe(dataframe_for_display(coverage_df), width="stretch", hide_index=True)
            # Pattern accumulation curve
            _matrix = getattr(model, "x_train_hup_", None)
            if _matrix is not None:
                try:
                    _arr = _matrix.toarray() if hasattr(_matrix, "toarray") else np.asarray(_matrix)
                    if _arr.ndim == 2 and _arr.shape[1] > 1:
                        _covered = np.zeros(_arr.shape[0], dtype=bool)
                        _cum = []
                        for _j in range(_arr.shape[1]):
                            _covered |= _arr[:, _j].astype(bool)
                            _cum.append(float(_covered.mean()))
                        _acc_df = pd.DataFrame({
                            "patterns_added": range(1, len(_cum) + 1),
                            "population_coverage": _cum,
                        })
                        st.markdown("#### Pattern accumulation curve")
                        st.caption("Cumulative population coverage as patterns are added in utility-rank order. The elbow shows where marginal coverage becomes negligible.")
                        _fig_acc, _ax_acc = plt.subplots(figsize=(7, 3.5))
                        _ax_acc.plot(_acc_df["patterns_added"], _acc_df["population_coverage"], color="#534AB7", linewidth=1.5)
                        _ax_acc.fill_between(_acc_df["patterns_added"], _acc_df["population_coverage"], alpha=0.12, color="#534AB7")
                        _ax_acc.set_xlabel("Number of patterns (ranked by utility)")
                        _ax_acc.set_ylabel("Fraction of population covered")
                        _ax_acc.set_ylim(0, 1.05)
                        _ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
                        _fig_acc.tight_layout()
                        st.pyplot(_fig_acc)
                        plt.close(_fig_acc)
                except Exception:
                    pass
    return {"support": support_df, "coverage": coverage_df}

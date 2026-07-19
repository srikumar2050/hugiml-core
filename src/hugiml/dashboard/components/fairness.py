"""Sensitive/proxy feature review component."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from hugiml.dashboard.components.feature_family import pattern_feature_audit
from hugiml.dashboard.components.patterns import (
    _get_rpte_final_term_rows,
    _is_direct_source_row,
    _rpte_source_family,
)
from hugiml.dashboard.display import dataframe_for_display


def _positive_probabilities(model: Any, X: pd.DataFrame | None) -> np.ndarray | None:
    if model is None or X is None:
        return None
    try:
        proba = model.predict_proba(X)
    except Exception:
        return None
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr[:, 1]
    return arr.ravel()


def predicted_positive_rate_by_group(
    model: Any,
    X: pd.DataFrame | None,
    sensitive_columns: list[str] | None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return predicted positive rates split by sensitive/proxy group value."""
    sensitive_columns = list(sensitive_columns or [])
    p = _positive_probabilities(model, X)
    if X is None or p is None or not sensitive_columns:
        return pd.DataFrame()
    n = min(len(X), len(p))
    if n == 0:
        return pd.DataFrame()
    proba = np.asarray(p[:n], dtype=float)
    pred = proba >= float(threshold)
    rows: list[dict[str, Any]] = []
    Xn = X.iloc[:n]
    for column in sensitive_columns:
        if column not in Xn.columns:
            continue
        values = Xn[column].astype("string").fillna("<missing>")
        for value, idx in values.groupby(values).groups.items():
            pos = pred[list(idx)]
            probs = proba[list(idx)]
            rows.append({
                "column": str(column),
                "group_value": str(value),
                "n_rows": int(len(idx)),
                "mean_predicted_probability": float(np.mean(probs)) if len(probs) else np.nan,
                "predicted_positive_rate": float(np.mean(pos)) if len(pos) else np.nan,
                "threshold": float(threshold),
            })
    return pd.DataFrame(rows).sort_values(["column", "predicted_positive_rate", "group_value"], ascending=[True, False, True]) if rows else pd.DataFrame()


def sensitive_pattern_flags(model: Any, sensitive_columns: list[str] | None) -> pd.DataFrame:
    """Return mined patterns that include sensitive/proxy source features."""
    if not sensitive_columns:
        return pd.DataFrame()
    try:
        df = pattern_feature_audit(model, sensitive_columns=sensitive_columns)
    except Exception:
        return pd.DataFrame()
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()
    out = df.loc[df["status"].astype(str).str.lower().eq("review")].copy()
    return out


def sensitive_rpte_rule_flags(model: Any, sensitive_columns: list[str] | None) -> pd.DataFrame:
    """Return sensitive/proxy flags across the complete RPTE final-LR terms."""
    if not sensitive_columns:
        return pd.DataFrame()
    rows = _get_rpte_final_term_rows(model, include_zero_direct=True)
    if not rows:
        return pd.DataFrame()
    sensitive = set(map(str, sensitive_columns))
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        coefficient = pd.to_numeric(
            pd.Series([row.get("final_logistic_coefficient")]), errors="coerce"
        ).iloc[0]
        if pd.isna(coefficient) or abs(float(coefficient)) < 1e-12:
            continue
        raw_sources = [str(s) for s in (row.get("raw_sources") or [])]
        hit = sensitive.intersection(raw_sources)
        if not hit:
            continue
        backend = str(row.get("backend") or "unknown")
        source_column = str(row.get("downstream_feature") or "")
        family = str(row.get("source_family") or _rpte_source_family(source_column))
        if backend in {"bounded_lookahead", "sequential_default"}:
            term_type = "RPTE leaf indicator"
            conditions = row.get("conditions") or []
            terms = [
                c.get("raw_condition") or c.get("downstream_condition") or "?"
                for c in conditions
            ]
            term = " AND ".join(str(t) for t in terms)
        elif _is_direct_source_row(row):
            term_type = {
                "original": "direct original term",
                "pattern": "direct HUG pattern term",
                "augmented_pair": "direct augmented-pair term",
            }.get(family, "direct source term")
            term = source_column
        else:
            term_type = "RPTE source fallback term"
            term = source_column or "(linear term)"
        out_rows.append({
            "final_term": term,
            "term_type": term_type,
            "source_family": family,
            "sensitive_features_used": ", ".join(sorted(hit)),
            "all_raw_sources": ", ".join(raw_sources),
            "coefficient": float(coefficient),
            "backend": backend,
            "support_count": row.get("support_count"),
            "status": "Review",
        })
    return pd.DataFrame(out_rows)


def render_fairness(
    sensitive_columns: list[str] | None = None,
    model: Any = None,
    X: pd.DataFrame | None = None,
    y: Any = None,
    *args,
    **kwargs,
) -> None:
    st.markdown("### Sensitive / Proxy Feature Review")
    sensitive_columns = list(sensitive_columns or [])

    if not sensitive_columns:
        st.info("No sensitive/proxy-review columns were provided.")
        return

    base = pd.DataFrame({
        "column": sensitive_columns,
        "role": ["Sensitive / proxy review"] * len(sensitive_columns),
        "required_action": ["Review usage in originals, HUG patterns, augmented/generated features, and group prediction rates"] * len(sensitive_columns),
    })

    tab1, tab2, tab3, tab4 = st.tabs([
        "Column review", "Group prediction rates", "Sensitive pattern flags", "Sensitive RPTE final-term flags",
    ])
    with tab1:
        st.dataframe(dataframe_for_display(base), width="stretch", hide_index=True)
    with tab2:
        rates = predicted_positive_rate_by_group(model, X, sensitive_columns)
        if rates.empty:
            st.info("Predicted positive-rate by group requires model, X, and sensitive/proxy columns present in X.")
        else:
            if "predicted_positive_rate" in rates.columns and "group_value" in rates.columns:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as _plt
                _plot_df = rates.copy()
                _plot_df["_label"] = _plot_df["column"].astype(str) + "=" + _plot_df["group_value"].astype(str)
                _plot_df = _plot_df.sort_values("predicted_positive_rate", ascending=True)
                _max_rate = float(_plot_df["predicted_positive_rate"].max()) if len(_plot_df) else 1.0
                _colors = [
                    "#E24B4A" if _max_rate > 0 and (abs(float(r) - _max_rate) / _max_rate) > 0.30 else "#378ADD"
                    for r in _plot_df["predicted_positive_rate"]
                ]
                _fig, _ax = _plt.subplots(figsize=(7, max(2.5, len(_plot_df) * 0.40)))
                _ax.barh(_plot_df["_label"], _plot_df["predicted_positive_rate"], color=_colors)
                _ax.set_xlabel("Predicted positive rate")
                _ax.set_title("Predicted positive rate by group")
                _ax.set_xlim(0, min(1.0, _max_rate * 1.25 + 0.05))
                _fig.tight_layout()
                st.pyplot(_fig)
                _plt.close(_fig)
                st.caption("Red bars differ from the highest group rate by > 30% — review for potential disparity.")
            st.dataframe(dataframe_for_display(rates), width="stretch", hide_index=True)
    with tab3:
        flags = sensitive_pattern_flags(model, sensitive_columns)
        if flags.empty:
            st.info("No mined HUG patterns containing sensitive/proxy source features were found, or pattern audit metadata is unavailable.")
        else:
            st.dataframe(dataframe_for_display(flags), width="stretch", hide_index=True)
    with tab4:
        st.caption(
            "RPTE can synthesize a split on a raw-feature pair that was never mined into a "
            "HUG pattern or augmented-pair transform, so it would not appear in the "
            "'Sensitive pattern flags' tab even if it uses a sensitive/proxy column directly. "
            "This checks the complete fitted RPTE final-LR representation instead of the mined-pattern list."
        )
        rpte_flags = sensitive_rpte_rule_flags(model, sensitive_columns)
        if rpte_flags.empty:
            st.info(
                "No non-zero RPTE final-LR terms reference the selected sensitive/proxy columns. "
                "This is also expected when the downstream estimator is not RPTE-based."
            )
        else:
            st.dataframe(dataframe_for_display(rpte_flags), width="stretch", hide_index=True)

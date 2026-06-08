"""Case-level prediction review component."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from hugiml.dashboard.display import dataframe_for_display


def _safe_predict(model: Any, row: pd.DataFrame) -> tuple[Any | None, str | None]:
    try:
        return model.predict(row)[0], None
    except Exception as exc:
        return None, repr(exc)


def _safe_predict_proba(model: Any, row: pd.DataFrame) -> tuple[float | None, str | None]:
    try:
        proba = model.predict_proba(row)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] > 1:
            return float(proba[0, 1]), None
        return float(np.asarray(proba).ravel()[0]), None
    except Exception as exc:
        return None, repr(exc)


def _is_missing_scalar(value: Any) -> bool:
    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        return False
    except Exception:
        return False


def _format_case_value(value: Any) -> Any:
    """Format a case value without accidentally blanking valid values.

    Avoid using a raw ``"" if pd.isna(v) else v`` expression because pd.isna
    can return array-like results for list/array/object values, and object
    values can also confuse Arrow display.
    """
    if _is_missing_scalar(value):
        return "<missing>"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(float(value)):
            return "<missing>"
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(map(str, value))
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return str(value)


def _row_to_feature_table(row: pd.DataFrame) -> pd.DataFrame:
    s = row.iloc[0]
    df = pd.DataFrame(
        {
            "feature": [str(x) for x in s.index],
            "value": [_format_case_value(v) for v in s.values],
            "missing": [_is_missing_scalar(v) for v in s.values],
        }
    )
    return df


def _top_case_values(row: pd.DataFrame, max_rows: int = 12) -> pd.DataFrame:
    """Show an evidence-oriented case summary without inventing explanations."""
    s = row.iloc[0]
    rows = []
    for name, value in s.items():
        missing = _is_missing_scalar(value)
        rows.append(
            {
                "feature": str(name),
                "value": _format_case_value(value),
                "missing": bool(missing),
                "evidence_type": "missing" if missing else ("numeric input" if isinstance(value, (int, float, np.integer, np.floating)) else "categorical/input"),
            }
        )
    df = pd.DataFrame(rows)
    df["_priority"] = df["evidence_type"].map({"missing": 0, "categorical/input": 1, "numeric input": 2}).fillna(3)
    return df.sort_values(["_priority", "feature"]).drop(columns=["_priority"]).head(max_rows)


def _get_pattern_frame(model: Any) -> pd.DataFrame:
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


def _activated_patterns(model: Any, row: pd.DataFrame) -> pd.DataFrame:
    """Best-effort extraction of active pattern indicators for a case.

    Different HUGIML versions expose different internals. This function tries
    public/semipublic transform APIs first and falls back gracefully.
    """
    pattern_df = _get_pattern_frame(model)

    # Public/semipublic transforms used by different versions.
    transform_candidates = [
        "transform_patterns",
        "transform_pattern_features",
        "_transform_patterns",
        "_transform_pattern_features",
        "transform",
    ]

    for name in transform_candidates:
        fn = getattr(model, name, None)
        if fn is None:
            continue
        try:
            Z = fn(row)
            if hasattr(Z, "toarray"):
                Z = Z.toarray()
            if isinstance(Z, pd.DataFrame):
                arr = Z.iloc[0].to_numpy()
                cols = list(map(str, Z.columns))
            else:
                arr = np.asarray(Z)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                arr = arr[0]
                cols = []
            active_idx = [i for i, v in enumerate(arr) if bool(v)]
            if not active_idx:
                return pd.DataFrame(columns=["rank", "pattern", "activation"])

            rows = []
            for i in active_idx[:25]:
                label = None
                if not pattern_df.empty and i < len(pattern_df):
                    label_col = next((c for c in pattern_df.columns if str(c).lower() in {"pattern", "label", "name", "feature"}), pattern_df.columns[0])
                    label = pattern_df.iloc[i][label_col]
                elif i < len(cols):
                    label = cols[i]
                else:
                    label = f"pattern_{i}"
                rows.append({"rank": int(i) + 1, "pattern": str(label), "activation": "active"})
            return pd.DataFrame(rows)
        except Exception:
            continue

    return pd.DataFrame()


def render_prediction(model: Any = None, X: pd.DataFrame | None = None, case_ids: Any = None, *args, **kwargs) -> None:
    st.subheader("Case Review")
    st.markdown(
        """
        <div class="hugiml-section-note">
          <p>This page provides case-level traceability: prediction, probability, source input values,
          and active HUG pattern indicators when exposed by the installed model version.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None or X is None or len(X) == 0:
        st.info("Train/load a model and data to inspect individual predictions.")
        return

    max_idx = max(len(X) - 1, 0)
    idx = int(st.number_input("Row index", min_value=0, max_value=max_idx, value=0, step=1))

    try:
        case_id = str(case_ids.iloc[idx] if hasattr(case_ids, "iloc") else case_ids[idx])
    except Exception:
        case_id = f"CASE-{idx+1:06d}"

    row = X.iloc[[idx]]

    pred, pred_error = _safe_predict(model, row)
    proba, proba_error = _safe_predict_proba(model, row)

    c1, c2, c3 = st.columns(3)
    c1.metric("Case ID", case_id)
    c2.metric("Predicted class", str(pred) if pred is not None else "N/A")
    c3.metric("Predicted probability", f"{proba:.4f}" if proba is not None else "N/A")

    if pred_error or proba_error:
        with st.expander("Prediction diagnostics", expanded=False):
            if pred_error:
                st.warning(f"predict failed: {pred_error}")
            if proba_error:
                st.warning(f"predict_proba failed: {proba_error}")

    tab1, tab2, tab3 = st.tabs(["Case summary", "All input values", "Pattern evidence"])

    with tab1:
        st.markdown("#### High-level case inputs")
        st.caption("Displayed from the row values only; no feature attribution is inferred.")
        st.dataframe(dataframe_for_display(_top_case_values(row)), width="stretch", hide_index=True)

    with tab2:
        st.markdown("#### Full case input vector")
        st.dataframe(dataframe_for_display(_row_to_feature_table(row)), width="stretch", hide_index=True)

    with tab3:
        active = _activated_patterns(model, row)
        if active.empty:
            st.info(
                "Active-pattern extraction is not exposed by this installed model version. "
                "Use the Pattern Inventory page for global pattern evidence and the Full case input vector "
                "above to manually review whether key pattern conditions apply."
            )
            pattern_df = _get_pattern_frame(model)
            if not pattern_df.empty:
                st.markdown("#### Available global pattern inventory")
                st.dataframe(dataframe_for_display(pattern_df.head(25)), width="stretch", hide_index=True)
        else:
            st.markdown("#### Active HUG pattern indicators for this case")
            st.dataframe(dataframe_for_display(active), width="stretch", hide_index=True)

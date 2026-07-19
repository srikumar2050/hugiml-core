"""Case-level prediction review component."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from hugiml.dashboard.components.patterns import format_conjunction
from hugiml.dashboard.display import dataframe_for_display


def _downstream_matrix_for_row(model: Any, row: pd.DataFrame) -> Any | None:
    """The HUGIML-mined feature matrix for one row -- what the fitted
    downstream estimator (including RPTE) actually sees -- built the
    same way HUGIMLClassifierNative.predict_proba() itself does: mine
    the HUG pattern matrix for the (possibly pre-binned) row, combine
    it with the raw-input downstream feature block, then apply the
    strict top-K budget transform if configured. self.model_ wraps
    only the downstream estimator (no separate sklearn Pipeline mining
    step to slice off), so this has to go through the classifier's own
    prediction-time feature construction directly.
    """
    if getattr(model, "_is_constant_prior_fallback_active", lambda: False)():
        return None
    try:
        X_for_downstream = row
        X_for_pattern = row
        if getattr(model, "adaptive_binning", False) and getattr(model, "_bin_edges_", None):
            X_for_pattern = model._prebin_for_predict(row)
        Z = model._build_test_hup(X_for_pattern)
        Xd = model._make_downstream_features(X_for_downstream, Z, fit=False)
        return model._apply_strict_topk_budget_transform(Xd)
    except Exception:
        return None


def _rpte_sub_estimators(model: Any) -> list[tuple[Any, Any | None]]:
    """(sub_estimator, class_label) pairs for every RPTE-capable fitted
    downstream estimator -- one pair for a plain binary RPTE fit, one
    per class for a OneVsRestClassifier-wrapped multiclass fit (see
    HUGIMLClassifierNative.rpte_rule_table's docstring for why the
    unwrapping has to handle both shapes)."""
    pipeline = getattr(model, "model_", None)
    estimator = pipeline.named_steps.get("clf", pipeline) if hasattr(pipeline, "named_steps") else pipeline
    if estimator is None:
        return []
    if hasattr(estimator, "estimators_") and hasattr(estimator, "classes_"):
        classes = list(getattr(estimator, "classes_", []))
        subs = list(getattr(estimator, "estimators_", []))
        return list(zip(subs, classes)) if len(classes) == len(subs) else [(s, None) for s in subs]
    if hasattr(estimator, "unified_rule_table"):
        return [(estimator, None)]
    return []


def _rpte_active_rules_for_case(model: Any, row: pd.DataFrame) -> pd.DataFrame:
    """Which specific leaf (rule) each RPTE sub-estimator's fitted trees
    assign this case to, joined against the same estimator's own
    unified_rule_table() to recover that leaf's full conjunction and
    coefficient -- i.e. the actual rule this case's prediction rests on,
    not just the global rule inventory. Handles both RPTE backends: the
    native bounded-lookahead engine and the sklearn-DecisionTreeClassifier
    -based sequential default (which "adaptive" mode picks whenever a
    single raw feature already carries enough marginal signal -- the
    common case in practice).

    Best-effort: returns an empty frame (never raises) if the fitted
    downstream estimator isn't RPTE-based, or its internal feature
    extractor doesn't expose leaf assignment in a recognized shape.
    """
    Xd = _downstream_matrix_for_row(model, row)
    if Xd is None:
        return pd.DataFrame()
    sub_estimators = _rpte_sub_estimators(model)
    if not sub_estimators:
        return pd.DataFrame()
    Xd_dense = Xd.toarray() if hasattr(Xd, "toarray") else np.asarray(Xd)

    records = []
    for sub, class_label in sub_estimators:
        fe = getattr(sub, "fe_", None)
        if fe is None:
            continue
        try:
            rule_rows = {
                (r.get("tree_index"), r.get("leaf_index")): r for r in sub.unified_rule_table()
            }
        except Exception:
            continue

        default_fe = getattr(fe, "_default_fe", None)
        if default_fe is not None:
            # Sequential backend: ordinary sklearn trees, each restricted
            # to its own reserved column subset (tree_columns_).
            for tree_index, (tree, columns) in enumerate(
                zip(getattr(default_fe, "trees_", []), getattr(default_fe, "tree_columns_", []))
            ):
                try:
                    leaf_id = int(tree.apply(Xd_dense[:, columns])[0])
                except Exception:
                    continue
                rule = rule_rows.get((tree_index, leaf_id))
                if rule is None:
                    continue
                records.append(_rpte_active_rule_record(class_label, tree_index, leaf_id, rule))
            continue

        if not hasattr(fe, "trees_") or not hasattr(fe, "tree_leaf_ids_"):
            continue
        # Bounded-lookahead backend: native trees, applied via the same
        # compiled kernel used at fit time.
        try:
            from hugiml import rpte_bounded_lookahead_leafwise as _rbl
        except Exception:
            continue
        for tree_index, tree_dict in enumerate(fe.trees_):
            try:
                leaf_ids = _rbl._native.rpte_apply_tree(np.asarray(Xd_dense, dtype=np.float64), tree_dict)
                leaf_id = int(leaf_ids[0])
            except Exception:
                continue
            rule = rule_rows.get((tree_index, leaf_id))
            if rule is None:
                continue
            records.append(_rpte_active_rule_record(class_label, tree_index, leaf_id, rule))
    return pd.DataFrame(records)


def _rpte_active_rule_record(class_label: Any, tree_index: int, leaf_id: int, rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "class": class_label if class_label is not None else rule.get("class"),
        "tree": tree_index,
        "leaf": leaf_id,
        "backend": rule.get("backend"),
        "conjunction": format_conjunction(rule.get("conditions") or []),
        "coefficient": rule.get("final_logistic_coefficient"),
        "vs_tree_baseline": rule.get("centered_tree_contribution"),
    }


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
          active HUG pattern indicators when exposed by the installed model version, and -- when the
          downstream estimator is RPTE -- the exact leaf conjunction each tree fired for this case.</p>
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

    rpte_active = _rpte_active_rules_for_case(model, row)
    tab_labels = ["Case summary", "All input values", "Pattern evidence"]
    if not rpte_active.empty:
        tab_labels.append("RPTE rule (this case)")
    tabs = st.tabs(tab_labels)
    tab1, tab2, tab3 = tabs[0], tabs[1], tabs[2]

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

    if not rpte_active.empty:
        with tabs[3]:
            st.markdown("#### RPTE rule conjunction fired for this case")
            st.caption(
                "One row per tree in the boosted RPTE ensemble: the specific leaf this case's "
                "input reached, its full conjunction of split conditions, and that leaf's "
                "contribution to the fitted logistic model. Sum the coefficient column (plus "
                "the model intercept) to reproduce this case's log-odds contribution from RPTE."
            )
            st.dataframe(dataframe_for_display(rpte_active), width="stretch", hide_index=True)
            total = rpte_active["coefficient"].sum()
            st.metric("Sum of fired-rule coefficients", f"{total:.4f}")

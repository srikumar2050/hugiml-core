"""Feature-family audit component."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from hugiml.dashboard.components.patterns import (
    _RPTE_LEAF_DIRECT_REPRESENTATIONS,
    _direct_family_counts,
    _get_rpte_feature_flow_audit,
    _get_rpte_final_term_rows,
    _get_rpte_rule_rows,
    _rpte_rules_to_frame,
)
from hugiml.dashboard.components.rpte_governance import (
    rpte_direct_source_terms_frame,
    rpte_raw_input_lineage_frame,
    rpte_representation_flow_frame,
    rpte_source_inventory_frame,
    rpte_split_usage_frame,
)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except Exception:
        return []


def _infer_source_features_from_label(label: str) -> list[str]:
    import re
    s = str(label)
    matches = re.findall(r"(?:^|,\s*)([^=,\[\]]+?)\s*=", s)
    return [m.strip() for m in matches if m.strip()]


def _string_list(value: Any) -> str:
    """Render list-like audit values as compact comma-separated text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return ", ".join(str(v) for v in list(value))
    return str(value)


def _safe_bool(value: Any) -> bool:
    """Coerce scalar or one-element array flags used by model metadata."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        if value.size == 1:
            return bool(value.reshape(-1)[0])
        return bool(np.any(value))
    try:
        return bool(value)
    except Exception:
        return False


def original_feature_audit(
    model: Any,
    X: pd.DataFrame | None = None,
    sensitive_columns: list[str] | None = None,
    excluded_columns: list[str] | None = None,
    id_column: str | None = None,
) -> pd.DataFrame:
    sensitive = set(map(str, _as_list(sensitive_columns)))
    excluded = set(map(str, _as_list(excluded_columns)))
    id_col = str(id_column) if id_column else None
    names = _as_list(getattr(model, "feature_names_in_", None)) or _as_list(getattr(model, "origColumns", None))
    if not names and X is not None:
        names = list(X.columns)

    rows = []
    for name in map(str, names):
        used = name not in excluded and name != id_col
        missing_pct = np.nan
        dtype = ""
        if X is not None and name in X.columns:
            missing_pct = float(pd.isna(X[name]).mean() * 100.0)
            dtype = str(X[name].dtype)
        role = "ID" if name == id_col else ("Sensitive/review" if name in sensitive else ("Excluded" if name in excluded else "Model feature"))
        risk = "Review" if name in sensitive and used else ("Excluded" if not used else "OK")
        rows.append({
            "feature": name,
            "family": "Original",
            "dtype": dtype,
            "used_in_model": bool(used),
            "role": role,
            "missing_pct": missing_pct,
            "status": risk,
        })
    return pd.DataFrame(rows)


def pattern_feature_audit(model: Any, sensitive_columns: list[str] | None = None) -> pd.DataFrame:
    sensitive = set(map(str, _as_list(sensitive_columns)))
    rows = []

    try:
        if hasattr(model, "get_pattern_info"):
            info = model.get_pattern_info()
            if isinstance(info, pd.DataFrame) and not info.empty:
                df = info.copy()
                label_col = next((c for c in df.columns if str(c).lower() in {"pattern", "label", "feature", "name"}), None)
                if label_col:
                    for i, row in df.iterrows():
                        label = str(row[label_col])
                        src = _infer_source_features_from_label(label)
                        rows.append({
                            "rank": int(i) + 1,
                            "pattern": label,
                            "source_features": ", ".join(src),
                            "order": len(src) if src else np.nan,
                            "pattern_origin": row.get("pattern_origin", "standard"),
                            "survivor_led": _safe_bool(row.get("survivor_led", False)),
                            "survivor_features": _string_list(row.get("survivor_features", "")),
                            "survivor_feature_count": row.get("survivor_feature_count", 0),
                            "survivor_min_marginal_ig": row.get("survivor_min_marginal_ig", np.nan),
                            "survivor_max_interaction_score": row.get("survivor_max_interaction_score", np.nan),
                            "survivor_best_partners": _string_list(row.get("survivor_best_partners", "")),
                            "status": "Review" if set(src).intersection(sensitive) else "OK",
                        })
                    return pd.DataFrame(rows)
    except Exception:
        pass

    labels = []
    for attr in ("pattern_labels_", "_pattern_labels_", "patterns_", "raw_patterns_"):
        try:
            labels = getattr(model, attr, None)
            if labels is not None:
                break
        except Exception:
            labels = None

    for i, label in enumerate(_as_list(labels), start=1):
        label = str(label)
        src = _infer_source_features_from_label(label)
        rows.append({
            "rank": i,
            "pattern": label,
            "source_features": ", ".join(src),
            "order": len(src) if src else np.nan,
            "status": "Review" if set(src).intersection(sensitive) else "OK",
        })
    return pd.DataFrame(rows)


def augmented_feature_audit(model: Any, sensitive_columns: list[str] | None = None) -> pd.DataFrame:
    sensitive = set(map(str, _as_list(sensitive_columns)))
    specs = _as_list(getattr(model, "augmented_pair_transforms_", []))
    rows = []
    for i, spec in enumerate(specs, start=1):
        if not isinstance(spec, dict):
            continue
        inputs = [str(x) for x in spec.get("inputs", [])]
        rows.append({
            "rank": i,
            "name": str(spec.get("name", f"augmented_{i}")),
            "operation": str(spec.get("operation", "")),
            "source_features": ", ".join(inputs),
            "formula": str(spec.get("formula", spec.get("raw_formula", ""))),
            "missing_policy": str(spec.get("pair_missing_policy", "")),
            "eligible_rate": spec.get("eligible_rate", np.nan),
            "missing_pair_rate": spec.get("missing_pair_rate", np.nan),
            "transform_ig": spec.get("transform_ig", np.nan),
            "status": "Review" if set(inputs).intersection(sensitive) else "OK",
        })
    return pd.DataFrame(rows)


def feature_family_summary(model: Any, X: pd.DataFrame | None = None) -> pd.DataFrame:
    meta = getattr(model, "fit_metadata_", None)
    counts = getattr(meta, "downstream_feature_counts", {}) if meta is not None else {}
    try:
        composition = model.get_model_composition()
    except Exception:
        composition = {}
    if isinstance(composition, dict) and isinstance(composition.get("downstream_feature_counts"), dict):
        counts = composition["downstream_feature_counts"]
    if not isinstance(counts, dict):
        counts = {}

    flow = _get_rpte_feature_flow_audit(model)
    representation = flow.get("final_representation")
    has_leaf_terms = representation == "rpte_leaf_rules" or representation in _RPTE_LEAF_DIRECT_REPRESENTATIONS
    raw_fallback = representation == "hugiml_source_features"

    p = len(_as_list(getattr(model, "feature_names_in_", None))) or (X.shape[1] if X is not None else 0)
    n_patterns = len(_as_list(getattr(model, "patterns_", []))) or len(_as_list(getattr(model, "raw_patterns_", [])))
    n_aug = len(_as_list(getattr(model, "augmented_pair_transforms_", [])))

    if has_leaf_terms:
        if flow.get("patterns_above_order_two_are_direct_only"):
            role = (
                "RPTE source family; patterns above order two are direct-only, "
                "while originals, generated pairs, and lower-order patterns may feed "
                "leaf indicators"
            )
        else:
            role = (
                "RPTE source family; split-used columns feed leaf indicators and "
                "remaining columns may stay direct"
            )
        direct = representation in _RPTE_LEAF_DIRECT_REPRESENTATIONS
    elif raw_fallback:
        role = "Direct final-LR source family in RPTE emergency fallback"
        direct = True
    else:
        role = "Downstream estimator feature family"
        direct = True

    rows = [
        {
            "feature_family": "Original features",
            "count": int(counts.get("original", p)),
            "used_as_direct_final_lr_family": direct,
            "rpte_role": role,
            "evidence": "Raw input columns after exclusions",
        },
        {
            "feature_family": "HUG pattern features",
            "count": int(counts.get("pattern", n_patterns)),
            "used_as_direct_final_lr_family": direct,
            "rpte_role": role,
            "evidence": "Human-readable mined patterns",
        },
        {
            "feature_family": "Augmented/generated features",
            "count": int(counts.get("augmented_pair", n_aug)),
            "used_as_direct_final_lr_family": direct,
            "rpte_role": role,
            "evidence": "Generated transforms with source-feature mapping",
        },
    ]
    total = sum(int(r["count"]) for r in rows)
    rows.append({
        "feature_family": "Source-input total" if flow else "Displayed total",
        "count": total,
        "used_as_direct_final_lr_family": direct,
        "rpte_role": role if flow else "Downstream feature total",
        "evidence": "Sum of original, pattern, and augmented source-family counts",
    })
    if has_leaf_terms:
        rows.append({
            "feature_family": "RPTE leaf-rule terms",
            "count": int(flow.get("leaf_rule_count", flow.get("final_term_count", 0)) or 0),
            "used_as_direct_final_lr_family": True,
            "rpte_role": "Actual final LR terms: one-hot root-to-leaf conjunction indicators",
            "evidence": "Fitted RPTE leaf matrix aligned to logistic-regression coefficients",
        })
        direct_counts = _direct_family_counts(flow)
        for family, label in (
            ("original", "Direct original terms"),
            ("pattern", "Direct HUG pattern terms"),
            ("augmented_pair", "Direct augmented-pair terms"),
        ):
            rows.append({
                "feature_family": label,
                "count": int(direct_counts.get(family, 0) or 0),
                "used_as_direct_final_lr_family": True,
                "rpte_role": "Source columns not selected in accepted RPTE splits; carried directly into LR",
                "evidence": "Fitted direct-input index layout aligned to the LR coefficient block after leaves",
            })
    elif representation == "constant":
        rows.append({
            "feature_family": "Constant fallback term",
            "count": int(flow.get("final_term_count", 1) or 1),
            "used_as_direct_final_lr_family": True,
            "rpte_role": "Actual final LR representation",
            "evidence": "Class-prior fallback after tree and raw-source fallbacks were unavailable",
        })
    return pd.DataFrame(rows)


def _render_rpte_feature_family_audit(
    model: Any,
    X: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    """Render the fitted leaf-plus-direct-source RPTE representation by role."""
    flow = _get_rpte_feature_flow_audit(model)
    flow_frame = rpte_representation_flow_frame(model, X)
    final_terms = _rpte_rules_to_frame(_get_rpte_final_term_rows(model, include_zero_direct=True))
    leaf_terms = (
        final_terms.loc[final_terms["is_leaf_term"]].copy()
        if not final_terms.empty and "is_leaf_term" in final_terms.columns
        else pd.DataFrame()
    )
    direct_terms = rpte_direct_source_terms_frame(model, include_zero=True)
    source_inventory = rpte_source_inventory_frame(model)
    split_usage = rpte_split_usage_frame(model)
    raw_lineage = rpte_raw_input_lineage_frame(model, X)

    st.info(str(flow.get("statement") or "RPTE representation evidence is available."))
    st.caption(
        "The final LR is partitioned into RPTE leaf indicators and direct source terms: source "
        "columns not selected by accepted RPTE splits. The direct block is shown only for originals, HUG patterns, "
        "and augmented pairs that actually enter the final LR."
    )

    family_counts = (
        direct_terms["family"].value_counts().to_dict()
        if not direct_terms.empty and "family" in direct_terms.columns
        else {}
    )
    metrics = st.columns(5)
    metrics[0].metric("RPTE leaf terms", f"{len(leaf_terms):,}")
    metrics[1].metric("Direct original features", f"{int(family_counts.get('original', 0)):,}")
    metrics[2].metric("Direct HUG patterns", f"{int(family_counts.get('pattern', 0)):,}")
    metrics[3].metric("Direct augmented pairs", f"{int(family_counts.get('augmented_pair', 0)):,}")
    metrics[4].metric(
        "Final LR terms",
        f"{int(flow.get('final_term_count', len(final_terms)) or len(final_terms)):,}",
    )

    tabs = st.tabs(
        [
            "Final LR composition",
            "Direct original features",
            "Direct HUG patterns",
            "Direct augmented pairs",
            "RPTE split usage",
            "Raw-input lineage",
        ]
    )
    with tabs[0]:
        st.markdown("#### Representation flow")
        st.dataframe(flow_frame, width="stretch", hide_index=True)
        st.markdown("#### Final LR terms")
        columns = [
            "class", "term_type", "rule_preview", "source_display_name", "source_column",
            "raw_sources", "coefficient", "odds_multiplier", "tree", "leaf", "backend",
        ]
        st.dataframe(
            final_terms[[c for c in columns if c in final_terms.columns]],
            width="stretch",
            hide_index=True,
        )

    family_tabs = [
        (tabs[1], "original", "direct original features"),
        (tabs[2], "pattern", "direct HUG patterns"),
        (tabs[3], "augmented_pair", "direct augmented pairs"),
    ]
    for tab, family, label in family_tabs:
        with tab:
            family_frame = (
                direct_terms.loc[direct_terms["family"].eq(family)].copy()
                if not direct_terms.empty and "family" in direct_terms.columns
                else pd.DataFrame()
            )
            st.caption(
                f"Only {label} not selected by accepted RPTE splits are listed. These columns enter "
                "the final LR directly after the leaf-indicator block."
            )
            if family_frame.empty:
                st.info(f"No {label} are present in the final LR.")
            else:
                display_columns = [
                    "class", "display_name", "source_column", "raw_sources", "coefficient",
                    "odds_multiplier", "nonzero", "final_term_type",
                ]
                st.dataframe(
                    family_frame[[c for c in display_columns if c in family_frame.columns]],
                    width="stretch",
                    hide_index=True,
                )

    with tabs[4]:
        st.caption(
            "This view lists source columns that occur in accepted RPTE split conditions. "
            "They are represented through leaf indicators rather than repeated as direct LR terms."
        )
        if split_usage.empty:
            st.info("No accepted RPTE split usage is available for this fitted model.")
        else:
            st.dataframe(split_usage, width="stretch", hide_index=True)
        with st.expander("Complete source-column role inventory", expanded=False):
            st.dataframe(source_inventory, width="stretch", hide_index=True)

    with tabs[5]:
        st.caption(
            "Raw inputs may feed selected original columns, mined patterns, augmented pairs, "
            "RPTE split conditions, and direct source terms."
        )
        st.dataframe(raw_lineage, width="stretch", hide_index=True)

    return {
        "flow": flow_frame,
        "final_terms": final_terms,
        "leaf_terms": leaf_terms,
        "direct_terms": direct_terms,
        "source_inventory": source_inventory,
        "split_usage": split_usage,
        "raw_lineage": raw_lineage,
    }


def _render_standard_feature_family_audit(
    model: Any,
    X: pd.DataFrame | None = None,
    sensitive_columns: list[str] | None = None,
    excluded_columns: list[str] | None = None,
    id_column: str | None = None,
) -> dict[str, pd.DataFrame]:
    st.markdown("### Feature Family Audit")
    flow = _get_rpte_feature_flow_audit(model)
    if flow:
        st.info(str(flow.get("statement") or "RPTE feature-flow audit is available."))
        st.caption(
            "Source-family counts describe what RPTE could split on. Final-term counts describe "
            "what the fitted logistic layer actually uses for prediction."
        )
    else:
        st.caption("Audits the downstream representation using explicit feature-family evidence, not opaque scores.")

    summary = feature_family_summary(model, X)
    originals = original_feature_audit(model, X, sensitive_columns, excluded_columns, id_column)
    patterns = pattern_feature_audit(model, sensitive_columns)
    augmented = augmented_feature_audit(model, sensitive_columns)
    rpte_rules = _rpte_rules_to_frame(_get_rpte_rule_rows(model))

    if flow:
        representation = flow.get("final_representation")
        direct_family = representation == "hugiml_source_features"
        family_role = (
            "RPTE tree-construction input; not a direct final-LR term"
            if representation == "rpte_leaf_rules"
            else "Direct final-LR source family in RPTE emergency fallback"
            if direct_family
            else "RPTE fallback input"
        )
        for frame in (originals, patterns, augmented):
            if not frame.empty:
                frame.insert(0, "used_as_direct_final_lr_family", direct_family)
                frame.insert(1, "rpte_role", family_role)

    cols = st.columns(5 if flow else 4)
    def _count(name: str) -> int:
        s = summary.loc[summary["feature_family"].eq(name), "count"]
        return int(s.iloc[0]) if len(s) else 0

    cols[0].metric("Original features", _count("Original features"))
    cols[1].metric("HUG patterns", _count("HUG pattern features"))
    cols[2].metric("Augmented features", _count("Augmented/generated features"))
    review_flags = int((originals.get("status", pd.Series(dtype=str)).eq("Review")).sum()) + int((patterns.get("status", pd.Series(dtype=str)).eq("Review")).sum()) + int((augmented.get("status", pd.Series(dtype=str)).eq("Review")).sum())
    if flow:
        cols[3].metric("Final LR terms", int(flow.get("final_term_count", len(rpte_rules)) or 0))
        cols[4].metric("Review flags", review_flags)
    else:
        cols[3].metric("Review flags", review_flags)

    st.dataframe(summary, width="stretch", hide_index=True)

    # Signed coefficient waterfall from feature_importances() when available;
    # falls back to rpte_rule_table() when the downstream estimator is RPTE
    # (feature_importances() requires a coef_-exposing estimator and raises
    # for RPTE -- without this fallback the waterfall silently disappears
    # for exactly the models "performance_ho", the default tuning grid, is
    # most likely to produce).
    try:
        _fi = model.feature_importances() if hasattr(model, "feature_importances") and callable(getattr(model, "feature_importances")) else None
    except Exception:
        _fi = None
    _fi_is_rpte_fallback = False
    if (_fi is None or _fi.empty) and hasattr(model, "rpte_rule_table") and callable(getattr(model, "rpte_rule_table")):
        try:
            _rpte_rows = model.rpte_rule_table()
        except Exception:
            _rpte_rows = []
        if _rpte_rows:
            _fi_is_rpte_fallback = True
            _ranked = sorted(
                _rpte_rows, key=lambda r: abs(r.get("final_logistic_coefficient") or 0.0), reverse=True
            )[:25]
            _fi_rows = []
            for _row in _ranked:
                _conditions = _row.get("conditions") or []
                _terms = [c.get("raw_condition") or c.get("downstream_condition") or "?" for c in _conditions]
                _fi_rows.append({
                    "pattern": " AND ".join(str(t) for t in _terms) if _terms else "(linear term)",
                    "coefficient": _row.get("final_logistic_coefficient"),
                    "feature_type": "rpte_rule",
                })
            _fi = pd.DataFrame(_fi_rows)
    if _fi is not None and not _fi.empty and "coefficient" in _fi.columns:
        _wf = _fi.copy()
        _wf["coefficient"] = pd.to_numeric(_wf["coefficient"], errors="coerce")
        _wf = _wf.dropna(subset=["coefficient"]).head(25).sort_values("coefficient")
        if not _wf.empty:
            st.markdown(
                "#### RPTE rule coefficient waterfall (signed)"
                if _fi_is_rpte_fallback
                else "#### Feature coefficient waterfall (signed)"
            )
            st.caption(
                "Each rule's coefficient is its own leaf's contribution to the fitted "
                "downstream model (see rpte_rule_table())."
                if _fi_is_rpte_fallback
                else "Positive coefficients (right) increase the model score; negative (left) decrease it. "
                "Color encodes the feature family: original, HUG pattern, or augmented/generated."
            )
            _family_colors = {
                "original": "#378ADD",
                "pattern": "#534AB7",
                "augmented_pair": "#1D9E75",
                "rpte_rule": "#C77D2E",
                "unknown": "#888780",
            }
            _family_labels = {
                "original": "Original",
                "pattern": "HUG pattern",
                "augmented_pair": "Augmented/generated",
                "rpte_rule": "RPTE rule",
                "unknown": "Unknown",
            }

            def _normalise_feature_family(value: Any, label: str = "") -> str:
                raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
                if raw == "rpte_rule":
                    return "rpte_rule"
                if raw in {"original", "original_feature", "original_features", "raw", "input", "input_feature"}:
                    return "original"
                if raw in {"pattern", "patterns", "hug_pattern", "hug_patterns", "pattern_feature", "pattern_features"}:
                    return "pattern"
                if raw in {
                    "augmented",
                    "augmented_pair",
                    "augmented_pair_feature",
                    "augmented_pair_features",
                    "augmented_generated",
                    "generated",
                    "generated_feature",
                    "interaction",
                    "interaction_feature",
                }:
                    return "augmented_pair"
                text = str(label or value or "").lower()
                if "augmented_pair" in text or text.startswith("augmented") or "*" in text or " abs(" in text or text.startswith("abs("):
                    return "augmented_pair"
                if "[" in text and "]" in text or "=" in text:
                    return "pattern"
                return "unknown"

            _type_col = "feature_type" if "feature_type" in _wf.columns else None
            _name_col = "display_name" if "display_name" in _wf.columns else "pattern" if "pattern" in _wf.columns else "feature"
            _wf["_feature_family"] = [
                _normalise_feature_family(_wf[_type_col].iloc[i] if _type_col else "", str(_wf[_name_col].iloc[i]))
                for i in range(len(_wf))
            ]
            _colors_wf = [_family_colors.get(str(v), _family_colors["unknown"]) for v in _wf["_feature_family"]]

            _fig_wf, _ax_wf = plt.subplots(figsize=(7, max(3.0, len(_wf) * 0.32)))
            _ax_wf.barh(_wf[_name_col].astype(str), _wf["coefficient"], color=_colors_wf)
            _ax_wf.axvline(0, color="#888780", linewidth=0.8)
            _ax_wf.set_xlabel("Coefficient (log-odds)")
            _ax_wf.set_title("Signed feature coefficients")
            _ax_wf.invert_yaxis()
            _present_families = [fam for fam in ("rpte_rule", "original", "pattern", "augmented_pair", "unknown") if fam in set(_wf["_feature_family"])]
            if _present_families:
                _handles = [
                    Line2D(
                        [0],
                        [0],
                        marker="s",
                        linestyle="",
                        color="none",
                        markerfacecolor=_family_colors[fam],
                        markeredgecolor="none",
                        markersize=8,
                        label=_family_labels[fam],
                    )
                    for fam in _present_families
                ]
                _ax_wf.legend(handles=_handles, fontsize=8, loc="lower right", title="Feature family")
            _fig_wf.tight_layout()
            st.pyplot(_fig_wf)
            plt.close(_fig_wf)

    tab_labels = ["Originals", "HUG patterns", "Augmented/generated"]
    if not rpte_rules.empty:
        tab_labels.insert(0, "Final RPTE LR terms")
    tabs = st.tabs(tab_labels)
    offset = 1 if not rpte_rules.empty else 0
    if not rpte_rules.empty:
        with tabs[0]:
            st.caption(
                "These are the fitted prediction terms. Source-family inventories in the other "
                "tabs are kept separate because they describe tree-construction inputs."
            )
            cols = [
                "rule_id", "effect", "coefficient", "odds_multiplier", "support_rate",
                "n_conditions", "rule_preview", "raw_sources", "class", "tree", "leaf", "backend",
            ]
            st.dataframe(rpte_rules[[c for c in cols if c in rpte_rules.columns]], width="stretch", hide_index=True)
    tab1, tab2, tab3 = tabs[offset], tabs[offset + 1], tabs[offset + 2]
    with tab1:
        st.dataframe(originals, width="stretch", hide_index=True)
    with tab2:
        if patterns.empty:
            st.info("Pattern metadata is not available for this model/version.")
        else:
            st.dataframe(patterns, width="stretch", hide_index=True)
    with tab3:
        if augmented.empty:
            st.info("No augmented/generated feature metadata was found for this model.")
        else:
            st.dataframe(augmented, width="stretch", hide_index=True)

    return {
        "summary": summary,
        "originals": originals,
        "patterns": patterns,
        "augmented": augmented,
        "rpte_rules": rpte_rules,
    }



def _render_rpte_tree_feature_family_audit(
    model: Any,
    X: pd.DataFrame | None = None,
    sensitive_columns: list[str] | None = None,
    excluded_columns: list[str] | None = None,
    id_column: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Render feature-family evidence for RPTE representations containing leaves."""
    return _render_rpte_feature_family_audit(model, X)

def render_feature_family_audit(
    model: Any,
    X: pd.DataFrame | None = None,
    sensitive_columns: list[str] | None = None,
    excluded_columns: list[str] | None = None,
    id_column: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Render representation evidence without mixing RPTE stages."""
    flow = _get_rpte_feature_flow_audit(model)
    if flow.get("final_representation") in {
        "rpte_leaf_rules",
        "rpte_leaves_plus_direct_terms",
    }:
        return _render_rpte_tree_feature_family_audit(
            model,
            X,
            sensitive_columns=sensitive_columns,
            excluded_columns=excluded_columns,
            id_column=id_column,
        )
    return _render_standard_feature_family_audit(
        model,
        X,
        sensitive_columns=sensitive_columns,
        excluded_columns=excluded_columns,
        id_column=id_column,
    )

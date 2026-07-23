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
from hugiml.rpte_interpretability import rpte_rule_tree_sections

_DIRECT_SOURCE_BACKENDS = {"direct_hugiml_feature"}
_RPTE_LEAF_DIRECT_REPRESENTATIONS = {"rpte_leaves_plus_direct_terms"}


def _is_direct_source_row(row: dict[str, Any]) -> bool:
    return (
        str(row.get("term_role") or "") == "direct_source_term"
        or str(row.get("backend") or "") in _DIRECT_SOURCE_BACKENDS
    )


def _direct_input_indices(sub: Any) -> np.ndarray:
    return np.asarray(getattr(sub, "direct_input_indices_", []), dtype=np.int64)


def _direct_input_coefficients(sub: Any) -> np.ndarray:
    method = getattr(sub, "direct_input_coefficients", None)
    try:
        return np.asarray(method(), dtype=float) if callable(method) else np.asarray([], dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def _direct_family_counts(flow: dict[str, Any]) -> dict[str, int]:
    value = flow.get("direct_source_family_counts", {})
    return dict(value) if isinstance(value, dict) else {}


def _direct_source_count(flow: dict[str, Any], fallback: int = 0) -> int:
    return int(flow.get("direct_source_term_count", fallback) or 0)


def _get_rpte_rule_rows(model: Any) -> list[dict[str, Any]]:
    """Return fitted RPTE prediction rows, or ``[]`` for non-RPTE models."""
    fn = getattr(model, "rpte_rule_table", None)
    if fn is None:
        return []
    try:
        rows = fn()
    except Exception:
        return []
    return list(rows) if rows else []


def _rpte_sub_estimators_for_audit(model: Any) -> list[tuple[Any, Any]]:
    """Return fitted RPTE sub-estimators paired with their governed class.

    HUGIML wraps the downstream estimator in a pipeline and may additionally
    wrap RPTE in OneVsRestClassifier.  The outer class labels are used for OVR
    because each binary sub-estimator otherwise reports only its local 0/1
    classes.
    """
    if model is None:
        return []
    estimator = getattr(model, "model_", model)
    named_steps = getattr(estimator, "named_steps", None)
    if named_steps is not None:
        try:
            estimator = named_steps.get("clf", estimator)
        except Exception:
            pass

    raw_sub_estimators = getattr(estimator, "estimators_", None)
    sub_estimators = list(raw_sub_estimators) if raw_sub_estimators is not None else []
    if sub_estimators:
        raw_labels = getattr(estimator, "classes_", None)
        classes = list(raw_labels) if raw_labels is not None else []
        labels = [classes[-1]] if len(sub_estimators) == 1 and len(classes) == 2 else classes
        return [
            (sub, labels[index] if index < len(labels) else index)
            for index, sub in enumerate(sub_estimators)
            if hasattr(sub, "unified_rule_table") or hasattr(sub, "logistic_")
        ]
    if hasattr(estimator, "unified_rule_table") or hasattr(estimator, "logistic_"):
        raw_classes = getattr(estimator, "classes_", None)
        classes = list(raw_classes) if raw_classes is not None else []
        label = classes[-1] if classes else None
        return [(estimator, label)]
    return []


def _get_rpte_alias_rows(model: Any) -> list[dict[str, Any]]:
    """Return fitted leaf-pattern canonicalization records."""
    fn = getattr(model, "rpte_representation_aliases", None)
    if callable(fn):
        try:
            return [dict(row) for row in (fn() or [])]
        except Exception:
            pass
    rows: list[dict[str, Any]] = []
    for sub, class_label in _rpte_sub_estimators_for_audit(model):
        method = getattr(sub, "representation_alias_table", None)
        if not callable(method):
            continue
        try:
            aliases = method() or []
        except Exception:
            continue
        for alias in aliases:
            item = dict(alias)
            item.setdefault("class", class_label)
            rows.append(item)
    return rows


def _rpte_uses_direct_only_higher_order_patterns(model: Any) -> bool:
    """Whether fitted RPTE estimators keep patterns above order two out of trees."""
    return any(
        bool(getattr(sub, "higher_order_patterns_direct_only_", False))
        for sub, _class_label in _rpte_sub_estimators_for_audit(model)
    )


def _rpte_source_feature_names(model: Any) -> list[str]:
    for method_name in ("get_downstream_features", "_get_downstream_feature_names"):
        method = getattr(model, method_name, None)
        if not callable(method):
            continue
        try:
            value = method()
            names = [str(item) for item in list(value)] if value is not None else []
        except Exception:
            continue
        if names:
            return names
    value = getattr(model, "_downstream_feature_names_", None)
    return [str(item) for item in list(value)] if value is not None else []


def _rpte_source_family(name: str) -> str:
    text = str(name)
    if text.startswith("orig:"):
        return "original"
    if text.startswith("pattern:"):
        return "pattern"
    if text.startswith("augmented_pair:"):
        return "augmented_pair"
    return "unknown"


def _rpte_source_display_name(name: str) -> str:
    text = str(name)
    for prefix in ("orig:", "pattern:", "augmented_pair:"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _get_rpte_direct_source_rows(model: Any, include_zero: bool = True) -> list[dict[str, Any]]:
    """Return every direct source term carried directly into final LR.

    ``unified_rule_table()`` intentionally omits zero-valued direct terms. For
    representation audit the fitted column layout is authoritative, so this
    helper reads ``direct_input_indices_`` and the
    aligned LR coefficient slice, then enriches those rows with structured
    explanation metadata exposed by the model.
    """
    source_names = _rpte_source_feature_names(model)
    exposed = [
        row for row in _get_rpte_rule_rows(model)
        if _is_direct_source_row(row)
    ]

    def _matching_exposed(class_label: Any, input_index: int, name: str) -> dict[str, Any]:
        for row in exposed:
            row_index = row.get("downstream_feature_index")
            row_name = row.get("downstream_feature")
            same_index = row_index is not None and int(row_index) == int(input_index)
            same_name = row_name is not None and str(row_name) == str(name)
            same_class = str(row.get("class")) == str(class_label)
            if (same_index or same_name) and same_class:
                return dict(row)
        for row in exposed:
            row_index = row.get("downstream_feature_index")
            row_name = row.get("downstream_feature")
            if (row_index is not None and int(row_index) == int(input_index)) or (
                row_name is not None and str(row_name) == str(name)
            ):
                return dict(row)
        return {}

    records: list[dict[str, Any]] = []
    for sub, class_label in _rpte_sub_estimators_for_audit(model):
        indices = _direct_input_indices(sub)
        coefficients = _direct_input_coefficients(sub)
        if coefficients.size != indices.size:
            logistic = getattr(sub, "logistic_", None)
            coefficient_matrix = np.asarray(getattr(logistic, "coef_", []), dtype=float)
            n_leaf = int(getattr(sub, "n_leaf_features_", 0) or 0)
            flat = coefficient_matrix.reshape(-1) if coefficient_matrix.size else np.asarray([])
            coefficients = flat[n_leaf : n_leaf + indices.size]

        for position, input_index in enumerate(indices.tolist()):
            coefficient = float(coefficients[position]) if position < coefficients.size else np.nan
            if not include_zero and (not np.isfinite(coefficient) or abs(coefficient) < 1e-12):
                continue
            name = (
                source_names[input_index]
                if 0 <= input_index < len(source_names)
                else f"col{input_index}"
            )
            row = _matching_exposed(class_label, input_index, name)
            row.update(
                {
                    "class": class_label,
                    "tree_index": None,
                    "leaf_index": int(input_index),
                    "backend": "direct_hugiml_feature",
                    "term_role": "direct_source_term",
                    "source_selection_status": "not_selected_in_tree_split",
                    "support_count": None,
                    "support_rate": None,
                    "final_logistic_coefficient": coefficient,
                    "centered_tree_contribution": None,
                    "newton_leaf_value": None,
                    "downstream_feature_index": int(input_index),
                    "downstream_feature": name,
                    "source_family": _rpte_source_family(name),
                    "source_display_name": _rpte_source_display_name(name),
                }
            )
            row.setdefault("conditions", [])
            row.setdefault("raw_conditions", [])
            row.setdefault("raw_sources", [])
            records.append(row)
    return records


def _get_rpte_final_term_rows(model: Any, include_zero_direct: bool = True) -> list[dict[str, Any]]:
    """Return the complete fitted RPTE final-LR representation in column order.

    Tree leaves are followed by source columns not selected by accepted RPTE splits.
    Raw-source and constant fallback rows are retained unchanged.
    """
    rows = [
        dict(row) for row in _get_rpte_rule_rows(model)
        if not _is_direct_source_row(row)
    ]
    rows.extend(_get_rpte_direct_source_rows(model, include_zero=include_zero_direct))
    return rows


def _get_rpte_feature_flow_audit(model: Any) -> dict[str, Any]:
    """Return the fitted RPTE source-to-prediction representation audit.

    Models may expose ``rpte_feature_flow_audit()`` directly. The inference
    path also supports fitted estimators and lightweight test doubles: normal
    RPTE backends imply leaf-rule LR terms, while
    ``raw_hugiml_features`` is the explicit direct-source emergency fallback.
    """
    if model is None:
        return {}

    explicit: dict[str, Any] = {}
    fn = getattr(model, "rpte_feature_flow_audit", None)
    if callable(fn):
        try:
            explicit = dict(fn() or {})
        except Exception:
            explicit = {}

    try:
        composition = model.get_model_composition()
    except Exception:
        composition = {}
    if isinstance(composition, dict):
        embedded = composition.get("rpte_feature_flow")
        if isinstance(embedded, dict) and embedded:
            explicit = {**dict(embedded), **explicit}

    rows = _get_rpte_final_term_rows(model, include_zero_direct=True)
    alias_rows = _get_rpte_alias_rows(model)
    sub_estimators = _rpte_sub_estimators_for_audit(model)
    direct_only_higher_order = _rpte_uses_direct_only_higher_order_patterns(model)
    if not rows and not sub_estimators and not explicit:
        return {}
    backends = {str(row.get("backend") or "unknown") for row in rows}
    counts = composition.get("downstream_feature_counts", {}) if isinstance(composition, dict) else {}
    counts = dict(counts) if isinstance(counts, dict) else {}

    leaf_backends = {"bounded_lookahead", "sequential_default"}
    leaf_rows = [row for row in rows if str(row.get("backend")) in leaf_backends]
    direct_rows = [
        row for row in rows if _is_direct_source_row(row)
    ]
    raw_rows = [row for row in rows if str(row.get("backend")) == "raw_hugiml_features"]
    constant_rows = [row for row in rows if str(row.get("backend")) == "constant"]

    coefficient_count = 0
    leaf_count = 0
    direct_count = 0
    tree_used_count = 0
    for sub, _ in sub_estimators:
        logistic = getattr(sub, "logistic_", None)
        coefficients = np.asarray(getattr(logistic, "coef_", []), dtype=float)
        coefficient_count += int(coefficients.shape[-1]) if coefficients.ndim else int(coefficients.size)
        leaf_count += int(getattr(sub, "n_leaf_features_", 0) or 0)
        direct_count += int(_direct_input_indices(sub).size)
        tree_used_count += int(np.asarray(getattr(sub, "tree_used_input_indices_", [])).size)
    if not sub_estimators:
        coefficient_count = len(rows)
        leaf_count = len(leaf_rows)
        direct_count = len(direct_rows)

    if raw_rows and not leaf_rows:
        representation = "hugiml_source_features"
        statement = (
            "RPTE could not form a valid tree. The final LR uses HUGIML "
            "original/pattern/augmented source columns directly."
        )
        uses_leaf_rules = False
        uses_source_directly = True
    elif constant_rows and not leaf_rows:
        representation = "constant"
        statement = "The fitted RPTE path reduced to a constant class-prior fallback."
        uses_leaf_rules = False
        uses_source_directly = False
    elif direct_count > 0 or direct_rows:
        representation = "rpte_leaves_plus_direct_terms"
        statement = (
            "The final LR uses RPTE leaf indicators plus direct HUGIML source terms. "
            + (
                "Patterns above order two are direct-only; originals, generated pairs, and "
                "lower-order patterns may be tree primitives. "
                if direct_only_higher_order
                else ""
            )
            + (
                "Structurally equivalent leaf-pattern aliases are represented once."
                if alias_rows
                else ""
            )
        ).strip()
        uses_leaf_rules = True
        uses_source_directly = True
    else:
        representation = "rpte_leaf_rules"
        statement = "The final LR uses only RPTE root-to-leaf conjunction indicators."
        if direct_only_higher_order:
            statement += " Patterns above order two are not used as tree primitives."
        uses_leaf_rules = True
        uses_source_directly = False

    source_names = _rpte_source_feature_names(model)
    direct_family_counts = {
        family: int(sum(_rpte_source_family(str(row.get("downstream_feature") or "")) == family for row in direct_rows))
        for family in ("original", "pattern", "augmented_pair")
    }
    inferred = {
        "is_rpte": True,
        "final_representation": representation,
        "backend": next(iter(backends)) if len(backends) == 1 else "mixed",
        "coefficient_count": coefficient_count,
        "final_term_count": coefficient_count,
        "leaf_rule_count": leaf_count if uses_leaf_rules else 0,
        "direct_source_term_count": direct_count,
        "direct_source_nonzero_term_count": int(sum(abs(float(row.get("final_logistic_coefficient") or 0.0)) >= 1e-12 for row in direct_rows)),
        "direct_source_family_counts": direct_family_counts,
        "tree_used_source_feature_count": tree_used_count,
        "suppressed_leaf_pattern_alias_count": len(alias_rows),
        "source_feature_count": int(counts.get("total", 0) or len(source_names)),
        "source_feature_counts": counts,
        "final_lr_uses_leaf_rules": uses_leaf_rules,
        "final_lr_uses_source_features_directly": uses_source_directly,
        "source_features_are_tree_inputs": uses_leaf_rules,
        "patterns_above_order_two_are_direct_only": direct_only_higher_order,
        "statement": statement,
    }
    return {**explicit, **inferred}


def format_conjunction(conditions: list[dict[str, Any]], max_terms: int = 6) -> str:
    """Render a root-to-leaf path with raw-unit conditions when available."""
    if not conditions:
        return "(no split conditions -- linear/fallback term)"
    terms = []
    for condition in conditions:
        text = condition.get("raw_condition") or condition.get("downstream_condition") or "?"
        terms.append(str(text))
    if len(terms) > max_terms:
        shown = terms[:max_terms]
        return " AND ".join(shown) + f" AND (+{len(terms) - max_terms} more)"
    return " AND ".join(terms)


def _numbered_conditions(conditions: list[dict[str, Any]]) -> str:
    if not conditions:
        return "1. (no split condition)"
    return "\n".join(
        f"{idx}. {condition.get('raw_condition') or condition.get('downstream_condition') or '?'}"
        for idx, condition in enumerate(conditions, start=1)
    )


def _rpte_rules_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize RPTE prediction terms into an audit-friendly table."""
    if not rows:
        return pd.DataFrame()
    records = []
    for source_index, row in enumerate(rows):
        conditions = list(row.get("conditions") or [])
        coefficient = pd.to_numeric(
            pd.Series([row.get("final_logistic_coefficient")]), errors="coerce"
        ).iloc[0]
        coefficient_value = float(coefficient) if pd.notna(coefficient) else np.nan
        if pd.isna(coefficient_value) or abs(coefficient_value) < 1e-12:
            effect = "Neutral / zero"
        elif coefficient_value > 0:
            effect = "Increases positive-class odds"
        else:
            effect = "Decreases positive-class odds"
        odds_multiplier = (
            float(np.exp(np.clip(coefficient_value, -50.0, 50.0)))
            if pd.notna(coefficient_value)
            else np.nan
        )
        class_value = row.get("class")
        tree_value = row.get("tree_index")
        leaf_value = row.get("leaf_index")
        backend = str(row.get("backend") or "unknown")
        source_column = str(row.get("downstream_feature") or "")
        source_family = str(row.get("source_family") or _rpte_source_family(source_column))
        is_leaf_term = backend in {"bounded_lookahead", "sequential_default"}
        is_direct_source_term = _is_direct_source_row(row)
        if is_leaf_term:
            term_type = "RPTE leaf indicator"
            rule_id = (
                f"class={class_value} | tree={tree_value} | leaf={leaf_value} | "
                f"backend={backend} | row={source_index + 1}"
            )
            preview = format_conjunction(conditions, max_terms=3)
            conjunction = format_conjunction(conditions, max_terms=10_000)
            condition_steps = _numbered_conditions(conditions)
        elif is_direct_source_term:
            family_label = {
                "original": "direct original feature",
                "pattern": "direct HUG pattern",
                "augmented_pair": "direct augmented pair",
            }.get(source_family, "direct source term")
            term_type = family_label
            display_name = str(row.get("source_display_name") or _rpte_source_display_name(source_column))
            rule_id = f"class={class_value} | {family_label} | {source_column}"
            preview = display_name or source_column
            conjunction = preview
            condition_steps = f"1. Direct linear term: {source_column}"
        elif backend == "raw_hugiml_features":
            term_type = "RPTE source fallback term"
            rule_id = f"class={class_value} | fallback source | {leaf_value}"
            preview = format_conjunction(conditions, max_terms=3)
            conjunction = format_conjunction(conditions, max_terms=10_000)
            condition_steps = _numbered_conditions(conditions)
        elif backend == "constant":
            term_type = "constant fallback"
            rule_id = f"class={class_value} | constant"
            preview = "constant class-prior term"
            conjunction = preview
            condition_steps = "1. Constant term"
        else:
            term_type = backend
            rule_id = f"class={class_value} | backend={backend} | row={source_index + 1}"
            preview = format_conjunction(conditions, max_terms=3)
            conjunction = format_conjunction(conditions, max_terms=10_000)
            condition_steps = _numbered_conditions(conditions)
        records.append(
            {
                "rule_id": rule_id,
                "class": class_value,
                "tree": tree_value,
                "leaf": leaf_value,
                "backend": backend,
                "term_type": term_type,
                "source_family": source_family,
                "source_column": source_column,
                "source_display_name": str(row.get("source_display_name") or _rpte_source_display_name(source_column)),
                "is_leaf_term": is_leaf_term,
                "is_direct_source_term": is_direct_source_term,
                "n_conditions": len(conditions),
                "rule_preview": preview,
                # Keep `conjunction` as the complete rule because Workbench
                # run-to-run deltas use it as part of the rule identity.
                "conjunction": conjunction,
                "condition_steps": condition_steps,
                "raw_sources": ", ".join(row.get("raw_sources") or []),
                "support_count": row.get("support_count"),
                "support_rate": row.get("support_rate"),
                "coefficient": coefficient_value,
                "abs_coefficient": abs(coefficient_value) if pd.notna(coefficient_value) else np.nan,
                "effect": effect,
                "odds_multiplier": odds_multiplier,
                "vs_tree_baseline": row.get("centered_tree_contribution"),
                "fallback_status": row.get("fallback_status"),
                "_source_row_index": source_index,
            }
        )
    df = pd.DataFrame(records)
    return df.sort_values("abs_coefficient", ascending=False, na_position="last").reset_index(drop=True)


def _rpte_flat_tree_sections(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Return the same flat-tree sections exposed by the RPTE text API."""
    if not rows:
        return []

    allowed: set[int] | None = None
    if frame is not None and not frame.empty and "_source_row_index" in frame.columns:
        allowed = {int(value) for value in frame["_source_row_index"].tolist()}

    selected_rows = []
    for source_index, row in enumerate(rows):
        if allowed is not None and source_index not in allowed:
            continue
        item = dict(row)
        item["_source_row_index"] = source_index
        selected_rows.append(item)

    sections = rpte_rule_tree_sections(
        selected_rows,
        condition_space="raw",
        detail_level="compact",
        precision=5,
    )
    for section in sections:
        section["coefficients"] = pd.DataFrame(section.get("coefficients") or [])
    return sections

def render_rpte_flat_tree_view(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame | None = None,
    *,
    key_prefix: str = "rpte_flat_tree",
) -> list[dict[str, Any]]:
    """Render one RPTE tree as an indented DT-style structure plus coefficients."""
    sections = _rpte_flat_tree_sections(rows, frame)
    st.markdown("#### Flat RPTE tree view")
    st.caption(
        "Shared root-to-leaf prefixes are merged like a decision-tree text view. "
        "The fitted LR coefficient is shown at each terminal leaf; internal split "
        "conditions do not have standalone LR coefficients."
    )
    if not sections:
        st.info("No RPTE leaf paths match the current filters.")
        return sections

    if len(sections) == 1:
        selected_index = 0
        st.markdown(f"**{sections[0]['title']}**")
    else:
        selected_index = st.selectbox(
            "RPTE tree",
            list(range(len(sections))),
            format_func=lambda idx: (
                f"{sections[idx]['title']} | {sections[idx]['leaf_count']} leaves | "
                f"depth {sections[idx]['max_depth']}"
            ),
            key=f"{key_prefix}_selector",
        )
    selected = sections[int(selected_index)]
    code_renderer = getattr(st, "code", None)
    if callable(code_renderer):
        code_renderer(selected["text"], language="text")
    else:  # pragma: no cover - compatibility with minimal Streamlit test doubles
        st.markdown(f"```text\n{selected['text']}\n```")
    st.markdown("##### Leaf coefficients")
    coefficient_frame = selected["coefficients"]
    if coefficient_frame.empty:
        st.info("No coefficients are available for the selected tree.")
    else:
        st.dataframe(
            dataframe_for_display(coefficient_frame.drop(columns=["_source_row_index"], errors="ignore")),
            width="stretch",
            hide_index=True,
        )
    return sections


def render_rpte_rule_conjunctions(model: Any = None) -> pd.DataFrame:
    """Render RPTE leaf indicators and direct source terms."""
    rows = _get_rpte_final_term_rows(model, include_zero_direct=True)
    if not rows:
        return pd.DataFrame()
    audit = _get_rpte_feature_flow_audit(model)
    representation = audit.get("final_representation", "rpte_leaf_rules")
    df = _rpte_rules_to_frame(rows)
    leaf_df = df.loc[df["is_leaf_term"]].copy() if "is_leaf_term" in df.columns else pd.DataFrame()
    direct_df = (
        df.loc[df["is_direct_source_term"].astype(bool)].copy()
        if "is_direct_source_term" in df.columns
        else pd.DataFrame()
    )

    if representation == "hugiml_source_features":
        st.markdown("### Final LR Source Terms (RPTE Source Fallback)")
    elif representation == "constant":
        st.markdown("### Final RPTE Constant Representation")
    else:
        st.markdown("### Final RPTE Logistic-Regression Representation")

    statement = audit.get("statement")
    if statement:
        st.info(str(statement))
    st.caption(
        "Leaf indicators encode root-to-leaf conjunctions. Source columns not selected by accepted "
        "RPTE splits are appended directly after the leaf block. Coefficients are fitted LR "
        "log-odds effects; exp(coefficient) is the corresponding odds multiplier."
    )

    metrics = st.columns(5)
    metrics[0].metric("Final LR terms", f"{int(audit.get('final_term_count', len(df)) or len(df)):,}")
    metrics[1].metric("Leaf terms", f"{len(leaf_df):,}")
    metrics[2].metric("Direct source terms", f"{_direct_source_count(audit, len(direct_df)):,}")
    tree_count = leaf_df["tree"].nunique() if not leaf_df.empty and leaf_df["tree"].notna().any() else 0
    metrics[3].metric("Trees", f"{tree_count:,}" if tree_count else "N/A")
    metrics[4].metric("Source inputs", f"{int(audit.get('source_feature_count', 0) or 0):,}")

    if not leaf_df.empty:
        st.markdown("#### RPTE leaf trees")
        query = st.text_input(
            "Search leaf rules",
            value="",
            placeholder="condition, raw feature, class, backend...",
            key="rpte_leaf_rule_search",
        )
        effect_filter = st.selectbox(
            "Leaf effect direction",
            ["All", "Increases positive-class odds", "Decreases positive-class odds", "Neutral / zero"],
            key="rpte_leaf_rule_effect_filter",
        )
        show_leaf = leaf_df
        if query:
            mask = leaf_df.astype(str).apply(
                lambda col: col.str.contains(query, case=False, na=False)
            ).any(axis=1)
            show_leaf = leaf_df.loc[mask]
        if effect_filter != "All":
            show_leaf = show_leaf.loc[show_leaf["effect"].eq(effect_filter)]
        render_rpte_flat_tree_view(rows, show_leaf, key_prefix="rpte_prediction_flat_tree")
        st.markdown("##### Leaf coefficient table")
        leaf_columns = [
            "class", "tree", "leaf", "effect", "coefficient", "odds_multiplier",
            "support_rate", "support_count", "n_conditions", "raw_sources", "backend",
        ]
        st.dataframe(
            dataframe_for_display(show_leaf[[c for c in leaf_columns if c in show_leaf.columns]]),
            width="stretch",
            hide_index=True,
        )

        with st.expander("Inspect one leaf path condition by condition", expanded=False):
            if show_leaf.empty:
                st.info("No leaf terms match the current filters.")
            else:
                options = list(show_leaf.index)
                pick = st.selectbox(
                    "Leaf term",
                    options,
                    format_func=lambda i: f"{show_leaf.loc[i, 'rule_id']} | coef={show_leaf.loc[i, 'coefficient']:.4f}",
                    key="rpte_leaf_rule_detail_pick",
                )
                selected = show_leaf.loc[pick]
                source_index = int(selected["_source_row_index"])
                original = rows[source_index]
                st.markdown(f"**Effect:** {selected['effect']}")
                st.markdown(
                    f"**Coefficient:** {selected['coefficient']:.6g} &nbsp; | &nbsp; "
                    f"**Odds multiplier:** {selected['odds_multiplier']:.6g}"
                )
                for line in str(selected["condition_steps"]).splitlines():
                    st.markdown(line)
                cond_df = pd.DataFrame(original.get("conditions") or [])
                if not cond_df.empty:
                    cond_df.insert(0, "step", np.arange(1, len(cond_df) + 1))
                    preferred = [
                        c for c in (
                            "step", "raw_condition", "downstream_condition", "downstream_feature",
                            "family", "operator", "raw_sources", "invertible",
                        ) if c in cond_df.columns
                    ]
                    st.dataframe(
                        dataframe_for_display(cond_df[preferred] if preferred else cond_df),
                        width="stretch",
                        hide_index=True,
                    )

    if not direct_df.empty:
        st.markdown("#### Direct source terms carried directly into LR")
        st.caption(
            "Only source columns not selected by accepted RPTE splits appear here. They remain grouped "
            "by their HUGIML family: original, pattern, or augmented pair."
        )
        family_options = ["All", "Original", "HUG pattern", "Augmented pair"]
        family_filter = st.selectbox(
            "Direct source family",
            family_options,
            key="rpte_direct_source_family_filter",
        )
        nonzero_only = st.checkbox(
            "Show only non-zero direct coefficients",
            value=False,
            key="rpte_direct_source_nonzero_only",
        )
        show_direct = direct_df.copy()
        family_map = {
            "Original": "original",
            "HUG pattern": "pattern",
            "Augmented pair": "augmented_pair",
        }
        if family_filter != "All":
            show_direct = show_direct.loc[show_direct["source_family"].eq(family_map[family_filter])]
        if nonzero_only:
            show_direct = show_direct.loc[show_direct["abs_coefficient"].fillna(0.0).ge(1e-12)]
        direct_columns = [
            "class", "term_type", "source_display_name", "source_column", "raw_sources",
            "effect", "coefficient", "odds_multiplier", "backend",
        ]
        st.dataframe(
            dataframe_for_display(show_direct[[c for c in direct_columns if c in show_direct.columns]]),
            width="stretch",
            hide_index=True,
        )
    elif representation in _RPTE_LEAF_DIRECT_REPRESENTATIONS:
        st.info("No direct source terms were exposed for this fitted model.")

    if leaf_df.empty and direct_df.empty:
        display_columns = [
            "class", "term_type", "effect", "coefficient", "odds_multiplier", "backend",
        ]
        st.dataframe(
            dataframe_for_display(df[[c for c in display_columns if c in df.columns]]),
            width="stretch",
            hide_index=True,
        )
    return df

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
    df = _patterns_to_frame(model)
    final_rows = _get_rpte_final_term_rows(model, include_zero_direct=True)
    audit = _get_rpte_feature_flow_audit(model)
    representation = audit.get("final_representation")

    if df.empty and not final_rows and not audit:
        st.subheader("Pattern Inventory")
        st.info("Pattern information is not available for this model/version.")
        return

    if final_rows:
        st.subheader("RPTE Prediction Representation")
        render_rpte_rule_conjunctions(model)
        st.divider()
    elif audit:
        st.subheader("RPTE Prediction Representation")
        st.info(str(audit.get("statement") or "RPTE feature-flow audit is available."))
        st.caption("No individual RPTE prediction terms were exposed by this fitted model.")
        st.divider()

    if representation == "rpte_leaf_rules" or representation in _RPTE_LEAF_DIRECT_REPRESENTATIONS:
        st.subheader("Direct HUG Patterns in the Final LR")
        st.caption(
            "This inventory is filtered to HUG pattern columns not selected by accepted RPTE splits. "
            "Patterns used in tree splits are represented through the leaf rules above and are not "
            "repeated as direct LR terms."
        )
        term_frame = _rpte_rules_to_frame(final_rows)
        direct_patterns = term_frame.loc[
            term_frame.get(
                "is_direct_source_term", pd.Series(False, index=term_frame.index)
            ).astype(bool)
            & term_frame.get("source_family", pd.Series("", index=term_frame.index)).eq("pattern")
        ].copy()
        direct_names = set(direct_patterns.get("source_display_name", pd.Series(dtype=str)).astype(str))
        if not df.empty and direct_names:
            label_col = next(
                (c for c in df.columns if str(c).lower() in {"pattern", "label", "feature", "name", "display_name"}),
                None,
            )
            if label_col is not None:
                filtered = df.loc[df[label_col].astype(str).isin(direct_names)].copy()
            else:
                filtered = pd.DataFrame()
        else:
            filtered = pd.DataFrame()
        if filtered.empty:
            st.info("No direct HUG pattern terms are present in the final LR.")
            return
        query = st.text_input(
            "Search direct HUG patterns",
            value="",
            placeholder="feature name, interval, category...",
        )
        show_df = filtered
        if query:
            mask = filtered.astype(str).apply(
                lambda col: col.str.contains(query, case=False, na=False)
            ).any(axis=1)
            show_df = filtered.loc[mask]
        st.dataframe(show_df, width="stretch", hide_index=True)
        st.divider()
        render_pattern_support_coverage_audit(model, filtered)
        return

    if representation == "hugiml_source_features":
        st.subheader("HUGIML Pattern Source Terms")
        st.warning(
            "This fitted object uses the explicit RPTE source fallback. Pattern columns shown here "
            "are direct LR terms rather than leaf-construction inputs."
        )
    else:
        st.subheader("Pattern Inventory")
        st.markdown(
            "Human-readable HUG patterns are downstream features for the standard HUGIML LR path. "
            "Search by feature name, category, interval, or pattern text."
        )

    if not df.empty:
        c1, c2 = st.columns([0.7, 0.3], gap="large")
        with c1:
            query = st.text_input(
                "Search patterns",
                value="",
                placeholder="feature name, interval, category...",
            )
        with c2:
            c2.metric("Patterns shown", f"{len(df):,}")
        show_df = df
        if query:
            mask = df.astype(str).apply(
                lambda col: col.str.contains(query, case=False, na=False)
            ).any(axis=1)
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

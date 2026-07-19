"""RPTE representation and governance helpers.

The fitted HUGIML/RPTE pipeline has distinct stages:
raw inputs -> HUGIML source columns -> RPTE tree splits and leaf indicators ->
final LR terms. In the current representation, source columns used by accepted
RPTE splits are represented through leaf indicators, while source columns not
used by any accepted split are carried directly into the final LR.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hugiml.dashboard.components.patterns import (
    _DIRECT_SOURCE_BACKENDS,
    _RPTE_LEAF_DIRECT_REPRESENTATIONS,
    _direct_input_coefficients,
    _direct_input_indices,
    _direct_source_count,
    _get_rpte_feature_flow_audit,
    _get_rpte_rule_rows,
    _is_direct_source_row,
    _rpte_sub_estimators_for_audit,
)

LEAF_BACKENDS = {"bounded_lookahead", "sequential_default"}
DIRECT_SOURCE_BACKENDS = _DIRECT_SOURCE_BACKENDS
RAW_FALLBACK_BACKEND = "raw_hugiml_features"


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


def rpte_is_active(model: Any) -> bool:
    """Return True when the fitted model exposes an RPTE representation."""
    return bool(_get_rpte_feature_flow_audit(model))


def rpte_has_tree_representation(model: Any) -> bool:
    """Return True when fitted RPTE leaf indicators are part of final LR."""
    representation = _get_rpte_feature_flow_audit(model).get("final_representation")
    return representation == "rpte_leaf_rules" or representation in _RPTE_LEAF_DIRECT_REPRESENTATIONS


def rpte_source_feature_names(model: Any) -> list[str]:
    """Names of HUGIML source columns supplied to the RPTE estimator."""
    if model is None:
        return []
    for attr in ("get_downstream_features", "_get_downstream_feature_names"):
        fn = getattr(model, attr, None)
        if not callable(fn):
            continue
        try:
            names = [str(v) for v in _as_list(fn())]
        except Exception:
            continue
        if names:
            return names
    names = getattr(model, "_downstream_feature_names_", None)
    return [str(v) for v in _as_list(names)]


def source_family(name: str) -> str:
    text = str(name)
    if text.startswith("orig:"):
        return "original"
    if text.startswith("pattern:"):
        return "pattern"
    if text.startswith("augmented_pair:"):
        return "augmented_pair"
    if text.startswith("synthetic_pair:"):
        return "rpte_synthetic_pair"
    return "unknown"


def source_display_name(name: str) -> str:
    text = str(name)
    for prefix in ("orig:", "pattern:", "augmented_pair:"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _split_pattern_assignments(label: str) -> list[str]:
    text = str(label)
    parts: list[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(text):
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            part = text[start:i].strip()
            if part:
                parts.append(part)
            start = i + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _raw_sources_for_source_column(model: Any, name: str) -> list[str]:
    family = source_family(name)
    display = source_display_name(name)
    if family == "original":
        return [display]
    if family == "pattern":
        sources: list[str] = []
        provenance = getattr(model, "get_pattern_provenance", None)
        if callable(provenance):
            try:
                item = dict(provenance() or {}).get(str(name), {})
                values = [str(v) for v in _as_list(item.get("raw_features"))]
                if values:
                    return values
            except Exception:
                pass
        for atom in _split_pattern_assignments(display):
            if "=" not in atom:
                continue
            feature = atom.split("=", 1)[0].strip()
            if feature and feature not in sources:
                sources.append(feature)
        return sources
    if family == "augmented_pair":
        for item in _as_list(getattr(model, "augmented_pair_transforms_", None)):
            if not isinstance(item, dict):
                continue
            item_name = str(item.get("name", ""))
            if item_name == display or f"augmented_pair:{item_name}" == str(name):
                return [str(v) for v in _as_list(item.get("inputs"))]
    return []


def _class_key(value: Any) -> str:
    return "binary" if value is None else str(value)


def rpte_direct_source_terms_frame(model: Any, include_zero: bool = True) -> pd.DataFrame:
    """Direct final-LR source terms not selected in accepted RPTE splits.

    Fitted estimator attributes retain zero-coefficient terms; unified explanation
    rows provide structured metadata for non-zero terms.
    """
    names = rpte_source_feature_names(model)
    records: list[dict[str, Any]] = []
    for sub, class_label in _rpte_sub_estimators_for_audit(model):
        indices = _direct_input_indices(sub)
        coefficients = _direct_input_coefficients(sub)
        if coefficients.size != indices.size:
            logistic = getattr(sub, "logistic_", None)
            coef = np.asarray(getattr(logistic, "coef_", []), dtype=float)
            n_leaf = int(getattr(sub, "n_leaf_features_", 0) or 0)
            coefficients = coef.reshape(-1)[n_leaf : n_leaf + indices.size] if coef.size else np.full(indices.size, np.nan)
        for position, input_idx in enumerate(indices.tolist()):
            coefficient = float(coefficients[position]) if position < coefficients.size else np.nan
            if not include_zero and (not np.isfinite(coefficient) or abs(coefficient) < 1e-12):
                continue
            name = names[input_idx] if 0 <= input_idx < len(names) else f"col{input_idx}"
            records.append(
                {
                    "class": class_label,
                    "source_index": int(input_idx),
                    "source_column": name,
                    "family": source_family(name),
                    "display_name": source_display_name(name),
                    "raw_sources": ", ".join(_raw_sources_for_source_column(model, name)),
                    "coefficient": coefficient,
                    "abs_coefficient": abs(coefficient) if np.isfinite(coefficient) else np.nan,
                    "odds_multiplier": float(np.exp(np.clip(coefficient, -50.0, 50.0))) if np.isfinite(coefficient) else np.nan,
                    "nonzero": bool(np.isfinite(coefficient) and abs(coefficient) >= 1e-12),
                    "final_term_type": "direct source term",
                }
            )
    if records:
        return pd.DataFrame(records).sort_values(
            ["abs_coefficient", "class", "source_index"],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)

    for row in _get_rpte_rule_rows(model):
        if not _is_direct_source_row(row):
            continue
        coefficient = pd.to_numeric(pd.Series([row.get("final_logistic_coefficient")]), errors="coerce").iloc[0]
        coefficient = float(coefficient) if pd.notna(coefficient) else np.nan
        if not include_zero and (not np.isfinite(coefficient) or abs(coefficient) < 1e-12):
            continue
        name = str(row.get("downstream_feature") or f"col{row.get('downstream_feature_index')}")
        records.append(
            {
                "class": row.get("class"),
                "source_index": row.get("downstream_feature_index"),
                "source_column": name,
                "family": source_family(name),
                "display_name": source_display_name(name),
                "raw_sources": ", ".join(str(v) for v in _as_list(row.get("raw_sources"))),
                "coefficient": coefficient,
                "abs_coefficient": abs(coefficient) if np.isfinite(coefficient) else np.nan,
                "odds_multiplier": float(np.exp(np.clip(coefficient, -50.0, 50.0))) if np.isfinite(coefficient) else np.nan,
                "nonzero": bool(np.isfinite(coefficient) and abs(coefficient) >= 1e-12),
                "final_term_type": "direct source term",
            }
        )
    return pd.DataFrame(records)


def rpte_split_usage_frame(model: Any) -> pd.DataFrame:
    """Summarise source and RPTE-synthesized columns used in accepted splits."""
    if not rpte_has_tree_representation(model):
        return pd.DataFrame()
    rows = [
        row for row in _get_rpte_rule_rows(model)
        if str(row.get("backend")) in LEAF_BACKENDS
    ]
    if not rows:
        return pd.DataFrame()

    stats: dict[str, dict[str, Any]] = {}
    for rule_index, rule in enumerate(rows):
        coefficient = pd.to_numeric(pd.Series([rule.get("final_logistic_coefficient")]), errors="coerce").iloc[0]
        abs_coefficient = float(abs(coefficient)) if pd.notna(coefficient) else np.nan
        tree = rule.get("tree_index")
        leaf = rule.get("leaf_index")
        class_label = rule.get("class")
        for condition in _as_list(rule.get("conditions")):
            name = condition.get("downstream_feature")
            if not name:
                text = str(condition.get("downstream_condition") or "")
                name = text.split(" ", 1)[0] if text else "<unknown>"
            name = str(name)
            item = stats.setdefault(
                name,
                {
                    "source_column": name,
                    "family": str(condition.get("family") or source_family(name)),
                    "display_name": source_display_name(name),
                    "condition_occurrences": 0,
                    "rule_ids": set(),
                    "trees": set(),
                    "classes": set(),
                    "raw_sources": set(),
                    "max_abs_leaf_coefficient": np.nan,
                },
            )
            item["condition_occurrences"] += 1
            item["rule_ids"].add((class_label, tree, leaf, rule_index))
            item["trees"].add((class_label, tree))
            item["classes"].add(class_label)
            item["raw_sources"].update(str(v) for v in _as_list(condition.get("raw_sources")))
            if pd.notna(abs_coefficient):
                current = item["max_abs_leaf_coefficient"]
                item["max_abs_leaf_coefficient"] = abs_coefficient if pd.isna(current) else max(float(current), abs_coefficient)

    out = [
        {
            "source_column": item["source_column"],
            "family": item["family"],
            "display_name": item["display_name"],
            "condition_occurrences": int(item["condition_occurrences"]),
            "leaf_rules_using_column": len(item["rule_ids"]),
            "trees_using_column": len(item["trees"]),
            "classes_using_column": len(item["classes"]),
            "raw_sources": ", ".join(sorted(item["raw_sources"])),
            "max_abs_leaf_coefficient": item["max_abs_leaf_coefficient"],
        }
        for item in stats.values()
    ]
    return pd.DataFrame(out).sort_values(
        ["leaf_rules_using_column", "condition_occurrences", "source_column"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def rpte_source_inventory_frame(model: Any) -> pd.DataFrame:
    """Inventory every HUGIML source column and its fitted RPTE role."""
    names = rpte_source_feature_names(model)
    if not names:
        return pd.DataFrame()
    flow = _get_rpte_feature_flow_audit(model)
    usage = rpte_split_usage_frame(model)
    usage_map = {str(row["source_column"]): row.to_dict() for _, row in usage.iterrows()} if not usage.empty else {}
    direct = rpte_direct_source_terms_frame(model, include_zero=True)
    direct_groups: dict[str, pd.DataFrame] = {
        str(name): group for name, group in direct.groupby("source_column", dropna=False)
    } if not direct.empty else {}

    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        used = usage_map.get(name, {})
        direct_group = direct_groups.get(name)
        direct_count = int(len(direct_group)) if direct_group is not None else 0
        direct_classes = [] if direct_group is None else [_class_key(v) for v in direct_group["class"].tolist()]
        max_abs_direct = (
            float(pd.to_numeric(direct_group["abs_coefficient"], errors="coerce").max())
            if direct_group is not None and not direct_group.empty
            else np.nan
        )
        family = source_family(name)
        used_in_split = bool(used)
        direct_in_lr = direct_count > 0 or flow.get("final_representation") == "hugiml_source_features"
        if used_in_split and direct_in_lr:
            role = "Used in RPTE splits for at least one class and carried directly for another class"
        elif used_in_split:
            role = "Used in accepted RPTE splits; represented through leaf indicators"
        elif direct_in_lr:
            role = "Not selected by accepted RPTE splits; carried directly into final LR"
        else:
            role = "Supplied to RPTE but not present in exposed final terms"
        rows.append(
            {
                "source_index": idx,
                "source_column": name,
                "family": family,
                "display_name": source_display_name(name),
                "raw_sources": ", ".join(_raw_sources_for_source_column(model, name)),
                "supplied_to_rpte": True,
                "used_in_fitted_split": used_in_split,
                "leaf_rules_using_column": int(used.get("leaf_rules_using_column", 0) or 0),
                "trees_using_column": int(used.get("trees_using_column", 0) or 0),
                "condition_occurrences": int(used.get("condition_occurrences", 0) or 0),
                "direct_final_lr_term": direct_in_lr,
                "direct_term_count": direct_count,
                "direct_nonzero_term_count": int(direct_group["nonzero"].sum()) if direct_group is not None else 0,
                "direct_term_classes": ", ".join(sorted(set(direct_classes))),
                "max_abs_direct_coefficient": max_abs_direct,
                "fitted_role": role,
            }
        )
    return pd.DataFrame(rows)


def rpte_raw_input_lineage_frame(model: Any, X: pd.DataFrame | None = None) -> pd.DataFrame:
    """Trace raw inputs into tree-based and direct final-LR terms."""
    names = [str(v) for v in _as_list(getattr(model, "feature_names_in_", None))]
    if not names and X is not None:
        names = [str(v) for v in X.columns]

    inventory = rpte_source_inventory_frame(model)
    leaf_rows = [row for row in _get_rpte_rule_rows(model) if str(row.get("backend")) in LEAF_BACKENDS]
    rows: list[dict[str, Any]] = []
    for raw in names:
        direct_original = pattern_count = augmented_count = 0
        split_sources = direct_lr_sources = 0
        if not inventory.empty:
            for _, item in inventory.iterrows():
                sources = [v.strip() for v in str(item.get("raw_sources", "")).split(",") if v.strip()]
                if raw not in sources:
                    continue
                family = str(item.get("family", ""))
                if family == "original":
                    direct_original += 1
                elif family == "pattern":
                    pattern_count += 1
                elif family == "augmented_pair":
                    augmented_count += 1
                split_sources += int(bool(item.get("used_in_fitted_split")))
                direct_lr_sources += int(bool(item.get("direct_final_lr_term")))
        affected_rules = 0
        affected_trees: set[tuple[Any, Any]] = set()
        for rule in leaf_rows:
            if raw in {str(v) for v in _as_list(rule.get("raw_sources"))}:
                affected_rules += 1
                affected_trees.add((rule.get("class"), rule.get("tree_index")))
        rows.append(
            {
                "raw_input": raw,
                "selected_original_source_columns": direct_original,
                "derived_pattern_source_columns": pattern_count,
                "derived_augmented_source_columns": augmented_count,
                "source_columns_used_in_rpte_splits": split_sources,
                "direct_source_columns_used_directly_in_lr": direct_lr_sources,
                "rpte_leaf_rules_referencing_raw_input": affected_rules,
                "rpte_trees_referencing_raw_input": len(affected_trees),
                "contributes_to_direct_final_lr_terms": direct_lr_sources > 0,
                "valid_removal_action": "Exclude the raw input and rebuild preprocessing, mining, generated pairs, RPTE trees, leaf indicators, direct source terms, and LR",
            }
        )
    return pd.DataFrame(rows)


def rpte_representation_flow_frame(model: Any, X: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fitted representation contract for Governance UI."""
    flow = _get_rpte_feature_flow_audit(model)
    if not flow:
        return pd.DataFrame()
    source = rpte_source_inventory_frame(model)
    raw_count = len(_as_list(getattr(model, "feature_names_in_", None)))
    if not raw_count and X is not None:
        raw_count = int(X.shape[1])
    source_count = int(flow.get("source_feature_count", len(source)) or len(source))
    used_count = int(source["used_in_fitted_split"].sum()) if not source.empty else int(flow.get("tree_used_source_feature_count", 0) or 0)
    leaf_count = int(flow.get("leaf_rule_count", 0) or 0)
    direct_count = _direct_source_count(flow)
    final_count = int(flow.get("final_term_count", flow.get("coefficient_count", leaf_count + direct_count)) or 0)
    rep = flow.get("final_representation")
    return pd.DataFrame(
        [
            {"stage": 1, "representation": "Raw model inputs", "count": raw_count, "role": "Inputs to preprocessing, HUG mining, and generated-pair construction", "direct_final_lr_terms": False},
            {"stage": 2, "representation": "HUGIML source columns supplied to RPTE", "count": source_count, "role": "Original, pattern, and augmented columns considered by RPTE", "direct_final_lr_terms": rep == "hugiml_source_features"},
            {"stage": 3, "representation": "Source columns used in accepted RPTE splits", "count": used_count, "role": "Represented indirectly through the fitted tree leaves", "direct_final_lr_terms": False},
            {"stage": 4, "representation": "RPTE leaf indicators", "count": leaf_count, "role": "One-hot root-to-leaf conjunction indicators", "direct_final_lr_terms": rep == "rpte_leaf_rules" or rep in _RPTE_LEAF_DIRECT_REPRESENTATIONS},
            {"stage": 5, "representation": "Direct source columns carried into LR", "count": direct_count, "role": "Supplied source columns not used by an accepted split; appended after leaf indicators", "direct_final_lr_terms": rep in _RPTE_LEAF_DIRECT_REPRESENTATIONS or rep == "hugiml_source_features"},
            {"stage": 6, "representation": "Final logistic-regression coefficients", "count": final_count, "role": (
                "Leaf coefficients followed by direct source coefficients"
                if rep in _RPTE_LEAF_DIRECT_REPRESENTATIONS
                else "Coefficients aligned one-to-one with RPTE leaf indicators"
                if rep == "rpte_leaf_rules"
                else "Coefficients aligned with HUGIML source columns in the explicit fallback"
                if rep == "hugiml_source_features"
                else "Constant fitted representation"
            ), "direct_final_lr_terms": True},
        ]
    )


def rpte_model_comparison_row(label: str, model: Any, score: float | None, X: pd.DataFrame | None = None) -> dict[str, Any]:
    """Separated RPTE source, leaf, and direct-term counts for comparisons."""
    flow = _get_rpte_feature_flow_audit(model)
    source = rpte_source_inventory_frame(model)
    counts = source["family"].value_counts().to_dict() if not source.empty else {}
    raw_count = len(_as_list(getattr(model, "feature_names_in_", None)))
    if not raw_count and X is not None:
        raw_count = int(X.shape[1])
    return {
        "run": label,
        "validation_roc_auc": score,
        "raw_input_features": int(raw_count),
        "rpte_source_original": int(counts.get("original", 0)),
        "rpte_source_patterns": int(counts.get("pattern", 0)),
        "rpte_source_augmented": int(counts.get("augmented_pair", 0)),
        "rpte_source_total": int(flow.get("source_feature_count", len(source)) or len(source)),
        "rpte_source_used_in_splits": int(source["used_in_fitted_split"].sum()) if not source.empty else 0,
        "rpte_leaf_terms": int(flow.get("leaf_rule_count", 0) or 0),
        "rpte_direct_source_terms": _direct_source_count(flow),
        "final_lr_terms": int(flow.get("final_term_count", flow.get("coefficient_count", 0)) or 0),
        "rpte_backend": str(flow.get("backend", "")),
        "final_representation": str(flow.get("final_representation", "")),
    }

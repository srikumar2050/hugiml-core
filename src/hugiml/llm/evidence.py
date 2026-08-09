"""Pure evidence-shaping helpers for the optional HUGIML LLM UI.

This module intentionally has no Streamlit dependency so data transformation
logic can be imported and tested in a core development environment.
"""

from __future__ import annotations

from typing import Any

_REDUNDANCY_AUDIT_LABELS = (
    ("input_columns", "Input columns"),
    ("retained_columns", "Retained columns"),
    ("removed_constant_columns", "Removed constants"),
    ("removed_duplicate_columns", "Removed duplicates"),
    ("removed_complementary_columns", "Removed complements"),
    ("vif_columns_above_threshold", "VIF-flagged columns"),
    ("removed_high_vif_pattern_columns", "Removed high-VIF patterns"),
    ("removed_high_vif_augmented_pair_columns", "Removed high-VIF augmented pairs"),
    ("maximum_vif", "Maximum VIF"),
    ("median_vif", "Median VIF"),
    ("vif_threshold", "VIF threshold"),
    ("representation_r2_threshold", "Representation R² threshold"),
)


def downstream_redundancy_audit_rows(model: Any) -> list[dict[str, Any]]:
    """Return the fitted training-only redundancy audit as display rows."""
    method = getattr(model, "get_downstream_redundancy_audit", None)
    if not callable(method):
        return []
    try:
        audit = method(include_feature_names=False)
    except Exception:
        return []
    if not isinstance(audit, dict):
        return []
    return [
        {"measure": label, "value": audit.get(key)}
        for key, label in _REDUNDANCY_AUDIT_LABELS
        if audit.get(key) is not None
    ]


def rpte_rule_rows_to_importance_rows(
    rpte_rows: list[dict[str, Any]], limit: int = 15
) -> list[dict[str, Any]]:
    """Convert RPTE rule-table rows into chart/table importance records.

    Rows are ranked by the absolute final logistic coefficient.  Each leaf's
    split conditions are joined into a readable conjunction compatible with
    the LLM UI's existing importance-chart shape.
    """
    ranked = sorted(
        rpte_rows,
        key=lambda row: abs(row.get("final_logistic_coefficient") or 0.0),
        reverse=True,
    )[:limit]

    output: list[dict[str, Any]] = []
    for row in ranked:
        conditions = row.get("conditions") or []
        terms = [
            condition.get("raw_condition") or condition.get("downstream_condition") or "?"
            for condition in conditions
        ]
        output.append(
            {
                "pattern": " AND ".join(str(term) for term in terms) if terms else "(linear term)",
                "coefficient": row.get("final_logistic_coefficient"),
                "class": row.get("class"),
                "support_count": row.get("support_count"),
                "backend": row.get("backend"),
            }
        )
    return output

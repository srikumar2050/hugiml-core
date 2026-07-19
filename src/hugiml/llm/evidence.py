"""Pure evidence-shaping helpers for the optional HUGIML LLM UI.

This module intentionally has no Streamlit dependency so data transformation
logic can be imported and tested in a core development environment.
"""

from __future__ import annotations

from typing import Any


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
            condition.get("raw_condition")
            or condition.get("downstream_condition")
            or "?"
            for condition in conditions
        ]
        output.append(
            {
                "pattern": " AND ".join(str(term) for term in terms)
                if terms
                else "(linear term)",
                "coefficient": row.get("final_logistic_coefficient"),
                "class": row.get("class"),
                "support_count": row.get("support_count"),
                "backend": row.get("backend"),
            }
        )
    return output

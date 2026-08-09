"""Text rendering helpers for RPTE prediction evidence.

The structured rule APIs remain the source of record.  This module converts
those rows into a compact decision-tree-style view that is suitable for
notebooks, terminals, reports, and the Governance Studio.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

ConditionSpace = Literal["raw", "downstream", "both"]
DetailLevel = Literal["compact", "full"]

_TREE_BACKENDS = {"bounded_lookahead", "sequential_default"}
_DIRECT_BACKENDS = {"direct_hugiml_feature"}
_FALLBACK_BACKENDS = {"raw_hugiml_features"}


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_number(value: Any, *, precision: int, signed: bool = False) -> str:
    number = _as_float(value)
    if number is None:
        return "N/A"
    sign = "+" if signed else ""
    return f"{number:{sign}.{precision}g}"


def _condition_text(condition: Mapping[str, Any], condition_space: ConditionSpace) -> str:
    raw = str(condition.get("raw_condition") or "").strip()
    downstream = str(condition.get("downstream_condition") or "").strip()
    if condition_space == "downstream":
        return downstream or raw or "?"
    if condition_space == "both":
        if raw and downstream and raw != downstream:
            return f"{raw}  [downstream: {downstream}]"
        return raw or downstream or "?"
    return raw or downstream or "?"


def _condition_order(condition: Mapping[str, Any], text: str) -> tuple[int, str]:
    direction = str(condition.get("direction") or "")
    rank = {
        "at_or_below_threshold": 0,
        "above_threshold": 1,
    }.get(direction, 2)
    return rank, text.casefold()


def _backend_label(backend: Any) -> str:
    return {
        "bounded_lookahead": "bounded lookahead",
        "sequential_default": "sequential RPTE",
        "direct_hugiml_feature": "direct source term",
        "raw_hugiml_features": "direct source fallback",
        "constant": "constant representation",
    }.get(str(backend), str(backend or "RPTE"))


def _source_family(row: Mapping[str, Any]) -> str:
    name = str(row.get("downstream_feature") or "")
    if name.startswith("orig:"):
        return "Original features"
    if name.startswith("pattern:"):
        return "HUG patterns"
    if name.startswith("augmented_pair:"):
        return "Augmented pairs"
    conditions = list(row.get("conditions") or [])
    family = str(conditions[0].get("family") or "") if conditions else ""
    return {
        "original": "Original features",
        "pattern": "HUG patterns",
        "augmented_pair": "Augmented pairs",
    }.get(family, "Other source terms")


def _support_text(row: Mapping[str, Any]) -> str | None:
    rate = _as_float(row.get("support_rate"))
    count = row.get("support_count")
    count_text = None
    try:
        if count is not None:
            count_text = f"n={int(count):,}"
    except (TypeError, ValueError):
        count_text = None
    if rate is not None and count_text:
        return f"support={rate:.1%} ({count_text})"
    if rate is not None:
        return f"support={rate:.1%}"
    if count_text:
        return f"support={count_text}"
    return None


def _coefficient_details(
    row: Mapping[str, Any],
    *,
    detail_level: DetailLevel,
    precision: int,
    include_generation_details: bool,
) -> list[str]:
    coefficient = _as_float(row.get("final_logistic_coefficient"))
    details = [f"beta={_format_number(coefficient, precision=precision, signed=True)}"]
    if coefficient is not None:
        odds = math.exp(max(-50.0, min(50.0, coefficient)))
        details.append(f"odds x{_format_number(odds, precision=precision)}")
    else:
        details.append("odds xN/A")

    support = _support_text(row)
    if support:
        details.append(support)

    if detail_level == "full":
        centered = _as_float(row.get("centered_tree_contribution"))
        if centered is not None:
            details.append(
                f"centered={_format_number(centered, precision=precision, signed=True)}"
            )
        sources = [str(value) for value in (row.get("raw_sources") or []) if str(value)]
        if sources:
            details.append("sources=" + ", ".join(sources))

    if include_generation_details:
        newton = _as_float(row.get("newton_leaf_value"))
        if newton is not None:
            details.append(
                f"tree-generation value={_format_number(newton, precision=precision, signed=True)}"
            )
        status = row.get("fallback_status")
        if status:
            details.append(f"backend status={status}")
    return details


def _leaf_annotation(
    row: Mapping[str, Any],
    *,
    detail_level: DetailLevel,
    precision: int,
    include_generation_details: bool,
) -> str:
    details = _coefficient_details(
        row,
        detail_level=detail_level,
        precision=precision,
        include_generation_details=include_generation_details,
    )
    return f"LEAF {row.get('leaf_index')}  |  " + "  |  ".join(details)


def _term_name(row: Mapping[str, Any], condition_space: ConditionSpace) -> str:
    name = str(row.get("downstream_feature") or "").strip()
    conditions = list(row.get("conditions") or [])
    rendered = _condition_text(conditions[0], condition_space) if conditions else ""
    if rendered and rendered != "?":
        return rendered
    return name or f"source column {row.get('leaf_index')}"


def _direct_annotation(
    row: Mapping[str, Any],
    *,
    condition_space: ConditionSpace,
    detail_level: DetailLevel,
    precision: int,
) -> str:
    details = _coefficient_details(
        row,
        detail_level=detail_level,
        precision=precision,
        include_generation_details=False,
    )
    return f"{_term_name(row, condition_space)}  |  " + "  |  ".join(details)


def _new_node() -> dict[str, Any]:
    return {"children": {}, "leaves": []}


def _render_node(
    node: Mapping[str, Any],
    *,
    detail_level: DetailLevel,
    precision: int,
    include_generation_details: bool,
    prefix: str = "",
) -> list[str]:
    children = list(node["children"].values())
    children.sort(key=lambda entry: entry["order"])
    leaves = sorted(
        node["leaves"],
        key=lambda row: (str(row.get("leaf_index")), str(row.get("final_logistic_coefficient"))),
    )
    entries: list[tuple[str, Any]] = [("condition", child) for child in children]
    entries.extend(("leaf", leaf) for leaf in leaves)

    lines: list[str] = []
    for position, (kind, payload) in enumerate(entries):
        is_last = position == len(entries) - 1
        connector = "└── " if is_last else "├── "
        continuation = "    " if is_last else "│   "
        if kind == "condition":
            lines.append(prefix + connector + payload["text"])
            lines.extend(
                _render_node(
                    payload["node"],
                    detail_level=detail_level,
                    precision=precision,
                    include_generation_details=include_generation_details,
                    prefix=prefix + continuation,
                )
            )
        else:
            lines.append(
                prefix
                + connector
                + _leaf_annotation(
                    payload,
                    detail_level=detail_level,
                    precision=precision,
                    include_generation_details=include_generation_details,
                )
            )
    return lines


def rpte_rule_tree_sections(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition_space: ConditionSpace = "raw",
    detail_level: DetailLevel = "full",
    precision: int = 5,
    include_generation_details: bool = False,
    class_label: Any | None = None,
    tree_index: int | None = None,
) -> list[dict[str, Any]]:
    """Build one flat-tree section per RPTE class and tree.

    The returned dictionaries contain ``title``, ``text``, tree metadata, and
    ``coefficients``.  ``format_rpte_rule_tree`` is the simpler text API.
    """
    if condition_space not in {"raw", "downstream", "both"}:
        raise ValueError("condition_space must be 'raw', 'downstream', or 'both'")
    if detail_level not in {"compact", "full"}:
        raise ValueError("detail_level must be 'compact' or 'full'")
    if precision < 1:
        raise ValueError("precision must be at least 1")

    grouped: dict[tuple[Any, Any, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        backend = str(row.get("backend") or "")
        if backend not in _TREE_BACKENDS:
            continue
        if class_label is not None and row.get("class") != class_label:
            continue
        if tree_index is not None and row.get("tree_index") != tree_index:
            continue
        key = (row.get("class"), row.get("tree_index"), backend)
        grouped.setdefault(key, []).append(row)

    sections: list[dict[str, Any]] = []
    for (class_value, tree_value, backend), members in sorted(
        grouped.items(),
        key=lambda item: (str(item[0][0]), str(item[0][1]), item[0][2]),
    ):
        root = _new_node()
        max_depth = 0
        for row in members:
            node = root
            conditions = list(row.get("conditions") or [])
            max_depth = max(max_depth, len(conditions))
            for condition in conditions:
                text = _condition_text(condition, condition_space)
                child = node["children"].setdefault(
                    text,
                    {
                        "text": text,
                        "order": _condition_order(condition, text),
                        "node": _new_node(),
                    },
                )
                node = child["node"]
            node["leaves"].append(row)

        coefficient_rows = []
        for row in members:
            coefficient = _as_float(row.get("final_logistic_coefficient"))
            odds = (
                math.exp(max(-50.0, min(50.0, coefficient)))
                if coefficient is not None
                else None
            )
            coefficient_rows.append(
                {
                    "leaf": row.get("leaf_index"),
                    "coefficient": coefficient,
                    "odds_multiplier": odds,
                    "centered_tree_contribution": _as_float(
                        row.get("centered_tree_contribution")
                    ),
                    "support_rate": _as_float(row.get("support_rate")),
                    "support_count": row.get("support_count"),
                    "n_conditions": len(row.get("conditions") or []),
                    "raw_sources": ", ".join(
                        str(value) for value in (row.get("raw_sources") or [])
                    ),
                    "_source_row_index": row.get("_source_row_index"),
                }
            )
        coefficient_rows.sort(
            key=lambda item: (
                -(abs(item["coefficient"]) if item["coefficient"] is not None else -1.0),
                str(item["leaf"]),
            )
        )

        title = f"Class {class_value} | Tree {tree_value} | {_backend_label(backend)}"
        text = "\n".join(
            ["ROOT"]
            + _render_node(
                root,
                detail_level=detail_level,
                precision=precision,
                include_generation_details=include_generation_details,
            )
        )
        sections.append(
            {
                "title": title,
                "text": text,
                "class": class_value,
                "tree": tree_value,
                "backend": backend,
                "leaf_count": len(members),
                "max_depth": max_depth,
                "coefficients": coefficient_rows,
            }
        )
    return sections


def _render_grouped_terms(
    rows: Sequence[Mapping[str, Any]],
    *,
    heading: str,
    condition_space: ConditionSpace,
    detail_level: DetailLevel,
    precision: int,
) -> str:
    families: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        families.setdefault(_source_family(row), []).append(row)
    family_order = ["Original features", "HUG patterns", "Augmented pairs", "Other source terms"]
    ordered = [(family, families[family]) for family in family_order if family in families]

    lines = [heading]
    for family_position, (family, members) in enumerate(ordered):
        family_last = family_position == len(ordered) - 1
        family_connector = "└── " if family_last else "├── "
        family_prefix = "    " if family_last else "│   "
        lines.append(family_connector + family)
        members = sorted(
            members,
            key=lambda row: (
                -abs(_as_float(row.get("final_logistic_coefficient")) or 0.0),
                _term_name(row, condition_space),
            ),
        )
        for term_position, row in enumerate(members):
            term_last = term_position == len(members) - 1
            term_connector = "└── " if term_last else "├── "
            lines.append(
                family_prefix
                + term_connector
                + _direct_annotation(
                    row,
                    condition_space=condition_space,
                    detail_level=detail_level,
                    precision=precision,
                )
            )
    return "\n".join(lines)


def format_rpte_rule_tree(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition_space: ConditionSpace = "raw",
    detail_level: DetailLevel = "full",
    precision: int = 5,
    include_direct_terms: bool = True,
    include_generation_details: bool = False,
    class_label: Any | None = None,
    tree_index: int | None = None,
) -> str:
    """Return RPTE prediction evidence as ready-to-print flat trees.

    Shared condition prefixes are merged, one split is shown per indentation
    level, and each terminal leaf reports its fitted LR coefficient, odds
    multiplier, support, centered contribution, and raw-feature mapping.
    Direct source terms are grouped by original, HUG-pattern, and augmented-pair
    families after the tree sections.

    Example::

        print(model.rpte_rule_tree())
    """
    sections = rpte_rule_tree_sections(
        rows,
        condition_space=condition_space,
        detail_level=detail_level,
        precision=precision,
        include_generation_details=include_generation_details,
        class_label=class_label,
        tree_index=tree_index,
    )
    blocks = [f"{section['title']}\n{section['text']}" for section in sections]

    filtered_rows = [
        row
        for row in rows
        if (class_label is None or row.get("class") == class_label)
        and (tree_index is None or row.get("tree_index") == tree_index)
    ]
    if include_direct_terms and tree_index is None:
        direct_rows = [
            row
            for row in filtered_rows
            if str(row.get("backend") or "") in _DIRECT_BACKENDS
            or str(row.get("term_role") or "") == "direct_source_term"
        ]
        direct_by_class: dict[Any, list[Mapping[str, Any]]] = {}
        for row in direct_rows:
            direct_by_class.setdefault(row.get("class"), []).append(row)
        for class_value, members in sorted(direct_by_class.items(), key=lambda item: str(item[0])):
            blocks.append(
                f"Class {class_value} | Direct source terms\n"
                + _render_grouped_terms(
                    members,
                    heading="DIRECT SOURCE TERMS",
                    condition_space=condition_space,
                    detail_level=detail_level,
                    precision=precision,
                )
            )

    fallback_rows = [
        row for row in filtered_rows if str(row.get("backend") or "") in _FALLBACK_BACKENDS
    ]
    fallback_by_class: dict[Any, list[Mapping[str, Any]]] = {}
    for row in fallback_rows:
        fallback_by_class.setdefault(row.get("class"), []).append(row)
    for class_value, members in sorted(fallback_by_class.items(), key=lambda item: str(item[0])):
        blocks.append(
            f"Class {class_value} | Direct source fallback\n"
            + _render_grouped_terms(
                members,
                heading="FINAL LR SOURCE TERMS",
                condition_space=condition_space,
                detail_level=detail_level,
                precision=precision,
            )
        )

    constant_rows = [
        row for row in filtered_rows if str(row.get("backend") or "") == "constant"
    ]
    for row in constant_rows:
        details = _coefficient_details(
            row,
            detail_level=detail_level,
            precision=precision,
            include_generation_details=include_generation_details,
        )
        blocks.append(
            f"Class {row.get('class')} | Constant representation\n"
            + "CONSTANT  |  "
            + "  |  ".join(details)
        )

    return "\n\n".join(blocks)


__all__ = ["format_rpte_rule_tree", "rpte_rule_tree_sections"]

"""Display-only formatting for the public governance views."""

from __future__ import annotations

import math
import re


def number(value):
    if isinstance(value, bool):
        return str(value)
    if not math.isfinite(value):
        return "—"
    if isinstance(value, int) or value.is_integer():
        return f"{int(value):,}"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:,.3g}"


def text(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return number(value)
    if isinstance(value, str):
        # Restrict replacement to complete numeric text nodes, never rule expressions or identifiers.
        if re.fullmatch(r"-?\d+\.\d{4,}(?:[eE][+-]?\d+)?", value):
            return number(float(value))
    return {"validation_roc_auc": "ROC-AUC"}.get(value, value) if isinstance(value, str) else value


def present(node):
    if isinstance(node, list):
        return [present(item) for item in node]
    if not isinstance(node, dict):
        return node
    if "props" not in node or "type" not in node:
        return {key: present(value) for key, value in node.items()}
    props = dict(node["props"])
    kind = node["type"]
    if kind == "DataTable":
        records = props.get("data", [])
        columns = []
        for column in props.get("columns", []):
            column = dict(column)
            values = [row.get(column["id"]) for row in records]
            values = [value for value in values if value is not None and value != ""]
            if values and all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
            ):
                integer = all(isinstance(value, int) for value in values)
                column.update(type="numeric", format={"specifier": ",d" if integer else ".3~g"})
            columns.append(column)
        props["columns"] = columns
        props["style_header"] = {
            **props.get("style_header", {}),
            "backgroundColor": "#edf3f7",
            "color": "#516579",
            "fontWeight": "600",
            "fontSize": "12px",
            "border": "0",
            "borderBottom": "1px solid #dce6ed",
        }
        props["style_cell"] = {
            **props.get("style_cell", {}),
            "fontFamily": "inherit",
            "fontSize": "14px",
            "padding": "12px 14px",
            "color": "#243b50",
            "border": "0",
            "borderBottom": "1px solid #edf2f6",
        }
        if props.get("id") == "pf-cv":
            compact = dict(props)
            labels = {
                "fold": "Fold",
                "validation_roc_auc": "Validation ROC-AUC",
                "status": "Status",
            }
            compact["columns"] = [
                {**column, "name": labels[column["id"]]}
                for column in columns
                if column["id"] in labels
            ]
            compact["style_table"] = {"overflowX": "auto"}
            detailed = {**props, "id": "pf-cv-settings"}
            return {
                "type": "Div",
                "namespace": "dash_html_components",
                "props": {
                    "children": [
                        {**node, "props": compact},
                        {
                            "type": "Details",
                            "namespace": "dash_html_components",
                            "props": {
                                "className": "detail-table",
                                "children": [
                                    {
                                        "type": "Summary",
                                        "namespace": "dash_html_components",
                                        "props": {
                                            "children": "Configuration details for each fold"
                                        },
                                    },
                                    {**node, "props": detailed},
                                ],
                            },
                        },
                    ]
                },
            }
        return {**node, "props": props}
    if kind == "Graph":
        figure = props.get("figure", {})
        layout = figure.setdefault("layout", {})
        layout.update(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"family": "Inter, system-ui, sans-serif", "color": "#516579", "size": 14},
        )
        for key in list(layout) + ["xaxis", "yaxis"]:
            if re.fullmatch(r"[xy]axis\d*", key):
                axis = layout.setdefault(key, {})
                axis.setdefault("tickformat", ".3~g")
                axis.setdefault("hoverformat", ".3~g")
                axis.update(gridcolor="#eef3f6", zerolinecolor="#dce6ed")
        props["figure"] = figure
        return {**node, "props": props}
    children = props.get("children")
    if kind != "Pre":
        if isinstance(children, list):
            props["children"] = [
                present(child) if isinstance(child, (dict, list)) else text(child)
                for child in children
            ]
        elif isinstance(children, (dict, list)):
            props["children"] = present(children)
        elif "children" in props:
            props["children"] = text(children)
    return {**node, "props": props}


def format_pages(pages):
    """Use raw evidence for headline metrics to avoid rounding an already rounded label."""
    scores = {}

    def collect(node):
        if isinstance(node, list):
            for child in node:
                collect(child)
        elif isinstance(node, dict):
            props = node.get("props", {})
            if node.get("type") == "DataTable":
                if props.get("id") == "pf-met":
                    metrics = {row["metric"]: row["value"] for row in props["data"]}
                    for label, key in (("ROC-AUC", "roc_auc"), ("Brier", "brier_score")):
                        if key in metrics:
                            scores[label] = metrics[key]
                elif props.get("id") == "pf-cv":
                    values = [
                        row["validation_roc_auc"]
                        for row in props["data"]
                        if isinstance(row.get("validation_roc_auc"), (int, float))
                    ]
                    if values:
                        scores["Mean CV ROC-AUC"] = scores["Best CV ROC-AUC"] = sum(values) / len(
                            values
                        )
            for value in node.values():
                collect(value)

    def restore(node):
        if isinstance(node, list):
            return [restore(child) for child in node]
        if not isinstance(node, dict):
            return node
        node = {key: restore(value) for key, value in node.items()}
        props = node.get("props", {})
        children = props.get("children")
        if props.get("className") == "mc" and isinstance(children, list) and len(children) == 2:
            label = children[0].get("props", {}).get("children")
            if label in scores:
                children[1]["props"]["children"] = scores[label]
        return node

    collect(pages)
    return present(restore(pages))

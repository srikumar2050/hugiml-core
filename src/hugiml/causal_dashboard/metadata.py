"""Validation and rendering helpers for optional causal metadata JSON."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any

import plotly.graph_objects as go

from hugiml.dashboard.dash_components.styles import get_tokens

_ALLOWED_ROLES = {
    "treatment",
    "outcome",
    "confounder",
    "mediator",
    "collider",
    "instrument",
    "effect modifier",
    "precision/effect modifier",
    "precision",
    "unknown",
}


def validate_causal_metadata(
    meta: dict[str, Any], columns: Iterable[str] | None = None
) -> dict[str, Any]:
    if not isinstance(meta, dict):
        raise ValueError("Causal metadata must be a JSON object.")
    out = dict(meta)
    for key in ("treatment", "outcome"):
        value = out.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Metadata field '{key}' must be a non-empty string.")
    if out["treatment"] == out["outcome"]:
        raise ValueError("Metadata treatment and outcome must be different columns.")
    adj = out.get("adjustment_set", [])
    if not isinstance(adj, list) or not all(isinstance(v, str) for v in adj):
        raise ValueError("adjustment_set must be a list of column names.")
    if len(adj) != len(set(adj)):
        raise ValueError("adjustment_set must not contain duplicate columns.")
    if out["treatment"] in adj or out["outcome"] in adj:
        raise ValueError("Treatment/outcome cannot be members of adjustment_set.")

    nodes = out.get("nodes", [])
    edges = out.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("nodes and edges must be lists.")
    ids = []
    treatment_nodes = 0
    outcome_nodes = 0
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise ValueError("Each DAG node requires a string id.")
        if node["id"] in ids:
            raise ValueError(f"Duplicate DAG node id: {node['id']}")
        role = str(node.get("role", "unknown")).lower()
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"Unsupported DAG role '{node.get('role')}' for node {node['id']}.")
        treatment_nodes += int(role == "treatment")
        outcome_nodes += int(role == "outcome")
        ids.append(node["id"])
    if nodes and (treatment_nodes != 1 or outcome_nodes != 1):
        raise ValueError(
            "A supplied DAG must contain exactly one treatment node and one outcome node."
        )
    if nodes:
        role_by_id = {str(node["id"]): str(node.get("role", "unknown")).lower() for node in nodes}
        if role_by_id.get(out["treatment"]) != "treatment":
            raise ValueError("The declared treatment must be the DAG node with role 'treatment'.")
        if role_by_id.get(out["outcome"]) != "outcome":
            raise ValueError("The declared outcome must be the DAG node with role 'outcome'.")
    idset = set(ids)
    normalized_edges: list[tuple[str, str]] = []
    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError("Each DAG edge must be [source, target].")
        source, target = str(edge[0]), str(edge[1])
        if source not in idset or target not in idset:
            raise ValueError(f"DAG edge {source}->{target} references an unknown node.")
        if source == target:
            raise ValueError("DAG self-loops are not allowed.")
        if (source, target) in normalized_edges:
            raise ValueError(f"Duplicate DAG edge: {source}->{target}")
        normalized_edges.append((source, target))
    _topological_layers(ids, normalized_edges)

    if columns is not None:
        cset = {str(c) for c in columns}
        required = {out["treatment"], out["outcome"], *adj}
        missing = sorted(required - cset)
        if missing:
            raise ValueError(
                "Metadata references columns not present in the dataset: " + ", ".join(missing)
            )
    out["edges"] = [list(edge) for edge in normalized_edges]
    return out


def _topological_layers(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    indeg = {node: 0 for node in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, list[str]] = defaultdict(list)
    for source, target in edges:
        indeg[target] += 1
        children[source].append(target)
        parents[target].append(source)
    queue = deque(sorted([node for node, degree in indeg.items() if degree == 0]))
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children[node]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    if len(order) != len(nodes):
        raise ValueError("DAG metadata contains a directed cycle.")
    layer: dict[str, int] = {}
    for node in order:
        layer[node] = 0 if not parents[node] else 1 + max(layer[parent] for parent in parents[node])
    return layer


def _role_family(role: str) -> str:
    role = role.lower()
    if role in {"confounder", "precision", "precision/effect modifier", "effect modifier"}:
        return "baseline"
    return role


def _compact_graph(nodes: list[dict[str, Any]], edges: list[tuple[str, str]]):
    node_map = {str(node["id"]): dict(node) for node in nodes}
    parents: dict[str, set[str]] = defaultdict(set)
    children: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        parents[target].add(source)
        children[source].add(target)

    groups: dict[tuple[str, tuple[str, ...], tuple[str, ...]], list[str]] = defaultdict(list)
    for node_id, node in node_map.items():
        role = str(node.get("role", "unknown")).lower()
        if role in {"treatment", "outcome", "mediator", "collider", "instrument"}:
            continue
        key = (
            _role_family(role),
            tuple(sorted(parents[node_id])),
            tuple(sorted(children[node_id])),
        )
        groups[key].append(node_id)

    replacement = {node_id: node_id for node_id in node_map}
    compact_nodes = dict(node_map)
    for group_no, members in enumerate(groups.values()):
        if len(members) < 3:
            continue
        group_id = f"__baseline_group_{group_no}"
        labels = [str(node_map[node_id].get("label", node_id)) for node_id in members]
        compact_nodes[group_id] = {
            "id": group_id,
            "label": f"Baseline covariates ({len(members)})",
            "role": "baseline covariates",
            "hover_label": "Baseline covariates: " + ", ".join(labels),
        }
        for node_id in members:
            replacement[node_id] = group_id
            compact_nodes.pop(node_id, None)

    compact_edges = []
    seen = set()
    for source, target in edges:
        edge = (replacement[source], replacement[target])
        if edge[0] == edge[1] or edge in seen:
            continue
        seen.add(edge)
        compact_edges.append(edge)
    return list(compact_nodes.values()), compact_edges


def dag_figure(meta: dict[str, Any], *, theme: str = "Ocean", height: int = 270) -> go.Figure:
    tokens = get_tokens(theme)
    nodes = list(meta.get("nodes", []))
    edges = [tuple(edge) for edge in meta.get("edges", [])]
    if not nodes:
        fig = go.Figure()
        fig.add_annotation(
            text="No causal graph metadata supplied.",
            showarrow=False,
            font=dict(color=tokens["mu"], size=12),
        )
        fig.update_layout(
            height=height,
            margin=dict(l=10, r=10, t=12, b=10),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    display_nodes, display_edges = _compact_graph(nodes, edges)
    ids = [str(node["id"]) for node in display_nodes]
    layers = _topological_layers(ids, display_edges)
    by_layer: dict[int, list[str]] = defaultdict(list)
    for node_id in ids:
        by_layer[layers[node_id]].append(node_id)

    max_layer = max(by_layer) if by_layer else 0
    pos: dict[str, tuple[float, float]] = {}
    for layer_no, members in sorted(by_layer.items()):
        ordered = sorted(
            members,
            key=lambda node_id: str(
                next(n for n in display_nodes if str(n["id"]) == node_id).get("label", node_id)
            ),
        )
        for index, node_id in enumerate(ordered):
            if len(ordered) == 1:
                y = 0.5
            else:
                span = 0.62
                y = 0.5 + span / 2 - index * span / (len(ordered) - 1)
            x = 0.08 + 0.84 * (layer_no / max(1, max_layer))
            pos[node_id] = (x, y)

    fig = go.Figure()
    for source, target in display_edges:
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.9,
            arrowwidth=1.3,
            arrowcolor=tokens["mu"],
            opacity=0.78,
            standoff=18,
            startstandoff=18,
        )

    symbol = {
        "treatment": "diamond",
        "outcome": "square",
        "confounder": "circle",
        "effect modifier": "circle",
        "precision/effect modifier": "circle",
        "precision": "circle",
        "mediator": "triangle-right",
        "collider": "x",
        "instrument": "triangle-up",
    }
    fill = {
        "treatment": tokens["a"],
        "outcome": tokens["a2"],
        "confounder": tokens["cb"],
        "effect modifier": tokens["cb"],
        "precision/effect modifier": tokens["cb"],
        "precision": tokens["cb"],
        "mediator": tokens["am"],
        "collider": tokens["rd"],
        "instrument": tokens["gr"],
    }
    for node in display_nodes:
        node_id = str(node["id"])
        x, y = pos[node_id]
        role = str(node.get("role", "unknown")).lower()
        label = str(node.get("label", node_id))
        hover = str(node.get("hover_label", label)).replace("<br>", "<br>• ")
        if hover != label:
            hover = "• " + hover
        marker_fill = fill.get(role, tokens["sf"])
        text_color = "#ffffff" if role in {"treatment", "outcome"} else tokens["tx"]
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                text=[label],
                textposition="bottom center",
                textfont=dict(color=tokens["tx"], size=11),
                marker=dict(
                    size=25,
                    symbol=symbol.get(role, "circle"),
                    color=marker_fill,
                    line=dict(
                        width=1.8, color=tokens["a"] if role == "treatment" else tokens["bd"]
                    ),
                ),
                customdata=[[role, hover]],
                hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
                showlegend=False,
            )
        )
        if role in {"treatment", "outcome"}:
            fig.add_annotation(
                x=x,
                y=y,
                text="T" if role == "treatment" else "Y",
                showarrow=False,
                font=dict(color=text_color, size=10),
            )

    fig.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=8, b=34),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0.02, 0.98]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=tokens["tx"]),
        hoverlabel=dict(bgcolor=tokens["sf"], font_color=tokens["tx"], bordercolor=tokens["bd"]),
    )
    return fig

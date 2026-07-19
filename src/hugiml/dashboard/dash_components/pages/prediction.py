"""Case Review page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from hugiml.dashboard.components.prediction import (
    _activated_patterns,
    _row_to_feature_table,
    _rpte_active_rules_for_case,
    _safe_predict,
    _safe_predict_proba,
    _top_case_values,
)
from hugiml.dashboard.dash_components.pages._shared import info, mc, sn
from hugiml.dashboard.dash_components.tables import make_table


def render_case(ctx, idx=0):
    model = ctx.get("model")
    X = ctx.get("X")
    cids = ctx.get("case_ids")
    if model is None or X is None or len(X) == 0:
        return info("No model/data.")
    idx = max(0, min(int(idx), len(X) - 1))
    try:
        cid = str(cids.iloc[idx] if hasattr(cids, "iloc") else cids[idx])
    except Exception:
        cid = f"CASE-{idx + 1:06d}"
    row = X.iloc[[idx]]
    pred, pe = _safe_predict(model, row)
    proba, pre = _safe_predict_proba(model, row)
    metrics = dbc.Row(
        [
            dbc.Col(mc("Case ID", cid), md=4, className="mb-3"),
            dbc.Col(
                mc("Predicted Class", str(pred) if pred is not None else "N/A"),
                md=4,
                className="mb-3",
            ),
            dbc.Col(
                mc("Probability", f"{proba:.4f}" if proba is not None else "N/A"),
                md=4,
                className="mb-3",
            ),
        ]
    )
    top = _top_case_values(row)
    all_vals = _row_to_feature_table(row)
    active = _activated_patterns(model, row)
    rpte = _rpte_active_rules_for_case(model, row)
    tabs = [
        dbc.Tab(
            make_table(top, tid="cs-top", height="300px", sortable=False),
            label="Case Summary",
            tab_id="case-summary",
        ),
        dbc.Tab(
            make_table(all_vals, tid="cs-all", height="400px"),
            label="All Inputs",
            tab_id="case-inputs",
        ),
        dbc.Tab(
            make_table(active, tid="cs-pat", height="300px")
            if not active.empty
            else info("Pattern activation not exposed."),
            label="Pattern Evidence",
            tab_id="case-patterns",
        ),
    ]
    if not rpte.empty:
        total = float(rpte["coefficient"].sum()) if "coefficient" in rpte.columns else 0
        tabs.append(
            dbc.Tab(
                html.Div(
                    [
                        make_table(rpte, tid="cs-rpte", height="300px"),
                        mc("Σ Coefficients", f"{total:.4f}"),
                    ]
                ),
                label="RPTE Rule",
                tab_id="case-rpte",
            )
        )
    return html.Div([metrics, dbc.Tabs(tabs, id="case-review-tabs", active_tab="case-summary")])


def render(ctx, row_idx=0):
    X = ctx.get("X")
    if X is None or len(X) == 0:
        return info("No data loaded.")
    mx = max(len(X) - 1, 0)
    return html.Div(
        [
            sn(
                "Case-level traceability: prediction, probability, input values, active patterns, RPTE leaf conjunctions."
            ),
            html.Div(
                [
                    html.Label("Row index", className="form-label"),
                    dcc.Input(
                        id="case-idx",
                        type="number",
                        value=row_idx,
                        min=0,
                        max=mx,
                        step=1,
                        style={
                            "width": "120px",
                            "fontSize": ".84rem",
                            "padding": "4px 8px",
                            "borderRadius": "6px",
                            "border": "1px solid var(--bs)",
                        },
                    ),
                ],
                className="mb-3",
                style={"maxWidth": "200px"},
            ),
            html.Div(id="case-content", children=render_case(ctx, row_idx)),
        ]
    )

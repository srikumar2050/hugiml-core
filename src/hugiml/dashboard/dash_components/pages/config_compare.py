"""Configuration Comparison page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from hugiml.dashboard.components.config_compare import _result_row
from hugiml.dashboard.dash_components.charts import bar_h
from hugiml.dashboard.dash_components.pages._shared import err, info, sn, warn
from hugiml.dashboard.dash_components.tables import make_table

try:
    from hugiml.rpte_bounded_lookahead_leafwise import LEAF_CONFIGS as RLC

    _RPTE = True
except Exception:
    _RPTE = False
    RLC = {}


def rpte_controls(show):
    if not show or not _RPTE:
        return html.Div()
    lo = [{"label": k, "value": k} for k in sorted(RLC)]
    dl = "3xD" if "3xD" in RLC else next(iter(RLC), "")
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6("RPTE Settings", className="fw-semibold mb-2"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Leaf Config", className="form-label"),
                                dcc.Dropdown(
                                    id="cc-rpte-lc", options=lo, value=dl, clearable=False
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Depth", className="form-label"),
                                dcc.Input(
                                    id="cc-rpte-d",
                                    type="number",
                                    value=4,
                                    min=1,
                                    max=10,
                                    className="form-control form-control-sm",
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("n_estimators", className="form-label"),
                                dcc.Input(
                                    id="cc-rpte-ne",
                                    type="number",
                                    value=10,
                                    min=1,
                                    max=100,
                                    className="form-control form-control-sm",
                                ),
                            ],
                            md=4,
                        ),
                    ]
                ),
            ]
        ),
        className="mb-3",
    )


def render_result(ctx, cand):
    if cand is None:
        return info("Fit a candidate to compare.")
    status = getattr(cand, "status_", "ok")
    blocks = []
    if status != "ok":
        blocks.append(err("Candidate did not produce a fitted model."))
        if status == "no_patterns":
            blocks.append(warn("Zero patterns mined. Adjust G, topK, or L."))
    rows = [
        _result_row("Current model", ctx["result"], ctx["X"]),
        _result_row("Candidate", cand, ctx["X"]),
    ]
    frame = pd.DataFrame(rows)
    if "validation_roc_auc" in frame and frame["validation_roc_auc"].notna().all():
        frame["auc_delta"] = [
            0.0,
            float(frame.loc[1, "validation_roc_auc"]) - float(frame.loc[0, "validation_roc_auc"]),
        ]

    comparison_table = make_table(frame, tid="cc-comp", height="255px")
    if "validation_roc_auc" in frame and frame["validation_roc_auc"].notna().sum() >= 1:
        figure = bar_h(
            frame["validation_roc_auc"].fillna(0).tolist(),
            frame["run"].tolist(),
            title="CV ROC-AUC",
            color=["#AFA9EC", "#534AB7"],
            xlabel="AUC",
            h=225,
        )
        comparison = dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dcc.Graph(figure=figure, config={"displayModeBar": False}),
                        className="results-side-card config-compare-chart",
                    ),
                    md=5,
                    className="mb-3",
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.H6("Configuration evidence", className="results-card-title"),
                            comparison_table,
                        ],
                        className="results-side-card",
                    ),
                    md=7,
                    className="mb-3",
                ),
            ]
        )
    else:
        comparison = comparison_table

    fold = pd.DataFrame(getattr(cand, "results_", []))
    fold_section = []
    if not fold.empty:
        fold_section = [
            html.H6("Candidate Fold Evidence", className="fw-semibold mb-2 mt-3"),
            make_table(fold, tid="cc-fold", height="240px"),
        ]
    return html.Div(blocks + [comparison] + fold_section)


def render(ctx, cand_result=None):
    model = ctx.get("model")
    current_params = getattr(ctx.get("result"), "best_params_", {}) or {}
    L = current_params.get("L", getattr(model, "L", 2) or 2)
    tk = int(current_params.get("topK", getattr(model, "topK", 50) or 50))
    current_G = float(current_params.get("G", getattr(model, "G", 0.01) or 0.01))
    current_mode = current_params.get(
        "feature_mode", getattr(model, "feature_mode", "original_plus_patterns")
    )
    current_strict = bool(
        current_params.get("topk_budget_strict", getattr(model, "topk_budget_strict", False))
    )
    ds_opts = [{"label": "HUGIML logistic regression", "value": "lr"}]
    if _RPTE:
        ds_opts.append({"label": "RPTE (bounded-lookahead)", "value": "rpte"})
    form = dbc.Card(
        dbc.CardBody(
            [
                html.H6("Candidate Configuration", className="fw-bold mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("L", className="form-label"),
                                dcc.Dropdown(
                                    id="cc-L",
                                    options=[
                                        {"label": "1", "value": 1},
                                        {"label": "2", "value": 2},
                                    ],
                                    value=min(2, int(L)),
                                    clearable=False,
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("topK", className="form-label"),
                                dcc.Input(
                                    id="cc-topK",
                                    type="text",
                                    value=str(tk),
                                    inputMode="numeric",
                                    pattern="[0-9]+",
                                    debounce=True,
                                    className="form-control form-control-sm",
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("G", className="form-label"),
                                dcc.Dropdown(
                                    id="cc-G",
                                    options=[
                                        {"label": str(v), "value": v}
                                        for v in [0.1, 0.01, 0.001, 0.0001]
                                    ],
                                    value=current_G
                                    if current_G in {0.1, 0.01, 0.001, 0.0001}
                                    else 0.01,
                                    clearable=False,
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("feature_mode", className="form-label"),
                                dcc.Dropdown(
                                    id="cc-fm",
                                    options=[
                                        {"label": v, "value": v}
                                        for v in [
                                            "patterns_only",
                                            "original_plus_patterns",
                                            "original_plus_interactions",
                                        ]
                                    ],
                                    value=current_mode,
                                    clearable=False,
                                ),
                            ],
                            md=8,
                        ),
                        dbc.Col(
                            [
                                html.Label("Strict Budget", className="form-label"),
                                dbc.Checklist(
                                    id="cc-strict",
                                    options=[{"label": "enabled", "value": True}],
                                    value=[True] if current_strict else [],
                                    switch=True,
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="mb-3",
                ),
                html.Label("Downstream", className="form-label"),
                dcc.RadioItems(
                    id="cc-downstream",
                    options=ds_opts,
                    value="lr",
                    inline=True,
                    inputStyle={"marginRight": "5px", "marginLeft": "12px"},
                    className="mb-2",
                ),
                html.Div(id="cc-rpte-box"),
                html.Button("Fit Candidate", id="cc-run", n_clicks=0, className="btn-fit mt-2"),
            ]
        ),
        className="mb-4",
    )
    return html.Div(
        [
            sn("Fit a different HUGIML configuration and compare."),
            form,
            html.Div(id="cc-result-box", children=render_result(ctx, cand_result)),
        ]
    )

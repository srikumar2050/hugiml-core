"""Overview page."""

from __future__ import annotations

import json

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from hugiml.dashboard.components.overview import evidence_status_frame, fit_metadata_frame
from hugiml.dashboard.dash_components.charts import timing_bar
from hugiml.dashboard.dash_components.pages._shared import info, mc, sn
from hugiml.dashboard.dash_components.tables import make_table
from hugiml.serialization import generate_sbom


def render(ctx):
    model = ctx.get("model")
    result = ctx.get("result")
    meta = ctx.get("meta", {})
    roles = ctx.get("roles", {})
    bs = getattr(result, "best_score_", None) if result else None
    pr = meta.get("positive_rate")
    metrics = dbc.Row(
        [
            dbc.Col(
                mc(
                    "Rows Reviewed",
                    f"{meta.get('n_rows', 0):,}" if isinstance(meta.get("n_rows"), int) else "N/A",
                ),
                md=3,
                className="mb-3",
            ),
            dbc.Col(
                mc(
                    "Model Features",
                    f"{meta.get('n_features', 0):,}"
                    if isinstance(meta.get("n_features"), int)
                    else "N/A",
                ),
                md=3,
                className="mb-3",
            ),
            dbc.Col(
                mc("Best CV ROC-AUC", f"{bs:.4f}" if isinstance(bs, float) else "N/A"),
                md=3,
                className="mb-3",
            ),
            dbc.Col(
                mc("Positive Rate", f"{pr:.2%}" if isinstance(pr, (int, float)) else "N/A"),
                md=3,
                className="mb-3",
            ),
        ]
    )
    # Workflow guide
    wf = html.Div(
        [
            html.Div(
                [html.Span(n, className="wf-num"), html.B(t), html.Span(d)], className="wf-step"
            )
            for n, t, d in [
                ("1", "Model overview", "Confirm roles, parameters, and score."),
                (
                    "2",
                    "Evidence inspection",
                    "Validation, representation, patterns, cases, quality.",
                ),
                (
                    "3",
                    "Governance actions",
                    "Sensitive lineage, rebuilds, config comparison, drift.",
                ),
            ]
        ],
        className="wf-grid",
    )
    # Evidence status
    ev_df = evidence_status_frame(model, result, ctx.get("X"), ctx.get("y"))
    # Fit metadata
    fit_df = fit_metadata_frame(model) if model else pd.DataFrame()
    fit_children = []
    if not fit_df.empty:
        fm = getattr(model, "fit_metadata_", None)
        st = (
            fm.get("stage_times_ms")
            if isinstance(fm, dict)
            else getattr(fm, "stage_times_ms", None)
        )
        if isinstance(st, dict) and st:
            fit_children.append(
                dcc.Graph(
                    figure=timing_bar(list(st.keys()), [float(v) for v in st.values()]),
                    config={"displayModeBar": False},
                )
            )
        fit_children.append(make_table(fit_df, tid="ov-fit", height="260px", page_size=15))
    else:
        fit_children = [info("fit_metadata_ is not available.")]
    # SBOM
    try:
        sbom = generate_sbom()
        sbom_j = json.dumps(sbom, indent=2)
        sbom_c = [
            html.Pre(
                sbom_j,
                style={
                    "fontSize": ".72rem",
                    "maxHeight": "240px",
                    "overflowY": "auto",
                    "background": "var(--cb)",
                    "borderRadius": "6px",
                    "padding": "10px",
                },
            ),
            html.A(
                dbc.Button("Download SBOM", size="sm", color="secondary", className="mt-2"),
                href="data:application/json;charset=utf-8," + sbom_j,
                download="hugiml_sbom.json",
            ),
        ]
    except Exception as e:
        sbom_c = [info(f"SBOM error: {e}")]
    # Column roles
    role_rows = [
        {"role": "Target", "columns": roles.get("target") or "N/A"},
        {"role": "ID", "columns": roles.get("id_column") or "Generated"},
        {"role": "Excluded", "columns": ", ".join(roles.get("excluded_columns", [])) or "None"},
        {
            "role": "Sensitive/Proxy",
            "columns": ", ".join(roles.get("sensitive_columns", [])) or "None",
        },
    ]
    # Config
    cfg = [
        {
            "parameter": "L",
            "value": str(getattr(model, "L", "N/A")),
            "meaning": "Max pattern order",
        },
        {
            "parameter": "topK",
            "value": str(getattr(model, "topK", "N/A")),
            "meaning": "Pattern budget",
        },
        {"parameter": "G", "value": str(getattr(model, "G", "N/A")), "meaning": "Mining threshold"},
        {
            "parameter": "feature_mode",
            "value": str(getattr(model, "feature_mode", "N/A")),
            "meaning": "Representation regime",
        },
    ]
    return html.Div(
        [
            sn(
                "Governance summary of configuration, column roles, fit metadata, and available evidence."
            ),
            wf,
            metrics,
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H6("Evidence Status", className="fw-bold mb-2"),
                            make_table(ev_df, tid="ov-ev", height="230px", sortable=False),
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(html.Div(fit_children), title="Fit Metadata"),
                                    dbc.AccordionItem(html.Div(sbom_c), title="SBOM"),
                                ],
                                start_collapsed=True,
                                flush=True,
                                className="mt-3",
                            ),
                        ],
                        md=7,
                    ),
                    dbc.Col(
                        [
                            html.H6("Column Roles", className="fw-bold mb-2"),
                            make_table(
                                pd.DataFrame(role_rows), tid="ov-roles", height=None, sortable=False
                            ),
                            html.H6("HUGIML Configuration", className="fw-bold mb-2 mt-3"),
                            make_table(
                                pd.DataFrame(cfg), tid="ov-cfg", height=None, sortable=False
                            ),
                        ],
                        md=5,
                    ),
                ]
            ),
        ]
    )

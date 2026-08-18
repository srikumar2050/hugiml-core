"""Dash UI for the optional HUGIML causal investigation studio."""

from __future__ import annotations

import hashlib
import json
import traceback
import uuid
from collections import OrderedDict
from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update
from sklearn.linear_model import LogisticRegression

from hugiml import HUGIMLClassifier
from hugiml.dashboard.dash_components.styles import THEMES, build_css, get_tokens

from .data import (
    dataset_roles,
    demo_catalog,
    load_demo,
    load_metadata_template,
    read_data_upload,
    read_metadata_upload,
)
from .diagnostics import diagnostic_summary
from .metadata import dag_figure, validate_causal_metadata
from .modeling import (
    baseline_grid,
    comparison_frame,
    fit_analysis,
    hug_grid_config,
    selected_configurations_frame,
    t_hug_region_frame,
    t_hug_rpte_frame,
    t_hug_selection_frame,
)

_MAX_DATASETS = 5
_MAX_ANALYSES = 4
_STATE: dict[str, OrderedDict[str, Any]] = {
    "datasets": OrderedDict(),
    "analyses": OrderedDict(),
}

_MODEL_DESCRIPTION = (
    "T-HUG fits separate control and treatment HUG outcome models using one shared supervised vocabulary. "
    "Shared bins and feature semantics keep the two models directly comparable, while utilities, mined patterns, "
    "and the LR/RPTE downstream choice are learned separately within each treatment group."
)

_EXTRA_CSS = """
.causal-page{width:100%;max-width:1440px;margin:0 auto}
.causal-stack{display:grid;gap:12px}
.causal-data-controls{display:flex;align-items:flex-end;gap:12px;flex-wrap:wrap}
.causal-data-controls>.setup-field{min-width:220px;flex:1}.causal-data-controls>.source-field{min-width:230px;flex:0 0 auto}
.causal-upload-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:9px}
.causal-config-fields{display:grid;grid-template-columns:repeat(3,minmax(190px,1fr));gap:10px}
.causal-model-fields{display:grid;grid-template-columns:minmax(220px,1.15fr) minmax(190px,.85fr) minmax(220px,1fr) minmax(130px,.48fr);gap:10px;margin-top:10px}
.causal-meta-row{display:flex;gap:7px;flex-wrap:wrap;margin-top:7px}.causal-summary{font-size:.73rem;color:var(--mu);line-height:1.45;margin-top:6px}
.causal-graph-wrap{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(260px,.6fr);gap:12px;align-items:start}
.causal-graph-note{font-size:.73rem;color:var(--mu);line-height:1.5}.causal-results-head{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;margin-bottom:12px}
.causal-results-title{font-size:1rem;font-weight:850;color:var(--tx)}.causal-results-meta{font-size:.72rem;color:var(--mu);margin-top:3px}
.causal-metrics{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:8px;margin-bottom:12px}
.causal-result-grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(260px,.65fr);gap:12px;align-items:start}
.causal-section-card{background:var(--sf);border:1px solid var(--bs);border-radius:10px;padding:11px 13px;box-shadow:var(--sh)}
.causal-card-title{font-size:.80rem;font-weight:800;color:var(--tx);margin-bottom:7px}.causal-help{font-size:.72rem;color:var(--mu);line-height:1.45;margin-bottom:8px}
.causal-status{font-size:.73rem;color:var(--mu);line-height:1.4}.causal-error{color:var(--rd)!important}.causal-success{color:var(--gr)!important}
.causal-page .Select-control,.causal-page .Select-menu-outer{background:var(--sf)!important;border-color:var(--bs)!important;color:var(--tx)!important}
.causal-page .Select-value-label,.causal-page .Select-placeholder,.causal-page .Select-input>input,.causal-page .VirtualizedSelectOption{color:var(--tx)!important}
.causal-page .VirtualizedSelectFocusedOption{background:var(--cb)!important;color:var(--tx)!important}
.causal-page .is-disabled .Select-control{background:var(--cb)!important}.causal-page .is-disabled .Select-value-label{color:var(--mu)!important}
.causal-page .dash-dropdown{background:var(--sf)!important;color:var(--tx)!important;border-color:var(--bd)!important}
.causal-page .dash-dropdown-content{background:var(--sf)!important;color:var(--tx)!important;border-color:var(--bd)!important}
.causal-page .dash-dropdown-option{color:var(--tx)!important;background:var(--sf)!important}
.causal-page .dash-dropdown-option:hover,.causal-page .dash-dropdown-option[aria-selected="true"]{color:var(--tx)!important;background:var(--cb)!important}
.causal-page .dash-dropdown-search{color:var(--tx)!important;background:var(--sf)!important;border-color:var(--bd)!important}
.causal-page .dash-dropdown:disabled{color:var(--mu)!important;background:var(--cb)!important}
.theme-dark .causal-page input,.theme-dark .causal-page textarea{color:var(--tx)!important;background-color:var(--sf)!important;border-color:var(--bd)!important}
.theme-dark .causal-page .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td,.theme-dark .causal-page .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th{background:var(--sf)!important;color:var(--tx)!important;border-color:var(--bs)!important}
.causal-page details{background:var(--sf);border:1px solid var(--bs);border-radius:10px;box-shadow:var(--sh);overflow:hidden}.causal-page details>summary{cursor:pointer;padding:10px 13px;background:var(--cb);font-size:.78rem;font-weight:800;color:var(--tx)}.causal-page details[open]>summary{border-bottom:1px solid var(--bs)}.causal-details-body{padding:11px 13px}
@media(max-width:950px){.causal-config-fields,.causal-model-fields,.causal-result-grid,.causal-graph-wrap,.causal-upload-row{grid-template-columns:1fr}.causal-metrics{grid-template-columns:1fr 1fr}}
@media(max-width:600px){.causal-config-fields,.causal-metrics{grid-template-columns:1fr}.causal-data-controls{align-items:stretch}.causal-data-controls>.setup-field,.causal-data-controls>.source-field{min-width:0;flex:1 1 100%}}
"""


def _bounded_put(bucket: str, key: str, value: Any, limit: int) -> None:
    store = _STATE[bucket]
    store[key] = value
    store.move_to_end(key)
    while len(store) > limit:
        store.popitem(last=False)


def _frame_key(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(str(df.shape).encode())
    digest.update("|".join(map(str, df.columns)).encode())
    try:
        digest.update(pd.util.hash_pandas_object(df.head(1000), index=True).values.tobytes())
    except Exception:
        pass
    return digest.hexdigest()[:20]


def _put_dataset(df: pd.DataFrame, meta: dict[str, Any] | None, label: str) -> str:
    key = _frame_key(df) + "-" + uuid.uuid4().hex[:6]
    _bounded_put("datasets", key, {"df": df, "meta": meta or {}, "label": label}, _MAX_DATASETS)
    return key


def resolve_dataset(
    source: str,
    demo_key: str,
    data_content: str | None,
    data_filename: str | None,
    metadata_content: str | None,
    metadata_filename: str | None,
) -> dict[str, Any]:
    metadata_backed = False
    if source == "upload":
        if not data_content:
            return {"error": "Upload a CSV, TSV, Parquet, or Excel dataset first."}
        df = read_data_upload(data_content, data_filename or "upload.csv")
        meta: dict[str, Any] = {}
        if metadata_content:
            meta = validate_causal_metadata(
                read_metadata_upload(metadata_content, metadata_filename or "causal_metadata.json"),
                df.columns,
            )
            metadata_backed = True
        label = f"Upload: {data_filename or 'dataset'}"
    else:
        df, meta = load_demo(demo_key or "credit_intervention")
        meta = validate_causal_metadata(meta, df.columns)
        metadata_backed = True
        label = meta.get("label", demo_key)

    roles = dataset_roles(df, meta, metadata_backed=metadata_backed)

    key = _put_dataset(df, meta, label)
    return {
        "key": key,
        "label": label,
        "meta": meta,
        "n": len(df),
        **roles,
    }


def _opts(values):
    return [{"label": str(value), "value": str(value)} for value in values]


def _grid_tooltip(grid_name: str):
    grid = (
        hug_grid_config(grid_name)
        if grid_name == "quick"
        else HUGIMLClassifier.default_param_grid(grid_name)
    )
    dimensions = (
        ("L", grid["L"]),
        ("topK", grid["topK"]),
        ("G", grid["G"]),
        (
            "Downstream",
            [
                "LR" if value is None or isinstance(value, LogisticRegression) else "RPTE"
                for value in grid["base_estimator"]
            ],
        ),
    )
    lines = [f"{label}: {', '.join(map(str, values))}" for label, values in dimensions]
    return html.Pre(
        "\n".join(lines), style={"margin": 0, "whiteSpace": "pre-wrap", "fontSize": "11px"}
    )


def _baseline_grid_tooltip(grid_name: str):
    labels = {"T-LR": "T-LR", "T-XGB": "T-XGB", "T-LGBM": "T-LGBM"}
    blocks = []
    for name, label in labels.items():
        grid = baseline_grid(name, grid_name)
        count = int(np.prod([len(values) for values in grid.values()]))
        params = "; ".join(f"{key}={list(values)}" for key, values in grid.items())
        noun = "configuration" if count == 1 else "configurations"
        blocks.append(f"{label} — {count} {noun}\n{params}")
    return html.Pre(
        "\n\n".join(blocks),
        style={"margin": 0, "whiteSpace": "pre-wrap", "fontSize": "11px", "maxWidth": "520px"},
    )


def _metric(
    label: str,
    value: str,
    sub: str = "",
    *,
    tooltip_id: str | None = None,
    tooltip: str | None = None,
):
    label_content = (
        [label, html.Span(" ⓘ", id=tooltip_id, style={"cursor": "help", "color": "var(--a)"})]
        if tooltip_id
        else label
    )
    children = [
        html.Div(label_content, className="mc-l"),
        html.Div(value, className="mc-v"),
        html.Div(sub, className="causal-summary") if sub else None,
    ]
    if tooltip_id and tooltip:
        children.append(dbc.Tooltip(tooltip, target=tooltip_id, placement="top"))
    return html.Div(children, className="mc")


def _table(df: pd.DataFrame, table_id: str, page_size: int = 12, style_data_conditional=None):
    if df is None or df.empty:
        return html.Div("No rows available.", className="causal-help")
    display = df.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(
            lambda value: None if pd.isna(value) else round(float(value), 5)
        )
    return dash_table.DataTable(
        id=table_id,
        data=display.to_dict("records"),
        columns=[{"name": column, "id": column} for column in display.columns],
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "Inter, sans-serif",
            "fontSize": "12px",
            "padding": "6px",
            "textAlign": "left",
            "maxWidth": "340px",
            "whiteSpace": "normal",
            "backgroundColor": "var(--sf)",
            "color": "var(--tx)",
            "borderColor": "var(--bs)",
        },
        style_header={"fontWeight": "700", "backgroundColor": "var(--cb)", "color": "var(--tx)"},
        style_data_conditional=list(style_data_conditional or []),
    )


def _rank_styles(df: pd.DataFrame, directions: dict[str, str]) -> list[dict]:
    styles = []
    if df is None or df.empty or "Model" not in df.columns:
        return styles
    for column, direction in directions.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        distinct = sorted(numeric.dropna().unique(), reverse=direction == "max")
        for rank, value in enumerate(distinct[:2]):
            models = df.loc[
                np.isclose(numeric, value, rtol=1e-10, atol=1e-12, equal_nan=False), "Model"
            ]
            for model in models.astype(str):
                condition = {
                    "filter_query": f'{{Model}} = "{model.replace(chr(34), chr(92) + chr(34))}"',
                    "column_id": column,
                }
                styles.append(
                    {"if": condition, "fontWeight": "800"}
                    if rank == 0
                    else {"if": condition, "fontStyle": "italic"}
                )
    return styles


def _effect_interval_text(estimate, low, high) -> str:
    if any(pd.isna(value) for value in (estimate, low, high)):
        return "Unavailable"
    return f"{float(estimate):+.3%} ({float(low):+.3%}, {float(high):+.3%})"


def _robust_estimates_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for record in df.to_dict("records"):
        rows.append(
            {
                "Model": record["Model"],
                "Repeated OOF ATE (95% CI)": _effect_interval_text(
                    record["Repeated OOF ATE"], record["ATE 95% CI low"], record["ATE 95% CI high"]
                ),
                "ATE repeat SD": record["ATE repeat SD"],
                "Cross-fitted DR ATE (95% CI)": _effect_interval_text(
                    record["Cross-fitted DR ATE"], record["DR 95% CI low"], record["DR 95% CI high"]
                ),
                "DR repeat SD": record["DR repeat SD"],
                "Repeats": record["Repeats"],
            }
        )
    return pd.DataFrame(rows)


def _overlap_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for record in df.to_dict("records"):
        rows.append(
            {
                "Model": record["Model"],
                "Propensity range": record["Propensity range"],
                "Retained": record["Retained"],
                "Cross-fitted DR ATE (95% CI)": _effect_interval_text(
                    record["Cross-fitted DR ATE"], record["95% CI low"], record["95% CI high"]
                ),
                "Repeat SD": record["Repeat SD"],
            }
        )
    return pd.DataFrame(rows)


def _plot_layout(
    fig: go.Figure, theme: str, *, height: int, x_title: str = "", y_title: str = ""
) -> go.Figure:
    tokens = get_tokens(theme)
    fig.update_layout(
        height=height,
        margin=dict(l=38, r=14, t=20, b=38),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=tokens["tx"], size=11),
        xaxis=dict(title=x_title, gridcolor=tokens["bs"], zerolinecolor=tokens["bd"]),
        yaxis=dict(title=y_title, gridcolor=tokens["bs"], zerolinecolor=tokens["bd"]),
        legend=dict(orientation="h", y=1.08),
        hoverlabel=dict(bgcolor=tokens["sf"], font_color=tokens["tx"], bordercolor=tokens["bd"]),
    )
    return fig


def _header():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "HUGIML Causal Investigation Studio", className="hero-ey"
                                    ),
                                    dcc.Dropdown(
                                        id="cd-theme",
                                        options=[
                                            {"label": theme, "value": theme} for theme in THEMES
                                        ],
                                        value="Ocean",
                                        clearable=False,
                                        searchable=False,
                                        className="theme-select",
                                    ),
                                ],
                                className="hero-topline",
                            ),
                            html.H1(
                                "Causal effect estimation with interpretable T-HUG",
                                className="hero-h1",
                            ),
                            html.P(
                                "Configure a binary-treatment study, run T-HUG with selected baselines, then investigate effects, support, structure, and model comparisons.",
                                className="hero-p",
                            ),
                        ],
                        className="hug-hero",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Button(
                                        "Setup",
                                        id="cd-nav-setup",
                                        n_clicks=1,
                                        className="tab-btn act",
                                    ),
                                    html.Button(
                                        "Results",
                                        id="cd-nav-results",
                                        n_clicks=0,
                                        className="tab-btn",
                                    ),
                                ],
                                className="workbench-nav-row",
                            ),
                        ],
                        className="hug-tabrow",
                    ),
                ],
                className="hug-hdr",
            ),
        ]
    )


def _setup_layout(catalog: dict[str, dict[str, Any]]):
    demos = [{"label": meta["label"], "value": key} for key, meta in catalog.items()]
    return html.Div(
        [
            html.Div(
                [
                    html.Div("DATA", className="setup-lbl"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Source", className="form-label"),
                                    dcc.RadioItems(
                                        id="cd-source",
                                        options=[
                                            {"label": " Demo", "value": "demo"},
                                            {"label": " Upload", "value": "upload"},
                                        ],
                                        value="demo",
                                        inline=True,
                                        className="compact-choice",
                                    ),
                                ],
                                className="setup-field source-field",
                            ),
                            html.Div(
                                [
                                    html.Label("Demo dataset", className="form-label"),
                                    dcc.Dropdown(
                                        id="cd-demo",
                                        options=demos,
                                        value="credit_intervention",
                                        clearable=False,
                                        className="compact-dropdown",
                                    ),
                                ],
                                id="cd-demo-box",
                                className="setup-field",
                            ),
                        ],
                        className="causal-data-controls",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Dataset file", className="form-label"),
                                    dcc.Upload(
                                        id="cd-data-upload",
                                        className="upload-dropzone compact-upload",
                                        children=html.Div(
                                            [
                                                html.Span(
                                                    "Choose data file", className="upload-title"
                                                ),
                                                html.Span(
                                                    " CSV · TSV · Excel · Parquet",
                                                    className="upload-subtitle",
                                                ),
                                            ]
                                        ),
                                        multiple=False,
                                    ),
                                ],
                                className="setup-field",
                            ),
                            html.Div(
                                [
                                    html.Label("Optional causal metadata", className="form-label"),
                                    dcc.Upload(
                                        id="cd-meta-upload",
                                        className="upload-dropzone compact-upload",
                                        children=html.Div(
                                            [
                                                html.Span(
                                                    "Choose metadata JSON", className="upload-title"
                                                )
                                            ]
                                        ),
                                        multiple=False,
                                    ),
                                ],
                                className="setup-field",
                            ),
                        ],
                        id="cd-upload-box",
                        className="causal-upload-row",
                        style={"display": "none"},
                    ),
                    html.Div(id="cd-dataset-note", className="causal-summary"),
                ],
                className="setup-card setup-card-compact",
            ),
            html.Div(
                [
                    html.Div("STUDY CONFIGURATION", className="setup-lbl"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Treatment", className="form-label"),
                                    dcc.Dropdown(id="cd-treatment", clearable=False),
                                ],
                                className="setup-field",
                            ),
                            html.Div(
                                [
                                    html.Label("Outcome", className="form-label"),
                                    dcc.Dropdown(id="cd-outcome", clearable=False),
                                ],
                                className="setup-field",
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Adjustment / pre-treatment covariates",
                                        className="form-label",
                                    ),
                                    dcc.Dropdown(id="cd-adjustment", multi=True),
                                ],
                                className="setup-field",
                            ),
                        ],
                        className="causal-config-fields",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        [
                                            "Baseline comparison ",
                                            html.Span(
                                                "ⓘ",
                                                id="cd-baseline-grid-help",
                                                style={"cursor": "help", "color": "var(--a)"},
                                            ),
                                        ],
                                        className="form-label",
                                    ),
                                    dcc.Dropdown(
                                        id="cd-baselines",
                                        multi=True,
                                        options=_opts(["T-LR", "T-XGB", "T-LGBM"]),
                                        value=["T-LR", "T-XGB"],
                                    ),
                                    dbc.Tooltip(
                                        _baseline_grid_tooltip("quick"),
                                        id="cd-baseline-grid-tooltip",
                                        target="cd-baseline-grid-help",
                                        placement="right",
                                    ),
                                ],
                                className="setup-field",
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        [
                                            "HUGIML grid ",
                                            html.Span(
                                                "ⓘ",
                                                id="cd-hug-grid-help",
                                                style={"cursor": "help", "color": "var(--a)"},
                                            ),
                                        ],
                                        className="form-label",
                                    ),
                                    dcc.Dropdown(
                                        id="cd-hug-grid",
                                        options=[
                                            {"label": "Quick — 1 configuration", "value": "quick"},
                                            {"label": "Performance", "value": "performance"},
                                            {
                                                "label": "Interpretability",
                                                "value": "interpretability",
                                            },
                                            {"label": "Performance HO", "value": "performance_ho"},
                                            {
                                                "label": "Interpretability HO",
                                                "value": "interpretability_ho",
                                            },
                                        ],
                                        value="quick",
                                        clearable=False,
                                    ),
                                    dbc.Tooltip(
                                        _grid_tooltip("quick"),
                                        id="cd-hug-grid-tooltip",
                                        target="cd-hug-grid-help",
                                        placement="right",
                                    ),
                                ],
                                className="setup-field",
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "T-HUG downstream selection metric", className="form-label"
                                    ),
                                    dcc.Dropdown(
                                        id="cd-selection-scoring",
                                        options=[
                                            {"label": "ROC AUC", "value": "roc_auc"},
                                            {"label": "Negative log loss", "value": "neg_log_loss"},
                                        ],
                                        value="roc_auc",
                                        clearable=False,
                                    ),
                                ],
                                className="setup-field",
                            ),
                            html.Div(
                                [
                                    html.Label("Analysis seeds", className="form-label"),
                                    html.Div(
                                        "42, 43, 44", className="form-control form-control-sm"
                                    ),
                                    html.Div(
                                        "Three reproducible cross-fitting seeds",
                                        className="causal-help",
                                    ),
                                ],
                                className="setup-field",
                            ),
                        ],
                        className="causal-model-fields",
                    ),
                    html.Div(_MODEL_DESCRIPTION, className="setup-help"),
                ],
                className="setup-card setup-card-compact",
            ),
            html.Details(
                [
                    html.Summary("Causal graph and assumptions"),
                    html.Div(
                        [
                            html.Div([dcc.Graph(id="cd-dag", config={"displayModeBar": False})]),
                            html.Div(
                                [
                                    html.Div(id="cd-adjust-note", className="info-b"),
                                    html.Div(
                                        id="cd-study-description",
                                        className="causal-graph-note",
                                        style={"marginTop": "9px"},
                                    ),
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Download metadata template",
                                                id="cd-meta-template",
                                                size="sm",
                                                outline=True,
                                                color="secondary",
                                                n_clicks=0,
                                            ),
                                            dcc.Download(id="cd-meta-template-download"),
                                        ],
                                        className="causal-meta-row",
                                    ),
                                ]
                            ),
                        ],
                        className="causal-details-body causal-graph-wrap",
                    ),
                ],
                open=True,
            ),
            html.Div(
                [
                    html.Div(id="cd-run-status", className="causal-status"),
                    html.Div(
                        [
                            dbc.Button(
                                "Run causal analysis",
                                id="cd-run",
                                color="primary",
                                n_clicks=0,
                                className="run-primary-action",
                            ),
                        ],
                        className="run-action-controls",
                    ),
                ],
                className="run-action-bar",
            ),
        ],
        className="causal-page causal-stack",
    )


def _results_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Causal analysis results", className="causal-results-title"),
                            html.Div(id="cd-results-meta", className="causal-results-meta"),
                        ]
                    ),
                    dbc.Button(
                        "Back to setup",
                        id="cd-back-setup",
                        size="sm",
                        outline=True,
                        color="secondary",
                        n_clicks=0,
                    ),
                ],
                className="causal-results-head",
            ),
            dbc.Tabs(
                [
                    dbc.Tab(html.Div(id="cd-effects"), label="Effects", tab_id="effects"),
                    dbc.Tab(
                        html.Div(id="cd-diagnostics"), label="Diagnostics", tab_id="diagnostics"
                    ),
                    dbc.Tab(html.Div(id="cd-regions"), label="Regions", tab_id="regions"),
                    dbc.Tab(html.Div(id="cd-comparison"), label="Comparison", tab_id="comparison"),
                ],
                id="cd-results-tabs",
                active_tab="effects",
                className="results-view-menu",
            ),
        ],
        className="causal-page",
    )


def build_layout(catalog: dict[str, dict[str, Any]]):
    return html.Div(
        [
            _header(),
            dcc.Loading(
                id="cd-main-loading",
                type="circle",
                color="var(--a)",
                overlay_style={"visibility": "visible", "filter": "blur(1px)"},
                parent_className="hug-content",
                children=html.Div(
                    [
                        html.Div(_setup_layout(catalog), id="cd-setup-page"),
                        html.Div(
                            _results_layout(), id="cd-results-page", style={"display": "none"}
                        ),
                    ]
                ),
            ),
            dcc.Store(id="cd-view", data="setup"),
            dcc.Store(id="cd-dataset-key"),
            dcc.Store(id="cd-analysis-key"),
            dcc.Store(id="cd-metadata", data={}),
        ],
        id="cd-root",
        className="hug-wrap theme-ocean",
    )


def create_app(debug: bool = False) -> Dash:
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="HUGIML Causal Investigation Studio",
        update_title=None,
    )
    app.index_string = app.index_string.replace(
        "</head>", f"<style id='hug-theme'>{build_css('Ocean')}{_EXTRA_CSS}</style></head>"
    )
    catalog = demo_catalog()
    app.layout = build_layout(catalog)

    @app.callback(Output("cd-root", "className"), Input("cd-theme", "value"))
    def _theme(theme):
        selected = theme if theme in THEMES else "Ocean"
        return f"hug-wrap theme-{selected.lower()}"

    @app.callback(
        Output("cd-hug-grid-tooltip", "children"),
        Output("cd-baseline-grid-tooltip", "children"),
        Input("cd-hug-grid", "value"),
    )
    def _hug_grid_tooltip(grid_name):
        selected = (
            grid_name
            if grid_name
            in {"quick", "performance", "interpretability", "performance_ho", "interpretability_ho"}
            else "quick"
        )
        return _grid_tooltip(selected), _baseline_grid_tooltip(selected)

    @app.callback(
        Output("cd-demo-box", "style"),
        Output("cd-upload-box", "style"),
        Input("cd-source", "value"),
    )
    def _source_view(source):
        if source == "upload":
            return {"display": "none"}, {}
        return {}, {"display": "none"}

    @app.callback(
        Output("cd-meta-template-download", "data"),
        Input("cd-meta-template", "n_clicks"),
        prevent_initial_call=True,
    )
    def _download_metadata_template(n_clicks):
        if not n_clicks:
            return no_update
        return {
            "content": json.dumps(load_metadata_template(), indent=2),
            "filename": "causal_metadata.json",
            "type": "application/json",
        }

    @app.callback(
        Output("cd-dataset-key", "data"),
        Output("cd-metadata", "data"),
        Output("cd-treatment", "options"),
        Output("cd-treatment", "value"),
        Output("cd-treatment", "disabled"),
        Output("cd-outcome", "options"),
        Output("cd-outcome", "value"),
        Output("cd-outcome", "disabled"),
        Output("cd-adjustment", "options"),
        Output("cd-adjustment", "value"),
        Output("cd-dataset-note", "children"),
        Input("cd-source", "value"),
        Input("cd-demo", "value"),
        Input("cd-data-upload", "contents"),
        Input("cd-meta-upload", "contents"),
        State("cd-data-upload", "filename"),
        State("cd-meta-upload", "filename"),
    )
    def _load(source, demo_key, data_content, meta_content, data_filename, meta_filename):
        try:
            info = resolve_dataset(
                source, demo_key, data_content, data_filename, meta_content, meta_filename
            )
            if info.get("error"):
                return no_update, {}, [], None, False, [], None, False, [], [], info["error"]
            lock_roles = bool(info["metadata_backed"])
            note = f"{info['label']} · {info['n']:,} rows · {len(info['columns'])} visible columns"
            return (
                info["key"],
                info["meta"],
                _opts(info["treatment_options"]),
                info["treatment"],
                lock_roles,
                _opts(info["outcome_options"]),
                info["outcome"],
                lock_roles,
                _opts(info["adjustment_options"]),
                info["adjustment"],
                note,
            )
        except Exception as exc:
            return no_update, {}, [], None, False, [], None, False, [], [], f"Dataset error: {exc}"

    @app.callback(
        Output("cd-dag", "figure"),
        Output("cd-adjust-note", "children"),
        Output("cd-study-description", "children"),
        Input("cd-metadata", "data"),
        Input("cd-theme", "value"),
        Input("cd-adjustment", "value"),
    )
    def _study_metadata(meta, theme, adjustment):
        meta = meta or {}
        selected_theme = theme if theme in THEMES else "Ocean"
        rationale = (
            meta.get("causal_rationale")
            or "No causal graph metadata supplied. Treatment, outcome, and adjustment variables are user-declared."
        )
        description = meta.get("description", "")
        adj = list(adjustment or [])
        adj_note = (
            "Adjustment set: " + ", ".join(adj) if adj else "No adjustment covariates selected."
        )
        return (
            dag_figure(meta, theme=selected_theme),
            adj_note,
            html.Div([html.P(description), html.P(rationale)]),
        )

    @app.callback(
        Output("cd-analysis-key", "data"),
        Output("cd-run-status", "children"),
        Output("cd-run-status", "className"),
        Output("cd-view", "data", allow_duplicate=True),
        Input("cd-run", "n_clicks"),
        State("cd-dataset-key", "data"),
        State("cd-treatment", "value"),
        State("cd-outcome", "value"),
        State("cd-adjustment", "value"),
        State("cd-baselines", "value"),
        State("cd-hug-grid", "value"),
        State("cd-selection-scoring", "value"),
        prevent_initial_call=True,
    )
    def _run(
        n_clicks,
        dataset_key,
        treatment,
        outcome,
        adjustment,
        baselines,
        hug_grid,
        selection_scoring,
    ):
        if not n_clicks:
            return no_update, no_update, no_update, no_update
        try:
            dataset = _STATE["datasets"].get(dataset_key)
            if dataset is None:
                raise ValueError("Dataset state is no longer available; reload the dataset.")
            seeds = [42, 43, 44]
            analysis = fit_analysis(
                dataset["df"],
                treatment=treatment,
                outcome=outcome,
                covariates=list(adjustment or []),
                baselines=list(baselines or []),
                hug_grid=str(hug_grid or "quick"),
                random_state=seeds[0],
                selection_scoring=str(selection_scoring or "roc_auc"),
                robust_seeds=seeds,
                robust_folds=2,
                study_type=str(dataset.get("meta", {}).get("study_type", "observational")),
            )
            meta = dataset.get("meta", {})
            oracle_col = meta.get("oracle_tau_column")
            if oracle_col and oracle_col in dataset["df"].columns:
                aligned = dataset["df"].iloc[analysis["source_positions"]][oracle_col]
                analysis["oracle_tau"] = pd.to_numeric(aligned, errors="coerce").to_numpy(
                    dtype=float
                )
            analysis["meta"] = meta
            analysis["dataset_label"] = dataset["label"]
            analysis_key = uuid.uuid4().hex
            _bounded_put("analyses", analysis_key, analysis, _MAX_ANALYSES)
            message = f"Analysis complete · {len(analysis['frame']):,} rows · {len(analysis['results'])} models"
            return analysis_key, message, "causal-status causal-success", "results"
        except Exception as exc:
            if debug:
                traceback.print_exc()
            return no_update, f"Analysis failed: {exc}", "causal-status causal-error", no_update

    @app.callback(
        Output("cd-view", "data"),
        Input("cd-nav-setup", "n_clicks"),
        Input("cd-nav-results", "n_clicks"),
        Input("cd-back-setup", "n_clicks"),
        State("cd-analysis-key", "data"),
        prevent_initial_call=True,
    )
    def _navigate(setup_clicks, results_clicks, back_clicks, analysis_key):
        trigger = dash.ctx.triggered_id
        if trigger in {"cd-nav-setup", "cd-back-setup"}:
            return "setup"
        if trigger == "cd-nav-results" and analysis_key in _STATE["analyses"]:
            return "results"
        return no_update

    @app.callback(
        Output("cd-setup-page", "style"),
        Output("cd-results-page", "style"),
        Output("cd-nav-setup", "className"),
        Output("cd-nav-results", "className"),
        Input("cd-view", "data"),
    )
    def _switch_view(view):
        if view == "results":
            return {"display": "none"}, {}, "tab-btn", "tab-btn act"
        return {}, {"display": "none"}, "tab-btn act", "tab-btn"

    @app.callback(
        Output("cd-results-meta", "children"),
        Output("cd-effects", "children"),
        Output("cd-diagnostics", "children"),
        Output("cd-regions", "children"),
        Output("cd-comparison", "children"),
        Input("cd-analysis-key", "data"),
        Input("cd-theme", "value"),
    )
    def _render_results(analysis_key, theme):
        if not analysis_key or analysis_key not in _STATE["analyses"]:
            return "", "", "", "", ""
        analysis = _STATE["analyses"][analysis_key]
        selected_theme = theme if theme in THEMES else "Ocean"
        frame = analysis["frame"]
        meta = analysis.get("meta", {})
        treatment = analysis["treatment"]
        outcome = analysis["outcome"]
        diagnostics = diagnostic_summary(
            frame,
            treatment,
            outcome,
            analysis["covariates"],
            meta.get("study_type", "observational"),
        )
        result = analysis["results"]["T-HUG"]
        robustness = analysis.get("robustness")
        robust_estimates = robustness["estimates"] if robustness else pd.DataFrame()
        t_hug_robust = (
            robust_estimates.loc[robust_estimates["Model"] == "T-HUG"]
            if not robust_estimates.empty
            else pd.DataFrame()
        )
        robust_display = _robust_estimates_display(robust_estimates)
        t_hug_robust_display = _robust_estimates_display(t_hug_robust)
        if not t_hug_robust.empty:
            robust_row = t_hug_robust.iloc[0]
            repeated_ate_text = f"{robust_row['Repeated OOF ATE']:+.3%}"
            repeated_ate_ci = f"95% CI {robust_row['ATE 95% CI low']:+.3%} to {robust_row['ATE 95% CI high']:+.3%}"
            dr_ate_text = f"{robust_row['Cross-fitted DR ATE']:+.3%}"
            dr_ate_ci = (
                f"95% CI {robust_row['DR 95% CI low']:+.3%} to {robust_row['DR 95% CI high']:+.3%}"
            )

        tau = result.tau
        direction = meta.get("outcome_direction", "higher_is_better")
        benefit = tau < 0 if direction == "lower_is_better" else tau > 0
        tau_fig = go.Figure()
        tau_fig.add_histogram(x=tau, nbinsx=35, marker_color=get_tokens(selected_theme)["a"])
        tau_fig.add_vline(
            x=float(np.mean(tau)), line_dash="dash", line_color=get_tokens(selected_theme)["a2"]
        )
        _plot_layout(
            tau_fig, selected_theme, height=320, x_title="Estimated CATE = P1 − P0", y_title="Count"
        )
        effects = html.Div(
            [
                html.Div(
                    [
                        _metric(
                            "T-HUG ATE",
                            f"{np.mean(tau):+.3%}",
                            tooltip_id="cd-tip-ate",
                            tooltip="Average estimated treatment effect: mean predicted treatment risk minus mean predicted control risk.",
                        ),
                        _metric(
                            "Mean control risk",
                            f"{np.mean(result.p0):.2%}",
                            tooltip_id="cd-tip-p0",
                            tooltip="Average predicted probability of the outcome if everyone were assigned to control.",
                        ),
                        _metric(
                            "Mean treatment risk",
                            f"{np.mean(result.p1):.2%}",
                            tooltip_id="cd-tip-p1",
                            tooltip="Average predicted probability of the outcome if everyone were assigned to treatment.",
                        ),
                        _metric(
                            "Estimated to benefit",
                            f"{np.mean(benefit):.1%}",
                            "Uses the declared outcome direction",
                            tooltip_id="cd-tip-benefit",
                            tooltip="Share of rows with a predicted treatment effect in the beneficial direction. This is model-estimated, not an observed response rate.",
                        ),
                    ],
                    className="causal-metrics",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "Treatment-effect distribution", className="causal-card-title"
                                ),
                                dcc.Graph(figure=tau_fig, config={"displayModeBar": False}),
                            ],
                            className="causal-section-card",
                        ),
                        html.Div(
                            [
                                html.Div("Observed and adjusted", className="causal-card-title"),
                                _metric("Raw group difference", f"{diagnostics['raw_diff']:+.3%}"),
                                html.Div(
                                    "The raw difference is descriptive. T-HUG estimates control and treatment potential-outcome risks after conditioning on the declared pre-treatment covariates.",
                                    className="info-b",
                                    style={"marginTop": "10px"},
                                ),
                            ],
                            className="causal-section-card",
                        ),
                    ],
                    className="causal-result-grid",
                ),
                html.Div(
                    [
                        html.Div("Repeated cross-fitted estimates", className="causal-card-title"),
                        html.Div(
                            "Three repeated two-fold runs provide out-of-fold predictions for every row. AIPW combines the outcome models with the treatment-assignment model.",
                            className="causal-help",
                        ),
                        html.Div(
                            [
                                _metric(
                                    "Repeated OOF T-HUG ATE", repeated_ate_text, repeated_ate_ci
                                ),
                                _metric("Cross-fitted DR/AIPW ATE", dr_ate_text, dr_ate_ci),
                            ],
                            className="causal-metrics",
                        )
                        if not t_hug_robust.empty
                        else None,
                        _table(t_hug_robust_display, "cd-t-hug-robust-table", 5)
                        if not t_hug_robust.empty
                        else html.Div(
                            "Robust estimates unavailable: "
                            + str(analysis.get("robustness_error") or "not calculated"),
                            className="info-b",
                        ),
                    ],
                    className="causal-section-card",
                    style={"marginTop": "12px"},
                ),
            ]
        )

        propensity = diagnostics["propensity"]
        treatment_values = pd.to_numeric(frame[treatment], errors="raise").astype(int).to_numpy()
        propensity_fig = go.Figure()
        propensity_fig.add_histogram(
            x=propensity[treatment_values == 0],
            name="Control",
            opacity=0.66,
            marker_color=get_tokens(selected_theme)["a2"],
        )
        propensity_fig.add_histogram(
            x=propensity[treatment_values == 1],
            name="Treatment",
            opacity=0.66,
            marker_color=get_tokens(selected_theme)["a"],
        )
        propensity_fig.update_layout(barmode="overlay")
        _plot_layout(
            propensity_fig,
            selected_theme,
            height=300,
            x_title="Estimated propensity",
            y_title="Count",
        )
        diagnostic_view = html.Div(
            [
                html.Div(
                    [
                        _metric("Rows", f"{diagnostics['n']:,}"),
                        _metric("Control", f"{diagnostics['n0']:,}"),
                        _metric("Treatment", f"{diagnostics['n1']:,}"),
                        _metric("Overlap 0.05–0.95", f"{100 * diagnostics['overlap_rate']:.1f}%"),
                    ],
                    className="causal-metrics",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "Treatment assignment and overlap",
                                    className="causal-card-title",
                                ),
                                dcc.Graph(figure=propensity_fig, config={"displayModeBar": False}),
                                html.Div(
                                    "Randomized demos use the observed assignment probability. Observational studies use a logistic propensity diagnostic.",
                                    className="causal-help",
                                ),
                            ],
                            className="causal-section-card",
                        ),
                        html.Div(
                            [
                                html.Div("Covariate balance", className="causal-card-title"),
                                _table(diagnostics["balance"], "cd-balance-table", 10),
                            ],
                            className="causal-section-card",
                        ),
                    ],
                    className="causal-result-grid",
                ),
                html.Div(
                    [
                        html.Div("Overlap sensitivity", className="causal-card-title"),
                        html.Div(
                            "Cross-fitted AIPW estimates are recalculated after successively excluding rows with extreme treatment probabilities. Retained shows the average share of rows remaining.",
                            className="causal-help",
                        ),
                        _table(
                            _overlap_display(robustness["overlap_sensitivity"]),
                            "cd-overlap-sensitivity-table",
                            20,
                        )
                        if robustness
                        else html.Div(
                            "Overlap sensitivity unavailable: "
                            + str(analysis.get("robustness_error") or "not calculated"),
                            className="info-b",
                        ),
                    ],
                    className="causal-section-card",
                    style={"marginTop": "12px"},
                ),
            ]
        )

        regions_df = t_hug_region_frame(analysis)
        rpte_df = t_hug_rpte_frame(analysis)
        selection_df = t_hug_selection_frame(analysis)
        regions_view = html.Div(
            [
                html.Div(
                    [
                        html.Div("T-HUG structural regions", className="causal-card-title"),
                        html.Div(
                            "Patterns use the shared vocabulary, so thresholds and feature semantics are directly comparable between the control and treatment outcome models.",
                            className="causal-help",
                        ),
                        _table(regions_df, "cd-region-table", 12),
                    ],
                    className="causal-section-card",
                ),
                html.Div(
                    [
                        html.Div("Downstream model selection", className="causal-card-title"),
                        html.Div(
                            "T-HUG selects LR or RPTE separately for the control and treatment outcome models using the configured predictive scoring rule.",
                            className="causal-help",
                        ),
                        _table(selection_df, "cd-selection-table", 5),
                    ],
                    className="causal-section-card",
                    style={"marginTop": "12px"},
                ),
                html.Div(
                    [
                        html.Div(
                            "RPTE leaf and direct-rule structure", className="causal-card-title"
                        ),
                        html.Div(
                            "Shown only for a group whose downstream model is RPTE. RPTE components retain source-disjoint structure and mutually exclusive leaves within each component.",
                            className="causal-help",
                        ),
                        _table(rpte_df, "cd-rpte-table", 12),
                    ],
                    className="causal-section-card",
                    style={"marginTop": "12px"},
                ),
            ]
        )

        comparison = comparison_frame(analysis, analysis.get("oracle_tau"))
        selected_configs = selected_configurations_frame(analysis)
        comparison_rank_styles = _rank_styles(
            comparison,
            {
                "Held-out AUC": "max",
                "Held-out Brier": "min",
                "Fit seconds": "min",
                "Oracle CATE RMSE": "min",
                "Oracle CATE corr": "max",
            },
        )
        robust_rank_styles = _rank_styles(
            robust_estimates,
            {
                "ATE repeat SD": "min",
                "DR repeat SD": "min",
            },
        )
        metric = (
            "Oracle CATE RMSE" if "Oracle CATE RMSE" in comparison.columns else "Held-out Brier"
        )
        comparison_fig = go.Figure(
            go.Bar(
                x=comparison["Model"],
                y=comparison[metric],
                marker_color=get_tokens(selected_theme)["a"],
            )
        )
        _plot_layout(comparison_fig, selected_theme, height=300, x_title="Model", y_title=metric)
        stability_fig = go.Figure()
        if not robust_estimates.empty:
            center = robust_estimates["Cross-fitted DR ATE"].to_numpy(dtype=float)
            stability_fig.add_trace(
                go.Scatter(
                    x=robust_estimates["Model"],
                    y=center,
                    mode="markers",
                    marker={"size": 10, "color": get_tokens(selected_theme)["a"]},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": robust_estimates["DR 95% CI high"].to_numpy(dtype=float) - center,
                        "arrayminus": center
                        - robust_estimates["DR 95% CI low"].to_numpy(dtype=float),
                    },
                )
            )
            stability_fig.add_hline(
                y=0, line_dash="dash", line_color=get_tokens(selected_theme)["a2"]
            )
            _plot_layout(
                stability_fig,
                selected_theme,
                height=300,
                x_title="Model",
                y_title="Cross-fitted DR ATE",
            )
        comparison_view = html.Div(
            [
                html.Div(
                    [
                        html.Div("Baseline comparison", className="causal-card-title"),
                        html.Div(
                            "T-HUG is always included. Selected baselines use separate control and treatment learners on the same train/test split.",
                            className="causal-help",
                        ),
                        _table(comparison, "cd-comparison-table", 10, comparison_rank_styles),
                    ],
                    className="causal-section-card",
                ),
                html.Div(
                    [
                        html.Div("Primary comparison metric", className="causal-card-title"),
                        dcc.Graph(figure=comparison_fig, config={"displayModeBar": False}),
                        html.Div(
                            "Oracle CATE metrics appear only for demos with known counterfactual probabilities. Uploaded real data is evaluated only with observable-data metrics.",
                            className="info-b",
                        ),
                    ],
                    className="causal-section-card",
                    style={"marginTop": "12px"},
                ),
                html.Div(
                    [
                        html.Div(
                            "Repeated-split stability and doubly robust estimates",
                            className="causal-card-title",
                        ),
                        html.Div(
                            f"Each method uses {robustness['repeats']} repeated {robustness['folds']}-fold runs. Propensity: {robustness['propensity_model']}."
                            if robustness
                            else "Repeated cross-fitted estimates were not available for this dataset.",
                            className="causal-help",
                        ),
                        _table(robust_display, "cd-robust-comparison-table", 10, robust_rank_styles)
                        if robustness
                        else html.Div(
                            str(analysis.get("robustness_error") or "Not calculated"),
                            className="info-b",
                        ),
                        dcc.Graph(figure=stability_fig, config={"displayModeBar": False})
                        if robustness
                        else None,
                    ],
                    className="causal-section-card",
                    style={"marginTop": "12px"},
                ),
                html.Details(
                    [
                        html.Summary("Selected configurations"),
                        html.Div(
                            [
                                html.Div(
                                    "Winning settings are shown separately for the control and treatment outcome models. Scores are internal tuning scores; the comparison metrics above remain held out.",
                                    className="causal-help",
                                ),
                                _table(selected_configs, "cd-selected-config-table", 10),
                            ],
                            className="causal-details-body",
                        ),
                    ],
                    style={"marginTop": "12px"},
                ),
                html.Div(
                    "Best values are shown in bold; second-best values are italicized. Ranking is applied only where higher or lower has an objective interpretation.",
                    className="causal-help",
                    style={"marginTop": "10px"},
                ),
            ]
        )

        scoring_label = (
            "ROC AUC" if analysis.get("selection_scoring") == "roc_auc" else "Negative log loss"
        )
        grid_label = str(analysis.get("hug_grid", "quick"))
        results_meta = f"{analysis.get('dataset_label', 'Dataset')} · Grid: {grid_label} · T-HUG downstream selection: {scoring_label}"
        return results_meta, effects, diagnostic_view, regions_view, comparison_view

    return app


def main():
    create_app().run(debug=False)

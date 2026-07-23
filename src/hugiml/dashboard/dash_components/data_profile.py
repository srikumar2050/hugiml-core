"""Interactive data profiling for the Dash Workbench Setup page."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html, no_update
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from hugiml.dashboard.dash_components.data_utils import load_demo_raw, read_upload
from hugiml.dashboard.dash_components.profile_stats import (
    CATEGORY_DISPLAY_LIMIT,
    CORRELATION_COLUMN_LIMIT,
    MISSINGNESS_MAP_ROW_LIMIT,
    categorical_profile,
    dataset_summary,
    datetime_profile,
    missingness_patterns,
    model_input_columns,
    numeric_profile,
    predictor_exclusions,
    profile_findings,
    profile_scope,
    semantic_type,
    target_associations,
    text_profile,
    variable_inventory,
)

_MAX_RAW_DATASETS = 8
_RAW_DATASETS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_PROFILE_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()

_PROFILE_BAR_COLOR = "#5CB98C"
_PROFILE_BAR_EDGE_COLOR = "#2F7F60"
_PROFILE_HISTOGRAM_COLOR = "#82D0AD"
_PROFILE_DENSITY_COLOR = "#23845F"
_PROFILE_MISSING_COLOR = "#E4A85C"


def _store_raw(
    key: str,
    frame: pd.DataFrame,
    label: str,
    roles: dict[str, Any],
    **reference_metadata: Any,
) -> dict[str, Any]:
    _RAW_DATASETS[key] = {"frame": frame, "label": label, "roles": dict(roles)}
    _RAW_DATASETS.move_to_end(key)
    while len(_RAW_DATASETS) > _MAX_RAW_DATASETS:
        dropped, _ = _RAW_DATASETS.popitem(last=False)
        for cache_key in list(_PROFILE_CACHE):
            if cache_key.startswith(dropped + ":"):
                _PROFILE_CACHE.pop(cache_key, None)
    return {
        "dataset_key": key,
        "label": label,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "roles": dict(roles),
        **reference_metadata,
    }


def register_profile_dataset(
    source: str,
    demo_key: str | None,
    upload_contents: str | None,
    upload_name: str | None,
) -> dict[str, Any] | None:
    """Register one raw dataset and return a browser-safe reference."""
    if source == "demo":
        key_name = demo_key or "credit"
        raw, roles, label = load_demo_raw(key_name)
        key = hashlib.sha256(f"demo:{key_name}".encode()).hexdigest()
        return _store_raw(key, raw, label, roles, source="demo", source_key=key_name)
    if upload_contents is None:
        return None
    raw = read_upload(upload_contents, upload_name or "file")
    content_hash = hashlib.sha256(upload_contents.encode()).hexdigest()
    key = hashlib.sha256(f"upload:{upload_name}:{content_hash}".encode()).hexdigest()
    roles = {"target": None, "id_column": None, "excluded_columns": [], "sensitive_columns": []}
    return _store_raw(
        key,
        raw,
        f"Upload: {upload_name or 'file'}",
        roles,
        source="upload",
        source_key=content_hash,
        upload_name=upload_name or "file",
    )


def profile_reference_matches(
    reference: dict[str, Any] | None,
    source: str,
    demo_key: str | None,
    upload_contents: str | None,
    upload_name: str | None,
) -> bool:
    """Return whether a browser reference describes the current source controls."""
    if not reference or reference.get("source") != source:
        return False
    if source == "demo":
        return reference.get("source_key") == (demo_key or "credit")
    if upload_contents is None:
        return False
    content_hash = hashlib.sha256(upload_contents.encode()).hexdigest()
    return reference.get("source_key") == content_hash and reference.get("upload_name") == (
        upload_name or "file"
    )


def get_profile_dataset(reference: dict[str, Any] | None) -> dict[str, Any] | None:
    if not reference:
        return None
    key = str(reference.get("dataset_key") or "")
    value = _RAW_DATASETS.get(key)
    if value is not None:
        _RAW_DATASETS.move_to_end(key)
    return value


def _roles(
    reference: dict[str, Any] | None, target, id_column, excluded, sensitive
) -> dict[str, Any]:
    dataset = get_profile_dataset(reference)
    default = dict((dataset or {}).get("roles", {}))
    if reference and reference.get("source") == "demo":
        default.setdefault("excluded_columns", [])
        default.setdefault("sensitive_columns", [])
        return default
    if target not in (None, ""):
        default["target"] = target
    if id_column is not None:
        default["id_column"] = None if id_column == "__none__" else id_column
    if excluded is not None:
        default["excluded_columns"] = list(excluded or [])
    if sensitive is not None:
        default["sensitive_columns"] = list(sensitive or [])
    default.setdefault("excluded_columns", [])
    default.setdefault("sensitive_columns", [])
    return default


def _profile_key(reference: dict[str, Any], mode: str, roles: dict[str, Any], view: str) -> str:
    payload = json.dumps({"mode": mode, "roles": roles, "view": view}, sort_keys=True, default=str)
    return f"{reference['dataset_key']}:{hashlib.sha256(payload.encode()).hexdigest()}"


def _profile_context(reference, mode, roles, view="raw") -> dict[str, Any]:
    dataset = get_profile_dataset(reference)
    if dataset is None:
        raise ValueError("The selected dataset is no longer available. Select it again.")
    key = _profile_key(reference, mode, roles, view)
    cached = _PROFILE_CACHE.get(key)
    if cached is not None:
        _PROFILE_CACHE.move_to_end(key)
        return cached
    raw = dataset["frame"]
    columns = list(map(str, raw.columns))
    raw_scope = profile_scope(raw, mode=mode, target=roles.get("target"))
    sampled_raw = raw_scope.frame
    if view == "model":
        selected = [name for name in model_input_columns(raw, roles) if name in columns]
        frame = sampled_raw.loc[:, selected] if selected else sampled_raw.iloc[:, :0]
    else:
        frame = sampled_raw
    inventory = variable_inventory(frame, roles)
    target = roles.get("target")
    relationship_frame = frame.copy(deep=False)
    if target in sampled_raw.columns and target not in relationship_frame.columns:
        relationship_frame = relationship_frame.copy()
        relationship_frame[target] = sampled_raw[target]
    result = {
        "scope": raw_scope,
        "frame": frame,
        "relationship_frame": relationship_frame,
        "inventory": inventory,
        "summary": dataset_summary(frame, inventory),
        "findings": profile_findings(relationship_frame, inventory, roles),
        "roles": roles,
        "view": view,
    }
    _PROFILE_CACHE[key] = result
    _PROFILE_CACHE.move_to_end(key)
    while len(_PROFILE_CACHE) > 24:
        _PROFILE_CACHE.popitem(last=False)
    return result


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20))
    return figure


def _figure_layout(figure: go.Figure, title: str, height: int = 300) -> go.Figure:
    figure.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=height,
        margin=dict(l=45, r=20, t=48, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figure


def _type_figure(inventory: pd.DataFrame) -> go.Figure:
    if inventory.empty:
        return _empty_figure("No variables")
    counts = inventory["semantic_type"].value_counts()
    figure = go.Figure(
        go.Pie(labels=counts.index.tolist(), values=counts.values.tolist(), hole=0.55)
    )
    return _figure_layout(figure, "Variable types", 290)


def _missing_bar(frame: pd.DataFrame, limit: int = 25) -> go.Figure:
    values = (frame.isna().mean() * 100).sort_values(ascending=True)
    values = values[values > 0].tail(limit)
    if values.empty:
        return _empty_figure("No missing values")
    figure = go.Figure(
        go.Bar(
            x=values.values,
            y=list(map(str, values.index)),
            orientation="h",
            marker=dict(
                color=_PROFILE_MISSING_COLOR,
                line=dict(color=_PROFILE_BAR_EDGE_COLOR, width=0.6),
            ),
        )
    )
    figure.update_xaxes(title="Missing (%)", range=[0, max(100.0, float(values.max()) * 1.05)])
    return _figure_layout(figure, "Missingness by variable", max(290, 28 * len(values) + 80))


def _target_figure(frame: pd.DataFrame, target: str | None) -> go.Figure:
    if target is None or target not in frame.columns:
        return _empty_figure("Select a target to show its distribution")
    counts = frame[target].fillna("<missing>").astype(str).value_counts().head(30)
    figure = go.Figure(
        go.Bar(
            x=counts.index.tolist(),
            y=counts.values.tolist(),
            marker=dict(
                color=_PROFILE_BAR_COLOR,
                line=dict(color=_PROFILE_BAR_EDGE_COLOR, width=0.6),
            ),
        )
    )
    figure.update_yaxes(title="Rows")
    return _figure_layout(figure, f"Target distribution · {target}", 290)


def _cards(summary: dict[str, Any]) -> html.Div:
    entries = [
        ("Rows", f"{summary['rows']:,}"),
        ("Variables", f"{summary['columns']:,}"),
        ("Missing", f"{summary['missing_pct']:.2f}%"),
        ("Duplicate rows", f"{summary['duplicate_rows']:,}"),
        ("Numeric", f"{summary['numeric']:,}"),
        ("Categorical", f"{summary['categorical']:,}"),
        ("Datetime", f"{summary['datetime']:,}"),
        ("Constant", f"{summary['constant']:,}"),
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(label, className="profile-card-label"),
                    html.Div(value, className="profile-card-value"),
                ],
                className="profile-card",
            )
            for label, value in entries
        ],
        className="profile-card-grid",
    )


def _findings(items: list[dict[str, str]]) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(item["title"], className="profile-finding-title"),
                    html.Div(item["detail"], className="profile-finding-detail"),
                ],
                className=f"profile-finding profile-finding-{item['level']}",
            )
            for item in items
        ],
        className="profile-findings",
    )


def _preview_table(frame: pd.DataFrame) -> html.Div:
    visible = frame.iloc[:5, :30].copy()
    visible.columns = list(map(str, visible.columns))
    for column in visible.columns:
        visible[column] = visible[column].map(
            lambda value: "" if pd.isna(value) else str(value)[:80]
        )
    note = None
    if frame.shape[1] > visible.shape[1]:
        note = html.Div(
            f"Preview shows the first {visible.shape[1]} of {frame.shape[1]:,} columns.",
            className="profile-preview-meta",
        )
    table = dash_table.DataTable(
        data=visible.to_dict("records"),
        columns=[{"name": column, "id": column} for column in visible.columns],
        page_size=5,
        style_table={"overflowX": "auto"},
        style_cell={
            "fontSize": ".72rem",
            "padding": "5px 7px",
            "minWidth": "95px",
            "maxWidth": "210px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_header={"fontWeight": "700"},
    )
    return html.Div([table, note] if note is not None else [table])


def compact_preview(reference: dict[str, Any] | None) -> html.Div:
    dataset = get_profile_dataset(reference)
    if dataset is None:
        return html.Div("Select or upload a dataset to preview it.", className="profile-empty")
    frame = dataset["frame"]
    dtype_counts = frame.dtypes.astype(str).value_counts()
    dtype_text = " · ".join(f"{count} {dtype}" for dtype, count in dtype_counts.head(4).items())
    return html.Div(
        [
            html.Div(
                [
                    html.Div(dataset["label"], className="profile-preview-title"),
                    html.Div(
                        f"{len(frame):,} rows × {frame.shape[1]:,} columns · {dtype_text}",
                        className="profile-preview-meta",
                    ),
                ],
                className="profile-preview-head",
            ),
            _preview_table(frame),
        ]
    )


def _scope_options(total_rows: int) -> tuple[list[dict[str, Any]], str]:
    if total_rows <= 10_000:
        return ([{"label": f"All {total_rows:,} rows", "value": "full", "disabled": True}], "full")
    return (
        [
            {"label": "Sample 10,000 rows", "value": "sample"},
            {"label": f"All {total_rows:,} rows", "value": "full"},
        ],
        "sample",
    )


def _predictor_view_summary(frame: pd.DataFrame, roles: dict[str, Any]) -> html.Div:
    omitted = predictor_exclusions(frame, roles)
    if omitted:
        chips = []
        for item in omitted:
            badges = [html.Span(item["reason"], className="profile-role-badge")]
            if item["sensitive"]:
                badges.append(
                    html.Span("Sensitive", className="profile-role-badge profile-role-sensitive")
                )
            chips.append(
                html.Div(
                    [html.Span(item["variable"], className="profile-excluded-name"), *badges],
                    className="profile-excluded-chip",
                )
            )
        omitted_block = html.Div(
            [
                html.Div(
                    f"Excluded from predictors ({len(omitted):,})",
                    className="profile-excluded-title",
                ),
                html.Div(chips, className="profile-excluded-list"),
            ]
        )
    else:
        omitted_block = html.Div(
            "No columns are currently excluded from predictors.",
            className="profile-excluded-empty",
        )
    return html.Div(
        [
            html.Div(
                [
                    html.Strong("All columns"),
                    " shows the selected or uploaded dataset. ",
                    html.Strong("Predictor columns"),
                    " shows source columns passed to model training after removing the target, ID, and user-excluded columns.",
                ],
                className="profile-view-help",
            ),
            omitted_block,
            html.Div(
                "Sensitive columns remain predictors unless separately excluded. Counts are before encoding, binning, pattern generation, or augmented-pair construction.",
                className="profile-view-footnote",
            ),
        ],
        className="profile-view-summary",
    )


def profile_section() -> html.Div:
    """Return the preview and collapsed detailed profiler used in Setup."""
    detailed = [
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Profiling rows", className="setup-lbl"),
                        dcc.RadioItems(
                            id="profile-scope-mode",
                            options=[
                                {"label": "Select a dataset", "value": "sample", "disabled": True}
                            ],
                            value="sample",
                            inline=True,
                            className="profile-scope-choice",
                        ),
                    ],
                    className="profile-control-block",
                ),
                html.Div(
                    [
                        html.Div("Profile view", className="setup-lbl"),
                        dcc.RadioItems(
                            id="profile-data-view",
                            options=[
                                {"label": "All columns", "value": "raw"},
                                {"label": "Predictor columns", "value": "model"},
                            ],
                            value="raw",
                            inline=True,
                            className="profile-scope-choice",
                        ),
                    ],
                    className="profile-control-block",
                ),
            ],
            className="profile-toolbar",
        ),
        html.Div(id="profile-view-summary"),
        html.Div(id="profile-scope-note", className="profile-scope-note"),
        dbc.Tabs(
            [
                dbc.Tab(
                    [
                        html.Div(id="profile-summary-cards"),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="profile-type-chart", config={"displayModeBar": False}
                                ),
                                dcc.Graph(
                                    id="profile-target-chart", config={"displayModeBar": False}
                                ),
                            ],
                            className="profile-two-col",
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="profile-overview-missing-chart",
                                    config={"displayModeBar": False},
                                ),
                                html.Div(
                                    [
                                        html.Div("Findings", className="profile-panel-title"),
                                        html.Div(id="profile-findings"),
                                    ],
                                    className="profile-panel",
                                ),
                            ],
                            className="profile-two-col",
                        ),
                    ],
                    label="Overview",
                    tab_id="profile-tab-overview",
                ),
                dbc.Tab(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Variables", className="profile-panel-title"),
                                    dcc.Dropdown(
                                        id="profile-variable-select",
                                        options=[],
                                        value=None,
                                        searchable=True,
                                        clearable=False,
                                    ),
                                    dash_table.DataTable(
                                        id="profile-variable-table",
                                        data=[],
                                        columns=[],
                                        page_size=12,
                                        sort_action="native",
                                        filter_action="native",
                                        row_selectable="single",
                                        selected_rows=[],
                                        style_table={
                                            "overflowX": "auto",
                                            "maxHeight": "520px",
                                            "overflowY": "auto",
                                        },
                                        style_cell={
                                            "fontSize": ".70rem",
                                            "padding": "5px 6px",
                                            "minWidth": "80px",
                                            "maxWidth": "165px",
                                            "overflow": "hidden",
                                            "textOverflow": "ellipsis",
                                        },
                                        style_header={"fontWeight": "700"},
                                    ),
                                ],
                                className="profile-variable-list",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        id="profile-variable-heading",
                                        className="profile-panel-title",
                                    ),
                                    html.Div(id="profile-variable-metrics"),
                                    dcc.Graph(
                                        id="profile-variable-chart",
                                        config={
                                            "displaylogo": False,
                                            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                        },
                                    ),
                                    html.Div(id="profile-variable-values"),
                                ],
                                className="profile-variable-detail",
                            ),
                        ],
                        className="profile-variable-grid",
                    ),
                    label="Variables",
                    tab_id="profile-tab-variables",
                ),
                dbc.Tab(
                    [
                        html.Div(id="profile-missing-cards"),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="profile-missing-bar", config={"displayModeBar": False}
                                ),
                                dcc.Graph(
                                    id="profile-missing-map", config={"displayModeBar": False}
                                ),
                            ],
                            className="profile-two-col",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Top missingness patterns", className="profile-panel-title"
                                ),
                                html.Div(id="profile-missing-patterns"),
                            ],
                            className="profile-panel",
                        ),
                    ],
                    label="Missingness",
                    tab_id="profile-tab-missingness",
                ),
                dbc.Tab(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id="profile-correlation-chart", config={"displayModeBar": False}
                                ),
                                dcc.Graph(
                                    id="profile-association-chart", config={"displayModeBar": False}
                                ),
                            ],
                            className="profile-relationship-grid",
                        ),
                        html.Div(id="profile-relationship-note", className="profile-scope-note"),
                    ],
                    label="Relationships",
                    tab_id="profile-tab-relationships",
                ),
            ],
            id="profile-tabs",
            active_tab="profile-tab-overview",
            className="profile-tabs",
        ),
    ]
    return html.Div(
        [
            dcc.Store(id="st-profile-dataset"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("DATA PROFILE", className="setup-lbl"),
                            html.Div(
                                "Preview is lightweight. Detailed statistics are calculated only after expansion.",
                                className="profile-preview-hint",
                            ),
                        ],
                        className="profile-preview-label-row",
                    ),
                    html.Div(id="profile-preview"),
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                detailed,
                                title="Open detailed profile",
                                item_id="profile-open",
                            )
                        ],
                        id="profile-accordion",
                        active_item=None,
                        always_open=False,
                        className="data-profile-accordion",
                    ),
                ],
                className="profile-preview-card",
            ),
        ],
        className="data-profile-section",
    )


def _metric_grid(values: dict[str, Any]) -> html.Div:
    items = []
    for label, value in values.items():
        if isinstance(value, float):
            rendered = f"{value:,.4g}"
        elif value is None:
            rendered = "N/A"
        else:
            rendered = f"{value:,}" if isinstance(value, int) else str(value)
        items.append(
            html.Div(
                [
                    html.Div(label, className="profile-mini-label"),
                    html.Div(rendered, className="profile-mini-value"),
                ],
                className="profile-mini-card",
            )
        )
    return html.Div(items, className="profile-mini-grid")


def _numeric_figure(series: pd.Series, name: str) -> go.Figure:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return _empty_figure("No finite numeric values")
    if values.nunique() <= 20:
        counts = values.value_counts().sort_index()
        figure = go.Figure(
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                name="Count",
                marker=dict(
                    color=_PROFILE_BAR_COLOR,
                    line=dict(color=_PROFILE_BAR_EDGE_COLOR, width=0.6),
                ),
            )
        )
        figure.update_yaxes(title="Rows")
        return _figure_layout(figure, f"Frequency · {name}", 350)
    plot_values = (
        values.sample(n=min(10_000, len(values)), random_state=2026)
        if len(values) > 10_000
        else values
    )
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Histogram(
            x=plot_values,
            nbinsx=min(60, max(15, int(np.sqrt(len(plot_values))))),
            histnorm="probability density",
            name="Histogram",
            opacity=0.78,
            marker=dict(
                color=_PROFILE_HISTOGRAM_COLOR,
                line=dict(color=_PROFILE_BAR_EDGE_COLOR, width=0.45),
            ),
        ),
        secondary_y=False,
    )
    if plot_values.nunique() > 2 and len(plot_values) >= 20:
        try:
            density = gaussian_kde(plot_values.to_numpy(dtype=float))
            x_grid = np.linspace(float(plot_values.min()), float(plot_values.max()), 240)
            figure.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=density(x_grid),
                    mode="lines",
                    name="Density",
                    line=dict(color=_PROFILE_DENSITY_COLOR, width=2.4),
                ),
                secondary_y=True,
            )
        except Exception:
            pass
    figure.update_xaxes(title=name)
    figure.update_yaxes(title="Density", secondary_y=False)
    figure.update_yaxes(title="KDE", secondary_y=True, showgrid=False)
    return _figure_layout(figure, f"Distribution · {name}", 360)


def _categorical_figure(series: pd.Series, name: str) -> tuple[go.Figure, pd.DataFrame]:
    counts = series.fillna("<missing>").astype(str).value_counts()
    top = counts.head(CATEGORY_DISPLAY_LIMIT).copy()
    if len(counts) > CATEGORY_DISPLAY_LIMIT:
        top.loc["Other"] = int(counts.iloc[CATEGORY_DISPLAY_LIMIT:].sum())
    figure = go.Figure(
        go.Bar(
            x=top.values.tolist(),
            y=top.index.tolist(),
            orientation="h",
            marker=dict(
                color=_PROFILE_BAR_COLOR,
                line=dict(color=_PROFILE_BAR_EDGE_COLOR, width=0.6),
            ),
        )
    )
    figure.update_xaxes(title="Rows")
    figure.update_yaxes(autorange="reversed")
    table = pd.DataFrame(
        {
            "value": counts.index.astype(str),
            "rows": counts.values.astype(int),
            "percentage": (counts.values / max(counts.sum(), 1) * 100).round(2),
        }
    )
    return _figure_layout(figure, f"Top values · {name}", max(340, 24 * len(top) + 90)), table.head(
        50
    )


def _datetime_figure(series: pd.Series, name: str) -> go.Figure:
    values = pd.to_datetime(series, errors="coerce", utc=True).dropna()
    if values.empty:
        return _empty_figure("No valid dates")
    span_days = max((values.max() - values.min()).days, 0)
    period = "D" if span_days <= 90 else "W" if span_days <= 730 else "M"
    counts = values.dt.to_period(period).astype(str).value_counts().sort_index()
    figure = go.Figure(
        go.Scatter(x=counts.index.tolist(), y=counts.values.tolist(), mode="lines+markers")
    )
    figure.update_yaxes(title="Rows")
    return _figure_layout(figure, f"Observations over time · {name}", 350)


def _missing_map(frame: pd.DataFrame) -> go.Figure:
    columns = [str(column) for column in frame.columns if frame[column].isna().any()]
    if not columns:
        return _empty_figure("No missing values")
    columns = sorted(columns, key=lambda column: frame[column].isna().mean(), reverse=True)[:40]
    sampled = frame[columns]
    if len(sampled) > MISSINGNESS_MAP_ROW_LIMIT:
        sampled = sampled.sample(n=MISSINGNESS_MAP_ROW_LIMIT, random_state=2026).sort_index()
    values = sampled.isna().astype(int).to_numpy().T
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=list(range(len(sampled))),
            y=columns,
            colorscale=[[0, "rgba(0,0,0,0.08)"], [1, "rgba(220,38,38,0.85)"]],
            showscale=False,
            hovertemplate="Variable: %{y}<br>Sample row: %{x}<br>Missing: %{z}<extra></extra>",
        )
    )
    figure.update_xaxes(title="Sampled row position")
    return _figure_layout(figure, "Missingness map", max(310, 20 * len(columns) + 110))


def _correlation_figure(frame: pd.DataFrame) -> tuple[go.Figure, str]:
    numeric = frame.select_dtypes(include=[np.number, "bool"]).replace([np.inf, -np.inf], np.nan)
    if numeric.shape[1] < 2:
        return _empty_figure("At least two numeric variables are required"), ""
    variances = numeric.var(numeric_only=True).sort_values(ascending=False)
    selected = list(map(str, variances.head(CORRELATION_COLUMN_LIMIT).index))
    corr = numeric[selected].corr()
    figure = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=selected,
            y=selected,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="r"),
        )
    )
    note = ""
    if numeric.shape[1] > CORRELATION_COLUMN_LIMIT:
        note = f"Correlation view is limited to the {CORRELATION_COLUMN_LIMIT} highest-variance numeric variables."
    return _figure_layout(figure, "Numeric correlations", 520), note


def _association_figure(associations: pd.DataFrame, target: str | None) -> go.Figure:
    if associations.empty:
        return _empty_figure("Select a target with at least two classes")
    ordered = associations.sort_values("association", ascending=False).copy()
    variables = ordered["variable"].astype(str).tolist()
    figure = go.Figure(
        go.Bar(
            x=variables,
            y=ordered["association"],
            marker=dict(
                color=_PROFILE_BAR_COLOR,
                line=dict(color=_PROFILE_BAR_EDGE_COLOR, width=0.6),
            ),
            customdata=ordered[["method"]],
            hovertemplate=(
                "Variable: %{x}<br>Association: %{y:.3f}<br>%{customdata[0]}<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(
        title="Variable",
        tickangle=-45,
        automargin=True,
        categoryorder="array",
        categoryarray=variables,
    )
    figure.update_yaxes(
        title="Association",
        range=[0, max(1.0, float(ordered["association"].max()) * 1.05)],
    )
    figure = _figure_layout(figure, f"Univariate association with {target}", 420)
    longest_name = max((len(name) for name in variables), default=0)
    bottom_margin = min(190, max(105, 70 + longest_name * 3))
    figure.update_layout(margin=dict(l=55, r=20, t=48, b=bottom_margin), bargap=0.24)
    return figure


def register_profile_callbacks(app) -> None:
    """Register lazy profiling callbacks on the Dash application."""

    @app.callback(
        Output("st-profile-dataset", "data"),
        Output("profile-preview", "children"),
        Input("ri-source", "value"),
        Input("dd-demo", "value"),
        Input("ul-file", "contents"),
        State("ul-file", "filename"),
    )
    def _dataset_reference(source, demo_key, upload_contents, upload_name):
        try:
            reference = register_profile_dataset(source, demo_key, upload_contents, upload_name)
        except Exception as exc:
            return None, html.Div(f"Profile preview unavailable: {exc}", className="warn-b")
        return reference, compact_preview(reference)

    @app.callback(
        Output("profile-scope-mode", "options"),
        Output("profile-scope-mode", "value"),
        Input("st-profile-dataset", "data"),
    )
    def _scope_control(reference):
        dataset = get_profile_dataset(reference)
        if dataset is None:
            return ([{"label": "Select a dataset", "value": "sample", "disabled": True}], "sample")
        return _scope_options(len(dataset["frame"]))

    @app.callback(
        Output("profile-data-view", "options"),
        Output("profile-view-summary", "children"),
        Input("st-profile-dataset", "data"),
        Input("ul-target", "value"),
        Input("ul-id", "value"),
        Input("ul-excl", "value"),
        Input("ul-sens", "value"),
    )
    def _view_control(reference, target, id_column, excluded, sensitive):
        dataset = get_profile_dataset(reference)
        if dataset is None:
            return (
                [
                    {"label": "All columns", "value": "raw"},
                    {"label": "Predictor columns", "value": "model"},
                ],
                html.Div(),
            )
        roles = _roles(reference, target, id_column, excluded, sensitive)
        frame = dataset["frame"]
        all_count = int(frame.shape[1])
        predictor_count = len(model_input_columns(frame, roles))
        options = [
            {"label": f"All columns ({all_count:,})", "value": "raw"},
            {"label": f"Predictor columns ({predictor_count:,})", "value": "model"},
        ]
        return options, _predictor_view_summary(frame, roles)

    @app.callback(
        Output("profile-scope-note", "children"),
        Output("profile-summary-cards", "children"),
        Output("profile-type-chart", "figure"),
        Output("profile-target-chart", "figure"),
        Output("profile-overview-missing-chart", "figure"),
        Output("profile-findings", "children"),
        Output("profile-variable-table", "data"),
        Output("profile-variable-table", "columns"),
        Output("profile-variable-select", "options"),
        Output("profile-variable-select", "value"),
        Output("profile-missing-cards", "children"),
        Output("profile-missing-bar", "figure"),
        Output("profile-missing-map", "figure"),
        Output("profile-missing-patterns", "children"),
        Output("profile-correlation-chart", "figure"),
        Output("profile-association-chart", "figure"),
        Output("profile-relationship-note", "children"),
        Input("profile-accordion", "active_item"),
        Input("profile-scope-mode", "value"),
        Input("profile-data-view", "value"),
        Input("st-profile-dataset", "data"),
        Input("ul-target", "value"),
        Input("ul-id", "value"),
        Input("ul-excl", "value"),
        Input("ul-sens", "value"),
        State("profile-variable-select", "value"),
    )
    def _profile_overview(
        active, mode, view, reference, target, id_column, excluded, sensitive, current_variable
    ):
        if active != "profile-open":
            return (no_update,) * 17
        if not reference:
            empty = _empty_figure("Select or upload a dataset")
            return (
                "Select or upload a dataset.",
                html.Div(),
                empty,
                empty,
                empty,
                html.Div(),
                [],
                [],
                [],
                None,
                html.Div(),
                empty,
                empty,
                html.Div(),
                empty,
                empty,
                "",
            )
        roles = _roles(reference, target, id_column, excluded, sensitive)
        try:
            context = _profile_context(reference, mode or "sample", roles, view or "raw")
        except Exception as exc:
            empty = _empty_figure(str(exc))
            return (
                str(exc),
                html.Div(),
                empty,
                empty,
                empty,
                html.Div(),
                [],
                [],
                [],
                None,
                html.Div(),
                empty,
                empty,
                html.Div(),
                empty,
                empty,
                "",
            )
        frame = context["frame"]
        relationship_frame = context["relationship_frame"]
        inventory = context["inventory"]
        scope = context["scope"]
        rows = inventory.drop(columns=["position"], errors="ignore").to_dict("records")
        visible_columns = [
            "variable",
            "role",
            "semantic_type",
            "missing_pct",
            "unique",
            "most_frequent",
            "warning_count",
        ]
        columns = [{"name": name.replace("_", " ").title(), "id": name} for name in visible_columns]
        options = [
            {"label": f"{row.variable} · {row.semantic_type}", "value": row.variable}
            for row in inventory.itertuples()
        ]
        names = set(inventory["variable"].tolist()) if not inventory.empty else set()
        if current_variable in names:
            selected = current_variable
        elif not inventory.empty:
            preferred = inventory[
                inventory["role"].eq("Predictor")
                & ~inventory["semantic_type"].isin(["identifier-like", "constant", "all missing"])
            ]
            selected = (preferred.iloc[0] if not preferred.empty else inventory.iloc[0])["variable"]
        else:
            selected = None
        summary = context["summary"]
        missing_columns = int((frame.isna().sum() > 0).sum())
        rows_with_missing = int(frame.isna().any(axis=1).sum()) if len(frame) else 0
        missing_cards = _metric_grid(
            {
                "Missing cells": int(frame.isna().sum().sum()),
                "Variables affected": missing_columns,
                "Rows affected": rows_with_missing,
                "Complete rows": int(len(frame) - rows_with_missing),
            }
        )
        patterns = missingness_patterns(frame)
        pattern_table = dash_table.DataTable(
            data=patterns.to_dict("records"),
            columns=[{"name": column.title(), "id": column} for column in patterns.columns],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": ".72rem", "padding": "5px 7px"},
            style_header={"fontWeight": "700"},
        )
        corr_figure, corr_note = _correlation_figure(frame)
        associations = target_associations(relationship_frame, inventory, roles.get("target"))
        view_note = "All columns" if context["view"] == "raw" else "Predictor columns"
        note = f"{scope.label} · {view_note}. Summary statistics use the selected profiling rows; chart rendering remains bounded."
        if mode == "full" and scope.total_rows > 10_000:
            note += " Full-data profiling can take longer on large datasets."
        relationship_note = corr_note
        if scope.sampled:
            relationship_note = (
                relationship_note + " " if relationship_note else ""
            ) + "Relationships are computed on the deterministic profiling sample."
        return (
            note,
            _cards(summary),
            _type_figure(inventory),
            _target_figure(relationship_frame, roles.get("target")),
            _missing_bar(frame),
            _findings(context["findings"]),
            rows,
            columns,
            options,
            selected,
            missing_cards,
            _missing_bar(frame, 40),
            _missing_map(frame),
            pattern_table,
            corr_figure,
            _association_figure(associations, roles.get("target")),
            relationship_note,
        )

    @app.callback(
        Output("profile-variable-select", "value", allow_duplicate=True),
        Input("profile-variable-table", "derived_virtual_selected_rows"),
        State("profile-variable-table", "derived_virtual_data"),
        State("profile-variable-table", "data"),
        prevent_initial_call=True,
    )
    def _table_selection(selected_rows, virtual_data, raw_data):
        if not selected_rows:
            return no_update
        records = virtual_data if virtual_data is not None else raw_data
        position = int(selected_rows[0])
        if not records or position >= len(records):
            return no_update
        return records[position].get("variable", no_update)

    @app.callback(
        Output("profile-variable-heading", "children"),
        Output("profile-variable-metrics", "children"),
        Output("profile-variable-chart", "figure"),
        Output("profile-variable-values", "children"),
        Input("profile-variable-select", "value"),
        Input("profile-scope-mode", "value"),
        Input("profile-data-view", "value"),
        Input("st-profile-dataset", "data"),
        Input("ul-target", "value"),
        Input("ul-id", "value"),
        Input("ul-excl", "value"),
        Input("ul-sens", "value"),
        State("profile-accordion", "active_item"),
    )
    def _variable_detail(
        variable, mode, view, reference, target, id_column, excluded, sensitive, active
    ):
        if active != "profile-open" or not variable or not reference:
            return "Variable detail", html.Div(), _empty_figure("Select a variable"), html.Div()
        roles = _roles(reference, target, id_column, excluded, sensitive)
        context = _profile_context(reference, mode or "sample", roles, view or "raw")
        frame = context["frame"]
        if variable not in frame.columns:
            return (
                "Variable detail",
                html.Div(),
                _empty_figure("Variable is not in this view"),
                html.Div(),
            )
        series = frame[variable]
        inferred = semantic_type(series)
        heading = html.Div(
            [html.Span(variable), html.Span(inferred, className="profile-type-badge")],
            className="profile-heading-row",
        )
        if inferred in {"continuous numeric", "discrete numeric", "numeric text"}:
            metrics = _metric_grid(numeric_profile(series))
            figure = _numeric_figure(series, variable)
            values = html.Div()
        elif inferred in {"datetime", "datetime text"}:
            metrics = _metric_grid(datetime_profile(series))
            figure = _datetime_figure(series, variable)
            values = html.Div()
        elif inferred in {"categorical", "binary", "constant", "all missing"}:
            metrics = _metric_grid(categorical_profile(series))
            figure, table = _categorical_figure(series, variable)
            values = dash_table.DataTable(
                data=table.to_dict("records"),
                columns=[{"name": column.title(), "id": column} for column in table.columns],
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": ".72rem", "padding": "5px 7px"},
                style_header={"fontWeight": "700"},
            )
        else:
            metrics = _metric_grid(text_profile(series))
            figure, table = _categorical_figure(series, variable)
            values = dash_table.DataTable(
                data=table.to_dict("records"),
                columns=[{"name": column.title(), "id": column} for column in table.columns],
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": ".72rem", "padding": "5px 7px"},
                style_header={"fontWeight": "700"},
            )
        return heading, metrics, figure, values

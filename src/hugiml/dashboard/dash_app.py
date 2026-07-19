"""HUGIML Governance Studio Dash interface.

The application keeps workspace selection and page navigation in a pinned header.
Workbench provides equal Setup and Results views. Dataset selection, uploads,
column roles, training controls, and experiment configuration live only in Setup.
Governance exposes its audit pages through the same pinned navigation region.
"""

from __future__ import annotations

import argparse
import threading
import traceback

import numpy as np

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback_context, dcc, html, no_update

from hugiml.dashboard.dash_components.data_utils import (
    DEMO_DATASETS,
    cache_key,
    fingerprint,
    load_demo,
    read_upload,
)
from hugiml.dashboard.dash_components.styles import THEMES, build_css
from hugiml.dashboard.data import prepare_model_frame
from hugiml.dashboard.runner import (
    fit_feature_pruned_hugiml,
    fit_hugiml_config,
    fit_representation_pruned_downstream,
)

try:
    from hugiml.rpte_bounded_lookahead_leafwise import LEAF_CONFIGS as RPTE_LC
    from hugiml.rpte_bounded_lookahead_leafwise import LeafWiseBoundedLookaheadRPTEFeatureLR

    _RPTE = True
except Exception:
    _RPTE = False
    RPTE_LC = {}

_CTX: dict[str, dict] = {}
_AUX: dict = {}
PAGES = [
    "Overview",
    "Validation",
    "Representation Audit",
    "Pattern Inventory",
    "Case Review",
    "Data Quality & Policy",
    "Configuration Comparison",
    "Representation Pruning",
    "Monitoring",
]

# ─── Layout ───────────────────────────────────────────────────────────────────


def _workbench_data_setup():
    demo_opts = [{"label": v["label"], "value": k} for k, v in DEMO_DATASETS.items()]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("DATA", className="setup-lbl"),
                            dcc.RadioItems(
                                id="ri-source",
                                options=[
                                    {"label": "Demo", "value": "demo"},
                                    {"label": "Upload", "value": "upload"},
                                ],
                                value="demo",
                                inline=True,
                                inputStyle={"marginRight": "4px", "marginLeft": "8px"},
                                className="compact-choice",
                            ),
                            html.Div(
                                id="demo-box",
                                children=[
                                    html.Label("Dataset", className="form-label"),
                                    dcc.Dropdown(
                                        id="dd-demo",
                                        options=demo_opts,
                                        value="credit",
                                        clearable=False,
                                        className="compact-dropdown",
                                    ),
                                ],
                            ),
                            html.Div(
                                id="upload-box",
                                style={"display": "none"},
                                children=[
                                    dcc.Upload(
                                        id="ul-file",
                                        className="upload-dropzone compact-upload",
                                        children=html.Div(
                                            [
                                                html.Span("Choose file", className="upload-title"),
                                                html.Span(
                                                    "  CSV · Excel · Parquet",
                                                    className="upload-subtitle",
                                                ),
                                            ]
                                        ),
                                        multiple=False,
                                    ),
                                    html.Div(id="ul-status", className="upload-status"),
                                ],
                            ),
                        ],
                        className="setup-card setup-card-compact",
                    ),
                    html.Div(
                        [
                            html.Div("RUN SETTINGS", className="setup-lbl"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("CV folds", className="form-label"),
                                            dcc.Input(
                                                id="in-cv",
                                                type="number",
                                                value=3,
                                                min=2,
                                                max=10,
                                                step=1,
                                                className="form-control form-control-sm",
                                            ),
                                        ],
                                        className="setup-field",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Random seed", className="form-label"),
                                            dcc.Input(
                                                id="in-seed",
                                                type="number",
                                                value=2026,
                                                min=0,
                                                max=999999,
                                                className="form-control form-control-sm",
                                            ),
                                        ],
                                        className="setup-field",
                                    ),
                                ],
                                className="setup-field-grid",
                            ),
                            html.Div(
                                "Dataset preparation, model fitting, and comparison run together from the single action below.",
                                className="setup-help",
                            ),
                        ],
                        className="setup-card setup-card-compact",
                    ),
                ],
                className="setup-command-grid",
            ),
            html.Div(
                id="upload-panel",
                style={"display": "none"},
                className="setup-card setup-card-compact upload-roles",
                children=[
                    html.Div("COLUMN ROLES", className="setup-lbl"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Target", className="form-label"),
                                    dcc.Dropdown(
                                        id="ul-target", options=[], value=None, clearable=False
                                    ),
                                ],
                                className="setup-field",
                            ),
                            html.Div(
                                [
                                    html.Label("ID column", className="form-label"),
                                    dcc.Dropdown(
                                        id="ul-id",
                                        options=[{"label": "<none>", "value": "__none__"}],
                                        value="__none__",
                                        clearable=False,
                                    ),
                                ],
                                className="setup-field",
                            ),
                            html.Div(
                                [
                                    html.Label("Exclude", className="form-label"),
                                    dcc.Dropdown(id="ul-excl", options=[], value=[], multi=True),
                                ],
                                className="setup-field",
                            ),
                            html.Div(
                                [
                                    html.Label("Sensitive", className="form-label"),
                                    dcc.Dropdown(id="ul-sens", options=[], value=[], multi=True),
                                ],
                                className="setup-field",
                            ),
                        ],
                        className="role-grid",
                    ),
                ],
            ),
        ],
        className="workbench-data-setup",
    )


def build_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("HUGIML Governance Studio", className="hero-ey"),
                                    dcc.Dropdown(
                                        id="theme-select",
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
                            html.Div(id="hero-box"),
                        ],
                        className="hug-hero",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Button(
                                        "Workbench", id="ws-wb", n_clicks=1, className="ws-btn act"
                                    ),
                                    html.Button(
                                        "Governance", id="ws-gov", n_clicks=0, className="ws-btn"
                                    ),
                                ],
                                className="workspace-switch",
                            ),
                        ],
                        className="hug-ctrl primary-nav",
                    ),
                    html.Div(
                        [
                            html.Div(
                                id="gov-nav-row",
                                style={"display": "none"},
                                children=[
                                    html.Button(
                                        page,
                                        id=f"nav-{i}",
                                        n_clicks=0,
                                        className="tab-btn" + (" act" if i == 0 else ""),
                                    )
                                    for i, page in enumerate(PAGES)
                                ],
                            ),
                            html.Div(
                                id="wb-tabs-row",
                                className="workbench-nav-row",
                                children=[
                                    html.Button(
                                        "Setup",
                                        id="wb-nav-setup",
                                        n_clicks=1,
                                        className="tab-btn act",
                                    ),
                                    html.Button(
                                        "Results",
                                        id="wb-nav-results",
                                        n_clicks=0,
                                        className="tab-btn",
                                    ),
                                ],
                            ),
                        ],
                        className="hug-tabrow",
                    ),
                ],
                className="hug-hdr",
            ),
            dcc.Loading(
                id="main-load",
                children=html.Div(id="main-out"),
                type="circle",
                color="var(--a)",
                overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                parent_className="hug-content",
            ),
            dcc.Store(id="st-ctx", storage_type="memory"),
            dcc.Store(id="st-ws", data="workbench"),
            dcc.Store(id="st-page", data="Overview"),
            dcc.Store(id="st-theme", data="Ocean"),
            dcc.Store(id="st-wb-sec", data="setup"),
        ],
        id="app-root",
        className="hug-wrap theme-ocean",
    )


# ─── Hero ─────────────────────────────────────────────────────────────────────


def _hero(ctx=None):
    """Always-visible dashboard summary below the studio title row."""
    if ctx is None:
        return html.Div(
            [
                html.H1("Workbench & Governance Dashboard", className="hero-h1"),
                html.P(
                    "Configure and compare models in Workbench, then promote a selected HUGIML run for governance review.",
                    className="hero-p",
                ),
            ]
        )
    return html.Div(
        [
            html.H1("Workbench & Governance Dashboard", className="hero-h1"),
            html.P(
                "Validation · Representation · Patterns · Cases · Data Quality · Configuration · Pruning · Monitoring",
                className="hero-p",
            ),
        ]
    )


# ─── Governance page router ───────────────────────────────────────────────────


def _gov_page(name, ctx):
    from hugiml.dashboard.dash_components.pages import (
        config_compare,
        data_quality,
        monitoring,
        overview,
        patterns,
        performance,
        prediction,
        pruning,
        representation,
    )

    _map = {
        "Overview": overview.render,
        "Validation": performance.render,
        "Representation Audit": representation.render,
        "Pattern Inventory": patterns.render,
        "Case Review": lambda c: prediction.render(c, row_idx=0),
        "Data Quality & Policy": data_quality.render,
        "Configuration Comparison": lambda c: config_compare.render(c, _AUX.get("cc_result")),
        "Representation Pruning": lambda c: pruning.render(
            c,
            prune_result=_AUX.get("pr_result"),
            prune_family=_AUX.get("pr_family", "Original features"),
            prune_removed=_AUX.get("pr_removed", []),
            rpte_rebuild_result=_AUX.get("rr_result"),
            rpte_removed=_AUX.get("rr_removed", []),
            rpte_X_pruned=_AUX.get("rr_X"),
        ),
        "Monitoring": monitoring.render,
    }
    fn = _map.get(name)
    if fn is None:
        return html.Div(f"Unknown: {name}", className="err-b")
    try:
        body = fn(ctx)
    except Exception:
        body = html.Div(
            [
                html.P(
                    f"Error rendering '{name}':", style={"color": "var(--rd)", "fontWeight": "bold"}
                ),
                html.Pre(
                    traceback.format_exc(),
                    style={"fontSize": ".72rem", "maxHeight": "300px", "overflowY": "auto"},
                ),
            ],
            className="err-b",
        )
    return body


def _prepare_data_context(
    source,
    demo_key,
    upload_contents,
    upload_name,
    target,
    id_column,
    excluded_columns,
    sensitive_columns,
    cv,
    random_state,
):
    """Create and cache the dataset context used by a Workbench experiment run."""
    cv = int(cv or 3)
    random_state = int(random_state or 2026)
    if source == "demo":
        data = load_demo(demo_key or "credit")
        key = cache_key(f"demo:{demo_key or 'credit'}", cv, random_state)
    else:
        if upload_contents is None:
            raise ValueError("Upload a file before running the selected models.")
        raw = read_upload(upload_contents, upload_name or "file")
        upload_fingerprint = fingerprint(raw)
        target = target or raw.columns[0]
        id_column = None if id_column == "__none__" else id_column
        excluded_columns = list(excluded_columns or [])
        sensitive_columns = list(sensitive_columns or [])
        X, y, case_ids, meta = prepare_model_frame(
            raw,
            target=target,
            id_column=id_column,
            excluded_columns=excluded_columns,
        )
        roles = {
            "target": target,
            "id_column": id_column,
            "excluded_columns": excluded_columns,
            "sensitive_columns": sensitive_columns,
        }
        data = {
            "mode": "Uploaded dataset",
            "X": X,
            "y": np.asarray(y, dtype=int),
            "case_ids": case_ids,
            "roles": roles,
            "meta": meta,
        }
        key = cache_key("upload", cv, random_state, upload_fingerprint, roles)
    existing = _CTX.get(key, {})
    context = {**existing, **data, "cv": cv, "random_state": random_state}
    _CTX[key] = context
    return key, context


# ─── App factory ──────────────────────────────────────────────────────────────


def create_app(debug=False):
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="HUGIML Governance Studio",
        suppress_callback_exceptions=True,
        update_title=None,
    )
    app.index_string = app.index_string.replace(
        "</head>", f"<style id='hug-theme'>{build_css('Ocean')}</style></head>"
    )
    app.layout = build_layout()
    _callbacks(app)
    try:
        from hugiml.dashboard.dash_components.workbench import register_callbacks

        register_callbacks(app)
    except Exception:
        pass
    return app


# ─── Callbacks ────────────────────────────────────────────────────────────────


def _callbacks(app):

    @app.callback(Output("st-theme", "data"), Input("theme-select", "value"))
    def _select_theme(theme):
        return theme if theme in THEMES else "Ocean"

    @app.callback(Output("app-root", "className"), Input("st-theme", "data"))
    def _apply_theme(theme):
        selected = theme if theme in THEMES else "Ocean"
        return f"hug-wrap theme-{selected.lower()}"

    # Data source toggle
    @app.callback(
        Output("demo-box", "style"),
        Output("upload-box", "style"),
        Output("upload-panel", "style"),
        Input("ri-source", "value"),
    )
    def _src(s):
        if s == "demo":
            return {}, {"display": "none"}, {"display": "none"}
        return {"display": "none"}, {}, {}

    # Upload → populate column dropdowns
    @app.callback(
        Output("ul-status", "children"),
        Output("ul-target", "options"),
        Output("ul-target", "value"),
        Output("ul-id", "options"),
        Output("ul-excl", "options"),
        Output("ul-sens", "options"),
        Input("ul-file", "contents"),
        State("ul-file", "filename"),
    )
    def _upload(content, fname):
        blank = ("", [], None, [{"label": "<none>", "value": "__none__"}], [], [])
        if content is None:
            return blank
        try:
            df = read_upload(content, fname or "file")
        except Exception as e:
            return (
                html.Span(f"Error: {e}", style={"color": "var(--rd)"}),
                [],
                None,
                [{"label": "<none>", "value": "__none__"}],
                [],
                [],
            )
        cols = [{"label": c, "value": c} for c in df.columns]
        id_opts = [{"label": "<none>", "value": "__none__"}] + cols
        tgt = next((c for c in df.columns if c.lower() == "target"), df.columns[0])
        return (
            html.Span(
                f"✓ {df.shape[0]:,} × {df.shape[1]:,} — {fname}", style={"color": "var(--gr)"}
            ),
            cols,
            tgt,
            id_opts,
            cols,
            cols,
        )

    # Workspace toggle → update button styles + tab row visibility
    @app.callback(
        Output("st-ws", "data"),
        Output("ws-wb", "className"),
        Output("ws-gov", "className"),
        Output("gov-nav-row", "style"),
        Output("wb-tabs-row", "style"),
        Input("ws-wb", "n_clicks"),
        Input("ws-gov", "n_clicks"),
        prevent_initial_call=True,
    )
    def _ws(wb, gov):
        tid = (
            callback_context.triggered[0]["prop_id"].split(".")[0]
            if callback_context.triggered
            else ""
        )
        ws = "governance" if tid == "ws-gov" else "workbench"
        gov_style = (
            {"display": "flex", "alignItems": "stretch", "gap": "0", "flexWrap": "nowrap"}
            if ws == "governance"
            else {"display": "none"}
        )
        wb_style = (
            {"display": "flex", "alignItems": "stretch", "gap": "0", "flexWrap": "nowrap"}
            if ws == "workbench"
            else {"display": "none"}
        )
        return (
            ws,
            "ws-btn act" if ws == "workbench" else "ws-btn",
            "ws-btn act" if ws == "governance" else "ws-btn",
            gov_style,
            wb_style,
        )

    # Sync workspace buttons when st-ws changes (e.g. after Promote)
    @app.callback(
        Output("ws-wb", "className", allow_duplicate=True),
        Output("ws-gov", "className", allow_duplicate=True),
        Output("gov-nav-row", "style", allow_duplicate=True),
        Output("wb-tabs-row", "style", allow_duplicate=True),
        Input("st-ws", "data"),
        prevent_initial_call=True,
    )
    def _sync_ws(ws):
        if ws is None:
            return no_update, no_update, no_update, no_update
        gov_style = (
            {"display": "flex", "alignItems": "stretch", "gap": "0", "flexWrap": "nowrap"}
            if ws == "governance"
            else {"display": "none"}
        )
        wb_style = (
            {"display": "flex", "alignItems": "stretch", "gap": "0", "flexWrap": "nowrap"}
            if ws == "workbench"
            else {"display": "none"}
        )
        return (
            "ws-btn act" if ws == "workbench" else "ws-btn",
            "ws-btn act" if ws == "governance" else "ws-btn",
            gov_style,
            wb_style,
        )

    @app.callback(
        Output("st-wb-sec", "data", allow_duplicate=True),
        Input("wb-nav-setup", "n_clicks"),
        Input("wb-nav-results", "n_clicks"),
        prevent_initial_call=True,
    )
    def _wb_nav(setup_clicks, result_clicks):
        tid = (
            callback_context.triggered[0]["prop_id"].split(".")[0]
            if callback_context.triggered
            else ""
        )
        return "results" if tid == "wb-nav-results" else "setup"

    @app.callback(
        Output("wb-nav-setup", "className"),
        Output("wb-nav-results", "className"),
        Input("st-wb-sec", "data"),
    )
    def _wb_nav_highlight(section):
        return (
            "tab-btn act" if section != "results" else "tab-btn",
            "tab-btn act" if section == "results" else "tab-btn",
        )

    # Governance nav: page selection
    @app.callback(
        Output("st-page", "data"),
        [Input(f"nav-{i}", "n_clicks") for i in range(len(PAGES))],
        prevent_initial_call=True,
    )
    def _nav(*_):
        tid = (
            callback_context.triggered[0]["prop_id"].split(".")[0]
            if callback_context.triggered
            else ""
        )
        for i, p in enumerate(PAGES):
            if tid == f"nav-{i}":
                return p
        return no_update

    @app.callback(
        [Output(f"nav-{i}", "className") for i in range(len(PAGES))], Input("st-page", "data")
    )
    def _navhl(page):
        return ["tab-btn act" if p == page else "tab-btn" for p in PAGES]

    # Main router: hero-box + main content (triggers on all state changes)
    @app.callback(
        Output("hero-box", "children"),
        Output("main-out", "children"),
        Input("st-ctx", "data"),
        Input("st-ws", "data"),
        Input("st-page", "data"),
        Input("st-wb-sec", "data"),
    )
    def _render(ctx_ref, ws, page, wb_sec):
        full = None
        if ctx_ref is not None and "ck" in ctx_ref:
            full = _CTX.get(ctx_ref.get("ck", ""))

        if full is None:
            hero = _hero(None)
            if ws == "workbench":
                if wb_sec == "results":
                    try:
                        from hugiml.dashboard.dash_components.workbench import _results_panel

                        return hero, html.Div(_results_panel(None), className="workbench-page")
                    except Exception:
                        return hero, html.Div("Results are not available yet.", className="info-b")
                try:
                    from hugiml.dashboard.dash_components.workbench import render_setup

                    return hero, html.Div(
                        [
                            _workbench_data_setup(),
                            html.Div(render_setup(None), className="experiment-setup-card"),
                        ],
                        className="workbench-page",
                    )
                except Exception:
                    return hero, html.Div(traceback.format_exc(), className="err-b")
            return hero, html.Div(
                [
                    html.Div(
                        "Run and promote a HUGIML model from Workbench before opening Governance.",
                        className="info-b",
                    )
                ]
            )

        hero = _hero(full)
        if ws == "workbench":
            try:
                from hugiml.dashboard.dash_components.workbench import _results_panel, render_setup

                if wb_sec == "results":
                    content = html.Div(_results_panel(full), className="workbench-page")
                else:
                    content = html.Div(
                        [
                            _workbench_data_setup(),
                            html.Div(render_setup(full), className="experiment-setup-card"),
                        ],
                        className="workbench-page",
                    )
            except Exception:
                content = html.Div(
                    [
                        html.P(
                            "Workbench error:", style={"color": "var(--rd)", "fontWeight": "bold"}
                        ),
                        html.Pre(
                            traceback.format_exc(),
                            style={"fontSize": ".72rem", "maxHeight": "300px", "overflowY": "auto"},
                        ),
                    ],
                    className="err-b",
                )
            return hero, content
        return hero, _gov_page(page or "Overview", full)

    # ── Governance page-specific callbacks ───────────────────────────────────

    @app.callback(
        Output("cc-rpte-box", "children"),
        Input("cc-downstream", "value"),
        prevent_initial_call=True,
    )
    def _cc_rpte(v):
        from hugiml.dashboard.dash_components.pages.config_compare import rpte_controls

        return rpte_controls(v == "rpte")

    @app.callback(
        Output("cc-result-box", "children"),
        Input("cc-run", "n_clicks"),
        State("cc-L", "value"),
        State("cc-topK", "value"),
        State("cc-G", "value"),
        State("cc-fm", "value"),
        State("cc-strict", "value"),
        State("cc-downstream", "value"),
        State("st-ctx", "data"),
        prevent_initial_call=True,
    )
    def _cc_run(nc, L, topK, G, fm, strict, ds, ctx_ref):
        if not nc or ctx_ref is None:
            return no_update
        full = _CTX.get(ctx_ref.get("ck", ""))
        if full is None:
            return html.Div("No model.", className="warn-b")
        be = None
        if ds == "rpte" and _RPTE:
            lc = "3xD" if "3xD" in RPTE_LC else next(iter(RPTE_LC), None)
            be = LeafWiseBoundedLookaheadRPTEFeatureLR(
                leaf_config=lc, depth=4, n_estimators=10, enable_lookahead="adaptive"
            )
        try:
            top_k = int(str(topK).strip())
            if top_k < 1:
                raise ValueError
        except (TypeError, ValueError):
            return html.Div("topK must be a positive whole number.", className="warn-b")
        params = {
            "adaptive_binning": True,
            "B": -1,
            "L": int(L or 2),
            "topK": top_k,
            "G": float(G or 0.01),
            "feature_mode": fm or "original_plus_patterns",
            "topk_budget_strict": bool(strict),
            "base_estimator": be,
        }
        result = fit_hugiml_config(
            full["X"],
            full["y"],
            params=params,
            cv=int(full.get("cv", 3)),
            scoring="roc_auc",
            random_state=int(full.get("random_state", 2026)),
        )
        _AUX["cc_result"] = result
        from hugiml.dashboard.dash_components.pages.config_compare import render_result

        return render_result(full, result)

    @app.callback(
        Output("pr-fam-tbl", "children"),
        Output("pr-items", "options"),
        Output("pr-items", "value"),
        Input("pr-family", "value"),
        State("st-ctx", "data"),
    )
    def _pr_fam(family, ctx_ref):
        if ctx_ref is None:
            return html.Div(), [], []
        full = _CTX.get(ctx_ref.get("ck", ""))
        if full is None:
            return html.Div(), [], []
        from hugiml.dashboard.dash_components.pages.pruning import family_table

        table, options = family_table(
            full.get("model"), full.get("X"), family or "Original features"
        )
        return table, options, []

    @app.callback(
        Output("pr-result-box", "children"),
        Input("pr-run", "n_clicks"),
        State("pr-family", "value"),
        State("pr-items", "value"),
        State("st-ctx", "data"),
        prevent_initial_call=True,
    )
    def _pr_run(nc, fam, items, ctx_ref):
        if not nc or not items or ctx_ref is None:
            return no_update
        full = _CTX.get(ctx_ref.get("ck", ""))
        if full is None:
            return html.Div("No model.", className="warn-b")
        m, X, y = full["model"], full["X"], full["y"]
        cv, rs = int(full.get("cv", 3)), int(full.get("random_state", 2026))
        try:
            if fam == "Original features":
                r, _ = fit_feature_pruned_hugiml(
                    X, y, base_model=m, remove_features=items, cv=cv, random_state=rs
                )
            else:
                r, _ = fit_representation_pruned_downstream(
                    m, X, y, remove_columns=items, family=fam, cv=cv, random_state=rs
                )
            _AUX.update(pr_result=r, pr_family=fam, pr_removed=items)
        except Exception as e:
            return html.Div(
                [
                    html.P("Pruning failed:", style={"color": "var(--rd)"}),
                    html.Pre(str(e), style={"fontSize": ".74rem"}),
                ],
                className="err-b",
            )
        from hugiml.dashboard.dash_components.pages.pruning import render_result as pr_render

        return pr_render(full, _AUX["pr_result"], fam, items)

    @app.callback(
        Output("rr-result-box", "children"),
        Input("rr-run", "n_clicks"),
        State("rr-excl", "value"),
        State("st-ctx", "data"),
        prevent_initial_call=True,
    )
    def _rr_run(nc, sel, ctx_ref):
        if not nc or not sel or ctx_ref is None:
            return no_update
        full = _CTX.get(ctx_ref.get("ck", ""))
        if full is None:
            return html.Div("No model.", className="warn-b")
        try:
            r, Xp = fit_feature_pruned_hugiml(
                full["X"],
                full["y"],
                base_model=full["model"],
                remove_features=sel,
                cv=int(full.get("cv", 3)),
                random_state=int(full.get("random_state", 2026)),
            )
            _AUX.update(rr_result=r, rr_removed=sel, rr_X=Xp)
        except Exception as e:
            return html.Div(str(e), className="err-b")
        from hugiml.dashboard.dash_components.pages.pruning import render_rpte_result

        return render_rpte_result(full, r, sel, Xp)

    @app.callback(
        Output("repr-profile-out", "children"),
        Input("repr-profile-dd", "value"),
        State("st-ctx", "data"),
        prevent_initial_call=True,
    )
    def _profile(feat, ctx_ref):
        if not feat or ctx_ref is None:
            return html.Div()
        full = _CTX.get(ctx_ref.get("ck", ""))
        if full is None:
            return html.Div()
        from hugiml.dashboard.dash_components.pages.representation import profile_chart

        return profile_chart(full.get("model"), full.get("X"), feat)

    @app.callback(
        Output("case-content", "children"),
        Input("case-idx", "value"),
        State("st-ctx", "data"),
        prevent_initial_call=True,
    )
    def _case(idx, ctx_ref):
        if ctx_ref is None:
            return html.Div()
        full = _CTX.get(ctx_ref.get("ck", ""))
        if full is None:
            return html.Div()
        from hugiml.dashboard.dash_components.pages.prediction import render_case

        return render_case(full, int(idx or 0))


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="HUGIML Governance Studio", add_help=False)
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--no-open", action="store_true", default=False)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--random-state", type=int, default=2026)
    args, _ = p.parse_known_args()
    app = create_app(args.debug)
    from hugiml.dashboard.launcher import _browser_url, _open_browser_when_ready

    url = _browser_url(args.host, args.port)
    print(f"\n  HUGIML Governance Studio (Dash)\n  {url}\n  Ctrl+C to quit.\n")
    if not args.no_open:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(args.host, args.port, url),
            daemon=True,
        ).start()
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()

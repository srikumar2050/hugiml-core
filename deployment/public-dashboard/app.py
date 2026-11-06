"""Public demonstration WSGI entry point."""

from __future__ import annotations

import atexit
import os
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from flask import abort, request, session
from guidance import CONTROL_HELP, help_label
from service import DATASETS, MODES, PAGES, Coordinator

PAGE_COPY = {
    "Overview": (
        "THE BIG PICTURE",
        "A clear view of your model.",
        "Review the model, its evidence, and the data behind each decision.",
    ),
    "Validation": (
        "MODEL PERFORMANCE",
        "Measure what matters.",
        "Explore cross-validation scores, calibration, and decision thresholds.",
    ),
    "Representation Audit": (
        "INSIDE THE MODEL",
        "See what the model learned.",
        "Inspect the features, patterns, and transformations in the representation.",
    ),
    "Pattern Inventory": (
        "LEARNED PATTERNS",
        "Find the rules in the data.",
        "Review interpretable patterns and how much of the population they cover.",
    ),
    "Case Review": (
        "INDIVIDUAL DECISIONS",
        "Follow a prediction.",
        "Choose an example and trace its inputs and model evidence.",
    ),
    "Data Quality & Policy": (
        "DATA & RESPONSIBILITY",
        "Understand your data.",
        "Review missing values, sensitive features, and representation quality.",
    ),
    "Pruning": (
        "CONTROLLED MODEL EDITING",
        "Remove. Refit. Compare.",
        "Explore the effect of removing inputs or patterns while preserving your original model.",
    ),
    "Monitoring": (
        "MODEL OBSERVABILITY",
        "Keep the evidence in view.",
        "Inspect prediction distributions and available monitoring diagnostics.",
    ),
}


def status_view(status, label):
    captions = {
        "ready": "Ready to explore",
        "queued": "Waiting for a training slot",
        "running": "Training your model",
        "complete": "Model ready",
        "pruning queued": "Pruning comparison queued",
        "pruning": "Building your pruning comparison",
        "compared": "Pruning comparison ready · original model preserved",
        "pruning failed": "Pruning could not complete. Your original model is available.",
        "pruning timed out": "Pruning reached the time limit. Your original model is available.",
        "failed": "Training could not complete. Please try again.",
        "timed out": "Training reached the time limit. Try a smaller pattern budget.",
    }
    return html.Div(
        [
            html.Span(className="status-dot"),
            html.Strong(captions[status]),
            html.Span(label, className="run-context"),
        ],
        className=f"status-message status-{status.replace(' ', '-')}",
    )


def content_view(page, body, status):
    eyebrow, title, description = PAGE_COPY[page]
    header = html.Div(
        [html.Div(eyebrow, className="eyebrow"), html.H1(title), html.P(description)],
        className="page-heading",
    )
    if body is None:
        busy = status in ("queued", "running")
        body = html.Div(
            [
                html.Div("◈", className="empty-symbol", **{"aria-hidden": "true"}),
                html.H3(
                    "Your model is taking shape." if busy else "Your next insight starts here."
                ),
                html.P(
                    "Training and validation run in the background. Your evidence will appear here."
                    if busy
                    else "Choose a dataset, set a pattern budget, and train your first model."
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("01"),
                                html.Strong("Train"),
                                html.P("Start with a curated dataset."),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span("02"),
                                html.Strong("Explore"),
                                html.P("Inspect patterns and performance."),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span("03"),
                                html.Strong("Understand"),
                                html.P("Connect predictions to evidence."),
                            ]
                        ),
                    ],
                    className="welcome-steps",
                ),
            ],
            className="empty-state",
        )
    return html.Div([header, html.Div(body, className="evidence-body")], className="page-content")


def create_app(*, coordinator=None, secret=None, testing=False):
    secret = secret or os.environ.get("HUGIML_DEMO_SECRET")
    if not secret or len(secret) < 32:
        raise ValueError("HUGIML_DEMO_SECRET must contain at least 32 random characters.")
    jobs = coordinator or Coordinator()
    if coordinator is None:
        atexit.register(jobs.close)
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        assets_folder=str(Path(__file__).with_name("assets")),
        title="HUGIML · Governance Studio",
    )
    server = app.server
    server.config.update(
        SECRET_KEY=secret,
        TESTING=testing,
        MAX_CONTENT_LENGTH=16 * 1024,
        SESSION_COOKIE_NAME="hugiml_demo",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SECURE=not (testing or os.environ.get("HUGIML_DEMO_LOCAL_HTTP") == "1"),
        SESSION_COOKIE_SAMESITE="Strict",
        PERMANENT_SESSION_LIFETIME=timedelta(minutes=15),
    )

    @server.before_request
    def protect():
        if request.path == "/healthz":
            return None
        if request.method == "POST":
            origin = urlsplit(request.headers.get("Origin", ""))
            if origin.netloc != request.host or origin.scheme not in ("http", "https"):
                abort(403)
        if request.path == "/" or request.path == "/_dash-update-component":
            try:
                payload = request.get_json(silent=True) if request.method == "POST" else None
                changes = payload.get("changedPropIds", []) if isinstance(payload, dict) else []
                polling = bool(changes) and all(
                    change in ("demo-poll.n_intervals", "demo-rendered.data") for change in changes
                )
                session["owner"] = jobs.session(session.get("owner"), touch=not polling)
                session.permanent = True
            except ValueError:
                return "The demo is busy. Please return in a few minutes.", 503
        return None

    @server.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        if not request.path.startswith(("/assets/", "/_dash-component-suites/")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @server.get("/healthz")
    def health():
        return {"status": "ok"}

    app.layout = html.Div(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.Span("H", className="brand-symbol"),
                            html.Div(
                                [
                                    html.Strong("HUGIML"),
                                    html.Span("GOVERNANCE STUDIO", className="brand-subtitle"),
                                ]
                            ),
                        ],
                        className="brand",
                    ),
                    html.Div(
                        [html.Span(className="live-dot"), "Interactive demo"],
                        className="live-label",
                    ),
                ],
                className="studio-header",
            ),
            html.Div(
                [
                    html.Aside(
                        [
                            html.Div("YOUR EXPERIMENT", className="eyebrow"),
                            html.H2("Train a model."),
                            html.P(
                                "Start with a curated synthetic dataset.",
                                className="sidebar-intro",
                            ),
                            html.Label(
                                help_label(
                                    "Dataset", CONTROL_HELP["Dataset"], "control-help-dataset"
                                ),
                                htmlFor="demo-dataset",
                                className="field-label",
                            ),
                            dcc.Dropdown(
                                id="demo-dataset",
                                options=DATASETS,
                                value="credit",
                                clearable=False,
                                searchable=False,
                            ),
                            html.Div(
                                "Three synthetic datasets, ready to explore.",
                                className="field-note",
                            ),
                            html.Label(
                                help_label(
                                    "Pattern budget",
                                    CONTROL_HELP["Pattern budget"],
                                    "control-help-topk",
                                ),
                                className="field-label control-gap",
                            ),
                            dcc.RadioItems(
                                id="demo-topk",
                                options=[{"label": str(n), "value": n} for n in (20, 50, 100)],
                                value=20,
                                className="segmented",
                            ),
                            html.Div(
                                "topK · More patterns allow a richer representation.",
                                className="field-note",
                            ),
                            html.Label(
                                help_label(
                                    "Pattern order",
                                    CONTROL_HELP["Pattern order"],
                                    "control-help-order",
                                ),
                                className="field-label control-gap",
                            ),
                            dcc.RadioItems(
                                id="demo-order",
                                options=[
                                    {"label": "1 · Individual", "value": 1},
                                    {"label": "2 · Pairs", "value": 2},
                                ],
                                value=2,
                                className="segmented",
                            ),
                            html.Div(
                                "L · Include single features or pair interactions.",
                                className="field-note",
                            ),
                            html.Details(
                                [
                                    html.Summary("Advanced"),
                                    html.Label(
                                        help_label(
                                            "Estimator selection",
                                            CONTROL_HELP["Estimator selection"],
                                            "control-help-mode",
                                        ),
                                        htmlFor="demo-mode",
                                        className="field-label",
                                    ),
                                    dcc.Dropdown(
                                        id="demo-mode",
                                        options=MODES,
                                        value="auto",
                                        clearable=False,
                                        searchable=False,
                                    ),
                                    html.P(
                                        "Automatic compares LR and RPTE using three-fold validation. "
                                        "Choose an estimator here to override it.",
                                        className="field-note",
                                    ),
                                ],
                                className="advanced-panel",
                            ),
                            html.Button(
                                ["Train model", html.Span("↗", **{"aria-hidden": "true"})],
                                id="demo-train",
                                n_clicks=0,
                                className="train-button",
                                title="Train with the selected dataset, pattern budget, and order. Only one job runs at a time; results replace your previous model.",
                            ),
                            html.Div(
                                [
                                    help_label(
                                        "3-fold validation",
                                        CONTROL_HELP["3-fold validation"],
                                        "control-help-cv",
                                    ),
                                    html.Span("Synthetic data"),
                                ],
                                className="assurance-tags",
                            ),
                            html.Div(
                                [
                                    html.Div("EXPLORE THE EVIDENCE", className="eyebrow"),
                                    dcc.RadioItems(
                                        id="demo-page",
                                        options=[
                                            {
                                                "label": html.Span(
                                                    [
                                                        html.Span(
                                                            f"{index:02}", className="nav-number"
                                                        ),
                                                        page,
                                                    ],
                                                    title=PAGE_COPY[page][2],
                                                ),
                                                "value": page,
                                            }
                                            for index, page in enumerate(PAGES, 1)
                                        ],
                                        value=PAGES[0],
                                        className="page-nav",
                                    ),
                                ],
                                className="navigation-block",
                            ),
                            html.P(
                                "Private session · Results expire after 15 minutes of inactivity.",
                                className="session-note",
                            ),
                        ],
                        className="studio-sidebar",
                    ),
                    html.Main(
                        [
                            html.Div(
                                id="demo-status",
                                role="status",
                                className="run-status",
                                **{"aria-live": "polite"},
                            ),
                            html.Div(id="demo-content"),
                            html.Div(
                                [
                                    html.Label("Row index", htmlFor="demo-case-index"),
                                    html.Div(
                                        [
                                            html.Button(
                                                "−",
                                                id="demo-case-prev",
                                                n_clicks=0,
                                                title="Previous case",
                                            ),
                                            dcc.Input(
                                                id="demo-case-index",
                                                type="number",
                                                value=0,
                                                min=0,
                                                max=0,
                                                step=1,
                                                debounce=True,
                                            ),
                                            html.Button(
                                                "+",
                                                id="demo-case-next",
                                                n_clicks=0,
                                                title="Next case",
                                            ),
                                            html.Span(id="demo-case-count"),
                                        ],
                                        className="case-navigation",
                                    ),
                                    html.Div(id="demo-case-body"),
                                    dcc.Store(id="demo-case-revision"),
                                ],
                                id="demo-case-controls",
                                className="evidence-body",
                                style={"display": "none"},
                            ),
                            html.Div(
                                [
                                    html.H3("Choose what to remove", className="prune-title"),
                                    dcc.RadioItems(
                                        id="demo-prune-kind",
                                        options=[
                                            {
                                                "label": "Raw inputs · full rebuild",
                                                "value": "features",
                                            },
                                            {
                                                "label": "Mined patterns · downstream refit",
                                                "value": "patterns",
                                            },
                                        ],
                                        value="features",
                                        className="segmented",
                                    ),
                                    html.Label(
                                        "Items to remove",
                                        htmlFor="demo-prune-items",
                                        className="field-label mt-3",
                                    ),
                                    dcc.Dropdown(
                                        id="demo-prune-items",
                                        options=[],
                                        value=[],
                                        multi=True,
                                        placeholder="Select items from your fitted model",
                                    ),
                                    html.P(
                                        "Select up to 3 raw inputs or 5 patterns. Each comparison starts from the original model. "
                                        "Allow 30 seconds between training or pruning requests.",
                                        className="field-note",
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Compare pruned model",
                                                id="demo-prune-run",
                                                n_clicks=0,
                                                disabled=True,
                                                className="train-button",
                                            ),
                                            html.Button(
                                                "Reset comparison",
                                                id="demo-prune-reset",
                                                n_clicks=0,
                                                disabled=True,
                                                className="reset-button",
                                            ),
                                        ],
                                        className="prune-actions",
                                    ),
                                    html.Div(
                                        id="demo-prune-notice",
                                        role="status",
                                        className="field-note",
                                    ),
                                    dcc.Store(id="demo-model-revision"),
                                ],
                                id="demo-prune-controls",
                                className="prune-controls",
                                style={"display": "none"},
                            ),
                            html.Footer(
                                "Interactive charts, evidence tables, and case navigation. "
                                "Pruning comparisons preserve the original model shown in the other governance pages.",
                                className="studio-footer",
                            ),
                        ],
                        className="studio-main",
                    ),
                ],
                className="studio-layout",
            ),
            dcc.Store(id="demo-rendered"),
            dcc.Interval(id="demo-poll", interval=2000),
        ],
        className="studio",
    )

    @app.callback(
        Output("demo-status", "children"),
        Output("demo-content", "children"),
        Output("demo-rendered", "data"),
        Input("demo-train", "n_clicks"),
        Input("demo-poll", "n_intervals"),
        Input("demo-page", "value"),
        State("demo-dataset", "value"),
        State("demo-mode", "value"),
        State("demo-rendered", "data"),
        State("demo-topk", "value"),
        State("demo-order", "value"),
    )
    def update(clicks, ticks, page, dataset, mode, rendered, topk, order):
        owner = session.get("owner")
        try:
            if ctx.triggered_id == "demo-train":
                jobs.submit(owner, dataset, mode, topk, order)
            status, label, revision, body = jobs.snapshot(owner, page)
            marker = [revision, status, page]
            return (
                status_view(status, label),
                no_update if marker == rendered else content_view(page, body, status),
                no_update if marker == rendered else marker,
            )
        except ValueError as exc:
            return html.Div(str(exc), className="status-message status-error"), no_update, no_update

    @app.callback(
        Output("demo-prune-items", "options"),
        Output("demo-prune-items", "value"),
        Output("demo-prune-notice", "children"),
        Output("demo-model-revision", "data"),
        Output("demo-prune-run", "disabled"),
        Output("demo-prune-reset", "disabled"),
        Output("demo-prune-controls", "style"),
        Input("demo-poll", "n_intervals"),
        Input("demo-prune-kind", "value"),
        Input("demo-prune-run", "n_clicks"),
        Input("demo-prune-reset", "n_clicks"),
        Input("demo-page", "value"),
        Input("demo-prune-items", "value"),
        State("demo-model-revision", "data"),
    )
    def pruning_controls(ticks, kind, run_clicks, reset_clicks, page, items, client_revision):
        style = {} if page == "Pruning" else {"display": "none"}
        owner = session.get("owner")
        try:
            revision, workflow = jobs.pruning_info(owner)
            if workflow is None:
                return (
                    [],
                    [],
                    "Train a model to inspect its editable inputs and patterns.",
                    None,
                    True,
                    True,
                    style,
                )
            if kind not in ("features", "patterns"):
                raise ValueError("Choose raw inputs or mined patterns.")
            options = (
                [{"label": name, "value": name} for name in workflow["features"]]
                if kind == "features"
                else workflow["patterns"]
            )
            if revision != client_revision or ctx.triggered_id == "demo-prune-kind":
                return (
                    options,
                    [],
                    "Select items, then compare their effect on the reserved holdout.",
                    revision,
                    True,
                    False,
                    style,
                )
            notice, value = no_update, no_update
            if ctx.triggered_id == "demo-prune-run":
                jobs.submit_pruning(owner, client_revision, kind, items)
                notice = "Comparison requested. The status above will update when it is ready."
            elif ctx.triggered_id == "demo-prune-reset":
                jobs.reset_pruning(owner, client_revision)
                notice, value = "Comparison cleared. Your original model is unchanged.", []
            status = jobs.snapshot(owner, "Pruning")[0]
            if ctx.triggered_id == "demo-poll":
                notice = {
                    "compared": "Comparison ready. The original model is preserved.",
                    "pruning failed": "The comparison could not complete. Your original model is available.",
                    "pruning timed out": "The comparison reached its time limit. Your original model is available.",
                }.get(status, no_update)
            busy = status in ("queued", "running", "pruning queued", "pruning")
            from pruning_workflow import validate_selection

            try:
                validate_selection(workflow, kind, [] if value == [] else items)
                valid = True
            except ValueError:
                valid = False
            return no_update, value, notice, revision, busy or not valid, busy, style
        except ValueError as exc:
            return no_update, no_update, str(exc), no_update, True, True, style

    @app.callback(
        Output("demo-case-body", "children"),
        Output("demo-case-index", "value"),
        Output("demo-case-index", "max"),
        Output("demo-case-index", "disabled"),
        Output("demo-case-prev", "disabled"),
        Output("demo-case-next", "disabled"),
        Output("demo-case-count", "children"),
        Output("demo-case-revision", "data"),
        Output("demo-case-controls", "style"),
        Input("demo-case-index", "value"),
        Input("demo-case-prev", "n_clicks"),
        Input("demo-case-next", "n_clicks"),
        Input("demo-page", "value"),
        Input("demo-rendered", "data"),
        State("demo-case-revision", "data"),
    )
    def review_case(index, previous, following, page, rendered, revision):
        if page != "Case Review":
            return (no_update,) * 8 + ({"display": "none"},)
        try:
            current, count, index, body = jobs.case(session.get("owner"), index, revision)
            if (
                current == revision
                and count
                and ctx.triggered_id in ("demo-case-prev", "demo-case-next")
            ):
                index = max(
                    0, min(count - 1, index + (1 if ctx.triggered_id == "demo-case-next" else -1))
                )
                current, count, index, body = jobs.case(session.get("owner"), index, current)
            return (
                body,
                index,
                max(0, count - 1),
                not count,
                not count or index == 0,
                not count or index == count - 1,
                f"{count} sampled training cases",
                current,
                {} if count else {"display": "none"},
            )
        except ValueError as exc:
            return (
                html.P(str(exc)),
                no_update,
                no_update,
                False,
                True,
                True,
                no_update,
                no_update,
                {},
            )

    return app


def application():
    """Gunicorn creates one application in its single worker."""
    return create_app().server

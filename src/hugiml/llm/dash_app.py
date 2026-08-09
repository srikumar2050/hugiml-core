"""Dash interface for the HUGIML natural-language workbench."""

from __future__ import annotations

import base64
import tempfile
import threading
from pathlib import Path
from typing import Any

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, ctx, dash_table, dcc, html, no_update

from hugiml.dashboard.dash_components.styles import THEMES, build_css
from hugiml.llm.orchestrator import HUGIMLActionOrchestrator
from hugiml.llm.runtime import (
    DEFAULT_OLLAMA_MODEL,
    FALLBACK_OLLAMA_MODEL,
    LIGHT_MODE_MODEL,
    check_ollama,
    get_memory_info,
    get_model_catalog,
    is_below_minimum_llm_model,
    is_lightweight_supported_model,
    model_availability,
    recommend_profile,
)
from hugiml.llm.ui_service import parse_launch_context, run_prompt

_CTX: dict[str, Any] = {}
_ACTION_LOCK = threading.Lock()
_QUICK_ACTIONS = {
    "llm-quick-describe": "Describe the active dataset",
    "llm-quick-build": "Build a HUGIML model on this dataset",
    "llm-quick-tune": "Tune this dataset for ROC AUC",
    "llm-quick-explain": "Explain the strongest patterns in plain English",
    "llm-quick-predict": "Generate a prediction table for 10 held-out rows",
    "llm-quick-governance": "Generate a governance report for the active model",
}


def _repo_root() -> Path | None:
    candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for base in candidates:
        if (base / "pyproject.toml").exists() and (base / "src" / "hugiml").exists():
            return base
    return None


def _dataset_rows(include_benchmarks: bool = False) -> list[dict[str, Any]]:
    orch: HUGIMLActionOrchestrator = _CTX["orchestrator"]
    return [
        item.to_dict()
        for item in orch.registry.list_datasets(
            include_profiles=False,
            include_benchmarks=include_benchmarks,
        )
    ]


def _dataset_options(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "label": f"{row['name']}  ·  {row.get('source', 'dataset')}",
            "value": str(row["name"]),
        }
        for row in rows
    ]


def _benchmark_catalog_enabled(selection: list[str] | None) -> bool:
    """Return whether the optional benchmark catalog is selected."""

    return "enabled" in (selection or [])


_DETERMINISTIC_MODEL = "deterministic router only"
_PROFILE_LABELS = {
    "no-llm": "Deterministic workspace",
    "light-mode": "Light mode",
    "lightweight-llm": "Default local LLM",
    "fallback-llm": "Fallback local LLM",
    "minimum-llm": "Minimum local LLM",
    "balanced": "Balanced local LLM",
    "expanded": "Expanded local LLM",
    "large-context": "Large-context local LLM",
}


def _preferred_model(selectable: list[str], recommended: str) -> str:
    for candidate in (DEFAULT_OLLAMA_MODEL, recommended, LIGHT_MODE_MODEL, FALLBACK_OLLAMA_MODEL):
        if candidate and candidate in selectable:
            return candidate
    return _DETERMINISTIC_MODEL


def _model_picker_state(
    memory: Any,
    profile: Any,
    status: dict[str, Any],
    *,
    current: str | None = None,
) -> tuple[list[dict[str, str]], str, list[dict[str, Any]]]:
    """Build the selectable models and full local-model catalog for Dash."""

    installed = {str(name) for name in status.get("models") or [] if str(name).strip()}
    catalog = get_model_catalog(_CTX.get("repo_root"))
    catalog_by_model = {option.model: option for option in catalog}
    recommended = str(getattr(profile, "recommended_model", "") or "")
    selectable = [_DETERMINISTIC_MODEL]
    labels = {_DETERMINISTIC_MODEL: "Deterministic routing only · no Ollama call"}
    rows: list[dict[str, Any]] = []

    for option in catalog:
        availability = model_availability(
            option,
            memory,
            installed,
            ollama_ok=bool(status.get("ok")),
        )
        if availability["selectable"]:
            selectable.append(option.model)
            suffix = " · recommended" if option.model == recommended else ""
            labels[option.model] = f"{option.label} · {option.model}{suffix}"
        rows.append(
            {
                "model": option.model,
                "profile": _PROFILE_LABELS.get(option.profile, option.profile),
                "minimum free RAM": f"{option.min_available_gb:.1f} GB",
                "installed": "yes" if availability["installed"] else "no",
                "status": "selectable" if availability["selectable"] else availability["reason"],
            }
        )

    free_gb = getattr(memory, "available_gb", None)
    for model_name in sorted(installed):
        if model_name in catalog_by_model:
            continue
        lightweight = is_lightweight_supported_model(model_name)
        minimum_gb = 5.0 if lightweight else 6.0
        too_small = is_below_minimum_llm_model(model_name)
        can_select = (
            bool(status.get("ok")) and not too_small and (free_gb is None or free_gb >= minimum_gb)
        )
        if can_select:
            selectable.append(model_name)
            labels[model_name] = f"Installed Ollama model · {model_name} · manual"
        reason = "selectable"
        if too_small:
            reason = "unsupported sub-3B manual model"
        elif not can_select:
            reason = f"manual model needs ≥{minimum_gb:.1f} GB free RAM"
        rows.append(
            {
                "model": model_name,
                "profile": "manual",
                "minimum free RAM": f"{minimum_gb:.1f} GB",
                "installed": "yes",
                "status": reason,
            }
        )

    selected = str(current or "").strip()
    if selected not in selectable:
        keep_loaded = (
            selected != _DETERMINISTIC_MODEL
            and bool(status.get("ok"))
            and selected in installed
            and not is_below_minimum_llm_model(selected)
        )
        if keep_loaded:
            selectable.append(selected)
            labels[selected] = f"Current session model · {selected} · keeping selection"
            for row in rows:
                if row["model"] == selected:
                    row["status"] = "kept for this session while Ollama holds memory"
                    break
        else:
            selected = _preferred_model(selectable, recommended)

    options = [{"label": labels[value], "value": value} for value in selectable]
    return options, selected, rows


def _model_catalog_view(rows: list[dict[str, Any]], status: dict[str, Any]) -> list[Any]:
    setup: list[Any] = []
    if not status.get("ok"):
        setup.extend(
            [
                html.P("Start Ollama, then refresh local models.", className="model-help"),
                html.Code("ollama serve", className="model-command"),
            ]
        )
    missing = [str(row["model"]) for row in rows if row.get("installed") == "no"]
    if missing:
        setup.extend(
            [
                html.P(
                    "Install a configured model, then refresh local models.", className="model-help"
                ),
                html.Pre(
                    "\n".join(f"ollama pull {name}" for name in missing[:5]),
                    className="model-command",
                ),
            ]
        )
    return [
        html.P(
            "Only installed models that satisfy the available-memory policy can be selected. "
            "The current selection is retained while an active Ollama model holds memory.",
            className="model-help",
        ),
        _table(rows, page_size=8),
        *setup,
    ]


def _model_options() -> tuple[list[dict[str, str]], str, dict[str, Any], list[dict[str, Any]]]:
    status = check_ollama(timeout=0.8)
    memory = get_memory_info()
    profile = recommend_profile(memory, repo_root=_CTX.get("repo_root"))
    options, selected, rows = _model_picker_state(memory, profile, status)
    return options, selected, status, rows


def _metric_card(label: str, value: Any) -> html.Div:
    if isinstance(value, float):
        display = f"{value:.4f}"
    elif value is None:
        display = "—"
    else:
        display = str(value)
    return html.Div(
        [html.Div(label, className="mc-l"), html.Div(display, className="mc-v")], className="mc"
    )


def _table(rows: Any, *, page_size: int = 8) -> Any:
    if not isinstance(rows, list) or not rows:
        return html.Div("No rows available.", className="empty-note")
    frame = pd.DataFrame(rows)
    if frame.empty:
        return html.Div("No rows available.", className="empty-note")
    frame = frame.iloc[:, :14].copy()
    for column in frame.columns:
        frame[column] = frame[column].map(_display_value)
    return dash_table.DataTable(
        data=frame.to_dict("records"),
        columns=[{"name": str(column), "id": str(column)} for column in frame.columns],
        page_size=min(page_size, len(frame)),
        sort_action="native",
        filter_action="native" if len(frame) > 8 else "none",
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "Inter, sans-serif",
            "fontSize": "12px",
            "padding": "8px 10px",
            "textAlign": "left",
            "maxWidth": "280px",
            "whiteSpace": "normal",
        },
        style_header={"fontWeight": 800, "backgroundColor": "var(--cb)"},
    )


def _display_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (str, int, bool)):
        return value
    text = str(value)
    return text if len(text) <= 240 else text[:237] + "..."


def _result_view(result: dict[str, Any]) -> list[Any]:
    blocks: list[Any] = []
    message = str(result.get("message") or "").strip()
    if message:
        blocks.append(dcc.Markdown(message, className="answer-copy"))
    for name, rows in (result.get("tables") or {}).items():
        blocks.extend([html.H5(str(name).replace("_", " ").title()), _table(rows)])
    artifacts = result.get("artifacts") or {}
    if artifacts:
        blocks.append(html.Details([html.Summary("Artifacts"), html.Pre(str(artifacts))]))
    if not blocks:
        blocks.append(
            html.Div("The action completed without display data.", className="empty-note")
        )
    return blocks


def _history_view(history: list[dict[str, Any]] | None) -> list[Any]:
    if not history:
        return [
            html.Div(
                [
                    html.Div("ASSISTANT", className="qa-label"),
                    html.Div(
                        "Select a dataset, then ask HUGIML to build, tune, score, explain, prune, or create governance evidence.",
                        className="qa-text",
                    ),
                ],
                className="qa-empty",
            )
        ]
    blocks: list[Any] = []
    for index, item in enumerate(history, start=1):
        blocks.append(
            html.Div(
                [
                    html.Div(f"QUESTION {index}", className="qa-label"),
                    html.Div(str(item.get("prompt") or ""), className="qa-text"),
                ],
                className="qa-card question",
            )
        )
        state = "answer" if item.get("ok") else "answer refusal"
        blocks.append(
            html.Div(
                [
                    html.Div(f"ANSWER {index}", className="qa-label"),
                    *_result_view(item.get("result") or {}),
                ],
                className=f"qa-card {state}",
            )
        )
    return blocks


def _active_session() -> Any | None:
    orch: HUGIMLActionOrchestrator = _CTX["orchestrator"]
    return orch.sessions.get(orch.last_session_id) if orch.last_session_id else None


def _evidence_view() -> list[Any]:
    session = _active_session()
    if session is None:
        return [
            html.Div("Build or tune a model to inspect its evidence.", className="empty-results")
        ]
    metrics = dict(session.metrics or {})
    preferred = ["roc_auc", "balanced_accuracy", "f1", "accuracy", "brier", "log_loss"]
    cards = [_metric_card(name.replace("_", " "), metrics.get(name)) for name in preferred]
    return [
        html.Div(
            [
                html.Span("ACTIVE MODEL", className="hero-ey"),
                html.H3(session.dataset),
                html.P(
                    f"Target: {session.target} · Session: {session.session_id}", className="hero-p"
                ),
            ],
            className="section-card",
        ),
        html.Div(cards, className="llm-metric-grid"),
        html.Div(
            [
                html.H4("Model artifacts"),
                html.Pre(str(session.artifacts or "No artifacts generated yet.")),
            ],
            className="section-card",
        ),
    ]


def _shell(
    rows: list[dict[str, Any]],
    model_options: list[dict[str, str]],
    model_value: str,
    status: dict[str, Any],
    model_rows: list[dict[str, Any]],
) -> html.Div:
    mem = get_memory_info()
    profile = recommend_profile(mem, repo_root=_CTX.get("repo_root"))
    available_gb = getattr(mem, "available_gb", None)
    memory_label = (
        f"{available_gb:.1f} GB free"
        if isinstance(available_gb, (int, float))
        else "Memory profile ready"
    )
    ollama_label = "Ollama connected" if status.get("ok") else "Deterministic mode ready"
    default_dataset = rows[0]["name"] if rows else None
    return html.Div(
        [
            dcc.Location(id="llm-url", refresh=False),
            dcc.Store(id="llm-chat-history", data=[]),
            dcc.Store(id="llm-session-version", data=0),
            dcc.Store(id="llm-upload-path"),
            dcc.Store(id="llm-launch-context", data={}),
            html.Header(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("HUGIML · INTERPRETABLE AI", className="hero-ey"),
                                    html.H1("LLM Assistant", className="hero-h1"),
                                    html.P(
                                        "Grounded natural-language modeling, evidence inspection, and governance actions.",
                                        className="hero-p",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        ollama_label, id="llm-ollama-status", className="chip"
                                    ),
                                    html.Span(memory_label, className="chip"),
                                    html.Span(
                                        str(getattr(profile, "name", "workspace")), className="chip"
                                    ),
                                ],
                                className="chip-row hero-status",
                            ),
                        ],
                        className="hero-topline",
                    )
                ],
                className="hug-hero llm-hero",
            ),
            html.Div(
                [
                    html.Span("Workspace", className="setup-lbl"),
                    dcc.Dropdown(
                        id="llm-dataset",
                        options=_dataset_options(rows),
                        value=default_dataset,
                        clearable=False,
                        className="llm-dataset-select",
                    ),
                    dcc.Dropdown(
                        id="llm-model",
                        options=model_options,
                        value=model_value,
                        clearable=False,
                        persistence=True,
                        persistence_type="session",
                        className="llm-model-select",
                    ),
                    dbc.Button(
                        "Refresh local models",
                        id="llm-refresh-models",
                        outline=True,
                        color="secondary",
                        size="sm",
                    ),
                    dcc.RadioItems(
                        id="llm-response-mode",
                        options=["Fast", "Thinking"],
                        value="Fast",
                        inline=True,
                        className="llm-mode",
                    ),
                    dcc.Checklist(
                        id="llm-include-benchmarks",
                        options=[{"label": "Benchmark catalog", "value": "enabled"}],
                        value=[],
                        className="llm-catalog-toggle",
                    ),
                    html.Div(className="hug-spacer"),
                    dcc.Dropdown(
                        id="llm-theme",
                        options=THEMES,
                        value="Ocean",
                        clearable=False,
                        className="theme-select",
                    ),
                ],
                className="hug-ctrl llm-toolbar",
            ),
            html.Details(
                [
                    html.Summary("Local model catalog and setup"),
                    html.Div(
                        _model_catalog_view(model_rows, status),
                        id="llm-model-catalog",
                        className="model-catalog-body",
                    ),
                ],
                className="model-catalog",
            ),
            dcc.Tabs(
                id="llm-nav",
                value="chat",
                children=[
                    dcc.Tab(label="Chat", value="chat"),
                    dcc.Tab(label="Dataset", value="dataset"),
                    dcc.Tab(label="Model evidence", value="evidence"),
                ],
                className="hug-tabrow llm-tabs",
            ),
            html.Main(
                [
                    html.Section(
                        [
                            html.Div(
                                id="llm-chat-thread",
                                children=_history_view([]),
                                className="chat-thread",
                            ),
                            html.Div(
                                [
                                    dcc.Textarea(
                                        id="llm-prompt",
                                        placeholder="Ask HUGIML to build, tune, explain, predict, or prepare governance evidence...",
                                        className="llm-prompt",
                                    ),
                                    dbc.Button(
                                        "Ask HUGIML",
                                        id="llm-submit",
                                        color="primary",
                                        className="btn-fit llm-submit",
                                    ),
                                ],
                                className="chat-composer",
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        label,
                                        id=identifier,
                                        outline=True,
                                        color="secondary",
                                        size="sm",
                                    )
                                    for identifier, label in [
                                        ("llm-quick-describe", "Describe data"),
                                        ("llm-quick-build", "Build model"),
                                        ("llm-quick-tune", "Tune model"),
                                        ("llm-quick-explain", "Explain model"),
                                        ("llm-quick-predict", "Predictions"),
                                        ("llm-quick-governance", "Governance report"),
                                    ]
                                ],
                                className="quick-row",
                            ),
                        ],
                        id="llm-page-chat",
                        className="llm-page",
                    ),
                    html.Section(
                        [html.Div(id="llm-dataset-profile"), _upload_panel()],
                        id="llm-page-dataset",
                        className="llm-page",
                        style={"display": "none"},
                    ),
                    html.Section(
                        id="llm-page-evidence", className="llm-page", style={"display": "none"}
                    ),
                ],
                className="hug-content llm-content",
            ),
            html.Div(id="llm-action-status", className="llm-status"),
        ],
        id="llm-root",
        className="hug-wrap theme-ocean",
    )


def _upload_panel() -> html.Div:
    return html.Div(
        [
            html.H4("Register a dataset"),
            html.P(
                "Upload CSV, TSV, or Parquet data, then choose its target column.",
                className="hero-p",
            ),
            dcc.Upload(
                id="llm-upload",
                children=html.Div(["Drop a file here or ", html.A("choose a file")]),
                className="upload-dropzone",
            ),
            html.Div(id="llm-upload-preview"),
            dcc.Dropdown(id="llm-upload-target", placeholder="Target column"),
            dbc.Input(id="llm-upload-name", placeholder="Dataset name"),
            dbc.Button("Register dataset", id="llm-register-upload", color="primary"),
            html.Div(id="llm-upload-status"),
        ],
        className="section-card upload-card",
    )


def _llm_css() -> str:
    return """
    .llm-hero{border-bottom:1px solid var(--bs)}.hero-status{justify-content:flex-end}
    .llm-toolbar{gap:12px;flex-wrap:wrap}.llm-dataset-select{width:280px}.llm-model-select{width:300px}
    .llm-mode label{margin-right:10px;font-size:.76rem}.llm-catalog-toggle label{display:flex;gap:7px;align-items:center;font-size:.76rem;margin:0}.llm-tabs{padding:0 22px}.llm-tabs .tab{padding:10px 18px!important}
    #llm-root .dash-dropdown{background:var(--sf)!important;color:var(--tx)!important;border-color:var(--bd)!important}
    #llm-root .dash-dropdown svg{color:var(--tx)!important}.theme-dark .llm-tabs .tab--selected{background:var(--sf)!important;color:var(--a)!important;border-color:var(--a)!important}
    .llm-content{max-width:1380px;width:100%;margin:0 auto}.llm-page{display:block}.chat-thread{display:grid;gap:10px}
    .qa-empty,.qa-card{background:var(--sf);border:1px solid var(--bs);border-radius:10px;padding:14px 16px;box-shadow:var(--sh)}
    .qa-card.question{margin-left:12%;border-left:3px solid var(--a2)}.qa-card.answer{margin-right:8%;border-left:3px solid var(--a)}
    .qa-card.refusal{border-left-color:var(--rd)}.qa-label{font-size:.60rem;font-weight:800;letter-spacing:.13em;color:var(--a);margin-bottom:6px}
    .qa-text,.answer-copy{font-size:.86rem;line-height:1.6;color:var(--tx)}.chat-composer{display:grid;grid-template-columns:1fr auto;gap:10px;margin-top:14px;background:var(--sf);border:1px solid var(--bs);border-radius:10px;padding:12px;box-shadow:var(--sh)}
    .llm-prompt{width:100%;min-height:88px;resize:vertical;border:1px solid var(--bs);border-radius:8px;padding:11px;background:var(--sf);color:var(--tx)}.llm-submit{min-width:150px}
    .quick-row{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}.section-card{background:var(--sf);border:1px solid var(--bs);border-radius:10px;padding:14px 16px;margin-bottom:12px;box-shadow:var(--sh)}
    .llm-metric-grid{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:9px;margin-bottom:12px}
    .upload-card{display:grid;gap:9px}.llm-status{position:fixed;right:18px;bottom:15px;z-index:100;font-size:.75rem;background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:7px 10px;box-shadow:var(--sh)}
    .model-catalog{margin:10px 22px 0;background:var(--sf);border:1px solid var(--bs);border-radius:10px;box-shadow:var(--sh)}.model-catalog summary{cursor:pointer;padding:10px 14px;font-size:.76rem;font-weight:800;color:var(--tx)}.model-catalog-body{padding:0 14px 14px}.model-help{font-size:.76rem;line-height:1.5;color:var(--mu);margin:8px 0}.model-command{display:block;white-space:pre-wrap;background:var(--cb);color:var(--tx);border:1px solid var(--bs);border-radius:7px;padding:8px 10px;font-size:.74rem}
    .empty-note{font-size:.78rem;color:var(--mu);padding:8px}.empty-results{padding:50px 20px;text-align:center;background:var(--sf);border:1px dashed var(--bd);border-radius:10px}
    @media(max-width:900px){.llm-metric-grid{grid-template-columns:repeat(3,1fr)}.llm-dataset-select,.llm-model-select{width:100%}.chat-composer{grid-template-columns:1fr}.qa-card.question{margin-left:0}.qa-card.answer{margin-right:0}}
    """


def create_app(debug: bool = False, *, repo_root: str | Path | None = None) -> dash.Dash:
    """Create the Dash LLM Assistant application."""

    del debug
    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    orchestrator = HUGIMLActionOrchestrator(repo_root=root)
    _CTX.clear()
    _CTX.update({"orchestrator": orchestrator, "repo_root": root, "upload_path": None})
    rows = _dataset_rows(False)
    model_options, model_value, status, model_rows = _model_options()
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="HUGIML LLM Assistant",
    )
    app.index_string = (
        """<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>"""
        + build_css("Ocean")
        + _llm_css()
        + """</style></head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""
    )
    app.layout = _shell(rows, model_options, model_value, status, model_rows)
    _register_callbacks(app)
    return app


def _register_callbacks(app: dash.Dash) -> None:
    @app.callback(Output("llm-root", "className"), Input("llm-theme", "value"))
    def theme_class(theme: str) -> str:
        name = str(theme or "Ocean").lower()
        return f"hug-wrap theme-{name}"

    @app.callback(
        Output("llm-page-chat", "style"),
        Output("llm-page-dataset", "style"),
        Output("llm-page-evidence", "style"),
        Input("llm-nav", "value"),
    )
    def switch_page(page: str):
        visible = {
            name: {"display": "block"} if page == name else {"display": "none"}
            for name in ("chat", "dataset", "evidence")
        }
        return visible["chat"], visible["dataset"], visible["evidence"]

    @app.callback(
        Output("llm-model", "options"),
        Output("llm-model", "value"),
        Output("llm-model-catalog", "children"),
        Output("llm-ollama-status", "children"),
        Input("llm-refresh-models", "n_clicks"),
        State("llm-model", "value"),
        prevent_initial_call=True,
    )
    def refresh_models(_clicks: int, current: str | None):
        status = check_ollama(timeout=0.8, use_cache=False)
        memory = get_memory_info()
        profile = recommend_profile(memory, repo_root=_CTX.get("repo_root"))
        options, selected, rows = _model_picker_state(
            memory,
            profile,
            status,
            current=current,
        )
        label = "Ollama connected" if status.get("ok") else "Deterministic mode ready"
        return options, selected, _model_catalog_view(rows, status), label

    @app.callback(
        Output("llm-dataset", "options"),
        Output("llm-dataset", "value"),
        Input("llm-include-benchmarks", "value"),
        State("llm-dataset", "value"),
    )
    def update_catalog(selection: list[str] | None, current: str | None):
        include = _benchmark_catalog_enabled(selection)
        rows = _dataset_rows(include)
        values = {str(row["name"]) for row in rows}
        selected = current if current in values else (str(rows[0]["name"]) if rows else None)
        return _dataset_options(rows), selected

    @app.callback(
        Output("llm-dataset", "value", allow_duplicate=True),
        Output("llm-nav", "value", allow_duplicate=True),
        Output("llm-launch-context", "data"),
        Input("llm-url", "search"),
        State("llm-dataset", "options"),
        prevent_initial_call=True,
    )
    def apply_context(search: str | None, options: list[dict[str, Any]]):
        launch_context = parse_launch_context(search)
        dataset = launch_context["dataset"]
        view = str(launch_context["view"] or "chat")
        allowed = {str(option.get("value")) for option in options or []}
        session_id = launch_context.get("session")
        orchestrator: HUGIMLActionOrchestrator = _CTX["orchestrator"]
        if session_id and session_id in orchestrator.sessions:
            orchestrator.last_session_id = session_id
        return (dataset if dataset in allowed else no_update), view, launch_context

    @app.callback(Output("llm-dataset-profile", "children"), Input("llm-dataset", "value"))
    def dataset_profile(dataset: str | None):
        if not dataset:
            return html.Div("Select a dataset.", className="empty-results")
        try:
            desc = _CTX["orchestrator"].registry.describe_dataset(dataset)
        except Exception as exc:
            return dbc.Alert(str(exc), color="danger")
        cards = [
            _metric_card("Rows", desc.get("rows")),
            _metric_card("Features", desc.get("features")),
            _metric_card("Classes", len(desc.get("class_counts") or {})),
            _metric_card("Missing columns", len(desc.get("missing_top") or {})),
        ]
        balance = [
            {"class": key, "count": value}
            for key, value in (desc.get("class_counts") or {}).items()
        ]
        missing = [
            {"feature": key, "missing fraction": value}
            for key, value in (desc.get("missing_top") or {}).items()
        ]
        return [
            html.Div(cards, className="llm-metric-grid"),
            html.Div([html.H4("Class balance"), _table(balance)], className="section-card"),
            html.Div([html.H4("Missingness"), _table(missing)], className="section-card"),
        ]

    action_inputs = [
        Input("llm-submit", "n_clicks"),
        *[Input(identifier, "n_clicks") for identifier in _QUICK_ACTIONS],
    ]

    @app.callback(
        Output("llm-chat-history", "data"),
        Output("llm-chat-thread", "children"),
        Output("llm-prompt", "value"),
        Output("llm-action-status", "children"),
        Output("llm-session-version", "data"),
        *action_inputs,
        State("llm-prompt", "value"),
        State("llm-dataset", "value"),
        State("llm-model", "value"),
        State("llm-response-mode", "value"),
        State("llm-include-benchmarks", "value"),
        State("llm-chat-history", "data"),
        State("llm-session-version", "data"),
        prevent_initial_call=True,
    )
    def submit_prompt(*values):
        prompt, dataset, model, mode, include, history, version = values[-7:]
        triggered = ctx.triggered_id
        if triggered in _QUICK_ACTIONS:
            prompt = _QUICK_ACTIONS[str(triggered)]
        prompt = str(prompt or "").strip()
        if not prompt:
            return no_update, no_update, no_update, "Enter a question.", no_update
        if not _ACTION_LOCK.acquire(blocking=False):
            return no_update, no_update, no_update, "Another action is running.", no_update
        try:
            result = run_prompt(
                _CTX["orchestrator"],
                prompt,
                dataset,
                model or "deterministic router only",
                response_mode=mode or "Fast",
                include_benchmarks=_benchmark_catalog_enabled(include),
                repo_root=_CTX.get("repo_root"),
            )
        finally:
            _ACTION_LOCK.release()
        updated = list(history or [])
        updated.append(
            {
                "prompt": prompt,
                "ok": bool(result.get("ok")),
                "action": result.get("action"),
                "result": result,
            }
        )
        state = "Completed" if result.get("ok") else "Action returned a review message"
        return updated, _history_view(updated), "", state, int(version or 0) + 1

    @app.callback(
        Output("llm-page-evidence", "children"),
        Input("llm-session-version", "data"),
        Input("llm-nav", "value"),
    )
    def refresh_evidence(_version: int, _page: str):
        return _evidence_view()

    @app.callback(
        Output("llm-upload-preview", "children"),
        Output("llm-upload-target", "options"),
        Output("llm-upload-target", "value"),
        Output("llm-upload-name", "value"),
        Output("llm-upload-path", "data"),
        Input("llm-upload", "contents"),
        State("llm-upload", "filename"),
        prevent_initial_call=True,
    )
    def preview_upload(contents: str | None, filename: str | None):
        if not contents or not filename:
            return no_update, no_update, no_update, no_update, no_update
        suffix = Path(filename).suffix.lower()
        if suffix not in {".csv", ".tsv", ".parquet"}:
            return (
                dbc.Alert("Use CSV, TSV, or Parquet data.", color="warning"),
                [],
                None,
                Path(filename).stem,
                None,
            )
        payload = base64.b64decode(contents.split(",", 1)[1])
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        handle.write(payload)
        handle.close()
        path = Path(handle.name)
        frame = _CTX["orchestrator"].registry._read_table(path, nrows=12)
        columns = [str(column) for column in frame.columns]
        return (
            _table(frame.to_dict("records"), page_size=6),
            columns,
            (columns[-1] if columns else None),
            Path(filename).stem,
            str(path),
        )

    @app.callback(
        Output("llm-upload-status", "children"),
        Output("llm-include-benchmarks", "value", allow_duplicate=True),
        Input("llm-register-upload", "n_clicks"),
        State("llm-upload-path", "data"),
        State("llm-upload-target", "value"),
        State("llm-upload-name", "value"),
        prevent_initial_call=True,
    )
    def register_upload(_clicks: int, path: str | None, target: str | None, name: str | None):
        if not path or not target:
            return dbc.Alert(
                "Upload data and choose a target column first.", color="warning"
            ), no_update
        try:
            info = _CTX["orchestrator"].registry.register_user_dataset(
                path, target_column=target, dataset_name=name
            )
        except Exception as exc:
            return dbc.Alert(str(exc), color="danger"), no_update
        return dbc.Alert(f"Registered {info.name} with target {info.target}.", color="success"), []


def main() -> None:
    """Launch the Dash LLM Assistant with default local settings."""

    app = create_app()
    app.run(host="127.0.0.1", port=8051, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

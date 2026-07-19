"""Dash Workbench experiment setup and result presentation.

The setup view supports Auto, Guided, and Advanced HUGIML configurations,
separate baseline controls, experiment execution, and promotion to Governance.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.tree import export_text

from hugiml.dashboard.dash_components.charts import bar_h, coef_waterfall
from hugiml.dashboard.dash_components.pages._shared import err, info, mc, sn, warn
from hugiml.dashboard.dash_components.tables import make_table
from hugiml.dashboard.workbench import (
    MODEL_CATALOG,
    _best_run_per_model,
    _coefficient_frame,
    _default_model_params,
    _feature_importance,
    _filter_hugiml_interaction_configs,
    _grid_varies_base_estimator,
    _hugiml_auto_params,
    _hugiml_core_default_grid,
    _hugiml_core_grid_names,
    _hugiml_features_frame,
    _hugiml_grid_count,
    _hugiml_guided_fast_tune_params,
    _hugiml_patterns_frame,
    _hugiml_runs_using_rpte,
    _hugiml_valid_feature_modes,
    _is_available,
    _leaderboard_frame,
    _model_short,
    _normalised_pattern_inventory,
    _normalised_rpte_rule_inventory,
    _normalised_rule_inventory,
    _scalar_metric,
    _successful_runs,
    _summarize_params_for_display,
)
from hugiml.dashboard.workbench import (
    run_experiments as _run,
)

try:
    from hugiml.dashboard.workbench import _promoted_evaluation_bundle
except Exception:
    _promoted_evaluation_bundle = None
try:
    from sklearn.multiclass import OneVsRestClassifier

    from hugiml.dashboard.workbench import DEFAULT_HUGIML_GRID_NAME
    from hugiml.rpte_bounded_lookahead_leafwise import LEAF_CONFIGS as RPTE_LC
    from hugiml.rpte_bounded_lookahead_leafwise import LeafWiseBoundedLookaheadRPTEFeatureLR

    _RPTE = True
except Exception:
    _RPTE = False
    RPTE_LC = {}
    DEFAULT_HUGIML_GRID_NAME = "performance_ho"
from hugiml.dashboard.components.patterns import (
    _get_rpte_final_term_rows,
    _rpte_flat_tree_sections,
    _rpte_rules_to_frame,
)
from hugiml.dashboard.runner import score_cases

_WB: dict[str, Any] = {"runs": []}


def _fmt(v):
    s = _scalar_metric(v)
    return "N/A" if s is None else f"{s:.4f}"


def _csv_values(value, cast, default, label):
    """Parse one value or a comma-separated candidate list."""
    raw = "" if value is None else str(value).strip()
    if not raw:
        return [default]
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        return [default]
    try:
        if cast is bool:
            parsed = []
            for item in items:
                token = item.lower()
                if token in {"1", "true", "yes", "y", "on"}:
                    parsed.append(True)
                elif token in {"0", "false", "no", "n", "off"}:
                    parsed.append(False)
                else:
                    raise ValueError
            return parsed
        return [cast(item) for item in items]
    except Exception as exc:
        raise ValueError(f"{label} must contain valid comma-separated values.") from exc


def _expand_candidates(base, grid):
    active = {key: list(values) for key, values in grid.items() if values}
    if not active:
        return [dict(base)]
    keys = list(active)
    candidates = []
    for combination in itertools.product(*(active[key] for key in keys)):
        params = dict(base)
        params.update(dict(zip(keys, combination)))
        candidates.append(params)
    return candidates


def _selected_values(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple, set)):
        values = [item for item in value if item not in (None, "")]
        return values or list(default)
    return [value]


def _configuration_combination_count(
    selected,
    hug_mode,
    grid_name,
    adv_L,
    adv_topK,
    adv_G,
    adv_fm,
    adv_strict,
    adv_aug,
    adv_relax,
    adv_ds,
    rpte_lc,
    rpte_d,
    rpte_ne,
    lr_C,
    lr_mi,
    dt_d,
    dt_l,
    rf_n,
    rf_d,
    rf_l,
    xgb_n,
    xgb_d,
    xgb_lr,
    lgbm_n,
    lgbm_d,
    lgbm_lr,
    ebm_b,
    ebm_i,
    rf2_ts,
    rf2_mr,
):
    """Return the number of selected model-parameter combinations."""
    total = 0
    for model_name in selected:
        if model_name == "HUGIML":
            if hug_mode == "auto":
                total += 1
                continue
            if hug_mode == "guided":
                guided = _hugiml_guided_fast_tune_params(grid_name)
                total += max(1, int(_hugiml_grid_count(guided.get("param_grid", {}))))
                continue

            feature_modes = _selected_values(adv_fm, ["original_plus_patterns"])
            downstream = _selected_values(adv_ds, ["lr"])
            base_estimator_count = 1 if "lr" in downstream else 0
            if "rpte" in downstream:
                if not _RPTE:
                    raise ValueError("RPTE is not available in this environment.")
                leaf_values = _selected_values(
                    rpte_lc,
                    ["3xD" if "3xD" in RPTE_LC else next(iter(RPTE_LC), "")],
                )
                depth_values = _csv_values(rpte_d, int, 4, "RPTE depth")
                estimator_values = _csv_values(rpte_ne, int, 10, "RPTE n_estimators")
                if any(int(value) < 1 for value in depth_values + estimator_values):
                    raise ValueError("RPTE depth and n_estimators must be positive.")
                base_estimator_count += len(leaf_values) * len(depth_values) * len(estimator_values)
            if base_estimator_count == 0:
                base_estimator_count = 1

            grid = {
                "L": _csv_values(adv_L, int, 2, "L"),
                "topK": _csv_values(adv_topK, int, 100, "topK"),
                "G": _csv_values(adv_G, float, 0.01, "G"),
                "feature_mode": feature_modes,
                "topk_budget_strict": _csv_values(adv_strict, bool, False, "Strict budget"),
                "augmented_pair_transforms": _csv_values(adv_aug, bool, True, "Augmented pair"),
                "interaction_relaxed_mining": _csv_values(adv_relax, bool, False, "Relaxed mining"),
                "base_estimator": list(range(base_estimator_count)),
            }
            if any(int(value) < 1 for value in grid["L"]):
                raise ValueError("L values must be at least 1.")
            if any(int(value) < 1 for value in grid["topK"]):
                raise ValueError("topK values must be at least 1.")
            if any(float(value) < 0 for value in grid["G"]):
                raise ValueError("G values must be non-negative.")
            candidates = _expand_candidates({}, grid)
            candidates, _ = _filter_hugiml_interaction_configs(candidates)
            total += len(candidates)
        elif model_name == "Logistic Regression":
            total += len(_csv_values(lr_C, float, 1.0, "Logistic C")) * len(
                _csv_values(lr_mi, int, 1000, "Logistic max_iter")
            )
        elif model_name == "Decision Tree":
            total += len(_csv_values(dt_d, int, 4, "Decision-tree max_depth")) * len(
                _csv_values(dt_l, int, 1, "Decision-tree min_samples_leaf")
            )
        elif model_name == "Random Forest":
            total += (
                len(_csv_values(rf_n, int, 200, "Random-forest n_estimators"))
                * len(_csv_values(rf_d, int, 0, "Random-forest max_depth"))
                * len(_csv_values(rf_l, int, 1, "Random-forest min_samples_leaf"))
            )
        elif model_name == "XGBoost":
            total += (
                len(_csv_values(xgb_n, int, 200, "XGBoost n_estimators"))
                * len(_csv_values(xgb_d, int, 4, "XGBoost max_depth"))
                * len(_csv_values(xgb_lr, float, 0.05, "XGBoost learning_rate"))
            )
        elif model_name == "LightGBM":
            total += (
                len(_csv_values(lgbm_n, int, 200, "LightGBM n_estimators"))
                * len(_csv_values(lgbm_d, int, -1, "LightGBM max_depth"))
                * len(_csv_values(lgbm_lr, float, 0.05, "LightGBM learning_rate"))
            )
        elif model_name == "EBM":
            total += len(_csv_values(ebm_b, int, 32, "EBM max_bins")) * len(
                _csv_values(ebm_i, int, 5, "EBM interactions")
            )
        elif model_name == "RuleFit":
            total += len(_csv_values(rf2_ts, int, 4, "RuleFit tree_size")) * len(
                _csv_values(rf2_mr, int, 100, "RuleFit max_rules")
            )
        else:
            total += 1
    return total


def _candidate_help(extra=""):
    message = (
        "Enter one value for a single configuration or comma-separated values to evaluate "
        "multiple candidates. Candidate combinations are selected by validation CV ROC-AUC."
    )
    if extra:
        message += " " + extra
    return html.Div(message, className="setup-help candidate-help")


# ─── Model picker ──────────────────────────────────────────────────────────────
def _model_picker():
    groups = []
    for cat, specs in MODEL_CATALOG.items():
        cid = cat.replace(" ", "_").lower()
        options = [
            {
                "label": f" {_model_short(s.name)}" + ("" if _is_available(s) else " ✗"),
                "value": s.name,
                "disabled": not _is_available(s),
            }
            for s in specs
        ]
        defaults = [
            s.name
            for s in specs
            if _is_available(s) and s.name in ("HUGIML", "Logistic Regression")
        ]
        groups.append(
            html.Div(
                [
                    html.Div(
                        cat,
                        style={
                            "fontSize": ".68rem",
                            "fontWeight": "700",
                            "letterSpacing": ".10em",
                            "textTransform": "uppercase",
                            "color": "var(--mu)",
                            "marginBottom": "4px",
                        },
                    ),
                    dbc.Checklist(
                        id=f"wb-chk-{cid}",
                        options=options,
                        value=defaults,
                        inputStyle={"marginRight": "5px"},
                        style={"fontSize": ".82rem"},
                        className="mb-2",
                    ),
                ]
            )
        )
    return html.Div(groups)


# ─── HUGIML config ─────────────────────────────────────────────────────────────
def _hugiml_config():
    grid_names = _hugiml_core_grid_names()
    feature_modes = _hugiml_valid_feature_modes()
    leaf_options = [{"label": key, "value": key} for key in sorted(RPTE_LC)] if _RPTE else []
    downstream_options = [{"label": "HUGIML logistic regression", "value": "lr"}]
    if _RPTE:
        downstream_options.append({"label": "RPTE (bounded-lookahead)", "value": "rpte"})
    default_grid = (
        DEFAULT_HUGIML_GRID_NAME
        if DEFAULT_HUGIML_GRID_NAME in grid_names
        else (grid_names[0] if grid_names else "performance_ho")
    )
    default_leaf = "3xD" if "3xD" in RPTE_LC else (next(iter(RPTE_LC), "") if RPTE_LC else "")
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6("HUGIML Configuration", className="fw-bold mb-2"),
                dbc.RadioItems(
                    id="wb-hug-mode",
                    options=[
                        {"label": "Auto (best defaults)", "value": "auto"},
                        {"label": "Guided (named grid)", "value": "guided"},
                        {"label": "Advanced (candidate grid)", "value": "advanced"},
                    ],
                    value="auto",
                    inline=True,
                    inputStyle={"marginRight": "5px", "marginLeft": "10px"},
                    className="mb-2",
                    style={"fontSize": ".82rem"},
                ),
                html.Div(
                    id="wb-auto-box",
                    children=[
                        html.Small(
                            "Runs one best-default HUGIML configuration with adaptive binning enabled.",
                            className="text-muted",
                        ),
                    ],
                ),
                html.Div(
                    id="wb-guided-box",
                    style={"display": "none"},
                    children=[
                        html.Small(
                            "Uses a named HUGIML core hyperparameter grid through the cached tuning path.",
                            className="text-muted d-block mb-2",
                        ),
                        html.Label("Grid", className="form-label"),
                        dcc.Dropdown(
                            id="wb-grid-name",
                            options=[{"label": name, "value": name} for name in grid_names],
                            value=default_grid,
                            clearable=False,
                            className="compact-dropdown mb-1",
                        ),
                        html.Div(id="wb-grid-info"),
                    ],
                ),
                html.Div(
                    id="wb-adv-box",
                    style={"display": "none"},
                    children=[
                        _candidate_help(
                            "Feature modes and downstream estimators may also be selected in combination."
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("L value(s)", className="form-label"),
                                        dcc.Input(
                                            id="wb-adv-L",
                                            type="text",
                                            value="2",
                                            className="form-control form-control-sm",
                                            placeholder="1, 2",
                                        ),
                                    ],
                                    className="setup-field",
                                ),
                                html.Div(
                                    [
                                        html.Label("topK value(s)", className="form-label"),
                                        dcc.Input(
                                            id="wb-adv-topK",
                                            type="text",
                                            value="100",
                                            className="form-control form-control-sm",
                                            placeholder="50, 100",
                                        ),
                                    ],
                                    className="setup-field",
                                ),
                                html.Div(
                                    [
                                        html.Label("G value(s)", className="form-label"),
                                        dcc.Input(
                                            id="wb-adv-G",
                                            type="text",
                                            value="0.01",
                                            className="form-control form-control-sm",
                                            placeholder="0.01, 0.001",
                                        ),
                                    ],
                                    className="setup-field",
                                ),
                                html.Div(
                                    [
                                        html.Label("Feature mode(s)", className="form-label"),
                                        dcc.Dropdown(
                                            id="wb-adv-fm",
                                            options=[
                                                {"label": mode, "value": mode}
                                                for mode in feature_modes
                                            ],
                                            value=["original_plus_patterns"]
                                            if "original_plus_patterns" in feature_modes
                                            else feature_modes[:1],
                                            multi=True,
                                            clearable=False,
                                            className="compact-dropdown",
                                        ),
                                    ],
                                    className="setup-field",
                                ),
                            ],
                            className="candidate-grid candidate-grid-four",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Strict budget value(s)", className="form-label"
                                        ),
                                        dcc.Input(
                                            id="wb-adv-strict",
                                            type="text",
                                            value="false",
                                            className="form-control form-control-sm",
                                            placeholder="false, true",
                                        ),
                                    ],
                                    className="setup-field",
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Augmented pair value(s)", className="form-label"
                                        ),
                                        dcc.Input(
                                            id="wb-adv-aug",
                                            type="text",
                                            value="true",
                                            className="form-control form-control-sm",
                                            placeholder="true, false",
                                        ),
                                    ],
                                    className="setup-field",
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Relaxed mining value(s)", className="form-label"
                                        ),
                                        dcc.Input(
                                            id="wb-adv-relax",
                                            type="text",
                                            value="false",
                                            className="form-control form-control-sm",
                                            placeholder="false, true",
                                        ),
                                    ],
                                    className="setup-field",
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Downstream estimator(s)", className="form-label"
                                        ),
                                        dcc.Dropdown(
                                            id="wb-adv-ds",
                                            options=downstream_options,
                                            value=["lr"],
                                            multi=True,
                                            clearable=False,
                                            className="compact-dropdown",
                                        ),
                                    ],
                                    className="setup-field",
                                ),
                            ],
                            className="candidate-grid candidate-grid-four mt-2",
                        ),
                        html.Div(
                            id="wb-rpte-box",
                            style={"display": "none"},
                            children=[
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Leaf configuration(s)", className="form-label"
                                                ),
                                                dcc.Dropdown(
                                                    id="wb-rpte-lc",
                                                    options=leaf_options,
                                                    value=[default_leaf] if default_leaf else [],
                                                    multi=True,
                                                    clearable=False,
                                                    className="compact-dropdown",
                                                ),
                                            ],
                                            className="setup-field",
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Depth value(s)", className="form-label"
                                                ),
                                                dcc.Input(
                                                    id="wb-rpte-d",
                                                    type="text",
                                                    value="4",
                                                    className="form-control form-control-sm",
                                                    placeholder="3, 4",
                                                ),
                                            ],
                                            className="setup-field",
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "n_estimators value(s)", className="form-label"
                                                ),
                                                dcc.Input(
                                                    id="wb-rpte-ne",
                                                    type="text",
                                                    value="10",
                                                    className="form-control form-control-sm",
                                                    placeholder="5, 10",
                                                ),
                                            ],
                                            className="setup-field",
                                        ),
                                    ],
                                    className="candidate-grid candidate-grid-three mt-2",
                                ),
                            ]
                            if _RPTE
                            else [html.Div(id="wb-rpte-box")],
                        ),
                    ],
                ),
            ]
        ),
        className="mb-3",
    )


# ─── Baseline configs (each model separate) ────────────────────────────────────
def _generic_config():
    help_text = _candidate_help(
        "Use 0 for an unlimited tree depth and -1 for an unlimited LightGBM depth."
    )

    def field(label, component_id, value, placeholder=""):
        return html.Div(
            [
                html.Label(label, className="form-label"),
                dcc.Input(
                    id=component_id,
                    type="text",
                    value=str(value),
                    placeholder=placeholder,
                    className="form-control form-control-sm",
                ),
            ],
            className="setup-field",
        )

    return html.Div(
        [
            help_text,
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        html.Div(
                            [
                                field("C value(s)", "wb-lr-C", 1.0, "0.1, 1.0, 10"),
                                field("max_iter value(s)", "wb-lr-mi", 1000, "500, 1000"),
                            ],
                            className="candidate-grid candidate-grid-two",
                        ),
                        title="Logistic Regression",
                        item_id="lr",
                        id="wb-param-lr",
                    ),
                    dbc.AccordionItem(
                        html.Div(
                            [
                                field("max_depth value(s)", "wb-dt-d", 4, "3, 4, 0"),
                                field("min_samples_leaf value(s)", "wb-dt-l", 1, "1, 2, 5"),
                            ],
                            className="candidate-grid candidate-grid-two",
                        ),
                        title="Decision Tree",
                        item_id="dt",
                        id="wb-param-dt",
                    ),
                    dbc.AccordionItem(
                        html.Div(
                            [
                                field("n_estimators value(s)", "wb-rf-n", 200, "100, 200"),
                                field("max_depth value(s)", "wb-rf-d", 0, "4, 8, 0"),
                                field("min_samples_leaf value(s)", "wb-rf-l", 1, "1, 2"),
                            ],
                            className="candidate-grid candidate-grid-three",
                        ),
                        title="Random Forest",
                        item_id="rf",
                        id="wb-param-rf",
                    ),
                    dbc.AccordionItem(
                        html.Div(
                            [
                                field("n_estimators value(s)", "wb-xgb-n", 200, "100, 200"),
                                field("max_depth value(s)", "wb-xgb-d", 4, "3, 4, 6"),
                                field(
                                    "learning_rate value(s)", "wb-xgb-lr", 0.05, "0.03, 0.05, 0.1"
                                ),
                            ],
                            className="candidate-grid candidate-grid-three",
                        ),
                        title="XGBoost",
                        item_id="xgb",
                        id="wb-param-xgb",
                    ),
                    dbc.AccordionItem(
                        html.Div(
                            [
                                field("n_estimators value(s)", "wb-lgbm-n", 200, "100, 200"),
                                field("max_depth value(s)", "wb-lgbm-d", -1, "4, 8, -1"),
                                field(
                                    "learning_rate value(s)", "wb-lgbm-lr", 0.05, "0.03, 0.05, 0.1"
                                ),
                            ],
                            className="candidate-grid candidate-grid-three",
                        ),
                        title="LightGBM",
                        item_id="lgbm",
                        id="wb-param-lgbm",
                    ),
                    dbc.AccordionItem(
                        html.Div(
                            [
                                field("max_bins value(s)", "wb-ebm-b", 32, "16, 32, 64"),
                                field("interactions value(s)", "wb-ebm-i", 5, "0, 5, 10"),
                            ],
                            className="candidate-grid candidate-grid-two",
                        ),
                        title="EBM (Explainable Boosting Machine)",
                        item_id="ebm",
                        id="wb-param-ebm",
                    ),
                    dbc.AccordionItem(
                        html.Div(
                            [
                                field("tree_size value(s)", "wb-rf2-ts", 4, "3, 4, 5"),
                                field("max_rules value(s)", "wb-rf2-mr", 100, "50, 100, 200"),
                            ],
                            className="candidate-grid candidate-grid-two",
                        ),
                        title="RuleFit",
                        item_id="rulefit",
                        id="wb-param-rulefit",
                    ),
                ],
                start_collapsed=True,
                flush=True,
                className="mb-3",
            ),
        ]
    )


# ─── Compact setup panel (left column) ─────────────────────────────────────────
def _lbl(text):
    return html.Span(text, className="setup-lbl")


def _mini(id_, type_="number", value=None, width="60px", **kw):
    return dcc.Input(
        id=id_,
        type=type_,
        value=value,
        style={
            "width": width,
            "fontSize": ".78rem",
            "padding": "2px 5px",
            "borderRadius": "5px",
            "border": "1px solid var(--bs)",
            "background": "transparent",
            "color": "var(--tx)",
        },
        **kw,
    )


def _setup_panel(ctx):
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("SELECT MODELS", className="setup-lbl"),
                            _model_picker(),
                        ],
                        className="setup-card setup-card-compact model-picker-card",
                    ),
                    html.Div(
                        [
                            html.Div("CONFIGURATION", className="setup-lbl"),
                            html.Div(_hugiml_config(), id="wb-hugiml-config-wrap"),
                            html.Div(
                                dbc.Accordion(
                                    [
                                        dbc.AccordionItem(
                                            _generic_config(),
                                            title="Comparison model parameters",
                                            item_id="comparison-params",
                                        ),
                                    ],
                                    start_collapsed=True,
                                    flush=True,
                                    className="config-accordion",
                                ),
                                id="wb-comparison-config-wrap",
                            ),
                        ],
                        className="setup-card setup-card-compact model-config-card",
                    ),
                ],
                className="setup-config-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("RUN EXPERIMENT", className="setup-lbl"),
                            html.Div(
                                "The selected dataset, CV settings, model configurations, fitting, and comparison are executed together.",
                                className="setup-help",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Run 2 model combinations",
                                id="wb-run-btn",
                                n_clicks=0,
                                className="btn-fit run-primary-action",
                            ),
                            dcc.Loading(
                                id="wb-run-load",
                                children=html.Div(id="wb-run-status"),
                                type="circle",
                                color="var(--a)",
                            ),
                        ],
                        className="run-action-controls",
                    ),
                ],
                className="run-action-bar",
            ),
        ],
        className="experiment-setup",
    )


# ─── Right results panel ──────────────────────────────────────────────────────
def _result_stubs():
    return html.Div(
        [
            html.Div(id="wb-promote-status", style={"display": "none"}),
            dcc.Dropdown(id="wb-promote-dd", options=[], value=None, style={"display": "none"}),
            html.Button(id="wb-promote-btn", n_clicks=0, style={"display": "none"}),
            dcc.Dropdown(id="wb-artifact-kind", options=[], value=None, style={"display": "none"}),
            dcc.Dropdown(id="wb-artifact-base", options=[], value=None, style={"display": "none"}),
            dcc.Dropdown(id="wb-artifact-alt", options=[], value=None, style={"display": "none"}),
            html.Div(id="wb-artifact-output", style={"display": "none"}),
            dcc.Dropdown(id="wb-inspect-run", options=[], value=None, style={"display": "none"}),
            html.Div(id="wb-inspect-output", style={"display": "none"}),
        ]
    )


def _curve_panel(runs, ctx):
    if not ctx or ctx.get("y") is None:
        return info("Run experiments to compare ROC and precision-recall curves.")
    y = np.asarray(ctx.get("y"), dtype=int).reshape(-1)
    best = list(_best_run_per_model(runs).values())
    roc_fig = go.Figure()
    pr_fig = go.Figure()
    palette = ["#2563eb", "#06b6d4", "#534AB7", "#1D9E75", "#EF9F27", "#E24B4A", "#0d9488"]
    plotted = 0
    for idx, run in enumerate(best):
        artifact = run.get("artifact") or {}
        proba = artifact.get("y_proba") if isinstance(artifact, dict) else None
        if proba is None:
            continue
        p = np.asarray(proba, dtype=float).reshape(-1)
        n = min(len(y), len(p))
        if n < 2 or len(np.unique(y[:n])) < 2:
            continue
        label = _model_short(str(run.get("model", "unknown")))
        color = palette[idx % len(palette)]
        fpr, tpr, _ = roc_curve(y[:n], p[:n])
        precision, recall, _ = precision_recall_curve(y[:n], p[:n])
        roc_fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=label, line={"color": color, "width": 2})
        )
        pr_fig.add_trace(
            go.Scatter(
                x=recall, y=precision, mode="lines", name=label, line={"color": color, "width": 2}
            )
        )
        plotted += 1
    if not plotted:
        return info("No out-of-fold probability artifacts are available for curve comparison.")
    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line={"dash": "dash", "color": "#888780", "width": 1},
        )
    )
    common = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=310,
        margin={"t": 42, "b": 45, "l": 55, "r": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1},
        font={"family": "Inter, sans-serif", "size": 11},
    )
    roc_fig.update_layout(
        **common,
        title={"text": "ROC comparison", "font": {"size": 13}},
        xaxis_title="False-positive rate",
        yaxis_title="True-positive rate",
    )
    pr_fig.update_layout(
        **common,
        title={"text": "Precision-recall comparison", "font": {"size": 13}},
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    for fig in (roc_fig, pr_fig):
        fig.update_xaxes(gridcolor="rgba(128,128,128,.10)", range=[0, 1])
        fig.update_yaxes(gridcolor="rgba(128,128,128,.10)", range=[0, 1])
    return html.Div(
        [
            sn(
                "Curves use stitched out-of-fold predictions from the best configuration of each model."
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=roc_fig, config={"displayModeBar": False}), md=6),
                    dbc.Col(dcc.Graph(figure=pr_fig, config={"displayModeBar": False}), md=6),
                ]
            ),
        ]
    )


def _rpte_tree_panel(model, rid):
    rows = _get_rpte_final_term_rows(model, include_zero_direct=True)
    if not rows:
        return None
    frame = _rpte_rules_to_frame(rows)
    if frame.empty:
        return None
    leaf_mask = frame.get("is_leaf_term", pd.Series(False, index=frame.index)).astype(bool)
    direct_mask = frame.get("is_direct_source_term", pd.Series(False, index=frame.index)).astype(
        bool
    )
    leaves = frame.loc[leaf_mask].copy()
    direct = frame.loc[direct_mask].copy()
    sections = _rpte_flat_tree_sections(rows, leaves) if not leaves.empty else []
    tree_count = len(sections)

    metrics = dbc.Row(
        [
            dbc.Col(mc("RPTE trees", f"{tree_count:,}"), md=4, className="mb-2"),
            dbc.Col(mc("Leaf terms", f"{len(leaves):,}"), md=4, className="mb-2"),
            dbc.Col(mc("Direct terms", f"{len(direct):,}"), md=4, className="mb-2"),
        ]
    )

    tree_items = []
    for index, section in enumerate(sections):
        coefficients = section.get("coefficients")
        if not isinstance(coefficients, pd.DataFrame):
            coefficients = pd.DataFrame(coefficients or [])
        coefficient_view = coefficients.drop(columns=["_source_row_index"], errors="ignore")
        title = (
            f"{section.get('title', f'RPTE tree {index + 1}')} | "
            f"{int(section.get('leaf_count', len(coefficient_view)) or 0)} leaves | "
            f"depth {int(section.get('max_depth', 0) or 0)}"
        )
        tree_items.append(
            dbc.AccordionItem(
                [
                    html.Div(
                        "Shared path prefixes are shown once; each terminal line is one leaf indicator in the final logistic model.",
                        className="setup-help mb-2",
                    ),
                    html.Pre(
                        str(section.get("text", "No tree text available.")),
                        className="tree-pre rpte-tree-pre",
                    ),
                    html.H6("Leaf coefficients", className="results-card-title mt-3"),
                    make_table(
                        coefficient_view,
                        tid=f"wb-rpte-tree-{rid}-{index}",
                        height="240px",
                    )
                    if not coefficient_view.empty
                    else info("No coefficients are available for this tree."),
                ],
                title=title,
                item_id=f"tree-{index}",
            )
        )

    tree_content = (
        html.Div(
            [
                sn(
                    "Select a tree to inspect its merged root-to-leaf structure and the fitted coefficient attached to every terminal leaf."
                ),
                dbc.Accordion(
                    tree_items,
                    active_item="tree-0" if tree_items else None,
                    start_collapsed=not bool(tree_items),
                    flush=True,
                    always_open=False,
                ),
            ]
        )
        if tree_items
        else info(
            "The fitted downstream estimator exposes direct terms but no accepted RPTE leaf trees."
        )
    )

    leaf_columns = [
        "rule_id",
        "class",
        "tree",
        "leaf",
        "effect",
        "coefficient",
        "odds_multiplier",
        "support_rate",
        "support_count",
        "n_conditions",
        "conjunction",
        "condition_steps",
        "raw_sources",
        "backend",
    ]
    direct_columns = [
        "class",
        "source_family",
        "source_column",
        "source_display_name",
        "effect",
        "coefficient",
        "odds_multiplier",
        "raw_sources",
        "backend",
    ]
    tab_items = [
        dbc.Tab(tree_content, label="Tree / leaf view", tab_id=f"{rid}-rpte-trees"),
        dbc.Tab(
            make_table(
                leaves[[column for column in leaf_columns if column in leaves.columns]],
                tid=f"wb-rpte-leaves-{rid}",
                height="330px",
                filterable=True,
            )
            if not leaves.empty
            else info("No RPTE leaf terms are available."),
            label="All leaf terms",
            tab_id=f"{rid}-rpte-leaves",
        ),
    ]
    if not direct.empty:
        tab_items.append(
            dbc.Tab(
                make_table(
                    direct[[column for column in direct_columns if column in direct.columns]],
                    tid=f"wb-rpte-direct-{rid}",
                    height="300px",
                    filterable=True,
                ),
                label="Direct terms",
                tab_id=f"{rid}-rpte-direct",
            )
        )
    return html.Div(
        [
            html.H6("RPTE representation", className="fw-bold mt-3 mb-2"),
            metrics,
            dbc.Tabs(
                tab_items,
                id=f"wb-rpte-tabs-{rid}",
                active_tab=f"{rid}-rpte-trees",
                className="rpte-result-tabs",
            ),
        ]
    )


def _run_detail(run, id_prefix="detail"):
    if not run:
        return info("Select a successful model run to inspect.")
    mn = run.get("model")
    rid = f"{id_prefix}-" + str(run.get("run_id", "run")).replace(":", "-").replace(" ", "-")
    artifact = run.get("artifact") or {}
    model = artifact.get("model") if isinstance(artifact, dict) else None
    X_model = artifact.get("X") if isinstance(artifact, dict) else None
    if X_model is None and isinstance(artifact, dict):
        X_model = artifact.get("feature_frame")
    body = [
        dbc.Row(
            [
                dbc.Col(mc("CV AUC", _fmt(run.get("cv_roc_auc"))), md=3, className="mb-2"),
                dbc.Col(mc("F1", _fmt(run.get("f1"))), md=3, className="mb-2"),
                dbc.Col(mc("Precision", _fmt(run.get("precision"))), md=3, className="mb-2"),
                dbc.Col(mc("Recall", _fmt(run.get("recall"))), md=3, className="mb-2"),
            ]
        ),
    ]
    if model is not None and X_model is not None:
        try:
            fi = _feature_importance(model, X_model, max_rows=20)
            if not fi.empty:
                name_col = next(
                    (c for c in fi.columns if c in ("feature", "display_name", "pattern")),
                    fi.columns[0],
                )
                value_col = next(
                    (
                        c
                        for c in fi.columns
                        if "coefficient" in c.lower() or "importance" in c.lower()
                    ),
                    None,
                )
                if value_col:
                    plotted = fi.head(12).sort_values(value_col)
                    body.append(
                        html.Div(
                            dcc.Graph(
                                figure=coef_waterfall(
                                    plotted[name_col].astype(str).tolist(),
                                    plotted[value_col].tolist(),
                                    title=f"Top features — {_model_short(str(mn))}",
                                    h=260,
                                ),
                                config={"displayModeBar": False},
                            ),
                            className="inspect-feature-chart",
                        )
                    )
                body.append(
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                make_table(fi.head(40), tid=f"wb-fi-{rid}", height="240px"),
                                title="Feature importance table",
                            )
                        ],
                        start_collapsed=True,
                        flush=True,
                    )
                )
        except Exception:
            pass
    if mn == "HUGIML" and model is not None:
        patterns = _hugiml_patterns_frame(model)
        features = _hugiml_features_frame(model)
        if not patterns.empty:
            body.append(
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            make_table(patterns.head(75), tid=f"wb-pat-{rid}", height="260px"),
                            title="Pattern inventory",
                        )
                    ],
                    start_collapsed=True,
                    flush=True,
                    className="mt-2",
                )
            )
        if not features.empty:
            body.append(
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            make_table(features.head(75), tid=f"wb-feat-{rid}", height="260px"),
                            title="Selected downstream features",
                        )
                    ],
                    start_collapsed=True,
                    flush=True,
                    className="mt-2",
                )
            )
        try:
            rpte_panel = _rpte_tree_panel(model, rid)
            if rpte_panel is not None:
                body.append(rpte_panel)
        except Exception:
            pass
    confusion = artifact.get("confusion_matrix") if isinstance(artifact, dict) else None
    if confusion is not None:
        try:
            cm_df = (
                pd.DataFrame(
                    confusion, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]
                )
                .reset_index()
                .rename(columns={"index": ""})
            )
            body.append(
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            make_table(cm_df, tid=f"wb-cm-{rid}", height=None),
                            title="Confusion matrix",
                        )
                    ],
                    start_collapsed=True,
                    flush=True,
                    className="mt-2",
                )
            )
        except Exception:
            pass
    params = {
        k: str(v)[:160] for k, v in (run.get("params") or {}).items() if k != "base_estimator"
    }
    body.append(
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    html.Pre(str(params), className="parameter-pre"), title="Parameters"
                )
            ],
            start_collapsed=True,
            flush=True,
            className="mt-2",
        )
    )
    return html.Div(
        [
            html.Div(
                [
                    html.Span(_model_short(str(mn)), className="inspect-title"),
                    html.Span(str(run.get("run_id", "")), className="inspect-run-id"),
                    html.Span(f"AUC {_fmt(run.get('cv_roc_auc'))}", className="chip"),
                ],
                className="inspect-heading",
            ),
            *body,
        ]
    )


def _interpretability_panel(runs):
    best = list(_best_run_per_model(runs).values())
    interpretable = [
        r
        for r in best
        if r.get("model") in {"HUGIML", "Logistic Regression", "Decision Tree", "EBM", "RuleFit"}
    ]
    if not interpretable:
        return info(
            "Run HUGIML, Logistic Regression, Decision Tree, EBM, or RuleFit to inspect interpretable artifacts."
        )
    rows = []
    for run in interpretable:
        artifact = run.get("artifact") or {}
        fi = artifact.get("feature_importance") if isinstance(artifact, dict) else None
        model_name = str(run.get("model"))
        artifact_name = {
            "HUGIML": "Patterns, selected features, RPTE terms",
            "Logistic Regression": "Coefficients",
            "Decision Tree": "Tree structure and importance",
            "EBM": "Additive terms",
            "RuleFit": "Rules and coefficients",
        }.get(model_name, "Feature importance")
        rows.append(
            {
                "run_id": run.get("run_id"),
                "model": _model_short(model_name),
                "artifact": artifact_name,
                "feature_importance": bool(isinstance(fi, pd.DataFrame) and not fi.empty),
                "cv_roc_auc": _scalar_metric(run.get("cv_roc_auc")),
            }
        )
    items = [
        dbc.AccordionItem(
            _run_detail(run, id_prefix="interpretability"),
            title=f"{_model_short(str(run.get('model')))} — {run.get('run_id')}",
        )
        for run in interpretable
    ]
    return html.Div(
        [
            make_table(pd.DataFrame(rows), tid="wb-interpretability-summary", height=None),
            dbc.Accordion(
                items,
                active_item="item-0" if items else None,
                start_collapsed=True,
                flush=True,
                className="mt-3",
            ),
        ]
    )


def _artifact_kind_options(runs):
    options = []
    counts = {
        name: len([r for r in _successful_runs(runs) if r.get("model") == name])
        for name in ("HUGIML", "RuleFit", "Logistic Regression", "Decision Tree")
    }
    if counts["HUGIML"] >= 2:
        options.append({"label": "HUGIML patterns", "value": "hugiml"})
    if len(_hugiml_runs_using_rpte(runs)) >= 2:
        options.append({"label": "RPTE final terms", "value": "rpte"})
    if counts["RuleFit"] >= 2:
        options.append({"label": "RuleFit rules", "value": "rulefit"})
    if counts["Logistic Regression"] >= 2:
        options.append({"label": "Logistic coefficients", "value": "lr"})
    if counts["Decision Tree"] >= 2:
        options.append({"label": "Decision-tree text", "value": "dt"})
    return options


def _artifact_runs(kind, runs):
    if kind == "rpte":
        return _hugiml_runs_using_rpte(runs)
    model_name = {
        "hugiml": "HUGIML",
        "rulefit": "RuleFit",
        "lr": "Logistic Regression",
        "dt": "Decision Tree",
    }.get(kind)
    return [r for r in _successful_runs(runs) if r.get("model") == model_name]


def _unwrap_downstream_estimator(estimator):
    target = estimator
    seen = set()
    while target is not None and id(target) not in seen:
        seen.add(id(target))
        nested = getattr(target, "estimator", None)
        if nested is None:
            break
        target = nested
    return target


def _downstream_estimator_label(run):
    if str(run.get("model")) != "HUGIML":
        return ""

    params = run.get("params") if isinstance(run.get("params"), dict) else {}
    selected = (
        params.get("selected_params") if isinstance(params.get("selected_params"), dict) else params
    )
    marker = object()
    estimator = selected.get("base_estimator", marker)

    artifact = run.get("artifact") if isinstance(run.get("artifact"), dict) else {}
    model = artifact.get("model") if isinstance(artifact, dict) else None
    if estimator is marker:
        estimator = getattr(model, "base_estimator", None) if model is not None else None

    if estimator is None:
        return "HUGIML logistic regression"

    target = _unwrap_downstream_estimator(estimator)
    name = type(target).__name__ if target is not None else type(estimator).__name__
    if "RPTE" in name.upper():
        details = []
        leaf_config = getattr(target, "leaf_config", None)
        depth = getattr(target, "depth", None)
        n_estimators = getattr(target, "n_estimators", None)
        if leaf_config not in (None, ""):
            details.append(f"leaf={leaf_config}")
        if depth is not None:
            details.append(f"depth={depth}")
        if n_estimators is not None:
            details.append(f"trees={n_estimators}")
        return "RPTE" + (" · " + " · ".join(details) if details else "")
    return name


def _run_config_values(run):
    values = {
        "run_id": run.get("run_id"),
        "cv_roc_auc": _scalar_metric(run.get("cv_roc_auc")),
    }
    if str(run.get("model")) == "HUGIML":
        values["downstream_estimator"] = _downstream_estimator_label(run)

    params = run.get("params") if isinstance(run.get("params"), dict) else {}
    if params.get("mode") == "guided_fast_tune":
        values["mode"] = "Guided fast tune"
        values["candidate_count"] = params.get("candidate_count")
        values["fast_path_used"] = params.get("fast_path_used")
        selected = (
            params.get("selected_params") if isinstance(params.get("selected_params"), dict) else {}
        )
        for key, value in selected.items():
            if key == "base_estimator" or str(key).startswith("__"):
                continue
            values[str(key)] = value
        return values

    for key, value in params.items():
        if key == "base_estimator" or str(key).startswith("__"):
            continue
        values[str(key)] = value
    return values


def _config_pair_frame(base_run, alt_run):
    base_values = _run_config_values(base_run)
    alt_values = _run_config_values(alt_run)
    preferred = [
        "run_id",
        "downstream_estimator",
        "cv_roc_auc",
        "mode",
        "candidate_count",
        "fast_path_used",
    ]
    keys = [key for key in preferred if key in base_values or key in alt_values]
    keys.extend(sorted((set(base_values) | set(alt_values)) - set(keys)))
    return pd.DataFrame(
        [
            {
                "field": key,
                "Base": str(base_values.get(key, ""))[:180],
                "Alternate": str(alt_values.get(key, ""))[:180],
            }
            for key in keys
        ]
    )


def _artifact_run_label(run):
    parts = [str(run.get("run_id"))]
    if str(run.get("model")) == "HUGIML":
        parts.append(_downstream_estimator_label(run))
    parts.append(f"AUC {_fmt(run.get('cv_roc_auc'))}")
    return " — ".join(part for part in parts if part)


def _set_delta(base_df, alt_df, key_col, label_col=None):
    label_col = label_col or key_col
    base_keys = (
        set(base_df.get(key_col, pd.Series(dtype=str)).astype(str)) if not base_df.empty else set()
    )
    alt_keys = (
        set(alt_df.get(key_col, pd.Series(dtype=str)).astype(str)) if not alt_df.empty else set()
    )
    rows = [
        {"change": "+", "status": "added", label_col: key} for key in sorted(alt_keys - base_keys)
    ]
    rows += [
        {"change": "−", "status": "removed", label_col: key} for key in sorted(base_keys - alt_keys)
    ]
    rows += [
        {"change": "", "status": "unchanged", label_col: key}
        for key in sorted(base_keys & alt_keys)
    ]
    return pd.DataFrame(rows), len(base_keys), len(alt_keys)


def _artifact_comparison(kind, base_id, alt_id):
    runs = _artifact_runs(kind, _WB.get("runs", []))
    by_id = {str(r.get("run_id")): r for r in runs}
    base_run = by_id.get(str(base_id))
    alt_run = by_id.get(str(alt_id))
    if base_run is None or alt_run is None:
        return info("Select two successful runs to compare.")
    config = dbc.Accordion(
        [
            dbc.AccordionItem(
                make_table(
                    _config_pair_frame(base_run, alt_run), tid="wb-artifact-config", height="240px"
                ),
                title="Selected run configurations",
            )
        ],
        active_item="item-0",
        start_collapsed=False,
        flush=True,
        className="mb-3",
    )
    if kind == "dt":
        panes = []
        for label, run in (("Base tree", base_run), ("Alternate tree", alt_run)):
            artifact = run.get("artifact") or {}
            model = artifact.get("model") if isinstance(artifact, dict) else None
            X_model = artifact.get("feature_frame") if isinstance(artifact, dict) else None
            text = "No tree text available."
            if model is not None and X_model is not None:
                try:
                    text = export_text(
                        model, feature_names=[str(c) for c in X_model.columns], max_depth=5
                    )
                except Exception:
                    pass
            panes.append(
                dbc.Col(
                    [html.H6(label, className="fw-semibold"), html.Pre(text, className="tree-pre")],
                    md=6,
                )
            )
        return html.Div([config, dbc.Row(panes)])
    if kind == "lr":
        frames = []
        for run in (base_run, alt_run):
            artifact = run.get("artifact") or {}
            model = artifact.get("model") if isinstance(artifact, dict) else None
            X_model = artifact.get("feature_frame") if isinstance(artifact, dict) else None
            frames.append(
                _coefficient_frame(model, X_model) if model is not None else pd.DataFrame()
            )
        base_df, alt_df = frames
        merged = (
            base_df[["feature", "coefficient"]]
            .rename(columns={"coefficient": "base_coefficient"})
            .merge(
                alt_df[["feature", "coefficient"]].rename(
                    columns={"coefficient": "alternate_coefficient"}
                ),
                on="feature",
                how="outer",
            )
            .fillna(0.0)
        )
        merged["delta"] = merged["alternate_coefficient"] - merged["base_coefficient"]
        merged["abs_delta"] = merged["delta"].abs()
        merged = merged.sort_values("abs_delta", ascending=False).drop(columns="abs_delta")
        return html.Div(
            [config, make_table(merged.head(100), tid="wb-artifact-delta", height="360px")]
        )
    if kind == "hugiml":
        base_df, alt_df, key, label = (
            _normalised_pattern_inventory(base_run),
            _normalised_pattern_inventory(alt_run),
            "pattern_key",
            "pattern",
        )
    elif kind == "rpte":
        base_df, alt_df, key, label = (
            _normalised_rpte_rule_inventory(base_run),
            _normalised_rpte_rule_inventory(alt_run),
            "term_key",
            "final_term",
        )
    else:
        base_df, alt_df, key, label = (
            _normalised_rule_inventory(base_run),
            _normalised_rule_inventory(alt_run),
            "rule_key",
            "rule",
        )
    delta, base_count, alt_count = _set_delta(base_df, alt_df, key, label)
    metrics = dbc.Row(
        [
            dbc.Col(mc("Base terms", f"{base_count:,}"), md=4, className="mb-2"),
            dbc.Col(mc("Alternate terms", f"{alt_count:,}"), md=4, className="mb-2"),
            dbc.Col(
                mc(
                    "Changed",
                    f"{int((delta['status'] != 'unchanged').sum()) if not delta.empty else 0:,}",
                ),
                md=4,
                className="mb-2",
            ),
        ]
    )
    return html.Div([config, metrics, make_table(delta, tid="wb-artifact-delta", height="380px")])


def _artifact_panel(runs):
    kinds = _artifact_kind_options(runs)
    if not kinds:
        return info(
            "Run at least two configurations of HUGIML, RPTE, RuleFit, Logistic Regression, or Decision Tree to compare artifacts."
        )
    first_kind = kinds[0]["value"]
    candidates = _artifact_runs(first_kind, runs)
    opts = [{"label": _artifact_run_label(r), "value": str(r.get("run_id"))} for r in candidates]
    base = opts[0]["value"] if opts else None
    alt = opts[1]["value"] if len(opts) > 1 else base
    return html.Div(
        [
            sn(
                "Compare model artifacts across two fitted configurations. Added and removed terms are reported relative to the base run."
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Artifact", className="form-label"),
                            dcc.Dropdown(
                                id="wb-artifact-kind",
                                options=kinds,
                                value=first_kind,
                                clearable=False,
                            ),
                        ],
                        className="setup-field",
                    ),
                    html.Div(
                        [
                            html.Label("Base run", className="form-label"),
                            dcc.Dropdown(
                                id="wb-artifact-base", options=opts, value=base, clearable=False
                            ),
                        ],
                        className="setup-field",
                    ),
                    html.Div(
                        [
                            html.Label("Alternate run", className="form-label"),
                            dcc.Dropdown(
                                id="wb-artifact-alt", options=opts, value=alt, clearable=False
                            ),
                        ],
                        className="setup-field",
                    ),
                ],
                className="artifact-control-grid",
            ),
            html.Div(id="wb-artifact-output", children=_artifact_comparison(first_kind, base, alt)),
        ]
    )


def _model_drilldown_panel(runs):
    best = list(_best_run_per_model(runs).values())
    if not best:
        return info("No successful runs are available for inspection.")
    opts = [
        {
            "label": f"{_model_short(str(r.get('model')))} — {r.get('run_id')}",
            "value": str(r.get("run_id")),
        }
        for r in best
    ]
    selected = opts[0]["value"]
    return html.Div(
        [
            sn(
                "Metrics use stitched out-of-fold predictions. Feature and rule artifacts come from the final refit on the complete dataset."
            ),
            html.Div(
                [
                    html.Label("Model run", className="form-label"),
                    dcc.Dropdown(
                        id="wb-inspect-run", options=opts, value=selected, clearable=False
                    ),
                ],
                className="inspect-selector",
            ),
            html.Div(id="wb-inspect-output", children=_run_detail(best[0], id_prefix="inspect")),
        ]
    )


def _leaderboard_panel(runs):
    leaderboard = _leaderboard_frame(runs)
    ok = leaderboard[leaderboard["status"] == "ok"] if not leaderboard.empty else pd.DataFrame()
    failed = leaderboard[leaderboard["status"] != "ok"] if not leaderboard.empty else pd.DataFrame()
    best = _best_run_per_model(runs)
    header = html.Div()
    if not ok.empty:
        top = ok.iloc[0]
        header = dbc.Row(
            [
                dbc.Col(
                    mc("Best Model", _model_short(str(top.get("model", "N/A")))),
                    md=3,
                    className="mb-2",
                ),
                dbc.Col(mc("Best CV AUC", _fmt(top.get("cv_roc_auc"))), md=3, className="mb-2"),
                dbc.Col(
                    mc("Successful", f"{len(ok):,}/{len(leaderboard):,}"), md=3, className="mb-2"
                ),
                dbc.Col(mc("Models Compared", f"{len(best):,}"), md=3, className="mb-2"),
            ]
        )
    compact_cols = [
        "run_id",
        "category",
        "display",
        "status",
        "best_config",
        "cv_roc_auc",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "fit_time_sec",
    ]
    table = make_table(
        leaderboard[[c for c in compact_cols if c in leaderboard.columns]],
        tid="wb-lb",
        height="280px",
    )
    comparison_row = html.Div()
    if best:
        best_rows = []
        for model_name, run in best.items():
            best_rows.append(
                {
                    "model": _model_short(model_name),
                    "best_run": run.get("run_id"),
                    "candidates": sum(
                        1 for r in runs if r.get("model") == model_name and r.get("status") == "ok"
                    ),
                    "cv_roc_auc": _scalar_metric(run.get("cv_roc_auc")),
                    "params": _summarize_params_for_display(run.get("params", {})),
                }
            )
        chart_df = pd.DataFrame(best_rows).sort_values("cv_roc_auc", ascending=False).head(8)
        colors = [
            "#2563eb" if "HUGIML" in str(label).upper() else "#AFA9EC"
            for label in chart_df["model"]
        ]
        chart = dcc.Graph(
            figure=bar_h(
                chart_df["cv_roc_auc"].tolist(),
                chart_df["model"].tolist(),
                title="Best CV ROC-AUC by model",
                color=colors,
                xlabel="AUC",
                h=230,
            ),
            config={"displayModeBar": False},
            className="compact-leaderboard-chart",
        )
        summary = make_table(pd.DataFrame(best_rows), tid="wb-bpm", height="285px")
        comparison_row = dbc.Row(
            [
                dbc.Col(html.Div(chart, className="results-side-card"), md=5, className="mb-3"),
                dbc.Col(
                    html.Div(
                        [
                            html.H6("Best configuration per model", className="results-card-title"),
                            summary,
                        ],
                        className="results-side-card",
                    ),
                    md=7,
                    className="mb-3",
                ),
            ],
            className="mt-3",
        )
    children = [header, table, comparison_row]
    if not failed.empty:
        children.append(
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        make_table(
                            failed[["run_id", "display", "status", "diagnostic"]],
                            tid="wb-fail",
                            height=None,
                        ),
                        title=f"Failed runs ({len(failed)})",
                    )
                ],
                start_collapsed=True,
                flush=True,
            )
        )
    return html.Div(children)


def _results_panel(ctx):
    runs = _WB.get("runs", [])
    if not runs:
        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            "Results appear after the experiment run.",
                            className="empty-results-title",
                        ),
                        html.Div(
                            "Configure the dataset and models in Setup, then use the single Run selected models action.",
                            className="setup-help",
                        ),
                    ],
                    className="empty-results",
                ),
                _result_stubs(),
            ]
        )
    tabs = dbc.Tabs(
        [
            dbc.Tab(_leaderboard_panel(runs), label="Leaderboard", tab_id="wb-res-leaderboard"),
            dbc.Tab(_curve_panel(runs, ctx), label="Compare curves", tab_id="wb-res-curves"),
            dbc.Tab(
                _interpretability_panel(runs),
                label="Interpretability",
                tab_id="wb-res-interpretability",
            ),
            dbc.Tab(_artifact_panel(runs), label="Artifact comparison", tab_id="wb-res-artifacts"),
            dbc.Tab(
                _model_drilldown_panel(runs), label="Model drill-down", tab_id="wb-res-drilldown"
            ),
        ],
        id="wb-res-tabs",
        active_tab="wb-res-leaderboard",
        className="results-view-menu",
    )
    successful_hugiml = [r for r in _successful_runs(runs) if r.get("model") == "HUGIML"]
    promotion = html.Div()
    if successful_hugiml:
        opts = [
            {"label": f"{r.get('run_id')} — AUC {_fmt(r.get('cv_roc_auc'))}", "value": i}
            for i, r in enumerate(successful_hugiml)
        ]
        promotion = dbc.Accordion(
            [
                dbc.AccordionItem(
                    html.Div(
                        [
                            html.Div(
                                "Transfer the selected fitted HUGIML model and its out-of-fold evaluation evidence to Governance.",
                                className="setup-help mb-2",
                            ),
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="wb-promote-dd",
                                        options=opts,
                                        value=0,
                                        clearable=False,
                                        className="promotion-select",
                                    ),
                                    html.Button(
                                        "Promote to Governance",
                                        id="wb-promote-btn",
                                        n_clicks=0,
                                        className="btn-fit",
                                    ),
                                ],
                                className="promotion-controls",
                            ),
                            html.Div(id="wb-promote-status"),
                        ]
                    ),
                    title="Promote a HUGIML run to Governance",
                )
            ],
            start_collapsed=True,
            flush=True,
            className="mt-3",
        )
    else:
        promotion = html.Div(
            [
                html.Div(id="wb-promote-status", style={"display": "none"}),
                dcc.Dropdown(id="wb-promote-dd", options=[], value=None, style={"display": "none"}),
                html.Button(id="wb-promote-btn", n_clicks=0, style={"display": "none"}),
            ]
        )
    return html.Div([tabs, promotion])


# ─── Top-level render ─────────────────────────────────────────────────────────
def render_setup(ctx):
    """Workbench setup panel — exposed for the main app router."""
    return _setup_panel(ctx)


def render(ctx, section="setup"):
    """Render one Workbench view without adding a second navigation surface."""
    if section == "results":
        return _results_panel(ctx)
    return _setup_panel(ctx)


# ─── Callbacks ─────────────────────────────────────────────────────────────────
def register_callbacks(app):
    from hugiml.dashboard.dash_app import _CTX, _prepare_data_context

    @app.callback(
        Output("wb-auto-box", "style"),
        Output("wb-guided-box", "style"),
        Output("wb-adv-box", "style"),
        Input("wb-hug-mode", "value"),
    )
    def _hug_mode(mode):
        h = {"display": "none"}
        return (
            {} if mode == "auto" else h,
            {} if mode == "guided" else h,
            {} if mode == "advanced" else h,
        )

    @app.callback(Output("wb-grid-info", "children"), Input("wb-grid-name", "value"))
    def _ginfo(gn):
        if not gn:
            return html.Div()
        try:
            g = _hugiml_core_default_grid(gn)
            nc = _hugiml_grid_count(g)
            vbe = _grid_varies_base_estimator(g)
        except Exception:
            return html.Div()
        c = [
            html.Div(
                [
                    html.Span(f"{nc:,} candidates", className="chip"),
                    html.Span(gn, className="chip", style={"marginLeft": "4px"}),
                ],
                style={"marginBottom": "6px"},
            )
        ]
        if vbe:
            c.append(
                html.Small(
                    "Varies base_estimator. Mining cached; only final fits repeat.",
                    className="text-muted d-block",
                )
            )
        return html.Div(c)

    @app.callback(Output("wb-rpte-box", "style"), Input("wb-adv-ds", "value"))
    def _rpte_vis(downstream):
        return {} if "rpte" in _selected_values(downstream, ["lr"]) else {"display": "none"}

    @app.callback(
        Output("wb-hugiml-config-wrap", "style"),
        Output("wb-comparison-config-wrap", "style"),
        Output("wb-param-lr", "style"),
        Output("wb-param-dt", "style"),
        Output("wb-param-rf", "style"),
        Output("wb-param-xgb", "style"),
        Output("wb-param-lgbm", "style"),
        Output("wb-param-ebm", "style"),
        Output("wb-param-rulefit", "style"),
        Output("wb-run-btn", "children"),
        Output("wb-run-btn", "disabled"),
        Input("wb-chk-baselines", "value"),
        Input("wb-chk-ensembles", "value"),
        Input("wb-chk-interpretable_models", "value"),
        Input("wb-hug-mode", "value"),
        Input("wb-grid-name", "value"),
        Input("wb-adv-L", "value"),
        Input("wb-adv-topK", "value"),
        Input("wb-adv-G", "value"),
        Input("wb-adv-fm", "value"),
        Input("wb-adv-strict", "value"),
        Input("wb-adv-aug", "value"),
        Input("wb-adv-relax", "value"),
        Input("wb-adv-ds", "value"),
        Input("wb-rpte-lc", "value"),
        Input("wb-rpte-d", "value"),
        Input("wb-rpte-ne", "value"),
        Input("wb-lr-C", "value"),
        Input("wb-lr-mi", "value"),
        Input("wb-dt-d", "value"),
        Input("wb-dt-l", "value"),
        Input("wb-rf-n", "value"),
        Input("wb-rf-d", "value"),
        Input("wb-rf-l", "value"),
        Input("wb-xgb-n", "value"),
        Input("wb-xgb-d", "value"),
        Input("wb-xgb-lr", "value"),
        Input("wb-lgbm-n", "value"),
        Input("wb-lgbm-d", "value"),
        Input("wb-lgbm-lr", "value"),
        Input("wb-ebm-b", "value"),
        Input("wb-ebm-i", "value"),
        Input("wb-rf2-ts", "value"),
        Input("wb-rf2-mr", "value"),
    )
    def _selected_model_controls(
        baselines,
        ensembles,
        interpretable,
        hug_mode,
        grid_name,
        adv_L,
        adv_topK,
        adv_G,
        adv_fm,
        adv_strict,
        adv_aug,
        adv_relax,
        adv_ds,
        rpte_lc,
        rpte_d,
        rpte_ne,
        lr_C,
        lr_mi,
        dt_d,
        dt_l,
        rf_n,
        rf_d,
        rf_l,
        xgb_n,
        xgb_d,
        xgb_lr,
        lgbm_n,
        lgbm_d,
        lgbm_lr,
        ebm_b,
        ebm_i,
        rf2_ts,
        rf2_mr,
    ):
        selected = set(baselines or []) | set(ensembles or []) | set(interpretable or [])

        def visible(model_name):
            return {} if model_name in selected else {"display": "none"}

        comparison_models = {
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "XGBoost",
            "LightGBM",
            "EBM",
            "RuleFit",
        }
        selected_comparison = selected & comparison_models
        if not selected:
            label = "Select models to run"
            disabled = True
        else:
            try:
                count = _configuration_combination_count(
                    selected,
                    hug_mode,
                    grid_name,
                    adv_L,
                    adv_topK,
                    adv_G,
                    adv_fm,
                    adv_strict,
                    adv_aug,
                    adv_relax,
                    adv_ds,
                    rpte_lc,
                    rpte_d,
                    rpte_ne,
                    lr_C,
                    lr_mi,
                    dt_d,
                    dt_l,
                    rf_n,
                    rf_d,
                    rf_l,
                    xgb_n,
                    xgb_d,
                    xgb_lr,
                    lgbm_n,
                    lgbm_d,
                    lgbm_lr,
                    ebm_b,
                    ebm_i,
                    rf2_ts,
                    rf2_mr,
                )
                label = (
                    "Run 1 model combination" if count == 1 else f"Run {count:,} model combinations"
                )
                disabled = count == 0
            except ValueError:
                label = "Review configuration values"
                disabled = True
        return (
            visible("HUGIML"),
            {} if selected_comparison else {"display": "none"},
            visible("Logistic Regression"),
            visible("Decision Tree"),
            visible("Random Forest"),
            visible("XGBoost"),
            visible("LightGBM"),
            visible("EBM"),
            visible("RuleFit"),
            label,
            disabled,
        )

    @app.callback(
        Output("wb-run-status", "children"),
        Output("st-wb-sec", "data"),
        Output("st-ctx", "data", allow_duplicate=True),
        Input("wb-run-btn", "n_clicks"),
        State("ri-source", "value"),
        State("dd-demo", "value"),
        State("ul-file", "contents"),
        State("ul-file", "filename"),
        State("ul-target", "value"),
        State("ul-id", "value"),
        State("ul-excl", "value"),
        State("ul-sens", "value"),
        State("in-cv", "value"),
        State("in-seed", "value"),
        State("wb-hug-mode", "value"),
        State("wb-grid-name", "value"),
        State("wb-adv-L", "value"),
        State("wb-adv-topK", "value"),
        State("wb-adv-G", "value"),
        State("wb-adv-fm", "value"),
        State("wb-adv-strict", "value"),
        State("wb-adv-aug", "value"),
        State("wb-adv-relax", "value"),
        State("wb-adv-ds", "value"),
        State("wb-rpte-lc", "value"),
        State("wb-rpte-d", "value"),
        State("wb-rpte-ne", "value"),
        State("wb-chk-baselines", "value"),
        State("wb-chk-ensembles", "value"),
        State("wb-chk-interpretable_models", "value"),
        State("wb-lr-C", "value"),
        State("wb-lr-mi", "value"),
        State("wb-dt-d", "value"),
        State("wb-dt-l", "value"),
        State("wb-rf-n", "value"),
        State("wb-rf-d", "value"),
        State("wb-rf-l", "value"),
        State("wb-xgb-n", "value"),
        State("wb-xgb-d", "value"),
        State("wb-xgb-lr", "value"),
        State("wb-lgbm-n", "value"),
        State("wb-lgbm-d", "value"),
        State("wb-lgbm-lr", "value"),
        State("wb-ebm-b", "value"),
        State("wb-ebm-i", "value"),
        State("wb-rf2-ts", "value"),
        State("wb-rf2-mr", "value"),
        prevent_initial_call=True,
    )
    def _wb_run(
        nc,
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
        hug_mode,
        grid_name,
        adv_L,
        adv_topK,
        adv_G,
        adv_fm,
        adv_strict,
        adv_aug,
        adv_relax,
        adv_ds,
        rpte_lc,
        rpte_d,
        rpte_ne,
        chk_base,
        chk_ens,
        chk_interp,
        lr_C,
        lr_mi,
        dt_d,
        dt_l,
        rf_n,
        rf_d,
        rf_l,
        xgb_n,
        xgb_d,
        xgb_lr,
        lgbm_n,
        lgbm_d,
        lgbm_lr,
        ebm_b,
        ebm_i,
        rf2_ts,
        rf2_mr,
    ):
        if not nc:
            return no_update, no_update, no_update
        all_checked = {
            "Baselines": list(chk_base or []),
            "Ensembles": list(chk_ens or []),
            "Interpretable Models": list(chk_interp or []),
        }
        selected = {category: names for category, names in all_checked.items() if names}
        if not selected:
            return warn("Select at least one model."), no_update, no_update
        try:
            context_key, full = _prepare_data_context(
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
            )
        except Exception as exc:
            return warn(str(exc)), no_update, no_update

        X, y = full["X"], full["y"]
        cv = int(full.get("cv", 3))
        random_state = int(full.get("random_state", 2026))
        param_map = {}
        try:
            for category, names in selected.items():
                for model_name in names:
                    if model_name == "HUGIML":
                        if hug_mode == "auto":
                            param_map[model_name] = [_hugiml_auto_params()]
                        elif hug_mode == "guided":
                            param_map[model_name] = [_hugiml_guided_fast_tune_params(grid_name)]
                        else:
                            params = _hugiml_auto_params()
                            params.update(adaptive_binning=True, B=-1)
                            feature_modes = _selected_values(adv_fm, ["original_plus_patterns"])
                            downstream = _selected_values(adv_ds, ["lr"])
                            base_estimators = []
                            if "lr" in downstream:
                                base_estimators.append(None)
                            if "rpte" in downstream:
                                if not _RPTE:
                                    raise ValueError("RPTE is not available in this environment.")
                                leaf_values = _selected_values(
                                    rpte_lc,
                                    ["3xD" if "3xD" in RPTE_LC else next(iter(RPTE_LC), "")],
                                )
                                depth_values = _csv_values(rpte_d, int, 4, "RPTE depth")
                                estimator_values = _csv_values(
                                    rpte_ne, int, 10, "RPTE n_estimators"
                                )
                                for leaf_config, depth, n_estimators in itertools.product(
                                    leaf_values, depth_values, estimator_values
                                ):
                                    if int(depth) < 1 or int(n_estimators) < 1:
                                        raise ValueError(
                                            "RPTE depth and n_estimators must be positive."
                                        )
                                    base_estimators.append(
                                        OneVsRestClassifier(
                                            LeafWiseBoundedLookaheadRPTEFeatureLR(
                                                leaf_config=str(leaf_config),
                                                depth=int(depth),
                                                n_estimators=int(n_estimators),
                                                enable_lookahead="adaptive",
                                            ),
                                            n_jobs=1,
                                        )
                                    )
                            if not base_estimators:
                                base_estimators = [None]
                            grid = {
                                "L": _csv_values(adv_L, int, 2, "L"),
                                "topK": _csv_values(adv_topK, int, 100, "topK"),
                                "G": _csv_values(adv_G, float, 0.01, "G"),
                                "feature_mode": feature_modes,
                                "topk_budget_strict": _csv_values(
                                    adv_strict, bool, False, "Strict budget"
                                ),
                                "augmented_pair_transforms": _csv_values(
                                    adv_aug, bool, True, "Augmented pair"
                                ),
                                "interaction_relaxed_mining": _csv_values(
                                    adv_relax, bool, False, "Relaxed mining"
                                ),
                                "base_estimator": base_estimators,
                            }
                            if any(int(value) < 1 for value in grid["L"]):
                                raise ValueError("L values must be at least 1.")
                            if any(int(value) < 1 for value in grid["topK"]):
                                raise ValueError("topK values must be at least 1.")
                            if any(float(value) < 0 for value in grid["G"]):
                                raise ValueError("G values must be non-negative.")
                            candidates = _expand_candidates(params, grid)
                            candidates, omitted = _filter_hugiml_interaction_configs(candidates)
                            if not candidates:
                                raise ValueError(
                                    "No valid HUGIML candidates remain. For L >= 2, do not enable "
                                    "augmented pairs and relaxed mining in the same candidate."
                                )
                            param_map[model_name] = candidates
                    elif model_name == "Logistic Regression":
                        base = {"C": 1.0, "max_iter": 1000}
                        param_map[model_name] = _expand_candidates(
                            base,
                            {
                                "C": _csv_values(lr_C, float, 1.0, "Logistic C"),
                                "max_iter": _csv_values(lr_mi, int, 1000, "Logistic max_iter"),
                            },
                        )
                    elif model_name == "Decision Tree":
                        depths = _csv_values(dt_d, int, 4, "Decision-tree max_depth")
                        param_map[model_name] = _expand_candidates(
                            {"max_depth": 4, "min_samples_leaf": 1},
                            {
                                "max_depth": [
                                    None if int(value) == 0 else int(value) for value in depths
                                ],
                                "min_samples_leaf": _csv_values(
                                    dt_l, int, 1, "Decision-tree min_samples_leaf"
                                ),
                            },
                        )
                    elif model_name == "Random Forest":
                        depths = _csv_values(rf_d, int, 0, "Random-forest max_depth")
                        param_map[model_name] = _expand_candidates(
                            {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
                            {
                                "n_estimators": _csv_values(
                                    rf_n, int, 200, "Random-forest n_estimators"
                                ),
                                "max_depth": [
                                    None if int(value) == 0 else int(value) for value in depths
                                ],
                                "min_samples_leaf": _csv_values(
                                    rf_l, int, 1, "Random-forest min_samples_leaf"
                                ),
                            },
                        )
                    elif model_name == "XGBoost":
                        param_map[model_name] = _expand_candidates(
                            {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
                            {
                                "n_estimators": _csv_values(
                                    xgb_n, int, 200, "XGBoost n_estimators"
                                ),
                                "max_depth": _csv_values(xgb_d, int, 4, "XGBoost max_depth"),
                                "learning_rate": _csv_values(
                                    xgb_lr, float, 0.05, "XGBoost learning_rate"
                                ),
                            },
                        )
                    elif model_name == "LightGBM":
                        param_map[model_name] = _expand_candidates(
                            {"n_estimators": 200, "max_depth": -1, "learning_rate": 0.05},
                            {
                                "n_estimators": _csv_values(
                                    lgbm_n, int, 200, "LightGBM n_estimators"
                                ),
                                "max_depth": _csv_values(lgbm_d, int, -1, "LightGBM max_depth"),
                                "learning_rate": _csv_values(
                                    lgbm_lr, float, 0.05, "LightGBM learning_rate"
                                ),
                            },
                        )
                    elif model_name == "EBM":
                        param_map[model_name] = _expand_candidates(
                            {"max_bins": 32, "interactions": 5},
                            {
                                "max_bins": _csv_values(ebm_b, int, 32, "EBM max_bins"),
                                "interactions": _csv_values(ebm_i, int, 5, "EBM interactions"),
                            },
                        )
                    elif model_name == "RuleFit":
                        param_map[model_name] = _expand_candidates(
                            {"tree_size": 4, "max_rules": 100},
                            {
                                "tree_size": _csv_values(rf2_ts, int, 4, "RuleFit tree_size"),
                                "max_rules": _csv_values(rf2_mr, int, 100, "RuleFit max_rules"),
                            },
                        )
                    else:
                        param_map[model_name] = [_default_model_params(model_name)]
        except ValueError as exc:
            return warn(str(exc)), no_update, {"ck": context_key}

        try:
            runs = _run(selected, param_map, X, y, cv, random_state)
            _WB["runs"] = runs
            successful = sum(1 for run in runs if run.get("status") == "ok")
            status = html.Div(
                [
                    html.Span("●", className="fit-dot", style={"marginRight": "6px"}),
                    html.Span(
                        f"Completed — {successful}/{len(runs)} candidate runs successful",
                        style={"fontSize": ".78rem", "color": "var(--gr)"},
                    ),
                ],
                className="fit-badge",
            )
            return status, "results", {"ck": context_key}
        except Exception as exc:
            status = html.Div(
                [
                    html.Span("✗ ", style={"color": "var(--rd)"}),
                    html.Span(str(exc)[:240], style={"fontSize": ".76rem"}),
                ]
            )
            return status, no_update, {"ck": context_key}

    @app.callback(
        Output("wb-artifact-base", "options"),
        Output("wb-artifact-base", "value"),
        Output("wb-artifact-alt", "options"),
        Output("wb-artifact-alt", "value"),
        Input("wb-artifact-kind", "value"),
    )
    def _artifact_run_options(kind):
        candidates = _artifact_runs(kind, _WB.get("runs", [])) if kind else []
        options = [
            {"label": _artifact_run_label(run), "value": str(run.get("run_id"))}
            for run in candidates
        ]
        base = options[0]["value"] if options else None
        alternate = options[1]["value"] if len(options) > 1 else base
        return options, base, options, alternate

    @app.callback(
        Output("wb-artifact-output", "children"),
        Input("wb-artifact-kind", "value"),
        Input("wb-artifact-base", "value"),
        Input("wb-artifact-alt", "value"),
    )
    def _artifact_output(kind, base_id, alternate_id):
        if not kind:
            return html.Div()
        return _artifact_comparison(kind, base_id, alternate_id)

    @app.callback(Output("wb-inspect-output", "children"), Input("wb-inspect-run", "value"))
    def _inspect_output(run_id):
        if not run_id:
            return html.Div()
        run = next(
            (
                r
                for r in _best_run_per_model(_WB.get("runs", [])).values()
                if str(r.get("run_id")) == str(run_id)
            ),
            None,
        )
        return _run_detail(run, id_prefix="inspect")

    # PROMOTION — outputs st-ws="governance" to switch workspace
    @app.callback(
        Output("wb-promote-status", "children"),
        Output("st-ctx", "data", allow_duplicate=True),
        Output("st-ws", "data", allow_duplicate=True),
        Input("wb-promote-btn", "n_clicks"),
        State("wb-promote-dd", "value"),
        State("st-ctx", "data"),
        prevent_initial_call=True,
    )
    def _promote(nc, ridx, ctx_ref):
        if not nc or ctx_ref is None:
            return no_update, no_update, no_update
        full = _CTX.get(ctx_ref.get("ck", ""))
        if full is None:
            return warn("No model."), no_update, no_update
        hr = [r for r in _successful_runs(_WB.get("runs", [])) if r.get("model") == "HUGIML"]
        if not hr or ridx is None or int(ridx) >= len(hr):
            return warn("No HUGIML run."), no_update, no_update
        sel = hr[int(ridx)]
        art = sel.get("artifact") or {}
        mo = art.get("model") if isinstance(art, dict) else None
        if mo is None:
            return err("No fitted model in artifact."), no_update, no_update
        sr = art.get("tuning_result") if isinstance(art, dict) else None
        if sr is None:
            sr = type("WBR", (), {})()
            sr.best_estimator_ = mo
            sr.best_params_ = sel.get("params", {})
            sr.best_score_ = sel.get("cv_roc_auc")
            sr.results_ = [{"status": "ok", "score": sel.get("cv_roc_auc")}]
            sr.status_ = "ok"
            sr.error_ = None
        try:
            preds = score_cases(mo, full["X"])
        except Exception:
            preds = pd.DataFrame()
        ev = {}
        if _promoted_evaluation_bundle:
            try:
                ev = _promoted_evaluation_bundle(sel, full.get("y"))
            except Exception:
                pass
        from hugiml.dashboard.dash_components.data_utils import cache_key

        ck = cache_key(
            f"workbench:{sel.get('run_id')}", full.get("cv", 3), full.get("random_state", 2026)
        )
        _CTX[ck] = {
            **full,
            "result": sr,
            "model": mo,
            "predictions": preds,
            "evaluation": ev,
            "promoted_run": {
                "run_id": sel.get("run_id"),
                "params": copy.deepcopy(sel.get("params", {})),
            },
        }
        return (
            html.Div(
                [
                    html.Span("●", className="fit-dot", style={"marginRight": "6px"}),
                    html.Span(
                        f"✓ Promoted {sel.get('run_id')} — switched to Governance",
                        style={"fontSize": ".78rem", "color": "var(--gr)"},
                    ),
                ],
                className="fit-badge",
            ),
            {"ck": ck},
            "governance",
        )

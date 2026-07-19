"""Validation Evidence page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from hugiml.dashboard.components.performance import (
    _diagnostic_inputs,
    normalize_validation_results,
    performance_diagnostic_frames,
    score_separation_frames,
)
from hugiml.dashboard.dash_components.charts import (
    calibration_chart,
    empty_fig,
    roc_pr_curves,
    separation_hist,
)
from hugiml.dashboard.dash_components.pages._shared import info, mc, sn
from hugiml.dashboard.dash_components.tables import make_table


def render(ctx):
    result = ctx.get("result")
    cv_df = pd.DataFrame(getattr(result, "results_", []) if result else [])
    cv_df = normalize_validation_results(cv_df)
    model, X, y, ev = ctx.get("model"), ctx.get("X"), ctx.get("y"), ctx.get("evaluation")
    diag = performance_diagnostic_frames(model, X, y, evaluation=ev)
    diag_ok = not diag["metrics"].empty
    # CV tab
    cv_children = []
    if cv_df.empty:
        cv_children = [info("No CV result rows.")]
    else:
        sc_cols = [
            c
            for c in cv_df.columns
            if pd.api.types.is_numeric_dtype(cv_df[c])
            and ("score" in c.lower() or "auc" in c.lower())
        ]
        sc_col = sc_cols[0] if sc_cols else None
        if sc_col:
            vals = pd.to_numeric(cv_df[sc_col], errors="coerce")
            is_fold = "fold" in cv_df.columns and sc_col in {"validation_roc_auc", "score"}
            s_val = float(vals.mean()) if is_fold else float(vals.max())
            cv_children.append(
                dbc.Row(
                    [
                        dbc.Col(
                            mc("Mean CV ROC-AUC" if is_fold else "Best CV ROC-AUC", f"{s_val:.4f}"),
                            md=4,
                            className="mb-3",
                        ),
                        dbc.Col(mc("Rows", f"{len(cv_df):,}"), md=4, className="mb-3"),
                        dbc.Col(mc("Metric", sc_col), md=4, className="mb-3"),
                    ]
                )
            )
        cv_children.append(make_table(cv_df, tid="pf-cv", height="340px"))
    # Diag tab
    diag_children = []
    if not diag_ok:
        diag_children = [info("Diagnostics require fitted model, X, and binary y.")]
    else:
        mm = dict(zip(diag["metrics"]["metric"], diag["metrics"]["value"]))
        diag_children.append(
            dbc.Row(
                [
                    dbc.Col(
                        mc("ROC-AUC", f"{mm.get('roc_auc', float('nan')):.4f}"),
                        md=3,
                        className="mb-3",
                    ),
                    dbc.Col(
                        mc("Brier", f"{mm.get('brier_score', float('nan')):.4f}"),
                        md=3,
                        className="mb-3",
                    ),
                    dbc.Col(mc("F1", f"{mm.get('f1', float('nan')):.2%}"), md=3, className="mb-3"),
                    dbc.Col(
                        mc("Accuracy", f"{mm.get('accuracy', float('nan')):.2%}"),
                        md=3,
                        className="mb-3",
                    ),
                ]
            )
        )
        yt, pb, _ = _diagnostic_inputs(model, X, y, ev)
        if yt is not None and pb is not None:
            sf = score_separation_frames(yt, pb)
            diag_children.append(
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                figure=separation_hist(pb, yt), config={"displayModeBar": False}
                            ),
                            md=7,
                        ),
                        dbc.Col(
                            dcc.Graph(
                                figure=calibration_chart(
                                    sf["calibration"]["predicted"].tolist(),
                                    sf["calibration"]["actual"].tolist(),
                                    sf["calibration"].get("n", pd.Series()).tolist(),
                                )
                                if not sf["calibration"].empty
                                else empty_fig("No calibration data"),
                                config={"displayModeBar": False},
                            ),
                            md=5,
                        ),
                    ]
                )
            )
        curves = diag.get("curves", pd.DataFrame())
        if not curves.empty:
            diag_children.append(
                dcc.Graph(
                    figure=roc_pr_curves(
                        curves["threshold"].tolist(),
                        curves["recall"].tolist(),
                        curves["precision"].tolist(),
                        curves["specificity"].tolist(),
                    ),
                    config={"displayModeBar": False},
                )
            )
        diag_children += [
            make_table(diag["confusion"], tid="pf-conf", height=None, page_size=5),
            make_table(diag["metrics"], tid="pf-met", height=None),
        ]
    # Threshold tab
    thr = diag["thresholds"]
    thr_child = (
        make_table(thr, tid="pf-thr", height="360px")
        if not thr.empty
        else info("No threshold data.")
    )
    return html.Div(
        [
            sn(
                "CV evidence and one consistent diagnostic evidence set. Diagnostics use stitched OOF predictions when available."
            ),
            dbc.Tabs(
                [
                    dbc.Tab(html.Div(cv_children), label="CV / Tuning", tab_id="pf-cv"),
                    dbc.Tab(html.Div(diag_children), label="Diagnostics", tab_id="pf-diag"),
                    dbc.Tab(thr_child, label="Threshold Sweep", tab_id="pf-thr"),
                ],
                active_tab="pf-cv",
            ),
        ]
    )

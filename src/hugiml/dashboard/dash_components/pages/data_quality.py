"""Data Quality & Policy page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from hugiml.dashboard.components.fairness import (
    predicted_positive_rate_by_group,
    sensitive_pattern_flags,
    sensitive_rpte_rule_flags,
)
from hugiml.dashboard.components.missingness import missingness_frame
from hugiml.dashboard.dash_components.charts import co_miss_bar, group_rates
from hugiml.dashboard.dash_components.pages._shared import info, mc, sn
from hugiml.dashboard.dash_components.tables import make_table


def render(ctx):
    X = ctx.get("X")
    model = ctx.get("model")
    roles = ctx.get("roles", {})
    sens = list(roles.get("sensitive_columns", []))
    # Missingness
    mdf = missingness_frame(X, model=model)
    miss_children = []
    if mdf.empty:
        miss_children = [info("No dataset.")]
    else:
        miss_children += [
            dbc.Row(
                [
                    dbc.Col(mc("Features", f"{len(mdf):,}"), md=3, className="mb-3"),
                    dbc.Col(
                        mc("Mean Miss", f"{mdf['missing_pct'].mean():.2f}%"), md=3, className="mb-3"
                    ),
                    dbc.Col(
                        mc("Max Miss", f"{mdf['missing_pct'].max():.2f}%"), md=3, className="mb-3"
                    ),
                    dbc.Col(
                        mc(
                            "Missing Edges",
                            f"{int(mdf['model_missing_edge'].sum()):,}"
                            if "model_missing_edge" in mdf
                            else "N/A",
                        ),
                        md=3,
                        className="mb-3",
                    ),
                ]
            ),
            make_table(mdf, tid="dq-miss", height="340px"),
        ]
        if X is not None:
            mc_ = [c for c in X.columns if X[c].isna().any()]
            if len(mc_) >= 2:
                try:
                    corr = X[mc_].isna().corr()
                    pairs = []
                    for i, c1 in enumerate(mc_):
                        for c2 in mc_[i + 1 :]:
                            r = float(corr.loc[c1, c2])
                            if abs(r) >= 0.3:
                                pairs.append((f"{c1} ↔ {c2}", r))
                    if pairs:
                        labs, vals = zip(*pairs)
                        miss_children += [
                            html.Hr(),
                            html.H6(
                                f"Co-missingness (|r|≥0.30) — {len(pairs)} pairs",
                                className="fw-semibold mb-2",
                            ),
                            dcc.Graph(
                                figure=co_miss_bar(list(labs), list(vals)),
                                config={"displayModeBar": False},
                            ),
                        ]
                except Exception:
                    pass
    # Fairness
    if not sens:
        fair_content = info("No sensitive/proxy columns configured.")
    else:
        base = pd.DataFrame(
            {
                "column": sens,
                "role": "Sensitive/proxy review",
                "action": "Review usage in features, patterns, and group rates",
            }
        )
        rates = predicted_positive_rate_by_group(model, X, sens)
        rate_chart = html.Div()
        if not rates.empty and "predicted_positive_rate" in rates.columns:
            mr = float(rates["predicted_positive_rate"].max())
            rd = rates.sort_values("predicted_positive_rate", ascending=True)
            labs = (rd["column"].astype(str) + "=" + rd["group_value"].astype(str)).tolist()
            rate_chart = html.Div(
                [
                    dcc.Graph(
                        figure=group_rates(labs, rd["predicted_positive_rate"].tolist(), mr),
                        config={"displayModeBar": False},
                    )
                ]
            )
        pf = sensitive_pattern_flags(model, sens)
        rf = sensitive_rpte_rule_flags(model, sens)
        fair_content = dbc.Tabs(
            [
                dbc.Tab(
                    make_table(base, tid="dq-cols", height=None),
                    label="Column Review",
                    tab_id="dq-columns",
                ),
                dbc.Tab(
                    html.Div([rate_chart, make_table(rates, tid="dq-rates", height="260px")])
                    if not rates.empty
                    else info("Rates require model+X+sensitive cols."),
                    label="Group Rates",
                    tab_id="dq-rates-tab",
                ),
                dbc.Tab(
                    make_table(pf, tid="dq-pf", height="260px")
                    if not pf.empty
                    else info("No sensitive pattern flags."),
                    label="Pattern Flags",
                    tab_id="dq-pattern-flags",
                ),
                dbc.Tab(
                    make_table(rf, tid="dq-rf", height="260px")
                    if not rf.empty
                    else info("No sensitive RPTE flags."),
                    label="RPTE Flags",
                    tab_id="dq-rpte-flags",
                ),
            ],
            id="dq-policy-tabs",
            active_tab="dq-columns",
        )
    return html.Div(
        [
            sn(
                "Missingness and sensitive/proxy review. Sensitive columns are surfaced for review, not removed."
            ),
            dbc.Tabs(
                [
                    dbc.Tab(html.Div(miss_children), label="Missingness", tab_id="dq-missingness"),
                    dbc.Tab(
                        fair_content, label="Sensitive / Proxy Review", tab_id="dq-sensitive-review"
                    ),
                ],
                id="dq-main-tabs",
                active_tab="dq-missingness",
            ),
        ]
    )

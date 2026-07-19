"""Monitoring page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dcc, html

from hugiml.dashboard.components.drift import (
    _cv_drift_frame,
    _cv_metadata_frame,
    _cv_scores_frame,
    _distribution_review,
    _get_rpte_feature_flow_audit,
    _missingness_summary,
    _pattern_activation_summary,
    _prediction_summary,
    _probability_band_table,
    _safe_cv_monitoring_report,
)
from hugiml.dashboard.dash_components.charts import bar_h, bar_v, line_chart, psi_bar
from hugiml.dashboard.dash_components.pages._shared import info, mc, sn
from hugiml.dashboard.dash_components.tables import make_table


def render(ctx):
    model = ctx.get("model")
    X = ctx.get("X")
    y = ctx.get("y")
    cv = int(ctx.get("cv", 5))
    ps, p = _prediction_summary(model, X)
    bands = _probability_band_table(p)
    miss = _missingness_summary(X)
    dist = _distribution_review(X)
    pat_sum = _pattern_activation_summary(model, X)
    rfl = _get_rpte_feature_flow_audit(model)
    rhl = rfl.get("final_representation") in {"rpte_leaf_rules", "rpte_leaves_plus_direct_terms"}
    # Header metrics
    mp = (
        float(ps.loc[ps["metric"] == "mean_probability", "value"].iloc[0])
        if not ps.empty and ps["metric"].eq("mean_probability").any()
        else float("nan")
    )
    hs = (
        float(ps.loc[ps["metric"] == "share_ge_0_80", "value"].iloc[0])
        if not ps.empty and ps["metric"].eq("share_ge_0_80").any()
        else float("nan")
    )
    mm = float(miss["missing_pct"].max()) if not miss.empty else float("nan")
    hdr = dbc.Row(
        [
            dbc.Col(mc("Cases", f"{len(X):,}" if X is not None else "N/A"), md=3, className="mb-3"),
            dbc.Col(
                mc("Mean Prob", f"{mp:.4f}" if np.isfinite(mp) else "N/A"), md=3, className="mb-3"
            ),
            dbc.Col(
                mc("Share ≥0.80", f"{hs:.2%}" if np.isfinite(hs) else "N/A"), md=3, className="mb-3"
            ),
            dbc.Col(
                mc("Max Miss", f"{mm:.2f}%" if np.isfinite(mm) else "N/A"), md=3, className="mb-3"
            ),
        ]
    )
    # Pred tab
    pt = []
    if ps.empty:
        pt = [info("No predictions.")]
    else:
        pt = [make_table(ps, tid="mo-ps", height=None)]
        if not bands.empty:
            pt += [
                dcc.Graph(
                    figure=bar_v(
                        bands["probability_band"].tolist(),
                        bands["n_cases"].tolist(),
                        title="Probability bands",
                        color="#534AB7",
                        h=220,
                    ),
                    config={"displayModeBar": False},
                )
            ]
    # Dist tab
    dt = []
    if dist.empty:
        dt = [info("Distribution review unavailable.")]
    else:
        nd = dist.loc[
            dist["review_type"].astype(str).str.contains("numeric", case=False, na=False)
        ].copy()
        if not nd.empty and "value" in nd:
            pdf = nd[["feature", "value"]].copy()
            pdf["value"] = pd.to_numeric(pdf["value"], errors="coerce")
            pdf = pdf.dropna(subset=["value"]).sort_values("value", ascending=False)
            if not pdf.empty:
                dt += [
                    dcc.Graph(
                        figure=psi_bar(pdf["feature"].tolist(), pdf["value"].tolist()),
                        config={"displayModeBar": False},
                    )
                ]
        dt.append(make_table(dist, tid="mo-dist", height="360px"))
    # Pattern tab — resolve pattern_N → actual HUG conjunctions
    pat_tab = []
    if pat_sum.empty:
        pat_tab = [info("Pattern activation not exposed. See Pattern Inventory page.")]
    else:
        try:
            from hugiml.dashboard.components.patterns import _patterns_to_frame

            pf_inv = _patterns_to_frame(model)
            if not pf_inv.empty and "pattern" in pf_inv.columns:

                def _res(p):
                    try:
                        n = int(str(p).replace("pattern_", "")) - 1
                        return str(pf_inv.iloc[n]["pattern"]) if 0 <= n < len(pf_inv) else str(p)
                    except Exception:
                        return str(p)

                pat_sum = pat_sum.copy()
                pat_sum["conjunction"] = pat_sum["pattern"].apply(_res)
                bar_labels = pat_sum["conjunction"].tolist()
            else:
                bar_labels = pat_sum["pattern"].astype(str).tolist()
                pat_sum["conjunction"] = pat_sum["pattern"]
        except Exception:
            bar_labels = pat_sum["pattern"].astype(str).tolist()
            pat_sum["conjunction"] = pat_sum["pattern"]
        title_lbl = "HUG Pattern Activation" if not rhl else "HUG Construction-Input Activation"
        pat_tab = [
            html.Small(
                "Each bar = fraction of training rows that activate the pattern. Hover for exact value.",
                className="text-muted d-block mb-2",
            ),
            dcc.Graph(
                figure=bar_h(
                    pat_sum["activation_rate"].tolist(),
                    bar_labels,
                    title=title_lbl,
                    xlabel="Activation rate",
                    color="#534AB7",
                    h=max(200, len(pat_sum) * 26 + 60),
                ),
                config={"displayModeBar": False},
            ),
            make_table(
                pat_sum[
                    [
                        c
                        for c in ["rank", "pattern", "conjunction", "activation_rate", "n_active"]
                        if c in pat_sum.columns
                    ]
                ],
                tid="mo-pat",
                height="280px",
            ),
        ]
    # CV monitoring
    rep, error = _safe_cv_monitoring_report(model, X, y, cv=cv)
    cv_tab = []
    if error:
        cv_tab = [info(error)]
    elif isinstance(rep, dict):
        sc = _cv_scores_frame(rep)
        dr = _cv_drift_frame(rep)
        md = _cv_metadata_frame(rep)
        if not sc.empty and "auc" in sc and "fold" in sc:
            cv_tab += [
                dcc.Graph(
                    figure=bar_v(
                        sc["fold"].tolist(),
                        pd.to_numeric(sc["auc"], errors="coerce").fillna(0).tolist(),
                        title="Fold AUC",
                        xlabel="Fold",
                        ylabel="AUC",
                        h=220,
                    ),
                    config={"displayModeBar": False},
                ),
                make_table(sc, tid="mo-cv-sc", height=None),
            ]
        if not dr.empty:
            try:
                ht = dr.pivot_table(index="feature", columns="fold", values="psi", aggfunc="mean")
                cv_tab.append(make_table(ht.reset_index(), tid="mo-cv-psi", height="260px"))
            except Exception:
                pass
        pc = next((c for c in ("pattern_count", "n_patterns", "patterns") if c in md.columns), None)
        if pc and not md.empty:
            cv_tab += [
                dcc.Graph(
                    figure=line_chart(md, "fold", [pc], title="Pattern stability"),
                    config={"displayModeBar": False},
                )
            ]
    else:
        cv_tab = [info("CV monitoring unavailable.")]
    pl = "HUG Input-Pattern Activation" if rhl else "Pattern Activation"
    return html.Div(
        [
            sn("Transparent monitoring evidence. No drift APIs called without external baseline."),
            hdr,
            dbc.Tabs(
                [
                    dbc.Tab(html.Div(pt), label="Prediction Distribution", tab_id="mon-pred"),
                    dbc.Tab(
                        make_table(miss, tid="mo-miss", height="360px"),
                        label="Missingness",
                        tab_id="mon-miss",
                    ),
                    dbc.Tab(html.Div(dt), label="Distribution Review", tab_id="mon-dist"),
                    dbc.Tab(html.Div(pat_tab), label=pl, tab_id="mon-pat"),
                    dbc.Tab(html.Div(cv_tab), label="CV Monitoring", tab_id="mon-cv"),
                ],
                active_tab="mon-pred",
            ),
        ]
    )

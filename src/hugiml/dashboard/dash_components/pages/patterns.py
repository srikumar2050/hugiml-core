"""Pattern Inventory page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dcc, html

from hugiml.dashboard.components.patterns import (
    _get_rpte_feature_flow_audit,
    _get_rpte_final_term_rows,
    _pattern_support_distribution,
    _patterns_to_frame,
    _rpte_flat_tree_sections,
    _rpte_rules_to_frame,
    population_coverage_frame,
)
from hugiml.dashboard.dash_components.charts import bar_v, coverage_line
from hugiml.dashboard.dash_components.pages._shared import info, mc
from hugiml.dashboard.dash_components.tables import make_table


def _support_section(model, pdf):
    sd = _pattern_support_distribution(pdf)
    cd = population_coverage_frame(model)
    c = []
    if not sd.empty:
        c += [
            dbc.Row(
                [
                    dbc.Col(
                        mc("Median Support", f"{sd['support'].median():.4f}"),
                        md=4,
                        className="mb-3",
                    ),
                    dbc.Col(
                        mc("Low Support (<0.05)", f"{(sd['support'] < 0.05).mean():.2%}"),
                        md=4,
                        className="mb-3",
                    ),
                ]
            )
        ]
        bins = np.linspace(0, 1, 21)
        counts, edges = np.histogram(sd["support"].clip(0, 1), bins=bins)
        labels = [f"[{edges[i]:.2f},{edges[i + 1]:.2f})" for i in range(len(counts))]
        c.append(
            dcc.Graph(
                figure=bar_v(
                    labels,
                    counts.tolist(),
                    title="Support distribution",
                    xlabel="Bin",
                    ylabel="Count",
                    color="#534AB7",
                    h=220,
                ),
                config={"displayModeBar": False},
            )
        )
    if not cd.empty:
        c.append(make_table(cd, tid="pat-cov", height=None))
        mx = getattr(model, "x_train_hup_", None)
        if mx is not None:
            try:
                arr = mx.toarray() if hasattr(mx, "toarray") else np.asarray(mx)
                if arr.ndim == 2 and arr.shape[1] > 1:
                    covered = np.zeros(arr.shape[0], dtype=bool)
                    cum = []
                    for j in range(arr.shape[1]):
                        covered |= arr[:, j].astype(bool)
                        cum.append(float(covered.mean()))
                    c.append(
                        dcc.Graph(
                            figure=coverage_line(list(range(1, len(cum) + 1)), cum),
                            config={"displayModeBar": False},
                        )
                    )
            except Exception:
                pass
    return html.Div(c) if c else info("No support/coverage data.")


def _tree_cards(rows, frame=None):
    secs = _rpte_flat_tree_sections(rows, frame)
    if not secs:
        return info("No RPTE leaf paths.")
    items = []
    for i, s in enumerate(secs):
        body = [
            html.Pre(
                s["text"],
                style={
                    "fontFamily": "'JetBrains Mono',monospace",
                    "fontSize": ".78rem",
                    "background": "var(--cb)",
                    "borderRadius": "6px",
                    "padding": "10px",
                    "overflowX": "auto",
                    "whiteSpace": "pre-wrap",
                    "maxHeight": "300px",
                    "overflowY": "auto",
                },
            )
        ]
        cdf = s.get("coefficients", pd.DataFrame())
        if not cdf.empty:
            body.append(
                make_table(
                    cdf.drop(columns=["_source_row_index"], errors="ignore"),
                    tid=f"rpte-tc-{i}",
                    height=None,
                )
            )
        items.append(
            dbc.AccordionItem(
                html.Div(body),
                title=f"{s['title']} | {s['leaf_count']} leaves | depth {s['max_depth']}",
            )
        )
    return dbc.Accordion(items, start_collapsed=len(items) > 1, flush=True)


def render(ctx):
    model = ctx.get("model")
    if model is None:
        return info("Train a model first.")
    pdf = _patterns_to_frame(model)
    fr = _get_rpte_final_term_rows(model, include_zero_direct=True)
    au = _get_rpte_feature_flow_audit(model)
    children = []
    if fr:
        df = _rpte_rules_to_frame(fr) if fr else pd.DataFrame()
        ldf = (
            df.loc[df["is_leaf_term"]].copy()
            if not df.empty and "is_leaf_term" in df.columns
            else pd.DataFrame()
        )
        ddf = (
            df.loc[
                df.get("is_direct_source_term", pd.Series(False, index=df.index)).astype(bool)
            ].copy()
            if not df.empty
            else pd.DataFrame()
        )
        stmt = au.get("statement")
        children += [html.H5("RPTE Prediction Representation", className="fw-bold mb-2")]
        if stmt:
            children.append(dbc.Alert(str(stmt), color="info", className="py-2 small"))
        children.append(
            dbc.Row(
                [
                    dbc.Col(
                        mc(
                            "Final LR Terms",
                            f"{int(au.get('final_term_count', len(df)) or len(df)):,}",
                        ),
                        md=2,
                        className="mb-3",
                    ),
                    dbc.Col(mc("Leaf Terms", f"{len(ldf):,}"), md=2, className="mb-3"),
                    dbc.Col(
                        mc("Source Inputs", f"{int(au.get('source_feature_count', 0) or 0):,}"),
                        md=2,
                        className="mb-3",
                    ),
                    dbc.Col(
                        mc(
                            "Trees",
                            f"{ldf['tree'].nunique()}"
                            if not ldf.empty and ldf["tree"].notna().any()
                            else "N/A",
                        ),
                        md=2,
                        className="mb-3",
                    ),
                    dbc.Col(mc("Direct Terms", f"{len(ddf):,}"), md=2, className="mb-3"),
                ]
            )
        )
        if not ldf.empty:
            lc = [
                "class",
                "tree",
                "leaf",
                "effect",
                "coefficient",
                "odds_multiplier",
                "support_rate",
                "n_conditions",
                "raw_sources",
                "backend",
            ]
            children += [
                html.H6("Leaf Trees", className="fw-semibold mb-2"),
                _tree_cards(fr, ldf),
                html.H6("Leaf Coefficient Table", className="fw-semibold mb-2 mt-2"),
                make_table(ldf[[c for c in lc if c in ldf.columns]], tid="rpte-lt", height="300px"),
            ]
        if not ddf.empty:
            dc = [
                "class",
                "term_type",
                "source_display_name",
                "source_column",
                "raw_sources",
                "effect",
                "coefficient",
                "odds_multiplier",
                "backend",
            ]
            children += [
                html.H6("Direct Source Terms", className="fw-semibold mb-2 mt-3"),
                make_table(ddf[[c for c in dc if c in ddf.columns]], tid="rpte-dt", height="260px"),
            ]
        children.append(html.Hr())
    if not pdf.empty:
        children += [
            html.H5("HUG Pattern Inventory", className="fw-bold mb-2"),
            dbc.Row([dbc.Col(mc("Patterns", f"{len(pdf):,}"), md=3, className="mb-3")]),
            make_table(pdf, tid="pat-inv", height="400px", filterable=True),
            html.Hr(),
            _support_section(model, pdf),
        ]
    elif not fr:
        children.append(info("Pattern info not available."))
    return html.Div(children)

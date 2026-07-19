"""Representation Audit page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from hugiml.dashboard.components.complexity import _base_estimator_structure
from hugiml.dashboard.components.feature_family import (
    augmented_feature_audit,
    feature_family_summary,
    original_feature_audit,
    pattern_feature_audit,
)
from hugiml.dashboard.components.governance_evidence import (
    _available_profile_features,
    adaptive_binning_table,
    augmented_pair_effects_frame,
    feature_shape_frame,
    plot_feature_shape,
    rpte_rule_evidence_frame,
    survivor_led_patterns_frame,
)
from hugiml.dashboard.components.patterns import (
    _get_rpte_feature_flow_audit,
    _get_rpte_final_term_rows,
    _get_rpte_rule_rows,
    _rpte_rules_to_frame,
)
from hugiml.dashboard.components.rpte_governance import (
    rpte_direct_source_terms_frame,
    rpte_has_tree_representation,
    rpte_is_active,
    rpte_raw_input_lineage_frame,
    rpte_representation_flow_frame,
    rpte_split_usage_frame,
)
from hugiml.dashboard.dash_components.charts import (
    FAMILY_COLOURS,
    coef_waterfall,
    fig_to_uri,
)
from hugiml.dashboard.dash_components.pages._shared import info, mc, sn
from hugiml.dashboard.dash_components.tables import make_table

try:
    from hugiml.compute_complexity import get_complexity, get_complexity_report

    _HC = True
except Exception:
    _HC = False


def _complexity(model):
    if model is None:
        return info("No model.")
    cfg = {
        k: getattr(model, k, None) for k in ("L", "topK", "G", "feature_mode", "topk_budget_strict")
    }
    be = getattr(model, "base_estimator", None)
    cfg["downstream"] = "RPTE" if be else "LR (built-in)"
    mu = au = cr = None
    if _HC:
        try:
            mu = get_complexity(model, "model units")
            au = get_complexity(model)
            cr = get_complexity_report(model)
        except Exception:
            pass
    met = dbc.Row(
        [
            dbc.Col(mc("Model Units", mu), md=3, className="mb-3"),
            dbc.Col(mc("Audit Units", au), md=3, className="mb-3"),
            dbc.Col(mc("L", cfg.get("L")), md=3, className="mb-3"),
            dbc.Col(mc("topK", cfg.get("topK")), md=3, className="mb-3"),
        ]
    )
    c = [met]
    if cr:
        c += [
            html.H6("Complexity Report", className="fw-semibold mb-2"),
            html.Pre(
                str(cr),
                style={
                    "fontSize": ".76rem",
                    "maxHeight": "200px",
                    "overflowY": "auto",
                    "background": "var(--cb)",
                    "borderRadius": "6px",
                    "padding": "8px",
                },
            ),
        ]
    c += [
        html.H6("Configuration", className="fw-semibold mb-2"),
        html.Pre(
            str(cfg),
            style={
                "fontSize": ".76rem",
                "background": "var(--cb)",
                "borderRadius": "6px",
                "padding": "8px",
            },
        ),
    ]
    rs = _base_estimator_structure(model)
    if rs:
        c += [
            html.H6("RPTE Complexity", className="fw-semibold mb-2"),
            html.Pre(
                str(rs),
                style={
                    "fontSize": ".76rem",
                    "background": "var(--cb)",
                    "borderRadius": "6px",
                    "padding": "8px",
                },
            ),
        ]
    return html.Div(c)


def _family_audit(model, X, sens, excl, idc):
    su = feature_family_summary(model, X)
    orig = original_feature_audit(model, X, sens, excl, idc)
    pats = pattern_feature_audit(model, sens)
    augs = augmented_feature_audit(model, sens)
    rpte = _rpte_rules_to_frame(_get_rpte_rule_rows(model))
    # Waterfall
    wf = []
    try:
        fi = (
            model.feature_importances()
            if callable(getattr(model, "feature_importances", None))
            else None
        )
    except Exception:
        fi = None
    rpte_fb = False
    if (fi is None or fi.empty) and hasattr(model, "rpte_rule_table"):
        try:
            rr = model.rpte_rule_table()
            if rr:
                rpte_fb = True
                ranked = sorted(
                    rr, key=lambda r: abs(r.get("final_logistic_coefficient") or 0), reverse=True
                )[:25]
                fi = pd.DataFrame(
                    [
                        {
                            "pattern": " AND ".join(
                                str(c.get("raw_condition") or "?")
                                for c in (r.get("conditions") or [])
                            )
                            or "(linear)",
                            "coefficient": r.get("final_logistic_coefficient"),
                            "feature_type": "rpte_rule",
                        }
                        for r in ranked
                    ]
                )
        except Exception:
            pass
    if fi is not None and not fi.empty and "coefficient" in fi:
        w = fi.copy()
        w["coefficient"] = pd.to_numeric(w["coefficient"], errors="coerce")
        w = w.dropna(subset=["coefficient"]).head(25).sort_values("coefficient")
        nc = "pattern" if "pattern" in w else ("display_name" if "display_name" in w else "feature")
        fc = "feature_type" if "feature_type" in w else None
        cs = [
            FAMILY_COLOURS.get(
                str(w[fc].iloc[i]).lower() if fc else "unknown", FAMILY_COLOURS["unknown"]
            )
            for i in range(len(w))
        ]
        wf = [
            dcc.Graph(
                figure=coef_waterfall(
                    w[nc].astype(str).tolist(),
                    w["coefficient"].tolist(),
                    cs,
                    title="RPTE Rule Coefficients" if rpte_fb else "Feature Coefficients",
                ),
                config={"displayModeBar": False},
            )
        ]
    tabs = [
        dbc.Tab(
            make_table(orig, tid="r-orig", height="360px"), label="Originals", tab_id="fa-orig"
        ),
        dbc.Tab(
            make_table(pats, tid="r-pats", height="360px")
            if not pats.empty
            else info("No patterns."),
            label="HUG Patterns",
            tab_id="fa-pats",
        ),
        dbc.Tab(
            make_table(augs, tid="r-augs", height="360px")
            if not augs.empty
            else info("No augmented."),
            label="Augmented",
            tab_id="fa-aug",
        ),
    ]
    if not rpte.empty:
        rc = [
            "rule_id",
            "effect",
            "coefficient",
            "odds_multiplier",
            "support_rate",
            "n_conditions",
            "rule_preview",
            "raw_sources",
            "class",
            "tree",
            "leaf",
            "backend",
        ]
        tabs.insert(
            0,
            dbc.Tab(
                make_table(
                    rpte[[c for c in rc if c in rpte.columns]], tid="r-rpte", height="360px"
                ),
                label="RPTE LR Terms",
                tab_id="fa-rpte",
            ),
        )
    return html.Div(
        [make_table(su, tid="r-su", height=None)]
        + wf
        + [dbc.Tabs(tabs, id="r-fam-tabs", active_tab=tabs[0].tab_id if tabs else "fa-orig")]
    )


def _binning(model, X):
    if not getattr(model, "adaptive_binning", False):
        return info("Adaptive binning off.")
    df = adaptive_binning_table(model)
    feats = _available_profile_features(model) if model else []
    c = [
        make_table(df, tid="r-bin", height="340px")
        if not df.empty
        else info("No binning metadata.")
    ]
    if feats:
        c += [
            html.H6("Feature Effect Profile", className="fw-semibold mb-2 mt-3"),
            dcc.Dropdown(
                id="repr-profile-dd",
                options=[{"label": f, "value": f} for f in feats],
                value=feats[0],
                clearable=False,
                style={"maxWidth": "360px", "marginBottom": "12px"},
            ),
            html.Div(id="repr-profile-out"),
        ]
    return html.Div(c)


def _survivor(model):
    df = survivor_led_patterns_frame(model)
    return (
        make_table(df, tid="r-surv", height="340px")
        if not df.empty
        else info("No survivor-led patterns.")
    )


def _aug_pair(model):
    df = augmented_pair_effects_frame(model)
    if df.empty:
        return info("No augmented pair effects.")
    pref = [
        "feature",
        "raw_formula",
        "formula",
        "coefficient_raw_scale",
        "risk_increases_when",
        "eligible_rate",
        "missing_pair_rate",
        "large_raw_effect",
        "raw_interpretation",
    ]
    cols = [c for c in pref if c in df.columns] + [c for c in df.columns if c not in pref]
    return make_table(df[cols], tid="r-aug-pair", height="360px")


def _rpte_rule(model):
    fl = _get_rpte_feature_flow_audit(model)
    df = rpte_rule_evidence_frame(model)
    if df.empty:
        return info("No RPTE evidence.")
    ldf = df.loc[df.get("is_leaf_term", False)].copy() if "is_leaf_term" in df else pd.DataFrame()
    ddf = (
        df.loc[
            df.get("is_direct_source_term", pd.Series(False, index=df.index)).astype(bool)
        ].copy()
        if not df.empty
        else pd.DataFrame()
    )
    met = dbc.Row(
        [
            dbc.Col(
                mc("LR Terms", f"{int(fl.get('final_term_count', len(df)) or len(df)):,}"),
                md=2,
                className="mb-3",
            ),
            dbc.Col(mc("Leaf", f"{len(ldf):,}"), md=2, className="mb-3"),
            dbc.Col(
                mc(
                    "D.Orig",
                    f"{int(ddf['source_family'].eq('original').sum()) if not ddf.empty else 0}",
                ),
                md=2,
                className="mb-3",
            ),
            dbc.Col(
                mc(
                    "D.Pattern",
                    f"{int(ddf['source_family'].eq('pattern').sum()) if not ddf.empty else 0}",
                ),
                md=2,
                className="mb-3",
            ),
            dbc.Col(
                mc(
                    "D.Aug",
                    f"{int(ddf['source_family'].eq('augmented_pair').sum()) if not ddf.empty else 0}",
                ),
                md=2,
                className="mb-3",
            ),
        ]
    )
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
    c = [
        met,
        make_table(ldf[[c for c in lc if c in ldf.columns]], tid="r-rpte-leaf", height="280px")
        if not ldf.empty
        else html.Div(),
    ]
    if not ddf.empty:
        dc = [
            "class",
            "source_display_name",
            "source_column",
            "raw_sources",
            "effect",
            "coefficient",
            "odds_multiplier",
            "backend",
        ]
        c += [
            html.H6("Direct Source Terms", className="fw-semibold mb-2 mt-2"),
            make_table(
                ddf[[c_ for c_ in dc if c_ in ddf.columns]], tid="r-rpte-dir", height="260px"
            ),
        ]
    return html.Div(c)


def _rpte_flow(model, X):
    fl = _get_rpte_feature_flow_audit(model)
    ff = rpte_representation_flow_frame(model, X)
    dr = rpte_direct_source_terms_frame(model, include_zero=True)
    su = rpte_split_usage_frame(model)
    rl = rpte_raw_input_lineage_frame(model, X)
    fr = _get_rpte_final_term_rows(model, include_zero_direct=True)
    tf = _rpte_rules_to_frame(fr) if fr else pd.DataFrame()
    stmt = fl.get("statement")
    c = []
    if stmt:
        c.append(dbc.Alert(str(stmt), color="info", className="py-2 small"))
    lc = int(fl.get("leaf_rule_count", 0) or 0)
    fc = dr["family"].value_counts().to_dict() if not dr.empty and "family" in dr.columns else {}
    c.append(
        dbc.Row(
            [
                dbc.Col(mc("Leaf Terms", f"{lc:,}"), md=2, className="mb-3"),
                dbc.Col(mc("D.Orig", f"{int(fc.get('original', 0)):,}"), md=2, className="mb-3"),
                dbc.Col(mc("D.Pattern", f"{int(fc.get('pattern', 0)):,}"), md=2, className="mb-3"),
                dbc.Col(
                    mc("D.Aug", f"{int(fc.get('augmented_pair', 0)):,}"), md=2, className="mb-3"
                ),
                dbc.Col(
                    mc("Final LR", f"{int(fl.get('final_term_count', len(tf)) or len(tf)):,}"),
                    md=2,
                    className="mb-3",
                ),
            ]
        )
    )
    c += [
        html.H6("Representation Flow", className="fw-semibold mb-2"),
        make_table(ff, tid="r-flow", height=None) if not ff.empty else html.Div(),
        html.H6("Split Usage", className="fw-semibold mb-2 mt-3"),
        make_table(su, tid="r-split", height="260px") if not su.empty else info("No split usage."),
        html.H6("Raw-Input Lineage", className="fw-semibold mb-2 mt-3"),
        make_table(rl, tid="r-lin", height="260px") if not rl.empty else info("No lineage."),
    ]
    return html.Div(c)


def profile_chart(model, X, feat):
    df = feature_shape_frame(model, feat, X)
    if df.empty:
        return info(f"No profile for '{feat}'.")
    fig = plot_feature_shape(model, X, feat)
    c = []
    if fig is not None:
        c.append(html.Img(src=fig_to_uri(fig), style={"maxWidth": "100%", "borderRadius": "8px"}))
        import matplotlib.pyplot as plt

        plt.close(fig)
    c.append(make_table(df, tid="r-prof", height="280px"))
    return html.Div(c)


def render(ctx):
    model = ctx.get("model")
    X = ctx.get("X")
    roles = ctx.get("roles", {})
    sens = list(roles.get("sensitive_columns", []))
    excl = list(roles.get("excluded_columns", []))
    idc = roles.get("id_column")
    is_rpte = rpte_is_active(model) if model else False
    is_leaf = rpte_has_tree_representation(model) if model else False
    is_fb = is_rpte and not is_leaf
    if is_leaf:
        title = "RPTE Representation Audit"
        note = "Audits raw inputs → source columns → RPTE splits → leaf indicators → direct source → final LR."
        tabs = [
            dbc.Tab(_rpte_flow(model, X), label="Feature-Flow Audit", tab_id="r-flow"),
            dbc.Tab(_rpte_rule(model), label="RPTE LR Evidence", tab_id="r-rule"),
            dbc.Tab(_binning(model, X), label="Preprocessing", tab_id="r-bin"),
            dbc.Tab(_survivor(model), label="HUG Patterns", tab_id="r-surv"),
            dbc.Tab(_aug_pair(model), label="Augmented Pairs", tab_id="r-aug"),
        ]
    elif is_fb:
        title = "RPTE Fallback Audit"
        note = "RPTE formed no valid tree; fallback uses HUGIML source columns directly."
        tabs = [
            dbc.Tab(
                _family_audit(model, X, sens, excl, idc), label="Fallback LR Terms", tab_id="r-fam"
            ),
            dbc.Tab(_binning(model, X), label="Preprocessing", tab_id="r-bin"),
            dbc.Tab(_survivor(model), label="Pattern Inputs", tab_id="r-surv"),
            dbc.Tab(_aug_pair(model), label="Aug. Pair Inputs", tab_id="r-aug"),
            dbc.Tab(_rpte_rule(model), label="RPTE Evidence", tab_id="r-rule"),
        ]
    else:
        title = "Representation Audit"
        note = "Complexity, feature-family provenance, and generated-feature governance evidence."
        tabs = [
            dbc.Tab(
                _family_audit(model, X, sens, excl, idc), label="Feature Families", tab_id="r-fam"
            ),
            dbc.Tab(_binning(model, X), label="Adaptive Binning", tab_id="r-bin"),
            dbc.Tab(_aug_pair(model), label="Augmented Pairs", tab_id="r-aug"),
            dbc.Tab(_survivor(model), label="Survivor-Led Patterns", tab_id="r-surv"),
            dbc.Tab(_rpte_rule(model), label="RPTE Evidence", tab_id="r-rule"),
        ]
    return html.Div(
        [
            sn(note),
            dbc.Card(dbc.CardBody(_complexity(model)), className="mb-4"),
            html.Hr(),
            html.H4(title, className="pg-h mb-3"),
            dbc.Tabs(tabs, active_tab=tabs[0].tab_id if tabs else None),
        ]
    )

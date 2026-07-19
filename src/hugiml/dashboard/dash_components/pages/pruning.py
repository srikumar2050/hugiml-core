"""Representation Pruning page."""

from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from hugiml.dashboard.components.patterns import _get_rpte_feature_flow_audit
from hugiml.dashboard.components.pruning import (
    _base_summary_row,
    _filter_selected_by_downstream,
    _pattern_inventory,
    _result_summary_row,
    _selected_augmented_features,
    _selected_original_features,
)
from hugiml.dashboard.components.rpte_governance import (
    rpte_direct_source_terms_frame,
    rpte_model_comparison_row,
    rpte_raw_input_lineage_frame,
)
from hugiml.dashboard.dash_components.charts import bar_h
from hugiml.dashboard.dash_components.pages._shared import err, info, sn, warn
from hugiml.dashboard.dash_components.tables import make_table


def _safe_filter_selected(df, model, X, column):
    if df.empty or column not in df.columns:
        return df
    try:
        return _filter_selected_by_downstream(df, model, X, column)
    except Exception:
        return df


def _pattern_label(row):
    representation = str(row.get("_repr_column", ""))
    pattern = str(row.get("_label", "") or representation)
    return pattern if pattern == representation else f"{pattern}  [{representation}]"


def _augmented_label(row):
    representation = str(row.get("repr_column", ""))
    for column in ("display_name", "feature", "name", "pair"):
        value = row.get(column)
        if value is not None and str(value).strip():
            text = str(value)
            return text if text == representation else f"{text}  [{representation}]"
    return representation


def _family_inventory(model, X, family):
    if family == "Original features":
        frame = _selected_original_features(model, X)
        options = [
            {"label": str(value), "value": str(value)}
            for value in frame.get("feature", pd.Series(dtype=str)).astype(str).tolist()
        ]
        return frame, options

    if family == "HUG patterns":
        frame = _pattern_inventory(model)
        frame = _safe_filter_selected(frame, model, X, "_repr_column")
        options = [
            {"label": _pattern_label(row), "value": str(row.get("_repr_column", ""))}
            for _, row in frame.iterrows()
            if str(row.get("_repr_column", "")).strip()
        ]
        return frame, options

    frame = _selected_augmented_features(model)
    frame = _safe_filter_selected(frame, model, X, "repr_column")
    options = [
        {"label": _augmented_label(row), "value": str(row.get("repr_column", ""))}
        for _, row in frame.iterrows()
        if str(row.get("repr_column", "")).strip()
    ]
    return frame, options


def family_counts(model, X):
    return {
        family: len(_family_inventory(model, X, family)[0])
        for family in ("Original features", "HUG patterns", "Augmented features")
    }


def family_table(model, X, family):
    frame, options = _family_inventory(model, X, family)
    if frame.empty:
        message = {
            "Original features": "No selected original features are available.",
            "HUG patterns": "No selected HUG patterns enter the fitted downstream representation.",
            "Augmented features": "No selected augmented features enter the fitted downstream representation.",
        }.get(family, "No representation items are available.")
        return info(message), []

    if family == "HUG patterns":
        columns = [
            column
            for column in (
                "rank",
                "_label",
                "_source_features",
                "pattern_origin",
                "survivor_led",
                "survivor_features",
                "survivor_feature_count",
                "_repr_column",
            )
            if column in frame.columns
        ]
        display = frame[columns].rename(
            columns={
                "_label": "pattern",
                "_source_features": "source_features",
                "_repr_column": "representation_column",
            }
        )
    else:
        display = frame

    return make_table(display, tid="pr-ft", height="280px", filterable=True), options


def render_result(ctx, result, family, removed):
    if result is None:
        return info("Run pruning to see results.")
    X = ctx.get("X")
    base_row = _base_summary_row(ctx)
    if hasattr(result, "best_estimator_"):
        if getattr(result, "best_estimator_", None) is None:
            children = [err("The reduced-input rerun did not produce a fitted model.")]
            if getattr(result, "status_", "") == "no_patterns":
                children.append(warn("The reduced-input rerun mined zero usable patterns."))
            return html.Div(children)
        pruned_model = result.best_estimator_
        metadata = getattr(pruned_model, "fit_metadata_", None)
        counts = getattr(metadata, "downstream_feature_counts", {}) if metadata else {}
        if not isinstance(counts, dict):
            counts = {}
        original = int(
            counts.get(
                "original", len(getattr(pruned_model, "feature_names_in_", [])) or X.shape[1]
            )
        )
        pattern = int(counts.get("pattern", len(getattr(pruned_model, "patterns_", []))))
        augmented = int(
            counts.get(
                "augmented_pair", len(getattr(pruned_model, "augmented_pair_transforms_", []))
            )
        )
        score = getattr(result, "best_score_", None)
        label = "Original-feature reduced rerun"
    else:
        original = base_row["original_features"]
        pattern = base_row["pattern_features"]
        augmented = base_row["augmented_features"]
        if family == "HUG patterns":
            pattern = max(0, pattern - len(removed))
        elif family == "Augmented features":
            augmented = max(0, augmented - len(removed))
        score = getattr(result, "score", None)
        label = f"{family} downstream refit"

    pruned_row = _result_summary_row(
        label,
        score,
        X.shape[1] if X is not None else 0,
        len(removed),
        original,
        pattern,
        augmented,
    )
    frame = pd.DataFrame([base_row, pruned_row])
    if frame["cv_score"].notna().all():
        frame["score_delta_vs_current"] = [
            0.0,
            float(frame.loc[1, "cv_score"]) - float(frame.loc[0, "cv_score"]),
        ]

    chart = []
    if frame["cv_score"].notna().sum() >= 1:
        chart.append(
            dcc.Graph(
                figure=bar_h(
                    frame["cv_score"].fillna(0).tolist(),
                    frame["run"].tolist(),
                    title="CV ROC-AUC — current vs reduced representation",
                    color=["#AFA9EC", "#534AB7"],
                    xlabel="ROC-AUC",
                ),
                config={"displayModeBar": False},
            )
        )
    return html.Div(
        chart
        + [
            make_table(frame, tid="pr-comp", height=None),
            html.H6("Removed representation items", className="fw-semibold mb-2 mt-3"),
            make_table(
                pd.DataFrame({"removed_item": removed, "family": family}),
                tid="pr-rem",
                height=None,
            ),
        ]
    )


def render_rpte_result(ctx, result, removed, X_pruned):
    if result is None:
        return info("Run rebuild to see results.")
    pruned_model = getattr(result, "best_estimator_", None)
    if pruned_model is None:
        children = [err("The rebuild did not produce a fitted model.")]
        if getattr(result, "status_", "") == "no_patterns":
            children.append(warn("The reduced raw-input set produced zero usable patterns."))
        return html.Div(children)
    base_row = rpte_model_comparison_row(
        "Current",
        ctx["model"],
        getattr(ctx.get("result"), "best_score_", None),
        ctx["X"],
    )
    rebuilt_row = rpte_model_comparison_row(
        "Rebuilt",
        pruned_model,
        getattr(result, "best_score_", None),
        X_pruned,
    )
    frame = pd.DataFrame([base_row, rebuilt_row])
    if frame["validation_roc_auc"].notna().all():
        frame["validation_roc_auc_delta_vs_current"] = [
            0.0,
            float(frame.loc[1, "validation_roc_auc"]) - float(frame.loc[0, "validation_roc_auc"]),
        ]
    return html.Div(
        [
            dcc.Graph(
                figure=bar_h(
                    frame["validation_roc_auc"].fillna(0).tolist(),
                    frame["run"].tolist(),
                    title="Validation ROC-AUC",
                    color=["#AFA9EC", "#534AB7"],
                    xlabel="ROC-AUC",
                ),
                config={"displayModeBar": False},
            ),
            make_table(frame, tid="rr-comp", height=None),
            html.H6("Excluded raw inputs", className="fw-semibold mb-2 mt-3"),
            make_table(pd.DataFrame({"raw_input_excluded": removed}), tid="rr-rem", height=None),
        ]
    )


def _direct_source_tabs(direct):
    specs = [
        ("original", "Direct original features"),
        ("pattern", "Direct HUG patterns"),
        ("augmented_pair", "Direct augmented pairs"),
    ]
    tabs = []
    family_series = direct.get("family", pd.Series(index=direct.index, dtype=str))
    for family, label in specs:
        frame = direct.loc[family_series.eq(family)].copy()
        count = len(frame)
        columns = [
            column
            for column in (
                "class",
                "display_name",
                "source_column",
                "raw_sources",
                "coefficient",
                "odds_multiplier",
                "nonzero",
            )
            if column in frame.columns
        ]
        content = (
            make_table(frame[columns], tid=f"rr-d-{family}", height="260px", filterable=True)
            if not frame.empty
            else info(f"No {label.lower()} enter the final logistic model.")
        )
        tabs.append(dbc.Tab(content, label=f"{label} ({count:,})", tab_id=f"rr-tab-{family}"))
    return dbc.Tabs(
        tabs,
        id="rr-direct-tabs",
        active_tab="rr-tab-original",
        className="rpte-result-tabs mb-3",
    )


def render(
    ctx,
    prune_result=None,
    prune_family="Original features",
    prune_removed=None,
    rpte_rebuild_result=None,
    rpte_removed=None,
    rpte_X_pruned=None,
):
    model = ctx.get("model")
    X = ctx.get("X")
    flow = _get_rpte_feature_flow_audit(model)
    if flow:
        representation = flow.get("final_representation", "")
        has_leaf_block = representation in {"rpte_leaf_rules", "rpte_leaves_plus_direct_terms"}
        direct = rpte_direct_source_terms_frame(model, include_zero=True)
        lineage = rpte_raw_input_lineage_frame(model, X)
        children = [
            sn(
                "RPTE rebuild sensitivity excludes raw inputs and reruns preprocessing, mining, pair generation, tree construction, leaf encoding, and the final logistic model."
            )
        ]
        if not direct.empty:
            children.extend(
                [
                    html.H6(
                        "Direct source terms in the final logistic model"
                        if has_leaf_block
                        else "Source terms in the RPTE fallback logistic model",
                        className="fw-semibold mb-2",
                    ),
                    _direct_source_tabs(direct),
                ]
            )
        if not lineage.empty:
            options = [
                {"label": value, "value": value}
                for value in lineage["raw_input"].astype(str).tolist()
            ]
            children.extend(
                [
                    html.H6("Raw-input lineage before rebuild", className="fw-semibold mb-2 mt-3"),
                    make_table(lineage, tid="rr-lin", height="300px", filterable=True),
                    html.Label(
                        "Raw model inputs to exclude from the full rebuild",
                        className="form-label mt-3",
                    ),
                    dcc.Dropdown(
                        id="rr-excl",
                        options=options,
                        value=[],
                        multi=True,
                        placeholder="Select raw inputs…",
                    ),
                    html.Button(
                        "Run full RPTE rebuild",
                        id="rr-run",
                        n_clicks=0,
                        className="btn-fit mt-2",
                        style={"maxWidth": "320px"},
                    ),
                    html.Div(
                        id="rr-result-box",
                        children=(
                            render_rpte_result(
                                ctx,
                                rpte_rebuild_result,
                                rpte_removed or [],
                                rpte_X_pruned,
                            )
                            if rpte_rebuild_result
                            else html.Div()
                        ),
                    ),
                ]
            )
        else:
            children.append(info("Raw-input lineage is not available for this fitted model."))
        return html.Div(children)

    counts = family_counts(model, X)
    family_options = [
        {
            "label": f"{family} ({counts.get(family, 0):,})",
            "value": family,
            "disabled": counts.get(family, 0) == 0,
        }
        for family in ("Original features", "HUG patterns", "Augmented features")
    ]
    initial_family = (
        prune_family
        if counts.get(prune_family, 0)
        else next(
            (option["value"] for option in family_options if not option["disabled"]),
            "Original features",
        )
    )
    initial_table, initial_options = family_table(model, X, initial_family)
    return html.Div(
        [
            sn(
                "Original-feature removal reruns the complete HUGIML pipeline. HUG-pattern and augmented-feature removal retain the fitted representation metadata and refit the downstream estimator without the selected columns."
            ),
            html.Label("Representation family", className="form-label"),
            dbc.RadioItems(
                id="pr-family",
                options=family_options,
                value=initial_family,
                inline=True,
                inputStyle={"marginRight": "5px", "marginLeft": "12px"},
                className="mb-3",
            ),
            html.Div(id="pr-fam-tbl", children=initial_table),
            html.Label("Representation items to remove", className="form-label mt-2"),
            dcc.Dropdown(
                id="pr-items",
                options=initial_options,
                value=[],
                multi=True,
                placeholder="Select items to remove…",
            ),
            html.Button(
                "Run representation pruning",
                id="pr-run",
                n_clicks=0,
                className="btn-fit mt-2",
                style={"maxWidth": "320px"},
            ),
            html.Div(
                id="pr-result-box",
                children=(
                    render_result(ctx, prune_result, prune_family, prune_removed or [])
                    if prune_result
                    else html.Div()
                ),
            ),
        ]
    )

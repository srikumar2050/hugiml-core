"""Bounded pruning comparisons on a fixed holdout partition."""

from __future__ import annotations

import hashlib
import json


def fingerprint(model, X):
    import numpy as np

    labels = json.dumps(list(model.get_hug_features()), ensure_ascii=True).encode()
    probabilities = np.round(model.predict_proba(X), 12).astype("<f8").tobytes()
    return hashlib.sha256(labels + probabilities).hexdigest()


def metrics(model, X, y):
    import numpy as np
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    probabilities = np.asarray(model.predict_proba(X)[:, 1])
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "brier": float(brier_score_loss(y, probabilities)),
        "accuracy": float(accuracy_score(y, model.predict(X))),
        "inputs": int(X.shape[1]),
        "patterns": len(model.patterns_),
    }, probabilities


def inventory(context, X_holdout, y_holdout):
    model = context["model"]
    score, _ = metrics(model, X_holdout, y_holdout)
    return {
        "fingerprint": fingerprint(model, context["X"]),
        "features": [str(name) for name in context["X"].columns],
        "patterns": [
            {"label": f"{i + 1}. {label}", "value": i}
            for i, label in enumerate(model.get_hug_features())
        ],
        "selected_mode": "rpte" if model.base_estimator is not None else "lr",
        "train_rows": len(context["X"]),
        "holdout_rows": len(X_holdout),
        "baseline": score,
    }


def introduction(workflow):
    from dash import html

    return html.Div(
        [
            html.H3("A smaller model. A measurable trade-off."),
            html.P(
                "Choose raw inputs to exclude, or mined patterns to remove. "
                "Compare the result with your original model on the same reserved examples."
            ),
            html.Ul(
                [
                    html.Li(
                        "Raw-input exclusion rebuilds the full pipeline without those columns."
                    ),
                    html.Li(
                        "Pattern removal keeps the remaining mined patterns and refits the downstream estimator."
                    ),
                    html.Li(
                        "The estimator family and training settings stay fixed. Each comparison starts from the original model."
                    ),
                ]
            ),
            html.P(
                f"{workflow['train_rows']} training rows · {workflow['holdout_rows']} holdout rows. "
                "Training and estimator selection never use the holdout. Repeated comparisons are exploratory; "
                "the holdout is not a final independent test once used to guide edits.",
                className="info-b",
            ),
        ],
        className="pruning-intro",
    )


def validate_selection(workflow, kind, items):
    if kind not in ("features", "patterns"):
        raise ValueError("Choose raw inputs or mined patterns.")
    if not isinstance(items, list) or not items or len(items) > (3 if kind == "features" else 5):
        raise ValueError("Select 1–3 raw inputs or 1–5 mined patterns.")
    allowed = (
        workflow["features"]
        if kind == "features"
        else [item["value"] for item in workflow["patterns"]]
    )
    expected_type = str if kind == "features" else int
    if any(type(item) is not expected_type or item not in allowed for item in items):
        raise ValueError(
            "The selection does not belong to this model. Refresh the pruning controls."
        )
    if len(set(items)) != len(items):
        raise ValueError("Choose each item only once.")
    if len(items) >= len(allowed):
        raise ValueError("Keep at least one input or mined pattern.")


def compare(fit_baseline, dataset, mode, topk, order, request):
    import numpy as np
    from sklearn.base import clone

    context, _, _, X_holdout, y_holdout = fit_baseline(dataset, mode, topk, order)
    original = context["model"]
    X_train, y_train = context["X"], context["y"]
    workflow = inventory(context, X_holdout, y_holdout)
    if workflow["fingerprint"] != request.get("fingerprint"):
        raise ValueError("The original model could not be reproduced")
    kind, items = request.get("kind"), request.get("items")
    validate_selection(workflow, kind, items)
    original_signature = workflow["fingerprint"]
    if kind == "features":
        retained = [name for name in X_train.columns if str(name) not in items]
        X_reduced, holdout_reduced = X_train.loc[:, retained], X_holdout.loc[:, retained]
        candidate = clone(original).fit(X_reduced, y_train)
        removed = items
    else:
        from hugiml.pruning import PatternEditor

        removed = [workflow["patterns"][index]["label"] for index in items]
        candidate = (
            PatternEditor(original)
            .remove(items, reason="Public demonstration selection")
            .refit(X_train, y_train)
            .finalize()
        )
        holdout_reduced = X_holdout
        if len(candidate.patterns_) != len(original.patterns_) - len(items):
            raise RuntimeError("Unexpected pattern count after pruning")
    baseline, before = metrics(original, X_holdout, y_holdout)
    pruned, after = metrics(candidate, holdout_reduced, y_holdout)
    if fingerprint(original, X_train) != original_signature:
        raise RuntimeError("The original model changed during pruning")
    summary = {
        "baseline": baseline,
        "pruned": pruned,
        "removed": removed,
        "kind": kind,
        "train_rows": len(X_train),
        "holdout_rows": len(X_holdout),
        "selected_mode": workflow["selected_mode"],
        "changed_predictions": int(
            np.sum(original.predict(X_holdout) != candidate.predict(holdout_reduced))
        ),
        "mean_probability_change": float(np.abs(before - after).mean()),
        "retained_inputs": [str(name) for name in holdout_reduced.columns],
    }
    return {"view": comparison_view(summary), "summary": summary}


def comparison_view(summary):
    import dash_bootstrap_components as dbc
    from dash import html
    from guidance import add_guidance
    from plotly.utils import PlotlyJSONEncoder
    from presentation import format_pages

    from hugiml.dashboard.dash_components.pages._shared import mc

    baseline, pruned = summary["baseline"], summary["pruned"]
    delta = pruned["roc_auc"] - baseline["roc_auc"]
    rows = []
    for key, label, direction in (
        ("roc_auc", "Holdout ROC-AUC", "Higher is better"),
        ("brier", "Holdout Brier score", "Lower is better"),
        ("accuracy", "Holdout accuracy", "Higher is better"),
        ("inputs", "Input features", "Count"),
        ("patterns", "Mined patterns", "Count"),
    ):
        difference = pruned[key] - baseline[key]
        rows.append(
            html.Tr(
                [
                    html.Td(label),
                    html.Td(baseline[key]),
                    html.Td(pruned[key]),
                    html.Td(
                        f"{difference:+.3f}"
                        if isinstance(difference, float)
                        else f"{difference:+d}"
                    ),
                    html.Td(direction),
                ]
            )
        )
    view = html.Div(
        [
            html.Div(
                "Original model preserved · Independent what-if comparison", className="info-b"
            ),
            dbc.Row(
                [
                    dbc.Col(mc("Removed items", len(summary["removed"])), md=4),
                    dbc.Col(mc("Holdout AUC change", f"{delta:+.3f}"), md=4),
                    dbc.Col(
                        mc(
                            "Changed predictions",
                            f"{summary['changed_predictions']} / {summary['holdout_rows']}",
                        ),
                        md=4,
                    ),
                ],
                className="mb-4 mt-3",
            ),
            html.Div(
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th(name)
                                    for name in (
                                        "Measure",
                                        "Original",
                                        "Pruned",
                                        "Change",
                                        "Interpretation",
                                    )
                                ]
                            )
                        ),
                        html.Tbody(rows),
                    ],
                    bordered=False,
                ),
                className="comparison-table",
            ),
            html.H6("Removed from this comparison"),
            html.Ul([html.Li(item) for item in summary["removed"]]),
            html.P(
                f"The {summary['selected_mode'].upper()} estimator family is retained. "
                f"Both models use {summary['train_rows']} training rows and the same {summary['holdout_rows']} holdout rows. "
                "Repeated holdout comparisons are exploratory and may influence later choices.",
                className="info-b",
            ),
            html.P(
                "A pattern can be redundant or unused by the downstream estimator, so its removal may leave predictions unchanged. "
                "Removing a pattern does not remove its raw inputs or other representations of them. "
                "Raw-input exclusion does not guarantee that related proxy information is absent.",
                className="field-note",
            ),
            html.P(
                "Other governance pages continue to describe the original model. Reset the comparison to return to the pruning introduction.",
                className="field-note",
            ),
        ]
    )
    return add_guidance(format_pages(json.loads(json.dumps(view, cls=PlotlyJSONEncoder))))

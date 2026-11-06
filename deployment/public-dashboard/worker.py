"""Train a predefined configuration and return bounded presentation JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from guidance import add_guidance
from presentation import format_pages
from service import PAGES, validate


def readonly(node):
    """Disable controls whose callbacks belong to the desktop application."""
    if isinstance(node, list):
        return [readonly(value) for value in node]
    if isinstance(node, dict):
        node = {key: readonly(value) for key, value in node.items()}
        if node.get("props", {}).get("id") == "repr-profile-dd":
            from dash import html

            return html.P("Feature selection is available in the local dashboard.").to_plotly_json()
        if node.get("type") in ("Dropdown", "Input", "Button") and node.get("props", {}).get("id"):
            node["props"]["disabled"] = True
        return node
    return node


def fit_baseline(dataset, mode, topk=20, order=2):
    validate(dataset, mode, topk, order)

    from hugiml.dashboard.dash_components.data_utils import load_demo
    from hugiml.dashboard.dash_components.model_modes import run_dash_experiments
    from hugiml.dashboard.runner import score_cases
    from hugiml.dashboard.workbench import _promoted_evaluation_bundle

    data = load_demo(dataset)
    X, y = data["X"], data["y"]
    if len(X) > 1000 or X.shape[1] > 32:
        raise ValueError("Dataset exceeds demonstration limits")
    from sklearn.model_selection import train_test_split

    X, X_holdout, y, y_holdout = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=2026
    )
    case_ids = data["case_ids"].iloc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)
    X_holdout = X_holdout.reset_index(drop=True)
    data = {
        **data,
        "case_ids": case_ids,
        "X": X,
        "y": y,
        "meta": {
            **data.get("meta", {}),
            "n_rows": len(X),
            "n_features": X.shape[1],
            "positive_rate": float(y.mean()),
        },
    }
    params = dict(
        L=order,
        topK=topk,
        G=0.0,
        adaptive_binning=True,
        feature_mode="original_plus_patterns",
        lr_source_policy="main_effect",
        n_jobs=1,
        execution_mode="audit",
    )
    candidates = [params] if mode in ("auto", "lr") else []
    if mode in ("auto", "rpte"):
        from hugiml.rpte_bounded_lookahead_leafwise import LeafWiseBoundedLookaheadRPTEFeatureLR

        candidates.append(
            {
                **params,
                "base_estimator": LeafWiseBoundedLookaheadRPTEFeatureLR(
                    n_estimators=2, depth=2, random_state=2026
                ),
            }
        )
    runs = run_dash_experiments({"HUGIML": ["HUGIML"]}, {"HUGIML": candidates}, X, y, 3, 2026)
    if any(candidate["status"] != "ok" for candidate in runs) or len(runs) != len(candidates):
        raise RuntimeError("Training did not complete")
    run = select_run(runs)
    artifact = run["artifact"]
    model = artifact["model"]
    context = {
        **data,
        "cv": 3,
        "random_state": 2026,
        "model": model,
        "result": artifact.get("tuning_result"),
        "predictions": score_cases(model, X),
        "evaluation": _promoted_evaluation_bundle(run, y),
    }
    return context, runs, run, X_holdout, y_holdout


def train(dataset, mode, topk=20, order=2):
    from case_review import prepare
    from dash import html
    from plotly.utils import PlotlyJSONEncoder
    from pruning_workflow import introduction, inventory

    from hugiml.dashboard.dash_components.pages import (
        data_quality,
        monitoring,
        overview,
        patterns,
        performance,
        representation,
    )

    context, runs, run, X_holdout, y_holdout = fit_baseline(dataset, mode, topk, order)
    workflow = inventory(context, X_holdout, y_holdout)
    renderers = (
        overview.render,
        performance.render,
        representation.render,
        patterns.render,
        lambda c: html.Div(),
        data_quality.render,
        monitoring.render,
    )
    pages = {name: render(context) for name, render in zip(PAGES[:7], renderers)}
    pages["Pruning"] = introduction(workflow)
    from service import MODES

    selected = "rpte" if run["artifact"]["model"].base_estimator is not None else "lr"
    selection = html.Details(
        [
            html.Summary("Model selection"),
            html.P(
                f"Selected estimator: {MODES[selected]}. "
                + (
                    "The highest mean three-fold CV ROC-AUC determines the automatic selection. "
                    "These scores are selection evidence, not an independent test estimate."
                    if mode == "auto"
                    else "Estimator chosen manually in Advanced."
                )
            ),
            html.Ul(
                [
                    html.Li(
                        f"{MODES['rpte' if candidate['artifact']['model'].base_estimator is not None else 'lr']}: "
                        f"{candidate['artifact']['tuning_result'].best_score_:.3f}"
                    )
                    for candidate in runs
                ]
            ),
        ],
        className="selection-detail",
    )
    pages["Overview"] = html.Div([selection, pages["Overview"]])
    pages = add_guidance(
        format_pages(readonly(json.loads(json.dumps(pages, cls=PlotlyJSONEncoder))))
    )
    pages["__pruning"] = workflow
    pages["__cases"] = prepare(context)
    return pages


def select_run(runs):
    """Choose the largest finite CV score; prefer the first candidate on ties."""
    import math

    scores = [candidate["artifact"]["tuning_result"].best_score_ for candidate in runs]
    if not scores or any(score is None or not math.isfinite(float(score)) for score in scores):
        raise ValueError("A finite validation score is required")
    return runs[max(range(len(runs)), key=lambda index: scores[index])]


if __name__ == "__main__":
    if len(sys.argv) == 7:
        from pruning_workflow import compare

        request = json.loads(Path(sys.argv[6]).read_text(encoding="utf-8"))
        result = compare(
            fit_baseline, sys.argv[1], sys.argv[2], int(sys.argv[4]), int(sys.argv[5]), request
        )
    else:
        result = train(sys.argv[1], sys.argv[2], int(sys.argv[4]), int(sys.argv[5]))
    Path(sys.argv[3]).write_text(json.dumps(result), encoding="utf-8")

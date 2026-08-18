from __future__ import annotations

import sys
import types
from importlib.util import find_spec, spec_from_loader

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

if find_spec("streamlit") is None:
    streamlit_module = types.ModuleType("streamlit")
    streamlit_module.__spec__ = spec_from_loader("streamlit", loader=None)
    sys.modules.setdefault("streamlit", streamlit_module)

_OPTIONAL_UI_MODULES = ("dash", "dash_bootstrap_components")
_MISSING_UI_MODULES = tuple(name for name in _OPTIONAL_UI_MODULES if find_spec(name) is None)
pytestmark = pytest.mark.skipif(
    bool(_MISSING_UI_MODULES),
    reason="optional dashboard dependencies are not installed: " + ", ".join(_MISSING_UI_MODULES),
)


def _report(model_units: int, model_inspection: int, instance_mean: float) -> dict:
    return {
        "model_type": "test_model",
        "model_units": {"value": model_units, "unit": "active_components"},
        "model_inspection_units": {
            "value": model_inspection,
            "unit": "expanded_complete_evidence",
        },
        "instance_inspection_units": {
            "available": True,
            "mean": instance_mean,
            "value": instance_mean,
            "ci_lower": instance_mean - 0.5,
            "ci_upper": instance_mean + 0.5,
            "min": max(0, int(instance_mean) - 1),
            "max": int(instance_mean) + 1,
            "n_samples": 40,
        },
    }


def test_complexity_frame_uses_best_run_per_model_and_caches_reports() -> None:
    from hugiml.dashboard.dash_components import workbench

    class DummyModel:
        def __init__(self, tag: str) -> None:
            self.tag = tag

    calls: list[str] = []
    reports = {
        "lr_best": _report(4, 4, 4.0),
        "dt": _report(8, 24, 3.5),
    }

    def fake_report(model, X=None):
        assert isinstance(X, pd.DataFrame)
        calls.append(model.tag)
        return reports[model.tag]

    original = workbench.get_complexity_report
    workbench.get_complexity_report = fake_report
    X = pd.DataFrame({"x1": np.arange(40), "x2": np.arange(40) % 3})
    runs = [
        {
            "model": "Logistic Regression",
            "run_id": "lr_low",
            "status": "ok",
            "cv_roc_auc": 0.71,
            "artifact": {"model": DummyModel("unused"), "feature_frame": X},
        },
        {
            "model": "Logistic Regression",
            "run_id": "lr_best",
            "status": "ok",
            "cv_roc_auc": 0.84,
            "artifact": {"model": DummyModel("lr_best"), "feature_frame": X},
        },
        {
            "model": "Decision Tree",
            "run_id": "dt",
            "status": "ok",
            "cv_roc_auc": 0.77,
            "artifact": {"model": DummyModel("dt"), "feature_frame": X},
        },
    ]

    try:
        first = workbench._complexity_frame(runs)
        second = workbench._complexity_frame(runs)
    finally:
        workbench.get_complexity_report = original

    assert first["run_id"].tolist() == ["lr_best", "dt"]
    assert first["model_units"].tolist() == [4.0, 8.0]
    assert first["model_inspection_units"].tolist() == [4.0, 24.0]
    assert first["instance_inspection_mean"].tolist() == [4.0, 3.5]
    assert first["instance_ci_margin"].tolist() == [0.5, 0.5]
    assert first.equals(second)
    assert calls == ["lr_best", "dt"]


def test_complexity_panel_contains_comparison_views_and_methodology() -> None:
    from dash import dcc

    from hugiml.dashboard.dash_components import workbench

    class DummyModel:
        pass

    original = workbench.get_complexity_report
    workbench.get_complexity_report = lambda model, X=None: _report(5, 12, 3.0)
    runs = [
        {
            "model": "Logistic Regression",
            "run_id": "lr",
            "status": "ok",
            "cv_roc_auc": 0.82,
            "artifact": {
                "model": DummyModel(),
                "feature_frame": pd.DataFrame({"x": np.arange(40)}),
            },
        }
    ]

    try:
        panel = workbench._complexity_panel(runs)
    finally:
        workbench.get_complexity_report = original

    def walk(component):
        if isinstance(component, (list, tuple)):
            for item in component:
                yield from walk(item)
            return
        yield component
        children = getattr(component, "children", None)
        if children is not None:
            yield from walk(children)

    components = list(walk(panel))
    graphs = [component for component in components if isinstance(component, dcc.Graph)]
    rendered = str(panel.to_plotly_json())

    assert len(graphs) == 4
    top_titles = [graph.figure.layout.title.text for graph in graphs[:3]]
    assert top_titles == [
        "Model units",
        "Model inspection units",
        "Mean instance inspection units",
    ]
    assert graphs[-1].figure.layout.title.text == "Performance vs model inspection effort"
    assert graphs[-1].figure.layout.xaxis.title.text == (
        "Model inspection units (lower is simpler)"
    )
    assert list(graphs[-1].figure.data[0].x) == [12.0]
    assert graphs[-1].figure.layout.height > max(graph.figure.layout.height for graph in graphs[:3])
    complexity_table = next(
        component
        for component in components
        if getattr(component, "id", None) == "wb-complexity-table"
    )
    table_columns = [column["name"] for column in complexity_table.columns]
    assert "Mean instance inspection units (95% CI)" in table_columns
    assert "95% CI" not in table_columns
    assert "Instance range" not in table_columns
    assert complexity_table.data[0]["Mean instance inspection units (95% CI)"] == (
        "3.00 (2.50–3.50)"
    )
    assert "Raw complexity values" in rendered
    assert "How complexity is computed" in rendered
    assert "model units = count(active fitted components)" in rendered
    assert "eval_terms(X) selects one score cell per additive term" in rendered
    assert "pairwise interaction as two" in rendered
    assert "wb-complexity-table" in rendered
    assert "wb-complexity-basis" in rendered
    assert "Availability" not in rendered
    assert '"Status"' not in rendered


def test_real_sklearn_models_produce_all_three_complexity_measures() -> None:
    from hugiml.dashboard.dash_components import workbench

    rng = np.random.default_rng(2026)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=100),
            "x2": rng.normal(size=100),
            "x3": rng.normal(size=100),
        }
    )
    y = ((1.2 * X["x1"] - 0.8 * X["x2"] + rng.normal(scale=0.6, size=100)) > 0).astype(int)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500).fit(X, y),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=7).fit(X, y),
        "Random Forest": RandomForestClassifier(
            n_estimators=8,
            max_depth=3,
            random_state=7,
        ).fit(X, y),
    }
    runs = [
        {
            "model": name,
            "run_id": name.lower().replace(" ", "_"),
            "status": "ok",
            "cv_roc_auc": 0.75 + index * 0.03,
            "artifact": {"model": model, "feature_frame": X},
        }
        for index, (name, model) in enumerate(models.items())
    ]

    frame = workbench._complexity_frame(runs).set_index("model")

    assert len(frame) == 3
    assert frame["model_units"].notna().all()
    assert frame["model_inspection_units"].notna().all()
    assert frame["instance_inspection_mean"].notna().all()
    assert (frame["instance_n"] == len(X)).all()
    assert frame.loc["LR", "model_units"] == frame.loc["LR", "model_inspection_units"]
    assert frame.loc["LR", "model_units"] == frame.loc["LR", "instance_inspection_mean"]
    assert frame.loc["DT", "model_inspection_units"] >= frame.loc["DT", "instance_inspection_mean"]
    assert frame.loc["RF", "model_inspection_units"] >= frame.loc["RF", "instance_inspection_mean"]


@pytest.mark.skipif(find_spec("interpret") is None, reason="optional EBM dependency is unavailable")
def test_real_ebm_instance_units_match_nonzero_selected_term_scores() -> None:
    from interpret.glassbox import ExplainableBoostingClassifier

    from hugiml.compute_complexity import get_complexity_report, get_instance_inspection_units

    rng = np.random.default_rng(2026)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=120),
            "x2": rng.normal(size=120),
            "x3": rng.normal(size=120),
        }
    )
    y = (
        1.1 * X["x1"] - 0.7 * X["x2"] + 0.4 * X["x1"] * X["x3"] + rng.normal(scale=0.7, size=len(X))
        > 0
    ).astype(int)
    model = ExplainableBoostingClassifier(
        interactions=2,
        max_bins=32,
        max_interaction_bins=16,
        outer_bags=2,
        max_rounds=200,
        random_state=17,
        n_jobs=1,
    ).fit(X, y)

    term_values = np.asarray(model.eval_terms(X), dtype=float)
    active = np.isfinite(term_values) & (np.abs(term_values) > 1e-12)
    term_arities = np.asarray([len(term) for term in model.term_features_], dtype=np.int64)
    expected = np.asarray(active, dtype=np.int64) @ term_arities
    actual = get_instance_inspection_units(model, X)
    report = get_complexity_report(model, X=X)

    np.testing.assert_array_equal(actual, expected)
    assert report is not None
    assert report["model_type"] == "explainable_boosting_machine"
    assert report["instance_inspection_units"]["mean"] == pytest.approx(float(expected.mean()))
    assert report["model_units"]["value"] == sum(
        bool(np.any(np.isfinite(scores) & (np.abs(scores) > 1e-12)))
        for scores in model.term_scores_
    )
    assert report["model_inspection_units"]["value"] == sum(
        int(np.sum(np.isfinite(scores) & (np.abs(scores) > 1e-12))) * len(term)
        for scores, term in zip(model.term_scores_, model.term_features_)
    )


def test_rulefit_adapter_produces_all_three_complexity_measures() -> None:
    from hugiml.dashboard import workbench as core_workbench
    from hugiml.dashboard.dash_components import workbench

    original_find_spec = core_workbench.find_spec

    def without_optional_rulefit(name: str):
        if name in {"imodels", "rulefit"}:
            return None
        return original_find_spec(name)

    core_workbench.find_spec = without_optional_rulefit
    rng = np.random.default_rng(2026)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=90),
            "x2": rng.normal(size=90),
            "x3": rng.normal(size=90),
        }
    )
    y = ((X["x1"] - 0.5 * X["x2"] + rng.normal(scale=0.7, size=90)) > 0).astype(int)
    try:
        model = core_workbench.RuleFitClassifierAdapter(
            tree_size=4,
            max_rules=20,
            random_state=17,
        ).fit(X, y.to_numpy())
    finally:
        core_workbench.find_spec = original_find_spec
    runs = [
        {
            "model": "RuleFit",
            "run_id": "rulefit",
            "status": "ok",
            "cv_roc_auc": 0.81,
            "artifact": {"model": model, "feature_frame": X},
        }
    ]

    frame = workbench._complexity_frame(runs)

    assert model.backend_ == "sklearn-rulefit-style"
    assert frame.loc[0, "model_type"] == "rulefit"
    assert frame.loc[0, "model_units"] > 0
    assert frame.loc[0, "model_inspection_units"] >= frame.loc[0, "model_units"]
    assert frame.loc[0, "instance_inspection_mean"] > 0
    assert frame.loc[0, "instance_n"] == len(X)
    assert frame.loc[0, "diagnostic"] is None


def test_results_panel_registers_complexity_tab() -> None:
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "src"
        / "hugiml"
        / "dashboard"
        / "dash_components"
        / "workbench.py"
    ).read_text(encoding="utf-8")

    assert 'label="Complexity"' in source
    assert 'tab_id="wb-res-complexity"' in source
    assert 'id="wb-complexity-output"' in source
    assert 'Output("wb-complexity-output", "children")' in source
    assert source.index('label="Leaderboard"') < source.index('label="Complexity"')
    assert source.index('label="Complexity"') < source.index('label="Compare curves"')


def test_complexity_tab_is_rendered_lazily() -> None:
    from dash import no_update

    from hugiml.dashboard.dash_app import create_app
    from hugiml.dashboard.dash_components import workbench

    class DummyModel:
        pass

    original = workbench.get_complexity_report
    workbench.get_complexity_report = lambda model, X=None: _report(3, 7, 2.5)
    workbench._WB["runs"] = [
        {
            "model": "Logistic Regression",
            "run_id": "lr",
            "status": "ok",
            "cv_roc_auc": 0.81,
            "artifact": {
                "model": DummyModel(),
                "feature_frame": pd.DataFrame({"x": np.arange(40)}),
            },
        }
    ]
    try:
        app = create_app()
        callback = app.callback_map["wb-complexity-output.children"]["callback"].__wrapped__
        assert callback("wb-res-leaderboard") is no_update
        rendered = str(callback("wb-res-complexity").to_plotly_json())
    finally:
        workbench.get_complexity_report = original

    assert "Raw complexity values" in rendered
    assert "wb-complexity-table" in rendered

"""Functional tests for dashboard governance panels.

These tests stay dashboard-scoped and avoid launching Streamlit. They validate
pure data-shaping helpers and app wiring used by the governance evidence views.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


class _FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getattr__(self, name):
        return _noop


def _noop(*args, **kwargs):
    return None


def _install_fake_streamlit(monkeypatch):
    fake = types.ModuleType("streamlit")
    fake.session_state = {}
    fake.markdown = _noop
    fake.caption = _noop
    fake.info = _noop
    fake.warning = _noop
    fake.write = _noop
    fake.metric = _noop
    fake.pyplot = _noop
    fake.dataframe = _noop
    fake.bar_chart = _noop
    fake.line_chart = _noop
    fake.json = _noop
    fake.download_button = _noop
    fake.divider = _noop
    fake.text_input = lambda *args, **kwargs: kwargs.get("value", "")
    fake.selectbox = lambda label, options, index=0, *args, **kwargs: list(options)[index]
    fake.toggle = lambda *args, **kwargs: kwargs.get("value", False)
    fake.columns = lambda spec, *args, **kwargs: [_FakeContext() for _ in range(spec if isinstance(spec, int) else len(spec))]
    fake.tabs = lambda labels, *args, **kwargs: [_FakeContext() for _ in labels]
    fake.container = lambda *args, **kwargs: _FakeContext()
    fake.expander = lambda *args, **kwargs: _FakeContext()
    fake.column_config = types.SimpleNamespace(TextColumn=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    return fake


def _import_with_fake_streamlit(monkeypatch, module_name: str):
    _install_fake_streamlit(monkeypatch)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_representation_audit_wires_new_governance_panels():
    app = Path("src/hugiml/dashboard/app.py").read_text()
    assert "render_adaptive_binning_evidence" in app
    assert "render_adaptive_binning_evidence(ctx[\"model\"], ctx.get(\"X\"))" in app
    assert "render_augmented_pair_traceability" in app
    assert "Adaptive binning evidence" in app
    assert "Augmented pair traceability" in app


def test_monitoring_wires_cv_report_with_targets():
    app = Path("src/hugiml/dashboard/app.py").read_text()
    drift = Path("src/hugiml/dashboard/components/drift.py").read_text()
    assert 'render_drift(ctx["model"], ctx["X"], ctx.get("y"))' in app
    assert "render_cv_monitoring(model, X, y, cv=5)" in drift
    assert "cross_validate_monitored" in drift


def test_pattern_inventory_wires_support_and_coverage_audit():
    source = Path("src/hugiml/dashboard/components/patterns.py").read_text()
    assert "render_pattern_support_coverage_audit(model, df)" in source
    assert "model.x_train_hup_" in source
    assert "rows with zero pattern coverage" in source


def test_adaptive_binning_table_melts_ig_scores(monkeypatch):
    ge = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.governance_evidence")

    class Model:
        per_feature_b_ = {"annual_income": 7, "age": 3}
        ig_scores_ = {"annual_income": {3: 0.12, 7: 0.31}, "age": {2: 0.1, 3: 0.11}}

    df = ge.adaptive_binning_table(Model())
    assert set(df["feature"]) == {"annual_income", "age"}
    chosen = df[(df["feature"] == "annual_income") & (df["B_candidate"] == 7)]
    assert bool(chosen["is_chosen_B"].iloc[0]) is True
    assert chosen["ig_score"].iloc[0] == 0.31


def test_feature_shape_uses_singleton_coefficients_and_zero_fills_edges(monkeypatch):
    ge = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.governance_evidence")

    class Model:
        _bin_edges_ = {"income": [38500.0, 55000.0, 80000.0, 100000.0]}

        def feature_importances(self):
            return pd.DataFrame(
                {
                    "display_name": [
                        "income=[38500, 55000)",
                        "income=[38500, 55000), age=[35, 45)",
                    ],
                    "feature_type": ["pattern", "pattern"],
                    "coefficient": [0.42, 9.99],
                    "pattern_support": [0.25, 0.10],
                }
            )

    X = pd.DataFrame({"income": [40000, 52000, 56000, 90000]})
    frame = ge.feature_shape_frame(Model(), "income", X)
    assert frame["bin"].tolist() == ["[38500, 55000)", "[55000, 80000)", "[80000, 100000]"]
    assert frame["log_odds_contribution"].tolist() == [0.42, 0.0, 0.0]
    assert frame["is_mined_singleton"].tolist() == [True, False, False]
    fig = ge.plot_feature_shape(Model(), X, "income")
    assert fig is not None
    assert ge.plot_feature_bin_profile(Model(), X, "income") is not None


def test_feature_effect_profile_source_uses_log_odds_not_utility():
    source = Path("src/hugiml/dashboard/components/governance_evidence.py").read_text()
    assert "render_feature_effect_profiles" in source
    assert "feature_shape_frame" in source
    assert "log_odds_contribution" in source
    assert "feature_importances()" in source
    assert "utility/coefficient-like" not in source


def test_validation_results_rename_generic_score(monkeypatch):
    perf = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.performance")
    df = perf.normalize_validation_results(pd.DataFrame({"score": [0.81], "L": [2]}))
    assert "validation_roc_auc" in df.columns
    assert "score" not in df.columns
    assert df["validation_roc_auc"].iloc[0] == 0.81


def test_augmented_pair_effects_marks_large_raw_effect(monkeypatch):
    ge = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.governance_evidence")

    class Model:
        def explain_augmented_pair_effects(self):
            return pd.DataFrame(
                {
                    "feature": ["pair_small", "pair_large"],
                    "coefficient_raw_scale": [0.1, 10.0],
                    "raw_interpretation": ["small", "large"],
                }
            )

    df = ge.augmented_pair_effects_frame(Model())
    assert "large_raw_effect" in df.columns
    assert bool(df.loc[df["feature"].eq("pair_large"), "large_raw_effect"].iloc[0])


def test_population_coverage_frame_uses_training_hup_matrix(monkeypatch):
    patterns = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.patterns")

    class Model:
        x_train_hup_ = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1]])

    df = patterns.population_coverage_frame(Model())
    zero = df.loc[df["threshold"].eq(0), "fraction_of_training_rows"].iloc[0]
    at_least_one = df.loc[df["threshold"].eq(1), "fraction_of_training_rows"].iloc[0]
    assert zero == 0.25
    assert at_least_one == 0.75


def test_render_pattern_support_coverage_audit_executes_support_distribution(monkeypatch):
    patterns = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.patterns")

    class Model:
        x_train_hup_ = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1]])

    pattern_df = pd.DataFrame({"pattern": ["p1", "p2", "p3"], "support": [0.01, 0.10, 0.80]})
    result = patterns.render_pattern_support_coverage_audit(Model(), pattern_df)
    assert result["support"]["support"].tolist() == [0.01, 0.10, 0.80]
    assert result["coverage"].loc[result["coverage"]["threshold"].eq(0), "fraction_of_training_rows"].iloc[0] == 0.25


def test_cv_monitoring_frames_parse_report(monkeypatch):
    drift = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.drift")
    report = {
        "test_scores": [{"auc": 0.7}, {"roc_auc": 0.8}],
        "fold_drift": [{"income": 0.02}, {"income": 0.04, "age": {"psi": 0.01}}],
        "fold_metadata": [{"pattern_count": 11}, {"pattern_count": 13}],
        "fit_times_ms": [100, 120],
    }
    scores = drift._cv_scores_frame(report)
    psi = drift._cv_drift_frame(report)
    meta = drift._cv_metadata_frame(report)
    assert scores["auc"].tolist() == [0.7, 0.8]
    assert set(psi["feature"]) == {"income", "age"}
    assert meta["pattern_count"].tolist() == [11, 13]
    assert meta["fit_time_ms"].tolist() == [100, 120]


def test_no_skip_markers_in_dashboard_tests():
    import ast as _ast
    for p in Path("tests/dashboard").glob("test_*.py"):
        source = p.read_text()
        try:
            tree = _ast.parse(source)
        except SyntaxError:
            continue
        offenders: list[str] = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call):
                func = node.func
                if isinstance(func, _ast.Attribute) and func.attr in {"skip", "importorskip", "skipif"}:
                    if isinstance(func.value, _ast.Attribute) and getattr(func.value, "attr", "") == "pytest":
                        offenders.append(f"{p.name} calls pytest.{func.attr}() on line {node.lineno}")
                    elif isinstance(func.value, _ast.Name) and func.value.id == "pytest":
                        offenders.append(f"{p.name} calls pytest.{func.attr}() on line {node.lineno}")
        assert offenders == [], "\n".join(offenders)


def test_representation_audit_helpers_accept_numpy_array_metadata(monkeypatch):
    ge = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.governance_evidence")

    class Model:
        adaptive_binning = np.array([True])
        feature_names_in_ = np.array(["income", "age"])
        per_feature_b_ = np.array([7, 3])
        ig_scores_ = {
            "income": {3: 0.10, 7: 0.30},
            "age": {2: 0.08, 3: 0.12},
        }

        def plot_bin_profiles(self):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1, 2], [1, 2])
            return fig, ax

        def ig_heatmap(self):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.imshow([[0.1, 0.2]])
            return fig, ax

    df = ge.render_adaptive_binning_evidence(Model())
    assert set(df["feature"]) == {"income", "age"}
    assert df.loc[df["feature"].eq("income"), "chosen_B"].iloc[0] == 7


def test_feature_family_audit_accepts_numpy_role_arrays(monkeypatch):
    ff = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.feature_family")

    class Model:
        feature_names_in_ = np.array(["id", "income", "age"])
        patterns_ = np.array(["income=high, age=old"])
        augmented_pair_transforms_ = np.array([], dtype=object)

    X = pd.DataFrame({"id": [1, 2], "income": [10, 20], "age": [30, 40]})
    result = ff.render_feature_family_audit(
        Model(),
        X,
        sensitive_columns=np.array(["age"]),
        excluded_columns=np.array(["id"]),
        id_column="id",
    )
    originals = result["originals"]
    assert originals.loc[originals["feature"].eq("age"), "status"].iloc[0] == "Review"
    assert originals.loc[originals["feature"].eq("id"), "role"].iloc[0] == "ID"


def test_dataframe_for_display_stringifies_array_values_without_ambiguous_truth():
    from hugiml.dashboard.display import dataframe_for_display

    df = pd.DataFrame(
        {
            "feature": ["income", "age"],
            "metadata": [np.array([1, 2, 3]), ["a", "b"]],
            "maybe_missing": [None, "ok"],
        }
    )
    out = dataframe_for_display(df)
    assert out.loc[0, "metadata"] == "[1, 2, 3]"
    assert out.loc[1, "metadata"] == "['a', 'b']"
    assert pd.isna(out.loc[0, "maybe_missing"]) or out.loc[0, "maybe_missing"] == ""



def test_validation_page_passes_model_x_y_to_performance():
    app = Path("src/hugiml/dashboard/app.py").read_text()
    assert "render_performance(" in app
    assert 'model=ctx["model"]' in app
    assert 'X=ctx["X"]' in app
    assert 'y=ctx.get("y")' in app
    assert 'evaluation=ctx.get("evaluation")' in app


def test_performance_diagnostics_compute_confusion_brier_and_thresholds(monkeypatch):
    perf = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.performance")

    class Model:
        def predict_proba(self, X):
            return np.column_stack([1 - np.array([0.1, 0.8, 0.7, 0.2]), np.array([0.1, 0.8, 0.7, 0.2])])

    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    frames = perf.performance_diagnostic_frames(Model(), X, np.array([0, 1, 1, 0]))
    metrics = dict(zip(frames["metrics"]["metric"], frames["metrics"]["value"]))
    assert metrics["brier_score"] < 0.1
    assert metrics["accuracy_at_0_50"] == 1.0
    assert set(frames["confusion"].columns) == {"actual", "predicted", "count"}
    assert 0.5 in frames["thresholds"]["threshold"].tolist()


def test_data_quality_passes_model_to_missingness_and_fairness():
    app = Path("src/hugiml/dashboard/app.py").read_text()
    assert 'render_missingness(ctx["X"], model=ctx["model"])' in app
    assert 'render_fairness(\n            ctx["roles"]["sensitive_columns"],' in app
    assert 'model=ctx["model"]' in app
    assert 'X=ctx["X"]' in app


def test_fairness_computes_group_prediction_rates_and_sensitive_pattern_flags(monkeypatch):
    fairness = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.fairness")

    class Model:
        def predict_proba(self, X):
            p = np.where(X["group"].eq("A"), 0.8, 0.2)
            return np.column_stack([1 - p, p])

        def get_pattern_info(self):
            return pd.DataFrame({"pattern": ["group=A", "income=high"]})

    X = pd.DataFrame({"group": ["A", "A", "B", "B"], "income": [1, 2, 3, 4]})
    rates = fairness.predicted_positive_rate_by_group(Model(), X, ["group"])
    assert rates.loc[rates["group_value"].eq("A"), "predicted_positive_rate"].iloc[0] == 1.0
    assert rates.loc[rates["group_value"].eq("B"), "predicted_positive_rate"].iloc[0] == 0.0
    flags = fairness.sensitive_pattern_flags(Model(), ["group"])
    assert flags["pattern"].astype(str).str.contains("group").any()


def test_overview_reads_fit_metadata_and_dynamic_evidence_status(monkeypatch):
    overview = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.overview")

    class Meta:
        n_patterns = 12
        total_fit_ms = 345
        stage_times_ms = {"mine": 100}
        matrix_density = 0.12
        degraded = False
        execution_mode = "native"
        memory_usage_mb = 10.5

    class Model:
        fit_metadata_ = Meta()
        patterns_ = ["x=1"]
        feature_names_in_ = ["x"]
        def predict_proba(self, X):
            return np.column_stack([0.5, 0.5])

    class Result:
        results_ = [{"fold": 1}]

    fit_df = overview.fit_metadata_frame(Model())
    assert "total_fit_ms" in fit_df["field"].tolist()
    status_df = overview.evidence_status_frame(Model(), Result(), pd.DataFrame({"x": [1]}), [0])
    assert set(status_df["status"]).issubset({"Available", "Missing"})
    assert status_df.loc[status_df["area"].eq("Pattern inventory"), "status"].iloc[0] == "Available"


def test_missingness_surfaces_model_edges_and_binary_categorical_cols(monkeypatch):
    missingness = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.missingness")

    class Model:
        _missing_col_edges_ = {"income": [0, 1]}
        binary_categorical_cols_ = ["flag"]

    X = pd.DataFrame({"income": [1.0, np.nan], "flag": [0, 1]})
    df = missingness.missingness_frame(X, Model())
    assert bool(df.loc[df["feature"].eq("income"), "model_missing_edge"].iloc[0])
    assert bool(df.loc[df["feature"].eq("flag"), "binary_inferred_categorical"].iloc[0])


def test_config_compare_result_row_includes_fit_timing(monkeypatch):
    cc = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.config_compare")

    class Meta:
        total_fit_ms = 987
        downstream_feature_counts = {"original": 2, "pattern": 3, "augmented_pair": 1}

    class Model:
        fit_metadata_ = Meta()
        L = 2
        topK = 100
        G = 0.01
        feature_mode = "original_plus_patterns"

    class Result:
        best_estimator_ = Model()
        best_params_ = {}
        best_score_ = 0.81
        status_ = "ok"

    row = cc._result_row("r", Result(), pd.DataFrame({"a": [1], "b": [2]}))
    assert row["total_fit_ms"] == 987
    assert row["displayed_total"] == 6


def test_overview_imports_and_renders_sbom_download():
    source = Path("src/hugiml/dashboard/components/overview.py").read_text()
    assert "generate_sbom" in source
    assert "download_button" in source
    assert "Software Bill of Materials" in source


def test_representation_audit_wires_survivor_led_pattern_audit():
    app = Path("src/hugiml/dashboard/app.py").read_text()
    assert "render_survivor_led_pattern_audit" in app
    assert "Survivor-led pattern audit" in app
    assert 'render_survivor_led_pattern_audit(ctx["model"])' in app


def test_survivor_led_pattern_frame_filters_audit_rows(monkeypatch):
    ge = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.governance_evidence")

    class Model:
        def get_pattern_info(self):
            return pd.DataFrame(
                {
                    "pattern": ["x0=low", "x1=high, x2=low"],
                    "pattern_origin": ["standard", "interaction_relaxed"],
                    "survivor_led": [False, True],
                    "survivor_features": [[], ["x1"]],
                    "survivor_feature_count": [0, 1],
                    "survivor_min_marginal_ig": [np.nan, 0.0002],
                    "survivor_max_interaction_score": [np.nan, 0.18],
                    "survivor_best_partners": [[], ["x2"]],
                    "information_gain": [0.03, 0.02],
                    "support": [0.40, 0.08],
                }
            )

    df = ge.survivor_led_patterns_frame(Model())
    assert df["pattern"].tolist() == ["x1=high, x2=low"]
    assert df["pattern_origin"].tolist() == ["interaction_relaxed"]
    assert df["survivor_feature_count"].tolist() == [1]


def test_survivor_led_pattern_render_executes(monkeypatch):
    ge = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.governance_evidence")

    class Model:
        def get_pattern_info(self):
            return pd.DataFrame(
                {
                    "pattern": ["x1=high, x2=low"],
                    "pattern_origin": ["interaction_relaxed"],
                    "survivor_led": [True],
                    "survivor_features": [["x1"]],
                    "survivor_feature_count": [1],
                }
            )

    df = ge.render_survivor_led_pattern_audit(Model())
    assert len(df) == 1
    assert bool(df["survivor_led"].iloc[0])


def test_feature_family_pattern_audit_surfaces_survivor_led_columns(monkeypatch):
    ff = _import_with_fake_streamlit(monkeypatch, "hugiml.dashboard.components.feature_family")

    class Model:
        def get_pattern_info(self):
            return pd.DataFrame(
                {
                    "pattern": ["x0=low", "x1=high, x2=low"],
                    "pattern_origin": ["standard", "interaction_relaxed"],
                    "survivor_led": [False, True],
                    "survivor_features": [[], ["x1"]],
                    "survivor_feature_count": [0, 1],
                    "survivor_min_marginal_ig": [np.nan, 0.0002],
                    "survivor_max_interaction_score": [np.nan, 0.18],
                    "survivor_best_partners": [[], ["x2"]],
                }
            )

    df = ff.pattern_feature_audit(Model())
    assert "pattern_origin" in df.columns
    assert "survivor_led" in df.columns
    row = df.loc[df["pattern_origin"].eq("interaction_relaxed")].iloc[0]
    assert bool(row["survivor_led"])
    assert row["survivor_features"] == "x1"

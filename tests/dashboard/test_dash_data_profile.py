from __future__ import annotations

import base64
import io
import sys
import types
from importlib.util import find_spec, spec_from_loader

import numpy as np
import pandas as pd
import pytest

if find_spec("_hugiml_core") is None:
    native_module = types.ModuleType("_hugiml_core")
    native_module.__spec__ = spec_from_loader("_hugiml_core", loader=None)
    sys.modules.setdefault("_hugiml_core", native_module)


_OPTIONAL_UI_MODULES = ("dash", "dash_bootstrap_components")
_MISSING_UI_MODULES = tuple(name for name in _OPTIONAL_UI_MODULES if find_spec(name) is None)
pytestmark = pytest.mark.skipif(
    bool(_MISSING_UI_MODULES),
    reason="optional dashboard dependencies are not installed: " + ", ".join(_MISSING_UI_MODULES),
)


def _profile_frame(rows: int = 12_500) -> pd.DataFrame:
    rng = np.random.default_rng(44)
    target = np.where(np.arange(rows) % 10 == 0, 1, 0)
    frame = pd.DataFrame(
        {
            "case_id": [f"C-{index:06d}" for index in range(rows)],
            "continuous": rng.normal(size=rows),
            "discrete": rng.integers(0, 5, size=rows),
            "category": rng.choice(["A", "B", "C"], size=rows),
            "date_text": pd.date_range("2024-01-01", periods=rows, freq="h").astype(str),
            "numeric_text": np.arange(rows).astype(str),
            "target": target,
        }
    )
    frame.loc[::7, "continuous"] = np.nan
    return frame


def test_profile_scope_defaults_to_deterministic_ten_thousand_row_sample() -> None:
    from hugiml.dashboard.dash_components.profile_stats import profile_scope

    frame = _profile_frame()
    first = profile_scope(frame, mode="sample", target="target")
    second = profile_scope(frame, mode="sample", target="target")

    assert first.sampled is True
    assert first.profile_rows == 10_000
    assert first.total_rows == 12_500
    assert first.frame.index.tolist() == second.frame.index.tolist()
    assert abs(float(first.frame["target"].mean()) - float(frame["target"].mean())) < 0.005


def test_profile_scope_full_mode_uses_every_row() -> None:
    from hugiml.dashboard.dash_components.profile_stats import profile_scope

    frame = _profile_frame()
    scope = profile_scope(frame, mode="full", target="target")

    assert scope.sampled is False
    assert scope.profile_rows == len(frame)
    assert scope.frame is not frame
    assert scope.frame.equals(frame)


def test_semantic_types_and_variable_inventory() -> None:
    from hugiml.dashboard.dash_components.profile_stats import semantic_type, variable_inventory

    frame = _profile_frame(500)
    frame["constant"] = 1
    frame["all_missing"] = np.nan
    inventory = variable_inventory(
        frame,
        {
            "target": "target",
            "id_column": "case_id",
            "excluded_columns": ["date_text"],
            "sensitive_columns": ["category"],
        },
    ).set_index("variable")

    assert semantic_type(frame["continuous"]) == "continuous numeric"
    assert semantic_type(frame["discrete"]) == "discrete numeric"
    assert semantic_type(frame["date_text"]) == "datetime text"
    assert semantic_type(frame["numeric_text"]) == "numeric text"
    assert inventory.loc["case_id", "semantic_type"] == "identifier-like"
    assert inventory.loc["case_id", "role"] == "ID"
    assert inventory.loc["target", "role"] == "Target"
    assert inventory.loc["date_text", "role"] == "Excluded"
    assert inventory.loc["category", "role"] == "Sensitive"
    assert inventory.loc["constant", "semantic_type"] == "constant"
    assert inventory.loc["all_missing", "semantic_type"] == "all missing"


def test_model_input_columns_respect_roles() -> None:
    from hugiml.dashboard.dash_components.profile_stats import model_input_columns

    frame = _profile_frame(20)
    columns = model_input_columns(
        frame,
        {
            "target": "target",
            "id_column": "case_id",
            "excluded_columns": ["date_text"],
            "sensitive_columns": ["category"],
        },
    )

    assert "target" not in columns
    assert "case_id" not in columns
    assert "date_text" not in columns
    assert "category" in columns


def test_predictor_exclusions_report_exact_training_omissions() -> None:
    from hugiml.dashboard.dash_components.profile_stats import predictor_exclusions

    frame = _profile_frame(20)
    frame["all_missing"] = np.nan
    omitted = predictor_exclusions(
        frame,
        {
            "target": "target",
            "id_column": "case_id",
            "excluded_columns": ["date_text"],
            "sensitive_columns": ["category", "date_text"],
        },
    )

    assert omitted == [
        {"variable": "case_id", "reason": "ID", "sensitive": False},
        {"variable": "date_text", "reason": "User excluded", "sensitive": True},
        {"variable": "target", "reason": "Target", "sensitive": False},
    ]
    assert "category" not in {item["variable"] for item in omitted}
    assert "all_missing" not in {item["variable"] for item in omitted}


def test_scope_control_is_fixed_for_small_data_and_selectable_for_large_data() -> None:
    from hugiml.dashboard.dash_app import create_app
    from hugiml.dashboard.dash_components.data_profile import register_profile_dataset

    app = create_app()
    callback = next(
        value["callback"].__wrapped__
        for key, value in app.callback_map.items()
        if "profile-scope-mode.options" in key
    )

    small_reference = register_profile_dataset("demo", "credit", None, None)
    small_options, small_value = callback(small_reference)
    assert small_value == "full"
    assert small_options == [{"label": "All 800 rows", "value": "full", "disabled": True}]

    frame = _profile_frame(12_500)
    from hugiml.dashboard.dash_components import data_profile

    large_reference = data_profile._store_raw(
        "large-scope-test",
        frame,
        "Large test",
        {
            "target": "target",
            "id_column": "case_id",
            "excluded_columns": [],
            "sensitive_columns": [],
        },
        source="upload",
        source_key="large-scope-test",
        upload_name="large.csv",
    )
    large_options, large_value = callback(large_reference)
    assert large_value == "sample"
    assert [option["value"] for option in large_options] == ["sample", "full"]
    assert all(not option.get("disabled", False) for option in large_options)


def test_profile_view_labels_counts_and_omitted_column_reasons() -> None:
    from hugiml.dashboard.dash_app import create_app
    from hugiml.dashboard.dash_components.data_profile import register_profile_dataset

    app = create_app()
    callback = next(
        value["callback"].__wrapped__
        for key, value in app.callback_map.items()
        if "profile-data-view.options" in key
    )
    reference = register_profile_dataset("demo", "credit", None, None)
    options, summary = callback(reference, None, None, None, None)

    assert options == [
        {"label": "All columns (14)", "value": "raw"},
        {"label": "Predictor columns (11)", "value": "model"},
    ]
    rendered = str(summary)
    assert "Excluded from predictors (3)" in rendered
    assert "default_risk" in rendered and "Target" in rendered
    assert "customer_id" in rendered and "ID" in rendered
    assert "date_issued" in rendered and "User excluded" in rendered
    assert "Sensitive columns remain predictors" in rendered
    assert "before encoding, binning, pattern generation" in rendered


def test_missingness_patterns_and_target_associations_are_bounded() -> None:
    from hugiml.dashboard.dash_components.profile_stats import (
        missingness_patterns,
        target_associations,
        variable_inventory,
    )

    frame = _profile_frame(800)
    inventory = variable_inventory(frame, {"target": "target"})
    patterns = missingness_patterns(frame)
    associations = target_associations(frame, inventory, "target", limit=4)

    assert len(patterns) <= 20
    assert patterns["rows"].sum() <= len(frame)
    assert len(associations) <= 4
    assert set(associations.columns) == {"variable", "association", "method"}
    assert associations["association"].between(0, 1).all()


def test_upload_parser_reuses_equal_frame_content() -> None:
    from hugiml.dashboard.dash_components.data_utils import read_upload

    frame = _profile_frame(20)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    payload = "data:text/csv;base64," + base64.b64encode(buffer.getvalue().encode()).decode()

    first = read_upload(payload, "profile.csv")
    second = read_upload(payload, "profile.csv")

    assert first.shape == frame.shape
    assert second.shape == frame.shape
    assert first["case_id"].tolist() == frame["case_id"].tolist()
    assert first["numeric_text"].astype(str).tolist() == frame["numeric_text"].astype(str).tolist()
    assert first is not second


def test_demo_raw_and_model_frame_use_the_same_source_rows() -> None:
    from hugiml.dashboard.dash_components.data_utils import load_demo, load_demo_raw

    raw, roles, label = load_demo_raw("credit")
    prepared = load_demo("credit")

    assert label == prepared["mode"]
    assert len(raw) == len(prepared["X"]) == len(prepared["y"])
    assert roles == prepared["roles"]


def test_dash_profile_layout_and_lazy_callbacks() -> None:
    from hugiml.dashboard.dash_app import _workbench_data_setup, create_app
    from hugiml.dashboard.dash_components.data_profile import register_profile_dataset

    app = create_app()
    layout = str(_workbench_data_setup())
    assert "st-profile-dataset" in layout
    assert layout.index("profile-preview") < layout.index("profile-accordion")
    callback_keys = list(app.callback_map)
    assert any("profile-scope-note" in key for key in callback_keys)
    assert any("profile-variable-heading" in key for key in callback_keys)

    reference = register_profile_dataset("demo", "credit", None, None)
    overview = next(
        value["callback"].__wrapped__
        for key, value in app.callback_map.items()
        if "profile-scope-note" in key
    )
    closed = overview(None, "sample", "raw", reference, None, None, None, None, None)
    assert len(closed) == 17

    opened = overview("profile-open", "sample", "raw", reference, None, None, None, None, None)
    assert len(opened) == 17
    assert "Full dataset" in opened[0]
    assert len(opened[6]) > 0
    assert len(opened[8]) == len(opened[6])


def test_profiled_demo_reference_is_reused_for_model_preparation() -> None:
    from hugiml.dashboard.dash_app import _prepare_data_context
    from hugiml.dashboard.dash_components.data_profile import register_profile_dataset

    reference = register_profile_dataset("demo", "credit", None, None)
    _, context = _prepare_data_context(
        "demo",
        "credit",
        None,
        None,
        None,
        None,
        [],
        [],
        2,
        2026,
        profile_reference=reference,
    )

    assert context["X"].shape == (800, 11)
    assert len(context["y"]) == 800


def test_profile_scope_handles_duplicate_indices() -> None:
    from hugiml.dashboard.dash_components.profile_stats import profile_scope

    frame = _profile_frame()
    frame.index = np.arange(len(frame)) // 2
    scope = profile_scope(frame, mode="sample", target="target")

    assert len(scope.frame) == 10_000
    assert scope.frame.index.duplicated().any()


def test_profile_reference_must_match_the_current_source_controls() -> None:
    from hugiml.dashboard.dash_components.data_profile import (
        profile_reference_matches,
        register_profile_dataset,
    )

    reference = register_profile_dataset("demo", "credit", None, None)
    assert profile_reference_matches(reference, "demo", "credit", None, None)
    assert not profile_reference_matches(reference, "demo", "churn", None, None)
    assert not profile_reference_matches(reference, "upload", None, None, None)


def test_model_view_keeps_target_analysis_but_excludes_role_columns() -> None:
    from hugiml.dashboard.dash_app import create_app
    from hugiml.dashboard.dash_components.data_profile import register_profile_dataset

    app = create_app()
    reference = register_profile_dataset("demo", "credit", None, None)
    overview = next(
        value["callback"].__wrapped__
        for key, value in app.callback_map.items()
        if "profile-scope-note" in key
    )
    opened = overview("profile-open", "sample", "model", reference, None, "__none__", [], [], None)

    variables = {row["variable"] for row in opened[6]}
    assert "default_risk" not in variables
    assert "customer_id" not in variables
    assert "date_issued" not in variables
    assert len(opened[3].data) == 1
    assert len(opened[15].data) == 1


def test_binary_variable_uses_frequency_chart() -> None:
    from hugiml.dashboard.dash_app import create_app
    from hugiml.dashboard.dash_components.data_profile import register_profile_dataset

    app = create_app()
    reference = register_profile_dataset("demo", "credit", None, None)
    detail = next(
        value["callback"].__wrapped__
        for key, value in app.callback_map.items()
        if "profile-variable-heading" in key
    )
    result = detail(
        "default_risk",
        "sample",
        "raw",
        reference,
        None,
        None,
        None,
        None,
        "profile-open",
    )

    assert len(result[2].data) == 1
    assert result[2].data[0].type == "bar"


def test_stale_demo_reference_is_not_used_for_model_preparation() -> None:
    from hugiml.dashboard.dash_app import _prepare_data_context
    from hugiml.dashboard.dash_components.data_profile import register_profile_dataset

    credit_reference = register_profile_dataset("demo", "credit", None, None)
    _, context = _prepare_data_context(
        "demo",
        "churn",
        None,
        None,
        None,
        None,
        [],
        [],
        2,
        2026,
        profile_reference=credit_reference,
    )

    assert context["roles"]["target"] == "churned"
    assert len(context["X"]) == 900


def test_profile_statistics_handle_mixed_objects_and_categorical_target() -> None:
    from hugiml.dashboard.dash_components.profile_stats import (
        dataset_summary,
        profile_scope,
        variable_inventory,
    )

    rows = 10_500
    target = pd.Series(pd.Categorical(np.where(np.arange(rows) % 3 == 0, "yes", "no")))
    target.iloc[::101] = np.nan
    frame = pd.DataFrame(
        {
            "mixed": [[index] if index % 2 else {"value": index} for index in range(rows)],
            "target": target,
        }
    )
    scope = profile_scope(frame, mode="sample", target="target")
    inventory = variable_inventory(scope.frame, {"target": "target"})
    summary = dataset_summary(scope.frame, inventory)

    assert len(scope.frame) == 10_000
    assert len(inventory) == 2
    assert summary["rows"] == 10_000


def test_relationship_association_chart_uses_readable_vertical_labels() -> None:
    from hugiml.dashboard.dash_components.data_profile import _association_figure

    associations = pd.DataFrame(
        {
            "variable": [
                "short_name",
                "a_much_longer_predictor_column_name",
                "another_predictor",
            ],
            "association": [0.42, 0.71, 0.18],
            "method": ["Cramer's V", "Correlation ratio", "Cramer's V"],
        }
    )
    figure = _association_figure(associations, "outcome")

    trace = figure.data[0]
    assert trace.orientation in (None, "v")
    assert list(trace.x) == [
        "a_much_longer_predictor_column_name",
        "short_name",
        "another_predictor",
    ]
    assert list(trace.y) == [0.71, 0.42, 0.18]
    assert figure.layout.xaxis.tickangle == -45
    assert figure.layout.xaxis.automargin is True
    assert figure.layout.xaxis.title.text == "Variable"
    assert figure.layout.yaxis.title.text == "Association"
    assert figure.layout.margin.b >= 105
    assert figure.layout.height == 420


def test_relationship_layout_allocates_more_width_to_association_chart() -> None:
    from hugiml.dashboard.dash_components.styles import build_css

    css = build_css()
    assert ".profile-relationship-grid" in css
    assert "minmax(0,.9fr) minmax(0,1.35fr)" in css


def test_profiler_bar_and_histogram_palette_is_light_and_consistent() -> None:
    from hugiml.dashboard.dash_components.data_profile import (
        _association_figure,
        _categorical_figure,
        _missing_bar,
        _numeric_figure,
        _target_figure,
    )

    frame = pd.DataFrame(
        {
            "target": ["yes", "no", "yes", "no"],
            "missing": [1.0, np.nan, 3.0, np.nan],
        }
    )
    missing = _missing_bar(frame)
    target = _target_figure(frame, "target")
    discrete = _numeric_figure(pd.Series([0, 1, 1, 2, 2, 2]), "discrete")
    continuous = _numeric_figure(pd.Series(np.linspace(-3.0, 3.0, 400)), "continuous")
    categorical, _ = _categorical_figure(pd.Series(["a", "b", "a", "c"]), "category")
    association = _association_figure(
        pd.DataFrame(
            {
                "variable": ["x1", "x2"],
                "association": [0.6, 0.3],
                "method": ["Correlation ratio", "Cramer's V"],
            }
        ),
        "target",
    )

    assert missing.data[0].marker.color == "#E4A85C"
    for figure in (target, discrete, categorical, association):
        assert figure.data[0].marker.color == "#5CB98C"
        assert figure.data[0].marker.line.color == "#2F7F60"

    histogram = next(trace for trace in continuous.data if trace.type == "histogram")
    density = next(trace for trace in continuous.data if trace.type == "scatter")
    assert histogram.marker.color == "#82D0AD"
    assert histogram.marker.line.color == "#2F7F60"
    assert density.line.color == "#23845F"

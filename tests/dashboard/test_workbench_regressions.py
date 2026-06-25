"""Regression tests for the Streamlit Workbench dashboard.

These tests intentionally stay in the dashboard path and focus on the UI wiring
that changed after the v1.1.10 Workbench release. They are static/AST checks
so CI can run them without launching a Streamlit server, browser, optional
Streamlit dependencies, or the native extension.
"""

from __future__ import annotations

import ast
import py_compile
import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.9/3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]
WORKBENCH = ROOT / "src" / "hugiml" / "dashboard" / "workbench.py"
RUNNER = ROOT / "src" / "hugiml" / "dashboard" / "runner.py"
PYPROJECT = ROOT / "pyproject.toml"
DASHBOARD_TEST_DIR = ROOT / "tests" / "dashboard"


def _source(path: Path = WORKBENCH) -> str:
    return path.read_text(encoding="utf-8")


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_dashboard_sources_compile() -> None:
    py_compile.compile(str(WORKBENCH), doraise=True)
    py_compile.compile(str(RUNNER), doraise=True)


def test_no_deprecated_or_invalid_streamlit_navigation_api_usage() -> None:
    src = _source()
    assert "streamlit.components.v1" not in src
    assert "components.v1.html" not in src
    assert "_st_components.html" not in src
    assert not re.search(r"\bst\.iframe\([^)]*scrolling\s*=", src, flags=re.S)
    assert not re.search(r"\bst\.iframe\([^)]*height\s*=\s*0(?:\D|$)", src, flags=re.S)


def test_no_forced_wording_in_hugiml_workbench_copy() -> None:
    assert "forced" not in _source().lower()


def test_hugiml_advanced_feature_modes_include_all_known_modes() -> None:
    src = _source()
    assert "original_plus_patterns" in src
    assert "patterns_only" in src
    assert "original_plus_interactions" in src
    assert "st.multiselect(\n        \"Feature modes\"" in src


def test_run_experiments_uses_pre_render_callback_navigation() -> None:
    src = _source()
    tree = ast.parse(src)
    callbacks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_selected_workbench_experiments"
    ]
    assert callbacks, "Run button callback is missing"
    callback_src = ast.get_source_segment(src, callbacks[0]) or ""
    assert "on_click=_run_selected_workbench_experiments" in src
    assert 'st.session_state["hugiml_workbench_section"] = "Results"' in callback_src
    assert "st.rerun()" not in callback_src


def test_no_mutation_of_old_widget_backed_section_key() -> None:
    tree = ast.parse(_source())
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not (
                isinstance(target.value, ast.Attribute)
                and target.value.attr == "session_state"
            ):
                continue
            key_node = target.slice
            if isinstance(key_node, ast.Constant) and key_node.value == "hugiml_workbench_section_choice":
                offenders.append(node.lineno)
    assert offenders == []


def test_advanced_adaptive_binning_b_contract() -> None:
    src = _source()
    assert "Adaptive binning" in src
    assert "B has no effect while adaptive binning is enabled" in src
    assert "value=-1" in src
    assert "disabled=True" in src
    assert "min_value=2" in src
    assert "choose a fixed bin count B >= 2" in src
    assert "B is fixed at -1 while adaptive binning is enabled" in src
    assert "B must be >= 2" in src


def test_artifact_comparison_contract() -> None:
    src = _source()
    assert "Artifact comparison" in src
    assert "Selected run configurations" in src
    assert '"Base"' in src
    assert '"Alternate"' in src
    assert '"base"' in src
    assert '"added"' in src
    assert '"removed"' in src
    assert "+ Added" in src
    assert "− Removed" in src
    assert "LR coefficients" in src
    assert "DT text tree" in src
    assert "Base tree" in src
    assert "Alternate tree" in src


def test_results_navigation_uses_stateful_radio_not_streamlit_tabs_or_js() -> None:
    src = _source()
    assert 'result_sections = [' in src
    assert 'st.radio(\n        "Results view"' in src
    assert 'hugiml_results_inner_section' in src
    assert 'st.tabs(\n        [\n            "Leaderboard"' not in src
    assert '_restore_results_inner_tab_if_requested' not in src
    assert '_render_hidden_tab_click_script' not in src
    assert 'hugiml_jump_to_results_inner_tab' not in src


def test_model_drilldown_callback_keeps_results_and_drilldown_section() -> None:
    src = _source()
    assert 'on_change=_request_results_drilldown_tab' in src
    assert 'st.session_state["hugiml_nav_section"] = "Results"' in src
    assert 'st.session_state["hugiml_results_inner_section"] = "Model drill-down"' in src


def test_dashboard_tests_are_not_skipped() -> None:
    """Dashboard regression tests should always run in normal CI.

    Keep these tests static/lightweight rather than using runtime skip calls or
    skip markers for optional dashboard dependencies. That ensures the
    navigation/API regressions are checked on every CI run.
    """
    offenders: list[str] = []
    for path in DASHBOARD_TEST_DIR.glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in {"skip", "importorskip"}:
                    offenders.append(f"{path.relative_to(ROOT)} calls {func.attr}() on line {node.lineno}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for dec in node.decorator_list:
                    text = ast.unparse(dec) if hasattr(ast, "unparse") else ""
                    if "skip" in text:
                        offenders.append(f"{path.relative_to(ROOT)} has skip decorator on line {node.lineno}")
    assert offenders == []


def test_dashboard_tests_are_discovered_and_core_coverage_gate_is_preserved() -> None:
    """Guard CI coverage without pulling the whole dashboard into the core gate.

    The full dashboard UI has many optional dependencies and is intentionally
    excluded from the package-wide coverage denominator. These dashboard tests
    still run on every CI pass, but they should not reduce the existing ~80%
    core coverage range or weaken the configured coverage minimum.
    """
    config = _pyproject()
    pytest_options = config["tool"]["pytest"]["ini_options"]
    coverage_run = config["tool"]["coverage"]["run"]
    coverage_report = config["tool"]["coverage"]["report"]

    assert "tests" in pytest_options["testpaths"]
    assert "*/dashboard/*" in coverage_run["omit"]
    assert "src/hugiml/dashboard/*" in coverage_report["omit"]
    assert coverage_report["fail_under"] >= 73


def test_hugiml_artifact_tables_include_survivor_led_audit_columns() -> None:
    src = _source()
    assert "survivor_led" in src
    assert "survivor_features" in src
    assert "survivor_feature_count" in src
    assert "pattern_origin" in src
    assert "Survivor-led pattern audit" not in src




def test_hugiml_advanced_exposes_adaptive_binning_sample_controls() -> None:
    src = _source()
    assert "Adaptive sample fraction values" in src
    assert "Adaptive sample seed values" in src
    assert "wb_hugiml_adaptive_sample_frac_grid" in src
    assert "wb_hugiml_adaptive_sample_seed_grid" in src
    assert "_parse_adaptive_sample_frac_grid" in src
    assert "adaptive_binning_sample_frac" in src
    assert "adaptive_binning_sample_random_state" in src

def test_hugiml_advanced_exposes_new_interaction_parameters() -> None:
    src = _source()
    assert "Augmented pair transforms" in src
    assert "Augmented pair scoring" in src
    assert "Augmented pair scoring values" in src
    assert "interaction_information" in src
    assert "marginal_ig" in src
    assert "Interaction relaxed mining" in src
    assert "Interaction relaxed values" in src
    assert "Augmented source count" in src
    assert "Max pair features" in src
    assert "Relaxed source count" in src
    assert "Interaction-information partner count" in src
    assert "wb_hugiml_augmented_pair_mode_grid" in src
    assert "wb_hugiml_interaction_relaxed_feature_size_grid" in src


def test_hugiml_guided_keeps_core_grid_and_advanced_filters_interaction_paths() -> None:
    src = _source()
    guided = src[src.index("if mode == \"Guided\":"):src.index("params[\"adaptive_binning\"] = st.toggle")]
    assert "Augmented pair transforms" not in guided
    assert "Interaction relaxed mining" not in guided
    assert "Augmented pair scoring" not in guided
    assert "_filter_hugiml_interaction_configs(configs)" in src
    assert "invalid HUGIML interaction-path candidate" in src


def test_dashboard_runner_preserves_new_hugiml_params_for_reruns() -> None:
    src = _source(RUNNER)
    for token in (
        "augmented_pair_transforms",
        "augmented_pair_mode",
        "aug_feature_size",
        "max_pair_features",
        "ii_partner_size",
        "interaction_relaxed_mining",
        "interaction_relaxed_feature_size",
        "adaptive_binning_sample_frac",
        "adaptive_binning_sample_random_state",
    ):
        assert token in src
    assert "_HUGIML_PARAM_KEYS" in src

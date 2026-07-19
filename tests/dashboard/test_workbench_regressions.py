"""Functional checks for the Workbench dashboard.

The checks remain lightweight so they can run without launching a browser or
requiring the native extension.
"""

from __future__ import annotations

import ast
import py_compile
import re
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.9/3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]
WORKBENCH = ROOT / "src" / "hugiml" / "dashboard" / "workbench.py"
RUNNER = ROOT / "src" / "hugiml" / "dashboard" / "runner.py"
PYPROJECT = ROOT / "pyproject.toml"
DASHBOARD_TEST_DIR = ROOT / "tests" / "dashboard"
_DASH_RUNTIME_AVAILABLE = (
    find_spec("dash") is not None and find_spec("dash_bootstrap_components") is not None
)
_REQUIRES_DASH_RUNTIME = pytest.mark.skipif(
    not _DASH_RUNTIME_AVAILABLE,
    reason="optional Dash dependencies are not installed",
)


def _source(path: Path = WORKBENCH) -> str:
    return path.read_text(encoding="utf-8")


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _compact_source(source: str) -> str:
    return re.sub(r"\s+", "", source)


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


def test_dashboard_tests_do_not_use_runtime_skip_calls() -> None:
    """Optional UI dependencies are handled by collection-time markers."""
    offenders: list[str] = []
    for path in DASHBOARD_TEST_DIR.glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in {"skip", "importorskip"}:
                    offenders.append(f"{path.relative_to(ROOT)} calls {func.attr}() on line {node.lineno}")
    assert offenders == []


def test_dashboard_tests_are_discovered_and_core_coverage_gate_is_preserved() -> None:
    """Keep dashboard discovery and the configured core coverage gate active."""
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


def test_dash_workbench_uses_governance_style_setup_results_navigation() -> None:
    dash_app = (ROOT / "src" / "hugiml" / "dashboard" / "dash_app.py").read_text(encoding="utf-8")
    styles = (ROOT / "src" / "hugiml" / "dashboard" / "dash_components" / "styles.py").read_text(encoding="utf-8")
    compact = _compact_source(dash_app)
    assert 'html.Button("Setup",id="wb-nav-setup",n_clicks=1,className="tab-btnact",)' in compact
    assert 'html.Button("Results",id="wb-nav-results",n_clicks=0,className="tab-btn",)' in compact
    assert '.workbench-nav-row{display:flex;' in styles
    assert '.workbench-nav-row{display:flex!important' not in styles
    assert '.wb-nav-btn{flex:1' not in styles
    assert 'id="ri-source"' in dash_app
    primary_nav = dash_app[dash_app.index('className="hug-ctrl primary-nav"') - 900:dash_app.index('className="hug-ctrl primary-nav"') + 100]
    assert 'id="ri-source"' not in primary_nav
    assert 'id="ul-file"' not in primary_nav


def test_dash_theme_is_in_studio_title_row() -> None:
    dash_app = (ROOT / "src" / "hugiml" / "dashboard" / "dash_app.py").read_text(encoding="utf-8")
    styles = (ROOT / "src" / "hugiml" / "dashboard" / "dash_components" / "styles.py").read_text(encoding="utf-8")
    assert 'className="hero-topline"' in dash_app
    assert 'id="theme-select"' in dash_app
    assert 'Output("app-root", "className")' in dash_app
    assert 'theme-{selected.lower()}' in dash_app
    assert "_THEME_STATE_CSS" in styles
    assert ".theme-dark" in styles
    assert "def _theme_class_css" in styles
    primary_nav = dash_app[dash_app.index('className="hug-ctrl primary-nav"') - 700:dash_app.index('className="hug-ctrl primary-nav"') + 100]
    assert 'id="theme-select"' not in primary_nav


def test_dash_setup_has_one_execution_action() -> None:
    dash_app = (ROOT / "src" / "hugiml" / "dashboard" / "dash_app.py").read_text(encoding="utf-8")
    workbench = (ROOT / "src" / "hugiml" / "dashboard" / "dash_components" / "workbench.py").read_text(encoding="utf-8")
    assert 'id="btn-fit"' not in dash_app
    assert 'id="wb-run-btn"' in workbench
    assert workbench.count('id="wb-run-btn"') == 1
    assert '_prepare_data_context' in dash_app
    assert 'Dataset preparation, model fitting, and comparison run together' in dash_app


def test_dash_results_restore_streamlit_analysis_views() -> None:
    workbench = (ROOT / "src" / "hugiml" / "dashboard" / "dash_components" / "workbench.py").read_text(encoding="utf-8")
    for label in ("Leaderboard", "Compare curves", "Interpretability", "Artifact comparison", "Model drill-down"):
        assert f'label="{label}"' in workbench
    assert 'className="compact-leaderboard-chart"' in workbench
    assert 'className="results-side-card"' in workbench


def test_dash_data_quality_has_explicit_default_tabs() -> None:
    source = (ROOT / "src" / "hugiml" / "dashboard" / "dash_components" / "pages" / "data_quality.py").read_text(encoding="utf-8")
    compact = _compact_source(source)
    assert 'id="dq-main-tabs",active_tab="dq-missingness"' in compact
    assert 'id="dq-policy-tabs",active_tab="dq-columns"' in compact


def test_dash_launcher_opens_browser_after_server_is_ready() -> None:
    launcher = (ROOT / "src" / "hugiml" / "dashboard" / "launcher.py").read_text(encoding="utf-8")
    assert "_open_browser_when_ready" in launcher
    assert "socket.create_connection" in launcher
    assert "webbrowser.open_new_tab" in launcher
    assert "os.startfile" in launcher
    assert "use_reloader=False" in launcher


def test_dash_setup_supports_comma_separated_candidate_grids() -> None:
    source = (ROOT / "src" / "hugiml" / "dashboard" / "dash_components" / "workbench.py").read_text(encoding="utf-8")
    assert "comma-separated values" in source
    assert "_csv_values" in source
    assert "_expand_candidates" in source
    assert 'id="wb-adv-L",type="text"' in _compact_source(source)
    assert 'field("C value(s)", "wb-lr-C"' in source
    assert 'type="text"' in source
    assert 'multi=True' in source
    assert "candidate runs successful" in source


def test_dash_setup_shows_only_selected_model_parameters_and_count() -> None:
    source = (ROOT / "src" / "hugiml" / "dashboard" / "dash_components" / "workbench.py").read_text(encoding="utf-8")
    for component_id in (
        "wb-hugiml-config-wrap", "wb-comparison-config-wrap", "wb-param-lr",
        "wb-param-dt", "wb-param-rf", "wb-param-xgb", "wb-param-lgbm",
        "wb-param-ebm", "wb-param-rulefit",
    ):
        assert f'id="{component_id}"' in source
    assert "def _selected_model_controls" in source
    assert 'Run 1 model combination' in source
    assert 'Run {count:,} model combinations' in source
    assert '_configuration_combination_count' in source
    assert 'Output("wb-run-btn", "disabled")' in source


def test_dash_case_review_and_nested_views_have_initial_content() -> None:
    source = (ROOT / "src" / "hugiml" / "dashboard" / "dash_components" / "pages" / "prediction.py").read_text(encoding="utf-8")
    assert 'id="case-review-tabs",active_tab="case-summary"' in _compact_source(source)
    assert 'tab_id="case-summary"' in source


def test_dash_configuration_comparison_uses_compact_two_column_layout() -> None:
    source = (ROOT / "src" / "hugiml" / "dashboard" / "dash_components" / "pages" / "config_compare.py").read_text(encoding="utf-8")
    assert 'className="results-side-card config-compare-chart"' in source
    assert "md=5" in source
    assert "md=7" in source
    assert "h=225" in source
    assert 'id="cc-topK",type="text"' in _compact_source(source)
    assert 'inputMode="numeric"' in source
    assert 'current_params = getattr(ctx.get("result"), "best_params_", {}) or {}' in source


def test_dash_header_omits_dataset_summary_chip_row() -> None:
    source = (ROOT / "src" / "hugiml" / "dashboard" / "dash_app.py").read_text(encoding="utf-8")
    hero_source = source[source.index("def _hero"):source.index("# ─── Governance page router")]
    assert "Rows:" not in hero_source
    assert "Feats:" not in hero_source
    assert "chip-row" not in hero_source


def test_dash_rpte_results_include_tree_and_leaf_inspection() -> None:
    source = (ROOT / "src" / "hugiml" / "dashboard" / "dash_components" / "workbench.py").read_text(encoding="utf-8")
    assert "_rpte_tree_panel" in source
    assert "_rpte_flat_tree_sections" in source
    assert 'label="Tree / leaf view"' in source
    assert "Select a tree to inspect" in source
    assert "Leaf coefficients" in source


@_REQUIRES_DASH_RUNTIME
def test_dash_workbench_counts_all_model_parameter_combinations() -> None:
    from hugiml.dashboard.dash_components.workbench import _configuration_combination_count

    count = _configuration_combination_count(
        {"HUGIML", "Logistic Regression", "Random Forest"},
        hug_mode="advanced",
        grid_name="performance_ho",
        adv_L="1, 2",
        adv_topK="50, 100",
        adv_G="0.01",
        adv_fm=["patterns_only", "original_plus_patterns"],
        adv_strict="false",
        adv_aug="true",
        adv_relax="false",
        adv_ds=["lr"],
        rpte_lc=[],
        rpte_d="4",
        rpte_ne="10",
        lr_C="0.1, 1.0",
        lr_mi="1000",
        dt_d="4",
        dt_l="1",
        rf_n="100, 200",
        rf_d="4, 8",
        rf_l="1, 2",
        xgb_n="200",
        xgb_d="4",
        xgb_lr="0.05",
        lgbm_n="200",
        lgbm_d="-1",
        lgbm_lr="0.05",
        ebm_b="32",
        ebm_i="5",
        rf2_ts="4",
        rf2_mr="100",
    )

    assert count == 18


def test_configuration_comparison_topk_uses_plain_numeric_text_entry() -> None:
    source = (
        ROOT
        / "src"
        / "hugiml"
        / "dashboard"
        / "dash_components"
        / "pages"
        / "config_compare.py"
    ).read_text(encoding="utf-8")
    assert 'id="cc-topK",type="text"' in _compact_source(source)
    assert 'inputMode="numeric"' in source
    assert 'pattern="[0-9]+"' in source
    assert 'type="number",value=tk' not in source


@_REQUIRES_DASH_RUNTIME
def test_dash_artifact_comparison_identifies_hugiml_downstream_estimators() -> None:
    from hugiml.dashboard.dash_components.workbench import (
        _artifact_run_label,
        _config_pair_frame,
        _downstream_estimator_label,
    )

    class LeafWiseBoundedLookaheadRPTEFeatureLR:
        leaf_config = "3xD"
        depth = 4
        n_estimators = 10

    class OneVsRest:
        estimator = LeafWiseBoundedLookaheadRPTEFeatureLR()

    lr_run = {
        "run_id": "hugiml_1",
        "model": "HUGIML",
        "cv_roc_auc": 0.81,
        "params": {"L": 2, "base_estimator": None},
    }
    rpte_run = {
        "run_id": "hugiml_2",
        "model": "HUGIML",
        "cv_roc_auc": 0.83,
        "params": {"L": 2, "base_estimator": OneVsRest()},
    }

    assert _downstream_estimator_label(lr_run) == "HUGIML logistic regression"
    assert _downstream_estimator_label(rpte_run) == "RPTE · leaf=3xD · depth=4 · trees=10"
    assert "HUGIML logistic regression" in _artifact_run_label(lr_run)
    assert "RPTE" in _artifact_run_label(rpte_run)

    frame = _config_pair_frame(lr_run, rpte_run)
    row = frame.loc[frame["field"].eq("downstream_estimator")].iloc[0]
    assert row["Base"] == "HUGIML logistic regression"
    assert row["Alternate"].startswith("RPTE")

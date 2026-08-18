"""Dash interface tests; skipped when optional UI dependencies are absent."""

from importlib.util import find_spec

import pandas as pd
import pytest

_MISSING = [module for module in ("dash", "dash_bootstrap_components") if find_spec(module) is None]
pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason="optional causal-dashboard dependencies not installed: " + ", ".join(_MISSING),
)


def test_create_app_callbacks_controls_and_flow():
    import dash

    from hugiml.causal_dashboard.dash_app import create_app

    app = create_app()
    assert isinstance(app, dash.Dash)
    assert len(app.callback_map) >= 8
    text = str(app.layout)
    for component_id in [
        "cd-source",
        "cd-demo",
        "cd-data-upload",
        "cd-meta-upload",
        "cd-treatment",
        "cd-outcome",
        "cd-adjustment",
        "cd-selection-scoring",
        "cd-hug-grid",
        "cd-hug-grid-tooltip",
        "cd-baseline-grid-tooltip",
        "cd-baselines",
        "cd-dag",
        "cd-run",
        "cd-setup-page",
        "cd-results-page",
        "cd-effects",
        "cd-diagnostics",
        "cd-regions",
        "cd-comparison",
    ]:
        assert component_id in text
    assert "html.Style" not in app.index_string
    assert ".dash-dropdown-content" in app.index_string
    assert ".dash-dropdown-option" in app.index_string
    for grid_name in ("performance", "interpretability", "performance_ho", "interpretability_ho"):
        assert grid_name in text
    assert "Quick — 1 configuration" in text
    assert "value='quick'" in text


def test_hug_grid_tooltip_uses_registered_grid():
    from hugiml.causal_dashboard.dash_app import _baseline_grid_tooltip, _grid_tooltip

    performance = str(_grid_tooltip("performance_ho"))
    interpretable = str(_grid_tooltip("interpretability_ho"))
    for tooltip in (performance, interpretable):
        assert "L:" in tooltip and "topK:" in tooltip and "G:" in tooltip
        assert "RPTE" in tooltip and "LR" in tooltip
        assert "configurations" not in tooltip
        assert "adaptive_binning" not in tooltip
    assert "RPTE" not in str(_grid_tooltip("performance"))
    assert "RPTE" not in str(_grid_tooltip("interpretability"))
    assert "8 configurations" in str(_baseline_grid_tooltip("performance"))
    assert "16 configurations" in str(_baseline_grid_tooltip("performance_ho"))
    assert "1 configuration" in str(_baseline_grid_tooltip("quick"))


def test_effect_metric_tooltips_render():
    from hugiml.causal_dashboard.dash_app import _metric

    rendered = str(_metric("ATE", "1.0%", tooltip_id="tip-test", tooltip="Average effect"))
    assert "tip-test" in rendered
    assert "Average effect" in rendered


def test_robust_tables_combine_estimates_and_intervals():
    from hugiml.causal_dashboard.dash_app import _overlap_display, _robust_estimates_display

    estimates = pd.DataFrame(
        [
            {
                "Model": "T-HUG",
                "Repeated OOF ATE": -0.01,
                "ATE 95% CI low": -0.02,
                "ATE 95% CI high": 0.0,
                "ATE repeat SD": 0.003,
                "Cross-fitted DR ATE": -0.012,
                "DR 95% CI low": -0.025,
                "DR 95% CI high": 0.001,
                "DR repeat SD": 0.004,
                "Repeats": 3,
            }
        ]
    )
    compact = _robust_estimates_display(estimates)
    assert "ATE 95% CI low" not in compact.columns
    assert compact.loc[0, "Repeated OOF ATE (95% CI)"] == "-1.000% (-2.000%, +0.000%)"

    overlap = pd.DataFrame(
        [
            {
                "Model": "T-HUG",
                "Propensity range": "0.05–0.95",
                "Retained": 0.9,
                "Cross-fitted DR ATE": -0.01,
                "95% CI low": -0.02,
                "95% CI high": 0.0,
                "Repeat SD": 0.003,
            }
        ]
    )
    overlap_compact = _overlap_display(overlap)
    assert "95% CI low" not in overlap_compact.columns
    assert overlap_compact.loc[0, "Cross-fitted DR ATE (95% CI)"] == "-1.000% (-2.000%, +0.000%)"


def test_comparison_rank_styles_mark_best_and_second_best():
    from hugiml.causal_dashboard.dash_app import _rank_styles

    frame = pd.DataFrame(
        {
            "Model": ["A", "B", "C"],
            "AUC": [0.71, 0.76, 0.73],
            "Brier": [0.19, 0.21, 0.20],
            "ATE": [-0.03, -0.01, -0.02],
        }
    )
    styles = _rank_styles(frame, {"AUC": "max", "Brier": "min"})
    assert len(styles) == 4
    assert any(
        style.get("fontWeight") == "800"
        and '"B"' in style["if"]["filter_query"]
        and style["if"]["column_id"] == "AUC"
        for style in styles
    )
    assert any(
        style.get("fontStyle") == "italic"
        and '"C"' in style["if"]["filter_query"]
        and style["if"]["column_id"] == "AUC"
        for style in styles
    )
    assert all(style["if"]["column_id"] != "ATE" for style in styles)


def test_flask_root_and_dash_layout():
    from hugiml.causal_dashboard.dash_app import create_app

    client = create_app().server.test_client()
    root = client.get("/")
    layout = client.get("/_dash-layout")
    assert root.status_code == 200
    assert layout.status_code == 200
    assert b"cd-source" in layout.data
    assert b"cd-results-page" in layout.data


def test_launcher_url():
    from hugiml.causal_dashboard.launcher import _browser_url

    assert _browser_url("127.0.0.1", 8052) == "http://localhost:8052/"

from __future__ import annotations

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


def _count_kwargs() -> dict[str, object]:
    return {
        "grid_name": "performance_ho",
        "adv_L": "2",
        "adv_topK": "100",
        "adv_G": "0.01",
        "adv_fm": ["original_plus_patterns"],
        "adv_strict": "false",
        "adv_aug": "true",
        "adv_relax": "false",
        "adv_ds": ["lr"],
        "rpte_lc": [],
        "rpte_d": "4",
        "rpte_ne": "10",
        "lr_C": "0.1, 1.0",
        "lr_mi": "500, 1000",
        "dt_d": "3, 4",
        "dt_l": "1, 2",
        "rf_n": "100, 200",
        "rf_d": "4, 8",
        "rf_l": "1, 2",
        "xgb_n": "100, 200",
        "xgb_d": "3, 4",
        "xgb_lr": "0.03, 0.1",
        "lgbm_n": "100, 200",
        "lgbm_d": "-1",
        "lgbm_lr": "0.03, 0.1",
        "ebm_b": "32, 64",
        "ebm_i": "0, 5",
        "rf2_ts": "5, 10",
        "rf2_mr": "50, 100",
    }


def test_guided_modes_count_as_one_outer_model_run() -> None:
    from hugiml.dashboard.dash_components.workbench import _configuration_combination_count

    count = _configuration_combination_count(
        {"HUGIML", "Logistic Regression", "Decision Tree", "Random Forest"},
        hug_mode="guided",
        model_modes={
            "Logistic Regression": "guided",
            "Decision Tree": "default",
            "Random Forest": "advanced",
        },
        guided_grid_names={
            "Logistic Regression": "standard",
            "Decision Tree": "standard",
            "Random Forest": "standard",
        },
        **_count_kwargs(),
    )

    assert count == 11  # HUGIML 1 + LR 1 + DT 1 + RF 2*2*2


def test_advanced_mode_expands_candidate_permutations() -> None:
    from hugiml.dashboard.dash_components.workbench import _configuration_combination_count

    count = _configuration_combination_count(
        {"Logistic Regression", "Decision Tree"},
        hug_mode="default",
        model_modes={
            "Logistic Regression": "advanced",
            "Decision Tree": "advanced",
        },
        **_count_kwargs(),
    )

    assert count == 8  # LR 2*2 + DT 2*2


def test_comparison_guided_run_returns_one_refitted_best_model() -> None:
    from hugiml.dashboard.dash_components.model_modes import (
        guided_model_tune_params,
        run_dash_experiments,
    )

    rng = np.random.default_rng(17)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=120),
            "x2": rng.normal(size=120),
        }
    )
    y = ((1.4 * X["x1"] - 0.7 * X["x2"] + rng.normal(scale=0.7, size=120)) > 0).astype(int)
    params = guided_model_tune_params(
        "Logistic Regression",
        "standard",
        {"C": [0.1, 1.0], "max_iter": [500]},
    )

    runs = run_dash_experiments(
        {"Baselines": ["Logistic Regression"]},
        {"Logistic Regression": [params]},
        X,
        y.to_numpy(),
        cv=3,
        random_state=2026,
    )

    assert len(runs) == 1
    run = runs[0]
    assert run["status"] == "ok"
    assert run["params"]["mode"] == "guided_tune"
    assert run["params"]["grid_name"] == "standard"
    assert run["params"]["candidate_count"] == 2
    assert run["params"]["selected_params"]["C"] in {0.1, 1.0}
    assert run["artifact"]["model"] is not None
    assert run["artifact"]["tuning_result"] is not None


def test_every_comparison_model_has_a_named_guided_grid() -> None:
    from hugiml.dashboard.dash_components.model_modes import (
        COMPARISON_MODEL_KEYS,
        comparison_guided_grid,
        comparison_guided_grid_names,
        parameter_grid_count,
    )

    for model_name in COMPARISON_MODEL_KEYS.values():
        names = comparison_guided_grid_names(model_name)
        assert names
        for name in names:
            grid = comparison_guided_grid(model_name, name)
            assert grid
            assert parameter_grid_count(grid) >= 2


def test_rulefit_guided_run_uses_one_outer_run() -> None:
    from hugiml.dashboard.dash_components.model_modes import (
        guided_model_tune_params,
        run_dash_experiments,
    )

    rng = np.random.default_rng(23)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=90),
            "x2": rng.normal(size=90),
        }
    )
    y = ((X["x1"] - 0.4 * X["x2"] + rng.normal(scale=0.8, size=90)) > 0).astype(int)
    params = guided_model_tune_params(
        "RuleFit",
        "standard",
        {"tree_size": [3, 5], "max_rules": [20]},
    )

    runs = run_dash_experiments(
        {"Interpretable Models": ["RuleFit"]},
        {"RuleFit": [params]},
        X,
        y.to_numpy(),
        cv=3,
        random_state=2026,
    )

    assert len(runs) == 1
    assert runs[0]["status"] == "ok"
    assert runs[0]["params"]["candidate_count"] == 2
    assert runs[0]["artifact"]["tuning_result"].best_estimator_ is runs[0]["artifact"]["model"]


def test_dash_setup_exposes_three_modes_for_every_model() -> None:
    from hugiml.dashboard.dash_components.workbench import (
        _COMPARISON_MODEL_KEYS,
        _generic_config,
        _hugiml_config,
    )

    expected = ["Default", "Guided (named grid)", "Advanced (candidate grid)"]
    hug_layout = str(_hugiml_config().to_plotly_json())
    comparison_layout = str(_generic_config().to_plotly_json())

    for label in expected:
        assert label in hug_layout
    for key in _COMPARISON_MODEL_KEYS:
        assert f"wb-mode-{key}" in comparison_layout
        assert f"wb-guided-grid-{key}" in comparison_layout
        assert f"wb-guided-grid-info-{key}" in comparison_layout
        assert f"wb-advanced-{key}-box" in comparison_layout
    assert comparison_layout.count("Guided (named grid)") == len(_COMPARISON_MODEL_KEYS)
    assert comparison_layout.count("Advanced (candidate grid)") == len(_COMPARISON_MODEL_KEYS)

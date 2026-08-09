from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.multiclass import OneVsRestClassifier

from hugiml.hyperparameter_configs import (
    get_baseline_grid,
    get_hugiml_grid,
    make_l1_logistic_base_estimator,
)


def _rpte_template(grid: dict[str, list]):
    wrapped = next(value for value in grid["base_estimator"] if value is not None)
    assert isinstance(wrapped, OneVsRestClassifier)
    return wrapped.estimator


def test_hugiml_l1_grid_is_the_complete_default_contract():
    l1 = get_hugiml_grid("performance_ho")
    default = get_hugiml_grid("performance_ho")

    assert len(list(ParameterGrid(l1))) == 16
    assert len(list(ParameterGrid(default))) == 16
    assert set(default) == set(l1)
    for key in set(l1) - {"base_estimator"}:
        assert default[key] == l1[key]
    assert [repr(value) for value in default["base_estimator"]] == [
        repr(value) for value in l1["base_estimator"]
    ]
    assert l1["lr_solver"] == ["adaptive_l1"]
    assert l1["lr_C"] == [0.5]
    assert _rpte_template(l1).lr_penalty == "l1"
    assert _rpte_template(l1).lr_C == 0.5


def test_named_l1_estimator_and_baseline_grid_sizes_are_literal():
    estimator = make_l1_logistic_base_estimator()
    assert isinstance(estimator, LogisticRegression)
    assert estimator.solver == "liblinear"
    assert estimator.penalty == "l1" or estimator.l1_ratio == 1.0
    assert estimator.C == 0.5
    assert estimator.max_iter == 300

    expected = {
        "XGBoost": 16,
        "LightGBM": 16,
        "RandomForest": 16,
        "LogisticRegression": 16,
        "EBM": 8,
        "RuleFit": 8,
    }
    for model, count in expected.items():
        assert len(list(ParameterGrid(get_baseline_grid(model)))) == count

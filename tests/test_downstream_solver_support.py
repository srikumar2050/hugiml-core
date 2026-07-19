# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for downstream LogisticRegression/SGD solver selection."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from hugiml import HUGIMLClassifierNative
from hugiml.exceptions import HUGIMLParamError
from hugiml.hyperparameter_configs import make_l1_logistic_base_estimator
from hugiml.serialization import MODEL_SCHEMA_VERSION, load_model, save_model
from tests.conftest import requires_extension

_SOLVER_CASES = [
    ("auto", LogisticRegression, {"solver": "liblinear"}),
    ("saga", LogisticRegression, {"solver": "saga"}),
    ("sgd", SGDClassifier, {"loss": "log_loss"}),
]


def _fit_solver_classifier(synthetic_split, solver: str) -> HUGIMLClassifierNative:
    X_tr, _, y_tr, _ = synthetic_split
    clf = HUGIMLClassifierNative(
        B=3,
        L=1,
        G=1e-3,
        topK=12,
        feature_mode="original_plus_patterns",
        execution_mode="production",
        lr_solver=solver,
    )
    X_p, y_p = clf.prepareXy(X_tr, y_tr)
    X_fit, _, y_fit, _ = train_test_split(
        X_p,
        y_p,
        test_size=0.25,
        random_state=0,
        stratify=y_p,
    )
    clf.fit(X_fit, y_fit)
    return clf


@requires_extension
@pytest.mark.parametrize(("solver", "expected_cls", "expected_attrs"), _SOLVER_CASES)
def test_lr_solver_selects_expected_downstream_estimator(
    synthetic_split,
    solver: str,
    expected_cls: type,
    expected_attrs: dict[str, str],
) -> None:
    clf = _fit_solver_classifier(synthetic_split, solver)
    estimator = clf.model_.named_steps["clf"]

    assert isinstance(estimator, expected_cls)
    for name, expected_value in expected_attrs.items():
        assert getattr(estimator, name) == expected_value
    assert estimator.random_state == 0
    assert estimator.max_iter == 500
    if solver in {"auto", "saga"}:
        assert getattr(estimator, "l1_ratio", None) == 1.0 or getattr(estimator, "penalty", None) == "l1"


@requires_extension
@pytest.mark.parametrize("solver", ["auto", "saga", "sgd"])
def test_lr_solver_predict_proba_shape(synthetic_split, solver: str) -> None:
    _, X_te, _, _ = synthetic_split
    clf = _fit_solver_classifier(synthetic_split, solver)

    proba = clf.predict_proba(X_te)

    assert proba.shape == (len(X_te), 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_te)), atol=1e-6)


@requires_extension
def test_lr_solver_round_trips_through_sklearn_params() -> None:
    clf = HUGIMLClassifierNative(B=4, L=1, lr_solver="saga")

    params = clf.get_params()
    clone = HUGIMLClassifierNative(**params)

    assert params["lr_solver"] == "saga"
    assert clone.get_params()["lr_solver"] == "saga"


@requires_extension
def test_invalid_lr_solver_raises_param_error(synthetic_split) -> None:
    X_tr, _, y_tr, _ = synthetic_split
    clf = HUGIMLClassifierNative(B=3, L=1, lr_solver="newton")
    X_p, y_p = clf.prepareXy(X_tr, y_tr)

    with pytest.raises(HUGIMLParamError, match="lr_solver"):
        clf.fit(X_p, y_p)


@requires_extension
def test_sgd_solver_uses_native_serialization(tmp_path, synthetic_split) -> None:
    _, X_te, _, _ = synthetic_split
    clf = _fit_solver_classifier(synthetic_split, "sgd")
    out = tmp_path / "sgd_solver.hugiml"

    save_model(clf, out)
    loaded = load_model(out)

    with zipfile.ZipFile(out, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        estimator_config = json.loads(zf.read("estimator.json"))

    assert manifest["schema_version"] == MODEL_SCHEMA_VERSION == 10
    assert estimator_config["class"] == "sklearn.pipeline.Pipeline"
    step_configs = {step["name"]: step["estimator"] for step in estimator_config["steps"]}
    assert step_configs["clf"]["class"] == "sklearn.linear_model.SGDClassifier"
    assert "_pickle_fallback" not in step_configs["clf"]
    np.testing.assert_allclose(
        clf.predict_proba(X_te),
        loaded.predict_proba(X_te),
        rtol=2e-6,
        atol=1e-8,
    )


def test_auto_solver_keeps_native_multiclass_linear_path() -> None:
    clf = HUGIMLClassifierNative(lr_solver="auto")

    estimator = clf._make_estimator(3)

    assert isinstance(estimator, LogisticRegression)
    assert estimator.solver == "saga"
    assert getattr(estimator, "penalty", None) == "l1"


def test_explicit_liblinear_base_estimator_is_class_aware() -> None:
    base = make_l1_logistic_base_estimator()
    clf = HUGIMLClassifierNative(base_estimator=base)

    binary = clf._make_estimator(2)
    multiclass = clf._make_estimator(3)

    assert isinstance(binary, LogisticRegression)
    assert binary.solver == "liblinear"
    assert isinstance(multiclass, OneVsRestClassifier)
    assert multiclass.n_jobs == 1
    assert isinstance(multiclass.estimator, LogisticRegression)
    assert multiclass.estimator.solver == "liblinear"
    assert (
        getattr(multiclass.estimator, "l1_ratio", None) == 1.0
        or getattr(multiclass.estimator, "penalty", None) == "l1"
    )
    assert clf.base_estimator is base


@requires_extension
def test_explicit_l1_base_estimator_fits_multiclass_iris() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0,
        stratify=y,
    )
    clf = HUGIMLClassifierNative(
        B=3,
        L=1,
        G=0.01,
        topK=10,
        feature_mode="original_plus_patterns",
        adaptive_binning=True,
        execution_mode="audit",
        base_estimator=make_l1_logistic_base_estimator(),
    )

    clf.fit(X_train, y_train)
    estimator = clf.model_.named_steps["clf"]
    probabilities = clf.predict_proba(X_test)

    assert isinstance(estimator, OneVsRestClassifier)
    assert len(estimator.estimators_) == 3
    importances = clf.feature_importances()

    assert probabilities.shape == (len(X_test), 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert len(importances) == clf.x_train_downstream_.shape[1]
    assert np.isfinite(importances["coefficient"]).all()
    assert clf.get_complexity() is not None


@requires_extension
def test_fast_grid_tune_supports_explicit_l1_multiclass_base_estimator() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=0,
        stratify=y,
    )
    grid = {
        "B": [-1],
        "adaptive_binning": [True],
        "L": [1],
        "topK": [10],
        "feature_mode": ["original_plus_patterns"],
        "G": [0.01],
        "convert_binary_to_categorical": [False],
        "base_estimator": [make_l1_logistic_base_estimator()],
    }

    result = HUGIMLClassifierNative.fast_grid_tune(
        X_train,
        y_train,
        X_valid,
        y_valid,
        param_grid=grid,
        base_params={"execution_mode": "production"},
        scoring="accuracy",
    )
    estimator = result["best_model"].model_.named_steps["clf"]

    assert isinstance(estimator, OneVsRestClassifier)
    assert len(estimator.estimators_) == 3
    assert result["best_model"].predict_proba(X_valid).shape == (len(X_valid), 3)


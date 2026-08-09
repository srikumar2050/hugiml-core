from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from experiments.benchmark import benchmark_engine, tabarena_protocol

_BUNDLED_FEATURES = (
    Path(benchmark_engine.__file__).resolve().parent
    / "tabarena"
    / "_dependencies"
    / "autogluon"
    / "features"
)
try:
    _HAS_AUTOGLUON_FEATURES = importlib.util.find_spec("autogluon.features") is not None
except ModuleNotFoundError:
    _HAS_AUTOGLUON_FEATURES = False
if not _HAS_AUTOGLUON_FEATURES and not _BUNDLED_FEATURES.exists():
    pytest.skip(
        "TabArena preprocessing tests use the benchmark dependency group",
        allow_module_level=True,
    )


def test_tabarena_preprocessing_is_fit_on_child_training_rows_only() -> None:
    X_fit = pd.DataFrame(
        {
            "signal": [0, 1, 2, 3, 4, 5],
            "training_constant": [1, 1, 1, 1, 1, 1],
            "category": pd.Series(["a", "b", "a", "b", "a", "b"], dtype="category"),
        }
    )
    X_validation = pd.DataFrame(
        {
            "signal": [6, 7],
            "training_constant": [2, 3],
            "category": pd.Series(["unseen", "a"], dtype="category"),
        }
    )
    generator, transformed_fit, transformed_validation = (
        benchmark_engine._tabarena_model_agnostic_transform(
            X_fit,
            np.array([0, 1, 0, 1, 0, 1]),
            X_validation,
        )
    )

    assert "training_constant" not in transformed_fit.columns
    assert list(transformed_fit.columns) == list(transformed_validation.columns)
    assert len(transformed_fit) == len(X_fit)
    assert len(transformed_validation) == len(X_validation)
    assert "training_constant" not in generator.features_in


def test_tabarena_categorical_labels_are_stable_across_missingness() -> None:
    training = pd.DataFrame(
        {
            "category": pd.Series(
                pd.Categorical([0, 1, None, 2], categories=[0, 1, 2])
            )
        }
    )
    outer = pd.DataFrame(
        {
            "category": pd.Series(
                pd.Categorical([0, 1, 2, 0], categories=[0, 1, 2])
            )
        }
    )

    stable_training = benchmark_engine._canonicalize_tabarena_categorical_frame(
        training
    )
    stable_outer = benchmark_engine._canonicalize_tabarena_categorical_frame(outer)

    training_values = stable_training["category"].astype(object).tolist()
    assert training_values[:2] == ["0", "1"]
    assert pd.isna(training_values[2])
    assert training_values[3] == "2"
    assert stable_outer["category"].astype(object).tolist() == ["0", "1", "2", "0"]
    training_dummy = pd.get_dummies(stable_training.astype("string"), dummy_na=True)
    outer_dummy = pd.get_dummies(stable_outer.astype("string"), dummy_na=True)
    assert set(training_dummy.columns) == set(outer_dummy.columns)


def test_tabarena_retains_each_fitted_generator_with_its_child() -> None:
    rng = np.random.default_rng(7)
    y = np.tile(np.array([0, 1], dtype=int), 48)
    X = pd.DataFrame(
        {
            "numeric": rng.normal(size=len(y)),
            "category": pd.Series(np.where(np.arange(len(y)) % 3, "a", "b"), dtype="category"),
        }
    )

    ensemble, _, _, _, metadata = benchmark_engine._fit_tabarena_cv_ensemble(
        model_name="Logistic Regression",
        candidates=[{"C": 1.0, "class_weight": None}],
        builder=lambda params: LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=100,
        ),
        X_train=X,
        y_train=y,
        random_state=3,
    )

    assert metadata["cv_ensemble_child_count"] == 8
    assert len(ensemble.estimators_) == 8
    assert all(
        isinstance(child, benchmark_engine.TabArenaPreprocessedEstimator)
        for child in ensemble.estimators_
    )
    probabilities = ensemble.predict_proba(X.iloc[:5])
    assert probabilities.shape == (5, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    restored = pickle.loads(pickle.dumps(ensemble))
    np.testing.assert_allclose(restored.predict_proba(X.iloc[:5]), probabilities)

    complexity = benchmark_engine.complexity_fields(
        ensemble,
        X.iloc[:5],
        benchmark_engine.benchmark_base.get_complexity,
    )
    assert complexity["complexity_model_units"] is not None
    assert complexity["complexity_model_inspection_units"] is not None
    assert complexity["complexity_instance_inspection_units_n_samples"] == 5
    assert complexity["complexity_instance_inspection_units_mean"] is not None
    assert complexity["complexity_instance_inspection_units_sum"] > 0


def test_fold_preprocessing_reuse_matches_independent_candidate_preprocessing() -> None:
    rng = np.random.default_rng(19)
    y = np.tile(np.array([0, 1], dtype=int), 36)
    X = pd.DataFrame(
        {
            "numeric": rng.normal(size=len(y)),
            "category": pd.Series(
                np.where(np.arange(len(y)) % 3, "a", "b"), dtype="category"
            ),
        }
    )
    candidates = [{"C": 0.5}, {"C": 1.0}]

    def make_child(params, X_fit, y_fit, X_validation, prepared=None):
        if prepared is None:
            prepared = benchmark_engine._tabarena_model_agnostic_transform(
                X_fit, y_fit, X_validation
            )
        generator, transformed_fit, _ = prepared
        estimator = LogisticRegression(
            C=params["C"], solver="liblinear", random_state=5
        ).fit(transformed_fit, y_fit)
        return benchmark_engine.TabArenaPreprocessedEstimator(generator, estimator)

    def independent_child(params, X_fit, y_fit, X_validation, y_validation):
        return make_child(params, X_fit, y_fit, X_validation)

    def reused_child(
        params, X_fit, y_fit, X_validation, y_validation, prepared
    ):
        return make_child(params, X_fit, y_fit, X_validation, prepared)

    common = {
        "candidates": candidates,
        "X": X,
        "y": y,
        "random_state": 13,
        "probability_fn": benchmark_engine.probability_matrix,
        "n_splits": 3,
    }
    independent = tabarena_protocol.fit_cross_validated_ensemble(
        **common, fit_child=independent_child
    )
    reused = tabarena_protocol.fit_cross_validated_ensemble(
        **common,
        fit_child=reused_child,
        prepare_fold=lambda X_fit, y_fit, X_validation, y_validation: (
            benchmark_engine._tabarena_model_agnostic_transform(
                X_fit, y_fit, X_validation
            )
        ),
    )

    assert independent[1] == reused[1]
    assert independent[2] == pytest.approx(reused[2], abs=1e-12)
    np.testing.assert_allclose(
        independent[0].predict_proba(X), reused[0].predict_proba(X), atol=1e-12
    )
    assert reused[4]["preprocessing_fit_count"] == 3
    assert reused[4]["preprocessing_reuse_count"] == 3


def test_tabarena_dependency_path_is_independent_of_suite_output_directory(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        benchmark_engine,
        "SCRIPT_DIR",
        benchmark_engine.BENCHMARK_ENGINE_DIR / "tabarena",
    )
    X_fit = pd.DataFrame({"value": [0, 1, 2, 3]})
    _, transformed_fit, transformed_validation = (
        benchmark_engine._tabarena_model_agnostic_transform(
            X_fit,
            np.array([0, 1, 0, 1]),
            pd.DataFrame({"value": [4, 5]}),
        )
    )

    assert transformed_fit.shape == (4, 1)
    assert transformed_validation.shape == (2, 1)


def test_split_telemetry_separates_all_timed_phases(monkeypatch) -> None:
    X = pd.DataFrame({"value": np.arange(24, dtype=float)})
    y = np.tile(np.array([0, 1], dtype=int), 12)

    def fake_fit_or_tune_model(model_name, X_train, y_train, **kwargs):
        del model_name, kwargs
        estimator = LogisticRegression(max_iter=100).fit(X_train, y_train)
        return estimator, {"C": 1.0}, 0.75, 12.0, 34.0, {
            "inner_tuning_performed": True,
            "selection_refit_performed": False,
        }, benchmark_engine.benchmark_base.get_complexity

    monkeypatch.setattr(
        benchmark_engine,
        "fit_or_tune_model",
        fake_fit_or_tune_model,
    )
    row, _, _ = benchmark_engine.evaluate_official_split(
        model_name="Logistic Regression",
        task_meta={
            "task_id": 1,
            "dataset_id": 2,
            "dataset_name": "telemetry",
            "n_classes": 2,
        },
        X=X,
        y=y,
        split={
            "split_id": 0,
            "repeat": 0,
            "fold": 0,
            "sample": 0,
            "train_idx": np.arange(16),
            "test_idx": np.arange(16, 24),
        },
        tune=True,
        inner_splits=3,
        random_state=42,
        hugiml_scenario="augmented_pair",
        hugiml_max_fit_seconds=None,
        validation_protocol="nested",
        early_stopping=False,
    )

    assert row["status"] == "ok"
    assert row["fit_seconds"] == 0.012
    assert row["tune_seconds"] == 0.034
    assert row["predict_seconds"] >= 0.0
    assert row["complexity_seconds"] >= 0.0
    assert row["phase_accounted_seconds"] == pytest.approx(
        row["tune_seconds"]
        + row["predict_seconds"]
        + row["complexity_seconds"]
    )
    assert row["evaluation_overhead_seconds"] >= 0.0
    assert row["complexity_instance_inspection_units_n_samples"] == 8

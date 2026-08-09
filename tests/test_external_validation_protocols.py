from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "experiments" / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_engine as runner  # noqa: E402
import tabarena_protocol  # noqa: E402


def _three_fold_partition() -> list[dict[str, object]]:
    rows = np.arange(12)
    splits = []
    for fold in range(3):
        test = np.arange(fold * 4, (fold + 1) * 4)
        splits.append(
            {
                "repeat": 0,
                "sample": 0,
                "fold": fold,
                "split_id": fold,
                "train_idx": np.setdiff1d(rows, test),
                "test_idx": test,
            }
        )
    return splits


def test_tabarena_retains_eight_folds_when_validation_folds_omit_a_rare_class() -> None:
    y = np.asarray([0] * 32 + [1] * 16 + [2] * 5, dtype=int)
    X = pd.DataFrame({"feature": np.arange(y.size, dtype=float)})
    child_training_classes: list[set[int]] = []

    def fit_child(params, X_fit, y_fit, X_validation, y_validation):
        child_training_classes.append(set(np.unique(y_fit).tolist()))
        return DummyClassifier(strategy="prior").fit(X_fit, y_fit)

    ensemble, _, _, _, telemetry = tabarena_protocol.fit_cross_validated_ensemble(
        candidates=[{}], X=X, y=y, random_state=42, fit_child=fit_child,
        probability_fn=runner.probability_matrix, n_splits=8,
    )

    assert len(ensemble.estimators_) == 8
    assert telemetry["cv_ensemble_child_count"] == 8
    assert child_training_classes == [{0, 1, 2}] * 8


def test_tabarena_reuses_fold_preparation_without_changing_oof_selection() -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=3,
        random_state=11,
    )
    frame = pd.DataFrame(X)

    def run(prepared: bool):
        preparation_calls = 0

        def prepare_fold(X_fit, y_fit, X_validation, y_validation):
            nonlocal preparation_calls
            preparation_calls += 1
            return X_fit, X_validation

        def fit_child(
            params, X_fit, y_fit, X_validation, y_validation, cached=None
        ):
            if cached is not None:
                X_fit, X_validation = cached
            return DummyClassifier(strategy=params["strategy"]).fit(X_fit, y_fit)

        result = tabarena_protocol.fit_cross_validated_ensemble(
            candidates=[{"strategy": "prior"}, {"strategy": "most_frequent"}],
            X=frame,
            y=y,
            random_state=7,
            fit_child=fit_child,
            probability_fn=runner.probability_matrix,
            n_splits=3,
            prepare_fold=prepare_fold if prepared else None,
        )
        return result, preparation_calls

    uncached, uncached_calls = run(False)
    cached, cached_calls = run(True)
    np.testing.assert_allclose(
        uncached[0].predict_proba(frame), cached[0].predict_proba(frame)
    )
    assert uncached[1] == cached[1]
    assert uncached[2] == pytest.approx(cached[2])
    assert uncached_calls == 0
    assert cached_calls == 3
    assert cached[4]["preprocessing_fit_count"] == 3
    assert cached[4]["preprocessing_reuse_count"] == 3


def test_protocol_cli_defaults_preserve_nested_cv() -> None:
    args = runner.build_parser().parse_args([])
    assert args.validation_protocol == "nested"
    assert args.early_stopping is None
    assert args.inner_splits == 3


def test_resume_order_finishes_partial_dataset_before_lower_workload_task() -> None:
    tasks = [
        {"task_id": 1, "n_rows": 100, "n_features": 2, "n_classes": 2},
        {"task_id": 2, "n_rows": 1000, "n_features": 20, "n_classes": 2},
        {"task_id": 3, "n_rows": 10, "n_features": 1, "n_classes": 2},
    ]
    models = ["HUGIML", "XGB standard", "LightGBM standard"]
    ordered = runner.order_execution_tasks(
        tasks,
        models=models,
        completed_pairs={(2, "HUGIML")},
        deferred_task_ids={3},
    )
    assert [row["task_id"] for row in ordered] == [2, 1, 3]


def test_protocol_cli_accepts_rotating_and_explicit_stopping() -> None:
    args = runner.build_parser().parse_args(
        ["--validation-protocol", "rotating", "--no-early-stopping"]
    )
    assert args.validation_protocol == "rotating"
    assert args.early_stopping is False


def test_rotating_fold_assignments_are_disjoint_and_exhaustive() -> None:
    rotated = runner.rotating_validation_splits(_three_fold_partition(), 12)
    assert [(row["fold"], row["validation_fold"]) for row in rotated] == [
        (0, 1),
        (1, 2),
        (2, 0),
    ]
    for row in rotated:
        train = np.asarray(row["train_idx"])
        validation = np.asarray(row["validation_idx"])
        test = np.asarray(row["test_idx"])
        assert np.intersect1d(train, validation).size == 0
        assert np.intersect1d(train, test).size == 0
        assert np.intersect1d(validation, test).size == 0
        assert np.unique(np.concatenate([train, validation, test])).size == 12


def test_rotating_protocol_rejects_independent_holdouts() -> None:
    splits = _three_fold_partition()
    splits[1]["test_idx"] = np.asarray(splits[0]["test_idx"]).copy()
    with pytest.raises(ValueError, match="disjoint exhaustive test folds"):
        runner.rotating_validation_splits(splits, 12)


def test_rpte_switch_is_applied_inside_multiclass_wrapper() -> None:
    _, candidates = runner.benchmark_base._hugiml_grid_for_scenario("augmented_pair")
    configured = runner._configure_hugiml_candidates(candidates, early_stopping=True)
    rpte = [
        row["base_estimator"]
        for row in configured
        if row["base_estimator"] is not None
        and (
            "early_stopping" in row["base_estimator"].get_params(deep=True)
            or "estimator__early_stopping" in row["base_estimator"].get_params(deep=True)
        )
    ]
    assert rpte
    for estimator in rpte:
        params = estimator.get_params(deep=True)
        key = "early_stopping" if "early_stopping" in params else "estimator__early_stopping"
        assert params[key] is True


def test_reuse_signature_separates_protocol_and_stopping_policy() -> None:
    nested = {
        "metadata": {
            "tune": True,
            "inner_splits": 3,
            "random_state": 42,
            "validation_protocol": "nested",
            "early_stopping": False,
            "grid_snapshot": {"XGB standard": {}},
            "source": {},
        }
    }
    rotating = {
        "metadata": {
            **nested["metadata"],
            "validation_protocol": "rotating",
            "early_stopping": True,
        }
    }
    assert runner._model_reuse_signature(nested, "XGB standard") != runner._model_reuse_signature(
        rotating, "XGB standard"
    )


def test_baseline_stopping_metadata_does_not_invalidate_hugiml_reuse() -> None:
    common = {
        "tune": True,
        "inner_splits": 3,
        "random_state": 42,
        "validation_protocol": "rotating",
        "early_stopping": True,
        "grid_snapshot": {"HUGIML": {"candidate_count": 16}},
        "source": {"package_version": "test-version"},
    }
    earlier = {"metadata": dict(common)}
    current = {
        "metadata": {
            **common,
            "baseline_validation_early_stopping": True,
            "baseline_early_stopping_max_estimators": 200,
            "baseline_early_stopping_patience": 20,
        }
    }
    assert runner._model_reuse_signature(
        earlier, "HUGIML"
    ) == runner._model_reuse_signature(current, "HUGIML")


def test_rotating_selection_uses_balanced_accuracy_when_validation_auc_is_undefined() -> None:
    X_fit = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y_fit = np.tile(np.arange(3), 4)
    X_validation = pd.DataFrame({"x": np.arange(6, dtype=float)})
    y_validation = np.tile(np.arange(2), 3)

    _, _, score, _, metadata = runner._fit_candidates_without_refit(
        model_name="HUGIML",
        candidates=[{}],
        builder=lambda _: DummyClassifier(strategy="prior"),
        complexity_fn=None,
        folds=[(X_fit, y_fit, X_validation, y_validation)],
    )

    assert np.isfinite(score)
    assert metadata["validation_selection_metrics"] == ["balanced_accuracy_fallback"]
    assert metadata["selection_refit_performed"] is False


def test_validation_baseline_grids_keep_sixteen_candidates() -> None:
    assert runner.benchmark_base.get_baseline_grid("RandomForest")["n_estimators"] == [
        200,
        400,
    ]
    expected = {
        "XGBoost": None,
        "LightGBM": None,
        "RandomForest": [100, 200],
    }
    for family, estimators in expected.items():
        grid = runner.rotating_baseline_grid(family)
        assert grid.get("n_estimators") == estimators
        assert len(list(runner.benchmark_base.ParameterGrid(grid))) == 16
    assert "reg_lambda" not in runner.benchmark_base.get_baseline_grid("XGBoost")
    assert "reg_lambda" not in runner.benchmark_base.get_baseline_grid("LightGBM")


@pytest.mark.parametrize("model_name", ["XGB standard", "LightGBM standard"])
def test_rotating_boosted_baselines_use_validation_early_stopping(model_name: str) -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=4,
        random_state=11,
    )
    X = pd.DataFrame(X)
    family = "XGBoost" if model_name == "XGB standard" else "LightGBM"
    dependency = (
        runner.benchmark_base.XGBClassifier
        if model_name == "XGB standard"
        else runner.benchmark_base.LGBMClassifier
    )
    if dependency is None:
        pytest.skip(f"{family} is available through the benchmark dependency group")
    grid = runner.rotating_baseline_grid(family)
    candidate = next(iter(runner.benchmark_base.ParameterGrid(grid)))
    _, builder, _, _ = runner.benchmark_base.get_model_spec(model_name)
    estimator = runner.baseline_pipeline(builder(candidate))
    fitted, metadata = runner._fit_baseline_with_rotating_validation(
        model_name,
        estimator,
        X.iloc[:80],
        y[:80],
        X.iloc[80:],
        y[80:],
    )
    assert metadata["baseline_early_stopping_used"] is True
    assert 1 <= metadata["baseline_selected_n_estimators"] <= 200
    assert fitted.predict_proba(X.iloc[80:]).shape[0] == 40


def test_pair_aggregation_preserves_metrics_timing_parameters_and_instance_moments() -> None:
    base = {
        "status": "ok",
        "error_count": 0,
        "repeat": 0,
        "sample": 0,
        "validation_protocol": "tabarena",
        "accuracy": 0.75,
        "balanced_accuracy": 0.75,
        "roc_auc": 0.8,
        "log_loss": 0.4,
        "avg_precision": 0.78,
        "brier": 0.18,
        "f1": 0.7,
        "best_inner_score": 0.79,
        "complexity": 12.0,
        "complexity_model_units": 10.0,
        "complexity_model_inspection_units": 12.0,
        "fit_ms": 10.0,
        "tune_ms": 100.0,
        "predict_ms": 2.0,
        "complexity_ms": 3.0,
        "fit_seconds": 0.01,
        "tune_seconds": 0.1,
        "predict_seconds": 0.002,
        "complexity_seconds": 0.003,
        "split_seconds": 0.12,
        "phase_accounted_seconds": 0.105,
        "evaluation_overhead_seconds": 0.015,
        "fit_time_included_in_tuning": True,
        "inner_scoring": "roc_auc",
        "selection_refit_performed": False,
        "retained_cv_ensemble": True,
        "cv_ensemble_child_count": 8,
        "candidate_fit_count": 128,
        "complexity_instance_inspection_units_confidence_level": 0.95,
    }
    split_rows = [
        {
            **base,
            "split_id": 0,
            "fold": 0,
            "best_params": '{"C": 0.5}',
            "best_params_json": '{"C": 0.5}',
            "complexity_instance_inspection_units_n_samples": 2,
            "complexity_instance_inspection_units_sum": 4.0,
            "complexity_instance_inspection_units_sum_squares": 10.0,
            "complexity_instance_inspection_units_min": 1,
            "complexity_instance_inspection_units_max": 3,
        },
        {
            **base,
            "split_id": 1,
            "fold": 1,
            "best_params": '{"C": 1.0}',
            "best_params_json": '{"C": 1.0}',
            "complexity_instance_inspection_units_n_samples": 2,
            "complexity_instance_inspection_units_sum": 6.0,
            "complexity_instance_inspection_units_sum_squares": 20.0,
            "complexity_instance_inspection_units_min": 2,
            "complexity_instance_inspection_units_max": 4,
        },
    ]
    result = runner.aggregate_pair_rows(
        split_rows,
        split_predictions={
            0: {"y_true": [0, 1], "proba": [[0.8, 0.2], [0.2, 0.8]]},
            1: {"y_true": [0, 1], "proba": [[0.7, 0.3], [0.1, 0.9]]},
        },
        task_meta={
            "task_id": 1,
            "dataset_id": 2,
            "dataset_name": "aggregate",
            "n_rows": 4,
            "n_features": 1,
            "n_classes": 2,
            "class_labels": [0, 1],
            "categorical_features": [],
            "target_name": "target",
            "official_repeats": 1,
            "official_folds": 2,
            "official_samples": 1,
        },
        model_name="Logistic Regression",
        model_feature_counts=[1, 1],
        tune=True,
        inner_splits=8,
        hugiml_scenario="augmented_pair",
        official_splits_expected=2,
    )

    assert result["status"] == "ok"
    assert result["log_loss"] == pytest.approx(0.4)
    assert result["total_tune_seconds"] == pytest.approx(0.2)
    assert result["total_complexity_seconds"] == pytest.approx(0.006)
    assert result["total_phase_accounted_seconds"] == pytest.approx(0.21)
    assert result["complexity_instance_inspection_units_n_samples"] == 4
    assert result["complexity_instance_inspection_units_mean"] == pytest.approx(2.5)
    assert result["complexity_instance_inspection_units_min"] == 1
    assert result["complexity_instance_inspection_units_max"] == 4
    assert result["best_params_unique_count"] == 2
    assert result["candidate_fit_count_total"] == 256
    assert result["candidate_fit_count"] == 128
    assert result["retained_cv_ensemble_split_count"] == 2
    assert result["retained_cv_ensemble"] is True
    assert result["cv_ensemble_child_count"] == 8
    assert result["inner_scoring"] == "roc_auc"
    assert result["scoring"] == "roc_auc_binary_or_multiclass_log_loss"

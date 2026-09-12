from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from joblib.externals import loky

from experiments.benchmark import benchmark_engine


def test_dataset_boundary_cleanup_precedes_memory_remeasurement(monkeypatch) -> None:
    calls: list[object] = []

    class _Executor:
        def shutdown(self, *, wait, kill_workers):
            calls.append(("shutdown", wait, kill_workers))

    available = iter([12.0, 19.5])
    monkeypatch.setattr(benchmark_engine, "_available_memory_gb", lambda: next(available))
    monkeypatch.setattr(loky, "get_reusable_executor", lambda: _Executor())
    monkeypatch.setattr(benchmark_engine.gc, "collect", lambda: calls.append("gc") or 7)

    result = benchmark_engine._flush_dataset_boundary_memory()

    assert calls == [("shutdown", True, True), "gc"]
    assert result["executor_shutdown"] is True
    assert result["collected_objects"] == 7
    assert result["available_memory_before_gb"] == 12.0
    assert result["available_memory_after_gb"] == 19.5
    assert result["available_memory_change_gb"] == 7.5


def test_parallelism_selection_respects_cpu_memory_and_all_cores() -> None:
    assert benchmark_engine._select_tabarena_parallelism(
        requested_outer_jobs=8,
        requested_inner_jobs=3,
        fallback_outer_jobs=6,
        available_memory_gb=40.0,
        logical_processors=24,
    )[:2] == (8, 3)
    assert benchmark_engine._select_tabarena_parallelism(
        requested_outer_jobs=3,
        requested_inner_jobs=8,
        fallback_outer_jobs=2,
        available_memory_gb=40.0,
        logical_processors=24,
    )[:2] == (3, 8)
    outer, inner, reasons = benchmark_engine._select_tabarena_parallelism(
        requested_outer_jobs=3,
        requested_inner_jobs=8,
        fallback_outer_jobs=2,
        available_memory_gb=25.0,
        logical_processors=24,
    )
    assert (outer, inner) == (3, 8)
    assert not reasons
    assert benchmark_engine._select_tabarena_parallelism(
        requested_outer_jobs=4,
        requested_inner_jobs=8,
        fallback_outer_jobs=None,
        available_memory_gb=100.0,
        logical_processors=24,
    )[:2] == (3, 8)
    assert benchmark_engine._select_tabarena_parallelism(
        requested_outer_jobs=3,
        requested_inner_jobs=-1,
        fallback_outer_jobs=2,
        available_memory_gb=40.0,
        logical_processors=24,
    )[:2] == (3, 8)


def test_parallelism_selection_reduces_outer_jobs_and_preserves_inner_pool() -> None:
    outer, inner, reasons = benchmark_engine._select_tabarena_parallelism(
        requested_outer_jobs=3,
        requested_inner_jobs=8,
        fallback_outer_jobs=2,
        available_memory_gb=18.0,
        logical_processors=24,
        estimated_memory_per_outer_job_gb=8.0,
    )
    assert (outer, inner) == (1, 8)
    assert reasons


def test_memory_estimate_accounts_for_high_cardinality_categories() -> None:
    rows = 1000
    low_cardinality = pd.DataFrame({"category": pd.Categorical(np.arange(rows) % 2)})
    high_cardinality = pd.DataFrame({"category": pd.Categorical(np.arange(rows))})
    y = np.arange(rows) % 2

    _, low_estimate = benchmark_engine._dataset_worker_memory_estimate_gb(
        low_cardinality, y
    )
    _, high_estimate = benchmark_engine._dataset_worker_memory_estimate_gb(
        high_cardinality, y
    )

    assert low_estimate >= 1.0
    assert high_estimate > low_estimate


def test_hugiml_memory_estimate_uses_sparse_categorical_storage() -> None:
    rows = 20_000
    categories = 4_000
    X = pd.DataFrame(
        {
            "category_a": pd.Categorical(np.arange(rows) % categories),
            "category_b": pd.Categorical((np.arange(rows) * 7) % categories),
            "numeric": np.arange(rows, dtype=np.float32),
        }
    )
    y = np.arange(rows) % 2

    _, dense_estimate = benchmark_engine._dataset_worker_memory_estimate_gb(X, y)
    _, sparse_estimate = benchmark_engine._dataset_worker_memory_estimate_gb(
        X, y, model_name="HUGIML"
    )

    assert sparse_estimate >= 1.0
    assert sparse_estimate < dense_estimate


def test_hugiml_sparse_estimate_can_retain_full_parallel_allocation() -> None:
    rows = 50_000
    categories = 8_000
    X = pd.DataFrame(
        {
            "category_a": pd.Categorical(np.arange(rows) % categories),
            "category_b": pd.Categorical((np.arange(rows) * 11) % categories),
        }
    )
    y = np.arange(rows) % 2
    _, estimate = benchmark_engine._dataset_worker_memory_estimate_gb(
        X, y, model_name="HUGIML"
    )

    outer, inner, reasons = benchmark_engine._select_tabarena_parallelism(
        requested_outer_jobs=3,
        requested_inner_jobs=8,
        fallback_outer_jobs=2,
        available_memory_gb=20.0,
        logical_processors=24,
        estimated_memory_per_outer_job_gb=estimate,
    )

    assert (outer, inner) == (3, 8)
    assert not reasons


def _synthetic_task():
    rng = np.random.default_rng(77)
    y = np.tile(np.array([0, 1], dtype=int), 45)
    X = pd.DataFrame(
        {
            "signal": y + rng.normal(scale=0.8, size=len(y)),
            "noise": rng.normal(size=len(y)),
        }
    )
    task_meta = {
        "task_id": 900001,
        "dataset_id": 900002,
        "dataset_name": "synthetic_outer_parallel",
        "n_rows": len(X),
        "n_features": X.shape[1],
        "n_classes": 2,
        "official_repeats": 1,
        "official_folds": 3,
        "official_samples": 1,
        "sha256": {"synthetic": "yes"},
    }
    splits = []
    indices = np.arange(len(y))
    for fold in range(3):
        test_idx = indices[fold::3]
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
        splits.append(
            {
                "split_id": fold,
                "repeat": 0,
                "fold": fold,
                "sample": 0,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
        )
    return X, y, task_meta, splits


def _force_parallel_capacity(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_engine, "_available_memory_gb", lambda: 64.0)
    monkeypatch.setattr(benchmark_engine.os, "cpu_count", lambda: 24)


def test_three_outer_folds_match_serial_and_checkpoint_in_order(tmp_path, monkeypatch) -> None:
    _force_parallel_capacity(monkeypatch)
    X, y, task_meta, splits = _synthetic_task()
    monkeypatch.setattr(
        benchmark_engine,
        "load_task",
        lambda task_id, retries: (object(), X, y, task_meta),
    )
    monkeypatch.setattr(
        benchmark_engine,
        "official_splits",
        lambda task, max_splits=None: list(splits),
    )
    common = {
        "task_id": task_meta["task_id"],
        "model_name": "Logistic Regression",
        "retries": 0,
        "tune": False,
        "inner_splits": 3,
        "random_state": 42,
        "max_official_splits": None,
        "hugiml_scenario": "augmented_pair",
        "hugiml_max_fit_seconds": None,
        "run_id": "same-run",
        "resume": False,
        "validation_protocol": "tabarena",
        "early_stopping": True,
        "inner_jobs": 2,
    }
    serial = benchmark_engine.run_pair(
        **common, out_dir=tmp_path / "serial", outer_jobs=1
    )
    parallel = benchmark_engine.run_pair(
        **common, out_dir=tmp_path / "parallel", outer_jobs=3
    )

    assert parallel["roc_auc"] == pytest.approx(serial["roc_auc"], abs=1e-12)
    assert parallel["balanced_accuracy"] == pytest.approx(
        serial["balanced_accuracy"], abs=1e-12
    )
    assert [row["split_id"] for row in parallel["split_rows"]] == [0, 1, 2]
    assert all(row["outer_jobs_effective"] == 3 for row in parallel["split_rows"])
    assert all(row["inner_jobs_effective"] == 2 for row in parallel["split_rows"])
    assert all(row["outer_jobs_scheduled"] == 3 for row in parallel["split_rows"])
    assert all(row["outer_executor_backend"] == "loky" for row in parallel["split_rows"])
    assert all(row["parallel_worker_budget"] == 6 for row in parallel["split_rows"])
    assert parallel["outer_jobs_scheduled_max"] == 3
    assert parallel["parallel_worker_budget_max"] == 6
    assert parallel["outer_executor_backends"] == ["loky"]
    assert parallel["outer_parallel_grouping"] == "official_repeat"


def test_outer_worker_uses_and_restores_lr_source_policy(monkeypatch) -> None:
    observed = []

    def evaluate(**kwargs):
        observed.append(benchmark_engine.BENCHMARK_LR_SOURCE_POLICY)
        return {"status": "ok"}, None, None

    monkeypatch.setattr(benchmark_engine, "evaluate_official_split", evaluate)
    monkeypatch.setattr(benchmark_engine, "BENCHMARK_LR_SOURCE_POLICY", "strict")

    split = {"split_id": 0}
    returned = benchmark_engine._evaluate_official_split_with_worker_budget(
        split=split,
        inner_jobs=8,
        common_kwargs={},
        suite_context={
            "nested_cv_ensemble_folds": None,
            "nested_early_stopping": False,
            "tabarena_reporting": False,
            "lr_source_policy": "main_effect",
        },
    )

    assert returned == (split, {"status": "ok"}, None, None)
    assert observed == ["main_effect"]
    assert benchmark_engine.BENCHMARK_LR_SOURCE_POLICY == "strict"


def test_parallel_resume_reuses_completed_outer_split(tmp_path, monkeypatch) -> None:
    X, y, task_meta, splits = _synthetic_task()
    exposed_splits = list(splits)
    monkeypatch.setattr(
        benchmark_engine,
        "load_task",
        lambda task_id, retries: (object(), X, y, task_meta),
    )
    monkeypatch.setattr(
        benchmark_engine,
        "official_splits",
        lambda task, max_splits=None: list(exposed_splits),
    )
    common = {
        "task_id": task_meta["task_id"],
        "model_name": "Logistic Regression",
        "out_dir": tmp_path / "resume",
        "retries": 0,
        "tune": False,
        "inner_splits": 3,
        "random_state": 42,
        "max_official_splits": None,
        "hugiml_scenario": "augmented_pair",
        "hugiml_max_fit_seconds": None,
        "run_id": "resume-run",
        "validation_protocol": "tabarena",
        "early_stopping": True,
        "inner_jobs": 2,
    }
    original_evaluate = benchmark_engine.evaluate_official_split
    calls = 0

    def interrupt_after_one(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original_evaluate(**kwargs)

    monkeypatch.setattr(
        benchmark_engine, "evaluate_official_split", interrupt_after_one
    )
    with pytest.raises(KeyboardInterrupt):
        benchmark_engine.run_pair(**common, resume=False, outer_jobs=1)

    monkeypatch.setattr(
        benchmark_engine, "evaluate_official_split", original_evaluate
    )
    resumed = benchmark_engine.run_pair(**common, resume=True, outer_jobs=3)
    assert resumed["official_splits_successful"] == 3
    assert [row["split_id"] for row in resumed["split_rows"]] == [0, 1, 2]
    assert resumed["split_rows"][0]["attempt_count"] == 1


def test_parallel_worker_termination_commits_completed_split_and_closes_pool(
    tmp_path, monkeypatch
) -> None:
    _force_parallel_capacity(monkeypatch)
    X, y, task_meta, splits = _synthetic_task()
    monkeypatch.setattr(
        benchmark_engine,
        "load_task",
        lambda task_id, retries: (object(), X, y, task_meta),
    )
    monkeypatch.setattr(
        benchmark_engine,
        "official_splits",
        lambda task, max_splits=None: list(splits),
    )
    closed: list[bool] = []
    monkeypatch.setattr(
        benchmark_engine,
        "_shutdown_parallel_workers",
        lambda *, wait: closed.append(wait) or None,
    )

    class _InterruptedParallel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, jobs):
            list(jobs)

            def results():
                yield (
                    splits[0],
                    {
                        "split_id": 0,
                        "repeat": 0,
                        "fold": 0,
                        "sample": 0,
                        "status": "ok",
                        "roc_auc": 0.75,
                        "balanced_accuracy": 0.7,
                        "last_error": None,
                    },
                    2,
                    {"y_true": [0, 1], "y_score": [[0.8, 0.2], [0.2, 0.8]]},
                )
                raise RuntimeError("worker process exited")

            return results()

    monkeypatch.setattr(benchmark_engine, "Parallel", _InterruptedParallel)
    with pytest.raises(RuntimeError, match="Parallel outer evaluation stopped"):
        benchmark_engine.run_pair(
            task_id=task_meta["task_id"],
            model_name="Logistic Regression",
            out_dir=tmp_path,
            retries=0,
            tune=False,
            inner_splits=3,
            random_state=42,
            max_official_splits=None,
            hugiml_scenario="augmented_pair",
            hugiml_max_fit_seconds=None,
            run_id="terminated-worker",
            resume=False,
            validation_protocol="tabarena",
            early_stopping=True,
            outer_jobs=3,
            inner_jobs=2,
        )

    assert closed == [False]
    checkpoint = next((tmp_path / "split_checkpoints").glob("*.json"))
    saved = benchmark_engine.read_json_with_backup(checkpoint)
    assert [row["split_id"] for row in saved["split_rows"]] == [0]
    assert saved["split_rows"][0]["status"] == "ok"


def test_rotating_outer_parallelism_matches_serial_results(tmp_path, monkeypatch) -> None:
    _force_parallel_capacity(monkeypatch)
    X, y, task_meta, splits = _synthetic_task()
    monkeypatch.setattr(
        benchmark_engine,
        "load_task",
        lambda task_id, retries: (object(), X, y, task_meta),
    )
    monkeypatch.setattr(
        benchmark_engine,
        "official_splits",
        lambda task, max_splits=None: list(splits),
    )
    common = {
        "task_id": task_meta["task_id"],
        "model_name": "Logistic Regression",
        "retries": 0,
        "tune": False,
        "inner_splits": 3,
        "random_state": 42,
        "max_official_splits": None,
        "hugiml_scenario": "augmented_pair",
        "hugiml_max_fit_seconds": None,
        "run_id": "rotating-run",
        "resume": False,
        "validation_protocol": "rotating",
        "early_stopping": True,
        "inner_jobs": 2,
    }
    serial = benchmark_engine.run_pair(
        **common, out_dir=tmp_path / "rotating_serial", outer_jobs=1
    )
    parallel = benchmark_engine.run_pair(
        **common, out_dir=tmp_path / "rotating_parallel", outer_jobs=3
    )

    assert parallel["roc_auc"] == pytest.approx(serial["roc_auc"], abs=1e-12)
    assert parallel["balanced_accuracy"] == pytest.approx(
        serial["balanced_accuracy"], abs=1e-12
    )
    assert [row["split_id"] for row in parallel["split_rows"]] == [0, 1, 2]
    assert all(row["outer_jobs_effective"] == 3 for row in parallel["split_rows"])


def test_nested_outer_parallelism_matches_serial_results(tmp_path, monkeypatch) -> None:
    _force_parallel_capacity(monkeypatch)
    X, y, task_meta, splits = _synthetic_task()
    monkeypatch.setattr(
        benchmark_engine,
        "load_task",
        lambda task_id, retries: (object(), X, y, task_meta),
    )
    monkeypatch.setattr(
        benchmark_engine,
        "official_splits",
        lambda task, max_splits=None: list(splits),
    )
    common = {
        "task_id": task_meta["task_id"],
        "model_name": "Logistic Regression",
        "retries": 0,
        "tune": False,
        "inner_splits": 3,
        "random_state": 42,
        "max_official_splits": None,
        "hugiml_scenario": "augmented_pair",
        "hugiml_max_fit_seconds": None,
        "run_id": "nested-run",
        "resume": False,
        "validation_protocol": "nested",
        "early_stopping": False,
        "inner_jobs": 3,
    }
    serial = benchmark_engine.run_pair(
        **common, out_dir=tmp_path / "nested_serial", outer_jobs=1
    )
    parallel = benchmark_engine.run_pair(
        **common, out_dir=tmp_path / "nested_parallel", outer_jobs=3
    )

    assert parallel["roc_auc"] == pytest.approx(serial["roc_auc"], abs=1e-12)
    assert parallel["balanced_accuracy"] == pytest.approx(
        serial["balanced_accuracy"], abs=1e-12
    )
    assert [row["split_id"] for row in parallel["split_rows"]] == [0, 1, 2]
    assert all(row["outer_jobs_effective"] == 3 for row in parallel["split_rows"])


def test_clean_checkpoint_json_keeps_metrics_and_resume_state(tmp_path, monkeypatch) -> None:
    target = tmp_path / "results.json"
    monkeypatch.setattr(benchmark_engine, "CLEAN_OUTPUT_JSON", True)
    benchmark_engine.atomic_write_json(
        target,
        {
            "active_run_id": "active",
            "runs": {
                "active": {"run_id": "active", "updated_at_utc": "removed"},
                "inactive": {"run_id": "inactive"},
            },
            "source": {"package_version": "removed"},
            "results": [{"status": "ok", "roc_auc": 0.875, "attempt_count": 1}],
            "machine_location": str(tmp_path / "data"),
        },
    )
    saved = benchmark_engine.read_json_with_backup(target)

    assert list(saved["runs"]) == ["active"]
    assert saved["results"][0] == {
        "status": "ok",
        "roc_auc": 0.875,
        "attempt_count": 1,
    }
    assert "source" not in saved
    assert "updated_at_utc" not in saved["runs"]["active"]
    assert saved["machine_location"] is None


def test_clean_rotating_checkpoint_continues_completed_split(tmp_path, monkeypatch) -> None:
    X, y, task_meta, splits = _synthetic_task()
    exposed_splits = list(splits)
    monkeypatch.setattr(benchmark_engine, "CLEAN_OUTPUT_JSON", True)
    monkeypatch.setattr(
        benchmark_engine,
        "load_task",
        lambda task_id, retries: (object(), X, y, task_meta),
    )
    monkeypatch.setattr(
        benchmark_engine,
        "official_splits",
        lambda task, max_splits=None: list(exposed_splits),
    )
    common = {
        "task_id": task_meta["task_id"],
        "model_name": "Logistic Regression",
        "out_dir": tmp_path / "clean_rotating_resume",
        "retries": 0,
        "tune": False,
        "inner_splits": 3,
        "random_state": 42,
        "max_official_splits": None,
        "hugiml_scenario": "augmented_pair",
        "hugiml_max_fit_seconds": None,
        "run_id": "clean-rotating-run",
        "validation_protocol": "rotating",
        "early_stopping": True,
        "inner_jobs": 2,
    }
    original_evaluate = benchmark_engine.evaluate_official_split
    calls = 0

    def interrupt_after_one(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original_evaluate(**kwargs)

    monkeypatch.setattr(
        benchmark_engine, "evaluate_official_split", interrupt_after_one
    )
    with pytest.raises(KeyboardInterrupt):
        benchmark_engine.run_pair(**common, resume=False, outer_jobs=1)

    monkeypatch.setattr(
        benchmark_engine, "evaluate_official_split", original_evaluate
    )
    resumed = benchmark_engine.run_pair(**common, resume=True, outer_jobs=3)
    assert resumed["official_splits_successful"] == 3
    assert resumed["split_rows"][0]["attempt_count"] == 1

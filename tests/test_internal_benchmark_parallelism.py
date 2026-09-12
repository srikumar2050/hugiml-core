from __future__ import annotations

import errno
import importlib.util
import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from experiments.benchmark import benchmark_dashboard


@contextmanager
def _replacement_failures(failures, failure):
    original = benchmark_dashboard.os.replace
    calls = []

    def replace(source, destination):
        calls.append((Path(source), Path(destination)))
        if len(calls) <= failures:
            raise failure
        return original(source, destination)

    benchmark_dashboard.os.replace = replace
    try:
        yield calls
    finally:
        benchmark_dashboard.os.replace = original


def test_checkpoint_replacement_retries_transient_access(tmp_path):
    destination = tmp_path / "checkpoint.json"
    benchmark_dashboard.save_checkpoint(destination, {"results": [1]})
    with _replacement_failures(2, PermissionError(errno.EACCES, "Busy")) as calls:
        benchmark_dashboard.save_checkpoint(destination, {"results": [1, 2]})
    assert len(calls) == 3
    assert json.loads(destination.read_text()) == {"results": [1, 2]}
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_persistent_access_preserves_both_files(tmp_path):
    destination = tmp_path / "checkpoint.json"
    temporary = tmp_path / "checkpoint.pending.tmp"
    destination.write_text('{"results": [1]}')
    temporary.write_text('{"results": [1, 2]}')
    with _replacement_failures(10, PermissionError(errno.EACCES, "Busy")) as calls:
        with pytest.raises(PermissionError):
            benchmark_dashboard._replace_checkpoint_file(temporary, destination, attempts=3, delay_seconds=0)
    assert len(calls) == 3
    assert json.loads(destination.read_text())["results"] == [1]
    assert json.loads(temporary.read_text())["results"] == [1, 2]


def test_checkpoint_non_lock_failure_is_not_retried(tmp_path):
    with _replacement_failures(10, FileNotFoundError(errno.ENOENT, "Absent")) as calls:
        with pytest.raises(FileNotFoundError):
            benchmark_dashboard._replace_checkpoint_file(tmp_path / "source", tmp_path / "target", delay_seconds=0)
    assert len(calls) == 1


def _binary_frame() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(41)
    y = np.tile(np.array([0, 1], dtype=int), 36)
    X = pd.DataFrame(
        {
            "signal": y + rng.normal(scale=0.7, size=len(y)),
            "noise": rng.normal(size=len(y)),
        }
    )
    return X, y


def _lr_builder(params):
    return LogisticRegression(
        C=float(params.get("C", 1.0)),
        solver="liblinear",
        random_state=17,
        max_iter=500,
    )


def test_internal_gridsearch_parallel_matches_serial() -> None:
    X, y = _binary_frame()
    kwargs = {
        "candidates": [{"C": 0.25}, {"C": 1.0}, {"C": 4.0}],
        "builder": _lr_builder,
        "X_tr": X,
        "y_tr": y,
        "inner_splits": 3,
        "random_state": 23,
    }
    serial = benchmark_dashboard._tune_pipeline_gridsearch(**kwargs, n_jobs=1)
    parallel = benchmark_dashboard._tune_pipeline_gridsearch(**kwargs, n_jobs=8)

    assert parallel[1] == serial[1]
    assert parallel[2] == pytest.approx(serial[2], abs=1e-12)
    np.testing.assert_allclose(
        parallel[0].predict_proba(X), serial[0].predict_proba(X), atol=1e-12
    )


def test_internal_budgeted_parallel_matches_serial_and_refits_once() -> None:
    X, y = _binary_frame()

    def complexity(_model):
        return 1.0

    kwargs = {
        "candidates": [{"C": 0.25}, {"C": 1.0}, {"C": 4.0}],
        "builder": _lr_builder,
        "complexity_fn": complexity,
        "X_tr": X,
        "y_tr": y,
        "budget": 2.0,
        "inner_splits": 3,
        "random_state": 23,
    }
    serial = benchmark_dashboard._tune_budgeted_pipeline_inner_cv(**kwargs, n_jobs=1)
    parallel = benchmark_dashboard._tune_budgeted_pipeline_inner_cv(**kwargs, n_jobs=8)

    assert parallel[1] == serial[1]
    assert parallel[2] == pytest.approx(serial[2], abs=1e-12)
    np.testing.assert_allclose(
        parallel[0].predict_proba(X), serial[0].predict_proba(X), atol=1e-12
    )
    assert parallel[4]["_final_refit_ms"] >= 0.0


@pytest.mark.parametrize("model, expected", [("HUGIML", (5, 3)), ("XGB standard", (5, 3)),
                                            ("XGB complexity-budgeted", (5, 3))])
def test_outer_inner_layout_respects_cpu_budget(model, expected):
    X, _ = _binary_frame()
    layout = benchmark_dashboard._parallel_layout(
        X, model, [{}], outer_jobs=5, n_jobs=4, n_splits=5, inner_splits=3,
        tune=True, available_bytes=64 * 1024**3, processors=24,
    )
    assert (layout["outer_jobs"], layout["n_jobs"]) == expected
    assert layout["outer_jobs"] * layout["n_jobs"] <= 24


def test_layout_reduces_workers_for_memory_and_cpu():
    X, _ = _binary_frame()
    for ram in (0, 1024**3):
        layout = benchmark_dashboard._parallel_layout(
            X, "HUGIML", [{}], outer_jobs=5, n_jobs=8, n_splits=5, inner_splits=3,
            tune=True, available_bytes=ram, processors=2,
        )
        assert layout["outer_jobs"] == layout["n_jobs"] == 1


@contextmanager
def _small_dataset_and_grid(model):
    loader = benchmark_dashboard.load_dataset
    spec = benchmark_dashboard.get_model_spec

    def small_spec(*args, **kwargs):
        grid, builder, complexity, budget = spec(*args, **kwargs)
        return grid[:2], builder, complexity, budget

    benchmark_dashboard.load_dataset = lambda _: (*_binary_frame(), "Synthetic")
    benchmark_dashboard.get_model_spec = small_spec
    try:
        yield
    finally:
        benchmark_dashboard.load_dataset = loader
        benchmark_dashboard.get_model_spec = spec


@pytest.mark.parametrize("model", [
    "HUGIML", "XGB standard", "XGB complexity-budgeted", "LightGBM standard",
    "LightGBM complexity-budgeted", "RandomForest standard", "RandomForest complexity-budgeted",
    "EBM", "RuleFit",
])
def test_outer_parallel_metrics_and_partial_resume(tmp_path, model):
    if model.startswith("XGB"):
        pytest.importorskip("xgboost")
    if model.startswith("LightGBM"):
        pytest.importorskip("lightgbm")
    if model == "EBM" and importlib.util.find_spec("interpret") is None:
        pytest.skip("EBM benchmark dependency is not installed")
    if model == "RuleFit" and importlib.util.find_spec("imodels") is None:
        pytest.skip("RuleFit benchmark dependency is not installed")
    kwargs = dict(dataset="SynthTest", model=model, n_splits=3, inner_splits=2,
                  lr_source_policy="main_effect", hugiml_scenario="augmented_pair", random_state=17)
    with _small_dataset_and_grid(model):
        serial = benchmark_dashboard.run_pair(**kwargs, n_jobs=1, outer_jobs=1)
        parallel = benchmark_dashboard.run_pair(
            **kwargs, n_jobs=2, outer_jobs=2, fold_checkpoint_dir=tmp_path,
        )
        assert serial["status"] == parallel["status"] == "ok"
        for metric in ("roc_auc", "accuracy", "balanced_accuracy", "f1", "brier",
                       "avg_precision", "complexity_model_inspection_units"):
            assert parallel[metric] == pytest.approx(serial[metric], abs=1e-10)
        checkpoint = next(tmp_path.glob("*.json"))
        saved = json.loads(checkpoint.read_text())
        assert saved["identity"]["lr_source_policy"] == "main_effect"
        assert len(saved["fold_rows"]) == 3
        retained = saved["fold_rows"][0]
        saved["fold_rows"] = [retained]
        checkpoint.write_text(json.dumps(saved))
        resumed = benchmark_dashboard.run_pair(
            **kwargs, n_jobs=2, outer_jobs=2, fold_checkpoint_dir=tmp_path,
        )
        assert resumed["status"] == "ok"
        assert resumed["timing_comparable"] is False
        assert resumed["roc_auc"] == pytest.approx(parallel["roc_auc"], abs=1e-12)
        final = json.loads(checkpoint.read_text())
        assert final["fold_rows"][0] == retained
        assert [r["fold"] for r in final["fold_rows"]] == [0, 1, 2]
        if model == "HUGIML":
            assert parallel["lr_source_policy"] == "main_effect"
            assert all(json.loads(r["best_params_json"])["lr_source_policy"] == "main_effect"
                       for r in final["fold_rows"])


def test_failed_pair_remains_pending():
    rows = [{"dataset": "d", "model": "HUGIML", "roc_auc": 0.8,
             "status": "error", "error_count": 1}]
    assert benchmark_dashboard.completed_keys({"results": rows}) == set()


def test_cli_policy_and_worker_resume_contract(tmp_path):
    args = ["--out-dir", str(tmp_path), "--max-pairs", "0", "--models", "HUGIML",
            "--outer-jobs", "5", "--n-jobs", "4", "--lr-source-policy", "main_effect"]
    assert benchmark_dashboard.main(args) == 0
    payload = json.loads((tmp_path / "benchmark_checkpoint.json").read_text())
    assert payload["metadata"]["lr_source_policy"] == "main_effect"
    assert payload["metadata"]["outer_jobs"] == 5
    assert payload["metadata"]["expected_result_rows"] == 200
    assert list(payload["metadata"]["hugiml_dashboard_scenarios"]) == [
        "augmented_pair", "interaction_relaxed",
    ]
    (tmp_path / "fold_checkpoints").mkdir()
    assert benchmark_dashboard.main(args + ["--resume"]) == 0
    with pytest.raises(RuntimeError, match="lr_source_policy"):
        benchmark_dashboard.main(args[:-1] + ["strict", "--resume"])


def test_hugiml_source_policy_without_tuning(tmp_path):
    with _small_dataset_and_grid("HUGIML"):
        row = benchmark_dashboard.run_pair(
            "SynthTest", "HUGIML", hugiml_scenario="augmented_pair", n_splits=2,
            tune=False, lr_source_policy="main_effect", fold_checkpoint_dir=tmp_path,
        )
    assert row["status"] == "ok"
    assert row["lr_source_policy"] == "main_effect"


def test_budgeted_search_rejects_partial_candidate():
    X, y = _binary_frame()
    X = pd.concat([X, X.iloc[:2]], ignore_index=True)
    y = np.concatenate([y, y[:2]])

    class ConditionalLR(LogisticRegression):
        def fit(self, X, y, sample_weight=None):
            if self.C == 0.25 and len(X) == 49:
                raise ValueError("candidate unavailable")
            return super().fit(X, y, sample_weight=sample_weight)

    def unreliable(params):
        return ConditionalLR(C=params["C"], solver="liblinear", random_state=17, max_iter=500)
    result = benchmark_dashboard._tune_budgeted_pipeline_inner_cv(
        [{"C": 0.25}, {"C": 1.0}], unreliable, lambda _: 1.0, X, y,
        budget=2.0, inner_splits=3, random_state=23, n_jobs=2,
    )
    assert result[1] == {"C": 1.0}
    assert result[4]["budgeted_inner_cv_errors"] == 2
    with pytest.raises(RuntimeError, match="No budgeted candidate"):
        benchmark_dashboard._tune_budgeted_pipeline_inner_cv(
            [{"C": 0.25}], unreliable, lambda _: 1.0, X, y,
            budget=2.0, inner_splits=3, random_state=23, n_jobs=2,
        )


def test_common_layout_obeys_every_model_memory_limit():
    X, _ = _binary_frame()
    models = [m for m in benchmark_dashboard.MODEL_ORDER if m != "Logistic Regression"]
    layout = benchmark_dashboard._dataset_parallel_layout(
        X, models, ["augmented_pair", "interaction_relaxed"], n_splits=5,
        inner_splits=3, n_jobs=-1, outer_jobs=5, tune=True,
    )
    assert layout["n_jobs"] <= 3
    assert sum(layout["inner_jobs_by_slot"]) <= 15
    assert len(set(layout["inner_jobs_by_slot"])) == 1


def test_result_serialization_preserves_metrics_and_removes_diagnostics(tmp_path):
    local_reference = str((tmp_path / "private" / "file.py").resolve())
    payload = {"results": [{"roc_auc": 0.8, "fit_seconds": 2.0, "status": "error",
                           "error_count": 1, "hugiml_path": "lr", "last_error": "details",
                           "warnings": ["details"], "created_at": "calendar value",
                           "working_path": local_reference,
                           "best_params_json": json.dumps({"topK": 50, "stderr": "details"})}]}
    path = tmp_path / "results.json"
    benchmark_dashboard.save_checkpoint(path, payload)
    serialized = path.read_text()
    row = json.loads(serialized)["results"][0]
    assert row["fit_seconds"] == 2.0 and row["hugiml_path"] == "lr"
    assert row["error_count"] == 1 and row["status"] == "error"
    assert not set(row).intersection({"last_error", "warnings", "created_at"})
    assert local_reference not in serialized
    assert json.loads(row["best_params_json"]) == {"topK": 50}


def test_invalid_saved_auc_is_recomputed(tmp_path):
    pytest.importorskip("xgboost")
    with _small_dataset_and_grid("XGB standard"):
        kwargs = dict(dataset="SynthTest", model="XGB standard", n_splits=2, inner_splits=2,
                      fold_checkpoint_dir=tmp_path)
        benchmark_dashboard.run_pair(**kwargs)
        checkpoint = next(tmp_path.glob("*.json"))
        saved = json.loads(checkpoint.read_text())
        saved["fold_rows"][0]["roc_auc"] = None
        checkpoint.write_text(json.dumps(saved))
        row = benchmark_dashboard.run_pair(**kwargs)
        assert row["status"] == "ok" and row["resumed_outer_folds"] == 1
        assert json.loads(checkpoint.read_text())["fold_rows"][0]["roc_auc"] is not None


def test_selected_panel_assembly_and_global_state(tmp_path):
    pytest.importorskip("xgboost")
    b = benchmark_dashboard
    models = ["HUGIML", "XGB standard"]
    rows = []
    with _small_dataset_and_grid("HUGIML"):
        for model in models:
            rows.append(b.run_pair("BreastCancerOriginal", model, hugiml_scenario="augmented_pair",
                                   n_splits=2, inner_splits=2, lr_source_policy="main_effect"))
    payload = {"metadata": {"dataset_names": ["BreastCancerOriginal"], "model_order": models,
                            "hugiml_dashboard_scenarios": {"augmented_pair": b.HUGIML_SCENARIOS["augmented_pair"]},
                            "n_splits": 2, "inner_splits": 2, "lr_source_policy": "main_effect"},
               "results": rows}
    checkpoint = tmp_path / "input.json"
    b.save_checkpoint(checkpoint, payload)
    saved_order = list(b.MODEL_ORDER)
    paths = b.assemble_outputs(checkpoint, tmp_path / "out", tmp_path / "absent.html")
    assert b.MODEL_ORDER == saved_order
    assert paths["html"].exists()
    for key in ("checkpoint", "data"):
        text = paths[key].read_text()
        assert '"last_error"' not in text and '"warnings"' not in text
        assert str(tmp_path) not in text
    payload["results"].append(rows[0])
    b.save_checkpoint(checkpoint, payload)
    with pytest.raises(RuntimeError):
        b.assemble_outputs(checkpoint, tmp_path / "out", tmp_path / "absent.html")
    assert b.MODEL_ORDER == saved_order



def test_cli_uses_one_allocation_for_every_model(tmp_path):
    pytest.importorskip("xgboost")
    b = benchmark_dashboard
    args = ["--out-dir", str(tmp_path), "--datasets", "BreastCancerOriginal",
            "--models", "HUGIML,XGB standard", "--hugiml-scenarios", "interaction_relaxed",
            "--n-splits", "2", "--inner-splits", "2", "--outer-jobs", "2", "--n-jobs", "2",
            "--lr-source-policy", "main_effect", "--resume"]
    with _small_dataset_and_grid("HUGIML"):
        assert b.main(args) == 0
        path = tmp_path / "benchmark_checkpoint.json"
        payload = json.loads(path.read_text())
        rows = payload["results"]
        assert len(rows) == 2 and all(r["status"] == "ok" for r in rows)
        assert len({(r["outer_jobs"], r["n_jobs"]) for r in rows}) == 1
        assert all(r["pair_seconds"] > 0 and r["timing_comparable"] for r in rows)
        assert b.main(args) == 0
        assert json.loads(path.read_text())["results"] == rows
        with pytest.raises(RuntimeError, match="model_order"):
            b.main(args + ["--models", "HUGIML"])


def test_wall_time_summary_uses_same_datasets_for_all_models():
    b = benchmark_dashboard
    rows = []
    for dataset in ("BreastCancerOriginal", "IrisSetosa"):
        for index, model in enumerate(b.MODEL_ORDER):
            rows.append({"dataset": dataset, "model": model, "auc": 0.8 + index * 0.001,
                         "pair_seconds": 10.0 if dataset == "BreastCancerOriginal" else 100.0,
                         "fit_seconds": 1.0, "complexity_model_inspection_units": 2.0,
                         "timing_comparable": not (dataset == "IrisSetosa" and index == 0)})
    summary = b._summary_for_scope(pd.DataFrame(rows), "Overall")
    assert all(row["mean_pair_seconds"] == 10.0 for row in summary["rows"])
    assert all(row["timed_dataset_count"] == 1 for row in summary["rows"])


def test_cli_reports_unsuccessful_run_without_saved_diagnostic_text(tmp_path):
    b = benchmark_dashboard
    with _small_dataset_and_grid("XGB standard"):
        original = b.get_model_spec
        def unavailable_spec(*args, **kwargs):
            grid, _, complexity, budget = original(*args, **kwargs)
            def builder(params):
                raise ValueError("unavailable model")
            return grid, builder, complexity, budget
        b.get_model_spec = unavailable_spec
        try:
            status = b.main(["--out-dir", str(tmp_path), "--datasets", "BreastCancerOriginal",
                             "--models", "XGB standard", "--n-splits", "2", "--inner-splits", "2"])
        finally:
            b.get_model_spec = original
    assert status == 1
    text = (tmp_path / "benchmark_checkpoint.json").read_text()
    row = json.loads(text)["results"][0]
    assert row["status"] == "error" and row["error_count"] == 2
    assert "unavailable model" not in text and "last_error" not in text


def test_categorical_detection_without_deprecation():
    b = benchmark_dashboard
    categorical = pd.Series(["a", "b"], dtype="category")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert b.is_categorical_dtype(categorical)
        assert b.is_categorical_dtype(categorical.dtype)
        assert b.is_categorical_dtype(pd.Categorical(["a", "b"]))
        for series in (pd.Series([1, 2]), pd.Series([True, False]),
                       pd.Series(["a", "b"]), pd.Series(["a", "b"], dtype="string")):
            assert not b.is_categorical_dtype(series)


def test_dataset_progress_preserves_resume(tmp_path, capsys):
    pytest.importorskip("xgboost")
    b = benchmark_dashboard
    args = ["--out-dir", str(tmp_path), "--datasets", "BreastCancerOriginal,IrisSetosa",
            "--models", "XGB standard", "--n-splits", "2", "--inner-splits", "2", "--resume"]
    with _small_dataset_and_grid("XGB standard"):
        assert b.main(args) == 0
        output = capsys.readouterr().out
        assert "[1/2] run BreastCancerOriginal" in output
        assert "[2/2] run IrisSetosa" in output
        assert "[2/2] workers IrisSetosa" in output
        assert "[2/2] fold 2/2 IrisSetosa" in output
        assert b.main(args) == 0
        output = capsys.readouterr().out
        assert "[1/2] skip BreastCancerOriginal" in output
        assert "[2/2] skip IrisSetosa" in output
        assert "workers" not in output


def test_hugiml_inner_workers_receive_arrays_without_memory_mapping():
    from joblib import Parallel, delayed

    b = benchmark_dashboard
    observed = []
    original = b.HUGIMLClassifierNative

    class ArrayTuner:
        @staticmethod
        def tune(X, y, **kwargs):
            observed.extend(Parallel(n_jobs=2)(
                delayed(isinstance)(X, np.memmap) for _ in range(2)
            ))
            return SimpleNamespace(best_estimator_=None, best_params_={}, best_score_=0.5,
                                   refit_time_=0.0)

    b.HUGIMLClassifierNative = ArrayTuner
    try:
        X = np.ones((2000, 100), dtype=np.float64)
        b._tune_hugiml_inner_cv(
            [{"topK": 50}], X, np.tile([0, 1], 1000), inner_splits=2,
            random_state=42, hugiml_max_fit_seconds=None, n_jobs=2,
        )
    finally:
        b.HUGIMLClassifierNative = original
    assert observed == [False, False]


def test_separate_model_execution_retains_completed_results(tmp_path):
    pytest.importorskip("xgboost")
    b = benchmark_dashboard
    args = ["--out-dir", str(tmp_path), "--datasets", "BreastCancerOriginal",
            "--models", "HUGIML,XGB standard", "--hugiml-scenarios", "augmented_pair",
            "--n-splits", "2", "--inner-splits", "2", "--lr-source-policy", "main_effect",
            "--resume"]
    with _small_dataset_and_grid("HUGIML"):
        assert b.main(args + ["--execute-models", "XGB standard"]) == 0
        path = tmp_path / "benchmark_checkpoint.json"
        first = json.loads(path.read_text())
        assert len(first["results"]) == 1
        assert first["metadata"]["expected_result_rows"] == 2
        assert first["metadata"]["execution_result_rows"] == 1
        assert b.main(args + ["--execute-models", "HUGIML"]) == 0
        second = json.loads(path.read_text())
        assert len(second["results"]) == 2
        retained = next(r for r in second["results"] if r["model"] == "XGB standard")
        assert retained == first["results"][0]
        assert second["metadata"]["dataset_parallel_layouts"] == first["metadata"]["dataset_parallel_layouts"]


def test_rulefit_internal_cv_uses_training_class_counts(monkeypatch):
    pytest.importorskip("imodels")
    from imodels.util import score
    from sklearn.base import clone

    original = score.cross_val_score
    observed = []

    def record(estimator, X, y, **kwargs):
        observed.append(kwargs['cv'])
        return original(estimator, X, y, **kwargs)

    monkeypatch.setattr(score, 'cross_val_score', record)
    rng = np.random.default_rng(14)
    X = rng.normal(size=(8, 3))
    y = np.array([0, 1] * 4)
    model = benchmark_dashboard.SampleAwareRuleFitClassifier(
        n_estimators=5, max_rules=5, tree_size=3, random_state=42
    )
    clone(model).fit(X, y)
    assert observed and set(observed) == {4}
    assert score.cross_val_score is record


def test_rulefit_five_fold_behavior_matches_dependency():
    pytest.importorskip("imodels")
    rng = np.random.default_rng(15)
    X = rng.normal(size=(20, 3))
    y = np.array([0, 1] * 10)
    params = dict(n_estimators=5, max_rules=5, tree_size=3, random_state=42)
    adapted = benchmark_dashboard.SampleAwareRuleFitClassifier(**params).fit(X, y)
    reference = benchmark_dashboard.RuleFitClassifier(**params).fit(X, y)
    assert adapted.internal_cv_splits_ == 5
    np.testing.assert_allclose(adapted.predict_proba(X), reference.predict_proba(X), rtol=0, atol=0)

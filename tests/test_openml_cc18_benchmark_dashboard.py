from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _benchmark_module_stub() -> ModuleType:
    module = ModuleType("benchmark_dashboard")
    module.__spec__ = importlib.util.spec_from_loader("benchmark_dashboard", loader=None)
    module.HUGIML_SCENARIOS = {
        "augmented_pair": {"grid_name": "performance_ho"},
        "interaction_relaxed": {"grid_name": "interpretability_ho"},
    }
    module.ParameterGrid = ParameterGrid
    module._hugiml_pkg = SimpleNamespace(__version__="test")
    grids = {
        "XGBoost": {"n_estimators": [100], "max_depth": [3]},
        "LightGBM": {"n_estimators": [100], "num_leaves": [15]},
        "RandomForest": {"n_estimators": [200], "max_depth": [8]},
        "LogisticRegression": {"C": [0.1, 1.0], "penalty": ["l1", "l2"]},
        "EBM": {"learning_rate": [0.01], "interactions": [0]},
        "RuleFit": {"n_estimators": [50], "max_rules": [50]},
    }
    module.get_baseline_grid = lambda name: dict(grids[name])
    module.baseline_constant_parameters = lambda name: {"family": name}
    module.get_hugiml_grid = lambda name: {"grid_name": [name], "topK": [50]}
    module._hugiml_grid_for_scenario = lambda scenario: (
        module.HUGIML_SCENARIOS[scenario]["grid_name"],
        [{"topK": 50}],
    )
    module._force_writable_frame = lambda frame: pd.DataFrame(frame).copy(deep=True)

    def wrap_non_hugiml(model):
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        prep = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), make_column_selector(dtype_include=np.number)),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("ohe", encoder),
                        ]
                    ),
                    make_column_selector(dtype_exclude=np.number),
                ),
            ],
            verbose_feature_names_out=False,
        )
        return Pipeline([("prep", prep), ("model", model)])

    module._wrap_non_hugiml_pipeline = wrap_non_hugiml
    module._model_feature_count = lambda estimator, fallback: int(fallback)
    module.get_complexity_report = lambda estimator, X=None: {}
    return module


sys.modules["benchmark_dashboard"] = _benchmark_module_stub()

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "benchmark" / "run_openml_cc18_offline_benchmark.py"
BENCH_DIR = SCRIPT.parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

spec = importlib.util.spec_from_file_location("openml_cc18_benchmark_dashboard", SCRIPT)
assert spec and spec.loader
cc18 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cc18
spec.loader.exec_module(cc18)


class ArrayOnlyClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            raise TypeError("expected ndarray input")
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError("expected ndarray input")
        return np.full((len(X), 2), 0.5, dtype=float)


def test_baseline_pipeline_removes_generated_feature_labels() -> None:
    X = pd.DataFrame(
        {
            "checking_status": pd.Series(["<0", "0<=X<200", ">=200", "<0"], dtype="category"),
            "duration": [6, 12, 24, 36],
        }
    )
    y = np.array([0, 1, 1, 0])
    pipeline = cc18.baseline_pipeline(ArrayOnlyClassifier())
    pipeline.fit(X, y)
    assert "array" in pipeline.named_steps
    assert pipeline.predict_proba(X).shape == (4, 2)


def test_dense_array_transformer_accepts_sparse_matrices() -> None:
    matrix = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    transformed = cc18.DenseArrayTransformer().fit_transform(matrix)
    assert isinstance(transformed, np.ndarray)
    assert transformed.dtype == np.float32
    assert transformed.flags.c_contiguous
    assert transformed.flags.writeable
    np.testing.assert_array_equal(transformed, np.eye(2, dtype=np.float32))


def test_safe_jsonable_replaces_nonfinite_values() -> None:
    value = {
        "nan": float("nan"),
        "inf": np.float64("inf"),
        "nested": [np.int64(3), pd.NA, np.array([1.0, np.nan])],
    }
    cleaned = cc18.safe_jsonable(value)
    assert cleaned == {"nan": None, "inf": None, "nested": [3, None, [1.0, None]]}
    json.dumps(cleaned, allow_nan=False)


def test_model_selectors_support_aliases_families_and_all() -> None:
    assert cc18.parse_models("ensemble") == [
        "XGB standard",
        "LightGBM standard",
        "RandomForest standard",
    ]
    assert cc18.parse_models("interpretable") == [
        "Logistic Regression",
        "EBM",
        "RuleFit",
    ]
    assert cc18.parse_models("hugiml,lr,rulefit") == [
        "HUGIML",
        "Logistic Regression",
        "RuleFit",
    ]
    assert cc18.parse_models("all") == cc18.MODEL_ORDER


def test_grid_snapshot_includes_selected_interpretable_models() -> None:
    snapshot = cc18.grid_snapshot(
        "augmented_pair",
        ["HUGIML", "Logistic Regression", "EBM", "RuleFit"],
    )
    assert list(snapshot) == ["HUGIML", "Logistic Regression", "EBM", "RuleFit"]
    assert snapshot["Logistic Regression"]["candidate_count"] == 4


def test_resolve_suite_tasks_uses_all_by_default_and_supports_smallest() -> None:
    task_ids = list(range(101, 109))
    listing = pd.DataFrame(
        {
            "tid": task_ids,
            "did": list(range(201, 209)),
            "name": [f"data-{i}" for i in task_ids],
            "NumberOfInstances": [900, 120, 450, 80, 300, 60, 700, 200],
            "NumberOfFeatures": [5] * 8,
            "NumberOfClasses": [2] * 8,
        }
    )

    fake_openml = SimpleNamespace(
        study=SimpleNamespace(
            get_suite=lambda _suite: SimpleNamespace(id=99, name="OpenML-CC18", tasks=task_ids)
        ),
        tasks=SimpleNamespace(list_tasks=lambda **_kwargs: listing),
    )
    original = cc18.require_openml
    cc18.require_openml = lambda: fake_openml
    try:
        suite, selected = cc18.resolve_suite_tasks(
            "OpenML-CC18", smallest=None, explicit_task_ids=[], retries=0
        )
        _, smallest = cc18.resolve_suite_tasks(
            "OpenML-CC18", smallest=6, explicit_task_ids=[], retries=0
        )
    finally:
        cc18.require_openml = original

    assert suite["suite_task_count"] == 8
    assert [row["task_id"] for row in selected] == [106, 104, 102, 108, 105, 103, 107, 101]
    assert [row["task_id"] for row in smallest] == [106, 104, 102, 108, 105, 103]


def test_select_offline_entries_without_task_ids() -> None:
    entries = {
        10: {"task_id": 10, "n_rows": 500},
        20: {"task_id": 20, "n_rows": 100},
        30: {"task_id": 30, "n_rows": 300},
    }
    assert [row["task_id"] for row in cc18.select_offline_entries(
        entries, smallest=None, explicit_task_ids=[]
    )] == [20, 30, 10]
    assert [row["task_id"] for row in cc18.select_offline_entries(
        entries, smallest=2, explicit_task_ids=[]
    )] == [20, 30]
    assert [row["task_id"] for row in cc18.select_offline_entries(
        entries, smallest=None, explicit_task_ids=[30, 10]
    )] == [30, 10]


def test_official_splits_uses_task_defined_indices() -> None:
    calls: list[tuple[int, int, int]] = []

    class FakeTask:
        def get_split_dimensions(self):
            return 2, 2, 1

        def get_train_test_split_indices(self, *, fold, repeat, sample):
            calls.append((repeat, fold, sample))
            offset = repeat * 4 + fold * 2
            return np.array([offset, offset + 1]), np.array([offset + 2, offset + 3])

    splits = cc18.official_splits(FakeTask())
    assert calls == [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
    assert [(s["repeat"], s["fold"], s["sample"]) for s in splits] == calls
    for split in splits:
        cc18.validate_official_split(split, n_rows=10)


def _pair(
    task_id: int,
    dataset: str,
    model: str,
    auc: float,
    *,
    model_units: float = 20.0,
    inspection_units: float = 10.0,
    instance_units: float = 4.0,
) -> dict:
    return {
        "task_id": task_id,
        "dataset_id": task_id + 1000,
        "dataset": dataset,
        "model": model,
        "status": "ok",
        "n_rows": 100 + task_id,
        "raw_features": 5,
        "model_features": 6,
        "n_classes": 2,
        "roc_auc": auc,
        "balanced_accuracy": auc - 0.05,
        "f1": auc - 0.04,
        "fit_seconds": 0.2,
        "tune_seconds": 1.2,
        "predict_seconds": 0.01,
        "complexity": inspection_units,
        "complexity_model_units": model_units,
        "complexity_model_inspection_units": inspection_units,
        "complexity_instance_inspection_units_mean": instance_units,
        "complexity_instance_inspection_units_ci_lower": instance_units - 0.5,
        "complexity_instance_inspection_units_ci_upper": instance_units + 0.5,
        "official_splits_expected": 2,
        "official_splits_completed": 2,
        "official_splits_successful": 2,
        "error_count": 0,
        "split_rows": [
            {
                "task_id": task_id,
                "dataset": dataset,
                "model": model,
                "split_id": 0,
                "repeat": 0,
                "fold": 0,
                "sample": 0,
                "roc_auc": auc,
                "status": "ok",
            }
        ],
    }


def _activate(payload: dict, label: str, task_ids: list[int], models: list[str]) -> str:
    tasks = [
        {"task_id": task_id, "dataset_name": f"tiny-{task_id}", "n_rows": 100 + task_id, "n_classes": 2}
        for task_id in task_ids
    ]
    return cc18.activate_run(
        payload,
        configuration={"label": label, "task_ids": task_ids, "models": models},
        metadata={
            "models": models,
            "inner_splits": 3,
            "tune": True,
            "selection_description": f"{len(task_ids)} selected tasks",
            "grid_snapshot": {"HUGIML": {"candidate_count": 16}},
        },
        selected_tasks=tasks,
    )


def test_checkpoint_separates_configurations_and_only_completes_successes() -> None:
    payload = cc18.new_checkpoint()
    first = _activate(payload, "first", [1], ["HUGIML", "RandomForest standard"])
    complete = _pair(1, "tiny-1", "HUGIML", 0.82)
    complete["run_id"] = first
    partial = _pair(1, "tiny-1", "RandomForest standard", 0.79)
    partial.update(
        {
            "run_id": first,
            "status": "partial",
            "official_splits_successful": 1,
            "error_count": 1,
        }
    )
    payload["results"].extend([complete, partial])

    second = _activate(payload, "second", [1], ["HUGIML"])
    assert second != first
    assert len(payload["runs"]) == 2
    assert cc18.completed_pair_keys(payload, first) == {(1, "HUGIML")}
    assert cc18.run_view(payload, second)["results"] == []
    assert len(cc18.run_view(payload, first)["results"]) == 2


def test_failed_splits_are_retried_without_duplicate_rows(tmp_path: Path) -> None:
    class FakeTask:
        def get_split_dimensions(self):
            return 1, 2, 1

        def get_train_test_split_indices(self, *, fold, repeat, sample):
            if fold == 0:
                return np.array([0, 1]), np.array([2, 3])
            return np.array([2, 3]), np.array([0, 1])

    task_meta = {
        "task_id": 7,
        "dataset_id": 70,
        "dataset_name": "retry-data",
        "n_rows": 4,
        "n_features": 2,
        "n_classes": 2,
        "categorical_features": [],
        "class_labels": ["0", "1"],
        "target_name": "target",
        "official_repeats": 1,
        "official_folds": 2,
        "official_samples": 1,
        "sha256": {"features.parquet": "abc"},
    }
    X = pd.DataFrame({"a": [0, 1, 2, 3], "b": [3, 2, 1, 0]})
    y = np.array([0, 0, 1, 1])
    original_load_task = cc18.load_task
    original_evaluate = cc18.evaluate_official_split
    cc18.load_task = lambda task_id, retries: (FakeTask(), X, y, task_meta)

    calls: list[int] = []

    def evaluate(**kwargs):
        split_id = int(kwargs["split"]["split_id"])
        calls.append(split_id)
        first_attempt_failure = split_id == 0 and calls.count(0) == 1
        return (
            {
                "task_id": 7,
                "dataset_id": 70,
                "dataset": "retry-data",
                "model": "HUGIML",
                "split_id": split_id,
                "repeat": 0,
                "fold": split_id,
                "sample": 0,
                "status": "error" if first_attempt_failure else "ok",
                "error_count": 1 if first_attempt_failure else 0,
                "last_error": "temporary" if first_attempt_failure else None,
                "roc_auc": None if first_attempt_failure else 0.8,
            },
            None if first_attempt_failure else 3,
        )

    cc18.evaluate_official_split = evaluate
    common = {
        "task_id": 7,
        "model_name": "HUGIML",
        "out_dir": tmp_path,
        "retries": 0,
        "tune": True,
        "inner_splits": 3,
        "random_state": 42,
        "max_official_splits": None,
        "hugiml_scenario": "augmented_pair",
        "hugiml_max_fit_seconds": None,
        "run_id": "run-one",
    }
    try:
        first = cc18.run_pair(**common, resume=False)
        second = cc18.run_pair(**common, resume=True)
    finally:
        cc18.load_task = original_load_task
        cc18.evaluate_official_split = original_evaluate

    assert first["status"] == "partial"
    assert first["official_splits_successful"] == 1
    assert second["status"] == "ok"
    assert second["official_splits_successful"] == 2
    assert calls == [0, 1, 0]
    assert len(second["split_rows"]) == 2
    attempts = {row["split_id"]: row["attempt_count"] for row in second["split_rows"]}
    assert attempts == {0: 2, 1: 1}


def test_dashboard_uses_only_active_selected_tasks_and_custom_template(tmp_path: Path) -> None:
    models = list(cc18.MODEL_ORDER)
    payload = cc18.new_checkpoint()

    older = _activate(payload, "older", [99], models)
    old_row = _pair(99, "old-data", "HUGIML", 0.99)
    old_row["run_id"] = older
    payload["results"].append(old_row)

    active = _activate(payload, "active", [1, 2], models)
    scores = {
        1: [0.82, 0.80, 0.79, 0.78, 0.81, 0.80, 0.79],
        2: [0.84, 0.81, 0.83, 0.80, 0.82, 0.83, 0.81],
    }
    inspection = {
        "HUGIML": 10.0,
        "XGB standard": 50.0,
        "LightGBM standard": 45.0,
        "RandomForest standard": 40.0,
        "Logistic Regression": 12.0,
        "EBM": 20.0,
        "RuleFit": 25.0,
    }
    for task_id, values in scores.items():
        for model, auc in zip(models, values):
            row = _pair(
                task_id,
                f"tiny-{task_id}",
                model,
                auc,
                model_units=inspection[model] * 2,
                inspection_units=inspection[model],
                instance_units=max(1.0, inspection[model] / 5.0),
            )
            row["run_id"] = active
            payload["results"].append(row)
    stale = _pair(99, "stale-current", "HUGIML", 1.0)
    stale["run_id"] = active
    payload["results"].append(stale)

    checkpoint = tmp_path / "checkpoint.json"
    cc18.atomic_write_json(checkpoint, payload)
    outputs = cc18.assemble_outputs(checkpoint, tmp_path / "built-in", None)
    for path in outputs.values():
        assert path.exists() and path.stat().st_size > 0

    data = json.loads(outputs["dashboard_data_json"].read_text(encoding="utf-8"))
    assert data["run_id"] == active
    assert [task["task_id"] for task in data["per_task"]] == [1, 2]
    assert all(row["tasks_completed"] == 2 for row in data["overall"])
    assert all(row["task_id"] in {1, 2} for row in data["pair_results"])
    assert data["complexity_points"]
    xgb_summary = next(
        row for row in data["complexity_summary"] if row["baseline_model"] == "XGB standard"
    )
    assert xgb_summary["comparable_or_better_tasks"] == 2
    assert xgb_summary["median_inspection_ratio_baseline_to_hugiml"] == 5.0
    assert xgb_summary["headline"]
    pair_csv = pd.read_csv(outputs["pair_results_csv"])
    assert set(pair_csv["task_id"]) == {1, 2}
    ratio_csv = pd.read_csv(outputs["complexity_ratios_csv"])
    assert set(ratio_csv["task_id"]) == {1, 2}

    html_text = outputs["dashboard_html"].read_text(encoding="utf-8")
    assert "Official OpenML train/test indices" in html_text
    assert "Model inspection units versus AUC" in html_text
    assert "HUGIML inspection efficiency" in html_text
    assert 'data-theme="indigo"' in html_text
    assert "The dashboard reports descriptive evidence" not in html_text
    assert "six smallest" not in html_text
    assert "Â" not in html_text
    assert "â" not in html_text
    assert "{{DATA_JSON}}" not in html_text

    custom = tmp_path / "template.html"
    custom.write_text(
        "<!doctype html><title>{{TITLE}}</title><p>{{GENERATED_AT}}</p><script>const x={{DATA_JSON}}</script>",
        encoding="utf-8",
    )
    custom_outputs = cc18.assemble_outputs(checkpoint, tmp_path / "custom", custom)
    custom_text = custom_outputs["dashboard_html"].read_text(encoding="utf-8")
    assert "const x={" in custom_text
    assert "tiny-1" in custom_text
    assert "old-data" not in custom_text


def test_assembly_and_template_export_do_not_require_dataset_cache(tmp_path: Path) -> None:
    models = ["HUGIML"]
    payload = cc18.new_checkpoint()
    run_id = _activate(payload, "assemble", [1], models)
    row = _pair(1, "tiny-1", "HUGIML", 0.82)
    row["run_id"] = run_id
    payload["results"].append(row)
    checkpoint = tmp_path / "checkpoint.json"
    cc18.atomic_write_json(checkpoint, payload)

    missing_cache = tmp_path / "missing-cache"
    output = tmp_path / "output"
    assert cc18.main(
        [
            "--datasets-dir",
            str(missing_cache),
            "--assemble",
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(output),
        ]
    ) == 0
    assert (output / cc18.DEFAULT_DASHBOARD_NAME).is_file()

    template = tmp_path / "dashboard-template.html"
    assert cc18.main(
        [
            "--datasets-dir",
            str(missing_cache),
            "--write-template",
            str(template),
        ]
    ) == 0
    assert template.is_file()


def test_atomic_json_reader_uses_previous_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    cc18.atomic_write_json(path, {"value": 1})
    cc18.atomic_write_json(path, {"value": 2})
    path.write_text("{not-json", encoding="utf-8")
    assert cc18.read_json_with_backup(path) == {"value": 1}


def test_pair_error_row_is_resumable() -> None:
    row = cc18.pair_error_row(
        task={"task_id": 5, "dataset_id": 50, "dataset_name": "data", "n_rows": 10, "n_features": 3, "n_classes": 2},
        model_name="HUGIML",
        tune=True,
        inner_splits=3,
        hugiml_scenario="augmented_pair",
        error=RuntimeError("unavailable"),
    )
    payload = cc18.new_checkpoint()
    run_id = _activate(payload, "error", [5], ["HUGIML"])
    row["run_id"] = run_id
    payload["results"].append(row)
    assert row["status"] == "error"
    assert row["last_error"] == "RuntimeError: unavailable"
    assert cc18.completed_pair_keys(payload, run_id) == set()


def test_optional_cache_validation_checks_hashes_and_split_files(tmp_path: Path) -> None:
    folder = tmp_path / "task_1_data"
    split_dir = folder / "splits"
    split_dir.mkdir(parents=True)
    features = folder / "features.parquet"
    target = folder / "target.parquet"
    features.write_bytes(b"features")
    target.write_bytes(b"target")
    np.save(split_dir / "split_000_train.npy", np.array([0, 1]), allow_pickle=False)
    np.save(split_dir / "split_000_test.npy", np.array([2, 3]), allow_pickle=False)
    pd.DataFrame(
        [
            {
                "split_id": 0,
                "repeat": 0,
                "fold": 0,
                "sample": 0,
                "train_file": "splits/split_000_train.npy",
                "test_file": "splits/split_000_test.npy",
            }
        ]
    ).to_csv(folder / "split_manifest.csv", index=False)
    metadata = {
        "task_id": 1,
        "dataset_id": 10,
        "dataset_name": "data",
        "target_name": "target",
        "n_rows": 4,
        "n_features": 2,
        "n_classes": 2,
        "official_repeats": 1,
        "official_folds": 1,
        "official_samples": 1,
        "official_split_count": 1,
        "features_file": "features.parquet",
        "target_file": "target.parquet",
        "split_manifest_file": "split_manifest.csv",
        "sha256": {
            "features.parquet": cc18.sha256_file(features),
            "target.parquet": cc18.sha256_file(target),
        },
    }
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    entry = {**metadata, "folder": folder.name}
    manifest = {
        "complete": True,
        "requested_dataset_count": 1,
        "dataset_count": 1,
        "datasets": [entry],
    }
    cc18.verify_dataset_manifest(tmp_path, manifest)
    target.write_bytes(b"changed")
    try:
        cc18.verify_dataset_manifest(tmp_path, manifest)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("cache validation accepted changed data")


def test_parser_defaults_use_all_available_tasks() -> None:
    args = cc18.build_parser().parse_args([])
    assert args.smallest is None
    assert args.inner_splits == 3
    assert args.models == "all"
    assert args.max_official_splits is None
    assert "np.asarray(y[train_idx], dtype=int, copy=True)" not in SCRIPT.read_text(encoding="utf-8")


def test_pair_aggregation_pools_instance_inspection_moments() -> None:
    rows = [
        {
            "status": "ok",
            "error_count": 0,
            "roc_auc": 0.8,
            "complexity_model_units": 20.0,
            "complexity_model_inspection_units": 10.0,
            "complexity": 10.0,
            "complexity_instance_inspection_units_n_samples": 2,
            "complexity_instance_inspection_units_sum": 3.0,
            "complexity_instance_inspection_units_sum_squares": 5.0,
            "complexity_instance_inspection_units_confidence_level": 0.95,
            "complexity_instance_inspection_units_min": 1,
            "complexity_instance_inspection_units_max": 2,
        },
        {
            "status": "ok",
            "error_count": 0,
            "roc_auc": 0.9,
            "complexity_model_units": 22.0,
            "complexity_model_inspection_units": 12.0,
            "complexity": 12.0,
            "complexity_instance_inspection_units_n_samples": 2,
            "complexity_instance_inspection_units_sum": 7.0,
            "complexity_instance_inspection_units_sum_squares": 25.0,
            "complexity_instance_inspection_units_confidence_level": 0.95,
            "complexity_instance_inspection_units_min": 3,
            "complexity_instance_inspection_units_max": 4,
        },
    ]
    result = cc18.aggregate_pair_rows(
        rows,
        task_meta={
            "task_id": 1,
            "dataset_id": 2,
            "dataset_name": "tiny",
            "n_rows": 100,
            "n_features": 5,
            "n_classes": 2,
            "official_repeats": 1,
            "official_folds": 2,
            "official_samples": 1,
        },
        model_name="HUGIML",
        model_feature_counts=[6, 6],
        tune=True,
        inner_splits=3,
        hugiml_scenario="augmented_pair",
        official_splits_expected=2,
    )
    assert result["complexity_instance_inspection_units_n_samples"] == 4
    assert result["complexity_instance_inspection_units_mean"] == 2.5
    assert result["complexity_instance_inspection_units_min"] == 1
    assert result["complexity_instance_inspection_units_max"] == 4
    assert result["complexity_instance_inspection_units_ci_lower"] < 2.5
    assert result["complexity_instance_inspection_units_ci_upper"] > 2.5

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "benchmark" / "benchmark_engine.py"
BENCH_DIR = SCRIPT.parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

spec = importlib.util.spec_from_file_location("openml_cc18_benchmark_dashboard", SCRIPT)
assert spec and spec.loader
cc18 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cc18
spec.loader.exec_module(cc18)


def test_safe_jsonable_replaces_nonfinite_values() -> None:
    value = {
        "nan": float("nan"),
        "inf": np.float64("inf"),
        "nested": [np.int64(3), pd.NA, np.array([1.0, np.nan])],
    }
    cleaned = cc18.safe_jsonable(value)
    assert cleaned == {"nan": None, "inf": None, "nested": [3, None, [1.0, None]]}
    json.dumps(cleaned, allow_nan=False)


def test_resolve_suite_tasks_selects_six_smallest(monkeypatch) -> None:
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
    monkeypatch.setattr(cc18, "require_openml", lambda: fake_openml)

    suite, selected = cc18.resolve_suite_tasks(
        "OpenML-CC18", smallest=6, explicit_task_ids=[], retries=0
    )
    assert suite["suite_task_count"] == 8
    assert [row["task_id"] for row in selected] == [106, 104, 102, 108, 105, 103]


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


def _pair(task_id: int, dataset: str, model: str, auc: float) -> dict:
    return {
        "task_id": task_id,
        "dataset_id": task_id + 1000,
        "dataset": dataset,
        "model": model,
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
        "complexity": 10.0,
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


def test_dashboard_assembly_and_custom_template(tmp_path: Path) -> None:
    models = list(cc18.MODEL_ORDER[:4])
    payload = cc18.new_checkpoint()
    payload["metadata"].update(
        {
            "models": models,
            "inner_splits": 3,
            "grid_snapshot": {"HUGIML": {"candidate_count": 16}},
        }
    )
    payload["selected_tasks"] = [
        {"task_id": 1, "dataset_name": "tiny-a", "n_rows": 101, "n_classes": 2},
        {"task_id": 2, "dataset_name": "tiny-b", "n_rows": 102, "n_classes": 2},
    ]
    scores = {
        1: [0.82, 0.80, 0.79, 0.78],
        2: [0.84, 0.81, 0.83, 0.80],
    }
    for task_id, values in scores.items():
        for model, auc in zip(models, values):
            payload["results"].append(_pair(task_id, f"tiny-{task_id}", model, auc))
    payload["selected_tasks"].append(
        {"task_id": 3, "dataset_name": "incomplete", "n_rows": 103, "n_classes": 2}
    )
    payload["results"].append(_pair(3, "incomplete", models[0], 0.91))

    checkpoint = tmp_path / "checkpoint.json"
    cc18.atomic_write_json(checkpoint, payload)
    outputs = cc18.assemble_outputs(checkpoint, tmp_path / "built-in", None)
    for path in outputs.values():
        assert path.exists() and path.stat().st_size > 0
    html_text = outputs["dashboard_html"].read_text(encoding="utf-8")
    assert "Official OpenML train/test indices" in html_text
    assert "{{DATA_JSON}}" not in html_text
    assert "incomplete" not in html_text

    custom = tmp_path / "template.html"
    custom.write_text(
        "<!doctype html><title>{{TITLE}}</title><p>{{GENERATED_AT}}</p><script>const x={{DATA_JSON}}</script>",
        encoding="utf-8",
    )
    custom_outputs = cc18.assemble_outputs(checkpoint, tmp_path / "custom", custom)
    custom_text = custom_outputs["dashboard_html"].read_text(encoding="utf-8")
    assert "const x={" in custom_text
    assert "tiny-a" in custom_text


def test_parser_defaults_match_requested_protocol() -> None:
    args = cc18.build_parser().parse_args([])
    assert args.smallest is None
    assert args.inner_splits == 3
    assert args.models == "all"
    assert args.max_official_splits is None
    assert args.validation_protocol == "nested"
    assert args.early_stopping is None

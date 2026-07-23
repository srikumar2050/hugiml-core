from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "benchmark" / "download_openml_cc18_datasets.py"

if "openml" not in sys.modules:
    module = ModuleType("openml")
    module.__spec__ = importlib.util.spec_from_loader("openml", loader=None)
    module.config = SimpleNamespace()
    module.study = SimpleNamespace()
    module.tasks = SimpleNamespace()
    sys.modules["openml"] = module

spec = importlib.util.spec_from_file_location("openml_cc18_dataset_downloader", SCRIPT)
assert spec and spec.loader
downloader = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = downloader
spec.loader.exec_module(downloader)


def test_add_selects_next_smallest_missing_tasks_and_retains_complete_cache(
    tmp_path: Path,
) -> None:
    ordered = [
        {"task_id": 1, "n_rows": 10},
        {"task_id": 2, "n_rows": 20},
        {"task_id": 3, "n_rows": 30},
        {"task_id": 4, "n_rows": 40},
        {"task_id": 5, "n_rows": 50},
    ]
    cached = {
        1: {"task_id": 1, "complete": True},
        2: {"task_id": 2, "complete": False},
        4: {"task_id": 4, "complete": True},
    }
    original = downloader.reusable_entry
    downloader.reusable_entry = lambda root, entry: bool(entry.get("complete"))
    try:
        selected, label = downloader.select_download_tasks(
            ordered,
            cached,
            tmp_path,
            smallest=None,
            add=2,
        )
    finally:
        downloader.reusable_entry = original
    assert [row["task_id"] for row in selected] == [1, 2, 3, 4]
    assert label == "complete_cache_plus_2_next_smallest_missing_tasks"


def test_smallest_selection_keeps_order(tmp_path: Path) -> None:
    ordered = [
        {"task_id": 5, "n_rows": 10},
        {"task_id": 6, "n_rows": 20},
        {"task_id": 7, "n_rows": 30},
    ]
    selected, label = downloader.select_download_tasks(
        ordered,
        {},
        tmp_path,
        smallest=2,
        add=None,
    )
    assert [row["task_id"] for row in selected] == [5, 6]
    assert label == "smallest_2_by_NumberOfInstances"

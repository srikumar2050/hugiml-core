"""Regression coverage for the deterministic pytest batching helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_pytest_batches.py"
SPEC = importlib.util.spec_from_file_location("hugiml_run_pytest_batches", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_display_path_is_relative_inside_repository() -> None:
    path = MODULE.ROOT / "build" / "pytest_batches" / "batch_001.log"
    assert MODULE._display_path(path) == "build/pytest_batches/batch_001.log"


def test_display_path_accepts_external_log_directory(tmp_path: Path) -> None:
    path = tmp_path / "pytest-logs" / "batch_001.log"
    expected = (
        path.resolve().relative_to(MODULE.ROOT).as_posix()
        if path.resolve().is_relative_to(MODULE.ROOT)
        else path.resolve().as_posix()
    )
    assert MODULE._display_path(path) == expected

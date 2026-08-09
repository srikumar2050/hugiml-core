"""Forward to the maintained OpenML-CC18 dataset downloader."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent / "openml_cc18" / "download_openml_cc18_datasets.py"
_SPEC = importlib.util.spec_from_file_location("hugiml_openml_cc18_downloader", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load the OpenML-CC18 downloader from {_MODULE_PATH}")
_implementation = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_SPEC.name, _implementation)
_SPEC.loader.exec_module(_implementation)

main = _implementation.main
reusable_entry = _implementation.reusable_entry


def select_download_tasks(*args, **kwargs):
    """Select cached and pending tasks through the maintained implementation."""
    original = _implementation.reusable_entry
    _implementation.reusable_entry = reusable_entry
    try:
        return _implementation.select_download_tasks(*args, **kwargs)
    finally:
        _implementation.reusable_entry = original

__all__ = ["main", "reusable_entry", "select_download_tasks"]

if __name__ == "__main__":
    raise SystemExit(main())

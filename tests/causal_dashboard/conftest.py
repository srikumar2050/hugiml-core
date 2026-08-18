"""Test classification for the causal dashboard."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    marker = pytest.mark.optional
    test_root = Path(__file__).parent
    for item in items:
        item_path = Path(item.path)
        if test_root in item_path.parents and item_path.name == "test_causal_dashboard.py":
            item.add_marker(marker)

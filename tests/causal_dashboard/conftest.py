"""Optional causal-dashboard test classification."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    marker = pytest.mark.optional
    test_root = Path(__file__).parent
    for item in items:
        if test_root in Path(item.path).parents:
            item.add_marker(marker)

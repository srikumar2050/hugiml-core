import warnings
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hugiml import HUGIMLClassifierNative
from hugiml._classifier_support import _MemoryTracker, psutil
from hugiml.exceptions import HUGIMLWarning


@contextmanager
def available_memory(value):
    original = psutil.virtual_memory

    def read():
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(available=value)

    psutil.virtual_memory = read
    try:
        yield
    finally:
        psutil.virtual_memory = original


def notify(X):
    _MemoryTracker.warn_fit_memory(X, 20, include_originals=True, max_length=2)


def test_occurrence_storage_scales_with_rows_and_sources():
    parts = _MemoryTracker.estimate_fit_bytes(
        1000, 12, 20, n_numeric=8, include_originals=True, max_length=2
    )
    assert parts["transactions"] == 1000 * 12 * 4 + 1000 * 24
    assert parts["mining"] == 1000 * 12 * 24 * 2
    assert parts["original_features"] == 1000 * 8 * 20 + 1000 * 4 * 8 + 1001 * 4


def test_feature_mode_and_search_depth_affect_estimate():
    args = dict(n=1000, p=12, K=20, n_numeric=8)
    patterns = _MemoryTracker.estimate_fit_bytes(**args, include_originals=False, max_length=1)
    originals = _MemoryTracker.estimate_fit_bytes(**args, include_originals=True, max_length=2)
    assert "original_features" not in patterns
    assert originals["mining"] == 2 * patterns["mining"]
    assert originals["downstream"] > patterns["downstream"]


@pytest.mark.parametrize("available", [0, 1])
def test_low_available_memory_emits_capacity_warning(available):
    with available_memory(available), pytest.warns(HUGIMLWarning, match="available RAM"):
        notify(np.ones((20, 3)))


@pytest.mark.parametrize("available", [128 * 1024**3, -1, OSError("unavailable")])
def test_sufficient_or_unknown_memory_does_not_warn(available):
    with available_memory(available), warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        notify(np.ones((20, 3)))
    assert not recorded


def test_warning_uses_available_memory_budget_boundary():
    X = np.ones((20, 3))
    required = sum(
        _MemoryTracker.estimate_fit_bytes(
            20, 3, 20, n_numeric=3, include_originals=True, max_length=2
        ).values()
    )
    enough = int(np.ceil(required / 0.8))
    with available_memory(enough), warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        notify(X)
    assert not recorded
    with available_memory(enough - 1), pytest.warns(HUGIMLWarning):
        notify(X)


def test_categorical_dataframe_memory_check():
    X = pd.DataFrame({"category": pd.Categorical(["a", "b", "c"]), "numeric": [1, 2, 3]})
    with available_memory(1), pytest.warns(HUGIMLWarning):
        notify(X)


def test_large_estimate_with_available_capacity_does_not_warn():
    X = SimpleNamespace(shape=(70000, 784), dtype=np.dtype("float64"))
    components = _MemoryTracker.estimate_fit_bytes(
        70000, 784, 20, n_numeric=784, include_originals=True, max_length=2
    )
    assert sum(components.values()) > 4 * 1024**3
    with available_memory(32 * 1024**3), warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        notify(X)
    assert not recorded


def test_memory_advisory_preserves_fitted_predictions():
    X = pd.DataFrame({"x": np.arange(40), "category": ["a", "b"] * 20})
    y = np.array([0, 1] * 20)
    params = dict(B=3, L=1, topK=10, n_jobs=1, feature_mode="original_plus_patterns")
    with available_memory(128 * 1024**3):
        first = HUGIMLClassifierNative(**params).fit(X, y)
    with available_memory(1), pytest.warns(HUGIMLWarning, match="additional fit memory"):
        second = HUGIMLClassifierNative(**params).fit(X, y)
    np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X))

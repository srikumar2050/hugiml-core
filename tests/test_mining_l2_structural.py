# Copyright 2026 Srikumar Krishnamoorthy
# Apache-2.0 License

"""Regression tests for L>1 mining and feature-level mutual exclusion."""

from __future__ import annotations

import numpy as np
import pytest


def _core():
    try:
        import _hugiml_core as core
    except ImportError:
        pytest.skip("native extension not built")
    return core


def _prepare_precoded(X: np.ndarray, y: np.ndarray):
    core = _core()
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.int64)
    p = X.shape[1]
    flags = np.zeros(p, dtype=np.uint8)
    precoded = np.ones(p, dtype=np.uint8)
    return core.prepare_transactions(
        X,
        y,
        2,
        [f"f{j}" for j in range(p)],
        flags,
        flags,
        None,
        precoded,
    )


def test_l2_mining_produces_valid_compound_patterns_on_and_signal():
    """A strong two-feature AND signal must produce at least one L=2 pattern.

    This catches regressions where pruning, EUCS changes, or structural-filter
    changes accidentally stop recursive L>1 pattern generation.
    """
    core = _core()
    rng = np.random.default_rng(123)
    base = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    X = np.tile(base, (80, 1))
    rng.shuffle(X, axis=0)
    y = ((X[:, 0] == 1) & (X[:, 1] == 1)).astype(np.int64)

    td = _prepare_precoded(X, y)
    patterns = core.mine_patterns(td, y, 2, 50, 2, 0.0, 0.0)

    compound = [pe for pe in patterns if len(pe.items) == 2]
    assert compound, "L=2 mining produced no length-2 patterns on a clear AND signal"


@pytest.mark.parametrize("L", [2, 3])
def test_patterns_never_contain_two_bins_from_same_source_feature(L):
    """Mined L>1 patterns must respect one-bin-per-feature exclusivity."""
    core = _core()
    rng = np.random.default_rng(321)
    n = 300
    f0 = rng.integers(0, 3, size=n)  # three mutually-exclusive bins
    f1 = rng.integers(0, 3, size=n)
    f2 = rng.integers(0, 2, size=n)
    X = np.column_stack([f0, f1, f2]).astype(np.float64)
    y = (((f0 == 2) & (f1 == 1)) | ((f1 == 2) & (f2 == 1))).astype(np.int64)

    td = _prepare_precoded(X, y)
    patterns = core.mine_patterns(td, y, 2, 100, L, 0.0, 0.0)

    assert patterns, "Expected at least one mined pattern"
    item_col = list(td.item_col)
    for pe in patterns:
        cols = [item_col[item_id - 1] for item_id in pe.items]
        assert len(cols) == len(set(cols)), (
            f"Pattern {pe.items} contains multiple bins from the same feature: {cols}"
        )


def test_direct_transaction_constructor_enforces_same_feature_skip():
    """The native miner must skip same-feature item pairs before child creation.

    This deliberately constructs rows that contain two items assigned to the
    same source feature.  Such rows cannot arise from prepared binned tabular
    data, but they make the structural guard observable and prevent future
    regressions from silently re-enabling same-feature L>1 candidates.
    """
    core = _core()
    transactions = [
        [(1, 5.0), (2, 5.0), (3, 1.0)],
        [(1, 4.0), (2, 4.0), (3, 1.0)],
        [(1, 1.0), (3, 5.0)],
        [(2, 1.0), (3, 5.0)],
    ]
    # item 1 and item 2 are alternative bins of feature 0; item 3 is feature 1.
    td = core.TransactionDataCpp(transactions, 3, [0, 0, 1])
    y = np.array([1, 1, 0, 0], dtype=np.int64)
    patterns = core.mine_patterns(td, y, 2, 20, 2, 0.0, 0.0)

    assert not any(set(pe.items) == {1, 2} for pe in patterns), (
        "Same-feature pair [1, 2] was mined despite item_col marking both "
        "items as bins of feature 0"
    )

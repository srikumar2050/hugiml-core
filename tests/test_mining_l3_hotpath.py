import math

import _hugiml_core as core
import numpy as np
import pytest


def _make_data(n=600, p=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    score = (
        (X[:, 0] > 0).astype(float)
        + (X[:, 1] > 0).astype(float)
        + 0.75 * ((X[:, 2] > 0) & (X[:, 3] > 0))
    )
    if p >= 8:
        score += 0.9 * ((X[:, 4] > 0) & (X[:, 5] > 0) & (X[:, 6] > 0))
    y = (score + 0.2 * rng.normal(size=n) > 1.6).astype(np.int64)
    return X, y


def _signature(patterns):
    def ig_value(v):
        v = float(v)
        return "nan" if math.isnan(v) else round(v, 10)

    return sorted(
        (round(float(p.utility), 10), tuple(p.items), ig_value(p.ig))
        for p in patterns
    )


@pytest.mark.parametrize("K,G", [(50, 1e-4), (128, 1e-5)])
def test_l3_hotpath_matches_generic_and_dispatch(K, G):
    X, y = _make_data(seed=K)
    p = X.shape[1]
    td = core.prepare_transactions(
        X,
        y,
        5,
        [f"x{i}" for i in range(p)],
        np.zeros(p, dtype=np.uint8),
        np.zeros(p, dtype=np.uint8),
        None,
        None,
    )

    generic = core.mine_patterns_generic(td, y, 2, K, 3, G, 0.0)
    l3 = core.mine_patterns_l3(td, y, 2, K, G, 0.0)
    dispatch = core.mine_patterns(td, y, 2, K, 3, G, 0.0)

    assert _signature(l3) == _signature(generic)
    assert _signature(dispatch) == _signature(l3)


def test_l3_patterns_never_contain_two_bins_from_same_source_feature():
    X, y = _make_data(n=800, p=12, seed=123)
    p = X.shape[1]
    td = core.prepare_transactions(
        X,
        y,
        5,
        [f"x{i}" for i in range(p)],
        np.zeros(p, dtype=np.uint8),
        np.zeros(p, dtype=np.uint8),
        None,
        None,
    )
    patterns = core.mine_patterns_l3(td, y, 2, 200, 1e-6, 0.0)
    for pattern in patterns:
        cols = [td.item_col[item - 1] for item in pattern.items]
        assert len(cols) == len(set(cols))

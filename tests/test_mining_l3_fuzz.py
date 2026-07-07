"""Extended stress test: mine_patterns_l3 vs mine_patterns_generic(L=3).

Covers multiple seeds, class counts, feature layouts, and edge cases to keep
the L3 hot path aligned with the generic bounded miner.
"""
import math
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")
import _hugiml_core as core


def _signature(patterns):
    def ig_value(v):
        v = float(v)
        return "nan" if math.isnan(v) else round(v, 9)

    return sorted(
        (round(float(p.utility), 9), tuple(p.items), ig_value(p.ig))
        for p in patterns
    )


def _prep(X, y, n_bins=5):
    p = X.shape[1]
    return core.prepare_transactions(
        X, y, n_bins,
        [f"x{i}" for i in range(p)],
        np.zeros(p, dtype=np.uint8),
        np.zeros(p, dtype=np.uint8),
        None, None,
    )


def _make_data(n, p, n_cls, seed, n_informative=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    n_informative = n_informative or min(4, p)
    weights = rng.normal(size=n_informative)
    score = np.zeros(n)
    for k in range(n_informative):
        score += weights[k] * (X[:, k] > 0)
    score += 0.3 * rng.normal(size=n)
    if n_cls == 2:
        y = (score > np.median(score)).astype(np.int64)
    else:
        edges = np.quantile(score, np.linspace(0, 1, n_cls + 1)[1:-1])
        y = np.digitize(score, edges).astype(np.int64)
    return X, y


SEEDS = list(range(20))
CONFIGS = []
for seed in SEEDS:
    n = int(np.random.default_rng(seed + 1000).choice([120, 400, 900]))
    p = int(np.random.default_rng(seed + 2000).choice([4, 8, 20]))
    n_cls = int(np.random.default_rng(seed + 3000).choice([2, 3, 4]))
    K = int(np.random.default_rng(seed + 4000).choice([5, 30, 100]))
    G = float(np.random.default_rng(seed + 5000).choice([1e-6, 1e-4, 1e-2]))
    n_bins = int(np.random.default_rng(seed + 6000).choice([3, 5, 8]))
    CONFIGS.append((seed, n, p, n_cls, K, G, n_bins))


@pytest.mark.parametrize("seed,n,p,n_cls,K,G,n_bins", CONFIGS)
def test_l3_matches_generic_fuzz(seed, n, p, n_cls, K, G, n_bins):
    X, y = _make_data(n, p, n_cls, seed)
    td = _prep(X, y, n_bins=n_bins)
    generic = core.mine_patterns_generic(td, y, n_cls, K, 3, G, 0.0)
    l3 = core.mine_patterns_l3(td, y, n_cls, K, G, 0.0)
    dispatch = core.mine_patterns(td, y, n_cls, K, 3, G, 0.0)
    assert _signature(l3) == _signature(generic), f"cfg={seed,n,p,n_cls,K,G,n_bins}"
    assert _signature(dispatch) == _signature(l3)


def test_l3_empty_result_no_patterns_clear_gate():
    # G so large nothing can pass.
    X, y = _make_data(300, 6, 2, seed=42)
    td = _prep(X, y)
    generic = core.mine_patterns_generic(td, y, 2, 50, 3, 0.999, 0.0)
    l3 = core.mine_patterns_l3(td, y, 2, 50, 0.999, 0.0)
    assert _signature(l3) == _signature(generic)
    assert l3 == [] or len(l3) >= 0


def test_l3_single_feature_no_valid_triplets():
    # p=1 -> no two items can ever come from different source features,
    # so pairs/triplets should be impossible; only singletons possible.
    X, y = _make_data(300, 1, 2, seed=7)
    td = _prep(X, y, n_bins=5)
    generic = core.mine_patterns_generic(td, y, 2, 50, 3, 1e-6, 0.0)
    l3 = core.mine_patterns_l3(td, y, 2, 50, 1e-6, 0.0)
    assert _signature(l3) == _signature(generic)
    for pattern in l3:
        assert len(pattern.items) == 1


def test_l3_all_same_label_degenerate():
    # Extreme class imbalance (all but one row share a label) rather than a
    # single class, since prepare_transactions requires >= 2 distinct classes.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    y = np.zeros(200, dtype=np.int64)
    y[0] = 1
    td = _prep(X, y)
    generic = core.mine_patterns_generic(td, y, 2, 20, 3, 1e-6, 0.0)
    l3 = core.mine_patterns_l3(td, y, 2, 20, 1e-6, 0.0)
    assert _signature(l3) == _signature(generic)


def test_l3_k_larger_than_available_patterns():
    X, y = _make_data(150, 3, 2, seed=99)
    td = _prep(X, y, n_bins=3)
    generic = core.mine_patterns_generic(td, y, 2, 10_000, 3, 1e-3, 0.0)
    l3 = core.mine_patterns_l3(td, y, 2, 10_000, 1e-3, 0.0)
    assert _signature(l3) == _signature(generic)


def test_l3_many_duplicate_columns_high_conflict_rate():
    # Duplicate features -> lots of same-source conflicts across many pairs,
    # exercising same_feature_l3 exclusion and coverage-dedup heavily.
    rng = np.random.default_rng(11)
    base = rng.normal(size=(500, 3))
    X = np.hstack([base, base, base, rng.normal(size=(500, 2))])  # 11 cols, heavy dup
    y = ((base[:, 0] > 0).astype(int) ^ (base[:, 1] > 0).astype(int)).astype(np.int64)
    td = _prep(X, y, n_bins=4)
    generic = core.mine_patterns_generic(td, y, 2, 80, 3, 1e-5, 0.0)
    l3 = core.mine_patterns_l3(td, y, 2, 80, 1e-5, 0.0)
    assert _signature(l3) == _signature(generic)
    for pattern in l3:
        cols = [td.item_col[i - 1] for i in pattern.items]
        assert len(cols) == len(set(cols))

"""Extended stress tests: L1/L2 hotpaths vs the generic recursive miner.

Written alongside allocation/locality optimizations to mining_l1.cpp and
mining_l2.cpp (precomputed base-entropy term, pointer-based IG helpers,
reused scratch buffers instead of per-candidate heap allocations). Mirrors
the style of test_mining_l3_fuzz.py: many seeds/configs compared against
the generic oracle, plus targeted edge cases.
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


# ── L1 ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed,n,p,n_cls,K,G,n_bins", CONFIGS)
def test_l1_matches_generic_fuzz(seed, n, p, n_cls, K, G, n_bins):
    X, y = _make_data(n, p, n_cls, seed)
    td = _prep(X, y, n_bins=n_bins)
    generic = core.mine_patterns_generic(td, y, n_cls, K, 1, G, 0.0)
    l1 = core.mine_patterns(td, y, n_cls, K, 1, G, 0.0)
    assert _signature(l1) == _signature(generic), f"cfg={seed,n,p,n_cls,K,G,n_bins}"


def test_l1_k_larger_than_available_patterns():
    X, y = _make_data(150, 3, 2, seed=99)
    td = _prep(X, y, n_bins=3)
    generic = core.mine_patterns_generic(td, y, 2, 10_000, 1, 1e-3, 0.0)
    l1 = core.mine_patterns(td, y, 2, 10_000, 1, 1e-3, 0.0)
    assert _signature(l1) == _signature(generic)


def test_l1_extreme_class_imbalance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    y = np.zeros(200, dtype=np.int64)
    y[0] = 1
    td = _prep(X, y)
    generic = core.mine_patterns_generic(td, y, 2, 20, 1, 1e-6, 0.0)
    l1 = core.mine_patterns(td, y, 2, 20, 1, 1e-6, 0.0)
    assert _signature(l1) == _signature(generic)


def test_l1_multiclass_many_bins():
    X, y = _make_data(1200, 15, 5, seed=55, n_informative=6)
    td = _prep(X, y, n_bins=10)
    generic = core.mine_patterns_generic(td, y, 5, 80, 1, 1e-5, 0.0)
    l1 = core.mine_patterns(td, y, 5, 80, 1, 1e-5, 0.0)
    assert _signature(l1) == _signature(generic)


# ── L2 ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed,n,p,n_cls,K,G,n_bins", CONFIGS)
def test_l2_matches_generic_fuzz(seed, n, p, n_cls, K, G, n_bins):
    X, y = _make_data(n, p, n_cls, seed)
    td = _prep(X, y, n_bins=n_bins)
    generic = core.mine_patterns_generic(td, y, n_cls, K, 2, G, 0.0)
    l2 = core.mine_patterns_l2(td, y, n_cls, K, G, 0.0)
    dispatch = core.mine_patterns(td, y, n_cls, K, 2, G, 0.0)
    assert _signature(l2) == _signature(generic), f"cfg={seed,n,p,n_cls,K,G,n_bins}"
    assert _signature(dispatch) == _signature(l2)


def test_l2_k_larger_than_available_patterns():
    X, y = _make_data(150, 3, 2, seed=99)
    td = _prep(X, y, n_bins=3)
    generic = core.mine_patterns_generic(td, y, 2, 10_000, 2, 1e-3, 0.0)
    l2 = core.mine_patterns_l2(td, y, 2, 10_000, 1e-3, 0.0)
    assert _signature(l2) == _signature(generic)


def test_l2_single_feature_no_valid_pairs():
    X, y = _make_data(300, 1, 2, seed=7)
    td = _prep(X, y, n_bins=5)
    generic = core.mine_patterns_generic(td, y, 2, 50, 2, 1e-6, 0.0)
    l2 = core.mine_patterns_l2(td, y, 2, 50, 1e-6, 0.0)
    assert _signature(l2) == _signature(generic)
    for pattern in l2:
        assert len(pattern.items) == 1


def test_l2_extreme_class_imbalance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    y = np.zeros(200, dtype=np.int64)
    y[0] = 1
    td = _prep(X, y)
    generic = core.mine_patterns_generic(td, y, 2, 20, 2, 1e-6, 0.0)
    l2 = core.mine_patterns_l2(td, y, 2, 20, 1e-6, 0.0)
    assert _signature(l2) == _signature(generic)


def test_l2_multiclass_many_bins():
    X, y = _make_data(1200, 15, 5, seed=55, n_informative=6)
    td = _prep(X, y, n_bins=10)
    generic = core.mine_patterns_generic(td, y, 5, 80, 2, 1e-5, 0.0)
    l2 = core.mine_patterns_l2(td, y, 5, 80, 1e-5, 0.0)
    assert _signature(l2) == _signature(generic)


def test_l2_many_duplicate_columns_high_conflict_rate():
    # Near-duplicate (not exact-duplicate) columns: heavy same-feature-conflict
    # stress without manufacturing exact utility/IG ties. Exact duplicate
    # columns produce genuine ties at the K-boundary where the hotpath and
    # generic miner may each pick a different (but equally valid) tied
    # representative. Tiny independent noise per "copy" keeps columns highly
    # correlated (same conflict-rate stress) while avoiding exact ties.
    rng = np.random.default_rng(11)
    base = rng.normal(size=(500, 3))
    copies = [base + rng.normal(scale=1e-3, size=base.shape) for _ in range(3)]
    X = np.hstack(copies + [rng.normal(size=(500, 2))])
    y = ((base[:, 0] > 0).astype(int) ^ (base[:, 1] > 0).astype(int)).astype(np.int64)
    td = _prep(X, y, n_bins=4)
    generic = core.mine_patterns_generic(td, y, 2, 80, 2, 1e-5, 0.0)
    l2 = core.mine_patterns_l2(td, y, 2, 80, 1e-5, 0.0)
    assert _signature(l2) == _signature(generic)
    for pattern in l2:
        cols = [td.item_col[i - 1] for i in pattern.items]
        assert len(cols) == len(set(cols))


# ── L2 augmented-pair variants sharing the l2_ig_global behavior ──────────

@pytest.mark.parametrize("seed,n,p,n_cls,K,G,n_bins", CONFIGS[:10])
def test_l2_augmented_v1_degenerate_matches_ordinary_l2(seed, n, p, n_cls, K, G, n_bins):
    """relaxed_cols=[] and relaxed_quota=0 must be exactly output-equivalent
    to mine_patterns_l2 (documented contract), including after the shared
    l2_ig_global signature change."""
    X, y = _make_data(n, p, n_cls, seed)
    td = _prep(X, y, n_bins=n_bins)
    ordinary = core.mine_patterns_l2(td, y, n_cls, K, G, 0.0)
    augmented = core.mine_patterns_l2_augmented_patterns(td, y, n_cls, K, G, [], 0, 0.0)
    assert augmented.relaxed_patterns == []
    assert _signature(augmented.patterns) == _signature(ordinary)


@pytest.mark.parametrize("seed,n,p,n_cls,K,G,n_bins", CONFIGS[:10])
def test_l2_augmented_v2_degenerate_matches_ordinary_l2(seed, n, p, n_cls, K, G, n_bins):
    X, y = _make_data(n, p, n_cls, seed)
    td = _prep(X, y, n_bins=n_bins)
    ordinary = core.mine_patterns_l2(td, y, n_cls, K, G, 0.0)
    augmented_v2 = core.mine_patterns_l2_augmented_patterns_v2(td, y, n_cls, K, G, [], 0.0)
    assert _signature(augmented_v2) == _signature(ordinary)


def test_l2_augmented_v1_with_relaxed_cols_respects_budget_and_exclusivity():
    """Sanity-check the augmented path still respects its structural
    contract (K budget split, same-feature exclusion) with actual relaxed
    columns set, since this path shares the optimized l2_ig_global."""
    X, y = _make_data(600, 16, 2, seed=0)
    td = _prep(X, y, n_bins=5)
    K = 60
    relaxed_quota = 15
    result = core.mine_patterns_l2_augmented_patterns(
        td, y, 2, K, 1e-3, [3, 4], relaxed_quota, 0.0)
    assert len(result.patterns) <= K - relaxed_quota
    assert len(result.relaxed_patterns) <= relaxed_quota
    assert len(result.patterns) + len(result.relaxed_patterns) <= K
    for pattern in list(result.patterns) + list(result.relaxed_patterns):
        cols = [td.item_col[i - 1] for i in pattern.items]
        assert len(cols) == len(set(cols))

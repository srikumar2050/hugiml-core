# Copyright 2026 Srikumar Krishnamoorthy
# Apache-2.0 License

"""Regression coverage for singleton mining dispatch."""

from __future__ import annotations

import math

import numpy as np
import pytest


def _core():
    try:
        import _hugiml_core as core
    except ImportError:
        pytest.skip("native extension not built")
    return core


def _signature(patterns):
    out = []
    for pe in patterns:
        ig = None if math.isnan(pe.ig) else round(float(pe.ig), 12)
        out.append((round(float(pe.utility), 12), tuple(pe.items), ig))
    return sorted(out)


def _prepare_precoded(core, X, y):
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


@pytest.mark.parametrize("K", [5, 20, 100])
@pytest.mark.parametrize("G", [0.0, 0.001, 0.1])
def test_l1_dispatch_matches_generic_singleton_miner(K, G):
    """The public L=1 dispatch must match the generic miner, including ties."""
    core = _core()
    rng = np.random.default_rng(20260628 + K)
    X = rng.integers(0, 4, size=(84, 7)).astype(np.float64)
    y = (rng.random(84) > 0.45).astype(np.int64)

    td = _prepare_precoded(core, X, y)
    hot = core.mine_patterns(td, y, 2, K, 1, G, 0.0)
    generic = core.mine_patterns_generic(td, y, 2, K, 1, G, 0.0)

    assert _signature(hot) == _signature(generic)

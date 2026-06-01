import math

from hugiml.classifier import HUGIMLClassifierNative, HUGIMLParamError


def test_mining_budget_equals_topk():
    clf = HUGIMLClassifierNative(topK=100)
    assert clf._effective_topK(n_items=500) == 100
    assert clf._effective_mining_topK(n_items=500) == 100


def test_effective_topk_and_mining_topk_are_equal():
    """Single-pass: effective topK == mining topK for all topK values."""
    for topK in [1, 30, 100, 200]:
        clf = HUGIMLClassifierNative(topK=topK)
        assert clf._effective_topK(n_items=500) == clf._effective_mining_topK(n_items=500)


def test_topk_minus_one_resolves_from_n_items_l1():
    """topK=-1 with L=1 auto-resolves to n_items (capped at 20 000)."""
    clf = HUGIMLClassifierNative(topK=-1, L=1)
    assert clf._effective_topK(n_items=101) == 101
    assert clf._effective_topK(n_items=20000) == 20000
    # Cap at 20 000
    assert clf._effective_topK(n_items=99999) == 20000


def test_topk_minus_one_resolves_from_n_items_l2():
    """topK=-1 with L=2 auto-resolves to C(n_items,2) (capped at 20 000)."""
    clf = HUGIMLClassifierNative(topK=-1, L=2)
    n = 20
    assert clf._effective_topK(n_items=n) == min(math.comb(n, 2), 20000)


def test_use_hotpath_default_true():
    clf = HUGIMLClassifierNative()
    assert clf.use_hotpath is True


def test_use_hotpath_can_be_disabled():
    clf = HUGIMLClassifierNative(use_hotpath=False)
    assert clf.use_hotpath is False


def test_validate_params_passes_valid():
    clf = HUGIMLClassifierNative(B=5, L=1, G=1e-3, topK=30)
    clf._validate_params()  # must not raise


def test_validate_params_rejects_bad_B():
    clf = HUGIMLClassifierNative(B=1)
    try:
        clf._validate_params()
    except HUGIMLParamError:
        pass
    else:
        raise AssertionError("B=1 should be rejected")


def test_validate_params_rejects_negative_G():
    clf = HUGIMLClassifierNative(G=-0.1)
    try:
        clf._validate_params()
    except HUGIMLParamError:
        pass
    else:
        raise AssertionError("G<0 should be rejected")


def test_default_param_grid_keys():
    grid = HUGIMLClassifierNative.default_param_grid()
    for key in ("B", "L", "topK", "G", "adaptive_binning", "feature_mode"):
        assert key in grid, f"missing key: {key}"

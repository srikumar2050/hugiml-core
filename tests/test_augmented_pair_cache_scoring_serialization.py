import pickle

import numpy as np
import pandas as pd
import pytest

import hugiml.classifier as clf_mod
from hugiml import HUGIMLClassifierNative


@pytest.mark.skipif(not clf_mod._CORE_AVAILABLE, reason="native extension unavailable")
def test_bounded_pair_scorer_is_exact_topk_and_legacy_order_is_unchanged():
    core = clf_mod._core
    if not hasattr(core, "score_pair_candidates_bounded"):
        pytest.skip("bounded pair scorer unavailable")
    rng = np.random.default_rng(20260628)
    n, p = 180, 12
    X = rng.normal(size=(n, p)).astype(np.float64)
    X[rng.random(size=X.shape) < 0.12] = np.nan
    y = (
        0.7 * np.nan_to_num(X[:, 0], nan=0.0) * np.nan_to_num(X[:, 1], nan=0.0)
        - 0.4 * np.nan_to_num(X[:, 2], nan=0.0)
        + 0.1 * rng.normal(size=n)
        > 0
    ).astype(np.int64)
    cols = [f"f{j}" for j in range(p)]

    legacy = list(core.score_pair_candidates(X, y, cols))
    legacy_again = list(core.score_pair_candidates(X.copy(), y.copy(), cols))
    assert [row["name"] for row in legacy] == [row["name"] for row in legacy_again]
    assert [float(row["transform_ig"]) for row in legacy] == [
        float(row["transform_ig"]) for row in legacy_again
    ]

    sorted_legacy = sorted(legacy, key=lambda item: (-float(item["transform_ig"]), str(item["name"])))
    bounded, total = core.score_pair_candidates_bounded(X, y, cols, 25)
    bounded = list(bounded)
    assert int(total) == len(legacy)
    assert [row["name"] for row in bounded] == [row["name"] for row in sorted_legacy[:25]]
    assert [float(row["transform_ig"]) for row in bounded] == [
        float(row["transform_ig"]) for row in sorted_legacy[:25]
    ]


@pytest.mark.skipif(not clf_mod._CORE_AVAILABLE, reason="native extension unavailable")
def test_augmented_pair_cache_exact_key_and_missing_transform_semantics():
    core = clf_mod._core
    if not hasattr(core, "AugmentedPairCache"):
        pytest.skip("augmented pair cache unavailable")
    rng = np.random.default_rng(17)
    n, p = 120, 8
    X = rng.normal(size=(n, p)).astype(np.float64)
    X[rng.random(size=X.shape) < 0.15] = np.nan
    y = (np.nan_to_num(X[:, 0]) - 0.5 * np.nan_to_num(X[:, 1]) > 0).astype(np.int64)
    cols = [f"x{j}" for j in range(p)]

    cache = core.AugmentedPairCache()
    first = cache.select_interaction_information_features(X, y, cols, 4, None)
    second = cache.select_interaction_information_features(X.copy(), y.copy(), cols, 4, None)
    assert [dict(row) for row in first] == [dict(row) for row in second]
    stats = cache.stats()
    assert int(stats["select_misses"]) == 1
    assert int(stats["select_hits"]) == 1

    X_changed = X.copy()
    X_changed[0, 0] = 123.0
    cache.select_interaction_information_features(X_changed, y, cols, 4, None)
    assert int(cache.stats()["select_misses"]) == 2

    left = np.array([0, 0, 2, 3], dtype=np.int64)
    right = np.array([1, 2, 3, 4], dtype=np.int64)
    ops = np.array([0, 1, 2, 3], dtype=np.int8)
    refs = np.array([1.25, -0.5, 0.0, 2.0], dtype=np.float64)
    means = refs.copy()
    scales = np.array([1.0, 2.0, 0.0, np.nan], dtype=np.float64)
    Z = np.asarray(core.transform_pair_features(X, left, right, ops, refs, means, scales))
    assert np.isfinite(Z).all()
    assert Z.dtype == np.float32


@pytest.mark.skipif(not clf_mod._CORE_AVAILABLE, reason="native extension unavailable")
def test_augmented_pair_serialization_and_interpretability_contract(tmp_path):
    if not hasattr(clf_mod._core, "AugmentedPairCache"):
        pytest.skip("augmented pair cache unavailable")
    rng = np.random.default_rng(101)
    n, p = 240, 12
    X = rng.normal(size=(n, p))
    X[rng.random(size=X.shape) < 0.08] = np.nan
    y = (
        0.8 * np.nan_to_num(X[:, 0])
        - 0.5 * np.nan_to_num(X[:, 1])
        + 0.9 * np.nan_to_num(X[:, 2]) * np.nan_to_num(X[:, 3])
        + 0.2 * rng.normal(size=n)
        > 0
    ).astype(int)
    Xdf = pd.DataFrame(X, columns=[f"x{j}" for j in range(p)])

    model = HUGIMLClassifierNative(
        L=2,
        topK=20,
        G=0.001,
        adaptive_binning=True,
        feature_mode="original_plus_patterns",
        n_jobs=1,
        execution_mode="audit",
        dense_downstream_max_width=10000,
    )
    model.fit(Xdf, y)
    proba = model.predict_proba(Xdf.iloc[:20])[:, 1]
    catalog = model.get_augmented_pair_transforms()
    assert isinstance(catalog, list)
    for item in catalog:
        assert item["pair_missing_policy"] == "reference_value_for_unavailable_pair"
        assert "standardization" in item
        assert item["used_in_hugiml_mining"] is False

    block = getattr(model, "_augmented_pair_block_", None)
    assert block is not None
    state = block.__getstate__()
    assert "native_cache" not in state
    assert "_fit_Z_cache_" not in state
    assert getattr(block, "native_cache", None) is None

    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    np.testing.assert_allclose(restored.predict_proba(Xdf.iloc[:20])[:, 1], proba, rtol=0, atol=0)
    assert restored.get_augmented_pair_transforms() == catalog

    path = tmp_path / "model.hugiml"
    model.save_model(path)
    loaded = HUGIMLClassifierNative.load_model(path)
    np.testing.assert_allclose(loaded.predict_proba(Xdf.iloc[:20])[:, 1], proba, rtol=0, atol=1e-12)
    assert loaded.get_augmented_pair_transforms() == catalog

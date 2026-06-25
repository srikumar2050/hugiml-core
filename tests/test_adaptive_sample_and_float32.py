import numpy as np
import pytest

from hugiml import HUGIMLClassifierNative


def test_float32_numeric_input_remains_float32():
    X = np.ones((8, 3), dtype=np.float32)
    X_num, cats = HUGIMLClassifierNative._to_float_array(X)
    assert X_num.dtype == np.float32
    assert X_num.flags.c_contiguous
    assert cats == [None, None, None]


def test_float32_dataframe_input_is_c_contiguous():
    X = np.ones((8, 3), dtype=np.float32)
    X_df = __import__("pandas").DataFrame(X, columns=["a", "b", "c"])
    X_num, cats = HUGIMLClassifierNative._to_float_array(X_df)
    assert X_num.dtype == np.float32
    assert X_num.flags.c_contiguous
    assert cats == [None, None, None]


def test_float32_noncontiguous_view_fits_and_predicts():
    rng = np.random.default_rng(77)
    X_base = rng.normal(size=(96, 6)).astype(np.float32)
    X = X_base[:, ::2]
    assert not X.flags.c_contiguous
    y = (X[:, 0] + 0.2 * X[:, 1] > np.median(X[:, 0])).astype(np.int64)
    clf = HUGIMLClassifierNative(
        L=1,
        topK=12,
        G=0.0,
        use_hotpath=True,
        feature_mode="patterns_only",
        execution_mode="production",
        n_jobs=1,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X[:12])
    assert proba.shape == (12, 2)
    assert np.all(np.isfinite(proba))


def test_float32_noncontiguous_adaptive_view_fits_and_predicts():
    rng = np.random.default_rng(78)
    X_base = rng.normal(size=(120, 8)).astype(np.float32)
    X = X_base[:, ::2]
    assert not X.flags.c_contiguous
    y = (X[:, 0] - 0.4 * X[:, 1] > np.median(X[:, 0])).astype(np.int64)
    clf = HUGIMLClassifierNative(
        L=1,
        topK=14,
        G=0.0,
        adaptive_binning=True,
        b_candidates=[2, 3, 5],
        use_hotpath=True,
        feature_mode="patterns_only",
        execution_mode="production",
        n_jobs=1,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X[:12])
    assert proba.shape == (12, 2)
    assert np.all(np.isfinite(proba))


def test_adaptive_sample_parameters_are_sklearn_visible():
    clf = HUGIMLClassifierNative(
        adaptive_binning_sample_frac=0.2,
        adaptive_binning_sample_random_state=17,
    )
    params = clf.get_params()
    assert params["adaptive_binning_sample_frac"] == 0.2
    assert params["adaptive_binning_sample_random_state"] == 17
    assert clf._adaptive_binning_sample_fraction_for_native() == 0.2


def test_adaptive_sample_validation():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    for bad in (True, 0, -0.1, 1.2, "0.2"):
        with pytest.raises(Exception):
            HUGIMLClassifierNative(adaptive_binning_sample_frac=bad).fit(X, y)


def test_adaptive_sample_full_matches_disabled():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(1500, 8)).astype(np.float32)
    z = X[:, 0] - 0.6 * X[:, 1] + 0.4 * (X[:, 2] > 0.0)
    y = (z > np.median(z)).astype(np.int64)
    base = dict(
        L=1,
        topK=30,
        G=0.01,
        adaptive_binning=True,
        b_candidates=[3, 5, 7],
        use_hotpath=True,
        augmented_pair_transforms=False,
        interaction_relaxed_mining=False,
        execution_mode="production",
        feature_mode="patterns_only",
        n_jobs=2,
    )
    clf_disabled = HUGIMLClassifierNative(**base, adaptive_binning_sample_frac=False)
    clf_full = HUGIMLClassifierNative(**base, adaptive_binning_sample_frac=1.0)
    clf_disabled.fit(X, y)
    clf_full.fit(X, y)
    p_disabled = clf_disabled.predict_proba(X[:200])[:, 1]
    p_full = clf_full.predict_proba(X[:200])[:, 1]
    np.testing.assert_allclose(p_disabled, p_full, rtol=0, atol=0)

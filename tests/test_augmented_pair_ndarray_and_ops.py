import _hugiml_core as _core
import numpy as np
import pandas as pd

from hugiml import HUGIMLClassifierNative


def test_native_augmented_pair_supports_all_four_operations():
    X = np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 1.0], [2.0, 0.0]])
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    specs = list(_core.score_pair_candidates(X, y, ["a", "b"]))
    assert [spec["operation"] for spec in specs] == [
        "product",
        "absolute_difference",
        "sum",
        "signed_difference",
    ]

    Z = _core.transform_pair_features(
        X,
        np.array([0, 0, 0, 0], dtype=np.int64),
        np.array([1, 1, 1, 1], dtype=np.int64),
        np.array([0, 1, 2, 3], dtype=np.int8),
        np.zeros(4, dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        np.ones(4, dtype=np.float64),
    )
    expected = np.column_stack(
        [X[:, 0] * X[:, 1], np.abs(X[:, 0] - X[:, 1]), X[:, 0] + X[:, 1], X[:, 0] - X[:, 1]]
    )
    np.testing.assert_allclose(np.asarray(Z), expected.astype(np.float32), rtol=1e-6, atol=1e-6)


def _interaction_data(n=420, p=8, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (
        1.6 * ((X[:, 0] > 0.45) & (X[:, 1] < -0.1)).astype(float)
        + 2.3 * ((X[:, 2] + X[:, 3]) > 0.9).astype(float)
        + 1.5 * ((X[:, 4] - X[:, 5]) > 1.1).astype(float)
        + 0.1 * rng.normal(size=n)
        > 1.0
    ).astype(int)
    return X, y


def _fit(X, y):
    clf = HUGIMLClassifierNative(
        L=2,
        topK=30,
        B=5,
        G=0.0,
        adaptive_binning=True,
        b_candidates=[2, 3, 5, 7, 10],
        feature_mode="original_plus_patterns",
        augmented_pair_transforms=True,
        aug_feature_size=8,
        use_hotpath=True,
        n_jobs=1,
        interaction_relaxed_mining=False,
    )
    clf.fit(X, y)
    return clf


def test_ndarray_augmented_pairs_use_synthetic_names_and_match_dataframe_behavior():
    X, y = _interaction_data()
    cols = [f"f{j}" for j in range(X.shape[1])]

    clf_np = _fit(X, y)
    clf_df = _fit(pd.DataFrame(X, columns=cols), y)

    assert clf_np.feature_names_in_ == [f"col{j}" for j in range(X.shape[1])]
    assert len(clf_np.get_augmented_pair_transforms()) > 0
    assert len(clf_df.get_augmented_pair_transforms()) > 0

    np_ops = {t["operation"] for t in clf_np.get_augmented_pair_transforms()}
    df_ops = {t["operation"] for t in clf_df.get_augmented_pair_transforms()}
    assert {"sum", "signed_difference"}.issubset(np_ops)
    assert {"sum", "signed_difference"}.issubset(df_ops)
    assert clf_np.augmented_pair_config_["num_candidates"] > 0
    assert clf_df.augmented_pair_config_["num_candidates"] > 0

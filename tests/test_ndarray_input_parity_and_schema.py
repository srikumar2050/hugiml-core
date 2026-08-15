import numpy as np
import pandas as pd
import pytest

from hugiml import HUGIMLClassifierNative
from hugiml.classifier import HUGIMLSchemaError


def _data(seed=123, n=180, p=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (
        1.4 * ((X[:, 0] + X[:, 1]) > 0.8)
        + 1.2 * ((X[:, 2] - X[:, 3]) > 0.5)
        + 0.7 * ((X[:, 4] * X[:, 5]) > 0.4)
        + 0.2 * rng.normal(size=n)
        > 0.9
    ).astype(int)
    return X, y, pd.DataFrame(X, columns=[f"col{j}" for j in range(p)])


def _clf(feature_mode="original_plus_patterns", adaptive_binning=True):
    return HUGIMLClassifierNative(
        L=2,
        topK=20,
        G=0,
        B=4,
        adaptive_binning=adaptive_binning,
        feature_mode=feature_mode,
        augmented_pair_transforms=True,
        aug_feature_size=20,
        use_hotpath=True,
        interaction_relaxed_mining=False,
    )


@pytest.mark.parametrize("feature_mode", ["patterns_only", "original_plus_patterns", "original_plus_interactions"])
@pytest.mark.parametrize("adaptive_binning", [True, False])
def test_ndarray_and_dataframe_colj_parity(feature_mode, adaptive_binning):
    X, y, X_df = _data()
    arr_model = _clf(feature_mode, adaptive_binning).fit(X, y)
    df_model = _clf(feature_mode, adaptive_binning).fit(X_df, y)

    np.testing.assert_allclose(arr_model.predict_proba(X), df_model.predict_proba(X_df), atol=0, rtol=0)
    np.testing.assert_array_equal(arr_model.predict(X), df_model.predict(X_df))
    arr_transform = arr_model.transform(X)
    df_transform = df_model.transform(X_df)
    np.testing.assert_allclose(
        arr_transform.toarray() if hasattr(arr_transform, "toarray") else np.asarray(arr_transform),
        df_transform.toarray() if hasattr(df_transform, "toarray") else np.asarray(df_transform),
        atol=0,
        rtol=0,
    )
    assert arr_model.fit_metadata_.n_augmented_pairs == df_model.fit_metadata_.n_augmented_pairs


def test_augmented_pairs_keep_ndarray_schema_order_when_selected_features_reorder():
    X, y, X_df = _data(n=220, p=8)
    model = _clf("original_plus_patterns", True).fit(X, y)
    # Cross input must be exactly aligned even when selected IG features are not
    # in raw column order. This checks ndarray-column assignment against the
    # fitted full feature schema rather than selected-feature order.
    np.testing.assert_allclose(model.predict_proba(X), model.predict_proba(X_df), atol=0, rtol=0)
    assert model.fit_metadata_.n_augmented_pairs > 0


def test_nonadaptive_ndarray_uses_synthetic_names_for_prebin_edges():
    X, y, X_df = _data(n=220, p=8)
    arr_model = _clf("patterns_only", False).fit(X, y)
    df_model = _clf("patterns_only", False).fit(X_df, y)
    np.testing.assert_allclose(arr_model.predict_proba(X), df_model.predict_proba(X_df), atol=0, rtol=0)
    arr_transform = arr_model.transform(X)
    df_transform = df_model.transform(X_df)
    np.testing.assert_allclose(
        arr_transform.toarray() if hasattr(arr_transform, "toarray") else np.asarray(arr_transform),
        df_transform.toarray() if hasattr(df_transform, "toarray") else np.asarray(df_transform),
        atol=0,
        rtol=0,
    )


def test_nonfinite_ndarray_original_features_do_not_break_downstream_scaler():
    X, y, _ = _data(n=180, p=8)
    X = X.copy()
    X[0, 0] = np.nan
    X[1, 1] = np.inf
    model = _clf("original_plus_patterns", False).fit(X, y)
    proba = model.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    assert np.isfinite(proba).all()


def test_wrong_width_ndarray_raises_schema_error_not_index_error():
    X, y, _ = _data(n=180, p=8)
    model = _clf("original_plus_patterns", True).fit(X, y)
    with pytest.raises(HUGIMLSchemaError):
        model.predict_proba(X[:5, :-1])
    with pytest.raises(HUGIMLSchemaError):
        model.transform(X[:5, :-1])

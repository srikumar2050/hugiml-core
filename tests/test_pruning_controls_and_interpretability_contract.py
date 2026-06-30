import pickle

import numpy as np
import pandas as pd
import pytest

import hugiml.classifier as clf_mod
from hugiml import HUGIMLClassifierNative


def _make_wide_numeric_frame(seed: int = 772, n: int = 180, p: int = 18) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(float)
    X[rng.random(size=X.shape) < 0.08] = np.nan
    score = (
        0.9 * np.nan_to_num(X[:, 0], nan=0.0)
        - 0.7 * np.nan_to_num(X[:, 1], nan=0.0)
        + 1.1
        * (
            np.nan_to_num(X[:, 2], nan=0.0)
            * np.nan_to_num(X[:, 3], nan=0.0)
            > 0.25
        )
        + 0.2 * rng.normal(size=n)
    )
    y = (score > np.nanmedian(score)).astype(int)
    return pd.DataFrame(X, columns=[f"x{j}" for j in range(p)]), y


@pytest.mark.skipif(not clf_mod._CORE_AVAILABLE, reason="native extension unavailable")
def test_eucs_enable_false_is_complete_noop_for_predictions_and_patterns(monkeypatch):
    X, y = _make_wide_numeric_frame()
    params = dict(
        L=2,
        topK=30,
        G=0.001,
        adaptive_binning=True,
        feature_mode="patterns_only",
        augmented_pair_transforms=False,
        execution_mode="audit",
        n_jobs=1,
    )

    monkeypatch.setenv("HUGIML_EUCS_ENABLE", "false")
    disabled = HUGIMLClassifierNative(**params).fit(X, y)
    disabled_sig = sorted(
        (tuple(int(i) for i in pe.items), round(float(pe.utility), 12), round(float(pe.ig), 12))
        for pe in disabled.patterns_
    )
    disabled_proba = disabled.predict_proba(X.iloc[:25])

    monkeypatch.setenv("HUGIML_EUCS_ENABLE", "true")
    # Force EUCS on for this small audit dataset so the test covers both branches.
    monkeypatch.setenv("HUGIML_EUCS_MIN_ITEMS", "0")
    monkeypatch.setenv("HUGIML_EUCS_MAX_DENSITY", "1")
    enabled = HUGIMLClassifierNative(**params).fit(X, y)
    enabled_sig = sorted(
        (tuple(int(i) for i in pe.items), round(float(pe.utility), 12), round(float(pe.ig), 12))
        for pe in enabled.patterns_
    )

    assert enabled_sig == disabled_sig
    np.testing.assert_allclose(enabled.predict_proba(X.iloc[:25]), disabled_proba, rtol=0, atol=0)


@pytest.mark.skipif(not clf_mod._CORE_AVAILABLE, reason="native extension unavailable")
def test_augmented_pair_interpretability_and_missing_policy_round_trip(tmp_path):
    X, y = _make_wide_numeric_frame(seed=913, n=220, p=12)
    model = HUGIMLClassifierNative(
        L=2,
        topK=24,
        G=0.0001,
        adaptive_binning=True,
        feature_mode="original_plus_patterns",
        augmented_pair_transforms=True,
        augmented_pair_mode="interaction_information",
        aug_feature_size=8,
        dense_downstream_max_width=10000,
        execution_mode="audit",
        n_jobs=1,
    ).fit(X, y)

    catalog = model.get_augmented_pair_transforms()
    assert catalog, "test dataset should retain at least one augmented pair"
    assert all(item["used_in_hugiml_mining"] is False for item in catalog)
    assert all(item["pair_missing_policy"] == "reference_value_for_unavailable_pair" for item in catalog)
    assert all("standardized_formula" in item for item in catalog)

    std = model.get_augmented_pair_standardization()
    effects = model.explain_augmented_pair_effects()
    assert len(std) == len(catalog)
    assert len(effects) == len(catalog)
    assert {"raw_formula", "coefficient_raw_scale", "pair_missing_policy_description"}.issubset(
        set(effects.columns)
    )
    assert effects["raw_interpretation"].str.contains("reference raw value", case=False).all()

    missing_row = X.iloc[[0]].copy()
    selected_sources = list(getattr(model._augmented_pair_block_, "selected_aug_features_", []))
    missing_row.loc[:, selected_sources] = np.nan
    z_missing = model._augmented_pair_block_.transform(missing_row).toarray()
    assert np.isfinite(z_missing).all()
    np.testing.assert_allclose(z_missing, np.zeros_like(z_missing), atol=1e-7)

    pickled = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    assert pickled.get_augmented_pair_transforms() == catalog
    pd.testing.assert_frame_equal(
        pickled.get_augmented_pair_standardization().reset_index(drop=True),
        std.reset_index(drop=True),
    )
    np.testing.assert_allclose(pickled.predict_proba(X.iloc[:20]), model.predict_proba(X.iloc[:20]))

    path = tmp_path / "augmented_pair_model.hugiml"
    model.save_model(path)
    loaded = HUGIMLClassifierNative.load_model(path)
    assert loaded.get_augmented_pair_transforms() == catalog
    pd.testing.assert_frame_equal(
        loaded.get_augmented_pair_standardization().reset_index(drop=True),
        std.reset_index(drop=True),
    )
    np.testing.assert_allclose(loaded.predict_proba(X.iloc[:20]), model.predict_proba(X.iloc[:20]))

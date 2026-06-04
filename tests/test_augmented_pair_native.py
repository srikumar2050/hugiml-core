import numpy as np
import pandas as pd

from hugiml import HUGIMLClassifierNative
from hugiml.classifier import NativeAugmentedPairTransformBlock


def _data(n=240, p=6, seed=7):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    X.iloc[::19, 1] = np.nan
    y = (
        (
            (
                X["f0"].fillna(0) * X["f1"].fillna(0)
                + np.abs(X["f2"] - X["f3"])
                + 0.1 * rng.normal(size=n)
            )
            > 0.8
        )
        .astype(int)
        .to_numpy()
    )
    return X, y


def test_native_transform_block_generates_expected_standardized_features():
    X, y = _data()
    ig_scores = {c: 0.05 + i * 0.01 for i, c in enumerate(X.columns)}
    block = NativeAugmentedPairTransformBlock(max_features=6, budget_topK=10, min_source_ig=0.001)
    block.fit(X, y, ig_scores, {}, list(X.columns))

    assert getattr(block, "augmented_pair_native_used_", False) is True
    assert hasattr(block, "pair_reference_values_")
    assert all(
        "eligible_rate" in spec and "reference_raw_value" in spec for spec in block.kept_specs_
    )
    z_native = block.transform(X).toarray()
    assert z_native.shape[0] == len(X)
    assert z_native.shape[1] == len(block.kept_specs_)
    assert np.isfinite(z_native).all()

    selected = block.selected_ig_features_
    X_raw = X[selected].to_numpy(dtype=float, copy=True)
    manual = []
    pos = {c: j for j, c in enumerate(selected)}
    for spec, ref in zip(block.kept_specs_, block.pair_reference_values_):
        a, b = (pos[c] for c in spec["inputs"])
        observed = np.isfinite(X_raw[:, a]) & np.isfinite(X_raw[:, b])
        vals = np.full(X_raw.shape[0], ref, dtype=float)
        vals[observed] = (
            X_raw[observed, a] * X_raw[observed, b]
            if spec["operation"] == "product"
            else np.abs(X_raw[observed, a] - X_raw[observed, b])
        )
        manual.append(vals)
    manual = np.vstack(manual).T
    manual = (manual - block.scaler_mean_) / block.scaler_scale_
    np.testing.assert_allclose(z_native, manual, rtol=1e-6, atol=1e-6)


def test_classifier_l1_prediction_matches_base_behavior_when_augmentation_disabled():
    X, y = _data(n=180, p=5, seed=11)
    Xt, _ = _data(n=50, p=5, seed=13)
    clf_plain = HUGIMLClassifierNative(
        allCols=[[], list(X.columns), []],
        origColumns=list(X.columns),
        B=6,
        L=1,
        G=0.001,
        topK=20,
        adaptive_binning=False,
        feature_mode="patterns_only",
        n_jobs=1,
    )
    clf_explicit_off = HUGIMLClassifierNative(
        allCols=[[], list(X.columns), []],
        origColumns=list(X.columns),
        B=6,
        L=1,
        G=0.001,
        topK=20,
        adaptive_binning=False,
        feature_mode="patterns_only",
        n_jobs=1,
        augmented_pair_transforms=False,
    )
    clf_plain.fit(X, y)
    clf_explicit_off.fit(X, y)
    np.testing.assert_allclose(
        clf_plain.predict_proba(Xt), clf_explicit_off.predict_proba(Xt), rtol=0, atol=0
    )
    assert getattr(clf_plain, "augmented_pair_transforms_", []) == []
    assert getattr(clf_explicit_off, "augmented_pair_transforms_", []) == []


def test_adaptive_l2_uses_native_augmented_pair_fit_path():
    X, y = _data(n=260, p=8, seed=17)
    clf = HUGIMLClassifierNative(
        allCols=[[], list(X.columns), []],
        origColumns=list(X.columns),
        B=6,
        L=2,
        G=0.001,
        topK=12,
        adaptive_binning=True,
        feature_mode="patterns_only",
        n_jobs=1,
    )
    clf.fit(X, y)
    assert len(clf.get_augmented_pair_transforms()) <= 12
    assert getattr(clf._augmented_pair_block_, "augmented_pair_native_used_", False) is True
    proba = clf.predict_proba(X.iloc[:20])
    assert proba.shape == (20, 2)
    assert np.isfinite(proba).all()

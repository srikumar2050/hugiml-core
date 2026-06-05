import pickle

import numpy as np
import pandas as pd

from hugiml import HUGIMLClassifierNative


def _interaction_data(n=140, p=12, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(p)})
    y = ((X["x0"] * X["x1"] + X["x2"] * X["x3"]) > 0).astype(int).to_numpy()
    return X, y


def test_augmented_pair_transforms_are_persisted_for_l2_and_capped_by_topk():
    X, y = _interaction_data()
    model = HUGIMLClassifierNative(
        B=-1,
        adaptive_binning=True,
        L=2,
        G=1e-3,
        topK=30,
        augmented_pair_max_features=10,
    )
    model.fit(X, y)
    transforms = model.get_augmented_pair_transforms()
    assert isinstance(transforms, list)
    assert len(transforms) <= 30
    assert all(t["kind"] == "augmented_pair_transform" for t in transforms)
    assert all(t["used_in_hugiml_mining"] is False for t in transforms)
    assert all(t["eligible_for_L2"] is False for t in transforms)
    assert all("transform_ig" in t for t in transforms)
    assert all(t["augmented_pair_max_features"] == 10 for t in transforms)
    assert [t["rank"] for t in transforms] == list(range(1, len(transforms) + 1))


def test_augmented_pair_max_features_controls_candidate_pool():
    X, y = _interaction_data(p=8)
    model = HUGIMLClassifierNative(
        B=-1,
        adaptive_binning=True,
        L=2,
        topK=100,
        augmented_pair_max_features=5,
    )
    model.fit(X, y)
    transforms = model.get_augmented_pair_transforms()
    # C(5, 2) * 4 candidate transforms at most (4 ops: product, abs_diff, sum, signed_diff).
    assert len(transforms) <= 40
    assert all(t["augmented_pair_max_features"] == 5 for t in transforms)


def test_no_augmented_pair_transforms_for_l1_or_disabled():
    X, y = _interaction_data(p=4)
    l1 = HUGIMLClassifierNative(B=-1, adaptive_binning=True, L=1, topK=30)
    l1.fit(X, y)
    assert l1.get_augmented_pair_transforms() == []

    disabled = HUGIMLClassifierNative(
        B=-1,
        adaptive_binning=True,
        L=2,
        topK=30,
        augmented_pair_transforms=False,
    )
    disabled.fit(X, y)
    assert disabled.get_augmented_pair_transforms() == []


def test_augmented_pair_params_follow_sklearn_get_set_params():
    model = HUGIMLClassifierNative()
    params = model.get_params()
    assert params["augmented_pair_transforms"] is True
    assert params["augmented_pair_max_features"] == 10
    model.set_params(augmented_pair_transforms=False, augmented_pair_max_features=5)
    assert model.augmented_pair_transforms is False
    assert model.augmented_pair_max_features == 5


def test_augmented_pair_serialization_pickle_roundtrip():
    X, y = _interaction_data(n=100, p=8)
    model = HUGIMLClassifierNative(B=-1, adaptive_binning=True, L=2, topK=30)
    model.fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.augmented_pair_transforms is True
    assert restored.augmented_pair_max_features == 10
    assert restored.get_augmented_pair_transforms() == model.get_augmented_pair_transforms()

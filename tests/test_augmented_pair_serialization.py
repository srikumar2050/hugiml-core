import numpy as np
import pandas as pd

from hugiml import HUGIMLClassifierNative
from hugiml.serialization import load_model, save_model


def test_augmented_pair_state_round_trips_v3_serialization(tmp_path):
    rng = np.random.default_rng(123)
    n = 260
    X = pd.DataFrame({name: rng.normal(size=n) for name in ["a", "b", "c", "d", "e"]})
    y = (
        ((X["a"] * X["b"] + np.abs(X["c"] - X["d"]) + 0.15 * rng.normal(size=n)) > 0.2)
        .astype(int)
        .to_numpy()
    )

    clf = HUGIMLClassifierNative(
        B=-1,
        L=2,
        G=1e-4,
        topK=30,
        adaptive_binning=True,
        feature_mode="original_plus_patterns",
        augmented_pair_max_features=5,
    )
    clf.fit(X, y)
    assert len(clf.augmented_pair_transforms_) > 0

    before = clf.predict_proba(X.iloc[:30])
    path = tmp_path / "augmented_pair.hugiml"
    save_model(clf, path)
    loaded = load_model(path)
    after = loaded.predict_proba(X.iloc[:30])

    assert loaded._augmented_pair_block_ is not None
    assert loaded.augmented_pair_transforms_enabled_ is True
    assert loaded.augmented_pair_transforms_ == clf.augmented_pair_transforms_
    assert loaded.augmented_pair_selected_features_ == clf.augmented_pair_selected_features_
    np.testing.assert_allclose(after, before, rtol=0, atol=0)

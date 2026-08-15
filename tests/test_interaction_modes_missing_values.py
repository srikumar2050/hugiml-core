import warnings

import numpy as np
import pandas as pd
import pytest

from hugiml import HUGIMLClassifierNative
from hugiml.serialization import load_model, save_model

pytest.importorskip("_hugiml_core")


def _interaction_frame(n: int = 180, seed: int = 20260619) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{j}": rng.normal(size=n) for j in range(6)})
    score = X["x0"] * X["x1"] + 0.6 * (X["x2"] > 0).astype(float) - 0.2 * X["x3"]
    y = (score > float(np.median(score))).astype(int).to_numpy()
    return X, y


def _inject_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    rows = np.array([0, 5, 11, 17, 23, 29, 31, 37, 41, 47, 53, 59])
    for offset, col in enumerate(["x0", "x1", "x2"]):
        out.loc[rows[offset::3], col] = np.nan
    return out


@pytest.mark.parametrize(
    ("mode_name", "params"),
    [
        (
            "default_augmented_pair_interaction_information",
            {
                "L": 2,
                "B": -1,
                "G": 1e-4,
                "topK": 30,
                "adaptive_binning": True,
                "feature_mode": "original_plus_patterns",
            },
        ),
        (
            "augmented_pair_interaction_information",
            {
                "L": 2,
                "B": -1,
                "G": 1e-4,
                "topK": 30,
                "adaptive_binning": True,
                "feature_mode": "original_plus_patterns",
                "augmented_pair_transforms": True,
                "augmented_pair_mode": "interaction_information",
                "interaction_relaxed_mining": False,
            },
        ),
        (
            "interaction_relaxed_mining",
            {
                "L": 2,
                "B": -1,
                "G": 1e-4,
                "topK": 30,
                "adaptive_binning": True,
                "feature_mode": "original_plus_patterns",
                "augmented_pair_transforms": False,
                "interaction_relaxed_mining": True,
                "interaction_relaxed_feature_size": 4,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    ("train_missing", "test_missing"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_missing_value_matrix_for_interaction_paths(
    mode_name: str,
    params: dict,
    train_missing: bool,
    test_missing: bool,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HUGIML_MODEL_HMAC_KEY", "00" * 32)
    X, y = _interaction_frame()
    X_train = X.iloc[:120].reset_index(drop=True)
    y_train = y[:120]
    X_test = X.iloc[120:].reset_index(drop=True)

    if train_missing:
        X_train = _inject_missing_values(X_train)
    if test_missing:
        X_test = _inject_missing_values(X_test)

    clf = HUGIMLClassifierNative(**params)
    clf.fit(X_train, y_train)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        proba = clf.predict_proba(X_test)
    assert not any(issubclass(item.category, RuntimeWarning) for item in seen)
    assert proba.shape == (len(X_test), 2)
    assert np.isfinite(proba).all()

    pattern_block = clf.transform_patterns(X_test)
    downstream = clf._make_downstream_features(X_test, pattern_block, fit=False)
    downstream_dense = downstream.toarray() if hasattr(downstream, "toarray") else np.asarray(downstream)
    assert np.isfinite(downstream_dense).all()

    pattern_info = clf.get_pattern_info()
    feature_info = clf.feature_importances()
    composition = clf.get_model_composition()
    assert not pattern_info.empty
    assert not feature_info.empty
    assert isinstance(composition, dict)
    assert clf.model_summary()

    if mode_name in {
        "default_augmented_pair_interaction_information",
        "augmented_pair_interaction_information",
    }:
        assert clf.interaction_relaxed_mining is False
        assert clf.augmented_pair_transforms_enabled_ is True
        assert clf.augmented_pair_mode == "interaction_information"
        transforms = clf.get_augmented_pair_transforms()
        assert transforms
        assert {item["selected_by"] for item in transforms} == {"interaction_information"}
        augmented_block = clf._make_augmented_pair_features(X_test)
        assert np.isfinite(augmented_block.toarray()).all()
        if train_missing:
            assert any(float(item.get("missing_pair_rate", 0.0)) > 0.0 for item in transforms)
    else:
        assert clf.interaction_relaxed_mining is True
        assert clf.augmented_pair_transforms_enabled_ is False
        assert clf.get_augmented_pair_transforms() == []
        assert clf.explain_augmented_pair_effects().empty
        for column in ["pattern_origin", "survivor_led", "survivor_features"]:
            assert column in pattern_info.columns
            assert column in feature_info.columns
        assert "pattern_origin_counts" in composition

    path = tmp_path / f"{mode_name}_{int(train_missing)}_{int(test_missing)}.hugiml"
    save_model(clf, str(path))
    restored = load_model(str(path))
    restored_proba = restored.predict_proba(X_test)
    assert np.allclose(proba, restored_proba, atol=1e-10)
    assert not restored.feature_importances().empty
    assert isinstance(restored.get_model_composition(), dict)


def test_augmented_pair_selected_sources_missing_at_inference_use_reference_values():
    X, y = _interaction_frame(n=220, seed=20260620)
    clf = HUGIMLClassifierNative(
        L=2,
        B=-1,
        G=1e-4,
        topK=30,
        adaptive_binning=True,
        feature_mode="original_plus_patterns",
    )
    clf.fit(X.iloc[:150], y[:150])
    selected = list(clf.augmented_pair_selected_features_)
    assert selected

    X_test = X.iloc[150:166].copy().reset_index(drop=True)
    X_test.loc[:, selected] = np.nan

    augmented_block = clf._make_augmented_pair_features(X_test)
    augmented_dense = augmented_block.toarray()
    assert augmented_dense.shape[0] == len(X_test)
    assert np.isfinite(augmented_dense).all()

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        proba = clf.predict_proba(X_test)
    assert not any(issubclass(item.category, RuntimeWarning) for item in seen)
    assert proba.shape == (len(X_test), 2)
    assert np.isfinite(proba).all()

    effects = clf.explain_augmented_pair_effects()
    assert not effects.empty
    assert set(effects["pair_missing_policy"]) == {"reference_value_for_unavailable_pair"}


def test_relaxed_survivor_sources_missing_at_inference_remain_original_bin_patterns():
    X, y = _interaction_frame(n=240, seed=20260621)
    clf = HUGIMLClassifierNative(
        L=2,
        B=-1,
        G=1e-4,
        topK=30,
        adaptive_binning=True,
        feature_mode="original_plus_patterns",
        augmented_pair_transforms=False,
        interaction_relaxed_mining=True,
        interaction_relaxed_feature_size=4,
    )
    clf.fit(X.iloc[:160], y[:160])
    survivor_names = [row["name"] for row in clf.interaction_relaxed_mining_survivors_]
    assert survivor_names

    X_test = X.iloc[160:178].copy().reset_index(drop=True)
    X_test.loc[:, survivor_names] = np.nan

    pattern_block = clf.transform_patterns(X_test)
    downstream = clf._make_downstream_features(X_test, pattern_block, fit=False)
    downstream_dense = downstream.toarray() if hasattr(downstream, "toarray") else np.asarray(downstream)
    assert np.isfinite(downstream_dense).all()
    assert not any(str(name).startswith("augmented_pair:") for name in clf.get_downstream_features())

    pattern_info = clf.get_pattern_info()
    assert pattern_info["survivor_led"].any()
    assert set(pattern_info.loc[pattern_info["survivor_led"], "pattern_origin"]) == {
        "interaction_relaxed"
    }

    proba = clf.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.isfinite(proba).all()

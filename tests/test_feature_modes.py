import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split

from hugiml import HUGIMLClassifierNative

MODES = [
    "patterns_only",
    "original_plus_patterns",
    "original_plus_interactions",
]


def _frame_dataset(kind="breast"):
    if kind == "moons":
        X, y = make_moons(n_samples=220, noise=0.22, random_state=7)
        cols = ["col0", "col1"]
    else:
        data = load_breast_cancer()
        X, y = data.data[:260], data.target[:260]
        cols = [f"col{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=cols)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


def _clf(mode="patterns_only"):
    # Small but non-trivial settings keep this suite fast while exercising L=1/L=2 patterns.
    return HUGIMLClassifierNative(
        allCols=[[], ["col0", "col1"], []],
        origColumns=["col0", "col1"],
        B=5,
        L=2,
        G=1e-3,
        topK=30,
        adaptive_binning=True,
        feature_mode=mode,
    )


def _clf_for_frame(X, mode="patterns_only"):
    cols = list(X.columns)
    return HUGIMLClassifierNative(
        allCols=[[], cols, []],
        origColumns=cols,
        B=5,
        L=2,
        G=1e-3,
        topK=40,
        adaptive_binning=True,
        feature_mode=mode,
    )


def test_default_feature_mode_is_patterns_only():
    clf = HUGIMLClassifierNative()
    assert clf.feature_mode == "patterns_only"
    assert clf.get_params()["feature_mode"] == "patterns_only"


@pytest.mark.parametrize("mode", MODES)
def test_three_feature_modes_fit_predict_proba_and_transform(mode):
    Xtr, Xte, ytr, yte = _frame_dataset("moons")
    clf = _clf(mode).fit(Xtr, ytr)
    preds = clf.predict(Xte)
    proba = clf.predict_proba(Xte)
    Z = clf.transform(Xte)

    assert preds.shape == (len(yte),)
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert Z.shape[0] == len(yte)
    assert Z.shape[1] == len(clf.patterns_)


@pytest.mark.parametrize("mode", MODES)
def test_fit_transform_remains_pattern_matrix_only(mode):
    Xtr, Xte, ytr, yte = _frame_dataset("moons")
    clf = _clf(mode)
    Z_fit = clf.fit_transform(Xtr, ytr)
    Z_transform = clf.transform(Xtr)
    assert Z_fit.shape == Z_transform.shape
    assert Z_fit.shape[1] == len(clf.patterns_)


def test_hybrid_modes_have_downstream_feature_count_at_least_patterns_only():
    Xtr, Xte, ytr, yte = _frame_dataset("moons")
    fitted = {mode: _clf(mode).fit(Xtr, ytr) for mode in MODES}
    n_patterns = fitted["patterns_only"].x_train_hup_.shape[1]
    assert fitted["patterns_only"].x_train_downstream_.shape[1] == n_patterns
    assert fitted["original_plus_patterns"].x_train_downstream_.shape[1] == Xtr.shape[1] + n_patterns
    assert fitted["original_plus_interactions"].x_train_downstream_.shape[1] <= Xtr.shape[1] + n_patterns
    assert fitted["original_plus_interactions"].x_train_downstream_.shape[1] >= Xtr.shape[1]


@pytest.mark.parametrize("mode", MODES)
def test_feature_importances_and_summary_reflect_mode(mode):
    Xtr, Xte, ytr, yte = _frame_dataset("moons")
    clf = _clf(mode).fit(Xtr, ytr)
    fi = clf.feature_importances()
    assert not fi.empty
    assert len(fi) == clf.x_train_downstream_.shape[1]
    summary = clf.model_summary()
    assert f"Feature mode: {mode}" in summary


def test_invalid_feature_mode_rejected_on_fit():
    Xtr, Xte, ytr, yte = _frame_dataset("moons")
    clf = _clf("not_a_mode")
    with pytest.raises(Exception):
        clf.fit(Xtr, ytr)


def test_sklearn_clone_preserves_feature_mode():
    clf = HUGIMLClassifierNative(feature_mode="original_plus_patterns")
    cloned = clone(clf)
    assert cloned.feature_mode == "original_plus_patterns"


@pytest.mark.parametrize("mode", MODES)
def test_save_load_preserves_feature_mode_and_predictions(mode, tmp_path):
    Xtr, Xte, ytr, yte = _frame_dataset("moons")
    clf = _clf(mode).fit(Xtr, ytr)
    p1 = clf.predict_proba(Xte)
    path = tmp_path / f"{mode}.hugiml"
    clf.save_model(path)
    loaded = HUGIMLClassifierNative.load_model(path)
    p2 = loaded.predict_proba(Xte)
    assert loaded.feature_mode == mode
    np.testing.assert_allclose(p1, p2, rtol=1e-10, atol=1e-10)


def test_all_modes_share_same_pattern_transform_when_mining_params_match():
    Xtr, Xte, ytr, yte = _frame_dataset("moons")
    matrices = []
    pattern_counts = []
    for mode in MODES:
        clf = _clf(mode).fit(Xtr, ytr)
        matrices.append(clf.transform(Xte).toarray())
        pattern_counts.append(len(clf.patterns_))
    assert len(set(pattern_counts)) == 1
    np.testing.assert_array_equal(matrices[0], matrices[1])
    np.testing.assert_array_equal(matrices[0], matrices[2])


def test_original_plus_interactions_uses_only_interaction_patterns_after_original_features():
    Xtr, Xte, ytr, yte = _frame_dataset("breast")
    clf = _clf_for_frame(Xtr, "original_plus_interactions").fit(Xtr, ytr)
    assert hasattr(clf, "_interaction_pattern_mask_")
    assert clf.x_train_downstream_.shape[1] == Xtr.shape[1] + int(clf._interaction_pattern_mask_.sum())

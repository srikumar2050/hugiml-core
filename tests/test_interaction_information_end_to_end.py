import numpy as np
import pandas as pd
import pytest

from hugiml import HUGIMLClassifier, HUGIMLClassifierNative

_core = pytest.importorskip("_hugiml_core")


def _xor_data(n=300, p=6, seed=11):
    """Two XOR-coupled binary features plus uninformative noise columns.

    f0/f1 individually carry little marginal signal but interact strongly,
    which is exactly the case interaction_relaxed_mining and
    augmented_pair_mode='interaction_information' are designed to surface.
    """
    rng = np.random.default_rng(seed)
    f0 = rng.integers(0, 2, size=n).astype(float)
    f1 = rng.integers(0, 2, size=n).astype(float)
    noise = rng.normal(size=(n, p - 2))
    X = pd.DataFrame(
        np.column_stack([f0, f1, noise]),
        columns=["f0", "f1", *[f"n{i}" for i in range(p - 2)]],
    )
    y = pd.Series(f0.astype(int) ^ f1.astype(int))
    return X, y


class TestInteractionPathEndToEnd:
    def test_default_l2_uses_augmented_pair_transforms(self):
        X, y = _xor_data()
        clf = HUGIMLClassifierNative(
            B=5,
            L=2,
            G=0.001,
            topK=20,
        )
        assert clf.interaction_relaxed_mining is False
        assert clf.augmented_pair_transforms is True
        assert clf.augmented_pair_mode == "interaction_information"
        clf.fit(X, y)
        assert clf.augmented_pair_transforms_enabled_ is True
        assert len(clf.augmented_pair_selected_features_) > 0
        proba = clf.predict_proba(X.iloc[:30])
        assert proba.shape == (30, 2)
        assert np.isfinite(proba).all()

    def test_survivors_recorded_and_bounded_by_feature_size(self):
        X, y = _xor_data(n=320, p=8, seed=29)
        clf = HUGIMLClassifierNative(
            B=5,
            L=2,
            G=0.001,
            topK=20,
            augmented_pair_transforms=False,
            interaction_relaxed_mining=True,
            interaction_relaxed_feature_size=3,
        )
        clf.fit(X, y)
        survivors = clf.interaction_relaxed_mining_survivors_
        assert 0 < len(survivors) <= 3
        assert {"f0", "f1"}.issubset({row["name"] for row in survivors})

    def test_survivor_led_patterns_are_auditable_without_new_feature_family(self):
        X, y = _xor_data(n=320, p=8, seed=97)
        clf = HUGIMLClassifierNative(
            B=5,
            L=2,
            G=0.001,
            topK=20,
            feature_mode="original_plus_patterns",
            augmented_pair_transforms=False,
            interaction_relaxed_mining=True,
            interaction_relaxed_feature_size=4,
        )
        clf.fit(X, y)

        info = clf.get_pattern_info()
        expected_cols = {
            "pattern_origin",
            "survivor_led",
            "survivor_features",
            "survivor_feature_count",
            "survivor_min_marginal_ig",
            "survivor_max_interaction_score",
            "survivor_best_partners",
        }
        assert expected_cols.issubset(info.columns)
        assert info["survivor_led"].any()
        assert set(info.loc[info["survivor_led"], "pattern_origin"]) == {
            "interaction_relaxed"
        }

        fi = clf.feature_importances()
        pattern_rows = fi[fi["feature_type"] == "pattern"]
        assert expected_cols.issubset(fi.columns)
        assert pattern_rows["survivor_led"].any()
        assert set(fi["feature_type"]).issubset({"original", "pattern"})
        assert not any(str(name).startswith("augmented_pair:") for name in clf.get_downstream_features())

        composition = clf.get_model_composition()
        assert composition["downstream_feature_counts"].get("augmented_pair") == 0
        assert composition["pattern_origin_counts"].get("survivor_led", 0) > 0

    def test_survivor_led_audit_columns_survive_serialization(self, tmp_path):
        from hugiml.serialization import load_model, save_model

        X, y = _xor_data(n=280, p=8, seed=101)
        clf = HUGIMLClassifierNative(
            B=5,
            L=2,
            G=0.001,
            topK=20,
            feature_mode="original_plus_patterns",
            augmented_pair_transforms=False,
            interaction_relaxed_mining=True,
            interaction_relaxed_feature_size=4,
        )
        clf.fit(X, y)
        path = tmp_path / "relaxed_audit.hugiml"
        save_model(clf, str(path))

        restored = load_model(str(path))
        fi = restored.feature_importances()
        assert "pattern_origin" in fi.columns
        assert "survivor_led" in fi.columns
        assert fi.loc[fi["feature_type"] == "pattern", "survivor_led"].any()
        composition = restored.get_model_composition()
        assert composition["pattern_origin_counts"].get("survivor_led", 0) > 0

    def test_disabling_relaxation_still_fits_and_predicts(self):
        X, y = _xor_data(seed=31)
        clf = HUGIMLClassifierNative(
            B=5,
            L=2,
            G=0.001,
            topK=20,
            augmented_pair_transforms=False,
            interaction_relaxed_mining=False,
        )
        clf.fit(X, y)
        proba = clf.predict_proba(X.iloc[:10])
        assert proba.shape == (10, 2)

    def test_explicit_relaxed_mining_is_mutually_exclusive_with_augmented_pairs_at_l2(self):
        X, y = _xor_data(seed=5)
        clf = HUGIMLClassifierNative(
            B=5,
            L=2,
            G=0.001,
            topK=20,
            augmented_pair_transforms=True,
            interaction_relaxed_mining=True,
        )
        with pytest.raises(Exception, match="mutually exclusive"):
            clf.fit(X, y)

    def test_invalid_l_raises_with_relaxation_enabled(self):
        X, y = _xor_data(seed=41)
        clf = HUGIMLClassifierNative(
            B=5,
            L=0,
            G=0.001,
            topK=20,
            augmented_pair_transforms=False,
            interaction_relaxed_mining=True,
        )
        with pytest.raises(Exception, match="interaction_relaxed_mining"):
            clf.fit(X, y)

    def test_high_level_classifier_round_trips_through_serialization(self, tmp_path):
        from hugiml.serialization import load_model, save_model

        X, y = _xor_data(n=260, seed=53)
        clf = HUGIMLClassifier(
            B=5,
            L=2,
            G=0.001,
            topK=15,
            augmented_pair_transforms=False,
            interaction_relaxed_mining=True,
            interaction_relaxed_feature_size=4,
        )
        clf.fit(X, y)
        proba_before = clf.predict_proba(X)

        path = tmp_path / "relaxed_model.hugiml"
        save_model(clf, str(path))
        reloaded = load_model(str(path))

        proba_after = reloaded.predict_proba(X)
        np.testing.assert_allclose(proba_before, proba_after)
        assert reloaded.interaction_relaxed_mining is True
        assert reloaded.interaction_relaxed_feature_size == 4


class TestAugmentedPairInteractionInformationEndToEnd:
    def test_default_mode_selects_sources_and_predicts(self):
        X, y = _xor_data(n=280, p=10, seed=61)
        clf = HUGIMLClassifierNative(
            B=5,
            L=2,
            G=0.001,
            topK=15,
        )
        assert clf.interaction_relaxed_mining is False
        assert clf.augmented_pair_transforms is True
        assert clf.augmented_pair_mode == "interaction_information"
        clf.fit(X, y)
        selected = clf.augmented_pair_selected_features_
        assert 0 < len(selected) <= clf.aug_feature_size
        proba = clf.predict_proba(X.iloc[:20])
        assert proba.shape == (20, 2)
        assert np.isfinite(proba).all()

    def test_ii_partner_size_restricts_candidate_search_without_changing_output_size(self):
        X, y = _xor_data(n=300, p=12, seed=67)
        clf = HUGIMLClassifierNative(
            B=5,
            L=2,
            G=0.001,
            topK=15,
            interaction_relaxed_mining=False,
            augmented_pair_transforms=True,
            augmented_pair_mode="interaction_information",
            aug_feature_size=4,
            ii_partner_size=3,
        )
        clf.fit(X, y)
        assert len(clf.augmented_pair_selected_features_) <= 4

    def test_marginal_ig_mode_still_available(self):
        X, y = _xor_data(n=260, seed=71)
        clf = HUGIMLClassifierNative(
            B=5,
            L=2,
            G=0.001,
            topK=15,
            interaction_relaxed_mining=False,
            augmented_pair_transforms=True,
            augmented_pair_mode="marginal_ig",
            max_pair_features=4,
        )
        clf.fit(X, y)
        proba = clf.predict_proba(X.iloc[:10])
        assert proba.shape == (10, 2)

    def test_invalid_mode_raises(self):
        X, y = _xor_data(seed=83)
        clf = HUGIMLClassifierNative(
            B=5,
            L=2,
            G=0.001,
            topK=15,
            interaction_relaxed_mining=False,
            augmented_pair_transforms=True,
            augmented_pair_mode="not_a_real_mode",
        )
        with pytest.raises(Exception, match="augmented_pair_mode"):
            clf.fit(X, y)


def test_relaxed_adaptive_uses_pair_context_without_pair_columns():
    rng = np.random.default_rng(113)
    n = 360
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    noise = rng.normal(size=(n, 4))
    y = ((x0 > 0.0) ^ (x1 > 0.0)).astype(np.int64)
    X = pd.DataFrame(
        np.column_stack([x0, x1, noise]),
        columns=["x0", "x1", "n0", "n1", "n2", "n3"],
    )
    clf = HUGIMLClassifierNative(
        L=2,
        G=0.001,
        topK=30,
        adaptive_binning=True,
        augmented_pair_transforms=False,
        interaction_relaxed_mining=True,
        interaction_relaxed_feature_size=4,
    )
    clf.fit(X, y)

    pairs = {(row["left"], row["right"]) for row in clf._interaction_relaxed_adaptive_pairs_}
    assert ("x0", "x1") in pairs or ("x1", "x0") in pairs
    assert clf._interaction_relaxed_adaptive_evidence_["x0"]["mode"] == "pair_aware"
    assert clf.per_feature_b_["x0"] <= 8
    assert clf.per_feature_b_["x1"] <= 8
    assert all("interaction_pairbin" not in str(name) for name in clf.feature_names_in_)
    assert all("interaction_pairbin" not in str(item) for item in clf.get_hug_features())


def test_relaxed_adaptive_survivors_reused_for_mining():
    rng = np.random.default_rng(127)
    n = 300
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    X = pd.DataFrame({"x0": x0, "x1": x1, "n0": rng.normal(size=n)})
    y = (np.abs(x0 - x1) < 0.5).astype(np.int64)
    clf = HUGIMLClassifierNative(
        L=2,
        G=0.001,
        topK=25,
        adaptive_binning=True,
        augmented_pair_transforms=False,
        interaction_relaxed_mining=True,
        interaction_relaxed_feature_size=3,
    )
    clf.fit(X, y)
    survivor_names = {row["name"] for row in clf.interaction_relaxed_mining_survivors_}
    assert {"x0", "x1"}.issubset(survivor_names)
    assert clf.predict_proba(X.iloc[:12]).shape == (12, 2)

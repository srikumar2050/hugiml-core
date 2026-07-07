# Copyright 2026 Srikumar Krishnamoorthy
#
# Regression coverage for the v1.1.15 change areas from v1.1.14:
#   (1) Interaction-relaxed mining and EU correlation
#   (2) No-pattern constant-prior fallback
#   (3) Benchmark runner configs for interpretable baseline models
# Plus serialization round-trip and downstream interpretability API checks.

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split

import hugiml
from hugiml import HUGIMLClassifier, HUGIMLClassifierNative
from hugiml.hyperparameter_configs import BASELINE_MODEL_GRIDS, get_hugiml_grid
from hugiml.serialization import MIN_SCHEMA_VERSION, MODEL_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trivial_data(n=60, p=4, random_state=42):
    """Small dataset for fast tests."""
    rng = np.random.RandomState(random_state)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(rng.randint(0, 2, size=n), name="target")
    return X, y


def _make_noise_only_data(n=50, p=3, random_state=99):
    """Pure noise data — no patterns should survive with strict G."""
    rng = np.random.RandomState(random_state)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"noise{i}" for i in range(p)])
    y = pd.Series(rng.randint(0, 2, size=n), name="target")
    return X, y


def _make_xor_data(n=200, random_state=0):
    """XOR-structured data where marginal correlations are near zero
    but joint (interaction) signal is strong."""
    rng = np.random.RandomState(random_state)
    x1 = rng.randint(0, 2, size=n).astype(float)
    x2 = rng.randint(0, 2, size=n).astype(float)
    y = (x1.astype(int) ^ x2.astype(int)).astype(int)
    # add noise columns
    X = pd.DataFrame({
        "x1": x1, "x2": x2,
        "noise1": rng.randn(n), "noise2": rng.randn(n),
    })
    return X, pd.Series(y, name="target")


# ===================================================================
# (1) ZERO-PATTERN FALLBACK TESTS
# ===================================================================

class TestZeroPatternFallback:
    """Test the constant-prior fallback when no patterns are mined."""

    @pytest.fixture
    def fallback_clf(self):
        """Fit a classifier with extreme G so no patterns survive."""
        X, y = _make_noise_only_data()
        clf = HUGIMLClassifier(
            B=2, L=1, G=1e6,   # very high G makes retained patterns unlikely
            topK=5,
            feature_mode="patterns_only",
            adaptive_binning=False,
            verbose=False,
        )
        X_prep, y_prep = clf.prepareXy(X, y)
        clf.fit(X_prep, y_prep)
        return clf, X_prep, y_prep

    def test_fit_succeeds_no_exception(self, fallback_clf):
        """fit() must not raise when no patterns found in patterns_only mode."""
        clf, _, _ = fallback_clf
        assert clf is not None

    def test_fallback_active_flag(self, fallback_clf):
        clf, _, _ = fallback_clf
        assert clf.fallback_active_ is True
        assert clf.fallback_strategy_ == "constant_prior"
        assert clf._is_constant_prior_fallback_active()

    def test_predict_proba_shape(self, fallback_clf):
        clf, X, _ = fallback_clf
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), len(clf.classes_))

    def test_predict_proba_valid_distribution(self, fallback_clf):
        clf, X, _ = fallback_clf
        proba = clf.predict_proba(X)
        # rows must sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
        # all non-negative
        assert np.all(proba >= 0)

    def test_predict_proba_constant_rows(self, fallback_clf):
        """All rows should return the same class-prior distribution."""
        clf, X, _ = fallback_clf
        proba = clf.predict_proba(X)
        for i in range(1, len(proba)):
            np.testing.assert_array_almost_equal(proba[0], proba[i])

    def test_predict_returns_majority_class(self, fallback_clf):
        clf, X, y = fallback_clf
        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        # all predictions should be the majority class
        unique_preds = np.unique(preds)
        assert len(unique_preds) == 1
        assert unique_preds[0] == clf.fallback_majority_class_

    def test_predict_dtype_matches_classes(self, fallback_clf):
        clf, X, _ = fallback_clf
        preds = clf.predict(X)
        assert preds.dtype == np.asarray(clf.classes_).dtype

    def test_transform_returns_empty_csr(self, fallback_clf):
        clf, X, _ = fallback_clf
        Z = clf.transform(X)
        assert issparse(Z)
        assert Z.shape == (len(X), 0)

    def test_patterns_list_empty(self, fallback_clf):
        clf, _, _ = fallback_clf
        assert len(clf.patterns_) == 0

    def test_model_summary_no_error(self, fallback_clf):
        clf, _, _ = fallback_clf
        summary = clf.model_summary()
        assert isinstance(summary, str)
        assert "HUGIMLClassifier" in summary

    def test_feature_importances_returns_empty_df(self, fallback_clf):
        clf, _, _ = fallback_clf
        imp = clf.feature_importances()
        assert isinstance(imp, pd.DataFrame)
        assert len(imp) == 0

    def test_get_model_composition(self, fallback_clf):
        clf, _, _ = fallback_clf
        comp = clf.get_model_composition()
        assert isinstance(comp, dict)
        assert comp["n_patterns_mined"] == 0
        assert comp["downstream_feature_counts"]["total"] == 0

    def test_fit_metadata_populated(self, fallback_clf):
        clf, _, _ = fallback_clf
        assert hasattr(clf, "fit_metadata_")
        assert clf.fit_metadata_.n_patterns == 0

    def test_fallback_class_prior_matches_training(self, fallback_clf):
        clf, _, y = fallback_clf
        y_arr = np.asarray(y)
        classes = clf.classes_
        expected_prior = np.array([np.mean(y_arr == c) for c in classes])
        np.testing.assert_allclose(clf.fallback_class_prior_, expected_prior, atol=1e-10)

    def test_get_pattern_info_empty(self, fallback_clf):
        clf, _, _ = fallback_clf
        info = clf.get_pattern_info()
        assert isinstance(info, pd.DataFrame)
        assert len(info) == 0

    def test_model_has_pipeline_model(self, fallback_clf):
        """model_ should be a Pipeline wrapping a DummyClassifier."""
        from sklearn.pipeline import Pipeline
        clf, _, _ = fallback_clf
        assert isinstance(clf.model_, Pipeline)
        assert "clf" in dict(clf.model_.steps)

    def test_check_health_does_not_raise(self, fallback_clf):
        clf, _, _ = fallback_clf
        # _check_health should return silently for fallback models
        clf._check_health()

    def test_monitoring_with_fallback(self, fallback_clf):
        clf, X, _ = fallback_clf
        clf.enable_monitoring()
        clf.predict_proba(X)
        report = clf.monitor.report()
        assert report is not None


class TestZeroPatternFallbackMulticlass:
    """Verify fallback works with >2 classes."""

    def test_multiclass_fallback(self):
        X, y = load_iris(return_X_y=True, as_frame=True)
        clf = HUGIMLClassifier(
            B=2, L=1, G=1e6,
            topK=5,
            feature_mode="patterns_only",
            adaptive_binning=False,
            verbose=False,
        )
        X_p, y_p = clf.prepareXy(X, y)
        clf.fit(X_p, y_p)
        assert clf.fallback_active_
        proba = clf.predict_proba(X_p)
        assert proba.shape == (len(X_p), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
        preds = clf.predict(X_p)
        assert preds.shape == (len(X_p),)


class TestZeroPatternNonPatternsOnlyMode:
    """In original_plus_patterns mode, zero patterns should NOT trigger
    fallback — the classifier should still work using original features."""

    def test_original_plus_patterns_no_fallback(self):
        X, y = _make_noise_only_data()
        clf = HUGIMLClassifier(
            B=2, L=1, G=1e6,
            topK=5,
            feature_mode="original_plus_patterns",
            adaptive_binning=False,
            verbose=False,
        )
        X_p, y_p = clf.prepareXy(X, y)
        clf.fit(X_p, y_p)
        # Should NOT be fallback — original features still available
        assert not clf._is_constant_prior_fallback_active()
        proba = clf.predict_proba(X_p)
        assert proba.shape[0] == len(X_p)


# ===================================================================
# (2) SERIALIZATION ROUND-TRIP TESTS
# ===================================================================

class TestFallbackSerialization:
    """Save/load round-trip for fallback models."""

    def test_save_load_roundtrip(self):
        X, y = _make_noise_only_data()
        clf = HUGIMLClassifier(
            B=2, L=1, G=1e6,
            topK=5,
            feature_mode="patterns_only",
            adaptive_binning=False,
        )
        X_p, y_p = clf.prepareXy(X, y)
        clf.fit(X_p, y_p)
        assert clf.fallback_active_

        with tempfile.NamedTemporaryFile(suffix=".hugiml", delete=False) as f:
            path = f.name
        try:
            clf.save_model(path)
            clf2 = HUGIMLClassifier.load_model(path)

            assert clf2.fallback_active_
            assert clf2.fallback_strategy_ == "constant_prior"

            proba1 = clf.predict_proba(X_p)
            proba2 = clf2.predict_proba(X_p)
            np.testing.assert_array_almost_equal(proba1, proba2)

            preds1 = clf.predict(X_p)
            preds2 = clf2.predict(X_p)
            np.testing.assert_array_equal(preds1, preds2)
        finally:
            os.unlink(path)

    def test_schema_version_matches_current_release(self):
        assert MODEL_SCHEMA_VERSION == 8

    def test_min_schema_backward_compat(self):
        assert MIN_SCHEMA_VERSION == 1

    def test_fallback_state_in_manifest(self):
        """Verify fallback_state is persisted in clf_fit.json."""
        import zipfile
        X, y = _make_noise_only_data()
        clf = HUGIMLClassifier(
            B=2, L=1, G=1e6,
            topK=5,
            feature_mode="patterns_only",
            adaptive_binning=False,
        )
        X_p, y_p = clf.prepareXy(X, y)
        clf.fit(X_p, y_p)

        with tempfile.NamedTemporaryFile(suffix=".hugiml", delete=False) as f:
            path = f.name
        try:
            clf.save_model(path)
            with zipfile.ZipFile(path, "r") as zf:
                clf_fit = json.loads(zf.read("clf_fit.json"))
            fb = clf_fit["fallback_state"]
            assert fb["active"] is True
            assert fb["strategy"] == "constant_prior"
            assert fb["majority_class"] is not None
        finally:
            os.unlink(path)

    def test_fallback_class_prior_array_persisted(self):
        """Verify fallback_class_prior_ survives npz round-trip."""
        X, y = _make_noise_only_data()
        clf = HUGIMLClassifier(
            B=2, L=1, G=1e6,
            topK=5,
            feature_mode="patterns_only",
            adaptive_binning=False,
        )
        X_p, y_p = clf.prepareXy(X, y)
        clf.fit(X_p, y_p)

        with tempfile.NamedTemporaryFile(suffix=".hugiml", delete=False) as f:
            path = f.name
        try:
            clf.save_model(path)
            clf2 = HUGIMLClassifier.load_model(path)
            np.testing.assert_array_almost_equal(
                clf.fallback_class_prior_, clf2.fallback_class_prior_
            )
        finally:
            os.unlink(path)


class TestNormalSerialization:
    """Ensure normal (non-fallback) models still serialize correctly."""

    def test_normal_model_roundtrip(self):
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = HUGIMLClassifier(B=3, L=1, G=1e-3, topK=20, adaptive_binning=False)
        X_tr_p, y_tr_p = clf.prepareXy(X_tr, y_tr)
        clf.fit(X_tr_p, y_tr_p)
        assert not getattr(clf, "fallback_active_", False)

        with tempfile.NamedTemporaryFile(suffix=".hugiml", delete=False) as f:
            path = f.name
        try:
            clf.save_model(path)
            clf2 = HUGIMLClassifier.load_model(path)
            assert not getattr(clf2, "fallback_active_", False)
            assert len(clf2.patterns_) == len(clf.patterns_)

            X_te_p, _ = clf2.prepareXy(X_te, y_te)
            proba1 = clf.predict_proba(X_te_p)
            proba2 = clf2.predict_proba(X_te_p)
            np.testing.assert_array_almost_equal(proba1, proba2, decimal=5)
        finally:
            os.unlink(path)


# ===================================================================
# (3) INTERACTION RELAXATION MINING
# ===================================================================

class TestInteractionRelaxedMining:
    """Tests for interaction_relaxed_mining and EU correlation logic."""

    def test_interaction_relaxed_mining_flag(self):
        """Verify the interaction_relaxed_mining parameter is accepted."""
        clf = HUGIMLClassifier(
            B=3, L=2, G=1e-3,
            topK=20,
            interaction_relaxed_mining=True,
            adaptive_binning=False,
        )
        assert clf.interaction_relaxed_mining is True

    def test_interaction_relaxed_fit_runs(self):
        """Fit with interaction_relaxed_mining=True should not raise."""
        X, y = _make_xor_data(n=200)
        clf = HUGIMLClassifier(
            B=3, L=2, G=1e-3,
            topK=30,
            interaction_relaxed_mining=True,
            interaction_relaxed_feature_size=4,
            augmented_pair_transforms=False,
            adaptive_binning=False,
            verbose=False,
        )
        X_p, y_p = clf.prepareXy(X, y)
        clf.fit(X_p, y_p)
        proba = clf.predict_proba(X_p)
        assert proba.shape == (len(X_p), 2)

    def test_interaction_relaxed_survivors_stored(self):
        """After fit, survivors should be stored as a list of dicts."""
        X, y = _make_xor_data(n=200)
        clf = HUGIMLClassifier(
            B=3, L=2, G=1e-3,
            topK=30,
            interaction_relaxed_mining=True,
            interaction_relaxed_feature_size=4,
            augmented_pair_transforms=False,
            adaptive_binning=False,
            verbose=False,
        )
        X_p, y_p = clf.prepareXy(X, y)
        clf.fit(X_p, y_p)
        survivors = getattr(clf, "interaction_relaxed_mining_survivors_", None)
        assert survivors is not None
        assert isinstance(survivors, list)

    def test_interaction_relaxed_serialization(self):
        """Interaction relaxed mining state should survive save/load."""
        X, y = _make_xor_data(n=200)
        clf = HUGIMLClassifier(
            B=3, L=2, G=1e-3,
            topK=30,
            interaction_relaxed_mining=True,
            interaction_relaxed_feature_size=4,
            augmented_pair_transforms=False,
            adaptive_binning=False,
        )
        X_p, y_p = clf.prepareXy(X, y)
        clf.fit(X_p, y_p)

        with tempfile.NamedTemporaryFile(suffix=".hugiml", delete=False) as f:
            path = f.name
        try:
            clf.save_model(path)
            clf2 = HUGIMLClassifier.load_model(path)
            assert clf2.interaction_relaxed_mining is True
            proba1 = clf.predict_proba(X_p)
            proba2 = clf2.predict_proba(X_p)
            np.testing.assert_array_almost_equal(proba1, proba2, decimal=5)
        finally:
            os.unlink(path)


# ===================================================================
# (4) BENCHMARK / HYPERPARAMETER CONFIG TESTS
# ===================================================================

class TestBenchmarkConfigs:
    """Verify the updated EBM and RuleFit benchmark grids are valid."""

    def test_ebm_grid_has_interactions(self):
        grid = BASELINE_MODEL_GRIDS.get("EBM", {})
        assert "interactions" in grid
        assert grid["interactions"] == [0, 5]

    def test_ebm_grid_max_bins_reduced(self):
        grid = BASELINE_MODEL_GRIDS.get("EBM", {})
        assert "max_bins" in grid
        assert grid["max_bins"] == [32, 64]
        # Verify old values [128, 256] are no longer present
        assert 128 not in grid["max_bins"]

    def test_ebm_grid_no_subsample(self):
        """The EBM grid omits subsample because that parameter is unsupported."""
        grid = BASELINE_MODEL_GRIDS.get("EBM", {})
        assert "subsample" not in grid

    def test_rulefit_grid_registered(self):
        grid = BASELINE_MODEL_GRIDS.get("RuleFit", {})
        assert "n_estimators" in grid
        assert "max_rules" in grid
        assert "tree_size" in grid

    def test_rulefit_grid_values(self):
        grid = BASELINE_MODEL_GRIDS["RuleFit"]
        assert grid["n_estimators"] == [50, 100]
        assert grid["max_rules"] == [50, 100]
        assert grid["tree_size"] == [5, 10]

    def test_lgbm_grid_no_subsample(self):
        """The LightGBM grid omits subsample for compact benchmark coverage."""
        grid = BASELINE_MODEL_GRIDS.get("LightGBM", {})
        assert "subsample" not in grid

    def test_hugiml_grid_unchanged(self):
        """HUG-IML's own grids should be unaffected."""
        grid = get_hugiml_grid()
        assert isinstance(grid, dict)


# ===================================================================
# (5) DOWNSTREAM INTERPRETABILITY API TESTS
# ===================================================================

class TestDownstreamAPI:
    """Verify interpretability APIs remain intact after release changes."""

    @pytest.fixture(scope="class")
    def fitted_clf(self):
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = HUGIMLClassifier(B=3, L=1, G=1e-3, topK=20, adaptive_binning=False)
        X_tr_p, y_tr_p = clf.prepareXy(X_tr, y_tr)
        clf.fit(X_tr_p, y_tr_p)
        X_te_p, _ = clf.prepareXy(X_te, y_te)
        return clf, X_tr_p, y_tr_p, X_te_p, y_te

    def test_predict_proba(self, fitted_clf):
        clf, _, _, X_te, _ = fitted_clf
        proba = clf.predict_proba(X_te)
        assert proba.shape[0] == len(X_te)
        assert proba.shape[1] == len(clf.classes_)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict(self, fitted_clf):
        clf, _, _, X_te, _ = fitted_clf
        preds = clf.predict(X_te)
        assert preds.shape == (len(X_te),)

    def test_transform(self, fitted_clf):
        clf, _, _, X_te, _ = fitted_clf
        Z = clf.transform(X_te)
        assert issparse(Z)
        assert Z.shape[0] == len(X_te)

    def test_model_summary(self, fitted_clf):
        clf, _, _, _, _ = fitted_clf
        summary = clf.model_summary()
        assert isinstance(summary, str)
        assert len(summary) > 100

    def test_feature_importances(self, fitted_clf):
        clf, _, _, _, _ = fitted_clf
        imp = clf.feature_importances()
        assert isinstance(imp, pd.DataFrame)
        assert len(imp) > 0
        assert "coefficient" in imp.columns

    def test_get_model_composition(self, fitted_clf):
        clf, _, _, _, _ = fitted_clf
        comp = clf.get_model_composition()
        assert "feature_mode" in comp
        assert "n_patterns_mined" in comp
        assert comp["n_patterns_mined"] > 0

    def test_get_pattern_info(self, fitted_clf):
        clf, _, _, _, _ = fitted_clf
        info = clf.get_pattern_info()
        assert isinstance(info, pd.DataFrame)
        assert len(info) == len(clf.patterns_)

    def test_get_hug_features(self, fitted_clf):
        clf, _, _, X_te, _ = fitted_clf
        hf = clf.get_hug_features()
        assert isinstance(hf, list)

    def test_detect_drift_runs(self, fitted_clf):
        clf, _, _, X_te, _ = fitted_clf
        if hasattr(clf, "_drift_det"):
            drift = clf.detect_drift(X_te)
            assert drift is not None


# ===================================================================
# (6) EDGE CASE TESTS
# ===================================================================

class TestEdgeCases:
    """Edge cases for fallback and interaction mining."""

    def test_single_sample_predict(self):
        """Fallback predict with a single row."""
        X, y = _make_noise_only_data(n=30)
        clf = HUGIMLClassifier(
            B=2, L=1, G=1e6,
            topK=5,
            feature_mode="patterns_only",
            adaptive_binning=False,
        )
        X_p, y_p = clf.prepareXy(X, y)
        clf.fit(X_p, y_p)
        assert clf.fallback_active_

        single = X_p.iloc[[0]]
        proba = clf.predict_proba(single)
        assert proba.shape == (1, len(clf.classes_))
        pred = clf.predict(single)
        assert pred.shape == (1,)

    def test_fallback_with_imbalanced_classes(self):
        """Fallback with heavily imbalanced target."""
        rng = np.random.RandomState(7)
        X = pd.DataFrame(rng.randn(100, 3), columns=["a", "b", "c"])
        y = pd.Series(np.concatenate([np.zeros(95), np.ones(5)]).astype(int))
        clf = HUGIMLClassifier(
            B=2, L=1, G=1e6,
            topK=5,
            feature_mode="patterns_only",
            adaptive_binning=False,
        )
        X_p, y_p = clf.prepareXy(X, y)
        clf.fit(X_p, y_p)
        assert clf.fallback_active_
        # Majority class should be 0
        assert clf.fallback_majority_class_ == 0
        proba = clf.predict_proba(X_p)
        # Prior for class 0 should be ~0.95
        assert proba[0, 0] > 0.9

    def test_fallback_enable_monitoring(self):
        """Monitoring should work with fallback models."""
        X, y = _make_noise_only_data(n=30)
        clf = HUGIMLClassifier(
            B=2, L=1, G=1e6,
            topK=5,
            feature_mode="patterns_only",
            adaptive_binning=False,
        )
        X_p, y_p = clf.prepareXy(X, y)
        clf.fit(X_p, y_p)
        clf.enable_monitoring()
        proba = clf.predict_proba(X_p)
        assert proba is not None
        report = clf.monitor.report()
        assert report is not None
        assert isinstance(report, (dict, str))

    def test_cached_fast_tune_with_zero_patterns(self):
        """fast_grid_tune path should not crash on zero-pattern candidates."""
        X, y = _make_noise_only_data(n=60)
        clf = HUGIMLClassifier(
            B=2, L=1, G=1e6,
            topK=5,
            feature_mode="patterns_only",
            adaptive_binning=True,
        )
        X_p, y_p = clf.prepareXy(X, y)
        # Should not raise
        clf.fit(X_p, y_p)
        proba = clf.predict_proba(X_p)
        assert proba.shape[0] == len(X_p)


# ===================================================================
# (7) INIT MODULE TESTS
# ===================================================================

class TestInitModule:

    def test_public_api_exports(self):
        assert hasattr(hugiml, "HUGIMLClassifier")
        assert hasattr(hugiml, "HUGIMLClassifierNative")
        assert hasattr(hugiml, "FitMetadata")

    def test_backward_compat_alias(self):
        assert HUGIMLClassifier is HUGIMLClassifierNative


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

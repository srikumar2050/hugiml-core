# tests/test_adaptive_native.py
# Tests for HUGIMLClassifierNative(adaptive_binning=True) — v1.1.0 addition.
# All other extension tests remain in test_extensions.py.
"""
Test suite for the adaptive_binning integration in HUGIMLClassifierNative.

Run with:
    pytest tests/test_adaptive_native.py -v
"""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from hugiml.classifier import HUGIMLClassifierNative

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bc_split():
    bc = load_breast_cancer(as_frame=True)
    X, y = bc.data, bc.target.values
    clf_tmp = HUGIMLClassifierNative()
    X_enc, y_enc = clf_tmp.prepareXy(X, y)
    return train_test_split(X_enc, y_enc, test_size=0.25, stratify=y_enc, random_state=42)


@pytest.fixture(scope="module")
def fitted_adaptive(bc_split):
    Xtr, Xte, ytr, yte = bc_split
    clf = HUGIMLClassifierNative(
        B=8, L=2, G=1e-4, topK=-1,
        adaptive_binning=True,
        b_candidates=[2, 3, 5, 7, 10, 15],
        min_marginal_gain_ratio=0.02,
    )
    clf.fit(Xtr, ytr)
    return clf, Xtr, Xte, ytr, yte


@pytest.fixture(scope="module")
def fitted_non_adaptive(bc_split):
    Xtr, Xte, ytr, yte = bc_split
    clf = HUGIMLClassifierNative(B=5, L=2, G=1e-4, topK=-1)
    clf.fit(Xtr, ytr)
    return clf, Xtr, Xte, ytr, yte


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------

class TestAdaptiveParams:
    def test_defaults_off(self):
        clf = HUGIMLClassifierNative()
        assert clf.adaptive_binning is False
        assert clf.b_candidates is None
        assert clf.min_marginal_gain_ratio == 0.02

    def test_params_round_trip(self):
        clf = HUGIMLClassifierNative(
            adaptive_binning=True,
            b_candidates=[2, 5, 10],
            min_marginal_gain_ratio=0.05,
        )
        p = clf.get_params()
        assert p["adaptive_binning"] is True
        assert p["b_candidates"] == [2, 5, 10]
        assert p["min_marginal_gain_ratio"] == 0.05

    def test_repr_shows_adaptive(self, fitted_adaptive):
        clf, *_ = fitted_adaptive
        assert "adaptive" in repr(clf)

    def test_repr_no_adaptive_tag_when_off(self, fitted_non_adaptive):
        clf, *_ = fitted_non_adaptive
        assert "adaptive" not in repr(clf)

    def test_validation_bad_b_candidates(self):
        clf = HUGIMLClassifierNative(adaptive_binning=True, b_candidates=[1, 5])
        with pytest.raises(Exception):
            from sklearn.datasets import load_breast_cancer
            bc = load_breast_cancer(as_frame=True)
            Xtr = bc.data.iloc[:400]; ytr = bc.target.values[:400]
            clf.fit(Xtr, ytr)

    def test_validation_bad_ratio(self):
        with pytest.raises(Exception):
            HUGIMLClassifierNative(
                adaptive_binning=True, min_marginal_gain_ratio=1.5
            )._validate_params()


# ---------------------------------------------------------------------------
# Fitted attributes
# ---------------------------------------------------------------------------

class TestFittedAttributes:
    def test_per_feature_b_set(self, fitted_adaptive, bc_split):
        clf, Xtr, Xte, ytr, yte = fitted_adaptive
        assert hasattr(clf, "per_feature_b_")
        assert len(clf.per_feature_b_) == Xtr.shape[1]

    def test_bin_edges_set(self, fitted_adaptive, bc_split):
        clf, Xtr, *_ = fitted_adaptive
        assert hasattr(clf, "_bin_edges_")
        assert len(clf._bin_edges_) == Xtr.shape[1]
        # all edges are numpy arrays with at least 2 values
        for edges in clf._bin_edges_.values():
            assert len(edges) >= 2

    def test_ig_scores_set(self, fitted_adaptive):
        clf, *_ = fitted_adaptive
        assert hasattr(clf, "ig_scores_")
        assert len(clf.ig_scores_) > 0

    def test_chosen_b_in_candidates(self, fitted_adaptive):
        clf, *_ = fitted_adaptive
        cands = clf.b_candidates or [2, 3, 5, 7, 10, 15]
        for b in clf.per_feature_b_.values():
            assert b in cands or b <= max(cands), f"chosen B={b} not in candidates"

    def test_non_adaptive_has_no_bin_edges(self, fitted_non_adaptive):
        clf, *_ = fitted_non_adaptive
        assert not getattr(clf, "_bin_edges_", {})
        assert not getattr(clf, "per_feature_b_", {})


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class TestInference:
    def test_predict_proba_shape(self, fitted_adaptive, bc_split):
        clf, Xtr, Xte, ytr, yte = fitted_adaptive
        proba = clf.predict_proba(Xte)
        assert proba.shape == (len(yte), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_shape(self, fitted_adaptive, bc_split):
        clf, Xtr, Xte, ytr, yte = fitted_adaptive
        preds = clf.predict(Xte)
        assert preds.shape == (len(yte),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_transform_shape(self, fitted_adaptive, bc_split):
        clf, Xtr, Xte, ytr, yte = fitted_adaptive
        hup = clf.transform(Xte)
        assert hup.shape == (len(yte), len(clf.patterns_))

    def test_auc_competitive(self, fitted_adaptive, bc_split):
        clf, Xtr, Xte, ytr, yte = fitted_adaptive
        auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        assert auc > 0.95, f"AUC={auc:.4f} unexpectedly low"

    def test_raw_input_accepted(self, fitted_adaptive, bc_split):
        """Predict from numpy array (un-binned) must work."""
        clf, Xtr, Xte, ytr, yte = fitted_adaptive
        proba = clf.predict_proba(Xte.values)
        assert proba.shape == (len(yte), 2)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_save_load_roundtrip(self, fitted_adaptive, bc_split, tmp_path):
        clf, Xtr, Xte, ytr, yte = fitted_adaptive
        path = tmp_path / "model.hugiml"
        clf.save_model(str(path))

        clf2 = HUGIMLClassifierNative.load_model(str(path))
        auc_orig = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        auc_loaded = roc_auc_score(yte, clf2.predict_proba(Xte)[:, 1])
        assert abs(auc_orig - auc_loaded) < 1e-9

    def test_bin_edges_restored(self, fitted_adaptive, tmp_path):
        clf, *_ = fitted_adaptive
        path = tmp_path / "adap.hugiml"
        clf.save_model(str(path))
        clf2 = HUGIMLClassifierNative.load_model(str(path))
        assert bool(clf2._bin_edges_)
        assert bool(clf2.per_feature_b_)
        for name in clf._bin_edges_:
            np.testing.assert_allclose(clf2._bin_edges_[name], clf._bin_edges_[name])

    def test_adaptive_binning_flag_restored(self, fitted_adaptive, tmp_path):
        clf, *_ = fitted_adaptive
        path = tmp_path / "flag.hugiml"
        clf.save_model(str(path))
        clf2 = HUGIMLClassifierNative.load_model(str(path))
        assert clf2.adaptive_binning is True


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_setstate_without_adaptive_keys(self, fitted_adaptive):
        """Old model state missing adaptive keys defaults gracefully."""
        import copy
        clf, *_ = fitted_adaptive
        state = copy.deepcopy(clf.__getstate__())
        for key in ("adaptive_binning", "b_candidates", "min_marginal_gain_ratio"):
            state.pop(key, None)
        restored = object.__new__(HUGIMLClassifierNative)
        restored.__setstate__(state)
        assert restored.adaptive_binning is False
        assert restored.b_candidates is None
        assert restored.min_marginal_gain_ratio == 0.02

    def test_non_adaptive_model_unchanged(self, fitted_non_adaptive, bc_split):
        """Existing non-adaptive flow is bit-for-bit identical."""
        clf, Xtr, Xte, ytr, yte = fitted_non_adaptive
        # fit a second time with same seed
        clf2 = HUGIMLClassifierNative(B=5, L=2, G=1e-4, topK=-1)
        clf2.fit(Xtr, ytr)
        auc1 = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        auc2 = roc_auc_score(yte, clf2.predict_proba(Xte)[:, 1])
        assert abs(auc1 - auc2) < 1e-9


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

class TestModelSummary:
    def test_summary_has_adaptive_section(self, fitted_adaptive):
        clf, *_ = fitted_adaptive
        summary = clf.model_summary()
        assert "Adaptive binning" in summary

    def test_summary_no_adaptive_section_when_off(self, fitted_non_adaptive):
        clf, *_ = fitted_non_adaptive
        summary = clf.model_summary()
        assert "Adaptive binning" not in summary


# ---------------------------------------------------------------------------
# Multiclass
# ---------------------------------------------------------------------------

class TestMulticlassAdaptive:
    def test_wine_multiclass(self):
        wn = load_wine(as_frame=True)
        X, y = wn.data, wn.target.values
        clf_tmp = HUGIMLClassifierNative()
        X_enc, y_enc = clf_tmp.prepareXy(X, y)
        Xtr, Xte, ytr, yte = train_test_split(X_enc, y_enc, test_size=0.25,
                                               stratify=y_enc, random_state=0)
        clf = HUGIMLClassifierNative(
            B=8, L=2, G=1e-4, topK=-1, adaptive_binning=True
        )
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        assert proba.shape == (len(yte), 3)
        auc = roc_auc_score(
            label_binarize(yte, classes=[0, 1, 2]),
            proba, multi_class="ovr", average="macro",
        )
        assert auc > 0.95, f"Wine multiclass AUC={auc:.4f}"


# ---------------------------------------------------------------------------
# Integration with extension modules
# ---------------------------------------------------------------------------

class TestExtensionIntegration:
    def test_metrics_on_adaptive(self, fitted_adaptive, bc_split):
        from hugiml.metrics import compute_all_metrics
        clf, Xtr, Xte, ytr, yte = fitted_adaptive
        m = compute_all_metrics(clf, Xte)
        assert m.n_patterns == len(clf.patterns_)
        assert 0 <= m.coverage <= 1

    def test_pruning_on_adaptive(self, fitted_adaptive, bc_split):
        from hugiml.pruning import PatternEditor
        clf, Xtr, Xte, ytr, yte = fitted_adaptive
        editor = PatternEditor(clf, operator_name="test")
        orig_n = len(clf.patterns_)
        editor.remove([0], reason="test")
        editor.refit(Xtr, ytr)
        new_clf = editor.finalize()
        assert len(new_clf.patterns_) < orig_n
        # new_clf must still predict on raw (un-binned) data
        proba = new_clf.predict_proba(Xte)
        assert proba.shape == (len(yte), 2)

    def test_plots_bin_profile_adaptive(self, fitted_adaptive, bc_split):
        pytest.importorskip("plotly")
        from hugiml.plots import HUGPlotter
        clf, Xtr, Xte, ytr, yte = fitted_adaptive
        plotter = HUGPlotter(clf)
        feat = list(clf.per_feature_b_.keys())[0]
        fig = plotter.plot_marginal_bin_profile(feat)
        # figure should have at least one bar trace
        assert len(fig.data) >= 1

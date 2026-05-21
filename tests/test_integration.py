# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end integration tests on real-world datasets.

German Credit dataset:
    1000 samples, 20 mixed features (categorical + numeric), binary target.

HELOC dataset:
    ~10 000 samples, 23 numeric features, binary target.

These tests exercise the full pipeline: prepareXy → fit → predict_proba →
explain → monitor → drift → serialise → reload.
"""

from __future__ import annotations

import numpy as np
import pytest

from hugiml import HUGIMLClassifierNative
from hugiml.calibration import evaluate_calibration
from hugiml.serialization import load_model, save_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roc_auc(y_true, proba_pos):
    """Simple AUC via trapezoidal rule (no sklearn needed at import time)."""
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y_true, proba_pos)


def _accuracy(y_true, y_pred):
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))


# ---------------------------------------------------------------------------
# German Credit — full pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGermanCreditPipeline:
    def test_fit_succeeds(self, german_split):
        X_tr, X_te, y_tr, y_te = german_split
        clf = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
        X_p, y_p = clf.prepareXy(
            X_tr.reset_index(drop=True),
            y_tr.reset_index(drop=True),
        )
        clf.fit(X_p, y_p)
        assert hasattr(clf, "classes_")

    def test_accuracy_above_baseline(self, fitted_clf_german):
        clf, X_te, y_te = fitted_clf_german
        preds = clf.predict(X_te)
        acc = _accuracy(y_te, preds)
        # German credit baseline accuracy (majority class) ~ 70%; we set a
        # conservative floor to guard against degenerate training
        assert acc > 0.55, f"Accuracy {acc:.3f} below floor"

    def test_roc_auc_reasonable(self, fitted_clf_german):
        clf, X_te, y_te = fitted_clf_german
        proba = clf.predict_proba(X_te)
        pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
        auc = _roc_auc(y_te, proba[:, pos_idx])
        assert auc > 0.55, f"AUC {auc:.3f} not above chance"

    def test_hug_features_nonempty(self, fitted_clf_german):
        clf, _, _ = fitted_clf_german
        feats = clf.get_hug_features()
        assert len(feats) > 0

    def test_pattern_info_nonempty(self, fitted_clf_german):
        clf, _, _ = fitted_clf_german
        df = clf.get_pattern_info()
        assert len(df) > 0

    def test_feature_importances_nonempty(self, fitted_clf_german):
        clf, _, _ = fitted_clf_german
        df = clf.feature_importances()
        assert len(df) > 0

    def test_model_summary(self, fitted_clf_german):
        clf, _, _ = fitted_clf_german
        summary = clf.model_summary()
        assert "HUG" in summary or "pattern" in summary.lower()

    def test_monitoring_pipeline(self, fitted_clf_german):
        clf, X_te, _ = fitted_clf_german
        clf.enable_monitoring(window_size=300)
        clf.predict_proba(X_te)
        report = clf.monitor.report()
        assert report is not None

    def test_drift_detection(self, fitted_clf_german, german_raw):
        clf, X_te, _ = fitted_clf_german
        report = clf.detect_drift(X_te)
        assert report is not None

    def test_calibration_on_german(self, fitted_clf_german):
        clf, X_te, y_te = fitted_clf_german
        proba = clf.predict_proba(X_te)
        pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
        result = evaluate_calibration(np.asarray(y_te), proba[:, pos_idx])
        assert 0.0 <= result.ece <= 1.0
        assert 0.0 <= result.brier_score <= 1.0

    def test_serialisation_roundtrip(self, fitted_clf_german, tmp_path):
        clf, X_te, _ = fitted_clf_german
        out = tmp_path / "german.hugiml"
        save_model(clf, out)
        clf2 = load_model(out)
        np.testing.assert_array_equal(clf.predict(X_te), clf2.predict(X_te))


# ---------------------------------------------------------------------------
# HELOC — full pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHELOCPipeline:
    @pytest.fixture(scope="class")
    def fitted_heloc(self, heloc_split):
        X_tr, X_te, y_tr, y_te = heloc_split
        clf = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
        X_p, y_p = clf.prepareXy(
            X_tr.reset_index(drop=True),
            y_tr.reset_index(drop=True),
        )
        clf.fit(X_p, y_p)
        return clf, X_te, y_te

    def test_fit_succeeds(self, heloc_split):
        X_tr, X_te, y_tr, y_te = heloc_split
        clf = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
        X_p, y_p = clf.prepareXy(
            X_tr.reset_index(drop=True),
            y_tr.reset_index(drop=True),
        )
        clf.fit(X_p, y_p)
        assert hasattr(clf, "classes_")

    def test_accuracy_above_baseline(self, fitted_heloc):
        clf, X_te, y_te = fitted_heloc
        preds = clf.predict(X_te)
        acc = _accuracy(y_te, preds)
        assert acc > 0.55, f"HELOC accuracy {acc:.3f} below floor"

    def test_roc_auc_heloc(self, fitted_heloc):
        clf, X_te, y_te = fitted_heloc
        proba = clf.predict_proba(X_te)
        pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
        auc = _roc_auc(y_te, proba[:, pos_idx])
        assert auc > 0.55, f"HELOC AUC {auc:.3f} not above chance"

    def test_patterns_nonempty(self, fitted_heloc):
        clf, _, _ = fitted_heloc
        assert len(clf.get_hug_features()) > 0

    def test_calibration_heloc(self, fitted_heloc):
        clf, X_te, y_te = fitted_heloc
        proba = clf.predict_proba(X_te)
        pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
        result = evaluate_calibration(np.asarray(y_te), proba[:, pos_idx])
        assert 0.0 <= result.brier_score <= 1.0


# ---------------------------------------------------------------------------
# Cross-dataset consistency
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCrossDatasetConsistency:
    def test_same_api_both_datasets(self, fitted_clf_german, fitted_heloc):
        """Both classifiers expose the same API contract."""
        for clf, X_te, y_te in [fitted_clf_german, fitted_heloc]:
            assert callable(clf.predict)
            assert callable(clf.predict_proba)
            assert callable(clf.transform)
            assert callable(clf.get_hug_features)
            assert callable(clf.get_pattern_info)
            assert callable(clf.feature_importances)
            assert callable(clf.model_summary)

    @pytest.fixture(scope="class")
    def fitted_heloc(self, heloc_split):
        X_tr, X_te, y_tr, y_te = heloc_split
        clf = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
        X_p, y_p = clf.prepareXy(
            X_tr.reset_index(drop=True),
            y_tr.reset_index(drop=True),
        )
        clf.fit(X_p, y_p)
        return clf, X_te, y_te

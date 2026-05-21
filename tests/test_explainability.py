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

"""Tests for explainability.py — HUGPatternExplainer, FeatureLineage,
ExplanationStabilityMetrics, ExplainabilityReport, SHAP bridge.
"""

from __future__ import annotations

import numpy as np
import pytest

from hugiml.explainability import (
    ExplainabilityReport,
    ExplanationStabilityMetrics,
    FeatureLineage,
    HUGPatternExplainer,
    aggregate_shap_to_features,
    shap_values_from_pattern_matrix,
)

# ---------------------------------------------------------------------------
# HUGPatternExplainer — instantiation
# ---------------------------------------------------------------------------


class TestHUGPatternExplainerInit:
    def test_instantiate_from_fitted_clf(self, fitted_clf_synthetic):
        clf, _, _ = fitted_clf_synthetic
        explainer = HUGPatternExplainer(clf)
        assert explainer is not None

    def test_raises_on_unfitted_clf(self, synthetic_Xy):
        from hugiml import HUGIMLClassifierNative

        clf_unfitted = HUGIMLClassifierNative(B=4)
        with pytest.raises(Exception):
            HUGPatternExplainer(clf_unfitted)


# ---------------------------------------------------------------------------
# feature_lineage()
# ---------------------------------------------------------------------------


class TestFeatureLineage:
    def test_feature_lineage_returns_list(self, fitted_clf_synthetic):
        clf, _, _ = fitted_clf_synthetic
        explainer = HUGPatternExplainer(clf)
        lineage = explainer.feature_lineage()
        assert isinstance(lineage, list)

    def test_feature_lineage_contains_FeatureLineage_objects(self, fitted_clf_synthetic):
        clf, _, _ = fitted_clf_synthetic
        explainer = HUGPatternExplainer(clf)
        lineage = explainer.feature_lineage()
        for item in lineage:
            assert isinstance(item, FeatureLineage)

    def test_feature_lineage_attributes(self, fitted_clf_synthetic):
        clf, _, _ = fitted_clf_synthetic
        explainer = HUGPatternExplainer(clf)
        lineage = explainer.feature_lineage()
        for fl in lineage:
            assert hasattr(fl, "feature_name")
            assert hasattr(fl, "feature_type")
            assert hasattr(fl, "derived_patterns")
            assert hasattr(fl, "pattern_indices")
            assert hasattr(fl, "total_importance")

    def test_feature_lineage_types_valid(self, fitted_clf_synthetic):
        clf, _, _ = fitted_clf_synthetic
        explainer = HUGPatternExplainer(clf)
        lineage = explainer.feature_lineage()
        valid_types = {"integer", "float", "categorical"}
        for fl in lineage:
            assert fl.feature_type in valid_types, f"Unexpected feature_type: {fl.feature_type!r}"

    def test_feature_lineage_importance_non_negative(self, fitted_clf_synthetic):
        clf, _, _ = fitted_clf_synthetic
        explainer = HUGPatternExplainer(clf)
        lineage = explainer.feature_lineage()
        for fl in lineage:
            assert fl.total_importance >= 0.0


# ---------------------------------------------------------------------------
# explanation_stability()
# ---------------------------------------------------------------------------


class TestExplanationStability:
    def test_stability_returns_metrics(self, synthetic_Xy):
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative

        X, y = synthetic_Xy
        clf = HUGIMLClassifierNative(B=4, L=1, G=1e-3)
        X_p, y_p = clf.prepareXy(X, y)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_p, y_p, test_size=0.2, random_state=0, stratify=y_p
        )
        clf.fit(X_tr, y_tr)
        explainer = HUGPatternExplainer(clf)

        # Split training data into two halves for stability measurement
        n = len(X_tr)
        X_a, y_a = X_tr[: n // 2], y_tr[: n // 2]
        X_b, y_b = X_tr[n // 2 :], y_tr[n // 2 :]
        metrics = explainer.explanation_stability(X_a, y_a, X_b, y_b)
        assert isinstance(metrics, ExplanationStabilityMetrics)

    def test_jaccard_in_unit_interval(self, synthetic_Xy):

        from hugiml import HUGIMLClassifierNative

        X, y = synthetic_Xy
        clf = HUGIMLClassifierNative(B=4, L=1, G=1e-3)
        X_p, y_p = clf.prepareXy(X, y)
        clf.fit(X_p, y_p)
        explainer = HUGPatternExplainer(clf)

        n = len(X_p)
        X_a, y_a = X_p[: n // 2], y_p[: n // 2]
        X_b, y_b = X_p[n // 2 :], y_p[n // 2 :]
        metrics = explainer.explanation_stability(X_a, y_a, X_b, y_b)
        assert 0.0 <= metrics.jaccard_similarity <= 1.0

    def test_same_data_high_stability(self, synthetic_Xy):
        """Same data for A and B should give Jaccard = 1.0."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative

        X, y = synthetic_Xy
        clf = HUGIMLClassifierNative(B=4, L=1, G=1e-3)
        X_p, y_p = clf.prepareXy(X, y)
        X_tr, _, y_tr, _ = train_test_split(X_p, y_p, test_size=0.2, random_state=0, stratify=y_p)
        clf.fit(X_tr, y_tr)
        explainer = HUGPatternExplainer(clf)
        metrics = explainer.explanation_stability(X_tr, y_tr, X_tr, y_tr)
        assert metrics.jaccard_similarity == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# generate_report()
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_generate_report_returns_report(self, fitted_clf_synthetic):
        clf, _, _ = fitted_clf_synthetic
        explainer = HUGPatternExplainer(clf)
        report = explainer.generate_report("test-model")
        assert isinstance(report, ExplainabilityReport)

    def test_report_has_n_patterns(self, fitted_clf_synthetic):
        clf, _, _ = fitted_clf_synthetic
        explainer = HUGPatternExplainer(clf)
        report = explainer.generate_report("test-model")
        assert report.n_patterns >= 0

    def test_report_to_json(self, fitted_clf_synthetic):
        import json

        clf, _, _ = fitted_clf_synthetic
        explainer = HUGPatternExplainer(clf)
        report = explainer.generate_report("test-model")
        j = report.to_json()
        parsed = json.loads(j)
        assert isinstance(parsed, dict)

    def test_report_save(self, fitted_clf_synthetic, tmp_path):
        import json

        clf, _, _ = fitted_clf_synthetic
        explainer = HUGPatternExplainer(clf)
        report = explainer.generate_report("test-model")
        out = tmp_path / "report.json"
        report.save(str(out))
        assert out.exists()
        data = json.loads(out.read_text())
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# SHAP bridge
# ---------------------------------------------------------------------------


class TestSHAPBridge:
    def test_shap_values_returns_array_or_none(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        sv = shap_values_from_pattern_matrix(clf, X_te)
        # Returns None when shap is not installed, or ndarray when it is
        assert sv is None or isinstance(sv, np.ndarray)

    def test_shap_values_finite_when_returned(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        sv = shap_values_from_pattern_matrix(clf, X_te)
        if sv is not None:
            assert np.all(np.isfinite(sv)), "SHAP values contain non-finite entries"

    def test_aggregate_shap_returns_dict_or_none(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        sv = shap_values_from_pattern_matrix(clf, X_te)
        if sv is None:
            pytest.skip("shap not installed")
        result = aggregate_shap_to_features(sv, clf)
        assert isinstance(result, dict)

    def test_aggregate_shap_feature_names_present(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        sv = shap_values_from_pattern_matrix(clf, X_te)
        if sv is None:
            pytest.skip("shap not installed")
        result = aggregate_shap_to_features(sv, clf)
        assert all(isinstance(k, str) for k in result.keys())

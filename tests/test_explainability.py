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
from sklearn.datasets import make_classification

from hugiml import HUGIMLClassifierNative
from hugiml.explainability import (
    ExplainabilityReport,
    ExplanationStabilityMetrics,
    FeatureLineage,
    HUGPatternExplainer,
    aggregate_shap_to_features,
    compute_shap_values,
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
        sv = compute_shap_values(clf, X_te)
        # Returns None when shap is not installed, or ndarray when it is
        assert sv is None or isinstance(sv, np.ndarray)

    def test_shap_values_finite_when_returned(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        sv = compute_shap_values(clf, X_te)
        if sv is not None:
            assert np.all(np.isfinite(sv)), "SHAP values contain non-finite entries"

    def test_aggregate_shap_returns_dict_or_none(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        sv = compute_shap_values(clf, X_te)
        if sv is None:
            pytest.skip("shap not installed")
        result = aggregate_shap_to_features(sv, clf)
        assert isinstance(result, dict)

    def test_aggregate_shap_feature_names_present(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        sv = compute_shap_values(clf, X_te)
        if sv is None:
            pytest.skip("shap not installed")
        result = aggregate_shap_to_features(sv, clf)
        assert all(isinstance(k, str) for k in result.keys())

    def test_shap_falls_back_to_kernel_explainer_for_rpte(self):
        """shap.LinearExplainer only works for a linear downstream
        estimator; RPTE is a boosted-tree ensemble, so this must fall
        through to the model-agnostic KernelExplainer branch (via
        classifier.model_.predict_proba, which works regardless of the
        downstream estimator) rather than raising or silently returning
        None for a reason unrelated to shap's availability.
        """
        from hugiml.hyperparameter_configs import get_hugiml_grid

        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("shap not installed")

        rng = np.random.RandomState(0)
        n = 300
        import pandas as pd

        X = pd.DataFrame({"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)})
        y = (X["a"] > 0.5).astype(int).to_numpy()

        rpte_estimator = get_hugiml_grid("performance_ho")["base_estimator"][1]
        clf = HUGIMLClassifierNative(
            L=1,
            topK=20,
            feature_mode="patterns_only",
            base_estimator=rpte_estimator,
        )
        clf.fit(X, y)

        sv = compute_shap_values(X=X.iloc[:40], classifier=clf, background_samples=20)
        assert sv is not None
        assert np.all(np.isfinite(sv))


def test_explanation_stability_reports_feature_type_specific_metrics():
    X, y = make_classification(n_samples=180, n_features=6, n_informative=4, random_state=42)
    clf = HUGIMLClassifierNative(
        B=-1,
        adaptive_binning=True,
        L=2,
        G=1e-3,
        topK=15,
        feature_mode="original_plus_patterns",
        augmented_pair_transforms=True,
        aug_feature_size=5,
        max_fit_seconds=5,
        interaction_relaxed_mining=False,
    )
    clf.fit(X[:90], y[:90])
    explainer = HUGPatternExplainer(clf)
    metrics = explainer.explanation_stability(X[:90], y[:90], X[90:], y[90:], top_n=10)
    assert isinstance(metrics.by_feature_type, dict)
    assert "pattern" in metrics.by_feature_type
    assert metrics.jaccard_similarity == metrics.by_feature_type["pattern"]["jaccard_similarity"]


def test_full_model_shap_supports_mixed_downstream_features():
    pytest.importorskip("shap")
    X, y = make_classification(n_samples=120, n_features=5, n_informative=3, random_state=7)
    clf = HUGIMLClassifierNative(
        B=-1,
        adaptive_binning=True,
        L=2,
        G=1e-3,
        topK=12,
        feature_mode="original_plus_patterns",
        augmented_pair_transforms=True,
        aug_feature_size=5,
        max_fit_seconds=5,
        interaction_relaxed_mining=False,
    ).fit(X, y)

    sv = compute_shap_values(clf, X[:5], background_samples=5)
    assert sv is not None
    assert sv.shape[0] == 5
    assert sv.shape[1] == len(clf.get_downstream_features())
    assert clf.transform(X[:5]).shape[1] == sv.shape[1]

    pattern_count = sum(str(name).startswith("pattern:") for name in clf.get_downstream_features())
    with pytest.warns(RuntimeWarning, match="Pattern-only SHAP reporting would omit"):
        assert shap_values_from_pattern_matrix(clf, X[:5]) is None
    sv_patterns = shap_values_from_pattern_matrix(
        clf, X[:5], background_samples=5, allow_incomplete=True
    )
    assert sv_patterns is not None
    assert sv_patterns.shape[1] == pattern_count

    with pytest.raises(ValueError, match="Pattern-only SHAP values omit"):
        aggregate_shap_to_features(sv_patterns, clf)
    aggregated = aggregate_shap_to_features(sv_patterns, clf, allow_incomplete=True)
    assert isinstance(aggregated, dict)


class _SHAPContractClassifier:
    def __init__(
        self,
        X_downstream,
        names,
        *,
        classes=(0, 1),
        feature_mode="original_plus_patterns",
    ):
        from types import SimpleNamespace

        self._X_downstream = np.asarray(X_downstream, dtype=float)
        self._names = list(names)
        self.feature_mode = feature_mode
        self.classes_ = np.asarray(classes)
        self.feature_names_in_ = ["age", "income"]
        self._original_cat_cols_ = []
        self.model_ = SimpleNamespace(
            named_steps={"clf": SimpleNamespace(n_features_in_=self._X_downstream.shape[1])},
            predict_proba=lambda Z: np.column_stack([np.full(len(Z), 0.4), np.full(len(Z), 0.6)]),
        )

    def transform(self, X):
        return self._X_downstream[: len(X)]

    def get_downstream_features(self):
        return list(self._names)

    def get_pattern_provenance(self):
        return {
            "pattern:age=high, income=high": {
                "raw_features": ["age", "income"],
            }
        }

    def get_augmented_pair_transforms(self):
        return [{"name": "age_times_income", "inputs": ["age", "income"]}]


def _install_fake_linear_shap(monkeypatch, calls, *, multiclass=False):
    import sys
    from types import SimpleNamespace

    class LinearExplainer:
        def __init__(self, model, background):
            calls["background_shape"] = tuple(background.shape)

        def shap_values(self, X):
            calls["explained_shape"] = tuple(X.shape)
            base = np.arange(X.shape[0] * X.shape[1], dtype=float).reshape(X.shape)
            if multiclass:
                return np.stack([base, base + 1.0, base + 2.0], axis=-1)
            return base

    fake_shap = SimpleNamespace(LinearExplainer=LinearExplainer)
    monkeypatch.setitem(sys.modules, "shap", fake_shap)


@pytest.mark.parametrize(
    "names",
    [
        ["pattern:age=high", "pattern:income=high"],
        ["pattern:age=high", "augmented_pair:age_times_income"],
        ["orig:age", "pattern:income=high"],
        ["orig:age", "pattern:age=high, income=high", "augmented_pair:age_times_income"],
    ],
)
def test_compute_shap_values_uses_complete_downstream_width(monkeypatch, names):
    X_downstream = np.arange(18, dtype=float).reshape(6, 3)[:, : len(names)]
    clf = _SHAPContractClassifier(X_downstream, names)
    calls = {}
    _install_fake_linear_shap(monkeypatch, calls)

    sv = compute_shap_values(clf, np.zeros((4, 2)))

    assert sv is not None
    assert sv.shape == (4, len(names))
    assert calls["background_shape"][1] == len(names)
    assert calls["explained_shape"][1] == len(names)


def test_pattern_scope_filters_after_full_model_explanation(monkeypatch):
    names = ["orig:age", "pattern:income=high", "augmented_pair:age_times_income"]
    clf = _SHAPContractClassifier(np.arange(18, dtype=float).reshape(6, 3), names)
    calls = {}
    _install_fake_linear_shap(monkeypatch, calls)

    with pytest.warns(RuntimeWarning, match="Pattern-only SHAP reporting would omit"):
        assert compute_shap_values(clf, np.zeros((4, 2)), feature_scope="patterns") is None

    sv = compute_shap_values(
        clf,
        np.zeros((4, 2)),
        feature_scope="patterns",
        allow_incomplete=True,
    )
    assert sv is not None
    assert sv.shape == (4, 1)
    assert calls["explained_shape"] == (4, 3)

    pattern_view = shap_values_from_pattern_matrix(
        clf,
        np.zeros((4, 2)),
        allow_incomplete=True,
    )
    assert pattern_view is not None
    np.testing.assert_allclose(pattern_view, sv)


@pytest.mark.parametrize(
    "names",
    [
        ["pattern:age=high", "pattern:income=high"],
        ["pattern:age=high", "augmented_pair:age_times_income"],
        ["augmented_pair:age_times_income"],
        [],
    ],
)
def test_pattern_matrix_wrapper_is_complete_for_patterns_only_models(monkeypatch, names):
    X_downstream = np.arange(24, dtype=float).reshape(6, 4)[:, : len(names)]
    clf = _SHAPContractClassifier(
        X_downstream,
        names,
        feature_mode="patterns_only",
    )
    calls = {}
    _install_fake_linear_shap(monkeypatch, calls)

    values = shap_values_from_pattern_matrix(clf, np.zeros((4, 2)))

    assert values is not None
    assert values.shape == (4, len(names))
    if names:
        assert calls["explained_shape"] == (4, len(names))
    else:
        assert calls == {}


@pytest.mark.parametrize(
    ("names", "expected_pattern_count"),
    [
        (["pattern:age=high", "augmented_pair:age_times_income"], 1),
        (["augmented_pair:age_times_income"], 0),
        ([], 0),
    ],
)
def test_patterns_only_wrapper_can_return_explicit_partial_view(
    monkeypatch, names, expected_pattern_count
):
    X_downstream = np.arange(24, dtype=float).reshape(6, 4)[:, : len(names)]
    clf = _SHAPContractClassifier(
        X_downstream,
        names,
        feature_mode="patterns_only",
    )
    calls = {}
    _install_fake_linear_shap(monkeypatch, calls)

    values = shap_values_from_pattern_matrix(
        clf,
        np.zeros((4, 2)),
        allow_incomplete=True,
    )

    assert values is not None
    assert values.shape == (4, expected_pattern_count)
    if names:
        assert calls["explained_shape"] == (4, len(names))
    else:
        assert calls == {}


def test_pattern_matrix_wrapper_preserves_multiclass_axis_for_patterns_only(monkeypatch):
    names = ["pattern:age=high", "augmented_pair:age_times_income"]
    clf = _SHAPContractClassifier(
        np.arange(12, dtype=float).reshape(6, 2),
        names,
        classes=(0, 1, 2),
        feature_mode="patterns_only",
    )
    calls = {}
    _install_fake_linear_shap(monkeypatch, calls, multiclass=True)

    values = shap_values_from_pattern_matrix(clf, np.zeros((4, 2)))

    assert values is not None
    assert values.shape == (4, 2, 3)


def test_compute_shap_values_preserves_multiclass_axis(monkeypatch):
    names = ["orig:age", "pattern:income=high"]
    clf = _SHAPContractClassifier(
        np.arange(12, dtype=float).reshape(6, 2), names, classes=(0, 1, 2)
    )
    calls = {}
    _install_fake_linear_shap(monkeypatch, calls, multiclass=True)

    sv = compute_shap_values(clf, np.zeros((4, 2)))

    assert sv is not None
    assert sv.shape == (4, 2, 3)


def test_aggregate_full_downstream_shap_uses_source_lineage():
    names = [
        "orig:age",
        "pattern:age=high, income=high",
        "augmented_pair:age_times_income",
    ]
    clf = _SHAPContractClassifier(np.ones((4, 3)), names)
    sv = np.ones((4, 3), dtype=float)

    aggregated = aggregate_shap_to_features(sv, clf)

    assert aggregated["age"] == pytest.approx(3.0)
    assert aggregated["income"] == pytest.approx(2.0)


def test_aggregate_pattern_subset_requires_explicit_incomplete_permission():
    names = ["orig:age", "pattern:age=high, income=high"]
    clf = _SHAPContractClassifier(np.ones((4, 2)), names)
    sv_patterns = np.ones((4, 1), dtype=float)

    with pytest.raises(ValueError, match="Pattern-only SHAP values omit"):
        aggregate_shap_to_features(sv_patterns, clf)

    aggregated = aggregate_shap_to_features(sv_patterns, clf, allow_incomplete=True)
    assert aggregated["age"] == pytest.approx(1.0)
    assert aggregated["income"] == pytest.approx(1.0)

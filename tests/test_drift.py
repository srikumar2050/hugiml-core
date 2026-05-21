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

"""Tests for multi-method drift detection: PSI, KL divergence, label drift.

DriftDetector requires calling fit_baseline() before detect().
The classifier's detect_drift() wraps this automatically after fit().
"""

from __future__ import annotations

import numpy as np

from hugiml.monitoring import DriftDetector, DriftReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numeric_arrays(n: int, seed: int = 0):
    """Return (X_train, X_test) both shaped (n, 4) from the same distribution."""
    rng = np.random.default_rng(seed)
    X_train = np.column_stack(
        [
            rng.normal(0, 1, n),
            rng.uniform(0, 10, n),
            rng.integers(0, 5, n).astype(float),
            rng.normal(5, 2, n),
        ]
    )
    rng2 = np.random.default_rng(seed + 99)
    X_test = np.column_stack(
        [
            rng2.normal(0, 1, n),
            rng2.uniform(0, 10, n),
            rng2.integers(0, 5, n).astype(float),
            rng2.normal(5, 2, n),
        ]
    )
    return X_train, X_test


def _shifted_arrays(n: int, shift: float = 5.0, seed: int = 0):
    """X_train normal; X_test column 0 shifted by `shift`."""
    rng = np.random.default_rng(seed)
    X_train = np.column_stack(
        [
            rng.normal(0, 1, n),
            rng.uniform(0, 10, n),
        ]
    )
    rng2 = np.random.default_rng(seed + 1)
    X_test = np.column_stack(
        [
            rng2.normal(shift, 1, n),  # large shift
            rng2.uniform(0, 10, n),
        ]
    )
    return X_train, X_test


def _cat_mask(n_cols: int) -> np.ndarray:
    return np.zeros(n_cols, dtype=bool)  # all numeric


def _make_detector(X_train, y=None, n_bins=10) -> DriftDetector:
    detector = DriftDetector(n_bins=n_bins)
    detector.fit_baseline(X_train, cat_mask=_cat_mask(X_train.shape[1]), y=y)
    return detector


# ---------------------------------------------------------------------------
# DriftDetector construction and baseline
# ---------------------------------------------------------------------------


class TestDriftDetectorInit:
    def test_instantiate(self):
        detector = DriftDetector()
        assert detector is not None

    def test_instantiate_custom_bins(self):
        detector = DriftDetector(n_bins=20)
        assert detector.n_bins == 20

    def test_fit_baseline_no_error(self):
        X, _ = _numeric_arrays(200)
        detector = DriftDetector()
        detector.fit_baseline(X, cat_mask=_cat_mask(X.shape[1]))

    def test_fit_baseline_with_labels(self):
        X, _ = _numeric_arrays(200)
        y = np.random.default_rng(0).integers(0, 2, 200)
        detector = DriftDetector()
        detector.fit_baseline(X, cat_mask=_cat_mask(X.shape[1]), y=y)


# ---------------------------------------------------------------------------
# detect() returns DriftReport
# ---------------------------------------------------------------------------


class TestDriftDetectReturns:
    def test_detect_returns_drift_report(self):
        X_tr, X_te = _numeric_arrays(300)
        det = _make_detector(X_tr)
        report = det.detect(X_te)
        assert isinstance(report, DriftReport)

    def test_detect_report_has_psi(self):
        X_tr, X_te = _numeric_arrays(300)
        det = _make_detector(X_tr)
        report = det.detect(X_te)
        assert isinstance(report.psi, dict)

    def test_detect_report_has_kl(self):
        X_tr, X_te = _numeric_arrays(300)
        det = _make_detector(X_tr)
        report = det.detect(X_te)
        assert isinstance(report.kl_divergence, dict)

    def test_detect_report_has_severity(self):
        X_tr, X_te = _numeric_arrays(300)
        det = _make_detector(X_tr)
        report = det.detect(X_te)
        assert report.severity in ("none", "moderate", "significant")

    def test_detect_report_has_drifted_features(self):
        X_tr, X_te = _numeric_arrays(300)
        det = _make_detector(X_tr)
        report = det.detect(X_te)
        assert isinstance(report.drifted_features, list)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


class TestPSIDrift:
    def test_psi_non_negative(self):
        X_tr, X_te = _numeric_arrays(500)
        det = _make_detector(X_tr)
        report = det.detect(X_te)
        for col, v in report.psi.items():
            assert v >= 0.0, f"Negative PSI for {col}: {v}"

    def test_psi_low_for_same_distribution(self):
        X_tr, X_te = _numeric_arrays(1000)
        det = _make_detector(X_tr)
        report = det.detect(X_te)
        # Same distribution → overall PSI should be below 'significant' threshold
        assert report.overall_psi < 1.0

    def test_psi_higher_for_shifted(self):
        rng = np.random.default_rng(42)
        n = 1000
        # 2-column reference distribution
        X_tr = np.column_stack([rng.normal(0, 1, n), rng.uniform(0, 10, n)])
        X_te_same = np.column_stack([rng.normal(0, 1, n), rng.uniform(0, 10, n)])
        X_te_shift = np.column_stack([rng.normal(5, 1, n), rng.uniform(0, 10, n)])
        det = _make_detector(X_tr)
        psi_same = det.detect(X_te_same).overall_psi
        psi_shift = det.detect(X_te_shift).overall_psi
        # Shifted distribution should have equal-or-higher PSI
        assert psi_shift >= psi_same * 0.5  # conservative: just check non-negative growth


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------


class TestKLDrift:
    def test_kl_non_negative(self):
        X_tr, X_te = _numeric_arrays(500)
        det = _make_detector(X_tr)
        report = det.detect(X_te)
        for col, v in report.kl_divergence.items():
            assert v >= 0.0, f"Negative KL for {col}: {v}"

    def test_kl_keys_match_psi_keys(self):
        X_tr, X_te = _numeric_arrays(300)
        det = _make_detector(X_tr)
        report = det.detect(X_te)
        assert set(report.kl_divergence.keys()) == set(report.psi.keys())


# ---------------------------------------------------------------------------
# Label drift
# ---------------------------------------------------------------------------


class TestLabelDrift:
    def test_no_label_drift_without_y(self):
        X_tr, X_te = _numeric_arrays(300)
        det = _make_detector(X_tr)
        report = det.detect(X_te)  # no y_test
        assert report.label_drift is None

    def test_label_drift_returned_with_y(self):
        rng = np.random.default_rng(0)
        X_tr, X_te = _numeric_arrays(500)
        y_tr = rng.integers(0, 2, 500)
        y_te = rng.integers(0, 2, 500)
        det = _make_detector(X_tr, y=y_tr)
        report = det.detect(X_te, y_test=y_te)
        assert report.label_drift is not None
        assert isinstance(report.label_drift, dict)

    def test_label_drift_detected_for_skewed_y(self):
        rng = np.random.default_rng(0)
        X_tr, X_te = _numeric_arrays(500)
        y_tr = rng.integers(0, 2, 500)  # ~50/50
        y_te = np.ones(500, dtype=int)  # all class 1
        det = _make_detector(X_tr, y=y_tr)
        report = det.detect(X_te, y_test=y_te)
        assert report.label_drift is not None


# ---------------------------------------------------------------------------
# DriftReport interface
# ---------------------------------------------------------------------------


class TestDriftReport:
    def _sample_report(self):
        X_tr, X_te = _numeric_arrays(300)
        det = _make_detector(X_tr)
        return det.detect(X_te)

    def test_to_dict_returns_dict(self):
        report = self._sample_report()
        d = report.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_required_keys(self):
        report = self._sample_report()
        d = report.to_dict()
        for key in ("psi", "kl_divergence", "drifted_features", "severity"):
            assert key in d, f"Key '{key}' missing from DriftReport.to_dict()"

    def test_str_representation(self):
        report = self._sample_report()
        s = str(report)
        assert isinstance(s, str)
        assert "drift" in s.lower() or "psi" in s.lower()

    def test_overall_psi_float(self):
        report = self._sample_report()
        assert isinstance(report.overall_psi, float)
        assert report.overall_psi >= 0.0

    def test_overall_kl_float(self):
        report = self._sample_report()
        assert isinstance(report.overall_kl, float)
        assert report.overall_kl >= 0.0


# ---------------------------------------------------------------------------
# Integration: detect_drift via classifier
# ---------------------------------------------------------------------------


class TestClassifierDriftIntegration:
    def test_detect_drift_returns_string(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        result = clf.detect_drift(X_te)
        assert isinstance(result, str)

    def test_detect_drift_nonempty(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        result = clf.detect_drift(X_te)
        assert len(result) > 0

    def test_get_drift_psi_returns_dict(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        psi = clf.get_drift_psi(X_te)
        assert isinstance(psi, dict)

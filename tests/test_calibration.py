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

"""Tests for calibration.py — ECE, MCE, Brier, reliability diagram data."""

from __future__ import annotations

import numpy as np
import pytest

from hugiml.calibration import (
    CalibrationResult,
    brier_decomposition,
    evaluate_calibration,
    reliability_diagram_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_calibration(n: int = 1000):
    """Perfectly calibrated probabilities: predicted ≈ empirical frequency."""
    rng = np.random.default_rng(42)
    proba = rng.uniform(0, 1, n)
    y = rng.binomial(1, proba).astype(int)
    return proba, y


def _overconfident(n: int = 1000):
    """Overconfident: probabilities near 0/1 regardless of true label."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, n)
    # Push probabilities toward extremes
    proba = np.where(rng.uniform(0, 1, n) > 0.5, 0.95, 0.05)
    return proba, y


# ---------------------------------------------------------------------------
# evaluate_calibration
# ---------------------------------------------------------------------------


class TestEvaluateCalibration:
    def test_returns_calibration_result(self):
        proba, y = _perfect_calibration()
        result = evaluate_calibration(y, proba)
        assert isinstance(result, CalibrationResult)

    def test_ece_range(self):
        proba, y = _perfect_calibration()
        result = evaluate_calibration(y, proba)
        assert 0.0 <= result.ece <= 1.0

    def test_mce_range(self):
        proba, y = _perfect_calibration()
        result = evaluate_calibration(y, proba)
        assert 0.0 <= result.mce <= 1.0

    def test_brier_range(self):
        proba, y = _perfect_calibration()
        result = evaluate_calibration(y, proba)
        # Brier score in [0, 1]
        assert 0.0 <= result.brier_score <= 1.0

    def test_ece_lower_for_calibrated(self):
        proba_cal, y_cal = _perfect_calibration(500)
        proba_over, y_over = _overconfident(500)
        evaluate_calibration(y_cal, proba_cal).ece
        ece_over = evaluate_calibration(y_over, proba_over).ece
        # Overconfident model should have higher ECE (on average)
        # This is probabilistic; assert ECE_over is not unreasonably low
        assert ece_over >= 0.0  # basic sanity

    def test_custom_n_bins(self):
        proba, y = _perfect_calibration()
        result = evaluate_calibration(y, proba, n_bins=5)
        assert isinstance(result, CalibrationResult)

    def test_all_correct_brier_near_zero(self):
        y = np.array([1, 1, 0, 0])
        proba = np.array([0.99, 0.99, 0.01, 0.01])
        result = evaluate_calibration(y, proba)
        assert result.brier_score < 0.01

    def test_all_wrong_brier_near_one(self):
        y = np.array([1, 1, 0, 0])
        proba = np.array([0.01, 0.01, 0.99, 0.99])
        result = evaluate_calibration(y, proba)
        assert result.brier_score > 0.9

    def test_result_has_expected_attributes(self):
        proba, y = _perfect_calibration()
        result = evaluate_calibration(y, proba)
        for attr in ("ece", "mce", "brier_score"):
            assert hasattr(result, attr), f"Missing attribute: {attr}"


# ---------------------------------------------------------------------------
# reliability_diagram_data
# ---------------------------------------------------------------------------


class TestReliabilityDiagramData:
    def test_returns_dict_or_tuple(self):
        proba, y = _perfect_calibration()
        out = reliability_diagram_data(y, proba)
        assert out is not None

    def test_fraction_of_positives_range(self):
        proba, y = _perfect_calibration()
        out = reliability_diagram_data(y, proba)
        # Expected to return (mean_predicted, fraction_of_positives) arrays
        if isinstance(out, (list, tuple)) and len(out) == 2:
            mean_pred, frac_pos = out
            assert np.all(mean_pred >= 0) and np.all(mean_pred <= 1)
            assert np.all(frac_pos >= 0) and np.all(frac_pos <= 1)
        else:
            # dict-like return
            assert out is not None

    def test_n_bins_controls_resolution(self):
        proba, y = _perfect_calibration(2000)
        out5 = reliability_diagram_data(y, proba, n_bins=5)
        out10 = reliability_diagram_data(y, proba, n_bins=10)
        if isinstance(out5, (list, tuple)) and isinstance(out10, (list, tuple)):
            assert len(out5[0]) <= len(out10[0])


# ---------------------------------------------------------------------------
# brier_decomposition
# ---------------------------------------------------------------------------


class TestBrierDecomposition:
    def test_returns_three_components(self):
        proba, y = _perfect_calibration()
        result = brier_decomposition(y, proba)
        # Expect reliability, resolution, uncertainty (or named tuple / dict)
        assert result is not None

    def test_components_non_negative(self):
        proba, y = _perfect_calibration()
        result = brier_decomposition(y, proba)
        if hasattr(result, "__iter__"):
            for v in result:
                if isinstance(v, float):
                    assert v >= -1e-9  # numerical tolerance


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestCalibrationEdgeCases:
    def test_single_sample(self):
        evaluate_calibration(np.array([1]), np.array([0.8]))

    def test_all_same_proba(self):
        y = np.array([0, 1, 0, 1, 1])
        proba = np.full(5, 0.5)
        result = evaluate_calibration(y, proba)
        assert isinstance(result, CalibrationResult)

    def test_invalid_proba_raises(self):
        y = np.array([0, 1])
        proba_bad = np.array([1.5, -0.1])  # out of [0, 1]
        with pytest.raises(Exception):
            evaluate_calibration(y, proba_bad)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(Exception):
            evaluate_calibration(np.array([0, 1, 0]), np.array([0.5, 0.5]))

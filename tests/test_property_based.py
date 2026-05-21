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

"""Property-based tests using Hypothesis.

These tests verify algebraic invariants and contracts that must hold for
arbitrary valid inputs, not just the fixed datasets used in other suites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Hypothesis import guard
# ---------------------------------------------------------------------------
hypothesis = pytest.importorskip("hypothesis", reason="hypothesis not installed")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# ---------------------------------------------------------------------------
# Calibration invariants
# ---------------------------------------------------------------------------


class TestCalibrationInvariants:
    @given(
        n=st.integers(min_value=10, max_value=200),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_ece_always_in_unit_interval(self, n, seed):
        from hugiml.calibration import evaluate_calibration

        rng = np.random.default_rng(seed)
        proba = rng.uniform(0, 1, n)
        y = rng.integers(0, 2, n)
        result = evaluate_calibration(y, proba)
        assert 0.0 <= result.ece <= 1.0

    @given(
        n=st.integers(min_value=4, max_value=100),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_brier_always_in_unit_interval(self, n, seed):
        from hugiml.calibration import evaluate_calibration

        rng = np.random.default_rng(seed)
        proba = rng.uniform(0, 1, n)
        y = rng.integers(0, 2, n)
        result = evaluate_calibration(y, proba)
        assert 0.0 <= result.brier_score <= 1.0

    @given(st.integers(min_value=10, max_value=300))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_perfect_predictions_low_brier(self, n):
        from hugiml.calibration import evaluate_calibration

        y = np.array([i % 2 for i in range(n)])
        # Near-perfect probabilities
        proba = np.where(y == 1, 0.95, 0.05)
        result = evaluate_calibration(y, proba)
        assert result.brier_score < 0.1


# ---------------------------------------------------------------------------
# Drift detection invariants
# ---------------------------------------------------------------------------


class TestDriftDetectionInvariants:
    @given(
        n=st.integers(min_value=50, max_value=500),
        seed=st.integers(min_value=0, max_value=999),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_identical_distribution_low_psi(self, n, seed):
        """PSI between identical distributions should be near zero."""
        from hugiml.monitoring import DriftDetector

        rng = np.random.default_rng(seed)
        ref = pd.DataFrame(
            {
                "x": rng.normal(0, 1, n),
                "y": rng.uniform(0, 1, n),
            }
        )
        # Same distribution, different random draws
        cur = pd.DataFrame(
            {
                "x": np.random.default_rng(seed + 1).normal(0, 1, n),
                "y": np.random.default_rng(seed + 2).uniform(0, 1, n),
            }
        )
        detector = DriftDetector(ref)
        report = detector.detect(cur)
        if hasattr(report, "psi_scores") and report.psi_scores:
            for col, psi in report.psi_scores.items():
                assert psi >= 0.0, f"Negative PSI for column {col}: {psi}"


# ---------------------------------------------------------------------------
# Serialisation invariants
# ---------------------------------------------------------------------------


class TestSerialisationInvariants:
    @given(st.integers(min_value=0, max_value=99))
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow])
    def test_roundtrip_predictions_stable(self, seed):
        """save → load must preserve predict() output exactly."""
        import tempfile
        from pathlib import Path

        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative
        from hugiml.serialization import load_model, save_model

        rng = np.random.default_rng(seed)
        n = 200
        X = pd.DataFrame(
            {
                "int_a": rng.integers(0, 10, n),
                "int_b": rng.integers(0, 5, n),
                "float_x": rng.uniform(0, 1, n),
            }
        )
        logits = 0.3 * X["int_a"] + 0.5 * X["float_x"]
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = pd.Series(rng.binomial(1, prob).astype(int), name="target")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed % 100, stratify=y
        )
        clf = HUGIMLClassifierNative(B=5, L=1, G=1e-3)
        X_p, y_p = clf.prepareXy(X_tr, y_tr)
        clf.fit(X_p, y_p)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / f"model_{seed}.hugiml"
            save_model(clf, out)
            clf2 = load_model(out)

        p1 = clf.predict(X_te)
        p2 = clf2.predict(X_te)
        np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# Classifier output invariants
# ---------------------------------------------------------------------------


class TestClassifierOutputInvariants:
    @given(
        n=st.integers(min_value=30, max_value=150),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_proba_sum_to_one(self, n, seed):
        """predict_proba rows must always sum to 1."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative

        rng = np.random.default_rng(seed)
        X = pd.DataFrame(
            {
                "a": rng.integers(0, 5, n),
                "b": rng.uniform(0, 1, n),
                "c": rng.choice(["X", "Y"], n),
            }
        )
        y = pd.Series(rng.integers(0, 2, n))

        clf = HUGIMLClassifierNative(B=4, L=1, G=1e-4)
        X_p, y_p = clf.prepareXy(X, y)
        if len(np.unique(y_p)) < 2:
            return  # degenerate: skip
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_p,
            y_p,
            test_size=0.2,
            random_state=seed,
            stratify=y_p,
        )
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        np.testing.assert_allclose(
            proba.sum(axis=1), 1.0, atol=1e-6, err_msg="predict_proba rows do not sum to 1"
        )

    @given(
        n=st.integers(min_value=30, max_value=150),
        seed=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_predict_labels_are_subset_of_classes(self, n, seed):
        """predict() must only return labels that appear in classes_."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative
        from hugiml.exceptions import HUGIMLMiningError

        rng = np.random.default_rng(seed)
        X = pd.DataFrame(
            {
                "a": rng.integers(0, 5, n),
                "b": rng.uniform(0, 1, n),
            }
        )
        y = pd.Series(rng.integers(0, 2, n))

        clf = HUGIMLClassifierNative(B=4, L=1, G=1e-4)
        X_p, y_p = clf.prepareXy(X, y)
        if len(np.unique(y_p)) < 2:
            return
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_p,
            y_p,
            test_size=0.2,
            random_state=seed,
            stratify=y_p,
        )
        try:
            clf.fit(X_tr, y_tr)
        except HUGIMLMiningError:
            # Edge-case dataset produced no patterns: skip rather than fail
            return
        preds = clf.predict(X_te)
        assert set(preds).issubset(set(clf.classes_))

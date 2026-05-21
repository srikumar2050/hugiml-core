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

"""Tests for telemetry.py — HUGIMLTracer, HUGIMLMetrics, instrument_classifier.

All telemetry is optional and degrades to no-ops when the backing libraries
(opentelemetry, prometheus_client) are not installed or when the corresponding
environment variable is set to 'false'.  These tests verify the no-op
behaviour and the instrumentation wrapper.
"""

from __future__ import annotations

import os

import numpy as np

from hugiml.telemetry import HUGIMLMetrics, HUGIMLTracer, instrument_classifier

# ---------------------------------------------------------------------------
# HUGIMLTracer (no-op path)
# ---------------------------------------------------------------------------


class TestHUGIMLTracer:
    def test_span_returns_context_manager(self):
        """span() must be usable as a context manager even without OTEL."""
        with HUGIMLTracer.span("test_span") as s:
            assert s is not None

    def test_span_with_attributes(self):
        attrs = {"model_id": "test", "n_samples": 100}
        with HUGIMLTracer.span("test_span", attrs) as s:
            s.set_attribute("extra", "value")  # must not raise

    def test_record_exception_noop(self):
        with HUGIMLTracer.span("test_span") as s:
            s.record_exception(ValueError("test"))  # must not raise

    def test_nested_spans(self):
        with HUGIMLTracer.span("outer"):
            with HUGIMLTracer.span("inner") as inner:
                inner.set_attribute("level", "inner")


# ---------------------------------------------------------------------------
# HUGIMLMetrics (no-op path)
# ---------------------------------------------------------------------------


class TestHUGIMLMetrics:
    def test_init_noop(self):
        """_init must not raise even without prometheus_client."""
        HUGIMLMetrics._init(model_id="test")

    def test_record_prediction_noop(self):
        HUGIMLMetrics._init(model_id="test")
        # Must not raise
        HUGIMLMetrics.record_prediction(
            model_id="test",
            n_samples=10,
            latency_s=0.005,
            mean_confidence=0.8,
        )

    def test_record_drift_noop(self):
        HUGIMLMetrics._init(model_id="test")
        HUGIMLMetrics.record_drift(
            model_id="test",
            psi_dict={"col_a": 0.05, "col_b": 0.15},
        )


# ---------------------------------------------------------------------------
# instrument_classifier
# ---------------------------------------------------------------------------


class TestInstrumentClassifier:
    def test_returns_same_instance(self, fitted_clf_synthetic):
        clf, _, _ = fitted_clf_synthetic
        result = instrument_classifier(clf, model_id="test")
        assert result is clf

    def test_predict_still_works(self, fitted_clf_synthetic):
        clf, X_te, y_te = fitted_clf_synthetic
        instrument_classifier(clf, model_id="test")
        preds = clf.predict(X_te)
        assert len(preds) == len(y_te)

    def test_predict_proba_still_works(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        instrument_classifier(clf, model_id="test")
        proba = clf.predict_proba(X_te)
        assert proba.shape[0] == len(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predictions_unchanged_after_instrumentation(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        proba_before = clf.predict_proba(X_te).copy()
        instrument_classifier(clf, model_id="test")
        proba_after = clf.predict_proba(X_te)
        np.testing.assert_allclose(proba_before, proba_after, atol=1e-10)

    def test_custom_model_id(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        instrument_classifier(clf, model_id="my-model-v2")
        # Should not raise
        clf.predict(X_te)


# ---------------------------------------------------------------------------
# Environment variable control
# ---------------------------------------------------------------------------


class TestTelemetryEnvControl:
    def test_otel_disabled_by_default(self):
        """HUGIML_OTEL_ENABLED defaults to false — spans are no-ops."""
        original = os.environ.pop("HUGIML_OTEL_ENABLED", None)
        try:
            with HUGIMLTracer.span("test") as s:
                # Should be a _NoopSpan
                assert s is not None
        finally:
            if original is not None:
                os.environ["HUGIML_OTEL_ENABLED"] = original

    def test_prometheus_disabled_by_default(self):
        """HUGIML_PROMETHEUS_ENABLED defaults to false — metrics are no-ops."""
        original = os.environ.pop("HUGIML_PROMETHEUS_ENABLED", None)
        try:
            HUGIMLMetrics._init(model_id="test")
            # Must not raise
            HUGIMLMetrics.record_prediction(
                model_id="test",
                n_samples=1,
                latency_s=0.001,
                mean_confidence=0.9,
            )
        finally:
            if original is not None:
                os.environ["HUGIML_PROMETHEUS_ENABLED"] = original

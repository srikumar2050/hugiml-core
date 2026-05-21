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

"""OpenTelemetry and Prometheus instrumentation for HUGIMLClassifierNative.

Both integrations are strictly optional: if the respective packages are not
installed the module degrades gracefully to no-op stubs.  Import and use of
this module never breaks the classifier itself.

OpenTelemetry
-------------
Wraps fit(), predict_proba(), and predict() with OTEL spans and attributes.
Set ``HUGIML_OTEL_ENABLED=1`` to activate.

Prometheus
----------
Exposes prediction count, latency histogram, and confidence gauge.
Set ``HUGIML_PROMETHEUS_ENABLED=1`` to activate.

Debug logging
-------------
All non-fatal telemetry and metrics failures are logged at DEBUG level
(``logger = logging.getLogger("hugiml.telemetry")``) with ``exc_info=True``
so that stack traces are available when the root logger is configured at
DEBUG without any user-visible noise at INFO or above.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

__all__ = [
    "HUGIMLTracer",
    "HUGIMLMetrics",
    "instrument_classifier",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
_OTEL_ENABLED = os.environ.get("HUGIML_OTEL_ENABLED", "0") == "1"
_PROM_ENABLED = os.environ.get("HUGIML_PROMETHEUS_ENABLED", "0") == "1"


# =============================================================================
# OpenTelemetry tracer
# =============================================================================


class _NoopSpan:
    """No-op span used when OpenTelemetry is not available."""

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def set_attribute(self, *_: object) -> None:
        pass

    def record_exception(self, *_: object) -> None:
        pass


class HUGIMLTracer:
    """OpenTelemetry tracer wrapper for HUGIMLClassifierNative.

    Emits spans for fit, predict_proba, and predict with attributes
    including n_samples, n_patterns, and latency.

    When ``opentelemetry-api`` is not installed all operations are no-ops.
    """

    _tracer: Any = None

    @classmethod
    def _get_tracer(cls) -> Any:
        if cls._tracer is not None:
            return cls._tracer
        if not _OTEL_ENABLED:
            return None
        try:
            from opentelemetry import trace

            cls._tracer = trace.get_tracer("hugiml.classifier")
        except ImportError:
            logger.debug(
                "opentelemetry-api is not installed; OTEL tracing disabled.",
                exc_info=True,
            )
        return cls._tracer

    @classmethod
    @contextmanager
    def span(cls, name: str, attributes: dict | None = None) -> Generator[Any, None, None]:
        """Context manager yielding an OTEL span (or no-op)."""
        tracer = cls._get_tracer()
        if tracer is None:
            yield _NoopSpan()
            return
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    try:
                        span.set_attribute(k, v)
                    except Exception:
                        logger.debug(
                            "span.set_attribute(%r, %r) failed on span '%s'.",
                            k,
                            v,
                            name,
                            exc_info=True,
                        )
            yield span


# =============================================================================
# Prometheus metrics
# =============================================================================


class _NoopCounter:
    def inc(self, *_: object) -> None:
        pass

    def labels(self, **_: object) -> _NoopCounter:
        return self


class _NoopHistogram:
    def observe(self, *_: object) -> None:
        pass

    def time(self) -> _NoopTimer:
        return _NoopTimer()

    def labels(self, **_: object) -> _NoopHistogram:
        return self


class _NoopGauge:
    def set(self, *_: object) -> None:
        pass

    def labels(self, **_: object) -> _NoopGauge:
        return self


class _NoopTimer:
    def __enter__(self) -> _NoopTimer:
        return self

    def __exit__(self, *_: object) -> None:
        pass


class HUGIMLMetrics:
    """Prometheus metrics for HUGIMLClassifierNative.

    Exposes:
      - ``hugiml_predictions_total`` counter
      - ``hugiml_prediction_latency_seconds`` histogram
      - ``hugiml_confidence_mean`` gauge
      - ``hugiml_drift_psi`` gauge (per-feature)

    When ``prometheus_client`` is not installed all metrics are no-ops.
    """

    _initialized: bool = False
    _predictions_total: Any = _NoopCounter()
    _prediction_latency: Any = _NoopHistogram()
    _confidence_mean: Any = _NoopGauge()
    _drift_psi: Any = _NoopGauge()

    @classmethod
    def _init(cls, model_id: str = "default") -> None:
        if cls._initialized:
            return
        if not _PROM_ENABLED:
            return
        try:
            from prometheus_client import Counter, Gauge, Histogram

            cls._predictions_total = Counter(
                "hugiml_predictions_total",
                "Total number of predictions made",
                ["model_id", "status"],
            )
            cls._prediction_latency = Histogram(
                "hugiml_prediction_latency_seconds",
                "Prediction latency in seconds",
                ["model_id"],
                buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
            )
            cls._confidence_mean = Gauge(
                "hugiml_confidence_mean",
                "Mean prediction confidence (rolling window)",
                ["model_id"],
            )
            cls._drift_psi = Gauge(
                "hugiml_drift_psi",
                "Population Stability Index per feature",
                ["model_id", "feature"],
            )
            cls._initialized = True
        except ImportError:
            logger.debug(
                "prometheus_client is not installed; Prometheus metrics disabled.",
                exc_info=True,
            )

    @classmethod
    def record_prediction(
        cls,
        model_id: str,
        n_samples: int,
        latency_s: float,
        mean_confidence: float,
        success: bool = True,
    ) -> None:
        """Record prediction metrics."""
        cls._init(model_id)
        status = "success" if success else "error"
        try:
            cls._predictions_total.labels(model_id=model_id, status=status).inc(n_samples)
            cls._prediction_latency.labels(model_id=model_id).observe(latency_s)
            cls._confidence_mean.labels(model_id=model_id).set(mean_confidence)
        except Exception:
            logger.debug(
                "Failed to record prediction metrics for model_id=%r (n_samples=%d).",
                model_id,
                n_samples,
                exc_info=True,
            )

    @classmethod
    def record_drift(cls, model_id: str, psi_dict: dict) -> None:
        """Update per-feature PSI gauges."""
        cls._init(model_id)
        for feature, psi_val in psi_dict.items():
            try:
                cls._drift_psi.labels(model_id=model_id, feature=feature).set(psi_val)
            except Exception:
                logger.debug(
                    "Failed to record drift PSI for model_id=%r, feature=%r.",
                    model_id,
                    feature,
                    exc_info=True,
                )


# =============================================================================
# Instrumentation helper
# =============================================================================


def instrument_classifier(
    classifier: Any,
    model_id: str = "default",
) -> Any:
    """Wrap a fitted classifier with telemetry instrumentation.

    Patches predict_proba and predict methods in-place to emit OTEL spans
    and Prometheus metrics.  The classifier itself is modified and returned.

    Parameters
    ----------
    classifier : HUGIMLClassifierNative
    model_id : str

    Returns
    -------
    The same classifier instance with patched methods.
    """
    original_predict_proba = classifier.predict_proba
    original_predict = classifier.predict

    def _instrumented_predict_proba(X_test: Any) -> Any:
        t0 = time.perf_counter()
        attrs = {
            "model_id": model_id,
            "n_samples": len(X_test) if hasattr(X_test, "__len__") else 0,
        }
        with HUGIMLTracer.span("hugiml.predict_proba", attrs) as span:
            try:
                proba = original_predict_proba(X_test)
                elapsed = time.perf_counter() - t0
                mean_conf = float(proba.max(axis=1).mean()) if len(proba) > 0 else 0.0
                HUGIMLMetrics.record_prediction(
                    model_id=model_id,
                    n_samples=len(proba),
                    latency_s=elapsed,
                    mean_confidence=mean_conf,
                    success=True,
                )
                try:
                    span.set_attribute("n_patterns", len(classifier.patterns_))
                    span.set_attribute("mean_confidence", mean_conf)
                    span.set_attribute("latency_ms", elapsed * 1000)
                except Exception:
                    logger.debug(
                        "Failed to set span attributes for predict_proba (model_id=%r).",
                        model_id,
                        exc_info=True,
                    )
                return proba
            except Exception as e:
                HUGIMLMetrics.record_prediction(
                    model_id=model_id,
                    n_samples=0,
                    latency_s=time.perf_counter() - t0,
                    mean_confidence=0.0,
                    success=False,
                )
                try:
                    span.record_exception(e)
                except Exception:
                    logger.debug(
                        "span.record_exception() failed for model_id=%r.",
                        model_id,
                        exc_info=True,
                    )
                raise

    def _instrumented_predict(X_test: Any) -> Any:
        with HUGIMLTracer.span("hugiml.predict", {"model_id": model_id}):
            return original_predict(X_test)

    classifier.predict_proba = _instrumented_predict_proba
    classifier.predict = _instrumented_predict
    return classifier

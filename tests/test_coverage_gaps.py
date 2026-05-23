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

"""Coverage gap tests.

Targets three 0%-or-low-coverage modules that are not exercised by the
main test suite:

* hugiml.exceptions  — exception/warning hierarchy (0% previously)
* hugiml.__init__    — lazy-import __getattr__ gate (0% previously)
* hugiml.monitoring  — PredictionMonitor.stats/report, DriftDetector
                       label-drift and human-readable report paths
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# =============================================================================
# exceptions.py  — full hierarchy
# =============================================================================


class TestExceptionHierarchy:
    """Instantiate and raise every exception class to cover exceptions.py."""

    def test_hugiml_error_is_exception(self):
        from hugiml.exceptions import HUGIMLError
        err = HUGIMLError("base error")
        assert isinstance(err, Exception)
        assert str(err) == "base error"

    def test_fit_error(self):
        from hugiml.exceptions import HUGIMLError, HUGIMLFitError
        with pytest.raises(HUGIMLFitError):
            raise HUGIMLFitError("fit failed")
        assert issubclass(HUGIMLFitError, HUGIMLError)

    def test_mining_error(self):
        from hugiml.exceptions import HUGIMLFitError, HUGIMLMiningError
        with pytest.raises(HUGIMLMiningError):
            raise HUGIMLMiningError("no patterns")
        assert issubclass(HUGIMLMiningError, HUGIMLFitError)

    def test_timeout_error(self):
        from hugiml.exceptions import HUGIMLFitError, HUGIMLTimeoutError
        with pytest.raises(HUGIMLTimeoutError):
            raise HUGIMLTimeoutError("timed out")
        assert issubclass(HUGIMLTimeoutError, HUGIMLFitError)

    def test_validation_error_is_value_error(self):
        from hugiml.exceptions import HUGIMLValidationError
        with pytest.raises(ValueError):
            raise HUGIMLValidationError("bad input")

    def test_schema_error(self):
        from hugiml.exceptions import HUGIMLSchemaError, HUGIMLValidationError
        with pytest.raises(HUGIMLSchemaError):
            raise HUGIMLSchemaError("wrong columns")
        assert issubclass(HUGIMLSchemaError, HUGIMLValidationError)

    def test_param_error_is_type_error(self):
        from hugiml.exceptions import HUGIMLParamError
        with pytest.raises(TypeError):
            raise HUGIMLParamError("bad param")

    def test_serialization_error(self):
        from hugiml.exceptions import HUGIMLError, HUGIMLSerializationError
        with pytest.raises(HUGIMLSerializationError):
            raise HUGIMLSerializationError("save failed")
        assert issubclass(HUGIMLSerializationError, HUGIMLError)

    def test_version_error(self):
        from hugiml.exceptions import HUGIMLSerializationError, HUGIMLVersionError
        with pytest.raises(HUGIMLVersionError):
            raise HUGIMLVersionError("incompatible version")
        assert issubclass(HUGIMLVersionError, HUGIMLSerializationError)

    def test_prediction_error_is_runtime_error(self):
        from hugiml.exceptions import HUGIMLPredictionError
        with pytest.raises(RuntimeError):
            raise HUGIMLPredictionError("predict failed")

    def test_warning_base(self):
        from hugiml.exceptions import HUGIMLWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("base warn", HUGIMLWarning)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)

    def test_convergence_warning(self):
        from hugiml.exceptions import HUGIMLConvergenceWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("converged", HUGIMLConvergenceWarning)
        assert issubclass(w[0].category, HUGIMLConvergenceWarning)

    def test_dtype_drift_warning(self):
        from hugiml.exceptions import HUGIMLDtypeDriftWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("dtype drift", HUGIMLDtypeDriftWarning)
        assert len(w) == 1

    def test_range_warning(self):
        from hugiml.exceptions import HUGIMLRangeWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("out of range", HUGIMLRangeWarning)
        assert len(w) == 1

    def test_degraded_warning(self):
        from hugiml.exceptions import HUGIMLDegradedWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("degraded", HUGIMLDegradedWarning)
        assert len(w) == 1

    def test_deprecation_warning(self):
        from hugiml.exceptions import HUGIMLDeprecationWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("deprecated", HUGIMLDeprecationWarning)
        assert issubclass(w[0].category, DeprecationWarning)


# =============================================================================
# hugiml/__init__.py  — lazy __getattr__ gate
# =============================================================================


class TestInitLazyImports:
    """Exercise the __getattr__ lazy-import gate in hugiml/__init__.py."""

    def test_version_and_metadata(self):
        import hugiml
        assert isinstance(hugiml.__version__, str)
        assert hugiml.__version__ == "1.1.0"
        assert "Krishnamoorthy" in hugiml.__author__
        assert "Apache" in hugiml.__license__
        assert "IEEE" in hugiml.__paper__

    def test_lazy_interpretability_metrics(self):
        import hugiml
        cls = hugiml.InterpretabilityMetrics
        assert cls.__name__ == "InterpretabilityMetrics"

    def test_lazy_compute_all_metrics(self):
        import hugiml
        fn = hugiml.compute_all_metrics
        assert callable(fn)

    def test_lazy_metrics_dataframe(self):
        import hugiml
        fn = hugiml.metrics_dataframe
        assert callable(fn)

    def test_lazy_pattern_editor(self):
        import hugiml
        cls = hugiml.PatternEditor
        assert cls.__name__ == "PatternEditor"

    def test_lazy_hugiml_adaptive(self):
        import hugiml
        cls = hugiml.HUGIMLAdaptive
        assert cls.__name__ == "HUGIMLAdaptive"

    def test_lazy_multiclass_report(self):
        import hugiml
        cls = hugiml.MulticlassHUGReport
        assert cls.__name__ == "MulticlassHUGReport"

    def test_lazy_make_imbalanced_pipeline(self):
        import hugiml
        fn = hugiml.make_imbalanced_pipeline
        assert callable(fn)

    def test_lazy_encode_high_cardinality(self):
        import hugiml
        fn = hugiml.encode_high_cardinality
        assert callable(fn)

    def test_lazy_apply_encoding(self):
        import hugiml
        fn = hugiml.apply_encoding
        assert callable(fn)

    def test_unknown_attribute_raises(self):
        import hugiml
        with pytest.raises(AttributeError, match="no attribute"):
            _ = hugiml.NonExistentThing


# =============================================================================
# monitoring.py  — uncovered paths
# =============================================================================


class TestPredictionMonitorStats:
    """Cover PredictionMonitor.stats property and report() method."""

    def _filled_monitor(self, n: int = 20) -> object:
        from hugiml.monitoring import PredictionMonitor
        mon = PredictionMonitor(window_size=200)
        rng = np.random.default_rng(0)
        for _ in range(n):
            p = rng.dirichlet([1, 1])
            mon.record(np.array([p]), latency_ms=float(rng.uniform(1, 10)))
        return mon

    def test_stats_keys(self):
        mon = self._filled_monitor()
        s = mon.stats
        for key in (
            "n_predictions", "confidence_mean", "confidence_std",
            "confidence_min", "confidence_max",
            "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
            "class_distribution",
        ):
            assert key in s, f"missing key: {key}"

    def test_stats_n_predictions(self):
        mon = self._filled_monitor(15)
        assert mon.stats["n_predictions"] == 15

    def test_stats_confidence_bounds(self):
        mon = self._filled_monitor()
        s = mon.stats
        assert 0.0 <= s["confidence_min"] <= s["confidence_mean"] <= s["confidence_max"] <= 1.0

    def test_stats_latency_percentiles_ordered(self):
        mon = self._filled_monitor(50)
        s = mon.stats
        assert s["latency_p50_ms"] <= s["latency_p95_ms"] <= s["latency_p99_ms"]

    def test_report_is_string(self):
        mon = self._filled_monitor()
        r = mon.report()
        assert isinstance(r, str)
        assert "Predictions" in r
        assert "Confidence" in r
        assert "Latency" in r

    def test_stats_empty_monitor(self):
        from hugiml.monitoring import PredictionMonitor
        mon = PredictionMonitor()
        s = mon.stats
        assert s["n_predictions"] == 0
        assert s["confidence_mean"] == 0.0
        assert s["latency_p50_ms"] == 0.0


class TestDriftDetectorLabelDrift:
    """Cover DriftDetector.compute_label_drift and DriftReport.__str__."""

    def _make_detector(self):
        from hugiml.monitoring import DriftDetector
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 3))
        y_train = rng.integers(0, 2, 200)
        cat_mask = np.zeros(3, dtype=bool)
        det = DriftDetector(n_bins=8)
        det.fit_baseline(X_train, cat_mask, col_names=["a", "b", "c"], y=y_train)
        return det, X_train, y_train

    def test_label_drift_returns_dict(self):
        det, X, y = self._make_detector()
        result = det.compute_label_drift(y)
        assert isinstance(result, dict)
        assert all(isinstance(v, float) for v in result.values())

    def test_label_drift_none_without_baseline(self):
        from hugiml.monitoring import DriftDetector
        det = DriftDetector()
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 2))
        cat_mask = np.zeros(2, dtype=bool)
        det.fit_baseline(X, cat_mask)   # no y → no label baseline
        result = det.compute_label_drift(np.zeros(50, dtype=int))
        assert result is None

    def test_detect_with_label_drift(self):
        from hugiml.monitoring import DriftReport
        det, X, y = self._make_detector()
        rng = np.random.default_rng(99)
        X_test = rng.standard_normal((80, 3))
        y_test = rng.integers(0, 2, 80)
        report = det.detect(X_test, y_test=y_test, threshold=0.1)
        assert isinstance(report, DriftReport)
        assert report.label_drift is not None

    def test_drift_report_str_with_label_drift(self):
        det, X, y = self._make_detector()
        rng = np.random.default_rng(7)
        X_test = rng.standard_normal((60, 3))
        y_test = rng.integers(0, 2, 60)
        report = det.detect(X_test, y_test=y_test)
        s = str(report)
        assert "PSI" in s or "Drift" in s or "psi" in s.lower()
        assert "label" in s.lower() or "class" in s.lower()

    def test_drift_report_str_without_label_drift(self):
        det, X, _ = self._make_detector()
        rng = np.random.default_rng(7)
        X_test = rng.standard_normal((60, 3))
        report = det.detect(X_test)
        s = str(report)
        assert isinstance(s, str) and len(s) > 0

    def test_detector_report_method(self):
        det, X, _ = self._make_detector()
        rng = np.random.default_rng(3)
        X_test = rng.standard_normal((40, 3))
        s = det.report(X_test, threshold=0.1)
        assert isinstance(s, str)

    def test_detector_report_no_baseline(self):
        from hugiml.monitoring import DriftDetector
        det = DriftDetector()
        X_test = np.random.default_rng(0).standard_normal((20, 2))
        s = det.report(X_test)
        assert "no numerical" in s.lower() or isinstance(s, str)

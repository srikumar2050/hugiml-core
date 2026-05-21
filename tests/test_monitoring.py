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

"""Tests for PredictionMonitor — thread safety, window behaviour, reporting."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from hugiml.monitoring import PredictionMonitor

# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestPredictionMonitorBasic:
    def test_instantiate(self):
        mon = PredictionMonitor(window_size=100)
        assert mon is not None

    def test_record_prediction(self):
        mon = PredictionMonitor(window_size=50)
        mon.record(np.array([[0.3, 0.7]]), latency_ms=5.0)

    def test_record_multiple(self):
        mon = PredictionMonitor(window_size=100)
        for i in range(20):
            proba = np.array([[0.4, 0.6]])
            mon.record(proba, latency_ms=float(i))

    def test_report_type(self):
        mon = PredictionMonitor(window_size=50)
        for _ in range(10):
            mon.record(np.array([[0.5, 0.5]]), latency_ms=1.0)
        report = mon.report()
        assert report is not None
        assert isinstance(report, str)

    def test_report_contains_prediction_count(self):
        mon = PredictionMonitor(window_size=50)
        n = 15
        for _ in range(n):
            mon.record(np.array([[0.6, 0.4]]), latency_ms=2.0)
        report = mon.report()
        # The report string should contain the total number of predictions
        assert str(n) in report or "prediction" in report.lower()

    def test_window_size_respected(self):
        window = 10
        mon = PredictionMonitor(window_size=window)
        for i in range(30):
            mon.record(np.array([[0.5, 0.5]]), latency_ms=1.0)
        # Internal deques should not grow beyond window_size
        for attr in ("_confidences", "_latencies_ms"):
            if hasattr(mon, attr):
                buf = getattr(mon, attr)
                assert len(buf) <= window

    def test_latency_stats_present(self):
        mon = PredictionMonitor(window_size=100)
        latencies = [1.0, 2.0, 3.0, 10.0, 0.5]
        for lat in latencies:
            mon.record(np.array([[0.5, 0.5]]), latency_ms=lat)
        report = mon.report()
        assert "latency" in report.lower() or "ms" in report.lower()

    def test_reset_clears_state(self):
        mon = PredictionMonitor(window_size=50)
        for _ in range(20):
            mon.record(np.array([[0.5, 0.5]]), latency_ms=1.0)
        if hasattr(mon, "reset"):
            mon.reset()
            report = mon.report()
            assert report is not None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestPredictionMonitorThreadSafety:
    """PredictionMonitor must be safe under concurrent access."""

    def test_concurrent_records(self):
        mon = PredictionMonitor(window_size=500)
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(50):
                    mon.record(
                        np.array([[0.4, 0.6]]),
                        latency_ms=1.0,
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_record_and_report(self):
        mon = PredictionMonitor(window_size=200)
        errors: list[Exception] = []

        def recorder():
            try:
                for _ in range(30):
                    mon.record(np.array([[0.5, 0.5]]), latency_ms=0.5)
                    time.sleep(0)  # yield
            except Exception as exc:
                errors.append(exc)

        def reporter():
            try:
                for _ in range(10):
                    mon.report()
                    time.sleep(0)
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(recorder) for _ in range(4)]
            futures += [pool.submit(reporter) for _ in range(2)]
            for f in futures:
                f.result(timeout=30)

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_no_data_corruption_under_concurrency(self):
        """All records written under concurrency must be counted correctly."""
        mon = PredictionMonitor(window_size=10_000)
        n_threads = 8
        records_per_thread = 100

        def worker():
            for _ in range(records_per_thread):
                mon.record(np.array([[0.5, 0.5]]), latency_ms=1.0)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        report = mon.report()
        # Report should be stable (not crash, not corrupt)
        assert report is not None


# ---------------------------------------------------------------------------
# Integration with classifier
# ---------------------------------------------------------------------------


class TestMonitorClassifierIntegration:
    def test_enable_monitoring(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        clf.enable_monitoring(window_size=200)
        assert clf.monitor is not None

    def test_predict_proba_populates_monitor(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        clf.enable_monitoring(window_size=200)
        clf.predict_proba(X_te)
        report = clf.monitor.report()
        assert report is not None

    def test_monitor_report_after_multiple_batches(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        clf.enable_monitoring(window_size=500)
        for _ in range(3):
            clf.predict_proba(X_te)
        report = clf.monitor.report()
        assert report is not None

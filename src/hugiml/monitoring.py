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

"""Operational monitoring for HUGIMLClassifierNative.

Provides thread-safe prediction statistics tracking and multi-method
distribution drift detection combining PSI, KL divergence, and label
drift monitoring.
"""

from __future__ import annotations

import logging
import threading
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


__all__ = [
    "PredictionMonitor",
    "DriftDetector",
    "DriftReport",
]


# =============================================================================
# Prediction monitoring
# =============================================================================


class PredictionMonitor:
    """Thread-safe prediction statistics tracker.

    Attach to a fitted classifier via ``clf.enable_monitoring()``.
    Access statistics via ``clf.monitor.report()`` or ``clf.monitor.stats``.

    Tracks prediction count, confidence distribution, per-class frequency,
    and latency percentiles over a rolling window.
    """

    def __init__(self, window_size: int = 1000) -> None:
        self._lock = threading.Lock()
        self.window_size = window_size
        self.reset()

    def __getstate__(self) -> dict:
        """Exclude threading.Lock from pickle payload."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore threading.Lock after unpickling."""
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        with self._lock:
            self.n_predictions: int = 0
            self.total_latency_s: float = 0.0
            self._confidences: deque = deque(maxlen=self.window_size)
            self._latencies_ms: deque = deque(maxlen=self.window_size)
            self._class_counts: dict[int, int] = {}

    def record(self, proba: np.ndarray, latency_ms: float) -> None:
        """Record one batch of predictions.

        Parameters
        ----------
        proba : np.ndarray, shape (n_samples, n_classes)
            Predicted class probabilities.
        latency_ms : float
            Wall-clock time for this batch in milliseconds.
        """
        max_conf = proba.max(axis=1)
        preds = proba.argmax(axis=1)

        with self._lock:
            self.n_predictions += len(proba)
            self.total_latency_s += latency_ms / 1000.0
            self._confidences.extend(max_conf.tolist())
            self._latencies_ms.append(latency_ms)
            for c in preds:
                key = int(c)
                self._class_counts[key] = self._class_counts.get(key, 0) + 1

    @property
    def stats(self) -> dict:
        """Current monitoring statistics as a plain dict."""
        with self._lock:
            confs = np.array(self._confidences) if self._confidences else np.array([])
            lats = np.array(self._latencies_ms) if self._latencies_ms else np.array([])
            return {
                "n_predictions": self.n_predictions,
                "confidence_mean": float(confs.mean()) if len(confs) else 0.0,
                "confidence_std": float(confs.std()) if len(confs) else 0.0,
                "confidence_min": float(confs.min()) if len(confs) else 0.0,
                "confidence_max": float(confs.max()) if len(confs) else 0.0,
                "latency_p50_ms": float(np.percentile(lats, 50)) if len(lats) else 0.0,
                "latency_p95_ms": float(np.percentile(lats, 95)) if len(lats) else 0.0,
                "latency_p99_ms": float(np.percentile(lats, 99)) if len(lats) else 0.0,
                "class_distribution": dict(self._class_counts),
            }

    def report(self) -> str:
        """Human-readable monitoring report."""
        s = self.stats
        return "\n".join(
            [
                "Prediction Monitor Report",
                f"  Predictions: {s['n_predictions']}",
                f"  Confidence:  mean={s['confidence_mean']:.4f}  "
                f"std={s['confidence_std']:.4f}  "
                f"range=[{s['confidence_min']:.4f}, {s['confidence_max']:.4f}]",
                f"  Latency:     p50={s['latency_p50_ms']:.1f}ms  "
                f"p95={s['latency_p95_ms']:.1f}ms  "
                f"p99={s['latency_p99_ms']:.1f}ms",
                f"  Classes:     {s['class_distribution']}",
            ]
        )


# =============================================================================
# Drift detection
# =============================================================================


class DriftReport:
    """Structured result from a drift detection run.

    Attributes
    ----------
    psi : dict[str, float]
        Population Stability Index per feature.
    kl_divergence : dict[str, float]
        Symmetric KL divergence per feature.
    label_drift : dict[str, float] or None
        Per-class label proportion shift (requires y_test).
    overall_psi : float
        Mean PSI across all numerical features.
    overall_kl : float
        Mean KL divergence across all numerical features.
    drifted_features : list[str]
        Features exceeding the PSI threshold.
    severity : str
        One of 'none', 'moderate', 'significant'.
    """

    def __init__(
        self,
        psi: dict,
        kl_divergence: dict,
        label_drift: dict | None,
        threshold: float,
    ) -> None:
        self.psi = psi
        self.kl_divergence = kl_divergence
        self.label_drift = label_drift

        vals = list(psi.values()) if psi else [0.0]
        self.overall_psi = float(np.mean(vals))
        kl_vals = list(kl_divergence.values()) if kl_divergence else [0.0]
        self.overall_kl = float(np.mean(kl_vals))

        self.drifted_features = [k for k, v in psi.items() if v > threshold]
        significant_threshold = max(0.25, threshold * 2.0)
        if self.overall_psi > significant_threshold:
            self.severity = "significant"
        elif self.overall_psi > threshold:
            self.severity = "moderate"
        else:
            self.severity = "none"

    def to_dict(self) -> dict:
        """Return all drift metrics as a plain dictionary."""
        return {
            "psi": self.psi,
            "kl_divergence": self.kl_divergence,
            "label_drift": self.label_drift,
            "overall_psi": self.overall_psi,
            "overall_kl": self.overall_kl,
            "drifted_features": self.drifted_features,
            "severity": self.severity,
        }

    def __str__(self) -> str:
        lines = [
            "Drift Detection Report",
            f"  Severity:    {self.severity.upper()}",
            f"  Overall PSI: {self.overall_psi:.4f}",
            f"  Overall KL:  {self.overall_kl:.4f}",
            f"  {'Feature':<30} {'PSI':>8} {'KL':>8} {'Status':>12}",
            f"  {'─' * 30} {'─' * 8} {'─' * 8} {'─' * 12}",
        ]
        all_features = sorted(set(list(self.psi.keys()) + list(self.kl_divergence.keys())))
        for feat in all_features:
            psi_val = self.psi.get(feat, 0.0)
            kl_val = self.kl_divergence.get(feat, 0.0)
            status = "⚠ DRIFT" if psi_val > 0.25 else "~ shift" if psi_val > 0.1 else "✓ stable"
            lines.append(f"  {feat:<30} {psi_val:>8.4f} {kl_val:>8.4f} {status:>12}")

        if self.label_drift:
            lines.append("")
            lines.append("  Label drift (class proportion shift):")
            for cls, delta in sorted(self.label_drift.items()):
                lines.append(f"    class {cls}: delta={delta:+.4f}")

        lines.append(
            f"\n  Drifted features ({len(self.drifted_features)}): "
            + (", ".join(self.drifted_features) or "none")
        )
        return "\n".join(lines)


class DriftDetector:
    """Multi-method distribution drift detector.

    Combines Population Stability Index (PSI) and symmetric KL divergence
    for robust drift assessment.  Optionally tracks label drift when ground
    truth is available.

    PSI thresholds:
      < 0.1  — stable
      0.1–0.25 — moderate shift
      > 0.25  — significant drift

    Parameters
    ----------
    n_bins : int
        Number of histogram bins for numerical features.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self._train_histograms: dict[int, tuple] = {}
        self._train_label_dist: np.ndarray | None = None
        self._col_names: list[str] = []

    def fit_baseline(
        self,
        X: np.ndarray,
        cat_mask: np.ndarray,
        col_names: list[str] | None = None,
        y: np.ndarray | None = None,
    ) -> None:
        """Store training distribution for later comparison.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
        cat_mask : np.ndarray of bool, shape (p,)
        col_names : list of str, optional
        y : np.ndarray of int, optional
            Training labels for label-drift baseline.
        """
        p = X.shape[1]
        self._col_names = col_names or [f"col{j}" for j in range(p)]
        self._train_histograms = {}
        for j in range(p):
            if j < len(cat_mask) and cat_mask[j]:
                continue
            col = X[:, j]
            finite = col[np.isfinite(col)]
            if len(finite) < 2:
                continue
            counts, edges = np.histogram(finite, bins=self.n_bins)
            self._train_histograms[j] = (counts.astype(float), edges)

        if y is not None:
            classes = np.unique(y)
            total = len(y)
            self._train_label_dist = np.array(
                [(y == c).sum() / total for c in classes], dtype=float
            )
            self._train_classes = classes

    def compute_psi(self, X_test: np.ndarray) -> dict[str, float]:
        """Compute PSI per numerical feature between training and test.

        Returns
        -------
        dict mapping column name to PSI value.
        """
        results: dict[str, float] = {}
        eps = 1e-6
        for j, (train_counts, edges) in self._train_histograms.items():
            name = self._col_names[j] if j < len(self._col_names) else f"col{j}"
            col = X_test[:, j]
            finite = col[np.isfinite(col)]
            if len(finite) == 0:
                results[name] = float("inf")
                continue
            test_counts = np.histogram(finite, bins=edges)[0].astype(float)
            train_p = train_counts / (train_counts.sum() + eps) + eps
            test_p = test_counts / (test_counts.sum() + eps) + eps
            psi = float(np.sum((test_p - train_p) * np.log(test_p / train_p)))
            results[name] = round(max(psi, 0.0), 6)
        return results

    def compute_kl(self, X_test: np.ndarray) -> dict[str, float]:
        """Compute symmetric KL divergence per feature.

        Returns
        -------
        dict mapping column name to KL value.
        """
        results: dict[str, float] = {}
        eps = 1e-6
        for j, (train_counts, edges) in self._train_histograms.items():
            name = self._col_names[j] if j < len(self._col_names) else f"col{j}"
            col = X_test[:, j]
            finite = col[np.isfinite(col)]
            if len(finite) == 0:
                results[name] = float("inf")
                continue
            test_counts = np.histogram(finite, bins=edges)[0].astype(float)
            p = train_counts / (train_counts.sum() + eps) + eps
            q = test_counts / (test_counts.sum() + eps) + eps
            kl_pq = float(np.sum(p * np.log(p / q)))
            kl_qp = float(np.sum(q * np.log(q / p)))
            results[name] = round((kl_pq + kl_qp) / 2.0, 6)
        return results

    def compute_label_drift(self, y_test: np.ndarray) -> dict[str, float] | None:
        """Compute per-class proportion shift between training and test labels.

        Returns None when no training label baseline is available.
        """
        if self._train_label_dist is None:
            return None
        classes = self._train_classes
        total = len(y_test)
        test_dist = np.array([(y_test == c).sum() / total for c in classes], dtype=float)
        return {
            str(int(c)): round(float(test_dist[i] - self._train_label_dist[i]), 6)
            for i, c in enumerate(classes)
        }

    def detect(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray | None = None,
        threshold: float = 0.1,
    ) -> DriftReport:
        """Run full multi-method drift detection.

        Parameters
        ----------
        X_test : np.ndarray
        y_test : np.ndarray of int, optional
        threshold : float
            PSI threshold above which a feature is flagged.

        Returns
        -------
        DriftReport
        """
        psi = self.compute_psi(X_test)
        kl = self.compute_kl(X_test)
        label_drift = self.compute_label_drift(y_test) if y_test is not None else None
        return DriftReport(psi, kl, label_drift, threshold)

    def report(self, X_test: np.ndarray, threshold: float = 0.1) -> str:
        """Return a human-readable drift report string (PSI only)."""
        if not self._train_histograms:
            return "Drift detection: no numerical features baselined."
        r = self.detect(X_test, threshold=threshold)
        return str(r)

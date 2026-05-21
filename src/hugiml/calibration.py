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

"""Calibration evaluation for HUGIMLClassifierNative.

Provides Expected Calibration Error (ECE), Brier score decomposition,
reliability diagram data, and calibration curve computation consistent
with best practices for interpretable classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CalibrationResult",
    "evaluate_calibration",
    "reliability_diagram_data",
    "brier_decomposition",
]


# =============================================================================
# Data containers
# =============================================================================


@dataclass
class CalibrationResult:
    """Calibration evaluation summary for a fitted classifier.

    Attributes
    ----------
    ece : float
        Expected Calibration Error (lower is better; 0 = perfect).
    mce : float
        Maximum Calibration Error across all bins.
    brier_score : float
        Mean Brier score (lower is better; 0 = perfect).
    brier_reliability : float
        Brier reliability component (miscalibration contribution).
    brier_resolution : float
        Brier resolution component (sharpness contribution).
    brier_uncertainty : float
        Brier uncertainty component (base rate uncertainty).
    n_bins : int
        Number of calibration bins used.
    bin_confidences : list of float
        Mean predicted confidence per bin.
    bin_accuracies : list of float
        Empirical accuracy per bin.
    bin_counts : list of int
        Sample count per bin.
    """

    ece: float
    mce: float
    brier_score: float
    brier_reliability: float
    brier_resolution: float
    brier_uncertainty: float
    n_bins: int
    bin_confidences: list[float] = field(default_factory=list)
    bin_accuracies: list[float] = field(default_factory=list)
    bin_counts: list[int] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable calibration summary."""
        lines = [
            "Calibration Summary",
            f"  ECE:              {self.ece:.4f}",
            f"  MCE:              {self.mce:.4f}",
            f"  Brier score:      {self.brier_score:.4f}",
            f"  Brier reliability:{self.brier_reliability:.4f}",
            f"  Brier resolution: {self.brier_resolution:.4f}",
            f"  Brier uncertainty:{self.brier_uncertainty:.4f}",
            "",
            f"  {'Bin':<6} {'Confidence':>12} {'Accuracy':>10} {'Count':>8}",
            f"  {'─' * 6} {'─' * 12} {'─' * 10} {'─' * 8}",
        ]
        for i, (conf, acc, cnt) in enumerate(
            zip(self.bin_confidences, self.bin_accuracies, self.bin_counts)
        ):
            lines.append(f"  {i:<6} {conf:>12.4f} {acc:>10.4f} {cnt:>8}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return metrics as a plain dictionary."""
        return {
            "ece": self.ece,
            "mce": self.mce,
            "brier_score": self.brier_score,
            "brier_reliability": self.brier_reliability,
            "brier_resolution": self.brier_resolution,
            "brier_uncertainty": self.brier_uncertainty,
            "n_bins": self.n_bins,
            "bin_confidences": self.bin_confidences,
            "bin_accuracies": self.bin_accuracies,
            "bin_counts": self.bin_counts,
        }


# =============================================================================
# Calibration computation
# =============================================================================


def evaluate_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> CalibrationResult:
    """Compute ECE, MCE, and Brier score decomposition.

    Parameters
    ----------
    y_true : np.ndarray of int, shape (n_samples,)
        True class labels (0 or 1 for binary; multi-class uses one-vs-rest).
    y_proba : np.ndarray of float, shape (n_samples,) or (n_samples, n_classes)
        Predicted probabilities.  For multi-class, pass the probability of the
        positive class or use the column for the class of interest.
    n_bins : int
        Number of calibration bins.
    strategy : {'uniform', 'quantile'}
        Bin strategy: uniform width or equal-frequency.

    Returns
    -------
    CalibrationResult
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}.")
    if strategy not in ("uniform", "quantile"):
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}.")

    # Validate probability range for 1-D arrays and the last axis of 2-D arrays
    _flat = y_proba.ravel()
    if np.any(np.isnan(_flat)) or np.any(_flat < 0.0) or np.any(_flat > 1.0):
        raise ValueError(
            "y_proba contains values outside [0, 1] or NaN.  "
            "Probabilities must be in the closed interval [0, 1]."
        )

    if y_proba.ndim == 2:
        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
        else:
            # Multi-class: use max probability for ECE
            y_conf = y_proba.max(axis=1)
            y_pred_cls = y_proba.argmax(axis=1)
            y_binary = (y_pred_cls == y_true).astype(float)
            return _compute_ece_multiclass(y_binary, y_conf, n_bins, strategy)

    return _compute_ece_binary(y_true, y_proba, n_bins, strategy)


def _compute_ece_binary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int,
    strategy: str,
) -> CalibrationResult:
    """ECE computation for binary predictions."""
    n = len(y_true)
    brier = float(np.mean((y_proba - y_true) ** 2))

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.percentile(y_proba, quantiles * 100)
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_confidences, bin_accuracies, bin_counts = [], [], []
    ece, mce = 0.0, 0.0

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)

        cnt = int(mask.sum())
        if cnt == 0:
            bin_confidences.append(float((lo + hi) / 2))
            bin_accuracies.append(0.0)
            bin_counts.append(0)
            continue

        conf = float(y_proba[mask].mean())
        acc = float(y_true[mask].mean())
        gap = abs(conf - acc)
        ece += cnt / n * gap
        mce = max(mce, gap)
        bin_confidences.append(round(conf, 6))
        bin_accuracies.append(round(acc, 6))
        bin_counts.append(cnt)

    rel, res, unc = brier_decomposition(y_true, y_proba)

    return CalibrationResult(
        ece=round(ece, 6),
        mce=round(mce, 6),
        brier_score=round(brier, 6),
        brier_reliability=round(rel, 6),
        brier_resolution=round(res, 6),
        brier_uncertainty=round(unc, 6),
        n_bins=n_bins,
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
    )


def _compute_ece_multiclass(
    y_correct: np.ndarray,
    y_conf: np.ndarray,
    n_bins: int,
    strategy: str,
) -> CalibrationResult:
    """ECE for multi-class using correctness vs. max confidence."""
    return _compute_ece_binary(y_correct, y_conf, n_bins, strategy)


def reliability_diagram_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> tuple[list[float], list[float], list[int]]:
    """Return bin-level data for plotting a reliability diagram.

    Parameters
    ----------
    y_true : np.ndarray
    y_proba : np.ndarray
    n_bins : int

    Returns
    -------
    (mean_predicted, fraction_positives, bin_counts)
        Three parallel lists, one entry per non-empty bin.
    """
    result = evaluate_calibration(y_true, y_proba, n_bins=n_bins)
    non_empty = [
        (c, a, n)
        for c, a, n in zip(result.bin_confidences, result.bin_accuracies, result.bin_counts)
        if n > 0
    ]
    if not non_empty:
        return [], [], []
    confs, accs, counts = zip(*non_empty)
    return list(confs), list(accs), list(counts)


def brier_decomposition(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float, float]:
    """Murphy decomposition of the Brier score.

    Decomposes Brier = Reliability - Resolution + Uncertainty.

    Parameters
    ----------
    y_true : np.ndarray of {0, 1}
    y_proba : np.ndarray of float in [0, 1]

    Returns
    -------
    (reliability, resolution, uncertainty)
        All three components as floats.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0

    base_rate = float(y_true.mean())
    uncertainty = base_rate * (1.0 - base_rate)

    n_bins = 10
    edges = np.linspace(0.0, 1.0 + 1e-10, n_bins + 1)
    bin_indices = np.digitize(y_proba, edges[1:])

    reliability, resolution = 0.0, 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        o_k = float(y_true[mask].mean())
        f_k = float(y_proba[mask].mean())
        reliability += cnt * (f_k - o_k) ** 2
        resolution += cnt * (o_k - base_rate) ** 2

    reliability /= n
    resolution /= n
    return round(reliability, 8), round(resolution, 8), round(uncertainty, 8)

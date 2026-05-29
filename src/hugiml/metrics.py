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

"""
Interpretability-complexity metrics for a fitted HUGIMLClassifierNative.

All functions accept a fitted ``HUGIMLClassifierNative`` and (optionally) a
data matrix ``X`` to compute sample-level statistics.  They never re-train
the model.

Quick reference
---------------
    from hugiml.metrics import compute_all_metrics
    m = compute_all_metrics(clf, X_test)
    print(m)

Available metrics
-----------------
n_patterns                  — total mined patterns
avg_pattern_length          — mean number of items per pattern
coverage                    — fraction of samples matched by ≥1 pattern
overlap_rate                — mean number of patterns active per sample
top_k_cumulative_contribution(k) — cumulative |coef| share of top-k patterns
active_patterns_per_prediction   — per-sample array
explanation_sparsity        — fraction of patterns *never* active (dead patterns)
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "InterpretabilityMetrics",
    "compute_all_metrics",
    "metrics_dataframe",
]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class InterpretabilityMetrics:
    """All interpretability metrics for one fitted model + dataset.

    Attributes
    ----------
    n_patterns : int
        Total number of mined HUG patterns.
    avg_pattern_length : float
        Mean items (conditions) per pattern.
    max_pattern_length : int
        Length of the longest pattern.
    coverage : float
        Fraction of samples covered by at least one active pattern.
    mean_active_patterns : float
        Average number of patterns active per sample.
    std_active_patterns : float
        Standard deviation of active patterns per sample.
    overlap_rate : float
        Alias for mean_active_patterns / n_patterns (normalised).
    explanation_sparsity : float
        Fraction of patterns that are never active on *X* ("dead" patterns).
    top_k_cumulative_contribution : dict[int, float]
        Mapping from k → cumulative share of total |coef| for the top-k patterns.
        Keys: [1, 5, 10, 20, 50].
    n_samples : int
        Number of rows in X used for sample-level metrics.
    """

    n_patterns: int = 0
    avg_pattern_length: float = 0.0
    max_pattern_length: int = 0
    coverage: float = 0.0
    mean_active_patterns: float = 0.0
    std_active_patterns: float = 0.0
    overlap_rate: float = 0.0
    explanation_sparsity: float = 0.0
    top_k_cumulative_contribution: dict = dataclasses.field(default_factory=dict)
    n_samples: int = 0

    def __str__(self) -> str:  # pragma: no cover
        lines = [
            "InterpretabilityMetrics",
            "=" * 42,
            f"  n_patterns              : {self.n_patterns}",
            f"  avg_pattern_length      : {self.avg_pattern_length:.2f}",
            f"  max_pattern_length      : {self.max_pattern_length}",
            f"  coverage                : {self.coverage:.4f}  ({self.coverage * 100:.1f}% of {self.n_samples} samples)",
            f"  mean_active_patterns    : {self.mean_active_patterns:.2f}",
            f"  std_active_patterns     : {self.std_active_patterns:.2f}",
            f"  overlap_rate (norm.)    : {self.overlap_rate:.4f}",
            f"  explanation_sparsity    : {self.explanation_sparsity:.4f}",
            "  top-k cumulative |coef|:",
        ]
        for k, v in sorted(self.top_k_cumulative_contribution.items()):
            lines.append(f"    top-{k:>3} : {v * 100:6.1f}%")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return a flat dict suitable for DataFrame construction."""
        d = dataclasses.asdict(self)
        topk = d.pop("top_k_cumulative_contribution", {})
        for k, v in topk.items():
            d[f"top_{k}_cumcontrib"] = v
        return d


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def n_patterns(clf: Any) -> int:
    """Return the total number of mined HUG patterns."""
    _require_fitted(clf)
    return len(clf.patterns_)


def avg_pattern_length(clf: Any) -> float:
    """Return the mean number of items (conditions) per pattern."""
    _require_fitted(clf)
    if not clf.patterns_:
        return 0.0
    return float(np.mean([len(pe.items) for pe in clf.patterns_]))


def max_pattern_length(clf: Any) -> int:
    """Return the length of the longest mined pattern."""
    _require_fitted(clf)
    if not clf.patterns_:
        return 0
    return int(max(len(pe.items) for pe in clf.patterns_))


def coverage(clf: Any, X: Any) -> float:
    """Fraction of samples in X matched by at least one active pattern.

    Parameters
    ----------
    clf : fitted HUGIMLClassifierNative
    X : array-like or DataFrame

    Returns
    -------
    float in [0, 1]
    """
    hup = _transform(clf, X)
    # row sums: how many patterns are active per sample
    row_sums = np.asarray(hup.sum(axis=1)).ravel()
    return float((row_sums > 0).mean())


def active_patterns_per_prediction(clf: Any, X: Any) -> np.ndarray:
    """Return a 1-D array with the number of active patterns for each sample.

    Parameters
    ----------
    clf : fitted HUGIMLClassifierNative
    X : array-like or DataFrame

    Returns
    -------
    np.ndarray of int, shape (n_samples,)
    """
    hup = _transform(clf, X)
    return np.asarray(hup.sum(axis=1)).ravel().astype(int)


def overlap_rate(clf: Any, X: Any) -> float:
    """Normalised overlap: mean active patterns / n_patterns.

    An overlap_rate of 0 means every sample activates exactly 0 patterns.
    A rate of 1 means every sample activates all patterns.
    """
    _require_fitted(clf)
    n_pats = len(clf.patterns_)
    if n_pats == 0:
        return 0.0
    ap = active_patterns_per_prediction(clf, X)
    return float(ap.mean()) / n_pats


def explanation_sparsity(clf: Any, X: Any) -> float:
    """Fraction of patterns that are never active on X ("dead" patterns).

    A sparsity of 1.0 means no pattern fires on any sample (degenerate model).
    A sparsity of 0.0 means every pattern fires on at least one sample.
    """
    hup = _transform(clf, X)
    # column sums: how many samples activate each pattern
    col_sums = np.asarray(hup.sum(axis=0)).ravel()
    n_pats = len(clf.patterns_)
    if n_pats == 0:
        return 1.0
    dead = int((col_sums == 0).sum())
    return float(dead) / n_pats


def top_k_cumulative_contribution(clf: Any, ks: list[int] | None = None) -> dict[int, float]:
    """Cumulative share of total |coef| held by the top-k patterns.

    Parameters
    ----------
    clf : fitted HUGIMLClassifierNative
    ks : list of int
        Values of k to evaluate.  Default: [1, 5, 10, 20, 50].

    Returns
    -------
    dict mapping k → cumulative fraction (float in [0, 1])
    """
    _require_fitted(clf)
    if ks is None:
        ks = [1, 5, 10, 20, 50]

    # Retrieve coefficients
    try:
        imp = clf.feature_importances()
        abs_coefs = imp["abs_coefficient"].values.astype(float)
    except Exception:
        # Fallback: use utility as proxy
        abs_coefs = np.array([pe.utility for pe in clf.patterns_], dtype=float)

    total = float(abs_coefs.sum())
    if total == 0:
        return {k: 0.0 for k in ks}

    sorted_coefs = np.sort(abs_coefs)[::-1]
    result: dict[int, float] = {}
    for k in ks:
        actual_k = min(k, len(sorted_coefs))
        result[k] = float(sorted_coefs[:actual_k].sum() / total)
    return result


# ---------------------------------------------------------------------------
# Composite helper
# ---------------------------------------------------------------------------


def compute_all_metrics(clf: Any, X: Any) -> InterpretabilityMetrics:
    """Compute all interpretability metrics in a single call.

    Parameters
    ----------
    clf : fitted HUGIMLClassifierNative
    X : array-like or DataFrame

    Returns
    -------
    InterpretabilityMetrics
    """
    _require_fitted(clf)
    hup = _transform(clf, X)
    n_samp = hup.shape[0]
    n_pats = len(clf.patterns_)

    lengths = [len(pe.items) for pe in clf.patterns_] if clf.patterns_ else [0]
    avg_len = float(np.mean(lengths))
    max_len = int(max(lengths))

    row_sums = np.asarray(hup.sum(axis=1)).ravel()
    col_sums = np.asarray(hup.sum(axis=0)).ravel()

    cov = float((row_sums > 0).mean())
    mean_ap = float(row_sums.mean())
    std_ap = float(row_sums.std())
    ol = float(mean_ap / n_pats) if n_pats > 0 else 0.0
    dead = int((col_sums == 0).sum())
    sparsity = float(dead / n_pats) if n_pats > 0 else 1.0
    topk = top_k_cumulative_contribution(clf)

    return InterpretabilityMetrics(
        n_patterns=n_pats,
        avg_pattern_length=round(avg_len, 3),
        max_pattern_length=max_len,
        coverage=round(cov, 4),
        mean_active_patterns=round(mean_ap, 3),
        std_active_patterns=round(std_ap, 3),
        overlap_rate=round(ol, 4),
        explanation_sparsity=round(sparsity, 4),
        top_k_cumulative_contribution=topk,
        n_samples=n_samp,
    )


def metrics_dataframe(results: dict[str, InterpretabilityMetrics]) -> pd.DataFrame:
    """Convert a mapping of {model_name: InterpretabilityMetrics} to a DataFrame.

    Useful for side-by-side comparisons across models or configurations.

    Parameters
    ----------
    results : dict
        Keys are model labels; values are InterpretabilityMetrics instances.

    Returns
    -------
    pd.DataFrame
    """
    rows = {}
    for name, m in results.items():
        rows[name] = m.to_dict()
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_fitted(clf: Any) -> None:
    if not hasattr(clf, "patterns_"):
        raise RuntimeError("Classifier must be fitted before computing interpretability metrics.")


def _transform(clf: Any, X: Any):
    """Return the sparse binary HUG pattern matrix for X."""
    _require_fitted(clf)
    return clf.transform(X)

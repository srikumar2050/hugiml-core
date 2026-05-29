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

"""Adaptive binning helpers — single source of truth for HUG-IML.

This private module holds every mathematical primitive used by the
adaptive-binning feature:

    _entropy                        Shannon entropy of a discrete label vector.
    _information_gain_from_sorted   IG(y ; bin(x)) from pre-sorted arrays (fast path).
    _information_gain               IG(y ; bin(x)) via equal-frequency binning.
    _select_b                       Elbow-stopping B selection (sort-once).
    _quantile_edges                 n_bins+1 unique quantile edges from finite values.
    _apply_edges                    Discretise a column and return string bin labels.

"""

from __future__ import annotations

import numpy as np

# Module is package-private; nothing is re-exported through the public API.
__all__: list[str] = []


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def _entropy(y: np.ndarray) -> float:
    """Shannon entropy of a discrete label vector *y*."""
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / n
    return float(-np.sum(p[p > 0] * np.log2(p[p > 0] + 1e-12)))


# ---------------------------------------------------------------------------
# Information gain
# ---------------------------------------------------------------------------


def _information_gain_from_sorted(
    x_sorted: np.ndarray,
    y_sorted: np.ndarray,
    n_bins: int,
) -> float:
    """IG(y ; bin(x)) using equal-frequency binning with *n_bins* bins.

    Operates on *pre-sorted* (x, y) pairs to avoid repeated sort/percentile
    work when evaluating multiple bin counts for the same column.

    Returns 0 when the column is constant or too coarse to split.
    """
    n = len(x_sorted)
    # Derive quantile edges directly from the sorted array — O(n_bins) slices
    # rather than a fresh np.percentile call per candidate bin count.
    indices = np.round(np.linspace(0, n - 1, n_bins + 1)).astype(int)
    edges = np.unique(x_sorted[indices])
    if len(edges) < 2:
        return 0.0
    # Digitise against the inner edges (exclude first and last boundary).
    binned = np.digitize(x_sorted, edges[1:-1])
    h_y = _entropy(y_sorted)
    weighted = sum((m := binned == b).sum() / n * _entropy(y_sorted[m]) for b in np.unique(binned))
    return float(max(0.0, h_y - weighted))


def _information_gain(x_col: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """IG(y ; bin(x)) using equal-frequency binning with *n_bins* bins.

    Public single-call entry point kept for backward compatibility and
    isolated use (e.g. diagnostics).  For scanning multiple bin counts on
    the same column, use :func:`_select_b` which sorts once and reuses.

    Returns 0 when the column is constant or too coarse to split.
    """
    order = np.argsort(x_col, kind="stable")
    return _information_gain_from_sorted(x_col[order], y[order], n_bins)


# ---------------------------------------------------------------------------
# Elbow-stopping B selection
# ---------------------------------------------------------------------------


def _select_b(
    x_col: np.ndarray,
    y: np.ndarray,
    candidates: list[int],
    min_marginal_gain_ratio: float,
) -> tuple[int, dict[int, float]]:
    """Return *(chosen_B, ig_scores)* using elbow-stopping.

    Iterates *candidates* in ascending order.  Stops when the incremental
    IG gain relative to the previous step falls below
    *min_marginal_gain_ratio*.  Pure argmax would otherwise always choose
    the largest candidate on continuously-valued features because IG is
    monotonically non-decreasing with bin count on training data
    (the elbow-stopping rule prevents that overfitting pattern).

    **Performance:** the column is sorted *once* here; every
    candidate bin count is then evaluated from the same sorted arrays via
    :func:`_information_gain_from_sorted`, replacing the previous pattern of
    calling :func:`_information_gain` (which called ``np.percentile``) once
    per candidate.  On a column with C candidates this reduces sort/quantile
    work from O(C · n log n) to O(n log n + C · n).

    Parameters
    ----------
    x_col:
        Finite numerical values of a single feature (training split).
    y:
        Integer class labels aligned with *x_col*.
    candidates:
        Sorted list of candidate bin counts (each >= 2).
    min_marginal_gain_ratio:
        Fractional threshold in (0, 1).  Stop when
        ``(ig_b - ig_prev) / (ig_prev + eps) < min_marginal_gain_ratio``.

    Returns
    -------
    chosen_B : int
        Selected bin count.
    ig_scores : dict[int, float]
        IG value for every candidate evaluated (for diagnostics).
    """
    if len(np.unique(x_col)) <= 1:
        return candidates[0], {b: 0.0 for b in candidates}

    # Sort once; all candidate evaluations reuse the same ordering.
    order = np.argsort(x_col, kind="stable")
    x_sorted = x_col[order]
    y_sorted = y[order]

    scores: dict[int, float] = {}
    prev_ig = 0.0
    chosen = candidates[0]

    for b in sorted(candidates):
        ig = _information_gain_from_sorted(x_sorted, y_sorted, b)
        scores[b] = ig
        if prev_ig > 0 and (ig - prev_ig) / (prev_ig + 1e-9) < min_marginal_gain_ratio:
            break
        chosen = b
        prev_ig = ig

    return chosen, scores


# ---------------------------------------------------------------------------
# Quantile edges
# ---------------------------------------------------------------------------


def _quantile_edges(x_col: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute *n_bins* + 1 unique quantile edges from finite values in *x_col*.

    Non-finite values are excluded before percentile computation so NaN/Inf
    do not perturb the bin boundaries.

    Returns a fallback ``[0.0, 1.0]`` array when no finite values exist.
    """
    finite = x_col[np.isfinite(x_col)]
    if len(finite) == 0:
        return np.array([0.0, 1.0])
    edges = np.unique(np.percentile(finite, np.linspace(0, 100, n_bins + 1)))
    return edges if len(edges) >= 2 else np.array([finite.min(), finite.max() + 1e-9])


# ---------------------------------------------------------------------------
# Apply edges → string bin labels
# ---------------------------------------------------------------------------


def _apply_edges(x_col: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Discretise *x_col* with pre-computed *edges* and return string bin labels.

    Label format: ``[lo,hi)`` — identical in appearance to the C++ native
    numeric format (e.g. ``duration=[12.0,24.0)``), so all downstream tools
    (pattern mining, explanations, serialisation) treat the output uniformly.

    Behaviour for special values:

    * Out-of-range finite values are clamped to the nearest bin (left or right
      outermost) via ``np.clip`` on the digitise index.
    * Non-finite values (NaN, ±Inf) receive ``np.nan`` in the output array.
      The C++ transaction builder skips cells whose value is ``nan``, so no
      item is generated for that (row, feature) pair — semantically
      "not observed".

    Performance note:
        The label universe is small (at most *n_bins* distinct strings).
        Pre-computing the full label array once and using numpy fancy-indexing
        replaces an O(n) Python-level loop with a single vectorised lookup,
        giving ~20–50x speedup on large columns.  Non-finite masking is then a
        single boolean assignment rather than a per-element branch.
    """
    n_bins = len(edges) - 1
    # Pre-compute the complete label array — at most B strings (typically 2–20).
    labels = np.empty(n_bins, dtype=object)
    for k in range(n_bins):
        labels[k] = f"[{edges[k]:.4g},{edges[k + 1]:.4g})"

    idx = np.clip(np.digitize(x_col, edges[1:-1]), 0, n_bins - 1)
    out: np.ndarray = labels[idx]  # vectorised fancy-index: O(n) C-speed
    out[~np.isfinite(x_col)] = np.nan  # single boolean-mask pass for NaN/Inf
    return out

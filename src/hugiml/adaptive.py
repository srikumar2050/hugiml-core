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
Per-feature adaptive binning for HUG-IML — ``HUGIMLAdaptive``.

``HUGIMLAdaptive`` is a thin, sklearn-compatible subclass of
``HUGIMLClassifierNative`` that hard-wires ``adaptive_binning=True`` and
exposes a simplified constructor (no ``B``, ``allCols``, or ``origColumns``
parameters — those are managed internally).

All adaptive-binning mathematics live in ``hugiml._binning`` (single source
of truth).  Both this module and ``hugiml.classifier`` import from there;
neither imports from the other at module level, so there is no circular
dependency.

Adaptive-binning algorithm (three steps)
-----------------------------------------
1. **Per-feature B selection** — for each numerical feature, evaluate
   candidate B values by computing information gain against ``y`` and stop
   when the marginal gain from adding more bins drops below
   ``min_marginal_gain_ratio`` of the gain already achieved (*elbow-stopping*).
2. **Pre-discretisation** — discretise each numerical feature to ``B_j``
   equal-frequency quantile bins, computed on the training split only.
   Bin boundaries are stored in ``_bin_edges_`` and reapplied at predict time.
   Each bin is encoded as a readable string label, e.g. ``"[12.0,24.0)"``.
3. **Categorical pass-through** — pre-binned columns are treated as
   categorical by the C++ layer; the global ``B`` parameter is set to the
   sentinel value ``2`` (no effect on already-categorical columns).

Non-finite value handling
--------------------------
Non-finite cells (NaN, ±Inf) in any pre-binned column receive ``np.nan``
in the label array.  The C++ transaction builder skips those cells,
generating no item for that (row, feature) pair — semantically
"not observed", with no imputation.

Usage
-----
    from hugiml.adaptive import HUGIMLAdaptive

    clf = HUGIMLAdaptive(b_candidates=[3, 5, 7, 10, 15],
                         min_marginal_gain_ratio=0.02,
                         L=2, G=1e-4)
    X_enc, y_enc = clf.prepareXy(X_df, y)
    X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y_enc, stratify=y_enc)
    clf.fit(X_tr, y_tr)

    print(clf.per_feature_b_)      # chosen B_j per feature
    print(clf.model_summary())
    clf.plot_bin_profiles()        # requires matplotlib
    clf.ig_heatmap()               # requires matplotlib

Diagnostic plots (``plot_bin_profiles``, ``ig_heatmap``) and fitted
attributes (``per_feature_b_``, ``ig_scores_``, ``_bin_edges_``) are
defined on ``HUGIMLClassifierNative`` and inherited here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hugiml.classifier import HUGIMLClassifierNative

# Re-export the binning helpers that were previously defined locally here.
# Tests do ``from hugiml.adaptive import _information_gain``; this import
# keeps that working while the canonical implementation lives in _binning.
from hugiml._binning import (  # noqa: F401  (public re-export)
    _information_gain,
    _select_b,
    _quantile_edges,
    _apply_edges,
)


__all__ = [
    "HUGIMLAdaptive",
    # binning helpers re-exported for backward compatibility
    "_information_gain",
    "_select_b",
    "_quantile_edges",
    "_apply_edges",
]


class HUGIMLAdaptive(HUGIMLClassifierNative):
    """HUG-IML with per-feature adaptive binning via elbow-stopping IG search.

    Thin subclass of ``HUGIMLClassifierNative`` with ``adaptive_binning=True``
    hard-wired and a simplified constructor that omits parameters which are
    managed internally (``B``, ``allCols``, ``origColumns``).

    All public methods, fitted attributes, serialisation, monitoring, drift
    detection, and explanation helpers are inherited from
    ``HUGIMLClassifierNative``.  No logic is duplicated.

    Parameters
    ----------
    b_candidates : list of int, optional
        Candidate bin counts to evaluate per feature.
        Default: ``[2, 3, 5, 7, 10, 15]``.
    min_marginal_gain_ratio : float, default 0.02
        Stop adding bins when the incremental IG gain relative to the
        current level falls below this fraction.  ``0.02`` means stop when
        a new candidate adds less than 2 % more IG than the previous step.
        Lower values allow finer bins; higher values enforce coarser bins.
    L : int, default 1
        Maximum HUG pattern length.  1 = singletons; 2 = pairs; -1 = unlimited.
    G : float, default 5e-3
        Minimum information-gain threshold.
    topK : int, default -1
        Maximum number of patterns to retain.  -1 computes automatically.
    n_jobs : int, default 1
        Number of OpenMP threads.  -1 uses all available cores.
    verbose : bool, default False
        Emit INFO-level log messages during fit.
    max_fit_seconds : float or None
        Wall-clock budget for the pattern-mining stage of fit().

    Attributes (after fit — inherited from HUGIMLClassifierNative)
    --------------------------------------------------------------
    per_feature_b_ : dict[str, int]
        Chosen bin count per numerical feature.
    ig_scores_ : dict[str, dict[int, float]]
        Full IG score grid ``{feature: {B: ig_value}}`` for diagnostics.
    _bin_edges_ : dict[str, np.ndarray]
        Quantile edges used during ``fit``, reapplied at predict time.
    patterns_ : list
        Mined HUG patterns.
    classes_ : ndarray
        Unique class labels.
    fit_metadata_ : FitMetadata
        Timings, memory, pattern count stats.
    """

    def __init__(
        self,
        b_candidates: list[int] | None = None,
        min_marginal_gain_ratio: float = 0.02,
        L: int = 1,
        G: float = 5e-3,
        topK: int = -1,
        n_jobs: int = 1,
        verbose: bool = False,
        max_fit_seconds: float | None = None,
    ) -> None:
        super().__init__(
            adaptive_binning=True,
            b_candidates=b_candidates,
            min_marginal_gain_ratio=float(min_marginal_gain_ratio),
            L=L,
            G=G,
            topK=topK,
            n_jobs=n_jobs,
            verbose=verbose,
            max_fit_seconds=max_fit_seconds,
        )

    # ── sklearn protocol ──────────────────────────────────────────────────────

    def get_params(self, deep: bool = True) -> dict:
        """Return the constructor parameters (sklearn protocol).

        Only the parameters that ``HUGIMLAdaptive.__init__`` accepts are
        returned, so ``sklearn.clone`` and cross-validation helpers
        reconstruct the correct subclass.
        """
        return dict(
            b_candidates=self.b_candidates,
            min_marginal_gain_ratio=self.min_marginal_gain_ratio,
            L=self.L,
            G=self.G,
            topK=self.topK,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            max_fit_seconds=self.max_fit_seconds,
        )

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X_train: Any, y_train: Any) -> "HUGIMLAdaptive":
        """Fit with per-feature adaptive binning.

        Delegates entirely to ``HUGIMLClassifierNative.fit`` with
        ``adaptive_binning=True``.  When ``X_train`` is a plain ndarray and
        ``prepareXy`` has been called previously, column names from
        ``feature_names_in_`` are applied so that feature-name-aware
        operations (adaptive binning, bin-edge lookup, schema validation) work
        correctly.

        Parameters
        ----------
        X_train : pd.DataFrame or ndarray
        y_train : array-like of int

        Returns
        -------
        self
        """
        # When the caller passes a raw ndarray (common after prepareXy +
        # train_test_split) and column names are already known, reconstruct
        # the named DataFrame so that adaptive binning and schema checks use
        # the correct feature names rather than integer column indices.
        if not isinstance(X_train, pd.DataFrame):
            feat_names = getattr(self, "feature_names_in_", None)
            if feat_names is not None:
                arr = np.asarray(X_train)
                cols = list(feat_names)[: arr.shape[1]]
                X_train = pd.DataFrame(arr, columns=cols)
        return super().fit(X_train, y_train)

    # ── Backward-compatibility shim ───────────────────────────────────────────

    @property
    def clf_(self) -> "HUGIMLAdaptive":
        """Backward-compatibility alias.

        Old code that accessed ``adaptive_clf.clf_`` to reach the inner
        ``HUGIMLClassifierNative`` now gets ``self``, because
        ``HUGIMLAdaptive`` *is* a ``HUGIMLClassifierNative``.
        All methods and fitted attributes are directly on ``self``.
        """
        return self

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if hasattr(self, "patterns_"):
            b_vals = list(getattr(self, "per_feature_b_", {}).values())
            rng = f"B_j∈[{min(b_vals)},{max(b_vals)}]" if b_vals else "fitted"
        else:
            rng = "not fitted"
        return (
            f"HUGIMLAdaptive("
            f"candidates={self.b_candidates or [2, 3, 5, 7, 10, 15]}, "
            f"threshold={self.min_marginal_gain_ratio:.0%}, "
            f"{rng})"
        )

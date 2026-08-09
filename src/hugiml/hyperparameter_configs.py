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

"""Centralized hyperparameter grid definitions.

The shared benchmark grids keep HUGIML and baseline configuration in one
module for classifier tuning, the benchmark runner, and dashboard Workbench
reuse, so recommended search spaces stay aligned across command-line,
Python, and UI entry points.

Four named HUGIML grids are provided:

``"performance"``
    LR-only first-pass grid. It uses adaptive binning, searches ``L`` and
    ``topK``, keeps ``feature_mode="original_plus_patterns"``, and evaluates
    ``G`` at ``0.01`` and ``0.001``. No longer the default (see
    ``"performance_ho"`` below); kept available by name for callers that
    want to restrict tuning to the built-in logistic-regression branch only.

``"interpretability"``
    Pattern-focused grid. It keeps ``feature_mode="patterns_only"``, enables
    interaction-relaxed mining, and disables augmented-pair transforms so the
    fitted representation remains a HUG pattern surface.

``"interpretability_ho"``
    Higher-order extension of ``"interpretability"``. It preserves the same
    pattern-only, interaction-relaxed representation and disables augmented
    pairs, while searching the built-in LR and sequential RPTE downstream
    branches. This is a controlled test of downstream RPTE effectiveness.

``"performance_ho"``
    **Default grid** (``DEFAULT_HUGIML_GRID_NAME``). Higher-order Hybrid
    grid. It searches the built-in automatic L2 lbfgs logistic-regression branch
    together with the adaptive RPTE downstream branch only
    at ``leaf_config="3xD"``. The automatic branch uses lbfgs directly for
    binary targets and through one-vs-rest for multiclass targets. RPTE uses
    L2 lbfgs for downstream leaf weighting.
    The mining dimensions search ``L`` in ``[1, 2]``,
    ``topK`` in ``[50, 100]``, and ``G`` in ``[0.01, 0.001]``. The RPTE
    estimators are wrapped in sklearn's ``OneVsRestClassifier``: binary
    problems still use a single binary RPTE fit, while a K-class problem
    fits K one-vs-rest adaptive-RPTE models. This produces 8 LR candidates
    plus 8 RPTE-OvR candidates, for 16 candidates in total.
    ``topk_budget_strict=False`` is fixed for every candidate, keeping the
    non-strict augmented-pair budget semantics.

    The named representation paths set binary-column handling explicitly:
    augmented-pair grids use ``convert_binary_to_categorical=False`` so
    numeric 0/1 columns remain eligible as pair-transform sources, while
    interaction-relaxed grids use ``True`` so those indicators enter the
    categorical item surface used by native pattern mining. Direct classifier
    construction remains user-controlled through the constructor parameter.

    ``base_estimator`` varying is supported by ``fast_grid_tune``'s cache:
    mining (native pattern discovery) is cached per ``(G, L, topK)`` exactly
    as for any other grid, and the downstream feature matrix built from it is
    additionally cached per ``(G, L, topK, feature_mode)`` and reused across
    every ``base_estimator`` candidate at that key -- only the final "fit an
    estimator on the already-built matrix" step repeats per candidate (cheap
    for the LR branch; the RPTE branch's own boosting loop is the one part
    of a candidate this grid cannot avoid recomputing). See
    ``classifier._hugiml_prepare_downstream_template_from_cached_base`` /
    ``classifier._hugiml_fit_downstream_estimator_from_template``.
    The underlying RPTE learner is binary, while the grid-level
    ``OneVsRestClassifier`` wrapper provides multiclass support.

``BASELINE_MODEL_GRIDS`` holds the standard non-HUGIML benchmark grids.
``BUDGETED_BASELINE_MODEL_GRIDS`` holds the corresponding 200-leaf ensemble
grids used by the dashboard benchmark. Models without a registered grid are
fitted once with their default estimator settings.
"""

from __future__ import annotations

import copy

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from ._compat import logistic_penalty_kwargs
from .rpte_bounded_lookahead_leafwise import (
    LeafWiseBoundedLookaheadRPTEFeatureLR,
)


def make_l1_logistic_base_estimator() -> LogisticRegression:
    """Create the L1 one-vs-rest logistic estimator used by named grids."""
    return LogisticRegression(
        solver="liblinear",
        C=0.5,
        random_state=0,
        max_iter=300,
        **logistic_penalty_kwargs("l1"),
    )

# ── HUGIML hyperparameter grids ──────────────────────────────────────────────

HUGIML_GRIDS: dict[str, dict[str, list]] = {
    "performance": {
        "B": [-1],
        "adaptive_binning": [True],
        "L": [1, 2],
        "topK": [50, 100],
        "feature_mode": ["original_plus_patterns"],
        "G": [0.01, 0.001],
        # Augmented-pair paths keep numeric 0/1 columns numeric so they remain
        # eligible as pair-transform sources.
        "convert_binary_to_categorical": [False],
        "base_estimator": [make_l1_logistic_base_estimator()],
    },
    "interpretability": {
        "B": [-1],
        "adaptive_binning": [True],
        "L": [1, 2],
        "topK": [50, 100],
        "feature_mode": ["patterns_only"],
        "G": [0.01, 0.001],
        "interaction_relaxed_mining": [True],
        # Interaction-relaxed pattern mining treats numeric 0/1 indicators as
        # categorical items, preserving the pattern surface used by this path.
        "convert_binary_to_categorical": [True],
        # For L>=2, interaction-relaxed mining and augmented-pair transforms
        # are distinct representation paths. The interpretability grid keeps
        # only the pattern-mining path active.
        "augmented_pair_transforms": [False],
        "base_estimator": [make_l1_logistic_base_estimator()],
    },
    "interpretability_ho": {
        "B": [-1],
        "adaptive_binning": [True],
        "L": [1, 2],
        "topK": [50, 100],
        "feature_mode": ["patterns_only"],
        "G": [0.01, 0.001],
        "interaction_relaxed_mining": [True],
        "augmented_pair_transforms": [False],
        # Preserve the interpretability grid's treatment of 0/1 columns as
        # categorical. The classifier-wide False default keeps binary numeric
        # columns eligible for augmented-pair grids, but would suppress this
        # pattern-only surface because augmented pairs are deliberately off.
        "convert_binary_to_categorical": [True],
        "topk_budget_strict": [False],
        # Higher-order downstream search over the same pattern-only surface
        # as the default interpretability grid. Lookahead is explicitly off:
        # mined two-feature patterns can also serve as lookahead roots, so
        # disabling augmented-pair transforms alone is insufficient to enforce
        # the sequential-RPTE path.
        "base_estimator": [
            make_l1_logistic_base_estimator(),
            OneVsRestClassifier(
                LeafWiseBoundedLookaheadRPTEFeatureLR(
                    leaf_config="3xD",
                    depth=4,
                    enable_lookahead=False,
                ),
                n_jobs=1,
            ),
        ],
    },
}

HUGIML_GRIDS["performance_ho"] = {
    "B": [-1],
    "adaptive_binning": [True],
    "L": [1, 2],
    "topK": [50, 100],
    "feature_mode": ["original_plus_patterns"],
    "G": [0.01, 0.001],
    "convert_binary_to_categorical": [False],
    "augmented_pair_transforms": [True],
    "topk_budget_strict": [False],
    "lr_solver": ["adaptive_l1"],
    "lr_C": [0.5],
    "base_estimator": [
        None,
        OneVsRestClassifier(
            LeafWiseBoundedLookaheadRPTEFeatureLR(
                leaf_config="3xD",
                depth=4,
                enable_lookahead="adaptive",
                lr_C=0.5,
                lr_penalty="l1",
            ),
            n_jobs=1,
        ),
    ],
}
HUGIML_GRIDS["interpretability_ho"] = {
    "B": [-1],
    "adaptive_binning": [True],
    "L": [1, 2],
    "topK": [50, 100],
    "feature_mode": ["patterns_only"],
    "G": [0.01, 0.001],
    "interaction_relaxed_mining": [True],
    "augmented_pair_transforms": [False],
    "convert_binary_to_categorical": [True],
    "topk_budget_strict": [False],
    "lr_solver": ["adaptive_l1"],
    "lr_C": [0.5],
    "base_estimator": [
        None,
        OneVsRestClassifier(
            LeafWiseBoundedLookaheadRPTEFeatureLR(
                leaf_config="3xD",
                depth=4,
                enable_lookahead=False,
                lr_C=0.5,
                lr_penalty="l1",
            ),
            n_jobs=1,
        ),
    ],
}

# As of the RPTE integration, "performance_ho" (the hybrid grid that also
# searches the adaptive-RPTE downstream branch) is the default grid used
# whenever a caller omits an explicit grid name -- see HUGIMLClassifierNative
# .DEFAULT_PARAM_GRID (classifier.py), HUGIMLActionOrchestrator._tune_model
# (llm/orchestrator.py), and the dashboard Workbench's grid selector
# (dashboard/workbench.py), all of which resolve through this constant
# rather than hardcoding a grid name, specifically so this one change
# updates every "run the base code with no grid specified" entry point at
# once. "performance" (LR-only), "interpretability" (pattern-only LR),
# and "interpretability_ho" (pattern-only LR versus sequential RPTE) remain
# available by name for callers that want a narrower representation or
# downstream search.
DEFAULT_HUGIML_GRID_NAME = "performance_ho"


def get_hugiml_grid(name: str | None = None) -> dict[str, list]:
    """Return a copy of the named HUGIML hyperparameter grid.

    Parameters
    ----------
    name : str or None, default None
        A key in ``HUGIML_GRIDS``. ``None`` resolves to
        ``DEFAULT_HUGIML_GRID_NAME``.

    Returns
    -------
    dict[str, list]
        A fresh copy of the grid so callers can narrow candidate values without
        mutating the shared definition.

    Raises
    ------
    KeyError
        If ``name`` does not match a known grid.
    """
    resolved = name or DEFAULT_HUGIML_GRID_NAME
    if resolved not in HUGIML_GRIDS:
        available = ", ".join(sorted(HUGIML_GRIDS))
        raise KeyError(f"Unknown HUGIML grid {resolved!r}. Available grids: {available}.")
    # Deep-copy because some grids contain sklearn estimator instances.
    # Returning the shared singleton would allow a caller or failed fit to
    # mutate the process-wide grid definition.
    return copy.deepcopy(HUGIML_GRIDS[resolved])


def list_hugiml_grids() -> list[str]:
    """Return the available HUGIML grid names for CLI and UI population."""
    return sorted(HUGIML_GRIDS)


# ── Baseline model hyperparameter grids (benchmark runner) ──────────────────

BASELINE_MODEL_GRIDS: dict[str, dict[str, list]] = {
    # 2 x 2 x 2 x 2 = 16 candidates.
    "RandomForest": {
        "n_estimators": [200, 400],
        "max_depth": [4, 8],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt", 0.5],
    },
    # 2 x 2 x 2 x 2 = 16 candidates.
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 4],
        "learning_rate": [0.03, 0.1],
        "min_child_weight": [1, 5],
    },
    # 2 x 2 x 2 x 2 = 16 candidates.
    "LightGBM": {
        "n_estimators": [100, 200],
        "learning_rate": [0.03, 0.1],
        "num_leaves": [15, 31],
        "min_child_samples": [10, 20],
    },
    "LogisticReg": {
        "lr__C": [0.1, 1.0, 10.0],
    },
    # 4 x 2 x 2 = 16 candidates.
    "LogisticRegression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "class_weight": [None, "balanced"],
    },
    # 2 x 2 x 2 x 1 = 8 candidates.
    "EBM": {
        "learning_rate": [0.01, 0.05],
        "max_bins": [32, 64],
        "interactions": [0, 5],
        "max_rounds": [500],
    },
    # 2 x 2 x 2 = 8 candidates.
    "RuleFit": {
        "n_estimators": [50, 100],
        "max_rules": [50, 100],
        "tree_size": [5, 10],
    },
    # GAM APIs vary across installations. They are intentionally left
    # unregistered so the benchmark runner fits them once with estimator
    # defaults.
}


# Every ensemble candidate below has a configured theoretical ceiling of
# 200 terminal leaves:
#   25 x 2 = 50, 25 x 4 = 100, 50 x 2 = 100, or 50 x 4 = 200.
BUDGETED_BASELINE_MODEL_GRIDS: dict[str, dict[str, list]] = {
    "XGBoost": {
        "n_estimators": [25, 50],
        "max_depth": [1, 2],
        "learning_rate": [0.03, 0.1],
        "min_child_weight": [1, 5],
    },
    "LightGBM": {
        "n_estimators": [25, 50],
        "num_leaves": [2, 4],
        "learning_rate": [0.03, 0.1],
        "min_child_samples": [10, 20],
    },
    "RandomForest": {
        "n_estimators": [25, 50],
        "max_leaf_nodes": [2, 4],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt", 0.5],
        "max_depth": [None],
    },
}


def _copy_grid(grid: dict[str, list] | None) -> dict[str, list] | None:
    return copy.deepcopy(grid) if grid is not None else None


def get_baseline_grid(model_name: str) -> dict[str, list] | None:
    """Return a copy of a standard baseline tuning grid, or ``None``."""
    return _copy_grid(BASELINE_MODEL_GRIDS.get(model_name))


def get_budgeted_baseline_grid(model_name: str) -> dict[str, list] | None:
    """Return a copy of a 200-leaf ensemble tuning grid, or ``None``."""
    return _copy_grid(BUDGETED_BASELINE_MODEL_GRIDS.get(model_name))

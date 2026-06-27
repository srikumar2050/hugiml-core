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

Relative to v1.1.14, the shared benchmark grids keep HUGIML and baseline
configuration in one module for classifier tuning, the benchmark runner, and
dashboard Workbench reuse. This keeps
the recommended search spaces aligned across command-line, Python, and UI entry
points.

Two named HUGIML grids are provided:

``"performance"``
    Default first-pass grid. It uses adaptive binning, searches ``L`` and
    ``topK``, keeps ``feature_mode="original_plus_patterns"``, and evaluates
    ``G`` at ``0.01`` and ``0.001``.

``"interpretability"``
    Pattern-focused grid. It keeps ``feature_mode="patterns_only"``, enables
    interaction-relaxed mining, and disables augmented-pair transforms so the
    fitted representation remains a HUG pattern surface.

``BASELINE_MODEL_GRIDS`` holds the non-HUGIML benchmark grids. Models without a
registered grid are fitted once with their default estimator settings.
"""

from __future__ import annotations

# ── HUGIML hyperparameter grids ──────────────────────────────────────────────

HUGIML_GRIDS: dict[str, dict[str, list]] = {
    "performance": {
        "B": [-1],
        "adaptive_binning": [True],
        "L": [1, 2],
        "topK": [50, 100],
        "feature_mode": ["original_plus_patterns"],
        "G": [0.01, 0.001],
    },
    "interpretability": {
        "B": [-1],
        "adaptive_binning": [True],
        "L": [1, 2],
        "topK": [50, 100],
        "feature_mode": ["patterns_only"],
        "G": [0.01, 0.001],
        "interaction_relaxed_mining": [True],
        # For L>=2, interaction-relaxed mining and augmented-pair transforms
        # are distinct representation paths. The interpretability grid keeps
        # only the pattern-mining path active.
        "augmented_pair_transforms": [False],
    },
}

DEFAULT_HUGIML_GRID_NAME = "performance"


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
    return {k: list(v) for k, v in HUGIML_GRIDS[resolved].items()}


def list_hugiml_grids() -> list[str]:
    """Return the available HUGIML grid names for CLI and UI population."""
    return sorted(HUGIML_GRIDS)


# ── Baseline model hyperparameter grids (benchmark runner) ──────────────────

BASELINE_MODEL_GRIDS: dict[str, dict[str, list]] = {
    "RandomForest": {
        "n_estimators": [200, 400],
        "max_depth": [4, 8, None],
        "min_samples_leaf": [1, 5],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 4],
        "learning_rate": [0.03, 0.1],
    },
    "LightGBM": {
        "n_estimators": [100, 200],
        "learning_rate": [0.03, 0.1],
        "num_leaves": [15, 31],
    },
    "LogisticReg": {
        "lr__C": [0.1, 1.0, 10.0],
    },
    "EBM": {
        "learning_rate": [0.01, 0.05],
        "max_bins": [32, 64],
        "interactions": [0, 5],
    },
    "RuleFit": {
        "n_estimators": [50, 100],
        "max_rules": [50, 100],
        "tree_size": [5, 10],
    },
    # GAM APIs vary across installations. They are intentionally left
    # unregistered so the benchmark runner fits them once with estimator
    # defaults.
}


def get_baseline_grid(model_name: str) -> dict[str, list] | None:
    """Return a copy of a baseline model's tuning grid, or ``None``.

    ``None`` means no grid is registered for ``model_name`` and the benchmark
    runner should fit that estimator once with default hyperparameters.
    """
    grid = BASELINE_MODEL_GRIDS.get(model_name)
    return {k: list(v) for k, v in grid.items()} if grid is not None else None

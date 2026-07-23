"""Experiment Workbench UI for HUGIML Studio.

The workbench is intentionally separated from Governance. It is the place to
configure HUGIML and optional comparison models, run evaluations, compare
metrics/plots, and promote a fitted HUGIML run into the governance workspace.
"""

from __future__ import annotations

import copy
import itertools
import time
import warnings
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import numpy as np

# Compatibility for older dependencies under NumPy 2.x.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from hugiml.dashboard.components.patterns import (
    _direct_family_counts,
    _direct_source_count,
    _get_rpte_feature_flow_audit,
    _get_rpte_final_term_rows,
    _get_rpte_rule_rows,
    _rpte_rules_to_frame,
    render_rpte_flat_tree_view,
)
from hugiml.dashboard.display import dataframe_for_display
from hugiml.dashboard.runner import _default_params, fit_hugiml_config, score_cases
from hugiml.hyperparameter_configs import (
    DEFAULT_HUGIML_GRID_NAME,
    get_hugiml_grid,
    list_hugiml_grids,
)
from hugiml.rpte_bounded_lookahead_leafwise import (
    LEAF_CONFIGS as RPTE_LEAF_CONFIGS,
)
from hugiml.rpte_bounded_lookahead_leafwise import (
    LeafWiseBoundedLookaheadRPTEFeatureLR,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    category: str
    description: str
    optional_dependency: str | None = None


MODEL_CATALOG: dict[str, list[ModelSpec]] = {
    "Baselines": [
        ModelSpec("Logistic Regression", "Baselines", "Linear baseline with calibrated probability output."),
        ModelSpec("Decision Tree", "Baselines", "Simple tree baseline with transparent splits."),
    ],
    "Ensembles": [
        ModelSpec("Random Forest", "Ensembles", "Bagged decision-tree ensemble."),
        ModelSpec("XGBoost", "Ensembles", "Gradient-boosted trees.", optional_dependency="xgboost"),
        ModelSpec("LightGBM", "Ensembles", "Fast gradient-boosted trees.", optional_dependency="lightgbm"),
    ],
    "Interpretable Models": [
        ModelSpec("HUGIML", "Interpretable Models", "Native high-utility pattern model."),
        ModelSpec("EBM", "Interpretable Models", "Explainable Boosting Machine.", optional_dependency="interpret"),
        ModelSpec(
            "RuleFit",
            "Interpretable Models",
            "Official imodels RuleFit when installed; otherwise a clearly labelled sklearn RuleFit-style fallback.",
        ),
    ],
}

PARAM_HINTS = {
    "B": "Number of bins. Use -1 for HUGIML adaptive/default behavior.",
    "L": "Maximum HUG pattern length / interaction depth.",
    "topK": "Maximum selected HUG patterns.",
    "G": "Minimum gain/utility threshold for pattern mining.",
    "feature_mode": "How original and mined features are combined.",
    "topk_budget_strict": "Whether strict top-K budget is applied.",
    "augmented_pair_transforms": "Enable generated pair features for L>=2 using original-feature pairs.",
    "augmented_pair_mode": "Source-column scoring for augmented-pair features.",
    "aug_feature_size": "Number of source columns considered for augmented-pair generation.",
    "max_pair_features": "Maximum generated pair features kept for downstream modeling.",
    "ii_partner_size": "Optional number of partner columns used by interaction-information scoring.",
    "interaction_relaxed_mining": (
        "Allow interaction-information survivor columns into the root or "
        "immediate first-child mining position without generated pair features; "
        "deeper positions receive no new relaxed admission."
    ),
    "interaction_relaxed_feature_size": "Number of source columns selected for interaction-relaxed mining.",
    "adaptive_binning_sample_frac": "Advanced only: False uses all rows for bin selection; a value in (0, 1] uses a deterministic stratified sample before full-data fitting.",
    "adaptive_binning_sample_random_state": "Random seed for deterministic adaptive-binning sample selection.",
}

AUGMENTED_PAIR_MODE_OPTIONS = ["interaction_information", "marginal_ig"]


def _hugiml_core_default_grid(grid_name: str | None = None) -> dict[str, list[Any]]:
    """Return a named HUGIML core tuning grid with dashboard safeguards.

    ``grid_name`` selects which grid in ``hugiml.hyperparameter_configs`` to
    use (``"performance"`` when omitted). The classifier's own
    ``default_param_grid`` is preferred so the grid a user sees mining run
    against is read from the same place the classifier itself would use if
    asked directly; the direct config-module lookup below is a fallback for
    the unlikely case that importing the classifier itself raises (its
    import pulls in numpy/pandas/scipy/sklearn, any of which failing would
    otherwise leave the Workbench with no tuning grid at all to show).
    """
    try:
        from hugiml import HUGIMLClassifierNative

        grid = copy.deepcopy(HUGIMLClassifierNative.default_param_grid(grid_name))
    except Exception:
        grid = get_hugiml_grid(grid_name)
    grid["adaptive_binning"] = [True]
    return grid


def _hugiml_core_grid_names() -> list[str]:
    """Return the selectable grid names for the Workbench tuning UI."""
    return list_hugiml_grids()




def _hugiml_valid_feature_modes() -> list[str]:
    """Return feature modes for the Advanced UI, preserving a stable Workbench order.

    Any single named tuning grid may intentionally restrict feature_mode to
    the one value it's built around, but Advanced should still expose every
    feature_mode that appears in any registered grid, plus
    original_plus_interactions for explicit user-driven tuning, since that
    mode isn't part of any named grid today.
    """
    fallback = ["original_plus_patterns", "patterns_only", "original_plus_interactions"]
    modes: list[str] = []
    for grid_name in _hugiml_core_grid_names():
        try:
            grid = _hugiml_core_default_grid(grid_name)
            modes.extend(grid.get("feature_mode") or [])
        except Exception:
            continue
    valid = []
    for mode in [*fallback, *modes]:
        if mode and mode not in valid:
            valid.append(mode)
    return valid or fallback

def _hugiml_auto_params() -> dict[str, Any]:
    """Best single-run Workbench default aligned with the core fast-tune grid."""
    params = _default_params()
    params.update({
        "adaptive_binning": True,
        "B": -1,
        "L": 2,
        "topK": 100,
        "G": 0.01,
        "feature_mode": "original_plus_patterns",
        "topk_budget_strict": False,
        "augmented_pair_transforms": True,
        "augmented_pair_mode": "interaction_information",
        "aug_feature_size": 10,
        "max_pair_features": 10,
        "ii_partner_size": None,
        "interaction_relaxed_mining": False,
        "interaction_relaxed_feature_size": 10,
    })
    return params


def _hugiml_guided_fast_tune_params(grid_name: str | None = None) -> dict[str, Any]:
    """Sentinel configuration consumed by _run_single_model for fast core tuning.

    ``grid_name`` selects which named grid in ``hugiml.hyperparameter_configs``
    drives the sweep. ``"performance_ho"`` is the default; ``"performance"``
    restricts the downstream search to LR, ``"interpretability"`` keeps a
    pattern-only LR surface, and ``"interpretability_ho"`` searches LR versus
    sequential RPTE over that same pattern-only surface. No
    execution_mode override is set here: tune() already evaluates every
    search candidate under 'production' for speed and refits the returned
    model under 'audit' by default, which is what's wanted, since a
    Workbench run can be promoted to Governance afterward and several
    Governance components (patterns, feature_family, pruning, prediction,
    governance_evidence, overview) call get_pattern_info() on whatever
    model they're given.
    """
    resolved = grid_name or DEFAULT_HUGIML_GRID_NAME
    label = "guided_fast_tune" if resolved == DEFAULT_HUGIML_GRID_NAME else f"guided_fast_tune_{resolved}"
    display = "Guided fast tune" if resolved == DEFAULT_HUGIML_GRID_NAME else f"Guided fast tune ({resolved})"
    return {
        "__hugiml_tune__": True,
        "__label__": label,
        "__display_name__": display,
        "param_grid": _hugiml_core_default_grid(resolved),
        "base_params": {"adaptive_binning": True},
        "use_fast_path": True,
    }


def _hugiml_interpretability_fast_tune_params() -> dict[str, Any]:
    """Sentinel configuration for a fast tune restricted to the interpretability grid."""
    return _hugiml_guided_fast_tune_params("interpretability")


def _summarize_base_estimator(value: Any) -> str:
    """Short, readable label for a base_estimator value -- 'HUGIML LR'
    for the built-in logistic-regression branch (base_estimator=None),
    or the wrapped estimator's class name(s) otherwise (e.g. 'RPTE via
    OneVsRestClassifier' for performance_ho's hybrid candidate)."""
    if value is None:
        return "HUGIML LR"
    inner = getattr(value, "estimator", None)
    if inner is not None:
        return f"{type(inner).__name__} via {type(value).__name__}"
    return type(value).__name__


def _summarize_params_for_display(params: dict[str, Any]) -> str:
    """Concise, single-line summary of a run's params for a leaderboard
    cell -- a raw str(dict) is unreadable once a value is a live
    sklearn estimator (e.g. performance_ho's RPTE base_estimator),
    whose full repr can run to dozens of lines."""
    if not params:
        return "{}"
    parts = []
    for key, value in params.items():
        text = _summarize_base_estimator(value) if key == "base_estimator" else value
        parts.append(f"{key}={text}")
    return ", ".join(parts)


def _hugiml_grid_count(grid: dict[str, list[Any]]) -> int:
    total = 1
    for values in (grid or {}).values():
        total *= max(1, len(values or []))
    return total


def _grid_varies_base_estimator(grid: dict[str, list[Any]]) -> bool:
    """True when a grid's base_estimator dimension has more than one
    distinct value (e.g. performance_ho's [None, RPTE-OvR]) -- the
    signal HUGIMLClassifierNative.tune() itself uses to skip the cached
    fast-tune path for that grid."""
    values = (grid or {}).get("base_estimator")
    return bool(values) and len(values) > 1


def _json_safe_value(value: Any) -> Any:
    """Best-effort JSON-safe rendering of one grid value for st.json()
    display. Streamlit's st.json() requires actual JSON-serializable
    data; a grid dimension can legitimately hold live sklearn estimator
    objects (performance_ho's base_estimator), which are not -- render
    those as a short human-readable summary instead of crashing."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if hasattr(value, "get_params"):
        # A fitted or unfitted sklearn estimator (e.g. RPTE, possibly
        # wrapped in OneVsRestClassifier): class name plus its own
        # constructor params, recursively made safe the same way.
        try:
            params = {k: _json_safe_value(v) for k, v in value.get_params(deep=False).items()}
        except Exception:
            params = {}
        return {"__estimator__": type(value).__name__, "params": params}
    return str(value)


def _json_safe_grid(grid: dict[str, list[Any]]) -> dict[str, Any]:
    """A copy of `grid` safe to pass to st.json() -- see _json_safe_value."""
    return {str(key): _json_safe_value(values) for key, values in (grid or {}).items()}

MODEL_SHORT_NAMES = {
    "Logistic Regression": "LR",
    "Decision Tree": "DT",
    "Random Forest": "RF",
    "XGBoost": "XGB",
    "LightGBM": "LGBM",
    "Explainable Boosting Machine": "EBM",
    "EBM": "EBM",
    "RuleFit": "RuleFit",
    "HUGIML": "HUGIML",
}


def _model_short(model_name: str) -> str:
    return MODEL_SHORT_NAMES.get(str(model_name), str(model_name))


def _model_caption(spec: ModelSpec) -> str:
    short = _model_short(spec.name)
    if spec.name == "RuleFit":
        return f"{short}: imodels RuleFit if installed; otherwise RuleFit-style fallback"
    return f"{short}: {spec.name}"


def _run_display_name(run: dict[str, Any]) -> str:
    model = str(run.get("model", ""))
    artifact = run.get("artifact") if isinstance(run.get("artifact"), dict) else {}
    implementation = str((artifact or {}).get("implementation", ""))
    if model == "RuleFit" and implementation == "sklearn-rulefit-style":
        return "RuleFit-style"
    return _model_short(model)


# ---------------------------------------------------------------------------
# Experiment helpers. Kept in this module so the governance runner remains
# HUGIML-centric and backwards-compatible.
# ---------------------------------------------------------------------------


def _is_available(spec: ModelSpec) -> bool:
    return spec.optional_dependency is None or find_spec(spec.optional_dependency) is not None


def _parse_grid(value: str, cast: type) -> list[Any]:
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    if not items:
        return []
    if cast is bool:
        return [x.lower() in {"1", "true", "yes", "y"} for x in items]
    return [cast(x) for x in items]


def _expand_param_grid(base: dict[str, Any], grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    active = {k: v for k, v in grid.items() if v}
    if not active:
        return [dict(base)]
    keys = list(active)
    rows = []
    for combo in itertools.product(*(active[k] for k in keys)):
        params = dict(base)
        params.update(dict(zip(keys, combo)))
        rows.append(params)
    return rows


def _parse_optional_int_grid(value: str) -> list[int | None]:
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    out: list[int | None] = []
    for item in items:
        if item.lower() in {"none", "null", "auto"}:
            out.append(None)
        else:
            out.append(int(item))
    return out


def _parse_adaptive_sample_frac_grid(value: str) -> list[float | bool]:
    """Parse Workbench Advanced adaptive-binning sample fractions."""
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    if not items:
        return [False]
    out: list[float | bool] = []
    for item in items:
        lower = item.lower()
        if lower in {"false", "none", "null", "off", "no", "full", "all"}:
            out.append(False)
            continue
        frac = float(item)
        if not 0 < frac <= 1:
            raise ValueError("adaptive_binning_sample_frac values must be False or in (0, 1].")
        out.append(frac)
    return out


def _l_allows_l2_interaction(params: dict[str, Any]) -> bool:
    try:
        return int(params.get("L", 1)) >= 2 or int(params.get("L", 1)) == -1
    except Exception:
        return True


def _filter_hugiml_interaction_configs(configs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    valid: list[dict[str, Any]] = []
    invalid = 0
    for cfg in configs:
        if (
            _l_allows_l2_interaction(cfg)
            and bool(cfg.get("interaction_relaxed_mining", False))
            and bool(cfg.get("augmented_pair_transforms", True))
        ):
            invalid += 1
            continue
        valid.append(cfg)
    return valid, invalid


class RuleFitClassifierAdapter:
    """RuleFit classifier adapter with explicit implementation labelling.

    Preferred backend: `imodels.RuleFitClassifier`.
    Compatibility backend: legacy standalone `rulefit.RuleFit` when present.
    Fallback backend: sklearn-generated tree-leaf rules + logistic regression.

    The fallback is intentionally labelled as "RuleFit-style fallback" in the UI
    so users do not confuse it with the official imodels implementation.
    """

    def __init__(self, tree_size: int = 4, max_rules: int = 100, random_state: int = 2026):
        self.tree_size = tree_size
        self.max_rules = max_rules
        self.random_state = random_state

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"tree_size": self.tree_size, "max_rules": self.max_rules, "random_state": self.random_state}

    def set_params(self, **params: Any) -> RuleFitClassifierAdapter:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _fit_imodels_rulefit(self, X_df: pd.DataFrame, y: np.ndarray) -> bool:
        if find_spec("imodels") is None:
            return False
        try:
            from imodels import RuleFitClassifier
        except Exception:
            return False

        self.feature_names_ = [str(c) for c in X_df.columns]
        ctor_attempts = [
            {"tree_size": int(self.tree_size), "max_rules": int(self.max_rules), "random_state": int(self.random_state)},
            {"tree_size": int(self.tree_size), "max_rules": int(self.max_rules)},
            {},
        ]
        for kwargs in ctor_attempts:
            try:
                model = RuleFitClassifier(**kwargs)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*Inconsistent values: penalty=.*l1_ratio=.*penalty is deprecated.*",
                        category=UserWarning,
                    )
                    model.fit(X_df, y)
                self.model_ = model
                self.backend_ = "imodels-rulefit"
                return True
            except Exception:
                continue
        return False

    def _fit_external_rulefit(self, X_df: pd.DataFrame, y: np.ndarray) -> bool:
        if find_spec("rulefit") is None:
            return False
        try:
            from rulefit import RuleFit
        except Exception:
            return False

        self.feature_names_ = [str(c) for c in X_df.columns]
        X_values = X_df.to_numpy(dtype=float)
        ctor_attempts = [
            {"tree_size": int(self.tree_size), "max_rules": int(self.max_rules), "random_state": int(self.random_state), "model_type": "c"},
            {"tree_size": int(self.tree_size), "max_rules": int(self.max_rules), "model_type": "c"},
            {"tree_size": int(self.tree_size), "max_rules": int(self.max_rules)},
        ]
        for kwargs in ctor_attempts:
            try:
                model = RuleFit(**kwargs)
                for fit_call in (
                    lambda: model.fit(X_values, y, feature_names=self.feature_names_),
                    lambda: model.fit(X_values, y),
                    lambda: model.fit(X_df, y),
                ):
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message=r".*Inconsistent values: penalty=.*l1_ratio=.*penalty is deprecated.*",
                                category=UserWarning,
                            )
                            fit_call()
                        self.model_ = model
                        self.backend_ = "legacy-rulefit"
                        return True
                    except Exception:
                        continue
            except Exception:
                continue
        return False

    @staticmethod
    def _new_one_hot_encoder() -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # sklearn < 1.2
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def _fit_internal_rulefit(self, X_df: pd.DataFrame, y: np.ndarray) -> None:
        n_estimators = max(20, min(300, int(self.max_rules) // max(1, int(self.tree_size))))
        max_leaf_nodes = max(2, int(self.tree_size))
        self.tree_model_ = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=0.05,
            subsample=0.9,
            random_state=int(self.random_state),
        )
        X_values = X_df.to_numpy(dtype=float)
        self.tree_model_.fit(X_values, y)
        leaves = self.tree_model_.apply(X_values)
        if leaves.ndim == 3:
            leaves = leaves[:, :, 0]
        leaves = leaves.astype(str)
        self.rule_encoder_ = self._new_one_hot_encoder()
        rule_matrix = np.asarray(self.rule_encoder_.fit_transform(leaves), dtype=float)
        design = np.hstack([X_values, rule_matrix])
        self.linear_model_ = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        self.linear_model_.fit(design, y)
        coef = np.asarray(getattr(self.linear_model_, "coef_", [[0.0]]), dtype=float)
        base_coef = np.abs(coef[0, : X_values.shape[1]]) if coef.ndim == 2 else np.zeros(X_values.shape[1])
        tree_imp = np.asarray(getattr(self.tree_model_, "feature_importances_", np.zeros(X_values.shape[1])), dtype=float)
        self.feature_importances_ = base_coef + tree_imp
        self.backend_ = "sklearn-rulefit-style"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RuleFitClassifierAdapter:
        X_df = pd.DataFrame(X).astype(float)
        self.feature_names_ = [str(c) for c in X_df.columns]
        if self._fit_imodels_rulefit(X_df, y):
            return self
        if self._fit_external_rulefit(X_df, y):
            return self
        self._fit_internal_rulefit(X_df, y)
        return self

    def _internal_rule_text(self, estimator: Any, leaf_id: int) -> str:
        tree = estimator.tree_
        feature_names = list(getattr(self, "feature_names_", []))
        stack: list[tuple[int, list[str]]] = [(0, [])]
        while stack:
            node_id, conditions = stack.pop()
            if node_id == int(leaf_id):
                return " and ".join(conditions) or "true"
            left = int(tree.children_left[node_id])
            right = int(tree.children_right[node_id])
            if left == right:
                continue
            feature_index = int(tree.feature[node_id])
            feature_name = (
                feature_names[feature_index]
                if 0 <= feature_index < len(feature_names)
                else f"x{feature_index}"
            )
            threshold = float(tree.threshold[node_id])
            stack.append(
                (right, [*conditions, f"{feature_name} > {threshold:.17g}"])
            )
            stack.append(
                (left, [*conditions, f"{feature_name} <= {threshold:.17g}"])
            )
        return f"leaf_{int(leaf_id)}"

    def _internal_rule_rows(self) -> list[dict[str, Any]]:
        coefficients = np.asarray(self.linear_model_.coef_, dtype=float)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)
        combined = coefficients[0]
        feature_names = list(getattr(self, "feature_names_", []))
        n_features = len(feature_names)
        rows: list[dict[str, Any]] = [
            {"rule": name, "type": "linear", "coef": float(combined[index])}
            for index, name in enumerate(feature_names)
        ]

        estimators = np.asarray(self.tree_model_.estimators_, dtype=object).reshape(-1)
        offset = n_features
        for estimator, categories in zip(estimators, self.rule_encoder_.categories_):
            for category in categories:
                leaf_id = int(float(category))
                coefficient = float(combined[offset])
                rows.append(
                    {
                        "rule": self._internal_rule_text(estimator, leaf_id),
                        "type": "rule",
                        "coef": coefficient,
                    }
                )
                offset += 1
        return rows

    def get_rules(self) -> pd.DataFrame:
        if getattr(self, "backend_", "") == "sklearn-rulefit-style":
            return pd.DataFrame(self._internal_rule_rows())

        model = getattr(self, "model_", None)
        get_rules = getattr(model, "get_rules", None)
        if callable(get_rules):
            rules = get_rules()
            if isinstance(rules, pd.DataFrame):
                return rules
            if hasattr(rules, "to_dict"):
                return pd.DataFrame(rules.to_dict(orient="records"))
            return pd.DataFrame(list(rules))

        rows: list[dict[str, Any]] = []
        for rule in list(getattr(model, "rules_", []) or []):
            coefficient = getattr(rule, "coef", None)
            if coefficient is None:
                args = getattr(rule, "args", None)
                if args is not None and len(args):
                    coefficient = args[0]
            rows.append(
                {
                    "rule": str(getattr(rule, "rule", rule)),
                    "type": "rule",
                    "coef": coefficient,
                }
            )
        return pd.DataFrame(rows)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if getattr(self, "backend_", "") == "sklearn-rulefit-style":
            X_values = pd.DataFrame(X).astype(float).to_numpy(dtype=float)
            leaves = self.tree_model_.apply(X_values)
            if leaves.ndim == 3:
                leaves = leaves[:, :, 0]
            return np.asarray(
                self.rule_encoder_.transform(leaves.astype(str)),
                dtype=float,
            )

        model = getattr(self, "model_", None)
        transform = getattr(model, "transform", None)
        if not callable(transform):
            raise AttributeError("The fitted RuleFit backend does not expose transform().")
        frame = pd.DataFrame(X).astype(float)
        for values in (frame, frame.to_numpy(dtype=float)):
            try:
                return np.asarray(transform(values), dtype=float)
            except Exception:
                continue
        raise ValueError("The fitted RuleFit backend could not transform the supplied rows.")

    def _internal_design(self, X: pd.DataFrame) -> np.ndarray:
        X_values = pd.DataFrame(X).astype(float).to_numpy(dtype=float)
        leaves = self.tree_model_.apply(X_values)
        if leaves.ndim == 3:
            leaves = leaves[:, :, 0]
        rule_matrix = np.asarray(self.rule_encoder_.transform(leaves.astype(str)), dtype=float)
        return np.hstack([X_values, rule_matrix])

    def _raw_score(self, X: pd.DataFrame) -> np.ndarray:
        X_values = pd.DataFrame(X).astype(float).to_numpy(dtype=float)
        if getattr(self, "backend_", "") == "sklearn-rulefit-style":
            proba = np.asarray(self.linear_model_.predict_proba(self._internal_design(X)), dtype=float)
            return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
        if hasattr(self.model_, "predict_proba"):
            proba = np.asarray(self.model_.predict_proba(X_values), dtype=float)
            if proba.ndim == 2 and proba.shape[1] > 1:
                return proba[:, 1]
            return proba.ravel()
        score = np.asarray(self.model_.predict(X_values), dtype=float).ravel()
        if np.nanmin(score) >= 0 and np.nanmax(score) <= 1:
            return score
        return 1.0 / (1.0 + np.exp(-score))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = np.clip(self._raw_score(X), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _build_estimator(model_name: str, params: dict[str, Any], random_state: int):
    if model_name == "Logistic Regression":
        # `penalty` is deprecated in scikit-learn 1.8; C controls regularization strength.
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            solver="lbfgs",
            class_weight=params.get("class_weight"),
            max_iter=int(params.get("max_iter", 1000)),
            random_state=None,
        )
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            class_weight=params.get("class_weight"),
            random_state=random_state,
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            class_weight=params.get("class_weight"),
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=1,
        )
    if model_name == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", -1)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            num_leaves=int(params.get("num_leaves", 31)),
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        )
    if model_name == "EBM":
        from interpret.glassbox import ExplainableBoostingClassifier

        return ExplainableBoostingClassifier(
            max_bins=int(params.get("max_bins", 32)),
            interactions=int(params.get("interactions", 5)),
            random_state=random_state,
        )
    if model_name == "RuleFit":
        return RuleFitClassifierAdapter(
            tree_size=int(params.get("tree_size", 4)),
            max_rules=int(params.get("max_rules", 100)),
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _default_model_params(model_name: str) -> dict[str, Any]:
    if model_name == "HUGIML":
        return _default_params()
    if model_name == "Logistic Regression":
        return {"C": 1.0, "max_iter": 1000, "class_weight": None}
    if model_name == "Decision Tree":
        return {"max_depth": None, "min_samples_leaf": 1, "class_weight": None}
    if model_name == "Random Forest":
        return {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "class_weight": None}
    if model_name in {"XGBoost", "LightGBM"}:
        return {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}
    if model_name == "EBM":
        return {"max_bins": 32, "interactions": 5}
    if model_name == "RuleFit":
        return {"tree_size": 4, "max_rules": 100}
    return {}




def _safe_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Return a dense numeric feature matrix suitable for sklearn/xgboost/lightgbm.

    HUGIML can consume the dashboard frame directly, but standard estimators
    frequently fail on object/category/datetime columns. This helper performs a
    deterministic, UI-friendly encoding step for the Workbench comparisons.
    """
    if X.empty:
        raise ValueError("No feature columns are available for modeling.")

    frame = X.copy()
    frame.columns = [str(c) for c in frame.columns]

    numeric_parts: list[pd.Series] = []
    categorical_parts: list[pd.Series] = []
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_bool_dtype(series):
            numeric_parts.append(series.astype(float).rename(col))
        elif pd.api.types.is_numeric_dtype(series):
            numeric_parts.append(pd.to_numeric(series, errors="coerce").rename(col))
        elif pd.api.types.is_datetime64_any_dtype(series):
            numeric_parts.append(pd.to_datetime(series, errors="coerce").astype("int64").replace(-9223372036854775808, np.nan).rename(col))
        else:
            categorical_parts.append(series.astype("string").fillna("__missing__").rename(col))

    pieces: list[pd.DataFrame] = []
    if numeric_parts:
        numeric = pd.concat(numeric_parts, axis=1)
        for col in numeric.columns:
            if numeric[col].isna().all():
                numeric[col] = 0.0
            else:
                numeric[col] = numeric[col].fillna(float(numeric[col].median()))
        pieces.append(numeric.astype(float))

    if categorical_parts:
        categorical = pd.concat(categorical_parts, axis=1)
        pieces.append(pd.get_dummies(categorical, dummy_na=False, dtype=float))

    encoded = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=frame.index)
    if encoded.empty:
        raise ValueError("Encoding produced no usable feature columns.")

    # XGBoost rejects feature names containing [, ] or <; deduplicate after encoding.
    seen: dict[str, int] = {}
    clean_cols: list[str] = []
    for raw in encoded.columns:
        name = str(raw).replace("[", "(").replace("]", ")").replace("<", "lt").replace(">", "gt")
        name = name.replace("\n", " ").strip() or "feature"
        count = seen.get(name, 0)
        seen[name] = count + 1
        clean_cols.append(name if count == 0 else f"{name}_{count}")
    encoded.columns = clean_cols
    return encoded.astype(float)


def _fresh_estimator(estimator: Any) -> Any:
    try:
        return clone(estimator)
    except Exception:
        return copy.deepcopy(estimator)


def _cv_predictions(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int,
    random_state: int,
    *,
    cv_splits: Any = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Return stitched out-of-fold predictions and mean fold ROC-AUC.

    When a HUGIML tune result exposes ``cv_splits_``, promotion consistency
    depends on reusing those exact folds rather than constructing a merely
    equivalent splitter.  Other estimators continue to use the Workbench's
    deterministic stratified folds.
    """
    y_arr = np.asarray(y, dtype=int)
    if cv_splits:
        split_iter = [
            (np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int))
            for train_idx, test_idx in cv_splits
        ]
    else:
        counts = np.bincount(y_arr)
        positive_counts = counts[counts > 0]
        max_splits = int(positive_counts.min()) if len(positive_counts) else 0
        n_splits = min(int(cv), max_splits)
        if n_splits < 2:
            raise ValueError("Cross-validation needs at least two rows in each target class.")
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = list(folds.split(X, y_arr))

    y_proba = np.zeros(len(y_arr), dtype=float)
    y_pred = np.zeros(len(y_arr), dtype=int)
    assigned = np.zeros(len(y_arr), dtype=bool)
    aucs: list[float] = []
    for train_idx, test_idx in split_iter:
        fitted = _fresh_estimator(estimator)
        fitted.fit(X.iloc[train_idx], y_arr[train_idx])
        fold_proba = _predict_proba_1(fitted, X.iloc[test_idx])
        y_proba[test_idx] = fold_proba
        y_pred[test_idx] = np.asarray(fitted.predict(X.iloc[test_idx]), dtype=int)
        assigned[test_idx] = True
        try:
            aucs.append(float(roc_auc_score(y_arr[test_idx], fold_proba)))
        except Exception:
            pass
    if not bool(assigned.all()):
        missing = np.flatnonzero(~assigned)
        raise ValueError(f"CV splits did not produce predictions for {len(missing)} row(s).")
    return y_pred, y_proba, float(np.mean(aucs)) if aucs else None

def _predict_proba_1(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        arr = np.asarray(proba)
        return arr[:, 1] if arr.ndim == 2 and arr.shape[1] > 1 else arr.ravel()
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-score))
    pred = np.asarray(model.predict(X), dtype=float).ravel()
    return pred



def _scalar_metric(value: Any) -> float | None:
    """Convert metric-like objects to a sortable/displayable scalar.

    HUGIML tuning artifacts and some third-party estimators may expose scores as
    NumPy arrays, pandas objects, or short sequences. Returns the nanmean of
    finite numeric values for arrays/sequences, and None when no scalar can be
    derived.
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes, dict)):
        return None
    try:
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return None
        arr = arr.reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(np.nanmean(arr))
    except Exception:
        try:
            scalar = float(value)
            return scalar if np.isfinite(scalar) else None
        except Exception:
            return None

def _metric_row(y_true, y_pred, y_proba) -> dict[str, float | int | None]:
    row: dict[str, float | int | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        row["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        row["roc_auc"] = None
    return row


def _feature_importance(model: Any, X: pd.DataFrame, max_rows: int = 80) -> pd.DataFrame:
    """Return a normalized, descending feature-importance table.

    HUGIML exposes a rich ``feature_importances()`` DataFrame with original,
    pattern, and augmented-pair feature families. Standard sklearn-like models
    expose either ``feature_importances_`` or ``coef_``. The Workbench keeps the
    richer columns when available and always provides ``feature`` +
    ``importance`` for plotting.
    """
    # RPTE has a staged representation. Source columns selected in accepted
    # splits are represented through leaf indicators, while non-tree
    # original/pattern/augmented columns remain direct LR terms. Rank the
    # actual final terms before
    # trying HUGIML's source-feature importance API so the Workbench cannot
    # mix the two stages.
    rpte_rows = _get_rpte_final_term_rows(model, include_zero_direct=True)
    rpte_flow = _get_rpte_feature_flow_audit(model)
    if rpte_rows and rpte_flow:
        rules = _rpte_rules_to_frame(rpte_rows)
        if not rules.empty:
            rules = rules.copy()
            rules["feature"] = rules["rule_id"].astype(str) + ": " + rules["rule_preview"].astype(str)
            rules["importance"] = pd.to_numeric(rules["abs_coefficient"], errors="coerce").fillna(0.0)
            rules["feature_type"] = np.where(
                rules.get("is_leaf_term", False),
                "rpte_rule",
                np.where(
                    rules.get("is_direct_source_term", False),
                    rules.get("source_family", "unknown").map(
                        {
                            "original": "original",
                            "pattern": "pattern",
                            "augmented_pair": "augmented_pair",
                        }
                    ).fillna("rpte_direct_source"),
                    "rpte_fallback_term",
                ),
            )
            return rules.sort_values("importance", ascending=False).head(max_rows).reset_index(drop=True)

    if hasattr(model, "feature_importances") and callable(getattr(model, "feature_importances")):
        try:
            rich = model.feature_importances()
            if isinstance(rich, pd.DataFrame) and not rich.empty:
                df = rich.copy()
                if "importance" not in df.columns:
                    if "abs_coefficient" in df.columns:
                        df["importance"] = pd.to_numeric(df["abs_coefficient"], errors="coerce")
                    elif "coefficient" in df.columns:
                        df["importance"] = pd.to_numeric(df["coefficient"], errors="coerce").abs()
                if "feature" not in df.columns:
                    if "display_name" in df.columns:
                        df["feature"] = df["display_name"].astype(str)
                    elif "pattern" in df.columns:
                        df["feature"] = df["pattern"].astype(str)
                    elif df.index.name:
                        df["feature"] = df.index.astype(str)
                if "feature" in df.columns and "importance" in df.columns:
                    df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
                    return df.sort_values("importance", ascending=False).head(max_rows).reset_index(drop=True)
        except Exception:
            pass

    cols = [str(c) for c in X.columns]
    values = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_"), dtype=float)
        values = np.abs(coef[0] if coef.ndim > 1 else coef)
    elif hasattr(model, "term_importances"):
        try:
            values = np.asarray(model.term_importances(), dtype=float)
            cols = [str(x) for x in getattr(model, "term_names_", cols)]
        except Exception:
            values = None

    if values is None or len(values) == 0:
        return pd.DataFrame()
    n = min(len(values), len(cols))
    df = pd.DataFrame({"feature": cols[:n], "importance": values[:n]})
    return df.sort_values("importance", ascending=False).head(max_rows).reset_index(drop=True)


def _run_single_model(
    model_name: str,
    category: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int,
    random_state: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    if model_name == "HUGIML":
        raw_params = dict(params)
        if raw_params.pop("__hugiml_tune__", False):
            from hugiml import HUGIMLClassifierNative

            param_grid = raw_params.pop("param_grid", None) or _hugiml_core_default_grid()
            param_grid["adaptive_binning"] = [True]
            base_params = raw_params.pop("base_params", {}) or {}
            base_params["adaptive_binning"] = True
            use_fast_path = bool(raw_params.pop("use_fast_path", True))
            display_name = raw_params.pop("__display_name__", "Guided fast tune")
            raw_params.pop("__label__", None)
            result = HUGIMLClassifierNative.tune(
                X,
                y,
                cv=cv,
                scoring="roc_auc",
                param_grid=param_grid,
                refit=True,
                base_params=base_params,
                random_state=random_state,
                shuffle=True,
                use_fast_path=use_fast_path,
                return_dataframe=True,
            )
            model = getattr(result, "best_estimator_", None)
            selected_params = dict(getattr(result, "best_params_", {}) or {})
            selected_params["adaptive_binning"] = True
            params = {
                "mode": "guided_fast_tune",
                "selected_params": selected_params,
                "candidate_count": _hugiml_grid_count(param_grid),
                "fast_path_used": bool(getattr(result, "fast_path_used_", False)),
                "param_grid": param_grid,
                "display_name": display_name,
            }
        else:
            params = dict(raw_params)
            result = fit_hugiml_config(X, y, params=params, cv=cv, scoring="roc_auc", random_state=random_state)
            model = getattr(result, "best_estimator_", None)
            selected_params = dict(params)
        if model is None:
            return {
                "model": model_name,
                "category": category,
                "status": getattr(result, "status_", "failed"),
                "params": params,
                "diagnostic": getattr(result, "error_", None),
                "fit_time_sec": round(time.perf_counter() - started, 4),
                "artifact": result,
            }
        cv_score = getattr(result, "best_score_", None)
        evaluation_scope = "out_of_fold"
        evaluation_error = None
        try:
            y_pred, y_proba, recomputed_cv_score = _cv_predictions(
                model,
                X,
                y,
                cv,
                random_state,
                cv_splits=getattr(result, "cv_splits_", None),
            )
            if cv_score is None:
                cv_score = recomputed_cv_score
        except Exception as exc:
            # Keep the run inspectable if an unusual third-party estimator
            # cannot be cloned.  The scope is carried into Governance so these
            # fallback numbers can never be mistaken for validation evidence.
            y_proba = _predict_proba_1(model, X)
            y_pred = np.asarray(model.predict(X))
            evaluation_scope = "full_data_refit_fallback"
            evaluation_error = repr(exc)
        metrics = _metric_row(y, y_pred, y_proba)
    else:
        estimator = _build_estimator(model_name, params, random_state)
        X_model = _safe_numeric_frame(X)
        y_pred, y_proba, cv_score = _cv_predictions(estimator, X_model, y, cv, random_state)
        model = _fresh_estimator(estimator)
        model.fit(X_model, y)
        metrics = _metric_row(y, y_pred, y_proba)
        X = X_model
        evaluation_scope = "out_of_fold"
        evaluation_error = None

    metrics["cv_roc_auc"] = _scalar_metric(cv_score)
    artifact = {
        "model": model,
        "params": params,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_importance": _feature_importance(model, X),
        "feature_frame": X,
        "confusion_matrix": confusion_matrix(y, y_pred),
        "evaluation_scope": evaluation_scope,
        "evaluation_error": evaluation_error,
        "evaluation_cv": int(cv),
        "evaluation_random_state": int(random_state),
        "implementation": getattr(model, "backend_", "native"),
        "tuning_result": result if model_name == "HUGIML" else None,
    }
    return {
        "model": model_name,
        "category": category,
        "status": "ok",
        "params": params,
        **metrics,
        "fit_time_sec": round(time.perf_counter() - started, 4),
        "artifact": artifact,
    }


def run_experiments(
    selected: dict[str, list[str]],
    param_map: dict[str, list[dict[str, Any]]],
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int,
    random_state: int,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for category, names in selected.items():
        for model_name in names:
            configs = param_map.get(model_name, [_default_model_params(model_name)])
            multi = len(configs) > 1  # only suffix when there are multiple configs
            for idx, params in enumerate(configs, start=1):
                label_params = dict(params)
                try:
                    row = _run_single_model(model_name, category, label_params, X, y, cv, random_state)
                except Exception as exc:
                    row = {
                        "model": model_name,
                        "category": category,
                        "status": "failed",
                        "params": label_params,
                        "diagnostic": repr(exc),
                        "fit_time_sec": None,
                        "artifact": None,
                    }
                base_id = model_name.replace(' ', '_').lower()
                if model_name == "HUGIML" and isinstance(label_params, dict) and label_params.get("__label__"):
                    base_id = f"{base_id}_{label_params.get('__label__')}"
                row["run_id"] = f"{base_id}_{idx}" if multi else base_id
                row["config_index"] = idx if multi else None
                runs.append(row)
    return runs


# ---------------------------------------------------------------------------
# UI rendering
# ---------------------------------------------------------------------------


def _render_intro(ctx: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="hugiml-hero hugiml-workbench-hero">
          <span class="hugiml-eyebrow">Experiment Workbench</span>
          <h1>Model Comparison &amp; Benchmarking</h1>
          <p>Configure HUGIML, benchmark it against baseline, ensemble, and interpretable models,
          inspect metrics and plots, then promote a fitted HUGIML run into Governance.</p>
          <div class="hugiml-chip-row">
            <span class="hugiml-chip">Data: {ctx['mode']}</span>
            <span class="hugiml-chip">Rows: {ctx['meta']['n_rows']:,}</span>
            <span class="hugiml-chip">Features: {ctx['meta']['n_features']:,}</span>
            <span class="hugiml-chip">Governance: HUGIML only</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_workbench_workflow() -> None:
    st.markdown(
        """
        <div class="hugiml-workflow">
          <div class="hugiml-step">
            <span class="hugiml-step-num">1</span>
            <b>Select models</b>
            <span>Choose baselines, ensembles, HUGIML, EBM, or RuleFit-style runs.</span>
          </div>
          <div class="hugiml-step">
            <span class="hugiml-step-num">2</span>
            <b>Configure candidates</b>
            <span>Use Auto, Guided, or Advanced grid settings for tuning.</span>
          </div>
          <div class="hugiml-step">
            <span class="hugiml-step-num">3</span>
            <b>Compare evidence</b>
            <span>Review leaderboard winners, ROC curves, PR curves, and feature importance.</span>
          </div>
          <div class="hugiml-step">
            <span class="hugiml-step-num">4</span>
            <b>Promote to Governance</b>
            <span>Send a fitted HUGIML run to the evidence dashboard for audits.</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_model_picker() -> dict[str, list[str]]:
    st.markdown("### 1. Select models")
    st.caption("Compact names are used to conserve space; hover or read the helper text for full names.")
    selected: dict[str, list[str]] = {}
    cols = st.columns(3)
    defaults = {
        "Baselines": ["Logistic Regression", "Decision Tree"],
        "Ensembles": ["Random Forest"],
        "Interpretable Models": ["HUGIML"],
    }
    for i, (category, specs) in enumerate(MODEL_CATALOG.items()):
        with cols[i]:
            available_specs = [spec for spec in specs if _is_available(spec)]
            unavailable_specs = [spec for spec in specs if not _is_available(spec)]
            options = [spec.name for spec in available_specs]
            chosen = st.multiselect(
                category,
                options=options,
                default=[m for m in defaults.get(category, []) if m in options],
                format_func=_model_short,
                help="Select one or more models in this family.",
                key=f"wb_models_{category}",
            )
            selected[category] = list(chosen)
            if available_specs:
                st.caption(" · ".join(_model_caption(spec) for spec in available_specs))
            if unavailable_specs:
                st.caption("Unavailable: " + " · ".join(f"{_model_short(spec.name)} needs {spec.optional_dependency}" for spec in unavailable_specs))
    return selected

def _render_hugiml_config() -> list[dict[str, Any]]:
    st.markdown("#### HUGIML")
    mode = st.radio(
        "Configuration mode",
        ["Auto", "Guided", "Advanced"],
        horizontal=True,
        key="wb_hugiml_mode",
        help=(
            "Auto runs one strong default. Guided uses the HUGIML core default hyperparameter "
            "grid through the fast tuning path. Advanced lets you edit a custom grid."
        ),
    )

    params = _hugiml_auto_params()
    params["adaptive_binning"] = True

    if mode == "Auto":
        st.caption("Runs one best-default HUGIML configuration. Adaptive binning is always enabled.")
        with st.expander("View auto parameters", expanded=False):
            st.json(params, expanded=False)
        return [params]

    if mode == "Guided":
        grid_names = _hugiml_core_grid_names()
        grid_name = (
            st.selectbox(
                "Tuning grid",
                grid_names,
                index=grid_names.index(DEFAULT_HUGIML_GRID_NAME)
                if DEFAULT_HUGIML_GRID_NAME in grid_names
                else 0,
                help=(
                    "'performance' searches feature_mode='original_plus_patterns' for "
                    "predictive performance. 'interpretability' restricts the downstream "
                    "model to feature_mode='patterns_only' with relaxed interaction mining, "
                    "so its explanation surface is the mined patterns alone. "
                    "'performance_ho' additionally searches a hybrid downstream branch: half "
                    "the candidates use HUGIML's built-in logistic regression, half use RPTE "
                    "(leaf-wise bounded-lookahead trees + logistic regression) for higher-order "
                    "feature interactions the logistic branch can't represent directly."
                ),
                key="wb_hugiml_grid_name",
            )
            if len(grid_names) > 1
            else (grid_names[0] if grid_names else DEFAULT_HUGIML_GRID_NAME)
        )
        guided = _hugiml_guided_fast_tune_params(grid_name)
        grid = guided["param_grid"]
        st.caption(
            f"Uses the '{grid_name}' HUGIML hyperparameter grid with `use_fast_path=True`; "
            "adaptive binning is enabled for every candidate."
        )
        if _grid_varies_base_estimator(grid):
            st.caption(
                "This grid varies `base_estimator` (HUGIML logistic regression vs. RPTE). "
                "The cached fast-tune path supports this: mining is cached per (G, L, topK) as "
                "usual, and the downstream feature matrix built from it is cached per (G, L, "
                "topK, feature_mode) and reused across every base_estimator candidate at that "
                "key -- only the final per-candidate estimator fit repeats."
            )
        c1, c2 = st.columns([1, 3])
        c1.metric("Candidates", f"{_hugiml_grid_count(grid):,}")
        c2.caption("The Workbench records the selected best configuration after CV tuning.")
        with st.expander("View guided grid", expanded=False):
            st.json(_json_safe_grid(grid), expanded=False)
        return [guided]

    params["adaptive_binning"] = st.toggle(
        "Adaptive binning",
        value=True,
        help="Default is on. When enabled, HUGIML uses adaptive/default binning and B is kept at -1.",
        key="wb_hugiml_adaptive_binning",
    )
    params["adaptive_binning_sample_frac"] = False
    params["adaptive_binning_sample_random_state"] = 42
    if params["adaptive_binning"]:
        params["B"] = -1
    c1, c2, c3, c4 = st.columns(4)
    if params["adaptive_binning"]:
        c1.number_input("B", value=-1, min_value=-1, max_value=-1, step=1, disabled=True, help="B has no effect while adaptive binning is enabled; it is kept at -1.")
    else:
        params["B"] = int(c1.number_input("B", value=2, min_value=2, step=1, help="With adaptive binning disabled, choose a fixed bin count B >= 2."))
    params["L"] = int(c2.number_input("L", value=int(params["L"]), min_value=1, max_value=5, step=1, help=PARAM_HINTS["L"]))
    params["topK"] = int(c3.number_input("topK", value=int(params["topK"]), min_value=1, step=1, help=PARAM_HINTS["topK"]))
    params["G"] = float(c4.number_input("G", value=float(params["G"]), min_value=0.0, step=0.001, format="%.5f", help=PARAM_HINTS["G"]))
    feature_modes = _hugiml_valid_feature_modes()
    default_feature_modes = [m for m in ["original_plus_patterns"] if m in feature_modes]
    if not default_feature_modes and feature_modes:
        default_feature_modes = [feature_modes[0]]
    selected_feature_modes = st.multiselect(
        "Feature modes",
        feature_modes,
        default=default_feature_modes,
        help=(
            PARAM_HINTS["feature_mode"]
            + " Select multiple values to include feature_mode in Advanced hyperparameter tuning."
        ),
        key="wb_hugiml_feature_modes",
    )
    if not selected_feature_modes:
        st.warning("Select at least one feature mode. Falling back to the first valid mode for this run.")
        selected_feature_modes = default_feature_modes or feature_modes[:1]
    params["feature_mode"] = selected_feature_modes[0]
    params["topk_budget_strict"] = st.toggle("Strict top-K budget", value=False, help=PARAM_HINTS["topk_budget_strict"], key="wb_hugiml_topk_strict")

    st.markdown("##### Interaction path")
    st.caption(
        "Guided keeps the HUGIML core grid. Advanced can tune augmented-pair scoring "
        "or interaction-relaxed mining explicitly."
    )
    i1, i2, i3 = st.columns(3)
    params["augmented_pair_transforms"] = bool(i1.toggle(
        "Augmented pair transforms",
        value=True,
        help=PARAM_HINTS["augmented_pair_transforms"],
        key="wb_hugiml_augmented_pair_transforms",
    ))
    params["interaction_relaxed_mining"] = bool(i2.toggle(
        "Interaction relaxed mining",
        value=False,
        help=PARAM_HINTS["interaction_relaxed_mining"],
        key="wb_hugiml_interaction_relaxed_mining",
    ))
    default_aug_mode = str(params.get("augmented_pair_mode", "interaction_information"))
    if default_aug_mode not in AUGMENTED_PAIR_MODE_OPTIONS:
        default_aug_mode = "interaction_information"
    params["augmented_pair_mode"] = str(i3.selectbox(
        "Augmented pair scoring",
        AUGMENTED_PAIR_MODE_OPTIONS,
        index=AUGMENTED_PAIR_MODE_OPTIONS.index(default_aug_mode),
        help=PARAM_HINTS["augmented_pair_mode"],
        key="wb_hugiml_augmented_pair_mode",
    ))
    s1, s2, s3 = st.columns(3)
    params["aug_feature_size"] = int(s1.number_input(
        "Augmented source count",
        value=int(params.get("aug_feature_size", 10)),
        min_value=2,
        step=1,
        help=PARAM_HINTS["aug_feature_size"],
        key="wb_hugiml_aug_feature_size",
    ))
    params["max_pair_features"] = int(s2.number_input(
        "Max pair features",
        value=int(params.get("max_pair_features", 10)),
        min_value=1,
        step=1,
        help=PARAM_HINTS["max_pair_features"],
        key="wb_hugiml_max_pair_features",
    ))
    params["interaction_relaxed_feature_size"] = int(s3.number_input(
        "Relaxed source count",
        value=int(params.get("interaction_relaxed_feature_size", 10)),
        min_value=2,
        step=1,
        help=PARAM_HINTS["interaction_relaxed_feature_size"],
        key="wb_hugiml_interaction_relaxed_feature_size",
    ))
    partner_raw = st.text_input(
        "Interaction-information partner count",
        value="None" if params.get("ii_partner_size") is None else str(params.get("ii_partner_size")),
        help=PARAM_HINTS["ii_partner_size"],
        key="wb_hugiml_ii_partner_size",
    ).strip()
    params["ii_partner_size"] = None if partner_raw.lower() in {"", "none", "null", "auto"} else int(partner_raw)
    if _l_allows_l2_interaction(params) and params["augmented_pair_transforms"] and params["interaction_relaxed_mining"]:
        st.warning(
            "For L>=2, interaction relaxed mining and augmented pair transforms are alternative paths. "
            "Advanced grid candidates that enable both are omitted."
        )

    st.markdown("##### Downstream estimator")
    st.caption(
        "HUGIML's built-in logistic regression fits directly on the mined pattern/original/"
        "augmented-pair matrix. RPTE (leaf-wise bounded-lookahead trees + logistic regression) "
        "additionally searches higher-order feature interactions the logistic branch can't "
        "represent directly -- see the `performance_ho` named grid (Guided mode) for a pre-built "
        "version of this hybrid search with a validated leaf configuration."
    )
    downstream_options = st.multiselect(
        "Downstream estimator(s) to compare",
        ["HUGIML logistic regression", "RPTE (bounded-lookahead trees)"],
        default=["HUGIML logistic regression"],
        help=(
            "Select both to run a hybrid search -- one candidate set per mining "
            "configuration for each selected estimator -- and let CV ROC-AUC pick the winner, "
            "the same comparison performance_ho makes with a single RPTE configuration."
        ),
        key="wb_hugiml_downstream_estimators",
    )
    base_estimator_values: list[Any] = []
    if "HUGIML logistic regression" in downstream_options:
        base_estimator_values.append(None)
    if "RPTE (bounded-lookahead trees)" in downstream_options:
        r1, r2, r3, r4 = st.columns(4)
        leaf_config_options = sorted(RPTE_LEAF_CONFIGS)
        rpte_leaf_config = r1.selectbox(
            "RPTE leaf config",
            leaf_config_options,
            index=leaf_config_options.index("3xD") if "3xD" in leaf_config_options else 0,
            help="Leaf budget per tree is LEAF_CONFIGS[leaf_config] * depth -- 3xD is the "
                 "configuration validated in the performance_ho grid.",
            key="wb_rpte_leaf_config",
        )
        rpte_depth = int(r2.number_input(
            "RPTE depth", value=4, min_value=1, max_value=10, step=1,
            help="Effective tree depth cap is 2x this value (root+child microtrees use 2 levels per split).",
            key="wb_rpte_depth",
        ))
        rpte_n_estimators = int(r3.number_input(
            "RPTE n_estimators", value=10, min_value=1, max_value=100, step=1,
            help="Maximum boosting rounds; a round is only kept if it lowers deviance, so fewer "
                 "than this may actually be fit.",
            key="wb_rpte_n_estimators",
        ))
        lookahead_label = r4.selectbox(
            "RPTE lookahead",
            ["adaptive", "always on", "always off"],
            index=0,
            help="'adaptive' engages the bounded-lookahead microtree search only when no single "
                 "raw feature has real marginal signal on its own (e.g. pure parity); 'always on' "
                 "forces it; 'always off' uses the plain sequential (sklearn-tree) backend.",
            key="wb_rpte_lookahead",
        )
        enable_lookahead_value: bool | str = {
            "adaptive": "adaptive", "always on": True, "always off": False,
        }[lookahead_label]
        rpte_estimator = LeafWiseBoundedLookaheadRPTEFeatureLR(
            leaf_config=rpte_leaf_config,
            depth=rpte_depth,
            n_estimators=rpte_n_estimators,
            enable_lookahead=enable_lookahead_value,
        )
        base_estimator_values.append(OneVsRestClassifier(rpte_estimator, n_jobs=1))
    if not base_estimator_values:
        st.warning("Select at least one downstream estimator. Falling back to HUGIML logistic regression.")
        base_estimator_values = [None]

    with st.expander("Advanced grid values", expanded=True):
        if params["adaptive_binning"]:
            st.caption("Comma-separated values create multiple candidate HUGIML runs. Blank fields keep the values above. B is locked at -1 while adaptive binning is enabled.")
        else:
            st.caption("Comma-separated values create multiple candidate HUGIML runs. Blank fields keep the values above. With adaptive binning off, every B value must be >= 2.")
        g1, g2, g3, g4 = st.columns(4)
        b_values = [-1] if params["adaptive_binning"] else _parse_grid(g4.text_input("B values", value=str(params["B"]), key="wb_hugiml_b_grid"), int)
        if not params["adaptive_binning"] and any(int(v) < 2 for v in b_values):
            st.error("Advanced validation: when adaptive binning is off, B must be >= 2. Invalid B values were removed.")
            b_values = [int(v) for v in b_values if int(v) >= 2] or [2]
        if params["adaptive_binning"]:
            g4.text_input("B values", value="-1", disabled=True, key="wb_hugiml_b_grid_locked", help="B is fixed at -1 while adaptive binning is enabled.")
        h1, h2, h3, h4 = st.columns(4)
        h5, h6, h7, h8 = st.columns(4)
        h9, h10 = st.columns(2)
        selected_aug_modes = h3.multiselect(
            "Augmented pair scoring values",
            AUGMENTED_PAIR_MODE_OPTIONS,
            default=[params["augmented_pair_mode"]],
            help="Values for augmented_pair_mode in Advanced tuning.",
            key="wb_hugiml_augmented_pair_mode_grid",
        ) or [params["augmented_pair_mode"]]
        sample_help = PARAM_HINTS["adaptive_binning_sample_frac"]
        sample_values = (
            _parse_adaptive_sample_frac_grid(
                h9.text_input(
                    "Adaptive sample fraction values",
                    value="False",
                    disabled=not params["adaptive_binning"],
                    help=sample_help,
                    key="wb_hugiml_adaptive_sample_frac_grid",
                )
            )
            if params["adaptive_binning"]
            else [False]
        )
        sample_seed_values = (
            _parse_grid(
                h10.text_input(
                    "Adaptive sample seed values",
                    value=str(params["adaptive_binning_sample_random_state"]),
                    disabled=not params["adaptive_binning"],
                    help=PARAM_HINTS["adaptive_binning_sample_random_state"],
                    key="wb_hugiml_adaptive_sample_seed_grid",
                ),
                int,
            )
            if params["adaptive_binning"]
            else [42]
        )
        grid = {
            "L": _parse_grid(g1.text_input("L values", value=str(params["L"]), key="wb_hugiml_l_grid"), int),
            "topK": _parse_grid(g2.text_input("topK values", value=str(params["topK"]), key="wb_hugiml_topk_grid"), int),
            "G": _parse_grid(g3.text_input("G values", value=str(params["G"]), key="wb_hugiml_g_grid"), float),
            "B": b_values,
            "feature_mode": selected_feature_modes,
            "adaptive_binning_sample_frac": sample_values,
            "adaptive_binning_sample_random_state": sample_seed_values,
            "augmented_pair_transforms": _parse_grid(h1.text_input("Augmented pair values", value=str(params["augmented_pair_transforms"]), key="wb_hugiml_augmented_pair_grid"), bool),
            "interaction_relaxed_mining": _parse_grid(h2.text_input("Interaction relaxed values", value=str(params["interaction_relaxed_mining"]), key="wb_hugiml_interaction_relaxed_grid"), bool),
            "augmented_pair_mode": selected_aug_modes,
            "aug_feature_size": _parse_grid(h4.text_input("Augmented source count values", value=str(params["aug_feature_size"]), key="wb_hugiml_aug_feature_size_grid"), int),
            "max_pair_features": _parse_grid(h5.text_input("Max pair feature values", value=str(params["max_pair_features"]), key="wb_hugiml_max_pair_features_grid"), int),
            "interaction_relaxed_feature_size": _parse_grid(h6.text_input("Relaxed source count values", value=str(params["interaction_relaxed_feature_size"]), key="wb_hugiml_interaction_relaxed_feature_size_grid"), int),
            "ii_partner_size": _parse_optional_int_grid(h7.text_input("Partner count values", value=partner_raw or "None", key="wb_hugiml_ii_partner_size_grid")),
            "base_estimator": base_estimator_values,
        }
    configs = _expand_param_grid(params, grid)
    if params["adaptive_binning"]:
        for cfg in configs:
            cfg["B"] = -1
    else:
        for cfg in configs:
            cfg["adaptive_binning_sample_frac"] = False
            cfg["adaptive_binning_sample_random_state"] = 42
    configs, invalid_configs = _filter_hugiml_interaction_configs(configs)
    if invalid_configs:
        st.warning(f"Omitted {invalid_configs:,} invalid HUGIML interaction-path candidate(s).")
    if not configs:
        st.error("No valid HUGIML Advanced candidates remain. Disable one interaction path or use L=1.")
        fallback = dict(params)
        if _l_allows_l2_interaction(fallback) and fallback.get("interaction_relaxed_mining"):
            fallback["augmented_pair_transforms"] = False
        configs = [fallback]
    return configs

def _render_generic_config(model_name: str) -> list[dict[str, Any]]:
    st.markdown(f"#### {_model_short(model_name)}")
    st.caption(model_name)
    params = _default_model_params(model_name)

    key = model_name.replace(" ", "_").lower()
    mode = st.radio(
        "Configuration mode",
        ["Single", "Hyperparameter grid"],
        horizontal=True,
        key=f"wb_{key}_mode",
        help="Single runs one configuration. Hyperparameter grid sweeps comma-separated values and selects the best on validation CV ROC-AUC.",
    )

    if model_name == "Logistic Regression":
        c1, c2 = st.columns(2)
        params["C"] = float(c1.number_input("Regularization strength C", value=1.0, min_value=0.0001, step=0.1, key="wb_lr_c"))
        params["max_iter"] = int(c2.number_input("Max iterations", value=1000, min_value=100, step=100, key="wb_lr_max_iter"))
        st.caption("Uses the stable lbfgs solver. Newer scikit-learn versions deprecate the explicit penalty argument, so C controls regularization.")
        if mode == "Hyperparameter grid":
            with st.expander("Grid values", expanded=True):
                st.caption("Comma-separated values sweep multiple candidates. Blank keeps the single value above.")
                g1, g2 = st.columns(2)
                grid = {
                    "C": _parse_grid(g1.text_input("C values", value=str(params["C"]), key="wb_lr_c_grid"), float),
                    "max_iter": _parse_grid(g2.text_input("max_iter values", value=str(params["max_iter"]), key="wb_lr_max_iter_grid"), int),
                }
            return _expand_param_grid(params, grid)

    elif model_name == "Decision Tree":
        c1, c2 = st.columns(2)
        depth = c1.number_input("Max depth (0 = none)", value=4, min_value=0, step=1, key="wb_dt_depth")
        params["max_depth"] = None if depth == 0 else int(depth)
        params["min_samples_leaf"] = int(c2.number_input("Min samples leaf", value=1, min_value=1, step=1, key="wb_dt_leaf"))
        if mode == "Hyperparameter grid":
            with st.expander("Grid values", expanded=True):
                st.caption("Comma-separated values. Use 0 for no depth limit. Blank keeps the value above.")
                g1, g2 = st.columns(2)
                raw_depths = _parse_grid(g1.text_input("max_depth values (0=none)", value=str(depth), key="wb_dt_depth_grid"), int)
                grid: dict[str, list[Any]] = {
                    "max_depth": [None if d == 0 else d for d in raw_depths],
                    "min_samples_leaf": _parse_grid(g2.text_input("min_samples_leaf values", value=str(params["min_samples_leaf"]), key="wb_dt_leaf_grid"), int),
                }
            return _expand_param_grid(params, grid)

    elif model_name == "Random Forest":
        c1, c2, c3 = st.columns(3)
        params["n_estimators"] = int(c1.number_input("Trees", value=200, min_value=10, step=10, key="wb_rf_trees"))
        depth = c2.number_input("Max depth (0 = none)", value=0, min_value=0, step=1, key="wb_rf_depth")
        params["max_depth"] = None if depth == 0 else int(depth)
        params["min_samples_leaf"] = int(c3.number_input("Min samples leaf", value=1, min_value=1, step=1, key="wb_rf_leaf"))
        if mode == "Hyperparameter grid":
            with st.expander("Grid values", expanded=True):
                st.caption("Comma-separated values. Use 0 for no depth limit.")
                g1, g2, g3 = st.columns(3)
                raw_depths = _parse_grid(g2.text_input("max_depth values (0=none)", value=str(depth), key="wb_rf_depth_grid"), int)
                grid = {
                    "n_estimators": _parse_grid(g1.text_input("n_estimators values", value=str(params["n_estimators"]), key="wb_rf_trees_grid"), int),
                    "max_depth": [None if d == 0 else d for d in raw_depths],
                    "min_samples_leaf": _parse_grid(g3.text_input("min_samples_leaf values", value=str(params["min_samples_leaf"]), key="wb_rf_leaf_grid"), int),
                }
            return _expand_param_grid(params, grid)

    elif model_name in {"XGBoost", "LightGBM"}:
        c1, c2, c3 = st.columns(3)
        params["n_estimators"] = int(c1.number_input("Estimators", value=200, min_value=10, step=10, key=f"wb_{key}_n"))
        params["max_depth"] = int(c2.number_input("Max depth", value=4, min_value=-1, step=1, key=f"wb_{key}_depth"))
        params["learning_rate"] = float(c3.number_input("Learning rate", value=0.05, min_value=0.001, step=0.01, format="%.3f", key=f"wb_{key}_lr"))
        if mode == "Hyperparameter grid":
            with st.expander("Grid values", expanded=True):
                st.caption("Comma-separated values sweep multiple candidates.")
                g1, g2, g3 = st.columns(3)
                grid = {
                    "n_estimators": _parse_grid(g1.text_input("n_estimators values", value=str(params["n_estimators"]), key=f"wb_{key}_n_grid"), int),
                    "max_depth": _parse_grid(g2.text_input("max_depth values", value=str(params["max_depth"]), key=f"wb_{key}_depth_grid"), int),
                    "learning_rate": _parse_grid(g3.text_input("learning_rate values", value=str(params["learning_rate"]), key=f"wb_{key}_lr_grid"), float),
                }
            return _expand_param_grid(params, grid)

    elif model_name == "EBM":
        c1, c2 = st.columns(2)
        params["max_bins"] = int(c1.number_input("Max bins", value=32, min_value=16, step=16, key="wb_ebm_bins"))
        params["interactions"] = int(c2.number_input("Interactions", value=5, min_value=0, step=1, key="wb_ebm_interactions"))
        if mode == "Hyperparameter grid":
            with st.expander("Grid values", expanded=True):
                st.caption("Comma-separated values sweep multiple candidates.")
                g1, g2 = st.columns(2)
                grid = {
                    "max_bins": _parse_grid(g1.text_input("max_bins values", value=str(params["max_bins"]), key="wb_ebm_bins_grid"), int),
                    "interactions": _parse_grid(g2.text_input("interactions values", value=str(params["interactions"]), key="wb_ebm_interactions_grid"), int),
                }
            return _expand_param_grid(params, grid)

    elif model_name == "RuleFit":
        c1, c2 = st.columns(2)
        params["tree_size"] = int(c1.number_input("Tree size", value=4, min_value=2, step=1, key="wb_rulefit_tree"))
        params["max_rules"] = int(c2.number_input("Max rules", value=100, min_value=100, step=100, key="wb_rulefit_rules"))
        if find_spec("imodels") is None:
            st.caption("imodels is not installed; this run will be labelled RuleFit-style fallback.")
        else:
            st.caption("Uses imodels.RuleFitClassifier when compatible.")
        if mode == "Hyperparameter grid":
            with st.expander("Grid values", expanded=True):
                st.caption("Comma-separated values sweep multiple candidates.")
                g1, g2 = st.columns(2)
                grid = {
                    "tree_size": _parse_grid(g1.text_input("tree_size values", value=str(params["tree_size"]), key="wb_rulefit_tree_grid"), int),
                    "max_rules": _parse_grid(g2.text_input("max_rules values", value=str(params["max_rules"]), key="wb_rulefit_rules_grid"), int),
                }
            return _expand_param_grid(params, grid)

    return [params]


def _render_configs(selected: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    st.markdown("### 2. Configure parameters")
    all_models = [m for models in selected.values() for m in models]
    if not all_models:
        st.info("Select at least one available model to configure.")
        return {}
    param_map: dict[str, list[dict[str, Any]]] = {}
    tabs = st.tabs([_model_short(m) for m in all_models])
    for tab, model_name in zip(tabs, all_models):
        with tab:
            if model_name == "HUGIML":
                param_map[model_name] = _render_hugiml_config()
            else:
                param_map[model_name] = _render_generic_config(model_name)
    total = sum(len(v) for v in param_map.values())
    tuning_models = [m for m, configs in param_map.items() if len(configs) > 1]
    if tuning_models:
        st.caption(
            f"Prepared {total:,} candidate configuration(s) across {len(param_map)} model(s). "
            f"Hyperparameter grid active for: {', '.join(_model_short(m) for m in tuning_models)}. "
            f"Best config per model will be selected by CV ROC-AUC on held-out validation folds."
        )
    else:
        st.caption(f"Prepared {total:,} candidate configuration(s).")
    return param_map


def _leaderboard_frame(runs: list[dict[str, Any]]) -> pd.DataFrame:
    # Determine best run_id per model for tuning annotation
    best_per_model = _best_run_per_model(runs)
    best_run_ids = {r.get("run_id") for r in best_per_model.values()}

    rows = []
    for r in runs:
        params = r.get("params", {}) or {}
        run_id = r.get("run_id")
        rows.append({
            "run_id": run_id,
            "category": r.get("category"),
            "model": r.get("model"),
            "display": _run_display_name(r),
            "status": r.get("status"),
            "best_config": "✓" if run_id in best_run_ids else "",
            "cv_roc_auc": _scalar_metric(r.get("cv_roc_auc")),
            "roc_auc": _scalar_metric(r.get("roc_auc")),
            "f1": _scalar_metric(r.get("f1")),
            "precision": _scalar_metric(r.get("precision")),
            "recall": _scalar_metric(r.get("recall")),
            "accuracy": _scalar_metric(r.get("accuracy")),
            "fit_time_sec": _scalar_metric(r.get("fit_time_sec")),
            "params": params,
            "diagnostic": r.get("diagnostic"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        sort_col = "cv_roc_auc" if "cv_roc_auc" in df.columns else "roc_auc"
        df = df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    return df



def _best_run_per_model(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the best successful run per model, selected by cv_roc_auc on validation data.

    When a model was run with multiple hyperparameter configurations the candidate
    with the highest mean CV ROC-AUC (out-of-fold validation score) is selected as
    the winner. Single-config runs are returned as-is. Only successful runs
    (status == 'ok') are considered.
    """
    best: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run.get("status") != "ok":
            continue
        model = str(run.get("model", ""))
        score = _scalar_metric(run.get("cv_roc_auc"))
        prev = best.get(model)
        prev_score = _scalar_metric(prev.get("cv_roc_auc")) if prev else None
        if prev is None or (score is not None and (prev_score is None or score > prev_score)):
            best[model] = run
    return best


def _run_label(row: dict[str, Any]) -> str:
    model = _run_display_name(row)
    idx = row.get("config_index")
    # config_index is None for single-config runs (no suffix), an int for multi-config
    return f"{model} #{idx}" if idx is not None else model


def _successful_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in runs if r.get("status") == "ok" and isinstance(r.get("artifact"), dict)]


def _render_curve_comparison(runs: list[dict[str, Any]], y: np.ndarray) -> None:
    ok_runs = _successful_runs(runs)
    if not ok_runs:
        st.info("Curves are available after at least one successful run.")
        return

    plot_runs = ok_runs
    st.caption("Each curve uses out-of-fold predictions produced during the Workbench run.")

    roc_grid = np.linspace(0.0, 1.0, 101)
    pr_grid = np.linspace(0.0, 1.0, 101)
    roc_data: dict[str, np.ndarray] = {"FPR": roc_grid}
    pr_data: dict[str, np.ndarray] = {"Recall": pr_grid}

    for run in plot_runs:
        artifact = run.get("artifact") or {}
        y_proba = artifact.get("y_proba")
        if y_proba is None:
            continue
        # Use the clean model name (not run_id) for the legend
        label = _model_short(str(run.get("model", "unknown")))
        try:
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_data[label] = np.interp(roc_grid, fpr, tpr)
        except Exception:
            pass
        try:
            precision, recall, _ = precision_recall_curve(y, y_proba)
            order = np.argsort(recall)
            pr_data[label] = np.interp(pr_grid, recall[order], precision[order])
        except Exception:
            pass

    left, right = st.columns(2)
    with left:
        st.markdown("##### ROC comparison")
        roc_df = pd.DataFrame(roc_data).set_index("FPR")
        if roc_df.shape[1] > 0:
            st.line_chart(roc_df, width="stretch")
        else:
            st.info("No ROC curves could be drawn for the successful runs.")
    with right:
        st.markdown("##### Precision-recall comparison")
        pr_df = pd.DataFrame(pr_data).set_index("Recall")
        if pr_df.shape[1] > 0:
            st.line_chart(pr_df, width="stretch")
        else:
            st.info("No precision-recall curves could be drawn for the successful runs.")



def _render_ordered_feature_importance(fi: Any, limit: int = 20) -> None:
    if not isinstance(fi, pd.DataFrame) or fi.empty:
        st.info("No feature-importance artifact was exposed by this model.")
        return
    plot_df = fi.copy()
    plot_df["importance"] = pd.to_numeric(plot_df["importance"], errors="coerce").fillna(0.0)
    # Sort descending and take top-N
    plot_df = plot_df.sort_values("importance", ascending=False).head(limit).reset_index(drop=True)
    st.caption("Ordered by descending importance for quick scanning.")

    # Horizontal bar chart via matplotlib for a deterministically sorted plot.
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams.update({"figure.dpi": 120})

        # barh renders row 0 at the bottom and the last row at the top.
        # Sort ascending so the most important feature ends up at the top.
        display_df = plot_df.sort_values("importance", ascending=True).reset_index(drop=True)
        feature_col = "display_name" if "display_name" in display_df.columns else "feature"

        fig, ax = plt.subplots(figsize=(7, max(2.5, len(display_df) * 0.32)))
        ax.set_facecolor("none")
        ax.barh(
            display_df[feature_col].astype(str),
            display_df["importance"],
            color="#2563eb",
            height=0.65,
        )
        ax.set_xlabel("Importance", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color((0.5, 0.5, 0.5, 0.3))  # valid RGBA tuple
        ax.xaxis.grid(True, alpha=0.2, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    except Exception:
        # Fallback: use st.bar_chart with reversed-sorted index
        chart_df = plot_df.set_index("feature" if "feature" in plot_df.columns else plot_df.columns[0])["importance"]
        st.bar_chart(chart_df, width="stretch")

    with st.expander("Feature importance table", expanded=False):
        st.dataframe(dataframe_for_display(_safe_stringify_objects(plot_df)), width="stretch", hide_index=True)


def _coefficient_frame(model: Any, X: pd.DataFrame | None) -> pd.DataFrame:
    if not hasattr(model, "coef_") or X is None:
        return pd.DataFrame()
    coef = np.asarray(getattr(model, "coef_"), dtype=float)
    values = coef[0] if coef.ndim > 1 else coef
    cols = [str(c) for c in X.columns]
    n = min(len(cols), len(values))
    df = pd.DataFrame({"feature": cols[:n], "coefficient": values[:n]})
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def _safe_df(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if value is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(value)
    except Exception:
        return pd.DataFrame({"value": [str(value)]})


def _has_items(value: Any) -> bool:
    """Type-safe non-empty check for lists, arrays, Series, DataFrames, and dicts."""
    if value is None:
        return False
    if isinstance(value, pd.DataFrame):
        return not value.empty
    if isinstance(value, pd.Series):
        return not value.empty
    if isinstance(value, np.ndarray):
        return value.size > 0
    try:
        return len(value) > 0  # type: ignore[arg-type]
    except Exception:
        return bool(value)


def _safe_mapping(value: Any) -> dict[str, Any]:
    """Return a dict only when the object is safely mapping-like."""
    if isinstance(value, dict):
        return value
    try:
        if hasattr(value, "items"):
            return dict(value.items())
    except Exception:
        pass
    return {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        arr = np.asarray(value)
        if arr.size != 1:
            return default
        return int(arr.item())
    except Exception:
        try:
            return int(value)
        except Exception:
            return default


def _hugiml_patterns_frame(model: Any) -> pd.DataFrame:
    try:
        value = model.get_pattern_info()
        df = _safe_df(value)
        if not df.empty:
            return df
    except Exception:
        pass
    try:
        features = model.get_hug_features()
        if _has_items(features):
            return pd.DataFrame({"pattern": [str(x) for x in list(features)]})
    except Exception:
        pass
    for attr in ("pattern_labels_", "_pattern_labels_", "raw_patterns_", "patterns_"):
        try:
            labels = getattr(model, attr, None)
            if labels is not None:
                return pd.DataFrame({"pattern": [str(x) for x in list(labels)]})
        except Exception:
            continue
    return pd.DataFrame()


def _hugiml_features_frame(model: Any) -> pd.DataFrame:
    try:
        df = model.feature_importances()
        df = _safe_df(df)
        if not df.empty:
            if "importance" not in df.columns:
                if "abs_coefficient" in df.columns:
                    df["importance"] = pd.to_numeric(df["abs_coefficient"], errors="coerce")
                elif "coefficient" in df.columns:
                    df["importance"] = pd.to_numeric(df["coefficient"], errors="coerce").abs()
            if "feature_type" not in df.columns:
                feature_col = "feature" if "feature" in df.columns else "pattern" if "pattern" in df.columns else None
                if feature_col:
                    def _kind(x: Any) -> str:
                        text = str(x)
                        if text.startswith("orig:"):
                            return "original"
                        if text.startswith("augmented_pair:"):
                            return "augmented_pair"
                        if text.startswith("pattern:"):
                            return "pattern"
                        return "feature"
                    df["feature_type"] = df[feature_col].map(_kind)
            if "display_name" not in df.columns:
                if "feature" in df.columns:
                    df["display_name"] = df["feature"].astype(str).str.replace(r"^(orig:|pattern:|augmented_pair:)", "", regex=True)
                elif "pattern" in df.columns:
                    df["display_name"] = df["pattern"].astype(str)
            if "importance" in df.columns:
                df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
                df = df.sort_values("importance", ascending=False).reset_index(drop=True)
            return df
    except Exception:
        pass

    try:
        names = model.get_downstream_features()
        if _has_items(names):
            rows = []
            for name in list(names):
                text = str(name)
                if text.startswith("orig:"):
                    kind = "original"
                    display = text[len("orig:"):]
                elif text.startswith("augmented_pair:"):
                    kind = "augmented_pair"
                    display = text[len("augmented_pair:"):]
                elif text.startswith("pattern:"):
                    kind = "pattern"
                    display = text[len("pattern:"):]
                else:
                    kind = "feature"
                    display = text
                rows.append({"feature": text, "display_name": display, "feature_type": kind})
            return pd.DataFrame(rows)
    except Exception:
        pass
    return pd.DataFrame()


def _safe_stringify_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Convert list/dict/ndarray cells to strings before display.

    Only object-dtype columns are touched; numeric/bool/datetime columns are
    left intact.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(
                lambda v: str(v) if isinstance(v, (list, dict, np.ndarray)) else v
            )
    return df


def _filter_text_table(label: str, df: pd.DataFrame, key: str, *, columns: list[str] | None = None, limit: int = 300) -> None:
    if df.empty:
        st.info(f"No {label.lower()} artifact is available for this HUGIML run.")
        return
    query = st.text_input(f"Search {label.lower()}", value="", key=key, placeholder="Type to filter...")
    show_df = df.copy()
    if query:
        mask = show_df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        show_df = show_df.loc[mask]
    if columns:
        existing = [c for c in columns if c in show_df.columns]
        if existing:
            show_df = show_df[existing]
    st.caption(f"Showing {len(show_df):,} of {len(df):,} rows.")
    st.dataframe(dataframe_for_display(_safe_stringify_objects(show_df.head(limit))), width="stretch", hide_index=True)


def _render_hugiml_artifacts(model: Any, run_id: str = "hugiml", key_prefix: str = "") -> None:
    """Render fitted HUGIML artifacts without mixing RPTE stages."""
    patterns = _hugiml_patterns_frame(model)
    features = _hugiml_features_frame(model)

    composition: dict[str, Any] = {}
    try:
        composition = _safe_mapping(model.get_model_composition())
    except Exception:
        composition = {}
    counts = _safe_mapping(composition.get("downstream_feature_counts"))
    has_counts = _has_items(counts)

    ns = f"{key_prefix}{run_id}" if key_prefix else run_id
    rpte_rows = _get_rpte_rule_rows(model)
    flow = _get_rpte_feature_flow_audit(model)

    if flow:
        statement = str(flow.get("statement") or "")
        if statement:
            st.info(statement)

        rpte_rows = _get_rpte_final_term_rows(model, include_zero_direct=True)
        rpte_df = _rpte_rules_to_frame(rpte_rows)
        leaf_df = rpte_df.loc[rpte_df.get("is_leaf_term", False)].copy() if not rpte_df.empty else pd.DataFrame()
        direct_flag = rpte_df.get("is_direct_source_term", False)
        direct_df = rpte_df.loc[direct_flag].copy() if not rpte_df.empty else pd.DataFrame()
        final_count = int(flow.get("final_term_count", len(rpte_df)) or len(rpte_df))
        source_count = int(flow.get("source_feature_count", counts.get("total", len(features))) or 0)
        tree_count = int(leaf_df["tree"].nunique()) if not leaf_df.empty and leaf_df["tree"].notna().any() else 0

        metric_cols = st.columns(5)
        metric_cols[0].metric("Final LR terms", f"{final_count:,}")
        metric_cols[1].metric("RPTE leaf terms", f"{len(leaf_df):,}")
        metric_cols[2].metric("Direct source terms", f"{_direct_source_count(flow, len(direct_df)):,}")
        metric_cols[3].metric("RPTE trees", f"{tree_count:,}" if tree_count else "N/A")
        metric_cols[4].metric("RPTE source inputs", f"{source_count:,}")

        tabs = st.tabs(
            [
                "RPTE leaf trees",
                "Direct original features",
                "Direct HUG patterns",
                "Direct augmented pairs",
                "Representation summary",
            ],
            key=f"{ns}_hugiml_inner_tabs",
        )

        with tabs[0]:
            st.caption(
                "Leaf indicators are the root-to-leaf conjunction terms in the final LR. "
                "Internal split columns do not also appear as direct LR terms."
            )
            query = st.text_input(
                "Search RPTE leaf terms",
                value="",
                key=f"{ns}_hugiml_rpte_leaf_search",
                placeholder="condition, source feature, class, backend...",
            )
            effect_filter = st.selectbox(
                "Leaf effect direction",
                ["All", "Increases positive-class odds", "Decreases positive-class odds", "Neutral / zero"],
                key=f"{ns}_hugiml_rpte_leaf_effect",
            )
            show_leaf = leaf_df.copy()
            if query and not show_leaf.empty:
                mask = show_leaf.astype(str).apply(
                    lambda col: col.str.contains(query, case=False, na=False)
                ).any(axis=1)
                show_leaf = show_leaf.loc[mask]
            if effect_filter != "All" and not show_leaf.empty:
                show_leaf = show_leaf.loc[show_leaf["effect"].eq(effect_filter)]
            render_rpte_flat_tree_view(rpte_rows, show_leaf, key_prefix=f"{ns}_rpte_flat_tree")
            display_cols = [
                "class", "tree", "leaf", "effect", "coefficient", "odds_multiplier",
                "support_rate", "support_count", "n_conditions", "raw_sources", "backend",
            ]
            st.dataframe(
                dataframe_for_display(_safe_stringify_objects(show_leaf[[c for c in display_cols if c in show_leaf.columns]].head(300))),
                width="stretch",
                hide_index=True,
            )
            with st.expander("Inspect one leaf path condition by condition", expanded=False, key=f"{ns}_rpte_leaf_detail_exp"):
                if show_leaf.empty:
                    st.info("No leaf terms match the current filters.")
                else:
                    options = list(show_leaf.index)
                    pick = st.selectbox(
                        "Leaf term",
                        options,
                        format_func=lambda i: f"{show_leaf.loc[i, 'rule_id']} | coef={show_leaf.loc[i, 'coefficient']:.4f}",
                        key=f"{ns}_rpte_leaf_detail_pick",
                    )
                    selected = show_leaf.loc[pick]
                    st.markdown(f"**Effect:** {selected['effect']}")
                    st.markdown(
                        f"**Coefficient:** {selected['coefficient']:.6g} &nbsp; | &nbsp; "
                        f"**Odds multiplier:** {selected['odds_multiplier']:.6g}"
                    )
                    for line in str(selected["condition_steps"]).splitlines():
                        st.markdown(line)

        family_specs = [
            (tabs[1], "original", "direct original features", f"{ns}_direct_original_search"),
            (tabs[2], "pattern", "direct HUG patterns", f"{ns}_direct_pattern_search"),
            (tabs[3], "augmented_pair", "direct augmented pairs", f"{ns}_direct_augmented_search"),
        ]
        for tab, family, label, key in family_specs:
            with tab:
                family_df = direct_df.loc[direct_df.get("source_family", pd.Series("", index=direct_df.index)).eq(family)].copy()
                st.caption(
                    f"Only {label} that were not selected in any accepted RPTE split are shown. "
                    "These columns are appended directly to the final LR after the leaf indicators."
                )
                if family_df.empty:
                    st.info(f"No {label} are present in the final LR.")
                else:
                    _filter_text_table(
                        label,
                        family_df,
                        key=key,
                        columns=[
                            "class", "source_display_name", "source_column", "raw_sources",
                            "effect", "coefficient", "odds_multiplier", "backend",
                        ],
                    )

        with tabs[4]:
            family_counts = _direct_family_counts(flow)
            family_counts = family_counts if isinstance(family_counts, dict) else {}
            summary_rows = [
                {
                    "representation": "HUGIML source columns supplied to RPTE",
                    "count": source_count,
                    "role": "Candidate split columns",
                },
                {
                    "representation": "RPTE leaf indicators",
                    "count": len(leaf_df),
                    "role": "Final LR conjunction terms",
                },
                {
                    "representation": "Direct original columns",
                    "count": int(family_counts.get("original", (direct_df.get("source_family", pd.Series(dtype=str)) == "original").sum()) or 0),
                    "role": "Direct final LR terms",
                },
                {
                    "representation": "Direct HUG pattern columns",
                    "count": int(family_counts.get("pattern", (direct_df.get("source_family", pd.Series(dtype=str)) == "pattern").sum()) or 0),
                    "role": "Direct final LR terms",
                },
                {
                    "representation": "Direct augmented-pair columns",
                    "count": int(family_counts.get("augmented_pair", (direct_df.get("source_family", pd.Series(dtype=str)) == "augmented_pair").sum()) or 0),
                    "role": "Direct final LR terms",
                },
                {
                    "representation": "Final logistic-regression coefficients",
                    "count": final_count,
                    "role": "Leaf terms followed by direct source terms",
                },
            ]
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
            if _has_items(composition):
                with st.expander("Full model composition", expanded=False, key=f"{ns}_model_composition_exp"):
                    st.json(composition)
        return

    # Standard HUGIML logistic-regression path: source features are also the
    # downstream LR terms, so the original four-tab representation remains
    # appropriate.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patterns", f"{len(patterns):,}" if not patterns.empty else "—")
    c2.metric("Original features", f"{_safe_int(counts.get('original')):,}" if has_counts else "—")
    c3.metric("Augmented features", f"{_safe_int(counts.get('augmented_pair')):,}" if has_counts else "—")
    downstream_total = _safe_int(counts.get("total"), len(features)) if has_counts or not features.empty else None
    c4.metric("Downstream total", f"{downstream_total:,}" if downstream_total is not None else "—")

    tabs = st.tabs(
        ["Patterns", "Original features", "Augmented features", "All downstream features"],
        key=f"{ns}_hugiml_inner_tabs",
    )
    with tabs[0]:
        _filter_text_table(
            "Patterns", patterns, key=f"{ns}_hugiml_pattern_search",
            columns=[
                "pattern", "pattern_origin", "survivor_led", "survivor_features",
                "survivor_feature_count", "survivor_min_marginal_ig",
                "survivor_max_interaction_score", "survivor_best_partners",
                "utility", "information_gain", "support",
            ],
        )
    orig = features[features["feature_type"].astype(str).eq("original")] if not features.empty and "feature_type" in features.columns else pd.DataFrame()
    with tabs[1]:
        _filter_text_table(
            "Original features", orig, key=f"{ns}_hugiml_orig_search",
            columns=["display_name", "feature", "importance", "coefficient", "abs_coefficient", "non_missing_rate", "variance"],
        )
    aug = features[features["feature_type"].astype(str).eq("augmented_pair")] if not features.empty and "feature_type" in features.columns else pd.DataFrame()
    with tabs[2]:
        _filter_text_table(
            "Augmented features", aug, key=f"{ns}_hugiml_aug_search",
            columns=["display_name", "feature", "importance", "coefficient", "abs_coefficient", "raw_formula", "operation", "inputs", "risk_increases_when", "raw_interpretation"],
        )
    with tabs[3]:
        _filter_text_table(
            "Downstream features", features, key=f"{ns}_hugiml_all_search",
            columns=[
                "feature_type", "display_name", "feature", "pattern_origin", "survivor_led",
                "survivor_features", "survivor_feature_count", "importance", "coefficient",
                "abs_coefficient", "support", "non_missing_rate", "variance",
            ],
        )
        if _has_items(composition):
            with st.expander("Model composition", expanded=False, key=f"{ns}_model_composition_exp"):
                st.json(composition)

def _render_lr_artifacts(model: Any, X: pd.DataFrame | None, key_prefix: str = "") -> None:
    coef_df = _coefficient_frame(model, X)
    if coef_df.empty:
        st.info("No coefficient artifact is available for this LR run.")
        return
    st.markdown("#### LR coefficients")
    st.caption("Signed coefficients show direction; absolute values are used for ranking.")
    st.dataframe(dataframe_for_display(coef_df.head(30)), width="stretch", hide_index=True)


def _render_dt_artifacts(model: Any, X: pd.DataFrame | None, key_prefix: str = "") -> None:
    if X is None:
        st.info("No feature frame is available for tree inspection.")
        return
    try:
        text_tree = export_text(model, feature_names=[str(c) for c in X.columns], max_depth=4)
    except Exception as exc:
        st.info(f"Could not export the tree text: {exc}")
        return
    st.markdown("#### Decision tree excerpt")
    st.caption("Top levels only; increase max depth in configuration if needed, then rerun.")
    st.code(text_tree, language="text")


def _ebm_terms_frame(model: Any) -> pd.DataFrame:
    names = list(map(str, getattr(model, "term_names_", [])))
    values: list[float] = []
    try:
        values = list(map(float, model.term_importances()))
    except Exception:
        values = []
    n = min(len(names), len(values))
    if n == 0:
        return pd.DataFrame()
    df = pd.DataFrame({"term": names[:n], "importance": values[:n]})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _render_ebm_artifacts(model: Any, key_prefix: str = "") -> None:
    terms = _ebm_terms_frame(model)
    if terms.empty:
        st.info("No EBM term artifact is available from this package/version.")
        return
    st.markdown("#### EBM terms")
    st.bar_chart(terms.head(20).set_index("term")["importance"], width="stretch")
    with st.expander("EBM term table", expanded=False, key=f"{key_prefix}ebm_term_table_exp"):
        st.dataframe(dataframe_for_display(terms), width="stretch", hide_index=True)


def _rulefit_rules_frame(model: Any, max_rows: int | None = 50) -> pd.DataFrame:
    """Best-effort RuleFit rule inventory across imodels, legacy rulefit, and fallback."""
    def _limit(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.head(max_rows).reset_index(drop=True) if max_rows is not None else df.reset_index(drop=True)

    # imodels commonly exposes a get_rules() table with rule, coef, support, importance.
    try:
        get_rules = getattr(model, "get_rules", None)
        if callable(get_rules):
            df = _safe_df(get_rules())
            if not df.empty:
                return _limit(df)
    except Exception:
        pass

    # imodels exposes rules through different attributes across versions.
    for attr in ("rules_", "rule_ensemble", "rule_ensemble_"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        try:
            if isinstance(obj, pd.DataFrame):
                return _limit(obj.copy())
            if hasattr(obj, "rules"):
                rules = getattr(obj, "rules")
                return _limit(pd.DataFrame({"rule": [str(r) for r in list(rules)]}))
            if isinstance(obj, (list, tuple)):
                return _limit(pd.DataFrame({"rule": [str(r) for r in obj]}))
        except Exception:
            continue
    wrapped = getattr(model, "model_", None)
    if wrapped is not None and wrapped is not model:
        return _rulefit_rules_frame(wrapped, max_rows=max_rows)
    if getattr(model, "backend_", "") == "sklearn-rulefit-style":
        encoder = getattr(model, "rule_encoder_", None)
        categories = getattr(encoder, "categories_", None)
        try:
            if categories is not None and len(categories):
                rules = [f"tree_leaf:{x}" for x in list(categories[0])]
                return _limit(pd.DataFrame({"rule": rules, "artifact": "boosted-tree leaf indicator"}))
        except Exception:
            pass
        return pd.DataFrame({
            "artifact": ["Rule indicators generated from boosted-tree leaves"],
            "count": [int(getattr(getattr(model, "rule_encoder_", None), "categories_", [[]])[0].size) if hasattr(getattr(model, "rule_encoder_", None), "categories_") else None],
        })
    return pd.DataFrame()


def _render_rulefit_artifacts(model: Any, implementation: str | None, key_prefix: str = "") -> None:
    if implementation == "sklearn-rulefit-style":
        st.info("This is labelled as RuleFit-style fallback, not official imodels.RuleFitClassifier.")
    elif implementation == "imodels-rulefit":
        st.caption("Implementation: imodels.RuleFitClassifier")
    rules = _rulefit_rules_frame(model)
    if rules.empty:
        st.info("No rule table was exposed by this RuleFit implementation.")
        return
    st.markdown("#### RuleFit rules / rule artifacts")
    st.dataframe(dataframe_for_display(rules), width="stretch", hide_index=True)


def _render_interpretable_artifacts(selected_run: dict[str, Any], key_prefix: str = "") -> None:
    """Render model-specific interpretability artifacts.

    ``key_prefix`` scopes all Streamlit widget keys within this call so that
    multiple tabs can render the same run without widget ID conflicts.
    """
    artifact = selected_run.get("artifact") if isinstance(selected_run.get("artifact"), dict) else {}
    model = artifact.get("model") if isinstance(artifact, dict) else None
    X_model = artifact.get("feature_frame") if isinstance(artifact, dict) else None
    model_name = selected_run.get("model")
    implementation = artifact.get("implementation") if isinstance(artifact, dict) else None

    if model is None:
        st.info("No fitted model artifact is available for this run.")
        return
    if model_name == "HUGIML":
        st.markdown("#### HUGIML interpretability artifacts")
        _render_hugiml_artifacts(model, run_id=str(selected_run.get("run_id", "hugiml")), key_prefix=key_prefix)
    elif model_name == "Logistic Regression":
        _render_lr_artifacts(model, X_model, key_prefix=key_prefix)
    elif model_name == "Decision Tree":
        _render_dt_artifacts(model, X_model, key_prefix=key_prefix)
    elif model_name == "EBM":
        _render_ebm_artifacts(model, key_prefix=key_prefix)
    elif model_name == "RuleFit":
        _render_rulefit_artifacts(model, implementation, key_prefix=key_prefix)
    else:
        st.info("This model exposes generic feature-importance artifacts only in the Workbench.")


def _pattern_compare_key(row: pd.Series) -> str:
    for col in ("pattern", "display_name", "feature", "name"):
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return " | ".join(str(x) for x in row.astype(str).tolist())


def _normalised_pattern_inventory(run: dict[str, Any]) -> pd.DataFrame:
    artifact = run.get("artifact") if isinstance(run.get("artifact"), dict) else {}
    model = artifact.get("model") if isinstance(artifact, dict) else None
    if model is None:
        return pd.DataFrame()
    df = _hugiml_patterns_frame(model)
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["pattern_key"] = out.apply(_pattern_compare_key, axis=1)
    out = out.drop_duplicates("pattern_key").reset_index(drop=True)
    return out



def _artifact_run_label(run_id: str, runs_by_id: dict[str, dict[str, Any]]) -> str:
    run = runs_by_id.get(str(run_id), {})
    summary = _short_config_summary(run)
    return f"{run_id} — {summary}" if summary else str(run_id)


def _successful_model_runs(runs: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    return [r for r in _successful_runs(runs) if r.get("model") == model_name]


def _select_base_alt_runs(model_runs: list[dict[str, Any]], *, key_prefix: str, model_label: str) -> tuple[dict[str, Any], dict[str, Any]] | tuple[None, None]:
    if len(model_runs) < 2:
        st.info(f"Run at least two successful {model_label} configurations to compare artifacts.")
        return None, None
    run_ids = [str(r.get("run_id")) for r in model_runs]
    runs_by_id = {str(r.get("run_id")): r for r in model_runs}
    c1, c2 = st.columns(2)
    base_id = c1.selectbox(
        f"Base {model_label} run",
        run_ids,
        index=0,
        key=f"{key_prefix}_base",
        format_func=lambda rid: _artifact_run_label(str(rid), runs_by_id),
        on_change=_request_results_pattern_delta_tab,
    )
    alt_id = c2.selectbox(
        f"Alternate {model_label} run",
        run_ids,
        index=1 if len(run_ids) > 1 else 0,
        key=f"{key_prefix}_alt",
        format_func=lambda rid: _artifact_run_label(str(rid), runs_by_id),
        on_change=_request_results_pattern_delta_tab,
    )
    return runs_by_id[str(base_id)], runs_by_id[str(alt_id)]


def _render_selected_config_pair(base_run: dict[str, Any], alt_run: dict[str, Any]) -> None:
    """Show selected configs as field rows with Base/Alternate value columns."""
    base_df = _run_config_frame(base_run, "Base")
    alt_df = _run_config_frame(alt_run, "Alternate")
    cfg_df = pd.concat([base_df, alt_df], ignore_index=True)
    if cfg_df.empty or not {"selection", "field", "value"}.issubset(cfg_df.columns):
        return
    cfg_df = cfg_df.copy()
    cfg_df["field"] = cfg_df["field"].astype(str)
    cfg_wide = (
        cfg_df.pivot_table(
            index="field",
            columns="selection",
            values="value",
            aggfunc=lambda values: next((v for v in values if pd.notna(v)), ""),
        )
        .reset_index()
    )
    for col in ("Base", "Alternate"):
        if col not in cfg_wide.columns:
            cfg_wide[col] = ""
    preferred_order = [
        "run_id", "model", "mode", "cv_roc_auc", "f1", "candidate_count", "fast_path_used",
        "adaptive_binning", "adaptive_binning_sample_frac", "adaptive_binning_sample_random_state",
        "B", "L", "topK", "G", "feature_mode", "topk_budget_strict",
        "augmented_pair_transforms", "augmented_pair_mode", "aug_feature_size",
        "max_pair_features", "ii_partner_size", "interaction_relaxed_mining",
        "interaction_relaxed_feature_size",
        "C", "max_iter", "max_depth", "min_samples_leaf", "tree_size", "max_rules",
        "n_estimators", "learning_rate", "max_bins", "interactions",
    ]
    order_map = {field: idx for idx, field in enumerate(preferred_order)}
    cfg_wide["__order"] = cfg_wide["field"].map(lambda x: order_map.get(str(x), len(order_map)))
    cfg_wide = cfg_wide.sort_values(["__order", "field"]).drop(columns="__order")
    cfg_wide = cfg_wide[["field", "Base", "Alternate"]]
    with st.expander("Selected run configurations", expanded=True):
        st.dataframe(dataframe_for_display(_safe_stringify_objects(cfg_wide)), width="stretch", hide_index=True)


def _rule_compare_key(row: pd.Series) -> str:
    for col in ("rule", "term", "feature", "name", "artifact"):
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return " | ".join(str(x) for x in row.astype(str).tolist())


def _normalised_rule_inventory(run: dict[str, Any]) -> pd.DataFrame:
    artifact = run.get("artifact") if isinstance(run.get("artifact"), dict) else {}
    model = artifact.get("model") if isinstance(artifact, dict) else None
    if model is None:
        return pd.DataFrame()
    df = _rulefit_rules_frame(model, max_rows=None)
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["rule_key"] = out.apply(_rule_compare_key, axis=1)
    out = out.drop_duplicates("rule_key").reset_index(drop=True)
    return out


def _render_rulefit_rule_delta(runs: list[dict[str, Any]]) -> None:
    rule_runs = _successful_model_runs(runs, "RuleFit")
    base_run, alt_run = _select_base_alt_runs(rule_runs, key_prefix="wb_rulefit_delta", model_label="RuleFit")
    if base_run is None or alt_run is None:
        return

    _render_selected_config_pair(base_run, alt_run)
    base_rules = _normalised_rule_inventory(base_run)
    alt_rules = _normalised_rule_inventory(alt_run)
    if base_rules.empty and alt_rules.empty:
        st.info("Neither selected RuleFit run exposes a rule inventory.")
        return

    base_keys = set(base_rules.get("term_key", pd.Series(dtype=str)).astype(str))
    alt_keys = set(alt_rules.get("term_key", pd.Series(dtype=str)).astype(str))
    added = sorted(alt_keys - base_keys)
    removed = sorted(base_keys - alt_keys)
    unchanged = len(base_keys & alt_keys)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Base final terms", f"{len(base_keys):,}")
    m2.metric("Alternate final terms", f"{len(alt_keys):,}")
    m3.metric("+ Added", f"{len(added):,}")
    m4.metric("− Removed", f"{len(removed):,}")
    st.caption(f"Unchanged rules: {unchanged:,}. Added means present in alternate but not base; removed means present in base but not alternate.")

    def _rows(keys: list[str], status: str, sign: str, source: pd.DataFrame) -> list[dict[str, Any]]:
        source = source.set_index("rule_key", drop=False) if not source.empty and "rule_key" in source.columns else pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for key in keys:
            row = {"change": sign, "status": status, "final_term": key}
            if not source.empty and key in source.index:
                src = source.loc[key]
                if isinstance(src, pd.DataFrame):
                    src = src.iloc[0]
                for col in ("coef", "coefficient", "support", "importance", "type"):
                    if col in src:
                        row[col] = src[col]
            rows.append(row)
        return rows

    base_inventory = _rows(sorted(base_keys), "base", "", base_rules)
    delta = pd.DataFrame(_rows(added, "added", "+", alt_rules) + _rows(removed, "removed", "−", base_rules))
    all_view = pd.DataFrame(base_inventory + _rows(added, "added", "+", alt_rules) + _rows(removed, "removed", "−", base_rules))
    if delta.empty:
        st.success("No rule differences: the selected RuleFit runs expose the same rule set. Showing the base run rules below.")
    change_filter = st.radio("Show", ["All", "+ Added", "− Removed"], horizontal=True, key="wb_rulefit_delta_filter")
    if change_filter == "+ Added":
        show_df = delta[delta["status"].eq("added")].copy() if not delta.empty else pd.DataFrame()
    elif change_filter == "− Removed":
        show_df = delta[delta["status"].eq("removed")].copy() if not delta.empty else pd.DataFrame()
    else:
        show_df = all_view.copy()
    query = st.text_input("Search RuleFit rule delta", value="", key="wb_rulefit_delta_search", placeholder="Type to filter rule text...")
    if query:
        show_df = show_df[show_df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)]
    st.dataframe(dataframe_for_display(_safe_stringify_objects(show_df)), width="stretch", hide_index=True)


def _run_model_and_frame(run: dict[str, Any]) -> tuple[Any, pd.DataFrame | None]:
    artifact = run.get("artifact") if isinstance(run.get("artifact"), dict) else {}
    return artifact.get("model") if isinstance(artifact, dict) else None, artifact.get("feature_frame") if isinstance(artifact, dict) else None


def _render_lr_coefficient_comparison(runs: list[dict[str, Any]]) -> None:
    lr_runs = _successful_model_runs(runs, "Logistic Regression")
    base_run, alt_run = _select_base_alt_runs(lr_runs, key_prefix="wb_lr_compare", model_label="LR")
    if base_run is None or alt_run is None:
        return
    _render_selected_config_pair(base_run, alt_run)

    base_model, base_X = _run_model_and_frame(base_run)
    alt_model, alt_X = _run_model_and_frame(alt_run)
    base_df = _coefficient_frame(base_model, base_X).rename(columns={"coefficient": "base_coefficient", "abs_coefficient": "base_abs_coefficient"})
    alt_df = _coefficient_frame(alt_model, alt_X).rename(columns={"coefficient": "alternate_coefficient", "abs_coefficient": "alternate_abs_coefficient"})
    if base_df.empty and alt_df.empty:
        st.info("Neither selected LR run exposes coefficients.")
        return
    keep_base = [c for c in ["feature", "base_coefficient", "base_abs_coefficient"] if c in base_df.columns]
    keep_alt = [c for c in ["feature", "alternate_coefficient", "alternate_abs_coefficient"] if c in alt_df.columns]
    merged = pd.merge(base_df[keep_base], alt_df[keep_alt], on="feature", how="outer")
    if "base_coefficient" in merged.columns and "alternate_coefficient" in merged.columns:
        merged["coefficient_delta"] = merged["alternate_coefficient"] - merged["base_coefficient"]
        merged["abs_delta"] = merged["coefficient_delta"].abs()
        merged = merged.sort_values("abs_delta", ascending=False, na_position="last")
    query = st.text_input("Search LR coefficients", value="", key="wb_lr_compare_search", placeholder="Type to filter feature names...")
    show_df = merged
    if query:
        show_df = show_df[show_df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)]
    st.dataframe(dataframe_for_display(_safe_stringify_objects(show_df)), width="stretch", hide_index=True)


def _decision_tree_text(run: dict[str, Any], max_depth: int = 10) -> str:
    model, X_model = _run_model_and_frame(run)
    if model is None or X_model is None:
        return ""
    try:
        return export_text(model, feature_names=[str(c) for c in X_model.columns], max_depth=max_depth)
    except Exception as exc:
        return f"Could not export tree text: {exc}"


def _render_dt_tree_comparison(runs: list[dict[str, Any]]) -> None:
    dt_runs = _successful_model_runs(runs, "Decision Tree")
    base_run, alt_run = _select_base_alt_runs(dt_runs, key_prefix="wb_dt_compare", model_label="DT")
    if base_run is None or alt_run is None:
        return
    _render_selected_config_pair(base_run, alt_run)
    depth = int(st.number_input("Tree text max depth", value=10, min_value=1, max_value=50, step=1, key="wb_dt_compare_text_depth"))
    base_text = _decision_tree_text(base_run, max_depth=depth)
    alt_text = _decision_tree_text(alt_run, max_depth=depth)
    table = pd.DataFrame([
        {"selection": "Base", "run_id": base_run.get("run_id"), "tree_text": base_text},
        {"selection": "Alternate", "run_id": alt_run.get("run_id"), "tree_text": alt_text},
    ])
    st.dataframe(dataframe_for_display(_safe_stringify_objects(table)), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Base tree")
        st.code(base_text or "No tree text available.", language="text")
    with c2:
        st.markdown("#### Alternate tree")
        st.code(alt_text or "No tree text available.", language="text")


def _render_artifact_comparison(runs: list[dict[str, Any]]) -> None:
    choices: list[tuple[str, str]] = []
    model_counts = {name: len(_successful_model_runs(runs, name)) for name in ("HUGIML", "RuleFit", "Logistic Regression", "Decision Tree")}
    if model_counts.get("HUGIML", 0) >= 2:
        choices.append(("HUGIML patterns", "HUGIML"))
    if model_counts.get("RuleFit", 0) >= 2:
        choices.append(("RuleFit rules", "RuleFit"))
    if model_counts.get("Logistic Regression", 0) >= 2:
        choices.append(("LR coefficients", "Logistic Regression"))
    if model_counts.get("Decision Tree", 0) >= 2:
        choices.append(("DT text tree", "Decision Tree"))
    if len(_hugiml_runs_using_rpte(runs)) >= 2:
        choices.append(("RPTE rule conjunctions", "RPTE"))

    if not choices:
        st.info("Run at least two successful configurations for HUGIML, RuleFit, LR, or DT to compare their interpretable artifacts.")
        return

    labels = [label for label, _ in choices]
    selected_label = st.selectbox(
        "Artifact comparison",
        labels,
        index=0,
        key="wb_artifact_compare_model",
        on_change=_request_results_pattern_delta_tab,
    )
    selected_model = dict(choices)[selected_label]
    if selected_model == "HUGIML":
        st.caption("Compare mined pattern inventories. `+` means added in the alternate run; `−` means removed relative to the base run.")
        _render_hugiml_pattern_delta(runs)
    elif selected_model == "RuleFit":
        st.caption("Compare RuleFit rule inventories. `+` means added in the alternate run; `−` means removed relative to the base run.")
        _render_rulefit_rule_delta(runs)
    elif selected_model == "Logistic Regression":
        st.caption("Compare LR coefficients for two configurations. The delta is alternate minus base.")
        _render_lr_coefficient_comparison(runs)
    elif selected_model == "Decision Tree":
        st.caption("Compare exported decision-tree text for two configurations.")
        _render_dt_tree_comparison(runs)
    elif selected_model == "RPTE":
        st.caption(
            "Compare RPTE rule conjunctions for two HUGIML runs whose downstream estimator is "
            "RPTE. `+` means added in the alternate run; `−` means removed relative to the base run."
        )
        _render_rpte_rule_delta(runs)


def _render_hugiml_pattern_delta(runs: list[dict[str, Any]]) -> None:
    hug_runs = [r for r in _successful_runs(runs) if r.get("model") == "HUGIML"]
    if len(hug_runs) < 2:
        st.info("Run at least two successful HUGIML configurations to compare pattern additions and removals.")
        return

    run_ids = [str(r.get("run_id")) for r in hug_runs]
    runs_by_id = {str(r.get("run_id")): r for r in hug_runs}
    c1, c2 = st.columns(2)
    base_id = c1.selectbox(
        "Base HUGIML run",
        run_ids,
        index=0,
        key="wb_pattern_delta_base",
        format_func=lambda rid: _pattern_delta_run_label(str(rid), runs_by_id),
        on_change=_request_results_pattern_delta_tab,
    )
    alt_default = 1 if len(run_ids) > 1 else 0
    alt_id = c2.selectbox(
        "Alternate HUGIML run",
        run_ids,
        index=alt_default,
        key="wb_pattern_delta_alt",
        format_func=lambda rid: _pattern_delta_run_label(str(rid), runs_by_id),
        on_change=_request_results_pattern_delta_tab,
    )

    base_run = next(r for r in hug_runs if str(r.get("run_id")) == base_id)
    alt_run = next(r for r in hug_runs if str(r.get("run_id")) == alt_id)
    _render_selected_config_pair(base_run, alt_run)

    base_patterns = _normalised_pattern_inventory(base_run)
    alt_patterns = _normalised_pattern_inventory(alt_run)

    if base_patterns.empty and alt_patterns.empty:
        st.info("Neither selected HUGIML run exposes a pattern inventory.")
        return

    base_keys = set(base_patterns.get("pattern_key", pd.Series(dtype=str)).astype(str))
    alt_keys = set(alt_patterns.get("pattern_key", pd.Series(dtype=str)).astype(str))
    added = sorted(alt_keys - base_keys)
    removed = sorted(base_keys - alt_keys)
    unchanged = len(base_keys & alt_keys)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Base patterns", f"{len(base_keys):,}")
    m2.metric("Alternate patterns", f"{len(alt_keys):,}")
    m3.metric("+ Added", f"{len(added):,}")
    m4.metric("− Removed", f"{len(removed):,}")
    st.caption(f"Unchanged patterns: {unchanged:,}. Added means present in alternate but not base; removed means present in base but not alternate.")

    def _rows(keys: list[str], status: str, sign: str, source: pd.DataFrame) -> list[dict[str, Any]]:
        source = source.set_index("pattern_key", drop=False) if not source.empty and "pattern_key" in source.columns else pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for key in keys:
            row = {"change": sign, "status": status, "pattern": key}
            if not source.empty and key in source.index:
                src = source.loc[key]
                if isinstance(src, pd.DataFrame):
                    src = src.iloc[0]
                for col in ("utility", "information_gain", "support"):
                    if col in src:
                        row[col] = src[col]
            rows.append(row)
        return rows

    base_inventory = _rows(sorted(base_keys), "base", "", base_patterns)
    delta = pd.DataFrame(
        _rows(added, "added", "+", alt_patterns)
        + _rows(removed, "removed", "−", base_patterns)
    )
    all_view = pd.DataFrame(
        base_inventory
        + _rows(added, "added", "+", alt_patterns)
        + _rows(removed, "removed", "−", base_patterns)
    )
    if delta.empty:
        st.success("No pattern differences: the selected HUGIML runs expose the same pattern set. Showing the base run patterns below.")

    change_filter = st.radio("Show", ["All", "+ Added", "− Removed"], horizontal=True, key="wb_pattern_delta_filter")
    if change_filter == "+ Added":
        show_df = delta[delta["status"].eq("added")].copy() if not delta.empty else pd.DataFrame()
    elif change_filter == "− Removed":
        show_df = delta[delta["status"].eq("removed")].copy() if not delta.empty else pd.DataFrame()
    else:
        show_df = all_view.copy()

    query = st.text_input("Search pattern delta", value="", key="wb_pattern_delta_search", placeholder="Type to filter pattern text...")
    if query:
        show_df = show_df[show_df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)]
    st.dataframe(dataframe_for_display(_safe_stringify_objects(show_df)), width="stretch", hide_index=True)


def _hugiml_runs_using_rpte(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Successful HUGIML runs whose downstream estimator is RPTE-based
    (i.e. rpte_rule_table() returns at least one row)."""
    out = []
    for run in _successful_runs(runs):
        if run.get("model") != "HUGIML":
            continue
        artifact = run.get("artifact") if isinstance(run.get("artifact"), dict) else {}
        model = artifact.get("model") if isinstance(artifact, dict) else None
        if model is not None and _get_rpte_rule_rows(model):
            out.append(run)
    return out


def _normalised_rpte_rule_inventory(run: dict[str, Any]) -> pd.DataFrame:
    artifact = run.get("artifact") if isinstance(run.get("artifact"), dict) else {}
    model = artifact.get("model") if isinstance(artifact, dict) else None
    if model is None:
        return pd.DataFrame()
    rows = _get_rpte_final_term_rows(model, include_zero_direct=True)
    if not rows:
        return pd.DataFrame()
    df = _rpte_rules_to_frame(rows)
    leaf_key = (
        df["class"].astype(str) + " | leaf | tree " + df["tree"].astype(str)
        + " | leaf " + df["leaf"].astype(str) + " | " + df["conjunction"].astype(str)
    )
    direct_key = (
        df["class"].astype(str) + " | " + df["term_type"].astype(str)
        + " | " + df["source_column"].astype(str)
    )
    df["term_key"] = np.where(df["is_leaf_term"], leaf_key, direct_key)
    return df


def _render_rpte_rule_delta(runs: list[dict[str, Any]]) -> None:
    rpte_runs = _hugiml_runs_using_rpte(runs)
    if len(rpte_runs) < 2:
        st.info("Run at least two successful HUGIML configurations with an RPTE downstream estimator to compare rule additions and removals.")
        return

    run_ids = [str(r.get("run_id")) for r in rpte_runs]
    runs_by_id = {str(r.get("run_id")): r for r in rpte_runs}
    c1, c2 = st.columns(2)
    base_id = c1.selectbox(
        "Base RPTE run", run_ids, index=0, key="wb_rpte_delta_base",
        format_func=lambda rid: _pattern_delta_run_label(str(rid), runs_by_id),
        on_change=_request_results_pattern_delta_tab,
    )
    alt_default = 1 if len(run_ids) > 1 else 0
    alt_id = c2.selectbox(
        "Alternate RPTE run", run_ids, index=alt_default, key="wb_rpte_delta_alt",
        format_func=lambda rid: _pattern_delta_run_label(str(rid), runs_by_id),
        on_change=_request_results_pattern_delta_tab,
    )

    base_run = next(r for r in rpte_runs if str(r.get("run_id")) == base_id)
    alt_run = next(r for r in rpte_runs if str(r.get("run_id")) == alt_id)
    _render_selected_config_pair(base_run, alt_run)

    base_rules = _normalised_rpte_rule_inventory(base_run)
    alt_rules = _normalised_rpte_rule_inventory(alt_run)
    if base_rules.empty and alt_rules.empty:
        st.info("Neither selected run exposes RPTE final-LR terms.")
        return

    base_keys = set(base_rules.get("term_key", pd.Series(dtype=str)).astype(str))
    alt_keys = set(alt_rules.get("term_key", pd.Series(dtype=str)).astype(str))
    added = sorted(alt_keys - base_keys)
    removed = sorted(base_keys - alt_keys)
    unchanged = len(base_keys & alt_keys)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Base final terms", f"{len(base_keys):,}")
    m2.metric("Alternate final terms", f"{len(alt_keys):,}")
    m3.metric("+ Added", f"{len(added):,}")
    m4.metric("− Removed", f"{len(removed):,}")
    st.caption(
        f"Unchanged final terms: {unchanged:,}. Leaf identities use their conjunctions; direct "
        "source terms use their source-column identities. Coefficient changes are shown separately "
        "from additions and removals."
    )

    def _rows(keys: list[str], status: str, sign: str, source: pd.DataFrame) -> list[dict[str, Any]]:
        source_idx = source.set_index("term_key", drop=False) if not source.empty and "term_key" in source.columns else pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for key in keys:
            row = {"change": sign, "status": status, "final_term": key}
            if not source_idx.empty and key in source_idx.index:
                src = source_idx.loc[key]
                if isinstance(src, pd.DataFrame):
                    src = src.iloc[0]
                for col in ("term_type", "source_family", "backend", "n_conditions", "support_count", "coefficient"):
                    if col in src:
                        row[col] = src[col]
            rows.append(row)
        return rows

    base_inventory = _rows(sorted(base_keys), "base", "", base_rules)
    delta = pd.DataFrame(_rows(added, "added", "+", alt_rules) + _rows(removed, "removed", "−", base_rules))
    all_view = pd.DataFrame(base_inventory + _rows(added, "added", "+", alt_rules) + _rows(removed, "removed", "−", base_rules))
    if delta.empty:
        st.success("No final-term differences: the selected RPTE runs expose the same leaf and direct-source term set. Showing the base run below.")

    change_filter = st.radio("Show", ["All", "+ Added", "− Removed"], horizontal=True, key="wb_rpte_delta_filter")
    if change_filter == "+ Added":
        show_df = delta[delta["status"].eq("added")].copy() if not delta.empty else pd.DataFrame()
    elif change_filter == "− Removed":
        show_df = delta[delta["status"].eq("removed")].copy() if not delta.empty else pd.DataFrame()
    else:
        show_df = all_view.copy()

    query = st.text_input("Search final-term delta", value="", key="wb_rpte_delta_search", placeholder="Type to filter leaf or source-term text...")
    if query:
        show_df = show_df[show_df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)]
    st.dataframe(dataframe_for_display(_safe_stringify_objects(show_df)), width="stretch", hide_index=True)


def _render_interpretability_comparison(runs: list[dict[str, Any]]) -> None:
    ok_runs = _successful_runs(runs)
    interpretable = [r for r in ok_runs if r.get("model") in {"HUGIML", "Logistic Regression", "Decision Tree", "EBM", "RuleFit"}]
    if not interpretable:
        st.info("No interpretable artifacts are available yet. Run HUGIML, LR, DT, EBM, or RuleFit.")
        return

    rows = []
    for run in interpretable:
        artifact = run.get("artifact") if isinstance(run.get("artifact"), dict) else {}
        fi = artifact.get("feature_importance") if isinstance(artifact, dict) else None
        model_name = run.get("model")
        impl = artifact.get("implementation") if isinstance(artifact, dict) else ""
        if model_name == "RuleFit" and impl == "sklearn-rulefit-style":
            artifact_type = "RuleFit-style fallback rules"
        elif model_name == "RuleFit":
            artifact_type = "RuleFit rules"
        elif model_name == "EBM":
            artifact_type = "EBM terms"
        elif model_name == "HUGIML":
            rpte_model = artifact.get("model") if isinstance(artifact, dict) else None
            artifact_type = (
                "RPTE leaf terms plus direct original/pattern/augmented LR terms"
                if rpte_model is not None and _get_rpte_feature_flow_audit(rpte_model)
                else "Patterns, original features, augmented features"
            )
        elif model_name == "Logistic Regression":
            artifact_type = "LR coefficients"
        elif model_name == "Decision Tree":
            artifact_type = "Tree excerpt"
        else:
            artifact_type = "Feature importance"
        rows.append({
            "run_id": run.get("run_id"),
            "model": _run_display_name(run),
            "artifact": artifact_type,
            "top_features_available": bool(isinstance(fi, pd.DataFrame) and not fi.empty),
            "cv_roc_auc": run.get("cv_roc_auc"),
        })

    st.dataframe(dataframe_for_display(pd.DataFrame(rows)), width="stretch", hide_index=True)

    # Each model's artifacts in a collapsible expander; first open by default.
    for run in interpretable:
        label = f"{_run_display_name(run)} — {run.get('run_id', '')}"
        # First model open by default, rest collapsed
        expanded = (run is interpretable[0])
        with st.expander(label, expanded=expanded):
            _render_interpretable_artifacts(run, key_prefix=f"interp_{run.get('run_id', '')}_")

def _render_selected_run_details(selected_run: dict[str, Any], y: np.ndarray) -> None:
    artifact = selected_run.get("artifact") or {}
    y_proba = artifact.get("y_proba") if isinstance(artifact, dict) else None
    y_pred = artifact.get("y_pred") if isinstance(artifact, dict) else None
    if y_proba is None or y_pred is None:
        st.info("Run details are available after a successful fitted run.")
        return

    implementation = artifact.get("implementation")
    if selected_run.get("model") == "RuleFit" and implementation == "sklearn-rulefit-style":
        st.info("This run used the sklearn RuleFit-style fallback, not imodels.RuleFitClassifier.")
    elif selected_run.get("model") == "RuleFit" and implementation == "imodels-rulefit":
        st.caption("Implementation: imodels.RuleFitClassifier")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV ROC AUC", _fmt(selected_run.get("cv_roc_auc")))
    c2.metric("F1", _fmt(selected_run.get("f1")))
    c3.metric("Precision", _fmt(selected_run.get("precision")))
    c4.metric("Recall", _fmt(selected_run.get("recall")))

    left, right = st.columns([1, 1])
    with left:
        cm = artifact.get("confusion_matrix")
        if cm is not None:
            st.markdown("#### Confusion matrix")
            st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]), width="stretch")
    with right:
        with st.expander("Parameters", expanded=False):
            st.json(_json_safe_value(selected_run.get("params", {})))

    fi = artifact.get("feature_importance")
    st.markdown("#### Feature importance")
    _render_ordered_feature_importance(fi)

    st.divider()
    st.markdown("#### Model artifact")
    _render_interpretable_artifacts(selected_run, key_prefix="inspect_")


def _promoted_evaluation_bundle(selected_run: dict[str, Any], y: Any) -> dict[str, Any]:
    """Freeze the Workbench evaluation evidence used by Governance.

    Governance must not silently regenerate metrics from the full-data refit:
    that would compare training diagnostics with the Workbench's CV evidence.
    The bundle is intentionally self-contained and is copied into session
    state at promotion time.
    """
    artifact = selected_run.get("artifact") if isinstance(selected_run.get("artifact"), dict) else {}
    y_true = np.asarray(y).reshape(-1)
    y_pred = np.asarray(artifact.get("y_pred")).reshape(-1) if artifact.get("y_pred") is not None else np.asarray([])
    y_proba = np.asarray(artifact.get("y_proba"), dtype=float).reshape(-1) if artifact.get("y_proba") is not None else np.asarray([])
    n = min(len(y_true), len(y_pred), len(y_proba))
    metric_names = ("cv_roc_auc", "roc_auc", "accuracy", "precision", "recall", "f1")
    metrics = {name: _scalar_metric(selected_run.get(name)) for name in metric_names}
    return {
        "source": "promoted_workbench_run",
        "scope": str(artifact.get("evaluation_scope") or "unknown"),
        "cv": artifact.get("evaluation_cv"),
        "random_state": artifact.get("evaluation_random_state"),
        "error": artifact.get("evaluation_error"),
        "run_id": selected_run.get("run_id"),
        "metrics": metrics,
        "y_true": y_true[:n].copy(),
        "y_pred": y_pred[:n].copy(),
        "y_proba": y_proba[:n].copy(),
        "confusion_matrix": np.asarray(artifact.get("confusion_matrix")).copy()
        if artifact.get("confusion_matrix") is not None
        else None,
    }


def _render_promotion(selected_run: dict[str, Any], ctx: dict[str, Any]) -> None:
    # Only HUGIML runs can be promoted to Governance.
    if selected_run.get("model") != "HUGIML":
        return
    if not st.button("Promote this HUGIML run to Governance", type="primary", width="stretch"):
        return

    artifact = selected_run.get("artifact")
    if isinstance(artifact, dict):
        model = artifact.get("model")
        # Preserve the original tuning/CV object whenever available.  The old
        # promotion path manufactured a one-row pseudo result, which discarded
        # fold/candidate evidence and could make Governance display a different
        # result table from the Workbench run that was promoted.
        source_result = artifact.get("tuning_result")
        if source_result is None:
            source_result = type("WorkbenchHUGIMLResult", (), {})()
            source_result.best_estimator_ = model
            source_result.best_params_ = selected_run.get("params", {})
            source_result.best_score_ = selected_run.get("cv_roc_auc")
            source_result.results_ = [
                {"status": "ok", "score": selected_run.get("cv_roc_auc"), **(selected_run.get("params", {}) or {})}
            ]
            source_result.status_ = "ok"
            source_result.error_ = None
        try:
            predictions = score_cases(model, ctx["X"])
        except Exception:
            predictions = pd.DataFrame()
        evaluation = _promoted_evaluation_bundle(selected_run, ctx.get("y"))
        st.session_state["hugiml_promoted_governance_ctx"] = {
            **ctx,
            "cache_key": f"workbench:{selected_run.get('run_id')}",
            "result": source_result,
            "model": model,
            "predictions": predictions,
            "evaluation": evaluation,
            "promoted_run": {
                "run_id": selected_run.get("run_id"),
                "params": copy.deepcopy(selected_run.get("params", {})),
                "metrics": copy.deepcopy(evaluation.get("metrics", {})),
            },
            "cv": ctx.get("cv"),
            "random_state": ctx.get("random_state"),
        }
        st.session_state["hugiml_governance_requested"] = True
        st.success("Promoted. Opening Governance now.")
        st.rerun()



def _request_results_drilldown_tab() -> None:
    """Keep the Workbench on Results → Model drill-down after model selection."""
    st.session_state["hugiml_jump_to_results_tab"] = True
    st.session_state["hugiml_nav_section"] = "Results"
    st.session_state["hugiml_results_inner_section"] = "Model drill-down"


def _request_results_pattern_delta_tab() -> None:
    """Keep the Workbench on Results → Artifact comparison."""
    st.session_state["hugiml_jump_to_results_tab"] = True
    st.session_state["hugiml_nav_section"] = "Results"
    st.session_state["hugiml_results_inner_section"] = "Artifact comparison"


def _short_config_summary(run: dict[str, Any]) -> str:
    params = run.get("params", {}) if isinstance(run.get("params"), dict) else {}
    if params.get("mode") == "guided_fast_tune":
        selected = params.get("selected_params", {}) if isinstance(params.get("selected_params"), dict) else {}
        bits = ["Guided fast tune"]
        if params.get("candidate_count") is not None:
            bits.append(f"{int(params.get('candidate_count')):,} candidates")
        if selected:
            bits.append(", ".join(
                f"{k}={selected.get(k)}"
                for k in (
                    "adaptive_binning", "adaptive_binning_sample_frac",
                    "B", "L", "topK", "G", "feature_mode"
                )
                if k in selected
            ))
        return " | ".join(bits)
    ordered = []
    preferred = (
        "adaptive_binning", "adaptive_binning_sample_frac", "adaptive_binning_sample_random_state",
        "B", "L", "topK", "G", "feature_mode", "topk_budget_strict",
        "C", "max_iter", "max_depth", "min_samples_leaf", "tree_size", "max_rules",
        "n_estimators", "learning_rate", "max_bins", "interactions",
    )
    for key in preferred:
        if key in params:
            ordered.append(key)
    for key in sorted(params):
        if key not in ordered and not str(key).startswith("__"):
            ordered.append(key)
    return ", ".join(f"{k}={params.get(k)}" for k in ordered)


def _pattern_delta_run_label(run_id: str, runs_by_id: dict[str, dict[str, Any]]) -> str:
    run = runs_by_id.get(str(run_id), {})
    summary = _short_config_summary(run)
    return f"{run_id} — {summary}" if summary else str(run_id)


def _run_config_frame(run: dict[str, Any], label: str) -> pd.DataFrame:
    params = run.get("params", {}) if isinstance(run.get("params"), dict) else {}
    rows = [
        {"selection": label, "field": "run_id", "value": run.get("run_id")},
        {"selection": label, "field": "model", "value": run.get("model")},
        {"selection": label, "field": "cv_roc_auc", "value": run.get("cv_roc_auc")},
        {"selection": label, "field": "f1", "value": run.get("f1")},
    ]
    if params.get("mode") == "guided_fast_tune":
        selected = params.get("selected_params", {}) if isinstance(params.get("selected_params"), dict) else {}
        rows.extend([
            {"selection": label, "field": "mode", "value": "Guided fast tune"},
            {"selection": label, "field": "candidate_count", "value": params.get("candidate_count")},
            {"selection": label, "field": "fast_path_used", "value": params.get("fast_path_used")},
        ])
        for key in (
            "adaptive_binning",
            "adaptive_binning_sample_frac",
            "adaptive_binning_sample_random_state",
            "B",
            "L",
            "topK",
            "G",
            "feature_mode",
            "topk_budget_strict",
            "augmented_pair_transforms",
            "augmented_pair_mode",
            "aug_feature_size",
            "max_pair_features",
            "ii_partner_size",
            "interaction_relaxed_mining",
            "interaction_relaxed_feature_size",
        ):
            if key in selected:
                rows.append({"selection": label, "field": key, "value": selected.get(key)})
        return pd.DataFrame(rows)
    for key, value in params.items():
        if not str(key).startswith("__"):
            rows.append({"selection": label, "field": str(key), "value": value})
    return pd.DataFrame(rows)


def _render_results(ctx: dict[str, Any]) -> None:
    runs = st.session_state.get("hugiml_workbench_runs", [])
    if not runs:
        st.info("Run experiments to populate the leaderboard and comparison plots.")
        return

    leaderboard = _leaderboard_frame(runs)
    ok_df = leaderboard[leaderboard["status"] == "ok"] if not leaderboard.empty else pd.DataFrame()
    failed_df = leaderboard[leaderboard["status"] != "ok"] if not leaderboard.empty else pd.DataFrame()

    # Best run per model (by CV ROC-AUC on held-out validation folds).
    # Curves, interpretability, and drill-down always show only these winners;
    # the leaderboard alone shows all runs with a ✓ marker on the winners.
    best_per_model = _best_run_per_model(runs)
    best_runs = list(best_per_model.values())
    result_sections = [
        "Leaderboard",
        "Compare curves",
        "Interpretability",
        "Artifact comparison",
        "Model drill-down",
    ]
    requested_inner = st.session_state.pop("hugiml_results_inner_section", None)
    if requested_inner in result_sections:
        st.session_state["hugiml_results_inner_section_current"] = requested_inner
    if "hugiml_results_inner_section_current" not in st.session_state:
        st.session_state["hugiml_results_inner_section_current"] = "Leaderboard"
    inner_index = result_sections.index(st.session_state["hugiml_results_inner_section_current"])
    result_section = st.radio(
        "Results view",
        result_sections,
        index=inner_index,
        horizontal=True,
        label_visibility="collapsed",
        key="hugiml_results_inner_section_choice",
    )
    st.session_state["hugiml_results_inner_section_current"] = result_section

    if result_section == "Leaderboard":
        st.markdown("### Leaderboard")
        compact_cols = ["run_id", "category", "display", "status", "best_config", "cv_roc_auc", "f1", "precision", "recall", "accuracy", "fit_time_sec"]
        st.dataframe(dataframe_for_display(leaderboard[[c for c in compact_cols if c in leaderboard.columns]]), width="stretch", hide_index=True)
        st.caption("**best_config ✓** marks the highest CV ROC-AUC configuration per model — the winner selected on held-out validation data.")

        # Tuning summary: show best-per-model table when any model ran multiple configs
        tuned_models = {
            str(r.get("model"))
            for r in runs
            if sum(1 for rr in runs if rr.get("model") == r.get("model") and rr.get("status") == "ok") > 1
        }
        best_per_model = _best_run_per_model(runs)
        if tuned_models and best_per_model:
            st.markdown("#### Hyperparameter tuning — best configuration per model")
            st.caption(
                "For models run with multiple hyperparameter candidates, the row below shows the "
                "winner selected by highest mean CV ROC-AUC on held-out validation folds. "
                "Use **Model drill-down** to inspect any specific configuration."
            )
            winner_rows = []
            for model_name, best_run in best_per_model.items():
                n_candidates = sum(1 for r in runs if r.get("model") == model_name and r.get("status") == "ok")
                winner_rows.append({
                    "model": _model_short(model_name),
                    "best_run_id": best_run.get("run_id"),
                    "candidates_evaluated": n_candidates,
                    "best_cv_roc_auc": _fmt(best_run.get("cv_roc_auc")),
                    "best_f1": _fmt(best_run.get("f1")),
                    "best_params": _summarize_params_for_display(best_run.get("params", {})),
                })
            st.dataframe(
                dataframe_for_display(pd.DataFrame(winner_rows)),
                width="stretch",
                hide_index=True,
            )

        if not ok_df.empty:
            best = ok_df.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Best model", _model_short(str(best.get("model", "N/A"))))
            c2.metric("Best CV ROC AUC", _fmt(best.get("cv_roc_auc")))
            c3.metric("Successful runs", f"{len(ok_df):,}/{len(leaderboard):,}")
        if not failed_df.empty:
            st.markdown("#### Failed runs")
            st.dataframe(dataframe_for_display(failed_df[["run_id", "category", "display", "status", "diagnostic"]]), width="stretch", hide_index=True)
            st.caption("Optional models can fail due to missing packages, unsupported versions, or data constraints. Successful runs are unaffected.")

    elif result_section == "Compare curves":
        _render_curve_comparison(best_runs, ctx["y"])

    elif result_section == "Interpretability":
        _render_interpretability_comparison(best_runs)

    elif result_section == "Artifact comparison":
        st.markdown("### Artifact comparison")
        st.caption("Compare interpretable artifacts across two runs of the same model: HUGIML patterns, RuleFit rules, LR coefficients, or DT text trees.")
        _render_artifact_comparison(runs)

    elif result_section == "Model drill-down":
        if not best_runs:
            st.warning("No successful runs to inspect.")
        else:
            st.markdown(
                """
                <div class="hugiml-section-note">
                  <p><b>Model drill-down</b> shows the best configuration per model, selected by CV ROC-AUC
                  on held-out validation folds. Metrics (CV ROC-AUC, F1, Precision, Recall) are computed
                  from <em>stitched out-of-fold predictions</em>. Each sample is predicted exactly once on
                  its held-out CV fold, so no data leaks into the score. The confusion matrix uses those same
                  out-of-fold predictions; feature importance and RPTE rules come from the final refit on the
                  complete dataset after cross-validation.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            best_run_ids_ordered = [r.get("run_id") for r in best_runs if r.get("status") == "ok"]
            previous_selected = st.session_state.get("hugiml_model_drilldown_selected_id")
            selected_index = best_run_ids_ordered.index(previous_selected) if previous_selected in best_run_ids_ordered else 0
            selected_id = st.selectbox(
                "Select model",
                best_run_ids_ordered,
                index=selected_index,
                key="hugiml_model_drilldown_selected_id",
                on_change=_request_results_drilldown_tab,
            )
            selected_run = next(r for r in best_runs if r.get("run_id") == selected_id)
            _render_selected_run_details(selected_run, ctx["y"])
            _render_promotion(selected_run, ctx)


def _fmt(value: Any) -> str:
    scalar = _scalar_metric(value)
    return "N/A" if scalar is None else f"{scalar:.4f}"



def render_workbench(ctx: dict[str, Any], section: str = "Setup") -> None:
    message = st.session_state.pop("hugiml_workbench_last_message", None)
    if message:
        st.success(message)

    _render_intro(ctx)
    _render_workbench_workflow()

    current_context_key = f"{ctx.get('cache_key')}:{ctx.get('cv')}:{ctx.get('random_state')}"
    previous_context_key = st.session_state.get("hugiml_workbench_context_key")
    has_previous_runs = bool(st.session_state.get("hugiml_workbench_runs"))
    if previous_context_key and previous_context_key != current_context_key and has_previous_runs:
        st.info(
            "Dataset or run settings changed. Existing Workbench results are preserved; "
            "switch to Results to compare, or re-run Setup to replace them."
        )

    # Setup/Results navigation.
    # Streamlit tabs cannot be selected reliably from Python, and iframe/JS-based
    # tab clicks are brittle across Streamlit releases. A horizontal radio keeps
    # the same two-section workflow while allowing us to set Results explicitly
    # after a run or artifact-comparison control rerun.
    setup_label = "⚙  Setup"
    results_label = "📊  Results"
    requested_section = st.session_state.pop("hugiml_nav_section", None)
    jump_to_results = st.session_state.pop("hugiml_jump_to_results_tab", False)
    if "hugiml_workbench_section" not in st.session_state:
        st.session_state["hugiml_workbench_section"] = "Setup"
    if "hugiml_workbench_section_token" not in st.session_state:
        st.session_state["hugiml_workbench_section_token"] = 0
    if jump_to_results or requested_section == "Results":
        st.session_state["hugiml_workbench_section"] = "Results"
        st.session_state["hugiml_workbench_section_token"] += 1

    section_index = 1 if st.session_state.get("hugiml_workbench_section") == "Results" else 0
    section_choice = st.radio(
        "Workbench section",
        [setup_label, results_label],
        index=section_index,
        horizontal=True,
        label_visibility="collapsed",
        key=f"hugiml_workbench_section_choice_{st.session_state['hugiml_workbench_section_token']}",
    )
    st.session_state["hugiml_workbench_section"] = "Results" if section_choice == results_label else "Setup"

    if section_choice == setup_label:
        selected = _render_model_picker()
        param_map = _render_configs(selected)

        st.markdown("### 3. Run experiments")
        total_runs = sum(len(param_map.get(m, [])) for models in selected.values() for m in models)
        tuning_models = [m for m, configs in param_map.items() if len(configs) > 1]
        run_disabled = total_runs == 0
        if total_runs == 0:
            run_label = "Select models to run experiments"
        elif tuning_models:
            run_label = f"Run {total_runs:,} candidate(s) — tuning {', '.join(_model_short(m) for m in tuning_models)}"
        else:
            run_label = f"Run {total_runs:,} candidate experiment(s)"
        def _run_selected_workbench_experiments() -> None:
            """Run experiments from the button callback, before the next render.

            Streamlit executes button callbacks before the script reruns. Setting
            the Workbench section here means the section selector is created with
            Results already selected on the next render, instead of trying to
            mutate widget state after the selector has been instantiated.
            """
            st.session_state["hugiml_workbench_runs"] = run_experiments(
                selected,
                param_map,
                ctx["X"],
                ctx["y"],
                int(ctx.get("cv", 3)),
                int(ctx.get("random_state", 2026)),
            )
            st.session_state["hugiml_workbench_context_key"] = current_context_key
            st.session_state["hugiml_workbench_section"] = "Results"
            st.session_state["hugiml_workbench_section_token"] = st.session_state.get("hugiml_workbench_section_token", 0) + 1
            st.session_state["hugiml_workbench_last_message"] = "Experiments complete. Showing Results."

        st.button(
            run_label,
            type="primary",
            disabled=run_disabled,
            width="stretch",
            on_click=_run_selected_workbench_experiments,
        )

    else:
        _render_results(ctx)

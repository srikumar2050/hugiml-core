"""Leaf-wise RPTE: a boosted ensemble of shallow trees over HUGIML features,
used as an optional downstream estimator for higher-order interactions.

Two backends, selected by `enable_lookahead` (True / False / "adaptive",
default "adaptive"):

Bounded lookahead (enable_lookahead=True). Each tree is grown natively
(see `_hugiml_core.rpte_grow_tree` in native/rpte_tree.cpp): leaf-wise
best-first, trying an ordinary greedy stump at each leaf first and
falling back to a bounded depth-two microtree search when the stump's
held-out gain stalls. A microtree's root is a raw-feature pair (mined
by HUGIML or synthesized on the fly) and its child is a single raw
feature or another pair, optionally extended one more level for
5-way/6-way interactions. A candidate is committed only when its
probe-set gain clears both fixed thresholds and a Bonferroni-corrected
statistical-significance bar. This module's role is orchestration: it
drives the boosting loop, resolves HUGIML's feature catalog into the
raw-feature-index form the native engine expects, and renders the
returned tree structure into human-readable rules -- it performs none
of the search itself.

Sequential default (enable_lookahead=False). An ordinary leaf-wise
ensemble of `sklearn.tree.DecisionTreeClassifier` stumps, with the
same raw-feature reservation but no augmented-pair/microtree search --
see _DefaultRPTEFeatureExtractor / _DefaultRPTEFeatureLR.

"adaptive" picks a backend per fit, from the data (see
_max_marginal_gain): if no single raw feature carries real marginal
signal on its own (e.g. pure parity), lookahead is worth its extra
cost; otherwise the sequential backend is generally at least as
accurate for less compute.

Both the root and any pair-valued child combine their two raw features
by binarizing first when both inputs are binary-like, or combining raw
values directly when either is continuous (e.g. GPA or a
log-coinsurance-rate, where a hard binary split first would discard
graded information the interaction might depend on).

Boosting. Both backends fit an ensemble of trees to the negative
gradient of binomial deviance (r_i = y_i - sigmoid(F(x_i))); each
tree's leaf values are the exact per-leaf Newton step, and a tree is
kept only if a backtracking line search verifies it lowers deviance.

Final downstream model. The L1 logistic layer receives every accepted tree
leaf indicator plus each supplied HUGIML downstream column that was not selected
in an accepted tree split. A supplied feature is therefore represented either
directly or through RPTE leaf conjunctions, never both.

Split acceptance. A partition's information gain is, via Wilks'
theorem, asymptotically a chi-squared statistic. Because the lookahead
search compares many candidates per leaf and keeps the best,
`use_statistical_acceptance` (default True) Bonferroni-corrects the
significance threshold by the actual number of candidates and
probe-set size compared at that leaf. See native/rpte_significance.hpp
for the calibration this module relies on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

# The bounded-lookahead tree grower, the boosting math (Newton leaf
# values, binomial deviance), and the underlying information-gain
# kernels all live in the compiled _hugiml_core extension -- see
# native/rpte_tree.cpp, native/rpte_scoring.cpp. This module is
# orchestration only: it drives the boosting loop and HUGIML metadata
# resolution around those native calls, and renders their output into
# human-readable rules. There is no Python fallback implementation; if
# the extension isn't importable, RPTE is unavailable.
try:
    import _hugiml_core as _native
except ImportError as exc:  # pragma: no cover - exercised only in a broken install
    raise ImportError(
        "RPTE requires the compiled _hugiml_core extension (native/rpte_tree.cpp, "
        "native/rpte_scoring.cpp). Reinstall hugiml-core from a wheel, or build the "
        "extension from source (see setup.py)."
    ) from exc

# Matches rpte_core::PairOp in native/rpte_core.hpp.
_PAIR_OP_CODES = {"absolute_difference": 0, "signed_difference": 1, "sum": 2, "product": 3}
_PAIR_OP_NAMES = {code: name for name, code in _PAIR_OP_CODES.items()}

LEAF_CONFIGS = {"2xD": 2, "3xD": 3, "4xD": 4}

_PATTERN_LABEL = re.compile(r"^pattern:(.+)$")
_ORIG_DUMMY_LABEL = re.compile(r"^orig:(.+)_([^_]+)$")


# ---------------------------------------------------------------------------
# Newton boosting on binomial deviance (Friedman 2001, "Greedy Function
# Approximation: A Gradient Boosting Machine"; the per-leaf 2nd-order
# step follows Friedman, Hastie & Tibshirani 2000, "Additive Logistic
# Regression"). Both RPTE backends fit their tree sequence this way:
#
#   F_0(x)      = log-odds of the base rate
#   p_hat(x)    = sigmoid(F(x))
#   r_i         = y_i - p_hat(x_i)             (negative gradient)
#   w_i         = p_hat(x_i) (1 - p_hat(x_i))  (Newton weight)
#   h_m         = a weak learner whose leaf values are the exact
#                 per-leaf Newton step sum(r)/(sum(w)+ridge)
#   F_m         = F_{m-1} + eta * h_m,   eta from a backtracking line
#                 search so the empirical deviance strictly decreases.
#
# Tree structure is found by each backend's own search, targeting
# sign(r) as a classification proxy for "regions correlated with the
# gradient". Leaf values are always the exact Newton step, computed
# natively (_hugiml_core.rpte_newton_leaf_values); a tree is kept only
# if the line search below finds a deviance-decreasing step.
# ---------------------------------------------------------------------------


def _sigmoid(F: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(F, -35.0, 35.0)))


def _apply_leaf_values(leaf_ids: np.ndarray, values: dict[int, float]) -> np.ndarray:
    return np.array([values.get(int(leaf), 0.0) for leaf in np.asarray(leaf_ids)], dtype=np.float64)


def _backtracking_boost_step(
    y01: np.ndarray,
    F: np.ndarray,
    h: np.ndarray,
    initial_eta: float,
    current_deviance: float,
    max_backtracks: int = 6,
    shrink: float = 0.5,
) -> tuple[float, np.ndarray, float, bool]:
    """Backtracking line search for one boosting step F -> F + eta*h:
    halve eta (from `initial_eta`, the configured learning rate, as an
    upper bound rather than a value applied unconditionally) until the
    resulting deviance strictly decreases, or give up after
    `max_backtracks` halvings. A tree is accepted into the ensemble
    only when this returns accepted=True."""
    eta = float(initial_eta)
    for _ in range(max_backtracks):
        F_new = F + eta * h
        deviance_new = float(_native.rpte_binomial_deviance(y01, _sigmoid(F_new), 1e-12))
        if deviance_new < current_deviance:
            return eta, F_new, deviance_new, True
        eta *= shrink
    return eta, F, current_deviance, False


@dataclass
class _BoostingRound:
    """One accepted round's leaf assignments plus a callback the driver
    invokes only when that round is actually kept."""

    leaf_ids: np.ndarray
    on_accept: Callable[[float, dict[int, float], float], None]


def _fit_newton_boosted_ensemble(
    y01: np.ndarray,
    n_estimators: int,
    learning_rate: float,
    min_tree_residual_gain: float,
    grow_round: Callable[[int, np.ndarray, np.ndarray, np.ndarray], _BoostingRound | None],
    stop_after_accept: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, list[float], float]:
    """Newton-boosting driver shared by both RPTE backends.

    Each round computes the negative gradient `r` and Newton weight `w`
    of binomial deviance at the current score, asks `grow_round` to fit
    one weak learner against them, takes its exact per-leaf Newton step
    (computed natively), and keeps the round only if a backtracking
    line search finds a step that lowers deviance. Stops when a round
    yields nothing, its step is rejected, or the deviance decrease
    falls below `min_tree_residual_gain`.

    Returns the final cumulative score, the deviance trajectory
    (monotone non-increasing by construction), and the base log-odds
    the score started from.
    """
    n = y01.shape[0]
    p_bar = float(np.clip(y01.mean(), 1e-6, 1 - 1e-6))
    F = np.full(n, np.log(p_bar / (1 - p_bar)), dtype=np.float64)
    base_log_odds = float(np.log(p_bar / (1 - p_bar)))
    deviance = float(_native.rpte_binomial_deviance(y01.astype(np.float64), _sigmoid(F), 1e-12))
    deviance_history = [deviance]
    for round_index in range(int(n_estimators)):
        p_hat = _sigmoid(F)
        r = y01 - p_hat
        w = np.clip(p_hat * (1.0 - p_hat), 1e-6, None)
        labels = (r > 0).astype(np.int8)
        if np.unique(labels).size < 2:
            break
        grown = grow_round(round_index, r, w, labels)
        if grown is None:
            break
        values = _native.rpte_newton_leaf_values(
            np.asarray(grown.leaf_ids, dtype=np.int64), r, w, 1.0
        )
        h = _apply_leaf_values(grown.leaf_ids, values)
        eta, F_proposed, deviance_proposed, accepted = _backtracking_boost_step(
            y01, F, h, learning_rate, deviance
        )
        if not accepted:
            break
        relative_decrease = (deviance - deviance_proposed) / max(deviance, 1e-12)
        if relative_decrease < min_tree_residual_gain and round_index > 0:
            break
        F = F_proposed
        deviance = deviance_proposed
        deviance_history.append(deviance)
        grown.on_accept(eta, values, relative_decrease)
        if stop_after_accept is not None and stop_after_accept():
            break
    return F, deviance_history, base_log_odds


def _max_marginal_gain(
    X: np.ndarray, y: np.ndarray, raw_indices: list[int], min_samples_leaf: int
) -> float:
    """Best single-raw-feature information gain against y, over a single
    median-or-binary split per feature. Used by enable_lookahead="adaptive"
    to decide whether the bounded-lookahead mechanism is likely worth its
    cost: if every raw feature's best split has effectively no gain (e.g.
    pure parity), no amount of ordinary greedy tree growing can get started
    at all, which is exactly the regime lookahead exists for."""
    y = np.asarray(y, dtype=np.int8)
    best = 0.0
    for idx in raw_indices:
        col = np.asarray(X[:, idx], dtype=np.float64)
        vals = np.unique(col[np.isfinite(col)])
        if vals.size < 2:
            continue
        threshold = float((vals[0] + vals[1]) / 2.0) if vals.size == 2 else float(np.median(vals))
        bit = col > threshold
        n0, n1 = int((~bit).sum()), int(bit.sum())
        if n0 < min_samples_leaf or n1 < min_samples_leaf:
            continue
        gain = float(_native.rpte_partition_ig_bits(bit.astype(np.int64), y))
        if gain > best:
            best = gain
    return best


# ---------------------------------------------------------------------------
# Rule rendering. Everything below operates on column NAMES and
# metadata dicts, or on sklearn's own DecisionTreeClassifier.tree_
# structure (the sequential backend) -- neither touches the native
# lookahead tree engine directly, so none of it changed when that
# engine moved to C++. The one exception, _walk_native_tree_leaf_
# conditions, walks the plain dict _hugiml_core.rpte_grow_tree returns.
# ---------------------------------------------------------------------------


def leaf_path_conditions(
    tree: DecisionTreeClassifier, leaf_id: int, feature_names: list[str]
) -> list[tuple[str, str]]:
    """Reconstruct the root-to-leaf conjunction of split tests for one leaf,
    as a list of (feature_name, "0"|"1") pairs in root-to-leaf order.

    Assumes binary (0/1) input features, so every split threshold sits at 0.5:
    the left child corresponds to feature == 0 and the right child to feature == 1.
    """
    tree_ = tree.tree_
    parent = {}
    for node in range(tree_.node_count):
        left, right = tree_.children_left[node], tree_.children_right[node]
        if left != -1:
            parent[left] = (node, "left")
        if right != -1:
            parent[right] = (node, "right")
    conditions = []
    node = leaf_id
    while node in parent:
        parent_node, side = parent[node]
        feature = feature_names[tree_.feature[parent_node]]
        conditions.append((feature, "0" if side == "left" else "1"))
        node = parent_node
    return list(reversed(conditions))


def leaf_path_conditions_with_thresholds(
    tree: DecisionTreeClassifier, leaf_id: int, feature_names: list[str]
) -> list[tuple[str, float, bool]]:
    """Same root-to-leaf path reconstruction as leaf_path_conditions, but
    keeps the split's actual threshold value and direction (is_right: True
    = right/">" branch) instead of collapsing every split to "0"/"1"."""
    tree_ = tree.tree_
    parent = {}
    for node in range(tree_.node_count):
        left, right = tree_.children_left[node], tree_.children_right[node]
        if left != -1:
            parent[left] = (node, False)
        if right != -1:
            parent[right] = (node, True)
    conditions = []
    node = leaf_id
    while node in parent:
        parent_node, is_right = parent[node]
        feature = feature_names[tree_.feature[parent_node]]
        conditions.append((feature, float(tree_.threshold[parent_node]), is_right))
        node = parent_node
    return list(reversed(conditions))


def _pattern_atoms(name: str) -> list[tuple[str, str]] | None:
    """Parse a HUGIML pattern column name like "pattern:x0=1, x1=1" into
    its atomic (feature, value) requirements, or None if `name` isn't a
    pattern column."""
    match = _PATTERN_LABEL.match(name)
    if match is None:
        return None
    atoms = []
    for term in match.group(1).split(", "):
        if "=" not in term:
            return None
        feature, value = term.rsplit("=", 1)
        atoms.append((feature, value))
    return atoms


def _orig_dummy_atom(name: str) -> tuple[str, str] | None:
    """Parse a HUGIML one-hot original-feature column name like
    "orig:x1_0" into (feature, value) = ("x1", "0"), or None for names
    this can't confidently decompose."""
    match = _ORIG_DUMMY_LABEL.match(name)
    return (match.group(1), match.group(2)) if match else None


def simplify_conditions(conditions: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop leaf-path conditions on a mined pattern column whose truth value
    is already logically guaranteed by an original-feature condition
    elsewhere in the same leaf path, so every surviving condition carries
    independent information."""
    known_values: dict[str, str] = {}
    for name, value in conditions:
        if _pattern_atoms(name) is not None:
            continue
        atom = _orig_dummy_atom(name)
        if atom is not None and value == "1":
            feature, atom_value = atom
            known_values[feature] = atom_value
    keep = []
    for name, value in conditions:
        atoms = _pattern_atoms(name)
        if atoms is None:
            keep.append((name, value))
            continue
        if value == "1":
            redundant = all(
                known_values.get(feature) == atom_value for feature, atom_value in atoms
            )
        else:
            redundant = any(
                feature in known_values and known_values[feature] != atom_value
                for feature, atom_value in atoms
            )
        if not redundant:
            keep.append((name, value))
    return keep
def _render_threshold(
    column_name: str,
    standardized_threshold: float,
    original_feature_standardization: dict[str, dict[str, Any]],
    augmented_catalog_by_name: dict[str, dict[str, Any]],
    pattern_provenance: dict[str, Any],
    *,
    is_right: bool,
) -> dict[str, Any]:
    """Render one downstream split in both estimator and raw units.

    ``is_right=True`` means the tree followed ``value > threshold``; the
    left branch is ``value <= threshold``.  Keeping the operator inside the
    rendering helper prevents a subtle but serious explanation error where a
    separately-reported direction field was correct while the human-readable
    condition text always used ``>``.
    """
    name = str(column_name)
    operator = ">" if is_right else "<="
    base = {
        "downstream_feature": name,
        "downstream_condition": (
            f"{name} {operator} {standardized_threshold:.6g} (standardized units)"
        ),
        "raw_condition": None,
        "raw_sources": [],
        "invertible": False,
        "transformation_formula": None,
        "pair_origin": None,
        "operation": None,
        "operator": operator,
        "missing_value_policy": None,
    }
    if name.startswith("orig:"):
        raw_feature = name[len("orig:") :]
        base["family"] = "original"
        base["raw_sources"] = [raw_feature]
        std = original_feature_standardization.get(raw_feature)
        if std is not None:
            base["missing_value_policy"] = std.get("missing_value_policy")
            scale = std.get("standardization_scale")
            mean = std.get("standardization_mean")
            if (
                scale not in (None, 0)
                and mean is not None
                and np.isfinite(scale)
                and np.isfinite(mean)
            ):
                raw_threshold = standardized_threshold * scale + mean
                base["raw_condition"] = f"{raw_feature} {operator} {raw_threshold:.6g}"
                base["invertible"] = True
                base["transformation_formula"] = f"({raw_feature} - {mean:.6g}) / {scale:.6g}"
        return base
    if name.startswith("augmented_pair:"):
        base["family"] = "augmented_pair"
        base["pair_origin"] = "hugiml"
        item = augmented_catalog_by_name.get(name) or augmented_catalog_by_name.get(
            name[len("augmented_pair:") :]
        )
        if item:
            inputs = [str(v) for v in item.get("inputs", [])]
            base["raw_sources"] = inputs
            base["operation"] = item.get("operation")
            base["missing_value_policy"] = item.get("pair_missing_policy")
            mean = item.get("standardization_mean")
            scale = item.get("standardization_scale")
            raw_formula = item.get("raw_formula") or item.get("formula")
            if (
                mean is not None
                and scale not in (None, 0)
                and np.isfinite(scale)
                and np.isfinite(mean)
            ):
                raw_threshold = standardized_threshold * scale + mean
                target = raw_formula or f"raw({'/'.join(inputs)})"
                base["raw_condition"] = f"{target} {operator} {raw_threshold:.6g}"
                base["invertible"] = True
                if raw_formula:
                    base["transformation_formula"] = f"({raw_formula} - {mean:.6g}) / {scale:.6g}"
        return base
    if name.startswith("synthetic_pair:"):
        # RPTE-synthesized pairs can combine binarized or continuous inputs;
        # the fitted descriptor currently does not retain enough information
        # to invert the split to one raw-domain scalar threshold.  Report the
        # exact operator in RPTE-combined units and preserve operation/source
        # provenance without inventing a raw equivalent.
        parts = name[len("synthetic_pair:") :].split("__")
        op = parts[0] if parts else "?"
        raw_sources = [
            part[len("orig:") :] if part.startswith("orig:") else part for part in parts[1:] if part
        ]
        base["family"] = "synthetic_pair"
        base["pair_origin"] = "rpte"
        base["operation"] = op
        base["raw_sources"] = raw_sources
        base["downstream_condition"] = (
            f"{name} {operator} {standardized_threshold:.6g} (RPTE-combined units)"
        )
        return base
    if name.startswith("pattern:"):
        base["family"] = "pattern"
        info = pattern_provenance.get(name)
        if isinstance(info, dict):
            base["raw_sources"] = list(info.get("raw_features", []))
        elif info:
            base["raw_sources"] = [str(feature) for feature in info]
        # HUG pattern columns are binary indicators.  A valid tree split is
        # normally at 0.5; only claim an exact Boolean interpretation when
        # the threshold separates 0 from 1.
        if 0.0 <= standardized_threshold < 1.0:
            holds = is_right
            state = "holds" if holds else "does not hold"
            base["downstream_condition"] = (
                f"{name} {state} (indicator {operator} {standardized_threshold:.6g})"
            )
            label = name[len("pattern:") :]
            base["raw_condition"] = label if holds else f"NOT ({label})"
            base["invertible"] = True
        return base
    base["family"] = "unknown"
    return base


def _render_linear_term(
    column_name: str,
    original_feature_standardization: dict[str, dict[str, Any]],
    augmented_catalog_by_name: dict[str, dict[str, Any]],
    pattern_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Describe a direct fallback logistic feature without inventing a split.

    The raw-feature fallback has no tree threshold: its coefficient multiplies
    the feature value directly.  Reusing ``_render_threshold(..., 0)`` used to
    falsely describe these rows as ``feature > 0`` rules.
    """
    info = _render_threshold(
        column_name,
        0.0,
        original_feature_standardization,
        augmented_catalog_by_name,
        pattern_provenance,
        is_right=True,
    )
    info["operator"] = "linear_term"
    info["standardized_threshold"] = None
    info["downstream_condition"] = f"linear term: {column_name}"
    family = info.get("family")
    if family == "original":
        raw = info.get("raw_sources", [column_name])
        info["raw_condition"] = f"linear value of {raw[0] if raw else column_name}"
    elif family == "augmented_pair":
        catalog_item = augmented_catalog_by_name.get(column_name) or augmented_catalog_by_name.get(
            str(column_name).removeprefix("augmented_pair:")
        )
        formula = (catalog_item or {}).get("raw_formula") or (catalog_item or {}).get("formula")
        info["raw_condition"] = f"linear value of {formula or column_name}"
    elif family == "pattern":
        info["raw_condition"] = f"indicator value of {str(column_name).removeprefix('pattern:')}"
    else:
        info["raw_condition"] = f"linear value of {column_name}"
    return info


def _simplify_threshold_path(
    conditions: list[tuple[str, float, bool]],
) -> list[tuple[str, float, bool]]:
    """Collapse repeated splits on one column to its tightest bounds.

    For ``x > t`` conditions only the largest threshold matters; for
    ``x <= t`` only the smallest matters.  If both directions are present,
    the result is the canonical interval ``lower < x <= upper`` represented
    by at most two conditions.
    """
    bounds: dict[str, dict[bool, tuple[float, int]]] = {}
    order: list[str] = []
    for position, (name, threshold, is_right) in enumerate(conditions):
        if name not in bounds:
            bounds[name] = {}
            order.append(name)
        current = bounds[name].get(is_right)
        tighter = (
            current is None
            or (is_right and threshold > current[0])
            or (not is_right and threshold < current[0])
        )
        if tighter:
            bounds[name][is_right] = (float(threshold), position)

    simplified: list[tuple[str, float, bool, int]] = []
    order_index = {name: index for index, name in enumerate(order)}
    for name in order:
        for is_right, (threshold, position) in bounds[name].items():
            simplified.append((name, threshold, is_right, position))
    simplified.sort(key=lambda item: (order_index[item[0]], item[3]))
    return [(name, threshold, is_right) for name, threshold, is_right, _ in simplified]


def _native_synthetic_column_name(spec: dict[str, Any], base_names: list[str]) -> str:
    """Display name for one column a bounded-lookahead tree synthesized
    on the fly (see native/rpte_tree.cpp's ColumnSpec), matching the
    naming convention HUGIML's own mined columns use. A synthetic
    column's two sources are always raw features (never another
    synthetic column), so they resolve directly against `base_names`."""
    op = _PAIR_OP_NAMES.get(spec["op"], "?")
    a_idx, b_idx = spec["a_idx"], spec["b_idx"]
    a_name = base_names[a_idx] if a_idx < len(base_names) else f"col{a_idx}"
    b_name = base_names[b_idx] if b_idx < len(base_names) else f"col{b_idx}"
    return f"synthetic_pair:{op}__{a_name}__{b_name}"


def _native_tree_column_names(tree_dict: dict[str, Any], base_names: list[str]) -> list[str]:
    """Full column-name list for a fitted native tree: `base_names`
    (HUGIML's downstream columns) followed by one entry per synthesized
    column, in the order native/rpte_tree.cpp materialized them --
    matching the column-index space `tree_dict`'s node/growth-log
    indices are expressed in."""
    names = list(base_names)
    for spec in tree_dict.get("synthetic_specs", []):
        names.append(_native_synthetic_column_name(spec, base_names))
    return names


def _walk_native_tree_leaf_conditions(tree_dict: dict[str, Any]) -> dict[int, list[tuple[int, float, bool]]]:
    """For a fitted bounded-lookahead tree (as returned by
    _hugiml_core.rpte_grow_tree), return, per leaf node id, the
    root-to-leaf list of (feature_index, threshold, is_right)
    conditions -- is_right=True means the path took the right (value >
    threshold) branch at that split."""
    is_leaf = tree_dict["node_is_leaf"]
    feature_index = tree_dict["node_feature_index"]
    threshold = tree_dict["node_threshold"]
    left = tree_dict["node_left"]
    right = tree_dict["node_right"]
    conditions_by_leaf: dict[int, list[tuple[int, float, bool]]] = {}

    def _walk(node_id: int, path: list[tuple[int, float, bool]]) -> None:
        if is_leaf[node_id]:
            conditions_by_leaf[node_id] = list(path)
            return
        _walk(left[node_id], path + [(feature_index[node_id], threshold[node_id], False)])
        _walk(right[node_id], path + [(feature_index[node_id], threshold[node_id], True)])

    _walk(0, [])
    return conditions_by_leaf

class _DefaultRPTEFeatureExtractor:
    """Builds up to `n_estimators` shallow trees, each fit on the sign of the
    residual left behind by the trees before it (so later trees target
    whatever earlier trees missed), with disjoint split-feature ownership
    across trees so each one is forced onto a fresh slice of the input.
    Every tree's root-to-leaf paths become binary match columns via
    fit_leaves/transform_leaves for a downstream linear model to weight.

    This is the enable_lookahead=False backend -- see the module docstring
    and the comment block above this class for why it's a genuinely
    different, simpler architecture rather than the bounded-lookahead tree
    with lookahead switched off.
    """

    def __init__(
        self,
        leaf_config: str = "3xD",
        depth: int = 4,
        n_estimators: int = 10,
        min_samples_leaf: int = 5,
        learning_rate: float = 0.3,
        random_state: int = 42,
        owner_by_column: dict[int, frozenset[str]] | None = None,
        reserve_raw_features: bool = True,
    ):
        if leaf_config not in LEAF_CONFIGS:
            raise ValueError(
                f"Unknown leaf configuration {leaf_config!r}; choose from {sorted(LEAF_CONFIGS)}."
            )
        self.leaf_config = leaf_config
        self.depth = depth
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.leaf_budget = LEAF_CONFIGS[leaf_config] * depth
        self.depth_cap = 2 * depth
        self.trees_: list[DecisionTreeClassifier] = []
        self.tree_columns_: list[list[int]] = []
        self.leaf_vocab_: list[np.ndarray] = []
        # Raw-feature-NAME ownership: when
        # owner_by_column is provided (see
        # LeafWiseBoundedLookaheadRPTEFeatureExtractor._fit_default_backend,
        # which always supplies it), this backend reserves by raw-feature
        # NAME -- once any tree uses a column derived from "age", every
        # later tree is blocked from using orig:age, any pattern containing
        # age, any augmented pair containing age, and any RPTE-synthesized
        # pair containing age, matching the bounded-lookahead backend's
        # ownership model. Without name-based grouping, two trees could
        # each claim a different downstream column that both derive from
        # the same raw feature (orig:age, pattern:age=[50,60),
        # augmented_pair:age*income are three columns, one raw source),
        # understating how much that feature is actually reused and
        # double-counting its apparent importance.
        #
        # When owner_by_column is None (standalone use outside a HUGIML-
        # fitted context), reservation falls back to plain column-index
        # tracking. reserve_raw_features=False also restores plain
        # per-column reservation even when owner_by_column is supplied, for
        # callers who find strict raw-source exclusion too restrictive for
        # their accuracy/diversity tradeoff.
        self.owner_by_column = dict(owner_by_column) if owner_by_column else None
        self.reserve_raw_features = bool(reserve_raw_features)
        self.reserved_raw_features_: list[set[str]] = []

    def fit_leaves(self, X, y) -> sparse.csr_matrix:
        """Fit the tree ensemble on (X, y) and return the training leaf-match matrix."""
        X = X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)
        y01 = np.asarray(y, dtype=float)
        y01 = (y01 == y01.max()).astype(np.int8)
        n_cols = X.shape[1]
        use_raw_ownership = self.owner_by_column is not None and self.reserve_raw_features
        if use_raw_ownership:
            available_raw: set[str] = set()
            for idx in range(n_cols):
                available_raw |= self.owner_by_column.get(idx, frozenset({str(idx)}))
        else:
            available: set[int] = set(range(n_cols))
        self.trees_, self.tree_columns_, self.leaf_vocab_ = [], [], []
        self.tree_leaf_values_: list[dict[int, float]] = []
        self.reserved_raw_features_ = []

        def grow_round(round_index, r, w, labels):
            if use_raw_ownership:
                columns = [
                    idx
                    for idx in range(n_cols)
                    if not self.owner_by_column.get(idx, frozenset())
                    or self.owner_by_column[idx].issubset(available_raw)
                ]
            else:
                columns = sorted(available)
            if not columns:
                return None
            tree = DecisionTreeClassifier(
                max_depth=self.depth_cap,
                max_leaf_nodes=max(2, self.leaf_budget),
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(self.random_state) + round_index,
            )
            tree.fit(X[:, columns], labels)
            used_local = {int(i) for i in tree.tree_.feature if i >= 0}
            if not used_local:
                return None
            leaf_ids = tree.apply(X[:, columns])

            def on_accept(eta, values, relative_decrease):
                nonlocal available_raw, available
                self.trees_.append(tree)
                self.tree_columns_.append(columns)
                self.tree_leaf_values_.append(
                    {leaf: eta * value for leaf, value in values.items()}
                )
                used_columns = {columns[i] for i in used_local}
                if use_raw_ownership:
                    used_raw: set[str] = set()
                    for idx in used_columns:
                        used_raw |= self.owner_by_column.get(idx, frozenset({str(idx)}))
                    self.reserved_raw_features_.append(used_raw)
                    available_raw -= used_raw
                else:
                    available -= used_columns

            return _BoostingRound(leaf_ids=leaf_ids, on_accept=on_accept)

        _F, self.deviance_history_, self.base_log_odds_ = _fit_newton_boosted_ensemble(
            y01, self.n_estimators, self.learning_rate, 1e-8, grow_round
        )
        if not self.trees_:
            raise ValueError("No RPTE-FE tree could be constructed.")
        return self._leaf_matrix(X, fit=True)

    def transform_leaves(self, X) -> sparse.csr_matrix:
        """Encode new rows into the leaf-match column space fit_leaves already built."""
        X = X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)
        return self._leaf_matrix(X, fit=False)

    def _leaf_matrix(self, X: sparse.csr_matrix, fit: bool) -> sparse.csr_matrix:
        blocks = []
        for tree_index, (tree, columns) in enumerate(zip(self.trees_, self.tree_columns_)):
            assign = tree.apply(X[:, columns])
            if fit:
                vocab = np.unique(assign)
                self.leaf_vocab_.append(vocab)
            else:
                vocab = self.leaf_vocab_[tree_index]
            blocks.append(sparse.csr_matrix((assign[:, None] == vocab[None, :]).astype(np.float32)))
        return sparse.hstack(blocks, format="csr")

    def rule_table(
        self, coefficients, feature_names: list[str] | None = None
    ) -> list[dict[str, object]]:
        """Translate a fitted downstream LogisticRegression's coefficients
        (one per leaf-match column, in fit_leaves' column order) back into
        human-readable root-to-leaf conjunctions, sorted by |coefficient|.

        When the extractor was fit on a feature space that mixes raw
        features with mined HUGIML pattern columns (as feature_mode=
        "original_plus_patterns" does), a leaf's path can legally include
        both a pattern condition and one of that pattern's own constituent
        raw conditions -- the tree doesn't know one determines the other, so
        it may split on both even though one is then logically redundant
        (see simplify_conditions). Every rule here has already had that
        redundancy removed, so each surviving condition is genuinely
        independent information, not just a technically-true restatement of
        another condition already in the same rule.
        """
        if not self.trees_:
            raise ValueError("rule_table() needs fit_leaves() to have run first.")
        columns = [
            (tree_index, int(leaf_id))
            for tree_index, leaf_ids in enumerate(self.leaf_vocab_)
            for leaf_id in leaf_ids
        ]
        rows = []
        for column_index, coefficient in enumerate(coefficients):
            if coefficient == 0:
                continue
            tree_index, leaf_id = columns[column_index]
            columns_used = self.tree_columns_[tree_index]
            names = (
                [feature_names[c] for c in columns_used]
                if feature_names is not None
                else [f"x{c}" for c in columns_used]
            )
            conditions = simplify_conditions(
                leaf_path_conditions(self.trees_[tree_index], leaf_id, names)
            )
            rows.append(
                {
                    "tree": tree_index,
                    "leaf": leaf_id,
                    "coefficient": float(coefficient),
                    "conditions": conditions,
                }
            )
        rows.sort(key=lambda row: -abs(row["coefficient"]))
        return rows


# solver="liblinear" only ever supports penalty in {"l1", "l2"}, so this
# covers every value `lr_penalty` can meaningfully take; anything else is
# passed through unchanged and left to raise sklearn's own error, exactly
# as it would have before this translation existed.
def _liblinear_logistic_regression(
    *, lr_penalty: str, lr_C: float, random_state: int, max_iter: int = 2000
) -> LogisticRegression:
    """Build the leaf-weighting LogisticRegression for solver="liblinear".

    Uses ``hugiml._compat.liblinear_penalty_kwargs`` rather than passing
    ``penalty=lr_penalty`` directly: sklearn >= 1.8 deprecates the
    ``penalty`` string in favor of ``l1_ratio`` (removal targeted for
    sklearn 1.10), but ``l1_ratio`` is honored only when
    ``penalty="elasticnet"`` on sklearn < 1.8 -- silently falling back to a
    different penalty otherwise. The two spellings are equivalent in
    result, not interchangeable in source, so which one to use must be
    chosen by installed sklearn version. See ``_compat``'s module
    docstring for the version boundary and verification.
    """
    from hugiml._compat import liblinear_penalty_kwargs

    return LogisticRegression(
        solver="liblinear",
        C=lr_C,
        random_state=random_state,
        max_iter=max_iter,
        **liblinear_penalty_kwargs(lr_penalty),
    )


class _DefaultRPTEFeatureLR(ClassifierMixin, BaseEstimator):
    """_DefaultRPTEFeatureExtractor + LogisticRegression as one
    sklearn-compatible classifier -- the enable_lookahead=False backend for
    LeafWiseBoundedLookaheadRPTEFeatureLR. Not meant to be constructed
    directly outside this module; use
    LeafWiseBoundedLookaheadRPTEFeatureLR(enable_lookahead=False) instead,
    which owns the dispatch and exposes a single consistent public API
    regardless of which backend ends up handling the fit.
    """

    def __init__(
        self,
        leaf_config: str = "3xD",
        depth: int = 4,
        n_estimators: int = 10,
        min_samples_leaf: int = 5,
        rpte_learning_rate: float = 0.3,
        lr_C: float = 1.0,
        lr_penalty: str = "l1",
        random_state: int = 42,
    ):
        self.leaf_config = leaf_config
        self.depth = depth
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.rpte_learning_rate = rpte_learning_rate
        self.lr_C = lr_C
        self.lr_penalty = lr_penalty
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def fit(self, X, y) -> _DefaultRPTEFeatureLR:
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                "_DefaultRPTEFeatureLR is binary; multiclass targets are not supported."
            )
        self.fe_ = _DefaultRPTEFeatureExtractor(
            leaf_config=self.leaf_config,
            depth=self.depth,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            learning_rate=self.rpte_learning_rate,
            random_state=self.random_state,
        )
        leaves = self.fe_.fit_leaves(X, y)
        self.logistic_ = _liblinear_logistic_regression(
            lr_penalty=self.lr_penalty, lr_C=self.lr_C, random_state=self.random_state,
        )
        self.logistic_.fit(leaves, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "logistic_")
        return self.logistic_.predict_proba(self.fe_.transform_leaves(X))

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def rule_table(self, feature_names: list[str] | None = None) -> list[dict[str, object]]:
        check_is_fitted(self, "logistic_")
        return self.fe_.rule_table(self.logistic_.coef_[0], feature_names=feature_names)



class LeafWiseBoundedLookaheadRPTEFeatureExtractor:
    """Sequential RPTE ensemble whose component trees grow leaf-wise via
    the native bounded-lookahead engine (native/rpte_tree.cpp)."""

    def __init__(
        self,
        leaf_config: str = "3xD",
        depth: int = 4,
        n_estimators: int = 10,
        min_samples_leaf: int = 5,
        learning_rate: float = 0.3,
        random_state: int = 42,
        lookahead_child_mode: str = "shared",
        lookahead_ops: tuple[str, ...] | None = ("absolute_difference",),
        lookahead_beam_width: int = 64,
        lookahead_probe_fraction: float = 0.25,
        lookahead_min_probe_ig: float = 0.05,
        lookahead_min_increment: float = 0.03,
        greedy_stall_probe_ig: float = 0.02,
        max_root_thresholds: int = 7,
        min_probe_leaf: int = 2,
        min_weighted_probe_gain: float = 0.01,
        reserve_raw_features: bool = True,
        min_tree_residual_gain: float = 1e-8,
        raw_pair_fallback: bool = True,
        raw_pair_max_candidates: int = 400,
        aug_child_enabled: bool = True,
        aug_child_max_candidates: int = 100,
        use_statistical_acceptance: bool = True,
        significance_alpha: float = 0.05,
        enable_lookahead: bool | str = "adaptive",
        adaptive_marginal_gain_threshold: float = 0.005,
    ):
        if leaf_config not in LEAF_CONFIGS:
            raise ValueError(f"Unknown leaf_config {leaf_config!r}.")
        if lookahead_child_mode not in {"shared", "branch_specific"}:
            raise ValueError("lookahead_child_mode must be 'shared' or 'branch_specific'.")
        self.leaf_config = leaf_config
        self.depth = int(depth)
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.lookahead_child_mode = lookahead_child_mode
        self.lookahead_ops = lookahead_ops
        self.lookahead_beam_width = int(lookahead_beam_width)
        self.lookahead_probe_fraction = float(lookahead_probe_fraction)
        self.lookahead_min_probe_ig = float(lookahead_min_probe_ig)
        self.lookahead_min_increment = float(lookahead_min_increment)
        self.greedy_stall_probe_ig = float(greedy_stall_probe_ig)
        self.max_root_thresholds = int(max_root_thresholds)
        self.min_probe_leaf = int(min_probe_leaf)
        self.min_weighted_probe_gain = float(min_weighted_probe_gain)
        self.reserve_raw_features = bool(reserve_raw_features)
        self.min_tree_residual_gain = float(min_tree_residual_gain)
        self.raw_pair_fallback = bool(raw_pair_fallback)
        self.raw_pair_max_candidates = int(raw_pair_max_candidates)
        # "Aug child": lets a microtree's CHILD be a pair (existing HUGIML
        # augmented pair or synthesized raw pair), not just a single raw
        # feature -- root pair + child pair together span 4 raw features,
        # extending 3-way detection to 4-way. See native/rpte_tree.cpp's
        # build_child_pool.
        self.aug_child_enabled = bool(aug_child_enabled)
        self.aug_child_max_candidates = int(aug_child_max_candidates)
        # Statistical acceptance calibration -- see native/rpte_significance.hpp.
        self.use_statistical_acceptance = bool(use_statistical_acceptance)
        self.significance_alpha = float(significance_alpha)
        # See the module docstring: False switches this entire extractor to
        # _DefaultRPTEFeatureExtractor (a plain DecisionTreeClassifier-per-
        # round ensemble with no augmented-pair/microtree machinery at all),
        # not "the bounded-lookahead tree with lookahead skipped".
        self.enable_lookahead = enable_lookahead
        # Deliberately its own threshold, not reused from
        # greedy_stall_probe_ig (a per-leaf, probe-set-based bar for a
        # different decision). This one is a single upfront, whole-fit,
        # grow-set-based bar: across this project's benchmark datasets,
        # only a genuinely zero-marginal-effect target (pure parity)
        # clearly benefits from lookahead.
        self.adaptive_marginal_gain_threshold = float(adaptive_marginal_gain_threshold)
        self.leaf_budget = LEAF_CONFIGS[leaf_config] * self.depth
        self.depth_cap = 2 * self.depth
        self.feature_names_: list[str] = []
        self.augmented_catalog_: list[dict[str, Any]] = []
        self.pattern_provenance_: dict[str, tuple[str, ...]] = {}
        self._default_fe: _DefaultRPTEFeatureExtractor | None = None

    def set_hugiml_feature_metadata(
        self,
        feature_names: list[str],
        augmented_catalog: list[dict[str, Any]],
        pattern_provenance: dict[str, tuple[str, ...] | dict[str, Any]] | None = None,
        original_feature_standardization: dict[str, dict[str, Any]] | None = None,
    ):
        self.feature_names_ = list(feature_names)
        self.augmented_catalog_ = [dict(item) for item in augmented_catalog]
        # Raw-feature provenance for mined HUG pattern columns -- see
        # HUGIMLClassifierNative.get_pattern_provenance(). Used below to
        # reserve a pattern's actual constituent raw features (not just
        # the pattern's own column name), so a later tree can't sidestep
        # reservation by reusing a different pattern built from the same
        # underlying raw feature.
        self.pattern_provenance_ = dict(pattern_provenance or {})
        # Standardization (mean, scale) and median-imputation value per raw
        # original feature. Used by rule_table()/unified_rule_table() to
        # render exact raw-space thresholds instead of leaving every split
        # in opaque standardized units.
        self.original_feature_standardization_ = dict(original_feature_standardization or {})
        return self

    @staticmethod
    def _dense(X) -> np.ndarray:
        if sparse.issparse(X):
            return np.asarray(X.toarray(), dtype=np.float64)
        return np.asarray(X, dtype=np.float64)

    def _metadata(self, n_cols: int):
        """Resolves HUGIML's downstream column catalog into the raw-
        feature-index form the native engine needs, plus the original
        name-based ownership map _DefaultRPTEFeatureExtractor still uses
        (kept for the fallback chain, which is unrelated to the native
        lookahead engine).

        Returns (names, raw_indices, roots, owner_by_column_names,
        reservable_names, owner_by_column_native, initial_available_native,
        raw_name_by_index):
          names: display name per downstream column.
          raw_indices: column indices whose name starts with "orig:".
          roots: (index, name, (source_a_name, source_b_name), operation)
            for every column usable as a lookahead root -- an existing
            HUGIML augmented pair, or a 2-raw-feature mined pattern.
          owner_by_column_names / reservable_names: name-keyed ownership,
            for _DefaultRPTEFeatureExtractor's own reservation logic.
          owner_by_column_native: raw-column-index ownership per column,
            for the native engine -- a column with no resolvable raw
            provenance owns itself (its own index), so it still gets
            reserved exactly once even though it can't be tied to a
            specific raw feature.
          initial_available_native: every ownership token that appears in
            owner_by_column_native, i.e. the reservation pool a fit starts
            with.
        """
        names = (
            list(self.feature_names_)
            if len(self.feature_names_) == n_cols
            else [f"x{j}" for j in range(n_cols)]
        )
        raw_indices = [i for i, name in enumerate(names) if str(name).startswith("orig:")]
        raw_index_set = set(raw_indices)
        raw_name_by_index = {i: str(names[i]).removeprefix("orig:") for i in raw_indices}
        raw_index_by_name = {name: i for i, name in raw_name_by_index.items()}

        owner_by_column_names: dict[int, frozenset[str]] = {
            i: frozenset({raw_name_by_index[i]}) for i in raw_indices
        }
        owner_by_column_native: list[list[int]] = [[] for _ in range(n_cols)]
        for i in raw_indices:
            owner_by_column_native[i] = [i]

        catalog_by_name = {
            f"augmented_pair:{item.get('name')}": item for item in self.augmented_catalog_
        }
        roots: list[tuple[int, str, tuple[str, str], str]] = []
        reservable_names = set(raw_name_by_index.values())

        for idx, name in enumerate(names):
            if idx in raw_index_set:
                continue
            item = catalog_by_name.get(str(name))
            if item is not None:
                inputs = tuple(map(str, item.get("inputs", [])))
                if len(inputs) == 2:
                    operation = str(item.get("operation", ""))
                    owner_by_column_names[idx] = frozenset(inputs)
                    resolved = [raw_index_by_name[s] for s in inputs if s in raw_index_by_name]
                    owner_by_column_native[idx] = sorted(set(resolved)) if len(resolved) == 2 else [idx]
                    roots.append((idx, str(name), (inputs[0], inputs[1]), operation))
                    continue
            raw_entry = self.pattern_provenance_.get(str(name))
            if isinstance(raw_entry, dict):
                pattern_sources = tuple(str(f) for f in raw_entry.get("raw_features", ()))
            elif raw_entry:
                pattern_sources = tuple(str(f) for f in raw_entry)
            else:
                pattern_sources = ()
            if pattern_sources:
                owner_by_column_names[idx] = frozenset(pattern_sources)
                reservable_names.update(pattern_sources)
                resolved = [raw_index_by_name[s] for s in pattern_sources if s in raw_index_by_name]
                owner_by_column_native[idx] = (
                    sorted(set(resolved)) if len(resolved) == len(pattern_sources) else [idx]
                )
                if len(pattern_sources) == 2:
                    roots.append((idx, str(name), (pattern_sources[0], pattern_sources[1]), "pattern"))
                continue
            own_name = str(name)
            owner_by_column_names.setdefault(idx, frozenset({own_name}))
            reservable_names.add(own_name)
            owner_by_column_native[idx] = [idx]

        initial_available_native: set[int] = set()
        for owners in owner_by_column_native:
            initial_available_native.update(owners)

        return (
            names,
            raw_indices,
            roots,
            owner_by_column_names,
            reservable_names,
            owner_by_column_native,
            initial_available_native,
            raw_name_by_index,
        )

    def _fit_default_backend(
        self,
        Xd: np.ndarray,
        y: np.ndarray,
        reason: str,
        owner_by_column: dict[int, frozenset[str]] | None = None,
    ) -> sparse.csr_matrix:
        """Dispatch to _DefaultRPTEFeatureExtractor (sequential RPTE, no
        augmented-pair/microtree machinery) and record why -- shared by all
        three routes that can lead here: enable_lookahead=False outright,
        "adaptive" finding strong-enough marginal signal not to bother, no
        augmented pairs to build a microtree root from in the first place
        (HUGIML's augmented_pair_transforms=False, or a mining config that
        simply didn't retain any), and the bounded search genuinely finding
        no structure at all (see fit_leaves' degenerate handling).

        Never raises. This is the full fallback chain, in order:
          1. Sequential RPTE (_DefaultRPTEFeatureExtractor) -- an ordinary
             greedy tree ensemble might still find SOMETHING even where
             bounded lookahead's stricter, more specific criteria found
             nothing. Reserves by raw-feature NAME across its own trees
             when `owner_by_column` is supplied -- the same ownership model the bounded-lookahead backend
             uses, rather than merely reserving whole downstream columns
             (which would let, say, orig:age and pattern:age=[50,60) be
             claimed by two different trees despite sharing a raw source).
          2. If sequential RPTE ALSO can't find a single valid split
             (raises ValueError -- happens on genuinely tiny samples,
             e.g. Longley's n=16, where min_samples_leaf/min_probe_leaf
             routinely can't be met at all): skip tree-based feature
             construction entirely and hand the downstream logistic layer
             HUGIML's own mined features directly -- exactly what
             HUGIML-LR (no RPTE at all) already uses. A real,
             still-informative feature matrix, not a placeholder, and
             this step itself essentially cannot fail: it doesn't require
             finding any particular structure, just handing the matrix
             HUGIML already built off to LogisticRegression.
          3. Belt-and-suspenders: if constructing even that sparse matrix
             somehow raises (Xd malformed in some way neither of the
             above anticipated), fall back to a single constant column --
             LogisticRegression can only fit an intercept on it, i.e.
             predict the class prior for every row. Legitimately bad
             (low-AUC) but always a VALID, fittable model; the point
             throughout this chain is that fit_leaves() never raises an
             exception a caller has to handle, however little signal
             ends up actually usable.

        Caution specifically for the "no augmented pairs" and "sequential
        RPTE also failed" routes: sequential RPTE is the best available
        option there, not a reliable substitute for lookahead on a
        genuinely zero-marginal-effect target. On a target like pure 3-way
        parity, a lucky seed can let the default backend score well by
        accident: with enough leaves (leaf_config/depth) and a large enough
        raw-feature pool, a plain greedy tree can occasionally stumble onto
        splitting on exactly the right combination of features across its
        many nodes (helped along by this class's own per-tree column
        reservation forcing later rounds onto different features than
        earlier ones), and with enough leaves can then effectively memorize
        the resulting fine partition -- but this is unreliable across
        seeds, unlike lookahead's targeted, statistically-validated
        root+child search. These routes exist because lookahead genuinely
        cannot do its job in these situations (no augmented pairs to build
        a root from; too few samples to support even one statistically-
        defensible split), not because the fallbacks are being claimed as
        adequate replacements for it.
        """
        try:
            self._default_fe = _DefaultRPTEFeatureExtractor(
                leaf_config=self.leaf_config,
                depth=self.depth,
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                owner_by_column=owner_by_column,
                reserve_raw_features=self.reserve_raw_features,
            )
            leaves = self._default_fe.fit_leaves(Xd, y)
            self.is_degenerate_ = False
            self.trees_ = []
            self.default_backend_reason_ = reason
            self._raw_feature_fallback_ = False
            return leaves
        except ValueError:
            pass
        try:
            self._default_fe = None
            leaves = sparse.csr_matrix(Xd)
            self.is_degenerate_ = True
            self.trees_ = []
            self.default_backend_reason_ = f"{reason}__sequential_also_failed__raw_hugiml_features"
            self._raw_feature_fallback_ = True
            return leaves
        except Exception:
            self._default_fe = None
            self.is_degenerate_ = True
            self.trees_ = []
            self.default_backend_reason_ = f"{reason}__sequential_also_failed__constant_column"
            self._raw_feature_fallback_ = False
            return sparse.csr_matrix(np.ones((Xd.shape[0], 1), dtype=np.float32))


    def fit_leaves(self, X, y) -> sparse.csr_matrix:
        if self.enable_lookahead not in (True, False, "adaptive"):
            raise ValueError(
                f"enable_lookahead must be True, False, or 'adaptive', got {self.enable_lookahead!r}."
            )
        Xd = self._dense(X)
        y = np.asarray(y)
        y01 = (y == np.max(y)).astype(np.int8)
        (
            names,
            raw_indices,
            roots,
            owner_by_column_names,
            available_names,
            owner_by_column_native,
            available_native,
            raw_name_by_index,
        ) = self._metadata(Xd.shape[1])
        self.default_backend_reason_: str | None = None
        self._raw_feature_fallback_ = False

        if not roots:
            # No augmented pairs at all to build a microtree root from --
            # see _fit_default_backend's docstring. Unconditional: even
            # enable_lookahead=True can't do its intended job here.
            self.remaining_raw_features_ = set(available_names)
            return self._fit_default_backend(Xd, y, "no_augmented_pairs", owner_by_column_names)

        use_lookahead = self.enable_lookahead
        if use_lookahead == "adaptive":
            # See _max_marginal_gain's docstring: engage the (slower)
            # bounded-lookahead mechanism only when no single raw feature
            # has real marginal signal on its own -- exactly the regime an
            # ordinary greedy tree (the default backend) cannot get
            # started in at all.
            self.adaptive_marginal_gain_ = _max_marginal_gain(
                Xd, y01, raw_indices, self.min_samples_leaf
            )
            use_lookahead = self.adaptive_marginal_gain_ < self.adaptive_marginal_gain_threshold
            self.adaptive_used_lookahead_ = use_lookahead

        if not use_lookahead:
            return self._fit_default_backend(
                Xd, y, "enable_lookahead_false_or_adaptive_strong_marginal_signal", owner_by_column_names
            )

        self._default_fe = None
        self.trees_: list[dict[str, Any]] = []
        self.tree_leaf_ids_: list[np.ndarray] = []
        self.tree_leaf_values_: list[dict[int, float]] = []
        self.tree_residual_gains_: list[float] = []
        self.reserved_raw_features_: list[set[str]] = []
        self._tree_names_: list[list[str]] = []
        available_raw = set(available_native)

        allowed_ops = None if self.lookahead_ops is None else set(self.lookahead_ops)
        eligible_root_columns_all = [
            idx for idx, _, _, op in roots if allowed_ops is None or op in allowed_ops
        ]
        synth_op_names = allowed_ops if allowed_ops is not None else set(_PAIR_OP_CODES)
        lookahead_op_codes = np.array(
            sorted({_PAIR_OP_CODES[op] for op in synth_op_names if op in _PAIR_OP_CODES}),
            dtype=np.int32,
        )

        def grow_round(round_index, r, w, labels):
            # A column is eligible for this tree iff it isn't owned by any
            # raw feature (or opaque self-token) a PREVIOUS tree already
            # used. Once used, permanent for the rest of this fit.
            eligible_columns = [
                idx
                for idx in range(Xd.shape[1])
                if not owner_by_column_native[idx] or set(owner_by_column_native[idx]).issubset(available_raw)
            ]
            if not eligible_columns:
                return None
            eligible_set = set(eligible_columns)
            eligible_raw = [idx for idx in raw_indices if idx in eligible_set]
            eligible_roots = [idx for idx in eligible_root_columns_all if idx in eligible_set]
            indices = np.arange(labels.size)
            try:
                grow_rows, probe_rows = train_test_split(
                    indices,
                    test_size=self.lookahead_probe_fraction,
                    stratify=labels,
                    random_state=self.random_state + round_index * 1009,
                )
            except ValueError:
                return None

            tree_dict = _native.rpte_grow_tree(
                Xd,
                labels,
                grow_rows.astype(np.int64),
                probe_rows.astype(np.int64),
                np.array(eligible_columns, dtype=np.int32),
                np.array(eligible_raw, dtype=np.int32),
                np.array(eligible_roots, dtype=np.int32),
                owner_by_column_native,
                self.depth_cap,
                self.leaf_budget,
                self.min_samples_leaf,
                self.min_probe_leaf,
                self.random_state + round_index * 7919,
                self.greedy_stall_probe_ig,
                self.lookahead_min_probe_ig,
                self.lookahead_min_increment,
                self.lookahead_beam_width,
                self.lookahead_child_mode == "shared",
                lookahead_op_codes,
                self.max_root_thresholds,
                self.min_weighted_probe_gain,
                self.raw_pair_fallback,
                self.raw_pair_max_candidates,
                self.aug_child_enabled,
                self.aug_child_max_candidates,
                self.use_statistical_acceptance,
                self.significance_alpha,
            )
            if not tree_dict["is_fitted"]:
                return None
            leaf_ids = tree_dict["train_leaf_ids"]

            def on_accept(eta, values, relative_decrease):
                nonlocal available_raw
                self.trees_.append(tree_dict)
                self.tree_leaf_ids_.append(np.array(sorted({int(x) for x in leaf_ids}), dtype=np.int64))
                self.tree_leaf_values_.append(
                    {leaf: eta * value for leaf, value in values.items()}
                )
                self.tree_residual_gains_.append(relative_decrease)
                used_raw = {int(x) for x in tree_dict["used_raw_features"]}
                self.reserved_raw_features_.append(used_raw)
                if self.reserve_raw_features:
                    available_raw -= used_raw
                self._tree_names_.append(_native_tree_column_names(tree_dict, names))

            return _BoostingRound(leaf_ids=leaf_ids, on_accept=on_accept)

        _F, self.deviance_history_, self.base_log_odds_ = _fit_newton_boosted_ensemble(
            y01,
            self.n_estimators,
            self.learning_rate,
            self.min_tree_residual_gain,
            grow_round,
            stop_after_accept=lambda: self.reserve_raw_features and not available_raw,
        )

        self.is_degenerate_ = not self.trees_
        if self.is_degenerate_:
            # No tree found even one valid split -- greedy stalled
            # everywhere and no candidate augmented-pair root cleared the
            # lookahead acceptance bar either. _fit_default_backend never
            # raises -- it tries sequential RPTE first, then degrades
            # further on its own if even that fails.
            self.remaining_raw_features_ = set(available_names)
            return self._fit_default_backend(Xd, y, "lookahead_degenerate", owner_by_column_names)
        self.remaining_raw_features_ = {
            raw_name_by_index.get(tok, names[tok] if tok < len(names) else str(tok))
            for tok in available_raw
        }
        return self._leaf_matrix(Xd)

    def _leaf_matrix(self, Xd: np.ndarray) -> sparse.csr_matrix:
        blocks = []
        for tree_dict, vocab in zip(self.trees_, self.tree_leaf_ids_):
            assign = _native.rpte_apply_tree(Xd, tree_dict)
            blocks.append(sparse.csr_matrix((assign[:, None] == vocab[None, :]).astype(np.float32)))
        return sparse.hstack(blocks, format="csr")

    def transform_leaves(self, X) -> sparse.csr_matrix:
        if self._default_fe is not None:
            return self._default_fe.transform_leaves(X)
        if not hasattr(self, "is_degenerate_"):
            raise ValueError("Extractor has not been fitted.")
        Xd = self._dense(X)
        if self.is_degenerate_:
            if getattr(self, "_raw_feature_fallback_", False):
                # No tree-based feature construction at all at fit time:
                # HUGIML's own downstream matrix handed directly to the
                # logistic layer. Reconstruct the same thing here.
                return sparse.csr_matrix(Xd)
            return sparse.csr_matrix(np.ones((Xd.shape[0], 1), dtype=np.float32))
        return self._leaf_matrix(Xd)

    def growth_summary(self) -> list[dict[str, Any]]:
        """Only meaningful when this fit actually used the bounded-
        lookahead mechanism -- returns [] when it used the default backend
        instead. Use rule_table() instead there."""
        if self._default_fe is not None:
            return []
        rows: list[dict[str, Any]] = []
        for tree_index, tree_dict in enumerate(self.trees_):
            tree_names = self._tree_names_[tree_index]
            for entry in tree_dict["growth_log"]:
                op_name = _PAIR_OP_NAMES.get(entry["operation"])
                source_features = [
                    tree_names[i] if i < len(tree_names) else f"col{i}" for i in entry["source_features"]
                ]
                rows.append(
                    {
                        "tree": tree_index,
                        "step": entry["step"],
                        "strategy": "leaf_wise_best_first",
                        "leaf_id": entry["leaf_id"],
                        "leaf_depth": entry["leaf_depth"],
                        "kind": entry["kind"],
                        "root_name": tree_names[entry["root_index"]]
                        if entry["root_index"] < len(tree_names)
                        else f"col{entry['root_index']}",
                        "operation": op_name,
                        "source_features": source_features,
                        "left_child_name": (
                            tree_names[entry["left_child_index"]]
                            if entry["left_child_index"] is not None
                            else None
                        ),
                        "right_child_name": (
                            tree_names[entry["right_child_index"]]
                            if entry["right_child_index"] is not None
                            else None
                        ),
                        "grandchild_name": (
                            tree_names[entry["grandchild_index"]]
                            if entry["grandchild_index"] is not None
                            else None
                        ),
                        "grow_gain": entry["grow_gain"],
                        "probe_gain": entry["probe_gain"],
                        "root_probe_gain": entry["root_probe_gain"],
                        "weighted_probe_gain": entry["weighted_probe_gain"],
                        "statistically_significant": entry["statistically_significant"],
                        "significance_p_value": entry["significance_p_value"],
                        "leaf_count_before": entry["leaf_count_before"],
                        "leaf_count_after": entry["leaf_count_after"],
                    }
                )
        return rows

    def rule_table(
        self, coefficients, feature_names: list[str] | None = None
    ) -> list[dict[str, object]]:
        """Default-mode only (see growth_summary's docstring for the
        complementary bounded-lookahead-mode diagnostic)."""
        if self._default_fe is None:
            raise NotImplementedError(
                "rule_table() is only available when this fit used the default "
                "backend (enable_lookahead=False, or 'adaptive' and marginal "
                "signal was strong enough not to engage lookahead). Use "
                "growth_summary() for the bounded-lookahead mechanism's "
                "root/child microtree structure instead."
            )
        return self._default_fe.rule_table(coefficients, feature_names=feature_names)

class LeafWiseBoundedLookaheadRPTEFeatureLR(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible leaf-wise bounded-look-ahead RPTE + logistic model."""

    def __init__(
        self,
        leaf_config: str = "3xD",
        depth: int = 4,
        n_estimators: int = 10,
        min_samples_leaf: int = 5,
        rpte_learning_rate: float = 0.3,
        lr_C: float = 1.0,
        lr_penalty: str = "l1",
        random_state: int = 42,
        lookahead_child_mode: str = "shared",
        lookahead_ops: tuple[str, ...] | None = ("absolute_difference",),
        lookahead_beam_width: int = 64,
        lookahead_probe_fraction: float = 0.25,
        lookahead_min_probe_ig: float = 0.05,
        lookahead_min_increment: float = 0.03,
        greedy_stall_probe_ig: float = 0.02,
        max_root_thresholds: int = 7,
        min_probe_leaf: int = 2,
        min_weighted_probe_gain: float = 0.01,
        reserve_raw_features: bool = True,
        min_tree_residual_gain: float = 1e-8,
        raw_pair_fallback: bool = True,
        raw_pair_max_candidates: int = 400,
        aug_child_enabled: bool = True,
        aug_child_max_candidates: int = 100,
        use_statistical_acceptance: bool = True,
        significance_alpha: float = 0.05,
        enable_lookahead: bool | str = "adaptive",
        adaptive_marginal_gain_threshold: float = 0.005,
        hugiml_feature_names: list[str] | None = None,
        hugiml_augmented_catalog: list[dict[str, Any]] | None = None,
        hugiml_pattern_provenance: dict[str, tuple[str, ...] | dict[str, Any]] | None = None,
        hugiml_original_feature_standardization: dict[str, dict[str, Any]] | None = None,
    ):
        self.leaf_config = leaf_config
        self.depth = depth
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.rpte_learning_rate = rpte_learning_rate
        self.lr_C = lr_C
        self.lr_penalty = lr_penalty
        self.random_state = random_state
        self.lookahead_child_mode = lookahead_child_mode
        self.lookahead_ops = lookahead_ops
        self.lookahead_beam_width = lookahead_beam_width
        self.lookahead_probe_fraction = lookahead_probe_fraction
        self.lookahead_min_probe_ig = lookahead_min_probe_ig
        self.lookahead_min_increment = lookahead_min_increment
        self.greedy_stall_probe_ig = greedy_stall_probe_ig
        self.max_root_thresholds = max_root_thresholds
        self.min_probe_leaf = min_probe_leaf
        self.min_weighted_probe_gain = min_weighted_probe_gain
        self.reserve_raw_features = reserve_raw_features
        self.min_tree_residual_gain = min_tree_residual_gain
        self.raw_pair_fallback = raw_pair_fallback
        self.raw_pair_max_candidates = raw_pair_max_candidates
        # "Aug child": lets a microtree's CHILD be a pair (existing HUGIML
        # augmented pair or synthesized raw pair), not just a single raw
        # feature -- root pair + child pair together span 4 raw features,
        # extending 3-way detection to 4-way.
        self.aug_child_enabled = aug_child_enabled
        self.aug_child_max_candidates = aug_child_max_candidates
        # Statistical acceptance calibration -- see the module note above
        # _g2_statistic in rpte_bounded_lookahead_leafwise.py.
        self.use_statistical_acceptance = use_statistical_acceptance
        self.significance_alpha = significance_alpha
        # True: always the bounded-lookahead mechanism. False: always the
        # plain default backend (_DefaultRPTEFeatureExtractor). "adaptive":
        # decide per fit from the data itself -- engage lookahead only when
        # no single raw feature has real marginal signal on its own (see
        # _max_marginal_gain's docstring), otherwise use the default
        # backend. See the module docstring and
        # LeafWiseBoundedLookaheadRPTEFeatureExtractor.fit_leaves.
        self.enable_lookahead = enable_lookahead
        self.adaptive_marginal_gain_threshold = adaptive_marginal_gain_threshold
        # Real, sklearn-visible constructor parameters -- deliberately NOT
        # private/underscore-prefixed instance state set only by the
        # set_hugiml_feature_metadata() convenience method below. This
        # class is regularly used as a base_estimator inside sklearn
        # meta-estimators (e.g. OneVsRestClassifier), which construct
        # per-class copies via sklearn.base.clone() -- clone() reconstructs
        # a fresh instance from get_params() (i.e. whatever __init__
        # accepts), it does NOT copy arbitrary attributes set by a method
        # call after construction. Metadata set only via
        # set_hugiml_feature_metadata() before wrapping/fitting would
        # silently vanish from the actual fitted per-class estimator(s)
        # the first time such a wrapper clones this instance -- verified
        # directly: OneVsRestClassifier.fit() with metadata set only the
        # old (private-attribute) way left every fitted sub-estimator with
        # feature_names_/augmented_catalog_ back at their [] defaults, so
        # RPTE silently fell back to having no augmented-pair root
        # candidates at all. Exposing these as real constructor parameters
        # means clone() carries the CURRENT value across automatically,
        # the same as any other hyperparameter.
        self.hugiml_feature_names = hugiml_feature_names
        self.hugiml_augmented_catalog = hugiml_augmented_catalog
        self.hugiml_pattern_provenance = hugiml_pattern_provenance
        # Standardization (mean, scale) and median-imputation value per raw
        # original feature -- used for exact threshold rendering
        # (interpretability: converting a standardized split
        # like "orig:age > -0.015" into "age > 48.85 years") and missing-
        # value disclosure. Same clone-safety rationale as
        # the three metadata channels above.
        self.hugiml_original_feature_standardization = hugiml_original_feature_standardization

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def set_hugiml_feature_metadata(
        self,
        feature_names: list[str],
        augmented_catalog: list[dict[str, Any]],
        pattern_provenance: dict[str, tuple[str, ...] | dict[str, Any]] | None = None,
        original_feature_standardization: dict[str, dict[str, Any]] | None = None,
    ):
        """Convenience setter -- equivalent to passing the same four
        values as constructor arguments (hugiml_feature_names /
        hugiml_augmented_catalog / hugiml_pattern_provenance /
        hugiml_original_feature_standardization), which is the
        clone()-safe way to do it (see their docstring above). This
        setter is still useful for callers that already have a
        constructed instance in hand (e.g. HUGIMLRPTEHybrid, which fits
        this class directly rather than through a cloning meta-estimator,
        so clone-safety isn't a concern there) and don't want to
        reconstruct it just to attach metadata.
        """
        self.hugiml_feature_names = list(feature_names)
        self.hugiml_augmented_catalog = [dict(item) for item in augmented_catalog]
        self.hugiml_pattern_provenance = dict(pattern_provenance or {})
        self.hugiml_original_feature_standardization = dict(original_feature_standardization or {})
        return self

    @staticmethod
    def _as_csr(X) -> sparse.csr_matrix:
        """Return the supplied downstream matrix as CSR without changing columns."""
        return X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)

    @staticmethod
    def _tree_used_input_columns(
        fe: LeafWiseBoundedLookaheadRPTEFeatureExtractor,
        n_input_features: int,
    ) -> set[int]:
        """Input-column indices selected in at least one accepted RPTE split.

        Native bounded-lookahead trees report their supplied-column indices in
        ``used_columns``. The sequential sklearn backend stores each tree's
        local-to-global column map in ``tree_columns_``. RPTE-synthesized pair
        columns are intentionally ignored because they were not part of the
        supplied HUGIML downstream matrix.
        """
        if bool(getattr(fe, "_raw_feature_fallback_", False)):
            # In this fallback the extractor's "leaf" representation is X
            # itself, so appending direct source inputs would duplicate every column.
            return set(range(n_input_features))

        default_fe = getattr(fe, "_default_fe", None)
        if default_fe is not None:
            used: set[int] = set()
            for tree, candidate_columns in zip(
                default_fe.trees_, default_fe.tree_columns_
            ):
                for local_idx in tree.tree_.feature:
                    local_idx = int(local_idx)
                    if 0 <= local_idx < len(candidate_columns):
                        global_idx = int(candidate_columns[local_idx])
                        if 0 <= global_idx < n_input_features:
                            used.add(global_idx)
            return used

        used = set()
        for tree_dict in getattr(fe, "trees_", []) or []:
            for idx in tree_dict.get("used_columns", []) or []:
                idx = int(idx)
                if 0 <= idx < n_input_features:
                    used.add(idx)
        return used

    def _fit_final_feature_layout(
        self,
        X_csr: sparse.csr_matrix,
        leaves: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """Record the leaf/direct partition and build the final LR matrix."""
        self.n_input_features_ = int(X_csr.shape[1])
        self.n_features_in_ = self.n_input_features_
        self.n_leaf_features_ = int(leaves.shape[1])

        used = self._tree_used_input_columns(self.fe_, self.n_input_features_)
        self.tree_used_input_indices_ = np.asarray(sorted(used), dtype=np.int64)
        direct = sorted(set(range(self.n_input_features_)) - used)
        self.direct_input_indices_ = np.asarray(direct, dtype=np.int64)

        if self.direct_input_indices_.size:
            final_X = sparse.hstack(
                [leaves, X_csr[:, self.direct_input_indices_]], format="csr"
            )
        else:
            final_X = leaves.tocsr()
        self.n_final_lr_features_ = int(final_X.shape[1])
        return final_X

    def _final_lr_matrix(self, X) -> sparse.csr_matrix:
        check_is_fitted(
            self, ["fe_", "logistic_", "n_input_features_", "direct_input_indices_"]
        )
        X_csr = self._as_csr(X)
        leaves = self.fe_.transform_leaves(X_csr)
        if X_csr.shape[1] != self.n_input_features_:
            raise ValueError(
                f"X has {X_csr.shape[1]} downstream columns; expected "
                f"{self.n_input_features_}."
            )
        direct_indices = np.asarray(self.direct_input_indices_, dtype=np.int64)
        if direct_indices.size:
            return sparse.hstack(
                [leaves, X_csr[:, direct_indices]], format="csr"
            )
        return leaves.tocsr()

    def leaf_coefficients(self) -> np.ndarray:
        """Coefficients assigned to RPTE leaf-indicator columns."""
        check_is_fitted(self, "logistic_")
        n_leaf = int(getattr(self, "n_leaf_features_", self.logistic_.coef_.shape[1]))
        return np.asarray(self.logistic_.coef_[0, :n_leaf], dtype=float)

    def direct_input_coefficients(self) -> np.ndarray:
        """Coefficients assigned to direct source columns after the leaf block.

        These are supplied HUGIML downstream columns that were not selected in
        an accepted RPTE split and therefore remain standalone terms in the
        final logistic regression.
        """
        check_is_fitted(self, "logistic_")
        n_leaf = int(getattr(self, "n_leaf_features_", self.logistic_.coef_.shape[1]))
        return np.asarray(self.logistic_.coef_[0, n_leaf:], dtype=float)

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("LeafWiseBoundedLookaheadRPTEFeatureLR is binary only.")
        self.fe_ = LeafWiseBoundedLookaheadRPTEFeatureExtractor(
            leaf_config=self.leaf_config,
            depth=self.depth,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            learning_rate=self.rpte_learning_rate,
            random_state=self.random_state,
            lookahead_child_mode=self.lookahead_child_mode,
            lookahead_ops=self.lookahead_ops,
            lookahead_beam_width=self.lookahead_beam_width,
            lookahead_probe_fraction=self.lookahead_probe_fraction,
            lookahead_min_probe_ig=self.lookahead_min_probe_ig,
            lookahead_min_increment=self.lookahead_min_increment,
            greedy_stall_probe_ig=self.greedy_stall_probe_ig,
            max_root_thresholds=self.max_root_thresholds,
            min_probe_leaf=self.min_probe_leaf,
            min_weighted_probe_gain=self.min_weighted_probe_gain,
            reserve_raw_features=self.reserve_raw_features,
            min_tree_residual_gain=self.min_tree_residual_gain,
            raw_pair_fallback=self.raw_pair_fallback,
            raw_pair_max_candidates=self.raw_pair_max_candidates,
            aug_child_enabled=self.aug_child_enabled,
            aug_child_max_candidates=self.aug_child_max_candidates,
            use_statistical_acceptance=self.use_statistical_acceptance,
            significance_alpha=self.significance_alpha,
            enable_lookahead=self.enable_lookahead,
            adaptive_marginal_gain_threshold=self.adaptive_marginal_gain_threshold,
        )
        self.fe_.set_hugiml_feature_metadata(
            self.hugiml_feature_names or [],
            self.hugiml_augmented_catalog or [],
            self.hugiml_pattern_provenance or {},
            self.hugiml_original_feature_standardization or {},
        )
        X_csr = self._as_csr(X)
        leaves = self.fe_.fit_leaves(X_csr, y)
        final_X = self._fit_final_feature_layout(X_csr, leaves)
        self.logistic_ = _liblinear_logistic_regression(
            lr_penalty=self.lr_penalty,
            lr_C=self.lr_C,
            random_state=self.random_state,
        )
        self.logistic_.fit(final_X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "logistic_")
        return self.logistic_.predict_proba(self._final_lr_matrix(X))

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def rule_table(self, feature_names: list[str] | None = None) -> list[dict[str, object]]:
        """Human-readable root-to-leaf conjunctions for the default backend.

        Direct source terms are intentionally not returned by this rules-only
        method. Use ``unified_rule_table()`` to obtain both leaf
        rules and direct source terms.
        """
        check_is_fitted(self, "logistic_")
        return self.fe_.rule_table(self.leaf_coefficients(), feature_names=feature_names)


    def unified_rule_table(self, feature_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Return the fitted prediction explanation surface.

        The result uses one schema for bounded-lookahead trees, sequential
        trees, direct supplied features, raw-feature fallbacks, and constant
        fallbacks. Callers therefore do not need to branch on the fitted tree
        backend. ``growth_summary()`` remains a separate algorithm-audit view
        of how a bounded-lookahead tree was constructed.

        Each returned row describes one active terminal leaf or direct term.
        Tree rows include the class label, tree and leaf indices, backend,
        structured split conditions, raw source lineage, training support,
        downstream logistic coefficient, centered contribution, and Newton
        leaf value. Direct-term rows identify supplied features that were not
        used in an accepted tree split and were retained by the final sparse
        logistic layer.

        ``centered_tree_contribution`` expresses a leaf coefficient relative
        to its tree's support-weighted coefficient baseline. This avoids
        treating the redundant one-hot leaf parameterization as an absolute
        importance scale. ``newton_leaf_value`` is a tree-construction
        diagnostic and is not the final prediction effect.

        Returns
        -------
        list of dict
            Structured, backend-independent explanation rows.

        Notes
        -----
        Per-leaf lookahead probe gain is not currently joined into this table.
        Use ``growth_summary()`` to inspect the accepted split-level probe
        evidence for bounded-lookahead fits.
        """
        check_is_fitted(self, "logistic_")
        fe = self.fe_
        names = (
            feature_names if feature_names is not None else list(self.hugiml_feature_names or [])
        )
        all_coef = np.asarray(self.logistic_.coef_[0], dtype=float)
        n_leaf_features = int(getattr(self, "n_leaf_features_", all_coef.size))
        direct_input_indices = np.asarray(self.direct_input_indices_, dtype=np.int64)
        coef = all_coef[:n_leaf_features]
        direct_coef = all_coef[n_leaf_features:]
        positive_class = self.classes_[1] if len(self.classes_) > 1 else self.classes_[0]
        original_std = self.hugiml_original_feature_standardization or {}
        augmented_catalog_by_name = {
            f"augmented_pair:{item.get('name')}": item
            for item in (self.hugiml_augmented_catalog or [])
        }
        pattern_prov = self.hugiml_pattern_provenance or {}
        fallback_status = getattr(fe, "default_backend_reason_", None)

        def _render_conditions(
            raw_conditions: list[tuple[str, float, bool]],
        ) -> list[dict[str, Any]]:
            rendered = []
            for name, threshold, is_right in raw_conditions:
                info = dict(
                    _render_threshold(
                        name,
                        threshold,
                        original_std,
                        augmented_catalog_by_name,
                        pattern_prov,
                        is_right=is_right,
                    )
                )
                info["direction"] = "above_threshold" if is_right else "at_or_below_threshold"
                info["standardized_threshold"] = float(threshold)
                rendered.append(info)
            return rendered

        rows: list[dict[str, Any]] = []

        def _append_direct_input_rows() -> None:
            for input_idx, coefficient in zip(direct_input_indices, direct_coef):
                coefficient = float(coefficient)
                if coefficient == 0.0:
                    continue
                input_idx = int(input_idx)
                name = names[input_idx] if input_idx < len(names) else f"col{input_idx}"
                condition = _render_linear_term(
                    name, original_std, augmented_catalog_by_name, pattern_prov
                )
                rows.append(
                    {
                        "class": positive_class,
                        "tree_index": None,
                        "leaf_index": input_idx,
                        "backend": "direct_hugiml_feature",
                        "term_role": "direct_source_term",
                        "source_selection_status": "not_selected_in_tree_split",
                        "conditions": [condition],
                        "raw_conditions": [condition["raw_condition"]]
                        if condition["raw_condition"]
                        else [],
                        "raw_sources": condition["raw_sources"],
                        "support_count": None,
                        "support_rate": None,
                        "final_logistic_coefficient": coefficient,
                        "centered_tree_contribution": None,
                        "newton_leaf_value": None,
                        "fallback_status": fallback_status,
                        "downstream_feature_index": input_idx,
                        "downstream_feature": name,
                    }
                )

        if fe.is_degenerate_:
            if getattr(fe, "_raw_feature_fallback_", False):
                for col_idx, coefficient in enumerate(coef):
                    if coefficient == 0:
                        continue
                    name = names[col_idx] if col_idx < len(names) else f"col{col_idx}"
                    condition = _render_linear_term(
                        name, original_std, augmented_catalog_by_name, pattern_prov
                    )
                    rows.append(
                        {
                            "class": positive_class,
                            "tree_index": None,
                            "leaf_index": col_idx,
                            "backend": "raw_hugiml_features",
                            "conditions": [condition],
                            "raw_conditions": [condition["raw_condition"]]
                            if condition["raw_condition"]
                            else [],
                            "raw_sources": condition["raw_sources"],
                            "support_count": None,
                            "support_rate": None,
                            "final_logistic_coefficient": float(coefficient),
                            "centered_tree_contribution": None,
                            "newton_leaf_value": None,
                            "fallback_status": fallback_status,
                        }
                    )
            else:
                rows.append(
                    {
                        "class": positive_class,
                        "tree_index": None,
                        "leaf_index": None,
                        "backend": "constant",
                        "conditions": [],
                        "raw_conditions": [],
                        "raw_sources": [],
                        "support_count": None,
                        "support_rate": None,
                        "final_logistic_coefficient": float(coef[0]) if coef.size else None,
                        "centered_tree_contribution": None,
                        "newton_leaf_value": None,
                        "fallback_status": fallback_status,
                    }
                )
            _append_direct_input_rows()
            rows.sort(key=lambda row: -abs(row["final_logistic_coefficient"]))
            return rows

        if fe._default_fe is not None:
            default_fe = fe._default_fe
            backend = "sequential_default"
            col_offset = 0
            for tree_index, (tree, columns, vocab) in enumerate(
                zip(default_fe.trees_, default_fe.tree_columns_, default_fe.leaf_vocab_)
            ):
                tree_ = tree.tree_
                tree_names = [names[c] if c < len(names) else f"col{c}" for c in columns]
                newton_values = (
                    default_fe.tree_leaf_values_[tree_index]
                    if tree_index < len(default_fe.tree_leaf_values_)
                    else {}
                )
                n_leaves = len(vocab)
                tree_coefs = coef[col_offset : col_offset + n_leaves]
                supports = [int(tree_.n_node_samples[int(leaf_id)]) for leaf_id in vocab]
                total_support = int(tree_.n_node_samples[0]) or 1
                weighted_avg = (
                    float(np.average(tree_coefs, weights=supports))
                    if n_leaves and sum(supports) > 0
                    else 0.0
                )
                for local_idx, leaf_id in enumerate(vocab):
                    leaf_id = int(leaf_id)
                    coefficient = float(tree_coefs[local_idx])
                    raw_conditions = _simplify_threshold_path(
                        leaf_path_conditions_with_thresholds(tree, leaf_id, tree_names)
                    )
                    simplified_binary = {
                        (name, value)
                        for name, value in simplify_conditions(
                            [(n, "1" if r else "0") for n, _, r in raw_conditions]
                        )
                    }
                    kept = [
                        (n, t, r)
                        for n, t, r in raw_conditions
                        if (n, "1" if r else "0") in simplified_binary
                    ]
                    conditions = _render_conditions(kept)
                    support = supports[local_idx]
                    rows.append(
                        {
                            "class": positive_class,
                            "tree_index": tree_index,
                            "leaf_index": leaf_id,
                            "backend": backend,
                            "conditions": conditions,
                            "raw_conditions": [
                                c["raw_condition"] for c in conditions if c["raw_condition"]
                            ],
                            "raw_sources": sorted(
                                {s for c in conditions for s in c["raw_sources"]}
                            ),
                            "support_count": support,
                            "support_rate": support / total_support,
                            "final_logistic_coefficient": coefficient,
                            "centered_tree_contribution": coefficient - weighted_avg,
                            "newton_leaf_value": newton_values.get(leaf_id),
                            "fallback_status": fallback_status,
                        }
                    )
                col_offset += n_leaves
            _append_direct_input_rows()
            rows.sort(key=lambda row: -abs(row["final_logistic_coefficient"]))
            return rows


        # Bounded-lookahead backend.
        backend = "bounded_lookahead"
        col_offset = 0
        for tree_index, (tree_dict, vocab) in enumerate(zip(fe.trees_, fe.tree_leaf_ids_)):
            leaf_conditions_by_id = _walk_native_tree_leaf_conditions(tree_dict)
            newton_values = (
                fe.tree_leaf_values_[tree_index] if tree_index < len(fe.tree_leaf_values_) else {}
            )
            n_leaves = len(vocab)
            tree_coefs = coef[col_offset : col_offset + n_leaves]
            train_leaf_ids = tree_dict["train_leaf_ids"]
            supports = [int(np.sum(train_leaf_ids == leaf_id)) for leaf_id in vocab]
            total_support = sum(supports) or 1
            weighted_avg = (
                float(np.average(tree_coefs, weights=supports))
                if n_leaves and sum(supports) > 0
                else 0.0
            )
            # tree_names (not the outer `names`/hugiml_feature_names) is
            # authoritative here: synthetic columns the raw-pair-fallback
            # search materializes mid-fit get appended to this tree's own
            # column-name list as they're created (see
            # _native_tree_column_names), so a synthetic root's real name
            # is only ever found there, not in the fixed HUGIML-mining-
            # time name list.
            tree_names = fe._tree_names_[tree_index]
            for local_idx, leaf_id in enumerate(vocab):
                leaf_id = int(leaf_id)
                coefficient = float(tree_coefs[local_idx])
                path = leaf_conditions_by_id.get(leaf_id, [])
                raw_conditions = _simplify_threshold_path(
                    [
                        (
                            tree_names[idx] if idx < len(tree_names) else f"col{idx}",
                            thr,
                            is_right,
                        )
                        for idx, thr, is_right in path
                    ]
                )
                conditions = _render_conditions(raw_conditions)
                support = supports[local_idx]
                rows.append(
                    {
                        "class": positive_class,
                        "tree_index": tree_index,
                        "leaf_index": leaf_id,
                        "backend": backend,
                        "conditions": conditions,
                        "raw_conditions": [
                            c["raw_condition"] for c in conditions if c["raw_condition"]
                        ],
                        "raw_sources": sorted({s for c in conditions for s in c["raw_sources"]}),
                        "support_count": support,
                        "support_rate": support / total_support,
                        "final_logistic_coefficient": coefficient,
                        "centered_tree_contribution": coefficient - weighted_avg,
                        "newton_leaf_value": newton_values.get(leaf_id),
                        "fallback_status": fallback_status,
                    }
                )
            col_offset += n_leaves
        _append_direct_input_rows()
        rows.sort(key=lambda row: -abs(row["final_logistic_coefficient"]))
        return rows

    def unified_rule_tree(
        self,
        feature_names: list[str] | None = None,
        *,
        condition_space: str = "raw",
        detail_level: str = "full",
        precision: int = 5,
        include_direct_terms: bool = True,
        include_generation_details: bool = False,
        class_label: Any | None = None,
        tree_index: int | None = None,
    ) -> str:
        """Return the fitted RPTE representation as ready-to-print flat trees.

        Shared root-to-leaf prefixes are merged into a decision-tree-style text
        layout.  Each terminal leaf includes its final LR coefficient, odds
        multiplier, support, centered contribution, and raw-feature provenance.
        Direct source terms are grouped after the leaf trees by source family.

        ``condition_space`` accepts ``"raw"``, ``"downstream"``, or ``"both"``.
        ``detail_level`` accepts ``"compact"`` or ``"full"``.  The returned
        string can be printed directly or embedded in a report::

            print(rpte.unified_rule_tree())
        """
        from .rpte_interpretability import format_rpte_rule_tree

        return format_rpte_rule_tree(
            self.unified_rule_table(feature_names=feature_names),
            condition_space=condition_space,
            detail_level=detail_level,
            precision=precision,
            include_direct_terms=include_direct_terms,
            include_generation_details=include_generation_details,
            class_label=class_label,
            tree_index=tree_index,
        )


def aggregate_rule_table_by_raw_source(
    rows: list[dict[str, Any]],
    allocation: str = "interaction_only",
) -> dict[str, Any]:
    """Aggregate a unified_rule_table() result by raw-feature provenance, producing two complementary views.

    downstream_contributions: keyed by each condition's own
    family-qualified downstream name (e.g. "orig:age",
    "pattern:age=[50,60)", "augmented_pair:age*income") -- how much each
    ENGINEERED representation of a raw feature contributes, kept separate
    even when several representations share the same underlying raw
    feature(s).

    raw_source_contributions: keyed by raw feature name (single features)
    or a tuple of raw feature names (interactions -- any condition whose
    raw_sources has more than one entry) -- how much each underlying raw
    feature, or interaction between two, contributes in total, aggregated
    ACROSS every downstream representation that touches it.

    `allocation` controls how one condition's coefficient is credited to
    raw_source_contributions when it has more than one raw source:

      "interaction_only" (default): the full coefficient goes to the
        (sourceA, sourceB) INTERACTION entry only -- never independently
        to sourceA's or sourceB's own main-effect entry. This is the safer
        default: crediting the same coefficient to both an interaction AND
        each of its main effects would double- (or triple-) count it and
        overstate every source's apparent importance.
      "equal_split": the coefficient is instead divided evenly across each
        of the k raw sources' own main-effect entries (no separate
        interaction entries at all). Useful when a single, simpler
        per-raw-feature ranking is wanted and the approximation is
        acceptable.

    Every entry in both views tracks `downstream_use_count` (how many
    distinct conditions/leaves touch it) separately from `raw_feature_count`
    (how many distinct raw features are involved in that one entry) --
    conflating the two would make an ensemble look more diverse than it
    really is, since a single raw feature reused across many leaves is not
    the same as many distinct raw features each contributing once.
    """
    if allocation not in ("interaction_only", "equal_split"):
        raise ValueError(
            f"allocation must be 'interaction_only' or 'equal_split', got {allocation!r}."
        )

    downstream: dict[str, dict[str, Any]] = {}
    raw_source: dict[Any, dict[str, Any]] = {}

    for row in rows:
        coefficient = row.get("final_logistic_coefficient")
        if coefficient is None:
            continue
        for condition in row.get("conditions", []):
            name = condition.get("downstream_feature")
            sources = tuple(sorted(set(condition.get("raw_sources", []))))
            if name is not None:
                entry = downstream.setdefault(
                    name,
                    {
                        "downstream_feature": name,
                        "family": condition.get("family"),
                        "raw_sources": list(sources),
                        "total_contribution": 0.0,
                        "downstream_use_count": 0,
                    },
                )
                entry["total_contribution"] += coefficient
                entry["downstream_use_count"] += 1
            if not sources:
                continue
            if len(sources) == 1 or allocation == "interaction_only":
                key = sources[0] if len(sources) == 1 else sources
                entry = raw_source.setdefault(
                    key,
                    {
                        "raw_source": key,
                        "kind": "main_effect" if len(sources) == 1 else "interaction",
                        "total_contribution": 0.0,
                        "downstream_use_count": 0,
                        "raw_feature_count": len(sources),
                    },
                )
                entry["total_contribution"] += coefficient
                entry["downstream_use_count"] += 1
            else:
                share = coefficient / len(sources)
                for source in sources:
                    entry = raw_source.setdefault(
                        source,
                        {
                            "raw_source": source,
                            "kind": "main_effect",
                            "total_contribution": 0.0,
                            "downstream_use_count": 0,
                            "raw_feature_count": 1,
                        },
                    )
                    entry["total_contribution"] += share
                    entry["downstream_use_count"] += 1

    downstream_list = sorted(downstream.values(), key=lambda e: -abs(e["total_contribution"]))
    raw_source_list = sorted(raw_source.values(), key=lambda e: -abs(e["total_contribution"]))
    return {
        "allocation": allocation,
        "downstream_contributions": downstream_list,
        "raw_source_contributions": raw_source_list,
    }


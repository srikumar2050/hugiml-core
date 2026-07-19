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

"""Uniform structural and inspection counts for fitted classifiers.

The public interface uses three coherent levels:

``model units``
    Coarse active components in the complete fitted model, such as terms,
    terminal leaves, or rules.

``model inspection units``
    Expanded evidence required to inspect the complete fitted model, such as
    HUG source elements, all active root-to-leaf conditions, active EBM cells,
    or RuleFit literals.

``instance inspection units``
    Expanded evidence reviewed for one prediction. Tree models count the
    reached contributing path in each tree. HUGIML RPTE adds active direct terms. HUGIML
    linear models count every active original/direct term and only the active
    pattern or augmented-pair terms with non-zero row-specific values.

Calling :func:`get_complexity` without a mode returns model inspection units.
Use :func:`get_instance_inspection_units` to obtain one integer count per row,
or pass X to :func:`get_complexity_report` for the mean and confidence interval.
Optional estimator libraries are recognized through fitted public attributes
and are not imported by this module.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from scipy import sparse
from scipy.stats import t as student_t
from sklearn.utils.validation import check_is_fitted

MODEL_UNITS = "model units"
MODEL_INSPECTION_UNITS = "model inspection units"
INSTANCE_INSPECTION_UNITS = "instance inspection units"
_VALID_MODES = {
    MODEL_UNITS,
    MODEL_INSPECTION_UNITS,
    INSTANCE_INSPECTION_UNITS,
}


def get_complexity(
    model: Any,
    mode: str | None = None,
    *,
    X: Any | None = None,
    coefficient_tolerance: float = 1e-12,
    confidence_level: float = 0.95,
) -> int | float | None:
    """Return one of the three complexity measures for a fitted estimator.

    Parameters
    ----------
    model : Any
        A fitted HUGIML estimator, supported baseline estimator, or a pipeline
        whose final step is a supported estimator.
    mode : {"model units", "model inspection units",
            "instance inspection units"} or None, default=None
        ``None`` selects ``"model inspection units"``.
    X : array-like or DataFrame, optional
        Required for ``"instance inspection units"``. The returned value is
        the arithmetic mean of the row-level counts.
    coefficient_tolerance : float, default=1e-12
        Absolute threshold used to identify active fitted coefficients, active
        leaf outputs, and non-zero row-specific transformed values.
    confidence_level : float, default=0.95
        Confidence level used when the instance-level summary is requested.

    Returns
    -------
    int, float, or None
        Global measures return integers. Instance inspection units return the
        mean row-level count. ``None`` indicates that the fitted structure is
        not available in a form that can be counted reliably.
    """
    selected_mode = _normalize_mode(mode)
    tolerance = _validate_tolerance(coefficient_tolerance)
    confidence = _validate_confidence_level(confidence_level)

    if selected_mode == INSTANCE_INSPECTION_UNITS:
        if X is None:
            raise ValueError("X is required for instance inspection units")
        values = get_instance_inspection_units(
            model,
            X,
            coefficient_tolerance=tolerance,
        )
        if values is None:
            return None
        return float(_instance_summary(values, confidence)["mean"])

    report = get_complexity_report(
        model,
        coefficient_tolerance=tolerance,
        confidence_level=confidence,
    )
    if report is None:
        return None
    key = "model_units" if selected_mode == MODEL_UNITS else "model_inspection_units"
    value = report.get(key, {}).get("value")
    return None if value is None else int(value)


def get_instance_inspection_units(
    model: Any,
    X: Any,
    *,
    coefficient_tolerance: float = 1e-12,
) -> np.ndarray | None:
    """Return one instance-inspection count for every row in X.

    The result is a one-dimensional integer array. ``None`` is returned when
    the estimator does not expose enough fitted structure to identify the
    row-specific evidence reliably.
    """
    tolerance = _validate_tolerance(coefficient_tolerance)
    if model is None:
        return None

    candidate, transformed_X = _resolve_estimator_and_input(model, X)
    if _is_hugiml(candidate):
        values = _hugiml_instance_inspection_units(candidate, transformed_X, tolerance)
    else:
        reporters = (
            _ebm_instance_inspection_units,
            _xgboost_instance_inspection_units,
            _lightgbm_instance_inspection_units,
            _rulefit_instance_inspection_units,
            _forest_instance_inspection_units,
            _single_tree_instance_inspection_units,
            _linear_instance_inspection_units,
        )
        values = None
        for reporter in reporters:
            try:
                values = reporter(candidate, transformed_X, tolerance)
            except Exception:
                values = None
            if values is not None:
                break

    if values is None:
        return None
    result = np.asarray(values, dtype=np.int64).reshape(-1)
    if result.shape[0] != _n_rows(transformed_X):
        return None
    if np.any(result < 0):
        return None
    return result


def get_complexity_report(
    model: Any,
    *,
    X: Any | None = None,
    coefficient_tolerance: float = 1e-12,
    confidence_level: float = 0.95,
) -> dict[str, Any] | None:
    """Return model, model-inspection, and optional instance details.

    Passing X adds an ``instance_inspection_units`` section containing the
    row count, mean, sample standard deviation, standard error, and two-sided
    Student-t confidence interval.
    """
    tolerance = _validate_tolerance(coefficient_tolerance)
    confidence = _validate_confidence_level(confidence_level)
    if model is None:
        return None

    candidate = _unwrap_estimator(model)
    if _is_hugiml(candidate):
        report = _hugiml_complexity_report(candidate, tolerance)
    else:
        reporters = (
            _ebm_complexity_report,
            _xgboost_complexity_report,
            _lightgbm_complexity_report,
            _rulefit_complexity_report,
            _forest_complexity_report,
            _single_tree_complexity_report,
            _linear_complexity_report,
        )
        report = None
        for reporter in reporters:
            try:
                report = reporter(candidate, tolerance)
            except Exception:
                report = None
            if report is not None:
                break

    if report is None:
        return None

    if X is not None:
        values = get_instance_inspection_units(
            model,
            X,
            coefficient_tolerance=tolerance,
        )
        if values is None:
            report["instance_inspection_units"] = {
                "available": False,
                "unit": "expanded_row_specific_evidence",
                "n_samples": int(_n_rows(X)),
                "confidence_level": confidence,
            }
        else:
            report["instance_inspection_units"] = _instance_summary(values, confidence)
    return report


def _normalize_mode(mode: str | None) -> str:
    if mode is None:
        return MODEL_INSPECTION_UNITS
    normalized = str(mode).strip().lower()
    if normalized not in _VALID_MODES:
        raise ValueError(
            "mode must be 'model units', 'model inspection units', "
            "'instance inspection units', or None"
        )
    return normalized


def _validate_tolerance(value: float) -> float:
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("coefficient_tolerance must be a finite non-negative number")
    return tolerance


def _validate_confidence_level(value: float) -> float:
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    return confidence


def _instance_summary(values: np.ndarray, confidence_level: float) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("instance inspection units require at least one row")
    mean = float(np.mean(arr))
    if arr.size == 1:
        std = 0.0
        standard_error = 0.0
        lower = mean
        upper = mean
    else:
        std = float(np.std(arr, ddof=1))
        standard_error = std / math.sqrt(arr.size)
        critical = float(student_t.ppf((1.0 + confidence_level) / 2.0, arr.size - 1))
        margin = critical * standard_error
        lower = mean - margin
        upper = mean + margin
    return {
        "available": True,
        "value": mean,
        "mean": mean,
        "std": std,
        "standard_error": standard_error,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": float(confidence_level),
        "n_samples": int(arr.size),
        "sum": float(np.sum(arr)),
        "sum_squares": float(np.dot(arr, arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "unit": "expanded_row_specific_evidence",
    }


def _n_rows(X: Any) -> int:
    if X is None:
        return 0
    if hasattr(X, "shape") and len(X.shape):
        return int(X.shape[0])
    return int(len(X))


def _unwrap_estimator(model: Any) -> Any:
    current = model
    visited: set[int] = set()
    for _ in range(8):
        marker = id(current)
        if marker in visited:
            break
        visited.add(marker)
        if _is_hugiml(current):
            break
        if hasattr(current, "best_estimator_"):
            current = current.best_estimator_
            continue
        steps = getattr(current, "steps", None)
        if steps:
            current = steps[-1][1]
            continue
        named_steps = getattr(current, "named_steps", None)
        if isinstance(named_steps, Mapping) and named_steps:
            current = list(named_steps.values())[-1]
            continue
        break
    return current


def _resolve_estimator_and_input(model: Any, X: Any) -> tuple[Any, Any]:
    current = model
    data = X
    visited: set[int] = set()
    for _ in range(8):
        marker = id(current)
        if marker in visited:
            break
        visited.add(marker)
        if _is_hugiml(current):
            break
        if hasattr(current, "best_estimator_"):
            current = current.best_estimator_
            continue
        steps = getattr(current, "steps", None)
        if steps:
            for _, transformer in steps[:-1]:
                if transformer is None or transformer == "passthrough":
                    continue
                data = transformer.transform(data)
            current = steps[-1][1]
            continue
        break
    return current, data


def _model_identity(model: Any) -> tuple[str, str]:
    cls = type(model)
    return cls.__module__.lower(), cls.__name__.lower()


def _is_hugiml(model: Any) -> bool:
    module_name, class_name = _model_identity(model)
    return bool(
        module_name.startswith("hugiml")
        and "hugimlclassifier" in class_name
        and hasattr(model, "rpte_rule_table")
        and hasattr(model, "_get_downstream_feature_names")
    )


def _is_active_number(value: Any, tolerance: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(number) and abs(number) > tolerance)


def _base_report(
    *,
    model_type: str,
    model_value: int,
    model_unit: str,
    model_inspection_value: int,
    model_inspection_unit: str,
    model_details: Mapping[str, Any] | None = None,
    model_inspection_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_units = {"value": int(model_value), "unit": model_unit}
    model_inspection_units = {
        "value": int(model_inspection_value),
        "unit": model_inspection_unit,
    }
    if model_details:
        model_units.update(dict(model_details))
    if model_inspection_details:
        model_inspection_units.update(dict(model_inspection_details))
    return {
        "model_type": model_type,
        "model_units": model_units,
        "model_inspection_units": model_inspection_units,
    }


def _coefficient_active_mask(
    estimator: Any,
    width: int,
    tolerance: float,
) -> tuple[np.ndarray, bool]:
    blocks: list[np.ndarray] = []
    if hasattr(estimator, "coef_"):
        blocks.append(np.asarray(estimator.coef_, dtype=float))
    elif hasattr(estimator, "estimators_"):
        for sub_estimator in _flatten_estimators(getattr(estimator, "estimators_", None)):
            if hasattr(sub_estimator, "coef_"):
                blocks.append(np.asarray(sub_estimator.coef_, dtype=float))
    if not blocks:
        return np.ones(width, dtype=bool), False

    active = np.zeros(width, dtype=bool)
    for block in blocks:
        matrix = np.atleast_2d(block)
        if matrix.shape[-1] != width:
            return np.ones(width, dtype=bool), False
        active |= np.any(np.isfinite(matrix) & (np.abs(matrix) > tolerance), axis=0)
    return active, True


# HUGIML --------------------------------------------------------------------


def _hugiml_pattern_inspection_sizes(model: Any) -> dict[str, int]:
    sizes: dict[str, int] = {}
    labels = list(model.get_hug_features())
    patterns = list(getattr(model, "patterns_", []) or [])
    for label, pattern in zip(labels, patterns):
        size = max(1, len(getattr(pattern, "items", []) or []))
        sizes[str(label)] = int(size)
        sizes[f"pattern:{label}"] = int(size)
    return sizes


def _hugiml_augmented_pair_inspection_sizes(model: Any) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for item in getattr(model, "augmented_pair_transforms_", []) or []:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        inputs = [str(value) for value in (item.get("inputs") or []) if str(value)]
        size = max(1, len(dict.fromkeys(inputs)))
        sizes[name] = int(size)
        sizes[f"augmented_pair:{name}"] = int(size)
    return sizes


def _hugiml_feature_inspection_size(
    model: Any,
    name: str,
    *,
    pattern_sizes: Mapping[str, int],
    augmented_pair_sizes: Mapping[str, int],
    condition: Mapping[str, Any] | None = None,
) -> int:
    feature_name = str(name or "")
    if feature_name.startswith("pattern:") or feature_name in pattern_sizes:
        display_name = model._downstream_feature_display_name(feature_name)
        return int(pattern_sizes.get(feature_name, pattern_sizes.get(display_name, 1)))
    if feature_name.startswith("augmented_pair:") or feature_name in augmented_pair_sizes:
        if feature_name in augmented_pair_sizes:
            return int(augmented_pair_sizes[feature_name])
        display_name = model._downstream_feature_display_name(feature_name)
        if display_name in augmented_pair_sizes:
            return int(augmented_pair_sizes[display_name])
    if condition:
        raw_sources = [
            str(value) for value in (condition.get("raw_sources") or []) if str(value)
        ]
        if raw_sources:
            return max(1, len(dict.fromkeys(raw_sources)))
    return 1


def _hugiml_active_linear_features(
    model: Any,
    tolerance: float,
) -> tuple[list[str], np.ndarray, bool]:
    names = list(model._get_downstream_feature_names())
    estimator = model.model_.named_steps.get("clf", model.model_)
    active, applied = _coefficient_active_mask(estimator, len(names), tolerance)
    return [name for name, keep in zip(names, active) if keep], active, applied


def _hugiml_complexity_report(model: Any, tolerance: float) -> dict[str, Any]:
    check_is_fitted(model, "model_")
    pattern_sizes = _hugiml_pattern_inspection_sizes(model)
    augmented_pair_sizes = _hugiml_augmented_pair_inspection_sizes(model)
    selected_pattern_count = int(len(getattr(model, "patterns_", []) or []))
    rows = list(model.rpte_rule_table())
    tree_backends = {"bounded_lookahead", "sequential_default"}

    candidate_tree_rows = [
        row
        for row in rows
        if row.get("tree_index") is not None and str(row.get("backend")) in tree_backends
    ]
    candidate_direct_rows = [
        row
        for row in rows
        if row.get("tree_index") is None
        and str(row.get("backend")) in {"direct_hugiml_feature", "raw_hugiml_features"}
        and row.get("source_selection_status") in {None, "not_selected_in_tree_split"}
    ]
    tree_rows = [
        row
        for row in candidate_tree_rows
        if _is_active_number(row.get("final_logistic_coefficient"), tolerance)
    ]
    direct_rows = [
        row
        for row in candidate_direct_rows
        if _is_active_number(row.get("final_logistic_coefficient"), tolerance)
    ]

    if candidate_tree_rows or candidate_direct_rows:
        direct_source_units = 0
        for row in direct_rows:
            conditions = list(row.get("conditions") or [])
            direct_source_units += _hugiml_feature_inspection_size(
                model,
                str(row.get("downstream_feature") or ""),
                pattern_sizes=pattern_sizes,
                augmented_pair_sizes=augmented_pair_sizes,
                condition=conditions[0] if conditions else None,
            )

        path_condition_count = 0
        expanded_path_units = 0
        path_lengths: list[int] = []
        for row in tree_rows:
            conditions = list(row.get("conditions") or [])
            path_lengths.append(len(conditions))
            path_condition_count += len(conditions)
            for condition in conditions:
                feature_name = str(
                    condition.get("downstream_feature") or condition.get("feature") or ""
                )
                expanded_path_units += _hugiml_feature_inspection_size(
                    model,
                    feature_name,
                    pattern_sizes=pattern_sizes,
                    augmented_pair_sizes=augmented_pair_sizes,
                    condition=condition,
                )

        leaf_count = int(len(tree_rows))
        direct_term_count = int(len(direct_rows))
        report = _base_report(
            model_type="hugiml_rpte",
            model_value=leaf_count + direct_term_count,
            model_unit="active_terminal_leaves_plus_direct_terms",
            model_inspection_value=expanded_path_units + direct_source_units,
            model_inspection_unit="expanded_active_leaf_paths_plus_direct_sources",
            model_details={
                "leaf_count": leaf_count,
                "direct_term_count": direct_term_count,
                "coefficient_tolerance": tolerance,
            },
            model_inspection_details={
                "path_condition_count": int(path_condition_count),
                "expanded_path_units": int(expanded_path_units),
                "direct_term_count": direct_term_count,
                "direct_source_units": int(direct_source_units),
                "mean_path_length": float(np.mean(path_lengths)) if path_lengths else 0.0,
                "max_path_length": int(max(path_lengths)) if path_lengths else 0,
            },
        )
        report["branch"] = "rpte" if candidate_tree_rows else "rpte_direct_only"
        return report

    active_names, _, coefficient_filter_applied = _hugiml_active_linear_features(
        model,
        tolerance,
    )
    family_term_counts = {"original": 0, "pattern": 0, "augmented_pair": 0}
    family_inspection_units = {"original": 0, "pattern": 0, "augmented_pair": 0}
    for name in active_names:
        family = model._downstream_feature_type(str(name))
        if family not in family_term_counts:
            family = "pattern"
        size = _hugiml_feature_inspection_size(
            model,
            str(name),
            pattern_sizes=pattern_sizes,
            augmented_pair_sizes=augmented_pair_sizes,
        )
        family_term_counts[family] += 1
        family_inspection_units[family] += int(size)

    model_value = int(len(active_names))
    model_inspection_value = int(sum(family_inspection_units.values()))
    report = _base_report(
        model_type="hugiml_linear",
        model_value=model_value,
        model_unit="active_downstream_terms",
        model_inspection_value=model_inspection_value,
        model_inspection_unit="expanded_all_active_sources",
        model_details={
            "active_term_count": model_value,
            "coefficient_filter_applied": bool(coefficient_filter_applied),
            "term_counts": family_term_counts,
            "selected_pattern_count": selected_pattern_count,
            "coefficient_tolerance": tolerance,
        },
        model_inspection_details={
            "active_term_count": model_value,
            "coefficient_filter_applied": bool(coefficient_filter_applied),
            "term_counts": family_term_counts,
            "units_by_family": family_inspection_units,
            "selected_pattern_model_inspection_units": int(
                sum(
                    pattern_sizes.get(f"pattern:{label}", 1)
                    for label in model.get_hug_features()
                )
            ),
        },
    )
    report["branch"] = "linear_or_empty_rpte"
    return report


def _hugiml_downstream_matrix(model: Any, X: Any) -> Any:
    if getattr(model, "_is_constant_prior_fallback_active", lambda: False)():
        return sparse.csr_matrix((_n_rows(X), 0), dtype=np.float32)
    raw_X = X
    pattern_X = X
    if getattr(model, "adaptive_binning", False) and getattr(model, "_bin_edges_", None):
        pattern_X = model._prebin_for_predict(pattern_X)
    Z = model._build_test_hup(pattern_X)
    downstream = model._make_downstream_features(raw_X, Z, fit=False)
    return model._apply_strict_topk_budget_transform(downstream)


def _rpte_leaf_column_keys(estimator: Any) -> list[tuple[int, int]]:
    fe = getattr(estimator, "fe_", None)
    if fe is None:
        return []
    default_fe = getattr(fe, "_default_fe", None)
    if default_fe is not None:
        return [
            (tree_index, int(leaf_id))
            for tree_index, vocab in enumerate(getattr(default_fe, "leaf_vocab_", []) or [])
            for leaf_id in vocab
        ]
    if getattr(fe, "is_degenerate_", False):
        return []
    return [
        (tree_index, int(leaf_id))
        for tree_index, vocab in enumerate(getattr(fe, "tree_leaf_ids_", []) or [])
        for leaf_id in vocab
    ]


def _hugiml_rpte_instance_inspection_units(
    model: Any,
    downstream_X: Any,
    tolerance: float,
    pattern_sizes: Mapping[str, int],
    augmented_pair_sizes: Mapping[str, int],
) -> np.ndarray | None:
    estimator = model.model_.named_steps.get("clf", model.model_)
    if hasattr(estimator, "estimators_") and hasattr(estimator, "classes_"):
        sub_estimators = list(getattr(estimator, "estimators_", []) or [])
    elif hasattr(estimator, "unified_rule_table"):
        sub_estimators = [estimator]
    else:
        return None
    if not sub_estimators or not all(hasattr(sub, "unified_rule_table") for sub in sub_estimators):
        return None

    names = list(model._get_downstream_feature_names())
    n = _n_rows(downstream_X)
    result = np.zeros(n, dtype=np.int64)
    tree_backends = {"bounded_lookahead", "sequential_default"}

    for sub in sub_estimators:
        rows = list(sub.unified_rule_table(feature_names=names))
        active_tree_rows = {
            (int(row["tree_index"]), int(row["leaf_index"])): row
            for row in rows
            if row.get("tree_index") is not None
            and row.get("leaf_index") is not None
            and str(row.get("backend")) in tree_backends
            and _is_active_number(row.get("final_logistic_coefficient"), tolerance)
        }
        active_direct_rows = [
            row
            for row in rows
            if row.get("tree_index") is None
            and str(row.get("backend")) in {"direct_hugiml_feature", "raw_hugiml_features"}
            and row.get("source_selection_status") in {None, "not_selected_in_tree_split"}
            and _is_active_number(row.get("final_logistic_coefficient"), tolerance)
        ]

        direct_units = 0
        for row in active_direct_rows:
            conditions = list(row.get("conditions") or [])
            direct_units += _hugiml_feature_inspection_size(
                model,
                str(row.get("downstream_feature") or ""),
                pattern_sizes=pattern_sizes,
                augmented_pair_sizes=augmented_pair_sizes,
                condition=conditions[0] if conditions else None,
            )
        result += int(direct_units)

        keys = _rpte_leaf_column_keys(sub)
        if not keys or not active_tree_rows:
            continue
        leaves = sub.fe_.transform_leaves(downstream_X)
        if leaves.shape[1] != len(keys):
            return None
        weights = np.zeros(len(keys), dtype=np.int64)
        for column_index, key in enumerate(keys):
            row = active_tree_rows.get(key)
            if row is None:
                continue
            units = 0
            for condition in list(row.get("conditions") or []):
                feature_name = str(
                    condition.get("downstream_feature") or condition.get("feature") or ""
                )
                units += _hugiml_feature_inspection_size(
                    model,
                    feature_name,
                    pattern_sizes=pattern_sizes,
                    augmented_pair_sizes=augmented_pair_sizes,
                    condition=condition,
                )
            weights[column_index] = int(units)
        result += np.asarray(leaves @ weights).reshape(-1).astype(np.int64)
    return result


def _hugiml_instance_inspection_units(
    model: Any,
    X: Any,
    tolerance: float,
) -> np.ndarray | None:
    check_is_fitted(model, "model_")
    n = _n_rows(X)
    if getattr(model, "_is_constant_prior_fallback_active", lambda: False)():
        return np.zeros(n, dtype=np.int64)

    downstream_X = _hugiml_downstream_matrix(model, X)
    pattern_sizes = _hugiml_pattern_inspection_sizes(model)
    augmented_pair_sizes = _hugiml_augmented_pair_inspection_sizes(model)

    rpte_values = _hugiml_rpte_instance_inspection_units(
        model,
        downstream_X,
        tolerance,
        pattern_sizes,
        augmented_pair_sizes,
    )
    if rpte_values is not None:
        return rpte_values

    names = list(model._get_downstream_feature_names())
    if downstream_X.shape[1] != len(names):
        return None
    _, active, _ = _hugiml_active_linear_features(model, tolerance)
    result = np.zeros(n, dtype=np.int64)

    local_indices: list[int] = []
    local_weights: list[int] = []
    for column_index, (name, keep) in enumerate(zip(names, active)):
        if not keep:
            continue
        family = model._downstream_feature_type(str(name))
        units = _hugiml_feature_inspection_size(
            model,
            str(name),
            pattern_sizes=pattern_sizes,
            augmented_pair_sizes=augmented_pair_sizes,
        )
        if family == "original":
            result += int(units)
        else:
            local_indices.append(column_index)
            local_weights.append(int(units))

    if local_indices:
        selected = downstream_X[:, local_indices]
        weights = np.asarray(local_weights, dtype=np.int64)
        if sparse.issparse(selected):
            present = selected.copy().tocsr()
            present.data = (np.abs(present.data) > tolerance).astype(np.int8)
            present.eliminate_zeros()
            result += np.asarray(present @ weights).reshape(-1).astype(np.int64)
        else:
            present = np.abs(np.asarray(selected, dtype=float)) > tolerance
            result += np.asarray(present @ weights).reshape(-1).astype(np.int64)
    return result


# Tree estimators -----------------------------------------------------------


def _json_tree_units(
    node: Mapping[str, Any],
    *,
    tolerance: float,
    depth: int,
) -> tuple[int, int]:
    if "leaf" in node:
        active = _is_active_number(node.get("leaf"), tolerance)
        return (1, depth) if active else (0, 0)
    leaves = 0
    paths = 0
    for child in node.get("children", []) or []:
        child_leaves, child_paths = _json_tree_units(
            child,
            tolerance=tolerance,
            depth=depth + 1,
        )
        leaves += child_leaves
        paths += child_paths
    return leaves, paths


def _json_tree_leaf_depths(
    node: Mapping[str, Any],
    *,
    tolerance: float,
    depth: int = 0,
) -> dict[int, int]:
    if "leaf" in node:
        node_id = int(node.get("nodeid", 0))
        return {node_id: depth if _is_active_number(node.get("leaf"), tolerance) else 0}
    depths: dict[int, int] = {}
    for child in node.get("children", []) or []:
        depths.update(
            _json_tree_leaf_depths(
                child,
                tolerance=tolerance,
                depth=depth + 1,
            )
        )
    return depths


def _xgboost_dump(model: Any) -> list[Any] | None:
    module_name, class_name = _model_identity(model)
    get_dump = getattr(model, "get_dump", None)
    get_booster = getattr(model, "get_booster", None)
    if not (
        module_name.startswith("xgboost")
        or "xgb" in class_name
        or callable(get_dump)
        or callable(get_booster)
    ):
        return None
    booster = model if callable(get_dump) else get_booster()
    dump_method = getattr(booster, "get_dump", None)
    if not callable(dump_method):
        return None
    return list(dump_method(dump_format="json"))


def _xgboost_complexity_report(model: Any, tolerance: float) -> dict[str, Any] | None:
    dumps = _xgboost_dump(model)
    if dumps is None:
        return None
    leaf_count = 0
    path_units = 0
    for tree_text in dumps:
        tree = json.loads(tree_text) if isinstance(tree_text, str) else tree_text
        leaves, paths = _json_tree_units(tree, tolerance=tolerance, depth=0)
        leaf_count += leaves
        path_units += paths
    return _base_report(
        model_type="xgboost",
        model_value=leaf_count,
        model_unit="active_terminal_leaves",
        model_inspection_value=path_units,
        model_inspection_unit="all_active_root_to_leaf_conditions",
        model_details={"tree_count": int(len(dumps)), "coefficient_tolerance": tolerance},
        model_inspection_details={"tree_count": int(len(dumps))},
    )


def _xgboost_instance_inspection_units(
    model: Any,
    X: Any,
    tolerance: float,
) -> np.ndarray | None:
    dumps = _xgboost_dump(model)
    apply = getattr(model, "apply", None)
    if dumps is None or not callable(apply):
        return None
    leaf_ids = np.asarray(apply(X))
    if leaf_ids.ndim == 1:
        leaf_ids = leaf_ids.reshape(-1, 1)
    else:
        leaf_ids = leaf_ids.reshape(leaf_ids.shape[0], -1)
    if leaf_ids.shape[1] != len(dumps):
        return None
    result = np.zeros(leaf_ids.shape[0], dtype=np.int64)
    for tree_index, tree_text in enumerate(dumps):
        tree = json.loads(tree_text) if isinstance(tree_text, str) else tree_text
        depths = _json_tree_leaf_depths(tree, tolerance=tolerance)
        result += np.asarray(
            [depths.get(int(leaf_id), 0) for leaf_id in leaf_ids[:, tree_index]],
            dtype=np.int64,
        )
    return result


def _lightgbm_tree_units(
    node: Mapping[str, Any],
    *,
    tolerance: float,
    depth: int,
) -> tuple[int, int]:
    if "leaf_index" in node:
        active = _is_active_number(node.get("leaf_value"), tolerance)
        return (1, depth) if active else (0, 0)
    leaves = 0
    paths = 0
    for child_name in ("left_child", "right_child"):
        child = node.get(child_name)
        if isinstance(child, Mapping):
            child_leaves, child_paths = _lightgbm_tree_units(
                child,
                tolerance=tolerance,
                depth=depth + 1,
            )
            leaves += child_leaves
            paths += child_paths
    return leaves, paths


def _lightgbm_leaf_depths(
    node: Mapping[str, Any],
    *,
    tolerance: float,
    depth: int = 0,
) -> dict[int, int]:
    if "leaf_index" in node:
        leaf_id = int(node["leaf_index"])
        return {
            leaf_id: depth
            if _is_active_number(node.get("leaf_value"), tolerance)
            else 0
        }
    depths: dict[int, int] = {}
    for child_name in ("left_child", "right_child"):
        child = node.get(child_name)
        if isinstance(child, Mapping):
            depths.update(
                _lightgbm_leaf_depths(
                    child,
                    tolerance=tolerance,
                    depth=depth + 1,
                )
            )
    return depths


def _lightgbm_dump(model: Any) -> tuple[Any, list[Mapping[str, Any]]] | None:
    module_name, class_name = _model_identity(model)
    direct_dump = getattr(model, "dump_model", None)
    fitted_booster = getattr(model, "booster_", None)
    if not (
        module_name.startswith("lightgbm")
        or "lgbm" in class_name
        or callable(direct_dump)
        or callable(getattr(fitted_booster, "dump_model", None))
    ):
        return None
    booster = model if callable(direct_dump) else fitted_booster
    dump_method = getattr(booster, "dump_model", None)
    if not callable(dump_method):
        return None
    dump = dump_method()
    return booster, list(dump.get("tree_info", []) or [])


def _lightgbm_complexity_report(model: Any, tolerance: float) -> dict[str, Any] | None:
    dumped = _lightgbm_dump(model)
    if dumped is None:
        return None
    _, tree_info = dumped
    leaf_count = 0
    path_units = 0
    for tree in tree_info:
        leaves, paths = _lightgbm_tree_units(
            tree.get("tree_structure", {}),
            tolerance=tolerance,
            depth=0,
        )
        leaf_count += leaves
        path_units += paths
    return _base_report(
        model_type="lightgbm",
        model_value=leaf_count,
        model_unit="active_terminal_leaves",
        model_inspection_value=path_units,
        model_inspection_unit="all_active_root_to_leaf_conditions",
        model_details={"tree_count": int(len(tree_info)), "coefficient_tolerance": tolerance},
        model_inspection_details={"tree_count": int(len(tree_info))},
    )


def _lightgbm_instance_inspection_units(
    model: Any,
    X: Any,
    tolerance: float,
) -> np.ndarray | None:
    dumped = _lightgbm_dump(model)
    if dumped is None:
        return None
    booster, tree_info = dumped
    predict = getattr(model, "predict", None)
    if not callable(predict):
        predict = getattr(booster, "predict", None)
    if not callable(predict):
        return None
    leaf_ids = np.asarray(predict(X, pred_leaf=True))
    if leaf_ids.ndim == 1:
        leaf_ids = leaf_ids.reshape(-1, 1)
    else:
        leaf_ids = leaf_ids.reshape(leaf_ids.shape[0], -1)
    if leaf_ids.shape[1] != len(tree_info):
        return None
    result = np.zeros(leaf_ids.shape[0], dtype=np.int64)
    for tree_index, tree in enumerate(tree_info):
        depths = _lightgbm_leaf_depths(
            tree.get("tree_structure", {}),
            tolerance=tolerance,
        )
        result += np.asarray(
            [depths.get(int(leaf_id), 0) for leaf_id in leaf_ids[:, tree_index]],
            dtype=np.int64,
        )
    return result


def _flatten_estimators(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [item for item in value.ravel().tolist() if item is not None]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        return [item for item in value if item is not None]
    return []


def _sklearn_tree_units(tree: Any) -> tuple[int, int]:
    children_left = np.asarray(tree.children_left, dtype=int)
    children_right = np.asarray(tree.children_right, dtype=int)
    leaf_count = 0
    path_units = 0
    stack: list[tuple[int, int]] = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        left = int(children_left[node_id])
        right = int(children_right[node_id])
        if left < 0 and right < 0:
            leaf_count += 1
            path_units += depth
            continue
        if right >= 0:
            stack.append((right, depth + 1))
        if left >= 0:
            stack.append((left, depth + 1))
    return int(leaf_count), int(path_units)


def _sklearn_tree_depths(tree: Any) -> np.ndarray:
    children_left = np.asarray(tree.children_left, dtype=int)
    children_right = np.asarray(tree.children_right, dtype=int)
    depths = np.zeros(int(tree.node_count), dtype=np.int64)
    stack: list[tuple[int, int]] = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        left = int(children_left[node_id])
        right = int(children_right[node_id])
        if right >= 0:
            stack.append((right, depth + 1))
        if left >= 0:
            stack.append((left, depth + 1))
    return depths


def _forest_complexity_report(model: Any, tolerance: float) -> dict[str, Any] | None:
    del tolerance
    estimators = _flatten_estimators(getattr(model, "estimators_", None))
    tree_estimators = [estimator for estimator in estimators if hasattr(estimator, "tree_")]
    if not tree_estimators or len(tree_estimators) != len(estimators):
        return None
    leaf_count = 0
    path_units = 0
    for estimator in tree_estimators:
        leaves, paths = _sklearn_tree_units(estimator.tree_)
        leaf_count += leaves
        path_units += paths
    return _base_report(
        model_type="tree_ensemble",
        model_value=leaf_count,
        model_unit="terminal_leaves",
        model_inspection_value=path_units,
        model_inspection_unit="all_root_to_leaf_conditions",
        model_details={"tree_count": int(len(tree_estimators))},
        model_inspection_details={"tree_count": int(len(tree_estimators))},
    )


def _forest_instance_inspection_units(
    model: Any,
    X: Any,
    tolerance: float,
) -> np.ndarray | None:
    del tolerance
    estimators = _flatten_estimators(getattr(model, "estimators_", None))
    tree_estimators = [estimator for estimator in estimators if hasattr(estimator, "tree_")]
    if not tree_estimators or len(tree_estimators) != len(estimators):
        return None
    result = np.zeros(_n_rows(X), dtype=np.int64)
    for estimator in tree_estimators:
        leaf_ids = np.asarray(estimator.apply(X), dtype=np.int64).reshape(-1)
        result += _sklearn_tree_depths(estimator.tree_)[leaf_ids]
    return result


def _single_tree_complexity_report(model: Any, tolerance: float) -> dict[str, Any] | None:
    del tolerance
    tree = getattr(model, "tree_", None)
    if tree is None:
        return None
    leaves, paths = _sklearn_tree_units(tree)
    return _base_report(
        model_type="decision_tree",
        model_value=leaves,
        model_unit="terminal_leaves",
        model_inspection_value=paths,
        model_inspection_unit="all_root_to_leaf_conditions",
    )


def _single_tree_instance_inspection_units(
    model: Any,
    X: Any,
    tolerance: float,
) -> np.ndarray | None:
    del tolerance
    tree = getattr(model, "tree_", None)
    apply = getattr(model, "apply", None)
    if tree is None or not callable(apply):
        return None
    leaf_ids = np.asarray(apply(X), dtype=np.int64).reshape(-1)
    return _sklearn_tree_depths(tree)[leaf_ids]


# Additive and rule models --------------------------------------------------


def _ebm_complexity_report(model: Any, tolerance: float) -> dict[str, Any] | None:
    module_name, class_name = _model_identity(model)
    scores = getattr(model, "term_scores_", None)
    if not (
        module_name.startswith("interpret")
        or "explainableboosting" in class_name
        or scores is not None
    ):
        return None
    if scores is None:
        return None

    active_terms = 0
    active_cells = 0
    cells_by_term: list[int] = []
    for term_scores in list(scores):
        values = np.asarray(term_scores, dtype=float)
        active = np.isfinite(values) & (np.abs(values) > tolerance)
        cell_count = int(np.sum(active))
        cells_by_term.append(cell_count)
        if cell_count:
            active_terms += 1
            active_cells += cell_count
    return _base_report(
        model_type="explainable_boosting_machine",
        model_value=active_terms,
        model_unit="active_additive_terms",
        model_inspection_value=active_cells,
        model_inspection_unit="all_active_term_score_cells",
        model_details={
            "available_term_count": int(len(cells_by_term)),
            "coefficient_tolerance": tolerance,
        },
        model_inspection_details={"active_cells_by_term": cells_by_term},
    )


def _ebm_instance_inspection_units(
    model: Any,
    X: Any,
    tolerance: float,
) -> np.ndarray | None:
    module_name, class_name = _model_identity(model)
    if not (
        module_name.startswith("interpret")
        or "explainableboosting" in class_name
        or hasattr(model, "term_scores_")
    ):
        return None
    evaluate = getattr(model, "eval_terms", None)
    if not callable(evaluate):
        return None
    values = np.asarray(evaluate(X), dtype=float)
    if values.ndim == 2:
        active = np.isfinite(values) & (np.abs(values) > tolerance)
    elif values.ndim >= 3:
        active = np.any(
            np.isfinite(values) & (np.abs(values) > tolerance),
            axis=tuple(range(2, values.ndim)),
        )
    else:
        return None
    return np.sum(active, axis=1, dtype=np.int64)


def _rulefit_rows(model: Any) -> list[dict[str, Any]] | None:
    get_rules = getattr(model, "get_rules", None)
    if callable(get_rules):
        table = get_rules()
        if hasattr(table, "to_dict"):
            return list(table.to_dict(orient="records"))
        if isinstance(table, Iterable) and not isinstance(table, (str, bytes, Mapping)):
            return [dict(row) for row in table]

    rules = list(getattr(model, "rules_", []) or [])
    if not rules:
        return None
    rows: list[dict[str, Any]] = []
    for rule in rules:
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
    return rows


def _count_rule_literals(rule_text: str) -> int:
    text = str(rule_text or "").strip()
    if not text:
        return 1
    parts = [
        part.strip()
        for part in re.split(r"\s+(?:and|AND)\s+|\s*&\s*", text)
        if part.strip()
    ]
    return max(1, len(parts))


def _rulefit_complexity_report(model: Any, tolerance: float) -> dict[str, Any] | None:
    module_name, class_name = _model_identity(model)
    if not (
        module_name.startswith("imodels")
        or "rulefit" in class_name
        or callable(getattr(model, "get_rules", None))
    ):
        return None
    rows = _rulefit_rows(model)
    if rows is None:
        return None

    active_linear_terms = 0
    active_rules = 0
    rule_literals = 0
    for row in rows:
        coefficient = row.get("coef", row.get("coefficient"))
        if not _is_active_number(coefficient, tolerance):
            continue
        term_type = str(row.get("type", row.get("term_type", "rule"))).lower()
        rule_text = str(row.get("rule", row.get("term", "")))
        if term_type == "linear":
            active_linear_terms += 1
        else:
            active_rules += 1
            rule_literals += _count_rule_literals(rule_text)

    model_value = active_linear_terms + active_rules
    model_inspection_value = active_linear_terms + rule_literals
    return _base_report(
        model_type="rulefit",
        model_value=model_value,
        model_unit="active_linear_terms_plus_rules",
        model_inspection_value=model_inspection_value,
        model_inspection_unit="all_linear_terms_plus_rule_conditions",
        model_details={
            "active_linear_term_count": int(active_linear_terms),
            "active_rule_count": int(active_rules),
            "coefficient_tolerance": tolerance,
        },
        model_inspection_details={
            "active_linear_term_count": int(active_linear_terms),
            "rule_condition_count": int(rule_literals),
        },
    )


def _rulefit_transform_matrix(model: Any, X: Any, n_rules: int) -> np.ndarray | None:
    transform = getattr(model, "transform", None)
    if not callable(transform):
        return None
    try:
        values = transform(X)
    except Exception:
        return None
    if sparse.issparse(values):
        values = values.toarray()
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[1] != n_rules:
        return None
    return matrix


def _rule_feature_values(model: Any, X: Any, name: str) -> np.ndarray | None:
    clean = str(name).strip().strip("`")
    if hasattr(X, "columns"):
        columns = [str(column) for column in X.columns]
        if clean in columns:
            return np.asarray(X.iloc[:, columns.index(clean)], dtype=float)

    feature_names = None
    for attr in ("feature_names_", "feature_names", "feature_names_in_"):
        candidate = getattr(model, attr, None)
        if candidate is not None and len(candidate):
            feature_names = [str(value) for value in candidate]
            break
    array = X.toarray() if sparse.issparse(X) else np.asarray(X)
    if array.ndim != 2:
        return None
    if feature_names is not None and clean in feature_names:
        return np.asarray(array[:, feature_names.index(clean)], dtype=float)
    match = re.fullmatch(r"(?:feature_|x_|X_|x|X)?(\d+)", clean)
    if match:
        index = int(match.group(1))
        if 0 <= index < array.shape[1]:
            return np.asarray(array[:, index], dtype=float)
    return None


def _evaluate_rulefit_rule(model: Any, X: Any, rule_text: str) -> np.ndarray | None:
    parts = [
        part.strip().strip("()")
        for part in re.split(r"\s+(?:and|AND)\s+|\s*&\s*", str(rule_text))
        if part.strip()
    ]
    result = np.ones(_n_rows(X), dtype=bool)
    pattern = re.compile(
        r"^\s*(.+?)\s*(<=|>=|==|!=|<|>)\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
    )
    for part in parts:
        match = pattern.match(part)
        if not match:
            return None
        feature_name, operator, threshold_text = match.groups()
        values = _rule_feature_values(model, X, feature_name)
        if values is None:
            return None
        threshold = float(threshold_text)
        if operator == "<=":
            current = values <= threshold
        elif operator == ">=":
            current = values >= threshold
        elif operator == "<":
            current = values < threshold
        elif operator == ">":
            current = values > threshold
        elif operator == "==":
            current = values == threshold
        else:
            current = values != threshold
        result &= np.asarray(current, dtype=bool)
    return result


def _rulefit_instance_inspection_units(
    model: Any,
    X: Any,
    tolerance: float,
) -> np.ndarray | None:
    module_name, class_name = _model_identity(model)
    if not (
        module_name.startswith("imodels")
        or "rulefit" in class_name
        or callable(getattr(model, "get_rules", None))
    ):
        return None
    rows = _rulefit_rows(model)
    if rows is None:
        return None

    active_linear_terms = 0
    all_rule_rows: list[dict[str, Any]] = []
    active_rule_positions: list[int] = []
    active_rule_weights: list[int] = []
    for row in rows:
        term_type = str(row.get("type", row.get("term_type", "rule"))).lower()
        if term_type == "linear":
            if _is_active_number(row.get("coef", row.get("coefficient")), tolerance):
                active_linear_terms += 1
            continue
        rule_position = len(all_rule_rows)
        all_rule_rows.append(row)
        if _is_active_number(row.get("coef", row.get("coefficient")), tolerance):
            active_rule_positions.append(rule_position)
            active_rule_weights.append(
                _count_rule_literals(str(row.get("rule", row.get("term", ""))))
            )

    result = np.full(_n_rows(X), active_linear_terms, dtype=np.int64)
    if not active_rule_positions:
        return result

    transformed = _rulefit_transform_matrix(model, X, len(all_rule_rows))
    if transformed is not None:
        present = np.abs(transformed[:, active_rule_positions]) > tolerance
        result += np.asarray(
            present @ np.asarray(active_rule_weights, dtype=np.int64)
        ).reshape(-1).astype(np.int64)
        return result

    for position, weight in zip(active_rule_positions, active_rule_weights):
        row = all_rule_rows[position]
        present = _evaluate_rulefit_rule(
            model,
            X,
            str(row.get("rule", row.get("term", ""))),
        )
        if present is None:
            return None
        result += present.astype(np.int64) * int(weight)
    return result


# Generic linear estimators ------------------------------------------------


def _linear_active_term_count(model: Any, tolerance: float) -> int | None:
    blocks: list[np.ndarray] = []
    if hasattr(model, "coef_"):
        blocks.append(np.asarray(model.coef_, dtype=float))
    elif hasattr(model, "estimators_"):
        for estimator in _flatten_estimators(model.estimators_):
            if hasattr(estimator, "coef_"):
                blocks.append(np.asarray(estimator.coef_, dtype=float))
    if not blocks:
        return None
    width = int(np.atleast_2d(blocks[0]).shape[-1])
    active = np.zeros(width, dtype=bool)
    for block in blocks:
        matrix = np.atleast_2d(block)
        if matrix.shape[-1] != width:
            return None
        active |= np.any(np.isfinite(matrix) & (np.abs(matrix) > tolerance), axis=0)
    return int(np.sum(active))


def _linear_complexity_report(model: Any, tolerance: float) -> dict[str, Any] | None:
    active_terms = _linear_active_term_count(model, tolerance)
    if active_terms is None:
        return None
    return _base_report(
        model_type="linear_model",
        model_value=active_terms,
        model_unit="active_terms",
        model_inspection_value=active_terms,
        model_inspection_unit="all_active_source_terms",
        model_details={"coefficient_tolerance": tolerance},
    )


def _linear_instance_inspection_units(
    model: Any,
    X: Any,
    tolerance: float,
) -> np.ndarray | None:
    active_terms = _linear_active_term_count(model, tolerance)
    if active_terms is None:
        return None
    return np.full(_n_rows(X), active_terms, dtype=np.int64)


__all__ = [
    "INSTANCE_INSPECTION_UNITS",
    "MODEL_INSPECTION_UNITS",
    "MODEL_UNITS",
    "get_complexity",
    "get_complexity_report",
    "get_instance_inspection_units",
]

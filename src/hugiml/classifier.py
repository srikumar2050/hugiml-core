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

"""HUGIMLClassifier — C++ accelerated, scikit-learn compatible classifier.

``HUGIMLClassifier`` is the primary public class name.
``HUGIMLClassifierNative`` remains as a backward-compatible alias.

Implements the High Utility Gain Interpretable Machine Learning (HUG-IML)
algorithm from:

    Krishnamoorthy, S. (2024). Interpretable Classifier Models for Decision
    Support Using High Utility Gain Patterns. IEEE Access, 12, 126088–126107.
    DOI: 10.1109/ACCESS.2024.3455563

Computationally intensive stages (discretisation, transaction construction,
pattern mining, matrix assembly) run at native speed via a compiled C++
extension with optional OpenMP parallelism.  The Python layer handles
DataFrame ingestion, column-type detection, downstream estimation,
explanation methods, monitoring, and drift detection.

Architecture
------------
C++ extension (_hugiml_core):
    Discretisation, transaction construction, top-K HUI pattern mining with
    information-gain filtering, bitmap-accelerated matrix assembly, OpenMP
    parallel pattern matching.

Python layer:
    Column-type detection (prepareXy), NaN/Inf imputation, downstream sklearn
    estimator (LogisticRegression default), explanation methods
    (get_hug_features, get_pattern_info, feature_importances), versioned
    model serialisation, prediction monitoring, multi-method drift detection,
    latency SLA enforcement, and graceful degradation under memory pressure.

Quick start
-----------
Two usage paths are supported:

**Path A — prepareXy** (recommended when the full dataset is available upfront)::

    from hugiml import HUGIMLClassifier

    clf = HUGIMLClassifier()
    X, y = clf.prepareXy(X_df, y_series)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)

    print(clf.model_summary())
    print(clf.feature_importances())

**Path B — allCols + origColumns** (cross-validation loops)::

    clf = HUGIMLClassifier(
        allCols=[int_cols, float_cols, cat_cols],
        origColumns=X_df.columns.tolist(),
    )
    clf.fit(X_train, y_train)

Monitoring and drift detection::

    clf.enable_monitoring()
    clf.predict_proba(X_new)
    print(clf.monitor.report())

    drift = clf.detect_drift(X_new)
    print(drift)

Versioned serialisation::

    clf.save_model("model.hugiml")
    clf2 = HUGIMLClassifier.load_model("model.hugiml")
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
import os
import threading
import time
import tracemalloc
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from hugiml._binning import (
    _apply_edges as _adap_apply_edges,
)
from hugiml._binning import (
    _quantile_edges as _adap_quantile_edges,
)
from hugiml._binning import (
    _select_b as _adap_select_b,
)
from hugiml._compat import check_array, check_X_y
from hugiml.exceptions import (
    HUGIMLConvergenceWarning,
    HUGIMLDtypeDriftWarning,
    HUGIMLMemoryError,
    HUGIMLMiningError,
    HUGIMLParamError,
    HUGIMLPredictionError,
    HUGIMLRangeWarning,
    HUGIMLSchemaError,
    HUGIMLTimeoutError,
    HUGIMLValidationError,
    HUGIMLVersionError,
    HUGIMLWarning,
)
from hugiml.hyperparameter_configs import DEFAULT_HUGIML_GRID_NAME, get_hugiml_grid
from hugiml.monitoring import DriftDetector, PredictionMonitor
from hugiml.serialization import MIN_SCHEMA_VERSION, MODEL_SCHEMA_VERSION
from hugiml.serialization import load_model as _load_model
from hugiml.serialization import save_model as _save_model

try:
    import _hugiml_core as _core

    _CORE_AVAILABLE: bool = True
    _CORE_IMPORT_ERROR: ImportError | None = None
except ImportError as _e:
    _core = None
    _CORE_AVAILABLE = False
    _CORE_IMPORT_ERROR: ImportError | None = _e
    import warnings as _w

    _w.warn(
        f"hugiml-core: the compiled C++ extension '_hugiml_core' could not be imported "
        f"({_e}). "
        f"All classifier fit/transform operations will raise ImportError. "
        f"On Linux x86_64 with glibc >= 2.17, a pre-built wheel should have been installed "
        f"automatically — verify your platform matches a published wheel at "
        f"https://pypi.org/project/hugiml-core/#files. "
        f"To build from source: pip install . --no-build-isolation",
        RuntimeWarning,
        stacklevel=2,
    )
    del _w


DEFAULT_AUGMENTED_PAIR_MAX_FEATURES = 10
DEFAULT_AUGMENTED_PAIR_UNBOUNDED_CAP = 100
AUGMENTED_PAIR_OPS = ("product", "absolute_difference", "sum", "signed_difference")
AUGMENTED_PAIR_MODES = ("interaction_information", "marginal_ig")
_II_N_BINS = 4


def _best_ig_score(score_obj: Any) -> float:
    """Return the best finite IG score from native adaptive-binning metadata."""
    if isinstance(score_obj, dict):
        vals: list[float] = []
        for value in score_obj.values():
            try:
                fval = float(value)
            except Exception:
                continue
            if np.isfinite(fval):
                vals.append(fval)
        return max(vals) if vals else 0.0
    try:
        fval = float(score_obj)
    except Exception:
        return 0.0
    return fval if np.isfinite(fval) else 0.0


def _dense_full_csr(Z: np.ndarray) -> csr_matrix:
    """Convert a dense, mostly non-zero float block to CSR without scanning."""
    n_rows, n_cols = Z.shape
    if n_cols == 0:
        return csr_matrix((n_rows, 0), dtype=np.float32)
    data = np.ascontiguousarray(Z, dtype=np.float32).ravel()
    indices = np.tile(np.arange(n_cols, dtype=np.int32), n_rows)
    indptr = np.arange(0, (n_rows + 1) * n_cols, n_cols, dtype=np.int32)
    return csr_matrix((data, indices, indptr), shape=(n_rows, n_cols), dtype=np.float32)


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = counts.astype(np.float64, copy=False) / total
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log2(probs)))


def _information_gain_from_codes(
    feature_codes: np.ndarray, y_codes: np.ndarray, n_classes: int
) -> float:
    """Return IG(y; feature) for integer-coded feature values."""
    y_codes = np.asarray(y_codes, dtype=np.int64)
    feature_codes = np.asarray(feature_codes, dtype=np.int64)
    valid = (feature_codes >= 0) & (y_codes >= 0)
    if not np.any(valid):
        return 0.0
    f = feature_codes[valid]
    yv = y_codes[valid]
    base = _entropy_from_counts(np.bincount(yv, minlength=n_classes))
    if base <= 0.0:
        return 0.0
    _, inv = np.unique(f, return_inverse=True)
    cond = 0.0
    n = float(len(yv))
    for code in range(int(inv.max()) + 1):
        mask = inv == code
        if not np.any(mask):
            continue
        weight = float(np.sum(mask)) / n
        cond += weight * _entropy_from_counts(np.bincount(yv[mask], minlength=n_classes))
    return max(0.0, float(base - cond))


def _continuous_to_quantile_codes(values: np.ndarray, max_bins: int = 16) -> np.ndarray:
    """Quantile-code a continuous column for strict topK IG ranking."""
    arr = np.asarray(values, dtype=np.float64)
    codes = np.full(arr.shape[0], -1, dtype=np.int64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return codes
    vals = arr[finite]
    uniq = np.unique(vals)
    if uniq.size <= max_bins:
        _, inv = np.unique(vals, return_inverse=True)
        codes[finite] = inv.astype(np.int64, copy=False)
        return codes
    qs = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
    edges = np.unique(np.quantile(vals, qs))
    if edges.size == 0:
        codes[finite] = 0
    else:
        codes[finite] = np.searchsorted(edges, vals, side="right").astype(np.int64, copy=False)
    return codes


def _codes_from_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    codes = np.full(arr.shape[0], -1, dtype=np.int64)
    finite = np.isfinite(arr)
    if not np.any(finite) or len(edges) < 2:
        return codes
    n_bins = max(1, len(edges) - 1)
    if len(edges) == 2:
        codes[finite] = 0
    else:
        codes[finite] = np.clip(np.digitize(arr[finite], edges[1:-1]), 0, n_bins - 1).astype(
            np.int64, copy=False
        )
    return codes


def _edge_information_gain(
    values: np.ndarray,
    y_codes: np.ndarray,
    n_classes: int,
    n_bins: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if int(finite.sum()) < 10:
        edges = (
            _adap_quantile_edges(arr[finite], max(2, int(n_bins)))
            if np.any(finite)
            else np.array([0.0, 1.0])
        )
        return 0.0, edges, _codes_from_edges(arr, edges)
    edges = _adap_quantile_edges(arr[finite], max(2, int(n_bins)))
    codes = _codes_from_edges(arr, edges)
    return _information_gain_from_codes(codes, y_codes, n_classes), edges, codes


def _joint_information_gain_from_binned_columns(
    left_values: np.ndarray,
    right_values: np.ndarray,
    y_codes: np.ndarray,
    n_classes: int,
    left_bins: int,
    right_bins: int,
) -> tuple[float, float, float]:
    left_score, left_edges, left_codes = _edge_information_gain(
        left_values, y_codes, n_classes, left_bins
    )
    right_score, right_edges, right_codes = _edge_information_gain(
        right_values, y_codes, n_classes, right_bins
    )
    valid = (left_codes >= 0) & (right_codes >= 0)
    if int(valid.sum()) < 10:
        return 0.0, left_score, right_score
    right_width = max(1, len(right_edges) - 1)
    joint_codes = np.full(left_codes.shape[0], -1, dtype=np.int64)
    joint_codes[valid] = left_codes[valid] * right_width + right_codes[valid]
    joint_score = _information_gain_from_codes(joint_codes, y_codes, n_classes)
    return float(joint_score), float(left_score), float(right_score)


class NativeAugmentedPairTransformBlock:
    """Native-backed augmented-pair source selection and feature generation.

    The default source-selection mode uses native interaction-information
    scoring over observed values. Missing values are skipped while source
    features and pair candidates are scored. Reference-value substitution is
    applied only when retained augmented-pair features are materialized.

    ``augmented_pair_mode='interaction_information'`` selects source columns
    using native interaction evidence. ``ii_partner_size`` optionally bounds
    the raw interaction search by sampling partner anchors. ``aug_feature_size``
    is the number of source columns retained for native pair generation.

    ``augmented_pair_mode='marginal_ig'`` uses the established marginal-IG
    source-selection path where ``max_pair_features`` controls the number of
    selected source columns.
    """

    def __init__(
        self,
        augmented_pair_mode: str = "interaction_information",
        aug_feature_size: int = DEFAULT_AUGMENTED_PAIR_MAX_FEATURES,
        max_pair_features: int = DEFAULT_AUGMENTED_PAIR_MAX_FEATURES,
        ii_partner_size: int | None = None,
        budget_topK: int | None = None,
        min_source_ig: float | None = None,
        unbounded_cap: int = DEFAULT_AUGMENTED_PAIR_UNBOUNDED_CAP,
    ) -> None:
        self.augmented_pair_mode = str(augmented_pair_mode)
        self.aug_feature_size = int(aug_feature_size)
        self.max_pair_features = int(max_pair_features)
        self.ii_partner_size = None if ii_partner_size is None else int(ii_partner_size)
        self.budget_topK = None if budget_topK is None else int(budget_topK)
        self.min_source_ig = None if min_source_ig is None else float(min_source_ig)
        self.unbounded_cap = int(unbounded_cap)

    def _as_frame(
        self, X: Any, cols: list[str], full_feature_names: list[str] | None = None
    ) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            schema = list(full_feature_names or getattr(self, "input_feature_names_", []) or [])
            if schema and len(schema) == arr.shape[1]:
                names = schema
            else:
                names = [f"col{j}" for j in range(arr.shape[1])]
            X_df = pd.DataFrame(arr, columns=names)
        missing = [col for col in cols if col not in X_df.columns]
        if missing:
            X_df = X_df.copy()
            for col in missing:
                X_df[col] = np.nan
        return X_df

    def _selected_numeric_matrix(self, X: Any, cols: list[str] | None = None) -> np.ndarray:
        selected = list(cols or getattr(self, "selected_aug_features_", []))
        n_rows = len(X) if hasattr(X, "__len__") else 0
        if not selected:
            return np.zeros((n_rows, 0), dtype=np.float64)
        X_df = self._as_frame(X, selected, list(getattr(self, "input_feature_names_", []) or []))
        try:
            mat = X_df.reindex(columns=selected).to_numpy(dtype=np.float64, copy=True)
        except Exception:
            mat = np.column_stack(
                [
                    pd.to_numeric(X_df[col], errors="coerce").to_numpy(dtype=np.float64)
                    for col in selected
                ]
            )
        return np.ascontiguousarray(mat, dtype=np.float64)

    def _pair_index_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = {col: idx for idx, col in enumerate(getattr(self, "selected_aug_features_", []))}
        left: list[int] = []
        right: list[int] = []
        ops: list[int] = []
        for spec in getattr(self, "kept_specs_", []):
            a, b = spec["inputs"]
            left.append(pos[a])
            right.append(pos[b])
            operation = str(spec["operation"])
            op_map = {
                "product": 0,
                "absolute_difference": 1,
                "sum": 2,
                "signed_difference": 3,
            }
            if operation not in op_map:
                raise HUGIMLParamError(f"Unknown augmented-pair operation: {operation!r}.")
            ops.append(op_map[operation])
        return (
            np.asarray(left, dtype=np.int64),
            np.asarray(right, dtype=np.int64),
            np.asarray(ops, dtype=np.int8),
        )

    def fit(
        self,
        X: Any,
        y: Any,
        ig_scores: dict[str, Any],
        bin_edges: dict[str, Any],
        numeric_cols: list[str],
        budget_topK: int | None = None,
        min_source_ig: float | None = None,
        full_feature_names: list[str] | None = None,
    ) -> NativeAugmentedPairTransformBlock:
        if not (
            _CORE_AVAILABLE
            and hasattr(_core, "score_pair_candidates")
            and hasattr(_core, "transform_pair_features")
        ):
            raise HUGIMLParamError(
                "Native augmented pair transforms require _hugiml_core.score_pair_candidates "
                "and _hugiml_core.transform_pair_features. Rebuild the native extension."
            )
        self.input_feature_names_ = list(full_feature_names or [])
        if budget_topK is not None:
            self.budget_topK = int(budget_topK)
        if min_source_ig is not None:
            self.min_source_ig = float(min_source_ig)
        # ``None`` means no augmented-pair pre-budget. This is used by
        # topk_budget_strict=True so the single global topK filter ranks
        # original + HUG pattern + augmented-pair features together exactly once.
        # Negative values retain a safety cap for explicitly
        # unbounded non-strict augmented-pair selection.
        if self.budget_topK is not None and self.budget_topK < 0:
            self.budget_topK = max(0, self.unbounded_cap)

        min_ig = max(1e-12, float(self.min_source_ig or 0.0))
        self.min_source_ig_ = min_ig

        # Augmented-pair source selection supports two transparent scoring
        # policies. ``marginal_ig`` ranks individual numeric columns by their
        # best single-feature adaptive-bin IG. ``interaction_information``
        # ranks source columns by pair-context evidence from the native
        # interaction-information scorer, retaining the best partner and
        # marginal IG values for traceability while keeping the generated
        # product/difference/sum transforms outside HUG pattern mining.
        if self.augmented_pair_mode == "interaction_information":
            if not (_CORE_AVAILABLE and hasattr(_core, "select_interaction_information_features")):
                raise HUGIMLParamError(
                    "interaction_information augmented-pair mode requires "
                    "_hugiml_core.select_interaction_information_features."
                )
            X_all_numeric = self._selected_numeric_matrix(X, list(numeric_cols))
            selected = _core.select_interaction_information_features(
                X_all_numeric,
                np.asarray(pd.factorize(np.asarray(y), sort=True)[0], dtype=np.int64),
                list(numeric_cols),
                int(self.aug_feature_size),
                self.ii_partner_size,
            )
            scored: list[tuple[float, str]] = [
                (float(item["score"]), str(item["name"])) for item in selected
            ]
            self.selected_aug_features_ = [col for _, col in scored]
            self.selected_aug_scores_ = {col: float(score) for score, col in scored}
            self.augmented_pair_source_scores_ = [dict(item) for item in selected]
        else:
            scored: list[tuple[float, str]] = []
            for col in numeric_cols:
                score = _best_ig_score((ig_scores or {}).get(col, {}))
                if score >= min_ig:
                    scored.append((score, col))
            scored.sort(key=lambda item: (-item[0], item[1]))
            self.selected_aug_features_ = [
                col for _, col in scored[: min(self.max_pair_features, len(scored))]
            ]
            self.selected_aug_scores_ = {col: float(score) for score, col in scored}
            self.augmented_pair_source_scores_ = [
                {"name": col, "score": float(score), "mode": "marginal_ig"}
                for score, col in scored[: min(self.max_pair_features, len(scored))]
            ]

        self.input_bin_edges_ = {
            col: (
                np.asarray(bin_edges[col], dtype=float).tolist()
                if col in (bin_edges or {})
                else None
            )
            for col in self.selected_aug_features_
        }

        X_selected = self._selected_numeric_matrix(X, self.selected_aug_features_)
        if X_selected.shape[1] == 0:
            self.source_observed_medians_ = {}
            self.source_observed_medians_array_ = np.zeros(0, dtype=np.float64)
            self.numeric_medians_ = {}
            self.numeric_medians_array_ = np.zeros(0, dtype=np.float64)
            self.kept_specs_ = []
            self.candidate_count_ = 0
            self.feature_names_ = []
            self.augmented_pair_transforms_ = []
            self.augmented_pair_native_used_ = True
            self.scaler_mean_ = np.zeros(0, dtype=np.float64)
            self.scaler_scale_ = np.zeros(0, dtype=np.float64)
            self.pair_reference_values_ = np.zeros(0, dtype=np.float64)
            return self

        observed_medians = np.nanmedian(
            np.where(np.isfinite(X_selected), X_selected, np.nan), axis=0
        )
        observed_medians = np.where(np.isfinite(observed_medians), observed_medians, 0.0).astype(
            np.float64, copy=False
        )
        self.source_observed_medians_array_ = observed_medians
        self.source_observed_medians_ = {
            col: float(observed_medians[j]) for j, col in enumerate(self.selected_aug_features_)
        }
        self.numeric_medians_array_ = observed_medians
        self.numeric_medians_ = dict(self.source_observed_medians_)

        y_codes, _ = pd.factorize(np.asarray(y), sort=True)
        native_specs = _core.score_pair_candidates(
            X_selected,
            np.asarray(y_codes, dtype=np.int64),
            list(self.selected_aug_features_),
        )
        self.augmented_pair_native_used_ = True
        candidates = list(native_specs)
        candidates.sort(key=lambda item: (-float(item["transform_ig"]), str(item["name"])))
        self.candidate_count_ = len(candidates)
        keep_n = (
            len(candidates)
            if self.budget_topK is None
            else min(int(self.budget_topK), len(candidates))
        )
        self.kept_specs_ = candidates[:keep_n]
        self.feature_names_ = [str(spec["name"]) for spec in self.kept_specs_]

        if self.kept_specs_:
            left, right, ops = self._pair_index_arrays()
            pair_refs = np.asarray(
                [float(spec.get("reference_raw_value", 0.0)) for spec in self.kept_specs_],
                dtype=np.float64,
            )
            self.pair_reference_values_ = pair_refs
            raw = _core.transform_pair_features(
                X_selected,
                left,
                right,
                ops,
                pair_refs,
                np.zeros(len(self.kept_specs_), dtype=np.float64),
                np.ones(len(self.kept_specs_), dtype=np.float64),
            ).astype(np.float64, copy=False)
            self.scaler_mean_ = pair_refs.copy()
            centered = raw - pair_refs.reshape(1, -1)
            scale = np.sqrt(np.mean(centered * centered, axis=0))
            self.scaler_scale_ = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0).astype(
                np.float64, copy=False
            )
            self.left_indices_ = left
            self.right_indices_ = right
            self.op_codes_ = ops
        else:
            self.scaler_mean_ = np.zeros(0, dtype=np.float64)
            self.scaler_scale_ = np.zeros(0, dtype=np.float64)
            self.left_indices_ = np.zeros(0, dtype=np.int64)
            self.right_indices_ = np.zeros(0, dtype=np.int64)
            self.op_codes_ = np.zeros(0, dtype=np.int8)
            self.pair_reference_values_ = np.zeros(0, dtype=np.float64)
        self.augmented_pair_transforms_ = self._build_catalog()
        return self

    def transform(self, X: Any) -> csr_matrix:
        n_rows = len(X) if hasattr(X, "__len__") else 0
        if not getattr(self, "kept_specs_", []):
            return csr_matrix((n_rows, 0), dtype=np.float32)
        X_selected = self._selected_numeric_matrix(
            X, list(getattr(self, "selected_aug_features_", []))
        )
        Z = _core.transform_pair_features(
            X_selected,
            np.asarray(getattr(self, "left_indices_", np.zeros(0)), dtype=np.int64),
            np.asarray(getattr(self, "right_indices_", np.zeros(0)), dtype=np.int64),
            np.asarray(getattr(self, "op_codes_", np.zeros(0)), dtype=np.int8),
            np.asarray(
                getattr(
                    self, "pair_reference_values_", np.zeros(len(getattr(self, "kept_specs_", [])))
                ),
                dtype=np.float64,
            ),
            np.asarray(getattr(self, "scaler_mean_", np.zeros(0)), dtype=np.float64),
            np.asarray(getattr(self, "scaler_scale_", np.ones(0)), dtype=np.float64),
        )
        return _dense_full_csr(np.asarray(Z, dtype=np.float32))

    def _build_catalog(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        candidate_count = int(getattr(self, "candidate_count_", 0))
        means = np.asarray(getattr(self, "scaler_mean_", np.zeros(0)), dtype=np.float64)
        scales = np.asarray(getattr(self, "scaler_scale_", np.ones(0)), dtype=np.float64)
        for rank, spec in enumerate(getattr(self, "kept_specs_", []), start=1):
            left, right = list(spec["inputs"])
            mean = float(means[rank - 1]) if rank - 1 < means.size else 0.0
            scale = float(scales[rank - 1]) if rank - 1 < scales.size else 1.0
            raw_formula = str(spec["formula"])
            out.append(
                {
                    "name": str(spec["name"]),
                    "kind": "augmented_pair_transform",
                    "operation": str(spec["operation"]),
                    "inputs": [left, right],
                    "formula": raw_formula,
                    "raw_formula": raw_formula,
                    "standardized_formula": f"({raw_formula} - {mean:.12g}) / {scale:.12g}",
                    "standardization": {"mean": mean, "scale": scale},
                    "standardization_mean": mean,
                    "standardization_scale": scale,
                    "source_observed_medians": {
                        left: float(getattr(self, "source_observed_medians_", {}).get(left, 0.0)),
                        right: float(getattr(self, "source_observed_medians_", {}).get(right, 0.0)),
                    },
                    "pair_missing_policy": "reference_value_for_unavailable_pair",
                    "reference_raw_value": float(spec.get("reference_raw_value", mean)),
                    "eligible_count": int(spec.get("eligible_count", 0)),
                    "eligible_rate": float(spec.get("eligible_rate", np.nan)),
                    "missing_pair_rate": float(spec.get("missing_pair_rate", np.nan)),
                    "selected_by": (
                        "interaction_information"
                        if self.augmented_pair_mode == "interaction_information"
                        else "marginal_ig"
                    ),
                    "source_ig": {
                        left: float(self.selected_aug_scores_.get(left, 0.0)),
                        right: float(self.selected_aug_scores_.get(right, 0.0)),
                    },
                    "source_bin_edges": {
                        left: self.input_bin_edges_.get(left),
                        right: self.input_bin_edges_.get(right),
                    },
                    "transform_ig": float(spec["transform_ig"]),
                    "transform_bin_edges": spec.get("transform_bin_edges"),
                    "rank": rank,
                    "budget_topK": None if self.budget_topK is None else int(self.budget_topK),
                    "candidate_count": candidate_count,
                    "aug_feature_size": int(self.aug_feature_size),
                    "max_pair_features": int(self.max_pair_features),
                    "ii_partner_size": self.ii_partner_size,
                    "used_in_hugiml_mining": False,
                    "eligible_for_L2": False,
                    "integration_point": "before_downstream_lr",
                }
            )
        return out


# ---------------------------------------------------------------------------
# Helpers: RSS memory (Unix) with Windows fallback
# ---------------------------------------------------------------------------
try:
    import resource as _resource

    def _get_peak_rss_kb() -> int:
        return int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)

except ImportError:

    def _get_peak_rss_kb() -> int:
        try:
            import psutil

            return int(psutil.Process().memory_info().peak_wset) // 1024
        except ImportError:
            return 0


logger = logging.getLogger(__name__)


def _is_binary_feature_series(s: pd.Series) -> bool:
    """Return True when a column has exactly two observed values.

    Numeric-looking binary features (for example 0/1 or 1/2 flags) should use
    HUG-IML's categorical item path instead of constant-width numeric binning.
    Missing values are ignored for the inference; constant columns are not
    considered binary and keep the existing constant-column warning behavior.
    """
    try:
        observed = s.dropna()
    except AttributeError:
        observed = pd.Series(s).dropna()
    if observed.empty:
        return False
    try:
        return int(observed.nunique(dropna=True)) == 2
    except TypeError:
        normalized = observed.map(lambda value: repr(value))
        return int(normalized.nunique(dropna=True)) == 2


# =============================================================================
# Configuration presets
# =============================================================================

_PRESETS: dict[str, dict] = {
    "quick": dict(B=5, L=1, G=1e-2, topK=50),
    "balanced": dict(B=7, L=1, G=5e-3, topK=-1),
    "thorough": dict(B=-1, L=2, G=1e-4, topK=-1),
}


# =============================================================================
# Fit metadata
# =============================================================================


@dataclasses.dataclass(frozen=True)
class FitMetadata:
    """Immutable record of everything that happened during fit().

    Attributes
    ----------
    n_samples, n_features : int
        Training set dimensions.
    n_classes : int
        Number of distinct target classes.
    n_items : int
        Number of utility-annotated items (bins + categories).
    n_patterns : int
        Number of HUG patterns mined and retained.
    n_compound : int
        Compound patterns (length > 1).
    n_augmented_pairs : int
        Number of augmented pair features retained for the downstream estimator.
    n_downstream_features : int
        Number of columns used by the downstream estimator after feature-mode
        construction and optional strict TopK filtering.
    downstream_feature_counts : dict
        Counts by downstream feature family, for example original, pattern, and
        augmented_pair.
    topK_used : int
        Effective topK budget used during mining.
    stage_times_ms : dict[str, float]
        Wall-clock milliseconds per fit stage.
    total_fit_ms : float
        Total fit wall-clock milliseconds.
    matrix_density : float
        Fraction of non-zero entries in the training pattern matrix.
    config : dict
        Snapshot of (B, L, G, topK) as used.
    memory_peak_mb : float
        Python-traced peak memory during fit.
    memory_rss_mb : float
        RSS delta during fit (Unix only).
    memory_cpp_mb : float
        Estimated C++ extension memory usage.
    openmp_threads : int
        Number of OpenMP threads used.
    degraded : bool
        True when fit fell back to reduced parameters.
    """

    n_samples: int
    n_features: int
    n_classes: int
    n_items: int
    n_patterns: int
    n_compound: int
    topK_used: int
    stage_times_ms: dict
    total_fit_ms: float
    matrix_density: float
    config: dict
    n_augmented_pairs: int = 0
    n_downstream_features: int = 0
    downstream_feature_counts: dict = dataclasses.field(default_factory=dict)
    memory_peak_mb: float = 0.0
    memory_rss_mb: float = 0.0
    memory_cpp_mb: float = 0.0
    openmp_threads: int = 1
    degraded: bool = False

    def summary(self) -> str:
        """Return a single-line human-readable summary of the fit outcome."""
        downstream_text = (
            f", {self.n_augmented_pairs} augmented pairs, "
            f"{self.n_downstream_features} downstream features"
            if self.n_downstream_features
            else f", {self.n_augmented_pairs} augmented pairs"
        )
        return (
            f"{self.n_patterns} patterns "
            f"({self.n_compound} compound){downstream_text} from "
            f"{self.n_samples}×{self.n_features} in "
            f"{self.total_fit_ms:.0f} ms  "
            f"[density={self.matrix_density:.4f}]"
        )


# =============================================================================
# Memory measurement context manager
# =============================================================================

# tracemalloc is a process-global resource.  Concurrent fits on separate
# instances would race on is_tracing / start / stop without this lock.
_tracemalloc_lock = threading.Lock()


class _MemoryTracker:
    """Track peak memory during a code block via tracemalloc + RSS.

    Thread-safe: a module-level lock ensures that only one fit() at a time
    owns the tracemalloc session.  Other concurrent fits skip tracing and
    report traced_peak_mb = 0.0, which is clearly distinguished from a
    real measurement rather than a corrupted one.

    Parameters
    ----------
    enable_tracing : bool, default True
        When True (the original, unconditional behaviour -- used for
        execution_mode='audit'), tracks both tracemalloc's line-level
        allocation breakdown (``traced_peak_mb``) and process RSS
        (``rss_mb``), exactly as before this parameter was added.
        When False (used for execution_mode='production'), skips
        tracemalloc entirely -- no start/stop/snapshot/compare -- and
        reports traced_peak_mb=0.0, while still tracking the much
        cheaper RSS delta in rss_mb. tracemalloc's per-allocation
        overhead can dominate fit() wall time on categorical-heavy
        inputs (every categorical cell takes a failed-cast/exception
        round trip in the native layer before the GIL is released;
        each such exception is itself further allocation that
        tracemalloc has to record), so production workloads that don't
        need the line-level breakdown can opt out of paying for it.
    """

    def __init__(self, enable_tracing: bool = True) -> None:
        self.enable_tracing = enable_tracing

    def __enter__(self) -> _MemoryTracker:
        self._rss_before = _get_peak_rss_kb()
        if not self.enable_tracing:
            # production fast path: RSS only, no tracemalloc session at all.
            self._lock_acquired = False
            self._snap_before = None
            self._started = False
            return self
        self._lock_acquired = _tracemalloc_lock.acquire(blocking=False)
        self._snap_before: tracemalloc.Snapshot | None = None
        if self._lock_acquired:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                self._started = True
            else:
                self._started = False
            self._snap_before = tracemalloc.take_snapshot()
        else:
            self._started = False
            self._snap_before = None
        return self

    def __exit__(self, *exc: object) -> None:
        if self._lock_acquired:
            try:
                if self._snap_before is not None:
                    snap_after = tracemalloc.take_snapshot()
                    stats = snap_after.compare_to(self._snap_before, "lineno")
                    self.traced_peak_mb = sum(s.size for s in stats if s.size > 0) / 1e6
                else:
                    self.traced_peak_mb = 0.0
            finally:
                if self._started:
                    tracemalloc.stop()
                _tracemalloc_lock.release()
        else:
            self.traced_peak_mb = 0.0
        self.rss_mb = (_get_peak_rss_kb() - self._rss_before) / 1024

    @staticmethod
    def estimate_fit_mb(n: int, p: int, n_items: int, K: int) -> float:
        """Rough peak-memory estimate in MB for a fit() call."""
        disc_mb = n * p * 4 / 1e6
        trans_mb = n * p * 16 / 1e6
        ul_mb = n_items * n * 24 / 1e6
        matrix_mb = n * min(K, n_items) * 4 / 1e6
        overhead = 50
        return disc_mb + trans_mb + ul_mb + matrix_mb + overhead


# =============================================================================
# Transaction data wrapper (C++ ↔ Python bridge)
# =============================================================================


class _TransactionDataWrapper:
    """Augments native TransactionDataCpp with Python-compatible attributes.

    Stores exact C++ state (prefixed _cpp_) so that deserialized models can
    still run predict() via the pure-Python fallback transform.
    """

    def __init__(self, td_native: Any, classifier: HUGIMLClassifierNative) -> None:
        self._td = td_native
        self._clf = classifier

        self._cpp_bn2id = dict(td_native.bn2id)
        self._cpp_bkey_stride = int(td_native.bkey_stride)
        self._cpp_col_min = np.array(td_native.col_min, dtype=np.float64)
        self._cpp_col_range = np.array(td_native.col_range, dtype=np.float64)
        self._cpp_all_edges = [np.array(e, dtype=np.float64) for e in td_native.all_edges]
        self._cpp_nb_col = list(td_native.nb_col)
        self._cpp_is_cat = list(td_native.is_cat_v)
        self._cpp_is_int = list(td_native.is_int_v)
        # Store is_precoded_v so the Python fallback uses direct code→bi mapping
        # rather than MinMax-scaling the integer codes through the float path.
        self._cpp_is_precoded = list(td_native.is_precoded_v) if td_native.is_precoded_v else []
        self._cpp_cat_categories = [list(c) for c in td_native.cat_categories]

        self.bn2id = self._build_compat_bn2id()
        self.all_edges = self._cpp_all_edges
        self.col_range = self._cpp_col_range
        self.col_min = self._cpp_col_min
        self.is_cat = classifier.cat_cols_mask_
        self.is_int = classifier.is_int_mask_

    def __getattr__(self, name: str) -> Any:
        return getattr(self._td, name)

    def __getstate__(self) -> dict:
        state = {k: v for k, v in self.__dict__.items() if k not in ("_td", "_clf")}
        if self._td is not None:
            state["item_map"] = dict(self._td.item_map)
            state["item_twu"] = list(self._td.item_twu)
            state["nb_col"] = list(self._td.nb_col)
        return state

    def __setstate__(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)
        self._td = None

    def _build_compat_bn2id(self) -> dict:
        bn2id: dict = {}
        item_map = self._td.item_map
        feature_items: dict[str, list] = {}
        for item_id, label in item_map.items():
            if "=" in label:
                feat_name = label.split("=")[0]
                feature_items.setdefault(feat_name, []).append(item_id)
        feature_names = getattr(self._clf, "feature_names_in_", None) or self._clf.origColumns
        if feature_names is None:
            return bn2id
        stride = self._cpp_bkey_stride
        for col_idx, feat_name in enumerate(feature_names):
            if feat_name in feature_items:
                for bin_idx, item_id in enumerate(sorted(feature_items[feat_name])):
                    bn2id[(col_idx * stride) + bin_idx] = item_id
        return bn2id


# =============================================================================
# HUGIMLClassifierNative
# =============================================================================


# =============================================================================
# ── Per-feature adaptive binning — module-level helpers ───────────────────────
#
# Imported from hugiml._binning — the single source of truth for all
# adaptive-binning maths.  Local aliases preserve every existing call-site
# inside this file without modification.
#
# =============================================================================


class HUGIMLClassifierNative(TransformerMixin, ClassifierMixin, BaseEstimator):
    """HUG-IML interpretable classifier — C++ accelerated, scikit-learn compatible.

    Extracts High Utility Gain (HUG) patterns from labelled tabular data,
    transforms the input into a binary pattern-presence matrix, and fits an
    interpretable downstream classifier.  The mined patterns are human-readable
    and serve as the primary source of model explanations.

    Parameters
    ----------
    allCols : list of 3 lists, optional
        ``[int_col_names, float_col_names, cat_col_names]``.
        Must be paired with ``origColumns``.
    origColumns : list of str, optional
        Ordered column names matching the columns of X passed to fit/predict.
    B : int, default 8
        Number of quantile bins per numerical feature when
        ``adaptive_binning=False``. With adaptive binning enabled, per-feature
        bin counts are selected from ``b_candidates``.
    L : int, default 1
        Maximum HUG pattern length. 1 = singletons; 2 = pairs; -1 = unlimited.
    G : float, default 1e-3
        Minimum information-gain threshold.
    topK : int, default 30
        Maximum number of patterns to retain. -1 computes automatically.
    base_estimator : sklearn estimator, optional
        Downstream classifier trained on the selected representation.
        Defaults to LogisticRegression.
    n_jobs : int, default 1
        Number of OpenMP threads. -1 uses all available cores.
    max_predict_ms : float or None
        Prediction latency budget in milliseconds.
    max_fit_seconds : float or None
        Wall-clock budget for the pattern-mining stage of fit(). Transaction
        preparation and downstream model fitting are not bounded by this value.
    adaptive_binning : bool, default True
        Select per-feature numeric bin counts using supervised information gain.
    b_candidates : list of int or None
        Candidate bin counts evaluated when adaptive binning is enabled.
    min_marginal_gain_ratio : float, default 0.02
        Elbow threshold for adaptive-binning marginal gain.
    feature_mode : {"patterns_only", "original_plus_patterns",
        "original_plus_interactions"}, default "patterns_only"
        Downstream representation used by fit/predict APIs. ``transform(X)``
        always returns the HUG pattern matrix.
    augmented_pair_transforms : bool, default True
        Enable downstream augmented-pair operator features for eligible
        ``L >= 2`` adaptive-binning models.
    augmented_pair_mode : {"interaction_information", "marginal_ig"},
        default "interaction_information"
        Source-column scorer for augmented-pair features.
    ii_partner_size : int or None
        Optional partner-search bound for interaction-information scoring.
    aug_feature_size : int, default 10
        Number of source columns retained for augmented-pair candidate
        generation in interaction-information mode.
    max_pair_features : int, default 10
        Source-column budget used by the marginal-IG augmented-pair mode.
    augmented_pair_max_features : int or None
        v1.1.11-compatible alias for the augmented-pair source budget. When
        provided with default new budgets, it maps to both ``aug_feature_size``
        and ``max_pair_features``.
    topk_budget_strict : bool, default False
        Apply one global ``topK`` cap across the constructed downstream
        representation.
    dense_downstream_max_width : int, default 200
        Width threshold below which downstream matrices may stay dense.
    execution_mode : {"audit", "production"}, default "audit"
        Artifact-retention mode.
    interaction_relaxed_mining : bool, default False
        Allow interaction-information survivors to participate in native mining
        as original-feature bins without creating augmented-pair operator
        columns. Mutually exclusive with augmented-pair transforms at
        ``L >= 2``.
    interaction_relaxed_feature_size : int, default 10
        Survivor-source budget for interaction-relaxed mining.
    verbose : bool, default False
        Emit INFO-level log messages during fit.

    Attributes (available after fit)
    ----------------------------------
    classes_           : ndarray — unique class labels.
    n_features_in_     : int — number of input features.
    feature_names_in_  : list or None — column names from training data.
    cat_cols_mask_     : ndarray[bool] — True for categorical columns.
    is_int_mask_       : ndarray[bool] — True for integer columns.
    td_                : _TransactionDataWrapper — discretisation artefacts.
    patterns_          : list — mined HUG patterns.
    x_train_hup_       : csr_matrix — binary training pattern matrix.
    model_             : Pipeline — fitted downstream estimator.
    fit_metadata_      : FitMetadata — timings, memory, pattern stats.
    monitor            : PredictionMonitor or None — prediction statistics.
    """

    _fit_lock: threading.RLock  # per-instance, created in __init__
    monitor: PredictionMonitor | None  # set by enable_monitoring() / disable_monitoring()
    feature_names_in_: list[str] | None  # set by prepareXy / _resolve_col_meta after fit

    # The grid values themselves live in hugiml.hyperparameter_configs, shared
    # with the benchmark runner and dashboard rather than declared here only.
    # See default_param_grid() below for the public, name-aware accessor.
    DEFAULT_PARAM_GRID: dict[str, list] = get_hugiml_grid(DEFAULT_HUGIML_GRID_NAME)

    def __init__(
        self,
        allCols: list | None = None,
        origColumns: list | None = None,
        B: int = 8,
        L: int = 1,
        G: float = 1e-3,
        topK: int = 30,
        base_estimator: Any = None,
        n_jobs: int = 1,
        max_predict_ms: float | None = None,
        max_fit_seconds: float | None = None,
        verbose: bool = False,
        # ── Adaptive binning ──────────────────────────────────────────────
        # When adaptive_binning=True each numerical feature is pre-discretised
        # to B_j quantile bins chosen by elbow-stopping IG search.  The
        # pre-binned columns are declared categorical before the C++ layer
        # (global B is overridden to sentinel 2).  Bin edges are stored in
        # _bin_edges_ and reapplied identically at predict/transform time.
        #
        # adaptive_binning=True avoids requiring the user to guess a single
        # global B: on internal benchmarks, accuracy with a constant B varied
        # by several points of ROC-AUC depending on the value chosen (e.g.
        # B=3 vs B=20), with no single constant value best across datasets.
        # Adaptive binning lands at parity with a well-chosen constant B
        # without requiring a search for one; head-to-head testing shows it
        # roughly tied with, not uniformly ahead of, a well-chosen constant B,
        # so the benefit is removing a sensitive hyperparameter from the
        # quick-start path rather than a raw accuracy gain. Pass
        # adaptive_binning=False to use a single constant B instead.
        # ─────────────────────────────────────────────────────────────────
        adaptive_binning: bool = True,
        b_candidates: list | None = None,
        min_marginal_gain_ratio: float = 0.02,
        adaptive_binning_sample_frac: float | bool = False,
        adaptive_binning_sample_random_state: int = 42,
        feature_mode: str = "patterns_only",
        use_hotpath: bool = True,
        augmented_pair_transforms: bool = True,
        augmented_pair_mode: str = "interaction_information",
        ii_partner_size: int | None = None,
        aug_feature_size: int = 10,
        max_pair_features: int = 10,
        augmented_pair_max_features: int | None = None,
        topk_budget_strict: bool = False,
        dense_downstream_max_width: int = 200,
        execution_mode: str = "audit",
        # When True, interaction-information survivors can enter native mining
        # as ordinary source columns even when their marginal IG is weak.  The
        # relaxed admission applies at the root (depth-0) position only; see
        # native/mining.hpp's THUIsl::relaxed_cols comment for the exact scope.
        # No augmented-pair operator features (sum/product/etc.) are generated
        # by this path.  The resulting HUG patterns remain conjunctions of
        # original feature bins and are annotated for audit APIs as
        # survivor-led patterns when they include a relaxed survivor column.
        # Mutually exclusive with augmented_pair_transforms at L >= 2
        # (validated in _validate_params). At L=1 this is a no-op because L=1
        # uses the fused hotpath, which has no relaxed variant.  Effective for
        # L in {2, 3, -1}.
        interaction_relaxed_mining: bool = False,
        # Number of columns select_interaction_information_features returns
        # for interaction_relaxed_mining's own column selection. Separate
        # from aug_feature_size, which controls the unrelated
        # augmented_pair_mode source-column count.
        interaction_relaxed_feature_size: int = 10,
    ) -> None:
        self.allCols = allCols
        self.origColumns = origColumns
        self.B = B
        self.L = L
        self.G = G
        self.topK = topK
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.max_predict_ms = max_predict_ms
        self.max_fit_seconds = max_fit_seconds
        self.verbose = verbose
        self.adaptive_binning = adaptive_binning
        self.b_candidates = b_candidates
        self.min_marginal_gain_ratio = min_marginal_gain_ratio
        self.adaptive_binning_sample_frac = adaptive_binning_sample_frac
        self.adaptive_binning_sample_random_state = adaptive_binning_sample_random_state
        self.feature_mode = feature_mode
        self.use_hotpath = use_hotpath
        self.augmented_pair_transforms = augmented_pair_transforms
        self.augmented_pair_mode = augmented_pair_mode
        self.ii_partner_size = ii_partner_size
        if isinstance(augmented_pair_max_features, int) and not isinstance(
            augmented_pair_max_features, bool
        ):
            if (
                isinstance(aug_feature_size, int)
                and not isinstance(aug_feature_size, bool)
                and aug_feature_size == DEFAULT_AUGMENTED_PAIR_MAX_FEATURES
            ):
                aug_feature_size = augmented_pair_max_features
            if (
                isinstance(max_pair_features, int)
                and not isinstance(max_pair_features, bool)
                and max_pair_features == DEFAULT_AUGMENTED_PAIR_MAX_FEATURES
            ):
                max_pair_features = augmented_pair_max_features
        self.aug_feature_size = aug_feature_size
        self.max_pair_features = max_pair_features
        self.augmented_pair_max_features = augmented_pair_max_features
        self.topk_budget_strict = topk_budget_strict
        self.dense_downstream_max_width = dense_downstream_max_width
        # sklearn estimator compatibility: constructor and set_params must not
        # validate parameter values.  execution_mode is validated in fit/load.
        self.execution_mode = execution_mode
        self.interaction_relaxed_mining = interaction_relaxed_mining
        self.interaction_relaxed_feature_size = interaction_relaxed_feature_size
        self._fit_lock = threading.RLock()

    # ── Execution-mode retention helpers ─────────────────────────────────────

    def _is_production_mode(self) -> bool:
        """Return True when audit/governance-heavy artifacts are not retained."""
        return getattr(self, "execution_mode", "audit") == "production"

    def _audit_artifact_message(self, artifact: str) -> str:
        return (
            f"{artifact} is not available because this model was fitted or loaded with "
            "execution_mode='production'. Refit the model with execution_mode='audit' "
            "or load a model file that was originally saved from audit mode for complete "
            "traceability and audit/governance artifacts."
        )

    def _require_audit_artifact(self, artifact: str, *required_attrs: str) -> None:
        """Raise a clear error when an audit/governance artifact is unavailable.

        Passing no required attributes is treated as an unconditional audit-only
        guard in production mode.  When required attributes are supplied, their
        presence is checked in every execution mode: production receives the
        governance-oriented retention message, while audit mode receives a
        fitted-state/corrupt-state message.  This keeps callers such as
        ``get_pattern_info()`` and drift helpers from falling through to an
        ``AttributeError`` if an expected audit artifact is absent.
        """
        is_prod = self._is_production_mode()
        if not required_attrs:
            if is_prod:
                raise RuntimeError(self._audit_artifact_message(artifact))
            return
        missing = [
            attr
            for attr in required_attrs
            if (not hasattr(self, attr)) or getattr(self, attr, None) is None
        ]
        if missing:
            if is_prod:
                raise RuntimeError(self._audit_artifact_message(artifact))
            raise RuntimeError(
                f"{artifact} is unavailable because required fitted artifact(s) "
                f"are missing: {', '.join(missing)}. Refit the model or reload a "
                "complete audit-mode model file."
            )

    def _apply_execution_mode_retention(self) -> None:
        """Drop audit/governance-heavy training artifacts in production mode.

        Prediction-critical state is retained: td_, patterns_, model_, bin/scaler
        metadata, selected downstream names/masks, augmented-pair transform
        metadata, and privacy-safe aggregate downstream metadata that was
        already cached before retention. Training matrices, drift baselines, and
        native-only transient score caches are audit/governance artifacts and are
        intentionally omitted in production mode.
        """
        if not self._is_production_mode():
            return
        x_hup = getattr(self, "x_train_hup_", None)
        if x_hup is not None:
            self._training_pattern_matrix_shape_ = tuple(int(v) for v in x_hup.shape)
            self._training_pattern_matrix_nnz_ = int(getattr(x_hup, "nnz", 0))
        x_down = getattr(self, "x_train_downstream_", None)
        if x_down is not None:
            self._training_downstream_matrix_shape_ = tuple(int(v) for v in x_down.shape)
            if hasattr(x_down, "nnz"):
                self._training_downstream_matrix_nnz_ = int(x_down.nnz)
            elif hasattr(x_down, "shape"):
                self._training_downstream_matrix_nnz_ = int(np.count_nonzero(x_down))
            else:
                self._training_downstream_matrix_nnz_ = 0
        for attr in (
            "x_train_hup_",
            "x_train_downstream_",
            # _downstream_* aggregate metadata is retained: it is aligned to
            # downstream feature names and lets feature_importances() report a
            # stable schema after production retention without needing training
            # matrices.
            "_native_original_feature_scores_downstream_",
            "_drift_det",
        ):
            self.__dict__.pop(attr, None)

    # ── Class methods ─────────────────────────────────────────────────────────

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> HUGIMLClassifierNative:
        """Create a classifier from a named configuration preset.

        Parameters
        ----------
        name : {'quick', 'balanced', 'thorough'}
            quick     — B=5, L=1, G=1e-2, topK=50
            balanced  — B=7, L=1, G=5e-3, topK=-1
            thorough  — B=-1, L=2, G=1e-4, topK=-1

        Returns
        -------
        HUGIMLClassifierNative
        """
        if name not in _PRESETS:
            raise HUGIMLParamError(f"Unknown preset '{name}'.  Available: {list(_PRESETS)}")
        params = {**_PRESETS[name], **overrides}
        return cls(**params)

    @classmethod
    def default_param_grid(cls, name: str | None = None) -> dict[str, list]:
        """Return a named validation grid for compact HUGIML tuning.

        ``name`` selects either ``performance`` or ``interpretability``. When
        omitted, the performance grid is returned. Grid definitions live in
        :mod:`hugiml.hyperparameter_configs` and are copied before return so
        callers can narrow candidate values locally.

        The performance grid uses adaptive binning with ``B=-1``, searches
        ``L`` values ``1`` and ``2``, searches ``topK`` values ``50`` and
        ``100``, uses ``feature_mode='original_plus_patterns'``, and evaluates
        ``G`` values ``0.01`` and ``0.001``. The interpretability grid uses the
        same ``L``, ``topK``, and ``G`` values with
        ``feature_mode='patterns_only'``, ``interaction_relaxed_mining=True``,
        and ``augmented_pair_transforms=False``.
        """
        return get_hugiml_grid(name)

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        fitted = hasattr(self, "patterns_")
        status = f", {len(self.patterns_)} patterns" if fitted else ", not fitted"
        adap = ", adaptive" if self.adaptive_binning else ""
        mode = f", feature_mode={self.feature_mode}"
        aug = f", augmented_pair_transforms={self.augmented_pair_transforms}"
        exec_mode = f", execution_mode={self.execution_mode}"
        return f"HUGIMLClassifier(B={self.B}, L={self.L}, G={self.G}{adap}{mode}{aug}{exec_mode}{status})"

    # ── sklearn protocol ──────────────────────────────────────────────────────

    def get_params(self, deep: bool = True) -> dict:
        """Return constructor parameters as a dict (sklearn protocol)."""
        return dict(
            allCols=self.allCols,
            origColumns=self.origColumns,
            B=self.B,
            L=self.L,
            G=self.G,
            topK=self.topK,
            base_estimator=(copy.deepcopy(self.base_estimator) if deep else self.base_estimator),
            n_jobs=self.n_jobs,
            max_predict_ms=self.max_predict_ms,
            max_fit_seconds=self.max_fit_seconds,
            verbose=self.verbose,
            adaptive_binning=self.adaptive_binning,
            b_candidates=self.b_candidates,
            min_marginal_gain_ratio=self.min_marginal_gain_ratio,
            adaptive_binning_sample_frac=self.adaptive_binning_sample_frac,
            adaptive_binning_sample_random_state=self.adaptive_binning_sample_random_state,
            feature_mode=self.feature_mode,
            use_hotpath=self.use_hotpath,
            augmented_pair_transforms=self.augmented_pair_transforms,
            augmented_pair_mode=self.augmented_pair_mode,
            ii_partner_size=self.ii_partner_size,
            aug_feature_size=self.aug_feature_size,
            max_pair_features=self.max_pair_features,
            augmented_pair_max_features=self.augmented_pair_max_features,
            topk_budget_strict=self.topk_budget_strict,
            dense_downstream_max_width=self.dense_downstream_max_width,
            execution_mode=self.execution_mode,
            interaction_relaxed_mining=self.interaction_relaxed_mining,
            interaction_relaxed_feature_size=self.interaction_relaxed_feature_size,
        )

    def set_params(self, **params: Any) -> HUGIMLClassifierNative:
        """Set constructor parameters in-place and return self (sklearn protocol)."""
        # Defer validation until fit(), matching sklearn estimator conventions.
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _more_tags(self) -> dict:
        return {
            "requires_y": True,
            "binary_only": False,
            "poor_score": False,
            "X_types": ["2darray", "dataframe"],
            "allow_nan": False,
        }

    def __sklearn_tags__(self) -> Any:
        """Declare sklearn 1.6+ Tags, including TransformerTags for transform().

        ``__sklearn_tags__`` was introduced in sklearn 1.6.  Base classes on
        older installations do not implement it, so ``super().__sklearn_tags__()``
        raises ``AttributeError``.  Guard that call and return ``None`` when the
        parent chain does not support the protocol — callers must handle ``None``.
        """
        try:
            tags = super().__sklearn_tags__()
        except AttributeError:
            logger.debug(
                "super().__sklearn_tags__() raised AttributeError; "
                "sklearn base classes do not implement the tag protocol "
                "(expected sklearn >= 1.6).",
            )
            return None
        try:
            from sklearn.utils._tags import TransformerTags

            tags.transformer_tags = TransformerTags()
        except ImportError:
            logger.debug(
                "sklearn.utils._tags.TransformerTags not available; "
                "TransformerTags will not be declared.",
                exc_info=True,
            )
        return tags

    # ── Pickle protocol ───────────────────────────────────────────────────────

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_fit_lock", None)
        # Remove instance-level methods set by instrument_classifier():\n        # these closures are not picklable.
        state.pop("predict_proba", None)
        state.pop("predict", None)
        state["_schema_version_"] = MODEL_SCHEMA_VERSION
        if "patterns_" in state and state["patterns_"]:
            state["patterns_"] = [
                {"utility": pe.utility, "items": list(pe.items), "ig": pe.ig}
                for pe in state["patterns_"]
            ]
            state["_patterns_pickled_"] = True
        # serialize raw_patterns_ (also holds PatternEntry objects) ──
        # __getstate__ must convert patterns_ and raw_patterns_ consistently as
        # native PatternEntry objects, which are not picklable/deepcopyable.
        # Mirror the same dict-serialisation used for patterns_.
        if "raw_patterns_" in state and state["raw_patterns_"]:
            try:
                state["raw_patterns_"] = [
                    {"utility": pe.utility, "items": list(pe.items), "ig": pe.ig}
                    for pe in state["raw_patterns_"]
                ]
                state["_raw_patterns_pickled_"] = True
            except (AttributeError, TypeError):
                # Already serialized (dicts) or empty — leave as-is
                pass
        return state

    def __setstate__(self, state: dict) -> None:
        schema_ver = state.pop("_schema_version_", 1)
        if schema_ver < MIN_SCHEMA_VERSION:
            raise HUGIMLVersionError(
                f"Model schema version {schema_ver} is too old.  "
                f"Minimum supported: {MIN_SCHEMA_VERSION}.  Re-fit the model."
            )

        if state.pop("_patterns_pickled_", False):

            class _PE:
                __slots__ = ("utility", "items", "ig")

                def __init__(self, d: dict) -> None:
                    self.utility = d["utility"]
                    self.items = d["items"]
                    self.ig = d["ig"]

            state["patterns_"] = [_PE(d) for d in state["patterns_"]]

        # restore raw_patterns_ from its serialized dict form ──────
        if state.pop("_raw_patterns_pickled_", False):

            class _PE2:
                __slots__ = ("utility", "items", "ig")

                def __init__(self, d: dict) -> None:
                    self.utility = d["utility"]
                    self.items = d["items"]
                    self.ig = d["ig"]

            if "raw_patterns_" in state and state["raw_patterns_"]:
                state["raw_patterns_"] = [_PE2(d) for d in state["raw_patterns_"]]

        self.__dict__.update(state)
        # Drop unsupported multi-round attributes when loading serialized estimators.
        for _attr in (
            "n_rounds",
            "g_decay_factor",
            "pattern_selection",
            "transaction_weighting",
            "_boosting_round_tds_",
            "_boosting_round_pats_",
        ):
            self.__dict__.pop(_attr, None)
        self._fit_lock = threading.RLock()

        # ── Backward compatibility ───────────────────────────────────────
        # Models saved with v1.1.13 do not have adaptive-sampling state in
        # their pickle payload. Fill in missing attributes so deserialized
        # models keep their established behavior after restore.
        if not hasattr(self, "adaptive_binning"):
            self.adaptive_binning = False
        if not hasattr(self, "adaptive_binning_sample_frac"):
            self.adaptive_binning_sample_frac = False
        if not hasattr(self, "adaptive_binning_sample_random_state"):
            self.adaptive_binning_sample_random_state = 42
        if not hasattr(self, "b_candidates"):
            self.b_candidates = None
        if not hasattr(self, "min_marginal_gain_ratio"):
            self.min_marginal_gain_ratio = 0.02
        if not hasattr(self, "use_hotpath"):
            self.use_hotpath = True
        if not hasattr(self, "augmented_pair_transforms"):
            self.augmented_pair_transforms = True
        if not hasattr(self, "topk_budget_strict"):
            self.topk_budget_strict = False
        if not hasattr(self, "dense_downstream_max_width"):
            self.dense_downstream_max_width = 200
        if not hasattr(self, "execution_mode"):
            self.execution_mode = "audit"
        if not hasattr(self, "augmented_pair_mode"):
            self.augmented_pair_mode = "interaction_information"
        if not hasattr(self, "ii_partner_size"):
            self.ii_partner_size = None
        if not hasattr(self, "aug_feature_size"):
            self.aug_feature_size = 10
        if not hasattr(self, "max_pair_features"):
            self.max_pair_features = 10
        if not hasattr(self, "augmented_pair_max_features"):
            self.augmented_pair_max_features = None
        if not hasattr(self, "augmented_pair_transforms_"):
            self.augmented_pair_transforms_ = []
        if not hasattr(self, "augmented_pair_selected_features_"):
            self.augmented_pair_selected_features_ = []
        if not hasattr(self, "_original_feature_mask_downstream_"):
            self._original_feature_mask_downstream_ = None
        if not hasattr(self, "_original_selected_feature_names_downstream_"):
            self._original_selected_feature_names_downstream_ = None
        if not hasattr(self, "binary_categorical_cols_"):
            self.binary_categorical_cols_ = []
        if not hasattr(self, "_strict_topk_applied_during_construction_"):
            self._strict_topk_applied_during_construction_ = False
        if not hasattr(self, "_missing_col_edges_"):
            self._missing_col_edges_ = {}
        if not hasattr(self, "_adaptive_code_label_map_"):
            self._adaptive_code_label_map_ = {}
        # Rebuild the code→label map from stored bin edges whenever it's absent
        # or empty but adaptive bin edges are present.  This handles save/load via
        # both pickle and the custom .hugiml format (serialization.py).
        if (
            self.adaptive_binning
            and not self._adaptive_code_label_map_
            and getattr(self, "_bin_edges_", None)
        ):
            self._rebuild_adaptive_code_label_map()
        # ──────────────────────────────────────────────────────────────────

        if hasattr(self, "td_") and self.td_ is not None:
            td = self.td_
            self._native_available_ = not (hasattr(td, "_td") and td._td is None)
        else:
            self._native_available_ = False

    # ── Versioned save / load ─────────────────────────────────────────────────

    def save_model(self, path: str | os.PathLike) -> None:
        """Persist the fitted model to a binary file with schema versioning.

        Parameters
        ----------
        path : str or Path

        Raises
        ------
        HUGIMLSerializationError
        """
        _save_model(self, path)

    @classmethod
    def load_model(cls, path: str | os.PathLike) -> HUGIMLClassifierNative:
        """Load a model saved with :meth:`save_model`.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        HUGIMLClassifierNative

        Raises
        ------
        HUGIMLVersionError, HUGIMLSerializationError
        """
        return _load_model(path, expected_type=cls)  # type: ignore[no-any-return]

    # ── Data preparation ──────────────────────────────────────────────────────

    def prepareXy(self, X: pd.DataFrame, y: Any) -> tuple[pd.DataFrame, np.ndarray]:
        """Detect column types and encode the target variable.

        Call on the full dataset **before** any train/test split.  Records
        which columns are integer, float, or categorical, and performs
        basic label validation.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series or array-like

        Returns
        -------
        X : pd.DataFrame (copy with string column names)
        y : np.ndarray of int64
        """
        if not isinstance(X, pd.DataFrame):
            raise HUGIMLParamError(f"X must be a pandas DataFrame, got {type(X).__name__}")

        X = X.copy()
        X.columns = [str(c) for c in X.columns]

        if len(set(X.columns)) < len(X.columns):
            dups = {c for c in X.columns if list(X.columns).count(c) > 1}
            warnings.warn(
                f"Duplicate column names detected: {dups}.  Results may be unreliable.",
                HUGIMLWarning,
                stacklevel=2,
            )

        catCols: list[str] = []
        binaryCatCols: list[str] = []
        for idx, c in enumerate(X.columns):
            col = X.iloc[:, idx]
            is_explicit_cat = (
                pd.api.types.is_object_dtype(col)
                or pd.api.types.is_string_dtype(col)
                or isinstance(col.dtype, pd.CategoricalDtype)
            )
            is_binary_numeric = (
                not is_explicit_cat
                and (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col))
                and _is_binary_feature_series(col)
            )
            if is_explicit_cat or is_binary_numeric:
                catCols.append(c)
            if is_binary_numeric:
                binaryCatCols.append(c)
                X[c] = col.astype("category")

        intCols = [
            c
            for idx, c in enumerate(X.columns)
            if c not in set(catCols) and pd.api.types.is_integer_dtype(X.iloc[:, idx])
        ]

        for idx, c in enumerate(X.columns):
            if c not in catCols and X.iloc[:, idx].nunique() <= 1:
                warnings.warn(
                    f"Column '{c}' is constant and will produce zero utility.",
                    HUGIMLConvergenceWarning,
                    stacklevel=2,
                )

        X = X.reset_index(drop=True)
        self.feature_names_in_ = X.columns.tolist()
        self.cat_cols_mask_ = np.array([c in set(catCols) for c in X.columns], dtype=bool)
        self.is_int_mask_ = np.array([c in set(intCols) for c in X.columns], dtype=bool)
        self.binary_categorical_cols_ = list(binaryCatCols)

        y = np.asarray(y)
        try:
            y_float = y.astype(float)
            if np.isnan(y_float).any():
                raise HUGIMLValidationError("y contains NaN values.")
        except (ValueError, TypeError) as e:
            if "NaN" in str(e):
                raise

        if np.issubdtype(y.dtype, np.floating):
            if np.allclose(y, y.astype(int)):
                y = y.astype(np.int64)
            else:
                raise HUGIMLValidationError(
                    "y contains non-integer float values.  HUG-IML requires integer class labels."
                )

        return X, y

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _safe_cast_y(y: Any) -> np.ndarray:
        """Cast label array to int64, raising a clear error if y is non-finite.

        ``np.asarray(y, dtype=np.int64)`` silently produces undefined values
        (typically INT64_MIN = -9223372036854775808) when y contains NaN or
        infinity; the C++ mining kernel then processes garbage class codes.
        This helper converts via float first and raises ``HUGIMLValidationError``
        if any non-finite values are present, which is the correct contract
        for a classifier that requires integer class labels.
        """
        y_arr = np.asarray(y)
        if np.issubdtype(y_arr.dtype, np.floating) or y_arr.dtype == object:
            y_float = y_arr.astype(float, copy=False)
            if not np.all(np.isfinite(y_float)):
                raise HUGIMLValidationError(
                    "y contains NaN or infinite values. HUG-IML requires integer class labels."
                )
            return y_float.astype(np.int64)
        return y_arr.astype(np.int64)

    def _adaptive_binning_sample_fraction_for_native(self) -> float:
        frac = getattr(self, "adaptive_binning_sample_frac", False)
        if frac is False or frac is None:
            return 1.0
        return float(frac)

    @staticmethod
    def _to_float_array(arr: Any, cat_mask: np.ndarray | None = None) -> tuple:
        """Split input into a numeric array and raw categorical arrays.

        All-numeric float32 inputs remain float32 for native L1 preparation;
        other numeric inputs use float64.

        Adversarial-input hardening:
        - Forces writable copies of read-only column views.
        - Non-finite cells (NaN/Inf) in numerical columns are pre-converted
          to np.nan string-label bins by _prebin_nan_cols (fit) or
          _handle_test_nan (predict), so they arrive here as categorical.
          No median imputation is performed.
        """
        is_df = isinstance(arr, pd.DataFrame)
        n = len(arr)
        if is_df:
            p = len(arr.columns)
            arr_np: np.ndarray | None = None
        else:
            arr_np = np.asarray(arr)
            p = arr_np.shape[1]

        if cat_mask is None:
            cat_mask = np.zeros(p, dtype=bool)

        # Hot all-numeric path: keep float32 inputs as float32.  The native L1
        # bindings accept float32 directly, so avoiding the unconditional
        # float64 upcast removes a full n×p copy in scalability/production runs.
        # Other numeric dtypes use float64 to preserve historical precision.
        if not np.any(cat_mask):
            try:
                target_dtype = np.float64
                if is_df:
                    try:
                        dtypes = [getattr(dt, "type", None) for dt in arr.dtypes]
                        if dtypes and all(dt is np.float32 for dt in dtypes):
                            target_dtype = np.float32
                    except Exception:
                        target_dtype = np.float64
                    out = arr.to_numpy(dtype=target_dtype, copy=True)
                    return np.ascontiguousarray(out, dtype=target_dtype), [None] * p
                assert arr_np is not None
                if arr_np.dtype == np.float32:
                    return np.ascontiguousarray(arr_np, dtype=np.float32), [None] * p
                return np.ascontiguousarray(arr_np, dtype=np.float64), [None] * p
            except Exception:
                pass

        X_num = np.zeros(
            (n, p), dtype=np.float32 if getattr(arr_np, "dtype", None) == np.float32 else np.float64
        )
        X_cat_raw = [None] * p

        for j in range(p):
            if is_df:
                raw = arr.iloc[:, j]
            else:
                assert arr_np is not None  # nosec B101 – guaranteed by control flow
                raw = arr_np[:, j]
            if cat_mask[j]:
                col_obj = np.asarray(raw, dtype=object).copy()
                for i, v in enumerate(col_obj):
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        col_obj[i] = np.nan
                X_cat_raw[j] = col_obj
                X_num[:, j] = 0.0
            else:
                col = np.array(raw, dtype=X_num.dtype, copy=True)
                # Non-finite cells (NaN/Inf) are pre-handled by
                # _prebin_nan_cols (fit) and _handle_test_nan (predict)
                # before reaching here.  No median imputation.
                X_num[:, j] = col

        return X_num, X_cat_raw

    def _effective_topK_total(self, n_items: int | None = None) -> int:
        """Return the user-facing total topK pattern budget."""
        if self.topK != -1:
            return int(self.topK)
        nitems = int(n_items) if n_items is not None else 100
        nitems = max(nitems, 1)
        cap = 20000
        if self.L == 1:
            return min(nitems, cap)
        if isinstance(self.L, int) and self.L >= 2:
            return min(math.comb(nitems, min(self.L, nitems)), cap)
        return min(sum(math.comb(nitems, r) for r in range(1, min(nitems, 6) + 1)), cap)

    def _effective_topK(self, n_items: int | None = None) -> int:
        """Return the effective topK budget used by the single-pass miner."""
        return max(1, int(self._effective_topK_total(n_items)))

    def _effective_mining_topK(self, n_items: int | None = None) -> int:
        """Return the topK passed to the native miner.

        Mining uses exactly the requested topK budget. No expanded
        information-gain candidate pool and no round-wise budget splitting are
        used.
        """
        return self._effective_topK(n_items)

    def _select_patterns_for_budget(self, patterns: list, n_items: int | None = None) -> list:
        """Return mined patterns without post-mining IG oversampling/filtering."""
        return list(patterns)

    def _deduplicate_patterns_by_coverage(
        self, patterns: list, n_rows: int
    ) -> tuple[list, tuple | None]:
        """Remove duplicate HUG patterns with identical training-row coverage.

        This is intentionally a post-mining HUGIML optimization.  Raw top-k
        mining output remains non-deduplicated and can be compared directly
        with THUI/HMiner/brute-force utility oracles.

        Among patterns that cover exactly the same set of training rows, the
        one with the highest information gain (and highest utility on ties) is
        retained; all lower-ranked duplicates are discarded.  Scanning forward
        through the descending-IG-sorted list achieves this: the first time a
        coverage key is encountered it belongs to the best representative, so
        every subsequent occurrence of that key is marked for removal.
        fsK is intentionally not applied.

        COO cache: the matrix built here for coverage-key computation
        is a superset of the Stage-4 matrix (all raw patterns vs. survivors).
        We filter the COO in Python to keep only survivor columns and return
        the result alongside the pattern list, so Stage 4 can skip the second
        C++ build_train_matrix call entirely.

        Returns
        -------
        survivors : list of PatternEntry
        cached_coo : (rows_array, cols_array) filtered to survivors, or None
        """
        if not patterns:
            return [], None
        ordered = sorted(patterns, key=lambda pe: (-pe.ig, -pe.utility))
        if hasattr(_core, "build_train_matrix_csr"):
            indptr_raw, indices_raw = _core.build_train_matrix_csr(self.td_, ordered)
            indptr_np = np.asarray(indptr_raw, dtype=np.int64)
            cols_np = np.asarray(indices_raw, dtype=np.int64)
            rows_np = np.repeat(
                np.arange(max(len(indptr_np) - 1, 0), dtype=np.int64), np.diff(indptr_np)
            )
        else:
            rows_raw, cols_raw = _core.build_train_matrix(self.td_, ordered)
            rows_np = np.asarray(rows_raw, dtype=np.int64)
            cols_np = np.asarray(cols_raw, dtype=np.int64)

        # build coverage keys without Python-level int() conversions.
        # Sort COO by column index, then use searchsorted to split rows into
        # per-pattern arrays.  Hash each array via .tobytes() rather than
        # materializing a Python tuple, which avoids O(n) object allocation
        # per pattern.
        n_ordered = len(ordered)
        seen: set[bytes] = set()
        remove: set[int] = set()
        if len(rows_np) > 0:
            order = np.argsort(cols_np, kind="stable")
            rows_sorted = rows_np[order].astype(np.int64)
            cols_sorted = cols_np[order]
            # split_points[i] = start index of column i in sorted arrays
            split_points = np.searchsorted(cols_sorted, np.arange(n_ordered))
            split_points_end = np.append(split_points[1:], len(rows_sorted))
            for i in range(n_ordered):
                seg = rows_sorted[split_points[i] : split_points_end[i]]
                # Sort within segment for a canonical key regardless of COO order
                key = np.sort(seg).tobytes()
                if key in seen:
                    remove.add(i)
                else:
                    seen.add(key)
        survivors = [pe for i, pe in enumerate(ordered) if i not in remove]

        # Build the column-index remap: old ordered-index → new survivor index.
        # Then filter the raw COO to keep only survivor entries, remapping cols.
        survivor_old_indices = [i for i in range(len(ordered)) if i not in remove]
        old_to_new = {old: new for new, old in enumerate(survivor_old_indices)}
        keep_mask = np.isin(cols_np, survivor_old_indices)
        rows_filtered = rows_np[keep_mask].astype(np.int32)
        cols_filtered = np.array([old_to_new[int(c)] for c in cols_np[keep_mask]], dtype=np.int32)
        return survivors, (rows_filtered, cols_filtered)

    def _make_estimator(self, n_cls: int) -> Any:
        if self.base_estimator is not None:
            return copy.deepcopy(self.base_estimator)
        solver = "liblinear" if n_cls == 2 else "lbfgs"
        return LogisticRegression(solver=solver, random_state=0, max_iter=500)

    def _validate_params(self) -> None:
        if not isinstance(self.B, int):
            raise HUGIMLParamError(f"B must be int, got {type(self.B).__name__}")
        if self.B != -1 and self.B < 2:
            raise HUGIMLParamError(f"B must be -1 (auto) or >= 2, got {self.B}")
        if not isinstance(self.L, int):
            raise HUGIMLParamError(f"L must be int, got {type(self.L).__name__}")
        if not isinstance(self.G, (float, int)):
            raise HUGIMLParamError(f"G must be numeric, got {type(self.G).__name__}")
        if self.G < 0:
            raise HUGIMLParamError(f"G must be >= 0, got {self.G}")
        dense_width = getattr(self, "dense_downstream_max_width", 200)
        if isinstance(dense_width, bool) or not isinstance(dense_width, int):
            raise HUGIMLParamError(
                f"dense_downstream_max_width must be an int >= 0, got {type(dense_width).__name__}"
            )
        if int(dense_width) < 0:
            raise HUGIMLParamError(f"dense_downstream_max_width must be >= 0, got {dense_width}")
        if getattr(self, "execution_mode", "audit") not in {"audit", "production"}:
            raise HUGIMLParamError(
                "execution_mode must be either 'audit' or 'production'. "
                "Use 'audit' for complete traceability/governance artifacts, "
                "or 'production' to retain only prediction-critical state."
            )
        if self.allCols is not None or self.origColumns is not None:
            if self.allCols is None or self.origColumns is None:
                raise HUGIMLParamError("allCols and origColumns must both be supplied together.")
            if not (isinstance(self.allCols, list) and len(self.allCols) == 3):
                raise HUGIMLParamError("allCols must be [int_cols, float_cols, cat_cols].")
        # ── Adaptive binning params ──────────────────────────────────────
        if not isinstance(self.adaptive_binning, bool):
            raise HUGIMLParamError("adaptive_binning must be bool.")
        if self.b_candidates is not None:
            if (
                not isinstance(self.b_candidates, list)
                or len(self.b_candidates) == 0
                or not all(isinstance(b, int) and b >= 2 for b in self.b_candidates)
            ):
                raise HUGIMLParamError("b_candidates must be a non-empty list of int >= 2.")
        if not isinstance(self.min_marginal_gain_ratio, (float, int)):
            raise HUGIMLParamError("min_marginal_gain_ratio must be numeric.")
        if not 0 < float(self.min_marginal_gain_ratio) < 1:
            raise HUGIMLParamError(
                f"min_marginal_gain_ratio must be in (0, 1), got {self.min_marginal_gain_ratio}."
            )
        frac = self.adaptive_binning_sample_frac
        if isinstance(frac, bool):
            if frac is True:
                raise HUGIMLParamError(
                    "adaptive_binning_sample_frac must be False or a float in (0, 1]."
                )
        elif frac is not None:
            if not isinstance(frac, (float, int)):
                raise HUGIMLParamError(
                    "adaptive_binning_sample_frac must be False or a float in (0, 1]."
                )
            if not 0 < float(frac) <= 1:
                raise HUGIMLParamError(
                    f"adaptive_binning_sample_frac must be in (0, 1], got {frac}."
                )
        if not isinstance(self.adaptive_binning_sample_random_state, int) or isinstance(
            self.adaptive_binning_sample_random_state, bool
        ):
            raise HUGIMLParamError("adaptive_binning_sample_random_state must be an int.")
        allowed_feature_modes = {
            "patterns_only",
            "original_plus_patterns",
            "original_plus_interactions",
        }
        if self.feature_mode not in allowed_feature_modes:
            raise HUGIMLParamError(
                f"feature_mode must be one of {sorted(allowed_feature_modes)}, "
                f"got {self.feature_mode!r}."
            )
        if not isinstance(self.augmented_pair_transforms, bool):
            raise HUGIMLParamError(
                "augmented_pair_transforms must be bool, "
                f"got {type(self.augmented_pair_transforms).__name__}."
            )
        if not isinstance(self.topk_budget_strict, bool):
            raise HUGIMLParamError(
                f"topk_budget_strict must be bool, got {type(self.topk_budget_strict).__name__}."
            )
        if self.augmented_pair_mode not in AUGMENTED_PAIR_MODES:
            raise HUGIMLParamError(
                f"augmented_pair_mode must be one of {list(AUGMENTED_PAIR_MODES)}, "
                f"got {self.augmented_pair_mode!r}."
            )
        if self.augmented_pair_max_features is not None:
            if not isinstance(self.augmented_pair_max_features, int):
                raise HUGIMLParamError(
                    "augmented_pair_max_features must be int or None, "
                    f"got {type(self.augmented_pair_max_features).__name__}."
                )
            if self.augmented_pair_max_features < 2:
                raise HUGIMLParamError(
                    "augmented_pair_max_features must be >= 2 when provided, "
                    f"got {self.augmented_pair_max_features}."
                )
        if not isinstance(self.aug_feature_size, int):
            raise HUGIMLParamError(
                f"aug_feature_size must be int, got {type(self.aug_feature_size).__name__}."
            )
        if self.aug_feature_size < 2:
            raise HUGIMLParamError(f"aug_feature_size must be >= 2, got {self.aug_feature_size}.")
        if self.ii_partner_size is not None:
            if not isinstance(self.ii_partner_size, int):
                raise HUGIMLParamError(
                    f"ii_partner_size must be int or None, got {type(self.ii_partner_size).__name__}."
                )
            if self.ii_partner_size < 2:
                raise HUGIMLParamError(f"ii_partner_size must be >= 2, got {self.ii_partner_size}.")
        if self.augmented_pair_mode == "marginal_ig":
            if not isinstance(self.max_pair_features, int):
                raise HUGIMLParamError(
                    f"max_pair_features must be int, got {type(self.max_pair_features).__name__}."
                )
            if self.max_pair_features < 2:
                raise HUGIMLParamError(
                    f"max_pair_features must be >= 2, got {self.max_pair_features}."
                )
        if not isinstance(self.interaction_relaxed_mining, bool):
            raise HUGIMLParamError(
                "interaction_relaxed_mining must be bool, "
                f"got {type(self.interaction_relaxed_mining).__name__}."
            )
        if self.interaction_relaxed_mining:
            # Mutually exclusive with the augmented-pair operator-feature
            # path at L >= 2.
            if bool(self.augmented_pair_transforms) and isinstance(self.L, int) and self.L >= 2:
                raise HUGIMLParamError(
                    "interaction_relaxed_mining=True is mutually exclusive with "
                    "augmented_pair_transforms=True at L >= 2. Set "
                    "augmented_pair_transforms=False to use interaction_relaxed_mining, "
                    "or set interaction_relaxed_mining=False to use the augmented-pair "
                    "operator-feature path instead."
                )
            # L=1 always uses the fused hotpath, which has no relaxed
            # variant; relaxation is a no-op there. L in {2, 3, -1} route
            # through mine_patterns_relaxed_cpp's dispatcher.
            _supported_relaxed_L = (-1, 1, 2, 3)
            if isinstance(self.L, int) and self.L not in _supported_relaxed_L:
                raise HUGIMLParamError(
                    f"interaction_relaxed_mining=True only supports L in "
                    f"{_supported_relaxed_L}, got L={self.L}. Set "
                    "interaction_relaxed_mining=False to use this L value with "
                    "ordinary (unrelaxed) mining instead."
                )
        if not isinstance(self.interaction_relaxed_feature_size, int):
            raise HUGIMLParamError(
                "interaction_relaxed_feature_size must be int, "
                f"got {type(self.interaction_relaxed_feature_size).__name__}."
            )
        if self.interaction_relaxed_feature_size < 2:
            raise HUGIMLParamError(
                "interaction_relaxed_feature_size must be >= 2, "
                f"got {self.interaction_relaxed_feature_size}."
            )

    def _resolve_col_meta(self, X_train: Any) -> np.ndarray:
        """Determine column names and type masks from whichever setup path was used."""
        if hasattr(self, "cat_cols_mask_"):
            return self.cat_cols_mask_

        if self.allCols is not None and self.origColumns is not None:
            cat_set = set(self.allCols[2])
            int_set = set(self.allCols[0])
            col_list = list(self.origColumns)
            self.cat_cols_mask_ = np.array([c in cat_set for c in col_list], dtype=bool)
            self.is_int_mask_ = np.array([c in int_set for c in col_list], dtype=bool)
            self.binary_categorical_cols_ = []
            self.feature_names_in_ = col_list
            return self.cat_cols_mask_

        if isinstance(X_train, pd.DataFrame):
            col_list = X_train.columns.astype(str).tolist()
            cat_mask_values: list[bool] = []
            int_mask_values: list[bool] = []
            binary_cat_cols: list[str] = []
            for c in X_train.columns:
                col = X_train[c]
                is_explicit_cat = (
                    pd.api.types.is_object_dtype(col)
                    or pd.api.types.is_string_dtype(col)
                    or isinstance(col.dtype, pd.CategoricalDtype)
                )
                is_binary_numeric = (
                    not is_explicit_cat
                    and (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col))
                    and _is_binary_feature_series(col)
                )
                cat_mask_values.append(bool(is_explicit_cat or is_binary_numeric))
                int_mask_values.append(
                    bool((not is_binary_numeric) and pd.api.types.is_integer_dtype(col))
                )
                if is_binary_numeric:
                    binary_cat_cols.append(str(c))
            self.cat_cols_mask_ = np.array(cat_mask_values, dtype=bool)
            self.is_int_mask_ = np.array(int_mask_values, dtype=bool)
            self.binary_categorical_cols_ = list(binary_cat_cols)
            self.feature_names_in_ = col_list
            return self.cat_cols_mask_

        arr = np.asarray(X_train)
        if arr.ndim < 2:
            raise ValueError(
                f"HUGIMLClassifierNative expects a 2D array, got array of shape {arr.shape}."
            )
        p = arr.shape[1]
        self.cat_cols_mask_ = np.zeros(p, dtype=bool)
        self.is_int_mask_ = np.zeros(p, dtype=bool)
        self.binary_categorical_cols_ = []
        # Array inputs have no native column labels, but downstream components
        # (notably augmented-pair transforms) require stable feature names to
        # align IG scores, selected source columns, and transform-time matrices.
        # Use deterministic synthetic names for ndarray inputs.
        self.feature_names_in_ = [f"col{j}" for j in range(p)]
        return self.cat_cols_mask_

    def _identify_zero_variance_columns(self, X_train: Any) -> list[str]:
        """Return the names of columns with at most one distinct observed value.

        A column where every non-missing cell holds the same value (or every
        cell is missing) carries no mutual information with the target by
        construction, so it can never contribute an eligible item to the
        registry built during transaction preparation -- the existing
        information-gain computation already assigns such a column zero
        utility for every candidate bin or category. Recognising this here,
        before that computation runs, lets the per-row work building the
        item registry (categorical label extraction in particular, which is
        comparatively the most expensive part of it) skip these columns
        entirely rather than touching every row only to confirm what is
        already known from one cheap, vectorised distinct-value count.

        Called once at fit time; the resulting column names are stored on
        the fitted estimator (``_zero_variance_cols_``) so predict-time
        processing excludes exactly the same columns regardless of what
        values they happen to hold in new data -- the model never learned
        anything from this column during training, so it consistently
        ignores it afterward rather than re-deciding per call.
        """
        names = list(getattr(self, "feature_names_in_", []) or [])
        if not names:
            return []
        if isinstance(X_train, pd.DataFrame):
            zero_variance = [
                name
                for name in names
                if name in X_train.columns and X_train[name].nunique(dropna=True) <= 1
            ]
        else:
            arr = np.asarray(X_train, dtype=object)
            zero_variance = []
            for j, name in enumerate(names):
                if j >= arr.shape[1]:
                    continue
                observed: set[Any] = set()
                for value in arr[:, j]:
                    try:
                        missing = pd.isna(value)
                    except (TypeError, ValueError):
                        missing = False
                    if isinstance(missing, (list, tuple, np.ndarray)):
                        missing = False
                    if bool(missing):
                        continue
                    try:
                        observed.add(value)
                    except TypeError:
                        observed.add(repr(value))
                    if len(observed) > 1:
                        break
                if len(observed) <= 1:
                    zero_variance.append(name)
        return zero_variance

    def _exclude_zero_variance_columns(
        self,
        X_num: np.ndarray,
        X_cat_raw: list[Any],
        cat_mask: np.ndarray,
        zero_variance_names: list[str],
    ) -> None:
        """Neutralise the listed columns in-place so native code skips them cheaply.

        Categorical columns are excluded by setting their ``X_cat_raw`` entry
        to ``None``, which the native categorical-extraction step (shared by
        the fit-time and predict-time code paths) already treats as "no data
        for this column" and skips outright -- no native-side change needed,
        since that behaviour already exists for the ordinary case where a
        caller has no categorical columns at all. Numeric columns are
        neutralised by writing a single constant value through the column;
        since the column is already constant (or entirely missing) by the
        definition used to select it, this changes nothing about what the
        column represents while letting every downstream numeric pass over
        it (range computation, binning, correlation) settle on its answer
        immediately rather than scanning genuinely varying values.
        """
        if not zero_variance_names:
            return
        names = list(getattr(self, "feature_names_in_", []) or [])
        name_to_index = {name: j for j, name in enumerate(names)}
        for name in zero_variance_names:
            j = name_to_index.get(name)
            if j is None or j >= len(cat_mask):
                continue
            if cat_mask[j]:
                if j < len(X_cat_raw):
                    X_cat_raw[j] = None
            else:
                if X_num.shape[1] > j:
                    X_num[:, j] = 0.0

    @staticmethod
    def _timer() -> Any:
        """Return a lightweight timer object."""

        class _T:
            def __init__(self) -> None:
                self.start = time.perf_counter()

            @property
            def ms(self) -> float:
                return (time.perf_counter() - self.start) * 1000

        return _T()

    # ── Core fit ──────────────────────────────────────────────────────────────

    # ── Adaptive binning methods ──────────────────────────────────────────────

    def _rebuild_adaptive_code_label_map(self) -> None:
        """Reconstruct ``_adaptive_code_label_map_`` from stored ``_bin_edges_``.

        Called automatically by ``__setstate__`` after deserialization (both
        pickle and the custom .hugiml format) when the map is absent or empty.
        The map is not stored explicitly in the .hugiml format; it is always
        derived from ``_bin_edges_``, which IS persisted.

        The reconstruction exactly mirrors the map built in
        ``_apply_adaptive_binning``: for each column whose edges are stored, and
        for each bin k, the key is the C++ item-map label
        ``"name=[k.000,(k+1).000]"`` and the value is the original-scale label
        ``"name=[edges[k]:.4g, edges[k+1]:.4g)"``.
        """
        bin_edges = getattr(self, "_bin_edges_", {})
        if not bin_edges:
            return
        # We only build entries for columns that were encoded as is_precoded
        # (i.e., had no NaN at training time).  After deserialization we cannot
        # distinguish which columns were NaN-fallback vs is_precoded.  Safe
        # approach: rebuild for all columns in _bin_edges_.  The entry is only
        # consulted when the C++ item_map actually contains the key, so spurious
        # entries for NaN-fallback columns are harmless.
        new_map: dict[str, str] = {}
        for name, edges in bin_edges.items():
            n_bins = len(edges) - 1
            for k in range(n_bins):
                cpp_label = f"{name}=[{float(k):.3f},{float(k + 1):.3f}]"
                orig_label = f"{name}=[{edges[k]:.4g},{edges[k + 1]:.4g})"
                new_map[cpp_label] = orig_label
        self._adaptive_code_label_map_ = new_map
        self._adaptive_precoded_features_ = set(bin_edges)

    def _apply_adaptive_binning_cpp(self, X_train: Any, y_arr: np.ndarray) -> Any:
        """C++ replacement for _apply_adaptive_binning.

        Calls _core.select_adaptive_bins (C++ elbow_stop_nb_cpp) instead of
        the Python _adap_select_b loop.  Produces identical _bin_edges_,
        per_feature_b_, ig_scores_, _adaptive_code_label_map_, cat_cols_mask_,
        and X_pre outputs.  Falls back to the Python path on any error.
        """
        try:
            return self._apply_adaptive_binning_cpp_impl(X_train, y_arr)
        except (MemoryError, HUGIMLMemoryError):
            # A native allocation failure means the Python fallback is very
            # likely to allocate even more memory.  Surface a clean OOM instead
            # of cascading into an OS-level kill.
            raise
        except RuntimeError as exc:
            if "hugiml_timeout" in str(exc):
                raise
            logger.warning("C++ adaptive binning failed (%s); falling back to Python path.", exc)
            return self._apply_adaptive_binning(X_train, y_arr)
        except Exception as exc:
            logger.warning("C++ adaptive binning failed (%s); falling back to Python path.", exc)
            return self._apply_adaptive_binning(X_train, y_arr)

    def _apply_adaptive_binning_cpp_impl(self, X_train: Any, y_arr: np.ndarray) -> Any:
        """Implementation of the C++ adaptive B-selection path.

        Works for numeric-only and mixed (numeric + string categorical) DataFrames.
        Extracts only the numeric columns into a float64 array before calling
        select_adaptive_bins, avoiding the ValueError that X_df.to_numpy(float64)
        raises when string-categorical columns are present.
        """
        is_df = isinstance(X_train, pd.DataFrame)
        X_df = (
            X_train
            if is_df
            else pd.DataFrame(
                X_train,
                columns=(
                    list(getattr(self, "feature_names_in_", []) or [])
                    if getattr(self, "feature_names_in_", None) is not None
                    and len(getattr(self, "feature_names_in_", [])) == np.asarray(X_train).shape[1]
                    else None
                ),
            )
        )
        candidates = sorted(set(self.b_candidates or [2, 3, 5, 7, 10, 15]))
        ratio = self.min_marginal_gain_ratio
        cat_mask = self.cat_cols_mask_
        col_names = list(X_df.columns)
        n_cls = len(np.unique(y_arr))

        self._bin_edges_: dict = {}
        self.per_feature_b_: dict = {}
        self.ig_scores_: dict = {}

        # ── Extract only the numeric (non-cat) columns ────────────────────────
        # Passing the full mixed DataFrame to to_numpy(float64) raises ValueError
        # when string-categorical columns are present.  select_adaptive_bins
        # already skips is_cat columns, so we can pass only the numeric subset
        # and map indices back afterwards via num_col_map.
        num_col_map = [j for j, v in enumerate(cat_mask) if not v]
        col_names_num = [col_names[j] for j in num_col_map]
        # C++ binding expects std::string column names.  ndarray inputs create
        # integer DataFrame column names, so pass string labels to C++ while
        # retaining the original names for pandas indexing below.
        col_names_num_cpp = [str(name) for name in col_names_num]

        if not num_col_map:
            # All columns are categorical — nothing to adapt
            self._adaptive_code_label_map_: dict[str, str] = {}
            return X_train

        # Build X_num: shape (n, len(num_col_map)), float64
        X_num = X_df.iloc[:, num_col_map].to_numpy(dtype=np.float64, na_value=np.nan)
        y_int = y_arr.astype(np.int64)
        # Pass all-zeros is_cat so C++ processes every column in X_num
        is_cat_zeros = np.zeros(len(num_col_map), dtype=np.uint8)

        adap_result = _core.select_adaptive_bins(
            X_num,
            y_int,
            n_cls,
            col_names_num_cpp,
            is_cat_zeros,
            candidates,
            ratio,
            adaptive_binning_sample_frac=self._adaptive_binning_sample_fraction_for_native(),
            adaptive_binning_sample_random_state=int(self.adaptive_binning_sample_random_state),
        )

        # Pack C++ results into Python model attributes.
        # adap_result.num_col_indices[ci] is the index within X_num (= within
        # col_names_num).  Map back to the original X_df column index via num_col_map.
        for ci, col_res in enumerate(adap_result.cols):
            j_num = adap_result.num_col_indices[ci]
            name = col_names_num[j_num]
            edges = np.array(col_res.edges)
            self._bin_edges_[name] = edges
            # Match the Python adaptive path, which records the effective number
            # of stored bins after duplicate quantile edges have collapsed.
            self.per_feature_b_[name] = len(edges) - 1
            # Pad missing candidates (early elbow-stop) with 0.0 for diagnostics
            scores: dict[int, float] = {}
            for k, b in enumerate(candidates):
                scores[b] = col_res.ig_scores[k] if k < len(col_res.ig_scores) else 0.0
            self.ig_scores_[name] = scores

        # Build _adaptive_code_label_map_ and update column-type masks
        self._adaptive_code_label_map_: dict[str, str] = {}
        self._adaptive_precoded_features_ = set(self._bin_edges_)
        new_cat = cat_mask.copy()
        new_int = getattr(self, "is_int_mask_", np.zeros(len(col_names), dtype=bool)).copy()

        for name, edges in self._bin_edges_.items():
            if name not in X_df.columns:
                continue
            j = col_names.index(name) if name in col_names else -1
            if j >= 0:
                new_cat[j] = False
                new_int[j] = True
            n_bins = len(edges) - 1
            for k in range(n_bins):
                cpp_label = f"{name}=[{float(k):.3f},{float(k + 1):.3f}]"
                orig_label = f"{name}=[{edges[k]:.4g},{edges[k + 1]:.4g})"
                self._adaptive_code_label_map_[cpp_label] = orig_label

        self.cat_cols_mask_ = new_cat
        self.is_int_mask_ = new_int

        # Apply integer codes to the pre-binned numeric columns in X_pre.
        # Native storage is int32 with -1 as the missing sentinel; fetch one
        # column at a time to avoid materialising a second full float64 code
        # matrix.  Cast only the one column that pandas needs so NaN can be
        # represented for the legacy pre-binned path.
        X_pre = X_df.copy()
        for ci in range(adap_result.n_num_cols):
            j_num = adap_result.num_col_indices[ci]
            name = col_names_num[j_num]
            col_raw = pd.to_numeric(X_df[name], errors="coerce").values
            if hasattr(adap_result, "get_X_codes_col"):
                codes_i32 = np.asarray(adap_result.get_X_codes_col(ci), dtype=np.int32)
                missing_codes = codes_i32 < 0
            else:  # compatibility with older native wheels
                # Older native wheels expose get_X_codes() as float64 with
                # np.nan as the missing sentinel.  Casting that matrix directly
                # to int32 can platform-dependently produce either INT32_MIN or
                # 0; the latter silently aliases a valid bin.  Detect missing
                # sentinels before the integer cast.
                codes_raw = np.asarray(adap_result.get_X_codes()[:, ci])
                missing_codes = ~np.isfinite(codes_raw)
                codes_i32 = np.zeros(codes_raw.shape, dtype=np.int32)
                finite_codes = ~missing_codes
                if np.any(finite_codes):
                    codes_i32[finite_codes] = codes_raw[finite_codes].astype(np.int32)
            codes = codes_i32.astype(np.float32, copy=False)
            codes[missing_codes | (codes_i32 < 0) | (~np.isfinite(col_raw))] = np.nan
            X_pre[name] = codes

        return X_pre if is_df else X_pre

    def _apply_adaptive_binning(self, X_train: Any, y_arr: np.ndarray) -> Any:
        """Pre-discretise numerical features using per-feature IG-selected B_j.

        Called by _fit_impl when adaptive_binning=True.  The method:

        1. Iterates over numerical columns, runs _adap_select_b to choose B_j.
        2. Computes quantile edges on the training column and stores them in
           _bin_edges_ so they can be reapplied at predict time.
        3. Replaces each numerical column with string bin labels ([lo,hi)).
        4. Updates cat_cols_mask_ and is_int_mask_ to mark pre-binned features
           as categorical, so Stage 1 of _fit_impl routes them through the
           C++ categorical path.

        Returns the pre-binned DataFrame (or the input unchanged if no
        numerical features are found).
        """
        is_df = isinstance(X_train, pd.DataFrame)
        X_df = (
            X_train
            if is_df
            else pd.DataFrame(
                X_train,
                columns=(
                    list(getattr(self, "feature_names_in_", []) or [])
                    if getattr(self, "feature_names_in_", None) is not None
                    and len(getattr(self, "feature_names_in_", [])) == np.asarray(X_train).shape[1]
                    else None
                ),
            )
        )
        candidates = sorted(set(self.b_candidates or [2, 3, 5, 7, 10, 15]))
        ratio = self.min_marginal_gain_ratio
        cat_mask = self.cat_cols_mask_
        col_names = list(X_df.columns)

        self._bin_edges_: dict = {}
        self.per_feature_b_: dict = {}
        self.ig_scores_: dict = {}
        pre_binned: set = set()

        # Identify which columns need adaptive B-selection.
        num_cols = [
            (j, name) for j, name in enumerate(col_names) if not (j < len(cat_mask) and cat_mask[j])
        ]

        # parallel B-selection:
        def _select_one(j: int, name: str) -> tuple:
            col = pd.to_numeric(X_df.iloc[:, j], errors="coerce").values
            finite_mask = np.isfinite(col)
            if finite_mask.sum() < 10:
                chosen = candidates[len(candidates) // 2]
                scores = {b: 0.0 for b in candidates}
            else:
                chosen, scores = _adap_select_b(
                    col[finite_mask], y_arr[finite_mask], candidates, ratio
                )
            edges = _adap_quantile_edges(col, chosen)
            return name, chosen, scores, edges

        try:
            from joblib import Parallel
            from joblib import delayed as _delayed

            _n_jobs = self.n_jobs if hasattr(self, "n_jobs") else 1
            _results = Parallel(n_jobs=_n_jobs, prefer="threads")(
                _delayed(_select_one)(j, name) for j, name in num_cols
            )
        except Exception:
            _results = [_select_one(j, name) for j, name in num_cols]

        for name, chosen, scores, edges in _results:
            self.ig_scores_[name] = scores
            self.per_feature_b_[name] = len(edges) - 1
            self._bin_edges_[name] = edges
            pre_binned.add(name)

        # ── Integer-code path ─────────────────────────────────────
        # Encode pre-binned columns as integer codes (0..B_j-1) and route them
        # through the C++ is_int path, which is 8–20x faster than the old
        # string-categorical std::string path.
        #
        # NaN handling (hybrid):
        #   Columns WITHOUT any NaN/Inf → is_int=True (fast integer path).
        #   Columns WITH NaN/Inf        → cat=True  (existing string path,
        #       correct because C++ categorical skips np.nan cells).
        # In practice most adaptive-mode datasets have no NaN; the string
        # fallback only fires when needed.
        #
        # Label translation: the C++ integer path stores item labels as
        # "feat=[k,k+1]" (integer-range format).  get_hug_features() remaps
        # these to original-scale "[lo,hi)" labels via _adaptive_code_label_map_.
        self._adaptive_code_label_map_: dict[str, str] = {}
        self._adaptive_precoded_features_ = set(self._bin_edges_)

        new_cat = cat_mask.copy()
        new_int = getattr(self, "is_int_mask_", np.zeros(len(col_names), dtype=bool)).copy()

        X_pre = X_df.copy()
        for name, edges in self._bin_edges_.items():
            if name not in X_df.columns:
                continue
            col = pd.to_numeric(X_df[name], errors="coerce").values
            n_bins = len(edges) - 1
            has_nan = not np.isfinite(col).all()

            j = col_names.index(name) if name in col_names else -1

            # Integer-code path for all adaptive columns, with or without NaN.
            # NaN/Inf cells are encoded as np.nan (float64), which the C++
            # is_precoded handler reads as non-finite and maps to -1 (skipped
            # — no item generated for that row/feature pair).
            # This keeps the column numeric rather than converting the whole
            # column to object/string dtype for a small number of NaNs, which
            # would force the slower C++ categorical path.
            codes = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1).astype(np.float64)
            if has_nan:
                codes[~np.isfinite(col)] = np.nan  # sentinel: C++ skips
            X_pre[name] = codes
            if j >= 0:
                new_cat[j] = False
                new_int[j] = True
            # Build C++ label -> original-scale label translation.
            # Key format matches the C++ is_precoded label exactly:
            # std::setprecision(3) -> "name=[0.000,1.000]"
            for k in range(n_bins):
                cpp_label = f"{name}=[{float(k):.3f},{float(k + 1):.3f}]"
                orig_label = f"{name}=[{edges[k]:.4g},{edges[k + 1]:.4g})"
                self._adaptive_code_label_map_[cpp_label] = orig_label

        self.cat_cols_mask_ = new_cat
        self.is_int_mask_ = new_int
        return X_pre if is_df else X_pre

    def _prebin_for_predict(self, X: Any) -> Any:
        """Apply stored adaptive bin edges before C++ inference.

        The common ndarray path is kept entirely in NumPy to avoid constructing
        and copying a pandas DataFrame for every predict()/transform() call.
        DataFrame input still preserves labels and mixed categorical columns.
        """
        bin_edges = getattr(self, "_bin_edges_", {})
        if not bin_edges:
            return X

        feat_names = getattr(self, "feature_names_in_", None)
        code_label_map = getattr(self, "_adaptive_code_label_map_", {})
        precoded_features = getattr(self, "_adaptive_precoded_features_", None)
        if precoded_features is None:
            # Backward-compatible fallback for models saved before this attribute.
            precoded_features = set(bin_edges) if code_label_map else set()

        # Fast path: numeric ndarray input.  Adaptive fused L1 stores ndarray
        # feature names as col0, col1, ...; keep the output as ndarray so
        # _to_float_array can consume it without pandas overhead.
        if not isinstance(X, pd.DataFrame):
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            names = (
                list(feat_names)
                if feat_names is not None and len(feat_names) == arr.shape[1]
                else [f"col{j}" for j in range(arr.shape[1])]
            )
            name_to_idx = {name: j for j, name in enumerate(names)}
            # If a name mismatch occurs, fall back to the labelled path rather
            # than silently applying edges to the wrong column.
            if all(name in name_to_idx for name in bin_edges):
                X_out = np.array(arr, dtype=np.float64, copy=True)
                for name, edges in bin_edges.items():
                    j = name_to_idx[name]
                    n_bins = len(edges) - 1
                    col = X_out[:, j]
                    if name in precoded_features:
                        codes = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1).astype(
                            np.float64
                        )
                        nan_mask = ~np.isfinite(col)
                        if nan_mask.any():
                            codes[nan_mask] = np.nan
                        X_out[:, j] = codes
                    else:
                        # Legacy string categorical fallback requires labels.
                        # It is rare in current adaptive models, but preserve
                        # correctness by using the DataFrame path below.
                        break
                else:
                    return X_out

        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            X_df = X
        else:
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            cols = (
                list(feat_names)
                if feat_names is not None and len(feat_names) == arr.shape[1]
                else [f"col{j}" for j in range(arr.shape[1])]
            )
            X_df = pd.DataFrame(arr, columns=cols)

        # Fast labelled path for the common adaptive case: all stored adaptive
        # columns are numeric pre-coded features.  Convert once to NumPy, edit
        # columns in-place, and rebuild one DataFrame instead of assigning one
        # pandas Series per feature.
        #
        # This path is only taken when every column outside of bin_edges is
        # non-categorical per the fitted schema. A blanket float64 cast across
        # the whole DataFrame would otherwise destroy the categorical-ness of
        # columns that are not in bin_edges at all (e.g. binary-numeric
        # auto-categorical columns admitted via
        # interaction_relaxed_mining/select_interaction_information_features)
        # before _build_test_hup ever saw them, which would prevent the native
        # build_test_matrix_csr categorical-matching path from recognizing
        # them -- any mined pattern touching such a column would then evaluate
        # to 0 for every row at predict/transform time, even though the
        # training-time pattern matrix (built on correctly-typed training
        # data) is unaffected.
        #
        # The raw input's pandas dtype is NOT a reliable signal here: at
        # predict time callers normally pass the original int64/object
        # columns (not a pre-cast 'category' dtype), so checking the input
        # dtype directly under-detects this case. The source of truth for
        # which columns are categorical is the fitted model's own
        # cat_cols_mask_/feature_names_in_ (set during fit's schema
        # detection). The fast path is therefore only taken when EVERY
        # column outside of bin_edges is, per the FITTED schema, not
        # categorical -- i.e. the blanket float64 cast cannot lose
        # information the native layer needs. Otherwise we always fall
        # through to the slower column-preserving path below, which only
        # mutates the specific bin_edges columns and leaves every other
        # column untouched (so its dtype, and the categorical match against
        # X_cat_raw downstream, stays correct).
        _fit_cat_mask = getattr(self, "cat_cols_mask_", None)
        _fit_feat_names = getattr(self, "feature_names_in_", None)
        if (
            _fit_cat_mask is not None
            and _fit_feat_names is not None
            and len(_fit_feat_names) == len(_fit_cat_mask)
        ):
            _fit_cat_names = {
                str(name) for name, is_cat in zip(_fit_feat_names, _fit_cat_mask) if is_cat
            }
        else:
            # Unknown schema: be conservative and assume any column could be
            # categorical, which simply routes to the always-correct slow path.
            _fit_cat_names = {str(c) for c in X_df.columns}
        non_bin_edge_cols_all_noncat = not any(
            str(c) in _fit_cat_names for c in X_df.columns if c not in bin_edges
        )
        if (
            is_df
            and non_bin_edge_cols_all_noncat
            and all(name in X_df.columns and name in precoded_features for name in bin_edges)
        ):
            try:
                cols = list(X_df.columns)
                name_to_idx = {str(c): j for j, c in enumerate(cols)}
                X_mat = X_df.to_numpy(dtype=np.float64, copy=True)
                for name, edges in bin_edges.items():
                    j = name_to_idx[name]
                    n_bins = len(edges) - 1
                    col = X_mat[:, j]
                    codes = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1).astype(np.float64)
                    nan_mask = ~np.isfinite(col)
                    if nan_mask.any():
                        codes[nan_mask] = np.nan
                    X_mat[:, j] = codes
                return pd.DataFrame(X_mat, columns=X_df.columns, index=X_df.index)
            except Exception:
                pass

        X_out = X_df.copy()
        for name, edges in bin_edges.items():
            if name not in X_df.columns:
                continue
            col = pd.to_numeric(X_df[name], errors="coerce").values
            n_bins = len(edges) - 1
            if name in precoded_features:
                codes = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1).astype(np.float64)
                nan_mask = ~np.isfinite(col)
                if nan_mask.any():
                    codes[nan_mask] = np.nan
                X_out[name] = codes
            else:
                X_out[name] = _adap_apply_edges(col, edges)
        return X_out if is_df else X_out

    # ── Missing value handling methods ───────────────────────────────────────
    #
    # NaN (and Inf) in a numerical feature is treated as "not observed" —
    # no item is generated in the transaction for that (row, feature) pair.
    # This matches the categorical path, where np.nan in X_cat_raw is already
    # silently skipped by the C++ transaction builder.
    #
    # Numerical columns that contain non-finite values in training are
    # pre-binned to string labels (same mechanism as adaptive_binning) so the
    # C++ sees them as categorical.  Non-finite cells become np.nan in the
    # label array → C++ skips → no item.  Edges are stored in
    # _missing_col_edges_ and reused at predict time.
    #
    # At predict time, columns that were non-finite-free in training but
    # receive non-finite test values are handled dynamically using the C++
    # edge arrays stored in td_._cpp_all_edges[j].
    #
    # The old median imputation in _to_float_array is removed entirely.
    # ────────────────────────────────────────────────────────────────────────────

    def _prebin_nan_cols(self, X_train: Any) -> Any:
        """Pre-bin constant-B numeric columns that must follow the string path.

        Contract for every L value:
        - Finite numeric columns stay numeric and use the native constant-B numeric
          path, including L > 1.
        - Numeric columns with NaN/Inf during training are pre-binned to the
          string/categorical path so the fitted transaction data and later
          predictions share the same missing-value representation.

        A new NaN/Inf at prediction time in a column that was clean during
        training is handled by the native numeric transaction builder, which
        skips item generation for that cell.  This fit-only helper resets
        ``_missing_col_edges_`` at the beginning of each call; callers outside
        the normal fit path should not invoke it incrementally.  For ndarray
        inputs, the returned object preserves ndarray type when a conversion is
        needed, avoiding accidental DataFrame type coercion.
        """
        is_df = isinstance(X_train, pd.DataFrame)
        if is_df:
            X_df = X_train
        else:
            arr = np.asarray(X_train)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            feat_names = list(getattr(self, "feature_names_in_", []) or [])
            cols = (
                feat_names
                if len(feat_names) == arr.shape[1]
                else [f"col{j}" for j in range(arr.shape[1])]
            )
            X_df = pd.DataFrame(arr, columns=cols)
        cat_mask = self.cat_cols_mask_
        col_names = list(X_df.columns)
        n_cols = len(col_names)

        self._missing_col_edges_: dict = {}
        new_cat = cat_mask.copy()
        new_int = getattr(self, "is_int_mask_", np.zeros(n_cols, dtype=bool)).copy()
        modified = False

        for j, name in enumerate(col_names):
            if j >= len(cat_mask) or cat_mask[j]:
                continue  # already categorical — C++ handles np.nan natively

            col = pd.to_numeric(X_df.iloc[:, j], errors="coerce").values
            finite_mask = np.isfinite(col)
            if bool(np.all(finite_mask)):
                # Consistent fast path for every L: finite numeric columns
                # remain numeric and are binned by native constant-B code.
                continue
            finite = col[finite_mask]
            edges = (
                _adap_quantile_edges(finite, self.B) if finite.size > 0 else np.array([0.0, 1.0])
            )
            self._missing_col_edges_[name] = edges
            new_cat[j] = True
            new_int[j] = False
            modified = True

        if not modified:
            return X_train

        self.cat_cols_mask_ = new_cat
        self.is_int_mask_ = new_int

        X_pre = X_df.copy()
        for name, edges in self._missing_col_edges_.items():
            col = pd.to_numeric(X_df[name], errors="coerce").values
            X_pre[name] = _adap_apply_edges(col, edges)
        return X_pre if is_df else X_pre.to_numpy()

    def _frame_from_input(self, X: Any, names: list[str] | None = None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        cols = list(names or [])
        if len(cols) != arr.shape[1]:
            cols = [f"col{j}" for j in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)

    def _select_interaction_relaxed_survivors_from_frame(
        self,
        X_train: Any,
        y_arr: np.ndarray,
        col_names: list[str] | None = None,
        cat_mask: np.ndarray | None = None,
    ) -> tuple[list[dict], list[int]]:
        if not (
            _CORE_AVAILABLE
            and hasattr(_core, "select_interaction_information_features")
            and hasattr(_core, "mine_patterns_relaxed")
        ):
            raise HUGIMLParamError(
                "interaction_relaxed_mining=True requires "
                "_hugiml_core.select_interaction_information_features and "
                "_hugiml_core.mine_patterns_relaxed."
            )
        X_df = self._frame_from_input(
            X_train, col_names or getattr(self, "feature_names_in_", None)
        )
        names = [str(name) for name in list(X_df.columns)]
        cat = (
            cat_mask
            if cat_mask is not None
            else getattr(self, "cat_cols_mask_", np.zeros(len(names), dtype=bool))
        )
        binary_cat_names = set(getattr(self, "binary_categorical_cols_", []) or [])
        score_idx: list[int] = []
        score_cols: list[np.ndarray] = []
        score_names: list[str] = []
        for j, name in enumerate(names):
            is_cat = bool(j < len(cat) and cat[j])
            if is_cat and name not in binary_cat_names:
                continue
            col = pd.to_numeric(X_df.iloc[:, j], errors="coerce").to_numpy(dtype=np.float64)
            if not np.isfinite(col).any():
                continue
            score_idx.append(j)
            score_names.append(name)
            score_cols.append(col)
        if not score_cols:
            return [], []
        X_scored = np.ascontiguousarray(np.column_stack(score_cols), dtype=np.float64)
        selected = _core.select_interaction_information_features(
            X_scored,
            np.asarray(y_arr, dtype=np.int64),
            score_names,
            int(self.interaction_relaxed_feature_size),
            self.ii_partner_size,
        )
        name_to_idx = {name: score_idx[k] for k, name in enumerate(score_names)}
        relaxed_cols = [
            name_to_idx[str(row["name"])] for row in selected if str(row.get("name")) in name_to_idx
        ]
        return [dict(row) for row in selected], relaxed_cols

    @staticmethod
    def _interaction_relaxed_pairs_from_survivors(
        survivor_rows: list[dict],
        numeric_names: list[str],
    ) -> list[tuple[str, str]]:
        numeric = {str(name) for name in numeric_names}
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add_pair(left: str, right: str) -> None:
            if left == right or left not in numeric or right not in numeric:
                return
            key = tuple(sorted((left, right)))
            if key in seen:
                return
            seen.add(key)
            pairs.append((left, right))

        for row in survivor_rows:
            left = str(row.get("name")) if row.get("name") is not None else ""
            right = str(row.get("best_partner")) if row.get("best_partner") is not None else ""
            add_pair(left, right)
        return pairs

    def _select_pair_aware_adaptive_b(
        self,
        feature_name: str,
        feature_values: np.ndarray,
        partner_values: list[tuple[str, np.ndarray]],
        y_codes: np.ndarray,
        n_classes: int,
        candidates: list[int],
        ratio: float,
    ) -> tuple[int, dict[int, float], dict]:
        finite = np.isfinite(feature_values)
        if int(finite.sum()) < 10:
            chosen = candidates[len(candidates) // 2]
            scores = {int(b): 0.0 for b in candidates}
            return int(chosen), scores, {"mode": "insufficient_finite_values", "best_partner": None}

        scores: dict[int, float] = {}
        marginal_scores: dict[int, float] = {}
        best_partner_by_b: dict[int, dict] = {}
        scan_candidates = sorted({int(v) for v in candidates if int(v) >= 2})
        if partner_values:
            scan_candidates = [b for b in scan_candidates if b <= 8] or scan_candidates
        for b in scan_candidates:
            partner_bins = [int(b)]
            marginal, _edges, _codes = _edge_information_gain(feature_values, y_codes, n_classes, b)
            marginal_scores[b] = float(marginal)
            best_pair_score = 0.0
            best_partner_name: str | None = None
            best_partner_bins = None
            for partner_name, partner_col in partner_values:
                if int(np.isfinite(partner_col).sum()) < 10:
                    continue
                for pb in partner_bins:
                    joint, left_score, right_score = _joint_information_gain_from_binned_columns(
                        feature_values,
                        partner_col,
                        y_codes,
                        n_classes,
                        b,
                        pb,
                    )
                    conditional = max(0.0, float(joint) - float(right_score))
                    pair_score = max(
                        conditional, min(float(joint), max(float(left_score), float(right_score)))
                    )
                    if pair_score > best_pair_score:
                        best_pair_score = float(pair_score)
                        best_partner_name = partner_name
                        best_partner_bins = int(pb)
            scores[b] = float(max(marginal, best_pair_score))
            best_partner_by_b[b] = {
                "partner": best_partner_name,
                "partner_bins": best_partner_bins,
                "pair_score": float(best_pair_score),
                "marginal_score": float(marginal),
            }

        if not scores:
            chosen = candidates[len(candidates) // 2]
            return (
                int(chosen),
                {int(b): 0.0 for b in candidates},
                {"mode": "no_scores", "best_partner": None},
            )
        best_score = max(scores.values())
        if best_score <= 0.0:
            chosen = min(scores)
        elif partner_values:
            threshold = best_score * 0.85
            eligible = [b for b in sorted(scores) if scores[b] >= threshold]
            chosen = eligible[0] if eligible else max(scores, key=scores.get)
        else:
            threshold = best_score * max(0.0, 1.0 - float(ratio))
            eligible = [b for b in sorted(scores) if scores[b] >= threshold]
            chosen = eligible[0] if eligible else max(scores, key=scores.get)
        evidence = dict(best_partner_by_b.get(int(chosen), {}))
        evidence.update(
            {
                "mode": "pair_aware" if partner_values else "marginal",
                "feature": feature_name,
                "chosen_b": int(chosen),
                "score": float(scores.get(int(chosen), 0.0)),
                "best_score": float(best_score),
                "marginal_scores": {int(k): float(v) for k, v in marginal_scores.items()},
            }
        )
        return int(chosen), {int(k): float(v) for k, v in scores.items()}, evidence

    def _apply_pair_aware_adaptive_binning(self, X_train: Any, y_arr: np.ndarray) -> Any:
        is_df = isinstance(X_train, pd.DataFrame)
        X_df = self._frame_from_input(X_train, getattr(self, "feature_names_in_", None))
        base_candidates = sorted(
            {int(v) for v in (self.b_candidates or [2, 3, 5, 7, 10, 15]) if int(v) >= 2}
        )
        candidates = sorted(set(base_candidates + [4, 6, 8]))
        ratio = float(self.min_marginal_gain_ratio)
        cat_mask = self.cat_cols_mask_
        col_names = list(X_df.columns)
        y_codes = np.asarray(pd.factorize(np.asarray(y_arr), sort=True)[0], dtype=np.int64)
        n_classes = int(len(np.unique(y_codes)))

        survivors, _ = self._select_interaction_relaxed_survivors_from_frame(
            X_df,
            y_codes,
            [str(name) for name in col_names],
            cat_mask,
        )
        self.interaction_relaxed_mining_survivors_ = [dict(row) for row in survivors]

        num_cols = [
            (j, name) for j, name in enumerate(col_names) if not (j < len(cat_mask) and cat_mask[j])
        ]
        num_names = [str(name) for _, name in num_cols]
        pair_names = self._interaction_relaxed_pairs_from_survivors(survivors, num_names)
        partners_by_feature: dict[str, list[str]] = {str(name): [] for name in num_names}
        for left, right in pair_names:
            partners_by_feature.setdefault(left, []).append(right)
            partners_by_feature.setdefault(right, []).append(left)

        raw_numeric: dict[str, np.ndarray] = {
            str(name): pd.to_numeric(X_df.iloc[:, j], errors="coerce").to_numpy(dtype=np.float64)
            for j, name in num_cols
        }
        self._bin_edges_: dict = {}
        self.per_feature_b_: dict = {}
        self.ig_scores_: dict = {}
        self._interaction_relaxed_adaptive_pairs_ = [
            {"left": left, "right": right} for left, right in pair_names
        ]
        self._interaction_relaxed_adaptive_evidence_ = {}

        # One native call computes chosen bin counts for every numeric
        # column at once: pair-aware scoring for survivor columns with a
        # known partner, ordinary marginal-IG elbow selection for everything
        # else. See native/augmented_pair.cpp::select_pair_aware_adaptive_bins
        # for the full algorithm. A pure-Python column-by-column path
        # (_select_pair_aware_adaptive_b below) is used when the native
        # function is unavailable or raises, matching the fallback pattern
        # used elsewhere in this module for other native/Python pairs.
        use_native_pair_aware = bool(
            _CORE_AVAILABLE and hasattr(_core, "select_pair_aware_adaptive_bins")
        )
        chosen_b_by_name: dict[str, int] = {}
        scores_by_name: dict[str, dict[int, float]] = {}
        evidence_by_name: dict[str, dict] = {}
        if use_native_pair_aware and num_names:
            name_to_idx = {name: k for k, name in enumerate(num_names)}
            pair_idx_list = [
                (name_to_idx[left], name_to_idx[right])
                for left, right in pair_names
                if left in name_to_idx and right in name_to_idx
            ]
            X_numeric_mat = np.ascontiguousarray(
                np.column_stack([raw_numeric[name] for name in num_names])
                if num_names
                else np.zeros((len(y_codes), 0)),
                dtype=np.float64,
            )
            try:
                native_result = _core.select_pair_aware_adaptive_bins(
                    X_numeric_mat,
                    y_codes,
                    num_names,
                    [int(c) for c in candidates],
                    pair_idx_list,
                    float(ratio),
                    0.85,
                    8,
                )
                chosen_b_by_name = {
                    str(k): int(v) for k, v in dict(native_result["chosen_b"]).items()
                }
                scores_by_name = {
                    str(k): {int(kk): float(vv) for kk, vv in dict(v).items()}
                    for k, v in dict(native_result["scores"]).items()
                }
                evidence_by_name = {
                    str(k): dict(v) for k, v in dict(native_result["evidence"]).items()
                }
            except Exception:
                use_native_pair_aware = False

        for _j, name_obj in num_cols:
            name = str(name_obj)
            col = raw_numeric[name]
            if use_native_pair_aware and name in chosen_b_by_name:
                chosen = chosen_b_by_name[name]
                scores = scores_by_name.get(name, {int(b): 0.0 for b in candidates})
                evidence = evidence_by_name.get(
                    name,
                    {
                        "mode": "marginal",
                        "feature": name,
                        "chosen_b": int(chosen),
                        "best_partner": None,
                    },
                )
            else:
                partner_values = [
                    (partner_name, raw_numeric[partner_name])
                    for partner_name in partners_by_feature.get(name, [])
                    if partner_name in raw_numeric
                ]
                if partner_values:
                    chosen, scores, evidence = self._select_pair_aware_adaptive_b(
                        name,
                        col,
                        partner_values,
                        y_codes,
                        n_classes,
                        candidates,
                        ratio,
                    )
                else:
                    finite_mask = np.isfinite(col)
                    if int(finite_mask.sum()) < 10:
                        chosen = candidates[len(candidates) // 2]
                        scores = {int(b): 0.0 for b in candidates}
                    else:
                        chosen, scores = _adap_select_b(
                            col[finite_mask], y_codes[finite_mask], candidates, ratio
                        )
                    evidence = {
                        "mode": "marginal",
                        "feature": name,
                        "chosen_b": int(chosen),
                        "best_partner": None,
                    }
            edges = _adap_quantile_edges(col, int(chosen))
            self._bin_edges_[name_obj] = edges
            self.per_feature_b_[name_obj] = len(edges) - 1
            self.ig_scores_[name_obj] = {int(k): float(v) for k, v in scores.items()}
            self._interaction_relaxed_adaptive_evidence_[name] = evidence

        self._adaptive_code_label_map_: dict[str, str] = {}
        self._adaptive_precoded_features_ = set(self._bin_edges_)
        new_cat = cat_mask.copy()
        new_int = getattr(self, "is_int_mask_", np.zeros(len(col_names), dtype=bool)).copy()
        X_pre = X_df.copy()
        for name_obj, edges in self._bin_edges_.items():
            if name_obj not in X_df.columns:
                continue
            name = str(name_obj)
            col = pd.to_numeric(X_df[name_obj], errors="coerce").values
            n_bins = len(edges) - 1
            has_nan = not np.isfinite(col).all()
            j = col_names.index(name_obj) if name_obj in col_names else -1
            codes = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1).astype(np.float64)
            if has_nan:
                codes[~np.isfinite(col)] = np.nan
            X_pre[name_obj] = codes
            if j >= 0:
                new_cat[j] = False
                new_int[j] = True
            for k in range(n_bins):
                cpp_label = f"{name_obj}=[{float(k):.3f},{float(k + 1):.3f}]"
                orig_label = f"{name_obj}=[{edges[k]:.4g},{edges[k + 1]:.4g})"
                self._adaptive_code_label_map_[cpp_label] = orig_label
        self.cat_cols_mask_ = new_cat
        self.is_int_mask_ = new_int
        return X_pre if is_df else X_pre

    def _handle_test_nan(self, X_test: Any) -> tuple:
        """Apply training-time missing-column bin edges at test time.

        In constant-B non-adaptive models, this converts only the columns recorded
        in ``_missing_col_edges_`` back to the exact string-label representation
        used at fit time.  The rule is identical for L == 1 and L > 1: only
        numeric columns that had NaN/Inf during training are recorded here.
        Numeric columns not recorded in ``_missing_col_edges_`` remain numeric
        and are binned by C++ directly; if they contain new non-finite values at
        prediction time, the native numeric transaction builder skips them.

        Returns ``(X_modified, local_cat_mask)``.  ``self.cat_cols_mask_``
        is never mutated; ``local_cat_mask`` is a per-call copy.
        """
        base_cat = getattr(self, "cat_cols_mask_", None)
        if base_cat is None:
            return X_test, base_cat

        is_df = isinstance(X_test, pd.DataFrame)
        feat_names = getattr(self, "feature_names_in_", None)
        if is_df:
            X_df = X_test
        else:
            arr = np.asarray(X_test)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            cols = (
                list(feat_names)
                if feat_names is not None and len(feat_names) == arr.shape[1]
                else [f"col{j}" for j in range(arr.shape[1])]
            )
            X_df = pd.DataFrame(arr, columns=cols)

        col_names = list(X_df.columns)
        missing_edges = getattr(self, "_missing_col_edges_", {})
        local_cat = base_cat.copy()
        modified = False

        for j, name in enumerate(col_names):
            if j >= len(base_cat):
                continue

            # Column was pre-binned at training time because it contained
            # NaN/Inf during fit: convert raw float values to the same string
            # bin labels so C++ item lookups match.
            if name in missing_edges:
                col = pd.to_numeric(X_df.iloc[:, j], errors="coerce").values
                if not modified:
                    X_df = X_df.copy()
                    modified = True
                X_df[name] = _adap_apply_edges(col, missing_edges[name])
                local_cat[j] = True
                continue

            # Genuinely categorical column or column not in model:
            # C++ already handles np.nan natively — no action needed.

        if not modified:
            return X_test, base_cat
        return (X_df if is_df else X_df.to_numpy()), local_cat

    # ── End missing value handling methods ───────────────────────────────────

    # ── End adaptive binning methods ─────────────────────────────────────────

    def _require_core(self) -> None:
        """Raise a diagnostic ImportError if the native extension is absent."""
        if not _CORE_AVAILABLE:
            import platform
            import sys

            cause = f"\n  Original error: {_CORE_IMPORT_ERROR}" if _CORE_IMPORT_ERROR else ""
            raise ImportError(
                "HUGIMLClassifier requires the compiled C++ extension '_hugiml_core', "
                "which was not importable at package load time."
                f"{cause}\n\n"
                f"  Platform : {platform.system()} {platform.machine()}\n"
                f"  Python   : {sys.version.split()[0]}\n"
                f"  sys.path : {sys.path}\n\n"
                "Likely causes and remedies:\n"
                "  1. No pre-built wheel for your platform (most common).\n"
                "     Check https://pypi.org/project/hugiml-core/#files for available wheels.\n"
                "     Linux x86_64 wheels require glibc >= 2.17. "
                "Run 'ldd --version' to check yours.\n"
                "  2. pip fell back to the sdist and the C++ build failed.\n"
                "     Look for gcc/g++ errors above in the pip install output.\n"
                "  3. The .so was installed but is not on sys.path (rare; "
                "check for a broken venv).\n\n"
                "To build from source (requires gcc/g++ and pybind11):\n"
                "  pip install . --no-build-isolation\n"
                "Fast development build:\n"
                "  HUGIML_FAST_BUILD=1 python setup.py build_ext --inplace\n"
                "Verify after install:\n"
                '  python -c "import hugiml; assert hugiml.check_native(), '
                "'native missing'\""
            )

    def fit(self, X: Any, y: Any) -> HUGIMLClassifierNative:
        """Fit the HUG-IML model on training data.

        Parameters
        ----------
        X : pd.DataFrame or ndarray, shape (n_samples, n_features)
        y : array-like of int, shape (n_samples,)

        Returns
        -------
        self

        Thread safety
        -------------
        fit() acquires an exclusive lock.  Concurrent fit() calls on the same
        instance are serialized.  predict/predict_proba/transform are read-only
        on fitted state and safe for concurrent use after fit() returns.
        """
        self._require_core()
        with self._fit_lock:
            try:
                return self._fit_impl(X, y)
            except MemoryError as exc:
                raise HUGIMLMemoryError(
                    "HUGIML fit failed cleanly because required memory could not be allocated. "
                    "Reduce n/p/topK/B, keep adaptive_binning=True/use_hotpath=True, or increase "
                    "the process memory limit. Original error: " + str(exc)
                ) from exc
            except RuntimeError as exc:
                if "hugiml_timeout" in str(exc):
                    raise HUGIMLTimeoutError(str(exc)) from exc
                raise

    def _fit_impl(self, X_train: Any, y_train: Any) -> HUGIMLClassifierNative:
        # Clear all fitted state so that re-fitting the same instance is
        # idempotent.  Without this, _resolve_col_meta() short-circuits on
        # the cached cat_cols_mask_ from the previous fit, causing the column
        # type masks to carry over and producing non-reproducible results.
        for _attr in (
            "cat_cols_mask_",
            "is_int_mask_",
            "feature_names_in_",
            "binary_categorical_cols_",
            "_bin_edges_",
            "_missing_col_edges_",
            "_adaptive_code_label_map_",
            "per_feature_b_",
            "ig_scores_",
            "patterns_",
            "model_",
            "classes_",
            "x_train_hup_",
            "fit_metadata_",
            "_original_scaler_",
            "_original_numeric_medians_",
            "_original_numeric_medians_array_",
            "_original_feature_names_downstream_",
            "_pattern_orders_",
            "_interaction_pattern_mask_",
            "x_train_downstream_",
            "_augmented_pair_block_",
            "augmented_pair_transforms_",
            "augmented_pair_selected_features_",
            "_native_original_feature_names_downstream_",
            "_native_original_feature_scores_downstream_",
            "_strict_topk_applied_during_construction_",
            "_strict_topk_feature_mask_",
            "_strict_topk_feature_scores_",
            "_strict_topk_selected_feature_names_",
            "_downstream_feature_names_full_",
            "_training_pattern_matrix_shape_",
            "_training_pattern_matrix_nnz_",
            "_training_downstream_matrix_shape_",
            "_training_downstream_matrix_nnz_",
            "_drift_det",
        ):
            self.__dict__.pop(_attr, None)

        t_total = self._timer()
        stage_times: dict[str, float] = {}

        # Reject sparse matrices with an informative message
        from scipy.sparse import issparse as _issparse

        if _issparse(X_train):
            raise ValueError(
                "HUGIMLClassifierNative does not support sparse input.  "
                "Convert to a dense array via X.toarray() first."
            )

        # Reject complex-valued arrays
        if hasattr(X_train, "dtype") and np.iscomplexobj(X_train):
            raise ValueError("Complex data not supported by HUGIMLClassifierNative.")

        self._validate_params()

        # Configure OpenMP before adaptive binning.  Adaptive B-selection is
        # column-parallel in the native path, so applying n_jobs after adaptive
        # preprocessing is too late.
        n_threads = _core.openmp_get_max_threads() if self.n_jobs == -1 else self.n_jobs
        if n_threads > 0:
            _core.openmp_set_num_threads(n_threads)
        actual_threads = _core.openmp_get_max_threads()

        # Preserve raw input if the downstream mode needs original features,
        # or if L>1 will create internal augmented_pair_transforms. This remains
        # an internal operation; no public hyperparameter is added.
        _needs_augmented_pairs = bool(self.L > 1 and bool(self.augmented_pair_transforms))
        X_train_original_for_downstream = (
            self._copy_input_for_downstream(X_train)
            if (self.feature_mode != "patterns_only" or _needs_augmented_pairs)
            else None
        )

        # Fused adaptive+L1 hot path can consume raw X directly and must not
        # materialise the intermediate X_codes matrix/DataFrame.
        _use_fused_adaptive_l1 = (
            self.adaptive_binning
            and self.use_hotpath
            and _CORE_AVAILABLE
            and self.L == 1
            and hasattr(_core, "prepare_and_mine_l1_adaptive")
        )

        # ── Adaptive pre-binning (C++ hot path or Python fallback) ────────
        # ── Adaptive B-selection always uses C++ ──────────────────────────
        # _apply_adaptive_binning_cpp calls _core.select_adaptive_bins
        # (elbow_stop_nb_cpp) whenever the C++ extension is available.
        # use_hotpath does NOT gate this: C++ adaptive selection is always
        # preferred because it produces identical outputs with no conflicts.
        # Python _apply_adaptive_binning is kept as a fallback for
        # environments where the C++ extension is absent.
        if self.adaptive_binning and not _use_fused_adaptive_l1:
            self._resolve_col_meta(X_train)  # prime cat_cols_mask_ first
            _y_for_ig = self._safe_cast_y(y_train)
            _use_pair_aware_adaptive = bool(
                self.L != 1 and bool(getattr(self, "interaction_relaxed_mining", False))
            )
            if _use_pair_aware_adaptive:
                X_train = self._apply_pair_aware_adaptive_binning(X_train, _y_for_ig)
            elif _CORE_AVAILABLE and hasattr(_core, "select_adaptive_bins"):
                X_train = self._apply_adaptive_binning_cpp(X_train, _y_for_ig)
            else:
                X_train = self._apply_adaptive_binning(X_train, _y_for_ig)
            if self.verbose:
                logger.info(
                    "  adaptive binning: %d features pre-binned, B_j in [%d, %d]",
                    len(self._bin_edges_),
                    min(self.per_feature_b_.values(), default=0),
                    max(self.per_feature_b_.values(), default=0),
                )
        # ─────────────────────────────────────────────────────────────────

        # ── Constant-B non-finite handling (non-adaptive path) ───────────────
        # Use one consistent scheme for every L: numeric columns stay numeric
        # unless they contain NaN/Inf during training.  Columns with training
        # non-finite cells are pre-binned to the string/categorical path so
        # fit and predict use the same missing-value representation.  Clean
        # numeric columns, including L>1 columns, use the native numeric path;
        # new test-time NaN/Inf values are skipped by native item generation.
        if not self.adaptive_binning:
            self._resolve_col_meta(X_train)
            X_train = self._prebin_nan_cols(X_train)
        # ─────────────────────────────────────────────────────────────────

        # Line-level allocation tracing (tracemalloc, inside _MemoryTracker)
        # is independent of the audit/production retention distinction below.
        # execution_mode='production' skips it, since it can dominate fit()
        # wall time on categorical-heavy inputs and the line-level breakdown
        # it provides is diagnostic rather than prediction-relevant.
        # execution_mode='audit' (the default) keeps it active, tracking the
        # same traced_peak_mb granularity as every other stage of fit().
        #
        # Internal cache-building fits performed by fast_grid_tune() set
        # _fast_tune_cache_only and are tagged execution_mode='audit' purely
        # so x_train_hup_ / x_train_downstream_ survive for reuse across the
        # feature_mode variants built from the same cached fit -- that need
        # is about which arrays are retained afterward, not about whether
        # allocation tracing runs during the fit itself, so those fits skip
        # tracing too regardless of the execution_mode tag they carry.
        mem = _MemoryTracker(
            enable_tracing=(
                getattr(self, "execution_mode", "audit") != "production"
                and not getattr(self, "_fast_tune_cache_only", False)
            )
        )
        with mem:
            # Stage 1: resolve column metadata
            t = self._timer()
            cat_mask = self._resolve_col_meta(X_train)
            int_mask = getattr(self, "is_int_mask_", None)

            X_num, X_cat_raw = self._to_float_array(X_train, cat_mask)
            y_train = self._safe_cast_y(y_train)
            # Native numeric paths treat non-finite feature cells as missing
            # observations and skip item generation.  Let NaN/Inf through
            # sklearn validation; y is already checked separately by _safe_cast_y.
            try:
                X_num, y_train = check_X_y(X_num, y_train, dtype=None, ensure_all_finite=False)
            except TypeError:
                X_num, y_train = check_X_y(X_num, y_train, dtype=None, force_all_finite=False)

            # Columns with at most one distinct observed value carry no
            # information for either path below to mine, so they're excluded
            # from the per-row work that would otherwise touch every row only
            # to confirm what a single distinct-value count already shows.
            # n_features_in_ / feature_names_in_ still reflect every column
            # exactly as given, so predict()'s schema check is unaffected --
            # only the per-column native processing is skipped.
            self._zero_variance_cols_ = self._identify_zero_variance_columns(X_train)
            self._exclude_zero_variance_columns(
                X_num, X_cat_raw, cat_mask, self._zero_variance_cols_
            )

            self.n_features_in_ = X_num.shape[1]
            self.classes_ = np.unique(y_train)
            n_cls = len(self.classes_)
            stage_times["resolve_meta"] = t.ms

            if n_cls < 2:
                raise HUGIMLValidationError(
                    f"y contains only {n_cls} class(es).  At least 2 are required."
                )
            if X_num.shape[0] < n_cls:
                raise HUGIMLValidationError(
                    f"Fewer samples ({X_num.shape[0]}) than classes ({n_cls})."
                )

            est_mb = _MemoryTracker.estimate_fit_mb(
                X_num.shape[0], X_num.shape[1], X_num.shape[1] * 10, self._effective_topK()
            )
            if est_mb > 4000:
                warnings.warn(
                    f"Estimated peak memory ~{est_mb:.0f} MB.  "
                    "Consider reducing topK or dataset size.",
                    HUGIMLWarning,
                    stacklevel=4,
                )

            if self.verbose:
                logger.info(
                    "HUGIMLClassifierNative.fit — %dx%d, %d classes",
                    X_num.shape[0],
                    X_num.shape[1],
                    n_cls,
                )

            # Stage 2+3+4: prepare / mine / build matrix
            # ── Fused L=1 hot path ─────────────────────────────────────────
            # When use_hotpath=True and L=1: a single C++ call replaces
            # prepare_transactions + mine_patterns + build_train_matrix.
            # No TransList, no hash-map lookups, direct COO from TID index.
            # Falls back to the original three-step path for L>1, adaptive
            # binning with Python fallback, or when explicitly disabled.
            t = self._timer()
            rss_before = _get_peak_rss_kb()
            col_names = getattr(self, "feature_names_in_", None)
            is_cat_np = cat_mask.astype(np.uint8)
            is_int_np = (
                int_mask if int_mask is not None else np.zeros(X_num.shape[1], dtype=bool)
            ).astype(np.uint8)

            # Build is_precoded mask without scanning every adaptive label key.
            is_precoded_np: np.ndarray | None = None
            if self.adaptive_binning:
                precoded_features = getattr(self, "_adaptive_precoded_features_", set())
                p_cols = X_num.shape[1]
                feat_names_list = (
                    col_names if col_names is not None else [f"col{j}" for j in range(p_cols)]
                )
                is_precoded_np = np.fromiter(
                    (name in precoded_features for name in feat_names_list),
                    dtype=np.uint8,
                    count=p_cols,
                )

            _use_fused = (
                self.use_hotpath
                and _CORE_AVAILABLE
                and self.L == 1
                and hasattr(_core, "prepare_and_mine_l1")
            )

            if _use_fused:
                # ── Fused path ────────────────────────────────────────────────
                fit_deadline = (
                    time.perf_counter() + self.max_fit_seconds if self.max_fit_seconds else None
                )
                remaining_s = max(fit_deadline - time.perf_counter(), 0.0) if fit_deadline else 0.0
                K_eff = self._effective_mining_topK()  # rough pre-estimate (no n_items yet)

                if (
                    os.environ.get("HUGIML_DISABLE_NUMERIC_L1_FASTPATH", "0") != "1"
                    and (not self.adaptive_binning)
                    and hasattr(_core, "prepare_and_mine_l1_fixed_numeric")
                    and not bool(np.any(is_cat_np))
                ):
                    _l1_result = _core.prepare_and_mine_l1_fixed_numeric(
                        X_num,
                        y_train,
                        self.B,
                        col_names,
                        is_int_np,
                        K_eff,
                        self.G,
                        remaining_s,
                        compute_original_scores=(self.feature_mode != "patterns_only"),
                    )
                elif self.adaptive_binning and hasattr(_core, "prepare_and_mine_l1_adaptive"):
                    candidates = sorted(set(self.b_candidates or [2, 3, 5, 7, 10, 15]))
                    _l1_result = _core.prepare_and_mine_l1_adaptive(
                        X_num,
                        y_train,
                        col_names,
                        is_cat_np,
                        is_int_np,
                        X_cat_raw if any(v is not None for v in X_cat_raw) else None,
                        candidates,
                        self.min_marginal_gain_ratio,
                        K_eff,
                        self.G,
                        remaining_s,
                        compute_original_scores=(self.feature_mode != "patterns_only"),
                        adaptive_binning_sample_frac=self._adaptive_binning_sample_fraction_for_native(),
                        adaptive_binning_sample_random_state=int(
                            self.adaptive_binning_sample_random_state
                        ),
                    )

                    # Install adaptive metadata for predict()/transform() so test
                    # data is pre-binned to the same integer-code representation
                    # used by the fitted td.
                    feat_names_list = (
                        list(col_names)
                        if col_names is not None
                        else [f"col{j}" for j in range(X_num.shape[1])]
                    )
                    self._bin_edges_ = {}
                    self.per_feature_b_ = {}
                    self.ig_scores_ = {}
                    self._adaptive_code_label_map_ = {}
                    self._adaptive_precoded_features_ = set()
                    new_cat = cat_mask.copy()
                    new_int = (
                        int_mask.copy()
                        if int_mask is not None
                        else np.zeros(X_num.shape[1], dtype=bool)
                    )
                    for ci, col_res in enumerate(getattr(_l1_result, "adaptive_cols", [])):
                        j = int(_l1_result.adaptive_num_col_indices[ci])
                        name = feat_names_list[j]
                        edges = np.array(col_res.edges)
                        self._bin_edges_[name] = edges
                        self._adaptive_precoded_features_.add(name)
                        self.per_feature_b_[name] = len(edges) - 1
                        scores: dict[int, float] = {}
                        for k, b in enumerate(candidates):
                            scores[b] = col_res.ig_scores[k] if k < len(col_res.ig_scores) else 0.0
                        self.ig_scores_[name] = scores
                        new_cat[j] = False
                        new_int[j] = True
                        for k in range(len(edges) - 1):
                            cpp_label = f"{name}=[{float(k):.3f},{float(k + 1):.3f}]"
                            orig_label = f"{name}=[{edges[k]:.4g},{edges[k + 1]:.4g})"
                            self._adaptive_code_label_map_[cpp_label] = orig_label
                    self.cat_cols_mask_ = new_cat
                    self.is_int_mask_ = new_int
                else:
                    _l1_result = _core.prepare_and_mine_l1(
                        X_num,
                        y_train,
                        2 if self.adaptive_binning else self.B,
                        col_names,
                        is_cat_np,
                        is_int_np,
                        X_cat_raw if any(v is not None for v in X_cat_raw) else None,
                        is_precoded_np,
                        K_eff,
                        self.G,
                        remaining_s,
                        compute_original_scores=(self.feature_mode != "patterns_only"),
                    )
                self.td_ = _l1_result.td
                if self.feature_mode != "patterns_only":
                    native_orig_names = [
                        f"orig:{name}"
                        for name in list(getattr(_l1_result, "original_feature_names", []) or [])
                    ]
                    native_orig_scores = np.asarray(
                        list(getattr(_l1_result, "original_feature_scores", []) or []),
                        dtype=np.float64,
                    )
                    if native_orig_names and len(native_orig_names) == len(native_orig_scores):
                        self._native_original_feature_names_downstream_ = native_orig_names
                        self._native_original_feature_scores_downstream_ = native_orig_scores
                cpp_mem_bytes = self.td_.memory_usage_bytes()
                n_items = len(self.td_.item_twu)
                K = self._effective_topK(n_items)
                stage_times["prepare_transactions"] = t.ms

                if self.verbose:
                    logger.info("  items=%d, K=%d [fused L=1 path]", n_items, K)

                t = self._timer()
                raw_patterns_list = list(_l1_result.patterns)
                self.raw_patterns_ = sorted(
                    raw_patterns_list, key=lambda pe: (-pe.utility, tuple(pe.items))
                )
                # L=1 singletons are unique by definition (each pattern is one
                # distinct item), so deduplication by coverage is a no-op.
                # Skip calling _deduplicate_patterns_by_coverage (which would
                # call build_train_matrix on the empty td.transactions) and use
                # the COO returned directly by the fused path.
                self.patterns_ = self.raw_patterns_
                stage_times["mine_patterns"] = t.ms

                if len(self.patterns_) == 0:
                    if self.feature_mode == "patterns_only":
                        raise HUGIMLMiningError(
                            "No HUG patterns found.  Try reducing G, increasing topK, or adjusting B / L."
                        )
                    # In original_plus_patterns mode, zero mined HUG patterns is
                    # not fatal: downstream fitting should fall back to the
                    # original feature block with an empty pattern matrix.
                    n_train = len(y_train)
                    self.x_train_hup_ = csr_matrix((n_train, 0), dtype=np.float32)
                    stage_times["build_matrix"] = 0.0

                if self.verbose:
                    logger.info(
                        "  %d patterns in %.0f ms [fused]",
                        len(self.patterns_),
                        stage_times["mine_patterns"],
                    )

                t = self._timer()
                n_train = len(y_train)
                n_pats = len(self.patterns_)
                if n_pats == 0:
                    # original_plus_* modes are allowed to continue with an
                    # empty pattern block; the downstream matrix will contain
                    # the original feature block and any enabled augmented block.
                    # Do not call native build/get_coo paths with an empty
                    # pattern list because native code rejects that as
                    # "patterns list is empty — nothing to build".
                    self.x_train_hup_ = csr_matrix((n_train, 0), dtype=np.float32)
                else:
                    # Build train matrix from fused native CSR when available.  This
                    # avoids copying COO rows/cols into Python and lets scipy consume the
                    # compact CSR structure directly.  get_coo remains as a compatibility
                    # fallback for older native wheels.
                    if hasattr(_l1_result, "get_csr"):
                        indptr, indices = _l1_result.get_csr(n_train, n_pats)
                        data = np.ones(len(indices), dtype=np.float32)
                        self.x_train_hup_ = csr_matrix(
                            (data, indices, indptr), shape=(n_train, n_pats), dtype=np.float32
                        )
                    else:
                        rows, cols = _l1_result.get_coo()
                        # The fused COO is ordered by pattern index matching raw_patterns_
                        # (both sorted by descending utility).  If patterns_ was reordered
                        # by dedup we would need to remap cols — but dedup is skipped here
                        # so the order is identical.
                        data = np.ones(len(rows), dtype=np.float32)
                        self.x_train_hup_ = csr_matrix(
                            (data, (rows, cols)), shape=(n_train, n_pats), dtype=np.float32
                        )
                stage_times["build_matrix"] = t.ms

            else:
                # ── Original three-step path (L>1, or hotpath disabled) ───────
                self.td_ = _core.prepare_transactions(
                    X_num,
                    y_train,
                    2 if self.adaptive_binning else self.B,
                    col_names,
                    is_cat_np,
                    is_int_np,
                    X_cat_raw if any(v is not None for v in X_cat_raw) else None,
                    is_precoded_np,
                )
                stage_times["prepare_transactions"] = t.ms
                cpp_mem_bytes = self.td_.memory_usage_bytes()

                n_items = len(self.td_.item_twu)
                K = self._effective_topK(n_items)
                K_mine = self._effective_mining_topK(n_items)

                if self.verbose:
                    logger.info(
                        "  items=%d, K=%d, K_mine=%d, td_mem=%.1fMB",
                        n_items,
                        K,
                        K_mine,
                        cpp_mem_bytes / 1e6,
                    )

                t = self._timer()
                fit_deadline = (
                    time.perf_counter() + self.max_fit_seconds if self.max_fit_seconds else None
                )
                relaxed_cols: list[int] | None = None
                if bool(getattr(self, "interaction_relaxed_mining", False)):
                    survivors = list(
                        getattr(self, "interaction_relaxed_mining_survivors_", []) or []
                    )
                    name_to_idx = {str(name): j for j, name in enumerate(list(col_names))}
                    if survivors:
                        relaxed_cols = [
                            name_to_idx[str(row["name"])]
                            for row in survivors
                            if str(row.get("name")) in name_to_idx
                        ]
                    else:
                        survivors, relaxed_cols = (
                            self._select_interaction_relaxed_survivors_from_frame(
                                X_train,
                                y_train,
                                [str(name) for name in list(col_names)],
                                is_cat_np.astype(bool, copy=False),
                            )
                        )
                        self.interaction_relaxed_mining_survivors_ = [
                            dict(row) for row in survivors
                        ]
                raw_patterns = self._mine_with_fallback(
                    y_train, n_cls, K_mine, fit_deadline, relaxed_cols=relaxed_cols
                )
                self.raw_patterns_ = sorted(
                    raw_patterns, key=lambda pe: (-pe.utility, tuple(pe.items))
                )
                selected_patterns = self._select_patterns_for_budget(self.raw_patterns_, n_items)
                self.patterns_, _cached_coo = self._deduplicate_patterns_by_coverage(
                    selected_patterns, len(y_train)
                )
                stage_times["mine_patterns"] = t.ms

                if len(self.patterns_) == 0:
                    if self.feature_mode == "patterns_only":
                        raise HUGIMLMiningError(
                            "No HUG patterns found.  Try reducing G, increasing topK, or adjusting B / L."
                        )
                    # In original_plus_patterns mode, zero mined HUG patterns is
                    # not fatal: downstream fitting should fall back to the
                    # original feature block with an empty pattern matrix.
                    n_train = len(y_train)
                    self.x_train_hup_ = csr_matrix((n_train, 0), dtype=np.float32)
                    stage_times["build_matrix"] = 0.0

                degraded = hasattr(self, "_degraded_reason")
                if degraded and self.verbose:
                    logger.warning("  DEGRADED: %s", self._degraded_reason)

                if self.verbose:
                    logger.info(
                        "  %d patterns in %.0f ms",
                        len(self.patterns_),
                        stage_times["mine_patterns"],
                    )

                t = self._timer()
                n_train = len(y_train)
                n_pats = len(self.patterns_)
                if n_pats == 0:
                    # original_plus_* modes are allowed to continue with an
                    # empty pattern block.  Avoid calling native
                    # build_train_matrix with an empty pattern list.
                    self.x_train_hup_ = csr_matrix((n_train, 0), dtype=np.float32)
                else:
                    if _cached_coo is not None:
                        rows, cols = _cached_coo
                        data = np.ones(len(rows), dtype=np.float32)
                        self.x_train_hup_ = csr_matrix(
                            (data, (rows, cols)), shape=(n_train, n_pats), dtype=np.float32
                        )
                    elif hasattr(_core, "build_train_matrix_csr"):
                        indptr, indices = _core.build_train_matrix_csr(self.td_, self.patterns_)
                        data = np.ones(len(indices), dtype=np.float32)
                        self.x_train_hup_ = csr_matrix(
                            (data, indices, indptr), shape=(n_train, n_pats), dtype=np.float32
                        )
                    else:
                        rows, cols = _core.build_train_matrix(self.td_, self.patterns_)
                        data = np.ones(len(rows), dtype=np.float32)
                        self.x_train_hup_ = csr_matrix(
                            (data, (rows, cols)), shape=(n_train, n_pats), dtype=np.float32
                        )
                stage_times["build_matrix"] = t.ms

            # Optional internal cache-only path used by fast_grid_tune().
            # At this point adaptive metadata, transaction data, mined patterns,
            # and the training HUG matrix are available.  Skipping downstream
            # fitting, rich feature metadata, and drift-baseline construction is
            # correctness-preserving for tuning because each evaluated candidate
            # rebuilds its own downstream matrix/model from these cached mining
            # artefacts.
            if bool(getattr(self, "_fast_tune_cache_only", False)):
                self.td_ = _TransactionDataWrapper(self.td_, self)
                self._native_available_ = True
                self._fast_tune_stage_times_ = dict(stage_times)
                return self

            # Stage 5: fit downstream classifier
            t = self._timer()
            self._setup_feature_mode_metadata()
            self._setup_augmented_pair_transforms(
                X_train_original_for_downstream, y_train, fit=True
            )
            self._current_y_for_downstream_topk_ = y_train
            try:
                self.x_train_downstream_ = self._make_downstream_features(
                    X_train_original_for_downstream, self.x_train_hup_, fit=True
                )
            finally:
                if hasattr(self, "_current_y_for_downstream_topk_"):
                    delattr(self, "_current_y_for_downstream_topk_")
            self.x_train_downstream_ = self._apply_strict_topk_budget_fit(
                self.x_train_downstream_, y_train
            )
            self._cache_downstream_feature_metadata()
            self.model_ = Pipeline([("clf", self._make_estimator(n_cls))])
            self.model_.fit(self.x_train_downstream_, y_train)
            stage_times["fit_downstream"] = t.ms

            # Stage 6: wrap C++ td_ for Python compatibility
            t = self._timer()
            self.td_ = _TransactionDataWrapper(self.td_, self)
            self._native_available_ = True
            stage_times["compat"] = t.ms

            t = self._timer()
            if self._is_production_mode():
                self.__dict__.pop("_drift_det", None)
                stage_times["drift_baseline"] = 0.0
            else:
                self._drift_det = DriftDetector()
                self._drift_det.fit_baseline(
                    X_num,
                    cat_mask,
                    getattr(self, "feature_names_in_", None)
                    or [f"col{j}" for j in range(X_num.shape[1])],
                    y=y_train,
                )
                stage_times["drift_baseline"] = t.ms

        rss_delta_mb = (_get_peak_rss_kb() - rss_before) / 1024
        n_compound = sum(1 for pe in self.patterns_ if len(pe.items) > 1)
        n_pats_final = len(self.patterns_)
        n_train_final = self.x_train_hup_.shape[0]
        nnz = self.x_train_hup_.nnz
        density = (
            nnz / (n_train_final * n_pats_final) if (n_train_final * n_pats_final) > 0 else 0.0
        )
        downstream_names_for_metadata = list(
            getattr(self, "_downstream_feature_names_", []) or self._get_downstream_feature_names()
        )
        downstream_feature_counts = {
            "original": sum(
                1 for name in downstream_names_for_metadata if str(name).startswith("orig:")
            ),
            "pattern": sum(
                1 for name in downstream_names_for_metadata if str(name).startswith("pattern:")
            ),
            "augmented_pair": sum(
                1
                for name in downstream_names_for_metadata
                if str(name).startswith("augmented_pair:")
            ),
        }
        downstream_feature_counts["total"] = len(downstream_names_for_metadata)

        self.fit_metadata_ = FitMetadata(
            n_samples=n_train_final,
            n_features=X_num.shape[1],
            n_classes=n_cls,
            n_items=len(getattr(self.td_, "item_twu", [])),
            n_patterns=n_pats_final,
            n_compound=n_compound,
            topK_used=self._effective_topK(len(getattr(self.td_, "item_twu", [])) or None),
            n_augmented_pairs=downstream_feature_counts.get("augmented_pair", 0),
            n_downstream_features=downstream_feature_counts.get("total", 0),
            downstream_feature_counts=downstream_feature_counts,
            stage_times_ms=stage_times,
            total_fit_ms=t_total.ms,
            matrix_density=density,
            config=dict(
                B=self.B,
                L=self.L,
                G=self.G,
                topK=self.topK,
                adaptive_binning=self.adaptive_binning,
                feature_mode=self.feature_mode,
                execution_mode=self.execution_mode,
            ),
            memory_peak_mb=round(mem.traced_peak_mb, 1),
            memory_rss_mb=round(rss_delta_mb, 1),
            memory_cpp_mb=round(cpp_mem_bytes / 1e6, 2),
            openmp_threads=actual_threads,
            degraded=hasattr(self, "_degraded_reason"),
        )

        self._apply_execution_mode_retention()

        if self.verbose:
            logger.info("  fit complete: %s", self.fit_metadata_.summary())

        return self

    def _mine_with_fallback(
        self,
        y_train: np.ndarray,
        n_cls: int,
        K: int,
        deadline: float | None,
        relaxed_cols: list[int] | None = None,
    ) -> list:
        """Mine patterns with graceful degradation on OOM or timeout.

        The ``deadline`` is forwarded into the C++ mining engine as a
        wall-clock ``timeout_s`` budget so the native layer can abort
        mid-run rather than only being checked between attempts.

        ``relaxed_cols``, when not None/empty, routes every attempt through
        ``mine_patterns_relaxed`` instead of ``mine_patterns`` (interaction_
        relaxed_mining). The minimal final fallback attempt drops G to 0.0
        for ordinary mining already; relaxation does not change that
        behavior, and the result remains bounded by the requested K.
        """
        use_relaxed = bool(relaxed_cols)
        mine_fn = _core.mine_patterns_relaxed if use_relaxed else _core.mine_patterns

        def _call(K_arg, L_arg, G_arg, timeout_arg):
            if use_relaxed:
                return mine_fn(
                    self.td_, y_train, n_cls, K_arg, L_arg, G_arg, relaxed_cols, timeout_arg
                )
            return mine_fn(self.td_, y_train, n_cls, K_arg, L_arg, G_arg, timeout_arg)

        attempts = [
            (K, self.L, self.G, "full"),
            (max(K // 2, 10), self.L, self.G, "K//2"),
            (max(K // 4, 10), 1, self.G, "K//4,L=1"),
            (50, 1, 0.0, "minimal"),
        ]
        for attempt_K, attempt_L, attempt_G, label in attempts:
            if deadline and time.perf_counter() > deadline:
                # Time budget exhausted — skip to minimal attempt immediately.
                minimal_K, minimal_L, minimal_G = 50, 1, 0.0
                self._degraded_reason = (
                    f"Time budget exceeded at '{label}'; "
                    f"falling back to minimal (K={minimal_K}, L={minimal_L})."
                )
                logger.warning("  fit timeout: %s", self._degraded_reason)
                # Give the minimal attempt a constant 5-second window; it is
                # cheap and must not run indefinitely on degenerate data.
                try:
                    return list(_call(minimal_K, minimal_L, minimal_G, 5.0))
                except Exception as exc:
                    raise HUGIMLTimeoutError(
                        f"fit() exceeded max_fit_seconds and the minimal fallback "
                        f"also failed: {exc}"
                    ) from exc
            # Compute remaining budget and pass it to the C++ engine so it
            # can abort mid-run rather than running past the wall-clock limit.
            remaining_s = max(deadline - time.perf_counter(), 0.0) if deadline else 0.0
            try:
                patterns: list = list(_call(attempt_K, attempt_L, attempt_G, remaining_s))
                if label != "full" and len(patterns) > 0:
                    self._degraded_reason = (
                        f"Recovered with {label}: K={attempt_K}, L={attempt_L}, G={attempt_G}"
                    )
                return patterns
            except MemoryError:
                logger.warning("MemoryError during mining (%s), retrying…", label)
                continue
            except Exception as e:
                if "bad_alloc" in str(e).lower() or "memory" in str(e).lower():
                    logger.warning("C++ memory error during mining (%s), retrying…", label)
                    continue
                raise
        return []

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_proba(self, X_test: Any) -> np.ndarray:
        """Predict class probabilities for X_test.

        When ``max_predict_ms`` is set large batches are processed in chunks.
        Rows exceeding the time budget receive uniform probabilities and a
        warning is emitted.

        Parameters
        ----------
        X_test : array-like or DataFrame

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        check_is_fitted(self)
        # Keep the same representation used to fit the downstream original
        # feature block: raw user input before adaptive/constant-B pre-binning.
        # _build_test_hup applies _handle_test_nan() internally for the HUG
        # pattern matrix only; original_plus_* downstream columns are fitted
        # from raw X_train_original_for_downstream and therefore must transform
        # the raw test input as well.
        X_test_original_for_downstream = X_test
        # ── Adaptive pre-binning ─────────────────────────────────────────
        if getattr(self, "adaptive_binning", False) and getattr(self, "_bin_edges_", None):
            X_test = self._prebin_for_predict(X_test)
        # ─────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()

        budget_ms = self.max_predict_ms
        if budget_ms is None or not isinstance(X_test, (pd.DataFrame, np.ndarray)):
            Z_test = self._build_test_hup(X_test)
            X_downstream = self._make_downstream_features(
                X_test_original_for_downstream, Z_test, fit=False
            )
            X_downstream = self._apply_strict_topk_budget_transform(X_downstream)
            proba = np.asarray(self.model_.predict_proba(X_downstream))
            _mon = getattr(self, "monitor", None)
            if _mon is not None:
                _mon.record(proba, (time.perf_counter() - t0) * 1000)
            return proba

        n = len(X_test)
        n_cls = len(self.classes_)
        chunk_size = max(100, n // 10)
        result = np.full((n, n_cls), 1.0 / n_cls, dtype=np.float64)
        completed = 0
        is_df = isinstance(X_test, pd.DataFrame)

        for start in range(0, n, chunk_size):
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms > budget_ms:
                warnings.warn(
                    f"Prediction SLA exceeded ({elapsed_ms:.0f}ms > {budget_ms}ms) "
                    f"after {completed}/{n} rows.  Remaining rows filled with uniform.",
                    HUGIMLWarning,
                    stacklevel=2,
                )
                break
            end = min(start + chunk_size, n)
            chunk = X_test.iloc[start:end] if is_df else X_test[start:end]  # type: ignore[union-attr]
            orig_chunk = (
                X_test_original_for_downstream.iloc[start:end]
                if isinstance(X_test_original_for_downstream, pd.DataFrame)
                else X_test_original_for_downstream[start:end]
            )
            Z_chunk = self._build_test_hup(chunk)
            X_downstream_chunk = self._make_downstream_features(orig_chunk, Z_chunk, fit=False)
            X_downstream_chunk = self._apply_strict_topk_budget_transform(X_downstream_chunk)
            result[start:end] = self.model_.predict_proba(X_downstream_chunk)
            completed = end

        _mon = getattr(self, "monitor", None)
        if _mon is not None:
            _mon.record(result[:completed], (time.perf_counter() - t0) * 1000)
        return result

    def predict(self, X_test: Any) -> np.ndarray:
        """Predict class labels for X_test.

        Parameters
        ----------
        X_test : array-like or DataFrame

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        check_is_fitted(self)
        # Keep the same representation used to fit the downstream original
        # feature block: raw user input before adaptive/constant-B pre-binning.
        # _build_test_hup applies _handle_test_nan() internally for the HUG
        # pattern matrix only; original_plus_* downstream columns are fitted
        # from raw X_train_original_for_downstream and therefore must transform
        # the raw test input as well.
        X_test_original_for_downstream = X_test
        # ── Adaptive pre-binning ─────────────────────────────────────────
        if getattr(self, "adaptive_binning", False) and getattr(self, "_bin_edges_", None):
            X_test = self._prebin_for_predict(X_test)
        # ─────────────────────────────────────────────────────────────────
        Z_test = self._build_test_hup(X_test)
        X_downstream = self._make_downstream_features(
            X_test_original_for_downstream, Z_test, fit=False
        )
        X_downstream = self._apply_strict_topk_budget_transform(X_downstream)
        return np.asarray(self.model_.predict(X_downstream))

    # ── Downstream feature modes ─────────────────────────────────────────────

    def _copy_input_for_downstream(self, X: Any) -> Any:
        """Preserve raw input before adaptive/pre-binning for hybrid modes.

        Downstream original-feature preparation is read-only.  For ndarray inputs
        the mining/pre-binning stages either consume X without mutation or bind a
        new pre-binned object to the local X_train variable, so a full eager copy
        here only adds O(n*p) time and memory.  DataFrames still get a shallow
        schema-stable copy because later preprocessing may add/reorder columns.
        """
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return np.asarray(X)

    def _pattern_order_from_label(self, label: str) -> int:
        """Infer pattern order from a human-readable HUG pattern label.

        .. deprecated::
            This method is retained for backward compatibility only.
            ``_setup_feature_mode_metadata`` derives pattern order directly
            from ``PatternEntry.items`` (the C++ item-ID list), which is the
            authoritative source of pattern length and is not affected by
            comma characters inside numeric interval notation such as
            ``age=[29.2, 38.4)``.

        The fallback parser intentionally counts feature assignments, not
        comma-separated chunks.  Numeric intervals contain commas, so a label
        like ``age=[29.2, 38.4)`` must remain order-1.  A conjunction such as
        ``age=[29,50), income=[50k,80k)`` is order-2 because it contains two
        top-level ``feature=...`` assignments.
        """
        import re

        s = str(label or "").strip()
        if not s:
            return 1

        # Fast path for native / ndarray labels such as
        # ``col0=[29.2, 38.4), col1=A``.  Count distinct column tokens rather
        # than commas so interval bounds do not inflate the order.
        col_matches = re.findall(r"\b(col\d+)\s*=", s)
        if col_matches:
            return max(1, len(set(col_matches)))

        # Human-readable labels are emitted as ``name=value`` assignments,
        # with conjunctions separated by either commas or explicit boolean
        # markers (``AND``, ``and`` or ``&``).  Count assignment starts that
        # occur at the beginning of the string or after one of those top-level
        # separators.  The feature-name pattern must start with a
        # letter/underscore, so commas inside numeric intervals, e.g.
        # ``[29.2, 38.4)``, are not mistaken for new assignments.
        assignment_matches = re.findall(
            r"(?:^|,|\s+(?:AND|and|&)\s+)\s*([A-Za-z_][A-Za-z0-9_ .:/\-]*)\s*=",
            s,
        )
        if assignment_matches:
            return max(1, len({name.strip() for name in assignment_matches if name.strip()}))

        # Last-resort fallback for unknown legacy formats: a label with an
        # explicit boolean conjunction marker is treated as an interaction;
        # otherwise keep the conservative singleton default.  Do not split on
        # commas here because interval labels contain commas.
        if re.search(r"\s+(?:AND|and|&)\s+", s):
            return 2
        return 1

    def _setup_feature_mode_metadata(self) -> None:
        """Cache pattern-order masks used by hybrid feature modes.

        Pattern order (number of features in a pattern) is read directly from
        ``PatternEntry.items`` — the C++ item-ID list — rather than inferred
        from the human-readable label string.  Label-string parsing
        (``_pattern_order_from_label``) mis-counts numeric singletons such as
        ``age=[29.2, 38.4)`` as order-2 because of the comma inside the
        interval notation, causing ``original_plus_interactions`` to
        incorrectly include numeric singletons in the downstream feature
        matrix.  Using ``len(pe.items)`` gives the correct structural count:
        1 for singletons, 2 for pair conjunctions, regardless of feature type
        or label format.
        """
        patterns = getattr(self, "patterns_", None)
        if hasattr(self, "x_train_hup_"):
            n_hup_cols = int(self.x_train_hup_.shape[1])
        elif getattr(self, "_training_pattern_matrix_shape_", None) is not None:
            n_hup_cols = int(self._training_pattern_matrix_shape_[1])
        elif patterns is not None:
            n_hup_cols = int(len(patterns))
        else:
            n_hup_cols = 0
        if patterns is not None and len(patterns) == n_hup_cols:
            # Primary path: read order from C++ PatternEntry.items directly.
            orders = np.asarray([len(pe.items) for pe in patterns], dtype=int)
        else:
            # Fallback: patterns_ unavailable or length mismatch — should not
            # occur after a completed fit, but guard defensively.
            features = self.get_hug_features()
            orders = np.asarray([self._pattern_order_from_label(f) for f in features], dtype=int)
        if len(orders) != n_hup_cols:
            orders = np.ones(n_hup_cols, dtype=int)
        self._pattern_orders_ = orders
        self._interaction_pattern_mask_ = orders > 1

    def _prepare_selected_original_features_for_downstream_transform(
        self, X: Any, selected_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Materialize only persisted selected original downstream columns at predict time.

        Fit still prepares the full original block once so scoring/serialization stay
        unchanged.  Prediction should not rebuild all original columns and then apply
        the fitted TopK mask: in original_plus_* modes the retained original columns
        are already known from ``_original_selected_feature_names_downstream_``.
        This helper constructs just those columns, preserving the exact fitted
        StandardScaler/median-imputation/dummy-column contract.
        """
        selected_names = list(selected_names or [])
        if not selected_names:
            n_rows = (
                len(X)
                if not isinstance(X, np.ndarray)
                else (1 if np.asarray(X).ndim == 1 else np.asarray(X).shape[0])
            )
            return self._empty_dense_block(n_rows), []

        selected_raw = [
            str(name)[5:] if str(name).startswith("orig:") else str(name) for name in selected_names
        ]
        num_cols = list(getattr(self, "_original_numeric_cols_", []))
        cat_cols = list(getattr(self, "_original_cat_cols_", []))
        dummy_cols = list(getattr(self, "_original_dummy_columns_", []))
        num_pos = {str(c): i for i, c in enumerate(num_cols)}
        dummy_set = {str(c) for c in dummy_cols}
        selected_numeric = [name for name in selected_raw if name in num_pos]
        selected_dummy = [name for name in selected_raw if name in dummy_set]

        train_names = list(getattr(self, "feature_names_in_", []) or [])
        train_pos = {str(c): i for i, c in enumerate(train_names)}
        is_df = isinstance(X, pd.DataFrame)
        arr = None if is_df else np.asarray(X)
        if arr is not None and arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n_rows = len(X) if is_df else int(arr.shape[0])

        blocks: list[np.ndarray] = []
        block_names: list[str] = []

        if selected_numeric:
            idx_in_num = np.asarray([num_pos[name] for name in selected_numeric], dtype=np.int64)
            if is_df:
                X_num_sel = pd.DataFrame(index=X.index)
                for name in selected_numeric:
                    X_num_sel[name] = X[name] if name in X.columns else np.nan
                raw_num = (
                    X_num_sel.apply(pd.to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .to_numpy(dtype=np.float64, copy=True)
                )
            else:
                assert arr is not None
                raw_num = np.empty((n_rows, len(selected_numeric)), dtype=np.float64)
                for out_j, name in enumerate(selected_numeric):
                    src_j = train_pos.get(name, out_j)
                    if 0 <= src_j < arr.shape[1]:
                        raw_num[:, out_j] = np.asarray(arr[:, src_j], dtype=np.float64)
                    else:
                        raw_num[:, out_j] = np.nan
                raw_num[~np.isfinite(raw_num)] = np.nan

            med_arr = getattr(self, "_original_numeric_medians_array_", None)
            if med_arr is None or len(med_arr) != len(num_cols):
                med = getattr(self, "_original_numeric_medians_", pd.Series(dtype=float))
                med_arr = med.reindex(num_cols).fillna(0.0).to_numpy(dtype=np.float64, copy=True)
            med_sel = np.asarray(med_arr, dtype=np.float64)[idx_in_num]
            bad = ~np.isfinite(raw_num)
            if bad.any():
                raw_num[bad] = np.take(med_sel, np.nonzero(bad)[1])

            scaler = self._original_scaler_
            mean = np.asarray(getattr(scaler, "mean_", np.zeros(len(num_cols))), dtype=np.float64)[
                idx_in_num
            ]
            scale = np.asarray(getattr(scaler, "scale_", np.ones(len(num_cols))), dtype=np.float64)[
                idx_in_num
            ]
            scale = np.where(scale == 0.0, 1.0, scale)
            blocks.append(((raw_num - mean) / scale).astype(np.float32, copy=False))
            block_names.extend([f"orig:{name}" for name in selected_numeric])

        if selected_dummy:
            if is_df:
                X_cat = X.reindex(columns=cat_cols)
            else:
                assert arr is not None
                data = {}
                for name in cat_cols:
                    src_j = train_pos.get(str(name), None)
                    if src_j is not None and 0 <= src_j < arr.shape[1]:
                        data[name] = arr[:, src_j]
                    else:
                        data[name] = np.full(n_rows, np.nan, dtype=object)
                X_cat = pd.DataFrame(data)
            X_cat_str = X_cat.astype("string") if len(cat_cols) else X_cat
            if len(cat_cols):
                # Collapse category values outside selected_dummy before one-hot
                # encoding, so get_dummies expands only the downstream levels
                # retained during fitting rather than every distinct value in
                # the prediction batch.
                #
                # Collapse to a sentinel value, not to missing/NaN. A valid but
                # unselected category is semantically different from a missing
                # cell, especially when a column's "_<NA>" dummy survived feature
                # selection. The sentinel-generated dummy is removed by the
                # reindex below, leaving all selected dummy columns at zero for
                # that row.
                #
                # This is vectorized per column: construct candidate dummy names
                # once and test membership against the fitted selected set.
                dummy_set_local = set(selected_dummy)
                _COLLAPSE_SENTINEL = "\ue000__hugiml_other__"
                for col in cat_cols:
                    candidate_names = col + "_" + X_cat_str[col]
                    not_keep = X_cat_str[col].notna() & ~candidate_names.isin(dummy_set_local)
                    if not_keep.any():
                        X_cat_str.loc[not_keep, col] = _COLLAPSE_SENTINEL
            X_cat_dum = (
                pd.get_dummies(X_cat_str, dummy_na=True)
                if len(cat_cols)
                else pd.DataFrame(index=range(n_rows))
            )
            X_cat_dum = X_cat_dum.reindex(columns=selected_dummy, fill_value=0)
            blocks.append(X_cat_dum.to_numpy(dtype=np.float32, copy=False))
            block_names.extend([f"orig:{name}" for name in selected_dummy])

        # Preserve fitted selected_names order even when numeric and dummy columns
        # are interleaved.  The two blocks above are built by type for speed; this
        # final gather restores the exact downstream coefficient alignment.
        if not blocks:
            return self._empty_dense_block(n_rows), []
        by_name = {}
        dense_concat = (
            np.hstack(blocks).astype(np.float32, copy=False) if len(blocks) > 1 else blocks[0]
        )
        for j, name in enumerate(block_names):
            by_name[name] = dense_concat[:, j]
        missing_selected = [name for name in selected_names if name not in by_name]
        if missing_selected:
            raise HUGIMLSchemaError(
                "Selected original downstream feature(s) are unavailable during transform: "
                f"{missing_selected[:10]!r}. This usually indicates schema drift or a "
                "model/metadata mismatch. Refit the model or provide input columns "
                "matching the training schema."
            )
        out = np.empty((n_rows, len(selected_names)), dtype=np.float32)
        for j, name in enumerate(selected_names):
            out[:, j] = by_name[name]
        return out, list(selected_names)

    def _prepare_original_features_for_downstream(self, X: Any, fit: bool = False):
        """Prepare original input features for hybrid downstream estimators.

        This intentionally does not affect transform(), get_hug_features(), or
        any pattern diagnostics.  It is used only by predict/fit when
        feature_mode includes original features.
        """
        # Fast all-numeric ndarray fit/transform path.  Avoid constructing a
        # DataFrame and running pandas apply/to_numeric over every column for the
        # common large-n benchmark path.  This preserves the exact fitted
        # StandardScaler/median-imputation/original feature-name contract.
        if not isinstance(X, pd.DataFrame):
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            names = getattr(self, "feature_names_in_", None)
            if names is None or len(names) != arr.shape[1]:
                names = [f"col{j}" for j in range(arr.shape[1])]
            cat_mask = getattr(self, "cat_cols_mask_", None)
            if cat_mask is not None and not bool(np.any(cat_mask)):
                raw_dtype = np.float32 if getattr(arr, "dtype", None) == np.float32 else np.float64
                raw = np.array(arr, dtype=raw_dtype, copy=True, order="C")
                raw[~np.isfinite(raw)] = np.nan
                if fit:
                    self._original_numeric_cols_ = list(names)
                    self._original_cat_cols_ = []
                    med_arr = (
                        np.nanmedian(raw, axis=0) if raw.shape[1] else np.empty(0, dtype=raw.dtype)
                    )
                    med_arr = np.where(np.isfinite(med_arr), med_arr, 0.0).astype(
                        raw.dtype, copy=False
                    )
                    self._original_numeric_medians_array_ = med_arr.astype(np.float64, copy=True)
                    self._original_numeric_medians_ = pd.Series(med_arr, index=list(names))
                    bad = ~np.isfinite(raw)
                    if bad.any():
                        raw[bad] = np.take(med_arr, np.nonzero(bad)[1])
                    self._original_scaler_ = StandardScaler()
                    X_num_arr = (
                        self._original_scaler_.fit_transform(raw)
                        if raw.shape[1]
                        else np.empty((raw.shape[0], 0))
                    )
                    self._original_dummy_columns_ = []
                    self._original_feature_names_downstream_ = list(names)
                    return X_num_arr.astype(np.float32, copy=False)
                num_cols = list(getattr(self, "_original_numeric_cols_", []))
                if (
                    num_cols
                    and list(names) == num_cols
                    and not getattr(self, "_original_cat_cols_", [])
                ):
                    med_arr = getattr(self, "_original_numeric_medians_array_", None)
                    if med_arr is None or len(med_arr) != raw.shape[1]:
                        med = getattr(self, "_original_numeric_medians_", pd.Series(dtype=float))
                        med_arr = (
                            med.reindex(num_cols).fillna(0.0).to_numpy(dtype=raw.dtype, copy=True)
                        )
                    else:
                        med_arr = np.asarray(med_arr, dtype=raw.dtype)
                    bad = ~np.isfinite(raw)
                    if bad.any():
                        raw[bad] = np.take(med_arr, np.nonzero(bad)[1])
                    X_num_arr = self._original_scaler_.transform(raw)
                    return X_num_arr.astype(np.float32, copy=False)

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            names = getattr(self, "feature_names_in_", None)
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if names is None or len(names) != arr.shape[1]:
                names = [f"col{j}" for j in range(arr.shape[1])]
            X_df = pd.DataFrame(arr, columns=list(names))

        # Stabilize column order against training schema when available.
        train_names = getattr(self, "feature_names_in_", None)
        if train_names is not None:
            for col in train_names:
                if col not in X_df.columns:
                    X_df[col] = np.nan
            X_df = X_df[list(train_names)]

        # Hot predict path: fitted all-numeric original_plus_* models can avoid
        # pandas apply/fillna/reindex/get_dummies.  This preserves the fitted
        # StandardScaler and median-imputation contract exactly.
        if not fit:
            num_cols = list(getattr(self, "_original_numeric_cols_", []))
            cat_cols = list(getattr(self, "_original_cat_cols_", []))
            dummy_cols = list(getattr(self, "_original_dummy_columns_", []))
            if num_cols and not cat_cols and not dummy_cols and list(X_df.columns) == num_cols:
                try:
                    X_num_arr_raw = X_df.to_numpy(dtype=np.float64, copy=True)
                    med_arr = getattr(self, "_original_numeric_medians_array_", None)
                    if med_arr is None or len(med_arr) != X_num_arr_raw.shape[1]:
                        med = getattr(self, "_original_numeric_medians_", pd.Series(dtype=float))
                        med_arr = (
                            med.reindex(num_cols).fillna(0.0).to_numpy(dtype=np.float64, copy=True)
                        )
                    bad = ~np.isfinite(X_num_arr_raw)
                    if bad.any():
                        X_num_arr_raw[bad] = np.take(med_arr, np.nonzero(bad)[1])
                    X_num_arr = self._original_scaler_.transform(X_num_arr_raw)
                    return X_num_arr.astype(np.float32, copy=False)
                except Exception:
                    pass

        # Numeric columns are scaled; categorical columns, including
        # numeric-looking binary features inferred during fit, are one-hot encoded.
        numeric = X_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        cat_mask = getattr(self, "cat_cols_mask_", None)
        cat_by_mask = set()
        if cat_mask is not None:
            cat_by_mask = {
                str(c) for c, is_cat in zip(X_df.columns.astype(str), cat_mask) if bool(is_cat)
            }
        numeric_cols = [
            c for c in X_df.columns if str(c) not in cat_by_mask and not numeric[c].isna().all()
        ]
        X_num = numeric[numeric_cols] if numeric_cols else pd.DataFrame(index=X_df.index)
        X_cat = X_df.drop(columns=numeric_cols, errors="ignore")

        if fit:
            self._original_numeric_cols_ = list(X_num.columns)
            self._original_cat_cols_ = list(X_cat.columns)
            self._original_numeric_medians_ = X_num.median(numeric_only=True).fillna(0.0)
            self._original_numeric_medians_array_ = self._original_numeric_medians_.reindex(
                self._original_numeric_cols_
            ).to_numpy(dtype=np.float64, copy=True)
            X_num_filled = X_num.fillna(self._original_numeric_medians_)
            self._original_scaler_ = StandardScaler()
            X_num_arr = (
                self._original_scaler_.fit_transform(
                    X_num_filled.to_numpy(dtype=np.float64, copy=False)
                )
                if len(self._original_numeric_cols_)
                else np.empty((len(X_df), 0))
            )
            X_cat_dum = (
                pd.get_dummies(X_cat.astype("string"), dummy_na=True)
                if len(self._original_cat_cols_)
                else pd.DataFrame(index=X_df.index)
            )
            self._original_dummy_columns_ = list(X_cat_dum.columns)
        else:
            num_cols = getattr(self, "_original_numeric_cols_", [])
            med = getattr(self, "_original_numeric_medians_", pd.Series(dtype=float))
            X_num = numeric.reindex(columns=num_cols)
            X_num_filled = X_num.fillna(med).fillna(0.0)
            if len(num_cols):
                X_num_arr = self._original_scaler_.transform(
                    X_num_filled.to_numpy(dtype=np.float64, copy=False)
                )
            else:
                X_num_arr = np.empty((len(X_df), 0))
            cat_cols = getattr(self, "_original_cat_cols_", [])
            X_cat = X_df.reindex(columns=cat_cols)
            dummy_cols = getattr(self, "_original_dummy_columns_", [])
            X_cat_str = X_cat.astype("string") if len(cat_cols) else X_cat
            if len(cat_cols):
                # Same sentinel approach as the selected-original transform.
                dummy_set_local = set(dummy_cols)
                _COLLAPSE_SENTINEL = "\ue000__hugiml_other__"
                for col in cat_cols:
                    candidate_names = col + "_" + X_cat_str[col]
                    not_keep = X_cat_str[col].notna() & ~candidate_names.isin(dummy_set_local)
                    if not_keep.any():
                        X_cat_str.loc[not_keep, col] = _COLLAPSE_SENTINEL
            X_cat_dum = (
                pd.get_dummies(X_cat_str, dummy_na=True)
                if len(cat_cols)
                else pd.DataFrame(index=X_df.index)
            )
            X_cat_dum = X_cat_dum.reindex(columns=dummy_cols, fill_value=0)

        X_cat_arr = (
            X_cat_dum.to_numpy(dtype=np.float64, copy=False)
            if X_cat_dum.shape[1]
            else np.empty((len(X_df), 0))
        )
        X_base = np.hstack([X_num_arr, X_cat_arr]) if X_cat_arr.shape[1] else X_num_arr
        if fit:
            self._original_feature_names_downstream_ = list(
                getattr(self, "_original_numeric_cols_", [])
            ) + list(getattr(self, "_original_dummy_columns_", []))
        return X_base.astype(np.float32, copy=False)

    def _as_dense_float32(self, X: Any) -> np.ndarray:
        """Return a dense float32 2-D array without changing estimator semantics."""
        arr = X.toarray() if issparse(X) else np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr.astype(np.float32, copy=False)

    def _empty_dense_block(self, n_rows: int) -> np.ndarray:
        return np.empty((int(n_rows), 0), dtype=np.float32)

    def _original_topk_budget_enabled(self) -> bool:
        """Whether original features should be pre-budgeted before downstream fit."""
        return self.topK is not None and int(self.topK) >= 0

    def _select_original_topk_fit(
        self, X_base: Any, y: Any, names: list[str]
    ) -> tuple[Any, list[str]]:
        """Select at most topK original downstream columns and persist the mask.

        The mask is applied before concatenating originals with pattern blocks.
        This makes original_plus_* non-strict mode symmetric with mined and
        augmented features: originals contribute at most topK columns when a
        finite topK budget is configured.  Strict mode then performs the global
        topK pass over this already-budgeted candidate set.
        """
        n_cols = len(names)
        self._original_feature_names_downstream_full_ = list(names)
        self._original_feature_scores_downstream_ = np.zeros(n_cols, dtype=np.float64)
        self._original_feature_mask_downstream_ = np.ones(n_cols, dtype=bool)
        self._original_selected_feature_names_downstream_ = list(names)
        if n_cols == 0 or not self._original_topk_budget_enabled():
            return X_base, list(names)
        budget = min(max(1, int(self.topK)), n_cols)
        if budget >= n_cols:
            native_scores = getattr(self, "_native_original_feature_scores_downstream_", None)
            if native_scores is not None and len(native_scores) == n_cols:
                self._original_feature_scores_downstream_ = np.asarray(
                    native_scores, dtype=np.float64
                )
            return X_base, list(names)

        native_names = list(getattr(self, "_native_original_feature_names_downstream_", []) or [])
        native_scores = getattr(self, "_native_original_feature_scores_downstream_", None)
        if (
            native_names == list(names)
            and native_scores is not None
            and len(native_scores) == n_cols
        ):
            scores = np.asarray(native_scores, dtype=np.float64)
            order = np.lexsort((np.arange(n_cols), -scores))
            keep_idx = np.sort(order[:budget])
            mask = np.zeros(n_cols, dtype=bool)
            mask[keep_idx] = True
        else:
            # Non-fused or schema-mismatch path: still native, but necessarily
            # uses the dense downstream block because no preparation-stage score
            # metadata is available for this fit.
            scores, mask = self._strict_topk_dense_column_scores(X_base, y, names, top_k=budget)

        selected_names = [name for name, keep in zip(names, mask) if keep]
        self._original_feature_scores_downstream_ = scores
        self._original_feature_mask_downstream_ = mask
        self._original_selected_feature_names_downstream_ = list(selected_names)
        return X_base[:, mask], selected_names

    def _select_original_topk_transform(
        self, X_base: Any, names: list[str]
    ) -> tuple[Any, list[str]]:
        """Apply the persisted original-feature prefilter at transform time."""
        mask = getattr(self, "_original_feature_mask_downstream_", None)
        selected_names = getattr(self, "_original_selected_feature_names_downstream_", None)
        if mask is None:
            return X_base, list(names)
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.size != len(names):
            return X_base, list(names)
        if bool(np.all(mask_arr)):
            return X_base, list(names if selected_names is None else selected_names)
        return X_base[:, mask_arr], list(names if selected_names is None else selected_names)

    def _selected_original_downstream_names(self) -> list[str]:
        names = getattr(self, "_original_selected_feature_names_downstream_", None)
        if names is not None:
            return list(names)
        return [f"orig:{name}" for name in getattr(self, "_original_feature_names_downstream_", [])]

    def _strict_topk_dense_column_scores(
        self, X: Any, y: Any, names: list[str], top_k: int = -1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score dense downstream columns and return ``(scores, mask)``.

        Prefer the native dense TopK helper when the extension exposes it,
        but fall back to the same Python IG scoring used by the sparse path.
        This preserves correctness for hybrid feature modes with
        ``topk_budget_strict=True``.
        """
        X_arr = np.ascontiguousarray(self._as_dense_float32(X), dtype=np.float32)
        n_cols = int(X_arr.shape[1])
        if n_cols == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool)
        y_codes, _ = pd.factorize(np.asarray(y), sort=True)
        y_codes = np.ascontiguousarray(y_codes.astype(np.int64, copy=False))
        discrete_mask_bool = self._strict_topk_discrete_mask(names).astype(bool, copy=False)
        max_bins = max(
            8,
            int(
                getattr(self, "B", 8) if getattr(self, "B", 8) and getattr(self, "B", 8) > 0 else 16
            ),
        )
        if _CORE_AVAILABLE and hasattr(_core, "strict_topk_filter_dense"):
            discrete_mask = np.ascontiguousarray(discrete_mask_bool.astype(np.uint8, copy=False))
            scores, mask = _core.strict_topk_filter_dense(
                X_arr, y_codes, discrete_mask, int(top_k), int(max_bins)
            )
            return np.asarray(scores, dtype=np.float64), np.asarray(mask, dtype=bool)

        n_classes = int(np.max(y_codes)) + 1 if y_codes.size else 0
        scores = np.zeros(n_cols, dtype=np.float64)
        if n_classes > 1:
            for j in range(n_cols):
                col = np.asarray(X_arr[:, j], dtype=np.float64)
                if bool(discrete_mask_bool[j]):
                    vals = np.where(col > 0.5, 1, 0).astype(np.int64, copy=False)
                else:
                    vals = _continuous_to_quantile_codes(col, max_bins=max_bins)
                scores[j] = _information_gain_from_codes(vals, y_codes, n_classes)

        budget = int(top_k)
        mask = np.ones(n_cols, dtype=bool)
        if budget >= 0 and budget < n_cols:
            budget = max(1, budget)
            order = np.lexsort((np.arange(n_cols), -scores))
            keep_idx = np.sort(order[:budget])
            mask = np.zeros(n_cols, dtype=bool)
            mask[keep_idx] = True
        return scores, mask

    def _combine_dense_blocks(self, blocks: list[Any], n_rows: int) -> np.ndarray:
        """Combine selected downstream blocks as one dense float32 matrix."""
        dense_blocks = []
        for block in blocks:
            if block is None or int(getattr(block, "shape", (n_rows, 0))[1]) == 0:
                continue
            dense_blocks.append(self._as_dense_float32(block))
        if not dense_blocks:
            return self._empty_dense_block(n_rows)
        if len(dense_blocks) == 1:
            return dense_blocks[0]
        return np.hstack(dense_blocks).astype(np.float32, copy=False)

    def _dense_downstream_width_threshold(self) -> int:
        """Width threshold for materializing hybrid downstream matrices as dense.

        Hybrid feature modes contain dense original columns and sparse pattern
        columns.  Dense output is fastest and smallest for small selected widths
        because LR sees a compact float32 ndarray.  For large selected widths,
        densifying sparse pattern columns wastes memory; keep the output CSR.

        The threshold is user-configurable via the sklearn parameter
        ``dense_downstream_max_width``.  Set it to 0 to keep hybrid matrices CSR
        except for the empty-width degenerate case.  Older loaded models that
        lack the public parameter can still fall back to the private compatibility
        attribute or the historical default of 200.
        """
        try:
            value = getattr(
                self,
                "dense_downstream_max_width",
                getattr(self, "_dense_downstream_max_width_", 200),
            )
            if isinstance(value, bool):
                return 200
            return max(0, int(value))
        except Exception:
            return 200

    def _should_use_dense_downstream(self, total_width: int) -> bool:
        mode = getattr(self, "feature_mode", "patterns_only")
        if mode == "patterns_only":
            return False
        threshold = self._dense_downstream_width_threshold()
        return int(total_width) <= int(threshold)

    def _combine_downstream_blocks(self, blocks: list[Any], n_rows: int):
        """Combine downstream blocks using a memory-aware output format.

        * patterns_only is handled earlier and remains CSR.
        * hybrid modes use dense float32 for small/moderate widths, avoiding
          sparse->dense churn before LR.
        * hybrid modes use CSR once width is large enough that densifying sparse
          pattern/augmented blocks would dominate memory.
        """
        live_blocks = [
            block
            for block in blocks
            if block is not None and int(getattr(block, "shape", (n_rows, 0))[1]) > 0
        ]
        total_width = sum(int(block.shape[1]) for block in live_blocks)
        if total_width == 0:
            return (
                self._empty_dense_block(n_rows)
                if self._should_use_dense_downstream(0)
                else csr_matrix((int(n_rows), 0), dtype=np.float32)
            )
        if self._should_use_dense_downstream(total_width):
            return self._combine_dense_blocks(live_blocks, n_rows)
        sparse_blocks = []
        for block in live_blocks:
            if issparse(block):
                sparse_blocks.append(block.astype(np.float32, copy=False).tocsr())
            else:
                sparse_blocks.append(csr_matrix(np.asarray(block, dtype=np.float32)))
        if len(sparse_blocks) == 1:
            return sparse_blocks[0]
        return hstack(sparse_blocks, format="csr", dtype=np.float32)

    def _select_strict_topk_from_blocks_fit(
        self,
        blocks: list[tuple[str, Any, list[str]]],
        y: Any,
    ) -> tuple[list[Any], list[str]]:
        """Apply strict global TopK before concatenating downstream feature blocks.

        This keeps strict mode as a compute budget: dense original columns are
        scored directly, sparse pattern columns are scored in sparse form, and
        only selected columns are materialized for the downstream estimator.
        The persisted full mask and scores retain the same public/serialization
        contract as the prior post-concatenation implementation.
        """
        full_names: list[str] = []
        full_scores_parts: list[np.ndarray] = []
        selected_blocks: list[Any] = []

        for _, block, names in blocks:
            full_names.extend(list(names))
        n_cols = len(full_names)
        self._downstream_feature_names_full_ = list(full_names)
        self._strict_topk_applied_during_construction_ = False
        self._strict_topk_feature_scores_ = np.zeros(n_cols, dtype=np.float64)
        self._strict_topk_feature_mask_ = np.ones(n_cols, dtype=bool)
        self._strict_topk_selected_feature_names_ = list(full_names)

        if (
            not bool(getattr(self, "topk_budget_strict", False))
            or self.topK is None
            or int(self.topK) < 0
            or n_cols == 0
            or int(self.topK) >= n_cols
        ):
            return [block for _, block, _ in blocks], list(full_names)

        for kind, block, names in blocks:
            if len(names) == 0:
                full_scores_parts.append(np.zeros(0, dtype=np.float64))
            elif kind == "dense":
                full_scores_parts.append(
                    self._strict_topk_dense_column_scores(block, y, names, top_k=-1)[0]
                )
            else:
                full_scores_parts.append(self._strict_topk_column_scores(block, y, names))
        scores = (
            np.concatenate(full_scores_parts).astype(np.float64, copy=False)
            if full_scores_parts
            else np.zeros(0, dtype=np.float64)
        )
        budget = min(max(1, int(self.topK)), n_cols)
        order = np.lexsort((np.arange(n_cols), -scores))
        keep_idx = np.sort(order[:budget])
        mask = np.zeros(n_cols, dtype=bool)
        mask[keep_idx] = True

        offset = 0
        selected_names: list[str] = []
        for kind, block, names in blocks:
            width = len(names)
            block_mask = mask[offset : offset + width]
            if width and bool(np.any(block_mask)):
                selected_names.extend([name for name, keep in zip(names, block_mask) if keep])
                selected_blocks.append(block[:, block_mask])
            offset += width

        self._strict_topk_feature_scores_ = scores
        self._strict_topk_feature_mask_ = mask
        self._strict_topk_selected_feature_names_ = list(selected_names)
        self._strict_topk_applied_during_construction_ = True
        return selected_blocks, selected_names

    def _select_strict_topk_from_blocks_transform(
        self, blocks: list[tuple[str, Any, list[str]]]
    ) -> list[Any]:
        """Select persisted strict TopK columns from logical blocks before concat."""
        mask = getattr(self, "_strict_topk_feature_mask_", None)
        if mask is None:
            return [block for _, block, _ in blocks]
        mask_arr = np.asarray(mask, dtype=bool)
        full_width = sum(len(names) for _, _, names in blocks)
        if mask_arr.size != full_width or bool(np.all(mask_arr)):
            return [block for _, block, _ in blocks]
        selected_blocks: list[Any] = []
        offset = 0
        for _, block, names in blocks:
            width = len(names)
            block_mask = mask_arr[offset : offset + width]
            if width and bool(np.any(block_mask)):
                selected_blocks.append(block[:, block_mask])
            offset += width
        return selected_blocks

    def _make_downstream_features(self, X_original: Any, Z_patterns: csr_matrix, fit: bool = False):
        """Build the estimator input matrix for the configured feature_mode."""
        mode = getattr(self, "feature_mode", "patterns_only")
        Z = Z_patterns if issparse(Z_patterns) else csr_matrix(Z_patterns)
        Z_aug = self._make_augmented_pair_features(X_original, fit=fit)
        n_rows = int(Z.shape[0])
        if mode == "patterns_only":
            return hstack([Z, Z_aug], format="csr") if Z_aug.shape[1] else Z

        if not fit:
            selected_original_names = getattr(
                self, "_original_selected_feature_names_downstream_", None
            )
            full_original_names = [
                f"orig:{name}" for name in getattr(self, "_original_feature_names_downstream_", [])
            ]
            mask = getattr(self, "_original_feature_mask_downstream_", None)
            use_selected_originals = (
                selected_original_names is not None
                and mask is not None
                and len(selected_original_names) <= len(full_original_names)
            )
            if use_selected_originals:
                X_base, original_names = (
                    self._prepare_selected_original_features_for_downstream_transform(
                        X_original, list(selected_original_names)
                    )
                )
            else:
                X_base = self._prepare_original_features_for_downstream(X_original, fit=False)
                X_base, original_names = self._select_original_topk_transform(
                    X_base, full_original_names
                )
        else:
            X_base = self._prepare_original_features_for_downstream(X_original, fit=True)
            original_names_full = [
                f"orig:{name}" for name in getattr(self, "_original_feature_names_downstream_", [])
            ]
            X_base, original_names = self._select_original_topk_fit(
                X_base, self._current_y_for_downstream_topk_, original_names_full
            )
        pattern_names = [f"pattern:{name}" for name in self.get_hug_features()]
        aug_names = [
            f"augmented_pair:{t['name']}" for t in getattr(self, "augmented_pair_transforms_", [])
        ]

        if mode == "original_plus_patterns":
            blocks = [("dense", X_base, original_names), ("sparse", Z, pattern_names)]
            if Z_aug.shape[1]:
                blocks.append(("sparse", Z_aug, aug_names))
            if fit and bool(getattr(self, "topk_budget_strict", False)):
                selected_blocks, _ = self._select_strict_topk_from_blocks_fit(
                    blocks, self._current_y_for_downstream_topk_
                )
                return self._combine_downstream_blocks(selected_blocks, n_rows)
            if bool(getattr(self, "topk_budget_strict", False)):
                selected_blocks = self._select_strict_topk_from_blocks_transform(blocks)
                return self._combine_downstream_blocks(selected_blocks, n_rows)
            return self._combine_downstream_blocks([X_base, Z, Z_aug], n_rows)

        if mode == "original_plus_interactions":
            mask = getattr(self, "_interaction_pattern_mask_", None)
            if mask is None:
                self._setup_feature_mode_metadata()
                mask = self._interaction_pattern_mask_
            Z_sel = Z[:, mask]
            selected_pattern_names = [name for name, keep in zip(pattern_names, mask) if keep]
            blocks = [("dense", X_base, original_names), ("sparse", Z_sel, selected_pattern_names)]
            if Z_aug.shape[1]:
                blocks.append(("sparse", Z_aug, aug_names))
            if fit and bool(getattr(self, "topk_budget_strict", False)):
                selected_blocks, _ = self._select_strict_topk_from_blocks_fit(
                    blocks, self._current_y_for_downstream_topk_
                )
                return self._combine_downstream_blocks(selected_blocks, n_rows)
            if bool(getattr(self, "topk_budget_strict", False)):
                selected_blocks = self._select_strict_topk_from_blocks_transform(blocks)
                return self._combine_downstream_blocks(selected_blocks, n_rows)
            return self._combine_downstream_blocks([X_base, Z_sel, Z_aug], n_rows)

        raise HUGIMLParamError(f"Unknown feature_mode={mode!r}.")

    def _get_downstream_feature_names_full(self) -> list[str]:
        """Names for the unfiltered downstream feature matrix."""
        mode = getattr(self, "feature_mode", "patterns_only")
        pattern_names = list(self.get_hug_features())
        aug_names = [
            f"augmented_pair:{t['name']}" for t in getattr(self, "augmented_pair_transforms_", [])
        ]
        if mode == "patterns_only":
            return [f"pattern:{name}" for name in pattern_names] + aug_names
        original_names = self._selected_original_downstream_names()
        if mode == "original_plus_patterns":
            return original_names + [f"pattern:{name}" for name in pattern_names] + aug_names
        if mode == "original_plus_interactions":
            mask = getattr(
                self, "_interaction_pattern_mask_", np.ones(len(pattern_names), dtype=bool)
            )
            selected = [name for name, keep in zip(pattern_names, mask) if keep]
            return original_names + [f"pattern:{name}" for name in selected] + aug_names
        return [f"pattern:{name}" for name in pattern_names]

    def _get_downstream_feature_names(self) -> list[str]:
        """Names aligned with coefficients of the downstream estimator."""
        names = self._get_downstream_feature_names_full()
        mask = getattr(self, "_strict_topk_feature_mask_", None)
        if mask is None:
            return names
        return [name for name, keep in zip(names, np.asarray(mask, dtype=bool)) if keep]

    def _is_discrete_downstream_feature(self, name: str) -> bool:
        return name.startswith("pattern:") or (
            name.startswith("orig:") and name in getattr(self, "_strict_topk_dummy_names_", set())
        )

    def _strict_topk_discrete_mask(self, names: list[str]) -> np.ndarray:
        """Boolean mask of downstream columns that should be IG-scored as discrete."""
        dummy_names = {f"orig:{c}" for c in getattr(self, "_original_dummy_columns_", [])}
        self._strict_topk_dummy_names_ = dummy_names
        return np.asarray(
            [name.startswith("pattern:") or name in dummy_names for name in names],
            dtype=np.uint8,
        )

    def _strict_topk_column_scores(self, X: csr_matrix, y: Any, names: list[str]) -> np.ndarray:
        """Compute comparable IG scores for strict global topK filtering.

        The native path scores CSC columns directly. The Python fallback is kept
        only for source-tree development before the extension has been rebuilt.
        """
        y_codes, _ = pd.factorize(np.asarray(y), sort=True)
        n_classes = int(np.max(y_codes)) + 1 if y_codes.size else 0
        if n_classes <= 1:
            return np.zeros(X.shape[1], dtype=np.float64)
        X_csc = X.tocsc() if issparse(X) else csr_matrix(X).tocsc()
        discrete_mask = self._strict_topk_discrete_mask(names)
        max_bins = max(
            8,
            int(
                getattr(self, "B", 8) if getattr(self, "B", 8) and getattr(self, "B", 8) > 0 else 16
            ),
        )
        if _CORE_AVAILABLE and hasattr(_core, "strict_topk_filter_csc"):
            scores, _ = _core.strict_topk_filter_csc(
                np.asarray(X_csc.data, dtype=np.float32),
                np.asarray(X_csc.indices, dtype=np.int32),
                np.asarray(X_csc.indptr, dtype=np.int32),
                int(X_csc.shape[0]),
                int(X_csc.shape[1]),
                np.asarray(y_codes, dtype=np.int64),
                discrete_mask,
                -1,
                int(max_bins),
            )
            return np.asarray(scores, dtype=np.float64)
        scores = np.zeros(X.shape[1], dtype=np.float64)
        for j in range(X.shape[1]):
            col = np.asarray(X_csc[:, j].toarray()).ravel()
            if bool(discrete_mask[j]):
                vals = np.where(col > 0.5, 1, 0).astype(np.int64, copy=False)
            else:
                vals = _continuous_to_quantile_codes(col, max_bins=max_bins)
            scores[j] = _information_gain_from_codes(vals, y_codes, n_classes)
        return scores

    def _apply_strict_topk_budget_fit(self, X: csr_matrix, y: Any) -> csr_matrix:
        """Optionally apply a single native global IG topK budget over all downstream features."""
        n_cols = int(X.shape[1])
        names = self._get_downstream_feature_names_full()
        existing_mask = getattr(self, "_strict_topk_feature_mask_", None)
        if bool(getattr(self, "_strict_topk_applied_during_construction_", False)):
            return X
        if (
            bool(getattr(self, "topk_budget_strict", False))
            and existing_mask is not None
            and len(existing_mask) != n_cols
        ):
            # Strict TopK was already applied during block-wise downstream
            # construction.  Preserve the full-length persisted mask/scores and
            # return the selected estimator matrix unchanged.
            return X
        self._downstream_feature_names_full_ = list(names)
        self._strict_topk_feature_scores_ = np.zeros(n_cols, dtype=np.float64)
        self._strict_topk_feature_mask_ = np.ones(n_cols, dtype=bool)
        self._strict_topk_selected_feature_names_ = list(names)
        if (
            not bool(getattr(self, "topk_budget_strict", False))
            or self.topK is None
            or int(self.topK) < 0
            or n_cols == 0
            or int(self.topK) >= n_cols
        ):
            return X
        budget = min(max(1, int(self.topK)), n_cols)
        y_codes, _ = pd.factorize(np.asarray(y), sort=True)
        X_csc = X.tocsc() if issparse(X) else csr_matrix(X).tocsc()
        discrete_mask = self._strict_topk_discrete_mask(names)
        max_bins = max(
            8,
            int(
                getattr(self, "B", 8) if getattr(self, "B", 8) and getattr(self, "B", 8) > 0 else 16
            ),
        )
        if _CORE_AVAILABLE and hasattr(_core, "strict_topk_filter_csc"):
            scores, mask_native = _core.strict_topk_filter_csc(
                np.asarray(X_csc.data, dtype=np.float32),
                np.asarray(X_csc.indices, dtype=np.int32),
                np.asarray(X_csc.indptr, dtype=np.int32),
                int(X_csc.shape[0]),
                int(X_csc.shape[1]),
                np.asarray(y_codes, dtype=np.int64),
                discrete_mask,
                int(budget),
                int(max_bins),
            )
            scores = np.asarray(scores, dtype=np.float64)
            mask = np.asarray(mask_native, dtype=bool)
        else:
            scores = self._strict_topk_column_scores(X, y, names)
            order = np.lexsort((np.arange(n_cols), -scores))
            keep_idx = np.sort(order[:budget])
            mask = np.zeros(n_cols, dtype=bool)
            mask[keep_idx] = True
        self._strict_topk_feature_scores_ = scores
        self._strict_topk_feature_mask_ = mask
        self._strict_topk_selected_feature_names_ = [
            name for name, keep in zip(names, mask) if keep
        ]
        return X[:, mask]

    def _apply_strict_topk_budget_transform(self, X: csr_matrix) -> csr_matrix:
        if bool(getattr(self, "_strict_topk_applied_during_construction_", False)):
            return X
        mask = getattr(self, "_strict_topk_feature_mask_", None)
        if mask is None:
            return X
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.size != X.shape[1] or bool(np.all(mask_arr)):
            return X
        return X[:, mask_arr]

    def _numeric_feature_names_for_augmented_pairs(self) -> list[str]:
        names = list(getattr(self, "feature_names_in_", []) or [])
        cat_mask = getattr(self, "cat_cols_mask_", np.zeros(len(names), dtype=bool))
        return [name for name, is_cat in zip(names, cat_mask) if not bool(is_cat)]

    def _setup_augmented_pair_transforms(
        self, X_original: Any, y: Any | None = None, fit: bool = False
    ) -> None:
        """Create internal augmented_pair_transforms when L>1.

        This reuses native adaptive-binning IG metadata and does not expose a
        public hyperparameter. Augmented pair transforms are not fed into HUGIML
        mining; they are appended only before the downstream estimator.
        """
        enabled = bool(self.augmented_pair_transforms)
        if not fit or self.L <= 1 or not enabled or X_original is None:
            self._augmented_pair_block_ = None
            self.augmented_pair_transforms_ = []
            self.augmented_pair_selected_features_ = []
            self.augmented_pair_transforms_enabled_ = False
            return
        if not getattr(self, "adaptive_binning", False):
            warnings.warn(
                "augmented_pair_transforms require adaptive_binning=True because they are selected from adaptive-binning IG metadata; no augmented pair features will be added.",
                HUGIMLWarning,
                stacklevel=2,
            )
            self._augmented_pair_block_ = None
            self.augmented_pair_transforms_ = []
            self.augmented_pair_selected_features_ = []
            self.augmented_pair_transforms_enabled_ = False
            self.augmented_pair_config_ = {
                "enabled": False,
                "reason": "adaptive_binning_required",
                "augmented_pair_mode": str(self.augmented_pair_mode),
                "aug_feature_size": int(self.aug_feature_size),
                "ii_partner_size": self.ii_partner_size,
                "max_pair_features": int(self.max_pair_features),
                "budget": int(self.topK) if self.topK is not None and self.topK >= 0 else None,
                "num_candidates": 0,
                "num_retained": 0,
            }
            return
        if not getattr(self, "ig_scores_", None):
            warnings.warn(
                "augmented_pair_transforms were requested but no adaptive-binning IG scores are available; no augmented pair features will be added.",
                HUGIMLWarning,
                stacklevel=2,
            )
            self._augmented_pair_block_ = None
            self.augmented_pair_transforms_ = []
            self.augmented_pair_selected_features_ = []
            self.augmented_pair_transforms_enabled_ = False
            self.augmented_pair_config_ = {
                "enabled": False,
                "reason": "missing_ig_scores",
                "augmented_pair_mode": str(self.augmented_pair_mode),
                "aug_feature_size": int(self.aug_feature_size),
                "ii_partner_size": self.ii_partner_size,
                "max_pair_features": int(self.max_pair_features),
                "budget": int(self.topK) if self.topK is not None and self.topK >= 0 else None,
                "num_candidates": 0,
                "num_retained": 0,
            }
            return
        pair_budget = None if bool(getattr(self, "topk_budget_strict", False)) else self.topK
        block = NativeAugmentedPairTransformBlock(
            augmented_pair_mode=self.augmented_pair_mode,
            aug_feature_size=self.aug_feature_size,
            ii_partner_size=self.ii_partner_size,
            max_pair_features=self.max_pair_features,
            budget_topK=pair_budget,
            min_source_ig=self.G,
        )
        block.fit(
            X_original,
            y,
            getattr(self, "ig_scores_", {}) or {},
            getattr(self, "_bin_edges_", {}) or {},
            self._numeric_feature_names_for_augmented_pairs(),
            budget_topK=pair_budget,
            min_source_ig=self.G,
            full_feature_names=list(getattr(self, "feature_names_in_", []) or []),
        )
        self._augmented_pair_block_ = block
        self.augmented_pair_transforms_ = list(block.augmented_pair_transforms_)
        self.augmented_pair_selected_features_ = list(block.selected_aug_features_)
        self.augmented_pair_transforms_enabled_ = bool(self.augmented_pair_transforms_)
        self.augmented_pair_config_ = {
            "enabled": self.augmented_pair_transforms_enabled_,
            "augmented_pair_mode": str(self.augmented_pair_mode),
            "aug_feature_size": int(self.aug_feature_size),
            "ii_partner_size": self.ii_partner_size,
            "max_pair_features": int(self.max_pair_features),
            "budget": int(self.topK) if self.topK is not None and self.topK >= 0 else None,
            "budget_source": "global_strict_topK"
            if bool(getattr(self, "topk_budget_strict", False))
            else "topK",
            "ops": ["product", "absolute_difference", "sum", "signed_difference"],
            "score": str(self.augmented_pair_mode),
            "min_source_ig": float(
                getattr(block, "min_source_ig_", max(1e-12, float(self.G or 0.0)))
            ),
            "num_candidates": int(getattr(block, "candidate_count_", 0)),
            "num_retained": len(self.augmented_pair_transforms_),
        }

    def _make_augmented_pair_features(self, X_original: Any, fit: bool = False):
        if self.L <= 1 or not bool(self.augmented_pair_transforms) or X_original is None:
            n_rows = 0 if X_original is None else len(X_original)
            return csr_matrix((n_rows, 0), dtype=np.float32)
        block = getattr(self, "_augmented_pair_block_", None)
        if block is None:
            return csr_matrix((len(X_original), 0), dtype=np.float32)
        return block.transform(X_original)

    def get_augmented_pair_transforms(self) -> list[dict[str, Any]]:
        """Return augmented pair transforms used by the downstream estimator.

        Each catalog entry includes the raw pair formula, source-feature IG
        provenance, candidate coverage, unavailable-pair policy, and the
        standardization parameters used before the downstream estimator sees
        the feature.  Candidate IG is scored on rows where both source values
        are observed.  For selected features, rows where the pair value cannot
        be computed receive the pair feature's training reference value before
        standardization, yielding a neutral standardized value.
        """
        return [dict(item) for item in getattr(self, "augmented_pair_transforms_", [])]

    def get_augmented_pair_standardization(self) -> pd.DataFrame:
        """Return standardization metadata for augmented pair features.

        The returned columns are aligned to ``get_augmented_pair_transforms()``
        and make the raw-to-estimator transformation explicit.
        """
        rows: list[dict[str, Any]] = []
        for item in self.get_augmented_pair_transforms():
            rows.append(
                {
                    "name": item.get("name"),
                    "operation": item.get("operation"),
                    "inputs": item.get("inputs"),
                    "raw_formula": item.get("raw_formula", item.get("formula")),
                    "standardization_mean": item.get("standardization_mean"),
                    "standardization_scale": item.get("standardization_scale"),
                    "standardized_formula": item.get("standardized_formula"),
                    "reference_raw_value": item.get("reference_raw_value"),
                    "pair_missing_policy": item.get("pair_missing_policy"),
                    "eligible_count": item.get("eligible_count"),
                    "eligible_rate": item.get("eligible_rate"),
                    "missing_pair_rate": item.get("missing_pair_rate"),
                    "source_observed_medians": item.get("source_observed_medians"),
                    "transform_ig": item.get("transform_ig"),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _format_source_observed_medians(source_observed_medians: Any) -> str:
        if not isinstance(source_observed_medians, dict) or not source_observed_medians:
            return "not available"
        parts: list[str] = []
        for key, value in source_observed_medians.items():
            try:
                parts.append(f"{key}={float(value):.6g}")
            except (TypeError, ValueError):
                parts.append(f"{key}={value}")
        return ", ".join(parts)

    @staticmethod
    def _augmented_pair_effect_text(
        *,
        raw_formula: str,
        operation: str,
        coefficient_raw_scale: float,
        standardization_mean: float,
        standardization_scale: float,
        source_observed_medians: Any,
        pair_missing_policy: str,
        eligible_rate: float,
        missing_pair_rate: float,
    ) -> dict[str, Any]:
        eligible_text = (
            f"Candidate scoring used rows where both source values were observed"
            f" (eligible_rate={eligible_rate:.3g})."
            if np.isfinite(eligible_rate)
            else "Candidate scoring used rows where both source values were observed."
        )
        missing_text = (
            f"training rows where the pair was unavailable: {missing_pair_rate:.3g}."
            if np.isfinite(missing_pair_rate)
            else "the unavailable-pair rate is not available."
        )
        reference_note = (
            f"The reference raw value {standardization_mean:.6g} is the training-cohort mean "
            f"of the observed {raw_formula} pair term after applying the selected pair operation. "
            "It is not a domain-specific baseline."
        )
        source_median_text = HUGIMLClassifierNative._format_source_observed_medians(
            source_observed_medians
        )
        missing_policy_note = (
            "If a selected pair cannot be computed for a row because one or both source values are missing, "
            "the augmented-pair feature is set to its training reference raw value before standardization. "
            "That gives the pair term a neutral standardized value of 0 for that row. "
            "This policy applies only to continuous augmented-pair features; HUGIML pattern features keep "
            "their native missing-value handling. "
            f"For diagnostics, source feature medians observed in training were: {source_median_text}."
        )

        if not np.isfinite(coefficient_raw_scale):
            return {
                "decision_direction": "effect_not_available",
                "risk_increases_when": "not_available",
                "unit_effect_interpretation": "Raw-scale log-odds effect is not available for this downstream estimator.",
                "reference_raw_value_description": "training_cohort_mean_of_observed_raw_pair_value",
                "source_observed_medians_description": "per-source-feature observed medians for diagnostics only; not used to construct pair values",
                "pair_missing_policy_description": "unavailable pair values are set to the pair reference raw value before standardization",
                "raw_scale_note": f"{reference_note} {eligible_text}",
                "raw_interpretation": (
                    f"The downstream estimator uses ({raw_formula} - {standardization_mean:.6g}) "
                    f"/ {standardization_scale:.6g}. {reference_note} {eligible_text} "
                    f"For selected-feature construction, {missing_policy_note}"
                ),
            }

        if coefficient_raw_scale > 0:
            direction = "higher_raw_value_increases_score"
            direction_text = f"Higher {raw_formula} increases the model score."
        elif coefficient_raw_scale < 0:
            direction = "higher_raw_value_decreases_score"
            direction_text = f"Higher {raw_formula} decreases the model score."
        else:
            direction = "raw_value_has_zero_linear_effect"
            direction_text = f"Higher {raw_formula} does not change the linear model score."

        if operation == "absolute_difference":
            risk_when = (
                "absolute_difference_increases"
                if coefficient_raw_scale > 0
                else "absolute_difference_decreases"
                if coefficient_raw_scale < 0
                else "not_applicable"
            )
            unit_text = (
                f"Each +1 increase in the absolute difference term changes the log-odds by "
                f"{coefficient_raw_scale:.6g}."
            )
            raw_scale_note = (
                "The raw-unit effect is expressed on the absolute-difference scale. "
                + reference_note
            )
        elif operation == "product":
            risk_when = (
                "product_value_increases"
                if coefficient_raw_scale > 0
                else "product_value_decreases"
                if coefficient_raw_scale < 0
                else "not_applicable"
            )
            unit_text = (
                f"A +1 change in the product term changes the log-odds by "
                f"{coefficient_raw_scale:.6g}. For a product feature, changing one source "
                "variable does not have a constant marginal effect; it depends on the current "
                "value of the other source variable."
            )
            raw_scale_note = (
                "The raw-unit effect is expressed on the product-term scale, not as a constant "
                "one-unit effect of either individual source feature. " + reference_note
            )
        elif operation == "sum":
            risk_when = (
                "sum_value_increases"
                if coefficient_raw_scale > 0
                else "sum_value_decreases"
                if coefficient_raw_scale < 0
                else "not_applicable"
            )
            unit_text = (
                f"Each +1 increase in the sum term changes the log-odds by "
                f"{coefficient_raw_scale:.6g}. The same coefficient applies to a one-unit "
                "increase in either source feature while the other source feature is kept constant."
            )
            raw_scale_note = (
                "The raw-unit effect is expressed on the pair sum scale. " + reference_note
            )
        elif operation == "signed_difference":
            risk_when = (
                "left_minus_right_increases"
                if coefficient_raw_scale > 0
                else "left_minus_right_decreases"
                if coefficient_raw_scale < 0
                else "not_applicable"
            )
            unit_text = (
                f"Each +1 increase in the signed difference term changes the log-odds by "
                f"{coefficient_raw_scale:.6g}. Increasing the left source feature raises this "
                "term, while increasing the right source feature lowers it."
            )
            raw_scale_note = (
                "The raw-unit effect is expressed on the signed left-minus-right difference scale. "
                + reference_note
            )
        else:
            risk_when = (
                "raw_value_increases"
                if coefficient_raw_scale > 0
                else "raw_value_decreases"
                if coefficient_raw_scale < 0
                else "not_applicable"
            )
            unit_text = (
                f"Each +1 raw-unit increase changes the log-odds by {coefficient_raw_scale:.6g}."
            )
            raw_scale_note = reference_note

        return {
            "decision_direction": direction,
            "risk_increases_when": risk_when,
            "unit_effect_interpretation": unit_text,
            "reference_raw_value_description": "training_cohort_mean_of_observed_raw_pair_value",
            "source_observed_medians_description": "per-source-feature observed medians for diagnostics only; not used to construct pair values",
            "pair_missing_policy_description": "unavailable pair values are set to the pair reference raw value before standardization",
            "raw_scale_note": f"{raw_scale_note} {eligible_text}",
            "raw_interpretation": (
                f"{direction_text} {unit_text} The downstream estimator uses "
                f"({raw_formula} - {standardization_mean:.6g}) / {standardization_scale:.6g}. "
                f"{reference_note} {eligible_text} For selected-feature construction, {missing_policy_note} "
                f"Among training rows, {missing_text}"
            ),
        }

    def _augmented_pair_effect_rows(self) -> list[dict[str, Any]]:
        """Return augmented-pair effect rows in raw and standardized units."""
        check_is_fitted(self)
        try:
            imp = self.feature_importances()
            coef_lookup = dict(zip(imp["feature"], imp["coefficient"]))
        except AttributeError:
            coef_lookup = {}

        rows: list[dict[str, Any]] = []
        for item in self.get_augmented_pair_transforms():
            name = str(item.get("name"))
            feature = f"augmented_pair:{name}"
            coef_std = float(coef_lookup.get(feature, np.nan))
            mean = float(item.get("standardization_mean", np.nan))
            scale = float(item.get("standardization_scale", np.nan))
            scale_safe = scale if np.isfinite(scale) and scale != 0.0 else np.nan
            coef_raw = (
                coef_std / scale_safe
                if np.isfinite(coef_std) and np.isfinite(scale_safe)
                else np.nan
            )
            operation = str(item.get("operation", ""))
            raw_formula = str(item.get("raw_formula", item.get("formula", name)))

            text = self._augmented_pair_effect_text(
                raw_formula=raw_formula,
                operation=operation,
                coefficient_raw_scale=coef_raw,
                standardization_mean=mean,
                standardization_scale=scale,
                source_observed_medians=item.get("source_observed_medians"),
                pair_missing_policy=str(
                    item.get("pair_missing_policy", "reference_value_for_unavailable_pair")
                ),
                eligible_rate=float(item.get("eligible_rate", np.nan)),
                missing_pair_rate=float(item.get("missing_pair_rate", np.nan)),
            )

            rows.append(
                {
                    "feature": feature,
                    "name": name,
                    "operation": operation,
                    "inputs": item.get("inputs"),
                    "raw_formula": raw_formula,
                    "standardized_formula": item.get("standardized_formula"),
                    "standardization_mean": mean,
                    "standardization_scale": scale,
                    "reference_raw_value": mean,
                    "reference_raw_value_description": text["reference_raw_value_description"],
                    "coefficient_standardized": coef_std,
                    "one_std_effect_on_log_odds": coef_std,
                    "coefficient_raw_scale": coef_raw,
                    "one_raw_unit_effect_on_log_odds": coef_raw,
                    "decision_direction": text["decision_direction"],
                    "risk_increases_when": text["risk_increases_when"],
                    "unit_effect_interpretation": text["unit_effect_interpretation"],
                    "raw_scale_note": text["raw_scale_note"],
                    "raw_interpretation": text["raw_interpretation"],
                    "pair_missing_policy": item.get("pair_missing_policy"),
                    "pair_missing_policy_description": text["pair_missing_policy_description"],
                    "eligible_count": item.get("eligible_count"),
                    "eligible_rate": item.get("eligible_rate"),
                    "missing_pair_rate": item.get("missing_pair_rate"),
                    "source_observed_medians": item.get("source_observed_medians"),
                    "source_observed_medians_description": text[
                        "source_observed_medians_description"
                    ],
                    "transform_ig": item.get("transform_ig"),
                }
            )
        return rows

    def explain_augmented_pair_effects(self) -> pd.DataFrame:
        """Explain augmented-pair effects in standardized and raw units.

        The downstream estimator is fit on standardized augmented-pair values.
        This method converts each standardized coefficient back to the raw pair
        scale and states that the reference value is the training-cohort mean
        of the observed pair term, not a domain-specific baseline. Candidate
        scoring uses rows where both source values are observed. For selected
        features, rows where the pair cannot be computed receive the pair
        feature's training reference raw value before standardization, yielding
        a neutral standardized value for that pair term. HUGIML pattern
        features keep their native missing-value handling.

        For logistic-regression downstream models, coefficient columns are
        log-odds effects.  Product-term effects are expressed on the product
        scale; changing one individual input does not have a constant marginal
        effect because it depends on the current value of the other input.
        """
        return pd.DataFrame(self._augmented_pair_effect_rows())

    def transform(self, X: Any) -> csr_matrix:
        """Return the binary HUG pattern matrix for X.

        Each column corresponds to one mined pattern.  Entry (i, j) is 1 when
        all items of pattern j appear in row i.

        Parameters
        ----------
        X : array-like or DataFrame

        Returns
        -------
        csr_matrix, shape (n_samples, n_patterns)
        """
        check_is_fitted(self)
        # ── Adaptive pre-binning ─────────────────────────────────────────
        if getattr(self, "adaptive_binning", False) and getattr(self, "_bin_edges_", None):
            X = self._prebin_for_predict(X)
        # ─────────────────────────────────────────────────────────────────
        return self._build_test_hup(X)

    def _build_test_hup(self, X_test: Any) -> csr_matrix:
        """Build the sparse binary pattern matrix for test data.

        This follows the single-pass path.
        """
        self._check_health()
        # In original_plus_patterns mode, a fitted model may legitimately have
        # zero mined patterns.  Return an empty pattern matrix and let
        # _make_downstream_features use the original feature block.
        if (
            len(getattr(self, "patterns_", [])) == 0
            and getattr(self, "feature_mode", "patterns_only") != "patterns_only"
        ):
            return csr_matrix((len(X_test), 0), dtype=np.float32)
        # ── Non-finite handling ──────────────────────────────────────────
        if not getattr(self, "adaptive_binning", False):
            X_test, _cat_mask = self._handle_test_nan(X_test)
        else:
            _cat_mask = getattr(self, "cat_cols_mask_", None)
        # ─────────────────────────────────────────────────────────────────
        self._validate_test_input(X_test)
        X_num, X_cat_raw = self._to_float_array(X_test, _cat_mask)
        try:
            X_num = check_array(X_num, dtype=None, ensure_all_finite=False)
        except TypeError:
            X_num = check_array(X_num, dtype=None, force_all_finite=False)

        # Exclude exactly the columns identified as zero-variance during
        # fit(), regardless of what they hold in this batch -- the fitted
        # model never learned anything from them, so prediction ignores them
        # the same way every time rather than re-deciding per call.
        zero_variance_cols = getattr(self, "_zero_variance_cols_", None)
        if zero_variance_cols and _cat_mask is not None:
            self._exclude_zero_variance_columns(X_num, X_cat_raw, _cat_mask, zero_variance_cols)

        n = X_num.shape[0]
        X_cat_arg = X_cat_raw if any(v is not None for v in X_cat_raw) else None

        # Single-pass path. Prefer native CSR output to avoid copying COO
        # row/column arrays into Python and then asking scipy to sort/compress
        # them again. build_test_matrix remains a compatibility fallback.
        n_pats = len(self.patterns_)
        if getattr(self, "_native_available_", True):
            try:
                if hasattr(_core, "build_test_matrix_csr"):
                    indptr, indices = _core.build_test_matrix_csr(
                        X_num,
                        self.td_,
                        X_cat_arg,
                        self.patterns_,
                    )
                    data = np.ones(len(indices), dtype=np.float32)
                    return csr_matrix((data, indices, indptr), shape=(n, n_pats), dtype=np.float32)
                rows, cols = _core.build_test_matrix(
                    X_num,
                    self.td_,
                    X_cat_arg,
                    self.patterns_,
                )
                data = np.ones(len(rows), dtype=np.float32)
                return csr_matrix((data, (rows, cols)), shape=(n, n_pats), dtype=np.float32)
            except Exception:
                logger.debug(
                    "Native build_test_matrix failed; falling back to Python path.",
                    exc_info=True,
                )

        return self._build_test_hup_fallback(X_num, X_cat_raw, n, n_pats)

    def _build_test_hup_fallback(
        self,
        X_num: np.ndarray,
        X_cat_raw: list,
        n: int,
        n_pats: int,
    ) -> csr_matrix:
        """Pure-Python fallback for deserialized models without C++ extension."""
        td = self.td_
        p = X_num.shape[1]

        cpp_bn2id = td._cpp_bn2id
        cpp_stride = td._cpp_bkey_stride
        cpp_all_edges = td._cpp_all_edges
        cpp_nb_col = td._cpp_nb_col
        cpp_col_min = td._cpp_col_min
        cpp_col_range = td._cpp_col_range
        cpp_is_cat = td._cpp_is_cat
        cpp_is_int = td._cpp_is_int
        cpp_is_precoded = getattr(td, "_cpp_is_precoded", [])
        # If the wrapper was deserialized without _cpp_is_precoded, reconstruct it
        # from the classifier's compact adaptive precoded feature set.
        if not cpp_is_precoded:
            precoded_features = getattr(self, "_adaptive_precoded_features_", None)
            if precoded_features is None and getattr(self, "_adaptive_code_label_map_", {}):
                precoded_features = set(getattr(self, "_bin_edges_", {}))
            if precoded_features:
                feat_names = getattr(self, "feature_names_in_", None) or []
                cpp_is_precoded = [name in precoded_features for name in feat_names]
        cpp_cat_cats = td._cpp_cat_categories

        label2code: list[dict[object, int] | None] = [None] * p
        for j in range(p):
            if j < len(cpp_is_cat) and cpp_is_cat[j]:
                if j < len(cpp_cat_cats) and cpp_cat_cats[j]:
                    label2code[j] = {v: i for i, v in enumerate(cpp_cat_cats[j])}

        def bkey(bi: int, j: int) -> int:
            return int(bi * cpp_stride + j)

        test_trans_sets = []
        for r in range(n):
            items: set = set()
            for j in range(p):
                if j < len(cpp_is_cat) and cpp_is_cat[j]:
                    if X_cat_raw[j] is None:
                        continue
                    v = X_cat_raw[j][r]
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    _lc = label2code[j]
                    if _lc is None:
                        continue
                    lc: dict[object, int] = _lc
                    # cpp_cat_cats stores category labels as strings (the
                    # native side stringifies every category value), while
                    # X_cat_raw preserves the original typed value. Match the
                    # native convention here, or every categorical lookup
                    # silently misses and that column is dropped from every
                    # test-time transaction.
                    code = lc.get(str(v))
                    if code is None:
                        continue
                    bi = code + 1
                elif j < len(cpp_is_precoded) and cpp_is_precoded[j]:
                    # Pre-coded column: X_num[r, j] IS the 0-indexed bin code.
                    # bi = code + 1 directly — no scaling, no upper_bound.
                    code_val = X_num[r, j]
                    if not math.isfinite(code_val):
                        continue
                    nb = cpp_nb_col[j] if j < len(cpp_nb_col) else 0
                    bi = max(1, min(int(code_val) + 1, nb))
                else:
                    edges = cpp_all_edges[j]
                    if edges is None or len(edges) < 2:
                        continue
                    nb = cpp_nb_col[j]
                    raw = X_num[r, j]
                    # Non-finite values must generate no item — same contract
                    # as every other column type and the C++ build_test_matrix.
                    if not math.isfinite(raw):
                        continue
                    if j < len(cpp_is_int) and cpp_is_int[j]:
                        val = raw
                    else:
                        val = (raw - cpp_col_min[j]) / cpp_col_range[j]
                    inner = edges[1:-1] if isinstance(edges, np.ndarray) else np.array(edges[1:-1])
                    bi = int(np.searchsorted(inner, val, side="right")) + 1
                    bi = max(1, min(bi, nb))

                bk = bkey(bi, j)
                iid = cpp_bn2id.get(bk)
                if iid is not None:
                    items.add(iid)
            test_trans_sets.append(frozenset(items))

        rows_v, cols_v = [], []
        for pi, pe in enumerate(self.patterns_):
            pat_items = frozenset(pe.items)
            for tid, ts in enumerate(test_trans_sets):
                if pat_items.issubset(ts):
                    rows_v.append(tid)
                    cols_v.append(pi)

        data = np.ones(len(rows_v), dtype=np.float32)
        return csr_matrix((data, (rows_v, cols_v)), shape=(n, n_pats), dtype=np.float32)

    def _check_health(self) -> None:
        check_is_fitted(self)
        if not hasattr(self, "patterns_"):
            raise HUGIMLPredictionError("Pattern state missing — fit() may have failed.")
        if (
            len(self.patterns_) == 0
            and getattr(self, "feature_mode", "patterns_only") == "patterns_only"
        ):
            raise HUGIMLPredictionError("Model has no patterns — fit() may have failed.")
        if not hasattr(self, "model_"):
            raise HUGIMLPredictionError("Downstream model missing — fit() incomplete.")
        if not hasattr(self, "td_") or self.td_ is None:
            raise HUGIMLPredictionError("Transaction data missing — model state corrupt.")

    def _validate_test_input(self, X_test: Any) -> None:
        """Validate test-time input against training schema."""
        from scipy.sparse import issparse as _issparse

        if _issparse(X_test):
            raise ValueError(
                "HUGIMLClassifierNative does not support sparse input.  "
                "Convert to a dense array via X.toarray() first."
            )
        is_df = isinstance(X_test, pd.DataFrame)
        arr = None
        if not is_df:
            arr = np.asarray(X_test)
            if arr.ndim == 1:
                raise ValueError(
                    f"HUGIMLClassifierNative expects a 2D array, got 1D array of shape {arr.shape}."
                )
        n_test_features = (
            len(X_test.columns) if is_df else arr.shape[1]  # type: ignore[union-attr]
        )

        expected = getattr(self, "n_features_in_", None)
        if expected is not None and n_test_features != expected:
            raise HUGIMLSchemaError(
                f"X has {n_test_features} features, but the model was fitted "
                f"with {expected} features."
            )

        expected_names = getattr(self, "feature_names_in_", None)
        if is_df and expected_names is not None:
            test_names = [str(c) for c in X_test.columns]
            if test_names != expected_names:
                missing = set(expected_names) - set(test_names)
                extra = set(test_names) - set(expected_names)
                parts = []
                if missing:
                    parts.append(f"missing: {sorted(missing)}")
                if extra:
                    parts.append(f"unexpected: {sorted(extra)}")
                if not missing and not extra:
                    parts.append("columns in different order")
                raise HUGIMLSchemaError(
                    "Column mismatch between training and test data.  " + "; ".join(parts)
                )

        cat_mask = getattr(self, "cat_cols_mask_", None)
        if is_df and cat_mask is not None and np.any(cat_mask):
            for j, is_cat in enumerate(cat_mask):
                if j >= n_test_features:
                    break
                col = X_test.iloc[:, j]
                if is_cat and pd.api.types.is_numeric_dtype(col):
                    warnings.warn(
                        f"Column '{X_test.columns[j]}' was categorical during "
                        f"training but has numeric dtype ({col.dtype}) in test data.",
                        HUGIMLDtypeDriftWarning,
                        stacklevel=4,
                    )

        if is_df and cat_mask is not None:
            td = self.td_
            cpp_all_edges = getattr(td, "_cpp_all_edges", None)
            if cpp_all_edges is not None:
                try:
                    numeric_idx = [
                        j
                        for j in range(min(n_test_features, len(cat_mask), len(cpp_all_edges)))
                        if not cat_mask[j]
                        and cpp_all_edges[j] is not None
                        and len(cpp_all_edges[j]) >= 2
                    ]
                    if numeric_idx:
                        train_min = np.asarray(
                            [float(cpp_all_edges[j][0]) for j in numeric_idx], dtype=float
                        )
                        train_max = np.asarray(
                            [float(cpp_all_edges[j][-1]) for j in numeric_idx], dtype=float
                        )
                        cpp_col_min = getattr(td, "_cpp_col_min", None)
                        cpp_col_range = getattr(td, "_cpp_col_range", None)
                        if cpp_col_min is not None and cpp_col_range is not None:
                            cm = np.asarray(
                                [float(cpp_col_min[j]) for j in numeric_idx], dtype=float
                            )
                            cr = np.asarray(
                                [float(cpp_col_range[j]) for j in numeric_idx], dtype=float
                            )
                            ok = np.isfinite(cr) & (cr > 0.0) & np.isfinite(cm)
                            train_min[ok] = train_min[ok] * cr[ok] + cm[ok]
                            train_max[ok] = train_max[ok] * cr[ok] + cm[ok]
                        train_span = train_max - train_min
                        valid = train_span > 0
                        if np.any(valid):
                            vals = X_test.iloc[:, numeric_idx].to_numpy(
                                dtype=np.float64, copy=False
                            )
                            finite_vals = np.isfinite(vals)
                            test_min = np.full(len(numeric_idx), np.nan, dtype=float)
                            test_max = np.full(len(numeric_idx), np.nan, dtype=float)
                            observed = finite_vals.any(axis=0)
                            if np.any(observed):
                                observed_vals = np.where(
                                    finite_vals[:, observed], vals[:, observed], np.nan
                                )
                                test_min[observed] = np.nanmin(observed_vals, axis=0)
                                test_max[observed] = np.nanmax(observed_vals, axis=0)
                            drift = (
                                valid
                                & np.isfinite(test_min)
                                & np.isfinite(test_max)
                                & (
                                    (test_min < train_min - train_span * 0.5)
                                    | (test_max > train_max + train_span * 0.5)
                                )
                            )
                            for pos in np.flatnonzero(drift):
                                j = numeric_idx[int(pos)]
                                warnings.warn(
                                    f"Column '{X_test.columns[j]}' has values "
                                    f"[{float(test_min[pos]):.4g}, {float(test_max[pos]):.4g}] outside training "
                                    f"range [{float(train_min[pos]):.4g}, {float(train_max[pos]):.4g}].",
                                    HUGIMLRangeWarning,
                                    stacklevel=4,
                                )
                except Exception:
                    # Preserve prediction behaviour if warning-only drift checks
                    # cannot be vectorized for mixed/object inputs.
                    pass

    # ── Monitoring and drift ──────────────────────────────────────────────────

    def enable_monitoring(self, window_size: int = 1000) -> HUGIMLClassifierNative:
        """Enable prediction monitoring.  Access via ``self.monitor``."""
        self.monitor = PredictionMonitor(window_size=window_size)
        return self

    def disable_monitoring(self) -> HUGIMLClassifierNative:
        """Disable prediction monitoring."""
        self.monitor = None
        return self

    def detect_drift(
        self,
        X_test: Any,
        y_test: np.ndarray | None = None,
        threshold: float = 0.1,
    ) -> str:
        """Run multi-method drift detection and return a human-readable report.

        Uses PSI + KL divergence.  When ``y_test`` is provided, also checks
        label distribution drift.

        Notes
        -----
        Drift metrics are computed on the numeric array retained by the mining
        path.  Constant-B numeric columns that contained NaN/Inf during training are
        converted to the categorical bin-label path so missingness is handled
        consistently at fit/predict time; those columns are therefore not
        represented as continuous numeric drift baselines.  PSI/KL alerts for
        such columns should be interpreted through pattern/feature-importance
        diagnostics rather than through ``detect_drift()``.

        Parameters
        ----------
        X_test : array-like or DataFrame
        y_test : array-like, optional
        threshold : float

        Returns
        -------
        str
        """
        check_is_fitted(self)
        if getattr(self, "_drift_det", None) is None:
            if self._is_production_mode():
                raise RuntimeError(self._audit_artifact_message("Drift-detection baseline"))
            return "Drift detection unavailable (no baseline stored)."
        cat_mask = getattr(self, "cat_cols_mask_", np.zeros(0, dtype=bool))
        X_num, _ = self._to_float_array(X_test, cat_mask)
        y_arr = np.asarray(y_test) if y_test is not None else None
        report = self._drift_det.detect(X_num, y_test=y_arr, threshold=threshold)
        return str(report)

    def get_drift_psi(self, X_test: Any) -> dict:
        """Return per-feature PSI values as a dict.

        See ``detect_drift()`` for the constant-B missing-numeric limitation: columns
        that were routed to categorical bin labels because they contained
        NaN/Inf during training do not have meaningful continuous PSI baselines.
        """
        check_is_fitted(self)
        if getattr(self, "_drift_det", None) is None:
            if self._is_production_mode():
                raise RuntimeError(self._audit_artifact_message("Drift PSI baseline"))
            return {}
        cat_mask = getattr(self, "cat_cols_mask_", np.zeros(0, dtype=bool))
        X_num, _ = self._to_float_array(X_test, cat_mask)
        return self._drift_det.compute_psi(X_num)

    def cross_validate_monitored(
        self,
        X: Any,
        y: Any,
        cv: Any = None,
        scoring: str = "roc_auc",
    ) -> dict:
        """Cross-validation with per-fold monitoring and drift detection.

        Parameters
        ----------
        X : pd.DataFrame or ndarray
        y : array-like
        cv : int or CV splitter (default: StratifiedKFold(5))
        scoring : str

        Returns
        -------
        dict with keys: test_scores, fit_times_ms, fold_monitors, fold_drift, fold_metadata
        """
        from sklearn.metrics import get_scorer
        from sklearn.model_selection import StratifiedKFold

        y = np.asarray(y)
        if cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        elif isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        scorer = get_scorer(scoring)
        results: dict = {
            "test_scores": [],
            "fit_times_ms": [],
            "fold_monitors": [],
            "fold_drift": [],
            "fold_metadata": [],
        }

        # parallelise CV folds.
        # Each fold is fully independent (separate clone, separate fit).
        # prefer="threads" avoids pickling the classifier and its C++ state;
        # the GIL is released during C++ mining so threads still scale.
        # Falls back to serial when n_jobs=1 (the default).
        base_params = {
            k: v for k, v in self.get_params().items() if k not in ("allCols", "origColumns")
        }
        clf_cls = self.__class__

        def _fit_fold(train_idx, test_idx):
            if isinstance(X, pd.DataFrame):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            fold_clf = clf_cls(**base_params)
            t0 = time.perf_counter()
            fold_clf.fit(X_tr, y_tr)
            fit_ms = (time.perf_counter() - t0) * 1000
            score = scorer(fold_clf, X_te, y_te)

            fold_clf.enable_monitoring()
            fold_clf.predict_proba(X_te)
            fold_mon = fold_clf.monitor
            mon_stats = fold_mon.stats if fold_mon is not None else {}

            psi: dict = {}
            if getattr(fold_clf, "_drift_det", None) is not None:
                cat_mask = getattr(fold_clf, "cat_cols_mask_", np.zeros(0, dtype=bool))
                X_te_num, _ = fold_clf._to_float_array(X_te, cat_mask)
                psi = fold_clf._drift_det.compute_psi(X_te_num)

            return score, fit_ms, fold_clf.fit_metadata_, mon_stats, psi

        try:
            from joblib import Parallel
            from joblib import delayed as _delayed

            _n_jobs = self.n_jobs if hasattr(self, "n_jobs") else 1
            fold_outputs = Parallel(n_jobs=_n_jobs, prefer="threads")(
                _delayed(_fit_fold)(tr, te) for tr, te in cv.split(X, y)
            )
        except Exception:
            fold_outputs = [_fit_fold(tr, te) for tr, te in cv.split(X, y)]

        for score, fit_ms, metadata, mon_stats, psi in fold_outputs:
            results["test_scores"].append(score)
            results["fit_times_ms"].append(fit_ms)
            results["fold_metadata"].append(metadata)
            results["fold_monitors"].append(mon_stats)
            results["fold_drift"].append(psi)

        return results

    # ── Explanation methods ───────────────────────────────────────────────────

    def get_hug_features(self) -> list[str]:
        """Return a human-readable label for each mined HUG pattern.

        Singleton patterns use the format ``feature=[lo,hi)`` for adaptive
        numerical columns (e.g. ``age=[35,50)``) and ``feature=value`` for
        categorical columns (e.g. ``gender=F``).  Compound patterns (L > 1)
        are comma-separated, e.g. ``age=[35,50), gender=F``.

        When ``adaptive_binning=True`` and the integer-code path was used,
        C++ stores bin labels as ``feature=[k,k+1]`` (integer range).  These
        are transparently remapped to the original-scale ``[lo,hi)`` labels
        via ``_adaptive_code_label_map_`` so that the output is identical in
        appearance to the string-path output.

        Production mode
        ---------------
        This method remains available in ``execution_mode='production'`` because
        it only needs retained pattern labels.  ``get_pattern_info()`` is
        intentionally audit-only because it additionally needs the retained
        training pattern matrix to compute support.

        Returns
        -------
        list of str
        """
        check_is_fitted(self)
        item_map = self.td_.item_map
        # _adaptive_code_label_map_ is set by _apply_adaptive_binning when the
        # integer-code path is used.  Empty dict (default) → no remapping needed.
        label_remap = getattr(self, "_adaptive_code_label_map_", {})

        def _resolve_item(item_id: int) -> str:
            raw_label = item_map.get(item_id, str(item_id))
            return label_remap.get(raw_label, raw_label)

        return [", ".join(_resolve_item(it) for it in pe.items) for pe in self.patterns_]

    @staticmethod
    def _feature_name_from_item_label(label: str) -> str:
        """Return the source feature name from a rendered transaction item."""
        text = str(label or "")
        if "=" not in text:
            return text.strip()
        return text.split("=", 1)[0].strip()

    def _survivor_audit_lookup(self) -> dict[str, dict[str, object]]:
        """Return interaction-relaxed survivor metadata keyed by feature name."""
        if not bool(getattr(self, "interaction_relaxed_mining", False)):
            return {}
        rows = list(getattr(self, "interaction_relaxed_mining_survivors_", []) or [])
        lookup: dict[str, dict[str, object]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            lookup[name] = dict(row)
        return lookup

    def _pattern_source_features(self, pattern_entry: Any) -> list[str]:
        """Return source feature names used by a native pattern entry."""
        item_map = getattr(getattr(self, "td_", None), "item_map", {}) or {}
        label_remap = getattr(self, "_adaptive_code_label_map_", {}) or {}
        names: list[str] = []
        seen: set[str] = set()
        for item_id in getattr(pattern_entry, "items", []) or []:
            raw_label = item_map.get(int(item_id), str(item_id))
            label = label_remap.get(raw_label, raw_label)
            feature_name = self._feature_name_from_item_label(str(label))
            if feature_name and feature_name not in seen:
                seen.add(feature_name)
                names.append(feature_name)
        return names

    @staticmethod
    def _finite_float_or_nan(value: Any) -> float:
        """Convert a scalar to float, using NaN for missing/non-finite values."""
        try:
            out = float(value)
        except Exception:
            return float("nan")
        return out if np.isfinite(out) else float("nan")

    def _pattern_survivor_audit(self, pattern_entry: Any) -> dict[str, object]:
        """Return survivor-led provenance for one mined pattern."""
        survivor_lookup = self._survivor_audit_lookup()
        source_features = self._pattern_source_features(pattern_entry)
        survivor_features = [name for name in source_features if name in survivor_lookup]
        survivor_rows = [survivor_lookup[name] for name in survivor_features]
        marginal_igs = [
            self._finite_float_or_nan(row.get("marginal_ig", np.nan)) for row in survivor_rows
        ]
        interaction_scores = [
            self._finite_float_or_nan(row.get("interaction_score", row.get("score", np.nan)))
            for row in survivor_rows
        ]
        best_partners: list[str] = []
        for row in survivor_rows:
            partner = str(row.get("best_partner", "")).strip()
            if partner and partner not in best_partners:
                best_partners.append(partner)
        finite_marginal = [v for v in marginal_igs if np.isfinite(v)]
        finite_interaction = [v for v in interaction_scores if np.isfinite(v)]
        survivor_led = bool(survivor_features)
        return {
            "pattern_origin": "interaction_relaxed" if survivor_led else "standard",
            "survivor_led": survivor_led,
            "survivor_features": survivor_features,
            "survivor_feature_count": len(survivor_features),
            "survivor_min_marginal_ig": min(finite_marginal) if finite_marginal else np.nan,
            "survivor_max_interaction_score": max(finite_interaction)
            if finite_interaction
            else np.nan,
            "survivor_best_partners": best_partners,
        }

    def _pattern_audit_lookup_by_label(self) -> dict[str, dict[str, object]]:
        """Return survivor-led metadata keyed by raw and namespaced labels."""
        labels = self.get_hug_features()
        lookup: dict[str, dict[str, object]] = {}
        for label, pe in zip(labels, getattr(self, "patterns_", []) or []):
            audit = self._pattern_survivor_audit(pe)
            lookup[label] = audit
            lookup[f"pattern:{label}"] = audit
        return lookup

    def _pattern_origin_counts_for_downstream_features(
        self, features: list[str] | None = None
    ) -> dict[str, int]:
        """Return origin counts for downstream pattern features."""
        names = list(features or self._get_downstream_feature_names())
        lookup = self._pattern_audit_lookup_by_label()
        counts = {"standard": 0, "interaction_relaxed": 0, "survivor_led": 0}
        for feat in names:
            if self._downstream_feature_type(str(feat)) != "pattern":
                continue
            display_name = self._downstream_feature_display_name(str(feat))
            audit = lookup.get(str(feat), lookup.get(display_name, {}))
            origin = str(audit.get("pattern_origin", "standard"))
            if origin not in counts:
                counts[origin] = 0
            counts[origin] += 1
            if bool(audit.get("survivor_led", False)):
                counts["survivor_led"] += 1
        counts["total_pattern_features"] = sum(
            1 for feat in names if self._downstream_feature_type(str(feat)) == "pattern"
        )
        return counts

    def get_transformed_shape(self) -> tuple[int, int]:
        """Return (n_samples, n_patterns) for the training pattern matrix.

        In production mode the matrix itself is not retained, but its shape is
        persisted as lightweight diagnostic metadata.
        """
        check_is_fitted(self)
        if hasattr(self, "x_train_hup_"):
            shape = self.x_train_hup_.shape
            return int(shape[0]), int(shape[1])
        cached = getattr(self, "_training_pattern_matrix_shape_", None)
        if cached is not None:
            return int(cached[0]), int(cached[1])
        raise RuntimeError(self._audit_artifact_message("Training pattern matrix shape"))

    def get_pattern_info(self) -> pd.DataFrame:
        """Summary DataFrame with one row per mined HUG pattern.

        Columns include pattern, utility, information_gain, support, and
        survivor-led provenance when interaction-relaxed mining is enabled.

        This is an audit/governance table.  Unlike ``get_hug_features()``, it
        requires the retained training pattern matrix to compute support and
        therefore raises a clear error in ``execution_mode='production'``.
        """
        check_is_fitted(self)
        self._require_audit_artifact("Pattern support and pattern-info audit table", "x_train_hup_")
        n_train = self.x_train_hup_.shape[0]
        features = self.get_hug_features()
        records: list[dict[str, object]] = []
        for i, pe in enumerate(self.patterns_):
            support = float(self.x_train_hup_[:, i].sum()) / n_train
            audit = self._pattern_survivor_audit(pe)
            records.append(
                {
                    "pattern": features[i],
                    "utility": round(pe.utility, 6),
                    "information_gain": round(pe.ig, 6),
                    "support": round(support, 4),
                    **audit,
                }
            )
        return pd.DataFrame(records)

    def _downstream_feature_display_name(self, name: str) -> str:
        """Return a compact display label for a downstream feature name."""
        for prefix in ("orig:", "pattern:", "augmented_pair:"):
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    def _downstream_feature_type(self, name: str) -> str:
        """Classify a downstream feature name by its explicit namespace."""
        if name.startswith("orig:"):
            return "original"
        if name.startswith("augmented_pair:"):
            return "augmented_pair"
        return "pattern"

    def _pattern_support_lookup(self) -> dict[str, float]:
        """Return training support by both raw and namespaced pattern label.

        Production-mode models intentionally drop the training HUG matrix.
        ``feature_importances()`` should prefer cached support metadata when it
        exists; this lookup is only a best-effort recomputation path for audit
        models or legacy objects that still retain ``x_train_hup_``.
        """
        if not hasattr(self, "x_train_hup_"):
            return {}
        n_train = int(self.x_train_hup_.shape[0])
        if n_train <= 0:
            return {}
        labels = self.get_hug_features()
        support = np.asarray(self.x_train_hup_.sum(axis=0)).ravel() / max(n_train, 1)
        lookup: dict[str, float] = {}
        for label, value in zip(labels, support):
            val = float(value)
            lookup[label] = val
            lookup[f"pattern:{label}"] = val
        return lookup

    def get_downstream_features(self) -> list[str]:
        """Return names aligned with the downstream estimator input columns.

        The returned names include a namespace prefix so feature provenance is
        explicit: ``orig:`` for original features, ``pattern:`` for mined HUG
        patterns, and ``augmented_pair:`` for augmented pair transforms.  When
        ``topk_budget_strict=True``, the returned list is already filtered to
        the columns retained by the fitted strict TopK mask.
        """
        check_is_fitted(self)
        return list(self._get_downstream_feature_names())

    def _downstream_feature_counts(self) -> dict[str, int]:
        """Return counts by downstream feature family for the fitted estimator."""
        names = list(
            getattr(self, "_downstream_feature_names_", []) or self._get_downstream_feature_names()
        )
        counts = {
            "original": sum(1 for name in names if str(name).startswith("orig:")),
            "pattern": sum(1 for name in names if str(name).startswith("pattern:")),
            "augmented_pair": sum(1 for name in names if str(name).startswith("augmented_pair:")),
        }
        counts["total"] = len(names)
        return counts

    def get_model_composition(self) -> dict[str, Any]:
        """Return downstream feature composition and relevant fitted configuration.

        The composition describes the actual feature families entering the
        downstream estimator after feature-mode construction and optional
        strict TopK filtering.
        """
        check_is_fitted(self)
        counts = self._downstream_feature_counts()
        aug_config = dict(getattr(self, "augmented_pair_config_", {}) or {})
        return {
            "feature_mode": getattr(self, "feature_mode", "patterns_only"),
            "topK": getattr(self, "topK", None),
            "topk_budget_strict": bool(getattr(self, "topk_budget_strict", False)),
            "augmented_pair_transforms_enabled": bool(
                getattr(self, "augmented_pair_transforms", False)
            ),
            "augmented_pair_config": aug_config,
            "n_input_features": int(getattr(self, "n_features_in_", 0)),
            "n_patterns_mined": int(len(getattr(self, "patterns_", []))),
            "n_downstream_features": counts["total"],
            "downstream_feature_counts": counts,
            "pattern_origin_counts": self._pattern_origin_counts_for_downstream_features(),
        }

    def _cache_downstream_feature_metadata(self) -> None:
        """Cache metadata aligned with the fitted downstream feature matrix.

        This method must run before production retention because production
        mode drops the training matrices.  It is intentionally best-effort:
        pattern support can be unavailable for cached/tuned candidates, but
        non-missing-rate and variance can still be computed from the fitted
        downstream matrix before it is discarded.
        """
        features = self._get_downstream_feature_names()
        self._downstream_feature_names_ = list(features)
        n_features = len(features)

        self._downstream_pattern_support_ = np.full(n_features, np.nan, dtype=np.float64)
        self._downstream_pattern_origin_ = np.full(n_features, "", dtype=object)
        self._downstream_survivor_led_ = np.zeros(n_features, dtype=np.bool_)
        self._downstream_survivor_features_ = [list() for _ in range(n_features)]
        self._downstream_survivor_feature_count_ = np.zeros(n_features, dtype=np.int64)
        self._downstream_survivor_min_marginal_ig_ = np.full(n_features, np.nan, dtype=np.float64)
        self._downstream_survivor_max_interaction_score_ = np.full(
            n_features, np.nan, dtype=np.float64
        )
        self._downstream_survivor_best_partners_ = [list() for _ in range(n_features)]
        try:
            support_lookup = self._pattern_support_lookup()
        except Exception:
            support_lookup = {}
        try:
            audit_lookup = self._pattern_audit_lookup_by_label()
        except Exception:
            audit_lookup = {}
        for idx, feat in enumerate(features):
            if self._downstream_feature_type(feat) == "pattern":
                display_name = self._downstream_feature_display_name(feat)
                self._downstream_pattern_support_[idx] = support_lookup.get(
                    feat, support_lookup.get(display_name, np.nan)
                )
                audit = audit_lookup.get(feat, audit_lookup.get(display_name, {}))
                self._downstream_pattern_origin_[idx] = str(audit.get("pattern_origin", "standard"))
                self._downstream_survivor_led_[idx] = bool(audit.get("survivor_led", False))
                survivor_features = list(audit.get("survivor_features", []) or [])
                self._downstream_survivor_features_[idx] = survivor_features
                self._downstream_survivor_feature_count_[idx] = int(
                    audit.get("survivor_feature_count", len(survivor_features))
                )
                self._downstream_survivor_min_marginal_ig_[idx] = self._finite_float_or_nan(
                    audit.get("survivor_min_marginal_ig", np.nan)
                )
                self._downstream_survivor_max_interaction_score_[idx] = self._finite_float_or_nan(
                    audit.get("survivor_max_interaction_score", np.nan)
                )
                self._downstream_survivor_best_partners_[idx] = list(
                    audit.get("survivor_best_partners", []) or []
                )

        X_meta = getattr(self, "x_train_downstream_", None)
        if X_meta is not None and n_features == getattr(X_meta, "shape", (0, 0))[1]:
            X_arr = X_meta.toarray() if issparse(X_meta) else np.asarray(X_meta)
            finite_mask = np.isfinite(X_arr)
            self._downstream_non_missing_rate_ = finite_mask.mean(axis=0).astype(np.float64)
            self._downstream_variance_ = np.nanvar(
                np.where(finite_mask, X_arr, np.nan), axis=0
            ).astype(np.float64)
        else:
            self._downstream_non_missing_rate_ = np.full(n_features, np.nan, dtype=np.float64)
            self._downstream_variance_ = np.full(n_features, np.nan, dtype=np.float64)

    def feature_importances(self) -> pd.DataFrame:
        """Map downstream estimator coefficients to final feature names.

        Returns a DataFrame sorted by absolute coefficient magnitude.  Feature
        names are aligned to the downstream estimator after feature-mode and
        optional strict TopK filtering have been applied.  The ``feature_type``
        column distinguishes original features, mined HUG patterns, and
        augmented pair transforms.  ``pattern_support`` is populated only for
        mined HUG patterns; original and augmented-pair features use
        ``support_type='not_applicable'`` and ``pattern_support=NaN``.

        Raises
        ------
        AttributeError
            When the downstream estimator does not expose ``coef_``
            (e.g. non-linear models).
        """
        check_is_fitted(self)
        production_without_training_artifacts = self._is_production_mode() and not hasattr(
            self, "x_train_downstream_"
        )
        audit_note = (
            self._audit_artifact_message("Training matrices and drift-baseline audit artifacts")
            if production_without_training_artifacts
            else ""
        )
        if production_without_training_artifacts:
            warnings.warn(
                audit_note,
                HUGIMLWarning,
                stacklevel=2,
            )
        clf_step = self.model_.named_steps.get("clf")
        if not hasattr(clf_step, "coef_"):
            raise AttributeError(
                "feature_importances requires the downstream estimator "
                "to expose coef_ (e.g. LogisticRegression)."
            )

        raw_coef = clf_step.coef_
        coef = (
            raw_coef.mean(axis=0)
            if raw_coef.ndim == 2 and raw_coef.shape[0] > 1
            else raw_coef.ravel()
        )
        features = self._get_downstream_feature_names()
        if len(features) != len(coef):
            raise RuntimeError(
                "Downstream feature names are not aligned with estimator coefficients: "
                f"{len(features)} names for {len(coef)} coefficients."
            )

        cached_pattern_support = getattr(self, "_downstream_pattern_support_", None)
        if cached_pattern_support is not None and len(cached_pattern_support) != len(features):
            cached_pattern_support = None
        support_lookup = self._pattern_support_lookup() if cached_pattern_support is None else {}
        cached_pattern_origin = getattr(self, "_downstream_pattern_origin_", None)
        cached_survivor_led = getattr(self, "_downstream_survivor_led_", None)
        cached_survivor_features = getattr(self, "_downstream_survivor_features_", None)
        cached_survivor_feature_count = getattr(self, "_downstream_survivor_feature_count_", None)
        cached_survivor_min_marginal_ig = getattr(
            self, "_downstream_survivor_min_marginal_ig_", None
        )
        cached_survivor_max_interaction_score = getattr(
            self, "_downstream_survivor_max_interaction_score_", None
        )
        cached_survivor_best_partners = getattr(self, "_downstream_survivor_best_partners_", None)
        has_cached_survivor_audit = all(
            value is not None and len(value) == len(features)
            for value in (
                cached_pattern_origin,
                cached_survivor_led,
                cached_survivor_features,
                cached_survivor_feature_count,
                cached_survivor_min_marginal_ig,
                cached_survivor_max_interaction_score,
                cached_survivor_best_partners,
            )
        )
        pattern_audit_lookup = (
            {} if has_cached_survivor_audit else self._pattern_audit_lookup_by_label()
        )
        strict_scores = getattr(self, "_strict_topk_feature_scores_", None)
        strict_score_lookup: dict[str, float] = {}
        if strict_scores is not None:
            full_names = getattr(self, "_downstream_feature_names_full_", None)
            if full_names is None or len(full_names) == 0:
                full_names = self._get_downstream_feature_names_full()
            strict_score_lookup = {
                name: float(score)
                for name, score in zip(full_names, np.asarray(strict_scores).ravel())
            }

        aug_lookup = {
            f"augmented_pair:{item.get('name')}": item
            for item in getattr(self, "augmented_pair_transforms_", [])
        }

        non_missing_rates = getattr(self, "_downstream_non_missing_rate_", None)
        variances = getattr(self, "_downstream_variance_", None)
        if (
            non_missing_rates is None
            or variances is None
            or len(non_missing_rates) != len(features)
            or len(variances) != len(features)
        ):
            X_meta = getattr(self, "x_train_downstream_", None)
            if X_meta is not None and len(features) == getattr(X_meta, "shape", (0, 0))[1]:
                X_arr = X_meta.toarray() if issparse(X_meta) else np.asarray(X_meta)
                finite_mask = np.isfinite(X_arr)
                non_missing_rates = finite_mask.mean(axis=0)
                variances = np.nanvar(np.where(finite_mask, X_arr, np.nan), axis=0)
            else:
                non_missing_rates = np.full(len(features), np.nan)
                variances = np.full(len(features), np.nan)

        rows: list[dict[str, object]] = []
        for idx, (feat, c) in enumerate(zip(features, coef)):
            feature_type = self._downstream_feature_type(feat)
            display_name = self._downstream_feature_display_name(feat)
            if feature_type == "pattern":
                if cached_pattern_support is not None:
                    pattern_support = float(cached_pattern_support[idx])
                else:
                    pattern_support = support_lookup.get(
                        feat, support_lookup.get(display_name, np.nan)
                    )
                support_type = "pattern_support"
                if has_cached_survivor_audit:
                    pattern_origin = str(cached_pattern_origin[idx]) or "standard"
                    survivor_led = bool(cached_survivor_led[idx])
                    survivor_features = list(cached_survivor_features[idx] or [])
                    survivor_feature_count = int(cached_survivor_feature_count[idx])
                    survivor_min_marginal_ig = self._finite_float_or_nan(
                        cached_survivor_min_marginal_ig[idx]
                    )
                    survivor_max_interaction_score = self._finite_float_or_nan(
                        cached_survivor_max_interaction_score[idx]
                    )
                    survivor_best_partners = list(cached_survivor_best_partners[idx] or [])
                else:
                    audit = pattern_audit_lookup.get(
                        feat, pattern_audit_lookup.get(display_name, {})
                    )
                    pattern_origin = str(audit.get("pattern_origin", "standard"))
                    survivor_led = bool(audit.get("survivor_led", False))
                    survivor_features = list(audit.get("survivor_features", []) or [])
                    survivor_feature_count = int(
                        audit.get("survivor_feature_count", len(survivor_features))
                    )
                    survivor_min_marginal_ig = self._finite_float_or_nan(
                        audit.get("survivor_min_marginal_ig", np.nan)
                    )
                    survivor_max_interaction_score = self._finite_float_or_nan(
                        audit.get("survivor_max_interaction_score", np.nan)
                    )
                    survivor_best_partners = list(audit.get("survivor_best_partners", []) or [])
            else:
                pattern_support = np.nan
                support_type = "not_applicable"
                pattern_origin = "not_applicable"
                survivor_led = False
                survivor_features = []
                survivor_feature_count = 0
                survivor_min_marginal_ig = np.nan
                survivor_max_interaction_score = np.nan
                survivor_best_partners = []

            support_value = (
                round(float(pattern_support), 4) if np.isfinite(pattern_support) else np.nan
            )
            aug_meta = aug_lookup.get(feat, {}) if feature_type == "augmented_pair" else {}
            std_mean = aug_meta.get("standardization_mean", np.nan)
            std_scale = aug_meta.get("standardization_scale", np.nan)
            std_scale_float = float(std_scale) if np.isfinite(std_scale) else np.nan
            coef_raw = (
                float(c) / std_scale_float
                if feature_type == "augmented_pair"
                and np.isfinite(std_scale_float)
                and std_scale_float != 0.0
                else np.nan
            )
            raw_formula = aug_meta.get("raw_formula", np.nan)
            if feature_type == "augmented_pair":
                aug_text = self._augmented_pair_effect_text(
                    raw_formula=str(raw_formula),
                    operation=str(aug_meta.get("operation", "")),
                    coefficient_raw_scale=coef_raw,
                    standardization_mean=float(std_mean) if np.isfinite(std_mean) else np.nan,
                    standardization_scale=std_scale_float,
                    source_observed_medians=aug_meta.get("source_observed_medians", np.nan),
                    pair_missing_policy=str(
                        aug_meta.get("pair_missing_policy", "reference_value_for_unavailable_pair")
                    ),
                    eligible_rate=float(aug_meta.get("eligible_rate", np.nan)),
                    missing_pair_rate=float(aug_meta.get("missing_pair_rate", np.nan)),
                )
                decision_direction = aug_text["decision_direction"]
                risk_increases_when = aug_text["risk_increases_when"]
                unit_effect_interpretation = aug_text["unit_effect_interpretation"]
                reference_raw_value_description = aug_text["reference_raw_value_description"]
                source_observed_medians_description = aug_text[
                    "source_observed_medians_description"
                ]
                pair_missing_policy_description = aug_text["pair_missing_policy_description"]
                raw_scale_note = aug_text["raw_scale_note"]
                raw_interpretation = aug_text["raw_interpretation"]
            else:
                decision_direction = np.nan
                risk_increases_when = np.nan
                unit_effect_interpretation = np.nan
                reference_raw_value_description = np.nan
                source_observed_medians_description = np.nan
                pair_missing_policy_description = np.nan
                raw_scale_note = np.nan
                raw_interpretation = np.nan
            rows.append(
                {
                    "pattern": display_name,
                    "feature": feat,
                    "display_name": display_name,
                    "feature_type": feature_type,
                    "pattern_origin": pattern_origin,
                    "survivor_led": survivor_led,
                    "survivor_features": survivor_features,
                    "survivor_feature_count": survivor_feature_count,
                    "survivor_min_marginal_ig": survivor_min_marginal_ig,
                    "survivor_max_interaction_score": survivor_max_interaction_score,
                    "survivor_best_partners": survivor_best_partners,
                    "coefficient": round(float(c), 6),
                    "abs_coefficient": round(abs(float(c)), 6),
                    "pattern_support": support_value,
                    "support": support_value,
                    "support_type": support_type,
                    "non_missing_rate": round(float(non_missing_rates[idx]), 6),
                    "variance": round(float(variances[idx]), 6),
                    "strict_topk_score": round(float(strict_score_lookup.get(feat, np.nan)), 6),
                    "standardization_mean": std_mean,
                    "standardization_scale": std_scale,
                    "operation": aug_meta.get("operation", np.nan),
                    "inputs": aug_meta.get("inputs", np.nan),
                    "raw_formula": raw_formula,
                    "standardized_formula": aug_meta.get("standardized_formula", np.nan),
                    "pair_missing_policy": aug_meta.get("pair_missing_policy", np.nan),
                    "eligible_count": aug_meta.get("eligible_count", np.nan),
                    "eligible_rate": aug_meta.get("eligible_rate", np.nan),
                    "missing_pair_rate": aug_meta.get("missing_pair_rate", np.nan),
                    "source_observed_medians": aug_meta.get("source_observed_medians", np.nan),
                    "transform_ig": aug_meta.get("transform_ig", np.nan),
                    "coefficient_standardized": round(float(c), 6)
                    if feature_type == "augmented_pair"
                    else np.nan,
                    "one_std_effect_on_log_odds": round(float(c), 6)
                    if feature_type == "augmented_pair"
                    else np.nan,
                    "coefficient_raw_scale": round(float(coef_raw), 12)
                    if np.isfinite(coef_raw)
                    else np.nan,
                    "one_raw_unit_effect_on_log_odds": round(float(coef_raw), 12)
                    if np.isfinite(coef_raw)
                    else np.nan,
                    "reference_raw_value": std_mean if feature_type == "augmented_pair" else np.nan,
                    "reference_raw_value_description": reference_raw_value_description,
                    "decision_direction": decision_direction,
                    "risk_increases_when": risk_increases_when,
                    "unit_effect_interpretation": unit_effect_interpretation,
                    "raw_scale_note": raw_scale_note,
                    "raw_interpretation": raw_interpretation,
                    "source_observed_medians_description": source_observed_medians_description,
                    "pair_missing_policy_description": pair_missing_policy_description,
                    "audit_note": (audit_note if production_without_training_artifacts else ""),
                }
            )
        result = (
            pd.DataFrame(rows)
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )
        if production_without_training_artifacts:
            result.attrs["audit_note"] = audit_note
        return pd.DataFrame(result)

    # ── Adaptive-binning diagnostic plots ────────────────────────────────────
    # These methods are available on any fitted HUGIMLClassifierNative instance
    # when adaptive_binning=True.  HUGIMLAdaptive inherits them automatically
    # as a subclass.  Both require matplotlib (optional dependency).

    def plot_bin_profiles(self, figsize: tuple | None = None):
        """Bar chart of the chosen B per numerical feature (adaptive binning only).

        Colour encodes position in the candidate range:
        blue = coarse end, green = mid, amber/red = fine end.

        Returns
        -------
        (fig, ax)

        Raises
        ------
        RuntimeError
            When called on a non-adaptive or unfitted model.
        ImportError
            When matplotlib is not installed.
        """
        self._check_adaptive_fitted("plot_bin_profiles")
        self._require_mpl()
        import matplotlib.pyplot as plt

        feats = list(self.per_feature_b_.keys())
        bvals = [self.per_feature_b_[f] for f in feats]
        cands = self.b_candidates or [2, 15]
        lo, hi = min(cands), max(cands)

        colors = [
            "#2166ac"
            if b <= lo + (hi - lo) / 3
            else "#1a9641"
            if b <= lo + 2 * (hi - lo) / 3
            else "#d7191c"
            for b in bvals
        ]

        fig, ax = plt.subplots(figsize=figsize or (max(7, len(feats) * 0.5 + 2), 4))
        ax.bar(range(len(feats)), bvals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Chosen B_j", fontsize=10)
        ax.set_title(
            f"Adaptive binning — chosen B per feature  "
            f"(threshold={self.min_marginal_gain_ratio:.0%})",
            fontsize=11,
        )
        for i, b in enumerate(bvals):
            ax.text(i, b + 0.05, str(b), ha="center", fontsize=8)
        fig.tight_layout()
        return fig, ax

    def ig_heatmap(self, figsize: tuple | None = None):
        """Heatmap of IG score at every (feature, B) grid point (adaptive binning only).

        The chosen B per feature is highlighted with a bounding box.

        Returns
        -------
        (fig, ax)

        Raises
        ------
        RuntimeError
            When called on a non-adaptive or unfitted model, or when
            ``ig_scores_`` is empty.
        ImportError
            When matplotlib is not installed.
        """
        self._check_adaptive_fitted("ig_heatmap")
        if not getattr(self, "ig_scores_", None):
            raise RuntimeError("ig_scores_ is empty — call fit() first.")
        self._require_mpl()
        import matplotlib.pyplot as plt

        feats = sorted(self.ig_scores_)
        bs = sorted({b for sc in self.ig_scores_.values() for b in sc})
        grid = np.array([[self.ig_scores_[f].get(b, 0.0) for b in bs] for f in feats])

        fig, ax = plt.subplots(
            figsize=figsize or (max(6, len(bs) * 0.9), max(4, len(feats) * 0.45))
        )
        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(bs)))
        ax.set_xticklabels([str(b) for b in bs], fontsize=9)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=8)
        ax.set_xlabel("B candidates", fontsize=10)
        ax.set_title("IG score per (feature, B)  — box = chosen B", fontsize=11)
        for i, f in enumerate(feats):
            chosen = self.per_feature_b_.get(f)
            if chosen and chosen in bs:
                j = bs.index(chosen)
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2,
                    )
                )
        plt.colorbar(im, ax=ax, label="Information gain")
        fig.tight_layout()
        return fig, ax

    def _check_adaptive_fitted(self, method_name: str) -> None:
        """Raise a clear error when an adaptive-only method is called incorrectly."""
        check_is_fitted(self)
        if not getattr(self, "adaptive_binning", False):
            raise RuntimeError(
                f"{method_name}() is only available when adaptive_binning=True.  "
                f"Re-fit with HUGIMLClassifier(adaptive_binning=True, ...) "
                f"or use HUGIMLAdaptive."
            )
        if not getattr(self, "per_feature_b_", None):
            raise RuntimeError(f"{method_name}() requires per_feature_b_ — call fit() first.")

    @staticmethod
    def _require_mpl() -> None:
        """Raise ImportError when matplotlib is not installed."""
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise ImportError(
                "matplotlib is required for diagnostic plots. "
                "Install with:  pip install matplotlib  "
                "or:  pip install 'hugiml-core[plots]'"
            )

    # ── End adaptive-binning diagnostic plots ────────────────────────────────

    def _summary_shape_text(self, matrix_attr: str, cached_shape_attr: str) -> str:
        """Return a stable summary shape for audit or production-retained models."""
        matrix = getattr(self, matrix_attr, None)
        if matrix is not None and hasattr(matrix, "shape"):
            return str(tuple(int(v) for v in matrix.shape))
        cached = getattr(self, cached_shape_attr, None)
        if cached is not None:
            return (
                f"{tuple(int(v) for v in cached)} (training matrix not retained in production mode)"
            )
        if self._is_production_mode():
            return "not retained in production mode"
        return "unavailable"

    def model_summary(self) -> str:
        """Human-readable model summary including top patterns."""
        check_is_fitted(self)
        composition = self.get_model_composition()
        counts = composition.get("downstream_feature_counts", {})
        lines = [
            "HUGIMLClassifier — Model Summary",
            "=" * 50,
            f"Config:       B={self.B}, L={self.L}, G={self.G}",
            f"Feature mode: {getattr(self, 'feature_mode', 'patterns_only')}",
            f"Training:     {self.fit_metadata_.n_samples} samples, "
            f"{self.fit_metadata_.n_features} features, "
            f"{self.fit_metadata_.n_classes} classes",
            f"Patterns:     {self.fit_metadata_.n_patterns} "
            f"({self.fit_metadata_.n_compound} compound)",
            f"Augmented pairs: {counts.get('augmented_pair', 0)} retained",
            f"Downstream composition: original={counts.get('original', 0)}, "
            f"patterns={counts.get('pattern', 0)}, "
            f"augmented_pair={counts.get('augmented_pair', 0)}, "
            f"total={counts.get('total', 0)}",
            f"Matrix:       {self._summary_shape_text('x_train_hup_', '_training_pattern_matrix_shape_')} "
            f"(density={self.fit_metadata_.matrix_density:.4f})",
            f"Downstream:   {self._summary_shape_text('x_train_downstream_', '_training_downstream_matrix_shape_')}",
            f"Fit time:     {self.fit_metadata_.total_fit_ms:.0f} ms",
            "",
            "Stage breakdown (ms):",
        ]
        for stage, ms in self.fit_metadata_.stage_times_ms.items():
            lines.append(f"  {stage:<25} {ms:>8.1f}")
        try:
            imp = self.feature_importances().head(10)
            has_non_pattern = bool((imp.get("feature_type", "pattern") != "pattern").any())
            has_augmented = bool((imp.get("feature_type", "pattern") == "augmented_pair").any())
            section = (
                "Top 10 downstream features by importance:"
                if has_non_pattern
                else "Top 10 patterns by importance:"
            )
            lines += ["", section]
            if has_augmented:
                lines.append(
                    "  (includes augmented pair transforms; use "
                    "explain_augmented_pair_effects() for raw-scale interpretation)"
                )
            for _, row in imp.iterrows():
                support_text = (
                    f"pattern_support={row['pattern_support']:.3f}"
                    if row.get("support_type") == "pattern_support"
                    else "pattern_support=n/a"
                )
                lines.append(
                    f"  [{row.get('feature_type', 'pattern')}] "
                    f"{row['pattern']:<40} "
                    f"coef={row['coefficient']:>+8.4f}  "
                    f"{support_text}"
                )
        except AttributeError:
            lines += ["", "Top downstream features by importance:"]
            lines.append("  (not available — non-LR downstream estimator)")

        # ── Adaptive binning section ────────────────────────────────────
        if getattr(self, "_missing_col_edges_", None):
            lines += [
                "",
                f"NaN handling: {len(self._missing_col_edges_)} numerical column(s) "
                f"pre-binned (NaN/Inf generates no transaction item at train or test time).",
            ]
        if self.adaptive_binning and getattr(self, "per_feature_b_", None):
            lines += ["", "Adaptive binning — chosen B per feature:"]
            for feat, b in sorted(self.per_feature_b_.items(), key=lambda kv: -kv[1]):
                edges = self._bin_edges_.get(feat, [])
                rng = f"  [{float(edges[0]):.4g}…{float(edges[-1]):.4g}]" if len(edges) >= 2 else ""
                lines.append(f"  {feat:<35} B={b:<3}{rng}")
        # ─────────────────────────────────────────────────────────────────

        return "\n".join(lines)


# =============================================================================
# Exact cached grid tuning helper
# =============================================================================


def _hugiml_auc_score_for_fast_grid(y_true: Any, proba: np.ndarray, classes: np.ndarray) -> float:
    """Internal validation AUC scorer used by fast_grid_tune()."""
    from sklearn.metrics import roc_auc_score

    y_arr = np.asarray(y_true)
    if proba.ndim != 2:
        raise HUGIMLValidationError("predict_proba must return a 2D array.")
    if proba.shape[1] == 2:
        return float(roc_auc_score(y_arr, proba[:, 1]))
    return float(roc_auc_score(y_arr, proba, multi_class="ovr", average="macro"))


def _hugiml_expand_grid_for_fast_tune(
    param_grid: dict[str, list] | str | None,
) -> list[dict[str, Any]]:
    """Expand a compact sklearn-style parameter grid for fast HUGIML tuning.

    ``param_grid`` may be a dict of parameter lists (the sklearn-style grid
    itself), the name of one of the grids in ``hugiml.hyperparameter_configs``
    (for example ``"interpretability"``), or ``None`` to use the default
    named grid.
    """
    from itertools import product

    if param_grid is None or isinstance(param_grid, str):
        grid = get_hugiml_grid(param_grid)
    else:
        grid = param_grid
    if not isinstance(grid, dict) or not grid:
        raise HUGIMLParamError("param_grid must be a non-empty dict of parameter lists.")
    keys = list(grid.keys())
    values = []
    for key in keys:
        val = grid[key]
        if isinstance(val, (str, bytes)) or not hasattr(val, "__iter__"):
            val = [val]
        val = list(val)
        if not val:
            raise HUGIMLParamError(f"param_grid[{key!r}] must contain at least one value.")
        values.append(val)
    return [dict(zip(keys, vals)) for vals in product(*values)]


def _hugiml_validate_fast_tune_grid(candidates: list[dict[str, Any]]) -> dict[str, list]:
    """Validate that a grid is safe for exact cached tuning.

    The fast path is exact when adaptive binning is enabled and only mining/
    representation dimensions vary: G, L, topK, and feature_mode.  Because G is
    part of the native mining call, candidates are cached in separate constant-G
    groups.  B may appear and even vary, but is ignored while
    adaptive_binning=True because per-feature binning supplies the effective
    discretisation.
    """
    if not candidates:
        raise HUGIMLParamError("No grid candidates supplied.")

    varying = {
        key
        for key in set().union(*(set(c.keys()) for c in candidates))
        if len({repr(c.get(key, None)) for c in candidates}) > 1
    }
    allowed_varying = {"B", "G", "L", "topK", "feature_mode"}
    disallowed = sorted(varying - allowed_varying)
    if disallowed:
        raise HUGIMLParamError(
            "fast_grid_tune requires only G, L, topK, and feature_mode to vary "
            f"(B is ignored under adaptive_binning=True). Varying unsupported keys: {disallowed}."
        )

    adaptive_values = {bool(c.get("adaptive_binning", True)) for c in candidates}
    if adaptive_values != {True}:
        raise HUGIMLParamError("fast_grid_tune requires adaptive_binning=True for every candidate.")

    g_values = sorted({float(c.get("G", 1e-2)) for c in candidates})
    L_values = sorted({int(c.get("L", 1)) for c in candidates})
    topk_values = sorted({int(c.get("topK", 30)) for c in candidates})
    feature_modes = sorted({str(c.get("feature_mode", "patterns_only")) for c in candidates})
    if any(k <= 0 for k in topk_values):
        raise HUGIMLParamError(
            "fast_grid_tune currently supports positive integer topK values only."
        )
    if any(L < 1 for L in L_values):
        raise HUGIMLParamError("fast_grid_tune currently supports L >= 1 only.")

    allowed_modes = {"patterns_only", "original_plus_patterns", "original_plus_interactions"}
    bad_modes = sorted(set(feature_modes) - allowed_modes)
    if bad_modes:
        raise HUGIMLParamError(f"Unsupported feature_mode values for fast_grid_tune: {bad_modes}.")

    return {
        "L_values": L_values,
        "topK_values": topk_values,
        "feature_modes": feature_modes,
        "G_values": g_values,
        # Backward-compatible key used by older callers/tests.
        "G": g_values,
    }


def _hugiml_shallow_candidate_from_base(base: HUGIMLClassifierNative) -> HUGIMLClassifierNative:
    """Create a candidate that shares immutable cached mining artefacts with base."""
    cand = base.__class__(**base.get_params(deep=False))
    share_attrs = [
        "cat_cols_mask_",
        "is_int_mask_",
        "feature_names_in_",
        "_bin_edges_",
        "_missing_col_edges_",
        "_adaptive_code_label_map_",
        "_adaptive_precoded_features_",
        "per_feature_b_",
        "ig_scores_",
        "td_",
        "raw_patterns_",
        "classes_",
        "n_features_in_",
        "_native_available_",
    ]
    for attr in share_attrs:
        if hasattr(base, attr):
            setattr(cand, attr, getattr(base, attr))
    return cand


def _hugiml_prepare_candidate_from_cached_base(
    base: HUGIMLClassifierNative,
    X_train_original: Any,
    y_train: Any,
    L_value: int,
    topK_value: int,
    feature_mode: str,
    execution_mode: str = "audit",
) -> HUGIMLClassifierNative:
    """Build and fit one exact candidate from a max-topK cached base model."""
    cand = _hugiml_shallow_candidate_from_base(base)
    cand.L = int(L_value)
    cand.topK = int(topK_value)
    cand.feature_mode = str(feature_mode)
    cand.execution_mode = str(execution_mode)

    raw_patterns = list(getattr(base, "raw_patterns_", []))[: int(topK_value)]
    n_train = len(y_train)
    base_hup = getattr(base, "x_train_hup_", None)
    if base_hup is None:
        raise RuntimeError(
            "fast_grid_tune requires cached training pattern matrices. "
            "The cache model was created in production mode or otherwise does not retain "
            "x_train_hup_; run tuning with execution_mode='audit'."
        )
    if int(L_value) == 1:
        cand.patterns_ = raw_patterns
        # Fused L=1 path returns columns in raw_patterns_ order; slicing is exact.
        cand.x_train_hup_ = base_hup[:, : len(cand.patterns_)]
    else:
        native_td = getattr(getattr(base, "td_", None), "_td", getattr(base, "td_", None))
        old_td = getattr(cand, "td_", None)
        cand.td_ = native_td
        cand.patterns_, cached_coo = cand._deduplicate_patterns_by_coverage(raw_patterns, n_train)
        cand.td_ = old_td
        if cached_coo is not None:
            rows, cols = cached_coo
        elif len(cand.patterns_) > 0:
            rows, cols = _core.build_train_matrix(native_td, cand.patterns_)
        else:
            rows = cols = np.zeros(0, dtype=np.int32)
        data = np.ones(len(rows), dtype=np.float32)
        cand.x_train_hup_ = csr_matrix(
            (data, (rows, cols)), shape=(n_train, len(cand.patterns_)), dtype=np.float32
        )

    if len(cand.patterns_) == 0 and cand.feature_mode == "patterns_only":
        raise HUGIMLMiningError(
            "No HUG patterns found for cached candidate. Try reducing G, increasing topK, "
            "or using original_plus_patterns."
        )

    cand._setup_feature_mode_metadata()
    cand._setup_augmented_pair_transforms(X_train_original, y_train, fit=True)
    cand._current_y_for_downstream_topk_ = y_train
    try:
        X_down = cand._make_downstream_features(X_train_original, cand.x_train_hup_, fit=True)
    finally:
        if hasattr(cand, "_current_y_for_downstream_topk_"):
            delattr(cand, "_current_y_for_downstream_topk_")
    X_down = cand._apply_strict_topk_budget_fit(X_down, y_train)
    cand.x_train_downstream_ = X_down
    cand._cache_downstream_feature_metadata()
    cand.model_ = Pipeline([("clf", cand._make_estimator(len(cand.classes_)))])
    cand.model_.fit(X_down, y_train)
    cand._apply_execution_mode_retention()
    # Intentionally avoid drift baseline and rich metadata during tuning. The
    # returned best_model is immediately usable for prediction; call fit() on the
    # selected params if full production metadata/drift baseline is required.
    return cand


def _hugiml_fast_grid_tune(
    cls,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    param_grid: dict[str, list] | str | None = None,
    *,
    base_params: dict[str, Any] | None = None,
    scoring: str = "roc_auc",
    refit_full: bool = False,
    return_results: bool = True,
) -> dict[str, Any]:
    """Exact cached tuner for the compact adaptive HUGIML grid.

    Requirements
    ------------
    - adaptive_binning=True for every candidate.
    - G may vary; the tuner partitions candidates into constant-G cache groups.
    - Only G, L, topK, and feature_mode vary. B may appear in the grid but is
      ignored for cache partitioning because adaptive binning chooses per-feature
      bins and fit() passes sentinel B=2 to the native transaction builder.
    - max_fit_seconds must be None to guarantee equivalence to the ordinary grid
      loop; timeout/degradation can make cached mining fits differ from
      standalone candidates.

    ``base_params['execution_mode']`` defaults to ``'production'`` when not
    supplied, since a tuning sweep evaluates many candidates and the
    line-level allocation diagnostics that ``'audit'`` provides are rarely
    useful for any individual one of them; pass
    ``base_params={'execution_mode': 'audit', ...}`` to opt back in. This
    only affects line-level tracing cost and the post-fit retention of
    training matrices -- it does not change which patterns are mined,
    scores, or rankings, all of which are identical either way (the
    per-(G, L, topK) cache fits that do the actual mining are tagged
    'audit' internally regardless, so that the candidates derived from
    them keep access to the cached pattern matrix; see the cache-building
    loop below).

    With ``refit_full=False`` (the default), ``best_model`` is one of these
    lightweight per-candidate objects: drift-baseline and rich metadata were
    never computed for it regardless of execution_mode (see below), and
    under the 'production' default its training pattern matrix is not
    retained either, so get_pattern_info() also raises on it -- call this
    function again with ``refit_full=True``, or call fit() directly with the
    selected params, for a model meant to be inspected afterward rather than
    just used for prediction.

    With ``refit_full=True``, the refit that produces the returned
    ``best_model`` defaults to ``execution_mode='audit'`` regardless of what
    the search candidates used, unless the caller's own ``base_params``
    explicitly named an ``execution_mode`` -- the search benefits from
    'production' for speed across many candidates, but the one model
    actually handed back for the caller to keep and inspect defaults to
    having full access to get_pattern_info(), detect_drift(),
    get_drift_psi(), and feature_importances() without missing-artifact
    warnings, rather than inheriting the search's speed-oriented default.

    Returns a dict with best_model, best_params, best_score, cv_results, and
    cache timings. Uses the same scorer as the ordinary grid path for all
    supported scoring values. During tuning it skips drift-baseline and rich final
    metadata; set refit_full=True to refit the selected model with normal fit().
    """
    t_start = time.perf_counter()
    candidates = _hugiml_expand_grid_for_fast_tune(param_grid)
    grid_info = _hugiml_validate_fast_tune_grid(candidates)
    params0 = dict(base_params or {})
    params0.setdefault("adaptive_binning", True)
    params0.setdefault("use_hotpath", True)
    # Recorded before the setdefault below so refit_full can tell "the caller
    # asked for production" apart from "this default applies to every search
    # candidate but a refit_full model should still get full audit access by
    # default" -- see the refit_full block further down.
    _caller_set_execution_mode = "execution_mode" in params0
    params0.setdefault("execution_mode", "production")
    # Do not set a single global G here; G is part of mining and is constant per
    # cache group below.  A caller-supplied base G is used only for candidates
    # that omit G from the grid.
    params0.setdefault("G", grid_info["G_values"][0])
    if params0.get("max_fit_seconds", None) is not None:
        raise HUGIMLParamError(
            "fast_grid_tune requires max_fit_seconds=None for exact equivalence."
        )

    requested_execution_mode = str(params0.get("execution_mode", "production"))
    if requested_execution_mode not in {"audit", "production"}:
        raise HUGIMLParamError(
            "execution_mode must be either 'audit' or 'production'. "
            f"Got {requested_execution_mode!r}."
        )
    y_train_arr = cls._safe_cast_y(y_train)
    y_val_arr = np.asarray(y_val)
    X_train_original = cls(**params0)._copy_input_for_downstream(X_train)

    # Correctness note: topK is NOT derived by mining max(topK) once and slicing.
    # Empirically, the native miner can return additional valid patterns when a
    # larger topK is requested, so a smaller standalone topK run is not always
    # equivalent to a prefix of the larger run.  To guarantee identical validation
    # scores to the ordinary grid loop, cache one mining fit per (G, L, topK)
    # group and reuse that cache only across feature_mode candidates.  Within
    # each cache fit, fit() already sorts raw_patterns_ by descending utility with
    # tuple(items) tie-breaking before downstream construction.
    base_by_G_L_topK: dict[tuple[float, int, int], HUGIMLClassifierNative] = {}
    cache_fit_seconds: dict[str, float] = {}
    needed_cache_keys = sorted(
        {
            (
                float(c.get("G", params0.get("G", 1e-2))),
                int(c.get("L", 1)),
                int(c.get("topK", 30)),
            )
            for c in candidates
        }
    )
    for G_value, L_value, topK_value in needed_cache_keys:
        base_fit_params = dict(params0)
        base_fit_params.update(
            {
                "adaptive_binning": True,
                "L": int(L_value),
                "topK": int(topK_value),
                # Use the richest ordinary mode so raw input is preserved and
                # empty-pattern fallbacks do not fail while building the cache.
                "feature_mode": "original_plus_patterns",
                "G": float(G_value),
                # Cached tuning needs training matrices.  Even if callers pass
                # production in base_params, the internal cache must retain audit
                # artifacts; final refit below can still use caller params.
                "execution_mode": "audit",
            }
        )
        # B may be present in the original grid, but it is intentionally ignored
        # under adaptive_binning=True.
        t_fit = time.perf_counter()
        base = cls(**base_fit_params)
        base._fast_tune_cache_only = True
        base.fit(X_train, y_train_arr)
        base.__dict__.pop("_fast_tune_cache_only", None)
        cache_fit_seconds[f"G={float(G_value):.12g},L={int(L_value)},topK={int(topK_value)}"] = (
            time.perf_counter() - t_fit
        )
        base_by_G_L_topK[(float(G_value), int(L_value), int(topK_value))] = base

    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_model: HUGIMLClassifierNative | None = None
    best_params: dict[str, Any] | None = None

    for candidate_params in candidates:
        L_value = int(candidate_params.get("L", 1))
        topK_value = int(candidate_params.get("topK", 30))
        feature_mode = str(candidate_params.get("feature_mode", "patterns_only"))
        G_value = float(candidate_params.get("G", params0.get("G", 1e-2)))
        t_cand = time.perf_counter()
        status = "ok"
        err = None
        score = np.nan
        model = None
        try:
            model = _hugiml_prepare_candidate_from_cached_base(
                base_by_G_L_topK[(G_value, L_value, topK_value)],
                X_train_original,
                y_train_arr,
                L_value,
                topK_value,
                feature_mode,
                requested_execution_mode,
            )
            score = _hugiml_score_model_for_tune(model, X_val, y_val_arr, scoring)
            if np.isfinite(score) and score > best_score:
                best_score = float(score)
                best_model = model
                best_params = dict(candidate_params)
                best_params["adaptive_binning"] = True
                best_params["G"] = G_value
                best_params.setdefault("execution_mode", requested_execution_mode)
        except (
            Exception
        ) as exc:  # keep failed candidates visible like GridSearchCV error_score=np.nan
            status = "failed"
            err = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "params": dict(candidate_params),
                "L": L_value,
                "topK": topK_value,
                "feature_mode": feature_mode,
                "G": G_value,
                "mean_test_score": score,
                "status": status,
                "error": err,
                "elapsed_seconds": time.perf_counter() - t_cand,
            }
        )

    if best_model is None or best_params is None:
        raise HUGIMLValidationError("All fast_grid_tune candidates failed.")

    if refit_full:
        refit_params = dict(params0)
        refit_params.update(best_params)
        # The search above runs every candidate under 'production' by default
        # for speed; refit_full's purpose is producing one normal, fully
        # inspectable model from the winning configuration, so unless the
        # caller explicitly asked for 'production', this refit uses 'audit'
        # so get_pattern_info(), detect_drift(), and similar audit-mode-only
        # methods work on the model handed back here without the caller
        # needing to know to ask for that separately.
        if not _caller_set_execution_mode:
            refit_params["execution_mode"] = "audit"
        # Keep user-supplied B if present; adaptive_binning ignores it for transaction B.
        best_model = cls(**refit_params).fit(X_train, y_train_arr)

    result = {
        "best_model": best_model,
        "best_params": best_params,
        "best_score": float(best_score),
        "cv_results": rows if return_results else None,
        "cache_fit_seconds_by_G_L_topK": cache_fit_seconds,
        "cache_topK_strategy": "exact_per_G_L_topK_utility_ordered",
        "elapsed_seconds": time.perf_counter() - t_start,
        "method": "exact_cached_adaptive_grid",
        "scoring": str(scoring),
    }
    return result


@dataclasses.dataclass
class HUGIMLTuneResult:
    """Result object returned by HUGIMLClassifier.tune().

    Attributes mirror the small subset of GridSearchCV-style fields users need
    for quick HUGIML tuning while keeping the API lightweight.
    """

    best_estimator_: HUGIMLClassifierNative
    best_params_: dict[str, Any]
    best_score_: float
    results_: Any
    fast_path_used_: bool
    elapsed_seconds_: float
    n_splits_: int
    scoring: str
    cv_splits_: list[tuple[np.ndarray, np.ndarray]]
    shuffle: bool
    random_state: int | None

    # Backward-compatible aliases for dict-style code in notebooks.
    @property
    def best_model(self) -> HUGIMLClassifierNative:
        return self.best_estimator_

    @property
    def best_params(self) -> dict[str, Any]:
        return self.best_params_

    @property
    def best_score(self) -> float:
        return self.best_score_

    @property
    def cv_results(self) -> Any:
        return self.results_

    @property
    def cv_splits(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self.cv_splits_


def _hugiml_params_key(params: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Stable hashable key for parameter dictionaries used in tuning results."""
    return tuple(sorted((str(k), repr(v)) for k, v in dict(params).items()))


def _hugiml_score_model_for_tune(
    model: HUGIMLClassifierNative,
    X_val: Any,
    y_val: Any,
    scoring: str,
) -> float:
    """Score one fitted model for HUGIMLClassifier.tune()."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    scoring_norm = str(scoring).lower()
    if scoring_norm in {"roc_auc", "auc"}:
        proba = model.predict_proba(X_val)
        return _hugiml_auc_score_for_fast_grid(y_val, proba, model.classes_)
    pred = model.predict(X_val)
    if scoring_norm == "accuracy":
        return float(accuracy_score(y_val, pred))
    if scoring_norm == "balanced_accuracy":
        return float(balanced_accuracy_score(y_val, pred))
    if scoring_norm in {"f1", "f1_binary"}:
        return float(f1_score(y_val, pred))
    if scoring_norm == "f1_macro":
        return float(f1_score(y_val, pred, average="macro"))
    if scoring_norm == "f1_weighted":
        return float(f1_score(y_val, pred, average="weighted"))
    raise HUGIMLParamError(
        "Unsupported scoring value. Supported: 'roc_auc', 'accuracy', "
        "'balanced_accuracy', 'f1', 'f1_macro', 'f1_weighted'."
    )


def _hugiml_standard_grid_tune_one_split(
    cls,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    candidates: list[dict[str, Any]],
    base_params: dict[str, Any],
    scoring: str,
) -> dict[str, Any]:
    """Ordinary per-candidate grid evaluation for grids not eligible for fast path."""
    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_model: HUGIMLClassifierNative | None = None
    best_params: dict[str, Any] | None = None
    y_train_arr = cls._safe_cast_y(y_train)
    for candidate_params in candidates:
        t_cand = time.perf_counter()
        params = dict(base_params)
        params.update(candidate_params)
        status = "ok"
        err = None
        score = np.nan
        model = None
        try:
            model = cls(**params).fit(X_train, y_train_arr)
            score = _hugiml_score_model_for_tune(model, X_val, y_val, scoring)
            if np.isfinite(score) and score > best_score:
                best_score = float(score)
                best_model = model
                best_params = dict(candidate_params)
        except Exception as exc:
            status = "failed"
            err = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "params": dict(candidate_params),
                "L": candidate_params.get("L", params.get("L")),
                "topK": candidate_params.get("topK", params.get("topK")),
                "feature_mode": candidate_params.get("feature_mode", params.get("feature_mode")),
                "mean_test_score": score,
                "status": status,
                "error": err,
                "elapsed_seconds": time.perf_counter() - t_cand,
            }
        )
    if best_model is None or best_params is None:
        raise HUGIMLValidationError("All tune candidates failed on a validation split.")
    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_score": float(best_score),
        "cv_results": rows,
        "elapsed_seconds": sum(float(r["elapsed_seconds"]) for r in rows),
        "method": "ordinary_grid",
    }


def _hugiml_tune(
    cls,
    X: Any,
    y: Any,
    *,
    cv: int | Any = 5,
    scoring: str = "roc_auc",
    param_grid: dict[str, list] | str | None = None,
    refit: bool = True,
    base_params: dict[str, Any] | None = None,
    random_state: int | None = 42,
    shuffle: bool = True,
    cv_splits: list[tuple[Any, Any]] | None = None,
    use_fast_path: bool = True,
    return_dataframe: bool = True,
) -> HUGIMLTuneResult:
    """Tune HUGIML on full X, y using stratified CV and optional fast-grid caching.

    This is the main public convenience API for quick HUGIML model selection.
    The regular constructor remains a single-configuration estimator; this
    method owns grid search, cross-validation, aggregation, and optional refit.

    Parameters
    ----------
    X, y : array-like or DataFrame/Series
        Full training data.
    cv : int or splitter, default=5
        Number of stratified folds, or any sklearn-compatible splitter with
        split(X, y). Integer cv uses StratifiedKFold.
    scoring : {'roc_auc', 'accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_weighted'}
        Validation metric. 'roc_auc' supports binary and multiclass OVR macro AUC.
    param_grid : dict or None
        A dict (sklearn-style grid), the name of a grid registered in
        hugiml.hyperparameter_configs (for example 'interpretability'),
        or None to use HUGIMLClassifier.default_param_grid().
    refit : bool, default=True
        If True, refit the best configuration on the full X, y with normal fit().
    base_params : dict or None
        Constructor parameters shared by every candidate. ``execution_mode``
        defaults to ``'production'`` for the candidates evaluated during the
        search, since per-candidate line-level allocation diagnostics are
        rarely useful mid-sweep; mining results, scores, and the resulting
        ranking are identical to ``'audit'`` either way. ``result.best_estimator_``
        -- the one model actually returned -- is refit under
        ``execution_mode='audit'`` regardless, so get_pattern_info(),
        detect_drift(), get_drift_psi(), and feature_importances() all work
        on it without missing-artifact warnings, unless ``base_params``
        explicitly names an ``execution_mode``, in which case that value is
        used everywhere (search and final refit alike) and the above
        defaulting is skipped entirely.
    random_state : int or None, default=42
        Random seed for StratifiedKFold when cv is an integer.
    shuffle : bool, default=True
        Whether StratifiedKFold shuffles before splitting.
    cv_splits : list of (train_idx, val_idx) or None, default=None
        Exact fold indices to use. When supplied, cv, shuffle, and random_state
        are ignored for split generation, and the same indices are returned in
        ``result.cv_splits_`` for reuse by other models.
    use_fast_path : bool, default=True
        Use exact cached fast-grid evaluation when the grid qualifies; otherwise
        fall back to ordinary per-candidate evaluation.
    return_dataframe : bool, default=True
        Return ``results_`` as a pandas DataFrame when pandas is available.

    Returns
    -------
    HUGIMLTuneResult
        GridSearchCV-like result object with ``best_estimator_``, ``best_params_``, ``best_score_``, ``results_``,
        ``fast_path_used_``, ``elapsed_seconds_``, and ``n_splits_``.
    """
    from sklearn.model_selection import StratifiedKFold

    t_start = time.perf_counter()
    base_params0 = dict(base_params or {})
    # Recorded before the setdefault below for the same reason as in
    # fast_grid_tune: the final refit further down restores 'audit' by
    # default even though every candidate during the search uses
    # 'production', unless the caller explicitly chose an execution_mode.
    _caller_set_execution_mode = "execution_mode" in base_params0
    base_params0.setdefault("execution_mode", "production")
    candidates = _hugiml_expand_grid_for_fast_tune(param_grid)

    y_arr = cls._safe_cast_y(y)
    n_samples = len(y_arr)
    if cv_splits is not None:
        splits = []
        for split_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
            tr = np.asarray(train_idx, dtype=np.int64)
            va = np.asarray(val_idx, dtype=np.int64)
            if tr.ndim != 1 or va.ndim != 1:
                raise HUGIMLParamError(
                    "Each cv_splits entry must contain 1D train and validation indices."
                )
            if tr.size == 0 or va.size == 0:
                raise HUGIMLParamError(
                    f"cv_splits entry {split_idx} has an empty train or validation index."
                )
            if (
                np.any(tr < 0)
                or np.any(va < 0)
                or np.any(tr >= n_samples)
                or np.any(va >= n_samples)
            ):
                raise HUGIMLParamError(
                    f"cv_splits entry {split_idx} contains indices outside [0, n_samples)."
                )
            if np.intersect1d(tr, va).size > 0:
                raise HUGIMLParamError(
                    f"cv_splits entry {split_idx} has overlapping train and validation indices."
                )
            splits.append((tr.copy(), va.copy()))
    elif isinstance(cv, int):
        if cv < 2:
            raise HUGIMLParamError("cv must be >= 2 when provided as an integer.")
        splitter = StratifiedKFold(
            n_splits=int(cv), shuffle=bool(shuffle), random_state=random_state
        )
        splits = [
            (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
            for tr, va in splitter.split(X, y_arr)
        ]
    else:
        splits = [
            (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
            for tr, va in cv.split(X, y_arr)
        ]
    if not splits:
        raise HUGIMLParamError("cv produced no splits.")

    def _take_rows(obj: Any, idx: np.ndarray) -> Any:
        if hasattr(obj, "iloc"):
            return obj.iloc[idx]
        return np.asarray(obj)[idx]

    fast_path_allowed = False
    if use_fast_path:
        try:
            _hugiml_validate_fast_tune_grid(candidates)
            # The cached fast path is exact only for adaptive-binning grids with
            # no fit-time timeout/degradation. Candidate grids that explicitly
            # set adaptive_binning=False are rejected above; this additional
            # guard covers the common case where adaptive_binning or
            # max_fit_seconds is supplied only through base_params.
            if (
                bool(base_params0.get("adaptive_binning", True))
                and base_params0.get("max_fit_seconds", None) is None
            ):
                fast_path_allowed = True
        except Exception:
            fast_path_allowed = False

    fold_rows: list[dict[str, Any]] = []
    fold_methods: list[str] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train = _take_rows(X, train_idx)
        X_val = _take_rows(X, val_idx)
        y_train = y_arr[train_idx]
        y_val = y_arr[val_idx]
        if fast_path_allowed:
            try:
                split_result = cls.fast_grid_tune(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    param_grid=param_grid,
                    base_params=base_params0,
                    scoring=scoring,
                    refit_full=False,
                    return_results=True,
                )
            except Exception:
                # Preserve correctness over speed: an unexpected cached-path
                # failure for one fold should fall back to the ordinary
                # per-candidate evaluation rather than aborting tuning.
                fast_path_allowed = False
                split_result = _hugiml_standard_grid_tune_one_split(
                    cls,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    candidates,
                    base_params0,
                    scoring,
                )
        else:
            split_result = _hugiml_standard_grid_tune_one_split(
                cls,
                X_train,
                y_train,
                X_val,
                y_val,
                candidates,
                base_params0,
                scoring,
            )
        fold_methods.append(str(split_result.get("method", "unknown")))
        for row in split_result.get("cv_results") or []:
            params = dict(row.get("params", {}))
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "params_key": _hugiml_params_key(params),
                    "params": params,
                    "L": row.get("L", params.get("L")),
                    "topK": row.get("topK", params.get("topK")),
                    "feature_mode": row.get("feature_mode", params.get("feature_mode")),
                    "split_test_score": row.get("mean_test_score", np.nan),
                    "status": row.get("status", "ok"),
                    "error": row.get("error"),
                    "elapsed_seconds": row.get("elapsed_seconds", np.nan),
                }
            )

    if not fold_rows:
        raise HUGIMLValidationError("No tuning results were produced.")

    grouped: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}
    for row in fold_rows:
        grouped.setdefault(row["params_key"], []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows_for_key in grouped.items():
        scores = np.asarray([float(r["split_test_score"]) for r in rows_for_key], dtype=float)
        finite = scores[np.isfinite(scores)]
        first_params = dict(rows_for_key[0]["params"])
        summary_rows.append(
            {
                "params": first_params,
                "L": first_params.get("L"),
                "topK": first_params.get("topK"),
                "feature_mode": first_params.get("feature_mode"),
                "mean_test_score": float(np.mean(finite)) if finite.size else np.nan,
                "std_test_score": float(np.std(finite, ddof=0)) if finite.size else np.nan,
                "n_successful_splits": int(finite.size),
                "n_splits": int(len(splits)),
                "mean_elapsed_seconds": float(
                    np.nanmean([r["elapsed_seconds"] for r in rows_for_key])
                ),
                "status": "ok" if finite.size == len(splits) else "partial_or_failed",
            }
        )
    summary_rows.sort(
        key=lambda r: (
            -float(r["mean_test_score"]) if np.isfinite(r["mean_test_score"]) else np.inf,
            repr(r["params"]),
        )
    )
    if not summary_rows or not np.isfinite(summary_rows[0]["mean_test_score"]):
        raise HUGIMLValidationError("All tune candidates failed across CV splits.")
    for rank, row in enumerate(summary_rows, start=1):
        row["rank_test_score"] = rank

    best_params = dict(base_params0)
    best_params.update(dict(summary_rows[0]["params"]))
    best_score = float(summary_rows[0]["mean_test_score"])
    # The search above evaluates every candidate under 'production' by
    # default for speed; best_estimator_ is the model the caller actually
    # keeps and is the natural target for get_pattern_info(), detect_drift(),
    # and similar audit-mode-only methods afterward, so unless the caller's
    # own base_params explicitly named an execution_mode, it's refit under
    # 'audit' here rather than inheriting the search's speed-oriented
    # default.
    if not _caller_set_execution_mode:
        best_params["execution_mode"] = "audit"
    if refit:
        best_estimator = cls(**best_params).fit(X, y_arr)
    else:
        # Return a fitted estimator from the first fold for convenience.  It is
        # valid for immediate inspection/prediction on that fold's fitted state,
        # but refit=True is recommended for production use.
        train_idx, val_idx = splits[0]
        best_estimator = cls(**best_params).fit(_take_rows(X, train_idx), y_arr[train_idx])

    if return_dataframe:
        try:
            results_obj = pd.DataFrame(summary_rows)
        except Exception:
            results_obj = summary_rows
    else:
        results_obj = summary_rows

    return HUGIMLTuneResult(
        best_estimator_=best_estimator,
        best_params_=best_params,
        best_score_=best_score,
        results_=results_obj,
        fast_path_used_=bool(
            fast_path_allowed and all(m == "exact_cached_adaptive_grid" for m in fold_methods)
        ),
        elapsed_seconds_=time.perf_counter() - t_start,
        n_splits_=int(len(splits)),
        scoring=str(scoring),
        cv_splits_=[(tr.copy(), va.copy()) for tr, va in splits],
        shuffle=bool(shuffle),
        random_state=random_state,
    )


HUGIMLClassifierNative.fast_grid_tune = classmethod(_hugiml_fast_grid_tune)
HUGIMLClassifierNative.tune = classmethod(_hugiml_tune)


# Backward-compatible public class name. HUGIMLClassifierNative remains
# available for existing code; HUGIMLClassifier is the cleaner end-user entry
# point and intentionally shares the exact implementation and serialization
# contract.
HUGIMLClassifier = HUGIMLClassifierNative

__all__ = [
    "HUGIMLClassifier",
    "HUGIMLClassifierNative",
    "FitMetadata",
    "HUGIMLTuneResult",
]

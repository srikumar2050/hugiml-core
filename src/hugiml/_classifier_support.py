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

"""Shared classifier data structures and numerical helpers."""

from __future__ import annotations

import dataclasses
import logging
import threading
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from hugiml._binning import _quantile_edges as _adap_quantile_edges
from hugiml._classifier_runtime import _CORE_AVAILABLE, _core
from hugiml.exceptions import HUGIMLParamError

HUGIMLClassifier = Any


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
        native_cache: Any | None = None,
        score_topK: int | None = None,
    ) -> None:
        self.augmented_pair_mode = str(augmented_pair_mode)
        self.aug_feature_size = int(aug_feature_size)
        self.max_pair_features = int(max_pair_features)
        self.ii_partner_size = None if ii_partner_size is None else int(ii_partner_size)
        self.budget_topK = None if budget_topK is None else int(budget_topK)
        self.min_source_ig = None if min_source_ig is None else float(min_source_ig)
        self.unbounded_cap = int(unbounded_cap)
        self.native_cache = native_cache
        self.score_topK = None if score_topK is None else int(score_topK)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Fit-time only acceleration objects must never become part of pickle
        # or versioned model state. The fitted transform catalog, indices,
        # reference values, and scaler arrays are sufficient for prediction,
        # explanations, and save/load reconstruction.
        state.pop("native_cache", None)
        state.pop("_fit_Z_cache_", None)
        state.pop("_consume_fit_cache_", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.native_cache = None

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
        y_codes = np.asarray(pd.factorize(np.asarray(y), sort=True)[0], dtype=np.int64)

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
            cache = getattr(self, "native_cache", None)
            if cache is not None and hasattr(cache, "select_interaction_information_features"):
                selected = cache.select_interaction_information_features(
                    X_all_numeric,
                    y_codes,
                    list(numeric_cols),
                    int(self.aug_feature_size),
                    self.ii_partner_size,
                )
            else:
                selected = _core.select_interaction_information_features(
                    X_all_numeric,
                    y_codes,
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
            self.native_cache = None
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

        score_top_k = (
            -1
            if self.budget_topK is None
            else max(0, int(max(int(self.budget_topK), int(self.score_topK or 0))))
        )
        cache = getattr(self, "native_cache", None)
        if cache is not None and hasattr(cache, "score_pair_candidates"):
            native_specs, candidate_count = cache.score_pair_candidates(
                X_selected,
                y_codes,
                list(self.selected_aug_features_),
                int(score_top_k),
            )
        elif hasattr(_core, "score_pair_candidates_bounded"):
            native_specs, candidate_count = _core.score_pair_candidates_bounded(
                X_selected,
                y_codes,
                list(self.selected_aug_features_),
                int(score_top_k),
            )
        else:
            native_specs = _core.score_pair_candidates(
                X_selected,
                y_codes,
                list(self.selected_aug_features_),
            )
            candidate_count = len(native_specs)
        self.augmented_pair_native_used_ = True
        candidates = list(native_specs)
        candidates.sort(key=lambda item: (-float(item["transform_ig"]), str(item["name"])))
        self.candidate_count_ = int(candidate_count)
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
            Z_train = _core.transform_pair_features(
                X_selected,
                left,
                right,
                ops,
                pair_refs,
                self.scaler_mean_,
                self.scaler_scale_,
            )
            self._fit_Z_cache_ = _dense_full_csr(np.asarray(Z_train, dtype=np.float32))
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
        self.native_cache = None
        return self

    def transform(self, X: Any) -> csr_matrix:
        n_rows = len(X) if hasattr(X, "__len__") else 0
        if not getattr(self, "kept_specs_", []):
            return csr_matrix((n_rows, 0), dtype=np.float32)
        if hasattr(self, "_fit_Z_cache_") and getattr(self, "_consume_fit_cache_", False):
            Z = self._fit_Z_cache_
            delattr(self, "_fit_Z_cache_")
            self._consume_fit_cache_ = False
            return Z
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


logger = logging.getLogger("hugiml.classifier")


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


_NUMERIC_DTYPE_KINDS = "fiub"  # float, signed int, unsigned int, bool


def _is_zero_variance_numeric_column(col: np.ndarray) -> bool:
    """Return True when a numeric column has at most one distinct value.

    NaN is the only value treated as missing (matching pandas'
    ``nunique(dropna=True)``); Inf is a real value and counts toward
    distinctness. A single ``np.unique`` call handles both jobs at once:
    it sorts NaNs to a single trailing entry for floating dtypes (this is
    guaranteed numpy behaviour, not incidental), so there is no separate
    NaN-masking pass over the column before counting distinct values.
    """
    u = np.unique(col)
    if u.size and np.issubdtype(u.dtype, np.floating) and np.isnan(u[-1]):
        u = u[:-1]
    return u.size <= 1


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
        True when fit completed in a degraded or fallback state.
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

    def __init__(self, td_native: Any, classifier: HUGIMLClassifier) -> None:
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
# HUGIMLClassifier
# =============================================================================


def _wire_hugiml_feature_metadata(
    estimator: Any,
    feature_names: list[str],
    augmented_catalog: list[dict[str, Any]],
    pattern_provenance: dict[str, dict[str, Any]] | None = None,
    original_feature_standardization: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Best-effort: if `estimator` (or, for a meta-estimator wrapper
    exposing `.estimator` -- e.g. sklearn's OneVsRestClassifier -- the
    estimator it wraps) knows how to consume HUGIML's own downstream
    feature names / augmented-pair catalog / pattern source mapping, wire it up
    before fitting.

    `base_estimator=` is a plain sklearn fit(X, y)/predict_proba(X)
    hand-off: X arrives as an anonymous numpy/sparse matrix with no column
    names attached, so a downstream estimator whose behavior depends on
    knowing which columns are original features versus mined augmented
    pairs versus mined patterns (duck-typed here via a
    `set_hugiml_feature_metadata` method -- e.g. an RPTE-style
    bounded-lookahead feature extractor) has no way to find that out on its
    own without this wiring.

    Two mechanisms are applied, both when available, because a single
    `set_hugiml_feature_metadata()` call is not enough once a meta-estimator
    is involved: sklearn.base.clone() (used internally by
    OneVsRestClassifier and other wrappers to create per-fit/per-class
    copies) reconstructs a fresh instance from get_params() and does not
    copy arbitrary instance state a method sets after construction.
      1. Call `set_hugiml_feature_metadata` if the estimator exposes it --
         covers direct (non-cloning) fits.
      2. If the estimator also exposes hugiml_feature_names /
         hugiml_augmented_catalog / hugiml_pattern_provenance /
         hugiml_original_feature_standardization as plain attributes (i.e.
         they are constructor parameters, the clone()-safe design), set
         those directly too -- these are the values clone() actually
         propagates, which is what makes metadata survive
         OneVsRestClassifier's internal per-class cloning.
    Silently does nothing for any estimator that recognizes neither --
    this is purely additive and never required for an ordinary fit
    (e.g. the built-in LogisticRegression downstream branch) to proceed.

    `feature_names` length is validated against nothing here (this
    function doesn't have the downstream matrix in hand) -- callers that
    do (both call sites below) should confirm feature_names' length
    matches the actual downstream column count before calling this, since
    a silent length mismatch would misattribute every later column.
    """
    targets = [estimator]
    inner = getattr(estimator, "estimator", None)
    if inner is not None:
        targets.append(inner)
    for target in targets:
        if hasattr(target, "set_hugiml_feature_metadata"):
            try:
                target.set_hugiml_feature_metadata(
                    feature_names,
                    augmented_catalog,
                    pattern_provenance,
                    original_feature_standardization,
                )
            except TypeError:
                try:
                    target.set_hugiml_feature_metadata(
                        feature_names, augmented_catalog, pattern_provenance
                    )
                except TypeError:
                    try:
                        target.set_hugiml_feature_metadata(feature_names, augmented_catalog)
                    except TypeError:
                        pass
        if hasattr(target, "hugiml_feature_names"):
            target.hugiml_feature_names = list(feature_names)
        if hasattr(target, "hugiml_augmented_catalog"):
            target.hugiml_augmented_catalog = [dict(item) for item in augmented_catalog]
        if hasattr(target, "hugiml_pattern_provenance"):
            target.hugiml_pattern_provenance = dict(pattern_provenance or {})
        if hasattr(target, "hugiml_original_feature_standardization"):
            target.hugiml_original_feature_standardization = dict(
                original_feature_standardization or {}
            )

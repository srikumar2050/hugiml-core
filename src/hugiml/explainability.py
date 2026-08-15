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

"""Enterprise explainability for HUGIMLClassifierNative.

Provides SHAP interoperability, feature lineage tracking, explanation
stability metrics, and audit artifact generation.  The core HUG patterns
are human-readable by design; this module adds depth for downstream
governance and audit workflows.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ExplainabilityReport",
    "ExplanationStabilityMetrics",
    "FeatureLineage",
    "HUGPatternExplainer",
    "aggregate_shap_to_features",
    "compute_shap_values",
    "shap_values_from_pattern_matrix",
]


# =============================================================================
# Data containers
# =============================================================================


@dataclass
class FeatureLineage:
    """Source-mapping record linking an original feature to downstream features.

    Attributes
    ----------
    feature_name : str
        Original feature name from the training DataFrame.
    feature_type : str
        One of 'integer', 'float', 'categorical'.
    derived_patterns : list of str
        Human-readable HUG pattern labels that include this feature.
    pattern_indices : list of int
        Indices into the pattern list for each derived pattern.
    derived_augmented_pairs : list of str
        Augmented-pair feature names that use this source feature.
    total_importance : float
        Sum of absolute downstream coefficients for original, HUG pattern,
        and augmented-pair features linked to this source feature.
    pattern_importance : float
        Pattern-only contribution to total_importance.
    augmented_pair_importance : float
        Augmented-pair contribution to total_importance.
    original_feature_importance : float
        Direct original-feature contribution when original features are included
        in the downstream estimator.
    """

    feature_name: str
    feature_type: str
    derived_patterns: list[str] = field(default_factory=list)
    pattern_indices: list[int] = field(default_factory=list)
    derived_augmented_pairs: list[str] = field(default_factory=list)
    total_importance: float = 0.0
    pattern_importance: float = 0.0
    augmented_pair_importance: float = 0.0
    original_feature_importance: float = 0.0


@dataclass
class ExplanationStabilityMetrics:
    """Stability metrics for pattern-based explanations.

    The top-level fields report stability for mined HUG patterns only.  When
    original or augmented-pair downstream features are present, per-feature-type
    metrics are available in ``by_feature_type`` so derived feature stability is
    not conflated with human-readable pattern-rule stability.
    """

    jaccard_similarity: float = 0.0
    rank_correlation: float = 0.0
    pattern_overlap_count: int = 0
    n_patterns_a: int = 0
    n_patterns_b: int = 0
    by_feature_type: dict[str, dict[str, float | int]] = field(default_factory=dict)


@dataclass
class ExplainabilityReport:
    """Full explainability report for a fitted classifier instance.

    Contains pattern importances, feature lineage, and stability metrics.
    Serializable to JSON for audit workflows.
    """

    model_id: str
    n_patterns: int
    n_features: int
    top_patterns: list[dict[str, Any]] = field(default_factory=list)
    feature_lineage: list[dict[str, Any]] = field(default_factory=list)
    model_composition: dict[str, Any] = field(default_factory=dict)
    augmented_pair_effects: list[dict[str, Any]] = field(default_factory=list)
    stability: dict[str, Any] | None = None
    shap_available: bool = False

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to a JSON string."""
        return json.dumps(asdict(self), indent=indent, default=str)

    def save(self, path: str) -> None:
        """Write the report to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())


# =============================================================================
# Main explainer
# =============================================================================


class HUGPatternExplainer:
    """Enterprise explainability layer over a fitted HUGIMLClassifierNative.

    Extracts feature lineage, computes explanation stability, and provides
    a SHAP-compatible interface where available.  Designed to operate on the
    already-mined HUG patterns without re-running the algorithm.

    Parameters
    ----------
    classifier : HUGIMLClassifierNative
        A fitted classifier instance.
    """

    def __init__(self, classifier: Any) -> None:
        self._clf = classifier
        self._validate_fitted()

    def _validate_fitted(self) -> None:
        if not hasattr(self._clf, "patterns_"):
            raise RuntimeError("Classifier must be fitted before creating an explainer.")

    # ------------------------------------------------------------------
    # Feature lineage
    # ------------------------------------------------------------------

    def feature_lineage(self) -> list[FeatureLineage]:
        """Build feature lineage mapping each input feature to its patterns.

        Returns
        -------
        list of FeatureLineage
            One entry per original input feature.
        """
        clf = self._clf
        feature_names = getattr(clf, "feature_names_in_", None) or []
        cat_mask = getattr(clf, "cat_cols_mask_", np.zeros(len(feature_names), dtype=bool))
        int_mask = getattr(clf, "is_int_mask_", np.zeros(len(feature_names), dtype=bool))
        pattern_labels = clf.get_hug_features()

        # Get downstream importances when available.  Pattern entries are
        # linked by HUG pattern label; augmented-pair entries are linked through
        # their source input metadata; direct original features are linked by
        # their ``orig:<name>`` namespace.
        pattern_importance: dict[str, float] = {}
        original_importance: dict[str, float] = {}
        augmented_importance: dict[str, float] = {}
        augmented_display: dict[str, str] = {}
        try:
            imp = clf.feature_importances()
            for _, row in imp.iterrows():
                ftype = str(row.get("feature_type", "pattern"))
                feature_key = str(row.get("feature", row.get("pattern", "")))
                display = str(row.get("display_name", row.get("pattern", feature_key)))
                importance = float(row.get("abs_coefficient", 0.0))
                if ftype == "pattern":
                    pattern_importance[str(row.get("pattern", display))] = importance
                elif ftype == "original" and feature_key.startswith("orig:"):
                    original_importance[feature_key[len("orig:") :]] = importance
                elif ftype == "augmented_pair":
                    augmented_importance[feature_key] = importance
                    augmented_display[feature_key] = display
        except Exception:
            logger.debug("feature_importances() unavailable for lineage scoring.", exc_info=True)

        lineage: dict[str, FeatureLineage] = {}
        for fname in feature_names:
            j = feature_names.index(fname)
            ftype = (
                "categorical"
                if j < len(cat_mask) and cat_mask[j]
                else "integer"
                if j < len(int_mask) and int_mask[j]
                else "float"
            )
            lineage[fname] = FeatureLineage(
                feature_name=fname,
                feature_type=ftype,
            )

        for fname, importance in original_importance.items():
            if fname in lineage:
                lineage[fname].original_feature_importance += importance
                lineage[fname].total_importance += importance

        for pat_idx, label in enumerate(pattern_labels):
            importance = pattern_importance.get(label, 0.0)
            parts = label.split(", ")
            for part in parts:
                if "=" in part:
                    fname = part.split("=")[0]
                    if fname in lineage:
                        lineage[fname].derived_patterns.append(label)
                        lineage[fname].pattern_indices.append(pat_idx)
                        lineage[fname].pattern_importance += importance
                        lineage[fname].total_importance += importance

        try:
            pair_catalog = clf.get_augmented_pair_transforms()
        except Exception:
            pair_catalog = []
        for item in pair_catalog:
            name = str(item.get("name", ""))
            feature_key = f"augmented_pair:{name}"
            importance = augmented_importance.get(feature_key, 0.0)
            if importance == 0.0:
                continue
            display = augmented_display.get(feature_key, str(item.get("raw_formula", name)))
            for src in item.get("inputs", []) or []:
                fname = str(src)
                if fname in lineage:
                    lineage[fname].derived_augmented_pairs.append(display)
                    lineage[fname].augmented_pair_importance += importance
                    lineage[fname].total_importance += importance

        return list(lineage.values())

    # ------------------------------------------------------------------
    # Explanation stability
    # ------------------------------------------------------------------

    def explanation_stability(
        self,
        X_a: np.ndarray,
        y_a: np.ndarray,
        X_b: np.ndarray,
        y_b: np.ndarray,
        top_n: int = 20,
    ) -> ExplanationStabilityMetrics:
        """Measure explanation stability across two data splits.

        Fits two copies of the classifier on split A and split B.  The
        headline metrics compare only mined HUG patterns.  Additional metrics
        are returned by feature type so original features, HUG patterns, and
        augmented-pair transforms are not mixed into a single stability score.

        Parameters
        ----------
        X_a, y_a : split A data
        X_b, y_b : split B data
        top_n : int
            How many top patterns to compare.

        Returns
        -------
        ExplanationStabilityMetrics
        """
        import copy

        clf_a = copy.deepcopy(self._clf)
        clf_b = copy.deepcopy(self._clf)

        # Reset fitted state and re-fit on each split
        for attr in ["patterns_", "td_", "model_", "x_train_hup_", "fit_metadata_"]:
            for c in (clf_a, clf_b):
                if hasattr(c, attr):
                    delattr(c, attr)

        clf_a.fit(X_a, y_a)
        clf_b.fit(X_b, y_b)

        try:
            imp_a = clf_a.feature_importances().head(top_n)
            imp_b = clf_b.feature_importances().head(top_n)
        except Exception:
            logger.debug(
                "explanation_stability(): could not compute feature_importances on split.",
                exc_info=True,
            )
            return ExplanationStabilityMetrics()

        def _stability_for(frame_a: Any, frame_b: Any, feature_type: str) -> dict[str, float | int]:
            fa = frame_a[frame_a.get("feature_type", "pattern") == feature_type].head(top_n)
            fb = frame_b[frame_b.get("feature_type", "pattern") == feature_type].head(top_n)
            key_col = (
                "feature" if "feature" in fa.columns and "feature" in fb.columns else "pattern"
            )
            set_a = set(fa[key_col].tolist())
            set_b = set(fb[key_col].tolist())
            overlap = set_a & set_b
            union = set_a | set_b
            jaccard = len(overlap) / len(union) if union else 0.0
            rank_corr = 0.0
            if len(overlap) >= 3:
                shared = list(overlap)
                ranks_a = [fa.index[fa[key_col] == item].tolist()[0] for item in shared]
                ranks_b = [fb.index[fb[key_col] == item].tolist()[0] for item in shared]
                if len(ranks_a) == len(ranks_b) and len(ranks_a) >= 3:
                    try:
                        from scipy.stats import spearmanr

                        corr, _ = spearmanr(ranks_a, ranks_b)
                        rank_corr = float(corr) if np.isfinite(corr) else 0.0
                    except Exception:
                        logger.debug("spearmanr rank correlation failed.", exc_info=True)
            return {
                "jaccard_similarity": round(jaccard, 4),
                "rank_correlation": round(rank_corr, 4),
                "overlap_count": len(overlap),
                "n_features_a": len(set_a),
                "n_features_b": len(set_b),
            }

        feature_types = sorted(
            set(imp_a.get("feature_type", "pattern").tolist())
            | set(imp_b.get("feature_type", "pattern").tolist())
        )
        by_type = {ft: _stability_for(imp_a, imp_b, ft) for ft in feature_types}
        pattern_metrics = by_type.get(
            "pattern",
            {
                "jaccard_similarity": 0.0,
                "rank_correlation": 0.0,
                "overlap_count": 0,
                "n_features_a": 0,
                "n_features_b": 0,
            },
        )

        return ExplanationStabilityMetrics(
            jaccard_similarity=float(pattern_metrics["jaccard_similarity"]),
            rank_correlation=float(pattern_metrics["rank_correlation"]),
            pattern_overlap_count=int(pattern_metrics["overlap_count"]),
            n_patterns_a=int(pattern_metrics["n_features_a"]),
            n_patterns_b=int(pattern_metrics["n_features_b"]),
            by_feature_type=by_type,
        )

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        model_id: str = "hugiml_model",
        top_n: int = 20,
    ) -> ExplainabilityReport:
        """Generate a complete explainability report.

        Parameters
        ----------
        model_id : str
            Identifier for this model instance.
        top_n : int
            Number of top patterns to include.

        Returns
        -------
        ExplainabilityReport
        """
        clf = self._clf

        def _json_value(value: Any) -> Any:
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                return None
            if isinstance(value, (list, tuple)):
                return [_json_value(v) for v in value]
            if isinstance(value, dict):
                return {str(k): _json_value(v) for k, v in value.items()}
            return value

        top_patterns: list[dict[str, Any]] = []
        try:
            imp = clf.feature_importances().head(top_n)
            governance_columns = [
                "pattern",
                "feature",
                "display_name",
                "feature_type",
                "coefficient",
                "abs_coefficient",
                "pattern_support",
                "support_type",
                "non_missing_rate",
                "variance",
                "strict_topk_score",
                "raw_formula",
                "standardized_formula",
                "standardization_mean",
                "standardization_scale",
                "reference_raw_value",
                "reference_raw_value_description",
                "pair_missing_policy",
                "pair_missing_policy_description",
                "eligible_count",
                "eligible_rate",
                "missing_pair_rate",
                "coefficient_standardized",
                "one_std_effect_on_log_odds",
                "coefficient_raw_scale",
                "one_raw_unit_effect_on_log_odds",
                "decision_direction",
                "risk_increases_when",
                "unit_effect_interpretation",
                "raw_scale_note",
                "raw_interpretation",
                "source_observed_medians",
                "source_observed_medians_description",
                "transform_ig",
            ]
            for _, row in imp.iterrows():
                record = {
                    col: _json_value(row.get(col)) for col in governance_columns if col in row
                }
                record.setdefault("pattern", _json_value(row.get("pattern")))
                record.setdefault("feature", _json_value(row.get("feature", row.get("pattern"))))
                record.setdefault("feature_type", _json_value(row.get("feature_type", "pattern")))
                top_patterns.append(record)
        except Exception:
            logger.debug(
                "feature_importances() unavailable in generate_report; "
                "falling back to get_pattern_info().",
                exc_info=True,
            )
            info = clf.get_pattern_info().head(top_n)
            for _, row in info.iterrows():
                top_patterns.append(
                    {
                        "pattern": row["pattern"],
                        "utility": float(row["utility"]),
                        "information_gain": float(row["information_gain"]),
                        "pattern_support": float(row["support"]),
                        "support_type": "pattern_support",
                    }
                )

        lineage = self.feature_lineage()
        lineage_dicts = [
            {
                "feature_name": fl.feature_name,
                "feature_type": fl.feature_type,
                "n_patterns": len(fl.derived_patterns),
                "n_augmented_pairs": len(fl.derived_augmented_pairs),
                "total_importance": round(fl.total_importance, 6),
                "pattern_importance": round(fl.pattern_importance, 6),
                "augmented_pair_importance": round(fl.augmented_pair_importance, 6),
                "original_feature_importance": round(fl.original_feature_importance, 6),
                "derived_patterns": fl.derived_patterns[:5],
                "derived_augmented_pairs": fl.derived_augmented_pairs[:5],
            }
            for fl in lineage
        ]

        try:
            model_composition = clf.get_model_composition()
        except Exception:
            model_composition = {
                "feature_mode": getattr(clf, "feature_mode", None),
                "topk_budget_strict": getattr(clf, "topk_budget_strict", None),
                "n_patterns_mined": len(getattr(clf, "patterns_", [])),
            }

        augmented_pair_effects: list[dict[str, Any]] = []
        try:
            effects = clf.explain_augmented_pair_effects()
            for _, row in effects.iterrows():
                augmented_pair_effects.append({str(k): _json_value(v) for k, v in row.items()})
        except Exception:
            logger.debug(
                "explain_augmented_pair_effects() unavailable in generate_report.", exc_info=True
            )

        shap_available = _shap_is_available()

        return ExplainabilityReport(
            model_id=model_id,
            n_patterns=len(clf.patterns_),
            n_features=getattr(clf, "n_features_in_", 0),
            top_patterns=top_patterns,
            feature_lineage=lineage_dicts,
            model_composition={str(k): _json_value(v) for k, v in model_composition.items()},
            augmented_pair_effects=augmented_pair_effects,
            shap_available=shap_available,
        )


# =============================================================================
# SHAP interoperability
# =============================================================================


def compute_shap_values(
    classifier: Any,
    X: Any,
    *,
    feature_scope: str = "all",
    background_samples: int = 100,
    check_additivity: bool = False,
    allow_incomplete: bool = False,
) -> np.ndarray | None:
    """Compute SHAP values for the fitted downstream representation.

    The explainer always evaluates the complete matrix returned by
    ``classifier.transform(X)`` so the SHAP input is aligned with the fitted
    downstream estimator in every feature mode. ``feature_scope="patterns"``
    optionally returns only pattern columns from that full-model explanation.

    Parameters
    ----------
    classifier : HUGIMLClassifierNative
        A fitted classifier.
    X : array-like
        Input data to explain.
    feature_scope : {"all", "patterns"}, default "all"
        Return all downstream SHAP columns or only HUG pattern columns.
    background_samples : int
        Number of background samples for KernelExplainer.
    check_additivity : bool
        Passed to KernelExplainer's SHAP call.
    allow_incomplete : bool, default False
        Required when ``feature_scope="patterns"`` would omit original or
        augmented-pair columns used by the fitted downstream estimator. SHAP is
        still computed on the complete fitted representation; this flag only
        permits returning a partial reporting view.

    Returns
    -------
    np.ndarray or None
        Binary models return ``(n_samples, n_features)`` values using the
        positive-class convention retained by the historical bridge. Multiclass
        models retain a class axis as ``(n_samples, n_features, n_classes)``.
        ``None`` is returned when SHAP is unavailable or the requested partial
        view is not explicitly permitted.
    """
    if feature_scope not in {"all", "patterns"}:
        raise ValueError("feature_scope must be either 'all' or 'patterns'.")

    downstream_names = [str(name) for name in classifier.get_downstream_features()]
    X_downstream = classifier.transform(X)
    expected_shape = getattr(X_downstream, "shape", None)
    if expected_shape is None or len(expected_shape) != 2:
        raise RuntimeError("transform(X) must return a two-dimensional downstream matrix.")
    if int(expected_shape[1]) != len(downstream_names):
        raise RuntimeError(
            "The downstream transform width does not match get_downstream_features()."
        )

    pattern_indices = [
        idx for idx, name in enumerate(downstream_names) if name.startswith("pattern:")
    ]
    partial_pattern_view = len(pattern_indices) != len(downstream_names)
    if feature_scope == "patterns" and partial_pattern_view and not allow_incomplete:
        warnings.warn(
            "Pattern-only SHAP reporting would omit downstream features used by the fitted "
            "model. Pass allow_incomplete=True to request that partial reporting view; "
            "the SHAP calculation itself will still use the complete transform(X) matrix.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    n_samples = int(expected_shape[0])
    n_features = int(expected_shape[1])
    if n_features == 0:
        return np.zeros((n_samples, 0), dtype=np.float64)

    if not _shap_is_available():
        warnings.warn(
            "SHAP is not installed. Install it with: pip install shap",
            ImportWarning,
            stacklevel=2,
        )
        return None

    import shap

    clf_step = classifier.model_.named_steps.get("clf")
    shap_values: Any | None = None
    try:
        explainer = shap.LinearExplainer(clf_step, X_downstream)
        shap_values = explainer.shap_values(X_downstream)
    except Exception:
        logger.debug("SHAP LinearExplainer was not applicable; trying KernelExplainer.", exc_info=True)

    if shap_values is None:
        try:
            bg_size = min(max(int(background_samples), 1), n_samples)
            bg_indices = np.random.choice(n_samples, bg_size, replace=False)
            bg = X_downstream[bg_indices]
            explainer = shap.KernelExplainer(classifier.model_.predict_proba, bg)
            shap_values = explainer.shap_values(
                X_downstream, check_additivity=check_additivity
            )
        except Exception as exc:
            warnings.warn(f"SHAP computation failed: {exc}", RuntimeWarning, stacklevel=2)
            return None

    classes = getattr(classifier, "classes_", None)
    values = _normalize_shap_values(
        shap_values,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=len(classes) if classes is not None else 0,
    )
    if feature_scope == "patterns":
        return _select_shap_feature_columns(values, pattern_indices)
    return values


def shap_values_from_pattern_matrix(
    classifier: Any,
    X: Any,
    *,
    background_samples: int = 100,
    check_additivity: bool = False,
    allow_incomplete: bool = False,
) -> np.ndarray | None:
    """Return the pattern-column view of full-model SHAP values.

    This historical function name is retained for compatibility. The fitted
    estimator is always evaluated on ``classifier.transform(X)``. When the
    model also uses non-pattern downstream columns, callers must explicitly set
    ``allow_incomplete=True`` to return only the pattern columns from the
    complete-model SHAP result.
    """
    return compute_shap_values(
        classifier,
        X,
        feature_scope="patterns",
        background_samples=background_samples,
        check_additivity=check_additivity,
        allow_incomplete=allow_incomplete,
    )


def aggregate_shap_to_features(
    shap_values_pattern: np.ndarray,
    classifier: Any,
    *,
    allow_incomplete: bool = False,
) -> dict[str, float]:
    """Aggregate downstream SHAP values back to original source features.

    ``shap_values_pattern`` retains its historical parameter name for API
    compatibility. The input may contain the complete downstream SHAP matrix,
    or a pattern-only subset when ``allow_incomplete=True``. Multiclass values
    may include a final class axis.
    """
    values = np.asarray(shap_values_pattern)
    if values.ndim not in {2, 3}:
        raise ValueError("SHAP values must have shape (samples, features[, classes]).")

    downstream_names = [str(name) for name in classifier.get_downstream_features()]
    pattern_names = [name for name in downstream_names if name.startswith("pattern:")]
    width = int(values.shape[1])
    if width == len(downstream_names):
        names = downstream_names
    elif width == len(pattern_names):
        if len(pattern_names) != len(downstream_names) and not allow_incomplete:
            raise ValueError(
                "Pattern-only SHAP values omit downstream features used by the fitted model. "
                "Pass allow_incomplete=True only when that partial aggregation is intended."
            )
        names = pattern_names
    else:
        raise ValueError(
            "SHAP feature width does not match either the complete downstream schema "
            "or its retained pattern subset."
        )

    reduce_axes = (0,) if values.ndim == 2 else (0, 2)
    mean_abs = np.abs(values).mean(axis=reduce_axes)
    feature_names = [str(name) for name in (getattr(classifier, "feature_names_in_", None) or [])]
    aggregated: dict[str, float] = {name: 0.0 for name in feature_names}

    provenance = dict(getattr(classifier, "get_pattern_provenance", lambda: {})() or {})
    pair_catalog = list(getattr(classifier, "get_augmented_pair_transforms", lambda: [])() or [])
    pair_sources = {
        f"augmented_pair:{row.get('name')}": [str(v) for v in (row.get("inputs") or [])]
        for row in pair_catalog
        if row.get("name") is not None
    }
    categorical_sources = sorted(
        [str(name) for name in (getattr(classifier, "_original_cat_cols_", []) or [])],
        key=len,
        reverse=True,
    )

    for idx, name in enumerate(names):
        sources: list[str] = []
        if name.startswith("pattern:"):
            sources = [str(v) for v in provenance.get(name, {}).get("raw_features", [])]
        elif name.startswith("augmented_pair:"):
            sources = list(pair_sources.get(name, []))
        elif name.startswith("orig:"):
            display = name[len("orig:") :]
            if display in aggregated:
                sources = [display]
            else:
                for source in categorical_sources:
                    if display.startswith(f"{source}_"):
                        sources = [source]
                        break
        for source in sources:
            if source in aggregated:
                aggregated[source] += float(mean_abs[idx])

    return aggregated


def _select_shap_feature_columns(values: np.ndarray, indices: list[int]) -> np.ndarray:
    if values.ndim == 2:
        return values[:, indices]
    return values[:, indices, :]


def _normalize_shap_values(
    values: Any,
    *,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    if isinstance(values, (list, tuple)):
        arrays = [np.asarray(value) for value in values]
        if len(arrays) == 2:
            arr = arrays[1]
        else:
            arr = np.stack(arrays, axis=-1)
    else:
        arr = np.asarray(values)

    if arr.ndim == 1 and n_features == 1 and arr.shape[0] == n_samples:
        arr = arr.reshape(n_samples, 1)
    if arr.ndim == 2:
        if arr.shape != (n_samples, n_features):
            raise RuntimeError(
                f"SHAP returned shape {arr.shape}, expected {(n_samples, n_features)}."
            )
        return arr
    if arr.ndim == 3:
        if arr.shape[:2] == (n_samples, n_features):
            normalized = arr
        elif n_classes and arr.shape == (n_classes, n_samples, n_features):
            normalized = np.moveaxis(arr, 0, -1)
        else:
            raise RuntimeError(
                "SHAP returned a three-dimensional array whose feature axis does not "
                "match the fitted downstream schema."
            )
        if normalized.shape[2] == 1:
            return normalized[:, :, 0]
        if n_classes == 2 and normalized.shape[2] == 2:
            return normalized[:, :, 1]
        return normalized
    raise RuntimeError("SHAP returned an unsupported output shape.")

def _shap_is_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except ImportError:
        return False

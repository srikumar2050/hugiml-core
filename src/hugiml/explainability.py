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
    "FeatureLineage",
    "ExplanationStabilityMetrics",
    "HUGPatternExplainer",
    "shap_values_from_pattern_matrix",
]


# =============================================================================
# Data containers
# =============================================================================


@dataclass
class FeatureLineage:
    """Provenance record linking an original feature to downstream features.

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


def shap_values_from_pattern_matrix(
    classifier: Any,
    X: Any,
    *,
    background_samples: int = 100,
    check_additivity: bool = False,
    allow_incomplete: bool = False,
) -> np.ndarray | None:
    """Compute SHAP values over the HUG pattern feature space.

    Applies SHAP's LinearExplainer (or KernelExplainer as fallback) on the
    binary pattern-presence matrix produced by the classifier's transform()
    method.  The resulting SHAP values are in pattern-space; use
    :func:`aggregate_shap_to_features` to roll them back to original features.

    When the fitted downstream estimator also uses original or augmented-pair
    features, pattern-space SHAP is incomplete relative to the fitted model.
    In that case this function warns and returns ``None`` unless
    ``allow_incomplete=True`` is passed explicitly.

    Requires the optional ``shap`` package (``pip install shap``).

    Parameters
    ----------
    classifier : HUGIMLClassifierNative
        A fitted classifier.
    X : array-like
        Input data to explain.
    background_samples : int
        Number of background samples for KernelExplainer.
    check_additivity : bool
        Pass to SHAP's explain call.
    allow_incomplete : bool
        If False, return None when the fitted downstream estimator uses
        original or augmented-pair features in addition to HUG patterns.

    Returns
    -------
    np.ndarray of shape (n_samples, n_patterns) or None
        SHAP values in pattern space.  Returns None when shap is not installed.
    """
    downstream_names = list(getattr(classifier, "get_downstream_features", lambda: [])())
    non_pattern = [name for name in downstream_names if not str(name).startswith("pattern:")]
    if non_pattern and not allow_incomplete:
        warnings.warn(
            "Pattern-space SHAP is incomplete because the fitted downstream estimator "
            "also uses original or augmented-pair features. Pass allow_incomplete=True "
            "only if a pattern-only diagnostic is intended.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if not _shap_is_available():
        warnings.warn(
            "SHAP is not installed.  Install it with: pip install shap",
            ImportWarning,
            stacklevel=2,
        )
        return None

    import shap

    X_hup = classifier.transform(X)

    # Try LinearExplainer first (works when downstream is LogisticRegression)
    clf_step = classifier.model_.named_steps.get("clf")
    try:
        explainer = shap.LinearExplainer(clf_step, X_hup)
        sv = explainer.shap_values(X_hup)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else np.array(sv).mean(axis=0)
        return np.array(sv)
    except Exception:
        logger.debug("SHAP LinearExplainer failed; trying KernelExplainer.", exc_info=True)
    try:
        bg_size = min(background_samples, X_hup.shape[0])
        bg_indices = np.random.choice(X_hup.shape[0], bg_size, replace=False)
        bg = X_hup[bg_indices]
        explainer = shap.KernelExplainer(classifier.model_.predict_proba, bg)
        sv = explainer.shap_values(X_hup, check_additivity=check_additivity)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else np.array(sv).mean(axis=0)
        return np.array(sv)
    except Exception as e:
        warnings.warn(f"SHAP computation failed: {e}", RuntimeWarning, stacklevel=2)
        return None


def aggregate_shap_to_features(
    shap_values_pattern: np.ndarray,
    classifier: Any,
) -> dict[str, float]:
    """Aggregate pattern-space SHAP values back to original features.

    Parameters
    ----------
    shap_values_pattern : np.ndarray, shape (n_samples, n_patterns)
    classifier : fitted HUGIMLClassifierNative

    Returns
    -------
    dict mapping feature name to mean absolute SHAP value.
    """
    downstream_names = list(getattr(classifier, "get_downstream_features", lambda: [])())
    non_pattern = [name for name in downstream_names if not str(name).startswith("pattern:")]
    if non_pattern:
        warnings.warn(
            "Aggregating pattern-space SHAP to original features omits original "
            "downstream columns and augmented-pair transforms. Use this only as a "
            "pattern-subspace diagnostic for this fitted model.",
            RuntimeWarning,
            stacklevel=2,
        )

    feature_names = getattr(classifier, "feature_names_in_", None) or []
    pattern_labels = classifier.get_hug_features()

    aggregated: dict[str, float] = {f: 0.0 for f in feature_names}
    mean_abs = np.abs(shap_values_pattern).mean(axis=0)

    for pat_idx, label in enumerate(pattern_labels):
        if pat_idx >= len(mean_abs):
            break
        parts = label.split(", ")
        for part in parts:
            if "=" in part:
                fname = part.split("=")[0]
                if fname in aggregated:
                    aggregated[fname] += float(mean_abs[pat_idx])

    return aggregated


def _shap_is_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except ImportError:
        return False

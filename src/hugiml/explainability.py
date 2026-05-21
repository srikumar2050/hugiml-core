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
    """Provenance record linking an original feature to its HUG patterns.

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
    total_importance : float
        Sum of absolute LR coefficients for all patterns using this feature.
    """

    feature_name: str
    feature_type: str
    derived_patterns: list[str] = field(default_factory=list)
    pattern_indices: list[int] = field(default_factory=list)
    total_importance: float = 0.0


@dataclass
class ExplanationStabilityMetrics:
    """Stability metrics for the pattern-based explanation.

    Attributes
    ----------
    jaccard_similarity : float
        Jaccard index between the top-N patterns across two data splits.
    rank_correlation : float
        Spearman rank correlation of pattern importances across splits.
    pattern_overlap_count : int
        Number of patterns shared between splits.
    n_patterns_a : int
        Pattern count in split A.
    n_patterns_b : int
        Pattern count in split B.
    """

    jaccard_similarity: float = 0.0
    rank_correlation: float = 0.0
    pattern_overlap_count: int = 0
    n_patterns_a: int = 0
    n_patterns_b: int = 0


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

        # Get importances when available
        coef_map: dict[int, float] = {}
        try:
            imp = clf.feature_importances()
            for idx, row in imp.iterrows():
                coef_map[idx] = float(row["abs_coefficient"])
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

        for pat_idx, label in enumerate(pattern_labels):
            importance = coef_map.get(pat_idx, 0.0)
            parts = label.split(", ")
            for part in parts:
                if "=" in part:
                    fname = part.split("=")[0]
                    if fname in lineage:
                        lineage[fname].derived_patterns.append(label)
                        lineage[fname].pattern_indices.append(pat_idx)
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

        Fits two copies of the classifier on split A and split B, then
        compares the top-N patterns via Jaccard similarity and rank
        correlation of importance scores.

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

        set_a = set(imp_a["pattern"].tolist())
        set_b = set(imp_b["pattern"].tolist())
        overlap = set_a & set_b
        union = set_a | set_b
        jaccard = len(overlap) / len(union) if union else 0.0

        # Rank correlation on shared patterns
        rank_corr = 0.0
        if len(overlap) >= 3:
            shared = list(overlap)
            ranks_a = [imp_a.index[imp_a["pattern"] == p].tolist()[0] for p in shared if p in set_a]
            ranks_b = [imp_b.index[imp_b["pattern"] == p].tolist()[0] for p in shared if p in set_b]
            if len(ranks_a) == len(ranks_b) and len(ranks_a) >= 3:
                try:
                    from scipy.stats import spearmanr

                    corr, _ = spearmanr(ranks_a, ranks_b)
                    rank_corr = float(corr) if np.isfinite(corr) else 0.0
                except Exception:
                    logger.debug("spearmanr rank correlation failed.", exc_info=True)

        return ExplanationStabilityMetrics(
            jaccard_similarity=round(jaccard, 4),
            rank_correlation=round(rank_corr, 4),
            pattern_overlap_count=len(overlap),
            n_patterns_a=len(set_a),
            n_patterns_b=len(set_b),
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

        top_patterns: list[dict[str, Any]] = []
        try:
            imp = clf.feature_importances().head(top_n)
            for _, row in imp.iterrows():
                top_patterns.append(
                    {
                        "pattern": row["pattern"],
                        "coefficient": float(row["coefficient"]),
                        "abs_coefficient": float(row["abs_coefficient"]),
                        "support": float(row["support"]),
                    }
                )
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
                        "support": float(row["support"]),
                    }
                )

        lineage = self.feature_lineage()
        lineage_dicts = [
            {
                "feature_name": fl.feature_name,
                "feature_type": fl.feature_type,
                "n_patterns": len(fl.derived_patterns),
                "total_importance": round(fl.total_importance, 6),
                "derived_patterns": fl.derived_patterns[:5],
            }
            for fl in lineage
        ]

        shap_available = _shap_is_available()

        return ExplainabilityReport(
            model_id=model_id,
            n_patterns=len(clf.patterns_),
            n_features=getattr(clf, "n_features_in_", 0),
            top_patterns=top_patterns,
            feature_lineage=lineage_dicts,
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
) -> np.ndarray | None:
    """Compute SHAP values over the HUG pattern feature space.

    Applies SHAP's LinearExplainer (or KernelExplainer as fallback) on the
    binary pattern-presence matrix produced by the classifier's transform()
    method.  The resulting SHAP values are in pattern-space; use
    :func:`aggregate_shap_to_features` to roll them back to original features.

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

    Returns
    -------
    np.ndarray of shape (n_samples, n_patterns) or None
        SHAP values in pattern space.  Returns None when shap is not installed.
    """
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

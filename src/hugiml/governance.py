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

"""Governance artifacts for HUGIMLClassifierNative.

Provides model card generation, audit artifact packaging, and governance
metadata consistent with responsible model deployment practices and the
HUG-IML paper's emphasis on interpretability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ModelCard",
    "AuditArtifact",
    "GovernanceMetadata",
    "generate_model_card",
    "package_audit_artifacts",
]


# =============================================================================
# Model card
# =============================================================================


@dataclass
class ModelCard:
    """Structured model card for a fitted HUGIMLClassifierNative.

    Follows the Google Model Cards framework adapted for rule-based
    interpretable classifiers.

    Attributes
    ----------
    model_id : str
        Unique identifier for this model version.
    model_type : str
        Always 'HUGIMLClassifierNative'.
    paper_reference : str
        Citation for the HUG-IML algorithm.
    license : str
        Software license.
    intended_use : str
        Describe the intended classification task.
    out_of_scope_use : str
        Describe uses not covered by this model.
    training_data_description : str
        Description of training data.
    evaluation_data_description : str
        Description of evaluation data.
    hyperparameters : dict
        B, L, G, topK as used during training.
    performance_metrics : dict
        Accuracy, F1, AUC, ECE, Brier score, etc.
    n_patterns : int
        Number of mined HUG patterns.
    n_compound : int
        Number of compound patterns.
    top_patterns : list of str
        Most important patterns.
    limitations : list of str
        Known limitations.
    ethical_considerations : str
        Fairness, bias, and ethical notes.
    created_at : str
        ISO 8601 timestamp of creation.
    framework_version : str
        hugiml-core version.
    """

    model_id: str
    model_type: str = "HUGIMLClassifierNative"
    paper_reference: str = (
        "Krishnamoorthy, S. (2024). Interpretable Classifier Models for "
        "Decision Support Using High Utility Gain Patterns. "
        "IEEE Access, 12, 126088-126107. DOI: 10.1109/ACCESS.2024.3455563"
    )
    license: str = "Apache-2.0"
    intended_use: str = ""
    out_of_scope_use: str = ""
    training_data_description: str = ""
    evaluation_data_description: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    n_patterns: int = 0
    n_compound: int = 0
    top_patterns: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    ethical_considerations: str = ""
    created_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    framework_version: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Render the model card as a Markdown document."""
        lines = [
            f"# Model Card: {self.model_id}",
            "",
            f"**Type:** {self.model_type}  ",
            f"**License:** {self.license}  ",
            f"**Created:** {self.created_at}  ",
            f"**Framework:** hugiml-core {self.framework_version}",
            "",
            "## Reference",
            "",
            self.paper_reference,
            "",
            "## Intended Use",
            "",
            self.intended_use or "_Not specified._",
            "",
            "## Out-of-Scope Use",
            "",
            self.out_of_scope_use or "_Not specified._",
            "",
            "## Training Data",
            "",
            self.training_data_description or "_Not specified._",
            "",
            "## Evaluation Data",
            "",
            self.evaluation_data_description or "_Not specified._",
            "",
            "## Hyperparameters",
            "",
        ]
        for k, v in self.hyperparameters.items():
            lines.append(f"- **{k}**: {v}")
        lines += [
            "",
            "## Performance Metrics",
            "",
        ]
        for k, v in self.performance_metrics.items():
            lines.append(f"- **{k}**: {v}")
        lines += [
            "",
            "## Patterns",
            "",
            f"- Total patterns: {self.n_patterns}",
            f"- Compound patterns: {self.n_compound}",
            "",
            "### Top Patterns",
            "",
        ]
        for p in self.top_patterns[:15]:
            lines.append(f"- `{p}`")
        lines += [
            "",
            "## Limitations",
            "",
        ]
        for lim in self.limitations:
            lines.append(f"- {lim}")
        lines += [
            "",
            "## Ethical Considerations",
            "",
            self.ethical_considerations or "_Not specified._",
        ]
        return "\n".join(lines)

    def save(self, path: str, fmt: str = "json") -> None:
        """Save the model card to a file.

        Parameters
        ----------
        path : str
            Output file path.
        fmt : {'json', 'markdown', 'md'}
            Output format.
        """
        if fmt in ("markdown", "md"):
            content = self.to_markdown()
        else:
            content = self.to_json()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)


@dataclass
class GovernanceMetadata:
    """Minimal governance metadata attached to a model instance.

    Attributes
    ----------
    model_id : str
    owner : str
        Person or team responsible for this model.
    purpose : str
        Business or scientific purpose.
    data_classification : str
        Sensitivity of training data (e.g. 'public', 'internal', 'confidential').
    review_status : str
        One of 'draft', 'reviewed', 'approved', 'deprecated'.
    approved_by : str or None
    approved_at : str or None
    tags : list of str
    """

    model_id: str
    owner: str = ""
    purpose: str = ""
    data_classification: str = "unclassified"
    review_status: str = "draft"
    approved_by: str | None = None
    approved_at: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return governance metadata as a plain dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise governance metadata to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# =============================================================================
# Audit artifact
# =============================================================================


@dataclass
class AuditArtifact:
    """Audit record for a model training run.

    Captures all information needed for regulatory review or internal audit.
    """

    model_id: str
    created_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    training_hash: str = ""
    model_card: dict[str, Any] | None = None
    governance: dict[str, Any] | None = None
    fit_metadata: dict[str, Any] | None = None
    pattern_info: list[dict[str, Any]] | None = None
    calibration: dict[str, Any] | None = None
    explainability: dict[str, Any] | None = None
    framework_version: str = ""

    def to_dict(self) -> dict:
        """Return audit artifact fields as a plain dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise the audit artifact to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        """Write the audit artifact to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())


# =============================================================================
# Factory functions
# =============================================================================


def generate_model_card(
    classifier: Any,
    model_id: str,
    *,
    intended_use: str = "",
    out_of_scope_use: str = "",
    training_data_description: str = "",
    evaluation_data_description: str = "",
    performance_metrics: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
    ethical_considerations: str = "",
) -> ModelCard:
    """Populate a ModelCard from a fitted classifier.

    Parameters
    ----------
    classifier : HUGIMLClassifierNative
        A fitted classifier.
    model_id : str
        Unique identifier.

    Returns
    -------
    ModelCard
    """
    from hugiml import __version__

    meta = getattr(classifier, "fit_metadata_", None)
    hparams: dict[str, object] = {}
    n_patterns = 0
    n_compound = 0
    top_patterns: list[str] = []

    if meta is not None:
        hparams = getattr(meta, "config", {})
        n_patterns = getattr(meta, "n_patterns", 0)
        n_compound = getattr(meta, "n_compound", 0)

    try:
        feat_imp = classifier.feature_importances().head(15)
        top_patterns = feat_imp["pattern"].tolist()
    except Exception:
        logger.debug(
            "feature_importances() unavailable for model card; trying get_hug_features().",
            exc_info=True,
        )
        try:
            top_patterns = classifier.get_hug_features()[:15]
        except Exception:
            logger.debug("get_hug_features() also unavailable for model card.", exc_info=True)

    default_limitations = [
        "Patterns are derived from training data; distribution shift may reduce accuracy.",
        "Auto-binning may not be optimal for all numerical feature distributions.",
        "Compound pattern mining (L > 1) increases interpretability but may reduce coverage.",
        "Downstream logistic regression may underfit when the pattern matrix is sparse.",
    ]

    return ModelCard(
        model_id=model_id,
        intended_use=intended_use,
        out_of_scope_use=out_of_scope_use,
        training_data_description=training_data_description,
        evaluation_data_description=evaluation_data_description,
        hyperparameters=hparams,
        performance_metrics=performance_metrics or {},
        n_patterns=n_patterns,
        n_compound=n_compound,
        top_patterns=top_patterns,
        limitations=limitations or default_limitations,
        ethical_considerations=ethical_considerations,
        framework_version=__version__,
    )


def package_audit_artifacts(
    classifier: Any,
    model_id: str,
    output_dir: str,
    *,
    model_card: ModelCard | None = None,
    governance: GovernanceMetadata | None = None,
    calibration_result: Any | None = None,
    explainability_report: Any | None = None,
) -> str:
    """Package all audit artifacts for a trained model.

    Writes model card, governance metadata, fit metadata, pattern info,
    and optional calibration/explainability reports to ``output_dir``.

    Returns
    -------
    str
        Path to the audit manifest JSON file.
    """
    from hugiml import __version__

    os.makedirs(output_dir, exist_ok=True)

    # Model card
    if model_card is None:
        model_card = generate_model_card(classifier, model_id)
    model_card.save(os.path.join(output_dir, "model_card.json"))
    model_card.save(os.path.join(output_dir, "model_card.md"), fmt="md")

    # Governance
    gov_dict = governance.to_dict() if governance else {"model_id": model_id}

    # Fit metadata
    meta = getattr(classifier, "fit_metadata_", None)
    meta_dict: dict[str, Any] = {}
    if meta is not None:
        for attr in [
            "n_samples",
            "n_features",
            "n_classes",
            "n_items",
            "n_patterns",
            "n_compound",
            "topK_used",
            "stage_times_ms",
            "total_fit_ms",
            "matrix_density",
            "config",
            "memory_peak_mb",
            "memory_rss_mb",
            "openmp_threads",
            "degraded",
        ]:
            val = getattr(meta, attr, None)
            if val is not None:
                meta_dict[attr] = val

    # Pattern info
    pattern_info_list: list[dict[str, Any]] = []
    try:
        df = classifier.get_pattern_info()
        pattern_info_list = df.to_dict(orient="records")
    except Exception:
        logger.warning(
            "get_pattern_info() failed; pattern_info will be empty in audit artifact.",
            exc_info=True,
        )

    # Calibration
    cal_dict = None
    if calibration_result is not None:
        try:
            cal_dict = calibration_result.to_dict()
        except Exception:
            logger.debug("calibration_result.to_dict() failed.", exc_info=True)

    # Explainability
    expl_dict = None
    if explainability_report is not None:
        try:
            expl_dict = json.loads(explainability_report.to_json())
        except Exception:
            logger.debug("explainability_report.to_json() failed.", exc_info=True)

    # Compute a hash of the pattern labels for traceability.
    try:
        pattern_labels = classifier.get_hug_features()
        label_str = json.dumps(sorted(pattern_labels), sort_keys=True)
        training_hash = hashlib.sha256(label_str.encode()).hexdigest()[:16]
    except Exception:
        logger.warning(
            "Training hash computation failed; the training hash is unavailable.", exc_info=True
        )
        training_hash = "unavailable"

    artifact = AuditArtifact(
        model_id=model_id,
        training_hash=training_hash,
        model_card=model_card.to_dict(),
        governance=gov_dict,
        fit_metadata=meta_dict,
        pattern_info=pattern_info_list,
        calibration=cal_dict,
        explainability=expl_dict,
        framework_version=__version__,
    )

    manifest_path = os.path.join(output_dir, "audit_manifest.json")
    artifact.save(manifest_path)
    return manifest_path

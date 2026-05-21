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

"""hugiml-core — High-performance interpretable rule-based ML infrastructure.

Implements the High Utility Gain Interpretable Machine Learning (HUG-IML)
framework described in:

    Krishnamoorthy, S. (2024). Interpretable Classifier Models for Decision
    Support Using High Utility Gain Patterns. IEEE Access, 12, 126088-126107.
    DOI: 10.1109/ACCESS.2024.3455563

Public API
----------
Core classifier::

    from hugiml import HUGIMLClassifierNative

    clf = HUGIMLClassifierNative(B=8, L=2, G=1e-4)
    X, y = clf.prepareXy(X_df, y_series)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    print(clf.model_summary())

Calibration::

    from hugiml.calibration import evaluate_calibration
    result = evaluate_calibration(y_test, proba)
    print(result.summary())

Explainability::

    from hugiml.explainability import HUGPatternExplainer
    explainer = HUGPatternExplainer(clf)
    report = explainer.generate_report("my_model")

Governance::

    from hugiml.governance import generate_model_card, package_audit_artifacts
    card = generate_model_card(clf, "my_model", intended_use="credit scoring")
    card.save("model_card.md", fmt="md")

Serialisation::

    from hugiml.serialization import generate_sbom
    sbom = generate_sbom(output_path="sbom.json")

Telemetry::

    from hugiml.telemetry import instrument_classifier
    clf = instrument_classifier(clf, model_id="v1")
"""

from __future__ import annotations

__version__ = "2.1.0"
__author__ = "Srikumar Krishnamoorthy"
__license__ = "Apache-2.0"
__paper__ = (
    "Krishnamoorthy, S. (2024). Interpretable Classifier Models for Decision "
    "Support Using High Utility Gain Patterns. "
    "IEEE Access, 12, 126088-126107. DOI: 10.1109/ACCESS.2024.3455563"
)

from hugiml.classifier import FitMetadata, HUGIMLClassifierNative
from hugiml.exceptions import (
    HUGIMLConvergenceWarning,
    HUGIMLDegradedWarning,
    HUGIMLDtypeDriftWarning,
    HUGIMLError,
    HUGIMLFitError,
    HUGIMLMiningError,
    HUGIMLParamError,
    HUGIMLPredictionError,
    HUGIMLRangeWarning,
    HUGIMLSchemaError,
    HUGIMLSerializationError,
    HUGIMLTimeoutError,
    HUGIMLValidationError,
    HUGIMLVersionError,
    HUGIMLWarning,
)
from hugiml.monitoring import DriftDetector, DriftReport, PredictionMonitor

__all__ = [
    # Classifier
    "HUGIMLClassifierNative",
    "FitMetadata",
    # Monitoring
    "PredictionMonitor",
    "DriftDetector",
    "DriftReport",
    # Exceptions
    "HUGIMLError",
    "HUGIMLFitError",
    "HUGIMLMiningError",
    "HUGIMLTimeoutError",
    "HUGIMLValidationError",
    "HUGIMLSchemaError",
    "HUGIMLParamError",
    "HUGIMLSerializationError",
    "HUGIMLVersionError",
    "HUGIMLPredictionError",
    "HUGIMLWarning",
    "HUGIMLConvergenceWarning",
    "HUGIMLDtypeDriftWarning",
    "HUGIMLRangeWarning",
    "HUGIMLDegradedWarning",
    # Version
    "__version__",
    "__paper__",
]

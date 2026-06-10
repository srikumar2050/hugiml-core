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

    from hugiml import HUGIMLClassifier

    clf = HUGIMLClassifier(B=7, L=1, G=5e-3)
    X, y = clf.prepareXy(X_df, y_series)   # schema/type prep — no fitting
    clf.fit(X_train, y_train)               # discretisation + mining + downstream fit
    proba = clf.predict_proba(X_test)
    print(clf.model_summary())

Adaptive binning (v1.1.0)::

    clf = HUGIMLClassifier(
        adaptive_binning=True,
        b_candidates=[2, 3, 5, 7, 10, 15],
        min_marginal_gain_ratio=0.02,
        L=2, G=1e-4,
    )
    clf.fit(X_train, y_train)
    print(clf.per_feature_b_)   # chosen B_j per feature

Visualisations (requires plotly)::

    from hugiml.plots import HUGPlotter
    plotter = HUGPlotter(clf)
    plotter.plot_marginal_bin_profile("age").show()
    plotter.plot_dashboard().show()

Interpretability metrics::

    from hugiml.metrics import compute_all_metrics
    m = compute_all_metrics(clf, X_test)
    print(m)

Pattern pruning::

    from hugiml.pruning import PatternEditor
    editor = PatternEditor(clf, operator_name="analyst")
    editor.remove_by_keyword("gender", reason="protected attribute")
    new_clf = editor.refit(X_tr, y_tr).calibrate(X_cal, y_cal).finalize()

Standalone adaptive wrapper::

    from hugiml.adaptive import HUGIMLAdaptive
    clf = HUGIMLAdaptive(b_candidates=[2, 3, 5, 7, 10, 15], L=2, G=1e-4)
    clf.fit(X_train, y_train)

Multiclass / imbalanced / high-cardinality::

    from hugiml.multiclass import MulticlassHUGReport, make_imbalanced_pipeline

Benchmark runner (CLI)::

    python -m hugiml.benchmarks.runner --datasets breast_cancer adult
"""

from __future__ import annotations

__version__ = "1.1.9"
__author__ = "Srikumar Krishnamoorthy"
__license__ = "Apache-2.0"
__paper__ = (
    "Krishnamoorthy, S. (2024). Interpretable Classifier Models for Decision "
    "Support Using High Utility Gain Patterns. "
    "IEEE Access, 12, 126088-126107. DOI: 10.1109/ACCESS.2024.3455563"
)

from hugiml.classifier import FitMetadata, HUGIMLClassifier, HUGIMLClassifierNative
from hugiml.exceptions import (
    HUGIMLConvergenceWarning,
    HUGIMLDegradedWarning,
    HUGIMLDtypeDriftWarning,
    HUGIMLError,
    HUGIMLFitError,
    HUGIMLMemoryError,
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


def __getattr__(name: str):
    """Lazy import gate for optional-dependency sub-modules."""
    _lazy = {
        "HUGPlotter": ("hugiml.plots", "HUGPlotter"),
        "InterpretabilityMetrics": ("hugiml.metrics", "InterpretabilityMetrics"),
        "compute_all_metrics": ("hugiml.metrics", "compute_all_metrics"),
        "metrics_dataframe": ("hugiml.metrics", "metrics_dataframe"),
        "PatternEditor": ("hugiml.pruning", "PatternEditor"),
        "HUGIMLAdaptive": ("hugiml.adaptive", "HUGIMLAdaptive"),
        "MulticlassHUGReport": ("hugiml.multiclass", "MulticlassHUGReport"),
        "make_imbalanced_pipeline": ("hugiml.multiclass", "make_imbalanced_pipeline"),
        "encode_high_cardinality": ("hugiml.multiclass", "encode_high_cardinality"),
        "apply_encoding": ("hugiml.multiclass", "apply_encoding"),
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(_lazy[name][0])
        return getattr(mod, _lazy[name][1])
    raise AttributeError(f"module 'hugiml' has no attribute {name!r}")


__all__ = [
    "HUGIMLClassifier",
    "HUGIMLClassifierNative",
    "FitMetadata",
    "PredictionMonitor",
    "DriftDetector",
    "DriftReport",
    "HUGIMLError",
    "HUGIMLFitError",
    "HUGIMLMiningError",
    "HUGIMLTimeoutError",
    "HUGIMLMemoryError",
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
    "HUGPlotter",
    "InterpretabilityMetrics",
    "compute_all_metrics",
    "metrics_dataframe",
    "PatternEditor",
    "HUGIMLAdaptive",
    "MulticlassHUGReport",
    "make_imbalanced_pipeline",
    "encode_high_cardinality",
    "apply_encoding",
    "__version__",
    "__paper__",
]

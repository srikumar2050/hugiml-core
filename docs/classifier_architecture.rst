Classifier architecture
=======================

The classifier implementation uses a public façade and focused internal modules.
The public estimator remains available from ``hugiml.classifier`` and from the
package root.

Public contract
---------------

``HUGIMLClassifier`` is the concrete estimator class. ``HUGIMLClassifierNative``
is an alternate public name for the same class. Both names therefore share the
same constructor, fitted state, sklearn behavior, and serialization contract.

``classifier.py`` contains:

* the public module documentation;
* native-extension availability state;
* the estimator constructor and parameter assignments;
* public helper and exception exports;
* the mixin composition that defines the estimator.

Internal modules
----------------

``_classifier_estimator.py``
   sklearn parameter handling, presets, estimator tags, object state, and model
   save/load methods.

``_classifier_binning.py``
   input preparation, parameter validation, column metadata, adaptive binning,
   missing-value preparation, and pair-aware bin selection.

``_classifier_training.py``
   native availability checks, fit orchestration, mining execution, fallback
   models, and probability/class prediction.

``_classifier_features.py``
   original, pattern, and augmented-pair feature assembly; strict ``topK``
   selection; downstream matrix construction; and feature-name alignment.

``_classifier_interpretation.py``
   pattern provenance, RPTE rule and alias views, complexity delegation,
   standardization metadata, and augmented-pair effect explanations.

``_classifier_prediction.py``
   transform-time pattern construction, schema and health checks, monitoring,
   drift APIs, and monitored cross-validation.

``_classifier_inspection.py``
   pattern and downstream-feature inventories, model composition, coefficient
   summaries, adaptive-bin visualizations, and model summaries.

``_classifier_tuning.py``
   cached and standard grid evaluation, scoring helpers, and tuning result data.

``_classifier_support.py``
   numerical helpers, augmented-pair transformation state, fit metadata,
   memory tracking, transaction wrappers, and shared constants.

``_classifier_runtime.py``
   runtime access to mutable native-extension and monitoring symbols exposed by
   ``hugiml.classifier``.

Dependency rules
----------------

Internal modules do not import the concrete classifier class. Type annotations
use deferred evaluation, and behavior is composed through ``self``. This avoids
cycles between the façade and the implementation modules.

Mutable native-extension and monitoring symbols are resolved through
``_classifier_runtime.py`` so test instrumentation and application-level
substitution through ``hugiml.classifier`` remain effective.

The public façade re-exports classifier-specific helpers and support types used
by package integrations. Public support types retain ``hugiml.classifier`` as
their module path for stable pickle globals.

Adding functionality
--------------------

Place new behavior in the module matching its responsibility. Cross-cutting
state should be documented where it is created and consumed. Constructor
parameters remain in ``HUGIMLClassifier.__init__`` so sklearn introspection and
cloning continue to derive the complete parameter contract from one location.

A new public method should be covered by tests for its owning module and, when
it affects fitted state, by serialization and sklearn-cloning tests. Changes to
feature construction should also cover DataFrame and ndarray inputs, binary and
multiclass targets, and audit and production execution modes.

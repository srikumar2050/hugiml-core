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
   pattern source metadata, RPTE rule and alias views, complexity delegation,
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

Lifecycle and fitted-state ownership
------------------------------------

The mixins compose one estimator instance and therefore share ``self.__dict__``. They are implementation modules rather than independent objects with isolated state. Extensions should use the public ``HUGIMLClassifier`` API instead of subclassing an internal mixin directly; private mixin methods and fitted attributes can change together as the estimator lifecycle evolves.

The canonical lifecycle is:

#. **Constructed** — constructor parameters and ``_fit_lock`` exist; fitted attributes are absent.
#. **Input preparation** — binning and schema metadata such as ``feature_names_in_``, ``cat_cols_mask_``, ``is_int_mask_``, and ``_bin_edges_`` are established.
#. **Mining** — transaction state, ``raw_patterns_``/``patterns_``, ``x_train_hup_``, mining audit state, and any resource-degradation record are created.
#. **Downstream assembly** — original-feature preprocessing, augmented-pair state, strict-budget masks, downstream feature names, and ``x_train_downstream_`` are created before ``model_`` is fitted.
#. **Fitted** — ``classes_`` records the learned class labels, ``model_`` stores the fitted downstream estimator pipeline, and ``fit_metadata_`` summarizes the completed fit. Production execution mode may then discard audit-only training matrices while retaining prediction-critical state.
#. **Transform/predict** — prediction paths read fitted schema, binning, pattern, feature-assembly, and estimator state. Monitoring may update monitoring counters, but fitted model structure is not rebuilt.
#. **Refit** — fitted lifecycle state is cleared before a new fit establishes a fresh schema, mining audit trail, downstream representation, and estimator.

The main ownership boundaries are:

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Module / mixin
     - Primarily creates or updates
     - Primarily reads from other stages
   * - ``_EstimatorMixin``
     - estimator protocol state, serialization compatibility, execution-mode retention
     - fitted artifacts retained or removed after training
   * - ``_BinningMixin``
     - ``feature_names_in_``, ``cat_cols_mask_``, ``is_int_mask_``, ``_bin_edges_``, adaptive-binning metadata
     - constructor parameters and training input schema
   * - ``_TrainingMixin``
     - ``classes_``, ``td_``, ``raw_patterns_``, ``patterns_``, ``x_train_hup_``, ``model_``, ``fit_metadata_``, ``mining_audit_log_``
     - binning metadata and downstream feature-assembly methods
   * - ``_FeatureAssemblyMixin``
     - ``_original_scaler_``, original-feature metadata, augmented-pair transforms, strict-budget/canonicalization masks, ``x_train_downstream_`` and downstream names
     - schema/binning state and mined pattern matrix
   * - ``_PredictionMixin``
     - transform-time validation/monitoring state
     - fitted schema, binning, patterns, feature-assembly metadata, and downstream estimator
   * - ``_InterpretationMixin`` / ``_InspectionMixin``
     - derived explanation and inspection views
     - retained fitted state from all earlier stages

When maintainers add a fitted attribute, document the stage that creates it, the modules that consume it, whether it is prediction-critical or audit-only, how refit clears it, and whether serialization must preserve it. This keeps the implicit shared-state surface reviewable even though the implementation is distributed across mixins.

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

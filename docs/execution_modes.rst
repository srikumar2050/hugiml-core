Execution modes
===============

HUGIML 1.1.9 adds an ``execution_mode`` parameter for choosing how much fitted training-review state is retained after ``fit``.

.. list-table::
   :header-rows: 1

   * - Mode
     - Retained state
     - Use when
   * - ``"audit"``
     - Full traceability state, including training matrices and drift baselines.
     - You need model cards, governance review, drift diagnostics, pattern-support tables, or detailed explanation workflows.
   * - ``"production"``
     - Prediction-critical state, fitted metadata, selected feature names, and lightweight shape summaries.
     - You need a smaller deployed model object for prediction and save/load workflows.

``audit`` is the default and preserves the v1.1.8 behavior for governance-heavy workflows.

Basic usage
-----------

.. code-block:: python

   from hugiml import HUGIMLClassifier

   audit_model = HUGIMLClassifier(execution_mode="audit")
   audit_model.fit(X_train, y_train)
   print(audit_model.get_pattern_info())

   production_model = HUGIMLClassifier(execution_mode="production")
   production_model.fit(X_train, y_train)
   proba = production_model.predict_proba(X_new)

Production-mode guidance
------------------------

Production mode keeps prediction, ``predict_proba``, ``score``, ``save_model``, ``load_model``, ``model_summary``, ``feature_importances``, and ``get_hug_features`` available. Methods that need retained training-review artifacts, such as pattern support details and drift baselines, raise a clear message asking you to refit or load an audit-mode model when complete traceability is required.

Recommended workflow
--------------------

Use audit mode during development, validation, governance review, and documentation. Save the audit-mode artifact for traceability. Use production mode for deployed scoring services when the model has already passed review and the retained training matrices are not needed in the live prediction object.

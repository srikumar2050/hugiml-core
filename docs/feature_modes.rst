Feature modes
=============

HUGIML supports three downstream feature representations. The mined pattern matrix is always available through ``transform(X)``. The ``feature_mode`` parameter controls only the matrix passed to ``fit``, ``predict``, ``predict_proba``, and ``score``.

.. list-table::
   :header-rows: 1

   * - ``feature_mode``
     - Downstream estimator input
     - Use when
   * - ``"patterns_only"``
     - Binary HUG pattern matrix only.
     - You want the standard HUGIML representation and the most compact explanation surface.
   * - ``"original_plus_patterns"``
     - Original features plus all mined binary patterns.
     - Original features contain strong marginal signal and HUG patterns add nonlinear refinements.
   * - ``"original_plus_interactions"``
     - Original features plus only higher-order ``L > 1`` patterns.
     - Original features should capture marginal effects while HUGIML contributes interaction regions.


Augmented pair features
-----------------------

When ``L > 1``, ``adaptive_binning=True``, and ``augmented_pair_transforms=True``, HUGIML can append native augmented-pair transforms to the downstream feature matrix. These are continuous product or absolute-difference features selected from informative numeric source features. They are not fed back into HUG pattern mining.

Use ``topk_budget_strict=True`` when a single global ``topK`` budget should apply to the final downstream feature space across original features, HUG patterns, and augmented-pair features.

.. code-block:: python

   clf = HUGIMLClassifier(
       B=-1,
       adaptive_binning=True,
       L=2,
       topK=50,
       feature_mode="original_plus_patterns",
       augmented_pair_transforms=True,
       topk_budget_strict=True,
   )

   clf.fit(X_train, y_train)
   print(clf.get_model_composition())
   print(clf.explain_augmented_pair_effects())

Examples
--------

.. code-block:: python

   from hugiml import HUGIMLClassifier

   clf = HUGIMLClassifier(
       B=10,
       L=2,
       G=1e-3,
       topK=150,
       adaptive_binning=True,
       feature_mode="patterns_only",
   )

   clf_hybrid = HUGIMLClassifier(
       B=10,
       L=2,
       G=1e-3,
       topK=150,
       adaptive_binning=True,
       feature_mode="original_plus_patterns",
   )

   clf_interactions = HUGIMLClassifier(
       B=10,
       L=2,
       G=1e-3,
       topK=150,
       adaptive_binning=True,
       feature_mode="original_plus_interactions",
   )

Downstream matrix policy
------------------------

In v1.1.9, ``patterns_only`` keeps the downstream representation sparse. Hybrid modes choose dense representation for small or moderate selected widths and CSR representation for larger selected widths. This keeps ordinary sklearn workflows convenient while reducing memory pressure for wider selected feature spaces.


Interpretation notes
--------------------

``get_hug_features`` and ``get_pattern_info`` are always pattern-only APIs. In hybrid modes, ``feature_importances`` and ``model_summary`` report the downstream feature representation used by the fitted model, which can include original features, mined patterns, and augmented-pair features. Use ``explain_augmented_pair_effects()`` for raw-scale interpretation of augmented-pair rows.



Compatibility with transform
----------------------------

``transform(X)`` and ``fit_transform(X, y)`` intentionally remain pattern-space APIs in every feature mode. This means existing explanation workflows that expect the binary HUG pattern matrix continue to work after enabling a hybrid downstream representation.

In hybrid modes, the fitted downstream estimator receives a private design matrix built from standardized original features, selected pattern columns, and any active augmented-pair columns. ``feature_importances()``, ``model_summary()``, and ``get_model_composition()`` report that downstream feature space so that diagnostics match what the estimator actually used.

Operational guidance
--------------------

* Use ``patterns_only`` as the default for audits, model cards, and compact explanations.
* Use ``original_plus_patterns`` when original columns contain strong marginal signal that should remain directly available to the downstream classifier.
* Use ``original_plus_interactions`` when the original columns should carry marginal effects and HUGIML should contribute higher-order regions only.
* Serialized models preserve the selected feature mode, original-feature preprocessing state, pattern-order masks, and downstream feature names.

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

Examples
--------

.. code-block:: python

   from hugiml import HUGIMLClassifierNative

   clf = HUGIMLClassifierNative(
       B=10,
       L=2,
       G=1e-3,
       topK=150,
       adaptive_binning=True,
       feature_mode="patterns_only",
   )

   clf_hybrid = HUGIMLClassifierNative(
       B=10,
       L=2,
       G=1e-3,
       topK=150,
       adaptive_binning=True,
       feature_mode="original_plus_patterns",
   )

   clf_interactions = HUGIMLClassifierNative(
       B=10,
       L=2,
       G=1e-3,
       topK=150,
       adaptive_binning=True,
       feature_mode="original_plus_interactions",
   )

Interpretation notes
--------------------

``get_hug_features`` and ``get_pattern_info`` are always pattern-only APIs. In hybrid modes, ``feature_importances`` and ``model_summary`` report the downstream feature representation used by the fitted model, which can include original features as well as mined patterns.



Compatibility with transform
----------------------------

``transform(X)`` and ``fit_transform(X, y)`` intentionally remain pattern-space APIs in every feature mode. This means existing explanation workflows that expect the binary HUG pattern matrix continue to work after enabling a hybrid downstream representation.

In hybrid modes, the fitted downstream estimator receives a private design matrix built from standardized original features plus the selected pattern columns. ``feature_importances()`` and ``model_summary()`` report that downstream feature space so that model diagnostics match what the estimator actually used.

Operational guidance
--------------------

* Use ``patterns_only`` as the default for audits, model cards, and compact explanations.
* Use ``original_plus_patterns`` when original columns contain strong marginal signal that should remain directly available to the downstream classifier.
* Use ``original_plus_interactions`` when the original columns should carry marginal effects and HUGIML should contribute higher-order regions only.
* Serialized models preserve the selected feature mode, original-feature preprocessing state, pattern-order masks, and downstream feature names.

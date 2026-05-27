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


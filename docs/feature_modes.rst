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


Recommended configurations
--------------------------

.. list-table::
   :header-rows: 1

   * - Goal
     - Key settings
     - Interpretability
     - Notes
   * - Pattern-only with interaction-relaxed mining
     - ``feature_mode="patterns_only"``, ``L=2``, ``adaptive_binning=True``, ``augmented_pair_transforms=False``, ``interaction_relaxed_mining=True``
     - Very high
     - Keeps the downstream model in original HUG pattern space. Interaction-information survivor sources can enter mining even when their marginal signal is weak.
   * - Pattern-only with augmented pairs
     - ``feature_mode="patterns_only"``, ``L=2``, ``adaptive_binning=True``, ``augmented_pair_transforms=True``, ``augmented_pair_mode="interaction_information"``
     - High
     - Often the faster interaction option because pair evidence is added as selected downstream features rather than widening the mining search.
   * - Original features plus patterns, relaxed mining
     - ``feature_mode="original_plus_patterns"``, ``L=2``, ``adaptive_binning=True``, ``augmented_pair_transforms=False``, ``interaction_relaxed_mining=True``
     - High
     - Useful when original columns have strong marginal signal and survivor-led HUG patterns add interaction evidence.
   * - Original features plus patterns, augmented pairs
     - ``feature_mode="original_plus_patterns"``, ``L=2``, ``adaptive_binning=True``, ``augmented_pair_transforms=True``, ``augmented_pair_mode="interaction_information"``
     - Moderate
     - Highest representation capacity among the recommended options; includes selected originals, HUG patterns, and pair features.

``augmented_pair_transforms=True`` and ``interaction_relaxed_mining=True`` are mutually exclusive for ``L >= 2``. Use augmented pairs when runtime is the main constraint. Use interaction-relaxed mining when preserving a pure original-feature pattern representation is the priority.

No-pattern behavior
-------------------

Starting in v1.1.15, a ``patterns_only`` fit that mines no HUG patterns installs a deterministic constant-prior fallback. ``predict_proba`` returns the empirical training class prior, ``predict`` returns the majority class, and ``transform`` returns an empty CSR pattern matrix with the requested row count. Hybrid modes continue to use the original feature block when no patterns are mined.


Augmented pair features
-----------------------

When ``L > 1``, ``adaptive_binning=True``, and ``augmented_pair_transforms=True``, HUGIML can append native augmented-pair transforms to the downstream feature matrix. These are continuous product, absolute-difference, sum, or signed-difference features selected from informative numeric source features. They are not fed back into HUG pattern mining.

The default ``augmented_pair_mode="interaction_information"`` uses pair-context evidence for source selection. Set ``augmented_pair_mode="marginal_ig"`` to use the v1.1.11 marginal-information-gain source selection behavior. ``aug_feature_size`` controls the interaction-information source budget, ``ii_partner_size`` optionally bounds partner search, and ``max_pair_features`` controls the marginal-IG source budget.

Use ``topk_budget_strict=True`` when a single global ``topK`` budget should apply to the final downstream feature space across original features, HUG patterns, and augmented-pair features.

.. code-block:: python

   clf = HUGIMLClassifier(
       B=-1,
       adaptive_binning=True,
       L=2,
       topK=50,
       feature_mode="original_plus_patterns",
       augmented_pair_transforms=True,
       augmented_pair_mode="interaction_information",
       aug_feature_size=10,
       topk_budget_strict=True,
   )

   clf.fit(X_train, y_train)
   print(clf.get_model_composition())
   print(clf.explain_augmented_pair_effects())


Interaction-relaxed mining
--------------------------

``interaction_relaxed_mining=True`` keeps the downstream representation in HUG pattern space but lets interaction-information survivor sources participate in native mining. This is useful when a feature has weak marginal information gain but becomes useful with a partner feature.

.. code-block:: python

   clf = HUGIMLClassifier(
       B=-1,
       adaptive_binning=True,
       L=2,
       topK=75,
       feature_mode="patterns_only",
       augmented_pair_transforms=False,
       interaction_relaxed_mining=True,
       interaction_relaxed_feature_size=12,
   )

   clf.fit(X_train, y_train)
   print(clf.feature_importances().head())

Survivor-led patterns remain ordinary HUG patterns over original feature bins. Governance and feature-importance outputs add survivor metadata so reviewers can see which pattern rows used the relaxed source admission path.


Examples
--------

.. code-block:: python

   from hugiml import HUGIMLClassifier

   clf = HUGIMLClassifier(
       B=-1,
       L=2,
       G=1e-3,
       topK=150,
       adaptive_binning=True,
       feature_mode="patterns_only",
   )

   clf_hybrid = HUGIMLClassifier(
       B=-1,
       L=2,
       G=1e-3,
       topK=150,
       adaptive_binning=True,
       feature_mode="original_plus_patterns",
   )

   clf_interactions = HUGIMLClassifier(
       B=-1,
       L=2,
       G=1e-3,
       topK=150,
       adaptive_binning=True,
       feature_mode="original_plus_interactions",
   )

Downstream matrix policy
------------------------

``patterns_only`` keeps the downstream representation sparse. Hybrid modes choose dense representation for small or moderate selected widths and CSR representation for larger selected feature spaces. This keeps ordinary sklearn workflows convenient while reducing memory pressure for wider selected feature spaces.


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
* Serialized models preserve the selected feature mode, original-feature preprocessing state, pattern-order masks, augmented-pair settings, relaxed-mining settings, and downstream feature names.

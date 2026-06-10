Augmented pair features
=======================

For ``L > 1`` adaptive-binning models, HUGIML can add native augmented-pair features to the downstream estimator. These features are continuous transforms built from informative numeric source features and are separate from the mined HUG pattern matrix.

When they are active
--------------------

Augmented-pair features are considered when all of the following are true:

* ``L > 1``;
* ``adaptive_binning=True``;
* ``augmented_pair_transforms=True``;
* adaptive-binning information-gain scores are available for numeric features.

The default value of ``augmented_pair_transforms`` is ``True``. If prerequisites are not met, no augmented-pair features are added and the model remains a standard HUG pattern model for the downstream estimator.

Feature families
----------------

A fitted downstream estimator can contain three feature families:

* ``pattern`` — mined binary HUG patterns;
* ``original`` — standardized original features, when ``feature_mode`` includes them;
* ``augmented_pair`` — native continuous pair transforms such as products, absolute differences, sums, and signed differences.

``transform(X)`` intentionally remains a pattern-space API and returns only the binary HUG pattern matrix. Use ``get_downstream_features()``, ``feature_importances()``, and ``get_model_composition()`` to inspect the fitted downstream feature space.

Example
-------

.. code-block:: python

   from hugiml import HUGIMLClassifier

   clf = HUGIMLClassifier(
       B=-1,
       adaptive_binning=True,
       L=2,
       topK=50,
       feature_mode="original_plus_patterns",
       augmented_pair_transforms=True,
   )
   clf.fit(X_train, y_train)

   print(clf.get_model_composition())
   print(clf.feature_importances().head(20))
   print(clf.explain_augmented_pair_effects())

Interpretation
--------------

Augmented-pair features use product, absolute-difference, sum, and signed-difference transforms. Public metadata reports the raw formula, standardized formula, information-gain provenance, observed-row coverage, and raw-scale coefficient interpretation.

For a product feature such as ``glucose * bmi``, the raw-unit effect is on the product scale and depends on the current value of the other source variable. For an absolute-difference feature such as ``abs(age - duration)``, the raw-unit effect is on the distance scale. Sum and signed-difference features expose additive pair effects while preserving the same raw-to-standardized metadata structure introduced in v1.1.6.

Candidate scoring uses rows where both source values are observed. For selected pair features, rows where the pair cannot be computed receive the pair feature's training reference raw value before standardization. This gives a neutral standardized value for that pair term and keeps HUG pattern features on their native missing-value handling path.

Budgeting
---------

By default, HUG patterns and augmented pairs have their own retained-feature budgets. Set ``topk_budget_strict=True`` when the downstream estimator should receive a single global ``topK`` budget across original features, HUG patterns, and augmented-pair features. This is useful when a hard fitted-feature count is required for review or deployment.

Review workflow
---------------

Useful public methods include:

* ``get_augmented_pair_transforms()`` for transform catalog metadata;
* ``get_augmented_pair_standardization()`` for raw-to-standardized mapping;
* ``explain_augmented_pair_effects()`` for raw-scale interpretation;
* ``get_model_composition()`` for downstream feature-family counts;
* ``PatternEditor.list_downstream_features()`` for editable pattern features plus non-editable context features.

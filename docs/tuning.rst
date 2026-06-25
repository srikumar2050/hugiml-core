Fast hyperparameter tuning
==========================

``HUGIMLClassifier.tune`` provides a GridSearchCV-like interface. Eligible
adaptive-binning grids reuse cached mining work for faster validation.
Unsupported grids automatically use ordinary per-candidate evaluation.

Recommended named grids
-----------------------

The recommended HUGIML grids are centralized in
``hugiml.hyperparameter_configs`` and are reused by the classifier, benchmark
runner, and dashboard Workbench.

``performance`` is the default first-pass grid:

.. code-block:: python

   from hugiml import HUGIMLClassifier

   performance_grid = HUGIMLClassifier.default_param_grid()

   # Equivalent explicit grid
   performance_grid = {
       "B": [-1],
       "adaptive_binning": [True],
       "L": [1, 2],
       "topK": [50, 100],
       "feature_mode": ["original_plus_patterns"],
       "G": [0.01, 0.001],
   }

``interpretability`` keeps the final representation pattern-focused:

.. code-block:: python

   interpretability_grid = HUGIMLClassifier.default_param_grid("interpretability")

   # Equivalent explicit grid
   interpretability_grid = {
       "B": [-1],
       "adaptive_binning": [True],
       "L": [1, 2],
       "topK": [50, 100],
       "feature_mode": ["patterns_only"],
       "G": [0.01, 0.001],
       "interaction_relaxed_mining": [True],
       "augmented_pair_transforms": [False],
   }

Cross-validated tuning
----------------------

.. code-block:: python

   result = HUGIMLClassifier.tune(
       X_train,
       y_train,
       cv=5,
       scoring="roc_auc",
       param_grid="performance",
       refit=True,
       use_fast_path=True,
   )

   print(result.best_params_)
   print(result.best_score_)
   print(result.results_)

Use ``param_grid="interpretability"`` for the pattern-only recommended grid, or
pass a custom sklearn-style grid when you need a focused search. For large
adaptive-binning datasets, include ``adaptive_binning_sample_frac`` as a fixed
value such as ``0.20`` when you want bin-count selection to use a deterministic
stratified sample before full-data fitting.

Focused follow-up grids
-----------------------

Keep focused grids compact. For ``L >= 2``, do not enable augmented-pair
transforms and interaction-relaxed mining in the same candidate.

.. code-block:: python

   patterns_relaxed = {
       "adaptive_binning": [True],
       "L": [2],
       "G": [1e-2, 5e-3],
       "topK": [50, 100],
       "feature_mode": ["patterns_only"],
       "augmented_pair_transforms": [False],
       "interaction_relaxed_mining": [True],
       "interaction_relaxed_feature_size": [8, 12],
   }

   patterns_augmented = {
       "adaptive_binning": [True],
       "L": [2],
       "G": [1e-2, 5e-3],
       "topK": [50, 100],
       "feature_mode": ["patterns_only"],
       "augmented_pair_transforms": [True],
       "augmented_pair_mode": ["interaction_information"],
       "aug_feature_size": [8, 12],
   }

   originals_plus_patterns = {
       "adaptive_binning": [True],
       "L": [2],
       "G": [1e-2, 5e-3],
       "topK": [50, 100],
       "feature_mode": ["original_plus_patterns"],
       "augmented_pair_transforms": [True],
       "augmented_pair_mode": ["interaction_information"],
       "aug_feature_size": [8, 12],
   }

Fast path eligibility
---------------------

The cached path is intended for adaptive-binning grids where the varying
dimensions are limited to mining and representation parameters such as ``G``,
``L``, ``topK``, and ``feature_mode``. Other grids remain valid and are
evaluated through the standard path.

Computational budget notes
--------------------------

``interaction_relaxed_mining=True`` changes the native mining search by
admitting interaction-information survivor sources. It does not add a
downstream augmented-pair feature family, so the downstream width budget remains
the same as the corresponding pattern-only or original-plus-patterns
configuration.

``augmented_pair_transforms=True`` usually keeps mining narrower and adds
selected pair features after mining. Its downstream width can add up to one
additional ``topK``-bounded feature family unless ``topk_budget_strict=True``
applies a single global cap.

Use ``interaction_relaxed_feature_size`` to bound the survivor-source pool for
relaxed mining. Use ``aug_feature_size`` and optional ``ii_partner_size`` to
bound augmented-pair source selection in interaction-information mode.

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

``performance_ho`` is the default first-pass grid (``DEFAULT_HUGIML_GRID_NAME``).
It searches the same mining/representation budget as ``performance`` below,
plus an RPTE (higher-order, boosted-tree) downstream branch alongside the
built-in logistic-regression branch -- see :doc:`explanations` for how to
explain an RPTE-based result. RPTE's final logistic representation is
``RPTE leaf indicators + direct source terms``. Direct source terms are supplied originals, HUG patterns, or augmented pairs that are not selected in accepted tree splits and therefore remain standalone LR terms.
``performance`` remains available by name for a linear-only
first pass:

.. code-block:: python

   from hugiml import HUGIMLClassifier

   default_grid = HUGIMLClassifier.default_param_grid()  # performance_ho
   performance_grid = HUGIMLClassifier.default_param_grid("performance")
   interpretability_grid = HUGIMLClassifier.default_param_grid("interpretability")
   interpretability_ho_grid = HUGIMLClassifier.default_param_grid("interpretability_ho")

   # Equivalent explicit performance_ho grid
   from sklearn.multiclass import OneVsRestClassifier
   from hugiml.rpte_bounded_lookahead_leafwise import LeafWiseBoundedLookaheadRPTEFeatureLR

   default_grid = {
       "B": [-1],
       "adaptive_binning": [True],
       "L": [1, 2],
       "topK": [50, 100],
       "feature_mode": ["original_plus_patterns"],
       "G": [0.01, 0.001],
       "convert_binary_to_categorical": [False],
       "augmented_pair_transforms": [True],
       "topk_budget_strict": [False],
       "base_estimator": [
           None,
           OneVsRestClassifier(
               LeafWiseBoundedLookaheadRPTEFeatureLR(
                   leaf_config="3xD", depth=4, enable_lookahead="adaptive"
               ),
               n_jobs=1,
           ),
       ],
   }

   # Equivalent explicit performance grid (LR-only, no RPTE search)
   performance_grid = {
       "B": [-1],
       "adaptive_binning": [True],
       "L": [1, 2],
       "topK": [50, 100],
       "feature_mode": ["original_plus_patterns"],
       "G": [0.01, 0.001],
       "convert_binary_to_categorical": [False],
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
       "convert_binary_to_categorical": [True],
       "augmented_pair_transforms": [False],
   }

The named augmented-pair grids explicitly retain numeric 0/1 columns as numeric sources with
``convert_binary_to_categorical=False``. The named interaction-relaxed grids
explicitly use ``True`` so 0/1 indicators enter the categorical item surface.

``interpretability_ho`` uses the same pattern-only mining surface and adds a
downstream choice between LR and sequential RPTE. It explicitly sets
``enable_lookahead=False`` and keeps augmented-pair generation inactive, so
the RPTE branch cannot enter the bounded-lookahead backend:

.. code-block:: python

   interpretability_ho_grid = HUGIMLClassifier.default_param_grid(
       "interpretability_ho"
   )

Cross-validated tuning
----------------------

.. code-block:: python

   result = HUGIMLClassifier.tune(
       X_train,
       y_train,
       cv=5,
       scoring="roc_auc",
       param_grid="performance_ho",  # default; omit param_grid for the same effect
       refit=True,
       use_fast_path=True,
   )

   print(result.best_params_)
   print(result.best_score_)
   print(result.results_)

   # If RPTE wins, use the readable tree view or structured rows.
   best_model = result.best_estimator_
   if hasattr(best_model, "rpte_rule_tree"):
       print(best_model.rpte_rule_tree())
       rules = best_model.rpte_rule_table()

Use ``param_grid="performance"`` to restrict tuning to the linear-only
first-pass grid, ``param_grid="interpretability"`` for the pattern-only LR
grid, ``param_grid="interpretability_ho"`` for pattern-only LR versus
sequential RPTE, or pass a custom sklearn-style grid when you need a focused
search. For large adaptive-binning datasets, include
``adaptive_binning_sample_frac`` as a fixed value such as ``0.20`` when you
want bin-count selection to use a deterministic stratified sample before
full-data fitting.

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
       "convert_binary_to_categorical": [True],
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
       "convert_binary_to_categorical": [False],
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
       "convert_binary_to_categorical": [False],
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
relaxed mining. Relaxed admission covers the root and immediate first-child
positions of the initial branch, not the root alone; later positions do not
receive a new exemption. Use ``aug_feature_size`` and optional
``ii_partner_size`` to bound augmented-pair source selection in
interaction-information mode.

Fast hyperparameter tuning
==========================

``HUGIMLClassifier.tune`` provides a GridSearchCV-like interface. Eligible adaptive-binning grids can reuse cached mining work for faster validation. Unsupported grids automatically use ordinary per-candidate evaluation.

.. code-block:: python

   result = HUGIMLClassifier.tune(
       X_train,
       y_train,
       cv=5,
       scoring="roc_auc",
       param_grid={
           "adaptive_binning": [True],
           "G": [1e-2, 5e-3, 1e-3],
           "L": [1, 2],
           "topK": [30, 50, 100],
           "feature_mode": ["patterns_only", "original_plus_patterns"],
       },
       refit=True,
       use_fast_path=True,
   )

   print(result.best_params_)
   print(result.best_score_)
   print(result.results_)

Fast path eligibility
---------------------

The cached path is intended for adaptive-binning grids where the varying dimensions are limited to mining and representation parameters such as ``G``, ``L``, ``topK``, and ``feature_mode``. Other grids remain valid and are evaluated through the standard path.

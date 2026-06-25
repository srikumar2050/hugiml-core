Core concepts
=============

HUG-IML process
---------------

HUG-IML extracts High Utility Gain patterns from labelled tabular data, transforms every row into a binary pattern-presence matrix, and trains an interpretable downstream classifier on that matrix.

.. image:: images/positioning-mosaic.png
   :alt: HUGIML positioning mosaic
   :width: 760px

The practical modeling sequence is:

#. Detect or provide feature types: integer, float, and categorical.
#. Discretize numerical columns into bins.
#. Build transactions from observed feature-bin/category items.
#. Mine top High Utility Gain patterns with information-gain filtering.
#. Transform samples into a sparse binary matrix indicating which patterns fire.
#. Fit the downstream estimator, logistic regression by default.
#. Expose pattern labels, support, utility, information gain, coefficients, and local active-pattern explanations.

Pattern anatomy
---------------

A learned pattern is readable because it is expressed in the vocabulary of the input data:

.. code-block:: text

   glucose=[157.1,177.3) AND bmi=[31.8,39.1)   coef=+1.41 support=0.067
   checking_status=no_checking                 coef=+1.12 support=0.390

Useful fields include:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``pattern``
     - Human-readable feature interval/category combination.
   * - ``support``
     - Fraction of training samples where the pattern is active.
   * - ``utility``
     - Pattern utility score used during mining.
   * - ``information_gain``
     - Supervised filtering signal for target relevance.
   * - ``coefficient``
     - Downstream classifier contribution, when available.

Key hyperparameters
-------------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - Role
     - Typical starting point
   * - ``B``
     - Numerical bin count, or upper bound when adaptive binning is enabled.
     - ``5`` to ``15``
   * - ``L``
     - Maximum pattern length. ``L=1`` mines singleton patterns; larger values allow interactions.
     - ``1`` for larger datasets, ``2`` for interaction discovery
   * - ``G``
     - Minimum information gain threshold.
     - ``1e-3`` to ``1e-5``
   * - ``topK``
     - Maximum number of retained patterns. In the native 1.1.x path this budget is applied inside mining, so it is both an interpretability cap and a performance control. ``-1`` computes an automatic budget from the mined item universe.
     - ``50`` to ``300`` for compact audits; use ``-1`` only when the candidate space is known to be manageable
   * - ``adaptive_binning``
     - Selects per-feature bin resolution with supervised information gain.
     - ``True`` when tuning ``B`` is undesirable
   * - ``adaptive_binning_sample_frac``
     - Optionally selects adaptive bin counts on a deterministic stratified row sample before full-data training.
     - ``False`` for full-data selection; ``0.1`` to ``0.3`` for larger adaptive runs

Missing values
--------------

Numerical ``NaN`` and infinite values are treated as not observed. The corresponding item is absent from the transaction, so patterns requiring that feature do not fire for that row. This avoids fabricating values through mean or median imputation.

When missingness is itself meaningful, add explicit binary indicators before fitting:

.. code-block:: python

   X_aug = X.copy()
   for col in X.columns:
       if X[col].isna().mean() > 0.05:
           X_aug[f"{col}__MISSING"] = X[col].isna().astype(int)

Adaptive binning
----------------

Adaptive binning selects a bin count per numerical feature by evaluating candidate ``B`` values using supervised information gain and stopping when marginal gains become small. On larger datasets, ``adaptive_binning_sample_frac`` can run the bin-count search on a deterministic stratified row sample; the selected edges are then applied to the full training data.

.. image:: images/adaptive-binning-impact.png
   :alt: Adaptive binning benchmark
   :width: 700px

.. code-block:: python

   from hugiml import HUGIMLClassifier

   clf = HUGIMLClassifier(
       adaptive_binning=True,
       b_candidates=[2, 3, 5, 7, 10, 15],
       min_marginal_gain_ratio=0.02,
       adaptive_binning_sample_frac=0.20,
       L=2,
       G=1e-4,
   )



Performance and mining behavior
--------------------------------

The 1.1.x native path keeps the modeling interface stable but changes where work is bounded internally. The effective ``topK`` value is passed into the native mining stage before candidate retention, rather than being treated only as a post-processing cap. This is closer to the original HUGIML Java implementation and is important for larger datasets because fewer non-final candidates need to be materialized.

For ``L=1`` fits, the native hot path fuses transaction preparation, singleton pattern mining, information-gain filtering, top-K retention, and sparse matrix construction. This hot path supports adaptive binning without first materializing a separate binned matrix, which is the recommended route for large adaptive singleton workflows. Set ``use_hotpath=False`` only when comparing against the older three-stage path for debugging or benchmarking.

Transaction construction is performed in row-stripe chunks on the non-fused path, and materialized native transactions now store compact item ids with shared item-level utility lookup. The resulting model is intended to match the previous transaction semantics while reducing repeated utility storage and making memory use less bursty. This is most useful for wide data, large batches, and cross-validation loops.

For interaction mining, structured constraints are applied exactly. EUCS pruning is disabled by default because aggressive pair-level pruning can remove valid higher-order patterns in edge cases. Prefer controlling complexity with ``L``, ``G``, and ``topK`` first.

Operational stability controls
------------------------------

Use ``n_jobs=-1`` to allow the native backend to use all available OpenMP threads. The adaptive bin-selection and bin-code application stages can use this parallelism before the main mining step, so it benefits adaptive workflows as well as the fused ``L=1`` path.

``max_fit_seconds`` is a wall-clock budget for the native mining stage. If the budget or memory pressure prevents the full configuration from completing, HUGIML attempts safer fallback configurations, records the degraded outcome in ``fit_metadata_.degraded``, and raises a clear ``HUGIMLTimeoutError`` or ``HUGIMLMemoryError`` only when it cannot recover. Inspect ``fit_metadata_`` after fitting to review pattern counts, stage timings, memory estimates, OpenMP thread count, and whether a fallback was used.

Constant and zero-utility columns
---------------------------------

A column that is constant after preprocessing cannot contribute useful pattern utility. The estimator may emit ``HUGIMLConvergenceWarning`` for these columns during ``prepareXy`` or ``fit``. This warning is expected: the column is ignored for pattern mining, while the rest of the dataset continues to train normally. In production pipelines, drop known constant columns before fitting if you want a quieter log.

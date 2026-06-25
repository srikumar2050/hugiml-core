Getting started
===============

Installation
------------

Install the package from PyPI:

.. code-block:: bash

   pip install hugiml-core

Install optional extras when needed:

.. code-block:: bash

   pip install "hugiml-core[plots]"          # Plotly dashboards and profile plots
   pip install "hugiml-core[benchmarks]"     # benchmark comparison dependencies
   pip install "hugiml-core[imbalanced]"     # imbalanced-learn helper pipeline
   pip install "hugiml-core[explainability]" # SHAP bridge
   pip install "hugiml-core[server]"         # FastAPI inference server dependencies
   pip install "hugiml-core[all]"            # all optional extras

Build from source when you need to edit the C++ extension or package internals:

.. code-block:: bash

   git clone https://github.com/srikumar2050/hugiml-core.git
   cd hugiml-core
   pip install -e ".[dev]"
   python setup.py build_ext --inplace

Minimal classifier workflow
---------------------------

``prepareXy`` performs schema and type preparation only. It does not mine patterns or fit the model. Mining and downstream classifier fitting happen inside ``fit``.

.. code-block:: python

   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import roc_auc_score
   from hugiml import HUGIMLClassifier

   clf = HUGIMLClassifier(adaptive_binning=True, L=1, G=5e-3, topK=100)

   X_enc, y_enc = clf.prepareXy(X_df, y)
   X_train, X_test, y_train, y_test = train_test_split(
       X_enc,
       y_enc,
       test_size=0.25,
       stratify=y_enc,
       random_state=42,
   )

   clf.fit(X_train, y_train)
   proba = clf.predict_proba(X_test)[:, 1]

   print("AUC:", roc_auc_score(y_test, proba))
   print(clf.model_summary())
   print(clf.get_pattern_info().head())

Cross-validation and production schemas
---------------------------------------

When you already know the feature schema, pass ``allCols`` and ``origColumns`` explicitly. This is often cleaner in cross-validation loops and production pipelines.

.. code-block:: python

   clf = HUGIMLClassifier(
       allCols=[integer_columns, float_columns, categorical_columns],
       origColumns=X_train.columns.tolist(),
       B=-1,
       adaptive_binning=True,
       b_candidates=[2, 3, 5, 7, 10, 15],
       adaptive_binning_sample_frac=0.20,
       L=1,
       G=1e-5,
       topK=150,
   )

   clf.fit(X_train, y_train)
   predictions = clf.predict(X_test)
   probabilities = clf.predict_proba(X_test)

For smaller datasets, keep ``adaptive_binning_sample_frac=False`` to select bins on all rows. For larger adaptive-binning runs, a fractional value such as ``0.20`` reduces the bin-selection workload while the fitted model still uses the full training data after edges are selected.

Recommended first checks
------------------------

After fitting, inspect both predictive behavior and explanation complexity:

.. code-block:: python

   print(clf.get_transformed_shape())
   print(clf.get_hug_features()[:10])
   print(clf.feature_importances().head(20))
   print(clf.get_pattern_info().head(20))



Recommended starting points
---------------------------

Start with a bounded pattern budget. Then choose the interaction path based on the representation you want reviewers to inspect.

.. list-table::
   :header-rows: 1

   * - Goal
     - Suggested settings
     - Interpretability
   * - Fast compact baseline
     - ``adaptive_binning=True``, ``L=1``, ``feature_mode="patterns_only"``
     - Very high
   * - Pattern-only interactions
     - ``adaptive_binning=True``, ``L=2``, ``feature_mode="patterns_only"``, ``augmented_pair_transforms=False``, ``interaction_relaxed_mining=True``
     - Very high
   * - Faster interaction lift
     - ``adaptive_binning=True``, ``L=2``, ``feature_mode="patterns_only"``, ``augmented_pair_transforms=True``, ``augmented_pair_mode="interaction_information"``
     - High
   * - Original features plus patterns
     - ``feature_mode="original_plus_patterns"`` with either interaction-relaxed mining or augmented pairs
     - High to moderate

.. code-block:: python

   clf = HUGIMLClassifier(
       B=-1,
       adaptive_binning=True,
       L=1,
       G=5e-3,
       topK=100,
       n_jobs=-1,
       use_hotpath=True,
   )

   clf.fit(X_train, y_train)
   print(clf.fit_metadata_.summary())
   print(clf.fit_metadata_)

Use ``L=2`` when interaction patterns or pair evidence are important, and compensate by tightening ``G`` or keeping ``topK`` bounded. Use ``topK=-1`` only for smaller datasets or controlled benchmark runs, because it allows the automatic budget to grow with the item universe.

For ``L >= 2``, ``augmented_pair_transforms=True`` and ``interaction_relaxed_mining=True`` are mutually exclusive. Augmented pairs usually keep runtime lower by adding selected downstream pair features. Interaction-relaxed mining keeps the model in original-feature HUG pattern space and adds survivor-led audit metadata.

If your logs show ``HUGIMLConvergenceWarning`` for a constant column, the model is telling you that the column has zero utility. Drop the column upstream if it is expected; otherwise, treat it as a data-quality signal.

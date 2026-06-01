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
   from hugiml import HUGIMLClassifierNative

   clf = HUGIMLClassifierNative(B=7, L=1, G=5e-3)

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

   clf = HUGIMLClassifierNative(
       allCols=[integer_columns, float_columns, categorical_columns],
       origColumns=X_train.columns.tolist(),
       B=15,
       L=1,
       G=1e-5,
       topK=150,
       adaptive_binning=True,
       b_candidates=[2, 3, 5, 7, 10, 15],
   )

   clf.fit(X_train, y_train)
   predictions = clf.predict(X_test)
   probabilities = clf.predict_proba(X_test)

Recommended first checks
------------------------

After fitting, inspect both predictive behavior and explanation complexity:

.. code-block:: python

   print(clf.get_transformed_shape())
   print(clf.get_hug_features()[:10])
   print(clf.feature_importances().head(20))
   print(clf.get_pattern_info().head(20))



Performance-oriented starting point
-----------------------------------

For the current 1.1.5 implementation, start with the native ``L=1`` hot path, a bounded pattern budget, and adaptive binning only when per-feature bin selection is useful. Increase complexity only when validation results justify it:

.. code-block:: python

   clf = HUGIMLClassifierNative(
       B=7,
       L=1,
       G=5e-3,
       topK=100,
       n_jobs=-1,
       use_hotpath=True,
   )

   clf.fit(X_train, y_train)
   print(clf.fit_metadata_.summary())
   print(clf.fit_metadata_)

Use ``adaptive_binning=True`` with ``L=1`` when you want supervised per-feature bin resolution without paying the cost of a fully materialized adaptive pre-binned matrix. Use ``L=2`` when interaction patterns are important, and compensate by tightening ``G`` or keeping ``topK`` bounded. Use ``topK=-1`` only for smaller datasets or controlled benchmark runs, because it allows the automatic budget to grow with the item universe.

If your logs show ``HUGIMLConvergenceWarning`` for a constant column, the model is telling you that the column has zero utility. Drop the column upstream if it is expected; otherwise, treat it as a data-quality signal.

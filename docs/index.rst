hugiml-core documentation
==========================

``hugiml-core`` is a high-performance implementation of the HUG-IML framework for intrinsic interpretable machine learning on tabular data. It learns High Utility Gain patterns, converts input rows into a binary pattern-presence representation, and fits an interpretable downstream classifier on top of those patterns.

The package is designed for teams that need competitive tabular classification with explanations that can be reviewed as intervals, categories, supports, utilities, information gain values, and coefficients rather than as post-hoc approximations of a black-box model.

.. image:: images/header-hugiml.png
   :alt: HUGIML interpretable tabular machine learning
   :width: 760px

Why HUGIML?
-----------

* **Intrinsic interpretability:** learned HUG patterns are the model representation, not an after-the-fact explanation.
* **Regulated-domain fit:** model cards, audit artifacts, pruning trails, calibration, drift monitoring, and deployment helpers are included.
* **Native performance:** computationally intensive mining, transaction construction, and matrix-building stages are C++ accelerated with optional OpenMP support. Version 1.1.5 extends the native L1 and adaptive-binning improvements with lower transaction memory use, fused adaptive L1 processing, earlier ``n_jobs`` parallelism, and cleaner memory/timeout fallback behavior.
* **Python ergonomics:** the estimator follows the scikit-learn API and works with pandas DataFrames or NumPy arrays.


Current release focus
---------------------

The 1.1.5 release keeps the public estimator API stable while improving scalability and native execution reliability:

* Materialized native transactions use compact item ids plus shared item-level utility lookup, reducing repeated per-transaction utility storage.
* The adaptive ``L=1`` workflow is fused into the native hot path, so adaptive bin selection and singleton pattern mining avoid an intermediate binned-matrix materialization.
* Native adaptive bin selection and bin-code application are parallelized, with ``n_jobs`` applied before adaptive preprocessing.
* Native memory and timeout failures now surface through cleaner error handling and safer fallback behavior under pressure.
* Earlier 1.1.x capabilities remain available: the 1.1.4 native ``L=1`` hot path, 1.1.3 topK-in-mining and chunking behavior, and the 1.1.2 feature modes ``patterns_only``, ``original_plus_patterns``, and ``original_plus_interactions``.

Installation
------------

.. code-block:: bash

   pip install hugiml-core
   pip install "hugiml-core[plots]"       # optional Plotly dashboards
   pip install "hugiml-core[benchmarks]"  # optional comparison suite

Paper reference
---------------

The implementation is based on:

   Krishnamoorthy, S. (2024). *Interpretable Classifier Models for Decision Support Using High Utility Gain Patterns*. IEEE Access, 12, 126088--126107. DOI: 10.1109/ACCESS.2024.3455563.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   concepts
   feature_modes
   explanations
   governance
   monitoring
   deployment
   benchmarks
   notebooks
   readthedocs
   paper

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
   changelog

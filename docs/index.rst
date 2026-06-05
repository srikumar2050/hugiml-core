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
* **Native performance:** computationally intensive mining, transaction construction, and matrix-building stages are C++ accelerated with optional OpenMP support. Version 1.1.7 adds a native ``L=2`` hot path and expands augmented-pair operations, while retaining the v1.1.6 augmented-feature APIs and earlier native scalability improvements from the 1.1.x series.
* **Python ergonomics:** the estimator follows the scikit-learn API and works with pandas DataFrames or NumPy arrays.


Current release focus
---------------------

The 1.1.7 release builds on the v1.1.6 augmented-feature interface and keeps the standard pattern-space APIs stable:

* Native mining now includes a dedicated ``L=2`` hot path for common two-item pattern workloads.
* ``L > 1`` adaptive-binning models can add product, absolute-difference, sum, and signed-difference augmented-pair features to the downstream estimator.
* The benchmark baseline notebook and exported HTML were refreshed with the current baseline run and HUGIML grid settings.
* Earlier 1.1.x capabilities remain available, including strict global ``topK`` budgeting, compact native transaction memory, and fused adaptive ``L=1`` execution.

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
   augmented_features
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
   :titlesonly:
   :caption: Project

   contributing
   changelog

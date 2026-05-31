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
* **Native performance:** computationally intensive mining, transaction construction, and matrix-building stages are C++ accelerated with optional OpenMP support. Version 1.1.4 adds a native L1 fit hot path and moves adaptive binning selection into the C++ backend, while retaining the previous topK and chunked transaction-construction improvements.
* **Python ergonomics:** the estimator follows the scikit-learn API and works with pandas DataFrames or NumPy arrays.


Current release focus
---------------------

The 1.1.4 release keeps the public estimator API stable while improving the native execution path:

* Native L1 hot path for ``L=1`` fits: transaction preparation, single-item mining, information-gain filtering, top-K retention, and sparse training-matrix construction are fused in the C++ path.
* Adaptive binning selection now runs in the C++ backend using supervised information-gain scoring and elbow-style stopping, while preserving Python metadata such as ``per_feature_b_``, ``_bin_edges_``, and ``ig_scores_``.
* The 1.1.3 mining improvements remain in place: ``topK`` is pushed into mining and transaction preparation uses row-stripe chunking to reduce avoidable memory pressure.
* The feature-mode APIs introduced in 1.1.2 remain unchanged: ``patterns_only``, ``original_plus_patterns``, and ``original_plus_interactions``.

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

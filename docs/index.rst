hugiml-core documentation
=========================

``hugiml-core`` is a high-performance implementation of the HUG-IML framework for intrinsic interpretable machine learning on tabular data. It learns High Utility Gain patterns, converts input rows into a binary pattern-presence representation, and fits an interpretable downstream classifier on top of those patterns.

The package is designed for teams that need competitive tabular classification with explanations that can be reviewed as intervals, categories, supports, utilities, information gain values, and coefficients rather than as post-hoc approximations of a black-box model.

Why HUGIML?
-----------

* **Intrinsic interpretability:** learned HUG patterns are the model representation, not an after-the-fact explanation.
* **Regulated-domain fit:** model cards, audit artifacts, pruning trails, calibration, drift monitoring, and deployment helpers are included.
* **Native performance:** computationally intensive mining, transaction construction, matrix-building, and adaptive-binning stages are C++ accelerated where available, with Python fallbacks for portability.
* **Python ergonomics:** the estimator follows the scikit-learn API and works with pandas DataFrames or NumPy arrays.

Current release focus
---------------------

The 1.1.4 release keeps the public training behavior stable while improving the native execution path for two high-impact areas:

* ``L=1`` training uses a native hot path for fused preparation, single-item mining, top-K retention, and sparse training-matrix construction.
* Adaptive binning can use a native supervised per-feature bin-selection path while preserving Python metadata for inspection and serialization.
* The broader documentation from earlier releases remains part of the published site: concepts, feature modes, explanations, governance, monitoring, deployment, benchmarks, notebooks, Read the Docs publishing notes, paper scope, and the API reference.

Installation
------------

.. code-block:: bash

   pip install hugiml-core
   pip install "hugiml-core[plots]"       # optional Plotly dashboards
   pip install "hugiml-core[benchmarks]"  # optional comparison suite

Paper reference
---------------

The implementation is based on:

   Krishnamoorthy, S. (2024). Interpretable Classifier Models for Decision Support Using High Utility Gain Patterns. IEEE Access, 12, 126088--126107. DOI: 10.1109/ACCESS.2024.3455563.

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

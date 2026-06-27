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
* **Native performance:** computationally intensive mining, transaction construction, interaction-information scoring, pair-aware adaptive binning, and matrix-building stages are C++ accelerated with optional OpenMP support. Version 1.1.15 builds on v1.1.14 with stronger interaction-relaxed mining, deterministic no-pattern fallback behavior, fallback-aware serialization, and refreshed interpretable benchmark grids. Earlier 1.1.x capabilities remain available, including production/audit execution modes, fast adaptive-binning tuning, higher-order interaction selection, the native ``L=2`` hot path, augmented-pair operations, strict global ``topK`` budgeting, compact native transaction memory, and fused adaptive ``L=1`` execution.
* **Python ergonomics:** the estimator follows the scikit-learn API and works with pandas DataFrames or NumPy arrays.


Current release focus
---------------------

The 1.1.15 release focuses on predictable behavior and refreshed
interpretable-model comparisons while retaining the v1.1.14 adaptive-binning
memory improvements:

* Interaction-relaxed mining can use survivor-pair joint evidence when native
  external-utility admission would otherwise drop weak-marginal interaction
  features.
* ``patterns_only`` fits that mine no HUG patterns now install a deterministic
  constant-prior fallback instead of failing.
* Versioned save/load persists fallback state so predictions remain stable after
  serialization.
* Benchmark grids now include compact EBM settings and a registered RuleFit grid
  for broader interpretable-baseline comparisons.

The v1.1.14 adaptive-binning sampling and float32-aware native L1 preparation
remain available.

Installation
------------

.. code-block:: bash

   pip install hugiml-core
   pip install "hugiml-core[plots]"       # optional Plotly dashboards
   pip install "hugiml-core[benchmarks]"  # optional comparison suite
   pip install "hugiml-core[dashboard]"   # optional Streamlit dashboard

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
   execution_modes
   augmented_features
   tuning
   explanations
   dashboard
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

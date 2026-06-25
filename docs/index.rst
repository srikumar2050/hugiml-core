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
* **Native performance:** computationally intensive mining, transaction construction, interaction-information scoring, pair-aware adaptive binning, and matrix-building stages are C++ accelerated with optional OpenMP support. Version 1.1.14 builds on v1.1.13 with lower-memory adaptive-binning selection, float32-aware native L1 preparation, and Workbench Advanced support for adaptive-binning sampling. Earlier 1.1.x capabilities remain available, including production/audit execution modes, fast adaptive-binning tuning, higher-order interaction selection, the native ``L=2`` hot path, augmented-pair operations, strict global ``topK`` budgeting, compact native transaction memory, and fused adaptive ``L=1`` execution.
* **Python ergonomics:** the estimator follows the scikit-learn API and works with pandas DataFrames or NumPy arrays.


Current release focus
---------------------

The 1.1.14 release focuses on memory-efficient adaptive-binning execution
while retaining the v1.1.13 runtime and configuration improvements:

* Adaptive bin-count selection can use a deterministic stratified row sample,
  then apply the selected bin edges to the full training data.
* All-numeric float32 inputs can stay float32 through native L1 preparation,
  reducing widening copies while preserving native C-contiguous input handling.
* Governance Studio Workbench Advanced exposes the adaptive-binning sampling
  option for larger adaptive workflows.

The v1.1.13 categorical handling, zero-variance exclusion, centralized grids,
and lean tuning defaults remain available.

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

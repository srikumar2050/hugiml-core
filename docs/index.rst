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
* **Native performance:** computationally intensive mining, transaction construction, interaction-information scoring, pair-aware adaptive binning, and matrix-building stages are C++ accelerated with optional OpenMP support. Version 1.1.17 builds on v1.1.16 with the native ``L=3`` mining hot path, explicit mining-stage time budgets, compact mining audit logs, and additional hot-path coverage. Earlier 1.1.x capabilities remain available, including production/audit execution modes, fast adaptive-binning tuning, higher-order interaction selection, the native ``L=2`` hot path, augmented-pair operations, strict global ``topK`` budgeting, compact native transaction memory, fused adaptive ``L=1`` execution, the optional LLM assistant, and native HUIM pruning controls.
* **Python ergonomics:** the estimator follows the scikit-learn API and works with pandas DataFrames or NumPy arrays.


Current release focus
---------------------

The 1.1.17 release focuses on performance optimization and L3 hot-path mining
while retaining the v1.1.16 LLM assistant and native pruning documentation:

* Added a native ``L=3`` mining hot path for bounded three-item pattern
  workloads.
* Added explicit mining-stage time budgets through ``max_mining_seconds`` while
  keeping ``max_fit_seconds`` as a backward-compatible alias.
* Added compact mining audit logs that record attempt status, timeout budget,
  elapsed time, and returned pattern counts.
* Improved native timeout handling so partial mining results can be retained
  when a deadline is reached.
* Added focused tests for L3 equivalence, timeout/audit behavior, relaxed
  bounded mining beyond ``L=3``, and zero-variance edge cases.
* Added downstream ``lr_solver`` choices for the default historical path,
  ``saga`` LogisticRegression, and ``sgd`` SGDClassifier.
* Added scalability-dashboard solver scenarios plus an optional
  privacy-sanitized reproducibility/SBOM manifest.

The full 1.1.x feature set remains documented in this guide, including the
v1.1.16 LLM assistant, v1.1.15 deterministic no-pattern fallback, production
execution mode, adaptive binning, augmented-pair features, governance tools,
pruning controls, monitoring, deployment, and benchmark workflows.

Installation
------------

.. code-block:: bash

   pip install hugiml-core
   pip install "hugiml-core[plots]"       # optional Plotly dashboards
   pip install "hugiml-core[benchmarks]"  # optional comparison suite
   pip install "hugiml-core[dashboard]"   # optional Streamlit dashboard
   pip install "hugiml-core[llm]"         # optional LLM assistant

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
   mining_pruning
   explanations
   dashboard
   llm_assistant
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

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
* **Native performance:** computationally intensive mining, transaction construction, interaction-information scoring, pair-aware adaptive binning, matrix construction, and RPTE tree search are C++ accelerated with optional OpenMP support. Version 1.1.20 extends validation, benchmark, and representation-audit capabilities while retaining the complete 1.1.x feature set.
* **Python ergonomics:** the estimator follows the scikit-learn API and works with pandas DataFrames or NumPy arrays.


Current release focus
---------------------

The 1.1.20 release extends reproducible validation, representation audit, and
benchmark coverage without changing the public estimator contract:

* Integrated workflows cover the internal 100-dataset evaluation and four
  external classification suites: OpenML-CC18, PMLBmini, TabZilla, and
  TabArena. Each workflow retains its suite-specific validation protocol.
* Rotating-fold validation uses a dedicated validation fold, and RPTE can use
  validation-guided tree growth with a bounded estimator budget.
* Training-only downstream representation controls remove constant, redundant,
  and highly represented terms before logistic fitting. Their audit metadata is
  available through the interpretability APIs and benchmark outputs.
* Benchmark checkpoints and dashboards consistently report performance,
  tuning, fit and prediction time, RPTE behavior, and model, model-inspection,
  and instance-inspection complexity.
* TabArena evaluation uses fold-local model-agnostic preprocessing and supports
  comparison with the official leaderboard and prediction-derived metrics.

The complete 1.1.x documentation remains in this guide, including execution
modes, adaptive binning, feature modes, augmented-pair features, RPTE,
mining controls, explanations, monitoring, deployment, and benchmark workflows.

Installation
------------

.. code-block:: bash

   pip install hugiml-core
   pip install "hugiml-core[plots]"       # optional Plotly dashboards
   pip install "hugiml-core[benchmarks]"  # optional comparison suite
   pip install "hugiml-core[dashboard]"   # optional Dash and Streamlit interfaces
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
   rpte
   complexity_units
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
   classifier_architecture

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Project

   contributing
   changelog

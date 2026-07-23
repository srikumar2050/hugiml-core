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
* **Native performance:** computationally intensive mining, transaction construction, interaction-information scoring, pair-aware adaptive binning, matrix construction, and RPTE tree search are C++ accelerated with optional OpenMP support. Version 1.1.19 improves RPTE representation clarity, modularizes the classifier implementation, and adds rich on-demand data profiling to the Dash workbench while retaining the complete 1.1.x feature set.
* **Python ergonomics:** the estimator follows the scikit-learn API and works with pandas DataFrames or NumPy arrays.


Current release focus
---------------------

The 1.1.19 release strengthens model representation, maintainability, and data
review without changing the public estimator contract:

* RPTE retains higher-order mined patterns as direct sparse terms and
  consolidates an equivalent leaf-pattern representation into one canonical
  model component. The omitted representation remains available as an audit
  alias, so interpretation and complexity views do not count the same fitted
  component twice.
* The classifier implementation is organized into focused internal modules for
  preparation, training, feature assembly, prediction, interpretation,
  inspection, tuning, and estimator state. ``HUGIMLClassifier`` and
  ``HUGIMLClassifierNative`` keep the same public behavior.
* The Dash workbench includes a compact data-profile preview and an expandable,
  on-demand profile with sample or full-data scope, raw-data and model-input
  views, variable roles, quality findings, missingness, distributions,
  correlations, and target associations.
* A reproducible OpenML-CC18 workflow evaluates 36 classification datasets on
  their official task-defined splits and assembles a static comparison dashboard
  with performance, runtime, and inspection-efficiency summaries.
* Dashboard model configuration uses consistent Default, Guided, and Advanced
  run modes for HUGIML and comparison estimators.

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

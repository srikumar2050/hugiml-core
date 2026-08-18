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
* **Native performance:** computationally intensive mining, transaction construction, interaction-information scoring, pair-aware adaptive binning, matrix construction, and RPTE tree search are C++ accelerated with optional OpenMP support.
* **Python ergonomics:** the estimator follows the scikit-learn API and works with pandas DataFrames or NumPy arrays.


Current release focus
---------------------

The 1.1.21 release adds causal-effect analysis and completes the fitted
downstream transformation contract:

* ``transform(X)`` returns the complete representation consumed by the fitted
  downstream estimator across all feature modes.
* SHAP evaluates the complete downstream representation and supports an
  explicitly requested partial pattern reporting view.
* Causal HUG estimates binary-treatment effects using a shared supervised HUG
  vocabulary, repeated cross-fitting, confidence intervals, doubly robust
  comparisons, and overlap sensitivity analysis.
* The optional Causal Investigation Dashboard provides causal metadata review,
  diagnostics, comparison methods, selected configurations, and robustness
  analysis.

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
   pip install "hugiml-core[causal-dashboard]"  # optional causal investigation UI
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
   causal
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

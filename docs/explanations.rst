Explanations and dashboards
===========================

HUGIML explanations are close to the fitted model. Pattern labels, supports, utilities, information-gain values, coefficients, and active-pattern indicators are exposed directly from the estimator.

Pattern tables
--------------

.. code-block:: python

   pattern_info = clf.get_pattern_info()
   importances = clf.feature_importances()

   print(pattern_info.head(20))
   print(importances.head(20))

Hybrid feature interpretation
-----------------------------

For models that include original features or augmented-pair features in the downstream estimator, ``feature_importances()`` reports the fitted downstream feature space and includes a ``feature_type`` column. Augmented-pair rows include raw formula, standardized formula, observed-row coverage, and raw-scale effect fields.

.. code-block:: python

   print(clf.get_downstream_features())
   print(clf.get_model_composition())
   print(clf.explain_augmented_pair_effects())

``model_summary()`` labels mixed models as downstream-feature summaries rather than pattern-only summaries. Pattern-only inspection remains available through ``get_hug_features()`` and ``get_pattern_info()``.

Interpretability metrics
------------------------

.. code-block:: python

   from hugiml.metrics import compute_all_metrics

   metrics = compute_all_metrics(clf, X_test)
   print(metrics)

Common metrics include pattern count, average pattern length, coverage, mean active patterns per prediction, overlap rate, explanation sparsity, and top-k cumulative contribution.

Plotly dashboard
----------------

The dashboard requires the ``plots`` extra.

.. code-block:: bash

   pip install "hugiml-core[plots]"

.. code-block:: python

   from hugiml.plots import HUGPlotter

   plotter = HUGPlotter(clf)
   plotter.plot_dashboard(
       X_test,
       dataset_name="Credit risk model",
       feature_names_for_profile=["age", "income", "duration"],
       output_path="hugiml_dashboard.html",
   )

Existing example dashboard panels:

.. image:: images/explanation_dashboard_credit.png
   :alt: HUGIML explanation dashboard for credit risk
   :width: 760px

Profile views
-------------

.. code-block:: python

   plotter.plot_marginal_bin_profile("age", X=X_test).show()
   plotter.plot_feature_combinations("age").show()
   plotter.plot_top_patterns(top_n=20).show()
   plotter.plot_active_patterns(X_test, sample_idx=0).show()
   plotter.plot_2d_profile("age", "income", X=X_test).show()

HUGIML versus additive shape models
-----------------------------------

EBM-style models learn smooth additive shape functions. HUGIML learns bin/category and compound-pattern profiles. Both can reveal directional behavior, but HUGIML expresses that behavior as compact intervals and categories that can be reviewed in audit packets.

.. image:: images/shape-profiles-hugiml-vs-ebm.png
   :alt: HUGIML native shape profiles compared with EBM shape functions
   :width: 760px

SHAP bridge
-----------

The optional explainability module can compute SHAP values over the HUG pattern matrix and aggregate them back to original features. This is a pattern-space diagnostic. When the fitted downstream estimator also uses original or augmented-pair features, the SHAP helper reports that the pattern-space result is incomplete unless incomplete reporting is explicitly allowed.

.. code-block:: python

   from hugiml.explainability import (
       HUGPatternExplainer,
       shap_values_from_pattern_matrix,
       aggregate_shap_to_features,
   )

   explainer = HUGPatternExplainer(clf)
   report = explainer.generate_report(model_id="credit-scorer-v1")
   report.save("explainability_report.json")


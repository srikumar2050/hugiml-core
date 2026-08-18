Causal analysis with HUGIML
===========================

HUGIML supports binary-treatment effect estimation through
``CausalHUGClassifier``. The estimator learns one supervised adaptive
vocabulary from pooled baseline covariates and uses the same numeric cutpoints
and feature semantics for the control and treatment outcome models. Each model
then learns its own utilities, patterns, fitted representation, and outcome
probabilities.

Methodology and assumptions
---------------------------

For baseline covariates :math:`X`, binary treatment :math:`T`, and binary
outcome :math:`Y`, the estimator learns potential-outcome probabilities
:math:`m_0(X)` and :math:`m_1(X)`. The conditional treatment effect is

.. math::

   \tau(X) = m_1(X) - m_0(X).

The average treatment effect is the sample mean of :math:`\tau(X)`. For
observational studies, causal interpretation requires consistency, an
appropriate pre-treatment adjustment set, conditional exchangeability, and
adequate treatment overlap. The estimator does not identify a causal graph or
account for unmeasured confounding.

Python workflow
---------------

Pass the treatment column in ``X`` and the binary outcome separately. Baseline
covariates should exclude post-treatment variables, identifiers, and the
treatment column itself.

.. code-block:: python

   from hugiml.causal import CausalHUGClassifier

   model = CausalHUGClassifier(
       treatment="treatment",
       covariates=["age", "income", "risk_score"],
       param_grid="performance",
       scoring="neg_log_loss",
       random_state=42,
   )
   model.fit(frame, frame["outcome"])

   cate = model.cate(frame)
   ate = model.ate(frame)
   p0, p1 = model.predict_potential_outcomes(frame)

``scoring="roc_auc"`` and ``scoring="neg_log_loss"`` are supported for model
selection. Named HUGIML grids include ``performance``, ``interpretability``,
``performance_ho``, and ``interpretability_ho``. The ``quick`` grid is provided
by the dashboard for a responsive initial analysis.

Robustness analysis
-------------------

``summarize_cross_fitted_effects()`` summarizes repeated out-of-fold
potential-outcome predictions and propensity estimates. It reports repeated
out-of-fold ATE, doubly robust AIPW estimates, confidence intervals across
complete cross-fitting runs, and overlap sensitivity under progressively
trimmed propensity ranges.

Repeated-split intervals describe variation across the configured runs. They
do not remove bias from an incomplete adjustment set, unmeasured confounding,
poor overlap, or misspecified treatment and outcome models.

Reusable comparisons
--------------------

``hugiml.causal_baselines`` provides the T-LR, T-XGB, and T-LGBM two-model
estimators and their aligned configuration grids. ``hugiml.causal_evaluation``
provides model fitting, held-out comparison tables, repeated cross-fitting,
doubly robust estimates, and overlap sensitivity summaries. These interfaces
can be used by scripts, notebooks, benchmarks, and the dashboard.

Causal Investigation Dashboard
-------------------------------

Install and launch the optional dashboard with:

.. code-block:: bash

   pip install "hugiml-core[causal-dashboard]"
   hugiml-causal-dashboard

The local interface opens at ``http://localhost:8052/`` by default. It supports
built-in demonstrations and uploaded CSV, TSV, Parquet, or Excel data. A causal
metadata JSON file can declare the treatment, outcome, baseline covariates,
study type, graph nodes and edges, hidden columns, and an optional oracle effect
column for synthetic demonstrations.

The results workspace includes:

* treatment and outcome diagnostics, covariate balance, and propensity overlap;
* potential-outcome, CATE, ATE, uplift, and Qini summaries where defined;
* T-HUG and two-model comparison methods with aligned configuration budgets;
* selected configurations and interpretable HUG regions;
* repeated cross-fitting, doubly robust estimates, confidence intervals, and
  overlap sensitivity.

Uploaded observational data is never assigned synthetic counterfactual truth.
Oracle CATE metrics are shown only when a supplied metadata file identifies an
oracle effect column.

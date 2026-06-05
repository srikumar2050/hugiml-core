Worked notebooks
================

The repository includes self-contained notebook folders under ``notebooks/``. Each folder contains an ``.ipynb`` notebook and, where available, matching ``.py``, ``.html``, data, and metadata files.

.. list-table::
   :header-rows: 1

   * - Folder
     - Focus
   * - ``00_quickstart``
     - End-to-end fitting, pattern extraction, and explanation walkthrough.
   * - ``01_benchmark_baselines``
     - Comparison against common tabular baselines; refreshed in v1.1.7 with the current baseline run and HUGIML grid settings.
   * - ``02_hug_vs_ebm``
     - HUGIML profiles versus EBM-style additive shape functions.
   * - ``03_modeling_special_cases``
     - Multiclass targets, imbalance, high-cardinality categoricals, adaptive binning, and pruning.
   * - ``04_credit_risk``
     - Credit-risk governance using German Credit-style data.
   * - ``05_aml``
     - Anti-money-laundering suspicious transaction patterns.
   * - ``06_mobile_money``
     - Mobile-money fraud review signals.
   * - ``07_basel_ca``
     - Basel capital-adequacy style risk analytics.
   * - ``08_clinical``
     - Clinical classification and healthcare pattern explanations.
   * - ``09_insurance``
     - Insurance underwriting risk-selection patterns.
   * - ``10_medicare``
     - Medicare program-integrity review.
   * - ``11_workforce_analytics``
     - Workforce attrition and HR risk patterns.

Publishing notebooks
--------------------

For Read the Docs, keep notebooks as repository examples and link to their HTML exports rather than executing all notebooks during documentation builds. This keeps documentation builds fast and avoids optional data or long-running benchmark dependencies.


Governance Studio dashboard
===========================

HUGIML Governance Studio is the Streamlit dashboard and workbench for preparing data, configuring HUGIML runs, comparing candidate models, reviewing model evidence, and assembling governance-ready summaries. The dashboard keeps the existing Workbench/Governance layout and exposes the current evidence views for adaptive binning, augmented pairs, interaction-relaxed mining, survivor-led patterns, feature families, validation, monitoring, and governance review.

Installation and launch
-----------------------

Install the dashboard optional dependencies, then launch the installed console command:

.. code-block:: bash

   pip install "hugiml-core[dashboard]"
   hugiml-dashboard

Dashboard arguments can be passed after the command and are forwarded to the Streamlit app.

For source-tree development, the app can also be launched directly:

.. code-block:: bash

   python -m streamlit run src/hugiml/dashboard/app.py
   python -m hugiml.dashboard.app

Workbench workflow
------------------

* Choose a demo dataset or upload a tabular file.
* Review and adjust target, identifier, protected, date, numeric, categorical, and excluded columns.
* Configure HUGIML and optional comparison models.
* Run a single configuration or a small candidate grid.
* Compare models using validation metrics and timing information.
* Inspect feature roles, pattern evidence, case-level explanations, policy checks, and governance summaries.

Evidence views
--------------

The dashboard includes the full workbench experience plus governance evidence panels:

* Overview, validation metrics, and active configuration summary.
* Representation audit across original features, HUG patterns, augmented pairs, and survivor-led pattern metadata.
* Adaptive-binning evidence, including selected bin counts and information-gain summaries where available.
* Augmented-pair traceability for product, absolute-difference, sum, and signed-difference features used by the downstream estimator.
* Interaction-relaxed mining evidence, including survivor-source counts, best partners, and survivor-led pattern flags where available.
* Feature-family review across original, pattern, and augmented representations.
* Pattern inventory, coefficient review, support coverage, and population-coverage summaries.
* Case-level prediction review with active pattern evidence.
* Missingness, drift, monitoring report parsing, and fairness-oriented checks.
* Configuration comparison, representation pruning analysis, and governance summary downloads.

Data inputs
-----------

The dashboard supports demo data and uploaded tabular files including CSV, TSV, Excel, and Parquet. Users can select target, identifier, protected, date, numeric, categorical, and excluded columns before fitting and reviewing a model.

Binary feature handling
-----------------------

Numeric columns with exactly two observed values are treated as categorical indicators during HUGIML preparation. In the dashboard this keeps binary flags, yes/no fields, and encoded indicators visible as discrete evidence rather than interval-binned numeric features.

Installed command behavior
--------------------------

The ``hugiml-dashboard`` console script points to ``hugiml.dashboard.app:main``. When invoked from an installed package, it starts Streamlit with the packaged app path. When already running under Streamlit, it executes the dashboard directly.

Dashboard API modules
---------------------

The following modules back the dashboard and are included in the Sphinx build so Read the Docs regenerates the complete dashboard reference from the source tree.

.. automodule:: hugiml.dashboard.app
   :members:
   :no-index:

.. automodule:: hugiml.dashboard.runner
   :members:
   :no-index:

.. automodule:: hugiml.dashboard.workbench
   :members:
   :no-index:

.. automodule:: hugiml.dashboard.display
   :members:
   :no-index:

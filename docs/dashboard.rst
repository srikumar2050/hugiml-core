Governance Studio dashboard
===========================

HUGIML Governance Studio is a Streamlit dashboard and workbench for model validation, comparison, and governance review workflows. Version 1.1.10 keeps the v1.1.9 dashboard views and adds a richer workbench flow for preparing data, configuring candidate runs, comparing models, drilling into results, and reviewing governance evidence in one place.

Installation and launch
-----------------------

Install the dashboard optional dependencies, then launch the installed console command:

.. code-block:: bash

   pip install "hugiml-core[dashboard]"
   hugiml-dashboard

The command delegates to Streamlit automatically. Dashboard arguments can be passed after the command and are forwarded to the Streamlit app.

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
* Drill into feature roles, pattern evidence, case-level review, policy checks, and governance summaries.

Evidence views
--------------

The dashboard includes the full review surface from v1.1.9 and the expanded v1.1.10 workbench experience:

* Overview and validation metrics.
* Representation audit across original features, HUG patterns, and augmented pairs.
* Pattern inventory and case-level prediction review.
* Data quality, missingness, drift, and fairness-oriented checks.
* Configuration comparison and representation pruning analysis.
* Workflow guidance, demo data, and governance evidence views for model review packets.

Data inputs
-----------

The dashboard supports demo data and uploaded tabular files including CSV, TSV, Excel, and Parquet. Users can select target, identifier, protected, date, numeric, categorical, and excluded columns before fitting and reviewing a model.

Installed command behavior
--------------------------

The ``hugiml-dashboard`` console script points to ``hugiml.dashboard.app:main``. When invoked from an installed package, it starts Streamlit with the packaged app path. When already running under Streamlit, it executes the dashboard directly.

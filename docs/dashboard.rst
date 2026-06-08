Governance Studio dashboard
===========================

HUGIML Governance Studio is a Streamlit dashboard for model validation and audit workflows.

Install the dashboard optional dependencies, then launch the dashboard console command:

.. code-block:: bash

   pip install "hugiml-core[dashboard]"
   hugiml-dashboard

For local source-tree development, the app can also be launched directly with Streamlit or through the module entry point:

.. code-block:: bash

   python -m streamlit run src/hugiml/dashboard/app.py
   python -m hugiml.dashboard.app

Workflow sections
-----------------

* Overview and validation metrics.
* Representation audit across original features, HUG patterns, and augmented pairs.
* Pattern inventory and case-level prediction review.
* Missingness, drift, and fairness-oriented checks.
* Configuration comparison and representation pruning analysis.
* Governance summaries for model review packets.

Data inputs
-----------

The dashboard supports demo data and uploaded tabular files including CSV, TSV, Excel, and Parquet. Users can select target, identifier, and excluded columns before fitting and reviewing a model.

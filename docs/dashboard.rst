Governance Studio dashboard
===========================

HUGIML Governance Studio provides a Dash interface for preparing data, configuring HUGIML and RPTE runs, comparing candidate models, reviewing model evidence, and assembling governance-ready summaries. The existing lightweight Streamlit interface remains available. Both interfaces use the same model, runner, and governance evidence layers.

Installation and launch
-----------------------

Install the dashboard optional dependencies, then launch the installed console command:

.. code-block:: bash

   pip install "hugiml-core[dashboard]"
   hugiml-dashboard                    # Dash, the default interface
   hugiml-dashboard --ui light         # lightweight Streamlit interface

The launcher accepts ``--host``, ``--port``, ``--debug``, ``--no-open``,
``--cv``, and ``--random-state``. For source-tree development, the interfaces
can also be launched directly:

.. code-block:: bash

   python -m hugiml.dashboard.dash_app
   python -m streamlit run src/hugiml/dashboard/app.py

Workbench workflow
------------------

* Choose a demo dataset or upload a tabular file.
* Review and adjust target, identifier, protected, date, numeric, categorical, and excluded columns.
* Configure HUGIML, the optional RPTE downstream model, and comparison models; advanced controls include adaptive binning, representation choices, and RPTE tree-growth settings.
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
* Feature-family review across original, pattern, augmented, RPTE tree-leaf, and RPTE direct-term representations.
* RPTE governance views for backend choice, accepted tree paths, direct source terms, coefficients, support, provenance, complexity, and instance inspection.
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

The ``hugiml-dashboard`` console script points to
``hugiml.dashboard.launcher:main``. It starts the Dash interface by default and
selects the Streamlit interface when ``--ui light`` is supplied or
``HUGIML_UI=light`` is set.

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

.. automodule:: hugiml.dashboard.dash_app
   :members:
   :no-index:

.. automodule:: hugiml.dashboard.components.rpte_governance
   :members:
   :no-index:

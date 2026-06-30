LLM assistant add-on
====================

The optional HUGIML LLM assistant adds a natural-language interface for common
modeling and governance workflows. It can answer package-usage questions from
the local documentation index, run supported HUGIML actions, summarize fitted
model evidence, and produce lightweight HTML outputs for review.

Installation
------------

Install the optional dependencies with one of the Streamlit-oriented extras:

.. code-block:: bash

   pip install "hugiml-core[llm]"
   # or install the dashboard stack if you want both Streamlit UIs
   pip install "hugiml-core[dashboard]"

Launch the assistant from an installed environment:

.. code-block:: bash

   hugiml-llm

Useful CLI commands include:

.. code-block:: bash

   hugiml-llm status
   hugiml-llm list-datasets
   hugiml-llm chat --dataset churn_synthetic --no-llm
   hugiml-llm ask "tune churn_synthetic and explain the strongest patterns" --dataset churn_synthetic
   hugiml-llm demo-html --output examples/governance_qna_churn.html

Fast and thinking response modes
--------------------------------

The assistant supports two response styles:

* **Fast** uses deterministic routing and local HUGIML summaries. This is the
  best default for quick API, dataset, tuning, and fitted-run questions.
* **Thinking** can use an Ollama-backed writing pass when an installed local
  model is available. This is useful for longer governance-style explanations
  or more polished narrative summaries.

The deterministic path remains available without Ollama. When Ollama is used,
the add-on checks lightweight local model options and keeps execution bounded by
the supported HUGIML action schema.

Supported workflows
-------------------

The add-on is designed around safe, structured workflows rather than unrestricted
source editing. Supported actions include dataset inspection, model fitting,
hyperparameter tuning, model comparison, pattern and feature-importance review,
controlled pruning summaries, governance report generation, prediction review,
and local documentation Q&A.

Built-in assets
---------------

The package includes small built-in demonstration datasets and model-profile
configuration under ``hugiml.llm`` package data. User datasets can be registered
through the UI or CLI, and installed environments store user-side LLM assets
under the configured HUGIML LLM home directory.

API modules
-----------

See :doc:`api` for the public Python modules in ``hugiml.llm``.

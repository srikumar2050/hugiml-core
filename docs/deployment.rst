Deployment
==========

FastAPI inference server
------------------------

A FastAPI-based inference server is included for containerized scoring.

.. code-block:: bash

   docker build -t hugiml-core:latest -f docker/Dockerfile .
   docker run -p 8080:8080 -v /path/to/models:/models hugiml-core:latest

Example request:

.. code-block:: bash

   curl -s -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"instances": [{"age": 35, "savings": "moderate"}]}'

Kubernetes
----------

A starter manifest is provided in ``kubernetes/deployment.yaml``. Review CPU/memory limits, model volume paths, network policy, and secrets before production use.

Production checklist
--------------------

* Pin ``hugiml-core`` and dependency versions.
* Save the trained model with ``save_model`` and record the model schema version.
* Export a model card and audit artifact.
* Capture calibration metrics and drift thresholds.
* Exercise prediction-time schema validation against representative production payloads.
* Configure monitoring, latency budgets, and rollback procedures.
* Validate any pattern-pruning actions and retain the pruning audit trail.


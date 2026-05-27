Monitoring, drift, and serialization
====================================

Prediction monitoring
---------------------

.. code-block:: python

   clf.enable_monitoring(window_size=1000)
   clf.predict_proba(X_new)

   print(clf.monitor.stats())
   print(clf.monitor.report())

The monitor tracks prediction volume, latency, probability summaries, and rolling-window state in a thread-safe object attached to the classifier.

Drift detection
---------------

.. code-block:: python

   drift_report = clf.detect_drift(X_new, current_labels=y_new)
   print(drift_report)

   psi_by_feature = clf.get_drift_psi(X_new)
   print(psi_by_feature)

The built-in detector combines Population Stability Index, symmetric KL divergence, and optional label drift when current labels are available.

.. image:: images/native-missing-value-schemes.png
   :alt: Native missing-value schemes across tabular models
   :width: 760px

Serialization
-------------

.. code-block:: python

   from hugiml.serialization import save_model, load_model, generate_sbom

   save_model(clf, "model.hugiml")
   restored = load_model("model.hugiml")

   sbom = generate_sbom(restored, output_path="sbom.json")

The serializer uses a versioned ZIP/JSON/NumPy model format and avoids unrestricted pickle loading for the current schema.

Telemetry
---------

Optional OpenTelemetry and Prometheus hooks are available with the ``telemetry`` extra.

.. code-block:: bash

   pip install "hugiml-core[telemetry]"

.. code-block:: python

   from hugiml.telemetry import instrument_classifier

   instrumented = instrument_classifier(clf, service_name="credit-risk-hugiml")
   instrumented.predict_proba(X_new)


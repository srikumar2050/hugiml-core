Governance, pruning, and model cards
====================================

HUGIML is designed for audit-heavy workflows where model reviewers need to inspect rules, trace feature lineage, remove unsafe patterns, and package validation artifacts.

Model cards
-----------

.. code-block:: python

   from hugiml.governance import generate_model_card

   card = generate_model_card(
       clf,
       model_id="credit-scorer-v1.0.0",
       intended_use="Credit risk assessment for SME lending.",
       training_data_description="German Credit dataset, 1000 samples",
   )

   print(card.to_markdown())
   card.save("model_card.json")
   card.save("model_card.md", fmt="markdown")

A starter template is included at ``docs/model_card_template.md`` and should be copied into validation packets or repository governance folders as needed.

.. image:: images/model-card-governance.png
   :alt: Model-card-ready HUGIML explanations
   :width: 760px

Pattern pruning
---------------

Analysts may need to remove patterns that reference protected attributes, have excessive drift, encode operationally invalid logic, or fail model-risk review. ``PatternEditor`` provides a controlled remove/refit/calibrate/finalize workflow with a JSON audit trail.

.. code-block:: python

   from hugiml.pruning import PatternEditor

   editor = PatternEditor(clf, operator_name="risk-team")
   print(editor.list_patterns().head(10))

   editor.remove([3, 7], reason="references protected attribute")
   editor.remove_by_keyword("postcode", reason="high PSI during monitoring")
   editor.remove_low_support(min_support=0.01, reason="low-support noise")

   editor.refit(X_train, y_train)
   editor.calibrate(X_calibration, y_calibration, method="isotonic")

   governed_clf = editor.finalize()
   editor.save_audit_report("pattern_pruning_audit.json")

Audit artifacts
---------------

.. code-block:: python

   from hugiml.governance import GovernanceMetadata, AuditArtifact

   metadata = GovernanceMetadata(
       model_id="credit-scorer-v1",
       owner="model-risk-team",
       purpose="Credit application risk ranking",
   )

   audit = AuditArtifact.from_model(clf, metadata=metadata)
   audit.save("training_audit.json")

Calibration
-----------

.. code-block:: python

   from hugiml.calibration import evaluate_calibration, reliability_diagram_data

   result = evaluate_calibration(y_test, proba[:, 1])
   print(result.summary())

   diagram = reliability_diagram_data(y_test, proba[:, 1], n_bins=10)


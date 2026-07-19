from __future__ import annotations

import numpy as np
import pandas as pd

from hugiml.llm.orchestrator import HUGIMLActionOrchestrator, ModelSession
from hugiml.llm.schemas import ActionRequest


class _RPTEEvidenceModel:
    def feature_importances(self):
        raise AttributeError("RPTE uses prediction terms")

    def get_pattern_info(self):
        return pd.DataFrame()

    def rpte_rule_table(self):
        return [
            {
                "class": 1,
                "tree_index": 0,
                "leaf_index": 2,
                "backend": "sequential_default",
                "conditions": [{"raw_condition": "age > 50"}],
                "final_logistic_coefficient": 0.8,
                "support_count": 30,
            },
            {
                "class": 1,
                "tree_index": None,
                "leaf_index": 4,
                "backend": "direct_hugiml_feature",
                "term_role": "direct_source_term",
                "conditions": [{"raw_condition": "income"}],
                "final_logistic_coefficient": -0.3,
                "support_count": None,
            },
        ]


def test_model_explanation_labels_leaf_and_direct_source_terms():
    orchestrator = object.__new__(HUGIMLActionOrchestrator)
    session = ModelSession(
        session_id="run-1",
        dataset="demo",
        target="target",
        info={},
        X_train=pd.DataFrame({"age": [40, 60]}),
        X_test=pd.DataFrame({"age": [45]}),
        y_train=np.asarray([0, 1]),
        y_test=np.asarray([0]),
        model=_RPTEEvidenceModel(),
        metrics={"roc_auc": 0.8},
    )
    orchestrator.sessions = {session.session_id: session}
    orchestrator.last_session_id = session.session_id

    result = orchestrator._action_explain_model(
        ActionRequest(action="explain_model", session_id=session.session_id, limit=10)
    )

    rows = result.tables["rpte_rule_conjunctions"]
    assert [row["term_type"] for row in rows] == ["rpte_leaf", "direct_source_term"]
    assert rows[0]["conjunction"] == "age > 50"
    assert rows[1]["conjunction"] == "income"

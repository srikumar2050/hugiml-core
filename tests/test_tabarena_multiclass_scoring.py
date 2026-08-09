from __future__ import annotations

import importlib

import numpy as np
from sklearn.metrics import log_loss


class _ProbabilityModel:
    def __init__(self, probabilities: np.ndarray) -> None:
        self.probabilities = probabilities

    def predict_proba(self, X):
        return self.probabilities


class _PreparedCandidate:
    def __init__(self, probabilities: np.ndarray) -> None:
        self.classes_ = np.arange(probabilities.shape[1])
        self.model_ = _ProbabilityModel(probabilities)

    def _apply_lr_downstream_canonical_transform(self, X):
        return X


class _PublicCandidate(_PreparedCandidate):
    max_predict_ms = None

    def _is_constant_prior_fallback_active(self) -> bool:
        return False

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def test_multiclass_log_loss_accepts_validation_fold_with_absent_class() -> None:
    tuning = importlib.import_module("hugiml._classifier_tuning")
    probabilities = np.asarray(
        [
            [0.04, 0.61, 0.15, 0.10, 0.10],
            [0.03, 0.12, 0.65, 0.10, 0.10],
            [0.02, 0.08, 0.10, 0.70, 0.10],
            [0.03, 0.08, 0.09, 0.10, 0.70],
        ]
    )
    y_validation = np.asarray([1, 2, 3, 4])
    expected = -log_loss(y_validation, probabilities, labels=np.arange(5))

    prepared = tuning._hugiml_score_prepared_downstream(
        _PreparedCandidate(probabilities), np.zeros((4, 1)), y_validation, "neg_log_loss"
    )
    public = tuning._hugiml_score_model_for_tune(
        _PublicCandidate(probabilities), np.zeros((4, 1)), y_validation, "neg_log_loss"
    )

    assert prepared == expected
    assert public == expected

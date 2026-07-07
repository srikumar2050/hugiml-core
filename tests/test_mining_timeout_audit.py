import pickle
import time
from types import SimpleNamespace

import numpy as np

import hugiml.classifier as classifier_mod
from hugiml import HUGIMLClassifier


class _FakeCore:
    @staticmethod
    def mine_patterns(td, y_train, n_cls, K, L, G, timeout_s):
        return [SimpleNamespace(utility=1.0, items=[1, 2], ig=0.5)]

    @staticmethod
    def mine_patterns_relaxed(td, y_train, n_cls, K, L, G, relaxed_cols, timeout_s):
        return [SimpleNamespace(utility=1.0, items=[1, 2], ig=0.5)]


def test_max_mining_seconds_is_sklearn_param_and_pickle_safe():
    clf = HUGIMLClassifier(L=6, max_mining_seconds=1800, execution_mode="audit")
    params = clf.get_params()
    assert params["max_mining_seconds"] == 1800

    clf.mining_audit_log_ = [
        {
            "attempt_index": 1,
            "label": "full",
            "K": 100,
            "L": 6,
            "G": 0.0,
            "timeout_s": 1800.0,
            "status": "ok_timeout_partial",
            "n_patterns_returned": 100,
        }
    ]
    restored = pickle.loads(pickle.dumps(clf))
    assert restored.max_mining_seconds == 1800
    assert restored.mining_audit_log_[0]["L"] == 6
    assert restored.get_mining_audit_log().iloc[0]["status"] == "ok_timeout_partial"


def test_mine_with_fallback_records_audit_log(monkeypatch):
    monkeypatch.setattr(classifier_mod, "_core", _FakeCore)
    clf = HUGIMLClassifier(L=6, G=0.0, topK=10, max_mining_seconds=1800)
    clf.td_ = object()
    deadline = time.perf_counter() + 1800.0

    pats = clf._mine_with_fallback(np.asarray([0, 1], dtype=np.int64), 2, 10, deadline)

    assert len(pats) == 1
    log = clf.get_mining_audit_log()
    assert list(log["label"]) == ["full"]
    assert int(log.iloc[0]["L"]) == 6
    assert float(log.iloc[0]["timeout_s"]) > 1700.0
    assert log.iloc[0]["status"] == "ok"
    assert int(log.iloc[0]["n_patterns_returned"]) == 1


def test_interaction_relaxed_audit_log_records_relaxed_cols(monkeypatch):
    monkeypatch.setattr(classifier_mod, "_core", _FakeCore)
    clf = HUGIMLClassifier(
        L=5,
        G=0.0,
        topK=10,
        max_mining_seconds=1800,
        interaction_relaxed_mining=True,
        augmented_pair_transforms=False,
    )
    clf.td_ = object()
    deadline = time.perf_counter() + 1800.0

    clf._mine_with_fallback(
        np.asarray([0, 1], dtype=np.int64),
        2,
        10,
        deadline,
        relaxed_cols=[0, 1, 2, 3],
    )

    row = clf.get_mining_audit_log().iloc[0]
    assert bool(row["interaction_relaxed_mining"]) is True
    assert int(row["relaxed_cols_count"]) == 4
    assert int(row["L"]) == 5

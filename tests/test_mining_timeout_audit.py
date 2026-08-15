import pickle
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import hugiml._classifier_training as training_mod
from hugiml import HUGIMLClassifier
from hugiml.exceptions import HUGIMLDegradedWarning, HUGIMLMemoryError, HUGIMLTimeoutError


class _FakeCore:
    @staticmethod
    def mine_patterns(td, y_train, n_cls, K, L, G, timeout_s):
        return [SimpleNamespace(utility=1.0, items=[1, 2], ig=0.5)]

    @staticmethod
    def mine_patterns_relaxed(td, y_train, n_cls, K, L, G, relaxed_cols, timeout_s):
        return [SimpleNamespace(utility=1.0, items=[1, 2], ig=0.5)]


def test_max_mining_seconds_is_sklearn_param_and_pickle_safe():
    clf = HUGIMLClassifier(
        L=6,
        max_mining_seconds=1800,
        mining_degradation_policy="raise",
        execution_mode="audit",
    )
    params = clf.get_params()
    assert params["max_mining_seconds"] == 1800
    assert params["mining_degradation_policy"] == "raise"

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
    assert restored.mining_degradation_policy == "raise"
    assert restored.mining_audit_log_[0]["L"] == 6
    assert restored.get_mining_audit_log().iloc[0]["status"] == "ok_timeout_partial"


def test_mine_with_fallback_records_audit_log(monkeypatch):
    monkeypatch.setattr(training_mod, "_core", _FakeCore)
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
    monkeypatch.setattr(training_mod, "_core", _FakeCore)
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


class _MemoryThenSuccessCore:
    calls = []

    @classmethod
    def mine_patterns(cls, td, y_train, n_cls, K, L, G, timeout_s):
        cls.calls.append((K, L, G))
        if len(cls.calls) == 1:
            raise MemoryError("simulated allocation pressure")
        return [SimpleNamespace(utility=1.0, items=[1], ig=0.5)]

    @classmethod
    def mine_patterns_relaxed(cls, td, y_train, n_cls, K, L, G, relaxed_cols, timeout_s):
        return cls.mine_patterns(td, y_train, n_cls, K, L, G, timeout_s)


class _AlwaysMemoryCore:
    calls = []

    @classmethod
    def mine_patterns(cls, td, y_train, n_cls, K, L, G, timeout_s):
        cls.calls.append((K, L, G))
        raise MemoryError("simulated allocation pressure")

    @classmethod
    def mine_patterns_relaxed(cls, td, y_train, n_cls, K, L, G, relaxed_cols, timeout_s):
        return cls.mine_patterns(td, y_train, n_cls, K, L, G, timeout_s)


class _SequenceClock:
    def __init__(self, values):
        self._values = iter(values)
        self._last = 0.0

    def perf_counter(self):
        try:
            self._last = float(next(self._values))
        except StopIteration:
            pass
        return self._last


class _RecordingCore:
    timeout_args = []

    @classmethod
    def mine_patterns(cls, td, y_train, n_cls, K, L, G, timeout_s):
        cls.timeout_args.append(float(timeout_s))
        return [SimpleNamespace(utility=1.0, items=[1], ig=0.5)]

    @classmethod
    def mine_patterns_relaxed(cls, td, y_train, n_cls, K, L, G, relaxed_cols, timeout_s):
        return cls.mine_patterns(td, y_train, n_cls, K, L, G, timeout_s)


def test_degradation_policy_validation():
    clf = HUGIMLClassifier(mining_degradation_policy="unexpected")
    with pytest.raises(Exception, match="mining_degradation_policy"):
        clf._validate_params()


def test_allow_policy_records_reduced_memory_recovery(monkeypatch):
    _MemoryThenSuccessCore.calls = []
    monkeypatch.setattr(training_mod, "_core", _MemoryThenSuccessCore)
    clf = HUGIMLClassifier(
        L=4,
        G=0.01,
        topK=20,
        mining_degradation_policy="allow",
    )
    clf.td_ = object()

    with pytest.warns(HUGIMLDegradedWarning, match="memory pressure"):
        patterns = clf._mine_with_fallback(np.asarray([0, 1]), 2, 20, None)

    assert len(patterns) == 1
    assert _MemoryThenSuccessCore.calls[:2] == [(20, 4, 0.01), (10, 4, 0.01)]
    assert clf._effective_mining_config_["K"] == 10
    assert clf.mining_audit_log_[0]["status"] == "memory_error"
    assert clf.mining_audit_log_[1]["status"] == "ok"
    assert hasattr(clf, "_degraded_reason")


def test_raise_policy_rejects_memory_recovery(monkeypatch):
    _AlwaysMemoryCore.calls = []
    monkeypatch.setattr(training_mod, "_core", _AlwaysMemoryCore)
    clf = HUGIMLClassifier(
        L=4,
        topK=20,
        mining_degradation_policy="raise",
    )
    clf.td_ = object()

    with pytest.raises(HUGIMLMemoryError, match="does not permit"):
        clf._mine_with_fallback(np.asarray([0, 1]), 2, 20, None)

    assert len(_AlwaysMemoryCore.calls) == 1
    assert clf.mining_audit_log_[0]["status"] == "memory_error"


def test_fallback_k_never_exceeds_requested_budget(monkeypatch):
    _AlwaysMemoryCore.calls = []
    monkeypatch.setattr(training_mod, "_core", _AlwaysMemoryCore)
    clf = HUGIMLClassifier(
        L=4,
        topK=8,
        mining_degradation_policy="allow",
    )
    clf.td_ = object()

    with pytest.warns(HUGIMLDegradedWarning, match="All staged mining attempts"):
        patterns = clf._mine_with_fallback(np.asarray([0, 1]), 2, 8, None)

    assert patterns == []
    attempted_k = [row[0] for row in _AlwaysMemoryCore.calls]
    assert attempted_k == [8, 4, 2, 2]
    assert max(attempted_k) <= 8
    assert clf._effective_mining_config_["K"] == 0


def test_raise_policy_rejects_partial_timeout_result(monkeypatch):
    _RecordingCore.timeout_args = []
    monkeypatch.setattr(training_mod, "_core", _RecordingCore)
    clock = _SequenceClock([99.0, 100.0, 100.0])
    monkeypatch.setattr(training_mod, "time", SimpleNamespace(perf_counter=clock.perf_counter))
    clf = HUGIMLClassifier(
        L=3,
        topK=12,
        mining_degradation_policy="raise",
    )
    clf.td_ = object()

    with pytest.raises(HUGIMLTimeoutError, match="rejects partial timeout results"):
        clf._mine_with_fallback(np.asarray([0, 1]), 2, 12, 100.0)

    assert _RecordingCore.timeout_args == [pytest.approx(1.0)]
    assert clf.mining_audit_log_[0]["status"] == "ok_timeout_partial"
    assert clf.mining_audit_log_[0]["n_patterns_returned"] == 1


def test_allow_policy_records_partial_timeout_result(monkeypatch):
    _RecordingCore.timeout_args = []
    monkeypatch.setattr(training_mod, "_core", _RecordingCore)
    clock = _SequenceClock([99.0, 100.0, 100.0])
    monkeypatch.setattr(training_mod, "time", SimpleNamespace(perf_counter=clock.perf_counter))
    clf = HUGIMLClassifier(
        L=3,
        topK=12,
        mining_degradation_policy="allow",
    )
    clf.td_ = object()

    with pytest.warns(HUGIMLDegradedWarning, match="partial pattern"):
        patterns = clf._mine_with_fallback(np.asarray([0, 1]), 2, 12, 100.0)

    assert len(patterns) == 1
    assert _RecordingCore.timeout_args == [pytest.approx(1.0)]
    assert clf._effective_mining_config_["K"] == 12
    assert clf._effective_mining_config_["status"] == "ok_timeout_partial"


def test_expired_deadline_never_uses_zero_native_timeout(monkeypatch):
    _RecordingCore.timeout_args = []
    monkeypatch.setattr(training_mod, "_core", _RecordingCore)
    clock = _SequenceClock([100.0, 100.0, 100.0])
    monkeypatch.setattr(training_mod, "time", SimpleNamespace(perf_counter=clock.perf_counter))
    clf = HUGIMLClassifier(
        L=3,
        topK=12,
        mining_degradation_policy="allow",
    )
    clf.td_ = object()

    with pytest.warns(HUGIMLDegradedWarning, match="deadline was exhausted"):
        patterns = clf._mine_with_fallback(np.asarray([0, 1]), 2, 12, 100.0)

    assert len(patterns) == 1
    assert _RecordingCore.timeout_args == [5.0]
    assert clf.mining_audit_log_[0]["status"] == "deadline_exhausted_before_attempt"
    assert clf.mining_audit_log_[1]["label"] == "minimal_after_timeout"
    assert all(timeout_s > 0.0 for timeout_s in _RecordingCore.timeout_args)



def test_no_deadline_uses_explicit_unlimited_native_timeout(monkeypatch):
    _RecordingCore.timeout_args = []
    monkeypatch.setattr(training_mod, "_core", _RecordingCore)
    clf = HUGIMLClassifier(L=3, topK=12)
    clf.td_ = object()

    patterns = clf._mine_with_fallback(np.asarray([0, 1]), 2, 12, None)

    assert len(patterns) == 1
    assert _RecordingCore.timeout_args == [0.0]
    assert clf.mining_audit_log_[0]["deadline_enabled"] is False
    assert clf.mining_audit_log_[0]["status"] == "ok"

def test_refit_clears_prior_degradation_and_audit_state():
    clf = HUGIMLClassifier()
    clf._degraded_reason = "prior run"
    clf.mining_audit_log_ = [{"status": "old"}]
    clf.mining_audit_config_ = {"requested_K": 999}
    clf._effective_mining_config_ = {"K": 999}

    with pytest.raises(ValueError, match="does not support sparse input"):
        clf._fit_impl(csr_matrix(np.eye(2)), np.asarray([0, 1]))

    assert not hasattr(clf, "_degraded_reason")
    assert clf.mining_audit_log_ == []
    assert clf.mining_audit_config_ == {}
    assert not hasattr(clf, "_effective_mining_config_")

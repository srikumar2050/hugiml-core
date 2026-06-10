from __future__ import annotations

import numpy as np
import pytest

from hugiml import HUGIMLClassifierNative
import hugiml.classifier as classifier_module


def _binary_data(n: int = 120, p: int = 4):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0).astype(int)
    return X, y


def _clf(execution_mode: str):
    return HUGIMLClassifierNative(
        B=3,
        L=1,
        G=0.01,
        topK=10,
        execution_mode=execution_mode,
        adaptive_binning=False,
        n_jobs=1,
        use_hotpath=False,
    )


def test_production_fit_does_not_build_drift_baseline(monkeypatch):
    X, y = _binary_data()
    calls = []
    original_fit_baseline = classifier_module.DriftDetector.fit_baseline

    def fit_baseline_spy(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original_fit_baseline(self, *args, **kwargs)

    monkeypatch.setattr(classifier_module.DriftDetector, "fit_baseline", fit_baseline_spy)
    clf = _clf("production").fit(X, y)

    assert calls == []
    assert not hasattr(clf, "_drift_det")
    assert clf.fit_metadata_.stage_times_ms.get("drift_baseline") == 0.0
    assert clf.predict(X[:5]).shape == (5,)


@pytest.mark.parametrize("method_name", ["detect_drift", "get_drift_psi"])
def test_production_drift_methods_raise_audit_artifact_error(method_name):
    X, y = _binary_data()
    clf = _clf("production").fit(X, y)

    with pytest.raises(RuntimeError, match="execution_mode='production'"):
        getattr(clf, method_name)(X[:20])


def test_audit_fit_builds_and_retains_drift_baseline():
    X, y = _binary_data()
    clf = _clf("audit").fit(X, y)

    assert getattr(clf, "_drift_det", None) is not None
    assert clf.fit_metadata_.stage_times_ms.get("drift_baseline", 0.0) >= 0.0
    assert isinstance(clf.get_drift_psi(X[:20]), dict)

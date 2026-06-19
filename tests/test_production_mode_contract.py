import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hugiml import HUGIMLClassifier, HUGIMLClassifierNative
from hugiml.exceptions import HUGIMLSerializationError
from hugiml.serialization import load_model, save_model


def _data(n=300):
    rng = np.random.default_rng(123)
    X = pd.DataFrame(
        {
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "cat": rng.choice(["a", "b"], size=n),
        }
    )
    y = (
        (X["x0"].to_numpy() + 0.4 * (X["cat"] == "a").to_numpy() + 0.2 * rng.normal(size=n)) > 0
    ).astype(int)
    return X, y


@pytest.mark.parametrize(
    "feature_mode", ["patterns_only", "original_plus_patterns", "original_plus_interactions"]
)
def test_production_save_load_predict_and_summary(feature_mode):
    X, y = _data()
    clf = HUGIMLClassifier(
        B=3,
        L=2,
        G=0.0,
        topK=10,
        adaptive_binning=False,
        feature_mode=feature_mode,
        topk_budget_strict=False,
        execution_mode="production",
        use_hotpath=False,
        interaction_relaxed_mining=False,
    ).fit(X, y)

    assert isinstance(clf, HUGIMLClassifierNative)
    assert not hasattr(clf, "x_train_hup_")
    assert clf.get_transformed_shape()[0] == len(X)
    assert "not retained in production mode" in clf.model_summary()
    proba = clf.predict_proba(X.head(9))
    assert np.isfinite(proba).all()

    fi = clf.feature_importances()
    assert "audit_note" in fi.columns
    assert fi["audit_note"].astype(str).str.contains("execution_mode='production'").all()

    with pytest.raises(RuntimeError, match="execution_mode='production'"):
        clf.detect_drift(X.head(10))
    with pytest.raises(RuntimeError, match="execution_mode='production'"):
        clf.get_drift_psi(X.head(10))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.hugiml"
        save_model(clf, path)
        loaded = load_model(path)
        assert loaded.execution_mode == "production"
        assert not hasattr(loaded, "x_train_hup_")
        assert loaded.get_transformed_shape()[0] == len(X)
        assert "not retained in production mode" in loaded.model_summary()
        assert np.allclose(proba, loaded.predict_proba(X.head(9)))


def test_production_fused_l1_hotpath_save_load_predict_and_summary():
    core = pytest.importorskip("hugiml._hugiml_core")
    if not hasattr(core, "prepare_and_mine_l1_adaptive"):
        pytest.skip("installed native extension does not expose fused adaptive L1 helper")
    X, y = _data()
    clf = HUGIMLClassifier(
        B=-1,
        L=1,
        G=0.0,
        topK=10,
        adaptive_binning=True,
        feature_mode="patterns_only",
        execution_mode="production",
        use_hotpath=True,
    ).fit(X[["x0", "x1"]], y)

    assert not hasattr(clf, "x_train_hup_")
    proba = clf.predict_proba(X[["x0", "x1"]].head(9))
    assert np.isfinite(proba).all()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.hugiml"
        save_model(clf, path)
        loaded = load_model(path)
        assert loaded.execution_mode == "production"
        assert not hasattr(loaded, "x_train_hup_")
        assert np.allclose(proba, loaded.predict_proba(X[["x0", "x1"]].head(9)))


def test_invalid_execution_mode_rejected_on_load(monkeypatch):
    monkeypatch.delenv("HUGIML_MODEL_HMAC_KEY", raising=False)
    monkeypatch.delenv("HUGIML_REQUIRE_MODEL_HMAC", raising=False)
    X, y = _data()
    clf = HUGIMLClassifier(
        B=3,
        L=1,
        G=0.0,
        topK=5,
        adaptive_binning=False,
        execution_mode="production",
        use_hotpath=False,
    ).fit(X[["x0", "x1"]], y)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.hugiml"
        bad = Path(tmp) / "bad.hugiml"
        save_model(clf, path)
        with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(bad, "w") as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename == "clf_fit.json":
                    state = json.loads(data)
                    state["execution_mode"] = "staging"
                    data = json.dumps(state).encode("utf-8")
                zout.writestr(info, data)
        with pytest.raises(HUGIMLSerializationError):
            load_model(bad)


def test_invalid_execution_mode_rejected_from_clf_init_on_load(monkeypatch):
    monkeypatch.delenv("HUGIML_MODEL_HMAC_KEY", raising=False)
    monkeypatch.delenv("HUGIML_REQUIRE_MODEL_HMAC", raising=False)
    X, y = _data()
    clf = HUGIMLClassifier(
        B=3,
        L=1,
        G=0.0,
        topK=5,
        adaptive_binning=False,
        execution_mode="production",
        use_hotpath=False,
    ).fit(X[["x0", "x1"]], y)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.hugiml"
        bad = Path(tmp) / "bad_init.hugiml"
        save_model(clf, path)
        with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(bad, "w") as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename == "clf_init.json":
                    state = json.loads(data)
                    state["execution_mode"] = "staging"
                    data = json.dumps(state).encode("utf-8")
                zout.writestr(info, data)
        with pytest.raises(HUGIMLSerializationError):
            load_model(bad)

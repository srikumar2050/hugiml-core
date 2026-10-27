"""Model trust boundaries and compatible serialization."""

import io
import json
import pickle
import struct
import zipfile

import numpy as np
import pandas as pd
import pytest

from hugiml import HUGIMLClassifier
from hugiml import serialization as ser
from hugiml.exceptions import HUGIMLSerializationError, HUGIMLValidationError


@pytest.mark.parametrize("name", ["forest", "tree", "sgd", "svc", "nb"])
def test_supported_estimator_fallback_parity(name):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    X = np.random.default_rng(19).normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)
    model = {
        "forest": RandomForestClassifier(n_estimators=3, random_state=19),
        "tree": DecisionTreeClassifier(max_depth=3, random_state=19),
        "sgd": SGDClassifier(loss="log_loss", random_state=19),
        "svc": SVC(probability=True, random_state=19),
        "nb": GaussianNB(),
    }[name].fit(X, y)
    config, arrays = ser._serialize_estimator(model)
    loaded = ser._deserialize_estimator(config, arrays)
    np.testing.assert_array_equal(model.predict_proba(X), loaded.predict_proba(X))


class Executable:
    def __reduce__(self):
        return eval, ("40 + 2",)


@pytest.fixture
def unsigned(monkeypatch):
    monkeypatch.delenv("HUGIML_MODEL_HMAC_KEY", raising=False)
    monkeypatch.delenv("HUGIML_REQUIRE_MODEL_HMAC", raising=False)


def test_legacy_executable_rejected(tmp_path, unsigned):
    path = tmp_path / "model.hugiml"
    path.write_bytes(b"HUGI" + struct.pack("<I", 2) + bytes(32) + pickle.dumps(Executable()))
    with pytest.raises(HUGIMLSerializationError, match="not allowed"):
        ser.load_model(path)


def test_v3_pickle_fallback_rejects_executable():
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        ser._deserialize_estimator(
            {"_pickle_fallback": True},
            {"pickle_payload": np.frombuffer(pickle.dumps(Executable()), dtype=np.uint8)},
        )


@pytest.mark.parametrize(
    "path", ["builtins.eval", "pandas.read_pickle", "numpy.load", "unknown.HUGIMLClassifier"]
)
def test_constructor_registry_rejects_functions(path):
    with pytest.raises(HUGIMLSerializationError, match="not approved"):
        ser._reconstruct_param_value(
            "base_estimator", {"__hugiml_estimator__": True, "class": path, "params": {}}
        )


def test_pickle_extension_rejected():
    with pytest.raises(pickle.UnpicklingError, match="extension"):
        ser._safe_unpickle(b"\x80\x02\x82\x01.")


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize("length", [1, 2])
def test_legacy_model_prediction_parity(tmp_path, unsigned, protocol, length):
    X = pd.DataFrame(np.random.default_rng(4).normal(size=(60, 3)), columns=list("abc"))
    y = (X.a > 0).astype(int)
    model = HUGIMLClassifier(L=length, topK=5, adaptive_binning=True).fit(X, y)
    path = tmp_path / "model.hugiml"
    path.write_bytes(
        b"HUGI" + struct.pack("<I", 2) + bytes(32) + pickle.dumps(model, protocol=protocol)
    )
    restored = ser.load_model(path)
    np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))


def test_v3_constructor_rejected_before_invocation(tmp_path, unsigned, monkeypatch):
    X = pd.DataFrame({"a": range(40), "b": np.arange(40) % 3})
    model = HUGIMLClassifier(L=1).fit(X, np.arange(40) % 2)
    path = tmp_path / "model.hugiml"
    ser.save_model(model, path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    config = json.loads(members["clf_init.json"])
    config["base_estimator"] = {
        "__hugiml_estimator__": True,
        "class": "pandas.read_pickle",
        "params": {},
    }
    members["clf_init.json"] = json.dumps(config).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)

    def forbidden(*args, **kwargs):
        pytest.fail("Unapproved constructor was invoked")

    monkeypatch.setattr(pd, "read_pickle", forbidden)
    with pytest.raises(HUGIMLSerializationError, match="not approved"):
        ser.load_model(path)


def test_archive_limit_precedes_expansion(tmp_path, monkeypatch, unsigned):
    path = tmp_path / "large.hugiml"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("extra", bytes(4096))
    monkeypatch.setenv("HUGIML_MODEL_MAX_BYTES", "1024")

    def forbidden(*args, **kwargs):
        pytest.fail("Oversized member was read")

    monkeypatch.setattr(zipfile.ZipFile, "read", forbidden)
    with pytest.raises(HUGIMLSerializationError, match="Expanded"):
        ser.load_model(path)


@pytest.mark.parametrize("kind", ["expanded", "shape", "duplicate"])
def test_nested_archive_validation(tmp_path, monkeypatch, unsigned, kind):
    path = tmp_path / "model.hugiml"
    X = pd.DataFrame({"a": range(40)})
    ser.save_model(HUGIMLClassifier(L=1, topK=3).fit(X, np.arange(40) % 2), path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    buf = io.BytesIO()
    if kind == "expanded":
        np.savez_compressed(buf, data=np.zeros(50000))
        monkeypatch.setenv("HUGIML_MODEL_MAX_BYTES", "16384")
    else:
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header, {"descr": "<f8", "fortran_order": False, "shape": (2**40,)}
        )
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("data.npy", header.getvalue())
            if kind == "duplicate":
                with pytest.warns(UserWarning):
                    archive.writestr("data.npy", header.getvalue())
    members["arrays.npz"] = buf.getvalue()
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    with pytest.raises(HUGIMLSerializationError):
        ser.load_model(path)


def test_dataset_identity_includes_last_row():
    from hugiml.dashboard.dash_components.data_utils import fingerprint

    a = pd.DataFrame({"x": range(1002)})
    b = a.copy()
    b.loc[1001, "x"] = -1
    assert fingerprint(a) != fingerprint(b)
    assert fingerprint(a) == fingerprint(a.copy())


@pytest.mark.parametrize(
    "labels",
    [
        [0.5, 1.5],
        [float("nan"), 1],
        [float("inf"), 1],
        ["cat", "dog"],
        np.array([2**63], dtype=np.uint64),
        np.array([2**63], dtype=object),
        np.array([0.5], dtype=object),
        [float(2**63)],
    ],
)
def test_invalid_labels_rejected(labels):
    with pytest.raises(HUGIMLValidationError):
        HUGIMLClassifier._safe_cast_y(labels)


@pytest.mark.parametrize(
    "labels",
    [
        [0, 1],
        [0.0, 1.0],
        np.array([0, 1], dtype=object),
        np.array([2**63 - 1, -(2**63)], dtype=object),
        np.array([True, False]),
    ],
)
def test_exact_label_conversion(labels):
    np.testing.assert_array_equal(
        HUGIMLClassifier._safe_cast_y(labels), np.asarray(labels, dtype=np.int64)
    )


def test_fit_and_prepare_reject_fractional_labels():
    X = pd.DataFrame({"a": range(20)})
    y = np.arange(20) % 2 + 0.5
    for action in (HUGIMLClassifier().fit, HUGIMLClassifier().prepareXy):
        with pytest.raises(HUGIMLValidationError):
            action(X, y)

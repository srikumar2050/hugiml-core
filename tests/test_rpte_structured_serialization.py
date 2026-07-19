import json
import pickle
import zipfile

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.multiclass import OneVsRestClassifier

import hugiml.serialization as serialization
from hugiml import HUGIMLClassifier
from hugiml.rpte_bounded_lookahead_leafwise import (
    LeafWiseBoundedLookaheadRPTEFeatureLR,
)


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def test_sequential_rpte_uses_structured_state_and_roundtrips_exactly():
    X, y = load_breast_cancer(return_X_y=True)
    X_train = X[:360, :8]
    X_test = X[360:, :8]
    model = LeafWiseBoundedLookaheadRPTEFeatureLR(
        n_estimators=3,
        depth=2,
        enable_lookahead=False,
        random_state=7,
    ).fit(X_train, y[:360])

    config, arrays = serialization._serialize_estimator(model)
    restored = serialization._deserialize_estimator(config, arrays)

    assert config["serialization"] == "structured_rpte_v1"
    assert not _contains_key(config, "_pickle_fallback")
    assert not any("pickle_payload" in key for key in arrays)
    np.testing.assert_allclose(model.predict_proba(X_test), restored.predict_proba(X_test))
    assert model.unified_rule_table() == restored.unified_rule_table()
    assert restored.fe_._default_fe is not None
    assert len(restored.fe_._default_fe.trees_) == len(model.fe_._default_fe.trees_)


def test_ovr_rpte_serialization_is_recursive_and_structured():
    X, y = load_iris(return_X_y=True)
    model = OneVsRestClassifier(
        LeafWiseBoundedLookaheadRPTEFeatureLR(
            n_estimators=2,
            depth=2,
            enable_lookahead=False,
            random_state=9,
        ),
        n_jobs=1,
    ).fit(X, y)

    config, arrays = serialization._serialize_estimator(model)
    restored = serialization._deserialize_estimator(config, arrays)

    assert config["serialization"] == "structured_ovr_v1"
    assert len(config["estimators"]) == len(model.estimators_)
    assert all(item["serialization"] == "structured_rpte_v1" for item in config["estimators"])
    assert not _contains_key(config, "_pickle_fallback")
    assert not any("pickle_payload" in key for key in arrays)
    np.testing.assert_allclose(model.predict_proba(X), restored.predict_proba(X))
    assert restored.label_binarizer_.y_type_ == model.label_binarizer_.y_type_


def test_hugiml_lookahead_rpte_archive_is_structured(tmp_path):
    rng = np.random.default_rng(11)
    raw = rng.integers(0, 2, size=(700, 4))
    X = pd.DataFrame(raw, columns=["x0", "x1", "x2", "x3"])
    y = raw[:, 0] ^ raw[:, 1] ^ raw[:, 2]

    rpte = LeafWiseBoundedLookaheadRPTEFeatureLR(
        n_estimators=2,
        depth=2,
        enable_lookahead=True,
        min_samples_leaf=5,
        random_state=11,
        use_statistical_acceptance=False,
        lookahead_min_probe_ig=0.0,
        min_weighted_probe_gain=0.0,
        min_tree_residual_gain=0.0,
    )
    model = HUGIMLClassifier(
        B=-1,
        adaptive_binning=True,
        L=2,
        topK=20,
        G=0.001,
        feature_mode="original_plus_patterns",
        augmented_pair_transforms=True,
        convert_binary_to_categorical=False,
        base_estimator=rpte,
    ).fit(X, y)

    before = model.predict_proba(X)
    rules_before = model.rpte_rule_table()
    path = tmp_path / "lookahead_rpte.hugiml"
    serialization.save_model(model, path)
    restored = serialization.load_model(path)

    with zipfile.ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        estimator_config = json.loads(archive.read("estimator.json"))
        estimator_array_names = set(np.load(archive.open("estimator_arrays.npz")).files)

    assert manifest["schema_version"] == serialization.MODEL_SCHEMA_VERSION == 10
    assert not _contains_key(estimator_config, "_pickle_fallback")
    assert not any("pickle_payload" in key for key in estimator_array_names)
    np.testing.assert_allclose(before, restored.predict_proba(X))
    assert rules_before == restored.rpte_rule_table()
    fitted = restored.model_.named_steps["clf"]
    assert fitted.fe_._default_fe is None
    assert fitted.fe_.trees_


def test_previous_rpte_fallback_payload_remains_loadable():
    X, y = load_breast_cancer(return_X_y=True)
    model = LeafWiseBoundedLookaheadRPTEFeatureLR(
        n_estimators=1,
        depth=1,
        enable_lookahead=False,
        random_state=3,
    ).fit(X[:250, :6], y[:250])
    payload = pickle.dumps(model, protocol=5)
    config = {
        "class": (
            "hugiml.rpte_bounded_lookahead_leafwise."
            "LeafWiseBoundedLookaheadRPTEFeatureLR"
        ),
        "_pickle_fallback": True,
    }
    arrays = {"pickle_payload": np.frombuffer(payload, dtype=np.uint8)}

    restored = serialization._deserialize_estimator(config, arrays)
    np.testing.assert_allclose(
        model.predict_proba(X[250:, :6]),
        restored.predict_proba(X[250:, :6]),
    )


def test_degenerate_rpte_feature_fallback_roundtrips_structurally():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(120, 3))
    y = np.arange(120) % 2
    model = LeafWiseBoundedLookaheadRPTEFeatureLR(
        n_estimators=1,
        depth=1,
        min_samples_leaf=80,
        enable_lookahead=False,
        random_state=21,
    ).fit(X, y)
    assert model.fe_._raw_feature_fallback_ is True

    config, arrays = serialization._serialize_estimator(model)
    restored = serialization._deserialize_estimator(config, arrays)

    assert not _contains_key(config, "_pickle_fallback")
    assert restored.fe_._raw_feature_fallback_ is True
    np.testing.assert_allclose(model.predict_proba(X), restored.predict_proba(X))
    assert model.unified_rule_table() == restored.unified_rule_table()

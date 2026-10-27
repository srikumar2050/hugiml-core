import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from hugiml import HUGIMLClassifier
from hugiml.exceptions import HUGIMLSerializationError
from hugiml.serialization import load_model, save_model


@pytest.mark.parametrize('policy', ['standard', 'main_effect'])
@pytest.mark.parametrize('columns', [list(range(6)), list('abcdef')])
@pytest.mark.parametrize('adaptive', [False, True])
def test_binning_column_keys_and_predictions_roundtrip(tmp_path, policy, columns, adaptive):
    X, y = make_classification(n_samples=200, n_features=6, n_informative=4,
                               n_redundant=0, random_state=91)
    X = pd.DataFrame(X, columns=columns)
    X.iloc[::17, 0] = np.nan
    train, test = train_test_split(np.arange(len(y)), test_size=0.25,
                                   random_state=42, stratify=y)
    model = HUGIMLClassifier(L=2, topK=16, adaptive_binning=adaptive,
                            feature_mode='original_plus_patterns',
                            lr_source_policy=policy, n_jobs=1).fit(X.iloc[train], y[train])
    path = tmp_path / 'model.hugiml'
    save_model(model, path)
    restored = load_model(path)
    for field in ('_missing_col_edges_', '_bin_edges_', 'per_feature_b_', 'ig_scores_'):
        original = getattr(model, field, None) or {}
        loaded = getattr(restored, field, None) or {}
        assert original.keys() == loaded.keys()
        for key in original:
            if isinstance(original[key], dict):
                assert original[key] == loaded[key]
            else:
                np.testing.assert_array_equal(original[key], loaded[key])
    np.testing.assert_array_equal(model.predict(X.iloc[test]), restored.predict(X.iloc[test]))
    np.testing.assert_allclose(model.predict_proba(X.iloc[test]),
                               restored.predict_proba(X.iloc[test]), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("records", [[[[0], [1, 2]]], [[0, [1, 2]], [0, [3, 4]]], None])
def test_invalid_column_key_records_rejected(tmp_path, records):
    X = pd.DataFrame({"a": np.arange(40) % 7, "b": np.arange(40) % 3})
    model = HUGIMLClassifier(L=1, topK=4).fit(X, np.arange(40) % 2)
    path = tmp_path / "model.hugiml"
    save_model(model, path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    state = json.loads(members["clf_fit.json"])
    state["column_keyed_binning_state"]["_bin_edges_"] = records
    members["clf_fit.json"] = json.dumps(state).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    with pytest.raises(HUGIMLSerializationError, match="column-keyed"):
        load_model(path)


def test_schema_11_named_columns_remain_loadable(tmp_path):
    X = pd.DataFrame({"a": np.arange(40) % 7, "b": np.arange(40) % 3})
    model = HUGIMLClassifier(L=2, topK=4, adaptive_binning=True).fit(X, np.arange(40) % 2)
    path = tmp_path / "model.hugiml"
    save_model(model, path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    state = json.loads(members["clf_fit.json"])
    del state["column_keyed_binning_state"]
    members["clf_fit.json"] = json.dumps(state).encode()
    manifest = json.loads(members["manifest.json"])
    manifest.update(schema_version=11, format_version=11)
    members["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    np.testing.assert_allclose(model.predict_proba(X), load_model(path).predict_proba(X))

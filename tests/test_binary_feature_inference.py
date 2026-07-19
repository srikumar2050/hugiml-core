from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from hugiml import HUGIMLClassifierNative
from hugiml.serialization import load_model, save_model


def _binary_frame() -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame(
        {
            "binary_num": [0, 1, 0, 1, 0, 1, 0, 1],
            "binary_float": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "continuous": [0.1, 0.4, 1.2, 1.8, 2.1, 2.5, 3.2, 3.9],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    return X, y


def test_prepareXy_infers_numeric_binary_columns_as_categorical() -> None:
    X, y = _binary_frame()
    clf = HUGIMLClassifierNative(B=4, L=1, convert_binary_to_categorical=True)

    X_prepared, _ = clf.prepareXy(X, y)

    assert clf.binary_categorical_cols_ == ["binary_num", "binary_float"]
    assert clf.cat_cols_mask_.tolist() == [True, True, False]
    assert clf.is_int_mask_.tolist() == [False, False, False]
    assert str(X_prepared["binary_num"].dtype) == "category"
    assert str(X_prepared["binary_float"].dtype) == "category"


def test_resolve_col_meta_infers_numeric_binary_columns_as_categorical() -> None:
    X, _ = _binary_frame()
    clf = HUGIMLClassifierNative(B=4, L=1, convert_binary_to_categorical=True)

    mask = clf._resolve_col_meta(X)

    assert mask.tolist() == [True, True, False]
    assert clf.binary_categorical_cols_ == ["binary_num", "binary_float"]
    assert clf.is_int_mask_.tolist() == [False, False, False]


def test_resolve_col_meta_keeps_numeric_binary_columns_numeric_by_default() -> None:
    """convert_binary_to_categorical's default is False: binary-valued
    numeric columns stay numeric (and so remain eligible for augmented-pair
    transforms) unless a caller opts into the old categorical-inference
    behavior explicitly."""
    X, _ = _binary_frame()
    clf = HUGIMLClassifierNative(B=4, L=1)

    mask = clf._resolve_col_meta(X)

    assert mask.tolist() == [False, False, False]
    assert clf.binary_categorical_cols_ == []


def test_original_feature_downstream_respects_binary_categorical_mask() -> None:
    X, _ = _binary_frame()
    clf = HUGIMLClassifierNative(
        B=4, L=1, feature_mode="original_plus_patterns", convert_binary_to_categorical=True
    )
    clf._resolve_col_meta(X)

    X_downstream = clf._prepare_original_features_for_downstream(X, fit=True)

    assert clf._original_numeric_cols_ == ["continuous"]
    assert clf._original_cat_cols_ == ["binary_num", "binary_float"]
    assert X_downstream.shape[0] == len(X)
    assert any(str(c).startswith("binary_num_") for c in clf._original_dummy_columns_)
    assert any(str(c).startswith("binary_float_") for c in clf._original_dummy_columns_)


def test_binary_categorical_columns_are_serialized(tmp_path) -> None:
    pytest.importorskip("hugiml._hugiml_core")
    X, y = _binary_frame()
    clf = HUGIMLClassifierNative(
        B=4, L=1, topK=5, G=0.0, feature_mode="patterns_only", convert_binary_to_categorical=True
    )
    clf.fit(X, y)

    out = tmp_path / "binary.hugiml"
    save_model(clf, out)

    with zipfile.ZipFile(out, "r") as zf:
        fit_state = json.loads(zf.read("clf_fit.json"))
    assert fit_state["binary_categorical_cols"] == ["binary_num", "binary_float"]

    loaded = load_model(out)
    assert loaded.binary_categorical_cols_ == ["binary_num", "binary_float"]
    assert loaded.cat_cols_mask_.tolist() == [True, True, False]

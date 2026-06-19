import numpy as np
import pytest

from hugiml import HUGIMLClassifier, HUGIMLClassifierNative
from hugiml.classifier import HUGIMLParamError
from hugiml.serialization import MODEL_SCHEMA_VERSION


def test_execution_mode_validation_is_deferred_until_public_fit():
    clf = HUGIMLClassifier(execution_mode="staging", use_hotpath=False)
    assert clf.execution_mode == "staging"
    clf.set_params(execution_mode="bad")
    assert clf.execution_mode == "bad"
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y = np.asarray([0, 1, 0, 1])
    with pytest.raises(HUGIMLParamError):
        clf.fit(X, y)


def test_dense_downstream_max_width_is_sklearn_parameter_and_validated():
    clf = HUGIMLClassifier(dense_downstream_max_width=0)
    assert clf.get_params()["dense_downstream_max_width"] == 0
    assert clf._dense_downstream_width_threshold() == 0
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y = np.asarray([0, 1, 0, 1])
    clf.set_params(dense_downstream_max_width=-1)
    with pytest.raises(HUGIMLParamError):
        clf.fit(X, y)
    clf.set_params(dense_downstream_max_width=True)
    with pytest.raises(HUGIMLParamError):
        clf.fit(X, y)


def test_hugimlclassifier_alias_is_public_class():
    assert HUGIMLClassifier is HUGIMLClassifierNative


def test_pickle_backward_compat_sets_dense_downstream_max_width_default():
    clf = HUGIMLClassifier()
    state = clf.__dict__.copy()
    state.pop("dense_downstream_max_width", None)
    restored = HUGIMLClassifier.__new__(HUGIMLClassifier)
    restored.__setstate__(state)
    assert restored.dense_downstream_max_width == 200
    assert restored.get_params()["dense_downstream_max_width"] == 200


def test_schema_version_bumped_for_v119_serialization_contract():
    assert MODEL_SCHEMA_VERSION >= 5


def test_require_audit_artifact_empty_attrs_is_unconditional_in_production():
    clf = HUGIMLClassifier(execution_mode="production")
    with pytest.raises(RuntimeError, match="execution_mode='production'"):
        clf._require_audit_artifact("Audit-only object")


def test_fixedB_L_gt_1_clean_numeric_columns_remain_numeric_and_ndarray_type_is_preserved():
    X = np.random.RandomState(0).randn(20, 4)
    clf = HUGIMLClassifier(B=4, L=2, adaptive_binning=False, interaction_relaxed_mining=False)
    clf.feature_names_in_ = [f"x{j}" for j in range(X.shape[1])]
    clf.cat_cols_mask_ = np.zeros(X.shape[1], dtype=bool)
    clf.is_int_mask_ = np.zeros(X.shape[1], dtype=bool)

    X_pre = clf._prebin_nan_cols(X)

    assert isinstance(X_pre, np.ndarray)
    assert clf._missing_col_edges_ == {}
    assert not clf.cat_cols_mask_.any()
    X_pred, cat_mask = clf._handle_test_nan(X)
    assert isinstance(X_pred, np.ndarray)
    assert not cat_mask.any()


def test_fixedB_L_eq_1_prebins_only_training_missing_numeric_columns_and_preserves_ndarray_type():
    X = np.random.RandomState(1).randn(20, 4)
    X[0, 2] = np.nan
    clf = HUGIMLClassifier(B=4, L=1, adaptive_binning=False)
    clf.feature_names_in_ = [f"x{j}" for j in range(X.shape[1])]
    clf.cat_cols_mask_ = np.zeros(X.shape[1], dtype=bool)
    clf.is_int_mask_ = np.zeros(X.shape[1], dtype=bool)

    X_pre = clf._prebin_nan_cols(X)

    assert isinstance(X_pre, np.ndarray)
    assert set(clf._missing_col_edges_) == {"x2"}
    assert clf.cat_cols_mask_.tolist() == [False, False, True, False]
    X_pred, cat_mask = clf._handle_test_nan(X)
    assert isinstance(X_pred, np.ndarray)
    assert cat_mask.tolist() == [False, False, True, False]


def test_fixedB_L_gt_1_prebins_only_training_missing_numeric_columns():
    X = np.random.RandomState(3).randn(20, 4)
    X[0, 1] = np.nan
    X[1, 3] = np.inf
    clf = HUGIMLClassifier(B=4, L=2, adaptive_binning=False, interaction_relaxed_mining=False)
    clf.feature_names_in_ = [f"x{j}" for j in range(X.shape[1])]
    clf.cat_cols_mask_ = np.zeros(X.shape[1], dtype=bool)
    clf.is_int_mask_ = np.zeros(X.shape[1], dtype=bool)

    X_pre = clf._prebin_nan_cols(X)

    assert isinstance(X_pre, np.ndarray)
    assert set(clf._missing_col_edges_) == {"x1", "x3"}
    assert clf.cat_cols_mask_.tolist() == [False, True, False, True]
    X_pred, cat_mask = clf._handle_test_nan(X)
    assert isinstance(X_pred, np.ndarray)
    assert cat_mask.tolist() == [False, True, False, True]


def test_model_summary_static_contract_has_no_eager_hup_default():
    import inspect

    import hugiml.classifier as classifier_mod

    src = inspect.getsource(classifier_mod.HUGIMLClassifierNative.model_summary)
    assert "getattr(self, 'x_train_downstream_', self.x_train_hup_)" not in src
    assert "_summary_shape_text" in src


def test_classifier_module_all_contains_legacy_public_symbols():
    import hugiml.classifier as classifier_mod

    assert "HUGIMLClassifier" in classifier_mod.__all__
    assert "HUGIMLClassifierNative" in classifier_mod.__all__
    assert "FitMetadata" in classifier_mod.__all__
    assert "HUGIMLTuneResult" in classifier_mod.__all__


def test_fixedB_L_gt_1_clean_named_dataframe_preserves_dataframe_column_names_and_numeric_path():
    import pandas as pd

    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "alpha": rng.normal(size=20),
            "beta": rng.normal(size=20),
            "gamma": rng.normal(size=20),
        }
    )
    clf = HUGIMLClassifier(B=4, L=2, adaptive_binning=False, interaction_relaxed_mining=False)
    clf.feature_names_in_ = list(X.columns)
    clf.cat_cols_mask_ = np.zeros(X.shape[1], dtype=bool)
    clf.is_int_mask_ = np.zeros(X.shape[1], dtype=bool)

    X_pre = clf._prebin_nan_cols(X)

    assert isinstance(X_pre, pd.DataFrame)
    assert list(X_pre.columns) == ["alpha", "beta", "gamma"]
    assert clf._missing_col_edges_ == {}
    assert not clf.cat_cols_mask_.any()
    X_pred, cat_mask = clf._handle_test_nan(X)
    assert isinstance(X_pred, pd.DataFrame)
    assert list(X_pred.columns) == ["alpha", "beta", "gamma"]
    assert not cat_mask.any()


def test_classifier_docstring_promotes_primary_alias_name():
    import hugiml.classifier as classifier_mod

    doc = classifier_mod.__doc__ or ""
    quick_start = doc.split("Quick start", 1)[-1]
    assert "from hugiml import HUGIMLClassifier" in quick_start
    assert "clf = HUGIMLClassifier()" in quick_start


def test_schema_v5_serializes_empty_missing_col_edges_explicitly(tmp_path):
    import json
    import zipfile

    X = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
    y = np.asarray([0, 0, 1, 1, 1, 0])
    clf = HUGIMLClassifier(B=3, L=1, G=0.0, adaptive_binning=False, use_hotpath=False)
    clf.fit(X, y)
    assert clf._missing_col_edges_ == {}

    path = tmp_path / "clean_l1.hugiml"
    clf.save_model(path)
    with zipfile.ZipFile(path, "r") as zf:
        fit_state = json.loads(zf.read("clf_fit.json"))
    assert "missing_col_edges" in fit_state
    assert fit_state["missing_col_edges"] == {}


def test_repr_uses_primary_public_alias_name():
    clf = HUGIMLClassifier(B=3, L=1, G=0.0)
    text = repr(clf)
    assert text.startswith("HUGIMLClassifier(")
    assert "HUGIMLClassifierNative(" not in text


def test_require_audit_artifact_checks_missing_attrs_in_audit_mode():
    clf = HUGIMLClassifier(execution_mode="audit")
    with pytest.raises(RuntimeError, match="required fitted artifact"):
        clf._require_audit_artifact("Audit-only object", "definitely_missing_attr_")


def test_fixedB_L_gt_1_clean_numeric_fit_path_keeps_missing_edges_empty():
    pytest.importorskip("hugiml._hugiml_core")
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = HUGIMLClassifier(
        B=4,
        L=2,
        G=0.0,
        adaptive_binning=False,
        use_hotpath=False,
        feature_mode="original_plus_patterns",
    )
    clf.fit(X, y)
    assert getattr(clf, "_missing_col_edges_", {}) == {}
    assert not np.any(getattr(clf, "cat_cols_mask_"))
    X_nan = X[:5].copy()
    X_nan[0, 0] = np.nan
    proba = clf.predict_proba(X_nan)
    assert np.isfinite(proba).all()


def test_fixedB_training_missing_numeric_full_fit_path_prebins_and_predict_handles_new_nan():
    pytest.importorskip("hugiml._hugiml_core")
    rng = np.random.default_rng(23)
    X = rng.normal(size=(48, 4))
    y = (X[:, 0] - 0.5 * X[:, 1] + X[:, 3] > 0).astype(int)
    X[0, 2] = np.nan
    X[5, 2] = np.inf

    clf = HUGIMLClassifier(
        B=4,
        L=2,
        G=0.0,
        adaptive_binning=False,
        use_hotpath=False,
        feature_mode="original_plus_patterns",
    )
    clf.fit(X, y)

    assert set(getattr(clf, "_missing_col_edges_", {})) == {"col2"}
    assert getattr(clf, "cat_cols_mask_").tolist() == [False, False, True, False]

    X_test = X[:8].copy()
    X_test[1, 2] = np.nan
    X_test[2, 2] = -np.inf
    proba = clf.predict_proba(X_test)

    assert proba.shape == (8, len(clf.classes_))
    assert np.isfinite(proba).all()


def test_dense_strict_topk_scores_fall_back_when_native_dense_helper_missing(monkeypatch):
    import hugiml.classifier as classifier_mod

    class _CoreWithoutDense:
        pass

    clf = HUGIMLClassifier(B=3, topk_budget_strict=True)
    X = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 1.0],
            [3.0, 0.0, 1.0],
            [4.0, 1.0, 1.0],
            [5.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.asarray([0, 0, 0, 1, 1, 1])
    names = ["orig:x0", "orig:x1", "orig:x2"]

    monkeypatch.setattr(classifier_mod, "_CORE_AVAILABLE", True)
    monkeypatch.setattr(classifier_mod, "_core", _CoreWithoutDense())

    scores, mask = clf._strict_topk_dense_column_scores(X, y, names, top_k=2)

    assert scores.shape == (3,)
    assert mask.shape == (3,)
    assert mask.dtype == np.bool_
    assert int(mask.sum()) == 2
    assert np.isfinite(scores).all()


def test_refit_clear_list_includes_production_shape_and_original_median_caches():
    import inspect

    import hugiml.classifier as classifier_mod

    src = inspect.getsource(classifier_mod.HUGIMLClassifierNative._fit_impl)
    for attr in (
        "_training_pattern_matrix_shape_",
        "_training_pattern_matrix_nnz_",
        "_training_downstream_matrix_shape_",
        "_training_downstream_matrix_nnz_",
        "_original_numeric_medians_array_",
        "_original_numeric_medians_",
    ):
        assert attr in src


def test_drift_docstrings_document_missing_numeric_limitations():
    assert "NaN/Inf during training" in (HUGIMLClassifier.detect_drift.__doc__ or "")
    assert "continuous PSI baselines" in (HUGIMLClassifier.get_drift_psi.__doc__ or "")

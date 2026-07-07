import numpy as np
import pandas as pd
import pytest

from hugiml import HUGIMLClassifier, HUGIMLClassifierNative


def _zero_variance_cols(X, names):
    """Call the private helper directly with a minimal stand-in estimator."""
    clf = HUGIMLClassifierNative.__new__(HUGIMLClassifierNative)
    clf.feature_names_in_ = names
    return clf._identify_zero_variance_columns(X)


# ---------------------------------------------------------------------------
# _identify_zero_variance_columns: numeric ndarray fast path
# ---------------------------------------------------------------------------

def test_zero_variance_ndarray_no_zero_variance_columns():
    X = np.column_stack([
        np.arange(20, dtype=np.float32),
        np.random.RandomState(0).randn(20).astype(np.float32),
    ])
    assert _zero_variance_cols(X, ["a", "b"]) == []


def test_zero_variance_ndarray_constant_column_detected():
    X = np.column_stack([np.full(20, 5.0, dtype=np.float32), np.arange(20, dtype=np.float32)])
    assert _zero_variance_cols(X, ["const", "var"]) == ["const"]


def test_zero_variance_ndarray_all_columns_constant():
    X = np.full((15, 3), 7.0, dtype=np.float32)
    assert _zero_variance_cols(X, ["a", "b", "c"]) == ["a", "b", "c"]


def test_zero_variance_all_nan_column_is_zero_variance():
    X = np.column_stack([np.full(10, np.nan), np.arange(10, dtype=np.float64)])
    assert _zero_variance_cols(X, ["allnan", "var"]) == ["allnan"]


def test_zero_variance_single_value_plus_nans_is_zero_variance():
    col = np.full(12, np.nan)
    col[3] = 42.0
    X = np.column_stack([col, np.arange(12, dtype=np.float64)])
    assert _zero_variance_cols(X, ["mostlynan", "var"]) == ["mostlynan"]


def test_zero_variance_nans_with_multiple_distinct_values_not_zero_variance():
    col = np.array([1.0, np.nan, 2.0, np.nan, 1.0, 3.0])
    X = np.column_stack([col, np.arange(6, dtype=np.float64)])
    assert _zero_variance_cols(X, ["mixednan", "var"]) == []


@pytest.mark.parametrize("dtype", [np.int64, np.uint32, bool])
def test_zero_variance_integer_and_bool_dtypes(dtype):
    if dtype is bool:
        const_col = np.zeros(10, dtype=bool)
        var_col = np.array([True, False] * 5)
    else:
        const_col = np.full(10, 3, dtype=dtype)
        var_col = np.arange(10, dtype=dtype)
    X = np.column_stack([const_col, var_col])
    assert _zero_variance_cols(X, ["c", "v"]) == ["c"]


def test_zero_variance_negative_zero_equals_positive_zero():
    X = np.column_stack([
        np.array([0.0, -0.0, 0.0, -0.0], dtype=np.float32),
        np.arange(4, dtype=np.float32),
    ])
    assert _zero_variance_cols(X, ["z", "var"]) == ["z"]


def test_zero_variance_inf_counts_as_a_real_distinct_value():
    X = np.column_stack([np.array([1.0, np.inf, 1.0, np.inf]), np.arange(4, dtype=np.float64)])
    assert _zero_variance_cols(X, ["infcol", "var"]) == []


def test_zero_variance_names_longer_than_column_count():
    X = np.arange(10, dtype=np.float64).reshape(5, 2)
    assert _zero_variance_cols(X, ["a", "b", "extra"]) == []


def test_zero_variance_empty_feature_names_short_circuits():
    assert _zero_variance_cols(np.zeros((5, 2)), []) == []


# ---------------------------------------------------------------------------
# _identify_zero_variance_columns: fallback path (non-numeric input)
# ---------------------------------------------------------------------------

def test_zero_variance_object_dtype_fallback_still_works():
    X = np.array([["a", "1"], ["a", "2"], ["a", "3"]], dtype=object)
    assert _zero_variance_cols(X, ["s1", "s2"]) == ["s1"]


def test_zero_variance_dataframe_input_still_works():
    df = pd.DataFrame({"x": [1, 1, 1], "y": [1, 2, 3]})
    assert _zero_variance_cols(df, ["x", "y"]) == ["x"]


def test_zero_variance_list_of_lists_input():
    X = [[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
    assert _zero_variance_cols(X, ["a", "b"]) == ["a"]


# ---------------------------------------------------------------------------
# End-to-end: fit()/predict() are unaffected by input container type
# ---------------------------------------------------------------------------

def _make_dataset(n=600, p=10, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p).astype(np.float32)
    X[:, 1] = 3.0  # zero-variance column
    y = ((X[:, :3].sum(axis=1) + 0.6 * rng.randn(n)) > 0).astype(np.int64)
    return X, y


@pytest.mark.parametrize("L", [1, 2, 3])
@pytest.mark.parametrize("adaptive_binning", [True, False])
def test_fit_predict_identical_for_ndarray_and_dataframe_input(L, adaptive_binning):
    """Regression guard: hotpath dispatch and input container type must
    never change the fitted model, only how fast fitting it is."""
    X, y = _make_dataset()
    Xdf = pd.DataFrame(X, columns=[f"c{i}" for i in range(X.shape[1])])

    clf_arr = HUGIMLClassifier(B=7, L=L, G=5e-3, topK=50,
                                adaptive_binning=adaptive_binning, use_hotpath=True, n_jobs=1)
    clf_arr.fit(X, y)

    clf_df = HUGIMLClassifier(B=7, L=L, G=5e-3, topK=50,
                               adaptive_binning=adaptive_binning, use_hotpath=True, n_jobs=1)
    clf_df.fit(Xdf, y)

    assert len(clf_arr.patterns_) == len(clf_df.patterns_)
    assert clf_arr.score(X, y) == pytest.approx(clf_df.score(Xdf, y))


def test_fit_with_zero_variance_column_and_hotpath_matches_non_hotpath():
    X, y = _make_dataset()

    clf_hot = HUGIMLClassifier(B=7, L=1, G=5e-3, topK=50,
                                adaptive_binning=True, use_hotpath=True, n_jobs=1)
    clf_hot.fit(X, y)

    clf_cold = HUGIMLClassifier(B=7, L=1, G=5e-3, topK=50,
                                 adaptive_binning=True, use_hotpath=False, n_jobs=1)
    clf_cold.fit(X, y)

    assert len(clf_hot.patterns_) == len(clf_cold.patterns_)
    assert clf_hot.score(X, y) == pytest.approx(clf_cold.score(X, y))
    assert list(np.asarray(clf_hot.predict(X))) == list(np.asarray(clf_cold.predict(X)))

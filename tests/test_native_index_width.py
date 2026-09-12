import numpy as np
import pytest
from scipy.sparse import csr_matrix

native = pytest.importorskip(
    "_hugiml_index_validation", reason="Build the native CSR validation module first"
)


def test_index_width_boundary_without_large_allocations():
    maximum = 2**31 - 1
    assert native.index_bits(5, 7, maximum, 64) == 32
    assert native.index_bits(5, 7, maximum + 1, 64) == 64
    assert native.index_bits(maximum + 1, 7, 0, 64) == 64
    assert native.index_bits(5, maximum + 1, 0, 64) == 64


def test_32_bit_process_never_narrows_large_structure():
    assert native.index_bits(3, 5, 8, 32) == 32
    with pytest.raises(OverflowError):
        native.index_bits(3, 5, 2**31, 32)
    with pytest.raises(OverflowError):
        native.index_bits(3, 5, 2**29, 32)
    with pytest.raises(OverflowError):
        native.index_bits(2**63 - 1, 1, 0, 64)


@pytest.mark.parametrize("rows,cols,nnz", [(-1, 1, 0), (1, -1, 0)])
def test_negative_dimensions(rows, cols, nnz):
    with pytest.raises(ValueError):
        native.index_bits(rows, cols, nnz)


def test_mining_dimension_is_checked_before_narrowing():
    assert native.native_dimension(2**31 - 2, "rows") == 2**31 - 2
    with pytest.raises(OverflowError):
        native.native_dimension(2**31 - 1, "rows")
    with pytest.raises(OverflowError):
        native.native_dimension(2**32 + 5, "rows")


@pytest.mark.parametrize("shape", [(0, 0), (10, 7), (4, 1), (1, 12)])
def test_both_index_widths_produce_identical_matrices(shape):
    rng = np.random.default_rng(4)
    n, p = shape
    rows = rng.integers(0, n, size=60).tolist() if n and p else []
    cols = rng.integers(0, p, size=60).tolist() if n and p else []
    outputs = []
    for method, dtype in [
        (native.csr32, np.int32),
        (native.csr64, np.int64),
        (native.csr_auto, np.int32),
    ]:
        indptr, indices = method(rows, cols, n, p)
        assert indptr.dtype == indices.dtype == dtype
        assert indptr[-1] == len(rows)
        matrix = csr_matrix((np.ones(len(rows)), indices, indptr), shape=shape)
        assert matrix.has_sorted_indices
        expected = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)
        np.testing.assert_array_equal(matrix.toarray(), expected.toarray())
        outputs.append((indptr, indices))
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


@pytest.mark.parametrize("method", [native.csr32, native.csr64, native.csr_auto])
@pytest.mark.parametrize(
    "rows,cols,n,p",
    [([0], [], 1, 1), ([-1], [0], 1, 1), ([1], [0], 1, 1), ([0], [-1], 1, 1), ([0], [1], 1, 1)],
)
def test_invalid_coordinates_are_rejected(method, rows, cols, n, p):
    with pytest.raises((ValueError, IndexError)):
        method(rows, cols, n, p)


def test_automatic_wide_matrix_without_large_storage():
    indptr, indices = native.csr_auto([0], [0], 1, 2**31)
    assert indptr.dtype == indices.dtype == np.int64
    np.testing.assert_array_equal(indptr, [0, 1])


def test_native_csc_scores_match_for_both_index_widths():
    import _hugiml_core as core
    from scipy.sparse import csc_matrix

    X = csc_matrix(np.random.default_rng(91).normal(size=(60, 7)).astype(np.float32))
    y = np.array([0, 1] * 30, dtype=np.int64)
    outputs = []
    for function, dtype in [
        (core.strict_topk_filter_csc, np.int32),
        (core.strict_topk_filter_csc64, np.int64),
    ]:
        outputs.append(
            function(
                X.data,
                X.indices.astype(dtype),
                X.indptr.astype(dtype),
                60,
                7,
                y,
                np.zeros(7, dtype=np.uint8),
                3,
                8,
            )
        )
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


def test_native_csc_rejects_implicit_index_narrowing():
    import _hugiml_core as core

    with pytest.raises(TypeError):
        core.strict_topk_filter_csc(
            np.ones(1, dtype=np.float32),
            np.array([2**32], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            1,
            1,
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.uint8),
            1,
            8,
        )


def test_python_width_policy_and_native_capability():
    from types import SimpleNamespace

    from hugiml._indexing import native_csc_scorer, sparse_index_dtype

    assert sparse_index_dtype(100, 50, 5000) == np.int32
    assert sparse_index_dtype(100, 50, 2**31) == np.int64
    matrix = SimpleNamespace(shape=(100, 50), nnz=2**31)
    with pytest.raises(OverflowError):
        native_csc_scorer(SimpleNamespace(), matrix)


def test_dense_full_csr_preserves_values():
    from hugiml._classifier_support import _dense_full_csr

    X = np.random.default_rng(29).normal(size=(12, 7)).astype(np.float32)
    result = _dense_full_csr(X)
    assert result.indices.dtype == np.int32
    np.testing.assert_array_equal(result.toarray(), X)


def test_large_input_rejected_without_allocating_elements():
    from hugiml import HUGIMLClassifierNative

    X = np.empty((2**31, 0), dtype=np.float32)
    with pytest.raises(OverflowError, match="index range"):
        HUGIMLClassifierNative().fit(X, np.array([0]))
    with pytest.raises(OverflowError, match="index range"):
        HUGIMLClassifierNative()._validate_test_input(X)

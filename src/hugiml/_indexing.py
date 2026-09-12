"""Checked sparse index selection for native feature matrices."""

import numpy as np


def sparse_index_dtype(n_rows, n_cols, nnz):
    if min(n_rows, n_cols, nnz) < 0:
        raise ValueError("Sparse dimensions and nonzero count must be nonnegative")
    maximum = np.iinfo(np.int32).max
    wide = max(n_rows, n_cols, nnz) > maximum
    if wide and np.dtype(np.intp).itemsize < 8:
        raise OverflowError("Sparse structure requires a 64-bit process")
    dtype = np.int64 if wide else np.int32
    address_max = np.iinfo(np.intp).max
    if (
        n_rows >= address_max // np.dtype(dtype).itemsize
        or nnz > address_max // np.dtype(dtype).itemsize
    ):
        raise OverflowError("Sparse index arrays exceed the process address range")
    return dtype


def native_csc_scorer(core, matrix):
    dtype = sparse_index_dtype(matrix.shape[1], matrix.shape[0], matrix.nnz)
    name = "strict_topk_filter_csc64" if dtype == np.int64 else "strict_topk_filter_csc"
    if not hasattr(core, name):
        raise OverflowError("The native extension does not support this sparse index width")
    return getattr(core, name), dtype


def validate_native_shape(shape):
    if len(shape) != 2:
        return
    if any(int(size) >= np.iinfo(np.int32).max for size in shape):
        raise OverflowError(
            "Input dimensions exceed the native mining index range; maximum dimension is 2147483646"
        )

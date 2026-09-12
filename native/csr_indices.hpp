// Copyright 2026 Srikumar Krishnamoorthy
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace hugiml {

inline int checked_native_dimension(int64_t value, const char* name) {
    if (value < 0) throw std::invalid_argument(std::string(name) + " must be nonnegative");
    if (value >= std::numeric_limits<int32_t>::max())
        throw std::overflow_error(std::string(name) + " exceeds the native mining index range");
    return static_cast<int>(value);
}

inline int csr_index_bits(int64_t n_rows, int64_t n_cols, uint64_t nnz,
                          unsigned address_bits = sizeof(void*) * 8) {
    if (n_rows < 0 || n_cols < 0)
        throw std::invalid_argument("CSR dimensions must be nonnegative");
    if (address_bits != 32 && address_bits != 64)
        throw std::invalid_argument("Address width must be 32 or 64");
    const uint64_t narrow_max = std::numeric_limits<int32_t>::max();
    const int bits = (static_cast<uint64_t>(n_rows) > narrow_max ||
                      static_cast<uint64_t>(n_cols) > narrow_max || nnz > narrow_max) ? 64 : 32;
    if (bits == 64 && address_bits == 32)
        throw std::overflow_error("CSR structure requires 64-bit indexing in a 64-bit process");
    const uint64_t address_max = address_bits == 32 ? narrow_max :
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    const uint64_t item_bytes = static_cast<uint64_t>(bits / 8);
    if (static_cast<uint64_t>(n_rows) >= address_max / item_bytes || nnz > address_max / item_bytes)
        throw std::overflow_error("CSR index arrays exceed the process address range");
    return bits;
}

// Index selects offset storage; row and column IDs retain the mining representation.
template <typename Index>
pybind11::tuple csr_from_coo_typed(const std::vector<int32_t>& rows,
                                 const std::vector<int32_t>& cols,
                                 int64_t n_rows, int64_t n_cols) {
    namespace py = pybind11;
    if (rows.size() != cols.size())
        throw std::invalid_argument("COO row/column vectors must have equal length");
    csr_index_bits(n_rows, n_cols, rows.size());
    const auto max_index = static_cast<uint64_t>(std::numeric_limits<Index>::max());
    if (rows.size() > max_index || static_cast<uint64_t>(n_rows) > max_index ||
        static_cast<uint64_t>(n_cols) > max_index)
        throw std::overflow_error("CSR structure exceeds the selected index width");
    const auto address_max = static_cast<uint64_t>(std::numeric_limits<py::ssize_t>::max());
    if (static_cast<uint64_t>(n_rows) >= address_max / sizeof(Index) || rows.size() > address_max / sizeof(Index))
        throw std::overflow_error("CSR index arrays exceed the process address range");
    for (size_t k = 0; k < rows.size(); ++k) {
        if (rows[k] < 0 || rows[k] >= n_rows) throw std::out_of_range("COO row out of CSR bounds");
        if (cols[k] < 0 || cols[k] >= n_cols) throw std::out_of_range("COO column out of CSR bounds");
    }
    py::array_t<Index> indptr_arr(static_cast<py::ssize_t>(n_rows + 1));
    py::array_t<Index> indices_arr(static_cast<py::ssize_t>(cols.size()));
    auto* indptr = indptr_arr.mutable_data();
    auto* indices = indices_arr.mutable_data();
    std::fill(indptr, indptr + n_rows + 1, Index{0});
    for (auto row : rows) ++indptr[static_cast<size_t>(row) + 1];
    for (int64_t i = 0; i < n_rows; ++i) indptr[i + 1] += indptr[i];
    std::vector<Index> cursor(indptr, indptr + n_rows);
    for (size_t k = 0; k < rows.size(); ++k)
        indices[cursor[static_cast<size_t>(rows[k])]++] = static_cast<Index>(cols[k]);
    for (int64_t i = 0; i < n_rows; ++i)
        std::sort(indices + indptr[i], indices + indptr[i + 1]);
    return py::make_tuple(indptr_arr, indices_arr);
}

inline pybind11::tuple csr_from_coo(const std::vector<int32_t>& rows,
                                   const std::vector<int32_t>& cols,
                                   int64_t n_rows, int64_t n_cols) {
    const int bits = csr_index_bits(n_rows, n_cols, rows.size());
    if (bits == 64) return csr_from_coo_typed<int64_t>(rows, cols, n_rows, n_cols);
    return csr_from_coo_typed<int32_t>(rows, cols, n_rows, n_cols);
}
} // namespace hugiml

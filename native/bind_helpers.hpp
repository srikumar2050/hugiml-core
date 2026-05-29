/**
 * bind_helpers.hpp — Shared validation and extraction helpers for Python bindings.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 *
 * Included by each bind_*.cpp translation unit so that inline helpers are
 * compiled once per TU (not once globally). Keeping them in a header avoids
 * cross-TU linkage and lets the compiler inline them freely.
 */

#pragma once
#include "pybind_common.hpp"
#include "transaction.hpp"
#include "matrix.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ── Array shape validators ────────────────────────────────────────────────────

inline void validate_2d_array(const py::array& arr, const char* name) {
    if (arr.ndim() != 2)
        throw std::invalid_argument(
            std::string(name) + " must be 2-D, got " +
            std::to_string(arr.ndim()) + "-D");
    if (arr.shape(0) == 0)
        throw std::invalid_argument(std::string(name) + " has 0 rows");
    if (!(arr.flags() & py::array::c_style))
        throw std::invalid_argument(
            std::string(name) + " must be C-contiguous");
}

inline void validate_1d_array(const py::array& arr, const char* name,
                               py::ssize_t expected_len = -1) {
    if (arr.ndim() != 1)
        throw std::invalid_argument(
            std::string(name) + " must be 1-D, got " +
            std::to_string(arr.ndim()) + "-D");
    if (expected_len >= 0 && arr.shape(0) != expected_len)
        throw std::invalid_argument(
            std::string(name) + " length " + std::to_string(arr.shape(0)) +
            " does not match expected " + std::to_string(expected_len));
}

inline void validate_mining_params(int K, int L, double G) {
    if (K <= 0)
        throw std::invalid_argument("K (topK) must be > 0, got " + std::to_string(K));
    if (L != -1 && (L < 1 || L > 3))
        throw std::invalid_argument("L must be -1, 1, 2, or 3, got " + std::to_string(L));
    if (G < 0.0)
        throw std::invalid_argument("G must be >= 0, got " + std::to_string(G));
}

// ── COO → py::tuple helper ────────────────────────────────────────────────────

inline py::tuple coo_to_tuple(hugiml::COO&& coo) {
    auto& rv = coo.first;
    auto& cv = coo.second;
    auto rows_arr = py::array_t<int32_t>(rv.size());
    auto cols_arr = py::array_t<int32_t>(cv.size());
    auto ra = rows_arr.mutable_unchecked<1>();
    auto ca = cols_arr.mutable_unchecked<1>();
    for (size_t k = 0; k < rv.size(); k++) { ra(k) = rv[k]; ca(k) = cv[k]; }
    return py::make_tuple(rows_arr, cols_arr);
}

// ── Categorical data pre-extraction (call with GIL held) ─────────────────────
// Converts list-of-column-arrays (or None) to C++ string/validity matrices.

inline void extract_cat_data(
    py::object       X_cat_raw_py,
    const py::array_t<uint8_t, py::array::forcecast>& is_cat_arr,
    int n, int p,
    std::vector<std::vector<std::string>>& out_strs,
    std::vector<std::vector<bool>>&        out_valid)
{
    out_strs.resize(p);
    out_valid.resize(p);
    if (X_cat_raw_py.is_none()) return;

    auto icb = is_cat_arr.unchecked<1>();
    py::list raw_list = X_cat_raw_py.cast<py::list>();

    for (int j = 0; j < p; j++) {
        if (!static_cast<bool>(icb(j))) continue;
        py::object col_obj = raw_list[j].cast<py::object>();
        if (col_obj.is_none()) continue;
        out_strs[j].resize(n);
        out_valid[j].resize(n, false);
        py::array arr = col_obj.cast<py::array>();
        py::list  lst = arr.attr("tolist")().cast<py::list>();
        for (int r = 0; r < n; r++) {
            py::object val = lst[r];
            if (val.is_none()) continue;
            try {
                double dv = val.cast<double>();
                if (std::isnan(dv)) continue;
            } catch (...) {}
            out_valid[j][r] = true;
            out_strs[j][r]  = py::str(val).cast<std::string>();
        }
    }
}

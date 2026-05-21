/**
 * bind_prepare_tx.cpp — Python binding for prepare_transactions.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 *
 * Split from bind_pattern.cpp so that the three heaviest pybind11 lambda TUs
 * can be compiled in parallel, reducing peak single-core build time from
 * ~55 s (O3, monolithic) to ~20 s per TU at O2.
 */

#include "bind_helpers.hpp"
#include "transaction.hpp"

namespace py = pybind11;
using namespace hugiml;

void bind_prepare_transactions(py::module_& m)
{
    m.def("prepare_transactions",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> X_num,
           py::array_t<int64_t, py::array::c_style | py::array::forcecast> y,
           int B,
           py::object col_names,
           py::array_t<uint8_t, py::array::forcecast> is_cat,
           py::array_t<uint8_t, py::array::forcecast> is_int,
           py::object X_cat_raw)
        {
            // ── Step 1: validate with GIL held ────────────────────────────
            validate_2d_array(X_num, "X_num");
            validate_1d_array(y,      "y",      X_num.shape(0));
            validate_1d_array(is_cat, "is_cat", X_num.shape(1));
            validate_1d_array(is_int, "is_int", X_num.shape(1));
            if (B != -1 && B < 2)
                throw std::invalid_argument(
                    "B must be -1 (auto) or >= 2, got " + std::to_string(B));

            int n = static_cast<int>(X_num.shape(0));
            int p = static_cast<int>(X_num.shape(1));

            // ── Step 2: pre-extract Python objects to C++ (GIL held) ──────
            std::vector<std::string> col_names_cpp;
            if (!col_names.is_none()) {
                py::list lst = col_names.cast<py::list>();
                col_names_cpp.reserve(lst.size());
                for (auto& item : lst)
                    col_names_cpp.push_back(item.cast<std::string>());
            }

            std::vector<std::vector<std::string>> cat_strs;
            std::vector<std::vector<bool>>        cat_valid;
            extract_cat_data(X_cat_raw, is_cat, n, p, cat_strs, cat_valid);

            // ── Step 3: release GIL, call pure-C++ ───────────────────────
            py::gil_scoped_release release;
            return prepare_transactions_cpp(
                X_num, y, B,
                std::move(col_names_cpp),
                is_cat, is_int,
                std::move(cat_strs),
                std::move(cat_valid));
        },
        py::arg("X_num"), py::arg("y"), py::arg("B"),
        py::arg("col_names"), py::arg("is_cat"), py::arg("is_int"),
        py::arg("X_cat_raw"),
        "Build utility-annotated TransactionDataCpp from training data.");
}

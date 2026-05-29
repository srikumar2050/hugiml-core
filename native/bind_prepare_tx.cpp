/**
 * bind_prepare_tx.cpp — Python binding for prepare_transactions.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
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
           py::object X_cat_raw,
           py::object is_precoded)
        {
            validate_2d_array(X_num, "X_num");
            validate_1d_array(y,      "y",      X_num.shape(0));
            validate_1d_array(is_cat, "is_cat", X_num.shape(1));
            validate_1d_array(is_int, "is_int", X_num.shape(1));
            if (B != -1 && B < 2)
                throw std::invalid_argument(
                    "B must be -1 (auto) or >= 2, got " + std::to_string(B));

            int n = static_cast<int>(X_num.shape(0));
            int p = static_cast<int>(X_num.shape(1));

            // Column names
            std::vector<std::string> col_names_cpp;
            if (!col_names.is_none()) {
                py::list lst = col_names.cast<py::list>();
                col_names_cpp.reserve(lst.size());
                for (auto& item : lst)
                    col_names_cpp.push_back(item.cast<std::string>());
            }

            // Categorical raw strings
            std::vector<std::vector<std::string>> cat_strs;
            std::vector<std::vector<bool>>        cat_valid;
            extract_cat_data(X_cat_raw, is_cat, n, p, cat_strs, cat_valid);

            // Pre-coded mask: convert to std::vector<bool> with GIL held
            std::vector<bool> ipc_vec(p, false);
            if (!is_precoded.is_none()) {
                auto ipc = is_precoded.cast<
                    py::array_t<uint8_t, py::array::forcecast>>();
                if (ipc.size() != p)
                    throw std::invalid_argument(
                        "is_precoded length must equal number of columns");
                auto ipcb = ipc.unchecked<1>();
                for (int j = 0; j < p; j++)
                    ipc_vec[j] = static_cast<bool>(ipcb(j));
            }

            // Release GIL before the heavy C++ computation
            py::gil_scoped_release release;
            return prepare_transactions_cpp(
                X_num, y, B,
                std::move(col_names_cpp),
                is_cat, is_int,
                std::move(ipc_vec),
                std::move(cat_strs),
                std::move(cat_valid));
        },
        py::arg("X_num"), py::arg("y"), py::arg("B"),
        py::arg("col_names"), py::arg("is_cat"), py::arg("is_int"),
        py::arg("X_cat_raw"),
        py::arg("is_precoded") = py::none(),
        "Build utility-annotated TransactionDataCpp from training data.\n\n"
        "is_precoded: optional uint8 array of length p. When is_precoded[j]=1,\n"
        "column j contains pre-computed 0-indexed bin codes; the C++ layer skips\n"
        "re-discretisation and uses the codes directly as bin indices.");
}

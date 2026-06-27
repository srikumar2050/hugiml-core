/**
 * bind_prepare_tx.cpp — Python binding for prepare_transactions.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 */

#include "bind_helpers.hpp"
#include "transaction.hpp"
#include "resource_guard.hpp"

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
           py::object is_precoded,
           py::object eu_pair_left_py,
           py::object eu_pair_right_py)
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

            // Optional interaction-survivor pairs: column index pairs whose
            // joint (product) correlation with y should be used as a shared
            // external-utility floor for both columns, bypassing the
            // marginal-correlation admission gate.  See the long comment in
            // prepare_transactions_cpp for the rationale: this exists so
            // that a feature whose only true signal is interactive (e.g. an
            // XOR component, whose population marginal correlation with y
            // is exactly zero by construction) is not dropped from the item
            // universe purely because its empirical marginal correlation
            // happens to land at/near zero on a given sample.
            std::vector<int> eu_pair_left_cpp, eu_pair_right_cpp;
            const std::vector<int>* eu_pair_left_ptr  = nullptr;
            const std::vector<int>* eu_pair_right_ptr = nullptr;
            if (!eu_pair_left_py.is_none() && !eu_pair_right_py.is_none()) {
                auto left_arr = eu_pair_left_py.cast<
                    py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
                auto right_arr = eu_pair_right_py.cast<
                    py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
                if (left_arr.size() != right_arr.size())
                    throw std::invalid_argument(
                        "eu_pair_left and eu_pair_right must have equal length");
                auto lb = left_arr.unchecked<1>();
                auto rb = right_arr.unchecked<1>();
                eu_pair_left_cpp.resize(static_cast<size_t>(lb.shape(0)));
                eu_pair_right_cpp.resize(static_cast<size_t>(rb.shape(0)));
                for (py::ssize_t k = 0; k < lb.shape(0); k++) {
                    eu_pair_left_cpp[static_cast<size_t>(k)]  = lb(k);
                    eu_pair_right_cpp[static_cast<size_t>(k)] = rb(k);
                }
                if (!eu_pair_left_cpp.empty()) {
                    eu_pair_left_ptr  = &eu_pair_left_cpp;
                    eu_pair_right_ptr = &eu_pair_right_cpp;
                }
            }

            // Release GIL before the heavy C++ computation.  Convert native
            // memory failures into Python MemoryError so severe stress exits
            // gracefully instead of surfacing as std::bad_alloc text.
            try {
                py::gil_scoped_release release;
                return prepare_transactions_cpp(
                    X_num, y, B,
                    std::move(col_names_cpp),
                    is_cat, is_int,
                    std::move(ipc_vec),
                    std::move(cat_strs),
                    std::move(cat_valid),
                    eu_pair_left_ptr,
                    eu_pair_right_ptr);
            } catch (const NativeMemoryError& e) {
                PyErr_SetString(PyExc_MemoryError, e.what());
                throw py::error_already_set();
            } catch (const std::bad_alloc&) {
                PyErr_SetString(PyExc_MemoryError,
                    "HUGIML native OOM while preparing transactions");
                throw py::error_already_set();
            }
        },
        py::arg("X_num"), py::arg("y"), py::arg("B"),
        py::arg("col_names"), py::arg("is_cat"), py::arg("is_int"),
        py::arg("X_cat_raw"),
        py::arg("is_precoded") = py::none(),
        py::arg("eu_pair_left") = py::none(),
        py::arg("eu_pair_right") = py::none(),
        "Build utility-annotated TransactionDataCpp from training data.\n\n"
        "is_precoded: optional uint8 array of length p. When is_precoded[j]=1,\n"
        "column j contains pre-computed 0-indexed bin codes; the C++ layer skips\n"
        "re-discretisation and uses the codes directly as bin indices.\n\n"
        "eu_pair_left, eu_pair_right: optional equal-length int32 arrays of\n"
        "column indices identifying interaction-survivor pairs (e.g. from\n"
        "interaction_relaxed_mining). For each pair, the native layer computes\n"
        "the correlation between the mean-centred PRODUCT of the two columns\n"
        "and y, and uses it as a shared external-utility floor for both\n"
        "columns -- bypassing the per-column marginal-correlation admission\n"
        "gate, which is blind to purely interactive signal by construction.");
}

/**
 * bind_build_matrix.cpp — Python bindings for build_train_matrix and
 *                          build_test_matrix.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 */

#include "bind_helpers.hpp"
#include "matrix.hpp"
#include "mining.hpp"

namespace py = pybind11;
using namespace hugiml;

void bind_build_matrix(py::module_& m)
{
    m.def("build_train_matrix",
        [](const TransactionDataCpp& td,
           const std::vector<PatternEntry>& patterns)
        {
            if (patterns.empty())
                throw std::invalid_argument(
                    "patterns list is empty — nothing to build");
            COO coo;
            {
                py::gil_scoped_release release;
                coo = build_train_matrix_cpp(td, patterns);
            }
            return coo_to_tuple(std::move(coo));
        },
        py::arg("td"), py::arg("patterns"),
        "Build COO (rows, cols) arrays for the training binary pattern matrix.");

    m.def("build_test_matrix",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> X_raw,
           const TransactionDataCpp& td,
           py::object X_cat_raw,
           const std::vector<PatternEntry>& patterns)
        {
            validate_2d_array(X_raw, "X_raw");
            if (patterns.empty())
                throw std::invalid_argument(
                    "patterns list is empty — nothing to build");

            int n = static_cast<int>(X_raw.shape(0));
            int p = static_cast<int>(X_raw.shape(1));

            std::vector<std::vector<std::string>> cat_strs;
            std::vector<std::vector<bool>>        cat_valid;

            if (!X_cat_raw.is_none() && p > 0) {
                const auto& is_cat_v = td.is_cat_v;
                if (static_cast<int>(is_cat_v.size()) == p) {
                    cat_strs.resize(p);
                    cat_valid.resize(p);
                    py::list raw_list = X_cat_raw.cast<py::list>();
                    for (int j = 0; j < p; j++) {
                        if (!is_cat_v[j]) continue;
                        py::object col_obj = raw_list[j].cast<py::object>();
                        if (col_obj.is_none()) continue;
                        cat_strs[j].resize(n);
                        cat_valid[j].resize(n, false);
                        py::array arr = col_obj.cast<py::array>();
                        py::list  lst = arr.attr("tolist")().cast<py::list>();
                        for (int r = 0; r < n; r++) {
                            py::object val = lst[r];
                            if (val.is_none()) continue;
                            try {
                                double dv = val.cast<double>();
                                if (std::isnan(dv)) continue;
                            } catch (...) {}
                            cat_valid[j][r] = true;
                            cat_strs[j][r]  = py::str(val).cast<std::string>();
                        }
                    }
                }
            }

            COO coo;
            {
                py::gil_scoped_release release;
                coo = build_test_matrix_cpp(X_raw, td,
                                             std::move(cat_strs),
                                             std::move(cat_valid),
                                             patterns);
            }
            return coo_to_tuple(std::move(coo));
        },
        py::arg("X_raw"), py::arg("td"), py::arg("X_cat_raw"),
        py::arg("patterns"),
        "Apply training bins to test data and build pattern matrix in one pass.");
}

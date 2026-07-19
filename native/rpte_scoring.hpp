/**
 * rpte_scoring.hpp — declarations for the kernels defined in
 * rpte_scoring.cpp, so other translation units (the tree-growth engine
 * in rpte_tree.cpp) can call them directly instead of only through the
 * Python bindings.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 */

#pragma once

#include "pybind_common.hpp"

namespace py = pybind11;

py::array_t<double> rpte_shared_child_scores(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> root_bit,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> child_bits,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> y,
    int64_t min_samples_leaf
);

py::array_t<double> rpte_shared_extension_scores(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> base_codes,
    int64_t n_base_cells,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> child_bits,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> y,
    int64_t min_samples_leaf
);

py::dict rpte_raw_pair_root_search(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xg,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> yg,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> raw_col_positions,
    py::array_t<double, py::array::c_style | py::array::forcecast> raw_thresholds,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> raw_is_binary,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> Cg_pool,
    const std::vector<int32_t>& ops,
    int64_t max_root_thresholds,
    int64_t min_samples_leaf,
    int64_t max_pairs,
    int64_t top_children_per_candidate,
    uint64_t seed
);

double rpte_partition_ig_bits(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> codes,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> y
);

py::dict rpte_newton_leaf_values(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> leaf_ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> r,
    py::array_t<double, py::array::c_style | py::array::forcecast> w,
    double ridge
);

double rpte_binomial_deviance(
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> p,
    double eps
);

void bind_rpte_scoring(py::module_& m);

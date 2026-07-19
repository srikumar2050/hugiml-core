/**
 * rpte_scoring.cpp — pybind11 bindings for RPTE's information-gain
 * scoring kernels.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Each function here is a thin marshaling wrapper: convert the
 * incoming numpy arrays to plain C++ types, call the corresponding
 * pure function in rpte_core.hpp / rpte_pair_search_core.hpp, convert
 * the result back. The actual math lives in those headers, shared
 * with the native tree-growth engine (rpte_tree.cpp), which calls
 * them directly (bypassing pybind11 marshaling and GIL management,
 * since it runs its entire search with the GIL already released).
 *
 * rpte_shared_child_scores / rpte_shared_extension_scores /
 * rpte_raw_pair_root_search are exposed here for direct use and
 * backward compatibility, though the tree-growth engine no longer
 * calls back into these bindings itself (see rpte_core.hpp's module
 * comment for why: calling a function that releases the GIL from
 * inside a region that already released it is unsafe).
 */

#include "pybind_common.hpp"
#include "rpte_scoring.hpp"

#include "rpte_core.hpp"
#include "rpte_pair_search_core.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace py = pybind11;
using namespace hugiml::rpte_core;

namespace {

template <typename T>
std::vector<T> to_vector(const py::array_t<T, py::array::c_style | py::array::forcecast>& arr) {
    auto info = arr.request();
    const T* ptr = static_cast<const T*>(info.ptr);
    return std::vector<T>(ptr, ptr + info.shape[0]);
}

py::array_t<double> to_array(const std::vector<double>& v) {
    py::array_t<double> out(static_cast<py::ssize_t>(v.size()));
    std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(double));
    return out;
}

}  // namespace


py::array_t<double> rpte_shared_child_scores(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> root_bit_arr,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> child_bits_arr,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> y_arr,
    int64_t min_samples_leaf
) {
    auto root_info = root_bit_arr.request();
    auto child_info = child_bits_arr.request();
    auto y_info = y_arr.request();
    if (root_info.ndim != 1) throw std::invalid_argument("root_bit must be 1D.");
    if (child_info.ndim != 2) throw std::invalid_argument("child_bits must be 2D (n, p).");
    if (y_info.ndim != 1) throw std::invalid_argument("y must be 1D.");
    const size_t n = static_cast<size_t>(root_info.shape[0]);
    const size_t p = static_cast<size_t>(child_info.shape[1]);
    if (static_cast<size_t>(child_info.shape[0]) != n) throw std::invalid_argument("child_bits row count must match root_bit.");
    if (static_cast<size_t>(y_info.shape[0]) != n) throw std::invalid_argument("y row count must match root_bit.");

    std::vector<uint8_t> root_bit = to_vector<uint8_t>(root_bit_arr);
    std::vector<int8_t> y = to_vector<int8_t>(y_arr);
    const uint8_t* child_bits = static_cast<const uint8_t*>(child_info.ptr);

    std::vector<double> scores;
    {
        py::gil_scoped_release release;
        scores = shared_child_scores(root_bit, child_bits, n, p, y, min_samples_leaf);
    }
    return to_array(scores);
}


py::array_t<double> rpte_shared_extension_scores(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> base_codes_arr,
    int64_t n_base_cells,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> child_bits_arr,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> y_arr,
    int64_t min_samples_leaf
) {
    auto base_info = base_codes_arr.request();
    auto child_info = child_bits_arr.request();
    auto y_info = y_arr.request();
    if (base_info.ndim != 1) throw std::invalid_argument("base_codes must be 1D.");
    if (child_info.ndim != 2) throw std::invalid_argument("child_bits must be 2D (n, p).");
    if (y_info.ndim != 1) throw std::invalid_argument("y must be 1D.");
    const size_t n = static_cast<size_t>(base_info.shape[0]);
    const size_t p = static_cast<size_t>(child_info.shape[1]);
    if (static_cast<size_t>(child_info.shape[0]) != n) throw std::invalid_argument("child_bits row count must match base_codes.");
    if (static_cast<size_t>(y_info.shape[0]) != n) throw std::invalid_argument("y row count must match base_codes.");
    if (n_base_cells <= 0) throw std::invalid_argument("n_base_cells must be positive.");

    std::vector<int64_t> base_codes = to_vector<int64_t>(base_codes_arr);
    std::vector<int8_t> y = to_vector<int8_t>(y_arr);
    const uint8_t* child_bits = static_cast<const uint8_t*>(child_info.ptr);

    std::vector<double> scores;
    {
        py::gil_scoped_release release;
        scores = shared_extension_scores(base_codes, n_base_cells, child_bits, n, p, y, min_samples_leaf);
    }
    return to_array(scores);
}


py::dict rpte_raw_pair_root_search(
    py::array_t<double, py::array::c_style | py::array::forcecast> Xg_arr,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> yg_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> raw_col_positions_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> raw_thresholds_arr,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> raw_is_binary_arr,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> Cg_pool_arr,
    const std::vector<int32_t>& ops,
    int64_t max_root_thresholds,
    int64_t min_samples_leaf,
    int64_t max_pairs,
    int64_t top_children_per_candidate,
    uint64_t seed
) {
    auto Xg_info = Xg_arr.request();
    auto yg_info = yg_arr.request();
    auto raw_pos_info = raw_col_positions_arr.request();
    auto Cg_info = Cg_pool_arr.request();
    if (Xg_info.ndim != 2) throw std::invalid_argument("Xg must be 2D (n, p_full).");
    if (yg_info.ndim != 1) throw std::invalid_argument("yg must be 1D.");
    if (raw_pos_info.ndim != 1) throw std::invalid_argument("raw_col_positions must be 1D.");
    if (Cg_info.ndim != 2) throw std::invalid_argument("Cg_pool must be 2D (n, m).");

    const size_t n = static_cast<size_t>(Xg_info.shape[0]);
    const size_t p_full = static_cast<size_t>(Xg_info.shape[1]);
    const size_t mu = static_cast<size_t>(Cg_info.shape[1]);
    if (static_cast<size_t>(yg_info.shape[0]) != n) throw std::invalid_argument("yg row count must match Xg.");
    if (static_cast<size_t>(Cg_info.shape[0]) != n) throw std::invalid_argument("Cg_pool row count must match Xg.");

    std::vector<int8_t> yg = to_vector<int8_t>(yg_arr);
    std::vector<int32_t> raw_col_positions = to_vector<int32_t>(raw_col_positions_arr);
    std::vector<double> raw_thresholds = to_vector<double>(raw_thresholds_arr);
    std::vector<uint8_t> raw_is_binary = to_vector<uint8_t>(raw_is_binary_arr);
    const double* Xg = static_cast<const double*>(Xg_info.ptr);
    const uint8_t* Cg_pool = static_cast<const uint8_t*>(Cg_info.ptr);

    PairSearchResult result;
    {
        py::gil_scoped_release release;
        result = raw_pair_root_search_core(
            Xg, n, p_full, yg.data(), raw_col_positions, raw_thresholds, raw_is_binary,
            Cg_pool, mu, ops, max_root_thresholds, min_samples_leaf, max_pairs,
            top_children_per_candidate, seed);
    }

    const py::ssize_t k_out = static_cast<py::ssize_t>(result.a_idx.size());
    const py::ssize_t width = static_cast<py::ssize_t>(std::max<int64_t>(1, top_children_per_candidate));
    py::array_t<int32_t> a_idx(k_out), b_idx(k_out), op_out(k_out);
    py::array_t<double> threshold_out(k_out), grow_gain_out(k_out);
    py::array_t<uint8_t> is_binarized_out(k_out);
    py::array_t<int32_t> child_positions(std::vector<py::ssize_t>{k_out, width});
    py::array_t<double> child_scores_out(std::vector<py::ssize_t>{k_out, width});
    {
        auto a_mut = a_idx.mutable_unchecked<1>();
        auto b_mut = b_idx.mutable_unchecked<1>();
        auto op_mut = op_out.mutable_unchecked<1>();
        auto thr_mut = threshold_out.mutable_unchecked<1>();
        auto gain_mut = grow_gain_out.mutable_unchecked<1>();
        auto bin_mut = is_binarized_out.mutable_unchecked<1>();
        auto cp_mut = child_positions.mutable_unchecked<2>();
        auto cs_mut = child_scores_out.mutable_unchecked<2>();
        for (py::ssize_t k = 0; k < k_out; ++k) {
            const size_t ku = static_cast<size_t>(k);
            a_mut(k) = result.a_idx[ku];
            b_mut(k) = result.b_idx[ku];
            op_mut(k) = result.op[ku];
            thr_mut(k) = result.threshold[ku];
            gain_mut(k) = result.grow_gain[ku];
            bin_mut(k) = result.is_binarized[ku];
            for (py::ssize_t w = 0; w < width; ++w) {
                const size_t wu = static_cast<size_t>(w);
                if (wu < result.child_positions[ku].size()) {
                    cp_mut(k, w) = result.child_positions[ku][wu];
                    cs_mut(k, w) = result.child_scores[ku][wu];
                } else {
                    cp_mut(k, w) = -1;
                    cs_mut(k, w) = -std::numeric_limits<double>::infinity();
                }
            }
        }
    }

    py::dict out;
    out["a_idx"] = a_idx;
    out["b_idx"] = b_idx;
    out["op"] = op_out;
    out["threshold"] = threshold_out;
    out["is_binarized"] = is_binarized_out;
    out["grow_gain"] = grow_gain_out;
    out["child_positions"] = child_positions;
    out["child_scores"] = child_scores_out;
    return out;
}


double rpte_partition_ig_bits(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> codes_arr,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> y_arr
) {
    auto codes_info = codes_arr.request();
    auto y_info = y_arr.request();
    if (codes_info.ndim != 1) throw std::invalid_argument("codes must be 1D.");
    if (y_info.ndim != 1) throw std::invalid_argument("y must be 1D.");
    if (codes_info.shape[0] != y_info.shape[0]) throw std::invalid_argument("y must match codes in length.");

    std::vector<int64_t> codes = to_vector<int64_t>(codes_arr);
    std::vector<int8_t> y = to_vector<int8_t>(y_arr);
    double result;
    {
        py::gil_scoped_release release;
        result = partition_ig_bits(codes, y);
    }
    return result;
}


py::dict rpte_newton_leaf_values(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> leaf_ids_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> r_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> w_arr,
    double ridge
) {
    auto leaf_info = leaf_ids_arr.request();
    auto r_info = r_arr.request();
    auto w_info = w_arr.request();
    if (leaf_info.ndim != 1) throw std::invalid_argument("leaf_ids must be 1D.");
    if (r_info.ndim != 1) throw std::invalid_argument("r must be 1D.");
    if (w_info.ndim != 1) throw std::invalid_argument("w must be 1D.");
    if (r_info.shape[0] != leaf_info.shape[0] || w_info.shape[0] != leaf_info.shape[0])
        throw std::invalid_argument("r and w must match leaf_ids in length.");

    std::vector<int64_t> leaf_ids = to_vector<int64_t>(leaf_ids_arr);
    std::vector<double> r = to_vector<double>(r_arr);
    std::vector<double> w = to_vector<double>(w_arr);
    std::unordered_map<int64_t, double> values;
    {
        py::gil_scoped_release release;
        values = newton_leaf_values(leaf_ids, r, w, ridge);
    }
    // Sorted-key order, matching the previous implementation, for
    // deterministic iteration on the Python side.
    std::vector<int64_t> keys;
    keys.reserve(values.size());
    for (const auto& kv : values) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    py::dict out;
    for (int64_t key : keys) out[py::int_(key)] = values[key];
    return out;
}


double rpte_binomial_deviance(
    py::array_t<double, py::array::c_style | py::array::forcecast> y_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> p_arr,
    double eps
) {
    auto y_info = y_arr.request();
    auto p_info = p_arr.request();
    if (y_info.ndim != 1) throw std::invalid_argument("y must be 1D.");
    if (p_info.ndim != 1) throw std::invalid_argument("p must be 1D.");
    if (p_info.shape[0] != y_info.shape[0]) throw std::invalid_argument("p must match y in length.");

    std::vector<double> y = to_vector<double>(y_arr);
    std::vector<double> p = to_vector<double>(p_arr);
    double result;
    {
        py::gil_scoped_release release;
        result = binomial_deviance(y, p, eps);
    }
    return result;
}


void bind_rpte_scoring(py::module_& m)
{
    m.def(
        "rpte_shared_child_scores",
        &rpte_shared_child_scores,
        py::arg("root_bit"),
        py::arg("child_bits"),
        py::arg("y"),
        py::arg("min_samples_leaf"),
        "Native 4-cell (root x child) partition-IG scoring for every candidate "
        "child column at once, given a fixed root split."
    );
    m.def(
        "rpte_shared_extension_scores",
        &rpte_shared_extension_scores,
        py::arg("base_codes"),
        py::arg("n_base_cells"),
        py::arg("child_bits"),
        py::arg("y"),
        py::arg("min_samples_leaf"),
        "Native extension-IG scoring: adds one more shared bit to an existing "
        "n_base_cells-cell partition for every candidate column at once."
    );
    m.def(
        "rpte_raw_pair_root_search",
        &rpte_raw_pair_root_search,
        py::arg("Xg"),
        py::arg("yg"),
        py::arg("raw_col_positions"),
        py::arg("raw_thresholds"),
        py::arg("raw_is_binary"),
        py::arg("Cg_pool"),
        py::arg("ops"),
        py::arg("max_root_thresholds"),
        py::arg("min_samples_leaf"),
        py::arg("max_pairs"),
        py::arg("top_children_per_candidate") = 3,
        py::arg("seed") = 0,
        "Native fused hot loop for the raw-pair lookahead root-candidate "
        "search: for many candidate raw-feature pairs x operations x "
        "thresholds, scores every combination against a precomputed "
        "child-candidate pool."
    );
    m.def(
        "rpte_partition_ig_bits",
        &rpte_partition_ig_bits,
        py::arg("codes"),
        py::arg("y"),
        "Information gain in bits of a partition (arbitrary integer cell "
        "ids) against binary y."
    );
    m.def(
        "rpte_newton_leaf_values",
        &rpte_newton_leaf_values,
        py::arg("leaf_ids"),
        py::arg("r"),
        py::arg("w"),
        py::arg("ridge") = 1.0,
        "Exact per-leaf Newton step sum(r)/(sum(w)+ridge) for each distinct "
        "leaf id."
    );
    m.def(
        "rpte_binomial_deviance",
        &rpte_binomial_deviance,
        py::arg("y"),
        py::arg("p"),
        py::arg("eps") = 1e-12,
        "Mean binomial deviance for binary y and predicted probabilities p."
    );
}

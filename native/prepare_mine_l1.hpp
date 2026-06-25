/**
 * prepare_mine_l1.hpp — Fused L=1 hot path: adaptive B-selection +
 *                        transaction-free prep + mining + COO matrix.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Overview
 * ────────
 * For L=1 singleton-only mining the general pipeline pays for several
 * abstractions it does not need:
 *
 *   • The TransList (n × avg_items entries of (iid, iu) pairs) —
 *     replaced here by a compact per-item TID inverted index.
 *   • Two hash-map lookups per (row, col) in Phase 2 (bn2id + tu) —
 *     replaced by direct array accesses into small per-column arrays.
 *   • A separate mine_patterns_l1 scan over the TransList —
 *     class counts accumulated inline during the single fused scan.
 *   • The O(n × n_pats) bitmap build_train_matrix —
 *     replaced by direct COO copy from item_tids.
 *   • TWU computation (inter-column row-sum) —
 *     not needed: RIU is both tighter and sufficient as the minU seed.
 *
 * For the adaptive path, C++ elbow_stop_nb_cpp replaces the Python
 * _apply_adaptive_binning() loop, giving ~2× speedup per column and
 * enabling OMP column-parallelism.
 *
 * Interface
 * ─────────
 * Two entry points share the same fused scan:
 *
 *   prepare_and_mine_l1_cpp(X_num, y, B, col_names, is_cat, is_int,
 *                           is_precoded, cat_raw_strs, cat_raw_valid,
 *                           K, G, timeout_s)
 *     → L1FitResult
 *
 *   select_adaptive_bins_cpp(X_num, y, n_cls, col_names, is_cat,
 *                             candidates, ratio)
 *     → AdaptiveBinResult
 *
 * The Python layer calls select_adaptive_bins_cpp first (when
 * adaptive_binning=True), packs its outputs into _bin_edges_ / per_feature_b_
 * / ig_scores_ / _adaptive_code_label_map_, then calls prepare_and_mine_l1_cpp
 * with the pre-coded X.  For fixed-B, prepare_and_mine_l1_cpp is called
 * directly.
 *
 * Both functions are GIL-safe (GIL released by the binding layer after
 * Python-object extraction).
 */

#pragma once

#include "mining.hpp"        // PatternEntry, TransactionDataCpp
#include "transaction.hpp"

#include <vector>
#include <string>

namespace hugiml {

// ── Result types ──────────────────────────────────────────────────────────────

/// Per-column output of C++ adaptive B selection.
struct ColAdaptResult {
    int                  chosen_b;     ///< Elbow-stop selected B_j
    std::vector<double>  edges;        ///< Quantile edges [0..B_j] for this column
    std::vector<double>  ig_scores;    ///< IG score per candidate evaluated
};

/// Output of the fused L=1 hot path.
struct L1FitResult {
    TransactionDataCpp           td;         ///< Phase-1 artefacts; td.transactions is EMPTY
    std::vector<PatternEntry>    patterns;   ///< Mined top-K singletons
    std::vector<int32_t>         coo_rows;   ///< COO row indices for x_train_hup
    std::vector<int32_t>         coo_cols;   ///< COO column indices for x_train_hup
    std::vector<ColAdaptResult>  adaptive_cols; ///< Metadata for fused adaptive path
    std::vector<int>             adaptive_num_col_indices; ///< Original numeric col indices
    std::vector<std::string>     original_feature_names; ///< Downstream original-feature names scored during native preparation
    std::vector<double>          original_feature_scores; ///< IG-like scores aligned with original_feature_names
};

/// Output of select_adaptive_bins_cpp.
struct AdaptiveBinResult {
    std::vector<ColAdaptResult>  cols;        ///< One entry per numerical column
    std::vector<int32_t>         X_codes_flat; ///< Row-major int32 bin codes (n × n_num_cols)
                                               ///< -1 for non-finite/missing cells
    int                          n_rows;
    int                          n_num_cols;   ///< Number of numeric (non-cat) columns
    std::vector<int>             num_col_indices; ///< Original column indices for each entry
};

// ── Entry points ──────────────────────────────────────────────────────────────

/// Fused prepare+mine+matrix for L=1.
/// Handles fixed-B (is_precoded=False) and adaptive pre-coded (is_precoded=True) columns.
/// For fixed-B float columns runs kbins in Phase 1; for precoded columns skips kbins.
/// timeout_s: 0 = no limit.
L1FitResult prepare_and_mine_l1_cpp(
    const pybind11::array_t<double,  pybind11::array::c_style | pybind11::array::forcecast>& X_num_arr,
    const pybind11::array_t<int64_t, pybind11::array::c_style | pybind11::array::forcecast>& y_arr,
    int B,
    std::vector<std::string>               col_names,
    const pybind11::array_t<uint8_t, pybind11::array::forcecast>& is_cat_arr,
    const pybind11::array_t<uint8_t, pybind11::array::forcecast>& is_int_arr,
    std::vector<bool>                      is_precoded,
    std::vector<std::vector<std::string>>  cat_raw_strs,
    std::vector<std::vector<bool>>         cat_raw_valid,
    int K, double G, double timeout_s, bool compute_original_scores);

L1FitResult prepare_and_mine_l1_cpp(
    const pybind11::array_t<float,  pybind11::array::c_style>& X_num_arr,
    const pybind11::array_t<int64_t, pybind11::array::c_style | pybind11::array::forcecast>& y_arr,
    int B,
    std::vector<std::string>               col_names,
    const pybind11::array_t<uint8_t, pybind11::array::forcecast>& is_cat_arr,
    const pybind11::array_t<uint8_t, pybind11::array::forcecast>& is_int_arr,
    std::vector<bool>                      is_precoded,
    std::vector<std::vector<std::string>>  cat_raw_strs,
    std::vector<std::vector<bool>>         cat_raw_valid,
    int K, double G, double timeout_s, bool compute_original_scores);

/// C++ adaptive B selection (replaces Python _apply_adaptive_binning).
/// Skips is_cat columns.  Returns per-column results + row-major int32 code matrix.
AdaptiveBinResult select_adaptive_bins_cpp(
    const pybind11::array_t<double,  pybind11::array::c_style | pybind11::array::forcecast>& X_num_arr,
    const pybind11::array_t<int64_t, pybind11::array::c_style | pybind11::array::forcecast>& y_arr,
    int n_cls,
    const std::vector<std::string>&        col_names,
    const pybind11::array_t<uint8_t, pybind11::array::forcecast>& is_cat_arr,
    const std::vector<int>&                candidates,
    double                                 ratio,
    double                                 adaptive_sample_frac = 1.0,
    uint64_t                               adaptive_sample_seed = 42);

/// Fast fixed-B dense numeric L=1 hot path.  It keeps numeric columns numeric,
/// skips non-finite cells, parallelises fixed-bin column preparation and the
/// first row scan, and avoids materialising transactions.
L1FitResult prepare_and_mine_l1_fixed_numeric_cpp(
    const pybind11::array_t<double,  pybind11::array::c_style | pybind11::array::forcecast>& X_num_arr,
    const pybind11::array_t<int64_t, pybind11::array::c_style | pybind11::array::forcecast>& y_arr,
    int B,
    std::vector<std::string> col_names,
    const pybind11::array_t<uint8_t, pybind11::array::forcecast>& is_int_arr,
    int K, double G, double timeout_s, bool compute_original_scores);

L1FitResult prepare_and_mine_l1_fixed_numeric_cpp(
    const pybind11::array_t<float,  pybind11::array::c_style>& X_num_arr,
    const pybind11::array_t<int64_t, pybind11::array::c_style | pybind11::array::forcecast>& y_arr,
    int B,
    std::vector<std::string> col_names,
    const pybind11::array_t<uint8_t, pybind11::array::forcecast>& is_int_arr,
    int K, double G, double timeout_s, bool compute_original_scores);

/// Fused adaptive-B + L1 hot path.  This combines adaptive edge selection and
/// the L1 scan without materialising X_codes_flat or a Python binned DataFrame.
L1FitResult prepare_and_mine_l1_adaptive_cpp(
    const pybind11::array_t<double,  pybind11::array::c_style | pybind11::array::forcecast>& X_num_arr,
    const pybind11::array_t<int64_t, pybind11::array::c_style | pybind11::array::forcecast>& y_arr,
    std::vector<std::string>               col_names,
    const pybind11::array_t<uint8_t, pybind11::array::forcecast>& is_cat_arr,
    const pybind11::array_t<uint8_t, pybind11::array::forcecast>& is_int_arr,
    std::vector<std::vector<std::string>>  cat_raw_strs,
    std::vector<std::vector<bool>>         cat_raw_valid,
    const std::vector<int>&                candidates,
    double                                 ratio,
    int K, double G, double timeout_s, bool compute_original_scores,
    double adaptive_sample_frac = 1.0, uint64_t adaptive_sample_seed = 42);

L1FitResult prepare_and_mine_l1_adaptive_cpp(
    const pybind11::array_t<float,  pybind11::array::c_style>& X_num_arr,
    const pybind11::array_t<int64_t, pybind11::array::c_style | pybind11::array::forcecast>& y_arr,
    std::vector<std::string>               col_names,
    const pybind11::array_t<uint8_t, pybind11::array::forcecast>& is_cat_arr,
    const pybind11::array_t<uint8_t, pybind11::array::forcecast>& is_int_arr,
    std::vector<std::vector<std::string>>  cat_raw_strs,
    std::vector<std::vector<bool>>         cat_raw_valid,
    const std::vector<int>&                candidates,
    double                                 ratio,
    int K, double G, double timeout_s, bool compute_original_scores,
    double adaptive_sample_frac = 1.0, uint64_t adaptive_sample_seed = 42);

}  // namespace hugiml

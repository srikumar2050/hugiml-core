/**
 * matrix.cpp — Feature matrix construction implementation
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Uses bitmap-based pattern matching (bitwise AND subset test) and
 * OpenMP-parallel scanning for large workloads (n*n_pats > 100k).
 *
 * Bitmaps are stored as flat row-major arrays of uint64_t words for
 * contiguous memory access and better cache performance than the previous
 * vector<vector<Word>> layout.
 *
 * Returns plain C++ COO vectors; callers create Python objects only after
 * reacquiring the GIL.
 */

#include "matrix.hpp"
#include "resource_guard.hpp"
#include <unordered_set>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hugiml {

static constexpr int BITS = 64;
using Word = uint64_t;

static inline int n_words(int max_item) {
    return (max_item + BITS) / BITS;
}

static inline void bit_set(Word* bv, int item) {
    if (item < 0) return;
    bv[item / BITS] |= (Word(1) << (item % BITS));
}

// pat is a dense row of nw words; txn is a dense row of nw words.
static inline bool bit_subset(const Word* pat, const Word* txn, int nw) {
    for (int w = 0; w < nw; w++)
        if ((pat[w] & txn[w]) != pat[w]) return false;
    return true;
}

COO build_matrix_cpp(const TransList& transactions,
                     const std::vector<PatternEntry>& patterns,
                     int n) {
    int n_pats = static_cast<int>(patterns.size());
    if (n_pats == 0 || n == 0) return {};

    // Find max item ID across patterns and transactions
    int max_item = 0;
    for (auto& pe : patterns)
        for (int it : pe.items)
            if (it > max_item) max_item = it;
    for (int i = 0; i < n && i < static_cast<int>(transactions.size()); i++)
        for (int it : transactions[i])
            if (it > 0 && it > max_item) max_item = it;

    int nw = n_words(max_item + 1);

    ensure_native_memory_available(
        static_cast<uint64_t>(n) * static_cast<uint64_t>(nw) * sizeof(Word) +
        static_cast<uint64_t>(n_pats) * static_cast<uint64_t>(nw) * sizeof(Word) +
        static_cast<uint64_t>(n) * static_cast<uint64_t>(std::max(n_pats, 1)) / 4ULL * sizeof(int32_t) * 2ULL,
        "build_matrix bitmap/COO buffers");

    // Flat row-major bitmap arrays: txn_bm[i * nw + w], pat_bm[pi * nw + w]
    std::vector<Word> txn_bm(static_cast<size_t>(n) * nw, 0);
    for (int i = 0; i < n && i < static_cast<int>(transactions.size()); i++) {
        Word* row = txn_bm.data() + static_cast<size_t>(i) * nw;
        for (int it : transactions[i])
            if (it > 0) bit_set(row, it);
    }

    std::vector<Word> pat_bm(static_cast<size_t>(n_pats) * nw, 0);
    for (int pi = 0; pi < n_pats; pi++) {
        Word* row = pat_bm.data() + static_cast<size_t>(pi) * nw;
        for (int it : patterns[pi].items)
            bit_set(row, it);
    }

    bool use_parallel = false;
#ifdef _OPENMP
    use_parallel = (static_cast<int64_t>(n) * n_pats > 100000);
#endif

    std::vector<int32_t> rows_v, cols_v;

    if (use_parallel) {
#ifdef _OPENMP
        int n_threads = omp_get_max_threads();
        std::vector<std::vector<int32_t>> local_rows(n_threads);
        std::vector<std::vector<int32_t>> local_cols(n_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& lr = local_rows[tid];
            auto& lc = local_cols[tid];
            int64_t est = static_cast<int64_t>(n) * n_pats / n_threads / 4;
            lr.reserve(static_cast<size_t>(est));
            lc.reserve(static_cast<size_t>(est));

            #pragma omp for schedule(dynamic, 16)
            for (int pi = 0; pi < n_pats; pi++) {
                const Word* prow = pat_bm.data() + static_cast<size_t>(pi) * nw;
                for (int r = 0; r < n; r++) {
                    const Word* trow = txn_bm.data() + static_cast<size_t>(r) * nw;
                    if (bit_subset(prow, trow, nw)) {
                        lr.push_back(r);
                        lc.push_back(pi);
                    }
                }
            }
        }

        size_t total = 0;
        for (auto& v : local_rows) total += v.size();
        rows_v.reserve(total);
        cols_v.reserve(total);
        for (int t = 0; t < n_threads; t++) {
            rows_v.insert(rows_v.end(), local_rows[t].begin(), local_rows[t].end());
            cols_v.insert(cols_v.end(), local_cols[t].begin(), local_cols[t].end());
        }
        return {std::move(rows_v), std::move(cols_v)};
#endif
    }

    rows_v.reserve(n);
    cols_v.reserve(n);
    for (int pi = 0; pi < n_pats; pi++) {
        const Word* prow = pat_bm.data() + static_cast<size_t>(pi) * nw;
        for (int r = 0; r < n; r++) {
            const Word* trow = txn_bm.data() + static_cast<size_t>(r) * nw;
            if (bit_subset(prow, trow, nw)) {
                rows_v.push_back(r);
                cols_v.push_back(pi);
            }
        }
    }
    return {std::move(rows_v), std::move(cols_v)};
}

COO build_train_matrix_cpp(const TransactionDataCpp& td,
                           const std::vector<PatternEntry>& patterns) {
    int n = static_cast<int>(td.transactions.size());
    return build_matrix_cpp(td.transactions, patterns, n);
}

COO build_test_matrix_cpp(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& X_raw_arr,
    const TransactionDataCpp& td,
    std::vector<std::vector<std::string>> cat_strs,
    std::vector<std::vector<bool>>        cat_valid,
    const std::vector<PatternEntry>& patterns) {

    // All Python object access has been done by the caller; this function is
    // pure C++ and safe to call with the GIL released.

    auto Xb = X_raw_arr.unchecked<2>();
    int n   = static_cast<int>(Xb.shape(0));
    int p   = static_cast<int>(Xb.shape(1));

    // Build label→code maps for categorical columns
    std::vector<std::unordered_map<std::string, int>> label2code(p);
    for (int j = 0; j < p; j++) {
        if (!td.is_cat_v[j]) continue;
        const auto& cats = td.cat_categories[j];
        label2code[j].reserve(cats.size());
        for (int i = 0; i < static_cast<int>(cats.size()); i++)
            label2code[j][cats[i]] = i;
    }

    // Validate pre-extracted cat data size
    bool have_cat_strs = (!cat_strs.empty() &&
                          static_cast<int>(cat_strs.size()) == p);
    bool have_cat_valid = (!cat_valid.empty() &&
                           static_cast<int>(cat_valid.size()) == p);

    ensure_native_memory_available(
        static_cast<uint64_t>(n) * (sizeof(Trans) + 8ULL) +
        static_cast<uint64_t>(n) * static_cast<uint64_t>(std::max(p, 1)) * sizeof(TItem),
        "build_test_matrix temporary transactions");
    TransList test_trans(n);
    for (int r = 0; r < n; r++) {
        Trans row;
        for (int j = 0; j < p; j++) {
            int bi;
            if (td.is_cat_v[j]) {
                if (!have_cat_strs || cat_strs[j].empty()) continue;
                if (!have_cat_valid || !cat_valid[j][r]) continue;
                auto it = label2code[j].find(cat_strs[j][r]);
                if (it == label2code[j].end()) continue;
                bi = it->second + 1;
            } else if (!td.is_precoded_v.empty() && td.is_precoded_v[j]) {
                // Pre-coded column: Xb(r,j) is the 0-indexed bin code stored
                // as float64.  np.nan encodes "no observation" — skip to
                // generate no item for this (row, feature) pair, matching the
                // behaviour of missing values on every other column type.
                // static_cast<int>(NaN) is undefined behaviour in C++, so the
                // finite check must come before the cast.
                double v = Xb(r, j);
                if (!std::isfinite(v)) continue;
                int code = static_cast<int>(v);
                int nb   = td.nb_col[j];
                bi = std::max(1, std::min(code + 1, nb));
            } else {
                // Numeric (float or integer) column.
                // Non-finite values (NaN, Inf) must generate no item — the
                // same "not observed" contract as every other column type.
                // std::upper_bound with a NaN comparand has undefined
                // behaviour, so guard before the binary search.
                const auto& edges = td.all_edges[j];
                int          nb   = td.nb_col[j];
                double raw = Xb(r, j);
                if (!std::isfinite(raw)) continue;
                double val = td.is_int_v[j]
                             ? raw
                             : (raw - td.col_min[j]) / td.col_range[j];
                auto it = std::upper_bound(
                    edges.begin() + 1, edges.end() - 1, val);
                bi = static_cast<int>(it - (edges.begin() + 1)) + 1;
                bi = std::max(1, std::min(bi, nb));
            }
            int iid = td.item_id_for_bin(bi, j);
            if (iid > 0) row.push_back(iid);
        }
        test_trans[r] = row.empty() ? Trans{-1} : row;
    }

    return build_matrix_cpp(test_trans, patterns, n);
}

}  // namespace hugiml

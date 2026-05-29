/**
 * transaction.hpp — Transaction data structures
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace hugiml {

using TItem     = std::pair<int, double>;
using Trans     = std::vector<TItem>;
using TransList = std::vector<Trans>;

struct TransactionDataCpp {
    TransList                                        transactions;
    std::vector<double>                              item_twu;
    std::unordered_map<int, std::string>             item_map;
    std::vector<double>                              RIU;
    // item_col[item_id - 1] = original source feature/column index.
    // Used by mining to enforce the exact structural constraint that a
    // pattern cannot contain two mutually-exclusive bins from the same feature.
    std::vector<int>                                 item_col;

    // cat_corr and per-column bin-code vectors are construction-time only;
    // they are kept as locals inside prepare_transactions_cpp.  Only the
    // active-column bin streams (O(n * n_active_cols)) are held during Pass 3
    // and freed before this struct is returned.

    int disc_n = 0, disc_p = 0;

    // bn2id: bname -> item ID.  Presence of a key means eiu > 0 for that bin.
    // colnew_set has been removed (Fix 4a); bn2id alone is sufficient for
    // Pass 3 filtering via a single find() instead of a set-find + map-at pair.
    std::unordered_map<int, int>                     bn2id;

    std::vector<int>                                 nb_col;
    std::vector<std::vector<double>>                 ber;
    std::vector<double>                              cv;
    std::vector<std::vector<double>>                 all_edges;
    std::vector<double>                              col_min, col_range;
    std::vector<bool>                                is_cat_v, is_int_v;
    // is_precoded_v[j] = true: column j contains pre-computed 0-indexed bin
    // codes from Python adaptive binning.  The C++ layer skips re-discretisation
    // and uses the code value directly as the 0-indexed bin (bi = code + 1).
    // Edges are set to [0, 1, ..., B_j] for compatibility with build_test_matrix.
    std::vector<bool>                                is_precoded_v;

    std::vector<std::vector<std::string>>            cat_categories;

    // -- Bin-key encoding -------------------------------------------------
    // bkey(bi, j) = bi * bkey_stride + j
    // stride = max(p, 10000) so datasets with >= 10000 columns don't
    // produce silent key collisions.  Backward-compatible for p < 10000.
    int bkey_stride = 10000;

    int bkey(int bi, int j) const { return bi * bkey_stride + j; }

    /// Estimate the memory footprint of this object in bytes.
    size_t memory_usage_bytes() const {
        size_t total = sizeof(*this);
        // transactions
        for (auto& t : transactions)
            total += sizeof(Trans) + t.capacity() * sizeof(TItem);
        total += transactions.capacity() * sizeof(Trans);
        // vectors
        total += item_twu.capacity() * sizeof(double);
        total += RIU.capacity() * sizeof(double);
        total += item_col.capacity() * sizeof(int);
        total += col_min.capacity() * sizeof(double);
        total += col_range.capacity() * sizeof(double);
        total += cv.capacity() * sizeof(double);
        // edges
        for (auto& e : all_edges)
            total += sizeof(std::vector<double>) + e.capacity() * sizeof(double);
        for (auto& b : ber)
            total += sizeof(std::vector<double>) + b.capacity() * sizeof(double);
        // cat_categories strings
        for (auto& cc : cat_categories)
            for (auto& s : cc)
                total += s.capacity() + sizeof(std::string);
        // item_map strings
        for (auto& kv : item_map)
            total += sizeof(int) + kv.second.capacity() + sizeof(std::string) + 32;
        // hash map overhead (approximate)
        total += bn2id.size() * 48;
        return total;
    }

    // Python-facing accessor
    py::dict get_item_map_py() const {
        py::dict d;
        for (auto& kv : item_map)
            d[py::int_(kv.first)] = py::str(kv.second);
        return d;
    }

    double riu_thresh(int k) const {
        if (k <= 0 || RIU.empty()) return 0.0;
        std::vector<double> s = RIU;
        std::sort(s.rbegin(), s.rend());
        return s[std::min(static_cast<size_t>(k - 1), s.size() - 1)];
    }
};

// -- GIL-safe entry point --------------------------------------------------
// All Python-derived data (col_names, cat strings) must be extracted by the
// caller (bind_pattern.cpp) before releasing the GIL.  This function operates
// entirely on C++ types and is safe to call with the GIL released.
//
//   col_names_in   -- pre-extracted column names; empty -> auto-generate
//   cat_raw_strs   -- [p][n] categorical strings (only cols where is_cat==1)
//   cat_raw_valid  -- [p][n] validity mask matching cat_raw_strs
TransactionDataCpp prepare_transactions_cpp(
    const py::array_t<double,  py::array::c_style | py::array::forcecast>& X_num_arr,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& y_arr,
    int B,
    std::vector<std::string>              col_names_in,
    const py::array_t<uint8_t, py::array::forcecast>& is_cat_arr,
    const py::array_t<uint8_t, py::array::forcecast>& is_int_arr,
    std::vector<bool>                     is_precoded_in,
    std::vector<std::vector<std::string>> cat_raw_strs,
    std::vector<std::vector<bool>>        cat_raw_valid);

}  // namespace hugiml

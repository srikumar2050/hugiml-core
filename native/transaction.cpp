/**
 * transaction.cpp — Transaction preparation implementation
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
 *
 * Design: row-stripe chunked transaction construction
 * ─────────────────────────────────────────────────────────────────────────
 * The function is divided into two phases:
 *
 *   Phase 1 — Column statistics (Pass 1+2, unchanged):
 *     Process each column to compute edges, Pearson r, NMI, and per-bin
 *     item utilities.  Requires all n rows per column.  Produces the item
 *     registry (tu, bn2id, item_map) and per-column metadata stored in
 *     ColDesc.  No bin_codes are materialised.
 *
 *   Phase 2 — Striped transaction construction (Pass 3):
 *     Iterate rows in stripes of STRIPE_ROWS.  For each stripe, re-read
 *     the raw values from X_num_arr and re-apply the stored column edges
 *     to derive the bin code on the fly.  Emit transactions directly.
 *     Peak transient memory for this phase is O(STRIPE_ROWS × n_active)
 *     rather than O(n × n_active).
 *
 * The old ColStream.bin_codes[n] array is gone entirely.  ColDesc stores
 * only the column index, type flags, and the code_to_bname lookup table
 * (length nb_col[j], typically 2–20 entries).
 */

#include "transaction.hpp"
#include "math.hpp"
#include "discretization.hpp"
#include "resource_guard.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

namespace hugiml {

// ── Stripe size ───────────────────────────────────────────────────────────────
// Each stripe allocates one int32 per row per active column (transient).
// 100 000 rows × 200 active cols × 4 B = 80 MB peak per stripe.
// Adjust downward for very memory-constrained environments.
static constexpr int STRIPE_ROWS = 100'000;

TransactionDataCpp prepare_transactions_cpp(
    const py::array_t<double,  py::array::c_style | py::array::forcecast>& X_num_arr,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& y_arr,
    int B,
    std::vector<std::string>              col_names_in,
    const py::array_t<uint8_t, py::array::forcecast>& is_cat_arr,
    const py::array_t<uint8_t, py::array::forcecast>& is_int_arr,
    std::vector<bool>                     is_precoded_in,
    std::vector<std::vector<std::string>> cat_raw_strs,
    std::vector<std::vector<bool>>        cat_raw_valid)
{
    auto Xb   = X_num_arr.unchecked<2>();
    auto yb   = y_arr.unchecked<1>();
    auto icb  = is_cat_arr.unchecked<1>();
    auto iib  = is_int_arr.unchecked<1>();

    int n = static_cast<int>(Xb.shape(0));
    int p = static_cast<int>(Xb.shape(1));

    // ── Input validation ─────────────────────────────────────────────────────
    if (n <= 0)
        throw std::invalid_argument("X must have at least one row");
    if (p <= 0)
        throw std::invalid_argument("X must have at least one column");
    validate_same_size(
        static_cast<size_t>(n),
        static_cast<size_t>(yb.shape(0)),
        "X.shape[0]", "y.shape[0]");
    if (B != -1 && (B < 2 || B > 100))
        throw std::invalid_argument(
            "B must be -1 (auto) or in range [2, 100], got " + std::to_string(B));

    // ── Native memory guard ─────────────────────────────────────────────────
    // Fail early with a Python MemoryError (via the binding layer) instead of
    // letting the OS kill the process during severe stress.  This is only a
    // conservative guard for native temporary/output buffers; it does not count
    // the already-owned NumPy input matrix.
    {
        const uint64_t dense_cells = static_cast<uint64_t>(n) * static_cast<uint64_t>(p);
        const uint64_t rough_native_bytes =
            dense_cells * sizeof(int32_t) +                         // stripe / code-like buffers
            static_cast<uint64_t>(n) * (sizeof(Trans) + 8ULL) +     // row vectors + allocator slack
            dense_cells * sizeof(TItem);                            // compact item IDs
        ensure_native_memory_available(rough_native_bytes,
                                       "prepare_transactions_cpp(n=" + std::to_string(n) +
                                       ", p=" + std::to_string(p) + ")");
    }

    // ── Bin-key stride ───────────────────────────────────────────────────────
    int bkey_stride = std::max(p, 10000);
    auto bk = [bkey_stride](int bi, int j) { return bi * bkey_stride + j; };

    // ── y as std::vector<int> ────────────────────────────────────────────────
    std::vector<int> y_vec(n);
    int max_label = 0;
    std::set<int> cls_set;
    for (int i = 0; i < n; i++) {
        y_vec[i] = static_cast<int>(yb(i));
        if (y_vec[i] < 0)
            throw std::invalid_argument(
                "y contains negative label at index " + std::to_string(i));
        if (y_vec[i] > max_label) max_label = y_vec[i];
        cls_set.insert(y_vec[i]);
    }
    int n_cls      = static_cast<int>(cls_set.size());
    int key_stride = max_label + 1;

    if (n_cls < 2)
        throw std::invalid_argument(
            "y must contain at least 2 distinct classes, got " + std::to_string(n_cls));

    // ── Column names ─────────────────────────────────────────────────────────
    std::vector<std::string> names(p);
    if (col_names_in.size() == static_cast<size_t>(p)) {
        names = std::move(col_names_in);
    } else {
        for (int j = 0; j < p; j++)
            names[j] = "col" + std::to_string(j);
    }

    // ── is_cat / is_int / is_precoded masks ─────────────────────────────────
    std::vector<bool> is_cat(p), is_int(p), is_precoded(p);
    for (int j = 0; j < p; j++) {
        is_cat[j]      = static_cast<bool>(icb(j));
        is_int[j]      = static_cast<bool>(iib(j));
        is_precoded[j] = (j < static_cast<int>(is_precoded_in.size()))
                         ? is_precoded_in[j] : false;
        if (is_precoded[j]) { is_cat[j] = false; is_int[j] = false; }
    }

    if (!cat_raw_strs.empty()) {
        if (static_cast<int>(cat_raw_strs.size()) != p)
            throw std::invalid_argument(
                "cat_raw_strs size mismatch: got " +
                std::to_string(cat_raw_strs.size()) + ", expected " + std::to_string(p));
    }

    // ── MinMax params for float columns ──────────────────────────────────────
    std::vector<double> col_min(p, 0.0), col_range(p, 1.0);
    for (int j = 0; j < p; j++) {
        if (is_cat[j] || is_int[j] || is_precoded[j]) continue;
        double cmin = std::numeric_limits<double>::max();
        double cmax = std::numeric_limits<double>::lowest();
        for (int r = 0; r < n; r++) {
            double v = Xb(r, j);
            if (std::isfinite(v)) {
                if (v < cmin) cmin = v;
                if (v > cmax) cmax = v;
            }
        }
        col_min[j]   = (cmin == std::numeric_limits<double>::max()) ? 0.0 : cmin;
        col_range[j] = (cmax > cmin) ? (cmax - cmin) : 1.0;
    }

    // ── Column metadata containers ────────────────────────────────────────────
    std::vector<int>                                     nb_col;
    std::vector<std::vector<double>>                     ber, all_edges;
    std::vector<double>                                  cv;
    std::vector<std::vector<std::string>>                cat_categories(p);
    std::vector<std::vector<std::pair<double,double>>>   bro(p);

    nb_col.reserve(p);
    ber.reserve(p);
    all_edges.reserve(p);
    cv.reserve(p);

    // ── Item registry ─────────────────────────────────────────────────────────
    std::unordered_map<int, std::string> item_map;
    std::unordered_map<int, int>         bn2id;
    std::unordered_map<int64_t, double>  tu;
    std::vector<double>                  tu_y(key_stride, 0.0);
    std::vector<int>                     item_col;
    int ic = 0;

    // ── Active-column descriptor ──────────────────────────────────────────────
    // Replaces the old ColStream which stored bin_codes[n] (O(n) per column).
    // ColDesc stores only the column index and the code_to_bname lookup table,
    // which is O(nb_col[j]) — negligible.
    //
    // Per-column label2int maps for categorical columns are stored separately
    // so the stripe loop can encode rows without re-scanning cat_categories.
    struct ColDesc {
        int              col_idx;       // original column index j
        std::vector<int> code_to_bname; // code 0-based → bname, -1 = no item
        // For categorical columns: label-string → integer code.
        // Built once in Phase 1 and reused across all stripes.
        std::unordered_map<std::string, int> label2int;
    };
    std::vector<ColDesc> active_cols;
    active_cols.reserve(p);

    // ── Phase 1 : Column statistics + item registry ───────────────────────────
    // For each column:
    //   a) Compute global stats (edges, Pearson r, NMI).  Requires all n rows.
    //   b) Register items in tu / bn2id / item_map.
    //   c) If any item was registered, add a ColDesc to active_cols.
    // No bin_codes are stored here.

    for (int j = 0; j < p; j++) {

        if (is_cat[j]) {
            // ── Categorical ──────────────────────────────────────────────────
            const bool have_cat   = (!cat_raw_strs.empty() && !cat_raw_strs[j].empty());
            const bool have_valid = (!cat_raw_valid.empty() && !cat_raw_valid[j].empty());

            std::set<std::string> uniq_set;
            if (have_cat)
                for (int r = 0; r < n; r++)
                    if (have_valid && cat_raw_valid[j][r])
                        uniq_set.insert(cat_raw_strs[j][r]);
            std::vector<std::string> uniq(uniq_set.begin(), uniq_set.end());
            cat_categories[j] = uniq;

            std::unordered_map<std::string, int> label2int;
            label2int.reserve(uniq.size());
            for (int i = 0; i < static_cast<int>(uniq.size()); i++)
                label2int[uniq[i]] = i;

            // Per-row codes — needed for NMI and Pearson sign, but only
            // for this column's stats pass.  Freed at end of this block.
            std::vector<int32_t> col_codes(n, -1);
            for (int r = 0; r < n; r++) {
                if (have_cat && have_valid && cat_raw_valid[j][r]) {
                    auto it = label2int.find(cat_raw_strs[j][r]);
                    if (it != label2int.end())
                        col_codes[r] = static_cast<int32_t>(it->second);
                }
            }

            // Single O(n) pass for Pearson sign across all categories
            int C = static_cast<int>(uniq.size());
            std::vector<int>    n_c(C, 0);
            std::vector<double> sum_y_c(C, 0.0);
            double sum_y = 0.0;
            for (int r = 0; r < n; r++) {
                int32_t code = col_codes[r];
                double  yv   = static_cast<double>(y_vec[r]);
                sum_y += yv;
                if (code >= 0 && code < C) {
                    n_c[code]    += 1;
                    sum_y_c[code] += yv;
                }
            }

            int nb_act = C;
            std::vector<int> code_to_bname(nb_act, -1);
            bool any_item = false;

            for (int idx = 0; idx < nb_act; idx++) {
                int bi = idx + 1;

                std::vector<int> x_bin(n, 0);
                for (int r = 0; r < n; r++)
                    if (col_codes[r] == static_cast<int32_t>(idx))
                        x_bin[r] = 1;
                double nmi = nmi_binary_cpp(x_bin, y_vec, n_cls);

                // sign(cov(x_bin, y)) = sign(sum_y_c*n - n_c*sum_y)
                double cov_num = sum_y_c[idx] * static_cast<double>(n)
                                 - static_cast<double>(n_c[idx]) * sum_y;
                double pb  = (cov_num > 0.0) ? 1.0 : (cov_num < 0.0) ? -1.0 : 0.0;
                double iu_t = (pb > 0.0) ? 1.0 : 0.05;
                double eiu  = nmi * iu_t;

                int bname = bk(bi, j);
                if (eiu > 0.0) {
                    ic++;
                    item_col.push_back(j);
                    item_map[ic] = names[j] + "=" + uniq[idx];
                    bn2id[bname] = ic;
                    code_to_bname[idx] = bname;
                    any_item = true;
                }
                for (int yi = 0; yi < n_cls; yi++) {
                    int64_t k = static_cast<int64_t>(bname) * key_stride + yi;
                    tu[k] = eiu;
                    if (eiu > tu_y[yi]) tu_y[yi] = eiu;
                }
            }

            nb_col.push_back(nb_act);
            all_edges.push_back({0.0, 1.0});
            ber.push_back({1.0});
            bro[j] = {{0.0, 1.0}};
            cv.push_back(0.0);

            if (any_item) {
                ColDesc cd;
                cd.col_idx      = j;
                cd.code_to_bname = std::move(code_to_bname);
                cd.label2int    = std::move(label2int);
                active_cols.push_back(std::move(cd));
            }
            // col_codes is freed here (goes out of scope).

        } else if (is_int[j]) {
            // ── Integer ──────────────────────────────────────────────────────
            std::vector<double> col_raw(n);
            for (int r = 0; r < n; r++) col_raw[r] = Xb(r, j);

            std::set<double> dset(col_raw.begin(), col_raw.end());
            int distinct = static_cast<int>(dset.size());
            int nb       = choose_nb_cpp(col_raw, y_vec, n_cls, B, distinct);

            auto [binned, edges] = kbins_cpp(col_raw, nb);
            int nb_act = static_cast<int>(edges.size()) - 1;
            nb_col.push_back(nb_act);
            all_edges.push_back(edges);

            double mx = *std::max_element(edges.begin(), edges.end());
            if (mx <= 0.0) mx = 1.0;
            std::vector<double> ber_j;
            ber_j.reserve(edges.size() - 1);
            for (int bi = 1; bi < static_cast<int>(edges.size()); bi++)
                ber_j.push_back(edges[bi] / mx);
            ber.push_back(ber_j);

            std::vector<std::pair<double,double>> bro_j;
            bro_j.reserve(nb_act);
            for (int bi = 0; bi < nb_act; bi++)
                bro_j.push_back({edges[bi], edges[bi + 1]});
            bro[j] = bro_j;

            std::vector<double> df(n), yf(n);
            for (int r = 0; r < n; r++) { df[r] = binned[r]; yf[r] = y_vec[r]; }
            cv.push_back(pearson_cpp(df, yf));

            double eu = std::abs(cv.back());
            std::vector<int> code_to_bname(nb_act, -1);
            bool any_item = false;
            for (int bi = 1; bi <= nb_act; bi++) {
                double iu_t = (cv.back() >= 0.0) ? ber_j[bi - 1] : ber_j[nb_act - bi];
                double eiu  = eu * iu_t;
                int bname = bk(bi, j);
                if (eiu > 0.0) {
                    ic++;
                    item_col.push_back(j);
                    auto& [lo, hi] = bro_j[bi - 1];
                    std::ostringstream oss;
                    oss << names[j] << "=[" << static_cast<int>(lo)
                        << "," << static_cast<int>(hi) << "]";
                    item_map[ic] = oss.str();
                    bn2id[bname] = ic;
                    code_to_bname[bi - 1] = bname;
                    any_item = true;
                }
                for (int yi = 0; yi < n_cls; yi++) {
                    int64_t k = static_cast<int64_t>(bname) * key_stride + yi;
                    tu[k] = eiu;
                    if (eiu > tu_y[yi]) tu_y[yi] = eiu;
                }
            }

            if (any_item) {
                ColDesc cd;
                cd.col_idx       = j;
                cd.code_to_bname = std::move(code_to_bname);
                active_cols.push_back(std::move(cd));
            }
            // binned, col_raw freed here.

        } else if (is_precoded[j]) {
            // ── Pre-coded (adaptive integer codes 0..B_j-1) ──────────────────
            std::vector<double> col_codes_d(n);
            for (int r = 0; r < n; r++) col_codes_d[r] = Xb(r, j);

            int max_code = 0;
            for (int r = 0; r < n; r++) {
                double v = col_codes_d[r];
                if (std::isfinite(v)) {
                    int c = static_cast<int>(v);
                    if (c > max_code) max_code = c;
                }
            }
            int nb_act = max_code + 1;

            std::vector<double> edges(static_cast<size_t>(nb_act + 1));
            for (int k = 0; k <= nb_act; k++) edges[k] = static_cast<double>(k);
            nb_col.push_back(nb_act);
            all_edges.push_back(edges);

            ber.push_back(std::vector<double>(nb_act, 1.0 / std::max(nb_act, 1)));

            std::vector<std::pair<double,double>> bro_j;
            bro_j.reserve(nb_act);
            for (int k = 0; k < nb_act; k++)
                bro_j.push_back({static_cast<double>(k), static_cast<double>(k + 1)});
            bro[j] = bro_j;

            std::vector<double> df2(n), yf2(n);
            for (int r = 0; r < n; r++) {
                df2[r] = std::isfinite(col_codes_d[r]) ? col_codes_d[r] : 0.0;
                yf2[r] = y_vec[r];
            }
            cv.push_back(pearson_cpp(df2, yf2));

            double eu = std::abs(cv.back());
            std::vector<int> code_to_bname(nb_act, -1);
            bool any_item = false;
            std::vector<double> ber_j(nb_act, 1.0 / std::max(nb_act, 1));
            for (int bi = 1; bi <= nb_act; bi++) {
                double iu_t = (cv.back() >= 0.0) ? ber_j[bi - 1] : ber_j[nb_act - bi];
                double eiu  = eu * iu_t;
                int bname = bk(bi, j);
                if (eiu > 0.0) {
                    ic++;
                    item_col.push_back(j);
                    auto& [lo, hi] = bro_j[bi - 1];
                    std::ostringstream oss;
                    oss << names[j] << "=["
                        << std::fixed << std::setprecision(3) << lo
                        << ","
                        << std::fixed << std::setprecision(3) << hi << "]";
                    item_map[ic] = oss.str();
                    bn2id[bname] = ic;
                    code_to_bname[bi - 1] = bname;
                    any_item = true;
                }
                for (int yi = 0; yi < n_cls; yi++) {
                    int64_t k = static_cast<int64_t>(bname) * key_stride + yi;
                    tu[k] = eiu;
                    if (eiu > tu_y[yi]) tu_y[yi] = eiu;
                }
            }

            if (any_item) {
                ColDesc cd;
                cd.col_idx       = j;
                cd.code_to_bname = std::move(code_to_bname);
                active_cols.push_back(std::move(cd));
            }
            // col_codes_d freed here.

        } else {
            // ── Float ────────────────────────────────────────────────────────
            std::vector<double> col_sc(n);
            for (int r = 0; r < n; r++) {
                double v = Xb(r, j);
                col_sc[r] = std::isfinite(v)
                    ? (v - col_min[j]) / col_range[j]
                    : std::numeric_limits<double>::quiet_NaN();
            }

            std::set<double> dset;
            for (double v : col_sc) if (std::isfinite(v)) dset.insert(v);
            int distinct = static_cast<int>(dset.size());
            int nb       = choose_nb_cpp(col_sc, y_vec, n_cls, B, distinct);

            auto [binned, edges] = kbins_cpp(col_sc, nb);
            int nb_act = static_cast<int>(edges.size()) - 1;
            nb_col.push_back(nb_act);
            all_edges.push_back(edges);

            double mx = *std::max_element(edges.begin(), edges.end());
            if (mx <= 0.0) mx = 1.0;
            std::vector<double> ber_j;
            ber_j.reserve(edges.size() - 1);
            for (int bi = 1; bi < static_cast<int>(edges.size()); bi++)
                ber_j.push_back(edges[bi] / mx);
            ber.push_back(ber_j);

            std::vector<std::pair<double,double>> bro_j;
            bro_j.reserve(nb_act);
            for (int bi = 0; bi < nb_act; bi++) {
                double lo_o = edges[bi]     * col_range[j] + col_min[j];
                double hi_o = edges[bi + 1] * col_range[j] + col_min[j];
                bro_j.push_back({lo_o, hi_o});
            }
            bro[j] = bro_j;

            std::vector<double> df(n), yf(n);
            for (int r = 0; r < n; r++) { df[r] = binned[r]; yf[r] = y_vec[r]; }
            cv.push_back(pearson_cpp(df, yf));

            double eu = std::abs(cv.back());
            std::vector<int> code_to_bname(nb_act, -1);
            bool any_item = false;
            for (int bi = 1; bi <= nb_act; bi++) {
                double iu_t = (cv.back() >= 0.0) ? ber_j[bi - 1] : ber_j[nb_act - bi];
                double eiu  = eu * iu_t;
                int bname = bk(bi, j);
                if (eiu > 0.0) {
                    ic++;
                    item_col.push_back(j);
                    auto& [lo, hi] = bro_j[bi - 1];
                    std::ostringstream oss;
                    oss << names[j] << "=["
                        << std::fixed << std::setprecision(3) << lo
                        << ","
                        << std::fixed << std::setprecision(3) << hi << "]";
                    item_map[ic] = oss.str();
                    bn2id[bname] = ic;
                    code_to_bname[bi - 1] = bname;
                    any_item = true;
                }
                for (int yi = 0; yi < n_cls; yi++) {
                    int64_t k = static_cast<int64_t>(bname) * key_stride + yi;
                    tu[k] = eiu;
                    if (eiu > tu_y[yi]) tu_y[yi] = eiu;
                }
            }

            if (any_item) {
                ColDesc cd;
                cd.col_idx       = j;
                cd.code_to_bname = std::move(code_to_bname);
                active_cols.push_back(std::move(cd));
            }
            // binned, col_sc freed here.
        }
    }

    // ── Normalise utility per class ───────────────────────────────────────────
    for (auto& kv : tu) {
        int yi = static_cast<int>(kv.first % key_stride);
        kv.second = (tu_y[yi] > 0.0) ? kv.second / tu_y[yi] : 0.0;
    }

    // Materialize one normalized utility per item/bin.  In v1.1.4 the class
    // dimension in tu[bname, yi] is redundant because class weights are not
    // exposed/applied; the same normalized value is stored for every yi.
    std::vector<double> item_iu(static_cast<size_t>(ic), 0.0);
    for (const auto& kv_bn : bn2id) {
        const int bname = kv_bn.first;
        const int iid   = kv_bn.second;
        if (iid <= 0 || iid > ic) continue;
        for (int yi_probe = 0; yi_probe < key_stride; ++yi_probe) {
            int64_t k = static_cast<int64_t>(bname) * key_stride + yi_probe;
            auto it = tu.find(k);
            if (it != tu.end()) {
                item_iu[static_cast<size_t>(iid - 1)] =
                    std::round(it->second * 1e6) / 1e6;
                break;
            }
        }
    }

    // ── Phase 2 : Striped transaction construction ────────────────────────────
    //
    // Rows are processed in stripes of STRIPE_ROWS.  For each stripe:
    //   1. Re-read raw values from X_num_arr and re-apply column edges to
    //      produce a temporary bin_codes matrix for the stripe rows only.
    //      This is a flat int32 array: stripe_codes[s * n_active + a]
    //      where s = row offset within stripe, a = index in active_cols.
    //   2. Walk the stripe's bin_codes row-by-row to build transactions and
    //      accumulate item_twu / RIU.
    //
    // The stripe bin_codes buffer is reused across stripes (resize-in-place).
    // Peak transient memory: STRIPE_ROWS × n_active × 4 bytes.
    //
    // Bin-code derivation per column type, identical to what the old
    // bin_codes[r] values encoded:
    //
    //   Categorical:  code = label2int[cat_raw_strs[j][r]] if valid, else -1.
    //                 bi   = code + 1  (1-indexed bin).
    //
    //   Integer/Float: val = raw value (int) or scaled value (float).
    //                  Non-finite val  → code = -1  (skip).
    //                  Finite val      → upper_bound on inner edges → bi.
    //                  bi clamped to [1, nb_col[j]].
    //                  code = bi - 1   (0-indexed, matching old bin_codes).
    //
    //   Pre-coded:    val = Xb(r, j) as float64.
    //                 Non-finite val → code = -1  (skip).
    //                 Finite val     → code = clip(floor(val), 0, nb_col[j]-1).
    //
    // All -1 codes are skipped during transaction assembly, producing the
    // same "no item for missing / zero-IU bin" behaviour as before.

    const int n_active = static_cast<int>(active_cols.size());

    TransList           transactions;
    std::vector<double> item_twu(ic, 0.0);
    std::vector<double> RIU(ic, 0.0);
    ensure_native_memory_available(
        static_cast<uint64_t>(n) * (sizeof(Trans) + 8ULL) +
        static_cast<uint64_t>(n) * static_cast<uint64_t>(std::max(n_active, 0)) * sizeof(TItem),
        "compact transaction output");
    transactions.reserve(n);

    // Flat stripe buffer: stripe_codes[s_row * n_active + a_col]
    // Allocated once, reused for every stripe.
    std::vector<int32_t> stripe_codes;

    for (int stripe_start = 0; stripe_start < n; stripe_start += STRIPE_ROWS) {
        const int stripe_end = std::min(stripe_start + STRIPE_ROWS, n);
        const int stripe_n   = stripe_end - stripe_start;

        // Resize the stripe buffer for the current stripe width (usually
        // constant except for the final stripe which may be shorter).
        const size_t stripe_cells = checked_mul_size_t(
            static_cast<uint64_t>(stripe_n), static_cast<uint64_t>(std::max(n_active, 0)),
            "prepare_transactions stripe_codes");
        ensure_native_memory_available(static_cast<uint64_t>(stripe_cells) * sizeof(int32_t),
                                       "prepare_transactions stripe buffer");
        stripe_codes.assign(stripe_cells, -1);

        // ── Fill stripe_codes from X_num_arr ──────────────────────────────
        for (int a = 0; a < n_active; a++) {
            const ColDesc& cd = active_cols[a];
            const int j = cd.col_idx;
            const int nb = static_cast<int>(cd.code_to_bname.size()); // = nb_col[j]

            if (is_cat[j]) {
                // Categorical: encode each row via label2int.
                const bool have_cat   = (!cat_raw_strs.empty()  && !cat_raw_strs[j].empty());
                const bool have_valid = (!cat_raw_valid.empty() && !cat_raw_valid[j].empty());
                for (int s = 0; s < stripe_n; s++) {
                    int r = stripe_start + s;
                    int32_t code = -1;
                    if (have_cat && have_valid && cat_raw_valid[j][r]) {
                        auto it = cd.label2int.find(cat_raw_strs[j][r]);
                        if (it != cd.label2int.end())
                            code = static_cast<int32_t>(it->second);
                    }
                    stripe_codes[static_cast<size_t>(s) * n_active + a] = code;
                }

            } else if (is_precoded[j]) {
                // Pre-coded: raw value IS the 0-indexed bin code.
                // nb_col[j] = max_code + 1, stored as code_to_bname.size().
                for (int s = 0; s < stripe_n; s++) {
                    int r = stripe_start + s;
                    double v = Xb(r, j);
                    int32_t code;
                    if (!std::isfinite(v)) {
                        code = -1;
                    } else {
                        int c = static_cast<int>(v);
                        code = static_cast<int32_t>(
                            std::max(0, std::min(c, nb - 1)));
                    }
                    stripe_codes[static_cast<size_t>(s) * n_active + a] = code;
                }

            } else {
                // Integer or Float: apply stored edges via upper_bound.
                // Edges are stored as all_edges[j] (scaled for float columns).
                // The inner edges are all_edges[j][1 .. size-2].
                const auto& edges = all_edges[j];
                const int n_inner = static_cast<int>(edges.size()) - 2;
                // Pointer to first inner edge for fast binary search.
                const double* inner_begin = edges.data() + 1;

                for (int s = 0; s < stripe_n; s++) {
                    int r = stripe_start + s;
                    double raw = Xb(r, j);
                    int32_t code;
                    if (!std::isfinite(raw)) {
                        code = -1;
                    } else {
                        // Scale raw value the same way Phase 1 did.
                        double val = is_int[j]
                            ? raw
                            : (raw - col_min[j]) / col_range[j];
                        // upper_bound on inner edges → 0-based bin index.
                        int pos = static_cast<int>(
                            std::upper_bound(inner_begin,
                                             inner_begin + n_inner,
                                             val)
                            - inner_begin);
                        // Clamp to [0, nb-1].
                        code = static_cast<int32_t>(
                            std::max(0, std::min(pos, nb - 1)));
                    }
                    stripe_codes[static_cast<size_t>(s) * n_active + a] = code;
                }
            }
        }

        // ── Build transactions for this stripe ────────────────────────────
        for (int s = 0; s < stripe_n; s++) {
            int r  = stripe_start + s;
            int yi = y_vec[r];
            double tutils = 0.0;
            Trans  trans;

            for (int a = 0; a < n_active; a++) {
                int32_t code = stripe_codes[static_cast<size_t>(s) * n_active + a];
                if (code < 0) continue;                         // missing / skip

                const ColDesc& cd = active_cols[a];
                int bname_idx = static_cast<int>(code);
                if (bname_idx >= static_cast<int>(cd.code_to_bname.size())) continue;
                int bname = cd.code_to_bname[bname_idx];
                if (bname < 0) continue;                        // zero-IU bin

                int64_t txk = static_cast<int64_t>(bname) * key_stride + yi;
                if (tu.find(txk) == tu.end()) continue;

                auto bn_it = bn2id.find(bname);
                if (bn_it == bn2id.end()) continue;
                int iid = bn_it->second;
                if (iid <= 0 || static_cast<size_t>(iid - 1) >= item_iu.size()) continue;
                double iu = item_iu[static_cast<size_t>(iid - 1)];
                trans.push_back(iid);
                tutils += iu;
            }

            if (tutils > 0.0) {
                for (int iid : trans) {
                    double iu = item_iu[static_cast<size_t>(iid - 1)];
                    item_twu[iid - 1] += tutils;
                    RIU[iid - 1]      += iu;
                }
                transactions.push_back(std::move(trans));
            } else {
                transactions.push_back({-1});
            }
        }
        // stripe_codes is reused next iteration (assign() at top of loop).
    }

    // ── Assemble result ──────────────────────────────────────────────────────
    TransactionDataCpp td;
    td.transactions    = std::move(transactions);
    td.item_twu        = std::move(item_twu);
    td.item_map        = std::move(item_map);
    td.RIU             = std::move(RIU);
    td.item_iu         = std::move(item_iu);
    td.item_col        = std::move(item_col);
    td.disc_n          = n;
    td.disc_p          = p;
    td.bn2id           = std::move(bn2id);
    td.nb_col          = std::move(nb_col);
    td.ber             = std::move(ber);
    td.cv              = std::move(cv);
    td.all_edges       = std::move(all_edges);
    td.col_min         = std::move(col_min);
    td.col_range       = std::move(col_range);
    td.is_cat_v        = std::move(is_cat);
    td.is_int_v        = std::move(is_int);
    td.is_precoded_v   = std::move(is_precoded);
    td.cat_categories  = std::move(cat_categories);
    td.bkey_stride     = bkey_stride;
    return td;
}

}  // namespace hugiml

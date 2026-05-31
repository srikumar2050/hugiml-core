/**
 * prepare_mine_l1.cpp — Fused L=1 hot path implementation.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * See prepare_mine_l1.hpp for design overview.
 */

#include "prepare_mine_l1.hpp"
#include "discretization.hpp"
#include "math.hpp"
#include "mining.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <set>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace py = pybind11;

namespace hugiml {

// ── Internal helpers ─────────────────────────────────────────────────────────

static double l1_entropy_ln(const std::vector<int>& counts, int total) {
    // Shannon entropy with natural log — matches ig_col_cpp / math.cpp.
    if (total <= 0) return 0.0;
    double h = 0.0;
    const double inv = 1.0 / static_cast<double>(total);
    for (int c : counts) {
        if (c > 0) {
            double p = static_cast<double>(c) * inv;
            h -= p * std::log(p);
        }
    }
    return std::max(h, 0.0);
}

/// Compute IG for a singleton item.  Matches compute_ig(parent=nullptr).
static double l1_ig(const std::vector<int>& cnt_global,
                    const int*              cnt_in_ptr,
                    int n_in, int n_train, int n_cls) {
    int n_out = n_train - n_in;
    if (n_out == 0) return std::numeric_limits<double>::quiet_NaN();

    std::vector<int> cnt_in(cnt_in_ptr, cnt_in_ptr + n_cls);
    std::vector<int> cnt_out(n_cls);
    for (int k = 0; k < n_cls; k++) cnt_out[k] = cnt_global[k] - cnt_in[k];

    double base = l1_entropy_ln(cnt_global, n_train);
    double ce   = (static_cast<double>(n_in)  / n_train * l1_entropy_ln(cnt_in,  n_in)
                 + static_cast<double>(n_out) / n_train * l1_entropy_ln(cnt_out, n_out));
    return base - ce;
}

/// Min-heap save — identical to THUIsl::save / l1_save in mining_l1.cpp.
static void l1_save(std::vector<PatternEntry>& heap, double& minU,
                    int K, int iid, double utility, double ig) {
    struct Cmp { bool operator()(const PatternEntry& a, const PatternEntry& b) const {
        return a.utility > b.utility; } };
    PatternEntry pe{utility, {iid}, ig};
    if (static_cast<int>(heap.size()) < K) {
        heap.push_back(pe);
        std::push_heap(heap.begin(), heap.end(), Cmp{});
        if (static_cast<int>(heap.size()) == K) minU = heap.front().utility;
    } else if (utility > minU) {
        std::pop_heap(heap.begin(), heap.end(), Cmp{});
        heap.back() = pe;
        std::push_heap(heap.begin(), heap.end(), Cmp{});
        minU = heap.front().utility;
    }
}

// ── Column descriptor (Phase 1 output) ───────────────────────────────────────

struct L1ColDesc {
    int col_idx = 0;
    int nb      = 0;          // number of bins

    // Per-bin arrays (indexed 0..nb-1)
    std::vector<double> bin_iu;   // normalised instantaneous utility
    std::vector<int>    bin_iid;  // item ID (1-based), 0 = no item

    // For categorical columns: label → 0-based code
    std::unordered_map<std::string, int> label2code;

    // Precoded column: direct int cast (no upper_bound)
    bool is_precoded_col = false;
    bool is_cat_col      = false;

    // For numeric/precoded: edges for bin assignment at inference time
    // (stored into td.all_edges by prepare_and_mine_l1_cpp)
    std::vector<double> edges;
    double col_min_v   = 0.0;
    double col_range_v = 1.0;
    bool   is_int_col  = false;
};

// ── select_adaptive_bins_cpp ──────────────────────────────────────────────────

AdaptiveBinResult select_adaptive_bins_cpp(
    const py::array_t<double,  py::array::c_style | py::array::forcecast>& X_num_arr,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& y_arr,
    int n_cls,
    const std::vector<std::string>&  col_names,
    const py::array_t<uint8_t, py::array::forcecast>& is_cat_arr,
    const std::vector<int>&          candidates,
    double                           ratio)
{
    auto Xb  = X_num_arr.unchecked<2>();
    auto yb  = y_arr.unchecked<1>();
    auto icb = is_cat_arr.unchecked<1>();

    int n = static_cast<int>(Xb.shape(0));
    int p = static_cast<int>(Xb.shape(1));

    std::vector<int> y_vec(n);
    for (int i = 0; i < n; i++) y_vec[i] = static_cast<int>(yb(i));

    AdaptiveBinResult result;
    result.n_rows = n;

    for (int j = 0; j < p; j++) {
        if (j < static_cast<int>(icb.size()) && static_cast<bool>(icb(j))) {
            // Categorical column — skip, will not be pre-coded
            continue;
        }

        // Extract raw column values
        std::vector<double> col_raw(n);
        for (int r = 0; r < n; r++) col_raw[r] = Xb(r, j);

        // Elbow-stop B selection
        std::vector<double> out_edges, out_ig;
        int chosen = elbow_stop_nb_cpp(col_raw, y_vec, n_cls,
                                        candidates, ratio,
                                        out_edges, out_ig);

        ColAdaptResult cr;
        cr.chosen_b  = chosen;
        cr.edges     = out_edges;
        cr.ig_scores = out_ig;
        result.cols.push_back(std::move(cr));
        result.num_col_indices.push_back(j);
    }

    // Build X_codes_flat: (n × n_num_cols) row-major float64
    int n_num = static_cast<int>(result.cols.size());
    result.n_num_cols = n_num;
    result.X_codes_flat.resize(static_cast<size_t>(n) * n_num, 0.0);

    for (int ci = 0; ci < n_num; ci++) {
        int j = result.num_col_indices[ci];
        const auto& edges = result.cols[ci].edges;
        int nb = result.cols[ci].chosen_b;
        const int n_inner = static_cast<int>(edges.size()) - 2;
        const double* inner_begin = edges.data() + 1;

        for (int r = 0; r < n; r++) {
            double raw = Xb(r, j);
            double code;
            if (!std::isfinite(raw)) {
                code = std::numeric_limits<double>::quiet_NaN();
            } else {
                int pos = static_cast<int>(
                    std::upper_bound(inner_begin, inner_begin + n_inner, raw)
                    - inner_begin);
                code = static_cast<double>(std::max(0, std::min(pos, nb - 1)));
            }
            result.X_codes_flat[static_cast<size_t>(r) * n_num + ci] = code;
        }
    }
    return result;
}

// ── prepare_and_mine_l1_cpp ───────────────────────────────────────────────────

L1FitResult prepare_and_mine_l1_cpp(
    const py::array_t<double,  py::array::c_style | py::array::forcecast>& X_num_arr,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& y_arr,
    int B,
    std::vector<std::string>               col_names,
    const py::array_t<uint8_t, py::array::forcecast>& is_cat_arr,
    const py::array_t<uint8_t, py::array::forcecast>& is_int_arr,
    std::vector<bool>                      is_precoded,
    std::vector<std::vector<std::string>>  cat_raw_strs,
    std::vector<std::vector<bool>>         cat_raw_valid,
    int K, double G, double timeout_s)
{
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    static constexpr int CHECK = 256;

    bool      has_deadline = (timeout_s > 0.0);
    TimePoint deadline_tp  = {};
    if (has_deadline)
        deadline_tp = Clock::now() +
            std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(timeout_s));

    auto Xb  = X_num_arr.unchecked<2>();
    auto yb  = y_arr.unchecked<1>();
    auto icb = is_cat_arr.unchecked<1>();
    auto iib = is_int_arr.unchecked<1>();

    int n = static_cast<int>(Xb.shape(0));
    int p = static_cast<int>(Xb.shape(1));

    // ── y vector + class count ────────────────────────────────────────────────
    std::vector<int> y_vec(n);
    int max_label = 0;
    for (int i = 0; i < n; i++) {
        y_vec[i] = static_cast<int>(yb(i));
        if (y_vec[i] > max_label) max_label = y_vec[i];
    }
    int n_cls      = max_label + 1;
    int key_stride = n_cls;

    // ── Column name defaults ──────────────────────────────────────────────────
    if (static_cast<int>(col_names.size()) != p) {
        col_names.resize(p);
        for (int j = 0; j < p; j++)
            if (col_names[j].empty())
                col_names[j] = "col" + std::to_string(j);
    }

    // ── Column type masks ─────────────────────────────────────────────────────
    std::vector<bool> is_cat_v(p), is_int_v(p), is_prec_v(p);
    for (int j = 0; j < p; j++) {
        is_cat_v[j]  = j < static_cast<int>(icb.size()) && static_cast<bool>(icb(j));
        is_int_v[j]  = j < static_cast<int>(iib.size()) && static_cast<bool>(iib(j));
        is_prec_v[j] = j < static_cast<int>(is_precoded.size()) && is_precoded[j];
        if (is_prec_v[j]) { is_cat_v[j] = false; is_int_v[j] = false; }
    }

    // ── Global class counts (parent population for IG) ────────────────────────
    std::vector<int> cnt_global(n_cls, 0);
    for (int lbl : y_vec)
        if (lbl >= 0 && lbl < n_cls) cnt_global[lbl]++;

    // ── bkey stride ──────────────────────────────────────────────────────────
    int bkey_stride = std::max(p, 10000);
    auto bk = [bkey_stride](int bi, int j) { return bi * bkey_stride + j; };

    // ── Phase 1: per-column statistics → L1ColDesc list ──────────────────────
    // Produces: edges, bin_iu, bin_iid, item_map, bn2id, all_edges, nb_col, etc.
    // These are identical to what prepare_transactions_cpp Phase 1 produces.

    TransactionDataCpp td;
    td.bkey_stride = bkey_stride;
    td.disc_n = n;
    td.disc_p = p;
    td.is_cat_v       = is_cat_v;
    td.is_int_v       = is_int_v;
    td.is_precoded_v  = is_prec_v;

    td.nb_col.reserve(p);
    td.all_edges.reserve(p);
    td.ber.reserve(p);
    td.cv.reserve(p);
    td.col_min.resize(p, 0.0);
    td.col_range.resize(p, 1.0);
    td.cat_categories.resize(p);

    std::vector<L1ColDesc> active_cols;
    active_cols.reserve(p);

    // tu_y: max raw eiu per class direction — for normalisation
    std::vector<double> tu_y(key_stride, 0.0);
    // raw eiu map (before normalisation): bname -> eiu
    std::unordered_map<int, double> raw_eiu;

    int ic = 0;  // item counter

    for (int j = 0; j < p; j++) {

        if (is_cat_v[j]) {
            // ── Categorical ───────────────────────────────────────────────────
            const bool have_cat   = (!cat_raw_strs.empty() && !cat_raw_strs[j].empty());
            const bool have_valid = (!cat_raw_valid.empty() && !cat_raw_valid[j].empty());

            // Collect unique categories
            std::vector<std::string> uniq;
            {
                std::set<std::string> us;
                if (have_cat)
                    for (int r = 0; r < n; r++)
                        if (have_valid && cat_raw_valid[j][r])
                            us.insert(cat_raw_strs[j][r]);
                uniq.assign(us.begin(), us.end());
            }
            td.cat_categories[j] = uniq;

            int C = static_cast<int>(uniq.size());
            std::unordered_map<std::string, int> label2code;
            label2code.reserve(C);
            for (int i = 0; i < C; i++) label2code[uniq[i]] = i;

            // Encode column
            std::vector<int32_t> col_codes(n, -1);
            for (int r = 0; r < n; r++)
                if (have_cat && have_valid && cat_raw_valid[j][r]) {
                    auto it = label2code.find(cat_raw_strs[j][r]);
                    if (it != label2code.end())
                        col_codes[r] = static_cast<int32_t>(it->second);
                }

            // Pearson sign for iu_t
            std::vector<int>    n_c(C, 0);
            std::vector<double> sum_y_c(C, 0.0);
            double sum_y = 0.0;
            for (int r = 0; r < n; r++) {
                double yv = static_cast<double>(y_vec[r]);
                sum_y += yv;
                int32_t code = col_codes[r];
                if (code >= 0 && code < C) { n_c[code]++; sum_y_c[code] += yv; }
            }

            L1ColDesc cd;
            cd.col_idx    = j;
            cd.nb         = C;
            cd.is_cat_col = true;
            cd.bin_iu.resize(C, 0.0);
            cd.bin_iid.resize(C, 0);
            cd.label2code = std::move(label2code);
            bool any_item = false;

            std::vector<int> code_to_bname(C, -1);
            for (int idx = 0; idx < C; idx++) {
                int bi = idx + 1;
                // NMI
                std::vector<int> x_bin(n, 0);
                for (int r = 0; r < n; r++)
                    if (col_codes[r] == static_cast<int32_t>(idx)) x_bin[r] = 1;
                double nmi = nmi_binary_cpp(x_bin, y_vec, n_cls);

                double cov_num = sum_y_c[idx]*n - static_cast<double>(n_c[idx])*sum_y;
                double pb  = (cov_num > 0.0) ? 1.0 : (cov_num < 0.0) ? -1.0 : 0.0;
                double eiu = nmi * ((pb > 0.0) ? 1.0 : 0.05);

                int bname = bk(bi, j);
                if (eiu > 0.0) {
                    ic++;
                    td.item_map[ic] = col_names[j] + "=" + uniq[idx];
                    td.bn2id[bname] = ic;
                    raw_eiu[bname]  = eiu;
                    cd.bin_iid[idx] = ic;
                    any_item = true;
                    for (int yi = 0; yi < n_cls; yi++) {
                        if (eiu > tu_y[yi]) tu_y[yi] = eiu;
                    }
                }
            }

            // Normalise immediately (cat columns have fixed eiu independent of row)
            for (int idx = 0; idx < C; idx++) {
                int bi = idx + 1;
                int bname = bk(bi, j);
                double eiu = raw_eiu.count(bname) ? raw_eiu[bname] : 0.0;
                // Normalise by max eiu seen so far for each class direction
                // For cat cols, iu_t is already baked in (1.0 or 0.05)
                // tu_y is updated above; normalise after all columns processed
                // (store raw eiu in bin_iu for now; normalise after all cols)
                cd.bin_iu[idx] = eiu;
            }

            td.nb_col.push_back(C);
            td.all_edges.push_back({0.0, 1.0});
            td.ber.push_back({1.0});
            td.cv.push_back(0.0);

            if (any_item) active_cols.push_back(std::move(cd));

        } else if (is_prec_v[j]) {
            // ── Pre-coded (adaptive integer codes) ───────────────────────────
            int max_code = 0;
            for (int r = 0; r < n; r++) {
                double v = Xb(r, j);
                if (std::isfinite(v)) {
                    int c = static_cast<int>(v);
                    if (c > max_code) max_code = c;
                }
            }
            int nb_act = max_code + 1;

            std::vector<double> edges(static_cast<size_t>(nb_act + 1));
            for (int k = 0; k <= nb_act; k++) edges[k] = static_cast<double>(k);
            td.nb_col.push_back(nb_act);
            td.all_edges.push_back(edges);

            std::vector<double> ber_j(nb_act, 1.0 / std::max(nb_act, 1));
            td.ber.push_back(ber_j);

            // Pearson of codes vs y
            std::vector<double> df(n), yf(n);
            for (int r = 0; r < n; r++) {
                df[r] = std::isfinite(Xb(r, j)) ? Xb(r, j) : 0.0;
                yf[r] = y_vec[r];
            }
            double cv_j = pearson_cpp(df, yf);
            td.cv.push_back(cv_j);
            double eu = std::abs(cv_j);

            L1ColDesc cd;
            cd.col_idx         = j;
            cd.nb              = nb_act;
            cd.is_precoded_col = true;
            cd.bin_iu.resize(nb_act, 0.0);
            cd.bin_iid.resize(nb_act, 0);
            cd.edges           = edges;
            cd.col_min_v       = 0.0;
            cd.col_range_v     = 1.0;
            bool any_item      = false;

            for (int bi = 1; bi <= nb_act; bi++) {
                double iu_t = (cv_j >= 0.0) ? ber_j[bi-1] : ber_j[nb_act-bi];
                double eiu  = eu * iu_t;
                int bname = bk(bi, j);
                if (eiu > 0.0) {
                    ic++;
                    std::ostringstream oss;
                    oss << col_names[j] << "=["
                        << std::fixed << std::setprecision(3) << static_cast<double>(bi-1)
                        << "," << std::fixed << std::setprecision(3) << static_cast<double>(bi) << "]";
                    td.item_map[ic] = oss.str();
                    td.bn2id[bname] = ic;
                    raw_eiu[bname]  = eiu;
                    cd.bin_iu[bi-1] = eiu;
                    cd.bin_iid[bi-1]= ic;
                    any_item = true;
                    for (int yi = 0; yi < n_cls; yi++)
                        if (eiu > tu_y[yi]) tu_y[yi] = eiu;
                }
            }

            td.col_min[j]   = 0.0;
            td.col_range[j] = 1.0;
            if (any_item) active_cols.push_back(std::move(cd));

        } else {
            // ── Numeric float / integer ───────────────────────────────────────
            // minmax
            double cmin = std::numeric_limits<double>::max();
            double cmax = std::numeric_limits<double>::lowest();
            for (int r = 0; r < n; r++) {
                double v = Xb(r, j);
                if (std::isfinite(v)) { cmin = std::min(cmin, v); cmax = std::max(cmax, v); }
            }
            td.col_min[j]   = (cmin == std::numeric_limits<double>::max()) ? 0.0 : cmin;
            td.col_range[j] = (cmax > cmin) ? (cmax - cmin) : 1.0;

            // Build scaled column for kbins
            std::vector<double> col_sc(n);
            for (int r = 0; r < n; r++) {
                double v = Xb(r, j);
                col_sc[r] = std::isfinite(v)
                    ? (is_int_v[j] ? v : (v - td.col_min[j]) / td.col_range[j])
                    : std::numeric_limits<double>::quiet_NaN();
            }

            std::set<double> dset;
            for (double v : col_sc) if (std::isfinite(v)) dset.insert(v);
            int distinct = static_cast<int>(dset.size());
            int nb_use   = choose_nb_cpp(col_sc, y_vec, n_cls, B, distinct);

            auto [binned, edges] = kbins_cpp(col_sc, nb_use);
            int nb_act = static_cast<int>(edges.size()) - 1;
            td.nb_col.push_back(nb_act);
            td.all_edges.push_back(edges);

            double mx = *std::max_element(edges.begin(), edges.end());
            if (mx <= 0.0) mx = 1.0;
            std::vector<double> ber_j;
            ber_j.reserve(edges.size() - 1);
            for (int bi = 1; bi < static_cast<int>(edges.size()); bi++)
                ber_j.push_back(edges[bi] / mx);
            td.ber.push_back(ber_j);

            std::vector<double> df(n), yf(n);
            for (int r = 0; r < n; r++) { df[r] = binned[r]; yf[r] = y_vec[r]; }
            double cv_j = pearson_cpp(df, yf);
            td.cv.push_back(cv_j);
            double eu = std::abs(cv_j);

            L1ColDesc cd;
            cd.col_idx     = j;
            cd.nb          = nb_act;
            cd.is_int_col  = is_int_v[j];
            cd.bin_iu.resize(nb_act, 0.0);
            cd.bin_iid.resize(nb_act, 0);
            cd.edges       = edges;
            cd.col_min_v   = td.col_min[j];
            cd.col_range_v = td.col_range[j];
            bool any_item  = false;

            for (int bi = 1; bi <= nb_act; bi++) {
                double iu_t = (cv_j >= 0.0) ? ber_j[bi-1] : ber_j[nb_act-bi];
                double eiu  = eu * iu_t;
                int bname = bk(bi, j);
                if (eiu > 0.0) {
                    ic++;
                    auto lo_o = edges[bi-1] * td.col_range[j] + td.col_min[j];
                    auto hi_o = edges[bi]   * td.col_range[j] + td.col_min[j];
                    std::ostringstream oss;
                    oss << col_names[j] << "=["
                        << std::fixed << std::setprecision(3) << lo_o << ","
                        << std::fixed << std::setprecision(3) << hi_o << "]";
                    td.item_map[ic] = oss.str();
                    td.bn2id[bname] = ic;
                    raw_eiu[bname]  = eiu;
                    cd.bin_iu[bi-1] = eiu;
                    cd.bin_iid[bi-1]= ic;
                    any_item = true;
                    for (int yi = 0; yi < n_cls; yi++)
                        if (eiu > tu_y[yi]) tu_y[yi] = eiu;
                }
            }

            if (any_item) active_cols.push_back(std::move(cd));
        }
    }

    // ── Normalise bin_iu across all columns ──────────────────────────────────
    // tu_y[0] = max raw eiu (same direction used for ALL items since L=1 has
    // no compound utilities).  Use tu_y[0] as the normaliser (conservative:
    // items are always normalised to the global max, same as prepare_transactions).
    double global_max_eiu = 0.0;
    for (double v : tu_y) global_max_eiu = std::max(global_max_eiu, v);
    if (global_max_eiu > 0.0) {
        for (auto& cd : active_cols)
            for (double& iu : cd.bin_iu)
                iu /= global_max_eiu;
    }

    // ── item_twu: not built (skipped — TWU not needed for L=1) ───────────────
    // RIU suffices as both utility and minU seed for L=1.
    int n_items = ic;
    td.item_twu.assign(n_items, 0.0);  // placeholder; not used by L=1 mining
    td.RIU.assign(n_items, 0.0);
    td.item_col.resize(n_items, 0);

    // ── Phase 2: single fused scan ────────────────────────────────────────────
    // For each (row, active_col): bin the value, accumulate class count and RIU.
    // Simultaneously append row to item_tids[iid-1] for the COO build.

    // Per-item accumulators
    std::vector<int>              item_n_in(n_items, 0);
    std::vector<std::vector<int>> item_cls_cnt(n_items, std::vector<int>(n_cls, 0));
    std::vector<std::vector<int32_t>> item_tids(n_items);

    int loop_ctr = 0;
    const int n_active = static_cast<int>(active_cols.size());

    for (int r = 0; r < n; r++) {
        if (has_deadline) {
            if ((++loop_ctr & (CHECK - 1)) == 0)
                if (Clock::now() >= deadline_tp)
                    return {};   // partial: return empty on timeout
        }
        int cls = (r < n) ? y_vec[r] : 0;
        bool cls_valid = (cls >= 0 && cls < n_cls);

        for (int ai = 0; ai < n_active; ai++) {
            const L1ColDesc& cd = active_cols[ai];
            int j = cd.col_idx;
            int code = -1;

            if (cd.is_cat_col) {
                if (cat_raw_strs.empty() || cat_raw_strs[j].empty()) continue;
                if (cat_raw_valid.empty() || !cat_raw_valid[j][r]) continue;
                auto it = cd.label2code.find(cat_raw_strs[j][r]);
                if (it == cd.label2code.end()) continue;
                code = it->second;
            } else if (cd.is_precoded_col) {
                double v = Xb(r, j);
                if (!std::isfinite(v)) continue;
                code = std::max(0, std::min(static_cast<int>(v), cd.nb - 1));
            } else {
                double raw = Xb(r, j);
                if (!std::isfinite(raw)) continue;
                double val = cd.is_int_col ? raw
                           : (raw - cd.col_min_v) / std::max(cd.col_range_v, 1e-15);
                const int n_inner = static_cast<int>(cd.edges.size()) - 2;
                const double* inner_b = cd.edges.data() + 1;
                int pos = static_cast<int>(
                    std::upper_bound(inner_b, inner_b + n_inner, val) - inner_b);
                code = std::max(0, std::min(pos, cd.nb - 1));
            }

            if (code < 0 || code >= cd.nb) continue;
            int iid = cd.bin_iid[code];
            if (iid <= 0) continue;   // zero-IU bin
            double iu = cd.bin_iu[code];

            int idx = iid - 1;
            item_n_in[idx]++;
            td.RIU[idx] += iu;
            if (cls_valid) item_cls_cnt[idx][cls]++;
            item_tids[idx].push_back(static_cast<int32_t>(r));
        }
    }

    // item_twu: use RIU as proxy (correct upper bound for singletons)
    for (int i = 0; i < n_items; i++) td.item_twu[i] = td.RIU[i];

    // ── Mining: IG + top-K heap ───────────────────────────────────────────────
    double minU = 0.0;
    {
        std::vector<double> riu_sorted = td.RIU;
        std::sort(riu_sorted.rbegin(), riu_sorted.rend());
        if (K > 0 && !riu_sorted.empty())
            minU = riu_sorted[std::min(K - 1, static_cast<int>(riu_sorted.size()) - 1)];
    }

    std::vector<PatternEntry> heap;
    heap.reserve(static_cast<size_t>(K) + 1);

    // Iterate items in descending RIU order to tighten minU quickly
    std::vector<int> iid_order(n_items);
    std::iota(iid_order.begin(), iid_order.end(), 0);
    std::sort(iid_order.begin(), iid_order.end(),
              [&](int a, int b) { return td.RIU[a] > td.RIU[b]; });

    for (int idx : iid_order) {
        double utility = td.RIU[idx];
        if (utility < minU || utility <= 0.0) continue;
        int n_in = item_n_in[idx];
        if (n_in == 0) continue;

        double ig = l1_ig(cnt_global, item_cls_cnt[idx].data(),
                          n_in, n, n_cls);
        if (!(ig > G)) continue;   // NaN > G is false — correct

        l1_save(heap, minU, K, idx + 1, utility, ig);
    }

    // ── Build COO from item_tids ──────────────────────────────────────────────
    // Patterns are sorted descending by utility then items (same as raw_patterns_ sort).
    std::vector<PatternEntry> sorted_heap = heap;
    std::sort(sorted_heap.begin(), sorted_heap.end(),
              [](const PatternEntry& a, const PatternEntry& b) {
                  if (a.utility != b.utility) return a.utility > b.utility;
                  return a.items < b.items;
              });

    std::vector<int32_t> coo_rows, coo_cols_out;
    {
        size_t total_nnz = 0;
        for (int pi = 0; pi < static_cast<int>(sorted_heap.size()); pi++) {
            int iid = sorted_heap[pi].items[0];
            total_nnz += item_tids[iid - 1].size();
        }
        coo_rows.reserve(total_nnz);
        coo_cols_out.reserve(total_nnz);
        for (int pi = 0; pi < static_cast<int>(sorted_heap.size()); pi++) {
            int iid = sorted_heap[pi].items[0];
            for (int32_t tid : item_tids[iid - 1]) {
                coo_rows.push_back(tid);
                coo_cols_out.push_back(static_cast<int32_t>(pi));
            }
        }
    }

    // td.transactions is intentionally left EMPTY.
    // build_test_matrix uses all_edges/bn2id/nb_col — all populated above.

    L1FitResult result;
    result.td       = std::move(td);
    result.patterns = std::move(sorted_heap);
    result.coo_rows = std::move(coo_rows);
    result.coo_cols = std::move(coo_cols_out);
    return result;
}

}  // namespace hugiml

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
#include "resource_guard.hpp"
#include "discretization.hpp"
#include "math.hpp"
#include "mining.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <set>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

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

static double l1_ig_multicodes(const std::vector<int>& codes,
                               const std::vector<int>& y_vec,
                               const std::vector<int>& cnt_global,
                               int n_valid, int n_cls) {
    if (n_valid <= 0 || n_cls <= 1) return 0.0;
    int max_code = -1;
    for (int c : codes) if (c >= 0 && c > max_code) max_code = c;
    if (max_code < 0) return 0.0;
    const int n_codes = max_code + 1;
    std::vector<int> code_counts(n_codes, 0);
    std::vector<std::vector<int>> code_cls(n_codes, std::vector<int>(n_cls, 0));
    for (int r = 0; r < static_cast<int>(codes.size()); r++) {
        int c = codes[r];
        if (c < 0) continue;
        int cls = y_vec[r];
        if (cls < 0 || cls >= n_cls) continue;
        code_counts[c]++;
        code_cls[c][cls]++;
    }
    double base = l1_entropy_ln(cnt_global, n_valid);
    double cond = 0.0;
    for (int c = 0; c < n_codes; c++) {
        if (code_counts[c] > 0) {
            cond += (static_cast<double>(code_counts[c]) / static_cast<double>(n_valid)) *
                    l1_entropy_ln(code_cls[c], code_counts[c]);
        }
    }
    return std::max(0.0, base - cond);
}

static double l1_ig_binary_indicator(const std::vector<int>& y_vec,
                                     const std::vector<int>& cnt_global,
                                     int n_cls,
                                     const std::vector<uint8_t>& present) {
    const int n = static_cast<int>(y_vec.size());
    if (n <= 0 || n_cls <= 1) return 0.0;
    std::vector<int> cnt_in(n_cls, 0);
    int n_in = 0;
    for (int r = 0; r < n; r++) {
        if (!present[r]) continue;
        int cls = y_vec[r];
        if (cls < 0 || cls >= n_cls) continue;
        cnt_in[cls]++;
        n_in++;
    }
    if (n_in <= 0 || n_in >= n) return 0.0;
    return std::max(0.0, l1_ig(cnt_global, cnt_in.data(), n_in, n, n_cls));
}

static std::string l1_dummy_name(const std::string& col, const std::string& value) {
    return col + "_" + value;
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
    bool is_precoded_col     = false;
    bool is_adaptive_raw_col = false;  // raw values + adaptive edges, no X_codes materialization
    bool is_cat_col          = false;

    // For adaptive raw columns: original-value adaptive quantile edges.
    std::vector<double> adaptive_edges;

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

    // Select the non-categorical columns first so the parallel workers can
    // write into pre-sized output slots.  This preserves the historical output
    // order exactly while avoiding a serial push_back bottleneck.
    std::vector<int> num_indices;
    num_indices.reserve(p);
    for (int j = 0; j < p; j++) {
        if (j < static_cast<int>(icb.size()) && static_cast<bool>(icb(j))) {
            continue;
        }
        num_indices.push_back(j);
    }

    const int n_num = static_cast<int>(num_indices.size());
    result.n_num_cols = n_num;
    result.cols.resize(n_num);
    result.num_col_indices.resize(n_num);

    std::exception_ptr first_error = nullptr;
    bool failed = false;

    // Per-feature adaptive-B selection is independent and therefore safe to
    // parallelise.  Each worker owns its column buffer and result slot.
    #pragma omp parallel for schedule(dynamic, 1) if(n_num > 1)
    for (int ci = 0; ci < n_num; ci++) {
        if (failed) continue;
        try {
            int j = num_indices[ci];

            std::vector<double> col_raw(n);
            for (int r = 0; r < n; r++) col_raw[r] = Xb(r, j);

            std::vector<double> out_edges, out_ig;
            int chosen = elbow_stop_nb_cpp(col_raw, y_vec, n_cls,
                                            candidates, ratio,
                                            out_edges, out_ig);

            ColAdaptResult cr;
            cr.chosen_b  = chosen;
            cr.edges     = std::move(out_edges);
            cr.ig_scores = std::move(out_ig);
            result.cols[ci] = std::move(cr);
            result.num_col_indices[ci] = j;
        } catch (...) {
            #pragma omp critical(hugiml_select_adaptive_error)
            {
                if (!first_error) first_error = std::current_exception();
                failed = true;
            }
        }
    }

    if (first_error) std::rethrow_exception(first_error);

    // Build X_codes_flat: (n × n_num_cols) row-major int32 codes.  Codes are
    // small bin ids; storing them as float64 doubled memory and forced another
    // Python float64 matrix in get_X_codes().  Use -1 as the missing/non-finite
    // sentinel and only cast one selected column to float in Python when a
    // legacy pre-binned DataFrame must be materialized.
    ensure_native_memory_available(
        static_cast<uint64_t>(n) * static_cast<uint64_t>(std::max(n_num, 0)) * sizeof(int32_t),
        "select_adaptive_bins_cpp X_codes_flat_int32(n=" + std::to_string(n) +
        ", n_num_cols=" + std::to_string(n_num) + ")");
    result.X_codes_flat.resize(static_cast<size_t>(n) * static_cast<size_t>(n_num), int32_t{-1});

    // Apply the selected edges row-wise.  The previous column-wise loop wrote
    // into row-major output with a stride of n_num_cols; this row-wise loop
    // writes contiguous output and is also parallel across rows.
    std::exception_ptr code_error = nullptr;
    failed = false;
    #pragma omp parallel for schedule(static) if(n > 4096 && n_num > 0)
    for (int r = 0; r < n; r++) {
        if (failed) continue;
        try {
            const size_t out_base = static_cast<size_t>(r) * static_cast<size_t>(n_num);
            for (int ci = 0; ci < n_num; ci++) {
                int j = result.num_col_indices[ci];
                const auto& edges = result.cols[ci].edges;
                int nb = result.cols[ci].chosen_b;
                const int n_inner = static_cast<int>(edges.size()) - 2;
                const double* inner_begin = edges.data() + 1;

                double raw = Xb(r, j);
                int32_t code = -1;
                if (std::isfinite(raw)) {
                    int pos = static_cast<int>(
                        std::upper_bound(inner_begin, inner_begin + n_inner, raw)
                        - inner_begin);
                    code = static_cast<int32_t>(std::max(0, std::min(pos, nb - 1)));
                }
                result.X_codes_flat[out_base + static_cast<size_t>(ci)] = code;
            }
        } catch (...) {
            #pragma omp critical(hugiml_select_adaptive_code_error)
            {
                if (!code_error) code_error = std::current_exception();
                failed = true;
            }
        }
    }
    if (code_error) std::rethrow_exception(code_error);

    return result;
}

// ── prepare_and_mine_l1_cpp ───────────────────────────────────────────────────

static L1FitResult prepare_and_mine_l1_cpp_impl(
    const py::array_t<double,  py::array::c_style | py::array::forcecast>& X_num_arr,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& y_arr,
    int B,
    std::vector<std::string>               col_names,
    const py::array_t<uint8_t, py::array::forcecast>& is_cat_arr,
    const py::array_t<uint8_t, py::array::forcecast>& is_int_arr,
    std::vector<bool>                      is_precoded,
    std::vector<std::vector<std::string>>  cat_raw_strs,
    std::vector<std::vector<bool>>         cat_raw_valid,
    int K, double G, double timeout_s,
    bool compute_original_scores,
    const std::vector<int>*                adaptive_num_indices,
    const std::vector<ColAdaptResult>*     adaptive_cols_meta)
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

    // Conservative native guard for the fused hot path.  The final COO and
    // per-item TID lists can both scale with active n*p occurrences.  Refuse
    // early under a configured/rlimit budget so callers receive MemoryError
    // instead of an OS-level kill.
    {
        const uint64_t dense_cells = static_cast<uint64_t>(n) * static_cast<uint64_t>(p);
        const uint64_t rough_native_bytes = dense_cells * (sizeof(int32_t) * 2ULL) +
                                           static_cast<uint64_t>(n) * 16ULL;
        ensure_native_memory_available(rough_native_bytes,
                                       "prepare_and_mine_l1_cpp(n=" + std::to_string(n) +
                                       ", p=" + std::to_string(p) + ")");
    }

    check_timeout_deadline(has_deadline, deadline_tp, "prepare_and_mine_l1 setup");

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

    // Optional fused-adaptive mode: numeric columns remain raw in X_num, but
    // Phase 1/2 use preselected adaptive edges.  The fitted td still marks them
    // as precoded for prediction, because predict() pre-bins raw values to codes.
    std::vector<int> adaptive_spec_idx(p, -1);
    if (adaptive_num_indices && adaptive_cols_meta) {
        for (int ci = 0; ci < static_cast<int>(adaptive_num_indices->size()); ci++) {
            int j = (*adaptive_num_indices)[ci];
            if (j >= 0 && j < p) {
                adaptive_spec_idx[j] = ci;
                is_cat_v[j] = false;
                is_int_v[j] = false;
                is_prec_v[j] = true;
            }
        }
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

    // Downstream original-feature prefilter scores are computed while the
    // native preparation pass already has per-column bins/categories in hand.
    // Numeric originals are represented by their raw column name; categorical
    // originals are represented by pandas get_dummies-compatible names
    // (<col>_<value> plus <col>_<NA> when missing values were observed).
    std::vector<std::string> original_numeric_names;
    std::vector<double>      original_numeric_scores;
    std::vector<std::string> original_dummy_names;
    std::vector<double>      original_dummy_scores;
    if (compute_original_scores) {
        original_numeric_names.reserve(p);
        original_numeric_scores.reserve(p);
    }

    for (int j = 0; j < p; j++) {
        if (has_deadline && ((j & (CHECK - 1)) == 0))
            check_timeout_deadline(has_deadline, deadline_tp, "prepare_and_mine_l1 column preparation");

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

            // Score downstream dummy originals only when original features can
            // participate downstream. patterns_only must not pay this cost.
            if (compute_original_scores) {
                for (int idx = 0; idx < C; idx++) {
                    std::vector<uint8_t> present(static_cast<size_t>(n), 0);
                    for (int r = 0; r < n; r++)
                        if (col_codes[r] == static_cast<int32_t>(idx)) present[static_cast<size_t>(r)] = 1;
                    original_dummy_names.push_back(l1_dummy_name(col_names[j], uniq[idx]));
                    original_dummy_scores.push_back(
                        l1_ig_binary_indicator(y_vec, cnt_global, n_cls, present));
                }
                bool any_missing_for_dummy = false;
                if (have_cat && have_valid) {
                    for (int r = 0; r < n; r++) {
                        if (!cat_raw_valid[j][r]) { any_missing_for_dummy = true; break; }
                    }
                }
                if (any_missing_for_dummy) {
                    std::vector<uint8_t> present(static_cast<size_t>(n), 0);
                    for (int r = 0; r < n; r++)
                        if (!cat_raw_valid[j][r]) present[static_cast<size_t>(r)] = 1;
                    original_dummy_names.push_back(l1_dummy_name(col_names[j], "<NA>"));
                    original_dummy_scores.push_back(
                        l1_ig_binary_indicator(y_vec, cnt_global, n_cls, present));
                }
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

        } else if (adaptive_spec_idx[j] >= 0) {
            // ── Adaptive raw numeric column (no X_codes materialization) ─────
            const ColAdaptResult& acr = (*adaptive_cols_meta)[adaptive_spec_idx[j]];
            std::vector<double> raw_edges = acr.edges;
            int nb_act = static_cast<int>(raw_edges.size()) - 1;
            if (nb_act <= 0) {
                td.nb_col.push_back(0);
                td.all_edges.push_back({0.0});
                td.ber.push_back({});
                td.cv.push_back(0.0);
                continue;
            }

            // Downstream predict() will pre-bin raw values to integer codes, so
            // td.all_edges remains the code-space [0, 1, ..., B_j] exactly like
            // the previous two-step adaptive path.
            std::vector<double> code_edges(static_cast<size_t>(nb_act + 1));
            for (int k = 0; k <= nb_act; k++) code_edges[k] = static_cast<double>(k);
            td.nb_col.push_back(nb_act);
            td.all_edges.push_back(code_edges);

            std::vector<double> ber_j(nb_act, 1.0 / std::max(nb_act, 1));
            td.ber.push_back(ber_j);

            const int n_inner = static_cast<int>(raw_edges.size()) - 2;
            const double* inner_begin = raw_edges.data() + 1;
            auto raw_to_code = [&](double raw) -> int {
                if (!std::isfinite(raw)) return -1;
                int pos = static_cast<int>(
                    std::upper_bound(inner_begin, inner_begin + n_inner, raw) - inner_begin);
                return std::max(0, std::min(pos, nb_act - 1));
            };

            std::vector<double> df(n), yf(n);
            std::vector<int> original_codes(n, -1);
            for (int r = 0; r < n; r++) {
                int code = raw_to_code(Xb(r, j));
                original_codes[r] = code;
                df[r] = (code >= 0) ? static_cast<double>(code) : 0.0;
                yf[r] = y_vec[r];
            }
            if (compute_original_scores) {
                original_numeric_names.push_back(col_names[j]);
                original_numeric_scores.push_back(
                    l1_ig_multicodes(original_codes, y_vec, cnt_global, n, n_cls));
            }
            double cv_j = pearson_cpp(df, yf);
            td.cv.push_back(cv_j);
            double eu = std::abs(cv_j);

            L1ColDesc cd;
            cd.col_idx             = j;
            cd.nb                  = nb_act;
            cd.is_adaptive_raw_col = true;
            cd.is_precoded_col     = false;
            cd.bin_iu.resize(nb_act, 0.0);
            cd.bin_iid.resize(nb_act, 0);
            cd.edges               = code_edges;
            cd.adaptive_edges      = raw_edges;
            cd.col_min_v           = 0.0;
            cd.col_range_v         = 1.0;
            bool any_item          = false;

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
            std::vector<int> original_codes(n, -1);
            for (int r = 0; r < n; r++) {
                if (std::isfinite(Xb(r, j))) {
                    original_codes[r] = std::max(0, std::min(static_cast<int>(Xb(r, j)), nb_act - 1));
                    df[r] = Xb(r, j);
                } else {
                    df[r] = 0.0;
                }
                yf[r] = y_vec[r];
            }
            if (compute_original_scores) {
                original_numeric_names.push_back(col_names[j]);
                original_numeric_scores.push_back(
                    l1_ig_multicodes(original_codes, y_vec, cnt_global, n, n_cls));
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
            std::vector<int> original_codes(n, -1);
            for (int r = 0; r < n; r++) {
                if (std::isfinite(col_sc[r])) original_codes[r] = static_cast<int>(binned[r]);
            }
            if (compute_original_scores) {
                original_numeric_names.push_back(col_names[j]);
                original_numeric_scores.push_back(
                    l1_ig_multicodes(original_codes, y_vec, cnt_global, n, n_cls));
            }
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

    // Preserve downstream original-feature ordering used by Python:
    // numeric original columns first, then categorical dummy columns.
    std::vector<std::string> original_feature_names;
    std::vector<double> original_feature_scores;
    if (compute_original_scores) {
        original_feature_names.reserve(original_numeric_names.size() + original_dummy_names.size());
        original_feature_scores.reserve(original_numeric_scores.size() + original_dummy_scores.size());
        original_feature_names.insert(original_feature_names.end(),
                                      original_numeric_names.begin(), original_numeric_names.end());
        original_feature_scores.insert(original_feature_scores.end(),
                                       original_numeric_scores.begin(), original_numeric_scores.end());
        original_feature_names.insert(original_feature_names.end(),
                                      original_dummy_names.begin(), original_dummy_names.end());
        original_feature_scores.insert(original_feature_scores.end(),
                                       original_dummy_scores.begin(), original_dummy_scores.end());
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
    const int n_active = static_cast<int>(active_cols.size());
    ensure_native_memory_available(
        static_cast<uint64_t>(n) * static_cast<uint64_t>(std::max(n_active, 0)) *
        sizeof(int32_t) * 2ULL,
        "prepare_and_mine_l1 item TID/COO buffers");
    std::vector<int>              item_n_in(n_items, 0);
    std::vector<std::vector<int>> item_cls_cnt(n_items, std::vector<int>(n_cls, 0));
    std::vector<std::vector<int32_t>> item_tids(n_items);

    int loop_ctr = 0;

    for (int r = 0; r < n; r++) {
        if (has_deadline) {
            if ((++loop_ctr & (CHECK - 1)) == 0)
                if (Clock::now() >= deadline_tp)
                    throw std::runtime_error("hugiml_timeout: prepare_and_mine_l1 scan exceeded timeout_s");
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
            } else if (cd.is_adaptive_raw_col) {
                double raw = Xb(r, j);
                if (!std::isfinite(raw)) continue;
                const int n_inner = static_cast<int>(cd.adaptive_edges.size()) - 2;
                const double* inner_b = cd.adaptive_edges.data() + 1;
                int pos = static_cast<int>(
                    std::upper_bound(inner_b, inner_b + n_inner, raw) - inner_b);
                code = std::max(0, std::min(pos, cd.nb - 1));
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
    result.original_feature_names = std::move(original_feature_names);
    result.original_feature_scores = std::move(original_feature_scores);
    return result;
}


// Metadata-only adaptive B selection used by prepare_and_mine_l1_adaptive_cpp.
// Unlike select_adaptive_bins_cpp, this deliberately does NOT allocate or fill
// X_codes_flat; the later fused scan bins raw values on the fly.
static AdaptiveBinResult select_adaptive_bins_metadata_only_cpp(
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
    std::vector<int> num_indices;
    num_indices.reserve(p);
    for (int j = 0; j < p; j++) {
        if (j < static_cast<int>(icb.size()) && static_cast<bool>(icb(j))) continue;
        num_indices.push_back(j);
    }
    const int n_num = static_cast<int>(num_indices.size());
    result.n_num_cols = n_num;
    result.cols.resize(n_num);
    result.num_col_indices.resize(n_num);

    std::exception_ptr first_error = nullptr;
    bool failed = false;
    #pragma omp parallel for schedule(dynamic, 1) if(n_num > 1)
    for (int ci = 0; ci < n_num; ci++) {
        if (failed) continue;
        try {
            int j = num_indices[ci];
            std::vector<double> col_raw(n);
            for (int r = 0; r < n; r++) col_raw[r] = Xb(r, j);
            std::vector<double> out_edges, out_ig;
            int chosen = elbow_stop_nb_cpp(col_raw, y_vec, n_cls,
                                            candidates, ratio,
                                            out_edges, out_ig);
            ColAdaptResult cr;
            cr.chosen_b  = chosen;
            cr.edges     = std::move(out_edges);
            cr.ig_scores = std::move(out_ig);
            result.cols[ci] = std::move(cr);
            result.num_col_indices[ci] = j;
        } catch (...) {
            #pragma omp critical(hugiml_select_adaptive_meta_error)
            {
                if (!first_error) first_error = std::current_exception();
                failed = true;
            }
        }
    }
    if (first_error) std::rethrow_exception(first_error);
    return result;
}

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
    int K, double G, double timeout_s, bool compute_original_scores)
{
    return prepare_and_mine_l1_cpp_impl(
        X_num_arr, y_arr, B, std::move(col_names), is_cat_arr, is_int_arr,
        std::move(is_precoded), std::move(cat_raw_strs), std::move(cat_raw_valid),
        K, G, timeout_s, compute_original_scores, nullptr, nullptr);
}

L1FitResult prepare_and_mine_l1_adaptive_cpp(
    const py::array_t<double,  py::array::c_style | py::array::forcecast>& X_num_arr,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& y_arr,
    std::vector<std::string>               col_names,
    const py::array_t<uint8_t, py::array::forcecast>& is_cat_arr,
    const py::array_t<uint8_t, py::array::forcecast>& is_int_arr,
    std::vector<std::vector<std::string>>  cat_raw_strs,
    std::vector<std::vector<bool>>         cat_raw_valid,
    const std::vector<int>&                candidates,
    double                                 ratio,
    int K, double G, double timeout_s, bool compute_original_scores)
{
    auto yb = y_arr.unchecked<1>();
    int n = static_cast<int>(yb.shape(0));
    int max_label = 0;
    for (int i = 0; i < n; i++) max_label = std::max(max_label, static_cast<int>(yb(i)));
    int n_cls = max_label + 1;

    AdaptiveBinResult meta = select_adaptive_bins_metadata_only_cpp(
        X_num_arr, y_arr, n_cls, col_names, is_cat_arr, candidates, ratio);

    L1FitResult out = prepare_and_mine_l1_cpp_impl(
        X_num_arr, y_arr, 2, std::move(col_names), is_cat_arr, is_int_arr,
        std::vector<bool>{}, std::move(cat_raw_strs), std::move(cat_raw_valid),
        K, G, timeout_s, compute_original_scores, &meta.num_col_indices, &meta.cols);
    out.adaptive_cols = std::move(meta.cols);
    out.adaptive_num_col_indices = std::move(meta.num_col_indices);
    return out;
}


// Fast fixed-B path for dense numeric L=1 hotpath.  It avoids the historical
// Python string pre-binning and avoids materialised transactions.  Missing or
// non-finite cells are skipped during item generation, for both train and test.
namespace {
struct FixedNumColDesc {
    int col_idx = 0;
    int nb = 0;
    bool is_int_col = false;
    bool valid = false;
    std::vector<double> edges;
    std::vector<double> ber;
    double col_min_v = 0.0;
    double col_range_v = 1.0;
    double cv = 0.0;
    std::vector<double> raw_eiu;
    std::vector<int> bin_iid;
    double original_score = 0.0;
};

static std::vector<double> percentile_edges_from_sorted_unique(std::vector<double>& sorted_vals, int nb) {
    const size_t m = sorted_vals.size();
    if (m == 0) return {0.0, 1.0};
    double step = 100.0 / std::max(nb, 1);
    std::vector<double> edges;
    edges.reserve(static_cast<size_t>(nb + 1));
    for (int i = 0; i <= nb; i++) {
        double q = step * static_cast<double>(i);
        double fidx = q / 100.0 * static_cast<double>(m - 1);
        size_t lo = static_cast<size_t>(fidx);
        size_t hi = std::min(lo + 1, m - 1);
        double frac = fidx - static_cast<double>(lo);
        edges.push_back(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]));
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    if (edges.size() < 2) edges = {sorted_vals.front(), sorted_vals.front() + 1e-9};
    return edges;
}

static inline int fixed_code_from_raw(double raw, const FixedNumColDesc& cd) {
    if (!std::isfinite(raw) || cd.nb <= 0) return -1;
    double val = cd.is_int_col ? raw : (raw - cd.col_min_v) / std::max(cd.col_range_v, 1e-15);
    const int n_inner = static_cast<int>(cd.edges.size()) - 2;
    const double* inner_b = cd.edges.data() + 1;
    int pos = static_cast<int>(std::upper_bound(inner_b, inner_b + n_inner, val) - inner_b);
    return std::max(0, std::min(pos, cd.nb - 1));
}
} // anonymous namespace

L1FitResult prepare_and_mine_l1_fixed_numeric_cpp(
    const py::array_t<double,  py::array::c_style | py::array::forcecast>& X_num_arr,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& y_arr,
    int B,
    std::vector<std::string> col_names,
    const py::array_t<uint8_t, py::array::forcecast>& is_int_arr,
    int K, double G, double timeout_s, bool compute_original_scores)
{
    using Clock = std::chrono::steady_clock;
    bool has_deadline = timeout_s > 0.0;
    auto deadline_tp = has_deadline
        ? Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout_s))
        : Clock::time_point{};

    auto Xb = X_num_arr.unchecked<2>();
    auto yb = y_arr.unchecked<1>();
    auto iib = is_int_arr.unchecked<1>();
    const int n = static_cast<int>(Xb.shape(0));
    const int p = static_cast<int>(Xb.shape(1));
    if (B < 2) throw std::invalid_argument("B must be >= 2 for fixed-B L1 hotpath");

    if (static_cast<int>(col_names.size()) != p) {
        col_names.resize(p);
        for (int j = 0; j < p; ++j) if (col_names[j].empty()) col_names[j] = "col" + std::to_string(j);
    }

    std::vector<int> y_vec(n);
    int max_label = 0;
    for (int i = 0; i < n; ++i) { y_vec[i] = static_cast<int>(yb(i)); if (y_vec[i] > max_label) max_label = y_vec[i]; }
    const int n_cls = max_label + 1;
    std::vector<int> cnt_global(n_cls, 0);
    for (int v : y_vec) if (v >= 0 && v < n_cls) cnt_global[v]++;

    std::vector<FixedNumColDesc> descs(p);
    std::exception_ptr first_error = nullptr;
    bool failed = false;

    #pragma omp parallel for schedule(dynamic, 1) if(p > 1)
    for (int j = 0; j < p; ++j) {
        if (failed) continue;
        try {
            if (has_deadline && Clock::now() >= deadline_tp)
                throw std::runtime_error("hugiml_timeout: fixed-B L1 column preparation exceeded timeout_s");
            FixedNumColDesc cd;
            cd.col_idx = j;
            cd.is_int_col = (j < static_cast<int>(iib.size()) && static_cast<bool>(iib(j)));

            double cmin = std::numeric_limits<double>::max();
            double cmax = std::numeric_limits<double>::lowest();
            for (int r = 0; r < n; ++r) {
                double v = Xb(r, j);
                if (std::isfinite(v)) { cmin = std::min(cmin, v); cmax = std::max(cmax, v); }
            }
            if (cmin == std::numeric_limits<double>::max()) {
                descs[j] = std::move(cd);
                continue;
            }
            cd.col_min_v = cd.is_int_col ? 0.0 : cmin;
            cd.col_range_v = cd.is_int_col ? 1.0 : ((cmax > cmin) ? (cmax - cmin) : 1.0);

            std::vector<double> col_sc(static_cast<size_t>(n));
            std::set<double> dset;
            for (int r = 0; r < n; ++r) {
                double raw = Xb(r, j);
                double scaled = std::isfinite(raw)
                    ? (cd.is_int_col ? raw : (raw - cd.col_min_v) / cd.col_range_v)
                    : std::numeric_limits<double>::quiet_NaN();
                col_sc[static_cast<size_t>(r)] = scaled;
                if (std::isfinite(scaled)) dset.insert(scaled);
            }
            const int distinct = static_cast<int>(dset.size());
            if (distinct <= 0) { descs[j] = std::move(cd); continue; }
            int nb_use = choose_nb_cpp(col_sc, y_vec, n_cls, B, distinct);
            auto kb = kbins_cpp(col_sc, nb_use);
            std::vector<int>& binned = kb.first;
            if (compute_original_scores) {
                std::vector<int> original_codes(static_cast<size_t>(n), -1);
                for (int r = 0; r < n; ++r) {
                    if (std::isfinite(col_sc[static_cast<size_t>(r)]))
                        original_codes[static_cast<size_t>(r)] = binned[static_cast<size_t>(r)];
                }
                cd.original_score = l1_ig_multicodes(original_codes, y_vec, cnt_global, n, n_cls);
            }
            cd.edges = std::move(kb.second);
            cd.nb = static_cast<int>(cd.edges.size()) - 1;
            if (cd.nb <= 0) { descs[j] = std::move(cd); continue; }

            double mx = *std::max_element(cd.edges.begin(), cd.edges.end());
            if (mx <= 0.0) mx = 1.0;
            cd.ber.reserve(static_cast<size_t>(cd.nb));
            for (int bi = 1; bi < static_cast<int>(cd.edges.size()); ++bi)
                cd.ber.push_back(cd.edges[bi] / mx);

            std::vector<double> df(static_cast<size_t>(n)), yf(static_cast<size_t>(n));
            for (int r = 0; r < n; ++r) {
                // Match the established fixed-B hotpath exactly: kbins_cpp
                // assigns missing/non-finite cells to code 0 for the Pearson
                // direction calculation; later row scans still skip missing
                // cells so no item is generated.
                df[static_cast<size_t>(r)] = static_cast<double>(binned[static_cast<size_t>(r)]);
                yf[static_cast<size_t>(r)] = static_cast<double>(y_vec[static_cast<size_t>(r)]);
            }
            cd.cv = pearson_cpp(df, yf);
            if (!std::isfinite(cd.cv)) cd.cv = 0.0;

            cd.raw_eiu.assign(static_cast<size_t>(cd.nb), 0.0);
            cd.bin_iid.assign(static_cast<size_t>(cd.nb), 0);
            double eu = std::abs(cd.cv);
            for (int bi = 1; bi <= cd.nb; ++bi) {
                double iu_t = (cd.cv >= 0.0) ? cd.ber[bi-1] : cd.ber[cd.nb-bi];
                cd.raw_eiu[bi-1] = eu * iu_t;
            }
            cd.valid = true;
            descs[j] = std::move(cd);
        } catch (...) {
            #pragma omp critical(hugiml_fixed_l1_prep_error)
            { if (!first_error) first_error = std::current_exception(); failed = true; }
        }
    }
    if (first_error) std::rethrow_exception(first_error);

    TransactionDataCpp td;
    td.disc_n = n; td.disc_p = p; td.bkey_stride = std::max(p, 10000);
    td.is_cat_v.assign(p, false);
    td.is_int_v.resize(p, false);
    td.is_precoded_v.assign(p, false);
    td.nb_col.assign(p, 0);
    td.all_edges.resize(p);
    td.ber.resize(p);
    td.cv.assign(p, 0.0);
    td.col_min.assign(p, 0.0);
    td.col_range.assign(p, 1.0);
    td.cat_categories.resize(p);

    double global_max_eiu = 0.0;
    for (const auto& cd : descs)
        for (double v : cd.raw_eiu) global_max_eiu = std::max(global_max_eiu, v);

    int ic = 0;
    std::vector<int> active_js;
    active_js.reserve(p);
    std::vector<std::string> original_feature_names;
    std::vector<double> original_feature_scores;
    if (compute_original_scores) {
        original_feature_names.reserve(static_cast<size_t>(p));
        original_feature_scores.reserve(static_cast<size_t>(p));
    }
    for (int j = 0; j < p; ++j) {
        auto& cd = descs[j];
        td.is_int_v[j] = cd.is_int_col;
        td.nb_col[j] = cd.nb;
        td.all_edges[j] = cd.edges.empty() ? std::vector<double>{0.0, 1.0} : cd.edges;
        td.ber[j] = cd.ber;
        td.cv[j] = cd.cv;
        td.col_min[j] = cd.col_min_v;
        td.col_range[j] = cd.col_range_v;
        if (!cd.valid || cd.nb <= 0) continue;
        if (compute_original_scores) {
            original_feature_names.push_back(col_names[j]);
            original_feature_scores.push_back(cd.original_score);
        }
        bool any = false;
        for (int bi = 1; bi <= cd.nb; ++bi) {
            double raw_eiu = cd.raw_eiu[bi-1];
            if (raw_eiu <= 0.0) continue;
            ic++;
            int bname = td.bkey(bi, j);
            double lo_o = cd.edges[bi-1] * cd.col_range_v + cd.col_min_v;
            double hi_o = cd.edges[bi]   * cd.col_range_v + cd.col_min_v;
            std::ostringstream oss;
            oss << col_names[j] << "=[" << std::fixed << std::setprecision(3) << lo_o
                << "," << std::fixed << std::setprecision(3) << hi_o << "]";
            td.item_map[ic] = oss.str();
            td.bn2id[bname] = ic;
            cd.bin_iid[bi-1] = ic;
            any = true;
        }
        if (any) active_js.push_back(j);
    }

    const int n_items = ic;
    td.RIU.assign(n_items, 0.0);
    td.item_twu.assign(n_items, 0.0);
    td.item_col.resize(n_items, 0);
    td.item_iu.assign(n_items, 0.0);
    for (int j : active_js) {
        auto& cd = descs[j];
        for (int bi = 1; bi <= cd.nb; ++bi) {
            int iid = cd.bin_iid[bi-1];
            if (iid <= 0) continue;
            double iu = (global_max_eiu > 0.0) ? cd.raw_eiu[bi-1] / global_max_eiu : 0.0;
            cd.raw_eiu[bi-1] = iu;
            td.item_iu[iid-1] = iu;
            td.item_col[iid-1] = j;
        }
    }

    const int n_threads =
#ifdef _OPENMP
        std::max(1, omp_get_max_threads());
#else
        1;
#endif
    std::vector<std::vector<int>> tl_n_in(n_threads, std::vector<int>(n_items, 0));
    std::vector<std::vector<double>> tl_riu(n_threads, std::vector<double>(n_items, 0.0));
    std::vector<std::vector<int>> tl_cls(n_threads, std::vector<int>(static_cast<size_t>(n_items) * std::max(n_cls,1), 0));

    #pragma omp parallel for schedule(static) if(n > 4096 && !active_js.empty())
    for (int r = 0; r < n; ++r) {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        int cls = y_vec[r];
        bool cls_valid = cls >= 0 && cls < n_cls;
        auto& loc_n = tl_n_in[tid];
        auto& loc_r = tl_riu[tid];
        auto& loc_c = tl_cls[tid];
        for (int j : active_js) {
            const auto& cd = descs[j];
            int code = fixed_code_from_raw(Xb(r, j), cd);
            if (code < 0 || code >= cd.nb) continue;
            int iid = cd.bin_iid[code];
            if (iid <= 0) continue;
            int idx = iid - 1;
            loc_n[idx]++;
            loc_r[idx] += cd.raw_eiu[code];
            if (cls_valid) loc_c[static_cast<size_t>(idx) * n_cls + cls]++;
        }
    }

    std::vector<int> item_n_in(n_items, 0);
    std::vector<std::vector<int>> item_cls_cnt(n_items, std::vector<int>(n_cls, 0));
    for (int t = 0; t < n_threads; ++t) {
        for (int i = 0; i < n_items; ++i) {
            item_n_in[i] += tl_n_in[t][i];
            td.RIU[i] += tl_riu[t][i];
            for (int c = 0; c < n_cls; ++c)
                item_cls_cnt[i][c] += tl_cls[t][static_cast<size_t>(i) * n_cls + c];
        }
    }
    td.item_twu = td.RIU;

    double minU = 0.0;
    {
        std::vector<double> riu_sorted = td.RIU;
        std::sort(riu_sorted.rbegin(), riu_sorted.rend());
        if (K > 0 && !riu_sorted.empty()) minU = riu_sorted[std::min(K-1, static_cast<int>(riu_sorted.size())-1)];
    }
    std::vector<PatternEntry> heap;
    heap.reserve(static_cast<size_t>(K) + 1);
    std::vector<int> iid_order(n_items);
    std::iota(iid_order.begin(), iid_order.end(), 0);
    std::sort(iid_order.begin(), iid_order.end(), [&](int a, int b){ return td.RIU[a] > td.RIU[b]; });
    for (int idx : iid_order) {
        double utility = td.RIU[idx];
        if (utility < minU || utility <= 0.0) continue;
        int n_in = item_n_in[idx];
        if (n_in == 0) continue;
        double ig = l1_ig(cnt_global, item_cls_cnt[idx].data(), n_in, n, n_cls);
        if (!(ig > G)) continue;
        l1_save(heap, minU, K, idx + 1, utility, ig);
    }
    std::vector<PatternEntry> sorted_heap = heap;
    std::sort(sorted_heap.begin(), sorted_heap.end(), [](const PatternEntry& a, const PatternEntry& b){
        if (a.utility != b.utility) return a.utility > b.utility;
        return a.items < b.items;
    });

    std::vector<int> iid_to_pat(n_items, -1);
    for (int pi = 0; pi < static_cast<int>(sorted_heap.size()); ++pi)
        iid_to_pat[sorted_heap[pi].items[0] - 1] = pi;
    std::vector<int32_t> coo_rows, coo_cols;
    coo_rows.reserve(static_cast<size_t>(n) * sorted_heap.size() / 2);
    coo_cols.reserve(coo_rows.capacity());
    for (int r = 0; r < n; ++r) {
        for (int j : active_js) {
            const auto& cd = descs[j];
            int code = fixed_code_from_raw(Xb(r, j), cd);
            if (code < 0 || code >= cd.nb) continue;
            int iid = cd.bin_iid[code];
            if (iid <= 0) continue;
            int pi = iid_to_pat[iid-1];
            if (pi >= 0) { coo_rows.push_back(static_cast<int32_t>(r)); coo_cols.push_back(static_cast<int32_t>(pi)); }
        }
    }

    L1FitResult result;
    result.td = std::move(td);
    if (compute_original_scores) {
        result.original_feature_names = std::move(original_feature_names);
        result.original_feature_scores = std::move(original_feature_scores);
    }
    result.patterns = std::move(sorted_heap);
    result.coo_rows = std::move(coo_rows);
    result.coo_cols = std::move(coo_cols);
    return result;
}

}  // namespace hugiml

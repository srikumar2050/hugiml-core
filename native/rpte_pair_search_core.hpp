/**
 * rpte_pair_search_core.hpp — fused raw-feature-pair root search, pure
 * C++ core (no pybind11 or GIL involvement).
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 *
 * For many candidate raw-feature pairs x operations x thresholds,
 * scores every combination against a precomputed child-candidate
 * pool, keeping the top few children per combination. This is the
 * fallback root-candidate search used when none of the mined
 * augmented-pair columns yield a viable lookahead candidate at a
 * leaf: it synthesizes root candidates directly from pairs of raw
 * features instead.
 *
 * `rpte_scoring.cpp`'s pybind11-facing `rpte_raw_pair_root_search`
 * wraps this for direct calls from Python. The native tree-growth
 * engine (rpte_tree.cpp) calls it directly, since it runs with the
 * GIL released for its entire search and cannot safely call a
 * function that releases the GIL a second time.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "rpte_core.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hugiml::rpte_core {

struct PairSearchResult {
    std::vector<int32_t> a_idx, b_idx, op;
    std::vector<double> threshold, grow_gain;
    std::vector<uint8_t> is_binarized;
    // child_positions[k] / child_scores[k]: up to top_children_per_candidate
    // entries for result k, best score first.
    std::vector<std::vector<int32_t>> child_positions;
    std::vector<std::vector<double>> child_scores;
};

// Xg: grow-set design matrix, row-major (n x p_full). Cg_pool: this
// leaf's child-candidate pool, row-major (n x mu), one thresholded
// bit column per candidate. raw_col_positions/raw_thresholds/
// raw_is_binary describe the eligible raw feature columns to pair up.
inline PairSearchResult raw_pair_root_search_core(
    const double* Xg, size_t n, size_t p_full,
    const int8_t* yg,
    const std::vector<int32_t>& raw_col_positions,
    const std::vector<double>& raw_thresholds,
    const std::vector<uint8_t>& raw_is_binary,
    const uint8_t* Cg_pool, size_t mu,
    const std::vector<int32_t>& ops,
    int64_t max_root_thresholds,
    int64_t min_samples_leaf,
    int64_t max_pairs,
    int64_t top_children_per_candidate,
    uint64_t seed
) {
    PairSearchResult empty;
    const size_t r = raw_col_positions.size();
    if (r < 2) return empty;

    std::vector<std::pair<size_t, size_t>> pairs;
    pairs.reserve(r * (r - 1) / 2);
    for (size_t a = 0; a < r; ++a)
        for (size_t b = a + 1; b < r; ++b) pairs.emplace_back(a, b);
    if (max_pairs > 0 && static_cast<int64_t>(pairs.size()) > max_pairs) {
        std::mt19937_64 rng(seed);
        std::shuffle(pairs.begin(), pairs.end(), rng);
        pairs.resize(static_cast<size_t>(max_pairs));
    }

    struct Result {
        int32_t a, b, op;
        double threshold;
        uint8_t is_binarized;
        double grow_gain;
        std::vector<int32_t> child_positions;
        std::vector<double> child_scores;
    };

    const int64_t top_k = std::max<int64_t>(1, top_children_per_candidate);
    const size_t n_pairs = pairs.size();

#ifdef _OPENMP
    const int n_threads = omp_get_max_threads();
#else
    const int n_threads = 1;
#endif
    std::vector<std::vector<Result>> per_thread_results(static_cast<size_t>(n_threads));

    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        std::vector<Result>& local_results = per_thread_results[static_cast<size_t>(tid)];
        std::vector<double> root_vals(n);
        std::vector<uint8_t> root_bit(n);
        std::vector<double> counts(8 * mu);
        std::vector<double> child_scores(mu);
        std::vector<int32_t> order(mu);

        #pragma omp for schedule(dynamic)
        for (long pi = 0; pi < static_cast<long>(n_pairs); ++pi) {
            const size_t ra = pairs[static_cast<size_t>(pi)].first;
            const size_t rb = pairs[static_cast<size_t>(pi)].second;
            const int32_t col_a = raw_col_positions[ra];
            const int32_t col_b = raw_col_positions[rb];
            const double thr_a = raw_thresholds[ra];
            const double thr_b = raw_thresholds[rb];
            const bool both_binary = raw_is_binary[ra] && raw_is_binary[rb];

            for (int32_t op : ops) {
                for (size_t i = 0; i < n; ++i) {
                    const double xa = Xg[i * p_full + static_cast<size_t>(col_a)];
                    const double xb = Xg[i * p_full + static_cast<size_t>(col_b)];
                    double va = xa, vb = xb;
                    if (both_binary) {
                        va = (xa > thr_a) ? 1.0 : 0.0;
                        vb = (xb > thr_b) ? 1.0 : 0.0;
                    }
                    root_vals[i] = apply_pair_op(va, vb, op);
                }
                std::vector<double> finite_vals;
                finite_vals.reserve(n);
                for (double v : root_vals) if (std::isfinite(v)) finite_vals.push_back(v);
                std::vector<double> thresholds =
                    candidate_thresholds(finite_vals, static_cast<int>(max_root_thresholds));

                for (double threshold : thresholds) {
                    size_t n0 = 0, n1 = 0;
                    for (size_t i = 0; i < n; ++i) {
                        const bool bit = root_vals[i] > threshold;
                        root_bit[i] = bit ? 1 : 0;
                        if (bit) ++n1; else ++n0;
                    }
                    if (std::min(n0, n1) < static_cast<size_t>(2 * min_samples_leaf)) continue;

                    std::fill(counts.begin(), counts.end(), 0.0);
                    for (size_t i = 0; i < n; ++i) {
                        const int root = root_bit[i];
                        const int cls = (yg[i] == 0) ? 0 : 1;
                        const uint8_t* row = Cg_pool + i * mu;
                        double* target0 = counts.data() + static_cast<size_t>((2 * root + 0) * 2 + cls) * mu;
                        double* target1 = counts.data() + static_cast<size_t>((2 * root + 1) * 2 + cls) * mu;
                        for (size_t j = 0; j < mu; ++j) {
                            const double bit = static_cast<double>(row[j]);
                            target1[j] += bit;
                            target0[j] += (1.0 - bit);
                        }
                    }
                    for (size_t j = 0; j < mu; ++j) {
                        double total0 = 0.0, total1 = 0.0;
                        bool under_min = false;
                        double cell_totals[4];
                        for (int c = 0; c < 4; ++c) {
                            const double c0 = counts[static_cast<size_t>(c * 2 + 0) * mu + j];
                            const double c1 = counts[static_cast<size_t>(c * 2 + 1) * mu + j];
                            cell_totals[c] = c0 + c1;
                            if (cell_totals[c] < static_cast<double>(min_samples_leaf)) under_min = true;
                            total0 += c0;
                            total1 += c1;
                        }
                        if (under_min || total0 <= 0.0 || total1 <= 0.0) {
                            child_scores[j] = -std::numeric_limits<double>::infinity();
                            continue;
                        }
                        const double base = entropy_binary_bits(total0, total1);
                        const double ntot = total0 + total1;
                        double cond = 0.0;
                        for (int c = 0; c < 4; ++c) {
                            if (cell_totals[c] <= 0.0) continue;
                            const double c0 = counts[static_cast<size_t>(c * 2 + 0) * mu + j];
                            const double c1 = counts[static_cast<size_t>(c * 2 + 1) * mu + j];
                            cond += (cell_totals[c] / ntot) * entropy_binary_bits(c0, c1);
                        }
                        child_scores[j] = std::max(0.0, base - cond);
                    }

                    for (size_t j = 0; j < mu; ++j) order[j] = static_cast<int32_t>(j);
                    const size_t take = static_cast<size_t>(std::min<int64_t>(top_k, static_cast<int64_t>(mu)));
                    std::partial_sort(
                        order.begin(), order.begin() + static_cast<long>(take), order.end(),
                        [&](int32_t x, int32_t y) { return child_scores[static_cast<size_t>(x)] > child_scores[static_cast<size_t>(y)]; }
                    );
                    if (take == 0 || !std::isfinite(child_scores[static_cast<size_t>(order[0])])) continue;

                    Result res;
                    res.a = col_a; res.b = col_b; res.op = op;
                    res.threshold = threshold;
                    res.is_binarized = both_binary ? 1 : 0;
                    res.grow_gain = child_scores[static_cast<size_t>(order[0])];
                    res.child_positions.reserve(take);
                    res.child_scores.reserve(take);
                    for (size_t k = 0; k < take; ++k) {
                        if (!std::isfinite(child_scores[static_cast<size_t>(order[k])])) break;
                        res.child_positions.push_back(order[k]);
                        res.child_scores.push_back(child_scores[static_cast<size_t>(order[k])]);
                    }
                    local_results.push_back(std::move(res));
                }
            }
        }
    } // end omp parallel

    PairSearchResult out;
    size_t total = 0;
    for (const auto& v : per_thread_results) total += v.size();
    out.a_idx.reserve(total); out.b_idx.reserve(total); out.op.reserve(total);
    out.threshold.reserve(total); out.grow_gain.reserve(total); out.is_binarized.reserve(total);
    out.child_positions.reserve(total); out.child_scores.reserve(total);
    for (auto& v : per_thread_results) {
        for (auto& res : v) {
            out.a_idx.push_back(res.a);
            out.b_idx.push_back(res.b);
            out.op.push_back(res.op);
            out.threshold.push_back(res.threshold);
            out.grow_gain.push_back(res.grow_gain);
            out.is_binarized.push_back(res.is_binarized);
            out.child_positions.push_back(std::move(res.child_positions));
            out.child_scores.push_back(std::move(res.child_scores));
        }
    }
    return out;
}

}  // namespace hugiml::rpte_core

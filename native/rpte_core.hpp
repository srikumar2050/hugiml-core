/**
 * rpte_core.hpp — pure C++ scoring primitives, with no pybind11 or GIL
 * involvement, safe to call from any native context including a
 * GIL-released region.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 *
 * `rpte_scoring.cpp`'s pybind11-facing kernels are thin marshaling
 * wrappers around these functions. `rpte_tree.cpp` (the native
 * tree-growth engine) calls them directly, since it runs its entire
 * search with the GIL released and cannot safely call a function that
 * tries to release the GIL a second time.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace hugiml::rpte_core {

inline double entropy_binary_bits(double n0, double n1) {
    const double n = n0 + n1;
    if (n <= 0.0) return 0.0;
    double h = 0.0;
    if (n0 > 0.0) { const double p = n0 / n; h -= p * std::log2(p); }
    if (n1 > 0.0) { const double p = n1 / n; h -= p * std::log2(p); }
    return h;
}

// One combined value per row from a pair operation: matches
// _adaptive_pair_value's four operations exactly.
enum PairOp : int32_t {
    kOpAbsoluteDifference = 0,
    kOpSignedDifference = 1,
    kOpSum = 2,
    kOpProduct = 3,
};

inline double apply_pair_op(double a, double b, int32_t op) {
    switch (op) {
        case kOpAbsoluteDifference: return std::fabs(a - b);
        case kOpSignedDifference:   return a - b;
        case kOpSum:                return a + b;
        case kOpProduct:            return a * b;
        default: return std::numeric_limits<double>::quiet_NaN();
    }
}

// Candidate thresholds for a value array: all midpoints between
// consecutive sorted unique values when there are few enough,
// otherwise a linspace-selected subset of them.
inline std::vector<double> candidate_thresholds(std::vector<double> values, int max_thresholds) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    const size_t n_vals = values.size();
    if (n_vals < 2) return {};
    std::vector<double> mids(n_vals - 1);
    for (size_t i = 0; i + 1 < n_vals; ++i) mids[i] = (values[i] + values[i + 1]) / 2.0;
    if (static_cast<int>(mids.size()) <= max_thresholds) return mids;

    std::vector<int> pos;
    pos.reserve(static_cast<size_t>(max_thresholds));
    const double hi = static_cast<double>(mids.size() - 1);
    for (int k = 0; k < max_thresholds; ++k) {
        const double frac = (max_thresholds <= 1)
            ? 0.0
            : static_cast<double>(k) / static_cast<double>(max_thresholds - 1);
        pos.push_back(static_cast<int>(std::lround(frac * hi)));
    }
    std::sort(pos.begin(), pos.end());
    pos.erase(std::unique(pos.begin(), pos.end()), pos.end());
    std::vector<double> out;
    out.reserve(pos.size());
    for (int p : pos) out.push_back(mids[static_cast<size_t>(p)]);
    return out;
}

// The midpoint of the two values if `values` is binary, else the
// median; NaN if fewer than 2 distinct finite values are present.
inline double binary_or_median_threshold(const std::vector<double>& values) {
    std::vector<double> vals;
    vals.reserve(values.size());
    for (double v : values) if (std::isfinite(v)) vals.push_back(v);
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    if (vals.size() < 2) return std::numeric_limits<double>::quiet_NaN();
    if (vals.size() == 2) return (vals[0] + vals[1]) / 2.0;
    const size_t mid = vals.size() / 2;
    if (vals.size() % 2 == 1) return vals[mid];
    return (vals[mid - 1] + vals[mid]) / 2.0;
}

// Information gain in bits of a partition (arbitrary integer cell
// ids) against binary y.
inline double partition_ig_bits(const std::vector<int64_t>& codes, const std::vector<int8_t>& y) {
    const size_t n = y.size();
    if (n == 0) return 0.0;
    double total0 = 0.0, total1 = 0.0;
    std::unordered_map<int64_t, std::pair<double, double>> cells;
    cells.reserve(n / 2 + 1);
    for (size_t i = 0; i < n; ++i) {
        auto& cell = cells[codes[i]];
        if (y[i] == 0) { cell.first += 1.0; total0 += 1.0; }
        else           { cell.second += 1.0; total1 += 1.0; }
    }
    if (total0 <= 0.0 || total1 <= 0.0) return 0.0;
    const double base = entropy_binary_bits(total0, total1);
    const double ntot = total0 + total1;
    double cond = 0.0;
    for (const auto& kv : cells) {
        const double ctotal = kv.second.first + kv.second.second;
        if (ctotal <= 0.0) continue;
        cond += (ctotal / ntot) * entropy_binary_bits(kv.second.first, kv.second.second);
    }
    return std::max(0.0, base - cond);
}

// Information gain of the 4-cell {root, child} x {0, 1} partition, for
// every candidate child column at once. child_bits is row-major (n x p).
// A candidate scores -inf if any of its 4 cells falls below min_samples_leaf.
inline std::vector<double> shared_child_scores(
    const std::vector<uint8_t>& root_bit,
    const uint8_t* child_bits, size_t n, size_t p,
    const std::vector<int8_t>& y,
    int64_t min_samples_leaf
) {
    std::vector<double> counts(8 * p, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const int root = root_bit[i] ? 1 : 0;
        const int cls = (y[i] == 0) ? 0 : 1;
        const uint8_t* row = child_bits + i * p;
        double* target0 = counts.data() + ((2 * root + 0) * 2 + cls) * p;
        double* target1 = counts.data() + ((2 * root + 1) * 2 + cls) * p;
        for (size_t j = 0; j < p; ++j) {
            if (row[j]) target1[j] += 1.0; else target0[j] += 1.0;
        }
    }
    std::vector<double> scores(p);
    for (size_t j = 0; j < p; ++j) {
        double total0 = 0.0, total1 = 0.0;
        bool under_min = false;
        double cell_totals[4];
        for (int c = 0; c < 4; ++c) {
            const double c0 = counts[static_cast<size_t>(c * 2 + 0) * p + j];
            const double c1 = counts[static_cast<size_t>(c * 2 + 1) * p + j];
            cell_totals[c] = c0 + c1;
            if (cell_totals[c] < static_cast<double>(min_samples_leaf)) under_min = true;
            total0 += c0;
            total1 += c1;
        }
        if (under_min || total0 <= 0.0 || total1 <= 0.0) {
            scores[j] = -std::numeric_limits<double>::infinity();
            continue;
        }
        const double base = entropy_binary_bits(total0, total1);
        const double ntot = total0 + total1;
        double cond = 0.0;
        for (int c = 0; c < 4; ++c) {
            if (cell_totals[c] <= 0.0) continue;
            const double c0 = counts[static_cast<size_t>(c * 2 + 0) * p + j];
            const double c1 = counts[static_cast<size_t>(c * 2 + 1) * p + j];
            cond += (cell_totals[c] / ntot) * entropy_binary_bits(c0, c1);
        }
        scores[j] = std::max(0.0, base - cond);
    }
    return scores;
}

// Extends an existing n_base_cells-cell partition (base_codes, integer
// in [0, n_base_cells)) with one more shared bit per candidate column,
// generalizing shared_child_scores from a 2-cell base to any size.
inline std::vector<double> shared_extension_scores(
    const std::vector<int64_t>& base_codes,
    int64_t n_base_cells,
    const uint8_t* child_bits, size_t n, size_t p,
    const std::vector<int8_t>& y,
    int64_t min_samples_leaf
) {
    const int64_t n_cells = 2 * n_base_cells;
    std::vector<double> counts(static_cast<size_t>(n_cells) * 2 * p, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const int64_t base = base_codes[i];
        const int cls = (y[i] == 0) ? 0 : 1;
        const uint8_t* row = child_bits + i * p;
        double* target0 = counts.data() + static_cast<size_t>((2 * base + 0) * 2 + cls) * p;
        double* target1 = counts.data() + static_cast<size_t>((2 * base + 1) * 2 + cls) * p;
        for (size_t j = 0; j < p; ++j) {
            if (row[j]) target1[j] += 1.0; else target0[j] += 1.0;
        }
    }
    std::vector<double> scores(p);
    std::vector<double> cell_totals(static_cast<size_t>(n_cells));
    for (size_t j = 0; j < p; ++j) {
        double total0 = 0.0, total1 = 0.0;
        bool under_min = false;
        for (int64_t c = 0; c < n_cells; ++c) {
            const double c0 = counts[static_cast<size_t>(c * 2 + 0) * p + j];
            const double c1 = counts[static_cast<size_t>(c * 2 + 1) * p + j];
            cell_totals[static_cast<size_t>(c)] = c0 + c1;
            if (cell_totals[static_cast<size_t>(c)] < static_cast<double>(min_samples_leaf)) under_min = true;
            total0 += c0;
            total1 += c1;
        }
        if (under_min || total0 <= 0.0 || total1 <= 0.0) {
            scores[j] = -std::numeric_limits<double>::infinity();
            continue;
        }
        const double base_h = entropy_binary_bits(total0, total1);
        const double ntot = total0 + total1;
        double cond = 0.0;
        for (int64_t c = 0; c < n_cells; ++c) {
            const double nc = cell_totals[static_cast<size_t>(c)];
            if (nc <= 0.0) continue;
            const double c0 = counts[static_cast<size_t>(c * 2 + 0) * p + j];
            const double c1 = counts[static_cast<size_t>(c * 2 + 1) * p + j];
            cond += (nc / ntot) * entropy_binary_bits(c0, c1);
        }
        scores[j] = std::max(0.0, base_h - cond);
    }
    return scores;
}

// Information gain of a 2-cell {child=0, child=1} partition, for every
// candidate child column at once, within a single root branch. Must
// not be implemented via shared_child_scores with a dummy empty
// branch: that scorer's min-count check looks at all 4 cells,
// including the deliberately-empty ones, which would mask every
// candidate to -inf regardless of the real data.
inline std::vector<double> single_branch_child_scores(
    const uint8_t* child_bits, size_t n, size_t p,
    const std::vector<int8_t>& y,
    int64_t min_samples_leaf
) {
    std::vector<double> counts(4 * p, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const int cls = (y[i] == 0) ? 0 : 1;
        const uint8_t* row = child_bits + i * p;
        double* target0 = counts.data() + (0 * 2 + cls) * p;
        double* target1 = counts.data() + (1 * 2 + cls) * p;
        for (size_t j = 0; j < p; ++j) {
            if (row[j]) target1[j] += 1.0; else target0[j] += 1.0;
        }
    }
    std::vector<double> scores(p);
    for (size_t j = 0; j < p; ++j) {
        const double c00 = counts[(0 * 2 + 0) * p + j], c01 = counts[(0 * 2 + 1) * p + j];
        const double c10 = counts[(1 * 2 + 0) * p + j], c11 = counts[(1 * 2 + 1) * p + j];
        const double n0 = c00 + c01, n1 = c10 + c11;
        if (n0 < static_cast<double>(min_samples_leaf) || n1 < static_cast<double>(min_samples_leaf)) {
            scores[j] = -std::numeric_limits<double>::infinity();
            continue;
        }
        const double total0 = c00 + c10, total1 = c01 + c11;
        if (total0 <= 0.0 || total1 <= 0.0) {
            scores[j] = -std::numeric_limits<double>::infinity();
            continue;
        }
        const double base = entropy_binary_bits(total0, total1);
        const double ntot = n0 + n1;
        const double cond = (n0 / ntot) * entropy_binary_bits(c00, c01) +
                             (n1 / ntot) * entropy_binary_bits(c10, c11);
        scores[j] = std::max(0.0, base - cond);
    }
    return scores;
}

// Index and score of the best-scoring child column within one root
// branch; (-1, -inf) if the branch is too small or has no candidate.
inline std::pair<int64_t, double> best_branch_child(
    const std::vector<uint8_t>& root_bit,
    const uint8_t* child_bits, size_t n, size_t p,
    const std::vector<int8_t>& y,
    bool branch,
    int64_t min_samples_leaf
) {
    std::vector<int64_t> rows;
    rows.reserve(n);
    bool has0 = false, has1 = false;
    for (size_t i = 0; i < n; ++i) {
        if (static_cast<bool>(root_bit[i]) == branch) {
            rows.push_back(static_cast<int64_t>(i));
            if (y[i] == 0) has0 = true; else has1 = true;
        }
    }
    if (static_cast<int64_t>(rows.size()) < 2 * min_samples_leaf || !(has0 && has1)) {
        return {-1, -std::numeric_limits<double>::infinity()};
    }
    std::vector<uint8_t> sub_child(rows.size() * p);
    std::vector<int8_t> sub_y(rows.size());
    for (size_t r = 0; r < rows.size(); ++r) {
        sub_y[r] = y[static_cast<size_t>(rows[r])];
        const uint8_t* src = child_bits + static_cast<size_t>(rows[r]) * p;
        std::copy(src, src + p, sub_child.data() + r * p);
    }
    std::vector<double> scores = single_branch_child_scores(sub_child.data(), rows.size(), p, sub_y, min_samples_leaf);
    int64_t best_idx = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < p; ++j) {
        if (std::isfinite(scores[j]) && scores[j] > best_score) {
            best_score = scores[j];
            best_idx = static_cast<int64_t>(j);
        }
    }
    return {best_idx, best_score};
}

// Exact per-leaf Newton step sum(r)/(sum(w)+ridge) for each distinct leaf id.
inline std::unordered_map<int64_t, double> newton_leaf_values(
    const std::vector<int64_t>& leaf_ids,
    const std::vector<double>& r,
    const std::vector<double>& w,
    double ridge
) {
    std::unordered_map<int64_t, std::pair<double, double>> sums;
    for (size_t i = 0; i < leaf_ids.size(); ++i) {
        auto& entry = sums[leaf_ids[i]];
        entry.first += r[i];
        entry.second += w[i];
    }
    std::unordered_map<int64_t, double> out;
    out.reserve(sums.size());
    for (const auto& kv : sums) out[kv.first] = kv.second.first / (kv.second.second + ridge);
    return out;
}

// Mean binomial deviance for binary y and predicted probabilities p.
inline double binomial_deviance(const std::vector<double>& y, const std::vector<double>& p, double eps) {
    if (y.empty()) return 0.0;
    double total = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        const double pc = std::min(std::max(p[i], eps), 1.0 - eps);
        total += y[i] * std::log(pc) + (1.0 - y[i]) * std::log(1.0 - pc);
    }
    return -2.0 * total / static_cast<double>(y.size());
}

}  // namespace hugiml::rpte_core

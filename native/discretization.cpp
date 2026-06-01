/**
 * discretization.cpp — Feature discretization implementation
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

#include "discretization.hpp"
#include "math.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace hugiml {

static double adaptive_entropy_log2(const std::vector<int>& y) {
    if (y.empty()) return 0.0;
    std::unordered_map<int, int> counts;
    counts.reserve(y.size());
    for (int v : y) counts[v]++;
    const double n = static_cast<double>(y.size());
    double h = 0.0;
    for (const auto& kv : counts) {
        double p = static_cast<double>(kv.second) / n;
        if (p > 0.0) h -= p * std::log2(p + 1e-12);
    }
    return h;
}

static double adaptive_ig_log2(const std::vector<int>& x_disc,
                               const std::vector<int>& y) {
    if (x_disc.empty() || y.empty()) return 0.0;
    const double total = static_cast<double>(y.size());
    const double base = adaptive_entropy_log2(y);
    std::unordered_map<int, std::vector<int>> groups;
    groups.reserve(32);
    for (size_t i = 0; i < x_disc.size(); ++i) groups[x_disc[i]].push_back(y[i]);
    double weighted = 0.0;
    for (const auto& kv : groups) {
        weighted += static_cast<double>(kv.second.size()) / total * adaptive_entropy_log2(kv.second);
    }
    return std::max(0.0, base - weighted);
}

std::pair<std::vector<int>, std::vector<double>>
kbins_cpp(const std::vector<double>& col, int nb) {
    size_t n = col.size();
    std::vector<double> sc = col;
    std::sort(sc.begin(), sc.end());

    // Compute nb+1 percentile points (numpy-compatible linear interpolation).
    // Must replicate NumPy's FP rounding path exactly:
    //   np.linspace(0,100,nb+1) computes step = 100.0/nb, then q = step*i
    //   np.percentile converts q to index: fidx = q / 100.0 * (n-1)
    double step = 100.0 / nb;
    std::vector<double> edges;
    edges.reserve(nb + 1);
    for (int i = 0; i <= nb; i++) {
        double q    = step * static_cast<double>(i);
        double fidx = q / 100.0 * (n - 1);
        size_t lo   = static_cast<size_t>(fidx);
        size_t hi   = std::min(lo + 1, n - 1);
        double frac = fidx - lo;
        edges.push_back(sc[lo] + frac * (sc[hi] - sc[lo]));
    }

    // Collapse duplicate edges (mirrors np.unique)
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    if (edges.size() < 2) {
        double lo_val = sc.front();
        edges = {lo_val, lo_val + 1e-9};
    }

    // Bin using upper_bound on edges[1:-1] (matches np.searchsorted side='right')
    std::vector<double> inner(edges.begin() + 1, edges.end() - 1);
    std::vector<int> binned(n);
    for (size_t i = 0; i < n; i++) {
        auto it   = std::upper_bound(inner.begin(), inner.end(), col[i]);
        binned[i] = static_cast<int>(it - inner.begin());
    }
    return {binned, edges};
}

int choose_nb_cpp(const std::vector<double>& col,
                  const std::vector<int>& y,
                  int n_cls, int B, int distinct) {
    if (B == -1) {
        double best_ig = 0.0;
        int    best_nb = 2;
        for (int nb = 2; nb <= 20; nb++) {
            int nb_cap = std::max(std::min(distinct - 1, nb), 2);
            auto [dv, _e] = kbins_cpp(col, nb_cap);
            double ig = ig_col_cpp(dv, y, n_cls);
            if (ig > best_ig) { best_ig = ig; best_nb = nb_cap; }
        }
        return best_nb;
    }
    return std::max(std::min(distinct - 1, B), 2);
}

}  // namespace hugiml

namespace hugiml {

// Helper: final quantile edges from an already-sorted finite column.
// This mirrors kbins_cpp edge generation (linear interpolation percentiles),
// but avoids a second column copy/sort during adaptive B selection.
static std::vector<double>
kbins_edges_from_sorted_linear(const std::vector<double>& sc, int nb) {
    const size_t n = sc.size();
    if (n == 0) return {0.0, 1.0};
    double step = 100.0 / nb;
    std::vector<double> edges;
    edges.reserve(nb + 1);
    for (int i = 0; i <= nb; i++) {
        double q    = step * static_cast<double>(i);
        double fidx = q / 100.0 * static_cast<double>(n - 1);
        size_t lo   = static_cast<size_t>(fidx);
        size_t hi   = std::min(lo + 1, n - 1);
        double frac = fidx - static_cast<double>(lo);
        edges.push_back(sc[lo] + frac * (sc[hi] - sc[lo]));
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    if (edges.size() < 2) edges = {sc.front(), sc.front() + 1e-9};
    return edges;
}

// Helper: rounded-index edges from an already-sorted finite column.
// This matches the scoring edge construction used by the previous
// kbins_presorted() implementation, but returns only the unique edge vector.
static std::vector<double>
kbins_score_edges_from_sorted(const std::vector<double>& sc, int nb) {
    const size_t n = sc.size();
    std::vector<double> edges;
    edges.reserve(nb + 1);
    for (int i = 0; i <= nb; i++) {
        double fidx = static_cast<double>(i) / static_cast<double>(nb) * static_cast<double>(n - 1);
        size_t idx  = static_cast<size_t>(std::nearbyint(fidx));
        idx = std::min(idx, n - 1);
        edges.push_back(sc[idx]);
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    if (edges.size() < 2) edges = {sc.front(), sc.front() + 1e-9};
    return edges;
}

static double entropy_log2_from_counts(const std::vector<int>& counts, int total) {
    if (total <= 0) return 0.0;
    const double inv = 1.0 / static_cast<double>(total);
    double h = 0.0;
    for (int c : counts) {
        if (c > 0) {
            double p = static_cast<double>(c) * inv;
            h -= p * std::log2(p + 1e-12);
        }
    }
    return std::max(0.0, h);
}

// Allocation-light IG for a candidate B.  The older implementation materialised
// a length-n binned vector and then built unordered_map<int, vector<int>> groups
// for every candidate.  This version scans the sorted (x, y) column once and
// accumulates dense class histograms by bin, preserving the same upper_bound
// binning semantics for duplicate edges and boundary values.
static double adaptive_ig_sorted_hist_log2(const std::vector<double>& x_sorted,
                                           const std::vector<int>& y_sorted,
                                           int n_cls,
                                           const std::vector<int>& global_counts,
                                           double base_entropy,
                                           int total,
                                           int nb) {
    if (total <= 0 || nb <= 0) return 0.0;
    std::vector<double> edges = kbins_score_edges_from_sorted(x_sorted, nb);
    const int n_bins = static_cast<int>(edges.size()) - 1;
    if (n_bins <= 0) return 0.0;

    std::vector<int> bin_counts(static_cast<size_t>(n_bins) * static_cast<size_t>(n_cls), 0);
    std::vector<int> bin_totals(n_bins, 0);
    int b = 0;
    const int n_inner = static_cast<int>(edges.size()) - 2;
    for (int i = 0; i < total; i++) {
        const double x = x_sorted[i];
        while (b < n_inner && x >= edges[static_cast<size_t>(b) + 1]) ++b;
        int yy = y_sorted[i];
        if (0 <= yy && yy < n_cls) {
            bin_counts[static_cast<size_t>(b) * static_cast<size_t>(n_cls) + static_cast<size_t>(yy)]++;
            bin_totals[b]++;
        }
    }

    double weighted = 0.0;
    std::vector<int> tmp_counts(n_cls);
    for (int bi = 0; bi < n_bins; bi++) {
        int bt = bin_totals[bi];
        if (bt <= 0) continue;
        for (int c = 0; c < n_cls; c++) {
            tmp_counts[c] = bin_counts[static_cast<size_t>(bi) * static_cast<size_t>(n_cls) + static_cast<size_t>(c)];
        }
        weighted += static_cast<double>(bt) / static_cast<double>(total) * entropy_log2_from_counts(tmp_counts, bt);
    }
    (void)global_counts;  // retained in signature for clarity and future reuse.
    return std::max(0.0, base_entropy - weighted);
}

int elbow_stop_nb_cpp(const std::vector<double>& col_raw,
                      const std::vector<int>&    y,
                      int                        n_cls,
                      const std::vector<int>&    candidates,
                      double                     ratio,
                      std::vector<double>&       out_edges,
                      std::vector<double>&       out_ig_scores) {
    // ── 1. Filter non-finite values while preserving labels ─────────────────
    std::vector<std::pair<double, int>> xy;
    xy.reserve(col_raw.size());
    for (size_t i = 0; i < col_raw.size(); i++) {
        if (std::isfinite(col_raw[i])) xy.emplace_back(col_raw[i], y[i]);
    }

    if (static_cast<int>(xy.size()) < 10 || candidates.empty()) {
        int chosen = candidates.empty() ? 2 : candidates[candidates.size() / 2];
        out_ig_scores.assign(candidates.size(), 0.0);
        if (xy.empty()) {
            out_edges = {0.0, 1.0};
        } else {
            std::sort(xy.begin(), xy.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            std::vector<double> sc;
            sc.reserve(xy.size());
            for (const auto& v : xy) sc.push_back(v.first);
            out_edges = kbins_edges_from_sorted_linear(sc, chosen);
        }
        return chosen;
    }

    // ── 2. Sort once and split into dense arrays ─────────────────────────────
    std::sort(xy.begin(), xy.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    const int total = static_cast<int>(xy.size());
    std::vector<double> x_sorted;
    std::vector<int> y_sorted;
    x_sorted.reserve(xy.size());
    y_sorted.reserve(xy.size());
    std::vector<int> global_counts(std::max(n_cls, 1), 0);
    for (const auto& v : xy) {
        x_sorted.push_back(v.first);
        y_sorted.push_back(v.second);
        if (0 <= v.second && v.second < n_cls) global_counts[v.second]++;
    }
    double base_entropy = entropy_log2_from_counts(global_counts, total);

    // ── 3. Elbow-stop over candidates with histogram-only scoring ────────────
    double prev_ig = 0.0;
    int chosen = candidates[0];
    out_ig_scores.clear();
    out_ig_scores.reserve(candidates.size());
    for (int b : candidates) {
        double ig = adaptive_ig_sorted_hist_log2(
            x_sorted, y_sorted, n_cls, global_counts, base_entropy, total, b);
        out_ig_scores.push_back(ig);
        if (prev_ig > 0.0 && (ig - prev_ig) / (prev_ig + 1e-9) < ratio) {
            break;
        }
        chosen = b;
        prev_ig = ig;
    }

    // ── 4. Final stored edges use the original percentile/interpolation path ─
    out_edges = kbins_edges_from_sorted_linear(x_sorted, chosen);
    return chosen;
}

}  // namespace hugiml

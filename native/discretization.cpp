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

namespace hugiml {

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

// Helper: kbins on pre-sorted column (avoids repeated sort).
// sc must already be sorted ascending.
//
// Uses the same boundary formula as Python _information_gain_from_sorted:
//   idx = round(linspace(0, n-1, nb+1))
// (integer index, no interpolation).  This matches np.round behaviour exactly
// so that elbow_stop_nb_cpp produces identical bin assignments to Python _select_b.
static std::pair<std::vector<int>, std::vector<double>>
kbins_presorted(const std::vector<double>& sc,
                const std::vector<double>& col_orig,
                int nb) {
    size_t n = sc.size();
    std::vector<double> edges;
    edges.reserve(nb + 1);
    for (int i = 0; i <= nb; i++) {
        // linspace(0, n-1, nb+1)[i] then round to nearest integer — matches
        // np.round(np.linspace(0, n-1, nb+1)).astype(int) in Python.
        double fidx = static_cast<double>(i) / static_cast<double>(nb) * (n - 1);
        size_t idx  = static_cast<size_t>(std::round(fidx));
        idx = std::min(idx, n - 1);
        edges.push_back(sc[idx]);
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    if (edges.size() < 2) {
        edges = {sc.front(), sc.front() + 1e-9};
    }
    std::vector<double> inner(edges.begin() + 1, edges.end() - 1);
    std::vector<int> binned(col_orig.size());
    for (size_t i = 0; i < col_orig.size(); i++) {
        auto it   = std::upper_bound(inner.begin(), inner.end(), col_orig[i]);
        binned[i] = static_cast<int>(it - inner.begin());
    }
    return {binned, edges};
}

int elbow_stop_nb_cpp(const std::vector<double>& col_raw,
                      const std::vector<int>&    y,
                      int                        n_cls,
                      const std::vector<int>&    candidates,
                      double                     ratio,
                      std::vector<double>&       out_edges,
                      std::vector<double>&       out_ig_scores) {
    // ── 1. Filter non-finite values ──────────────────────────────────────────
    std::vector<double> col_f;
    std::vector<int>    y_f;
    col_f.reserve(col_raw.size());
    y_f.reserve(col_raw.size());
    for (size_t i = 0; i < col_raw.size(); i++) {
        if (std::isfinite(col_raw[i])) {
            col_f.push_back(col_raw[i]);
            y_f.push_back(y[i]);
        }
    }

    if (static_cast<int>(col_f.size()) < 10 || candidates.empty()) {
        int chosen = candidates.empty() ? 2 : candidates[candidates.size() / 2];
        out_ig_scores.assign(candidates.size(), 0.0);
        // Preserve Python _quantile_edges semantics for the final stored edges:
        // finite-only np.percentile(..., linspace(0, 100, chosen+1)), unique,
        // then [min, max + 1e-9] fallback for collapsed edges.
        if (col_f.empty()) {
            out_edges = {0.0, 1.0};
        } else {
            auto [dummy_binned, edges] = kbins_cpp(col_f, chosen);
            out_edges = edges;
        }
        return chosen;
    }

    // ── 2. Sort ONCE — reuse across all candidate evaluations ────────────────
    // Mirrors Python _select_b which sorts once and calls
    // _information_gain_from_sorted(x_sorted, y_sorted, b) per candidate.
    std::vector<double> sc = col_f;
    std::sort(sc.begin(), sc.end());

    // ── 3. Elbow-stop over candidates using pre-sorted column ─────────────────
    double prev_ig = 0.0;
    int    chosen  = candidates[0];
    out_ig_scores.clear();
    out_ig_scores.reserve(candidates.size());
    for (int b : candidates) {
        // Match Python _select_b exactly: candidate b is passed directly to
        // _information_gain_from_sorted.  That function may collapse duplicate
        // edges internally; it does not cap b by distinct-1 before scoring.
        auto [binned, score_edges] = kbins_presorted(sc, col_f, b);
        double ig = ig_col_cpp(binned, y_f, n_cls);
        out_ig_scores.push_back(ig);

        if (prev_ig > 0.0 &&
            (ig - prev_ig) / (prev_ig + 1e-9) < ratio) {
            break;
        }
        chosen  = b;
        prev_ig = ig;
    }

    // Preserve Python _apply_adaptive_binning semantics: after _select_b chooses
    // B, final stored edges come from _quantile_edges(col, chosen), not from
    // the rounded-index edges used only for fast IG scoring.
    auto [dummy_binned, final_edges] = kbins_cpp(col_f, chosen);
    out_edges = final_edges;
    return chosen;
}

}  // namespace hugiml

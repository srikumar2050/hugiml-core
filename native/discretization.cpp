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

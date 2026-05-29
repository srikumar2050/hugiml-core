/**
 * math.cpp — Mathematical utilities implementation
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

#include "math.hpp"

#include <algorithm>
#include <unordered_map>

namespace hugiml {

double entropy_vec(const std::vector<int>& y, int n_cls) {
    if (y.empty()) return 0.0;
    // Java THUIsl's UtilityList.getEntropy uses natural-log Shannon entropy
    // without normalising by log(num_classes).  Keep the miner on that scale
    int max_y = 0;
    for (int v : y)
        if (v > max_y) max_y = v;
    std::vector<double> cnts(max_y + 1, 0.0);
    for (int v : y)
        if (v >= 0) cnts[v] += 1.0;
    double total = static_cast<double>(y.size());
    double h     = 0.0;
    for (double c : cnts) {
        if (c > 0.0) {
            double p = c / total;
            h -= p * std::log(p);
        }
    }
    return std::max(h, 0.0);
}

double ig_col_cpp(const std::vector<int>& x_disc,
                  const std::vector<int>& y, int n_cls) {
    double base  = entropy_vec(y, n_cls);
    double total = static_cast<double>(y.size());
    std::unordered_map<int, std::vector<int>> groups;
    groups.reserve(n_cls * 2);
    for (size_t i = 0; i < x_disc.size(); i++)
        groups[x_disc[i]].push_back(y[i]);
    double ce = 0.0;
    for (auto& kv : groups)
        ce += static_cast<double>(kv.second.size()) / total
              * entropy_vec(kv.second, n_cls);
    return base - ce;
}

double pearson_cpp(const std::vector<double>& x,
                   const std::vector<double>& y) {
    size_t n = x.size();
    if (n == 0) return 0.0;
    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n;
    my /= n;
    double sx = 0.0, sy = 0.0, sxy = 0.0;
    for (size_t i = 0; i < n; i++) {
        double dx = x[i] - mx, dy = y[i] - my;
        sx  += dx * dx;
        sy  += dy * dy;
        sxy += dx * dy;
    }
    if (sx == 0.0 || sy == 0.0) return 0.0;
    double r = sxy / std::sqrt(sx * sy);
    return std::isfinite(r) ? r : 0.0;
}

double nmi_binary_cpp(const std::vector<int>& x_bin,
                      const std::vector<int>& y, int n_cls) {
    size_t n = x_bin.size();
    std::unordered_map<int, int> x_counts;
    for (int v : x_bin) x_counts[v]++;
    if (x_counts.size() <= 1) return 0.0;

    double hx = 0.0;
    for (auto& kv : x_counts) {
        double p = static_cast<double>(kv.second) / n;
        if (p > 0.0) hx -= p * safe_log(p);
    }

    int max_y = 0;
    for (int v : y)
        if (v > max_y) max_y = v;
    int arr_sz = max_y + 1;
    std::vector<double> yc(arr_sz, 0.0);
    for (int v : y)
        if (v >= 0) yc[v] += 1.0;
    double hy = 0.0;
    for (double c : yc) {
        double p = c / n;
        if (p > 0.0) hy -= p * safe_log(p);
    }
    if (hx == 0.0 || hy == 0.0) return 0.0;

    double mi = 0.0;
    for (auto& kv : x_counts) {
        int xv   = kv.first;
        int nx_v = kv.second;
        std::vector<double> xyc(arr_sz, 0.0);
        for (size_t i = 0; i < n; i++)
            if (x_bin[i] == xv && y[i] >= 0) xyc[y[i]] += 1.0;
        for (int c = 0; c < arr_sz; c++) {
            double nxy = xyc[c];
            if (nxy > 0.0) {
                double pxy = nxy / n;
                double px  = static_cast<double>(nx_v) / n;
                double py_c = yc[c] / n;
                mi += pxy * safe_log(pxy / (px * py_c));
            }
        }
    }
    return std::max(0.0, std::min(1.0, safe_div(mi, safe_sqrt(hx * hy))));
}

}  // namespace hugiml

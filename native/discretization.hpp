/**
 * discretization.hpp — Feature discretization
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

#include <utility>
#include <vector>

namespace hugiml {

/// Quantile-based discretisation into nb bins (_kbins).
/// Returns (binned 0-based indices, sorted unique edges).
std::pair<std::vector<int>, std::vector<double>>
kbins_cpp(const std::vector<double>& col, int nb);

/// Select the number of bins for one column.
/// B == -1 → supervised auto-selection (maximises IG over [2, 20]).
int choose_nb_cpp(const std::vector<double>& col,
                  const std::vector<int>& y,
                  int n_cls, int B, int distinct);

// ── Elbow-stopping adaptive B selection ──────────────────────────────────────
// Mirrors Python _select_b in hugiml._binning exactly.
//
// Algorithm (sort-once):
//   1. Filter non-finite values from col and y (same as Python finite= mask).
//   2. Sort filtered col once; reuse for all candidate evaluations.
//   3. For each candidate B in ascending order:
//        a. Call kbins_cpp to get quantile edges and bin codes.
//        b. Compute ig_col_cpp (natural log — scale-invariant for elbow test).
//        c. Elbow-stop: if (ig - ig_prev) / (ig_prev + eps) < ratio → break.
//   4. Return chosen B, the full edges for chosen B, and the IG score grid.
//
// The elbow-stopping criterion is log-scale invariant (ratio = Δig/ig_prev),
// so using natural-log ig_col_cpp gives the same chosen_B as Python's log2
// _information_gain_from_sorted.  The returned ig_scores may differ from the
// Python ig_scores_ by a constant factor (ln 2 ≈ 0.693) but that only affects
// the ig_heatmap() diagnostic — not any correctness-critical downstream path.
//
// Parameters
// ──────────
//   col_raw       — raw float64 column values (may contain NaN/Inf)
//   y             — integer class labels aligned with col_raw
//   n_cls         — number of distinct classes
//   candidates    — sorted list of candidate B values (each >= 2)
//   ratio         — elbow-stop threshold (min_marginal_gain_ratio)
//   out_edges     — populated with the quantile edges for chosen B
//   out_ig_scores — populated with ig score per candidate evaluated
//
// Returns chosen B.
int elbow_stop_nb_cpp(const std::vector<double>& col_raw,
                      const std::vector<int>&    y,
                      int                        n_cls,
                      const std::vector<int>&    candidates,
                      double                     ratio,
                      std::vector<double>&       out_edges,
                      std::vector<double>&       out_ig_scores);

}  // namespace hugiml

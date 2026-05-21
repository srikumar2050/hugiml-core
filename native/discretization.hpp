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

}  // namespace hugiml

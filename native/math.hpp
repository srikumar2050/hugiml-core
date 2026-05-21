/**
 * math.hpp — Mathematical utilities for HUG-IML
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
 *
 * Provides normalised entropy, information gain, Pearson correlation,
 * and normalised mutual information, along with input-validation and
 * numerical-safety helpers used across the C++ extension.
 */

#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace hugiml {

// ── Input validation helpers ─────────────────────────────────────────────────

inline void validate_positive(int val, const char* name) {
    if (val <= 0)
        throw std::invalid_argument(
            std::string(name) + " must be positive, got " + std::to_string(val));
}

inline void validate_not_empty(size_t size, const char* name) {
    if (size == 0)
        throw std::invalid_argument(std::string(name) + " cannot be empty");
}

inline void validate_same_size(size_t s1, size_t s2,
                                const char* n1, const char* n2) {
    if (s1 != s2)
        throw std::invalid_argument(
            std::string(n1) + " and " + n2 + " size mismatch: " +
            std::to_string(s1) + " vs " + std::to_string(s2));
}

// ── Numerical safety ─────────────────────────────────────────────────────────

inline double safe_div(double num, double den, double fallback = 0.0) {
    constexpr double EPS = 1e-15;
    if (std::abs(den) < EPS) return fallback;
    double r = num / den;
    return std::isfinite(r) ? r : fallback;
}

inline double safe_sqrt(double x) {
    return (x < 0.0) ? 0.0 : std::sqrt(x);
}

inline double safe_log(double x) {
    constexpr double EPS = 1e-15;
    return (x <= EPS) ? 0.0 : std::log(x);
}

// ── Statistical functions ────────────────────────────────────────────────────

/// Normalised Shannon entropy of integer class labels, in [0, 1].
double entropy_vec(const std::vector<int>& y, int n_cls);

/// Information gain of a discretised column w.r.t. class labels.
double ig_col_cpp(const std::vector<int>& x_disc,
                  const std::vector<int>& y, int n_cls);

/// Pearson correlation coefficient.  Returns 0 for zero-variance inputs.
double pearson_cpp(const std::vector<double>& x,
                   const std::vector<double>& y);

/// Normalised Mutual Information between a binarised (0/1) column and y.
double nmi_binary_cpp(const std::vector<int>& x_bin,
                      const std::vector<int>& y, int n_cls);

}  // namespace hugiml

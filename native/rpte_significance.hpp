/**
 * rpte_significance.hpp — statistical calibration for split acceptance.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 *
 * An information-gain statistic IG = H(y) - H(y|partition), computed
 * over a K-cell partition of a binary target, is (up to a constant)
 * the same quantity a G-test of independence in a K x 2 contingency
 * table is built from: G^2 = 2 n IG_nats is asymptotically
 * chi-squared(df = K-1) under the null hypothesis that the partition
 * carries no information about y (Wilks' theorem). This gives IG a
 * reference distribution instead of treating "IG > some constant" as
 * self-justifying.
 *
 * The lookahead search compares many candidates at each leaf and
 * keeps the best. Testing the best of many statistics against a
 * single-test critical value is the standard multiple-comparisons
 * mistake: even under a global null, the best of m candidates clears
 * any fixed bar far more often than the nominal test level suggests.
 * `bonferroni_critical_ig` corrects for this by dividing the
 * significance budget by the number of candidates actually compared
 * at that decision point.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "rpte_chi2.hpp"

namespace hugiml::rpte {

// G^2 = 2 n IG_nats: the likelihood-ratio statistic for an
// information-gain-in-bits value computed over n rows.
inline double g2_statistic(double ig_bits, int64_t n) {
    return 2.0 * static_cast<double>(std::max<int64_t>(0, n)) * ig_bits * std::log(2.0);
}

// Asymptotic p-value for a single partition's information gain,
// against the chi-squared(df) null reference distribution. Not
// multiplicity-corrected; for diagnostics, not acceptance decisions.
inline double g2_p_value(double ig_bits, int64_t n, double df) {
    return chi2::sf(g2_statistic(ig_bits, n), df);
}

// The information-gain-in-bits threshold a partition's G^2 statistic
// must clear to be significant at family-wise level `alpha` after
// Bonferroni-correcting for `n_tests` simultaneous comparisons, given
// `n` rows and `df` degrees of freedom.
inline double bonferroni_critical_ig(double alpha, double df, int64_t n_tests, int64_t n) {
    n_tests = std::max<int64_t>(1, n_tests);
    const double corrected_alpha =
        std::min(0.999999, std::max(1e-300, alpha / static_cast<double>(n_tests)));
    const double critical_g2 = chi2::isf(corrected_alpha, df);
    const double denom = 2.0 * static_cast<double>(std::max<int64_t>(1, n)) * std::log(2.0);
    return critical_g2 / denom;
}

}  // namespace hugiml::rpte

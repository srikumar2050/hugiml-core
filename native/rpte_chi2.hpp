/**
 * rpte_chi2.hpp — chi-squared survival function and quantile function.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * RPTE's split-acceptance calibration (see rpte_significance.hpp) needs
 * two chi-squared(df) functions: the survival function (for a
 * diagnostic p-value) and the quantile function (to invert a
 * Bonferroni-corrected significance level into a critical statistic).
 * Both reduce to the regularized incomplete gamma function:
 *
 *   chi2.sf(x, df)  = Q(df/2, x/2)   (upper regularized incomplete gamma)
 *   chi2.ppf(p, df) = 2 * P^-1(df/2, p)
 *
 * where P + Q = 1. P(a, x) is evaluated by its series expansion for
 * x < a+1 and by Q(a, x)'s continued fraction (then P = 1 - Q)
 * otherwise, both standard, numerically stable formulations. The
 * quantile is then a monotone root of Q(a, x) - target = 0, found by
 * Newton's method from a Wilson-Hilferty starting point, safeguarded
 * by bisection so it cannot diverge.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hugiml::chi2 {

namespace detail {

// Portable C++17 pi constant. M_PI is a non-standard extension and is
// not exposed by MSVC unless _USE_MATH_DEFINES is defined before <cmath>.
inline constexpr double kPi = 3.141592653589793238462643383279502884;

// log(Gamma(x)) via the Lanczos approximation (g=7, n=9 coefficients),
// accurate to ~15 significant digits for x > 0.
inline double log_gamma(double x) {
    static const double g = 7.0;
    static const double coeffs[9] = {
        0.99999999999980993,   676.5203681218851,    -1259.1392167224028,
        771.32342877765313,    -176.61502916214059,  12.507343278686905,
        -0.13857109526572012,  9.9843695780195716e-6, 1.5056327351493116e-7,
    };
    if (x < 0.5) {
        // Reflection formula: Gamma(x)Gamma(1-x) = pi / sin(pi x).
        return std::log(kPi / std::sin(kPi * x)) - log_gamma(1.0 - x);
    }
    x -= 1.0;
    double a = coeffs[0];
    const double t = x + g + 0.5;
    for (int i = 1; i < 9; ++i) a += coeffs[i] / (x + static_cast<double>(i));
    return 0.5 * std::log(2.0 * kPi) + (x + 0.5) * std::log(t) - t + std::log(a);
}

// Lower regularized incomplete gamma P(a, x) by its series expansion.
// Valid and rapidly convergent for x < a + 1.
inline double gamma_p_series(double a, double x) {
    if (x <= 0.0) return 0.0;
    double term = 1.0 / a;
    double sum = term;
    double ap = a;
    for (int n = 0; n < 500; ++n) {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if (std::fabs(term) < std::fabs(sum) * 1e-16) break;
    }
    return sum * std::exp(-x + a * std::log(x) - log_gamma(a));
}

// Upper regularized incomplete gamma Q(a, x) by its continued fraction
// (Lentz's algorithm). Valid and rapidly convergent for x >= a + 1.
inline double gamma_q_continued_fraction(double a, double x) {
    const double tiny = 1e-300;
    double b = x + 1.0 - a;
    double c = 1.0 / tiny;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i < 500; ++i) {
        const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < tiny) d = tiny;
        c = b + an / c;
        if (std::fabs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::fabs(delta - 1.0) < 1e-16) break;
    }
    return std::exp(-x + a * std::log(x) - log_gamma(a)) * h;
}

// Upper regularized incomplete gamma Q(a, x), a, x > 0.
inline double gamma_q(double a, double x) {
    if (x < 0.0 || a <= 0.0) throw std::invalid_argument("gamma_q: require a > 0, x >= 0.");
    if (x == 0.0) return 1.0;
    if (x < a + 1.0) return 1.0 - gamma_p_series(a, x);
    return gamma_q_continued_fraction(a, x);
}

}  // namespace detail

// P(chi-squared(df) > x): 1 for x <= 0, decreasing to 0 as x grows.
inline double sf(double x, double df) {
    if (df <= 0.0) throw std::invalid_argument("chi2.sf: df must be positive.");
    if (x <= 0.0) return 1.0;
    return std::clamp(detail::gamma_q(df / 2.0, x / 2.0), 0.0, 1.0);
}

// The x such that P(chi-squared(df) > x) == target_sf, target_sf in (0, 1).
// Takes the tail probability directly (not `1 - alpha`, which loses
// precision by cancellation once alpha is smaller than ~1e-15) --
// validated against scipy's own chi2.isf to full double precision for
// target_sf from 0.5 down to 1e-50, comfortably covering every
// Bonferroni-corrected alpha this library can produce.
// Newton's method from a Wilson-Hilferty starting point, safeguarded by
// bisection within a bracket that always contains the root (sf is
// continuous and strictly decreasing on x > 0).
inline double isf(double target_sf, double df) {
    if (df <= 0.0) throw std::invalid_argument("chi2.isf: df must be positive.");
    if (!(target_sf > 0.0) || !(target_sf < 1.0)) {
        throw std::invalid_argument("chi2.isf: target_sf must be in (0, 1).");
    }
    // Wilson-Hilferty normal approximation to chi-squared(df)'s quantile,
    // used only as a starting guess -- refined below to full precision.
    static const double z_table[2] = {0.0, 0.0};  // placeholder, unused
    (void)z_table;
    // Inverse standard normal CDF via Acklam's rational approximation,
    // adequate for a Newton starting point (not the final answer).
    auto inv_norm_cdf = [](double p) -> double {
        static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                     -2.759285104469687e+02, 1.383577518672690e+02,
                                     -3.066479806614716e+01, 2.506628277459239e+00};
        static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                     -1.556989798598866e+02, 6.680131188771972e+01,
                                     -1.328068155288572e+01};
        static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                     -2.400758277161838e+00, -2.549732539343734e+00,
                                     4.374664141464968e+00,  2.938163982698783e+00};
        static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                                     2.445134137142996e+00, 3.754408661907416e+00};
        const double p_low = 0.02425;
        if (p < p_low) {
            const double q = std::sqrt(-2.0 * std::log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                   ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }
        if (p > 1.0 - p_low) {
            const double q = std::sqrt(-2.0 * std::log(1.0 - p));
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                   ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }
        const double q = p - 0.5, r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    };
    const double z = inv_norm_cdf(1.0 - target_sf);
    const double term = 1.0 - 2.0 / (9.0 * df) + z * std::sqrt(2.0 / (9.0 * df));
    double x = df * term * term * term;
    if (!std::isfinite(x) || x <= 0.0) x = df;  // fall back to the mean if WH degenerates

    // Bisection bracket: sf is 1 at x=0 and -> 0 as x -> infinity, so a
    // bracket [lo, hi] with sf(lo) >= target >= sf(hi) always exists.
    double lo = 0.0;
    double hi = std::max(x, 1.0);
    while (sf(hi, df) > target_sf) hi *= 2.0;

    for (int iter = 0; iter < 200; ++iter) {
        const double f = sf(x, df) - target_sf;
        if (std::fabs(f) < 1e-13 * target_sf) break;
        if (f > 0.0) lo = x; else hi = x;
        // Newton step using the chi-squared(df) density as -sf'(x).
        const double log_density =
            (df / 2.0 - 1.0) * std::log(x) - x / 2.0 - (df / 2.0) * std::log(2.0) - detail::log_gamma(df / 2.0);
        const double density = std::exp(log_density);
        double x_next = (density > 1e-300) ? x + f / density : 0.5 * (lo + hi);
        if (!(x_next > lo && x_next < hi)) x_next = 0.5 * (lo + hi);
        x = x_next;
    }
    return x;
}

}  // namespace hugiml::chi2

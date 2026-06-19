/**
 * augmented_pair.cpp — Native helpers for L>1 augmented pair transforms.
 *
 * These functions are built from native/ into the optional
 * _hugiml_core extension.
 */

#include "pybind_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace {

inline bool finite_double(double v) { return std::isfinite(v); }

double entropy_from_counts(const std::vector<int64_t>& counts, int64_t total) {
    if (total <= 0) return 0.0;
    double h = 0.0;
    const double inv_total = 1.0 / static_cast<double>(total);
    for (int64_t c : counts) {
        if (c <= 0) continue;
        const double p = static_cast<double>(c) * inv_total;
        h -= p * std::log2(p + 1e-15);
    }
    return h;
}

double entropy_labels(const int64_t* y, py::ssize_t n, int64_t n_classes) {
    std::vector<int64_t> counts(static_cast<size_t>(n_classes), 0);
    for (py::ssize_t i = 0; i < n; ++i) {
        const int64_t cls = y[i];
        if (cls >= 0 && cls < n_classes) counts[static_cast<size_t>(cls)]++;
    }
    return entropy_from_counts(counts, static_cast<int64_t>(n));
}

double discrete_ig(
    const std::vector<int32_t>& z,
    const int64_t* y,
    py::ssize_t n,
    int64_t n_classes,
    double base_entropy
) {
    if (n <= 0) return 0.0;
    int32_t max_bin = 0;
    for (int32_t v : z) {
        if (v > max_bin) max_bin = v;
    }
    const size_t nbins = static_cast<size_t>(max_bin + 1);
    std::vector<std::vector<int64_t>> counts(
        nbins,
        std::vector<int64_t>(static_cast<size_t>(n_classes), 0)
    );
    std::vector<int64_t> totals(nbins, 0);
    for (py::ssize_t i = 0; i < n; ++i) {
        int32_t b = z[static_cast<size_t>(i)];
        if (b < 0) b = 0;
        if (static_cast<size_t>(b) >= nbins) continue;
        const int64_t cls = y[i];
        if (cls >= 0 && cls < n_classes) {
            counts[static_cast<size_t>(b)][static_cast<size_t>(cls)]++;
            totals[static_cast<size_t>(b)]++;
        }
    }
    double cond = 0.0;
    for (size_t b = 0; b < nbins; ++b) {
        if (totals[b] <= 0) continue;
        cond += (static_cast<double>(totals[b]) / static_cast<double>(n))
            * entropy_from_counts(counts[b], totals[b]);
    }
    return base_entropy - cond;
}

std::vector<double> quantile_edges(const std::vector<double>& sorted_values, int q) {
    std::vector<double> edges;
    const size_t n = sorted_values.size();
    if (n == 0 || q < 2) return edges;
    edges.reserve(static_cast<size_t>(q + 1));
    for (int k = 0; k <= q; ++k) {
        const double pos = (static_cast<double>(n - 1) * static_cast<double>(k)) / static_cast<double>(q);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = static_cast<size_t>(std::ceil(pos));
        const double frac = pos - static_cast<double>(lo);
        const double val = sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac;
        if (edges.empty() || val != edges.back()) edges.push_back(val);
    }
    return edges;
}

std::pair<double, py::object> adaptive_numeric_ig(
    const std::vector<double>& values,
    const int64_t* y,
    py::ssize_t n,
    int64_t n_classes,
    double base_entropy,
    int max_bins
) {
    std::vector<double> finite_vals;
    finite_vals.reserve(values.size());
    for (double v : values) {
        if (finite_double(v)) finite_vals.push_back(v);
    }
    if (finite_vals.size() < 3) return {0.0, py::none()};
    std::sort(finite_vals.begin(), finite_vals.end());
    finite_vals.erase(std::unique(finite_vals.begin(), finite_vals.end()), finite_vals.end());
    if (finite_vals.size() < 2) return {0.0, py::none()};

    const int max_q = std::min(max_bins, static_cast<int>(finite_vals.size()));
    double best_score = 0.0;
    std::vector<double> best_edges;
    std::vector<int32_t> z(static_cast<size_t>(n), 0);
    for (int q = 2; q <= max_q; ++q) {
        std::vector<double> edges = quantile_edges(finite_vals, q);
        if (edges.size() < 3) continue;
        for (py::ssize_t i = 0; i < n; ++i) {
            const double v = values[static_cast<size_t>(i)];
            if (!finite_double(v)) {
                z[static_cast<size_t>(i)] = 0;
            } else {
                auto it = std::upper_bound(edges.begin() + 1, edges.end() - 1, v);
                z[static_cast<size_t>(i)] = static_cast<int32_t>(it - (edges.begin() + 1));
            }
        }
        const double score = discrete_ig(z, y, n, n_classes, base_entropy);
        if (score > best_score) {
            best_score = score;
            best_edges = std::move(edges);
        }
    }
    if (best_edges.empty()) return {best_score, py::none()};
    return {best_score, py::cast(best_edges)};
}

py::list score_pair_candidates(
    py::array_t<double, py::array::c_style | py::array::forcecast> X_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> y_arr,
    const std::vector<std::string>& col_names
) {
    auto X = X_arr.unchecked<2>();
    auto y = y_arr.unchecked<1>();
    const py::ssize_t n = X.shape(0);
    const py::ssize_t p = X.shape(1);
    if (y.shape(0) != n) throw std::invalid_argument("X and y row counts do not match.");
    if (static_cast<py::ssize_t>(col_names.size()) != p) throw std::invalid_argument("col_names length does not match X columns.");

    int64_t n_classes = 0;
    for (py::ssize_t i = 0; i < n; ++i) {
        if (y(i) >= n_classes) n_classes = y(i) + 1;
    }
    if (n_classes <= 0) n_classes = 1;

    py::list out;
    std::vector<double> vals;
    std::vector<int64_t> y_eligible;
    vals.reserve(static_cast<size_t>(n));
    y_eligible.reserve(static_cast<size_t>(n));
    for (py::ssize_t a = 0; a < p; ++a) {
        for (py::ssize_t b = a + 1; b < p; ++b) {
            for (int op = 0; op < 4; ++op) {
                vals.clear();
                y_eligible.clear();
                double sum_raw = 0.0;
                for (py::ssize_t i = 0; i < n; ++i) {
                    const double xl = X(i, a);
                    const double xr = X(i, b);
                    if (!finite_double(xl) || !finite_double(xr)) continue;
                    double raw;
                    if (op == 0) {
                        raw = xl * xr;
                    } else if (op == 1) {
                        raw = std::fabs(xl - xr);
                    } else if (op == 2) {
                        raw = xl + xr;
                    } else {
                        raw = xl - xr;
                    }
                    if (!finite_double(raw)) continue;
                    vals.push_back(raw);
                    y_eligible.push_back(y(i));
                    sum_raw += raw;
                }
                const py::ssize_t m = static_cast<py::ssize_t>(vals.size());
                if (m < 3) continue;
                const double reference_raw_value = sum_raw / static_cast<double>(m);
                const double base_entropy = entropy_labels(y_eligible.data(), m, n_classes);
                auto scored = adaptive_numeric_ig(vals, y_eligible.data(), m, n_classes, base_entropy, 12);
                std::string operation;
                std::string prefix;
                std::string formula;
                if (op == 0) {
                    operation = "product";
                    prefix = "augmented_pair_prod__";
                    formula = col_names[static_cast<size_t>(a)] + " * " + col_names[static_cast<size_t>(b)];
                } else if (op == 1) {
                    operation = "absolute_difference";
                    prefix = "augmented_pair_absdiff__";
                    formula = "abs(" + col_names[static_cast<size_t>(a)] + " - " + col_names[static_cast<size_t>(b)] + ")";
                } else if (op == 2) {
                    operation = "sum";
                    prefix = "augmented_pair_sum__";
                    formula = col_names[static_cast<size_t>(a)] + " + " + col_names[static_cast<size_t>(b)];
                } else {
                    operation = "signed_difference";
                    prefix = "augmented_pair_diff__";
                    formula = col_names[static_cast<size_t>(a)] + " - " + col_names[static_cast<size_t>(b)];
                }
                py::dict d;
                d["name"] = prefix + col_names[static_cast<size_t>(a)] + "__" + col_names[static_cast<size_t>(b)];
                d["operation"] = operation;
                d["inputs"] = py::make_tuple(col_names[static_cast<size_t>(a)], col_names[static_cast<size_t>(b)]);
                d["formula"] = formula;
                d["transform_ig"] = scored.first;
                d["transform_bin_edges"] = scored.second;
                d["eligible_count"] = static_cast<int64_t>(m);
                d["eligible_rate"] = static_cast<double>(m) / static_cast<double>(std::max<py::ssize_t>(n, 1));
                d["missing_pair_rate"] = 1.0 - (static_cast<double>(m) / static_cast<double>(std::max<py::ssize_t>(n, 1)));
                d["reference_raw_value"] = reference_raw_value;
                d["pair_missing_policy"] = "reference_value_for_unavailable_pair";
                out.append(std::move(d));
            }
        }
    }
    return out;
}

py::array_t<float> transform_pair_features(
    py::array_t<double, py::array::c_style | py::array::forcecast> X_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> left_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> right_arr,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ops_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reference_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> mean_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> scale_arr
) {
    auto X = X_arr.unchecked<2>();
    auto left = left_arr.unchecked<1>();
    auto right = right_arr.unchecked<1>();
    auto ops = ops_arr.unchecked<1>();
    auto refs = reference_arr.unchecked<1>();
    auto means = mean_arr.unchecked<1>();
    auto scales = scale_arr.unchecked<1>();
    const py::ssize_t n = X.shape(0);
    const py::ssize_t p = X.shape(1);
    const py::ssize_t k = left.shape(0);
    if (right.shape(0) != k || ops.shape(0) != k || refs.shape(0) != k || means.shape(0) != k || scales.shape(0) != k) {
        throw std::invalid_argument("Array lengths do not match.");
    }
    py::array_t<float> out({n, k});
    auto Z = out.mutable_unchecked<2>();
    for (py::ssize_t t = 0; t < k; ++t) {
        const int64_t a = left(t);
        const int64_t b = right(t);
        if (a < 0 || b < 0 || a >= p || b >= p) throw std::out_of_range("Pair index out of bounds.");
        double sc = scales(t);
        if (!finite_double(sc) || sc == 0.0) sc = 1.0;
        double ref = refs(t);
        if (!finite_double(ref)) ref = means(t);
        if (!finite_double(ref)) ref = 0.0;
        for (py::ssize_t i = 0; i < n; ++i) {
            const double xl = X(i, a);
            const double xr = X(i, b);
            double raw = ref;
            if (finite_double(xl) && finite_double(xr)) {
                const int8_t op = ops(t);
                if (op == 0) {
                    raw = xl * xr;
                } else if (op == 1) {
                    raw = std::fabs(xl - xr);
                } else if (op == 2) {
                    raw = xl + xr;
                } else if (op == 3) {
                    raw = xl - xr;
                } else {
                    throw std::invalid_argument("Unknown augmented pair op code.");
                }
                if (!finite_double(raw)) raw = ref;
            }
            Z(i, t) = static_cast<float>((raw - means(t)) / sc);
        }
    }
    return out;
}


std::vector<int32_t> continuous_to_quantile_codes_fixed(const std::vector<double>& values, int max_bins) {
    const size_t n = values.size();
    std::vector<int32_t> codes(n, -1);
    std::vector<std::pair<double, size_t>> finite;
    finite.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (finite_double(values[i])) finite.emplace_back(values[i], i);
    }
    if (finite.empty()) return codes;
    std::sort(finite.begin(), finite.end(), [](const auto& a, const auto& b) {
        if (a.first == b.first) return a.second < b.second;
        return a.first < b.first;
    });
    std::vector<double> sorted_vals;
    sorted_vals.reserve(finite.size());
    std::vector<double> uniq;
    uniq.reserve(finite.size());
    for (const auto& item : finite) {
        sorted_vals.push_back(item.first);
        if (uniq.empty() || item.first != uniq.back()) uniq.push_back(item.first);
    }
    if (static_cast<int>(uniq.size()) <= std::max(1, max_bins)) {
        for (const auto& item : finite) {
            auto it = std::lower_bound(uniq.begin(), uniq.end(), item.first);
            codes[item.second] = static_cast<int32_t>(it - uniq.begin());
        }
        return codes;
    }

    std::vector<double> edges;
    edges.reserve(static_cast<size_t>(max_bins > 1 ? max_bins - 1 : 0));
    for (int k = 1; k < max_bins; ++k) {
        const double q = static_cast<double>(k) / static_cast<double>(max_bins);
        const double pos = q * static_cast<double>(sorted_vals.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = static_cast<size_t>(std::ceil(pos));
        const double frac = pos - static_cast<double>(lo);
        const double edge = sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac;
        if (edges.empty() || edge != edges.back()) edges.push_back(edge);
    }
    if (edges.empty()) {
        for (const auto& item : finite) codes[item.second] = 0;
    } else {
        for (const auto& item : finite) {
            auto it = std::upper_bound(edges.begin(), edges.end(), item.first);
            codes[item.second] = static_cast<int32_t>(it - edges.begin());
        }
    }
    return codes;
}




struct InteractionSelectionRecord {
    int64_t index = -1;
    double score = 0.0;
    double marginal_ig = 0.0;
    int64_t best_partner = -1;
};

double conditional_entropy_from_table(
    const std::vector<int64_t>& counts,
    const std::vector<int64_t>& totals,
    int64_t n_classes,
    int64_t eligible_count
) {
    if (eligible_count <= 0) return 0.0;
    std::vector<int64_t> cls_counts(static_cast<size_t>(n_classes), 0);
    double cond = 0.0;
    for (size_t g = 0; g < totals.size(); ++g) {
        const int64_t total = totals[g];
        if (total <= 0) continue;
        const size_t offset = g * static_cast<size_t>(n_classes);
        for (int64_t cls = 0; cls < n_classes; ++cls) {
            cls_counts[static_cast<size_t>(cls)] = counts[offset + static_cast<size_t>(cls)];
        }
        cond += (static_cast<double>(total) / static_cast<double>(eligible_count))
            * entropy_from_counts(cls_counts, total);
    }
    return cond;
}

double marginal_ig_skip_missing(
    const std::vector<int32_t>& codes,
    const int64_t* y,
    py::ssize_t n,
    int64_t n_classes,
    int32_t n_bins
) {
    if (n <= 0 || n_bins <= 0) return 0.0;
    std::vector<int64_t> label_counts(static_cast<size_t>(n_classes), 0);
    std::vector<int64_t> counts(static_cast<size_t>(n_bins) * static_cast<size_t>(n_classes), 0);
    std::vector<int64_t> totals(static_cast<size_t>(n_bins), 0);
    int64_t eligible = 0;
    for (py::ssize_t i = 0; i < n; ++i) {
        const int32_t c = codes[static_cast<size_t>(i)];
        const int64_t cls = y[i];
        if (c < 0 || c >= n_bins || cls < 0 || cls >= n_classes) continue;
        ++eligible;
        label_counts[static_cast<size_t>(cls)]++;
        totals[static_cast<size_t>(c)]++;
        counts[static_cast<size_t>(c) * static_cast<size_t>(n_classes) + static_cast<size_t>(cls)]++;
    }
    if (eligible < 3) return 0.0;
    const double base = entropy_from_counts(label_counts, eligible);
    if (base <= 0.0) return 0.0;
    const double cond = conditional_entropy_from_table(counts, totals, n_classes, eligible);
    return std::max(0.0, base - cond);
}

py::list select_interaction_information_features(
    py::array_t<double, py::array::c_style | py::array::forcecast> X_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> y_arr,
    const std::vector<std::string>& col_names,
    int64_t aug_feature_size,
    py::object ii_partner_size_obj
) {
    auto X = X_arr.unchecked<2>();
    auto y = y_arr.unchecked<1>();
    const py::ssize_t n = X.shape(0);
    const py::ssize_t p = X.shape(1);
    if (y.shape(0) != n) throw std::invalid_argument("X and y row counts do not match.");
    if (static_cast<py::ssize_t>(col_names.size()) != p) throw std::invalid_argument("col_names length does not match X columns.");

    py::list out;
    if (n <= 0 || p <= 0 || aug_feature_size <= 0) return out;

    int64_t partner_size = -1;
    if (!ii_partner_size_obj.is_none()) partner_size = py::cast<int64_t>(ii_partner_size_obj);

    std::vector<InteractionSelectionRecord> selected;
    selected.reserve(static_cast<size_t>(std::min<int64_t>(aug_feature_size, static_cast<int64_t>(p))));

    {
        py::gil_scoped_release release;

        int64_t n_classes = 0;
        const int64_t* y_ptr = y_arr.data();
        for (py::ssize_t i = 0; i < n; ++i) {
            if (y_ptr[i] >= n_classes) n_classes = y_ptr[i] + 1;
        }
        if (n_classes <= 0) n_classes = 1;

        constexpr int fixed_bins = 4;
        std::vector<std::vector<int32_t>> codes(static_cast<size_t>(p));
        std::vector<int32_t> n_bins(static_cast<size_t>(p), 1);
        std::vector<double> marginal_ig(static_cast<size_t>(p), 0.0);
        std::vector<double> values(static_cast<size_t>(n), 0.0);

        for (py::ssize_t j = 0; j < p; ++j) {
            for (py::ssize_t i = 0; i < n; ++i) {
                const double v = X(i, j);
                values[static_cast<size_t>(i)] = finite_double(v) ? v : std::numeric_limits<double>::quiet_NaN();
            }
            codes[static_cast<size_t>(j)] = continuous_to_quantile_codes_fixed(values, fixed_bins);
            int32_t max_code = -1;
            for (int32_t c : codes[static_cast<size_t>(j)]) {
                if (c > max_code) max_code = c;
            }
            n_bins[static_cast<size_t>(j)] = std::max<int32_t>(1, max_code + 1);
            marginal_ig[static_cast<size_t>(j)] = marginal_ig_skip_missing(
                codes[static_cast<size_t>(j)], y_ptr, n, n_classes, n_bins[static_cast<size_t>(j)]
            );
        }

        std::vector<uint8_t> is_partner(static_cast<size_t>(p), 1);
        if (partner_size > 0 && partner_size < static_cast<int64_t>(p)) {
            std::fill(is_partner.begin(), is_partner.end(), static_cast<uint8_t>(0));
            std::vector<int64_t> order(static_cast<size_t>(p));
            std::iota(order.begin(), order.end(), 0);
            std::mt19937 rng(0);
            std::shuffle(order.begin(), order.end(), rng);
            for (int64_t r = 0; r < partner_size; ++r) {
                is_partner[static_cast<size_t>(order[static_cast<size_t>(r)])] = 1;
            }
        }

        std::vector<double> best_interaction(static_cast<size_t>(p), 0.0);
        std::vector<int64_t> best_partner(static_cast<size_t>(p), -1);
        std::vector<int64_t> label_counts;
        std::vector<int64_t> joint_counts;
        std::vector<int64_t> joint_totals;
        std::vector<int64_t> a_counts;
        std::vector<int64_t> a_totals;
        std::vector<int64_t> b_counts;
        std::vector<int64_t> b_totals;

        for (py::ssize_t a = 0; a < p; ++a) {
            for (py::ssize_t b = a + 1; b < p; ++b) {
                if (is_partner[static_cast<size_t>(a)] == 0 && is_partner[static_cast<size_t>(b)] == 0) continue;
                const int32_t nb_a = n_bins[static_cast<size_t>(a)];
                const int32_t nb_b = n_bins[static_cast<size_t>(b)];
                if (nb_a <= 0 || nb_b <= 0) continue;
                const int64_t n_joint = static_cast<int64_t>(nb_a) * static_cast<int64_t>(nb_b);

                label_counts.assign(static_cast<size_t>(n_classes), 0);
                joint_counts.assign(static_cast<size_t>(n_joint) * static_cast<size_t>(n_classes), 0);
                joint_totals.assign(static_cast<size_t>(n_joint), 0);
                a_counts.assign(static_cast<size_t>(nb_a) * static_cast<size_t>(n_classes), 0);
                a_totals.assign(static_cast<size_t>(nb_a), 0);
                b_counts.assign(static_cast<size_t>(nb_b) * static_cast<size_t>(n_classes), 0);
                b_totals.assign(static_cast<size_t>(nb_b), 0);

                int64_t eligible = 0;
                for (py::ssize_t i = 0; i < n; ++i) {
                    const int32_t ca = codes[static_cast<size_t>(a)][static_cast<size_t>(i)];
                    const int32_t cb = codes[static_cast<size_t>(b)][static_cast<size_t>(i)];
                    const int64_t cls = y_ptr[i];
                    if (ca < 0 || cb < 0 || ca >= nb_a || cb >= nb_b || cls < 0 || cls >= n_classes) continue;

                    ++eligible;
                    label_counts[static_cast<size_t>(cls)]++;
                    a_totals[static_cast<size_t>(ca)]++;
                    b_totals[static_cast<size_t>(cb)]++;
                    a_counts[static_cast<size_t>(ca) * static_cast<size_t>(n_classes) + static_cast<size_t>(cls)]++;
                    b_counts[static_cast<size_t>(cb) * static_cast<size_t>(n_classes) + static_cast<size_t>(cls)]++;
                    const int64_t joint = static_cast<int64_t>(ca) * static_cast<int64_t>(nb_b) + static_cast<int64_t>(cb);
                    joint_totals[static_cast<size_t>(joint)]++;
                    joint_counts[static_cast<size_t>(joint) * static_cast<size_t>(n_classes) + static_cast<size_t>(cls)]++;
                }
                if (eligible < 3) continue;

                const double base = entropy_from_counts(label_counts, eligible);
                if (base <= 0.0) continue;
                const double cond_joint = conditional_entropy_from_table(joint_counts, joint_totals, n_classes, eligible);
                const double cond_a = conditional_entropy_from_table(a_counts, a_totals, n_classes, eligible);
                const double cond_b = conditional_entropy_from_table(b_counts, b_totals, n_classes, eligible);
                const double joint_ig = std::max(0.0, base - cond_joint);
                const double ig_a = std::max(0.0, base - cond_a);
                const double ig_b = std::max(0.0, base - cond_b);
                const double interaction = joint_ig - ig_a - ig_b;

                if (interaction > best_interaction[static_cast<size_t>(a)]) {
                    best_interaction[static_cast<size_t>(a)] = interaction;
                    best_partner[static_cast<size_t>(a)] = b;
                }
                if (interaction > best_interaction[static_cast<size_t>(b)]) {
                    best_interaction[static_cast<size_t>(b)] = interaction;
                    best_partner[static_cast<size_t>(b)] = a;
                }
            }
        }

        std::vector<int64_t> order(static_cast<size_t>(p));
        std::iota(order.begin(), order.end(), 0);
        auto cmp = [&](int64_t lhs, int64_t rhs) {
            const double li = best_interaction[static_cast<size_t>(lhs)];
            const double ri = best_interaction[static_cast<size_t>(rhs)];
            if (li != ri) return li > ri;
            const double lm = marginal_ig[static_cast<size_t>(lhs)];
            const double rm = marginal_ig[static_cast<size_t>(rhs)];
            if (lm != rm) return lm > rm;
            return col_names[static_cast<size_t>(lhs)] < col_names[static_cast<size_t>(rhs)];
        };
        const int64_t keep = std::min<int64_t>(aug_feature_size, static_cast<int64_t>(p));
        if (keep < static_cast<int64_t>(p)) {
            std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(keep), order.end(), cmp);
        } else {
            std::sort(order.begin(), order.end(), cmp);
        }

        for (int64_t r = 0; r < keep; ++r) {
            const int64_t j = order[static_cast<size_t>(r)];
            InteractionSelectionRecord rec;
            rec.index = j;
            rec.score = best_interaction[static_cast<size_t>(j)];
            rec.marginal_ig = marginal_ig[static_cast<size_t>(j)];
            rec.best_partner = best_partner[static_cast<size_t>(j)];
            selected.push_back(rec);
        }
    }

    for (size_t r = 0; r < selected.size(); ++r) {
        const InteractionSelectionRecord& rec = selected[r];
        py::dict d;
        d["name"] = col_names[static_cast<size_t>(rec.index)];
        d["score"] = rec.score;
        d["interaction_score"] = rec.score;
        d["marginal_ig"] = rec.marginal_ig;
        if (rec.best_partner >= 0) {
            d["best_partner"] = col_names[static_cast<size_t>(rec.best_partner)];
        } else {
            d["best_partner"] = py::none();
        }
        d["mode"] = "interaction_information";
        d["rank"] = static_cast<int64_t>(r + 1);
        out.append(std::move(d));
    }
    return out;
}

py::tuple strict_topk_filter_dense(
    py::array_t<float, py::array::c_style | py::array::forcecast> X_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> y_arr,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> discrete_mask_arr,
    int64_t top_k,
    int max_bins
) {
    auto X = X_arr.unchecked<2>();
    auto y = y_arr.unchecked<1>();
    auto discrete_mask = discrete_mask_arr.unchecked<1>();
    const int64_t n_rows = static_cast<int64_t>(X.shape(0));
    const int64_t n_cols = static_cast<int64_t>(X.shape(1));
    if (y.shape(0) != static_cast<py::ssize_t>(n_rows)) throw std::invalid_argument("y length does not match n_rows.");
    if (discrete_mask.shape(0) != static_cast<py::ssize_t>(n_cols)) throw std::invalid_argument("discrete_mask length does not match n_cols.");

    py::array_t<double> scores_arr({static_cast<py::ssize_t>(n_cols)});
    py::array_t<uint8_t> mask_arr({static_cast<py::ssize_t>(n_cols)});
    auto scores = scores_arr.mutable_unchecked<1>();
    auto mask = mask_arr.mutable_unchecked<1>();
    for (int64_t j = 0; j < n_cols; ++j) {
        scores(j) = 0.0;
        mask(j) = 1;
    }
    if (n_cols == 0 || n_rows == 0) return py::make_tuple(scores_arr, mask_arr);

    int64_t n_classes = 0;
    for (int64_t i = 0; i < n_rows; ++i) {
        const int64_t cls = y(static_cast<py::ssize_t>(i));
        if (cls >= n_classes) n_classes = cls + 1;
    }
    if (n_classes <= 1) {
        if (top_k >= 0 && top_k < n_cols) {
            for (int64_t j = top_k; j < n_cols; ++j) mask(j) = 0;
        }
        return py::make_tuple(scores_arr, mask_arr);
    }
    const int64_t* y_ptr = y_arr.data();
    const double base_entropy = entropy_labels(y_ptr, static_cast<py::ssize_t>(n_rows), n_classes);
    if (base_entropy <= 0.0) return py::make_tuple(scores_arr, mask_arr);

    std::vector<int32_t> z(static_cast<size_t>(n_rows), 0);
    std::vector<double> vals(static_cast<size_t>(n_rows), 0.0);

    for (int64_t j = 0; j < n_cols; ++j) {
        if (discrete_mask(static_cast<py::ssize_t>(j)) != 0) {
            for (int64_t i = 0; i < n_rows; ++i) {
                const float v = X(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j));
                z[static_cast<size_t>(i)] = (std::isfinite(v) && v > 0.5f) ? 1 : 0;
            }
            scores(j) = std::max(0.0, discrete_ig(z, y_ptr, static_cast<py::ssize_t>(n_rows), n_classes, base_entropy));
        } else {
            for (int64_t i = 0; i < n_rows; ++i) {
                const float v = X(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j));
                vals[static_cast<size_t>(i)] = std::isfinite(v) ? static_cast<double>(v) : std::numeric_limits<double>::quiet_NaN();
            }
            z = continuous_to_quantile_codes_fixed(vals, std::max(2, max_bins));
            scores(j) = std::max(0.0, discrete_ig(z, y_ptr, static_cast<py::ssize_t>(n_rows), n_classes, base_entropy));
        }
    }

    if (top_k >= 0 && top_k < n_cols) {
        std::vector<int64_t> order(static_cast<size_t>(n_cols));
        std::iota(order.begin(), order.end(), 0);
        const size_t k = static_cast<size_t>(top_k);
        auto cmp = [&](int64_t a, int64_t b) {
            const double sa = scores(static_cast<py::ssize_t>(a));
            const double sb = scores(static_cast<py::ssize_t>(b));
            if (sa == sb) return a < b;
            return sa > sb;
        };
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(), cmp);
        for (int64_t j = 0; j < n_cols; ++j) mask(j) = 0;
        for (size_t r = 0; r < k; ++r) mask(static_cast<py::ssize_t>(order[r])) = 1;
    }
    return py::make_tuple(scores_arr, mask_arr);
}

py::tuple strict_topk_filter_csc(
    py::array_t<float, py::array::c_style | py::array::forcecast> data_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> indices_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> indptr_arr,
    int64_t n_rows,
    int64_t n_cols,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> y_arr,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> discrete_mask_arr,
    int64_t top_k,
    int max_bins
) {
    if (n_rows < 0 || n_cols < 0) throw std::invalid_argument("n_rows and n_cols must be non-negative.");
    auto data = data_arr.unchecked<1>();
    auto indices = indices_arr.unchecked<1>();
    auto indptr = indptr_arr.unchecked<1>();
    auto y = y_arr.unchecked<1>();
    auto discrete_mask = discrete_mask_arr.unchecked<1>();
    if (y.shape(0) != static_cast<py::ssize_t>(n_rows)) throw std::invalid_argument("y length does not match n_rows.");
    if (discrete_mask.shape(0) != static_cast<py::ssize_t>(n_cols)) throw std::invalid_argument("discrete_mask length does not match n_cols.");
    if (indptr.shape(0) != static_cast<py::ssize_t>(n_cols + 1)) throw std::invalid_argument("indptr length does not match n_cols + 1.");

    py::array_t<double> scores_arr({static_cast<py::ssize_t>(n_cols)});
    py::array_t<uint8_t> mask_arr({static_cast<py::ssize_t>(n_cols)});
    auto scores = scores_arr.mutable_unchecked<1>();
    auto mask = mask_arr.mutable_unchecked<1>();
    for (int64_t j = 0; j < n_cols; ++j) {
        scores(j) = 0.0;
        mask(j) = 1;
    }
    if (n_cols == 0 || n_rows == 0) return py::make_tuple(scores_arr, mask_arr);

    int64_t n_classes = 0;
    for (int64_t i = 0; i < n_rows; ++i) {
        const int64_t cls = y(static_cast<py::ssize_t>(i));
        if (cls >= n_classes) n_classes = cls + 1;
    }
    if (n_classes <= 1) {
        if (top_k >= 0 && top_k < n_cols) {
            for (int64_t j = top_k; j < n_cols; ++j) mask(j) = 0;
        }
        return py::make_tuple(scores_arr, mask_arr);
    }
    const int64_t* y_ptr = y_arr.data();
    const double base_entropy = entropy_labels(y_ptr, static_cast<py::ssize_t>(n_rows), n_classes);
    if (base_entropy <= 0.0) return py::make_tuple(scores_arr, mask_arr);

    std::vector<int32_t> z(static_cast<size_t>(n_rows), 0);
    std::vector<double> vals(static_cast<size_t>(n_rows), 0.0);

    for (int64_t j = 0; j < n_cols; ++j) {
        const py::ssize_t start = indptr(static_cast<py::ssize_t>(j));
        const py::ssize_t end = indptr(static_cast<py::ssize_t>(j + 1));
        if (start < 0 || end < start || end > data.shape(0)) throw std::invalid_argument("Invalid CSC indptr.");
        if (discrete_mask(static_cast<py::ssize_t>(j)) != 0) {
            std::fill(z.begin(), z.end(), 0);
            for (py::ssize_t pos = start; pos < end; ++pos) {
                const int64_t row = indices(pos);
                if (row < 0 || row >= n_rows) throw std::invalid_argument("Invalid CSC row index.");
                z[static_cast<size_t>(row)] = (data(pos) > 0.5f) ? 1 : 0;
            }
            scores(j) = std::max(0.0, discrete_ig(z, y_ptr, static_cast<py::ssize_t>(n_rows), n_classes, base_entropy));
        } else {
            std::fill(vals.begin(), vals.end(), 0.0);
            for (py::ssize_t pos = start; pos < end; ++pos) {
                const int64_t row = indices(pos);
                if (row < 0 || row >= n_rows) throw std::invalid_argument("Invalid CSC row index.");
                vals[static_cast<size_t>(row)] = static_cast<double>(data(pos));
            }
            z = continuous_to_quantile_codes_fixed(vals, std::max(2, max_bins));
            scores(j) = std::max(0.0, discrete_ig(z, y_ptr, static_cast<py::ssize_t>(n_rows), n_classes, base_entropy));
        }
    }

    if (top_k >= 0 && top_k < n_cols) {
        std::vector<int64_t> order(static_cast<size_t>(n_cols));
        std::iota(order.begin(), order.end(), 0);
        const size_t k = static_cast<size_t>(top_k);
        auto cmp = [&](int64_t a, int64_t b) {
            const double sa = scores(static_cast<py::ssize_t>(a));
            const double sb = scores(static_cast<py::ssize_t>(b));
            if (sa == sb) return a < b;
            return sa > sb;
        };
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(), cmp);
        for (int64_t j = 0; j < n_cols; ++j) mask(j) = 0;
        for (size_t r = 0; r < k; ++r) mask(static_cast<py::ssize_t>(order[r])) = 1;
    }
    return py::make_tuple(scores_arr, mask_arr);
}

// ─────────────────────────────────────────────────────────────────────────
// select_pair_aware_adaptive_bins
//
// Chooses a bin count per numeric column for interaction_relaxed_mining,
// using each survivor column's known best-partner pairing (already
// computed by select_interaction_information_features) to score candidate
// bin counts by JOINT information against that partner instead of purely
// marginal information against the target -- so a column whose value is
// almost entirely interaction-driven (near-zero marginal IG at any bin
// count) is not forced toward the finest available resolution by an
// elbow-stop rule that only ever sees marginal IG.
//
// Algorithm:
//   1. For each numeric column, scan candidate bin counts b (capped to
//      <= pair_aware_max_bins when the column has at least one partner,
//      which keeps the criterion well-behaved rather than monotonically
//      rewarding finer bins regardless of real structure).
//   2. marginal = IG(quantile_code(col, b); y).
//   3. For each partner, joint = IG(quantile_code(col,b) combined with
//      quantile_code(partner, min(b, pair_aware_max_bins)); y), and
//      pair_score = max(conditional, min(joint, max(left, right))) where
//      conditional = max(0, joint - right). This caps the pair score
//      against what either marginal IG alone already explains.
//   4. score[b] = max(marginal, best_pair_score_over_partners).
//   5. Elbow-stop: pick the SMALLEST b whose score clears
//      best_score * pair_aware_threshold_ratio (when partnered) or
//      best_score * (1 - min_marginal_gain_ratio) (when not) -- i.e.
//      coarsen by default, only move finer if it buys a real improvement.
//
// Every column's candidate codes are computed once and reused for every
// pairing that needs them within the same scan; IG and joint IG are
// single-pass count-table accumulations.
// ─────────────────────────────────────────────────────────────────────────
py::dict select_pair_aware_adaptive_bins(
    py::array_t<double, py::array::c_style | py::array::forcecast> X_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> y_arr,
    const std::vector<std::string>& col_names,
    const std::vector<int64_t>& candidate_bins,
    const std::vector<std::pair<int64_t, int64_t>>& pair_indices,
    double min_marginal_gain_ratio,
    double pair_aware_threshold_ratio,
    int64_t pair_aware_max_bins
) {
    auto X = X_arr.unchecked<2>();
    const py::ssize_t n = X.shape(0);
    const py::ssize_t p = X.shape(1);
    py::dict out;
    if (n <= 0 || p <= 0 || candidate_bins.empty()) return out;
    if (static_cast<py::ssize_t>(col_names.size()) != p) {
        throw std::invalid_argument("col_names length does not match X columns.");
    }

    std::vector<int64_t> sorted_candidates(candidate_bins.begin(), candidate_bins.end());
    std::sort(sorted_candidates.begin(), sorted_candidates.end());
    sorted_candidates.erase(
        std::unique(sorted_candidates.begin(), sorted_candidates.end()), sorted_candidates.end()
    );

    // Build, per column, the set of partner column indices (symmetric).
    std::vector<std::vector<int64_t>> partners(static_cast<size_t>(p));
    for (const auto& pr : pair_indices) {
        const int64_t a = pr.first;
        const int64_t b = pr.second;
        if (a < 0 || a >= p || b < 0 || b >= p || a == b) continue;
        partners[static_cast<size_t>(a)].push_back(b);
        partners[static_cast<size_t>(b)].push_back(a);
    }

    // Plain C++ accumulator for per-column results computed while the GIL
    // is released; no py:: objects are touched until after re-acquiring it.
    struct PairAwareColumnResult {
        py::ssize_t col = 0;
        int64_t chosen_b = 0;
        std::map<int64_t, double> scores;
        std::map<int64_t, double> marginal_scores;
        bool has_partner = false;
        int64_t best_partner_col = -1;
        int64_t best_partner_bins = -1;
        double best_score = 0.0;
    };
    std::vector<PairAwareColumnResult> column_results;
    column_results.reserve(static_cast<size_t>(p));

    py::gil_scoped_release release;

    int64_t n_classes = 0;
    const int64_t* y_ptr = y_arr.data();
    for (py::ssize_t i = 0; i < n; ++i) {
        if (y_ptr[i] >= n_classes) n_classes = y_ptr[i] + 1;
    }
    if (n_classes <= 0) n_classes = 1;

    std::vector<double> values(static_cast<size_t>(n), 0.0);

    // Cache, per column, codes/n_bins at every candidate bin count actually
    // needed (every column needs its own scan_candidates; partner columns
    // additionally need codes at min(b, pair_aware_max_bins) for whichever
    // b values their partners scan). To keep this simple and still avoid
    // recomputation within one column's own scan, codes are computed once
    // per (column, bin_count) the first time they are needed and reused.
    std::map<std::pair<int64_t, int64_t>, std::vector<int32_t>> code_cache;
    std::map<std::pair<int64_t, int64_t>, int32_t> nbins_cache;

    auto get_codes = [&](int64_t col, int64_t b) -> const std::vector<int32_t>& {
        auto key = std::make_pair(col, b);
        auto it = code_cache.find(key);
        if (it != code_cache.end()) return it->second;
        for (py::ssize_t i = 0; i < n; ++i) {
            const double v = X(i, col);
            values[static_cast<size_t>(i)] = finite_double(v) ? v : std::numeric_limits<double>::quiet_NaN();
        }
        std::vector<int32_t> codes = continuous_to_quantile_codes_fixed(values, static_cast<int>(b));
        int32_t max_code = -1;
        for (int32_t c : codes) {
            if (c > max_code) max_code = c;
        }
        nbins_cache[key] = std::max<int32_t>(1, max_code + 1);
        auto& slot = code_cache[key];
        slot = std::move(codes);
        return slot;
    };
    auto get_nbins = [&](int64_t col, int64_t b) -> int32_t {
        get_codes(col, b);
        return nbins_cache[std::make_pair(col, b)];
    };

    for (py::ssize_t col = 0; col < p; ++col) {
        const bool has_partner = !partners[static_cast<size_t>(col)].empty();
        std::vector<int64_t> scan = sorted_candidates;
        if (has_partner) {
            std::vector<int64_t> capped;
            for (int64_t b : scan) if (b <= pair_aware_max_bins) capped.push_back(b);
            if (!capped.empty()) scan = capped;
        }

        std::map<int64_t, double> scores;
        std::map<int64_t, double> marginal_scores;
        int64_t best_partner_overall = -1;
        int64_t best_partner_bins_overall = -1;
        double best_pair_score_overall = 0.0;

        for (int64_t b : scan) {
            const auto& codes_x = get_codes(col, b);
            const int32_t nb_x = get_nbins(col, b);
            const double marginal = marginal_ig_skip_missing(codes_x, y_ptr, n, n_classes, nb_x);
            marginal_scores[b] = marginal;

            double best_pair_score = 0.0;
            int64_t best_partner_name = -1;
            int64_t best_partner_bins = -1;

            for (int64_t partner_col : partners[static_cast<size_t>(col)]) {
                const int64_t pb = std::min<int64_t>(b, pair_aware_max_bins);
                const auto& codes_y = get_codes(partner_col, pb);
                const int32_t nb_y = get_nbins(partner_col, pb);
                const double right_score = marginal_ig_skip_missing(codes_y, y_ptr, n, n_classes, nb_y);

                const int64_t n_joint = static_cast<int64_t>(nb_x) * static_cast<int64_t>(nb_y);
                std::vector<int64_t> label_counts(static_cast<size_t>(n_classes), 0);
                std::vector<int64_t> joint_counts(static_cast<size_t>(n_joint) * static_cast<size_t>(n_classes), 0);
                std::vector<int64_t> joint_totals(static_cast<size_t>(n_joint), 0);
                int64_t eligible = 0;
                for (py::ssize_t i = 0; i < n; ++i) {
                    const int32_t cx = codes_x[static_cast<size_t>(i)];
                    const int32_t cy = codes_y[static_cast<size_t>(i)];
                    const int64_t cls = y_ptr[i];
                    if (cx < 0 || cy < 0 || cx >= nb_x || cy >= nb_y || cls < 0 || cls >= n_classes) continue;
                    ++eligible;
                    label_counts[static_cast<size_t>(cls)]++;
                    const int64_t joint = static_cast<int64_t>(cx) * static_cast<int64_t>(nb_y) + static_cast<int64_t>(cy);
                    joint_totals[static_cast<size_t>(joint)]++;
                    joint_counts[static_cast<size_t>(joint) * static_cast<size_t>(n_classes) + static_cast<size_t>(cls)]++;
                }
                double joint_score = 0.0;
                if (eligible >= 3) {
                    const double base = entropy_from_counts(label_counts, eligible);
                    if (base > 0.0) {
                        const double cond_joint = conditional_entropy_from_table(
                            joint_counts, joint_totals, n_classes, eligible
                        );
                        joint_score = std::max(0.0, base - cond_joint);
                    }
                }
                const double left_score = marginal;  // IG of col itself at its own b
                const double conditional = std::max(0.0, joint_score - right_score);
                const double pair_score = std::max(
                    conditional, std::min(joint_score, std::max(left_score, right_score))
                );
                if (pair_score > best_pair_score) {
                    best_pair_score = pair_score;
                    best_partner_name = partner_col;
                    best_partner_bins = nb_y;
                }
            }
            scores[b] = std::max(marginal, best_pair_score);
            if (best_pair_score > best_pair_score_overall) {
                best_pair_score_overall = best_pair_score;
                best_partner_overall = best_partner_name;
                best_partner_bins_overall = best_partner_bins;
            }
        }

        if (scores.empty()) continue;
        double best_score = 0.0;
        for (const auto& kv : scores) best_score = std::max(best_score, kv.second);

        int64_t chosen;
        if (best_score <= 0.0) {
            chosen = scores.begin()->first;
        } else {
            const double threshold = has_partner
                ? best_score * pair_aware_threshold_ratio
                : best_score * std::max(0.0, 1.0 - min_marginal_gain_ratio);
            chosen = -1;
            for (const auto& kv : scores) {  // map is sorted ascending by key
                if (kv.second >= threshold) { chosen = kv.first; break; }
            }
            if (chosen < 0) {
                chosen = scores.begin()->first;
                double best_val = scores.begin()->second;
                for (const auto& kv : scores) {
                    if (kv.second > best_val) { best_val = kv.second; chosen = kv.first; }
                }
            }
        }

        // Stash results in plain C++ containers only -- no py:: object
        // construction here, since the GIL is released for this entire
        // outer loop. All py::dict/py::str building happens in a separate
        // pass below, after the GIL is reacquired.
        PairAwareColumnResult rec;
        rec.col = col;
        rec.chosen_b = chosen;
        rec.scores = scores;
        rec.marginal_scores = marginal_scores;
        rec.has_partner = has_partner;
        rec.best_partner_col = (has_partner ? best_partner_overall : -1);
        rec.best_partner_bins = best_partner_bins_overall;
        rec.best_score = best_score;
        column_results.push_back(std::move(rec));
    }

    py::gil_scoped_acquire acquire;
    py::dict chosen_b_out;
    py::dict scores_out;
    py::dict evidence_out;
    for (const auto& rec : column_results) {
        const std::string& name = col_names[static_cast<size_t>(rec.col)];
        chosen_b_out[py::str(name)] = rec.chosen_b;

        py::dict score_dict;
        for (const auto& kv : rec.scores) score_dict[py::int_(kv.first)] = kv.second;
        scores_out[py::str(name)] = score_dict;

        py::dict evidence;
        evidence["mode"] = rec.has_partner ? "pair_aware" : "marginal";
        evidence["feature"] = name;
        evidence["chosen_b"] = rec.chosen_b;
        auto it = rec.scores.find(rec.chosen_b);
        evidence["score"] = (it != rec.scores.end()) ? it->second : 0.0;
        evidence["best_score"] = rec.best_score;
        if (rec.has_partner && rec.best_partner_col >= 0) {
            evidence["best_partner"] = col_names[static_cast<size_t>(rec.best_partner_col)];
            evidence["partner_bins"] = rec.best_partner_bins;
        } else {
            evidence["best_partner"] = py::none();
            evidence["partner_bins"] = py::none();
        }
        py::dict marginal_dict;
        for (const auto& kv : rec.marginal_scores) marginal_dict[py::int_(kv.first)] = kv.second;
        evidence["marginal_scores"] = marginal_dict;
        evidence_out[py::str(name)] = evidence;
    }

    out["chosen_b"] = chosen_b_out;
    out["scores"] = scores_out;
    out["evidence"] = evidence_out;
    return out;
}

} // namespace

void bind_augmented_pair(py::module_& m)
{
    m.def(
        "score_pair_candidates",
        &score_pair_candidates,
        py::arg("X"),
        py::arg("y"),
        py::arg("col_names"),
        "Score product/absolute-difference/sum/signed-difference pair candidates on observed source pairs with adaptive-binned IG."
    );
    m.def(
        "transform_pair_features",
        &transform_pair_features,
        py::arg("X"),
        py::arg("left_indices"),
        py::arg("right_indices"),
        py::arg("op_codes"),
        py::arg("reference_values"),
        py::arg("means"),
        py::arg("scales"),
        "Generate standardized augmented pair features natively; unavailable pairs use per-pair reference values."
    );

    m.def(
        "select_interaction_information_features",
        &select_interaction_information_features,
        py::arg("X"),
        py::arg("y"),
        py::arg("col_names"),
        py::arg("aug_feature_size"),
        py::arg("ii_partner_size") = py::none(),
        "Select source features for augmented pair generation using native interaction-information scoring."
    );

    m.def(
        "select_pair_aware_adaptive_bins",
        &select_pair_aware_adaptive_bins,
        py::arg("X"),
        py::arg("y"),
        py::arg("col_names"),
        py::arg("candidate_bins"),
        py::arg("pair_indices"),
        py::arg("min_marginal_gain_ratio"),
        py::arg("pair_aware_threshold_ratio"),
        py::arg("pair_aware_max_bins"),
        "Native pair-aware adaptive bin-count selection for interaction_relaxed_mining: "
        "scores candidate bin counts per column by joint information against its known "
        "best-partner column(s) instead of purely marginal information against the target."
    );

    m.def(
        "strict_topk_filter_dense",
        &strict_topk_filter_dense,
        py::arg("X"),
        py::arg("y"),
        py::arg("discrete_mask"),
        py::arg("top_k"),
        py::arg("max_bins"),
        "Score dense downstream columns by IG and return scores plus a topK mask."
    );

    m.def(
        "strict_topk_filter_csc",
        &strict_topk_filter_csc,
        py::arg("data"),
        py::arg("indices"),
        py::arg("indptr"),
        py::arg("n_rows"),
        py::arg("n_cols"),
        py::arg("y"),
        py::arg("discrete_mask"),
        py::arg("top_k"),
        py::arg("max_bins"),
        "Score final downstream CSC columns by IG and return scores plus a topK mask."
    );
}


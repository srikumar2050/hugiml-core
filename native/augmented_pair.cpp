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
#include <numeric>
#include <string>
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


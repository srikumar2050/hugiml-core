/**
 * transaction.cpp — Transaction preparation implementation
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

#include "transaction.hpp"
#include "math.hpp"
#include "discretization.hpp"

#include <iomanip>
#include <limits>
#include <set>
#include <sstream>

namespace hugiml {

TransactionDataCpp prepare_transactions_cpp(
    const py::array_t<double,  py::array::c_style | py::array::forcecast>& X_num_arr,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& y_arr,
    int B,
    std::vector<std::string>              col_names_in,
    const py::array_t<uint8_t, py::array::forcecast>& is_cat_arr,
    const py::array_t<uint8_t, py::array::forcecast>& is_int_arr,
    std::vector<std::vector<std::string>> cat_raw_strs,
    std::vector<std::vector<bool>>        cat_raw_valid)
{
    auto Xb  = X_num_arr.unchecked<2>();
    auto yb  = y_arr.unchecked<1>();
    auto icb = is_cat_arr.unchecked<1>();
    auto iib = is_int_arr.unchecked<1>();

    int n = static_cast<int>(Xb.shape(0));
    int p = static_cast<int>(Xb.shape(1));

    // ── Input validation ─────────────────────────────────────────────────────
    if (n <= 0)
        throw std::invalid_argument("X must have at least one row");
    if (p <= 0)
        throw std::invalid_argument("X must have at least one column");
    validate_same_size(
        static_cast<size_t>(n),
        static_cast<size_t>(yb.shape(0)),
        "X.shape[0]", "y.shape[0]");
    if (B != -1 && (B < 2 || B > 100))
        throw std::invalid_argument(
            "B must be -1 (auto) or in range [2, 100], got " + std::to_string(B));
    if (n > 10000000)
        throw std::runtime_error("Dataset too large: max 10M rows supported");

    // Memory guard: disc_mat would be n*p int32 — warn early on huge datasets
    {
        uint64_t disc_bytes = static_cast<uint64_t>(n) * p * sizeof(int32_t);
        if (disc_bytes > 2ULL * 1024 * 1024 * 1024)  // 2 GiB
            throw std::runtime_error(
                "n*p too large: discretisation matrix would exceed 2 GiB "
                "(n=" + std::to_string(n) + ", p=" + std::to_string(p) + ")");
    }

    // ── Bin-key stride ───────────────────────────────────────────────────────
    int bkey_stride = std::max(p, 10000);
    auto bk = [bkey_stride](int bi, int j) { return bi * bkey_stride + j; };

    // ── y as std::vector<int> ────────────────────────────────────────────────
    std::vector<int> y_vec(n);
    int max_label = 0;
    std::set<int> cls_set;
    for (int i = 0; i < n; i++) {
        y_vec[i] = static_cast<int>(yb(i));
        if (y_vec[i] < 0)
            throw std::invalid_argument(
                "y contains negative label at index " + std::to_string(i));
        if (y_vec[i] > max_label) max_label = y_vec[i];
        cls_set.insert(y_vec[i]);
    }
    int n_cls      = static_cast<int>(cls_set.size());
    int key_stride = max_label + 1;

    if (n_cls < 2)
        throw std::invalid_argument(
            "y must contain at least 2 distinct classes, got " + std::to_string(n_cls));

    // ── Column names ─────────────────────────────────────────────────────────
    std::vector<std::string> names(p);
    if (col_names_in.size() == static_cast<size_t>(p)) {
        names = std::move(col_names_in);
    } else {
        for (int j = 0; j < p; j++)
            names[j] = "col" + std::to_string(j);
    }

    // ── is_cat / is_int masks ────────────────────────────────────────────────
    std::vector<bool> is_cat(p), is_int(p);
    for (int j = 0; j < p; j++) {
        is_cat[j] = static_cast<bool>(icb(j));
        is_int[j] = static_cast<bool>(iib(j));
    }

    // Validate pre-extracted cat data dimensions
    if (!cat_raw_strs.empty()) {
        if (static_cast<int>(cat_raw_strs.size()) != p)
            throw std::invalid_argument(
                "cat_raw_strs size mismatch: got " +
                std::to_string(cat_raw_strs.size()) + ", expected " + std::to_string(p));
    }

    // ── MinMax params for float columns ──────────────────────────────────────
    std::vector<double> col_min(p, 0.0), col_range(p, 1.0);
    for (int j = 0; j < p; j++) {
        if (is_cat[j] || is_int[j]) continue;
        double cmin = std::numeric_limits<double>::max();
        double cmax = std::numeric_limits<double>::lowest();
        for (int r = 0; r < n; r++) {
            double v = Xb(r, j);
            if (v < cmin) cmin = v;
            if (v > cmax) cmax = v;
        }
        col_min[j]   = cmin;
        col_range[j] = (cmax > cmin) ? (cmax - cmin) : 1.0;
    }

    // ── disc_flat: flat row-major discretisation matrix (local only) ──────────
    // Layout: disc_flat[r * p + j] = bin index for row r, column j.
    // Kept local to this function; NOT stored in TransactionDataCpp (saves O(n*p) mem).
    std::vector<int32_t> disc_flat(static_cast<size_t>(n) * p, 0);

    // ── Pass 1 : Discretise / encode each column ─────────────────────────────
    std::vector<int>                  nb_col;
    std::vector<std::vector<double>>  ber, all_edges;
    std::vector<double>               cv;
    std::vector<std::vector<std::string>>            cat_categories(p);
    // cat_corr is local — pb-sign per category, only used for IU computation.
    std::vector<std::unordered_map<std::string, double>> cat_corr(p);
    std::vector<std::vector<std::pair<double, double>>>  bro(p);

    nb_col.reserve(p);
    ber.reserve(p);
    all_edges.reserve(p);
    cv.reserve(p);

    for (int j = 0; j < p; j++) {
        if (is_cat[j]) {
            // ── Categorical ──────────────────────────────────────────────────
            const bool have_cat = (!cat_raw_strs.empty() &&
                                   !cat_raw_strs[j].empty());
            const bool have_valid = (!cat_raw_valid.empty() &&
                                     !cat_raw_valid[j].empty());

            std::set<std::string> uniq_set;
            if (have_cat) {
                for (int r = 0; r < n; r++)
                    if (have_valid && cat_raw_valid[j][r])
                        uniq_set.insert(cat_raw_strs[j][r]);
            }
            std::vector<std::string> uniq(uniq_set.begin(), uniq_set.end());
            cat_categories[j] = uniq;

            std::unordered_map<std::string, int> label2int;
            label2int.reserve(uniq.size());
            for (int i = 0; i < static_cast<int>(uniq.size()); i++)
                label2int[uniq[i]] = i;

            for (int r = 0; r < n; r++) {
                int32_t code = -1;
                if (have_cat && have_valid && cat_raw_valid[j][r]) {
                    auto it = label2int.find(cat_raw_strs[j][r]);
                    if (it != label2int.end()) code = static_cast<int32_t>(it->second);
                }
                disc_flat[static_cast<size_t>(r) * p + j] = code;
            }

            // Point-biserial sign per label for IU
            std::unordered_map<std::string, double> pb_sign;
            pb_sign.reserve(uniq.size());
            for (auto& v : uniq) {
                int code = label2int.at(v);
                std::vector<double> x_bin_d(n, 0.0), y_d(n);
                for (int r = 0; r < n; r++) {
                    x_bin_d[r] = (disc_flat[static_cast<size_t>(r) * p + j] == code) ? 1.0 : 0.0;
                    y_d[r]     = static_cast<double>(y_vec[r]);
                }
                pb_sign[v] = pearson_cpp(x_bin_d, y_d);
            }
            cat_corr[j] = std::move(pb_sign);

            nb_col.push_back(static_cast<int>(uniq.size()));
            all_edges.push_back({0.0, 1.0});
            ber.push_back({1.0});
            bro[j] = {{0.0, 1.0}};
            cv.push_back(0.0);

        } else if (is_int[j]) {
            // ── Integer : bin on raw values, no MinMax scaling ────────────────
            std::vector<double> col_raw(n);
            for (int r = 0; r < n; r++) col_raw[r] = Xb(r, j);

            std::set<double> dset(col_raw.begin(), col_raw.end());
            int distinct = static_cast<int>(dset.size());
            int nb       = choose_nb_cpp(col_raw, y_vec, n_cls, B, distinct);

            auto [binned, edges] = kbins_cpp(col_raw, nb);
            int nb_act = static_cast<int>(edges.size()) - 1;

            for (int r = 0; r < n; r++)
                disc_flat[static_cast<size_t>(r) * p + j] = static_cast<int32_t>(binned[r]);
            nb_col.push_back(nb_act);
            all_edges.push_back(edges);

            double mx = *std::max_element(edges.begin(), edges.end());
            if (mx <= 0.0) mx = 1.0;
            std::vector<double> ber_j;
            ber_j.reserve(edges.size() - 1);
            for (int bi = 1; bi < static_cast<int>(edges.size()); bi++)
                ber_j.push_back(edges[bi] / mx);
            ber.push_back(ber_j);

            std::vector<std::pair<double, double>> bro_j;
            bro_j.reserve(nb_act);
            for (int bi = 0; bi < nb_act; bi++)
                bro_j.push_back({edges[bi], edges[bi + 1]});
            bro[j] = bro_j;

            std::vector<double> df(n), yf(n);
            for (int r = 0; r < n; r++) { df[r] = binned[r]; yf[r] = y_vec[r]; }
            cv.push_back(pearson_cpp(df, yf));

        } else {
            // ── Float : MinMax-scale then bin ────────────────────────────────
            std::vector<double> col_sc(n);
            for (int r = 0; r < n; r++)
                col_sc[r] = (Xb(r, j) - col_min[j]) / col_range[j];

            std::set<double> dset(col_sc.begin(), col_sc.end());
            int distinct = static_cast<int>(dset.size());
            int nb       = choose_nb_cpp(col_sc, y_vec, n_cls, B, distinct);

            auto [binned, edges] = kbins_cpp(col_sc, nb);
            int nb_act = static_cast<int>(edges.size()) - 1;

            for (int r = 0; r < n; r++)
                disc_flat[static_cast<size_t>(r) * p + j] = static_cast<int32_t>(binned[r]);
            nb_col.push_back(nb_act);
            all_edges.push_back(edges);

            double mx = *std::max_element(edges.begin(), edges.end());
            if (mx <= 0.0) mx = 1.0;
            std::vector<double> ber_j;
            ber_j.reserve(edges.size() - 1);
            for (int bi = 1; bi < static_cast<int>(edges.size()); bi++)
                ber_j.push_back(edges[bi] / mx);
            ber.push_back(ber_j);

            // Inverse-transform edges to original scale for labels
            std::vector<std::pair<double, double>> bro_j;
            bro_j.reserve(nb_act);
            for (int bi = 0; bi < nb_act; bi++) {
                double lo_o = edges[bi]     * col_range[j] + col_min[j];
                double hi_o = edges[bi + 1] * col_range[j] + col_min[j];
                bro_j.push_back({lo_o, hi_o});
            }
            bro[j] = bro_j;

            std::vector<double> df(n), yf(n);
            for (int r = 0; r < n; r++) { df[r] = binned[r]; yf[r] = y_vec[r]; }
            cv.push_back(pearson_cpp(df, yf));
        }
    }

    // ── Pass 2 : Build item registry (EU × IU per bin/category) ──────────────
    std::unordered_map<int, std::string> item_map;
    std::unordered_map<int, int>         bn2id;
    std::unordered_set<int>              colnew_set;

    std::unordered_map<int64_t, double> tu;
    std::vector<double>                 tu_y(key_stride, 0.0);
    int ic = 0;

    for (int j = 0; j < p; j++) {
        int nb = nb_col[j];
        if (is_cat[j]) {
            for (int idx = 0; idx < static_cast<int>(cat_categories[j].size()); idx++) {
                int bi = idx + 1;
                std::vector<int> x_bin(n, 0);
                for (int r = 0; r < n; r++)
                    if (disc_flat[static_cast<size_t>(r) * p + j] == static_cast<int32_t>(idx))
                        x_bin[r] = 1;

                double nmi = nmi_binary_cpp(x_bin, y_vec, n_cls);
                const std::string& v = cat_categories[j][idx];
                double pb = 0.0;
                auto it = cat_corr[j].find(v);
                if (it != cat_corr[j].end()) pb = it->second;
                double iu_t = (pb > 0.0) ? 1.0 : 0.05;
                double eiu  = nmi * iu_t;

                int bname = bk(bi, j);
                if (eiu > 0.0) {
                    ic++;
                    item_map[ic] = names[j] + "=" + v;
                    colnew_set.insert(bname);
                    bn2id[bname] = ic;
                }
                for (int yi = 0; yi < n_cls; yi++) {
                    int64_t k = static_cast<int64_t>(bname) * key_stride + yi;
                    tu[k] = eiu;
                    if (eiu > tu_y[yi]) tu_y[yi] = eiu;
                }
            }
        } else {
            double eu = std::abs(cv[j]);
            for (int bi = 1; bi <= nb; bi++) {
                double iu_t = (cv[j] >= 0.0) ? ber[j][bi - 1] : ber[j][nb - bi];
                double eiu  = eu * iu_t;

                int bname = bk(bi, j);
                if (eiu > 0.0) {
                    ic++;
                    auto& [lo, hi] = bro[j][bi - 1];
                    std::ostringstream oss;
                    if (is_int[j]) {
                        oss << names[j] << "=[" << static_cast<int>(lo)
                            << "," << static_cast<int>(hi) << "]";
                    } else {
                        oss << names[j] << "=["
                            << std::fixed << std::setprecision(3) << lo
                            << ","
                            << std::fixed << std::setprecision(3) << hi << "]";
                    }
                    item_map[ic] = oss.str();
                    colnew_set.insert(bname);
                    bn2id[bname] = ic;
                }
                for (int yi = 0; yi < n_cls; yi++) {
                    int64_t k = static_cast<int64_t>(bname) * key_stride + yi;
                    tu[k] = eiu;
                    if (eiu > tu_y[yi]) tu_y[yi] = eiu;
                }
            }
        }
    }

    // Normalise utility per class
    for (auto& kv : tu) {
        int yi = static_cast<int>(kv.first % key_stride);
        kv.second = (tu_y[yi] > 0.0) ? kv.second / tu_y[yi] : 0.0;
    }

    // ── Pass 3 : Build transaction list ──────────────────────────────────────
    TransList           transactions;
    std::vector<double> item_twu(ic, 0.0);
    std::vector<double> RIU(ic, 0.0);
    transactions.reserve(n);

    for (int r = 0; r < n; r++) {
        int yi = y_vec[r];
        double tutils = 0.0;
        Trans  trans;

        for (int j = 0; j < p; j++) {
            int32_t disc_val = disc_flat[static_cast<size_t>(r) * p + j];
            int bi;
            if (is_cat[j]) {
                if (disc_val < 0) continue;
                bi = static_cast<int>(disc_val) + 1;
            } else {
                bi = static_cast<int>(disc_val) + 1;
            }

            int bname = bk(bi, j);
            if (colnew_set.find(bname) == colnew_set.end()) continue;

            int64_t txk = static_cast<int64_t>(bname) * key_stride + yi;
            auto tit = tu.find(txk);
            if (tit == tu.end()) continue;

            // Round to 6 d.p. (matches Python round(tu[txk], 6))
            double iu = std::round(tit->second * 1e6) / 1e6;
            int iid   = bn2id.at(bname);
            trans.push_back({iid, iu});
            tutils += iu;
        }

        if (tutils > 0.0) {
            for (auto& [iid, iu] : trans) {
                item_twu[iid - 1] += tutils;
                RIU[iid - 1]      += iu;
            }
            transactions.push_back(std::move(trans));
        } else {
            transactions.push_back({{-1, 0.0}});
        }
    }

    // ── Assemble result ──────────────────────────────────────────────────────
    TransactionDataCpp td;
    td.transactions    = std::move(transactions);
    td.item_twu        = std::move(item_twu);
    td.item_map        = std::move(item_map);
    td.RIU             = std::move(RIU);
    td.disc_n          = n;
    td.disc_p          = p;
    td.bn2id           = std::move(bn2id);
    td.colnew_set      = std::move(colnew_set);
    td.nb_col          = std::move(nb_col);
    td.ber             = std::move(ber);
    td.cv              = std::move(cv);
    td.all_edges       = std::move(all_edges);
    td.col_min         = std::move(col_min);
    td.col_range       = std::move(col_range);
    td.is_cat_v        = std::move(is_cat);
    td.is_int_v        = std::move(is_int);
    td.cat_categories  = std::move(cat_categories);
    td.bkey_stride     = bkey_stride;
    return td;
}

}  // namespace hugiml

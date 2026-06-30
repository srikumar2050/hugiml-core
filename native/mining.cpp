/**
 * mining.cpp — Pattern mining implementation
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Implements the top-K High Utility Itemset (HUI) miner with information-gain
 * filtering.  Uses LIU pruning, EUCS pruning (for L > 1), and an IG threshold.
 *
 * Graceful timeout: explore() checks a wall-clock deadline every
 * _timeout_check_interval recursive calls and throws std::runtime_error
 * ("mining_timeout") if the deadline is exceeded.  The Python layer
 * (_mine_with_fallback) catches this and retries with reduced parameters.
 */

#include "mining.hpp"
#include "mining_l1.hpp"
#include "mining_l2.hpp"
#include "math.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <limits>
#include <string>
#include <cmath>

namespace hugiml {

namespace {

constexpr size_t kEucsMaxCellsDefault = 6000000ULL;
constexpr int kEucsMinItemsDefault = 32;
constexpr double kEucsMaxDensityDefault = 0.20;

static std::string eucs_env_lower(const char* value) {
    std::string out(value ? value : "");
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

static bool eucs_env_enabled() {
    const char* raw = std::getenv("HUGIML_EUCS_ENABLE");
    if (!raw) raw = std::getenv("HUGIML_EUCS_ENABLED");
    if (!raw || !*raw) return true;
    const std::string value = eucs_env_lower(raw);
    if (value == "0" || value == "false" || value == "no" || value == "off" ||
        value == "disable" || value == "disabled") {
        return false;
    }
    if (value == "1" || value == "true" || value == "yes" || value == "on" ||
        value == "enable" || value == "enabled") {
        return true;
    }
    // Invalid values preserve the default rather than surprising the caller.
    return true;
}

static int eucs_env_int(const char* name, int fallback, int min_value) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    try {
        size_t idx = 0;
        long value = std::stol(raw, &idx, 10);
        if (idx != std::string(raw).size() || value < min_value ||
            value > static_cast<long>(std::numeric_limits<int>::max())) {
            return fallback;
        }
        return static_cast<int>(value);
    } catch (...) {
        return fallback;
    }
}

static size_t eucs_env_size(const char* name, size_t fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    try {
        size_t idx = 0;
        unsigned long long value = std::stoull(raw, &idx, 10);
        if (idx != std::string(raw).size()) return fallback;
        return static_cast<size_t>(value);
    } catch (...) {
        return fallback;
    }
}

static double eucs_env_double(const char* name, double fallback, double min_value) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    try {
        size_t idx = 0;
        double value = std::stod(raw, &idx);
        if (idx != std::string(raw).size() || value < min_value) return fallback;
        return value;
    } catch (...) {
        return fallback;
    }
}

static int eucs_min_items_config() {
    return eucs_env_int("HUGIML_EUCS_MIN_ITEMS", kEucsMinItemsDefault, 0);
}

static size_t eucs_max_cells_config() {
    return eucs_env_size("HUGIML_EUCS_MAX_CELLS", kEucsMaxCellsDefault);
}

static double eucs_max_density_config() {
    return eucs_env_double("HUGIML_EUCS_MAX_DENSITY", kEucsMaxDensityDefault, 0.0);
}

}  // namespace

// ── UL methods ───────────────────────────────────────────────────────────────

void UL::add(int t, double i, double r) {
    sI += i;
    sR += r;
    els.push_back({t, i, r});
}

void UL::seal() {
    if (sealed) return;
    size_t m = els.size();
    tid_arr.resize(m);
    iu_arr.resize(m);
    ru_arr.resize(m);
    for (size_t k = 0; k < m; k++) {
        tid_arr[k] = els[k].tid;
        iu_arr[k]  = els[k].iu;
        ru_arr[k]  = els[k].ru;
    }
    sealed = true;
}

void UL::seal_from_arrays(std::vector<int32_t>&& ta,
                           std::vector<double>&&  ia,
                           std::vector<double>&&  ra) {
    tid_arr = std::move(ta);
    iu_arr  = std::move(ia);
    ru_arr  = std::move(ra);
    sI = 0.0;
    sR = 0.0;
    for (double v : iu_arr) sI += v;
    for (double v : ru_arr) sR += v;
    // Build els for compute_ig (minimal: just tid + iu needed for IG entropy)
    els.clear();
    els.reserve(tid_arr.size());
    for (size_t k = 0; k < tid_arr.size(); k++)
        els.push_back({static_cast<int>(tid_arr[k]), iu_arr[k], ru_arr[k]});
    sealed = true;
}

static double entropy_from_counts(const std::vector<int>& counts, int total) {
    if (total <= 0) return 0.0;
    double h = 0.0;
    const double inv = 1.0 / static_cast<double>(total);
    for (int c : counts) {
        if (c > 0) {
            double p = static_cast<double>(c) * inv;
            h -= p * std::log(p);
        }
    }
    return std::max(h, 0.0);
}

static double information_gain_from_child_parent_counts(
    const std::vector<int>& child_counts,
    int child_total,
    const std::vector<int>& parent_counts,
    int parent_total
) {
    if (child_total <= 0 || parent_total <= 0 || child_total > parent_total) {
        return 0.0;
    }
    const int n_out = parent_total - child_total;
    if (n_out == 0) {
        // Match UL::compute_ig: a pattern covering the complete parent
        // population is not considered informative for the G gate.
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<int> out_counts(parent_counts.size(), 0);
    for (size_t k = 0; k < parent_counts.size(); ++k) {
        const int child = (k < child_counts.size()) ? child_counts[k] : 0;
        out_counts[k] = parent_counts[k] - child;
    }
    const double base = entropy_from_counts(parent_counts, parent_total);
    const double cond = (static_cast<double>(child_total) / parent_total)
            * entropy_from_counts(child_counts, child_total)
        + (static_cast<double>(n_out) / parent_total)
            * entropy_from_counts(out_counts, n_out);
    return base - cond;
}

void UL::compute_ig(const UL* parent,
                     const std::vector<int>& y_arr, int n_cls) {
    // Use sealed tid_arr throughout — els may have been released already.
    if (tid_arr.empty()) { ig = 0.0; return; }

    // Build parent population counts first — needed for base entropy and cnt_out.
    const std::vector<int32_t>* parent_tids = nullptr;
    int n_parent = 0;
    if (parent == nullptr) {
        n_parent = static_cast<int>(y_arr.size());
    } else {
        if (parent->tid_arr.empty()) { ig = 0.0; return; }
        parent_tids = &parent->tid_arr;
        n_parent    = static_cast<int>(parent->tid_arr.size());
    }
    if (n_parent == 0) { ig = 0.0; return; }

    std::vector<int> cnt_parent(n_cls, 0);
    if (parent == nullptr) {
        for (int lbl : y_arr)
            if (lbl >= 0 && lbl < n_cls) cnt_parent[lbl]++;
    } else {
        for (int32_t tid : *parent_tids)
            if (y_arr[tid] >= 0 && y_arr[tid] < n_cls) cnt_parent[y_arr[tid]]++;
    }
    double base = entropy_from_counts(cnt_parent, n_parent);

    // cnt_in: class counts for tids covered by this pattern.
    std::vector<int> cnt_in(n_cls, 0);
    for (int32_t tid : tid_arr)
        if (y_arr[tid] >= 0 && y_arr[tid] < n_cls) cnt_in[y_arr[tid]]++;

    int n_in  = static_cast<int>(tid_arr.size());
    int n_out = n_parent - n_in;

    if (n_out == 0) {
        // Pattern covers the entire parent population.
        ig = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    // cnt_out = cnt_parent - cnt_in (no set-membership scan needed).
    std::vector<int> cnt_out(n_cls, 0);
    for (int k = 0; k < n_cls; k++) cnt_out[k] = cnt_parent[k] - cnt_in[k];

    double ce = (static_cast<double>(n_in)  / n_parent * entropy_from_counts(cnt_in,  n_in)
               + static_cast<double>(n_out) / n_parent * entropy_from_counts(cnt_out, n_out));
    ig = base - ce;
}

// ── THUIsl methods ───────────────────────────────────────────────────────────

/// Min-heap comparator (smallest utility on top, matching Python heapq).
struct MinHeapCmp {
    bool operator()(const PatternEntry& a, const PatternEntry& b) const {
        return a.utility > b.utility;
    }
};

struct IgMinHeapCmp {
    bool operator()(const PatternEntry& a, const PatternEntry& b) const {
        return a.ig > b.ig;
    }
};

void THUIsl::save(const std::vector<int>& items, const UL& ul) {
    // interaction_relaxed_mining: a pattern whose root (items[0]) is a
    // relaxed item is routed ONLY to relaxed_heap (IG-ranked, independent
    // K-sized budget), never into the ordinary utility-ranked heap -- this
    // mirrors mine_patterns_l2_augmented_patterns_v2_cpp's mutually
    // exclusive routing so a single pattern cannot occupy a slot in both
    // pools. When relaxed_cols is empty this check is false for every item
    // and behavior is identical to the unmodified single-heap path.
    if (!items.empty() && is_relaxed_root(items.front())) {
        save_by_ig(items, ul);
        return;
    }
    double u = ul.sI;
    PatternEntry pe{u, items, ul.ig};
    if (static_cast<int>(heap.size()) < K) {
        heap.push_back(pe);
        std::push_heap(heap.begin(), heap.end(), MinHeapCmp{});
        if (static_cast<int>(heap.size()) == K) minU = heap.front().utility;
    } else if (u > minU) {
        std::pop_heap(heap.begin(), heap.end(), MinHeapCmp{});
        heap.back() = pe;
        std::push_heap(heap.begin(), heap.end(), MinHeapCmp{});
        minU = heap.front().utility;
    }
}

void THUIsl::save_by_ig(const std::vector<int>& items, const UL& ul) {
    PatternEntry pe{ul.sI, items, ul.ig};
    if (static_cast<int>(relaxed_heap.size()) < K) {
        relaxed_heap.push_back(pe);
        std::push_heap(relaxed_heap.begin(), relaxed_heap.end(), IgMinHeapCmp{});
        if (static_cast<int>(relaxed_heap.size()) == K) minIg = relaxed_heap.front().ig;
    } else if (ul.ig > minIg) {
        std::pop_heap(relaxed_heap.begin(), relaxed_heap.end(), IgMinHeapCmp{});
        relaxed_heap.back() = pe;
        std::push_heap(relaxed_heap.begin(), relaxed_heap.end(), IgMinHeapCmp{});
        minIg = relaxed_heap.front().ig;
    }
}

UL THUIsl::child_ul(const UL* prefix_ul, const UL& px_ul, const UL& py_ul,
                         bool allow_utility_abandon) {
    // Construct the utility list for Pxy from utility lists for Px and Py.
    // For a 2-item extension (P is empty), utility is u(x)+u(y).
    // For deeper extensions, subtract the shared prefix utility u(P) so the
    // already-counted prefix is not added twice.
    UL c(py_ul.item);
    if (px_ul.tid_arr.empty() || py_ul.tid_arr.empty()) return c;

    const auto& xt = px_ul.tid_arr;
    const auto& yt = py_ul.tid_arr;
    std::vector<int32_t> c_tid;
    std::vector<double>  c_iu, c_ru;
    c_tid.reserve(std::min(xt.size(), yt.size()));
    c_iu.reserve(c_tid.capacity());
    c_ru.reserve(c_tid.capacity());

    // LA-style early abandonment: when Px entries are skipped because Py is
    // absent, the remaining upper bound for any Pxy descendant falls by
    // iu(Px,t)+ru(Px,t).  Once that bound is below the current utility floor,
    // this child cannot produce a retained pattern.  Relaxed-root branches opt
    // out because their separate IG-ranked track intentionally ignores minU.
    double remaining_bound = px_ul.sI + px_ul.sR;
    auto drop_px_entry = [&](size_t pos) -> bool {
        if (!allow_utility_abandon) return false;
        remaining_bound -= px_ul.iu_arr[pos] + px_ul.ru_arr[pos];
        return remaining_bound < minU;
    };

    size_t xi = 0, yi = 0, pi = 0;
    const auto* pt = prefix_ul ? &prefix_ul->tid_arr : nullptr;
    while (xi < xt.size() && yi < yt.size()) {
        if (xt[xi] < yt[yi]) {
            if (drop_px_entry(xi)) return UL(py_ul.item);
            ++xi;
            continue;
        }
        if (yt[yi] < xt[xi]) { ++yi; continue; }

        const int32_t tid = xt[xi];
        double prefix_iu = 0.0;
        if (prefix_ul != nullptr) {
            while (pi < pt->size() && (*pt)[pi] < tid) ++pi;
            if (pi == pt->size() || (*pt)[pi] != tid) {
                ++xi; ++yi;
                continue;
            }
            prefix_iu = prefix_ul->iu_arr[pi];
        }

        c_tid.push_back(tid);
        c_iu.push_back(px_ul.iu_arr[xi] + py_ul.iu_arr[yi] - prefix_iu);
        c_ru.push_back(py_ul.ru_arr[yi]);
        ++xi; ++yi;
    }
    while (xi < xt.size()) {
        if (drop_px_entry(xi)) return UL(py_ul.item);
        ++xi;
    }
    if (c_tid.empty()) return c;
    c.seal_from_arrays(std::move(c_tid), std::move(c_iu), std::move(c_ru));
    return c;
}


bool THUIsl::same_feature(int item_a, int item_b) const {
    if (item_col == nullptr) return false;
    if (item_a <= 0 || item_b <= 0) return false;
    const size_t ia = static_cast<size_t>(item_a - 1);
    const size_t ib = static_cast<size_t>(item_b - 1);
    if (ia >= item_col->size() || ib >= item_col->size()) return false;
    return (*item_col)[ia] == (*item_col)[ib];
}

bool THUIsl::candidate_conflicts_with_prefix(const std::vector<int>& prefix,
                                             int prefix_item,
                                             int candidate_item) const {
    if (item_col == nullptr) return false;
    if (same_feature(prefix_item, candidate_item)) return true;
    for (int existing : prefix) {
        if (same_feature(existing, candidate_item)) return true;
    }
    return false;
}

void THUIsl::mine(const TransList& transactions,
                   const std::vector<double>& item_twu,
                   const std::vector<int>& ytrain,
                   int n_cls,
                   const std::vector<int>* item_col_in,
                   const std::vector<double>* item_iu_in,
                   const std::vector<std::vector<double>>* transaction_utils_in) {
    item_col = (item_col_in && item_col_in->size() >= item_twu.size())
        ? item_col_in : nullptr;
    item_iu = (item_iu_in && item_iu_in->size() >= item_twu.size())
        ? item_iu_in : nullptr;
    transaction_utils = (transaction_utils_in && !transaction_utils_in->empty())
        ? transaction_utils_in : nullptr;

    // minU is seeded by mine_patterns_cpp (td.riu_thresh(K)) before this call.
    // Do NOT reset it here — only clear the heap and the recursion counter.
    heap.clear();
    // relaxed_heap/minIg ARE reset here (unlike minU/heap, there is no
    // external seeding contract for them) so that reusing a THUIsl
    // instance across multiple mine() calls cannot leak relaxed-heap
    // entries from a previous run into a new one. No current call site
    // reuses a THUIsl instance, but nothing in the public API prevents it.
    relaxed_heap.clear();
    minIg = -std::numeric_limits<double>::infinity();
    _explore_calls = 0;
    int  n_items  = static_cast<int>(item_twu.size());
    bool use_eucs = (L != 1);
    const int maxd_for_leaf = (L == -1 || L == 0) ? 99 : L;
    const bool use_leaf_lb = (L == -1 || L == 0);
    // LIU-Exact is safe in classifier mining only after the contiguous
    // candidate is validated against the same source-feature, relaxed-root,
    // and information-gain gates that the utility-list search will apply.
    // LIU-LB remains restricted to legacy leaf/pure-utility mode because a
    // lower-bound descendant does not carry exact coverage for the IG gate.
    bool use_leaf = (L != 1);

    FMap fmap;

    struct LeafInfo {
        double utility = 0.0;
        int support = 0;
        std::vector<int> class_counts;
    };

    // leaf_map_by_end[end_pos][start_pos] stores exact utility and coverage
    // counts for the consecutive item run [start_pos .. end_pos] in
    // sorted_items order.  A vector for the outer dimension avoids hashing the
    // leaf endpoint on every transaction-row update.
    using LeafMapByEnd = std::vector<std::unordered_map<int, LeafInfo>>;
    LeafMapByEnd leaf_map_by_end;

    // Build utility-list map for items that pass the initial threshold.
    // interaction_relaxed_mining: items whose column is in relaxed_cols are
    // admitted regardless of the TWU floor (root-only relaxation; see the
    // THUIsl::relaxed_cols comment in mining.hpp). When relaxed_cols is
    // empty this is identical to the unmodified filter below.
    std::unordered_map<int, UL> ul_map;
    ul_map.reserve(n_items);
    for (int iid = 1; iid <= n_items; iid++)
        if (item_twu[iid - 1] >= minU || is_relaxed_root(iid))
            ul_map.emplace(iid, UL(iid));

    // Sorted by TWU ascending (same as Python sorted_items)
    std::vector<int> sorted_items;
    sorted_items.reserve(ul_map.size());
    for (auto& kv : ul_map) sorted_items.push_back(kv.first);
    std::sort(sorted_items.begin(), sorted_items.end(),
              [&](int a, int b) {
                  double ta = item_twu[a - 1], tb = item_twu[b - 1];
                  return (ta < tb) || (ta == tb && a < b);
              });

    // pre-compute item_rank[id] = position in sorted_items.
    // Built unconditionally because it replaces the per-transaction TWU sort
    // comparator with a dense integer lookup.
    std::vector<int> item_rank(static_cast<size_t>(n_items) + 1, -1);
    for (int pos = 0; pos < static_cast<int>(sorted_items.size()); pos++)
        item_rank[static_cast<size_t>(sorted_items[pos])] = pos;

    if (use_leaf) leaf_map_by_end.resize(sorted_items.size());

    // EUCS is quadratic in the number of admitted items and pays off when the
    // pair co-occurrence space is sparse enough to prune many intersections.
    // Gate it by both cell count and observed active-item density so dense
    // workloads keep the original utility-list path without the extra O(|T|^2)
    // scan work.  Skipping EUCS never changes completeness.
    eucs_enabled = false;
    eucs_m = static_cast<int>(sorted_items.size());
    eucs_pos_for_item.assign(static_cast<size_t>(n_items) + 1, -1);
    eucs_twu.clear();
    if (use_eucs && eucs_env_enabled() && eucs_m > 1) {
        const int eucs_min_items = eucs_min_items_config();
        const size_t eucs_max_cells = eucs_max_cells_config();
        const double eucs_max_density = eucs_max_density_config();
        if (eucs_m <= eucs_min_items) {
            // Leave EUCS disabled: small item universes are already cheap to
            // intersect, and the density probe/cache build can dominate.
        } else {
            size_t active_total = 0;
            size_t active_rows = 0;
            for (const Trans& trans : transactions) {
                if (trans.size() == 1 && trans[0] == -1) continue;
                size_t active_count = 0;
                for (int it : trans) {
                    if (it > 0 && static_cast<size_t>(it) < item_rank.size()
                        && item_rank[static_cast<size_t>(it)] >= 0) {
                        ++active_count;
                    }
                }
                if (active_count > 0) {
                    active_total += active_count;
                    ++active_rows;
                }
            }
            const double active_density = (active_rows == 0)
                ? 1.0
                : (static_cast<double>(active_total) / static_cast<double>(active_rows))
                    / static_cast<double>(eucs_m);
            const size_t cells = (static_cast<size_t>(eucs_m) * static_cast<size_t>(eucs_m - 1)) / 2;
            if (cells <= eucs_max_cells && active_density <= eucs_max_density) {
                for (int pos = 0; pos < eucs_m; ++pos)
                    eucs_pos_for_item[static_cast<size_t>(sorted_items[static_cast<size_t>(pos)])] = pos;
                eucs_twu.assign(cells, 0.0);
                eucs_enabled = true;
            }
        }
    }

    // Single pass: populate ULs and EUCS map
    for (int tid = 0; tid < static_cast<int>(transactions.size()); tid++) {
        // Deadline check during EUCS construction — this loop dominates
        // wall-clock time for large datasets before explore() is reached.
        if (has_deadline) {
            if ((++_explore_calls & (_timeout_check_interval - 1)) == 0) {
                if (Clock::now() >= deadline_tp)
                    throw std::runtime_error("mining_timeout");
            }
        }
        const Trans& trans = transactions[tid];
        if (trans.size() == 1 && trans[0] == -1) continue;

        std::vector<std::pair<int, double>> active;
        active.reserve(trans.size());
        for (size_t pos = 0; pos < trans.size(); ++pos) {
            int it = trans[pos];
            double u = 0.0;
            if (transaction_utils &&
                static_cast<size_t>(tid) < transaction_utils->size() &&
                pos < (*transaction_utils)[static_cast<size_t>(tid)].size()) {
                u = (*transaction_utils)[static_cast<size_t>(tid)][pos];
            } else if (item_iu && it > 0 && static_cast<size_t>(it - 1) < item_iu->size()) {
                u = (*item_iu)[static_cast<size_t>(it - 1)];
            }
            if (ul_map.count(it)) active.push_back({it, u});
        }
        if (active.empty()) continue;

        double new_twu = 0.0;
        for (auto& [it, u] : active) new_twu += u;

        // sort by pre-computed rank (O(1) lookup) rather than
        // re-fetching item_twu on every comparison inside the sort.
        // item_rank[id] == position of id in sorted_items; since sorted_items
        // was built with the same comparator
        std::sort(active.begin(), active.end(),
                  [&](auto& a, auto& b) {
                      return item_rank[a.first] < item_rank[b.first];
                  });

        double rem = 0.0;
        for (int i = static_cast<int>(active.size()) - 1; i >= 0; i--) {
            int it = active[i].first;
            double u = active[i].second;
            ul_map.at(it).add(tid, u, rem);
            if (eucs_enabled) {
                const int it_pos = item_rank[static_cast<size_t>(it)];
                for (int j2 = i + 1; j2 < static_cast<int>(active.size()); j2++) {
                    int oj = active[j2].first;
                    if (oj == it || same_feature(it, oj)) continue;
                    const int oj_pos = item_rank[static_cast<size_t>(oj)];
                    if (it_pos < 0 || oj_pos < 0 || it_pos == oj_pos) continue;
                    const size_t idx = eucs_index_from_pos(it_pos, oj_pos);
                    eucs_twu[idx] += new_twu;
                }
            }
            // ── Leaf-map update ──────────────────────
            // For each item, walk backward through the transaction's active
            // items that are consecutive in sorted_items order, accumulating
            // cumulative utilities.  Only the consecutive run is tracked.
            if (use_leaf) {
                int end_pos = item_rank[static_cast<size_t>(it)];
                const int lbl = (static_cast<size_t>(tid) < ytrain.size())
                    ? ytrain[static_cast<size_t>(tid)]
                    : -1;
                const bool lbl_valid = (lbl >= 0 && lbl < n_cls);
                if (end_pos >= 0 && static_cast<size_t>(end_pos) < leaf_map_by_end.size()) {
                    auto& leaf_entry = leaf_map_by_end[static_cast<size_t>(end_pos)];
                    double cutil = u;
                    int follow_pos = end_pos;
                    for (int j = i - 1; j >= 0; j--) {
                        int prec = active[j].first;
                        int prec_pos = item_rank[static_cast<size_t>(prec)];
                        if (prec_pos != follow_pos - 1) break;
                        follow_pos = prec_pos;
                        cutil += active[j].second;
                        auto& info = leaf_entry[follow_pos];
                        info.utility += cutil;
                        info.support += 1;
                        if (lbl_valid) {
                            if (info.class_counts.empty()) info.class_counts.assign(n_cls, 0);
                            info.class_counts[static_cast<size_t>(lbl)] += 1;
                        }
                    }
                }
            }
            rem += u;
        }
    }


    // Seal all 1-item ULs, compute IG, then release transient els buffers.
    // compute_ig uses tid_arr, so recursive child IG remains correct after release.
    for (auto& kv : ul_map) kv.second.seal();
    for (auto& kv : ul_map) {
        kv.second.compute_ig(nullptr, ytrain, n_cls);
        kv.second.release_els();  // reclaim els memory before recursion
    }

    // ── Leaf raise ───────────────────────────────
    // LIU-Exact can safely raise the ordinary utility floor only with exact
    // candidates that would pass the same classifier gates as the recursive
    // utility-list search.  For bounded L>=2 classifier mining, this block
    // therefore checks same-source conflicts, relaxed-root routing, singleton
    // IG gates, and the exact contiguous-prefix IG gates before contributing a
    // LIU value to the K-sized threshold heap.  LIU-LB is kept only for the
    // legacy leaf/pure-utility mode because lower-bound descendants do not
    // carry exact coverage information for the classifier IG gate.
    if (use_leaf && !leaf_map_by_end.empty()) {
        std::vector<double> leaf_utils;
        leaf_utils.reserve(static_cast<size_t>(K) + 1);

        auto add_to_leaf = [&](double val) {
            if (val < minU) return;
            leaf_utils.push_back(val);
            std::push_heap(leaf_utils.begin(), leaf_utils.end(),
                           std::greater<double>{});
            if (static_cast<int>(leaf_utils.size()) > K) {
                std::pop_heap(leaf_utils.begin(), leaf_utils.end(),
                              std::greater<double>{});
                leaf_utils.pop_back();
            }
        };

        std::vector<int> global_counts(n_cls, 0);
        for (int lbl : ytrain)
            if (lbl >= 0 && lbl < n_cls) global_counts[static_cast<size_t>(lbl)]++;
        const int global_support = static_cast<int>(ytrain.size());

        std::vector<std::vector<int>> singleton_counts(sorted_items.size(), std::vector<int>(n_cls, 0));
        std::vector<int> singleton_support(sorted_items.size(), 0);
        for (int pos = 0; pos < static_cast<int>(sorted_items.size()); ++pos) {
            const int iid = sorted_items[static_cast<size_t>(pos)];
            const UL& ul = ul_map.at(iid);
            singleton_support[static_cast<size_t>(pos)] = static_cast<int>(ul.tid_arr.size());
            for (int32_t tid : ul.tid_arr) {
                if (tid >= 0 && static_cast<size_t>(tid) < ytrain.size()) {
                    const int lbl = ytrain[static_cast<size_t>(tid)];
                    if (lbl >= 0 && lbl < n_cls)
                        singleton_counts[static_cast<size_t>(pos)][static_cast<size_t>(lbl)]++;
                }
            }
        }

        auto leaf_info = [&](int start_pos, int end_pos) -> const LeafInfo* {
            if (end_pos < 0 || start_pos < 0) return nullptr;
            if (static_cast<size_t>(end_pos) >= leaf_map_by_end.size()) return nullptr;
            const auto& by_start = leaf_map_by_end[static_cast<size_t>(end_pos)];
            auto it = by_start.find(start_pos);
            if (it == by_start.end()) return nullptr;
            return &it->second;
        };

        std::unordered_map<long long, double> liu_ig_cache;
        auto liu_key = [](int start_pos, int end_pos) -> long long {
            return (static_cast<long long>(start_pos) << 32)
                ^ static_cast<unsigned int>(end_pos);
        };

        auto liu_ig = [&](int start_pos, int end_pos) -> double {
            const long long key = liu_key(start_pos, end_pos);
            auto cached = liu_ig_cache.find(key);
            if (cached != liu_ig_cache.end()) return cached->second;
            const LeafInfo* child = leaf_info(start_pos, end_pos);
            if (!child) {
                liu_ig_cache[key] = 0.0;
                return 0.0;
            }
            const int length = end_pos - start_pos + 1;
            double out = 0.0;
            if (length == 2) {
                out = information_gain_from_child_parent_counts(
                    child->class_counts, child->support, global_counts, global_support);
            } else if (length == 3) {
                out = information_gain_from_child_parent_counts(
                    child->class_counts,
                    child->support,
                    singleton_counts[static_cast<size_t>(start_pos)],
                    singleton_support[static_cast<size_t>(start_pos)]);
            } else if (length > 3) {
                const LeafInfo* parent = leaf_info(start_pos, end_pos - 2);
                if (parent) {
                    out = information_gain_from_child_parent_counts(
                        child->class_counts, child->support,
                        parent->class_counts, parent->support);
                }
            }
            liu_ig_cache[key] = out;
            return out;
        };

        auto sequence_has_source_conflict = [&](int start_pos, int end_pos) -> bool {
            for (int a = start_pos; a <= end_pos; ++a)
                for (int b = a + 1; b <= end_pos; ++b)
                    if (same_feature(sorted_items[static_cast<size_t>(a)],
                                     sorted_items[static_cast<size_t>(b)]))
                        return true;
            return false;
        };

        auto sequence_passes_classifier_gates = [&](int start_pos, int end_pos) -> bool {
            const int length = end_pos - start_pos + 1;
            if (length < 2 || length > maxd_for_leaf) return false;
            if (sequence_has_source_conflict(start_pos, end_pos)) return false;
            if (is_relaxed_root(sorted_items[static_cast<size_t>(start_pos)])) return false;
            for (int pos = start_pos; pos <= end_pos; ++pos) {
                const UL& ul = ul_map.at(sorted_items[static_cast<size_t>(pos)]);
                if (!(ul.ig > G)) return false;
            }
            for (int prefix_end = start_pos + 1; prefix_end <= end_pos; ++prefix_end) {
                const double ig = liu_ig(start_pos, prefix_end);
                if (!(ig > G)) return false;
            }
            return true;
        };

        // LIU-Exact for bounded classifier mining and legacy utility mode.
        for (int end_pos = 0; end_pos < static_cast<int>(leaf_map_by_end.size()); ++end_pos) {
            for (auto& kv : leaf_map_by_end[static_cast<size_t>(end_pos)]) {
                const int start_pos = kv.first;
                const LeafInfo& info = kv.second;
                if (use_leaf_lb || sequence_passes_classifier_gates(start_pos, end_pos)) {
                    add_to_leaf(info.utility);
                }
            }
        }

        // LIU-LB: legacy leaf/pure-utility mode only.  These lower bounds are
        // utility-safe but do not have exact IG coverage, so they are not used
        // to raise minU for bounded classifier paths.
        if (use_leaf_lb) {
            int n_sorted = static_cast<int>(sorted_items.size());
            for (int end_pos = 0; end_pos < static_cast<int>(leaf_map_by_end.size()); ++end_pos) {
                for (auto& kv : leaf_map_by_end[static_cast<size_t>(end_pos)]) {
                    const int start_pos = kv.first;
                    const double val = kv.second.utility;
                    if (val < minU) continue;
                    for (int i = start_pos + 1; i < end_pos - 1; i++) {
                        if (i >= n_sorted) break;
                        double v1 = val - ul_map.at(sorted_items[static_cast<size_t>(i)]).sI;
                        add_to_leaf(v1);
                        for (int j = i + 1; j < end_pos - 1; j++) {
                            if (j >= n_sorted) break;
                            double v2 = v1 - ul_map.at(sorted_items[static_cast<size_t>(j)]).sI;
                            add_to_leaf(v2);
                            for (int k = j + 1; k + 1 < end_pos - 1; k++) {
                                if (k >= n_sorted) break;
                                add_to_leaf(v2 - ul_map.at(sorted_items[static_cast<size_t>(k)]).sI);
                            }
                        }
                    }
                }
            }

            // All 1-item actual utilities, legacy leaf/pure-utility mode only.
            for (auto& kv : ul_map) add_to_leaf(kv.second.sI);
        }

        if (static_cast<int>(leaf_utils.size()) >= K && leaf_utils.front() > minU)
            minU = leaf_utils.front();

        leaf_map_by_end.clear();
    }

    // Build the top-level UL list without a post-raise sI+sR filter.
    // A low-utility / zero-remaining singleton may still be a high-utility
    // suffix for an earlier item (e.g. the last item in TWU order).  The
    // explore() loop will prune it when it appears as the prefix item, while
    // still allowing earlier prefixes to combine with it as an extension.
    std::vector<UL*> uls;
    uls.reserve(sorted_items.size());
    for (int iid : sorted_items)
        uls.push_back(&ul_map.at(iid));

    explore({}, uls, ytrain, n_cls, 0, fmap, nullptr);
}

void THUIsl::explore(std::vector<int>  prefix,
                      std::vector<UL*>& uls,
                      const std::vector<int>& y_arr,
                      int n_cls, int depth, FMap& fmap,
                      const UL* prefix_ul) {
    int maxd = (L == -1 || L == 0) ? 99 : L;

    for (int i = static_cast<int>(uls.size()) - 1; i >= 0; --i) {
        if (has_deadline) {
            if ((++_explore_calls & (_timeout_check_interval - 1)) == 0) {
                if (Clock::now() >= deadline_tp)
                    throw std::runtime_error("mining_timeout");
            }
        }
        UL* ux = uls[static_cast<size_t>(i)];
        // pat's root is prefix.front() if a prefix already exists (depth
        // >= 1), or ux->item itself if this is the first item appended
        // (depth == 0, prefix is empty). Either way this determines which
        // heap save() will route into; the eligibility condition here must
        // match that routing, or a relaxed-rooted pattern can be skipped
        // before save() is ever called regardless of how save() itself
        // would route it. interaction_relaxed_mining: for a relaxed-rooted
        // pattern, neither ux->ig > G nor the ordinary heap's minU
        // utility floor apply -- both are requirements specific to the
        // ordinary, utility-ranked heap, and save() routes this pattern to
        // the IG-ranked relaxed_heap unconditionally (matching the L2 v2
        // hot path's precedent: a relaxed-rooted pattern is saved there
        // with only an sI > 0.0 sanity check, no G threshold at all).
        const bool root_is_relaxed = prefix.empty()
            ? is_relaxed_root(ux->item)
            : is_relaxed_root(prefix.front());
        const bool eligible = root_is_relaxed
            ? (ux->sI > 0.0)
            : (ux->sI >= minU && ux->sI > 0.0 && ux->ig > G);
        if (eligible) {
            std::vector<int> pat = prefix;
            pat.push_back(ux->item);
            save(pat, *ux);
        }
    }

    if (depth + 1 >= maxd) return;

    for (int i = static_cast<int>(uls.size()) - 2; i >= 0; --i) {
        if (has_deadline) {
            if ((++_explore_calls & (_timeout_check_interval - 1)) == 0) {
                if (Clock::now() >= deadline_tp)
                    throw std::runtime_error("mining_timeout");
            }
        }
        UL* ux = uls[static_cast<size_t>(i)];
        // interaction_relaxed_mining exemptions. Two distinct cases are
        // both needed, since without case (a) relaxed columns would never
        // produce a pattern of length >1: the first relaxed root would be
        // blocked from ever extending past depth 0.
        //
        //   (a) depth == 0 AND ux is a relaxed root candidate: this loop's
        //       ux is itself about to become prefix[0]. Its own ig <= G is
        //       exactly why it needed relaxing in the first place, so this
        //       check must not apply to it here -- it was already let into
        //       ul_map past the TWU/RIU filter for the same reason; gating
        //       it again here on ig would undo that. The depth==0 guard is
        //       required, not optional: without it, a relaxed-column item
        //       that happens to also be considered as a depth>=1 PARTNER
        //       candidate would wrongly get the same exemption, letting it
        //       appear at a non-root position -- silently widening scope
        //       beyond the agreed root-only design.
        //   (b) the CURRENT prefix's root (prefix.front(), depth >= 1) was
        //       already a relaxed root established at depth 0. ux here is
        //       a depth 1+ partner being considered for extension; its own
        //       ig <= G is NOT exempted regardless of whether ux itself is
        //       also a relaxed column -- only the growth-bound check below
        //       is exempted for this case, so the branch is not pruned
        //       purely on accumulated utility while the partner's own G
        //       gate still applies normally.
        const bool ux_is_relaxed_root_at_depth0 =
            (depth == 0) && is_relaxed_root(ux->item);
        const bool prefix_rooted_in_relaxed =
            !prefix.empty() && is_relaxed_root(prefix.front());
        if (!ux_is_relaxed_root_at_depth0 && ux->ig <= G) continue;
        if (!(ux_is_relaxed_root_at_depth0 || prefix_rooted_in_relaxed) &&
            (ux->sI + ux->sR < minU || ux->sI <= 0.0)) continue;
        if ((ux_is_relaxed_root_at_depth0 || prefix_rooted_in_relaxed) &&
            ux->sI <= 0.0) continue;


        std::vector<std::unique_ptr<UL>> ext_owned;
        std::vector<UL*>                 ext;
        std::unordered_set<size_t> ext_cov_hashes;

        auto fnv1a_tids = [](const std::vector<int32_t>& v) -> size_t {
            size_t h = 14695981039346656037ULL;
            for (int32_t x : v) {
                const unsigned char* b = reinterpret_cast<const unsigned char*>(&x);
                for (int byte = 0; byte < 4; byte++) {
                    h ^= static_cast<size_t>(b[byte]);
                    h *= 1099511628211ULL;
                }
            }
            return h;
        };

        for (size_t j = static_cast<size_t>(i + 1); j < uls.size(); j++) {
            if (has_deadline) {
                if ((++_explore_calls & (_timeout_check_interval - 1)) == 0) {
                    if (Clock::now() >= deadline_tp)
                        throw std::runtime_error("mining_timeout");
                }
            }
            UL* uy = uls[j];
            if (candidate_conflicts_with_prefix(prefix, ux->item, uy->item))
                continue;
            if (uy->ig <= G) continue;
            const bool utility_floor_applies =
                !(ux_is_relaxed_root_at_depth0 || prefix_rooted_in_relaxed);
            if (utility_floor_applies && L != 1 && eucs_enabled) {
                double eucs_twu_val = 0.0;
                if (!eucs_lookup(ux->item, uy->item, eucs_twu_val)
                    || eucs_twu_val < minU) {
                    continue;
                }
            }

            const bool allow_utility_abandon = utility_floor_applies;
            auto ch = std::make_unique<UL>(child_ul(prefix_ul, *ux, *uy, allow_utility_abandon));
            if (!ch->tid_arr.empty()) {
                ch->compute_ig(prefix_ul, y_arr, n_cls);
                if (ch->ig <= G) continue;

                // Hash gate: O(1) check; full compare only on collision.
                size_t h = fnv1a_tids(ch->tid_arr);
                bool duplicate_tids = false;
                if (ext_cov_hashes.count(h)) {
                    for (UL* existing : ext) {
                        if (existing->tid_arr == ch->tid_arr) {
                            duplicate_tids = true;
                            break;
                        }
                    }
                }
                if (duplicate_tids) continue;
                ext_cov_hashes.insert(h);

                ch->release_els();
                ext.push_back(ch.get());
                ext_owned.push_back(std::move(ch));
            }
        }
        if (!ext.empty()) {
            std::vector<int> np2 = prefix;
            np2.push_back(ux->item);
            explore(np2, ext, y_arr, n_cls, depth + 1, fmap, ux);
        }
    }
}

// ── Entry point ──────────────────────────────────────────────────────────────

std::vector<PatternEntry> mine_patterns_generic_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, int L, double G,
    double timeout_s) {

    THUIsl miner(K, L, G);

    // ── RIU seed ────────────
    // Set minU to the K-th largest per-item instantaneous utility before
    // mine() runs.  mine() no longer resets minU, so this value propagates
    // into every threshold check in the transaction scan, CUD raise, leaf
    // raise, and explore().  td.RIU is populated by prepare_transactions_cpp.
    miner.minU = td.riu_thresh(K);

    if (timeout_s > 0.0) {
        miner.has_deadline = true;
        miner.deadline_tp  = THUIsl::Clock::now() +
            std::chrono::duration_cast<THUIsl::Clock::duration>(
                std::chrono::duration<double>(timeout_s));
    }

    try {
        miner.mine(td.transactions, td.item_twu, ytrain, n_cls, &td.item_col, &td.item_iu, &td.transaction_utils);
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()) == "mining_timeout") {
            // Return whatever was collected before the timeout.
            return miner.heap;
        }
        throw;
    }
    return miner.heap;
}

std::vector<PatternEntry> mine_patterns_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, int L, double G,
    double timeout_s) {
    if (L == 1)
        return mine_patterns_l1_cpp(td, ytrain, n_cls, K, G, timeout_s);
    if (L == 2)
        return mine_patterns_l2_cpp(td, ytrain, n_cls, K, G, timeout_s);
    return mine_patterns_generic_cpp(td, ytrain, n_cls, K, L, G, timeout_s);
}

// ── Extended: interaction_relaxed_mining, generic L (incl. L>2) path ──
std::vector<PatternEntry> mine_patterns_generic_relaxed_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, int L, double G,
    const std::vector<int>&   relaxed_cols,
    double timeout_s) {

    THUIsl miner(K, L, G);
    miner.relaxed_cols = relaxed_cols;
    miner.minU = td.riu_thresh(K);

    if (timeout_s > 0.0) {
        miner.has_deadline = true;
        miner.deadline_tp  = THUIsl::Clock::now() +
            std::chrono::duration_cast<THUIsl::Clock::duration>(
                std::chrono::duration<double>(timeout_s));
    }

    try {
        miner.mine(td.transactions, td.item_twu, ytrain, n_cls,
                   &td.item_col, &td.item_iu, &td.transaction_utils);
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()) != "mining_timeout") throw;
        // Fall through and partition/merge whatever was collected so far.
    }

    if (relaxed_cols.empty() || td.item_col.empty()) {
        // No relaxation requested: behave exactly like mine_patterns_generic_cpp.
        return miner.heap;
    }

    // THUIsl::save() already routes a pattern into EITHER miner.heap
    // (ordinary, utility-ranked, capped at K) OR miner.relaxed_heap
    // (relaxed-root, IG-ranked, capped at K) at write time -- see save().
    // No post-hoc partitioning of miner.heap is needed or correct here:
    // relaxed-root patterns never land in miner.heap in the first place
    // (save() routes them directly into relaxed_heap), so partitioning
    // miner.heap alone would always produce an empty relaxed pool and
    // silently discard every rescued pattern. Read both pools directly
    // and merge them here, matching the L2 v2 hot path exactly: each pool
    // independently capped at K by THUIsl's own two heaps, then globally
    // re-ranked by IG and truncated to the final K.
    std::vector<PatternEntry> merged(miner.heap);
    merged.insert(merged.end(), miner.relaxed_heap.begin(), miner.relaxed_heap.end());
    std::sort(merged.begin(), merged.end(),
              [](const PatternEntry& a, const PatternEntry& b) { return a.ig > b.ig; });
    if (static_cast<int>(merged.size()) > K) merged.resize(static_cast<size_t>(K));
    return merged;
}

std::vector<PatternEntry> mine_patterns_relaxed_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, int L, double G,
    const std::vector<int>&   relaxed_cols,
    double timeout_s) {
    if (L == 2)
        return mine_patterns_l2_augmented_patterns_v2_cpp(
            td, ytrain, n_cls, K, G, relaxed_cols, timeout_s);
    return mine_patterns_generic_relaxed_cpp(
        td, ytrain, n_cls, K, L, G, relaxed_cols, timeout_s);
}

}  // namespace hugiml

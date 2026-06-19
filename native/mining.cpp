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
#include "mining_l2.hpp"
#include "math.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <limits>

namespace hugiml {

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

UL THUIsl::child_ul(const UL* prefix_ul, const UL& px_ul, const UL& py_ul) {
    // Construct the utility list for Pxy from utility lists for Px and Py.
    // For a 2-item extension (P is empty), utility is u(x)+u(y).
    // For deeper extensions, subtract the shared prefix utility u(P), Without this subtraction,
    // every prefix item is double-counted for L >= 3.
    UL c(py_ul.item);
    if (px_ul.tid_arr.empty() || py_ul.tid_arr.empty()) return c;

    const auto& xt = px_ul.tid_arr;
    const auto& yt = py_ul.tid_arr;
    std::vector<int32_t> c_tid;
    std::vector<double>  c_iu, c_ru;
    c_tid.reserve(std::min(xt.size(), yt.size()));
    c_iu.reserve(c_tid.capacity());
    c_ru.reserve(c_tid.capacity());

    size_t xi = 0, yi = 0;
    while (xi < xt.size() && yi < yt.size()) {
        if (xt[xi] < yt[yi]) { ++xi; continue; }
        if (yt[yi] < xt[xi]) { ++yi; continue; }

        const int32_t tid = xt[xi];
        double prefix_iu = 0.0;
        if (prefix_ul != nullptr) {
            auto pit = std::lower_bound(prefix_ul->tid_arr.begin(),
                                        prefix_ul->tid_arr.end(), tid);
            if (pit == prefix_ul->tid_arr.end() || *pit != tid) {
                ++xi; ++yi;
                continue;
            }
            const size_t pi = static_cast<size_t>(pit - prefix_ul->tid_arr.begin());
            prefix_iu = prefix_ul->iu_arr[pi];
        }

        c_tid.push_back(tid);
        c_iu.push_back(px_ul.iu_arr[xi] + py_ul.iu_arr[yi] - prefix_iu);
        c_ru.push_back(py_ul.ru_arr[yi]);
        ++xi; ++yi;
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
    // EUCS (Extended Utility Co-occurrence Structure) is disabled by default.
    // It builds a quadratic-in-active-items map during the transaction scan
    // which causes O(w²) memory overhead per transaction row for wide datasets
    // (w = active items per row).  For large datasets this easily exhausts
    // available RAM before mining starts.  The pruning benefit EUCS provides
    // is outweighed by the memory cost on typical wide tabular data.
    //
    // To re-enable: replace the line below with:
    //     bool use_eucs = (L != 1);
    // Note that doing so will restore the quadratic memory behaviour.
    bool use_eucs = false;
    bool use_leaf = (L == -1 || L == 0);   

    FMap fmap;

    // leaf_map[end_pos][start_pos] = cumulative utility of the consecutive
    // item run [start_pos .. end_pos) in sorted_items order.
    using LeafMap = std::unordered_map<int, std::unordered_map<int, double>>;
    LeafMap leaf_map;

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
    // comparator (O(1) lookup vs O(1) array access — same complexity but
    // avoids two floating-point loads and a conditional inside every comparison).
    // item_to_pos serves the leaf-map update (built only when use_leaf).
    std::vector<int> item_rank(static_cast<size_t>(n_items) + 1, 0);
    for (int pos = 0; pos < static_cast<int>(sorted_items.size()); pos++)
        item_rank[sorted_items[pos]] = pos;

    std::unordered_map<int, int> item_to_pos;
    if (use_leaf) {
        item_to_pos.reserve(sorted_items.size());
        for (int pos = 0; pos < static_cast<int>(sorted_items.size()); pos++)
            item_to_pos[sorted_items[pos]] = pos;
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
            if (use_eucs) {
                auto& fm = fmap[it];
                for (int j2 = i + 1; j2 < static_cast<int>(active.size()); j2++) {
                    int oj = active[j2].first;
                    double ou = active[j2].second;
                    if (oj != it && !same_feature(it, oj)) {
                        fm[oj].first  += new_twu;
                        fm[oj].second += u + ou;
                    }
                }
            }
            // ── Leaf-map update ──────────────────────
            // For each item, walk backward through the transaction's active
            // items that are CONSECUTIVE in sorted_items order, accumulating
            // cumulative utilities.  Only the consecutive run is tracked
            if (use_leaf) {
                auto pos_it = item_to_pos.find(it);
                if (pos_it != item_to_pos.end()) {
                    int end_pos = pos_it->second;
                    auto& leaf_entry = leaf_map[end_pos];
                    double cutil = u;
                    int follow_pos = end_pos;
                    for (int j = i - 1; j >= 0; j--) {
                        int prec = active[j].first;
                        auto pit = item_to_pos.find(prec);
                        if (pit == item_to_pos.end()) break;
                        if (pit->second != follow_pos - 1) break;
                        follow_pos = pit->second;
                        cutil += active[j].second;
                        leaf_entry[follow_pos] += cutil;
                    }
                }
            }
            rem += u;
        }
    }

    // ── CUD raise  ────────────────────────
    // Active only when use_eucs (L != 1).  Scans EUCS pair utilities,
    // maintains a K-sized min-heap, and raises minU to the K-th largest
    // pair utility if it exceeds the current floor.  Then removes EUCS
    // entries whose co-occurrence TWU is now below the raised minU
    if (use_eucs && !fmap.empty()) {
        std::vector<double> top_utils;
        top_utils.reserve(static_cast<size_t>(K) + 1);
        for (auto& [a, inner] : fmap) {
            for (auto& [b, vals] : inner) {
                double util = vals.second;          // util_sum
                if (util < minU) continue;
                top_utils.push_back(util);
                std::push_heap(top_utils.begin(), top_utils.end(),
                               std::greater<double>{});
                if (static_cast<int>(top_utils.size()) > K) {
                    std::pop_heap(top_utils.begin(), top_utils.end(),
                                  std::greater<double>{});
                    top_utils.pop_back();
                }
            }
        }
        if (static_cast<int>(top_utils.size()) >= K &&
                top_utils.front() > minU)
            minU = top_utils.front();

        // Remove EUCS entries below new threshold 
        for (auto& [a, inner] : fmap)
            for (auto it = inner.begin(); it != inner.end(); )
                it = (it->second.first < minU) ? inner.erase(it) : ++it;
    }

    // Seal all 1-item ULs, compute IG, then release transient els buffers.
    // compute_ig uses tid_arr, so recursive child IG remains correct after release.
    for (auto& kv : ul_map) kv.second.seal();
    for (auto& kv : ul_map) {
        kv.second.compute_ig(nullptr, ytrain, n_cls);
        kv.second.release_els();  // reclaim els memory before recursion
    }

    // ── Leaf raise ───────────────────────────────
    // Active only when use_leaf (L == -1).  Three sources feed the K-sized
    // min-heap: LIU-Exact (direct leaf_map entries), LIU-LB (sub-run estimates
    // obtained by subtracting individual item utilities from exact entries), and
    // all 1-item actual utilities.  minU is raised to the K-th largest value
    // found if it exceeds the current floor.
    if (use_leaf && !leaf_map.empty()) {
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

        // LIU-Exact
        for (auto& [end_pos, inner] : leaf_map)
            for (auto& [start_pos, val] : inner)
                add_to_leaf(val);

        // LIU-LB: subtract intermediate-item utilities to get lower bounds
        int n_sorted = static_cast<int>(sorted_items.size());
        for (auto& [end_pos, inner] : leaf_map) {
            for (auto& [start_pos, val] : inner) {
                if (val < minU) continue;
                for (int i = start_pos + 1; i < end_pos - 1; i++) {
                    if (i >= n_sorted) break;
                    double v1 = val - ul_map.at(sorted_items[i]).sI;
                    add_to_leaf(v1);
                    for (int j = i + 1; j < end_pos - 1; j++) {
                        if (j >= n_sorted) break;
                        double v2 = v1 - ul_map.at(sorted_items[j]).sI;
                        add_to_leaf(v2);
                        for (int k = j + 1; k + 1 < end_pos - 1; k++) {
                            if (k >= n_sorted) break;
                            add_to_leaf(v2 - ul_map.at(sorted_items[k]).sI);
                        }
                    }
                }
            }
        }

        // All 1-item actual utilities
        for (auto& kv : ul_map) add_to_leaf(kv.second.sI);

        if (static_cast<int>(leaf_utils.size()) >= K &&
                leaf_utils.front() > minU)
            minU = leaf_utils.front();

        leaf_map.clear();   
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

        if (L != 1 && !fmap.empty() && fmap.find(ux->item) == fmap.end()) continue;

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

            auto ch = std::make_unique<UL>(child_ul(prefix_ul, *ux, *uy));
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

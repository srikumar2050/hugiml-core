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
#include "math.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

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

void UL::compute_ig(const UL* parent,
                     const std::vector<int>& y_arr, int n_cls) {
    if (els.empty()) { ig = 0.0; return; }

    std::vector<int> y_in;
    y_in.reserve(els.size());
    for (auto& e : els) y_in.push_back(y_arr[e.tid]);

    double base;
    int    n_parent;
    if (parent == nullptr) {
        base     = entropy_vec(y_arr, n_cls);
        n_parent = static_cast<int>(y_arr.size());
    } else {
        std::vector<int> py_v;
        py_v.reserve(parent->els.size());
        for (auto& e : parent->els) py_v.push_back(y_arr[e.tid]);
        base     = entropy_vec(py_v, n_cls);
        n_parent = static_cast<int>(parent->els.size());
    }
    if (n_parent == 0) { ig = 0.0; return; }

    // y_out: elements in parent (or all) NOT in this UL
    std::unordered_set<int> tid_set;
    tid_set.reserve(els.size());
    for (auto& e : els) tid_set.insert(e.tid);

    std::vector<int> y_out;
    if (parent == nullptr) {
        for (int i = 0; i < static_cast<int>(y_arr.size()); i++)
            if (tid_set.find(i) == tid_set.end())
                y_out.push_back(y_arr[i]);
    } else {
        for (auto& e : parent->els)
            if (tid_set.find(e.tid) == tid_set.end())
                y_out.push_back(y_arr[e.tid]);
    }

    double ce = (static_cast<double>(y_in.size())  / n_parent
                     * entropy_vec(y_in, n_cls)
                 + static_cast<double>(y_out.size()) / n_parent
                     * entropy_vec(y_out, n_cls));
    ig = std::max(base - ce, 0.0);
}

// ── THUIsl methods ───────────────────────────────────────────────────────────

/// Min-heap comparator (smallest utility on top, matching Python heapq).
struct MinHeapCmp {
    bool operator()(const PatternEntry& a, const PatternEntry& b) const {
        return a.utility > b.utility;
    }
};

void THUIsl::save(const std::vector<int>& items, const UL& ul) {
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

UL THUIsl::child_ul(const UL& p_ul, const UL& x_ul) {
    UL c(x_ul.item);
    if (p_ul.tid_arr.empty() || x_ul.tid_arr.empty()) return c;

    const auto& pt = p_ul.tid_arr;
    const auto& xt = x_ul.tid_arr;
    std::vector<int32_t> c_tid;
    std::vector<double>  c_iu, c_ru;
    c_tid.reserve(std::min(pt.size(), xt.size()));
    c_iu.reserve(c_tid.capacity());
    c_ru.reserve(c_tid.capacity());

    for (size_t xi = 0; xi < xt.size(); xi++) {
        auto it = std::lower_bound(pt.begin(), pt.end(), xt[xi]);
        if (it == pt.end() || *it != xt[xi]) continue;
        size_t pi = static_cast<size_t>(it - pt.begin());
        c_tid.push_back(xt[xi]);
        c_iu.push_back(p_ul.iu_arr[pi] + x_ul.iu_arr[xi]);
        c_ru.push_back(x_ul.ru_arr[xi]);
    }
    if (c_tid.empty()) return c;
    c.seal_from_arrays(std::move(c_tid), std::move(c_iu), std::move(c_ru));
    return c;
}

void THUIsl::mine(const TransList& transactions,
                   const std::vector<double>& item_twu,
                   const std::vector<int>& ytrain,
                   int n_cls) {
    minU = 0.0;
    heap.clear();
    _explore_calls = 0;
    int  n_items  = static_cast<int>(item_twu.size());
    bool use_eucs = (L != 1);

    FMap fmap;

    // Build utility-list map for items that pass the initial threshold
    std::unordered_map<int, UL> ul_map;
    ul_map.reserve(n_items);
    for (int iid = 1; iid <= n_items; iid++)
        if (item_twu[iid - 1] >= minU)
            ul_map.emplace(iid, UL(iid));

    // Sorted by TWU ascending (same as Python sorted_items)
    std::vector<int> sorted_items;
    sorted_items.reserve(ul_map.size());
    for (auto& kv : ul_map) sorted_items.push_back(kv.first);
    std::sort(sorted_items.begin(), sorted_items.end(),
              [&](int a, int b) { return item_twu[a - 1] < item_twu[b - 1]; });

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
        if (trans.size() == 1 && trans[0].first == -1) continue;

        std::vector<std::pair<int, double>> active;
        active.reserve(trans.size());
        for (auto& [it, u] : trans)
            if (ul_map.count(it)) active.push_back({it, u});
        if (active.empty()) continue;

        double new_twu = 0.0;
        for (auto& [it, u] : active) new_twu += u;

        std::sort(active.begin(), active.end(),
                  [&](auto& a, auto& b) {
                      return item_twu[a.first - 1] < item_twu[b.first - 1];
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
                    if (oj != it) {
                        fm[oj].first  += new_twu;
                        fm[oj].second += u + ou;
                    }
                }
            }
            rem += u;
        }
    }

    // Seal all 1-item ULs, compute IG, then release staging buffers
    for (auto& kv : ul_map) kv.second.seal();
    for (auto& kv : ul_map) {
        kv.second.compute_ig(nullptr, ytrain, n_cls);
        kv.second.release_els();  // reclaim els memory before recursion
    }

    std::vector<UL*> uls;
    uls.reserve(sorted_items.size());
    for (int iid : sorted_items) uls.push_back(&ul_map.at(iid));

    explore({}, uls, ytrain, n_cls, 0, fmap);
}

void THUIsl::explore(std::vector<int>  prefix,
                      std::vector<UL*>& uls,
                      const std::vector<int>& y_arr,
                      int n_cls, int depth, FMap& fmap) {
    int maxd = (L == -1 || L == 0) ? 99 : L;
    for (size_t i = 0; i < uls.size(); i++) {
        // Timeout check: every _timeout_check_interval outer-loop iterations.
        // Counting here (not per explore() call) ensures the clock is polled
        // for L=1 and L=2 where total explore() calls stay well below the
        // check interval.
        if (has_deadline) {
            if ((++_explore_calls & (_timeout_check_interval - 1)) == 0) {
                if (Clock::now() >= deadline_tp)
                    throw std::runtime_error("mining_timeout");
            }
        }
        UL* ux = uls[i];
        if (ux->sI + ux->sR < minU) continue;

        if (ux->sI >= minU && ux->ig >= G) {
            std::vector<int> pat = prefix;
            pat.push_back(ux->item);
            save(pat, *ux);
        }
        if (depth + 1 >= maxd) continue;

        std::vector<std::unique_ptr<UL>> ext_owned;
        std::vector<UL*>                 ext;
        for (size_t j = i + 1; j < uls.size(); j++) {
            // Also check the deadline in the inner pair-building loop.
            // This is where most wall-clock time is spent at L=2.
            if (has_deadline) {
                if ((++_explore_calls & (_timeout_check_interval - 1)) == 0) {
                    if (Clock::now() >= deadline_tp)
                        throw std::runtime_error("mining_timeout");
                }
            }
            UL* uy = uls[j];
            // EUCS pruning
            if (L != 1) {
                double eucs_val = 0.0;
                auto fi = fmap.find(ux->item);
                if (fi != fmap.end()) {
                    auto fj = fi->second.find(uy->item);
                    if (fj != fi->second.end()) eucs_val = fj->second.first;
                }
                if (eucs_val < minU) continue;
            }
            auto ch = std::make_unique<UL>(child_ul(*ux, *uy));
            if (ch->sI + ch->sR >= minU) {
                // Rebuild els from arrays for compute_ig, then release
                ch->compute_ig(ux, y_arr, n_cls);
                ch->release_els();
                ext.push_back(ch.get());
                ext_owned.push_back(std::move(ch));
            }
        }
        if (!ext.empty()) {
            std::vector<int> np2 = prefix;
            np2.push_back(ux->item);
            explore(np2, ext, y_arr, n_cls, depth + 1, fmap);
        }
    }
}

// ── Entry point ──────────────────────────────────────────────────────────────

std::vector<PatternEntry> mine_patterns_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, int L, double G,
    double timeout_s) {

    THUIsl miner(K, L, G);
    if (timeout_s > 0.0) {
        miner.has_deadline = true;
        miner.deadline_tp  = THUIsl::Clock::now() +
            std::chrono::duration_cast<THUIsl::Clock::duration>(
                std::chrono::duration<double>(timeout_s));
    }

    try {
        miner.mine(td.transactions, td.item_twu, ytrain, n_cls);
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()) == "mining_timeout") {
            // Return whatever was collected before the timeout.
            return miner.heap;
        }
        throw;
    }
    return miner.heap;
}

}  // namespace hugiml

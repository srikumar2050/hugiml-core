/**
 * mining.hpp — Pattern mining structures and algorithms
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Defines PatternEntry (the heap element exposed to Python via pybind11),
 * the internal utility-list structures (El, UL), and the THUIsl miner.
 */

#pragma once

#include "transaction.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <chrono>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace hugiml {

// ── PatternEntry — exposed to Python via pybind11 ────────────────────────────

struct PatternEntry {
    double           utility;
    std::vector<int> items;
    double           ig;
};

// ── Internal structures (not exposed to Python) ──────────────────────────────

struct El {
    int    tid;
    double iu, ru;
};

class UL {
public:
    int    item = 0;
    double sI   = 0.0;
    double sR   = 0.0;
    double ig   = 0.0;

    // els is used only as a construction/staging buffer.
    // After sealing, tid_arr/iu_arr/ru_arr are the canonical data used by
    // recursion and information-gain computation; els may be released.
    std::vector<El>      els;
    std::vector<int32_t> tid_arr;
    std::vector<double>  iu_arr, ru_arr;
    bool sealed = false;

    explicit UL(int it) : item(it) {}

    void add(int t, double i, double r);
    void seal();
    void seal_from_arrays(std::vector<int32_t>&& ta,
                          std::vector<double>&&  ia,
                          std::vector<double>&&  ra);
    void compute_ig(const UL* parent,
                    const std::vector<int>& y_arr, int n_cls);

    // Release the els staging buffer after sealing/IG computation to free memory.
    // Correctness must not depend on els after seal(); use tid_arr instead.
    void release_els() { std::vector<El>().swap(els); }
};

// ── Top-K HUI miner ─────────────────────────────────────────────────────────

using FMap = std::unordered_map<int,
                std::unordered_map<int, std::pair<double, double>>>;

class THUIsl {
public:
    int    K, L;
    double G, minU = 0.0;
    std::vector<PatternEntry> heap;

    // Dense EUCS cache for admitted items.  When enabled, it stores the
    // pair co-occurrence transaction utility upper bound in sorted-item order.
    // It is built only under a cell cap; when absent, mining falls back to the
    // original utility-list search without EUCS pruning.
    bool eucs_enabled = false;
    int eucs_m = 0;
    std::vector<int> eucs_pos_for_item;
    std::vector<double> eucs_twu;

    size_t eucs_index_from_pos(int left_pos, int right_pos) const {
        if (left_pos > right_pos) std::swap(left_pos, right_pos);
        return static_cast<size_t>(left_pos) * static_cast<size_t>(eucs_m)
            - (static_cast<size_t>(left_pos) * static_cast<size_t>(left_pos + 1)) / 2
            + static_cast<size_t>(right_pos - left_pos - 1);
    }

    bool eucs_lookup(int item_a, int item_b, double& twu) const {
        if (!eucs_enabled || item_a <= 0 || item_b <= 0) return false;
        if (static_cast<size_t>(item_a) >= eucs_pos_for_item.size()
            || static_cast<size_t>(item_b) >= eucs_pos_for_item.size()) return false;
        int pa = eucs_pos_for_item[static_cast<size_t>(item_a)];
        int pb = eucs_pos_for_item[static_cast<size_t>(item_b)];
        if (pa < 0 || pb < 0 || pa == pb) return false;
        const size_t idx = eucs_index_from_pos(pa, pb);
        if (idx >= eucs_twu.size()) return false;
        twu = eucs_twu[idx];
        return true;
    }
    // item_col[item_id - 1] gives the original feature for each item.
    // When present, mining enforces the exact mutual-exclusion constraint
    // that a pattern cannot contain two bins/categories from the same feature.
    const std::vector<int>* item_col = nullptr;
    const std::vector<double>* item_iu = nullptr;
    const std::vector<std::vector<double>>* transaction_utils = nullptr;

    // interaction_relaxed_mining support: when non-empty, item_col values
    // listed here are admitted into the depth-0 (root) candidate set even
    // if they fail the TWU/RIU seed filter. This relaxation now extends to
    // TWO positions in an explored branch: items[0] (the root) and, when
    // depth == 0, its immediate pairing partner items[1] (the first
    // child) -- see the uy_is_relaxed_child_at_depth0 gate in explore().
    // A relaxed root's own IG/TWU no longer needs to clear G to be
    // extended, and a relaxed first-child partner no longer needs to
    // individually clear G to be paired with it; this rescues the
    // classic case where two individually-uninformative-but-jointly-
    // informative columns (e.g. XOR/parity-style interactions) would
    // otherwise never be paired, since previously the *partner* still had
    // to pass the ordinary IG gate even when the root was relaxed. Every
    // other gate inside explore() (G checks, the TWU/RIU bound at depth
    // 2+, pair-construction IG checks for grandchildren and beyond) runs
    // completely unmodified. A relaxed item can therefore only ever
    // appear at items[0] or items[1] of an explored branch, never spliced
    // in deeper. This is a deliberate scope limitation, not an oversight:
    // it avoids threading relaxation state through the recursion (which
    // would compound across depths and risk search-space blowup) at the
    // cost of not rescuing patterns where the informative interaction
    // only appears among items at depth 2+ with no individually-
    // qualifying root or first child. Given that interactions worth
    // rescuing are expected to be low-order (2-3 features), the
    // root+first-child restriction does not need depth >1 relaxation to
    // be useful: any interaction the survivor(s) participate in still
    // gets their columns as the anchor of a full, ordinarily-pruned
    // exploration from that root/first-child pair.
    std::vector<int> relaxed_cols;

    // Dual-heap budget (mirrors mine_patterns_l2_augmented_patterns_v2_cpp):
    // when relaxed_cols is non-empty, save() routes a pattern into EITHER
    // heap (ordinary, utility-ranked, capped at K) OR relaxed_heap
    // (IG-ranked, capped at K) based on whether its root item is relaxed --
    // never both, and each capped independently at K so relaxed-root
    // patterns do not have to out-compete ordinary high-utility patterns
    // for the same K slots. The caller merges both pools and globally
    // re-ranks by IG, truncating to the final K, exactly as in the L2
    // hotpath's v2 design. When relaxed_cols is empty, save() is identical
    // to the unmodified single-heap behavior.
    std::vector<PatternEntry> relaxed_heap;
    double minIg = -std::numeric_limits<double>::infinity();

    bool is_relaxed_root(int item) const {
        if (relaxed_cols.empty() || !item_col) return false;
        if (item <= 0 || static_cast<size_t>(item - 1) >= item_col->size()) return false;
        const int col = (*item_col)[static_cast<size_t>(item - 1)];
        return std::find(relaxed_cols.begin(), relaxed_cols.end(), col) != relaxed_cols.end();
    }

    // Timeout support: set deadline_tp before calling mine().
    // explore() checks this every _timeout_check_interval calls and throws
    // std::runtime_error("mining_timeout") when the deadline is exceeded.
    using Clock    = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    TimePoint deadline_tp;
    bool      has_deadline = false;

    static constexpr int _timeout_check_interval = 256;

    THUIsl(int K, int L, double G) : K(K), L(L), G(G) {}

    void save(const std::vector<int>& items, const UL& ul);
    void save_by_ig(const std::vector<int>& items, const UL& ul);
    UL   child_ul(const UL* prefix_ul, const UL& px_ul, const UL& py_ul,
                  bool allow_utility_abandon = true);
    void mine(const TransList& transactions,
              const std::vector<double>& item_twu,
              const std::vector<int>& ytrain, int n_cls,
              const std::vector<int>* item_col_in = nullptr,
              const std::vector<double>* item_iu_in = nullptr,
              const std::vector<std::vector<double>>* transaction_utils_in = nullptr);

private:
    int _explore_calls = 0;

    void explore(std::vector<int>  prefix,
                 std::vector<UL*>& uls,
                 const std::vector<int>& y_arr,
                 int n_cls, int depth, FMap& fmap,
                 const UL* prefix_ul);

    bool same_feature(int item_a, int item_b) const;
    bool candidate_conflicts_with_prefix(const std::vector<int>& prefix,
                                         int prefix_item,
                                         int candidate_item) const;
};

// ── Entry point called from Python bindings ──────────────────────────────────
// ytrain is pre-extracted to std::vector<int> by the binding layer (GIL held)
// so this function is safe to call with the GIL released.
// timeout_s: wall-clock seconds before mining is aborted (0 = no limit).
std::vector<PatternEntry> mine_patterns_generic_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, int L, double G,
    double timeout_s = 0.0);

std::vector<PatternEntry> mine_patterns_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, int L, double G,
    double timeout_s = 0.0);

// ── Extended: interaction_relaxed_mining for the generic L (incl. L>2) path ──
//
// Same THUIsl::mine()/explore() recursion as mine_patterns_generic_cpp, with
// two changes: items whose source column is in relaxed_cols are admitted
// into the depth-0 root candidate set even if they fail the TWU/RIU seed
// filter, AND, when pairing at depth 0, a relaxed-column partner (the
// root's first child) may bypass its own IG gate the same way the root
// does. No other gate is relaxed at any deeper depth -- see the
// THUIsl::relaxed_cols comment in mining.hpp for why this scope is
// deliberate. THUIsl::save() routes each pattern, at write time, into
// either the ordinary utility-ranked heap or the relaxed-root IG-ranked
// relaxed_heap (mutually exclusive, never both), each independently capped
// at K. This function reads both pools, merges them, and globally
// re-ranks by IG, truncated to the final top-K -- the same v2 budget
// contract used by mine_patterns_l2_augmented_patterns_v2_cpp. With
// relaxed_cols empty, relaxed_heap stays empty and this reduces to
// mine_patterns_generic_cpp's behavior (same root set, so same output,
// modulo the harmless IG-vs-utility final sort order).
std::vector<PatternEntry> mine_patterns_generic_relaxed_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, int L, double G,
    const std::vector<int>&   relaxed_cols,
    double timeout_s = 0.0);

// Dispatcher mirroring mine_patterns_cpp: routes L=2 to the L2 hotpath's
// root-relaxed variant, everything else to mine_patterns_generic_relaxed_cpp.
std::vector<PatternEntry> mine_patterns_relaxed_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, int L, double G,
    const std::vector<int>&   relaxed_cols,
    double timeout_s = 0.0);

}  // namespace hugiml

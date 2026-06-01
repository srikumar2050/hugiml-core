/**
 * mining_l1.cpp — Hot path implementation for L=1 pattern mining.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Design
 * ──────
 * For L=1 the miner only ever emits singleton patterns {item}.  Their
 * utility is exactly RIU[iid-1] (the instantaneous-utility sum already
 * accumulated by prepare_transactions_cpp) and their information gain
 * requires only two class-count vectors:
 *
 *   cnt_global[c]   – class c count over ALL n training rows (parent pop.)
 *   cnt_item[iid][c] – class c count over rows where iid appears (cnt_in)
 *
 * Both are built in ONE forward pass over td.transactions.  No utility-list
 * objects, no els[] staging buffers, no iu_arr / ru_arr arrays are created.
 *
 * IG formula (matches compute_ig(parent=nullptr) in mining.cpp exactly):
 *
 *   n_parent = len(ytrain)            // ALL rows incl. zero-utility ones
 *   n_in     = |rows containing iid|  // only non-placeholder tx rows
 *   n_out    = n_parent - n_in
 *
 *   base_entropy  = H(cnt_global,  n_parent)
 *   cond_entropy  = (n_in/n_parent)*H(cnt_in, n_in)
 *                 + (n_out/n_parent)*H(cnt_out, n_out)
 *   ig = base_entropy - cond_entropy
 *
 *   Special case: n_out == 0 → ig = NaN  (matches original; NaN > G is
 *   false so these items are not saved — identical filtering behaviour).
 *
 * Top-K heap
 * ──────────
 * Same min-heap + minU update logic as THUIsl::save().  minU is seeded at
 * td.riu_thresh(K) and rises as the heap fills, tightening the threshold
 * exactly as the general path does.
 *
 * Timeout
 * ───────
 * A wall-clock deadline is checked every _CHECK_INTERVAL iterations of the
 * transaction scan loop (same cadence as THUIsl::explore()). On expiry,
 * whatever has been collected so far is returned — same graceful-degradation
 * contract as the general path.
 */

#include "mining_l1.hpp"
#include "transaction.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hugiml {

// ── Internal helpers ─────────────────────────────────────────────────────────

// Shannon entropy from integer class counts (natural-log scale).
// Mirrors the static entropy_from_counts() in mining.cpp exactly.
static double l1_entropy(const std::vector<int>& counts, int total) {
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

// Compute information gain for a single item.
// Matches UL::compute_ig(parent=nullptr, ytrain, n_cls) precisely.
//
//   cnt_global  — class counts over ALL n_train rows (the "parent" population)
//   cnt_in      — class counts over rows where this item appeared
//   n_in        — number of rows where this item appeared
//   n_train     — total training rows (= ytrain.size())
//   n_cls       — number of classes
static double l1_ig(const std::vector<int>& cnt_global,
                    const std::vector<int>& cnt_in,
                    int n_in, int n_train, int n_cls) {
    int n_out = n_train - n_in;

    // n_out == 0: item covers entire population — matches NaN branch in compute_ig.
    if (n_out == 0)
        return std::numeric_limits<double>::quiet_NaN();

    double base = l1_entropy(cnt_global, n_train);

    // cnt_out = cnt_global - cnt_in  (identical to mining.cpp's cnt_out computation)
    std::vector<int> cnt_out(n_cls);
    for (int k = 0; k < n_cls; k++)
        cnt_out[k] = cnt_global[k] - cnt_in[k];

    double ce = (static_cast<double>(n_in)  / n_train * l1_entropy(cnt_in,  n_in)
               + static_cast<double>(n_out) / n_train * l1_entropy(cnt_out, n_out));
    return base - ce;
}

// Min-heap comparator — smallest utility on top (matches THUIsl's MinHeapCmp).
struct L1MinHeapCmp {
    bool operator()(const PatternEntry& a, const PatternEntry& b) const {
        return a.utility > b.utility;
    }
};

// Push a candidate into the top-K heap; update minU if the heap fills.
// Direct translation of THUIsl::save() for singleton items.
static void l1_save(std::vector<PatternEntry>& heap, double& minU,
                    int K, int item_id, double utility, double ig) {
    PatternEntry pe{utility, {item_id}, ig};
    if (static_cast<int>(heap.size()) < K) {
        heap.push_back(pe);
        std::push_heap(heap.begin(), heap.end(), L1MinHeapCmp{});
        if (static_cast<int>(heap.size()) == K)
            minU = heap.front().utility;
    } else if (utility > minU) {
        std::pop_heap(heap.begin(), heap.end(), L1MinHeapCmp{});
        heap.back() = pe;
        std::push_heap(heap.begin(), heap.end(), L1MinHeapCmp{});
        minU = heap.front().utility;
    }
}

// ── Entry point ──────────────────────────────────────────────────────────────

std::vector<PatternEntry> mine_patterns_l1_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, double G,
    double timeout_s)
{
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    static constexpr int _CHECK_INTERVAL = 256;  // matches THUIsl

    // ── Deadline setup ────────────────────────────────────────────────────────
    bool      has_deadline = (timeout_s > 0.0);
    TimePoint deadline_tp  = {};
    if (has_deadline)
        deadline_tp = Clock::now() +
            std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(timeout_s));

    // ── minU seed — identical to mine_patterns_cpp ────────────────────────────
    // Seed with the K-th largest per-item instantaneous utility so that the
    // initial threshold matches exactly what the general path uses.
    double minU = td.riu_thresh(K);

    const int n_train = static_cast<int>(ytrain.size());
    const int n_items = static_cast<int>(td.item_twu.size());

    // ── Identify surviving items (TWU ≥ minU, same filter as mine()) ──────────
    // We use TWU as the initial gate (matches ul_map construction in mine()).
    // Items whose actual sI (= RIU) < minU will be skipped later in the save
    // condition, exactly as explore()'s save loop does.
    std::vector<int> candidates;
    candidates.reserve(static_cast<size_t>(n_items));
    for (int iid = 1; iid <= n_items; iid++)
        if (td.item_twu[iid - 1] >= minU)
            candidates.push_back(iid);

    if (candidates.empty())
        return {};

    // ── Build item → dense index map for the count matrix ────────────────────
    // Map surviving item IDs to [0, m) for compact 2-D storage.
    // m = candidates.size()
    const int m = static_cast<int>(candidates.size());

    // iid_to_idx[iid] = dense column index; 0 for items not in candidates.
    // Valid only for iid in [1, n_items].
    std::vector<int> iid_to_idx(static_cast<size_t>(n_items) + 1, -1);
    for (int ci = 0; ci < m; ci++)
        iid_to_idx[static_cast<size_t>(candidates[ci])] = ci;

    // ── Per-item class-count matrix — flat row-major [m × n_cls] ─────────────
    // cnt_item[ci * n_cls + c] = count of class c among transactions containing
    // candidate item candidates[ci].
    std::vector<int> cnt_item(static_cast<size_t>(m) * n_cls, 0);

    // Per-item transaction count (= n_in for the IG computation).
    std::vector<int> n_in_item(m, 0);

    // ── Global class counts — over ALL ytrain rows (parent population) ────────
    // This matches compute_ig(parent=nullptr): n_parent = ytrain.size(),
    // cnt_parent iterates ALL y_arr entries regardless of tx utility.
    std::vector<int> cnt_global(n_cls, 0);
    for (int lbl : ytrain)
        if (lbl >= 0 && lbl < n_cls) cnt_global[lbl]++;

    // ── Single forward pass over transactions ─────────────────────────────────
    // For each non-placeholder transaction, for each surviving item it contains,
    // increment that item's class count for the row's label.
    // Timeout is checked every _CHECK_INTERVAL iterations.
    int loop_ctr = 0;
    const int n_tx = static_cast<int>(td.transactions.size());

    for (int tid = 0; tid < n_tx; tid++) {
        // Deadline check — same cadence as THUIsl::explore().
        // On expiry, return whatever partial heap has been built so far.
        // This matches mine_patterns_cpp's graceful-degradation contract:
        // never throw, always return a valid (possibly empty) list.
        if (has_deadline) {
            if ((++loop_ctr & (_CHECK_INTERVAL - 1)) == 0) {
                if (Clock::now() >= deadline_tp)
                    return {};   // scan incomplete; no heap built yet
            }
        }

        const Trans& tx = td.transactions[tid];

        // Skip placeholder rows ({-1, 0.0} sentinel emitted by prepare_transactions_cpp
        // for zero-utility rows).
        if (tx.size() == 1 && tx[0] == -1) continue;

        const int lbl = ytrain[tid];
        const bool lbl_valid = (lbl >= 0 && lbl < n_cls);

        for (int iid : tx) {
            if (iid <= 0 || iid > n_items) continue;
            const int ci = iid_to_idx[static_cast<size_t>(iid)];
            if (ci < 0) continue;  // item didn't pass TWU threshold

            ++n_in_item[ci];
            if (lbl_valid)
                cnt_item[static_cast<size_t>(ci) * n_cls + lbl]++;
        }
    }

    // ── Build top-K heap ──────────────────────────────────────────────────────
    // Iterate candidates in descending TWU order so that the highest-utility
    // items are offered to the heap first, mirroring explore()'s reverse-index
    // iteration (sorted_items is ascending TWU; explore iterates i from
    // uls.size()-1 downto 0, i.e. highest TWU first).
    // Iteration order doesn't affect the final heap membership (top-K is
    // order-independent) but matching the direction keeps minU dynamics
    // consistent and helps tighten the threshold sooner.
    std::vector<PatternEntry> heap;
    heap.reserve(static_cast<size_t>(K) + 1);

    // candidates is in ascending TWU order (mirroring sorted_items).
    // Iterate in reverse to process highest TWU first.
    for (int ci = m - 1; ci >= 0; --ci) {
        const int iid = candidates[ci];

        // Utility = RIU[iid-1] = sI of the 1-item UL.
        // save condition in explore(): sI >= minU AND sI > 0.0 AND ig > G.
        const double utility = td.RIU[static_cast<size_t>(iid - 1)];
        if (utility < minU || utility <= 0.0) continue;

        // Compute IG from the accumulated count vectors.
        const int ni = n_in_item[ci];
        if (ni == 0) continue;  // item never appeared (shouldn't happen if utility > 0)

        // Build cnt_in view from the flat matrix row.
        std::vector<int> cnt_in(cnt_item.data() + static_cast<size_t>(ci) * n_cls,
                                cnt_item.data() + static_cast<size_t>(ci) * n_cls + n_cls);

        const double ig = l1_ig(cnt_global, cnt_in, ni, n_train, n_cls);

        // NaN ig (n_out == 0) and ig <= G are both skipped — NaN > G is false
        // in IEEE 754, matching explore()'s ux->ig > G guard exactly.
        if (!(ig > G)) continue;  // also rejects NaN

        l1_save(heap, minU, K, iid, utility, ig);
    }

    return heap;
}

}  // namespace hugiml

/**
 * mining_l1.hpp — Hot path for L=1 (singleton-only) pattern mining.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * When L=1 the general THUIsl miner (mining.cpp) builds full utility-list
 * structures — els[] staging buffers, tid_arr[], iu_arr[], ru_arr[] — for
 * every surviving item, runs seal() to copy els into typed arrays, and
 * finally calls explore() which immediately returns at depth 0 after a
 * single save-loop pass.  The iu_arr and ru_arr arrays are never read;
 * the els staging buffer is freed immediately after seal().
 *
 * This module replaces that work for L=1 with a single forward pass over
 * td.transactions that accumulates per-item class counts directly.  No UL
 * objects are constructed; IG is computed from count vectors; the same
 * top-K min-heap and minU update logic is applied.  The PatternEntry
 * output is bit-for-bit equivalent to the general path.
 *
 * Caller contract
 * ───────────────
 *   • Called only when L == 1.
 *   • td.RIU must be populated (by prepare_transactions_cpp) before call.
 *   • ytrain must be pre-extracted to std::vector<int> (GIL already held
 *     by binding layer before release); same requirement as mine_patterns_cpp.
 *   • timeout_s semantics identical to mine_patterns_cpp (0 = no limit).
 */

#pragma once

#include "mining.hpp"   // PatternEntry, TransactionDataCpp

namespace hugiml {

/// Hot path entry point: mine singleton patterns (L=1) without building ULs.
/// Bit-for-bit equivalent output to mine_patterns_cpp(..., L=1, ...).
std::vector<PatternEntry> mine_patterns_l1_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, double G,
    double timeout_s = 0.0);

}  // namespace hugiml

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

#include <atomic>
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

    // els is used only during construction (add/seal/compute_ig).
    // After sealing it is cleared to reclaim memory on large datasets.
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

    // Release the els staging buffer after compute_ig to free memory.
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
    UL   child_ul(const UL& p_ul, const UL& x_ul);
    void mine(const TransList& transactions,
              const std::vector<double>& item_twu,
              const std::vector<int>& ytrain, int n_cls);

private:
    int _explore_calls = 0;

    void explore(std::vector<int>  prefix,
                 std::vector<UL*>& uls,
                 const std::vector<int>& y_arr,
                 int n_cls, int depth, FMap& fmap);
};

// ── Entry point called from Python bindings ──────────────────────────────────
// ytrain is pre-extracted to std::vector<int> by the binding layer (GIL held)
// so this function is safe to call with the GIL released.
// timeout_s: wall-clock seconds before mining is aborted (0 = no limit).
std::vector<PatternEntry> mine_patterns_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, int L, double G,
    double timeout_s = 0.0);

}  // namespace hugiml

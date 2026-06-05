/**
 * mining_l2.hpp — Hot path for L=2 pattern mining.
 */
#pragma once

#include "mining.hpp"

namespace hugiml {

/// Hot path entry point: mine global top-K patterns of length 1 or 2 only.
/// The result is intended to be output-equivalent to mine_patterns_cpp(..., L=2, ...).
std::vector<PatternEntry> mine_patterns_l2_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, double G,
    double timeout_s = 0.0);

}  // namespace hugiml

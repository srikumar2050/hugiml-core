/**
 * mining_l3.hpp — Hot path for L=3 pattern mining.
 */
#pragma once

#include "mining.hpp"

namespace hugiml {

std::vector<PatternEntry> mine_patterns_l3_cpp(
    const TransactionDataCpp& td,
    const std::vector<int>&   ytrain,
    int n_cls, int K, double G,
    double timeout_s = 0.0);

}  // namespace hugiml

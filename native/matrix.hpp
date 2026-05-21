/**
 * matrix.hpp — Feature matrix construction
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include "mining.hpp"
#include <utility>

namespace hugiml {

using COO = std::pair<std::vector<int32_t>, std::vector<int32_t>>;

COO build_matrix_cpp(const TransList& transactions,
                     const std::vector<PatternEntry>& patterns,
                     int n);

COO build_train_matrix_cpp(const TransactionDataCpp& td,
                           const std::vector<PatternEntry>& patterns);

// GIL-safe variant: categorical data is pre-extracted to C++ by the binding
// layer before the GIL is released.
COO build_test_matrix_cpp(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& X_raw_arr,
    const TransactionDataCpp& td,
    std::vector<std::vector<std::string>> cat_strs,
    std::vector<std::vector<bool>>        cat_valid,
    const std::vector<PatternEntry>& patterns);

}  // namespace hugiml

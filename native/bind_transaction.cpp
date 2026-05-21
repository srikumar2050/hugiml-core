/**
 * bind_transaction.cpp — Python bindings for TransactionDataCpp.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "pybind_common.hpp"
#include "transaction.hpp"

namespace py = pybind11;
using namespace hugiml;

void bind_transaction(py::module_& m)
{
    py::class_<TransactionDataCpp>(m, "TransactionDataCpp",
        "C++ counterpart of _TransactionData; holds all training-time artefacts.")
        .def(py::init<>())
        .def_property_readonly("item_map", &TransactionDataCpp::get_item_map_py,
             "dict {item_id (int) -> label (str)}")
        .def_readonly("item_twu", &TransactionDataCpp::item_twu,
             "Transaction-Weighted Utility per item (0-indexed).")
        .def_readonly("nb_col", &TransactionDataCpp::nb_col,
             "Number of bins / categories per column.")
        .def_readonly("all_edges", &TransactionDataCpp::all_edges,
             "Raw bin edges per column.")
        .def_readonly("col_min", &TransactionDataCpp::col_min,
             "Per-column minimum for MinMax scaling.")
        .def_readonly("col_range", &TransactionDataCpp::col_range,
             "Per-column range for MinMax scaling.")
        .def_readonly("is_cat_v", &TransactionDataCpp::is_cat_v,
             "Boolean mask: True for categorical columns.")
        .def_readonly("is_int_v", &TransactionDataCpp::is_int_v,
             "Boolean mask: True for integer columns.")
        .def_readonly("cat_categories", &TransactionDataCpp::cat_categories,
             "Sorted category labels per categorical column.")
        .def_readonly("bn2id", &TransactionDataCpp::bn2id,
             "Bin-key to item-ID mapping (bkey = bi*stride+j).")
        .def_readonly("bkey_stride", &TransactionDataCpp::bkey_stride,
             "Stride used in bin-key encoding: bkey(bi, j) = bi * stride + j.")
        .def("riu_thresh", &TransactionDataCpp::riu_thresh,
             py::arg("k"),
             "Return the k-th largest RIU value (0.0 when k exceeds list length).")
        .def("memory_usage_bytes", &TransactionDataCpp::memory_usage_bytes,
             "Estimate the memory footprint of this object in bytes.");
}

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

#include <algorithm>


static TransactionDataCpp make_transaction_data_from_transactions(
    const TransList& transactions,
    int n_items,
    std::vector<int> item_col_in = {})
{
    int max_item = n_items;
    for (const auto& t : transactions)
        for (const auto& iu : t)
            if (iu.first > max_item) max_item = iu.first;

    TransactionDataCpp td;
    td.transactions = transactions;
    td.item_twu.assign(static_cast<size_t>(std::max(max_item, 0)), 0.0);
    td.RIU.assign(static_cast<size_t>(std::max(max_item, 0)), 0.0);
    if (!item_col_in.empty()) {
        if (static_cast<int>(item_col_in.size()) < max_item)
            throw std::invalid_argument("item_col length must be >= number of items");
        td.item_col.assign(item_col_in.begin(), item_col_in.begin() + max_item);
    } else {
        td.item_col.resize(static_cast<size_t>(std::max(max_item, 0)));
        for (int iid = 1; iid <= max_item; ++iid)
            td.item_col[static_cast<size_t>(iid - 1)] = iid - 1;
    }
    td.disc_n = static_cast<int>(transactions.size());
    td.disc_p = max_item;

    for (const auto& t : td.transactions) {
        if (t.size() == 1 && t[0].first == -1) continue;
        double tu = 0.0;
        for (const auto& [iid, u] : t) tu += u;
        for (const auto& [iid, u] : t) {
            if (iid <= 0 || iid > max_item) continue;
            td.item_twu[static_cast<size_t>(iid - 1)] += tu;
            td.RIU[static_cast<size_t>(iid - 1)] += u;
            td.item_map.emplace(iid, std::to_string(iid));
        }
    }
    return td;
}


void bind_transaction(py::module_& m)
{
    py::class_<TransactionDataCpp>(m, "TransactionDataCpp",
        "C++ counterpart of _TransactionData; holds all training-time artefacts.")
        .def(py::init<>())
        .def(py::init(&make_transaction_data_from_transactions),
             py::arg("transactions"), py::arg("n_items") = 0,
             py::arg("item_col") = std::vector<int>{},
             "Build TransactionDataCpp directly from [[(item, utility), ...], ...].")
        .def_readonly("transactions", &TransactionDataCpp::transactions,
             "Raw transaction list as [(item_id, utility), ...] per row.")
        .def_readonly("RIU", &TransactionDataCpp::RIU,
             "Real item utilities per item (0-indexed).")
        .def_property_readonly("item_map", &TransactionDataCpp::get_item_map_py,
             "dict {item_id (int) -> label (str)}")
        .def_readonly("item_twu", &TransactionDataCpp::item_twu,
             "Transaction-Weighted Utility per item (0-indexed).")
        .def_readonly("item_col", &TransactionDataCpp::item_col,
             "Original source column for each item (0-indexed by item ID - 1).")
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
        .def_readonly("is_precoded_v", &TransactionDataCpp::is_precoded_v,
             "Boolean mask: True for pre-coded adaptive-binning columns.")
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

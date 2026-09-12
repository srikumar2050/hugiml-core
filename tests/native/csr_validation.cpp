// Copyright 2026 Srikumar Krishnamoorthy
// SPDX-License-Identifier: Apache-2.0
#include "csr_indices.hpp"
#include <pybind11/stl.h>

PYBIND11_MODULE(_hugiml_index_validation, m) {
    m.def("index_bits", &hugiml::csr_index_bits,
          pybind11::arg("rows"), pybind11::arg("cols"), pybind11::arg("nnz"),
          pybind11::arg("address_bits") = sizeof(void*) * 8);
    m.def("native_dimension", &hugiml::checked_native_dimension);
    m.def("csr32", &hugiml::csr_from_coo_typed<int32_t>);
    m.def("csr64", &hugiml::csr_from_coo_typed<int64_t>);
    m.def("csr_auto", &hugiml::csr_from_coo);
}

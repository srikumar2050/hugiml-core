/**
 * bind_openmp.cpp — Python bindings for OpenMP thread control.
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

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

void bind_openmp(py::module_& m)
{
#ifdef _OPENMP
    m.def("openmp_available", []() { return true; },
          "Returns True if the extension was compiled with OpenMP support.");
    m.def("openmp_get_max_threads", []() {
        return omp_get_max_threads();
    }, "Return the maximum number of OpenMP threads.");
    m.def("openmp_set_num_threads", [](int n) {
        if (n < 1)
            throw std::invalid_argument(
                "Thread count must be >= 1, got " + std::to_string(n));
        omp_set_num_threads(n);
    }, py::arg("n"), "Set the number of OpenMP threads for matrix construction.");
#else
    m.def("openmp_available", []() { return false; },
          "Returns True if the extension was compiled with OpenMP support.");
    m.def("openmp_get_max_threads", []() { return 1; },
          "Return 1 (OpenMP not available).");
    m.def("openmp_set_num_threads", [](int) {},
          py::arg("n"), "No-op (OpenMP not available).");
#endif
}

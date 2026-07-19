/**
 * bindings.cpp — Module entry point; delegates to sub-binders.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The module is deliberately split into focused TUs so that each one
 * can be compiled in parallel and the per-TU pybind11 template burden
 * remains manageable.  The precompiled header (pybind_common.hpp) is
 * shared across all TUs to avoid redundant template instantiation.
 */

#include "pybind_common.hpp"

namespace py = pybind11;

void bind_transaction(py::module_& m);
void bind_pattern(py::module_& m);
void bind_openmp(py::module_& m);
void bind_augmented_pair(py::module_& m);
void bind_rpte_scoring(py::module_& m);
void bind_rpte_tree(py::module_& m);

PYBIND11_MODULE(_hugiml_core, m)
{
    m.doc() = "HUGIMLClassifierNative — C++ core extension (pybind11)";
    bind_transaction(m);
    bind_pattern(m);
    bind_openmp(m);
    bind_augmented_pair(m);
    bind_rpte_scoring(m);
    bind_rpte_tree(m);
}

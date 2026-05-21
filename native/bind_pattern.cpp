/**
 * bind_pattern.cpp — Thin aggregator that calls the split binding registrars.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 *
 * The three heavy pybind11 lambda bodies live in separate translation units
 * (bind_prepare_tx.cpp, bind_mine_patterns.cpp, bind_build_matrix.cpp) so
 * that the parallel build scheduler can compile them concurrently.  This file
 * simply calls all three registrars from the top-level bind_pattern() entry
 * point invoked from bindings.cpp.
 *
 * Before this split, bind_pattern.cpp was a single 297-line TU.  At -O3 on a
 * single-core machine it could take > 60 s due to pybind11 template expansion
 * inside large lambdas.  With three ~80-line TUs compiled in parallel the
 * wall-clock time falls to the cost of the slowest individual TU (~20 s at O2).
 */

#include "pybind_common.hpp"

namespace py = pybind11;

// Declared in each split TU:
void bind_prepare_transactions(py::module_& m);
void bind_mine_patterns(py::module_& m);
void bind_build_matrix(py::module_& m);

void bind_pattern(py::module_& m)
{
    bind_prepare_transactions(m);
    bind_mine_patterns(m);
    bind_build_matrix(m);
}

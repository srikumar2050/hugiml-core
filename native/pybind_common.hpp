/**
 * pybind_common.hpp — shared pybind11 include chain.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Intended as a precompiled header (PCH).  Every translation unit that
 * participates in the pybind11 module includes this file first so that
 * the compiler can reuse the PCH instead of re-instantiating the full
 * pybind11 template machinery in each TU.
 *
 * Usage in setup.py when GCC/Clang PCH support is available:
 *   extra_compile_args += ["-include", "native/pybind_common.hpp"]
 * The build system pre-compiles this header to pybind_common.hpp.gch
 * and the compiler finds it automatically.
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

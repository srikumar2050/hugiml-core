# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build configuration for the _hugiml_core C++ extension.

Build modes
-----------
Local development (fast, O0 binding TUs + O1 algo TUs):
    python setup.py build_ext --inplace

    # Or explicitly:
    HUGIML_FAST_BUILD=1 python setup.py build_ext --inplace

Release / production (O2 everywhere, default when invoked via pip):
    pip install . --no-build-isolation
    pip wheel . -w dist/

Sanitizers (Linux/macOS):
    HUGIML_SANITIZE=address,undefined pip install . --no-build-isolation

Debug build:
    HUGIML_DEBUG=1 pip install . --no-build-isolation

Parallel workers (default: CPU count, min 1):
    HUGIML_BUILD_JOBS=4 pip install . --no-build-isolation

ccache acceleration (auto-detected):
    CC="ccache gcc" CXX="ccache g++" pip install . --no-build-isolation

Expected build times (single core, 2 GHz, Linux):
    Development (HUGIML_FAST_BUILD=1):  ~35 s
    Release (default pip install):       ~60 s

Design note: binding TUs (bind_*.cpp, bindings.cpp) are always compiled at
O0 regardless of build mode.  pybind11 template glue code is latency-bound
on the compiler front-end and optimizer passes provide no runtime benefit
for pure-dispatch code.  Algorithm TUs (mining.cpp, transaction.cpp, etc.)
keep the release optimization level.
"""

import glob
import multiprocessing
import os
import platform
import shutil

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

is_windows = platform.system() == "Windows"
is_macos = platform.system() == "Darwin"

# ── Compiler selection ────────────────────────────────────────────────────────


def _wrap_ccache(var: str, fallback: str) -> str:
    if os.environ.get(var):
        return os.environ[var]
    if shutil.which("ccache") and shutil.which(fallback):
        return f"ccache {fallback}"
    return fallback


if not is_windows:
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    
# ── Base flags (platform-specific) ───────────────────────────────────────────

if is_windows:
    omp_compile = ["/openmp"]
    omp_link: list = []
    _opt_release = ["/O2", "/W3"]
    _opt_fast = ["/O1", "/W3"]
    _opt_bind = ["/Od", "/W3"]
    _opt_debug = ["/Od", "/Zi", "/DHUGIML_DEBUG"]
elif is_macos:
    omp_compile = ["-Xpreprocessor", "-fopenmp"]
    omp_link = ["-lomp"]
    _opt_release = ["-O2", "-Wall", "-Wextra", "-Wno-unused-parameter"]
    _opt_fast = ["-O1", "-Wall", "-Wno-unused-parameter"]
    _opt_bind = ["-O0", "-g0", "-Wall", "-Wno-unused-parameter"]
    _opt_debug = ["-O0", "-g", "-DHUGIML_DEBUG"]
else:
    omp_compile = ["-fopenmp"]
    omp_link = ["-fopenmp"]
    _opt_release = ["-O2", "-Wall", "-Wextra", "-Wno-unused-parameter"]
    _opt_fast = ["-O1", "-Wall", "-Wno-unused-parameter"]
    _opt_bind = ["-O0", "-g0", "-Wall", "-Wno-unused-parameter"]
    _opt_debug = ["-O0", "-g", "-DHUGIML_DEBUG"]

# ── Build mode ────────────────────────────────────────────────────────────────
# When invoked as "python setup.py build_ext --inplace" (local dev),
# default to HUGIML_FAST_BUILD to avoid long O2 waits.
# pip / build-backend invocations use the full O2 release path.

_is_direct_build = "build_ext" in __import__("sys").argv
_fast = bool(os.environ.get("HUGIML_FAST_BUILD") or _is_direct_build)

sanitize = os.environ.get("HUGIML_SANITIZE", "")

if os.environ.get("HUGIML_DEBUG"):
    algo_args = _opt_debug + omp_compile
    link_args = list(omp_link)
elif sanitize and not is_windows:
    san_flags = [f"-fsanitize={sanitize}", "-fno-omit-frame-pointer", "-g"]
    algo_args = ["-O1"] + san_flags + omp_compile
    link_args = san_flags + list(omp_link)
elif _fast:
    algo_args = _opt_fast + omp_compile
    link_args = list(omp_link)
else:
    algo_args = _opt_release + omp_compile
    link_args = list(omp_link)

# Binding TUs always use O0: optimizer provides zero benefit for pybind11 glue
# and drives >50% of total build time on single-core dev machines.
bind_args = _opt_bind + omp_compile

# ── Sources ───────────────────────────────────────────────────────────────────

_ALGO_SOURCES = [
    "native/math.cpp",
    "native/discretization.cpp",
    "native/transaction.cpp",
    "native/mining.cpp",
    "native/matrix.cpp",
]

_BIND_SOURCES = [
    "native/bind_transaction.cpp",
    "native/bind_pattern.cpp",
    "native/bind_prepare_tx.cpp",
    "native/bind_mine_patterns.cpp",
    "native/bind_build_matrix.cpp",
    "native/bind_openmp.cpp",
    "native/bindings.cpp",
]

_BIND_BASENAMES = frozenset(os.path.basename(s) for s in _BIND_SOURCES)

# ── Extension ─────────────────────────────────────────────────────────────────
# All sources share the same Extension object (required for a single .so).
# Per-TU compile flags are applied in the build_ext command below.

ext = Pybind11Extension(
    "_hugiml_core",
    sources=_ALGO_SOURCES + _BIND_SOURCES,
    include_dirs=["native"],
    cxx_std=17,
    # algo_args are the default; _SplitOptBuildExt overrides for bind TUs.
    extra_compile_args=algo_args,
    extra_link_args=link_args,
)


# ── Custom build_ext: parallel + per-TU optimization ─────────────────────────


class _SplitOptBuildExt(build_ext):
    """Compile algorithm TUs at release level; binding TUs at O0.

    pybind11 binding glue (bind_*.cpp, bindings.cpp) contains only
    type-dispatch boilerplate that the optimizer cannot improve at runtime.
    Compiling them at O0 cuts their individual TU time from ~10s to ~3s
    without any reduction in extension performance.
    """

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.parallel is None:
            try:
                jobs = int(os.environ.get("HUGIML_BUILD_JOBS", "0"))
                self.parallel = jobs if jobs > 0 else multiprocessing.cpu_count()
            except (ValueError, NotImplementedError):
                self.parallel = 1

    def build_extensions(self) -> None:
        # UnixCCompiler._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
        # MSVCCompiler._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
        # Both have the same signature; extra_postargs holds our compile flags.
        orig_compile = self.compiler._compile

        def _per_source_compile(
            obj: str,
            src: str,
            ext_suffix: str,
            cc_args: list,
            extra_postargs: list,
            pp_opts: list,
        ) -> None:
            if os.path.basename(src) in _BIND_BASENAMES:
                extra_postargs = bind_args
            orig_compile(obj, src, ext_suffix, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = _per_source_compile  # type: ignore[method-assign]
        try:
            super().build_extensions()
        finally:
            self.compiler._compile = orig_compile  # type: ignore[method-assign]


# ── Custom build_py: strip native/ from Python package staging ────────────────


class _BuildPyNoNative(_build_py):
    def run(self) -> None:
        super().run()
        for pattern in ("_native", "native"):
            for d in glob.glob(os.path.join(self.build_lib, pattern)):
                shutil.rmtree(d, ignore_errors=True)


# ── Custom sdist: abort if compiled artefacts are present ────────────────────


class _SdistNoSo(_sdist):
    def run(self) -> None:
        so_files = (
            glob.glob("native/**/*.so", recursive=True)
            + glob.glob("native/**/*.pyd", recursive=True)
            + glob.glob("src/**/*.so", recursive=True)
            + glob.glob("src/**/*.pyd", recursive=True)
        )
        if so_files:
            raise RuntimeError(
                "Pre-built compiled artefacts found in the source tree. "
                "Remove them before building an sdist:\n" + "\n".join(f"  {f}" for f in so_files)
            )
        super().run()


setup(
    ext_modules=[ext],
    cmdclass={
        "build_ext": _SplitOptBuildExt,
        "build_py": _BuildPyNoNative,
        "sdist": _SdistNoSo,
    },
)

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
    python -m pip install -e .
    python scripts/build_batched.py --inplace

    # Or explicitly:
    HUGIML_FAST_BUILD=1 python setup.py build_ext --inplace

Release / production (O2 everywhere, default when invoked via pip):
    python -m pip install .
    python -m pip wheel . -w dist/

Sanitizers (Linux/macOS):
    HUGIML_SANITIZE=address,undefined python -m pip install .

Debug build:
    HUGIML_DEBUG=1 python -m pip install .

Batching / parallelism controls:
    HUGIML_BUILD_BATCH_SIZE=4 python scripts/build_batched.py --inplace
    HUGIML_BUILD_JOBS=2 python scripts/build_batched.py --inplace

Avoid --no-build-isolation unless build requirements from pyproject.toml
(pybind11, setuptools, wheel) are already installed in the active environment.

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
import sys
from collections.abc import Sequence
from typing import Any, Callable, cast

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ModuleNotFoundError as exc:  # pragma: no cover - exercised before setup imports complete
    if exc.name != "pybind11":
        raise
    raise SystemExit(
        "pybind11 is required to build hugiml-core from source. "
        "Use an isolated PEP 517 build such as `python -m pip install .` or "
        "install build requirements first with `python -m pip install pybind11 wheel`. "
        "Avoid `--no-build-isolation` unless those requirements are already installed."
    ) from exc
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

is_windows = platform.system() == "Windows"
is_macos = platform.system() == "Darwin"

# ── Compiler selection ────────────────────────────────────────────────────────

if not is_windows:
    # Strip any accidental 'ccache' prefix
    cc = os.environ.get("CC", "gcc").split()[-1]
    cxx = os.environ.get("CXX", "g++").split()[-1]
    os.environ["CC"] = cc
    os.environ["CXX"] = cxx


# ── Base flags (platform-specific) ───────────────────────────────────────────

if is_windows:
    omp_compile = ["/openmp"]
    omp_link: list = []
    _opt_release = ["/O2", "/W3"]
    _opt_fast = ["/O1", "/W3"]
    _opt_bind = ["/Od", "/W3"]
    _opt_debug = ["/Od", "/Zi", "/DHUGIML_DEBUG"]
elif is_macos:
    import re
    import subprocess

    def _env_bool(name: str, *, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _version_tuple(value: str) -> tuple[int, ...]:
        parts = []
        for piece in value.split("."):
            if not piece.isdigit():
                break
            parts.append(int(piece))
        return tuple(parts or [0])

    def _macos_dylib_minos(dylib: str) -> str:
        """Return the LC_BUILD_VERSION minos value for a dylib, or ''."""
        try:
            output = subprocess.check_output(
                ["otool", "-l", dylib],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            return ""
        match = re.search(r"\bminos\s+(\d+(?:\.\d+)*)", output)
        return match.group(1) if match else ""

    def _candidate_libomp_prefix() -> str:
        # CI-safe override.  This must win over Homebrew so cibuildwheel can point
        # at a deployment-target-compatible OpenMP runtime.
        explicit = os.environ.get("HUGIML_LIBOMP_ROOT", "").strip()
        if explicit:
            return explicit

        # Homebrew bottles can be built for the runner OS.  On newer macOS runners
        # that can produce a libomp.dylib with minos newer than the wheel tag
        # (e.g. libomp minos 26.0 inside a macosx_15_0 wheel), which delocate
        # rejects.  Make Homebrew fallback opt-out-able and validate it below.
        if _env_bool("HUGIML_NO_BREW_LIBOMP"):
            return ""

        try:
            return (
                subprocess.check_output(["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL)
                .strip()
                .decode()
            )
        except Exception:
            return ""

    libomp_prefix = _candidate_libomp_prefix()
    if libomp_prefix:
        deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "").strip()
        libomp_dylib = os.path.join(libomp_prefix, "lib", "libomp.dylib")
        libomp_minos = _macos_dylib_minos(libomp_dylib) if os.path.exists(libomp_dylib) else ""

        if (
            deployment_target
            and libomp_minos
            and _version_tuple(libomp_minos) > _version_tuple(deployment_target)
            and not _env_bool("HUGIML_ALLOW_INCOMPATIBLE_LIBOMP")
        ):
            # Do not silently create a wheel that delocate will later reject.
            raise RuntimeError(
                "Selected libomp.dylib is newer than the requested macOS deployment "
                f"target: {libomp_dylib} has minos {libomp_minos}, but "
                f"MACOSX_DEPLOYMENT_TARGET={deployment_target}. "
                "Use HUGIML_LIBOMP_ROOT to point at a compatible libomp build, "
                "set HUGIML_NO_BREW_LIBOMP=1 to build the macOS wheel without "
                "OpenMP, or intentionally raise MACOSX_DEPLOYMENT_TARGET."
            )

        omp_compile = [f"-I{libomp_prefix}/include", "-Xpreprocessor", "-fopenmp"]
        omp_link = [f"-L{libomp_prefix}/lib", "-lomp", f"-Wl,-rpath,{libomp_prefix}/lib"]
    elif _env_bool("HUGIML_NO_BREW_LIBOMP"):
        # Serial macOS build: avoids adding any libomp dependency to the wheel.
        # The native code already guards OpenMP-only calls with _OPENMP.
        omp_compile = []
        omp_link = []
    else:
        # Last-resort local-source build path.  This may work when the compiler
        # toolchain already knows where libomp lives; CI wheels should prefer
        # HUGIML_LIBOMP_ROOT or HUGIML_NO_BREW_LIBOMP=1.
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
# Editable wheels use the development path; normal wheels use full O2.

_is_direct_build = "build_ext" in sys.argv
_is_editable_build = any(arg in {"editable_wheel", "develop"} for arg in sys.argv)
_fast = bool(
    os.environ.get("HUGIML_FAST_BUILD")
    or _is_direct_build
    or _is_editable_build
)


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _default_build_jobs() -> int:
    try:
        cpu_count = multiprocessing.cpu_count()
    except NotImplementedError:
        cpu_count = 1
    # Keep the default conservative for memory-constrained pybind11 builds.
    # Users can raise this with HUGIML_BUILD_JOBS.
    return max(1, min(cpu_count, _env_int("HUGIML_BUILD_BATCH_SIZE", 4, minimum=1), 2))


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
    "native/mining_l1.cpp",
    "native/mining_l2.cpp",
    "native/mining_l3.cpp",
    "native/matrix.cpp",
    "native/prepare_mine_l1.cpp",
    "native/augmented_pair.cpp",
    "native/rpte_scoring.cpp",
    "native/rpte_tree.cpp",
]

_BIND_SOURCES = [
    "native/bind_transaction.cpp",
    "native/bind_pattern.cpp",
    "native/bind_prepare_tx.cpp",
    "native/bind_mine_patterns.cpp",
    "native/bind_prepare_mine_l1.cpp",
    "native/bind_build_matrix.cpp",
    "native/bind_openmp.cpp",
    "native/bindings.cpp",
    "native/bind_rpte_tree.cpp",
]

_BIND_BASENAMES = frozenset(os.path.basename(s) for s in _BIND_SOURCES)

# ── Extension ─────────────────────────────────────────────────────────────────
# All sources share the same Extension object (required for a single .so).
# Per-TU compile flags are applied in the build_ext command below.

core_ext = Pybind11Extension(
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
        parallel = cast(Any, self).parallel
        if parallel is None:
            cast(Any, self).parallel = _env_int(
                "HUGIML_BUILD_JOBS", _default_build_jobs(), minimum=1
            )

    def build_extensions(self) -> None:
        # UnixCCompiler._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
        # MSVCCompiler._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
        # Both have the same signature; extra_postargs holds our compile flags.
        compiler = cast(Any, self.compiler)
        orig_compile_one: Callable[..., None] = compiler._compile
        orig_compile_many: Callable[..., list[str]] = compiler.compile

        def _per_source_compile(
            obj: str,
            src: str,
            ext_suffix: str,
            cc_args: list[Any],
            extra_postargs: list[Any],
            pp_opts: list[Any],
        ) -> None:
            if os.path.basename(src) in _BIND_BASENAMES:
                extra_postargs = bind_args
            orig_compile_one(obj, src, ext_suffix, cc_args, extra_postargs, pp_opts)

        def _batched_compile(sources: Sequence[str], *args: Any, **kwargs: Any) -> list[str]:
            batch_size = _env_int("HUGIML_BUILD_BATCH_SIZE", 4, minimum=0)
            source_list = list(sources)
            if batch_size <= 0 or len(source_list) <= batch_size:
                return orig_compile_many(source_list, *args, **kwargs)

            objects: list[str] = []
            total_batches = (len(source_list) + batch_size - 1) // batch_size
            for batch_index, start in enumerate(range(0, len(source_list), batch_size), start=1):
                batch = source_list[start : start + batch_size]
                self.announce(
                    f"building _hugiml_core source batch {batch_index}/{total_batches} "
                    f"({len(batch)} translation units)",
                    level=2,
                )
                objects.extend(orig_compile_many(batch, *args, **kwargs))
            return objects

        compiler._compile = _per_source_compile
        compiler.compile = _batched_compile
        try:
            super().build_extensions()
        finally:
            compiler.compile = orig_compile_many
            compiler._compile = orig_compile_one


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
    ext_modules=[core_ext],
    cmdclass={
        "build_ext": _SplitOptBuildExt,
        "build_py": _BuildPyNoNative,
        "sdist": _SdistNoSo,
    },
)

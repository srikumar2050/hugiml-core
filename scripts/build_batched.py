#!/usr/bin/env python3
"""Build the native HUG-IML extension with conservative batching defaults.

This helper is intended for local development and constrained CI workers. It
keeps the standard setuptools/PEP 517 build path intact, but supplies safer
defaults for direct ``setup.py build_ext`` rebuilds where build isolation is not
available.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "build" / "logs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("HUGIML_BUILD_BATCH_SIZE", "4")),
        help="Number of C++ translation units per compiler batch. Use 0 to disable batching.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("HUGIML_BUILD_JOBS", "2")),
        help="Maximum compiler workers passed to build_ext.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        default=True,
        help="Build the extension into the source tree for local tests. This is the default.",
    )
    parser.add_argument(
        "--no-inplace",
        action="store_false",
        dest="inplace",
        help="Build under build/lib instead of the source tree.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove build/temp.* object files before compiling.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Use the local-development fast build flags. This is the default.",
    )
    parser.add_argument(
        "--release-flags",
        action="store_false",
        dest="fast",
        help="Do not set HUGIML_FAST_BUILD before invoking setup.py.",
    )
    return parser.parse_args()


def _check_pybind11() -> None:
    try:
        __import__("pybind11")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "pybind11 is not installed in this environment. Run `python -m pip install pybind11` "
            "for direct batched builds, or use an isolated build such as `python -m pip install .`."
        ) from exc


def _clean_temp_objects() -> None:
    build_dir = ROOT / "build"
    if not build_dir.exists():
        return
    for path in build_dir.glob("temp.*"):
        if path.is_dir():
            for obj in path.rglob("*"):
                if obj.is_file() and obj.suffix.lower() in {".o", ".obj"}:
                    obj.unlink()


def main() -> int:
    args = _parse_args()
    _check_pybind11()
    if args.clean:
        _clean_temp_objects()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HUGIML_BUILD_BATCH_SIZE"] = str(max(0, args.batch_size))
    env["HUGIML_BUILD_JOBS"] = str(max(1, args.jobs))
    if args.fast:
        env.setdefault("HUGIML_FAST_BUILD", "1")

    cmd = [
        sys.executable,
        "setup.py",
        "build_ext",
        f"--parallel={max(1, args.jobs)}",
    ]
    if args.inplace:
        cmd.append("--inplace")

    log_path = LOG_DIR / "build_batched.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + " ".join(cmd) + "\n")
        log.write(f"HUGIML_BUILD_BATCH_SIZE={env['HUGIML_BUILD_BATCH_SIZE']}\n")
        log.write(f"HUGIML_BUILD_JOBS={env['HUGIML_BUILD_JOBS']}\n")
        log.write(f"HUGIML_FAST_BUILD={env.get('HUGIML_FAST_BUILD', '')}\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        print(f"Batched build failed; see {log_path}", file=sys.stderr)
        return proc.returncode

    print(f"Batched build completed; log written to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

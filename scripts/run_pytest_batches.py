#!/usr/bin/env python3
"""Run pytest in deterministic file batches with per-batch logs.

The full suite can be heavy on constrained workers. This helper keeps pytest as
the underlying runner while making the default validation path deterministic,
restartable, and easier to audit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "build" / "pytest_batches"


def _display_path(path: Path) -> str:
    """Return a stable repository-relative path, or an absolute external path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["tests"], help="Test files or directories.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("HUGIML_PYTEST_BATCH_SIZE", "5")),
        help="Number of test files per pytest process.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("HUGIML_PYTEST_BATCH_TIMEOUT", "0")),
        help="Timeout per batch in seconds. Use 0 for no timeout.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for pytest batch logs and summary.json.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=None,
        help=(
            "Extra pytest arguments. Put this option last, for example "
            "`--pytest-args -q -m 'not stress'`. If omitted, HUGIML_PYTEST_ARGS "
            "or '-q' is used."
        ),
    )
    parser.add_argument(
        "--no-rerun-timeouts",
        action="store_true",
        help="Do not retry timed-out batches one file at a time.",
    )
    return parser.parse_args()


def _discover_test_files(paths: list[str]) -> list[Path]:
    files: set[Path] = set()
    for raw in paths:
        path = (ROOT / raw).resolve()
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(p.resolve() for p in path.rglob("test*.py") if p.is_file())
        else:
            raise SystemExit(f"Test path does not exist: {raw}")
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def _split_pytest_args(raw: str) -> list[str]:
    import shlex

    return shlex.split(raw)


def _resolve_pytest_args(cli_args: list[str] | None) -> list[str]:
    if cli_args is None:
        return _split_pytest_args(os.environ.get("HUGIML_PYTEST_ARGS", "-q"))
    if len(cli_args) == 1:
        # Support both `--pytest-args=-q` and `--pytest-args '-q -m smoke'`.
        return _split_pytest_args(cli_args[0])
    return cli_args


def _run_pytest(files: list[Path], pytest_args: list[str], log_path: Path, timeout: int) -> dict[str, object]:
    rel_files = [p.relative_to(ROOT).as_posix() for p in files]
    cmd = [sys.executable, "-m", "pytest", *pytest_args, *rel_files]
    start = time.monotonic()
    status = "passed"
    returncode: int | None = None
    timed_out = False

    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + " ".join(cmd) + "\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout if timeout > 0 else None,
            )
            returncode = proc.returncode
            if returncode != 0:
                status = "failed"
        except subprocess.TimeoutExpired:
            status = "timed_out"
            timed_out = True
            log.write(f"\nTimed out after {timeout} seconds.\n")

    elapsed = round(time.monotonic() - start, 3)
    return {
        "files": rel_files,
        "log": _display_path(log_path),
        "returncode": returncode,
        "status": status,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
    }


def main() -> int:
    args = _parse_args()
    test_files = _discover_test_files(args.paths)
    if not test_files:
        raise SystemExit("No pytest files discovered.")

    batch_size = max(1, args.batch_size)
    pytest_args = _resolve_pytest_args(args.pytest_args)
    log_dir = args.log_dir if args.log_dir.is_absolute() else ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "test_file_count": len(test_files),
        "batch_size": batch_size,
        "timeout_seconds": args.timeout,
        "pytest_args": pytest_args,
        "batches": [],
        "single_file_reruns": [],
    }

    failures = 0
    timed_out_batches: list[list[Path]] = []
    batches = [test_files[i : i + batch_size] for i in range(0, len(test_files), batch_size)]
    for index, batch in enumerate(batches, start=1):
        log_path = log_dir / f"batch_{index:03d}.log"
        result = _run_pytest(batch, pytest_args, log_path, args.timeout)
        cast_batches = summary["batches"]
        assert isinstance(cast_batches, list)
        cast_batches.append(result)
        print(f"batch {index}/{len(batches)}: {result['status']} ({result['log']})")
        if result["timed_out"]:
            timed_out_batches.append(batch)
        elif result["returncode"] != 0:
            failures += 1

    if timed_out_batches and not args.no_rerun_timeouts:
        for batch in timed_out_batches:
            for file_path in batch:
                rel = file_path.relative_to(ROOT).as_posix().replace("/", "__")
                log_path = log_dir / f"rerun_{rel}.log"
                result = _run_pytest([file_path], pytest_args, log_path, args.timeout)
                cast_reruns = summary["single_file_reruns"]
                assert isinstance(cast_reruns, list)
                cast_reruns.append(result)
                print(f"rerun {file_path.relative_to(ROOT)}: {result['status']} ({result['log']})")
                if result["returncode"] != 0:
                    failures += 1
    else:
        failures += len(timed_out_batches)

    summary["failed_units"] = failures
    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"summary written to {_display_path(summary_path)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

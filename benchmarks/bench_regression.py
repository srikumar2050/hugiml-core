#!/usr/bin/env python3
# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Performance regression gate for CI.

Measures key benchmarks and compares them against ``baseline.json``.  Exits
with a non-zero status code if any metric exceeds its threshold by more than
``--threshold`` (default 1.5×).  The CI bench-gate job runs this script and
fails the build on regression.

Regression factor precedence
-----------------------------
Each benchmark entry in ``baseline.json`` may define a ``regression_factor``
that represents the upper bound the *baseline author* guarantees.  The CLI
``--threshold`` acts as a **global override** in either direction:

* Passing ``--threshold 2.0`` relaxes ALL checks to 2× regardless of the
  per-benchmark factor.
* Passing ``--threshold 1.2`` tightens ALL checks to 1.2× regardless of the
  per-benchmark factor.

This makes CI-wide tightening/relaxation a one-flag operation without having
to edit every baseline entry.

Usage::

    # Run regression check (exits non-zero on regression)
    python benchmarks/bench_regression.py --check

    # Run without failing (report only)
    python benchmarks/bench_regression.py

    # Override regression factor
    python benchmarks/bench_regression.py --check --threshold 2.0

    # Save result JSON alongside baseline
    python benchmarks/bench_regression.py --check --output benchmarks/results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
BASELINE_PATH = ROOT / "benchmarks" / "baseline.json"
TESTS_DIR = ROOT / "tests"

sys.path.insert(0, str(ROOT / "src"))

GERMAN_COLS = [
    "checking_acct",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def _load_german():
    import pandas as pd

    df = pd.read_csv(
        TESTS_DIR / "german.data",
        sep=" ",
        header=None,
        names=GERMAN_COLS + ["target"],
    )
    df["target"] = (df["target"] == 2).astype(int)
    return df.drop(columns=["target"]), df["target"]


def _load_heloc():
    import pandas as pd

    df = pd.read_csv(TESTS_DIR / "heloc.csv")
    df["target"] = (df["RiskPerformance"] == "Bad").astype(int)
    df = df.drop(columns=["RiskPerformance"])
    return df.drop(columns=["target"]), df["target"]


# ---------------------------------------------------------------------------
# Benchmark cases aligned to baseline.json keys
# ---------------------------------------------------------------------------


def run_benchmarks() -> dict[str, float]:
    """Return a dict mapping benchmark keys (as in baseline.json) to seconds."""
    import tempfile

    import pandas as pd
    from sklearn.model_selection import train_test_split

    from hugiml import HUGIMLClassifierNative
    from hugiml.serialization import load_model, save_model

    results: dict[str, float] = {}

    # ── German Credit (L=1 baseline) ──────────────────────────────────────────
    X_g, y_g = _load_german()
    clf_g = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
    X_gp, y_gp = clf_g.prepareXy(X_g.copy(), y_g.copy())
    X_gtr, X_gte, y_gtr, y_gte = train_test_split(
        X_gp, y_gp, test_size=0.2, random_state=42, stratify=y_gp
    )

    clf_tmp = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
    _, t = _timed(clf_tmp.prepareXy, X_g.copy(), y_g.copy())
    results["prepareXy_german"] = t

    clf_fit = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
    Xgp2, ygp2 = clf_fit.prepareXy(X_g.copy(), y_g.copy())
    Xgtr2, _, ygtr2, _ = train_test_split(Xgp2, ygp2, test_size=0.2, random_state=42, stratify=ygp2)
    _, t = _timed(clf_fit.fit, Xgtr2, ygtr2)
    results["fit_german_credit"] = t
    clf_g.fit(X_gtr, y_gtr)

    n = 1000
    reps = (n // len(X_gte)) + 1
    X_g1k = (
        pd.concat([X_gte] * reps, ignore_index=True).iloc[:n]
        if isinstance(X_gte, pd.DataFrame)
        else X_gte
    )
    _, t = _timed(clf_g.predict_proba, X_g1k)
    results["predict_proba_german_1k"] = t

    _, t = _timed(clf_g.transform, X_g1k)
    results["transform_german_1k"] = t

    with tempfile.NamedTemporaryFile(suffix=".hugiml", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        _, t1 = _timed(save_model, clf_g, tmp_path)
        _, t2 = _timed(load_model, tmp_path)
        results["serialise_roundtrip"] = t1 + t2
    finally:
        tmp_path.unlink(missing_ok=True)

    # ── German Credit — L=2 depth-2 patterns ─────────────────────────────────
    clf_g_l2 = HUGIMLClassifierNative(B=5, L=2, G=0.0, topK=60, max_fit_seconds=60)
    Xg_l2, yg_l2 = clf_g_l2.prepareXy(X_g.copy(), y_g.copy())
    Xgtr_l2, Xgte_l2, ygtr_l2, _ = train_test_split(
        Xg_l2, yg_l2, test_size=0.2, random_state=42, stratify=yg_l2
    )
    _, t = _timed(clf_g_l2.fit, Xgtr_l2, ygtr_l2)
    results["fit_german_l2"] = t

    n_l2 = min(500, len(Xgte_l2))
    reps_l2 = (n_l2 // len(Xgte_l2)) + 1
    X_g_l2_pred = (
        pd.concat([Xgte_l2] * reps_l2, ignore_index=True).iloc[:n_l2]
        if isinstance(Xgte_l2, pd.DataFrame)
        else Xgte_l2
    )
    _, t = _timed(clf_g_l2.predict_proba, X_g_l2_pred)
    results["predict_proba_german_l2_500"] = t

    # ── HELOC ─────────────────────────────────────────────────────────────────
    X_h, y_h = _load_heloc()
    clf_h = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
    X_hp, y_hp = clf_h.prepareXy(X_h.copy(), y_h.copy())
    X_htr, X_hte, y_htr, y_hte = train_test_split(
        X_hp, y_hp, test_size=0.2, random_state=42, stratify=y_hp
    )

    clf_hfit = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
    Xhp2, yhp2 = clf_hfit.prepareXy(X_h.copy(), y_h.copy())
    Xhtr2, _, yhtr2, _ = train_test_split(Xhp2, yhp2, test_size=0.2, random_state=42, stratify=yhp2)
    _, t = _timed(clf_hfit.fit, Xhtr2, yhtr2)
    results["fit_heloc_full"] = t
    clf_h.fit(X_htr, y_htr)

    n = 10_000
    reps = (n // len(X_hte)) + 1
    X_h10k = (
        pd.concat([X_hte] * reps, ignore_index=True).iloc[:n]
        if isinstance(X_hte, pd.DataFrame)
        else X_hte
    )
    _, t = _timed(clf_h.predict_proba, X_h10k)
    results["predict_proba_heloc_10k"] = t

    return results


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------


def check_regression(
    measured: dict[str, float],
    baseline: dict,
    threshold: float,
) -> list[str]:
    """Return a list of failure messages (empty → all passed).

    The effective limit for each benchmark is::

        limit = baseline_seconds * threshold

    The CLI ``--threshold`` is the sole multiplier.  Per-benchmark
    ``regression_factor`` entries in ``baseline.json`` are reported but do
    not override the CLI value, making global CI adjustments a single flag.
    """
    failures = []
    for key, spec in baseline.get("benchmarks", {}).items():
        if key not in measured:
            continue
        limit = spec["baseline_seconds"] * threshold
        actual = measured[key]
        # Report per-benchmark factor as informational guidance
        per_bench_factor = spec.get("regression_factor", threshold)
        guidance = f"(per-bench factor={per_bench_factor}×, CLI threshold={threshold}×)"
        if actual > limit:
            failures.append(
                f"REGRESSION  {key}: {actual:.3f}s > limit {limit:.3f}s "
                f"(baseline {spec['baseline_seconds']:.3f}s × {threshold}) {guidance}"
            )
        else:
            print(f"  OK  {key}: {actual:.3f}s  (limit {limit:.3f}s) {guidance}")
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="HUG-IML performance regression gate")
    parser.add_argument(
        "--check", action="store_true", help="Exit non-zero if a regression is detected"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Global regression factor applied to all benchmarks (default 1.5×)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Directory to write result JSON")
    parser.add_argument(
        "--baseline", type=Path, default=BASELINE_PATH, help="Path to baseline.json"
    )
    args = parser.parse_args()

    print("Running benchmarks …")
    measured = run_benchmarks()

    print("\n--- Measured times ---")
    for k, v in measured.items():
        print(f"  {k:55s}  {v:.3f}s")

    baseline = json.loads(args.baseline.read_text())
    print("\n--- Regression check ---")
    failures = check_regression(measured, baseline, args.threshold)

    if args.output:
        import datetime

        args.output.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out = args.output / f"regression_{ts}.json"
        out.write_text(
            json.dumps(
                {
                    "timestamp": ts,
                    "threshold": args.threshold,
                    "measured": measured,
                    "failures": failures,
                },
                indent=2,
            )
        )
        print(f"\nResult written to {out}")

    if failures:
        print("\nFAILURES:")
        for msg in failures:
            print(f"  {msg}")
        if args.check:
            sys.exit(1)
    else:
        print("\nAll benchmarks within threshold.")


if __name__ == "__main__":
    main()

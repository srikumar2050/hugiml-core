#!/usr/bin/env python3
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

"""Micro-benchmarks for HUG-IML core operations.

Measures wall-clock time for individual pipeline stages.  Results are written
to ``benchmarks/results/`` as JSON so they can be tracked over time in CI.

Usage::

    # Run all benchmarks and print a summary
    python benchmarks/bench_core.py

    # Run with a specific dataset only
    python benchmarks/bench_core.py --dataset german

    # Save results to a custom directory
    python benchmarks/bench_core.py --output benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

TESTS_DIR = ROOT / "tests"
RESULTS_DIR = ROOT / "benchmarks" / "results"

GERMAN_COLS = [
    "checking_acct", "duration", "credit_history", "purpose",
    "credit_amount", "savings", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_plans", "housing", "existing_credits", "job",
    "num_dependents", "telephone", "foreign_worker",
]


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _timed(fn, *args, **kwargs):
    """Return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_german():
    import pandas as pd
    path = TESTS_DIR / "german.data"
    df = pd.read_csv(path, sep=" ", header=None,
                     names=GERMAN_COLS + ["target"])
    df["target"] = (df["target"] == 2).astype(int)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def _load_heloc():
    import pandas as pd
    path = TESTS_DIR / "heloc.csv"
    df = pd.read_csv(path)
    df["target"] = (df["RiskPerformance"] == "Bad").astype(int)
    df = df.drop(columns=["RiskPerformance"])
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------

def bench_prepareXy(clf_cls, X, y, label: str) -> dict:
    clf = clf_cls(B=7, L=1, G=5e-3)
    _, elapsed = _timed(clf.prepareXy, X.copy(), y.copy())
    return {"name": f"prepareXy_{label}", "seconds": elapsed}


def bench_fit(clf_cls, X_tr, y_tr, label: str) -> dict:
    from sklearn.model_selection import train_test_split
    clf = clf_cls(B=7, L=1, G=5e-3)
    X_p, y_p = clf.prepareXy(X_tr.copy(), y_tr.copy())
    X_f, _, y_f, _ = train_test_split(X_p, y_p, test_size=0.1,
                                      random_state=0, stratify=y_p)
    clf2 = clf_cls(B=7, L=1, G=5e-3)
    X_p2, y_p2 = clf2.prepareXy(X_tr.copy(), y_tr.copy())
    _, elapsed = _timed(clf2.fit, X_p2, y_p2)
    return {"name": f"fit_{label}", "seconds": elapsed}


def bench_predict_proba(clf, X_te, label: str) -> dict:
    import pandas as pd
    # Tile to 1 000+ rows for a stable measurement
    n_target = max(1000, len(X_te))
    reps = (n_target // len(X_te)) + 1
    if isinstance(X_te, pd.DataFrame):
        X_large = pd.concat([X_te] * reps, ignore_index=True).iloc[:n_target]
    else:
        import numpy as np
        X_large = np.tile(X_te, (reps, 1))[:n_target]
    _, elapsed = _timed(clf.predict_proba, X_large)
    return {"name": f"predict_proba_{label}_{n_target}", "seconds": elapsed}


def bench_transform(clf, X_te, label: str) -> dict:
    _, elapsed = _timed(clf.transform, X_te)
    return {"name": f"transform_{label}", "seconds": elapsed}


def bench_serialise_roundtrip(clf, label: str) -> dict:
    import tempfile
    from pathlib import Path as _Path

    from hugiml.serialization import load_model, save_model

    with tempfile.NamedTemporaryFile(suffix=".hugiml", delete=False) as f:
        path = _Path(f.name)
    try:
        _, elapsed_save = _timed(save_model, clf, path)
        _, elapsed_load = _timed(load_model, path)
        elapsed = elapsed_save + elapsed_load
    finally:
        path.unlink(missing_ok=True)
    return {"name": f"serialise_roundtrip_{label}", "seconds": elapsed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_dataset(clf_cls, X, y, label: str):
    from sklearn.model_selection import train_test_split

    results = []

    results.append(bench_prepareXy(clf_cls, X, y, label))

    clf = clf_cls(B=7, L=1, G=5e-3)
    X_p, y_p = clf.prepareXy(X.copy(), y.copy())
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_p, y_p, test_size=0.2, random_state=42, stratify=y_p
    )
    clf.fit(X_tr, y_tr)

    results.append(bench_fit(clf_cls, X, y, label))
    results.append(bench_predict_proba(clf, X_te, label))
    results.append(bench_transform(clf, X_te, label))
    results.append(bench_serialise_roundtrip(clf, label))

    return results


def main():
    parser = argparse.ArgumentParser(description="HUG-IML core micro-benchmarks")
    parser.add_argument("--dataset", choices=["german", "heloc", "all"],
                        default="all")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    from hugiml import HUGIMLClassifierNative

    all_results = []

    if args.dataset in ("german", "all"):
        print("Loading German Credit dataset …")
        X_g, y_g = _load_german()
        print(f"  {len(X_g)} samples, {X_g.shape[1]} features")
        all_results += _run_dataset(HUGIMLClassifierNative, X_g, y_g, "german")

    if args.dataset in ("heloc", "all"):
        print("Loading HELOC dataset …")
        X_h, y_h = _load_heloc()
        print(f"  {len(X_h)} samples, {X_h.shape[1]} features")
        all_results += _run_dataset(HUGIMLClassifierNative, X_h, y_h, "heloc")

    # Print summary
    print("\n--- Benchmark Results ---")
    for r in all_results:
        print(f"  {r['name']:50s}  {r['seconds']:.3f}s")

    # Save JSON
    args.output.mkdir(parents=True, exist_ok=True)
    import datetime
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output / f"bench_{ts}.json"
    out_path.write_text(json.dumps({"timestamp": ts, "results": all_results}, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

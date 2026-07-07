#!/usr/bin/env python3
"""Benchmark: mine_patterns_generic(L=3) vs mine_patterns_l3.

Prints a table of wall-clock times (median of N repeats) across a grid of
dataset sizes / feature counts / K / G, and the resulting speedup factor.
Also verifies output-equivalence at each config as a safety net.
"""
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
import _hugiml_core as core  # noqa: E402


def make_data(n, p, n_cls, seed, n_informative=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    n_informative = min(n_informative, p)
    weights = rng.normal(size=n_informative)
    score = np.zeros(n)
    for k in range(n_informative):
        score += weights[k] * (X[:, k] > 0)
    score += 0.3 * rng.normal(size=n)
    if n_cls == 2:
        y = (score > np.median(score)).astype(np.int64)
    else:
        edges = np.quantile(score, np.linspace(0, 1, n_cls + 1)[1:-1])
        y = np.digitize(score, edges).astype(np.int64)
    return X, y


def prep(X, y, n_bins=5):
    p = X.shape[1]
    return core.prepare_transactions(
        X, y, n_bins,
        [f"x{i}" for i in range(p)],
        np.zeros(p, dtype=np.uint8),
        np.zeros(p, dtype=np.uint8),
        None, None,
    )


def timeit(fn, repeats=5):
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, statistics.median(times), min(times)


def sig(patterns):
    def igv(v):
        v = float(v)
        return "nan" if math.isnan(v) else round(v, 9)
    return sorted((round(float(p.utility), 9), tuple(p.items), igv(p.ig)) for p in patterns)


GRID = [
    # (n, p, n_cls, n_bins, K, G, label)
    (400, 12, 2, 5, 50, 1e-4, "small"),
    (1500, 20, 2, 5, 50, 1e-4, "medium"),
    (1500, 20, 2, 5, 200, 1e-5, "medium_bigK"),
    (4000, 25, 2, 6, 50, 1e-4, "large"),
    (4000, 25, 3, 6, 100, 1e-5, "large_3cls"),
    (2500, 40, 2, 5, 50, 1e-4, "wide"),
]


def main():
    rows = []
    for n, p, n_cls, n_bins, K, G, label in GRID:
        X, y = make_data(n, p, n_cls, seed=hash(label) % 1000)
        td = prep(X, y, n_bins=n_bins)

        generic_res, generic_med, generic_min = timeit(
            lambda: core.mine_patterns_generic(td, y, n_cls, K, 3, G, 0.0), repeats=5)
        l3_res, l3_med, l3_min = timeit(
            lambda: core.mine_patterns_l3(td, y, n_cls, K, G, 0.0), repeats=5)

        match = sig(generic_res) == sig(l3_res)
        speedup_med = generic_med / l3_med if l3_med > 0 else float("inf")
        speedup_min = generic_min / l3_min if l3_min > 0 else float("inf")

        row = dict(
            label=label, n=n, p=p, n_cls=n_cls, n_bins=n_bins, K=K, G=G,
            n_patterns=len(l3_res),
            generic_median_ms=generic_med * 1000,
            l3_median_ms=l3_med * 1000,
            generic_min_ms=generic_min * 1000,
            l3_min_ms=l3_min * 1000,
            speedup_median=speedup_med,
            speedup_min=speedup_min,
            outputs_match=match,
        )
        rows.append(row)
        print(f"{label:14s} n={n:5d} p={p:3d} cls={n_cls} K={K:4d} G={G:<8g} "
              f"| generic={generic_med*1000:9.3f}ms  l3={l3_med*1000:9.3f}ms  "
              f"speedup={speedup_med:5.2f}x  match={match}")

    out_path = ROOT / "benchmarks" / "results" / "l3_hotpath_bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

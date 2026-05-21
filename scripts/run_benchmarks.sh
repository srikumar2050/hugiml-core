#!/usr/bin/env bash
# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Convenience wrapper: run micro-benchmarks then check for regressions.
#
# Usage:
#   ./scripts/run_benchmarks.sh            # run + check
#   ./scripts/run_benchmarks.sh --no-check # run without failing on regression
#   THRESHOLD=2.0 ./scripts/run_benchmarks.sh  # custom factor

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

THRESHOLD="${THRESHOLD:-1.5}"
CHECK="${1:-}"

echo "==> Running core benchmarks …"
python benchmarks/bench_core.py --output benchmarks/results/

echo ""
echo "==> Running regression gate (threshold ${THRESHOLD}×) …"
if [[ "$CHECK" == "--no-check" ]]; then
    python benchmarks/bench_regression.py \
        --threshold "$THRESHOLD" \
        --output benchmarks/results/
else
    python benchmarks/bench_regression.py \
        --check \
        --threshold "$THRESHOLD" \
        --output benchmarks/results/
fi

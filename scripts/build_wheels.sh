#!/usr/bin/env bash
# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Builds manylinux-compatible wheels locally using cibuildwheel.
# Requires Docker to be running.
#
# Usage:
#   ./scripts/build_wheels.sh               # build for current platform
#   ./scripts/build_wheels.sh --all         # build linux / macOS / windows
#   CIBW_BUILD="cp311-*" ./scripts/build_wheels.sh  # specific Python version

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WHEELDIR="${ROOT}/dist/wheels"
mkdir -p "$WHEELDIR"

echo "==> Installing cibuildwheel …"
pip install --quiet cibuildwheel

echo "==> Building wheels → ${WHEELDIR}"
CIBW_BUILD="${CIBW_BUILD:-cp39-* cp310-* cp311-* cp312-*}" \
CIBW_ARCHS="${CIBW_ARCHS:-auto}" \
    cibuildwheel --output-dir "$WHEELDIR" "$@"

echo ""
echo "Built wheels:"
ls -1 "$WHEELDIR"

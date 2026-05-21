#!/usr/bin/env python3
# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Cross-platform wheel smoke test.

Verifies that the native extension loads correctly and that a full
fit → predict_proba → save_model → load_model round-trip works on the
current platform / Python interpreter.

Run by cibuildwheel after building each wheel::

    python {project}/scripts/wheel_smoke_test.py

Exit code 0 = all checks passed; non-zero = failure.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path


def _fail(message: str, exc: BaseException | None = None) -> None:
    print(f"\nFAIL: {message}", file=sys.stderr)
    if exc is not None:
        traceback.print_exc()
    sys.exit(1)


def main() -> None:
    print("=== HUG-IML wheel smoke test ===")
    print(f"Python {sys.version}")

    # ── 1. Import ─────────────────────────────────────────────────────────────
    try:
        import hugiml
        from hugiml import HUGIMLClassifierNative

        print(f"[OK] import hugiml {hugiml.__version__}")
    except Exception as exc:
        _fail("import hugiml failed", exc)

    # ── 2. Native extension ───────────────────────────────────────────────────
    try:
        import _hugiml_core as _core

        threads = _core.openmp_get_max_threads()
        print(f"[OK] _hugiml_core loaded (OpenMP threads={threads})")
    except Exception as exc:
        _fail("_hugiml_core native extension not importable", exc)

    # ── 3. Instantiation ──────────────────────────────────────────────────────
    try:
        clf = HUGIMLClassifierNative(B=4, L=1, G=0.0, topK=20)
        print(f"[OK] HUGIMLClassifierNative instantiated: {clf!r}")
    except Exception as exc:
        _fail("HUGIMLClassifierNative() constructor raised", exc)

    # ── 4. Data preparation ───────────────────────────────────────────────────
    try:
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(42)
        n = 120
        X_df = pd.DataFrame(
            {
                "feat_int": rng.integers(0, 8, n),
                "feat_float": rng.standard_normal(n),
                "feat_cat": rng.choice(["A", "B", "C"], n),
            }
        )
        logits = 0.4 * X_df["feat_int"] + 0.6 * X_df["feat_float"]
        prob = 1.0 / (1.0 + np.exp(-logits))
        y_s = pd.Series(rng.binomial(1, prob).astype(int), name="target")

        X_p, y_p = clf.prepareXy(X_df.copy(), y_s.copy())
        print(f"[OK] prepareXy: {X_p.shape}")
    except Exception as exc:
        _fail("prepareXy() raised", exc)

    # ── 5. fit ────────────────────────────────────────────────────────────────
    try:
        from sklearn.model_selection import train_test_split

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_p, y_p, test_size=0.25, random_state=0, stratify=y_p
        )
        clf.fit(X_tr, y_tr)
        n_pats = len(clf.patterns_)
        print(f"[OK] fit() produced {n_pats} patterns")
        assert n_pats > 0, "fit() produced 0 patterns"
    except Exception as exc:
        _fail("fit() raised", exc)

    # ── 6. predict_proba ──────────────────────────────────────────────────────
    try:
        proba = clf.predict_proba(X_te)
        assert proba.shape == (len(X_te), 2), f"unexpected proba shape {proba.shape}"
        row_sums = proba.sum(axis=1)
        assert abs(row_sums.mean() - 1.0) < 1e-5, "probabilities do not sum to 1"
        assert not (proba < 0).any(), "negative probability"
        print(f"[OK] predict_proba shape={proba.shape}, row_sum_mean={row_sums.mean():.6f}")
    except Exception as exc:
        _fail("predict_proba() raised or returned invalid output", exc)

    # ── 7. predict ────────────────────────────────────────────────────────────
    try:
        preds = clf.predict(X_te)
        assert preds.shape == (len(X_te),), f"unexpected preds shape {preds.shape}"
        assert set(preds.tolist()).issubset({0, 1}), "unexpected label values"
        print(f"[OK] predict shape={preds.shape}")
    except Exception as exc:
        _fail("predict() raised or returned invalid output", exc)

    # ── 8. transform ─────────────────────────────────────────────────────────
    try:
        mat = clf.transform(X_te)
        assert mat.shape[0] == len(X_te)
        print(f"[OK] transform shape={mat.shape}")
    except Exception as exc:
        _fail("transform() raised", exc)

    # ── 9. save_model / load_model round-trip ─────────────────────────────────
    try:
        from hugiml.serialization import load_model, save_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "smoke_test.hugiml"
            save_model(clf, model_path)
            assert model_path.exists(), "save_model() did not create file"
            clf2 = load_model(model_path)

        import numpy as np

        p1 = clf.predict(X_te)
        p2 = clf2.predict(X_te)
        assert (p1 == p2).all(), "predictions differ after round-trip"
        print("[OK] save_model / load_model round-trip predictions match")
    except Exception as exc:
        _fail("save_model / load_model round-trip failed", exc)

    # ── 10. sklearn compatibility ─────────────────────────────────────────────
    try:
        from sklearn.utils.estimator_checks import parametrize_with_checks

        checks = list(parametrize_with_checks([HUGIMLClassifierNative()]))
        print(f"[OK] sklearn parametrize_with_checks returned {len(checks)} checks")
    except Exception as exc:
        # Non-fatal: sklearn version may not support parametrize_with_checks
        print(f"[SKIP] sklearn parametrize_with_checks not available: {exc}")

    print("\n=== All smoke tests PASSED ===")


if __name__ == "__main__":
    main()

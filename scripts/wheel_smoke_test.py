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

    # ── 11. HUGIMLMemoryError ───────────────────────────────────────
    try:
        from hugiml.exceptions import HUGIMLFitError, HUGIMLMemoryError

        err = HUGIMLMemoryError("OOM: reduce B")
        assert isinstance(err, HUGIMLFitError), "HUGIMLMemoryError must subclass HUGIMLFitError"
        assert isinstance(err, MemoryError), "HUGIMLMemoryError must subclass MemoryError"
        assert "HUGIMLMemoryError" in __import__("hugiml").__all__
        print("[OK] HUGIMLMemoryError importable and hierarchy correct")
    except Exception as exc:
        _fail("HUGIMLMemoryError check failed", exc)

    # ── 12. TransactionDataCpp item_iu + get_transactions_py ─────────
    try:
        import _hugiml_core as _core

        # Use a fresh all-numeric array — X_tr contains a categorical column
        # and prepare_transactions expects a float64 numeric array.
        rng12 = np.random.default_rng(12)
        n12 = len(y_tr)
        X_num12 = rng12.standard_normal((n12, 3)).astype(np.float64)
        y_num12 = np.asarray(y_tr, dtype=np.int64)
        p12 = X_num12.shape[1]
        td_smoke = _core.prepare_transactions(
            X_num12,
            y_num12,
            4,
            None,
            np.zeros(p12, dtype=np.uint8),
            np.zeros(p12, dtype=np.uint8),
            None,
            [False] * p12,
        )
        n_items = len(td_smoke.item_twu)
        iu_vec = list(td_smoke.item_iu)
        assert len(iu_vec) == n_items, f"item_iu length {len(iu_vec)} != n_items {n_items}"
        assert all(0.0 <= v <= 1.0 + 1e-9 for v in iu_vec), "item_iu out of [0,1] range"
        txs = td_smoke.transactions
        for iid, u in txs[0]:
            if iid != -1:
                assert abs(u - iu_vec[iid - 1]) < 1e-9, "get_transactions_py utility mismatch"
        print(f"[OK] item_iu vector ({n_items} items) and get_transactions_py consistent")
    except Exception as exc:
        _fail("item_iu / get_transactions_py smoke test failed", exc)

    # ── 13. Fused adaptive+L1 hotpath  ────────────────────────────────
    try:
        clf_adap = HUGIMLClassifierNative(
            B=5,
            L=1,
            G=0.0,
            topK=20,
            adaptive_binning=True,
            use_hotpath=True,
        )
        clf_adap.fit(X_tr, y_tr)
        assert len(clf_adap.patterns_) > 0, "fused adaptive L1 fit returned 0 patterns"
        assert hasattr(clf_adap, "_bin_edges_") and len(clf_adap._bin_edges_) > 0, (
            "_bin_edges_ not populated after fused adaptive fit"
        )
        preds_adap = clf_adap.predict(X_te)
        assert preds_adap.shape == (len(X_te),)
        print(
            f"[OK] fused adaptive+L1 hotpath: {len(clf_adap.patterns_)} patterns, "
            f"{len(clf_adap._bin_edges_)} adaptive cols"
        )
    except Exception as exc:
        _fail("fused adaptive+L1 hotpath smoke test failed", exc)

    # ── 14. Fixed-B numeric L1 hotpath ──────────────────
    try:
        import os

        os.environ["HUGIML_ENABLE_FIXED_NUMERIC_L1_FASTPATH"] = "1"
        clf_fixed = HUGIMLClassifierNative(B=5, L=1, G=0.0, topK=20, adaptive_binning=False)
        clf_fixed.fit(X_tr, y_tr)
        assert len(clf_fixed.patterns_) > 0, "fixed-numeric L1 fit returned 0 patterns"
        preds_fixed = clf_fixed.predict(X_te)
        assert preds_fixed.shape == (len(X_te),)
        print(f"[OK] fixed-numeric L1 hotpath: {len(clf_fixed.patterns_)} patterns")
    except Exception as exc:
        _fail("fixed-numeric L1 hotpath smoke test failed", exc)
    finally:
        os.environ.pop("HUGIML_ENABLE_FIXED_NUMERIC_L1_FASTPATH", None)

    print("\n=== All smoke tests PASSED ===")


if __name__ == "__main__":
    main()

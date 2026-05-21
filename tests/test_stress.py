# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Stress tests — memory footprint, large n/p grids, concurrent inference,
mining timeout, and graceful degradation on extreme inputs.

Marked ``stress``; excluded from the default test run.
Run manually::

    pytest tests/test_stress.py -v -m stress
"""

from __future__ import annotations

import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.stress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    """Return process RSS in MiB (requires psutil; returns 0 if absent)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def _make_dataset(n: int, p: int, n_cls: int = 2, seed: int = 0) -> tuple:
    """Return (X: DataFrame, y: Series) with n rows and p numeric columns."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, p)),
        columns=[f"f{j}" for j in range(p)],
    )
    y = pd.Series(rng.integers(0, n_cls, size=n), name="target")
    return X, y


def _quick_clf(**kw):
    from hugiml import HUGIMLClassifierNative

    defaults = dict(B=4, L=1, G=0.0, topK=30)
    defaults.update(kw)
    return HUGIMLClassifierNative(**defaults)


def _fit_clf(n: int, p: int, **kw):
    """Fit a classifier on a synthetic (n, p) dataset and return (clf, X_te, y_te)."""
    from sklearn.model_selection import train_test_split

    X, y = _make_dataset(n, p)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

    clf = _quick_clf(**kw)
    X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
    clf.fit(X_p, y_p)
    return clf, X_te.reset_index(drop=True), y_te.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Memory footprint
# ---------------------------------------------------------------------------


class TestMemoryFootprint:
    def test_fit_memory_reasonable(self, german_split):
        """Fitting on German Credit should not exhaust typical CI memory."""
        X_tr, X_te, y_tr, y_te = german_split
        from hugiml import HUGIMLClassifierNative

        before = _rss_mb()
        clf = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
        clf.fit(X_p, y_p)
        after = _rss_mb()
        growth = after - before
        assert growth < 500, f"Memory growth {growth:.1f} MiB exceeds 500 MiB"

    def test_no_memory_leak_repeated_predict(self, fitted_clf_synthetic):
        """Memory should not grow unboundedly across repeated predictions."""
        clf, X_te, _ = fitted_clf_synthetic
        gc.collect()
        before = _rss_mb()
        for _ in range(100):
            clf.predict_proba(X_te)
        gc.collect()
        after = _rss_mb()
        growth = after - before
        assert growth < 100, f"Memory growth {growth:.1f} MiB over 100 predict calls"

    def test_td_memory_usage_bytes_returns_positive(self, fitted_clf_synthetic):
        """td_.memory_usage_bytes() must return a sensible positive value."""
        clf, _, _ = fitted_clf_synthetic
        mem = clf.td_.memory_usage_bytes()
        assert mem > 0, "memory_usage_bytes() returned non-positive"
        # Should be far less than 100 MiB for a small synthetic dataset
        assert mem < 100 * 1024 * 1024, f"memory_usage_bytes() suspiciously large: {mem}"


# ---------------------------------------------------------------------------
# Large n × p grid stress tests
# ---------------------------------------------------------------------------


class TestLargeNP:
    """Sweep n and p to confirm no hangs and graceful behaviour."""

    @pytest.mark.parametrize(
        "n,p",
        [
            (500, 10),
            (500, 50),
            (1_000, 20),
            (2_000, 15),
            (5_000, 10),
        ],
    )
    def test_fit_predict_grid(self, n, p):
        """fit + predict_proba must complete without hanging on (n, p) grid."""
        clf, X_te, _ = _fit_clf(n, p, B=4, L=1, topK=50)
        proba = clf.predict_proba(X_te)
        assert proba.shape[0] == len(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.parametrize(
        "n,p",
        [
            (200, 100),
            (300, 150),
            (500, 80),
        ],
    )
    def test_wide_datasets(self, n, p):
        """Wide datasets (many columns) must fit within a reasonable time."""
        t0 = time.perf_counter()
        clf, X_te, _ = _fit_clf(n, p, B=3, L=1, topK=20)
        elapsed = time.perf_counter() - t0
        # Wide datasets with L=1 should finish well under 60 s
        assert elapsed < 60.0, f"n={n}, p={p} took {elapsed:.1f} s (expected < 60 s)"
        proba = clf.predict_proba(X_te)
        assert proba.shape[0] == len(X_te)

    def test_tall_dataset(self):
        """Tall dataset (large n, small p) must fit and predict correctly."""
        clf, X_te, _ = _fit_clf(10_000, 8, B=5, L=1, topK=30)
        proba = clf.predict_proba(X_te)
        assert proba.shape[0] == len(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_multiclass_stress(self):
        """Multi-class (4 classes, moderate n, p) must produce valid proba."""
        from sklearn.model_selection import train_test_split

        X, y = _make_dataset(800, 12, n_cls=4)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=7)
        clf = _quick_clf(B=4, L=1, topK=40)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
        clf.fit(X_p, y_p)
        proba = clf.predict_proba(X_te.reset_index(drop=True))
        assert proba.shape == (len(X_te), 4)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Large batch inference
# ---------------------------------------------------------------------------


class TestLargeBatchInference:
    def test_predict_10k_samples(self, fitted_clf_synthetic, synthetic_Xy):
        """predict_proba must handle a 10 000-sample batch without error."""
        clf, _, _ = fitted_clf_synthetic
        X, _ = synthetic_Xy
        X_large = pd.concat([X] * 50, ignore_index=True).iloc[:10_000]
        proba = clf.predict_proba(X_large)
        assert proba.shape[0] == 10_000
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_transform_large_batch(self, fitted_clf_synthetic, synthetic_Xy):
        clf, _, _ = fitted_clf_synthetic
        X, _ = synthetic_Xy
        X_large = pd.concat([X] * 50, ignore_index=True).iloc[:5_000]
        mat = clf.transform(X_large)
        assert mat.shape[0] == 5_000


# ---------------------------------------------------------------------------
# Mining timeout (C++ level)
# ---------------------------------------------------------------------------


class TestMiningTimeout:
    def test_timeout_returns_partial_results(self):
        """mine_patterns with timeout_s=0.01 must not hang and may return patterns."""
        try:
            import _hugiml_core as _core
        except ImportError:
            pytest.skip("native extension not built")

        X, y = _make_dataset(300, 15)
        clf = _quick_clf(B=4, L=2, topK=200)
        X_p, y_p = clf.prepareXy(X.reset_index(drop=True), y.reset_index(drop=True))
        # Only prepare transactions; call mine_patterns directly with a tiny timeout
        import numpy as np

        clf.td_ = _core.prepare_transactions(
            X_p.values.astype(np.float64),
            y_p.astype(np.int64),
            4,
            None,
            np.zeros(X_p.shape[1], dtype=np.uint8),
            np.zeros(X_p.shape[1], dtype=np.uint8),
            None,
        )
        t0 = time.perf_counter()
        patterns = _core.mine_patterns(
            clf.td_,
            y_p.astype(np.int64),
            2,
            200,
            2,
            0.0,
            timeout_s=0.05,  # 50 ms hard timeout
        )
        elapsed = time.perf_counter() - t0
        # Should finish quickly (not hang for seconds)
        assert elapsed < 5.0, f"mine_patterns with timeout ran {elapsed:.2f} s"
        # patterns may be empty or partial — both are acceptable
        assert isinstance(patterns, list)

    def test_no_hang_on_large_L2(self):
        """L=2 with moderate data must complete within max_fit_seconds."""
        from hugiml import HUGIMLClassifierNative

        X, y = _make_dataset(400, 12, seed=99)
        clf = HUGIMLClassifierNative(B=4, L=2, G=0.0, topK=100, max_fit_seconds=10)
        X_p, y_p = clf.prepareXy(X.reset_index(drop=True), y.reset_index(drop=True))
        t0 = time.perf_counter()
        clf.fit(X_p, y_p)
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"L=2 fit took {elapsed:.1f} s — expected < 30 s"
        assert len(clf.patterns_) > 0


# ---------------------------------------------------------------------------
# Graceful handling of edge-case inputs
# ---------------------------------------------------------------------------


class TestEdgeCaseInputs:
    def test_single_binary_feature(self):
        """Single binary column must not crash or produce NaN probabilities."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative
        from hugiml.exceptions import HUGIMLMiningError

        rng = np.random.default_rng(5)
        X = pd.DataFrame({"a": rng.integers(0, 2, size=200).astype(float)})
        y = pd.Series(rng.integers(0, 2, size=200))
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1)
        # G=0.0 ensures patterns are found when discriminative structure exists.
        # For a single binary column with random labels, no patterns may be found
        # at all — HUGIMLMiningError is an acceptable outcome for degenerate data.
        clf = HUGIMLClassifierNative(B=2, L=1, G=0.0, topK=10)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
        try:
            clf.fit(X_p, y_p)
        except HUGIMLMiningError:
            return  # no patterns from degenerate data is acceptable
        proba = clf.predict_proba(X_te.reset_index(drop=True))
        assert not np.isnan(proba).any(), "NaN in predict_proba output"

    def test_constant_column_ignored(self):
        """Constant columns must not cause crashes or inf utilities."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative

        rng = np.random.default_rng(7)
        X = pd.DataFrame(
            {
                "const": np.ones(300),
                "a": rng.standard_normal(300),
                "b": rng.standard_normal(300),
            }
        )
        y = pd.Series(rng.integers(0, 2, size=300))
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=2)
        clf = HUGIMLClassifierNative(B=3, L=1, topK=20)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
        clf.fit(X_p, y_p)
        proba = clf.predict_proba(X_te.reset_index(drop=True))
        assert not np.isnan(proba).any()

    def test_all_same_label_raises(self):
        """y with only one class must raise a clear error, not crash or hang."""
        from hugiml import HUGIMLClassifierNative

        rng = np.random.default_rng(3)
        X = pd.DataFrame({"a": rng.standard_normal(100), "b": rng.standard_normal(100)})
        y = pd.Series(np.zeros(100, dtype=int))
        clf = HUGIMLClassifierNative(B=3, L=1, topK=10)
        X_p, y_p = clf.prepareXy(X, y)
        with pytest.raises(Exception):
            clf.fit(X_p, y_p)

    def test_nan_in_numeric_features(self):
        """NaN values in numeric features must not cause crashes."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative

        rng = np.random.default_rng(11)
        X = pd.DataFrame(rng.standard_normal((300, 5)), columns=[f"f{i}" for i in range(5)])
        # Inject 5% NaN
        mask = rng.random(X.shape) < 0.05
        X[mask] = np.nan
        y = pd.Series(rng.integers(0, 2, size=300))
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=3)
        clf = HUGIMLClassifierNative(B=4, L=1, topK=20)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
        clf.fit(X_p, y_p)
        proba = clf.predict_proba(X_te.reset_index(drop=True))
        assert not np.isnan(proba).any()

    def test_predict_on_single_row(self, fitted_clf_synthetic, synthetic_Xy):
        """Single-row prediction must not crash or produce NaN."""
        clf, _, _ = fitted_clf_synthetic
        X, _ = synthetic_Xy
        proba = clf.predict_proba(X.iloc[:1])
        assert proba.shape[0] == 1
        assert not np.isnan(proba).any()


# ---------------------------------------------------------------------------
# Concurrent inference
# ---------------------------------------------------------------------------


class TestConcurrentInference:
    def test_concurrent_predict_proba(self, fitted_clf_synthetic):
        """Concurrent predict_proba calls must return identical results."""
        clf, X_te, _ = fitted_clf_synthetic
        expected = clf.predict_proba(X_te)
        errors: list[Exception] = []
        results: list[np.ndarray] = [None] * 8  # type: ignore[list-item]

        def worker(idx: int):
            try:
                results[idx] = clf.predict_proba(X_te)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in concurrent predict: {errors}"
        for i, res in enumerate(results):
            if res is not None:
                np.testing.assert_allclose(
                    res, expected, atol=1e-10, err_msg=f"Thread {i} produced different predictions"
                )

    def test_concurrent_transform(self, fitted_clf_synthetic):
        clf, X_te, _ = fitted_clf_synthetic
        errors: list[Exception] = []

        def worker():
            try:
                clf.transform(X_te)
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker) for _ in range(8)]
            for f in as_completed(futures):
                f.result(timeout=60)

        assert len(errors) == 0, f"Errors in concurrent transform: {errors}"

    def test_concurrent_fit_inference_isolation(self, synthetic_Xy):
        """New fits must not corrupt existing fitted classifiers."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative

        X, y = synthetic_Xy
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
        clf_ref = HUGIMLClassifierNative(B=5, L=1, G=1e-3)
        X_p, y_p = clf_ref.prepareXy(X_tr, y_tr)
        clf_ref.fit(X_p, y_p)
        expected = clf_ref.predict(X_te)

        errors: list[Exception] = []

        def fit_new():
            try:
                clf_new = HUGIMLClassifierNative(B=4, L=1)
                Xn, yn = clf_new.prepareXy(X_tr, y_tr)
                clf_new.fit(Xn, yn)
            except Exception as exc:
                errors.append(exc)

        def predict_ref():
            try:
                preds = clf_ref.predict(X_te)
                np.testing.assert_array_equal(preds, expected)
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(fit_new) for _ in range(3)] + [
                pool.submit(predict_ref) for _ in range(3)
            ]
            for f in as_completed(futures):
                f.result(timeout=120)

        assert len(errors) == 0, f"Isolation errors: {errors}"


# ---------------------------------------------------------------------------
# Monitor stress
# ---------------------------------------------------------------------------


class TestMonitorStress:
    def test_monitor_high_throughput(self, fitted_clf_synthetic):
        """Monitor must remain stable under high-throughput record writes."""
        clf, X_te, _ = fitted_clf_synthetic
        clf.enable_monitoring(window_size=1000)
        errors: list[Exception] = []

        def writer():
            try:
                for _ in range(200):
                    clf.predict_proba(X_te[:1])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        report = clf.monitor.report()
        assert report is not None


# ---------------------------------------------------------------------------
# Bitmap / transform correctness at scale
# ---------------------------------------------------------------------------


class TestBitmapAtScale:
    def test_build_train_matrix_large(self):
        """build_train_matrix must produce a valid sparse COO for n=2000."""
        try:
            import _hugiml_core as _core
        except ImportError:
            pytest.skip("native extension not built")
        from scipy.sparse import csr_matrix

        clf, X_te, _ = _fit_clf(2_000, 10)
        rows, cols = _core.build_train_matrix(clf.td_, clf.patterns_)
        n = clf.fit_metadata_.n_samples
        n_pats = len(clf.patterns_)
        assert max(rows) < n
        assert max(cols) < n_pats
        mat = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n_pats))
        assert mat.shape == (n, n_pats)

    def test_pattern_matrix_values_binary(self, fitted_clf_synthetic):
        """All entries of the pattern feature matrix must be 0 or 1."""
        clf, X_te, _ = fitted_clf_synthetic
        mat = clf.transform(X_te)
        vals = mat.data
        assert set(vals.round(6)).issubset({0.0, 1.0}), "Non-binary values in pattern matrix"


# =============================================================================
# L=2 (depth-2 compound patterns) — memory and time stress tests
# =============================================================================


class TestL2Stress:
    """Stress tests specifically for L=2 (depth-2 compound patterns).

    L=2 triggers the combinatorial branch of the mining algorithm.  These
    tests guard against memory blowup and wall-clock regressions at depth 2.
    """

    @pytest.mark.parametrize(
        "n,p",
        [
            (300, 10),
            (500, 12),
            (800, 8),
        ],
    )
    def test_l2_fit_predict_completes(self, n, p):
        """L=2 fit + predict_proba must complete within a generous wall-clock budget."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative

        t0 = time.perf_counter()
        X, y = _make_dataset(n, p, seed=1234)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        clf = HUGIMLClassifierNative(B=4, L=2, G=0.0, topK=50, max_fit_seconds=30)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
        clf.fit(X_p, y_p)
        elapsed_fit = time.perf_counter() - t0
        assert elapsed_fit < 60.0, f"L=2 fit (n={n}, p={p}) took {elapsed_fit:.1f}s (> 60s)"

        proba = clf.predict_proba(X_te.reset_index(drop=True))
        assert proba.shape[0] == len(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert len(clf.patterns_) > 0

    def test_l2_memory_footprint(self):
        """L=2 fit memory growth must stay within 1 GiB for moderate n, p."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative

        X, y = _make_dataset(400, 10, seed=77)
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=0)
        clf = HUGIMLClassifierNative(B=4, L=2, G=0.0, topK=60, max_fit_seconds=20)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))

        gc.collect()
        before = _rss_mb()
        clf.fit(X_p, y_p)
        gc.collect()
        after = _rss_mb()

        growth = after - before
        assert growth < 1024, (
            f"L=2 fit memory growth {growth:.1f} MiB exceeds 1 GiB.  "
            "Check for combinatorial memory blowup in mine_patterns."
        )

    def test_l2_produces_compound_patterns(self):
        """At least some L=2 patterns must have len(items) == 2."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative

        X, y = _make_dataset(500, 8, seed=42)
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=0)
        clf = HUGIMLClassifierNative(B=4, L=2, G=0.0, topK=80)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
        clf.fit(X_p, y_p)

        compound = [pe for pe in clf.patterns_ if len(pe.items) == 2]
        assert len(compound) > 0, (
            "No compound (length-2) patterns found with L=2.  "
            "The mining engine may be ignoring the L parameter."
        )

    def test_l2_predict_valid_proba(self):
        """Predictions from an L=2 model must be valid probabilities."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative

        X, y = _make_dataset(400, 8, seed=55)
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.25, random_state=0)
        clf = HUGIMLClassifierNative(B=4, L=2, G=0.0, topK=40, max_fit_seconds=20)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
        clf.fit(X_p, y_p)
        proba = clf.predict_proba(X_te.reset_index(drop=True))

        assert not np.isnan(proba).any(), "NaN in L=2 predict_proba output"
        assert not np.isinf(proba).any(), "Inf in L=2 predict_proba output"
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert (proba >= 0).all()

    def test_l2_serialization_roundtrip(self, tmp_path):
        """L=2 models must survive a save/load round-trip with identical predictions."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative
        from hugiml.serialization import load_model, save_model

        X, y = _make_dataset(300, 8, seed=99)
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=0)
        clf = HUGIMLClassifierNative(B=4, L=2, G=0.0, topK=30, max_fit_seconds=15)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
        clf.fit(X_p, y_p)
        preds_before = clf.predict(X_te.reset_index(drop=True))

        out = tmp_path / "l2_model.hugiml"
        save_model(clf, out)
        clf2 = load_model(out)
        preds_after = clf2.predict(X_te.reset_index(drop=True))
        np.testing.assert_array_equal(preds_before, preds_after)


# =============================================================================
# High-cardinality categorical data stress tests
# =============================================================================


class TestHighCardinalityCategorical:
    """Stress tests for datasets with many categorical levels.

    High cardinality (100+ categories) is a known source of memory pressure
    in pattern mining because the bin count grows with the number of distinct
    values.
    """

    @staticmethod
    def _make_highcard_dataset(
        n: int = 500,
        n_cat_cols: int = 3,
        cardinality: int = 100,
        n_num_cols: int = 5,
        seed: int = 0,
    ) -> tuple:
        """Return (X: DataFrame, y: Series) with high-cardinality categorical columns."""
        rng = np.random.default_rng(seed)
        cols: dict = {}
        for j in range(n_num_cols):
            cols[f"num_{j}"] = rng.standard_normal(n)
        for j in range(n_cat_cols):
            categories = [f"cat_{j}_val_{k}" for k in range(cardinality)]
            cols[f"cat_{j}"] = [categories[i] for i in rng.integers(0, cardinality, n)]
        X = pd.DataFrame(cols)
        # Target correlated with first numeric feature
        logits = cols.get("num_0", rng.standard_normal(n))
        prob = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
        y = pd.Series(rng.binomial(1, prob).astype(int), name="target")
        return X, y

    def test_fit_highcard_100(self):
        """100-category columns must fit without crash or memory explosion."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative

        X, y = self._make_highcard_dataset(n=400, n_cat_cols=2, cardinality=100)
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=0)
        clf = HUGIMLClassifierNative(B=4, L=1, G=0.0, topK=50)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
        clf.fit(X_p, y_p)
        proba = clf.predict_proba(X_te.reset_index(drop=True))
        assert proba.shape[0] == len(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_fit_highcard_500(self):
        """500-category columns (extreme) must complete without hang."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative
        from hugiml.exceptions import HUGIMLMiningError

        X, y = self._make_highcard_dataset(n=300, n_cat_cols=1, cardinality=500)
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=0)
        clf = HUGIMLClassifierNative(B=4, L=1, G=0.0, topK=30, max_fit_seconds=30)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
        t0 = time.perf_counter()
        try:
            clf.fit(X_p, y_p)
        except HUGIMLMiningError:
            pass  # acceptable if no patterns found for degenerate high-card data
        elapsed = time.perf_counter() - t0
        assert elapsed < 60.0, f"High-cardinality (500) fit took {elapsed:.1f}s (> 60s)"

    def test_highcard_predict_no_nan(self):
        """Predictions on high-cardinality data must not contain NaN."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative

        X, y = self._make_highcard_dataset(n=400, n_cat_cols=2, cardinality=50)
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=7)
        clf = HUGIMLClassifierNative(B=4, L=1, G=0.0, topK=40)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
        clf.fit(X_p, y_p)
        proba = clf.predict_proba(X_te.reset_index(drop=True))
        assert not np.isnan(proba).any(), "NaN in predict_proba with high-cardinality cats"

    def test_highcard_unseen_category_at_predict(self):
        """Unseen categories at predict time must not crash (soft-miss fallback)."""

        from hugiml import HUGIMLClassifierNative

        rng = np.random.default_rng(13)
        n = 300
        train_cats = [f"val_{k}" for k in range(50)]
        test_cats = train_cats + [f"unseen_{k}" for k in range(20)]

        X_tr = pd.DataFrame(
            {
                "num": rng.standard_normal(n),
                "cat": [train_cats[i] for i in rng.integers(0, len(train_cats), n)],
            }
        )
        y_tr = pd.Series(rng.integers(0, 2, n))

        clf = HUGIMLClassifierNative(B=4, L=1, G=0.0, topK=30)
        X_p, y_p = clf.prepareXy(X_tr, y_tr)
        clf.fit(X_p, y_p)

        X_te = pd.DataFrame(
            {
                "num": rng.standard_normal(40),
                "cat": [test_cats[i] for i in rng.integers(0, len(test_cats), 40)],
            }
        )
        proba = clf.predict_proba(X_te)
        assert not np.isnan(proba).any(), "NaN with unseen categories at predict time"
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_l2_highcard_combined(self):
        """L=2 + high-cardinality categoricals must finish within budget."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative
        from hugiml.exceptions import HUGIMLMiningError

        X, y = self._make_highcard_dataset(n=300, n_cat_cols=2, cardinality=30, n_num_cols=4)
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=0)
        clf = HUGIMLClassifierNative(B=4, L=2, G=0.0, topK=40, max_fit_seconds=30)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))
        t0 = time.perf_counter()
        try:
            clf.fit(X_p, y_p)
            proba = clf.predict_proba(X_te.reset_index(drop=True))
            assert not np.isnan(proba).any()
        except HUGIMLMiningError:
            pass  # acceptable if data is too sparse for patterns
        elapsed = time.perf_counter() - t0
        assert elapsed < 60.0, f"L=2 + high-card fit took {elapsed:.1f}s (> 60s)"

    def test_highcard_memory_growth(self):
        """Fitting on 200-category data must not exhaust typical CI memory."""
        from sklearn.model_selection import train_test_split

        from hugiml import HUGIMLClassifierNative

        X, y = self._make_highcard_dataset(n=400, n_cat_cols=2, cardinality=200)
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=0)
        clf = HUGIMLClassifierNative(B=4, L=1, G=0.0, topK=40)
        X_p, y_p = clf.prepareXy(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True))

        gc.collect()
        before = _rss_mb()
        clf.fit(X_p, y_p)
        gc.collect()
        after = _rss_mb()

        growth = after - before
        assert growth < 512, (
            f"High-cardinality (200 cats) fit memory growth {growth:.1f} MiB exceeds 512 MiB."
        )

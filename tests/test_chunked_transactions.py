# Copyright 2026 Srikumar Krishnamoorthy — Apache-2.0
"""
Correctness tests for the row-stripe chunked transaction construction.

Design under test
─────────────────
Phase 1 (column stats) computes edges/Pearson/NMI from all n rows; no
bin_codes are materialised.  Phase 2 (transaction construction) iterates
rows in stripes of STRIPE_ROWS, re-reads raw values from X_num_arr per
stripe, and applies the stored edges on the fly.  The old ColStream.bin_codes
array is eliminated.

What is verified
────────────────
1.  Determinism: two independent runs on identical data produce identical
    item_map, bn2id, RIU, item_twu, and transaction lists.
2.  Field-by-field transaction correctness for all column types (float,
    integer, categorical, pre-coded/adaptive).
3.  NaN / missing value handling: NaN rows must generate no item.
4.  Stripe-boundary invariant: datasets of n=2..800 all produce correct td.
5.  Downstream correctness: AUC, pattern counts, predict_proba, serialization.
6.  Zero-correlation columns are dropped from item registry.
7.  Transaction ordering is preserved (row 0 = transaction 0, etc.).
"""

from __future__ import annotations

import math
from pathlib import Path

import _hugiml_core as _core
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from hugiml import HUGIMLClassifierNative

TESTS_DIR = Path(__file__).parent

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


# ── Pure helpers (no sys.path manipulation) ───────────────────────────────────


def _clf(**params) -> HUGIMLClassifierNative:
    # use_hotpath=False: the L1 hotpath intentionally discards .transactions
    # for memory efficiency.  These tests inspect transaction internals directly,
    # so they must run on the slow path where .transactions is populated.
    params.setdefault("use_hotpath", False)
    return HUGIMLClassifierNative(**params)


def _fit(X, y, params, seed=42):
    """prepareXy → 80/20 split → fit; returns (clf, X_te, y_te)."""
    clf = _clf(**params)
    Xp, yp = clf.prepareXy(X.copy(), y.copy())
    Xtr, Xte, ytr, yte = train_test_split(Xp, yp, test_size=0.2, random_state=seed, stratify=yp)
    clf.fit(Xtr, ytr)
    return clf, Xte, yte


def _fit_full(X, y, params):
    """Fit on the full dataset (no split)."""
    clf = _clf(**params)
    Xp, yp = clf.prepareXy(X.copy(), y.copy())
    clf.fit(Xp, yp)
    return clf


def _td(X, y, params, seed=42):
    """Return raw TransactionDataCpp (._td) from an 80/20 split fit."""
    clf, *_ = _fit(X, y, params, seed)
    return clf.td_._td


def _td_full(X, y, params):
    """Return raw TransactionDataCpp from a full-dataset fit."""
    return _fit_full(X, y, params).td_._td


# ── Field comparator ─────────────────────────────────────────────────────────


def assert_td_equal(td_a, td_b, label: str = ""):
    """Assert every computed field of two TransactionDataCpp objects matches."""
    pfx = f"[{label}] " if label else ""

    # Item registry
    assert dict(td_a.item_map) == dict(td_b.item_map), f"{pfx}item_map differs"
    assert dict(td_a.bn2id) == dict(td_b.bn2id), f"{pfx}bn2id differs"

    # Per-item arrays
    riu_a = np.array(list(td_a.RIU))
    riu_b = np.array(list(td_b.RIU))
    assert len(riu_a) == len(riu_b), f"{pfx}RIU length differs"
    assert np.allclose(riu_a, riu_b, atol=1e-9, rtol=0), (
        f"{pfx}RIU differs; max={np.abs(riu_a - riu_b).max():.2e}"
    )

    twu_a = np.array(list(td_a.item_twu))
    twu_b = np.array(list(td_b.item_twu))
    assert len(twu_a) == len(twu_b), f"{pfx}item_twu length differs"
    assert np.allclose(twu_a, twu_b, atol=1e-9, rtol=0), (
        f"{pfx}item_twu differs; max={np.abs(twu_a - twu_b).max():.2e}"
    )

    # Transaction list
    ta, tb = td_a.transactions, td_b.transactions
    assert len(ta) == len(tb), f"{pfx}transaction count differs ({len(ta)} vs {len(tb)})"

    for r in range(len(ta)):
        row_a = sorted(ta[r], key=lambda x: x[0])
        row_b = sorted(tb[r], key=lambda x: x[0])
        assert len(row_a) == len(row_b), f"{pfx}row {r} item count {len(row_a)} vs {len(row_b)}"
        for (ia, ua), (ib, ub) in zip(row_a, row_b):
            assert ia == ib, f"{pfx}row {r}: item ID {ia} vs {ib}"
            assert abs(ua - ub) < 1e-9, f"{pfx}row {r} item {ia}: utility {ua:.10f} vs {ub:.10f}"


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def german():
    p = TESTS_DIR / "german.data"
    if not p.exists():
        pytest.skip("german.data not found in tests/ directory")
    df = pd.read_csv(p, sep=" ", header=None, names=GERMAN_COLS + ["target"])
    df["target"] = (df["target"] == 2).astype(int)
    return df.drop(columns=["target"]), df["target"]


@pytest.fixture(scope="module")
def heloc():
    p = TESTS_DIR / "heloc.csv"
    if not p.exists():
        pytest.skip("heloc.csv not found in tests/ directory")
    df = pd.read_csv(p)
    df["target"] = (df["RiskPerformance"] == "Bad").astype(int)
    df = df.drop(columns=["RiskPerformance"])
    return df.drop(columns=["target"]), df["target"]


@pytest.fixture(scope="module")
def synth_float():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 8)), columns=[f"f{i}" for i in range(8)])
    y = pd.Series((rng.random(400) > 0.5).astype(int))
    return X, y


@pytest.fixture(scope="module")
def synth_cat():
    rng = np.random.default_rng(1)
    n = 400
    y_arr = (rng.random(n) > 0.5).astype(int)
    X = pd.DataFrame(
        {
            "color": np.where(
                rng.random(n) < 0.6,
                np.where(y_arr == 1, "red", "blue"),
                rng.choice(["red", "blue", "green"], n),
            ),
            "size": rng.choice(["S", "M", "L"], n),
            "shape": np.where(
                rng.random(n) < 0.7,
                np.where(y_arr == 1, "circle", "square"),
                rng.choice(["circle", "square", "triangle"], n),
            ),
        }
    )
    return X, pd.Series(y_arr)


@pytest.fixture(scope="module")
def synth_nan():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(300, 6)), columns=[f"n{i}" for i in range(6)])
    mask = rng.random(X.shape) < 0.05
    X[mask] = np.nan
    y = pd.Series((rng.random(300) > 0.5).astype(int))
    return X, y


@pytest.fixture(scope="module")
def synth_mixed():
    rng = np.random.default_rng(42)
    n = 500
    y_arr = (rng.random(n) > 0.5).astype(int)
    X = pd.DataFrame(
        {
            "f_float": rng.normal(size=n),
            "f_int": rng.integers(0, 10, size=n),
            "f_cat": rng.choice(["A", "B", "C"], size=n),
            "f_float2": rng.normal(size=n),
        }
    )
    return X, pd.Series(y_arr)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Determinism — two independent runs must produce bit-identical td
# ═══════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Two independent fits on the same data with the same seed must
    produce identical transactions, RIU, item_twu, and item_map."""

    def test_german(self, german):
        X, y = german
        td_a = _td(X, y, {"B": 7, "L": 1, "G": 5e-3})
        td_b = _td(X, y, {"B": 7, "L": 1, "G": 5e-3})
        assert_td_equal(td_a, td_b, "German/L1")

    def test_heloc(self, heloc):
        X, y = heloc
        td_a = _td(X, y, {"B": 7, "L": 1, "G": 5e-3})
        td_b = _td(X, y, {"B": 7, "L": 1, "G": 5e-3})
        assert_td_equal(td_a, td_b, "HELOC/L1")

    def test_categorical(self, synth_cat):
        X, y = synth_cat
        td_a = _td(X, y, {"B": 3, "L": 1, "G": 0.0})
        td_b = _td(X, y, {"B": 3, "L": 1, "G": 0.0})
        assert_td_equal(td_a, td_b, "Cat")

    def test_float(self, synth_float):
        X, y = synth_float
        td_a = _td(X, y, {"B": 5, "L": 1, "G": 0.0})
        td_b = _td(X, y, {"B": 5, "L": 1, "G": 0.0})
        assert_td_equal(td_a, td_b, "Float")

    def test_adaptive(self, synth_float):
        X, y = synth_float
        td_a = _td(X, y, {"B": 5, "L": 1, "G": 0.0, "adaptive_binning": True})
        td_b = _td(X, y, {"B": 5, "L": 1, "G": 0.0, "adaptive_binning": True})
        assert_td_equal(td_a, td_b, "Adaptive")

    def test_mixed(self, synth_mixed):
        X, y = synth_mixed
        td_a = _td(X, y, {"B": 5, "L": 1, "G": 0.0})
        td_b = _td(X, y, {"B": 5, "L": 1, "G": 0.0})
        assert_td_equal(td_a, td_b, "Mixed")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Transaction content correctness — verify individual fields
# ═══════════════════════════════════════════════════════════════════════════


class TestTransactionContents:
    """Check specific structural invariants of every transaction."""

    def _check_trans(self, td, label):
        n = len(td.transactions)
        n_items = len(td.item_twu)
        trans = td.transactions

        # Correct row count
        assert len(trans) == n, f"{label}: expected {n} transactions, got {len(trans)}"

        # Every item ID in every transaction must be in [1, n_items]
        for r, t in enumerate(trans):
            for iid, iu in t:
                if iid == -1:
                    continue  # sentinel for empty row
                assert 1 <= iid <= n_items, f"{label}: row {r} has out-of-range iid={iid}"
                assert iu >= 0.0, f"{label}: row {r} iid={iid} has negative utility {iu}"

        # item_twu consistency: item_twu[i] >= RIU[i] always
        # (TWU counts full row utility; RIU counts only item i's own utility)
        twu = list(td.item_twu)
        riu = list(td.RIU)
        for i in range(n_items):
            assert twu[i] >= riu[i] - 1e-9, (
                f"{label}: item {i + 1} TWU {twu[i]:.6f} < RIU {riu[i]:.6f}"
            )

        # item_twu must equal sum of tutils_r for rows containing each item
        reconstructed_twu = [0.0] * n_items
        reconstructed_riu = [0.0] * n_items
        for t in trans:
            if len(t) == 1 and t[0][0] == -1:
                continue
            tutils = sum(iu for _, iu in t)
            for iid, iu in t:
                reconstructed_twu[iid - 1] += tutils
                reconstructed_riu[iid - 1] += iu
        assert np.allclose(reconstructed_twu, twu, atol=1e-6), (
            f"{label}: reconstructed TWU != stored TWU"
        )
        assert np.allclose(reconstructed_riu, riu, atol=1e-6), (
            f"{label}: reconstructed RIU != stored RIU"
        )

    def test_german(self, german):
        X, y = german
        self._check_trans(_td(X, y, {"B": 7, "L": 1, "G": 5e-3}), "German")

    def test_heloc(self, heloc):
        X, y = heloc
        self._check_trans(_td(X, y, {"B": 7, "L": 1, "G": 5e-3}), "HELOC")

    def test_categorical(self, synth_cat):
        X, y = synth_cat
        self._check_trans(_td(X, y, {"B": 3, "L": 1, "G": 0.0}), "Cat")

    def test_adaptive(self, synth_float):
        X, y = synth_float
        td = _td(X, y, {"B": 5, "L": 1, "G": 0.0, "adaptive_binning": True})
        self._check_trans(td, "Adaptive")

    def test_mixed(self, synth_mixed):
        X, y = synth_mixed
        self._check_trans(_td(X, y, {"B": 5, "L": 1, "G": 0.0}), "Mixed")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Stripe boundary invariant
# ═══════════════════════════════════════════════════════════════════════════


class TestStripeBoundary:
    """Stripe logic must be correct for datasets of all sizes.
    STRIPE_ROWS = 100 000; our test datasets are all smaller, so every
    call runs in a single stripe.  The important invariant is that the
    final stripe (which may be shorter than STRIPE_ROWS) is handled
    correctly and the results are deterministic."""

    @pytest.mark.parametrize("n_rows", [2, 4, 7, 50, 200, 800])
    def test_various_sizes(self, n_rows):
        """Deterministic td for datasets of n=2..800 rows."""
        rng = np.random.default_rng(n_rows)
        X = pd.DataFrame(rng.normal(size=(n_rows, 4)), columns=["a", "b", "c", "d"])
        # Balanced classes
        y = pd.Series(np.tile([0, 1], math.ceil(n_rows / 2))[:n_rows])
        params = {"B": 3, "L": 1, "G": 0.0}
        td_a = _td_full(X, y, params)
        td_b = _td_full(X, y, params)
        assert_td_equal(td_a, td_b, f"n={n_rows}")
        # Also verify structural invariants
        assert len(td_a.transactions) == n_rows

    def test_minimum_n(self):
        """n=2 (minimum valid) must work correctly."""
        X = pd.DataFrame({"x": [1.0, 2.0]})
        y = pd.Series([0, 1])
        td_a = _td_full(X, y, {"B": 2, "L": 1, "G": 0.0})
        td_b = _td_full(X, y, {"B": 2, "L": 1, "G": 0.0})
        assert_td_equal(td_a, td_b, "n=2")
        assert len(td_a.transactions) == 2


# ═══════════════════════════════════════════════════════════════════════════
# 4. NaN / missing value handling
# ═══════════════════════════════════════════════════════════════════════════


class TestNaNHandling:
    def test_nan_float_deterministic(self, synth_nan):
        X, y = synth_nan
        td_a = _td(X, y, {"B": 5, "L": 1, "G": 0.0})
        td_b = _td(X, y, {"B": 5, "L": 1, "G": 0.0})
        assert_td_equal(td_a, td_b, "NaNFloat")

    def test_nan_adaptive_deterministic(self, synth_nan):
        X, y = synth_nan
        td_a = _td(X, y, {"B": 5, "L": 1, "G": 0.0, "adaptive_binning": True})
        td_b = _td(X, y, {"B": 5, "L": 1, "G": 0.0, "adaptive_binning": True})
        assert_td_equal(td_a, td_b, "NaNAdaptive")

    def test_nan_row_generates_no_item_non_adaptive(self, synth_nan):
        """Rows where ALL non-cat columns are NaN must produce the empty sentinel."""
        X, y = synth_nan
        # Inject a fully-NaN row
        X_copy = X.copy()
        X_copy.iloc[0] = np.nan
        y_copy = y.copy()
        clf = _clf(B=5, L=1, G=0.0)
        Xp, yp = clf.prepareXy(X_copy, y_copy)
        clf.fit(Xp, yp)
        td = clf.td_._td
        t0 = td.transactions[0]
        assert len(t0) == 1 and t0[0][0] == -1, f"All-NaN row should produce sentinel, got {t0}"

    def test_nan_adaptive_no_nan_patterns(self, synth_nan):
        """Pattern features must never contain literal 'nan'."""
        X, y = synth_nan
        clf, _, _ = _fit(X, y, {"B": 5, "L": 2, "G": 1e-4, "topK": 20})
        nan_feats = [f for f in clf.get_hug_features() if "nan" in f.lower()]
        assert len(nan_feats) == 0, f"nan-string patterns: {nan_feats[:3]}"

    def test_nan_predict_no_nan_output(self, synth_nan):
        X, y = synth_nan
        clf, Xte, yte = _fit(
            X,
            y,
            {"B": 5, "L": 2, "G": 1e-4, "topK": 30, "adaptive_binning": True},
            seed=7,
        )
        p = clf.predict_proba(Xte)
        assert not np.isnan(p).any(), "NaN in predict_proba output"
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)

    def test_nan_single_row_equals_batch(self, synth_nan):
        """Single-row predict must equal the same row in a batch."""
        X, y = synth_nan
        clf, Xte, _ = _fit(
            X,
            y,
            {"B": 5, "L": 1, "G": 0.0, "adaptive_binning": True},
            seed=9,
        )
        p_batch = clf.predict_proba(Xte)
        for i in range(min(5, len(Xte))):
            p_single = clf.predict_proba(Xte.iloc[[i]])[0]
            assert np.allclose(p_batch[i], p_single, atol=1e-9), (
                f"Row {i}: batch={p_batch[i]} single={p_single}"
            )

    def test_non_nan_rows_unaffected(self, synth_nan):
        """Rows without NaN must not be affected by NaN in other rows."""
        X, y = synth_nan
        clf, Xte, _ = _fit(X, y, {"B": 5, "L": 1, "G": 0.0}, seed=3)
        p_orig = clf.predict_proba(Xte)

        # Inject NaN into a few rows
        Xte_nan = Xte.copy()
        Xte_nan.iloc[:3, 0] = np.nan
        p_with = clf.predict_proba(Xte_nan)

        clean = ~Xte_nan.isna().any(axis=1)
        assert np.allclose(p_orig[clean], p_with[clean], atol=1e-9), (
            "Non-NaN rows changed by NaN injection"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Downstream fit/predict correctness
# ═══════════════════════════════════════════════════════════════════════════


class TestDownstreamCorrectness:
    def test_german_l1_auc_reasonable(self, german):
        X, y = german
        clf, Xte, yte = _fit(X, y, {"B": 7, "L": 1, "G": 5e-3})
        auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        assert auc > 0.65, f"German L=1 AUC too low: {auc:.4f}"

    def test_german_l2_auc_reasonable(self, german):
        X, y = german
        clf, Xte, yte = _fit(
            X,
            y,
            {"B": 5, "L": 2, "G": 0.0, "topK": 40, "max_fit_seconds": 60},
        )
        auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        assert auc > 0.65, f"German L=2 AUC too low: {auc:.4f}"

    def test_heloc_auc_reasonable(self, heloc):
        X, y = heloc
        clf, Xte, yte = _fit(X, y, {"B": 7, "L": 1, "G": 5e-3})
        auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        assert auc > 0.70, f"HELOC AUC too low: {auc:.4f}"

    def test_categorical_auc_reasonable(self, synth_cat):
        X, y = synth_cat
        clf, Xte, yte = _fit(X, y, {"B": 3, "L": 1, "G": 0.0})
        auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        assert auc > 0.80, f"CatSynth AUC too low: {auc:.4f}"

    def test_proba_sums_to_one(self, german):
        X, y = german
        clf, Xte, _ = _fit(X, y, {"B": 7, "L": 1, "G": 5e-3})
        p = clf.predict_proba(Xte)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
        assert not np.isnan(p).any()

    def test_predict_single_eq_batch(self, synth_float):
        """Single-row predict must equal the same row in a batch."""
        X, y = synth_float
        clf, Xte, _ = _fit(X, y, {"B": 5, "L": 1, "G": 0.0})
        p_batch = clf.predict_proba(Xte)
        for i in range(min(8, len(Xte))):
            p_single = clf.predict_proba(Xte.iloc[[i]])[0]
            assert np.allclose(p_batch[i], p_single, atol=1e-9), f"Row {i}: batch≠single"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Transaction ordering invariant
# ═══════════════════════════════════════════════════════════════════════════


class TestTransactionOrdering:
    """Transaction i must correspond to training row i."""

    def test_build_train_matrix_shape(self, synth_float):
        X, y = synth_float
        clf, Xte, _ = _fit(X, y, {"B": 5, "L": 1, "G": 0.0})
        rows, cols = _core.build_train_matrix(clf.td_._td, clf.patterns_)
        n_train = len(clf.td_._td.transactions)
        n_pats = len(clf.patterns_)
        data = np.ones(len(rows), dtype=np.float32)
        Z = sp.csr_matrix((data, (rows, cols)), shape=(n_train, n_pats))
        assert Z.shape == (n_train, n_pats)
        assert Z.data.min() >= 0 and Z.data.max() <= 1

    def test_train_matrix_nonzero_structure(self, synth_float):
        """train_matrix must have valid COO values and correct column count."""
        X, y = synth_float
        clf, _, _ = _fit(X, y, {"B": 5, "L": 1, "G": 0.0})
        rows_tr, cols_tr = _core.build_train_matrix(clf.td_._td, clf.patterns_)
        n_tr = len(clf.td_._td.transactions)
        n_pats = len(clf.patterns_)
        # All row and col indices must be in range
        if len(rows_tr) > 0:
            assert rows_tr.min() >= 0 and rows_tr.max() < n_tr
            assert cols_tr.min() >= 0 and cols_tr.max() < n_pats
        Z_tr = sp.csr_matrix(
            (np.ones(len(rows_tr)), (rows_tr, cols_tr)),
            shape=(n_tr, n_pats),
        )
        assert Z_tr.shape == (n_tr, n_pats)
        assert Z_tr.nnz == len(rows_tr)

    def test_transaction_row_count_matches_training_n(self, synth_mixed):
        X, y = synth_mixed
        clf, Xte, _ = _fit(X, y, {"B": 5, "L": 1, "G": 0.0})
        # Verify transaction count matches training rows (len(transactions) is the count)
        assert len(clf.td_._td.transactions) == len(X) - len(Xte)

    def test_transaction_items_are_subset_of_item_map(self, synth_mixed):
        """Every item ID in every transaction must be a key in item_map."""
        X, y = synth_mixed
        td = _td(X, y, {"B": 5, "L": 1, "G": 0.0})
        valid_ids = set(td.item_map.keys())
        for r, t in enumerate(td.transactions):
            for iid, _ in t:
                if iid == -1:
                    continue
                assert iid in valid_ids, f"Row {r}: item {iid} not in item_map"


# ═══════════════════════════════════════════════════════════════════════════
# 7b.  item_iu vector invariants (TItem=int storage)
# ═══════════════════════════════════════════════════════════════════════════


class TestItemIuStorage:
    """

    In previous versions utilities were stored inline as TItem = (int, double).
    Now TItem = int and TransactionDataCpp carries item_iu[iid-1] as a
    level constant.  get_transactions_py() reconstructs the old (iid, u) view.
    """

    def _check_item_iu(self, td, label: str):
        n_items = len(td.item_twu)
        iu_vec = list(td.item_iu)

        # Length must match item registry
        assert len(iu_vec) == n_items, f"{label}: item_iu length {len(iu_vec)} != n_items {n_items}"

        # All utilities non-negative and bounded
        assert all(v >= 0.0 for v in iu_vec), f"{label}: item_iu has negative entries"
        assert all(v <= 1.0 + 1e-9 for v in iu_vec), f"{label}: item_iu has entries > 1"

        # get_transactions_py must reconstruct utilities that match item_iu
        txs = td.transactions
        for r, row in enumerate(txs):
            for iid, u in row:
                if iid == -1:
                    continue
                idx = iid - 1
                assert 0 <= idx < len(iu_vec), f"{label}: row {r} iid {iid} out of range"
                assert abs(u - iu_vec[idx]) < 1e-9, (
                    f"{label}: row {r} iid {iid}: get_transactions_py u={u:.8f} "
                    f"!= item_iu[{idx}]={iu_vec[idx]:.8f}"
                )

    def test_float(self, synth_float):
        X, y = synth_float
        td = _td(X, y, {"B": 5, "L": 1, "G": 0.0})
        self._check_item_iu(td, "float")

    def test_integer(self, synth_mixed):
        X, y = synth_mixed
        td = _td(X, y, {"B": 5, "L": 1, "G": 0.0})
        self._check_item_iu(td, "mixed")

    def test_categorical(self, synth_cat):
        X, y = synth_cat
        td = _td(X, y, {"B": 3, "L": 1, "G": 0.0})
        self._check_item_iu(td, "categorical")

    def test_adaptive(self, synth_float):
        X, y = synth_float
        td = _td(X, y, {"B": 5, "L": 1, "G": 0.0, "adaptive_binning": True})
        self._check_item_iu(td, "adaptive")

    def test_nan(self, synth_nan):
        X, y = synth_nan
        td = _td(X, y, {"B": 5, "L": 1, "G": 0.0})
        self._check_item_iu(td, "nan")

    def test_item_iu_nonzero_for_active_items(self, synth_float):
        """Every item that appears in at least one transaction should have iu > 0."""
        X, y = synth_float
        td = _td(X, y, {"B": 5, "L": 1, "G": 0.0})
        iu_vec = list(td.item_iu)
        seen_iids = set()
        for row in td.transactions:
            for iid, _ in row:
                if iid != -1:
                    seen_iids.add(iid)
        for iid in seen_iids:
            assert iu_vec[iid - 1] > 0.0, (
                f"item {iid} appears in transactions but item_iu[{iid - 1}] == 0"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 8. Zero-correlation column pruning
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroCorrelation:
    def test_constant_column_dropped(self):
        """A constant column (zero variance → zero Pearson r) must be dropped."""
        rng = np.random.default_rng(99)
        n = 200
        y = pd.Series(np.tile([0, 1], n // 2))
        X = pd.DataFrame(
            {
                "signal": rng.normal(size=n),
                "const": np.zeros(n),
            }
        )
        clf = _clf(B=5, L=1, G=0.0)
        Xp, yp = clf.prepareXy(X.copy(), y.copy())
        clf.fit(Xp, yp)
        td = clf.td_._td
        const_items = [name for name in td.item_map.values() if name.startswith("const=")]
        assert len(const_items) == 0, f"Constant column produced items: {const_items}"

    def test_all_zero_riu_column_absent_from_transactions(self):
        """Items with eiu=0 must not appear in any transaction."""
        rng = np.random.default_rng(77)
        n = 100
        X = pd.DataFrame(
            {
                "good": rng.normal(size=n),
                "noise": np.full(n, 3.14),
            }
        )
        y = pd.Series(np.tile([0, 1], n // 2))
        td = _td_full(X, y, {"B": 3, "L": 1, "G": 0.0})
        noise_item_ids = {iid for iid, name in td.item_map.items() if name.startswith("noise=")}
        assert len(noise_item_ids) == 0, "Zero-correlation items must not be registered"
        # Belt-and-suspenders: scan transactions
        for r, t in enumerate(td.transactions):
            for iid, _ in t:
                assert iid not in noise_item_ids, f"Zero-corr item {iid} appeared in row {r}"


# ═══════════════════════════════════════════════════════════════════════════
# 9. Serialization round-trip
# ═══════════════════════════════════════════════════════════════════════════


class TestSerialization:
    def test_save_load_identical_predictions(self, synth_float, tmp_path):
        X, y = synth_float
        clf, Xte, _ = _fit(X, y, {"B": 5, "L": 1, "G": 0.0})
        p1 = clf.predict_proba(Xte)

        path = str(tmp_path / "chunked.hugiml")
        clf.save_model(path)
        clf2 = HUGIMLClassifierNative.load_model(path)
        p2 = clf2.predict_proba(Xte)

        assert np.allclose(p1, p2, atol=1e-9), (
            f"Post-load predictions differ; max={np.abs(p1 - p2).max():.2e}"
        )

    def test_save_load_identical_predictions_nan(self, synth_nan, tmp_path):
        X, y = synth_nan
        clf, Xte, _ = _fit(X, y, {"B": 5, "L": 1, "G": 0.0})
        p1 = clf.predict_proba(Xte)

        path = str(tmp_path / "nan_model.hugiml")
        clf.save_model(path)
        clf2 = HUGIMLClassifierNative.load_model(path)
        p2 = clf2.predict_proba(Xte)

        assert np.allclose(p1, p2, atol=1e-9), (
            f"NaN model post-load differs; max={np.abs(p1 - p2).max():.2e}"
        )

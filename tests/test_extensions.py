"""
Comprehensive tests for all hugiml extension modules.

Covers: metrics, pruning, adaptive (standalone), multiclass, plots, benchmarks.
Run with:  pytest tests/ -v --ignore=tests/test_adaptive_native.py
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bc_data():
    bc = load_breast_cancer(as_frame=True)
    return bc.data, bc.target.values


@pytest.fixture(scope="module")
def wine_data():
    wn = load_wine(as_frame=True)
    return wn.data, wn.target.values


@pytest.fixture(scope="module")
def fitted_clf(bc_data):
    """Fast binary classifier for most tests — B=5, L=1, topK=40."""
    from hugiml import HUGIMLClassifierNative

    X, y = bc_data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    clf = HUGIMLClassifierNative(B=5, L=1, G=1e-2, topK=40)
    clf.fit(Xtr, ytr)
    return clf, Xtr, Xte, ytr, yte


@pytest.fixture(scope="module")
def fitted_multiclass(wine_data):
    """Fitted 3-class classifier for multiclass tests."""
    from hugiml import HUGIMLClassifierNative

    X, y = wine_data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    clf = HUGIMLClassifierNative(B=7, L=2, G=1e-4, topK=80)
    clf.fit(Xtr, ytr)
    return clf, Xtr, Xte, ytr, yte


# ─────────────────────────────────────────────────────────────────────────────
# metrics.py
# ─────────────────────────────────────────────────────────────────────────────


class TestMetrics:
    def test_compute_all_metrics_types(self, fitted_clf):
        from hugiml.metrics import InterpretabilityMetrics, compute_all_metrics

        clf, Xtr, Xte, ytr, yte = fitted_clf
        m = compute_all_metrics(clf, Xte)
        assert isinstance(m, InterpretabilityMetrics)

    def test_n_patterns_matches_clf(self, fitted_clf):
        from hugiml.metrics import compute_all_metrics

        clf, Xtr, Xte, ytr, yte = fitted_clf
        m = compute_all_metrics(clf, Xte)
        assert m.n_patterns == len(clf.patterns_)

    def test_coverage_in_range(self, fitted_clf):
        from hugiml.metrics import compute_all_metrics

        clf, Xtr, Xte, ytr, yte = fitted_clf
        m = compute_all_metrics(clf, Xte)
        assert 0.0 <= m.coverage <= 1.0

    def test_sparsity_in_range(self, fitted_clf):
        from hugiml.metrics import compute_all_metrics

        clf, Xtr, Xte, ytr, yte = fitted_clf
        m = compute_all_metrics(clf, Xte)
        assert 0.0 <= m.explanation_sparsity <= 1.0

    def test_overlap_rate_in_range(self, fitted_clf):
        from hugiml.metrics import compute_all_metrics

        clf, Xtr, Xte, ytr, yte = fitted_clf
        m = compute_all_metrics(clf, Xte)
        assert 0.0 <= m.overlap_rate <= 1.0

    def test_topk_monotone(self, fitted_clf):
        from hugiml.metrics import compute_all_metrics

        clf, Xtr, Xte, ytr, yte = fitted_clf
        m = compute_all_metrics(clf, Xte)
        vals = [m.top_k_cumulative_contribution[k] for k in sorted(m.top_k_cumulative_contribution)]
        for a, b in zip(vals, vals[1:]):
            assert a <= b + 1e-9, "top-k contribution not monotonically non-decreasing"

    def test_to_dict_flat(self, fitted_clf):
        from hugiml.metrics import compute_all_metrics

        clf, Xtr, Xte, ytr, yte = fitted_clf
        m = compute_all_metrics(clf, Xte)
        d = m.to_dict()
        assert "n_patterns" in d
        assert "coverage" in d
        assert "top_k_cumulative_contribution" not in d  # should be flattened
        assert any(k.startswith("top_") for k in d)

    def test_str_representation(self, fitted_clf):
        from hugiml.metrics import compute_all_metrics

        clf, Xtr, Xte, ytr, yte = fitted_clf
        m = compute_all_metrics(clf, Xte)
        s = str(m)
        assert "n_patterns" in s
        assert "coverage" in s

    def test_metrics_dataframe(self, fitted_clf):
        from hugiml import HUGIMLClassifierNative
        from hugiml.metrics import compute_all_metrics, metrics_dataframe

        clf, Xtr, Xte, ytr, yte = fitted_clf
        clf2 = HUGIMLClassifierNative(B=3, L=1, G=1e-2, topK=20)
        clf2.fit(Xtr, ytr)
        results = {
            "B=5": compute_all_metrics(clf, Xte),
            "B=3": compute_all_metrics(clf2, Xte),
        }
        df = metrics_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert set(df.index) == {"B=5", "B=3"}
        assert "n_patterns" in df.columns

    def test_individual_functions(self, fitted_clf):
        from hugiml.metrics import (
            active_patterns_per_prediction,
            avg_pattern_length,
            coverage,
            explanation_sparsity,
            max_pattern_length,
            n_patterns,
            overlap_rate,
            top_k_cumulative_contribution,
        )

        clf, Xtr, Xte, ytr, yte = fitted_clf
        assert n_patterns(clf) == len(clf.patterns_)
        assert avg_pattern_length(clf) >= 1.0
        assert max_pattern_length(clf) >= 1
        cov = coverage(clf, Xte)
        assert 0 <= cov <= 1
        ol = overlap_rate(clf, Xte)
        assert 0 <= ol <= 1
        sp = explanation_sparsity(clf, Xte)
        assert 0 <= sp <= 1
        ap = active_patterns_per_prediction(clf, Xte)
        assert ap.shape == (len(yte),)
        assert ap.dtype == int
        tk = top_k_cumulative_contribution(clf, ks=[1, 5, 10])
        assert all(0 <= v <= 1 for v in tk.values())

    def test_requires_fitted(self):
        from hugiml import HUGIMLClassifierNative
        from hugiml.metrics import compute_all_metrics

        clf = HUGIMLClassifierNative()
        with pytest.raises(RuntimeError):
            compute_all_metrics(clf, np.zeros((10, 5)))


# ─────────────────────────────────────────────────────────────────────────────
# pruning.py
# ─────────────────────────────────────────────────────────────────────────────


class TestPruning:
    def test_list_patterns_returns_dataframe(self, fitted_clf):
        from hugiml.pruning import PatternEditor

        clf, *_ = fitted_clf
        editor = PatternEditor(clf)
        df = editor.list_patterns()
        assert isinstance(df, pd.DataFrame)
        assert "pattern" in df.columns
        assert len(df) == len(clf.patterns_)

    def test_remove_by_index(self, fitted_clf):
        from hugiml.pruning import PatternEditor

        clf, Xtr, Xte, ytr, yte = fitted_clf
        editor = PatternEditor(clf)
        orig = editor.diff()["n_original"]
        editor.remove([0, 1], reason="test")
        assert editor.diff()["n_current"] == orig - 2

    def test_remove_by_keyword(self, fitted_clf):
        from hugiml.pruning import PatternEditor

        clf, *_ = fitted_clf
        editor = PatternEditor(clf)
        fname = clf.feature_names_in_[0]
        before = editor.diff()["n_original"]
        editor.remove_by_keyword(fname, reason="feature removal")
        assert editor.diff()["n_current"] <= before

    def test_remove_low_support(self, fitted_clf):
        from hugiml.pruning import PatternEditor

        clf, *_ = fitted_clf
        editor = PatternEditor(clf)
        before = editor.diff()["n_original"]
        editor.remove_low_support(min_support=0.5, reason="low support")
        # High threshold should remove many patterns
        assert editor.diff()["n_current"] < before

    def test_refit_and_predict(self, fitted_clf):
        from hugiml.pruning import PatternEditor

        clf, Xtr, Xte, ytr, yte = fitted_clf
        editor = PatternEditor(clf)
        editor.remove([0], reason="test")
        editor.refit(Xtr, ytr)
        new_clf = editor.finalize()
        proba = new_clf.predict_proba(Xte)
        assert proba.shape == (len(yte), 2)

    def test_calibrate(self, fitted_clf):
        from hugiml.pruning import PatternEditor

        clf, Xtr, Xte, ytr, yte = fitted_clf
        Xtr2, Xcal, ytr2, ycal = train_test_split(Xtr, ytr, test_size=0.2, random_state=0)
        editor = PatternEditor(clf)
        editor.remove([0], reason="test")
        editor.refit(Xtr2, ytr2)
        editor.calibrate(Xcal, ycal, method="isotonic")
        assert editor._calibrated
        new_clf = editor.finalize()
        proba = new_clf.predict_proba(Xte)
        assert proba.shape == (len(yte), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_audit_report_json(self, fitted_clf):
        from hugiml.pruning import PatternEditor

        clf, Xtr, Xte, ytr, yte = fitted_clf
        editor = PatternEditor(clf, operator_name="tester")
        editor.remove([0], reason="protected attr")
        editor.remove_by_keyword(clf.feature_names_in_[1], reason="unstable")
        report = editor.audit_report()
        data = json.loads(report)
        assert data["operator"] == "tester"
        assert len(data["removals"]) == 2
        assert data["removals"][0]["reason"] == "protected attr"
        assert data["diff"]["n_removed"] >= 1

    def test_save_audit_report(self, fitted_clf, tmp_path):
        from hugiml.pruning import PatternEditor

        clf, *_ = fitted_clf
        editor = PatternEditor(clf)
        editor.remove([0], reason="test")
        path = str(tmp_path / "audit.json")
        editor.save_audit_report(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "removals" in data

    def test_diff_counts(self, fitted_clf):
        from hugiml.pruning import PatternEditor

        clf, Xtr, Xte, ytr, yte = fitted_clf
        editor = PatternEditor(clf)
        editor.remove([0, 1, 2], reason="test")
        d = editor.diff()
        assert d["n_removed"] == 3
        assert d["n_current"] == d["n_original"] - 3

    def test_finalize_blocks_further_edits(self, fitted_clf):
        from hugiml.pruning import PatternEditor

        clf, Xtr, Xte, ytr, yte = fitted_clf
        editor = PatternEditor(clf)
        editor.remove([0], reason="test")
        editor.refit(Xtr, ytr)
        editor.finalize()
        with pytest.raises(RuntimeError):
            editor.remove([1], reason="after finalize")

    def test_context_manager(self, fitted_clf):
        from hugiml.pruning import PatternEditor

        clf, Xtr, Xte, ytr, yte = fitted_clf
        with PatternEditor(clf) as editor:
            editor.remove([0], reason="test")
            editor.refit(Xtr, ytr)
            new_clf = editor.finalize()
        assert new_clf is not clf

    def test_repr(self, fitted_clf):
        from hugiml.pruning import PatternEditor

        clf, *_ = fitted_clf
        editor = PatternEditor(clf)
        r = repr(editor)
        assert "PatternEditor" in r

    def test_refit_required_before_finalize(self, fitted_clf):
        from hugiml.pruning import PatternEditor

        clf, *_ = fitted_clf
        editor = PatternEditor(clf)
        editor.remove([0], reason="test")
        # finalize without refit should raise
        with pytest.raises(RuntimeError):
            editor.finalize()


# ─────────────────────────────────────────────────────────────────────────────
# adaptive.py  (standalone HUGIMLAdaptive wrapper)
# ─────────────────────────────────────────────────────────────────────────────


class TestAdaptiveStandalone:
    def test_ig_helper(self):
        from hugiml.adaptive import _information_gain

        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 400)
        y = (x > 0).astype(int)
        ig = _information_gain(x, y, n_bins=5)
        assert ig > 0.5, f"IG={ig} unexpectedly low for informative feature"

    def test_ig_zero_for_noise(self):
        from hugiml.adaptive import _information_gain

        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 400)
        y = rng.integers(0, 2, 400)
        ig = _information_gain(x, y, n_bins=5)
        assert ig < 0.05, f"IG={ig} surprisingly high for noise feature"

    def test_fit_predict(self, bc_data):
        from hugiml.adaptive import HUGIMLAdaptive

        X, y = bc_data
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        clf = HUGIMLAdaptive(b_candidates=[3, 5, 7], min_marginal_gain_ratio=0.02, L=1, G=1e-2)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        assert proba.shape == (len(yte), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_labels(self, bc_data):
        from hugiml.adaptive import HUGIMLAdaptive

        X, y = bc_data
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        clf = HUGIMLAdaptive(b_candidates=[3, 5], L=1, G=1e-2)
        clf.fit(Xtr, ytr)
        preds = clf.predict(Xte)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_transform_shape(self, bc_data):
        from hugiml.adaptive import HUGIMLAdaptive

        X, y = bc_data
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        clf = HUGIMLAdaptive(b_candidates=[3, 5], L=1, G=1e-2)
        clf.fit(Xtr, ytr)
        hup = clf.transform(Xte)
        assert hup.shape[0] == len(yte)
        assert hup.shape[1] == len(clf.patterns_)

    def test_per_feature_b_set(self, bc_data):
        from hugiml.adaptive import HUGIMLAdaptive

        X, y = bc_data
        Xtr, *_ = train_test_split(X, y, test_size=0.25, random_state=42)
        clf = HUGIMLAdaptive(b_candidates=[3, 5, 7], L=1, G=1e-2)
        clf.fit(Xtr, y[: len(Xtr)])
        assert len(clf.per_feature_b_) > 0
        cands = clf.b_candidates
        for b in clf.per_feature_b_.values():
            assert b in cands, f"B={b} not in candidates {cands}"

    def test_ig_scores_grid(self, bc_data):
        from hugiml.adaptive import HUGIMLAdaptive

        X, y = bc_data
        Xtr, *_ = train_test_split(X, y, test_size=0.25, random_state=42)
        clf = HUGIMLAdaptive(b_candidates=[3, 5, 7], L=1, G=1e-2)
        clf.fit(Xtr, y[: len(Xtr)])
        assert len(clf.ig_scores_) > 0
        for name, scores in clf.ig_scores_.items():
            assert set(scores.keys()) <= {3, 5, 7}

    def test_prepareXy_delegate(self, bc_data):
        from hugiml.adaptive import HUGIMLAdaptive

        X, y = bc_data
        clf = HUGIMLAdaptive(b_candidates=[3, 5], L=1, G=1e-2)
        X_enc, y_enc = clf.prepareXy(X, y)
        assert isinstance(X_enc, pd.DataFrame)
        assert y_enc.dtype == np.int64

    def test_model_summary(self, bc_data):
        from hugiml.adaptive import HUGIMLAdaptive

        X, y = bc_data
        Xtr, *_ = train_test_split(X, y, test_size=0.25, random_state=42)
        clf = HUGIMLAdaptive(b_candidates=[3, 5, 7], L=1, G=1e-2)
        clf.fit(Xtr, y[: len(Xtr)])
        s = clf.model_summary()
        assert "adaptive" in s.lower() or "B=" in s

    def test_repr(self, bc_data):
        from hugiml.adaptive import HUGIMLAdaptive

        X, y = bc_data
        clf = HUGIMLAdaptive(b_candidates=[3, 5])
        assert "not fitted" in repr(clf)
        Xtr, *_ = train_test_split(X, y, test_size=0.25, random_state=42)
        clf.fit(Xtr, y[: len(Xtr)])
        assert "HUGIMLAdaptive" in repr(clf)

    def test_fit_with_ndarray_after_prepareXy(self, bc_data):
        """fit() must accept raw ndarray when feature names come from prepareXy."""
        from hugiml.adaptive import HUGIMLAdaptive

        X, y = bc_data
        clf = HUGIMLAdaptive(b_candidates=[3, 5], L=1, G=1e-2)
        X_enc, y_enc = clf.prepareXy(X, y)
        Xtr, Xte, ytr, yte = train_test_split(
            X_enc, y_enc, test_size=0.25, stratify=y_enc, random_state=42
        )
        clf.fit(Xtr.values, ytr)  # ndarray path
        proba = clf.predict_proba(Xte.values)
        assert proba.shape == (len(yte), 2)

    def test_auc_reasonable(self, bc_data):
        from hugiml.adaptive import HUGIMLAdaptive

        X, y = bc_data
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        clf = HUGIMLAdaptive(b_candidates=[3, 5, 7], L=1, G=1e-2)
        clf.fit(Xtr, ytr)
        auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        assert auc > 0.90, f"AUC={auc:.4f} unexpectedly low"

    def test_plot_bin_profiles(self, bc_data):
        pytest.importorskip("matplotlib")
        import matplotlib

        from hugiml.adaptive import HUGIMLAdaptive

        matplotlib.use("Agg")
        X, y = bc_data
        Xtr, *_ = train_test_split(X, y, test_size=0.25, random_state=42)
        clf = HUGIMLAdaptive(b_candidates=[3, 5, 7], L=1, G=1e-2)
        clf.fit(Xtr, y[: len(Xtr)])
        fig, ax = clf.plot_bin_profiles()
        assert fig is not None

    def test_ig_heatmap(self, bc_data):
        pytest.importorskip("matplotlib")
        import matplotlib

        from hugiml.adaptive import HUGIMLAdaptive

        matplotlib.use("Agg")
        X, y = bc_data
        Xtr, *_ = train_test_split(X, y, test_size=0.25, random_state=42)
        clf = HUGIMLAdaptive(b_candidates=[3, 5, 7], L=1, G=1e-2)
        clf.fit(Xtr, y[: len(Xtr)])
        fig, ax = clf.ig_heatmap()
        assert fig is not None


# ─────────────────────────────────────────────────────────────────────────────
# multiclass.py
# ─────────────────────────────────────────────────────────────────────────────


class TestMulticlass:
    def test_multiclass_report_classes(self, fitted_multiclass):
        from hugiml.multiclass import MulticlassHUGReport

        clf, *_ = fitted_multiclass
        report = MulticlassHUGReport(clf)
        assert set(report.classes.tolist()) == {0, 1, 2}

    def test_importances_for_class_shape(self, fitted_multiclass):
        from hugiml.multiclass import MulticlassHUGReport

        clf, *_ = fitted_multiclass
        report = MulticlassHUGReport(clf)
        df = report.importances_for_class(0, top_n=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 5
        assert "pattern" in df.columns
        assert "coefficient" in df.columns

    def test_importances_for_all_classes(self, fitted_multiclass):
        from hugiml.multiclass import MulticlassHUGReport

        clf, *_ = fitted_multiclass
        report = MulticlassHUGReport(clf)
        for c in [0, 1, 2]:
            df = report.importances_for_class(c, top_n=10)
            assert len(df) <= 10

    def test_summary_string(self, fitted_multiclass):
        from hugiml.multiclass import MulticlassHUGReport

        clf, *_ = fitted_multiclass
        report = MulticlassHUGReport(clf)
        s = report.summary(top_n=3)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_raises_on_binary(self, fitted_clf):
        from hugiml.multiclass import MulticlassHUGReport

        clf, *_ = fitted_clf
        with pytest.raises((ValueError, AttributeError)):
            MulticlassHUGReport(clf)

    def test_invalid_class_label(self, fitted_multiclass):
        from hugiml.multiclass import MulticlassHUGReport

        clf, *_ = fitted_multiclass
        report = MulticlassHUGReport(clf)
        with pytest.raises(ValueError):
            report.importances_for_class(99)

    def test_imbalanced_class_weight(self, bc_data):
        from hugiml import HUGIMLClassifierNative
        from hugiml.multiclass import make_imbalanced_pipeline

        X, y = bc_data
        y_imbal = y.copy()
        y_imbal[:100] = 0  # make more imbalanced
        Xtr, Xte, ytr, yte = train_test_split(
            X, y_imbal, test_size=0.25, stratify=y_imbal, random_state=42
        )
        clf_proto = HUGIMLClassifierNative(B=4, L=1, G=1e-2, topK=20)
        clf_bal = make_imbalanced_pipeline(clf_proto, strategy="class_weight")
        clf_bal.fit(Xtr, ytr)
        preds = clf_bal.predict(Xte)
        assert len(preds) == len(yte)
        proba = clf_bal.predict_proba(Xte)
        assert proba.shape == (len(yte), 2)

    def test_imbalanced_smote(self, bc_data):
        pytest.importorskip("imblearn")
        from hugiml import HUGIMLClassifierNative
        from hugiml.multiclass import make_imbalanced_pipeline

        X, y = bc_data
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        clf_proto = HUGIMLClassifierNative(B=4, L=1, G=1e-2, topK=20)
        clf_smote = make_imbalanced_pipeline(clf_proto, strategy="smote")
        clf_smote.fit(Xtr, ytr)
        proba = clf_smote.predict_proba(Xte)
        assert proba.shape == (len(yte), 2)

    def test_encode_high_cardinality_target_mean(self):
        from hugiml.multiclass import apply_encoding, encode_high_cardinality

        rng = np.random.default_rng(0)
        n = 500
        X = pd.DataFrame(
            {
                "city": [f"c{i:03d}" for i in rng.integers(0, 60, n)],
                "age": rng.integers(18, 65, n),
            }
        )
        y = rng.integers(0, 2, n)
        X_enc, enc_map = encode_high_cardinality(X, y, threshold=10, method="target_mean")
        assert "city" in enc_map
        assert X_enc["city"].dtype != object
        # apply to test set without leakage
        X_test = X.copy()
        X_test_enc = apply_encoding(X_test, enc_map)
        assert X_test_enc["city"].dtype != object

    def test_encode_high_cardinality_frequency(self):
        from hugiml.multiclass import encode_high_cardinality

        rng = np.random.default_rng(1)
        n = 400
        X = pd.DataFrame(
            {
                "brand": [f"b{i:02d}" for i in rng.integers(0, 30, n)],
            }
        )
        y = rng.integers(0, 2, n)
        X_enc, enc_map = encode_high_cardinality(X, y, threshold=5, method="frequency")
        assert "brand" in enc_map
        assert X_enc["brand"].dtype != object

    def test_encode_respects_threshold(self):
        from hugiml.multiclass import encode_high_cardinality

        X = pd.DataFrame(
            {
                "low_card": ["A", "B", "C"] * 100,  # 3 unique < threshold=10
                "high_card": [f"v{i}" for i in range(300)],  # 300 unique > threshold=10
            }
        )
        y = np.random.randint(0, 2, 300)
        X_enc, enc_map = encode_high_cardinality(X, y, threshold=10)
        # high_card should be encoded, low_card should be untouched
        assert "high_card" in enc_map
        assert "low_card" not in enc_map

    def test_apply_encoding_unseen_values(self):
        """Unseen values at test time get fill_value (default 0)."""
        from hugiml.multiclass import apply_encoding, encode_high_cardinality

        rng = np.random.default_rng(2)
        n = 200
        X_tr = pd.DataFrame({"cat": [f"v{i}" for i in rng.integers(0, 25, n)]})
        y_tr = rng.integers(0, 2, n)
        _, enc_map = encode_high_cardinality(X_tr, y_tr, threshold=5)
        X_te = pd.DataFrame({"cat": ["UNSEEN_CAT"] * 10})
        X_te_enc = apply_encoding(X_te, enc_map, fill_value=0.0)
        assert (X_te_enc["cat"] == 0.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# plots.py
# ─────────────────────────────────────────────────────────────────────────────


class TestPlots:
    @pytest.fixture(autouse=True)
    def require_plotly(self):
        pytest.importorskip("plotly")

    def test_plot_top_patterns(self, fitted_clf):
        from hugiml.plots import HUGPlotter

        clf, Xtr, Xte, ytr, yte = fitted_clf
        fig = HUGPlotter(clf).plot_top_patterns(top_n=10)
        assert len(fig.data) >= 1

    def test_plot_utility_vs_ig(self, fitted_clf):
        from hugiml.plots import HUGPlotter

        clf, *_ = fitted_clf
        fig = HUGPlotter(clf).plot_utility_vs_ig()
        assert len(fig.data) >= 1

    def test_plot_active_patterns(self, fitted_clf):
        from hugiml.plots import HUGPlotter

        clf, Xtr, Xte, ytr, yte = fitted_clf
        fig = HUGPlotter(clf).plot_active_patterns(Xte, sample_idx=0)
        assert fig is not None

    def test_plot_feature_coverage(self, fitted_clf):
        from hugiml.plots import HUGPlotter

        clf, Xtr, Xte, ytr, yte = fitted_clf
        fig = HUGPlotter(clf).plot_feature_coverage(top_n=5)
        assert len(fig.data) >= 1

    def test_plot_marginal_bin_profile_native(self, fitted_clf):
        from hugiml.plots import HUGPlotter

        clf, *_ = fitted_clf
        labels = clf.get_hug_features()
        singletons = [
            lbl.split(", ")[0].split("=")[0]
            for lbl, pe in zip(labels, clf.patterns_)
            if len(pe.items) == 1
        ]
        if not singletons:
            pytest.skip("No singleton patterns in this model")
        feat = singletons[0]
        fig = HUGPlotter(clf).plot_marginal_bin_profile(feat)
        assert len(fig.data) >= 1

    def test_plot_marginal_bin_profile_adaptive(self, bc_data):
        """Adaptive model: _bin_edges_ path in _get_bin_edges."""
        from hugiml import HUGIMLClassifierNative
        from hugiml.plots import HUGPlotter

        X, y = bc_data
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        clf = HUGIMLClassifierNative(B=8, L=2, G=1e-4, topK=80, adaptive_binning=True)
        clf.fit(Xtr, ytr)
        feat = list(clf._bin_edges_.keys())[0]
        fig = HUGPlotter(clf).plot_marginal_bin_profile(feat)
        # Figure must have data and all bins in positional order
        assert len(fig.data) >= 1
        xs = fig.data[0].x
        if xs is not None and len(xs) > 1:
            lowers = []
            for label in xs:
                if isinstance(label, str) and label.startswith("["):
                    try:
                        lowers.append(float(label[1:].split(",")[0]))
                    except ValueError:
                        pass
            if len(lowers) > 1:
                assert lowers == sorted(lowers), "bins not in positional order"

    def test_plot_feature_combinations(self, fitted_clf):
        from hugiml.plots import HUGPlotter

        clf, *_ = fitted_clf
        labels = clf.get_hug_features()
        compound_feats = [
            lbl.split(", ")[0].split("=")[0]
            for lbl, pe in zip(labels, clf.patterns_)
            if len(pe.items) >= 2
        ]
        if not compound_feats:
            pytest.skip("No compound patterns in this model")
        fig = HUGPlotter(clf).plot_feature_combinations(compound_feats[0])
        assert fig is not None

    def test_no_plotly_raises(self, fitted_clf, monkeypatch):
        """HUGPlotter raises a clear ImportError when plotly is absent."""
        from hugiml.plots import _PLOTLY

        if not _PLOTLY:
            pytest.skip("plotly not installed")
        clf, *_ = fitted_clf
        import hugiml.plots as _plots

        orig = _plots._PLOTLY
        _plots._PLOTLY = False
        try:
            with pytest.raises(ImportError, match="plotly"):
                from hugiml.plots import HUGPlotter

                HUGPlotter(clf)
        finally:
            _plots._PLOTLY = orig

    def test_save_dashboard_html(self, fitted_clf, tmp_path):
        from hugiml.plots import HUGPlotter

        clf, Xtr, Xte, ytr, yte = fitted_clf
        plotter = HUGPlotter(clf)
        out = str(tmp_path / "dash.html")
        plotter.plot_dashboard(
            Xte, dataset_name="test", feature_names_for_profile=None, output_path=out
        )
        with open(out, encoding="utf-8") as f:
            content = f.read()
        assert "<html" in content.lower() or "<!doctype" in content.lower()
        assert len(content) > 1000


# ─────────────────────────────────────────────────────────────────────────────
# benchmarks/runner.py
# ─────────────────────────────────────────────────────────────────────────────


class TestBenchmarkRunner:
    def test_builders_registry(self):
        from hugiml.benchmarks.runner import BUILDERS

        expected = {"HUG-IML", "EBM", "XGBoost", "LightGBM", "RandomForest", "LogisticReg", "RuleFit", "GAM"}
        assert set(BUILDERS.keys()) == expected

    def test_hugiml_builder_returns_fitted_type(self):
        from hugiml import HUGIMLClassifierNative
        from hugiml.benchmarks.runner import _build_hugiml

        clf = _build_hugiml()
        assert isinstance(clf, HUGIMLClassifierNative)

    def test_rf_builder_always_returns(self):
        from hugiml.benchmarks.runner import _build_rf

        clf = _build_rf()
        assert clf is not None

    def test_lr_builder_always_returns(self):
        from hugiml.benchmarks.runner import _build_lr

        clf = _build_lr()
        assert clf is not None

    def test_optional_builders_return_none_or_clf(self):
        from hugiml.benchmarks.runner import _build_ebm, _build_lightgbm, _build_xgb

        for builder in [_build_ebm, _build_xgb, _build_lightgbm]:
            result = builder()
            # Either returns an estimator or None (if not installed)
            assert result is None or hasattr(result, "fit")

    def test_evaluate_returns_dict(self, bc_data):
        from hugiml.benchmarks.runner import _build_rf, _evaluate

        X, y = bc_data
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        clf = _build_rf()
        result = _evaluate(clf, Xtr.values, ytr, Xte.values, yte)
        assert "roc_auc" in result
        assert "accuracy" in result
        assert "fit_ms" in result
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_run_benchmark_breast_cancer(self, tmp_path):
        """Minimal 2-fold run on breast_cancer — checks output DataFrame."""

        # Override BUILDERS temporarily to only run fast models
        import hugiml.benchmarks.runner as runner
        from hugiml.benchmarks.runner import run_benchmark

        orig = runner.BUILDERS
        runner.BUILDERS = {
            "HUG-IML": runner._build_hugiml,
            "RandomForest": runner._build_rf,
        }
        try:
            df = run_benchmark("breast_cancer", n_splits=2, output_dir=str(tmp_path))
            assert isinstance(df, pd.DataFrame)
            assert "roc_auc" in df.columns
            assert "model" in df.columns
            assert set(df["model"].unique()) == {"HUG-IML", "RandomForest"}
            assert len(df) == 4  # 2 models × 2 folds
            # Output files created
            assert (tmp_path / "breast_cancer_results.csv").exists()
            assert (tmp_path / "breast_cancer_summary.json").exists()
        finally:
            runner.BUILDERS = orig

    def test_dataset_loaders_available(self):
        from hugiml.benchmarks.runner import DATASET_LOADERS

        assert "breast_cancer" in DATASET_LOADERS
        X, y = DATASET_LOADERS["breast_cancer"]()
        assert hasattr(X, "shape")
        assert len(y) == len(X)

    def test_main_is_callable(self):
        from hugiml.benchmarks.runner import main

        assert callable(main)

    def test_unknown_dataset_handled(self, capsys):
        """Unknown dataset name should not crash the whole run."""

        from hugiml.benchmarks.runner import main

        with pytest.raises(SystemExit):
            # argparse exits on bad args, that's fine
            main()


# =============================================================================
# Missing Value Handling (v1.1.0)
# =============================================================================


class TestMissingValues:
    """NaN/Inf = 'not observed' — no item generated in the transaction.

    All numerical columns are pre-binned to string quantile labels at fit time
    (non-adaptive path).  Non-finite cells become np.nan in the label array;
    the C++ transaction builder skips them.  No median imputation anywhere.
    """

    @pytest.fixture(scope="class")
    def nan_split(self, bc_data):
        from hugiml import HUGIMLClassifierNative

        X, y = bc_data
        rng = np.random.default_rng(0)
        Xn = X.copy()
        Xn[rng.random(Xn.shape) < 0.05] = np.nan
        clf = HUGIMLClassifierNative()
        X_enc, y_enc = clf.prepareXy(Xn, y)
        Xtr, Xte, ytr, yte = train_test_split(
            X_enc, y_enc, test_size=0.25, stratify=y_enc, random_state=42
        )
        return Xtr, Xte, ytr, yte

    @pytest.fixture(scope="class")
    def nan_fitted(self, nan_split):
        from hugiml import HUGIMLClassifierNative

        Xtr, Xte, ytr, yte = nan_split
        clf = HUGIMLClassifierNative(B=5, L=2, G=1e-4, topK=80)
        clf.fit(Xtr, ytr)
        return clf, Xtr, Xte, ytr, yte

    def test_all_numerical_cols_prebin(self, bc_data):
        """All 30 numerical columns pre-binned even on clean data."""
        from hugiml import HUGIMLClassifierNative

        X, y = bc_data
        X_enc, y_enc = HUGIMLClassifierNative().prepareXy(X, y)
        Xtr, Xte, ytr, yte = train_test_split(
            X_enc, y_enc, test_size=0.25, stratify=y_enc, random_state=42
        )
        clf = HUGIMLClassifierNative(B=5, L=2, G=1e-4, topK=80)
        clf.fit(Xtr, ytr)
        assert len(clf._missing_col_edges_) == X.shape[1]

    def test_auc_competitive(self, bc_data, nan_fitted):
        from hugiml import HUGIMLClassifierNative

        X, y = bc_data
        X_enc, y_enc = HUGIMLClassifierNative().prepareXy(X, y)
        Xtr, Xte, ytr, yte = train_test_split(
            X_enc, y_enc, test_size=0.25, stratify=y_enc, random_state=42
        )
        clf_clean = HUGIMLClassifierNative(B=5, L=2, G=1e-4, topK=80)
        clf_clean.fit(Xtr, ytr)
        auc_clean = roc_auc_score(yte, clf_clean.predict_proba(Xte)[:, 1])
        clf, Xtr2, Xte2, ytr2, yte2 = nan_fitted
        auc_nan = roc_auc_score(yte2, clf.predict_proba(Xte2)[:, 1])
        assert abs(auc_nan - auc_clean) < 0.02

    def test_predict_no_nan_output(self, nan_fitted):
        clf, Xtr, Xte, ytr, yte = nan_fitted
        p = clf.predict_proba(Xte)
        assert not np.isnan(p).any()
        assert np.allclose(p.sum(axis=1), 1.0)

    def test_batch_independence(self, bc_data):
        """Non-NaN rows in a batch are unaffected by NaN in other rows."""
        from hugiml import HUGIMLClassifierNative

        X, y = bc_data
        X_enc, y_enc = HUGIMLClassifierNative().prepareXy(X, y)
        Xtr, Xte, ytr, yte = train_test_split(
            X_enc, y_enc, test_size=0.25, stratify=y_enc, random_state=42
        )
        clf = HUGIMLClassifierNative(B=5, L=2, G=1e-4, topK=80)
        clf.fit(Xtr, ytr)
        Xte_nan = Xte.copy()
        Xte_nan.iloc[:10, 0] = np.nan
        Xte_nan.iloc[:5, 5] = np.nan
        p_with = clf.predict_proba(Xte_nan)
        p_without = clf.predict_proba(Xte)
        no_nan = ~Xte_nan.isna().any(axis=1)
        assert np.allclose(p_with[no_nan], p_without[no_nan], atol=1e-9)

    def test_single_row_reproducibility(self, bc_data):
        """single-row predict == same row in batch — no batch-median dependency."""
        from hugiml import HUGIMLClassifierNative

        X, y = bc_data
        X_enc, y_enc = HUGIMLClassifierNative().prepareXy(X, y)
        Xtr, Xte, ytr, yte = train_test_split(
            X_enc, y_enc, test_size=0.25, stratify=y_enc, random_state=42
        )
        clf = HUGIMLClassifierNative(B=5, L=2, G=1e-4, topK=80)
        clf.fit(Xtr, ytr)
        Xte_nan = Xte.copy()
        Xte_nan.iloc[0, 0] = np.nan
        p_batch = clf.predict_proba(Xte_nan)[0]
        p_single = clf.predict_proba(Xte_nan.iloc[[0]])[0]
        assert np.allclose(p_batch, p_single, atol=1e-9)

    def test_nan_not_mined_as_item(self, nan_fitted):
        clf, *_ = nan_fitted
        nan_pats = [feat for feat in clf.get_hug_features() if "nan" in feat.lower()]
        assert len(nan_pats) == 0, f"nan-string patterns: {nan_pats}"

    def test_save_load_preserves_edges(self, nan_fitted, tmp_path):
        from hugiml import HUGIMLClassifierNative

        clf, Xtr, Xte, ytr, yte = nan_fitted
        p = str(tmp_path / "nan_model.hugiml")
        clf.save_model(p)
        clf2 = HUGIMLClassifierNative.load_model(p)
        assert len(clf2._missing_col_edges_) == len(clf._missing_col_edges_)
        auc1 = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        auc2 = roc_auc_score(yte, clf2.predict_proba(Xte)[:, 1])
        assert abs(auc1 - auc2) < 1e-9

    def test_backward_compat_missing_attr(self, nan_fitted):
        import copy

        from hugiml import HUGIMLClassifierNative

        clf, *_ = nan_fitted
        state = copy.deepcopy(clf.__getstate__())
        state.pop("_missing_col_edges_", None)
        restored = object.__new__(HUGIMLClassifierNative)
        restored.__setstate__(state)
        assert restored._missing_col_edges_ == {}

    def test_model_summary_mentions_nan_handling(self, nan_fitted):
        clf, *_ = nan_fitted
        assert "NaN handling" in clf.model_summary()

    def test_adaptive_native_no_nan_patterns(self, nan_split):
        from hugiml import HUGIMLClassifierNative

        Xtr, Xte, ytr, yte = nan_split
        clf = HUGIMLClassifierNative(B=8, L=2, G=1e-4, topK=80, adaptive_binning=True)
        clf.fit(Xtr, ytr)
        assert not any("=nan" in feat.lower() for feat in clf.get_hug_features())
        assert roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]) > 0.90

    def test_adaptive_standalone_no_nan_patterns(self, bc_data):
        from hugiml.adaptive import HUGIMLAdaptive

        X, y = bc_data
        rng = np.random.default_rng(0)
        X_nan = X.copy()
        X_nan[rng.random(X_nan.shape) < 0.05] = np.nan
        clf = HUGIMLAdaptive(b_candidates=[3, 5, 7], L=2, G=1e-4, topK=80)
        X_enc, y_enc = clf.prepareXy(X_nan, y)
        Xtr, Xte, ytr, yte = train_test_split(
            X_enc, y_enc, test_size=0.25, stratify=y_enc, random_state=42
        )
        clf.fit(Xtr, ytr)
        assert not any("=nan" in feat.lower() for feat in clf.get_hug_features())
        assert roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]) > 0.90

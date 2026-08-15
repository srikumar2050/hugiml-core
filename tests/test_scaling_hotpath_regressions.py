import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score

from hugiml import HUGIMLClassifierNative


def _make_signal(n=600, p=30, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = (X[:, : min(5, p)].sum(axis=1) + 0.15 * rng.normal(size=n) > 0).astype(int)
    return X, y


def test_fixed_numeric_l1_hotpath_transform_is_not_all_zero(monkeypatch):
    monkeypatch.delenv("HUGIML_DISABLE_FIXED_NUMERIC_L1_FASTPATH", raising=False)
    X, y = _make_signal()
    clf = HUGIMLClassifierNative(
        B=5,
        L=1,
        G=0.0,
        topK=120,
        adaptive_binning=False,
        use_hotpath=True,
        n_jobs=1,
    ).fit(X, y)

    Z = clf.transform_patterns(X[:25])
    assert isinstance(Z, csr_matrix)
    assert Z.shape[0] == 25
    assert Z.shape[1] == len(clf.patterns_)
    assert Z.nnz > 0
    assert not np.any(clf.cat_cols_mask_)
    assert getattr(clf, "_missing_col_edges_", {}) == {}
    assert roc_auc_score(y, clf.predict_proba(X)[:, 1]) > 0.80


def test_fixed_numeric_l1_hotpath_skips_nonfinite_cells(monkeypatch):
    monkeypatch.delenv("HUGIML_DISABLE_FIXED_NUMERIC_L1_FASTPATH", raising=False)
    X, y = _make_signal(n=300, p=12)
    X[0, 0] = np.nan
    X[1, 1] = np.inf
    clf = HUGIMLClassifierNative(
        B=4,
        L=1,
        G=0.0,
        topK=80,
        adaptive_binning=False,
        use_hotpath=True,
        n_jobs=1,
    ).fit(X, y)
    Z = clf.transform_patterns(X[:20])
    assert Z.nnz > 0
    proba = clf.predict_proba(X[:20])
    assert np.all(np.isfinite(proba))


def test_adaptive_ndarray_prebin_keeps_numpy_fast_path():
    X, y = _make_signal(n=400, p=40)
    clf = HUGIMLClassifierNative(
        B=5,
        L=1,
        G=0.0,
        topK=100,
        adaptive_binning=True,
        use_hotpath=True,
        n_jobs=1,
    ).fit(X, y)
    X_pre = clf._prebin_for_predict(X[:10])
    assert isinstance(X_pre, np.ndarray)
    assert X_pre.shape == X[:10].shape
    Z = clf.transform_patterns(X[:10])
    assert Z.nnz > 0

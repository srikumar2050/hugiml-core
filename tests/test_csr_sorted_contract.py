import numpy as np
import pytest
from scipy.sparse import csr_matrix


def _assert_csr_sorted(indptr, indices):
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    for r in range(len(indptr) - 1):
        row = indices[indptr[r] : indptr[r + 1]]
        assert np.all(row[:-1] <= row[1:])


def test_csr_sorted_contract_helper_shape():
    """Document the downstream scipy/sklearn sorted-index requirement."""
    indptr = np.array([0, 3, 5, 5], dtype=np.int32)
    indices = np.array([1, 3, 7, 0, 9], dtype=np.int32)
    X = csr_matrix((np.ones(indices.size, dtype=np.float32), indices, indptr), shape=(3, 10))
    assert X.has_sorted_indices
    _assert_csr_sorted(indptr, indices)


def test_native_csr_helpers_return_sorted_indices_when_available():
    core = pytest.importorskip("hugiml._hugiml_core")
    if not hasattr(core, "build_train_matrix_csr"):
        pytest.skip(
            "installed native extension does not expose build_train_matrix_csr; rebuild the current native extension to exercise this contract"
        )

    from hugiml import HUGIMLClassifier

    X = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.asarray([0, 0, 1, 1, 1, 0])

    clf = HUGIMLClassifier(B=3, L=1, G=0.0, adaptive_binning=False, use_hotpath=False)
    clf.fit(X, y)
    assert getattr(clf, "patterns_", None), (
        "CSR sorted-index fixture must mine at least one pattern"
    )

    native_td = getattr(clf.td_, "_td", clf.td_)
    indptr, indices = core.build_train_matrix_csr(native_td, clf.patterns_)
    _assert_csr_sorted(indptr, indices)
    X_csr = csr_matrix(
        (np.ones(len(indices), dtype=np.float32), np.asarray(indices), np.asarray(indptr)),
        shape=(X.shape[0], len(clf.patterns_)),
    )
    assert X_csr.has_sorted_indices


def test_fused_l1_result_get_csr_returns_sorted_indices_when_available():
    core = pytest.importorskip("hugiml._hugiml_core")
    if not hasattr(core, "prepare_and_mine_l1_adaptive"):
        pytest.skip("installed native extension does not expose fused adaptive L1 helper")

    from hugiml import HUGIMLClassifier

    X = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [3.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.asarray([0, 0, 1, 1, 1, 0, 1, 0])

    clf = HUGIMLClassifier(B=-1, L=1, G=0.0, adaptive_binning=True, use_hotpath=True, topK=-1)
    clf.fit(X, y)
    assert getattr(clf, "patterns_", None), "fused CSR fixture must mine at least one pattern"
    assert getattr(clf.x_train_hup_, "has_sorted_indices", True)
    _assert_csr_sorted(clf.x_train_hup_.indptr, clf.x_train_hup_.indices)

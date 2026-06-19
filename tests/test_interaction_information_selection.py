import numpy as np
import pytest

_core = pytest.importorskip("_hugiml_core")


def test_interaction_information_selects_xor_sources_with_missing_rows_skipped():
    rng = np.random.default_rng(17)
    n = 360
    x0 = rng.integers(0, 2, size=n).astype(float)
    x1 = rng.integers(0, 2, size=n).astype(float)
    y = (x0.astype(int) ^ x1.astype(int)).astype(np.int64)
    noise = rng.normal(size=(n, 4))
    X = np.column_stack([x0, x1, noise]).astype(np.float64)

    X[::13, 0] = np.nan
    X[::17, 1] = np.nan

    selected = _core.select_interaction_information_features(
        X,
        y,
        ["x0", "x1", "n0", "n1", "n2", "n3"],
        2,
        None,
    )
    names = {row["name"] for row in selected}
    assert names == {"x0", "x1"}


def test_interaction_information_partner_size_limits_anchors_without_losing_output_budget():
    rng = np.random.default_rng(23)
    n = 260
    X = rng.normal(size=(n, 12)).astype(np.float64)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)

    selected = _core.select_interaction_information_features(
        X,
        y,
        [f"f{i}" for i in range(X.shape[1])],
        5,
        4,
    )
    assert len(selected) == 5
    assert all("interaction_score" in row for row in selected)
    assert all(row["mode"] == "interaction_information" for row in selected)

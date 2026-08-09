import numpy as np
import pytest

_core = pytest.importorskip("_hugiml_core")


def _entropy(counts: np.ndarray, total: int) -> float:
    if total <= 0:
        return 0.0
    positive = counts[counts > 0].astype(float)
    probabilities = positive / float(total)
    return float(-np.sum(probabilities * np.log2(probabilities + 1e-15)))


def _quantile_codes(values: np.ndarray, max_bins: int = 4) -> np.ndarray:
    codes = np.full(values.size, -1, dtype=np.int32)
    finite_mask = np.isfinite(values)
    finite = np.sort(values[finite_mask])
    if finite.size == 0:
        return codes
    unique = np.unique(finite)
    if unique.size <= max(1, max_bins):
        codes[finite_mask] = np.searchsorted(unique, values[finite_mask], side="left")
        return codes
    edges: list[float] = []
    for k in range(1, max_bins):
        position = (k / max_bins) * (finite.size - 1)
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        fraction = position - lower
        edge = float(finite[lower] * (1.0 - fraction) + finite[upper] * fraction)
        if not edges or edge != edges[-1]:
            edges.append(edge)
    codes[finite_mask] = np.searchsorted(np.asarray(edges), values[finite_mask], side="right")
    return codes


def _conditional_entropy(counts: np.ndarray, totals: np.ndarray, eligible: int) -> float:
    value = 0.0
    for group, total in enumerate(totals):
        if total > 0:
            value += (float(total) / eligible) * _entropy(counts[group], int(total))
    return value


def _marginal_ig(codes: np.ndarray, y: np.ndarray, n_classes: int, n_bins: int) -> float:
    valid = (codes >= 0) & (codes < n_bins) & (y >= 0) & (y < n_classes)
    eligible = int(valid.sum())
    if eligible < 3:
        return 0.0
    label_counts = np.bincount(y[valid], minlength=n_classes)
    counts = np.zeros((n_bins, n_classes), dtype=np.int64)
    totals = np.zeros(n_bins, dtype=np.int64)
    np.add.at(counts, (codes[valid], y[valid]), 1)
    np.add.at(totals, codes[valid], 1)
    base = _entropy(label_counts, eligible)
    return max(0.0, base - _conditional_entropy(counts, totals, eligible))


def _reference_selection(X: np.ndarray, y: np.ndarray, names: list[str], keep: int) -> list[dict]:
    n_classes = max(1, int(np.max(y)) + 1)
    codes = [_quantile_codes(X[:, column]) for column in range(X.shape[1])]
    n_bins = [max(1, int(np.max(column_codes)) + 1) for column_codes in codes]
    marginal = [
        _marginal_ig(column_codes, y, n_classes, column_bins)
        for column_codes, column_bins in zip(codes, n_bins)
    ]
    best_score = np.zeros(X.shape[1], dtype=float)
    best_partner = np.full(X.shape[1], -1, dtype=np.int64)

    for left in range(X.shape[1]):
        for right in range(left + 1, X.shape[1]):
            left_codes = codes[left]
            right_codes = codes[right]
            valid = (
                (left_codes >= 0)
                & (left_codes < n_bins[left])
                & (right_codes >= 0)
                & (right_codes < n_bins[right])
                & (y >= 0)
                & (y < n_classes)
            )
            eligible = int(valid.sum())
            if eligible < 3:
                continue
            labels = y[valid]
            label_counts = np.bincount(labels, minlength=n_classes)
            base = _entropy(label_counts, eligible)
            if base <= 0.0:
                continue

            left_values = left_codes[valid]
            right_values = right_codes[valid]
            joint_values = left_values * n_bins[right] + right_values
            joint_counts = np.zeros((n_bins[left] * n_bins[right], n_classes), dtype=np.int64)
            joint_totals = np.zeros(n_bins[left] * n_bins[right], dtype=np.int64)
            left_counts = np.zeros((n_bins[left], n_classes), dtype=np.int64)
            left_totals = np.zeros(n_bins[left], dtype=np.int64)
            right_counts = np.zeros((n_bins[right], n_classes), dtype=np.int64)
            right_totals = np.zeros(n_bins[right], dtype=np.int64)
            np.add.at(joint_counts, (joint_values, labels), 1)
            np.add.at(joint_totals, joint_values, 1)
            np.add.at(left_counts, (left_values, labels), 1)
            np.add.at(left_totals, left_values, 1)
            np.add.at(right_counts, (right_values, labels), 1)
            np.add.at(right_totals, right_values, 1)

            joint_ig = max(0.0, base - _conditional_entropy(joint_counts, joint_totals, eligible))
            left_ig = max(0.0, base - _conditional_entropy(left_counts, left_totals, eligible))
            right_ig = max(0.0, base - _conditional_entropy(right_counts, right_totals, eligible))
            score = joint_ig - left_ig - right_ig
            if score > best_score[left]:
                best_score[left] = score
                best_partner[left] = right
            if score > best_score[right]:
                best_score[right] = score
                best_partner[right] = left

    order = sorted(range(X.shape[1]), key=lambda column: (-best_score[column], -marginal[column], names[column]))
    return [
        {
            "name": names[column],
            "score": float(best_score[column]),
            "marginal_ig": float(marginal[column]),
            "best_partner": names[best_partner[column]] if best_partner[column] >= 0 else None,
        }
        for column in order[:keep]
    ]


@pytest.mark.parametrize("case", ["complete_binary", "complete_multiclass", "effective_bins", "mixed_missing"])
def test_interaction_information_matches_pairwise_contingency_reference(case):
    rng = np.random.default_rng(20260809)
    if case == "complete_binary":
        X = rng.normal(size=(521, 9))
        y = (X[:, 0] * X[:, 1] + 0.25 * X[:, 2] > 0).astype(np.int64)
    elif case == "complete_multiclass":
        X = rng.normal(size=(607, 10))
        y = np.argmax(np.column_stack((X[:, 0], X[:, 1] - X[:, 2], -X[:, 0] + X[:, 3])), axis=1).astype(np.int64)
    elif case == "effective_bins":
        n = 560
        X = np.column_stack(
            (rng.integers(0, 2, n), rng.integers(0, 3, n), rng.integers(0, 4, n), np.ones(n), rng.normal(size=n))
        ).astype(float)
        y = ((X[:, 0] + X[:, 1] + (X[:, 2] == 3)) % 3).astype(np.int64)
    else:
        X = rng.normal(size=(577, 10))
        X[::7, 1] = np.nan
        X[::11, 4] = np.inf
        X[::13, 8] = -np.inf
        X[:, 9] = np.nan
        y = (np.nan_to_num(X[:, 0]) - 0.4 * np.nan_to_num(X[:, 3]) > 0).astype(np.int64)

    names = [f"f{column}" for column in range(X.shape[1])]
    keep = min(7, X.shape[1])
    expected = _reference_selection(X, y, names, keep)
    actual = [dict(row) for row in _core.select_interaction_information_features(X, y, names, keep, None)]

    assert [row["name"] for row in actual] == [row["name"] for row in expected]
    assert [row["best_partner"] for row in actual] == [row["best_partner"] for row in expected]
    assert [row["score"] for row in actual] == pytest.approx([row["score"] for row in expected], abs=1e-14)
    assert [row["marginal_ig"] for row in actual] == pytest.approx(
        [row["marginal_ig"] for row in expected], abs=1e-14
    )


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


def test_interaction_information_partner_restriction_is_repeatable_within_build():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(340, 15)).astype(np.float64)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0) ^ (X[:, 2] > 0)).astype(np.int64)
    names = [f"f{i}" for i in range(X.shape[1])]

    first = [dict(row) for row in _core.select_interaction_information_features(X, y, names, 7, 5)]
    second = [dict(row) for row in _core.select_interaction_information_features(X, y, names, 7, 5)]

    assert first == second
    assert len(first) == 7
    assert all(row["best_partner"] is None or row["best_partner"] in names for row in first)

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.benchmark import benchmark_engine


def test_training_limits_are_stratified_and_leave_all_test_rows() -> None:
    rng = np.random.default_rng(7)
    y = np.repeat(np.arange(3), [5000, 3000, 2000])
    signal = y + rng.normal(0, 0.05, len(y))
    X = pd.DataFrame(
        {
            "signal": signal,
            **{f"noise_{index}": rng.normal(size=len(y)) for index in range(39)},
        }
    )
    X_test = X.iloc[:1234].copy()

    limited_X, limited_y, limited_test, fields = benchmark_engine.limit_outer_training_data(
        X, y, X_test, max_rows=6000, max_predictors=30, random_state=42
    )

    assert limited_X.shape == (6000, 30)
    assert limited_test.shape == (1234, 30)
    assert "signal" in limited_X.columns
    assert limited_test.columns.tolist() == limited_X.columns.tolist()
    assert np.bincount(limited_y).tolist() == [3000, 1800, 1200]
    assert fields["outer_test_rows_evaluated"] == 1234
    assert fields["outer_train_rows_original"] == 10000
    assert fields["outer_train_rows_used"] == 6000
    assert len(json.loads(fields["outer_selected_predictors_json"])) == 30


def test_categorical_selection_is_invariant_to_category_labels() -> None:
    y = np.tile([0, 1], 100)
    original = pd.Series(np.where(y == 1, "high", "low"))
    renamed = original.map({"high": "zebra", "low": "apple"})
    first = benchmark_engine._categorical_target_correlation(original, y)
    second = benchmark_engine._categorical_target_correlation(renamed, y)
    assert first == second == 1.0


def test_no_limit_preserves_training_and_test_frames() -> None:
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})
    y = np.array([0, 1, 0, 1])
    X_test = X.iloc[:2].copy()
    limited_X, limited_y, limited_test, fields = benchmark_engine.limit_outer_training_data(
        X, y, X_test, max_rows=None, max_predictors=None, random_state=3
    )
    pd.testing.assert_frame_equal(limited_X, X)
    pd.testing.assert_frame_equal(limited_test, X_test)
    np.testing.assert_array_equal(limited_y, y)
    assert fields["outer_predictor_selection"] == "all"

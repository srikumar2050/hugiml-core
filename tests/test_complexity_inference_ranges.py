import json

import numpy as np

from hugiml.compute_complexity import (
    get_complexity_report,
    get_instance_inspection_units,
)


def _stump():
    return json.dumps(
        {
            "nodeid": 0,
            "split": "x0",
            "children": [
                {"nodeid": 1, "leaf": 0.7},
                {"nodeid": 2, "leaf": -0.4},
            ],
        }
    )


class _EarlyStoppedBooster:
    best_iteration = 0

    def num_boosted_rounds(self):
        return 2

    def get_dump(self, dump_format):
        assert dump_format == "json"
        return [_stump(), _stump(), _stump(), _stump()]


class _EarlyStoppedMulticlassXGB:
    best_iteration = 0

    def get_booster(self):
        return _EarlyStoppedBooster()

    def apply(self, X, iteration_range=None):
        assert iteration_range == (0, 1)
        leaf = np.where(np.asarray(X)[:, 0] <= 0, 1, 2)
        return np.column_stack([leaf, leaf])


class _CompleteBooster:
    def get_dump(self, dump_format):
        assert dump_format == "json"
        return [_stump(), _stump()]


class _CompleteXGB:
    def get_booster(self):
        return _CompleteBooster()

    def apply(self, X):
        leaf = np.where(np.asarray(X)[:, 0] <= 0, 1, 2)
        return np.column_stack([leaf, leaf])


def test_early_stopped_xgboost_counts_only_inference_trees():
    X = np.asarray([[-1.0], [1.0]])
    model = _EarlyStoppedMulticlassXGB()
    report = get_complexity_report(model, X=X)

    assert report["model_units"]["tree_count"] == 2
    assert report["model_units"]["trained_tree_count"] == 4
    assert report["model_units"]["effective_boosting_rounds"] == 1
    assert report["instance_inspection_units"]["available"] is True
    np.testing.assert_array_equal(get_instance_inspection_units(model, X), [2, 2])


def test_fully_fitted_xgboost_counts_the_complete_estimator():
    X = np.asarray([[-1.0], [1.0]])
    report = get_complexity_report(_CompleteXGB(), X=X)

    assert report["model_units"]["tree_count"] == 2
    assert report["model_units"]["trained_tree_count"] == 2
    assert report["model_units"]["effective_boosting_rounds"] is None
    assert report["instance_inspection_units"]["available"] is True

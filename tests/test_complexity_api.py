from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hugiml import (
    HUGIMLClassifierNative,
    get_complexity,
    get_complexity_report,
    get_instance_inspection_units,
)
from hugiml.rpte_bounded_lookahead_leafwise import (
    LeafWiseBoundedLookaheadRPTEFeatureLR,
)


def _interaction_dataset(n: int = 420, seed: int = 913) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    y = (((X["x0"] > 0.1) & (X["x1"] < -0.1)) | (X["x2"] > 1.0)).astype(int)
    return X, y.to_numpy()


def _base_kwargs() -> dict:
    return {
        "B": -1,
        "adaptive_binning": True,
        "L": 2,
        "topK": 40,
        "feature_mode": "patterns_only",
        "G": 0.001,
        "convert_binary_to_categorical": False,
        "augmented_pair_transforms": False,
        "topk_budget_strict": False,
        "execution_mode": "production",
        "n_jobs": 1,
    }


def _is_active_row(row: dict, tolerance: float = 1e-12) -> bool:
    try:
        value = float(row.get("final_logistic_coefficient"))
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and abs(value) > tolerance)


def _hug_feature_inspection_size(clf: HUGIMLClassifierNative, name: str) -> int:
    pattern_sizes: dict[str, int] = {}
    for label, pattern in zip(clf.get_hug_features(), clf.patterns_):
        size = max(1, len(getattr(pattern, "items", []) or []))
        pattern_sizes[str(label)] = size
        pattern_sizes[f"pattern:{label}"] = size

    pair_sizes: dict[str, int] = {}
    for item in getattr(clf, "augmented_pair_transforms_", []) or []:
        pair_name = str(item.get("name", ""))
        inputs = [str(value) for value in (item.get("inputs") or []) if str(value)]
        size = max(1, len(dict.fromkeys(inputs)))
        pair_sizes[pair_name] = size
        pair_sizes[f"augmented_pair:{pair_name}"] = size

    if name in pattern_sizes:
        return int(pattern_sizes[name])
    if name.startswith("pattern:"):
        return int(pattern_sizes.get(name, 1))
    if name in pair_sizes:
        return int(pair_sizes[name])
    if name.startswith("augmented_pair:"):
        return int(pair_sizes.get(name, 1))
    return 1


def _tree_leaf_depth_sum(tree) -> int:
    total = 0
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left < 0 and right < 0:
            total += depth
            continue
        if right >= 0:
            stack.append((right, depth + 1))
        if left >= 0:
            stack.append((left, depth + 1))
    return total


def _tree_depths(tree) -> np.ndarray:
    depths = np.zeros(tree.node_count, dtype=np.int64)
    stack = [0]
    while stack:
        node = stack.pop()
        for child in (int(tree.children_left[node]), int(tree.children_right[node])):
            if child >= 0:
                depths[child] = depths[node] + 1
                stack.append(child)
    return depths


def test_hugiml_linear_three_level_report_and_instance_counts():
    X, y = _interaction_dataset()
    kwargs = _base_kwargs()
    kwargs.update(feature_mode="original_plus_patterns")
    clf = HUGIMLClassifierNative(**kwargs).fit(X, y)

    report = clf.get_complexity_report(X=X)
    assert report["branch"] == "linear_or_empty_rpte"
    assert report["model_units"]["unit"] == "active_downstream_terms"
    assert report["model_inspection_units"]["unit"] == "expanded_all_active_sources"

    names = list(clf._get_downstream_feature_names())
    coefficients = np.atleast_2d(clf.model_.named_steps["clf"].coef_)
    active = np.any(np.isfinite(coefficients) & (np.abs(coefficients) > 1e-12), axis=0)
    expected_model_units = int(active.sum())
    expected_model_inspection_units = sum(
        _hug_feature_inspection_size(clf, str(name))
        for name, keep in zip(names, active)
        if keep
    )

    downstream = clf._apply_strict_topk_budget_transform(
        clf._make_downstream_features(
            X,
            clf._build_test_hup(clf._prebin_for_predict(X)),
            fit=False,
        )
    )
    expected_instance = np.zeros(len(X), dtype=np.int64)
    for index, (name, keep) in enumerate(zip(names, active)):
        if not keep:
            continue
        weight = _hug_feature_inspection_size(clf, str(name))
        if clf._downstream_feature_type(str(name)) == "original":
            expected_instance += weight
        else:
            present = np.abs(np.asarray(downstream[:, index]).reshape(-1)) > 1e-12
            expected_instance += present * weight

    actual_instance = clf.get_instance_inspection_units(X)
    np.testing.assert_array_equal(actual_instance, expected_instance)
    assert report["model_units"]["value"] == expected_model_units
    assert report["model_inspection_units"]["value"] == expected_model_inspection_units
    assert report["instance_inspection_units"]["mean"] == pytest.approx(
        float(expected_instance.mean())
    )
    assert report["instance_inspection_units"]["ci_lower"] <= expected_instance.mean()
    assert report["instance_inspection_units"]["ci_upper"] >= expected_instance.mean()
    assert clf.get_complexity() == expected_model_inspection_units
    assert clf.get_complexity("model units") == expected_model_units
    assert clf.get_complexity("model inspection units") == expected_model_inspection_units
    assert clf.get_complexity("instance inspection units", X=X) == pytest.approx(
        float(expected_instance.mean())
    )


def test_hugiml_rpte_counts_one_reached_leaf_per_tree_plus_direct_terms():
    X, y = _interaction_dataset(n=520, seed=1031)
    rpte = LeafWiseBoundedLookaheadRPTEFeatureLR(
        leaf_config="3xD",
        depth=3,
        n_estimators=3,
        min_samples_leaf=10,
        enable_lookahead=False,
    )
    kwargs = _base_kwargs()
    kwargs.update(feature_mode="original_plus_patterns", base_estimator=rpte)
    clf = HUGIMLClassifierNative(**kwargs).fit(X, y)

    rows = clf.rpte_rule_table()
    tree_rows = [
        row
        for row in rows
        if row.get("tree_index") is not None
        and row.get("backend") in {"bounded_lookahead", "sequential_default"}
        and _is_active_row(row)
    ]
    direct_rows = [
        row
        for row in rows
        if row.get("tree_index") is None
        and row.get("backend") in {"direct_hugiml_feature", "raw_hugiml_features"}
        and row.get("source_selection_status") in {None, "not_selected_in_tree_split"}
        and _is_active_row(row)
    ]
    assert tree_rows

    report = clf.get_complexity_report(X=X)
    expected_model_units = len(tree_rows) + len(direct_rows)
    assert report["branch"] == "rpte"
    assert report["model_units"]["value"] == expected_model_units
    assert report["model_inspection_units"]["value"] == (
        report["model_inspection_units"]["expanded_path_units"]
        + report["model_inspection_units"]["direct_source_units"]
    )

    estimator = clf.model_.named_steps["clf"]
    downstream = clf._apply_strict_topk_budget_transform(
        clf._make_downstream_features(
            X,
            clf._build_test_hup(clf._prebin_for_predict(X)),
            fit=False,
        )
    )
    leaf_matrix = estimator.fe_.transform_leaves(downstream)
    active_leaf_rows = {
        (int(row["tree_index"]), int(row["leaf_index"])): row for row in tree_rows
    }
    feature_extractor = estimator.fe_
    default_extractor = getattr(feature_extractor, "_default_fe", None)
    leaf_vocabularies = (
        default_extractor.leaf_vocab_
        if default_extractor is not None
        else feature_extractor.tree_leaf_ids_
    )
    keys = [
        (tree_index, int(leaf_id))
        for tree_index, vocab in enumerate(leaf_vocabularies)
        for leaf_id in vocab
    ]
    weights = np.zeros(len(keys), dtype=np.int64)
    for column, key in enumerate(keys):
        row = active_leaf_rows.get(key)
        if row is None:
            continue
        weights[column] = sum(
            _hug_feature_inspection_size(
                clf,
                str(condition.get("downstream_feature") or condition.get("feature") or ""),
            )
            for condition in row.get("conditions") or []
        )
    direct_units = sum(
        _hug_feature_inspection_size(clf, str(row.get("downstream_feature") or ""))
        for row in direct_rows
    )
    expected_instance = np.asarray(leaf_matrix @ weights).reshape(-1).astype(np.int64)
    expected_instance += direct_units

    actual_instance = get_instance_inspection_units(clf, X)
    np.testing.assert_array_equal(actual_instance, expected_instance)
    assert report["instance_inspection_units"]["mean"] == pytest.approx(
        float(expected_instance.mean())
    )
    assert report["instance_inspection_units"]["mean"] < report["model_inspection_units"]["value"]


def test_mode_contract_requires_clean_public_names():
    X, y = _interaction_dataset(n=240, seed=22)
    clf = HUGIMLClassifierNative(**_base_kwargs()).fit(X, y)

    for invalid_mode in (
        "unknown",
        "model_units",
        "coarse",
        "granular",
    ):
        with pytest.raises(
            ValueError,
            match="model units.*model inspection units.*instance inspection units",
        ):
            clf.get_complexity(invalid_mode)

    with pytest.raises(ValueError, match="X is required"):
        clf.get_complexity("instance inspection units")
    with pytest.raises(ValueError, match="non-negative"):
        clf.get_complexity_report(coefficient_tolerance=-1.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        clf.get_complexity_report(X=X, confidence_level=1.0)


def test_sklearn_linear_tree_forest_and_pipeline_use_uniform_interface():
    X, y = _interaction_dataset(n=320, seed=45)

    linear = LogisticRegression(max_iter=500).fit(X, y)
    active_linear = int(np.any(np.abs(np.atleast_2d(linear.coef_)) > 1e-12, axis=0).sum())
    linear_values = get_instance_inspection_units(linear, X)
    np.testing.assert_array_equal(linear_values, np.full(len(X), active_linear))
    assert get_complexity(linear, "model units") == active_linear
    assert get_complexity(linear) == active_linear

    pipeline = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500))]
    ).fit(X, y)
    pipeline_values = get_instance_inspection_units(pipeline, X)
    assert pipeline_values is not None
    assert pipeline_values.shape == (len(X),)

    forest = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=7).fit(X, y)
    expected_leaves = int(sum(tree.tree_.n_leaves for tree in forest.estimators_))
    expected_model_inspection = int(
        sum(_tree_leaf_depth_sum(tree.tree_) for tree in forest.estimators_)
    )
    expected_instance = np.zeros(len(X), dtype=np.int64)
    for tree in forest.estimators_:
        expected_instance += _tree_depths(tree.tree_)[tree.apply(X)]

    report = get_complexity_report(forest, X=X)
    assert report is not None
    assert get_complexity(forest, "model units") == expected_leaves
    assert get_complexity(forest) == expected_model_inspection
    assert get_complexity(forest, "model inspection units") == expected_model_inspection
    assert report["model_inspection_units"]["unit"] == "all_root_to_leaf_conditions"
    np.testing.assert_array_equal(get_instance_inspection_units(forest, X), expected_instance)


class _FakeXGBooster:
    def get_dump(self, dump_format: str):
        assert dump_format == "json"
        return [
            json.dumps(
                {
                    "nodeid": 0,
                    "split": "x0",
                    "children": [
                        {"nodeid": 1, "leaf": 0.7},
                        {"nodeid": 2, "leaf": -0.4},
                    ],
                }
            ),
            json.dumps({"nodeid": 0, "leaf": 0.0}),
        ]


class FakeXGBClassifier:
    def get_booster(self):
        return _FakeXGBooster()

    def apply(self, X):
        return np.column_stack([np.where(np.asarray(X)[:, 0] <= 0, 1, 2), np.zeros(len(X))])


class _FakeLightGBMBooster:
    def dump_model(self):
        return {
            "tree_info": [
                {
                    "tree_structure": {
                        "split_index": 0,
                        "left_child": {"leaf_index": 0, "leaf_value": 0.3},
                        "right_child": {
                            "split_index": 1,
                            "left_child": {"leaf_index": 1, "leaf_value": -0.2},
                            "right_child": {"leaf_index": 2, "leaf_value": 0.0},
                        },
                    }
                }
            ]
        }


class FakeLGBMClassifier:
    booster_ = _FakeLightGBMBooster()

    def predict(self, X, pred_leaf=False):
        assert pred_leaf
        values = np.asarray(X)
        leaves = np.where(values[:, 0] <= 0, 0, np.where(values[:, 1] <= 0, 1, 2))
        return leaves.reshape(-1, 1)


class FakeExplainableBoostingClassifier:
    term_scores_ = [
        np.asarray([0.0, 1.0, -2.0]),
        np.asarray([[0.0, 0.0], [3.0, 0.0]]),
        np.asarray([0.0, 0.0]),
    ]

    def eval_terms(self, X):
        n = len(X)
        return np.column_stack(
            [np.resize([0.0, 1.0], n), np.resize([3.0, 0.0], n), np.zeros(n)]
        )


class FakeRuleFitClassifier:
    def get_rules(self):
        return pd.DataFrame(
            [
                {"rule": "x0", "type": "linear", "coef": 0.8},
                {"rule": "x0 > 0 & x1 <= 1", "type": "rule", "coef": -0.6},
                {"rule": "x2 > 0", "type": "rule", "coef": 0.0},
            ]
        )


def test_optional_model_shapes_include_instance_counts_without_optional_imports():
    X = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]])

    xgb = FakeXGBClassifier()
    assert get_complexity(xgb, "model units") == 2
    assert get_complexity(xgb) == 2
    assert get_complexity(xgb, "model inspection units") == 2
    np.testing.assert_array_equal(get_instance_inspection_units(xgb, X), [1, 1, 1])

    lgb = FakeLGBMClassifier()
    assert get_complexity(lgb, "model units") == 2
    assert get_complexity(lgb) == 3
    assert get_complexity(lgb, "model inspection units") == 3
    np.testing.assert_array_equal(get_instance_inspection_units(lgb, X), [1, 2, 0])

    ebm = FakeExplainableBoostingClassifier()
    assert get_complexity(ebm, "model units") == 2
    assert get_complexity(ebm) == 4
    assert get_complexity(ebm, "model inspection units") == 4
    np.testing.assert_array_equal(get_instance_inspection_units(ebm, X), [2, 1, 2])

    rulefit = FakeRuleFitClassifier()
    assert get_complexity(rulefit, "model units") == 2
    assert get_complexity(rulefit) == 3
    assert get_complexity(rulefit, "model inspection units") == 3
    np.testing.assert_array_equal(get_instance_inspection_units(rulefit, X), [1, 3, 1])



def test_ebm_interaction_arity_expands_inspection_but_not_model_units():
    class EBMWithHigherOrderTerms:
        term_features_ = [(0,), (0, 1), (0, 1, 2)]
        term_scores_ = [
            np.asarray([0.0, 1.0]),
            np.asarray([[0.0, 2.0], [0.0, 0.0]]),
            np.asarray([[[0.0, 0.0], [0.0, 3.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        ]

        def eval_terms(self, X):
            n = len(X)
            return np.tile(np.asarray([[1.0, 2.0, 3.0]]), (n, 1))

    model = EBMWithHigherOrderTerms()
    X = np.zeros((2, 3), dtype=float)
    report = get_complexity_report(model, X=X)

    assert report is not None
    assert report["model_units"]["value"] == 3
    assert report["model_inspection_units"]["value"] == 6
    assert report["model_inspection_units"]["active_cells_by_term"] == [1, 1, 1]
    assert report["model_inspection_units"]["term_arities"] == [1, 2, 3]
    assert report["model_inspection_units"]["inspection_units_by_term"] == [1, 2, 3]
    np.testing.assert_array_equal(get_instance_inspection_units(model, X), [6, 6])


def test_unsupported_model_returns_none():
    unsupported = object()
    assert get_complexity(unsupported) is None
    assert get_complexity(unsupported, "model units") is None
    assert get_instance_inspection_units(unsupported, np.zeros((2, 1))) is None
    assert get_complexity_report(unsupported) is None


class FakeUnavailableXGBClassifier:
    def get_booster(self):
        raise ImportError("optional runtime is unavailable")


def test_optional_runtime_failures_are_graceful():
    assert get_complexity(FakeUnavailableXGBClassifier()) is None
    assert get_instance_inspection_units(
        FakeUnavailableXGBClassifier(), np.zeros((2, 1))
    ) is None


def test_every_supported_report_uses_the_same_three_keys():
    X = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    models = [
        FakeXGBClassifier(),
        FakeLGBMClassifier(),
        FakeExplainableBoostingClassifier(),
        FakeRuleFitClassifier(),
    ]
    for model in models:
        report = get_complexity_report(model, X=X)
        assert report is not None
        assert set(report) >= {
            "model_type",
            "model_units",
            "model_inspection_units",
            "instance_inspection_units",
        }
        assert get_complexity(model) == report["model_inspection_units"]["value"]
        assert get_complexity(model, "model units") == report["model_units"]["value"]
        assert get_complexity(model, "model inspection units") == (
            report["model_inspection_units"]["value"]
        )
        assert get_complexity(model, "instance inspection units", X=X) == pytest.approx(
            report["instance_inspection_units"]["mean"]
        )

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from sklearn.multiclass import OneVsRestClassifier

import hugiml.serialization as serialization
from hugiml import HUGIMLClassifierNative
from hugiml.rpte_bounded_lookahead_leafwise import (
    LeafWiseBoundedLookaheadRPTEFeatureExtractor,
    LeafWiseBoundedLookaheadRPTEFeatureLR,
)


def _equals_atom(feature: str, value: str = "1") -> dict[str, object]:
    return {"feature": feature, "operator": "equals", "value": value}


def _fit_exact_pattern_alias(
    order: int,
) -> tuple[LeafWiseBoundedLookaheadRPTEFeatureLR, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(100 + order)

    if order == 1:
        first = rng.integers(0, 2, size=4000)
        X = np.column_stack([first, first]).astype(float)
        y = first
        names = ["pattern:A=1", "pattern:A=1 copy"]
        provenance = {
            name: {
                "order": 1,
                "raw_features": ["A"],
                "atoms": [_equals_atom("A")],
            }
            for name in names
        }
        random_state = 2
        depth = 1
    elif order == 2:
        raw = rng.integers(0, 2, size=(5000, 2))
        first, second = raw.T
        pair = first * second
        row_index = np.arange(raw.shape[0])
        y = np.where(first == 0, 0, np.where(second == 0, row_index % 2, 1)).astype(int)
        X = np.column_stack([first, second, pair]).astype(float)
        names = ["pattern:A=1", "pattern:B=1", "pattern:A=1, B=1"]
        provenance = {
            names[0]: {"order": 1, "raw_features": ["A"], "atoms": [_equals_atom("A")]},
            names[1]: {"order": 1, "raw_features": ["B"], "atoms": [_equals_atom("B")]},
            names[2]: {
                "order": 2,
                "raw_features": ["A", "B"],
                "atoms": [_equals_atom("A"), _equals_atom("B")],
            },
        }
        random_state = 2
        depth = 2
    else:
        raw = rng.integers(0, 2, size=(8000, 3))
        first, second, third = raw.T
        pair = first * second
        triplet = pair * third
        row_index = np.arange(raw.shape[0])
        y = np.where(pair == 0, 0, np.where(third == 0, row_index % 2, 1)).astype(int)
        X = np.column_stack([pair, third, triplet]).astype(float)
        names = [
            "pattern:A=1, B=1",
            "pattern:C=1",
            "pattern:A=1, B=1, C=1",
        ]
        provenance = {
            names[0]: {
                "order": 2,
                "raw_features": ["A", "B"],
                "atoms": [_equals_atom("A"), _equals_atom("B")],
            },
            names[1]: {"order": 1, "raw_features": ["C"], "atoms": [_equals_atom("C")]},
            names[2]: {
                "order": 3,
                "raw_features": ["A", "B", "C"],
                "atoms": [_equals_atom("A"), _equals_atom("B"), _equals_atom("C")],
            },
        }
        random_state = 0
        depth = 2

    model = LeafWiseBoundedLookaheadRPTEFeatureLR(
        n_estimators=1,
        depth=depth,
        min_samples_leaf=5,
        enable_lookahead=False,
        lr_C=10.0,
        random_state=random_state,
        hugiml_feature_names=names,
        hugiml_pattern_provenance=provenance,
    ).fit(X, y)
    unused_patterns = sorted(
        set(range(X.shape[1]))
        - set(model.tree_used_input_indices_.tolist())
        - set(model.direct_input_indices_.tolist())
    )
    assert len(unused_patterns) == 1
    return model, X, y, unused_patterns[0]


@pytest.mark.parametrize("order", [1, 2, 3])
def test_exact_leaf_pattern_aliases_are_suppressed_across_pattern_orders(order: int):
    model, X, y, pattern_index = _fit_exact_pattern_alias(order)

    if order > 2:
        assert pattern_index in set(model.direct_only_pattern_indices_.tolist())
        assert pattern_index not in set(model.tree_eligible_input_indices_.tolist())
    else:
        assert pattern_index not in set(model.direct_only_pattern_indices_.tolist())
        assert pattern_index in set(model.tree_eligible_input_indices_.tolist())
    assert pattern_index not in set(model.tree_used_input_indices_.tolist())
    assert pattern_index in set(model.candidate_direct_input_indices_.tolist())
    assert pattern_index not in set(model.direct_input_indices_.tolist())
    assert pattern_index in set(model.suppressed_direct_alias_indices_.tolist())

    aliases = model.representation_alias_table()
    assert len(aliases) == 1
    assert aliases[0]["downstream_feature_index"] == pattern_index
    assert aliases[0]["raw_sources"] == [chr(ord("A") + idx) for idx in range(order)]
    assert aliases[0]["canonical_role"] == "rpte_leaf"

    leaves = model.fe_.transform_leaves(X).toarray()
    leaf_index = aliases[0]["leaf_feature_index"]
    np.testing.assert_array_equal(leaves[:, leaf_index], X[:, pattern_index])
    assert model.n_final_lr_features_ == model.n_leaf_features_ + model.direct_input_indices_.size
    minimum_accuracy = 0.85 if order == 2 else (0.90 if order == 3 else 0.99)
    assert np.mean(model.predict(X) == y) >= minimum_accuracy

    direct_rule_names = {
        row.get("downstream_feature")
        for row in model.unified_rule_table()
        if row.get("term_role") == "direct_source_term"
    }
    assert model.hugiml_feature_names[pattern_index] not in direct_rule_names


def test_pair_patterns_and_augmented_pairs_keep_their_tree_roles():
    names = [
        "orig:A",
        "orig:B",
        "pattern:A=1, B=1",
        "augmented_pair:product__A__B",
    ]
    provenance = {
        names[2]: {"raw_features": ["A", "B"], "atoms": []},
    }
    augmented = [
        {
            "name": "product__A__B",
            "inputs": ["A", "B"],
            "operation": "product",
        }
    ]
    extractor = LeafWiseBoundedLookaheadRPTEFeatureExtractor(enable_lookahead=True)
    extractor.set_hugiml_feature_metadata(names, augmented, provenance, {})
    metadata = extractor._metadata(len(names))
    roots = metadata[2]
    tree_eligible = metadata[8]
    direct_only_patterns = metadata[9]

    assert [root[0] for root in roots] == [2, 3]
    assert [root[3] for root in roots] == ["pattern", "product"]
    assert tree_eligible == [0, 1, 2, 3]
    assert direct_only_patterns == []


def test_higher_order_patterns_only_mode_uses_direct_source_fallback():
    rng = np.random.default_rng(31)
    first = rng.integers(0, 2, size=500)
    second = rng.integers(0, 2, size=500)
    X = np.column_stack([first, second]).astype(float)
    y = first
    names = ["pattern:A=1, B=1, C=1", "pattern:D=1, E=1, F=1"]
    provenance = {
        names[0]: {"raw_features": ["A", "B", "C"], "atoms": []},
        names[1]: {"raw_features": ["D", "E", "F"], "atoms": []},
    }

    model = LeafWiseBoundedLookaheadRPTEFeatureLR(
        n_estimators=1,
        depth=2,
        enable_lookahead=False,
        random_state=31,
        hugiml_feature_names=names,
        hugiml_pattern_provenance=provenance,
    ).fit(X, y)

    assert model.fe_._raw_feature_fallback_ is True
    assert model.tree_eligible_input_indices_.size == 0
    assert model.direct_only_pattern_indices_.tolist() == [0, 1]
    assert model.direct_input_indices_.size == 0
    assert model.representation_alias_table() == []
    assert model.n_final_lr_features_ == X.shape[1]
    assert np.mean(model.predict(X) == y) >= 0.99


def test_alias_layout_roundtrips_through_structured_serialization():
    model, X, _y, _pattern_index = _fit_exact_pattern_alias(2)
    config, arrays = serialization._serialize_estimator(model)
    restored = serialization._deserialize_estimator(config, arrays)

    assert restored.representation_alias_table() == model.representation_alias_table()
    np.testing.assert_array_equal(
        restored.direct_input_indices_, model.direct_input_indices_
    )
    np.testing.assert_allclose(
        restored.predict_proba(X),
        model.predict_proba(X),
        rtol=1e-7,
        atol=2e-8,
    )


def test_higher_order_direct_only_role_survives_one_vs_rest_cloning():
    rng = np.random.default_rng(44)
    raw = rng.integers(0, 2, size=(900, 3))
    triplet = raw[:, 0] * raw[:, 1] * raw[:, 2]
    X = np.column_stack([raw, triplet]).astype(float)
    y = raw[:, 0] + raw[:, 1]
    names = ["orig:A", "orig:B", "orig:C", "pattern:A=1, B=1, C=1"]
    provenance = {
        names[-1]: {"raw_features": ["A", "B", "C"], "atoms": []},
    }

    model = OneVsRestClassifier(
        LeafWiseBoundedLookaheadRPTEFeatureLR(
            n_estimators=1,
            depth=2,
            enable_lookahead=False,
            random_state=44,
            hugiml_feature_names=names,
            hugiml_pattern_provenance=provenance,
        ),
        n_jobs=1,
    ).fit(X, y)

    for submodel in model.estimators_:
        assert submodel.direct_only_pattern_indices_.tolist() == [3]
        assert 3 not in set(submodel.tree_used_input_indices_.tolist())


def test_pattern_without_order_metadata_preserves_existing_tree_eligibility():
    names = ["orig:A", "pattern:A=1, B=1, C=1"]
    extractor = LeafWiseBoundedLookaheadRPTEFeatureExtractor(enable_lookahead=False)
    extractor.set_hugiml_feature_metadata(names, [], {}, {})

    metadata = extractor._metadata(len(names))

    assert metadata[8] == [0, 1]
    assert metadata[9] == []
    assert metadata[10] == [1]


def test_classifier_exposes_rpte_alias_records():
    model, _X, _y, _pattern_index = _fit_exact_pattern_alias(2)
    classifier = object.__new__(HUGIMLClassifierNative)
    classifier.model_ = SimpleNamespace(named_steps={"clf": model})

    assert classifier.rpte_representation_aliases() == model.representation_alias_table()



def test_training_support_match_without_structural_equivalence_is_retained():
    values = np.tile(np.asarray([0.0, 0.1, 0.2, 0.3, 0.8, 0.9, 1.0, 1.1]), 50)
    indicator = (values > 0.6).astype(float)
    X = np.column_stack([values, indicator])
    name = "pattern:A=[0.6, inf)"
    metadata = {
        name: {
            "order": 1,
            "raw_features": ["A"],
            "atoms": [
                {
                    "feature": "A",
                    "operator": "interval",
                    "lower": 0.6,
                    "upper": float("inf"),
                    "lower_inclusive": False,
                    "upper_inclusive": False,
                }
            ],
        }
    }

    model = LeafWiseBoundedLookaheadRPTEFeatureLR(
        n_estimators=1,
        depth=1,
        min_samples_leaf=5,
        enable_lookahead=False,
        random_state=2,
        hugiml_feature_names=["orig:A", name],
        hugiml_pattern_provenance=metadata,
    ).fit(X, indicator.astype(int))

    assert model.tree_used_input_indices_.tolist() == [0]
    assert model.direct_input_indices_.tolist() == [1]
    assert model.representation_alias_table() == []
    test_rows = np.asarray([[0.57, 0.0], [0.65, 1.0]])
    leaf_rows = model.fe_.transform_leaves(test_rows).toarray()
    assert not np.array_equal(leaf_rows[:, 1], test_rows[:, 1])


def test_explicit_pattern_order_controls_tree_eligibility():
    name = "pattern:opaque_high_order_term"
    extractor = LeafWiseBoundedLookaheadRPTEFeatureExtractor(enable_lookahead=False)
    extractor.set_hugiml_feature_metadata(
        [name],
        [],
        {
            name: {
                "order": 4,
                "raw_features": ["A"],
                "atoms": [_equals_atom("A")],
            }
        },
        {},
    )

    metadata = extractor._metadata(1)

    assert metadata[8] == []
    assert metadata[9] == [0]


def test_bounded_backend_excludes_higher_order_pattern_columns():
    rng = np.random.default_rng(71)
    raw = rng.integers(0, 2, size=(4000, 3))
    first, second, third = raw.T
    pair = first * second
    triplet = pair * third
    X = np.column_stack([first, second, third, pair, triplet]).astype(float)
    names = [
        "orig:A",
        "orig:B",
        "orig:C",
        "augmented_pair:product__A__B",
        "pattern:A=1, B=1, C=1",
    ]
    augmented = [{"name": "product__A__B", "inputs": ["A", "B"], "operation": "product"}]
    metadata = {
        names[-1]: {
            "order": 3,
            "raw_features": ["A", "B", "C"],
            "atoms": [_equals_atom("A"), _equals_atom("B"), _equals_atom("C")],
        }
    }

    model = LeafWiseBoundedLookaheadRPTEFeatureLR(
        n_estimators=1,
        depth=2,
        min_samples_leaf=10,
        enable_lookahead=True,
        lookahead_ops=("product",),
        lookahead_min_probe_ig=0.0,
        lookahead_min_increment=0.0,
        min_weighted_probe_gain=0.0,
        use_statistical_acceptance=False,
        random_state=71,
        hugiml_feature_names=names,
        hugiml_augmented_catalog=augmented,
        hugiml_pattern_provenance=metadata,
    ).fit(X, triplet)

    assert model.fe_._default_fe is None
    assert model.direct_only_pattern_indices_.tolist() == [4]
    assert 4 not in set(model.tree_used_input_indices_.tolist())


def test_sparse_pattern_matrix_uses_the_same_canonical_layout():
    model, X, _y, pattern_index = _fit_exact_pattern_alias(3)
    sparse_model = LeafWiseBoundedLookaheadRPTEFeatureLR(**model.get_params(deep=False)).fit(
        sparse.csr_matrix(X), model.predict(X)
    )

    assert pattern_index in set(sparse_model.suppressed_direct_alias_indices_.tolist())
    np.testing.assert_allclose(
        sparse_model.predict_proba(sparse.csr_matrix(X)),
        sparse_model.predict_proba(X),
    )


def test_binary_one_vs_rest_alias_rows_use_the_positive_class_label():
    model, _X, _y, _pattern_index = _fit_exact_pattern_alias(2)
    wrapper = SimpleNamespace(classes_=np.asarray(["negative", "positive"]), estimators_=[model])
    classifier = object.__new__(HUGIMLClassifierNative)
    classifier.model_ = SimpleNamespace(named_steps={"clf": wrapper})

    rows = classifier.rpte_representation_aliases()

    assert rows
    assert {row["class"] for row in rows} == {"positive"}



def test_bounded_backend_canonicalizes_pattern_conjunction_alias():
    rng = np.random.default_rng(500)
    raw = rng.integers(0, 2, size=(6000, 4))
    first, second, third, fourth = raw.T
    pair = first * second
    extra_pair = third * fourth
    row_index = np.arange(raw.shape[0])
    y = np.where(first == 0, 0, np.where(second == 0, row_index % 2, 1)).astype(int)
    X = np.column_stack([first, second, pair, third, fourth, extra_pair]).astype(float)
    names = [
        "pattern:A=1",
        "pattern:B=1",
        "pattern:A=1, B=1",
        "orig:D",
        "orig:E",
        "augmented_pair:product__D__E",
    ]
    metadata = {
        names[0]: {"order": 1, "raw_features": ["A"], "atoms": [_equals_atom("A")]},
        names[1]: {"order": 1, "raw_features": ["B"], "atoms": [_equals_atom("B")]},
        names[2]: {
            "order": 2,
            "raw_features": ["A", "B"],
            "atoms": [_equals_atom("A"), _equals_atom("B")],
        },
    }
    augmented = [{"name": "product__D__E", "inputs": ["D", "E"], "operation": "product"}]

    model = LeafWiseBoundedLookaheadRPTEFeatureLR(
        n_estimators=1,
        depth=2,
        min_samples_leaf=10,
        enable_lookahead=True,
        lookahead_ops=("product",),
        lookahead_min_probe_ig=0.0,
        lookahead_min_increment=0.0,
        min_weighted_probe_gain=0.0,
        use_statistical_acceptance=False,
        random_state=0,
        hugiml_feature_names=names,
        hugiml_augmented_catalog=augmented,
        hugiml_pattern_provenance=metadata,
    ).fit(X, y)

    assert model.fe_._default_fe is None
    assert model.suppressed_direct_alias_indices_.tolist() == [2]
    assert model.representation_alias_table()[0]["backend"] == "bounded_lookahead"

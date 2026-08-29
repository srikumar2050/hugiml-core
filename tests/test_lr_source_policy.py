from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.datasets import make_classification

from hugiml import HUGIMLClassifierNative
from hugiml._classifier_support import _select_lr_source_policy_mask
from hugiml.rpte_bounded_lookahead_leafwise import LeafWiseBoundedLookaheadRPTEFeatureLR


def test_lr_source_policy_default_and_clone_safe():
    clf = HUGIMLClassifierNative()
    assert clf.lr_source_policy == "standard"
    assert clf.get_params()["lr_source_policy"] == "standard"
    assert clone(clf).lr_source_policy == "standard"


def test_lr_source_policy_validation():
    clf = HUGIMLClassifierNative(lr_source_policy="invalid")
    with pytest.raises(Exception, match="lr_source_policy"):
        clf._validate_params()


def test_patterns_only_main_effect_is_effectively_strict():
    clf = HUGIMLClassifierNative(feature_mode="patterns_only", lr_source_policy="main_effect")
    assert clf._effective_lr_source_policy() == "strict"


def test_source_policy_selector_preserves_original_group_and_bounds_contexts():
    names = [
        "orig:occupation_manager",
        "orig:occupation_teacher",
        "orig:income",
        "pattern:occupation=manager, income=high",
        "pattern:income=high, age=old",
        "augmented_pair:age_x_balance",
    ]
    sources = [
        frozenset({"occupation"}),
        frozenset({"occupation"}),
        frozenset({"income"}),
        frozenset({"occupation", "income"}),
        frozenset({"income", "age"}),
        frozenset({"age", "balance"}),
    ]

    standard, _ = _select_lr_source_policy_mask(names, sources, "standard")
    assert standard.tolist() == [True] * len(names)

    main, _ = _select_lr_source_policy_mask(names, sources, "main_effect")
    # Both surviving occupation dummies are one retained main-effect group.
    assert main[:3].tolist() == [True, True, True]
    # First context claims occupation+income; later overlapping contexts are removed.
    assert main[3:].tolist() == [True, False, True]

    strict, _ = _select_lr_source_policy_mask(names, sources, "strict")
    # Contexts claim sources first. occupation/income originals then disappear as groups.
    assert strict.tolist() == [False, False, False, True, False, True]


def test_categorical_dummy_lineage_maps_every_surviving_dummy_to_raw_parent():
    X = pd.DataFrame(
        {
            "occupation": ["manager", "teacher", "manager", None, "clerical"],
            "age": [30.0, 42.0, 37.0, 51.0, 46.0],
        }
    )
    clf = HUGIMLClassifierNative(feature_mode="original_plus_patterns")
    clf.feature_names_in_ = list(X.columns)
    clf.cat_cols_mask_ = np.asarray([True, False], dtype=bool)
    clf._prepare_original_features_for_downstream(X, fit=True)

    source_map = clf._original_downstream_source_map_
    assert source_map["age"] == "age"
    dummy_names = list(clf._original_dummy_columns_)
    assert dummy_names
    assert {source_map[name] for name in dummy_names} == {"occupation"}


def _manual_rpte(policy: str):
    model = LeafWiseBoundedLookaheadRPTEFeatureLR(
        enable_lookahead=False,
        lr_source_policy=policy,
        hugiml_feature_names=[
            "orig:a",
            "pattern:a=high, b=high",
            "orig:b",
            "pattern:b=high, c=high",
        ],
        hugiml_feature_source_sets=[
            frozenset({"a"}),
            frozenset({"a", "b"}),
            frozenset({"b"}),
            frozenset({"b", "c"}),
        ],
    )
    model.fe_ = SimpleNamespace(
        _raw_feature_fallback_=False,
        owner_by_column_names_={
            0: frozenset({"a"}),
            1: frozenset({"a", "b"}),
            2: frozenset({"b"}),
            3: frozenset({"b", "c"}),
        },
        reserved_raw_features_=[],
        tree_eligible_input_indices_=np.asarray([0, 1, 2, 3], dtype=np.int64),
        direct_only_pattern_indices_=np.asarray([], dtype=np.int64),
        pattern_input_indices_=np.asarray([1, 3], dtype=np.int64),
        higher_order_patterns_direct_only_=False,
    )
    model._tree_used_input_columns = lambda _fe, _n: {0}
    model._find_leaf_pattern_aliases = lambda _X, _L, _D: []
    X = sparse.csr_matrix(np.arange(16, dtype=float).reshape(4, 4))
    leaves = sparse.csr_matrix(np.asarray([[1.0], [0.0], [1.0], [0.0]]))
    model._fit_final_feature_layout(X, leaves)
    return model


def test_rpte_main_effect_retains_tree_used_original_as_direct_main_effect():
    standard = _manual_rpte("standard")
    main = _manual_rpte("main_effect")
    strict = _manual_rpte("strict")

    assert standard.direct_input_indices_.tolist() == [1, 2, 3]
    # orig:a is tree-used but intentionally survives as its main-effect channel.
    assert main.direct_input_indices_.tolist() == [0, 2, 3]
    # Strict gives contextual terms first claim; only b/c pattern survives here.
    assert strict.direct_input_indices_.tolist() == [3]


def _small_realistic_frame():
    X, y = make_classification(
        n_samples=220,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=7,
    )
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    return frame, y


@pytest.mark.parametrize(
    ("feature_mode", "policy", "effective"),
    [
        ("patterns_only", "standard", "standard"),
        ("patterns_only", "main_effect", "strict"),
        ("patterns_only", "strict", "strict"),
        ("original_plus_patterns", "main_effect", "main_effect"),
        ("original_plus_patterns", "strict", "strict"),
        ("original_plus_interactions", "main_effect", "main_effect"),
        ("original_plus_interactions", "strict", "strict"),
    ],
)
def test_lr_source_policy_all_feature_modes_fit_transform(feature_mode, policy, effective):
    X, y = _small_realistic_frame()
    clf = HUGIMLClassifierNative(
        B=5,
        L=2,
        G=1e-3,
        topK=20,
        adaptive_binning=True,
        augmented_pair_transforms=False,
        feature_mode=feature_mode,
        lr_source_policy=policy,
    ).fit(X, y)

    audit = clf.get_downstream_redundancy_audit()
    assert audit["lr_source_policy"] == policy
    assert audit["lr_source_policy_effective"] == effective
    Z = clf.transform(X.iloc[:20])
    assert Z.shape[1] == clf.model_.named_steps["clf"].n_features_in_
    assert Z.shape[1] == len(clf.get_downstream_features())

    names = clf.get_downstream_features()
    sources = clf._get_downstream_feature_source_sets()
    assert len(names) == len(sources)
    context_sources: set[str] = set()
    for name, source_set in zip(names, sources):
        if str(name).startswith("orig:"):
            continue
        if effective != "standard":
            assert set(source_set).isdisjoint(context_sources)
        context_sources.update(source_set)

    if effective == "strict":
        original_sources = {
            next(iter(source_set))
            for name, source_set in zip(names, sources)
            if str(name).startswith("orig:") and len(source_set) == 1
        }
        assert original_sources.isdisjoint(context_sources)

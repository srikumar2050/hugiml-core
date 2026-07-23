import numpy as np

from hugiml.rpte_bounded_lookahead_leafwise import (
    LeafWiseBoundedLookaheadRPTEFeatureExtractor,
)


def _extractor():
    return LeafWiseBoundedLookaheadRPTEFeatureExtractor(
        enable_lookahead=True,
        leaf_config="2xD",
        depth=1,
        n_estimators=1,
        min_samples_leaf=1,
        random_state=0,
    )


def test_opaque_pattern_name_is_reservable_without_provenance():
    fe = _extractor().set_hugiml_feature_metadata(
        ["pattern:opaque"], [], pattern_provenance={}
    )
    metadata = fe._metadata(1)
    assert metadata[3][0] == frozenset({"pattern:opaque"})
    assert metadata[4] == {"pattern:opaque"}
    assert metadata[5][0] == [0]
    assert metadata[8] == [0]
    assert metadata[9] == []
    assert metadata[10] == [0]


def test_opaque_pattern_name_is_reported_in_no_pair_fallback():
    X = np.array([[0.0], [0.0], [1.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    fe = _extractor().set_hugiml_feature_metadata(
        ["pattern:opaque"], [], pattern_provenance={}
    )
    leaves = fe.fit_leaves(X, y)
    assert leaves.shape[0] == X.shape[0]
    assert fe.default_backend_reason_ == "no_augmented_pairs"
    assert fe.remaining_raw_features_ == {"pattern:opaque"}


def test_known_pattern_sources_remain_raw_source_names():
    fe = _extractor().set_hugiml_feature_metadata(
        ["pattern:A=1,B=1"],
        [],
        pattern_provenance={
            "pattern:A=1,B=1": {"raw_features": ["A", "B"], "order": 2}
        },
    )
    metadata = fe._metadata(1)
    assert metadata[3][0] == frozenset({"A", "B"})
    assert metadata[4] == {"A", "B"}
    assert metadata[5][0] == [0]
    assert metadata[2] == [(0, "pattern:A=1,B=1", ("A", "B"), "pattern")]

"""Readable RPTE rule-tree API coverage."""

from __future__ import annotations

from hugiml.rpte_bounded_lookahead_leafwise import LeafWiseBoundedLookaheadRPTEFeatureLR
from hugiml.rpte_interpretability import format_rpte_rule_tree, rpte_rule_tree_sections


def _rows():
    return [
        {
            "class": 1,
            "tree_index": 0,
            "leaf_index": 0,
            "backend": "sequential_default",
            "conditions": [
                {
                    "raw_condition": "age <= 50",
                    "downstream_condition": "orig:age <= 0.15",
                    "direction": "at_or_below_threshold",
                },
                {
                    "raw_condition": "income > 30000",
                    "downstream_condition": "orig:income > -0.25",
                    "direction": "above_threshold",
                },
            ],
            "raw_sources": ["age", "income"],
            "final_logistic_coefficient": 1.25,
            "centered_tree_contribution": 0.85,
            "newton_leaf_value": 0.42,
            "support_rate": 0.20,
            "support_count": 20,
        },
        {
            "class": 1,
            "tree_index": 0,
            "leaf_index": 1,
            "backend": "sequential_default",
            "conditions": [
                {
                    "raw_condition": "age <= 50",
                    "downstream_condition": "orig:age <= 0.15",
                    "direction": "at_or_below_threshold",
                },
                {
                    "raw_condition": "income <= 30000",
                    "downstream_condition": "orig:income <= -0.25",
                    "direction": "at_or_below_threshold",
                },
            ],
            "raw_sources": ["age", "income"],
            "final_logistic_coefficient": -0.75,
            "centered_tree_contribution": -1.15,
            "newton_leaf_value": -0.31,
            "support_rate": 0.30,
            "support_count": 30,
        },
        {
            "class": 1,
            "tree_index": 0,
            "leaf_index": 2,
            "backend": "sequential_default",
            "conditions": [
                {
                    "raw_condition": "age > 50",
                    "downstream_condition": "orig:age > 0.15",
                    "direction": "above_threshold",
                }
            ],
            "raw_sources": ["age"],
            "final_logistic_coefficient": 0.10,
            "centered_tree_contribution": -0.30,
            "support_rate": 0.50,
            "support_count": 50,
        },
        {
            "class": 1,
            "tree_index": None,
            "leaf_index": 3,
            "backend": "direct_hugiml_feature",
            "term_role": "direct_source_term",
            "conditions": [{"raw_condition": "balance", "family": "original"}],
            "raw_sources": ["balance"],
            "downstream_feature": "orig:balance",
            "final_logistic_coefficient": 0.45,
            "support_rate": None,
            "support_count": None,
        },
        {
            "class": 1,
            "tree_index": None,
            "leaf_index": 4,
            "backend": "direct_hugiml_feature",
            "term_role": "direct_source_term",
            "conditions": [
                {"raw_condition": "age=[40,60)", "family": "pattern"}
            ],
            "raw_sources": ["age"],
            "downstream_feature": "pattern:age=[40,60)",
            "final_logistic_coefficient": -0.22,
            "support_rate": None,
            "support_count": None,
        },
        {
            "class": 1,
            "tree_index": None,
            "leaf_index": 5,
            "backend": "direct_hugiml_feature",
            "term_role": "direct_source_term",
            "conditions": [
                {"raw_condition": "age * income", "family": "augmented_pair"}
            ],
            "raw_sources": ["age", "income"],
            "downstream_feature": "augmented_pair:age*income",
            "final_logistic_coefficient": 0.12,
            "support_rate": None,
            "support_count": None,
        },
    ]


def test_rule_tree_merges_shared_prefixes_and_attaches_leaf_details():
    text = format_rpte_rule_tree(_rows())
    assert text.count("age <= 50") == 1
    assert "income <= 30000" in text
    assert "income > 30000" in text
    assert "LEAF 0" in text
    assert "beta=+1.25" in text
    assert "odds x3.4903" in text
    assert "support=20.0% (n=20)" in text
    assert "centered=+0.85" in text
    assert "sources=age, income" in text


def test_rule_tree_groups_direct_source_term_families():
    text = format_rpte_rule_tree(_rows())
    assert "DIRECT SOURCE TERMS" in text
    assert "Original features" in text
    assert "HUG patterns" in text
    assert "Augmented pairs" in text
    assert "balance" in text
    assert "age=[40,60)" in text
    assert "age * income" in text


def test_rule_tree_supports_compact_downstream_and_filtered_views():
    text = format_rpte_rule_tree(
        _rows(),
        condition_space="downstream",
        detail_level="compact",
        include_direct_terms=False,
        class_label=1,
        tree_index=0,
    )
    assert "orig:age <= 0.15" in text
    assert "centered=" not in text
    assert "sources=" not in text
    assert "DIRECT SOURCE TERMS" not in text


def test_rule_tree_sections_expose_coefficient_details_for_dashboard_use():
    sections = rpte_rule_tree_sections(_rows(), detail_level="compact")
    assert len(sections) == 1
    section = sections[0]
    assert section["leaf_count"] == 3
    assert section["max_depth"] == 2
    assert {item["leaf"] for item in section["coefficients"]} == {0, 1, 2}
    assert section["coefficients"][0]["coefficient"] == 1.25


def test_estimator_exposes_unified_rule_tree_companion_api():
    estimator = object.__new__(LeafWiseBoundedLookaheadRPTEFeatureLR)
    estimator.unified_rule_table = lambda feature_names=None: _rows()
    text = LeafWiseBoundedLookaheadRPTEFeatureLR.unified_rule_tree(estimator)
    assert text.startswith("Class 1 | Tree 0")
    assert "ROOT" in text
    assert "LEAF" in text
    assert "beta=" in text
    assert "Direct source terms" in text


def test_classifier_exposes_rpte_rule_tree_companion_api():
    from hugiml import HUGIMLClassifierNative

    model = object.__new__(HUGIMLClassifierNative)
    model.rpte_rule_table = lambda feature_names=None: _rows()
    text = HUGIMLClassifierNative.rpte_rule_tree(model, detail_level="compact")
    assert "Class 1 | Tree 0" in text
    assert "DIRECT SOURCE TERMS" in text


def test_rule_tree_empty_and_argument_validation():
    import pytest

    assert format_rpte_rule_tree([]) == ""
    with pytest.raises(ValueError, match="condition_space"):
        format_rpte_rule_tree(_rows(), condition_space="invalid")
    with pytest.raises(ValueError, match="detail_level"):
        format_rpte_rule_tree(_rows(), detail_level="invalid")
    with pytest.raises(ValueError, match="precision"):
        format_rpte_rule_tree(_rows(), precision=0)


def test_rule_tree_both_condition_spaces_retains_raw_and_downstream_text():
    text = format_rpte_rule_tree(
        _rows(),
        condition_space="both",
        include_direct_terms=False,
    )
    assert "age <= 50  [downstream: orig:age <= 0.15]" in text

"""Interpretability + native-acceleration regression tests. Run with:

    python3 -m pytest test_rpte_interpretability.py -v

or directly:

    python3 test_rpte_interpretability.py
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.multiclass import OneVsRestClassifier

from hugiml import HUGIMLClassifierNative
from hugiml.rpte_bounded_lookahead_leafwise import (
    LeafWiseBoundedLookaheadRPTEFeatureLR,
    _render_linear_term,
    _render_threshold,
    _simplify_threshold_path,
    aggregate_rule_table_by_raw_source,
)

MINING = dict(
    B=-1,
    adaptive_binning=True,
    L=2,
    topK=100,
    feature_mode="original_plus_patterns",
    G=0.01,
    convert_binary_to_categorical=False,
    augmented_pair_transforms=True,
    topk_budget_strict=False,
    execution_mode="production",
    n_jobs=1,
)


def _xor3_dataset(n=1400, seed=2026):
    rng = np.random.default_rng(seed)
    Xi = rng.integers(0, 2, size=(n, 32))
    y = Xi[:, 0] ^ Xi[:, 1] ^ Xi[:, 2]
    flip = rng.random(n) < 0.05
    y = np.where(flip, 1 - y, y)
    X = pd.DataFrame(Xi, columns=[f"x{i}" for i in range(32)])
    return X, y


# ---------------------------------------------------------------------------
# Native acceleration correctness (not just presence).
# ---------------------------------------------------------------------------


def test_native_extension_available():
    """RPTE has no Python fallback -- the tree grower, boosting math, and
    scoring kernels are native-only (see native/rpte_tree.cpp,
    native/rpte_scoring.cpp). This suite is vacuous if the extension
    didn't build."""
    import _hugiml_core

    for name in ("rpte_grow_tree", "rpte_apply_tree", "rpte_newton_leaf_values", "rpte_binomial_deviance"):
        assert hasattr(_hugiml_core, name), f"native _hugiml_core.{name} not available -- rebuild the extension"


def test_native_lookahead_solves_xor3_parity():
    """The bounded-lookahead mechanism's core claim: a target whose every
    raw feature has ~zero marginal effect (pure 3-way parity) is
    unsolvable by an ordinary greedy tree but solvable once the native
    root+child microtree search is engaged."""
    X, y = _xor3_dataset()
    rpte = LeafWiseBoundedLookaheadRPTEFeatureLR(leaf_config="4xD", depth=4, enable_lookahead=True)
    clf = HUGIMLClassifierNative(**MINING, base_estimator=rpte)
    clf.fit(X.iloc[:1000], y[:1000])
    from sklearn.metrics import roc_auc_score

    p = clf.predict_proba(X.iloc[1000:])[:, 1]
    auc = roc_auc_score(y[1000:], p)
    assert auc > 0.85


# ---------------------------------------------------------------------------
# Priority #1: raw-feature provenance reservation, both backends.
# ---------------------------------------------------------------------------


def test_sequential_backend_reserves_by_raw_feature_name():
    X, y = _xor3_dataset()
    rpte = LeafWiseBoundedLookaheadRPTEFeatureLR(leaf_config="4xD", depth=4, enable_lookahead=False)
    clf = HUGIMLClassifierNative(**MINING, base_estimator=rpte)
    clf.fit(X.iloc[:1000], y[:1000])
    fitted = clf.model_.named_steps["clf"]
    default_fe = fitted.fe_._default_fe
    assert default_fe is not None
    seen: set[str] = set()
    for reserved in default_fe.reserved_raw_features_:
        assert not (seen & reserved), f"raw-feature overlap across trees: {seen & reserved}"
        seen |= reserved


# ---------------------------------------------------------------------------
# Default final LR representation: leaves plus direct source inputs.
# ---------------------------------------------------------------------------


def _mixed_additive_tree_dataset(n=500, seed=773):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    score = 3.0 * X[:, 0] + 1.4 * X[:, 1] + 0.35 * rng.normal(size=n)
    y = (score > 0.0).astype(int)
    return X, y


def _one_stump_rpte(**kwargs):
    return LeafWiseBoundedLookaheadRPTEFeatureLR(
        leaf_config="2xD",
        depth=1,
        n_estimators=1,
        min_samples_leaf=8,
        lr_C=100.0,
        enable_lookahead=False,
        hugiml_feature_names=[f"orig:x{i}" for i in range(6)],
        **kwargs,
    )


def test_default_final_lr_uses_leaves_plus_direct_source_inputs():
    X, y = _mixed_additive_tree_dataset()
    fitted = _one_stump_rpte().fit(X, y)

    used = set(fitted.tree_used_input_indices_.tolist())
    direct = set(fitted.direct_input_indices_.tolist())
    assert used
    assert used.isdisjoint(direct)
    assert used | direct == set(range(X.shape[1]))
    assert fitted.n_final_lr_features_ == (
        fitted.n_leaf_features_ + fitted.direct_input_indices_.size
    )

    leaves = fitted.fe_.transform_leaves(sparse.csr_matrix(X))
    expected = sparse.hstack(
        [leaves, sparse.csr_matrix(X)[:, fitted.direct_input_indices_]],
        format="csr",
    )
    actual = fitted._final_lr_matrix(X)
    assert actual.shape == expected.shape
    assert (actual != expected).nnz == 0
    assert np.allclose(
        fitted.predict_proba(X), fitted.logistic_.predict_proba(expected)
    )


def test_final_lr_layout_is_not_a_public_hyperparameter():
    estimator = _one_stump_rpte()
    assert "final_lr_feature_mode" not in clone(estimator).get_params()
    with pytest.raises(TypeError, match="final_lr_feature_mode"):
        _one_stump_rpte(final_lr_feature_mode="leaves_only")


def test_unified_rule_table_includes_nonzero_direct_source_terms():
    X, y = _mixed_additive_tree_dataset()
    fitted = _one_stump_rpte().fit(X, y)
    rows = fitted.unified_rule_table()
    direct_rows = [r for r in rows if r["backend"] == "direct_hugiml_feature"]

    assert direct_rows, "the additive x1 signal should survive as a direct source term"
    direct_position = {
        int(input_idx): pos for pos, input_idx in enumerate(fitted.direct_input_indices_)
    }
    direct_coef = fitted.direct_input_coefficients()
    for row in direct_rows:
        input_idx = int(row["downstream_feature_index"])
        assert input_idx in direct_position
        assert row["final_logistic_coefficient"] == pytest.approx(
            direct_coef[direct_position[input_idx]]
        )
        assert row["tree_index"] is None
        assert row["centered_tree_contribution"] is None


# ---------------------------------------------------------------------------
# Priority #2/#3: pattern provenance wiring and structured atoms.
# ---------------------------------------------------------------------------


def test_pattern_provenance_structured_and_correctly_split():
    """A compound pattern label whose numeric interval bounds contain a
    comma must not be mis-split into spurious atoms."""
    X = pd.DataFrame(
        {
            "INCOME": np.random.default_rng(0).uniform(0, 15, 300),
            "AGE": np.random.default_rng(1).uniform(20, 60, 300),
        }
    )
    X["INCOMESQ"] = X["INCOME"] ** 2
    y = ((X["INCOME"] > 5) & (X["AGE"] > 35)).astype(int).to_numpy()
    clf = HUGIMLClassifierNative(
        B=-1,
        adaptive_binning=True,
        L=2,
        topK=200,
        feature_mode="original_plus_patterns",
        G=0.0001,
        convert_binary_to_categorical=False,
        execution_mode="production",
        n_jobs=1,
    )
    clf.fit(X, y)
    provenance = clf.get_pattern_provenance()
    for name, entry in provenance.items():
        assert entry["family"] == "pattern"
        assert set(entry["raw_features"]) <= {"INCOME", "INCOMESQ", "AGE"}, (
            f"spurious raw feature parsed from {name!r}: {entry['raw_features']}"
        )
        for atom in entry["atoms"]:
            assert atom["feature"] in {"INCOME", "INCOMESQ", "AGE"}
            if atom["operator"] == "interval":
                assert atom["lower"] < atom["upper"]


def test_pattern_metadata_survives_onevsrest_cloning():
    """Metadata set only via a post-hoc method call does not survive
    OneVsRestClassifier's internal per-class cloning; metadata set through
    clone-safe constructor parameters must survive it."""
    X, y = _xor3_dataset()
    rpte = OneVsRestClassifier(
        LeafWiseBoundedLookaheadRPTEFeatureLR(
            leaf_config="2xD", depth=4, enable_lookahead="adaptive"
        ),
        n_jobs=1,
    )
    clf = HUGIMLClassifierNative(**MINING, base_estimator=rpte)
    clf.fit(X.iloc[:1000], y[:1000])
    fitted_ovr = clf.model_.named_steps["clf"]
    fitted_rpte = fitted_ovr.estimators_[0]
    assert len(fitted_rpte.hugiml_feature_names or []) > 0
    assert (
        fitted_rpte.fe_.default_backend_reason_ is None
        or fitted_rpte.fe_.adaptive_used_lookahead_ is not None
    )


# ---------------------------------------------------------------------------
# Priority #4: exact threshold rendering.
# ---------------------------------------------------------------------------


def test_threshold_rendering_respects_left_and_right_branches():
    original_std = {
        "age": {
            "standardization_mean": 50.0,
            "standardization_scale": 10.0,
            "missing_value_policy": "median",
        }
    }
    pair_name = "augmented_pair:product__age__income"
    pair_catalog = {
        pair_name: {
            "name": "product__age__income",
            "inputs": ["age", "income"],
            "operation": "product",
            "raw_formula": "age * income",
            "standardization_mean": 1000.0,
            "standardization_scale": 100.0,
            "pair_missing_policy": "reference_value_for_unavailable_pair",
        }
    }
    pattern_name = "pattern:age=[50,60)"
    pattern_provenance = {
        pattern_name: {"raw_features": ["age"], "atoms": []},
    }

    age_left = _render_threshold(
        "orig:age",
        0.5,
        original_std,
        pair_catalog,
        pattern_provenance,
        is_right=False,
    )
    age_right = _render_threshold(
        "orig:age",
        0.5,
        original_std,
        pair_catalog,
        pattern_provenance,
        is_right=True,
    )
    assert age_left["operator"] == "<="
    assert age_left["raw_condition"] == "age <= 55"
    assert age_right["operator"] == ">"
    assert age_right["raw_condition"] == "age > 55"

    pair_left = _render_threshold(
        pair_name,
        0.5,
        original_std,
        pair_catalog,
        pattern_provenance,
        is_right=False,
    )
    assert pair_left["raw_condition"] == "age * income <= 1050"

    pattern_right = _render_threshold(
        pattern_name,
        0.5,
        original_std,
        pair_catalog,
        pattern_provenance,
        is_right=True,
    )
    pattern_left = _render_threshold(
        pattern_name,
        0.5,
        original_std,
        pair_catalog,
        pattern_provenance,
        is_right=False,
    )
    assert pattern_right["raw_condition"] == "age=[50,60)"
    assert pattern_left["raw_condition"] == "NOT (age=[50,60))"


def test_repeated_thresholds_collapse_to_tightest_interval():
    path = [
        ("orig:age", -1.0, True),
        ("orig:age", 0.5, True),
        ("orig:age", 1.2, False),
        ("orig:age", 0.9, False),
        ("orig:bmi", 0.0, True),
    ]
    assert _simplify_threshold_path(path) == [
        ("orig:age", 0.5, True),
        ("orig:age", 0.9, False),
        ("orig:bmi", 0.0, True),
    ]


def test_raw_feature_fallback_is_rendered_as_linear_term_not_fake_split():
    term = _render_linear_term(
        "orig:age",
        {
            "age": {
                "standardization_mean": 50.0,
                "standardization_scale": 10.0,
                "missing_value_policy": "median",
            }
        },
        {},
        {},
    )
    assert term["operator"] == "linear_term"
    assert term["standardized_threshold"] is None
    assert term["downstream_condition"] == "linear term: orig:age"
    assert term["raw_condition"] == "linear value of age"


def test_original_feature_threshold_inverts_to_raw_units():
    X = pd.DataFrame(
        {
            "age": np.random.default_rng(0).uniform(20, 80, 500),
            "noise": np.random.default_rng(1).normal(size=500),
        }
    )
    y = (X["age"] > 50).astype(int).to_numpy()
    rpte = LeafWiseBoundedLookaheadRPTEFeatureLR(
        leaf_config="4xD", depth=4, enable_lookahead=False, min_samples_leaf=5
    )
    clf = HUGIMLClassifierNative(
        B=-1,
        adaptive_binning=True,
        L=1,
        topK=50,
        feature_mode="original_plus_patterns",
        G=0.01,
        convert_binary_to_categorical=False,
        execution_mode="production",
        n_jobs=1,
        base_estimator=rpte,
    )
    clf.fit(X, y)
    table = clf.rpte_rule_table()
    assert table, "expected at least one rule"
    invertible_age_conditions = [
        c
        for row in table
        for c in row["conditions"]
        if c["family"] == "original" and c["invertible"] and "age" in c["raw_sources"]
    ]
    assert invertible_age_conditions, "expected at least one invertible age condition"
    for c in invertible_age_conditions:
        assert c["raw_condition"] is not None
        assert "age" in c["raw_condition"]
        expected_operator = ">" if c["direction"] == "above_threshold" else "<="
        assert c["operator"] == expected_operator
        assert f"age {expected_operator}" in c["raw_condition"]


# ---------------------------------------------------------------------------
# Priority #5/#6/#9/#13: unified rule table schema and content.
# ---------------------------------------------------------------------------


def test_unified_rule_table_schema_and_newton_logistic_separation():
    X, y = _xor3_dataset()
    rpte = LeafWiseBoundedLookaheadRPTEFeatureLR(leaf_config="4xD", depth=4, enable_lookahead=True)
    clf = HUGIMLClassifierNative(**MINING, base_estimator=rpte)
    clf.fit(X.iloc[:1000], y[:1000])
    fitted = clf.model_.named_steps["clf"]
    table = fitted.unified_rule_table()
    assert table
    required_fields = {
        "class",
        "tree_index",
        "leaf_index",
        "backend",
        "conditions",
        "raw_conditions",
        "raw_sources",
        "support_count",
        "support_rate",
        "final_logistic_coefficient",
        "centered_tree_contribution",
        "newton_leaf_value",
        "fallback_status",
    }
    for row in table:
        assert required_fields <= set(row.keys())
        # Newton value and final logistic coefficient are DIFFERENT
        # numbers describing different things -- not the same value under
        # two names.
        if row["newton_leaf_value"] is not None:
            assert row["newton_leaf_value"] != row["final_logistic_coefficient"]


def test_centered_tree_contribution_is_weighted_average_zero():
    X, y = _xor3_dataset()
    rpte = LeafWiseBoundedLookaheadRPTEFeatureLR(leaf_config="4xD", depth=4, enable_lookahead=True)
    clf = HUGIMLClassifierNative(**MINING, base_estimator=rpte)
    clf.fit(X.iloc[:1000], y[:1000])
    fitted = clf.model_.named_steps["clf"]
    table = fitted.unified_rule_table()
    by_tree: dict[int, list[dict]] = {}
    for row in table:
        if row["tree_index"] is not None:
            by_tree.setdefault(row["tree_index"], []).append(row)
    for tree_index, rows in by_tree.items():
        supports = np.array([r["support_count"] for r in rows], dtype=float)
        centered = np.array([r["centered_tree_contribution"] for r in rows], dtype=float)
        if supports.sum() > 0:
            weighted_mean = np.average(centered, weights=supports)
            assert abs(weighted_mean) < 1e-6, (
                f"tree {tree_index} centered contributions don't average to 0"
            )


def test_backend_and_fallback_status_reported():
    X, y = _xor3_dataset()
    mining_no_pairs = dict(MINING)
    mining_no_pairs["augmented_pair_transforms"] = False
    rpte = LeafWiseBoundedLookaheadRPTEFeatureLR(leaf_config="4xD", depth=4, enable_lookahead=True)
    clf = HUGIMLClassifierNative(**mining_no_pairs, base_estimator=rpte)
    clf.fit(X.iloc[:1000], y[:1000])
    table = clf.rpte_rule_table()
    assert table
    assert table[0]["fallback_status"] == "no_augmented_pairs"
    assert table[0]["backend"] in {"sequential_default", "raw_hugiml_features", "constant"}


# ---------------------------------------------------------------------------
# Priority #8: raw-source aggregation.
# ---------------------------------------------------------------------------


def test_raw_source_aggregation_no_double_counting_interactions():
    rows = [
        {
            "final_logistic_coefficient": 2.0,
            "conditions": [
                {
                    "downstream_feature": "augmented_pair:product__age__income",
                    "family": "augmented_pair",
                    "raw_sources": ["age", "income"],
                }
            ],
        },
        {
            "final_logistic_coefficient": 0.5,
            "conditions": [
                {
                    "downstream_feature": "orig:age",
                    "family": "original",
                    "raw_sources": ["age"],
                }
            ],
        },
    ]
    agg = aggregate_rule_table_by_raw_source(rows, allocation="interaction_only")
    by_source = {
        tuple(entry["raw_source"])
        if isinstance(entry["raw_source"], (list, tuple))
        else entry["raw_source"]: entry
        for entry in agg["raw_source_contributions"]
    }

    assert by_source[("age", "income")]["total_contribution"] == pytest.approx(2.0)
    assert by_source[("age", "income")]["kind"] == "interaction"
    assert by_source["age"]["total_contribution"] == pytest.approx(0.5)
    assert "income" not in by_source


# ---------------------------------------------------------------------------
# Priority #14: serialization.
# ---------------------------------------------------------------------------


def test_pickle_roundtrip_preserves_predictions_and_explanations():
    X, y = _xor3_dataset()
    rpte = LeafWiseBoundedLookaheadRPTEFeatureLR(leaf_config="4xD", depth=4, enable_lookahead=True)
    clf = HUGIMLClassifierNative(**MINING, base_estimator=rpte)
    clf.fit(X.iloc[:1000], y[:1000])
    p_before = clf.predict_proba(X.iloc[1000:])[:, 1]
    table_before = clf.rpte_rule_table()

    clf2 = pickle.loads(pickle.dumps(clf))
    p_after = clf2.predict_proba(X.iloc[1000:])[:, 1]
    table_after = clf2.rpte_rule_table()

    assert np.allclose(p_before, p_after)
    assert table_before == table_after


def test_save_model_load_model_roundtrip_for_bare_rpte(tmp_path):
    """hugiml.serialization.save_model() -- the package's versioned model
    persistence API -- must round-trip
    an RPTE-based model.

    `get_params()` includes `base_estimator`, which is a live (unfitted)
    RPTE estimator instance rather than a JSON type when the classifier was
    fit through RPTE. RPTE is reachable through the package's own
    "performance_ho" grid (the default grid), so this is the primary,
    documented model-persistence path for the default grid's RPTE branch.
    See serialization.py's MODEL_SCHEMA_VERSION notes.
    """
    from hugiml.serialization import load_model, save_model

    X, y = _xor3_dataset()
    rpte = LeafWiseBoundedLookaheadRPTEFeatureLR(leaf_config="3xD", depth=3, enable_lookahead=True)
    clf = HUGIMLClassifierNative(**MINING, base_estimator=rpte)
    clf.fit(X.iloc[:1000], y[:1000])
    p_before = clf.predict_proba(X.iloc[1000:])[:, 1]
    table_before = clf.rpte_rule_table()

    path = tmp_path / "rpte_model.hugiml"
    save_model(clf, path)  # must not raise
    clf2 = load_model(path)

    p_after = clf2.predict_proba(X.iloc[1000:])[:, 1]
    assert np.allclose(p_before, p_after)
    assert clf2.rpte_rule_table() == table_before

    # base_estimator itself (the unfitted hyperparameter template, not the
    # fitted model_ Pipeline -- that's covered by the predictions/rule-table
    # equality above) must also survive the round trip: this is what a
    # refit (e.g. dashboard/runner.py's feature-pruning refit, which
    # deliberately preserves base_estimator across a re-fit) or
    # sklearn.base.clone() would read, and previously silently reverted to
    # None (the built-in LR branch) after any load_model() call.
    assert clf2.base_estimator is not None
    assert type(clf2.base_estimator).__name__ == type(clf.base_estimator).__name__
    assert clf2.base_estimator.get_params()["leaf_config"] == "3xD"
    assert clf2.base_estimator.get_params()["depth"] == 3


def test_save_model_load_model_roundtrip_for_ovr_wrapped_rpte(tmp_path):
    """Same as test_save_model_load_model_roundtrip_for_bare_rpte, but for
    the OneVsRestClassifier-wrapped shape hyperparameter_configs.py's
    "performance_ho" grid actually uses -- reconstruction must recurse into
    the wrapper's own `estimator=` param, not just the outer class.
    """
    from sklearn.multiclass import OneVsRestClassifier

    from hugiml.serialization import load_model, save_model

    X, y = _xor3_dataset()
    rpte = OneVsRestClassifier(
        LeafWiseBoundedLookaheadRPTEFeatureLR(leaf_config="3xD", depth=4, enable_lookahead="adaptive"),
        n_jobs=1,
    )
    clf = HUGIMLClassifierNative(**MINING, base_estimator=rpte)
    clf.fit(X.iloc[:1000], y[:1000])
    p_before = clf.predict_proba(X.iloc[1000:])[:, 1]

    path = tmp_path / "rpte_ovr_model.hugiml"
    save_model(clf, path)  # must not raise
    clf2 = load_model(path)

    p_after = clf2.predict_proba(X.iloc[1000:])[:, 1]
    assert np.allclose(p_before, p_after)
    assert type(clf2.base_estimator).__name__ == "OneVsRestClassifier"
    inner = clf2.base_estimator.estimator
    assert type(inner).__name__ == "LeafWiseBoundedLookaheadRPTEFeatureLR"
    assert inner.get_params()["leaf_config"] == "3xD"
    assert inner.get_params()["enable_lookahead"] == "adaptive"


def test_save_model_load_model_roundtrip_unaffected_for_plain_lr(tmp_path):
    """Backward-compatibility check: a model fit with base_estimator=None
    (the built-in logistic-regression branch, the only case that existed
    before RPTE) must round-trip exactly as before -- the versioned serialization
    change must be a no-op for this case, not just avoid a crash for RPTE.
    """
    from hugiml.serialization import load_model, save_model

    X, y = _xor3_dataset()
    clf = HUGIMLClassifierNative(**MINING)
    clf.fit(X.iloc[:1000], y[:1000])
    p_before = clf.predict_proba(X.iloc[1000:])[:, 1]

    path = tmp_path / "lr_model.hugiml"
    save_model(clf, path)
    clf2 = load_model(path)

    p_after = clf2.predict_proba(X.iloc[1000:])[:, 1]
    assert np.allclose(p_before, p_after)
    assert clf2.base_estimator is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

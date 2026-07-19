from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from hugiml.rpte_bounded_lookahead_leafwise import (
    LeafWiseBoundedLookaheadRPTEFeatureLR,
)


class _LeafMatrix:
    is_degenerate_ = True
    _raw_feature_fallback_ = False
    default_backend_reason_ = None

    def transform_leaves(self, X):
        return sparse.csr_matrix(np.asarray([[1.0], [0.0], [1.0]]))


def _fitted_direct_term_model():
    model = LeafWiseBoundedLookaheadRPTEFeatureLR(
        hugiml_feature_names=[
            "orig:age",
            "orig:income",
            "pattern:age=[40,60)",
            "augmented_pair:age_x_income",
        ]
    )
    model.fe_ = _LeafMatrix()
    model.classes_ = np.asarray([0, 1])
    model.n_input_features_ = 4
    model.n_leaf_features_ = 1
    model.n_final_lr_features_ = 3
    model.direct_input_indices_ = np.asarray([1, 3], dtype=np.int64)
    model.logistic_ = SimpleNamespace(coef_=np.asarray([[0.8, 0.2, -0.3]]))
    return model


def test_final_lr_matrix_orders_leaf_then_direct_source_terms():
    model = _fitted_direct_term_model()
    X = sparse.csr_matrix(
        np.asarray(
            [
                [10.0, 20.0, 0.0, 2.0],
                [11.0, 21.0, 1.0, 3.0],
                [12.0, 22.0, 0.0, 4.0],
            ]
        )
    )

    actual = model._final_lr_matrix(X).toarray()
    expected = np.asarray(
        [
            [1.0, 20.0, 2.0],
            [0.0, 21.0, 3.0],
            [1.0, 22.0, 4.0],
        ]
    )
    assert np.array_equal(actual, expected)
    assert np.array_equal(model.direct_input_coefficients(), np.asarray([0.2, -0.3]))


def test_unified_rule_table_uses_direct_source_identifiers():
    model = _fitted_direct_term_model()
    rows = model.unified_rule_table()
    direct_rows = [row for row in rows if row.get("term_role") == "direct_source_term"]

    assert len(direct_rows) == 2
    assert {row["backend"] for row in direct_rows} == {"direct_hugiml_feature"}
    assert {row["downstream_feature"] for row in direct_rows} == {
        "orig:income",
        "augmented_pair:age_x_income",
    }
    assert {row["source_selection_status"] for row in direct_rows} == {
        "not_selected_in_tree_split"
    }

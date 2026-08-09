from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from hugiml import HUGIMLClassifier


def test_exact_downstream_redundancies_are_removed_deterministically():
    model = HUGIMLClassifier()
    indicator = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.float64)
    other = np.asarray([0.2, 0.4, 0.1, 0.7, 0.9, 0.3], dtype=np.float64)
    matrix = np.column_stack(
        [
            np.ones(indicator.size),
            indicator,
            indicator.copy(),
            1.0 - indicator,
            other,
        ]
    )

    reduced = model._canonicalize_lr_downstream_fit(matrix)

    assert reduced.shape == (indicator.size, 2)
    np.testing.assert_array_equal(reduced, np.column_stack([indicator, other]))
    expected_counts = {
        "input_columns": 5,
        "retained_columns": 2,
        "removed_constant_columns": 1,
        "removed_duplicate_columns": 1,
        "removed_complementary_columns": 1,
    }
    assert {
        key: model._downstream_lr_canonicalization_[key] for key in expected_counts
    } == expected_counts
    assert model._downstream_lr_canonicalization_["vif_columns_above_threshold"] == 0
    assert model._downstream_lr_canonicalization_["vif_threshold"] == 5.0
    np.testing.assert_array_equal(model._apply_lr_downstream_canonical_transform(matrix), reduced)

    sparse_model = HUGIMLClassifier()
    sparse_reduced = sparse_model._canonicalize_lr_downstream_fit(csr_matrix(matrix))
    np.testing.assert_array_equal(sparse_reduced.toarray(), reduced)
    sparse_telemetry = dict(sparse_model._downstream_lr_canonicalization_)
    dense_telemetry = dict(model._downstream_lr_canonicalization_)
    assert sparse_telemetry.pop("vif_analysis_seconds") >= 0.0
    assert dense_telemetry.pop("vif_analysis_seconds") >= 0.0
    assert sparse_telemetry == dense_telemetry


def test_fitted_lr_schema_and_predictions_survive_serialization():
    rng = np.random.RandomState(7)
    frame = pd.DataFrame(
        {
            "x0": rng.normal(size=120),
            "x1": rng.normal(size=120),
            "binary": np.tile([0, 1], 60),
        }
    )
    target = ((frame["x0"] + 0.4 * frame["x1"]) > 0).astype(int).to_numpy()
    model = HUGIMLClassifier(
        adaptive_binning=True,
        G=1e-3,
        L=1,
        topK=20,
        feature_mode="original_plus_patterns",
        augmented_pair_transforms=True,
        execution_mode="audit",
    )
    model.fit(frame, target)

    coefficients = np.asarray(model.model_.named_steps["clf"].coef_)
    assert coefficients.shape[1] == len(model._get_downstream_feature_names())
    assert coefficients.shape[1] == model.x_train_downstream_.shape[1]
    before = model.predict_proba(frame)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict_proba(frame), before, rtol=0, atol=0)

    from hugiml.compute_complexity import get_complexity_report

    report = get_complexity_report(restored)
    audit = report["downstream_redundancy_audit"]
    assert audit["input_columns"] >= audit["retained_columns"]
    assert "removed_constant_columns" in audit
    assert "removed_duplicate_columns" in audit
    assert "removed_complementary_columns" in audit
    assert "removed_high_vif_pattern_columns" in audit
    assert "removed_high_vif_augmented_pair_columns" in audit
    assert "vif_analysis_seconds" not in audit

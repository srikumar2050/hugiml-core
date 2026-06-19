import pickle
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from hugiml.classifier import HUGIMLClassifierNative, NativeAugmentedPairTransformBlock


class _CoefEstimator:
    def __init__(self, coef):
        self.coef_ = np.asarray([coef], dtype=float)


def _fake_fitted_classifier_with_augmented_ops():
    clf = HUGIMLClassifierNative(
        B=4, L=2, feature_mode="patterns_only", interaction_relaxed_mining=False
    )
    clf.classes_ = np.array([0, 1])
    clf.n_features_in_ = 2
    clf.feature_names_in_ = ["col0", "col1"]
    clf.cat_cols_mask_ = np.array([False, False])
    clf.is_int_mask_ = np.array([False, False])
    clf.patterns_ = []
    clf.raw_patterns_ = []
    clf.td_ = SimpleNamespace(item_map={})
    clf.x_train_hup_ = csr_matrix((5, 0), dtype=np.float32)
    clf.x_train_downstream_ = csr_matrix(np.ones((5, 2), dtype=np.float32))
    clf.model_ = SimpleNamespace(named_steps={"clf": _CoefEstimator([0.8, -0.4])})
    clf.augmented_pair_transforms = True
    clf.aug_feature_size = 2
    clf.augmented_pair_selected_features_ = ["col0", "col1"]
    clf.augmented_pair_transforms_enabled_ = True
    clf.augmented_pair_config_ = {"enabled": True, "ops": ["sum", "signed_difference"]}
    clf.augmented_pair_transforms_ = [
        {
            "name": "augmented_pair_sum__col0__col1",
            "operation": "sum",
            "inputs": ("col0", "col1"),
            "raw_formula": "col0 + col1",
            "standardized_formula": "(col0 + col1 - 1.0) / 2.0",
            "standardization_mean": 1.0,
            "standardization_scale": 2.0,
            "reference_raw_value": 1.0,
            "pair_missing_policy": "reference_value_for_unavailable_pair",
            "eligible_count": 5,
            "eligible_rate": 1.0,
            "missing_pair_rate": 0.0,
            "source_observed_medians": {"col0": 0.5, "col1": 0.5},
            "transform_ig": 0.12,
        },
        {
            "name": "augmented_pair_diff__col0__col1",
            "operation": "signed_difference",
            "inputs": ("col0", "col1"),
            "raw_formula": "col0 - col1",
            "standardized_formula": "(col0 - col1 - 0.0) / 0.5",
            "standardization_mean": 0.0,
            "standardization_scale": 0.5,
            "reference_raw_value": 0.0,
            "pair_missing_policy": "reference_value_for_unavailable_pair",
            "eligible_count": 5,
            "eligible_rate": 1.0,
            "missing_pair_rate": 0.0,
            "source_observed_medians": {"col0": 0.5, "col1": 0.5},
            "transform_ig": 0.10,
        },
    ]
    clf._downstream_non_missing_rate_ = np.array([1.0, 1.0])
    clf._downstream_variance_ = np.array([1.0, 1.0])
    clf._downstream_pattern_support_ = np.array([])
    return clf


def test_feature_importances_exposes_augmented_pair_operation_and_inputs():
    clf = _fake_fitted_classifier_with_augmented_ops()
    imp = clf.feature_importances()
    aug = imp[imp["feature_type"] == "augmented_pair"].sort_values("feature")
    assert set(aug["operation"]) == {"sum", "signed_difference"}
    assert "inputs" in aug.columns
    assert aug["raw_formula"].str.contains("col0").all()
    assert aug["coefficient_raw_scale"].notna().all()


def test_augmented_pair_effects_have_operation_specific_language():
    clf = _fake_fitted_classifier_with_augmented_ops()
    effects = clf.explain_augmented_pair_effects()
    by_op = {row["operation"]: row for _, row in effects.iterrows()}
    assert by_op["sum"]["risk_increases_when"] == "sum_value_increases"
    assert "pair sum scale" in by_op["sum"]["raw_scale_note"]
    assert by_op["signed_difference"]["risk_increases_when"] == "left_minus_right_decreases"
    assert "left-minus-right difference" in by_op["signed_difference"]["raw_scale_note"]


def test_pickle_preserves_augmented_pair_metadata_for_new_ops():
    clf = _fake_fitted_classifier_with_augmented_ops()
    restored = pickle.loads(pickle.dumps(clf))
    assert restored.get_augmented_pair_transforms()[0]["operation"] == "sum"
    assert restored.get_augmented_pair_transforms()[1]["operation"] == "signed_difference"
    imp = restored.feature_importances()
    assert set(imp["operation"].dropna()) == {"sum", "signed_difference"}


def test_native_block_state_contains_opcode_and_schema_fields_for_new_ops():
    block = NativeAugmentedPairTransformBlock(aug_feature_size=2)
    block.input_feature_names_ = ["col0", "col1"]
    block.selected_aug_features_ = ["col0", "col1"]
    block.selected_aug_scores_ = {"col0": 0.5, "col1": 0.4}
    block.input_bin_edges_ = {"col0": [0, 1], "col1": [0, 1]}
    block.source_observed_medians_ = {"col0": 0.5, "col1": 0.5}
    block.source_observed_medians_array_ = np.array([0.5, 0.5])
    block.numeric_medians_ = dict(block.source_observed_medians_)
    block.numeric_medians_array_ = block.source_observed_medians_array_
    block.kept_specs_ = [
        {"name": "augmented_pair_sum__col0__col1", "operation": "sum", "inputs": ("col0", "col1"), "formula": "col0 + col1", "reference_raw_value": 1.0, "eligible_count": 5, "eligible_rate": 1.0, "missing_pair_rate": 0.0, "transform_ig": 0.2},
        {"name": "augmented_pair_diff__col0__col1", "operation": "signed_difference", "inputs": ("col0", "col1"), "formula": "col0 - col1", "reference_raw_value": 0.0, "eligible_count": 5, "eligible_rate": 1.0, "missing_pair_rate": 0.0, "transform_ig": 0.1},
    ]
    block.feature_names_ = [s["name"] for s in block.kept_specs_]
    block.pair_reference_values_ = np.array([1.0, 0.0])
    block.scaler_mean_ = np.array([1.0, 0.0])
    block.scaler_scale_ = np.array([2.0, 0.5])
    block.left_indices_ = np.array([0, 0], dtype=np.int64)
    block.right_indices_ = np.array([1, 1], dtype=np.int64)
    block.op_codes_ = np.array([2, 3], dtype=np.int8)
    block.augmented_pair_transforms_ = block._build_catalog()
    restored = pickle.loads(pickle.dumps(block))
    assert restored.op_codes_.tolist() == [2, 3]
    assert [t["operation"] for t in restored.augmented_pair_transforms_] == ["sum", "signed_difference"]

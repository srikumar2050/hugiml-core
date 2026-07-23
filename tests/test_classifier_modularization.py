from __future__ import annotations

import pickle

from hugiml import FitMetadata, HUGIMLClassifier, HUGIMLClassifierNative
from hugiml.adaptive import HUGIMLAdaptive
from hugiml.classifier import HUGIMLTuneResult, NativeAugmentedPairTransformBlock


def test_primary_and_compatibility_names_share_one_class():
    assert HUGIMLClassifierNative is HUGIMLClassifier
    assert HUGIMLClassifier.__name__ == "HUGIMLClassifier"
    assert HUGIMLClassifier.__module__ == "hugiml.classifier"


def test_legacy_pickle_global_resolves_to_primary_class():
    loaded = pickle.loads(b"chugiml.classifier\nHUGIMLClassifierNative\n.")
    assert loaded is HUGIMLClassifier


def test_behavior_is_owned_by_focused_modules():
    assert HUGIMLClassifier.fit.__module__ == "hugiml._classifier_training"
    assert HUGIMLClassifier.prepareXy.__module__ == "hugiml._classifier_binning"
    assert HUGIMLClassifier.transform.__module__ == "hugiml._classifier_prediction"
    assert HUGIMLClassifier.get_pattern_provenance.__module__ == (
        "hugiml._classifier_interpretation"
    )
    assert HUGIMLClassifier.feature_importances.__module__ == ("hugiml._classifier_inspection")


def test_adaptive_estimator_uses_primary_classifier_contract():
    model = HUGIMLAdaptive()
    assert isinstance(model, HUGIMLClassifier)
    assert isinstance(model, HUGIMLClassifierNative)


def test_tuning_methods_remain_bound_to_primary_class():
    assert HUGIMLClassifier.tune.__self__ is HUGIMLClassifier
    assert HUGIMLClassifier.fast_grid_tune.__self__ is HUGIMLClassifier


def test_public_support_types_keep_classifier_module_path():
    assert FitMetadata.__module__ == "hugiml.classifier"
    assert NativeAugmentedPairTransformBlock.__module__ == "hugiml.classifier"
    assert HUGIMLTuneResult.__module__ == "hugiml.classifier"


def test_classifier_specific_module_symbols_remain_available():
    import hugiml.classifier as classifier_module

    expected = {
        "AUGMENTED_PAIR_MODES",
        "AUGMENTED_PAIR_OPS",
        "DEFAULT_AUGMENTED_PAIR_UNBOUNDED_CAP",
        "MIN_SCHEMA_VERSION",
        "MODEL_SCHEMA_VERSION",
        "_hugiml_build_fast_tune_adaptive_context",
        "_is_zero_variance_numeric_column",
    }
    assert expected.issubset(set(dir(classifier_module)))


def test_public_support_type_pickle_globals_resolve():
    block = NativeAugmentedPairTransformBlock(aug_feature_size=2)
    restored = pickle.loads(pickle.dumps(block))
    assert isinstance(restored, NativeAugmentedPairTransformBlock)

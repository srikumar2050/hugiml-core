import pytest

from hugiml.classifier import HUGIMLClassifierNative, HUGIMLParamError


def test_interaction_relaxed_accepts_bounded_l_gt3():
    clf = HUGIMLClassifierNative(
        L=4,
        interaction_relaxed_mining=True,
        augmented_pair_transforms=False,
    )
    clf._validate_params()


def test_interaction_relaxed_accepts_l5():
    clf = HUGIMLClassifierNative(
        L=5,
        interaction_relaxed_mining=True,
        augmented_pair_transforms=False,
    )
    clf._validate_params()


def test_interaction_relaxed_rejects_invalid_zero_depth():
    clf = HUGIMLClassifierNative(
        L=0,
        interaction_relaxed_mining=True,
        augmented_pair_transforms=False,
    )
    with pytest.raises(HUGIMLParamError, match="L=-1 or L>=1"):
        clf._validate_params()

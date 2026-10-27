"""Validate native tree inputs before pointer access or traversal."""

import subprocess
import sys

import _hugiml_core as core
import numpy as np
import pytest


def tree():
    return dict(
        is_fitted=True,
        node_is_leaf=[0, 1, 1],
        node_feature_index=[0, -1, -1],
        node_threshold=[0.5, 0.0, 0.0],
        node_left=[1, -1, -1],
        node_right=[2, -1, -1],
        synthetic_specs=[],
        used_columns=[0],
        used_raw_features=[0],
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("node_feature_index", []),
        ("node_is_leaf", []),
        ("node_left", [0, -1, -1]),
        ("node_left", [10, -1, -1]),
        ("node_left", [-1, -1, -1]),
        ("node_right", [1, -1, -1]),
        ("node_feature_index", [2, -1, -1]),
        ("node_feature_index", [-1, -1, -1]),
        ("node_threshold", [float("nan"), 0.0, 0.0]),
    ],
)
def test_malformed_tree_rejected_in_child(key, value):
    value_tree = tree()
    value_tree[key] = value
    # Subprocess isolation also turns crashes and hangs into test failures.
    code = "import numpy as np; import _hugiml_core as c\n"
    code += "nan=float('nan')\nt=" + repr(value_tree) + "\n"
    code += "try: c.rpte_apply_tree(np.zeros((2,1)),t)\nexcept ValueError: pass\nelse: raise AssertionError('accepted')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_valid_tree_predictions():
    np.testing.assert_array_equal(core.rpte_apply_tree(np.array([[0.0], [1.0]]), tree()), [1, 2])


@pytest.mark.parametrize("a,b,op", [(-1, 0, 0), (0, 5, 0), (0, 0, 99)])
def test_synthetic_reference_rejected(a, b, op):
    t = tree()
    t["synthetic_specs"] = [
        dict(
            a_idx=a,
            b_idx=b,
            op=op,
            a_threshold=0.0,
            b_threshold=0.0,
            is_binarized=False,
            owners=[0],
        )
    ]
    with pytest.raises(ValueError):
        core.rpte_apply_tree(np.zeros((2, 1)), t)


def grow_args():
    return dict(
        X=np.zeros((10, 2)),
        labels=np.arange(10, dtype=np.int8) % 2,
        grow_rows=np.arange(5, dtype=np.int64),
        probe_rows=np.arange(5, 10, dtype=np.int64),
        eligible_columns=np.array([0, 1], dtype=np.int32),
        raw_indices=np.array([0, 1], dtype=np.int32),
        roots=np.array([], dtype=np.int32),
        owner_by_column=[[0], [1]],
        max_depth=2,
        max_leaf_nodes=4,
        min_samples_leaf=2,
        min_probe_leaf=2,
        random_state=42,
        greedy_stall_probe_ig=0.02,
        lookahead_min_probe_ig=0.05,
        lookahead_min_increment=0.03,
        lookahead_beam_width=4,
        lookahead_child_mode_shared=True,
        lookahead_ops=np.array([0, 1, 2, 3], dtype=np.int32),
        max_root_thresholds=7,
        min_weighted_probe_gain=0.01,
        raw_pair_fallback=True,
        raw_pair_max_candidates=20,
        aug_child_enabled=True,
        aug_child_max_candidates=10,
        use_statistical_acceptance=True,
        significance_alpha=0.05,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("labels", np.array([0], dtype=np.int8)),
        ("grow_rows", np.array([-1], dtype=np.int64)),
        ("probe_rows", np.array([10], dtype=np.int64)),
        ("eligible_columns", np.array([2], dtype=np.int32)),
        ("roots", np.array(-1, dtype=np.int32)),
        ("raw_indices", np.zeros((1, 1), dtype=np.int32)),
        ("lookahead_ops", np.array([4], dtype=np.int32)),
    ],
)
def test_grow_bounds(key, value):
    args = grow_args()
    args[key] = value
    with pytest.raises(ValueError):
        core.rpte_grow_tree(**args)

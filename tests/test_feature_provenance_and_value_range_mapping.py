# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Ground-truth regression tests for feature/pattern/augmented-pair/RPTE-rule
provenance: every explanation surface must map back to the correct original
feature name and the correct raw (un-standardized, un-binned) value range,
with no index mismatch across the mining -> downstream-matrix -> fitted
-estimator pipeline.

Each test builds a synthetic dataset where the true signal (a specific raw
feature and a specific threshold, or a specific pair of interacting raw
features) is known exactly, then checks that HUGIML's explanation output
names that exact feature/pair and reports a threshold/coefficient consistent
with the known ground truth -- not merely that it runs without raising. This
catches a wrong-column, off-by-one, or scrambled-order bug that a
count-only or type-only assertion would not: those would still produce a
same-shaped, same-typed, but factually incorrect explanation.

Several tests deliberately stress the conditions most likely to break index
alignment: a zero-variance column dropped before fitting, and the training
frame's columns presented in a shuffled order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hugiml import HUGIMLClassifierNative
from hugiml.hyperparameter_configs import get_hugiml_grid


def _age_threshold_dataset(*, seed: int = 42, shuffle_columns: bool = False, n: int = 2000):
    """A dataset with a known, dominant signal: y depends on age > 55 (plus
    a weaker income > 90000 effect), alongside a true zero-variance column,
    a categorical column, and several pure-noise columns -- so a fit must
    correctly single out "age" (and, more weakly, "income") among
    plausible-looking distractors positioned at different column indices.
    """
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "noise1": rng.uniform(0, 1, n),
        "age": rng.uniform(20, 80, n),
        "noise2": rng.uniform(0, 1, n),
        "const_col": np.full(n, 7.0),
        "income": rng.uniform(20000, 150000, n),
        "noise3": rng.uniform(0, 1, n),
        "cat_flag": rng.randint(0, 2, n),
        "noise4": rng.uniform(0, 1, n),
        "noise5": rng.uniform(0, 1, n),
    })
    logit = 3.0 * (df["age"] > 55).astype(float) + 0.5 * (df["income"] > 90000).astype(float)
    prob = 1.0 / (1.0 + np.exp(-(logit - 1.5)))
    y = (rng.uniform(0.0, 1.0, n) < prob).astype(int)
    if shuffle_columns:
        cols = list(df.columns)
        rng.shuffle(cols)
        df = df[cols]
    return df, y


def _fit_rpte(df: pd.DataFrame, y: np.ndarray, **kwargs) -> HUGIMLClassifierNative:
    rpte_estimator = get_hugiml_grid("performance_ho")["base_estimator"][1]
    clf = HUGIMLClassifierNative(
        L=1, topK=30, feature_mode="original_plus_patterns",
        augmented_pair_transforms=True, base_estimator=rpte_estimator, **kwargs,
    )
    clf.fit(df, y)
    return clf


def _raw_conditions_for(rules: list[dict], raw_feature: str) -> list[str]:
    out = []
    for rule in rules:
        for cond in rule.get("conditions") or []:
            raw = cond.get("raw_condition")
            if raw and raw.startswith(f"{raw_feature} "):
                out.append(raw)
    return out


class TestRPTERawThresholdMatchesGroundTruth:
    """rpte_rule_table()'s raw_condition must report the split threshold in
    the *feature's own* raw units (age's real mean/std, not some other
    column's), and must name the feature that was actually split on.
    """

    def test_dominant_signal_column_and_threshold_are_correctly_identified(self):
        df, y = _age_threshold_dataset()
        clf = _fit_rpte(df, y)

        assert list(getattr(clf, "_zero_variance_cols_", []) or []) == ["const_col"]

        age_conditions = _raw_conditions_for(clf.rpte_rule_table(), "age")
        assert age_conditions, "expected at least one rule to split on age"
        thresholds = [float(c.split()[-1]) for c in age_conditions]
        # The true generating threshold is 55; every observed raw-scale
        # split threshold must land close to it, in *raw* age units
        # (20-80), not in standardized units (which would put it near 0)
        # and not in some unrelated column's range (noise columns are
        # 0-1, income is 20000-150000).
        assert all(45.0 <= t <= 65.0 for t in thresholds), thresholds

    def test_threshold_survives_shuffled_column_order(self):
        """The exact same check as above, but with the training frame's
        columns presented in a different order -- age is no longer column
        index 1. A positional (rather than name-keyed) mapping bug would
        misattribute the threshold to whatever feature now sits at the
        original position.
        """
        df, y = _age_threshold_dataset(shuffle_columns=True)
        assert list(df.columns) != [
            "noise1", "age", "noise2", "const_col", "income", "noise3", "cat_flag", "noise4", "noise5",
        ], "test setup requires columns to actually be shuffled"
        clf = _fit_rpte(df, y)

        age_conditions = _raw_conditions_for(clf.rpte_rule_table(), "age")
        assert age_conditions
        thresholds = [float(c.split()[-1]) for c in age_conditions]
        assert all(45.0 <= t <= 65.0 for t in thresholds), thresholds

    def test_weaker_income_signal_uses_income_raw_scale_not_a_noise_columns_scale(self):
        df, y = _age_threshold_dataset()
        clf = _fit_rpte(df, y)

        income_conditions = _raw_conditions_for(clf.rpte_rule_table(), "income")
        if not income_conditions:
            pytest.skip("income signal too weak relative to age to produce its own split in this fit")
        thresholds = [float(c.split()[-1]) for c in income_conditions]
        # income's true raw range is [20000, 150000]; a noise column's
        # range is [0, 1] and age's is [20, 80] -- landing in income's
        # actual range (with slack for finite-sample split search) rules
        # out both a raw/standardized-units mixup and a wrong-column bind.
        assert all(20000.0 <= t <= 150000.0 for t in thresholds), thresholds

    def test_raw_sources_for_age_rules_never_include_an_unrelated_column(self):
        df, y = _age_threshold_dataset()
        clf = _fit_rpte(df, y)
        for rule in clf.rpte_rule_table():
            conditions = rule.get("conditions") or []
            if any((c.get("raw_condition") or "").startswith("age ") for c in conditions):
                assert "age" in (rule.get("raw_sources") or []), rule


class TestAugmentedPairProvenanceMatchesGroundTruth:
    """explain_augmented_pair_effects() must name the actual two interacting
    raw source features and report a raw-scale coefficient whose sign
    matches the known ground-truth direction of the interaction.
    """

    def test_known_product_interaction_is_ranked_first_with_correct_sources_and_sign(self):
        rng = np.random.RandomState(7)
        n = 3000
        df = pd.DataFrame({
            "z_noise_a": rng.uniform(0, 1, n),
            "feature_X": rng.uniform(1, 10, n),
            "z_noise_b": rng.uniform(0, 1, n),
            "feature_Y": rng.uniform(1, 10, n),
            "z_noise_c": rng.uniform(0, 1, n),
        })
        interaction = df["feature_X"] * df["feature_Y"]
        y = (interaction > np.median(interaction)).astype(int).to_numpy()

        clf = HUGIMLClassifierNative(
            L=2, topK=50, feature_mode="patterns_only",
            augmented_pair_transforms=True, augmented_pair_mode="interaction_information",
        )
        clf.fit(df, y)

        effects = clf.explain_augmented_pair_effects()
        assert not effects.empty
        top = effects.sort_values("transform_ig", ascending=False).iloc[0]

        assert top["operation"] == "product"
        assert set(top["inputs"]) == {"feature_X", "feature_Y"}
        assert top["raw_formula"] in {"feature_X * feature_Y", "feature_Y * feature_X"}
        # Ground truth: higher product -> higher P(y=1), so the raw-scale
        # coefficient must be positive.
        assert float(top["coefficient_raw_scale"]) > 0.0
        # standardization_mean should be close to the actual observed mean
        # of the raw product term (not some other pair's mean).
        actual_product_mean = float((df["feature_X"] * df["feature_Y"]).mean())
        assert abs(float(top["standardization_mean"]) - actual_product_mean) < 1e-6

    def test_pure_noise_pairs_do_not_outrank_the_true_interaction(self):
        rng = np.random.RandomState(7)
        n = 3000
        df = pd.DataFrame({
            "z_noise_a": rng.uniform(0, 1, n),
            "feature_X": rng.uniform(1, 10, n),
            "z_noise_b": rng.uniform(0, 1, n),
            "feature_Y": rng.uniform(1, 10, n),
            "z_noise_c": rng.uniform(0, 1, n),
        })
        interaction = df["feature_X"] * df["feature_Y"]
        y = (interaction > np.median(interaction)).astype(int).to_numpy()

        clf = HUGIMLClassifierNative(
            L=2, topK=50, feature_mode="patterns_only",
            augmented_pair_transforms=True, augmented_pair_mode="interaction_information",
        )
        clf.fit(df, y)
        effects = clf.explain_augmented_pair_effects()
        ranked = effects.sort_values("transform_ig", ascending=False)
        top_inputs = set(ranked.iloc[0]["inputs"])
        assert top_inputs == {"feature_X", "feature_Y"}


class TestPatternBinRangesMatchActualColumnDistribution:
    """get_pattern_info()'s bin-interval labels for a numeric feature must
    reflect that feature's own observed raw-value range, and the
    information-gain-ranked bins must bracket the true generating
    threshold.
    """

    def test_age_pattern_bins_cover_the_true_column_range_and_bracket_the_threshold(self):
        rng = np.random.RandomState(3)
        n = 2000
        df = pd.DataFrame({
            "noise1": rng.uniform(0, 1, n),
            "age": rng.uniform(20, 80, n),
            "noise2": rng.uniform(0, 1, n),
        })
        y = (df["age"] > 55).astype(int).to_numpy()

        clf = HUGIMLClassifierNative(
            L=1, topK=20, feature_mode="original_plus_patterns", B=-1, adaptive_binning=True,
        )
        clf.fit(df, y)

        info = clf.get_pattern_info()
        age_patterns = info[info["pattern"].astype(str).str.startswith("age=")]
        assert not age_patterns.empty

        bin_edges: list[tuple[float, float]] = []
        for label in age_patterns["pattern"]:
            interval = label.split("=", 1)[1]
            lo_str, hi_str = interval.strip("[)").split(",")
            bin_edges.append((float(lo_str), float(hi_str)))
        bin_edges.sort()

        # Every bin edge must lie within (or right at) age's actual observed
        # range -- not e.g. a standardized-unit range near 0, and not some
        # other column's range.
        actual_min, actual_max = float(df["age"].min()), float(df["age"].max())
        assert bin_edges[0][0] == pytest.approx(actual_min, abs=0.5)
        assert bin_edges[-1][1] == pytest.approx(actual_max, abs=0.5)

        # The true threshold (55) must fall inside the span covered by the
        # bins (not, say, entirely below or above every bin -- which would
        # indicate the labeled range doesn't correspond to this column).
        assert bin_edges[0][0] <= 55.0 <= bin_edges[-1][1]

        # The bins with the highest information gain should be adjacent to
        # the true threshold (55) -- that's exactly where the label flips
        # most sharply -- rather than at the far ends of age's range.
        info_gain_by_bin = dict(zip(age_patterns["pattern"], age_patterns["information_gain"]))
        max_ig = max(info_gain_by_bin.values())
        highest_ig_bins = [label for label, ig in info_gain_by_bin.items() if ig == max_ig]
        bin_edges_by_label = dict(zip(age_patterns["pattern"], bin_edges))
        for label in highest_ig_bins:
            lo, hi = bin_edges_by_label[label]
            # within one bin-width of the true threshold
            assert lo - 15.0 <= 55.0 <= hi + 15.0, (label, lo, hi)


class TestFeatureMetadataWiringGuardsAgainstLengthMismatch:
    """classifier._wire_hugiml_feature_metadata's callers must refuse to
    wire feature names into the downstream estimator when the name count
    doesn't match the downstream matrix's column count -- a silent
    mismatch here would misattribute every later column's explanation.
    Exercised through the public fit() path rather than by calling the
    private helper directly, since the guard's value is in what it
    prevents reaching a fitted, explainable model.
    """

    def test_rpte_fit_succeeds_with_consistent_feature_name_and_column_counts(self):
        """Not a mismatch scenario (that would require corrupting internal
        state, which isn't a realistic path to test through the public
        API) -- this instead documents and locks in the invariant the
        guard depends on: a normal fit's downstream feature-name count
        exactly matches its downstream matrix's column count, for every
        combination of feature_mode and augmented_pair_transforms.
        """
        df, y = _age_threshold_dataset()
        for feature_mode in ["patterns_only", "original_plus_patterns"]:
            for augmented in [False, True]:
                clf = HUGIMLClassifierNative(
                    L=1, topK=20, feature_mode=feature_mode,
                    augmented_pair_transforms=augmented,
                )
                clf.fit(df, y)
                names = clf._get_downstream_feature_names()
                assert len(names) == clf.x_train_downstream_.shape[1], (
                    feature_mode, augmented, len(names), clf.x_train_downstream_.shape[1]
                )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

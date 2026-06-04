import numpy as np
import pandas as pd

from hugiml.classifier import HUGIMLClassifierNative
from hugiml.pruning import PatternEditor
from hugiml.serialization import load_model, save_model


def _interaction_frame(n=260, seed=123):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(6)})
    y = (
        (X["f0"] * X["f1"] + 0.7 * X["f2"] - 0.4 * X["f3"] + rng.normal(scale=0.35, size=n)) > 0
    ).astype(int)
    return X, y.to_numpy()


def _fit_mixed(strict=False):
    X, y = _interaction_frame()
    clf = HUGIMLClassifierNative(
        B=-1,
        adaptive_binning=True,
        L=2,
        G=1e-3,
        topK=20,
        feature_mode="original_plus_patterns",
        augmented_pair_transforms=True,
        augmented_pair_max_features=6,
        topk_budget_strict=strict,
        max_fit_seconds=5,
    )
    clf.fit(X, y)
    return clf, X, y


def test_feature_importances_align_with_patterns_originals_and_augmented_pairs():
    clf, _, _ = _fit_mixed(strict=False)
    names = clf.get_downstream_features()
    coef_count = clf.model_.named_steps["clf"].coef_.ravel().shape[0]
    assert len(names) == coef_count
    assert any(name.startswith("orig:") for name in names)
    assert any(name.startswith("pattern:") for name in names)
    assert any(name.startswith("augmented_pair:") for name in names)

    imp = clf.feature_importances()
    assert set(imp["feature"]) == set(names)
    assert {"original", "pattern", "augmented_pair"}.issubset(set(imp["feature_type"]))
    assert not imp["coefficient"].isna().any()
    assert {"pattern_support", "support_type", "non_missing_rate", "variance"}.issubset(imp.columns)

    pattern_labels = set(clf.get_hug_features())
    pattern_rows = imp[imp["feature_type"] == "pattern"]
    non_pattern_rows = imp[imp["feature_type"] != "pattern"]
    assert set(pattern_rows["pattern"]).issubset(pattern_labels)
    assert pattern_rows["pattern_support"].between(0.0, 1.0).all()
    assert (pattern_rows["support_type"] == "pattern_support").all()
    assert non_pattern_rows["pattern_support"].isna().all()
    assert (non_pattern_rows["support_type"] == "not_applicable").all()
    assert imp["non_missing_rate"].between(0.0, 1.0).all()


def test_strict_feature_importances_respect_filtered_downstream_names():
    clf, _, _ = _fit_mixed(strict=True)
    names = clf.get_downstream_features()
    assert len(names) <= clf.topK
    assert len(names) == clf.model_.named_steps["clf"].coef_.ravel().shape[0]

    imp = clf.feature_importances()
    assert set(imp["feature"]) == set(names)
    assert len(imp) == len(names)
    assert not imp["strict_topk_score"].isna().any()
    assert set(imp["feature_type"]).issubset({"original", "pattern", "augmented_pair"})


def test_plots_and_pruning_handle_mixed_feature_spaces():
    pytest = __import__("pytest")
    pytest.importorskip("plotly")
    from hugiml.plots import HUGPlotter

    clf, X, y = _fit_mixed(strict=True)
    plotter = HUGPlotter(clf)
    assert plotter.plot_top_patterns(top_n=5) is not None
    assert plotter.plot_active_patterns(X, sample_idx=0, max_patterns=5) is not None

    editor = PatternEditor(clf)
    listed = editor.list_patterns()
    assert {"idx", "pattern", "coefficient", "support"}.issubset(listed.columns)
    if len(listed) > 0:
        editor.remove([0], reason="test removal")
    editor.refit(X, y)
    pruned = editor.finalize()
    pred = pruned.predict(X.iloc[:10])
    assert pred.shape == (10,)
    assert len(pruned.get_downstream_features()) <= pruned.topK
    pruned.feature_importances()


def test_interpretability_metadata_round_trips_through_serialization(tmp_path):
    clf, X, _ = _fit_mixed(strict=False)
    before = clf.feature_importances().sort_values("feature").reset_index(drop=True)

    path = tmp_path / "mixed_interpretability.hugiml"
    save_model(clf, path)
    loaded = load_model(path)
    after = loaded.feature_importances().sort_values("feature").reset_index(drop=True)

    assert loaded.get_downstream_features() == clf.get_downstream_features()
    np.testing.assert_allclose(
        loaded.predict_proba(X.iloc[:25]), clf.predict_proba(X.iloc[:25]), rtol=0, atol=0
    )
    assert before["feature"].tolist() == after["feature"].tolist()
    assert before["feature_type"].tolist() == after["feature_type"].tolist()
    assert before["support_type"].tolist() == after["support_type"].tolist()
    np.testing.assert_allclose(
        before["pattern_support"].fillna(-1).to_numpy(),
        after["pattern_support"].fillna(-1).to_numpy(),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        before["non_missing_rate"].fillna(-1).to_numpy(),
        after["non_missing_rate"].fillna(-1).to_numpy(),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        before["variance"].fillna(-1).to_numpy(),
        after["variance"].fillna(-1).to_numpy(),
        rtol=0,
        atol=0,
    )
    assert after.loc[after["feature_type"] != "pattern", "pattern_support"].isna().all()


def test_strict_interpretability_metadata_round_trips_through_serialization(tmp_path):
    clf, X, _ = _fit_mixed(strict=True)
    before = clf.feature_importances().sort_values("feature").reset_index(drop=True)

    path = tmp_path / "strict_mixed_interpretability.hugiml"
    save_model(clf, path)
    loaded = load_model(path)
    after = loaded.feature_importances().sort_values("feature").reset_index(drop=True)

    assert loaded.get_downstream_features() == clf.get_downstream_features()
    assert len(loaded.get_downstream_features()) <= loaded.topK
    np.testing.assert_allclose(
        loaded.predict_proba(X.iloc[:25]), clf.predict_proba(X.iloc[:25]), rtol=0, atol=0
    )
    assert before["feature"].tolist() == after["feature"].tolist()
    np.testing.assert_allclose(
        before["pattern_support"].fillna(-1).to_numpy(),
        after["pattern_support"].fillna(-1).to_numpy(),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        before["non_missing_rate"].fillna(-1).to_numpy(),
        after["non_missing_rate"].fillna(-1).to_numpy(),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        before["variance"].fillna(-1).to_numpy(),
        after["variance"].fillna(-1).to_numpy(),
        rtol=0,
        atol=0,
    )


def test_augmented_pair_standardization_is_public_and_serialized(tmp_path):
    clf, _, _ = _fit_mixed(strict=False)
    catalog = clf.get_augmented_pair_transforms()
    assert catalog
    first = catalog[0]
    assert "standardization_mean" in first
    assert "standardization_scale" in first
    assert "standardized_formula" in first
    assert "source_observed_medians" in first
    assert first["pair_missing_policy"] == "reference_value_for_unavailable_pair"
    assert "eligible_rate" in first
    assert np.isfinite(first["standardization_mean"])
    assert np.isfinite(first["standardization_scale"])
    assert first["standardization_scale"] > 0

    std_df = clf.get_augmented_pair_standardization()
    assert len(std_df) == len(catalog)
    assert {"name", "raw_formula", "standardization_mean", "standardization_scale"}.issubset(
        std_df.columns
    )

    imp = clf.feature_importances()
    aug_imp = imp[imp["feature_type"] == "augmented_pair"]
    assert not aug_imp.empty
    assert aug_imp["standardization_mean"].notna().all()
    assert aug_imp["standardization_scale"].notna().all()
    assert aug_imp["standardized_formula"].notna().all()

    path = tmp_path / "aug_std.hugiml"
    save_model(clf, path)
    loaded = load_model(path)
    assert (
        loaded.get_augmented_pair_transforms()[0]["standardization_mean"]
        == first["standardization_mean"]
    )
    assert (
        loaded.get_augmented_pair_transforms()[0]["standardization_scale"]
        == first["standardization_scale"]
    )


def test_pattern_editor_exposes_full_downstream_context_but_edits_patterns_only():
    clf, _, _ = _fit_mixed(strict=False)
    editor = PatternEditor(clf)
    patterns = editor.list_patterns()
    assert (patterns["feature_type"] == "pattern").all()
    assert patterns["editable"].all()

    downstream = editor.list_downstream_features()
    assert {"feature", "feature_type", "editable", "editor_scope"}.issubset(downstream.columns)
    assert set(downstream["feature_type"]).issuperset({"original", "pattern", "augmented_pair"})
    assert downstream.loc[downstream["feature_type"] == "pattern", "editable"].all()
    assert not downstream.loc[downstream["feature_type"] != "pattern", "editable"].any()

    diff = editor.diff()
    assert diff["scope"] == "hug_patterns_only"
    assert diff["n_downstream_features_current"] == len(downstream)
    assert diff["n_non_editable_downstream_features_current"] > 0


def test_augmented_pair_raw_scale_effects_are_exposed_and_correct():
    clf, _, _ = _fit_mixed(strict=False)
    effects = clf.explain_augmented_pair_effects()
    assert not effects.empty
    required = {
        "feature",
        "raw_formula",
        "standardized_formula",
        "standardization_mean",
        "standardization_scale",
        "coefficient_standardized",
        "coefficient_raw_scale",
        "one_raw_unit_effect_on_log_odds",
        "one_std_effect_on_log_odds",
        "decision_direction",
        "risk_increases_when",
        "unit_effect_interpretation",
        "reference_raw_value",
        "reference_raw_value_description",
        "source_observed_medians",
        "source_observed_medians_description",
        "pair_missing_policy",
        "pair_missing_policy_description",
        "eligible_count",
        "eligible_rate",
        "missing_pair_rate",
        "raw_scale_note",
        "raw_interpretation",
    }
    assert required.issubset(effects.columns)
    assert effects["standardization_scale"].gt(0).all()
    np.testing.assert_allclose(
        effects["coefficient_raw_scale"].to_numpy(),
        effects["coefficient_standardized"].to_numpy()
        / effects["standardization_scale"].to_numpy(),
        rtol=1e-5,
        atol=1e-8,
    )
    assert (
        effects["raw_interpretation"].str.contains("downstream estimator uses", regex=False).all()
    )
    assert effects["raw_interpretation"].str.contains("training-cohort mean", regex=False).all()
    assert (
        effects["raw_interpretation"].str.contains("neutral standardized value", regex=False).all()
    )
    assert (
        effects["raw_interpretation"]
        .str.contains("native missing-value handling", regex=False)
        .all()
    )
    assert (
        effects["reference_raw_value_description"]
        == "training_cohort_mean_of_observed_raw_pair_value"
    ).all()
    assert (
        effects["source_observed_medians_description"]
        == "per-source-feature observed medians for diagnostics only; not used to construct pair values"
    ).all()
    assert (effects["pair_missing_policy"] == "reference_value_for_unavailable_pair").all()
    assert effects["eligible_rate"].between(0, 1).all()
    assert effects["missing_pair_rate"].between(0, 1).all()
    prod = effects[effects["operation"] == "product"]
    if not prod.empty:
        assert (
            prod["unit_effect_interpretation"]
            .str.contains("does not have a fixed marginal effect", regex=False)
            .all()
        )
    absdiff = effects[effects["operation"] == "absolute_difference"]
    if not absdiff.empty:
        assert (
            absdiff["unit_effect_interpretation"]
            .str.contains("absolute difference term", regex=False)
            .all()
        )

    imp = clf.feature_importances()
    aug_imp = imp[imp["feature_type"] == "augmented_pair"]
    assert not aug_imp.empty
    assert {
        "coefficient_standardized",
        "coefficient_raw_scale",
        "one_raw_unit_effect_on_log_odds",
        "decision_direction",
        "risk_increases_when",
        "unit_effect_interpretation",
        "reference_raw_value_description",
        "source_observed_medians_description",
        "pair_missing_policy_description",
        "raw_scale_note",
        "raw_interpretation",
    }.issubset(aug_imp.columns)
    assert aug_imp["raw_interpretation"].notna().all()
    assert aug_imp["raw_interpretation"].str.contains("training-cohort mean", regex=False).all()


def test_augmented_pair_raw_scale_effects_round_trip_through_serialization(tmp_path):
    clf, _, _ = _fit_mixed(strict=False)
    before = clf.explain_augmented_pair_effects().sort_values("feature").reset_index(drop=True)
    path = tmp_path / "aug_raw_effects.hugiml"
    save_model(clf, path)
    loaded = load_model(path)
    after = loaded.explain_augmented_pair_effects().sort_values("feature").reset_index(drop=True)

    assert before["feature"].tolist() == after["feature"].tolist()
    for col in [
        "standardization_mean",
        "standardization_scale",
        "coefficient_standardized",
        "coefficient_raw_scale",
        "one_raw_unit_effect_on_log_odds",
    ]:
        np.testing.assert_allclose(
            before[col].to_numpy(), after[col].to_numpy(), rtol=1e-5, atol=1e-8
        )
    assert (
        before["unit_effect_interpretation"].tolist()
        == after["unit_effect_interpretation"].tolist()
    )
    assert before["raw_scale_note"].tolist() == after["raw_scale_note"].tolist()
    assert before["raw_interpretation"].tolist() == after["raw_interpretation"].tolist()


def test_model_summary_and_fit_metadata_expose_hybrid_composition():
    clf, _, _ = _fit_mixed(strict=False)
    summary = clf.model_summary()
    composition = clf.get_model_composition()
    assert composition["downstream_feature_counts"]["augmented_pair"] > 0
    assert "Augmented pairs:" in summary
    assert "Downstream composition:" in summary
    assert "Top 10 downstream features by importance:" in summary
    assert "explain_augmented_pair_effects()" in summary
    assert (
        clf.fit_metadata_.n_augmented_pairs
        == composition["downstream_feature_counts"]["augmented_pair"]
    )
    assert (
        clf.fit_metadata_.n_downstream_features == composition["downstream_feature_counts"]["total"]
    )
    assert "augmented pairs" in clf.fit_metadata_.summary()


def test_explainability_report_includes_hybrid_metadata_and_pair_interpretation():
    from hugiml.explainability import HUGPatternExplainer

    clf, _, _ = _fit_mixed(strict=False)
    report = HUGPatternExplainer(clf).generate_report(top_n=20)
    assert report.model_composition["downstream_feature_counts"]["augmented_pair"] > 0
    assert report.augmented_pair_effects
    assert any(row.get("feature_type") == "augmented_pair" for row in report.top_patterns)
    aug_rows = [row for row in report.top_patterns if row.get("feature_type") == "augmented_pair"]
    assert aug_rows
    assert "raw_interpretation" in aug_rows[0]
    assert "coefficient_raw_scale" in aug_rows[0]
    assert "pair_missing_policy_description" in aug_rows[0]
    as_json = report.to_json()
    assert "model_composition" in as_json
    assert "augmented_pair_effects" in as_json
    assert "raw_interpretation" in as_json


def test_feature_lineage_includes_augmented_pair_source_contributions():
    from hugiml.explainability import HUGPatternExplainer

    clf, _, _ = _fit_mixed(strict=False)
    lineage = HUGPatternExplainer(clf).feature_lineage()
    assert any(fl.derived_augmented_pairs for fl in lineage)
    for fl in lineage:
        assert fl.total_importance >= fl.pattern_importance
        assert fl.total_importance >= fl.augmented_pair_importance
    report = HUGPatternExplainer(clf).generate_report(top_n=10)
    assert any(row["n_augmented_pairs"] > 0 for row in report.feature_lineage)

"""Runtime-independent causal dashboard model and data tests."""

from __future__ import annotations

import base64
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest


def _tiny_grid():
    return {
        "B": [-1],
        "adaptive_binning": [True],
        "L": [1],
        "topK": [30],
        "feature_mode": ["original_plus_patterns"],
        "G": [0.01],
        "convert_binary_to_categorical": [False],
        "augmented_pair_transforms": [False],
        "topk_budget_strict": [False],
        "lr_solver": ["adaptive_l1"],
        "lr_C": [0.5],
        "base_estimator": [None],
    }


def _upload(data: bytes, mime: str = "text/csv") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def test_all_demo_metadata_valid():
    from hugiml.causal_dashboard.data import demo_catalog, load_demo
    from hugiml.causal_dashboard.metadata import validate_causal_metadata

    for key in demo_catalog():
        df, meta = load_demo(key)
        out = validate_causal_metadata(meta, df.columns)
        assert out["treatment"] in df.columns
        assert out["outcome"] in df.columns


def test_metadata_cycle_unknown_column_and_duplicate_role_rejected():
    from hugiml.causal_dashboard.metadata import validate_causal_metadata

    cyc = {
        "treatment": "T",
        "outcome": "Y",
        "adjustment_set": [],
        "nodes": [{"id": "T", "role": "treatment"}, {"id": "Y", "role": "outcome"}],
        "edges": [["T", "Y"], ["Y", "T"]],
    }
    with pytest.raises(ValueError, match="cycle"):
        validate_causal_metadata(cyc, ["T", "Y"])
    missing = {"treatment": "T", "outcome": "Y", "adjustment_set": ["Z"], "nodes": [], "edges": []}
    with pytest.raises(ValueError, match="not present"):
        validate_causal_metadata(missing, ["T", "Y"])
    duplicate_treatment = {
        "treatment": "T",
        "outcome": "Y",
        "adjustment_set": [],
        "nodes": [
            {"id": "T", "role": "treatment"},
            {"id": "T2", "role": "treatment"},
            {"id": "Y", "role": "outcome"},
        ],
        "edges": [["T", "Y"]],
    }
    with pytest.raises(ValueError, match="exactly one treatment"):
        validate_causal_metadata(duplicate_treatment, ["T", "T2", "Y"])
    mismatched = {
        "treatment": "T2",
        "outcome": "Y",
        "adjustment_set": [],
        "nodes": [
            {"id": "T", "role": "treatment"},
            {"id": "T2", "role": "unknown"},
            {"id": "Y", "role": "outcome"},
        ],
        "edges": [["T", "Y"]],
    }
    with pytest.raises(ValueError, match="declared treatment"):
        validate_causal_metadata(mismatched, ["T", "T2", "Y"])


def test_demo_shapes_binary_roles_and_compact_dag():
    from hugiml.causal_dashboard.data import load_demo
    from hugiml.causal_dashboard.metadata import dag_figure

    for key in ["credit_intervention", "claims_outreach", "collections_overlap"]:
        df, meta = load_demo(key)
        assert len(df) >= 3000
        assert set(df[meta["treatment"]].unique()) == {0, 1}
        assert set(df[meta["outcome"]].unique()) == {0, 1}
        assert meta["oracle_tau_column"] in df.columns
        fig = dag_figure(meta, theme="Dark")
        assert fig.layout.height <= 300
        assert len(fig.data) < len(meta["nodes"])
        hover_values = [
            str(trace.customdata)
            for trace in fig.data
            if getattr(trace, "customdata", None) is not None
        ]
        assert any("Baseline covariates" in value for value in hover_values)


def test_demo_roles_are_locked_to_metadata():
    from hugiml.causal_dashboard.data import dataset_roles, load_demo
    from hugiml.causal_dashboard.metadata import validate_causal_metadata

    df, metadata = load_demo("credit_intervention")
    metadata = validate_causal_metadata(metadata, df.columns)
    info = dataset_roles(df, metadata, metadata_backed=True)
    assert info["metadata_backed"] is True
    assert info["treatment_options"] == [info["treatment"]]
    assert info["outcome_options"] == [info["outcome"]]


def test_upload_validation_and_duplicate_headers():
    from hugiml.causal_dashboard.data import read_data_upload, read_metadata_upload

    frame = read_data_upload(_upload(b"a,b\n1,0\n2,1\n"), "x.csv")
    assert list(frame.columns) == ["a", "b"]
    with pytest.raises(ValueError, match="duplicate column"):
        read_data_upload(_upload(b"a,a\n1,0\n"), "x.csv")
    with pytest.raises(ValueError, match="base64"):
        read_data_upload("data:text/csv;base64,%%%", "x.csv")
    with pytest.raises(ValueError, match="empty"):
        read_data_upload(_upload(b""), "x.csv")
    with pytest.raises(ValueError, match="JSON object"):
        read_metadata_upload(_upload(b"[]", "application/json"), "meta.json")


def test_upload_size_limit(monkeypatch):
    import hugiml.causal_dashboard.data as data_module

    monkeypatch.setattr(data_module, "_MAX_UPLOAD_BYTES", 10)
    with pytest.raises(ValueError, match="100 MB limit"):
        data_module.read_data_upload(_upload(b"a,b\n12345678901,1\n"), "x.csv")


@pytest.fixture(scope="module")
def small_analysis():
    from hugiml.causal_dashboard.data import load_demo
    from hugiml.causal_dashboard.modeling import fit_analysis

    df, meta = load_demo("claims_outreach")
    df = df.sample(900, random_state=9).reset_index(drop=True)
    return fit_analysis(
        df,
        treatment=meta["treatment"],
        outcome=meta["outcome"],
        covariates=meta["adjustment_set"],
        baselines=["T-LR"],
        random_state=9,
        hug_grid=_tiny_grid(),
        selection_scoring="neg_log_loss",
    )


def test_t_hug_shared_and_baseline_fit(small_analysis):
    assert set(small_analysis["results"]) == {"T-HUG", "T-LR"}
    result = small_analysis["results"]["T-HUG"]
    assert np.isfinite(result.tau).all()
    assert np.all((result.p0 > 0) & (result.p0 < 1))
    assert np.all((result.p1 > 0) & (result.p1 < 1))
    t_hug = result.model
    assert t_hug.control_model_ is not t_hug.treatment_model_
    assert (
        t_hug.control_model_._shared_supervised_vocabulary
        is t_hug.treatment_model_._shared_supervised_vocabulary
    )
    assert t_hug.control_model_.execution_mode == "audit"
    assert t_hug.treatment_model_.execution_mode == "audit"
    assert small_analysis["selection_scoring"] == "neg_log_loss"


def test_auc_and_log_loss_are_forwarded_even_when_winner_is_same():
    from hugiml.causal_dashboard.data import load_demo
    from hugiml.causal_dashboard.modeling import fit_analysis

    df, meta = load_demo("claims_outreach")
    df = df.sample(900, random_state=19).reset_index(drop=True)
    analyses = {
        scoring: fit_analysis(
            df,
            treatment=meta["treatment"],
            outcome=meta["outcome"],
            covariates=meta["adjustment_set"],
            baselines=[],
            random_state=19,
            hug_grid=_tiny_grid(),
            selection_scoring=scoring,
        )
        for scoring in ("roc_auc", "neg_log_loss")
    }
    assert analyses["roc_auc"]["selection_scoring"] == "roc_auc"
    assert analyses["neg_log_loss"]["selection_scoring"] == "neg_log_loss"
    auc_scores = analyses["roc_auc"]["results"]["T-HUG"].model.best_scores_
    log_scores = analyses["neg_log_loss"]["results"]["T-HUG"].model.best_scores_
    assert all(0 <= score <= 1 for score in auc_scores.values())
    assert all(score <= 0 for score in log_scores.values())


def test_region_selection_and_comparison_tables(small_analysis):
    from hugiml.causal_dashboard.modeling import (
        comparison_frame,
        selected_configurations_frame,
        t_hug_region_frame,
        t_hug_selection_frame,
    )

    comp = comparison_frame(small_analysis)
    assert {"Model", "ATE", "Held-out Brier"}.issubset(comp.columns)
    regions = t_hug_region_frame(small_analysis, max_patterns_per_group=5)
    assert isinstance(regions, pd.DataFrame)
    if not regions.empty:
        assert {"Region / HUG pattern", "Mean CATE", "Structure"}.issubset(regions.columns)
    selection = t_hug_selection_frame(small_analysis)
    assert set(selection["Group"]) == {"Control", "Treatment"}
    assert "Downstream model" in selection.columns
    assert list(selection.columns[:2]) == ["Group", "Downstream model"]
    configs = selected_configurations_frame(small_analysis)
    assert set(configs["Group"]) == {"Control", "Treatment"}
    assert set(configs["Model"]) == set(small_analysis["results"])
    assert {"Best score", "Candidates evaluated", "Selected parameters"}.issubset(configs.columns)
    t_hug_configs = configs.loc[configs["Model"] == "T-HUG"]
    assert t_hug_configs["Selected parameters"].str.contains('"downstream"').all()
    assert not t_hug_configs["Selected parameters"].str.contains("LogisticRegression").any()


def test_oracle_alignment_and_safe_constant_correlation():
    from hugiml.causal_dashboard.data import load_demo
    from hugiml.causal_dashboard.modeling import comparison_frame, fit_analysis

    df, meta = load_demo("claims_outreach")
    df = df.sample(850, random_state=14).reset_index(drop=True)
    df.loc[[3, 21], meta["treatment"]] = np.nan
    analysis = fit_analysis(
        df,
        treatment=meta["treatment"],
        outcome=meta["outcome"],
        covariates=meta["adjustment_set"],
        baselines=["T-LR"],
        random_state=14,
        hug_grid=_tiny_grid(),
    )
    oracle = df.loc[analysis["row_index"], meta["oracle_tau_column"]].to_numpy(dtype=float)
    comp = comparison_frame(analysis, oracle)
    assert "Oracle CATE RMSE" in comp.columns
    constant = np.zeros(len(analysis["frame"]), dtype=float)
    comp_constant = comparison_frame(analysis, constant)
    assert comp_constant["Oracle CATE corr"].isna().all()


def test_diagnostics_randomized_overlap_and_input_checks():
    from hugiml.causal_dashboard.data import load_demo
    from hugiml.causal_dashboard.diagnostics import diagnostic_summary

    claims, meta = load_demo("claims_outreach")
    summary = diagnostic_summary(
        claims, meta["treatment"], meta["outcome"], meta["adjustment_set"], meta["study_type"]
    )
    assert np.std(summary["propensity"]) < 1e-12
    assert summary["overlap_rate"] == 1.0

    collections, meta2 = load_demo("collections_overlap")
    summary2 = diagnostic_summary(
        collections,
        meta2["treatment"],
        meta2["outcome"],
        meta2["adjustment_set"],
        meta2["study_type"],
    )
    assert summary2["overlap_rate"] < 0.98
    assert summary2["extreme_high"] > 0.02

    with pytest.raises(ValueError, match="Missing diagnostic columns"):
        diagnostic_summary(
            claims, meta["treatment"], meta["outcome"], ["not_a_column"], meta["study_type"]
        )
    broken = claims.copy()
    broken[meta["treatment"]] = 1
    with pytest.raises(ValueError, match="both 0 and 1"):
        diagnostic_summary(
            broken, meta["treatment"], meta["outcome"], meta["adjustment_set"], meta["study_type"]
        )


def test_all_baseline_adapters_smoke():
    from hugiml.causal_dashboard.data import load_demo
    from hugiml.causal_dashboard.modeling import fit_analysis

    df, meta = load_demo("credit_intervention")
    df = df.sample(1000, random_state=12).reset_index(drop=True)
    baselines = ["T-LR"]
    if find_spec("xgboost") is not None:
        baselines.append("T-XGB")
    if find_spec("lightgbm") is not None:
        baselines.append("T-LGBM")
    analysis = fit_analysis(
        df,
        treatment=meta["treatment"],
        outcome=meta["outcome"],
        covariates=meta["adjustment_set"],
        baselines=baselines,
        random_state=12,
        hug_grid=_tiny_grid(),
    )
    assert set(analysis["results"]) == {"T-HUG", *baselines}
    for name in baselines:
        assert analysis["results"][name].details["candidate_count"] == 16
        assert len(analysis["results"][name].details["groups"]) == 2


def test_baseline_candidate_budget_tracks_hug_grid_family():
    from sklearn.model_selection import ParameterGrid

    from hugiml.causal_dashboard.modeling import baseline_grid

    for name in ("T-LR", "T-XGB", "T-LGBM"):
        assert len(list(ParameterGrid(baseline_grid(name, "quick")))) == 1
        assert len(list(ParameterGrid(baseline_grid(name, "performance")))) == 8
        assert len(list(ParameterGrid(baseline_grid(name, "interpretability")))) == 8
        assert len(list(ParameterGrid(baseline_grid(name, "performance_ho")))) == 16
        assert len(list(ParameterGrid(baseline_grid(name, "interpretability_ho")))) == 16


def test_quick_hug_grid_has_one_candidate():
    from sklearn.model_selection import ParameterGrid

    from hugiml.causal_dashboard.modeling import hug_grid_config

    grid = hug_grid_config("quick")
    assert len(list(ParameterGrid(grid))) == 1
    assert len(grid["base_estimator"]) == 1


def test_repeated_cross_fitting_produces_complete_robustness_tables():
    from hugiml.causal_dashboard.data import load_demo
    from hugiml.causal_dashboard.modeling import fit_analysis

    df, meta = load_demo("claims_outreach")
    df = df.sample(700, random_state=31).reset_index(drop=True)
    analysis = fit_analysis(
        df,
        treatment=meta["treatment"],
        outcome=meta["outcome"],
        covariates=meta["adjustment_set"],
        baselines=[],
        random_state=42,
        hug_grid=_tiny_grid(),
        selection_scoring="neg_log_loss",
        robust_seeds=[42, 43, 44],
        robust_folds=2,
        study_type=meta["study_type"],
    )
    robust = analysis["robustness"]
    assert analysis["robustness_error"] is None
    assert robust["seeds"] == [42, 43, 44]
    assert robust["folds"] == 2 and robust["repeats"] == 3
    assert list(robust["estimates"]["Model"]) == ["T-HUG"]
    assert robust["estimates"].filter(regex="ATE|CI|SD").notna().all().all()
    assert list(robust["overlap_sensitivity"]["Propensity range"]) == [
        "Untrimmed",
        "0.01–0.99",
        "0.05–0.95",
        "0.10–0.90",
    ]

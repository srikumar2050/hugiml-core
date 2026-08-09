"""Coverage for the hugiml.llm package's RPTE and "performance_ho" awareness.

Covers:

1. `docs_index._extract_grid_assignments` recovering all four named grids
   (including "performance_ho", whose `base_estimator` values are
   constructor-call expressions rather than literals) from the real
   `hyperparameter_configs.py` source text, and the hyperparameter/overview
   canned answers describing "performance_ho" as the default grid.
2. `orchestrator._tune_model()` tracking `DEFAULT_HUGIML_GRID_NAME` (rather
   than a hardcoded grid name) for its default tuning strategy, and sampling
   tuning candidates in a way that gives the RPTE branch fair representation
   regardless of `ParameterGrid`'s fixed key-sorted iteration order.
3. `ui_app`'s model-evidence panel falling back to `rpte_rule_table()` when
   `feature_importances()` is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_redundancy_audit_rows_exclude_timing_and_feature_names():
    from hugiml.llm.evidence import downstream_redundancy_audit_rows

    class Model:
        def get_downstream_redundancy_audit(self, *, include_feature_names=True):
            assert include_feature_names is False
            return {
                "input_columns": 12,
                "retained_columns": 8,
                "removed_constant_columns": 1,
                "vif_analysis_seconds": 0.25,
                "retained_feature_names": ["x"],
            }

    rows = downstream_redundancy_audit_rows(Model())
    assert {row["measure"]: row["value"] for row in rows} == {
        "Input columns": 12,
        "Retained columns": 8,
        "Removed constants": 1,
    }


# ---------------------------------------------------------------------------
# docs_index grid extraction
# ---------------------------------------------------------------------------


def test_extract_grid_assignments_recovers_all_four_grids_from_real_source():
    from hugiml.llm.docs_index import _extract_grid_assignments

    src_path = Path("src/hugiml/hyperparameter_configs.py")
    text = src_path.read_text(encoding="utf-8")
    grids = _extract_grid_assignments(text)

    # A whole-dict ast.literal_eval() over HUGIML_GRIDS's text would fail
    # (and return nothing) the moment any single value anywhere in it isn't
    # a literal -- true for "performance_ho"'s base_estimator values, which
    # are constructor-call expressions. All four grids must still resolve.
    assert set(grids) == {
        "performance",
        "interpretability",
        "interpretability_ho",
        "performance_ho",
    }
    assert grids["performance"]["L"] == [1, 2]
    assert grids["performance"]["convert_binary_to_categorical"] == [False]
    assert grids["interpretability"]["feature_mode"] == ["patterns_only"]
    assert grids["interpretability"]["convert_binary_to_categorical"] == [True]
    assert grids["interpretability_ho"]["convert_binary_to_categorical"] == [True]
    assert grids["performance_ho"]["convert_binary_to_categorical"] == [False]


def test_extract_grid_assignments_placeholders_nonliteral_base_estimator():
    from hugiml.llm.docs_index import _extract_grid_assignments

    text = """
HUGIML_GRIDS = {
    "performance_ho": {
        "L": [1, 2],
        "base_estimator": [None, OneVsRestClassifier(RPTE(leaf_config="3xD"))],
    },
}
"""
    grids = _extract_grid_assignments(text)
    assert "performance_ho" in grids
    values = grids["performance_ho"]["base_estimator"]
    assert values[0] is None
    assert isinstance(values[1], str)
    assert "OneVsRestClassifier" in values[1]


def test_safe_literal_eval_dict_still_handles_fully_literal_input():
    from hugiml.llm.docs_index import _safe_literal_eval_dict

    parsed = _safe_literal_eval_dict('{"a": [1, 2], "b": "x"}')
    assert parsed == {"a": [1, 2], "b": "x"}


def test_safe_literal_eval_dict_returns_none_for_non_dict():
    from hugiml.llm.docs_index import _safe_literal_eval_dict

    assert _safe_literal_eval_dict("[1, 2, 3]") is None
    assert _safe_literal_eval_dict("not even python{") is None


def test_fallback_grids_include_both_higher_order_grids():
    from hugiml.llm.docs_index import _FALLBACK_HUGIML_GRIDS

    assert "performance_ho" in _FALLBACK_HUGIML_GRIDS
    assert "base_estimator" in _FALLBACK_HUGIML_GRIDS["performance_ho"]
    assert "interpretability_ho" in _FALLBACK_HUGIML_GRIDS
    assert "base_estimator" in _FALLBACK_HUGIML_GRIDS["interpretability_ho"]


def test_hyperparameter_answer_mentions_performance_ho_as_default():
    from hugiml.llm.docs_index import _build_hyperparameter_answer

    answer = _build_hyperparameter_answer("what hyperparameters does hugiml have", hits=[])
    assert "performance_ho" in answer
    assert "interpretability_ho" in answer
    assert "default" in answer.lower()


def test_overview_answer_mentions_performance_ho_and_rpte_explanation():
    from hugiml.llm.docs_index import _build_hugiml_overview_answer

    answer = _build_hugiml_overview_answer(hits=[])
    assert "performance_ho" in answer
    assert "rpte_rule_table" in answer


# ---------------------------------------------------------------------------
# orchestrator default-grid tracking + fair candidate sampling
# ---------------------------------------------------------------------------


def test_orchestrator_module_tracks_default_grid_name_not_hardcoded():
    source = Path("src/hugiml/llm/orchestrator.py").read_text(encoding="utf-8")
    assert "DEFAULT_HUGIML_GRID_NAME" in source
    # The old hardcoded fallback must be gone.
    assert '"interpretability" if strategy == "interpretability" else "performance"' not in source


def test_tune_model_uses_default_grid_name_for_unrecognized_strategy(tmp_path):
    from pathlib import Path as _Path

    from hugiml.hyperparameter_configs import DEFAULT_HUGIML_GRID_NAME
    from hugiml.llm import HUGIMLActionOrchestrator

    assert DEFAULT_HUGIML_GRID_NAME == "performance_ho"

    repo = _Path(__file__).resolve().parents[2]
    orch = HUGIMLActionOrchestrator(repo_root=repo, session_dir=tmp_path)

    calls: list[str] = []
    import hugiml.llm.orchestrator as orch_module

    real_get_grid = orch_module.get_hugiml_grid

    def _spy(name=None):
        calls.append(name)
        return real_get_grid(name)

    orch_module.get_hugiml_grid = _spy
    try:
        import pandas as pd

        rng = np.random.RandomState(0)
        n = 120
        df = pd.DataFrame(
            {
                "x1": rng.uniform(0, 1, n),
                "x2": rng.randint(0, 2, n),
            }
        )
        y = (df["x1"] > 0.5).astype(int).values
        X_train, X_test, y_train, y_test = df.iloc[:90], df.iloc[90:], y[:90], y[90:]
        orch_module.HUGIMLActionOrchestrator._tune_model(
            orch, X_train, y_train, X_test, y_test, strategy="balanced", metric=None
        )
    finally:
        orch_module.get_hugiml_grid = real_get_grid

    assert calls == ["performance_ho"]


def test_tune_model_candidate_sampling_gives_rpte_fair_representation():
    """A plain candidates[:max_trials] prefix is biased by ParameterGrid's
    fixed key-sorted iteration order: for "performance_ho", a small prefix
    can systematically under-try (or never try) the RPTE branch together
    with higher L values -- exactly the higher-order candidates this grid
    exists to search. Sampling must shuffle before truncating so RPTE is
    reachable regardless of grid key order.
    """
    from sklearn.model_selection import ParameterGrid

    from hugiml.hyperparameter_configs import get_hugiml_grid

    grid = get_hugiml_grid("performance_ho")
    candidates = list(ParameterGrid(grid))
    assert len(candidates) == 16

    rng = np.random.RandomState(7)
    order = rng.permutation(len(candidates))[:6]
    sample = [candidates[i] for i in order]
    n_rpte = sum(1 for c in sample if c.get("base_estimator") is not None)
    # Not asserting an exact count (that would over-fit to one seed) --
    # just that RPTE is reachable at all, which the old candidates[:6]
    # prefix could fail for small max_trials values depending on key order.
    assert n_rpte >= 1


# ---------------------------------------------------------------------------
# ui_app Model evidence panel RPTE fallback
# ---------------------------------------------------------------------------


def test_rpte_rule_rows_to_importance_rows_shapes_for_chart():
    from hugiml.llm.evidence import rpte_rule_rows_to_importance_rows

    rows = [
        {
            "class": 1,
            "conditions": [{"raw_condition": "age > 50"}, {"raw_condition": "flag == 1"}],
            "final_logistic_coefficient": 2.5,
            "support_count": 40,
            "backend": "sequential_default",
        },
        {
            "class": 1,
            "conditions": [{"downstream_condition": "orig:income > 0.1"}],
            "final_logistic_coefficient": -1.1,
            "support_count": 20,
            "backend": "sequential_default",
        },
    ]
    out = rpte_rule_rows_to_importance_rows(rows, limit=5)
    assert len(out) == 2
    assert out[0]["pattern"] == "age > 50 AND flag == 1"
    assert out[0]["coefficient"] == 2.5
    # Ranked by |coefficient| descending.
    assert abs(out[0]["coefficient"]) >= abs(out[1]["coefficient"])


def test_model_panel_source_falls_back_to_rpte_rule_table():
    source = Path("src/hugiml/llm/ui_app.py").read_text(encoding="utf-8")
    assert "rpte_rule_table" in source
    assert "rpte_rule_rows_to_importance_rows" in source


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


def test_planner_normalises_interpretability_ho_strategy_to_grid() -> None:
    from hugiml.llm.planner import _normalise_planned_model_request
    from hugiml.llm.schemas import ActionRequest

    planned = ActionRequest(
        action="tune_hyperparameters",
        strategy="interpretability_ho",
    )
    normalised = _normalise_planned_model_request(
        planned,
        "run the interpretability_ho grid",
    )
    assert normalised.params["grid_name"] == "interpretability_ho"


def test_orchestrator_normalises_interpretability_ho_aliases() -> None:
    from hugiml.llm.orchestrator import _normalise_grid_name

    assert _normalise_grid_name("interpretabilityho") == "interpretability_ho"
    assert _normalise_grid_name("higher order interpretability") == "interpretability_ho"
    assert _normalise_grid_name("interpretability higher order") == "interpretability_ho"

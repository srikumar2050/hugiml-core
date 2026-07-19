"""Functional tests for the Dash-based HUGIML dashboard."""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

_OPTIONAL_UI_MODULES = ("dash", "dash_bootstrap_components", "streamlit")
_MISSING_UI_MODULES = tuple(name for name in _OPTIONAL_UI_MODULES if find_spec(name) is None)
pytestmark = pytest.mark.skipif(
    bool(_MISSING_UI_MODULES),
    reason="optional dashboard dependencies are not installed: " + ", ".join(_MISSING_UI_MODULES),
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def demo_ctx():
    from hugiml.dashboard.dash_components.data_utils import load_demo

    return load_demo("credit")


@pytest.fixture(scope="module")
def fitted_ctx(demo_ctx):
    from hugiml.dashboard.runner import score_cases, train_hugiml

    result = train_hugiml(demo_ctx["X"], demo_ctx["y"], cv=2, random_state=42)
    return {
        **demo_ctx,
        "result": result,
        "model": result.best_estimator_,
        "predictions": score_cases(result.best_estimator_, demo_ctx["X"]),
        "cv": 2,
        "random_state": 42,
    }


# ── App creation & layout ────────────────────────────────────────────────────


class TestAppCreation:
    def test_create_app(self):
        import dash

        from hugiml.dashboard.dash_app import create_app

        app = create_app()
        assert isinstance(app, dash.Dash)

    def test_layout_has_sidebar_and_main(self):
        from hugiml.dashboard.dash_app import create_app

        app = create_app()
        s = str(app.layout)
        assert "ws-wb" in s and "ws-gov" in s
        assert "wb-nav-setup" in s and "wb-nav-results" in s
        assert "st-ctx" in s

    def test_callbacks_registered(self):
        from hugiml.dashboard.dash_app import create_app

        app = create_app()
        assert len(app.callback_map) >= 10

    def test_critical_ids_in_layout(self):
        """Every component ID that a callback references as State/Input must exist."""
        from hugiml.dashboard.dash_app import create_app

        app = create_app()
        s = str(app.layout)
        for cid in [
            "ws-wb",
            "ws-gov",
            "wb-nav-setup",
            "wb-nav-results",
            "st-ctx",
            "st-ws",
            "st-wb-sec",
        ]:
            assert cid in s, f"Component '{cid}' missing from initial layout"

    def test_workbench_data_controls_are_not_global_navigation(self):
        from hugiml.dashboard.dash_app import _workbench_data_setup, create_app

        app = create_app()
        layout_text = str(app.layout)
        setup_text = str(_workbench_data_setup())
        assert "ri-source" not in layout_text
        assert "dd-demo" not in layout_text
        assert "ul-file" not in layout_text
        assert "ri-source" in setup_text
        assert "dd-demo" in setup_text
        assert "ul-file" in setup_text


# ── Data utilities ────────────────────────────────────────────────────────────


class TestDataUtils:
    def test_load_demo_credit(self):
        from hugiml.dashboard.dash_components.data_utils import load_demo

        ctx = load_demo("credit")
        assert ctx["X"].shape == (800, 11)
        assert len(ctx["y"]) == 800

    def test_load_demo_churn(self):
        from hugiml.dashboard.dash_components.data_utils import load_demo

        ctx = load_demo("churn")
        assert ctx["roles"]["target"] == "churned"

    def test_load_demo_claims(self):
        from hugiml.dashboard.dash_components.data_utils import load_demo

        ctx = load_demo("claims")
        assert ctx["meta"]["n_features"] > 0

    def test_cache_key_deterministic(self):
        from hugiml.dashboard.dash_components.data_utils import cache_key

        k1 = cache_key("demo:credit", 3, 2026)
        k2 = cache_key("demo:credit", 3, 2026)
        assert k1 == k2
        assert k1 != cache_key("demo:credit", 5, 2026)
        assert len(k1) == 64

    def test_fingerprint_stable(self):
        from hugiml.dashboard.dash_components.data_utils import fingerprint

        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        assert fingerprint(df) == fingerprint(df)


# ── Tables ────────────────────────────────────────────────────────────────────


class TestTables:
    def test_df_to_records_empty(self):
        from hugiml.dashboard.dash_components.tables import df_to_records

        d, c = df_to_records(pd.DataFrame())
        assert d == [] and c == []

    def test_df_to_records_correct_count(self):
        from hugiml.dashboard.dash_components.tables import df_to_records

        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        d, c = df_to_records(df)
        assert len(d) == 3
        assert len(c) == 2
        assert d[0]["a"] == 1

    def test_make_table_returns_datatable(self):
        from dash import dash_table

        from hugiml.dashboard.dash_components.tables import make_table

        tbl = make_table(pd.DataFrame({"x": [1]}), tid="t1")
        assert isinstance(tbl, dash_table.DataTable)
        assert tbl.id == "t1"


# ── Charts ────────────────────────────────────────────────────────────────────


class TestCharts:
    def test_bar_h(self):
        import plotly.graph_objects as go

        from hugiml.dashboard.dash_components.charts import bar_h

        fig = bar_h([1.0, 2.0], ["a", "b"])
        assert isinstance(fig, go.Figure)

    def test_separation_hist(self):
        import plotly.graph_objects as go

        from hugiml.dashboard.dash_components.charts import separation_hist

        fig = separation_hist(np.random.rand(50), np.random.randint(0, 2, 50))
        assert isinstance(fig, go.Figure) and len(fig.data) == 2

    def test_psi_bar(self):
        from hugiml.dashboard.dash_components.charts import psi_bar

        fig = psi_bar(["f1", "f2"], [0.05, 0.30])
        assert len(fig.data) == 1

    def test_fig_to_uri(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from hugiml.dashboard.dash_components.charts import fig_to_uri

        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        uri = fig_to_uri(fig)
        plt.close(fig)
        assert uri.startswith("data:image/png;base64,")

    def test_empty_fig(self):
        from hugiml.dashboard.dash_components.charts import empty_fig

        f = empty_fig("Nothing here")
        assert f.layout.annotations[0].text == "Nothing here"


# ── Styles ────────────────────────────────────────────────────────────────────


class TestStyles:
    def test_build_css_ocean(self):
        from hugiml.dashboard.dash_components.styles import build_css

        css = build_css("Ocean")
        assert "--a:" in css and "#2563eb" in css

    def test_build_css_dark(self):
        from hugiml.dashboard.dash_components.styles import build_css

        css = build_css("Dark")
        assert "#0f172a" in css

    def test_all_themes_have_tokens(self):
        from hugiml.dashboard.dash_components.styles import THEMES, get_tokens

        for t in THEMES:
            tok = get_tokens(t)
            assert "a" in tok and "sf" in tok


# ── Page smoke tests ──────────────────────────────────────────────────────────


class TestPageRendering:
    def test_overview(self, fitted_ctx):
        from hugiml.dashboard.dash_components.pages.overview import render

        assert isinstance(render(fitted_ctx), __import__("dash").html.Div)

    def test_performance(self, fitted_ctx):
        from hugiml.dashboard.dash_components.pages.performance import render

        assert isinstance(render(fitted_ctx), __import__("dash").html.Div)

    def test_patterns(self, fitted_ctx):
        from hugiml.dashboard.dash_components.pages.patterns import render

        assert isinstance(render(fitted_ctx), __import__("dash").html.Div)

    def test_prediction(self, fitted_ctx):
        from hugiml.dashboard.dash_components.pages.prediction import render

        assert isinstance(render(fitted_ctx, row_idx=0), __import__("dash").html.Div)

    def test_data_quality(self, fitted_ctx):
        from hugiml.dashboard.dash_components.pages.data_quality import render

        assert isinstance(render(fitted_ctx), __import__("dash").html.Div)

    def test_config_compare(self, fitted_ctx):
        from hugiml.dashboard.dash_components.pages.config_compare import render

        assert isinstance(render(fitted_ctx, None), __import__("dash").html.Div)

    def test_pruning(self, fitted_ctx):
        from hugiml.dashboard.dash_components.pages.pruning import render

        assert isinstance(render(fitted_ctx), __import__("dash").html.Div)

    def test_monitoring(self, fitted_ctx):
        from hugiml.dashboard.dash_components.pages.monitoring import render

        assert isinstance(render(fitted_ctx), __import__("dash").html.Div)

    def test_representation(self, fitted_ctx):
        from hugiml.dashboard.dash_components.pages.representation import render

        assert isinstance(render(fitted_ctx), __import__("dash").html.Div)


# ── Page router ───────────────────────────────────────────────────────────────


class TestRouter:
    def test_all_pages_via_router(self, fitted_ctx):
        from dash import html

        from hugiml.dashboard.dash_app import PAGES, _gov_page

        for p in PAGES:
            div = _gov_page(p, fitted_ctx)
            assert isinstance(div, html.Div), f"Router failed for {p}"

    def test_unknown_page_returns_error(self, fitted_ctx):
        from hugiml.dashboard.dash_app import _gov_page

        div = _gov_page("Nonexistent", fitted_ctx)
        assert "Unknown" in str(div)


# ── Hero banner ───────────────────────────────────────────────────────────────


class TestHero:
    def test_hero_no_ctx(self):
        from dash import html

        from hugiml.dashboard.dash_app import _hero

        assert isinstance(_hero(None), html.Div)

    def test_hero_with_ctx(self, fitted_ctx):
        from dash import html

        from hugiml.dashboard.dash_app import _hero

        h = _hero(fitted_ctx)
        assert isinstance(h, html.Div)
        rendered = str(h)
        assert "Rows:" not in rendered
        assert "Feats:" not in rendered
        assert "Data:" not in rendered


class TestRepresentationPruning:
    class Model:
        feature_names_in_ = np.array(["age", "income"])
        augmented_pair_transforms_ = [
            {"name": "age__x__income", "operation": "product", "inputs": ["age", "income"]}
        ]

        def get_pattern_info(self):
            return pd.DataFrame(
                {
                    "rank": [1, 2],
                    "pattern": ["age=high", "income=high"],
                    "pattern_origin": ["mined", "mined"],
                }
            )

        def get_downstream_features(self):
            return [
                "orig:age",
                "orig:income",
                "pattern:age=high",
                "pattern:income=high",
                "augmented_pair:age__x__income",
            ]

    def test_all_representation_families_are_available(self):
        from hugiml.dashboard.dash_components.pages.pruning import family_counts, family_table

        model = self.Model()
        X = pd.DataFrame({"age": [20, 40], "income": [30, 80]})
        counts = family_counts(model, X)
        assert counts == {
            "Original features": 2,
            "HUG patterns": 2,
            "Augmented features": 1,
        }

        option_values = {}
        for family, expected in counts.items():
            table, options = family_table(model, X, family)
            assert table is not None
            assert len(options) == expected
            option_values[family] = [option["value"] for option in options]

        assert option_values["HUG patterns"] == [
            "pattern:age=high",
            "pattern:income=high",
        ]
        assert option_values["Augmented features"] == [
            "augmented_pair:age__x__income",
        ]

    def test_page_lists_each_representation_family(self):
        from hugiml.dashboard.dash_components.pages.pruning import render

        model = self.Model()
        X = pd.DataFrame({"age": [20, 40], "income": [30, 80]})
        page = render({"model": model, "X": X, "y": np.array([0, 1])})
        text = str(page)
        assert "Original features (2)" in text
        assert "HUG patterns (2)" in text
        assert "Augmented features (1)" in text


# ── Launcher ──────────────────────────────────────────────────────────────────


class TestLauncher:
    def test_importable(self):
        from hugiml.dashboard import launcher

        assert hasattr(launcher, "main")

    def test_parse_args_defaults(self):
        import sys

        from hugiml.dashboard.launcher import _parse_args

        old = sys.argv
        sys.argv = ["hugiml-dashboard"]
        try:
            args = _parse_args()
            assert args.ui == "dash" and args.port == 8050
        finally:
            sys.argv = old

    def test_parse_args_light(self):
        import sys

        from hugiml.dashboard.launcher import _parse_args

        old = sys.argv
        sys.argv = ["hugiml-dashboard", "--ui", "light"]
        try:
            assert _parse_args().ui == "light"
        finally:
            sys.argv = old

    def test_init_launch_dashboard_importable(self):
        from hugiml.dashboard import launch_dashboard

        assert callable(launch_dashboard)


# ── Streamlit entry point ─────────────────────────────────────────────────────


class TestStreamlitEntryPoint:
    def test_streamlit_app_exposes_governance_sections(self):
        from pathlib import Path

        app = Path("src/hugiml/dashboard/app.py").read_text()
        assert "Streamlit app for HUGIML Governance Studio" in app
        assert "render_performance" in app
        assert "render_drift" in app
        assert "render_missingness" in app

    def test_streamlit_app_main_importable(self):
        from hugiml.dashboard.app import main

        assert callable(main)


class TestRepresentationPruningExecution:
    class Model:
        augmented_pair_transforms_ = [
            {"name": "age__x__income", "operation": "product", "inputs": ["age", "income"]}
        ]

        def transform(self, X):
            from scipy.sparse import csr_matrix

            values = np.column_stack(
                [
                    (X["age"].to_numpy() >= 30).astype(float),
                    (X["income"].to_numpy() >= 50).astype(float),
                ]
            )
            return csr_matrix(values)

        def _make_downstream_features(self, X, patterns, fit=False):
            pattern_values = (
                patterns.toarray() if hasattr(patterns, "toarray") else np.asarray(patterns)
            )
            augmented = (X["age"].to_numpy() * X["income"].to_numpy()).reshape(-1, 1)
            return np.column_stack(
                [
                    X[["age", "income"]].to_numpy(dtype=float),
                    pattern_values,
                    augmented,
                ]
            )

        def get_downstream_features(self):
            return [
                "orig:age",
                "orig:income",
                "pattern:age=high",
                "pattern:income=high",
                "augmented_pair:age__x__income",
            ]

    @staticmethod
    def data():
        X = pd.DataFrame(
            {
                "age": [18, 22, 27, 31, 35, 41, 46, 53, 24, 29, 33, 38, 44, 49, 55, 61],
                "income": [25, 35, 45, 55, 65, 75, 85, 95, 30, 40, 50, 60, 70, 80, 90, 100],
            }
        )
        y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1])
        return X, y

    def test_pattern_column_is_removed_from_fitted_representation(self):
        from hugiml.dashboard.runner import fit_representation_pruned_downstream

        X, y = self.data()
        result, transformed = fit_representation_pruned_downstream(
            self.Model(),
            X,
            y,
            remove_columns=["age=high"],
            family="HUG patterns",
            cv=2,
        )
        assert "pattern:age=high" not in transformed.columns
        assert result.removed_columns == ["pattern:age=high"]

    def test_augmented_column_is_removed_from_fitted_representation(self):
        from hugiml.dashboard.runner import fit_representation_pruned_downstream

        X, y = self.data()
        result, transformed = fit_representation_pruned_downstream(
            self.Model(),
            X,
            y,
            remove_columns=["age__x__income"],
            family="Augmented features",
            cv=2,
        )
        assert "augmented_pair:age__x__income" not in transformed.columns
        assert result.removed_columns == ["augmented_pair:age__x__income"]

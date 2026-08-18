"""Layout/callback contract test used when Dash is not installed in the base test env."""

from __future__ import annotations

import sys
import types
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    find_spec("dash") is not None, reason="real Dash tests cover this environment"
)


class _Dep:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _Comp:
    def __init__(self, *children, **props):
        self.children = props.pop("children", children[0] if len(children) == 1 else list(children))
        self.props = props
        self.id = props.get("id")

    def __repr__(self):
        return f"Comp(id={self.id!r}, children={self.children!r})"

    __str__ = __repr__


class _Factory:
    def __getattr__(self, name):
        return type(name, (_Comp,), {})


class _FakeDash:
    def __init__(self, *args, **kwargs):
        self.layout = None
        self.callback_map = {}
        self.title = kwargs.get("title", "")
        self.index_string = ""

    def callback(self, *deps, **kwargs):
        def deco(fn):
            self.callback_map[fn.__name__] = {"fn": fn, "deps": deps, "kwargs": kwargs}
            return fn

        return deco

    def run(self, *args, **kwargs):
        return None


def _install_stubs(monkeypatch):
    dash = types.ModuleType("dash")
    dash.Dash = _FakeDash
    dash.Input = _Dep
    dash.Output = _Dep
    dash.State = _Dep
    dash.no_update = object()
    dash.dcc = _Factory()
    dash.html = _Factory()
    dash.dash_table = types.SimpleNamespace(DataTable=type("DataTable", (_Comp,), {}))
    dbc = types.ModuleType("dash_bootstrap_components")
    dbc.themes = types.SimpleNamespace(BOOTSTRAP="bootstrap")
    dbc.__getattr__ = lambda name: type(name, (_Comp,), {})
    monkeypatch.setitem(sys.modules, "dash", dash)
    monkeypatch.setitem(sys.modules, "dash_bootstrap_components", dbc)


def test_layout_callback_and_render_contract(monkeypatch):
    _install_stubs(monkeypatch)
    sys.modules.pop("hugiml.causal_dashboard.dash_app", None)
    import hugiml.causal_dashboard.dash_app as da

    app = da.create_app(debug=True)
    assert {"_theme", "_download_metadata_template", "_load", "_run", "_render_results"}.issubset(
        app.callback_map
    )
    text = str(app.layout)
    for cid in [
        "cd-source",
        "cd-demo",
        "cd-meta-template",
        "cd-treatment",
        "cd-outcome",
        "cd-adjustment",
        "cd-baselines",
        "cd-dag",
        "cd-run",
        "cd-diagnostics",
        "cd-effects",
        "cd-regions",
    ]:
        assert cid in text

    load = app.callback_map["_load"]["fn"]
    loaded = load("demo", "credit_intervention", None, None, None, None)
    assert (
        loaded[0] in da._STATE["datasets"]
        and loaded[3] == "enhanced_review"
        and loaded[4] is True
        and loaded[6] == "default_12m"
        and loaded[7] is True
    )

    download = app.callback_map["_download_metadata_template"]["fn"](1)
    assert (
        download["filename"] == "causal_metadata.json" and '"treatment": "T"' in download["content"]
    )

    # Exercise result page construction without running model code in the UI contract test.
    n = 20
    frame = pd.DataFrame({"x": np.arange(n), "T": [0, 1] * 10, "Y": [0, 0, 1, 0] * 5})
    result = types.SimpleNamespace(
        p0=np.full(n, 0.2), p1=np.full(n, 0.15), tau=np.full(n, -0.05), details={"arms": []}
    )
    akey = "contract-analysis"
    da._STATE["analyses"][akey] = {
        "frame": frame,
        "source_df": frame,
        "meta": {"study_type": "randomized", "outcome_direction": "lower_is_better"},
        "treatment": "T",
        "outcome": "Y",
        "covariates": ["x"],
        "results": {"T-HUG": result},
        "test_idx": np.arange(10, 20),
    }
    monkeypatch.setattr(
        da,
        "diagnostic_summary",
        lambda *a, **k: {
            "n": n,
            "n0": 10,
            "n1": 10,
            "raw_diff": 0.0,
            "overlap_rate": 1.0,
            "propensity": np.full(n, 0.5),
            "balance": pd.DataFrame([{"Covariate": "x", "SMD": 0.0, "Flag": "OK"}]),
        },
    )
    monkeypatch.setattr(
        da,
        "t_hug_region_frame",
        lambda a: pd.DataFrame(
            [{"Region / HUG pattern": "x > 5", "Mean CATE": -0.05, "Structure": "shared"}]
        ),
    )
    monkeypatch.setattr(da, "t_hug_rpte_frame", lambda a: pd.DataFrame())
    monkeypatch.setattr(
        da,
        "t_hug_selection_frame",
        lambda a: pd.DataFrame([{"Group": "Control", "Downstream model": "LR"}]),
    )
    monkeypatch.setattr(
        da,
        "selected_configurations_frame",
        lambda a: pd.DataFrame(
            [
                {
                    "Model": "T-HUG",
                    "Group": "Control",
                    "Best score": 0.5,
                    "Candidates evaluated": 1,
                    "Selected parameters": "{}",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        da,
        "comparison_frame",
        lambda a, o=None: pd.DataFrame([{"Model": "T-HUG", "Held-out Brier": 0.2, "ATE": -0.05}]),
    )
    pages = app.callback_map["_render_results"]["fn"](akey, "Ocean")
    assert len(pages) == 5
    assert isinstance(pages[0], str)
    assert all(isinstance(page, _Comp) for page in pages[1:])

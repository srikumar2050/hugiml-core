"""Interface tests for the Dash-first HUGIML LLM Assistant."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MISSING = [
    name for name in ("dash", "dash_bootstrap_components") if importlib.util.find_spec(name) is None
]
pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason="optional LLM interface dependencies are not installed: " + ", ".join(_MISSING),
)


def _component_ids(component) -> set[str]:
    found: set[str] = set()
    identifier = getattr(component, "id", None)
    if identifier:
        found.add(str(identifier))
    children = getattr(component, "children", None)
    if children is None:
        return found
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or getattr(child, "id", None):
            found.update(_component_ids(child))
    return found


def test_dash_app_is_constructible_with_primary_workspaces(tmp_path: Path) -> None:
    from hugiml.llm.dash_app import create_app

    app = create_app(repo_root=tmp_path)
    identifiers = _component_ids(app.layout)

    assert app.title == "HUGIML LLM Assistant"
    assert len(app.callback_map) >= 9
    assert {
        "llm-page-chat",
        "llm-page-dataset",
        "llm-page-evidence",
        "llm-chat-history",
        "llm-dataset",
        "llm-model",
        "llm-refresh-models",
        "llm-model-catalog",
    }.issubset(identifiers)


def test_dash_layout_uses_governance_studio_visual_tokens(tmp_path: Path) -> None:
    from hugiml.llm.dash_app import create_app

    app = create_app(repo_root=tmp_path)
    assert "hug-wrap" in str(app.layout.className)
    assert "--bg" in app.index_string
    assert "LLM Assistant" in str(app.layout)


def test_shared_prompt_service_runs_deterministic_dataset_request(tmp_path: Path) -> None:
    from hugiml.llm.orchestrator import HUGIMLActionOrchestrator
    from hugiml.llm.ui_service import run_prompt

    orchestrator = HUGIMLActionOrchestrator(repo_root=tmp_path, session_dir=tmp_path / "sessions")
    result = run_prompt(
        orchestrator,
        "List the available datasets",
        None,
        "deterministic router only",
        include_benchmarks=False,
        repo_root=tmp_path,
    )

    assert result["ok"] is True
    assert result["action"] == "list_datasets"
    assert result["tables"]["datasets"]


def test_benchmark_catalog_state_supports_repeated_transitions() -> None:
    from hugiml.llm.dash_app import _benchmark_catalog_enabled

    states = [[], ["enabled"], [], ["enabled"]]
    assert [_benchmark_catalog_enabled(state) for state in states] == [False, True, False, True]


def test_dash_model_picker_applies_catalog_memory_and_recommendation(monkeypatch) -> None:
    from hugiml.llm import dash_app
    from hugiml.llm.runtime import MemoryInfo, ModelOption, ModelProfile

    catalog = [
        ModelOption("lightweight-llm", "qwen3:1.7b", "Qwen3 1.7B", 5.0, 4096, 768, "default", ""),
        ModelOption("balanced", "qwen3:4b", "Qwen3 4B", 10.0, 8192, 768, "balanced", ""),
    ]
    monkeypatch.setattr(dash_app, "get_model_catalog", lambda _root=None: catalog)
    memory = MemoryInfo(total_gb=16.0, available_gb=7.0, source="test")
    profile = ModelProfile("lightweight-llm", "qwen3:1.7b", 4096, "validated_llm", 1, 0, "")
    status = {"ok": True, "models": ["qwen3:1.7b", "qwen3:4b"], "error": None}

    options, selected, rows = dash_app._model_picker_state(memory, profile, status)

    assert selected == "qwen3:1.7b"
    assert [option["value"] for option in options] == [
        "deterministic router only",
        "qwen3:1.7b",
    ]
    assert "recommended" in options[1]["label"]
    assert rows[1]["status"] == "needs ≥10.0 GB free RAM"


def test_dash_model_picker_retains_loaded_session_model(monkeypatch) -> None:
    from hugiml.llm import dash_app
    from hugiml.llm.runtime import MemoryInfo, ModelOption, ModelProfile

    catalog = [ModelOption("balanced", "qwen3:4b", "Qwen3 4B", 10.0, 8192, 768, "balanced", "")]
    monkeypatch.setattr(dash_app, "get_model_catalog", lambda _root=None: catalog)
    memory = MemoryInfo(total_gb=16.0, available_gb=4.0, source="test")
    profile = ModelProfile("light-mode", "gemma3:1b", 3072, "validated_llm", 1, 0, "")
    status = {"ok": True, "models": ["qwen3:4b"], "error": None}

    options, selected, rows = dash_app._model_picker_state(
        memory,
        profile,
        status,
        current="qwen3:4b",
    )

    assert selected == "qwen3:4b"
    assert "qwen3:4b" in [option["value"] for option in options]
    assert rows[0]["status"].startswith("kept for this session")


def test_dash_model_picker_falls_back_without_ollama(monkeypatch) -> None:
    from hugiml.llm import dash_app
    from hugiml.llm.runtime import MemoryInfo, ModelOption, ModelProfile

    catalog = [ModelOption("light-mode", "gemma3:1b", "Gemma 3 1B", 3.5, 3072, 512, "light", "")]
    monkeypatch.setattr(dash_app, "get_model_catalog", lambda _root=None: catalog)
    memory = MemoryInfo(total_gb=8.0, available_gb=6.0, source="test")
    profile = ModelProfile("light-mode", "gemma3:1b", 3072, "validated_llm", 1, 0, "")

    options, selected, rows = dash_app._model_picker_state(
        memory,
        profile,
        {"ok": False, "models": [], "error": "offline"},
    )

    assert options == [
        {
            "label": "Deterministic routing only · no Ollama call",
            "value": "deterministic router only",
        }
    ]
    assert selected == "deterministic router only"
    assert rows[0]["status"] == "Ollama is not running"


def test_launch_context_preserves_governance_link_fields() -> None:
    from hugiml.llm.ui_service import parse_launch_context

    context = parse_launch_context(
        "?dataset=credit_risk&session=model-17&view=evidence&source=governance"
    )
    assert context == {
        "dataset": "credit_risk",
        "session": "model-17",
        "view": "evidence",
        "source": "governance",
    }
    assert parse_launch_context("?view=unknown")["view"] == "chat"
    assert parse_launch_context("?view=governance")["view"] == "chat"


def test_cli_ui_routes_to_dash_by_default(monkeypatch) -> None:
    from hugiml.llm import launcher
    from hugiml.llm.cli import _cmd_ui

    captured = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(launcher, "launch", fake_launch)
    assert _cmd_ui(no_open=True) == 0
    assert captured["ui"] == "dash"
    assert captured["port"] is None


def test_cli_can_select_light_interface(monkeypatch) -> None:
    from hugiml.llm import launcher
    from hugiml.llm.cli import _cmd_ui

    captured = {}
    monkeypatch.setattr(launcher, "launch", lambda **kwargs: captured.update(kwargs) or 0)
    assert _cmd_ui(ui="light", port=8510, no_open=True) == 0
    assert captured["ui"] == "light"
    assert captured["port"] == 8510

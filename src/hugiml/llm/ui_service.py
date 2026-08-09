"""Framework-neutral prompt execution for HUGIML interactive interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from .guardrails import deterministic_refusal
from .orchestrator import HUGIMLActionOrchestrator
from .planner import plan_request
from .schemas import ActionRequest

_SKIP_WRITER_ACTIONS = frozenset(
    {
        "list_datasets",
        "describe_dataset",
        "generate_tabular_output",
        "generate_predictions",
        "build_model",
        "tune_hyperparameters",
        "compare_model_configs",
        "prune_patterns",
        "generate_governance_report",
    }
)

_INTERFACE_VIEWS = frozenset({"chat", "dataset", "evidence"})


def parse_launch_context(search: str | None) -> dict[str, str | None]:
    """Parse a stable URL context shared with Governance Studio."""

    query = parse_qs((search or "").lstrip("?"))
    view = str((query.get("view") or ["chat"])[0] or "chat").lower()
    if view not in _INTERFACE_VIEWS:
        view = "chat"
    return {
        "dataset": _query_value(query, "dataset"),
        "session": _query_value(query, "session"),
        "view": view,
        "source": _query_value(query, "source"),
    }


def _query_value(query: dict[str, list[str]], name: str) -> str | None:
    value = str((query.get(name) or [""])[0] or "").strip()
    return value or None


def run_prompt(
    orchestrator: HUGIMLActionOrchestrator,
    prompt: str,
    selected_dataset: str | None,
    model_choice: str,
    *,
    response_mode: str = "Fast",
    include_benchmarks: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Plan and execute one guarded interactive request."""

    refusal = deterministic_refusal(prompt)
    if refusal is not None:
        return refusal.to_dict()
    thinking_mode = str(response_mode).lower().startswith("think")
    prefer_llm = thinking_mode and model_choice != "deterministic router only"
    model = model_choice if prefer_llm else None
    active_session = None
    if orchestrator.last_session_id:
        active_session = orchestrator.sessions.get(orchestrator.last_session_id)
    context = {
        "dataset": selected_dataset,
        "active_session_id": orchestrator.last_session_id,
        "active_session_dataset": getattr(active_session, "dataset", None),
        "has_active_model": active_session is not None,
        "note": "If the user asks to summarize findings after a model build, use explain_model.",
    }
    planned = plan_request(
        prompt,
        model=model,
        prefer_llm=prefer_llm,
        context=context,
        repo_root=repo_root,
    )
    if hasattr(planned, "to_dict") and getattr(planned, "ok", None) is False:
        return planned.to_dict()
    if not isinstance(planned, ActionRequest):
        return {"ok": False, "message": "Unable to plan request.", "action": "refuse"}
    if planned.action == "list_datasets":
        visible = orchestrator.registry.list_datasets(
            include_profiles=True,
            include_benchmarks=include_benchmarks,
        )
        rows = [info.to_dict() for info in visible]
        scope = "built-in and user"
        if include_benchmarks:
            scope = "built-in, user, and benchmark"
        return {
            "ok": True,
            "action": "list_datasets",
            "message": f"Found {len(rows)} visible datasets from {scope} sources.",
            "tables": {"datasets": rows},
            "data": {"count": len(rows), "include_benchmarks": include_benchmarks},
            "artifacts": {},
            "refusal_reason": None,
        }
    dataset_actions = {
        "describe_dataset",
        "build_model",
        "tune_hyperparameters",
        "generate_predictions",
        "generate_tabular_output",
        "compare_model_configs",
        "explain_model",
        "explain_prediction",
    }
    if selected_dataset and planned.dataset is None and planned.action in dataset_actions:
        planned.dataset = selected_dataset
    if planned.action in {"build_model", "tune_hyperparameters", "compare_model_configs"}:
        planned.params = dict(planned.params or {})
        if not asks_for_full_dataset(prompt):
            limit = 8000 if planned.action == "build_model" else 6000
            planned.params.setdefault("_chat_max_rows", limit)
    if prefer_llm and model and planned.action not in _SKIP_WRITER_ACTIONS:
        planned.params = dict(planned.params or {})
        planned.params["_writer_model"] = model
    return orchestrator.execute(planned).to_dict()


def asks_for_full_dataset(prompt: str) -> bool:
    """Return whether a prompt explicitly requests unrestricted row usage."""

    low = (prompt or "").lower()
    return any(
        term in low
        for term in (
            "full dataset",
            "entire dataset",
            "all rows",
            "without sampling",
            "no sampling",
        )
    )

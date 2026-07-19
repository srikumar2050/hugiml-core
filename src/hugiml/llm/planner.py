"""Planning helpers for the optional HUGIML natural-language interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .guardrails import deterministic_refusal, infer_model_request_hints, route_without_llm
from .ollama_client import OllamaPlanner
from .runtime import (
    FALLBACK_OLLAMA_MODEL,
    check_ollama,
    is_below_minimum_llm_model,
    recommend_profile,
)
from .schemas import ActionRequest, ActionResult

_MODEL_ACTIONS = {"build_model", "tune_hyperparameters", "compare_model_configs"}


def plan_request(
    user_text: str,
    *,
    model: str | None = None,
    context: dict[str, Any] | None = None,
    prefer_llm: bool = True,
    repo_root: str | Path | None = None,
) -> ActionRequest | ActionResult:
    """Plan a user request into a validated, schema-safe action request."""

    refusal = deterministic_refusal(user_text)
    if refusal is not None:
        return refusal

    # Common execution commands are deterministic so a local model cannot
    # accidentally turn "run performance_ho" or "use RPTE downstream" into
    # generic API help.
    routed = route_without_llm(user_text, repo_root=repo_root)
    if routed is not None:
        return routed

    if prefer_llm:
        profile = recommend_profile(repo_root=repo_root)
        selected_model = model or profile.recommended_model
        if selected_model and is_below_minimum_llm_model(selected_model):
            selected_model = ""
        status = check_ollama()
        if status.get("ok"):
            installed = set(status.get("models", []))
            candidate_models: list[str] = []
            if selected_model:
                candidate_models.append(selected_model)
            if FALLBACK_OLLAMA_MODEL not in candidate_models:
                candidate_models.append(FALLBACK_OLLAMA_MODEL)
            for candidate in candidate_models:
                if not candidate or candidate not in installed or is_below_minimum_llm_model(candidate):
                    continue
                try:
                    planned = OllamaPlanner(candidate).plan(user_text, context=context or {})
                    return _normalise_planned_model_request(planned, user_text)
                except Exception:
                    continue

    return ActionRequest(action="answer_api_question", question=user_text)


def _normalise_planned_model_request(planned: ActionRequest, user_text: str) -> ActionRequest:
    """Recover model options that small local planners commonly omit or rename."""

    hints = infer_model_request_hints(user_text)
    params = dict(planned.params or {})

    # Accept common JSON forms produced by local models even when they do not
    # exactly follow the prompt's preferred names.
    model_params = dict(params.get("model_params") or {})
    grid_value = (
        params.pop("grid", None)
        or params.get("grid_name")
        or model_params.pop("grid", None)
        or model_params.pop("grid_name", None)
    )
    downstream_value = (
        params.pop("base_estimator", None)
        or params.pop("downstream", None)
        or params.get("downstream_estimator")
        or model_params.pop("base_estimator", None)
        or model_params.pop("downstream_estimator", None)
    )
    if model_params:
        params["model_params"] = model_params
    elif "model_params" in params:
        params.pop("model_params", None)
    strategy_text = str(planned.strategy or "").strip().lower().replace("-", "_").replace(" ", "_")
    if strategy_text in {
        "performance_ho",
        "performanceho",
        "interpretability_ho",
        "interpretabilityho",
        "interpretability",
        "performance",
    }:
        if strategy_text in {"performance_ho", "performanceho"}:
            grid_value = grid_value or "performance_ho"
        elif strategy_text in {"interpretability_ho", "interpretabilityho"}:
            grid_value = grid_value or "interpretability_ho"
        elif "grid" in user_text.lower() or planned.action == "tune_hyperparameters":
            grid_value = grid_value or strategy_text

    grid_value = hints.get("grid_name") or grid_value
    downstream_value = hints.get("downstream_estimator") or downstream_value
    if grid_value:
        params["grid_name"] = str(grid_value)
    if downstream_value:
        params["downstream_estimator"] = str(downstream_value)

    hinted_params = dict(hints.get("model_params") or {})
    if hinted_params:
        merged = dict(params.get("model_params") or {})
        merged.update(hinted_params)
        params["model_params"] = merged

    if planned.action == "answer_api_question" and hints.get("is_execution_command"):
        routed = route_without_llm(user_text)
        if routed is not None:
            return routed

    if planned.action in _MODEL_ACTIONS:
        if params.get("grid_name"):
            planned.action = "tune_hyperparameters"
        hinted_strategy = str(hints.get("strategy") or "balanced")
        if hinted_strategy != "balanced" or not planned.strategy:
            planned.strategy = hinted_strategy
        planned.params = params
    return planned

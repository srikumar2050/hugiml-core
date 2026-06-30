"""Planning helpers for the optional HUGIML NLP interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .guardrails import deterministic_refusal, route_without_llm
from .ollama_client import OllamaPlanner
from .runtime import (
    FALLBACK_OLLAMA_MODEL,
    check_ollama,
    is_below_minimum_llm_model,
    recommend_profile,
)
from .schemas import ActionRequest, ActionResult


def plan_request(
    user_text: str,
    *,
    model: str | None = None,
    context: dict[str, Any] | None = None,
    prefer_llm: bool = True,
    repo_root: str | Path | None = None,
) -> ActionRequest | ActionResult:
    """Plan a user request into a safe action request.

    Deterministic refusals happen before any model call.  Common actions can be
    routed without Ollama.  If ``prefer_llm`` is true and Ollama is available,
    the local model is asked for strict JSON, then the orchestrator still
    validates it before execution.  ``repo_root`` is only used by the no-LLM
    fallback router, to look up known dataset names mentioned in the text.
    """

    refusal = deterministic_refusal(user_text)
    if refusal is not None:
        return refusal

    # Route high-confidence intents deterministically first.  The selected local
    # model is still useful as a writer/synthesis layer, but it should not be
    # allowed to randomly reclassify obvious documentation questions such as
    # "what is HUGIML" or obvious model-result requests such as
    # "summarize findings".
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
            # Try the explicit tiny fallback before abandoning Ollama.  This keeps
            # the chat on a local model when Qwen/Gemma is temporarily unavailable
            # or returns malformed planning JSON.
            if FALLBACK_OLLAMA_MODEL not in candidate_models:
                candidate_models.append(FALLBACK_OLLAMA_MODEL)
            for candidate in candidate_models:
                if not candidate or candidate not in installed or is_below_minimum_llm_model(candidate):
                    continue
                try:
                    planned = OllamaPlanner(candidate).plan(user_text, context=context or {})
                    # If the model falls back to a generic API answer but the
                    # deterministic router found a concrete workflow action, prefer
                    # the concrete action.
                    if planned.action == "answer_api_question" and routed is not None:
                        return routed
                    return planned
                except Exception:
                    # Try the next configured candidate, then fall through to the
                    # deterministic router if all local models fail.
                    continue

    if routed is not None:
        return routed
    return ActionRequest(action="answer_api_question", question=user_text)

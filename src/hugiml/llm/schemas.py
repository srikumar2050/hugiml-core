"""Strict action schemas for the optional HUGIML natural-language interface.

The LLM add-on intentionally exposes a narrow, deterministic action surface.
Models may propose these actions as JSON, but Python validation decides what
can run.  No arbitrary code, shell execution, source editing, or package
modification action exists in this schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ActionName = Literal[
    "list_datasets",
    "describe_dataset",
    "build_model",
    "tune_hyperparameters",
    "generate_predictions",
    "generate_tabular_output",
    "compare_model_configs",
    "explain_model",
    "explain_prediction",
    "prune_patterns",
    "generate_governance_report",
    "answer_api_question",
    "refuse",
]

ALLOWED_ACTIONS: tuple[str, ...] = (
    "list_datasets",
    "describe_dataset",
    "build_model",
    "tune_hyperparameters",
    "generate_predictions",
    "generate_tabular_output",
    "compare_model_configs",
    "explain_model",
    "explain_prediction",
    "prune_patterns",
    "generate_governance_report",
    "answer_api_question",
    "refuse",
)

REFUSAL_REASONS: dict[str, str] = {
    "code_modification_not_supported": (
        "This NLP interface cannot modify package source code, write scripts, run shell "
        "commands, or rewrite repository files. It can help you build, tune, evaluate, "
        "generate tabular outputs, interpret HUGIML models, and answer API/feature questions."
    ),
    "unsupported_action": (
        "That request is outside the supported HUGIML NLP action set. Supported actions are "
        "dataset discovery, HUGIML-only model build/tune/evaluate, tabular outputs, "
        "interpretation, controlled pruning/governance reports, and API help."
    ),
    "baseline_model_not_supported": (
        "The initial NLP interface runs HUGIML modeling and tuning only. Baseline models such "
        "as XGBoost, LightGBM, random forests, EBM, RuleFit, or logistic-regression comparisons "
        "are intentionally not exposed through this action layer."
    ),
    "invalid_action_json": (
        "The model response could not be validated as a supported HUGIML action."
    ),
    "dataset_required": (
        "I need to know which dataset to use first. Pass --dataset <name>, say \"list datasets\" "
        "to see what's available, or mention the dataset by name in your question."
    ),
}


@dataclass(frozen=True)
class DatasetInfo:
    """A compact dataset registry entry."""

    name: str
    source: str
    path: str | None = None
    task_type: str = "binary_classification"
    target: str | None = None
    rows: int | None = None
    features: int | None = None
    description: str = ""
    origin_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "path": self.path,
            "task_type": self.task_type,
            "target": self.target,
            "rows": self.rows,
            "features": self.features,
            "description": self.description,
            "origin_detail": self.origin_detail,
        }


@dataclass
class ActionRequest:
    """Validated action request proposed by an LLM or by deterministic routing."""

    action: str
    dataset: str | None = None
    target: str | None = None
    metric: str | None = None
    strategy: str = "balanced"
    params: dict[str, Any] = field(default_factory=dict)
    output_format: str = "table"
    session_id: str | None = None
    question: str | None = None
    pattern_indices: list[int] = field(default_factory=list)
    keyword: str | None = None
    min_support: float | None = None
    reason: str | None = None
    limit: int = 10

    def validate(self) -> None:
        if self.action not in ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported action {self.action!r}.")
        if self.limit <= 0:
            raise ValueError("limit must be positive.")
        if self.action in {
            "describe_dataset",
            "build_model",
            "tune_hyperparameters",
            "generate_predictions",
            "compare_model_configs",
        } and not self.dataset:
            raise ValueError(f"action {self.action!r} requires a dataset.")
        if self.action == "prune_patterns" and not (
            self.pattern_indices or self.keyword or self.min_support is not None
        ):
            raise ValueError("prune_patterns requires pattern_indices, keyword, or min_support.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionRequest:
        """Build an ActionRequest from a dict without validating yet.

        Validation is intentionally deferred to the caller (normally
        ``HUGIMLActionOrchestrator.execute``) so that missing-but-recoverable
        fields, such as a dataset that can be inferred from an active session,
        can be filled in before validation runs.
        """

        allowed_fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        clean = {k: v for k, v in dict(data).items() if k in allowed_fields}
        return cls(**clean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "dataset": self.dataset,
            "target": self.target,
            "metric": self.metric,
            "strategy": self.strategy,
            "params": dict(self.params),
            "output_format": self.output_format,
            "session_id": self.session_id,
            "question": self.question,
            "pattern_indices": list(self.pattern_indices),
            "keyword": self.keyword,
            "min_support": self.min_support,
            "reason": self.reason,
            "limit": self.limit,
        }


@dataclass
class ActionResult:
    """Structured result returned by the deterministic orchestrator."""

    ok: bool
    action: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    refusal_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "message": self.message,
            "data": self.data,
            "tables": self.tables,
            "artifacts": self.artifacts,
            "refusal_reason": self.refusal_reason,
        }


def refusal_result(reason: str = "unsupported_action", action: str = "refuse") -> ActionResult:
    return ActionResult(
        ok=False,
        action=action,
        message=REFUSAL_REASONS.get(reason, REFUSAL_REASONS["unsupported_action"]),
        refusal_reason=reason,
    )

"""Optional natural-language interface for HUGIML.

This namespace is additive and lazy: importing ``hugiml`` does not import or
require Ollama.  Use ``hugiml-llm`` after installing the optional extra, ``python -m hugiml.llm.cli``
from a source checkout, or import ``HUGIMLActionOrchestrator`` directly for
programmatic use.
"""

from __future__ import annotations

from .dataset_registry import DatasetRegistry
from .orchestrator import HUGIMLActionOrchestrator, generate_qna_html
from .runtime import check_ollama, get_memory_info, recommend_profile
from .schemas import ActionRequest, ActionResult

__all__ = [
    "ActionRequest",
    "ActionResult",
    "DatasetRegistry",
    "HUGIMLActionOrchestrator",
    "check_ollama",
    "generate_qna_html",
    "get_memory_info",
    "recommend_profile",
]

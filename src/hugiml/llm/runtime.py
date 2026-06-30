"""Runtime profile and local-model detection for the optional NLP interface."""

from __future__ import annotations

import json
import os
import platform
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MemoryInfo:
    total_gb: float | None
    available_gb: float | None
    source: str


@dataclass(frozen=True)
class ModelProfile:
    name: str
    recommended_model: str
    max_context_tokens: int
    planning_mode: str
    branches: int
    lookahead: int
    description: str


@dataclass(frozen=True)
class ModelOption:
    """One visible local-model choice in the workbench model picker.

    ``min_available_gb`` is intentionally conservative.  It is a UI/runtime
    guardrail, not a claim about the model file size: browser, OS, Ollama KV
    cache, dataset profiling, and HUGIML fitting all share the same free RAM.
    """

    profile: str
    model: str
    label: str
    min_available_gb: float
    max_context_tokens: int
    max_output_tokens: int
    usage: str
    notes: str


DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
LIGHT_MODE_MODEL = "gemma3:1b"
FALLBACK_OLLAMA_MODEL = "llama3.2:1b"

LIGHTWEIGHT_SUPPORTED_MODELS = {DEFAULT_OLLAMA_MODEL, LIGHT_MODE_MODEL, FALLBACK_OLLAMA_MODEL}

DEFAULT_PROFILES: dict[str, ModelProfile] = {
    "no-llm": ModelProfile(
        name="no-llm",
        recommended_model="",
        max_context_tokens=0,
        planning_mode="deterministic",
        branches=0,
        lookahead=0,
        description="Less than about 3.5 GB free RAM; use deterministic routing without calling Ollama.",
    ),
    "light-mode": ModelProfile(
        name="light-mode",
        recommended_model=LIGHT_MODE_MODEL,
        max_context_tokens=3072,
        planning_mode="validated_llm",
        branches=1,
        lookahead=0,
        description="Lowest-memory Ollama mode using Gemma 3 1B for compact answer writing and routing.",
    ),
    "lightweight-llm": ModelProfile(
        name="lightweight-llm",
        recommended_model=DEFAULT_OLLAMA_MODEL,
        max_context_tokens=4096,
        planning_mode="validated_llm",
        branches=1,
        lookahead=0,
        description="Default lightweight Ollama mode using Qwen3 1.7B; deterministic HUGIML execution remains available.",
    ),
    "fallback-llm": ModelProfile(
        name="fallback-llm",
        recommended_model=FALLBACK_OLLAMA_MODEL,
        max_context_tokens=3072,
        planning_mode="validated_llm",
        branches=1,
        lookahead=0,
        description="Fallback local model used when the preferred lightweight writer/planner is unavailable or fails.",
    ),
    "minimum-llm": ModelProfile(
        name="minimum-llm",
        recommended_model="llama3.2:3b",
        max_context_tokens=4096,
        planning_mode="validated_llm",
        branches=1,
        lookahead=0,
        description="Minimum recommended local model for a real conversational LLM experience.",
    ),
    "balanced": ModelProfile(
        name="balanced",
        recommended_model="qwen3:4b",
        max_context_tokens=8192,
        planning_mode="validated_llm",
        branches=1,
        lookahead=0,
        description="Good everyday local model for summaries, follow-up questions, and interpretation.",
    ),
    "expanded": ModelProfile(
        name="expanded",
        recommended_model="qwen3:8b",
        max_context_tokens=12288,
        planning_mode="validated_llm",
        branches=1,
        lookahead=0,
        description="Stronger local model with richer grounded interpretation when more RAM is free.",
    ),
    "large-context": ModelProfile(
        name="large-context",
        recommended_model="gemma3:12b",
        max_context_tokens=16384,
        planning_mode="expanded",
        branches=1,
        lookahead=0,
        description="Larger-context local model for long model cards, bigger outputs, and richer analysis.",
    ),
}


DEFAULT_MODEL_CATALOG: list[ModelOption] = [
    ModelOption(
        profile="lightweight-llm",
        model=DEFAULT_OLLAMA_MODEL,
        label="Qwen3 1.7B",
        min_available_gb=5.0,
        max_context_tokens=4096,
        max_output_tokens=768,
        usage="default local LLM",
        notes="Default HUGIML local writer/planner for polished grounded answers. Deterministic HUGIML execution remains the source of truth.",
    ),
    ModelOption(
        profile="light-mode",
        model=LIGHT_MODE_MODEL,
        label="Gemma 3 1B",
        min_available_gb=3.5,
        max_context_tokens=3072,
        max_output_tokens=512,
        usage="light mode",
        notes="Lowest-memory Ollama mode for compact grounded answers when Qwen3 1.7B is too heavy or unavailable.",
    ),
    ModelOption(
        profile="fallback-llm",
        model=FALLBACK_OLLAMA_MODEL,
        label="Llama 3.2 1B",
        min_available_gb=3.5,
        max_context_tokens=3072,
        max_output_tokens=512,
        usage="fallback",
        notes="Fallback writer/planner tried before deterministic routing when the selected model fails or is unavailable.",
    ),
    ModelOption(
        profile="minimum-llm",
        model="llama3.2:3b",
        label="Llama 3.2 3B",
        min_available_gb=6.0,
        max_context_tokens=4096,
        max_output_tokens=768,
        usage="minimum real LLM",
        notes="Minimum recommended model for a conversational HUGIML assistant. Below this, use deterministic routing only.",
    ),
    ModelOption(
        profile="balanced",
        model="qwen3:4b",
        label="Qwen3 4B",
        min_available_gb=10.0,
        max_context_tokens=8192,
        max_output_tokens=1024,
        usage="balanced",
        notes="Recommended for everyday local NLP workbench use when about 10-16 GB RAM is free.",
    ),
    ModelOption(
        profile="balanced",
        model="gemma3:4b",
        label="Gemma 3 4B",
        min_available_gb=10.0,
        max_context_tokens=8192,
        max_output_tokens=1024,
        usage="balanced alternative",
        notes="Alternative 4B-class model for summaries, interpretation, and follow-up questions.",
    ),
    ModelOption(
        profile="expanded",
        model="qwen3:8b",
        label="Qwen3 8B",
        min_available_gb=16.0,
        max_context_tokens=12288,
        max_output_tokens=1536,
        usage="stronger local LLM",
        notes="Recommended when about 16-32 GB RAM is free and richer explanations are desired.",
    ),
    ModelOption(
        profile="large-context",
        model="gemma3:12b",
        label="Gemma 3 12B",
        min_available_gb=32.0,
        max_context_tokens=16384,
        max_output_tokens=2048,
        usage="large context",
        notes="Largest default recommendation; use only when ample RAM is free.",
    ),
]


def _config_search_paths(repo_root: str | Path | None = None) -> list[Path]:
    """Source-checkout copy first (user-editable), then the packaged copy."""

    paths: list[Path] = []
    if repo_root:
        paths.append(Path(repo_root) / "LLM" / "config" / "models.yaml")
    paths.append(Path(__file__).resolve().parent / "assets" / "config" / "models.yaml")
    return paths


def _load_yaml_config(repo_root: str | Path | None = None) -> dict[str, Any] | None:
    try:
        import yaml  # type: ignore
    except Exception:
        return None

    for path in _config_search_paths(repo_root):
        if not path.exists():
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict):
                return raw
        except Exception:
            continue
    return None


def _load_profiles_from_yaml(repo_root: str | Path | None = None) -> dict[str, ModelProfile] | None:
    """Load tier profiles from ``models.yaml`` if PyYAML and file are available."""

    raw = _load_yaml_config(repo_root)
    if not raw:
        return None
    entries = raw.get("profiles") or {}
    if not entries:
        return None
    profiles: dict[str, ModelProfile] = {}
    for name, entry in entries.items():
        entry = entry or {}
        fallback = DEFAULT_PROFILES.get(name)
        profiles[name] = ModelProfile(
            name=name,
            recommended_model=str(
                entry.get("recommended_model")
                or (fallback.recommended_model if fallback else "qwen3:4b")
            ),
            max_context_tokens=int(
                entry.get("max_context_tokens")
                or (fallback.max_context_tokens if fallback else 8192)
            ),
            planning_mode=str(
                entry.get("planning_mode") or (fallback.planning_mode if fallback else "guided")
            ),
            branches=int(entry.get("branches") if entry.get("branches") is not None else (fallback.branches if fallback else 3)),
            lookahead=int(entry.get("lookahead") if entry.get("lookahead") is not None else (fallback.lookahead if fallback else 1)),
            description=str(entry.get("notes") or (fallback.description if fallback else "")),
        )
    return profiles or None


def _load_model_catalog_from_yaml(repo_root: str | Path | None = None) -> list[ModelOption] | None:
    raw = _load_yaml_config(repo_root)
    if not raw:
        return None
    entries = raw.get("models") or []
    if not isinstance(entries, list) or not entries:
        return None
    options: list[ModelOption] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("model"):
            continue
        profile = str(entry.get("profile") or "manual")
        model = str(entry["model"])
        fallback = next((m for m in DEFAULT_MODEL_CATALOG if m.model == model), None)
        options.append(
            ModelOption(
                profile=profile,
                model=model,
                label=str(entry.get("label") or (fallback.label if fallback else model)),
                min_available_gb=float(entry.get("min_available_gb") if entry.get("min_available_gb") is not None else (fallback.min_available_gb if fallback else 8.0)),
                max_context_tokens=int(entry.get("max_context_tokens") or (fallback.max_context_tokens if fallback else 4096)),
                max_output_tokens=int(entry.get("max_output_tokens") or (fallback.max_output_tokens if fallback else 768)),
                usage=str(entry.get("usage") or (fallback.usage if fallback else "manual")),
                notes=str(entry.get("notes") or (fallback.notes if fallback else "")),
            )
        )
    return options or None


def get_profiles(repo_root: str | Path | None = None) -> dict[str, ModelProfile]:
    """Tier profiles, preferring ``models.yaml`` over the hardcoded defaults."""

    profiles = DEFAULT_PROFILES.copy()
    loaded = _load_profiles_from_yaml(repo_root)
    if loaded:
        profiles.update(loaded)
    return profiles


def get_model_catalog(repo_root: str | Path | None = None) -> list[ModelOption]:
    """Visible local-model catalog used by CLI/UI.

    The list is fixed by config and independent of which models are currently
    pulled in Ollama.  UI callers should display every entry, then disable
    entries that are not feasible for the current machine or not installed.
    """

    return _load_model_catalog_from_yaml(repo_root) or DEFAULT_MODEL_CATALOG


def estimated_model_size_b(model_name: str) -> float | None:
    """Best-effort parameter-size estimate parsed from common Ollama tags.

    Examples: ``llama3.2:3b`` -> 3.0, ``gemma3:270m`` -> 0.27.
    The value is used for RAM/UI guidance; explicitly supported lightweight
    models such as ``qwen3:1.7b``, ``gemma3:1b``, and ``llama3.2:1b`` are allowed even though they are below 3B.
    """

    import re

    text = model_name.lower().replace(" ", "")
    matches = re.findall(r"(\d+(?:\.\d+)?)(b|m)(?![a-z])", text)
    if not matches:
        return None
    value, unit = matches[-1]
    size = float(value)
    return size if unit == "b" else size / 1000.0


def _normalise_model_name(model_name: str) -> str:
    return model_name.strip().lower()


def is_lightweight_supported_model(model_name: str) -> bool:
    """Return True for explicitly supported sub-3B local LLM choices."""

    return _normalise_model_name(model_name) in LIGHTWEIGHT_SUPPORTED_MODELS


def is_below_minimum_llm_model(model_name: str) -> bool:
    """Return True for unsupported tiny models.

    The configured tiny models are intentionally allowed for default/light/fallback modes, but
    arbitrary sub-3B manual models remain disabled because they often produce
    brittle planning JSON and generic explanations.
    """

    if is_lightweight_supported_model(model_name):
        return False
    size_b = estimated_model_size_b(model_name)
    return size_b is not None and size_b < 3.0


def get_memory_info() -> MemoryInfo:
    """Best-effort total/available RAM detection without mandatory psutil."""

    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return MemoryInfo(
            total_gb=round(float(vm.total) / (1024**3), 2),
            available_gb=round(float(vm.available) / (1024**3), 2),
            source="psutil",
        )
    except Exception:
        pass

    if platform.system() == "Linux":
        try:
            data: dict[str, int] = {}
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    key, value = line.split(":", 1)
                    amount_kb = int(value.strip().split()[0])
                    data[key] = amount_kb
            total = data.get("MemTotal")
            available = data.get("MemAvailable", data.get("MemFree"))
            return MemoryInfo(
                total_gb=None if total is None else round(total / (1024**2), 2),
                available_gb=None if available is None else round(available / (1024**2), 2),
                source="/proc/meminfo",
            )
        except Exception:
            pass
    return MemoryInfo(total_gb=None, available_gb=None, source="unknown")


def recommend_profile(
    memory: MemoryInfo | None = None, *, repo_root: str | Path | None = None
) -> ModelProfile:
    """Recommend a profile from available RAM, not merely total RAM."""

    memory = memory or get_memory_info()
    profiles = get_profiles(repo_root)
    avail = memory.available_gb
    if avail is None:
        return profiles.get("minimum-llm", DEFAULT_PROFILES["minimum-llm"])
    # Conservative thresholds because browser/dashboard, Ollama KV cache, data
    # profiling, and HUGIML fitting share the same free memory pool.
    # Gemma 3 1B is the low-memory light mode from about 3.5 GB free RAM.
    # Qwen3 1.7B remains the normal default once there is enough headroom,
    # while deterministic routing remains available at all tiers.
    if avail < 3.5:
        return profiles.get("no-llm", DEFAULT_PROFILES["no-llm"])
    if avail < 5:
        return profiles.get("light-mode", DEFAULT_PROFILES["light-mode"])
    if avail < 6:
        return profiles.get("lightweight-llm", DEFAULT_PROFILES["lightweight-llm"])
    if avail < 10:
        return profiles.get("minimum-llm", DEFAULT_PROFILES["minimum-llm"])
    if avail < 16:
        return profiles.get("balanced", DEFAULT_PROFILES["balanced"])
    if avail < 32:
        return profiles.get("expanded", DEFAULT_PROFILES["expanded"])
    return profiles.get("large-context", DEFAULT_PROFILES["large-context"])


def model_availability(
    option: ModelOption,
    memory: MemoryInfo | None = None,
    pulled_models: list[str] | set[str] | tuple[str, ...] | None = None,
    *,
    ollama_ok: bool = True,
) -> dict[str, Any]:
    """Return UI-friendly selectability information for one model option."""

    memory = memory or get_memory_info()
    avail = memory.available_gb
    ram_ok = avail is None or avail >= option.min_available_gb
    pulled = set(pulled_models or [])
    installed = option.model in pulled
    if not ram_ok:
        reason = f"needs ≥{option.min_available_gb:.1f} GB free RAM"
    elif not ollama_ok:
        reason = "Ollama is not running"
    elif not installed:
        reason = f"pull with: ollama pull {option.model}"
    else:
        reason = "ready"
    return {
        "model": option.model,
        "label": option.label,
        "profile": option.profile,
        "min_available_gb": option.min_available_gb,
        "available_gb": avail,
        "ram_ok": ram_ok,
        "installed": installed,
        "ollama_ok": ollama_ok,
        "selectable": bool(ram_ok and installed and ollama_ok),
        "reason": reason,
    }


def ollama_base_url() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


# Speed: a single chat turn typically triggers two or more independent
# callers that each want fresh Ollama status (the sidebar render, the
# planner, the answer-writer fallback selection).  Each call is a blocking
# HTTP round trip to localhost.  A short TTL cache collapses these into one
# real request per Streamlit rerun / CLI turn without meaningfully risking
# staleness -- Ollama coming up or down mid-turn is rare, and the next
# user turn (almost always >2s later) gets a fresh check anyway.
_OLLAMA_STATUS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_OLLAMA_STATUS_TTL_SECONDS = 3.0


def check_ollama(
    base_url: str | None = None, timeout: float = 2.0, *, use_cache: bool = True
) -> dict[str, Any]:
    """Return Ollama server/model status.  Never starts or downloads models.

    Results are cached per base URL for ``_OLLAMA_STATUS_TTL_SECONDS`` to
    avoid redundant round trips when multiple callers check status within
    the same user-facing turn. Pass ``use_cache=False`` to force a fresh
    check (e.g. immediately after the user pulls/starts Ollama).
    """

    base = (base_url or ollama_base_url()).rstrip("/")
    now = time.monotonic()
    if use_cache:
        cached = _OLLAMA_STATUS_CACHE.get(base)
        if cached is not None and (now - cached[0]) < _OLLAMA_STATUS_TTL_SECONDS:
            return cached[1]
    url = f"{base}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - local user configured URL
            payload = json.loads(resp.read().decode("utf-8"))
        models = [m.get("name", "") for m in payload.get("models", [])]
        result = {"ok": True, "base_url": base, "models": sorted(models), "error": None}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        result = {"ok": False, "base_url": base, "models": [], "error": str(exc)}
    _OLLAMA_STATUS_CACHE[base] = (now, result)
    return result

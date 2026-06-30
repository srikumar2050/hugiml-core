"""Guardrails and deterministic intent pre-routing for the optional LLM add-on."""

from __future__ import annotations

import re
from pathlib import Path

from .schemas import ActionRequest, ActionResult, refusal_result

# NOTE on "implement": matched as a simple bare word below, same as the other
# triggers. A *narrower* qualifier pattern (e.g. requiring "implement a new
# <noun>") was tried and reverted: real requests have arbitrary words in
# between ("implement a new pruning criterion in mining.cpp"), which a tight
# qualifier pattern misses. False positives on legitimate questions ("how
# does hugiml implement pruning?") are instead avoided by the broadened
# _API_HELP_PATTERNS exemption below, which now also covers "how does",
# "what features", etc.
_CODE_EDIT_PATTERNS = re.compile(
    r"\b("
    r"modify|edit|change source|rewrite|refactor|implement|add code|"
    r"create a script|write a script|run shell|shell command|pip install|"
    r"delete file|overwrite file|commit|git|pull request|build wheel|rebuild package"
    r")\b",
    re.IGNORECASE,
)

# Exemption: questions ABOUT the API/feature/docs should not be refused even
# if they happen to share vocabulary with _CODE_EDIT_PATTERNS (e.g. a question
# that mentions "implement" while asking how something already works).
_API_HELP_PATTERNS = re.compile(
    r"\b("
    r"api|parameter|parameters|method|methods|feature|features|"
    r"how do i|how does|how is|how are|what does|what is|what are|"
    r"what feature|what features|what parameter|what parameters|what method|what methods|"
    r"tell me about|more about|what's in|what is in|"
    r"explain|docs?|documentation|usage"
    r")\b",
    re.IGNORECASE,
)

_DATASET_PATTERNS = re.compile(r"\b(list|show|available|describe|dataset|data set|load)\b", re.I)
# `_DATASET_PATTERNS` above is intentionally broad (it also matches generic
# verbs like "show"/"available"/"load" that have nothing inherently to do
# with datasets). Using it alone to decide "this is a dataset request" meant
# ANY message containing the word "show" -- e.g. "show me top 3 most
# interesting patterns found", or "show me pattern distribution by original
# and patterns_only" -- got routed to list_datasets before it ever reached
# the pattern/explain-model branches below (or the LLM, since the
# deterministic router runs first regardless of Thinking mode). Listing or
# describing a dataset should require the request to actually say
# "dataset(s)"/"data set(s)", not just contain a generic listing verb.
_DATASET_NOUN_PATTERN = re.compile(r"\b(datasets?|data sets?)\b", re.I)
_DATASET_DESCRIBE_HINTS = (
    "describe", "profile", "summar", "tell me about", "more about", "what's in", "what is in",
)
_DATASET_LIST_HINTS = ("list", "available", "show")

_BUILD_PATTERNS = re.compile(r"\b(build|fit|train|make a model|create model)\b", re.I)
_TUNE_PATTERNS = re.compile(r"\b(tune|optimi[sz]e|hyperparameter|best auc|best f1|grid)\b", re.I)
_EXPLAIN_PATTERNS = re.compile(r"\b(explain|interpret|pattern|patterns|importance|why|model card)\b", re.I)
_MODEL_FINDINGS_PATTERNS = re.compile(
    r"\b("
    r"summari[sz]e|summary|findings|what did you find|what have you found|"
    r"what does the model show|what did the model learn|model findings|"
    r"key takeaways|takeaways|insights|results summary|summarise"
    r")\b",
    re.I,
)
_PRUNE_PATTERNS = re.compile(r"\b(prune|remove pattern|remove low support|governance action)\b", re.I)
_PREDICT_PATTERNS = re.compile(r"\b(predict|predicts|predicting|predictions?|score applicants|score accounts|tabular output|table)\b", re.I)
_RESULT_METRIC_PATTERNS = re.compile(
    r"\b(precision|recall|f1|auc|roc|accuracy|balanced accuracy|average precision|confusion matrix|metric|metrics|performance)\b",
    re.I,
)
_PARAMETER_USED_PATTERNS = re.compile(
    r"\b(parameters? did you use|used parameters?|model parameters?|configuration|config|best parameters?|best params?|chosen parameters?)\b",
    re.I,
)
_REPRESENTATION_PATTERNS = re.compile(
    r"\b(interpretability|feature importance|feature importances|drivers|representation|composition|breakdown|original features?|augmented features?|augmented pairs?|pattern details?|details of patterns|patterns? breakup)\b",
    re.I,
)
_SUMMARY_STATS_PATTERNS = re.compile(
    r"\b(summary statistics|predictors?|describe predictors?|column summary|data profile|missingness|missing values|target distribution)\b",
    re.I,
)
_COMPARE_CONFIG_PATTERNS = re.compile(r"\b(compare|comparison|versus|vs\.?|against)\b", re.I)
_GOV_PATTERNS = re.compile(r"\b(governance|audit|model card|compliance)\b", re.I)
_DOCS_TOPIC_PATTERNS = re.compile(
    r"\b("
    r"hugiml|hugimlclassifier|api|docs?|documentation|usage|method|methods|"
    r"parameter|parameters|hyperparameter|hyperparameters|execution mode|feature mode|"
    r"pruning|prune|governance|audit|model card|classifier"
    r")\b",
    re.I,
)
_DOCS_QUESTION_PATTERNS = re.compile(
    r"\b(what|how|explain|tell me about|usage|api|docs?|documentation|parameters?)\b",
    re.I,
)
_MODEL_RESULT_CONTEXT_PATTERNS = re.compile(
    r"\b(this model|active model|built model|trained model|current model|findings|results|drivers|takeaways|predictions?)\b",
    re.I,
)
_BASELINE_PATTERNS = re.compile(
    r"\b(xgb|xgboost|lightgbm|lgbm|random forest|baseline|baselines|rulefit|ebm|logistic regression|sklearn model)\b",
    re.I,
)

_DATASET_REQUIRING_ACTIONS = {
    "describe_dataset",
    "build_model",
    "tune_hyperparameters",
    "generate_predictions",
    "generate_tabular_output",
}


def deterministic_refusal(user_text: str) -> ActionResult | None:
    """Return a canned refusal for requests outside the NLP interface scope."""

    text = (user_text or "").strip()
    if _CODE_EDIT_PATTERNS.search(text):
        # Allow harmless API questions that mention code only as examples.
        if not _API_HELP_PATTERNS.search(text):
            return refusal_result("code_modification_not_supported")
    if _BASELINE_PATTERNS.search(text) and (
        _BUILD_PATTERNS.search(text) or _TUNE_PATTERNS.search(text) or _PREDICT_PATTERNS.search(text)
    ):
        return refusal_result("baseline_model_not_supported")
    return None


def route_without_llm(user_text: str, *, repo_root: str | Path | None = None) -> ActionRequest | None:
    """Cheap deterministic fallback router.

    This is intentionally conservative.  More ambiguous requests can be routed
    through Ollama, but common commands work even without a live LLM runtime.

    When ``repo_root`` is given, a best-effort attempt is made to fill in
    ``dataset`` by matching a known dataset name mentioned in the text. This
    matters because actions like ``build_model`` *require* a dataset
    (``ActionRequest.validate``); without this, every fallback-routed request
    that doesn't separately pass ``--dataset`` would otherwise need to fail.
    """

    text = (user_text or "").strip()
    low = text.lower()
    if not text:
        return None
    if _CODE_EDIT_PATTERNS.search(text) and not _API_HELP_PATTERNS.search(text):
        return ActionRequest(action="refuse", reason="code_modification_not_supported")
    if _BASELINE_PATTERNS.search(text) and (
        _BUILD_PATTERNS.search(text) or _TUNE_PATTERNS.search(text) or _PREDICT_PATTERNS.search(text)
    ):
        return ActionRequest(action="refuse", reason="baseline_model_not_supported")

    request = _route_intent(text, low)
    if request is not None and request.action in _DATASET_REQUIRING_ACTIONS and not request.dataset:
        match = _match_known_dataset(text, repo_root=repo_root)
        if match:
            request.dataset = match
    return request


def _route_intent(text: str, low: str) -> ActionRequest | None:
    compare_request = _comparison_request(text, low)
    if compare_request is not None:
        return compare_request
    if _RESULT_METRIC_PATTERNS.search(text) or _PARAMETER_USED_PATTERNS.search(text) or _REPRESENTATION_PATTERNS.search(text):
        return ActionRequest(action="explain_model", question=text, metric=_infer_metric(low), limit=_infer_limit(low))
    if _is_documentation_question(text, low):
        return ActionRequest(action="answer_api_question", question=text)
    if _DATASET_NOUN_PATTERN.search(text) and any(hint in low for hint in _DATASET_LIST_HINTS):
        return ActionRequest(action="list_datasets", question=text)
    if (_DATASET_NOUN_PATTERN.search(text) and any(hint in low for hint in _DATASET_DESCRIBE_HINTS)) or _SUMMARY_STATS_PATTERNS.search(text):
        return ActionRequest(action="describe_dataset", question=text, limit=_infer_limit(low))
    # Build/tune are checked ahead of prune/governance: a message like "build
    # and tune this, then prune it" is most naturally continued by building
    # first -- the deterministic router only ever returns one action, so this
    # ordering picks the action that starts the workflow rather than the one
    # that ends it. (A live LLM does not have this limitation since it reads
    # the whole request; this only governs the no-LLM fallback path.)
    if _BUILD_PATTERNS.search(text):
        return ActionRequest(action="build_model", question=text, params={"model_params": _extract_model_params(text)}, metric=_infer_metric(low))
    if _TUNE_PATTERNS.search(text):
        return ActionRequest(action="tune_hyperparameters", question=text, metric=_infer_metric(low))
    if _PREDICT_PATTERNS.search(text):
        return ActionRequest(action="generate_tabular_output", question=text)
    if _MODEL_FINDINGS_PATTERNS.search(text):
        return ActionRequest(action="explain_model", question=text, limit=_infer_limit(low))
    if _PRUNE_PATTERNS.search(text):
        return ActionRequest(action="prune_patterns", question=text, min_support=0.02)
    if _GOV_PATTERNS.search(text):
        return ActionRequest(action="generate_governance_report", question=text)
    if _EXPLAIN_PATTERNS.search(text):
        return ActionRequest(action="explain_model", question=text, limit=_infer_limit(low))
    if _DATASET_NOUN_PATTERN.search(text):
        # Matched a dataset-ish phrase but none of the more specific hints
        # above -- default to describing it rather than dropping the request.
        return ActionRequest(action="describe_dataset", question=text)
    if _API_HELP_PATTERNS.search(text):
        return ActionRequest(action="answer_api_question", question=text)
    return None



def _is_documentation_question(text: str, low: str) -> bool:
    """True when the user wants docs/concept guidance, not to mutate a run.

    This prevents questions like "how does pruning work?" or "what governance
    artifacts are created?" from being routed to destructive/action handlers.
    """

    if not (_DOCS_QUESTION_PATTERNS.search(text) and _DOCS_TOPIC_PATTERNS.search(text)):
        return False
    if _MODEL_RESULT_CONTEXT_PATTERNS.search(text) and not any(
        term in low for term in ("api", "docs", "documentation", "parameter", "hyperparameter", "usage")
    ):
        return False
    # Imperative action phrases should still execute the workflow.
    if re.search(r"\b(generate|create|package|run|do|perform)\b.*\b(governance|audit|model card)\b", low):
        return False
    if re.search(r"\b(prune|remove)\b.*\b(now|this model|active model|patterns? below|low support)\b", low):
        return False
    return True


def _match_known_dataset(text: str, *, repo_root: str | Path | None) -> str | None:
    """Best-effort: find a known dataset name mentioned in free text.

    Cheap by design: only lists dataset *names* (``include_profiles=False``),
    never fits or loads data. Picks the longest matching name so a specific
    name (e.g. "churn_synthetic") wins over an accidental shorter overlap.
    """

    try:
        from .dataset_registry import DatasetRegistry  # local import: avoid a hard

        registry = DatasetRegistry(repo_root=repo_root)
        names = [info.name for info in registry.list_datasets(include_profiles=False)]
    except Exception:
        return None

    low = text.lower()
    best: str | None = None
    for name in names:
        token = name.lower().replace("_", " ")
        candidates = {name.lower(), token}
        if any(c and c in low for c in candidates):
            if best is None or len(name) > len(best):
                best = name
    return best


def _infer_limit(text: str) -> int:
    match = re.search(r"\btop\s+(\d+)\b|\blimit\s*[=:]?\s*(\d+)\b", text, re.I)
    if not match:
        return 10
    value = match.group(1) or match.group(2)
    try:
        return max(1, min(int(value), 100))
    except Exception:
        return 10


def _comparison_request(text: str, low: str) -> ActionRequest | None:
    if not _COMPARE_CONFIG_PATTERNS.search(text):
        return None
    modes = []
    for mode in ("patterns_only", "original_plus_patterns", "original_plus_interactions"):
        if mode in low:
            modes.append(mode)
    if len(modes) < 2:
        return None
    base = _extract_model_params(text)
    configs = []
    for mode in modes:
        params = dict(base)
        params["feature_mode"] = mode
        params["label"] = mode
        configs.append(params)
    return ActionRequest(
        action="compare_model_configs",
        question=text,
        metric=_infer_metric(low),
        params={"model_params_list": configs},
        limit=_infer_limit(low),
    )


def _extract_model_params(text: str) -> dict:
    out: dict[str, object] = {}
    patterns = {
        "B": r"\bB\s*(?:=|as\s+)\s*([-+]?\d+)",
        "L": r"\bL\s*(?:=|as\s+)\s*([-+]?\d+)",
        "topK": r"\btopK\s*(?:=|as\s+)\s*([-+]?\d+)",
        "G": r"\bG\s*(?:=|as\s+)\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)",
        "aug_feature_size": r"\baug_feature_size\s*(?:=|as\s+)\s*([-+]?\d+)",
        "interaction_relaxed_feature_size": r"\binteraction_relaxed_feature_size\s*(?:=|as\s+)\s*([-+]?\d+)",
        "n_jobs": r"\bn_jobs\s*(?:=|as\s+)\s*([-+]?\d+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.I)
        if not m:
            continue
        raw = m.group(1)
        out[key] = float(raw) if key == "G" else int(raw)
    for key in ("adaptive_binning", "augmented_pair_transforms", "interaction_relaxed_mining"):
        m = re.search(rf"\b{key}\s*(?:=|as\s+)\s*(true|false)", text, re.I)
        if m:
            out[key] = m.group(1).lower() == "true"
    m = re.search(r"\bfeature_mode\s*(?:=|as\s+)?\s*['\"]?([a-z_]+)['\"]?", text, re.I)
    if m and m.group(1) in {"patterns_only", "original_plus_patterns", "original_plus_interactions"}:
        out["feature_mode"] = m.group(1)
    return out


def _infer_metric(text: str) -> str | None:
    if "average precision" in text or "avg precision" in text:
        return "average_precision"
    if "auc" in text or "roc" in text:
        return "roc_auc"
    if "precision" in text:
        return "precision"
    if "recall" in text:
        return "recall"
    if "f1" in text:
        return "f1"
    if "balanced accuracy" in text:
        return "balanced_accuracy"
    if "accuracy" in text:
        return "accuracy"
    return None

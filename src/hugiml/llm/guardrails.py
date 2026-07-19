"""Guardrails and deterministic intent routing for the optional LLM add-on."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .schemas import ActionRequest, ActionResult, refusal_result

_CODE_EDIT_PATTERNS = re.compile(
    r"\b("
    r"modify|edit|change source|rewrite|refactor|implement|add code|"
    r"create a script|write a script|run shell|shell command|pip install|"
    r"delete file|overwrite file|commit|git|pull request|build wheel|rebuild package"
    r")\b",
    re.IGNORECASE,
)

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

_DATASET_NOUN_PATTERN = re.compile(r"\b(datasets?|data sets?)\b", re.I)
_DATASET_DESCRIBE_HINTS = (
    "describe", "profile", "summar", "tell me about", "more about", "what's in", "what is in",
)
_DATASET_LIST_HINTS = ("list", "available", "show")

_BUILD_PATTERNS = re.compile(
    r"\b(build|fit|train|construct|make a model|create(?: a)? model|produce(?: a)? model)\b",
    re.I,
)
_TUNE_COMMAND_PATTERNS = re.compile(
    r"\b(tune|optimi[sz]e|hyperparameter search|grid search|search the grid|select (?:the )?best|best auc|best f1)\b",
    re.I,
)
_RUN_MODEL_PATTERNS = re.compile(
    r"\b(run|execute|apply|use)\b.{0,60}\b(hugiml|model|grid|rpte|downstream estimator|base estimator)\b",
    re.I,
)
_EXPLAIN_PATTERNS = re.compile(r"\b(explain|interpret|pattern|patterns|importance|why|model card)\b", re.I)
_RPTE_RULE_REQUEST_PATTERNS = re.compile(
    r"("
    r"\b(?:show|display|list|print|view|give|provide)\b.{0,40}\brpte\b.{0,24}\b(?:rule|rules|tree|trees|leaf|leaves)\b|"
    r"\brpte\b.{0,24}\b(?:rule|rules|tree|trees|leaf|leaves)\b.{0,40}\b(?:show|display|list|print|view|give|provide)\b|"
    r"\b(?:rpte rule|rpte rules|rpte tree|rpte trees|rpte leaves)\b"
    r")",
    re.I,
)

# These vocabularies describe user intent rather than exact commands. The
# router combines them with active-model wording (selected, fitted, current,
# predictions, results, and similar terms), so ordinary paraphrases resolve to
# model evidence while conceptual questions continue to resolve to API help.
_ACTIVE_EVIDENCE_TERMS = {
    "active",
    "built",
    "chosen",
    "current",
    "fitted",
    "learned",
    "model",
    "prediction",
    "predictions",
    "result",
    "results",
    "selected",
    "trained",
    "used",
    "uses",
    "using",
}
_SHOW_EVIDENCE_TERMS = {
    "display",
    "give",
    "list",
    "print",
    "provide",
    "report",
    "show",
    "summarise",
    "summarize",
    "view",
    "which",
}
_DRIVER_TERMS = {
    "coefficient",
    "coefficients",
    "contribute",
    "contribution",
    "driver",
    "drivers",
    "drives",
    "driving",
    "effect",
    "effects",
    "evidence",
    "explain",
    "explanation",
    "factor",
    "factors",
    "impact",
    "impacts",
    "importance",
    "important",
    "influence",
    "influences",
    "matter",
    "reason",
    "reasoning",
    "reasons",
    "signal",
    "signals",
    "weight",
    "weights",
    "why",
}
_RULE_TERMS = {
    "branch",
    "branches",
    "condition",
    "conditions",
    "conjunction",
    "decision",
    "leaf",
    "leaves",
    "logic",
    "path",
    "paths",
    "route",
    "routes",
    "rule",
    "rules",
    "tree",
    "trees",
}
_CONFIGURATION_TERMS = {
    "config",
    "configuration",
    "hyperparameter",
    "hyperparameters",
    "option",
    "options",
    "parameter",
    "parameters",
    "setting",
    "settings",
    "setup",
    "value",
    "values",
}
_KNOWN_MODEL_PARAMETER_TERMS = {
    "adaptive_binning",
    "aug_feature_size",
    "augmented_pair_transforms",
    "b",
    "base_estimator",
    "convert_binary_to_categorical",
    "execution_mode",
    "feature_mode",
    "g",
    "interaction_relaxed_feature_size",
    "interaction_relaxed_mining",
    "l",
    "lr_solver",
    "n_jobs",
    "topk",
    "topk_budget_strict",
}
_METRIC_TERMS = {
    "accuracy",
    "auc",
    "f1",
    "metric",
    "metrics",
    "performance",
    "precision",
    "recall",
    "roc",
    "score",
    "scores",
}
_PATTERN_TERMS = {"hug", "inventory", "pattern", "patterns", "representation"}
_CONCEPTUAL_QUESTION_STARTS = (
    "what is ",
    "what are ",
    "what does ",
    "how does ",
    "how do ",
    "how is ",
    "explain how ",
    "explain ",
    "tell me about ",
    "define ",
)
_IMPERATIVE_MODEL_COMMAND_PATTERNS = re.compile(
    r"^\s*(?:please\s+)?(?:build|fit|train|construct|run|execute|apply|use|make|create|produce)\b",
    re.I,
)
_CONFIGURE_MODEL_COMMAND_PATTERNS = re.compile(
    r"^\s*(?:please\s+)?(?:set|enable|disable|turn\s+on|turn\s+off)\b",
    re.I,
)
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
    r"grid|performance_ho|performance ho|interpretability_ho|interpretability ho|interpretability|rpte|downstream estimator|base estimator|"
    r"augmented pairs?|augmented_pair_transforms|interaction relaxed mining|interaction_relaxed_mining|"
    r"adaptive binning|adaptive_binning|convert_binary_to_categorical|topk|feature mode|lr solver|"
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
_RPTE_PATTERNS = re.compile(
    r"("
    r"\brpte\b|\bresidual pattern tree ensemble\b|\bresidual pattern trees?\b|"
    r"\bleaf[ -]?wise(?: bounded lookahead)?\b|\bbounded lookahead(?: rpte)?\b|"
    r"\btree(?:[ -]based| ensemble)? downstream\b|\bhigher[ -]?order downstream\b|"
    r"LeafWiseBoundedLookaheadRPTEFeatureLR"
    r")",
    re.I,
)
_LINEAR_DOWNSTREAM_PATTERNS = re.compile(
    r"("
    r"\b(no rpte|without rpte|lr downstream|logistic(?: regression)? downstream|linear downstream|built[ -]?in logistic regression)\b|"
    r"base_estimator\s*=\s*None"
    r")",
    re.I,
)
_DOWNSTREAM_ROLE_PATTERNS = re.compile(r"\b(downstream|base estimator|within hugiml|hugiml)\b", re.I)

_DATASET_REQUIRING_ACTIONS = {
    "describe_dataset",
    "build_model",
    "tune_hyperparameters",
    "generate_predictions",
    "generate_tabular_output",
}


def infer_model_request_hints(user_text: str) -> dict[str, Any]:
    """Extract executable model preferences from common natural-language forms.

    The returned values are plain schema-safe strings. The orchestrator resolves
    them into actual grid definitions and estimator instances only at execution.
    """

    text = (user_text or "").strip()
    low = text.lower()
    return {
        "grid_name": _infer_grid_name(text),
        "downstream_estimator": _infer_downstream_estimator(text),
        "strategy": _infer_strategy(low),
        "model_params": _extract_model_params(text),
        "is_execution_command": _is_model_execution_command(text),
    }


def deterministic_refusal(user_text: str) -> ActionResult | None:
    """Return a canned refusal for requests outside the NLP interface scope."""

    text = (user_text or "").strip()
    if _CODE_EDIT_PATTERNS.search(text) and not _API_HELP_PATTERNS.search(text):
        return refusal_result("code_modification_not_supported")
    if _is_unsupported_baseline_request(text):
        return refusal_result("baseline_model_not_supported")
    return None


def route_without_llm(user_text: str, *, repo_root: str | Path | None = None) -> ActionRequest | None:
    """Route common HUGIML requests without requiring a local language model."""

    text = (user_text or "").strip()
    low = text.lower()
    if not text:
        return None
    if _CODE_EDIT_PATTERNS.search(text) and not _API_HELP_PATTERNS.search(text):
        return ActionRequest(action="refuse", reason="code_modification_not_supported")
    if _is_unsupported_baseline_request(text):
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

    if _DATASET_NOUN_PATTERN.search(text) and any(hint in low for hint in _DATASET_LIST_HINTS):
        return ActionRequest(action="list_datasets", question=text)
    if (_DATASET_NOUN_PATTERN.search(text) and any(hint in low for hint in _DATASET_DESCRIBE_HINTS)) or _SUMMARY_STATS_PATTERNS.search(text):
        return ActionRequest(action="describe_dataset", question=text, limit=_infer_limit(low))

    # Explicit commands must take precedence over API-topic words such as
    # "RPTE" and "downstream estimator". Instructional questions such as
    # "how do I build ..." still remain documentation requests because they
    # do not begin with an imperative verb.
    if _IMPERATIVE_MODEL_COMMAND_PATTERNS.search(text) or _CONFIGURE_MODEL_COMMAND_PATTERNS.search(text):
        model_request = _model_execution_request(text, low)
        if model_request is not None:
            return model_request

    # Active-model questions are routed by meaning, then interpreted against
    # the fitted estimator by the orchestrator. This lets "what drives the
    # predictions?" surface RPTE rules for an RPTE model, linear coefficients
    # for an LR model, and the relevant path evidence for other configurations.
    explanation_request = _active_model_explanation_request(text, low)
    if explanation_request is not None:
        return explanation_request

    # Questions about a grid or estimator remain documentation requests.
    if _is_documentation_question(text, low):
        return ActionRequest(action="answer_api_question", question=text)

    model_request = _model_execution_request(text, low)
    if model_request is not None:
        return model_request

    if _RESULT_METRIC_PATTERNS.search(text) or _PARAMETER_USED_PATTERNS.search(text) or _REPRESENTATION_PATTERNS.search(text):
        return ActionRequest(
            action="explain_model",
            question=text,
            metric=_infer_metric(low),
            limit=_infer_limit(low),
            params={"focus": _infer_explanation_focus(text, low) or "overview"},
        )
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
        return ActionRequest(action="describe_dataset", question=text)
    if _API_HELP_PATTERNS.search(text):
        return ActionRequest(action="answer_api_question", question=text)
    return None


def _semantic_tokens(text: str) -> set[str]:
    """Return lightweight normalized tokens for deterministic intent routing."""

    tokens = set(re.findall(r"[a-z0-9_]+", (text or "").lower().replace("-", " ")))
    expanded = set(tokens)
    for token in list(tokens):
        if token.endswith("ies") and len(token) > 4:
            expanded.add(token[:-3] + "y")
        if token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
        if token.endswith("ing") and len(token) > 5:
            expanded.add(token[:-3])
        if token.endswith("ed") and len(token) > 4:
            expanded.add(token[:-2])
    return expanded


def _mentions_augmented_pair_path(low: str, tokens: set[str]) -> bool:
    if "augmented_pair" in low or "augmented pair" in low or "pairwise transform" in low:
        return True
    if tokens & {"operator", "operators", "transform", "transforms", "transformation", "transformations"}:
        return bool(tokens & {"augment", "augmented", "pair", "pairwise", "selected"})
    return bool(tokens & {"pair", "pairwise"}) and bool(
        tokens & {"augment", "augmented", "difference", "effect", "effects", "product", "sum"}
    )


def _mentions_interaction_relaxed_path(low: str, tokens: set[str]) -> bool:
    if "interaction_relaxed" in low or "interaction relaxed" in low:
        return True
    if tokens & {"survivor", "survivors"}:
        return True
    return bool(tokens & {"admission", "admitted", "relaxed"}) and bool(
        tokens & {"feature", "interaction", "mining", "pattern", "source"}
    )


def _is_conceptual_question(text: str, low: str, tokens: set[str]) -> bool:
    stripped = low.strip()
    if any(stripped.startswith(prefix) for prefix in _CONCEPTUAL_QUESTION_STARTS):
        # These forms ask about the fitted run despite beginning with an
        # interrogative: "what parameters were used" and "what drives the
        # predictions", for example.
        if tokens & {
            "active",
            "chosen",
            "current",
            "driver",
            "drivers",
            "drives",
            "fitted",
            "prediction",
            "predictions",
            "selected",
            "used",
        }:
            return False
        return True
    return bool(tokens & {"api", "define", "docs", "documentation", "meaning", "usage"})


def _infer_explanation_focus(text: str, low: str) -> str | None:
    """Infer the kind of fitted-model evidence requested by meaning."""

    tokens = _semantic_tokens(text)
    if not tokens:
        return None
    configuration_context = tokens & (
        _ACTIVE_EVIDENCE_TERMS
        | _SHOW_EVIDENCE_TERMS
        | {"chosen", "disabled", "enabled", "selected", "set", "used", "value", "values"}
    )
    # Explicit parameter/configuration questions take precedence over path
    # evidence. For example, "which value of augmented_pair_transforms was
    # used?" asks for the fitted configuration, while "which augmented pairs
    # were selected?" asks for the path evidence.
    if configuration_context and (
        tokens & _CONFIGURATION_TERMS or tokens & _KNOWN_MODEL_PARAMETER_TERMS
    ):
        return "configuration"
    if _mentions_augmented_pair_path(low, tokens):
        return "augmented_pairs"
    if _mentions_interaction_relaxed_path(low, tokens):
        return "interaction_relaxed"
    if tokens & _METRIC_TERMS and tokens & (_ACTIVE_EVIDENCE_TERMS | _SHOW_EVIDENCE_TERMS):
        return "metrics"
    if tokens & _RULE_TERMS and tokens & (
        _ACTIVE_EVIDENCE_TERMS | _SHOW_EVIDENCE_TERMS | _DRIVER_TERMS | {"rpte"}
    ):
        return "rules"
    if tokens & _DRIVER_TERMS and tokens & (
        _ACTIVE_EVIDENCE_TERMS
        | _SHOW_EVIDENCE_TERMS
        | {"decision", "outcome", "prediction", "predictions"}
    ):
        return "prediction_drivers"
    if tokens & _PATTERN_TERMS and tokens & (
        _ACTIVE_EVIDENCE_TERMS | _SHOW_EVIDENCE_TERMS | _DRIVER_TERMS
    ):
        return "patterns"
    if _MODEL_FINDINGS_PATTERNS.search(text):
        return "overview"
    return None


def _active_model_explanation_request(text: str, low: str) -> ActionRequest | None:
    tokens = _semantic_tokens(text)
    focus = _infer_explanation_focus(text, low)
    if focus is None or _is_conceptual_question(text, low, tokens):
        return None
    params: dict[str, Any] = {"focus": focus}
    if "rpte" in tokens:
        params["requested_representation"] = "rpte"
    return ActionRequest(
        action="explain_model",
        question=text,
        metric=_infer_metric(low),
        limit=_infer_limit(low),
        params=params,
    )


def _model_execution_request(text: str, low: str) -> ActionRequest | None:
    if not _is_model_execution_command(text):
        return None

    grid_name = _infer_grid_name(text)
    downstream = _infer_downstream_estimator(text)
    model_params = _extract_model_params(text)
    params: dict[str, Any] = {}
    if model_params:
        params["model_params"] = model_params
    if grid_name:
        params["grid_name"] = grid_name
    if downstream:
        params["downstream_estimator"] = downstream

    # A named grid denotes candidate search even when the natural-language verb
    # is "build" or "run". Direct estimator requests without a grid remain a
    # single model build unless tuning is explicitly requested.
    action = "tune_hyperparameters" if grid_name or _TUNE_COMMAND_PATTERNS.search(text) else "build_model"
    strategy = _infer_strategy(low)
    if downstream == "rpte" and strategy == "balanced":
        strategy = "performance"
    return ActionRequest(
        action=action,
        question=text,
        strategy=strategy,
        params=params,
        metric=_infer_metric(low),
    )


def _is_model_execution_command(text: str) -> bool:
    if _BUILD_PATTERNS.search(text) or _TUNE_COMMAND_PATTERNS.search(text):
        return True
    if _RUN_MODEL_PATTERNS.search(text):
        return True
    # Common concise commands omit the word "model": "run performance_ho",
    # "use RPTE downstream", "enable augmented pairs", and "set G=0.01"
    # still clearly request execution on the selected or active dataset.
    if re.search(r"\b(run|execute|apply|use)\b", text, re.I) and (
        _infer_grid_name(text) is not None or _infer_downstream_estimator(text) is not None
    ):
        return True
    if _CONFIGURE_MODEL_COMMAND_PATTERNS.search(text):
        low = text.lower().replace("-", " ")
        tokens = _semantic_tokens(text)
        return bool(
            _extract_explicit_model_params(text)
            or _mentions_augmented_pair_path(low, tokens)
            or _mentions_interaction_relaxed_path(low, tokens)
            or _infer_grid_name(text)
            or _infer_downstream_estimator(text)
        )
    return False


def _infer_grid_name(text: str) -> str | None:
    low = (text or "").lower()
    try:
        from hugiml.hyperparameter_configs import list_hugiml_grids

        names = list_hugiml_grids()
    except Exception:
        names = ["performance", "interpretability", "performance_ho", "interpretability_ho"]

    # Longest names first prevents "performance" from swallowing
    # "performance_ho".
    for name in sorted(names, key=len, reverse=True):
        forms = {
            name.lower(),
            name.lower().replace("_", " "),
            name.lower().replace("_", "-"),
        }
        if name == "performance_ho":
            forms.update({"performance higher order", "higher order performance"})
        elif name == "interpretability_ho":
            forms.update({"interpretability higher order", "higher order interpretability"})
        for form in forms:
            if re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", low):
                # A bare strategy word is treated as a named grid only when the
                # user actually says grid/search, except for unambiguous names
                # such as performance_ho.
                if "_" in name or "grid" in low or "search" in low:
                    return name
    return None


def _infer_downstream_estimator(text: str) -> str | None:
    if _LINEAR_DOWNSTREAM_PATTERNS.search(text):
        return "logistic_regression"
    if _RPTE_PATTERNS.search(text):
        return "rpte"
    return None


def _infer_strategy(low: str) -> str:
    normalised = low.replace("-", "_").replace(" ", "_")
    if "small_memory" in normalised or "low_memory" in normalised:
        return "small_memory"
    for strategy in ("interpretability", "performance", "balanced", "fast"):
        if re.search(rf"\b{strategy}\b", low):
            return strategy
    return "balanced"


def _is_supported_hugiml_downstream_request(text: str) -> bool:
    downstream = _infer_downstream_estimator(text)
    if downstream is None:
        return False
    if downstream == "rpte":
        return True
    return bool(_DOWNSTREAM_ROLE_PATTERNS.search(text))


def _is_unsupported_baseline_request(text: str) -> bool:
    if not _BASELINE_PATTERNS.search(text):
        return False
    if not (_is_model_execution_command(text) or _PREDICT_PATTERNS.search(text)):
        return False
    return not _is_supported_hugiml_downstream_request(text)


def _is_documentation_question(text: str, low: str) -> bool:
    if not (_DOCS_QUESTION_PATTERNS.search(text) and _DOCS_TOPIC_PATTERNS.search(text)):
        return False
    if _MODEL_RESULT_CONTEXT_PATTERNS.search(text) and not any(
        term in low for term in ("api", "docs", "documentation", "parameter", "hyperparameter", "usage")
    ):
        return False
    if re.search(r"\b(generate|create|package|run|do|perform)\b.*\b(governance|audit|model card)\b", low):
        return False
    if re.search(r"\b(prune|remove)\b.*\b(now|this model|active model|patterns? below|low support)\b", low):
        return False
    return True


def _match_known_dataset(text: str, *, repo_root: str | Path | None) -> str | None:
    try:
        from .dataset_registry import DatasetRegistry

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


def _extract_explicit_model_params(text: str) -> dict[str, Any]:
    """Extract only parameter values explicitly stated by the user."""

    out: dict[str, Any] = {}
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
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        raw = match.group(1)
        out[key] = float(raw) if key == "G" else int(raw)
    for key in (
        "adaptive_binning",
        "augmented_pair_transforms",
        "convert_binary_to_categorical",
        "interaction_relaxed_mining",
        "topk_budget_strict",
    ):
        match = re.search(rf"\b{key}\s*(?:=|as\s+)\s*(true|false)", text, re.I)
        if match:
            out[key] = match.group(1).lower() == "true"

    match = re.search(r"\bfeature_mode\s*(?:=|as\s+)?\s*['\"]?([a-z_]+)['\"]?", text, re.I)
    if match and match.group(1) in {"patterns_only", "original_plus_patterns", "original_plus_interactions"}:
        out["feature_mode"] = match.group(1)
    elif re.search(r"\bpatterns?[ _-]?only\b", text, re.I):
        out["feature_mode"] = "patterns_only"
    elif re.search(r"\boriginal(?:s| features?)?\s*(?:plus|and|with)\s*patterns?\b", text, re.I):
        out["feature_mode"] = "original_plus_patterns"
    elif re.search(r"\boriginal(?:s| features?)?\s*(?:plus|and|with)\s*interactions?\b", text, re.I):
        out["feature_mode"] = "original_plus_interactions"
    return out


def _extract_model_params(text: str) -> dict[str, Any]:
    """Extract explicit values plus semantic path defaults for model execution."""

    out = _extract_explicit_model_params(text)
    low = text.lower().replace("-", " ")
    tokens = _semantic_tokens(text)
    disable_pair = bool(
        re.search(r"\b(?:disable|without|turn\s+off|no)\b.{0,32}\b(?:augmented|pair)", low)
    )
    disable_relaxed = bool(
        re.search(r"\b(?:disable|without|turn\s+off|no)\b.{0,32}\b(?:interaction|relaxed|survivor)", low)
    )
    if _mentions_augmented_pair_path(low, tokens):
        out.setdefault("L", 2)
        out.setdefault("augmented_pair_transforms", not disable_pair)
        if not disable_pair:
            out.setdefault("interaction_relaxed_mining", False)
            out.setdefault("convert_binary_to_categorical", False)
    if _mentions_interaction_relaxed_path(low, tokens):
        out.setdefault("L", 2)
        out.setdefault("interaction_relaxed_mining", not disable_relaxed)
        if not disable_relaxed:
            out.setdefault("augmented_pair_transforms", False)
            out.setdefault("convert_binary_to_categorical", True)
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

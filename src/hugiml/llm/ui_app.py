"""Streamlit UI for the optional HUGIML natural-language workbench.

This module is packaged so the installed ``hugiml-llm`` command can launch the
UI directly.  The repo-root ``LLM/ui/hugiml_llm_chat.py`` file is a thin wrapper
for source-checkout usage.
"""

from __future__ import annotations

import ast
import html
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:  # Plotly is part of the optional llm extra.
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - fallback message shown in UI
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]

from hugiml.llm import ActionRequest, DatasetRegistry, HUGIMLActionOrchestrator
from hugiml.llm.evidence import (
    downstream_redundancy_audit_rows,
    rpte_rule_rows_to_importance_rows,
)
from hugiml.llm.runtime import (
    DEFAULT_OLLAMA_MODEL,
    FALLBACK_OLLAMA_MODEL,
    LIGHT_MODE_MODEL,
    check_ollama,
    get_memory_info,
    get_model_catalog,
    is_below_minimum_llm_model,
    is_lightweight_supported_model,
    model_availability,
    recommend_profile,
)
from hugiml.llm.ui_service import run_prompt


def _guess_repo_root() -> Path | None:
    """Return a source-checkout root when available; installed wheels return None."""

    candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for base in candidates:
        if (base / "pyproject.toml").exists() and (base / "src" / "hugiml").exists():
            return base
        if (base / "LLM").exists() and (base / "src" / "hugiml").exists():
            return base
    return None


REPO_ROOT = _guess_repo_root()

_PALETTE = ["#2C6E63", "#B5651D", "#6B5CA0", "#5B6358", "#A1442E"]
_COLOR_MAP = {
    "positive": "#B5651D",
    "negative": "#2C6E63",
    "toward positive class": "#B5651D",
    "toward negative class": "#2C6E63",
    "before": "#1C2420",
    "after": "#2C6E63",
}

st.set_page_config(
    page_title="HUGIML LLM Chat",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)

_PROFILE_LABELS = {
    "no-llm": "Deterministic workspace",
    "light-mode": "Light mode workspace",
    "lightweight-llm": "Default LLM workspace",
    "fallback-llm": "Fallback LLM workspace",
    "minimum-llm": "Minimum LLM workspace",
    "balanced": "Balanced LLM workspace",
    "expanded": "Expanded LLM workspace",
    "large-context": "Large-context LLM workspace",
}

_PROFILE_COPY = {
    "no-llm": "Less than about 3.5 GB free RAM. Use deterministic routing without Ollama.",
    "light-mode": "Light mode uses gemma3:1b for compact local answers when memory is tight.",
    "lightweight-llm": "Default local mode uses qwen3:1.7b for polished grounded answers, with deterministic routing still available.",
    "fallback-llm": "Fallback mode uses llama3.2:1b before returning to deterministic routing.",
    "minimum-llm": "Minimum recommended local model for a fuller conversational LLM experience.",
    "balanced": "Good default for summaries, follow-up questions, and interpretation.",
    "expanded": "Stronger local model with richer grounded interpretation when more RAM is free.",
    "large-context": "Best for long model cards, larger result summaries, and richer one-pass explanations.",
}

_ALLOWED_SUGGESTIONS = [
    "List the available datasets",
    "Describe the active dataset",
    "Build a HUGIML model",
    "Tune this dataset for ROC AUC",
    "Show prediction table for 10 rows",
    "Explain the strongest patterns",
    "Prune low-support patterns",
    "Generate governance report",
]

_EXAMPLE_REQUESTS = [
    ("List data", "List the available datasets"),
    ("Describe data", "Describe the active dataset"),
    ("Build model", "Build a HUGIML model on this dataset"),
    ("Tune model", "Tune this dataset for ROC AUC"),
    ("Prediction table", "Generate a prediction table for 10 held-out rows"),
    ("Top patterns", "Explain the strongest patterns in plain English"),
    ("One prediction", "Explain one prediction from the held-out set"),
    ("Prune rules", "Prune low-support patterns and show the before/after metrics"),
    ("Governance", "Generate a governance report for the active model"),
    ("API help", "What HUGIML APIs and actions can I use here?"),
]

_CLI_COMMANDS = [
    ("Launch UI", "hugiml-llm"),
    ("Check setup", "hugiml-llm status"),
    ("List datasets", "hugiml-llm list-datasets"),
    ("Terminal chat", "hugiml-llm chat --dataset churn_synthetic --no-llm"),
    (
        "One-shot request",
        'hugiml-llm ask "build a model and explain it" --dataset churn_synthetic --no-llm',
    ),
    ("Demo HTML", "hugiml-llm demo-html"),
]


@st.cache_resource(show_spinner=False)
def get_orchestrator() -> HUGIMLActionOrchestrator:
    return HUGIMLActionOrchestrator(repo_root=REPO_ROOT)


def main() -> None:
    _inject_css()
    st.session_state["_hugiml_chart_counter"] = 0
    orch = get_orchestrator()
    registry: DatasetRegistry = orch.registry

    include_benchmarks = bool(st.session_state.get("include_benchmark_catalog", False))
    dataset_rows = [
        d.to_dict()
        for d in registry.list_datasets(
            include_profiles=True,
            include_benchmarks=include_benchmarks,
        )
    ]
    builtin_rows = [d for d in dataset_rows if d.get("source") == "llm_builtin"]
    mem = get_memory_info()
    profile = recommend_profile(mem, repo_root=REPO_ROOT)
    status = check_ollama(timeout=0.8)

    _hero(mem, profile, status, len(builtin_rows), len(dataset_rows), include_benchmarks)

    with st.sidebar:
        selected_dataset, model_choice = _sidebar(registry, dataset_rows, mem, profile, status)

    _chat_panel(orch, selected_dataset, model_choice, include_benchmarks=include_benchmarks)

    with st.expander("Quick view and workspace details", expanded=False):
        _overview_panel(dataset_rows, selected_dataset, mem, profile, status)

    with st.expander("Dataset details", expanded=False):
        _data_panel(registry, dataset_rows, selected_dataset)

    with st.expander("Active model evidence", expanded=False):
        _model_panel(orch)

    with st.expander("Governance actions", expanded=False):
        _governance_panel(orch)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          :root{
            --paper:#ECEFE6; --panel:#F8FAF3; --ink:#1C2420; --soft:#5B6358;
            --rule:#C7CFC0; --teal:#2C6E63; --teal-soft:#E4EEEA;
            --amber:#B5651D; --amber-soft:#F3E7D8; --violet:#6B5CA0; --violet-soft:#ECE8F5;
          }
          .stApp { background: var(--paper); color: var(--ink); }
          .block-container { padding-top: 1.25rem; padding-bottom: 4rem; max-width: 1240px; }
          section[data-testid="stSidebar"] { background: #F8FAF3; border-right: 1px solid var(--rule); }
          .hugiml-hero {
            border: 1px solid var(--rule); background: linear-gradient(135deg, #F8FAF3 0%, #ECEFE6 60%, #E4EEEA 100%);
            padding: 1.5rem 1.6rem; margin: 0 0 1rem; box-shadow: 0 1px 0 rgba(28,36,32,.04);
          }
          .hugiml-eyebrow { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .73rem; letter-spacing: .12em; text-transform: uppercase; color: var(--soft); margin-bottom: .4rem; }
          .hugiml-title { font-family: Georgia, 'Times New Roman', serif; font-size: clamp(2rem, 4vw, 3.1rem); line-height: 1.05; font-weight: 600; margin: 0 0 .55rem; letter-spacing: -.02em; }
          .hugiml-dek { max-width: 780px; color: var(--soft); font-size: 1.03rem; margin-bottom: .85rem; }
          .chip-row { display:flex; flex-wrap: wrap; gap: .45rem; }
          .chip { border:1px solid var(--rule); background: #fff; padding: .32rem .6rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.78rem; color: var(--soft); }
          .chip b { color: var(--ink); }
          .panel-card { border: 1px solid var(--rule); background: var(--panel); padding: 1rem; margin-bottom: .8rem; }
          .panel-card h3 { margin: 0 0 .35rem; font-family: Georgia, 'Times New Roman', serif; font-size: 1.15rem; }
          .panel-note { color: var(--soft); font-size: .92rem; margin: 0 0 .75rem; }
          .stage-list { display:grid; gap:.45rem; }
          .stage-item { display:flex; align-items:flex-start; gap:.65rem; background:#fff; border:1px solid var(--rule); padding:.68rem .75rem; }
          .stage-num { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--amber); border:1px solid var(--amber); padding:.03rem .34rem; font-size:.78rem; }
          .stage-text b { display:block; font-weight:600; }
          .stage-text span { color:var(--soft); font-size:.87rem; }
          .scope-ok { border-left: 3px solid var(--teal); }
          .scope-no { border-left: 3px solid var(--violet); }
          .dataset-pill { border:1px solid var(--rule); background:white; padding:.35rem .55rem; font-size:.82rem; display:inline-block; margin:.15rem .2rem .15rem 0; }
          div[data-testid="stMetric"] { background: #fff; border:1px solid var(--rule); padding:.62rem .72rem; min-width:0; overflow:hidden; }
          div[data-testid="stMetric"] label { color: var(--soft); font-size:.78rem; line-height:1.15; white-space:normal; }
          div[data-testid="stMetricValue"] { font-size: clamp(.92rem, 1.15vw, 1.25rem); line-height:1.18; white-space: normal; overflow-wrap:anywhere; word-break:break-word; }
          div[data-testid="stMetricDelta"] { font-size:.78rem; }
          div[data-testid="stDataFrame"] { max-width:100%; overflow:hidden; }
          .stDataFrame, .stTable { font-size:.88rem; }
          [data-testid="stVerticalBlock"] { min-width:0; }
          .small-muted { color: var(--soft); font-size:.86rem; }
          .qa-tip { border-left:3px solid var(--teal); background: var(--teal-soft); padding:.8rem .95rem; margin:.55rem 0; }
          .refusal-box { border-left:3px solid var(--violet); background: var(--violet-soft); padding:.8rem .95rem; }
          .command-row { display:grid; grid-template-columns: 1fr; gap:.45rem; margin:.45rem 0 .85rem; }
          .command-card { border:1px solid var(--rule); background:#fff; padding:.58rem .68rem; }
          .command-card b { display:block; color:var(--ink); margin-bottom:.18rem; }
          .command-card code { display:block; white-space:pre-wrap; overflow-wrap:anywhere; color:var(--soft); font-size:.78rem; }
          .example-note { color:var(--soft); font-size:.86rem; margin:.2rem 0 .55rem; }
          .viz-shell { border:1px solid var(--rule); background:#fff; padding:.65rem .75rem; margin:.55rem 0 .85rem; }
          .viz-title { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); margin-bottom:.35rem; }
          .evidence-grid { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:.6rem; margin:.75rem 0; }
          .evidence-card { background:#fff; border:1px solid var(--rule); padding:.75rem .85rem; min-height:86px; }
          .evidence-card b { display:block; font-size:1.4rem; margin:.1rem 0; }
          .evidence-card span { color:var(--soft); font-size:.84rem; }

          .chat-thread { display:grid; gap:.85rem; margin: .85rem 0 1rem; }
          .qa-empty { border:1px dashed var(--rule); background:#fff; padding:1rem; color:var(--soft); }
          .qa-card { border:1px solid var(--rule); background:#fff; padding:.82rem .95rem; margin:.45rem 0; }
          .qa-card.question { border-left:4px solid var(--teal); background:var(--teal-soft); }
          .qa-card.answer { border-left:4px solid var(--amber); background:#fff; }
          .qa-label { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; color:var(--soft); margin-bottom:.28rem; }
          .qa-text { font-size:.98rem; line-height:1.55; color:var(--ink); white-space:pre-wrap; overflow-wrap:anywhere; }
          .qa-answer-body { border:1px solid var(--rule); border-top:0; background:#fff; padding:.72rem .85rem; margin:-.45rem 0 .85rem; }
          .chat-composer-card { border:1px solid var(--rule); background:var(--panel); padding:.9rem 1rem; margin:1rem 0 .85rem; }
          .chat-composer-title { font-family:Georgia, 'Times New Roman', serif; font-size:1.12rem; font-weight:600; margin:0 0 .2rem; }
          .bottom-tools { border-top:1px solid var(--rule); padding-top:.7rem; margin-top:.7rem; }
          @media (max-width: 900px) { .evidence-grid { grid-template-columns:1fr; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero(
    mem: Any,
    profile: Any,
    status: dict[str, Any],
    builtin_count: int,
    visible_count: int,
    include_benchmarks: bool,
) -> None:
    profile_label = _PROFILE_LABELS.get(profile.name, profile.name)
    ollama_state = "ready" if status.get("ok") else "not connected"
    catalog_note = (
        "including benchmark catalog" if include_benchmarks else "built-in first-run catalog"
    )
    st.markdown(
        f"""
        <div class="hugiml-hero">
          <div class="hugiml-eyebrow">hugiml-core add-on · natural-language workbench</div>
          <div class="hugiml-title">HUGIML Chat</div>
          <div class="hugiml-dek">Ask, inspect the answer, then ask follow-up questions in one continuous LLM workbench over existing HUGIML APIs.</div>
          <div class="chip-row">
            <span class="chip">built-in datasets <b>{builtin_count}</b></span>
            <span class="chip">visible catalog <b>{visible_count}</b> · {catalog_note}</span>
            <span class="chip">available RAM <b>{_fmt_gb(mem.available_gb)}</b></span>
            <span class="chip">workspace <b>{profile_label}</b></span>
            <span class="chip">Ollama <b>{ollama_state}</b></span>
            <span class="chip">modeling <b>HUGIML only</b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar(
    registry: DatasetRegistry,
    dataset_rows: list[dict[str, Any]],
    mem: Any,
    profile: Any,
    status: dict[str, Any],
) -> tuple[str | None, str]:
    st.header("Workspace")
    profile_label = _PROFILE_LABELS.get(profile.name, profile.name)
    st.info(f"**{profile_label}**  \n{_PROFILE_COPY.get(profile.name, profile.description)}")
    c1, c2 = st.columns(2)
    c1.metric("Total RAM", _fmt_gb(mem.total_gb))
    c2.metric("Free RAM", _fmt_gb(mem.available_gb))

    model_choice = _model_picker(mem, profile, status)

    st.divider()
    st.header("Datasets")
    selected_dataset = _dataset_picker(dataset_rows)
    st.divider()
    _upload_panel(registry)
    st.divider()
    _cli_commands_panel()
    return selected_dataset, model_choice


def _model_picker(mem: Any, profile: Any, status: dict[str, Any]) -> str:
    """Choose the routing model from a dropdown.

    Streamlit does not support disabled options inside ``selectbox``.  To keep
    the UX simple and safe, the dropdown contains only choices that are usable
    on the current machine.  The full catalog is still shown below with the
    reason each unavailable model is not selectable.
    """

    default_model = "deterministic router only"
    installed = set(status.get("models") or [])
    catalog = get_model_catalog(REPO_ROOT)
    catalog_by_model = {option.model: option for option in catalog}
    recommended_model = getattr(profile, "recommended_model", "") or ""

    st.markdown("#### Local model")
    st.caption(
        "Choose deterministic routing or an installed recommended Ollama model. Default is `qwen3:1.7b`; light mode is `gemma3:1b`; fallback is `llama3.2:1b`. Other sub-3B manual models stay disabled."
    )

    rows: list[dict[str, Any]] = []
    selectable_values: list[str] = [default_model]
    labels: dict[str, str] = {default_model: "Deterministic routing only · no Ollama call"}

    for option in catalog:
        avail = model_availability(
            option,
            mem,
            installed,
            ollama_ok=bool(status.get("ok")),
        )
        is_recommended = option.model == recommended_model
        if avail["selectable"]:
            selectable_values.append(option.model)
            label = f"{option.label} · {option.model}"
            if is_recommended:
                label += " · recommended"
            labels[option.model] = label
        rows.append(
            {
                "model": option.model,
                "tier": _PROFILE_LABELS.get(option.profile, option.profile),
                "min free RAM": f"{option.min_available_gb:.1f} GB",
                "status": "selectable" if avail["selectable"] else str(avail["reason"]),
                "installed": "yes" if avail["installed"] else "no",
            }
        )

    # Preserve the older behavior where already-pulled Ollama models are visible.
    # Explicitly supported lightweight models such as qwen3:1.7b are selectable;
    # other sub-3B manual models remain disabled. Unknown/manual models are
    # allowed only when they clear the same minimum-memory gate as the 3B tier.
    for model_name in sorted(installed):
        if model_name in catalog_by_model:
            continue
        free_gb = getattr(mem, "available_gb", None)
        lightweight = is_lightweight_supported_model(model_name)
        min_free_gb = 5.0 if lightweight else 6.0
        too_small = is_below_minimum_llm_model(model_name)
        selectable = (
            bool(status.get("ok")) and not too_small and (free_gb is None or free_gb >= min_free_gb)
        )
        if selectable:
            selectable_values.append(model_name)
            labels[model_name] = f"Installed Ollama model · {model_name} · manual"
        rows.append(
            {
                "model": model_name,
                "tier": "manual",
                "min free RAM": f"{min_free_gb:.1f} GB*",
                "status": (
                    "unsupported sub-3B manual model; use deterministic routing"
                    if too_small
                    else (
                        "selectable"
                        if selectable
                        else f"manual model needs ≥{min_free_gb:.1f} GB free RAM"
                    )
                ),
                "installed": "yes",
            }
        )

    selected = str(st.session_state.get("_hugiml_selected_model") or "").strip()
    if not selected:
        selected = _preferred_initial_model(selectable_values, recommended_model, default_model)
        st.session_state["_hugiml_selected_model"] = selected
    if selected not in selectable_values:
        # Do not drop back to deterministic routing just because the selected
        # Ollama model is currently resident in memory and has reduced free RAM
        # during the Streamlit rerun.  Keep a previously selected, installed,
        # supported model selectable for this session; new low-memory choices
        # are still blocked by the normal catalog checks above.
        can_preserve_loaded_choice = (
            selected != default_model
            and bool(status.get("ok"))
            and selected in installed
            and not is_below_minimum_llm_model(selected)
        )
        if can_preserve_loaded_choice:
            selectable_values.append(selected)
            labels[selected] = f"Current session model · {selected} · keeping selection"
            for row in rows:
                if row.get("model") == selected:
                    row["status"] = (
                        "kept for this session; memory may be temporarily held by Ollama"
                    )
                    break
        else:
            selected = default_model
            st.session_state["_hugiml_selected_model"] = default_model

    choice = st.selectbox(
        "Routing mode / Ollama model",
        selectable_values,
        index=selectable_values.index(selected),
        format_func=lambda value: labels.get(str(value), str(value)),
        key="_hugiml_model_dropdown",
        help="Installed Ollama models appear here only when they are safe for the currently available RAM.",
    )
    st.session_state["_hugiml_selected_model"] = str(choice)

    if choice == default_model:
        st.info(
            "Using deterministic routing. HUGIML actions still run normally; no Ollama call is made."
        )
    else:
        st.success(
            f"Using Ollama model `{choice}` for conversational routing. HUGIML still executes the modeling actions."
        )

    with st.expander("Model catalog and setup", expanded=False):
        _show_dataframe(rows, height=260)
        if not status.get("ok"):
            st.caption("Start Ollama first, then pull one of the enabled models.")
            st.code("ollama serve", language="bash")
        not_installed = [r["model"] for r in rows if r.get("installed") == "no"]
        if not_installed:
            st.caption("Pull a recommended model, then refresh this page:")
            st.code("\n".join(f"ollama pull {name}" for name in not_installed[:5]), language="bash")
        if any(r.get("tier") == "manual" for r in rows):
            st.caption(
                "*Manual models are not in the HUGIML recommendation catalog. qwen3:1.7b, gemma3:1b, and llama3.2:1b are allowed as explicit HUGIML tiny-model choices; other sub-3B manual models are shown but not selectable."
            )

    return str(choice)


def _preferred_initial_model(
    selectable_values: list[str], recommended_model: str, default_model: str
) -> str:
    """Pick the first-run model according to HUGIML's local model policy."""

    for candidate in (
        DEFAULT_OLLAMA_MODEL,
        recommended_model,
        LIGHT_MODE_MODEL,
        FALLBACK_OLLAMA_MODEL,
    ):
        if candidate and candidate in selectable_values:
            return candidate
    return default_model


def _dataset_picker(dataset_rows: list[dict[str, Any]]) -> str | None:
    st.checkbox(
        "Include benchmark catalog",
        value=bool(st.session_state.get("include_benchmark_catalog", False)),
        key="include_benchmark_catalog",
        help="Keep this off for a simple first-run view. Turn it on to expose the larger benchmark dataset catalog.",
    )
    if not dataset_rows:
        st.warning("No datasets found.")
        return None
    df = pd.DataFrame(dataset_rows)
    if "source" in df.columns:
        priority = {"llm_builtin": 0, "user": 1, "benchmark": 2}
        df["_source_order"] = df["source"].map(priority).fillna(9)
        df = df.sort_values(["_source_order", "name"]).drop(columns=["_source_order"])
    columns = [
        c for c in ["name", "source", "rows", "features", "target", "task_type"] if c in df.columns
    ]
    st.caption(
        "Built-in sample datasets are shown first. Enable the benchmark catalog only when needed."
    )
    st.dataframe(
        _display_dataset_frame(df[columns]),
        hide_index=True,
        width="stretch",
        height=min(230, 54 + 36 * max(1, min(len(df), 5))),
    )
    options = df["name"].tolist()
    selected = st.selectbox("Active dataset", options, index=0)
    return str(selected) if selected else None


def _cli_commands_panel() -> None:
    st.header("Commands")
    st.caption("Common ways to launch or script the optional NLP workbench.")
    for label, command in _CLI_COMMANDS[:5]:
        st.markdown(
            f"<div class='command-card'><b>{label}</b><code>{command}</code></div>",
            unsafe_allow_html=True,
        )
    with st.expander("More command-line help", expanded=False):
        st.code("hugiml-llm --help\nhugiml-llm chat --help\nhugiml-llm ask --help", language="bash")


def _cli_examples_readout() -> None:
    st.markdown("### Command-line shortcuts")
    st.markdown(
        "<p class='example-note'>The same workflows are also available outside the browser.</p>",
        unsafe_allow_html=True,
    )
    command_text = "\n".join(command for _, command in _CLI_COMMANDS)
    st.code(command_text, language="bash")


def _queue_chat_prompt(prompt: str) -> None:
    """Store exactly one pending prompt for the next Streamlit rerun."""

    st.session_state["_hugiml_pending_chat_prompt"] = prompt


def _submit_inline_chat_prompt() -> None:
    """Queue the inline composer contents before Streamlit reruns."""

    prompt = str(st.session_state.get("_hugiml_inline_chat_prompt") or "").strip()
    if prompt:
        st.session_state["_hugiml_pending_chat_prompt"] = prompt
        st.session_state["_hugiml_inline_chat_prompt"] = ""


def _consume_pending_chat_prompt() -> str | None:
    prompt = st.session_state.get("_hugiml_pending_chat_prompt")
    if prompt:
        del st.session_state["_hugiml_pending_chat_prompt"]
        return str(prompt)
    return None


def _example_question_panel(selected_dataset: str | None, *, show_buttons: bool) -> None:
    if not show_buttons:
        st.markdown("#### Chat examples")
        st.markdown(
            "<p class='example-note'>Ollama routing is active, so the fixed quick-action buttons are hidden. Type a natural-language request below, for example: <code>build a model and explain the strongest patterns</code>.</p>",
            unsafe_allow_html=True,
        )
        with st.expander("Command-line equivalents", expanded=False):
            st.code("\n".join(command for _, command in _CLI_COMMANDS), language="bash")
        return

    st.markdown("#### Quick actions")
    st.markdown(
        "<p class='example-note'>Deterministic routing is active. Click one to run only that request. Previous requests are kept as compact history, not replayed.</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for idx, (label, prompt) in enumerate(_EXAMPLE_REQUESTS):
        prompt_text = prompt
        if selected_dataset and "this dataset" in prompt_text.lower():
            prompt_text = prompt_text.replace("this dataset", selected_dataset)
        cols[idx % 2].button(
            label,
            key=f"example_prompt_{idx}",
            width="stretch",
            on_click=_queue_chat_prompt,
            args=(prompt_text,),
        )
    with st.expander("Command-line equivalents", expanded=False):
        st.code("\n".join(command for _, command in _CLI_COMMANDS), language="bash")


def _upload_panel(registry: DatasetRegistry) -> None:
    st.header("Upload data")
    st.caption("Preview columns first, then select the target column before registration.")
    upload = st.file_uploader(
        "CSV, TSV, Excel, or Parquet",
        type=["csv", "tsv", "txt", "xlsx", "xls", "parquet", "pq"],
        label_visibility="collapsed",
    )
    if upload is None:
        return

    suffix = Path(upload.name).suffix.lower()
    upload_key = f"{upload.name}:{upload.size}"
    if st.session_state.get("upload_key") != upload_key:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload.getbuffer())
            st.session_state.upload_tmp_path = tmp.name
            st.session_state.upload_key = upload_key

    tmp_path = Path(st.session_state.upload_tmp_path)
    try:
        preview = DatasetRegistry._read_table(tmp_path, nrows=80)
    except Exception as exc:
        st.error(f"Could not read upload: {exc}")
        return

    st.caption(f"Previewing {preview.shape[0]} rows × {preview.shape[1]} columns")
    st.dataframe(preview.head(8), width="stretch", height=190)
    target = st.selectbox("Target column", list(map(str, preview.columns)))
    dataset_name = st.text_input("Registered name", value=Path(upload.name).stem)
    overwrite = st.checkbox("Overwrite same-name user dataset", value=False)
    if st.button("Register dataset", type="primary", width="stretch"):
        try:
            info = registry.register_user_dataset(
                tmp_path,
                target_column=target,
                dataset_name=dataset_name,
                overwrite=overwrite,
            )
            st.success(f"Registered `{info.name}` with target `{info.target}`.")
            st.rerun()
        except Exception as exc:
            st.error(f"Registration failed: {exc}")


def _overview_panel(
    dataset_rows: list[dict[str, Any]],
    selected_dataset: str | None,
    mem: Any,
    profile: Any,
    status: dict[str, Any],
) -> None:
    st.markdown("### Workflow at a glance")
    st.markdown(
        """
        <div class="stage-list">
          <div class="stage-item"><div class="stage-num">01</div><div class="stage-text"><b>Select data</b><span>Use curated examples, benchmark datasets, or registered user uploads.</span></div></div>
          <div class="stage-item"><div class="stage-num">02</div><div class="stage-text"><b>Build or tune</b><span>Only HUGIML modeling actions are exposed in this first version.</span></div></div>
          <div class="stage-item"><div class="stage-num">03</div><div class="stage-text"><b>Score and tabulate</b><span>Generate prediction tables, metrics, and exportable summaries.</span></div></div>
          <div class="stage-item"><div class="stage-num">04</div><div class="stage-text"><b>Interpret</b><span>Summaries are grounded in HUGIML metrics, coefficients, and pattern artifacts.</span></div></div>
          <div class="stage-item"><div class="stage-num">05</div><div class="stage-text"><b>Govern</b><span>Prune, refit, and create model-card/audit artifacts through existing APIs.</span></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Readiness")
    builtin_rows = [r for r in dataset_rows if r.get("source") == "llm_builtin"]
    user_rows = [r for r in dataset_rows if r.get("source") == "user"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Built-in datasets", len(builtin_rows))
    c2.metric("Active dataset", _clip_text(selected_dataset or "none", 22))
    c3.metric("Workspace", _PROFILE_LABELS.get(profile.name, profile.name))
    c4.metric("Ollama", "ready" if status.get("ok") else "optional")

    first_run_rows = builtin_rows + user_rows
    if first_run_rows:
        st.markdown("### Built-in and user datasets")
        _render_catalog_size_bars(first_run_rows, title="First-run dataset sizes")
        _render_catalog_scatter(first_run_rows, title="First-run dataset shape map")

    st.markdown("### Suggested requests")
    cols = st.columns(4)
    for idx, text in enumerate(_ALLOWED_SUGGESTIONS):
        cols[idx % 4].markdown(f"<div class='dataset-pill'>{text}</div>", unsafe_allow_html=True)
    _cli_examples_readout()


def _data_panel(
    registry: DatasetRegistry, dataset_rows: list[dict[str, Any]], selected_dataset: str | None
) -> None:
    st.markdown("### Dataset catalog")
    if dataset_rows:
        df = pd.DataFrame(dataset_rows)
        cols = [
            c
            for c in ["name", "source", "rows", "features", "target", "description"]
            if c in df.columns
        ]
        st.caption(
            "The default view is intentionally small. Use the sidebar checkbox to include the benchmark catalog when you want the full experiment library."
        )
        st.dataframe(_display_dataset_frame(df[cols]), hide_index=True, width="stretch", height=300)
        _render_catalog_size_bars(dataset_rows, title="Visible dataset sizes")
        _render_catalog_scatter(dataset_rows, title="Visible dataset shape map")
    if selected_dataset:
        st.markdown("### Active dataset profile")
        try:
            desc = registry.describe_dataset(selected_dataset)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", _fmt_int(desc.get("rows")))
            c2.metric("Features", _fmt_int(desc.get("features")))
            c3.metric("Target", _clip_text(desc.get("target") or "unknown", 18))
            c4.metric("Task", _clip_text(desc.get("task_type") or "classification", 18))
            class_counts = desc.get("class_counts") or {}
            if class_counts:
                st.markdown("#### Class balance")
                _render_class_balance_chart(class_counts)
            missing = desc.get("missing_top") or {}
            missing_df = pd.DataFrame(
                [{"feature": k, "missing_fraction": v} for k, v in missing.items() if v]
            )
            if not missing_df.empty:
                st.markdown("#### Missingness")
                _render_missingness_chart(missing_df)
        except Exception as exc:
            st.warning(f"Could not profile active dataset: {exc}")


def _chat_panel(
    orch: HUGIMLActionOrchestrator,
    selected_dataset: str | None,
    model_choice: str,
    *,
    include_benchmarks: bool,
) -> None:
    st.markdown("### Chat")
    st.markdown(
        "<div class='qa-tip'>Ask one question, read the answer, then continue with the next question in the same thread. The input is inline, not pinned to the browser bottom.</div>",
        unsafe_allow_html=True,
    )
    deterministic_mode = model_choice == "deterministic router only"

    if "_hugiml_chat_history" not in st.session_state:
        st.session_state["_hugiml_chat_history"] = []
    if "_hugiml_response_mode" not in st.session_state:
        st.session_state["_hugiml_response_mode"] = "Fast"

    response_mode = str(st.session_state.get("_hugiml_response_mode") or "Fast")
    pending_prompt = _consume_pending_chat_prompt()
    if pending_prompt:
        with st.spinner("Running HUGIML request..."):
            result = _handle_prompt(
                orch,
                pending_prompt,
                selected_dataset,
                model_choice,
                response_mode=response_mode,
                include_benchmarks=include_benchmarks,
            )
        st.session_state["_hugiml_chat_history"].append(
            {
                "prompt": pending_prompt,
                "message": result.get("message", ""),
                "result": result,
                "action": result.get("action", ""),
                "ok": bool(result.get("ok")),
            }
        )

    history = st.session_state.get("_hugiml_chat_history") or []
    _render_qa_thread(history)
    _inline_chat_composer(selected_dataset, model_choice)

    st.markdown("<div class='bottom-tools'></div>", unsafe_allow_html=True)
    with st.expander("Quick actions and examples", expanded=False):
        _example_question_panel(selected_dataset, show_buttons=deterministic_mode)

    with st.expander("Chat controls", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Turns", len(history))
        last_action = history[-1].get("action") if history else "none"
        c2.metric("Last action", _clip_text(last_action or "none", 28))
        c3.metric("Mode", str(st.session_state.get("_hugiml_response_mode") or "Fast"))
        if st.button("Clear chat", width="stretch"):
            st.session_state["_hugiml_chat_history"] = []
            st.session_state["_hugiml_inline_chat_prompt"] = ""
            st.rerun()


def _render_qa_thread(history: list[dict[str, Any]]) -> None:
    """Render the conversation as explicit question/answer cards."""

    if not history:
        st.markdown(
            """
            <div class='qa-empty'>
              <div class='qa-label'>Assistant</div>
              <div class='qa-text'>Select a dataset, then ask me to build, tune, score, explain, prune, or create governance artifacts for a HUGIML model.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown("<div class='chat-thread'>", unsafe_allow_html=True)
    for idx, item in enumerate(history, start=1):
        prompt = html.escape(str(item.get("prompt", "")))
        st.markdown(
            f"""
            <div class='qa-card question'>
              <div class='qa-label'>Question {idx}</div>
              <div class='qa-text'>{prompt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class='qa-card answer'>
              <div class='qa-label'>Answer {idx}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not item.get("ok"):
            message = html.escape(str(item.get("message", "")))
            st.markdown(f"<div class='refusal-box'>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(str(item.get("message", "")))
        _render_result(item.get("result") or {}, suppress_summary=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _inline_chat_composer(selected_dataset: str | None, model_choice: str) -> None:
    """Render an inline follow-up composer directly below the Q&A thread."""

    active_dataset = html.escape(str(selected_dataset)) if selected_dataset else ""
    active_note = (
        f"Active dataset: {active_dataset}" if active_dataset else "No active dataset selected yet."
    )
    st.markdown(
        f"""
        <div class='chat-composer-card'>
          <div class='chat-composer-title'>Ask the next question</div>
          <div class='small-muted'>{active_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mode_options = ["Fast", "Thinking"]
    current_mode = str(st.session_state.get("_hugiml_response_mode") or "Fast")
    if current_mode not in mode_options:
        current_mode = "Fast"
    mode_index = mode_options.index(current_mode)
    selected_mode = st.radio(
        "Response mode",
        mode_options,
        index=mode_index,
        horizontal=True,
        key="_hugiml_response_mode",
        help="Fast answers direct dataset/model questions from HUGIML artifacts. Thinking also asks the selected Ollama model to rewrite grounded results.",
    )
    if selected_mode == "Fast":
        st.caption(
            "Fast mode uses direct HUGIML artifacts and skips the Ollama writer for lower latency."
        )
    else:
        if model_choice == "deterministic router only":
            st.caption(
                "Thinking mode is selected, but no Ollama model is active; responses will use deterministic routing."
            )
        else:
            st.caption(
                f"Thinking mode uses `{model_choice}` after HUGIML produces grounded results."
            )

    with st.form("hugiml_inline_chat_form", clear_on_submit=False):
        st.text_area(
            "Question",
            key="_hugiml_inline_chat_prompt",
            placeholder="Example: build a model, then explain the strongest patterns",
            height=95,
            label_visibility="collapsed",
        )
        st.form_submit_button(
            "Ask HUGIML",
            type="primary",
            width="stretch",
            on_click=_submit_inline_chat_prompt,
        )


def _handle_prompt(
    orch: HUGIMLActionOrchestrator,
    prompt: str,
    selected_dataset: str | None,
    model_choice: str,
    *,
    response_mode: str = "Fast",
    include_benchmarks: bool,
) -> dict[str, Any]:
    return run_prompt(
        orch,
        prompt,
        selected_dataset,
        model_choice,
        response_mode=response_mode,
        include_benchmarks=include_benchmarks,
        repo_root=REPO_ROOT,
)


def _session_panel(orch: HUGIMLActionOrchestrator) -> None:
    st.markdown("### Active session")
    if not orch.last_session_id:
        st.info("No model session yet. Ask the chat to build or tune a HUGIML model.")
        return
    session = orch.sessions[orch.last_session_id]
    st.markdown(
        f"<div class='panel-card'><h3>{session.dataset}</h3><p class='panel-note'>Session {session.session_id} · target <b>{session.target}</b></p></div>",
        unsafe_allow_html=True,
    )
    _render_metrics(session.metrics)
    _render_confusion_matrix(session.metrics)
    st.markdown("#### Artifacts")
    if session.artifacts:
        st.json(session.artifacts)
    else:
        st.caption("No governance artifacts generated yet.")


def _model_panel(orch: HUGIMLActionOrchestrator) -> None:
    st.markdown("### Model evidence")
    if not orch.last_session_id:
        st.info("Build or tune a model first.")
        return
    session = orch.sessions[orch.last_session_id]
    _render_metrics(session.metrics)
    redundancy_rows = downstream_redundancy_audit_rows(session.model)
    if redundancy_rows:
        st.markdown("#### Downstream redundancy audit")
        st.caption(
            "Computed from the fitted training partition; prediction uses the "
            "stored retained-column mask."
        )
        _show_dataframe(redundancy_rows, height=300)
    try:
        imp = session.model.feature_importances().head(15)
        st.markdown("#### Top patterns and features")
        _render_importance_chart(_df_records(imp))
        _show_dataframe(_df_records(imp), height=300)
    except Exception as exc:
        # feature_importances() requires a coef_-exposing downstream
        # estimator and raises for RPTE (see classifier.feature_importances'
        # docstring) -- which is now common, since "performance_ho" (the
        # default tuning grid) can select the RPTE branch. Without this
        # fallback the panel would just show the warning below and nothing
        # else, right when a user most needs the RPTE-specific explanation.
        rpte_rows: list[dict[str, Any]] = []
        try:
            rpte_rows = (
                session.model.rpte_rule_table() if hasattr(session.model, "rpte_rule_table") else []
            )
        except Exception:
            rpte_rows = []
        if rpte_rows:
            st.markdown("#### Top RPTE rules")
            st.caption(
                "The downstream estimator is RPTE; feature_importances() doesn't apply "
                "(it requires a coef_-exposing estimator), so rules are ranked by each "
                "leaf's own logistic coefficient instead."
            )
            importance_rows = rpte_rule_rows_to_importance_rows(rpte_rows)
            _render_importance_chart(importance_rows)
            _show_dataframe(importance_rows, height=300)
        else:
            st.warning(f"Feature importance unavailable: {exc}")
    try:
        pred = orch._prediction_rows(
            session, limit=12
        )  # dashboard-only readout from active session
        st.markdown("#### Held-out prediction snapshot")
        _render_prediction_chart(pred)
        _show_dataframe(pred, height=300)
    except Exception as exc:
        st.warning(f"Prediction snapshot unavailable: {exc}")


def _governance_panel(orch: HUGIMLActionOrchestrator) -> None:
    st.markdown("### Governance actions")
    st.markdown(
        """
        <div class="stage-list">
          <div class="stage-item scope-ok"><div class="stage-num">✓</div><div class="stage-text"><b>Allowed</b><span>Prune patterns, refit downstream model, generate model card, package audit artifacts.</span></div></div>
          <div class="stage-item scope-no"><div class="stage-num">×</div><div class="stage-text"><b>Not allowed here</b><span>Source edits, shell commands, scripts, package installs, or baseline-model experiments.</span></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not orch.last_session_id:
        st.info("Build or tune a model before governance actions are available.")
        return
    session = orch.sessions[orch.last_session_id]
    c1, c2, c3 = st.columns(3)
    c1.metric("Active session", session.session_id)
    c2.metric("Dataset", session.dataset)
    c3.metric("Current score", _fmt_float(session.metrics.get("primary_score")))

    st.markdown("#### Guided pruning")
    min_support = st.slider(
        "Remove patterns below support", min_value=0.0, max_value=0.20, value=0.03, step=0.01
    )
    reason = st.text_input(
        "Audit reason", value="Governance review: remove very low-support patterns before sign-off."
    )
    if st.button("Prune and refit", type="primary"):
        result = orch.execute(
            ActionRequest(action="prune_patterns", min_support=min_support, reason=reason)
        )
        st.markdown(result.message)
        _render_result(result.to_dict(), suppress_summary=True)

    st.markdown("#### Model card")
    if st.button("Generate governance report"):
        result = orch.execute(ActionRequest(action="generate_governance_report"))
        st.markdown(result.message)
        _render_result(result.to_dict(), suppress_summary=True)


def _chart_key(base: str) -> str:
    counter = int(st.session_state.get("_hugiml_chart_counter", 0)) + 1
    st.session_state["_hugiml_chart_counter"] = counter
    safe = "".join(ch if ch.isalnum() else "_" for ch in base.lower()).strip("_") or "chart"
    return f"hugiml_{safe}_{counter}"


def _plotly_chart(fig: Any, base_key: str) -> None:
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=_chart_key(base_key),
    )


def _display_dataset_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "source" in out.columns:
        out["source"] = out["source"].replace(
            {"llm_builtin": "built-in", "user": "user", "benchmark": "benchmark"}
        )
    for col in ["rows", "features"]:
        if col in out.columns:
            out[col] = out[col].map(lambda v: _fmt_int(v) if pd.notna(v) else "n/a")
    if "description" in out.columns:
        out["description"] = out["description"].map(
            lambda v: _clip_text(str(v), 92) if pd.notna(v) else ""
        )
    return out


def _clip_text(value: Any, max_chars: int = 48) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= max_chars else text[: max(0, max_chars - 1)] + "…"


def _fmt_compact(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "n/a" if value is None else str(value)
    if math.isnan(val):
        return "n/a"
    sign = "-" if val < 0 else ""
    val = abs(val)
    if val >= 1_000_000:
        return f"{sign}{val / 1_000_000:.1f}M"
    if val >= 1_000:
        return f"{sign}{val / 1_000:.1f}k"
    if float(val).is_integer():
        return f"{sign}{int(val)}"
    return f"{sign}{val:.3g}"


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _as_record_list(value: Any) -> list[dict[str, Any]]:
    """Normalize arbitrary structured output to records Streamlit can render safely."""

    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return _df_records(value)
    if isinstance(value, dict):
        return [_flatten_record(value)]
    if isinstance(value, (list, tuple)):
        records: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                records.append(_flatten_record(item))
            else:
                records.append({"value": _json_preview(item)})
        return records
    return [{"value": _json_preview(value)}]


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (dict, list, tuple)):
            out[str(key)] = _json_preview(value)
        else:
            out[str(key)] = value
    return out


def _arrow_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns that mix text with numbers/booleans so Arrow can render them.

    ``st.dataframe`` converts pandas DataFrames to Arrow internally. Arrow
    infers one type per column by scanning every value; for an ``object``
    dtype column that mixes numeric/boolean values with a string -- exactly
    what ``model_configuration``-style "parameter"/"value" tables produce,
    since ``feature_mode`` (a string) sits next to ``B``/``G``/``topK``
    (numbers) and ``adaptive_binning`` (a bool) -- it can infer "double" from
    the numeric-looking majority and then raise on the string it didn't
    expect::

        pyarrow.lib.ArrowInvalid: ("Could not convert 'original_plus_patterns'
        with type str: tried to convert to double", ...)

    This affects every action whose result includes a hyperparameter table
    (build_model, tune_hyperparameters, compare_model_configs, explain_model)
    and is independent of Thinking/Fast mode or any writer-model setting.

    Only columns that actually mix incompatible Python types are touched, so
    purely numeric or purely textual columns keep their native dtype (and
    therefore Streamlit's native numeric formatting/sorting/alignment).
    """

    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object:
            continue
        kinds: set[str] = set()
        for v in out[col]:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            if isinstance(v, bool):
                kinds.add("bool")
            elif isinstance(v, (int, float, np.integer, np.floating)):
                kinds.add("number")
            elif isinstance(v, str):
                kinds.add("str")
            else:
                kinds.add("other")
        if "str" in kinds and len(kinds) > 1:
            out[col] = out[col].map(
                lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
            )
    return out


def _safe_dataframe(value: Any) -> pd.DataFrame:
    records = _as_record_list(value)
    if not records:
        return pd.DataFrame()
    try:
        return _arrow_safe_dataframe(pd.DataFrame.from_records(records))
    except Exception:
        return pd.DataFrame({"value": [_json_preview(value)]})


def _json_preview(value: Any, max_chars: int = 240) -> str:
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        text = str(value)
    return _clip_text(text, max_chars)


def _show_dataframe(value: Any, *, height: int | None = None) -> pd.DataFrame:
    df = _safe_dataframe(value)
    if df.empty:
        return df
    display_df = df.copy()
    for col in display_df.columns:
        if display_df[col].dtype == object:
            display_df[col] = display_df[col].map(
                lambda v: _clip_text(v, 140) if isinstance(v, str) else v
            )
    try:
        st.dataframe(display_df, hide_index=True, width="stretch", height=height)
    except Exception:
        # Last-resort fallback so one malformed/unexpected column shape can
        # never crash the whole chat turn -- stringify everything and retry.
        st.dataframe(display_df.astype(str), hide_index=True, width="stretch", height=height)
    return df


def _render_result(result: dict[str, Any], *, suppress_summary: bool = False) -> None:
    if not result:
        return
    tables = result.get("tables") or {}
    data = result.get("data") or {}
    summary_text = data.get("summary")
    if summary_text and not suppress_summary:
        st.markdown("#### Findings")
        st.markdown(str(summary_text))
    model_summary = data.get("model_summary")
    if model_summary:
        with st.expander("Model summary", expanded=False):
            st.code(str(model_summary), language="text")
    if "metrics" in tables and tables["metrics"]:
        _render_metrics(tables["metrics"][0])
        _render_confusion_matrix(tables["metrics"][0])
    if "datasets" in tables and tables["datasets"]:
        df = _show_dataframe(
            _display_dataset_frame(_safe_dataframe(tables["datasets"])), height=260
        )
        if "source" in df.columns:
            counts = df.groupby("source").size().reset_index(name="count")
            _render_source_chart(counts)
    if "class_balance" in tables and tables["class_balance"]:
        st.markdown("#### Class balance")
        _render_class_balance_chart(
            {row.get("class"): row.get("count") for row in tables["class_balance"]}
        )
    if "feature_importance" in tables and tables["feature_importance"]:
        st.markdown("#### Pattern / feature influence")
        _render_importance_chart(tables["feature_importance"])
    if "predictions" in tables and tables["predictions"]:
        st.markdown("#### Prediction output")
        _render_prediction_chart(tables["predictions"])
    if "pruning_summary" in tables and tables["pruning_summary"]:
        st.markdown("#### Pruning summary")
        _show_dataframe(tables["pruning_summary"], height=160)
    for name, rows in tables.items():
        if name in {
            "metrics",
            "datasets",
            "class_balance",
            "feature_importance",
            "predictions",
            "pruning_summary",
        }:
            continue
        if rows:
            with st.expander(f"Table: {name}", expanded=False):
                _show_dataframe(rows, height=260)
    artifacts = result.get("artifacts") or {}
    if artifacts:
        with st.expander("Artifacts", expanded=True):
            st.json(artifacts)
    data = result.get("data") or {}
    if result.get("action") == "answer_api_question":
        return
    slim = {k: v for k, v in data.items() if k not in {"model_card", "model_summary"}}
    if slim:
        with st.expander("Structured data", expanded=False):
            st.code(json.dumps(slim, indent=2, default=str), language="json")


def _render_metrics(metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    cols = st.columns(5)
    display = [
        ("Primary", metrics.get("primary_score")),
        ("ROC AUC", metrics.get("roc_auc")),
        ("Accuracy", metrics.get("accuracy")),
        ("Balanced acc.", metrics.get("balanced_accuracy")),
        ("Precision", metrics.get("precision")),
        ("Recall", metrics.get("recall")),
        ("F1", metrics.get("f1")),
        ("Avg precision", metrics.get("average_precision")),
    ]
    for col, (label, value) in zip(cols, display):
        col.metric(label, _fmt_float(value))
    chart_rows = [
        {"metric": label, "value": float(value)}
        for label, value in display[1:]
        if _is_number(value) and not math.isnan(float(value))
    ]
    if chart_rows:
        df = pd.DataFrame(chart_rows)
        if px is None:
            st.bar_chart(df.set_index("metric"), width="stretch")
            return
        fig = px.bar(
            df,
            x="metric",
            y="value",
            text=df["value"].map(lambda v: f"{v:.3f}"),
            title="Held-out metric profile",
            range_y=[0, max(1.0, float(df["value"].max()) * 1.08)],
        )
        fig.update_traces(
            marker_color=_PALETTE[0],
            texttemplate="%{text}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        )
        _polish_fig(fig, y_title="Score", height=330)
        _plotly_chart(fig, "metrics")


def _render_confusion_matrix(metrics: dict[str, Any]) -> None:
    cm = _coerce_matrix(metrics.get("confusion_matrix"))
    if not cm:
        return
    st.markdown("#### Confusion matrix")
    rows = []
    for i, row in enumerate(cm):
        record = {"actual": f"class {i}"}
        for j, value in enumerate(row):
            record[f"pred {j}"] = value
        rows.append(record)
    if go is not None:
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[f"Pred {i}" for i in range(len(cm[0]))],
                y=[f"Actual {i}" for i in range(len(cm))],
                text=cm,
                texttemplate="%{text}",
                colorscale=[[0, "#F8FAF3"], [1, "#2C6E63"]],
                showscale=False,
                hovertemplate="%{y} · %{x}: %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            height=280,
            margin=dict(l=35, r=20, t=25, b=30),
            plot_bgcolor="#F8FAF3",
            paper_bgcolor="#F8FAF3",
            font=dict(color="#1C2420"),
        )
        fig.update_xaxes(side="bottom", automargin=True)
        fig.update_yaxes(automargin=True)
        _plotly_chart(fig, "confusion_matrix")
    else:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_importance_chart(rows: list[dict[str, Any]]) -> None:
    df = _safe_dataframe(rows)
    if df.empty:
        return
    label_col = next(
        (c for c in ["display_name", "pattern", "feature"] if c in df.columns), df.columns[0]
    )
    value_col = (
        "coefficient"
        if "coefficient" in df.columns
        else "abs_coefficient"
        if "abs_coefficient" in df.columns
        else None
    )
    if value_col is None:
        _show_dataframe(_df_records(df), height=300)
        return
    view = df[[label_col, value_col]].copy()
    view["full_label"] = view[label_col].astype(str)
    view["label"] = view["full_label"].map(lambda x: _clip_text(x, 42))
    view[value_col] = _coerce_numeric_series(view[value_col])
    view = view.dropna(subset=[value_col]).head(15)
    if view.empty:
        return
    signed = value_col == "coefficient" and (view[value_col] < 0).any()
    plot = view.sort_values(value_col, ascending=True).copy()
    plot["value_label"] = plot[value_col].map(lambda v: f"{v:.3f}")
    if px is None:
        st.bar_chart(plot.set_index("label")[[value_col]], width="stretch")
    else:
        height = max(360, 120 + 30 * len(plot))
        if signed:
            plot["direction"] = plot[value_col].map(
                lambda v: "toward positive class" if v >= 0 else "toward negative class"
            )
            fig = px.bar(
                plot,
                x=value_col,
                y="label",
                orientation="h",
                color="direction",
                text="value_label",
                color_discrete_map=_COLOR_MAP,
                hover_data={
                    "full_label": True,
                    "label": False,
                    value_col: ":.4f",
                    "value_label": False,
                },
                title="Feature and pattern influence",
            )
            fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="#5B6358")
            min_v = float(plot[value_col].min())
            max_v = float(plot[value_col].max())
            pad = max(0.2, (max_v - min_v) * 0.18)
            fig.update_xaxes(range=[min_v - pad, max_v + pad])
            _polish_fig(
                fig,
                x_title="Signed coefficient",
                y_title="",
                height=height,
                margin_l=210,
                margin_r=70,
            )
        else:
            fig = px.bar(
                plot,
                x=value_col,
                y="label",
                orientation="h",
                text="value_label",
                hover_data={
                    "full_label": True,
                    "label": False,
                    value_col: ":.4f",
                    "value_label": False,
                },
                title="Feature and pattern importance",
            )
            fig.update_traces(marker_color=_PALETTE[0])
            max_v = float(plot[value_col].max())
            fig.update_xaxes(range=[0, max_v * 1.22 if max_v > 0 else 1])
            _polish_fig(
                fig, x_title="Importance", y_title="", height=height, margin_l=210, margin_r=70
            )
        fig.update_traces(textposition="outside", cliponaxis=False)
        _plotly_chart(fig, "importance")
    _show_dataframe(_df_records(df), height=340)


def _render_prediction_chart(rows: list[dict[str, Any]]) -> None:
    df = _safe_dataframe(rows)
    if df.empty:
        return
    prob_cols = [c for c in df.columns if "prob" in c.lower() or c.lower().startswith("p(")]
    if prob_cols:
        col = prob_cols[-1]
        view = df[[col]].copy()
        view["row"] = [f"row {i}" for i in range(len(view))]
        view[col] = pd.to_numeric(view[col], errors="coerce")
        if px is None:
            st.bar_chart(view.set_index("row")[[col]], width="stretch")
        else:
            fig = px.bar(
                view,
                x="row",
                y=col,
                text=view[col].map(lambda v: "" if pd.isna(v) else f"{v:.2f}"),
                title="Prediction probability snapshot",
                range_y=[0, 1],
            )
            fig.update_traces(
                marker_color=_PALETTE[1],
                texttemplate="%{text}",
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            )
            _polish_fig(fig, y_title="Probability", height=330)
            fig.update_yaxes(range=[0, 1.08])
            _plotly_chart(fig, "prediction")
    _show_dataframe(_df_records(df), height=300)


def _render_source_chart(src_counts: pd.DataFrame) -> None:
    if src_counts.empty:
        return
    if px is None:
        st.bar_chart(src_counts.set_index("source"), width="stretch")
        return
    plot = src_counts.sort_values("count").copy()
    plot["source"] = plot["source"].replace(
        {"llm_builtin": "built-in", "user": "user", "benchmark": "benchmark"}
    )
    fig = px.bar(
        plot,
        x="count",
        y="source",
        orientation="h",
        text="count",
        title="Dataset catalog by source",
    )
    fig.update_traces(
        marker_color=_PALETTE[0],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x}<extra></extra>",
    )
    max_count = max(1, int(plot["count"].max()))
    fig.update_xaxes(range=[0, max_count * 1.35])
    _polish_fig(fig, x_title="Datasets", y_title="", height=280, margin_l=110, margin_r=45)
    _plotly_chart(fig, "source_catalog")


def _render_catalog_size_bars(
    dataset_rows: list[dict[str, Any]], title: str = "Dataset sizes"
) -> None:
    if not dataset_rows or px is None:
        return
    df = pd.DataFrame(dataset_rows)
    if not {"name", "rows"}.issubset(df.columns):
        return
    df = df.dropna(subset=["rows"]).copy()
    df["rows"] = _coerce_numeric_series(df["rows"])
    df = df.dropna(subset=["rows"]).sort_values("rows", ascending=True).tail(12)
    if df.empty:
        return
    df["label"] = df["name"].map(lambda v: _clip_text(v, 32))
    df["rows_label"] = df["rows"].map(_fmt_compact)
    color = "source" if "source" in df.columns else None
    fig = px.bar(
        df,
        x="rows",
        y="label",
        orientation="h",
        color=color,
        text="rows_label",
        title=title,
        hover_data={"name": True, "rows": ":,.0f", "rows_label": False, "label": False},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    max_rows = float(df["rows"].max())
    fig.update_xaxes(range=[0, max_rows * 1.30 if max_rows > 0 else 1])
    _polish_fig(
        fig,
        x_title="Rows",
        y_title="",
        height=max(300, 110 + 30 * len(df)),
        margin_l=175,
        margin_r=65,
    )
    _plotly_chart(fig, "catalog_size")


def _render_catalog_scatter(
    dataset_rows: list[dict[str, Any]], title: str = "Dataset shape map"
) -> None:
    if not dataset_rows or px is None:
        return
    df = pd.DataFrame(dataset_rows)
    if not {"rows", "features", "source", "name"}.issubset(df.columns):
        return
    df = df.dropna(subset=["rows", "features"]).copy()
    df["rows"] = _coerce_numeric_series(df["rows"])
    df["features"] = _coerce_numeric_series(df["features"])
    df = df.dropna(subset=["rows", "features"])
    if df.empty:
        return
    df["rows_label"] = df["rows"].map(_fmt_int)
    df["features_label"] = df["features"].map(_fmt_int)
    df["source"] = df["source"].replace(
        {"llm_builtin": "built-in", "user": "user", "benchmark": "benchmark"}
    )
    st.markdown(f"### {title}")
    fig = px.scatter(
        df,
        x="features",
        y="rows",
        color="source",
        hover_name="name",
        log_y=True,
        title="Rows versus features",
        custom_data=["rows_label", "features_label", "source"],
    )
    fig.update_traces(
        marker=dict(size=13, line=dict(width=1, color="#1C2420")),
        hovertemplate="%{hovertext}<br>Rows: %{customdata[0]}<br>Features: %{customdata[1]}<br>Source: %{customdata[2]}<extra></extra>",
    )
    _polish_fig(
        fig, x_title="Features", y_title="Rows (log scale)", height=330, margin_l=55, margin_r=20
    )
    _plotly_chart(fig, "catalog_scatter")


def _render_class_balance_chart(class_counts: dict[Any, Any]) -> None:
    rows = []
    for k, v in class_counts.items():
        if v is None:
            continue
        try:
            count = int(v)
        except Exception:
            continue
        rows.append({"class": str(k), "count": count})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    if px is None:
        st.bar_chart(df.set_index("class"), width="stretch")
        return
    col1, col2 = st.columns([0.48, 0.52])
    with col1:
        fig = px.pie(df, names="class", values="count", hole=0.52, title="Class mix")
        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate="%{label}: %{value:,}<extra></extra>",
        )
        _polish_fig(fig, height=300)
        _plotly_chart(fig, "class_mix")
    with col2:
        df["count_label"] = df["count"].map(_fmt_compact)
        fig = px.bar(df, x="class", y="count", text="count_label", title="Class counts")
        fig.update_traces(
            marker_color=_PALETTE[0],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        )
        max_count = max(1, int(df["count"].max()))
        fig.update_yaxes(range=[0, max_count * 1.22])
        _polish_fig(fig, y_title="Rows", height=300)
        _plotly_chart(fig, "class_counts")


def _render_missingness_chart(missing_df: pd.DataFrame) -> None:
    if px is None:
        st.bar_chart(missing_df.set_index("feature"))
        return
    view = missing_df.sort_values("missing_fraction", ascending=True)
    fig = px.bar(
        view,
        x="missing_fraction",
        y="feature",
        orientation="h",
        title="Top missing-value fractions",
    )
    fig.update_traces(marker_color=_PALETTE[3], hovertemplate="%{y}: %{x:.2%}<extra></extra>")
    _polish_fig(
        fig,
        x_title="Missing fraction",
        y_title="",
        height=max(280, 100 + 28 * len(view)),
        margin_l=190,
    )
    _plotly_chart(fig, "missingness")


def _polish_fig(
    fig: Any,
    x_title: str | None = None,
    y_title: str | None = None,
    *,
    height: int = 360,
    margin_l: int = 40,
    margin_r: int = 35,
) -> None:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=margin_l, r=margin_r, t=58, b=42),
        paper_bgcolor="#F8FAF3",
        plot_bgcolor="#F8FAF3",
        font=dict(family="Arial, sans-serif", color="#1C2420", size=12),
        title=dict(font=dict(size=16), x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uniformtext_minsize=10,
        uniformtext_mode="show",
    )
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(showgrid=True, gridcolor="#E1E6DC", zeroline=False, automargin=True)
    fig.update_yaxes(showgrid=False, automargin=True, tickfont=dict(size=11))


def _fmt_gb(value: Any) -> str:
    return "unknown" if value is None else f"{float(value):.1f} GB"


def _fmt_float(value: Any) -> str:
    try:
        val = float(value)
        if math.isnan(val):
            return "n/a"
        return f"{val:.4f}"
    except Exception:
        return "n/a" if value is None else str(value)


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "n/a" if value is None else str(value)


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _coerce_matrix(value: Any) -> list[list[Any]] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            return None
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(row, (list, tuple)) for row in value)
    ):
        return [list(row) for row in value]
    return None


def _df_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


if __name__ == "__main__":
    main()

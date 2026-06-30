"""Command-line interface for the optional HUGIML NLP add-on.

After installation, the intended entry point is::

    hugiml-llm

which launches the Streamlit UI.  Subcommands provide status, dataset listing,
one-shot requests, terminal chat, and demo HTML generation.  Optional UI/LLM
imports are lazy so importing the base package remains lightweight.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from .orchestrator import HUGIMLActionOrchestrator, generate_qna_html
from .planner import plan_request
from .runtime import (
    check_ollama,
    get_memory_info,
    get_model_catalog,
    model_availability,
    recommend_profile,
)
from .schemas import ActionRequest, ActionResult

_OPTIONAL_INSTALL_HINT = """The HUGIML NLP assistant is optional.

Install it with:
  pip install "hugiml-core[llm]"

Then run:
  hugiml-llm
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hugiml-llm")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Source checkout root. Usually unnecessary for installed packages.",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_ui = sub.add_parser("ui", help="Launch the Streamlit natural-language workbench.")
    p_ui.add_argument("--port", type=int, default=None)
    p_ui.add_argument("--headless", action="store_true", help="Start Streamlit in headless mode.")

    sub.add_parser("status", help="Show memory/profile/Ollama status.")

    p_list = sub.add_parser("list-datasets", help="List available datasets.")
    p_list.add_argument("--repo-root", default=None, help="Source checkout root for local package development.")

    p_chat = sub.add_parser("chat", help="Run a small terminal chat loop.")
    p_chat.add_argument("--repo-root", default=None, help="Source checkout root for local package development.")
    p_chat.add_argument("--dataset", default=None)
    p_chat.add_argument("--target", default=None)
    p_chat.add_argument("--model", default=None, help="Ollama model name override.")
    p_chat.add_argument("--no-llm", action="store_true", help="Use deterministic routing only.")

    p_ask = sub.add_parser("ask", help="Plan and execute one natural-language request.")
    p_ask.add_argument("question")
    p_ask.add_argument("--repo-root", default=None, help="Source checkout root for local package development.")
    p_ask.add_argument("--dataset", default=None)
    p_ask.add_argument("--target", default=None)
    p_ask.add_argument("--model", default=None, help="Ollama model name override.")
    p_ask.add_argument("--no-llm", action="store_true", help="Use deterministic routing only.")

    p_demo = sub.add_parser("demo-html", help="Generate an end-to-end Q&A governance HTML demo.")
    p_demo.add_argument("--repo-root", default=None, help="Source checkout root for local package development.")
    p_demo.add_argument("--dataset", default="churn_synthetic")
    p_demo.add_argument("--output", default="LLM/examples/governance_qna_churn.html")

    args = parser.parse_args(argv)
    cmd = args.cmd or "ui"

    if cmd == "ui":
        return _cmd_ui(port=getattr(args, "port", None), headless=getattr(args, "headless", False))
    if cmd == "status":
        return _cmd_status(repo_root=args.repo_root)
    if cmd == "list-datasets":
        orch = HUGIMLActionOrchestrator(repo_root=args.repo_root)
        return _print_result(orch.execute(ActionRequest(action="list_datasets")))
    if cmd == "ask":
        return _cmd_ask(
            args.question,
            repo_root=args.repo_root,
            dataset=args.dataset,
            target=args.target,
            model=args.model,
            no_llm=args.no_llm,
        )
    if cmd == "chat":
        return _cmd_chat(
            repo_root=args.repo_root,
            dataset=args.dataset,
            target=args.target,
            model=args.model,
            no_llm=args.no_llm,
        )
    if cmd == "demo-html":
        return _cmd_demo_html(args.dataset, args.output, args.repo_root)
    return 2


def _cmd_ui(*, port: int | None = None, headless: bool = False) -> int:
    if importlib.util.find_spec("streamlit") is None:
        print(_OPTIONAL_INSTALL_HINT, file=sys.stderr)
        return 1
    app_path = Path(__file__).with_name("ui_app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if port is not None:
        cmd.extend(["--server.port", str(port)])
    if headless:
        cmd.extend(["--server.headless", "true"])
    return subprocess.call(cmd)


def _cmd_status(*, repo_root: str | None = None) -> int:
    mem = get_memory_info()
    profile = recommend_profile(mem, repo_root=repo_root)
    status = check_ollama()
    pulled = status.get("models") or []
    catalog = []
    for option in get_model_catalog(repo_root):
        avail = model_availability(
            option,
            mem,
            pulled,
            ollama_ok=bool(status.get("ok")),
        )
        catalog.append(
            {
                "model": option.model,
                "label": option.label,
                "profile": option.profile,
                "min_available_gb": option.min_available_gb,
                "installed": avail["installed"],
                "selectable": avail["selectable"],
                "status": avail["reason"],
            }
        )
    payload = {
        "memory": {
            "total_gb": mem.total_gb,
            "available_gb": mem.available_gb,
            "source": mem.source,
        },
        "recommended_workspace": {
            "name": profile.name,
            "recommended_model": profile.recommended_model,
            "max_context_tokens": profile.max_context_tokens,
            "description": profile.description,
        },
        "model_catalog": catalog,
        "ollama": status,
        "note": "Ollama models are not auto-pulled. Pull explicitly with the Ollama CLI.",
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _cmd_ask(
    question: str,
    *,
    repo_root: str | None,
    dataset: str | None,
    target: str | None,
    model: str | None,
    no_llm: bool,
) -> int:
    orch = HUGIMLActionOrchestrator(repo_root=repo_root)
    planned = plan_request(
        question,
        model=model,
        prefer_llm=not no_llm,
        context={"dataset": dataset, "target": target},
        repo_root=repo_root,
    )
    if isinstance(planned, ActionResult):
        return _print_result(planned)
    _apply_dataset_context(planned, dataset, target)
    return _print_result(orch.execute(planned))


def _cmd_chat(
    *,
    repo_root: str | None,
    dataset: str | None,
    target: str | None,
    model: str | None,
    no_llm: bool,
) -> int:
    orch = HUGIMLActionOrchestrator(repo_root=repo_root)
    print("HUGIML NLP chat. Type 'exit' or 'quit' to stop.")
    if dataset:
        print(f"Active dataset: {dataset}")
    while True:
        try:
            question = input("hugiml> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            return 0
        planned = plan_request(
            question,
            model=model,
            prefer_llm=not no_llm,
            context={"dataset": dataset, "target": target, "last_session": orch.last_session_id},
            repo_root=repo_root,
        )
        if isinstance(planned, ActionResult):
            _print_result(planned)
            continue
        _apply_dataset_context(planned, dataset, target)
        _print_result(orch.execute(planned))


def _apply_dataset_context(planned: ActionRequest, dataset: str | None, target: str | None) -> None:
    if dataset and planned.dataset is None and planned.action in {
        "describe_dataset",
        "build_model",
        "tune_hyperparameters",
        "generate_predictions",
        "generate_tabular_output",
        "explain_model",
        "explain_prediction",
    }:
        planned.dataset = dataset
    if target and planned.target is None:
        planned.target = target


def _cmd_demo_html(dataset: str, output: str, repo_root: str | None) -> int:
    orch = HUGIMLActionOrchestrator(repo_root=repo_root)
    desc = orch.registry.describe_dataset(dataset)
    build = orch.execute(
        ActionRequest(action="tune_hyperparameters", dataset=dataset, strategy="fast", metric="roc_auc")
    )
    predictions = orch.execute(ActionRequest(action="generate_predictions", dataset=dataset, limit=8))
    explain = orch.execute(ActionRequest(action="explain_model", limit=12))
    prune = orch.execute(
        ActionRequest(
            action="prune_patterns",
            min_support=0.03,
            reason="Governance demo: remove very low-support patterns before sign-off.",
        )
    )
    gov = orch.execute(ActionRequest(action="generate_governance_report"))
    path = generate_qna_html(
        Path(output),
        title="HUGIML NLP Q&A: Churn Risk Build, Interpretation, Pruning, and Governance",
        dataset_description=desc,
        build_result=build,
        prediction_result=predictions,
        explain_result=explain,
        prune_result=prune,
        governance_result=gov,
    )
    print(path)
    return 0


def _print_result(result: ActionResult) -> int:
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

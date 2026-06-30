"""Deterministic executor for validated HUGIML NLP actions.

The orchestrator is the safety boundary.  LLMs may propose ``ActionRequest``
JSON, but only this module loads data, fits models, tunes hyperparameters,
produces tables, prunes patterns, and writes governance artifacts.
"""

from __future__ import annotations

import html
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split

from hugiml import HUGIMLClassifier
from hugiml.governance import generate_model_card, package_audit_artifacts
from hugiml.hyperparameter_configs import get_hugiml_grid
from hugiml.pruning import PatternEditor

from .dataset_registry import DatasetRegistry
from .docs_index import DocumentationSearchIndex, build_docs_answer, build_docs_context
from .schemas import ActionRequest, ActionResult, refusal_result


@dataclass
class ModelSession:
    session_id: str
    dataset: str
    target: str
    info: dict[str, Any]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    model: Any
    metrics: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    artifacts: dict[str, str] = field(default_factory=dict)


class HUGIMLActionOrchestrator:
    """Execute validated natural-language actions using existing HUGIML APIs."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        session_dir: str | Path | None = None,
        random_state: int = 42,
    ) -> None:
        self.registry = DatasetRegistry(repo_root=repo_root)
        self.repo_root = self.registry.repo_root or Path.cwd().resolve()
        self.session_dir = Path(session_dir or (self.repo_root / "LLM" / "sessions")).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = int(random_state)
        self.docs_index = DocumentationSearchIndex(
            repo_root=self.repo_root,
            cache_dir=self.session_dir / "_cache",
        )
        self.sessions: dict[str, ModelSession] = {}
        self.last_session_id: str | None = None
        self._dataset_answer_cache: dict[tuple[str, int], tuple[dict[str, Any], dict[str, list[dict[str, Any]]], str]] = {}

    def execute(self, request: ActionRequest | dict[str, Any]) -> ActionResult:
        if isinstance(request, dict):
            request = ActionRequest.from_dict(request)
        request = self._fill_missing_context(request)
        try:
            request.validate()
        except ValueError as exc:
            return self._validation_failure(request, exc)
        if request.action == "refuse":
            return refusal_result(request.reason or "unsupported_action")
        try:
            handler = getattr(self, f"_action_{request.action}", None)
            if handler is None:
                return refusal_result("unsupported_action", action=request.action)
            return handler(request)
        except Exception as exc:
            return ActionResult(
                ok=False,
                action=request.action,
                message=f"Action {request.action!r} failed: {exc}",
                data={"error_type": type(exc).__name__},
            )

    def _fill_missing_context(self, request: ActionRequest) -> ActionRequest:
        """Best-effort recovery for requests that omit a dataset.

        Deterministic routing (used when no LLM is available) cannot reliably
        extract a dataset name from free text. Rather than fail outright, reuse
        the active session's dataset/target when the request doesn't supply its
        own -- this is exactly what a person means by "now tune it" after
        already building on a dataset earlier in the same conversation.
        """

        needs_dataset = request.action in {
            "describe_dataset",
            "build_model",
            "tune_hyperparameters",
            "generate_predictions",
            "compare_model_configs",
        }
        if needs_dataset and not request.dataset:
            sid = request.session_id or self.last_session_id
            session = self.sessions.get(sid) if sid else None
            if session is not None:
                request.dataset = session.dataset
                if not request.target:
                    request.target = session.target
        return request

    @staticmethod
    def _validation_failure(request: ActionRequest, exc: ValueError) -> ActionResult:
        message = str(exc)
        if "requires a dataset" in message:
            return refusal_result("dataset_required", action=request.action)
        return ActionResult(
            ok=False,
            action=request.action,
            message=f"Could not run {request.action!r}: {message}",
            data={"error_type": "ValueError"},
        )

    def _action_list_datasets(self, request: ActionRequest) -> ActionResult:
        infos = self.registry.list_datasets(include_profiles=True)
        rows = [info.to_dict() for info in infos]
        return ActionResult(
            ok=True,
            action=request.action,
            message=f"Found {len(rows)} datasets across curated, user, and benchmark sources.",
            tables={"datasets": rows},
            data={"count": len(rows)},
        )

    def _action_describe_dataset(self, request: ActionRequest) -> ActionResult:
        cache_key = (request.dataset or "", max(int(request.limit or 10), 12))
        cached = self._dataset_answer_cache.get(cache_key)
        if cached is None:
            desc = self.registry.describe_dataset(request.dataset or "")
            tables: dict[str, list[dict[str, Any]]] = {
                "class_balance": [
                    {"class": key, "count": value} for key, value in desc["class_counts"].items()
                ],
                "missing_top": [
                    {"feature": key, "missing_fraction": value}
                    for key, value in desc["missing_top"].items()
                ],
            }
            try:
                X, _y, _info = self.registry.load_dataset(request.dataset or "")
                tables["predictor_overview"] = _predictor_overview_rows(X)
                tables["summary_statistics"] = _summary_statistics_rows(X, limit=max(request.limit, 12))
            except Exception as exc:
                desc["summary_statistics_error"] = str(exc)
            message = _dataset_decision_summary(desc, tables)
            self._dataset_answer_cache[cache_key] = (desc, tables, message)
        else:
            desc, tables, message = cached
            desc = dict(desc)
            tables = {key: [dict(row) for row in rows] for key, rows in tables.items()}
        writer_model = str(request.params.get("_writer_model") or "").strip()
        if writer_model:
            polished = _polish_run_answer(
                writer_model=writer_model,
                question=request.question or "Describe this dataset.",
                draft=message,
                run_context=json.dumps({"dataset": desc, "tables": tables}, indent=2, default=str),
            )
            if polished:
                message = polished
        return ActionResult(
            ok=True,
            action=request.action,
            message=message,
            data=desc,
            tables=tables,
        )

    def _action_build_model(self, request: ActionRequest) -> ActionResult:
        session = self._fit_session(request, tune=False)
        return self._session_result(session, request.action, "Built HUGIML model.", request=request)

    def _action_tune_hyperparameters(self, request: ActionRequest) -> ActionResult:
        session = self._fit_session(request, tune=True)
        return self._session_result(session, request.action, "Tuned and selected HUGIML model.", request=request)

    def _action_compare_model_configs(self, request: ActionRequest) -> ActionResult:
        X, y, info = self.registry.load_dataset(request.dataset or "")
        target = request.target or info.target or "target"
        X, y, run_scope = self._limit_working_dataset(X, np.asarray(y), request)
        counts = pd.Series(y).value_counts(dropna=False)
        stratify = y if len(counts) > 1 and int(counts.min()) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(request.params.get("test_size", 0.25)),
            random_state=self.random_state, stratify=stratify,
        )
        base_params = self._params_for_strategy(request.strategy)
        configs = request.params.get("model_params_list") or []
        if not configs:
            configs = [
                {"feature_mode": "patterns_only", "label": "patterns_only"},
                {"feature_mode": "original_plus_patterns", "label": "original_plus_patterns"},
            ]
        rows: list[dict[str, Any]] = []
        sessions: list[ModelSession] = []
        for i, config in enumerate(configs, start=1):
            params = dict(base_params)
            params.update({k: v for k, v in dict(config).items() if k != "label"})
            model = HUGIMLClassifier(**params)
            model.fit(X_train, y_train)
            metrics = self._evaluate(model, X_test, y_test, request.metric)
            metrics.update({
                "n_train": int(len(y_train)),
                "n_rows_used": int(run_scope.get("rows_used", len(y))),
                "source_rows": int(run_scope.get("source_rows", len(y))),
                "sampled_for_chat": bool(run_scope.get("sampled", False)),
            })
            label = str(config.get("label") or params.get("feature_mode") or f"config_{i}")
            row = {"configuration": label, **_compact_param_row(params), **_compact_metric_row(metrics)}
            try:
                composition = model.get_model_composition()
                row.update({
                    "downstream_features": composition.get("n_downstream_features"),
                    "patterns_mined": composition.get("n_patterns_mined"),
                    "original_features": (composition.get("downstream_feature_counts") or {}).get("original"),
                    "pattern_features": (composition.get("downstream_feature_counts") or {}).get("pattern"),
                    "augmented_pair_features": (composition.get("downstream_feature_counts") or {}).get("augmented_pair"),
                })
            except Exception:
                pass
            rows.append(row)
            sid = self._new_session_id(f"{info.name}-{label}")
            sessions.append(ModelSession(
                session_id=sid, dataset=info.name, target=target, info=info.to_dict(),
                X_train=X_train, X_test=X_test, y_train=np.asarray(y_train), y_test=np.asarray(y_test),
                model=model, metrics={**metrics, "chosen_params": params, "run_label": label},
            ))
        primary = request.metric or "roc_auc"
        best_idx = max(range(len(rows)), key=lambda idx: _score_for_selection(sessions[idx].metrics, primary))
        session = sessions[best_idx]
        self.sessions[session.session_id] = session
        self.last_session_id = session.session_id
        self._write_session_manifest(session)
        message = _comparison_decision_summary(info.name, rows, primary, rows[best_idx].get("configuration"))
        writer_model = str(request.params.get("_writer_model") or "").strip()
        if writer_model:
            message = _polish_run_answer(
                writer_model=writer_model,
                question=request.question or "Compare these HUGIML model configurations.",
                draft=message,
                run_context=json.dumps({"comparison": rows, "selected_session": session.session_id}, indent=2, default=str),
            ) or message
        return ActionResult(
            ok=True,
            action=request.action,
            message=message,
            data={"session_id": session.session_id, "selected_configuration": rows[best_idx].get("configuration"), "dataset": info.name, "run_scope": run_scope},
            tables={"configuration_comparison": rows, "metrics": [session.metrics], "model_configuration": _model_config_rows(session)},
        )

    def _action_generate_predictions(self, request: ActionRequest) -> ActionResult:
        session = self._session_for(request)
        pred_rows = self._prediction_rows(session, limit=request.limit)
        return ActionResult(
            ok=True,
            action=request.action,
            message=f"Generated prediction table for session {session.session_id}.",
            tables={"predictions": pred_rows},
            data={"session_id": session.session_id},
        )

    def _action_generate_tabular_output(self, request: ActionRequest) -> ActionResult:
        session = self._session_for(request)
        tables = {
            "metrics": [session.metrics],
            "predictions": self._prediction_rows(session, limit=request.limit),
        }
        try:
            tables["feature_importance"] = _df_records(
                session.model.feature_importances().head(request.limit)
            )
        except Exception:
            pass
        return ActionResult(
            ok=True,
            action=request.action,
            message=f"Generated tabular outputs for session {session.session_id}.",
            tables=tables,
            data={"session_id": session.session_id},
        )

    def _action_explain_model(self, request: ActionRequest) -> ActionResult:
        session = self._session_for(request)
        tables: dict[str, list[dict[str, Any]]] = {"metrics": [session.metrics]}
        data: dict[str, Any] = {"session_id": session.session_id, "summary": ""}
        try:
            imp = session.model.feature_importances().head(request.limit)
            tables["feature_importance"] = _df_records(imp)
        except Exception as exc:
            data["feature_importance_error"] = str(exc)
        try:
            pat = session.model.get_pattern_info().head(request.limit)
            tables["pattern_info"] = _df_records(pat)
        except Exception as exc:
            data["pattern_info_error"] = str(exc)
        tables["model_configuration"] = _model_config_rows(session)
        try:
            composition = session.model.get_model_composition()
            data["model_composition"] = composition
            tables["model_composition"] = _model_composition_rows(composition)
        except Exception as exc:
            data["model_composition_error"] = str(exc)
        rep_tables = _representation_breakdown_tables(session, request.limit)
        tables.update(rep_tables)
        try:
            data["model_summary"] = session.model.model_summary()
        except Exception:
            data["model_summary"] = _grounded_summary(session.metrics, tables)

        summary = _decision_summary(session, tables, question=request.question or "")
        writer_model = str(request.params.get("_writer_model") or "").strip()
        if writer_model:
            summary = _polish_run_answer(
                writer_model=writer_model,
                question=request.question or "Summarize the active HUGIML model findings.",
                draft=summary,
                run_context=_run_context_for_writer(session, tables, data),
            ) or summary
        data["summary"] = summary
        return ActionResult(
            ok=True,
            action=request.action,
            message=summary,
            data=data,
            tables=tables,
        )

    def _action_explain_prediction(self, request: ActionRequest) -> ActionResult:
        # v1 returns predictions plus top global patterns as grounded explanation.
        session = self._session_for(request)
        result = self._action_generate_predictions(request)
        try:
            result.tables["feature_importance"] = _df_records(
                session.model.feature_importances().head(request.limit)
            )
            result.message += " Per-row local explanations can be layered on later; v1 returns prediction rows and global pattern evidence."
        except Exception:
            pass
        return result

    def _action_prune_patterns(self, request: ActionRequest) -> ActionResult:
        session = self._session_for(request)
        before = int(len(getattr(session.model, "patterns_", [])))
        previous_metrics = dict(session.metrics or {})
        editor = PatternEditor(session.model, operator_name="hugiml-llm")
        reason = request.reason or "NLP-governed pruning request"
        if request.pattern_indices:
            editor.remove(request.pattern_indices, reason=reason)
        elif request.keyword:
            editor.remove_by_keyword(request.keyword, reason=reason)
        elif request.min_support is not None:
            editor.remove_low_support(float(request.min_support), reason=reason)
        editor.refit(session.X_train, session.y_train)
        pruned_model = editor.finalize()
        after = int(len(getattr(pruned_model, "patterns_", [])))
        metrics = self._evaluate(pruned_model, session.X_test, session.y_test, request.metric)
        session.model = pruned_model
        session.metrics = metrics
        tables = {
            "pruning_summary": [
                {
                    "patterns_before": before,
                    "patterns_after": after,
                    "patterns_removed": before - after,
                    "reason": reason,
                    "metric_before_pruning": previous_metrics.get(request.metric or previous_metrics.get("primary_metric"), previous_metrics.get("primary_score")),
                    "metric_after_pruning": metrics.get(request.metric or metrics.get("primary_metric"), metrics.get("primary_score")),
                    "primary_metric": metrics.get("primary_metric"),
                }
            ],
            "metrics_before": [previous_metrics],
            "metrics": [metrics],
        }
        try:
            tables["audit_log"] = json.loads(editor.audit_report())
        except Exception:
            tables["audit_log"] = []
        message = _pruning_decision_summary(session, before, after, reason, metrics)
        writer_model = str(request.params.get("_writer_model") or "").strip()
        if writer_model:
            message = _polish_run_answer(
                writer_model=writer_model,
                question=request.question or "Summarize the pruning result.",
                draft=message,
                run_context=json.dumps({"pruning_summary": tables["pruning_summary"], "metrics_before": previous_metrics, "metrics_after": metrics}, indent=2, default=str),
            ) or message
        return ActionResult(
            ok=True,
            action=request.action,
            message=message,
            data={"session_id": session.session_id, "patterns_before": before, "patterns_after": after},
            tables=tables,
        )

    def _action_generate_governance_report(self, request: ActionRequest) -> ActionResult:
        session = self._session_for(request)
        out_dir = self._session_path(session.session_id) / "governance"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_id = f"hugiml-llm-{session.session_id}"
        card = generate_model_card(
            session.model,
            model_id=model_id,
            intended_use=f"Demonstration classification workflow for dataset {session.dataset}.",
            out_of_scope_use=(
                "Not for production decisioning without domain validation, fairness review, "
                "external test data, and human approval."
            ),
            training_data_description=json.dumps(session.info, default=str),
            evaluation_data_description="Held-out test split created by the HUGIML NLP orchestrator.",
            performance_metrics=session.metrics,
            ethical_considerations=(
                "Review features and mined patterns for proxy variables before deployment. "
                "Use the pruning workflow to remove operationally invalid or sensitive patterns."
            ),
        )
        card_path = out_dir / "model_card.md"
        card.save(str(card_path), fmt="md")
        manifest = package_audit_artifacts(session.model, model_id=model_id, output_dir=str(out_dir), model_card=card)
        session.artifacts.update({"model_card_md": str(card_path), "audit_manifest": str(manifest)})
        message = _governance_decision_summary(session, card_path, manifest)
        writer_model = str(request.params.get("_writer_model") or "").strip()
        if writer_model:
            message = _polish_run_answer(
                writer_model=writer_model,
                question=request.question or "Summarize the governance artifacts.",
                draft=message,
                run_context=json.dumps({
                    "session_id": session.session_id,
                    "metrics": session.metrics,
                    "artifacts": {"model_card_md": str(card_path), "audit_manifest": str(manifest)},
                }, indent=2, default=str),
            ) or message
        return ActionResult(
            ok=True,
            action=request.action,
            message=message,
            data={"session_id": session.session_id, "model_card": card.to_dict()},
            artifacts={"model_card_md": str(card_path), "audit_manifest": str(manifest)},
        )

    def _action_answer_api_question(self, request: ActionRequest) -> ActionResult:
        """Answer API/help questions from a cached local documentation index.

        Retrieval is only the first step.  The deterministic path turns the
        retrieved Sphinx/API material into a compact answer; when the UI has a
        supported Ollama model selected, that model is used only as a writer to
        polish the grounded answer, not as the executor or source of truth.
        """

        question = request.question or ""
        hits = self.docs_index.search(question, top_k=10)
        answer = build_docs_answer(question, hits)
        writer_model = str(request.params.get("_writer_model") or "").strip()
        if writer_model:
            answer = _polish_docs_answer(
                writer_model=writer_model,
                question=question,
                draft=answer,
                docs_context=build_docs_context(question, hits),
            ) or answer
        source_refs = [hit.to_dict() for hit in hits[:4]]
        return ActionResult(
            ok=True,
            action=request.action,
            message=answer,
            data={
                "question": request.question,
                "retrieval": "local_docs_index",
                "synthesis": "ollama_writer" if writer_model else "deterministic_docs_synthesis",
                "docs_index": self.docs_index.stats(),
                "source_count": len(hits),
                "sources": source_refs,
            },
            tables={},
        )

    def _limit_working_dataset(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        request: ActionRequest,
    ) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
        max_rows = request.params.get("_chat_max_rows")
        if max_rows is None:
            return X, y, {"sampled": False, "source_rows": int(len(y)), "rows_used": int(len(y))}
        try:
            limit = int(max_rows)
        except Exception:
            return X, y, {"sampled": False, "source_rows": int(len(y)), "rows_used": int(len(y))}
        if limit <= 0 or len(y) <= limit:
            return X, y, {"sampled": False, "source_rows": int(len(y)), "rows_used": int(len(y))}
        counts = pd.Series(y).value_counts(dropna=False)
        if len(counts) > 1:
            limit = max(limit, int(len(counts) * 2))
            stratify = y if int(counts.min()) >= 2 else None
        else:
            stratify = None
        X_work, _X_unused, y_work, _y_unused = train_test_split(
            X,
            y,
            train_size=min(limit, len(y) - 1),
            random_state=self.random_state,
            stratify=stratify,
        )
        return (
            X_work.reset_index(drop=True),
            np.asarray(y_work),
            {"sampled": True, "source_rows": int(len(y)), "rows_used": int(len(y_work))},
        )

    def _fit_session(self, request: ActionRequest, tune: bool) -> ModelSession:
        X, y, info = self.registry.load_dataset(request.dataset or "")
        target = request.target or info.target or "target"
        X, y, run_scope = self._limit_working_dataset(X, np.asarray(y), request)
        counts = pd.Series(y).value_counts(dropna=False)
        stratify = y if len(counts) > 1 and int(counts.min()) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(request.params.get("test_size", 0.25)),
            random_state=self.random_state,
            stratify=stratify,
        )
        if tune:
            model, metrics, chosen = self._tune_model(
                X_train, y_train, X_test, y_test, request.strategy, request.metric
            )
        else:
            params = self._params_for_strategy(request.strategy)
            params.update(_extract_model_params_from_request(request.params))
            model = HUGIMLClassifier(**params)
            model.fit(X_train, y_train)
            metrics = self._evaluate(model, X_test, y_test, request.metric)
            chosen = params
        metrics.update({
            "n_train": int(len(y_train)),
            "n_rows_used": int(run_scope.get("rows_used", len(y))),
            "source_rows": int(run_scope.get("source_rows", len(y))),
            "sampled_for_chat": bool(run_scope.get("sampled", False)),
        })
        sid = self._new_session_id(info.name)
        session = ModelSession(
            session_id=sid,
            dataset=info.name,
            target=target,
            info=info.to_dict(),
            X_train=X_train,
            X_test=X_test,
            y_train=np.asarray(y_train),
            y_test=np.asarray(y_test),
            model=model,
            metrics={**metrics, "chosen_params": chosen},
        )
        self.sessions[sid] = session
        self.last_session_id = sid
        self._write_session_manifest(session)
        return session

    def _tune_model(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        strategy: str,
        metric: str | None,
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        grid_name = "interpretability" if strategy == "interpretability" else "performance"
        try:
            grid = get_hugiml_grid(grid_name)
        except Exception:
            grid = {}
        # Keep v1 NLP tuning bounded. HUGIML owns the model-level hyperparameter semantics;
        # this loop just tries a few existing grid candidates.
        candidates = list(ParameterGrid(grid or {"B": [8], "L": [1], "G": [0.001], "topK": [30]}))
        max_trials = 4 if strategy in {"fast", "small_memory"} else 6
        candidates = candidates[:max_trials]
        best_model = None
        best_metrics: dict[str, Any] = {}
        best_params: dict[str, Any] = {}
        best_score = -np.inf
        primary = metric or "roc_auc"
        for params in candidates:
            params = {**self._params_for_strategy(strategy), **params}
            try:
                model = HUGIMLClassifier(**params)
                model.fit(X_train, y_train)
                metrics = self._evaluate(model, X_test, y_test, metric)
                score = _score_for_selection(metrics, primary)
            except Exception as exc:
                metrics = {"error": str(exc), "error_type": type(exc).__name__}
                score = -np.inf
            if score > best_score:
                best_score = score
                best_model = locals().get("model")
                best_metrics = metrics
                best_params = params
        if best_model is None:
            raise RuntimeError("All HUGIML tuning candidates failed.")
        return best_model, best_metrics, best_params

    @staticmethod
    def _params_for_strategy(strategy: str) -> dict[str, Any]:
        if strategy == "interpretability":
            return {
                "B": -1,
                "adaptive_binning": True,
                "L": 1,
                "G": 0.01,
                "topK": 30,
                "feature_mode": "patterns_only",
                "augmented_pair_transforms": False,
                "interaction_relaxed_mining": True,
                "n_jobs": 1,
            }
        if strategy in {"fast", "small_memory"}:
            return {
                "B": 6,
                "adaptive_binning": False,
                "L": 1,
                "G": 0.01,
                "topK": 20,
                "feature_mode": "patterns_only",
                "augmented_pair_transforms": False,
                "n_jobs": 1,
            }
        if strategy == "balanced":
            return {
                "B": -1,
                "adaptive_binning": True,
                "L": 1,
                "G": 0.003,
                "topK": 35,
                "feature_mode": "patterns_only",
                "augmented_pair_transforms": False,
                "interaction_relaxed_mining": True,
                "n_jobs": 1,
            }
        if strategy == "performance":
            return {
                "B": -1,
                "adaptive_binning": True,
                "L": 2,
                "G": 0.001,
                "topK": 50,
                "feature_mode": "original_plus_patterns",
                "augmented_pair_transforms": True,
                "n_jobs": 1,
            }
        # Unrecognized strategy strings fall back to the same setup as "performance".
        return {
            "B": -1,
            "adaptive_binning": True,
            "L": 1,
            "G": 0.001,
            "topK": 50,
            "feature_mode": "original_plus_patterns",
            "augmented_pair_transforms": True,
            "n_jobs": 1,
        }

    def _evaluate(self, model: Any, X_test: pd.DataFrame, y_test: np.ndarray, metric: str | None) -> dict[str, Any]:
        pred = model.predict(X_test)
        metrics: dict[str, Any] = {
            "n_test": int(len(y_test)),
            "accuracy": round(float(accuracy_score(y_test, pred)), 6),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 6),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 6),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 6),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 6),
        }
        try:
            proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 6)
            metrics["average_precision"] = round(float(average_precision_score(y_test, proba)), 6)
        except Exception:
            pass
        cm = confusion_matrix(y_test, pred).tolist()
        metrics["confusion_matrix"] = cm
        metrics["primary_metric"] = metric or ("roc_auc" if "roc_auc" in metrics else "f1")
        metrics["primary_score"] = metrics.get(metrics["primary_metric"], metrics.get("f1"))
        return metrics

    def _prediction_rows(self, session: ModelSession, limit: int = 10) -> list[dict[str, Any]]:
        X_preview = session.X_test.head(limit)
        pred = session.model.predict(X_preview)
        rows: list[dict[str, Any]] = []
        proba = None
        try:
            proba = session.model.predict_proba(X_preview)[:, 1]
        except Exception:
            pass
        for i, (_, row) in enumerate(X_preview.iterrows()):
            rec = {f"feature_{k}": _jsonable(v) for k, v in row.head(8).items()}
            rec["prediction"] = _jsonable(pred[i])
            if proba is not None:
                rec["probability_class_1"] = round(float(proba[i]), 6)
            rows.append(rec)
        return rows

    def _session_for(self, request: ActionRequest) -> ModelSession:
        sid = request.session_id or self.last_session_id
        if not sid or sid not in self.sessions:
            if request.dataset:
                return self._fit_session(ActionRequest(action="build_model", dataset=request.dataset), tune=False)
            raise KeyError("No active model session. Build or tune a model first.")
        return self.sessions[sid]

    def _session_result(self, session: ModelSession, action: str, prefix: str, request: ActionRequest | None = None) -> ActionResult:
        tables: dict[str, list[dict[str, Any]]] = {"metrics": [session.metrics], "model_configuration": _model_config_rows(session)}
        data: dict[str, Any] = {"session_id": session.session_id, "dataset": session.dataset, "target": session.target}
        try:
            tables["feature_importance"] = _df_records(session.model.feature_importances().head(10))
        except Exception:
            pass
        try:
            composition = session.model.get_model_composition()
            data["model_composition"] = composition
            tables["model_composition"] = _model_composition_rows(composition)
        except Exception:
            pass
        message = _build_result_summary(prefix, session, tables)
        writer_model = str(((request.params if request else {}) or {}).get("_writer_model") or "").strip()
        if writer_model:
            message = _polish_run_answer(
                writer_model=writer_model,
                question=(request.question if request else None) or prefix,
                draft=message,
                run_context=_run_context_for_writer(session, tables, data),
            ) or message
        data["summary"] = message
        return ActionResult(
            ok=True,
            action=action,
            message=message,
            data=data,
            tables=tables,
        )

    def _new_session_id(self, dataset: str) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        base = f"{stamp}-{dataset}".replace(" ", "_")
        sid = base
        counter = 1
        while sid in self.sessions or self._session_path(sid).exists():
            counter += 1
            sid = f"{base}-{counter}"
        return sid

    def _session_path(self, session_id: str) -> Path:
        return self.session_dir / session_id

    def _write_session_manifest(self, session: ModelSession) -> None:
        path = self._session_path(session.session_id)
        path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "session_id": session.session_id,
            "dataset": session.dataset,
            "target": session.target,
            "info": session.info,
            "metrics": session.metrics,
            "created_at": session.created_at,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def generate_qna_html(
    output_path: str | Path,
    *,
    title: str,
    dataset_description: dict[str, Any],
    build_result: ActionResult,
    explain_result: ActionResult,
    prune_result: ActionResult,
    governance_result: ActionResult,
    prediction_result: ActionResult | None = None,
) -> str:
    """Create a standalone, visual Q&A HTML page for an end-to-end use case."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metrics = (build_result.tables.get("metrics") or [{}])[0]
    feature_rows = explain_result.tables.get("feature_importance", [])[:12]
    prediction_rows = (prediction_result.tables.get("predictions", []) if prediction_result else [])[:8]
    prune_rows = prune_result.tables.get("pruning_summary", [])
    prune_summary = prune_rows[0] if prune_rows else {}
    class_rows = [
        {"label": str(k), "value": v}
        for k, v in (dataset_description.get("class_counts") or {}).items()
    ]
    confusion = _coerce_html_matrix(metrics.get("confusion_matrix"))

    pattern_total = len(feature_rows) if feature_rows else "—"
    rows = dataset_description.get("rows", "—")
    features = dataset_description.get("features", "—")
    target = dataset_description.get("target", "—")
    primary_metric = metrics.get("primary_metric", "primary score")
    primary_score = _fmt_html_num(metrics.get("primary_score"))
    accuracy = _fmt_html_num(metrics.get("accuracy"))
    f1 = _fmt_html_num(metrics.get("f1"))
    roc_auc = _fmt_html_num(metrics.get("roc_auc"))

    sections = [
        {
            "id": "s1",
            "num": "01",
            "tab": "Data",
            "title": "Read and profile the dataset",
            "note": "The assistant begins by summarizing the selected dataset before any model is fit.",
            "question": "Can you read the data and tell me what I am modeling?",
            "answer": (
                f"The active dataset is <b>{html.escape(str(dataset_description.get('name')))}</b>. "
                f"It has <b>{html.escape(str(rows))}</b> rows, <b>{html.escape(str(features))}</b> features, "
                f"and target column <code>{html.escape(str(target))}</code>."
                + _metric_cards([
                    ("Rows", rows),
                    ("Features", features),
                    ("Target", target),
                    ("Source", dataset_description.get("source", "registered")),
                ])
                + _svg_bar_chart(class_rows, title="Class balance")
                + _svg_dataset_profile_chart(dataset_description)
            ),
        },
        {
            "id": "s2",
            "num": "02",
            "tab": "Build",
            "title": "Build and tune a HUGIML model",
            "note": "The request is executed through the HUGIML-only action layer; no baseline models are exposed in this interface.",
            "question": "Build a HUGIML model and tune it for a strong held-out score.",
            "answer": (
                html.escape(build_result.message)
                + _metric_cards([
                    (str(primary_metric).upper(), primary_score),
                    ("ROC AUC", roc_auc),
                    ("Accuracy", accuracy),
                    ("F1", f1),
                ])
                + _svg_metric_profile_chart(metrics)
                + _html_table(build_result.tables.get("metrics", []), max_rows=1, max_cols=8)
                + _confusion_matrix_html(confusion)
            ),
        },
        {
            "id": "s3",
            "num": "03",
            "tab": "Patterns",
            "title": "Explain what drives the model",
            "note": "Interpretation is grounded in HUGIML artifacts: metrics, coefficients, supports, and pattern tables.",
            "question": "Which features or patterns are driving the predictions?",
            "answer": (
                html.escape(explain_result.data.get("summary", explain_result.message))
                + _svg_diverging_importance_chart(feature_rows, title="Top feature / pattern influence")
                + _svg_bar_chart(_importance_chart_rows(feature_rows), title="Absolute influence ranking")
                + _html_table(feature_rows, max_rows=8, max_cols=7)
            ),
        },
        {
            "id": "s4",
            "num": "04",
            "tab": "Predictions",
            "title": "Generate export-ready tabular output",
            "note": "The same action interface produces prediction tables and compact visual checks from active session artifacts.",
            "question": "Give me a table I can review or export.",
            "answer": (
                "The workbench returns structured tables alongside the plain-language answer. "
                "These tables are generated from HUGIML session artifacts, not free-form text."
                + _svg_prediction_chart(prediction_rows)
                + _html_table(prediction_rows or explain_result.tables.get("feature_importance", []), max_rows=8, max_cols=8)
            ),
        },
        {
            "id": "s5",
            "num": "05",
            "tab": "Prune",
            "title": "Prune with an audit reason",
            "note": "Pruning is a controlled governance action: remove patterns, refit the downstream estimator, and preserve the reason.",
            "question": "Can you simplify the model for review without retraining from scratch?",
            "answer": (
                html.escape(prune_result.message)
                + _metric_cards([
                    ("Before", prune_summary.get("patterns_before", "—")),
                    ("After", prune_summary.get("patterns_after", "—")),
                    ("Removed", prune_summary.get("patterns_removed", "—")),
                    ("Score after", _fmt_html_num(prune_summary.get("metric_after_pruning"))),
                ])
                + _svg_before_after_chart(prune_summary)
                + _svg_metric_profile_chart((prune_result.tables.get("metrics") or [{}])[0], title="Metrics after pruning")
                + _html_table(prune_rows, max_rows=3, max_cols=7)
            ),
        },
        {
            "id": "s6",
            "num": "06",
            "tab": "Govern",
            "title": "Create governance artifacts",
            "note": "The final step packages model-card and audit outputs for review.",
            "question": "Generate something the review board can file.",
            "answer": (
                html.escape(governance_result.message)
                + _metric_cards([
                    ("Model score", primary_score),
                    ("Pattern rows shown", pattern_total),
                    ("Audit status", "packaged" if governance_result.ok else "check"),
                    ("Interface scope", "HUGIML only"),
                ])
                + _svg_governance_flow_chart()
                + "<p>The model card and audit manifest are emitted by existing HUGIML governance APIs. "
                "The chat layer does not edit code, run shell commands, or invoke baseline models.</p>"
            ),
        },
    ]

    section_html = "\n".join(_qa_section(section) for section in sections)
    tabs_html = "\n".join(
        f'<a class="stage-tab" href="#{s["id"]}"><span class="num">{s["num"]}</span>{html.escape(s["tab"])}</a>'
        for s in sections
    )

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
  :root{{
    --paper:#ECEFE6; --panel:#F8FAF3; --ink:#1C2420; --ink-soft:#5B6358; --rule:#C7CFC0;
    --teal:#2C6E63; --teal-soft:#E4EEEA; --amber:#B5651D; --amber-soft:#F3E7D8;
    --violet:#6B5CA0; --violet-soft:#ECE8F5;
  }}
  *{{box-sizing:border-box;}}
  html{{scroll-behavior:smooth;}}
  body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;line-height:1.55;overflow-x:hidden;}}
  .mono, code{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}}
  p{{overflow-wrap:anywhere;}}
  header.masthead{{width:min(1060px,100%);margin:0 auto;padding:54px 24px 36px;border-bottom:1px solid var(--rule);}}
  .eyebrow{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-soft);margin:0 0 12px;}}
  h1{{font-family:Georgia,'Times New Roman',serif;font-size:clamp(34px,5vw,54px);line-height:1.05;margin:0 0 14px;letter-spacing:-.02em;}}
  .dek{{font-size:17px;color:var(--ink-soft);max-width:760px;margin:0 0 20px;}}
  .meta-row{{display:flex;flex-wrap:wrap;gap:10px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;}}
  .meta-chip{{border:1px solid var(--rule);padding:5px 10px;color:var(--ink-soft);background:var(--panel);}}
  .meta-chip b{{color:var(--ink);}}
  .stage-strip-wrap{{position:sticky;top:0;z-index:50;background:var(--paper);border-bottom:1px solid var(--rule);}}
  .stage-strip{{width:min(1060px,100%);margin:0 auto;display:flex;overflow-x:auto;}}
  .stage-tab{{flex:0 0 auto;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;letter-spacing:.04em;color:var(--ink-soft);text-decoration:none;padding:11px 16px 9px;border-right:1px solid var(--rule);border-bottom:3px solid transparent;white-space:nowrap;}}
  .stage-tab:hover,.stage-tab.active{{background:var(--panel);color:var(--ink);border-bottom-color:var(--amber);}}
  .stage-tab .num{{color:var(--amber);margin-right:6px;}}
  main{{width:min(1060px,100%);margin:0 auto;padding:0 24px 88px;}}
  section.stage{{padding:48px 0 12px;border-bottom:1px solid var(--rule);scroll-margin-top:54px;}}
  .stage-head{{display:flex;align-items:baseline;gap:14px;margin-bottom:7px;}}
  .stage-num{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;color:var(--amber);border:1px solid var(--amber);padding:2px 7px;}}
  h2{{font-family:Georgia,'Times New Roman',serif;font-size:25px;margin:0;}}
  .stage-note{{color:var(--ink-soft);font-size:14.5px;margin:6px 0 26px;max-width:720px;}}
  .bubble{{display:flex;gap:14px;align-items:flex-start;margin-bottom:14px;}}
  .tag{{flex:0 0 auto;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;font-weight:700;border:1px solid currentColor;margin-top:2px;}}
  .tag.q{{color:var(--teal);}} .tag.a{{color:var(--amber);}}
  .bubble-body{{flex:1;min-width:0;max-width:100%;overflow:hidden;}}
  .bubble.q .bubble-body{{background:var(--teal-soft);border-left:3px solid var(--teal);padding:12px 16px;}}
  .bubble.a .bubble-body{{background:var(--panel);border-left:3px solid var(--amber);padding:14px 16px;}}
  .speaker-label{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;margin-bottom:5px;}}
  .bubble.q .speaker-label{{color:var(--teal);}} .bubble.a .speaker-label{{color:var(--amber);}}
  .metric-row{{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:8px;margin:15px 0;}}
  .metric{{border:1px solid var(--rule);background:#fff;padding:10px 12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}}
  .metric .k{{display:block;color:var(--ink-soft);font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;margin-bottom:3px;}}
  .metric .v{{font-size:18px;font-weight:700;color:var(--ink);}}
  .readout{{position:relative;border:1px solid var(--rule);background:#fff;margin:14px 0 6px;overflow-x:auto;}}
  .readout-tag{{position:absolute;top:-1px;left:-1px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10px;letter-spacing:.06em;background:var(--ink);color:var(--paper);padding:3px 8px 2px;}}
  .table-wrap{{max-width:100%;overflow-x:auto;overflow-y:hidden;border:1px solid var(--rule);background:#fff;margin:14px 0 6px;-webkit-overflow-scrolling:touch;}}
  .table-wrap:focus{{outline:2px solid var(--teal);outline-offset:2px;}}
  table{{width:100%;min-width:720px;border-collapse:collapse;table-layout:fixed;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;}}
  th,td{{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top;overflow-wrap:anywhere;word-break:break-word;}}
  thead th{{color:var(--ink-soft);font-weight:600;text-transform:uppercase;font-size:10.5px;letter-spacing:.04em;border-bottom:1px solid var(--ink-soft);}}
  tbody tr:last-child td{{border-bottom:none;}}
  .chart-card{{border:1px solid var(--rule);background:#fff;margin:14px 0;padding:14px;max-width:100%;overflow:hidden;}}
  .chart-card svg{{display:block;max-width:100%;}}
  .chart-title{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-soft);margin-bottom:8px;}}
  .chart-legend{{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:var(--ink-soft);}}
  .dot{{display:inline-block;width:9px;height:9px;margin-right:5px;vertical-align:-1px;}}
  .dot.amber{{background:var(--amber);}} .dot.teal{{background:var(--teal);}}
  .bar-label{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;fill:var(--ink-soft);}}
  .bar-value{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;fill:var(--ink);font-weight:700;}}
  .matrix{{display:grid;grid-template-columns:repeat(2,minmax(110px,1fr));gap:6px;max-width:420px;margin-top:10px;}}
  .cell{{padding:16px;text-align:center;border:1px solid var(--rule);background:#fff;}}
  .cell.hot{{background:var(--amber-soft);}} .cell.cool{{background:var(--teal-soft);}}
  .cell b{{display:block;font-size:24px;}}
  footer{{width:min(1060px,100%);margin:0 auto;padding:36px 24px 70px;color:var(--ink-soft);font-size:13px;}}
  footer .rule{{border:none;border-top:1px solid var(--rule);margin:0 0 22px;}}
  @media (max-width:720px){{.metric-row{{grid-template-columns:1fr 1fr;}}main,header.masthead{{padding-left:18px;padding-right:18px;}}.bubble{{gap:10px;}}.tag{{width:24px;height:24px;font-size:11px;}}table{{min-width:620px;font-size:11.5px;}}}}
  @media (max-width:480px){{.metric-row{{grid-template-columns:1fr;}}section.stage{{padding-top:38px;}}table{{min-width:560px;}}}}
</style>
</head>
<body>
<header class="masthead">
  <p class="eyebrow">hugiml-core add-on · natural-language model workflow</p>
  <h1>{html.escape(title)}</h1>
  <p class="dek">A polished end-to-end walkthrough for a focused NLP interface: select data, build and tune a HUGIML model, inspect outputs, simplify the rule set, and produce governance artifacts. User-facing language stays practical; implementation details stay behind the scenes.</p>
  <div class="meta-row">
    <span class="meta-chip">dataset <b>{html.escape(str(dataset_description.get('name')))}</b></span>
    <span class="meta-chip">rows <b>{html.escape(str(rows))}</b></span>
    <span class="meta-chip">features <b>{html.escape(str(features))}</b></span>
    <span class="meta-chip">modeling <b>HUGIML only</b></span>
  </div>
</header>
<div class="stage-strip-wrap"><nav class="stage-strip" id="stageStrip" aria-label="Workflow stages">{tabs_html}</nav></div>
<main>{section_html}</main>
<footer><hr class="rule"><p><b>About this page.</b> This page is generated from the optional HUGIML NLP orchestrator. The interface translates natural-language workflow requests into a fixed set of HUGIML actions, then summarizes structured outputs such as metrics, pattern tables, pruning records, and governance artifacts.</p></footer>
<script>
(function(){{
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.stage-tab'));
  var sections = Array.prototype.slice.call(document.querySelectorAll('section.stage'));
  if(!('IntersectionObserver' in window)) return;
  var map = {{}};
  sections.forEach(function(s){{ map[s.id] = tabs.find(function(t){{ return t.getAttribute('href') === '#' + s.id; }}); }});
  var current = null;
  var observer = new IntersectionObserver(function(entries){{
    entries.forEach(function(entry){{
      if(entry.isIntersecting){{
        var tab = map[entry.target.id];
        if(tab && tab !== current){{
          if(current) current.classList.remove('active');
          tab.classList.add('active'); current = tab;
          document.getElementById('stageStrip').scrollTo({{left: tab.offsetLeft - 24, behavior: 'smooth'}});
        }}
      }}
    }});
  }}, {{ rootMargin: '-40% 0px -55% 0px', threshold: 0 }});
  sections.forEach(function(s){{ observer.observe(s); }});
}})();
</script>
</body>
</html>"""
    output.write_text(doc, encoding="utf-8")
    return str(output)


def _qa_section(section: dict[str, Any]) -> str:
    return f"""
<section class="stage" id="{html.escape(section['id'])}">
  <div class="stage-head"><span class="stage-num">{html.escape(section['num'])}</span><h2>{html.escape(section['title'])}</h2></div>
  <p class="stage-note">{html.escape(section['note'])}</p>
  <div class="bubble q"><div class="tag q">Q</div><div class="bubble-body"><div class="speaker-label">You</div><p>{html.escape(section['question'])}</p></div></div>
  <div class="bubble a"><div class="tag a">A</div><div class="bubble-body"><div class="speaker-label">Assistant</div>{section['answer']}</div></div>
</section>
"""


def _metric_cards(items: list[tuple[str, Any]]) -> str:
    cards = []
    for key, value in items:
        cards.append(
            f"<div class='metric'><span class='k'>{html.escape(str(key))}</span>"
            f"<span class='v'>{html.escape(str(value))}</span></div>"
        )
    return "<div class='metric-row'>" + "".join(cards) + "</div>"


def _importance_chart_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows[:10]:
        label = row.get("display_name") or row.get("pattern") or row.get("feature") or "pattern"
        value = row.get("abs_coefficient", row.get("coefficient", 0))
        out.append({"label": str(label), "value": value})
    return out


def _svg_bar_chart(rows: list[dict[str, Any]], title: str) -> str:
    clean = []
    for row in rows:
        try:
            value = abs(float(row.get("value", 0)))
        except Exception:
            continue
        if not np.isfinite(value):
            continue
        label = str(row.get("label", ""))[:46]
        clean.append((label, value))
    if not clean:
        return ""
    max_value = max(v for _, v in clean) or 1.0
    row_h = 30
    width = 760
    height = 28 + row_h * len(clean)
    svg_rows = []
    for i, (label, value) in enumerate(clean):
        y = 30 + i * row_h
        bar_w = 10 + int((width - 250) * (value / max_value))
        value_x = min(width - 54, 262 + bar_w)
        label_text = html.escape(label)
        svg_rows.append(
            f"<text class='bar-label' x='0' y='{y + 15}'><title>{label_text}</title>{label_text}</text>"
            f"<rect x='250' y='{y}' width='{min(bar_w, width - 255)}' height='18' rx='0'></rect>"
            f"<text class='bar-value' x='{value_x}' y='{y + 14}'>{html.escape(_fmt_html_num(value))}</text>"
        )
    return (
        f"<div class='chart-card'><div class='chart-title'>{html.escape(title)}</div>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}' width='100%' height='{height}'>"
        "<style>rect{fill:#2C6E63;opacity:.82;}</style>"
        + "".join(svg_rows)
        + "</svg></div>"
    )


def _svg_dataset_profile_chart(desc: dict[str, Any]) -> str:
    rows = []
    for key, value in (
        ("numeric", len(desc.get("numeric_features") or [])),
        ("categorical", len(desc.get("categorical_features") or [])),
        ("missing fields", len(desc.get("missing_top") or {})),
    ):
        rows.append({"label": key, "value": value})
    return _svg_bar_chart(rows, title="Dataset profile checks")


def _svg_metric_profile_chart(metrics: dict[str, Any], title: str = "Held-out metric profile") -> str:
    rows = []
    for key, label in (
        ("roc_auc", "ROC AUC"),
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced acc."),
        ("f1", "F1"),
    ):
        value = metrics.get(key)
        try:
            value_f = float(value)
        except Exception:
            continue
        if np.isfinite(value_f):
            rows.append({"label": label, "value": value_f})
    return _svg_bar_chart(rows, title=title)


def _svg_diverging_importance_chart(rows: list[dict[str, Any]], title: str) -> str:
    clean = []
    for row in rows[:12]:
        label = str(row.get("display_name") or row.get("pattern") or row.get("feature") or "pattern")
        raw = row.get("coefficient", row.get("importance", row.get("abs_coefficient", 0)))
        try:
            value = float(raw)
        except Exception:
            continue
        if np.isfinite(value):
            clean.append((label, value))
    if not clean:
        return ""
    max_abs = max(abs(v) for _, v in clean) or 1.0
    width = 820
    mid = 400
    row_h = 29
    height = 40 + row_h * len(clean)
    bits = [
        f"<line x1='{mid}' y1='26' x2='{mid}' y2='{height-12}' stroke='#5B6358' stroke-dasharray='3 4'></line>",
        f"<text class='bar-label' x='{mid}' y='{height-2}' text-anchor='middle'>0</text>",
    ]
    for i, (label, value) in enumerate(clean):
        y = 32 + i * row_h
        bar_w = max(2, int(260 * abs(value) / max_abs))
        color = "#B5651D" if value >= 0 else "#2C6E63"
        if value >= 0:
            rect = f"<rect x='{mid}' y='{y}' width='{bar_w}' height='16' fill='{color}' opacity='.86'></rect>"
            value_x = min(width - 34, mid + bar_w + 8)
            anchor = "start"
        else:
            rect = f"<rect x='{mid - bar_w}' y='{y}' width='{bar_w}' height='16' fill='{color}' opacity='.86'></rect>"
            value_x = max(226, mid - bar_w - 8)
            anchor = "end"
        short = html.escape(label[:64])
        full = html.escape(label, quote=True)
        bits.append(
            f"<text class='bar-label' x='0' y='{y+13}'><title>{full}</title>{short}</text>"
            + rect
            + f"<text class='bar-value' x='{value_x}' y='{y+13}' text-anchor='{anchor}'>{_fmt_html_num(value)}</text>"
        )
    return (
        f"<div class='chart-card'><div class='chart-title'>{html.escape(title)}</div>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}' width='100%' height='{height}'>"
        + "".join(bits)
        + "</svg><div class='chart-legend'><span><i class='dot amber'></i>pushes positive class</span><span><i class='dot teal'></i>pushes negative class</span></div></div>"
    )


def _svg_prediction_chart(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    prob_key = None
    for key in rows[0].keys():
        low = str(key).lower()
        if "prob" in low or low.startswith("p("):
            prob_key = key
    if prob_key is None:
        return ""
    clean = []
    for i, row in enumerate(rows[:8]):
        try:
            value = float(row.get(prob_key, 0))
        except Exception:
            continue
        if np.isfinite(value):
            clean.append((f"row {i+1}", max(0.0, min(1.0, value))))
    if not clean:
        return ""
    width = 760
    height = 170
    base_y = 136
    gap = 78
    bits = [
        f"<line x1='48' y1='{base_y}' x2='{width-30}' y2='{base_y}' stroke='#5B6358'></line>",
        "<text class='bar-label' x='4' y='44'>1.0</text><line x1='48' y1='40' x2='730' y2='40' stroke='#C7CFC0' stroke-dasharray='3 4'></line>",
        "<text class='bar-label' x='4' y='138'>0.0</text>",
    ]
    for i, (label, value) in enumerate(clean):
        x = 62 + i * gap
        bar_h = max(2, int(96 * value))
        y = base_y - bar_h
        bits.append(
            f"<rect x='{x}' y='{y}' width='34' height='{bar_h}' fill='#B5651D' opacity='.86'></rect>"
            f"<text class='bar-value' x='{x+17}' y='{max(18, y-6)}' text-anchor='middle'>{value:.2f}</text>"
            f"<text class='bar-label' x='{x+17}' y='158' text-anchor='middle'>{html.escape(label)}</text>"
        )
    return (
        "<div class='chart-card'><div class='chart-title'>Prediction probability snapshot</div>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Prediction probability snapshot' width='100%' height='{height}'>"
        + "".join(bits)
        + "</svg></div>"
    )


def _svg_before_after_chart(summary: dict[str, Any]) -> str:
    return _svg_bar_chart([
        {"label": "Before pruning", "value": summary.get("patterns_before", 0)},
        {"label": "After pruning", "value": summary.get("patterns_after", 0)},
        {"label": "Removed", "value": summary.get("patterns_removed", 0)},
    ], title="Pattern count before and after pruning")


def _svg_governance_flow_chart() -> str:
    items = [
        ("Model card", "Performance, use, limits"),
        ("Audit manifest", "Files and timestamps"),
        ("Pruning log", "Reasons and operator"),
        ("Review package", "Board-ready evidence"),
    ]
    width = 820
    height = 150
    card_w = 170
    bits = []
    for i, (title, caption) in enumerate(items):
        x = 24 + i * 198
        bits.append(
            f"<rect x='{x}' y='35' width='{card_w}' height='72' fill='#fff' stroke='#C7CFC0'></rect>"
            f"<text x='{x+14}' y='62' class='bar-value'>{html.escape(title)}</text>"
            f"<text x='{x+14}' y='86' class='bar-label'>{html.escape(caption)}</text>"
        )
        if i < len(items) - 1:
            bits.append(f"<path d='M{x+card_w+8} 71 L{x+card_w+28} 71' stroke='#B5651D' stroke-width='2'></path><path d='M{x+card_w+28} 71 L{x+card_w+21} 65 M{x+card_w+28} 71 L{x+card_w+21} 77' stroke='#B5651D' stroke-width='2' fill='none'></path>")
    return (
        "<div class='chart-card'><div class='chart-title'>Governance package flow</div>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Governance package flow' width='100%' height='{height}'>"
        + "".join(bits)
        + "</svg></div>"
    )


def _confusion_matrix_html(matrix: list[list[Any]] | None) -> str:
    if not matrix or len(matrix) < 2 or len(matrix[0]) < 2:
        return ""
    cells = []
    for i, row in enumerate(matrix[:2]):
        for j, value in enumerate(row[:2]):
            klass = "cool" if i == j else "hot"
            cells.append(f"<div class='cell {klass}'><span>actual {i} · pred {j}</span><b>{html.escape(str(value))}</b></div>")
    return "<div class='chart-card'><div class='chart-title'>Confusion matrix</div><div class='matrix'>" + "".join(cells) + "</div></div>"


def _coerce_html_matrix(value: Any) -> list[list[Any]] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            import ast
            value = ast.literal_eval(value)
        except Exception:
            return None
    if isinstance(value, (list, tuple)) and value and all(isinstance(row, (list, tuple)) for row in value):
        return [list(row) for row in value]
    return None


def _fmt_html_num(value: Any) -> str:
    try:
        val = float(value)
        if not np.isfinite(val):
            return "n/a"
        return f"{val:.4f}"
    except Exception:
        return "n/a" if value is None else str(value)



def _extract_model_params_from_request(params: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "B", "L", "G", "topK", "adaptive_binning", "feature_mode",
        "augmented_pair_transforms", "interaction_relaxed_mining", "aug_feature_size",
        "interaction_relaxed_feature_size", "n_jobs", "execution_mode",
    }
    model_params = dict(params.get("model_params", {}) or {})
    for key in allowed:
        if key in params and key not in model_params:
            model_params[key] = params[key]
    return {key: value for key, value in model_params.items() if key in allowed}


def _compact_metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ["primary_metric", "primary_score", "roc_auc", "average_precision", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "n_test"]
    return {key: metrics.get(key) for key in keys if metrics.get(key) is not None}


def _compact_param_row(params: dict[str, Any]) -> dict[str, Any]:
    keys = ["feature_mode", "B", "L", "G", "topK", "adaptive_binning", "augmented_pair_transforms", "interaction_relaxed_mining"]
    return {key: params.get(key) for key in keys if key in params}


def _model_config_rows(session: ModelSession) -> list[dict[str, Any]]:
    params = dict((session.metrics or {}).get("chosen_params") or {})
    if not params:
        params = {
            "B": getattr(session.model, "B", None),
            "L": getattr(session.model, "L", None),
            "G": getattr(session.model, "G", None),
            "topK": getattr(session.model, "topK", None),
            "adaptive_binning": getattr(session.model, "adaptive_binning", None),
            "feature_mode": getattr(session.model, "feature_mode", None),
        }
    return [{"parameter": key, "value": _jsonable(value)} for key, value in sorted(params.items())]


def _model_composition_rows(composition: dict[str, Any]) -> list[dict[str, Any]]:
    counts = composition.get("downstream_feature_counts") or {}
    origin = composition.get("pattern_origin_counts") or {}
    rows = [
        {"item": "feature_mode", "value": composition.get("feature_mode")},
        {"item": "input_features", "value": composition.get("n_input_features")},
        {"item": "patterns_mined", "value": composition.get("n_patterns_mined")},
        {"item": "downstream_features", "value": composition.get("n_downstream_features")},
        {"item": "original_features", "value": counts.get("original")},
        {"item": "pattern_features", "value": counts.get("pattern")},
        {"item": "augmented_pair_features", "value": counts.get("augmented_pair")},
    ]
    for key, value in sorted(origin.items()):
        rows.append({"item": f"pattern_origin_{key}", "value": value})
    return rows


def _representation_breakdown_tables(session: ModelSession, limit: int) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    try:
        imp = session.model.feature_importances()
        if "feature_type" in imp.columns:
            for feature_type, table_name in (
                ("original", "original_feature_details"),
                ("pattern", "pattern_feature_details"),
                ("augmented_pair", "augmented_pair_details"),
            ):
                subset = imp[imp["feature_type"].astype(str).eq(feature_type)].head(limit)
                if not subset.empty:
                    tables[table_name] = _df_records(subset)
    except Exception:
        pass
    try:
        features = list(session.model.get_downstream_features())
        grouped: dict[str, list[dict[str, Any]]] = {"original": [], "pattern": [], "augmented_pair": []}
        for name in features:
            text = str(name)
            if text.startswith("orig:"):
                grouped["original"].append({"feature": text[5:]})
            elif text.startswith("augmented_pair:"):
                grouped["augmented_pair"].append({"feature": text[len("augmented_pair:"):]})
            else:
                grouped["pattern"].append({"feature": text.removeprefix("pattern:")})
        if "original_feature_details" not in tables and grouped["original"]:
            tables["original_feature_details"] = grouped["original"][:limit]
        if "augmented_pair_details" not in tables and grouped["augmented_pair"]:
            tables["augmented_pair_details"] = grouped["augmented_pair"][:limit]
        if "pattern_feature_details" not in tables and grouped["pattern"]:
            tables["pattern_feature_details"] = grouped["pattern"][:limit]
    except Exception:
        pass
    return tables


def _predictor_overview_rows(X: pd.DataFrame) -> list[dict[str, Any]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    rows = [
        {"role": "numeric_predictors", "count": len(numeric_cols), "examples": ", ".join(map(str, numeric_cols[:10]))},
        {"role": "categorical_predictors", "count": int(X.shape[1] - len(numeric_cols)), "examples": ", ".join(map(str, [c for c in X.columns if c not in numeric_cols][:10]))},
        {"role": "total_predictors", "count": int(X.shape[1]), "examples": ", ".join(map(str, X.columns[:10]))},
    ]
    return rows


def _summary_statistics_rows(X: pd.DataFrame, *, limit: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    numeric = X.select_dtypes(include=[np.number]).head(0).columns.tolist()
    for col in numeric[:limit]:
        series = pd.to_numeric(X[col], errors="coerce")
        rows.append({
            "feature": str(col), "type": "numeric", "missing_fraction": round(float(series.isna().mean()), 6),
            "mean": round(float(series.mean()), 6) if series.notna().any() else None,
            "std": round(float(series.std()), 6) if series.notna().sum() > 1 else None,
            "min": round(float(series.min()), 6) if series.notna().any() else None,
            "max": round(float(series.max()), 6) if series.notna().any() else None,
            "unique": int(series.nunique(dropna=True)),
        })
    remaining = [c for c in X.columns if c not in numeric]
    for col in remaining[: max(0, limit - len(rows))]:
        series = X[col]
        mode = series.mode(dropna=True)
        rows.append({
            "feature": str(col), "type": "categorical", "missing_fraction": round(float(series.isna().mean()), 6),
            "unique": int(series.nunique(dropna=True)), "most_common": _jsonable(mode.iloc[0]) if not mode.empty else None,
        })
    return rows


def _dataset_decision_summary(desc: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> str:
    rows = desc.get("rows")
    features = desc.get("features")
    target = desc.get("target")
    balance = tables.get("class_balance") or []
    predictors = tables.get("predictor_overview") or []
    lines = [
        f"### Dataset summary — `{desc.get('name')}`", "",
        "**Shape and target**",
        f"- Rows: **{rows}**",
        f"- Predictors: **{features}**",
        f"- Target: `{target}`",
    ]
    if balance:
        lines.extend(["", "**Target distribution**", "", "| Class | Count |", "|---|---:|"])
        for row in balance:
            lines.append(f"| {row.get('class')} | {row.get('count')} |")
    if predictors:
        lines.extend(["", "**Predictor groups**", "", "| Group | Count | Examples |", "|---|---:|---|"])
        for row in predictors:
            lines.append(f"| {row.get('role')} | {row.get('count')} | {row.get('examples') or ''} |")
    if tables.get("missing_top"):
        lines.append("\n**Data quality note**\n- Some predictors contain missing values; review the missingness table before modeling.")
    else:
        lines.append("\n**Data quality note**\n- No non-zero missingness was found in the top missingness scan.")
    return "\n".join(lines).strip()


def _build_result_summary(prefix: str, session: ModelSession, tables: dict[str, list[dict[str, Any]]]) -> str:
    metrics = session.metrics or {}
    lines = [
        f"### {prefix} — `{session.dataset}`", "",
        "**Run configuration**", "", "| Parameter | Value |", "|---|---|",
    ]
    for row in tables.get("model_configuration", [])[:12]:
        lines.append(f"| {row.get('parameter')} | `{row.get('value')}` |")
    lines.extend(["", "**Held-out performance**", "", "| Metric | Value |", "|---|---:|"])
    for label, value in _metric_rows(metrics):
        lines.append(f"| {label} | {_fmt_markdown_num(value)} |")
    comp = tables.get("model_composition") or []
    if comp:
        lines.extend(["", "**Representation breakdown**", "", "| Item | Value |", "|---|---:|"])
        for row in comp:
            lines.append(f"| {row.get('item')} | {row.get('value')} |")
    return "\n".join(lines).strip()


def _comparison_decision_summary(dataset: str, rows: list[dict[str, Any]], primary: str, selected: Any) -> str:
    lines = [
        f"### Configuration comparison — `{dataset}`", "",
        f"Compared **{len(rows)}** HUGIML configurations on the same held-out split. The selected active session is **{selected}** using `{primary}`.", "",
        "| Configuration | Feature mode | Primary score | ROC AUC | Precision | Recall | F1 | Downstream features |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('configuration')} | {row.get('feature_mode')} | {_fmt_markdown_num(row.get('primary_score'))} | "
            f"{_fmt_markdown_num(row.get('roc_auc'))} | {_fmt_markdown_num(row.get('precision'))} | {_fmt_markdown_num(row.get('recall'))} | "
            f"{_fmt_markdown_num(row.get('f1'))} | {_fmt_markdown_num(row.get('downstream_features'))} |"
        )
    lines.extend(["", "**Decision note**", "- Prefer the configuration that meets the business metric target while keeping the representation small enough for review."])
    return "\n".join(lines).strip()


def _decision_summary(session: ModelSession, tables: dict[str, list[dict[str, Any]]], *, question: str = "") -> str:
    metrics = session.metrics or {}
    primary_metric = str(metrics.get("primary_metric") or "primary_score")
    primary_score = metrics.get("primary_score")
    metric_rows = _metric_rows(metrics)
    driver_rows = _driver_rows(tables.get("feature_importance") or tables.get("pattern_info") or [], limit=6)
    confusion_note = _confusion_note(metrics.get("confusion_matrix"))
    band = _score_band(primary_score)
    config_rows = tables.get("model_configuration") or []

    lines: list[str] = [
        f"### Findings summary — `{session.dataset}`",
        "",
        "**Decision summary**",
        f"- The active HUGIML model was evaluated on a held-out test split with `{primary_metric}` = **{_fmt_markdown_num(primary_score)}** ({band}).",
        f"- The target is `{session.target}` and the test set contains **{metrics.get('n_test', 'n/a')}** rows.",
    ]
    if confusion_note:
        lines.append(f"- {confusion_note}")
    if driver_rows:
        top = driver_rows[0]
        lines.append(f"- The strongest reported driver is **{top['driver']}** with effect/importance **{top['effect']}**.")
    else:
        lines.append("- Feature/pattern driver details were not available from the fitted model object.")

    # Surface the hyperparameters that were actually used unconditionally
    # (not just when the question explicitly asks for them). This is cheap
    # -- the table is already computed for every explain_model call -- and
    # it means questions phrased as "what parameters/configuration did you
    # use" are answered directly from this deterministic draft rather than
    # depending on an LLM polish pass that previously did not
    # received the configuration table in its context.
    if config_rows:
        lines.extend(["", "**Model configuration used**", "", "| Parameter | Value |", "|---|---:|"])
        for row in config_rows:
            lines.append(f"| {row.get('parameter')} | `{row.get('value')}` |")

    lines.extend(["", "**Performance snapshot**", "", "| Metric | Value |", "|---|---:|"])
    for label, value in metric_rows:
        lines.append(f"| {label} | {_fmt_markdown_num(value)} |")

    if driver_rows:
        lines.extend(["", "**Main drivers to review**", "", "| Rank | Driver | Effect / importance | Support |", "|---:|---|---:|---:|"])
        for row in driver_rows:
            lines.append(f"| {row['rank']} | {row['driver']} | {row['effect']} | {row['support']} |")

    lines.extend([
        "",
        "**Interpretability / governance implications**",
        "- Treat the top drivers as the first review queue: check whether they are domain-valid, stable, and free of sensitive/proxy leakage.",
        "- If a driver is invalid or too brittle, use the pruning workflow to remove it and compare metrics before/after pruning.",
        "- Do not approve the model from score alone; pair the validation metrics with pattern review and a model-card/audit package.",
        "",
        "**Recommended next steps**",
        "1. Review the highest-impact patterns/features with a domain owner.",
        "2. Generate predictions for a small held-out sample and inspect false positives/false negatives.",
        "3. Prune low-support or policy-invalid patterns, then compare the new score and pattern count.",
        "4. Generate governance artifacts once the evidence is acceptable.",
    ])
    return "\n".join(lines).strip()


def _metric_rows(metrics: dict[str, Any]) -> list[tuple[str, Any]]:
    preferred = [
        ("Primary score", metrics.get("primary_score")),
        ("ROC AUC", metrics.get("roc_auc")),
        ("Accuracy", metrics.get("accuracy")),
        ("Balanced accuracy", metrics.get("balanced_accuracy")),
        ("Precision", metrics.get("precision")),
        ("Recall", metrics.get("recall")),
        ("F1", metrics.get("f1")),
        ("Average precision", metrics.get("average_precision")),
        ("Test rows", metrics.get("n_test")),
    ]
    return [(label, value) for label, value in preferred if value is not None]


def _driver_rows(rows: list[dict[str, Any]], *, limit: int = 6) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i, row in enumerate(rows[:limit], start=1):
        driver = (
            row.get("pattern")
            or row.get("display_name")
            or row.get("feature")
            or row.get("feature_name")
            or row.get("name")
            or f"driver_{i}"
        )
        effect = (
            row.get("coefficient")
            if row.get("coefficient") is not None
            else row.get("importance")
            if row.get("importance") is not None
            else row.get("abs_coefficient")
            if row.get("abs_coefficient") is not None
            else row.get("score")
        )
        support = (
            row.get("support")
            if row.get("support") is not None
            else row.get("pattern_support")
            if row.get("pattern_support") is not None
            else row.get("coverage")
        )
        out.append({
            "rank": str(i),
            "driver": _trim_markdown(str(driver), 120),
            "effect": _fmt_markdown_num(effect),
            "support": _fmt_markdown_num(support),
        })
    return out


def _confusion_note(matrix: Any) -> str:
    try:
        if not isinstance(matrix, (list, tuple)) or len(matrix) < 2:
            return ""
        tn, fp = matrix[0][0], matrix[0][1]
        fn, tp = matrix[1][0], matrix[1][1]
        return f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}; review FP/FN trade-offs against business cost."
    except Exception:
        return ""


def _score_band(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "score unavailable"
    if val >= 0.85:
        return "strong initial signal"
    if val >= 0.75:
        return "useful but needs review"
    if val >= 0.65:
        return "moderate; validate carefully"
    return "weak/early-stage; improve before use"


def _fmt_markdown_num(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        val = float(value)
        if not np.isfinite(val):
            return "n/a"
        if abs(val) >= 1000 or float(val).is_integer():
            return f"{val:,.0f}"
        return f"{val:.4f}"
    except Exception:
        return str(value)


def _trim_markdown(value: str, limit: int) -> str:
    clean = " ".join(value.replace("\n", " ").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _pruning_decision_summary(session: ModelSession, before: int, after: int, reason: str, metrics: dict[str, Any]) -> str:
    removed = before - after
    metric = metrics.get("primary_metric") or "primary_score"
    score = metrics.get("primary_score")
    return "\n".join([
        f"### Pruning result — `{session.dataset}`",
        "",
        "**What changed**",
        f"- Removed **{removed}** pattern(s): **{before} → {after}**.",
        "- Refit the downstream classifier after pruning.",
        f"- Review reason: {reason}",
        "",
        "**Performance after pruning**",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| {metric} | {_fmt_markdown_num(score)} |",
        f"| Accuracy | {_fmt_markdown_num(metrics.get('accuracy'))} |",
        f"| F1 | {_fmt_markdown_num(metrics.get('f1'))} |",
        "",
        "**Decision note**",
        "Use this as a before/after governance checkpoint. If the performance change is acceptable, keep the pruned model and package the audit artifacts; otherwise inspect which removed patterns carried signal.",
    ]).strip()


def _governance_decision_summary(session: ModelSession, card_path: Path, manifest: Any) -> str:
    return "\n".join([
        f"### Governance artifacts — `{session.dataset}`",
        "",
        "Generated a model-card file and audit manifest for the active HUGIML session.",
        "",
        "**What reviewers now have**",
        f"- Session: `{session.session_id}`",
        f"- Target: `{session.target}`",
        f"- Primary score: **{_fmt_markdown_num(session.metrics.get('primary_score'))}** using `{session.metrics.get('primary_metric')}`",
        f"- Model card: `{card_path}`",
        f"- Audit manifest: `{manifest}`",
        "",
        "**Review checklist**",
        "1. Confirm intended use and out-of-scope use are correct.",
        "2. Review top patterns/features for proxy or policy issues.",
        "3. Confirm pruning history is justified and retained.",
        "4. Approve only after external/domain validation is complete.",
    ]).strip()


def _run_context_for_writer(session: ModelSession, tables: dict[str, list[dict[str, Any]]], data: dict[str, Any]) -> str:
    payload = {
        "session_id": session.session_id,
        "dataset": session.dataset,
        "target": session.target,
        "dataset_info": session.info,
        "metrics": session.metrics,
        "model_configuration": tables.get("model_configuration") or [],
        "top_feature_importance": (tables.get("feature_importance") or [])[:8],
        "top_pattern_info": (tables.get("pattern_info") or [])[:8],
        "model_summary": data.get("model_summary"),
    }
    return json.dumps(payload, indent=2, default=str)[:12000]


def _polish_docs_answer(*, writer_model: str, question: str, draft: str, docs_context: str) -> str:
    try:
        from .ollama_client import OllamaAnswerWriter
        from .runtime import FALLBACK_OLLAMA_MODEL

        candidates = [writer_model]
        if FALLBACK_OLLAMA_MODEL not in candidates:
            candidates.append(FALLBACK_OLLAMA_MODEL)
        for candidate in candidates:
            if not candidate:
                continue
            try:
                answer = OllamaAnswerWriter(candidate).polish_docs_answer(
                    question=question,
                    draft=draft,
                    docs_context=docs_context,
                )
                if answer:
                    return answer
            except Exception:
                continue
    except Exception:
        return ""
    return ""


def _polish_run_answer(*, writer_model: str, question: str, draft: str, run_context: str) -> str:
    try:
        from .ollama_client import OllamaAnswerWriter
        from .runtime import FALLBACK_OLLAMA_MODEL

        candidates = [writer_model]
        if FALLBACK_OLLAMA_MODEL not in candidates:
            candidates.append(FALLBACK_OLLAMA_MODEL)
        for candidate in candidates:
            if not candidate:
                continue
            try:
                answer = OllamaAnswerWriter(candidate).polish_run_answer(
                    question=question,
                    draft=draft,
                    run_context=run_context,
                )
                if answer:
                    return answer
            except Exception:
                continue
    except Exception:
        return ""
    return ""


def _grounded_summary(metrics: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> str:
    parts = [
        f"Held-out primary score is {metrics.get('primary_score')} using {metrics.get('primary_metric')}.",
        f"Accuracy is {metrics.get('accuracy')} and F1 is {metrics.get('f1')}.",
    ]
    fi = tables.get("feature_importance") or []
    if fi:
        top = fi[0]
        parts.append(
            "The highest-ranked artifact is "
            f"{top.get('pattern', top.get('feature', 'the first feature'))} "
            f"with coefficient {top.get('coefficient', 'n/a')}."
        )
    return " ".join(parts)


def _score_for_selection(metrics: dict[str, Any], metric: str) -> float:
    value = metrics.get(metric)
    if value is None:
        value = metrics.get("primary_score", metrics.get("f1", metrics.get("accuracy")))
    try:
        return float(value)
    except Exception:
        return -np.inf


def _df_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): _jsonable(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_jsonable(v) for v in value]
    if pd.isna(value):
        return None
    return value


def _html_table(rows: list[dict[str, Any]], max_rows: int = 10, max_cols: int = 12) -> str:
    if not rows:
        return ""
    rows = rows[:max_rows]
    keys: list[str] = []
    preferred = [
        "pattern", "feature", "display_name", "feature_type", "coefficient",
        "abs_coefficient", "support", "pattern_support", "n_test", "accuracy",
        "balanced_accuracy", "f1", "roc_auc", "primary_metric", "primary_score",
        "patterns_before", "patterns_after", "patterns_removed", "reason",
        "metric_after_pruning", "chosen_params",
    ]
    for key in preferred:
        if any(key in row for row in rows) and key not in keys:
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
            if len(keys) >= max_cols:
                break
        if len(keys) >= max_cols:
            break
    keys = keys[:max_cols]
    head = "".join(f"<th>{html.escape(str(k).replace('_', ' '))}</th>" for k in keys)
    body = ""
    for row in rows:
        cells = "".join(_html_cell(row.get(k, "")) for k in keys)
        body += f"<tr>{cells}</tr>"
    return (
        "<div class='table-wrap' role='region' aria-label='Scrollable data table' tabindex='0'>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        "</div>"
    )


def _html_cell(value: Any, max_chars: int = 110) -> str:
    if value is None:
        text = "—"
    elif isinstance(value, float):
        text = _fmt_html_num(value)
    elif isinstance(value, (dict, list, tuple)):
        try:
            import json
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
    else:
        text = str(value)
    full = text
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    title = f" title='{html.escape(full, quote=True)}'" if full != text else ""
    return f"<td{title}>{html.escape(text)}</td>"

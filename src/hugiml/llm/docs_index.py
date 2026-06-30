"""Lightweight local documentation retrieval for deterministic LLM answers.

The optional ``hugiml-llm`` interface must be useful even in deterministic
mode, where no local LLM is called.  This module builds a small cached search
index over the source checkout's documentation and, when docs are unavailable,
public API docstrings from the installed package/source tree.

The implementation intentionally uses only the Python standard library so it
works anywhere the core package runs.  It is not intended to replace a vector
store; it is a fast, explainable retrieval layer for API/help questions.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TEXT_EXTENSIONS = {".md", ".rst", ".txt", ".html", ".htm"}
_SOURCE_EXTENSIONS = {".py"}
_MAX_DOC_BYTES = 2_000_000
_MAX_CHUNK_CHARS = 4_500
_TOP_TERMS_PER_CHUNK = 250
_INDEX_VERSION = 6

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Za-z0-9_ .,/()\-]+\n[-=]{3,}\s*)$", re.M)
_STOP_WORDS = {
    "a",
    "about",
    "above",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "between",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "their",
    "then",
    "there",
    "this",
    "to",
    "use",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
    "your",
}
_API_HINT_WORDS = {
    "api",
    "class",
    "function",
    "method",
    "parameter",
    "parameters",
    "hyperparameter",
    "hyperparameters",
    "argument",
    "arguments",
    "usage",
    "example",
    "examples",
    "import",
    "fit",
    "predict",
    "explain",
    "prune",
    "governance",
    "model",
    "dataset",
}


@dataclass(frozen=True)
class DocumentationHit:
    """One retrieved documentation chunk."""

    path: str
    title: str
    snippet: str
    score: float
    line_start: int | None = None
    line_end: int | None = None
    source_kind: str = "docs"
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return compact source metadata suitable for UI display.

        The full retrieved text is intentionally not exposed here.  API-help
        answers should synthesize from the retrieved content instead of dumping
        snippets, paths, and line ranges into the chat.
        """

        return {
            "path": self.path,
            "title": self.title,
            "score": round(float(self.score), 4),
            "source_kind": self.source_kind,
        }


@dataclass
class _Chunk:
    path: str
    title: str
    text: str
    line_start: int | None
    line_end: int | None
    source_kind: str
    term_counts: dict[str, int]
    length: int


class DocumentationSearchIndex:
    """Cached lexical search over local HUGIML docs and public API docstrings."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root else None
        self.package_root = Path(__file__).resolve().parents[1]
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else self._default_cache_dir()
        self.cache_file = self.cache_dir / "hugiml_llm_docs_index.json"
        self._loaded = False
        self._chunks: list[_Chunk] = []
        self._df: dict[str, int] = {}
        self._avg_len = 1.0
        self._source_signature: list[dict[str, Any]] = []

    def search(self, query: str, *, top_k: int = 5) -> list[DocumentationHit]:
        """Return the most relevant local documentation chunks for ``query``."""

        self._ensure_loaded()
        query_terms = _tokens(query)
        if not query_terms or not self._chunks:
            return []
        expanded = _expand_query_terms(query_terms)
        query_counter = Counter(expanded)
        n_docs = max(len(self._chunks), 1)
        scored: list[tuple[float, _Chunk]] = []
        raw_query = " ".join(query_terms)
        for chunk in self._chunks:
            score = 0.0
            for term, q_count in query_counter.items():
                tf = chunk.term_counts.get(term, 0)
                if not tf:
                    continue
                df = max(self._df.get(term, 1), 1)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                # Tiny BM25-style normalisation. It is deterministic and fast,
                # while avoiding giant files dominating short API entries.
                denom = tf + 1.2 * (0.25 + 0.75 * chunk.length / max(self._avg_len, 1.0))
                score += idf * ((tf * 2.2) / denom) * min(q_count, 3)
            title_low = chunk.title.lower()
            text_low = chunk.text.lower()
            if raw_query and raw_query in text_low:
                score += 2.5
            for term in set(query_terms):
                if term in title_low:
                    score += 1.75
                if term in _path_tokens(chunk.path):
                    score += 0.8
            if chunk.source_kind == "docs":
                score *= 1.25
            elif chunk.source_kind == "readme":
                score *= 1.15
            elif any(term in _API_HINT_WORDS for term in query_terms):
                # Source docstrings are very useful for API questions, but docs
                # and README content should still win when available.
                score *= 1.05
            if score > 0:
                original_matches = {
                    term
                    for term in set(query_terms)
                    if term in chunk.term_counts or term in title_low or term in text_low or term in _path_tokens(chunk.path)
                }
                weak_help_terms = {"api", "apis", "parameter", "parameters", "argument", "arguments", "usage"}
                if original_matches and original_matches <= weak_help_terms:
                    score *= 0.35
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._hit_from_chunk(chunk, score, query_terms) for score, chunk in scored[:top_k]]

    def stats(self) -> dict[str, Any]:
        """Return basic index diagnostics for UI/CLI result metadata."""

        self._ensure_loaded()
        docs = len({chunk.path for chunk in self._chunks})
        return {
            "documents": docs,
            "chunks": len(self._chunks),
            "cache_file": str(self.cache_file),
            "repo_root": str(self.repo_root) if self.repo_root else None,
        }

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        sources = list(self._discover_sources())
        signature = [_file_signature(path, kind) for path, kind in sources]
        if self._load_cache(signature):
            self._loaded = True
            return
        chunks: list[_Chunk] = []
        for path, kind in sources:
            try:
                if path.suffix.lower() in _TEXT_EXTENSIONS:
                    chunks.extend(self._chunks_from_text_file(path, kind))
                elif path.suffix.lower() in _SOURCE_EXTENSIONS:
                    chunks.extend(self._chunks_from_python_file(path, kind))
            except Exception:
                # A single unreadable doc should not break deterministic help.
                continue
        self._chunks = chunks
        self._source_signature = signature
        self._recompute_stats()
        self._save_cache()
        self._loaded = True

    def _default_cache_dir(self) -> Path:
        root = self.repo_root if self.repo_root and self.repo_root.exists() else None
        if root is not None:
            return root / "LLM" / ".cache"
        return Path(tempfile.gettempdir()) / "hugiml_llm"

    def _discover_sources(self) -> Iterable[tuple[Path, str]]:
        """Yield docs first, then README/LLM notes, then public source fallback."""

        seen: set[Path] = set()
        roots: list[tuple[Path, str]] = []
        candidate_roots = []
        if self.repo_root is not None:
            candidate_roots.append(self.repo_root)
        # The installed package may not know the source checkout. Include the
        # package root so source docstrings remain searchable in wheel installs.
        candidate_roots.append(self.package_root.parent)
        for base in candidate_roots:
            for docs_name in ("docs", "doc"):
                docs_dir = base / docs_name
                if docs_dir.exists():
                    html_dir = docs_dir / "_build" / "html"
                    if html_dir.exists():
                        roots.append((html_dir, "docs"))
                    roots.append((docs_dir, "docs"))
            for rel in (
                "README.md",
                "LLM/README.md",
                "LLM/OVERLAY_MANIFEST.md",
                "LLM/prompts/system.md",
                "experiments/benchmark/benchmarkREADME.md",
                "experiments/scalability/scalabilityREADME.md",
            ):
                p = base / rel
                if p.exists():
                    roots.append((p, "readme"))
        for root, kind in roots:
            for path in _iter_files(root, _TEXT_EXTENSIONS):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved, kind
        for path in self._public_api_source_files():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved, "source"

    def _public_api_source_files(self) -> Iterable[Path]:
        source_root: Path | None = None
        if self.repo_root is not None and (self.repo_root / "src" / "hugiml").exists():
            source_root = self.repo_root / "src" / "hugiml"
        elif self.package_root.exists():
            source_root = self.package_root
        if source_root is None:
            return []
        preferred = [
            "__init__.py",
            "classifier.py",
            "adaptive.py",
            "calibration.py",
            "explainability.py",
            "governance.py",
            "hyperparameter_configs.py",
            "metrics.py",
            "monitoring.py",
            "multiclass.py",
            "plots.py",
            "pruning.py",
            "serialization.py",
            "llm/__init__.py",
            "llm/cli.py",
            "llm/dataset_registry.py",
            "llm/guardrails.py",
            "llm/orchestrator.py",
            "llm/planner.py",
            "llm/runtime.py",
            "llm/schemas.py",
        ]
        files = [source_root / rel for rel in preferred]
        return [path for path in files if path.exists()]

    def _chunks_from_text_file(self, path: Path, kind: str) -> list[_Chunk]:
        text = _read_text(path)
        if path.suffix.lower() in {".html", ".htm"}:
            text = _html_to_text(text)
        if not text.strip():
            return []
        rel = self._relative_path(path)
        pieces = _split_markdown_like(text)
        chunks: list[_Chunk] = []
        for title, body, line_start, line_end in pieces:
            for sub_text, sub_start, sub_end in _split_long_block(body, line_start):
                chunk_title = title or rel
                chunks.append(self._make_chunk(rel, chunk_title, sub_text, sub_start, sub_end, kind))
        return chunks

    def _chunks_from_python_file(self, path: Path, kind: str) -> list[_Chunk]:
        text = _read_text(path)
        if not text.strip():
            return []
        rel = self._relative_path(path)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [self._make_chunk(rel, rel, text[:_MAX_CHUNK_CHARS], 1, None, kind)]
        chunks: list[_Chunk] = []
        module_doc = ast.get_docstring(tree) or ""
        if module_doc:
            chunks.append(self._make_chunk(rel, f"{rel} module", module_doc, 1, None, kind))
        for node in tree.body:
            constant_name = _public_constant_name(node)
            if not constant_name:
                continue
            source = ast.get_source_segment(text, node) or ""
            if source:
                chunks.append(
                    self._make_chunk(
                        rel,
                        f"{rel}: {constant_name}",
                        source,
                        getattr(node, "lineno", None),
                        getattr(node, "end_lineno", None),
                        kind,
                    )
                )
        public_nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in public_nodes:
            name = getattr(node, "name", "")
            if not name:
                continue
            include_private_action = rel.endswith("llm/orchestrator.py") and name.startswith("_action_")
            if name.startswith("_") and not name.startswith("__") and not include_private_action:
                continue
            doc = ast.get_docstring(node) or ""
            signature = _signature_from_node(node)
            fields = _class_field_summary(node) if isinstance(node, ast.ClassDef) else ""
            title = f"{rel}: {name}"
            body = f"{signature}\n\n{doc}\n\n{fields}".strip()
            if body:
                chunks.append(
                    self._make_chunk(
                        rel,
                        title,
                        body,
                        getattr(node, "lineno", None),
                        getattr(node, "end_lineno", None),
                        kind,
                    )
                )
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    child_name = getattr(child, "name", "")
                    include_private_action_child = rel.endswith("llm/orchestrator.py") and child_name.startswith("_action_")
                    if not child_name or (
                        child_name.startswith("_")
                        and child_name not in {"__init__"}
                        and not include_private_action_child
                    ):
                        continue
                    child_doc = ast.get_docstring(child) or ""
                    child_sig = _signature_from_node(child, owner=name)
                    child_body = f"{child_sig}\n\n{child_doc}".strip()
                    if child_body:
                        chunks.append(
                            self._make_chunk(
                                rel,
                                f"{rel}: {name}.{child_name}",
                                child_body,
                                getattr(child, "lineno", None),
                                getattr(child, "end_lineno", None),
                                kind,
                            )
                        )
        if not chunks and text:
            chunks.append(self._make_chunk(rel, rel, text[:_MAX_CHUNK_CHARS], 1, None, kind))
        return chunks

    def _make_chunk(
        self,
        rel_path: str,
        title: str,
        text: str,
        line_start: int | None,
        line_end: int | None,
        kind: str,
    ) -> _Chunk:
        compact = _compact_text(text)
        counts = Counter(_tokens(f"{title}\n{compact}"))
        most_common = dict(counts.most_common(_TOP_TERMS_PER_CHUNK))
        return _Chunk(
            path=rel_path,
            title=_compact_text(title)[:180] or rel_path,
            text=compact[:_MAX_CHUNK_CHARS],
            line_start=line_start,
            line_end=line_end,
            source_kind=kind,
            term_counts=most_common,
            length=max(sum(most_common.values()), 1),
        )

    def _relative_path(self, path: Path) -> str:
        bases = [self.repo_root, self.package_root.parent, self.package_root]
        for base in bases:
            if base is None:
                continue
            try:
                return path.relative_to(base).as_posix()
            except ValueError:
                continue
        return path.name

    def _hit_from_chunk(self, chunk: _Chunk, score: float, query_terms: list[str]) -> DocumentationHit:
        return DocumentationHit(
            path=chunk.path,
            title=chunk.title,
            snippet=_snippet(chunk.text, query_terms),
            score=score,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            source_kind=chunk.source_kind,
            content=chunk.text,
        )

    def _recompute_stats(self) -> None:
        df: Counter[str] = Counter()
        total_len = 0
        for chunk in self._chunks:
            total_len += chunk.length
            for term in chunk.term_counts:
                df[term] += 1
        self._df = dict(df)
        self._avg_len = total_len / max(len(self._chunks), 1)

    def _load_cache(self, signature: list[dict[str, Any]]) -> bool:
        try:
            payload = json.loads(self.cache_file.read_text(encoding="utf-8"))
        except Exception:
            return False
        if payload.get("version") != _INDEX_VERSION:
            return False
        if payload.get("signature") != signature:
            return False
        try:
            self._chunks = [
                _Chunk(
                    path=item["path"],
                    title=item["title"],
                    text=item["text"],
                    line_start=item.get("line_start"),
                    line_end=item.get("line_end"),
                    source_kind=item.get("source_kind", "docs"),
                    term_counts=dict(item.get("term_counts") or {}),
                    length=int(item.get("length") or 1),
                )
                for item in payload.get("chunks", [])
            ]
            self._source_signature = signature
            self._recompute_stats()
            return True
        except Exception:
            self._chunks = []
            return False

    def _save_cache(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": _INDEX_VERSION,
                "signature": self._source_signature,
                "chunks": [
                    {
                        "path": chunk.path,
                        "title": chunk.title,
                        "text": chunk.text,
                        "line_start": chunk.line_start,
                        "line_end": chunk.line_end,
                        "source_kind": chunk.source_kind,
                        "term_counts": chunk.term_counts,
                        "length": chunk.length,
                    }
                    for chunk in self._chunks
                ],
            }
            self.cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            # Cache is an optimisation only. Deterministic help should work even
            # on read-only installs.
            pass


def build_docs_answer(question: str, hits: list[DocumentationHit]) -> str:
    """Create a concise, synthesized API-help answer from retrieved docs.

    Retrieval is only the grounding step.  The user-facing response should read
    like documentation help: explain the concept, identify the relevant call,
    summarise inputs/outputs, show a tiny usage example, and mention caveats.
    Raw snippets, line numbers, and source dumps stay out of the chat answer.
    """

    useful_hits = _rank_answer_hits(question, hits)
    if not useful_hits:
        return (
            "I could not find enough local Sphinx/API documentation to answer that cleanly. "
            "Ask about a specific class, method, parameter, workflow, pruning step, "
            "governance artifact, or `hugiml-llm` command."
        )

    specialised = _specialised_docs_answer(question, useful_hits)
    if specialised:
        return specialised

    focus = _answer_focus(question, useful_hits)
    content = "\n\n".join(_clean_retrieved_content(hit.content or hit.snippet) for hit in useful_hits[:5])
    paragraphs = _extract_explanatory_paragraphs(content)
    signatures = _extract_signatures(content)
    params = _extract_parameters(content)
    returns = _extract_returns(content)
    example = _extract_example(content, question, focus, signatures)
    caveats = _extract_caveats(content)

    lines: list[str] = [f"### {focus}", ""]
    summary = _compose_summary(question, focus, paragraphs, useful_hits)
    lines.append(summary)

    best_sig = _choose_signature(signatures, question, focus)
    if best_sig:
        lines.extend(["", "**Call shape**", "", f"`{best_sig}`"])

    if params:
        lines.extend(["", "**Key inputs**"])
        for name, desc in params[:6]:
            lines.append(f"- `{name}` — {desc}")

    if returns:
        lines.extend(["", "**Returns**"])
        for item in returns[:3]:
            lines.append(f"- {item}")

    if example:
        lines.extend(["", "**Typical usage**", "", "```python", example.strip(), "```"])

    if caveats:
        lines.extend(["", "**Important notes**"])
        for item in caveats[:4]:
            lines.append(f"- {item}")

    return "\n".join(lines).strip()


def build_docs_context(question: str, hits: list[DocumentationHit], *, max_chars: int = 7000) -> str:
    """Return compact, non-user-facing context for an optional answer writer.

    This intentionally includes cleaned excerpts because it is sent only to a
    local synthesis model.  The chat UI should display ``build_docs_answer`` or
    the polished writer output, not this raw context.
    """

    useful_hits = _rank_answer_hits(question, hits)
    pieces: list[str] = []
    for hit in useful_hits[:6]:
        text = _clean_retrieved_content(hit.content or hit.snippet)
        if not text:
            continue
        text = _compact_text(text)
        if len(text) > 1200:
            text = text[:1200].rsplit(" ", 1)[0].strip() + "…"
        title = _normalise_title(hit.title) or hit.path
        pieces.append(f"Source: {title} ({hit.path})\n{text}")
        if sum(len(piece) for piece in pieces) >= max_chars:
            break
    return "\n\n".join(pieces)[:max_chars].strip()


_HYPERPARAMETER_TERMS = {
    "hyperparameter",
    "hyperparameters",
    "parameter",
    "parameters",
    "param",
    "params",
    "grid",
    "grids",
    "tune",
    "tuning",
    "performance",
    "interpretability",
}

_FALLBACK_HUGIML_GRIDS: dict[str, dict[str, list[Any]]] = {
    "performance": {
        "B": [-1],
        "adaptive_binning": [True],
        "L": [1, 2],
        "topK": [50, 100],
        "feature_mode": ["original_plus_patterns"],
        "G": [0.01, 0.001],
    },
    "interpretability": {
        "B": [-1],
        "adaptive_binning": [True],
        "L": [1, 2],
        "topK": [50, 100],
        "feature_mode": ["patterns_only"],
        "G": [0.01, 0.001],
        "interaction_relaxed_mining": [True],
        "augmented_pair_transforms": [False],
    },
}

_GRID_PARAM_NOTES: dict[str, str] = {
    "B": "numeric bin count; `-1` means adaptive/binning-controlled selection in the recommended grids",
    "adaptive_binning": "lets HUGIML select per-feature numeric bins instead of forcing one global `B`",
    "L": "maximum HUG pattern length; `1` means singleton patterns, `2` allows pairs",
    "topK": "maximum retained pattern count / representation budget",
    "feature_mode": "downstream representation: pattern-only or original features plus HUG patterns",
    "G": "minimum information-gain threshold for mined patterns",
    "interaction_relaxed_mining": "allows interaction-information survivor columns into native mining while preserving pattern features",
    "augmented_pair_transforms": "adds downstream pair/operator features when enabled; disabled in the pattern-focused interpretability grid",
}

_PUBLIC_CONSTANT_NAMES = {
    "HUGIML_GRIDS",
    "DEFAULT_HUGIML_GRID_NAME",
    "BASELINE_MODEL_GRIDS",
    "CLASSIFIER_GRID",
    "DASHBOARD_GRID",
}


def _specialised_docs_answer(question: str, hits: list[DocumentationHit]) -> str:
    """Return a high-signal synthesized answer for common API-help intents."""

    if _is_hugiml_overview_question(question):
        return _build_hugiml_overview_answer(hits)
    if _is_hyperparameter_question(question):
        return _build_hyperparameter_answer(question, hits)
    if _is_pruning_docs_question(question):
        return _build_pruning_answer(hits)
    if _is_governance_docs_question(question):
        return _build_governance_answer(hits)
    return ""



def _is_hugiml_overview_question(question: str) -> bool:
    low = _compact_text(question).lower().rstrip("?")
    if not low:
        return False
    api_specific_terms = (
        "hugimlclassifier", "classifier", ".fit", " fit ", "predict", "predict_proba",
        "default_param_grid", "param_grid", "parameter", "parameters", "method", "methods",
        "api", "signature", "argument", "arguments", "return", "returns",
    )
    if any(term in f" {low} " for term in api_specific_terms):
        # Do not answer class/method/API questions with the broad product overview.
        return low in {"what is hugiml", "what's hugiml", "about hugiml"}
    overview_phrases = (
        "what is hugiml",
        "what's hugiml",
        "tell me about hugiml",
        "explain hugiml",
        "what does hugiml do",
        "why hugiml",
        "hugiml overview",
        "overview of hugiml",
    )
    return any(phrase in low for phrase in overview_phrases) or low in {"hugiml", "about hugiml"}


def _build_hugiml_overview_answer(hits: list[DocumentationHit]) -> str:
    """Answer broad product/concept questions from the overview docs.

    Broad questions should not inherit the title of an arbitrary retrieved page
    such as a benchmark or comparison section.  The docs repeatedly frame
    HUGIML as interpretable tabular ML based on High Utility Gain patterns, so
    present that as a concise overview with usage and decision context.
    """

    has_docs = bool(hits)
    lines = [
        "### What is HUGIML?",
        "",
        "HUGIML is an interpretable tabular machine-learning library built around **High Utility Gain (HUG) patterns**. Instead of training a black-box model and explaining it afterward, HUGIML mines human-readable bins, categories, and compound patterns, then uses those patterns as the model evidence itself.",
        "",
        "**Core idea**",
        "- Convert tabular data into interpretable candidate items such as numeric intervals and categorical conditions.",
        "- Mine supervised HUG patterns using information gain / utility thresholds.",
        "- Fit a downstream classifier on pattern-based or hybrid feature representations.",
        "- Expose pattern tables, coefficients, supports, feature importance, model summaries, pruning, and governance artifacts for review.",
        "",
        "**What it is useful for**",
        "- Classification problems where reviewers need to understand which conditions drive predictions.",
        "- Regulated or audited workflows such as credit-risk, healthcare-triage, churn, eligibility, or policy-review models.",
        "- Cases where a compact set of readable patterns is more useful than a purely post-hoc explanation.",
        "",
        "**Typical workflow**",
        "1. Fit `HUGIMLClassifier` on a tabular dataset.",
        "2. Inspect held-out metrics and confusion matrix.",
        "3. Review `feature_importances()`, `get_pattern_info()`, and `model_summary()`.",
        "4. Tune with the `performance` grid when score matters most, or `interpretability` when the final model should remain pattern-focused.",
        "5. Use pruning and governance APIs to remove invalid patterns and package audit artifacts.",
        "",
        "**Minimal usage**",
        "",
        "```python",
        "from hugiml import HUGIMLClassifier",
        "",
        "model = HUGIMLClassifier(adaptive_binning=True, L=1, G=5e-3, topK=100)",
        "model.fit(X_train, y_train)",
        "pred = model.predict(X_test)",
        "patterns = model.get_pattern_info()",
        "importance = model.feature_importances()",
        "```",
        "",
        "**Decision-maker view**",
        "HUGIML is best understood as a model-development and review tool: it gives a validation score, but it also gives the evidence trail behind that score. That makes it suitable when the question is not only “does the model perform?” but also “can a human review, prune, govern, and explain the model?”",
    ]
    if not has_docs:
        lines.append("\n_The local docs index did not return overview pages, so this answer uses the packaged HUGIML API conventions._")
    return "\n".join(lines).strip()


def _is_pruning_docs_question(question: str) -> bool:
    low = question.lower()
    if not any(term in low for term in ("prune", "pruning", "pattern editor", "remove pattern")):
        return False
    # Actual imperative pruning is handled as an orchestrator action.  This is
    # for explanatory/API questions about how pruning works.
    return any(term in low for term in ("how", "what", "explain", "api", "docs", "usage", "works"))


def _build_pruning_answer(hits: list[DocumentationHit]) -> str:
    return "\n".join([
        "### Pruning in HUGIML",
        "",
        "Pruning is the controlled way to remove mined patterns after review, then refit the downstream classifier while keeping an audit trail of what changed and why.",
        "",
        "**When to prune**",
        "- A pattern references a sensitive/proxy feature or is operationally invalid.",
        "- A pattern has very low support and is too brittle for review.",
        "- A pattern has excessive drift, PSI, or fails a model-risk/gov check.",
        "- The model should be simplified before sign-off.",
        "",
        "**Typical workflow**",
        "1. Build or tune a model.",
        "2. Inspect `get_pattern_info()` / `feature_importances()`.",
        "3. Use `PatternEditor` to remove patterns by index, keyword, or minimum support.",
        "4. Refit the downstream classifier on the remaining pattern representation.",
        "5. Finalize the model and keep the audit report.",
        "",
        "**API shape**",
        "",
        "```python",
        "from hugiml.pruning import PatternEditor",
        "",
        "editor = PatternEditor(model, operator_name=\"reviewer\")",
        "editor.remove([0, 3], reason=\"Rejected during model-risk review\")",
        "# or: editor.remove_low_support(0.02, reason=\"Remove brittle low-support patterns\")",
        "editor.refit(X_train, y_train)",
        "model = editor.finalize()",
        "audit_log = editor.audit_report()",
        "```",
        "",
        "**What to watch**",
        "After pruning, compare pattern count and validation metrics before/after. A small score drop may be acceptable if the remaining model is materially easier to audit and defend.",
    ]).strip()


def _is_governance_docs_question(question: str) -> bool:
    low = question.lower()
    if not any(term in low for term in ("governance", "audit", "model card", "compliance")):
        return False
    return any(term in low for term in ("how", "what", "explain", "api", "docs", "usage", "artifact", "report"))


def _build_governance_answer(hits: list[DocumentationHit]) -> str:
    return "\n".join([
        "### Governance and audit in HUGIML",
        "",
        "HUGIML’s governance path packages the model evidence that reviewers usually need: model purpose, training/evaluation context, performance metrics, pattern evidence, pruning history, and audit artifacts.",
        "",
        "**What governance covers**",
        "- Model-card style documentation for intended use, out-of-scope use, metrics, and ethical considerations.",
        "- Pattern evidence from the fitted HUGIML model, including supports, coefficients, and selected feature families.",
        "- Pruning records that explain which patterns were removed and the review reason.",
        "- Audit packaging so the review bundle can be filed or shared.",
        "",
        "**Typical workflow**",
        "1. Build or tune the model and inspect validation metrics.",
        "2. Review pattern/feature evidence and prune anything invalid.",
        "3. Generate a model card and audit manifest.",
        "4. Compare metrics and pattern counts before approving the model.",
        "",
        "**API shape**",
        "",
        "```python",
        "from hugiml.governance import generate_model_card, package_audit_artifacts",
        "",
        "card = generate_model_card(model, performance_metrics=metrics)",
        "manifest = package_audit_artifacts(model, model_id=\"reviewed-model\", model_card=card)",
        "```",
        "",
        "**Decision-maker view**",
        "The governance output should answer: how well did the model perform, what evidence drove it, what was removed or changed, and what residual risks still need human review.",
    ]).strip()


def _is_hyperparameter_question(question: str) -> bool:
    terms = set(_tokens(question))
    joined = " ".join(terms)
    return bool(terms & _HYPERPARAMETER_TERMS) or "param_grid" in joined or "default_param_grid" in joined


def _build_hyperparameter_answer(question: str, hits: list[DocumentationHit]) -> str:
    """Synthesize named-grid docs into a compact human answer.

    The docs contain the exact named grids.  This function extracts the values
    from retrieved Sphinx/source content, then presents the two user-facing
    choices side by side instead of dumping the retrieved chunks.
    """

    grids = _hyperparameter_grids_from_hits(hits)
    if not grids:
        grids = {name: dict(values) for name, values in _FALLBACK_HUGIML_GRIDS.items()}
    performance = grids.get("performance") or _FALLBACK_HUGIML_GRIDS["performance"]
    interpretability = grids.get("interpretability") or _FALLBACK_HUGIML_GRIDS["interpretability"]

    params = _ordered_grid_params(performance, interpretability)
    rows = [
        "| Hyperparameter | Performance grid | Interpretability grid | Meaning |",
        "|---|---:|---:|---|",
    ]
    for name in params:
        rows.append(
            "| "
            f"`{name}` | "
            f"{_format_grid_value(performance.get(name))} | "
            f"{_format_grid_value(interpretability.get(name))} | "
            f"{_GRID_PARAM_NOTES.get(name, 'documented HUGIML tuning option')} |"
        )

    lines = [
        "### HUGIML hyperparameters",
        "",
        "HUGIML exposes two recommended named grids for tuning: `performance` and `interpretability`. They share the same basic mining budget, but differ in the final representation and audit posture.",
        "",
        "**Recommended grids**",
        "",
        *rows,
        "",
        "**When to use which**",
        "",
        "| Goal | Use | Why |",
        "|---|---|---|",
        "| Strong first-pass validation score | `performance` | Keeps original features plus mined HUG patterns. |",
        "| Cleaner audit / pattern-focused model | `interpretability` | Uses `patterns_only`, enables interaction-relaxed mining, and disables augmented-pair transforms. |",
        "",
        "**How to use them**",
        "",
        "```python",
        "from hugiml import HUGIMLClassifier",
        "",
        "performance_grid = HUGIMLClassifier.default_param_grid('performance')",
        "interpretability_grid = HUGIMLClassifier.default_param_grid('interpretability')",
        "",
        "result = HUGIMLClassifier.tune(",
        "    X_train,",
        "    y_train,",
        "    cv=5,",
        "    scoring='roc_auc',",
        "    param_grid='performance',  # or 'interpretability'",
        "    refit=True,",
        "    use_fast_path=True,",
        ")",
        "```",
        "",
        "**Practical notes**",
        "- Keep follow-up grids compact after the recommended first pass.",
        "- For `L >= 2`, do not enable interaction-relaxed mining and augmented-pair transforms in the same candidate.",
    ]
    return "\n".join(lines).strip()


def _compact_source_paths(hits: list[DocumentationHit], *, limit: int = 3) -> str:
    paths: list[str] = []
    for hit in hits:
        path = hit.path
        if path not in paths:
            paths.append(path)
        if len(paths) >= limit:
            break
    return "; ".join(f"`{path}`" for path in paths)


def _prioritise_hyperparameter_sources(hits: list[DocumentationHit]) -> list[DocumentationHit]:
    def score(hit: DocumentationHit) -> tuple[int, float]:
        combined = f"{hit.path} {hit.title}".lower()
        priority = 0
        if "docs/tuning" in combined:
            priority += 5
        if "hyperparameter_configs" in combined:
            priority += 4
        if "default_param_grid" in combined:
            priority += 3
        if "docs/" in combined:
            priority += 2
        if hit.source_kind == "readme":
            priority += 1
        return (priority, float(hit.score))

    filtered = [hit for hit in hits if hit.source_kind in {"docs", "source", "readme"}]
    return sorted(filtered, key=score, reverse=True)


def _hyperparameter_grids_from_hits(hits: list[DocumentationHit]) -> dict[str, dict[str, list[Any]]]:
    grids: dict[str, dict[str, list[Any]]] = {}
    for hit in hits:
        text = _clean_retrieved_content(hit.content or hit.snippet)
        for name, grid in _extract_grid_assignments(text).items():
            grids[name] = grid
    return grids


def _extract_grid_assignments(text: str) -> dict[str, dict[str, list[Any]]]:
    """Extract performance/interpretability grids from docs or source chunks."""

    grids: dict[str, dict[str, list[Any]]] = {}
    for assignment_name, value in _assignment_dicts_from_text(text).items():
        if assignment_name == "HUGIML_GRIDS":
            for name, grid in value.items():
                if isinstance(name, str) and isinstance(grid, dict):
                    grids[name] = _normalise_grid_dict(grid)
        elif assignment_name in {"performance_grid", "interpretability_grid"}:
            grid_name = assignment_name.replace("_grid", "")
            if isinstance(value, dict):
                grids[grid_name] = _normalise_grid_dict(value)
    return grids


def _assignment_dicts_from_text(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    names = ("HUGIML_GRIDS", "performance_grid", "interpretability_grid")
    for name in names:
        for pattern in (rf"\b{name}\s*[:=][^=]*=\s*{{", rf"\b{name}\s*=\s*{{"):
            for match in re.finditer(pattern, text):
                brace_start = text.find("{", match.start())
                if brace_start < 0:
                    continue
                literal = _balanced_brace_text(text, brace_start)
                if not literal:
                    continue
                parsed = _safe_literal_eval_dict(literal)
                if isinstance(parsed, dict):
                    out[name] = parsed
        # Sphinx code blocks often have a simple assignment without annotation.
        simple = re.search(rf"\b{name}\s*=\s*{{", text)
        if simple and name not in out:
            literal = _balanced_brace_text(text, text.find("{", simple.start()))
            parsed = _safe_literal_eval_dict(literal) if literal else None
            if isinstance(parsed, dict):
                out[name] = parsed
    return out


def _balanced_brace_text(text: str, start: int) -> str:
    if start < 0 or start >= len(text) or text[start] != "{":
        return ""
    depth = 0
    quote = ""
    escaped = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    return ""


def _safe_literal_eval_dict(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"#.*", "", text)
    try:
        value = ast.literal_eval(cleaned)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _normalise_grid_dict(grid: dict[Any, Any]) -> dict[str, list[Any]]:
    normalised: dict[str, list[Any]] = {}
    for key, value in grid.items():
        name = str(key)
        if isinstance(value, list):
            normalised[name] = value
        elif isinstance(value, tuple):
            normalised[name] = list(value)
        else:
            normalised[name] = [value]
    return normalised


def _ordered_grid_params(*grids: dict[str, list[Any]]) -> list[str]:
    preferred = [
        "B",
        "adaptive_binning",
        "L",
        "topK",
        "feature_mode",
        "G",
        "interaction_relaxed_mining",
        "augmented_pair_transforms",
    ]
    present = {key for grid in grids for key in grid}
    ordered = [name for name in preferred if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _format_grid_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, list):
        if not value:
            return "-"
        if len(value) == 1:
            return f"`{value[0]!r}`".replace("'", "") if not isinstance(value[0], str) else f"`{value[0]}`"
        rendered = []
        for item in value:
            if isinstance(item, str):
                rendered.append(item)
            else:
                rendered.append(repr(item))
        return "`[" + ", ".join(rendered) + "]`"
    return f"`{value}`"


def _public_constant_name(node: ast.AST) -> str:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in _PUBLIC_CONSTANT_NAMES:
                return target.id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        if node.target.id in _PUBLIC_CONSTANT_NAMES:
            return node.target.id
    return ""

_GENERIC_TITLES = {
    "api reference",
    "basic usage",
    "quick start",
    "example",
    "usage",
    "section",
}
_SECTION_NAMES = {
    "parameters",
    "returns",
    "raises",
    "examples",
    "notes",
    "thread safety",
    "see also",
}
_CAVEAT_WORDS = {
    "requires",
    "must",
    "warning",
    "caution",
    "fallback",
    "production",
    "audit",
    "thread",
    "safe",
    "unavailable",
    "raises",
    "only",
}


def _rank_answer_hits(question: str, hits: list[DocumentationHit]) -> list[DocumentationHit]:
    q_terms = set(_tokens(question))
    ranked: list[tuple[float, int, DocumentationHit]] = []
    for pos, hit in enumerate(hits):
        text = _clean_retrieved_content(hit.content or hit.snippet)
        if not _has_substantive_content(text):
            continue
        score = float(hit.score)
        title_low = hit.title.lower()
        path_low = hit.path.lower()
        if hit.source_kind == "docs":
            score += 1.0
        if "docs/" in path_low:
            score += 0.8
        if hit.source_kind == "source" and _extract_signatures(text):
            score += 1.4
        if _title_is_generic(title_low):
            score -= 1.2
        if q_terms & set(_tokens(hit.title)):
            score += 0.9
        if any(term in text.lower() for term in q_terms):
            score += 0.4
        ranked.append((score, -pos, hit))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [hit for _, _, hit in ranked]


def _has_substantive_content(text: str) -> bool:
    low = text.lower()
    if len(_tokens(text)) < 12:
        return False
    if ".. automodule::" in low and len(_tokens(text)) < 80:
        return False
    return True


def _title_is_generic(title: str) -> bool:
    clean = re.sub(r"[^a-z0-9_ .-]+", "", title).strip().lower()
    return clean in _GENERIC_TITLES or clean.startswith("api reference")


def _answer_focus(question: str, hits: list[DocumentationHit]) -> str:
    q = _compact_text(question).rstrip("?")
    for hit in hits:
        text = _clean_retrieved_content(hit.content or hit.snippet)
        sig = _choose_signature(_extract_signatures(text), question, hit.title)
        if sig:
            name = sig.split("(", 1)[0].strip()
            name = name.replace("HUGIMLClassifierNative", "HUGIMLClassifier")
            if "." in name or name.lower() in " ".join(_tokens(question)):
                return name
        title = _normalise_title(hit.title)
        if title and not _title_is_generic(title.lower()):
            return title
    return q[:90] or "HUGIML API help"


def _normalise_title(title: str) -> str:
    clean = _compact_text(title)
    if ": " in clean:
        clean = clean.split(": ", 1)[1]
    clean = clean.replace("HUGIMLClassifierNative", "HUGIMLClassifier")
    return clean[:90]


def _clean_retrieved_content(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^[-=~`#*]{3,}\s*$", "", text, flags=re.M)
    text = re.sub(r"\.\.\s+(automodule|autoclass|autofunction|currentmodule)::.*", "", text)
    text = re.sub(r"^\s+:(members|show-inheritance|undoc-members|special-members):.*", "", text, flags=re.M)
    text = re.sub(r":class:`~?([^`]+)`", r"`\1`", text)
    text = re.sub(r":meth:`~?([^`]+)`", r"`\1`", text)
    text = re.sub(r":func:`~?([^`]+)`", r"`\1`", text)
    text = re.sub(r":mod:`~?([^`]+)`", r"`\1`", text)
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    text = re.sub(r"`([^`<>]+)\s*<[^`>]+>`_", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_signatures(text: str) -> list[str]:
    signatures: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or len(line) > 220:
            continue
        if "(" not in line or ")" not in line:
            continue
        if line.startswith(("if ", "for ", "while ", "return ", "print(", "raise ")):
            continue
        if re.match(r"^(?:[A-Za-z_]\w*\.){0,3}[A-Za-z_]\w*\([^\n]*\)(?:\s*->\s*[^\n]+)?$", line):
            signatures.append(line.replace("HUGIMLClassifierNative", "HUGIMLClassifier"))
    return _dedupe(signatures)


def _extract_explanatory_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        clean = _paragraph_to_sentence(para)
        if not clean:
            continue
        low = clean.lower().strip()
        if low in _SECTION_NAMES:
            continue
        if low.startswith(("parameters", "returns", "raises", "examples", "fields/attributes")):
            continue
        if clean.startswith((".. ", ":", "* ", "- ")):
            continue
        if "(" in clean and ")" in clean and len(clean.split()) <= 5:
            continue
        if len(clean.split()) < 5:
            continue
        paragraphs.append(clean)
    return _dedupe(paragraphs)


def _paragraph_to_sentence(para: str) -> str:
    lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
    if not lines:
        return ""
    # Skip code-heavy blocks; examples are handled separately.
    if sum(1 for ln in lines if ln.startswith((">>>", "from ", "import ", "clf.", "model.", "editor."))) >= 2:
        return ""
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(#+\s*)", "", text)
    text = text.strip("-* ")
    if not text:
        return ""
    # Keep the first two sentences max for a compact answer.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:2]).strip()


def _extract_parameters(text: str) -> list[tuple[str, str]]:
    sections = _named_sections(text, {"parameters", "fields/attributes"})
    params: list[tuple[str, str]] = []
    for section in sections:
        if section.lower().startswith("fields/attributes:"):
            fields = section.split(":", 1)[1]
            for item in fields.split(","):
                name = item.strip()
                if not name:
                    continue
                params.append((name.split(":", 1)[0].strip(), _compact_text(name)))
            continue
        current_name = ""
        current_desc: list[str] = []
        for raw in section.splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped or set(stripped) in ({"-"}, {"="}):
                continue
            match = re.match(r"^([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*:\s*(.+)$", stripped)
            if match:
                if current_name:
                    params.append((current_name, _clean_description(" ".join(current_desc))))
                current_name = match.group(1).replace(" ", "")
                current_desc = [match.group(2)]
            elif current_name and (raw.startswith(" ") or raw.startswith("\t")):
                current_desc.append(stripped)
        if current_name:
            params.append((current_name, _clean_description(" ".join(current_desc))))
    cleaned: list[tuple[str, str]] = []
    for name, desc in params:
        if not name or name.lower() in _SECTION_NAMES:
            continue
        cleaned.append((name, desc or "documented input"))
    return _dedupe_pairs(cleaned)


def _extract_returns(text: str) -> list[str]:
    sections = _named_sections(text, {"returns"})
    returns: list[str] = []
    for section in sections:
        lines = [ln.strip() for ln in section.splitlines() if ln.strip() and set(ln.strip()) not in ({"-"}, {"="})]
        if not lines:
            continue
        joined = _clean_description(" ".join(lines[:4]))
        if joined:
            returns.append(joined)
    return _dedupe(returns)


def _named_sections(text: str, names: set[str]) -> list[str]:
    lines = text.splitlines()
    sections: list[str] = []
    i = 0
    while i < len(lines):
        name = lines[i].strip().lower().rstrip(":")
        if name in names:
            j = i + 1
            if j < len(lines) and set(lines[j].strip()) in ({"-"}, {"="}):
                j += 1
            start = j
            while j < len(lines):
                candidate = lines[j].strip().lower().rstrip(":")
                if candidate in _SECTION_NAMES:
                    break
                if j + 1 < len(lines) and set(lines[j + 1].strip()) in ({"-"}, {"="}) and candidate:
                    break
                j += 1
            sections.append("\n".join(lines[start:j]).strip())
            i = j
        elif name.startswith("fields/attributes") and "fields/attributes" in names:
            sections.append(lines[i].strip())
            i += 1
        else:
            i += 1
    return [section for section in sections if section]


def _extract_example(text: str, question: str, focus: str, signatures: list[str]) -> str:
    blocks = _code_blocks(text)
    q_terms = set(_tokens(question + " " + focus))
    best = ""
    best_score = -1
    for block in blocks:
        compact = block.strip()
        if not compact or len(compact.splitlines()) > 12 or len(compact) > 900:
            continue
        score = sum(1 for term in q_terms if term in compact.lower())
        if "hugiml" in compact.lower():
            score += 2
        if score > best_score:
            best = compact
            best_score = score
    if best:
        return _normalise_code_block(best)
    return _synthetic_example(question, focus, signatures)


def _code_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    # Markdown fenced blocks.
    for match in re.finditer(r"```(?:python)?\s*\n(.*?)\n```", text, flags=re.S | re.I):
        blocks.append(match.group(1))
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(".. code-block::") or stripped.endswith("::"):
            i += 1
            block: list[str] = []
            while i < len(lines):
                raw = lines[i]
                if not raw.strip() and block:
                    block.append("")
                    i += 1
                    continue
                if raw.startswith(("   ", "\t")):
                    block.append(raw[3:] if raw.startswith("   ") else raw.lstrip("\t"))
                    i += 1
                    continue
                if block:
                    break
                i += 1
            if block:
                blocks.append("\n".join(block).strip())
            continue
        i += 1
    return blocks


def _synthetic_example(question: str, focus: str, signatures: list[str]) -> str:
    low = (question + " " + focus).lower()
    if "predict_proba" in low:
        return "proba = model.predict_proba(X_test)"
    if re.search(r"\bpredict\b", low):
        return "labels = model.predict(X_test)"
    if "feature_importance" in low or "importance" in low:
        return "importance = model.feature_importances()"
    if "pattern" in low and "prun" in low:
        return "from hugiml.pruning import PatternEditor\n\neditor = PatternEditor(model, operator_name=\"reviewer\")\neditor.remove([0, 3], reason=\"Reviewed and rejected by model risk\")\neditor.refit(X_train, y_train)\nmodel = editor.finalize()"
    if "fit" in low or "classifier" in low:
        return "from hugiml import HUGIMLClassifier\n\nmodel = HUGIMLClassifier()\nmodel.fit(X_train, y_train)"
    if signatures:
        call = signatures[0]
        name = call.split("(", 1)[0].split(".")[-1]
        if "." in call.split("(", 1)[0]:
            return f"result = model.{name}(...)"
        return f"result = {name}(...)"
    return "# See the sources below for the exact call used by this topic."


def _normalise_code_block(block: str) -> str:
    lines = [ln.rstrip() for ln in block.strip().splitlines()]
    # Remove doctest prompts while preserving useful code.
    cleaned = []
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith(">>> "):
            cleaned.append(stripped[4:])
        elif stripped.startswith("... "):
            cleaned.append(stripped[4:])
        else:
            cleaned.append(ln)
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(cleaned)


def _extract_caveats(text: str) -> list[str]:
    candidates: list[str] = []
    for para in _extract_explanatory_paragraphs(text):
        low = para.lower()
        if any(word in low for word in _CAVEAT_WORDS):
            candidates.append(_trim_sentence(para, 220))
    for section_name in ("raises", "thread safety", "notes"):
        for section in _named_sections(text, {section_name}):
            clean = _clean_description(" ".join(ln.strip() for ln in section.splitlines() if ln.strip()))
            if clean:
                candidates.append(_trim_sentence(clean, 220))
    return _dedupe(candidates)


def _compose_summary(question: str, focus: str, paragraphs: list[str], hits: list[DocumentationHit]) -> str:
    selected: list[str] = []
    q_terms = set(_tokens(question + " " + focus))
    for para in paragraphs:
        low_tokens = set(_tokens(para))
        if q_terms & low_tokens or len(selected) < 1:
            selected.append(_trim_sentence(para, 260))
        if len(selected) >= 2:
            break
    if selected:
        return " ".join(selected)
    title = _normalise_title(hits[0].title)
    return f"`{title or focus}` is documented in the local HUGIML Sphinx/API materials. The retrieved entries describe how to use it and what it returns."


def _choose_signature(signatures: list[str], question: str, focus: str) -> str:
    if not signatures:
        return ""
    terms = set(_tokens(question + " " + focus))
    best = signatures[0]
    best_score = -1
    for sig in signatures:
        sig_terms = set(_tokens(sig))
        score = len(terms & sig_terms)
        if "__init__" in sig:
            score -= 1
        if score > best_score:
            best = sig
            best_score = score
    return best


def _compact_sources(hits: list[DocumentationHit]) -> list[str]:
    sources: list[str] = []
    for hit in hits:
        title = _normalise_title(hit.title)
        if title and not _title_is_generic(title.lower()):
            sources.append(f"{title} (`{hit.path}`)")
        else:
            sources.append(f"`{hit.path}`")
    return _dedupe(sources)


def _clean_description(text: str) -> str:
    text = _compact_text(text)
    text = re.sub(r"\s*--\s*", " — ", text)
    return _trim_sentence(text, 220)


def _trim_sentence(text: str, max_chars: int) -> str:
    text = _compact_text(text)
    if len(text) <= max_chars:
        return text.rstrip(".") + ("." if text and text[-1] not in ".!?`" else "")
    clipped = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return clipped + "…"


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        clean = _compact_text(str(item))
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _dedupe_pairs(items: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for name, desc in items:
        key = name.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((_compact_text(name), _compact_text(desc)))
    return out


def _iter_files(root: Path, extensions: set[str]) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() in extensions and root.stat().st_size <= _MAX_DOC_BYTES:
            yield root
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        parts = set(path.parts)
        if {".git", ".venv", "venv", "node_modules", "__pycache__"} & parts:
            continue
        try:
            if path.stat().st_size <= _MAX_DOC_BYTES:
                yield path
        except OSError:
            continue



def _html_to_text(text: str) -> str:
    """Convert built Sphinx HTML to compact searchable text without dependencies."""

    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|section|article|h[1-6]|li|tr|pre|code)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    try:
        import html as _html

        text = _html.unescape(text)
    except Exception:
        pass
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _file_signature(path: Path, kind: str) -> dict[str, Any]:
    try:
        stat = path.stat()
        rel = path.as_posix()
        marker = f"{rel}:{int(stat.st_mtime_ns)}:{int(stat.st_size)}:{kind}".encode()
        digest = hashlib.sha1(marker).hexdigest()[:12]
        return {"path": rel, "mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size), "kind": kind, "id": digest}
    except OSError:
        return {"path": path.as_posix(), "missing": True, "kind": kind}


def _split_markdown_like(text: str) -> list[tuple[str, str, int, int]]:
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[tuple[str, list[str], int]] = []
    title = ""
    body: list[str] = []
    start_line = 1
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        is_hash_heading = stripped.startswith("#") and bool(stripped.lstrip("#").strip())
        is_rst_heading = i < len(lines) and set(lines[i].strip()) in ({"="}, {"-"}) and len(lines[i].strip()) >= 3
        if is_hash_heading or is_rst_heading:
            if body:
                chunks.append((title, body, start_line))
            title = stripped.lstrip("#").strip() if is_hash_heading else stripped
            body = [line]
            start_line = i
        else:
            body.append(line)
    if body:
        chunks.append((title, body, start_line))
    if len(chunks) == 1 and not chunks[0][0]:
        return [("", "\n".join(chunks[0][1]), 1, len(lines))]
    return [(title or "section", "\n".join(body), start, start + len(body) - 1) for title, body, start in chunks]


def _split_long_block(text: str, line_start: int | None) -> list[tuple[str, int | None, int | None]]:
    if len(text) <= _MAX_CHUNK_CHARS:
        line_count = len(text.splitlines()) or 1
        line_end = (line_start + line_count - 1) if line_start else None
        return [(text, line_start, line_end)]
    lines = text.splitlines()
    parts: list[tuple[str, int | None, int | None]] = []
    current: list[str] = []
    current_start = line_start
    for offset, line in enumerate(lines):
        if sum(len(x) + 1 for x in current) + len(line) + 1 > _MAX_CHUNK_CHARS and current:
            part_text = "\n".join(current)
            part_start = current_start
            part_end = (part_start + len(current) - 1) if part_start else None
            parts.append((part_text, part_start, part_end))
            current = []
            current_start = (line_start + offset) if line_start else None
        current.append(line)
    if current:
        part_start = current_start
        part_end = (part_start + len(current) - 1) if part_start else None
        parts.append(("\n".join(current), part_start, part_end))
    return parts


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        for piece in raw.split("_"):
            if len(piece) < 2 or piece in _STOP_WORDS:
                continue
            out.append(piece)
            if piece == "pruning":
                out.append("prune")
            elif piece == "tuning":
                out.append("tune")
            elif piece == "scoring":
                out.append("score")
            elif piece == "loading":
                out.append("load")
            elif piece == "explaining":
                out.append("explain")
        if "_" in raw and len(raw) >= 3 and raw not in _STOP_WORDS:
            out.append(raw)
    return out


def _expand_query_terms(terms: list[str]) -> list[str]:
    expanded = list(terms)
    for term in list(terms):
        if term.endswith("s") and len(term) > 3:
            expanded.append(term[:-1])
        if term.endswith("ing") and len(term) > 5:
            expanded.append(term[:-3])
        if term.endswith("ed") and len(term) > 4:
            expanded.append(term[:-2])
        if term in {"api", "apis"}:
            expanded.extend(["class", "function", "method", "usage"])
        elif term in {"parameter", "parameters", "argument", "arguments"}:
            expanded.extend(["param", "params", "argument", "arguments"])
        elif term in {"hyperparameter", "hyperparameters", "tune", "tuning", "grid", "grids"}:
            expanded.extend([
                "hyperparameter", "hyperparameters", "tune", "tuning", "param_grid",
                "default_param_grid", "get_hugiml_grid", "HUGIML_GRIDS",
                "performance", "interpretability", "feature_mode", "topK", "adaptive_binning",
            ])
        elif term in {"prediction", "predict"}:
            expanded.extend(["predict", "predict_proba", "score"])
        elif term in {"pruning", "prune", "pruned"}:
            expanded.extend(["prune", "prune_patterns", "PatternEditor", "remove", "min_support", "pattern_indices"])
        elif term in {"explain", "explanation", "interpret"}:
            expanded.extend(["feature_importances", "get_pattern_info", "model_summary"])
    return expanded


def _path_tokens(path: str) -> set[str]:
    return set(_tokens(path.replace("/", " ").replace(".", " ")))


def _compact_text(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def _snippet(text: str, query_terms: list[str], *, max_chars: int = 520) -> str:
    compact = _compact_text(text)
    if len(compact) <= max_chars:
        return compact
    low = compact.lower()
    positions = [low.find(term) for term in query_terms if term and low.find(term) >= 0]
    if positions:
        center = min(positions)
        start = max(0, center - max_chars // 3)
    else:
        start = 0
    end = min(len(compact), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)
    snippet = compact[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(compact):
        snippet += "…"
    return snippet


def _class_field_summary(node: ast.ClassDef) -> str:
    """Return a compact field/attribute list for dataclass-like API docs."""

    fields: list[str] = []
    for child in node.body:
        targets: list[str] = []
        annotation = ""
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            targets = [child.target.id]
            try:
                annotation = ast.unparse(child.annotation)
            except Exception:
                annotation = ""
        elif isinstance(child, ast.Assign):
            targets = [target.id for target in child.targets if isinstance(target, ast.Name)]
        for target in targets:
            if target.startswith("_"):
                continue
            item = target
            if annotation:
                item += f": {annotation}"
            fields.append(item)
    if not fields:
        return ""
    return "Fields/attributes: " + ", ".join(fields)


def _signature_from_node(node: ast.AST, *, owner: str | None = None) -> str:
    name = getattr(node, "name", "")
    try:
        args = getattr(node, "args", None)
        pieces: list[str] = []
        if args is not None:
            pos = list(getattr(args, "posonlyargs", [])) + list(getattr(args, "args", []))
            defaults = list(getattr(args, "defaults", []))
            default_offset = len(pos) - len(defaults)
            for i, arg in enumerate(pos):
                if owner and i == 0 and arg.arg in {"self", "cls"}:
                    continue
                part = arg.arg
                if i >= default_offset:
                    part += "=..."
                pieces.append(part)
            vararg = getattr(args, "vararg", None)
            if vararg is not None:
                pieces.append("*" + vararg.arg)
            kwonly = list(getattr(args, "kwonlyargs", []))
            if kwonly and vararg is None:
                pieces.append("*")
            kw_defaults = list(getattr(args, "kw_defaults", []))
            for arg, default in zip(kwonly, kw_defaults):
                part = arg.arg
                if default is not None:
                    part += "=..."
                pieces.append(part)
            kwarg = getattr(args, "kwarg", None)
            if kwarg is not None:
                pieces.append("**" + kwarg.arg)
        prefix = f"{owner}." if owner else ""
        return f"{prefix}{name}({', '.join(pieces)})"
    except Exception:
        prefix = f"{owner}." if owner else ""
        return f"{prefix}{name}(...)"

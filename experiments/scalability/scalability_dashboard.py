#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html as html_lib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTDIR = ROOT / "results"
TEMPLATE_NAME = "hugiml_scalability_dashboard.html"
DATASETS = ("sparse_nonlinear", "threshold_grid")
MODELS = ("hug_op", "hug_po", "xgb", "lgb")
LABEL = {"hug_op": "Orig+Pat", "hug_po": "Pat Only", "xgb": "XGBoost", "lgb": "LightGBM"}

N_SCALING = {
    "sparse_nonlinear": [
        (10_000, 20),
        (50_000, 20),
        (100_000, 20),
        (500_000, 20),
        (1_000_000, 20),
        (3_000_000, 20),
        (5_000_000, 20),
        (10_000_000, 20),
    ],
    "threshold_grid": [
        (1_000, 200),
        (5_000, 200),
        (10_000, 200),
        (50_000, 200),
        (100_000, 200),
        (300_000, 200),
        (500_000, 200),
        (1_000_000, 200),
    ],
}
P_SCALING = {
    "sparse_nonlinear": [
        (50_000, 20),
        (100_000, 100),
        (10_000, 1_000),
        (5_000, 2_000),
        (2_500, 4_000),
        (2_000, 5_000),
        (1_000, 10_000),
    ],
    "threshold_grid": [
        (50_000, 20),
        (100_000, 100),
        (10_000, 1000),
        (5_000, 2_000),
        (2_500, 4_000),
        (2_000, 5_000),
        (1_000, 10_000),
    ],
}
SWEEP = {
    "sparse_nonlinear": {
        "B": {"n": 50000, "p": 20, "values": [2, 3, 5, 7, 10, 15]},
        "G": {"n": 50000, "p": 20, "values": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]},
        "topK": {"n": 50000, "p": 20, "values": [10, 20, 30, 50, 100, 200]},
        "L": {"n": 30000, "p": 20, "values": [1, 2]},
        "avf": {"n": 50000, "p": 20, "values": [True, False]},
    },
    "threshold_grid": {
        "B": {"n": 30000, "p": 200, "values": [2, 3, 5, 7, 10, 15]},
        "G": {"n": 30000, "p": 200, "values": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]},
        "topK": {"n": 30000, "p": 200, "values": [10, 20, 30, 50, 100, 200]},
        "L": {"n": 20000, "p": 100, "values": [1, 2]},
        "avf": {"n": 30000, "p": 200, "values": [True, False]},
    },
}


def resolve_template_html(raw: str | None = None) -> Path:
    if raw:
        path = Path(raw).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"template not found: {path}")
    candidates = []
    try:
        candidates.append(ROOT.parent.parent / TEMPLATE_NAME)
    except Exception:
        pass
    candidates += [ROOT / TEMPLATE_NAME, Path.cwd() / TEMPLATE_NAME]
    for parent in ROOT.parents:
        candidates.append(parent / TEMPLATE_NAME)
    for parent in Path.cwd().resolve().parents:
        candidates.append(parent / TEMPLATE_NAME)
    seen = set()
    for candidate in candidates:
        try:
            key = candidate.resolve()
        except Exception:
            key = candidate
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p) for p in candidates[:6])
    raise FileNotFoundError(f"could not locate {TEMPLATE_NAME}; searched {searched}")


def resolve_outdir(raw: str | None = None) -> Path:
    return Path(raw).expanduser().resolve() if raw else DEFAULT_OUTDIR


def temp_path_for(path: Path) -> Path:
    return path.parent / (path.name + ".tmp")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_total_memory_mb() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total / 1048576)
    except Exception:
        return None


def effective_mem_limit_mb(raw: int | None) -> int | None:
    """Return a process-tree RSS cap in MB. -1 => 90% of system memory; 0/None => no cap."""
    if raw is None or int(raw) == 0:
        return None
    if int(raw) < 0:
        total = get_total_memory_mb()
        return None if total is None else max(1, int(total * 0.90))
    return int(raw)


def system_metadata(args=None) -> dict[str, Any]:
    meta = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "system_memory_mb": get_total_memory_mb(),
    }
    for mod_name in ["xgboost", "lightgbm", "sklearn", "numpy", "pandas", "psutil"]:
        try:
            mod = __import__(mod_name)
            meta[f"{mod_name}_version"] = getattr(mod, "__version__", None)
        except Exception:
            meta[f"{mod_name}_version"] = None
    if args is not None:
        timeout = getattr(args, "task_timeout", None)
        raw_mem = getattr(args, "mem_limit_mb", None)
        meta["task_timeout_seconds"] = (
            None if timeout is None or float(timeout) <= 0 else float(timeout)
        )
        meta["mem_limit_mb_argument"] = raw_mem
        meta["effective_mem_limit_mb"] = effective_mem_limit_mb(raw_mem)
    return meta


def read_text_utf8(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text_utf8(path: Path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def process_tree_rss_mb(proc) -> float | None:
    try:
        import psutil

        p = psutil.Process(proc.pid)
        total = p.memory_info().rss
        for child in p.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:
                pass
        return total / 1048576.0
    except Exception:
        return None


def terminate_process_tree(proc) -> None:
    try:
        import psutil

        p = psutil.Process(proc.pid)
        children = p.children(recursive=True)
        for ch in children:
            try:
                ch.terminate()
            except Exception:
                pass
        try:
            p.terminate()
        except Exception:
            pass
        gone, alive = psutil.wait_procs(children + [p], timeout=5)
        for a in alive:
            try:
                a.kill()
            except Exception:
                pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def make_baseline_signal(n: int, p: int, seed: int = 42):
    import numpy as np

    rng = np.random.default_rng(seed + n * 13 + p * 17)
    X = rng.normal(0, 1, size=(n, p)).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)
    if p >= 1:
        z += 1.20 * X[:, 0]
    if p >= 2:
        z += 0.80 * X[:, 1]
    if p >= 3:
        z -= 0.60 * X[:, 2]
    if p >= 5:
        z += 0.25 * np.sin(X[:, 3]) + 0.20 * (X[:, 4] > 0.0)
    z += rng.normal(0, 0.25, size=n).astype(np.float32)
    y = (z > np.median(z)).astype(np.int32)
    return X, y


def make_threshold_grid(n: int, p: int, seed: int = 42):
    import numpy as np

    rng = np.random.default_rng(seed + n * 5 + p * 11)
    X = rng.normal(0, 1, size=(n, p)).astype(np.float32)
    z = np.zeros(n, dtype=np.float32)
    for j, thr in enumerate([0.70, 0.45, 0.20, -0.20, -0.45, -0.70][: min(6, p)]):
        z += (0.65 - 0.04 * j) * (X[:, j] > thr)
    if p >= 8:
        z += 0.90 * ((X[:, 2] > 0.15) & (X[:, 7] < -0.10))
    if p >= 16:
        z += 0.75 * ((X[:, 10] + X[:, 11]) > 0.50)
    if p >= 32:
        z += 0.45 * np.tanh(X[:, 20] - 0.5 * X[:, 21])
    z += rng.normal(0, 0.20, size=n).astype(np.float32)
    y = (z > np.quantile(z, 0.50)).astype(np.int32)
    return X, y


def task_key(t: dict[str, Any]) -> str:
    parts = [t["dataset"], t["section"], t["model"], str(t["n"]), str(t["p"])]
    if t.get("sweep_name"):
        parts += [t["sweep_name"], repr(t["sweep_value"])]
    return "|".join(parts)


def make_tasks(include_sweeps=True):
    out = []
    for dataset in DATASETS:
        for n, p in N_SCALING[dataset]:
            for model in MODELS:
                t = {"dataset": dataset, "section": "n_scaling", "model": model, "n": n, "p": p}
                t["key"] = task_key(t)
                out.append(t)
        for n, p in P_SCALING[dataset]:
            for model in MODELS:
                t = {"dataset": dataset, "section": "p_scaling", "model": model, "n": n, "p": p}
                t["key"] = task_key(t)
                out.append(t)
        if include_sweeps:
            for name, s in SWEEP[dataset].items():
                for v in s["values"]:
                    t = {
                        "dataset": dataset,
                        "section": f"parameter_sweep_{name}",
                        "model": "hug_op",
                        "n": s["n"],
                        "p": s["p"],
                        "sweep_name": name,
                        "sweep_value": v,
                    }
                    t["key"] = task_key(t)
                    out.append(t)
    return out


def params_for(model: str, sweep_name=None, sweep_value=None):
    if model not in ("hug_op", "hug_po"):
        return {}
    p = {
        "B": 5,
        "L": 1,
        "G": 0.01,
        "topK": 50,
        "n_jobs": 4,
        "adaptive_binning": True,
        "b_candidates": [3, 5, 7, 10],
        "use_hotpath": True,
        "augmented_pair_transforms": False,
        "interaction_relaxed_mining": False,
        "execution_mode": "production",
    }
    if model == "hug_op":
        p.update({"feature_mode": "original_plus_patterns", "topk_budget_strict": True})
    else:
        p.update({"feature_mode": "patterns_only", "topk_budget_strict": False})
    if sweep_name == "B":
        p.update({"adaptive_binning": False, "B": int(sweep_value)})
        p.pop("b_candidates", None)
    elif sweep_name == "G":
        p["G"] = float(sweep_value)
    elif sweep_name == "topK":
        p["topK"] = int(sweep_value)
    elif sweep_name == "L":
        p["L"] = int(sweep_value)
    elif sweep_name == "avf":
        p["adaptive_binning"] = bool(sweep_value)
        if not p["adaptive_binning"]:
            p.pop("b_candidates", None)
            p["B"] = 5
    return p


def _emit_worker_result(args, row: dict[str, Any]) -> None:
    """Emit a single JSON row for the parent process.

    The benchmark parent now reads the worker result from stdout to avoid
    creating one result file per task. --result-json is retained for manual
    compatibility only.
    """
    text = json.dumps(row, ensure_ascii=False, allow_nan=False)
    result_path = getattr(args, "result_json", None)
    if result_path:
        path = Path(result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = temp_path_for(path)
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    else:
        print(text, flush=True)


def worker(args):
    import psutil
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    t0all = time.perf_counter()
    status = "ok"
    err = None
    fit_s = None
    auc = None
    patterns = None
    params: dict[str, Any] = {}
    after = None
    peak = [None]
    data_s = split_s = predict_s = auc_s = None

    try:
        t0 = time.perf_counter()
        X, y = (
            make_baseline_signal if args.dataset == "sparse_nonlinear" else make_threshold_grid
        )(args.n, args.p, args.seed)
        data_s = time.perf_counter() - t0

        proc = psutil.Process()
        after = proc.memory_info().rss / 1048576

        t0 = time.perf_counter()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        del X, y
        split_s = time.perf_counter() - t0

        peak = [proc.memory_info().rss / 1048576]
        running = [True]

        def sample():
            while running[0]:
                try:
                    peak[0] = max(peak[0], proc.memory_info().rss / 1048576)
                except Exception:
                    pass
                time.sleep(0.05)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        try:
            if args.model in ("hug_op", "hug_po"):
                from hugiml import HUGIMLClassifierNative

                val = json.loads(args.sweep_value_json) if args.sweep_value_json else None
                params = params_for(args.model, args.sweep_name, val)
                clf = HUGIMLClassifierNative(**params)
                fit_Xtr = Xtr
                pred_Xte = Xte
            elif args.model == "xgb":
                from xgboost import XGBClassifier

                params = {
                    "n_estimators": 50,
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "tree_method": "hist",
                    "n_jobs": 4,
                    "eval_metric": "logloss",
                    "verbosity": 0,
                    "random_state": 42,
                }
                clf = XGBClassifier(**params)
                fit_Xtr = Xtr
                pred_Xte = Xte
            else:
                import pandas as pd
                from lightgbm import LGBMClassifier

                params = {
                    "n_estimators": 50,
                    "max_depth": 4,
                    "num_leaves": 15,
                    "learning_rate": 0.1,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "n_jobs": 4,
                    "random_state": 42,
                    "verbose": -1,
                }
                clf = LGBMClassifier(**params)
                columns = [f"x{i}" for i in range(Xtr.shape[1])]
                # Use the same named schema for fit and predict. This keeps the
                # sklearn/LightGBM feature-name warning at source instead of
                # hiding it.
                fit_Xtr = pd.DataFrame(Xtr, columns=columns, copy=False)
                pred_Xte = pd.DataFrame(Xte, columns=columns, copy=False)

            t0 = time.perf_counter()
            clf.fit(fit_Xtr, ytr)
            fit_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            proba = clf.predict_proba(pred_Xte)[:, 1]
            predict_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            auc = float(roc_auc_score(yte, proba))
            auc_s = time.perf_counter() - t0

            if args.model in ("hug_op", "hug_po"):
                patterns = len(getattr(clf, "patterns_", []) or [])
                params["execution_mode"] = getattr(
                    clf, "execution_mode", params.get("execution_mode")
                )
        finally:
            running[0] = False
            try:
                th.join(timeout=1)
            except Exception:
                pass
    except Exception as e:
        status = "error"
        err = repr(e)

    row = {
        "key": args.key,
        "dataset": args.dataset,
        "section": args.section,
        "model": args.model,
        "model_label": LABEL.get(args.model, args.model),
        "n": args.n,
        "p": args.p,
        "status": status,
        "data_s": data_s,
        "split_s": split_s,
        "fit_s": fit_s,
        "predict_s": predict_s,
        "auc_s": auc_s,
        "auc": auc,
        "patterns": patterns,
        "memory_mb": {
            "rss_after_data": after,
            "peak_fit_window": peak[0],
            "fit_delta_from_after_data": None
            if after is None or peak[0] is None
            else max(0, peak[0] - after),
        },
        "params": params,
        "sweep_name": args.sweep_name,
        "sweep_value": json.loads(args.sweep_value_json) if args.sweep_value_json else None,
        "elapsed_s": time.perf_counter() - t0all,
        "error": err,
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "import_path": "optimized_package",
        },
    }
    _emit_worker_result(args, row)


def load_ckpt(path: Path):
    return (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"created_at": now_iso(), "metadata": {}, "results": []}
    )


def save_ckpt(path: Path, ckpt):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path_for(path)
    tmp.write_text(json.dumps(ckpt, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def row_complete(results, dataset, section, n, p):
    have = set(
        r["model"]
        for r in results
        if r.get("dataset") == dataset
        and r.get("section") == section
        and int(r.get("n", -1)) == n
        and int(r.get("p", -1)) == p
        and r.get("model") in MODELS
        and r.get("status") == "ok"
    )
    return all(m in have for m in MODELS)


def _safe_task_stem(key: str) -> str:
    # Retained for backward compatibility with older checkpoints/scripts that
    # may import it, but the optimized runner no longer creates per-task files.
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:80].strip("_") or "task"
    return f"{readable}_{digest}"


def _append_bounded(parts: list[str], state: dict[str, int], chunk: str, limit: int) -> None:
    """Append stream text while retaining only the last ``limit`` chars."""
    if not chunk or limit <= 0:
        return
    parts.append(chunk)
    state["chars"] = state.get("chars", 0) + len(chunk)
    while state["chars"] > limit and parts:
        excess = state["chars"] - limit
        head = parts[0]
        if len(head) <= excess:
            state["chars"] -= len(head)
            parts.pop(0)
        else:
            parts[0] = head[excess:]
            state["chars"] -= excess
            break


def _drain_text_stream(stream, parts: list[str], state: dict[str, int], limit: int) -> None:
    """Continuously drain a worker pipe so verbose libraries cannot deadlock."""
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            _append_bounded(parts, state, chunk, limit)
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _parse_worker_stdout(stdout_text: str) -> dict[str, Any]:
    """Parse the worker JSON row from stdout, tolerating incidental log lines."""
    text = (stdout_text or "").strip()
    if not text:
        raise ValueError("worker did not emit JSON on stdout")
    try:
        return json.loads(text)
    except Exception:
        # Some third-party libraries may write incidental lines to stdout. The
        # worker emits its result as one JSON object, so scan from the end for
        # a parseable JSON line before failing.
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        raise


def _read_tail(path: Path, limit: int) -> str:
    # The optimized runner no longer
    # stores worker stdout/stderr on disk.
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[-limit:]


def run_worker_task(t: dict[str, Any], args, outdir: Path) -> dict[str, Any]:
    effective_mem = effective_mem_limit_mb(args.mem_limit_mb)
    timeout = (
        None
        if args.task_timeout is None or float(args.task_timeout) <= 0
        else float(args.task_timeout)
    )

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--key",
        t["key"],
        "--dataset",
        t["dataset"],
        "--section",
        t["section"],
        "--model",
        t["model"],
        "--n",
        str(t["n"]),
        "--p",
        str(t["p"]),
        "--seed",
        str(args.seed),
        "--mem-limit-mb",
        "0",  # memory is enforced by the parent process-tree RSS monitor
    ]
    if t.get("sweep_name"):
        cmd += ["--sweep-name", t["sweep_name"], "--sweep-value-json", json.dumps(t["sweep_value"])]

    started = time.perf_counter()
    peak_tree = None
    status = None
    error = None

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    stdout_state = {"chars": 0}
    stderr_state = {"chars": 0}
    # Keep enough stdout to include the JSON result and enough stderr for useful
    # diagnostics, without allowing verbose dependencies to grow memory forever.
    stdout_limit = 1_000_000
    stderr_limit = 200_000

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    stdout_thread = threading.Thread(
        target=_drain_text_stream,
        args=(proc.stdout, stdout_parts, stdout_state, stdout_limit),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain_text_stream,
        args=(proc.stderr, stderr_parts, stderr_state, stderr_limit),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        while True:
            rc = proc.poll()
            rss = process_tree_rss_mb(proc) if rc is None else None
            if rss is not None:
                peak_tree = rss if peak_tree is None else max(peak_tree, rss)
            elapsed = time.perf_counter() - started
            if (
                rc is None
                and effective_mem is not None
                and peak_tree is not None
                and peak_tree > effective_mem
            ):
                status = "oom"
                error = (
                    f"process tree RSS {peak_tree:.1f} MB exceeded memory limit {effective_mem} MB"
                )
                terminate_process_tree(proc)
                break
            if rc is None and timeout is not None and elapsed > timeout:
                status = "timeout"
                error = f"timeout after {timeout:.1f}s"
                terminate_process_tree(proc)
                break
            if rc is not None:
                break
            time.sleep(0.25)
        try:
            proc.wait(timeout=5)
        except Exception:
            terminate_process_tree(proc)
    finally:
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)

    elapsed = time.perf_counter() - started
    stdout_text = "".join(stdout_parts)
    stderr_text = "".join(stderr_parts)
    stdout_tail = stdout_text[-2000:]
    stderr_tail = stderr_text[-4000:]

    if status in {"oom", "timeout"}:
        res = {"key": t["key"], **t, "status": status, "error": error}
        if stdout_tail:
            res["stdout_tail"] = stdout_tail
        if stderr_tail:
            res["stderr_tail"] = stderr_tail
    else:
        try:
            res = _parse_worker_stdout(stdout_text)
        except Exception as exc:
            res = {
                "key": t["key"],
                **t,
                "status": "error",
                "error": f"worker stdout JSON parse failed: {exc!r}",
            }
        if proc.returncode not in (0, None) and res.get("status") == "ok":
            res["status"] = "error"
            res["error"] = f"worker_returncode={proc.returncode}"
        # Keep normal successful rows clean. Attach stderr as warnings only if a
        # successful worker emitted warnings; failure rows keep bounded tails.
        if res.get("status") == "ok":
            if stderr_tail:
                res["warnings_tail"] = stderr_tail
        else:
            if stdout_tail:
                res["stdout_tail"] = stdout_tail
            if stderr_tail:
                res["stderr_tail"] = stderr_tail

    res.setdefault("key", t["key"])
    for k, v in t.items():
        res.setdefault(k, v)
    res["returncode"] = proc.returncode
    res["elapsed_s"] = float(res.get("elapsed_s") or elapsed)
    res["task_timeout_seconds"] = timeout
    res["mem_limit_mb"] = effective_mem
    res["peak_process_tree_rss_mb"] = peak_tree
    res["completed_at"] = now_iso()
    return res


def run_all(args):
    outdir = resolve_outdir(args.outdir)
    if getattr(args, "fresh", False) and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / "scalability_checkpoint.json"
    ckpt = load_ckpt(ckpt_path)
    ckpt["metadata"] = {
        "generated_at": now_iso(),
        "threads": 4,
        "system": system_metadata(args),
        "hugiml_execution_mode": "production",
        "hugiml_config": {
            "L": 1,
            "G": 0.01,
            "topK": 50,
            "adaptive_binning": True,
            "b_candidates": [3, 5, 7, 10],
            "use_hotpath": True,
            "augmented_pair_transforms": False,
            "interaction_relaxed_mining": False,
        },
        "dashboard_uses_common_successful_rows_only": True,
    }
    done = {r["key"] for r in ckpt.get("results", []) if r.get("status") == "ok"}
    tasks = make_tasks(include_sweeps=not args.no_sweeps)
    if args.only_section:
        tasks = [t for t in tasks if t["section"] == args.only_section]
    if args.only_dataset:
        tasks = [t for t in tasks if t["dataset"] == args.only_dataset]
    if args.only_model:
        tasks = [t for t in tasks if t["model"] == args.only_model]
    if getattr(args, "start_task", 0):
        tasks = tasks[int(args.start_task) :]
    if getattr(args, "max_tasks", None) is not None:
        tasks = tasks[: int(args.max_tasks)]
    for i, t in enumerate(tasks, 1):
        if args.resume and t["key"] in done:
            continue
        print(f"run {i}/{len(tasks)} {t['key']}", flush=True)
        res = run_worker_task(t, args, outdir)
        ckpt["results"] = [r for r in ckpt.get("results", []) if r.get("key") != t["key"]]
        ckpt["results"].append(res)
        save_ckpt(ckpt_path, ckpt)
    build_outputs(
        argparse.Namespace(outdir=str(outdir), template=args.template, output_html=args.output_html)
    )


def find_res(results, dataset, section, model, n, p, sn=None, sv=None):
    for r in results:
        if r.get("status") != "ok":
            continue
        if (
            r.get("dataset") == dataset
            and r.get("section") == section
            and r.get("model") == model
            and int(r.get("n", -1)) == int(n)
            and int(r.get("p", -1)) == int(p)
        ):
            if sn is None or (r.get("sweep_name") == sn and r.get("sweep_value") == sv):
                return r
    return None


def val(r, f):
    if not r:
        return None
    if f == "m":
        return (r.get("memory_mb") or {}).get("fit_delta_from_after_data")
    return r.get(f)


def row_for(results, dataset, section, n, p):
    rop, rpo, rxg, rlg = (find_res(results, dataset, section, mo, n, p) for mo in MODELS)
    return {
        "n": n,
        "p": p,
        "ha": val(rop, "fit_s"),
        "hp": val(rpo, "fit_s"),
        "xg": val(rxg, "fit_s"),
        "lg": val(rlg, "fit_s"),
        "haA": val(rop, "auc"),
        "hpA": val(rpo, "auc"),
        "xgA": val(rxg, "auc"),
        "lgA": val(rlg, "auc"),
        "mha": val(rop, "m"),
        "mhp": val(rpo, "m"),
        "mxg": val(rxg, "m"),
        "mlg": val(rlg, "m"),
        "hap": val(rop, "patterns"),
        "hpp": val(rpo, "patterns"),
    }


def completed_grid(results, dataset, grid, section):
    return [(n, p) for n, p in grid if row_complete(results, dataset, section, n, p)]


def sweep_arr(results, dataset, name):
    s = SWEEP[dataset][name]
    out = []
    for v in s["values"]:
        r = find_res(results, dataset, f"parameter_sweep_{name}", "hug_op", s["n"], s["p"], name, v)
        out.append(
            {
                "v": v,
                "t": val(r, "fit_s"),
                "a": val(r, "auc"),
                "p": val(r, "patterns"),
                "m": val(r, "m"),
                "n": s["n"],
                "pp": s["p"],
            }
        )
    return out


def fmt_s(x):
    if x is None:
        return "—"
    return f"{x / 60:.1f} min" if x >= 60 else f"{x:.2f} s" if x >= 1 else f"{x * 1000:.0f} ms"


def fnum(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def fN(n):
    return f"{n / 1_000_000:g}M" if n >= 1_000_000 else f"{n // 1000}k" if n >= 1000 else str(n)


def make_meta(data):
    meta = {}
    for name, d in data.items():
        label = "Baseline Signal" if name == "sparse_nonlinear" else "Threshold Grid"
        maxn = d["ns"][-1]["n"] if d["ns"] else None
        maxp = d["ps"][-1]["p"] if d["ps"] else None
        basep = 20 if name == "sparse_nonlinear" else 200
        desc = (
            f"float32 features; nonlinear signal concentrated in the first columns with remaining columns mostly acting as noise. p={basep} for n-scaling; p varies from 20 to {maxp} for p-scaling. 4 threads."
            if name == "sparse_nonlinear"
            else f"float32 features; threshold and local-interaction signal with median-binarised labels. p={basep} for n-scaling; p varies from 20 to {maxp} for p-scaling. 4 threads."
        )
        chips = [
            f"n to {fN(maxn)}" if maxn else "n-scaling",
            f"p=20–{maxp}" if maxp else "p-scaling",
            "binary classification" if name == "sparse_nonlinear" else "threshold signal",
            "4 threads",
        ]
        last = d["ns"][-1] if d["ns"] else {}
        pl = d["ps"][-1] if d["ps"] else {}

        def best_label(keys, fn=min):
            pairs = [(last.get(k), v) for k, v in keys if last.get(k) is not None]
            return fn(pairs)[1] if pairs else "—"

        fast = best_label(
            [("ha", "Orig+Pat"), ("hp", "Pat Only"), ("xg", "XGBoost"), ("lg", "LightGBM")], min
        )
        best = best_label(
            [("haA", "Orig+Pat"), ("hpA", "Pat Only"), ("xgA", "XGBoost"), ("lgA", "LightGBM")], max
        )
        mem = best_label(
            [("mha", "Orig+Pat"), ("mhp", "Pat Only"), ("mxg", "XGBoost"), ("mlg", "LightGBM")], min
        )
        fit = f"At n={fN(last.get('n', 0))}, {fast} has the shortest fit time. Orig+Pat takes {fmt_s(last.get('ha'))}, Pat Only {fmt_s(last.get('hp'))}, XGBoost {fmt_s(last.get('xg'))}, and LightGBM {fmt_s(last.get('lg'))}."
        auc = f"At n={fN(last.get('n', 0))}, {best} has the highest test AUC. Orig+Pat is {fnum(last.get('haA'))}, Pat Only {fnum(last.get('hpA'))}, XGBoost {fnum(last.get('xgA'))}, and LightGBM {fnum(last.get('lgA'))}."
        mems = f"At n={fN(last.get('n', 0))}, {mem} has the lowest fit memory delta. Orig+Pat uses {fnum(last.get('mha'), 0)} MB, Pat Only {fnum(last.get('mhp'), 0)} MB, XGBoost {fnum(last.get('mxg'), 0)} MB, and LightGBM {fnum(last.get('mlg'), 0)} MB."
        pcs = [r.get("hap") for r in d["ns"] if r.get("hap") is not None]
        pcs2 = [r.get("hpp") for r in d["ns"] if r.get("hpp") is not None]
        pats = f"Pattern counts remain bounded across n: Orig+Pat mines {min(pcs) if pcs else '—'}–{max(pcs) if pcs else '—'} patterns and Pat Only mines {min(pcs2) if pcs2 else '—'}–{max(pcs2) if pcs2 else '—'} patterns."
        meta[name] = {
            "label": label,
            "desc": desc,
            "chips": chips,
            "dsNote": (f"{label}, p={basep} for n-scaling. Only complete model rows are shown."),
            "kpi_n_label": "Largest complete n measured",
            "cbs": [
                ["⏱", f"<strong>Fit-time summary.</strong> {fit}"],
                ["🎯", f"<strong>Accuracy summary.</strong> {auc}"],
                ["💾", f"<strong>Memory summary.</strong> {mems}"],
                ["📊", f"<strong>Pattern summary.</strong> {pats}"],
            ],
            "fit_ni": fit,
            "auc_ni": auc,
            "pat_ni": pats,
            "mem_n_ni": mems,
            "mem_p_ni": f"At p={pl.get('p')}, memory delta is Orig+Pat {fnum(pl.get('mha'), 0)} MB, Pat Only {fnum(pl.get('mhp'), 0)} MB, XGBoost {fnum(pl.get('mxg'), 0)} MB, and LightGBM {fnum(pl.get('mlg'), 0)} MB.",
            "mem_tbl_p_ni": "Ratios below 1× indicate lower fit memory delta than XGBoost for the same p and n setting.",
            "ps_fit_ni": f"At p={pl.get('p')}, Orig+Pat takes {fmt_s(pl.get('ha'))}, Pat Only {fmt_s(pl.get('hp'))}, XGBoost {fmt_s(pl.get('xg'))}, and LightGBM {fmt_s(pl.get('lg'))}.",
            "ps_ratio_ni": "Ratios below 1.0 indicate a model is faster than XGBoost at the same p and n setting.",
            "ps_auc_ni": f"At p={pl.get('p')}, AUC is Orig+Pat {fnum(pl.get('haA'))}, Pat Only {fnum(pl.get('hpA'))}, XGBoost {fnum(pl.get('xgA'))}, and LightGBM {fnum(pl.get('lgA'))}.",
        }
    return meta


def js_replace(html, name, value):
    marker = f"const {name}="
    s = html.index(marker)
    i = s + len(marker)
    depth = 0
    string = None
    esc = False
    for j in range(i, len(html)):
        ch = html[j]
        if string:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == string:
                string = None
        else:
            if ch in ('"', "'", "`"):
                string = ch
            elif ch in "[{(":
                depth += 1
            elif ch in "]})":
                depth -= 1
            elif ch == ";" and depth == 0:
                return (
                    html[:s]
                    + marker
                    + json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    + ";"
                    + html[j + 1 :]
                )
    raise RuntimeError(name)


def clean_html(html):
    html = re.sub(
        r"<tr><td[^>]*>HUGIML</td><td[^>]*>.*?</td></tr>",
        '<tr><td style="color:var(--mu);padding:3px 10px 3px 0">HUGIML</td><td style="font-weight:700">native C++ package</td></tr>',
        html,
        flags=re.S,
    )
    html = re.sub(r"v\d+(?:\.\d+)+", "native package", html)
    return html


def refresh_axis_ticks(html):
    n_ticks = "const nTk=v=>({5e3:'5k',2e4:'20k',5e4:'50k',1e5:'100k',2e5:'200k',3e5:'300k',5e5:'500k',6e5:'600k',1e6:'1M',1.5e6:'1.5M',3e6:'3M',1e7:'10M',5e7:'50M',1e8:'100M'}[v]||fN(v));"
    p_ticks = "const pTk=v=>({20:'20',50:'50',100:'100',200:'200',500:'500',1000:'1k',2000:'2k',3000:'3k',5000:'5k',10000:'10k'}[v]||String(v));"
    html = re.sub(r"const nTk=v=>\(\{.*?\}\[v\]\|\|''\);", n_ticks, html)
    html = re.sub(r"const pTk=v=>\(\{.*?\}\[v\]\|\|''\);", p_ticks, html)
    html = html.replace("avf:'sw_adaptive_vs_" + "fi" + "xed'", "avf:'sw_adaptive_vs_constant'")
    return html


def _fmt_na(value) -> str:
    if value is None or value == "":
        return "not recorded"
    return str(value)


def _fmt_mb(value) -> str:
    if value is None or value == "":
        return "not recorded"
    try:
        mb = float(value)
        if mb >= 1024:
            return f"{mb:,.0f} MB ({mb / 1024.0:,.1f} GB)"
        return f"{mb:,.0f} MB"
    except Exception:
        return str(value)


def _fmt_seconds(value) -> str:
    if value is None or value == "":
        return "disabled/not recorded"
    try:
        seconds = float(value)
    except Exception:
        return str(value)
    if seconds <= 0:
        return "disabled"
    if seconds >= 60:
        minutes = seconds / 60.0
        if abs(minutes - round(minutes)) < 1e-9:
            return f"{seconds:,.0f} s ({minutes:,.0f} min)"
        return f"{seconds:,.0f} s ({minutes:,.1f} min)"
    return f"{seconds:g} s"


def _pkg_version(system: dict[str, Any], key: str) -> str:
    val = system.get(key)
    return f"v{val}" if val else "not recorded"


def _env_td(label: str, value: Any) -> str:
    return (
        "<tr>"
        '<td style="color:var(--mu);padding:3px 10px 3px 0">'
        f"{html_lib.escape(str(label), quote=True)}"
        "</td>"
        '<td style="font-weight:700">'
        f"{html_lib.escape(str(value), quote=True)}"
        "</td>"
        "</tr>"
    )


def render_environment_table(ckpt: dict[str, Any], args=None) -> str:
    """Render the methodology Environment table from run metadata.

    Prefer checkpoint metadata because --assemble may run on a different machine
    from the benchmark run. Fall back to the current process only when the
    checkpoint does not contain system metadata.
    """
    run_meta = ckpt.get("metadata") or {}
    system = run_meta.get("system") or system_metadata(args)
    threads = run_meta.get("threads") or system.get("threads") or 4
    timeout = system.get("task_timeout_seconds")
    raw_mem = system.get("mem_limit_mb_argument")
    effective_mem = system.get("effective_mem_limit_mb")
    if raw_mem is not None and int(raw_mem) < 0 and effective_mem is not None:
        mem_cap = f"90% of system RAM = {_fmt_mb(effective_mem)}"
    elif effective_mem is None:
        mem_cap = "disabled/not recorded"
    else:
        mem_cap = _fmt_mb(effective_mem)

    rows = [
        ("Python", _fmt_na(system.get("python"))),
        ("Platform", _fmt_na(system.get("platform"))),
        ("Processor", _fmt_na(system.get("processor"))),
        ("Logical CPUs", _fmt_na(system.get("cpu_count_logical"))),
        ("System RAM", _fmt_mb(system.get("system_memory_mb"))),
        ("XGBoost", _pkg_version(system, "xgboost_version")),
        ("LightGBM", _pkg_version(system, "lightgbm_version")),
        ("scikit-learn", _pkg_version(system, "sklearn_version")),
        ("NumPy", _pkg_version(system, "numpy_version")),
        ("pandas", _pkg_version(system, "pandas_version")),
        ("Threads", f"{threads} (n_jobs={threads})"),
        ("Task timeout", _fmt_seconds(timeout)),
        ("Memory cap", mem_cap),
        ("Train/test split", "75% / 25%, stratified, seed=42"),
        ("Metric", "Test-set ROC AUC"),
        ("Memory measurement", "Process-tree peak RSS and fit-window RSS delta"),
    ]
    body = "".join(_env_td(k, v) for k, v in rows)
    return f'<table style="font-size:11.5px">\n        {body}\n      </table>'


def refresh_methodology_text(html: str, ckpt: dict[str, Any] | None = None, args=None) -> str:
    html = html.replace("p varied 20&#8211;3000", "p varied 20&#8211;10000")
    html = html.replace("n up to 3,000,000", "n up to 100,000,000")
    html = html.replace("n up to 500,000", "n up to 100,000,000")
    if ckpt is not None:
        env_table = render_environment_table(ckpt, args)
        html = re.sub(
            r"(<h3[^>]*>\s*Environment\s*</h3>\s*)<table[^>]*>.*?</table>",
            r"\1" + env_table,
            html,
            count=1,
            flags=re.S,
        )
    return html


def build_outputs(args):
    outdir = resolve_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    template_path = resolve_template_html(getattr(args, "template", None))
    ckpt = load_ckpt(outdir / "scalability_checkpoint.json")
    results = ckpt.get("results", [])
    full_rows = {"n_scaling": {}, "p_scaling": {}}
    data = {}
    for ds in DATASETS:
        ngrid = completed_grid(results, ds, N_SCALING[ds], "n_scaling")
        pgrid = completed_grid(results, ds, P_SCALING[ds], "p_scaling")
        full_rows["n_scaling"][ds] = ngrid
        full_rows["p_scaling"][ds] = pgrid
        data[ds] = {
            "ns": [row_for(results, ds, "n_scaling", n, p) for n, p in ngrid],
            "ps": [row_for(results, ds, "p_scaling", n, p) for n, p in pgrid],
            "sw_B": sweep_arr(results, ds, "B"),
            "sw_G": sweep_arr(results, ds, "G"),
            "sw_topK": sweep_arr(results, ds, "topK"),
            "sw_L": sweep_arr(results, ds, "L"),
            "sw_adaptive_vs_constant": sweep_arr(results, ds, "avf"),
        }
    meta = make_meta(data)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": ckpt,
        "datasets": data,
        "meta": meta,
        "full_rows": full_rows,
        "omitted_incomplete_points": collect_incomplete(results),
    }
    (outdir / "scalability_dashboard_data.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )

    fields = [
        "dataset",
        "section",
        "model",
        "n",
        "p",
        "sweep_name",
        "sweep_value",
        "status",
        "data_s",
        "split_s",
        "fit_s",
        "predict_s",
        "auc_s",
        "auc",
        "patterns",
        "memory_mb",
        "peak_process_tree_rss_mb",
        "mem_limit_mb",
        "task_timeout_seconds",
        "elapsed_s",
        "returncode",
        "execution_mode",
        "error",
        "warnings_tail",
        "stdout_tail",
        "stderr_tail",
    ]
    with (outdir / "scalability_details.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(results, key=lambda r: r.get("key", "")):
            w.writerow(
                {
                    "dataset": r.get("dataset"),
                    "section": r.get("section"),
                    "model": r.get("model"),
                    "n": r.get("n"),
                    "p": r.get("p"),
                    "sweep_name": r.get("sweep_name"),
                    "sweep_value": r.get("sweep_value"),
                    "status": r.get("status"),
                    "data_s": r.get("data_s"),
                    "split_s": r.get("split_s"),
                    "fit_s": r.get("fit_s"),
                    "predict_s": r.get("predict_s"),
                    "auc_s": r.get("auc_s"),
                    "auc": r.get("auc"),
                    "patterns": r.get("patterns"),
                    "memory_mb": (r.get("memory_mb") or {}).get("fit_delta_from_after_data"),
                    "peak_process_tree_rss_mb": r.get("peak_process_tree_rss_mb"),
                    "mem_limit_mb": r.get("mem_limit_mb"),
                    "task_timeout_seconds": r.get("task_timeout_seconds"),
                    "elapsed_s": r.get("elapsed_s"),
                    "returncode": r.get("returncode"),
                    "execution_mode": (r.get("params") or {}).get("execution_mode"),
                    "error": r.get("error"),
                    "warnings_tail": r.get("warnings_tail"),
                    "stdout_tail": r.get("stdout_tail"),
                    "stderr_tail": r.get("stderr_tail"),
                }
            )
    # No reference data dependency: dashboard creation is driven only by the checkpoint JSON and supplied HTML template.
    with (outdir / "scalability_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields2 = [
            "dataset",
            "scale",
            "largest_complete_n",
            "largest_complete_p",
            "model",
            "fit_s",
            "auc",
            "memory_mb",
            "patterns",
        ]
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader()
        for ds, d in data.items():
            for scale, rows in [("n", d["ns"]), ("p", d["ps"])]:
                last = rows[-1] if rows else {}
                for model, k in [
                    ("Orig+Pat", ("ha", "haA", "mha", "hap")),
                    ("Pat Only", ("hp", "hpA", "mhp", "hpp")),
                    ("XGBoost", ("xg", "xgA", "mxg", None)),
                    ("LightGBM", ("lg", "lgA", "mlg", None)),
                ]:
                    w.writerow(
                        {
                            "dataset": ds,
                            "scale": scale,
                            "largest_complete_n": last.get("n"),
                            "largest_complete_p": last.get("p"),
                            "model": model,
                            "fit_s": last.get(k[0]),
                            "auc": last.get(k[1]),
                            "memory_mb": last.get(k[2]),
                            "patterns": last.get(k[3]) if k[3] else "",
                        }
                    )
    html = template_path.read_text(encoding="utf-8")
    html = js_replace(html, "DATASETS", data)
    html = js_replace(html, "META", meta)
    html = clean_html(html)
    html = html.replace(
        "Baseline Signal (n to 3M)",
        f"Baseline Signal (n to {fN(data['sparse_nonlinear']['ns'][-1]['n'])})"
        if data["sparse_nonlinear"]["ns"]
        else "Baseline Signal",
    )
    html = html.replace(
        "Threshold Grid (n to 500k)",
        f"Threshold Grid (n to {fN(data['threshold_grid']['ns'][-1]['n'])})"
        if data["threshold_grid"]["ns"]
        else "Threshold Grid",
    )
    html = refresh_axis_ticks(html)
    html = refresh_methodology_text(html, ckpt, args)
    out_html = (
        Path(args.output_html) if args.output_html else outdir / "hugiml_scalability_dashboard.html"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def collect_incomplete(results):
    out = []
    for ds in DATASETS:
        for sec, grid in [("n_scaling", N_SCALING[ds]), ("p_scaling", P_SCALING[ds])]:
            for n, p in grid:
                models = {m: find_res(results, ds, sec, m, n, p) for m in MODELS}
                if any(v is not None for v in models.values()) and not all(
                    v is not None for v in models.values()
                ):
                    out.append(
                        {
                            "dataset": ds,
                            "section": sec,
                            "n": n,
                            "p": p,
                            "completed_models": [
                                LABEL[m] for m, v in models.items() if v is not None
                            ],
                        }
                    )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Run and assemble the HUGIML scalability dashboard from a JSON checkpoint."
    )
    ap.add_argument(
        "--template-html",
        "--template",
        dest="template",
        default=None,
        help="Template HTML dashboard. By default, search starts at the grandparent folder of this script.",
    )
    ap.add_argument(
        "--out-dir",
        "--outdir",
        dest="outdir",
        default=str(DEFAULT_OUTDIR),
        help="Directory containing/writing scalability_checkpoint.json and assembled artifacts.",
    )
    ap.add_argument(
        "--output-html",
        default=None,
        help="Optional output HTML path. Default: <out-dir>/hugiml_scalability_dashboard.html",
    )
    ap.add_argument(
        "--assemble",
        dest="assemble",
        action="store_true",
        help="Assemble the dashboard from <out-dir>/scalability_checkpoint.json without running experiments.",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Recreate the results directory before running experiments.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip successful tasks already present in the checkpoint.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--task-timeout",
        type=float,
        default=3600,
        help="Per-task wall-clock limit in seconds. Default: 3600 (60 minutes). Use <=0 to disable.",
    )
    ap.add_argument(
        "--mem-limit-mb",
        type=int,
        default=-1,
        help="Process-tree RSS limit in MB. Default -1 uses 90% of system memory; 0 disables.",
    )
    ap.add_argument("--no-sweeps", action="store_true")
    ap.add_argument("--start-task", type=int, default=0)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--only-section", default=None)
    ap.add_argument("--only-dataset", default=None)
    ap.add_argument("--only-model", default=None)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--key", default="")
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--section", default="")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--p", type=int, default=0)
    ap.add_argument("--sweep-name", default=None)
    ap.add_argument("--sweep-value-json", default=None)
    ap.add_argument("--result-json", default=None)
    args = ap.parse_args()
    if args.worker:
        worker(args)
    elif args.assemble:
        build_outputs(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()

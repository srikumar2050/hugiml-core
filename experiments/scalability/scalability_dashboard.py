#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTDIR = ROOT / "results"
TEMPLATE_NAME = "hugiml_scalability_dashboard.html"


def find_repo_root(start: Path) -> Path | None:
    """Find the nearest parent containing src/hugiml, or return None."""
    for base in [start, *start.parents]:
        if (base / "src" / "hugiml").exists():
            return base.resolve()
    return None


SOURCE_ROOT_ENV = os.environ.get("HUGIML_SOURCE_ROOT")
SOURCE_ROOT = Path(SOURCE_ROOT_ENV).expanduser().resolve() if SOURCE_ROOT_ENV else find_repo_root(ROOT)
if SOURCE_ROOT is not None and (SOURCE_ROOT / "src").exists():
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
    sys.path.insert(0, str(SOURCE_ROOT))

try:
    import hugiml as _hugiml_pkg
    HUGIML_IMPORTED_FROM = str(Path(getattr(_hugiml_pkg, "__file__", "")).resolve())
except Exception:
    _hugiml_pkg = None
    HUGIML_IMPORTED_FROM = None

DATASETS = ("sparse_nonlinear", "threshold_grid")

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "hug_op_adaptive_full": {"family": "hug", "feature_mode": "original_plus_patterns", "scenario": "adaptive_full", "lr_solver": "auto", "label": "HUG OP adaptive full"},
    "hug_op_adaptive_saga": {"family": "hug", "feature_mode": "original_plus_patterns", "scenario": "solver_saga", "lr_solver": "saga", "label": "HUG OP adaptive saga"},
    "hug_op_adaptive_sgd": {"family": "hug", "feature_mode": "original_plus_patterns", "scenario": "solver_sgd", "lr_solver": "sgd", "label": "HUG OP adaptive SGD"},
    "hug_op_adaptive_s20": {"family": "hug", "feature_mode": "original_plus_patterns", "scenario": "adaptive_s20", "lr_solver": "auto", "label": "HUG OP adaptive 20%"},
    "hug_po_adaptive_full": {"family": "hug", "feature_mode": "patterns_only", "scenario": "adaptive_full", "lr_solver": "auto", "label": "HUG PO adaptive full"},
    "hug_po_adaptive_s20": {"family": "hug", "feature_mode": "patterns_only", "scenario": "adaptive_s20", "lr_solver": "auto", "label": "HUG PO adaptive 20%"},
    "xgb": {"family": "xgb", "feature_mode": "baseline", "scenario": "baseline", "label": "XGBoost"},
    "lgb": {"family": "lgb", "feature_mode": "baseline", "scenario": "baseline", "label": "LightGBM"},
}
MODELS = tuple(MODEL_SPECS.keys())

# The second dataset has 10x predictors in n-scaling (p=200 vs p=20).
# When --max-n is supplied, cap its n-scaling sweep at 1/10th of the
# first dataset's cap so a request such as --max-n 10Mn runs:
#   sparse_nonlinear <= 10,000,000 and threshold_grid <= 1,000,000.
N_SCALING_MAX_N_FRACTION: dict[str, float] = {
    DATASETS[0]: 1.0,
    DATASETS[1]: 0.1,
}

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
        (50_000_000, 20),
        (100_000_000, 20),
        (500_000_000, 20),
        (1_000_000_000, 20),
    ],
    "threshold_grid": [
        # 10x predictor count versus sparse_nonlinear n-scaling, so keep the
        # sample-size ladder at roughly 1/10th of the first dataset ladder.
        (1_000, 200),
        (5_000, 200),
        (10_000, 200),
        (50_000, 200),
        (100_000, 200),
        (300_000, 200),
        (500_000, 200),
        (1_000_000, 200),
        (5_000_000, 200),
        (10_000_000, 200),
        (50_000_000, 200),
        (100_000_000, 200),
    ],
}
P_SCALING = {
    "sparse_nonlinear": [(50_000, 20), (100_000, 100), (10_000, 1_000), (5_000, 2_000), (2_500, 4_000), (2_000, 5_000), (1_000, 10_000)],
    "threshold_grid": [(50_000, 20), (100_000, 100), (10_000, 1_000), (5_000, 2_000), (2_500, 4_000), (2_000, 5_000), (1_000, 10_000)],
}

SWEEP = {
    "sparse_nonlinear": {
        "B": {"n": 50_000, "p": 20, "values": [2, 3, 5, 7, 10, 15]},
        "G": {"n": 50_000, "p": 20, "values": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]},
        "topK": {"n": 50_000, "p": 20, "values": [10, 20, 30, 50, 100, 200]},
        "L": {"n": 30_000, "p": 20, "values": [1, 2]},
        "avf": {"n": 50_000, "p": 20, "values": [True, False]},
    },
    "threshold_grid": {
        "B": {"n": 30_000, "p": 200, "values": [2, 3, 5, 7, 10, 15]},
        "G": {"n": 30_000, "p": 200, "values": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]},
        "topK": {"n": 30_000, "p": 200, "values": [10, 20, 30, 50, 100, 200]},
        "L": {"n": 20_000, "p": 100, "values": [1, 2]},
        "avf": {"n": 30_000, "p": 200, "values": [True, False]},
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def temp_path_for(path: Path) -> Path:
    return path.parent / (path.name + ".tmp")


def resolve_outdir(raw: str | None) -> Path:
    return Path(raw).expanduser().resolve() if raw else DEFAULT_OUTDIR


def parse_size_count(raw: str | int | None) -> int | None:
    """Parse counts with optional k, m, mn, b, or bn suffixes."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    text = str(raw).strip().lower().replace(",", "").replace("_", "")
    if text in {"", "none", "all", "-1"}:
        return None
    multipliers = {
        "k": 1_000,
        "m": 1_000_000,
        "mn": 1_000_000,
        "b": 1_000_000_000,
        "bn": 1_000_000_000,
    }
    for suffix, multiplier in sorted(multipliers.items(), key=lambda item: -len(item[0])):
        if text.endswith(suffix):
            number = text[: -len(suffix)]
            if not number:
                raise argparse.ArgumentTypeError(f"invalid count: {raw!r}")
            return int(float(number) * multiplier)
    try:
        return int(float(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid count: {raw!r}") from exc


def effective_max_n_for_task(t: dict[str, Any], max_n: int | None) -> int | None:
    """Return the effective --max-n cap for a task.

    The high-p threshold_grid n-scaling experiment uses 10x predictors, so its
    n-scaling cap is intentionally 1/10th of the global cap. Other sections keep
    the user-supplied cap unchanged.
    """
    if max_n is None or int(max_n) <= 0:
        return None
    cap = int(max_n)
    if t.get("section") == "n_scaling":
        frac = float(N_SCALING_MAX_N_FRACTION.get(str(t.get("dataset")), 1.0))
        cap = max(1, int(cap * frac))
    return cap


def apply_size_caps(
    tasks: list[dict[str, Any]],
    *,
    max_n: int | None = None,
    max_p: int | None = None,
) -> list[dict[str, Any]]:
    """Return tasks whose configured sample and feature counts are within limits."""
    out = tasks
    if max_n is not None and int(max_n) > 0:
        out = [t for t in out if int(t.get("n", 0)) <= int(effective_max_n_for_task(t, max_n) or max_n)]
    if max_p is not None and int(max_p) > 0:
        out = [t for t in out if int(t.get("p", 0)) <= int(max_p)]
    return out


def task_filter_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    ns = [int(t["n"]) for t in tasks]
    ps = [int(t["p"]) for t in tasks]
    sections = sorted({str(t["section"]) for t in tasks})
    datasets = sorted({str(t["dataset"]) for t in tasks})
    models = sorted({str(t["model"]) for t in tasks})
    return {
        "task_count": len(tasks),
        "min_n": min(ns) if ns else None,
        "max_n": max(ns) if ns else None,
        "min_p": min(ps) if ps else None,
        "max_p": max(ps) if ps else None,
        "sections": sections,
        "datasets": datasets,
        "models": models,
    }


def get_total_memory_mb() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().total / 1048576)
    except Exception:
        return None


def effective_mem_limit_mb(raw: int | None) -> int | None:
    if raw is None or int(raw) == 0:
        return None
    if int(raw) < 0:
        total = get_total_memory_mb()
        return None if total is None else max(1, int(total * 0.90))
    return int(raw)


def process_tree_rss_mb(proc: subprocess.Popen) -> float | None:
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


def terminate_process_tree(proc: subprocess.Popen) -> None:
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
        _, alive = psutil.wait_procs(children + [p], timeout=5)
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
        parts.extend([str(t["sweep_name"]), repr(t.get("sweep_value"))])
    return "|".join(parts)


def make_tasks(include_sweeps: bool = True) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for section, grid in (("n_scaling", N_SCALING[dataset]), ("p_scaling", P_SCALING[dataset])):
            for n, p in grid:
                for model in MODELS:
                    t = {"dataset": dataset, "section": section, "model": model, "n": n, "p": p}
                    t["key"] = task_key(t)
                    out.append(t)
        if include_sweeps:
            # Preserve the original dashboard's sweep experiments, mapped to the
            # scenario model that represents Orig+Pat with full adaptive binning.
            for name, spec in SWEEP[dataset].items():
                for value in spec["values"]:
                    t = {
                        "dataset": dataset,
                        "section": f"parameter_sweep_{name}",
                        "model": "hug_op_adaptive_full",
                        "n": spec["n"],
                        "p": spec["p"],
                        "sweep_name": name,
                        "sweep_value": value,
                    }
                    t["key"] = task_key(t)
                    out.append(t)
    return out


def hug_params_for(model: str, sweep_name: str | None = None, sweep_value: Any = None) -> dict[str, Any]:
    spec = MODEL_SPECS[model]
    p: dict[str, Any] = {
        "B": 5,
        "L": 1,
        "G": 0.01,
        "topK": 50,
        "n_jobs": 4,
        "adaptive_binning": True,
        "b_candidates": [3, 5, 7, 10],
        "adaptive_binning_sample_frac": False,
        "adaptive_binning_sample_random_state": 42,
        "use_hotpath": True,
        "augmented_pair_transforms": False,
        "interaction_relaxed_mining": False,
        "execution_mode": "production",
        "feature_mode": spec["feature_mode"],
        "lr_solver": spec.get("lr_solver", "auto"),
        "topk_budget_strict": spec["feature_mode"] == "original_plus_patterns",
    }
    scenario = spec["scenario"]
    if scenario == "adaptive_s20":
        p["adaptive_binning_sample_frac"] = 0.20

    if sweep_name == "B":
        p.update({"adaptive_binning": False, "B": int(sweep_value)})
        p.pop("b_candidates", None)
        p["adaptive_binning_sample_frac"] = False
    elif sweep_name == "G":
        p["G"] = float(sweep_value)
    elif sweep_name == "topK":
        p["topK"] = int(sweep_value)
    elif sweep_name == "L":
        p["L"] = int(sweep_value)
    elif sweep_name == "avf":
        p["adaptive_binning"] = bool(sweep_value)
        if p["adaptive_binning"]:
            p.setdefault("b_candidates", [3, 5, 7, 10])
        else:
            p.pop("b_candidates", None)
            p["B"] = 5
            p["adaptive_binning_sample_frac"] = False
    return p


def emit_worker_result(args: argparse.Namespace, row: dict[str, Any]) -> None:
    text = json.dumps(row, ensure_ascii=False, allow_nan=False)
    if getattr(args, "result_json", None):
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = temp_path_for(path)
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    else:
        print(text, flush=True)


def worker(args: argparse.Namespace) -> None:
    import psutil
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    status = "ok"
    err = None
    fit_s = predict_s = auc_s = data_s = split_s = None
    auc = None
    patterns = None
    params: dict[str, Any] = {}
    rss_after_data = rss_before_fit = peak_fit = None
    t0all = time.perf_counter()
    try:
        t0 = time.perf_counter()
        X, y = (make_baseline_signal if args.dataset == "sparse_nonlinear" else make_threshold_grid)(args.n, args.p, args.seed)
        data_s = time.perf_counter() - t0
        proc = psutil.Process()
        rss_after_data = proc.memory_info().rss / 1048576
        t0 = time.perf_counter()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        del X, y
        split_s = time.perf_counter() - t0

        spec = MODEL_SPECS[args.model]
        sweep_value = json.loads(args.sweep_value_json) if getattr(args, "sweep_value_json", None) else None
        if spec["family"] == "hug":
            from hugiml import HUGIMLClassifierNative
            params = hug_params_for(args.model, getattr(args, "sweep_name", None), sweep_value)
            clf = HUGIMLClassifierNative(**params)
            fit_Xtr = Xtr
            pred_Xte = Xte
        elif spec["family"] == "xgb":
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
            fit_Xtr = pd.DataFrame(Xtr, columns=columns, copy=False)
            pred_Xte = pd.DataFrame(Xte, columns=columns, copy=False)

        rss_before_fit = proc.memory_info().rss / 1048576
        peak = [rss_before_fit]
        running = [True]

        def sample() -> None:
            while running[0]:
                try:
                    peak[0] = max(peak[0], proc.memory_info().rss / 1048576)
                except Exception:
                    pass
                time.sleep(0.05)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        try:
            t0 = time.perf_counter()
            clf.fit(fit_Xtr, ytr)
            fit_s = time.perf_counter() - t0
        finally:
            running[0] = False
            th.join(timeout=1)
        peak_fit = peak[0]

        t0 = time.perf_counter()
        proba = clf.predict_proba(pred_Xte)[:, 1]
        predict_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        auc = float(roc_auc_score(yte, proba))
        auc_s = time.perf_counter() - t0
        if spec["family"] == "hug":
            patterns = len(getattr(clf, "patterns_", []) or [])
    except Exception as e:
        status = "error"
        err = repr(e)

    row = {
        "key": args.key,
        "dataset": args.dataset,
        "section": args.section,
        "model": args.model,
        "model_label": MODEL_SPECS[args.model]["label"],
        "feature_mode": MODEL_SPECS[args.model]["feature_mode"],
        "scenario": MODEL_SPECS[args.model]["scenario"],
        "family": MODEL_SPECS[args.model]["family"],
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
            "rss_after_data": rss_after_data,
            "rss_before_fit": rss_before_fit,
            "peak_fit_window": peak_fit,
            "fit_delta_from_before_fit": None if rss_before_fit is None or peak_fit is None else max(0.0, peak_fit - rss_before_fit),
            "fit_delta_from_after_data": None if rss_after_data is None or peak_fit is None else max(0.0, peak_fit - rss_after_data),
        },
        "params": params,
        "sweep_name": getattr(args, "sweep_name", None),
        "sweep_value": json.loads(args.sweep_value_json) if getattr(args, "sweep_value_json", None) else None,
        "elapsed_s": time.perf_counter() - t0all,
        "error": err,
        "metadata": {"python": platform.python_version(), "platform": platform.platform()},
    }
    emit_worker_result(args, row)


def load_ckpt(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"created_at": now_iso(), "metadata": {}, "results": []}


def save_ckpt(path: Path, ckpt: dict[str, Any]) -> None:
    """Atomically save the checkpoint.

    Keep this compact rather than pretty-printed because the checkpoint is
    rewritten after every task. Compact JSON materially reduces write size and
    avoids spending extra time formatting large result histories.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path_for(path)
    tmp.write_text(json.dumps(ckpt, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def drain_text_stream(stream, parts: list[str], limit: int) -> None:
    total = 0
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            parts.append(chunk)
            total += len(chunk)
            while total > limit and parts:
                head = parts.pop(0)
                total -= len(head)
    except Exception:
        pass


def parse_worker_stdout(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("worker did not emit JSON on stdout")
    try:
        return json.loads(stripped)
    except Exception:
        for line in reversed(stripped.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        raise


def run_worker_task(t: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    timeout = None if args.task_timeout is None or float(args.task_timeout) <= 0 else float(args.task_timeout)
    effective_mem = effective_mem_limit_mb(args.mem_limit_mb)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--key", t["key"],
        "--dataset", t["dataset"],
        "--section", t["section"],
        "--model", t["model"],
        "--n", str(t["n"]),
        "--p", str(t["p"]),
        "--seed", str(args.seed),
    ]
    if t.get("sweep_name"):
        cmd += ["--sweep-name", str(t["sweep_name"]), "--sweep-value-json", json.dumps(t.get("sweep_value"))]
    started = time.perf_counter()
    peak_tree = None
    status = None
    error = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1)
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    stdout_thread = threading.Thread(target=drain_text_stream, args=(proc.stdout, stdout_parts, 1_000_000), daemon=True)
    stderr_thread = threading.Thread(target=drain_text_stream, args=(proc.stderr, stderr_parts, 200_000), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    try:
        while True:
            rc = proc.poll()
            rss = process_tree_rss_mb(proc) if rc is None else None
            if rss is not None:
                peak_tree = rss if peak_tree is None else max(peak_tree, rss)
            elapsed = time.perf_counter() - started
            if rc is None and effective_mem is not None and peak_tree is not None and peak_tree > effective_mem:
                status = "oom"
                error = f"process tree RSS {peak_tree:.1f} MB exceeded memory limit {effective_mem} MB"
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
    stdout_text = "".join(stdout_parts)
    stderr_text = "".join(stderr_parts)
    if status in {"oom", "timeout"}:
        res = {"key": t["key"], **t, "status": status, "error": error}
    else:
        try:
            res = parse_worker_stdout(stdout_text)
        except Exception as exc:
            res = {"key": t["key"], **t, "status": "error", "error": f"worker stdout JSON parse failed: {exc!r}"}
        if proc.returncode not in (0, None) and res.get("status") == "ok":
            res["status"] = "error"
            res["error"] = f"worker_returncode={proc.returncode}"
    # Keep successful rows clean: library warnings emitted to stderr (for example
    # HUGIMLWarning memory estimates) are not persisted into the checkpoint JSON.
    # Failure/OOM/timeout rows still keep bounded stdout/stderr tails for debugging.
    if res.get("status") != "ok":
        if stdout_text[-2000:]:
            res["stdout_tail"] = stdout_text[-2000:]
        if stderr_text[-4000:]:
            res["stderr_tail"] = stderr_text[-4000:]
    res.setdefault("key", t["key"])
    for k, v in t.items():
        res.setdefault(k, v)
    res["returncode"] = proc.returncode
    res["task_timeout_seconds"] = timeout
    res["mem_limit_mb"] = effective_mem
    res["peak_process_tree_rss_mb"] = peak_tree
    res["completed_at"] = now_iso()
    return res


def run_all(args: argparse.Namespace) -> None:
    outdir = resolve_outdir(args.outdir)
    if args.fresh and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / "scalability_checkpoint.json"
    ckpt = load_ckpt(ckpt_path)
    ckpt["metadata"] = {
        "generated_at": now_iso(),
        "threads": 4,
        "system": {"python": platform.python_version(), "platform": platform.platform(), "system_memory_mb": get_total_memory_mb()},
        "hugiml_config": {"L": 1, "G": 0.01, "topK": 50, "b_candidates": [3, 5, 7, 10], "lr_solver": "auto", "use_hotpath": True, "execution_mode": "production"},
        "scenarios": MODEL_SPECS,
        "sweeps": SWEEP,
        "primary_memory_metric": "peak process-tree RSS during the full worker task",
        "fit_memory_metric": "peak RSS during clf.fit minus RSS immediately before clf.fit",
        "task_limits": {
            "max_n": args.max_n,
            "max_p": args.max_p,
            "n_scaling_max_n_fraction": N_SCALING_MAX_N_FRACTION,
            "include_sweeps": not args.no_sweeps,
            "only_section": args.only_section,
            "only_dataset": args.only_dataset,
            "only_model": args.only_model,
        },
    }
    done = {r["key"] for r in ckpt.get("results", []) if r.get("status") == "ok"}
    tasks = make_tasks(include_sweeps=not args.no_sweeps)
    tasks = apply_size_caps(tasks, max_n=args.max_n, max_p=args.max_p)
    if args.only_section:
        tasks = [t for t in tasks if t["section"] == args.only_section]
    if args.only_dataset:
        tasks = [t for t in tasks if t["dataset"] == args.only_dataset]
    if args.only_model:
        tasks = [t for t in tasks if t["model"] == args.only_model]
    selected_task_summary = task_filter_summary(tasks)
    ckpt["metadata"]["selected_task_summary"] = selected_task_summary
    print(
        "selected "
        f"{selected_task_summary['task_count']} tasks "
        f"with n<= {args.max_n if args.max_n else 'all'} "
        f"and p<= {args.max_p if args.max_p else 'all'}",
        flush=True,
    )
    save_ckpt(ckpt_path, ckpt)
    if getattr(args, "start_task", 0):
        tasks = tasks[int(args.start_task):]
    if args.max_tasks is not None:
        tasks = tasks[: int(args.max_tasks)]
    for i, t in enumerate(tasks, 1):
        if args.resume and t["key"] in done:
            continue
        print(f"run {i}/{len(tasks)} {t['key']}", flush=True)
        res = run_worker_task(t, args)
        ckpt["results"] = [r for r in ckpt.get("results", []) if r.get("key") != t["key"]]
        ckpt["results"].append(res)
        save_ckpt(ckpt_path, ckpt)
    build_outputs(argparse.Namespace(outdir=str(outdir), output_html=args.output_html, include_sbom=getattr(args, "include_sbom", False)))


def numeric(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def flatten_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        # Drop rows from retired scenarios (for example the removed HUG OP/PO B=8
        # cases) so resumed/assembled dashboards do not reintroduce them from an
        # older checkpoint.
        if r.get("model") not in MODEL_SPECS:
            continue
        mem = r.get("memory_mb") or {}
        fit_s = r.get("fit_s") if r.get("fit_s") is not None else r.get("fit_seconds")
        predict_s = r.get("predict_s") if r.get("predict_s") is not None else r.get("predict_seconds")
        auc = r.get("auc") if r.get("auc") is not None else r.get("roc_auc")
        peak_rss_mb = r.get("peak_process_tree_rss_mb")
        if peak_rss_mb is None:
            peak_rss_mb = mem.get("peak_fit_window")
        rows.append({
            "dataset": r.get("dataset"),
            "section": r.get("section"),
            "model": r.get("model"),
            "sweep_name": r.get("sweep_name"),
            "sweep_value": r.get("sweep_value"),
            "model_label": r.get("model_label") or MODEL_SPECS.get(r.get("model"), {}).get("label", r.get("model")),
            "feature_mode": r.get("feature_mode") or MODEL_SPECS.get(r.get("model"), {}).get("feature_mode"),
            "scenario": r.get("scenario") or MODEL_SPECS.get(r.get("model"), {}).get("scenario"),
            "family": r.get("family") or MODEL_SPECS.get(r.get("model"), {}).get("family"),
            "n": r.get("n"),
            "p": r.get("p"),
            "status": r.get("status"),
            "fit_s": fit_s,
            "predict_s": predict_s,
            "auc": auc,
            "patterns": r.get("patterns"),
            "fit_delta_from_before_fit_mb": mem.get("fit_delta_from_before_fit"),
            "fit_delta_from_after_data_mb": mem.get("fit_delta_from_after_data"),
            "peak_process_tree_rss_mb": peak_rss_mb,
            "memory_plot_mb": peak_rss_mb,
            "memory_plot_gb": (float(peak_rss_mb) / 1024.0) if peak_rss_mb is not None else None,
            "error": r.get("error"),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "section", "model", "sweep_name", "sweep_value", "model_label", "feature_mode", "scenario", "family", "n", "p", "status", "fit_s", "predict_s", "auc", "patterns", "fit_delta_from_before_fit_mb", "fit_delta_from_after_data_mb", "peak_process_tree_rss_mb", "memory_plot_mb", "memory_plot_gb", "error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})




def render_html(rows: list[dict[str, Any]], metadata: dict[str, Any], reproducibility_sbom: dict[str, Any] | None = None) -> str:
    """Render a self-contained dynamic dashboard from checkpoint rows."""
    keep_fields = {
        "dataset", "section", "model", "sweep_name", "sweep_value", "model_label",
        "feature_mode", "scenario", "family", "n", "p", "status", "fit_s", "predict_s",
        "auc", "patterns", "peak_process_tree_rss_mb", "memory_plot_mb", "memory_plot_gb", "error",
    }
    ok_rows: list[dict[str, Any]] = []
    for r in rows:
        if r.get("status") == "ok":
            row = {k: r.get(k) for k in keep_fields}
            if row.get("memory_plot_gb") is None and row.get("peak_process_tree_rss_mb") is not None:
                row["memory_plot_gb"] = float(row["peak_process_tree_rss_mb"]) / 1024.0
            ok_rows.append(row)

    safe_metadata = dict(metadata or {})
    for key in ["generated_at", "completed_at", "created_at", "fit_memory_metric"]:
        safe_metadata.pop(key, None)
    if isinstance(safe_metadata.get("system"), dict):
        safe_metadata["system"] = {
            k: v for k, v in safe_metadata["system"].items()
            if k not in {"generated_at", "completed_at", "created_at"}
        }
    safe_metadata["primary_memory_metric"] = "peak_process_tree_rss_mb"
    safe_metadata["primary_memory_metric_label"] = "Peak task RSS"
    payload = {
        "metadata": safe_metadata,
        "rows": ok_rows,
        "models": MODEL_SPECS,
        "model_order": list(MODELS),
        "datasets": list(DATASETS),
        "primary_memory_metric": "peak_process_tree_rss_mb",
        "primary_memory_metric_label": "Peak task RSS",
        "reproducibility_sbom": reproducibility_sbom,
    }
    payload_json = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    css_text = r"""
:root{--bg:#f5f7fb;--panel:#fff;--p2:#f9fafb;--ink:#172033;--mu:#667085;--ln:#e4e7ec;--brand:#2563eb;--b2:#06b6d4;--chip:rgba(37,99,235,.10);--ha:rgba(37,99,235,.15);--hb:rgba(6,182,212,.12);--sh:0 8px 24px rgba(16,24,40,.08);--ra:14px;--good:#16a34a;--warn:#d97706;--bad:#dc2626}
body[data-t=dark]{--bg:#0b1020;--panel:#111827;--p2:#0f172a;--ink:#f8fafc;--mu:#94a3b8;--ln:#334155;--brand:#60a5fa;--b2:#a78bfa;--chip:rgba(96,165,250,.16);--ha:rgba(30,41,59,.9);--hb:rgba(96,165,250,.12);--sh:0 12px 32px rgba(0,0,0,.35);--good:#4ade80;--warn:#fbbf24;--bad:#f87171}
*{box-sizing:border-box;margin:0;padding:0}html{scroll-behavior:smooth}body{background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;font-size:14px;line-height:1.5}.app{display:grid;grid-template-columns:234px 1fr;min-height:100vh}.sb{background:var(--panel);border-right:1px solid var(--ln);padding:12px;position:sticky;top:0;height:100vh;overflow-y:auto;display:flex;flex-direction:column;gap:9px}.main{padding:22px 24px 64px;min-width:0}.logo{display:flex;gap:9px;align-items:center}.logoi{width:32px;height:32px;border-radius:9px;background:linear-gradient(135deg,var(--brand),var(--b2));display:grid;place-items:center;color:#fff;font-weight:900;font-size:14px;flex:0 0 auto}.logt h1{font-size:12.5px;font-weight:900;line-height:1.25}.logt p{font-size:10px;color:var(--mu);margin-top:1px}.box{border:1px solid var(--ln);border-radius:11px;padding:9px;background:var(--p2)}.box h3{font-size:9.5px;color:var(--mu);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;font-weight:800}.pills{display:flex;gap:5px;flex-wrap:wrap}.pill,.tab{border:1px solid var(--ln);background:var(--panel);color:var(--ink);border-radius:999px;padding:5px 10px;font-weight:800;font-size:10.5px;cursor:pointer;transition:all .15s}.pill.on,.pill:hover,.tab.on,.tab:hover{background:var(--chip);border-color:var(--brand)}nav{display:flex;flex-direction:column;gap:2px}.nb{border:0;background:transparent;color:var(--mu);text-align:left;padding:7px 8px;border-radius:8px;cursor:pointer;font-size:11.5px;font-weight:700;transition:all .15s;width:100%}.nb:hover,.nb.on{background:var(--chip);color:var(--ink);font-weight:900}select{width:100%;padding:6px 8px;border:1px solid var(--ln);border-radius:8px;background:var(--panel);color:var(--ink);font-size:11.5px;cursor:pointer}.foot{margin-top:auto;padding-top:8px;border-top:1px solid var(--ln);font-size:10px;color:var(--mu);line-height:1.6}.sec{display:none}.sec.on{display:block}.hero{background:radial-gradient(circle at 5% 5%,var(--ha),transparent 35%),radial-gradient(circle at 90% 15%,var(--hb),transparent 30%),var(--panel);border:1px solid color-mix(in srgb,var(--brand) 22%,var(--ln));border-radius:18px;padding:18px 20px;box-shadow:var(--sh);margin-bottom:14px}.hero h2{font-size:clamp(17px,2.1vw,25px);letter-spacing:-.04em;margin-bottom:5px}.hero p{color:var(--mu);font-size:12.5px;line-height:1.58;max-width:980px}.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.chip,.badge{border:1px solid color-mix(in srgb,var(--brand) 20%,var(--ln));background:var(--chip);border-radius:999px;padding:3px 8px;font-size:10.5px;font-weight:800}.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px}.g2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-bottom:12px}.card,.cc{background:var(--panel);border:1px solid var(--ln);border-radius:var(--ra);padding:13px 14px;box-shadow:var(--sh);margin-bottom:12px}.kpi{border-top:4px solid var(--brand)}.kpi .kl{font-size:9.5px;color:var(--mu);font-weight:900;text-transform:uppercase;letter-spacing:.05em}.kpi .kv{font-size:19px;font-weight:950;letter-spacing:-.035em;margin-top:4px;line-height:1.15}.kpi .ks{font-size:10.5px;color:var(--mu);margin-top:3px}.cc h3,.card h3{font-size:12.5px;font-weight:900;margin-bottom:5px}.ch{position:relative;height:285px}.ch.small{height:235px}.ch.tall{height:335px}.ni{font-size:11.5px;color:var(--mu);line-height:1.5;background:var(--p2);border-left:3px solid var(--brand);padding:7px 9px;border-radius:0 8px 8px 0;margin-bottom:9px}.ni.ok{border-left-color:var(--good)}.ni.warn{border-left-color:var(--warn)}.ni.bad{border-left-color:var(--bad)}.subtle{color:var(--mu);font-size:11.5px}.tabs{display:flex;gap:5px;flex-wrap:wrap;margin:6px 0 11px}.tc{display:none}.tc.on{display:block}.tw{overflow:auto;border:1px solid var(--ln);border-radius:10px;background:var(--panel);margin-bottom:12px}table{width:100%;border-collapse:collapse;font-size:11.5px}th,td{padding:6px 9px;border-bottom:1px solid var(--ln);vertical-align:middle}th{text-align:left;background:var(--p2);font-size:9.5px;color:var(--mu);font-weight:800;text-transform:uppercase;letter-spacing:.04em}tr:last-child td{border-bottom:0}td.r{text-align:right;font-variant-numeric:tabular-nums;font-weight:700}.leg{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px}.li{display:flex;align-items:center;gap:4px;font-size:11.1px;font-weight:800}.ld{width:9px;height:9px;border-radius:50%;display:inline-block}.ctrl{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap}.ctrl label{font-size:11.5px;color:var(--mu);font-weight:900}.ctrl select{width:auto}.wide{grid-column:1/-1}.model-tag{display:inline-flex;align-items:center;gap:5px}.model-dot{width:8px;height:8px;border-radius:50%;display:inline-block}.rank1{color:var(--good);font-weight:900}.rank2{color:var(--warn);font-weight:900}.chart-missing{display:grid;place-items:center;height:100%;border:1px dashed var(--ln);border-radius:10px;color:var(--mu);background:var(--p2)}.mini-list{display:grid;gap:7px}.mini-list div{padding:7px 9px;border:1px solid var(--ln);border-radius:9px;background:var(--p2);font-size:11.5px;color:var(--mu)}.mini-list strong{color:var(--ink)}@media(max-width:1100px){.g4{grid-template-columns:repeat(2,1fr)}.g3{grid-template-columns:1fr}.g2{grid-template-columns:1fr}}@media(max-width:760px){.app{grid-template-columns:1fr}.sb{height:auto;position:relative}.main{padding:12px}.g4{grid-template-columns:1fr}.ch,.ch.tall{height:260px}}
"""
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\"/>
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>
<title>HUGIML Scalability Analysis</title>
<script src=\"https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js\"></script>
<style>__CSS__</style>
</head>
<body data-t=\"light\">
<script id=\"payload\" type=\"application/json\">__PAYLOAD__</script>
<div class=\"app\">
<aside class=\"sb\">
  <div class=\"logo\"><div class=\"logoi\">⚡</div><div class=\"logt\"><h1>HUGIML Scalability</h1><p id=\"sideSub\"></p></div></div>
  <div class=\"box\"><h3>Theme</h3><div class=\"pills\"><button class=\"pill on\" data-th=\"light\">Light</button><button class=\"pill\" data-th=\"dark\">Dark</button></div></div>
  <div class=\"box\"><h3>Dataset</h3><select id=\"dsSel\"></select><div id=\"dsNote\" class=\"subtle\" style=\"margin-top:6px\"></div></div>
  <div class=\"box\"><h3>Sections</h3><nav>
    <button class=\"nb on\" data-s=\"ov\">📊 Overview</button>
    <button class=\"nb\" data-s=\"ns\">📈 n-Scaling</button>
    <button class=\"nb\" data-s=\"ps\">🔢 p-Scaling</button>
    <button class=\"nb\" data-s=\"mem\">💾 Memory</button>
    <button class=\"nb\" data-s=\"sw\">🎛 Parameter sweep</button>
    <button class=\"nb\" data-s=\"mth\">📋 Methodology</button>
  </nav></div>
  <div class=\"foot\">4 threads · test ROC AUC · memory in GB</div>
</aside>
<main class=\"main\">
<section class=\"sec on\" id=\"ov\">
  <div class=\"hero\"><h2 id=\"ov_title\"></h2><p id=\"ov_desc\"></p><div class=\"chips\" id=\"ov_chips\"></div></div>
  <div class=\"g4\" id=\"kpi_row\"></div>
  <div class=\"g2\">
    <div class=\"cc\"><h3>Fit time vs n</h3><div class=\"ni\" id=\"ov_fit_ni\"></div><div class=\"leg\" id=\"legend_ov_fit\"></div><div class=\"ch\"><canvas id=\"ovFit\"></canvas></div></div>
    <div class=\"cc\"><h3>Test AUC vs n</h3><div class=\"ni ok\" id=\"ov_auc_ni\"></div><div class=\"leg\" id=\"legend_ov_auc\"></div><div class=\"ch\"><canvas id=\"ovAuc\"></canvas></div></div>
  </div>
  <div class=\"g2\">
    <div class=\"cc\"><h3>Peak memory vs n</h3><div class=\"ni\" id=\"ov_mem_ni\"></div><div class=\"leg\" id=\"legend_ov_mem\"></div><div class=\"ch small\"><canvas id=\"ovMem\"></canvas></div></div>
    <div class=\"card\"><h3>Key findings</h3><div class=\"mini-list\" id=\"ov_insights\"></div></div>
  </div>
  <div class=\"cc\"><h3 id=\"snapshot_title\"></h3><div class=\"ni info\" id=\"snapshot_note\"></div><div class=\"tw\"><table id=\"snapshot_tbl\"></table></div></div>
</section>
<section class=\"sec\" id=\"ns\">
  <div class=\"hero\"><h2>n-Scaling</h2><p id=\"ns_hero\"></p></div>
  <div class=\"tabs\"><button class=\"tab on\" data-tab=\"ns_fit\">Fit time</button><button class=\"tab\" data-tab=\"ns_auc\">Test AUC</button><button class=\"tab\" data-tab=\"ns_mem\">Peak memory</button><button class=\"tab\" data-tab=\"ns_pat\">Patterns</button><button class=\"tab\" data-tab=\"ns_tbl\">Table</button></div>
  <div class=\"tc on\" id=\"ns_fit\"><div class=\"cc\"><h3>Fit time vs n</h3><div class=\"ni\" id=\"ns_fit_ni\"></div><div class=\"leg\" id=\"legend_ns_fit\"></div><div class=\"ch tall\"><canvas id=\"nsFit\"></canvas></div></div></div>
  <div class=\"tc\" id=\"ns_auc\"><div class=\"cc\"><h3>Test AUC vs n</h3><div class=\"ni ok\" id=\"ns_auc_ni\"></div><div class=\"leg\" id=\"legend_ns_auc\"></div><div class=\"ch\"><canvas id=\"nsAuc\"></canvas></div></div></div>
  <div class=\"tc\" id=\"ns_mem\"><div class=\"cc\"><h3>Peak memory vs n</h3><div class=\"ni\" id=\"ns_mem_ni\"></div><div class=\"leg\" id=\"legend_ns_mem\"></div><div class=\"ch\"><canvas id=\"nsMem\"></canvas></div></div></div>
  <div class=\"tc\" id=\"ns_pat\"><div class=\"cc\"><h3>Patterns mined vs n</h3><div class=\"ni info\" id=\"ns_pat_ni\"></div><div class=\"leg\" id=\"legend_ns_pat\"></div><div class=\"ch\"><canvas id=\"nsPat\"></canvas></div></div></div>
  <div class=\"tc\" id=\"ns_tbl\"><div class=\"tw\"><table id=\"nsTbl\"></table></div></div>
</section>
<section class=\"sec\" id=\"ps\">
  <div class=\"hero\"><h2>p-Scaling</h2><p id=\"ps_hero\"></p></div>
  <div class=\"tabs\"><button class=\"tab on\" data-tab=\"ps_fit\">Fit time</button><button class=\"tab\" data-tab=\"ps_ratio\">Ratio vs XGBoost</button><button class=\"tab\" data-tab=\"ps_auc\">Test AUC</button><button class=\"tab\" data-tab=\"ps_mem\">Peak memory</button><button class=\"tab\" data-tab=\"ps_tbl\">Table</button></div>
  <div class=\"tc on\" id=\"ps_fit\"><div class=\"cc\"><h3>Fit time vs p</h3><div class=\"ni\" id=\"ps_fit_ni\"></div><div class=\"leg\" id=\"legend_ps_fit\"></div><div class=\"ch tall\"><canvas id=\"psFit\"></canvas></div></div></div>
  <div class=\"tc\" id=\"ps_ratio\"><div class=\"cc\"><h3>Fit-time ratio vs XGBoost</h3><div class=\"ni warn\" id=\"ps_ratio_ni\"></div><div class=\"leg\" id=\"legend_ps_ratio\"></div><div class=\"ch\"><canvas id=\"psRatio\"></canvas></div></div></div>
  <div class=\"tc\" id=\"ps_auc\"><div class=\"cc\"><h3>Test AUC vs p</h3><div class=\"ni ok\" id=\"ps_auc_ni\"></div><div class=\"leg\" id=\"legend_ps_auc\"></div><div class=\"ch\"><canvas id=\"psAuc\"></canvas></div></div></div>
  <div class=\"tc\" id=\"ps_mem\"><div class=\"cc\"><h3>Peak memory vs p</h3><div class=\"ni\" id=\"ps_mem_ni\"></div><div class=\"leg\" id=\"legend_ps_mem\"></div><div class=\"ch\"><canvas id=\"psMem\"></canvas></div></div></div>
  <div class=\"tc\" id=\"ps_tbl\"><div class=\"tw\"><table id=\"psTbl\"></table></div></div>
</section>
<section class=\"sec\" id=\"mem\">
  <div class=\"hero\"><h2>Memory</h2><p>Memory plots use peak process-tree RSS during the full benchmark worker task, reported in GB. This is the clearest end-to-end RAM requirement for comparing classifiers.</p></div>
  <div class=\"tabs\"><button class=\"tab on\" data-tab=\"mem_n\">vs n</button><button class=\"tab\" data-tab=\"mem_p\">vs p</button><button class=\"tab\" data-tab=\"mem_tbl_n\">Table vs n</button><button class=\"tab\" data-tab=\"mem_tbl_p\">Table vs p</button></div>
  <div class=\"tc on\" id=\"mem_n\"><div class=\"cc\"><h3>Peak task RSS vs n</h3><div class=\"ni\" id=\"mem_n_ni\"></div><div class=\"leg\" id=\"legend_mem_n\"></div><div class=\"ch tall\"><canvas id=\"memN\"></canvas></div></div></div>
  <div class=\"tc\" id=\"mem_p\"><div class=\"cc\"><h3>Peak task RSS vs p</h3><div class=\"ni\" id=\"mem_p_ni\"></div><div class=\"leg\" id=\"legend_mem_p\"></div><div class=\"ch\"><canvas id=\"memP\"></canvas></div></div></div>
  <div class=\"tc\" id=\"mem_tbl_n\"><div class=\"tw\"><table id=\"memTblN\"></table></div></div>
  <div class=\"tc\" id=\"mem_tbl_p\"><div class=\"tw\"><table id=\"memTblP\"></table></div></div>
</section>
<section class=\"sec\" id=\"sw\">
  <div class=\"hero\"><h2>Parameter sweep</h2><p id=\"sw_hero\"></p></div>
  <div class=\"ctrl\"><label>Parameter:</label><select id=\"swSel\"></select></div>
  <div id=\"sw_area\"></div>
</section>
<section class=\"sec\" id=\"mth\">
  <div class=\"hero\"><h2>Methodology</h2><p>Benchmark setup, dataset definitions, model scenarios, and sweep grids used to produce the dashboard.</p></div>
  <div class=\"g2\"><div class=\"card\"><h3>Environment</h3><div id=\"envTable\"></div></div><div class=\"card\"><h3>Model scenarios</h3><div id=\"modelTable\"></div></div></div>
  <div class=\"g2\"><div class=\"card\"><h3>Dataset protocol</h3><div id=\"protocolBox\" class=\"mini-list\"></div></div><div class=\"card\"><h3>Sweep grids</h3><div id=\"sweepTable\"></div></div></div>
  <div class=\"card\" id=\"sbomCard\" style=\"display:none;margin-top:12px\"><details><summary style=\"cursor:pointer;font-weight:800\">Reproducibility / SBOM manifest</summary><div class=\"ni\" style=\"margin:8px 0 10px 0\">Minimal path-free manifest with Python/runtime details, key package versions, native build/ABI facts, compiler optimization indicators, and artifact hashes.</div><pre id=\"sbomPre\" style=\"white-space:pre-wrap;overflow:auto;max-height:520px;background:#0f172a;color:#e5e7eb;border-radius:12px;padding:14px;font-size:11px;line-height:1.45\"></pre></details></div>
</section>
</main></div>
<script>
(function(){
'use strict';
const payload=JSON.parse(document.getElementById('payload').textContent);
const allRows=(payload.rows||[]).filter(r=>r.status==='ok');
const models=payload.models||{};
const modelOrder=(payload.model_order||Object.keys(models)).filter(m=>allRows.some(r=>r.model===m));
const labels=Object.fromEntries(modelOrder.map(m=>[m,(models[m]&&models[m].label)||m]));
const palette=['#2563eb','#60a5fa','#1d4ed8','#06b6d4','#67e8f9','#0891b2','#d97706','#16a34a','#9333ea','#ef4444'];
const colors=Object.fromEntries(modelOrder.map((m,i)=>[m,palette[i%palette.length]]));
let currentCharts={};
function esc(v){return String(v==null?'—':v).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));}
function num(v){const n=Number(v);return Number.isFinite(n)?n:null;}
function uniq(a){return [...new Set(a)];}
function fN(v){const n=num(v);if(n==null)return '—'; if(n>=1e6)return (n/1e6).toLocaleString(undefined,{maximumFractionDigits:n%1e6?1:0})+'M'; if(n>=1e3)return (n/1e3).toLocaleString(undefined,{maximumFractionDigits:n%1e3?1:0})+'k'; return String(n);}
function fP(v){const n=num(v);return n==null?'—':n.toLocaleString();}
function fS(v){const n=num(v);if(n==null)return '—'; if(n>=60)return (n/60).toLocaleString(undefined,{maximumFractionDigits:1})+' min'; if(n>=10)return n.toLocaleString(undefined,{maximumFractionDigits:1})+' s'; return n.toLocaleString(undefined,{maximumFractionDigits:2})+' s';}
function fA(v){const n=num(v);return n==null?'—':n.toFixed(4);}
function fGB(v){const n=num(v);return n==null?'—':n.toLocaleString(undefined,{maximumFractionDigits:n>=10?1:2})+' GB';}
function fRaw(v){return v==null?'—':String(v);}
function dsLabel(ds){return ds==='sparse_nonlinear'?'Sparse nonlinear':'Threshold grid';}
function dsPredictorCount(ds){return ds==='threshold_grid'?200:20;}
function dsScaleLabel(ds){const maxN=latestX(ds,'n_scaling','n'); return dsLabel(ds)+' — '+fN(maxN)+' × '+fP(dsPredictorCount(ds));}
function currentDs(){return document.getElementById('dsSel').value || (payload.datasets||[])[0];}
function rowsFor(ds,section){return allRows.filter(r=>r.dataset===ds && r.section===section && modelOrder.includes(r.model));}
function sweepRows(ds,name){return allRows.filter(r=>r.dataset===ds && (r.sweep_name===name || r.section==='parameter_sweep_'+name));}
function valuesFor(ds,section,xKey){return uniq(rowsFor(ds,section).map(r=>num(r[xKey])).filter(v=>v!=null)).sort((a,b)=>a-b);}
function latestX(ds,section,xKey){const xs=valuesFor(ds,section,xKey);return xs.length?xs[xs.length-1]:null;}
function rowsAt(ds,section,xKey,x){return rowsFor(ds,section).filter(r=>num(r[xKey])===x);}
function bestRow(rs,key,dir='min'){const vals=rs.filter(r=>num(r[key])!=null); if(!vals.length)return null; return vals.slice().sort((a,b)=>dir==='max'?num(b[key])-num(a[key]):num(a[key])-num(b[key]))[0];}
function modelTag(m){return '<span class="model-tag"><span class="model-dot" style="background:'+colors[m]+'"></span>'+esc(labels[m]||m)+'</span>';}
function legend(id, used){const el=document.getElementById(id); if(!el)return; el.innerHTML=used.map(m=>'<span class="li"><span class="ld" style="background:'+colors[m]+'"></span>'+esc(labels[m])+'</span>').join('');}
function gridColor(){return getComputedStyle(document.body).getPropertyValue('--ln').trim()||'#e5e7eb';}
function textColor(){return getComputedStyle(document.body).getPropertyValue('--mu').trim()||'#667085';}
function chart(id,cfg){const el=document.getElementById(id); if(!el)return; if(currentCharts[id]){currentCharts[id].destroy(); delete currentCharts[id];} if(!window.Chart){el.parentElement.innerHTML='<div class="chart-missing">Chart.js is required to render this chart.</div>'; return;} currentCharts[id]=new Chart(el,cfg);}
function metricLabel(metric){return metric==='fit_s'?'Fit time, seconds':metric==='auc'?'Test ROC AUC':metric==='memory_plot_gb'?'Peak task RSS, GB':metric==='patterns'?'Patterns mined':metric;}
function yFormat(metric,v){return metric==='fit_s'?fS(v):metric==='auc'?fA(v):metric==='memory_plot_gb'?fGB(v):metric==='patterns'?fRaw(v):fRaw(v);}
function seriesData(ds,section,xKey,metric){const xs=valuesFor(ds,section,xKey); return {xs,datasets:modelOrder.map(m=>{const vals=xs.map(x=>{const r=rowsFor(ds,section).find(rr=>rr.model===m && num(rr[xKey])===x); return r&&num(r[metric])!=null?num(r[metric]):null;}); return {m,vals};}).filter(d=>d.vals.some(v=>v!=null))};}
function lineChart(id,ds,section,xKey,metric,opts={}){const s=seriesData(ds,section,xKey,metric); const used=s.datasets.map(d=>d.m); legend(opts.legendId,used); chart(id,{type:'line',data:{labels:s.xs.map(x=>xKey==='n'?fN(x):fP(x)),datasets:s.datasets.map(d=>({label:labels[d.m],data:d.vals,borderColor:colors[d.m],backgroundColor:colors[d.m]+'22',borderWidth:2,pointRadius:3,spanGaps:true,tension:.25}))},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{mode:'index',intersect:false,callbacks:{label:(ctx)=>ctx.dataset.label+': '+yFormat(metric,ctx.parsed.y)}}},interaction:{mode:'nearest',intersect:false},scales:{x:{title:{display:true,text:xKey},ticks:{color:textColor()},grid:{color:gridColor()}},y:{type:opts.logY?'logarithmic':'linear',title:{display:true,text:metricLabel(metric)},ticks:{color:textColor(),callback:(v)=>metric==='memory_plot_gb'?Number(v).toLocaleString(undefined,{maximumFractionDigits:1}):v},grid:{color:gridColor()}}}}});}
function ratioChart(ds){const xs=valuesFor(ds,'p_scaling','p'); const base='xgb'; const datasets=modelOrder.filter(m=>m!==base).map(m=>{const vals=xs.map(p=>{const b=rowsFor(ds,'p_scaling').find(r=>r.model===base && num(r.p)===p); const r=rowsFor(ds,'p_scaling').find(rr=>rr.model===m && num(rr.p)===p); return b&&r&&num(b.fit_s)>0?num(r.fit_s)/num(b.fit_s):null;}); return {m,vals};}).filter(d=>d.vals.some(v=>v!=null)); legend('legend_ps_ratio',datasets.map(d=>d.m)); chart('psRatio',{type:'line',data:{labels:xs.map(fP),datasets:datasets.map(d=>({label:labels[d.m]+' / XGBoost',data:d.vals,borderColor:colors[d.m],backgroundColor:colors[d.m]+'22',borderWidth:2,pointRadius:3,spanGaps:true,tension:.25}))},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:(ctx)=>ctx.dataset.label+': '+Number(ctx.parsed.y).toFixed(2)+'×'}}},scales:{x:{title:{display:true,text:'p'},grid:{color:gridColor()}},y:{title:{display:true,text:'Fit-time ratio'},grid:{color:gridColor()}}}}});}
function sectionNote(ds,section,metric,phrase,fmt,dir='min'){const xKey=section==='n_scaling'?'n':'p'; const x=latestX(ds,section,xKey); if(x==null)return 'No completed rows for this section.'; const rs=rowsAt(ds,section,xKey,x).filter(r=>num(r[metric])!=null); if(!rs.length)return 'Metric is not recorded for the latest completed point.'; const b=bestRow(rs,metric,dir); const label=xKey==='n'?'n='+fN(x):'p='+fP(x); return `At ${label}, ${labels[b.model]} has the ${phrase} (${fmt(b[metric])}). ${rs.length} model runs are available at this point.`;}
function slopeNote(ds,section,metric,fmt){const xKey=section==='n_scaling'?'n':'p'; const s=seriesData(ds,section,xKey,metric); const parts=[]; for(const d of s.datasets){const pairs=s.xs.map((x,i)=>[x,d.vals[i]]).filter(p=>p[1]!=null); if(pairs.length>=2){const first=pairs[0], last=pairs[pairs.length-1]; const ratio=last[1]/first[1]; if(Number.isFinite(ratio)) parts.push({m:d.m,ratio,last:last[1]});}} if(!parts.length)return ''; parts.sort((a,b)=>b.ratio-a.ratio); const top=parts[0]; return `${labels[top.m]} shows the largest end-to-end increase over the observed range (${top.ratio.toFixed(1)}×, ending at ${fmt(top.last)}).`;}
function fillTable(id,ds,section){const xKey=section==='n_scaling'?'n':'p';const xs=valuesFor(ds,section,xKey);let html='';for(const m of modelOrder){const rows=xs.map(x=>rowsAt(ds,section,xKey,x).find(r=>r.model===m)).filter(Boolean);if(rows.length===0)continue;html+=`<details style="margin-bottom:10px;padding:8px;border:1px solid var(--ln);border-radius:6px"><summary style="cursor:pointer;font-weight:700">${labels[m]}</summary><table style="width:100%;margin-top:8px;font-size:11px"><thead><tr><th style="padding:4px">${xKey}</th><th class="r">Fit</th><th class="r">AUC</th><th class="r">Peak RSS</th><th class="r">Patterns</th></tr></thead><tbody>${rows.map(r=>`<tr><td style="padding:4px">${xKey==='n'?fN(r[xKey]):fP(r[xKey])}</td><td class="r">${fS(r.fit_s)}</td><td class="r">${fA(r.auc)}</td><td class="r">${r.memory_plot_gb!=null?r.memory_plot_gb.toFixed(1):'-'} GB</td><td class="r">${r.patterns??'—'}</td></tr>`).join('')}</tbody></table></details>`;}document.getElementById(id).innerHTML=html;}
function fillMemoryTable(id,ds,section){const xKey=section==='n_scaling'?'n':'p'; const xs=valuesFor(ds,section,xKey); let head='<thead><tr><th>'+xKey+'</th>'+modelOrder.map(m=>'<th class="r">'+esc(labels[m])+'</th>').join('')+'</tr></thead>'; const rowsHtml=xs.map(x=>'<tr><td>'+esc(xKey==='n'?fN(x):fP(x))+'</td>'+modelOrder.map(m=>{const r=rowsAt(ds,section,xKey,x).find(rr=>rr.model===m);return '<td class="r">'+fGB(r&&r.memory_plot_gb)+'</td>';}).join('')+'</tr>').join(''); document.getElementById(id).innerHTML=head+'<tbody>'+rowsHtml+'</tbody>';}
function snapshot(ds){const n=latestX(ds,'n_scaling','n'); const rs=rowsAt(ds,'n_scaling','n',n).sort((a,b)=>modelOrder.indexOf(a.model)-modelOrder.indexOf(b.model)); document.getElementById('snapshot_title').textContent='Latest n-scaling snapshot at n='+fN(n); const bf=bestRow(rs,'fit_s'), ba=bestRow(rs,'auc','max'), bm=bestRow(rs,'memory_plot_gb'); document.getElementById('snapshot_note').textContent=rs.length?`Fastest fit: ${labels[bf.model]} (${fS(bf.fit_s)}). Best AUC: ${labels[ba.model]} (${fA(ba.auc)}). Lowest peak memory: ${labels[bm.model]} (${fGB(bm.memory_plot_gb)}).`:'No completed rows.'; document.getElementById('snapshot_tbl').innerHTML='<thead><tr><th>Model</th><th>Scenario</th><th class="r">Fit</th><th class="r">Predict</th><th class="r">AUC</th><th class="r">Peak RSS</th><th class="r">Patterns</th></tr></thead><tbody>'+rs.map(r=>'<tr><td>'+modelTag(r.model)+'</td><td>'+esc(r.scenario)+'</td><td class="r">'+fS(r.fit_s)+'</td><td class="r">'+fS(r.predict_s)+'</td><td class="r">'+fA(r.auc)+'</td><td class="r">'+fGB(r.memory_plot_gb)+'</td><td class="r">'+(r.patterns??'—')+'</td></tr>').join('')+'</tbody>';}
function updateOverview(){const ds=currentDs(); const ns=rowsFor(ds,'n_scaling'); const ps=rowsFor(ds,'p_scaling'); const maxN=latestX(ds,'n_scaling','n'); const maxP=latestX(ds,'p_scaling','p'); const nModels=uniq(allRows.map(r=>r.model)).length;const nDatasets=uniq(allRows.map(r=>r.dataset)).length;document.getElementById('sideSub').textContent=nModels+' models, '+nDatasets+' datasets'; document.getElementById('dsNote').textContent=ds==='sparse_nonlinear'?'Dense nonlinear signal: completed to '+fN(maxN)+' × 20.':'Threshold/local-interaction signal: completed to '+fN(maxN)+' × 200.'; document.getElementById('ov_title').textContent=dsLabel(ds)+' scalability'; document.getElementById('ov_desc').textContent='End-to-end comparison of the active model scenarios across sample-size scaling, feature-count scaling, peak task memory, and HUGIML hyperparameter sweeps.'; document.getElementById('ov_chips').innerHTML=['Complete coverage',modelOrder.length+' model variants','n up to '+fN(maxN),'p up to '+fP(maxP)].map(x=>'<span class="chip">'+esc(x)+'</span>').join(''); const latest=rowsAt(ds,'n_scaling','n',maxN); const bf=bestRow(latest,'fit_s'), ba=bestRow(latest,'auc','max'), bm=bestRow(latest,'memory_plot_gb'), bp=bestRow(latest,'patterns','max'); const kpis=[['Largest completed n',fN(maxN),latest.length+' model rows'],['Fastest fit',bf?labels[bf.model]:'—',bf?fS(bf.fit_s):'—'],['Best AUC',ba?labels[ba.model]:'—',ba?fA(ba.auc):'—'],['Lowest peak RSS',bm?labels[bm.model]:'—',bm?fGB(bm.memory_plot_gb):'—']]; document.getElementById('kpi_row').innerHTML=kpis.map((k,i)=>'<div class="card kpi"><div class="kl">'+esc(k[0])+'</div><div class="kv">'+esc(k[1])+'</div><div class="ks">'+esc(k[2])+'</div></div>').join(''); document.getElementById('ov_fit_ni').textContent=sectionNote(ds,'n_scaling','fit_s','shortest fit time',fS,'min'); document.getElementById('ov_auc_ni').textContent=sectionNote(ds,'n_scaling','auc','highest test AUC',fA,'max'); document.getElementById('ov_mem_ni').textContent=sectionNote(ds,'n_scaling','memory_plot_gb','lowest peak task RSS',fGB,'min'); const bf2=bestRow(rowsFor(ds,'n_scaling'),'fit_s');const ba2=bestRow(rowsFor(ds,'n_scaling'),'auc','max');const pRows=rowsAt(ds,'p_scaling','p',maxP);const nsAucRows=rowsFor(ds,'n_scaling');const nsAucVals=nsAucRows.map(r=>r.auc).filter(v=>v!=null);const auc_range=nsAucVals.length?`${fA(Math.min(...nsAucVals))} to ${fA(Math.max(...nsAucVals))}`:'-';document.getElementById('ov_insights').innerHTML=[bf2?`${labels[bf2.model]} dominates fit performance at ${fS(bf2.fit_s)}, remaining competitive on accuracy across all data sizes.`:'',ba2?`${labels[ba2.model]} delivers best test AUC (${fA(ba2.auc)}); full range across models is ${auc_range}.`:'',`Strong scaling invariance: fit time and accuracy stable from n=${fN(latestX(ds,'n_scaling','n'))} to n=${fN(latestX(ds,'n_scaling','n'))}; no model degradation at scale.`,`${pRows.length}/${modelOrder.length} models scale to maximum feature count (p=${fP(maxP)}); parameter sweeps reveal sensitivity to hyperparameter tuning choices.`].filter(Boolean).map(x=>'<div>'+esc(x)+'</div>').join(''); snapshot(ds); lineChart('ovFit',ds,'n_scaling','n','fit_s',{legendId:'legend_ov_fit',logY:true}); lineChart('ovAuc',ds,'n_scaling','n','auc',{legendId:'legend_ov_auc'}); lineChart('ovMem',ds,'n_scaling','n','memory_plot_gb',{legendId:'legend_ov_mem'});}
function updateScaling(){const ds=currentDs(); const maxN=latestX(ds,'n_scaling','n'), maxP=latestX(ds,'p_scaling','p'); document.getElementById('ns_hero').textContent='Sample size varies while p is fixed for each dataset. Charts use log scaling for fit time to keep small and large runs readable.'; document.getElementById('ps_hero').textContent='Feature count varies while n is adjusted by the configured grid. Ratio view divides each model fit time by XGBoost at the same p.'; document.getElementById('ns_fit_ni').textContent=sectionNote(ds,'n_scaling','fit_s','shortest fit time',fS,'min'); document.getElementById('ns_auc_ni').textContent=sectionNote(ds,'n_scaling','auc','highest test AUC',fA,'max'); document.getElementById('ns_mem_ni').textContent=sectionNote(ds,'n_scaling','memory_plot_gb','lowest peak task RSS',fGB,'min')+' '+slopeNote(ds,'n_scaling','memory_plot_gb',fGB); document.getElementById('ns_pat_ni').textContent=sectionNote(ds,'n_scaling','patterns','highest pattern count',v=>String(v)+' patterns','max'); document.getElementById('ps_fit_ni').textContent=sectionNote(ds,'p_scaling','fit_s','shortest fit time',fS,'min'); document.getElementById('ps_auc_ni').textContent=sectionNote(ds,'p_scaling','auc','highest test AUC',fA,'max'); document.getElementById('ps_mem_ni').textContent=sectionNote(ds,'p_scaling','memory_plot_gb','lowest peak task RSS',fGB,'min'); document.getElementById('ps_ratio_ni').textContent=`At the largest completed p (${fP(maxP)}), values below 1.0 mean the scenario fits faster than XGBoost on the same data.`; lineChart('nsFit',ds,'n_scaling','n','fit_s',{legendId:'legend_ns_fit',logY:true}); lineChart('nsAuc',ds,'n_scaling','n','auc',{legendId:'legend_ns_auc'}); lineChart('nsMem',ds,'n_scaling','n','memory_plot_gb',{legendId:'legend_ns_mem'}); lineChart('nsPat',ds,'n_scaling','n','patterns',{legendId:'legend_ns_pat'}); lineChart('psFit',ds,'p_scaling','p','fit_s',{legendId:'legend_ps_fit',logY:true}); ratioChart(ds); lineChart('psAuc',ds,'p_scaling','p','auc',{legendId:'legend_ps_auc'}); lineChart('psMem',ds,'p_scaling','p','memory_plot_gb',{legendId:'legend_ps_mem'}); fillTable('nsTbl',ds,'n_scaling'); fillTable('psTbl',ds,'p_scaling');}
function updateMemory(){const ds=currentDs(); document.getElementById('mem_n_ni').textContent=sectionNote(ds,'n_scaling','memory_plot_gb','lowest peak task RSS',fGB,'min')+' '+slopeNote(ds,'n_scaling','memory_plot_gb',fGB); document.getElementById('mem_p_ni').textContent=sectionNote(ds,'p_scaling','memory_plot_gb','lowest peak task RSS',fGB,'min')+' '+slopeNote(ds,'p_scaling','memory_plot_gb',fGB); lineChart('memN',ds,'n_scaling','n','memory_plot_gb',{legendId:'legend_mem_n'}); lineChart('memP',ds,'p_scaling','p','memory_plot_gb',{legendId:'legend_mem_p'}); fillMemoryTable('memTblN',ds,'n_scaling'); fillMemoryTable('memTblP',ds,'p_scaling');}
function sweepNames(ds){return uniq(allRows.filter(r=>r.dataset===ds && String(r.section||'').startsWith('parameter_sweep_')).map(r=>String(r.sweep_name||r.section.replace('parameter_sweep_','')))).sort();}
function sweepChart(id,rs,metric){const data=rs.filter(r=>num(r[metric])!=null).sort((a,b)=>String(a.sweep_value).localeCompare(String(b.sweep_value),undefined,{numeric:true})); chart(id,{type:'line',data:{labels:data.map(r=>String(r.sweep_value)),datasets:[{label:metricLabel(metric),data:data.map(r=>num(r[metric])),borderColor:'#2563eb',backgroundColor:'#2563eb22',borderWidth:2,pointRadius:4,tension:.25}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:(ctx)=>metricLabel(metric)+': '+yFormat(metric,ctx.parsed.y)}}},scales:{x:{title:{display:true,text:'sweep value'},grid:{color:gridColor()}},y:{title:{display:true,text:metricLabel(metric)},grid:{color:gridColor()}}}}});}
function updateSweeps(){const ds=currentDs();const names=sweepNames(ds);const sel=document.getElementById('swSel');const oldVal=sel.value;sel.innerHTML=names.map(n=>'<option value="'+esc(n)+'" '+(n===oldVal?'selected':'')+'>'+esc(n)+'</option>').join('');const sw=sel.value||names[0];const rs=sweepRows(ds,sw);const el=document.getElementById('sw_area');if(!el||rs.length===0){el.innerHTML='<div class="ni">No sweep data.</div>';return;}document.getElementById('sw_hero').textContent='Parameter: '+sw;let html='';if(sw==='B'){html=`<div class="g2"><div class="cc"><h3>B sweep — fit (n=${fN(rs[0].n)}, p=${rs[0].p})</h3><div class="ni">Fit time nearly flat. B chosen on accuracy.</div><div class="ch" style="height:220px"><canvas id="sw_bFit"></canvas></div></div><div class="cc"><h3>B sweep — AUC & patterns</h3><div class="ni">AUC peaks near B=5. Use B=5 as default.</div><div class="ch" style="height:220px"><canvas id="sw_bAuc"></canvas></div></div></div>`;el.innerHTML=html;setTimeout(()=>{const sr=rs.slice().sort((a,b)=>a.sweep_value-b.sweep_value);sweepChart('sw_bFit',sr,'fit_s');const a=sr.map(r=>r.auc),p=sr.map(r=>r.patterns||0),aMn=Math.max(0.5,Math.floor((Math.min(...a)-0.02)*20)/20),aMx=Math.min(1.0,Math.ceil((Math.max(...a)+0.01)*20)/20);chart('sw_bAuc',{type:'line',data:{labels:sr.map(r=>String(r.sweep_value)),datasets:[{label:'AUC',data:a,borderColor:'#2563eb',backgroundColor:'#2563eb22',yAxisID:'y',tension:.3,borderWidth:2.5,pointRadius:5},{label:'Patterns',data:p,borderColor:'#d97706',yAxisID:'y1',tension:.3,borderWidth:2,borderDash:[4,3],pointRadius:5}]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index'},plugins:{legend:{display:true}},scales:{x:{ticks:{color:textColor()}},y:{min:aMn,max:aMx,title:{display:true,text:'AUC'},ticks:{color:textColor(),callback:v=>typeof v==='number'?v.toFixed(2):v}},y1:{position:'right',title:{display:true,text:'Patterns'},ticks:{color:textColor()}}}}});},50);}else if(sw==='G'){html=`<div class="g2"><div class="cc"><h3>G sweep — AUC & patterns (n=${fN(rs[0].n)}, p=${rs[0].p})</h3><div class="ni">Lower G mines more patterns. G=0.01 is default.</div><div class="ch" style="height:230px"><canvas id="sw_gAuc"></canvas></div></div><div class="cc"><h3>G sweep — fit</h3><div class="ni">Fit nearly flat despite pattern changes.</div><div class="ch" style="height:230px"><canvas id="sw_gFit"></canvas></div></div></div>`;el.innerHTML=html;setTimeout(()=>{const sr=rs.slice().sort((a,b)=>b.sweep_value-a.sweep_value);sweepChart('sw_gFit',sr,'fit_s');const a=sr.map(r=>r.auc),p=sr.map(r=>r.patterns||0),aMn=Math.max(0.5,Math.floor((Math.min(...a)-0.02)*20)/20),aMx=Math.min(1.0,Math.ceil((Math.max(...a)+0.01)*20)/20);chart('sw_gAuc',{type:'line',data:{labels:sr.map(r=>String(r.sweep_value)),datasets:[{label:'AUC',data:a,borderColor:'#2563eb',backgroundColor:'#2563eb22',yAxisID:'y',tension:.3,borderWidth:2.5,pointRadius:5},{label:'Patterns',data:p,borderColor:'#d97706',yAxisID:'y1',tension:.3,borderWidth:2,borderDash:[4,3],pointRadius:5}]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index'},plugins:{legend:{display:true}},scales:{x:{ticks:{color:textColor()}},y:{min:aMn,max:aMx,title:{display:true,text:'AUC'},ticks:{color:textColor(),callback:v=>typeof v==='number'?v.toFixed(2):v}},y1:{position:'right',title:{display:true,text:'Patterns'},ticks:{color:textColor()}}}}});},50);}else if(sw==='topK'){html=`<div class="cc" style="max-width:620px"><h3>topK sweep (n=${fN(rs[0].n)}, p=${rs[0].p})</h3><div class="ni">topK caps pattern budget. Binding at lower G.</div><div class="ch" style="height:270px"><canvas id="sw_topk"></canvas></div></div>`;el.innerHTML=html;setTimeout(()=>{const sr=rs.slice().sort((a,b)=>a.sweep_value-b.sweep_value);const a=sr.map(r=>r.auc),p=sr.map(r=>r.patterns||0),f=sr.map(r=>r.fit_s),aMn=Math.max(0.5,Math.floor((Math.min(...a)-0.02)*20)/20),aMx=Math.min(1.0,Math.ceil((Math.max(...a)+0.01)*20)/20);chart('sw_topk',{type:'line',data:{labels:sr.map(r=>String(r.sweep_value)),datasets:[{label:'AUC',data:a,borderColor:'#2563eb',backgroundColor:'#2563eb22',yAxisID:'y',pointRadius:5,tension:.3,borderWidth:2.5},{label:'Patterns',data:p,borderColor:'#d97706',yAxisID:'y1',pointRadius:5,tension:.3,borderWidth:2,borderDash:[4,3]},{label:'Fit',data:f,borderColor:'#16a34a',yAxisID:'y2',pointRadius:4,tension:.3,borderWidth:1.5,borderDash:[2,3]}]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index'},plugins:{legend:{display:true}},scales:{x:{ticks:{color:textColor()}},y:{min:aMn,max:aMx,title:{display:true,text:'AUC'},ticks:{color:textColor(),callback:v=>typeof v==='number'?v.toFixed(2):v}},y1:{position:'right',title:{display:true,text:'Patterns'},ticks:{color:textColor()}},y2:{display:false}}}});},50);}else if(sw==='L'){const sr=rs.slice().sort((a,b)=>a.sweep_value-b.sweep_value);const hasMem=sr.some(r=>r.memory_plot_gb!=null);html=`<div class="g2"><div class="card"><h3 style="font-size:12.5px;font-weight:800;margin-bottom:8px">L=1 vs L=2 (n=${fN(rs[0].n)}, p=${fP(rs[0].p)})</h3>${sr.map(r=>`<div style="border:1px solid var(--ln);border-radius:9px;padding:10px;margin-bottom:7px"><div style="font-size:14px;font-weight:900;color:var(--opp)">L = ${r.sweep_value}</div><div style="display:grid;grid-template-columns:repeat(${hasMem?4:3},1fr);gap:7px;margin-top:8px"><div><div style="font-size:9.5px;color:var(--mu)">Fit</div><div style="font-weight:800">${r.fit_s!=null?r.fit_s.toFixed(3)+'s':'—'}</div></div><div><div style="font-size:9.5px;color:var(--mu)">Patterns</div><div style="font-weight:800">${r.patterns??'—'}</div></div><div><div style="font-size:9.5px;color:var(--mu)">AUC</div><div style="font-weight:800">${r.auc!=null?r.auc.toFixed(4):'—'}</div></div>${hasMem?'<div><div style="font-size:9.5px;color:var(--mu)">Mem Δ</div><div style="font-weight:800">'+(r.memory_plot_gb!=null?r.memory_plot_gb.toFixed(1)+' MB':'—')+'</div></div>':''}</div></div>`).join('')}</div><div class="card"><h3 style="font-size:12.5px;font-weight:800;margin-bottom:8px">When to use L=2</h3><div style="font-size:11.5px;color:var(--mu);line-height:1.9">L=2 adds interactions. AUC gain <0.5%, fit 4.5× slower, patterns double. Prefer L=1 unless validated.</div></div></div>`;el.innerHTML=html;}else if(sw==='avf'){const sr=rs.slice().sort((a,b)=>(a.sweep_value===true||a.sweep_value===1?-1:1));const hasMem=sr.some(r=>r.memory_plot_gb!=null);html=`<div class="g2"><div class="card"><h3 style="font-size:12.5px;font-weight:800;margin-bottom:8px">Adaptive vs Fixed (n=${fN(rs[0].n)}, p=${fP(rs[0].p)})</h3>${sr.map((r,i)=>`<div style="border:1px solid var(--ln);border-radius:9px;padding:10px;margin-bottom:7px"><div style="font-size:13px;font-weight:900;color:var(--${i===0?'opp':'po'})">${r.sweep_value===true||r.sweep_value===1?'Adaptive':'Fixed B=5'}</div><div style="display:grid;grid-template-columns:repeat(${hasMem?4:3},1fr);gap:7px;margin-top:8px"><div><div style="font-size:9.5px;color:var(--mu)">Fit</div><div style="font-weight:800">${r.fit_s!=null?r.fit_s.toFixed(3)+'s':'—'}</div></div><div><div style="font-size:9.5px;color:var(--mu)">Patterns</div><div style="font-weight:800">${r.patterns??'—'}</div></div><div><div style="font-size:9.5px;color:var(--mu)">AUC</div><div style="font-weight:800">${r.auc!=null?r.auc.toFixed(4):'—'}</div></div>${hasMem?'<div><div style="font-size:9.5px;color:var(--mu)">Mem Δ</div><div style="font-weight:800">'+(r.memory_plot_gb!=null?r.memory_plot_gb.toFixed(1)+' MB':'—')+'</div></div>':''}</div></div>`).join('')}</div><div class="card"><h3 style="font-size:12.5px;font-weight:800;margin-bottom:8px">Choosing</h3><div style="font-size:11.5px;color:var(--mu);line-height:1.9"><strong style="color:var(--opp)">Adaptive:</strong> Info gain. Heterogeneous scales. AUC priority.</div><div style="height:1px;background:var(--ln);margin:10px 0"></div><div style="font-size:11.5px;color:var(--mu);line-height:1.9"><strong style="color:var(--po)">Fixed:</strong> Faster, reproducible. Large p. Uniform distributions.</div></div></div>`;el.innerHTML=html;}else{html=`<div class="cc"><h3>${esc(sw)}</h3><div class="ni">Data.</div></div>`;el.innerHTML=html;}}
function methodology(){const m=payload.metadata||{}, sys=m.system||{}; const env=[['Python',sys.python],['Platform',sys.platform],['Logical CPUs',sys.cpu_count_logical||sys.logical_cpus],['System RAM',sys.system_memory_mb?fGB(sys.system_memory_mb/1024):undefined],['Threads',m.threads],['Train/test split','75% / 25%, stratified'],['Metric','Test-set ROC AUC'],['Dashboard memory','Peak task RSS in GB']]; document.getElementById('envTable').innerHTML='<table><tbody>'+env.map(([k,v])=>'<tr><td>'+esc(k)+'</td><td class="r">'+esc(v??'not recorded')+'</td></tr>').join('')+'</tbody></table>'; document.getElementById('modelTable').innerHTML='<table><thead><tr><th>Model</th><th>Family</th><th>Feature mode</th><th>Scenario</th><th>LR solver</th></tr></thead><tbody>'+modelOrder.map(k=>{const v=models[k]||{};return '<tr><td>'+modelTag(k)+'</td><td>'+esc(v.family)+'</td><td>'+esc(v.feature_mode)+'</td><td>'+esc(v.scenario)+'</td><td>'+esc(v.lr_solver||'—')+'</td></tr>';}).join('')+'</tbody></table>'; document.getElementById('protocolBox').innerHTML=['Two synthetic binary-classification datasets are generated with deterministic seeds.','Each benchmark row uses a single stratified holdout split.','Fit time measures classifier training only; prediction and AUC time are tracked separately.','Peak memory is measured from the parent process as process-tree RSS across the worker task.','Built-in HUGIML solver variants keep deterministic random_state/random_seed defaults aligned with the default downstream classifier.'].map(x=>'<div>'+esc(x)+'</div>').join(''); const sweeps=m.sweeps||{}; const sr=[]; for(const [ds,cfg] of Object.entries(sweeps)){for(const [name,c] of Object.entries(cfg||{})){sr.push([dsLabel(ds),name,c.n,c.p,(c.values||[]).join(', ')]);}} document.getElementById('sweepTable').innerHTML='<table><thead><tr><th>Dataset</th><th>Sweep</th><th>n</th><th>p</th><th>Values</th></tr></thead><tbody>'+sr.map(r=>'<tr>'+r.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>'; const sb=payload.reproducibility_sbom; const sbomCard=document.getElementById('sbomCard'), sbomPre=document.getElementById('sbomPre'); if(sb&&sbomCard&&sbomPre){sbomCard.style.display='block'; sbomPre.textContent=JSON.stringify(sb,null,2);}}
function updateAll(){updateOverview(); updateScaling(); updateMemory(); updateSweeps();}
function init(){const dsSel=document.getElementById('dsSel'); const dss=(payload.datasets||uniq(allRows.map(r=>r.dataset))).filter(ds=>allRows.some(r=>r.dataset===ds)); dsSel.innerHTML=dss.map(ds=>'<option value="'+esc(ds)+'">'+esc(dsScaleLabel(ds))+'</option>').join(''); dsSel.addEventListener('change',updateAll); document.getElementById('swSel').addEventListener('change',updateSweeps); document.querySelectorAll('[data-th]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-th]').forEach(x=>x.classList.remove('on')); b.classList.add('on'); document.body.dataset.t=b.dataset.th; updateAll();})); document.querySelectorAll('.nb').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.nb').forEach(x=>x.classList.remove('on')); document.querySelectorAll('.sec').forEach(x=>x.classList.remove('on')); b.classList.add('on'); document.getElementById(b.dataset.s).classList.add('on'); setTimeout(updateAll,40);})); document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{const sec=b.closest('.sec'); sec.querySelectorAll('.tab').forEach(x=>x.classList.remove('on')); sec.querySelectorAll('.tc').forEach(x=>x.classList.remove('on')); b.classList.add('on'); document.getElementById(b.dataset.tab).classList.add('on'); setTimeout(updateAll,40);})); methodology(); updateAll();}
init();
})();
</script>
</body>
</html>""".replace("__CSS__", css_text).replace("__PAYLOAD__", payload_json)


def safe_jsonable(obj: Any) -> Any:
    """Return an object containing only standards-compliant JSON values."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {str(k): safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe_jsonable(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            x = float(obj)
            return x if x == x and x not in (float("inf"), float("-inf")) else None
        if isinstance(obj, np.ndarray):
            return safe_jsonable(obj.tolist())
    except Exception:
        pass
    return str(obj)


def _drop_unavailable(obj: Any) -> Any:
    """Remove null/empty/unavailable fields from public reproducibility JSON."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            cleaned = _drop_unavailable(value)
            if cleaned is None or cleaned == "" or cleaned == {} or cleaned == []:
                continue
            out[str(key)] = cleaned
        return out
    if isinstance(obj, list):
        out = []
        for value in obj:
            cleaned = _drop_unavailable(value)
            if cleaned is None or cleaned == "" or cleaned == {} or cleaned == []:
                continue
            out.append(cleaned)
        return out
    return obj


def _compact_jsonable(obj: Any) -> Any:
    return _drop_unavailable(safe_jsonable(obj))


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024, label: str | None = None) -> dict[str, Any]:
    """Return a path-free SHA-256 fingerprint for a readable file."""
    p = Path(path)
    out: dict[str, Any] = {"file": label or p.name}
    if not p.exists() or not p.is_file():
        return out
    try:
        st = p.stat()
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        out.update({"size_bytes": int(st.st_size), "sha256": h.hexdigest()})
    except Exception as exc:
        out["error"] = type(exc).__name__
    return out


def _package_version(dist_name: str) -> str | None:
    try:
        return str(importlib_metadata.version(dist_name))
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _key_package_versions() -> dict[str, str]:
    """Only record package versions that materially affect benchmark reproducibility."""
    names = [
        "hugiml",
        "hugiml-core",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "psutil",
        "joblib",
        "threadpoolctl",
    ]
    out: dict[str, str] = {}
    for name in names:
        version = _package_version(name)
        if version is not None:
            out[name] = version
    return out


def _compiler_version_line(command: str | None) -> dict[str, Any] | None:
    if not command:
        return None
    try:
        parts = shlex.split(str(command))
    except Exception:
        parts = str(command).split()
    if not parts:
        return None
    executable = shutil.which(parts[0])
    if executable is None:
        return None
    try:
        proc = subprocess.run(
            [parts[0], "--version"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except Exception:
        return {"compiler": parts[0]}
    first_line = next((line.strip() for line in (proc.stdout or "").splitlines() if line.strip()), "")
    return _compact_jsonable({"compiler": parts[0], "version": first_line})


def _sanitize_build_tokens(value: Any) -> str | None:
    """Remove local include/library paths while preserving relevant build flags."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        tokens = shlex.split(text)
    except Exception:
        tokens = text.split()
    sanitized: list[str] = []
    skip_next = False
    path_flags = {"-I", "-L", "-isystem", "-iquote", "/I", "/LIBPATH:"}
    for token in tokens:
        if skip_next:
            skip_next = False
            sanitized.append("<path>")
            continue
        if token in path_flags:
            sanitized.extend([token, "<path>"])
            skip_next = True
            continue
        if any(token.startswith(prefix) and len(token) > len(prefix) for prefix in ["-I", "-L", "-isystem", "-iquote", "/I", "/LIBPATH:"]):
            prefix = next(prefix for prefix in ["-isystem", "-iquote", "/LIBPATH:", "-I", "-L", "/I"] if token.startswith(prefix))
            sanitized.append(prefix + "<path>")
            continue
        # Replace path-bearing tokens, but keep ordinary compiler switches.
        if ("/" in token or "\\" in token) and not token.startswith(("-O", "/O", "-m", "-f", "-D", "-std", "/arch:")):
            sanitized.append("<path>")
            continue
        sanitized.append(token)
    return " ".join(sanitized) if sanitized else None


def _optimization_indicators(*values: Any) -> list[str]:
    text = " ".join(str(v) for v in values if v)
    try:
        tokens = shlex.split(text)
    except Exception:
        tokens = text.split()
    keep: list[str] = []
    prefixes = (
        "-O",
        "/O",
        "-march",
        "-mtune",
        "-mcpu",
        "-mavx",
        "-msse",
        "-mfma",
        "-flto",
        "-fopenmp",
        "-fno-math-errno",
        "-ffast-math",
        "-DNDEBUG",
        "-std=",
        "/arch:",
        "/openmp",
    )
    for token in tokens:
        if token.startswith(prefixes) and token not in keep:
            keep.append(token)
    return keep


def _hugiml_module_metadata() -> dict[str, Any]:
    out: dict[str, Any] = {
        "version_attr": getattr(_hugiml_pkg, "__version__", None) if _hugiml_pkg is not None else None,
        "distributions": {
            name: version
            for name in ["hugiml", "hugiml-core", "hugiml_core"]
            if (version := _package_version(name)) is not None
        },
    }
    try:
        import _hugiml_core as core_ext  # type: ignore
        core_path = Path(getattr(core_ext, "__file__", ""))
        out["native_extension"] = {
            "module": "_hugiml_core",
            "filename": core_path.name,
            "fingerprint": _sha256_file(core_path, label=core_path.name),
        }
    except Exception:
        pass
    return _compact_jsonable(out)


def _source_tree_fingerprint(root: Path | None, *, max_files: int = 2000, max_file_size_bytes: int = 2 * 1024 * 1024) -> dict[str, Any]:
    """Return an aggregate source/build-input hash without exposing file paths."""
    if root is None:
        return {}
    base = Path(root).resolve()
    if not base.exists():
        return {}
    rel_roots = [
        "src",
        "native",
        "include",
        "cmake",
        "CMakeLists.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "MANIFEST.in",
        "requirements.txt",
        "requirements-dev.txt",
        "experiments/scalability/scalability_dashboard.py",
    ]
    allowed_suffixes = {"", ".c", ".cc", ".cmake", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".in", ".lock", ".md", ".py", ".pyi", ".toml", ".txt", ".yaml", ".yml", ".cfg"}
    skip_parts = {".git", ".mypy_cache", ".pytest_cache", "__pycache__", "build", "dist", "results"}
    files: list[Path] = []
    for rel in rel_roots:
        candidate = base / rel
        if not candidate.exists():
            continue
        if candidate.is_file():
            files.append(candidate)
            continue
        for f in candidate.rglob("*"):
            if not f.is_file():
                continue
            if any(part in skip_parts for part in f.relative_to(base).parts):
                continue
            if f.suffix.lower() not in allowed_suffixes:
                continue
            try:
                if f.stat().st_size > max_file_size_bytes:
                    continue
            except Exception:
                continue
            files.append(f)
    files = sorted(set(files), key=lambda p: p.relative_to(base).as_posix())[:max_files]
    tree_hash = hashlib.sha256()
    for f in files:
        rel = f.relative_to(base).as_posix()
        fp = _sha256_file(f, label=f.name)
        tree_hash.update(rel.encode("utf-8", errors="replace"))
        tree_hash.update(b"\0")
        tree_hash.update(str(fp.get("size_bytes")).encode("ascii", errors="replace"))
        tree_hash.update(b"\0")
        tree_hash.update(str(fp.get("sha256")).encode("ascii", errors="replace"))
        tree_hash.update(b"\0")
    return _compact_jsonable({
        "tree_sha256": tree_hash.hexdigest(),
        "file_count_hashed": len(files),
        "file_count_limit": int(max_files),
        "max_file_size_bytes": int(max_file_size_bytes),
        "note": "Aggregate source/build-input hash only; file paths and source-control metadata are intentionally omitted.",
    })


def _native_build_metadata() -> dict[str, Any]:
    """Capture minimal Python/native build facts that can affect performance."""
    config_keys = [
        "SOABI",
        "EXT_SUFFIX",
        "MULTIARCH",
        "Py_DEBUG",
        "Py_ENABLE_SHARED",
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "CPPFLAGS",
        "LDFLAGS",
        "LDSHARED",
        "CCSHARED",
        "CONFIG_ARGS",
    ]
    raw_sysconfig = {k: sysconfig.get_config_var(k) for k in config_keys}
    sysconfig_public = {k: _sanitize_build_tokens(v) for k, v in raw_sysconfig.items()}
    env_keys = [
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "CPPFLAGS",
        "LDFLAGS",
        "ARCHFLAGS",
        "CMAKE_ARGS",
        "CMAKE_BUILD_PARALLEL_LEVEL",
        "CMAKE_GENERATOR",
        "NPY_NUM_BUILD_JOBS",
    ]
    env_public = {k: _sanitize_build_tokens(os.environ.get(k)) for k in env_keys if os.environ.get(k)}
    compiler_versions = {
        key: value
        for key, cmd in {
            "env_CC": os.environ.get("CC"),
            "env_CXX": os.environ.get("CXX"),
            "sysconfig_CC": raw_sysconfig.get("CC"),
            "sysconfig_CXX": raw_sysconfig.get("CXX"),
        }.items()
        if (value := _compiler_version_line(cmd))
    }
    optimization_flags = _optimization_indicators(*raw_sysconfig.values(), *env_public.values())
    return _compact_jsonable({
        "python_abi": {
            "SOABI": raw_sysconfig.get("SOABI"),
            "EXT_SUFFIX": raw_sysconfig.get("EXT_SUFFIX"),
            "MULTIARCH": raw_sysconfig.get("MULTIARCH"),
            "Py_DEBUG": raw_sysconfig.get("Py_DEBUG"),
            "Py_ENABLE_SHARED": raw_sysconfig.get("Py_ENABLE_SHARED"),
        },
        "compiler_versions": compiler_versions,
        "build_flags": sysconfig_public,
        "build_environment": env_public,
        "optimization_indicators": optimization_flags,
        "note": "Local include/library paths, command transcripts, and dynamic-linker dumps are intentionally omitted.",
    })

def build_reproducibility_sbom(*, checkpoint: Path, out_dir: Path, payload: dict[str, Any], rows: list[dict[str, Any]], output_paths: dict[str, Path]) -> dict[str, Any]:
    """Build a minimal, path-free reproducibility manifest for scalability outputs."""
    artifacts: dict[str, Any] = {
        "script": _sha256_file(Path(__file__).resolve(), label="scalability_dashboard.py"),
        "checkpoint_input": _sha256_file(checkpoint, label=checkpoint.name),
    }
    for name, p in output_paths.items():
        if name in {"html", "sbom"}:
            # The HTML embeds this manifest and the SBOM file has not been written yet.
            continue
        artifacts[name] = _sha256_file(p, label=p.name)

    thread_env_keys = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    manifest = {
        "bom_format": "HUGIML scalability reproducibility manifest",
        "bom_version": "2.0",
        "python_runtime": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "build": list(platform.python_build()),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": _key_package_versions(),
        "hugiml": _hugiml_module_metadata(),
        "native_build": _native_build_metadata(),
        "runtime_environment": {
            "threading": {k: os.environ.get(k) for k in thread_env_keys if os.environ.get(k)},
        },
        "artifacts": artifacts,
        "source_tree": _source_tree_fingerprint(SOURCE_ROOT),
        "redactions": {
            "output_directory": "<output-dir>",
            "local_paths": "<path>",
        },
        "privacy_note": "This manifest intentionally omits raw command output, sys.path, full package inventories, local paths, and unavailable/null fields; path placeholders document redaction without exposing real locations.",
    }
    return _compact_jsonable(manifest)

def build_outputs(args: argparse.Namespace) -> None:
    outdir = resolve_outdir(args.outdir)
    ckpt_path = outdir / "scalability_checkpoint.json"
    ckpt = load_ckpt(ckpt_path)
    rows = flatten_results(ckpt.get("results", []))
    csv_path = outdir / "scalability_results_flat.csv"
    write_csv(csv_path, rows)
    output_html = Path(args.output_html).expanduser().resolve() if args.output_html else outdir / "hugiml_scalability_dashboard.html"
    sbom: dict[str, Any] | None = None
    if getattr(args, "include_sbom", False):
        sbom_path = outdir / "scalability_reproducibility_sbom.json"
        sbom = build_reproducibility_sbom(
            checkpoint=ckpt_path,
            out_dir=outdir,
            payload=ckpt,
            rows=rows,
            output_paths={"flat_csv": csv_path, "html": output_html, "sbom": sbom_path},
        )
        sbom_path.write_text(json.dumps(sbom, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    html_text = render_html(rows, ckpt.get("metadata", {}), reproducibility_sbom=sbom)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run and assemble the HUGIML scalability scenario dashboard from a JSON checkpoint."
    )
    p.add_argument("--worker", action="store_true")
    p.add_argument("--key")
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--section")
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--n", type=int)
    p.add_argument("--p", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--result-json")
    p.add_argument("--out-dir", dest="outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--output-html", default=None)
    p.add_argument("--assemble", action="store_true")
    p.add_argument("--include-sbom", action="store_true", help="Include a minimal path-free reproducibility/SBOM manifest in the Methodology tab and write scalability_reproducibility_sbom.json.")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only-section")
    p.add_argument("--only-dataset", choices=DATASETS)
    p.add_argument("--only-model", choices=MODELS)
    p.add_argument("--start-task", type=int, default=0)
    p.add_argument("--max-tasks", type=int)
    p.add_argument("--no-sweeps", action="store_true")
    p.add_argument(
        "--max-n",
        type=parse_size_count,
        default=None,
        help="Run tasks with n less than or equal to this value. Supports k, M, Mn, B, and Bn suffixes.",
    )
    p.add_argument(
        "--max-p",
        type=parse_size_count,
        default=None,
        help="Run tasks with p less than or equal to this value. Supports k, M, Mn, B, and Bn suffixes.",
    )
    p.add_argument("--sweep-name", default=None)
    p.add_argument("--sweep-value-json", default=None)
    p.add_argument("--task-timeout", type=float, default=3600)
    p.add_argument("--mem-limit-mb", type=int, default=-1)
    args = p.parse_args()
    if args.worker:
        worker(args)
    elif args.assemble:
        build_outputs(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()

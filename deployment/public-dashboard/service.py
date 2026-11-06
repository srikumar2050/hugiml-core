"""Bounded in-memory job coordination for a single web process."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

DATASETS = {"credit": "Credit risk", "churn": "Customer churn", "claims": "Claims review"}

MODES = {"auto": "Automatic", "lr": "Logistic regression", "rpte": "RPTE"}

TOP_K = (20, 50, 100)

ORDERS = (1, 2)

PAGES = (
    "Overview",
    "Validation",
    "Representation Audit",
    "Pattern Inventory",
    "Case Review",
    "Data Quality & Policy",
    "Monitoring",
    "Pruning",
)

MAX_RESULT_BYTES = 4 * 1024 * 1024


def validate(dataset, mode, topk=20, order=2):

    if not isinstance(dataset, str) or dataset not in DATASETS:
        raise ValueError("Choose a predefined dataset.")
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("Choose Automatic, logistic regression, or RPTE.")
    if type(topk) is not int or topk not in TOP_K:
        raise ValueError("Pattern budget must be 20, 50, or 100.")
    if type(order) is not int or order not in ORDERS:
        raise ValueError("Pattern order must be 1 or 2.")


def terminate_tree(proc):

    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        psutil.wait_procs(children, timeout=5)
    except psutil.NoSuchProcess:
        pass
    proc.wait(timeout=5)


def run_job(dataset, mode, timeout, topk=20, order=2, pruning=None):

    validate(dataset, mode, topk, order)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    with tempfile.TemporaryDirectory(prefix="hugiml-demo-") as directory:
        output = Path(directory) / "result.json"
        extra = []
        if pruning is not None:
            request_file = Path(directory) / "request.json"
            request_file.write_text(json.dumps(pruning), encoding="utf-8")
            extra = [str(request_file)]
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("worker.py")),
                dataset,
                mode,
                str(output),
                str(topk),
                str(order),
            ]
            + extra,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            proc.wait(timeout=timeout)
            if proc.returncode != 0:
                raise RuntimeError("Training could not complete. Please try again.")
            if output.stat().st_size > MAX_RESULT_BYTES:
                raise RuntimeError("Result exceeds the demonstration size limit.")
            result = json.loads(output.read_text(encoding="utf-8"))
            expected = (
                set(PAGES) | {"__pruning", "__cases"} if pruning is None else {"view", "summary"}
            )
            if set(result) != expected:
                raise RuntimeError("Incomplete demonstration result.")
            return result
        finally:
            if proc.poll() is None:
                terminate_tree(proc)


class Coordinator:
    """One active child, two waiting jobs, and bounded session retention."""

    def __init__(self, runner=run_job, *, timeout=120, ttl=900, capacity=8, cooldown=30):
        self.runner, self.timeout, self.ttl = runner, timeout, ttl
        self.capacity, self.cooldown = capacity, cooldown
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demo")
        self.sessions = {}
        self.outstanding = 0
        self.closed = False

    def _expire(self):
        now = time.monotonic()
        for key, state in list(self.sessions.items()):
            if now - state["seen"] > self.ttl:
                del self.sessions[key]

    def session(self, token=None, *, touch=True):
        with self.lock:
            self._expire()
            if token in self.sessions:
                if touch:
                    self.sessions[token]["seen"] = time.monotonic()
                return token
            if not touch:
                return token
            if len(self.sessions) >= self.capacity:
                raise ValueError("The demo is busy. Please return in a few minutes.")
            token = secrets.token_urlsafe(32)
            self.sessions[token] = dict(
                seen=time.monotonic(),
                status="ready",
                result=None,
                revision=0,
                last_submit=-float("inf"),
                label="",
                config=None,
                workflow=None,
                model_revision=None,
                comparison=None,
                cases=[],
            )
            return token

    def submit(self, owner, dataset, mode="auto", topk=20, order=2):
        validate(dataset, mode, topk, order)
        with self.lock:
            self._expire()
            state = self.sessions.get(owner)
            if state is None or self.closed:
                raise ValueError("Session expired. Reload the page.")
            if state["status"] in ("queued", "running", "pruning queued", "pruning"):
                raise ValueError("Your training job is already queued or running.")
            if self.outstanding >= 3:
                raise ValueError("The training queue is full. Please try again shortly.")
            if time.monotonic() - state["last_submit"] < self.cooldown:
                raise ValueError("Please wait 30 seconds between training requests.")
            state.update(
                status="queued",
                result=None,
                workflow=None,
                model_revision=None,
                comparison=None,
                cases=[],
                config=(dataset, mode, topk, order),
                revision=state["revision"] + 1,
                last_submit=time.monotonic(),
                seen=time.monotonic(),
                label=f"{DATASETS[dataset]} · topK {topk} · L {order}",
            )
            self.outstanding += 1
            return self.pool.submit(
                self._execute, owner, state["revision"], dataset, mode, topk, order
            )

    def _execute(self, owner, revision, dataset, mode, topk, order):
        result = None
        status = "failed"
        try:
            with self.lock:
                self._expire()
                state = self.sessions.get(owner)
                if state is None or state["revision"] != revision:
                    return
                state["status"] = "running"
            result = self.runner(dataset, mode, self.timeout, topk, order)
            workflow = result.pop("__pruning", None)
            cases = result.pop("__cases", [])
            if not isinstance(cases, list) or len(cases) > 1000:
                raise ValueError("Invalid case evidence")
            result = {page: json.dumps(body).encode("utf-8") for page, body in result.items()}
            if (
                sum(map(len, result.values())) + len(json.dumps(cases)) + len(json.dumps(workflow))
                > MAX_RESULT_BYTES
            ):
                raise ValueError("Result too large")
            status = "complete"
        except subprocess.TimeoutExpired:
            status = "timed out"
        except Exception:
            status = "failed"
        finally:
            with self.lock:
                self._expire()
                state = self.sessions.get(owner)
                if state is not None and state["revision"] == revision:
                    state.update(status=status, result=result if status == "complete" else None)
                    if status == "complete":
                        state.update(workflow=workflow, model_revision=revision, cases=cases)
                self.outstanding -= 1

    def snapshot(self, owner, page):
        if not isinstance(page, str) or page not in PAGES:
            raise ValueError("Choose a governance page.")
        with self.lock:
            self._expire()
            state = self.sessions.get(owner)
            if state is None:
                raise ValueError("Session expired. Reload the page.")
            return (
                state["status"],
                state["label"],
                state["revision"],
                json.loads(state["comparison"])
                if page == "Pruning" and state["comparison"]
                else json.loads(state["result"][page])
                if state["result"] and page in state["result"]
                else None,
            )

    def pruning_info(self, owner):
        with self.lock:
            self._expire()
            state = self.sessions.get(owner)
            if state is None:
                raise ValueError("Session expired. Reload the page.")
            return state["model_revision"], json.loads(json.dumps(state["workflow"]))

    def case(self, owner, index, revision=None):
        from case_review import unpack

        with self.lock:
            self._expire()
            state = self.sessions.get(owner)
            if state is None:
                raise ValueError("Session expired. Reload the page.")
            count = len(state["cases"])
            current = state["model_revision"]
            if not count:
                return current, 0, 0, None
            if revision != current:
                index = 0
            if type(index) is not int or not 0 <= index < count:
                raise ValueError(f"Choose a row index from 0 to {count - 1}.")
            value = state["cases"][index]
        return current, count, index, unpack(value)

    def submit_pruning(self, owner, model_revision, kind, items):
        from pruning_workflow import validate_selection

        with self.lock:
            self._expire()
            state = self.sessions.get(owner)
            if state is None or self.closed or not state["workflow"]:
                raise ValueError("Train a model before pruning.")
            if type(model_revision) is not int or state["model_revision"] != model_revision:
                raise ValueError("The model changed. Refresh the pruning selection.")
            validate_selection(state["workflow"], kind, items)
            if state["status"] in ("queued", "running", "pruning queued", "pruning"):
                raise ValueError("Your session already has a running or queued job.")
            if self.outstanding >= 3:
                raise ValueError("The training queue is full. Please try again shortly.")
            if time.monotonic() - state["last_submit"] < self.cooldown:
                raise ValueError("Please wait 30 seconds between training or pruning requests.")
            state.update(
                status="pruning queued",
                revision=state["revision"] + 1,
                comparison=None,
                last_submit=time.monotonic(),
                seen=time.monotonic(),
            )
            request = {
                "kind": kind,
                "items": list(items),
                "fingerprint": state["workflow"]["fingerprint"],
            }
            self.outstanding += 1
            return self.pool.submit(self._prune, owner, state["revision"], state["config"], request)

    def _prune(self, owner, revision, config, request):
        status, result = "pruning failed", None
        try:
            with self.lock:
                self._expire()
                state = self.sessions.get(owner)
                if state is None or state["revision"] != revision:
                    return
                state["status"] = "pruning"
            dataset, mode, topk, order = config
            result = run_job(dataset, mode, self.timeout, topk, order, pruning=request)
            result = json.dumps(result["view"]).encode("utf-8")
            with self.lock:
                state = self.sessions.get(owner)
                retained = (
                    sum(map(len, state["result"].values())) if state and state["result"] else 0
                )
                if state:
                    retained += len(json.dumps(state["cases"])) + len(json.dumps(state["workflow"]))
            if len(result) + retained > MAX_RESULT_BYTES:
                raise ValueError("Result exceeds the session size limit")
            status = "compared"
        except subprocess.TimeoutExpired:
            status = "pruning timed out"
        except Exception:
            status = "pruning failed"
        finally:
            with self.lock:
                self._expire()
                state = self.sessions.get(owner)
                if state is not None and state["revision"] == revision:
                    state.update(status=status, comparison=result if status == "compared" else None)
                self.outstanding -= 1

    def reset_pruning(self, owner, model_revision):
        with self.lock:
            self._expire()
            state = self.sessions.get(owner)
            if (
                state is None
                or type(model_revision) is not int
                or state["model_revision"] != model_revision
            ):
                raise ValueError("The model changed. Refresh the pruning selection.")
            if state["status"] in ("queued", "running", "pruning queued", "pruning"):
                raise ValueError("Wait for the current job before resetting the comparison.")
            state.update(comparison=None, status="complete", revision=state["revision"] + 1)

    def close(self):
        with self.lock:
            self.closed = True
        self.pool.shutdown(wait=True)

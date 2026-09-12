import argparse
import errno
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.scalability import scalability_dashboard as d


def arguments(tmp_path, **changes):
    values = dict(
        outdir=str(tmp_path / "results"),
        fresh=False,
        resume=True,
        plan=False,
        seed=42,
        n_jobs=1,
        lr_source_policy="main_effect",
        max_n=None,
        max_p=None,
        no_sweeps=True,
        only_section=None,
        only_dataset=None,
        only_model=None,
        start_task=0,
        max_tasks=None,
        mem_limit_mb=2048,
        task_timeout=60,
        output_html=None,
        include_sbom=False,
    )
    values.update(changes)
    return argparse.Namespace(**values)


def task(n=300, model="hug_op_adaptive_full"):
    t = dict(dataset="sparse_nonlinear", section="n_scaling", model=model, n=n, p=8)
    t["key"] = d.task_key(t)
    return t


def successful(t):
    return {**t, "status": "ok", "auc": 0.8, "fit_s": 0.1}


def test_chunked_normal_stream_matches_full_generation():
    a = np.random.default_rng(17)
    b = np.random.default_rng(17)
    expected = a.normal(0, 1, size=(501, 13)).astype(np.float32)
    np.testing.assert_array_equal(d.normal_float32(b, 501, 13, block_rows=17), expected)
    np.testing.assert_array_equal(a.normal(size=20), b.normal(size=20))


def test_stages_include_requested_sizes_and_stable_keys():
    tasks = d.make_tasks(False)
    for cap in (10**6, 10**7, 3 * 10**7, 5 * 10**7):
        selected = d.apply_size_caps(tasks, max_n=cap, max_p=None)
        assert any(t["n"] == cap and t["dataset"] == "sparse_nonlinear" for t in selected)
        assert (
            max(
                t["n"]
                for t in selected
                if t["dataset"] == "threshold_grid" and t["section"] == "n_scaling"
            )
            == cap // 10
        )
    assert len({t["key"] for t in tasks}) == len(tasks)


def test_same_memory_budget_for_every_model():
    assert len({d.memory_estimate_mb(task(model=m)) for m in d.MODELS}) == 1
    assert d.index_capacity_exceeded(task(n=10**10))
    assert not d.index_capacity_exceeded(task(n=10**6))


def test_available_memory_caps_user_limit(monkeypatch):
    monkeypatch.setattr(d, "available_memory_mb", lambda: 1000)
    assert d.effective_mem_limit_mb(-1) == 800
    assert d.effective_mem_limit_mb(500) == 500
    assert d.effective_mem_limit_mb(10000) == 800


def setup_run(monkeypatch):
    monkeypatch.setattr(
        d,
        "execution_signature",
        lambda args: {"n_jobs": args.n_jobs, "lr_source_policy": args.lr_source_policy},
    )
    monkeypatch.setattr(d, "build_outputs", lambda args: None)
    monkeypatch.setattr(d, "available_memory_mb", lambda: 10000)


def test_resume_preserves_success_retries_failure_and_guards_settings(tmp_path, monkeypatch):
    setup_run(monkeypatch)
    tasks = [task(300), task(400)]
    monkeypatch.setattr(d, "selected_tasks", lambda args: tasks)
    calls = []

    def execute(t, args):
        calls.append(t["key"])
        return successful(t)

    monkeypatch.setattr(d, "run_worker_task", execute)
    args = arguments(tmp_path, max_tasks=1)
    d.run_all(args)
    path = Path(args.outdir) / "scalability_checkpoint.json"
    saved = json.loads(path.read_text())["results"][0]
    d.run_all(args)
    assert calls == [t["key"] for t in tasks]
    data = json.loads(path.read_text())
    assert data["results"][0] == saved
    data["results"][1]["auc"] = None
    d.save_ckpt(path, data)
    d.run_all(args)
    assert calls[-1] == tasks[1]["key"] and len(calls) == 3
    with pytest.raises(ValueError, match="settings differ"):
        d.run_all(arguments(tmp_path, n_jobs=2))
    assert json.loads(path.read_text())["results"][0] == saved


def test_capacity_and_plan_never_start_worker(tmp_path, monkeypatch):
    setup_run(monkeypatch)
    monkeypatch.setattr(d, "selected_tasks", lambda args: [task(10**11)])

    def prohibited(*args):
        raise AssertionError("worker started")

    monkeypatch.setattr(d, "run_worker_task", prohibited)
    args = arguments(tmp_path, plan=True)
    d.run_all(args)
    assert not Path(args.outdir).exists()
    args.plan = False
    with pytest.raises(SystemExit) as status:
        d.run_all(args)
    assert status.value.code == 2
    data = json.loads((Path(args.outdir) / "scalability_checkpoint.json").read_text())
    assert data["results"][0]["status"] == "index_capacity_exceeded"


def test_fresh_never_removes_existing_files(tmp_path):
    args = arguments(tmp_path, fresh=True, resume=False)
    root = Path(args.outdir)
    root.mkdir()
    keep = root / "keep.txt"
    keep.write_text("keep")
    with pytest.raises(ValueError, match="empty output"):
        d.run_all(args)
    assert keep.read_text() == "keep"


def test_output_lock_excludes_second_runner(tmp_path):
    with d.output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="active runner"):
            with d.output_lock(tmp_path):
                pass
    with d.output_lock(tmp_path):
        pass


def test_atomic_save_retries_and_sanitizes(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.json"
    original = d.os.replace
    calls = []

    def replace(a, b):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(errno.EACCES, "busy")
        return original(a, b)

    monkeypatch.setattr(d.os, "replace", replace)
    monkeypatch.setattr(d.time, "sleep", lambda _: None)
    d.save_ckpt(
        target,
        {
            "results": [
                {
                    "status": "error",
                    "error": "private",
                    "stderr_tail": "private",
                    "completed_at": "private",
                }
            ]
        },
    )
    assert len(calls) == 3
    assert json.loads(target.read_text()) == {"results": [{"status": "error"}]}


def test_atomic_failure_preserves_destination(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.json"
    target.write_text('{"results":[]}')

    def fail(a, b):
        raise PermissionError(errno.EACCES, "busy")

    monkeypatch.setattr(d.os, "replace", fail)
    monkeypatch.setattr(d.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        d.save_ckpt(target, {"results": [1]})
    assert json.loads(target.read_text()) == {"results": []}
    assert len(list(tmp_path.glob("*.tmp"))) == 1


@pytest.mark.parametrize(
    "model", ["hug_op_adaptive_full", "hug_op_adaptive_saga", "hug_op_adaptive_sgd", "xgb", "lgb"]
)
def test_real_worker_threads_policy_and_metrics(tmp_path, model):
    if model == "xgb":
        pytest.importorskip("xgboost")
    if model == "lgb":
        pytest.importorskip("lightgbm")
    row = d.run_worker_task(task(model=model), arguments(tmp_path))
    assert d.successful_task(row), row
    assert row["params"]["n_jobs"] == 1
    if model.startswith("hug"):
        assert row["params"]["lr_source_policy"] == "main_effect"
    assert not {"error", "stderr_tail", "stdout_tail", "completed_at"} & set(row)


def test_worker_timeout_is_retryable(tmp_path):
    row = d.run_worker_task(task(), arguments(tmp_path, task_timeout=0.001))
    assert row["status"] == "timeout" and not d.successful_task(row)


def test_capacity_envelope_and_fallback():
    for dataset, p, n, peak in [("sparse_nonlinear", 20, 15_000_000, 13.35),
                                ("threshold_grid", 200, 5_000_000, 13.86)]:
        t = dict(dataset=dataset, section="n_scaling", n=n, p=p)
        assert d.memory_estimate_mb(t) == pytest.approx(512 + 1.2 * peak * 1024)
        assert len({d.memory_estimate_mb({**t, "model": m}) for m in d.MODELS}) == 1
        t["section"] = "p_scaling"
        assert d.memory_estimate_mb(t) == pytest.approx((40*n*p + 16*n*50 + 128*n)/1048576 + 512)
    assert max(t["n"] for t in d.make_tasks(False)) == 50_000_000


def test_sparse_extension_contains_all_models():
    tasks = [t for t in d.make_tasks(False) if t["dataset"] == "sparse_nonlinear" and t["section"] == "n_scaling" and t["n"] == 15_000_000]
    assert len(tasks) == 8
    assert {t["model"] for t in tasks} == set(d.MODELS)


def test_larger_levels_include_baselines():
    for dataset, n in [("sparse_nonlinear", 20_000_000), ("threshold_grid", 7_500_000), ("threshold_grid", 10_000_000)]:
        tasks = [t for t in d.make_tasks(False) if t["dataset"] == dataset and t["n"] == n and t["section"] == "n_scaling"]
        assert len(tasks) == 8
        assert {t["model"] for t in tasks} == set(d.MODELS)

"""Public entry point limits, isolation, and actual demonstration training."""

import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "deployment" / "public-dashboard"
AVAILABLE = DEPLOY.exists() and all(
    importlib.util.find_spec(name) is not None for name in ("dash", "dash_bootstrap_components")
)
pytestmark = pytest.mark.skipif(
    not AVAILABLE, reason="Public dashboard sources or dependencies unavailable"
)
if AVAILABLE:
    sys.path.insert(0, str(DEPLOY))
    spec = importlib.util.spec_from_file_location("service", DEPLOY / "service.py")
    service = importlib.util.module_from_spec(spec)
    sys.modules["service"] = service
    spec.loader.exec_module(service)
    spec = importlib.util.spec_from_file_location("public_demo_app", DEPLOY / "app.py")
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)


def test_limits_and_isolation():
    gate = threading.Event()
    started = threading.Event()
    active = [0, 0]

    def runner(dataset, mode, timeout, topk, order):
        active[0] += 1
        active[1] = max(active)
        started.set()
        assert gate.wait(5)
        active[0] -= 1
        return {page: dataset for page in service.PAGES}

    jobs = service.Coordinator(runner, cooldown=0)
    try:
        owners = [jobs.session() for _ in range(4)]
        first = jobs.submit(owners[0], "credit", "lr")
        assert started.wait(2)
        with pytest.raises(ValueError, match="already"):
            jobs.submit(owners[0], "credit", "lr")
        second = jobs.submit(owners[1], "churn", "rpte")
        third = jobs.submit(owners[2], "claims", "lr")
        with pytest.raises(ValueError, match="queue is full"):
            jobs.submit(owners[3], "credit", "lr")
        assert jobs.snapshot(owners[1], "Overview")[3] is None
        gate.set()
        for future in (first, second, third):
            future.result(timeout=5)
        assert active[1] == 1
        for owner, dataset in zip(owners, ("credit", "churn", "claims")):
            assert jobs.snapshot(owner, "Overview")[3] == dataset
        with pytest.raises(ValueError, match="expired"):
            jobs.snapshot("foreign", "Overview")
    finally:
        gate.set()
        jobs.close()


def test_expiry_and_stale_result():
    gate = threading.Event()
    jobs = service.Coordinator(lambda *args: gate.wait(3) and {"Overview": "old"}, cooldown=0)
    try:
        owner = jobs.session()
        future = jobs.submit(owner, "credit", "lr")
        with jobs.lock:
            jobs.sessions[owner]["seen"] -= 1000
        replacement = jobs.session(owner)
        assert replacement != owner
        gate.set()
        future.result(5)
        assert jobs.snapshot(replacement, "Overview")[3] is None
        assert jobs.outstanding == 0
    finally:
        gate.set()
        jobs.close()


def test_background_polling_does_not_extend_session():
    jobs = service.Coordinator()
    try:
        owner = jobs.session()
        jobs.sessions[owner]["seen"] -= 30
        seen = jobs.sessions[owner]["seen"]
        assert jobs.session(owner, touch=False) == owner
        jobs.snapshot(owner, "Overview")
        assert jobs.sessions[owner]["seen"] == seen
        jobs.session(owner)
        assert jobs.sessions[owner]["seen"] > seen
        jobs.sessions[owner]["seen"] -= 1000
        assert jobs.session(owner, touch=False) == owner
        assert owner not in jobs.sessions
        with pytest.raises(ValueError, match="expired"):
            jobs.snapshot(owner, "Overview")
    finally:
        jobs.close()


@pytest.mark.parametrize("dataset,mode", [("../../secret", "lr"), ([], "lr"), ("credit", "exec")])
def test_invalid_config(dataset, mode):
    with pytest.raises(ValueError):
        service.validate(dataset, mode)


@pytest.mark.parametrize(
    "failure,status",
    [
        (RuntimeError("private path"), "failed"),
        (subprocess.TimeoutExpired("private", 1), "timed out"),
    ],
)
def test_failure_recovery(failure, status):
    def runner(*args):
        raise failure

    jobs = service.Coordinator(runner, cooldown=0)
    try:
        owner = jobs.session()
        jobs.submit(owner, "credit", "lr").result(5)
        assert jobs.snapshot(owner, "Overview")[0] == status
        assert jobs.outstanding == 0
        jobs.runner = lambda *args: {"Overview": "ok"}
        jobs.submit(owner, "credit", "lr").result(5)
        assert jobs.snapshot(owner, "Overview")[3] == "ok"
    finally:
        jobs.close()


def test_capacity_cooldown_and_payload():
    jobs = service.Coordinator(
        lambda *args: {"Overview": "x" * (service.MAX_RESULT_BYTES + 1)}, capacity=1
    )
    try:
        owner = jobs.session()
        with pytest.raises(ValueError, match="busy"):
            jobs.session()
        jobs.submit(owner, "credit", "lr").result(5)
        assert jobs.snapshot(owner, "Overview")[0] == "failed"
        with pytest.raises(ValueError, match="30 seconds"):
            jobs.submit(owner, "credit", "lr")
    finally:
        jobs.close()


def test_http_and_actual_callback():
    jobs = service.Coordinator(lambda *args: {page: "isolated result" for page in service.PAGES})
    app = app_module.create_app(coordinator=jobs, secret="s" * 32, testing=True)
    try:
        client = app.server.test_client()
        stranger = app.server.test_client()
        assert client.get("/healthz").json == {"status": "ok"}
        for path in ("/", "/_dash-layout", "/_dash-dependencies"):
            response = client.get(path)
            assert response.status_code == 200
            assert response.headers["Cache-Control"] == "no-store"
        assert stranger.get("/").status_code == 200
        assert client.post("/_dash-update-component").status_code == 403
        output = next(iter(app.callback_map))
        payload = dict(
            output=output,
            outputs=[
                {"id": "demo-status", "property": "children"},
                {"id": "demo-content", "property": "children"},
                {"id": "demo-rendered", "property": "data"},
            ],
            changedPropIds=["demo-train.n_clicks"],
            inputs=[
                {"id": "demo-train", "property": "n_clicks", "value": 1},
                {"id": "demo-poll", "property": "n_intervals", "value": 0},
                {"id": "demo-page", "property": "value", "value": "Overview"},
            ],
            state=[
                {"id": "demo-dataset", "property": "value", "value": "credit"},
                {"id": "demo-mode", "property": "value", "value": "lr"},
                {"id": "demo-rendered", "property": "data", "value": None},
                {"id": "demo-topk", "property": "value", "value": 20},
                {"id": "demo-order", "property": "value", "value": 2},
            ],
        )
        headers = {"Origin": "http://localhost"}
        response = client.post("/_dash-update-component", json=payload, headers=headers)
        assert response.status_code == 200
        for _ in range(100):
            if jobs.outstanding == 0:
                break
            time.sleep(0.01)
        payload["changedPropIds"] = ["demo-poll.n_intervals"]
        response = client.post("/_dash-update-component", json=payload, headers=headers)
        assert "isolated result" in json.dumps(
            response.json["response"]["demo-content"]["children"]
        )
        marker = response.json["response"]["demo-rendered"]["data"]
        payload["state"][2]["value"] = marker
        response = client.post("/_dash-update-component", json=payload, headers=headers)
        assert "demo-content" not in response.json["response"]
        assert "demo-rendered" not in response.json["response"]
        with client.session_transaction() as state:
            owner = state["owner"]
        jobs.sessions[owner]["seen"] -= 1000
        response = client.post("/_dash-update-component", json=payload, headers=headers)
        assert "Session expired" in response.get_data(as_text=True)
        assert owner not in jobs.sessions
        payload["state"][2]["value"] = None
        response = stranger.post("/_dash-update-component", json=payload, headers=headers)
        assert "isolated result" not in response.get_data(as_text=True)
        assert (
            client.post(
                "/_dash-update-component",
                data="x" * 20000,
                content_type="application/json",
                headers=headers,
            ).status_code
            == 413
        )
    finally:
        jobs.close()


def test_secret_required(monkeypatch):
    monkeypatch.delenv("HUGIML_DEMO_SECRET", raising=False)
    with pytest.raises(ValueError, match="32"):
        app_module.create_app()


@pytest.mark.parametrize("dataset", ("credit", "churn", "claims"))
@pytest.mark.parametrize("mode", ("auto", "lr", "rpte"))
@pytest.mark.parametrize("topk", (20, 50, 100))
@pytest.mark.parametrize("order", (1, 2))
def test_real_worker(dataset, mode, topk, order, monkeypatch):
    source = DEPLOY.parents[1] / "src"
    monkeypatch.setenv("PYTHONPATH", str(source))
    pages = service.run_job(dataset, mode, 90, topk, order)
    assert set(pages) == set(service.PAGES) | {"__pruning", "__cases"}
    text = json.dumps(pages)
    assert "Traceback" not in text
    assert len(text.encode()) <= service.MAX_RESULT_BYTES
    inventory = find_component(pages["Pattern Inventory"], "pat-inv")["data"]
    monitoring = find_component(pages["Monitoring"], "mo-pat")["data"]
    assert len(inventory) == len(pages["__pruning"]["patterns"]) > 0
    for row in monitoring:
        pattern = inventory[row["rank"] - 1]
        assert row["conjunction"] == pattern["pattern"]
        assert row["activation_rate"] == pytest.approx(pattern["support"], abs=0.000051)
    from case_review import unpack

    cases = pages["__cases"]
    assert len(cases) == pages["__pruning"]["train_rows"]
    assert unpack(cases[0]) != unpack(cases[-1])


def find_component(node, key):
    if isinstance(node, dict):
        if node.get("props", {}).get("id") == key:
            return node["props"]
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = find_component(child, key)
        if found is not None:
            return found
    return None


def test_case_navigation_callback_and_isolation():
    import base64
    import zlib

    cases = [
        base64.b64encode(zlib.compress(json.dumps({"case": n}).encode())).decode() for n in range(3)
    ]
    jobs = service.Coordinator(lambda *args: {"Overview": "ready", "__cases": cases}, cooldown=0)
    app = app_module.create_app(coordinator=jobs, secret="s" * 32, testing=True)
    try:
        client = app.server.test_client()
        client.get("/")
        with client.session_transaction() as state:
            owner = state["owner"]
        jobs.submit(owner, "credit").result(5)
        output = next(key for key in app.callback_map if key.startswith("..demo-case-body"))
        properties = [
            ("demo-case-body", "children"),
            ("demo-case-index", "value"),
            ("demo-case-index", "max"),
            ("demo-case-index", "disabled"),
            ("demo-case-prev", "disabled"),
            ("demo-case-next", "disabled"),
            ("demo-case-count", "children"),
            ("demo-case-revision", "data"),
            ("demo-case-controls", "style"),
        ]
        payload = dict(
            output=output,
            outputs=[{"id": i, "property": p} for i, p in properties],
            inputs=[
                {"id": "demo-case-index", "property": "value", "value": 0},
                {"id": "demo-case-prev", "property": "n_clicks", "value": 0},
                {"id": "demo-case-next", "property": "n_clicks", "value": 0},
                {"id": "demo-page", "property": "value", "value": "Case Review"},
                {"id": "demo-rendered", "property": "data", "value": None},
            ],
            state=[{"id": "demo-case-revision", "property": "data", "value": None}],
            changedPropIds=["demo-page.value"],
        )

        def update(trigger, index):
            payload["inputs"][0]["value"] = index
            payload["changedPropIds"] = [trigger]
            response = client.post(
                "/_dash-update-component", json=payload, headers={"Origin": "http://localhost"}
            )
            assert response.status_code == 200
            data = response.json["response"]
            if "demo-case-revision" in data:
                payload["state"][0]["value"] = data["demo-case-revision"]["data"]
            return data

        assert update("demo-page.value", 0)["demo-case-body"]["children"] == {"case": 0}
        assert update("demo-case-next.n_clicks", 0)["demo-case-body"]["children"] == {"case": 1}
        assert update("demo-case-index.value", 2)["demo-case-next"]["disabled"]
        assert update("demo-case-prev.n_clicks", 2)["demo-case-body"]["children"] == {"case": 1}
        for invalid in (-1, 3, 1.5, True, "1", None):
            with pytest.raises(ValueError, match="row index"):
                jobs.case(owner, invalid, 1)
        foreign = jobs.session()
        assert jobs.case(foreign, 0, 1)[3] is None
        jobs.submit(owner, "churn").result(5)
        assert update("demo-rendered.data", 2)["demo-case-body"]["children"] == {"case": 0}
    finally:
        jobs.close()


def test_timeout_stops_worker(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(DEPLOY.parents[1] / "src"))
    with pytest.raises(subprocess.TimeoutExpired):
        service.run_job("credit", "lr", 0.001)


@pytest.mark.parametrize(
    "topk,order", [(21, 2), (10000, 2), (20, 3), (20, True), ("20", 2), (20, 2.0)]
)
def test_invalid_bounds(topk, order):
    with pytest.raises(ValueError):
        service.validate("credit", "auto", topk, order)


def test_numeric_presentation_preserves_values():
    spec = importlib.util.spec_from_file_location("demo_presentation", DEPLOY / "presentation.py")
    presentation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(presentation)
    value = 0.123456789
    table = {
        "type": "DataTable",
        "props": {"data": [{"score": value, "n": 12}], "columns": [{"id": "score"}, {"id": "n"}]},
    }
    result = presentation.present(table)
    assert result["props"]["data"][0]["score"] == value
    assert result["props"]["columns"][0]["format"]["specifier"] == ".3~g"
    assert presentation.text(value) == "0.123"
    assert presentation.text(0.0000001234) == "1.23e-07"
    assert presentation.text("x <= 0.123456789") == "x <= 0.123456789"
    assert presentation.text("0.770666666") == "0.771"
    graph = {"type": "Graph", "props": {"figure": {"data": [{"y": [value]}], "layout": {}}}}
    result = presentation.present(graph)
    assert result["props"]["figure"]["data"][0]["y"][0] == value
    assert result["props"]["figure"]["layout"]["yaxis"]["hoverformat"] == ".3~g"


def test_selection_uses_validation_score(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(DEPLOY))
    spec = importlib.util.spec_from_file_location("demo_worker", DEPLOY / "worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    candidates = [
        {"artifact": {"tuning_result": SimpleNamespace(best_score_=score)}}
        for score in (0.71, 0.79)
    ]
    assert worker.select_run(candidates) is candidates[1]
    candidates[0]["artifact"]["tuning_result"].best_score_ = 0.79
    assert worker.select_run(candidates) is candidates[0]
    candidates[1]["artifact"]["tuning_result"].best_score_ = float("nan")
    with pytest.raises(ValueError):
        worker.select_run(candidates)


def test_controls_and_assets():
    jobs = service.Coordinator()
    try:
        app = app_module.create_app(coordinator=jobs, secret="s" * 32, testing=True)
        client = app.server.test_client()
        layout = client.get("/_dash-layout").json
        serialized = json.dumps(layout)
        assert '"id": "demo-topk"' in serialized
        assert '"id": "demo-order"' in serialized
        assert '"value": "auto"' in serialized
        assert client.get("/assets/studio.css").status_code == 200
    finally:
        jobs.close()


def test_headline_rounding_uses_raw_evidence():
    spec = importlib.util.spec_from_file_location("demo_presentation", DEPLOY / "presentation.py")
    presentation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(presentation)
    raw = 0.8144999
    card = {
        "type": "Div",
        "props": {
            "className": "mc",
            "children": [
                {"type": "Div", "props": {"children": "ROC-AUC"}},
                {"type": "Div", "props": {"children": "0.8145"}},
            ],
        },
    }
    table = {
        "type": "DataTable",
        "props": {
            "id": "pf-met",
            "data": [{"metric": "roc_auc", "value": raw}],
            "columns": [{"id": "metric"}, {"id": "value"}],
        },
    }
    result = presentation.format_pages([card, table])
    assert result[0]["props"]["children"][1]["props"]["children"] == "0.814"
    assert result[1]["props"]["data"][0]["value"] == raw


def test_context_help_preserves_values_and_has_unique_targets():
    import guidance

    metrics = [
        {"type": "Div", "props": {"className": "mc-l", "children": "ROC-AUC"}},
        {"type": "Div", "props": {"className": "mc-l", "children": "Brier"}},
    ]
    table = {
        "type": "DataTable",
        "props": {"columns": [{"id": "threshold"}], "data": [{"threshold": 0.123456789}]},
    }
    result = guidance.add_guidance(metrics + [table])
    targets = []

    def walk(node):
        if isinstance(node, list):
            for child in node:
                walk(child)
        elif isinstance(node, dict):
            if node.get("type") == "Tooltip":
                targets.append(node["props"]["target"])
                assert node["props"]["trigger"] == "hover focus"
            for child in node.values():
                walk(child)

    walk(result)
    assert len(targets) == len(set(targets)) == 2
    assert result[2]["props"]["data"][0]["threshold"] == 0.123456789
    assert "threshold" in result[2]["props"]["tooltip_header"]


@pytest.mark.parametrize("dataset", ("credit", "churn", "claims"))
@pytest.mark.parametrize("mode", ("lr", "rpte"))
@pytest.mark.parametrize("kind", ("features", "patterns"))
@pytest.mark.parametrize("topk,maximum", [(20, False), (100, True)])
def test_real_pruning(dataset, mode, kind, topk, maximum, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(DEPLOY.parents[1] / "src"))
    pages = service.run_job(dataset, mode, 90, topk, 2)
    workflow = pages["__pruning"]
    count = (3 if kind == "features" else 5) if maximum else 1
    items = workflow["features"][:count] if kind == "features" else list(range(count))
    result = service.run_job(
        dataset,
        mode,
        90,
        topk,
        2,
        pruning={"fingerprint": workflow["fingerprint"], "kind": kind, "items": items},
    )
    summary = result["summary"]
    assert summary["baseline"] == workflow["baseline"]
    assert summary["selected_mode"] == mode
    assert 0 <= summary["changed_predictions"] <= workflow["holdout_rows"]
    if kind == "features":
        assert summary["pruned"]["inputs"] == summary["baseline"]["inputs"] - count
        assert items[0] not in summary["retained_inputs"]
    else:
        assert summary["pruned"]["patterns"] == summary["baseline"]["patterns"] - count
    assert "Traceback" not in json.dumps(result)


def demo_workflow():
    return {
        "fingerprint": "abc",
        "features": ["a", "b", "c", "d"],
        "patterns": [{"label": str(i), "value": i} for i in range(7)],
    }


@pytest.mark.parametrize(
    "kind,items",
    [
        ("features", []),
        ("features", ["unknown"]),
        ("features", ["a", "a"]),
        ("features", ["a", "b", "c", "d"]),
        ("patterns", [True]),
        ("patterns", ["0"]),
        ("patterns", [-1]),
        ("patterns", [0, 1, 2, 3, 4, 5]),
        ("invalid", [0]),
    ],
)
def test_pruning_selection_limits(kind, items):
    from pruning_workflow import validate_selection

    with pytest.raises(ValueError):
        validate_selection(demo_workflow(), kind, items)


def test_pruning_preserves_original_and_rejects_stale_model(monkeypatch):
    def baseline(*args):
        return {**{page: "original" for page in service.PAGES}, "__pruning": demo_workflow()}

    jobs = service.Coordinator(baseline, cooldown=0)
    monkeypatch.setattr(
        service, "run_job", lambda *args, **kwargs: {"view": "comparison", "summary": {}}
    )
    try:
        owner, foreign = jobs.session(), jobs.session()
        jobs.submit(owner, "credit", "lr").result(5)
        revision, _ = jobs.pruning_info(owner)
        original = dict(jobs.sessions[owner]["result"])
        with pytest.raises(ValueError):
            jobs.submit_pruning(foreign, revision, "features", ["a"])
        jobs.submit_pruning(owner, revision, "features", ["a"]).result(5)
        assert jobs.snapshot(owner, "Pruning")[3] == "comparison"
        assert jobs.snapshot(owner, "Overview")[3] == "original"
        assert jobs.sessions[owner]["result"] == original
        jobs.reset_pruning(owner, revision)
        assert jobs.snapshot(owner, "Pruning")[3] == "original"
        jobs.submit(owner, "claims", "rpte").result(5)
        with pytest.raises(ValueError, match="model changed"):
            jobs.submit_pruning(owner, revision, "features", ["a"])
    finally:
        jobs.close()


def test_pruning_failure_and_expiry_preserve_isolation(monkeypatch):
    def baseline(*args):
        return {**{page: "original" for page in service.PAGES}, "__pruning": demo_workflow()}

    gate = threading.Event()
    entered = threading.Event()

    def compute(*args, **kwargs):
        entered.set()
        assert gate.wait(5)
        raise RuntimeError("private diagnostic")

    jobs = service.Coordinator(baseline, cooldown=0)
    monkeypatch.setattr(service, "run_job", compute)
    try:
        owner = jobs.session()
        jobs.submit(owner, "credit", "lr").result(5)
        revision, _ = jobs.pruning_info(owner)
        future = jobs.submit_pruning(owner, revision, "patterns", [0])
        assert entered.wait(2)
        with pytest.raises(ValueError):
            jobs.submit_pruning(owner, revision, "patterns", [1])
        with pytest.raises(ValueError):
            jobs.reset_pruning(owner, revision)
        gate.set()
        future.result(5)
        assert jobs.snapshot(owner, "Pruning")[0] == "pruning failed"
        assert jobs.snapshot(owner, "Overview")[3] == "original"
        gate.clear()
        entered.clear()
        future = jobs.submit_pruning(owner, revision, "patterns", [1])
        assert entered.wait(2)
        with jobs.lock:
            jobs.sessions[owner]["seen"] -= 1000
        replacement = jobs.session(owner)
        gate.set()
        future.result(5)
        assert jobs.pruning_info(replacement) == (None, None)
        assert jobs.outstanding == 0
    finally:
        gate.set()
        jobs.close()

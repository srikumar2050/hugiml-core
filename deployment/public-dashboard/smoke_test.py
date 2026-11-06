"""Exercise the deployed HTTP entry point using only the standard library."""

import http.cookiejar
import json
import sys
import time
import urllib.request


def check(base, dataset="credit", mode="auto"):
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )

    def get(path):
        with opener.open(base + path, timeout=15) as response:
            return response.read()

    for attempt in range(60):
        try:
            assert json.loads(get("/healthz"))["status"] == "ok"
            break
        except (OSError, AssertionError):
            if attempt == 59:
                raise
            time.sleep(1)
    get("/")
    layout = json.loads(get("/_dash-layout"))
    dependencies = json.loads(get("/_dash-dependencies"))
    output = dependencies[0]["output"]
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
            {"id": "demo-dataset", "property": "value", "value": dataset},
            {"id": "demo-mode", "property": "value", "value": mode},
            {"id": "demo-rendered", "property": "data", "value": None},
            {"id": "demo-topk", "property": "value", "value": 100},
            {"id": "demo-order", "property": "value", "value": 2},
        ],
    )

    def update(body=None):
        request = urllib.request.Request(
            base + "/_dash-update-component",
            data=json.dumps(payload if body is None else body).encode(),
            headers={"Content-Type": "application/json", "Origin": base},
        )
        with opener.open(request, timeout=15) as response:
            return json.load(response)["response"]

    started = time.monotonic()
    result = update()
    payload["changedPropIds"] = ["demo-poll.n_intervals"]
    deadline = time.monotonic() + 150
    while "Model ready" not in json.dumps(result["demo-status"]["children"]):
        assert not any(
            word in json.dumps(result["demo-status"]["children"])
            for word in ("could not complete", "time limit")
        ), f"{dataset}: {json.dumps(result['demo-status']['children'])}"
        if time.monotonic() > deadline:
            raise TimeoutError("Training deadline exceeded")
        time.sleep(1)
        result = update()
    assert isinstance(result["demo-content"]["children"], dict)
    original_view = result["demo-content"]["children"]

    def find(node, key):
        if isinstance(node, list):
            for item in node:
                found = find(item, key)
                if found is not None:
                    return found
        elif isinstance(node, dict):
            if node.get("props", {}).get("id") == key:
                return node["props"]
            for item in node.values():
                found = find(item, key)
                if found is not None:
                    return found
        return None

    pages = [option["value"] for option in find(layout, "demo-page")["options"]]
    for page in pages:
        payload["inputs"][2]["value"] = page
        payload["changedPropIds"] = ["demo-page.value"]
        result = update()
        assert "Traceback" not in json.dumps(result), page
        assert result["demo-content"]["children"], page
        if page == "Pattern Inventory":
            assert find(result, "pat-inv")["data"]
    case_output = next(
        item["output"] for item in dependencies if item["output"].startswith("..demo-case-body")
    )
    case_properties = [
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
    case_payload = dict(
        output=case_output,
        outputs=[{"id": key, "property": prop} for key, prop in case_properties],
        changedPropIds=["demo-page.value"],
        inputs=[
            {"id": "demo-case-index", "property": "value", "value": 0},
            {"id": "demo-case-prev", "property": "n_clicks", "value": 0},
            {"id": "demo-case-next", "property": "n_clicks", "value": 0},
            {"id": "demo-page", "property": "value", "value": "Case Review"},
            {"id": "demo-rendered", "property": "data", "value": None},
        ],
        state=[{"id": "demo-case-revision", "property": "data", "value": None}],
    )
    first_case = update(case_payload)
    assert first_case["demo-case-index"]["max"] == 19
    assert first_case["demo-case-count"]["children"] == "20 sampled training cases"
    case_payload["state"][0]["value"] = first_case["demo-case-revision"]["data"]
    case_payload["inputs"][0]["value"] = first_case["demo-case-index"]["max"]
    case_payload["changedPropIds"] = ["demo-case-index.value"]
    last_case = update(case_payload)
    assert last_case["demo-case-next"]["disabled"]
    assert first_case["demo-case-body"] != last_case["demo-case-body"]
    control_payload = dict(
        output=next(
            item["output"] for item in dependencies if "demo-prune-items.options" in item["output"]
        ),
        outputs=[
            {"id": name, "property": prop}
            for name, prop in (
                ("demo-prune-items", "options"),
                ("demo-prune-items", "value"),
                ("demo-prune-notice", "children"),
                ("demo-model-revision", "data"),
                ("demo-prune-run", "disabled"),
                ("demo-prune-reset", "disabled"),
                ("demo-prune-controls", "style"),
            )
        ],
        changedPropIds=["demo-page.value"],
        inputs=[
            {"id": name, "property": prop, "value": value}
            for name, prop, value in (
                ("demo-poll", "n_intervals", 1),
                ("demo-prune-kind", "value", "features"),
                ("demo-prune-run", "n_clicks", 0),
                ("demo-prune-reset", "n_clicks", 0),
                ("demo-page", "value", "Pruning"),
                ("demo-prune-items", "value", []),
            )
        ],
        state=[
            {"id": "demo-model-revision", "property": "data", "value": None},
        ],
    )
    controls = update(control_payload)
    control_payload["inputs"][5]["value"] = [controls["demo-prune-items"]["options"][0]["value"]]
    control_payload["state"][0]["value"] = controls["demo-model-revision"]["data"]
    remaining = 31 - (time.monotonic() - started)
    if remaining > 0:
        time.sleep(remaining)
    control_payload["changedPropIds"] = ["demo-prune-run.n_clicks"]
    control_payload["inputs"][2]["value"] = 1
    controls = update(control_payload)
    assert "Comparison requested" in json.dumps(controls)
    payload["inputs"][2]["value"] = "Pruning"
    payload["changedPropIds"] = ["demo-poll.n_intervals"]
    deadline = time.monotonic() + 150
    while True:
        result = update()
        status = json.dumps(result["demo-status"]["children"])
        if "Pruning comparison ready" in status:
            break
        assert "could not complete" not in status and "time limit" not in status, status
        if time.monotonic() > deadline:
            raise TimeoutError("Pruning deadline exceeded")
        time.sleep(1)
    assert "Original model preserved" in json.dumps(result["demo-content"])
    control_payload["changedPropIds"] = ["demo-prune-reset.n_clicks"]
    control_payload["inputs"][3]["value"] = 1
    assert "Comparison cleared" in json.dumps(update(control_payload))
    payload["inputs"][2]["value"] = "Overview"
    assert update()["demo-content"]["children"] == original_view
    print(
        "HTTP startup, all governance routes, pruning comparison, reset, and original model preservation passed."
    )


if __name__ == "__main__":
    check(sys.argv[1].rstrip("/"), *(sys.argv[2:4]))

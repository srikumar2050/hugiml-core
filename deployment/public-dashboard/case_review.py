"""Bounded, model-free case navigation using prepared presentation evidence."""

import base64
import json
import random
import zlib

MAX_CASE_BYTES = 128 * 1024
MAX_REVIEW_CASES = 100


def sample_indices(labels, limit=MAX_REVIEW_CASES):
    """Select row positions proportionally across observed target classes."""
    if type(limit) is not int or limit < 1:
        raise ValueError("Case sample size must be a positive integer.")
    if len(labels) <= limit:
        return list(range(len(labels)))
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    if len(groups) > limit:
        raise ValueError("Case sample size must cover every observed class.")
    targets = {label: limit * len(group) / len(labels) for label, group in groups.items()}
    counts = {label: max(1, int(target)) for label, target in targets.items()}
    while sum(counts.values()) > limit:
        label = max((k for k in counts if counts[k] > 1), key=lambda k: counts[k] - targets[k])
        counts[label] -= 1
    while sum(counts.values()) < limit:
        label = max((k for k in counts if counts[k] < len(groups[k])), key=lambda k: targets[k] - counts[k])
        counts[label] += 1
    rng = random.Random(2026)
    selected = []
    for label, group in groups.items():
        selected.extend(rng.sample(group, counts[label]))
    return sorted(selected)


def prepare(context):
    from guidance import add_guidance
    from plotly.utils import PlotlyJSONEncoder
    from presentation import format_pages

    from hugiml.dashboard.dash_components.pages.prediction import render_case

    cases = []
    for index in sample_indices(context["y"]):
        view = json.loads(json.dumps(render_case(context, index), cls=PlotlyJSONEncoder))
        raw = json.dumps(add_guidance(format_pages(view))).encode("utf-8")
        if len(raw) > MAX_CASE_BYTES:
            raise ValueError("Case evidence exceeds the demonstration size limit.")
        cases.append(base64.b64encode(zlib.compress(raw)).decode("ascii"))
    return cases


def unpack(value):
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(value, validate=True), MAX_CASE_BYTES + 1)
    if len(raw) > MAX_CASE_BYTES or not decoder.eof or decoder.unused_data:
        raise ValueError("Invalid case evidence.")
    return json.loads(raw)

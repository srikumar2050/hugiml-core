"""Bounded, model-free case navigation using prepared presentation evidence."""

import base64
import json
import zlib

MAX_CASE_BYTES = 128 * 1024


def prepare(context):
    from guidance import add_guidance
    from plotly.utils import PlotlyJSONEncoder
    from presentation import format_pages

    from hugiml.dashboard.dash_components.pages.prediction import render_case

    cases = []
    for index in range(len(context["X"])):
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

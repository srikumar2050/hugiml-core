"""Row-specific pattern evidence returned by the inference service."""

import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

pytest.importorskip("fastapi")
pytest.importorskip("slowapi")
from starlette.requests import Request


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("HUGIML_API_KEYS", "test-token")
    monkeypatch.setenv("HUGIML_MODEL_HMAC_KEY", "ab" * 32)
    path = Path(__file__).resolve().parents[1] / "docker" / "server.py"
    spec = importlib.util.spec_from_file_location("inference_server", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class Model:
    def predict(self, X):
        return np.array([0, 1, 0])[: len(X)]

    def predict_proba(self, X):
        return np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])[: len(X)]

    def transform_patterns(self, X):
        return csr_matrix([[1, 0], [0, 1], [0, 0]])[: len(X)]

    def get_hug_features(self):
        return ["a=low", "a=high"]


@pytest.mark.parametrize("return_proba", [True, False])
def test_explanations_match_each_input_row(server, return_proba):
    server._clf = Model()
    request = Request(
        {
            "type": "http",
            "app": server.app,
            "client": ("127.0.0.1", 1),
            "path": "/predict/explain",
            "headers": [],
        }
    )
    payload = server.PredictRequest(
        instances=[{"a": 0}, {"a": 1}, {"a": 2}], return_proba=return_proba
    )
    result = asyncio.run(server.predict_explain(request=request, req=payload, _=None))
    assert result.predictions == [0, 1, 0]
    assert result.explanations == [
        {"patterns": ["a=low"]},
        {"patterns": ["a=high"]},
        {"patterns": []},
    ]
    assert (result.probabilities is not None) == return_proba


def test_explanations_preserve_empty_pattern_model(server):
    model = Model()
    model.transform_patterns = lambda X: csr_matrix((len(X), 0))
    model.get_hug_features = lambda: []
    server._clf = model
    request = Request(
        {
            "type": "http",
            "app": server.app,
            "client": ("127.0.0.1", 1),
            "path": "/predict/explain",
            "headers": [],
        }
    )
    result = asyncio.run(
        server.predict_explain(
            request=request, req=server.PredictRequest(instances=[{"a": 0}]), _=None
        )
    )
    assert result.explanations == [{"patterns": []}]

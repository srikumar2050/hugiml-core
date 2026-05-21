#!/usr/bin/env python3
# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastAPI inference server for HUGIMLClassifierNative.

Environment variables
---------------------
HUGIML_MODEL_PATH           Path to the .hugiml model file (required).
HUGIML_HOST                 Bind host  (default: 127.0.0.1)
HUGIML_PORT                 Bind port  (default: 8080)
HUGIML_WORKERS              Uvicorn worker count (default: 1)
HUGIML_OTEL_ENABLED         Enable OpenTelemetry tracing (default: false)
HUGIML_PROMETHEUS_ENABLED   Expose /metrics endpoint (default: true)
HUGIML_API_KEYS             Comma-separated list of valid bearer tokens.
                            Required unless HUGIML_AUTH_DISABLED=1.
                            The server will refuse to start if this is unset
                            and auth is not explicitly disabled.
HUGIML_AUTH_DISABLED        Set to "1" to bypass auth (development only).
HUGIML_RATE_LIMIT           Requests per minute per client (default: 60).
HUGIML_MAX_BODY_BYTES       Maximum request body size in bytes (default: 1MB).
HUGIML_MODEL_HMAC_KEY       Hex HMAC key forwarded to the serialization layer.

Endpoints
---------
GET  /health           Liveness probe — returns 200 when the process is alive
                       and the model loaded without error; 503 otherwise.
GET  /ready            Readiness probe — returns 200 when warm.
POST /predict          Batch prediction endpoint (JSON).
POST /predict/explain  Prediction + top pattern explanations.
GET  /model/info       Redacted model summary (no raw feature data).
GET  /metrics          Prometheus metrics (if enabled).
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import PlainTextResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from hugiml.serialization import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hugiml.server")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = Path(os.environ.get("HUGIML_MODEL_PATH", "/models/model.hugiml"))
_RATE_LIMIT = os.environ.get("HUGIML_RATE_LIMIT", "60")
_MAX_BODY_BYTES = int(os.environ.get("HUGIML_MAX_BODY_BYTES", str(1 * 1024 * 1024)))
_AUTH_DISABLED = os.environ.get("HUGIML_AUTH_DISABLED", "0") == "1"
_ENABLE_DOCS = os.environ.get("HUGIML_ENABLE_DOCS", "false").lower() == "true"

_valid_keys: frozenset[str] = frozenset(
    k.strip() for k in os.environ.get("HUGIML_API_KEYS", "").split(",") if k.strip()
)
if not _valid_keys and not _AUTH_DISABLED:
    raise RuntimeError(
        "HUGIML_API_KEYS is not set and HUGIML_AUTH_DISABLED is not '1'.  "
        "The server cannot start in this state: every request would be rejected "
        "with HTTP 401.  Set HUGIML_API_KEYS to a comma-separated list of bearer "
        "tokens, or set HUGIML_AUTH_DISABLED=1 for local development only."
    )
if _AUTH_DISABLED:
    logger.warning(
        "Authentication is disabled (HUGIML_AUTH_DISABLED=1). This must not be used in production."
    )

# Validate HMAC configuration.  In production HUGIML_REQUIRE_MODEL_HMAC=true
# must be paired with a strong HUGIML_MODEL_HMAC_KEY, otherwise model files
# could be tampered with or substituted without detection.
_require_hmac_env = os.environ.get("HUGIML_REQUIRE_MODEL_HMAC", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_hmac_key_set = bool(os.environ.get("HUGIML_MODEL_HMAC_KEY", "").strip())
if _require_hmac_env and not _hmac_key_set:
    raise RuntimeError(
        "HUGIML_REQUIRE_MODEL_HMAC is enabled but HUGIML_MODEL_HMAC_KEY is not set.  "
        "Configure the hex-encoded HMAC key (32+ bytes recommended) before starting "
        "the server in production."
    )
if not _hmac_key_set and not _AUTH_DISABLED:
    logger.warning(
        "HUGIML_MODEL_HMAC_KEY is not set.  Model files will be loaded without "
        "authentication.  Set HUGIML_MODEL_HMAC_KEY and HUGIML_REQUIRE_MODEL_HMAC=true "
        "to reject unsigned or tampered model files."
    )

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_clf: Any = None
_explainer: Any = None
_load_time: float | None = None
_load_error: str | None = None


def _load_model_at_startup() -> None:
    global _clf, _explainer, _load_time, _load_error
    if not MODEL_PATH.exists():
        _load_error = f"Model file not found: {MODEL_PATH}"
        logger.error(_load_error)
        return
    try:
        logger.info("Loading model from %s …", MODEL_PATH)
        t0 = time.perf_counter()
        _clf = load_model(MODEL_PATH)
        _clf.enable_monitoring(window_size=10_000)
        try:
            from hugiml.explainability import HUGPatternExplainer

            _explainer = HUGPatternExplainer(_clf)
        except Exception:
            logger.exception("Explainer initialization failed")
            _explainer = None
        _load_time = time.perf_counter() - t0
        logger.info("Model loaded in %.3fs", _load_time)
    except Exception as exc:
        _load_error = str(exc)
        logger.exception("Model load failed: %s", exc)


def _get_model() -> Any:
    if _clf is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available.",
        )
    return _clf


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
_bearer = HTTPBearer(auto_error=False)


def _require_auth(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    if _AUTH_DISABLED:
        return
    if credentials is None or not any(
        hmac.compare_digest(credentials.credentials, key) for key in _valid_keys
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Body size enforcement middleware
# ---------------------------------------------------------------------------
async def _enforce_body_size(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > _MAX_BODY_BYTES:
                return PlainTextResponse(
                    f"Request body exceeds {_MAX_BODY_BYTES} bytes.",
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                )
        except ValueError:
            return PlainTextResponse(
                "Invalid Content-Length header.",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

    body = await request.body()
    if len(body) > _MAX_BODY_BYTES:
        return PlainTextResponse(
            f"Request body exceeds {_MAX_BODY_BYTES} bytes.",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(request.scope, receive)
    return await call_next(request)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="HUG-IML Inference Server",
    description=(
        "High-performance interpretable rule-based ML inference server "
        "powered by HUGIMLClassifierNative."
    ),
    version="2.1.0",
    docs_url="/docs" if _ENABLE_DOCS else None,
    redoc_url="/redoc" if _ENABLE_DOCS else None,
    openapi_url="/openapi.json" if _ENABLE_DOCS else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.middleware("http")(_enforce_body_size)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    instances: list[dict[str, Any]] = Field(
        ...,
        description="List of feature dicts. Keys must match training features.",
        min_length=1,
        max_length=1_000,
    )
    return_proba: bool = Field(
        True,
        description="If true, return class probabilities in addition to labels.",
    )


class PredictResponse(BaseModel):
    predictions: list[int | str]
    probabilities: list[list[float]] | None = None
    latency_ms: float


class ExplainResponse(BaseModel):
    predictions: list[int | str]
    probabilities: list[list[float]] | None = None
    explanations: list[dict[str, Any]]
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def _startup() -> None:
    _load_model_at_startup()


@app.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    """Liveness probe.  Returns 503 if model load failed."""
    if _load_error is not None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model load failed.",
        )
    return {"status": "ok"}


@app.get("/ready", status_code=status.HTTP_200_OK)
async def ready() -> dict[str, Any]:
    if _clf is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not yet loaded",
        )
    return {"status": "ready", "model_load_time_s": _load_time}


@app.post("/predict", response_model=PredictResponse)
@limiter.limit(f"{_RATE_LIMIT}/minute")
async def predict(
    request: Request,
    req: PredictRequest,
    _: None = Depends(_require_auth),
) -> PredictResponse:
    clf = _get_model()
    t0 = time.perf_counter()
    try:
        X = pd.DataFrame(req.instances)
        preds = clf.predict(X).tolist()
        proba = clf.predict_proba(X).tolist() if req.return_proba else None
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=400, detail="Invalid prediction request.") from exc
    latency_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        predictions=preds,
        probabilities=proba,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/explain", response_model=ExplainResponse)
@limiter.limit(f"{_RATE_LIMIT}/minute")
async def predict_explain(
    request: Request,
    req: PredictRequest,
    _: None = Depends(_require_auth),
) -> ExplainResponse:
    clf = _get_model()
    t0 = time.perf_counter()
    try:
        X = pd.DataFrame(req.instances)
        preds = clf.predict(X).tolist()
        proba = clf.predict_proba(X).tolist() if req.return_proba else None
        if _explainer is None:
            raise RuntimeError("Explainer is not available")
        raw = _explainer.explain(X)

        if hasattr(raw, "patterns"):
            explanations = [{"patterns": raw.patterns}] * len(preds)
        elif isinstance(raw, list):
            explanations = [e if isinstance(e, dict) else {"patterns": e} for e in raw]
        else:
            explanations = [{"patterns": str(raw)}] * len(preds)
    except Exception as exc:
        logger.exception("Explain error")
        raise HTTPException(status_code=400, detail="Invalid explanation request.") from exc
    latency_ms = (time.perf_counter() - t0) * 1000
    return ExplainResponse(
        predictions=preds,
        probabilities=proba,
        explanations=explanations,
        latency_ms=round(latency_ms, 3),
    )


@app.get("/model/info")
async def model_info(_: None = Depends(_require_auth)) -> dict[str, Any]:
    """Return a redacted model summary suitable for operational monitoring.

    Raw HUG feature lists are not exposed to avoid leaking model internals.
    """
    clf = _get_model()
    summary = clf.model_summary()
    return {
        "summary": summary,
        "n_patterns": len(clf.get_hug_features()),
        "n_classes": len(clf.classes_.tolist()),
        "classes": clf.classes_.tolist(),
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    """Prometheus text-format metrics (if enabled)."""
    enabled = os.environ.get("HUGIML_PROMETHEUS_ENABLED", "true").lower() == "true"
    if not enabled:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return PlainTextResponse(
            generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        return PlainTextResponse(
            "# prometheus_client not installed\n",
            media_type="text/plain",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    host = os.environ.get("HUGIML_HOST", "127.0.0.1")
    port = int(os.environ.get("HUGIML_PORT", "8080"))
    workers = int(os.environ.get("HUGIML_WORKERS", "1"))
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        workers=workers,
        access_log=True,
    )

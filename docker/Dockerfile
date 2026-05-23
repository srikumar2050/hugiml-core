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

# ---------------------------------------------------------------------------
# Stage 1 — Build wheel
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System build dependencies for the C++ extension
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Upgrade pip/setuptools/wheel to fix CVEs in base image vendored copies
RUN pip install --no-cache-dir --upgrade "pip>=26.1" "setuptools>=78.0" "wheel>=0.46.2"

# Copy only the files needed to build the wheel
COPY pyproject.toml setup.py MANIFEST.in README.md LICENSE NOTICE ./
COPY src/ ./src/
COPY native/ ./native/

RUN pip install --no-cache-dir --upgrade pip build \
    && python -m build --wheel --outdir /dist \
    && ls /dist/*.whl

# ---------------------------------------------------------------------------
# Stage 2 — Runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime: libgomp for OpenMP parallelism
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1001 hugiml

WORKDIR /app

# Upgrade pip/setuptools/wheel to fix CVEs in base image vendored copies
RUN pip install --no-cache-dir --upgrade "pip>=26.1" "setuptools>=78.0" "wheel>=0.46.2"

# Copy entire dist directory — avoids glob-matches-nothing silently copying 0 files
COPY --from=builder /dist/ /staging/

# Resolve the wheel path with find so the shell glob never reaches pip as a literal
RUN whl=$(find /staging -name "*.whl" | head -1) \
    && [ -n "$whl" ] || { echo "ERROR: no wheel found in /staging"; exit 1; } \
    && pip install --no-cache-dir "${whl}[server,telemetry]" \
    && rm -rf /staging

# Smoke-test: core + new modules import cleanly
RUN python -c "\
from hugiml import HUGIMLClassifierNative; \
from hugiml.metrics import compute_all_metrics; \
from hugiml.pruning import PatternEditor; \
from hugiml.adaptive import HUGIMLAdaptive; \
from hugiml.multiclass import MulticlassHUGReport; \
import hugiml; print('hugiml', hugiml.__version__, 'runtime image OK')"

# Copy inference server
COPY docker/server.py ./server.py

# Model path is mounted at runtime; default location for the volume mount
RUN mkdir -p /models && chown hugiml:hugiml /models

USER hugiml

# Expose HTTP port
EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENV HUGIML_MODEL_PATH=/models/model.hugiml
ENV HUGIML_OTEL_ENABLED=false
ENV HUGIML_PROMETHEUS_ENABLED=true
ENV HUGIML_REQUIRE_MODEL_HMAC=true
ENV HUGIML_ENABLE_DOCS=false

ENTRYPOINT ["python", "server.py"]

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
FROM python:3.11-slim AS builder

# System build dependencies for the C++ extension
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the files needed to build the wheel
COPY pyproject.toml setup.py MANIFEST.in README.md LICENSE NOTICE ./
COPY src/ ./src/
COPY native/ ./native/

RUN pip install --no-cache-dir --upgrade pip build \
    && python -m build --wheel --outdir /dist

# ---------------------------------------------------------------------------
# Stage 2 — Runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runtime: libgomp for OpenMP parallelism
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1001 hugiml

WORKDIR /app

# Install the wheel from the build stage
COPY --from=builder /dist/*.whl /staging/
RUN pip install --no-cache-dir /staging/*.whl[server,telemetry] \
    && rm -rf /staging

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

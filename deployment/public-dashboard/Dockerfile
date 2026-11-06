FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 HUGIML_BUILD_JOBS=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml setup.py MANIFEST.in README.md LICENSE NOTICE ./
COPY src ./src
COPY native ./native
COPY THIRD_PARTY_LICENSES ./THIRD_PARTY_LICENSES
RUN pip wheel --no-deps . --wheel-dir /wheels

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY deployment/public-dashboard/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir /wheels/*.whl -r requirements.txt && rm -rf /wheels && useradd --create-home --uid 10001 demo
COPY deployment/public-dashboard/app.py deployment/public-dashboard/service.py deployment/public-dashboard/worker.py deployment/public-dashboard/presentation.py deployment/public-dashboard/guidance.py deployment/public-dashboard/pruning_workflow.py ./
COPY deployment/public-dashboard/assets ./assets
COPY deployment/public-dashboard/case_review.py ./
USER demo
EXPOSE 10000
CMD ["sh", "-c", "exec gunicorn 'app:application()' --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 180 --graceful-timeout 130 --worker-tmp-dir /tmp"]

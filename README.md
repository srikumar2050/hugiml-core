# hugiml-core

> **High-performance interpretable rule-based ML infrastructure** built on the
> HUG-IML algorithm published in IEEE Access (2024).

[![CI](https://github.com/srikumar2050/hugiml-core/actions/workflows/ci.yml/badge.svg)](https://github.com/srikumar2050/hugiml-core/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hugiml-core.svg)](https://pypi.org/project/hugiml-core/)
[![Python](https://img.shields.io/pypi/pyversions/hugiml-core.svg)](https://pypi.org/project/hugiml-core/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2024.3455563-blue)](https://doi.org/10.1109/ACCESS.2024.3455563)

---

## What Is HUG-IML?

The **High Utility Gain Interpretable Machine Learning (HUG-IML)** framework
extracts *High Utility Gain patterns* from labelled tabular data, transforms
the input into a binary pattern-presence matrix, and fits an interpretable
downstream classifier (logistic regression by default) on that matrix.
The resulting patterns are human-readable and serve as the primary source of
model explanations, making the system suitable for regulated domains such as
credit scoring, healthcare, and risk management.

**Key reference:**
> Krishnamoorthy, S. (2024). Interpretable Classifier Models for Decision
> Support Using High Utility Gain Patterns. *IEEE Access*, 12, 126088–126107.
> DOI: [10.1109/ACCESS.2024.3455563](https://doi.org/10.1109/ACCESS.2024.3455563)

---

## Features

| Capability | Details |
|---|---|
| **HUG pattern mining** | C++ accelerated via pybind11; optional OpenMP parallelism |
| **scikit-learn API** | Full `BaseEstimator` / `ClassifierMixin` compliance |
| **Mixed feature types** | Integer, float, categorical — auto-detected |
| **Interpretability** | Human-readable patterns, feature importances, SHAP bridge |
| **Calibration** | ECE, MCE, Brier score, reliability diagram data |
| **Drift detection** | PSI + symmetric KL divergence + label drift |
| **Monitoring** | Thread-safe `PredictionMonitor`, latency tracking |
| **Governance** | Model cards (JSON + Markdown), audit artifacts, SBOM |
| **Observability** | OpenTelemetry tracing, Prometheus metrics (both optional) |
| **Secure serialisation** | Allowlist-based `_RestrictedUnpickler`, versioned schema |
| **Deployment** | FastAPI inference server, Docker image, Kubernetes manifests |
| **CI/CD** | GitHub Actions: lint → coverage → native tests → wheels → PyPI |

---

## Installation

```bash
# From PyPI (pre-built wheels for Linux / macOS / Windows, Python 3.9–3.12)
pip install hugiml-core

# With optional telemetry (OpenTelemetry + Prometheus)
pip install "hugiml-core[telemetry]"

# With SHAP interoperability
pip install "hugiml-core[explainability]"

# With MLflow integration
pip install "hugiml-core[mlflow]"

# Everything
pip install "hugiml-core[all]"
```

**Build from source** (requires a C++17 compiler and CMake or pybind11):

```bash
git clone https://github.com/srikumar2050/hugiml-core.git
cd hugiml-core
pip install -e ".[dev]"
python setup.py build_ext --inplace
```

---

## Quick Start

### Path A — `prepareXy` (recommended)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from hugiml import HUGIMLClassifierNative

# Load your data
X = pd.read_csv("your_data.csv")
y = X.pop("target")

# Instantiate and prepare
clf = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
X_enc, y_enc = clf.prepareXy(X, y)          # detects column types automatically

X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y_enc, stratify=y_enc)
clf.fit(X_tr, y_tr)

# Predict
proba = clf.predict_proba(X_te)
labels = clf.predict(X_te)

# Explain
print(clf.get_hug_features())    # e.g. ['age=[35,50]', 'savings=low']
print(clf.get_pattern_info())    # utility / info-gain / support table
print(clf.feature_importances()) # original-feature importances
print(clf.model_summary())
```

### Path B — `allCols` (cross-validation loops)

```python
clf = HUGIMLClassifierNative(
    allCols=[int_col_names, float_col_names, cat_col_names],
    origColumns=X.columns.tolist(),
    B=7, L=1, G=5e-3,
)
clf.fit(X_train, y_train)
clf.predict(X_test)
```

---

## Drift Detection & Monitoring

```python
# Enable in-process prediction monitoring
clf.enable_monitoring(window_size=1000)
clf.predict_proba(X_new)
print(clf.monitor.report())

# Multi-method drift detection (PSI + KL + label drift)
report = clf.detect_drift(X_new, current_labels=y_new)
print(report)
```

---

## Serialisation

```python
from hugiml.serialization import save_model, load_model, generate_sbom

# Save
save_model(clf, "model.hugiml")

# Reload (safe allowlist-based deserialization)
clf2 = load_model("model.hugiml")

# Generate SBOM (CycloneDX-lite)
sbom = generate_sbom(clf)
```

---

## Governance & Model Cards

```python
from hugiml.governance import generate_model_card

card = generate_model_card(
    clf,
    model_id="credit-scorer-v1.0.0",
    intended_use="Credit risk assessment for SME lending.",
    training_data_description="German Credit dataset, 1000 samples",
)
print(card.to_markdown())
card.save("model_card.json")
```

---

## Calibration

```python
from hugiml.calibration import evaluate_calibration

proba = clf.predict_proba(X_te)
result = evaluate_calibration(y_te.values, proba[:, 1])
print(f"ECE:   {result.ece:.4f}")
print(f"Brier: {result.brier_score:.4f}")
```

---

## Inference Server

A FastAPI-based inference server is included for containerised deployments.

```bash
# Build image
docker build -t hugiml-core:latest -f docker/Dockerfile .

# Run (mount a directory containing model.hugiml)
docker run -p 8080:8080 -v /path/to/models:/models hugiml-core:latest

# POST /predict
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"age": 35, "savings": "moderate", ...}]}'
```

Kubernetes manifests are in [`kubernetes/deployment.yaml`](kubernetes/deployment.yaml).

---

## CI / CD

| Workflow | Trigger | What it does |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | Every push / PR | Lint, type-check, coverage gate (≥80%), native tests (3 OS × 3 Python), sanitizer build, bench regression, wheel build |
| [`release.yml`](.github/workflows/release.yml) | Git tag `v*.*.*` | Build all platform wheels, generate SBOM, publish to PyPI, create GitHub Release |
| [`nightly.yml`](.github/workflows/nightly.yml) | Nightly UTC | Property-based tests (Hypothesis), calibration validation, memory safety, full benchmarks |

---

## Repository Structure

```
hugiml-core/
├── src/
│   ├── _native/          C++ extension sources (pybind11)
│   └── hugiml/           Python package
│       ├── classifier.py     HUGIMLClassifierNative
│       ├── calibration.py    ECE, Brier, reliability diagrams
│       ├── explainability.py SHAP bridge, feature lineage, stability
│       ├── governance.py     Model cards, audit artifacts
│       ├── monitoring.py     PredictionMonitor, DriftDetector
│       ├── serialization.py  save/load, SBOM, restricted unpickler
│       ├── telemetry.py      OpenTelemetry, Prometheus (optional)
│       └── exceptions.py     Exception hierarchy
├── tests/                Pytest suite (unit + integration + stress)
├── benchmarks/           Micro-benchmarks and regression gate
├── docker/               Dockerfile + FastAPI inference server
├── kubernetes/           Deployment manifests
├── scripts/              Build and utility scripts
├── docs/                 Model card template
├── .github/workflows/    CI/CD pipelines
├── pyproject.toml
└── setup.py
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

## Citation

If you use hugiml-core in research or commercial work, please cite:

```bibtex
@article{krishnamoorthy2024hugIML,
  author  = {Krishnamoorthy, Srikumar},
  title   = {Interpretable Classifier Models for Decision Support Using
             High Utility Gain Patterns},
  journal = {IEEE Access},
  volume  = {12},
  pages   = {126088--126107},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.3455563}
}
```

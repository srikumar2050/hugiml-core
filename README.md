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
| **Profile visualisations** | EBM-style 1-D/2-D HUG profiles, active-pattern explanations, coefficient-support views (plotly) |
| **Interpretability metrics** | Pattern count, coverage, overlap, sparsity, top-k cumulative contribution |
| **Adaptive binning** | Per-feature supervised B selection — fixes the B-sensitivity trap |
| **Pattern pruning** | Regulated remove/refit/calibrate workflow with full JSON audit trail |
| **Multiclass & imbalance** | Multiclass report, SMOTE/class-weight pipeline, high-cardinality encoding |
| **Benchmark suite** | Reproducible CV comparison vs EBM, XGBoost, RF, LR, RuleFit, GAM |
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
# Core (pre-built wheels for Linux / macOS / Windows, Python 3.9–3.13)
pip install hugiml-core

# With profile plots (Plotly-based, EBM-style visualisations)
pip install "hugiml-core[plots]"

# With benchmark comparison suite (EBM / XGBoost / RuleFit / GAM)
pip install "hugiml-core[benchmarks]"

# With imbalanced-data helpers (SMOTE etc.)
pip install "hugiml-core[imbalanced]"

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

> **Note on `prepareXy`:** `prepareXy` performs schema and type preparation
> only — it detects integer, float, and categorical columns and encodes the
> target. Discretisation, HUG pattern mining, and downstream classifier
> fitting all occur inside `fit()` on the training data supplied to that call.
> Always call `prepareXy` on the **full** dataset before splitting, and pass
> only the training split to `fit()`.

### Path A — `prepareXy` (recommended)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from hugiml import HUGIMLClassifierNative

clf = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
X_enc, y_enc = clf.prepareXy(X_df, y)          # schema/type prep — no fitting

X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y_enc, stratify=y_enc)
clf.fit(X_tr, y_tr)                             # mining + downstream fit on train only

proba = clf.predict_proba(X_te)
print(clf.get_hug_features())    # e.g. ['age=[35,50]', 'savings=low']
print(clf.feature_importances())
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

## Model Explanation Dashboard

The explanation dashboard is available directly from a fitted model:

```python
from hugiml.plots import HUGPlotter
plotter = HUGPlotter(clf, X_test=Xte, y_test=yte)
plotter.plot_dashboard().show()          # interactive Plotly dashboard
```

Each panel shows the **logistic-regression coefficient per quantile bin** for one feature —
directly analogous to EBM shape functions.  Positive values signal the target class;
negative values signal the complementary class.  The grey step overlay shows training support.

**Breast cancer classification** (n=569, p=30, AUC=0.990):

![Feature shape profiles — breast cancer](docs/images/explanation_dashboard_bc.png)

**Credit risk scoring** (n=1000, p=9 engineered features, AUC=0.850):

![Feature shape profiles — credit risk](docs/images/explanation_dashboard_credit.png)


## Profile Visualisations

```python
from hugiml.plots import HUGPlotter

plotter = HUGPlotter(clf)

# EBM-style 1-D shape function: utility per bin + support overlay
plotter.plot_marginal_bin_profile("age").show()

# All patterns that involve a feature (compound interactions coloured by degree)
plotter.plot_feature_combinations("age").show()

# Top patterns by importance (bar chart)
plotter.plot_top_patterns(n=20).show()

# Coefficient vs support scatter
plotter.plot_coef_support_scatter().show()

# Local explanation for one sample
plotter.plot_active_patterns(X_test, sample_idx=0).show()

# Full interactive dashboard (feature selector, all charts)
plotter.plot_dashboard().show()
```

---

## Interpretability Metrics

```python
from hugiml.metrics import compute_all_metrics

m = compute_all_metrics(clf, X_test)
print(m)
# InterpretabilityMetrics
# ==========================================
#   n_patterns              : 87
#   avg_pattern_length      : 1.34
#   coverage                : 0.9812  (98.1% of 285 samples)
#   mean_active_patterns    : 6.21
#   overlap_rate (norm.)    : 0.0714
#   explanation_sparsity    : 0.0230
#   top-k cumulative |coef|:
#     top-  1 :   8.4%
#     top-  5 :  31.2%
#     top- 10 :  54.7%
```

---

## Pattern Pruning (Regulated Workflow)

In regulated domains, analysts often need to remove patterns that reference
protected attributes, have high PSI, or are operationally invalid. HUG-IML
provides an EBM-inspired controlled editing workflow with a full JSON audit
trail.

```python
from hugiml.pruning import PatternEditor

editor = PatternEditor(clf, operator_name="risk-team")

# Preview what's in the model
print(editor.list_patterns().head(10))

# Remove by index, keyword, or support threshold
editor.remove([3, 7], reason="references protected attribute 'gender'")
editor.remove_by_keyword("postcode", reason="high PSI — unstable feature")
editor.remove_low_support(min_support=0.01, reason="noise patterns")

# Refit downstream classifier + optional calibration
editor.refit(X_tr, y_tr)
editor.calibrate(X_cal, y_cal, method="isotonic")

# Get edited model and audit report
new_clf = editor.finalize()
print(editor.audit_report())   # JSON: timestamps, reasons, diff summary
```

---

## Adaptive Binning

The global `B` parameter controls how many quantile bins each numerical feature
is discretised into.  Choosing a single `B` for all features can miss the
optimal resolution for informative features or over-fragment noisy ones.
`HUGIMLAdaptive` selects the optimal bin count **per feature** via a supervised
entropy search with elbow-stopping.

```python
from hugiml.adaptive import HUGIMLAdaptive

clf = HUGIMLAdaptive(b_candidates=[3, 5, 7, 10, 15], L=2, G=1e-4)
X_enc, y_enc = clf.prepareXy(X_df, y)
clf.fit(X_tr, y_tr)

print(clf.per_feature_b_)   # {'age': 7, 'income': 10, 'duration': 5, ...}
clf.plot_bin_profiles()      # bar chart of chosen B per feature
clf.ig_heatmap()             # IG score grid across (feature × B candidates)
```

Alternatively, enable adaptive binning directly on `HUGIMLClassifierNative`:

```python
clf = HUGIMLClassifierNative(
    B=10,                   # upper bound; adaptive search finds optimal B_j ≤ B
    adaptive_binning=True,
    b_candidates=[3,5,7,10],
    min_marginal_gain_ratio=0.02,   # elbow threshold (default 2%)
)
```

**How it works:**  For each numerical feature, the algorithm evaluates the
information gain (IG) at each candidate `B` value.  It stops when the marginal
IG gain falls below `min_marginal_gain_ratio × current_IG` — preventing
overfitting by stopping at the natural elbow rather than always picking the
maximum.

**NaN handling:**  Adaptive pre-binning correctly treats non-finite cells as
`np.nan` (no transaction item) — consistent with the native-path NaN handling
introduced in v1.1.0.

---

## Missing Value Handling

HugiML v1.1.0 treats NaN and Inf values as **"not observed"** — no imputation,
no special parameter needed.  The behaviour is always-on and transparent.

**How it works:**  All numerical columns are pre-binned to string quantile
labels at fit time (equal-frequency, using the same `B` as the model).
Non-finite cells become `np.nan` in the label array; the C++ transaction
builder skips them — that (row, feature) item is simply absent from the
transaction.  Patterns that require a missing feature do not fire for that row.

```python
import numpy as np
from hugiml import HUGIMLClassifierNative

# NaN in training data — cells with NaN generate no transaction item
X_train.iloc[5, 2] = np.nan   # row 5, feature 2: no item mined for this cell

clf = HUGIMLClassifierNative(B=5, L=2, G=1e-4)
clf.fit(X_train, y_train)      # _missing_col_edges_ stores quantile edges

# NaN at prediction time — same behaviour, no imputation
X_test.iloc[0, 0] = np.nan
proba = clf.predict_proba(X_test)  # row 0 scored using only available features
```

See the [missing value robustness analysis](#missing-value-robustness) below.

---

## Multiclass, Imbalanced Data, High-Cardinality Categoricals

```python
from hugiml.multiclass import (
    MulticlassHUGReport,
    make_imbalanced_pipeline,
    encode_high_cardinality,
    apply_encoding,
)

# Per-class pattern importances for multiclass models
report = MulticlassHUGReport(clf)
print(report.importances_for_class(class_label=2, top_n=10))
print(report.summary())

# SMOTE / class-weight pipeline for imbalanced data
clf_bal = make_imbalanced_pipeline(clf_proto, strategy="smote")
clf_bal.fit(X_tr, y_tr)

# Target-mean / frequency encoding for high-cardinality categoricals
X_enc, enc_map = encode_high_cardinality(X_tr, y_tr, threshold=20, method="target_mean")
X_te_enc = apply_encoding(X_te, enc_map)
```

---

## Benchmark Suite

Reproduce paper claims or benchmark on your own datasets:

```bash
# Run full CV comparison (HUG vs EBM / XGBoost / RF / LR / RuleFit / GAM)
python -m hugiml.benchmarks.runner

# Specific datasets
python -m hugiml.benchmarks.runner --datasets breast_cancer adult german_credit

# Save results to JSON
python -m hugiml.benchmarks.runner --output benchmarks/results/
```

Or use the installed console script:

```bash
hugiml-bench --datasets breast_cancer --output results/
```

See also the worked notebooks in [`notebooks/`](notebooks/):

| Notebook | Description |
|---|---|
| [`01_benchmark_baselines.ipynb`](notebooks/01_benchmark_baselines.ipynb) | 5-fold CV across HUG-IML, XGBoost, LightGBM, RF, LogReg on Breast Cancer |
| [`02_hug_vs_ebm.ipynb`](notebooks/02_hug_vs_ebm.ipynb) | Side-by-side HUG-IML vs EBM: shape functions, feature importance, performance |
| [`03_special_cases.ipynb`](notebooks/03_special_cases.ipynb) | Multiclass, imbalanced data, high-cardinality categoricals, adaptive binning, pattern pruning |

### Observed results (breast cancer · n=569 · p=30 · 5-fold CV)

![Benchmark comparison](docs/images/benchmark_comparison.png)

| Model | AUC (mean±std) | Fit time/fold | Complexity budget | Remarks |
|---|---|---|---|---|
| HUG B=3 | 0.9907 ± 0.0031 | 0.32 s | **topK** patterns | `topK` is an explicit cap. Actual mined patterns can be much lower. |
| HUG B=5 | 0.9909 ± 0.0028 | 0.34 s | **topK** patterns | More bins per feature. |
| HUG adaptive | 0.9954 ± 0.0022 | 1.20 s | **topK** patterns | Per-feature B increases fit time. |
| EBM (p=30) | 0.9940 ± 0.0025 | 11.0 s | **p × bins + interactions** | With default 256 bins and 10 interactions: ≈30×256 + 10×256²≈660k terms. EBM is the reference interpretable baseline. |
| XGBoost (n=200, d=4) | 0.9882 ± 0.0040 | 0.12 s | **trees × leaves** | 200 trees × (2⁴−1)=15 leaves = 3,000 decision nodes. Ensemble — not directly interpretable. |
| LightGBM (n=200, d=4) | 0.9921 ± 0.0028 | 0.07 s | **leaves × trees** | Leaf-wise growth; similar node count to XGBoost. Faster training via histogram binning. |

> Complexity budget reflects the number of distinct learned components — patterns/rules/nodes — used at inference.
> HUG-IML's budget is always exactly `topK` patterns regardless of feature count or bin resolution.

### Missing value robustness <a name="missing-value-robustness"></a>

![Missing value benchmark](docs/images/missing_value_benchmark.png)

**Simulation setup:**  Wine dataset (n=178, p=13 continuous features, binary: Cultivar 1
vs rest), 3-fold stratified CV, missing rates 0–40%. Three mechanisms:

- **MCAR** — each cell (i, j) masked independently with probability `p`. No relationship to observed or unobserved values. `P(missing | X, y) = p`.
- **MAR** — `P(missing_ij)` is proportional to the row's mean z-score across all features (rows with high observed values are more likely to have missing cells). Missingness depends on what we can see, not on the missing value itself: `P(missing | X_obs)` varies with observed data.
- **MNAR** — `P(missing_ij)` is proportional to the normalised value of cell (i, j) itself. High-value cells — often the most discriminative — are the ones most likely to be absent. The true missing value is systematically higher than any imputed substitute. This is the hardest regime for imputation-based methods.

**Observations:**

- All models maintain AUC > 0.95 across most conditions on this small dataset (n=178); the signal is
  strong enough that partial data remains informative.
- HUGIML fit time *decreases* with more missing data — fewer transaction items
  means the C++ miner processes shorter transactions faster (0.34 s → 0.23 s at 40% MCAR).
  Tree models show the opposite trend (more NaN-routing decisions per split).
- Under MAR, imputation-based methods benefit from a consistent train/test bias:
  the imputed distribution is the same in both splits, so the model learns to work with
  it. This advantage would not hold under a mechanism shift at deployment.
- Under MNAR (the hardest case for imputation), HUGIML and EBM show comparable
  degradation. HUGIML avoids imputation distortion; EBM may fit the imputed values
  as if they were real observations.
- HUGIML's "no item" semantics give interpretable absence: a pattern simply does not
  fire if one of its features is unavailable, which is the correct behaviour for
  a transaction-based model.

---

## Drift Detection & Monitoring

```python
clf.enable_monitoring(window_size=1000)
clf.predict_proba(X_new)
print(clf.monitor.report())

report = clf.detect_drift(X_new, current_labels=y_new)
print(report)
```

---

## Serialisation

```python
from hugiml.serialization import save_model, load_model, generate_sbom

save_model(clf, "model.hugiml")
clf2 = load_model("model.hugiml")   # safe allowlist-based deserialisation
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

result = evaluate_calibration(y_te.values, proba[:, 1])
print(f"ECE:   {result.ece:.4f}")
print(f"Brier: {result.brier_score:.4f}")
```

---

## Inference Server

A FastAPI-based inference server is included for containerised deployments.

```bash
docker build -t hugiml-core:latest -f docker/Dockerfile .
docker run -p 8080:8080 -v /path/to/models:/models hugiml-core:latest

curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"age": 35, "savings": "moderate"}]}'
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
│   ├── _native/              C++ extension sources (pybind11)
│   └── hugiml/               Python package
│       ├── classifier.py         HUGIMLClassifierNative
│       ├── calibration.py        ECE, Brier, reliability diagrams
│       ├── explainability.py     SHAP bridge, feature lineage, stability
│       ├── governance.py         Model cards, audit artifacts
│       ├── monitoring.py         PredictionMonitor, DriftDetector
│       ├── serialization.py      save/load, SBOM, restricted unpickler
│       ├── telemetry.py          OpenTelemetry, Prometheus (optional)
│       ├── exceptions.py         Exception hierarchy
│       ├── metrics.py            Interpretability-complexity metrics  [new v1.1.0]
│       ├── plots.py              EBM-style profile visualisations     [new v1.1.0]
│       ├── pruning.py            Pattern editor + audit trail         [new v1.1.0]
│       ├── adaptive.py           Per-feature adaptive binning         [new v1.1.0]
│       ├── multiclass.py         Multiclass / imbalanced / encoding   [new v1.1.0]
│       └── benchmarks/           CV comparison suite                  [new v1.1.0]
│           ├── __init__.py
│           └── runner.py
├── notebooks/                Worked examples
│   ├── 01_benchmark_baselines.py
│   ├── 02_hug_vs_ebm.py
│   └── 03_special_cases.py
├── tests/                    Pytest suite (unit + integration + stress)
├── benchmarks/               C++ micro-benchmarks and regression gate
├── docker/                   Dockerfile + FastAPI inference server
├── kubernetes/               Deployment manifests
├── scripts/                  Build and utility scripts
├── docs/                     Model card template
├── .github/workflows/        CI/CD pipelines
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

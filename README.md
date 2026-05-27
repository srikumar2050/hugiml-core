# hugiml-core

> **High-performance interpretable rule-based ML infrastructure** built on the
> HUG-IML algorithm published in IEEE Access (2024).

[![CI](https://github.com/srikumar2050/hugiml-core/actions/workflows/ci.yml/badge.svg)](https://github.com/srikumar2050/hugiml-core/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hugiml-core.svg)](https://pypi.org/project/hugiml-core/)
[![Docs](https://readthedocs.org/projects/hugiml-core/badge/?version=latest)](https://hugiml-core.readthedocs.io/en/latest/?badge=latest)
[![Python](https://img.shields.io/pypi/pyversions/hugiml-core.svg)](https://pypi.org/project/hugiml-core/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2024.3455563-blue)](https://doi.org/10.1109/ACCESS.2024.3455563)

<p align="left">
  <img src="docs/images/header-hugiml.png" alt="HUGIML: interpretable tabular ML through compact human-readable patterns" width="800" height="370">
</p>

HUGIML learns **human-readable High Utility Gain patterns** and uses those patterns as the model representation itself. Instead of explaining a black-box after training, the learned model is already composed of inspectable intervals, categories, supports, utilities, and coefficients.

```text
glucose=[157.1,177.3)                coef= +1.4077   support=0.067
bmi=[31.8,39.1)                      coef= +1.0839   support=0.200
duration=[24,48)                     coef= +0.84     support=0.28
checking_status=no_checking          coef= +1.12     support=0.39
```

<p align="left">
  <img src="docs/images/positioning-mosaic.png" alt="Where HUGIML fits" width="800" height="500">
</p>

---

## What Is HUG-IML?

The **High Utility Gain Interpretable Machine Learning (HUG-IML)** framework extracts *High Utility Gain patterns* from labelled tabular data, transforms the input into a binary pattern-presence matrix, and fits an interpretable downstream classifier (logistic regression by default) on that matrix.

The resulting patterns are human-readable and serve as the primary source of model explanations, making the system suitable for regulated domains such as credit scoring, healthcare, and risk management.

**Key reference:**

> Krishnamoorthy, S. (2024). Interpretable Classifier Models for Decision
> Support Using High Utility Gain Patterns. *IEEE Access*, 12, 126088–126107.
> DOI: [10.1109/ACCESS.2024.3455563](https://doi.org/10.1109/ACCESS.2024.3455563)

---

## Validation Highlights

> The finance panels use German Credit / HELOC-style risk features such as loan duration, credit amount, checking status, and repayment-risk signals. The healthcare panels use Pima diabetes-style features such as glucose, BMI, pregnancies, pedigree, and age.

This section summarizes practical comparisons against established interpretable and high-performance tabular baselines. The goal is not to claim that HUGIML always beats boosted trees or EBM; the goal is to show where HUGIML provides a distinct trade-off: **compact, auditable pattern explanations with competitive predictive behavior**.

### HUGIML vs EBM shape profiles

EBM learns smooth additive shape functions. HUGIML learns threshold/category profiles using native bin-level pattern contributions. On Pima diabetes and credit-risk examples, the models can express similar directional behavior, but HUGIML presents it as compact intervals/categories that are easier to audit.

<p align="left">
  <img src="docs/images/shape-profiles-hugiml-vs-ebm.png" alt="HUGIML native shape profiles compared with EBM shape functions" width="700"  height="300">
</p>

EBM is excellent for smooth effect inspection; HUGIML is strong when the explanation needs to be reviewed as a set of readable thresholds and pattern contributions.

### Real-world and synthetic non-monotonic benchmarks

The following benchmark panels compare HUGIML with LR, XGBoost, LightGBM, Random Forest, and EBM under explicit fitted-complexity budgets. They are intended as validation visuals, not as a universal ranking of models.

<p align="left">
  <img src="docs/images/realworld-credit-risk-benchmark.png" alt="Real-world credit risk benchmark comparing HUGIML, LR, XGBoost, LightGBM, Random Forest, and EBM" width="800">
</p>

**Real-world credit-risk benchmark.** On Give Me Some Credit and Taiwan Credit Card Default, LightGBM and XGBoost achieve the highest raw ROC-AUC. HUGIML improves over LR while remaining in a compact fitted-complexity range. EBM is shown with capped bins (`max_bins=8`, `interactions=0`) so that it is assessed as a competitive interpretable baseline rather than being penalized by excessive bin-score complexity.

<p align="left">
  <img src="docs/images/synthetic-nonmonotonic-benchmark.png" alt="Synthetic non-monotonic benchmark comparing HUGIML, LR, XGBoost, LightGBM, Random Forest, and EBM" width="800">
</p>

**Synthetic non-monotonic benchmark.** The synthetic panels use two controlled datasets with interval-like and oscillatory feature effects. They illustrate model behavior when LR is structurally misspecified by a single global slope. HUGIML provides a compact nonlinear lift over LR, while LightGBM and XGBoost remain strongest in raw ROC-AUC. EBM is shown with capped bins (`max_bins=16`, `interactions=0`) to provide a fairer interpretable baseline.

### Native missing-value handling

HUGIML, XGBoost, LightGBM, and EBM can all operate without an external imputation pipeline, but they treat missingness differently.

<p align="left">
  <img src="docs/images/native-missing-value-schemes.png" alt="Native missing-value schemes in HUGIML, XGBoost, LightGBM, and EBM" width="760">
</p>

| Model | Native missing-value behavior | What to monitor |
|---|---|---|
| **HUGIML** | Missing numerical values are absent from the transaction. Patterns requiring that feature item do not fire. | Missingness rate and activation frequency of top patterns. |
| **XGBoost** | Each split learns a default route for missing values. | Whether default-route behavior changes under deployment shift. |
| **LightGBM** | Histogram splits learn how missing values are routed. | Missing-value routing and feature missingness drift. |
| **EBM** | Missing values can be modeled as a separate bin/effect. | Size and sign of each missing-bin effect. |

Native missing handling is usually preferable to blindly injecting mean/median values. It does not remove all risk: if the missingness mechanism changes, all models that use missingness as signal can drift.

### Adaptive binning

The global `B` parameter controls numerical bin resolution. A single fixed `B` can be too coarse for informative features or too fragmented for noisy ones. Adaptive binning selects per-feature resolution using supervised information gain and elbow stopping.

<p align="left">
  <img src="docs/images/adaptive-binning-impact.png" alt="Adaptive binning benchmark against fixed bin counts" width="700" height="300">
</p>

Adaptive binning is a safe default when you do not want to tune `B`; fixed `B=5` is a useful fast baseline; very large fixed `B` can over-fragment patterns.

### Pattern explanations on finance and healthcare datasets

HUGIML keeps explanations close to the model: feature intervals/categories, coefficients, support, utility, and information gain are available directly from the fitted classifier.

<p align="left">
  <img src="docs/images/pattern-explanations-real-datasets.png" alt="HUGIML pattern explanations on finance and healthcare datasets" width="760">
</p>

Learned patterns map naturally to domain narratives: glucose/BMI/age in diabetes and duration/checking-status/credit-amount style signals in German Credit or HELOC-style risk scoring.

### Model-card-ready artifacts

Because HUGIML’s explanations are compact, they can be copied directly into model cards, audit packets, validation reports, and deployment reviews.

<p align="left">
  <img src="docs/images/model-card-governance.png" alt="Model-card-ready HUGIML explanations" width="760">
</p>

---

## Features

| Capability | Details |
|---|---|
| **HUG pattern mining** | C++ accelerated via pybind11; optional OpenMP parallelism |
| **scikit-learn API** | Full `BaseEstimator` / `ClassifierMixin` compliance |
| **Mixed feature types** | Integer, float, categorical — auto-detected or explicitly supplied |
| **Profile visualisations** | EBM-style 1-D/2-D HUG profiles, active-pattern explanations, coefficient-support views (Plotly) |
| **Interpretability metrics** | Pattern count, coverage, overlap, sparsity, top-k cumulative contribution |
| **Adaptive binning** | Per-feature supervised `B` selection — fixes the B-sensitivity trap |
| **Feature modes** | Pattern-only, original-plus-patterns, and original-plus-interactions downstream representations |
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
# Core
pip install hugiml-core

# With profile plots
pip install "hugiml-core[plots]"

# With benchmark comparison suite
pip install "hugiml-core[benchmarks]"

# With imbalanced-data helpers
pip install "hugiml-core[imbalanced]"

# With SHAP interoperability
pip install "hugiml-core[explainability]"

# With MLflow integration
pip install "hugiml-core[mlflow]"

# Everything
pip install "hugiml-core[all]"
```

**Build from source** requires a C++17 compiler and CMake or pybind11:

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
> target. Discretisation, HUG pattern mining, and downstream classifier fitting
> occur inside `fit()` on the training data supplied to that call.

### Path A — `prepareXy`

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from hugiml import HUGIMLClassifierNative

clf = HUGIMLClassifierNative(B=7, L=1, G=5e-3)

X_enc, y_enc = clf.prepareXy(X_df, y)   # schema/type prep — no model fitting

X_tr, X_te, y_tr, y_te = train_test_split(
    X_enc, y_enc, stratify=y_enc, random_state=42
)

clf.fit(X_tr, y_tr)                     # mining + downstream fit on train only
proba = clf.predict_proba(X_te)

print(clf.get_hug_features())
print(clf.feature_importances())
print(clf.model_summary())
```

### Path B — explicit `allCols` for CV and production pipelines

```python
from hugiml import HUGIMLClassifierNative

clf = HUGIMLClassifierNative(
    allCols=[int_col_names, float_col_names, cat_col_names],
    origColumns=X.columns.tolist(),
    B=15,
    L=1,
    G=1e-5,
    topK=150,
    adaptive_binning=True,
    b_candidates=[2, 3, 5, 7, 10, 15],
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)
```


## Feature Modes

HUGIML can use the mined binary pattern matrix in three downstream feature modes. The default remains the original pattern-only behavior, so existing code keeps the same semantics unless `feature_mode` is set explicitly.

| `feature_mode` | Downstream estimator input | When to use |
|---|---|---|
| `"patterns_only"` | HUGIML binary pattern matrix only | Standard HUGIML; best when the mined pattern space itself captures the decision boundary. |
| `"original_plus_patterns"` | Original features plus all mined binary patterns | Useful when the original features contain strong marginal signal and HUGIML patterns add supervised nonlinear refinements. |
| `"original_plus_interactions"` | Original features plus only `L > 1` mined patterns | Useful when original features should handle marginal effects and HUGIML should contribute interaction/compound-region features. |

```python
from hugiml import HUGIMLClassifierNative

# Backward-compatible default: pattern matrix only
clf = HUGIMLClassifierNative(
    B=10,
    L=2,
    G=1e-3,
    topK=150,
    adaptive_binning=True,
    feature_mode="patterns_only",
)

# Hybrid: original features + all binary HUGIML patterns
clf_hybrid_all = HUGIMLClassifierNative(
    B=10,
    L=2,
    G=1e-3,
    topK=150,
    adaptive_binning=True,
    feature_mode="original_plus_patterns",
)

# Hybrid: original features + higher-order/interaction patterns only
clf_hybrid_interactions = HUGIMLClassifierNative(
    B=10,
    L=2,
    G=1e-3,
    topK=150,
    adaptive_binning=True,
    feature_mode="original_plus_interactions",
)
```

`transform(X)` always returns the HUGIML binary pattern matrix, regardless of `feature_mode`. The feature mode only changes the matrix passed to the downstream estimator inside `fit()`, `predict()`, `predict_proba()`, and `score()`.

For hybrid modes, HUGIML standardizes numeric original features internally before concatenating them with the sparse binary pattern matrix. `feature_importances()` and `model_summary()` report the downstream feature representation used by the fitted model, while `get_hug_features()` and `get_pattern_info()` remain pattern-only APIs.

---

## Model Explanation Dashboard

The explanation dashboard is available directly from a fitted model:

```python
from hugiml.plots import HUGPlotter

plotter = HUGPlotter(clf)

plotter.plot_dashboard(
    X_test,
    dataset_name="My Dataset",
    feature_names_for_profile=["age", "income", "glucose"],
    output_path="hugiml_dashboard.html",
)

plotter.plot_marginal_bin_profile("glucose", X=X_test).show()
plotter.plot_top_patterns(top_n=20).show()
plotter.plot_feature_importance(top_n=15).show()
plotter.plot_active_patterns(X_test, sample_idx=0).show()
```

Each profile panel shows the learned bin/pattern behavior for a feature: utility or coefficient-like contribution per bin, with support overlay where available. Positive values signal the target class; negative values signal the complementary class.

Existing example dashboards:

**public tabular benchmark classification**  
![Feature shape profiles — public tabular benchmark](docs/images/explanation_dashboard_bc.png)

**Credit risk scoring**  
![Feature shape profiles — credit risk](docs/images/explanation_dashboard_credit.png)

---

## Profile Visualisations

```python
from hugiml.plots import HUGPlotter

plotter = HUGPlotter(clf)

# EBM-style 1-D shape function: utility per bin + support overlay
plotter.plot_marginal_bin_profile("age", X=X_test).show()

# Feature-combination view for compound patterns
plotter.plot_feature_combinations("age").show()

# Top patterns by importance
plotter.plot_top_patterns(top_n=20).show()

# Local explanation for one sample
plotter.plot_active_patterns(X_test, sample_idx=0).show()

# Full interactive dashboard
plotter.plot_dashboard(X_test, dataset_name="Dataset", output_path="dashboard.html")
```

---

## Interpretability Metrics

```python
from hugiml.metrics import compute_all_metrics

m = compute_all_metrics(clf, X_test)
print(m)
```

Example output:

```text
InterpretabilityMetrics
==========================================
n_patterns              : 87
avg_pattern_length       : 1.34
coverage                 : 0.9812
mean_active_patterns     : 6.21
overlap_rate             : 0.0714
explanation_sparsity     : 0.0230

top-k cumulative |coef|:
top- 1 : 8.4%
top- 5 : 31.2%
top-10 : 54.7%
```

---

## Pattern Pruning: Regulated Editing Workflow

In regulated domains, analysts often need to remove patterns that reference protected attributes, have high PSI, or are operationally invalid. HUGIML provides an EBM-inspired controlled editing workflow with a JSON audit trail.

```python
from hugiml.pruning import PatternEditor

editor = PatternEditor(clf, operator_name="risk-team")

print(editor.list_patterns().head(10))

editor.remove([3, 7], reason="references protected attribute 'gender'")
editor.remove_by_keyword("postcode", reason="high PSI — unstable feature")
editor.remove_low_support(min_support=0.01, reason="noise patterns")

editor.refit(X_tr, y_tr)
editor.calibrate(X_cal, y_cal, method="isotonic")

new_clf = editor.finalize()
print(editor.audit_report())
```

---

## Adaptive Binning

The global `B` parameter controls how many quantile bins each numerical feature is discretised into. Choosing one `B` for all features can miss the optimal resolution for informative features or over-fragment noisy ones.

`HUGIMLAdaptive` selects the optimal bin count **per feature** via supervised entropy/information-gain search with elbow stopping.

```python
from hugiml.adaptive import HUGIMLAdaptive

clf = HUGIMLAdaptive(
    b_candidates=[3, 5, 7, 10, 15],
    L=2,
    G=1e-4,
)

X_enc, y_enc = clf.prepareXy(X_df, y)
clf.fit(X_tr, y_tr)

print(clf.per_feature_b_)
clf.plot_bin_profiles()
clf.ig_heatmap()
```

Alternatively, enable adaptive binning directly on `HUGIMLClassifierNative`:

```python
from hugiml import HUGIMLClassifierNative

clf = HUGIMLClassifierNative(
    B=10,                              # upper bound
    adaptive_binning=True,
    b_candidates=[3, 5, 7, 10],
    min_marginal_gain_ratio=0.02,       # elbow threshold
)
```

**How it works:** for each numerical feature, HUGIML evaluates information gain at candidate `B` values and stops when the marginal gain falls below `min_marginal_gain_ratio × current_IG`. This prevents blindly selecting the maximum bin count.

**NaN handling:** adaptive pre-binning treats non-finite cells as `np.nan`, consistent with native missing-value handling.

---

## Missing Value Handling

HUGIML v1.1.0 treats NaN and Inf values as **not observed** — no imputation and no special parameter are required.

**How it works:** numerical columns are pre-binned at fit time. Non-finite cells become `np.nan` in the label array, and the C++ transaction builder skips them. The corresponding item is absent from the transaction. Patterns requiring that feature do not fire for that row.

```python
import numpy as np
from hugiml import HUGIMLClassifierNative

X_train.iloc[5, 2] = np.nan

clf = HUGIMLClassifierNative(B=5, L=2, G=1e-4)
clf.fit(X_train, y_train)

X_test.iloc[0, 0] = np.nan
proba = clf.predict_proba(X_test)       # scored using available feature items
```

This differs from median/mean imputation: HUGIML does not fabricate replacement values. It also differs from row dropping: incomplete rows remain usable.

See the benchmark section below for missing-value robustness.

### Mining Patterns About Missingness

In healthcare and finance, **absence of data is often informative**, a missing
critical lab may indicate the patient was too unstable for testing, or a missing
field in a credit application may signal intentional concealment.

To mine patterns that **involve** missingness (e.g., `Glucose_MISSING=1 AND HeartRate=[110,140]`),
add binary missingness indicators as preprocessing features:

```python
def add_missingness_indicators(X, threshold=0.05):
    X_aug = X.copy()
    for col in X.columns:
        if X[col].isna().mean() > threshold:
            X_aug[f"{col}__MISSING"] = X[col].isna().astype(int)
    return X_aug

# Usage
X_with_indicators = add_missingness_indicators(X_raw)
clf = HUGIMLClassifierNative(B=7, L=2, G=1e-4)  
clf.fit(X_with_indicators, y)

# Extract missingness-related patterns
all_patterns = clf.get_hug_features()
missing_patterns = [p for p in all_patterns if '__MISSING' in p]

# Example patterns discovered:
# ['Glucose__MISSING=1 AND Age=[50,65] AND EmergencyAdmit=1',
#  'Troponin__MISSING=1 AND HeartRate=[110,140]',
#  'CurrentDebt__MISSING=1 AND Income=low']
```

**Note:** Apply the same `add_missingness_indicators()` transformation to test/production data. Use `sklearn.pipeline.Pipeline` with `FunctionTransformer` to ensure consistency.

---

## Multiclass, Imbalanced Data, High-Cardinality Categoricals

### Multiclass Classification

```python
from hugiml.multiclass import (
    MulticlassHUGReport,
    make_imbalanced_pipeline,
    encode_high_cardinality,
    apply_encoding,
)

report = MulticlassHUGReport(clf)
print(report.importances_for_class(class_label=2, top_n=10))
print(report.summary())
```

### Imbalanced Data Handling

```python
from hugiml.multiclass import make_imbalanced_pipeline
clf_bal = make_imbalanced_pipeline(clf_proto, strategy="smote")
clf_bal.fit(X_tr, y_tr)
```

### High-Cardinality Categorical Reduction

When categorical features have hundreds or thousands of unique values (ZIP codes, ICD-10 diagnoses, merchant IDs), the pattern mining engine faces combinatorial explosion and dramatically slower mining. To reduce cardinality while **preserving discriminative signal**, group rare categories as '__OTHER__':

```python
def reduce_high_cardinality(X, y, threshold=50, min_frequency=0.01):
    """Group rare categories (<min_frequency) as '__OTHER__' for high-cardinality columns"""
    X_reduced = X.copy()
    
    for col in X.select_dtypes(include=['object', 'category']).columns:
        if X[col].nunique() <= threshold:
            continue  # Skip low-cardinality columns
        
        # Compute frequency per category
        value_counts = X[col].value_counts()
        min_count = len(X) * min_frequency
        
        # Map rare categories to '__OTHER__'
        rare_categories = value_counts[value_counts < min_count].index
        X_reduced[col] = X[col].apply(
            lambda x: '__OTHER__' if x in rare_categories else x
        )
    
    return X_reduced

# Usage
X_reduced = reduce_high_cardinality(X_raw, y, threshold=50, min_frequency=0.01)
clf = HUGIMLClassifierNative(B=7, L=2, G=1e-4)
clf.fit(X_reduced, y)

# Example: 10,000 ZIP codes → ~100 frequent ZIPs + '__OTHER__'
# Patterns: "ZIP_CODE=10001 AND Income=low" or "ZIP_CODE=__OTHER__ AND CreditScore=[600,650]"

# Or use existing utility (basic target encoding):
from hugiml.multiclass import encode_high_cardinality, apply_encoding
X_enc, enc_map = encode_high_cardinality(X_tr, y_tr, threshold=20, method="target_mean")
X_te_enc = apply_encoding(X_te, enc_map)
```

**Note:** Learn category groupings on training data only, then apply the same mapping to test/production data. Use a custom sklearn transformer or save the learned `rare_categories` mapping separately to ensure consistency.

---

## Benchmark Suite

Reproduce paper claims or benchmark on your own datasets:

```bash
# Run full CV comparison
python -m hugiml.benchmarks.runner

# Specific datasets
python -m hugiml.benchmarks.runner --datasets german_credit pima adult

# Save results
python -m hugiml.benchmarks.runner --output benchmarks/results/
```

Or use the installed console script:

```bash
hugiml-bench --datasets german_credit --output results/
```

Worked notebooks in [`notebooks/`](notebooks/) are organized as 12 self-contained folders. Each folder includes an `.ipynb` notebook and, where available, matching `.py`, `.html`, data, and metadata artifacts.

| Folder | Notebook | Brief description |
|---|---|---|
| [`00_quickstart`](notebooks/00_quickstart/) | [`nb00_pattern_explanation_walkthrough.ipynb`](notebooks/00_quickstart/nb00_pattern_explanation_walkthrough.ipynb) | Quick end-to-end walkthrough of fitting HUGIML, extracting patterns, and reading pattern-level explanations. |
| [`01_benchmark_baselines`](notebooks/01_benchmark_baselines/) | [`nb01_benchmark_baselines.ipynb`](notebooks/01_benchmark_baselines/nb01_benchmark_baselines.ipynb) | Benchmark comparison across HUGIML and common tabular baselines such as XGBoost, LightGBM, Random Forest, and logistic regression. |
| [`02_hug_vs_ebm`](notebooks/02_hug_vs_ebm/) | [`nb02_hug_vs_ebm.ipynb`](notebooks/02_hug_vs_ebm/nb02_hug_vs_ebm.ipynb) | Side-by-side comparison of HUGIML pattern profiles and EBM-style additive shape functions. |
| [`03_modeling_special_cases`](notebooks/03_modeling_special_cases/) | [`nb03_modeling_special_cases.ipynb`](notebooks/03_modeling_special_cases/nb03_modeling_special_cases.ipynb) | Practical modeling cases including multiclass targets, imbalance, high-cardinality categoricals, adaptive binning, and pruning workflows. |
| [`04_credit_risk`](notebooks/04_credit_risk/) | [`nb04_credit_risk.ipynb`](notebooks/04_credit_risk/nb04_credit_risk.ipynb) | Credit-risk governance example using German Credit-style data, scorecard-style features, and auditable risk patterns. |
| [`05_aml`](notebooks/05_aml/) | [`nb05_aml.ipynb`](notebooks/05_aml/nb05_aml.ipynb) | Anti-money-laundering example focused on suspicious transaction pattern discovery and model review artifacts. |
| [`06_mobile_money`](notebooks/06_mobile_money/) | [`nb06_mobile_money_fraud.ipynb`](notebooks/06_mobile_money/nb06_mobile_money_fraud.ipynb) | Mobile-money fraud example showing compact transaction-risk patterns and operational fraud-review signals. |
| [`07_basel_ca`](notebooks/07_basel_ca/) | [`nb07_basel_ca.ipynb`](notebooks/07_basel_ca/nb07_basel_ca.ipynb) | Basel capital-adequacy oriented example for regulated risk analytics and explainable model validation. |
| [`08_clinical`](notebooks/08_clinical/) | [`nb08_healthcare_breast_cancer.ipynb`](notebooks/08_clinical/nb08_healthcare_breast_cancer.ipynb) | Clinical classification example using breast-cancer features to demonstrate interpretable healthcare pattern explanations. |
| [`09_insurance`](notebooks/09_insurance/) | [`nb09_insurance_underwriting.ipynb`](notebooks/09_insurance/nb09_insurance_underwriting.ipynb) | Insurance underwriting example with risk-selection patterns and model-card-friendly feature narratives. |
| [`10_medicare`](notebooks/10_medicare/) | [`nb10_medicare_program_integrity.ipynb`](notebooks/10_medicare/nb10_medicare_program_integrity.ipynb) | Medicare program-integrity example for suspicious provider/claim behavior and audit-ready pattern summaries. |
| [`11_workforce_analytics`](notebooks/11_workforce_analytics/) | [`nb11_workforce_attrition.ipynb`](notebooks/11_workforce_analytics/nb11_workforce_attrition.ipynb) | Workforce attrition analytics example showing HR risk patterns, explanation tables, and governance-oriented summaries. |

### Observed results: public benchmark snapshot

![Benchmark comparison](docs/images/benchmark_comparison.png)

> The shipped benchmark runner supports multiple public datasets; the top validation visuals above emphasize healthcare and credit-risk examples.

| Model | AUC (mean±std) | Fit time/fold | Complexity budget | Remarks |
|---|---:|---:|---|---|
| HUG B=3 | 0.9907 ± 0.0031 | 0.32 s | **topK** patterns | `topK` is an explicit cap; actual mined patterns can be lower. |
| HUG B=5 | 0.9909 ± 0.0028 | 0.34 s | **topK** patterns | More bins per feature. |
| HUG adaptive | 0.9954 ± 0.0022 | 1.20 s | **topK** patterns | Per-feature B increases fit time. |
| EBM | 0.9940 ± 0.0025 | 11.0 s | Additive terms + interactions | Reference interpretable baseline. |
| XGBoost | 0.9882 ± 0.0040 | 0.12 s | Trees × leaves | High-performing ensemble; not directly pattern-interpretable. |
| LightGBM | 0.9921 ± 0.0028 | 0.07 s | Leaves × trees | Fast histogram boosting. |

### Missing value robustness

![Missing value benchmark](docs/images/missing_value_benchmark.png)

Simulation setup: Wine dataset, 3-fold stratified CV, missing rates 0–40%, mechanisms MCAR, MAR, and MNAR.

**Observations:**

- HUGIML fit time can decrease with more missing data because the transaction miner processes shorter transactions.
- Tree models often route missing values through learned default directions.
- Imputation-based pipelines may learn artifacts when the imputed distribution is stable in train/test but shifts in deployment.
- HUGIML’s “no item” semantics give interpretable absence: a pattern simply does not fire if one of its features is unavailable.

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
clf2 = load_model("model.hugiml")

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

Model cards should include top positive/negative patterns, missing-value behavior, calibration metrics, drift-monitoring plan, and any pattern-pruning audit trail.

---

## Calibration

```python
from hugiml.calibration import evaluate_calibration

result = evaluate_calibration(y_te.values, proba[:, 1])

print(f"ECE: {result.ece:.4f}")
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
| [`ci.yml`](.github/workflows/ci.yml) | Every push / PR | Lint, type-check, coverage gate, native tests, sanitizer build, benchmark regression, wheel build |
| [`release.yml`](.github/workflows/release.yml) | Git tag `v*.*.*` | Build platform wheels, generate SBOM, publish to PyPI, create GitHub 

---

## Repository Structure

```text
hugiml-core/
├── src/
│   ├── _native/                 C++ extension sources
│   └── hugiml/
│       ├── classifier.py        HUGIMLClassifierNative
│       ├── calibration.py       ECE, Brier, reliability diagrams
│       ├── explainability.py    SHAP bridge, feature lineage, stability
│       ├── governance.py        Model cards, audit artifacts
│       ├── monitoring.py        PredictionMonitor, DriftDetector
│       ├── serialization.py     save/load, SBOM, restricted unpickler
│       ├── telemetry.py         OpenTelemetry, Prometheus
│       ├── exceptions.py        Exception hierarchy
│       ├── metrics.py           Interpretability-complexity metrics
│       ├── plots.py             EBM-style profile visualisations
│       ├── pruning.py           Pattern editor + audit trail
│       ├── adaptive.py          Per-feature adaptive binning
│       ├── multiclass.py        Multiclass / imbalanced / encoding
│       └── benchmarks/          CV comparison suite
├── notebooks/                   Worked examples
├── tests/                       Pytest suite
├── benchmarks/                  Micro-benchmarks and regression gate
├── docker/                      Dockerfile + FastAPI inference server
├── kubernetes/                  Deployment manifests
├── scripts/                     Build and utility scripts
├── docs/                        Documentation and model-card templates
├── .github/workflows/           CI/CD pipelines
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
  title   = {Interpretable Classifier Models for Decision Support Using High Utility Gain Patterns},
  journal = {IEEE Access},
  volume  = {12},
  pages   = {126088--126107},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.3455563}
}
```

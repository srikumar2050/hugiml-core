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

## Table of Contents

1. [What Is HUG-IML?](#what-is-hug-iml)
2. [Why HUG-IML?](#why-hug-iml)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Feature Modes](#feature-modes)
6. [Execution Modes](#execution-modes)
7. [Hyperparameter Search](#hyperparameter-search)
8. [RPTE — Higher-Order Interactions](#rpte--higher-order-interactions)
9. [Governance Studio Dashboard](#governance-studio-dashboard)
10. [LLM Assistant](#llm-assistant)
11. [Augmented Pair Features](#augmented-pair-features)
12. [Adaptive Binning](#adaptive-binning)
13. [Missing Value Handling](#missing-value-handling)
14. [Model Explanation and Visualisations](#model-explanation-and-visualisations)
15. [Native Mining Pruning Controls](#native-mining-pruning-controls)
16. [Pattern Pruning](#pattern-pruning)
17. [Interpretability Metrics](#interpretability-metrics)
18. [Multiclass, Imbalanced Data, High-Cardinality](#multiclass-imbalanced-data-high-cardinality)
19. [Drift Detection & Monitoring](#drift-detection--monitoring)
20. [Calibration](#calibration)
21. [Serialisation](#serialisation)
22. [Governance & Model Cards](#governance--model-cards)
23. [Benchmark Suite](#benchmark-suite)
24. [Validation Highlights](#validation-highlights)
25. [Inference Server](#inference-server)
26. [CI / CD](#ci--cd)
27. [Repository Structure](#repository-structure)
28. [License](#license)
29. [Citation](#citation)

---

## What Is HUG-IML?

The **High Utility Gain Interpretable Machine Learning (HUG-IML)** framework extracts *High Utility Gain patterns* from labelled tabular data, transforms the input into a binary pattern-presence matrix, and fits an interpretable downstream classifier on that matrix — logistic regression by default for a single fit, or, when tuning with the default `performance_ho` grid, either logistic regression or RPTE (Residual Pattern Tree Ensemble, an optional constrained higher-order rule ensemble — see [RPTE — Higher-Order Interactions](#rpte--higher-order-interactions)), whichever scores better.

The resulting patterns are human-readable and serve as the primary source of model explanations, making the system suitable for regulated domains such as credit scoring, healthcare, and risk management.

**Key references:**

> Krishnamoorthy, S. (2024). Interpretable Classifier Models for Decision
> Support Using High Utility Gain Patterns. *IEEE Access*, 12, 126088–126107.
> DOI: [10.1109/ACCESS.2024.3455563](https://doi.org/10.1109/ACCESS.2024.3455563)

The interaction-aware extensions such as adaptive discretisation, interaction-supported
pattern admission, explicit pair terms, and bounded explanation budgets are
described in [Complexity-Budgeted, Interaction-Aware Interpretable Model for
Tabular Data](https://arxiv.org/abs/2607.07060).

---

## Why HUG-IML?

Most interpretable ML methods force a trade-off: either the model is readable but its complexity grows with the data (EBMs scale as features x bins plus interaction terms), or it performs well but explanations are post-hoc approximations of a black box (SHAP on XGBoost). HUG-IML sidesteps both problems. The model's complexity budget is always exactly **topK patterns**, regardless of how many features or bins the dataset has. Each pattern is a human-readable conjunction of intervals and categories. There is no separate explanation layer to trust or validate: the patterns *are* the learned representation, and a logistic regression over them *is* the classifier. On standard benchmarks, HUG-IML matches or exceeds EBM and XGBoost accuracy at a fraction of the complexity budget (see [Benchmark Suite](#benchmark-suite) and [Validation Highlights](#validation-highlights)).

This design makes HUG-IML a natural fit for regulated domains like credit scoring, healthcare risk, insurance underwriting, and AML, where reviewers need to inspect, edit, and sign off on individual model components. The [pattern pruning](#pattern-pruning) workflow lets analysts remove patterns that reference protected attributes or show drift, refit, recalibrate, and produce a full JSON audit trail, all within a single API.

---

## Installation

```bash
# Core
pip install hugiml-core

# With profile plots
pip install "hugiml-core[plots]"

# With the Dash Governance Studio and dashboard dependencies
pip install "hugiml-core[dashboard]"

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

**Build from source** requires a C++17 compiler. The recommended source-build path uses the build requirements declared in `pyproject.toml` so that `pybind11` is installed before compilation:

```bash
git clone https://github.com/srikumar2050/hugiml-core.git
cd hugiml-core
python -m pip install -e ".[dev]"
python scripts/build_batched.py --inplace
```

The helper enables conservative native build batching by default (`HUGIML_BUILD_BATCH_SIZE=4`, `HUGIML_BUILD_JOBS=2`) to avoid memory spikes in constrained environments. Direct `python setup.py build_ext --inplace` still works when local build requirements are already installed. Avoid `--no-build-isolation` unless `pybind11`, `setuptools`, and `wheel` are already available in the active environment.

Recommended local validation uses deterministic pytest batches instead of one long pytest process:

```bash
python scripts/run_pytest_batches.py
ruff check .
```

`make build`, `make test`, `make lint`, and `make validate` provide the same default workflows when `make` is available.

---

## Quick Start

`HUGIMLClassifier` is the primary public class name. `HUGIMLClassifierNative` remains available as a backward-compatible alias for existing code.

> **Note on `prepareXy`:** `prepareXy` performs schema and type preparation
> only — it detects integer, float, and categorical columns and encodes the
> target. Discretisation, HUG pattern mining, and downstream classifier fitting
> occur inside `fit()` on the training data supplied to that call.

### Path A — `prepareXy`

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from hugiml import HUGIMLClassifier

clf = HUGIMLClassifier(adaptive_binning=True, L=1, G=5e-3, topK=100)

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
from hugiml import HUGIMLClassifier

clf = HUGIMLClassifier(
    allCols=[int_col_names, float_col_names, cat_col_names],
    origColumns=X.columns.tolist(),
    B=-1,
    adaptive_binning=True,
    b_candidates=[2, 3, 5, 7, 10, 15],
    L=1,
    G=1e-5,
    topK=150,
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)
```

---

## Feature Modes

HUGIML can use the mined binary pattern matrix in three downstream feature modes. The default remains pattern-only behavior, so existing code keeps the same high-interpretability semantics unless `feature_mode` is set explicitly.

| `feature_mode` | Downstream estimator input | When to use |
|---|---|---|
| `"patterns_only"` | HUGIML binary pattern matrix only | Standard HUGIML; best when the mined pattern space itself captures the decision boundary. |
| `"original_plus_patterns"` | Original features plus all mined binary patterns | Useful when original features contain strong marginal signal and HUGIML patterns add supervised nonlinear refinements. |
| `"original_plus_interactions"` | Original features plus only `L > 1` mined patterns | Useful when original features should handle marginal effects and HUGIML should contribute interaction/compound-region features only. |

The recommended tuning grid and configuration choices are described in [Hyperparameter Search](#hyperparameter-search). Start there for first-pass model selection, then select a representation based on interpretability and runtime needs.

```python
from hugiml import HUGIMLClassifier

# Backward-compatible default: pattern matrix only
clf = HUGIMLClassifier(B=-1, L=2, G=1e-2, topK=150,
                              adaptive_binning=True, feature_mode="patterns_only")

# Hybrid: original features + all binary HUGIML patterns
clf_hybrid = HUGIMLClassifier(B=-1, L=2, G=1e-2, topK=150,
                                    adaptive_binning=True, feature_mode="original_plus_patterns")

# Hybrid: original features + higher-order/interaction patterns only
clf_interactions = HUGIMLClassifier(B=-1, L=2, G=1e-2, topK=150,
                                          adaptive_binning=True,
                                          feature_mode="original_plus_interactions")
```

`transform(X)` always returns the HUGIML binary pattern matrix, regardless of `feature_mode`. The feature mode only changes the matrix passed to the downstream estimator inside `fit()`, `predict()`, `predict_proba()`, and `score()`.

For hybrid modes, HUGIML standardizes numeric original features internally before concatenating them with the sparse binary pattern matrix and any active augmented-pair columns. `feature_importances()`, `model_summary()`, and `get_model_composition()` report the downstream feature representation, while `get_hug_features()` and `get_pattern_info()` remain pattern-only APIs.

---

## Downstream Solver Support

When `base_estimator` is not supplied, HUGIML now exposes the built-in downstream linear classifier choice through `lr_solver`. The default remains `"auto"`, which preserves the historical behavior: binary problems use `LogisticRegression(solver="liblinear")`, and multiclass problems use `LogisticRegression(solver="lbfgs")`.

| `lr_solver` | Downstream estimator | When to use |
|---|---|---|
| `"auto"` | `LogisticRegression` with the historical binary/multiclass solver choice | Recommended default for most datasets and for backward-compatible results. |
| `"saga"` | `LogisticRegression(solver="saga")` | Useful for larger or sparse downstream matrices when you still want logistic-regression coefficients and probability estimates. |
| `"sgd"` | `SGDClassifier(loss="log_loss")` | Useful for very large downstream matrices where stochastic optimization can reduce memory pressure or wall-clock time. Validate accuracy because SGD can be more sensitive to scaling and convergence settings. |

All built-in solver choices keep deterministic defaults aligned with the existing classifier path: `random_state=0` and `max_iter=500`. If you need complete control over solver-specific hyperparameters, pass a fully configured `base_estimator`; that continues to override `lr_solver`.

A `base_estimator` is not limited to `LogisticRegression`/`SGDClassifier` — see [RPTE — Higher-Order Interactions](#rpte--higher-order-interactions) for the constrained residual-rule branch reachable the same way (and, by default, automatically considered through the `performance_ho` grid).

```python
from hugiml import HUGIMLClassifier

# Historical default
clf_default = HUGIMLClassifier(lr_solver="auto")

# LogisticRegression through saga
clf_saga = HUGIMLClassifier(lr_solver="saga", feature_mode="original_plus_patterns")

# Logistic loss through SGDClassifier
clf_sgd = HUGIMLClassifier(lr_solver="sgd", feature_mode="original_plus_patterns")
```

The versioned `.hugiml` serializer records `lr_solver` in the classifier initialization state and natively round-trips both `LogisticRegression` and the built-in `SGDClassifier` downstream estimator; an RPTE (or otherwise custom) `base_estimator` round-trips too, including the unfitted `base_estimator` hyperparameter itself — see [Serialization](#serialization) under the RPTE section above.

## Execution Modes

HUGIML supports two execution modes:

| `execution_mode` | Purpose | Behavior |
|---|---|---|
| `"audit"` | Default mode for development, validation, governance, and regulated review | Keeps the complete training and traceability artifacts needed by audit, governance, and dashboard APIs. |
| `"production"` | Lean mode for deployment after validation | Keeps prediction, probability scoring, save, and load behavior, while dropping training/audit-heavy artifacts to reduce retained memory. |

```python
from hugiml import HUGIMLClassifier

# Full traceability; this is the default.
audit_model = HUGIMLClassifier(execution_mode="audit")
audit_model.fit(X_train, y_train)

# Lean retained state for deployment.
prod_model = HUGIMLClassifier(execution_mode="production")
prod_model.fit(X_train, y_train)
prod_model.save_model("model.hugiml")
loaded = HUGIMLClassifier.load_model("model.hugiml")
```

In production mode, audit-oriented methods return a clear guidance result or raise a clear message asking you to refit with `execution_mode="audit"` when complete traceability is required.

---

## Hyperparameter Search

HUGIML provides a fast cached tuning path for adaptive-binning grids. When `adaptive_binning=True`, the binning and transaction construction work is reused across eligible candidates, so compact grids can be evaluated without rebuilding the same mining inputs repeatedly.

### Recommended named parameter grids

HUGIML tuning reads four named grids from `hugiml.hyperparameter_configs`. **`"performance_ho"` is the default**: it evaluates the eight `L × topK × G` configurations used by `"performance"` with both the built-in LR branch and the adaptive RPTE branch, for 16 candidates. It sets `feature_mode="original_plus_patterns"`, enables augmented pairs, keeps numeric 0/1 columns numeric with `convert_binary_to_categorical=False`, and keeps `topk_budget_strict=False`. Use `"performance"` for the same mining search with LR only, `"interpretability"` for pattern-only relaxed mining with LR, or `"interpretability_ho"` for the same pattern-only surface with LR versus sequential RPTE. Both interaction-relaxed grids set `convert_binary_to_categorical=True` so numeric 0/1 indicators enter the categorical pattern surface.

```python
from hugiml import HUGIMLClassifier

default_grid = HUGIMLClassifier.default_param_grid()          # performance_ho
performance_grid = HUGIMLClassifier.default_param_grid("performance")
interpretability_grid = HUGIMLClassifier.default_param_grid("interpretability")
interpretability_ho_grid = HUGIMLClassifier.default_param_grid("interpretability_ho")

# Equivalent default (performance_ho) grid:
from sklearn.multiclass import OneVsRestClassifier
from hugiml.rpte_bounded_lookahead_leafwise import LeafWiseBoundedLookaheadRPTEFeatureLR

default_grid = {
    "B": [-1],
    "adaptive_binning": [True],
    "L": [1, 2],
    "topK": [50, 100],
    "feature_mode": ["original_plus_patterns"],
    "G": [0.01, 0.001],
    "convert_binary_to_categorical": [False],
    "augmented_pair_transforms": [True],
    "topk_budget_strict": [False],
    "base_estimator": [
        None,  # HUGIML's built-in logistic-regression branch
        OneVsRestClassifier(
            LeafWiseBoundedLookaheadRPTEFeatureLR(
                leaf_config="3xD", depth=4, enable_lookahead="adaptive"
            ),
            n_jobs=1,
        ),
    ],
}

# Equivalent performance grid (LR-only, no RPTE search):
performance_grid = {
    "B": [-1],
    "adaptive_binning": [True],
    "L": [1, 2],
    "topK": [50, 100],
    "feature_mode": ["original_plus_patterns"],
    "G": [0.01, 0.001],
    "convert_binary_to_categorical": [False],
}

# Equivalent interpretability grid:
interpretability_grid = {
    "B": [-1],
    "adaptive_binning": [True],
    "L": [1, 2],
    "topK": [50, 100],
    "feature_mode": ["patterns_only"],
    "G": [0.01, 0.001],
    "interaction_relaxed_mining": [True],
    "convert_binary_to_categorical": [True],
    "augmented_pair_transforms": [False],
}

# Equivalent interpretability_ho grid:
interpretability_ho_grid = {
    **interpretability_grid,
    "topk_budget_strict": [False],
    "base_estimator": [
        None,
        OneVsRestClassifier(
            LeafWiseBoundedLookaheadRPTEFeatureLR(
                leaf_config="3xD", depth=4, enable_lookahead=False
            ),
            n_jobs=1,
        ),
    ],
}
```

| Grid | Recommended use | Main values |
|---|---|---|
| `performance_ho` **(default)** | First-pass tuning that also considers higher-order interactions | Same mining budget as `performance`, `convert_binary_to_categorical=False`, plus `base_estimator=[None, RPTE(...)]` |
| `performance` | First-pass predictive tuning, linear downstream only | `feature_mode=["original_plus_patterns"]`, `convert_binary_to_categorical=False`, `L=[1,2]`, `topK=[50,100]`, `G=[0.01,0.001]` |
| `interpretability` | Pattern-only representation review | `feature_mode=["patterns_only"]`, `interaction_relaxed_mining=True`, `convert_binary_to_categorical=True`, `augmented_pair_transforms=False` |
| `interpretability_ho` | Pattern-only higher-order review | Same binary-aware mining surface as `interpretability`, plus `base_estimator=[None, sequential RPTE(...)]`, `enable_lookahead=False` |

All four grids keep `B=[-1]` and `adaptive_binning=[True]`, so each numerical feature chooses a supervised bin count. The performance/augmented-pair grids explicitly use `convert_binary_to_categorical=False`; the interpretability/interaction-relaxed grids explicitly use `True`. Do not enable `interaction_relaxed_mining=True` and `augmented_pair_transforms=True` in the same `L >= 2` candidate.

Use focused follow-up grids when you want to explore interaction-relaxed mining or augmented-pair transforms.

### `tune()` — cross-validated search with automatic fast path

```python
result = HUGIMLClassifier.tune(
    X, y,
    param_grid="performance_ho",  # default; omit param_grid for the same effect
    cv=5,
    shuffle=True,
    random_state=42,
    scoring="roc_auc",
    refit=True,
)

print(result.best_params_)
print(f"CV score: {result.best_score_:.4f}")
print(f"Fast path used: {result.fast_path_used_}")

best_model = result.best_estimator_

# If RPTE wins, print the readable tree or use structured rows.
if hasattr(best_model, "rpte_rule_tree"):
    print(best_model.rpte_rule_tree())
    rules = best_model.rpte_rule_table()
```

A custom grid is supplied via `param_grid`. For the cached adaptive-binning path, keep the varying dimensions compact and centered on mining or representation choices such as `G`, `L`, `topK`, and `feature_mode`. Fixed values such as `B=-1` and `adaptive_binning=True` may be included for clarity.

```python
custom_grid = {
    "B": [-1],
    "adaptive_binning": [True],
    "G": [1e-2, 5e-3],
    "L": [1, 2],
    "topK": [50, 100],
    "feature_mode": ["patterns_only", "original_plus_patterns"],
}

result = HUGIMLClassifier.tune(
    X, y,
    param_grid=custom_grid,
    cv=3,
    scoring="roc_auc",
    refit=True,
)
```

### Choosing the model configuration

After the default grid identifies a useful budget range, choose one of these focused configurations based on the representation you want.

| Option | Feature mode | Interaction path | Extra downstream pair columns? | Interpretability | Runtime profile | Good default when... |
|---|---|---|---|---|---|---|
| Pure HUG patterns | `patterns_only` | Standard `L=1` or `L=2` mining | No | Very high | Lowest to moderate | You want the simplest pattern-only model. |
| Patterns + interaction-relaxed mining | `patterns_only` | `interaction_relaxed_mining=True` | No | Very high | Higher than augmented pairs | You want interaction evidence to affect HUG pattern discovery without adding a new feature family. |
| Patterns + augmented pairs | `patterns_only` | `augmented_pair_transforms=True` | Yes | High | Often faster than relaxed mining | You want selected pair evidence with better runtime control. |
| Originals + patterns | `original_plus_patterns` | Standard `L=1` or `L=2` mining | No | High | Moderate | Original variables have strong marginal signal and patterns add readable refinements. |
| Originals + patterns + relaxed mining | `original_plus_patterns` | `interaction_relaxed_mining=True` | No | High | Higher than augmented pairs | You want original features plus survivor-led HUG patterns, but no pair-operator columns. |
| Originals + patterns + augmented pairs | `original_plus_patterns` | `augmented_pair_transforms=True` | Yes | Moderate | Moderate to higher | You want the highest representation capacity among the recommended options. |

A **survivor** is a source feature that remains after interaction-information screening. It may not be one of the strongest features by itself, but it has useful pairwise or synergy evidence with another feature. In interaction-relaxed mining, these survivor source features are allowed to participate in native HUG pattern mining. A survivor is not automatically a final model feature; it is a candidate source that can help form mined patterns.

`interaction_relaxed_mining=True` relaxes the usual entry path for interaction-useful source features. Instead of adding product, difference, or sum columns to the downstream estimator, it lets a small survivor pool enter the native mining step, so the final representation remains HUG patterns plus any original features selected by `feature_mode`.

Use these focused follow-up grids:

```python
# Pattern-only with interaction-relaxed mining.
patterns_relaxed_grid = {
    "B": [-1],
    "adaptive_binning": [True],
    "L": [2],
    "G": [1e-2, 5e-3],
    "topK": [50, 100],
    "feature_mode": ["patterns_only"],
    "convert_binary_to_categorical": [True],
    "augmented_pair_transforms": [False],
    "interaction_relaxed_mining": [True],
    "interaction_relaxed_feature_size": [8, 12],
}

# Pattern-only with augmented pair features.
patterns_augmented_grid = {
    "B": [-1],
    "adaptive_binning": [True],
    "L": [2],
    "G": [1e-2, 5e-3],
    "topK": [50, 100],
    "feature_mode": ["patterns_only"],
    "convert_binary_to_categorical": [False],
    "augmented_pair_transforms": [True],
    "augmented_pair_mode": ["interaction_information"],
    "aug_feature_size": [8, 12],
}

# Originals plus patterns with interaction-relaxed mining.
originals_relaxed_grid = {
    "B": [-1],
    "adaptive_binning": [True],
    "L": [2],
    "G": [1e-2, 5e-3],
    "topK": [50, 100],
    "feature_mode": ["original_plus_patterns"],
    "convert_binary_to_categorical": [True],
    "augmented_pair_transforms": [False],
    "interaction_relaxed_mining": [True],
    "interaction_relaxed_feature_size": [8, 12],
}

# Originals plus patterns with augmented pair features.
originals_augmented_grid = {
    "B": [-1],
    "adaptive_binning": [True],
    "L": [2],
    "G": [1e-2, 5e-3],
    "topK": [50, 100],
    "feature_mode": ["original_plus_patterns"],
    "convert_binary_to_categorical": [False],
    "augmented_pair_transforms": [True],
    "augmented_pair_mode": ["interaction_information"],
    "aug_feature_size": [8, 12],
}
```

### `fast_grid_tune()` — single-split cached path for custom CV loops

```python
tune_result = HUGIMLClassifier.fast_grid_tune(
    X_train, y_train,
    X_val,   y_val,
    param_grid="performance_ho",  # default
    scoring="roc_auc",
    refit_full=False,
)

print(tune_result["best_params"])
print(f"Validation score: {tune_result['best_score']:.4f}")
```

---

## RPTE: Higher-Order Interactions

**Residual Pattern Tree Ensemble** (**RPTE**) is an optional downstream model for cases where individual HUG patterns or original features are readable, but their joint effect is not adequately represented by one linear coefficient.

RPTE is not a conventional tree-voting ensemble. Its trees create a small rule representation, and a final sparse logistic regression assigns the prediction weights.

### What keeps RPTE interpretable

- **Residual construction.** Each accepted tree is trained on signal left after the earlier trees, so later trees have a distinct role.
- **Source ownership across trees.** Once an accepted tree uses a raw source, later trees do not use that source again through another original, pattern, or pair-derived column.
- **Bounded trees.** Depth and leaf budgets limit the length and number of displayed rules.
- **One reached leaf per tree.** Every case follows one deterministic path through each accepted tree and reaches exactly one terminal leaf. The tree uses only its selected splits; it does not enumerate every possible input combination.
- **No repeated input column.** A downstream input column used in an accepted tree is not also included as a direct term in the final logistic layer.
- **Additive final score.** The prediction is the logistic intercept plus one reached-leaf coefficient from each accepted tree and any retained direct terms.

### Breast-cancer example generated by the package

The following example uses six measurements from scikit-learn's Breast Cancer Wisconsin dataset. The target is recoded so **Class 1 means malignant**. The small tree budget keeps the rendered output compact.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from hugiml import HUGIMLClassifier
from hugiml.rpte_bounded_lookahead_leafwise import (
    LeafWiseBoundedLookaheadRPTEFeatureLR,
)

data = load_breast_cancer(as_frame=True)
columns = [
    "mean radius",
    "mean texture",
    "mean concavity",
    "worst radius",
    "worst concavity",
    "worst symmetry",
]
X = data.data[columns]
y = (data.target == 0).astype(int)  # malignant = 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

clf = HUGIMLClassifier(
    B=-1,
    adaptive_binning=True,
    b_candidates=[3, 5, 7],
    L=1,
    G=0.5,
    topK=6,
    feature_mode="original_plus_patterns",
    augmented_pair_transforms=False,
    base_estimator=LeafWiseBoundedLookaheadRPTEFeatureLR(
        leaf_config="2xD",
        depth=2,
        n_estimators=2,
        min_samples_leaf=20,
        enable_lookahead=False,
        random_state=42,
    ),
)
clf.fit(X_train, y_train)

print(clf.rpte_rule_tree(detail_level="compact", precision=3))
```

This produces the following package-rendered representation:

```text
Class 1 | Tree 0 | sequential RPTE
ROOT
├── worst radius <= 16.795
│   ├── mean concavity <= 0.11185
│   │   └── LEAF 3  |  beta=-2.67  |  odds x0.0695  |  support=59.4% (n=253)
│   └── mean concavity > 0.11185
│       └── LEAF 4  |  beta=+0  |  odds x1  |  support=6.3% (n=27)
└── worst radius > 16.795
    ├── mean concavity <= 0.07344
    │   └── LEAF 5  |  beta=+0  |  odds x1  |  support=4.7% (n=20)
    └── mean concavity > 0.07344
        └── LEAF 6  |  beta=+3.74  |  odds x42.3  |  support=29.6% (n=126)

Class 1 | Tree 1 | sequential RPTE
ROOT
├── mean radius <= 15.045
│   ├── worst concavity <= 0.3663
│   │   └── LEAF 3  |  beta=-1.24  |  odds x0.29  |  support=59.6% (n=254)
│   └── worst concavity > 0.3663
│       ├── worst symmetry <= 0.33645
│       │   └── LEAF 5  |  beta=+0  |  odds x1  |  support=4.7% (n=20)
│       └── worst symmetry > 0.33645
│           └── LEAF 6  |  beta=+1.37  |  odds x3.94  |  support=4.9% (n=21)
└── mean radius > 15.045
    └── LEAF 2  |  beta=+0.395  |  odds x1.48  |  support=30.8% (n=131)

Class 1 | Direct source terms
DIRECT SOURCE TERMS
└── Original features
    └── linear value of mean texture  |  beta=+1.14  |  odds x3.13
```

### Reading the example

Tree 0 uses `worst radius` and `mean concavity`. A case with `worst radius > 16.795` and `mean concavity > 0.07344` reaches Leaf 6, whose positive coefficient raises the malignant-class log-odds. A case with `worst radius <= 16.795` and `mean concavity <= 0.11185` reaches Leaf 3, whose negative coefficient lowers them.

Tree 1 uses a different source set: `mean radius`, `worst concavity`, and `worst symmetry`. The remaining `mean texture` input is retained as a direct linear term. For any case, the final logistic layer combines exactly one reached leaf from Tree 0, one reached leaf from Tree 1, and the direct `mean texture` term.

A leaf with `beta=0` is still a valid terminal region, but the final sparse logistic regression assigns it no additional contribution after considering the other reached leaves and direct terms.

The renderer reports:

- the complete root-to-leaf conditions;
- `beta`, the leaf or direct-term coefficient in the final logistic model;
- the corresponding odds multiplier; and
- training support for every leaf.

Use `rpte_rule_table()` when the same evidence is needed as structured rows for governance checks, filtering, or export.

### Using RPTE

The default `performance_ho` grid compares the built-in logistic branch with an RPTE branch and retains the better cross-validated model. The `interpretability_ho` grid performs the same comparison on the interaction-relaxed, pattern-only representation using sequential RPTE.

RPTE can also be supplied directly:

```python
from hugiml import HUGIMLClassifier
from hugiml.rpte_bounded_lookahead_leafwise import LeafWiseBoundedLookaheadRPTEFeatureLR

clf = HUGIMLClassifier(
    L=1,
    topK=100,
    feature_mode="original_plus_patterns",
    base_estimator=LeafWiseBoundedLookaheadRPTEFeatureLR(
        leaf_config="3xD",
        depth=4,
        enable_lookahead="adaptive",
    ),
)
clf.fit(X_train, y_train)

print(clf.rpte_rule_tree(detail_level="compact"))
rules = clf.rpte_rule_table()
```

For multiclass targets, wrap RPTE in `sklearn.multiclass.OneVsRestClassifier`; the named higher-order grids already do this.

### Serialization

RPTE-based models are persisted through `hugiml.serialization.save_model()` and `load_model()` using the versioned `.hugiml` archive format. The archive retains the fitted trees, final logistic layer, and reconstructable `base_estimator` configuration. See [Serialisation](#serialisation) for details.

---

## Governance Studio Dashboard

The **HUGIML Governance Studio** is a Dash-based interface for configuring model
experiments, comparing candidate runs, inspecting fitted representations, and
promoting a selected HUGIML model into governance review. The default Ocean
theme and compact Workbench/Governance navigation are designed for desktop use;
Forest and Dark themes are also available.

### Installation

```bash
pip install "hugiml-core[dashboard]"
```

### Launch

```bash
# Dash interface (default); opens the browser when the server is ready
hugiml-dashboard

# Common options
hugiml-dashboard --port 8050 --cv 5 --random-state 42
hugiml-dashboard --no-open

# Lightweight Streamlit interface
hugiml-dashboard --ui light

# Source-tree development
python -m hugiml.dashboard.dash_app --port 8050
```

The Dash launcher uses `127.0.0.1:8050` by default. Use `--host`, `--port`,
`--debug`, or `--no-open` as needed.

### Main areas

| Area | What it supports |
|---|---|
| **Workbench · Setup** | Demo or uploaded data, column roles, model selection, comma-separated parameter candidates, downstream LR/RPTE choices, and the total expanded model-combination count |
| **Workbench · Results** | Leaderboard, ROC/PR comparisons, interpretability summaries, artifact comparison, RPTE tree/leaf inspection, and model drill-down |
| **Governance** | Overview, validation, representation audit, pattern inventory, case review, data quality and policy, configuration comparison, representation pruning, and monitoring |

### Governance evidence

| View | What it shows |
|---|---|
| **Representation Audit** | Fitted source families, accepted RPTE splits, individual trees and leaves, and final LR leaf/direct-term composition |
| **Pattern Inventory** | Tree-used and direct patterns, coefficients, support, utility, information gain, and coverage |
| **Case Review** | Row-level probability and active leaf/direct-term contributions |
| **Data Quality & Policy** | Missingness, sensitive/proxy use, and policy checks |
| **Configuration Comparison** | Side-by-side model settings, downstream estimator, performance, and representation size |
| **Representation Pruning** | Original, pattern, or augmented representation pruning; raw-input exclusion rebuilds the full pipeline |
| **Monitoring** | PSI, KL-divergence, leaf activation, direct-term, and representation-stability signals |

Uploaded inputs may be CSV, TSV, Excel, or Parquet. Numeric two-value columns
remain numeric unless `convert_binary_to_categorical=True`; the named grids set
this explicitly according to their mining design.

### Demo preview

- [Open the HUGIML Governance Studio Demo](https://srikumar2050.github.io/hugiml-core/hugiml_governance_studio_demo.html)

---

## LLM Assistant

The **HUGIML LLM Assistant** is a chat-first Streamlit interface for working with HUGIML models and documentation from one place. It is designed for model-review workflows where a user asks natural-language questions, sees a structured answer, and then continues with follow-up questions in the same review session.

Unlike a general chatbot, the assistant grounds its responses in two HUGIML-specific sources:

| Query type | Grounding source | Typical output |
|---|---|---|
| **Model/run questions** | The fitted classifier, run artifacts, pattern inventory, validation metrics, pruning state, and governance metadata | Decision-maker summaries, key model drivers, risk/audit checks, and next actions |
| **API/documentation questions** | The local Sphinx/API documentation index, with fallback to package README and public API docstrings | Condensed API guidance, usage examples, parameter notes, and caveats |
| **Mixed questions** | Both model artifacts and documentation | Practical guidance that explains what the current model shows and which API or governance action applies |

### Launch

```bash
# Installed console script
hugiml-llm

# Source-tree development
python -m streamlit run src/hugiml/llm/ui_app.py
```

The UI uses a single Q&A-style view. The session appears as a chronological transcript, the follow-up input stays inline below the latest answer, and quick model/run details are available in expandable panes below the chat.

### Local model policy

The assistant can run in deterministic mode, or use a supported local Ollama model as a writer/synthesis layer over retrieved HUGIML evidence. The default policy favours small models that work on modest local machines, while still exposing larger configured models when they are installed and memory is available.

| Role | Model | Purpose |
|---|---|---|
| **Default** | `qwen3:1.7b` | Primary local writer for grounded API and model-review answers |
| **Light mode** | `gemma3:1b` | Lower-memory local response generation |
| **Fallback** | `llama3.2:1b` | Retry model before falling back to deterministic routing |
| **Deterministic router** | built in | Always available when no supported local model is selected or available |

Additional configured Ollama models remain visible in the selector when available:

| Model | Profile |
|---|---|
| `llama3.2:3b` | Minimum larger LLM |
| `qwen3:4b` | Balanced local LLM |
| `gemma3:4b` | Balanced alternative |
| `qwen3:8b` | Stronger local LLM |
| `gemma3:12b` | Large-context local LLM |

The selector uses available system memory as a safety check. These thresholds are available-RAM requirements, not model file sizes:

| Model | Role / profile | Minimum available RAM |
|---|---|---:|
| `qwen3:1.7b` | Default local LLM | 5.0 GB |
| `gemma3:1b` | Light mode | 3.5 GB |
| `llama3.2:1b` | Fallback before deterministic routing | 3.5 GB |
| `llama3.2:3b` | Minimum larger LLM | 6.0 GB |
| `qwen3:4b` | Balanced local LLM | 10.0 GB |
| `gemma3:4b` | Balanced alternative | 10.0 GB |
| `qwen3:8b` | Stronger local LLM | 16.0 GB |
| `gemma3:12b` | Large-context local LLM | 32.0 GB |

Supported tiny models are shown explicitly. Other extra small Ollama models that are not part of the configured policy are omitted from the selector, while deterministic routing remains available without Ollama.

Install the preferred Ollama models before launching the assistant:

```bash
ollama pull qwen3:1.7b
ollama pull gemma3:1b
ollama pull llama3.2:1b
```

Optional larger models can also be installed and selected when the local machine has enough available memory:

```bash
ollama pull llama3.2:3b
ollama pull qwen3:4b
ollama pull gemma3:4b
ollama pull qwen3:8b
ollama pull gemma3:12b
```

Only configured supported models are listed in the UI. Extra experimental or unsupported small Ollama models that may be installed locally are omitted from the selector.

### Documentation-aware answers

For API questions such as “what hyperparameters should I tune?”, “how does pruning work?”, or “what governance artifacts are created?”, the deterministic runner builds a lightweight local index over the Sphinx documentation and package docs. It retrieves the relevant sections internally, then presents a concise, structured response with the API details needed to act.

### Model-review answers

For run-specific questions such as “summarize findings”, “what changed after pruning?”, or “is this model ready for governance review?”, the assistant analyzes the active classifier outputs and produces a structured decision summary covering performance, main evidence drivers, interpretability constraints, audit concerns, and recommended next steps.

### Static examples

These GitHub Pages examples show the intended Q&A-style review flow:

- [LLM lending credit-risk example](https://srikumar2050.github.io/hugiml-core/LLM/examples/lending_credit_risk.html)
- [LLM card default example](https://srikumar2050.github.io/hugiml-core/LLM/examples/card_default_taiwan.html)

---

## Augmented Pair Features

For interaction-oriented models, HUGIML can add native augmented-pair features to the downstream estimator. These are continuous product or absolute-difference transforms built from informative numeric features, for example:

```text
glucose * bmi
abs(age - duration)
```

They are active when `L > 1`, `adaptive_binning=True`, and `augmented_pair_transforms=True` (the default). They are appended only to the downstream estimator; the mined HUG pattern matrix and `transform(X)` remain pattern-space APIs.

The default `augmented_pair_mode="interaction_information"` scores candidate source columns using pair context before building product, absolute-difference, sum, and signed-difference features. Set `augmented_pair_mode="marginal_ig"` to use the v1.1.11 marginal-information-gain source selection behavior. `aug_feature_size` controls how many source columns are retained in interaction-information mode; `ii_partner_size` optionally bounds partner search; `max_pair_features` controls the source budget for marginal-IG mode.

```python
clf = HUGIMLClassifier(
    B=-1,
    adaptive_binning=True,
    L=2,
    topK=50,
    G=1e-2,
    feature_mode="original_plus_patterns",
    augmented_pair_transforms=True,
    augmented_pair_mode="interaction_information",
    aug_feature_size=10,
    topk_budget_strict=True,
)
clf.fit(X_train, y_train)

print(clf.get_model_composition())
print(clf.explain_augmented_pair_effects())
```

For selected pair features, HUGIML reports the raw formula, standardized formula, observed-row coverage, missing-pair policy, and raw-scale coefficient interpretation.

---

## Adaptive Binning

The global `B` parameter controls how many quantile bins each numerical feature is discretised into. Adaptive binning selects the optimal bin count per feature via supervised information-gain search and elbow stopping. For larger datasets, `adaptive_binning_sample_frac` can choose bin counts from a deterministic stratified row sample, then apply the selected bin edges to the full training data.

```python
from hugiml.adaptive import HUGIMLAdaptive

clf = HUGIMLAdaptive(b_candidates=[3, 5, 7, 10, 15], L=2, G=1e-2)

X_enc, y_enc = clf.prepareXy(X_df, y)
clf.fit(X_tr, y_tr)

print(clf.per_feature_b_)
clf.plot_bin_profiles()
clf.ig_heatmap()
```

Alternatively, enable adaptive binning directly on `HUGIMLClassifier`:

```python
from hugiml import HUGIMLClassifier

clf = HUGIMLClassifier(
    adaptive_binning=True,
    b_candidates=[3, 5, 7, 10],
    min_marginal_gain_ratio=0.02,
    adaptive_binning_sample_frac=0.20,  # optional for large adaptive-binning runs
)
```

**How it works:** for each numerical feature, HUGIML evaluates information gain at candidate `B` values and stops when the marginal gain falls below `min_marginal_gain_ratio × current_IG`. This prevents blindly selecting the maximum bin count. Set `adaptive_binning_sample_frac=False` for full-data bin selection, or a float in `(0, 1]` to use a stratified sample for the selection step.

---

## Missing Value Handling

HUGIML treats NaN and Inf values as **not observed** — no imputation and no special parameter are required.

**How it works:** numerical columns are pre-binned at fit time. Non-finite cells become `np.nan` in the label array, and the C++ transaction builder skips them. The corresponding item is absent from the transaction. Patterns requiring that feature do not fire for that row.

```python
import numpy as np
from hugiml import HUGIMLClassifier

X_train.iloc[5, 2] = np.nan

clf = HUGIMLClassifier(B=5, L=2, G=1e-4)
clf.fit(X_train, y_train)

X_test.iloc[0, 0] = np.nan
proba = clf.predict_proba(X_test)       # scored using available feature items
```

### Mining Patterns About Missingness

To mine patterns that involve missingness (e.g., `Glucose_MISSING=1 AND HeartRate=[110,140]`), add binary missingness indicators as preprocessing features:

```python
def add_missingness_indicators(X, threshold=0.05):
    X_aug = X.copy()
    for col in X.columns:
        if X[col].isna().mean() > threshold:
            X_aug[f"{col}__MISSING"] = X[col].isna().astype(int)
    return X_aug

X_with_indicators = add_missingness_indicators(X_raw)
clf = HUGIMLClassifier(B=7, L=2, G=1e-4)
clf.fit(X_with_indicators, y)
```

The Governance Studio **Data Quality & Policy** view shows feature-level missingness rates alongside sensitive column review.

---

## Model Explanation and Visualisations

### Interactive Plotly dashboard

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

Each profile panel shows the learned bin/pattern behavior for a feature: utility or coefficient-like contribution per bin, with support overlay where available.

Existing example dashboards:

**Public tabular benchmark classification**
![Feature shape profiles — public tabular benchmark](docs/images/explanation_dashboard_bc.png)

**Credit risk scoring**
![Feature shape profiles — credit risk](docs/images/explanation_dashboard_credit.png)

- [Open the HUGIML Benchmark Analysis Dashboard](https://srikumar2050.github.io/hugiml-core/hugiml_benchmark_analysis_dashboard.html)

The static benchmark dashboard is reproducible from the repository source with
[`experiments/benchmark/benchmark_dashboard.py`](experiments/benchmark/benchmark_dashboard.py); see
[Benchmark Suite](#benchmark-suite) for the exact rerun and assemble commands.

### Profile visualisations

```python
plotter.plot_marginal_bin_profile("age", X=X_test).show()  # EBM-style 1-D shape function
plotter.plot_feature_combinations("age").show()             # Feature-combination view
plotter.plot_top_patterns(top_n=20).show()                  # Top patterns by importance
plotter.plot_active_patterns(X_test, sample_idx=0).show()   # Local explanation for one sample
```

---

## Native Mining Pruning Controls

This section covers native HUIM search pruning, which is different from the user-facing [Pattern Pruning](#pattern-pruning) workflow below. Native pruning controls how the C++ miner avoids unnecessary candidate work during `fit()` while preserving the same public model outputs.

| Pruning path | When it is active | What it does | User-facing controls |
|---|---|---|---|
| **LIU** | Active for compound-pattern mining (`L > 1`, including the `L=2` hot path). Bounded classifier mining uses exact candidate evidence before raising the utility floor. | Raises the utility threshold from locally strong candidate sequences so low-utility branches can be skipped earlier. | Tune `L`, `G`, and `topK`; there is no separate public LIU switch. |
| **LA** | Active during generic utility-list child construction when the current branch uses the ordinary utility-ranked path. | Stops building a child utility list once the remaining upper bound can no longer pass the current utility floor. | Tune `topK` and `G`; relaxed-root interaction branches bypass this utility-floor shortcut where needed. |
| **EUCS** | Considered for `L > 1` after the admitted item set is known. It is skipped for small, dense, or very wide pair spaces where the cache would not pay off. | Builds a pair co-occurrence utility cache and skips pair intersections whose pair-level utility cannot enter the retained set. | Environment variables below. |

EUCS is enabled by default for eligible `L > 1` native mining paths, but it has safety gates so small or dense workloads continue without the extra cache. The relevant environment variables are:

| Variable | Default | Meaning |
|---|---:|---|
| `HUGIML_EUCS_ENABLE` or `HUGIML_EUCS_ENABLED` | enabled | Set to `0`, `false`, `no`, `off`, `disable`, or `disabled` to disable EUCS. Set to `1`, `true`, `yes`, `on`, `enable`, or `enabled` to enable it. Invalid values keep the default. |
| `HUGIML_EUCS_MIN_ITEMS` | `32` | EUCS is skipped when the admitted item universe is this size or smaller. |
| `HUGIML_EUCS_MAX_CELLS` | `6000000` | Maximum pair-cache cells allowed before EUCS is skipped. |
| `HUGIML_EUCS_MAX_DENSITY` | `0.20` | Maximum observed active-item density allowed before EUCS is skipped. |

Typical users should leave these settings at their defaults and tune model-level parameters first: `L` for maximum pattern length, `G` for the information-gain gate, and `topK` for the retained pattern budget. EUCS controls are mainly useful when benchmarking native mining behavior or diagnosing a workload whose pair space is unusually sparse or dense.

---

## Pattern Pruning

In regulated domains, analysts often need to remove patterns that reference protected attributes, have high PSI, or are operationally invalid. HUGIML provides a controlled editing workflow with a JSON audit trail.

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

The **Representation Pruning** view in the Governance Studio provides an interactive version of this workflow without writing code.

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

## Multiclass, Imbalanced Data, High-Cardinality

### Multiclass Classification

```python
from hugiml.multiclass import MulticlassHUGReport

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

When categorical features have hundreds or thousands of unique values (ZIP codes, ICD-10 diagnoses, merchant IDs), grouping rare categories prevents combinatorial explosion in pattern mining:

```python
def reduce_high_cardinality(X, y, threshold=50, min_frequency=0.01):
    """Group rare categories (<min_frequency) as '__OTHER__' for high-cardinality columns."""
    X_reduced = X.copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        if X[col].nunique() <= threshold:
            continue
        value_counts = X[col].value_counts()
        min_count = len(X) * min_frequency
        rare_categories = value_counts[value_counts < min_count].index
        X_reduced[col] = X[col].apply(
            lambda x: "__OTHER__" if x in rare_categories else x
        )
    return X_reduced

X_reduced = reduce_high_cardinality(X_raw, y, threshold=50, min_frequency=0.01)
clf = HUGIMLClassifier(B=7, L=2, G=1e-4)
clf.fit(X_reduced, y)

# Or use built-in target encoding:
from hugiml.multiclass import encode_high_cardinality, apply_encoding
X_enc, enc_map = encode_high_cardinality(X_tr, y_tr, threshold=20, method="target_mean")
X_te_enc = apply_encoding(X_te, enc_map)
```

**Note:** Learn category groupings on training data only, then apply the same mapping to test/production data.

---

## Drift Detection & Monitoring

```python
clf.enable_monitoring(window_size=1000)

clf.predict_proba(X_new)

print(clf.monitor.report())

report = clf.detect_drift(X_new, current_labels=y_new)
print(report)
```

The **Monitoring** view in the Governance Studio shows PSI and KL-divergence drift signals per feature from the fitted model's training baseline.

---

## Calibration

```python
from hugiml.calibration import evaluate_calibration

result = evaluate_calibration(y_te.values, proba[:, 1])

print(f"ECE: {result.ece:.4f}")
print(f"Brier: {result.brier_score:.4f}")
```

---

## Serialisation

```python
from hugiml.serialization import save_model, load_model, generate_sbom

save_model(clf, "model.hugiml")
clf2 = load_model("model.hugiml")

sbom = generate_sbom(clf)
```

`save_model()`/`load_model()` fully round-trip RPTE-based models (schema version 9+), including the fitted trees, final logistic layer, and reconstructable `base_estimator` configuration. Built-in logistic-regression and SGD estimators are stored as JSON/NumPy records. RPTE estimator state uses the archive's allowlisted restricted-pickle payload, while the surrounding `.hugiml` artifact remains versioned and can be HMAC-authenticated when `HUGIML_MODEL_HMAC_KEY` is configured. `load_model()` accepts recognized HUGIML model artifacts; it does not load arbitrary raw pickle files. Earlier supported schema versions remain readable.

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

The **Governance Studio dashboard** provides interactive governance evidence views that complement programmatic model cards with visual audit artifacts.

---

## Benchmark Suite

HUGIML includes two reproducible benchmark workflows:

1. **Package benchmark runner** for quick CV-style comparisons from the installed package.
2. **Experiment dashboard runners** in [`experiments/`](experiments/) for regenerating the published static benchmark and scalability dashboards.

The package-level runner is useful for ad hoc benchmark checks:

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

### Reproduce the benchmark analysis dashboard

The public benchmark analysis dashboard is generated from
[`experiments/benchmark/benchmark_dashboard.py`](experiments/benchmark/benchmark_dashboard.py).
This script defines the 100-dataset panel (50 real-world and 50 synthetic tasks), model grids, preprocessing policy, checkpointing, result aggregation,
and static HTML assembly used for the dashboard.

The root-level `experiments/` runners and their documentation are included in the source distribution. They are not installed as importable wheel package modules; the installed wheel provides the `hugiml-bench` package runner described above.

From the repository root:

```bash
# Full fresh run; writes checkpoint, CSV summaries, and revised HTML
python experiments/benchmark/benchmark_dashboard.py --fresh

# Resume a partially completed run from checkpoint
python experiments/benchmark/benchmark_dashboard.py --resume

# Rebuild only the HTML/CSV summaries from an existing checkpoint
python experiments/benchmark/benchmark_dashboard.py --assemble
```

Default outputs are written under:

```text
experiments/benchmark/results/
```

The dashboard runner is deterministic for a fixed code version and dependency environment: dataset generation,
train/validation/test splits, row subsampling, and model seeds are all controlled by the script. The generated
artifacts include `details.csv`, `summary_by_scope.csv`, `scope_tests.csv`, `overall.csv`, and
`hugiml_benchmark_analysis_dashboard_revised.html`.

### Scalability dashboard

For runtime and memory scaling evidence, see the static scalability dashboard:

- [Open the HUGIML Scalability Dashboard](https://srikumar2050.github.io/hugiml-core/hugiml_scalability_dashboard.html)

The dashboard summarizes measured fit time, prediction latency, memory delta, pattern counts, and test AUC against XGBoost and LightGBM. It covers sample-size scaling, feature-count scaling, and parameter sweeps over `B`, `G`, `topK`, `L`, and adaptive binning. HUGIML retains many training and test artifacts to support governance and audit requirements.

The scalability dashboard is reproducible from
[`experiments/scalability/scalability_dashboard.py`](experiments/scalability/scalability_dashboard.py):

```bash
# Full scalability run with checkpointing
python experiments/scalability/scalability_dashboard.py --fresh

# Resume a partially completed scalability run
python experiments/scalability/scalability_dashboard.py --resume

# Rebuild only the static dashboard from an existing checkpoint
python experiments/scalability/scalability_dashboard.py --assemble

# Assemble with a privacy-sanitized reproducibility/SBOM manifest embedded in Methodology
python experiments/scalability/scalability_dashboard.py --assemble --include-sbom
```

Default outputs are written under the scalability results directory configured by the script and include the
JSON checkpoint, flat CSV export, and `hugiml_scalability_dashboard.html`. With `--include-sbom`, assembly also writes `scalability_reproducibility_sbom.json` and embeds the same privacy-sanitized manifest as a collapsed block under the Methodology tab.

Worked notebooks in [`notebooks/`](notebooks/) are organized as 12 self-contained folders:

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

---

## Validation Highlights

> The finance panels use German Credit / HELOC-style risk features such as loan duration, credit amount, checking status, and repayment-risk signals. The healthcare panels use Pima diabetes-style features such as glucose, BMI, pregnancies, pedigree, and age.

### HUGIML vs EBM shape profiles

<p align="left">
  <img src="docs/images/shape-profiles-hugiml-vs-ebm.png" alt="HUGIML native shape profiles compared with EBM shape functions" width="700"  height="300">
</p>

EBM is excellent for smooth effect inspection; HUGIML is strong when the explanation needs to be reviewed as a set of readable thresholds and pattern contributions.

### Real-world and synthetic benchmarks

<p align="left">
  <img src="docs/images/realworld-credit-risk-benchmark.png" alt="Real-world credit risk benchmark comparing HUGIML, LR, XGBoost, LightGBM, Random Forest, and EBM" width="800">
</p>

<p align="left">
  <img src="docs/images/synthetic-nonmonotonic-benchmark.png" alt="Synthetic non-monotonic benchmark comparing HUGIML, LR, XGBoost, LightGBM, Random Forest, and EBM" width="800">
</p>

### Native missing-value handling

<p align="left">
  <img src="docs/images/native-missing-value-schemes.png" alt="Native missing-value schemes in HUGIML, XGBoost, LightGBM, and EBM" width="760">
</p>

| Model | Native missing-value behavior | What to monitor |
|---|---|---|
| **HUGIML** | Missing numerical values are absent from the transaction. Patterns requiring that feature item do not fire. | Missingness rate and activation frequency of top patterns. |
| **XGBoost** | Each split learns a default route for missing values. | Whether default-route behavior changes under deployment shift. |
| **LightGBM** | Histogram splits learn how missing values are routed. | Missing-value routing and feature missingness drift. |
| **EBM** | Missing values can be modeled as a separate bin/effect. | Size and sign of each missing-bin effect. |

### Adaptive binning

<p align="left">
  <img src="docs/images/adaptive-binning-impact.png" alt="Adaptive binning benchmark against fixed bin counts" width="700" height="300">
</p>

Adaptive binning is a safe default when you do not want to tune `B`; fixed `B=5` is a useful fast baseline. For larger adaptive workflows, the sampling option reduces bin-selection memory while preserving full-data training after edges are selected; in Governance Studio this option is exposed from the Workbench Advanced configuration path.

### Pattern explanations

<p align="left">
  <img src="docs/images/pattern-explanations-real-datasets.png" alt="HUGIML pattern explanations on finance and healthcare datasets" width="760">
</p>

### Model-card-ready artifacts

<p align="left">
  <img src="docs/images/model-card-governance.png" alt="Model-card-ready HUGIML explanations" width="760">
</p>

### Observed benchmark results

![Benchmark comparison](docs/images/benchmark_comparison.png)

| Model | AUC (mean±std) | Fit time/fold | Complexity budget | Remarks |
|---|---:|---:|---|---|
| HUG B=3 | 0.9907 ± 0.0031 | 0.32 s | **topK** patterns | `topK` is an explicit cap; actual mined patterns can be lower. |
| HUG B=5 | 0.9909 ± 0.0028 | 0.34 s | **topK** patterns | More bins per feature. |
| HUG adaptive | 0.9954 ± 0.0022 | 1.20 s | **topK** patterns | Per-feature B increases fit time. |
| EBM | 0.9940 ± 0.0025 | 11.0 s | Additive terms + interactions | Reference interpretable baseline. |
| XGBoost | 0.9882 ± 0.0040 | 0.12 s | Trees × leaves | High-performing ensemble; not directly pattern-interpretable. |
| LightGBM | 0.9921 ± 0.0028 | 0.07 s | Leaves × trees | Fast histogram boosting. |

### Complexity budget

`topK` is the feature-selection budget **K**. It caps each selected feature family before the final estimator is built, unless `topk_budget_strict=True` is used to apply one global cap. The effective downstream width **D** can be lower than these limits when fewer valid features are mined or selected.

| Configuration | Downstream feature budget when `topK = K` |
|---|---|
| `patterns_only`, `L = 1` | Up to **K** HUG pattern features. |
| `patterns_only`, `L > 1`, `interaction_relaxed_mining=True` | Up to **K** HUG pattern features. The mining search may admit up to `interaction_relaxed_feature_size` interaction-information survivor source columns, but no extra downstream feature family is added. |
| `patterns_only`, `L > 1`, augmented pairs enabled | Up to **K** HUG pattern features + up to **K** augmented-pair features, so **D ≤ 2K**. |
| `original_plus_patterns`, `L = 1` | Up to **K** selected original features + up to **K** HUG pattern features, so **D ≤ 2K**. |
| `original_plus_patterns`, `L > 1`, `interaction_relaxed_mining=True` | Up to **K** selected original features + up to **K** HUG pattern features, so **D ≤ 2K**. Survivor-led mining affects which patterns are available, not the number of downstream feature families. |
| `original_plus_patterns`, `L > 1`, augmented pairs enabled | Up to **K** selected original features + up to **K** HUG pattern features + up to **K** augmented-pair features, so **D ≤ 3K**. |
| `original_plus_interactions` | Original features are capped at **K**; retained interaction/pattern features are also bounded by the HUG pattern budget. With augmented pairs enabled, the same additional **K** augmented-pair cap applies. |
| `topk_budget_strict=True` | HUGIML first avoids oversized family blocks, then applies one global TopK selection across the constructed original, pattern, and augmented-pair candidates, so final **D ≤ K**. |

#### Feature-family budgets

`topK` defines the per-family selection budget used by HUGIML when constructing downstream representations. A configuration may include one, two, or three selected feature families:

- HUG pattern features
- selected original input features
- augmented-pair features, when enabled for higher-order configurations

Interaction-relaxed mining changes the native search path but does not add a separate downstream feature family; its budget is `interaction_relaxed_feature_size`, which controls survivor-source admission before pattern mining.

Each active family can contribute up to `topK` downstream columns before strict global selection. Therefore, the maximum downstream width is the number of active selected families multiplied by `topK`:

- one active family: up to `topK` columns
- two active families: up to `2 × topK` columns
- three active families: up to `3 × topK` columns

For example, with `topK=150`, `original_plus_patterns` at `L=1` can retain up to `150` selected original columns and up to `150` HUG pattern columns, for a maximum downstream width of `300`. With `L>1` and `interaction_relaxed_mining=True`, the same downstream width bound remains `300`; the relaxed path affects pattern discovery rather than adding feature columns. With `L>1` and augmented-pair transforms enabled, the same configuration can retain up to `150` selected original columns, `150` HUG pattern columns, and `150` augmented-pair columns, for a maximum downstream width of `450`. When `topk_budget_strict=True`, HUGIML applies one final global TopK selection across the constructed downstream candidates, so the final downstream width is capped at `topK`.

With strict budgeting enabled, HUGIML applies the TopK budget during feature construction rather than after building a full expanded matrix. This keeps the practical downstream width bounded and avoids large intermediate matrices. In hybrid modes, original features are scored and preselected before prediction-time preparation, so prediction prepares only the retained original columns.

### Missing value robustness

![Missing value benchmark](docs/images/missing_value_benchmark.png)

---

## Capabilities Summary

| Capability | Details |
|---|---|
| **HUG pattern mining** | C++ accelerated via pybind11; optional OpenMP parallelism |
| **scikit-learn API** | Full `BaseEstimator` / `ClassifierMixin` compliance |
| **Mixed feature types** | Integer, float, categorical — auto-detected or explicitly supplied |
| **Feature modes** | Pattern-only, original-plus-patterns, original-plus-interactions, augmented-pair downstream features |
| **Fast hyperparameter search** | Cached adaptive-binning grid; mining runs once per unique `(G, L, topK)` group |
| **Governance Studio** | Dash Workbench/Governance dashboard with model comparison, representation inspection, pruning, monitoring, and upload support |
| **Profile visualisations** | EBM-style 1-D/2-D HUG profiles, active-pattern explanations, coefficient-support views (Plotly) |
| **Interpretability metrics** | Pattern count, coverage, overlap, sparsity, top-k cumulative contribution |
| **Adaptive binning** | Per-feature supervised `B` selection with optional stratified sampling — addresses the B-sensitivity trap |
| **Pattern pruning** | Regulated remove/refit/calibrate workflow with full JSON audit trail |
| **Multiclass & imbalance** | Multiclass report, SMOTE/class-weight pipeline, high-cardinality encoding |
| **Benchmark suite** | Reproducible CV comparison and dashboard regeneration via `experiments/benchmark/benchmark_dashboard.py` |
| **Scalability dashboard** | Static runtime, latency, memory, n-scaling, p-scaling, and parameter-sweep evidence reproducible via `experiments/scalability/scalability_dashboard.py` |
| **Calibration** | ECE, MCE, Brier score, reliability diagram data |
| **Drift detection** | PSI + symmetric KL divergence + label drift |
| **Monitoring** | Thread-safe `PredictionMonitor`, latency tracking |
| **Governance** | Model cards (JSON + Markdown), audit artifacts, SBOM |
| **Observability** | OpenTelemetry tracing, Prometheus metrics (both optional) |
| **Secure serialisation** | Allowlist-based `_RestrictedUnpickler`, versioned schema |
| **Deployment** | FastAPI inference server, Docker image, Kubernetes manifests |
| **CI/CD** | GitHub Actions: lint → coverage → native tests → wheels → PyPI |

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
| [`release.yml`](.github/workflows/release.yml) | Git tag `v*.*.*` or manual dispatch | Build platform wheels and the source archive, generate an SBOM, assemble an advisory Docker scan image from the prebuilt CPython 3.12 Linux wheel, publish to PyPI, and create a GitHub release. The scan is retained for security visibility but does not block Python publication; manual runs may disable it. |

---

## Repository Structure

```text
hugiml-core/
├── native/                      C++ extension sources
├── src/
│   └── hugiml/
│       ├── classifier.py        HUGIMLClassifier / HUGIMLClassifierNative
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
│       ├── dashboard/           Governance Studio interfaces and shared runtime
│       │   ├── launcher.py      hugiml-dashboard console entry point
│       │   ├── dash_app.py      Primary Dash application
│       │   ├── dash_components/ Dash pages, controls, tables, charts, and themes
│       │   ├── app.py           Lightweight Streamlit interface
│       │   ├── runner.py        Shared model training and scoring helpers
│       │   └── components/      Streamlit evidence-view renderers
│       ├── llm/                 LLM Assistant package and runtime
│       │   ├── cli.py           hugiml-llm console entry point
│       │   ├── ui_app.py        Streamlit chat interface
│       │   ├── orchestrator.py  Evidence routing and answer assembly
│       │   ├── docs_index.py    Local documentation index
│       │   ├── assets/          Packaged configs and demo datasets
│       │   └── ...
│       └── benchmarks/          CV comparison suite
├── LLM/                         Static assistant assets and GitHub Pages examples
│   ├── config/                  Assistant model policy config
│   ├── datasets/                Built-in and user dataset folders
│   ├── examples/                Static demo pages and source data
│   ├── prompts/                 Assistant prompt templates
│   └── ui/                      Standalone chat UI helpers
├── notebooks/                   Worked examples (12 domain folders)
├── tests/                       Pytest suite
│   ├── dashboard/               Dashboard component tests
│   └── llm/                     Optional LLM Assistant tests
├── benchmarks/                  Micro-benchmarks and regression gate
├── experiments/                 Reproducible dashboard-generation workflows
│   ├── benchmark/               Benchmark analysis runner, checkpointing, CSV summaries, HTML assembly
│   └── scalability/             Scalability runner, checkpointing, flat exports, HTML assembly
├── docker/                      Dockerfile + FastAPI inference server
├── kubernetes/                  Deployment manifests
├── scripts/                     Build and utility scripts
├── docs/                        Sphinx documentation, LLM assistant docs, and model-card templates
├── .github/workflows/           CI/CD pipelines
├── pyproject.toml
└── setup.py
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

## Citation

If you use `hugiml-core`, cite the paper or papers relevant to the functionality
used:

```bibtex
@misc{krishnamoorthy2026complexity,
  title         = {Complexity-Budgeted, Interaction-Aware Interpretable Model for Tabular Data},
  author        = {Krishnamoorthy, Srikumar},
  year          = {2026},
  eprint        = {2607.07060},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2607.07060},
  url           = {https://arxiv.org/abs/2607.07060}
}

@article{krishnamoorthy2026interpretability,
  title   = {Interpretability Myopia: Governance Fitness in Financial Risk Models},
  author  = {Krishnamoorthy, Srikumar},
  journal = {SSRN Electronic Journal},
  year    = {2026},
  doi     = {10.2139/ssrn.6821418},
  url     = {https://ssrn.com/abstract=6821418}
}

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

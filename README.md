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


## Explore HUGIML

See HUGIML in action through interactive benchmark dashboards, Governance Studio, guided video demos, and LLM examples.

**[Interactive Demos](https://srikumar2050.github.io/hugiml-core/)** · **[Open in Colab](https://colab.research.google.com/github/srikumar2050/hugiml-core/blob/gh-pages/quickstart.ipynb)** · **[Documentation](https://hugiml-core.readthedocs.io/)** · **[PyPI](https://pypi.org/project/hugiml-core/)**


---

## Table of Contents

1. [What Is HUG-IML?](#what-is-hug-iml)
2. [Core Philosophy](#core-philosophy)
3. [Why HUG-IML?](#why-hug-iml)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Feature Modes](#feature-modes)
7. [Execution Modes](#execution-modes)
8. [Hyperparameter Search](#hyperparameter-search)
9. [RPTE: Higher-Order Interactions](#rpte--higher-order-interactions)
10. [Causal Analysis with T-HUG](#causal-analysis-with-t-hug)
11. [Governance Studio Dashboard](#governance-studio-dashboard)
12. [LLM Assistant](#llm-assistant)
13. [Augmented Pair Features](#augmented-pair-features)
14. [Adaptive Binning](#adaptive-binning)
15. [Missing Value Handling](#missing-value-handling)
16. [Model Explanation and Visualisations](#model-explanation-and-visualisations)
17. [Native Mining Pruning Controls](#native-mining-pruning-controls)
18. [Pattern Pruning](#pattern-pruning)
19. [Interpretability Metrics](#interpretability-metrics)
20. [Multiclass, Imbalanced Data, High-Cardinality](#multiclass-imbalanced-data-high-cardinality)
21. [Drift Detection & Monitoring](#drift-detection--monitoring)
22. [Calibration](#calibration)
23. [Serialisation](#serialisation)
24. [Governance & Model Cards](#governance--model-cards)
25. [Benchmark Suite](#benchmark-suite)
26. [Example Notebooks: General and Domain-Specific](#example-notebooks-general-and-domain-specific)
27. [Validation Highlights](#validation-highlights)
28. [Inference Server](#inference-server)
29. [CI / CD](#ci--cd)
30. [Repository Structure](#repository-structure)
31. [License](#license)
32. [Citation](#citation)

---

## What Is HUG-IML?

The **High Utility Gain Interpretable Machine Learning (HUG-IML)** framework extracts *High Utility Gain patterns* from labelled tabular data, transforms the input into a binary pattern-presence matrix, and fits an interpretable downstream classifier on that matrix  -  logistic regression by default for a single fit, or, when tuning with the default `performance_ho` grid, either logistic regression or RPTE (Residual Pattern Tree Ensemble, an optional constrained higher-order rule ensemble  -  see [RPTE  -  Higher-Order Interactions](#rpte--higher-order-interactions)), whichever scores better.

The resulting patterns are human-readable and serve as the primary source of model explanations, making the system suitable for regulated domains such as credit scoring, healthcare, and risk management.

**Key references:**

> Krishnamoorthy, S. (2024). Interpretable Classifier Models for Decision
> Support Using High Utility Gain Patterns. *IEEE Access*, 12, 126088-126107.
> DOI: [10.1109/ACCESS.2024.3455563](https://doi.org/10.1109/ACCESS.2024.3455563)

The interaction-aware extensions such as adaptive discretisation, interaction-supported
pattern admission, explicit pair terms, and bounded explanation budgets are
described in [Complexity-Budgeted, Interaction-Aware Interpretable Model for
Tabular Data](https://arxiv.org/abs/2607.07060).

---

## Core Philosophy

A hugimal is a soft toy designed for clinical use, placed in a person's hands before surgery, during a scan, or through the disorientation of a hospital ward. It is not a distraction. It is a therapeutic object with a documented role, earning its place not by being pleasant but by doing something measurable: giving a person in their care something they can hold and recognise while a system too large and complex to understand does its work on them. That same obligation, of placing something graspable in the hands of someone a powerful system acts upon, is what gives HUG-IML its name and its purpose. HUG stands for High Utility Gain, and IML for Interpretable Machine Learning, and together they describe a system built on a single conviction: that in the regulated domains this library targets, the person most affected by a model's output often cannot inspect it, appeal to it, or refuse it on informed grounds. What they can be given, through the expert who must own the decision, is something legible: patterns drawn from the vocabulary of their own domain, each with a demonstrated utility, each readable, challengeable, and if necessary removable on the record.

Rather than explaining a black box after training, HUG-IML builds a model from interpretability up. The learned representation is anchored in High Utility Gain patterns: human-readable intervals and categories, each carrying its own support, utility, and coefficient. There is no hidden layer behind the explanation. The explanation is the model.

---

## Why HUG-IML?

Interpretable ML methods tend to navigate a tradeoff between readability and capacity: additive models such as EBMs grow in complexity with the number of features and interaction terms, while post-hoc methods such as SHAP explain a trained model after the fact rather than producing an inherently inspectable one. HUGIML takes a different approach. The complexity budget is capped at **topK patterns** and this cap holds regardless of the number of features or bin counts in the data. Each pattern is a human-readable conjunction of intervals and categories, and the patterns themselves form the learned representation  -  the downstream logistic regression operates directly on a binary pattern-presence matrix, with no separate explanation layer added after training. On the benchmark datasets evaluated so far, HUGIML achieves AUC broadly comparable to EBM and XGBoost while keeping the representation more compact (see [Benchmark Suite](#benchmark-suite) and [Validation Highlights](#validation-highlights)).

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

# With the focused causal-effect investigation dashboard
pip install "hugiml-core[causal-dashboard]"

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
> only  -  it detects integer, float, and categorical columns and encodes the
> target. Discretisation, HUG pattern mining, and downstream classifier fitting
> occur inside `fit()` on the training data supplied to that call.

### Path A  -  `prepareXy`

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from hugiml import HUGIMLClassifier

clf = HUGIMLClassifier(adaptive_binning=True, L=1, G=5e-3, topK=100)

X_enc, y_enc = clf.prepareXy(X_df, y)   # schema/type prep  -  no model fitting

X_tr, X_te, y_tr, y_te = train_test_split(
    X_enc, y_enc, stratify=y_enc, random_state=42
)

clf.fit(X_tr, y_tr)                     # mining + downstream fit on train only
proba = clf.predict_proba(X_te)

print(clf.get_hug_features())
print(clf.feature_importances())
print(clf.model_summary())
```

### Path B  -  explicit `allCols` for CV and production pipelines

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

For downstream LR fits, `lr_source_policy` controls reuse of raw sources in the final LR representation: `"standard"` preserves the current layout, `"main_effect"` retains surviving original-feature main effects while making generated contextual components source-disjoint, and `"strict"` permits each raw source in at most one final component. In `patterns_only`, `"main_effect"` is equivalent to `"strict"` because no original-feature main-effect block is present. RPTE tree construction is unchanged; the policy is applied only immediately before its final LR fit.

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

When `base_estimator` is not supplied, HUGIML exposes the built-in downstream linear classifier choice through `lr_solver`. The default remains `"auto"`: binary problems use `LogisticRegression(solver="liblinear")`, and multiclass problems use `LogisticRegression(solver="saga")`.

| `lr_solver` | Downstream estimator | When to use |
|---|---|---|
| `"auto"` | `LogisticRegression`: liblinear for binary tasks and saga for multiclass tasks | Recommended default for most datasets and for the established solver-selection behavior. |
| `"saga"` | `LogisticRegression(solver="saga")` | Useful for larger or sparse downstream matrices when you still want logistic-regression coefficients and probability estimates. |
| `"sgd"` | `SGDClassifier(loss="log_loss")` | Useful for very large downstream matrices where stochastic optimization can reduce memory pressure or wall-clock time. Validate accuracy because SGD can be more sensitive to scaling and convergence settings. |

All built-in solver choices keep deterministic defaults aligned with the existing classifier path: `random_state=0` and `max_iter=500`. If you need complete control over solver-specific hyperparameters, pass a fully configured `base_estimator`; that continues to override `lr_solver`.

A `base_estimator` is not limited to `LogisticRegression`/`SGDClassifier`  -  see [RPTE  -  Higher-Order Interactions](#rpte--higher-order-interactions) for the constrained residual-rule branch reachable the same way (and, by default, automatically considered through the `performance_ho` grid).

```python
from hugiml import HUGIMLClassifier

# Historical default
clf_default = HUGIMLClassifier(lr_solver="auto")

# LogisticRegression through saga
clf_saga = HUGIMLClassifier(lr_solver="saga", feature_mode="original_plus_patterns")

# Logistic loss through SGDClassifier
clf_sgd = HUGIMLClassifier(lr_solver="sgd", feature_mode="original_plus_patterns")
```

The versioned `.hugiml` serializer records `lr_solver` in the classifier initialization state and natively round-trips both `LogisticRegression` and the built-in `SGDClassifier` downstream estimator; an RPTE (or otherwise custom) `base_estimator` round-trips too, including the unfitted `base_estimator` hyperparameter itself  -  see [Serialization](#serialization) under the RPTE section above.

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

HUGIML provides a fast cached tuning path for adaptive-binning grids. When `adaptive_binning=True`, the binning and transaction construction work is reused across eligible candidates, so compact grids can be evaluated without rebuilding the same mining inputs repeatedly. This caching follows from how HUGIML's pipeline is factored: adaptive binning and transaction preparation depend only on the data, so their outputs are shared across all candidates; pattern mining depends on `(G, L, topK)`, so its outputs are shared within each mining group; and only the lightweight downstream estimator fit varies per candidate. As a result, the most expensive stages run once regardless of grid size, and the downstream fit results are numerically identical to those from independent per-candidate runs.

The reason this multi-level reuse is achievable in HUGIML is that its pipeline has a genuine separation between upstream data preparation and downstream fitting. Discretization and pattern mining are upstream stages whose outputs do not depend on the downstream estimator choice or its regularization parameters, so they can be computed once and shared. Tree-ensemble methods such as XGBoost and LightGBM support their own forms of caching and incremental training  -  most notably the ability to extend a fitted model by adding more boosting rounds without restarting from scratch. However, other hyperparameters such as tree depth, learning rate, subsampling ratios, and regularization terms affect the splitting criterion at every node throughout training, so changing them generally requires a full retrain rather than only updating a downstream stage. This reflects the nature of boosting as a joint optimization over the ensemble, not a limitation of ensemble methods as predictors. XGBoost and LightGBM remain competitive baselines on raw predictive performance across the evaluated datasets.

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

### `tune()`  -  cross-validated search with automatic fast path

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

A custom grid is supplied via `param_grid`. For the cached adaptive-binning path, keep the varying dimensions compact and centered on mining or representation choices such as `G`, `L`, `topK`, `feature_mode`, and `lr_source_policy`. Fixed values such as `B=-1` and `adaptive_binning=True` may be included for clarity.

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

### `fast_grid_tune()`  -  single-split cached path for custom CV loops

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

## Causal Analysis with T-HUG

The optional **HUGIML Causal Investigation Studio** explores binary-treatment
effects with shared-vocabulary T-HUG outcome models. It provides potential-outcome
and CATE summaries, repeated cross-fitted estimates, DR/AIPW comparisons, overlap
sensitivity, balance diagnostics, interpretable treatment-effect regions, and
control and treatment outcome-model details. The default Quick grid evaluates one configuration
per model for a responsive first run; the four larger HUGIML grids remain available.

### Install and launch

```bash
pip install "hugiml-core[causal-dashboard]"
hugiml-causal-dashboard
```

The launcher opens `http://localhost:8052/` and binds locally to `127.0.0.1:8052`.
Choose a demo or upload a supported dataset with causal metadata, confirm the
treatment, outcome, and baseline covariates, select comparison methods and a
grid, then run the analysis. T-HUG selection can use ROC AUC or negative log loss.
Synthetic demos may include oracle CATE values; uploaded real datasets never
receive fabricated counterfactual truth.

### Interactive preview

- [Open the HUGIML Causal Investigation Studio Demo](https://srikumar2050.github.io/hugiml-core/hugiml_causal_dashboard_demo.html)

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
# Launch the Dash Governance Studio (opens the browser when the server is ready)
hugiml-dashboard

# Common options
hugiml-dashboard --port 8050 --cv 5 --random-state 42
hugiml-dashboard --no-open

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

The **HUGIML LLM Assistant** is a chat-first Dash interface for working with HUGIML models and documentation from one place. It shares the visual language of Governance Studio and keeps the existing Streamlit interface as a lightweight option. It is designed for model-review workflows where a user asks natural-language questions, sees a structured answer, and then continues with follow-up questions in the same review session.

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

# Lightweight Streamlit interface
hugiml-llm --ui light

# Dash host, port, and browser controls
hugiml-llm --host 127.0.0.1 --port 8051 --no-open

# Source-tree development
python -m hugiml.llm.dash_app
```

The Dash UI provides Chat, Dataset, and Model evidence workspaces. The session appears as a chronological transcript, the follow-up input stays inline below the latest answer, and the evidence workspaces remain available throughout the review. Governance reports and pruning requests can be made directly in Chat. Governance Studio can pass dataset, model-session, workspace, and source context through a stable URL contract such as `?dataset=credit_risk&session=model-17&view=chat&source=governance`.

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

The Dash and lightweight interfaces use the same model catalog and available-memory policy. The Dash selector recommends an installed model for the current memory profile, retains the selected model for the browser session, and provides a refresh control, availability details, and Ollama setup commands. These thresholds are available-RAM requirements, not model file sizes:

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

HUGIML treats NaN and Inf values as **not observed**  -  no imputation and no special parameter are required.

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
![Feature shape profiles  -  public tabular benchmark](docs/images/explanation_dashboard_bc.png)

**Credit risk scoring**
![Feature shape profiles  -  credit risk](docs/images/explanation_dashboard_credit.png)

- [Open the HUGIML Benchmark Analysis Dashboard](https://srikumar2050.github.io/hugiml-core/hugiml_benchmark_analysis_dashboard.html)  -  internal 100-dataset suite (50 real-world + 50 synthetic)
- [Open the OpenML-CC18 Benchmark Dashboard](https://srikumar2050.github.io/hugiml-core/openml_cc18_benchmark_dashboard.html)  -  evaluation on official OpenML-CC18 splits

The static benchmark dashboards are reproducible from the repository source; see
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
editor.remove_by_keyword("postcode", reason="high PSI  -  unstable feature")
editor.remove_low_support(min_support=0.01, reason="noise patterns")

editor.refit(X_tr, y_tr)
editor.calibrate(X_cal, y_cal, method="isotonic")

new_clf = editor.finalize()
print(editor.audit_report())
```

The **Representation Pruning** view in the Governance Studio provides an interactive version of this workflow without writing code.

---

## Interpretability Metrics

HUGIML exposes two complementary views of interpretability. Pattern metrics
describe the mined HUG representation on supplied data, while complexity
metrics quantify how much fitted evidence must be reviewed at model and
prediction level.

### Pattern metrics

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
  avg_pattern_length      : 1.34
  max_pattern_length      : 2
  coverage                : 0.9812  (98.1% of 500 samples)
  mean_active_patterns    : 6.21
  std_active_patterns     : 2.08
  overlap_rate (norm.)    : 0.0714
  explanation_sparsity    : 0.0230
  top-k cumulative |coef|:
    top-  1 :    8.4%
    top-  5 :   31.2%
    top- 10 :   54.7%
```

`coverage` is the fraction of supplied rows activating at least one mined
pattern. `overlap_rate` is the mean number of active patterns per row divided
by the number of mined patterns. `explanation_sparsity` is the fraction of
mined patterns that do not activate on the supplied rows. Cumulative
contribution is based on absolute downstream coefficients.

### Structural and inspection complexity

```python
from hugiml.compute_complexity import (
    get_complexity,
    get_complexity_report,
    get_instance_inspection_units,
)

model_units = get_complexity(clf, "model units")
model_inspection_units = get_complexity(clf, "model inspection units")
per_row_units = get_instance_inspection_units(clf, X_test)
report = get_complexity_report(clf, X=X_test)

# Equivalent estimator methods are also available.
same_report = clf.get_complexity_report(X=X_test)
redundancy_audit = clf.get_downstream_redundancy_audit()
```

| Measure | Interpretation |
|---|---|
| **Model units** | Coarse active components in the complete fitted model, such as terms, rules, or terminal leaves. |
| **Model inspection units** | Expanded evidence required to inspect the complete fitted model, including source elements and active root-to-leaf conditions. |
| **Instance inspection units** | Expanded evidence used for one prediction. The API returns one count per row; the report adds the mean, sample standard deviation, standard error, and confidence interval. |

For an RPTE model, counts use the final active representation: active leaf
rules and active direct terms, rather than every generated tree or leaf. The
complexity report also includes `downstream_redundancy_audit`, which records
the training-only removal of constant, duplicate, complementary, and
high-dependence generated columns. Its summary includes the VIF threshold,
the representation R² threshold, retained-column counts, and VIF statistics.
These fitted audit details are retained by model serialization and are
available to the Dash and Streamlit inspection interfaces.

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

HUGIML is evaluated on one internal panel and four external classification suites. The internal panel contains 50 real-world and 50 synthetic datasets. OpenML-CC18, PMLBmini, and TabZilla retain their suite-specific validation structures and matched ensemble baselines. TabArena uses repeated outer folds, retained inner-fold ensembles, and the official reference leaderboard.

**Evaluation environment.** All reported runs were performed on a 64-bit Windows 11 25H2 system with a 13th-generation Intel Core i9-13900H processor (14 cores and 20 logical processors), 16 GB of RAM, Python 3.12.7, and HUGIML 1.1.20. GPU acceleration was not used. Parallel execution followed each benchmark runner's configured worker settings, with the same computational environment applied to HUGIML and its matched baselines within each suite. Reported runtimes therefore support within-suite comparisons; cross-suite timing comparisons should also account for differences in validation protocols, fold counts, and retained ensembles.

| Benchmark | Completed / catalog | Validation protocol | HUGIML / reference configurations |
|---|---:|---|---:|
| Internal panel | 100 / 100 | Controlled outer evaluation with inner model selection | 16 / 16 |
| OpenML-CC18 | 50 / 72 | Official outer splits plus nested 3-fold selection | 16 / 16 |
| PMLBmini | 44 / 44 | Three repeated rotating 3-fold partitions | 16 / 16 |
| TabZilla | 31 / 36 | Official rotating train, validation, and test folds | 16 / 16 |
| TabArena classification | 32 / 38 | Repeated outer 3-fold plus retained inner 8-fold ensemble | 16 / 200 |

### Internal benchmark

The internal benchmark evaluates 100 datasets using controlled outer evaluation with inner model selection. It separates 50 real-world datasets from 50 synthetic datasets designed to exercise interactions, missingness, and varied feature structures.

#### HUGIML predictive results

| Scope | Datasets | Mean ROC AUC | Mean balanced accuracy | Mean F1 | Mean Brier score |
|---|---:|---:|---:|---:|---:|
| Overall | 100 | 0.8809 | 0.8265 | 0.8110 | 0.1077 |
| Real-world | 50 | 0.9020 | 0.8431 | 0.8359 | 0.0989 |
| Synthetic | 50 | 0.8598 | 0.8099 | 0.7861 | 0.1165 |

#### Inspection complexity

Model-inspection units estimate the work required to audit the complete fitted model.

<table>
  <thead><tr><th rowspan="2">Scope</th><th rowspan="2">N</th><th colspan="4">Mean model-inspection units</th><th colspan="2">RPTE-active HUGIML fits</th></tr><tr><th>HUGIML</th><th>XGB</th><th>LGBM</th><th>RF</th><th>Mean active trees</th><th>Mean active leaf path length</th></tr></thead>
  <tbody>
    <tr><td>Overall</td><td>100</td><td><strong>67.4</strong></td><td>3,971.1</td><td>12,477.7</td><td>157,420.5</td><td>2.78</td><td>3.34</td></tr>
    <tr><td>Real-world</td><td>50</td><td><strong>24.6</strong></td><td>1,839.3</td><td>6,919.7</td><td>37,007.3</td><td>1.91</td><td>2.46</td></tr>
    <tr><td>Synthetic</td><td>50</td><td><strong>110.2</strong></td><td>6,102.9</td><td>18,035.7</td><td>277,833.8</td><td>3.19</td><td>3.75</td></tr>
  </tbody>
</table>

[Open the internal benchmark dashboard](https://srikumar2050.github.io/hugiml-core/hugiml_benchmark_analysis_dashboard.html) for dataset-level results, statistical comparisons, timing, RPTE behavior, and methodology.

### External benchmarks: OpenML-CC18, TabZilla, and PMLBmini

OpenML-CC18 reports 50 of 72 catalog tasks using official outer splits and nested 3-fold model selection. TabZilla reports 31 of 36 tasks using stored rotating train, validation, and test folds. PMLBmini reports all 44 datasets using three repeated rotating 3-fold partitions. The tables use datasets with matched HUGIML, XGBoost, LightGBM, and Random Forest results.

#### HUGIML predictive and inspection results

| Benchmark | N | Mean ROC AUC | Mean balanced accuracy | Mean F1 | Mean Brier score | Mean model-inspection units |
|---|---:|---:|---:|---:|---:|---:|
| OpenML-CC18 | 50 | 0.9105 | 0.7910 | 0.7330 | 0.1372 | 119.4 |
| TabZilla | 31 | 0.8934 | 0.7627 | 0.7345 | 0.1858 | 175.8 |
| PMLBmini | 44 | 0.8248 | 0.7438 | 0.6973 | 0.1399 | 15.0 |

#### Complexity and RPTE behavior

<table>
  <thead><tr><th rowspan="2">Benchmark</th><th rowspan="2">N</th><th colspan="4">Mean model-inspection units</th><th colspan="4">Mean instance-inspection units</th><th colspan="3">HUGIML active RPTE trees</th><th rowspan="2">HUGIML mean active leaf path length</th></tr><tr><th>HUGIML</th><th>XGB</th><th>LGBM</th><th>RF</th><th>HUGIML</th><th>XGB</th><th>LGBM</th><th>RF</th><th>Mean</th><th>Median</th><th>Maximum</th></tr></thead>
  <tbody>
    <tr><td>OpenML-CC18</td><td>50</td><td><strong>119.4 (1x)</strong></td><td>12,338.8 (103.3x)</td><td>50,476.2 (422.7x)</td><td>154,884.9 (1,297.0x)</td><td><strong>54.3 (1x)</strong></td><td>1,458.1 (26.8x)</td><td>2,024.1 (37.3x)</td><td>2,078.6 (38.3x)</td><td>4.66</td><td>4.00</td><td>10</td><td>3.97</td></tr>
    <tr><td>TabZilla</td><td>31</td><td><strong>175.8 (1x)</strong></td><td>15,326.9 (87.2x)</td><td>60,824.4 (346.0x)</td><td>103,388.0 (588.1x)</td><td><strong>80.1 (1x)</strong></td><td>1,829.3 (22.8x)</td><td>3,184.0 (39.7x)</td><td>1,057.4 (13.2x)</td><td>5.69</td><td>4.00</td><td>15</td><td>4.19</td></tr>
    <tr><td>PMLBmini</td><td>44</td><td><strong>15.0 (1x)</strong></td><td>481.8 (32.0x)</td><td>630.9 (41.9x)</td><td>3,644.6 (242.2x)</td><td><strong>9.7 (1x)</strong></td><td>110.6 (11.4x)</td><td>126.4 (13.0x)</td><td>380.8 (39.1x)</td><td>1.93</td><td>2.00</td><td>7</td><td>2.26</td></tr>
  </tbody>
</table>

Detailed analysis: [OpenML-CC18](https://srikumar2050.github.io/hugiml-core/openml_cc18_benchmark_dashboard.html), [TabZilla](https://srikumar2050.github.io/hugiml-core/tabzilla_benchmark_dashboard.html), and [PMLBmini](https://srikumar2050.github.io/hugiml-core/pmlbmini_benchmark_dashboard.html).

### External benchmark: TabArena

TabArena classification uses repeated outer 3-fold evaluation with a retained inner 8-fold ensemble. The current comparison covers 32 of the 38 classification datasets and aligns HUGIML outer-fold results with the official reference pool. The leaderboard dashboard begins with a balanced 2 x 2 dataset-scale analysis using the median row and predictor counts of the completed datasets. TabArena's 2,500-row boundary continues to determine whether the evaluation uses ten or three repeated outer partitions.

HUGIML evaluates only 16 configurations, while each other tuned method evaluates 200 configurations. The comparison therefore aligns datasets, outer test folds, and evaluation metrics, but not search budget. It shows HUGIML performance under substantially more limited tuning rather than an equal-compute comparison.

#### Official Elo comparison

<table>
  <thead><tr><th rowspan="2">Scope</th><th rowspan="2">N</th><th colspan="2">Default pool</th><th colspan="2">Tuned pool</th><th colspan="2">All official variants</th></tr><tr><th>Elo</th><th>Rank</th><th>Elo</th><th>Rank</th><th>Elo</th><th>Rank</th></tr></thead>
  <tbody>
    <tr><td>Overall</td><td>32</td><td>1158.6</td><td>11 / 17</td><td>1080.2</td><td>12 / 15</td><td>1056.6</td><td>35 / 46</td></tr>
    <tr><td>Binary</td><td>26</td><td>1152.7</td><td>11 / 17</td><td>1101.9</td><td>11 / 15</td><td>1074.9</td><td>32 / 46</td></tr>
    <tr><td>Multiclass</td><td>6</td><td>1205.6</td><td>10 / 17</td><td>972.5</td><td>13 / 15</td><td>968.4</td><td>35 / 46</td></tr>
  </tbody>
</table>

#### Tuned-pool metric comparison

The delta is HUGIML's dataset-balanced mean minus the strongest official mean. For Brier score, the sign is reversed so a positive value always favors HUGIML.

<table>
  <thead><tr><th rowspan="2">Scope</th><th colspan="3">ROC AUC, mean (median)</th><th colspan="3">Balanced accuracy, mean (median)</th><th colspan="3">F1, mean (median)</th><th colspan="3">Brier score, mean (median)</th></tr><tr><th>HUGIML</th><th>Best official</th><th>Delta</th><th>HUGIML</th><th>Best official</th><th>Delta</th><th>HUGIML</th><th>Best official</th><th>Delta</th><th>HUGIML</th><th>Best official</th><th>Delta</th></tr></thead>
  <tbody>
    <tr><td>Overall</td><td>0.8425 (0.8365)</td><td>0.8747 (0.8913)</td><td>-0.0321 (-0.0547)</td><td>0.6821 (0.6981)</td><td>0.7244 (0.7268)</td><td>-0.0422 (-0.0287)</td><td>0.5509 (0.6004)</td><td>0.6145 (0.6869)</td><td>-0.0635 (-0.0866)</td><td>0.1169 (0.0976)</td><td>0.1087 (0.0929)</td><td>-0.0082 (-0.0047)</td></tr>
    <tr><td>Binary</td><td>0.8220 (0.7947)</td><td>0.8520 (0.8309)</td><td>-0.0300 (-0.0362)</td><td>0.6696 (0.6807)</td><td>0.7119 (0.7255)</td><td>-0.0423 (-0.0448)</td><td>0.5067 (0.5607)</td><td>0.5691 (0.6505)</td><td>-0.0623 (-0.0897)</td><td>0.1019 (0.0904)</td><td>0.0971 (0.0896)</td><td>-0.0048 (-0.0008)</td></tr>
    <tr><td>Multiclass</td><td>0.9315 (0.9348)</td><td>0.9501 (0.9622)</td><td>-0.0187 (-0.0274)</td><td>0.7364 (0.7965)</td><td>0.7668 (0.8670)</td><td>-0.0304 (-0.0704)</td><td>0.7425 (0.7979)</td><td>0.7686 (0.8592)</td><td>-0.0261 (-0.0613)</td><td>0.1820 (0.1668)</td><td>0.1529 (0.1523)</td><td>-0.0290 (-0.0145)</td></tr>
  </tbody>
</table>

Each table entry reports mean (median), and positive delta favors HUGIML; the Brier delta follows its lower-is-better direction. For ROC AUC, HUGIML trails the strongest displayed tuned official mean by 0.0300 across binary datasets and by 0.0187 across multiclass datasets. These differences are achieved with 16 HUGIML configurations versus 200 for each tuned official method. HUGIML combines this predictive performance with source-disjoint, interpretable micro-ensembles and commonly uses one to two orders of magnitude fewer model-inspection units. The median active RPTE size is 3 trees in the internal benchmark, 2 trees on PMLBmini, and 4 trees on both OpenML-CC18 and TabZilla.

[Open the TabArena official leaderboard dashboard](https://srikumar2050.github.io/hugiml-core/tabarena_official_leaderboard_dashboard.html) for model filters, rankings, metric distributions, and methodology.

### Running benchmark evaluations

The installed package provides a compact cross-validation runner:

```bash
python -m hugiml.benchmarks.runner --datasets german_credit pima adult --output benchmarks/results/
```

Repository runners support dataset download, checkpoint resume, and dashboard assembly. The central [reproduction guide](experiments/benchmark/REPRODUCING.md) describes creation of a constrained environment, internal and external dataset preparation, cache verification, complete and staged execution, checkpoint continuation, result assembly, and dataset-integrity checks. Suite-specific README files define the corresponding panels, validation protocols, grids, metrics, and output schemas.

### Scalability dashboard

The scalability evaluation measures fit time, prediction latency, memory, pattern counts, and test AUC across row-count, feature-count, and mining-budget sweeps. It compares HUGIML with XGBoost and LightGBM while retaining the evidence needed for model review.

[Open the HUGIML scalability dashboard](https://srikumar2050.github.io/hugiml-core/hugiml_scalability_dashboard.html) for the complete scaling analysis and methodology.

## Example Notebooks: General and Domain-Specific

The notebooks provide self-contained examples that move from model fitting to interpretation, governance review, and domain-specific analysis. They are organized in [`notebooks/`](notebooks/) as 12 focused folders.

### General Modeling and Evaluation

These notebooks introduce the core workflow, baseline evaluation, model comparison, and special modeling cases.

| Folder | Notebook | Brief description |
|---|---|---|
| [`00_quickstart`](notebooks/00_quickstart/) | [`nb00_pattern_explanation_walkthrough.ipynb`](notebooks/00_quickstart/nb00_pattern_explanation_walkthrough.ipynb) | Quick end-to-end walkthrough of fitting HUGIML, extracting patterns, and reading pattern-level explanations. |
| [`01_benchmark_baselines`](notebooks/01_benchmark_baselines/) | [`nb01_benchmark_baselines.ipynb`](notebooks/01_benchmark_baselines/nb01_benchmark_baselines.ipynb) | Benchmark comparison across HUGIML and common tabular baselines such as XGBoost, LightGBM, Random Forest, and logistic regression. |
| [`02_hug_vs_ebm`](notebooks/02_hug_vs_ebm/) | [`nb02_hug_vs_ebm.ipynb`](notebooks/02_hug_vs_ebm/nb02_hug_vs_ebm.ipynb) | Side-by-side comparison of HUGIML pattern profiles and EBM-style additive shape functions. |
| [`03_modeling_special_cases`](notebooks/03_modeling_special_cases/) | [`nb03_modeling_special_cases.ipynb`](notebooks/03_modeling_special_cases/nb03_modeling_special_cases.ipynb) | Practical modeling cases including multiclass targets, imbalance, high-cardinality categoricals, adaptive binning, and pruning workflows. |

### Domain-Specific Examples

These notebooks apply the modeling and governance workflow to finance, fraud, healthcare, insurance, public-program integrity, and workforce analytics.

| Folder | Notebook | Brief description |
|---|---|---|
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
| **Mixed feature types** | Integer, float, categorical  -  auto-detected or explicitly supplied |
| **Feature modes** | Pattern-only, original-plus-patterns, original-plus-interactions, augmented-pair downstream features |
| **Fast hyperparameter search** | Multi-level compositional caching: binning and transactions shared across all candidates, mining shared per `(G, L, topK)` group; downstream fit results numerically identical to independent per-candidate runs, achievable because data preparation and pattern mining are upstream stages independent of the downstream estimator |
| **Governance Studio** | Dash Workbench/Governance dashboard with model comparison, representation inspection, pruning, monitoring, and upload support |
| **Profile visualisations** | EBM-style 1-D/2-D HUG profiles, active-pattern explanations, coefficient-support views (Plotly) |
| **Interpretability metrics** | Pattern count, coverage, overlap, sparsity, top-k cumulative contribution |
| **Adaptive binning** | Per-feature supervised `B` selection with optional stratified sampling  -  addresses the B-sensitivity trap |
| **Pattern pruning** | Regulated remove/refit/calibrate workflow with full JSON audit trail |
| **Multiclass & imbalance** | Multiclass report, SMOTE/class-weight pipeline, high-cardinality encoding |
| **Benchmark suite** | Reproducible internal and external evaluations covering the 100-dataset internal panel, PMLBmini, TabZilla, OpenML-CC18, and a reserved TabArena track; runners share a common benchmark engine while retaining suite-specific validation rules. |
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

Apache License 2.0  -  see [LICENSE](LICENSE).

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

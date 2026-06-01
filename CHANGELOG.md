# Changelog

All notable changes to hugiml-core are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---
## [1.1.5] - 2026-06-01

### Changed

- Reduced memory usage for materialized native transactions by storing item ids with shared item-level utility lookup instead of repeating utility values in every transaction entry.
- Integrated adaptive binning directly into the native L1 hot path, avoiding intermediate binned-matrix materialization for `L=1` adaptive workflows.
- Parallelized native adaptive bin selection and bin-code application, with `n_jobs` applied before adaptive preprocessing.
- Improved large-data stability with cleaner native memory/timeout error handling and safer fallback behavior under memory pressure.

---
## [1.1.4] - 2026-05-31

- Added the native L1 hot path for `L=1` fits. The C++ path fuses transaction preparation, single-item pattern mining, information-gain filtering, top-K retention, and sparse matrix construction to reduce Python/C++ overhead for the common L1 workflow.
- Moved adaptive binning selection into the C++ backend. Per-feature bin counts are now selected using supervised information-gain scoring and elbow-style stopping, while keeping Python metadata (`per_feature_b_`, `_bin_edges_`, and `ig_scores_`) available for inspection and serialization.

## [1.1.3] — 2026-05-29

### Changed

- Optimized code and pushed K value inside mining, in line with the original java implementation (hugiml)
- Included performance and memory optimizations - chunked transaction processing, exact structured mining constraints, disabled eucs pruning by default 
- Revised domain specific notebook examples 

## [1.1.2] — 2026-05-27

### Added

- Added `feature_mode` to `HUGIMLClassifierNative` with three downstream feature modes:
  - `patterns_only` — default, backward-compatible mode using only the HUGIML binary pattern matrix.
  - `original_plus_patterns` — hybrid mode using standardized original features plus all mined binary patterns.
  - `original_plus_interactions` — hybrid mode using standardized original features plus higher-order (`L > 1`) mined patterns.
- Added feature-mode regression tests covering fit, predict, predict_proba, transform, fit_transform, feature_importances, model_summary, sklearn clone compatibility, invalid-mode validation, and save/load behavior.

### Changed

- `HUGIMLClassifierNative.fit`, `predict`, `predict_proba`, and `score` now route the downstream estimator through the selected `feature_mode` while preserving the existing pattern-mining pipeline.
- `transform(X)`, `fit_transform(X, y)`, `get_hug_features()`, and `get_pattern_info()` remain pattern-space APIs and continue to return/report only HUGIML binary pattern features.
- `feature_importances()` now reports downstream feature names correctly for both pattern-only and hybrid modes.
- `model_summary()` now reports the active feature mode and downstream feature shape.
- `save_model()` / `load_model()` now persist and restore feature-mode metadata, original-feature preprocessing state, pattern-order masks, and hybrid downstream feature names.
- README documentation updated with the three feature modes and revised notebook-folder descriptions.

### Compatibility

- Default behavior remains unchanged because `feature_mode="patterns_only"` is the default. Existing code that relies on `transform()` returning the HUGIML binary pattern matrix should continue to work unchanged.

---

## [1.1.1] — 2026-05-26

### Added

- New notebooks and examples added

### Changed

- `native/mining.cpp` / `native/mining.hpp`: compute compound-pattern IG from sealed `tid_arr` rather than transient `els`, so L2+ patterns survive positive `G` after `release_els()`.

## [1.1.0] — 2026-05-23

### Added

#### Adaptive binning (`hugiml.adaptive`, `HUGIMLClassifierNative`)
- **`HUGIMLAdaptive`** — standalone classifier that selects the optimal bin count
  per numerical feature via supervised entropy search with elbow-stopping.
  Eliminates the B-sensitivity trap: instead of a single global B, each feature
  gets its own `B_j` from a candidate set, stopping when marginal IG gain falls
  below `min_marginal_gain_ratio`.
- **`HUGIMLClassifierNative(adaptive_binning=True)`** — enables per-feature adaptive
  binning directly on the native classifier via `b_candidates` and
  `min_marginal_gain_ratio` parameters.
- `per_feature_b_`, `ig_scores_` fitted attributes expose the chosen B and IG
  grid per feature.
- `plot_bin_profiles()` and `ig_heatmap()` visualisations.

#### Missing value handling (`HUGIMLClassifierNative`, `HUGIMLAdaptive`)
- **NaN and Inf treated as "not observed"** — no imputation, no new parameter,
  always-on.  All numerical columns are pre-binned to string quantile labels at
  fit time; non-finite cells become `np.nan` → C++ transaction builder skips →
  no item for that (row, feature) pair.
- `_missing_col_edges_` stores quantile edges for all numerical columns;
  persisted in `clf_fit.json`.
- `__setstate__` defaults `_missing_col_edges_ = {}` for backward compat with
  pre-v1.1.0 saved models.

#### Extension modules 
- **`hugiml.plots`** — EBM-style 1-D/2-D HUG profiles, active-pattern
  explanations, coefficient-support views, interactive Plotly dashboard.
- **`hugiml.pruning`** — regulated remove/refit/calibrate workflow with full
  JSON audit trail (`PatternEditor`).
- **`hugiml.metrics`** — interpretability metrics: pattern count, coverage,
  overlap rate, explanation sparsity, top-k cumulative contribution.
- **`hugiml.multiclass`** — per-class pattern importances, SMOTE/class-weight
  pipeline, target-mean / frequency encoding for high-cardinality categoricals.
- **`hugiml.benchmarks`** — reproducible CV comparison suite (console script
  `hugiml-bench`).

#### Documentation / assets
- `docs/images/explanation_dashboard.png` — static model explanation dashboard
  (pattern coefficients, support, ROC curve, summary metrics).
- `docs/images/benchmark_comparison.png` — comparative benchmark chart with
  complexity budget (HUG B=3/5/adaptive, EBM, XGBoost, LightGBM).
- `docs/images/missing_value_benchmark.png` — missing value robustness
  analysis across MCAR / MAR / MNAR mechanisms and 0–40% missing rates.

### Changed
- `_adap_apply_edges` (both `classifier.py` and `adaptive.py`): non-finite
  inputs now return `np.nan` instead of the string `"nan"`.
- `pyproject.toml`: version `1.0.0` → `1.1.0`; new extras `[plots]`,
  `[benchmarks]`, `[imbalanced]`; `hugiml-bench` console script added.
- `Dockerfile`: multi-stage build with `inference`, `research`, and `dev`
  targets.
- CI workflow: installs `.[dev,plots]` to run plot tests.

---

## [1.0.0] — initial release

- C++ accelerated HUG pattern mining via pybind11.
- `HUGIMLClassifierNative`: sklearn-compatible estimator with `prepareXy`
  helper, mixed feature type support, `topK` pattern cap.
- Secure serialisation with allowlist-based unpickler and versioned schema.
- Calibration, drift detection, monitoring, governance, telemetry modules.
- FastAPI inference server, Kubernetes manifests.
- Full CI/CD pipeline (GitHub Actions).

# Changelog

All notable changes to hugiml-core are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---
## [1.1.11] - 2026-06-17

### Added

- Added Governance Studio dashboard evidence views for adaptive binning, augmented-pair traceability, feature-family review, pattern coverage, cross-validation monitoring, and export-ready governance summaries.
- Added binary feature conversion: two-value numeric columns are treated as categorical indicators before HUG pattern mining and downstream feature preparation.

---
## [1.1.10] - 2026-06-12

### Changed

- Strengthened build and import reliability for the native backend with clearer install diagnostics, a public `check_native()` verification helper, cleaned package exports, and backward-compatible classifier names.
- Expanded Governance Studio into a richer Workbench/Governance dashboard for dataset setup, candidate model comparison, result drill-downs, demo datasets, workflow guidance, and complete governance evidence views.

### Compatibility

- Existing v1.1.9 APIs and classifier import paths remain available.
- The installed `hugiml-dashboard` command launches the packaged Streamlit dashboard entry point.

---
## [1.1.9] - 2026-06-10

### Added

- Added `HUGIMLClassifier` as the primary public class name. `HUGIMLClassifierNative` remains available for existing code.
- Added `execution_mode="audit"|"production"`. Audit mode keeps full traceability data by default; production mode keeps prediction and save/load state while omitting large training-review artifacts.
- Added clearer production-mode guidance for audit, governance, drift, and pattern-support methods when full traceability data is not retained.

### Changed

- Reduced strict and hybrid memory use by selecting original features earlier and applying strict `topK` during downstream matrix construction.
- Moved original-feature scoring into native and fused construction paths so `patterns_only` remains pattern-focused and avoids retaining original-feature scoring state.
- Improved fixed-B numeric handling so finite numeric columns stay numeric, while only training columns with missing or infinite values use missing-aware bin labels.
- Reduced native memory pressure with compact integer code storage, CSR-returning matrix paths, and a conditional dense/CSR downstream policy.
- Optimized hybrid prediction so only selected original columns are prepared at prediction time.

### Compatibility

- Default behavior remains `execution_mode="audit"`, preserving the v1.1.8 audit and explanation workflow unless production mode is explicitly selected.
- Existing imports of `HUGIMLClassifierNative` continue to work. New examples use `HUGIMLClassifier`.

---
## [1.1.8] - 2026-06-08

### Added

- Added fast tuning for eligible adaptive-binning hyperparameter searches, reducing repeated mining work during cross-validation while preserving ordinary tuning behavior for unsupported grids.
- Added HUGIML Governance Studio, a Streamlit dashboard for model validation, representation audit, pattern inventory, case review, configuration comparison, pruning analysis, monitoring, and report-oriented review workflows.
- Added dashboard helpers for demo data, uploaded tabular data, governance summaries, prediction review, drift/missingness checks, and benchmark-oriented validation views.

### Changed

- Improved `feature_mode="original_plus_interactions"` so higher-order pattern selection is based on structural pattern order rather than parsing display labels. This keeps numeric interval commas from being treated as multi-feature interactions.
- Expanded benchmark and dashboard runner utilities so HUGIML configurations can be compared more consistently across validation folds.
- Reorganized README content to make installation, quick start, feature modes, tuning, dashboard usage, and governance workflows easier to scan.

---
## [1.1.7] - 2026-06-05

### Added

- Added a native `L=2` mining hot path for common two-item pattern workloads.
- Expanded augmented-pair features with two additional operations: pair sums and signed differences.

### Changed

- Revised the benchmark baseline notebook and exported HTML with the updated baseline run and current HUGIML grid settings.
- Kept the v1.1.6 augmented-feature APIs, feature-lineage metadata, and serialization paths compatible with the expanded pair operations.

---
## [1.1.6] - 2026-06-04

### Added

- Added native augmented-pair features for `L > 1` adaptive-binning models, including product and absolute-difference transforms selected from informative numeric features.
- Added strict global `topK` budgeting across original features, HUG patterns, and augmented-pair features.
- Added public augmented-pair interpretation APIs with raw-scale effects, feature coverage, missing-pair policy, and downstream feature composition metadata.

### Changed

- Improved model summaries, explainability reports, feature lineage, serialization, and pattern-pruning workflows so hybrid downstream models clearly distinguish original features, HUG patterns, and augmented-pair features.

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

---
## [1.1.3] — 2026-05-29

### Changed

- Optimized code and pushed K value inside mining, in line with the original java implementation (hugiml).
- Included performance and memory optimizations: chunked transaction processing, exact structured mining constraints, and disabled EUCS pruning by default.
- Revised domain-specific notebook examples.

---
## [1.1.2] — 2026-05-27

### Added

- Added `feature_mode` to `HUGIMLClassifierNative` with three downstream feature modes:
  - `patterns_only` — default, backward-compatible mode using only the HUGIML binary pattern matrix.
  - `original_plus_patterns` — hybrid mode using standardized original features plus all mined binary patterns.
  - `original_plus_interactions` — hybrid mode using standardized original features plus higher-order (`L > 1`) mined patterns.
- Added feature-mode regression tests covering fit, predict, predict_proba, transform, fit_transform, feature_importances, model_summary, sklearn clone compatibility, invalid-mode validation, and save/load behavior.

### Changed

- `HUGIMLClassifierNative.fit`, `predict`, `predict_proba`, and `score` route the downstream estimator through the selected `feature_mode` while preserving the existing pattern-mining pipeline.
- `transform(X)`, `fit_transform(X, y)`, `get_hug_features()`, and `get_pattern_info()` remain pattern-space APIs and continue to return/report only HUGIML binary pattern features.
- `feature_importances()` reports downstream feature names correctly for both pattern-only and hybrid modes.
- `model_summary()` reports the active feature mode and downstream feature shape.
- `save_model()` / `load_model()` persist and restore feature-mode metadata, original-feature preprocessing state, pattern-order masks, and hybrid downstream feature names.

### Compatibility

- Default behavior remains unchanged because `feature_mode="patterns_only"` is the default. Existing code that relies on `transform()` returning the HUGIML binary pattern matrix should continue to work unchanged.

---
## [1.1.1] — 2026-05-26

### Added

- New notebooks and examples added.

### Changed

- `native/mining.cpp` / `native/mining.hpp`: compute compound-pattern IG from sealed `tid_arr` rather than transient `els`, so L2+ patterns survive positive `G` after `release_els()`.

---
## [1.1.0] — 2026-05-23

### Added

#### Adaptive binning (`hugiml.adaptive`, `HUGIMLClassifierNative`)

- Added `HUGIMLAdaptive`, a standalone classifier that selects the bin count per numerical feature via supervised entropy search with elbow-style stopping.
- Added `HUGIMLClassifierNative(adaptive_binning=True)` with per-feature adaptive binning through `b_candidates` and `min_marginal_gain_ratio`.
- Added fitted inspection attributes including `per_feature_b_` and `ig_scores_`.
- Added `plot_bin_profiles()` and `ig_heatmap()` visualizations.

#### Missing value handling (`HUGIMLClassifierNative`, `HUGIMLAdaptive`)

- NaN and Inf numerical cells are treated as not observed during transaction building.
- Added `_missing_col_edges_` metadata for numerical columns that need missing-aware pre-binning.

#### Extension modules

- Added `hugiml.plots`, `hugiml.pruning`, `hugiml.metrics`, `hugiml.multiclass`, and `hugiml.benchmarks` extension areas.

### Changed

- `_adap_apply_edges` maps non-finite inputs to `np.nan` instead of the string `"nan"`.
- Added optional package extras for plots, benchmarks, imbalanced-data helpers, telemetry, server, explainability, and MLflow integrations.

---
## [1.0.0] — initial release

- C++ accelerated HUG pattern mining via pybind11.
- `HUGIMLClassifierNative`: sklearn-compatible estimator with `prepareXy` helper, mixed feature type support, and `topK` pattern cap.
- Secure serialization with allowlist-based unpickler and versioned schema.
- Calibration, drift detection, monitoring, governance, telemetry modules.
- FastAPI inference server, Kubernetes manifests.
- Full CI/CD pipeline.

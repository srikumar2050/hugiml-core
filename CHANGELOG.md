# Changelog

All notable changes to hugiml-core are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---
## [1.1.20] - 2026-08-09

### Changed

- Made Dash the primary HUGIML LLM Assistant interface, retained Streamlit as the lightweight interface, and aligned the Chat, Dataset, and Model evidence workspaces with Governance Studio. Dataset catalogs refresh on every catalog-selection change, governance requests remain available through Chat, and the interface accepts a stable dataset, model-session, workspace, and source launch-context contract. Both interfaces apply the same Ollama catalog, memory eligibility, recommended-model selection, session retention, and local setup guidance.
- Strengthened benchmark analysis across the internal benchmark and four external suites: OpenML-CC18, PMLBmini, TabZilla, and TabArena. The benchmark workflows retain suite-specific official validation protocols and splits while reporting performance, runtime, RPTE behavior, and model complexity consistently.
- Improved downstream logistic-regression canonicalization by removing constant, duplicate, and complementary columns, then filtering highly collinear generated terms only when they are strongly represented by semantically preferred retained terms. Original features remain protected, pattern terms take precedence over augmented-pair terms, and the fitted transformation is retained for prediction, interpretation, and serialization.
- Accelerated complete-data interaction-information selection by reusing per-feature marginal information gain and global target entropy while retaining joint-observation scoring for feature pairs containing missing values.

---
## [1.1.19] - 2026-07-23

### Added

- Added a reproducible OpenML-CC18 benchmark workflow and static dashboard covering 36 classification datasets with official task-defined splits.
- Added rich, on-demand data profiling to the Dash workbench, including role-aware previews, quality summaries, missingness, distributions, correlations, and target associations.

### Changed

- Improved the RPTE representation path by retaining higher-order patterns as direct terms and consolidating equivalent leaf and pattern terms into one canonical component with auditable aliases, preventing duplicate interpretation and complexity counts.
- Refactored ``classifier.py`` into focused internal modules while preserving the public estimator API, sklearn behavior, serialization, and package imports.
- Added consistent Default, Guided, and Advanced model configuration modes and a dedicated Results complexity view in the Dash interface.
- Aligned EBM inspection complexity with the other interpretable models by expanding active interaction score cells by source-feature arity at both model and instance levels.

---
## [1.1.18] - 2026-07-19

### Added

- Added the optional RPTE downstream model for learning higher-order relationships with adaptive sequential or bounded-lookahead tree growth.
- Added interaction relaxation through the first child split, with held-out acceptance checks for jointly useful features that may have weak individual signal.
- Added RPTE rule-tree and rule-table views, direct-term source metadata, model and instance inspection units, and structured save/load support.
- Added a Dash-based Governance Studio interface while retaining the lightweight Streamlit interface.
- Expanded benchmark analysis to 100 diverse datasets: 50 real-world and 50 synthetic datasets, with instance-level inspection and confidence summaries.

### Changed

- Extended governance, dashboard, LLM evidence, complexity, and serialization workflows to recognize RPTE models and their interpretable direct and tree-based terms.
- Extended the native build and release package with RPTE scoring, tree-growth, binding, and third-party notice files.

---
## [1.1.17] - 2026-07-07

### Added

- Added a native `L=3` mining hot path for bounded three-item pattern workloads.
- Added `max_mining_seconds` and `get_mining_audit_log()` so long mining runs can keep an explicit timeout budget and a compact audit trail.
- Added focused fuzz, timeout, L3 hot-path, and zero-variance tests to cover the new mining paths and edge cases.
- Added `lr_solver` support for built-in downstream `saga` LogisticRegression and `sgd` SGDClassifier paths while preserving the existing default solver behavior.
- Added scalability dashboard solver scenarios for the new saga and SGD downstream paths.
- Added a privacy-sanitized reproducibility/SBOM manifest option for the scalability and benchmark dashboard.

### Changed

- Optimized L3 mining with tighter same-feature checks, pair reuse, coverage de-duplication, and gated pair-cache pruning.
- Improved native timeout behavior so miners return partial results where possible instead of discarding completed work.
- Reduced Python-side overhead in numeric zero-variance detection and refreshed Sphinx documentation for the full 1.1.17 release surface.
- Updated README and Sphinx guidance for choosing downstream solvers, including when to use the default, saga, or SGD path.
- Updated versioned model serialization to record and restore the new built-in solver option, including native SGDClassifier estimator state.

---
## [1.1.16] - 2026-06-30

### Added

- Added the optional HUGIML LLM assistant with Streamlit and CLI entry points, built-in demo datasets, deterministic fast mode, and optional Ollama-backed thinking mode.
- Added documentation for native HUIM pruning behavior, including LIU, LA, gated EUCS activation, and EUCS environment controls.

### Changed

- Improved augmented-pair adaptive path reuse so repeated tuning and scoring flows can reuse prepared path state more efficiently.
- Expanded native compound-pattern mining pruning with LIU, LA, and gated EUCS paths while preserving public model behavior.
- Refreshed Sphinx documentation, Read the Docs navigation, and static benchmark/scalability outputs for the new release.

---
## [1.1.15] - 2026-06-27

### Added

- Added a no-pattern fallback for pattern-only fits, returning a deterministic constant-prior classifier instead of failing on null, tiny, or very strict mining runs.
- Added serialization support for fallback state so saved models keep the same predictions after load.

### Changed

- Strengthened interaction-relaxed mining by allowing survivor-pair joint evidence to help admit weak-marginal interaction features into native mining.
- Refreshed benchmark grids with compact EBM settings and a registered RuleFit grid for broader interpretable-baseline comparisons.

---
## [1.1.14] - 2026-06-25

### Changed

- Reduced memory pressure for adaptive-binning workflows by allowing native bin selection to use a deterministic stratified sample before applying the selected bin edges to the full training data.
- Preserved float32 numeric inputs through native L1 preparation where possible, avoiding unnecessary widening copies while keeping C-contiguous arrays for native execution.
- Exposed the sampling option in Workbench Advanced configuration and refreshed the Governance Studio static preview to match the new control.

---
## [1.1.13] - 2026-06-24

### Added

- Added `experiments/` reproducibility workflows for regenerating benchmark and scalability dashboard outputs.


### Changed

- Improved fit and predict runtime through faster categorical handling, reusable categorical-code paths, and consistent zero-variance column exclusion.
- Centralized HUGIML and benchmark hyperparameter grids in `hugiml.hyperparameter_configs`, with shared use by classifier tuning, benchmarks, and the Workbench UI.
- Updated tuning defaults so search candidates run with leaner retained state while the returned refit model remains audit-ready unless production mode is explicitly requested.

---
## [1.1.12] - 2026-06-20

### Added

- Added interaction-information source selection for augmented-pair features, with `aug_feature_size`, `ii_partner_size`, and `augmented_pair_mode` controls.
- Added `interaction_relaxed_mining`, which lets interaction-information survivor features participate in native pattern mining without adding pair-operator columns.
- Added pair-aware adaptive binning metadata for the interaction-relaxed path and survivor-led audit fields for pattern and downstream-feature review.

### Changed

- Updated augmented-pair defaults to use `augmented_pair_mode="interaction_information"`; `augmented_pair_mode="marginal_ig"` remains available for the v1.1.11 source-selection behavior.
- Updated dashboard and governance evidence views to separate survivor-led pattern evidence from augmented-pair feature evidence.
- Kept `augmented_pair_max_features` as a v1.1.11-compatible alias for the new augmented-pair source budgets.

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

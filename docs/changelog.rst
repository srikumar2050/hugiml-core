Changelog
=========

The canonical release notes live in ``CHANGELOG.md`` at the repository root. Keep that file synchronized with the package version, PyPI release, GitHub tag, and Read the Docs build.

Recent releases
---------------

1.1.19 — 2026-07-23
~~~~~~~~~~~~~~~~~~~

* Added a reproducible OpenML-CC18 benchmark workflow and static dashboard for 36 classification datasets using official task-defined splits.
* Added rich, on-demand Dash data profiling with role-aware previews, quality findings, missingness, distributions, correlations, and target associations.
* Improved RPTE representation handling by retaining higher-order patterns as direct terms and consolidating equivalent leaf and pattern terms into one canonical component with auditable aliases.
* Refactored the classifier implementation into focused internal modules while preserving the public estimator API, sklearn behavior, serialization, and package imports.
* Added consistent Default, Guided, and Advanced model configuration modes and a dedicated Results complexity view in the Dash interface.
* Aligned EBM inspection complexity with the other interpretable models by expanding active interaction score cells by source-feature arity at both model and instance levels.

1.1.18 — 2026-07-19
~~~~~~~~~~~~~~~~~~~

* Added the optional RPTE downstream model for higher-order relationships with adaptive sequential or bounded-lookahead tree growth.
* Added interaction relaxation through the first child split, with held-out acceptance checks for jointly useful features.
* Added interpretable RPTE tree, rule-table, direct-term provenance, complexity, instance inspection, and structured serialization views.
* Added a Dash-based Governance Studio while retaining the lightweight Streamlit interface.
* Expanded benchmark analysis to 100 diverse datasets: 50 real-world and 50 synthetic datasets.

1.1.17 — 2026-07-07
~~~~~~~~~~~~~~~~~~~

* Added a native ``L=3`` mining hot path for bounded three-item pattern workloads.
* Added ``max_mining_seconds`` and ``get_mining_audit_log()`` for explicit mining-stage time budgets and compact attempt-level audit trails.
* Optimized L3 mining with tighter same-feature checks, pair reuse, coverage de-duplication, and gated pair-cache pruning.
* Improved native timeout behavior so miners can return partial results where possible.
* Added focused fuzz, timeout, L3 hot-path, and zero-variance tests for the new mining paths and edge cases.
* Added ``lr_solver`` support for built-in downstream ``saga`` LogisticRegression and ``sgd`` SGDClassifier paths while preserving the existing default solver behavior.
* Added scalability dashboard solver scenarios for the new saga and SGD downstream paths.
* Added a privacy-sanitized reproducibility/SBOM manifest option for the scalability and benchmark dashboard.
* Updated README and Sphinx guidance for choosing downstream solvers, and updated serialization for the new built-in SGD solver state.

1.1.16 — 2026-06-30
~~~~~~~~~~~~~~~~~~~~~

* Added the optional HUGIML LLM assistant with Streamlit and CLI entry points, built-in demo datasets, deterministic fast mode, and optional Ollama-backed thinking mode.
* Improved augmented-pair adaptive path reuse so repeated tuning and scoring flows can reuse prepared path state more efficiently.
* Expanded native HUIM pruning behavior around LIU, LA, and gated EUCS, with documented EUCS environment controls.
* Refreshed Sphinx documentation, Read the Docs navigation, and static benchmark/scalability outputs for the new release.

1.1.15 — 2026-06-27
~~~~~~~~~~~~~~~~~~~

* Added a deterministic constant-prior fallback for ``patterns_only`` fits when no HUG patterns are mined.
* Persisted fallback state in the versioned model format so fallback predictions round-trip through save/load.
* Strengthened ``interaction_relaxed_mining`` by using joint survivor-pair evidence during native external-utility admission.
* Refreshed interpretable benchmark grids, including compact EBM settings and a registered RuleFit grid.

1.1.14 — 2026-06-25
~~~~~~~~~~~~~~~~~~~

* Reduced memory pressure for adaptive-binning workflows by adding deterministic stratified sampling for native bin-count selection.
* Preserved float32 numeric inputs through the native L1 preparation path where possible, avoiding unnecessary widening copies while keeping C-contiguous arrays for native execution.
* Exposed the sampling option in Workbench Advanced configuration and refreshed the Governance Studio static preview to match the new control.

1.1.13 — 2026-06-24
~~~~~~~~~~~~~~~~~~~

* Improved fit and predict runtime through faster categorical processing, optional categorical-code reuse, and consistent zero-variance column exclusion.
* Added centralized HUGIML and benchmark hyperparameter grid configuration used by classifier tuning, the benchmark runner, and the dashboard Workbench.
* Added ``experiments/`` reproducibility workflows for regenerating benchmark and scalability dashboard outputs.
* Updated tuning defaults so candidate sweeps use leaner execution while the returned refit model remains audit-ready unless production mode is explicitly requested.

1.1.12 — 2026-06-20
~~~~~~~~~~~~~~~~~~~

* Added interaction-information source selection for augmented-pair features, including ``augmented_pair_mode``, ``aug_feature_size``, and ``ii_partner_size``.
* Added ``interaction_relaxed_mining`` for survivor-led native mining without pair-operator feature columns.
* Added pair-aware adaptive-binning metadata and survivor-led audit fields for governance review.
* Kept ``augmented_pair_max_features`` as a v1.1.11-compatible alias for the new augmented-pair source budgets.

1.1.11 — 2026-06-17
~~~~~~~~~~~~~~~~~~~

* Added expanded Governance Studio evidence views for dashboard review, including adaptive binning, augmented-pair traceability, feature-family review, pattern coverage, cross-validation monitoring, and export-ready governance summaries.
* Added binary feature conversion: two-value numeric columns are treated as categorical indicators before HUG pattern mining and downstream feature preparation.

1.1.10 — 2026-06-12
~~~~~~~~~~~~~~~~~~~

* Strengthened build and import reliability for the native backend with clearer install diagnostics, a public ``check_native()`` verification helper, cleaned package exports, and preserved classifier naming compatibility.
* Upgraded Governance Studio into a richer workbench experience for dataset setup, candidate model comparison, drill-down review, workflow guidance, demo data, and complete governance evidence views.

1.1.9 — 2026-06-10
~~~~~~~~~~~~~~~~~~

* Added ``HUGIMLClassifier`` as the primary public class name while keeping ``HUGIMLClassifierNative`` backward-compatible.
* Added ``execution_mode="audit"|"production"``. Audit is the default; production keeps prediction and save/load state while omitting large training-review artifacts.
* Added clear guidance from audit/governance methods when production-mode models do not retain full traceability data.
* Reduced memory pressure in strict and hybrid modes through earlier original-feature selection, compact native code storage, CSR-returning matrix paths, and conditional dense/CSR downstream handling.
* Improved fixed-B numeric handling so clean numeric columns remain numeric and only training columns with missing or infinite values use missing-aware bin labels.
* Optimized hybrid prediction so selected original columns are prepared directly.

1.1.8 — 2026-06-08
~~~~~~~~~~~~~~~~~~

* Added fast tuning for eligible adaptive-binning hyperparameter searches, reducing repeated mining work during cross-validation while preserving ordinary tuning behavior for unsupported grids.
* Added HUGIML Governance Studio, a Streamlit dashboard for validation, representation audit, pattern inventory, case review, configuration comparison, pruning analysis, monitoring, and report-oriented review workflows.
* Improved ``feature_mode="original_plus_interactions"`` so higher-order pattern selection is based on structural pattern order rather than parsing display labels.
* Expanded benchmark and dashboard runner utilities for more consistent validation-fold comparisons.

1.1.7 — 2026-06-05
~~~~~~~~~~~~~~~~~~

* Added a native ``L=2`` mining hot path for common two-item pattern workloads.
* Expanded augmented-pair features with two additional operations: pair sums and signed differences.
* Revised the benchmark baseline notebook and exported HTML with the updated baseline run and current HUGIML grid settings.
* Preserved the v1.1.6 augmented-feature APIs, feature-lineage metadata, and serialization compatibility while extending pair-operation coverage.

1.1.6 — 2026-06-04
~~~~~~~~~~~~~~~~~~

* Added native augmented-pair features for ``L > 1`` adaptive-binning models, including product and absolute-difference transforms.
* Added strict global ``topK`` budgeting across original features, HUG patterns, and augmented-pair features.
* Added raw-scale augmented-pair interpretation metadata, downstream feature composition summaries, and hybrid-model explainability report fields.

1.1.5 — 2026-06-01
~~~~~~~~~~~~~~~~~~

* Reduced native transaction memory use by storing compact item ids with shared item-level utility lookup instead of repeating utility values in every materialized transaction entry.
* Integrated adaptive binning into the fused native ``L=1`` execution path, avoiding an intermediate binned-matrix materialization for common adaptive workflows.
* Parallelized native adaptive bin selection and bin-code application; ``n_jobs`` is now applied before adaptive preprocessing when native support is available.
* Improved large-data stability with clearer native memory and timeout errors, plus safer fallback behavior under memory pressure.

1.1.4 — 2026-05-31
~~~~~~~~~~~~~~~~~~

* Added the native L1 hot path for ``L=1`` fits, fusing transaction preparation, single-item pattern mining, information-gain filtering, top-K retention, and sparse matrix construction in the C++ path.
* Moved adaptive binning selection into the C++ backend with supervised information-gain scoring and elbow-style stopping.
* Preserved Python-side adaptive-binning metadata including ``per_feature_b_``, ``_bin_edges_``, and ``ig_scores_`` for inspection and serialization.

1.1.3 — 2026-05-29
~~~~~~~~~~~~~~~~~~

* Optimized the native mining path by pushing the effective ``topK`` budget into mining, closer to the original Java implementation.
* Added performance and memory improvements, including row-stripe chunked transaction construction.
* Enforced structured mining constraints exactly and kept EUCS pruning disabled by default for predictable compound-pattern behavior.
* Revised domain-specific notebook examples.

1.1.2 — 2026-05-27
~~~~~~~~~~~~~~~~~~

* Added ``feature_mode`` to ``HUGIMLClassifierNative``.
* Supported ``patterns_only``, ``original_plus_patterns``, and ``original_plus_interactions`` downstream representations.
* Preserved pattern-space behavior for ``transform()``, ``fit_transform()``, ``get_hug_features()``, and ``get_pattern_info()``.
* Updated serialization, summaries, feature importances, and tests for hybrid feature modes.

1.1.1 — 2026-05-26
~~~~~~~~~~~~~~~~~~

* Fixed compound-pattern information-gain handling so L2+ patterns survive positive ``G`` after item-list release.
* Added notebooks and examples.

1.1.0 — 2026-05-23
~~~~~~~~~~~~~~~~~~

* Added adaptive binning and supervised per-feature ``B`` selection.
* Added native missing-value handling for numerical ``NaN`` and infinite values.
* Added plotting, pruning, interpretability metrics, multiclass/imbalance helpers, benchmarks, and expanded documentation assets.

Changelog
=========

The canonical release notes live in ``CHANGELOG.md`` at the repository root. Keep that file synchronized with the package version, PyPI release, GitHub tag, and Read the Docs build.

Recent releases
---------------

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

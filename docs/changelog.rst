Changelog
=========

The canonical release notes live in ``CHANGELOG.md`` at the repository root. Keep that file synchronized with the package version, PyPI release, GitHub tag, and Read the Docs build.

Recent releases
---------------

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

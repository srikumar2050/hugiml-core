Native mining pruning controls
==============================

Native mining pruning controls reduce C++ HUIM search work during ``fit()``.
They are separate from the user-facing ``PatternEditor`` workflow, which edits
an already fitted representation for governance review.

Pruning paths
-------------

.. list-table::
   :header-rows: 1
   :widths: 18 28 34 20

   * - Path
     - When active
     - Purpose
     - Controls
   * - LIU
     - Compound-pattern mining, including the ``L=2`` hot path. Bounded
       classifier mining uses exact candidate evidence before raising the
       utility floor.
     - Raises the utility threshold from locally strong candidate sequences so
       weaker branches can be skipped earlier.
     - Tune ``L``, ``G``, and ``topK``. There is no separate public LIU switch.
   * - LA
     - Generic utility-list child construction on the ordinary utility-ranked
       branch.
     - Stops constructing a child utility list once the remaining upper bound
       cannot pass the current retained-pattern utility floor.
     - Tune ``topK`` and ``G``. Relaxed-root interaction branches bypass this
       utility-floor shortcut when needed.
   * - EUCS
     - Eligible ``L > 1`` native mining paths after the admitted item set is
       known. Small, dense, or very wide pair spaces are skipped automatically.
     - Builds a pair co-occurrence utility cache and skips pair intersections
       whose pair-level utility cannot enter the retained set.
     - Environment variables described below.

EUCS parameters
---------------

EUCS is enabled by default for eligible ``L > 1`` native mining paths, subject
to workload gates that avoid building the cache when it is unlikely to pay off.

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Variable
     - Default
     - Meaning
   * - ``HUGIML_EUCS_ENABLE`` or ``HUGIML_EUCS_ENABLED``
     - enabled
     - Set to ``0``, ``false``, ``no``, ``off``, ``disable``, or ``disabled``
       to disable EUCS. Set to ``1``, ``true``, ``yes``, ``on``, ``enable``,
       or ``enabled`` to enable it. Invalid values keep the default.
   * - ``HUGIML_EUCS_MIN_ITEMS``
     - ``32``
     - Skip EUCS when the admitted item universe is this size or smaller.
   * - ``HUGIML_EUCS_MAX_CELLS``
     - ``6000000``
     - Maximum pair-cache cells allowed before EUCS is skipped.
   * - ``HUGIML_EUCS_MAX_DENSITY``
     - ``0.20``
     - Maximum observed active-item density allowed before EUCS is skipped.

Practical guidance
------------------

Most users should leave these native controls at their defaults and tune the
model-level parameters first:

* ``L`` controls maximum pattern length.
* ``G`` controls the information-gain gate.
* ``topK`` controls the retained pattern budget.

EUCS environment controls are most useful for benchmarking, profiling, or
workloads with unusually sparse or dense pair spaces.

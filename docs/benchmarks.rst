Benchmarks
==========

HUGIML provides one internal evaluation and four external classification
workflows. Each workflow preserves its benchmark-specific split and validation
structure, stores progress in a resumable checkpoint, and assembles predictive,
timing, complexity, and RPTE analysis artifacts.

Install the optional benchmark dependencies before using the runners:

.. code-block:: bash

   pip install "hugiml-core[benchmarks]"

The repository runners live under ``experiments/benchmark``. They are included
in the source distribution and source checkout, rather than installed as wheel
package modules. Commands below are run from the repository root.

Evaluation suites
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 14 42 24

   * - Suite
     - Catalog
     - Validation protocol
     - Search size
   * - Internal panel
     - 100 datasets
     - Five-fold shuffled stratified outer CV with three-fold inner selection
     - 16 configurations for HUGIML and primary baselines
   * - OpenML-CC18
     - 72 tasks
     - Official outer splits with nested three-fold selection
     - 16 configurations for HUGIML and primary baselines
   * - PMLBmini
     - 44 datasets
     - Three independently shuffled rotating three-fold partitions
     - 16 configurations for HUGIML and primary baselines
   * - TabZilla
     - 36 tasks
     - Official rotating train, validation, and test folds
     - 16 configurations for HUGIML and primary baselines
   * - TabArena classification
     - 38 datasets
     - Repeated outer three-fold evaluation with retained inner eight-fold ensembles
     - 16 HUGIML configurations; official tuned methods use 200

The primary baselines are XGBoost, LightGBM, Random Forest, and logistic
regression. EBM and RuleFit are available in the shared external runner with
eight configurations because their searches are substantially more expensive.
All preprocessing is learned from training rows only.

Internal benchmark
------------------

The internal panel contains 50 real-world and 50 synthetic binary
classification datasets. The synthetic panel exercises interactions,
missingness, and varied feature structures. The selected estimator is fitted
on the complete outer-training partition after inner model selection.

.. code-block:: powershell

   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\benchmark_dashboard.py --fresh
   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\benchmark_dashboard.py --resume
   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\benchmark_dashboard.py --assemble

Use ``--datasets`` and ``--models`` for explicit panels, ``--max-pairs`` for
staged execution, and ``--row-cap`` for a stratified row cap. The checkpoint and
assembled artifacts are written to ``experiments/benchmark/results`` unless an
alternative output directory is supplied.

External benchmark workflow
---------------------------

External evaluations use local offline dataset folders containing prepared
features, targets, metadata, checksums, and official or deterministic split
definitions. Each suite follows the same operational sequence:

#. Download the dataset panel and split definitions.
#. Verify the local cache.
#. Execute or resume selected task and model pairs.
#. Assemble the checkpoint into JSON, CSV, and HTML reports.

All dataset and result locations are resolved relative to the repository.

OpenML-CC18
~~~~~~~~~~~

.. code-block:: powershell

   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\openml_cc18\download_openml_cc18_datasets.py --smallest 72
   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\openml_cc18\run_openml_cc18_offline_benchmark.py --verify-cache
   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\openml_cc18\run_openml_cc18_offline_benchmark.py --models all --validation-protocol nested --resume

The workflow evaluates stored OpenML task partitions and performs three-fold
selection inside every official outer-training partition. See
``experiments/benchmark/openml_cc18/README.md`` for panel selection and staged
download options.

PMLBmini
~~~~~~~~

.. code-block:: powershell

   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\pmlbmini\download_pmlbmini_datasets.py
   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\pmlbmini\run_pmlbmini_offline_benchmark.py --verify-cache
   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\pmlbmini\run_pmlbmini_offline_benchmark.py --models all --validation-protocol rotating --resume

The rotating protocol uses fold ``F_i`` for testing, fold ``F_(i+1) mod K``
for validation, and the remaining fold for training. The validation winner is
retained without a post-selection refit.

TabZilla
~~~~~~~~

.. code-block:: powershell

   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\tabzilla\download_tabzilla_datasets.py
   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\tabzilla\run_tabzilla_offline_benchmark.py --verify-cache
   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\tabzilla\run_tabzilla_offline_benchmark.py --models all --validation-protocol rotating --resume

TabZilla uses its stored official folds. Candidate selection uses validation
ROC AUC when defined and balanced accuracy when a validation fold omits a
class. Final multiclass reporting uses fold AUC where defined and pooled
out-of-fold probabilities for affected tasks.

TabArena
~~~~~~~~

.. code-block:: powershell

   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\tabarena\download_tabarena_datasets.py
   .\.venv-hugiml\Scripts\python.exe -m pip install --no-deps --target experiments\benchmark\tabarena\_dependencies -r experiments\benchmark\tabarena\requirements-tabarena.txt
   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\tabarena\run_tabarena_offline_benchmark.py --verify-cache
   .\.venv-hugiml\Scripts\python.exe experiments\benchmark\tabarena\run_tabarena_offline_benchmark.py --models all --validation-protocol tabarena --resume

TabArena uses official outer partitions and eight-fold stratified selection
inside each outer-training partition. The eight fitted children belonging to
the selected configuration are retained and their outer-test probabilities are
averaged. Binary configurations are selected by ROC AUC and multiclass
configurations by log loss. AutoGluon's model-agnostic feature generator is
fitted independently within every child training fold and retained with that
child for outer-test transformation.

Resume behavior
---------------

``--fresh`` starts a new checkpoint. ``--resume`` preserves successful,
compatible dataset/model pairs and continues incomplete pairs. Compatibility
includes the dataset hashes, split definitions, selected models, search grids,
validation protocol, random state, and HUGIML scenario. A specific stored run
can be selected with ``--resume-run-id``.

Use ``--task-ids`` or ``--task-ids-file`` for explicit task panels,
``--smallest`` or ``--first`` for ordered subsets, ``--defer-task-ids`` to move
expensive tasks to the end, and ``--max-pairs`` to bound one invocation.

Results and dashboards
----------------------

Each external suite stores a sanitized ``benchmark_results.json`` beside its
runner. Assembly produces dataset/model CSV files, fold-level metrics, model
summaries, complexity tables, RPTE distributions, and a self-contained HTML
dashboard. Reported predictive metrics include ROC AUC, balanced accuracy, F1,
accuracy, and Brier score where applicable. Timing fields distinguish tuning,
fit, prediction, and total pair time and state when fitting is included in
tuning.

Complexity reporting uses three levels:

* model units for coarse active components;
* model-inspection units for all fitted evidence reviewed during a complete
  model audit; and
* instance-inspection units for the evidence used by one prediction.

HUGIML reports the selected LR or RPTE route and, for RPTE, input count, active
trees, active leaves, active direct terms, and active average leaf path length.
Enable the RPTE dashboard section during assembly with
``--include-rpte-dashboard``.

.. code-block:: powershell

   .\.venv-hugiml\Scripts\python.exe RUNNER.py --assemble --include-rpte-dashboard

The repository root contains the assembled internal, OpenML-CC18, PMLBmini,
TabZilla, and TabArena leaderboard dashboards linked by ``index.html``.

Package benchmark runner
------------------------

The installed package also provides a compact benchmark runner for packaged or
user-supplied datasets:

.. code-block:: bash

   python -m hugiml.benchmarks.runner --datasets breast_cancer adult credit --output benchmarks/results/
   hugiml-bench --datasets breast_cancer --output results/

Use ``--data`` and ``--target`` for a user-supplied CSV, TSV, Excel, or Parquet
classification dataset. Add ``--tune`` for nested parameter selection. This
compact runner is separate from the suite-specific offline workflows above.

Reproducibility
---------------

``experiments/benchmark/REPRODUCING.md`` defines the release-neutral
environment setup, dependency constraints, cache verification, smoke tests,
complete runner commands, resume behavior, and artifact expectations for both
the internal and external evaluations.

Benchmark comparisons should be interpreted as a predictive-performance,
runtime, and inspection-complexity trade-off rather than as a universal model
ranking. Compare models only on matched datasets and splits, and account for
the different fold counts and retained-ensemble structures across suites.

# TabArena offline benchmark

This folder targets the official **TabArena-v0.1 classification subset**: 38
classification datasets from the 51-dataset mixed benchmark using its published
classification catalog.

The downloader writes to `../../../datasetsTabarena` by default, stores features and encoded/original targets as Parquet, and stores every prescribed split separately. For the supported classification catalog, it uses 10 repeated three-fold partitions below 2,500 rows and 3 repeated three-fold partitions otherwise.

```powershell
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\tabarena\download_tabarena_datasets.py
```

The benchmark runner is offline-only and exits with download guidance when
`datasetsTabarena` is absent. It uses the shared benchmark engine and dashboard
in the parent folder; this folder intentionally duplicates neither module.

```powershell
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\tabarena\run_tabarena_offline_benchmark.py
```

Use `--smallest N` for the `N` smallest datasets by row count, `--first N` for
the first `N` datasets in official TabArena order, or `--task-ids` /
`--task-ids-file` for an explicit selection. `--verify-cache` validates the
offline cache and exits; `--verify-cache-before-run` validates and then runs.

Validation defaults to `--validation-protocol tabarena`. Every outer training
partition uses eight-fold stratified CV for configuration selection. The eight
child models belonging to the selected configuration are retained and their
probabilities are averaged for the outer test fold; there is no final refit.
Binary configurations are selected by ROC AUC and multiclass configurations by
log loss.

The TabArena protocol additionally requires AutoGluon's model-agnostic feature
generator:

```powershell
.\.venv-hugiml\Scripts\python.exe -m pip install --no-deps --target experiments\benchmark\tabarena\_dependencies -r experiments\benchmark\tabarena\requirements-tabarena.txt
```

The dedicated target directory and `--no-deps` preserve the benchmark
environment's established NumPy, pandas, scikit-learn, and PyArrow versions.
The runner adds this directory only when the TabArena protocol reaches its
model-agnostic preprocessing step.

For every inner fold, `AutoMLPipelineFeatureGenerator` is fitted only on that
child model's training rows. The transformed training and validation matrices
are reused across hyperparameter candidates for the same fold. Its fitted state
remains attached to the retained child for outer-test transformation. Other
validation protocols retain their existing preprocessing.

Every outer split records performance metrics, selected parameters, candidate
and retained-child counts, model units, model-inspection units, and pooled
instance-inspection moments. Timing is separated into tuning, final fitting,
prediction, complexity measurement, and remaining evaluation overhead. Under
the TabArena protocol the retained ensemble is produced during tuning, so
`fit_seconds` is zero and `fit_time_included_in_tuning` is true; no additional
outer-training refit is performed.

The assembled outputs include Elo ratings with 95% dataset-bootstrap intervals,
normalized score, average rank, harmonic mean rank, pairwise win rate, and
improvability. `--validation-protocol nested` and
`--validation-protocol rotating` remain available for separate analyses but do
not represent the TabArena evaluation protocol.

When the official lightweight result artifact is present at
`official_tabarena_results/tabarena_v0_1_classification_official_results.parquet`,
assembly also produces an official-reference comparison. Local HUGIML errors
are matched by dataset and exact outer split, then averaged within each dataset
before calculating Elo, ranks, normalized score, improvability, and pairwise
win/tie/loss results. Binary error is one minus ROC AUC and multiclass error is
log loss. The outer test partitions and metrics are aligned; inner selection,
search spaces, compute budgets, and retained ensembles remain method-specific.
Assembly creates the regular benchmark dashboard and a separate official-reference
leaderboard dashboard. The latter provides overall, binary, and multiclass views
for default, tuned, and combined official pools; default overall is selected initially.
Mean and median comparison statistics are displayed compactly as mean (median).

The official dashboard opens with a 2 x 2 tuned-model chart. The row and predictor
boundaries are the median row and raw predictor counts among completed HUGIML
datasets. Bars show mean ROC-AUC delta
from the strongest tuned baseline, dots show the median delta, and the right column
reports actual mean and median ROC AUC. Ordering follows TabArena rank, which uses
ROC-AUC error for binary datasets and log loss for multiclass datasets. The
methodology section at the bottom is collapsed by default. When detailed official
predictions cover only part of a quadrant, the chart reports the available-data
mean and median.

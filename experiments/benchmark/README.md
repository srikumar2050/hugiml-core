# Internal benchmark

The complete panel contains 100 datasets. The selection below evaluates two
HUGIML scenarios, standard and complexity-budgeted XGBoost, LightGBM and Random
Forest, plus EBM and RuleFit. This produces 1,000 aggregate results and 5,000
outer-fold results. Standalone Logistic Regression is excluded.

Run the first 800 configurations from the repository root in PowerShell:

```powershell
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\benchmark_dashboard.py `
  --out-dir experiments\benchmark\internal_results_100datasets `
  --datasets all `
  --hugiml-scenarios augmented_pair,interaction_relaxed `
  --models "HUGIML,XGB standard,XGB complexity-budgeted,LightGBM standard,LightGBM complexity-budgeted,RandomForest standard,RandomForest complexity-budgeted,EBM,RuleFit" `
  --execute-models "HUGIML,XGB standard,XGB complexity-budgeted,LightGBM standard,LightGBM complexity-budgeted,RandomForest standard,RandomForest complexity-budgeted" `
  --n-splits 5 --inner-splits 3 `
  --outer-jobs 5 --n-jobs 3 `
  --lr-source-policy main_effect --resume
```

Run EBM and RuleFit separately for the remaining 200 configurations:

```powershell
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\benchmark_dashboard.py `
  --out-dir experiments\benchmark\internal_results_100datasets `
  --datasets all `
  --hugiml-scenarios "augmented_pair,interaction_relaxed" `
  --models "HUGIML,XGB standard,XGB complexity-budgeted,LightGBM standard,LightGBM complexity-budgeted,RandomForest standard,RandomForest complexity-budgeted,EBM,RuleFit" `
  --execute-models "EBM,RuleFit" `
  --n-splits 5 --inner-splits 3 `
  --outer-jobs 5 --n-jobs 3 `
  --lr-source-policy main_effect --resume
```

Both commands use the same complete panel and checkpoint. `--execute-models`
selects which models run in that invocation and preserves every completed result.
Worker allocations are chosen for the entire panel and remain common across
both invocations. Either command can be repeated to resume its selected models.

Each outer-training partition performs three-fold inner tuning, then refits the
selected candidate before evaluating its untouched outer-test partition.
`--lr-source-policy` accepts `standard`, `main_effect`, and `strict` and applies
to both HUGIML scenarios, including runs with `--no-tune`.

`--outer-jobs` limits concurrent outer folds. `--n-jobs` limits inner validation
workers per outer fold. With 24 processors and sufficient available RAM, the
command uses five outer folds with three inner workers for every model. Individual
fits use one worker; native worker thread pools are limited to one thread.
HUGIML retains its cached tuning path. Final refits occur within the outer workers.
Automatic joblib memory mapping is disabled in outer and inner workers; arrays
are transferred without temporary memory-map files. Execution messages include
the selected dataset position, such as `[1/100]`, along with fold progress.

The scheduler checks available RAM before each dataset and reserves 30 percent.
It estimates all selected models and uses one common allocation that accommodates
the most demanding model. That allocation is recorded before training and used
by every model on the dataset, including after resume. Its working-memory estimate
includes data copies, feature matrices, model construction and process overhead.
It reduces concurrency for the entire dataset when needed.
This estimate is a planning allowance, not a guaranteed peak-memory bound. If
available RAM cannot be read, execution uses one outer and one inner worker.
Use the full worker budget when other benchmark runners have finished.

Completed folds are written atomically by the parent process as they arrive.
Checkpoint writes use unique temporary files and retry transient file-access
conflicts before replacing the destination. If replacement remains blocked,
the existing checkpoint and the complete temporary file are retained.
The same command resumes successful folds and retries unsuccessful folds.
The checkpoint records the source policy, validation settings and requested
worker limits; resume rejects incompatible settings. Effective worker counts
are also recorded with each fold.

`pair_seconds` measures an uninterrupted dataset/model run. A resumed pair has
`pair_seconds=null` and `timing_comparable=false`; `invocation_seconds` measures
only its current invocation. Full-run timing summaries include only datasets
with uninterrupted timings for every selected model and report that dataset
count. Fold fit and tuning durations remain available separately. Tuning time
already includes the final refit, so fit time must not be added to it.

Five outer folds and three inner folds permit 15 concurrent fits under this
common allocation. This retains HUGIML's cached tuner and avoids giving another
model a larger worker budget. Filling 24 processors would require scheduling
across datasets and measuring throughput under that different schedule.

Saved data excludes diagnostic text, absolute machine paths and calendar
timestamps. Completion status, failure counts and numeric durations are retained.

After completion, assemble the dashboard using:

```powershell
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\benchmark_dashboard.py `
  --out-dir experiments\benchmark\internal_results_100datasets --assemble
```

RuleFit uses stratified internal regularization validation with up to five folds,
bounded by the smallest class in the training data passed to each fit. At least
two examples per class are required. This internal selection is separate from
the benchmark's five outer folds and three inner folds.

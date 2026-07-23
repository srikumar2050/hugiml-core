# OpenML-CC18 benchmark dashboard

This workflow evaluates HUGIML and a mixed panel of predictive and interpretable classifiers on the official OpenML-CC18 train/test splits. It supports an offline dataset cache, task and model selection, resumable execution, statistical comparisons, inspection-unit analysis, and CSV, JSON, and HTML reports.

## Evaluation protocol

- Suite: `OpenML-CC18`.
- Default task panel: every task available in the local dataset cache.
- Optional task selection: use `--smallest` for the smallest cached tasks, `--task-ids` for a comma-separated list, or `--task-ids-file` for one or more text files.
- Default model panel: all supported models.
- Outer evaluation: every official `(repeat, fold, sample)` split stored by OpenML. No outer folds are generated locally.
- Inner model selection: stratified three-fold cross-validation inside each official training partition by default.
- Primary score: binary ROC AUC or multiclass one-vs-rest macro ROC AUC.
- Reporting: predictive metrics, execution times, model inspection measures, pair summaries, statistical comparisons, task-level ratios, and per-split records.

The model definitions, preprocessing policy, parameter grids, HUGIML scenarios, and inspection-unit calculations are shared with `benchmark_dashboard.py`. Non-HUGIML estimators receive the numeric matrix produced by imputation and one-hot encoding, independent of generated feature labels.

## Supported models

The canonical model labels are:

- `HUGIML`
- `XGB standard`
- `LightGBM standard`
- `RandomForest standard`
- `Logistic Regression`
- `EBM`
- `RuleFit`

`--models` accepts canonical labels, short aliases, model families, or `all`.

| Selector | Models |
|---|---|
| `all` | Every supported model |
| `ensemble` | XGBoost, LightGBM, and Random Forest |
| `interpretable` | Logistic Regression, EBM, and RuleFit |
| `hugiml` | HUGIML only |
| `xgb`, `lightgbm`, `rf`, `lr`, `ebm`, `rulefit` | A specific model |

Selectors can be combined. For example, `--models hugiml,interpretable` runs HUGIML and all interpretable baselines.

## Environment

Run the commands from the repository root so the benchmark scripts and the local HUGIML source package are available together.

```bash
python -m pip install -e ".[benchmarks]"
python -m pip install \
  openml pyarrow numpy pandas scipy scikit-learn \
  xgboost lightgbm interpret imodels
```

Dependency roles:

- `pyarrow`: Parquet storage for the offline dataset cache.
- `xgboost`: XGBoost baseline.
- `lightgbm`: LightGBM baseline.
- `interpret`: `ExplainableBoostingClassifier` for EBM.
- `imodels`: `RuleFitClassifier` for RuleFit.

Logistic Regression uses scikit-learn and does not require an additional package. RuleFit is evaluated through a one-vs-rest wrapper so the same model selector can be used for binary and multiclass CC18 tasks.

## Download datasets

The downloader stores feature data, encoded targets, task metadata, and every official OpenML split. With no selection argument, it caches the smallest 36 tasks:

```bash
python experiments/benchmark/download_openml_cc18_datasets.py
```

Choose a different total:

```bash
python experiments/benchmark/download_openml_cc18_datasets.py --smallest 72
```

Add the next-smallest tasks that are not already complete in the cache. Existing complete tasks are retained, including tasks outside the smallest contiguous prefix:

```bash
python experiments/benchmark/download_openml_cc18_datasets.py --add 6
```

Re-download tasks that are already present in the cache:

```bash
python experiments/benchmark/download_openml_cc18_datasets.py --refresh
```

Use a different cache location when needed:

```bash
python experiments/benchmark/download_openml_cc18_datasets.py \
  --output /path/to/datasetsOpenml
```

## Run the benchmark

Run every cached task and every supported model:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py
```

Start with an empty results directory:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py --fresh
```

Run only the six smallest cached tasks:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --smallest 6 --fresh
```

Run explicit task IDs:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --task-ids 3,6,11 --fresh
```

Read task IDs from a file. Blank lines and text after `#` are ignored:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --task-ids-file experiments/benchmark/task_ids.example.txt --fresh
```

Use a non-default cache location:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --datasets-dir /path/to/datasetsOpenml
```

## Select models

Run all ensemble baselines:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --models ensemble
```

Run all interpretable baselines:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --models interpretable
```

Run HUGIML with the interpretable family:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --models hugiml,interpretable
```

Run specific models by alias:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --models hugiml,lr,ebm,rulefit
```

Canonical labels are also accepted:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --models "HUGIML,Logistic Regression,EBM,RuleFit"
```

## Inspection-unit comparisons

The dashboard keeps three measures separate:

1. **Model units**: the model family's native structural units.
2. **Model inspection units**: the complete fitted model content a reviewer would inspect.
3. **Instance inspection units**: the amount inspected for individual predictions, including confidence intervals when available.

For each task with HUGIML and a baseline, the report calculates:

```text
baseline model inspection units / HUGIML model inspection units
```

The ratio is included in the headline summary only when HUGIML AUC is within the configured tolerance of the baseline or higher. The default tolerance is `0.01` AUC:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --complexity-auc-tolerance 0.01
```

Use `0` for a strict HUGIML-at-least-as-good comparison:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --complexity-auc-tolerance 0
```

The HTML dashboard includes:

- explicit model-unit and inspection-unit columns;
- an interactive model-inspection-units versus AUC scatter;
- task-level instance inspection values, with confidence intervals when they are available for the complete displayed panel;
- per-task baseline-to-HUGIML ratios;
- baseline-specific summary statements with paired task count, mean AUC, median inspection units, and median ratio.

The dashboard follows the same card, summary-grid, responsive layout, and Indigo, Pearl, and Forest theme choices as the main benchmark dashboard. The interactive scatter uses the Plotly browser library. All underlying values remain available in the embedded JSON and exported CSV files if the chart library is unavailable.

## Resume and checkpoint behavior

Each distinct execution configuration receives its own run ID inside the checkpoint. The identity includes the selected datasets, resolved model list, tuning settings, random state, split limit, HUGIML settings, inspection comparison tolerance, parameter grids, source identity, and dataset signatures.

Resume the matching run:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py --resume
```

Only fully successful task/model pairs are skipped. A pair with unsuccessful official splits is opened again, successful splits are retained, and only the remaining splits are attempted. Each split keeps one current row with an incrementing attempt count.

When configuration values change, a separate run record is created in the same checkpoint. Results from other run IDs are retained but are not combined with the active dashboard or exported CSV files.

Process a portion of the task/model plan for batch scheduling:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --resume --start-pair 8 --max-pairs 4
```

A lightweight execution check can limit official splits and omit inner tuning:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --models lr \
  --max-official-splits 1 --max-pairs 1 --no-tune --fresh
```

## Optional cache validation

Normal execution checks the files required for each selected task as it is loaded. Use `--verify-cache` to validate the complete manifest, dataset checksums, split counts, and every referenced split file before model execution:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --verify-cache
```

## Assemble reports

Report assembly only needs a checkpoint; the dataset cache and OpenML connection are not required:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py --assemble
```

To assemble a retained run other than the active run, provide its run ID:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --assemble --run-id RUN_ID
```

Default outputs under `experiments/benchmark/openml_cc18_results/` are:

- `openml_cc18_checkpoint.json`
- `openml_cc18_pair_results.csv`
- `openml_cc18_official_split_results.csv`
- `openml_cc18_model_summary.csv`
- `openml_cc18_complexity_points.csv`
- `openml_cc18_complexity_ratios.csv`
- `openml_cc18_dashboard_data.json`
- `openml_cc18_benchmark_dashboard.html`
- `split_checkpoints/*.json`

The checkpoint writer uses atomic replacement and retains the previous valid JSON file as a backup.

## Dashboard template

Export the built-in template without a dataset cache:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --write-template openml_cc18_template.html
```

A custom template must preserve `{{TITLE}}`, `{{GENERATED_AT}}`, and `{{DATA_JSON}}`:

```bash
python experiments/benchmark/run_openml_cc18_offline_benchmark.py \
  --assemble --template-html openml_cc18_template.html
```

The built-in dashboard derives its protocol wording from the active run metadata, including task selection, model selection, tuning state, inner-fold count, and inspection comparison tolerance.

## Validation commands

```bash
PYTHONPATH=src:. pytest -q \
  tests/test_openml_cc18_benchmark_dashboard.py \
  tests/test_openml_cc18_dataset_downloader.py

python -m py_compile \
  experiments/benchmark/benchmark_dashboard.py \
  experiments/benchmark/download_openml_cc18_datasets.py \
  experiments/benchmark/run_openml_cc18_offline_benchmark.py
```

# HUGIML benchmark dashboard

The script runs a reproducible classifier benchmark for HUGIML and tree-ensemble baselines, checkpoints every dataset/model pair, and assembles the HTML dashboard from the checkpoint.

## Benchmark panel

The default panel contains 50 datasets:

- 30 public real-world datasets from scikit-learn and statsmodels.
- 20 synthetic datasets generated inside the script.

The synthetic datasets are deterministic and use `RANDOM_STATE = 2026` plus dataset-specific offsets. The real-world datasets are loaded from installed Python packages; no network access is required during a run.

## Evaluation protocol

The runner uses a stratified train/validation/test holdout protocol rather than k-fold cross-validation.

For each dataset:

1. `--row-cap` defaults to `-1`, so the full dataset is used. If `--row-cap` is set to a positive integer and the dataset has more rows than that value, the runner keeps a stratified subsample using `RANDOM_STATE = 2026`.
2. It creates a stratified test split: 20% test for datasets with at least 60 rows, otherwise 25% test.
3. It splits the remaining rows into train and validation, with validation equal to 25% of the train+validation pool.
4. Hyperparameters are selected by validation AUC on the train/validation split.
5. The selected configuration is refit on train+validation.
6. AUC, F1, and accuracy are reported on the held-out test split.

All splits use the same random seed so the benchmark is reproducible.

## Preprocessing policy

HUGIML receives native pandas dataframes. Categorical, object, string, and boolean columns are preserved as categorical-style features.

All non-HUGIML models receive train-fitted one-hot encoded matrices. The encoder is fitted once for validation selection and once for final train+validation evaluation. Final `fit_seconds` includes the final one-hot encoding fit/transform time plus the selected estimator fit time.

The checkpoint also records:

- `model_fit_seconds`
- `preprocess_fit_transform_seconds`
- `selection_preprocess_seconds`
- `model_features`
- `categorical_features`
- `preprocessing_policy`

## Models and hyperparameter grids

### HUGIML

```python
{
    "B": [-1],
    "adaptive_binning": [True],
    "L": [1, 2],
    "topK": [50, 100],
    "feature_mode": ["original_plus_patterns"],
    "G": [0.01, 0.001],
}
```

The runner also sets `execution_mode="production"` and `n_jobs=1` for HUGIML.

### XGB standard

```python
{
    "n_estimators": [100, 200],
    "max_depth": [3, 4],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0],
}
```

### XGB complexity-budgeted

```python
{
    "n_estimators": [25, 50, 75],
    "max_depth": [1, 2, 3],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
```

### LightGBM standard

```python
{
    "n_estimators": [100, 200],
    "learning_rate": [0.03, 0.1],
    "num_leaves": [15, 31],
    "subsample": [0.8, 1.0],
}
```

### LightGBM complexity-budgeted

```python
{
    "n_estimators": [25, 50, 75],
    "num_leaves": [2, 4, 8],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0],
    "min_child_samples": [5],
}
```

### RandomForest standard

```python
{
    "n_estimators": [200, 400],
    "max_depth": [4, 8, None],
    "min_samples_leaf": [1, 5],
}
```

### RandomForest complexity-budgeted

```python
{
    "n_estimators": [20, 50, 75],
    "max_leaf_nodes": [2, 4, 8],
    "min_samples_leaf": [1, 5],
    "max_depth": [None],
}
```

Complexity-budgeted tree models are selected subject to a 300-leaf budget when leaf complexity is available. HUGIML reports its selected pattern count and uses the selected `topK` as its displayed budget.

## Run a full benchmark

From the repo root:

```bash
python experiments/benchmark/benchmark_dashboard.py --fresh
```

On Windows Command Prompt:

```bat
python experiments\benchmark\benchmark_dashboard.py --fresh
```

`--fresh` recreates the local results directory before the run.

By default, outputs are written to:

```text
repo/experiments/benchmark/results/
```

## Resume a run

```bash
python experiments/benchmark/benchmark_dashboard.py --resume
```

This skips dataset/model pairs already present in the checkpoint.

## Run a subset

```bash
python experiments/benchmark/benchmark_dashboard.py \
  --datasets InterestInflationHighRate,SynthMixedMissing \
  --models HUGIML,"XGB standard"
```

On Windows Command Prompt:

```bat
python experiments\benchmark\benchmark_dashboard.py ^
  --datasets InterestInflationHighRate,SynthMixedMissing ^
  --models HUGIML,"XGB standard"
```

## Assemble the dashboard

After a complete run, assemble with:

```bash
python experiments/benchmark/benchmark_dashboard.py --assemble
```

The script automatically searches upward from `experiments/benchmark/` for:

```text
hugiml_benchmark_analysis_dashboard.html
```

So if the template is in the repo root, no template argument is needed.

To override the template:

```bash
python experiments/benchmark/benchmark_dashboard.py \
  --assemble \
  --template-html path/to/hugiml_benchmark_analysis_dashboard.html
```

## Output files

The results directory contains:

```text
benchmark_checkpoint.json
dashboard_data.json
details.csv
overall.csv
summary_by_scope.csv
scope_tests.csv
summary_comparison.csv
hugiml_benchmark_analysis_dashboard_revised.html
```

The HTML dashboard includes:

- Overall model summary.
- Real-world and synthetic summary tables.
- Pairwise model comparisons.
- Complexity vs performance chart with extra plot padding.
- Dataset column profile table with dtype, missingness, uniqueness, and summary statistics.

## Useful options

```text
--fresh                         recreate the results directory before running
--resume                        skip completed dataset/model pairs
--assemble                      build dashboard files from the checkpoint
--datasets A,B,C                run selected datasets
--models M1,M2                  run selected model labels
--start-pair N                  start at pair index N
--max-pairs N                   run at most N pairs
--row-cap N                     cap rows per dataset; default -1 uses the full dataset
--checkpoint PATH               custom checkpoint path
--out-dir PATH                  custom output directory
--template-html PATH            custom dashboard template path
--hugiml-max-fit-seconds FLOAT  optional HUGIML runtime cap
```

## Notes

- A complete dashboard assemble expects all default dataset/model pairs to be present in the checkpoint.
- For partial runs, inspect the checkpoint or CSVs directly.

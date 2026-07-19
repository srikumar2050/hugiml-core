# HUGIML benchmark dashboard

The script runs a reproducible classifier benchmark for HUGIML and baseline models, checkpoints every dataset/model/configuration pair, and assembles the HTML dashboard from the checkpoint. The dashboard provides two HUGIML path options in one selector while sharing the same baseline results.

The dashboard evaluation protocol is aligned with `hugiml.benchmarks.runner` when runner is invoked with tuning enabled:

```bash
python -m hugiml.benchmarks.runner --tune --n-splits 5 --inner-splits 3 --random-state 42
```

## Benchmark panel

The default panel contains **100 binary classification tasks**:

- **50 public real-world tasks** from scikit-learn and statsmodels.
- **50 synthetic tasks** generated inside the script.

The real-world panel contains package-bundled binary classification tasks covering biomedical, demographic, economic, behavioral, environmental, and social-science domains.

The synthetic panel covers linear and nonlinear decision boundaries, categorical and numeric interactions, missingness, imbalance, high-dimensional noise, correlated features, threshold and piecewise rules, parity/XOR, exact-cardinality, DNF/CNF, multiplexers, hierarchical and gated interactions, sparse conjunctions, and mixed-order logic.

All stochastic benchmark operations use `--random-state`; the default is **42**, matching `hugiml.benchmarks.runner`. Synthetic generators derive deterministic dataset-specific streams or offsets from that value. Real-world source data are fixed, while any randomized row subsampling, cross-validation splitting, tuning, and supported model initialization also use seed 42 by default.

The real-world datasets are loaded from installed Python packages; no network access is required during a run. Install the benchmark dependency set with `pip install "hugiml-core[benchmarks]"`; it includes the `statsmodels` dependency used by the expanded real-world panel.

This experiment runner and its documentation are included in the project source archive and source checkout. They are not installed as importable wheel-package modules under `hugiml`.

## Evaluation protocol

The default dashboard protocol is nested cross-validation:

```text
outer_cv_inner_cv_tuning
```

The defaults are:

```text
outer folds:  5
inner folds:  3
random seed:  42
scoring:      roc_auc
tuning:       on
```

For each dataset/model pair:

1. `--row-cap` defaults to `-1`, so the full dataset is used. If `--row-cap` is set to a positive integer and the dataset has more rows than that value, the script keeps a stratified subsample using `--random-state`.
2. The benchmark creates an outer `StratifiedKFold(n_splits=--n-splits, shuffle=True, random_state=--random-state)` split.
3. For each outer train fold, hyperparameters are selected with inner CV using `--inner-splits` and ROC-AUC scoring.
4. The selected model is refit on the full outer train fold.
5. Metrics are computed on the held-out outer test fold.
6. The checkpoint row for one dataset/model stores the mean across outer folds.

This means each row in `details.csv` is an outer-CV aggregate, not a single holdout test result.

To disable inner-CV tuning and evaluate the model defaults with outer CV only, use:

```bash
python experiments/benchmark/benchmark_dashboard.py --fresh --no-tune
```

That protocol is recorded as:

```text
outer_cv_no_inner_tuning
```

## Runner consistency

The dashboard defaults match `hugiml.benchmarks.runner` with tuning enabled:

| Setting | Dashboard default | Runner equivalent |
|---|---:|---:|
| Outer CV folds | `--n-splits 5` | `--n-splits 5` |
| Inner CV folds | `--inner-splits 3` | `--inner-splits 3` |
| Random seed | `--random-state 42` | `--random-state 42` |
| Tuning | on by default | `--tune` required |
| Scoring | `roc_auc` | `roc_auc` |

Runner command with the same validation protocol:

```bash
python -m hugiml.benchmarks.runner --tune --n-splits 5 --inner-splits 3 --random-state 42
```

Dashboard command with the same validation protocol:

```bash
python experiments/benchmark/benchmark_dashboard.py --fresh --n-splits 5 --inner-splits 3 --random-state 42
```

## Environment manifest

The assemble step can optionally write a privacy-sanitized environment manifest:

```bash
python experiments/benchmark/benchmark_dashboard.py --assemble --include-sbom
```

When enabled, the results directory includes:

```text
benchmark_reproducibility_sbom.json
```

The manifest is stored as a separate JSON file and is not included in the HTML dashboard.

## Metrics

The dashboard records runner-style metrics for each dataset/model aggregate:

- `roc_auc`
- `balanced_accuracy`
- `avg_precision`
- `brier`
- `accuracy`
- `f1`
- `fit_ms`
- `predict_ms`
- `tune_ms`
- `best_inner_score`

For dashboard compatibility, the following aliases are also written:

| Alias | Meaning under the nested-CV protocol |
|---|---|
| `auc` | same value as outer-CV mean `roc_auc` |
| `valid_auc` | same value as mean `best_inner_score` across outer folds |
| `fit_seconds` | `fit_ms / 1000`, averaged across outer folds |

`fit_ms` measures only the final selected estimator refit on each outer-training fold. `tune_ms` measures the inner-CV model search and excludes that final refit. For HUGIML, the tuning API exposes the final refit duration as `HUGIMLTuneResult.refit_time_`, which the benchmark writes into each outer-fold checkpoint. Missing timings remain unavailable rather than being converted to zero.

`--assemble` does not recompute model metrics. It reads the checkpoint and builds CSV/JSON/HTML outputs from the aggregate rows already stored there. Use a fresh checkpoint when changing the validation protocol.

## Statistical tests

Dashboard statistical tests use datasets as paired observations.

For Wilcoxon comparisons, the sample size is the number of datasets where both compared models have valid outer-CV aggregate ROC-AUC values. Inner folds and outer folds are not counted as independent observations for Wilcoxon or Friedman tests.

Using more outer folds can make each dataset-level estimate more stable, but the statistical tests mainly depend on the number of datasets. For stronger summaries, prefer many datasets. A heavier but common dashboard run is:

```bash
python experiments/benchmark/benchmark_dashboard.py --fresh --n-splits 10 --inner-splits 3 --random-state 42
python experiments/benchmark/benchmark_dashboard.py --assemble
```

Use the same scheme in runner when comparing outputs:

```bash
python -m hugiml.benchmarks.runner --tune --n-splits 10 --inner-splits 3 --random-state 42
```

## Fast HUGIML tuning and cached hot path

When tuning is enabled, HUGIML uses the fast tuning entry point:

```python
HUGIMLClassifierNative.tune(
    X_train,
    y_train,
    cv=inner_splits,
    shuffle=True,
    random_state=random_state,
    scoring="roc_auc",
    param_grid=grid_dict,
    base_params=base_params,
    refit=True,
    use_fast_path=True,
)
```

This requests HUGIML's cached fast path. When the grid is eligible, HUGIML can reuse cached mining artifacts across compatible candidates. The checkpoint records HUGIML fast-path metadata, including:

- `hugiml_fast_path_requested`
- `hugiml_fast_path_used`
- `hugiml_tune_elapsed_seconds`
- `hugiml_tune_n_splits`

The fast tuning path is not used when `--no-tune` is supplied.

## Preprocessing policy

HUGIML receives native pandas dataframes. Categorical, object, string, and boolean columns are preserved as categorical-style features.

Non-HUGIML baselines use runner-compatible sklearn pipelines. Numeric columns are median-imputed. Non-numeric columns are most-frequent-imputed and one-hot encoded. The pipeline is fit inside each inner CV fold during hyperparameter selection and refit on each outer train fold before evaluation.

The checkpoint records:

- `model_features`
- `categorical_features`
- `preprocessing_policy`
- `fit_ms`
- `predict_ms`
- `tune_ms`

## Models and hyperparameter grids

### HUGIML scenarios

The dashboard contains two HUGIML path options in the existing selector. Baselines are shared across both options.

#### Augmented pair path

```python
{
    "B": [-1],
    "adaptive_binning": [True],
    "L": [1, 2],
    "topK": [50, 100],
    "feature_mode": ["original_plus_patterns"],
    "G": [0.01, 0.001],
    "convert_binary_to_categorical": [False],
    "augmented_pair_transforms": [True],
    "interaction_relaxed_mining": [False],
}
```

#### Interaction-relaxed mining

```python
{
    "B": [-1],
    "adaptive_binning": [True],
    "L": [1, 2],
    "topK": [50, 100],
    "feature_mode": ["patterns_only"],
    "G": [0.01, 0.001],
    "interaction_relaxed_mining": [True],
    "convert_binary_to_categorical": [True],
    "augmented_pair_transforms": [False],
}
```

#### Higher-order interpretability grid (`interpretability_ho`)

The Interaction-relaxed option uses the pattern-only `interpretability_ho` grid. It does not generate augmented pairs, treats numeric 0/1 indicators categorically, and lets inner CV select between the built-in LR path and sequential RPTE with `enable_lookahead=False`. Survivor admission covers the root and immediate first-child positions of the initial branch, allowing weak-marginal pair evidence to be evaluated without recursively relaxing deeper positions.

```python
{
    "B": [-1],
    "adaptive_binning": [True],
    "L": [1, 2],
    "topK": [50, 100],
    "feature_mode": ["patterns_only"],
    "G": [0.01, 0.001],
    "interaction_relaxed_mining": [True],
    "convert_binary_to_categorical": [True],
    "augmented_pair_transforms": [False],
    "topk_budget_strict": [False],
    "base_estimator": [
        None,
        OneVsRestClassifier(
            LeafWiseBoundedLookaheadRPTEFeatureLR(
                leaf_config="3xD",
                depth=4,
                enable_lookahead=False,
            ),
            n_jobs=1,
        ),
    ],
}
```

#### Higher-order performance grid (`performance_ho`)

The shared `hyperparameter_configs.py` module defines the higher-order hybrid grid as follows:

```python
{
    "B": [-1],
    "adaptive_binning": [True],
    "L": [1, 2],
    "topK": [50, 100],
    "feature_mode": ["original_plus_patterns"],
    "G": [0.01, 0.001],
    "convert_binary_to_categorical": [False],
    "augmented_pair_transforms": [True],
    "topk_budget_strict": [False],
    "base_estimator": [
        None,
        OneVsRestClassifier(
            LeafWiseBoundedLookaheadRPTEFeatureLR(
                leaf_config="3xD",
                depth=4,
                enable_lookahead="adaptive",
            ),
            n_jobs=1,
        ),
    ],
}
```

`base_estimator=None` selects HUGIML's built-in logistic-regression downstream branch. The second candidate selects the adaptive RPTE downstream branch with the single `3xD` leaf configuration. With the listed `L`, `topK`, and `G` values, the grid contains 8 logistic-regression candidates and 8 RPTE candidates, for 16 candidates in total.

`convert_binary_to_categorical=False` keeps binary-valued numeric columns eligible for augmented-pair transformations. `topk_budget_strict=False` retains the non-strict augmented-pair budget semantics. The cached HUGIML tuning path reuses compatible mining artifacts while evaluating both downstream estimator branches.

The script also sets `execution_mode="production"` and `n_jobs=1` for HUGIML.

### XGB standard

```python
{
    "n_estimators": [100, 200],
    "max_depth": [3, 4],
    "learning_rate": [0.03, 0.1],
    "min_child_weight": [1, 5],
}
```

This grid contains **16 candidates**.

### XGB complexity-budgeted

```python
{
    "n_estimators": [25, 50],
    "max_depth": [1, 2],
    "learning_rate": [0.03, 0.1],
    "min_child_weight": [1, 5],
}
```

This grid contains **16 candidates**. The largest configured structure is 50 depth-2 trees, giving a theoretical ceiling of 200 leaves.

### LightGBM standard

```python
{
    "n_estimators": [100, 200],
    "learning_rate": [0.03, 0.1],
    "num_leaves": [15, 31],
    "min_child_samples": [10, 20],
}
```

This grid contains **16 candidates**.

### LightGBM complexity-budgeted

```python
{
    "n_estimators": [25, 50],
    "num_leaves": [2, 4],
    "learning_rate": [0.03, 0.1],
    "min_child_samples": [10, 20],
}
```

This grid contains **16 candidates**. The largest configured structure is 50 trees with four leaves each, for a ceiling of 200 leaves.

### RandomForest standard

```python
{
    "n_estimators": [200, 400],
    "max_depth": [4, 8],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt", 0.5],
}
```

This grid contains **16 candidates**.

### RandomForest complexity-budgeted

```python
{
    "n_estimators": [25, 50],
    "max_leaf_nodes": [2, 4],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt", 0.5],
    "max_depth": [None],
}
```

This grid contains **16 candidates**. The largest configured structure is 50 trees with four leaves each, for a ceiling of 200 leaves.

### EBM

```python
{
    "learning_rate": [0.01, 0.05],
    "max_bins": [32, 64],
    "interactions": [0, 5],
    "max_rounds": [500],
}
```

The EBM grid contains **8 candidates**. The builder also sets:

```python
{
    "random_state": 42,
    "n_jobs": 1,
    "outer_bags": 4,
    "max_rounds": 500,
}
```

### RuleFit

```python
{
    "n_estimators": [50, 100],
    "max_rules": [50, 100],
    "tree_size": [5, 10],
}
```

The RuleFit grid contains **8 candidates**. The builder sets `alpha=None` explicitly so `max_rules` remains authoritative.

Complexity-budgeted XGBoost, LightGBM, and Random Forest candidates are structurally capped at **200 terminal leaves**. The four ensemble sizes represented are 50, 100, 100, and 200 maximum leaves before regularization effects. HUGIML reports its representation complexity separately using selected patterns and path-specific derived features.

## Run a full benchmark

From the repo root:

```bash
python experiments/benchmark/benchmark_dashboard.py --fresh
```

Equivalent explicit command:

```bash
python experiments/benchmark/benchmark_dashboard.py --fresh --n-splits 5 --inner-splits 3 --random-state 42
```

On Windows Command Prompt:

```bat
python experiments\benchmark\benchmark_dashboard.py --fresh --n-splits 5 --inner-splits 3 --random-state 42
```

`--fresh` recreates the local results directory before the run when it is not combined with `--assemble`.

By default, outputs are written to:

```text
repo/experiments/benchmark/results/
```

## Resume a run

```bash
python experiments/benchmark/benchmark_dashboard.py --resume
```

This skips dataset/model/scenario pairs already present in the checkpoint.

## Run a subset

```bash
python experiments/benchmark/benchmark_dashboard.py \
  --datasets InterestInflationHighRate,SynthMixedMissing \
  --models HUGIML,"XGB standard" \
  --n-splits 5 \
  --inner-splits 3 \
  --random-state 42
```

On Windows Command Prompt:

```bat
python experiments\benchmark\benchmark_dashboard.py ^
  --datasets InterestInflationHighRate,SynthMixedMissing ^
  --models HUGIML,"XGB standard" ^
  --n-splits 5 ^
  --inner-splits 3 ^
  --random-state 42
```

## Assemble the dashboard

After a complete run, assemble with:

```bash
python experiments/benchmark/benchmark_dashboard.py --assemble
```

To write the privacy-sanitized environment manifest as a separate JSON file:

```bash
python experiments/benchmark/benchmark_dashboard.py --assemble --include-sbom
```

Run and assemble as separate steps:

```bash
python experiments/benchmark/benchmark_dashboard.py --fresh
python experiments/benchmark/benchmark_dashboard.py --assemble
```

Run, assemble, and write the environment manifest as separate steps:

```bash
python experiments/benchmark/benchmark_dashboard.py --fresh --n-splits 5 --inner-splits 3 --random-state 42
python experiments/benchmark/benchmark_dashboard.py --assemble --include-sbom
```

The script automatically searches upward from `experiments/benchmark/` for:

```text
hugiml_benchmark_analysis_dashboard_100.html
```

So if the template is in the repo root, no template argument is needed.

To override the template:

```bash
python experiments/benchmark/benchmark_dashboard.py \
  --assemble \
  --template-html path/to/hugiml_benchmark_analysis_dashboard_100.html
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
hugiml_benchmark_analysis_dashboard.html
```

When `--include-sbom` is supplied during assembly, the results directory also contains:

```text
benchmark_reproducibility_sbom.json
```

The HTML dashboard includes:

- Augmented pair and interaction-relaxed HUGIML path options in one selector.
- A collapsed methodology section at the end with the nested-CV protocol, preprocessing boundaries, timing definitions, constant estimator settings, and every parameter value considered for HUGIML and the baselines.
- Overall model summary.
- Real-world and synthetic summary tables.
- Pairwise model comparisons.
- Friedman and Wilcoxon tests across dataset-level outer-CV aggregates.
- Complexity vs performance chart.
- Dataset column profile table with dtype, missingness, uniqueness, and summary statistics.

## Useful options

```text
--fresh                         recreate the results directory before running
--resume                        skip completed dataset/model/scenario pairs
--assemble                      build dashboard files from the checkpoint
--include-sbom                  write a separate sanitized environment-manifest JSON; used with --assemble
--datasets A,B,C                run selected datasets
--models M1,M2                  run selected model labels
--start-pair N                  start at pair index N
--max-pairs N                   run at most N pairs
--row-cap N                     cap rows per dataset; default -1 uses the full dataset
--n-splits N                    outer StratifiedKFold split count; default 5
--inner-splits N                inner tuning StratifiedKFold split count; default 3
--random-state N                random seed for data generation and CV; default 42
--no-tune                       disable inner-CV hyperparameter tuning
--checkpoint PATH               custom checkpoint path
--out-dir PATH                  custom output directory
--template-html PATH            custom dashboard template path
--hugiml-max-fit-seconds FLOAT  optional HUGIML runtime cap
```

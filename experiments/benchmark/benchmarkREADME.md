# HUGIML benchmark dashboard

The script runs a reproducible classifier benchmark for HUGIML and baseline models, checkpoints every dataset/model pair, and assembles the HTML dashboard from the checkpoint. When requested at assemble time, it can also write a privacy-sanitized SBOM-style reproducibility manifest and embed the same manifest in a collapsed section at the bottom of the dashboard HTML.

The dashboard evaluation protocol is aligned with `hugiml.benchmarks.runner` when runner is invoked with tuning enabled:

```bash
python -m hugiml.benchmarks.runner --tune --n-splits 5 --inner-splits 3 --random-state 42
```

## Benchmark panel

The default panel contains 50 datasets:

- 30 public real-world datasets from scikit-learn and statsmodels.
- 20 synthetic datasets generated inside the script.

Synthetic datasets are deterministic and use `--random-state` plus dataset-specific offsets. The default random seed is `42`, matching `hugiml.benchmarks.runner`.

The real-world datasets are loaded from installed Python packages; no network access is required during a run. Statsmodels-backed datasets require `statsmodels` to be installed.

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

## Reproducibility SBOM

The dashboard can optionally include an SBOM-style reproducibility manifest during the assemble step:

```bash
python experiments/benchmark/benchmark_dashboard.py --assemble --include-sbom
```

`--include-sbom` is an assemble-time option. A fresh benchmark command such as:

```bash
python experiments/benchmark/benchmark_dashboard.py --fresh --n-splits 5 --inner-splits 3 --random-state 42
```

does not by itself generate the HTML dashboard or the SBOM. Use the two-step flow when you want a public dashboard with reproducibility metadata:

```bash
python experiments/benchmark/benchmark_dashboard.py --fresh --n-splits 5 --inner-splits 3 --random-state 42
python experiments/benchmark/benchmark_dashboard.py --assemble --include-sbom
```

When enabled, the assemble step writes:

```text
benchmark_reproducibility_sbom.json
hugiml_benchmark_analysis_dashboard_revised.html
```

The HTML dashboard contains the same SBOM JSON at the bottom inside a collapsed `<details>` block titled:

```text
Reproducibility / SBOM manifest
```

The SBOM is hidden by default in the HTML and can be expanded by the reader.

### SBOM contents

The manifest is intended to make another person able to understand and reproduce the benchmark environment. It includes best-effort metadata for:

- Benchmark configuration: datasets, model order, HUGIML scenarios, grids, row cap, random state, budget, split counts, and tuning mode where available.
- Dashboard inputs and outputs: checkpoint/template/output fingerprints and SHA-256 hashes.
- Python runtime: interpreter version, implementation, platform, executable label, and relevant environment variables.
- Python packages: installed distribution versions and `pip freeze --all`, with local paths sanitized.
- HUGIML package metadata: imported package location label, package version, native extension label, extension size, and extension SHA-256.
- Source fingerprints: SHA-256 hashes and sizes for relevant source files under locations such as `src/`, `native/`, `include/`, `cmake/`, and benchmark scripts.
- Git metadata: commit, branch, dirty status, short status, diff stat, and redacted remotes.
- Native/C++ build evidence: compiler/linker settings that are discoverable at runtime.

### C++ and native-extension reproducibility

C++ build information is included because HUGIML uses the native `_hugiml_core` extension. Native compiler, ABI, optimization flags, OpenMP runtime, dynamic linkage, and whether the package came from a wheel or local source build can affect benchmark performance.

The SBOM records native build evidence where discoverable:

```text
sysconfig compiler/linker variables
CC, CXX, CFLAGS, CXXFLAGS, CPPFLAGS, LDFLAGS, and related environment variables
compiler --version probes
_hugiml_core binary hash and size
ldd/readelf/otool linkage output when available
installed wheel/package metadata where available
native source fingerprints
```

This is best-effort metadata. If HUGIML was installed from a prebuilt wheel, the original compiler command line used by the wheel builder may not be fully recoverable from the installed package. In that case, the native extension hash, ABI tag, package metadata, runtime linkage, and source/git fingerprints are the main reproducibility anchors.

### SBOM path privacy

The SBOM is privacy-sanitized before it is written to both outputs:

```text
benchmark_reproducibility_sbom.json
the collapsed SBOM section in hugiml_benchmark_analysis_dashboard_revised.html
```

The same sanitized manifest is used for both files. It avoids exposing raw absolute local paths by using labels such as:

```text
repo:...
script_dir:...
out_dir:...
cwd:...
<redacted-path>/...
<redacted-remote>
```

The sanitizer also covers nested locations where paths commonly leak, including `sys.path`, `PYTHONPATH`, package install locations, HUGIML import paths, `_hugiml_core` paths, `ldd`/`readelf`/`otool` output, compiler probe output, `pip freeze --all`, and git remotes.

The raw benchmark artifacts are still local working files. For public sharing, prefer sharing the dashboard HTML and the SBOM JSON. Avoid publishing raw checkpoints or intermediate JSON/CSV files if they were produced by an older script version or if you have not inspected them for local-path content.

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

The dashboard evaluates HUGIML through scenario-specific grids. Baselines are shared across scenarios.

#### Augmented pair path

```python
{
    "B": [-1],
    "adaptive_binning": [True],
    "L": [1, 2],
    "topK": [50, 100],
    "feature_mode": ["original_plus_patterns"],
    "G": [0.01, 0.001],
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
    "augmented_pair_transforms": [False],
}
```

The script also sets `execution_mode="production"` and `n_jobs=1` for HUGIML.

### XGB standard

```python
{
    "n_estimators": [100, 200],
    "max_depth": [3, 4],
    "learning_rate": [0.03, 0.1],
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
    "max_depth": [4, 8],
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

### EBM

```python
{
    "learning_rate": [0.01, 0.05],
    "max_bins": [32, 64],
    "interactions": [0, 5],
}
```

The EBM builder sets these defaults unless already supplied by the grid:

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

Complexity-budgeted tree models are selected subject to a 100-leaf budget when leaf complexity is available. HUGIML reports its selected pattern count and uses the selected `topK` as its displayed budget.

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

To include the privacy-sanitized reproducibility SBOM in both JSON and HTML form:

```bash
python experiments/benchmark/benchmark_dashboard.py --assemble --include-sbom
```

Run and assemble as separate steps:

```bash
python experiments/benchmark/benchmark_dashboard.py --fresh
python experiments/benchmark/benchmark_dashboard.py --assemble
```

Run, assemble, and include the SBOM as separate steps:

```bash
python experiments/benchmark/benchmark_dashboard.py --fresh --n-splits 5 --inner-splits 3 --random-state 42
python experiments/benchmark/benchmark_dashboard.py --assemble --include-sbom
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

When `--include-sbom` is supplied during assembly, the results directory also contains:

```text
benchmark_reproducibility_sbom.json
```

The HTML dashboard also embeds the same sanitized SBOM in a collapsed section at the bottom.

The HTML dashboard includes:

- Overall model summary.
- Real-world and synthetic summary tables.
- Pairwise model comparisons.
- Friedman and Wilcoxon tests across dataset-level outer-CV aggregates.
- Complexity vs performance chart.
- Dataset column profile table with dtype, missingness, uniqueness, and summary statistics.
- Optional collapsed reproducibility/SBOM manifest when assembled with `--include-sbom`.

## Useful options

```text
--fresh                         recreate the results directory before running
--resume                        skip completed dataset/model/scenario pairs
--assemble                      build dashboard files from the checkpoint
--include-sbom                  include sanitized SBOM JSON and embed it in the HTML; used with --assemble
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

## Notes

- A complete dashboard assemble expects all requested dataset/model/scenario pairs to be present in the checkpoint.
- `--include-sbom` only affects assembly. It does not rerun benchmarks or recompute metrics.
- For public sharing, prefer the assembled HTML plus `benchmark_reproducibility_sbom.json`; inspect raw checkpoint/data artifacts before publishing them.
- For partial runs, inspect the checkpoint or CSVs directly.
- Do not assemble old holdout-protocol checkpoints together with nested-CV checkpoints.
- For comparable dashboard and runner outputs, keep `--n-splits`, `--inner-splits`, `--random-state`, and tuning settings aligned.

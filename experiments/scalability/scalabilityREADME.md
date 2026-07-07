# HUGIML Scalability Dashboard

Interactive benchmark dashboard comparing HUGIML model variants against XGBoost and LightGBM. The runner executes scalability tasks, writes a JSON checkpoint after each task, and assembles a self-contained HTML dashboard with visualizations, interactive filters, and data-driven insights from the checkpoint.

## Current model scenarios

The benchmark uses the following model set:

| Model key | Description |
|---|---|
| `hug_op_adaptive_full` | HUG original + patterns, full adaptive binning, default downstream LR solver |
| `hug_op_adaptive_saga` | HUG original + patterns, full adaptive binning, downstream LogisticRegression with `solver="saga"` |
| `hug_op_adaptive_sgd` | HUG original + patterns, full adaptive binning, downstream `SGDClassifier(loss="log_loss")` |
| `hug_op_adaptive_s20` | HUG original + patterns, adaptive binning sampled at 20% |
| `hug_po_adaptive_full` | HUG patterns only, full adaptive binning |
| `hug_po_adaptive_s20` | HUG patterns only, adaptive binning sampled at 20% |
| `xgb` | XGBoost baseline |
| `lgb` | LightGBM baseline |

The static `B` comparison is now handled through the `B` parameter sweep rather than as separate full-grid model scenarios. The saga and SGD rows isolate downstream solver scaling while keeping the same deterministic dataset seeds, train/test split, HUGIML mining defaults, `random_state=0`, and `max_iter=500` built-in estimator defaults used by the main classifier path.

## Datasets

The scalability benchmark uses two deterministic synthetic binary-classification datasets.

| Dataset | Purpose | Signal structure |
|---|---|---|
| `sparse_nonlinear` | Large-row, low-feature scalability | Mostly sparse signal in the first few features, with mild nonlinear terms and median-balanced labels |
| `threshold_grid` | Threshold, interaction, and higher-feature scalability | Threshold rules, local interactions, and median-balanced labels |

## Scaling grids

The runner uses predefined `(n, p)` task grids. Size caps filter these predefined tasks; they do not resize an individual task.

### n-scaling

| Dataset | Grid |
|---|---|
| `sparse_nonlinear` | `(10k,20)`, `(50k,20)`, `(100k,20)`, `(500k,20)`, `(1M,20)`, `(3M,20)`, `(5M,20)`, `(10M,20)`, `(50M,20)`, `(100M,20)`, `(500M,20)`, `(1B,20)` |
| `threshold_grid` | `(1k,200)`, `(5k,200)`, `(10k,200)`, `(50k,200)`, `(100k,200)`, `(300k,200)`, `(500k,200)`, `(1M,200)`, `(5M,200)`, `(10M,200)`, `(50M,200)`, `(100M,200)` |

### p-scaling

Both datasets use the same p-scaling grid. The maximum feature count remains `p=10,000`.

```text
(50k,20), (100k,100), (10k,1000), (5k,2000), (2500,4000), (2000,5000), (1000,10000)
```

## Size caps and staged runs

Use `--max-n` and `--max-p` to run only tasks whose configured `n` and `p` are within the selected limits. For n-scaling, `threshold_grid` uses 10x as many predictors as `sparse_nonlinear`, so its effective `--max-n` cap is 10% of the user-supplied value. For example, `--max-n 10M` selects `sparse_nonlinear` n-scaling tasks up to 10M rows and `threshold_grid` n-scaling tasks up to 1M rows. p-scaling and parameter-sweep tasks use the user-supplied `--max-n` value directly.

Supported count formats:

```text
10000
10k
1M
50M
100Mn
500M
1B
1Bn
```

Examples:

```bash
python experiments/scalability/scalability_dashboard.py --fresh --max-n 1M --max-p 1000
```

Later, expand the same checkpoint with larger tasks:

```bash
python experiments/scalability/scalability_dashboard.py --resume --max-n 50M --max-p 10000
```

Then expand again up to the largest n-scaling points:

```bash
python experiments/scalability/scalability_dashboard.py --resume --max-n 1B --max-p 10000
```

Task keys do not include the cap values. This is intentional: a small capped run records completed task keys, and a later larger capped run skips those completed tasks while adding newly eligible larger tasks.

Caps apply to n-scaling, p-scaling, and parameter-sweep tasks. If a cap is lower than a sweep task's configured `n` or `p`, that sweep task is not selected for that run. The checkpoint metadata records the selected task summary, the active caps, the selected dataset/model/section filters, and the n-scaling cap fraction used for each dataset.

## Parameter sweeps

The original sweep experiments are preserved and run against `hug_op_adaptive_full` when they are within the selected size caps:

| Sweep | Values |
|---|---|
| `B` | `2, 3, 5, 7, 10, 15` |
| `G` | `0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05` |
| `topK` | `10, 20, 30, 50, 100, 200` |
| `L` | `1, 2` |
| `avf` | adaptive binning `True` / static `False` |

With no caps and sweeps enabled, the current grid contains:

```text
192 n-scaling tasks
112 p-scaling tasks
44 sweep tasks
348 total tasks
```

Use `--no-sweeps` to run only n/p scaling. Rows from retired model keys, such as earlier static `B=8` full-grid scenarios, are ignored during CSV and dashboard assembly so older checkpoints can be resumed without reintroducing obsolete scenarios.

## Memory metrics

The dashboard memory plots use:

```text
peak_process_tree_rss_mb
```

This is the maximum observed RSS of the worker process tree during the full task. It is the primary memory metric for classifier comparison because it captures the practical peak RAM footprint of the run.

Fit-window memory deltas are still logged and exported as diagnostics:

```text
fit_delta_from_before_fit_mb
fit_delta_from_after_data_mb
```

## CLI

From the repo root:

```bash
python experiments/scalability/scalability_dashboard.py --fresh --max-n 1M --max-p 1000
```

Resume with a larger cap:

```bash
python experiments/scalability/scalability_dashboard.py --resume --max-n 50M --max-p 10000
```

Assemble from an existing checkpoint without running benchmarks:

```bash
python experiments/scalability/scalability_dashboard.py --assemble --out-dir experiments/scalability/results
```

Assemble with the privacy-sanitized reproducibility/SBOM manifest embedded in the Methodology tab:

```bash
python experiments/scalability/scalability_dashboard.py --assemble --include-sbom --out-dir experiments/scalability/results
```

Optional output path:

```bash
python experiments/scalability/scalability_dashboard.py \
  --assemble \
  --out-dir experiments/scalability/results \
  --output-html experiments/scalability/results/dashboard.html
```

Supported relevant options:

```text
--assemble              Assemble HTML dashboard from checkpoint only
--include-sbom          Write scalability_reproducibility_sbom.json and embed the same sanitized manifest under Methodology
--out-dir               Output directory; default is ./experiments/scalability/results
--output-html           Custom output HTML path
--fresh                 Start fresh by recreating the output directory
--resume                Resume from existing checkpoint
--no-sweeps             Run only n/p scaling; skip parameter sweeps
--max-n VALUE           Select tasks with n <= VALUE; supports k, M, Mn, B, Bn endings
--max-p VALUE           Select tasks with p <= VALUE; supports k, M, Mn, B, Bn endings
--start-task N          Start at selected task index N after filtering
--max-tasks N           Run at most N selected tasks
--only-section NAME     Run one section: n_scaling, p_scaling, or parameter_sweep_* 
--only-dataset NAME     Run one dataset: sparse_nonlinear or threshold_grid
--only-model NAME       Run one model key from the model scenarios table
--task-timeout SECONDS  Timeout per task in seconds
--mem-limit-mb MB       Memory limit per task in MB; -1 disables the limit
```

## Common workflows

### Small smoke run

```bash
python experiments/scalability/scalability_dashboard.py --fresh \
  --only-dataset sparse_nonlinear \
  --only-model hug_op_adaptive_full \
  --max-n 100k \
  --max-p 1000 \
  --max-tasks 4
```

### Full p-scaling up to 10k features, moderate n-scaling

```bash
python experiments/scalability/scalability_dashboard.py --resume \
  --max-n 10M \
  --max-p 10000
```

### Largest predefined n-scaling tasks

```bash
python experiments/scalability/scalability_dashboard.py --resume \
  --only-section n_scaling \
  --max-n 1B \
  --max-p 10000
```

### Run only the highest-cap HUGIML scenario first

```bash
python experiments/scalability/scalability_dashboard.py --resume \
  --only-model hug_op_adaptive_full \
  --max-n 1B \
  --max-p 10000
```

## Dashboard features

The HTML dashboard is self-contained and requires no external dependencies or network access. It includes:

### Overview section

- Key findings on fit performance, accuracy ranges, scaling behavior, and model coverage
- KPIs for largest n, fastest fit, best AUC, and lowest peak memory
- Charts for n-scaling fit time, test AUC, and peak memory trends
- Snapshot at the latest completed n-scaling point

### n-scaling and p-scaling sections

- Interactive charts for fit time, test AUC, peak memory, and pattern count
- Ratio chart for p-scaling fit time relative to XGBoost
- Grouped tables by model
- Peak RSS displayed in GB

### Parameter sweep sections

- `B` sweep: fit time plus AUC/pattern views
- `G` sweep: AUC/pattern and fit-time views
- `topK` sweep: AUC, patterns, and fit-time views
- `L` comparison: L=1 vs L=2 metrics
- `avf` comparison: adaptive vs static binning

### Methodology section

- System info: Python, platform, CPU count, RAM, worker threads
- Model scenarios: family, feature mode, scenario label, and downstream LR solver where applicable
- Benchmark grid: datasets, n-scaling settings, p-scaling settings
- Sweep grids: parameter ranges and values
- Task limits metadata: `max_n`, `max_p`, selected dataset/model/section filters, selected task summary, and n-scaling cap fractions
- Optional collapsed reproducibility/SBOM manifest when assembled with `--include-sbom`

### Theme and responsiveness

- Light/dark mode toggle
- Dataset selector
- Responsive design for desktop, tablet, and mobile
- Embedded Chart.js visualizations

## Outputs

Assembly writes:

| File | Contents |
|---|---|
| `scalability_checkpoint.json` | Full task checkpoint with task metadata and results |
| `scalability_results_flat.csv` | Flat CSV with fit time, AUC, patterns, memory, and diagnostics |
| `hugiml_scalability_dashboard.html` | Self-contained HTML dashboard |
| `scalability_reproducibility_sbom.json` | Optional privacy-sanitized SBOM-style reproducibility manifest written when `--include-sbom` is supplied |


## Reproducibility/SBOM path privacy

`--include-sbom` captures an SBOM-style manifest with artifact hashes, source fingerprints, git metadata, Python runtime details, installed distributions, `pip freeze --all`, selected environment variables, HUGIML package metadata, and discoverable native-extension build/linkage metadata. The same sanitized manifest is written to `scalability_reproducibility_sbom.json` and embedded in the HTML Methodology tab inside a collapsed, expandable block.

The sanitizer avoids publishing raw absolute local paths by replacing the source root, working directory, and user home directory with labels such as `<source-root>`, `<output-dir>`, `<cwd>`, and `<home>`. It also redacts credential-like URL userinfo in remotes and package output. For public sharing, prefer the assembled HTML and SBOM JSON over raw checkpoints from older script versions.

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

Static `B` values are evaluated through the `B` parameter sweep. The saga and SGD rows isolate downstream solver scaling while keeping the same deterministic dataset seeds, train/test split, HUGIML mining defaults, `random_state=0`, and `max_iter=500` built-in estimator defaults used by the main classifier path.

## Datasets

The scalability benchmark uses two deterministic synthetic binary-classification datasets.

| Dataset | Purpose | Signal structure |
|---|---|---|
| `sparse_nonlinear` | Large-row, low-feature scalability | Mostly sparse signal in the first few features, with mild nonlinear terms and median-balanced labels |
| `threshold_grid` | Threshold, interaction, and higher-feature scalability | Threshold rules, local interactions, and median-balanced labels |

## Scaling grids

The tables list completed `(n, p)` measurements. The panel uses 24 threads and `lr_source_policy=standard`.

### n-scaling

| Dataset | Grid |
|---|---|
| `sparse_nonlinear` | `(10k,20)`, `(50k,20)`, `(100k,20)`, `(500k,20)`, `(1M,20)`, `(3M,20)`, `(5M,20)`, `(10M,20)`, `(15M,20)` |
| `threshold_grid` | `(1k,200)`, `(5k,200)`, `(10k,200)`, `(50k,200)`, `(100k,200)`, `(300k,200)`, `(500k,200)`, `(1M,200)`, `(3M,200)`, `(5M,200)` |

### p-scaling

Both datasets use the same p-scaling grid. The maximum feature count remains `p=10,000`.

```text
(50k,20), (100k,100), (10k,1000), (5k,2000), (2500,4000), (2000,5000), (1000,10000)
```

## Size caps and staged runs

Size caps select predefined tasks without resizing them. For n-scaling, `threshold_grid` uses 10% of the supplied `--max-n` cap. Run the completed grid with these two dataset selections:

```bash
python experiments/scalability/scalability_dashboard.py --resume --only-dataset sparse_nonlinear --max-n 15Mn --max-p 10000 --n-jobs 24 --lr-source-policy standard
python experiments/scalability/scalability_dashboard.py --resume --only-dataset threshold_grid --max-n 50Mn --max-p 10000 --n-jobs 24 --lr-source-policy standard
```

The threshold-grid cap of 50Mn selects at most 5 million rows. Both commands include predictor scaling and parameter sweeps. Use a separate output directory for a new experiment. Resume preserves completed tasks and requires matching execution settings; `--fresh` requires an empty directory.

## Parameter sweeps

The sweep experiments run against `hug_op_adaptive_full` when they are within the selected size caps:

| Sweep | Values |
|---|---|
| `B` | `2, 3, 5, 7, 10, 15` |
| `G` | `0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05` |
| `topK` | `10, 20, 30, 50, 100, 200` |
| `L` | `1, 2` |
| `avf` | adaptive binning `True` / static `False` |

The completed panel contains:

```text
152 n-scaling tasks
112 p-scaling tasks
44 sweep tasks
308 total tasks
```

Use `--no-sweeps` to run only n/p scaling. CSV and dashboard assembly include only model keys in the model scenarios table.

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

From the repo root. To reproduce the completed panel, add `--n-jobs 24 --lr-source-policy standard` to execution commands and keep these settings unchanged when resuming:

```bash
python experiments/scalability/scalability_dashboard.py --fresh --max-n 1M --max-p 1000
```

Resume with a larger cap:

```bash
python experiments/scalability/scalability_dashboard.py --resume --only-dataset sparse_nonlinear --max-n 15Mn --max-p 10000 --n-jobs 24 --lr-source-policy standard
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
--fresh                 Start a new run; requires an empty output directory
--resume                Continue a compatible checkpoint, or initialize an empty output directory
--plan                  Show selected tasks and capacity estimates without fitting or writing results
--n-jobs N              Common thread budget for all models; default 4
--lr-source-policy P    HUGIML policy: standard, main_effect, or strict
--no-sweeps             Run only n/p scaling; skip parameter sweeps
--max-n VALUE           Select tasks with n <= VALUE; supports k, M, Mn, B, Bn endings
--max-p VALUE           Select tasks with p <= VALUE; supports k, M, Mn, B, Bn endings
--start-task N          Start at selected task index N after filtering
--max-tasks N           Run at most N pending tasks after completed tasks are skipped
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

### Largest completed n-scaling point

```bash
python experiments/scalability/scalability_dashboard.py --resume \
  --only-section n_scaling \
  --only-dataset sparse_nonlinear \
  --max-n 15Mn \
  --max-p 10000
```

### Run only the highest-cap HUGIML scenario first

```bash
python experiments/scalability/scalability_dashboard.py --resume \
  --only-model hug_op_adaptive_full \
  --only-dataset sparse_nonlinear \
  --max-n 15Mn \
  --max-p 10000
```

## Dashboard features

The HTML dashboard embeds its results and loads Chart.js from a CDN. It includes:

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
- Chart.js visualizations

## Outputs

Assembly writes:

| File | Contents |
|---|---|
| `scalability_checkpoint.json` | Full task checkpoint with task metadata and results |
| `scalability_results_flat.csv` | Flat CSV with fit time, AUC, patterns, memory, and diagnostics |
| `hugiml_scalability_dashboard.html` | Self-contained HTML dashboard |
| `scalability_reproducibility_sbom.json` | Optional privacy-sanitized SBOM-style reproducibility manifest written when `--include-sbom` is supplied |


## Reproducibility/SBOM path privacy

`--include-sbom` captures artifact hashes, source fingerprints, Python runtime details, key dependency versions, selected threading environment variables, HUGIML package metadata, and discoverable native-extension build/linkage metadata. The same sanitized manifest is written to `scalability_reproducibility_sbom.json` and embedded in the HTML Methodology tab inside a collapsed, expandable block.

The manifest omits raw command output, local paths, `sys.path`, and full package inventories. Path placeholders such as `<path>` and `<output-dir>` identify redacted locations. Public result JSON and dashboard artifacts contain completed measurements without terminal diagnostics, timestamps, or local absolute paths.

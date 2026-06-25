# HUGIML Scalability Dashboard

Interactive benchmark dashboard comparing HUGIML model variants against XGBoost and LightGBM. The runner executes scalability tasks, writes a JSON checkpoint after each task, and assembles a self-contained HTML dashboard with professional visualizations, interactive filters, and data-driven insights from the checkpoint.

## Current model scenarios

The benchmark uses the following model set:

| Model key | Description |
|---|---|
| `hug_op_adaptive_full` | HUG original + patterns, full adaptive binning |
| `hug_op_adaptive_s20` | HUG original + patterns, adaptive binning sampled at 20% |
| `hug_op_b8` | HUG original + patterns, fixed `B=8` |
| `hug_po_adaptive_full` | HUG patterns only, full adaptive binning |
| `hug_po_adaptive_s20` | HUG patterns only, adaptive binning sampled at 20% |
| `hug_po_b8` | HUG patterns only, fixed `B=8` |
| `xgb` | XGBoost baseline |
| `lgb` | LightGBM baseline |

## Scaling grids

### n-scaling

| Dataset | Grid |
|---|---|
| `sparse_nonlinear` | `(10k,20)`, `(50k,20)`, `(100k,20)`, `(500k,20)`, `(1M,20)`, `(3M,20)`, `(5M,20)`, `(10M,20)` |
| `threshold_grid` | `(1k,200)`, `(5k,200)`, `(10k,200)`, `(50k,200)`, `(100k,200)`, `(300k,200)`, `(500k,200)`, `(1M,200)` |

### p-scaling

Both datasets use:

```text
(50k,20), (100k,100), (10k,1000), (5k,2000), (2500,4000), (2000,5000), (1000,10000)
```

## Parameter sweeps

The original sweep experiments are preserved and run against `hug_op_adaptive_full`:

| Sweep | Values |
|---|---|
| `B` | `2, 3, 5, 7, 10, 15` |
| `G` | `0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05` |
| `topK` | `10, 20, 30, 50, 100, 200` |
| `L` | `1, 2` |
| `avf` | adaptive binning `True` / fixed `False` |

With the current model set, there are 240 n/p scaling tasks plus 44 sweep tasks, for 284 tasks total when sweeps are enabled.

Use `--no-sweeps` to run only n/p scaling.

## Memory metrics

The dashboard memory plots now use:

```text
peak_process_tree_rss_mb
```

This is the maximum observed RSS of the worker process tree during the full task. It is the primary metric for classifier memory comparison because it captures the practical peak RAM footprint of the run.

Fit-window memory deltas are still logged/exported as diagnostics:

```text
fit_delta_from_before_fit_mb
fit_delta_from_after_data_mb
```

## CLI

The runner uses the following argument style:

```bash
python scalability_dashboard_improved.py --out-dir results --resume
```

To assemble from an existing checkpoint without running benchmarks:

```bash
python scalability_dashboard_improved.py --assemble --out-dir results
```

Optional output path:

```bash
python scalability_dashboard_improved.py --assemble --out-dir results --output-html results/dashboard.html
```

Supported relevant options:

```text
--assemble              # Assemble HTML dashboard from checkpoint only
--out-dir              # Output directory (default: ./scalability_results)
--output-html          # Custom output HTML path
--fresh                # Start fresh (remove old checkpoint)
--resume               # Resume from existing checkpoint
--no-sweeps            # Run only n/p scaling (skip parameter sweeps)
--start-task           # Start task index
--max-tasks            # Maximum number of tasks to run
--only-section         # Run specific section (n_scaling, p_scaling, parameter_sweep_*)
--only-dataset         # Run specific dataset (sparse_nonlinear, threshold_grid)
--only-model           # Run specific model (see model scenarios table)
--task-timeout         # Timeout per task in seconds
--mem-limit-mb         # Memory limit per task in MB
```

## Dashboard Features

The HTML dashboard is self-contained and requires no external dependencies or network access. It includes:

### Overview Section
- **Key findings**: Data-driven insights on fit performance, accuracy ranges, scaling invariance, and model coverage
- **KPIs**: Largest n, fastest fit, best AUC, and lowest peak memory
- **Charts**: n-scaling fit time, test AUC, and peak memory trends
- **Snapshot**: Latest results at maximum n with model comparison

### n-Scaling & p-Scaling Sections
- **Interactive charts**: Fit time (log scale), test AUC, peak memory, pattern count
- **Ratio chart**: p-scaling fit time relative to XGBoost baseline
- **Grouped tables**: Collapsible sections by model for compact, scannable layout
- **Memory precision**: Peak RSS displayed to 1 decimal place (e.g., 1.2 GB)

### Parameter Sweep Sections
Professional visualizations for each hyperparameter:

- **B sweep**: Fit time + AUC/patterns dual-axis chart
- **G sweep**: AUC/patterns sorted (descending) + fit time chart
- **topK sweep**: Triple-axis chart (AUC, patterns, fit time)
- **L comparison**: L=1 vs L=2 side-by-side metrics with decision guidance
- **avf comparison**: Adaptive vs Fixed binning with pros/cons

Parameter values remain responsive—all 5 sweeps populate dynamically from checkpoint data.

### Methodology Section
- **System info**: Python, platform, CPU count, RAM, worker threads
- **Model scenarios**: Filterable by family (HugOP, HugPO, XGBoost, LightGBM)
- **Benchmark grid**: Datasets, feature-count scaling, n-scaling settings
- **Sweep grids**: Parameter ranges with values filterable by dataset

### Theme & Responsiveness
- Light/dark mode toggle
- Dataset selector
- Responsive design (desktop, tablet, mobile)
- Embedded Chart.js for all visualizations

## Outputs

Assembly writes:

| File | Contents |
|---|---|
| `scalability_checkpoint.json` | Full task checkpoint with 284+ results |
| `scalability_results_flat.csv` | Flat CSV: fit time, AUC, patterns, memory, fit deltas |
| `hugiml_scalability_dashboard.html` | Self-contained HTML dashboard (170 KB) |

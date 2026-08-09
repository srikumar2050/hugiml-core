# PMLBmini offline benchmark

This workflow evaluates the canonical 44 binary classification datasets in
PMLBmini. Data are pinned to PMLB commit
`c936c9e8ca0e5995816a42e1e8c829b8bcd9a8c9`, and dataset order follows the
published suite.

## Files and protocol

- Dataset cache: `datasetsPmlbmini/`
- Results: `experiments/benchmark/pmlbmini/pmlbmini_results/`
- Checkpoint: `pmlbmini_checkpoint.json`
- Dashboard: `pmlbmini_benchmark_dashboard.html`
- Splits: three independently shuffled stratified three-fold partitions
- Default validation: nested cross-validation
- Optional validation: rotating train/validation/test folds

The rotating protocol uses fold `F_i` for testing, `F_(i+1) mod K` for
validation, and the remaining folds for training. The validation winner is
retained without a post-selection refit. HUGIML, XGBoost, LightGBM, Random
Forest, and logistic regression use 16 configurations. EBM and RuleFit use
eight configurations.

## Commands

Run commands from the repository root.

```bash
python experiments/benchmark/pmlbmini/download_pmlbmini_datasets.py
python experiments/benchmark/pmlbmini/run_pmlbmini_offline_benchmark.py --verify-cache
python experiments/benchmark/pmlbmini/run_pmlbmini_offline_benchmark.py --models hugiml,ensemble --validation-protocol rotating --resume
python experiments/benchmark/pmlbmini/run_pmlbmini_offline_benchmark.py --assemble --include-rpte-dashboard
```

Use `--smallest N` to select by row count, `--first N` to follow suite order,
and `--task-ids` or `--task-ids-file` for explicit selections. `--fresh`
starts a new checkpoint run. `--resume` retains successful compatible pairs and
continues incomplete pairs.

The assembled CSV, JSON, and HTML files report predictive metrics, split and
pair timing, complexity units, statistical comparisons, and optional RPTE
structure distributions.

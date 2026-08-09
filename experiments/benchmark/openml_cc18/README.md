# OpenML-CC18 offline benchmark

This workflow evaluates cached OpenML-CC18 tasks on their official train/test splits. It supports staged downloads, task and model selection, resumable execution, split-level metrics, complexity analysis, RPTE telemetry, and CSV, JSON, and HTML reports.

## Protocol

- Dataset cache: `datasetsOpenml/`
- Results: `experiments/benchmark/openml_cc18/openml_cc18_results/`
- Default validation: three-fold stratified model selection inside every official outer-training partition
- Primary score: ROC AUC for binary tasks and macro one-vs-rest ROC AUC for multiclass tasks
- Default task panel: every complete task in the local cache

HUGIML, XGBoost, LightGBM, Random Forest, and logistic regression use 16 configurations. EBM and RuleFit use eight. Preprocessing state is fitted only on training rows. Every model uses the same stored outer partitions.

## Download and verify

```bash
python experiments/benchmark/openml_cc18/download_openml_cc18_datasets.py --smallest 72
python experiments/benchmark/openml_cc18/download_openml_cc18_datasets.py --add 10
python experiments/benchmark/openml_cc18/run_openml_cc18_offline_benchmark.py --verify-cache
```

`--smallest N` defines a target panel and retains complete cached tasks. `--add N` adds the next `N` missing tasks. Feature data, encoded targets, metadata, checksums, and every official split are stored locally.

## Run and assemble

```bash
python experiments/benchmark/openml_cc18/run_openml_cc18_offline_benchmark.py --models hugiml,ensemble --resume
python experiments/benchmark/openml_cc18/run_openml_cc18_offline_benchmark.py --models all --resume
python experiments/benchmark/openml_cc18/run_openml_cc18_offline_benchmark.py --assemble --include-rpte-dashboard
```

Run commands from the repository root. Model families are `hugiml`, `ensemble`, and `interpretable`; individual aliases are `xgb`, `lightgbm`, `rf`, `lr`, `ebm`, and `rulefit`. Use `--task-ids`, `--task-ids-file`, `--smallest N`, or `--first N` to select tasks. `--max-pairs N` runs the next `N` incomplete task/model pairs.

`--resume` retains fully successful compatible pairs. Incomplete pairs retain successful splits and continue with remaining splits. Compatibility includes dataset hashes, stored split definitions, model grid, validation settings, random state, HUGIML scenario, and model-source identity.

The dashboard presents predictive metrics, tuning/fit/prediction time, statistical comparisons, model units, model-inspection units, instance-inspection units, and optional RPTE route and structure distributions.

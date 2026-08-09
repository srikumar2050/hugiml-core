# TabZilla offline benchmark

This workflow evaluates the 36 classification tasks in the TabZilla suite using the official OpenML partitions stored with each cached task.

## Protocol

- Dataset cache: `datasetsTabzilla/`
- Results: `experiments/benchmark/tabzilla/tabzilla_results/`
- Default validation: nested cross-validation
- Suite-aligned validation: rotating train/validation/test folds

For rotating validation, test fold `F_i` is paired with validation fold `F_(i+1) mod K`; all remaining folds train the candidate. The selected fitted candidate is retained without a post-selection refit. Selection uses validation ROC AUC when defined and balanced accuracy when the validation fold does not contain every class. Final reporting uses fold AUC when defined and pooled out-of-fold predictions for affected multiclass tasks.

HUGIML, XGBoost, LightGBM, Random Forest, and logistic regression use 16 configurations. EBM and RuleFit use eight. XGBoost and LightGBM monitor rotating-validation log loss with a maximum of 200 trees and patience 20; Random Forest selects 100 or 200 trees.

## Commands

```bash
python experiments/benchmark/tabzilla/download_tabzilla_datasets.py
python experiments/benchmark/tabzilla/run_tabzilla_offline_benchmark.py --verify-cache
python experiments/benchmark/tabzilla/run_tabzilla_offline_benchmark.py --models hugiml,ensemble --validation-protocol rotating --resume
python experiments/benchmark/tabzilla/run_tabzilla_offline_benchmark.py --assemble --include-rpte-dashboard
```

Run commands from the repository root. Use `--smallest N` to select by row count, `--first N` for suite order, and `--task-ids` or `--task-ids-file` for explicit selections. `--defer-task-ids` moves tasks to the end; `--max-pairs N` limits an invocation to the next `N` incomplete pairs. `--resume` preserves successful compatible results.

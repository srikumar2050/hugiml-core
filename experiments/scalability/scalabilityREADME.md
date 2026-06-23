# HUGIML scalability experiment

The dashboard template is discovered automatically from the grandparent folder of the script, so the command below works when the template is in the repo root. A custom template can still be supplied with `--template-html`.

## Quick start

Run the full experiment from the repository root:

```bash
python experiments/scalability/scalability_dashboard.py --fresh
```

Assemble the dashboard from an existing checkpoint:

```bash
python experiments/scalability/scalability_dashboard.py --assemble
```

All generated files are written under:

```text
experiments/scalability/results/
```

Main outputs:

```text
results/scalability_checkpoint.json
results/scalability_dashboard_data.json
results/scalability_details.csv
results/scalability_summary.csv
results/hugiml_scalability_dashboard.html
```

## Experiment design

Two synthetic binary-classification datasets are generated inside the script using a reproducible seed.

`sparse_nonlinear` uses dense float32 Gaussian features with signal concentrated in the first columns and remaining columns acting mostly as noise. The target is median-binarised from a nonlinear score.

`threshold_grid` uses dense float32 Gaussian features with threshold and local-interaction signal terms. The target is median-binarised from the generated score.

The default random seed is `42`. The dataset generators derive deterministic sub-seeds from `n`, `p`, and the user seed, so repeated runs reproduce the same data for the same task.

## Evaluation protocol

Each task uses a single stratified holdout split:

```text
75% train / 25% test
```

The split seed is `42`. The dashboard reports test ROC AUC. The details CSV also records phase timings:

```text
data_s, split_s, fit_s, predict_s, auc_s
```

`fit_s` measures only `clf.fit(X_train, y_train)`. `predict_s` measures `predict_proba(X_test)`, and `auc_s` measures ROC-AUC calculation on the test predictions.

No cross-validation is used in this scalability experiment. The goal is to measure training time, memory growth, prediction time, and test AUC under controlled increases in `n` and `p`, not to tune hyperparameters.

## Models

The displayed models are:

```text
hug_op   HUGIML original_plus_patterns
hug_po   HUGIML patterns_only
xgb      XGBoost
lgb      LightGBM
```

## Model configurations

Shared HUGIML settings:

```python
B = -1
b_candidates = [3,5,7,10]
adaptive_binning=True
L=1
G=0.01
topK=50
n_jobs=4
use_hotpath=True
augmented_pair_transforms=False
interaction_relaxed_mining=False
execution_mode="production"
```

HUGIML mode settings:

```python
hug_op:
  feature_mode="original_plus_patterns"
  topk_budget_strict=True

hug_po:
  feature_mode="patterns_only"
  topk_budget_strict=False
```

The `avf` sweep still compares adaptive binning against fixed binning. When adaptive binning is enabled only for that sweep, `b_candidates=[3, 5, 7, 10]`.

XGBoost settings:

```python
n_estimators=50
max_depth=4
learning_rate=0.1
subsample=0.85
colsample_bytree=0.85
tree_method="hist"
n_jobs=4
eval_metric="logloss"
random_state=42
```

LightGBM settings:

```python
n_estimators=50
max_depth=4
num_leaves=15
learning_rate=0.1
subsample=0.85
colsample_bytree=0.85
n_jobs=4
random_state=42
verbose=-1
```

## Parameter sweeps

Parameter sweeps run HUGIML `hug_op` only.

For `sparse_nonlinear`:

```python
B:    n=50_000, p=20, values=[2, 3, 5, 7, 10, 15]
G:    n=50_000, p=20, values=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
topK: n=50_000, p=20, values=[10, 20, 30, 50, 100, 200]
L:    n=30_000, p=20, values=[1, 2]
avf:  n=50_000, p=20, values=[True, False]
```

For `threshold_grid`:

```python
B:    n=30_000, p=200, values=[2, 3, 5, 7, 10, 15]
G:    n=30_000, p=200, values=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
topK: n=30_000, p=200, values=[10, 20, 30, 50, 100, 200]
L:    n=20_000, p=100, values=[1, 2]
avf:  n=30_000, p=200, values=[True, False]
```

## Useful run controls

Run a subset:

```bash
python experiments/scalability/scalability_dashboard.py --fresh --only-dataset sparse_nonlinear --only-section n_scaling --only-model hug_po
```

Skip successful tasks already present in the checkpoint:

```bash
python experiments/scalability/scalability_dashboard.py --resume
```

Run a small batch from the task list:

```bash
python experiments/scalability/scalability_dashboard.py --fresh --no-sweeps --max-tasks 8
```

Runtime limit:

```bash
python experiments/scalability/scalability_dashboard.py --task-timeout 3600
```

`--task-timeout` is a per-task wall-clock limit in seconds. The default is `3600`, which is 60 minutes. Use `--task-timeout 0` to disable the time limit. If the limit is exceeded, the checkpoint row is marked `timeout`.

Memory limit:

```bash
python experiments/scalability/scalability_dashboard.py --mem-limit-mb -1
```

`--mem-limit-mb` controls the process-tree RSS limit. The default is `-1`, which uses 90% of detected system memory. Use `--mem-limit-mb 0` to disable the memory limit, or a positive value such as `65536` to set an explicit MB cap. If the limit is exceeded, the checkpoint row is marked `oom`.

# Internal benchmark dashboard

The internal workflow evaluates 50 real-world and 50 synthetic binary
classification datasets. It checkpoints each dataset/model pair and assembles
predictive, timing, statistical, robustness, complexity, and HUGIML structure
views into a self-contained HTML dashboard.

The [reproduction guide](REPRODUCING.md) covers the tested environment setup
and the complete internal and external benchmark workflows.

## Protocol

- Outer evaluation: five-fold shuffled stratified cross-validation
- Inner selection: three-fold stratified cross-validation
- Random seed: 42
- Selection metric: ROC AUC
- Preprocessing: fitted independently inside each training fold
- HUGIML search: 16 configurations
- XGBoost, LightGBM, Random Forest, and logistic regression: 16 configurations
- EBM and RuleFit: eight configurations

HUGIML receives native pandas columns. Baselines receive training-fitted median
numeric imputation, most-frequent categorical imputation, and one-hot encoding.
The selected estimator is fitted on the full outer-training fold and evaluated
on the held-out outer fold.

## Commands

```bash
python experiments/benchmark/benchmark_dashboard.py --fresh
python experiments/benchmark/benchmark_dashboard.py --resume
python experiments/benchmark/benchmark_dashboard.py --assemble
```

Use `--datasets` and `--models` for explicit panels, `--max-pairs N` for staged
execution, `--row-cap N` for a stratified row cap, and `--no-tune` for outer CV
without inner selection.

The checkpoint and assembled artifacts are written to
`experiments/benchmark/results/` unless `--out-dir` is supplied. The dashboard
includes augmented-pair and interaction-relaxed HUGIML selectors, route counts,
RPTE input/tree/leaf/direct-term distributions, model-inspection units, and
instance-inspection units.

The package runner can execute a matching CV design:

```bash
python -m hugiml.benchmarks.runner --tune --n-splits 5 --inner-splits 3 --random-state 42
```

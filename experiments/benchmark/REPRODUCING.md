# Reproducing the HUGIML benchmarks

This guide provides a common entry point for the internal and external HUGIML benchmark evaluations. The benchmark-specific README files describe dataset definitions, model grids, validation protocols, metrics, and output schemas in greater detail.

## Reference environment

The published evaluations use Python 3.12 on 64-bit Windows with fixed benchmark seeds and single-threaded model execution unless a benchmark states otherwise. The accompanying constraints file records a tested CPython 3.12 benchmark environment. Results produced by parallel numerical libraries or different hardware may differ slightly because of floating-point ordering. Comparisons should use the stored metric precision rather than require bit-for-bit identity.

The instructions are release-neutral. The repository installation reads the package version from the selected source checkout. When referring to a published artifact, `hugiml-core==x.y.z` denotes the release being evaluated; the checkout, installed package, and recorded benchmark metadata should identify the same `x.y.z` release.

From the repository root, create the environment with:

```powershell
python -m venv .venv-hugiml
.\.venv-hugiml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-hugiml\Scripts\python.exe -m pip install -c experiments\benchmark\constraints-benchmarks-py312.txt ".[benchmarks]"
.\.venv-hugiml\Scripts\python.exe -m pip check
```

The constraints are tested with CPython 3.12. Other supported Python versions can install HUGIML normally, but they may require platform-specific dependency resolution.

## Internal benchmark

The internal benchmark contains 50 real-world classification tasks built from standard Python-library datasets and 50 deterministic synthetic stress tests. It does not require a separate dataset-download stage. The benchmark uses five outer stratified folds, three inner stratified folds for model selection, and random seed 42 by default.

Run all 100 datasets and all models:

```powershell
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\benchmark_dashboard.py --datasets all --models all --fresh
```

Resume the current checkpoint without repeating completed dataset/model pairs:

```powershell
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\benchmark_dashboard.py --datasets all --models all --resume
```

Assemble the checkpoint into JSON, CSV, and HTML outputs:

```powershell
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\benchmark_dashboard.py --assemble --template-html hugiml_benchmark_analysis_dashboard.html
```

Use `--datasets`, `--models`, and `--max-pairs` for focused evaluations. The internal benchmark checkpoint is committed after each complete dataset/model pair, and fold checkpoints support continuation within a pair.

## External benchmarks

External benchmarks use local offline dataset folders containing the prepared data, targets, metadata, and official or deterministic split definitions. Each workflow has four stages:

1. Download the designated datasets and split definitions.
2. Verify the complete local dataset cache.
3. Execute or resume the benchmark.
4. Assemble the checkpoint into JSON, CSV, and HTML outputs.

Dataset and result locations are resolved relative to the repository. No absolute paths are required.

### OpenML-CC18

```powershell
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\openml_cc18\download_openml_cc18_datasets.py
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\openml_cc18\run_openml_cc18_offline_benchmark.py --verify-cache
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\openml_cc18\run_openml_cc18_offline_benchmark.py --models all --validation-protocol nested --fresh
```

See `openml_cc18/README.md` for the evaluated panel and nested-CV methodology.

### PMLBmini

```powershell
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\pmlbmini\download_pmlbmini_datasets.py
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\pmlbmini\run_pmlbmini_offline_benchmark.py --verify-cache
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\pmlbmini\run_pmlbmini_offline_benchmark.py --models all --validation-protocol rotating --fresh
```

See `pmlbmini/README.md` for the 44-dataset panel and rotating-validation methodology.

### TabZilla

```powershell
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\tabzilla\download_tabzilla_datasets.py
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\tabzilla\run_tabzilla_offline_benchmark.py --verify-cache
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\tabzilla\run_tabzilla_offline_benchmark.py --models all --validation-protocol rotating --fresh
```

See `tabzilla/README.md` for the 36-dataset panel, official folds, multiclass scoring, and pooled-AUC handling.

### TabArena

```powershell
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\tabarena\download_tabarena_datasets.py
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\tabarena\run_tabarena_offline_benchmark.py --verify-cache
.\.venv-hugiml\Scripts\python.exe experiments\benchmark\tabarena\run_tabarena_offline_benchmark.py --models all --validation-protocol tabarena --fresh
```

TabArena is substantially more computationally demanding. The packaged analysis currently covers 32 of the 38 classification datasets. It uses official outer splits, retained eight-fold inner ensembles, model-agnostic preprocessing fitted independently within each child, and official leaderboard comparisons. See `tabarena/README.md` for details.

## Resume behavior

`--fresh` starts a new evaluation and should be used only when a complete new run is intended.

For external benchmarks, continue a stored run with its identifier:

```powershell
.\.venv-hugiml\Scripts\python.exe RUNNER.py --models MODEL_SELECTION --validation-protocol PROTOCOL --resume --resume-run-id RUN_ID
```

The runner validates the stored task panel, models, grid, validation protocol, and tuning settings. Completed dataset/model pairs are preserved.

## Result assembly

Use the applicable runner with `--assemble`, `--run-id RUN_ID`, and `--include-rpte-dashboard` when RPTE distribution reporting is required. Assembly reads the checkpoint and does not fit models. The benchmark-specific README lists the generated filenames and dashboard options.

## Dataset integrity

Each external dataset folder contains a manifest describing task and dataset identifiers, dimensions, class labels, feature and target files, split files, and file hashes. Run `--verify-cache` before a new evaluation or after transferring a dataset folder between systems. Resolve any verification failure before fitting models.

## Reproducibility scope

Published checkpoint JSON files contain fold-level metrics, timing information, model-selection details, HUGIML route telemetry, RPTE structure fields, and complexity measurements. Dashboards are derived views of these artifacts. Exact wall-clock durations depend on hardware and system load, while predictive metrics should reproduce within normal numerical tolerance when the constrained environment, fixed seeds, stored datasets, and specified splits are used.


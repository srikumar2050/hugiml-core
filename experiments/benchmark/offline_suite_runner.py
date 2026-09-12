from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import benchmark_engine as engine


@dataclass(frozen=True)
class OfflineSuiteConfig:
    script_dir: Path
    project_root: Path
    suite: str
    benchmark_label: str
    suite_task_count: int
    datasets_dir_name: str
    download_script_name: str
    results_dir_name: str
    checkpoint_name: str
    dashboard_name: str
    output_prefix: str
    methodology_note: str | None = None
    nested_cv_ensemble_folds: int | None = None
    nested_early_stopping_default: bool = False
    tabarena_reporting: bool = False
    tabarena_official_results: Path | None = None
    tabarena_detailed_metrics: Path | None = None
    default_validation_protocol: str = "nested"
    default_n_jobs: int = 1
    default_outer_jobs: int = 1
    default_outer_jobs_fallback: int | None = None
    default_defer_task_ids: tuple[int, ...] = ()
    clean_output_json: bool = False


def configure_engine(config: OfflineSuiteConfig) -> None:
    """Apply suite-specific names and paths to the single benchmark engine."""
    engine.SCRIPT_DIR = config.script_dir.resolve()
    engine.PROJECT_ROOT = config.project_root.resolve()
    engine.DEFAULT_SUITE = config.suite
    engine.DEFAULT_DATASETS_DIR = engine.PROJECT_ROOT / config.datasets_dir_name
    engine.DOWNLOAD_SCRIPT = engine.SCRIPT_DIR / config.download_script_name
    engine.DEFAULT_RESULTS_DIR = config.results_dir_name
    engine.DEFAULT_CHECKPOINT_NAME = config.checkpoint_name
    engine.DEFAULT_DASHBOARD_NAME = config.dashboard_name
    engine.DEFAULT_OUTPUT_PREFIX = config.output_prefix
    engine.DEFAULT_BENCHMARK_LABEL = config.benchmark_label
    engine.DEFAULT_SUITE_TASK_COUNT = int(config.suite_task_count)
    engine.DEFAULT_METHODOLOGY_NOTE = config.methodology_note
    engine.DEFAULT_NESTED_CV_ENSEMBLE_FOLDS = config.nested_cv_ensemble_folds
    if config.nested_cv_ensemble_folds is not None:
        engine.DEFAULT_INNER_SPLITS = int(config.nested_cv_ensemble_folds)
    engine.DEFAULT_NESTED_EARLY_STOPPING = bool(config.nested_early_stopping_default)
    engine.DEFAULT_TABARENA_REPORTING = bool(config.tabarena_reporting)
    engine.DEFAULT_TABARENA_OFFICIAL_RESULTS = (
        None if config.tabarena_official_results is None else config.tabarena_official_results.resolve()
    )
    engine.DEFAULT_TABARENA_DETAILED_METRICS = (
        None if config.tabarena_detailed_metrics is None else config.tabarena_detailed_metrics.resolve()
    )
    engine.DEFAULT_VALIDATION_PROTOCOL = str(config.default_validation_protocol)
    engine.DEFAULT_WORKER_JOBS = int(config.default_n_jobs)
    engine.DEFAULT_OUTER_JOBS = int(config.default_outer_jobs)
    engine.DEFAULT_OUTER_JOBS_FALLBACK = config.default_outer_jobs_fallback
    engine.DEFAULT_DEFER_TASK_IDS = tuple(int(value) for value in config.default_defer_task_ids)
    engine.CLEAN_OUTPUT_JSON = bool(config.clean_output_json)


def run_suite(config: OfflineSuiteConfig, argv: Sequence[str] | None = None) -> int:
    configure_engine(config)
    return engine.main(argv)

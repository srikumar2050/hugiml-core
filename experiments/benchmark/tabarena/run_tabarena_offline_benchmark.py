from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from offline_suite_runner import OfflineSuiteConfig, run_suite  # noqa: E402

CONFIG = OfflineSuiteConfig(
    script_dir=SCRIPT_DIR,
    project_root=PROJECT_ROOT,
    suite="TabArena-v0.1-classification",
    benchmark_label="TabArena Classification",
    suite_task_count=38,
    datasets_dir_name="datasetsTabarena",
    download_script_name="download_tabarena_datasets.py",
    results_dir_name="tabarena_results",
    checkpoint_name="tabarena_checkpoint.json",
    dashboard_name="tabarena_benchmark_dashboard.html",
    output_prefix="tabarena",
    nested_cv_ensemble_folds=8,
    nested_early_stopping_default=True,
    tabarena_reporting=True,
    tabarena_official_results=(
        PROJECT_ROOT
        / "official_tabarena_results"
        / "tabarena_v0_1_classification_official_results.parquet"
    ),
    tabarena_detailed_metrics=(
        PROJECT_ROOT
        / "utilities"
        / "tabarena_official_metrics"
        / "output"
        / "tabarena_official_metrics.parquet"
    ),
    default_validation_protocol="tabarena",
)


def main(argv: list[str] | None = None) -> int:
    return run_suite(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())

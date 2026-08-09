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
    suite="PMLBmini",
    benchmark_label="PMLBmini",
    suite_task_count=44,
    datasets_dir_name="datasetsPmlbmini",
    download_script_name="download_pmlbmini_datasets.py",
    results_dir_name="pmlbmini_results",
    checkpoint_name="pmlbmini_checkpoint.json",
    dashboard_name="pmlbmini_benchmark_dashboard.html",
    output_prefix="pmlbmini",
)


def main(argv: list[str] | None = None) -> int:
    return run_suite(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())

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
    suite="TabZilla",
    benchmark_label="TabZilla",
    suite_task_count=36,
    datasets_dir_name="datasetsTabzilla",
    download_script_name="download_tabzilla_datasets.py",
    results_dir_name="tabzilla_results",
    checkpoint_name="tabzilla_checkpoint.json",
    dashboard_name="tabzilla_benchmark_dashboard.html",
    output_prefix="tabzilla",
    methodology_note=(
        "Multiclass ROC-AUC uses macro one-vs-rest on each official test fold and is "
        "averaged across folds when every fold is defined. If rare classes are absent "
        "from one or more test folds, ROC-AUC is calculated from pooled out-of-fold "
        "predictions and recorded as pooled_oof_fallback. If an outer training split "
        "has fewer than two samples in a class, stratified inner CV is impossible; "
        "that split is fitted without tuning and recorded as an inner-tuning fallback."
        " For rotating validation, if a class occurs in the test and validation folds "
        "but not in the remaining training folds, model selection is undefined; the "
        "complete outer-training partition is fitted without validation selection, "
        "and the split remains available for pooled out-of-fold metrics."
    ),
)


def main(argv: list[str] | None = None) -> int:
    return run_suite(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip(
    "pyarrow",
    reason="Parquet comparison tests use the benchmark dependency group",
)

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "experiments" / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from tabarena_protocol import (  # noqa: E402
    detailed_metric_comparison,
    official_reference_comparison,
)


def test_rpte_presentation_uses_only_recorded_branches_and_statistics():
    from benchmark_engine import tabarena_sections
    data = {"pair_results": [{"model": "HUGIML", "dataset": "example",
            "best_params_by_split_json": [
                {"best_params": {"base_estimator": None}},
                {"best_params": {"base_estimator": "RPTE()"}}]}],
            "hugiml_rpte_distributions": [{"dataset": "example",
                "rpte_active_tree_count_n": 1, "rpte_active_tree_count_median": 3,
                "rpte_active_tree_count_q25": 3, "rpte_active_tree_count_q75": 3}]}
    _, section = tabarena_sections(data)
    assert "LR 1" in section and "RPTE 1" in section
    assert section.count("<table>") == 1 and "<details>" not in section
    assert "Other" not in section and "other" not in section
    assert "Active trees" in section and "3 [3, 3]" in section
    assert "Active leaves" not in section and "path length" not in section


def test_rpte_presentation_omits_unrecorded_selection():
    from benchmark_engine import tabarena_sections
    _, section = tabarena_sections({"pair_results": [{"model": "HUGIML",
        "dataset": "example", "best_params_by_split_json": [{"best_params": {}}]}]})
    assert section == ""


def test_official_reference_comparison_matches_complete_outer_splits(tmp_path: Path) -> None:
    official_path = tmp_path / "official.parquet"
    pd.DataFrame(
        [
            {"dataset": dataset, "fold": fold, "method": method, "metric_error": error}
            for dataset, method, values in [
                ("binary", "A (default)", [0.20, 0.30]),
                ("binary", "RF (default)", [0.30, 0.40]),
                ("multi", "A (default)", [0.50, 0.60]),
                ("multi", "RF (default)", [0.70, 0.80]),
            ]
            for fold, error in enumerate(values)
        ]
    ).to_parquet(official_path, index=False)
    rows = [
        {
            "dataset": "binary",
            "model": "HUGIML",
            "status": "ok",
            "n_classes": 2,
            "official_splits_expected": 2,
            "official_splits_successful": 2,
            "split_rows": [
                {"split_id": 0, "status": "ok", "roc_auc": 0.85},
                {"split_id": 1, "status": "ok", "roc_auc": 0.75},
            ],
        },
        {
            "dataset": "multi",
            "model": "HUGIML",
            "status": "ok",
            "n_classes": 3,
            "official_splits_expected": 2,
            "official_splits_successful": 2,
            "split_rows": [
                {"split_id": 0, "status": "ok", "log_loss": 0.40},
                {"split_id": 1, "status": "ok", "log_loss": 0.50},
            ],
        },
        {
            "dataset": "partial",
            "model": "HUGIML",
            "status": "ok",
            "n_classes": 2,
            "official_splits_expected": 2,
            "official_splits_successful": 1,
            "split_rows": [{"split_id": 0, "status": "ok", "roc_auc": 0.8}],
        },
    ]

    result = official_reference_comparison(
        rows, official_path, regime="default", bootstrap_rounds=5
    )

    assert result["available"] is True
    assert result["matched_dataset_count"] == 2
    assert result["matched_datasets"] == ["binary", "multi"]
    assert [row["method"] for row in result["leaderboard"]].count("HUGIML") == 1
    assert result["elo_reference_method"] == "RF (default)"
    against_b = next(
        row for row in result["hugiml_pairwise"] if row["method"] == "RF (default)"
    )
    assert (against_b["wins"], against_b["ties"], against_b["losses"]) == (2, 0, 0)

    binary = official_reference_comparison(
        rows,
        official_path,
        regime="default",
        problem_scope="binary",
        bootstrap_rounds=5,
    )
    assert binary["matched_datasets"] == ["binary"]
    assert binary["problem_scope"] == "binary"


def test_detailed_metric_comparison_uses_paired_dataset_means(tmp_path: Path) -> None:
    detailed_path = tmp_path / "detailed.parquet"
    pd.DataFrame(
        [
            {
                "dataset": "binary",
                "fold": fold,
                "source_model": "XGBoost",
                "regime": "tuned",
                "problem_type": "binary",
                "accuracy": auc - 0.05,
                "roc_auc": auc,
                "balanced_accuracy": auc - 0.1,
                "f1": auc - 0.2,
                "log_loss": 1.0 - auc,
            }
            for fold, auc in enumerate((0.8, 0.9))
        ]
    ).to_parquet(detailed_path, index=False)
    rows = [
        {
            "dataset": "binary",
            "model": "HUGIML",
            "status": "ok",
            "n_classes": 2,
            "official_splits_expected": 2,
            "official_splits_successful": 2,
            "split_rows": [
                {
                    "split_id": fold,
                    "status": "ok",
                    "accuracy": auc,
                    "roc_auc": auc,
                    "balanced_accuracy": auc - 0.1,
                    "f1": auc - 0.2,
                    "log_loss": 0.9 - auc,
                }
                for fold, auc in enumerate((0.85, 0.95))
            ],
        }
    ]

    result = detailed_metric_comparison(rows, detailed_path)["tuned_overall"]
    xgb = next(row for row in result if row["method"] == "XGBoost")
    assert xgb["dataset_count"] == 1
    assert xgb["roc_auc_mean"] == pytest.approx(0.85)
    assert xgb["roc_auc_hugiml_delta"] == pytest.approx(0.05)
    assert xgb["accuracy_hugiml_delta"] == pytest.approx(0.1)
    assert xgb["log_loss_hugiml_delta"] == pytest.approx(0.15)

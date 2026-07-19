# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = Path(__file__).parents[1] / "experiments" / "benchmark" / "benchmark_dashboard.py"
SPEC = importlib.util.spec_from_file_location("benchmark_dashboard", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark_dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_dashboard)


def test_fold_aggregation_pools_out_of_fold_instance_counts():
    fold_rows = [
        {
            "roc_auc": 0.8,
            "complexity": 20.0,
            "complexity_model_units": 10.0,
            "complexity_model_inspection_units": 20.0,
            "complexity_instance_inspection_units_n_samples": 2,
            "complexity_instance_inspection_units_sum": 7.0,
            "complexity_instance_inspection_units_sum_squares": 25.0,
            "complexity_instance_inspection_units_min": 3,
            "complexity_instance_inspection_units_max": 4,
            "complexity_instance_inspection_units_confidence_level": 0.95,
            "complexity_report_json": "{}",
        },
        {
            "roc_auc": 0.9,
            "complexity": 30.0,
            "complexity_model_units": 12.0,
            "complexity_model_inspection_units": 30.0,
            "complexity_instance_inspection_units_n_samples": 3,
            "complexity_instance_inspection_units_sum": 18.0,
            "complexity_instance_inspection_units_sum_squares": 110.0,
            "complexity_instance_inspection_units_min": 5,
            "complexity_instance_inspection_units_max": 7,
            "complexity_instance_inspection_units_confidence_level": 0.95,
            "complexity_report_json": "{}",
        },
    ]
    result = benchmark_dashboard._aggregate_fold_rows(fold_rows)
    values = np.asarray([3, 4, 5, 6, 7], dtype=float)

    assert result["complexity_model_units"] == pytest.approx(11.0)
    assert result["complexity_model_inspection_units"] == pytest.approx(25.0)
    assert result["complexity_instance_inspection_units_n_samples"] == 5
    assert result["complexity_instance_inspection_units_mean"] == pytest.approx(values.mean())
    assert result["complexity_instance_inspection_units_std"] == pytest.approx(
        values.std(ddof=1)
    )
    assert result["complexity_instance_inspection_units_min"] == 3
    assert result["complexity_instance_inspection_units_max"] == 7


def test_scope_summary_uses_equal_weight_dataset_means():
    rows = []
    for dataset, instance_mean in (("A", 2.0), ("B", 8.0)):
        for model_index, model in enumerate(benchmark_dashboard.MODEL_ORDER):
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_group": "Synthetic",
                    "raw_features": 4,
                    "model": model,
                    "auc": 0.9 - model_index * 0.001,
                    "f1": 0.8,
                    "accuracy": 0.8,
                    "fit_seconds": 1.0,
                    "pair_seconds": 2.0,
                    "complexity": 20.0,
                    "complexity_model_units": 10.0,
                    "complexity_model_inspection_units": 20.0,
                    "complexity_instance_inspection_units_mean": instance_mean
                    if model == "HUGIML"
                    else 4.0,
                }
            )
    summary = benchmark_dashboard._summary_for_scope(pd.DataFrame(rows), "Overall")
    hugiml = next(row for row in summary["rows"] if row["model"] == "HUGIML")

    assert hugiml["mean_instance_inspection_units"] == pytest.approx(5.0)
    assert hugiml["instance_inspection_n_datasets"] == 2
    assert hugiml["instance_inspection_ci_lower"] < 5.0
    assert hugiml["instance_inspection_ci_upper"] > 5.0

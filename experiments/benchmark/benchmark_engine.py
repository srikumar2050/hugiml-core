from __future__ import annotations

# ruff: noqa: E402
"""OpenML-CC18 benchmark runner and dashboard assembler.

This runner intentionally reuses the model grids and preprocessing policy from
``benchmark_dashboard.py`` while replacing its generated outer
``StratifiedKFold`` with the exact train/test indices stored in each OpenML
classification task.

Default experiment
------------------
* Suite: OpenML-CC18
* Datasets: every available task, with optional task-count or task-ID selection
* Models: HUGIML, tree ensembles, logistic regression, EBM, and RuleFit
* Outer evaluation: every OpenML-defined repeat/fold/sample split
* Inner selection: three-fold stratified CV on each official training partition
* Primary metric: ROC AUC (binary, or multiclass OVR macro)

The script records pair and official-split progress. Resume operations reuse only
successful results produced with the same run configuration.
"""

import argparse
import copy
import hashlib
import html
import importlib
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ENGINE_DIR = SCRIPT_DIR
SOURCE_ROOT = next(
    (p for p in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (p / "src" / "hugiml").exists()),
    None,
)
if SOURCE_ROOT is not None:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
    sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_dashboard as benchmark_base
import numpy as np
import pandas as pd
import tabarena_protocol
from scipy import sparse
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from scipy.stats import t as student_t
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, label_binarize

try:
    from lightgbm import early_stopping as lightgbm_early_stopping
except ImportError:  # pragma: no cover - optional benchmark dependency
    lightgbm_early_stopping = None

MODEL_ORDER = [
    "HUGIML",
    "XGB standard",
    "LightGBM standard",
    "RandomForest standard",
    "Logistic Regression",
    "EBM",
    "RuleFit",
]
MODEL_FAMILIES = {
    "ensemble": ["XGB standard", "LightGBM standard", "RandomForest standard"],
    "interpretable": ["Logistic Regression", "EBM", "RuleFit"],
    "hugiml": ["HUGIML"],
}
REUSABLE_BASELINE_MODELS = {
    "XGB standard",
    "LightGBM standard",
    "RandomForest standard",
    "Logistic Regression",
    "EBM",
    "RuleFit",
}
MODEL_ALIASES = {
    "hugiml": "HUGIML",
    "xgb": "XGB standard",
    "xgboost": "XGB standard",
    "xgb standard": "XGB standard",
    "lightgbm": "LightGBM standard",
    "lgbm": "LightGBM standard",
    "lightgbm standard": "LightGBM standard",
    "rf": "RandomForest standard",
    "randomforest": "RandomForest standard",
    "random forest": "RandomForest standard",
    "randomforest standard": "RandomForest standard",
    "lr": "Logistic Regression",
    "logistic": "Logistic Regression",
    "logistic regression": "Logistic Regression",
    "logisticregression": "Logistic Regression",
    "ebm": "EBM",
    "rulefit": "RuleFit",
    "rule fit": "RuleFit",
}
HUGIML_MODELS = ("HUGIML",)


def is_hugiml_model(model: str) -> bool:
    return str(model) in HUGIML_MODELS


DEFAULT_SUITE = "OpenML-CC18"
DEFAULT_SMALLEST: int | None = None
DEFAULT_INNER_SPLITS = 3
DEFAULT_RANDOM_STATE = 42
DEFAULT_HUGIML_SCENARIO = "augmented_pair"
DEFAULT_RESULTS_DIR = "openml_cc18_results"
DEFAULT_CHECKPOINT_NAME = "openml_cc18_checkpoint.json"
DEFAULT_DASHBOARD_NAME = "openml_cc18_benchmark_dashboard.html"
DEFAULT_OUTPUT_PREFIX = "openml_cc18"
DEFAULT_BENCHMARK_LABEL = "OpenML-CC18"
DEFAULT_SUITE_TASK_COUNT = 72
DEFAULT_METHODOLOGY_NOTE: str | None = None
DEFAULT_NESTED_CV_ENSEMBLE_FOLDS: int | None = None
DEFAULT_NESTED_EARLY_STOPPING = False
DEFAULT_TABARENA_REPORTING = False
DEFAULT_TABARENA_OFFICIAL_RESULTS: Path | None = None
DEFAULT_TABARENA_DETAILED_METRICS: Path | None = None
DEFAULT_VALIDATION_PROTOCOL = "nested"
OFFLINE_SELECTION_MODE = "size"
VALIDATION_PROTOCOLS = ("nested", "rotating", "tabarena")
BASELINE_EARLY_STOPPING_MAX_ESTIMATORS = 200
BASELINE_EARLY_STOPPING_PATIENCE = 20
ROTATING_BASELINE_GRIDS: dict[str, dict[str, list[Any]]] = {
    "XGBoost": {
        "max_depth": [3, 4],
        "learning_rate": [0.03, 0.1],
        "min_child_weight": [1, 5],
        "reg_lambda": [0.1, 1.0],
    },
    "LightGBM": {
        "learning_rate": [0.03, 0.1],
        "num_leaves": [15, 31],
        "min_child_samples": [10, 20],
        "reg_lambda": [0.0, 1.0],
    },
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [4, 8],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt", 0.5],
    },
}


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)


def _sanitize_json(value: Any) -> Any:
    """Recursively replace non-finite and third-party scalar values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_sanitize_json(item) for item in value.tolist()]
    if isinstance(value, (pd.Series, pd.Index)):
        return [_sanitize_json(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except Exception:
        pass
    return str(value)


def safe_jsonable(value: Any) -> Any:
    return _sanitize_json(value)


def _public_dashboard_json(value: Any) -> Any:
    """Remove execution-only metadata from embedded dashboard data."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if any(token in normalized for token in ("sha", "hash", "checksum")):
                continue
            if any(token in normalized for token in ("revision", "rerun", "history")):
                continue
            if normalized.endswith(("_at_utc", "_timestamp")):
                continue
            cleaned[str(key)] = _public_dashboard_json(item)
        return cleaned
    if isinstance(value, list):
        return [_public_dashboard_json(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(safe_jsonable(payload), indent=2, allow_nan=False)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    backup = path.with_suffix(path.suffix + ".bak")
    backup_temporary = backup.with_name(f".{backup.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        json.loads(temporary.read_text(encoding="utf-8"))
        if path.is_file():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(current, dict):
                    raise ValueError("top-level JSON value must be an object")
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            else:
                shutil.copy2(path, backup_temporary)
                os.replace(backup_temporary, backup)
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
        backup_temporary.unlink(missing_ok=True)


def read_json_with_backup(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    for candidate in (path, path.with_suffix(path.suffix + ".bak")):
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("top-level JSON value must be an object")
            return value
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    if errors:
        raise ValueError("Unable to read JSON data:\n  " + "\n  ".join(errors))
    raise FileNotFoundError(path)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hugiml_fit_core_sha256() -> str | None:
    """Fingerprint HUGIML sources that can change fitted benchmark models."""
    package_file = getattr(benchmark_base._hugiml_pkg, "__file__", None)
    if not package_file:
        return None
    package_dir = Path(package_file).resolve().parent
    source_names = (
        "classifier.py",
        "_classifier_binning.py",
        "_classifier_training.py",
        "_classifier_tuning.py",
        "compute_complexity.py",
        "hyperparameter_configs.py",
        "rpte_bounded_lookahead_leafwise.py",
    )
    digest = hashlib.sha256()
    for name in source_names:
        path = package_dir / name
        value = sha256_file(path)
        if value is None:
            return None
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_models(value: str | None) -> list[str]:
    selectors = parse_csv(value or "all") or ["all"]
    selected: set[str] = set()
    unknown: list[str] = []
    for selector in selectors:
        normalized = " ".join(selector.strip().lower().replace("_", " ").replace("-", " ").split())
        if normalized == "all":
            selected.update(MODEL_ORDER)
        elif normalized in MODEL_FAMILIES:
            selected.update(MODEL_FAMILIES[normalized])
        elif normalized in MODEL_ALIASES:
            selected.add(MODEL_ALIASES[normalized])
        else:
            canonical = next((model for model in MODEL_ORDER if model.lower() == normalized), None)
            if canonical is None:
                unknown.append(selector)
            else:
                selected.add(canonical)
    if unknown:
        families = ", ".join(sorted(MODEL_FAMILIES))
        aliases = ", ".join(["hugiml", "xgb", "lightgbm", "rf", "lr", "ebm", "rulefit"])
        raise ValueError(
            f"Unknown model selectors: {unknown}. Use all; families {families}; "
            f"or model aliases {aliases}."
        )
    return [model for model in MODEL_ORDER if model in selected]


def as_suite_identifier(value: str) -> str | int:
    stripped = value.strip()
    return int(stripped) if stripped.isdigit() else stripped


def retry_call(fn, *, retries: int, label: str):
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # OpenML raises transport and API-specific errors.
            last_error = exc
            if attempt >= retries:
                break
            delay = min(30.0, 2.0**attempt)
            print(
                f"{label} failed ({type(exc).__name__}: {exc}); retrying in {delay:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def mean_or_none(values: Iterable[Any]) -> float | None:
    nums = [v for value in values if (v := finite_or_none(value)) is not None]
    return None if not nums else float(np.mean(nums))


def std_or_none(values: Iterable[Any]) -> float | None:
    nums = [v for value in values if (v := finite_or_none(value)) is not None]
    return None if len(nums) < 2 else float(np.std(nums, ddof=1))


def summary_from_moments(
    n_samples: int,
    total: float,
    sum_squares: float,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    n = int(n_samples)
    if n <= 0:
        raise ValueError("n_samples must be positive")
    mean = float(total) / n
    if n == 1:
        std = 0.0
        standard_error = 0.0
        lower = mean
        upper = mean
    else:
        variance = max(0.0, (float(sum_squares) - n * mean * mean) / (n - 1))
        std = math.sqrt(variance)
        standard_error = std / math.sqrt(n)
        critical = float(student_t.ppf((1.0 + confidence_level) / 2.0, n - 1))
        margin = critical * standard_error
        lower = mean - margin
        upper = mean + margin
    return {
        "mean": mean,
        "std": std,
        "standard_error": standard_error,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": float(confidence_level),
        "n_samples": n,
        "sum": float(total),
        "sum_squares": float(sum_squares),
    }


def series_mean_ci(values: Iterable[Any], *, confidence_level: float = 0.95) -> dict[str, Any]:
    nums = [value for item in values if (value := finite_or_none(item)) is not None]
    if not nums:
        return {
            "mean": None,
            "ci_lower": None,
            "ci_upper": None,
            "confidence_level": float(confidence_level),
            "n_samples": 0,
        }
    array = np.asarray(nums, dtype=float)
    return summary_from_moments(
        len(array),
        float(array.sum()),
        float(np.square(array).sum()),
        confidence_level=confidence_level,
    )


def execution_workload_key(task: dict[str, Any]) -> tuple[Any, ...]:
    """Order datasets by an estimate of increasing evaluation work."""
    n_rows = int(task.get("n_rows", -1) or -1)
    n_features = int(task.get("n_features", -1) or -1)
    n_classes = int(task.get("n_classes", -1) or -1)
    task_id = int(task.get("task_id", -1) or -1)
    dimensions_available = n_rows > 0 and n_features > 0
    classes_available = n_classes > 0
    if dimensions_available and classes_available:
        row_overhead_feature_equivalent = 32
        multiclass_factor = max(1, n_classes - 1)
        estimated_work = n_rows * (n_features + row_overhead_feature_equivalent) * multiclass_factor
    else:
        estimated_work = math.inf
    return (
        not (dimensions_available and classes_available),
        estimated_work,
        n_rows if dimensions_available else math.inf,
        n_features if dimensions_available else math.inf,
        n_classes if classes_available else math.inf,
        task_id,
    )


# ---------------------------------------------------------------------------
# OpenML task discovery and data handling
# ---------------------------------------------------------------------------


def require_openml():
    try:
        import openml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "The OpenML benchmark requires the 'openml' package. Install with "
            "`python -m pip install 'hugiml-core[benchmarks]'`."
        ) from exc
    return openml


def configure_openml(cache_dir: Path | None) -> Any:
    openml = require_openml()
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(openml.config, "set_root_cache_directory"):
            openml.config.set_root_cache_directory(str(cache_dir))
        else:  # pragma: no cover - older openml-python
            openml.config.cache_directory = str(cache_dir)
    return openml


def _first_present(row: pd.Series | dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        try:
            value = row[name]
        except Exception:
            continue
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            return value
    return None


def _normalize_task_listing(frame: pd.DataFrame, task_ids: Sequence[int]) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    df = frame.copy()
    if "tid" not in df.columns:
        df["tid"] = df.index
    allowed = {int(x) for x in task_ids}
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        tid = _first_present(row, ["tid", "task_id"])
        if tid is None or int(tid) not in allowed:
            continue
        rows.append(
            {
                "task_id": int(tid),
                "dataset_id": int(_first_present(row, ["did", "data_id", "dataset_id"]) or -1),
                "dataset_name": str(
                    _first_present(row, ["name", "data_name"]) or f"task-{int(tid)}"
                ),
                "n_rows": int(
                    float(_first_present(row, ["NumberOfInstances", "number_instances"]) or -1)
                ),
                "n_features": int(
                    float(_first_present(row, ["NumberOfFeatures", "number_features"]) or -1)
                ),
                "n_classes": int(
                    float(_first_present(row, ["NumberOfClasses", "number_classes"]) or -1)
                ),
            }
        )
    return rows


def _task_metadata_from_object(task: Any) -> dict[str, Any]:
    dataset = task.get_dataset(download_qualities=True)
    qualities = getattr(dataset, "qualities", {}) or {}

    def q(name: str, default: int = -1) -> int:
        value = qualities.get(name, default)
        try:
            return int(float(value))
        except Exception:
            return default

    return {
        "task_id": int(getattr(task, "task_id", getattr(task, "id", -1))),
        "dataset_id": int(getattr(task, "dataset_id", getattr(dataset, "dataset_id", -1))),
        "dataset_name": str(
            getattr(dataset, "name", f"task-{getattr(task, 'task_id', 'unknown')}")
        ),
        "n_rows": q("NumberOfInstances"),
        "n_features": q("NumberOfFeatures"),
        "n_classes": q("NumberOfClasses"),
    }


def resolve_suite_tasks(
    suite_id: str | int,
    *,
    smallest: int | None,
    explicit_task_ids: Sequence[int],
    retries: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    openml = require_openml()
    suite = retry_call(
        lambda: openml.study.get_suite(suite_id), retries=retries, label=f"suite {suite_id}"
    )
    suite_task_ids = [int(x) for x in suite.tasks]
    selected_ids = list(dict.fromkeys(int(x) for x in explicit_task_ids))
    if selected_ids:
        not_in_suite = sorted(set(selected_ids).difference(suite_task_ids))
        if not_in_suite:
            raise ValueError(f"Task IDs not present in suite {suite_id}: {not_in_suite}")
        candidate_ids = selected_ids
    else:
        candidate_ids = suite_task_ids

    metadata: list[dict[str, Any]] = []
    try:
        listing = retry_call(
            lambda: openml.tasks.list_tasks(
                task_id=candidate_ids,
                output_format="dataframe",
            ),
            retries=retries,
            label="OpenML task metadata listing",
        )
        metadata = _normalize_task_listing(listing, candidate_ids)
    except Exception as exc:
        print(
            f"Bulk task metadata unavailable ({type(exc).__name__}: {exc}); falling back to task objects",
            file=sys.stderr,
            flush=True,
        )

    found = {int(r["task_id"]) for r in metadata}
    for task_id in candidate_ids:
        if task_id in found and next(r for r in metadata if r["task_id"] == task_id)["n_rows"] >= 0:
            continue
        task = retry_call(
            lambda task_id=task_id: openml.tasks.get_task(task_id, download_splits=False),
            retries=retries,
            label=f"task {task_id}",
        )
        fresh = _task_metadata_from_object(task)
        metadata = [r for r in metadata if int(r["task_id"]) != task_id]
        metadata.append(fresh)

    metadata.sort(
        key=lambda r: (
            int(r.get("n_rows", -1)) < 0,
            int(r.get("n_rows", -1)),
            int(r["task_id"]),
        )
    )
    if selected_ids:
        order = {tid: i for i, tid in enumerate(selected_ids)}
        metadata.sort(key=lambda r: order[int(r["task_id"])])
    elif smallest is not None:
        metadata = metadata[: int(smallest)]

    suite_meta = {
        "suite_id": int(getattr(suite, "id", -1))
        if str(getattr(suite, "id", "")).isdigit()
        else getattr(suite, "id", None),
        "suite_name": str(getattr(suite, "name", suite_id)),
        "suite_alias": str(suite_id),
        "suite_task_count": len(suite_task_ids),
    }
    return suite_meta, metadata


def load_task(
    task_id: int, *, retries: int
) -> tuple[Any, pd.DataFrame, np.ndarray, dict[str, Any]]:
    openml = require_openml()
    task = retry_call(
        lambda: openml.tasks.get_task(task_id, download_splits=True),
        retries=retries,
        label=f"task {task_id}",
    )
    dataset = task.get_dataset(download_qualities=True)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    X = pd.DataFrame(X).copy()
    if attribute_names and len(attribute_names) == X.shape[1]:
        X.columns = [str(c) for c in attribute_names]
    else:
        X.columns = [str(c) for c in X.columns]

    # OpenML column names are not guaranteed unique; pipelines require unique names.
    seen: dict[str, int] = {}
    unique_columns: list[str] = []
    for col in X.columns:
        count = seen.get(col, 0)
        unique_columns.append(col if count == 0 else f"{col}__{count}")
        seen[col] = count + 1
    X.columns = unique_columns

    y_series = pd.Series(y)
    if y_series.isna().any():
        raise ValueError(
            f"Task {task_id} contains missing target values; dropping rows would invalidate official split indices."
        )
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_series.astype(str)).astype(int)

    categorical_indicator = list(categorical_indicator or [False] * X.shape[1])
    for idx, col in enumerate(X.columns):
        marked = idx < len(categorical_indicator) and bool(categorical_indicator[idx])
        if (
            marked
            or not pd.api.types.is_numeric_dtype(X[col])
            or pd.api.types.is_bool_dtype(X[col])
        ):
            X[col] = X[col].astype("category")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = benchmark_base._force_writable_frame(X.reset_index(drop=True))

    repeats, folds, samples = task.get_split_dimensions()
    meta = _task_metadata_from_object(task)
    meta.update(
        {
            "target_name": str(task.target_name),
            "class_labels": [str(x) for x in encoder.classes_],
            "n_classes": int(len(encoder.classes_)),
            "n_rows": int(len(y_encoded)),
            "n_features": int(X.shape[1]),
            "categorical_features": benchmark_base._categorical_columns(X),
            "official_repeats": int(repeats),
            "official_folds": int(folds),
            "official_samples": int(samples),
            "estimation_procedure": str(
                getattr(task, "estimation_procedure_type", "") or "OpenML-defined"
            ),
        }
    )
    return task, X, y_encoded, meta


def official_splits(task: Any, *, max_splits: int | None = None) -> list[dict[str, Any]]:
    repeats, folds, samples = (int(x) for x in task.get_split_dimensions())
    out: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for fold in range(folds):
            for sample in range(samples):
                train_idx, test_idx = task.get_train_test_split_indices(
                    fold=fold,
                    repeat=repeat,
                    sample=sample,
                )
                out.append(
                    {
                        "split_id": len(out),
                        "repeat": repeat,
                        "fold": fold,
                        "sample": sample,
                        "train_idx": np.asarray(train_idx, dtype=np.int64),
                        "test_idx": np.asarray(test_idx, dtype=np.int64),
                    }
                )
                if max_splits is not None and len(out) >= int(max_splits):
                    return out
    return out


def validate_official_split(split: dict[str, Any], n_rows: int) -> None:
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(f"OpenML split {split['split_id']} has an empty train or test partition")
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError(f"OpenML split {split['split_id']} has overlapping train/test indices")
    if (
        train_idx.min() < 0
        or test_idx.min() < 0
        or train_idx.max() >= n_rows
        or test_idx.max() >= n_rows
    ):
        raise IndexError(f"OpenML split {split['split_id']} contains indices outside [0, {n_rows})")


# ---------------------------------------------------------------------------
# Model tuning, prediction, and metrics
# ---------------------------------------------------------------------------


def inner_scoring(y: np.ndarray) -> str:
    return "roc_auc" if np.unique(y).size == 2 else "roc_auc_ovr"


def _with_rpte_early_stopping(value: Any, enabled: bool) -> Any:
    """Clone an estimator value and toggle every nested RPTE switch it exposes."""
    try:
        fitted = clone(value)
        params = fitted.get_params(deep=True)
    except (TypeError, AttributeError):
        return value
    updates = {
        key: bool(enabled)
        for key in params
        if key == "early_stopping" or key.endswith("__early_stopping")
    }
    return fitted.set_params(**updates) if updates else fitted


def _configure_hugiml_candidates(
    candidates: Sequence[dict[str, Any]], *, early_stopping: bool
) -> list[dict[str, Any]]:
    configured: list[dict[str, Any]] = []
    for candidate in candidates:
        item = copy.deepcopy(candidate)
        if "base_estimator" in item:
            item["base_estimator"] = _with_rpte_early_stopping(
                item["base_estimator"], early_stopping
            )
        configured.append(item)
    return configured


class DenseArrayTransformer(BaseEstimator, TransformerMixin):
    """Convert preprocessed baseline features to a dense numeric array."""

    def fit(self, X: Any, y: Any = None) -> DenseArrayTransformer:
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = X.toarray() if sparse.issparse(X) else X
        return np.array(values, dtype=np.float32, order="C", copy=True)


class TabArenaPreprocessedEstimator:
    """Retain fold-local TabArena preprocessing with one fitted child model."""

    def __init__(
        self,
        feature_generator: Any,
        estimator: Any,
        canonicalize_categorical: bool = False,
    ):
        self.feature_generator_ = feature_generator
        self.estimator_ = estimator
        self.canonicalize_categorical_ = bool(canonicalize_categorical)
        self.classes_ = np.asarray(getattr(estimator, "classes_", np.array([], dtype=int)))

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self.feature_generator_.transform(X.copy())
        if self.canonicalize_categorical_:
            transformed = _canonicalize_tabarena_categorical_frame(transformed)
        return transformed

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict_proba(self.transform_features(X))

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.decision_function(self.transform_features(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict(self.transform_features(X))


def _tabarena_model_agnostic_transform(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_validation: pd.DataFrame,
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    """Fit AutoGluon's standard feature generator on child-training rows only."""
    dependency_dir = BENCHMARK_ENGINE_DIR / "tabarena" / "_dependencies"
    dependency_namespace = dependency_dir / "autogluon"
    if dependency_dir.exists():
        if str(dependency_dir) not in sys.path:
            sys.path.insert(0, str(dependency_dir))
        loaded_namespace = sys.modules.get("autogluon")
        namespace_path = getattr(loaded_namespace, "__path__", None)
        if namespace_path is not None and str(dependency_namespace) not in namespace_path:
            namespace_path.append(str(dependency_namespace))
        importlib.invalidate_caches()
    try:
        from autogluon.features import AutoMLPipelineFeatureGenerator
    except ImportError as exc:
        raise ImportError(
            "The TabArena validation protocol requires autogluon.features==1.5.0 "
            f"from {dependency_dir}. Import failed: {type(exc).__name__}: {exc}"
        ) from exc
    feature_generator = AutoMLPipelineFeatureGenerator(verbosity=0)
    transformed_fit = feature_generator.fit_transform(X_fit.copy(), y=pd.Series(y_fit))
    transformed_validation = feature_generator.transform(X_validation.copy())
    return feature_generator, transformed_fit, transformed_validation


def _canonicalize_tabarena_categorical_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Return stable string labels for categorical columns.

    AutoGluon's categorical output uses integral codes. A code can materialize
    as an integer in a complete batch and as an integral float in a batch that
    also contains missing values. Stable labels keep transaction items and
    downstream indicator names identical across child and outer partitions.
    Missing values remain missing.
    """
    categorical = [
        column for column in X.columns if isinstance(X[column].dtype, pd.CategoricalDtype)
    ]
    if not categorical:
        return X
    output = X.copy()

    def stable_label(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            if math.isfinite(numeric) and numeric.is_integer():
                return str(int(numeric))
        return str(value)

    for column in categorical:
        labels = [stable_label(value) for value in output[column].astype(object)]
        output[column] = pd.Series(labels, index=output.index, dtype="string").astype("category")
    return output


def baseline_pipeline(estimator: Any) -> Pipeline:
    """Build the shared baseline pipeline with array-only estimator input.

    Category values can become generated one-hot feature labels containing
    characters that some estimators reserve in their dataframe interfaces.
    Converting after preprocessing keeps the numerical matrix unchanged while
    preventing those generated labels from reaching the estimator.
    """

    wrapped = benchmark_base._wrap_non_hugiml_pipeline(estimator)
    steps = list(getattr(wrapped, "steps", []))
    if not steps or steps[-1][0] != "model":
        raise TypeError("Baseline preprocessing must return a Pipeline ending in a 'model' step")
    return Pipeline([*steps[:-1], ("array", DenseArrayTransformer()), steps[-1]])


def rotating_baseline_grid(model: str) -> dict[str, list[Any]]:
    """Return the baseline grid associated with rotating validation."""
    if model not in ROTATING_BASELINE_GRIDS:
        return benchmark_base.get_baseline_grid(model)
    return copy.deepcopy(ROTATING_BASELINE_GRIDS[model])


def rotating_hugiml_candidates(hugiml_scenario: str) -> list[dict[str, Any]]:
    """Return the package-standard HUGIML search space."""
    _, candidates = benchmark_base._hugiml_grid_for_scenario(hugiml_scenario)
    return copy.deepcopy(candidates)


def _fit_baseline_with_rotating_validation(
    model_name: str,
    estimator: Pipeline,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit one baseline candidate, using native stopping for boosted trees."""
    if model_name not in {"XGB standard", "LightGBM standard"}:
        estimator.fit(X_fit, y_fit)
        return estimator, {
            "baseline_early_stopping_used": False,
            "baseline_selected_n_estimators": int(
                estimator.named_steps["model"].get_params().get("n_estimators", 0)
            ),
        }

    preprocessing = Pipeline(estimator.steps[:-1])
    model = estimator.named_steps["model"]
    model.set_params(n_estimators=BASELINE_EARLY_STOPPING_MAX_ESTIMATORS)
    transformed_fit: Any = X_fit
    transformed_validation: Any = X_validation
    for _, transformer in preprocessing.steps:
        transformer.fit(transformed_fit, y_fit)
        transformed_fit = transformer.transform(transformed_fit)
        transformed_validation = transformer.transform(transformed_validation)
    fit_kwargs: dict[str, Any] = {"eval_set": [(transformed_validation, y_validation)]}

    if model_name == "XGB standard":
        stopping_metric = "logloss" if np.unique(y_fit).size == 2 else "mlogloss"
        model.set_params(eval_metric=stopping_metric)
        model.set_params(early_stopping_rounds=BASELINE_EARLY_STOPPING_PATIENCE)
        fit_kwargs["verbose"] = False
    else:
        if lightgbm_early_stopping is None:
            raise ImportError("lightgbm early_stopping callback is required")
        fit_kwargs["callbacks"] = [
            lightgbm_early_stopping(
                BASELINE_EARLY_STOPPING_PATIENCE,
                first_metric_only=True,
                verbose=False,
            )
        ]
        fit_kwargs["eval_metric"] = (
            "binary_logloss" if np.unique(y_fit).size == 2 else "multi_logloss"
        )
        stopping_metric = str(fit_kwargs["eval_metric"])

    model.fit(transformed_fit, y_fit, **fit_kwargs)
    if model_name == "XGB standard":
        selected = int(getattr(model, "best_iteration", -1)) + 1
    else:
        selected = int(getattr(model, "best_iteration_", 0) or 0)
    if selected <= 0:
        selected = BASELINE_EARLY_STOPPING_MAX_ESTIMATORS

    fitted = Pipeline([*preprocessing.steps, ("model", model)])
    return fitted, {
        "baseline_early_stopping_used": True,
        "baseline_selected_n_estimators": selected,
        "baseline_max_n_estimators": BASELINE_EARLY_STOPPING_MAX_ESTIMATORS,
        "baseline_early_stopping_patience": BASELINE_EARLY_STOPPING_PATIENCE,
        "baseline_early_stopping_metric": stopping_metric,
    }


def tune_baseline(
    candidates: list[dict[str, Any]],
    builder,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    inner_splits: int,
    random_state: int,
) -> tuple[Any, dict[str, Any], float, float, dict[str, Any]]:
    splits = benchmark_base._validated_stratified_splits(
        y_train, inner_splits, label="Inner OpenML training-partition CV"
    )
    grid = benchmark_base._candidate_grid_dict(candidates)
    pipeline = baseline_pipeline(builder({}))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        pipeline,
        benchmark_base._prefix_grid_for_wrapped_model(grid),
        scoring=inner_scoring(y_train),
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score=np.nan,
        return_train_score=False,
    )
    started = time.perf_counter()
    search.fit(X_train, y_train)
    tune_ms = (time.perf_counter() - started) * 1000.0
    final_refit_ms = float(search.refit_time_) * 1000.0 if hasattr(search, "refit_time_") else None
    params = benchmark_base._strip_model_prefix(dict(search.best_params_))
    return (
        search.best_estimator_,
        params,
        float(search.best_score_),
        float(tune_ms),
        {"_final_refit_ms": final_refit_ms, "inner_scoring": inner_scoring(y_train)},
    )


def _fit_candidates_without_refit(
    *,
    model_name: str,
    candidates: Sequence[dict[str, Any]],
    builder: Any,
    complexity_fn: Any,
    folds: Sequence[tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]],
) -> tuple[Any, dict[str, Any], float, float, dict[str, Any]]:
    """Select on explicit validation folds and retain a fitted winner."""
    started = time.perf_counter()
    best = None
    errors: list[str] = []
    selection_metrics: set[str] = set()
    for params in candidates:
        scores: list[float] = []
        fitted: list[tuple[float, Any, float, dict[str, Any]]] = []
        for X_fit, y_fit, X_val, y_val in folds:
            try:
                estimator = (
                    builder(params)
                    if is_hugiml_model(model_name)
                    else baseline_pipeline(builder(params))
                )
                fit_started = time.perf_counter()
                stopping_info: dict[str, Any] = {}
                if is_hugiml_model(model_name):
                    estimator.fit(X_fit, y_fit)
                else:
                    estimator, stopping_info = _fit_baseline_with_rotating_validation(
                        model_name,
                        estimator,
                        X_fit,
                        y_fit,
                        X_val,
                        y_val,
                    )
                fit_ms = (time.perf_counter() - fit_started) * 1000.0
                n_classes = int(np.unique(y_fit).size)
                proba = probability_matrix(
                    estimator,
                    X_val,
                    n_classes,
                    trained_class_labels=np.arange(n_classes, dtype=int),
                )
                metrics = classification_metrics(y_val, proba)
                score = finite_or_none(metrics.get("roc_auc"))
                selection_metric = "roc_auc"
                if score is None:
                    score = finite_or_none(metrics.get("balanced_accuracy"))
                    selection_metric = "balanced_accuracy_fallback"
                if score is None:
                    raise ValueError("Validation ROC-AUC and balanced accuracy are undefined.")
                selection_metrics.add(selection_metric)
                scores.append(float(score))
                fitted.append((float(score), estimator, fit_ms, stopping_info))
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if len(scores) != len(folds):
            continue
        retained = max(fitted, key=lambda item: item[0])
        complexity = complexity_fn(retained[1]) if complexity_fn else None
        item = (
            float(np.mean(scores)),
            -float(complexity or 0.0),
            retained[1],
            copy.deepcopy(params),
            retained[2],
            retained[3],
        )
        if best is None or item[:2] > best[:2]:
            best = item
    if best is None:
        raise RuntimeError("All validation candidates failed: " + " | ".join(errors[-3:]))
    return (
        best[2],
        best[3],
        best[0],
        (time.perf_counter() - started) * 1000.0,
        {
            "_final_refit_ms": best[4],
            "inner_scoring": inner_scoring(folds[0][1]),
            "selection_refit_performed": False,
            "retained_validation_fold_model": True,
            "candidate_error_count": len(errors),
            "validation_selection_metrics": sorted(selection_metrics),
            **best[5],
        },
    )


def _fit_hugiml_with_rotating_validation_cached(
    *,
    candidates: Sequence[dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
) -> tuple[Any, dict[str, Any], float, float, dict[str, Any]]:
    """Select HUGIML on one explicit validation fold with shared representations."""
    grid = benchmark_base._candidate_grid_dict(list(candidates))
    base_params: dict[str, Any] = {
        key: copy.deepcopy(values[0]) for key, values in grid.items() if len(values) == 1
    }
    base_params.setdefault("execution_mode", "production")
    base_params.setdefault("n_jobs", 1)
    selection_scoring = (
        "roc_auc"
        if set(np.unique(y_validation).tolist()) == set(np.unique(y_train).tolist())
        else "balanced_accuracy"
    )
    started = time.perf_counter()
    result = benchmark_base.HUGIMLClassifierNative.fast_grid_tune(
        X_train,
        y_train,
        X_validation,
        y_validation,
        param_grid=grid,
        base_params=base_params,
        scoring=selection_scoring,
        refit_full=False,
        return_results=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    best_params = dict(result.get("best_params") or {})
    best_row_seconds = None
    for row in result.get("cv_results") or []:
        params = dict(row.get("params") or {})
        if all(repr(params.get(key)) == repr(value) for key, value in best_params.items()):
            value = finite_or_none(row.get("elapsed_seconds"))
            if value is not None:
                best_row_seconds = value
                break
    return (
        result["best_model"],
        best_params,
        float(result["best_score"]),
        elapsed_ms,
        {
            "_final_refit_ms": (
                0.0 if best_row_seconds is None else float(best_row_seconds) * 1000.0
            ),
            "inner_scoring": selection_scoring,
            "selection_refit_performed": False,
            "retained_validation_fold_model": True,
            "validation_selection_metrics": [
                "roc_auc" if selection_scoring == "roc_auc" else "balanced_accuracy_fallback"
            ],
            "hugiml_fast_tune_used": True,
            "hugiml_fast_tune_method": result.get("method", "cached_validation"),
            "hugiml_fast_tune_cache_fit_seconds_json": json.dumps(
                result.get("cache_fit_seconds_by_G_L_topK", {}),
                sort_keys=True,
                default=json_default,
            ),
            "hugiml_fast_tune_validation_cache_entries": result.get("validation_cache_entries"),
            "hugiml_fast_tune_transaction_cache_entries": result.get("transaction_cache_entries"),
            "hugiml_fast_tune_equivalent_downstream_fit_reuses": result.get(
                "equivalent_downstream_fit_reuses", 0
            ),
            "hugiml_fast_tune_template_prepare_seconds_json": json.dumps(
                result.get("template_prepare_seconds", {}),
                sort_keys=True,
                default=json_default,
            ),
            "hugiml_fast_tune_validation_prepare_seconds_json": json.dumps(
                result.get("validation_prepare_seconds", {}),
                sort_keys=True,
                default=json_default,
            ),
            "hugiml_fast_tune_fingerprint_seconds_json": json.dumps(
                result.get("fingerprint_seconds", {}),
                sort_keys=True,
                default=json_default,
            ),
            "hugiml_fast_tune_downstream_fit_seconds": float(
                result.get("downstream_fit_seconds", 0.0)
            ),
            "hugiml_fast_tune_downstream_score_seconds": float(
                result.get("downstream_score_seconds", 0.0)
            ),
            "hugiml_fast_tune_downstream_template_count": int(
                result.get("downstream_template_count", 0)
            ),
            "hugiml_fast_tune_validation_structure_reuses": int(
                result.get("validation_structure_reuses", 0)
            ),
            "hugiml_fast_tune_original_feature_cache_stats_json": json.dumps(
                result.get("original_feature_cache_stats"),
                sort_keys=True,
                default=json_default,
            ),
        },
    )


def _fit_tabarena_cv_ensemble(
    *,
    model_name: str,
    candidates: Sequence[dict[str, Any]],
    builder: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
) -> tuple[Any, dict[str, Any], float, float, dict[str, Any]]:
    """Fit the selected configuration as TabArena's eight-child CV ensemble."""

    canonicalize_hugiml_categories = is_hugiml_model(model_name)

    def prepare_fold(X_fit, y_fit, X_validation, y_validation):
        feature_generator, transformed_fit, transformed_validation = (
            _tabarena_model_agnostic_transform(X_fit, y_fit, X_validation)
        )
        if canonicalize_hugiml_categories:
            transformed_fit = _canonicalize_tabarena_categorical_frame(transformed_fit)
            transformed_validation = _canonicalize_tabarena_categorical_frame(
                transformed_validation
            )
        return feature_generator, transformed_fit, transformed_validation

    def fit_child(
        params,
        X_fit,
        y_fit,
        X_validation,
        y_validation,
        prepared=None,
    ):
        if prepared is None:
            feature_generator, X_fit, X_validation = prepare_fold(
                X_fit, y_fit, X_validation, y_validation
            )
        else:
            feature_generator, X_fit, X_validation = prepared
        if is_hugiml_model(model_name):
            grid = benchmark_base._candidate_grid_dict([params])
            base_params = {
                key: copy.deepcopy(values[0]) for key, values in grid.items() if len(values) == 1
            }
            base_params.setdefault("execution_mode", "production")
            base_params.setdefault("n_jobs", 1)
            child_scoring = "roc_auc" if np.unique(y_fit).size == 2 else "neg_log_loss"
            result = benchmark_base.HUGIMLClassifierNative.fast_grid_tune(
                X_fit,
                y_fit,
                X_validation,
                y_validation,
                param_grid=grid,
                base_params=base_params,
                scoring=child_scoring,
                refit_full=False,
                return_results=False,
            )
            return TabArenaPreprocessedEstimator(
                feature_generator=feature_generator,
                estimator=result["best_model"],
                canonicalize_categorical=True,
            )
        estimator = baseline_pipeline(builder(params))
        if model_name in {"XGB standard", "LightGBM standard"}:
            estimator, _ = _fit_baseline_with_rotating_validation(
                model_name, estimator, X_fit, y_fit, X_validation, y_validation
            )
            return TabArenaPreprocessedEstimator(
                feature_generator=feature_generator,
                estimator=estimator,
            )
        estimator.fit(X_fit, y_fit)
        return TabArenaPreprocessedEstimator(
            feature_generator=feature_generator,
            estimator=estimator,
        )

    return tabarena_protocol.fit_cross_validated_ensemble(
        candidates=candidates,
        X=X_train,
        y=y_train,
        random_state=random_state,
        fit_child=fit_child,
        probability_fn=probability_matrix,
        n_splits=int(DEFAULT_NESTED_CV_ENSEMBLE_FOLDS or 8),
        prepare_fold=prepare_fold,
    )


def order_execution_tasks(
    tasks: Sequence[dict[str, Any]],
    *,
    models: Sequence[str],
    completed_pairs: set[tuple[int, str]],
    deferred_task_ids: set[int],
) -> list[dict[str, Any]]:
    """Finish partially completed datasets before starting untouched datasets."""

    def ordering_key(task: dict[str, Any]) -> tuple[Any, ...]:
        task_id = int(task["task_id"])
        completed_count = sum(pair_key(task_id, model) in completed_pairs for model in models)
        partially_completed = 0 < completed_count < len(models)
        return (
            task_id in deferred_task_ids,
            not partially_completed,
            execution_workload_key(task),
        )

    return sorted(tasks, key=ordering_key)


def fit_or_tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    tune: bool,
    inner_splits: int,
    random_state: int,
    hugiml_scenario: str,
    hugiml_max_fit_seconds: float | None,
    validation_protocol: str = "nested",
    early_stopping: bool = False,
    X_validation: pd.DataFrame | None = None,
    y_validation: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any], float | None, float, float, dict[str, Any], Any]:
    benchmark_base.RANDOM_STATE = int(random_state)
    candidates, builder, complexity_fn, budget = benchmark_base.get_model_spec(
        model_name,
        hugiml_scenario=hugiml_scenario if is_hugiml_model(model_name) else None,
        hugiml_max_fit_seconds=hugiml_max_fit_seconds,
    )
    if is_hugiml_model(model_name):
        candidates = _configure_hugiml_candidates(candidates, early_stopping=early_stopping)
    elif validation_protocol in {"rotating", "tabarena"}:
        rotating_family = {
            "XGB standard": "XGBoost",
            "LightGBM standard": "LightGBM",
            "RandomForest standard": "RandomForest",
        }.get(model_name)
        if rotating_family is not None:
            candidates = list(benchmark_base.ParameterGrid(rotating_baseline_grid(rotating_family)))
    if model_name == "XGB standard" and benchmark_base.XGBClassifier is None:
        raise ImportError("xgboost is required for XGB standard")
    if model_name == "LightGBM standard" and benchmark_base.LGBMClassifier is None:
        raise ImportError("lightgbm is required for LightGBM standard")

    _, class_counts = np.unique(y_train, return_counts=True)
    tuning_fallback_reason = None
    if tune and (class_counts.size < 2 or int(class_counts.min()) < 2):
        tuning_fallback_reason = (
            "inner_stratified_cv_unavailable:"
            f"smallest_training_class={int(class_counts.min()) if class_counts.size else 0}"
        )

    if tuning_fallback_reason is not None:
        fallback_params = (
            candidates[0]
            if validation_protocol == "rotating" and not is_hugiml_model(model_name)
            else {}
        )
        estimator = (
            builder(fallback_params)
            if is_hugiml_model(model_name)
            else baseline_pipeline(builder(fallback_params))
        )
        started = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_ms = (time.perf_counter() - started) * 1000.0
        return (
            estimator,
            {},
            None,
            fit_ms,
            0.0,
            {
                "inner_scoring": None,
                "inner_tuning_performed": False,
                "inner_tuning_fallback": tuning_fallback_reason,
            },
            complexity_fn,
        )

    if tune and validation_protocol == "rotating":
        if X_validation is None or y_validation is None:
            raise ValueError("Rotating validation requires an explicit validation fold.")
        if is_hugiml_model(model_name):
            estimator, params, best_score, tune_ms, info = (
                _fit_hugiml_with_rotating_validation_cached(
                    candidates=candidates,
                    X_train=X_train,
                    y_train=y_train,
                    X_validation=X_validation,
                    y_validation=y_validation,
                )
            )
        else:
            estimator, params, best_score, tune_ms, info = _fit_candidates_without_refit(
                model_name=model_name,
                candidates=candidates,
                builder=builder,
                complexity_fn=complexity_fn,
                folds=[(X_train, y_train, X_validation, y_validation)],
            )
        info["inner_tuning_performed"] = True
        info["inner_tuning_fallback"] = None
        return estimator, params, best_score, info["_final_refit_ms"], tune_ms, info, complexity_fn

    if tune and validation_protocol == "tabarena" and DEFAULT_NESTED_CV_ENSEMBLE_FOLDS is not None:
        estimator, params, best_score, tune_ms, info = _fit_tabarena_cv_ensemble(
            model_name=model_name,
            candidates=candidates,
            builder=builder,
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
        )
        info["inner_tuning_performed"] = True
        info["inner_tuning_fallback"] = None
        return estimator, params, best_score, 0.0, tune_ms, info, complexity_fn

    if tune:
        if is_hugiml_model(model_name):
            if early_stopping:
                split_count = benchmark_base._validated_stratified_splits(
                    y_train, inner_splits, label="Inner HUGIML training-partition CV"
                )
                cv = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=random_state)
                folds = [
                    (
                        benchmark_base._force_writable_frame(
                            X_train.iloc[tr].reset_index(drop=True)
                        ),
                        y_train[tr],
                        benchmark_base._force_writable_frame(
                            X_train.iloc[va].reset_index(drop=True)
                        ),
                        y_train[va],
                    )
                    for tr, va in cv.split(X_train, y_train)
                ]
                estimator, params, best_score, tune_ms, info = _fit_candidates_without_refit(
                    model_name=model_name,
                    candidates=candidates,
                    builder=builder,
                    complexity_fn=complexity_fn,
                    folds=folds,
                )
            else:
                estimator, params, best_score, tune_ms, info = benchmark_base._tune_hugiml_inner_cv(
                    candidates,
                    X_train,
                    y_train,
                    inner_splits=inner_splits,
                    random_state=random_state,
                    hugiml_max_fit_seconds=hugiml_max_fit_seconds,
                )
        else:
            estimator, params, best_score, tune_ms, info = tune_baseline(
                candidates,
                builder,
                X_train,
                y_train,
                inner_splits=inner_splits,
                random_state=random_state,
            )
        fit_ms = benchmark_base._final_refit_ms_from_info(estimator, info)
        info["inner_tuning_performed"] = True
        info["inner_tuning_fallback"] = None
        return estimator, params, finite_or_none(best_score), fit_ms, tune_ms, info, complexity_fn

    estimator = builder({}) if is_hugiml_model(model_name) else baseline_pipeline(builder({}))
    started = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_ms = (time.perf_counter() - started) * 1000.0
    return (
        estimator,
        {},
        None,
        fit_ms,
        0.0,
        {
            "inner_scoring": None,
            "inner_tuning_performed": False,
            "inner_tuning_fallback": None,
        },
        complexity_fn,
    )


def probability_matrix(
    estimator: Any,
    X: pd.DataFrame,
    n_classes: int,
    *,
    trained_class_labels: np.ndarray | None = None,
) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X), dtype=float)
    elif hasattr(estimator, "decision_function"):
        score = np.asarray(estimator.decision_function(X), dtype=float)
        if score.ndim == 1:
            positive = 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))
            proba = np.column_stack([1.0 - positive, positive])
        else:
            shifted = score - np.max(score, axis=1, keepdims=True)
            exp = np.exp(np.clip(shifted, -40.0, 40.0))
            proba = exp / exp.sum(axis=1, keepdims=True)
    else:
        pred = np.asarray(estimator.predict(X), dtype=int)
        proba = np.zeros((len(pred), n_classes), dtype=float)
        proba[np.arange(len(pred)), pred] = 1.0

    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.shape[1] == 1 and n_classes == 2:
        proba = np.column_stack([1.0 - proba[:, 0], proba[:, 0]])

    classes = np.asarray(getattr(estimator, "classes_", np.arange(proba.shape[1])))
    if trained_class_labels is not None:
        trained_class_labels = np.asarray(trained_class_labels, dtype=int)
        mapped_classes = np.full(classes.shape, -1, dtype=int)
        for index, cls in enumerate(classes):
            local_class = int(cls)
            if 0 <= local_class < len(trained_class_labels):
                mapped_classes[index] = int(trained_class_labels[local_class])
        classes = mapped_classes
    if proba.shape[1] != n_classes or not np.array_equal(
        classes.astype(int, copy=False), np.arange(n_classes)
    ):
        aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
        for col, cls in enumerate(classes):
            cls_int = int(cls)
            if 0 <= cls_int < n_classes and col < proba.shape[1]:
                aligned[:, cls_int] = proba[:, col]
        proba = aligned

    row_sums = proba.sum(axis=1, keepdims=True)
    invalid = ~np.isfinite(proba).all(axis=1) | (row_sums[:, 0] <= 0)
    if invalid.any():
        proba[invalid] = 1.0 / n_classes
        row_sums = proba.sum(axis=1, keepdims=True)
    return np.clip(proba / row_sums, 0.0, 1.0)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float | None]:
    n_classes = proba.shape[1]
    pred = np.argmax(proba, axis=1)
    out: dict[str, float | None] = {
        "accuracy": finite_or_none(accuracy_score(y_true, pred)),
        "balanced_accuracy": finite_or_none(balanced_accuracy_score(y_true, pred)),
        "f1": finite_or_none(
            f1_score(
                y_true,
                pred,
                average="binary" if n_classes == 2 else "macro",
                zero_division=0,
            )
        ),
    }
    if n_classes == 2:
        try:
            out["log_loss"] = finite_or_none(log_loss(y_true, proba, labels=[0, 1]))
        except (TypeError, ValueError):
            out["log_loss"] = None
        try:
            out["roc_auc"] = finite_or_none(roc_auc_score(y_true, proba[:, 1]))
        except (TypeError, ValueError):
            out["roc_auc"] = None
        try:
            out["avg_precision"] = finite_or_none(average_precision_score(y_true, proba[:, 1]))
        except (TypeError, ValueError):
            out["avg_precision"] = None
        try:
            out["brier"] = finite_or_none(np.mean((proba[:, 1] - y_true) ** 2))
        except (TypeError, ValueError):
            out["brier"] = None
    else:
        try:
            out["log_loss"] = finite_or_none(log_loss(y_true, proba, labels=np.arange(n_classes)))
        except (TypeError, ValueError):
            out["log_loss"] = None
        try:
            out["roc_auc"] = finite_or_none(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
            )
        except (TypeError, ValueError):
            out["roc_auc"] = None
        y_bin = label_binarize(y_true, classes=np.arange(n_classes))
        try:
            out["avg_precision"] = finite_or_none(
                average_precision_score(y_bin, proba, average="macro")
            )
        except (TypeError, ValueError):
            out["avg_precision"] = None
        try:
            out["brier"] = finite_or_none(np.mean(np.sum((proba - y_bin) ** 2, axis=1)))
        except (TypeError, ValueError):
            out["brier"] = None
    return out


def rotating_validation_splits(
    splits: Sequence[dict[str, Any]], n_rows: int, *, max_splits: int | None = None
) -> list[dict[str, Any]]:
    """Build train/validation/test rotations from genuine K-fold partitions.

    Within each OpenML repeat/sample group, test fold ``F_i`` is paired with
    validation fold ``F_(i+1) mod K`` and every remaining fold is training
    data. Independent holdouts are rejected because overlapping test sets
    would leak rows between validation and test.
    """
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for split in splits:
        groups.setdefault((int(split["repeat"]), int(split["sample"])), []).append(split)

    rotated: list[dict[str, Any]] = []
    all_rows = np.arange(int(n_rows), dtype=np.int64)
    for group_key in sorted(groups):
        folds = sorted(groups[group_key], key=lambda row: int(row["fold"]))
        if len(folds) < 3:
            raise ValueError(
                "Rotating validation requires at least three folds per repeat/sample; "
                f"group {group_key} has {len(folds)}."
            )
        test_sets = [np.asarray(row["test_idx"], dtype=np.int64) for row in folds]
        concatenated = np.concatenate(test_sets)
        if np.unique(concatenated).size != concatenated.size or np.unique(concatenated).size != int(
            n_rows
        ):
            raise ValueError(
                "Rotating validation requires disjoint exhaustive test folds. "
                f"Repeat/sample group {group_key} is an independent-holdout collection."
            )
        for index, split in enumerate(folds):
            test_idx = test_sets[index]
            validation_idx = test_sets[(index + 1) % len(folds)]
            excluded = np.union1d(test_idx, validation_idx)
            train_idx = np.setdiff1d(all_rows, excluded, assume_unique=False)
            item = dict(split)
            item.update(
                {
                    "split_id": len(rotated),
                    "train_idx": train_idx,
                    "validation_idx": validation_idx,
                    "test_idx": test_idx,
                    "validation_fold": int(folds[(index + 1) % len(folds)]["fold"]),
                    "training_folds": [
                        int(row["fold"])
                        for offset, row in enumerate(folds)
                        if offset not in {index, (index + 1) % len(folds)}
                    ],
                }
            )
            rotated.append(item)
    return rotated[: int(max_splits)] if max_splits is not None else rotated


def _instance_inspection_values(
    estimator: Any,
    X_test: pd.DataFrame,
    instance_fn,
) -> np.ndarray | None:
    """Return row-level inspection units through retained TabArena wrappers."""
    if not callable(instance_fn):
        return None
    if isinstance(estimator, TabArenaPreprocessedEstimator):
        transformed = estimator.transform_features(X_test)
        return _instance_inspection_values(estimator.estimator_, transformed, instance_fn)
    if isinstance(estimator, tabarena_protocol.CrossValidatedEnsemble):
        arrays: list[np.ndarray] = []
        for child in estimator.estimators_:
            values = _instance_inspection_values(child, X_test, instance_fn)
            if values is None:
                return None
            values = np.asarray(values, dtype=float).ravel()
            if values.size != len(X_test) or not np.isfinite(values).all():
                return None
            arrays.append(values)
        return None if not arrays else np.sum(np.vstack(arrays), axis=0)
    try:
        values = instance_fn(estimator, X_test)
    except Exception:
        return None
    if values is None:
        return None
    values = np.asarray(values, dtype=float).ravel()
    if values.size != len(X_test) or not np.isfinite(values).all():
        return None
    return values


def complexity_fields(estimator: Any, X_test: pd.DataFrame, complexity_fn) -> dict[str, Any]:
    if isinstance(estimator, TabArenaPreprocessedEstimator):
        return complexity_fields(
            estimator.estimator_,
            estimator.transform_features(X_test),
            complexity_fn,
        )
    if isinstance(estimator, tabarena_protocol.CrossValidatedEnsemble):
        child_rows = [
            complexity_fields(child, X_test, complexity_fn) for child in estimator.estimators_
        ]
        out = {}
        for key in ("complexity_model_units", "complexity_model_inspection_units"):
            values = [row[key] for row in child_rows if row.get(key) is not None]
            out[key] = float(np.sum(values)) if values else None
        out["complexity"] = out["complexity_model_inspection_units"]
        out["complexity_report_json"] = json.dumps(
            {
                "model_type": "cross_validated_ensemble",
                "child_count": len(child_rows),
                "aggregation": "sum_across_retained_children",
            },
            sort_keys=True,
        )
        out["complexity_model_units_per_child_mean"] = (
            out["complexity_model_units"] / len(child_rows) if child_rows else None
        )
        out["complexity_model_inspection_units_per_child_mean"] = (
            out["complexity_model_inspection_units"] / len(child_rows) if child_rows else None
        )
        instance_fn = getattr(
            getattr(benchmark_base, "_hugiml_pkg", None),
            "get_instance_inspection_units",
            None,
        )
        combined = _instance_inspection_values(estimator, X_test, instance_fn)
        if combined is not None:
            summary = summary_from_moments(
                int(combined.size),
                float(combined.sum()),
                float(np.square(combined).sum()),
            )
            summary["min"] = float(combined.min())
            summary["max"] = float(combined.max())
            for key, value in summary.items():
                out[f"complexity_instance_inspection_units_{key}"] = value
        else:
            for key in (
                "mean",
                "std",
                "standard_error",
                "ci_lower",
                "ci_upper",
                "confidence_level",
                "n_samples",
                "sum",
                "sum_squares",
                "min",
                "max",
            ):
                out[f"complexity_instance_inspection_units_{key}"] = None
        return out
    instance_keys = (
        "mean",
        "std",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "confidence_level",
        "n_samples",
        "sum",
        "sum_squares",
        "min",
        "max",
    )
    out: dict[str, Any] = {
        "complexity": None,
        "complexity_model_units": None,
        "complexity_model_inspection_units": None,
        "complexity_report_json": "{}",
    }
    for key in instance_keys:
        out[f"complexity_instance_inspection_units_{key}"] = None
    try:
        report = benchmark_base.get_complexity_report(estimator, X=X_test) or {}
        model_units = report.get("model_units", {}).get("value")
        model_inspection = report.get("model_inspection_units", {}).get("value")
        instance = report.get("instance_inspection_units", {})
        fallback = complexity_fn(estimator) if model_inspection is None and complexity_fn else None
        resolved_inspection = model_inspection if model_inspection is not None else fallback
        out.update(
            {
                "complexity": finite_or_none(resolved_inspection),
                "complexity_model_units": finite_or_none(model_units),
                "complexity_model_inspection_units": finite_or_none(resolved_inspection),
                "complexity_report_json": json.dumps(report, sort_keys=True, default=json_default),
            }
        )
        if instance.get("available"):
            for key in instance_keys:
                value = instance.get(key)
                if key in {"n_samples", "min", "max"}:
                    out[f"complexity_instance_inspection_units_{key}"] = (
                        None if value is None else int(value)
                    )
                else:
                    out[f"complexity_instance_inspection_units_{key}"] = finite_or_none(value)
    except Exception as exc:
        try:
            out["complexity"] = finite_or_none(complexity_fn(estimator)) if complexity_fn else None
        except Exception:
            pass
        out["complexity_error"] = f"{type(exc).__name__}: {exc}"
    return out


RPTE_BACKEND_TO_PATH = {
    "sequential_default": "rpte_sequential",
    "bounded_lookahead": "rpte_lookahead",
}
RPTE_DISTRIBUTION_METRICS = (
    "rpte_inputs_passed",
    "rpte_tree_count",
    "rpte_active_tree_count",
    "rpte_leaf_count",
    "rpte_active_leaf_count",
    "rpte_direct_term_count",
    "rpte_candidate_direct_term_count",
    "rpte_average_leaf_path_length",
    "rpte_active_average_leaf_path_length",
    "rpte_max_leaf_path_length",
    "rpte_selected_tree_count",
    "rpte_early_stopping_evaluated_tree_count",
)


def _fitted_rpte_estimators(estimator: Any) -> list[Any]:
    """Find fitted binary RPTE estimators through HUGIML/sklearn wrappers."""
    found: list[Any] = []
    seen: set[int] = set()
    stack = [estimator]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if current.__class__.__name__ == "LeafWiseBoundedLookaheadRPTEFeatureLR" and hasattr(
            current, "fe_"
        ):
            found.append(current)
            continue
        model = getattr(current, "model_", None)
        if model is not None:
            stack.append(model)
        for attr in ("best_estimator_", "estimator_", "estimator"):
            child = getattr(current, attr, None)
            if child is not None and child is not current:
                stack.append(child)
        for child in getattr(current, "estimators_", []) or []:
            stack.append(child)
        for _, child in getattr(current, "steps", []) or []:
            stack.append(child)
        named_steps = getattr(current, "named_steps", None)
        if named_steps:
            stack.extend(named_steps.values())
    return found


def _rpte_rule_rows(estimator: Any, fitted_rpte: Sequence[Any]) -> list[dict[str, Any]]:
    for candidate in (estimator, getattr(estimator, "best_estimator_", None)):
        method = getattr(candidate, "rpte_rule_table", None)
        if callable(method):
            return [dict(row) for row in (method() or [])]
    rows: list[dict[str, Any]] = []
    for class_index, candidate in enumerate(fitted_rpte):
        method = getattr(candidate, "unified_rule_table", None)
        if not callable(method):
            continue
        for raw in method() or []:
            row = dict(raw)
            row.setdefault("class", class_index)
            rows.append(row)
    return rows


def hugiml_run_fields(estimator: Any) -> dict[str, Any]:
    """Describe the route and fitted RPTE structure for one official split."""
    rpte_estimators = _fitted_rpte_estimators(estimator)
    rows = _rpte_rule_rows(estimator, rpte_estimators)
    tree_rows = [
        row
        for row in rows
        if row.get("tree_index") is not None and str(row.get("backend")) in RPTE_BACKEND_TO_PATH
    ]
    direct_rows = [
        row
        for row in rows
        if row.get("tree_index") is None
        and str(row.get("backend")) in {"direct_hugiml_feature", "raw_hugiml_features"}
        and row.get("source_selection_status") in {None, "not_selected_in_tree_split"}
    ]
    active_tree_rows = [
        row for row in tree_rows if abs(float(row.get("final_logistic_coefficient") or 0.0)) > 1e-12
    ]
    active_direct_rows = [
        row
        for row in direct_rows
        if abs(float(row.get("final_logistic_coefficient") or 0.0)) > 1e-12
    ]

    backend_counts: dict[str, int] = {}
    for rpte in rpte_estimators:
        fe = getattr(rpte, "fe_", None)
        if getattr(fe, "_default_fe", None) is not None:
            backend = "sequential_default"
        elif getattr(fe, "trees_", None):
            backend = "bounded_lookahead"
        else:
            continue
        backend_counts[RPTE_BACKEND_TO_PATH[backend]] = (
            backend_counts.get(RPTE_BACKEND_TO_PATH[backend], 0) + 1
        )
    if not backend_counts:
        for backend in sorted({str(row.get("backend")) for row in tree_rows}):
            path = RPTE_BACKEND_TO_PATH.get(backend)
            if path:
                backend_counts[path] = sum(
                    1 for row in tree_rows if str(row.get("backend")) == backend
                )

    if not rpte_estimators and not tree_rows and not direct_rows:
        path = "lr"
    elif len(backend_counts) == 1:
        path = next(iter(backend_counts))
    else:
        # Multiclass OvR estimators may independently take different adaptive
        # routes. Preserve those routes below instead of inventing one label.
        path = None
    paths = ["lr"] if path == "lr" else sorted(backend_counts)

    input_counts = [
        int(rpte.n_input_features_)
        for rpte in rpte_estimators
        if getattr(rpte, "n_input_features_", None) is not None
    ]
    tree_keys = {
        (str(row.get("class")), int(row["tree_index"]), str(row.get("backend")))
        for row in tree_rows
    }
    active_tree_keys = {
        (str(row.get("class")), int(row["tree_index"]), str(row.get("backend")))
        for row in active_tree_rows
    }
    path_lengths = [len(row.get("conditions") or []) for row in tree_rows]
    active_path_lengths = [len(row.get("conditions") or []) for row in active_tree_rows]
    early_stopping_used = [
        bool(getattr(rpte, "early_stopping_used_", False)) for rpte in rpte_estimators
    ]
    selected_tree_counts = [
        int(rpte.n_estimators_)
        for rpte in rpte_estimators
        if getattr(rpte, "n_estimators_", None) is not None
    ]
    stopping_scores = [
        float(rpte.early_stopping_best_score_)
        for rpte in rpte_estimators
        if getattr(rpte, "early_stopping_best_score_", None) is not None
    ]
    stopping_losses = [
        float(rpte.early_stopping_best_loss_)
        for rpte in rpte_estimators
        if getattr(rpte, "early_stopping_best_loss_", None) is not None
    ]
    stopping_metrics = sorted(
        {
            str(rpte.early_stopping_metric_)
            for rpte in rpte_estimators
            if getattr(rpte, "early_stopping_metric_", None)
        }
    )
    validation_sources = sorted(
        {
            str(rpte.early_stopping_validation_source_)
            for rpte in rpte_estimators
            if getattr(rpte, "early_stopping_validation_source_", None)
        }
    )
    evaluated_tree_counts = [
        len(getattr(rpte, "early_stopping_evaluated_estimators_", []) or [])
        for rpte in rpte_estimators
        if getattr(rpte, "early_stopping_used_", False)
    ]
    stopping_fallbacks = [
        str(rpte.early_stopping_fallback_reason_)
        for rpte in rpte_estimators
        if getattr(rpte, "early_stopping_fallback_reason_", None)
    ]
    canonical = dict(getattr(estimator, "_downstream_lr_canonicalization_", None) or {})
    return {
        "hugiml_path": path,
        "hugiml_benchmark_lr_C": finite_or_none(getattr(estimator, "_benchmark_lr_C", None)),
        "hugiml_lr_solver": str(getattr(estimator, "lr_solver", "auto")),
        "downstream_columns_before_reduction": canonical.get("input_columns"),
        "downstream_columns_after_reduction": canonical.get("retained_columns"),
        "downstream_vif_columns_above_threshold": canonical.get("vif_columns_above_threshold"),
        "downstream_vif_removed_patterns": canonical.get("removed_high_vif_pattern_columns"),
        "downstream_vif_removed_augmented_pairs": canonical.get(
            "removed_high_vif_augmented_pair_columns"
        ),
        "downstream_vif_maximum": canonical.get("maximum_vif"),
        "downstream_vif_median": canonical.get("median_vif"),
        "downstream_vif_analysis_seconds": canonical.get("vif_analysis_seconds"),
        "hugiml_paths_json": json.dumps(paths),
        "hugiml_path_counts_json": json.dumps(
            {"lr": 1} if path == "lr" else backend_counts, sort_keys=True
        ),
        "rpte_binary_estimator_count": len(rpte_estimators) or None,
        "rpte_sequential_estimator_count": backend_counts.get("rpte_sequential", 0),
        "rpte_lookahead_estimator_count": backend_counts.get("rpte_lookahead", 0),
        # OvR estimators receive the same downstream matrix; do not multiply its width.
        "rpte_inputs_passed": (
            int(input_counts[0])
            if input_counts and len(set(input_counts)) == 1
            else (int(max(input_counts)) if input_counts else None)
        ),
        "rpte_inputs_passed_per_estimator_json": json.dumps(input_counts),
        "rpte_tree_count": len(tree_keys) if path != "lr" else None,
        "rpte_active_tree_count": len(active_tree_keys) if path != "lr" else None,
        "rpte_early_stopping_used": any(early_stopping_used),
        "rpte_early_stopping_used_estimator_count": sum(early_stopping_used),
        "rpte_selected_tree_counts_json": json.dumps(selected_tree_counts),
        "rpte_selected_tree_count": (
            int(max(selected_tree_counts)) if selected_tree_counts else None
        ),
        "rpte_early_stopping_best_score_mean": (
            float(np.mean(stopping_scores)) if stopping_scores else None
        ),
        "rpte_early_stopping_best_log_loss_mean": (
            float(np.mean(stopping_losses)) if stopping_losses else None
        ),
        "rpte_early_stopping_metrics_json": json.dumps(stopping_metrics),
        "rpte_early_stopping_validation_sources_json": json.dumps(validation_sources),
        "rpte_early_stopping_evaluated_tree_count": (
            int(max(evaluated_tree_counts)) if evaluated_tree_counts else None
        ),
        "rpte_early_stopping_fallbacks_json": json.dumps(stopping_fallbacks),
        "rpte_leaf_count": len(tree_rows) if path != "lr" else None,
        "rpte_active_leaf_count": len(active_tree_rows) if path != "lr" else None,
        "rpte_direct_term_count": len(active_direct_rows) if path != "lr" else None,
        "rpte_candidate_direct_term_count": len(direct_rows) if path != "lr" else None,
        "rpte_average_leaf_path_length": (
            float(np.mean(path_lengths)) if path_lengths else (0.0 if path != "lr" else None)
        ),
        "rpte_active_average_leaf_path_length": (
            float(np.mean(active_path_lengths))
            if active_path_lengths
            else (0.0 if path != "lr" else None)
        ),
        "rpte_max_leaf_path_length": (
            int(max(path_lengths)) if path_lengths else (0 if path != "lr" else None)
        ),
    }


def distribution_summary(values: Iterable[Any]) -> dict[str, Any]:
    clean = [value for raw in values if (value := finite_or_none(raw)) is not None]
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }
    array = np.asarray(clean, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def evaluate_official_split(
    *,
    model_name: str,
    task_meta: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    split: dict[str, Any],
    tune: bool,
    inner_splits: int,
    random_state: int,
    hugiml_scenario: str,
    hugiml_max_fit_seconds: float | None,
    validation_protocol: str = "nested",
    early_stopping: bool = False,
) -> tuple[dict[str, Any], int | None, dict[str, Any] | None]:
    validate_official_split(split, len(y))
    split_started = time.perf_counter()
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    X_train = benchmark_base._force_writable_frame(X.iloc[train_idx].reset_index(drop=True))
    X_test = benchmark_base._force_writable_frame(X.iloc[test_idx].reset_index(drop=True))
    y_train_global = np.array(y[train_idx], dtype=int, copy=True)
    trained_class_labels = np.unique(y_train_global)
    y_train = np.searchsorted(trained_class_labels, y_train_global).astype(int, copy=False)
    y_test = np.array(y[test_idx], dtype=int, copy=True)
    X_validation = None
    y_validation = None
    validation_error = None
    validation_idx = np.asarray(split.get("validation_idx", []), dtype=np.int64)
    if validation_protocol == "rotating":
        if validation_idx.size == 0:
            validation_error = "Rotating validation split is missing validation indices."
        else:
            X_validation = benchmark_base._force_writable_frame(
                X.iloc[validation_idx].reset_index(drop=True)
            )
            y_validation_global = np.array(y[validation_idx], dtype=int, copy=True)
            if not np.isin(y_validation_global, trained_class_labels).all():
                validation_error = (
                    "Rotating validation contains a class absent from its training folds."
                )
            else:
                y_validation = np.searchsorted(trained_class_labels, y_validation_global).astype(
                    int, copy=False
                )
    split_seed = int(random_state + split["repeat"] * 10007 + split["fold"] * 101 + split["sample"])

    row: dict[str, Any] = {
        "task_id": int(task_meta["task_id"]),
        "dataset_id": int(task_meta["dataset_id"]),
        "dataset": str(task_meta["dataset_name"]),
        "model": model_name,
        "split_id": int(split["split_id"]),
        "repeat": int(split["repeat"]),
        "fold": int(split["fold"]),
        "sample": int(split["sample"]),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_validation": int(validation_idx.size),
        "validation_protocol": validation_protocol,
        "validation_fold": split.get("validation_fold"),
        "training_folds": split.get("training_folds"),
        "early_stopping": bool(early_stopping),
        "training_class_labels": trained_class_labels.tolist(),
        "training_missing_class_count": int(
            int(task_meta["n_classes"]) - len(trained_class_labels)
        ),
        "inner_n_splits": (
            int(inner_splits) if tune and validation_protocol in {"nested", "tabarena"} else None
        ),
        "random_state": split_seed,
        "status": "error",
        "error_count": 1,
        "last_error": None,
    }
    try:
        effective_tune = bool(tune)
        effective_early_stopping = bool(early_stopping)
        fit_validation_protocol = validation_protocol
        rotating_fallback_reason = None
        if validation_error is not None and validation_protocol == "rotating":
            # Some rare-class K-fold assignments place every available example
            # of a class in the test and validation folds. Model selection is
            # then undefined. Use the complete outer-training partition
            # (training + validation), omit validation selection, and retain a
            # prediction for pooled out-of-fold metrics.
            train_idx = np.union1d(train_idx, validation_idx)
            X_train = benchmark_base._force_writable_frame(X.iloc[train_idx].reset_index(drop=True))
            y_train_global = np.array(y[train_idx], dtype=int, copy=True)
            trained_class_labels = np.unique(y_train_global)
            y_train = np.searchsorted(trained_class_labels, y_train_global).astype(int, copy=False)
            X_validation = None
            y_validation = None
            effective_tune = False
            effective_early_stopping = False
            row["early_stopping"] = False
            fit_validation_protocol = "rotating"
            rotating_fallback_reason = (
                "validation_class_absent_from_training_folds:"
                "fit_complete_outer_training_without_validation_selection"
            )
            row["n_train"] = int(len(train_idx))
            row["training_class_labels"] = trained_class_labels.tolist()
            row["training_missing_class_count"] = int(
                int(task_meta["n_classes"]) - len(trained_class_labels)
            )
        elif validation_error is not None:
            raise ValueError(validation_error)
        estimator, params, best_inner_score, fit_ms, tune_ms, info, complexity_fn = (
            fit_or_tune_model(
                model_name,
                X_train,
                y_train,
                tune=effective_tune,
                inner_splits=inner_splits,
                random_state=split_seed,
                hugiml_scenario=hugiml_scenario,
                hugiml_max_fit_seconds=hugiml_max_fit_seconds,
                validation_protocol=fit_validation_protocol,
                early_stopping=effective_early_stopping,
                X_validation=X_validation,
                y_validation=y_validation,
            )
        )
        started = time.perf_counter()
        proba = probability_matrix(
            estimator,
            X_test,
            int(task_meta["n_classes"]),
            trained_class_labels=trained_class_labels,
        )
        predict_ms = (time.perf_counter() - started) * 1000.0
        row.update(classification_metrics(y_test, proba))
        complexity_started = time.perf_counter()
        row.update(complexity_fields(estimator, X_test, complexity_fn))
        complexity_ms = (time.perf_counter() - complexity_started) * 1000.0
        if is_hugiml_model(model_name):
            row.update(hugiml_run_fields(estimator))
        row.update(
            {
                "auc": row.get("roc_auc"),
                "best_inner_score": best_inner_score,
                "best_params": json.dumps(params, sort_keys=True, default=json_default),
                "best_params_json": json.dumps(params, sort_keys=True, default=json_default),
                "fit_ms": finite_or_none(fit_ms),
                "tune_ms": finite_or_none(tune_ms),
                "predict_ms": finite_or_none(predict_ms),
                "complexity_ms": finite_or_none(complexity_ms),
                "fit_seconds": finite_or_none(fit_ms / 1000.0),
                "tune_seconds": finite_or_none(tune_ms / 1000.0),
                "predict_seconds": finite_or_none(predict_ms / 1000.0),
                "complexity_seconds": finite_or_none(complexity_ms / 1000.0),
                "fit_time_included_in_tuning": bool(
                    effective_tune and finite_or_none(tune_ms) is not None and tune_ms > 0.0
                ),
                "fit_time_accounting": (
                    "included_in_tuning"
                    if effective_tune and finite_or_none(tune_ms) is not None and tune_ms > 0.0
                    else "separate_from_tuning"
                ),
                "tuned": bool(info.get("inner_tuning_performed", tune)),
                "inner_tuning_fallback": info.get("inner_tuning_fallback"),
                "rotating_validation_fallback": rotating_fallback_reason,
                "status": "ok",
                "error_count": 0,
                "last_error": None,
                "inner_scoring": info.get("inner_scoring", "roc_auc"),
                "selection_refit_performed": info.get("selection_refit_performed", True),
                "retained_cv_ensemble": info.get("retained_cv_ensemble", False),
                "cv_ensemble_child_count": info.get("cv_ensemble_child_count"),
                "candidate_fit_count": info.get("candidate_fit_count"),
                "preprocessing_fit_count": info.get("preprocessing_fit_count"),
                "preprocessing_reuse_count": info.get("preprocessing_reuse_count"),
                "validation_selection_metrics": info.get("validation_selection_metrics", []),
                "baseline_early_stopping_used": info.get("baseline_early_stopping_used", False),
                "baseline_selected_n_estimators": info.get("baseline_selected_n_estimators"),
                "baseline_max_n_estimators": info.get("baseline_max_n_estimators"),
                "baseline_early_stopping_patience": info.get("baseline_early_stopping_patience"),
                "baseline_early_stopping_metric": info.get("baseline_early_stopping_metric"),
            }
        )
        if is_hugiml_model(model_name):
            row.update({key: value for key, value in info.items() if key.startswith("hugiml_")})
        row["split_seconds"] = float(time.perf_counter() - split_started)
        separately_accounted_fit_seconds = (
            0.0 if row.get("fit_time_included_in_tuning") else (row.get("fit_seconds") or 0.0)
        )
        row["phase_accounted_seconds"] = float(
            separately_accounted_fit_seconds
            + (row.get("tune_seconds") or 0.0)
            + (row.get("predict_seconds") or 0.0)
            + (row.get("complexity_seconds") or 0.0)
        )
        row["evaluation_overhead_seconds"] = max(
            0.0,
            row["split_seconds"] - row["phase_accounted_seconds"],
        )
        return (
            row,
            benchmark_base._model_feature_count(estimator, X.shape[1]),
            {"y_true": y_test.tolist(), "proba": proba.tolist()},
        )
    except Exception as exc:
        row["last_error"] = f"{type(exc).__name__}: {exc}"
        for key in (
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "log_loss",
            "auc",
            "avg_precision",
            "brier",
            "f1",
            "best_inner_score",
            "complexity",
            "complexity_model_units",
            "complexity_model_inspection_units",
            "complexity_instance_inspection_units_mean",
            "complexity_instance_inspection_units_std",
            "complexity_instance_inspection_units_standard_error",
            "complexity_instance_inspection_units_ci_lower",
            "complexity_instance_inspection_units_ci_upper",
            "complexity_instance_inspection_units_confidence_level",
            "complexity_instance_inspection_units_n_samples",
            "complexity_instance_inspection_units_sum",
            "complexity_instance_inspection_units_sum_squares",
            "complexity_instance_inspection_units_min",
            "complexity_instance_inspection_units_max",
            "fit_ms",
            "tune_ms",
            "predict_ms",
            "complexity_ms",
            "fit_seconds",
            "tune_seconds",
            "predict_seconds",
            "complexity_seconds",
            "phase_accounted_seconds",
            "evaluation_overhead_seconds",
            "rpte_inputs_passed",
            "rpte_tree_count",
            "rpte_active_tree_count",
            "rpte_leaf_count",
            "rpte_active_leaf_count",
            "rpte_direct_term_count",
            "rpte_candidate_direct_term_count",
            "rpte_average_leaf_path_length",
            "rpte_active_average_leaf_path_length",
            "rpte_max_leaf_path_length",
            "downstream_columns_before_reduction",
            "downstream_columns_after_reduction",
            "downstream_vif_columns_above_threshold",
            "downstream_vif_removed_patterns",
            "downstream_vif_removed_augmented_pairs",
            "downstream_vif_maximum",
            "downstream_vif_median",
            "downstream_vif_analysis_seconds",
        ):
            row.setdefault(key, None)
        row.setdefault("best_params", "{}")
        row.setdefault("best_params_json", "{}")
        row["split_seconds"] = float(time.perf_counter() - split_started)
        return row, None, None


# ---------------------------------------------------------------------------
# Pair checkpointing and aggregation
# ---------------------------------------------------------------------------


def pair_key(task_id: int, model: str) -> tuple[int, str]:
    return int(task_id), str(model)


def active_run_id(payload: dict[str, Any], run_id: str | None = None) -> str:
    selected = run_id or payload.get("active_run_id")
    runs = payload.get("runs", {})
    if not selected or selected not in runs:
        available = ", ".join(sorted(str(value) for value in runs)) or "none"
        raise ValueError(f"Checkpoint run is unavailable. Available run IDs: {available}")
    return str(selected)


def run_view(payload: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    selected = active_run_id(payload, run_id)
    run = dict(payload["runs"][selected])
    run["run_id"] = selected
    run["results"] = [
        dict(row) for row in payload.get("results", []) if str(row.get("run_id")) == selected
    ]
    return run


def completed_pair_keys(payload: dict[str, Any], run_id: str | None = None) -> set[tuple[int, str]]:
    completed: set[tuple[int, str]] = set()
    for row in run_view(payload, run_id).get("results", []):
        expected = int(
            row.get("official_splits_expected", row.get("official_splits_completed", 0)) or 0
        )
        successful = int(row.get("official_splits_successful", 0) or 0)
        status = str(row.get("status", ""))
        if status == "ok" and expected > 0 and successful == expected:
            completed.add(pair_key(int(row["task_id"]), str(row["model"])))
    return completed


def _task_reuse_signature(run: dict[str, Any], task_id: int) -> dict[str, Any] | None:
    for task in run.get("selected_tasks", []):
        if int(task.get("task_id", -1)) == int(task_id):
            return {
                "task_id": int(task_id),
                "dataset_id": int(task.get("dataset_id", -1)),
                "official_split_count": int(task.get("official_split_count", 0)),
                "sha256": safe_jsonable(task.get("sha256", {})),
            }
    return None


def _model_reuse_signature(run: dict[str, Any], model: str) -> dict[str, Any]:
    metadata = run.get("metadata", {})
    source = metadata.get("source", {})
    grid = metadata.get("grid_snapshot", {}).get(model)
    signature: dict[str, Any] = {
        "model": model,
        "tune": bool(metadata.get("tune", True)),
        "inner_splits": metadata.get("inner_splits"),
        "random_state": metadata.get("random_state"),
        "max_official_splits": metadata.get("max_official_splits"),
        "validation_protocol": metadata.get("validation_protocol", "nested"),
        "early_stopping": bool(metadata.get("early_stopping", False)),
        "grid": safe_jsonable(grid),
    }
    if model in {
        "XGB standard",
        "LightGBM standard",
        "RandomForest standard",
    }:
        signature.update(
            {
                "baseline_validation_early_stopping": bool(
                    metadata.get("baseline_validation_early_stopping", False)
                ),
                "baseline_early_stopping_max_estimators": metadata.get(
                    "baseline_early_stopping_max_estimators"
                ),
                "baseline_early_stopping_patience": metadata.get(
                    "baseline_early_stopping_patience"
                ),
            }
        )
    if is_hugiml_model(model):
        signature.update(
            {
                "package_version": source.get("package_version"),
                "hugiml_fit_core_sha256": source.get("hugiml_fit_core_sha256"),
                "benchmark_dashboard_sha256": source.get("benchmark_dashboard_sha256"),
                "hugiml_scenario": metadata.get("hugiml_scenario"),
                "hugiml_max_fit_seconds": metadata.get("hugiml_max_fit_seconds"),
            }
        )
    return signature


def reuse_completed_compatible_results(
    payload: dict[str, Any],
    *,
    target_run_id: str,
    task_ids: Sequence[int],
    models: Sequence[str],
) -> dict[str, int]:
    """Copy fully completed, fitting-compatible rows into a broader staged run."""
    allowed_tasks = {int(task_id) for task_id in task_ids}
    allowed_models = set(models)
    if not allowed_models:
        return {}

    target_run = payload["runs"][target_run_id]
    existing = {
        pair_key(int(row["task_id"]), str(row["model"]))
        for row in payload.get("results", [])
        if str(row.get("run_id")) == target_run_id
    }
    reusable: dict[tuple[int, str], dict[str, Any]] = {}
    for row in reversed(payload.get("results", [])):
        task_id = int(row.get("task_id", -1))
        model = str(row.get("model", ""))
        key = pair_key(task_id, model)
        source_run_id = str(row.get("run_id", ""))
        source_run = payload.get("runs", {}).get(source_run_id)
        expected = int(
            row.get("official_splits_expected", row.get("official_splits_completed", 0)) or 0
        )
        successful = int(row.get("official_splits_successful", 0) or 0)
        status = str(row.get("status", ""))
        complete = (
            expected > 0
            and successful == expected
            and not int(row.get("error_count", 0) or 0)
            and status in {"", "ok"}
        )
        if (
            source_run_id != target_run_id
            and source_run is not None
            and task_id in allowed_tasks
            and model in allowed_models
            and complete
            and key not in reusable
            and _task_reuse_signature(source_run, task_id)
            == _task_reuse_signature(target_run, task_id)
            and _model_reuse_signature(source_run, model)
            == _model_reuse_signature(target_run, model)
        ):
            reusable[key] = row

    copied_by_model: dict[str, int] = {}
    for key, source_row in reusable.items():
        if key in existing:
            continue
        cloned = dict(source_row)
        cloned["reused_compatible_result"] = True
        cloned["reused_baseline_result"] = str(source_row.get("model")) in REUSABLE_BASELINE_MODELS
        cloned["reused_from_run_id"] = source_row.get("reused_from_run_id") or source_row.get(
            "run_id"
        )
        cloned["reused_via_run_id"] = source_row.get("run_id")
        cloned["run_id"] = target_run_id
        cloned.setdefault("status", "ok")
        cloned.setdefault(
            "official_splits_expected",
            int(cloned.get("official_splits_completed", 0) or 0),
        )
        payload["results"].append(cloned)
        existing.add(key)
        model = str(cloned["model"])
        copied_by_model[model] = copied_by_model.get(model, 0) + 1
    return copied_by_model


def aggregate_pair_rows(
    split_rows: list[dict[str, Any]],
    *,
    split_predictions: dict[int, dict[str, Any]],
    task_meta: dict[str, Any],
    model_name: str,
    model_feature_counts: list[int],
    tune: bool,
    inner_splits: int,
    hugiml_scenario: str,
    official_splits_expected: int,
) -> dict[str, Any]:
    ok = [row for row in split_rows if row.get("status") == "ok"]
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "log_loss",
        "avg_precision",
        "brier",
        "f1",
        "best_inner_score",
        "complexity",
        "complexity_model_units",
        "complexity_model_inspection_units",
        "fit_ms",
        "tune_ms",
        "predict_ms",
        "complexity_ms",
        "fit_seconds",
        "tune_seconds",
        "predict_seconds",
        "complexity_seconds",
        "split_seconds",
        "phase_accounted_seconds",
        "evaluation_overhead_seconds",
        "downstream_columns_before_reduction",
        "downstream_columns_after_reduction",
        "downstream_vif_columns_above_threshold",
        "downstream_vif_removed_patterns",
        "downstream_vif_removed_augmented_pairs",
        "downstream_vif_maximum",
        "downstream_vif_median",
        "downstream_vif_analysis_seconds",
    ]
    repeat_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in ok:
        repeat_groups.setdefault((int(row.get("repeat", 0)), int(row.get("sample", 0))), []).append(
            row
        )
    out: dict[str, Any] = {}
    for metric in metrics:
        repeat_values = [
            mean_or_none(row.get(metric) for row in rows) for rows in repeat_groups.values()
        ]
        out[metric] = mean_or_none(repeat_values)
        out[f"{metric}_std"] = std_or_none(repeat_values)
    for metric in (
        "fit_seconds",
        "tune_seconds",
        "predict_seconds",
        "complexity_seconds",
        "split_seconds",
        "phase_accounted_seconds",
        "evaluation_overhead_seconds",
    ):
        out[f"total_{metric}"] = float(
            sum(
                float(value) for row in ok if (value := finite_or_none(row.get(metric))) is not None
            )
        )

    # Preserve conventional cross-validation aggregation whenever every
    # official fold has a valid metric. Pool out-of-fold predictions only as a
    # fallback for metrics that are undefined on one or more rare-class folds.
    probability_metric_rows: dict[tuple[int, int], dict[str, float | None]] = {}
    if len(ok) == official_splits_expected:
        grouped_predictions: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in ok:
            prediction = split_predictions.get(int(row["split_id"]))
            if prediction is None:
                grouped_predictions = {}
                break
            group = (int(row.get("repeat", 0)), int(row.get("sample", 0)))
            grouped_predictions.setdefault(group, []).append(prediction)
        for group, predictions in grouped_predictions.items():
            y_true = np.concatenate([np.asarray(item["y_true"], dtype=int) for item in predictions])
            proba = np.vstack([np.asarray(item["proba"], dtype=float) for item in predictions])
            pooled_metrics = classification_metrics(y_true, proba)
            probability_metric_rows[group] = {
                metric: pooled_metrics.get(metric)
                for metric in ("roc_auc", "avg_precision", "brier")
            }
    pooled_fallback_metrics: list[str] = []
    for metric in ("roc_auc", "avg_precision", "brier"):
        repeat_metric_values = []
        used_fallback = False
        for group, rows in repeat_groups.items():
            valid_fold_count = sum(finite_or_none(row.get(metric)) is not None for row in rows)
            if valid_fold_count == len(rows):
                repeat_metric_values.append(mean_or_none(row.get(metric) for row in rows))
            elif group in probability_metric_rows:
                repeat_metric_values.append(probability_metric_rows[group].get(metric))
                used_fallback = True
            else:
                repeat_metric_values.append(None)
        if repeat_metric_values:
            out[metric] = mean_or_none(repeat_metric_values)
            out[f"{metric}_std"] = std_or_none(repeat_metric_values)
        if used_fallback:
            pooled_fallback_metrics.append(metric)
    out["probability_metrics_aggregation"] = (
        "mean_of_repeat_metrics_with_pooled_oof_fallback"
        if pooled_fallback_metrics
        else "mean_of_repeat_metrics"
    )
    out["pooled_oof_fallback_metrics"] = pooled_fallback_metrics
    out["roc_auc_aggregation"] = (
        "mean_of_repeat_metrics_with_pooled_oof_fallback"
        if "roc_auc" in pooled_fallback_metrics
        else "mean_of_repeat_metrics"
    )

    pooled_n = int(
        sum(int(row.get("complexity_instance_inspection_units_n_samples", 0) or 0) for row in ok)
    )
    pooled_sum = float(
        sum(float(row.get("complexity_instance_inspection_units_sum", 0.0) or 0.0) for row in ok)
    )
    pooled_sum_squares = float(
        sum(
            float(row.get("complexity_instance_inspection_units_sum_squares", 0.0) or 0.0)
            for row in ok
        )
    )
    confidence_values = [
        value
        for row in ok
        if (
            value := finite_or_none(
                row.get("complexity_instance_inspection_units_confidence_level")
            )
        )
        is not None
    ]
    if pooled_n > 0:
        pooled = summary_from_moments(
            pooled_n,
            pooled_sum,
            pooled_sum_squares,
            confidence_level=confidence_values[0] if confidence_values else 0.95,
        )
        for key, value in pooled.items():
            out[f"complexity_instance_inspection_units_{key}"] = value
        minimums = [
            value
            for row in ok
            if (value := finite_or_none(row.get("complexity_instance_inspection_units_min")))
            is not None
        ]
        maximums = [
            value
            for row in ok
            if (value := finite_or_none(row.get("complexity_instance_inspection_units_max")))
            is not None
        ]
        out["complexity_instance_inspection_units_min"] = (
            None if not minimums else int(min(minimums))
        )
        out["complexity_instance_inspection_units_max"] = (
            None if not maximums else int(max(maximums))
        )
    else:
        for key in (
            "mean",
            "std",
            "standard_error",
            "ci_lower",
            "ci_upper",
            "confidence_level",
            "n_samples",
            "sum",
            "sum_squares",
            "min",
            "max",
        ):
            out[f"complexity_instance_inspection_units_{key}"] = None

    out["auc"] = out["roc_auc"]
    out["auc_std"] = out["roc_auc_std"]
    if is_hugiml_model(model_name):
        path_counts: dict[str, int] = {}
        for row in ok:
            if row.get("hugiml_path") == "lr":
                path_counts["lr"] = path_counts.get("lr", 0) + 1
            for path, field in (
                ("rpte_sequential", "rpte_sequential_estimator_count"),
                ("rpte_lookahead", "rpte_lookahead_estimator_count"),
            ):
                path_counts[path] = path_counts.get(path, 0) + int(row.get(field, 0) or 0)
        path_counts = {path: count for path, count in path_counts.items() if count}
        out["hugiml_path_counts_json"] = json.dumps(path_counts, sort_keys=True)
        out["hugiml_dominant_path"] = (
            sorted(path_counts, key=lambda value: (-path_counts[value], value))[0]
            if path_counts
            else None
        )
        out["hugiml_paths_json"] = json.dumps(sorted(path_counts))
        for metric in RPTE_DISTRIBUTION_METRICS:
            summary = distribution_summary(row.get(metric) for row in ok)
            for statistic, value in summary.items():
                out[f"{metric}_{statistic}"] = value
        stopping_losses = [
            float(value)
            for row in ok
            if (value := row.get("rpte_early_stopping_best_log_loss_mean")) is not None
        ]
        out["rpte_early_stopping_best_log_loss_mean"] = (
            float(np.mean(stopping_losses)) if stopping_losses else None
        )
        out["rpte_early_stopping_metrics_json"] = json.dumps(
            sorted(
                {
                    value
                    for row in ok
                    for value in json.loads(row.get("rpte_early_stopping_metrics_json", "[]"))
                }
            )
        )
        out["rpte_early_stopping_validation_sources_json"] = json.dumps(
            sorted(
                {
                    value
                    for row in ok
                    for value in json.loads(
                        row.get("rpte_early_stopping_validation_sources_json", "[]")
                    )
                }
            )
        )
    selected_estimators = [
        int(value) for row in ok if (value := row.get("baseline_selected_n_estimators")) is not None
    ]
    out["baseline_early_stopping_used"] = any(
        bool(row.get("baseline_early_stopping_used")) for row in ok
    )
    out["baseline_selected_n_estimators_mean"] = (
        None if not selected_estimators else float(np.mean(selected_estimators))
    )
    out["baseline_selected_n_estimators_min"] = (
        None if not selected_estimators else int(min(selected_estimators))
    )
    out["baseline_selected_n_estimators_max"] = (
        None if not selected_estimators else int(max(selected_estimators))
    )
    parameter_rows = [
        {
            "split_id": int(row.get("split_id", index)),
            "best_params": json.loads(
                row.get("best_params_json") or row.get("best_params") or "{}"
            ),
        }
        for index, row in enumerate(ok)
    ]
    parameter_counts: dict[str, int] = {}
    for item in parameter_rows:
        canonical = json.dumps(item["best_params"], sort_keys=True, default=json_default)
        parameter_counts[canonical] = parameter_counts.get(canonical, 0) + 1
    parameter_mode = (
        sorted(parameter_counts, key=lambda value: (-parameter_counts[value], value))[0]
        if parameter_counts
        else "{}"
    )
    out["best_params_by_split_json"] = json.dumps(
        parameter_rows, sort_keys=True, default=json_default
    )
    out["best_params_frequency_json"] = json.dumps(parameter_counts, sort_keys=True)
    out["best_params_mode_json"] = parameter_mode
    out["best_params"] = parameter_mode
    out["best_params_json"] = parameter_mode
    out["best_params_unique_count"] = int(len(parameter_counts))
    inner_scoring_counts: dict[str, int] = {}
    for row in ok:
        value = row.get("inner_scoring")
        if value is not None:
            key = str(value)
            inner_scoring_counts[key] = inner_scoring_counts.get(key, 0) + 1
    out["inner_scoring_counts_json"] = json.dumps(inner_scoring_counts, sort_keys=True)
    out["inner_scoring"] = (
        next(iter(inner_scoring_counts))
        if len(inner_scoring_counts) == 1
        else "mixed"
        if inner_scoring_counts
        else None
    )
    out["selection_refit_performed_split_count"] = int(
        sum(bool(row.get("selection_refit_performed")) for row in ok)
    )
    selection_refit_values = {bool(row.get("selection_refit_performed")) for row in ok}
    out["selection_refit_performed"] = (
        next(iter(selection_refit_values)) if len(selection_refit_values) == 1 else None
    )
    out["retained_cv_ensemble_split_count"] = int(
        sum(bool(row.get("retained_cv_ensemble")) for row in ok)
    )
    retained_values = {bool(row.get("retained_cv_ensemble")) for row in ok}
    out["retained_cv_ensemble"] = next(iter(retained_values)) if len(retained_values) == 1 else None
    child_counts = [
        int(value) for row in ok if (value := row.get("cv_ensemble_child_count")) is not None
    ]
    out["cv_ensemble_child_count_mean"] = None if not child_counts else float(np.mean(child_counts))
    out["cv_ensemble_child_count_min"] = None if not child_counts else int(min(child_counts))
    out["cv_ensemble_child_count_max"] = None if not child_counts else int(max(child_counts))
    out["cv_ensemble_child_count"] = (
        child_counts[0] if child_counts and len(set(child_counts)) == 1 else None
    )
    candidate_fit_counts = [
        int(value) for row in ok if (value := row.get("candidate_fit_count")) is not None
    ]
    out["candidate_fit_count_mean"] = (
        None if not candidate_fit_counts else float(np.mean(candidate_fit_counts))
    )
    out["candidate_fit_count_total"] = int(sum(candidate_fit_counts))
    out["candidate_fit_count"] = (
        candidate_fit_counts[0]
        if candidate_fit_counts and len(set(candidate_fit_counts)) == 1
        else None
    )
    for field in ("preprocessing_fit_count", "preprocessing_reuse_count"):
        values = [int(value) for row in ok if (value := row.get(field)) is not None]
        out[f"{field}_mean"] = None if not values else float(np.mean(values))
        out[f"{field}_total"] = int(sum(values))
        out[field] = values[0] if values and len(set(values)) == 1 else None
    out["complexity_aggregation_json"] = json.dumps(
        {
            "across_retained_children": "sum",
            "across_outer_splits_model_units": "mean_of_repeat_means",
            "across_outer_test_rows_instance_units": "pooled_moments",
        },
        sort_keys=True,
    )
    error_count = int(sum(int(row.get("error_count", 0)) for row in split_rows))
    if len(ok) == official_splits_expected and error_count == 0:
        pair_status = "ok"
    elif ok:
        pair_status = "partial"
    else:
        pair_status = "error"
    out.update(
        {
            "task_id": int(task_meta["task_id"]),
            "dataset_id": int(task_meta["dataset_id"]),
            "dataset": str(task_meta["dataset_name"]),
            "model": model_name,
            "status": pair_status,
            "n_rows": int(task_meta["n_rows"]),
            "raw_features": int(task_meta["n_features"]),
            "model_features": (
                int(round(float(np.mean(model_feature_counts))))
                if model_feature_counts
                else int(task_meta["n_features"])
            ),
            "categorical_features": list(task_meta.get("categorical_features", [])),
            "n_classes": int(task_meta["n_classes"]),
            "class_labels": list(task_meta.get("class_labels", [])),
            "target_name": task_meta.get("target_name"),
            "official_repeats": int(task_meta["official_repeats"]),
            "official_folds": int(task_meta["official_folds"]),
            "official_samples": int(task_meta["official_samples"]),
            "official_splits_expected": int(official_splits_expected),
            "official_splits_completed": len(split_rows),
            "official_splits_successful": len(ok),
            "repeat_count_successful": len(repeat_groups),
            "repeat_metrics_aggregation": "mean_of_repeat_metrics",
            "error_count": error_count,
            "last_error": next(
                (
                    str(row.get("last_error"))
                    for row in reversed(split_rows)
                    if row.get("last_error")
                ),
                None,
            ),
            "evaluation_protocol": (
                "rotating_fold_train_validation_test_no_refit"
                if split_rows and split_rows[0].get("validation_protocol") == "rotating"
                else "tabarena_repeated_outer_3fold_inner_8fold_cv_ensemble_no_refit"
                if split_rows and split_rows[0].get("validation_protocol") == "tabarena"
                else (
                    "openml_defined_outer_splits_with_inner_cv_tuning"
                    if tune
                    else "openml_defined_outer_splits_no_inner_tuning"
                )
            ),
            "validation_protocol": (
                split_rows[0].get("validation_protocol", "nested") if split_rows else "nested"
            ),
            "early_stopping": bool(split_rows and split_rows[0].get("early_stopping")),
            "inner_n_splits": (
                int(inner_splits)
                if tune
                and not (split_rows and split_rows[0].get("validation_protocol") == "rotating")
                else None
            ),
            "scoring": (
                "roc_auc_binary_or_multiclass_log_loss"
                if split_rows and split_rows[0].get("validation_protocol") == "tabarena"
                else "roc_auc_binary_or_multiclass_ovr_macro"
            ),
            "tuned": bool(tune),
            "inner_tuning_fallback_split_count": int(
                sum(bool(row.get("inner_tuning_fallback")) for row in ok)
            ),
            "preprocessing_policy": (
                (
                    "fold-local AutoMLPipelineFeatureGenerator followed by HUGIML native handling"
                    if is_hugiml_model(model_name)
                    else "fold-local AutoMLPipelineFeatureGenerator followed by numeric imputation and categorical one-hot encoding"
                )
                if split_rows and split_rows[0].get("validation_protocol") == "tabarena"
                else (
                    "HUGIML native pandas categorical handling"
                    if is_hugiml_model(model_name)
                    else "fold-local numeric imputation and categorical one-hot encoding Pipeline"
                )
            ),
            "hugiml_scenario": hugiml_scenario if is_hugiml_model(model_name) else None,
            "split_rows": split_rows,
        }
    )
    return annotate_pair_timing(out)


def timing_inclusion_marker(row: dict[str, Any]) -> bool | None:
    """State whether the reported fit component is contained in tuning time."""
    explicit = row.get("fit_time_included_in_tuning")
    if isinstance(explicit, bool):
        return explicit
    tune_seconds = finite_or_none(row.get("tune_seconds"))
    if tune_seconds is None:
        return None
    return bool(row.get("tuned") and tune_seconds > 0.0)


def annotate_pair_timing(row: dict[str, Any]) -> dict[str, Any]:
    """Attach machine-readable timing-accounting fields to one pair result."""
    out = dict(row)
    split_rows = []
    for source in row.get("split_rows", []):
        split = dict(source)
        marker = timing_inclusion_marker(split)
        split["fit_time_included_in_tuning"] = marker
        split["fit_time_accounting"] = (
            "included_in_tuning"
            if marker is True
            else "separate_from_tuning"
            if marker is False
            else "unavailable"
        )
        split_rows.append(split)
    out["split_rows"] = split_rows
    successful = [split for split in split_rows if split.get("status") == "ok"]
    markers = [timing_inclusion_marker(split) for split in successful]
    known = [marker for marker in markers if marker is not None]
    included_count = sum(marker is True for marker in known)
    separate_count = sum(marker is False for marker in known)
    out["fit_time_included_in_tuning_split_count"] = int(included_count)
    out["fit_time_separate_from_tuning_split_count"] = int(separate_count)
    out["fit_time_accounting_unknown_split_count"] = int(len(markers) - len(known))
    if known and included_count == len(known) and len(known) == len(markers):
        aggregate_marker: bool | None = True
        accounting = "included_in_tuning"
    elif known and separate_count == len(known) and len(known) == len(markers):
        aggregate_marker = False
        accounting = "separate_from_tuning"
    elif known:
        aggregate_marker = None
        accounting = "mixed"
    else:
        aggregate_marker = None
        accounting = "unavailable"
    out["fit_time_included_in_tuning"] = aggregate_marker
    out["fit_time_accounting"] = accounting
    return out


def build_hugiml_distribution_rows(
    pair_results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-dataset RPTE distributions and preserve original route counts."""
    rows: list[dict[str, Any]] = []
    for pair in pair_results:
        if not is_hugiml_model(str(pair.get("model"))):
            continue
        successful = [
            dict(split) for split in pair.get("split_rows", []) if split.get("status") == "ok"
        ]
        rpte_observations = [
            split
            for split in successful
            if int(split.get("rpte_binary_estimator_count", 0) or 0) > 0
        ]
        output: dict[str, Any] = {
            "task_id": int(pair["task_id"]),
            "dataset": str(pair["dataset"]),
            "model": str(pair["model"]),
            "split_count": len(successful),
            "lr_split_count": sum(split.get("hugiml_path") == "lr" for split in successful),
            "rpte_split_count": len(rpte_observations),
            "rpte_sequential_estimator_count": sum(
                int(split.get("rpte_sequential_estimator_count", 0) or 0) for split in successful
            ),
            "rpte_lookahead_estimator_count": sum(
                int(split.get("rpte_lookahead_estimator_count", 0) or 0) for split in successful
            ),
        }
        for metric in RPTE_DISTRIBUTION_METRICS:
            for statistic, value in distribution_summary(
                row.get(metric) for row in rpte_observations
            ).items():
                output[f"{metric}_{statistic}"] = value
        rows.append(output)
    return rows


def run_pair(
    *,
    task_id: int,
    model_name: str,
    out_dir: Path,
    retries: int,
    tune: bool,
    inner_splits: int,
    random_state: int,
    max_official_splits: int | None,
    hugiml_scenario: str,
    hugiml_max_fit_seconds: float | None,
    run_id: str,
    resume: bool,
    validation_protocol: str = "nested",
    early_stopping: bool = False,
) -> dict[str, Any]:
    task, X, y, task_meta = load_task(task_id, retries=retries)
    raw_splits = official_splits(
        task,
        max_splits=max_official_splits if validation_protocol in {"nested", "tabarena"} else None,
    )
    splits = (
        rotating_validation_splits(raw_splits, len(y), max_splits=max_official_splits)
        if validation_protocol == "rotating"
        else raw_splits
    )
    identity = {
        "run_id": run_id,
        "task_id": int(task_id),
        "dataset_id": int(task_meta["dataset_id"]),
        "model": model_name,
        "tune": bool(tune),
        "inner_splits": int(inner_splits),
        "random_state": int(random_state),
        "max_official_splits": max_official_splits,
        "hugiml_scenario": hugiml_scenario if is_hugiml_model(model_name) else None,
        "official_dimensions": [
            int(task_meta["official_repeats"]),
            int(task_meta["official_folds"]),
            int(task_meta["official_samples"]),
        ],
        "dataset_sha256": safe_jsonable(task_meta.get("sha256", {})),
        "validation_protocol": validation_protocol,
        "early_stopping": bool(early_stopping),
    }
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    split_checkpoint = out_dir / "split_checkpoints" / f"{token}.json"
    split_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[int, dict[str, Any]] = {}
    feature_counts_by_split: dict[int, int] = {}
    predictions_by_split: dict[int, dict[str, Any]] = {}
    if resume and split_checkpoint.exists():
        try:
            saved = read_json_with_backup(split_checkpoint)
            if saved.get("identity") == identity:
                for row in saved.get("split_rows", []):
                    if row.get("split_id") is not None:
                        rows_by_split[int(row["split_id"])] = dict(row)
                raw_counts = saved.get("model_feature_counts", {})
                if isinstance(raw_counts, dict):
                    feature_counts_by_split = {
                        int(key): int(value) for key, value in raw_counts.items()
                    }
                raw_predictions = saved.get("split_predictions", {})
                if isinstance(raw_predictions, dict):
                    predictions_by_split = {
                        int(key): dict(value) for key, value in raw_predictions.items()
                    }
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            rows_by_split = {}
            feature_counts_by_split = {}
            predictions_by_split = {}

    successful = {
        split_id
        for split_id, row in rows_by_split.items()
        if row.get("status") == "ok" and split_id in predictions_by_split
    }
    for split in splits:
        split_id = int(split["split_id"])
        if split_id in successful:
            row = rows_by_split[split_id]
            print(
                f"  split r={split['repeat']} f={split['fold']} s={split['sample']} (reused)",
                flush=True,
            )
            print(
                json.dumps(
                    {
                        "split_id": row["split_id"],
                        "status": row["status"],
                        "roc_auc": row.get("roc_auc"),
                        "balanced_accuracy": row.get("balanced_accuracy"),
                        "error": row.get("last_error"),
                        "reused_split_checkpoint": True,
                    },
                    default=json_default,
                ),
                flush=True,
            )
            continue
        previous = rows_by_split.get(split_id, {})
        print(
            f"  split r={split['repeat']} f={split['fold']} s={split['sample']}",
            flush=True,
        )
        row, feature_count, prediction = evaluate_official_split(
            model_name=model_name,
            task_meta=task_meta,
            X=X,
            y=y,
            split=split,
            tune=tune,
            inner_splits=inner_splits,
            random_state=random_state,
            hugiml_scenario=hugiml_scenario,
            hugiml_max_fit_seconds=hugiml_max_fit_seconds,
            validation_protocol=validation_protocol,
            early_stopping=early_stopping,
        )
        row["attempt_count"] = int(previous.get("attempt_count", 0)) + 1
        row["attempted_at_utc"] = utc_now()
        rows_by_split[split_id] = row
        if feature_count is not None:
            feature_counts_by_split[split_id] = int(feature_count)
        if row.get("status") == "ok" and prediction is not None:
            predictions_by_split[split_id] = prediction
        elif row.get("status") != "ok":
            feature_counts_by_split.pop(split_id, None)
            predictions_by_split.pop(split_id, None)
        split_rows = [rows_by_split[key] for key in sorted(rows_by_split)]
        atomic_write_json(
            split_checkpoint,
            {
                "identity": identity,
                "task_meta": task_meta,
                "split_rows": split_rows,
                "model_feature_counts": {
                    str(key): feature_counts_by_split[key]
                    for key in sorted(feature_counts_by_split)
                },
                "split_predictions": {
                    str(key): predictions_by_split[key] for key in sorted(predictions_by_split)
                },
                "updated_at_utc": utc_now(),
            },
        )
        print(
            json.dumps(
                {
                    "split_id": row["split_id"],
                    "status": row["status"],
                    "roc_auc": row.get("roc_auc"),
                    "balanced_accuracy": row.get("balanced_accuracy"),
                    "error": row.get("last_error"),
                },
                default=json_default,
            ),
            flush=True,
        )

    split_rows = [rows_by_split[key] for key in sorted(rows_by_split)]
    return aggregate_pair_rows(
        split_rows,
        split_predictions=predictions_by_split,
        task_meta=task_meta,
        model_name=model_name,
        model_feature_counts=[
            feature_counts_by_split[key] for key in sorted(feature_counts_by_split)
        ],
        tune=tune,
        inner_splits=inner_splits,
        hugiml_scenario=hugiml_scenario,
        official_splits_expected=len(splits),
    )


def grid_snapshot(
    hugiml_scenario: str,
    models: Sequence[str] | None = None,
    *,
    validation_protocol: str = "nested",
) -> dict[str, Any]:
    selected = list(models or MODEL_ORDER)
    snapshot: dict[str, Any] = {}
    for hugiml_model in HUGIML_MODELS:
        if hugiml_model not in selected:
            continue
        hug_grid_name, hug_candidates = benchmark_base._hugiml_grid_for_model(
            hugiml_model, hugiml_scenario
        )
        hug_grid = benchmark_base.get_hugiml_grid(hug_grid_name)
        snapshot[hugiml_model] = {
            "scenario": hugiml_scenario,
            "grid_name": hug_grid_name,
            "candidate_count": len(hug_candidates),
            "grid": hug_grid,
        }
    families = {
        "XGB standard": "XGBoost",
        "LightGBM standard": "LightGBM",
        "RandomForest standard": "RandomForest",
        "Logistic Regression": "LogisticRegression",
        "EBM": "EBM",
        "RuleFit": "RuleFit",
    }
    for label, family in families.items():
        if label not in selected:
            continue
        grid = (
            rotating_baseline_grid(family)
            if validation_protocol in {"rotating", "tabarena"}
            else benchmark_base.get_baseline_grid(family)
        ) or {}
        snapshot[label] = {
            "candidate_count": len(list(benchmark_base.ParameterGrid(grid))) if grid else 1,
            "grid": grid,
            "constants": benchmark_base.baseline_constant_parameters(family),
        }
    return safe_jsonable(snapshot)


def configuration_id(configuration: dict[str, Any]) -> str:
    canonical = json.dumps(
        safe_jsonable(configuration), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def task_signature(task: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "task_id",
        "dataset_id",
        "dataset_name",
        "n_rows",
        "n_features",
        "n_classes",
        "official_repeats",
        "official_folds",
        "official_samples",
        "official_split_count",
        "sha256",
    )
    return {key: safe_jsonable(task.get(key)) for key in keys if key in task}


def new_checkpoint() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "active_run_id": None,
        "runs": {},
        "metadata": {},
        "selected_tasks": [],
        "results": [],
    }


def normalize_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("metadata", {})
    payload.setdefault("selected_tasks", [])
    payload.setdefault("results", [])
    payload.setdefault("runs", {})
    if not isinstance(payload["results"], list) or not isinstance(payload["runs"], dict):
        raise ValueError("Checkpoint structure is invalid")
    if not payload["runs"] and (
        payload["metadata"] or payload["selected_tasks"] or payload["results"]
    ):
        configuration = {
            "metadata": safe_jsonable(payload["metadata"]),
            "task_ids": [
                int(row["task_id"])
                for row in payload["selected_tasks"]
                if row.get("task_id") is not None
            ],
        }
        run_id = configuration_id(configuration)
        payload["runs"][run_id] = {
            "run_id": run_id,
            "created_at_utc": payload["metadata"].get("created_at_utc", utc_now()),
            "updated_at_utc": payload["metadata"].get("updated_at_utc", utc_now()),
            "configuration": configuration,
            "metadata": dict(payload["metadata"]),
            "selected_tasks": list(payload["selected_tasks"]),
        }
        for row in payload["results"]:
            row.setdefault("run_id", run_id)
            if "status" not in row:
                expected = int(row.get("official_splits_completed", 0) or 0)
                successful = int(row.get("official_splits_successful", 0) or 0)
                row["official_splits_expected"] = expected
                row["status"] = (
                    "ok"
                    if expected > 0 and expected == successful and not row.get("error_count")
                    else "partial"
                )
        payload["active_run_id"] = run_id
    payload["schema_version"] = 2
    if payload.get("active_run_id") in payload["runs"]:
        run = payload["runs"][payload["active_run_id"]]
        payload["metadata"] = dict(run.get("metadata", {}))
        payload["selected_tasks"] = list(run.get("selected_tasks", []))
    return payload


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists() and not path.with_suffix(path.suffix + ".bak").exists():
        return new_checkpoint()
    return normalize_checkpoint(read_json_with_backup(path))


def activate_run(
    payload: dict[str, Any],
    *,
    configuration: dict[str, Any],
    metadata: dict[str, Any],
    selected_tasks: list[dict[str, Any]],
) -> str:
    run_id = configuration_id(configuration)
    now = utc_now()
    existing = payload["runs"].get(run_id, {})
    run = {
        "run_id": run_id,
        "created_at_utc": existing.get("created_at_utc", now),
        "updated_at_utc": now,
        "configuration": safe_jsonable(configuration),
        "metadata": {**metadata, "run_id": run_id, "updated_at_utc": now},
        "selected_tasks": safe_jsonable(selected_tasks),
    }
    payload["runs"][run_id] = run
    payload["active_run_id"] = run_id
    payload["metadata"] = dict(run["metadata"])
    payload["selected_tasks"] = list(run["selected_tasks"])
    return run_id


def update_active_run_metadata(payload: dict[str, Any], **values: Any) -> None:
    run_id = active_run_id(payload)
    run = payload["runs"][run_id]
    run.setdefault("metadata", {}).update(values)
    run["updated_at_utc"] = values.get("updated_at_utc", utc_now())
    payload["metadata"] = dict(run["metadata"])


# ---------------------------------------------------------------------------
# Dashboard data and rendering
# ---------------------------------------------------------------------------


def _holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * float(p_values[idx]))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def model_inspection_value(row: dict[str, Any] | pd.Series) -> float | None:
    primary = finite_or_none(row.get("complexity_model_inspection_units"))
    return primary if primary is not None else finite_or_none(row.get("complexity"))


def complete_matched_task_ids(rows: Sequence[dict[str, Any]], models: Sequence[str]) -> set[int]:
    """Return tasks with one successful, complete result for every model."""
    complete_by_task: dict[int, set[str]] = {}
    recognized = set(str(model) for model in models)
    for row in rows:
        model = str(row.get("model"))
        if model not in recognized or str(row.get("status", "ok")) != "ok":
            continue
        expected = int(
            row.get("official_splits_expected") or row.get("official_splits_completed") or 0
        )
        successful = int(
            row.get("official_splits_successful") or row.get("official_splits_completed") or 0
        )
        if expected > 0 and successful < expected:
            continue
        if int(row.get("error_count") or 0) > 0:
            continue
        complete_by_task.setdefault(int(row["task_id"]), set()).add(model)
    return {
        task_id
        for task_id, completed_models in complete_by_task.items()
        if completed_models == recognized
    }


def build_dashboard_data(payload: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    view = run_view(normalize_checkpoint(payload), run_id)
    results = [annotate_pair_timing(row) for row in view.get("results", [])]
    pair_df = pd.DataFrame(results)
    if pair_df.empty:
        raise ValueError("Checkpoint contains no completed model/task results")
    configured_models = view.get("metadata", {}).get("models") or []
    active_models = [model for model in MODEL_ORDER if model in configured_models]
    if not active_models:
        present_models = set(str(model) for model in pair_df.get("model", []))
        active_models = [model for model in MODEL_ORDER if model in present_models]
    pair_df = pair_df[pair_df["model"].isin(active_models)].copy()
    if pair_df.empty:
        raise ValueError("Checkpoint contains no recognized model/task results")
    pair_df["roc_auc"] = pd.to_numeric(pair_df["roc_auc"], errors="coerce")

    task_order = [int(row["task_id"]) for row in view.get("selected_tasks", [])]
    if task_order:
        pair_df = pair_df[pair_df["task_id"].isin(task_order)].copy()
    matched_task_ids = complete_matched_task_ids(results, active_models)
    pair_df = pair_df[pair_df["task_id"].isin(matched_task_ids)].copy()
    task_order = [task_id for task_id in task_order if task_id in matched_task_ids]
    if pair_df.empty:
        raise ValueError("Checkpoint run contains no fully matched completed tasks")
    task_names = {
        int(row["task_id"]): str(row.get("dataset_name", f"task-{row['task_id']}"))
        for row in view.get("selected_tasks", [])
    }
    per_task: list[dict[str, Any]] = []
    for task_id in task_order:
        subset = pair_df[pair_df["task_id"] == task_id].copy()
        if subset.empty:
            continue
        valid = subset["roc_auc"].notna() & subset.get(
            "status", pd.Series("ok", index=subset.index)
        ).eq("ok")
        ranks: dict[str, float | None] = {m: None for m in active_models}
        if valid.any():
            rank_vals = rankdata(
                -subset.loc[valid, "roc_auc"].astype(float).to_numpy(), method="average"
            )
            for model, rank in zip(subset.loc[valid, "model"], rank_vals):
                ranks[str(model)] = float(rank)
        rows: list[dict[str, Any]] = []
        for model in active_models:
            match = subset[subset["model"] == model]
            if match.empty:
                continue
            row = match.iloc[0].to_dict()
            row["rank"] = ranks[model]
            row["model_inspection_units"] = model_inspection_value(row)
            row.pop("split_rows", None)
            rows.append(safe_jsonable(row))
        per_task.append(
            {
                "task_id": task_id,
                "dataset": task_names.get(task_id, str(subset.iloc[0].get("dataset", task_id))),
                "n_rows": int(subset.iloc[0].get("n_rows", 0)),
                "n_classes": int(subset.iloc[0].get("n_classes", 0)),
                "rows": rows,
            }
        )

    aggregate_rows: list[dict[str, Any]] = []
    for model in active_models:
        model_subset = pair_df[pair_df["model"] == model]
        status_series = model_subset.get("status", pd.Series("ok", index=model_subset.index))
        subset = model_subset[status_series.eq("ok")].copy()
        ranks = [
            row["rank"]
            for task in per_task
            for row in task["rows"]
            if row["model"] == model and row.get("rank") is not None
        ]
        aucs = pd.to_numeric(
            subset.get("roc_auc", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        inspection_values = [model_inspection_value(row) for _, row in subset.iterrows()]
        model_units = [finite_or_none(value) for value in subset.get("complexity_model_units", [])]
        instance_values = [
            finite_or_none(value)
            for value in subset.get("complexity_instance_inspection_units_mean", [])
        ]
        instance_ci = series_mean_ci(instance_values)
        wins = sum(
            1
            for task in per_task
            for row in task["rows"]
            if row["model"] == model and row.get("rank") == 1.0
        )
        valid_inspection = [value for value in inspection_values if value is not None]
        valid_model_units = [value for value in model_units if value is not None]
        aggregate_rows.append(
            {
                "model": model,
                "tasks_attempted": int(len(model_subset)),
                "tasks_completed": int(len(aucs)),
                "tasks_failed_or_partial": int(len(model_subset) - len(aucs)),
                "mean_auc": None if aucs.empty else float(aucs.mean()),
                "std_auc": None if len(aucs) < 2 else float(aucs.std(ddof=1)),
                "mean_rank": None if not ranks else float(np.mean(ranks)),
                "wins": int(wins),
                "mean_balanced_accuracy": mean_or_none(subset.get("balanced_accuracy", [])),
                "mean_f1": mean_or_none(subset.get("f1", [])),
                "mean_brier": mean_or_none(subset.get("brier", [])),
                "mean_fit_seconds": mean_or_none(subset.get("fit_seconds", [])),
                "mean_tune_seconds": mean_or_none(subset.get("tune_seconds", [])),
                "mean_predict_seconds": mean_or_none(subset.get("predict_seconds", [])),
                "mean_complexity_seconds": mean_or_none(subset.get("complexity_seconds", [])),
                "mean_split_seconds": mean_or_none(subset.get("split_seconds", [])),
                "mean_evaluation_overhead_seconds": mean_or_none(
                    subset.get("evaluation_overhead_seconds", [])
                ),
                "mean_complexity": None
                if not valid_inspection
                else float(np.mean(valid_inspection)),
                "mean_model_units": None
                if not valid_model_units
                else float(np.mean(valid_model_units)),
                "median_model_units": None
                if not valid_model_units
                else float(np.median(valid_model_units)),
                "mean_model_inspection_units": None
                if not valid_inspection
                else float(np.mean(valid_inspection)),
                "median_model_inspection_units": None
                if not valid_inspection
                else float(np.median(valid_inspection)),
                "mean_instance_inspection_units": instance_ci.get("mean"),
                "instance_inspection_ci_lower": instance_ci.get("ci_lower"),
                "instance_inspection_ci_upper": instance_ci.get("ci_upper"),
                "instance_inspection_confidence_level": instance_ci.get("confidence_level"),
                "instance_inspection_n_tasks": instance_ci.get("n_samples"),
            }
        )

    complete_tasks: list[dict[str, Any]] = []
    for task in per_task:
        scores = {row["model"]: row.get("roc_auc") for row in task["rows"]}
        if len(active_models) >= 2 and all(
            finite_or_none(scores.get(model)) is not None for model in active_models
        ):
            complete_tasks.append(task)
    friedman = {"n_tasks": len(complete_tasks), "statistic": None, "p_value": None}
    if len(active_models) >= 3 and len(complete_tasks) >= 2:
        arrays = [
            [
                float(next(row["roc_auc"] for row in task["rows"] if row["model"] == model))
                for task in complete_tasks
            ]
            for model in active_models
        ]
        try:
            stat, p_value = friedmanchisquare(*arrays)
            friedman.update({"statistic": finite_or_none(stat), "p_value": finite_or_none(p_value)})
        except Exception:
            pass

    versus_hugiml: list[dict[str, Any]] = []
    raw_p: list[float] = []
    active_hugiml_models = [model for model in active_models if is_hugiml_model(model)]
    comparison_models = [model for model in active_models if not is_hugiml_model(model)]
    primary_hugiml_model = active_hugiml_models[0] if active_hugiml_models else None
    for hugiml_model in active_hugiml_models:
        for model in comparison_models:
            pairs: list[tuple[float, float]] = []
            for task in per_task:
                scores = {row["model"]: row.get("roc_auc") for row in task["rows"]}
                hug_auc = finite_or_none(scores.get(hugiml_model))
                baseline_auc = finite_or_none(scores.get(model))
                if hug_auc is not None and baseline_auc is not None:
                    pairs.append((hug_auc, baseline_auc))
            p_value = None
            statistic = None
            if len(pairs) >= 2 and any(abs(a - b) > 1e-15 for a, b in pairs):
                try:
                    statistic, p_value = wilcoxon([a for a, _ in pairs], [b for _, b in pairs])
                    statistic, p_value = finite_or_none(statistic), finite_or_none(p_value)
                except Exception:
                    pass
            versus_hugiml.append(
                {
                    "comparison": f"{hugiml_model} vs {model}",
                    "hugiml_model": hugiml_model,
                    "model": model,
                    "n_tasks": len(pairs),
                    "mean_auc_difference_hugiml_minus_baseline": (
                        None if not pairs else float(np.mean([a - b for a, b in pairs]))
                    ),
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )
            raw_p.append(1.0 if p_value is None else float(p_value))
    adjusted = _holm_adjust(raw_p) if raw_p else []
    for row, p_adj in zip(versus_hugiml, adjusted):
        row["p_value_holm"] = float(p_adj)
        row["significant_holm_0_05"] = bool(p_adj < 0.05 and row.get("p_value") is not None)

    auc_tolerance = float(view.get("metadata", {}).get("complexity_auc_tolerance", 0.01))
    complexity_points: list[dict[str, Any]] = []
    complexity_ratios: list[dict[str, Any]] = []
    for task in per_task:
        by_model = {row["model"]: row for row in task["rows"]}
        for row in task["rows"]:
            auc = finite_or_none(row.get("roc_auc"))
            inspection = model_inspection_value(row)
            if auc is None or inspection is None or inspection <= 0:
                continue
            complexity_points.append(
                {
                    "task_id": task["task_id"],
                    "dataset": task["dataset"],
                    "model": row["model"],
                    "roc_auc": auc,
                    "model_units": finite_or_none(row.get("complexity_model_units")),
                    "model_inspection_units": inspection,
                    "instance_inspection_units_mean": finite_or_none(
                        row.get("complexity_instance_inspection_units_mean")
                    ),
                    "instance_inspection_units_ci_lower": finite_or_none(
                        row.get("complexity_instance_inspection_units_ci_lower")
                    ),
                    "instance_inspection_units_ci_upper": finite_or_none(
                        row.get("complexity_instance_inspection_units_ci_upper")
                    ),
                }
            )
        for hugiml_model in active_hugiml_models:
            hug = by_model.get(hugiml_model)
            if hug is None:
                continue
            hug_auc = finite_or_none(hug.get("roc_auc"))
            hug_inspection = model_inspection_value(hug)
            if hug_auc is None or hug_inspection is None or hug_inspection <= 0:
                continue
            for model in comparison_models:
                baseline = by_model.get(model)
                if baseline is None:
                    continue
                baseline_auc = finite_or_none(baseline.get("roc_auc"))
                baseline_inspection = model_inspection_value(baseline)
                if baseline_auc is None or baseline_inspection is None or baseline_inspection <= 0:
                    continue
                auc_difference = hug_auc - baseline_auc
                comparable = auc_difference >= -auc_tolerance
                ratio = baseline_inspection / hug_inspection
                complexity_ratios.append(
                    {
                        "task_id": task["task_id"],
                        "dataset": task["dataset"],
                        "hugiml_model": hugiml_model,
                        "baseline_model": model,
                        "hugiml_auc": hug_auc,
                        "baseline_auc": baseline_auc,
                        "auc_difference_hugiml_minus_baseline": auc_difference,
                        "auc_tolerance": auc_tolerance,
                        "hugiml_model_inspection_units": hug_inspection,
                        "baseline_model_inspection_units": baseline_inspection,
                        "inspection_ratio_baseline_to_hugiml": ratio,
                        "hugiml_reduction_percent": 100.0 * (1.0 - 1.0 / ratio),
                        "comparable_or_better_auc": comparable,
                    }
                )

    complexity_summary: list[dict[str, Any]] = []
    for hugiml_model in active_hugiml_models:
        for model in comparison_models:
            rows = [
                row
                for row in complexity_ratios
                if row["hugiml_model"] == hugiml_model and row["baseline_model"] == model
            ]
            comparable = [row for row in rows if row["comparable_or_better_auc"]]
            ratios = [float(row["inspection_ratio_baseline_to_hugiml"]) for row in comparable]
            hug_units = [float(row["hugiml_model_inspection_units"]) for row in comparable]
            baseline_units = [float(row["baseline_model_inspection_units"]) for row in comparable]
            hug_auc_values = [float(row["hugiml_auc"]) for row in comparable]
            baseline_auc_values = [float(row["baseline_auc"]) for row in comparable]
            median_ratio = None if not ratios else float(np.median(ratios))
            summary = {
                "hugiml_model": hugiml_model,
                "baseline_model": model,
                "paired_tasks": len(rows),
                "comparable_or_better_tasks": len(comparable),
                "comparable_or_better_share": None
                if not rows
                else float(len(comparable) / len(rows)),
                "auc_tolerance": auc_tolerance,
                "mean_hugiml_auc": None if not hug_auc_values else float(np.mean(hug_auc_values)),
                "mean_baseline_auc": None
                if not baseline_auc_values
                else float(np.mean(baseline_auc_values)),
                "median_hugiml_model_inspection_units": None
                if not hug_units
                else float(np.median(hug_units)),
                "median_baseline_model_inspection_units": None
                if not baseline_units
                else float(np.median(baseline_units)),
                "median_inspection_ratio_baseline_to_hugiml": median_ratio,
                "mean_inspection_ratio_baseline_to_hugiml": None
                if not ratios
                else float(np.mean(ratios)),
                "median_hugiml_reduction_percent": None
                if median_ratio is None
                else 100.0 * (1.0 - 1.0 / median_ratio),
            }
            if comparable and median_ratio is not None:
                summary["headline"] = (
                    f"Across {len(comparable)} tasks where {hugiml_model} AUC was within "
                    f"{auc_tolerance:.3f} of or above {model}, mean AUC was "
                    f"{np.mean(hug_auc_values):.3f} versus {np.mean(baseline_auc_values):.3f}, "
                    f"with median model inspection units {np.median(hug_units):.1f} versus "
                    f"{np.median(baseline_units):.1f} ({median_ratio:.2f}x baseline-to-HUGIML ratio)."
                )
            else:
                summary["headline"] = None
            complexity_summary.append(summary)

    for task in per_task:
        primary_ratios = {
            row["baseline_model"]: row
            for row in complexity_ratios
            if int(row["task_id"]) == int(task["task_id"])
            and row["hugiml_model"] == primary_hugiml_model
        }
        for row in task["rows"]:
            if row["model"] == primary_hugiml_model:
                row["inspection_ratio_to_hugiml"] = 1.0
                row["comparable_or_better_auc_to_hugiml"] = True
            elif row["model"] in primary_ratios:
                ratio_row = primary_ratios[row["model"]]
                row["inspection_ratio_to_hugiml"] = ratio_row["inspection_ratio_baseline_to_hugiml"]
                row["comparable_or_better_auc_to_hugiml"] = ratio_row["comparable_or_better_auc"]
    selected_task_ids = set(task_order)
    filtered_results = [
        row
        for row in results
        if not selected_task_ids or int(row.get("task_id", -1)) in selected_task_ids
    ]
    split_rows: list[dict[str, Any]] = []
    for result in filtered_results:
        for split in result.get("split_rows", []):
            split_rows.append(dict(split))
    dashboard_tasks = [
        dict(task)
        for task in view.get("selected_tasks", [])
        if int(task["task_id"]) in selected_task_ids
    ]
    dashboard_metadata = copy.deepcopy(view.get("metadata", {}))
    dashboard_metadata["selected_task_count"] = len(dashboard_tasks)
    dashboard_metadata["selection_description"] = (
        f"{len(dashboard_tasks)} fully matched completed tasks"
    )
    dashboard_metadata["assembly_scope"] = "fully_matched_completed_tasks"

    return safe_jsonable(
        {
            "run_id": view["run_id"],
            "metadata": dashboard_metadata,
            "selected_tasks": dashboard_tasks,
            "model_order": active_models,
            "model_families": MODEL_FAMILIES,
            "overall": aggregate_rows,
            "per_task": per_task,
            "friedman": friedman,
            "versus_hugiml": versus_hugiml,
            "primary_hugiml_model": primary_hugiml_model,
            "complexity_auc_tolerance": auc_tolerance,
            "complexity_points": complexity_points,
            "complexity_ratios": complexity_ratios,
            "complexity_summary": complexity_summary,
            "hugiml_rpte_distributions": build_hugiml_distribution_rows(filtered_results),
            "pair_results": [
                {k: v for k, v in row.items() if k != "split_rows"} for row in filtered_results
            ],
            "split_result_count": len(split_rows),
        }
    )


DEFAULT_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{TITLE}}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#0b1020;--panel:#121a31;--panel2:#18223d;--ink:#edf2ff;--muted:#aeb9d4;--border:#2a3658;--accent:#82aaff;--accent2:#86efcf;--grid:rgba(255,255,255,.10);--good:#86efcf;--warn:#ffd166;--danger:#ff8fab}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;background:radial-gradient(circle at 16% 0%,color-mix(in srgb,var(--accent) 18%,transparent),transparent 34%),linear-gradient(180deg,var(--bg),color-mix(in srgb,var(--bg) 92%,black));color:var(--ink);line-height:1.5}
.wrap{max-width:1520px;margin:0 auto;padding:24px}
.hero{padding:24px 28px;background:linear-gradient(135deg,color-mix(in srgb,var(--accent) 16%,var(--panel)),color-mix(in srgb,var(--accent2) 10%,var(--panel)));border:1px solid var(--border);border-radius:22px;box-shadow:0 14px 34px rgba(0,0,0,.18)}
.topline{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap}
h1{margin:0 0 10px;font-size:34px;line-height:1.1;letter-spacing:-.02em}
h2{margin:0;font-size:20px}
h3{margin:0;font-size:16px}
p{color:var(--muted);line-height:1.55}
.hero p{max-width:900px;margin:0;color:color-mix(in srgb,var(--ink) 80%,var(--muted))}
.generated{margin-top:8px;font-size:13px;color:var(--muted)}
.theme-switcher{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end;margin-left:auto}
.theme-btn{border:1px solid var(--border);background:var(--panel2);color:var(--ink);border-radius:999px;padding:8px 12px;cursor:pointer;font-weight:650;min-width:74px}
.theme-btn.active{outline:2px solid var(--accent);background:color-mix(in srgb,var(--accent) 18%,var(--panel2))}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}
.chip{padding:6px 10px;border-radius:999px;background:color-mix(in srgb,var(--panel2) 72%,transparent);border:1px solid var(--border);font-size:13px}
.card{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--border);border-radius:18px;padding:18px;box-shadow:0 8px 22px rgba(0,0,0,.13)}
.summary-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin-top:18px}
.metric-comparison-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:14px}
.metric-comparison .value{display:flex;align-items:baseline;gap:9px}
.metric-comparison .winner{color:var(--accent2);font-size:14px;font-weight:700}
.metric-ranking{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.metric-rank{background:var(--panel2);border:1px solid var(--border);border-radius:999px;color:var(--muted);font-size:11px;padding:4px 7px}
.metric-rank.hugiml{border-color:var(--accent2);color:var(--ink);font-weight:700}
.stat .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.055em}
.stat .value{font-size:29px;font-weight:760;margin-top:8px}
.stat .value.small{font-size:20px}
.stat .sub{color:var(--muted);font-size:13px;margin-top:8px;line-height:1.42}
.grid-2{display:grid;grid-template-columns:minmax(0,1.18fr) minmax(360px,.82fr);gap:16px;margin-top:16px}
.section{margin-top:16px}
.section-title{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin:0 0 12px}
.meta,.note{color:var(--muted);font-size:13px}
.meta{text-align:right}
.plot{width:100%;min-height:450px;overflow:visible}
.table-wrap{max-height:520px;overflow:auto;border:1px solid var(--border);border-radius:14px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:right;color:color-mix(in srgb,var(--ink) 88%,var(--muted));padding:10px;border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--panel);z-index:2;white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid color-mix(in srgb,var(--border) 60%,transparent);vertical-align:top;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
tr:hover{background:color-mix(in srgb,var(--accent) 9%,transparent)}
.best{font-weight:760;color:var(--good)}
.error{color:var(--danger)}
.pill{display:inline-block;padding:4px 9px;border-radius:999px;background:color-mix(in srgb,var(--accent) 15%,var(--panel2));border:1px solid color-mix(in srgb,var(--accent) 35%,var(--border));color:var(--ink)}
.claim{padding:13px 15px;border-radius:14px;background:color-mix(in srgb,var(--accent2) 10%,var(--panel));border:1px solid color-mix(in srgb,var(--accent2) 40%,var(--border));margin:9px 0;color:var(--ink)}
.overall-tables-grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,.8fr);gap:18px;margin-top:12px}
.subsection-label{font-size:0.78rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em;margin:0 0 8px}
.claims-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:18px}
.claims-grid .claim{margin:0;font-size:0.87rem;line-height:1.58}
.ratio-good{color:var(--good);font-weight:700}.ratio-warn{color:var(--warn)}
.dataset-metric-wrap{max-height:720px}
.dataset-metric-table{min-width:1500px}
.dataset-metric-table th.metric-group-head{text-align:center;font-size:13px;color:var(--ink);background:color-mix(in srgb,var(--accent) 12%,var(--panel))}
.dataset-metric-table .group-start{border-left:2px solid color-mix(in srgb,var(--accent) 55%,var(--border))}
.dataset-metric-table th.dataset-sticky,.dataset-metric-table td.dataset-sticky{position:sticky;left:0;text-align:left;background:var(--panel);z-index:3}
.dataset-metric-table th.task-sticky,.dataset-metric-table td.task-sticky{position:sticky;left:190px;text-align:right;background:var(--panel);z-index:3}
.dataset-metric-table th.dataset-sticky{z-index:5;min-width:190px}
.dataset-metric-table th.task-sticky{z-index:5;min-width:76px}
.dataset-metric-table td.dataset-sticky{font-weight:650;min-width:190px}
.dataset-metric-table td.task-sticky{min-width:76px;color:var(--muted)}
.dataset-metric-table tbody tr:hover td.dataset-sticky,.dataset-metric-table tbody tr:hover td.task-sticky{background:color-mix(in srgb,var(--accent) 9%,var(--panel))}
.controls{display:grid;grid-template-columns:minmax(240px,1fr);gap:8px;max-width:620px}
label{font-size:13px;color:var(--muted)}
select{width:100%;background:var(--panel2);color:var(--ink);border:1px solid var(--border);border-radius:12px;padding:10px 12px;outline:none}
.kv{display:grid;grid-template-columns:180px 1fr;gap:8px;font-size:13px}
.kv div{padding:7px 0;border-bottom:1px dashed color-mix(in srgb,var(--border) 70%,transparent)}
.paired-row{padding:11px 0;border-bottom:1px solid color-mix(in srgb,var(--border) 70%,transparent)}
.badge{display:inline-block;margin-left:7px;padding:2px 7px;border-radius:999px;font-size:11px;font-weight:700;background:color-mix(in srgb,var(--accent2) 15%,var(--panel2));border:1px solid color-mix(in srgb,var(--accent2) 38%,var(--border))}
details{border:1px solid var(--border);border-radius:14px;background:var(--panel2)}
summary{cursor:pointer;padding:12px 14px;font-weight:700}
pre{margin:0;border-top:1px solid var(--border);white-space:pre-wrap;word-break:break-word;background:color-mix(in srgb,var(--bg) 84%,black);color:var(--ink);padding:14px;max-height:420px;overflow:auto;border-radius:0 0 14px 14px;font-size:12px}
.methodology-content{padding:2px 14px 16px}.methodology-content h3{margin:18px 0 8px;font-size:16px}.methodology-content ul{margin:8px 0;padding-left:22px}.methodology-content li{margin:5px 0;color:var(--muted)}
.methodology-model-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:12px}.methodology-model{border:1px solid var(--border);border-radius:12px;padding:14px;background:var(--panel)}.methodology-model h4{margin:0 0 5px}.methodology-model-meta{font-size:12px;color:var(--muted);margin-bottom:10px}.methodology-table{width:100%;font-size:12px}.methodology-table th,.methodology-table td{text-align:left;position:static;background:transparent;padding:6px 7px;white-space:normal}.methodology-table th:first-child{width:38%}.methodology-model p{font-size:12px;line-height:1.5;margin:10px 0 0;color:var(--muted)}
.empty{color:var(--muted);font-style:italic}
@media(max-width:1180px){.summary-grid,.metric-comparison-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.grid-2{grid-template-columns:1fr}.overall-tables-grid{grid-template-columns:1fr}.claims-grid{grid-template-columns:1fr}.section-title{align-items:flex-start;flex-direction:column}.meta{text-align:left}}
@media(max-width:760px){.wrap{padding:14px}.summary-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.metric-comparison-grid{grid-template-columns:1fr}h1{font-size:28px}.hero{padding:22px}.theme-switcher{width:100%;justify-content:flex-start}.kv{grid-template-columns:1fr}.plot{min-height:390px}.methodology-model-grid{grid-template-columns:1fr}}
.js-plotly-plot .scatterlayer .trace path{vector-effect:non-scaling-stroke}
</style>
</head>
<body><main class="wrap">
<header class="hero">
  <div class="topline">
    <div>
      <h1>{{TITLE}}</h1>
      <p>Official OpenML train/test indices, fold-local model selection, predictive performance, and model-inspection evidence.</p>
    </div>
    <div class="theme-switcher" aria-label="Dashboard theme">
      <button class="theme-btn active" data-theme="indigo" type="button">Indigo</button>
      <button class="theme-btn" data-theme="pearl" type="button">Pearl</button>
      <button class="theme-btn" data-theme="forest" type="button">Forest</button>
    </div>
  </div>
  <div id="heroChips" class="chips"></div>
  <section id="metrics" class="summary-grid"></section>
  <section id="metricComparisons" class="metric-comparison-grid"></section>
</header>

<section class="section card">
  <div class="section-title"><h2>Overall model comparison</h2><div class="meta">Performance, ranks, inspection efficiency, and runtime</div></div>
  <div class="overall-tables-grid">
    <div>
      <p class="subsection-label">Performance, ranks &amp; runtime</p>
      <div class="table-wrap"><table id="overallTable"></table></div>
    </div>
    <div>
      <p class="subsection-label">HUGIML inspection efficiency — baseline / HUGIML</p>
      <p class="note" style="margin:0 0 8px">Ratios are baseline model inspection units divided by each HUGIML variant?s model inspection units, on tasks where that variant?s AUC is within tolerance.</p>
      <div class="table-wrap"><table id="ratioTable"></table></div>
    </div>
  </div>
  <div id="claims" class="claims-grid"></div>
</section>

<section class="section card">
  <div class="section-title"><h2>Model inspection units versus AUC</h2><div class="meta">One point per completed task/model pair</div></div>
  <p class="note">The x-axis is logarithmic when every displayed inspection value is positive.</p>
  <div id="complexityPlot" class="plot"></div><div id="plotFallback" class="note"></div>
</section>

<section class="section card">
  <div class="section-title"><h2>Dataset explorer</h2><div class="meta">Inspect paired results for one OpenML task</div></div>
  <div class="controls"><label for="taskSelect">Dataset</label><select id="taskSelect"></select></div>
  <div id="taskMeta" class="note" style="margin-top:10px"></div>
  <div class="table-wrap" style="margin-top:12px"><table id="taskTable"></table></div>
</section>

<section class="section card" id="datasetMetricComparison">
  <div class="section-title"><h2>Dataset-level metric comparison</h2><div class="meta">Three measures × all models</div></div>
  <p class="note">Each row compares ROC-AUC, balanced accuracy, and F1 for the same OpenML task. <strong>Bold</strong> marks the best score; <em>italics</em> mark the second-best score. Ranking is based on the displayed four-decimal values, so displayed ties receive identical styling.</p>
  <div class="table-wrap dataset-metric-wrap">
    <table id="datasetMetricTable" class="dataset-metric-table"></table>
  </div>
</section>

{{RPTE_SECTION}}

<section class="grid-2">
  <div class="card"><div class="section-title"><h2>HUGIML variant paired tests</h2><div class="meta">Wilcoxon tests with Holm adjustment</div></div><div id="paired"></div></div>
  <div class="card"><div class="section-title"><h2>Evaluation protocol</h2><div class="meta">Run configuration</div></div><div id="protocol" class="kv"></div></div>
</section>

<section class="section card">
  <details id="benchmarkMethodology">
    <summary>Methodology and parameter search space</summary>
    <div class="methodology-content">
      <h3>Cross-validation protocol</h3><ul id="methodologyProtocol"></ul>
      <h3>Preprocessing and timing</h3><ul id="methodologyPreprocessing"></ul>
      <h3>Complexity unit definitions</h3><ul id="methodologyComplexity"></ul>
      <h3>Model search spaces</h3><div id="methodologyModels" class="methodology-model-grid"></div>
    </div>
  </details>
</section>
</main>
<script>
const DATA={{DATA_JSON}};
const THEMES={
 indigo:{bg:'#0b1020',panel:'#121a31',panel2:'#18223d',ink:'#edf2ff',muted:'#aeb9d4',border:'#2a3658',accent:'#82aaff',accent2:'#86efcf',grid:'rgba(255,255,255,.10)',good:'#86efcf',warn:'#ffd166',danger:'#ff8fab',colors:['#82aaff','#9ad7ff','#86efcf','#d6b4ff','#ffd166','#ff8fab','#a3e635']},
 pearl:{bg:'#f7f8fb',panel:'#ffffff',panel2:'#f1f4f9',ink:'#172033',muted:'#5f6b7e',border:'#dce3ee',accent:'#335cce',accent2:'#0f8b8d',grid:'rgba(20,35,60,.12)',good:'#087f5b',warn:'#b45309',danger:'#b91c1c',colors:['#335cce','#0f8b8d','#6f42c1','#d95f02','#2ca25f','#e7298a','#6c757d']},
 forest:{bg:'#07140f',panel:'#0f2019',panel2:'#153027',ink:'#ecfff6',muted:'#a6c7b6',border:'#28473a',accent:'#52d273',accent2:'#f7c948',grid:'rgba(236,255,246,.10)',good:'#86efac',warn:'#f7c948',danger:'#fb7185',colors:['#52d273','#2dd4bf','#f7c948','#7dd3fc','#c084fc','#fb7185','#a3e635']}
};
let currentTheme='indigo';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=v=>v==null||!Number.isFinite(Number(v))?null:Number(v);
const fmt=(v,d=3)=>{const n=num(v);if(n==null)return '';if(Math.abs(n)>0&&Math.abs(n)<Math.pow(10,-d))return n.toExponential(2);return n.toFixed(d).replace(/0+$/,'').replace(/\.$/,'')};
const pct=v=>num(v)==null?'':fmt(100*Number(v),1)+'%';
const ciText=(mean,lo,hi,d=1)=>{const m=num(mean),l=num(lo),h=num(hi);if(m==null)return '';return l==null||h==null?fmt(m,d):`${fmt(m,d)} (${fmt(l,d)}–${fmt(h,d)})`};
const textOr=(value,fallback='Not available')=>String(value??'').trim()||fallback;
function theme(){return THEMES[currentTheme]}
function plotLayout(extra={}){const t=theme();return Object.assign({paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:t.ink,size:12},colorway:t.colors,margin:{l:70,r:24,t:24,b:80},legend:{orientation:'h',y:-.2}},extra)}
function applyTheme(name){currentTheme=name;const t=theme(),root=document.documentElement;['bg','panel','panel2','ink','muted','border','accent','accent2','grid','good','warn','danger'].forEach(k=>root.style.setProperty('--'+k,t[k]));document.querySelectorAll('.theme-btn').forEach(b=>b.classList.toggle('active',b.dataset.theme===name));renderComplexity()}
function tableMarkup(headers,rows){return `<thead><tr>${headers.map((h,i)=>`<th${i===0?' scope="col"':''}>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.length?rows.join(''):`<tr><td colspan="${headers.length}" class="empty">No rows available.</td></tr>`}</tbody>`}
function renderHero(){const m=DATA.metadata||{};const taskCount=(DATA.per_task||[]).length;const modelCount=(DATA.model_order||[]).length;const tuning=m.tune?`${m.inner_splits??3}-fold inner CV`:'Tuning disabled';const chips=[`Primary metric: AUC`,`{{SUITE_LABEL}}: ${taskCount} of {{SUITE_TASK_COUNT}} tasks`,`Models: ${modelCount}`,tuning,`AUC tolerance: ${fmt(DATA.complexity_auc_tolerance,3)}`];document.getElementById('heroChips').innerHTML=chips.map(x=>`<span class="chip">${esc(x)}</span>`).join('')}
function renderMetrics(){const overall=DATA.overall||[];const valid=overall.filter(r=>num(r.mean_auc)!=null).sort((a,b)=>b.mean_auc-a.mean_auc);const best=valid[0]||{};const primary=DATA.primary_hugiml_model||'HUGIML';const hug=overall.find(r=>r.model===primary)||{};const comp=(DATA.complexity_summary||[]).find(r=>r.hugiml_model===primary&&r.baseline_model==='XGB standard')||(DATA.complexity_summary||[]).find(r=>num(r.median_inspection_ratio_baseline_to_hugiml)!=null)||{};const fr=DATA.friedman||{};const cards=[['Best mean AUC',fmt(best.mean_auc),textOr(best.model)],[primary+' mean AUC',fmt(hug.mean_auc),`mean rank ${textOr(fmt(hug.mean_rank,2))}`],[primary+' median inspection',fmt(hug.median_model_inspection_units,1),'model inspection units'],['Comparable inspection ratio',num(comp.median_inspection_ratio_baseline_to_hugiml)==null?'Not available':fmt(comp.median_inspection_ratio_baseline_to_hugiml,2)+'×',comp.baseline_model?`vs ${comp.baseline_model} on ${comp.comparable_or_better_tasks||0} tasks`:'insufficient paired data'],['Friedman p-value',textOr(fmt(fr.p_value,5)),fr.n_tasks?`${fr.n_tasks} complete paired tasks`:'insufficient paired tasks']];document.getElementById('metrics').innerHTML=cards.map(x=>`<div class="card stat"><div class="label">${esc(x[0])}</div><div class="value ${String(x[1]).length>12?'small':''}">${esc(x[1])}</div><div class="sub">${esc(x[2])}</div></div>`).join('');const specs=[['Balanced accuracy','mean_balanced_accuracy',true],['F1 score','mean_f1',true],['Brier score','mean_brier',false]];document.getElementById('metricComparisons').innerHTML=specs.map(([label,key,higher])=>{const rows=overall.filter(r=>num(r[key])!=null).sort((a,b)=>higher?Number(b[key])-Number(a[key]):Number(a[key])-Number(b[key]));const winner=rows[0]||{};const ranks=rows.map(r=>`<span class="metric-rank ${String(r.model).startsWith('HUGIML')?'hugiml':''}">${esc(r.model.replace(' standard',''))} ${fmt(r[key])}</span>`).join('');return `<div class="card stat metric-comparison"><div class="label">${esc(label)} · ${higher?'higher':'lower'} is better</div><div class="value">${textOr(fmt(winner[key]))}<span class="winner">${esc(winner.model||'Not available')}</span></div><div class="metric-ranking">${ranks}</div></div>`}).join('')}
function renderOverall(){const rows=DATA.overall||[];const best=Math.max(...rows.map(r=>num(r.mean_auc)??-Infinity));const hasCi=rows.length>0&&rows.every(r=>num(r.mean_instance_inspection_units)!=null&&num(r.instance_inspection_ci_lower)!=null&&num(r.instance_inspection_ci_upper)!=null);const hasInstance=rows.length>0&&rows.every(r=>num(r.mean_instance_inspection_units)!=null);const headers=['Model','Completed','Mean AUC','Std AUC','Mean rank','Wins','Mean model inspection','Median model inspection'];if(hasCi)headers.push('Mean instance inspection (95% CI)');else if(hasInstance)headers.push('Mean instance inspection');headers.push('Tune s','Fit s','Predict s','Complexity s','Overhead s');const body=rows.map(r=>{const cells=[`<span class="pill">${esc(r.model)}</span>`,`${r.tasks_completed??0}/${r.tasks_attempted??r.tasks_completed??0}`,`<span class="${num(r.mean_auc)===best?'best':''}">${fmt(r.mean_auc)}</span>`,fmt(r.std_auc),fmt(r.mean_rank,2),String(r.wins??0),fmt(r.mean_model_inspection_units,1),fmt(r.median_model_inspection_units,1)];if(hasCi)cells.push(ciText(r.mean_instance_inspection_units,r.instance_inspection_ci_lower,r.instance_inspection_ci_upper,1));else if(hasInstance)cells.push(fmt(r.mean_instance_inspection_units,1));cells.push(fmt(r.mean_tune_seconds,2),fmt(r.mean_fit_seconds,2),fmt(r.mean_predict_seconds,2),fmt(r.mean_complexity_seconds,2),fmt(r.mean_evaluation_overhead_seconds,2));return `<tr>${cells.map(c=>`<td>${c}</td>`).join('')}</tr>`});document.getElementById('overallTable').innerHTML=tableMarkup(headers,body)}
function renderComplexity(){const rows=DATA.complexity_points||[];const fallback=document.getElementById('plotFallback');fallback.textContent='';if(!rows.length){fallback.textContent='No model-inspection values are available.';return}if(!window.Plotly){fallback.textContent='Interactive chart unavailable; the same values remain available in the tables and exported CSV files.';return}const models=[...new Set(rows.map(r=>r.model))];const traces=models.map(model=>{const sub=rows.filter(r=>r.model===model);return {type:'scatter',mode:'markers',name:model,x:sub.map(r=>r.model_inspection_units),y:sub.map(r=>r.roc_auc),text:sub.map(r=>`${r.dataset} · task ${r.task_id}`),customdata:sub.map(r=>[ciText(r.instance_inspection_units_mean,r.instance_inspection_units_ci_lower,r.instance_inspection_units_ci_upper,1)]),hovertemplate:'%{text}<br>Model inspection=%{x:.2f}<br>Instance inspection=%{customdata[0]}<br>AUC=%{y:.4f}<extra>%{fullData.name}</extra>',marker:{size:10,opacity:.78}}});const positive=rows.every(r=>Number(r.model_inspection_units)>0);const t=theme();Plotly.react('complexityPlot',traces,plotLayout({xaxis:{title:'Model inspection units',type:positive?'log':'linear',automargin:true,gridcolor:t.grid},yaxis:{title:'ROC AUC',range:[0,1.02],automargin:true,gridcolor:t.grid},hovermode:'closest'}),{responsive:true,displaylogo:false})}
function renderRatios(){const rows=DATA.complexity_summary||[];const headers=['HUGIML variant','Baseline','Comparable / paired','Comparable share','Mean AUC HUGIML','Mean AUC baseline','Median HUGIML inspection','Median baseline inspection','Median ratio','Median reduction'];const body=rows.map(r=>`<tr><td>${esc(r.hugiml_model)}</td><td>${esc(r.baseline_model)}</td><td>${r.comparable_or_better_tasks??0}/${r.paired_tasks??0}</td><td>${pct(r.comparable_or_better_share)}</td><td>${fmt(r.mean_hugiml_auc)}</td><td>${fmt(r.mean_baseline_auc)}</td><td>${fmt(r.median_hugiml_model_inspection_units,1)}</td><td>${fmt(r.median_baseline_model_inspection_units,1)}</td><td class="${num(r.median_inspection_ratio_baseline_to_hugiml)>=1?'ratio-good':'ratio-warn'}">${num(r.median_inspection_ratio_baseline_to_hugiml)==null?'':fmt(r.median_inspection_ratio_baseline_to_hugiml,2)+'×'}</td><td>${num(r.median_hugiml_reduction_percent)==null?'':fmt(r.median_hugiml_reduction_percent,1)+'%'}</td></tr>`);document.getElementById('ratioTable').innerHTML=tableMarkup(headers,body);document.getElementById('claims').innerHTML=rows.filter(r=>r.headline).map(r=>`<div class="claim">${esc(r.headline)}</div>`).join('')||'<p class="note">No comparable task-level inspection ratios are available.</p>'}
function renderTask(){const id=Number(document.getElementById('taskSelect').value);const task=(DATA.per_task||[]).find(x=>x.task_id===id)||(DATA.per_task||[])[0];if(!task)return;document.getElementById('taskMeta').textContent=`OpenML task ${task.task_id} · ${task.n_rows} rows · ${task.n_classes} classes`;const allRows=(DATA.per_task||[]).flatMap(t=>t.rows||[]);const hasCi=allRows.some(r=>num(r.complexity_instance_inspection_units_mean)!=null&&num(r.complexity_instance_inspection_units_ci_lower)!=null&&num(r.complexity_instance_inspection_units_ci_upper)!=null);const hasInstance=allRows.some(r=>num(r.complexity_instance_inspection_units_mean)!=null);const headers=['Model','AUC','Rank','Model inspection'];if(hasCi)headers.push('Instance inspection (95% CI)');else if(hasInstance)headers.push('Instance inspection');headers.push('Baseline / '+(DATA.primary_hugiml_model||'HUGIML')+' ratio','Tune s','Fit s','Predict s','Complexity s','Overhead s','Splits','Errors');const body=(task.rows||[]).map(r=>{const comparable=r.comparable_or_better_auc_to_hugiml;const ratioClass=r.model===(DATA.primary_hugiml_model||'HUGIML')?'':(comparable?'ratio-good':'ratio-warn');const cells=[esc(r.model),`<span class="${r.rank===1?'best':''}">${fmt(r.roc_auc)}</span>`,fmt(r.rank,2),fmt(r.model_inspection_units,1)];if(hasCi)cells.push(ciText(r.complexity_instance_inspection_units_mean,r.complexity_instance_inspection_units_ci_lower,r.complexity_instance_inspection_units_ci_upper,1));else if(hasInstance)cells.push(fmt(r.complexity_instance_inspection_units_mean,1));cells.push(`<span class="${ratioClass}">${num(r.inspection_ratio_to_hugiml)==null?'':fmt(r.inspection_ratio_to_hugiml,2)+'×'}</span>`,fmt(r.tune_seconds,2),fmt(r.fit_seconds,2),fmt(r.predict_seconds,2),fmt(r.complexity_seconds,2),fmt(r.evaluation_overhead_seconds,2),`${r.official_splits_successful??0}/${r.official_splits_expected??r.official_splits_completed??0}`,`<span class="${r.error_count?'error':''}">${r.error_count??0}</span>`);return `<tr>${cells.map(c=>`<td>${c}</td>`).join('')}</tr>`});document.getElementById('taskTable').innerHTML=tableMarkup(headers,body)}
function metricScoreClass(values,index){const rounded=values.map(v=>num(v)==null?null:Number(Number(v).toFixed(4)));const unique=[...new Set(rounded.filter(v=>v!=null))].sort((a,b)=>b-a);const value=rounded[index];if(value==null)return '';if(value===unique[0])return 'best score-best';if(unique.length>1&&value===unique[1])return 'score-second';return ''}
function renderDatasetMetrics(){const models=DATA.model_order||[];const short={'HUGIML':'HUGIML','HUGIML':'HUGIML','XGB standard':'XGB','LightGBM standard':'LightGBM','RandomForest standard':'RandomForest'};const metrics=[['roc_auc','ROC-AUC'],['balanced_accuracy','Balanced accuracy'],['f1','F1 (binary / macro)']];const row1=`<tr><th rowspan="2" class="dataset-sticky">Dataset</th><th rowspan="2" class="task-sticky">Task</th>${metrics.map(([,label])=>`<th colspan="${models.length}" class="metric-group-head">${esc(label)}</th>`).join('')}</tr>`;const row2=`<tr>${metrics.flatMap(()=>models).map((model,i)=>`<th class="${i%models.length===0?'group-start':''}">${esc(short[model]||model)}</th>`).join('')}</tr>`;const body=(DATA.per_task||[]).map(task=>{const byModel=Object.fromEntries((task.rows||[]).map(row=>[row.model,row]));const cells=[];for(const [key] of metrics){const values=models.map(model=>byModel[model]?.[key]);values.forEach((value,i)=>{const classes=[metricScoreClass(values,i),i===0?'group-start':''].filter(Boolean).join(' ');cells.push(`<td class="${classes}" title="${num(value)==null?'':esc(String(value))}">${fmt(value,4)}</td>`)});}return `<tr><td class="dataset-sticky">${esc(task.dataset)}</td><td class="task-sticky">${task.task_id}</td>${cells.join('')}</tr>`}).join('');document.getElementById('datasetMetricTable').innerHTML=`<thead>${row1}${row2}</thead><tbody>${body||`<tr><td colspan="${2+metrics.length*models.length}" class="empty">No rows available.</td></tr>`}</tbody>`}
{{RPTE_SCRIPT}}
function renderPaired(){const rows=DATA.versus_hugiml||[];document.getElementById('paired').innerHTML=rows.map(r=>`<div class="paired-row"><strong>${esc(r.comparison)}</strong>${r.significant_holm_0_05?'<span class="badge">Significant</span>':''}<div class="note">Mean AUC difference: ${fmt(r.mean_auc_difference_hugiml_minus_baseline)} · raw p=${fmt(r.p_value,5)} · Holm p=${fmt(r.p_value_holm,5)} · paired tasks=${r.n_tasks??0}</div></div>`).join('')||'<div class="note">Insufficient complete paired tasks.</div>'}
function renderProtocol(){const m=DATA.metadata||{};const rotating=m.validation_protocol==='rotating';const outer=rotating?'For test fold F_i, validation is F_(i+1) mod K and all remaining folds train the model':'Every stored task-defined train/test split';const selection=rotating?(m.tune?'Select by validation ROC AUC (balanced accuracy only when AUC is undefined); retain the fitted winner without refit':'Tuning disabled; fit estimator defaults on the rotating training folds'):(m.tune?`${m.inner_splits??3}-fold stratified CV inside each outer training partition`:'Disabled; estimator defaults fitted on each outer training partition');const stopping=m.early_stopping?`RPTE staged early stopping monitors log loss on the ${rotating?'rotating validation fold':'private stratified training holdout'} (maximum 15 trees, patience 3)`:'RPTE early stopping disabled; n_estimators=10';const baselineStopping=rotating?'XGBoost and LightGBM: maximum 200 trees, patience 20; Random Forest: tune 100 or 200 trees':'Defined by the nested-protocol baseline grids';const rows=[['Outer evaluation',outer],['Model selection',selection],['Downstream redundancy','Training-only single-pass reduction of generated terms with VIF greater than 5 when an earlier preferred term explains at least 80 percent of their variance; originals are preserved and patterns precede augmented pairs'],['RPTE early stopping',stopping],['Tree baseline budgets',baselineStopping],['Primary metric','Binary ROC AUC or multiclass OVR macro ROC AUC'],['Models',(DATA.model_order||[]).join(', ')],['Inspection comparison',`HUGIML AUC within ${fmt(DATA.complexity_auc_tolerance,3)} of or above the baseline`],['Task selection',m.selection_description||'Selected tasks']];if(m.methodology_note)rows.push(['Fallback methodology',m.methodology_note]);document.getElementById('protocol').innerHTML=rows.map(([k,v])=>`<div><strong>${esc(k)}</strong></div><div>${esc(v)}</div>`).join('');renderMethodology(rows,rotating)}
function renderComplexityDefinitions(){const familyDefinitions={'HUGIML':'HUGIML: sparse linear branches count active source contributions; RPTE branches count conditions across active terminal paths plus active direct terms.','HUGIML':'HUGIML: linear branches count active source contributions; RPTE branches count conditions across active terminal paths plus active direct terms.','XGB standard':'XGBoost: complete-model units sum conditions across all active root-to-leaf paths; instance units count the reached path in every tree.','LightGBM standard':'LightGBM: complete-model units sum conditions across all active root-to-leaf paths; instance units count the reached path in every tree.','RandomForest standard':'Random Forest: complete-model units sum conditions across every root-to-leaf path; instance units count the reached path in every tree.','Logistic Regression':'Logistic regression: complete-model units count nonzero coefficients across fitted class models; instance units count active source contributions for one row.','EBM':'EBM: active term-score cells are weighted by source-feature arity; instance units count the selected cell for each term with the same arity weighting.','RuleFit':'RuleFit: complete-model units count active linear terms and all conditions in active rules; instance units count active linear contributions and conditions in rules satisfied by one row.'};const complexity=['Model inspection units measure the expanded evidence needed to inspect the complete fitted model. They count reviewed source elements, rule conditions, active score cells, or all tree-path conditions according to model family.','Instance inspection units measure the expanded evidence used for one prediction. Reported values are means across held-out instances, with a two-sided 95% Student-t confidence interval when available.','Fitted numeric components and terminal outputs are active when their absolute value exceeds 1e-12; intercepts are excluded.',...(DATA.model_order||[]).map(model=>familyDefinitions[model]).filter(Boolean)];document.getElementById('methodologyComplexity').innerHTML=complexity.map(x=>`<li>${esc(x)}</li>`).join('')}
function methodologyValue(model,name,value){if(String(model).startsWith('HUGIML')&&name==='base_estimator'){if(value===null)return 'Automatic sparse logistic regression (binary: liblinear/L1; multiclass: SAGA/L1; C=0.5)';if(String(value).includes('LeafWiseBoundedLookaheadRPTEFeatureLR'))return 'RPTE adaptive sequential/lookahead (liblinear/L1 downstream; One-vs-Rest multiclass; leaf_config=3xD; depth=4; lr_C=0.5)'}if(value===null)return 'None';if(value===true)return 'True';if(value===false)return 'False';return String(value)}
function renderMethodology(protocolRows,rotating){const m=DATA.metadata||{};const bullets=protocolRows.map(([k,v])=>`${k}: ${v}`);document.getElementById('methodologyProtocol').innerHTML=bullets.map(x=>`<li>${esc(x)}</li>`).join('');const ta=m.tabarena_protocol||{};const preprocessing=ta.model_agnostic_preprocessing?['AutoMLPipelineFeatureGenerator is fitted independently on each inner child training fold.','The corresponding fitted generator transforms that child validation fold and remains attached to the child for outer-test prediction.','HUGIML then applies its native feature handling; baseline models apply their existing model-specific numeric and categorical transformations.','No validation or outer-test rows contribute to preprocessing fit state.']:['All preprocessing is fitted using training data only within each outer split.','Numeric features use median imputation; categorical features use most-frequent imputation and one-hot encoding for baseline models.','Tuning, fitting, and prediction times are recorded separately for every official split.'];document.getElementById('methodologyPreprocessing').innerHTML=preprocessing.map(x=>`<li>${esc(x)}</li>`).join('');const grids=m.grid_snapshot||{};const cards=(DATA.model_order||[]).map(model=>{const spec=grids[model]||{};const grid=spec.grid||{};const params=Object.entries(grid).map(([name,values])=>`<tr><th scope="row">${esc(name)}</th><td>${esc((Array.isArray(values)?values:[values]).map(value=>methodologyValue(model,name,value)).join(', '))}</td></tr>`).join('');let runtime=[];if(String(model).startsWith('HUGIML'))runtime=m.early_stopping?['RPTE early_stopping=True','maximum estimators=15','patience=3','monitoring metric=log loss',rotating?'validation source=rotating validation fold':'validation source=private stratified 10% training holdout','minimum improvement=1e-4']:['RPTE early_stopping=False','n_estimators=10'];else if(model==='XGB standard'&&rotating)runtime=[`validation early stopping enabled`,`maximum estimators=${m.baseline_early_stopping_max_estimators??200}`,`patience=${m.baseline_early_stopping_patience??20}`];else if(model==='LightGBM standard'&&rotating)runtime=[`validation early stopping callback enabled`,`maximum estimators=${m.baseline_early_stopping_max_estimators??200}`,`patience=${m.baseline_early_stopping_patience??20}`];else if(model==='RandomForest standard'&&rotating)runtime=['no iterative early stopping','validation selects n_estimators from 100 or 200'];else runtime=['no additional validation-time stopping setting'];const constants=Object.entries(spec.constants||{}).map(([k,v])=>`${k}=${methodologyValue(model,k,v)}`);return `<article class="methodology-model"><h4>${esc(model)}</h4><div class="methodology-model-meta">Search space: ${esc(spec.grid_name||model)} · ${esc(spec.candidate_count??1)} candidate configurations</div><table class="methodology-table"><thead><tr><th>Parameter</th><th>Values considered</th></tr></thead><tbody>${params}</tbody></table><p><strong>Constant settings:</strong> ${esc(constants.join('; ')||'None recorded')}</p><p><strong>Validation-time behavior:</strong> ${esc(runtime.join('; '))}</p></article>`});document.getElementById('methodologyModels').innerHTML=cards.join('')}
const select=document.getElementById('taskSelect');select.innerHTML=(DATA.per_task||[]).map(t=>`<option value="${t.task_id}">${esc(t.dataset)} · task ${t.task_id}</option>`).join('');select.addEventListener('change',renderTask);document.querySelectorAll('.theme-btn').forEach(button=>button.addEventListener('click',()=>applyTheme(button.dataset.theme)));renderHero();renderMetrics();renderOverall();renderRatios();renderTask();renderDatasetMetrics();{{RPTE_RENDER_CALL}}renderPaired();renderProtocol();renderComplexityDefinitions();renderComplexity();
</script></body></html>
"""


def write_template(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_TEMPLATE, encoding="utf-8")
    return path


def render_dashboard(
    data: dict[str, Any],
    out_html: Path,
    template_html: Path | None,
    *,
    include_rpte: bool = False,
    include_official_reference: bool = False,
) -> None:
    template = (
        template_html.read_text(encoding="utf-8") if template_html is not None else DEFAULT_TEMPLATE
    )
    required = ["{{TITLE}}", "{{DATA_JSON}}"]
    missing = [token for token in required if token not in template]
    if missing:
        raise ValueError(f"Dashboard template is missing placeholders: {missing}")
    title = f"HUGIML {DEFAULT_BENCHMARK_LABEL} Benchmark Dashboard"
    rpte_section = (
        """
<section class="section card">
  <div class="section-title"><h2>HUGIML fitted paths and RPTE distributions</h2><div class="meta">Actual route counts and official-split structural distributions</div></div>
  <p class="note">Inputs are the downstream columns supplied to each binary RPTE estimator. Trees and leaves are totals in the fitted one-vs-rest model; path length is averaged across generated terminal leaves.</p>
  <div class="table-wrap"><table id="rpteDistributionTable"></table></div>
</section>
"""
        if include_rpte
        else ""
    )
    rpte_script = (
        """
function rangeText(r,p,d=1){return !num(r[`${p}_n`])?'':`${fmt(r[`${p}_median`],d)} [${fmt(r[`${p}_q25`],d)}, ${fmt(r[`${p}_q75`],d)}]`}
function renderRpteDistributions(){const rows=DATA.hugiml_rpte_distributions||[];const headers=['Variant','Dataset','Task','Splits','LR splits','RPTE splits','Sequential estimators','Lookahead estimators','Inputs median [IQR]','Generated trees median [IQR]','Active trees median [IQR]','Generated leaves median [IQR]','Active leaves median [IQR]','Active direct terms median [IQR]','Average active leaf path median [IQR]'];const body=rows.map(r=>`<tr><td>${esc(r.model)}</td><td>${esc(r.dataset)}</td><td>${r.task_id}</td><td>${r.split_count}</td><td>${r.lr_split_count}</td><td>${r.rpte_split_count}</td><td>${r.rpte_sequential_estimator_count}</td><td>${r.rpte_lookahead_estimator_count}</td><td>${rangeText(r,'rpte_inputs_passed')}</td><td>${rangeText(r,'rpte_tree_count')}</td><td>${rangeText(r,'rpte_active_tree_count')}</td><td>${rangeText(r,'rpte_leaf_count')}</td><td>${rangeText(r,'rpte_active_leaf_count')}</td><td>${rangeText(r,'rpte_direct_term_count')}</td><td>${rangeText(r,'rpte_active_average_leaf_path_length',2)}</td></tr>`);document.getElementById('rpteDistributionTable').innerHTML=tableMarkup(headers,body)}
"""
        if include_rpte
        else ""
    )
    tabarena_section = ""
    if DEFAULT_TABARENA_REPORTING:
        leaderboard = data.get("tabarena_leaderboard", [])

        def display(value: Any, digits: int = 3) -> str:
            return "—" if value is None else f"{float(value):.{digits}f}"

        body = "".join(
            "<tr>"
            f"<td>{html.escape(str(row.get('model', '')))}</td>"
            f"<td>{display(row.get('elo'), 1)}</td>"
            f"<td>{display(row.get('elo_ci_lower'), 1)}–{display(row.get('elo_ci_upper'), 1)}</td>"
            f"<td>{display(row.get('normalized_score'))}</td>"
            f"<td>{display(row.get('average_rank'), 2)}</td>"
            f"<td>{display(row.get('harmonic_mean_rank'), 2)}</td>"
            f"<td>{display(row.get('pairwise_win_rate'))}</td>"
            f"<td>{display(row.get('improvability_percent'), 2)}%</td>"
            "</tr>"
            for row in leaderboard
        )
        tabarena_section = f"""
<section class="section card">
  <div class="section-title"><h2>TabArena leaderboard</h2><div class="meta">Protocol-specific comparative metrics</div></div>
  <div class="table-wrap"><table><thead><tr><th>Model</th><th>Elo</th><th>95% interval</th><th>Normalized score</th><th>Average rank</th><th>Harmonic rank</th><th>Pairwise win rate</th><th>Improvability</th></tr></thead><tbody>{body}</tbody></table></div>
  <p class="note"><strong>Methodology.</strong> Every prescribed outer split uses eight-fold stratified CV inside its training partition. The selected configuration remains an eight-child ensemble whose probabilities are averaged on the outer test fold; no final refit is performed. Binary configurations are selected by ROC AUC and multiclass configurations by log loss. Elo intervals use 200 dataset-bootstrap rounds. Normalized score maps each dataset’s best error to 1 and median error to 0, clipped at 0. Improvability is the mean percentage error reduction available relative to the best model on each dataset.</p>
</section>
"""
        official_comparison = data.get("tabarena_official_comparison", {})
        if include_official_reference and official_comparison.get("available"):
            official_rows = official_comparison.get("leaderboard", [])
            official_body = "".join(
                (
                    '<tr class="official-reference-hugiml">'
                    if row.get("method") == "HUGIML"
                    else "<tr>"
                )
                + f"<td>{html.escape(str(row.get('method', '')))}</td>"
                + f"<td>{display(row.get('elo'), 1)}</td>"
                + f"<td>{display(row.get('elo_ci_lower'), 1)}â€“{display(row.get('elo_ci_upper'), 1)}</td>"
                + f"<td>{display(row.get('normalized_score'))}</td>"
                + f"<td>{display(row.get('average_rank'), 2)}</td>"
                + f"<td>{display(row.get('mean_error'), 4)}</td>"
                + f"<td>{display(row.get('median_error'), 4)}</td>"
                + f"<td>{display(row.get('mean_improvability_percent'), 2)}%</td>"
                + f"<td>{display(row.get('median_improvability_percent'), 2)}%</td>"
                + "</tr>"
                for row in official_rows
            )
            pairwise_rows = official_comparison.get("hugiml_pairwise", [])
            pairwise_body = "".join(
                "<tr>"
                f"<td>{html.escape(str(row.get('method', '')))}</td>"
                f"<td>{int(row.get('wins', 0))}/{int(row.get('ties', 0))}/{int(row.get('losses', 0))}</td>"
                f"<td>{display(row.get('mean_hugiml_error_reduction_percent'), 2)}%</td>"
                f"<td>{display(row.get('median_hugiml_error_reduction_percent'), 2)}%</td>"
                "</tr>"
                for row in pairwise_rows
            )
            matched = int(official_comparison.get("matched_dataset_count", 0))
            official_count = int(official_comparison.get("official_method_count", 0))
            reference = html.escape(
                str(official_comparison.get("elo_reference_method", "RF (tuned)"))
            )
            tabarena_section += f"""
<section class="section card" id="officialTabArenaComparison">
  <div class="section-title"><h2>HUGIML against the official TabArena reference pool</h2><div class="meta">{matched} completed datasets · {official_count} official model variants</div></div>
  <p class="note"><strong>Approach.</strong> Local HUGIML results are joined to TabArena's published results by exact dataset name and outer-split number. Binary error is 1 − ROC AUC; multiclass error is log loss. Split errors are averaged within each dataset before aggregation, giving every dataset equal weight. Only fully completed HUGIML datasets with complete official coverage for every reference method are included. Elo is anchored at {reference}=1000 with 200 dataset-bootstrap rounds.</p>
  <p class="note"><strong>Interpretation boundary.</strong> The outer test partitions and metrics are aligned, but inner model selection, configuration counts, compute budgets, and retained ensembles remain specific to each method. HUGIML evaluates 16 configurations, while each tuned official baseline evaluates 200. This is therefore an official-test comparison, not an equal-compute comparison. Mean and median raw error are descriptive because binary and multiclass datasets use different error metrics; Elo, ranks, normalized score, improvability, and paired comparisons are calculated within dataset before aggregation.</p>
  <div class="official-regime-tabs" role="tablist" aria-label="Official TabArena comparison pool"><button type="button" class="official-regime-tab active" data-regime-tab="default">Default</button><button type="button" class="official-regime-tab" data-regime-tab="tuned">Tuned</button><button type="button" class="official-regime-tab" data-regime-tab="tuned_ensemble">Tuned + ensemble</button><button type="button" class="official-regime-tab" data-regime-tab="all">All</button></div>
  <p class="note" id="officialRegimeDescription">Tuned reference pool. Comparative metrics are recomputed within this pool.</p>
  <h3>Leaderboard</h3>
  <div class="table-wrap" style="max-height:560px"><table><thead><tr><th>Method</th><th>Elo</th><th>95% interval</th><th>Normalized score</th><th>Average rank</th><th>Mean error</th><th>Median error</th><th>Mean improvability</th><th>Median improvability</th></tr></thead><tbody id="officialRegimeLeaderboard">{official_body}</tbody></table></div>
  <h3>HUGIML pairwise comparison</h3>
  <p class="note">Win/tie/loss is from HUGIML's perspective. Positive error reduction means HUGIML has lower error.</p>
  <div class="table-wrap" style="max-height:440px"><table><thead><tr><th>Official method</th><th>HUGIML W/T/L</th><th>Mean error reduction</th><th>Median error reduction</th></tr></thead><tbody id="officialRegimePairwise">{pairwise_body}</tbody></table></div>
</section>
<style>.official-reference-hugiml{{background:color-mix(in srgb,var(--accent2) 16%,transparent);font-weight:750}}.official-regime-tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}}.official-regime-tab{{border:1px solid var(--border);border-radius:999px;padding:8px 14px;background:var(--panel2);color:var(--muted);font:inherit;font-weight:700;cursor:pointer}}.official-regime-tab.active{{background:color-mix(in srgb,var(--accent) 22%,var(--panel2));border-color:var(--accent);color:var(--ink)}}</style>
<script>document.addEventListener('DOMContentLoaded',()=>{{const root=DATA.tabarena_official_comparison||{{}};const views=root.views||{{tuned:root}};const labels={{default:'Default',tuned:'Tuned',tuned_ensemble:'Tuned + ensemble',all:'All'}};const render=viewName=>{{const view=views[viewName];if(!view)return;document.getElementById('officialRegimeDescription').textContent=`${{labels[viewName]}} reference pool: HUGIML plus ${{view.official_method_count}} official variants. Elo is recomputed for this pool and anchored at ${{view.elo_reference_method}}=1000.`;document.getElementById('officialRegimeLeaderboard').innerHTML=(view.leaderboard||[]).map(row=>`<tr class="${{row.method==='HUGIML'?'official-reference-hugiml':''}}"><td>${{esc(row.method)}}</td><td>${{fmt(row.elo,1)}}</td><td>${{fmt(row.elo_ci_lower,1)}}–${{fmt(row.elo_ci_upper,1)}}</td><td>${{fmt(row.normalized_score)}}</td><td>${{fmt(row.average_rank,2)}}</td><td>${{fmt(row.mean_error,4)}}</td><td>${{fmt(row.median_error,4)}}</td><td>${{fmt(row.mean_improvability_percent,2)}}%</td><td>${{fmt(row.median_improvability_percent,2)}}%</td></tr>`).join('');document.getElementById('officialRegimePairwise').innerHTML=(view.hugiml_pairwise||[]).map(row=>`<tr><td>${{esc(row.method)}}</td><td>${{row.wins}}/${{row.ties}}/${{row.losses}}</td><td>${{fmt(row.mean_hugiml_error_reduction_percent,2)}}%</td><td>${{fmt(row.median_hugiml_error_reduction_percent,2)}}%</td></tr>`).join('')}};document.querySelectorAll('[data-regime-tab]').forEach(button=>button.addEventListener('click',()=>{{document.querySelectorAll('[data-regime-tab]').forEach(item=>item.classList.toggle('active',item===button));render(button.dataset.regimeTab)}}));render(root.default_view||'default')}});</script>
"""
    rpte_section += tabarena_section
    data_json = json.dumps(
        _public_dashboard_json(safe_jsonable(data)),
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    text = (
        template.replace("{{TITLE}}", html.escape(title))
        .replace("{{GENERATED_AT}}", "")
        .replace("{{SUITE_LABEL}}", html.escape(DEFAULT_BENCHMARK_LABEL))
        .replace("{{SUITE_TASK_COUNT}}", str(DEFAULT_SUITE_TASK_COUNT))
        .replace("{{RPTE_SECTION}}", rpte_section)
        .replace("{{RPTE_SCRIPT}}", rpte_script)
        .replace("{{RPTE_RENDER_CALL}}", "renderRpteDistributions();" if include_rpte else "")
        .replace("{{DATA_JSON}}", data_json)
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(text, encoding="utf-8")


def build_tabarena_quadrant_analysis(
    rows: Sequence[dict[str, Any]],
    official_results_path: Path,
    official_metrics_path: Path,
) -> dict[str, Any]:
    """Build tuned-model rankings and ROC-AUC summaries by dataset scale."""
    metrics = pd.read_parquet(official_metrics_path)
    required = {"dataset", "fold", "regime", "official_method", "roc_auc"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"TabArena detailed metrics are missing columns: {missing}")
    metrics = metrics[metrics["regime"].astype(str) == "tuned"].copy()
    hugiml_rows = [
        row
        for row in rows
        if row.get("model") == "HUGIML"
        and row.get("status") == "ok"
        and row.get("roc_auc") is not None
    ]
    if not hugiml_rows:
        return {}
    n_threshold = float(np.median([float(row["n_rows"]) for row in hugiml_rows]))
    p_threshold = float(np.median([float(row["raw_features"]) for row in hugiml_rows]))
    p_floor = int(math.floor(p_threshold))
    groups = {
        "Low n, low p": [row for row in hugiml_rows if float(row["n_rows"]) <= n_threshold and int(row["raw_features"]) <= p_floor],
        "Low n, high p": [row for row in hugiml_rows if float(row["n_rows"]) <= n_threshold and int(row["raw_features"]) > p_floor],
        "High n, low p": [row for row in hugiml_rows if float(row["n_rows"]) > n_threshold and int(row["raw_features"]) <= p_floor],
        "High n, high p": [row for row in hugiml_rows if float(row["n_rows"]) > n_threshold and int(row["raw_features"]) > p_floor],
    }
    label_map = {
        "CAT (tuned)": "CatBoost", "EBM (tuned)": "EBM", "FASTAI (tuned)": "FastAI",
        "GBM (tuned)": "LightGBM", "KNN (tuned)": "KNN", "LR (tuned)": "Linear model",
        "MNCA_GPU (tuned)": "ModernNCA", "NN_TORCH (tuned)": "NN Torch",
        "REALMLP (tuned)": "RealMLP", "RF (tuned)": "Random forest",
        "TABM_GPU (tuned)": "TabM", "TABPFNV2_GPU (tuned)": "TABPFNv2",
        "XGB (tuned)": "XGBoost", "XT (tuned)": "ExtraTrees", "HUGIML": "HUGIML",
    }
    output_groups: dict[str, Any] = {}
    observed_deltas: list[float] = []
    for name, subset in groups.items():
        if not subset:
            continue
        comparison = tabarena_protocol.official_reference_comparison(
            subset, official_results_path, regime="tuned"
        )
        ranking = list(comparison.get("leaderboard", []))
        dataset_names = {str(row["dataset"]) for row in subset}
        scores: dict[str, np.ndarray] = {
            "HUGIML": np.asarray([float(row["roc_auc"]) for row in subset], dtype=float)
        }
        detailed = (
            metrics[metrics["dataset"].astype(str).isin(dataset_names)]
            .groupby(["official_method", "dataset"], as_index=False)["roc_auc"]
            .mean()
        )
        for method, frame in detailed.groupby("official_method"):
            frame = frame.dropna(subset=["roc_auc"])
            if frame.empty:
                continue
            scores[str(method)] = frame["roc_auc"].to_numpy(dtype=float)
        official_scores = {method: values for method, values in scores.items() if method != "HUGIML"}
        best_mean_method = max(official_scores, key=lambda method: float(np.mean(official_scores[method])))
        best_median_method = max(official_scores, key=lambda method: float(np.median(official_scores[method])))
        best_mean = float(np.mean(official_scores[best_mean_method]))
        best_median = float(np.median(official_scores[best_median_method]))
        models: list[dict[str, Any]] = []
        for rank, ranking_row in enumerate(ranking, start=1):
            method = str(ranking_row["method"])
            values = scores.get(method)
            mean_auc = None if values is None else float(np.mean(values))
            median_auc = None if values is None else float(np.median(values))
            mean_delta = None if mean_auc is None else mean_auc - best_mean
            median_delta = None if median_auc is None else median_auc - best_median
            if mean_delta is not None:
                observed_deltas.extend([mean_delta, median_delta])
            models.append({
                "rank": rank,
                "method": method,
                "label": label_map.get(method, method),
                "mean_auc": mean_auc,
                "median_auc": median_auc,
                "mean_delta": mean_delta,
                "median_delta": median_delta,
            })
        hugiml_rank = next(model["rank"] for model in models if model["method"] == "HUGIML")
        output_groups[name] = {
            "dataset_count": len(subset),
            "definition": (
                f"n {'<=' if name.startswith('Low n') else '>'} {n_threshold:g} | "
                f"p {'<=' if name.endswith('low p') else '>'} {p_floor}"
            ),
            "hugiml_rank": hugiml_rank,
            "reference": (
                f"Best AUC: {label_map.get(best_mean_method, best_mean_method)} mean; "
                f"{label_map.get(best_median_method, best_median_method)} median"
            ),
            "models": models,
        }
    axis_min = min(-0.04, min(observed_deltas, default=-0.04))
    return {
        "n_threshold": n_threshold,
        "p_threshold": p_threshold,
        "delta_axis_min": math.floor(axis_min / 0.04) * 0.04,
        "delta_axis_step": 0.04,
        "groups": output_groups,
    }


def render_tabarena_official_dashboard(data: dict[str, Any], out_html: Path) -> None:
    """Render the standalone official-reference leaderboard analysis."""
    comparison = data.get("tabarena_official_comparison", {})
    views = comparison.get("views", {})
    if not views:
        raise ValueError("Official TabArena comparison data is unavailable")

    labels = {
        "default": "Default · overall",
        "default_binary": "Default · binary",
        "default_multiclass": "Default · multiclass",
        "tuned": "Tuned · overall",
        "tuned_binary": "Tuned · binary",
        "tuned_multiclass": "Tuned · multiclass",
        "all": "Combined · overall",
        "all_binary": "Combined · binary",
        "all_multiclass": "Combined · multiclass",
    }
    view_order = (
        "default",
        "default_binary",
        "default_multiclass",
        "tuned",
        "tuned_binary",
        "tuned_multiclass",
        "all",
        "all_binary",
        "all_multiclass",
    )

    def display(value: Any, digits: int = 3) -> str:
        return "—" if value is None else f"{float(value):.{digits}f}"

    def mean_median(mean: Any, median: Any, digits: int = 2, suffix: str = "") -> str:
        return f"{display(mean, digits)}{suffix} ({display(median, digits)}{suffix})"

    buttons: list[str] = []
    panels: list[str] = []
    for view_name in view_order:
        view = views.get(view_name)
        if not view or not view.get("available"):
            continue
        active = view_name == "default"
        buttons.append(
            f'<button class="pool-tab{" active" if active else ""}" '
            f'data-pool-tab="{view_name}" type="button">{labels[view_name]}</button>'
        )
        leaderboard = list(view.get("leaderboard", []))
        hugiml = next((row for row in leaderboard if row.get("method") == "HUGIML"), {})
        position = next(
            (
                index
                for index, row in enumerate(leaderboard, start=1)
                if row.get("method") == "HUGIML"
            ),
            None,
        )
        leaderboard_body = "".join(
            ('<tr class="hugiml-row">' if row.get("method") == "HUGIML" else "<tr>")
            + f'<td><span class="rank">{index}</span></td>'
            + f"<td>{html.escape(str(row.get('method', '')))}</td>"
            + f"<td>{display(row.get('elo'), 1)}</td>"
            + f"<td>{display(row.get('elo_ci_lower'), 1)}–{display(row.get('elo_ci_upper'), 1)}</td>"
            + f"<td>{display(row.get('normalized_score'))}</td>"
            + f"<td>{display(row.get('average_rank'), 2)}</td>"
            + f"<td>{mean_median(row.get('mean_improvability_percent'), row.get('median_improvability_percent'), suffix='%')}</td>"
            + "</tr>"
            for index, row in enumerate(leaderboard, start=1)
        )
        pairwise_body = "".join(
            "<tr>"
            + f"<td>{html.escape(str(row.get('method', '')))}</td>"
            + f"<td>{int(row.get('wins', 0))}</td>"
            + f"<td>{int(row.get('ties', 0))}</td>"
            + f"<td>{int(row.get('losses', 0))}</td>"
            + f'<td class="{"positive" if float(row.get("mean_hugiml_error_reduction_percent", 0)) >= 0 else "negative"}">{mean_median(row.get("mean_hugiml_error_reduction_percent"), row.get("median_hugiml_error_reduction_percent"), suffix="%")}</td>'
            + "</tr>"
            for row in view.get("hugiml_pairwise", [])
        )
        hidden = "" if active else " hidden"
        panels.append(
            f"""
<section class="pool-panel" data-pool-panel="{view_name}"{hidden}>
  <div class="pool-intro"><div><span class="eyebrow">Reference pool</span><h2>{labels[view_name]}</h2></div><p>HUGIML plus {int(view.get("official_method_count", 0))} official variants · {int(view.get("matched_dataset_count", 0))} completed datasets · Elo anchor {html.escape(str(view.get("elo_reference_method")))} = 1000</p></div>
  <div class="metric-grid">
    <article class="metric"><span>HUGIML position</span><strong>{position or "—"} / {len(leaderboard)}</strong><small>within this recomputed pool</small></article>
    <article class="metric"><span>Elo</span><strong>{display(hugiml.get("elo"), 1)}</strong><small>95% interval {display(hugiml.get("elo_ci_lower"), 1)}–{display(hugiml.get("elo_ci_upper"), 1)}</small></article>
    <article class="metric"><span>Average rank</span><strong>{display(hugiml.get("average_rank"), 2)}</strong><small>lower is better</small></article>
    <article class="metric"><span>Normalized score</span><strong>{display(hugiml.get("normalized_score"))}</strong><small>best = 1, dataset median = 0</small></article>
    <article class="metric"><span>Improvability</span><strong>{mean_median(hugiml.get("mean_improvability_percent"), hugiml.get("median_improvability_percent"), suffix="%")}</strong><small>mean (median); lower is better</small></article>
  </div>
  <article class="panel-card"><div class="section-head"><div><span class="eyebrow">Pool-relative results</span><h3>Leaderboard</h3></div><p>Every displayed statistic is recalculated using only this tab's methods.</p></div><div class="table-scroll leaderboard-scroll"><table><thead><tr><th>#</th><th>Method</th><th>Elo</th><th>95% interval</th><th>Normalized score</th><th>Average rank</th><th>Improvability · mean (median)</th></tr></thead><tbody>{leaderboard_body}</tbody></table></div></article>
  <article class="panel-card"><div class="section-head"><div><span class="eyebrow">Dataset-level pairing</span><h3>HUGIML versus each official method</h3></div><p>Positive error reduction favors HUGIML.</p></div><div class="table-scroll pairwise-scroll"><table><thead><tr><th>Official method</th><th>Wins</th><th>Ties</th><th>Losses</th><th>Error reduction · mean (median)</th></tr></thead><tbody>{pairwise_body}</tbody></table></div></article>
</section>"""
        )

    detailed_sections: list[str] = []
    detailed = data.get("tabarena_detailed_metric_comparison", {})
    metric_specs = (
        ("roc_auc", "ROC AUC"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("f1", "F1"),
        ("brier", "Brier"),
    )
    for regime in ("default", "tuned"):
        for scope in ("overall", "binary", "multiclass"):
            rows = list(detailed.get(f"{regime}_{scope}", []))
            if not rows:
                continue
            best_delta = {
                metric: max(
                    (
                        float(row[f"{metric}_hugiml_delta"])
                        for row in rows
                        if row.get(f"{metric}_hugiml_delta") is not None
                        and row.get("method") != "HUGIML"
                    ),
                    default=None,
                )
                for metric, _ in metric_specs
            }
            body: list[str] = []
            for row in rows:
                cells = [f"<td>{html.escape(str(row.get('method', '')))}</td>"]
                for metric, _ in metric_specs:
                    delta = row.get(f"{metric}_hugiml_delta")
                    emphasized = (
                        delta is not None
                        and best_delta[metric] is not None
                        and math.isclose(float(delta), float(best_delta[metric]), abs_tol=1e-12)
                        and row.get("method") != "HUGIML"
                    )
                    delta_text = display(delta, 4)
                    if emphasized:
                        delta_text = f"<strong><em>{delta_text}</em></strong>"
                    cells.extend(
                        [
                            f"<td>{display(row.get(f'{metric}_mean'), 4)}</td>",
                            f"<td>{display(row.get(f'{metric}_median'), 4)}</td>",
                            f'<td class="{"positive" if delta is not None and float(delta) >= 0 else "negative"}">{delta_text}</td>',
                        ]
                    )
                body.append("<tr>" + "".join(cells) + "</tr>")
            group_headers = "".join(f'<th colspan="3">{label}</th>' for _, label in metric_specs)
            subheaders = "".join(
                "<th>Mean</th><th>Median</th><th>HUGIML delta</th>" for _ in metric_specs
            )
            detailed_view_name = regime if scope == "overall" else f"{regime}_{scope}"
            detailed_hidden = "" if detailed_view_name == "default" else " hidden"
            detailed_sections.append(
                f'<article class="panel-card detailed-panel" data-detailed-panel="{detailed_view_name}"{detailed_hidden}><div class="section-head"><div><span class="eyebrow">{regime.title()} · {scope}</span><h3>Predictive metrics and HUGIML deltas</h3></div><p>Positive delta favors HUGIML. Rows are ordered by mean ROC AUC.</p></div><div class="table-scroll leaderboard-scroll"><table><thead><tr><th rowspan="2">Method</th>{group_headers}</tr><tr>{subheaders}</tr></thead><tbody>{"".join(body)}</tbody></table></div></article>'
            )

    matched = int(comparison.get("matched_dataset_count", 0))
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HUGIML · TabArena official leaderboard analysis</title>
<style>
:root{{--bg:#07101f;--panel:#101b31;--panel2:#15233d;--ink:#edf4ff;--muted:#9eacc6;--border:#293a59;--accent:#7da7ff;--accent2:#63e6be;--good:#69db7c;--bad:#ff8787}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 10% 0,#152749 0,transparent 34%),var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,Segoe UI,sans-serif}}.wrap{{width:min(1540px,calc(100% - 32px));margin:auto;padding:30px 0 54px}}.hero{{padding:28px;border:1px solid var(--border);border-radius:22px;background:linear-gradient(135deg,#142747,#0d172a);box-shadow:0 22px 70px #0005}}.eyebrow{{color:var(--accent2);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.14em}}h1{{font-size:clamp(30px,4vw,52px);line-height:1.05;margin:9px 0 12px}}h2{{font-size:28px;margin:4px 0}}h3{{font-size:20px;margin:4px 0}}p{{color:var(--muted);line-height:1.6}}.hero-meta{{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}}.chip{{padding:7px 11px;border:1px solid var(--border);border-radius:999px;background:#ffffff08;color:var(--muted);font-size:12px}}.tabs{{display:flex;gap:9px;flex-wrap:wrap;margin:24px 0 18px}}.pool-tab{{border:1px solid var(--border);border-radius:999px;padding:10px 17px;background:var(--panel);color:var(--muted);font:inherit;font-weight:750;cursor:pointer}}.pool-tab.active{{background:color-mix(in srgb,var(--accent) 24%,var(--panel));border-color:var(--accent);color:var(--ink)}}.pool-panel[hidden],.detailed-panel[hidden]{{display:none}}.pool-intro{{display:flex;align-items:end;justify-content:space-between;gap:20px;margin:8px 2px 16px}}.pool-intro p{{max-width:760px;text-align:right;margin:0}}.metric-grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}}.metric,.panel-card,.methodology{{border:1px solid var(--border);border-radius:16px;background:color-mix(in srgb,var(--panel) 96%,transparent)}}.metric{{padding:16px}}.metric span{{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.07em}}.metric strong{{display:block;font-size:27px;margin:8px 0 5px}}.metric small{{color:var(--muted)}}.panel-card{{padding:18px;margin-top:16px}}.section-head{{display:flex;align-items:end;justify-content:space-between;gap:18px;margin-bottom:12px}}.section-head p{{margin:0;text-align:right;font-size:12px}}.table-scroll{{overflow:auto;border:1px solid var(--border);border-radius:12px}}.leaderboard-scroll{{max-height:570px}}.pairwise-scroll{{max-height:470px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:10px 12px;border-bottom:1px solid #293a5988;text-align:right;white-space:nowrap}}th{{position:sticky;top:0;background:var(--panel2);color:var(--muted);z-index:2}}th:nth-child(2),td:nth-child(2),.pairwise-scroll th:first-child,.pairwise-scroll td:first-child{{text-align:left}}.rank{{display:inline-grid;place-items:center;min-width:26px;height:26px;border-radius:50%;background:#ffffff0a;color:var(--muted)}}.hugiml-row{{background:color-mix(in srgb,var(--accent2) 14%,transparent);font-weight:750}}.positive{{color:var(--good);font-weight:750}}.negative{{color:var(--bad);font-weight:750}}.methodology{{padding:22px;margin-top:20px}}.methodology ul{{color:var(--muted);line-height:1.65}}footer{{color:var(--muted);font-size:12px;margin-top:18px;text-align:right}}@media(max-width:1000px){{.metric-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.pool-intro,.section-head{{align-items:flex-start;flex-direction:column}}.pool-intro p{{text-align:right;margin:0}}.section-head p{{text-align:left}}}}@media(max-width:560px){{.wrap{{width:min(100% - 18px,1540px)}}.hero{{padding:20px}}.metric-grid{{grid-template-columns:1fr}}}}
</style></head><body><main class="wrap"><header class="hero"><span class="eyebrow">Official-test comparison</span><h1>HUGIML in the TabArena reference landscape</h1><p>A dedicated leaderboard view for aligned outer-test comparisons. Search budgets and ensemble construction remain method-specific, so each pool is presented separately.</p><div class="hero-meta"><span class="chip">{matched} completed datasets</span><span class="chip">Exact outer-split matching</span><span class="chip">Dataset-balanced aggregation</span><span class="chip">200 Elo bootstraps</span></div></header><nav class="tabs" role="tablist">{"".join(buttons)}</nav>{"".join(panels)}{"".join(detailed_sections)}<section class="methodology"><span class="eyebrow">Methodology and interpretation</span><h3>Comparison design</h3><ul><li>Local HUGIML results are joined to published TabArena results by exact dataset name and outer-split number.</li><li>Binary error is 1 − ROC AUC; multiclass error is log loss. Split errors are averaged within dataset before aggregation, giving every dataset equal weight.</li><li>Each tab is an independent comparison pool. Elo, ranks, normalized score, improvability, confidence intervals, and HUGIML pairwise results are recomputed within that pool.</li><li>AutoMLPipelineFeatureGenerator is fitted separately on every inner child training fold and retained with that child for validation and outer-test transformation; no validation or test rows contribute to preprocessing state.</li><li>The outer tests and metrics align. HUGIML evaluates 16 configurations, while each tuned official baseline evaluates 200; retained ensembles and compute budgets remain method-specific.</li><li>Detailed metric tables show mean, median, and the paired HUGIML advantage. Positive deltas favor HUGIML; Brier uses the lower-is-better direction.</li></ul></section></main><script>const tabs=[...document.querySelectorAll('[data-pool-tab]')],panels=[...document.querySelectorAll('[data-pool-panel]')],details=[...document.querySelectorAll('[data-detailed-panel]')];tabs.forEach(tab=>tab.addEventListener('click',()=>{{tabs.forEach(item=>item.classList.toggle('active',item===tab));panels.forEach(panel=>panel.hidden=panel.dataset.poolPanel!==tab.dataset.poolTab);details.forEach(detail=>detail.hidden=detail.dataset.detailedPanel!==tab.dataset.poolTab)}}));</script></body></html>"""
    quadrant_data = data.get("tabarena_quadrant_analysis", {})
    if quadrant_data.get("groups"):
        quadrant_section = """
<section class="quadrant-section">
  <div class="section-head"><div><span class="eyebrow">Dataset-scale analysis</span><h2>Tuned-model AUC gaps</h2></div><p>Models follow their within-quadrant Elo rank. Bars show mean ROC-AUC delta; dots show median delta.</p></div>
  <div class="quadrant-grid" id="quadrantGrid"></div>
  <div class="quadrant-legend"><span><i class="legend-swatch hug"></i>HUGIML mean delta</span><span><i class="legend-swatch peer"></i>Official tuned model mean delta</span><span><i class="legend-dot"></i>Median delta</span></div>
  <p class="quadrant-order-note">Ordering is by TabArena rank, which uses ROC-AUC error for binary datasets and log loss for multiclass datasets. The displayed order is therefore not a ranking by the ROC-AUC values shown at right.</p>
</section>"""
        document = document.replace(
            "</header><nav class=\"tabs\"",
            f"</header>{quadrant_section}<nav class=\"tabs\"",
            1,
        )
        quadrant_json = json.dumps(
            safe_jsonable(quadrant_data), separators=(",", ":"), allow_nan=False
        ).replace("</", "<\\/")
        quadrant_script = """
const QUADRANTS=__QUADRANT_DATA__;
function renderQuadrants(){
  const host=document.getElementById('quadrantGrid');
  if(!host||!QUADRANTS.groups)return;
  const domainMin=QUADRANTS.delta_axis_min??-0.16;
  const step=QUADRANTS.delta_axis_step??0.04;
  Object.entries(QUADRANTS.groups).forEach(([name,group])=>{
    const panel=document.createElement('section');
    panel.className='quadrant-panel';
    panel.innerHTML=`<h3>${name}</h3><p class="quadrant-note">${group.dataset_count} datasets | ${group.definition}</p><p class="quadrant-reference"><strong>HUGIML rank ${group.hugiml_rank}/${group.models.length}</strong> | ${group.reference}</p><svg role="img" aria-label="${name} tuned-model ROC-AUC deltas"></svg>`;
    host.appendChild(panel);
    const svg=panel.querySelector('svg');
    const draw=()=>{
      const width=Math.max(320,panel.getBoundingClientRect().width),height=350,left=132,right=108,top=30,bottom=38,plotWidth=width-left-right,rowHeight=(height-top-bottom)/group.models.length,x=value=>left+((value-domainMin)/-domainMin)*plotWidth,signed=value=>`${value>0?'+':value<0?'-':''}${Math.abs(value).toFixed(4)}`;
      svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
      let markup='<desc>Models are ordered by within-quadrant Elo rank. Bars show mean ROC-AUC delta and dots show median ROC-AUC delta from the best official tuned baseline.</desc>';
      markup+=`<text class="muted" x="${width-2}" y="12" text-anchor="end">AUC mean (median)</text>`;
      for(let value=domainMin;value<=0.000001;value+=step){
        const tick=Math.abs(value)<1e-9?0:value;
        markup+=`<line class="gridline" x1="${x(tick)}" y1="${top-4}" x2="${x(tick)}" y2="${height-bottom+2}"/><text class="muted" x="${x(tick)}" y="${height-20}" text-anchor="middle">${tick.toFixed(2)}</text>`;
      }
      group.models.forEach((model,index)=>{
        const y=top+index*rowHeight+rowHeight/2,hug=model.method==='HUGIML',delta=model.mean_delta;
        markup+=`<text class="${hug?'hug-label':''}" x="${left-8}" y="${y+4}" text-anchor="end">${model.rank}. ${model.label}</text>`;
        if(delta!=null){
          const barX=Math.min(x(delta),x(0)),barWidth=Math.max(1.5,Math.abs(x(0)-x(delta)));
          markup+=`<rect class="${hug?'hug-bar':'peer-bar'}" x="${barX}" y="${y-4.5}" width="${barWidth}" height="9"><title>${model.label} mean delta ${signed(delta)}</title></rect><circle class="median-dot" cx="${x(model.median_delta)}" cy="${y}" r="3.2"><title>${model.label} median delta ${signed(model.median_delta)}</title></circle>`;
        }
        const auc=model.mean_auc==null?'n/a':`${model.mean_auc.toFixed(4)} (${model.median_auc.toFixed(4)})`;
        markup+=`<text class="${hug?'hug-label':'muted'}" x="${width-2}" y="${y+4}" text-anchor="end">${auc}</text>`;
      });
      markup+=`<text x="${left+plotWidth/2}" y="${height-4}" text-anchor="middle">ROC-AUC delta from best tuned baseline</text>`;
      svg.innerHTML=markup;
    };
    new ResizeObserver(draw).observe(panel);
    draw();
  });
}
renderQuadrants();
""".replace("__QUADRANT_DATA__", quadrant_json)
        document = document.replace("</script></body>", quadrant_script + "</script></body>")
    extra_css = """
.tabs{display:grid;grid-template-columns:repeat(9,minmax(0,1fr));gap:5px;margin:20px 0 18px}.pool-tab{min-width:0;padding:8px 5px;font-size:10.5px;line-height:1.2;white-space:nowrap}.quadrant-section{padding:20px;margin-top:20px;border:1px solid var(--border);border-radius:16px;background:color-mix(in srgb,var(--panel) 96%,transparent)}.quadrant-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.quadrant-panel{min-width:0}.quadrant-panel h3{margin-bottom:2px}.quadrant-note,.quadrant-reference{margin:0 0 4px;font-size:12px;color:var(--muted)}.quadrant-panel svg{display:block;width:100%}.quadrant-panel text{fill:var(--ink);font-size:11px}.quadrant-panel .muted{fill:var(--muted)}.quadrant-panel .gridline{stroke:var(--border)}.quadrant-panel .peer-bar{fill:var(--accent)}.quadrant-panel .hug-bar{fill:var(--accent2)}.quadrant-panel .median-dot{fill:var(--ink);stroke:var(--bg);stroke-width:1}.quadrant-panel .hug-label{font-weight:800}.quadrant-legend{display:flex;justify-content:center;gap:18px;flex-wrap:wrap;margin-top:8px;color:var(--muted);font-size:12px}.legend-swatch,.legend-dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.legend-swatch.hug{background:var(--accent2)}.legend-swatch.peer{background:var(--accent)}.legend-dot{background:var(--ink)}.quadrant-order-note{margin:12px 0 0;color:var(--muted);font-size:12px}.methodology{padding:0}.methodology summary{cursor:pointer;padding:20px 22px;font-weight:800}.methodology-content{padding:0 22px 20px}@media(max-width:1160px){.tabs{display:flex;flex-wrap:wrap}.pool-tab{padding:8px 12px}}@media(max-width:760px){.quadrant-grid{grid-template-columns:1fr}}
"""
    document = document.replace("</style>", extra_css + "</style>", 1)
    methodology_start = '<section class="methodology"><span class="eyebrow">'
    if methodology_start in document:
        document = document.replace(
            methodology_start,
            '<section class="methodology"><details><summary>Methodology and interpretation</summary><div class="methodology-content"><span class="eyebrow">',
            1,
        )
        document = document.replace(
            "</ul></section></main>",
            "</ul></div></details></section></main>",
            1,
        )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(document, encoding="utf-8")


def assemble_outputs(
    checkpoint: Path,
    out_dir: Path,
    template_html: Path | None,
    run_id: str | None = None,
    include_rpte_dashboard: bool = False,
) -> dict[str, Path]:
    payload = load_checkpoint(checkpoint)
    view = run_view(payload, run_id)
    data = build_dashboard_data(payload, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_task_ids = {int(row["task_id"]) for row in view.get("selected_tasks", [])}
    candidate_results = [
        annotate_pair_timing(row)
        for row in view["results"]
        if not selected_task_ids or int(row.get("task_id", -1)) in selected_task_ids
    ]
    active_models = list(data.get("model_order", []))
    matched_task_ids = complete_matched_task_ids(candidate_results, active_models)
    selected_results = [row for row in candidate_results if int(row["task_id"]) in matched_task_ids]
    pair_rows = [{k: v for k, v in row.items() if k != "split_rows"} for row in selected_results]
    split_rows = [split for row in selected_results for split in row.get("split_rows", [])]
    tabarena_rows: list[dict[str, Any]] = []
    if DEFAULT_TABARENA_REPORTING:
        alternatives = tabarena_protocol.tabarena_aggregate_metrics(selected_results)
        elo = tabarena_protocol.tabarena_elo_metrics(selected_results)
        elo_by_model = {row["model"]: row for row in elo}
        tabarena_rows = [{**row, **elo_by_model.get(row["model"], {})} for row in alternatives]
        data["tabarena_leaderboard"] = tabarena_rows
        if (
            DEFAULT_TABARENA_OFFICIAL_RESULTS is not None
            and DEFAULT_TABARENA_OFFICIAL_RESULTS.exists()
        ):
            official_view_specs = {
                "default": ("default", "all", "all"),
                "default_binary": ("default", "all", "binary"),
                "default_multiclass": ("default", "all", "multiclass"),
                "tuned": ("tuned", "all", "all"),
                "tuned_binary": ("tuned", "all", "binary"),
                "tuned_multiclass": ("tuned", "all", "multiclass"),
                "all": ("all", "all", "all"),
                "all_binary": ("all", "all", "binary"),
                "all_multiclass": ("all", "all", "multiclass"),
            }
            official_views = {
                view_name: tabarena_protocol.official_reference_comparison(
                    selected_results,
                    DEFAULT_TABARENA_OFFICIAL_RESULTS,
                    regime=regime,
                    model_scope=model_scope,
                    problem_scope=problem_scope,
                )
                for view_name, (regime, model_scope, problem_scope) in official_view_specs.items()
            }
            data["tabarena_official_comparison"] = {
                **official_views["tuned"],
                "default_view": "default",
                "views": official_views,
            }
        if (
            DEFAULT_TABARENA_DETAILED_METRICS is not None
            and DEFAULT_TABARENA_DETAILED_METRICS.exists()
        ):
            data["tabarena_detailed_metric_comparison"] = (
                tabarena_protocol.detailed_metric_comparison(
                    selected_results,
                    DEFAULT_TABARENA_DETAILED_METRICS,
                )
            )
            if (
                DEFAULT_TABARENA_OFFICIAL_RESULTS is not None
                and DEFAULT_TABARENA_OFFICIAL_RESULTS.exists()
            ):
                data["tabarena_quadrant_analysis"] = build_tabarena_quadrant_analysis(
                    selected_results,
                    DEFAULT_TABARENA_OFFICIAL_RESULTS,
                    DEFAULT_TABARENA_DETAILED_METRICS,
                )
        data.setdefault("metadata", {})["tabarena_protocol"] = {
            "outer_cv": "repeated_stratified_3_fold",
            "inner_cv_folds": 8,
            "retained_child_models": 8,
            "selection": "binary_roc_auc_or_multiclass_log_loss",
            "outer_refit": False,
            "elo_bootstrap_rounds": 200,
            "elo_reference_model": "RandomForest standard",
            "model_agnostic_preprocessing": "AutoMLPipelineFeatureGenerator",
            "model_agnostic_preprocessing_version": "autogluon.features==1.5.0",
            "preprocessing_fit_scope": "each inner child training fold only",
            "preprocessing_validation_behavior": "transform with the corresponding fitted child generator",
            "preprocessing_outer_test_behavior": "transform independently through every retained child generator",
        }
    pair_csv = out_dir / f"{DEFAULT_OUTPUT_PREFIX}_pair_results.csv"
    split_csv = out_dir / f"{DEFAULT_OUTPUT_PREFIX}_official_split_results.csv"
    summary_csv = out_dir / f"{DEFAULT_OUTPUT_PREFIX}_model_summary.csv"
    complexity_points_csv = out_dir / f"{DEFAULT_OUTPUT_PREFIX}_complexity_points.csv"
    complexity_ratios_csv = out_dir / f"{DEFAULT_OUTPUT_PREFIX}_complexity_ratios.csv"
    hugiml_rpte_distributions_csv = (
        out_dir / f"{DEFAULT_OUTPUT_PREFIX}_hugiml_rpte_distributions.csv"
    )
    tabarena_leaderboard_csv = out_dir / f"{DEFAULT_OUTPUT_PREFIX}_leaderboard.csv"
    tabarena_official_leaderboard_csv = (
        out_dir / f"{DEFAULT_OUTPUT_PREFIX}_official_reference_leaderboard.csv"
    )
    tabarena_official_pairwise_csv = (
        out_dir / f"{DEFAULT_OUTPUT_PREFIX}_official_reference_pairwise.csv"
    )
    data_json = out_dir / f"{DEFAULT_OUTPUT_PREFIX}_dashboard_data.json"
    dashboard_html = out_dir / DEFAULT_DASHBOARD_NAME
    tabarena_official_dashboard_html = (
        out_dir / f"{DEFAULT_OUTPUT_PREFIX}_official_leaderboard_dashboard.html"
    )

    pd.DataFrame(pair_rows).to_csv(pair_csv, index=False)
    pd.DataFrame(split_rows).to_csv(split_csv, index=False)
    pd.DataFrame(data["overall"]).to_csv(summary_csv, index=False)
    pd.DataFrame(data.get("complexity_points", [])).to_csv(complexity_points_csv, index=False)
    pd.DataFrame(data.get("complexity_ratios", [])).to_csv(complexity_ratios_csv, index=False)
    pd.DataFrame(data.get("hugiml_rpte_distributions", [])).to_csv(
        hugiml_rpte_distributions_csv, index=False
    )
    if DEFAULT_TABARENA_REPORTING:
        pd.DataFrame(tabarena_rows).to_csv(tabarena_leaderboard_csv, index=False)
        official_comparison = data.get("tabarena_official_comparison", {})
        official_views = official_comparison.get("views", {})
        leaderboard_export = [
            {
                "view": view_name,
                "regime": comparison.get("regime"),
                "model_scope": comparison.get("model_scope"),
                **row,
            }
            for view_name, comparison in official_views.items()
            for row in comparison.get("leaderboard", [])
        ] or official_comparison.get("leaderboard", [])
        pairwise_export = [
            {
                "view": view_name,
                "regime": comparison.get("regime"),
                "model_scope": comparison.get("model_scope"),
                **row,
            }
            for view_name, comparison in official_views.items()
            for row in comparison.get("hugiml_pairwise", [])
        ] or official_comparison.get("hugiml_pairwise", [])
        pd.DataFrame(leaderboard_export).to_csv(tabarena_official_leaderboard_csv, index=False)
        pd.DataFrame(pairwise_export).to_csv(tabarena_official_pairwise_csv, index=False)
    atomic_write_json(data_json, data)
    if not DEFAULT_TABARENA_REPORTING:
        render_dashboard(
            data,
            dashboard_html,
            template_html,
            include_rpte=include_rpte_dashboard,
        )
    if DEFAULT_TABARENA_REPORTING and data.get("tabarena_official_comparison", {}).get("available"):
        render_tabarena_official_dashboard(data, tabarena_official_dashboard_html)
    outputs = {
        "pair_results_csv": pair_csv,
        "official_split_results_csv": split_csv,
        "model_summary_csv": summary_csv,
        "complexity_points_csv": complexity_points_csv,
        "complexity_ratios_csv": complexity_ratios_csv,
        "hugiml_rpte_distributions_csv": hugiml_rpte_distributions_csv,
        "dashboard_data_json": data_json,
    }
    if not DEFAULT_TABARENA_REPORTING:
        outputs["dashboard_html"] = dashboard_html
    if DEFAULT_TABARENA_REPORTING:
        outputs["tabarena_leaderboard_csv"] = tabarena_leaderboard_csv
        outputs["tabarena_official_leaderboard_csv"] = tabarena_official_leaderboard_csv
        outputs["tabarena_official_pairwise_csv"] = tabarena_official_pairwise_csv
        if tabarena_official_dashboard_html.exists():
            outputs["tabarena_official_dashboard_html"] = tabarena_official_dashboard_html
    return outputs


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Benchmark HUGIML, ensemble models, and interpretable models on official {DEFAULT_BENCHMARK_LABEL} splits"
    )
    parser.add_argument("--suite", default=DEFAULT_SUITE, help="OpenML suite alias or numeric ID")
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated task IDs. When omitted, use all suite tasks unless --smallest is set.",
    )
    parser.add_argument(
        "--smallest",
        type=int,
        default=DEFAULT_SMALLEST,
        help="Use only the requested number of smallest suite tasks; default uses all available tasks",
    )
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Comma-separated model names or families. Families: ensemble, interpretable, "
            "hugiml. Aliases: xgb, lightgbm, rf, lr, ebm, rulefit. Default: all."
        ),
    )
    parser.add_argument("--inner-splits", type=int, default=DEFAULT_INNER_SPLITS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--no-tune", action="store_true", help="Fit defaults without inner CV")
    parser.add_argument(
        "--validation-protocol",
        choices=VALIDATION_PROTOCOLS,
        default=DEFAULT_VALIDATION_PROTOCOL,
        help="Nested CV, rotating validation, or TabArena retained 8-fold CV ensemble",
    )
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable RPTE validation early stopping. Defaults off for nested CV "
            "and on for rotating validation."
        ),
    )
    parser.add_argument(
        "--max-official-splits",
        type=int,
        default=None,
        help="Development-only cap; default evaluates every OpenML-defined split",
    )
    parser.add_argument(
        "--hugiml-scenario",
        choices=sorted(benchmark_base.HUGIML_SCENARIOS),
        default=DEFAULT_HUGIML_SCENARIO,
    )
    parser.add_argument("--hugiml-max-fit-seconds", type=float, default=None)
    parser.add_argument(
        "--complexity-auc-tolerance",
        type=float,
        default=0.01,
        help=(
            "Maximum AUC shortfall allowed when labeling HUGIML comparable to a baseline "
            "for inspection-unit ratios"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / DEFAULT_RESULTS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-id", default=None, help="Checkpoint run ID to assemble")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-run-id",
        default=None,
        help=(
            "Resume an existing checkpoint run explicitly. All run settings except "
            "recorded source hashes must match the current invocation."
        ),
    )
    parser.add_argument(
        "--reuse-baseline-results",
        action="store_true",
        help=(
            "Deprecated compatibility alias. With --resume, all fitting-compatible "
            "completed results are reused automatically."
        ),
    )
    parser.add_argument(
        "--no-reuse-compatible-results",
        action="store_true",
        help="Do not import compatible completed results from retained checkpoint runs",
    )
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--assemble", action="store_true", help="Build CSV/JSON/HTML from checkpoint"
    )
    parser.add_argument(
        "--include-rpte-dashboard",
        action="store_true",
        help="Include the RPTE path/distribution section in the assembled dashboard",
    )
    parser.add_argument(
        "--template-html", type=Path, default=None, help="Custom dashboard template"
    )
    parser.add_argument(
        "--write-template",
        nargs="?",
        const=f"{DEFAULT_OUTPUT_PREFIX}_dashboard_template.html",
        default=None,
        help="Write the built-in HTML template and exit; optional output path",
    )
    offline = parser.add_argument_group("offline dataset selection")
    offline.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    offline.add_argument("--task-ids-file", type=Path, action="append", default=[])
    offline.add_argument(
        "--defer-task-ids",
        default="",
        help=(
            "Comma-separated task IDs to execute after every other selected task. "
            "This changes execution order only, not checkpoint identity."
        ),
    )
    offline.add_argument(
        "--first",
        type=int,
        default=None,
        help="Use the first N tasks in the benchmark manifest's canonical order",
    )
    offline.add_argument(
        "--verify-cache",
        action="store_true",
        help="Validate the complete offline cache and exit without fitting models",
    )
    offline.add_argument(
        "--verify-cache-before-run",
        action="store_true",
        help="Validate the complete offline cache before starting a benchmark",
    )
    return parser


def pair_error_row(
    *,
    task: dict[str, Any],
    model_name: str,
    tune: bool,
    inner_splits: int,
    hugiml_scenario: str,
    validation_protocol: str,
    early_stopping: bool,
    error: Exception,
) -> dict[str, Any]:
    return {
        "task_id": int(task["task_id"]),
        "dataset_id": int(task.get("dataset_id", -1)),
        "dataset": str(task.get("dataset_name", f"task-{task['task_id']}")),
        "model": model_name,
        "status": "error",
        "n_rows": int(task.get("n_rows", -1)),
        "raw_features": int(task.get("n_features", -1)),
        "model_features": int(task.get("n_features", -1)),
        "n_classes": int(task.get("n_classes", -1)),
        "official_splits_expected": 0,
        "official_splits_completed": 0,
        "official_splits_successful": 0,
        "error_count": 1,
        "last_error": f"{type(error).__name__}: {error}",
        "evaluation_protocol": (
            "tabarena_repeated_outer_3fold_inner_8fold_cv_ensemble_no_refit"
            if validation_protocol == "tabarena"
            else "rotating_fold_train_validation_test_no_refit"
            if validation_protocol == "rotating"
            else (
                "openml_defined_outer_splits_with_inner_cv_tuning"
                if tune
                else "openml_defined_outer_splits_no_inner_tuning"
            )
        ),
        "validation_protocol": validation_protocol,
        "early_stopping": bool(early_stopping),
        "inner_n_splits": (
            int(inner_splits) if tune and validation_protocol in {"nested", "tabarena"} else None
        ),
        "tuned": bool(tune),
        "hugiml_scenario": hugiml_scenario if is_hugiml_model(model_name) else None,
        "accuracy": None,
        "roc_auc": None,
        "auc": None,
        "log_loss": None,
        "balanced_accuracy": None,
        "avg_precision": None,
        "brier": None,
        "f1": None,
        "fit_ms": None,
        "tune_ms": None,
        "predict_ms": None,
        "complexity_ms": None,
        "fit_seconds": None,
        "tune_seconds": None,
        "predict_seconds": None,
        "complexity_seconds": None,
        "split_seconds": None,
        "phase_accounted_seconds": None,
        "evaluation_overhead_seconds": None,
        "total_fit_seconds": 0.0,
        "total_tune_seconds": 0.0,
        "total_predict_seconds": 0.0,
        "total_complexity_seconds": 0.0,
        "total_split_seconds": 0.0,
        "total_phase_accounted_seconds": 0.0,
        "total_evaluation_overhead_seconds": 0.0,
        "fit_time_included_in_tuning": None,
        "fit_time_accounting": "unavailable",
        "best_params": "{}",
        "best_params_json": "{}",
        "best_params_mode_json": "{}",
        "best_params_by_split_json": "[]",
        "best_params_frequency_json": "{}",
        "best_params_unique_count": 0,
        "inner_scoring_counts_json": "{}",
        "selection_refit_performed_split_count": 0,
        "retained_cv_ensemble_split_count": 0,
        "cv_ensemble_child_count_mean": None,
        "cv_ensemble_child_count_min": None,
        "cv_ensemble_child_count_max": None,
        "candidate_fit_count_mean": None,
        "candidate_fit_count_total": 0,
        "complexity": None,
        "complexity_model_units": None,
        "complexity_model_inspection_units": None,
        "complexity_instance_inspection_units_mean": None,
        "complexity_instance_inspection_units_std": None,
        "complexity_instance_inspection_units_standard_error": None,
        "complexity_instance_inspection_units_ci_lower": None,
        "complexity_instance_inspection_units_ci_upper": None,
        "complexity_instance_inspection_units_confidence_level": None,
        "complexity_instance_inspection_units_n_samples": None,
        "complexity_instance_inspection_units_sum": None,
        "complexity_instance_inspection_units_sum_squares": None,
        "complexity_instance_inspection_units_min": None,
        "complexity_instance_inspection_units_max": None,
        "split_rows": [],
    }


def benchmark_main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    effective_early_stopping = (
        bool(args.early_stopping)
        if args.early_stopping is not None
        else (
            args.validation_protocol == "rotating"
            or (args.validation_protocol == "tabarena" and DEFAULT_NESTED_EARLY_STOPPING)
        )
    )
    if args.validation_protocol == "tabarena" and not DEFAULT_TABARENA_REPORTING:
        raise ValueError(
            "The tabarena validation protocol is available only in the TabArena runner."
        )
    out_dir = args.out_dir.expanduser().resolve()
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else out_dir / DEFAULT_CHECKPOINT_NAME
    )
    template_html = args.template_html.expanduser().resolve() if args.template_html else None

    if args.write_template is not None:
        path = Path(args.write_template).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        print(write_template(path.resolve()))
        return 0

    if args.fresh and not args.assemble:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.assemble:
        outputs = assemble_outputs(
            checkpoint,
            out_dir,
            template_html,
            args.run_id,
            include_rpte_dashboard=args.include_rpte_dashboard,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.smallest is not None and args.smallest <= 0:
        raise ValueError("--smallest must be greater than zero")
    if args.complexity_auc_tolerance < 0:
        raise ValueError("--complexity-auc-tolerance must be zero or greater")
    configure_openml(
        args.cache_dir.expanduser().resolve() if args.cache_dir else out_dir / "openml_cache"
    )
    models = parse_models(args.models)
    explicit_task_ids = [int(x) for x in parse_csv(args.task_ids)]
    suite_meta, selected_tasks = resolve_suite_tasks(
        as_suite_identifier(args.suite),
        smallest=args.smallest,
        explicit_task_ids=explicit_task_ids,
        retries=args.retries,
    )
    if not selected_tasks:
        raise RuntimeError("No OpenML tasks selected")

    grid = grid_snapshot(
        args.hugiml_scenario,
        models,
        validation_protocol=args.validation_protocol,
    )
    source = {
        "script_sha256": sha256_file(Path(__file__)),
        "benchmark_dashboard_sha256": sha256_file(Path(benchmark_base.__file__)),
        "package_version": getattr(benchmark_base._hugiml_pkg, "__version__", None),
        "hugiml_fit_core_sha256": hugiml_fit_core_sha256(),
    }
    if explicit_task_ids:
        selection = "explicit_task_ids"
        selection_description = f"{len(selected_tasks)} explicitly selected tasks"
    elif args.smallest is not None:
        selection = f"smallest_{args.smallest}_by_NumberOfInstances"
        selection_description = f"{len(selected_tasks)} smallest available tasks"
    else:
        selection = "all_available_tasks"
        selection_description = f"all {len(selected_tasks)} available tasks"
    metadata = {
        **suite_meta,
        "selected_task_count": len(selected_tasks),
        "selection": selection,
        "selection_description": selection_description,
        "models": models,
        "model_order": models,
        "inner_splits": int(args.inner_splits),
        "tune": not args.no_tune,
        "random_state": int(args.random_state),
        "max_official_splits": args.max_official_splits,
        "validation_protocol": args.validation_protocol,
        "early_stopping": effective_early_stopping,
        "baseline_validation_early_stopping": bool(
            args.validation_protocol == "rotating" and not args.no_tune
        ),
        "baseline_early_stopping_max_estimators": (
            BASELINE_EARLY_STOPPING_MAX_ESTIMATORS
            if args.validation_protocol == "rotating"
            else None
        ),
        "baseline_early_stopping_patience": (
            BASELINE_EARLY_STOPPING_PATIENCE if args.validation_protocol == "rotating" else None
        ),
        "evaluation_protocol": (
            "Rotating K-fold train/validation/test without refit"
            if args.validation_protocol == "rotating"
            else "OpenML task-defined train/test indices with optional inner stratified CV"
        ),
        "primary_metric": "ROC AUC; multiclass OVR macro",
        "hugiml_scenario": args.hugiml_scenario,
        "hugiml_max_fit_seconds": args.hugiml_max_fit_seconds,
        "complexity_auc_tolerance": float(args.complexity_auc_tolerance),
        "methodology_note": DEFAULT_METHODOLOGY_NOTE,
        "grid_snapshot": grid,
        "source": source,
    }
    configuration = {
        "suite": safe_jsonable(suite_meta),
        "selection": selection,
        "tasks": [task_signature(task) for task in selected_tasks],
        "models": models,
        "inner_splits": int(args.inner_splits),
        "tune": not args.no_tune,
        "random_state": int(args.random_state),
        "max_official_splits": args.max_official_splits,
        "validation_protocol": args.validation_protocol,
        "early_stopping": effective_early_stopping,
        "baseline_validation_early_stopping": bool(
            args.validation_protocol == "rotating" and not args.no_tune
        ),
        "hugiml_scenario": args.hugiml_scenario,
        "hugiml_max_fit_seconds": args.hugiml_max_fit_seconds,
        "complexity_auc_tolerance": float(args.complexity_auc_tolerance),
        "grid_snapshot": grid,
        "source": source,
    }
    payload = load_checkpoint(checkpoint)
    if args.resume_run_id is not None:
        if not args.resume:
            raise ValueError("--resume-run-id requires --resume")
        requested_run_id = str(args.resume_run_id)
        existing_run = payload.get("runs", {}).get(requested_run_id)
        if existing_run is None:
            raise ValueError(f"Checkpoint run ID {requested_run_id!r} does not exist")
        existing_configuration = dict(existing_run.get("configuration", {}))
        current_comparable = dict(configuration)
        existing_configuration.pop("source", None)
        current_comparable.pop("source", None)
        if safe_jsonable(existing_configuration) != safe_jsonable(current_comparable):
            raise ValueError(
                "--resume-run-id settings do not match the stored run configuration. "
                "Tasks, models, grids, protocol, and tuning settings must be identical."
            )
        run_id = requested_run_id
        payload["active_run_id"] = run_id
        payload["metadata"] = dict(existing_run.get("metadata", {}))
        payload["selected_tasks"] = [
            dict(task) for task in existing_run.get("selected_tasks", selected_tasks)
        ]
    else:
        run_id = activate_run(
            payload,
            configuration=configuration,
            metadata=metadata,
            selected_tasks=selected_tasks,
        )
    if not args.resume:
        payload["results"] = [row for row in payload["results"] if str(row.get("run_id")) != run_id]
    reused_by_model: dict[str, int] = {}
    if args.resume and not args.no_reuse_compatible_results:
        reused_by_model = reuse_completed_compatible_results(
            payload,
            target_run_id=run_id,
            task_ids=[int(task["task_id"]) for task in selected_tasks],
            models=models,
        )
        reused_total = int(sum(reused_by_model.values()))
        payload["runs"][run_id]["metadata"]["reused_compatible_results"] = reused_total
        payload["runs"][run_id]["metadata"]["reused_compatible_results_by_model"] = reused_by_model
        payload["metadata"]["reused_compatible_results"] = reused_total
        payload["metadata"]["reused_compatible_results_by_model"] = reused_by_model
        if reused_total:
            detail = ", ".join(f"{model}: {count}" for model, count in reused_by_model.items())
            print(f"reused {reused_total} compatible completed result(s) ({detail})", flush=True)
    atomic_write_json(checkpoint, payload)

    deferred_task_ids = {int(value) for value in parse_csv(args.defer_task_ids)}
    selected_task_ids = {int(task["task_id"]) for task in selected_tasks}
    unknown_deferred = deferred_task_ids - selected_task_ids
    if unknown_deferred:
        raise ValueError(
            "--defer-task-ids contains task IDs outside the selected panel: "
            + ", ".join(str(value) for value in sorted(unknown_deferred))
        )
    done = completed_pair_keys(payload, run_id) if args.resume else set()
    execution_tasks = order_execution_tasks(
        selected_tasks,
        models=models,
        completed_pairs=done,
        deferred_task_ids=deferred_task_ids,
    )
    plan = [(int(task["task_id"]), model) for task in execution_tasks for model in models]
    plan = [item for item in plan if pair_key(*item) not in done]
    pending_task_ids = list(dict.fromkeys(task_id for task_id, _ in plan))
    completed_task_count = len(execution_tasks) - len(pending_task_ids)
    dataset_position_by_id = {
        task_id: completed_task_count + position
        for position, task_id in enumerate(pending_task_ids, start=1)
    }
    pending_by_model: dict[str, int] = {}
    for _, model in plan:
        pending_by_model[model] = pending_by_model.get(model, 0) + 1
    pending_detail = ", ".join(f"{model}: {pending_by_model.get(model, 0)}" for model in models)
    print(f"pending {len(plan)} pair(s) ({pending_detail})", flush=True)
    if args.start_pair:
        plan = plan[int(args.start_pair) :]
    if args.max_pairs is not None:
        plan = plan[: int(args.max_pairs)]

    selected_by_id = {int(row["task_id"]): row for row in selected_tasks}
    for task_id, model_name in plan:
        key = pair_key(task_id, model_name)
        task_label = selected_by_id[task_id]["dataset_name"]
        position = dataset_position_by_id[task_id]
        progress = f"dataset {position}/{len(execution_tasks)}"
        if key in done:
            print(f"skip {progress} task {task_id} ({task_label}) :: {model_name}", flush=True)
            continue
        print(f"run {progress} task {task_id} ({task_label}) :: {model_name}", flush=True)
        started = time.perf_counter()
        try:
            result = run_pair(
                task_id=task_id,
                model_name=model_name,
                out_dir=out_dir,
                retries=args.retries,
                tune=not args.no_tune,
                inner_splits=args.inner_splits,
                random_state=args.random_state,
                max_official_splits=args.max_official_splits,
                hugiml_scenario=args.hugiml_scenario,
                hugiml_max_fit_seconds=args.hugiml_max_fit_seconds,
                run_id=run_id,
                resume=args.resume,
                validation_protocol=args.validation_protocol,
                early_stopping=effective_early_stopping,
            )
        except Exception as exc:
            result = pair_error_row(
                task=selected_by_id[task_id],
                model_name=model_name,
                tune=not args.no_tune,
                inner_splits=args.inner_splits,
                hugiml_scenario=args.hugiml_scenario,
                validation_protocol=args.validation_protocol,
                early_stopping=effective_early_stopping,
                error=exc,
            )
        result["run_id"] = run_id
        result["pair_seconds"] = float(time.perf_counter() - started)
        payload["results"] = [
            row
            for row in payload["results"]
            if not (
                str(row.get("run_id")) == run_id
                and pair_key(int(row["task_id"]), str(row["model"])) == key
            )
        ]
        payload["results"].append(result)
        update_active_run_metadata(payload, updated_at_utc=utc_now())
        atomic_write_json(checkpoint, payload)
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "dataset": task_label,
                    "model": model_name,
                    "roc_auc": result.get("roc_auc"),
                    "balanced_accuracy": result.get("balanced_accuracy"),
                    "official_splits_successful": result.get("official_splits_successful"),
                    "errors": result.get("error_count"),
                    "pair_seconds": result.get("pair_seconds"),
                },
                default=json_default,
            ),
            flush=True,
        )
    return 0


# Offline dataset adapter and command-line entry point
benchmark = sys.modules[__name__]

PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASETS_DIR = PROJECT_ROOT / "datasetsOpenml"
DOWNLOAD_SCRIPT = SCRIPT_DIR / "download_openml_cc18_datasets.py"


def download_hint(datasets_dir: Path) -> str:
    return (
        "Offline datasets are not available. Download them before running the benchmark:\n"
        f'  "{sys.executable}" "{DOWNLOAD_SCRIPT}" --output "{datasets_dir}"'
    )


def read_manifest(datasets_dir: Path) -> dict[str, Any]:
    manifest_path = datasets_dir / "manifest.json"
    if not datasets_dir.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(download_hint(datasets_dir))
    try:
        manifest = read_json_with_backup(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"The offline dataset manifest is unreadable: {manifest_path}\n"
            + download_hint(datasets_dir)
        ) from exc
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(
            f"The offline dataset manifest contains no datasets: {manifest_path}\n"
            + download_hint(datasets_dir)
        )
    return manifest


def verify_dataset_manifest(datasets_dir: Path, manifest: dict[str, Any]) -> None:
    datasets = manifest.get("datasets", [])
    task_ids = [int(entry["task_id"]) for entry in datasets]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("The offline dataset manifest contains duplicate task IDs")
    if manifest.get("complete") is not True:
        raise ValueError("The offline dataset manifest is not marked complete")
    if int(manifest.get("dataset_count", len(datasets))) != len(datasets):
        raise ValueError("The offline dataset manifest dataset count does not match its entries")
    requested = manifest.get("requested_dataset_count")
    if requested is not None and int(requested) != len(datasets):
        raise ValueError("The offline dataset manifest requested count does not match its entries")
    for entry in datasets:
        validate_dataset_entry(datasets_dir, entry, verify_integrity=True)


def parse_task_id_file(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Task-ID file does not exist: {path}")
    values: list[int] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            try:
                values.append(int(token))
            except ValueError as exc:
                raise ValueError(f"Invalid task ID {token!r} in {path}:{line_number}") from exc
    return values


def unique(values: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


class OfflineTask:
    def __init__(self, folder: Path, metadata: dict[str, Any]) -> None:
        self.folder = folder
        self.metadata = metadata
        self.task_id = int(metadata["task_id"])
        self.dataset_id = int(metadata["dataset_id"])
        self.target_name = str(metadata["target_name"])
        self.estimation_procedure_type = "OpenML-defined (offline cache)"
        frame = pd.read_csv(folder / metadata["split_manifest_file"])
        self._splits = {
            (int(row.repeat), int(row.fold), int(row.sample)): row
            for row in frame.itertuples(index=False)
        }

    def get_split_dimensions(self) -> tuple[int, int, int]:
        return (
            int(self.metadata["official_repeats"]),
            int(self.metadata["official_folds"]),
            int(self.metadata["official_samples"]),
        )

    def get_train_test_split_indices(
        self, *, fold: int, repeat: int, sample: int
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            row = self._splits[(int(repeat), int(fold), int(sample))]
        except KeyError as exc:
            raise KeyError(
                f"Missing offline split repeat={repeat}, fold={fold}, sample={sample} "
                f"for task {self.task_id}"
            ) from exc
        return (
            np.load(self.folder / row.train_file, allow_pickle=False),
            np.load(self.folder / row.test_file, allow_pickle=False),
        )


def validate_dataset_entry(
    datasets_dir: Path,
    entry: dict[str, Any],
    *,
    verify_integrity: bool = False,
) -> Path:
    folder = datasets_dir / str(entry["folder"])
    required = [
        folder / "metadata.json",
        folder / str(entry["features_file"]),
        folder / str(entry["target_file"]),
        folder / str(entry["split_manifest_file"]),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Offline dataset files are incomplete:\n  "
            + "\n  ".join(missing)
            + "\n"
            + download_hint(datasets_dir)
        )
    if not verify_integrity:
        return folder
    metadata = read_json_with_backup(folder / "metadata.json")
    if int(metadata.get("task_id", -1)) != int(entry["task_id"]):
        raise ValueError(f"Offline task metadata does not match manifest entry {entry['task_id']}")
    expected_hashes = metadata.get("sha256") or entry.get("sha256") or {}
    for file_name, legacy_key in (
        (metadata["features_file"], "features.parquet"),
        (metadata["target_file"], "target.parquet"),
    ):
        expected = expected_hashes.get(str(file_name), expected_hashes.get(legacy_key))
        if expected and sha256_file(folder / str(file_name)) != expected:
            raise ValueError(
                f"Offline dataset checksum mismatch for task {entry['task_id']}: {file_name}"
            )
    split_frame = pd.read_csv(folder / str(metadata["split_manifest_file"]))
    if len(split_frame) != int(metadata["official_split_count"]):
        raise ValueError(f"Offline split count mismatch for task {entry['task_id']}")
    for split in split_frame.itertuples(index=False):
        for relative in (split.train_file, split.test_file):
            if not (folder / str(relative)).is_file():
                raise FileNotFoundError(
                    f"Offline split file is missing for task {entry['task_id']}: {relative}"
                )
    return folder


def select_offline_entries(
    entries: dict[int, dict[str, Any]],
    *,
    smallest: int | None,
    first: int | None = None,
    explicit_task_ids: Sequence[int],
) -> list[dict[str, Any]]:
    requested = unique(explicit_task_ids)
    if requested:
        missing = [task_id for task_id in requested if task_id not in entries]
        if missing:
            available = ", ".join(str(task_id) for task_id in sorted(entries))
            raise ValueError(
                f"Task IDs are not present in the offline dataset folder: {missing}. "
                f"Available task IDs: {available}"
            )
        return [dict(entries[task_id]) for task_id in requested]
    if first is not None:
        selected = [dict(entry) for entry in entries.values()]
        return selected[: int(first)]
    selected = sorted(
        (dict(entry) for entry in entries.values()),
        key=lambda entry: (
            int(entry.get("n_rows", -1)) < 0,
            int(entry.get("n_rows", -1)),
            int(entry["task_id"]),
        ),
    )
    return selected if smallest is None else selected[: int(smallest)]


def install_offline_loaders(
    datasets_dir: Path,
    manifest: dict[str, Any],
    *,
    verify_cache: bool = False,
) -> dict[int, dict[str, Any]]:
    entries = {int(entry["task_id"]): entry for entry in manifest["datasets"]}

    def resolve_suite_tasks(
        suite_id: str | int,
        *,
        smallest: int | None,
        explicit_task_ids: Sequence[int],
        retries: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        del suite_id, retries
        selected = select_offline_entries(
            entries,
            smallest=smallest,
            first=None,
            explicit_task_ids=explicit_task_ids,
        )
        suite_meta = {
            "suite_id": manifest.get("suite_id"),
            "suite_name": manifest.get("suite", DEFAULT_BENCHMARK_LABEL),
            "suite_alias": manifest.get("suite", DEFAULT_BENCHMARK_LABEL),
            "suite_task_count": len(entries),
            "data_source": "offline_dataset_cache",
            "datasets_dir": datasets_dir.name or f"{DEFAULT_OUTPUT_PREFIX}_datasets",
        }
        return suite_meta, selected

    def load_task(
        task_id: int, *, retries: int
    ) -> tuple[Any, pd.DataFrame, np.ndarray, dict[str, Any]]:
        del retries
        entry = entries[int(task_id)]
        folder = validate_dataset_entry(datasets_dir, entry, verify_integrity=verify_cache)
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        features_path = folder / metadata["features_file"]
        if features_path.suffix.lower() == ".npy":
            feature_array = np.load(features_path, allow_pickle=False)
            if feature_array.ndim != 2:
                raise ValueError(
                    f"Offline feature array must be two-dimensional for task {task_id}"
                )
            features = pd.DataFrame(feature_array)
        else:
            features = pd.read_parquet(features_path)
        target = pd.read_parquet(folder / metadata["target_file"])
        y = target["target_encoded"].to_numpy(dtype=np.int64, copy=True)
        if len(features) != len(y) or len(y) != int(metadata["n_rows"]):
            raise ValueError(f"Offline row-count mismatch for task {task_id}")
        features = benchmark.benchmark_base._force_writable_frame(features)
        return OfflineTask(folder, metadata), features, y, metadata

    benchmark.configure_openml = lambda cache_dir: None
    benchmark.resolve_suite_tasks = resolve_suite_tasks
    benchmark.load_task = load_task
    return entries


def strip_option(argv: Sequence[str], option: str) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == option:
            index += 2
            continue
        if item.startswith(option + "="):
            index += 1
            continue
        result.append(item)
        index += 1
    return result


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    pre_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre_parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    pre_parser.add_argument("--task-ids-file", type=Path, action="append", default=[])
    pre_parser.add_argument(
        "--first",
        type=int,
        default=None,
        help="Use the first N tasks in the benchmark manifest's canonical order",
    )
    pre_parser.add_argument(
        "--verify-cache",
        action="store_true",
        help="Validate cache metadata, checksums, and every official split file, then exit",
    )
    pre_parser.add_argument(
        "--verify-cache-before-run",
        action="store_true",
        help="Validate the complete offline cache before starting the requested benchmark",
    )
    offline_args, benchmark_argv = pre_parser.parse_known_args(raw_argv)
    datasets_dir = offline_args.datasets_dir.expanduser().resolve()

    parsed_benchmark_args, _ = benchmark.build_parser().parse_known_args(benchmark_argv)
    if parsed_benchmark_args.assemble or parsed_benchmark_args.write_template is not None:
        return benchmark_main(benchmark_argv)

    try:
        if parsed_benchmark_args.smallest is not None and parsed_benchmark_args.smallest <= 0:
            raise ValueError("--smallest must be greater than zero")
        if offline_args.first is not None and offline_args.first <= 0:
            raise ValueError("--first must be greater than zero")
        if offline_args.first is not None and parsed_benchmark_args.smallest is not None:
            raise ValueError("--first and --smallest are mutually exclusive")
        manifest = read_manifest(datasets_dir)
        if offline_args.verify_cache or offline_args.verify_cache_before_run:
            verify_dataset_manifest(datasets_dir, manifest)
        if offline_args.verify_cache:
            print(
                f"Offline cache verified: {len(manifest['datasets'])} task(s) in {datasets_dir}",
                flush=True,
            )
            return 0
        entries = install_offline_loaders(
            datasets_dir,
            manifest,
            verify_cache=False,
        )
        task_ids = [int(value) for value in benchmark.parse_csv(parsed_benchmark_args.task_ids)]
        for path in offline_args.task_ids_file:
            task_ids.extend(parse_task_id_file(path.expanduser().resolve()))
        task_ids = unique(task_ids)
        selected = select_offline_entries(
            entries,
            smallest=parsed_benchmark_args.smallest,
            first=offline_args.first,
            explicit_task_ids=task_ids,
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
        print(f"Cannot start offline benchmark.\n{exc}", file=sys.stderr)
        return 2

    forwarded = list(benchmark_argv)
    if task_ids or offline_args.first is not None:
        forwarded = strip_option(forwarded, "--task-ids")
        forwarded = strip_option(forwarded, "--smallest")
        forwarded.extend(["--task-ids", ",".join(str(int(entry["task_id"])) for entry in selected)])
    selected_ids = [int(entry["task_id"]) for entry in selected]
    print(
        f"Offline benchmark: {len(selected_ids)} task(s) from {datasets_dir}\n"
        f"Task IDs: {','.join(str(task_id) for task_id in selected_ids)}",
        flush=True,
    )
    return benchmark_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())

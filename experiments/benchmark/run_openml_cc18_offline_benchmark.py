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
import hashlib
import html
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = next(
    (p for p in [SCRIPT_DIR, *SCRIPT_DIR.parents] if (p / "src" / "hugiml").exists()),
    None,
)
if SOURCE_ROOT is not None:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
    sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from scipy.stats import t as student_t
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, label_binarize

import benchmark_dashboard as benchmark_base

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
DEFAULT_SUITE = "OpenML-CC18"
DEFAULT_SMALLEST: int | None = None
DEFAULT_INNER_SPLITS = 3
DEFAULT_RANDOM_STATE = 42
DEFAULT_HUGIML_SCENARIO = "augmented_pair"
DEFAULT_RESULTS_DIR = "openml_cc18_results"
DEFAULT_CHECKPOINT_NAME = "openml_cc18_checkpoint.json"
DEFAULT_DASHBOARD_NAME = "openml_cc18_benchmark_dashboard.html"


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
                "dataset_name": str(_first_present(row, ["name", "data_name"]) or f"task-{int(tid)}"),
                "n_rows": int(float(_first_present(row, ["NumberOfInstances", "number_instances"]) or -1)),
                "n_features": int(float(_first_present(row, ["NumberOfFeatures", "number_features"]) or -1)),
                "n_classes": int(float(_first_present(row, ["NumberOfClasses", "number_classes"]) or -1)),
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
        "dataset_name": str(getattr(dataset, "name", f"task-{getattr(task, 'task_id', 'unknown')}")),
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
        "suite_id": int(getattr(suite, "id", -1)) if str(getattr(suite, "id", "")).isdigit() else getattr(suite, "id", None),
        "suite_name": str(getattr(suite, "name", suite_id)),
        "suite_alias": str(suite_id),
        "suite_task_count": len(suite_task_ids),
    }
    return suite_meta, metadata


def load_task(task_id: int, *, retries: int) -> tuple[Any, pd.DataFrame, np.ndarray, dict[str, Any]]:
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
        if marked or not pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col]):
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
            "estimation_procedure": str(getattr(task, "estimation_procedure_type", "") or "OpenML-defined"),
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
    if train_idx.min() < 0 or test_idx.min() < 0 or train_idx.max() >= n_rows or test_idx.max() >= n_rows:
        raise IndexError(f"OpenML split {split['split_id']} contains indices outside [0, {n_rows})")


# ---------------------------------------------------------------------------
# Model tuning, prediction, and metrics
# ---------------------------------------------------------------------------


def inner_scoring(y: np.ndarray) -> str:
    return "roc_auc" if np.unique(y).size == 2 else "roc_auc_ovr"


class DenseArrayTransformer(BaseEstimator, TransformerMixin):
    """Convert preprocessed baseline features to a dense numeric array."""

    def fit(self, X: Any, y: Any = None) -> "DenseArrayTransformer":
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = X.toarray() if sparse.issparse(X) else X
        return np.array(values, dtype=np.float32, order="C", copy=True)


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
    final_refit_ms = (
        float(search.refit_time_) * 1000.0 if hasattr(search, "refit_time_") else None
    )
    params = benchmark_base._strip_model_prefix(dict(search.best_params_))
    return (
        search.best_estimator_,
        params,
        float(search.best_score_),
        float(tune_ms),
        {"_final_refit_ms": final_refit_ms, "inner_scoring": inner_scoring(y_train)},
    )


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
) -> tuple[Any, dict[str, Any], float | None, float, float, dict[str, Any], Any]:
    benchmark_base.RANDOM_STATE = int(random_state)
    candidates, builder, complexity_fn, budget = benchmark_base.get_model_spec(
        model_name,
        hugiml_scenario=hugiml_scenario if model_name == "HUGIML" else None,
        hugiml_max_fit_seconds=hugiml_max_fit_seconds,
    )
    if model_name == "XGB standard" and benchmark_base.XGBClassifier is None:
        raise ImportError("xgboost is required for XGB standard")
    if model_name == "LightGBM standard" and benchmark_base.LGBMClassifier is None:
        raise ImportError("lightgbm is required for LightGBM standard")

    if tune:
        if model_name == "HUGIML":
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
        return estimator, params, finite_or_none(best_score), fit_ms, tune_ms, info, complexity_fn

    estimator = builder({}) if model_name == "HUGIML" else baseline_pipeline(builder({}))
    started = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_ms = (time.perf_counter() - started) * 1000.0
    return estimator, {}, None, fit_ms, 0.0, {"inner_scoring": None}, complexity_fn


def probability_matrix(estimator: Any, X: pd.DataFrame, n_classes: int) -> np.ndarray:
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
    if proba.shape[1] != n_classes or not np.array_equal(classes.astype(int, copy=False), np.arange(n_classes)):
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
    try:
        if n_classes == 2:
            out["roc_auc"] = finite_or_none(roc_auc_score(y_true, proba[:, 1]))
            out["avg_precision"] = finite_or_none(average_precision_score(y_true, proba[:, 1]))
            out["brier"] = finite_or_none(np.mean((proba[:, 1] - y_true) ** 2))
        else:
            out["roc_auc"] = finite_or_none(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
            )
            y_bin = label_binarize(y_true, classes=np.arange(n_classes))
            out["avg_precision"] = finite_or_none(
                average_precision_score(y_bin, proba, average="macro")
            )
            out["brier"] = finite_or_none(np.mean(np.sum((proba - y_bin) ** 2, axis=1)))
    except Exception:
        out.setdefault("roc_auc", None)
        out.setdefault("avg_precision", None)
        out.setdefault("brier", None)
    return out


def complexity_fields(estimator: Any, X_test: pd.DataFrame, complexity_fn) -> dict[str, Any]:
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
) -> tuple[dict[str, Any], int | None]:
    validate_official_split(split, len(y))
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    X_train = benchmark_base._force_writable_frame(X.iloc[train_idx].reset_index(drop=True))
    X_test = benchmark_base._force_writable_frame(X.iloc[test_idx].reset_index(drop=True))
    y_train = np.array(y[train_idx], dtype=int, copy=True)
    y_test = np.array(y[test_idx], dtype=int, copy=True)
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
        "inner_n_splits": int(inner_splits) if tune else None,
        "random_state": split_seed,
        "status": "error",
        "error_count": 1,
        "last_error": None,
    }
    try:
        estimator, params, best_inner_score, fit_ms, tune_ms, info, complexity_fn = fit_or_tune_model(
            model_name,
            X_train,
            y_train,
            tune=tune,
            inner_splits=inner_splits,
            random_state=split_seed,
            hugiml_scenario=hugiml_scenario,
            hugiml_max_fit_seconds=hugiml_max_fit_seconds,
        )
        started = time.perf_counter()
        proba = probability_matrix(estimator, X_test, int(task_meta["n_classes"]))
        predict_ms = (time.perf_counter() - started) * 1000.0
        row.update(classification_metrics(y_test, proba))
        row.update(complexity_fields(estimator, X_test, complexity_fn))
        row.update(
            {
                "auc": row.get("roc_auc"),
                "best_inner_score": best_inner_score,
                "best_params": json.dumps(params, sort_keys=True, default=json_default),
                "best_params_json": json.dumps(params, sort_keys=True, default=json_default),
                "fit_ms": finite_or_none(fit_ms),
                "tune_ms": finite_or_none(tune_ms),
                "predict_ms": finite_or_none(predict_ms),
                "fit_seconds": finite_or_none(fit_ms / 1000.0),
                "tune_seconds": finite_or_none(tune_ms / 1000.0),
                "predict_seconds": finite_or_none(predict_ms / 1000.0),
                "tuned": bool(tune),
                "status": "ok",
                "error_count": 0,
                "last_error": None,
                "inner_scoring": info.get("inner_scoring", "roc_auc"),
            }
        )
        return row, benchmark_base._model_feature_count(estimator, X.shape[1])
    except Exception as exc:
        row["last_error"] = f"{type(exc).__name__}: {exc}"
        for key in (
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
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
            "fit_seconds",
            "tune_seconds",
            "predict_seconds",
        ):
            row.setdefault(key, None)
        row.setdefault("best_params", "{}")
        row.setdefault("best_params_json", "{}")
        return row, None


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
        expected = int(row.get("official_splits_expected", row.get("official_splits_completed", 0)) or 0)
        successful = int(row.get("official_splits_successful", 0) or 0)
        status = str(row.get("status", ""))
        if status == "ok" and expected > 0 and successful == expected:
            completed.add(pair_key(int(row["task_id"]), str(row["model"])))
    return completed


def aggregate_pair_rows(
    split_rows: list[dict[str, Any]],
    *,
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
        "fit_seconds",
        "tune_seconds",
        "predict_seconds",
    ]
    out: dict[str, Any] = {}
    for metric in metrics:
        out[metric] = mean_or_none(row.get(metric) for row in ok)
        out[f"{metric}_std"] = std_or_none(row.get(metric) for row in ok)

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
        if (value := finite_or_none(row.get("complexity_instance_inspection_units_confidence_level")))
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
            if (value := finite_or_none(row.get("complexity_instance_inspection_units_min"))) is not None
        ]
        maximums = [
            value
            for row in ok
            if (value := finite_or_none(row.get("complexity_instance_inspection_units_max"))) is not None
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
            "error_count": error_count,
            "last_error": next(
                (str(row.get("last_error")) for row in reversed(split_rows) if row.get("last_error")),
                None,
            ),
            "evaluation_protocol": "openml_defined_outer_splits_with_inner_cv_tuning"
            if tune
            else "openml_defined_outer_splits_no_inner_tuning",
            "inner_n_splits": int(inner_splits) if tune else None,
            "scoring": "roc_auc_binary_or_multiclass_ovr_macro",
            "tuned": bool(tune),
            "preprocessing_policy": (
                "HUGIML native pandas categorical handling"
                if model_name == "HUGIML"
                else "fold-local numeric imputation and categorical one-hot encoding Pipeline"
            ),
            "hugiml_scenario": hugiml_scenario if model_name == "HUGIML" else None,
            "split_rows": split_rows,
        }
    )
    return out


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
) -> dict[str, Any]:
    task, X, y, task_meta = load_task(task_id, retries=retries)
    splits = official_splits(task, max_splits=max_official_splits)
    identity = {
        "run_id": run_id,
        "task_id": int(task_id),
        "dataset_id": int(task_meta["dataset_id"]),
        "model": model_name,
        "tune": bool(tune),
        "inner_splits": int(inner_splits),
        "random_state": int(random_state),
        "max_official_splits": max_official_splits,
        "hugiml_scenario": hugiml_scenario if model_name == "HUGIML" else None,
        "official_dimensions": [
            int(task_meta["official_repeats"]),
            int(task_meta["official_folds"]),
            int(task_meta["official_samples"]),
        ],
        "dataset_sha256": safe_jsonable(task_meta.get("sha256", {})),
    }
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    split_checkpoint = out_dir / "split_checkpoints" / f"{token}.json"
    split_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[int, dict[str, Any]] = {}
    feature_counts_by_split: dict[int, int] = {}
    if resume and split_checkpoint.exists():
        try:
            saved = read_json_with_backup(split_checkpoint)
            if saved.get("identity") == identity:
                for row in saved.get("split_rows", []):
                    if row.get("split_id") is not None:
                        rows_by_split[int(row["split_id"])] = dict(row)
                raw_counts = saved.get("model_feature_counts", {})
                if isinstance(raw_counts, dict):
                    feature_counts_by_split = {int(key): int(value) for key, value in raw_counts.items()}
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            rows_by_split = {}
            feature_counts_by_split = {}

    successful = {
        split_id for split_id, row in rows_by_split.items() if row.get("status") == "ok"
    }
    for split in splits:
        split_id = int(split["split_id"])
        if split_id in successful:
            continue
        previous = rows_by_split.get(split_id, {})
        print(
            f"  split r={split['repeat']} f={split['fold']} s={split['sample']}",
            flush=True,
        )
        row, feature_count = evaluate_official_split(
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
        )
        row["attempt_count"] = int(previous.get("attempt_count", 0)) + 1
        row["attempted_at_utc"] = utc_now()
        rows_by_split[split_id] = row
        if feature_count is not None:
            feature_counts_by_split[split_id] = int(feature_count)
        elif row.get("status") != "ok":
            feature_counts_by_split.pop(split_id, None)
        split_rows = [rows_by_split[key] for key in sorted(rows_by_split)]
        atomic_write_json(
            split_checkpoint,
            {
                "identity": identity,
                "task_meta": task_meta,
                "split_rows": split_rows,
                "model_feature_counts": {
                    str(key): feature_counts_by_split[key] for key in sorted(feature_counts_by_split)
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
        task_meta=task_meta,
        model_name=model_name,
        model_feature_counts=[feature_counts_by_split[key] for key in sorted(feature_counts_by_split)],
        tune=tune,
        inner_splits=inner_splits,
        hugiml_scenario=hugiml_scenario,
        official_splits_expected=len(splits),
    )

def grid_snapshot(hugiml_scenario: str, models: Sequence[str] | None = None) -> dict[str, Any]:
    selected = list(models or MODEL_ORDER)
    snapshot: dict[str, Any] = {}
    if "HUGIML" in selected:
        hug_grid_name, hug_candidates = benchmark_base._hugiml_grid_for_scenario(hugiml_scenario)
        snapshot["HUGIML"] = {
            "scenario": hugiml_scenario,
            "grid_name": hug_grid_name,
            "candidate_count": len(hug_candidates),
            "grid": benchmark_base.get_hugiml_grid(hug_grid_name),
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
        grid = benchmark_base.get_baseline_grid(family) or {}
        snapshot[label] = {
            "candidate_count": len(list(benchmark_base.ParameterGrid(grid))) if grid else 1,
            "grid": grid,
            "constants": benchmark_base.baseline_constant_parameters(family),
        }
    return safe_jsonable(snapshot)


def configuration_id(configuration: dict[str, Any]) -> str:
    canonical = json.dumps(safe_jsonable(configuration), sort_keys=True, separators=(",", ":"), allow_nan=False)
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
    if not payload["runs"] and (payload["metadata"] or payload["selected_tasks"] or payload["results"]):
        configuration = {
            "metadata": safe_jsonable(payload["metadata"]),
            "task_ids": [int(row["task_id"]) for row in payload["selected_tasks"] if row.get("task_id") is not None],
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
                row["status"] = "ok" if expected > 0 and expected == successful and not row.get("error_count") else "partial"
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


def build_dashboard_data(payload: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    view = run_view(normalize_checkpoint(payload), run_id)
    results = [dict(row) for row in view.get("results", [])]
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
    if pair_df.empty:
        raise ValueError("Checkpoint run contains no results for its selected tasks")
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
        aucs = pd.to_numeric(subset.get("roc_auc", pd.Series(dtype=float)), errors="coerce").dropna()
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
                "mean_fit_seconds": mean_or_none(subset.get("fit_seconds", [])),
                "mean_tune_seconds": mean_or_none(subset.get("tune_seconds", [])),
                "mean_predict_seconds": mean_or_none(subset.get("predict_seconds", [])),
                "mean_complexity": None if not valid_inspection else float(np.mean(valid_inspection)),
                "mean_model_units": None if not valid_model_units else float(np.mean(valid_model_units)),
                "median_model_units": None if not valid_model_units else float(np.median(valid_model_units)),
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
            friedman.update(
                {"statistic": finite_or_none(stat), "p_value": finite_or_none(p_value)}
            )
        except Exception:
            pass

    versus_hugiml: list[dict[str, Any]] = []
    raw_p: list[float] = []
    comparison_models = (
        [model for model in active_models if model != "HUGIML"]
        if "HUGIML" in active_models
        else []
    )
    for model in comparison_models:
        pairs: list[tuple[float, float]] = []
        for task in per_task:
            scores = {row["model"]: row.get("roc_auc") for row in task["rows"]}
            hug_auc = finite_or_none(scores.get("HUGIML"))
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
                "comparison": f"HUGIML vs {model}",
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
        row["significant_holm_0_05"] = bool(
            p_adj < 0.05 and row.get("p_value") is not None
        )

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
        hug = by_model.get("HUGIML")
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
    for model in comparison_models:
        rows = [row for row in complexity_ratios if row["baseline_model"] == model]
        comparable = [row for row in rows if row["comparable_or_better_auc"]]
        ratios = [float(row["inspection_ratio_baseline_to_hugiml"]) for row in comparable]
        hug_units = [float(row["hugiml_model_inspection_units"]) for row in comparable]
        baseline_units = [float(row["baseline_model_inspection_units"]) for row in comparable]
        hug_auc_values = [float(row["hugiml_auc"]) for row in comparable]
        baseline_auc_values = [float(row["baseline_auc"]) for row in comparable]
        median_ratio = None if not ratios else float(np.median(ratios))
        summary = {
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
                f"Across {len(comparable)} tasks where HUGIML AUC was within {auc_tolerance:.3f} "
                f"of or above {model}, mean AUC was {np.mean(hug_auc_values):.3f} versus "
                f"{np.mean(baseline_auc_values):.3f}, with median model inspection units "
                f"{np.median(hug_units):.1f} versus {np.median(baseline_units):.1f} "
                f"({median_ratio:.2f}× baseline-to-HUGIML ratio)."
            )
        else:
            summary["headline"] = None
        complexity_summary.append(summary)

    for task in per_task:
        task_ratios = {
            row["baseline_model"]: row
            for row in complexity_ratios
            if int(row["task_id"]) == int(task["task_id"])
        }
        for row in task["rows"]:
            if row["model"] == "HUGIML":
                row["inspection_ratio_to_hugiml"] = 1.0
                row["comparable_or_better_auc_to_hugiml"] = True
            elif row["model"] in task_ratios:
                ratio_row = task_ratios[row["model"]]
                row["inspection_ratio_to_hugiml"] = ratio_row[
                    "inspection_ratio_baseline_to_hugiml"
                ]
                row["comparable_or_better_auc_to_hugiml"] = ratio_row[
                    "comparable_or_better_auc"
                ]

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

    return safe_jsonable(
        {
            "run_id": view["run_id"],
            "metadata": view.get("metadata", {}),
            "selected_tasks": view.get("selected_tasks", []),
            "model_order": active_models,
            "model_families": MODEL_FAMILIES,
            "overall": aggregate_rows,
            "per_task": per_task,
            "friedman": friedman,
            "versus_hugiml": versus_hugiml,
            "complexity_auc_tolerance": auc_tolerance,
            "complexity_points": complexity_points,
            "complexity_ratios": complexity_ratios,
            "complexity_summary": complexity_summary,
            "pair_results": [
                {k: v for k, v in row.items() if k != "split_rows"}
                for row in filtered_results
            ],
            "split_result_count": len(split_rows),
            "generated_at_utc": utc_now(),
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
.empty{color:var(--muted);font-style:italic}
@media(max-width:1180px){.summary-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.grid-2{grid-template-columns:1fr}.overall-tables-grid{grid-template-columns:1fr}.claims-grid{grid-template-columns:1fr}.section-title{align-items:flex-start;flex-direction:column}.meta{text-align:left}}
@media(max-width:760px){.wrap{padding:14px}.summary-grid{grid-template-columns:repeat(2,minmax(0,1fr))}h1{font-size:28px}.hero{padding:22px}.theme-switcher{width:100%;justify-content:flex-start}.kv{grid-template-columns:1fr}.plot{min-height:390px}}
.js-plotly-plot .scatterlayer .trace path{vector-effect:non-scaling-stroke}
</style>
</head>
<meta name="generated-at" content="{{GENERATED_AT}}" hidden>
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
      <p class="note" style="margin:0 0 8px">Ratios are baseline model inspection units divided by HUGIML model inspection units, on tasks where HUGIML AUC is within tolerance.</p>
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

<section class="grid-2">
  <div class="card"><div class="section-title"><h2>HUGIML paired tests</h2><div class="meta">Wilcoxon tests with Holm adjustment</div></div><div id="paired"></div></div>
  <div class="card"><div class="section-title"><h2>Evaluation protocol</h2><div class="meta">Run configuration</div></div><div id="protocol" class="kv"></div></div>
</section>

<section class="section card">
  <div class="section-title"><h2>Model search spaces</h2><div class="meta">Exact configuration recorded with this run</div></div>
  <details><summary>Show model grids</summary><pre id="grids"></pre></details>
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
function renderHero(){const m=DATA.metadata||{};const taskCount=(DATA.per_task||[]).length;const modelCount=(DATA.model_order||[]).length;const tuning=m.tune?`${m.inner_splits??3}-fold inner CV`:'Tuning disabled';const chips=[`Primary metric: AUC`,`OpenML-CC18: ${taskCount} of 72 tasks`,`Models: ${modelCount}`,tuning,`AUC tolerance: ${fmt(DATA.complexity_auc_tolerance,3)}`];document.getElementById('heroChips').innerHTML=chips.map(x=>`<span class="chip">${esc(x)}</span>`).join('')}
function renderMetrics(){const valid=(DATA.overall||[]).filter(r=>num(r.mean_auc)!=null).sort((a,b)=>b.mean_auc-a.mean_auc);const best=valid[0]||{};const hug=(DATA.overall||[]).find(r=>r.model==='HUGIML')||{};const comp=(DATA.complexity_summary||[]).find(r=>r.baseline_model==='XGB standard')||(DATA.complexity_summary||[]).find(r=>num(r.median_inspection_ratio_baseline_to_hugiml)!=null)||{};const fr=DATA.friedman||{};const cards=[['Best mean AUC',fmt(best.mean_auc),textOr(best.model)],['HUGIML mean AUC',fmt(hug.mean_auc),`mean rank ${textOr(fmt(hug.mean_rank,2))}`],['HUGIML median inspection',fmt(hug.median_model_inspection_units,1),'model inspection units'],['Comparable inspection ratio',num(comp.median_inspection_ratio_baseline_to_hugiml)==null?'Not available':fmt(comp.median_inspection_ratio_baseline_to_hugiml,2)+'×',comp.baseline_model?`vs ${comp.baseline_model} on ${comp.comparable_or_better_tasks||0} tasks`:'insufficient paired data'],['Friedman p-value',textOr(fmt(fr.p_value,5)),fr.n_tasks?`${fr.n_tasks} complete paired tasks`:'insufficient paired tasks']];document.getElementById('metrics').innerHTML=cards.map(x=>`<div class="card stat"><div class="label">${esc(x[0])}</div><div class="value ${String(x[1]).length>12?'small':''}">${esc(x[1])}</div><div class="sub">${esc(x[2])}</div></div>`).join('')}
function renderOverall(){const rows=DATA.overall||[];const best=Math.max(...rows.map(r=>num(r.mean_auc)??-Infinity));const hasCi=rows.length>0&&rows.every(r=>num(r.mean_instance_inspection_units)!=null&&num(r.instance_inspection_ci_lower)!=null&&num(r.instance_inspection_ci_upper)!=null);const hasInstance=rows.length>0&&rows.every(r=>num(r.mean_instance_inspection_units)!=null);const headers=['Model','Completed','Mean AUC','Std AUC','Mean rank','Wins','Mean model units','Mean model inspection','Median model inspection'];if(hasCi)headers.push('Mean instance inspection (95% CI)');else if(hasInstance)headers.push('Mean instance inspection');headers.push('Tune s','Fit s');const body=rows.map(r=>{const cells=[`<span class="pill">${esc(r.model)}</span>`,`${r.tasks_completed??0}/${r.tasks_attempted??r.tasks_completed??0}`,`<span class="${num(r.mean_auc)===best?'best':''}">${fmt(r.mean_auc)}</span>`,fmt(r.std_auc),fmt(r.mean_rank,2),String(r.wins??0),fmt(r.mean_model_units,1),fmt(r.mean_model_inspection_units,1),fmt(r.median_model_inspection_units,1)];if(hasCi)cells.push(ciText(r.mean_instance_inspection_units,r.instance_inspection_ci_lower,r.instance_inspection_ci_upper,1));else if(hasInstance)cells.push(fmt(r.mean_instance_inspection_units,1));cells.push(fmt(r.mean_tune_seconds,2),fmt(r.mean_fit_seconds,2));return `<tr>${cells.map(c=>`<td>${c}</td>`).join('')}</tr>`});document.getElementById('overallTable').innerHTML=tableMarkup(headers,body)}
function renderComplexity(){const rows=DATA.complexity_points||[];const fallback=document.getElementById('plotFallback');fallback.textContent='';if(!rows.length){fallback.textContent='No model-inspection values are available.';return}if(!window.Plotly){fallback.textContent='Interactive chart unavailable; the same values remain available in the tables and exported CSV files.';return}const models=[...new Set(rows.map(r=>r.model))];const traces=models.map(model=>{const sub=rows.filter(r=>r.model===model);return {type:'scatter',mode:'markers',name:model,x:sub.map(r=>r.model_inspection_units),y:sub.map(r=>r.roc_auc),text:sub.map(r=>`${r.dataset} · task ${r.task_id}`),customdata:sub.map(r=>[fmt(r.model_units,1),ciText(r.instance_inspection_units_mean,r.instance_inspection_units_ci_lower,r.instance_inspection_units_ci_upper,1)]),hovertemplate:'%{text}<br>Model inspection=%{x:.2f}<br>AUC=%{y:.4f}<br>Model units=%{customdata[0]}<br>Instance inspection=%{customdata[1]}<extra>%{fullData.name}</extra>',marker:{size:10,opacity:.78}}});const positive=rows.every(r=>Number(r.model_inspection_units)>0);const t=theme();Plotly.react('complexityPlot',traces,plotLayout({xaxis:{title:'Model inspection units',type:positive?'log':'linear',automargin:true,gridcolor:t.grid},yaxis:{title:'ROC AUC',range:[0,1.02],automargin:true,gridcolor:t.grid},hovermode:'closest'}),{responsive:true,displaylogo:false})}
function renderRatios(){const rows=DATA.complexity_summary||[];const headers=['Baseline','Comparable / paired','Comparable share','Mean AUC HUGIML','Mean AUC baseline','Median HUGIML inspection','Median baseline inspection','Median ratio','Median reduction'];const body=rows.map(r=>`<tr><td>${esc(r.baseline_model)}</td><td>${r.comparable_or_better_tasks??0}/${r.paired_tasks??0}</td><td>${pct(r.comparable_or_better_share)}</td><td>${fmt(r.mean_hugiml_auc)}</td><td>${fmt(r.mean_baseline_auc)}</td><td>${fmt(r.median_hugiml_model_inspection_units,1)}</td><td>${fmt(r.median_baseline_model_inspection_units,1)}</td><td class="${num(r.median_inspection_ratio_baseline_to_hugiml)>=1?'ratio-good':'ratio-warn'}">${num(r.median_inspection_ratio_baseline_to_hugiml)==null?'':fmt(r.median_inspection_ratio_baseline_to_hugiml,2)+'×'}</td><td>${num(r.median_hugiml_reduction_percent)==null?'':fmt(r.median_hugiml_reduction_percent,1)+'%'}</td></tr>`);document.getElementById('ratioTable').innerHTML=tableMarkup(headers,body);document.getElementById('claims').innerHTML=rows.filter(r=>r.headline).map(r=>`<div class="claim">${esc(r.headline)}</div>`).join('')||'<p class="note">No comparable task-level inspection ratios are available.</p>'}
function renderTask(){const id=Number(document.getElementById('taskSelect').value);const task=(DATA.per_task||[]).find(x=>x.task_id===id)||(DATA.per_task||[])[0];if(!task)return;document.getElementById('taskMeta').textContent=`OpenML task ${task.task_id} · ${task.n_rows} rows · ${task.n_classes} classes`;const allRows=(DATA.per_task||[]).flatMap(t=>t.rows||[]);const hasCi=allRows.length>0&&allRows.every(r=>num(r.complexity_instance_inspection_units_mean)!=null&&num(r.complexity_instance_inspection_units_ci_lower)!=null&&num(r.complexity_instance_inspection_units_ci_upper)!=null);const hasInstance=allRows.length>0&&allRows.every(r=>num(r.complexity_instance_inspection_units_mean)!=null);const headers=['Model','AUC','Rank','Model units','Model inspection'];if(hasCi)headers.push('Instance inspection (95% CI)');else if(hasInstance)headers.push('Instance inspection');headers.push('Baseline / HUGIML ratio','Tune s','Fit s','Splits','Errors');const body=(task.rows||[]).map(r=>{const comparable=r.comparable_or_better_auc_to_hugiml;const ratioClass=r.model==='HUGIML'?'':(comparable?'ratio-good':'ratio-warn');const cells=[esc(r.model),`<span class="${r.rank===1?'best':''}">${fmt(r.roc_auc)}</span>`,fmt(r.rank,2),fmt(r.complexity_model_units,1),fmt(r.model_inspection_units,1)];if(hasCi)cells.push(ciText(r.complexity_instance_inspection_units_mean,r.complexity_instance_inspection_units_ci_lower,r.complexity_instance_inspection_units_ci_upper,1));else if(hasInstance)cells.push(fmt(r.complexity_instance_inspection_units_mean,1));cells.push(`<span class="${ratioClass}">${num(r.inspection_ratio_to_hugiml)==null?'':fmt(r.inspection_ratio_to_hugiml,2)+'×'}</span>`,fmt(r.tune_seconds,2),fmt(r.fit_seconds,2),`${r.official_splits_successful??0}/${r.official_splits_expected??r.official_splits_completed??0}`,`<span class="${r.error_count?'error':''}">${r.error_count??0}</span>`);return `<tr>${cells.map(c=>`<td>${c}</td>`).join('')}</tr>`});document.getElementById('taskTable').innerHTML=tableMarkup(headers,body)}
function metricScoreClass(values,index){const rounded=values.map(v=>num(v)==null?null:Number(Number(v).toFixed(4)));const unique=[...new Set(rounded.filter(v=>v!=null))].sort((a,b)=>b-a);const value=rounded[index];if(value==null)return '';if(value===unique[0])return 'best score-best';if(unique.length>1&&value===unique[1])return 'score-second';return ''}
function renderDatasetMetrics(){const models=DATA.model_order||[];const short={'HUGIML':'HUGIML','XGB standard':'XGB','LightGBM standard':'LightGBM','RandomForest standard':'RandomForest'};const metrics=[['roc_auc','ROC-AUC'],['balanced_accuracy','Balanced accuracy'],['f1','F1 (binary / macro)']];const row1=`<tr><th rowspan="2" class="dataset-sticky">Dataset</th><th rowspan="2" class="task-sticky">Task</th>${metrics.map(([,label])=>`<th colspan="${models.length}" class="metric-group-head">${esc(label)}</th>`).join('')}</tr>`;const row2=`<tr>${metrics.flatMap(()=>models).map((model,i)=>`<th class="${i%models.length===0?'group-start':''}">${esc(short[model]||model)}</th>`).join('')}</tr>`;const body=(DATA.per_task||[]).map(task=>{const byModel=Object.fromEntries((task.rows||[]).map(row=>[row.model,row]));const cells=[];for(const [key] of metrics){const values=models.map(model=>byModel[model]?.[key]);values.forEach((value,i)=>{const classes=[metricScoreClass(values,i),i===0?'group-start':''].filter(Boolean).join(' ');cells.push(`<td class="${classes}" title="${num(value)==null?'':esc(String(value))}">${fmt(value,4)}</td>`)});}return `<tr><td class="dataset-sticky">${esc(task.dataset)}</td><td class="task-sticky">${task.task_id}</td>${cells.join('')}</tr>`}).join('');document.getElementById('datasetMetricTable').innerHTML=`<thead>${row1}${row2}</thead><tbody>${body||`<tr><td colspan="${2+metrics.length*models.length}" class="empty">No rows available.</td></tr>`}</tbody>`}
function renderPaired(){const rows=DATA.versus_hugiml||[];document.getElementById('paired').innerHTML=rows.map(r=>`<div class="paired-row"><strong>${esc(r.comparison)}</strong>${r.significant_holm_0_05?'<span class="badge">Significant</span>':''}<div class="note">Mean AUC difference: ${fmt(r.mean_auc_difference_hugiml_minus_baseline)} · raw p=${fmt(r.p_value,5)} · Holm p=${fmt(r.p_value_holm,5)} · paired tasks=${r.n_tasks??0}</div></div>`).join('')||'<div class="note">Insufficient complete paired tasks.</div>'}
function renderProtocol(){const m=DATA.metadata||{};const inner=m.tune?`${m.inner_splits??3}-fold stratified CV inside each official training partition`:'Disabled; estimator defaults fitted on each official training partition';const rows=[['Outer evaluation','Every OpenML-defined repeat/fold/sample split; no generated outer folds'],['Inner selection',inner],['Primary metric','Binary ROC AUC or multiclass OVR macro ROC AUC'],['Models',(DATA.model_order||[]).join(', ')],['Inspection comparison',`HUGIML AUC within ${fmt(DATA.complexity_auc_tolerance,3)} of or above the baseline`],['Task selection',m.selection_description||'Selected tasks']];document.getElementById('protocol').innerHTML=rows.map(([k,v])=>`<div><strong>${esc(k)}</strong></div><div>${esc(v)}</div>`).join('');document.getElementById('grids').textContent=JSON.stringify(m.grid_snapshot||{},null,2)}
const select=document.getElementById('taskSelect');select.innerHTML=(DATA.per_task||[]).map(t=>`<option value="${t.task_id}">${esc(t.dataset)} · task ${t.task_id}</option>`).join('');select.addEventListener('change',renderTask);document.querySelectorAll('.theme-btn').forEach(button=>button.addEventListener('click',()=>applyTheme(button.dataset.theme)));renderHero();renderMetrics();renderOverall();renderRatios();renderTask();renderDatasetMetrics();renderPaired();renderProtocol();renderComplexity();
</script></body></html>
"""


def write_template(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_TEMPLATE, encoding="utf-8")
    return path


def render_dashboard(data: dict[str, Any], out_html: Path, template_html: Path | None) -> None:
    template = template_html.read_text(encoding="utf-8") if template_html is not None else DEFAULT_TEMPLATE
    required = ["{{TITLE}}", "{{GENERATED_AT}}", "{{DATA_JSON}}"]
    missing = [token for token in required if token not in template]
    if missing:
        raise ValueError(f"Dashboard template is missing placeholders: {missing}")
    title = "HUGIML OpenML-CC18 Benchmark Dashboard"
    data_json = json.dumps(safe_jsonable(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    text = (
        template.replace("{{TITLE}}", html.escape(title))
        .replace("{{GENERATED_AT}}", html.escape(str(data.get("generated_at_utc", utc_now()))))
        .replace("{{DATA_JSON}}", data_json)
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(text, encoding="utf-8")


def assemble_outputs(
    checkpoint: Path,
    out_dir: Path,
    template_html: Path | None,
    run_id: str | None = None,
) -> dict[str, Path]:
    payload = load_checkpoint(checkpoint)
    view = run_view(payload, run_id)
    data = build_dashboard_data(payload, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_task_ids = {int(row["task_id"]) for row in view.get("selected_tasks", [])}
    selected_results = [
        row
        for row in view["results"]
        if not selected_task_ids or int(row.get("task_id", -1)) in selected_task_ids
    ]
    pair_rows = [{k: v for k, v in row.items() if k != "split_rows"} for row in selected_results]
    split_rows = [split for row in selected_results for split in row.get("split_rows", [])]
    pair_csv = out_dir / "openml_cc18_pair_results.csv"
    split_csv = out_dir / "openml_cc18_official_split_results.csv"
    summary_csv = out_dir / "openml_cc18_model_summary.csv"
    complexity_points_csv = out_dir / "openml_cc18_complexity_points.csv"
    complexity_ratios_csv = out_dir / "openml_cc18_complexity_ratios.csv"
    data_json = out_dir / "openml_cc18_dashboard_data.json"
    dashboard_html = out_dir / DEFAULT_DASHBOARD_NAME

    pd.DataFrame(pair_rows).to_csv(pair_csv, index=False)
    pd.DataFrame(split_rows).to_csv(split_csv, index=False)
    pd.DataFrame(data["overall"]).to_csv(summary_csv, index=False)
    pd.DataFrame(data.get("complexity_points", [])).to_csv(complexity_points_csv, index=False)
    pd.DataFrame(data.get("complexity_ratios", [])).to_csv(complexity_ratios_csv, index=False)
    atomic_write_json(data_json, data)
    render_dashboard(data, dashboard_html, template_html)
    return {
        "pair_results_csv": pair_csv,
        "official_split_results_csv": split_csv,
        "model_summary_csv": summary_csv,
        "complexity_points_csv": complexity_points_csv,
        "complexity_ratios_csv": complexity_ratios_csv,
        "dashboard_data_json": data_json,
        "dashboard_html": dashboard_html,
    }


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark HUGIML, ensemble models, and interpretable models on official OpenML-CC18 splits"
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
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--assemble", action="store_true", help="Build CSV/JSON/HTML from checkpoint")
    parser.add_argument("--template-html", type=Path, default=None, help="Custom dashboard template")
    parser.add_argument(
        "--write-template",
        nargs="?",
        const="openml_cc18_dashboard_template.html",
        default=None,
        help="Write the built-in HTML template and exit; optional output path",
    )
    return parser


def pair_error_row(
    *,
    task: dict[str, Any],
    model_name: str,
    tune: bool,
    inner_splits: int,
    hugiml_scenario: str,
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
        "evaluation_protocol": "openml_defined_outer_splits_with_inner_cv_tuning"
        if tune
        else "openml_defined_outer_splits_no_inner_tuning",
        "inner_n_splits": int(inner_splits) if tune else None,
        "tuned": bool(tune),
        "hugiml_scenario": hugiml_scenario if model_name == "HUGIML" else None,
        "roc_auc": None,
        "balanced_accuracy": None,
        "f1": None,
        "fit_seconds": None,
        "tune_seconds": None,
        "predict_seconds": None,
        "complexity": None,
        "complexity_model_units": None,
        "complexity_model_inspection_units": None,
        "complexity_instance_inspection_units_mean": None,
        "complexity_instance_inspection_units_ci_lower": None,
        "complexity_instance_inspection_units_ci_upper": None,
        "split_rows": [],
    }


def benchmark_main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        outputs = assemble_outputs(checkpoint, out_dir, template_html, args.run_id)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.smallest is not None and args.smallest <= 0:
        raise ValueError("--smallest must be greater than zero")
    if args.complexity_auc_tolerance < 0:
        raise ValueError("--complexity-auc-tolerance must be zero or greater")
    configure_openml(args.cache_dir.expanduser().resolve() if args.cache_dir else out_dir / "openml_cache")
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

    grid = grid_snapshot(args.hugiml_scenario, models)
    source = {
        "script_sha256": sha256_file(Path(__file__)),
        "benchmark_dashboard_sha256": sha256_file(Path(benchmark_base.__file__)),
        "package_version": getattr(benchmark_base._hugiml_pkg, "__version__", None),
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
        "evaluation_protocol": "OpenML task-defined train/test indices with optional inner stratified CV",
        "primary_metric": "ROC AUC; multiclass OVR macro",
        "hugiml_scenario": args.hugiml_scenario,
        "hugiml_max_fit_seconds": args.hugiml_max_fit_seconds,
        "complexity_auc_tolerance": float(args.complexity_auc_tolerance),
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
        "hugiml_scenario": args.hugiml_scenario,
        "hugiml_max_fit_seconds": args.hugiml_max_fit_seconds,
        "complexity_auc_tolerance": float(args.complexity_auc_tolerance),
        "grid_snapshot": grid,
        "source": source,
    }
    payload = load_checkpoint(checkpoint)
    run_id = activate_run(
        payload,
        configuration=configuration,
        metadata=metadata,
        selected_tasks=selected_tasks,
    )
    if not args.resume:
        payload["results"] = [
            row for row in payload["results"] if str(row.get("run_id")) != run_id
        ]
    atomic_write_json(checkpoint, payload)

    plan = [(int(task["task_id"]), model) for task in selected_tasks for model in models]
    if args.start_pair:
        plan = plan[int(args.start_pair) :]
    if args.max_pairs is not None:
        plan = plan[: int(args.max_pairs)]
    done = completed_pair_keys(payload, run_id) if args.resume else set()

    selected_by_id = {int(row["task_id"]): row for row in selected_tasks}
    for task_id, model_name in plan:
        key = pair_key(task_id, model_name)
        task_label = selected_by_id[task_id]["dataset_name"]
        if key in done:
            print(f"skip task {task_id} ({task_label}) :: {model_name}", flush=True)
            continue
        print(f"run task {task_id} ({task_label}) :: {model_name}", flush=True)
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
            )
        except Exception as exc:
            result = pair_error_row(
                task=selected_by_id[task_id],
                model_name=model_name,
                tune=not args.no_tune,
                inner_splits=args.inner_splits,
                hugiml_scenario=args.hugiml_scenario,
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
    for file_name, key in ((metadata["features_file"], "features.parquet"), (metadata["target_file"], "target.parquet")):
        expected = expected_hashes.get(key)
        if expected and sha256_file(folder / str(file_name)) != expected:
            raise ValueError(f"Offline dataset checksum mismatch for task {entry['task_id']}: {file_name}")
    split_frame = pd.read_csv(folder / str(metadata["split_manifest_file"]))
    if len(split_frame) != int(metadata["official_split_count"]):
        raise ValueError(f"Offline split count mismatch for task {entry['task_id']}")
    for split in split_frame.itertuples(index=False):
        for relative in (split.train_file, split.test_file):
            if not (folder / str(relative)).is_file():
                raise FileNotFoundError(f"Offline split file is missing for task {entry['task_id']}: {relative}")
    return folder


def select_offline_entries(
    entries: dict[int, dict[str, Any]],
    *,
    smallest: int | None,
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
            explicit_task_ids=explicit_task_ids,
        )
        suite_meta = {
            "suite_id": manifest.get("suite_id"),
            "suite_name": manifest.get("suite", "OpenML-CC18"),
            "suite_alias": manifest.get("suite", "OpenML-CC18"),
            "suite_task_count": len(entries),
            "data_source": "offline_dataset_cache",
            "datasets_dir": datasets_dir.name or "openml_cc18_datasets",
        }
        return suite_meta, selected

    def load_task(task_id: int, *, retries: int) -> tuple[Any, pd.DataFrame, np.ndarray, dict[str, Any]]:
        del retries
        entry = entries[int(task_id)]
        folder = validate_dataset_entry(datasets_dir, entry, verify_integrity=verify_cache)
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        features = pd.read_parquet(folder / metadata["features_file"])
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
        "--verify-cache",
        action="store_true",
        help="Validate cache metadata, checksums, and every official split file before execution",
    )
    offline_args, benchmark_argv = pre_parser.parse_known_args(raw_argv)
    datasets_dir = offline_args.datasets_dir.expanduser().resolve()

    parsed_benchmark_args, _ = benchmark.build_parser().parse_known_args(benchmark_argv)
    if parsed_benchmark_args.assemble or parsed_benchmark_args.write_template is not None:
        return benchmark_main(benchmark_argv)

    try:
        if parsed_benchmark_args.smallest is not None and parsed_benchmark_args.smallest <= 0:
            raise ValueError("--smallest must be greater than zero")
        manifest = read_manifest(datasets_dir)
        if offline_args.verify_cache:
            verify_dataset_manifest(datasets_dir, manifest)
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
            explicit_task_ids=task_ids,
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
        print(f"Cannot start offline benchmark.\n{exc}", file=sys.stderr)
        return 2

    forwarded = list(benchmark_argv)
    if task_ids:
        forwarded = strip_option(forwarded, "--task-ids")
        forwarded.extend(["--task-ids", ",".join(str(task_id) for task_id in task_ids)])
    selected_ids = [int(entry["task_id"]) for entry in selected]
    print(
        f"Offline benchmark: {len(selected_ids)} task(s) from {datasets_dir}\n"
        f"Task IDs: {','.join(str(task_id) for task_id in selected_ids)}",
        flush=True,
    )
    return benchmark_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())


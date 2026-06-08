# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0.

"""Benchmark runner for HUG-IML and baseline classifiers.

This runner extends the original v1.1.7 built-in benchmark workflow with:

* LightGBM baseline support when ``lightgbm`` is installed.
* Custom dataset support via ``--data`` and ``--target``.
* Optional hyperparameter tuning via ``--tune`` for both built-in and custom
  datasets.

Examples
--------
Built-in datasets, standard configs::

    hugiml-bench --datasets breast_cancer adult credit

Built-in datasets, inner-CV tuning::

    hugiml-bench --datasets breast_cancer --tune

Custom dataset::

    hugiml-bench --data data.csv --target target --output benchmarks/results/custom

Custom dataset with tuning::

    hugiml-bench --data data.csv --target target --tune --n-splits 5 --inner-splits 3
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

__all__ = ["run_benchmark", "run_custom_benchmark", "main"]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _load_breast_cancer():
    from sklearn.datasets import load_breast_cancer

    d = load_breast_cancer(as_frame=True)
    return d.data, d.target.values


def _load_adult():
    try:
        from sklearn.datasets import fetch_openml

        data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        X = data.data.copy().fillna("MISSING")
        y = (data.target.str.strip().str.rstrip(".") == ">50K").astype(int).values
        X = X.dropna(axis=1, thresh=int(0.7 * len(X)))
        cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        for c in cat_cols:
            X[c] = X[c].astype("category").cat.codes
        return X, y
    except Exception:
        np.random.seed(42)
        n = 3000
        X = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n),
                "edu_num": np.random.randint(1, 16, n),
                "capital_gain": np.random.exponential(500, n),
                "hours_per_week": np.random.randint(10, 80, n),
            }
        )
        y = ((X["age"] > 40) & (X["hours_per_week"] > 40)).astype(int).values
        return X, y


def _load_credit():
    try:
        from sklearn.datasets import fetch_openml

        data = fetch_openml(
            "default-of-credit-card-clients", version=1, as_frame=True, parser="auto"
        )
        return data.data.astype(float), data.target.astype(int).values
    except Exception:
        np.random.seed(7)
        n = 3000
        X = pd.DataFrame(
            {
                "limit_bal": np.random.exponential(100000, n),
                "age": np.random.randint(21, 75, n),
                "bill_amt1": np.random.normal(20000, 15000, n),
            }
        )
        y = (np.random.random(n) < 0.22).astype(int)
        return X, y


DATASET_LOADERS = {
    "breast_cancer": _load_breast_cancer,
    "adult": _load_adult,
    "credit": _load_credit,
}


# ---------------------------------------------------------------------------
# Generic dataset preparation
# ---------------------------------------------------------------------------


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset file extension: {suffix}")


def _split_csv(value: str | None) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def _encode_target(y_raw: pd.Series, positive_label: str | None = None) -> np.ndarray:
    y = y_raw.copy()
    if positive_label is not None:
        return (y.astype(str) == str(positive_label)).astype(int).to_numpy()

    unique = pd.Series(y.dropna().unique())
    if len(unique) != 2:
        raise ValueError(
            "Benchmark runner currently expects binary classification. "
            f"Found {len(unique)} target classes."
        )

    # Preserve numeric 0/1 when already present.
    try:
        y_num = pd.to_numeric(y, errors="raise")
        vals = sorted(pd.Series(y_num.dropna().unique()).tolist())
        if vals == [0, 1]:
            return y_num.astype(int).to_numpy()
    except Exception:
        pass

    classes = sorted([str(v) for v in unique.tolist()])
    positive = classes[-1]
    return (y.astype(str) == positive).astype(int).to_numpy()


def _prepare_features_for_benchmarks(X: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric dataframe suitable for all benchmark baselines.

    HUG-IML can handle richer feature metadata, but all baselines in this runner
    need numeric matrix input. We use deterministic category codes and simple
    median/zero fills. This mirrors the original v1.1.7 runner style and keeps
    the benchmark dependency-light.
    """
    Xp = X.copy()

    # Drop columns that are entirely missing.
    Xp = Xp.dropna(axis=1, how="all")

    cat_cols = Xp.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    for c in cat_cols:
        Xp[c] = Xp[c].astype("string").fillna("MISSING").astype("category").cat.codes

    for c in Xp.columns:
        if not pd.api.types.is_numeric_dtype(Xp[c]):
            Xp[c] = pd.to_numeric(Xp[c], errors="coerce")
        if Xp[c].isna().any():
            med = Xp[c].median()
            if pd.isna(med):
                med = 0
            Xp[c] = Xp[c].fillna(med)

    return Xp


def _load_custom_dataset(
    data_path: str,
    target: str,
    id_column: str | None = None,
    exclude_columns: str | None = None,
    positive_label: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    raw = _read_table(data_path)
    if target not in raw.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    drop_cols = [target]
    if id_column:
        if id_column not in raw.columns:
            raise ValueError(f"ID column '{id_column}' not found in dataset.")
        drop_cols.append(id_column)

    for c in _split_csv(exclude_columns):
        if c not in raw.columns:
            raise ValueError(f"Excluded column '{c}' not found in dataset.")
        drop_cols.append(c)

    y = _encode_target(raw[target], positive_label=positive_label)
    X = raw.drop(columns=drop_cols)
    X = _prepare_features_for_benchmarks(X)
    return X, y


# ---------------------------------------------------------------------------
# Model builders and tuning grids
# ---------------------------------------------------------------------------


def _build_hugiml(allCols=None, origColumns=None, random_state: int = 42):
    from hugiml import HUGIMLClassifierNative

    kw: dict[str, Any] = {}
    if allCols is not None:
        kw = {"allCols": allCols, "origColumns": origColumns}

    # Keep the original v1.1.7 baseline configuration.
    return HUGIMLClassifierNative(B=7, L=2, G=5e-3, topK=100, **kw)


def _build_ebm(random_state: int = 42):
    try:
        from interpret.glassbox import ExplainableBoostingClassifier

        return ExplainableBoostingClassifier(random_state=random_state)
    except ImportError:
        return None


def _build_xgb(random_state: int = 42):
    try:
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        )
    except ImportError:
        return None


def _build_lightgbm(random_state: int = 42):
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        return None


def _build_rf(random_state: int = 42):
    return RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=random_state, n_jobs=-1
    )


def _build_lr(random_state: int = 42):
    return Pipeline(
        [
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, random_state=random_state)),
        ]
    )


def _build_rulefit(random_state: int = 42):
    try:
        from rulefit import RuleFit

        return RuleFit(tree_size=4, max_rules=100, rfmode="classify", random_state=random_state)
    except ImportError:
        return None


def _build_pygam(random_state: int = 42):
    try:
        from pygam import LogisticGAM

        return LogisticGAM()
    except ImportError:
        return None


BUILDERS = {
    "HUG-IML": _build_hugiml,
    "EBM": _build_ebm,
    "XGBoost": _build_xgb,
    "LightGBM": _build_lightgbm,
    "RandomForest": _build_rf,
    "LogisticReg": _build_lr,
    "RuleFit": _build_rulefit,
    "GAM": _build_pygam,
}


TUNING_GRIDS: dict[str, dict[str, list[Any]]] = {
    "RandomForest": {
        "n_estimators": [200, 400],
        "max_depth": [4, 8, None],
        "min_samples_leaf": [1, 5],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 4],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [100, 200],
        "learning_rate": [0.03, 0.1],
        "num_leaves": [15, 31],
        "subsample": [0.8, 1.0],
    },
    "LogisticReg": {
        "lr__C": [0.1, 1.0, 10.0],
    },
    "EBM": {
        "learning_rate": [0.01, 0.05],
        "max_bins": [128, 256],
    },
    # RuleFit and GAM APIs vary across installations; keep static by default.
}


def _build_hugiml_for_fold(X_tr: pd.DataFrame, random_state: int = 42):
    if isinstance(X_tr, pd.DataFrame):
        cat_cols = X_tr.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        int_cols = X_tr.select_dtypes(include=["int", "int32", "int64"]).columns.tolist()
        flt_cols = [c for c in X_tr.columns if c not in cat_cols and c not in int_cols]
        return _build_hugiml([int_cols, flt_cols, cat_cols], X_tr.columns.tolist(), random_state)
    return _build_hugiml(random_state=random_state)


def _tune_hugiml(
    X_tr,
    y_tr,
    inner_splits: int,
    random_state: int,
):
    from hugiml import HUGIMLClassifierNative

    inner_splits = _validated_stratified_splits(y_tr, inner_splits, label="Inner HUG-IML tuning CV")
    if hasattr(HUGIMLClassifierNative, "tune"):
        result = HUGIMLClassifierNative.tune(
            X_tr,
            y_tr,
            cv=inner_splits,
            shuffle=True,
            random_state=random_state,
            scoring="roc_auc",
            param_grid=None,
            refit=True,
            use_fast_path=True,
        )
        return result.best_estimator_, getattr(result, "best_params_", {}), getattr(
            result, "best_score_", float("nan")
        )

    clf = _build_hugiml_for_fold(X_tr, random_state=random_state)
    clf.fit(X_tr, y_tr)
    return clf, {"static": True}, float("nan")


def _tune_sklearn_like(
    model_name: str,
    clf,
    X_tr,
    y_tr,
    inner_splits: int,
    random_state: int,
):
    grid = TUNING_GRIDS.get(model_name)
    if not grid:
        clf.fit(X_tr, y_tr)
        return clf, {"static": True}, float("nan")

    inner_splits = _validated_stratified_splits(y_tr, inner_splits, label="Inner tuning CV")
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        clf,
        grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score=np.nan,
    )
    search.fit(X_tr, y_tr)
    return search.best_estimator_, search.best_params_, float(search.best_score_)




def _validated_stratified_splits(y: np.ndarray, requested: int, *, label: str) -> int:
    """Return a safe StratifiedKFold split count for binary benchmark data."""
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    if classes.size < 2:
        raise ValueError(f"{label} requires at least two target classes.")
    min_count = int(np.min(counts))
    if min_count < 2:
        raise ValueError(
            f"{label} needs at least two samples in every class; smallest class has {min_count}."
        )
    return max(2, min(int(requested), min_count))


def _select_models(model_names: list[str] | None = None) -> dict[str, Any]:
    if not model_names:
        return BUILDERS
    unknown = [m for m in model_names if m not in BUILDERS]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {sorted(BUILDERS)}")
    return {m: BUILDERS[m] for m in model_names}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate(clf, X_tr, y_tr, X_te, y_te, tune_ms: float = 0.0, tune_info: dict | None = None) -> dict:
    t0 = time.perf_counter()
    try:
        # Tuned estimators are already fitted. We mark them with tune_info.
        if not tune_info:
            clf.fit(X_tr, y_tr)
    except Exception as e:
        return {"error": str(e)}
    fit_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    try:
        proba = clf.predict_proba(X_te)[:, 1]
    except Exception:
        try:
            scores = clf.decision_function(X_te)
            proba = 1.0 / (1.0 + np.exp(-scores))
        except Exception:
            proba = np.full(len(y_te), 0.5)
    predict_ms = (time.perf_counter() - t1) * 1000

    preds = (proba >= 0.5).astype(int)

    def _s(fn, *a, **k):
        try:
            return float(fn(*a, **k))
        except Exception:
            return float("nan")

    out = {
        "accuracy": _s(accuracy_score, y_te, preds),
        "balanced_accuracy": _s(balanced_accuracy_score, y_te, preds),
        "roc_auc": _s(roc_auc_score, y_te, proba),
        "avg_precision": _s(average_precision_score, y_te, proba),
        "brier": _s(brier_score_loss, y_te, proba),
        "f1": _s(f1_score, y_te, preds, zero_division=0),
        "fit_ms": fit_ms,
        "predict_ms": predict_ms,
        "tune_ms": tune_ms,
    }
    if tune_info:
        out["tuned"] = True
        out["best_inner_score"] = tune_info.get("best_inner_score", float("nan"))
        out["best_params"] = json.dumps(tune_info.get("best_params", {}), default=str)
    else:
        out["tuned"] = False
        out["best_inner_score"] = float("nan")
        out["best_params"] = "{}"
    return out


def _run_cv_benchmark(
    X: pd.DataFrame,
    y: np.ndarray,
    dataset_name: str,
    n_splits: int,
    output_dir: str | None,
    tune: bool,
    inner_splits: int,
    random_state: int,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    print(f"\n{'=' * 60}\nDataset: {dataset_name}\n{'=' * 60}")
    print(f"  Shape: {X.shape}  |  class balance: {np.mean(y):.3f}")
    print(f"  Tuning: {'on' if tune else 'off'}")

    n_splits = _validated_stratified_splits(y, n_splits, label="Outer benchmark CV")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records = []

    builders = _select_models(model_names)

    for model_name, builder in builders.items():
        clf_proto = builder(random_state=random_state)
        if clf_proto is None:
            print(f"  [{model_name}] SKIPPED (package not installed)")
            continue

        print(f"  [{model_name}] ...", end="", flush=True)
        fold_aucs = []

        for fold_idx, (tr, te) in enumerate(cv.split(X, y)):
            X_tr = X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr]
            X_te = X.iloc[te] if isinstance(X, pd.DataFrame) else X[te]
            y_tr, y_te = y[tr], y[te]

            tune_info = None
            tune_ms = 0.0

            try:
                if tune:
                    t0 = time.perf_counter()
                    if model_name == "HUG-IML":
                        clf, best_params, best_inner_score = _tune_hugiml(
                            X_tr, y_tr, inner_splits=inner_splits, random_state=random_state
                        )
                    else:
                        clf0 = builder(random_state=random_state)
                        clf, best_params, best_inner_score = _tune_sklearn_like(
                            model_name,
                            clf0,
                            X_tr,
                            y_tr,
                            inner_splits=inner_splits,
                            random_state=random_state,
                        )
                    tune_ms = (time.perf_counter() - t0) * 1000
                    tune_info = {
                        "best_params": best_params,
                        "best_inner_score": best_inner_score,
                    }
                elif model_name == "HUG-IML":
                    clf = _build_hugiml_for_fold(X_tr, random_state=random_state)
                else:
                    clf = copy.deepcopy(clf_proto)

                m = _evaluate(
                    clf,
                    X_tr,
                    y_tr,
                    X_te,
                    y_te,
                    tune_ms=tune_ms,
                    tune_info=tune_info,
                )
            except Exception as e:
                m = {"error": str(e)}

            m.update({"fold": fold_idx, "model": model_name, "dataset": dataset_name})
            records.append(m)
            fold_aucs.append(m.get("roc_auc", float("nan")))

        mean_auc = np.nanmean(fold_aucs)
        print(f" ROC-AUC = {mean_auc:.4f}")

    df = pd.DataFrame(records)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        safe_name = str(dataset_name).replace("/", "_").replace("\\", "_")
        df.to_csv(f"{output_dir}/{safe_name}_results.csv", index=False)

        metric_cols = [
            c
            for c in ["accuracy", "balanced_accuracy", "roc_auc", "avg_precision", "f1", "brier", "fit_ms", "predict_ms", "tune_ms"]
            if c in df.columns
        ]
        summary = df.groupby("model")[metric_cols].agg(["mean", "std"]).round(4)
        summary.to_json(f"{output_dir}/{safe_name}_summary.json", indent=2)
        print(f"  → Saved results to {output_dir}/")
    return df


def run_benchmark(
    dataset_name: str,
    n_splits: int = 5,
    output_dir: str | None = None,
    tune: bool = False,
    inner_splits: int = 3,
    random_state: int = 42,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Run CV benchmark for one built-in dataset and return per-fold results."""
    X, y = DATASET_LOADERS[dataset_name]()
    X = _prepare_features_for_benchmarks(X)
    return _run_cv_benchmark(
        X,
        y,
        dataset_name=dataset_name,
        n_splits=n_splits,
        output_dir=output_dir,
        tune=tune,
        inner_splits=inner_splits,
        random_state=random_state,
        model_names=models,
    )


def run_custom_benchmark(
    data: str,
    target: str,
    dataset_name: str | None = None,
    id_column: str | None = None,
    exclude_columns: str | None = None,
    positive_label: str | None = None,
    n_splits: int = 5,
    output_dir: str | None = None,
    tune: bool = False,
    inner_splits: int = 3,
    random_state: int = 42,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Run benchmark for a user-supplied binary classification dataset."""
    X, y = _load_custom_dataset(
        data,
        target=target,
        id_column=id_column,
        exclude_columns=exclude_columns,
        positive_label=positive_label,
    )
    name = dataset_name or Path(data).stem
    return _run_cv_benchmark(
        X,
        y,
        dataset_name=name,
        n_splits=n_splits,
        output_dir=output_dir,
        tune=tune,
        inner_splits=inner_splits,
        random_state=random_state,
        model_names=models,
    )


def main():
    parser = argparse.ArgumentParser(description="HUG-IML benchmark runner")
    parser.add_argument("--datasets", nargs="+", default=None, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--data", default=None, help="Path to a custom CSV/TSV/Excel/Parquet dataset.")
    parser.add_argument("--target", default=None, help="Target column for --data.")
    parser.add_argument("--dataset-name", default=None, help="Display/output name for custom dataset.")
    parser.add_argument("--id-column", default=None, help="Optional ID column to exclude from modeling.")
    parser.add_argument("--exclude-columns", default="", help="Comma-separated columns to exclude.")
    parser.add_argument("--positive-label", default=None, help="Positive class label for custom dataset.")
    parser.add_argument("--models", nargs="+", default=None, choices=list(BUILDERS.keys()))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tune", action="store_true", help="Enable inner-CV hyperparameter tuning.")
    parser.add_argument("--output", default="benchmarks/results/")
    args = parser.parse_args()

    if args.data and not args.target:
        parser.error("--target is required when --data is supplied.")

    all_dfs = []

    if args.data:
        try:
            all_dfs.append(
                run_custom_benchmark(
                    data=args.data,
                    target=args.target,
                    dataset_name=args.dataset_name,
                    id_column=args.id_column,
                    exclude_columns=args.exclude_columns,
                    positive_label=args.positive_label,
                    n_splits=args.n_splits,
                    output_dir=args.output,
                    tune=args.tune,
                    inner_splits=args.inner_splits,
                    random_state=args.random_state,
                    models=args.models,
                )
            )
        except Exception as e:
            print(f"  ERROR on custom dataset '{args.data}': {e}")
    else:
        datasets = args.datasets or list(DATASET_LOADERS.keys())
        for ds in datasets:
            try:
                all_dfs.append(
                    run_benchmark(
                        ds,
                        n_splits=args.n_splits,
                        output_dir=args.output,
                        tune=args.tune,
                        inner_splits=args.inner_splits,
                        random_state=args.random_state,
                        models=args.models,
                    )
                )
            except Exception as e:
                print(f"  ERROR on '{ds}': {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        combined.to_csv(f"{args.output}/full_report.csv", index=False)
        if "roc_auc" in combined.columns:
            print("\n=== FINAL SUMMARY ===")
            print(combined.groupby("model")["roc_auc"].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()

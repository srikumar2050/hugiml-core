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

"""
Benchmark runner: HUG-IML vs EBM, XGBoost, Random Forest,
Logistic Regression, RuleFit, and GAM.

Usage
-----
    python -m hugiml.benchmarks.runner
    python -m hugiml.benchmarks.runner --datasets breast_cancer adult
    python -m hugiml.benchmarks.runner --output benchmarks/results/
"""

from __future__ import annotations

import argparse
import copy
import time
import warnings
from pathlib import Path

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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


__all__ = [
    "run_benchmark",
    "main",
]


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
        cat_cols = X.select_dtypes(include="object").columns.tolist()
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
# Model builders
# ---------------------------------------------------------------------------


def _build_hugiml(allCols=None, origColumns=None):
    from hugiml import HUGIMLClassifierNative

    kw = {}
    if allCols is not None:
        kw = {"allCols": allCols, "origColumns": origColumns}
    return HUGIMLClassifierNative(B=7, L=2, G=5e-3, topK=100, **kw)


def _build_ebm():
    try:
        from interpret.glassbox import ExplainableBoostingClassifier

        return ExplainableBoostingClassifier(random_state=42)
    except ImportError:
        return None


def _build_xgb():
    try:
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
    except ImportError:
        return None


def _build_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)


def _build_lr():
    return Pipeline(
        [("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=500, random_state=42))]
    )


def _build_rulefit():
    try:
        from rulefit import RuleFit

        return RuleFit(tree_size=4, max_rules=100, rfmode="classify", random_state=42)
    except ImportError:
        return None


def _build_pygam():
    try:
        from pygam import LogisticGAM

        return LogisticGAM()
    except ImportError:
        return None


BUILDERS = {
    "HUG-IML": _build_hugiml,
    "EBM": _build_ebm,
    "XGBoost": _build_xgb,
    "RandomForest": _build_rf,
    "LogisticReg": _build_lr,
    "RuleFit": _build_rulefit,
    "GAM": _build_pygam,
}


# ---------------------------------------------------------------------------
# Single-fold evaluation
# ---------------------------------------------------------------------------


def _evaluate(clf, X_tr, y_tr, X_te, y_te) -> dict:
    t0 = time.perf_counter()
    try:
        clf.fit(X_tr, y_tr)
    except Exception as e:
        return {"error": str(e)}
    fit_ms = (time.perf_counter() - t0) * 1000

    try:
        proba = clf.predict_proba(X_te)[:, 1]
    except Exception:
        proba = np.full(len(y_te), 0.5)
    preds = (proba >= 0.5).astype(int)

    def _s(fn, *a, **k):
        try:
            return float(fn(*a, **k))
        except Exception:
            return float("nan")

    return {
        "accuracy": _s(accuracy_score, y_te, preds),
        "balanced_accuracy": _s(balanced_accuracy_score, y_te, preds),
        "roc_auc": _s(roc_auc_score, y_te, proba),
        "avg_precision": _s(average_precision_score, y_te, proba),
        "brier": _s(brier_score_loss, y_te, proba),
        "f1": _s(f1_score, y_te, preds, zero_division=0),
        "fit_ms": fit_ms,
    }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark(
    dataset_name: str, n_splits: int = 5, output_dir: str | None = None
) -> pd.DataFrame:
    """Run CV benchmark for one dataset and return per-fold results."""
    print(f"\n{'=' * 60}\nDataset: {dataset_name}\n{'=' * 60}")

    X, y = DATASET_LOADERS[dataset_name]()
    print(f"  Shape: {X.shape}  |  class balance: {y.mean():.3f}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    records = []

    for model_name, builder in BUILDERS.items():
        # Build a fresh instance for each model (some builders return None)
        clf_proto = builder()
        if clf_proto is None:
            print(f"  [{model_name}] SKIPPED (package not installed)")
            continue

        print(f"  [{model_name}] ...", end="", flush=True)
        fold_aucs = []

        for fold_idx, (tr, te) in enumerate(cv.split(X, y)):
            X_tr = X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr]
            X_te = X.iloc[te] if isinstance(X, pd.DataFrame) else X[te]
            y_tr, y_te = y[tr], y[te]

            # HUG-IML uses Path B for CV correctness
            if model_name == "HUG-IML":
                if isinstance(X_tr, pd.DataFrame):
                    cat_cols = X_tr.select_dtypes(include="object").columns.tolist()
                    int_cols = X_tr.select_dtypes(include="int").columns.tolist()
                    flt_cols = [c for c in X_tr.columns if c not in cat_cols and c not in int_cols]
                    clf = _build_hugiml([int_cols, flt_cols, cat_cols], X_tr.columns.tolist())
                else:
                    clf = _build_hugiml()
            else:
                clf = copy.deepcopy(clf_proto)

            m = _evaluate(clf, X_tr, y_tr, X_te, y_te)
            m.update({"fold": fold_idx, "model": model_name, "dataset": dataset_name})
            records.append(m)
            fold_aucs.append(m.get("roc_auc", float("nan")))

        mean_auc = np.nanmean(fold_aucs)
        print(f" ROC-AUC = {mean_auc:.4f}")

    df = pd.DataFrame(records)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(f"{output_dir}/{dataset_name}_results.csv", index=False)
        summary = (
            df.groupby("model")[
                ["accuracy", "balanced_accuracy", "roc_auc", "f1", "brier", "fit_ms"]
            ]
            .agg(["mean", "std"])
            .round(4)
        )
        summary.to_json(f"{output_dir}/{dataset_name}_summary.json", indent=2)
        print(f"  → Saved results to {output_dir}/")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_LOADERS.keys()))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output", default="benchmarks/results/")
    args = parser.parse_args()

    all_dfs = []
    for ds in args.datasets:
        try:
            all_dfs.append(run_benchmark(ds, args.n_splits, args.output))
        except Exception as e:
            print(f"  ERROR on '{ds}': {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        combined.to_csv(f"{args.output}/full_report.csv", index=False)
        print("\n=== FINAL SUMMARY ===")
        print(combined.groupby("model")["roc_auc"].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()

# %% [markdown]
# # HUG-IML vs Baselines — Benchmark Notebook
#
# Compares HUG-IML against EBM, XGBoost, Random Forest, Logistic Regression,
# RuleFit, and GAM on three datasets: Breast Cancer, Adult (income), and
# Default-of-Credit-Card (UCI).
#
# **Evaluation-safe usage note**
# `prepareXy` performs schema/type preparation only; discretization, HUG pattern
# mining, and downstream classifier fitting occur inside `fit()` on the training
# data supplied to that call.  All splits below are created before any `fit()` call.

# %% [markdown]
# ## Setup

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, time
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

# hugiml-core
from hugiml import HUGIMLClassifierNative

# Extension layer
import sys; sys.path.insert(0, str(Path("../").resolve()))
from hugiml.metrics import compute_all_metrics
from hugiml.plots import HUGPlotter

# %% [markdown]
# ## 1. Load datasets

# %%
from hugiml.benchmarks.runner import (
    _load_breast_cancer, _load_adult, _load_credit,
)

datasets = {
    "Breast Cancer": _load_breast_cancer(),
    "Adult":         _load_adult(),
    "Credit":        _load_credit(),
}
for name, (X, y) in datasets.items():
    print(f"{name}: {X.shape}, pos_rate={y.mean():.3f}")

# %% [markdown]
# ## 2. Define model grid

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_models(X_tr):
    models = {}
    # HUG-IML (Path B — CV-safe)
    if isinstance(X_tr, pd.DataFrame):
        cat_cols = X_tr.select_dtypes(include="object").columns.tolist()
        int_cols = X_tr.select_dtypes(include="int").columns.tolist()
        flt_cols = [c for c in X_tr.columns if c not in cat_cols + int_cols]
        models["HUG-IML"] = HUGIMLClassifierNative(
            allCols=[int_cols, flt_cols, cat_cols],
            origColumns=X_tr.columns.tolist(),
            B=7, L=2, G=5e-3,
        )
    else:
        models["HUG-IML"] = HUGIMLClassifierNative(B=7, L=2, G=5e-3)

    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        models["EBM"] = ExplainableBoostingClassifier(random_state=42)
    except ImportError:
        print("install interpret-learn for EBM: pip install interpret")

    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, eval_metric="logloss", verbosity=0)
    except ImportError:
        print("install xgboost for XGBoost: pip install xgboost")

    models["RandomForest"] = RandomForestClassifier(n_estimators=200, random_state=42)
    models["LogisticReg"] = Pipeline([("sc", StandardScaler()),
                                       ("lr", LogisticRegression(max_iter=500))])
    return models

# %% [markdown]
# ## 3. Cross-validated benchmark

# %%
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_rows = []

for ds_name, (X, y) in datasets.items():
    print(f"\n── {ds_name} ──")
    for fold, (tr, te) in enumerate(CV.split(X, y)):
        X_tr = X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr]
        X_te = X.iloc[te] if isinstance(X, pd.DataFrame) else X[te]
        y_tr, y_te = y[tr], y[te]

        for mname, clf in get_models(X_tr).items():
            t0 = time.perf_counter()
            try:
                clf.fit(X_tr, y_tr)
                proba = clf.predict_proba(X_te)[:, 1]
                auc  = roc_auc_score(y_te, proba)
                bal  = balanced_accuracy_score(y_te, proba >= 0.5)
                f1   = f1_score(y_te, proba >= 0.5, zero_division=0)
            except Exception as e:
                auc = bal = f1 = float("nan")
            fit_s = time.perf_counter() - t0

            row = {"dataset": ds_name, "model": mname, "fold": fold,
                   "roc_auc": auc, "balanced_acc": bal, "f1": f1, "fit_s": fit_s}
            all_rows.append(row)
            print(f"  {mname:15s} fold={fold} AUC={auc:.4f}")

results = pd.DataFrame(all_rows)

# %% [markdown]
# ## 4. Summary table

# %%
summary = (results.groupby(["dataset","model"])
           [["roc_auc","balanced_acc","f1","fit_s"]]
           .agg(["mean","std"])
           .round(4))
print(summary.to_string())

# %% [markdown]
# ## 5. Interpretability metrics for HUG-IML

# %%
for ds_name, (X, y) in list(datasets.items())[:1]:  # show for first dataset
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                stratify=y, random_state=42)
    clf = HUGIMLClassifierNative(B=7, L=2, G=5e-3)
    if isinstance(X_tr, pd.DataFrame):
        cat_cols = X_tr.select_dtypes(include="object").columns.tolist()
        int_cols = X_tr.select_dtypes(include="int").columns.tolist()
        flt_cols = [c for c in X_tr.columns if c not in cat_cols + int_cols]
        clf = HUGIMLClassifierNative(
            allCols=[int_cols, flt_cols, cat_cols],
            origColumns=X_tr.columns.tolist(), B=7, L=2, G=5e-3)
    clf.fit(X_tr, y_tr)
    m = compute_all_metrics(clf, X_te)
    print(f"\n{ds_name} — Interpretability metrics:")
    print(m)

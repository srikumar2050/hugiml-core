#!/usr/bin/env python
# coding: utf-8

# ## HUG-IML Benchmark Against Common Baselines
# ### A reproducible, evaluation-safe benchmark with bounded cross-validation and interpretable diagnostics.

# The comparison uses local datasets and synthetic tabular problems so the workflow is reproducible without network access. Cross-validation is stratified, and each dataset is capped before CV when needed. HUG-IML preprocessing and pattern mining are performed inside the estimator fit on each training fold.

# In[1]:


import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from hugiml import HUGIMLClassifierNative

RANDOM_STATE = 42
MAX_ROWS_FOR_CV = 500
N_SPLITS = 2


# ## 1. Dataset builders
# 
# The benchmark includes a medical dataset and two controlled synthetic risk datasets. Synthetic datasets are useful here because they allow nonlinear interactions and class imbalance to be created deliberately.

# In[2]:


def make_healthcare_dataset():
    X, y = load_breast_cancer(as_frame=True, return_X_y=True)
    return X, pd.Series(y, name="benign")


def make_credit_risk_dataset(n=650):
    X, y = make_classification(
        n_samples=n,
        n_features=18,
        n_informative=8,
        n_redundant=4,
        weights=[0.68, 0.32],
        class_sep=0.9,
        flip_y=0.025,
        random_state=RANDOM_STATE,
    )
    cols = [f"risk_signal_{i:02d}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["utilization_band"] = pd.qcut(df["risk_signal_00"], 4, labels=["low", "medium", "high", "very_high"]).astype(str)
    df["channel"] = np.where(df["risk_signal_01"] > 0, "digital", "branch")
    return df, pd.Series(y, name="default_flag")


def make_operational_risk_dataset(n=650):
    X, y = make_classification(
        n_samples=n,
        n_features=14,
        n_informative=6,
        n_redundant=2,
        weights=[0.82, 0.18],
        class_sep=0.75,
        flip_y=0.035,
        random_state=RANDOM_STATE + 7,
    )
    df = pd.DataFrame(X, columns=[f"control_metric_{i:02d}" for i in range(X.shape[1])])
    df["review_queue"] = pd.qcut(df["control_metric_02"], 3, labels=["standard", "watch", "urgent"]).astype(str)
    return df, pd.Series(y, name="case_escalated")


datasets = {
    "Healthcare diagnostic": make_healthcare_dataset(),
    "Credit risk": make_credit_risk_dataset(),
    "Operational risk": make_operational_risk_dataset(),
}

for name, (X, y) in datasets.items():
    print(f"{name:<24} rows={len(X):>5,}  features={X.shape[1]:>2}  positive_rate={y.mean():.3f}")


# ## 2. Modeling utilities
# 
# HUG-IML receives explicit integer, float, and categorical column groups when a pandas DataFrame is used. Baseline models receive one-hot encoded features through pandas utilities inside the fold.

# In[3]:


def cap_for_cv(X, y, max_rows=MAX_ROWS_FOR_CV):
    if len(X) <= max_rows:
        return X.reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    _, X_s, _, y_s = next(StratifiedKFold(n_splits=int(np.ceil(len(X) / max_rows)), shuffle=True, random_state=RANDOM_STATE).split(X, y)), None, None, None


def stratified_cap(X, y, max_rows=MAX_ROWS_FOR_CV):
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if len(X) <= max_rows:
        return X, y
    from sklearn.model_selection import train_test_split
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=RANDOM_STATE)
    return X_sample.reset_index(drop=True), y_sample.reset_index(drop=True)


def hug_for_frame(X_train):
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    int_cols = X_train.select_dtypes(include=["int", "int32", "int64"]).columns.tolist()
    flt_cols = [c for c in X_train.columns if c not in cat_cols + int_cols]
    return HUGIMLClassifierNative(
        allCols=[int_cols, flt_cols, cat_cols],
        origColumns=X_train.columns.tolist(),
        B=-1,
        adaptive_binning=True,
        L=2,
        G=0.01,
        feature_mode="original_plus_patterns",
        topK=50,
    )

def encode_for_baseline(X_train, X_test):
    X_all = pd.concat([X_train, X_test], axis=0)
    X_all = pd.get_dummies(X_all, drop_first=False)
    return X_all.iloc[:len(X_train)].copy(), X_all.iloc[len(X_train):].copy()


def baseline_models():
    models = {
        "Logistic regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=800, class_weight="balanced")),
        ]),
        "Random forest": RandomForestClassifier(n_estimators=60, min_samples_leaf=4, random_state=RANDOM_STATE, n_jobs=1),
    }
    return models


# ## 3. Cross-validated benchmark

# In[4]:


rows = []
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for dataset_name, (X_raw, y_raw) in datasets.items():
    X, y = stratified_cap(X_raw, y_raw)
    print(f"\n{dataset_name} — CV rows used: {len(X):,}")
    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
        y_tr, y_te = y.iloc[tr].to_numpy(), y.iloc[te].to_numpy()

        model_specs = {"HUG-IML": hug_for_frame(X_tr)}
        model_specs.update(baseline_models())

        X_tr_enc, X_te_enc = encode_for_baseline(X_tr, X_te)
        for model_name, model in model_specs.items():
            t0 = time.perf_counter()
            try:
                if model_name == "HUG-IML":
                    model.fit(X_tr, y_tr)
                    score = model.predict_proba(X_te)[:, 1]
                else:
                    model.fit(X_tr_enc, y_tr)
                    score = model.predict_proba(X_te_enc)[:, 1]
                pred = (score >= 0.5).astype(int)
                row = {
                    "dataset": dataset_name,
                    "fold": fold,
                    "model": model_name,
                    "roc_auc": roc_auc_score(y_te, score),
                    "balanced_accuracy": balanced_accuracy_score(y_te, pred),
                    "f1": f1_score(y_te, pred, zero_division=0),
                    "fit_seconds": time.perf_counter() - t0,
                    "status": "ok",
                }
            except Exception as exc:
                row = {
                    "dataset": dataset_name,
                    "fold": fold,
                    "model": model_name,
                    "roc_auc": np.nan,
                    "balanced_accuracy": np.nan,
                    "f1": np.nan,
                    "fit_seconds": time.perf_counter() - t0,
                    "status": type(exc).__name__,
                }
            rows.append(row)
            print(f"  fold={fold} {model_name:<20} AUC={row['roc_auc'] if pd.notna(row['roc_auc']) else np.nan:.3f} status={row['status']}")

results = pd.DataFrame(rows)
results.head(6)


# ## 4. Summary table

# In[5]:


summary = (
    results.groupby(["dataset", "model"], as_index=False)
    .agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        f1_mean=("f1", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"),
    )
    .sort_values(["dataset", "roc_auc_mean"], ascending=[True, False])
)
summary.round(4)


# In[6]:


fig, ax = plt.subplots(figsize=(10, 5.2))
plot_df = summary.pivot(index="model", columns="dataset", values="roc_auc_mean")
plot_df.plot(kind="bar", ax=ax)
ax.set_title("Mean ROC-AUC by model and dataset")
ax.set_ylabel("Mean ROC-AUC")
ax.set_xlabel("")
ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# ## 5. HUG-IML interpretability diagnostic
# 
# The diagnostic below fits one HUG-IML model on the healthcare dataset and reports its most influential patterns.

# In[7]:


X_demo, y_demo = datasets["Healthcare diagnostic"]
X_demo, y_demo = stratified_cap(X_demo, y_demo, max_rows=600)
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X_demo, y_demo, test_size=0.25, stratify=y_demo, random_state=RANDOM_STATE)

hug = hug_for_frame(X_tr)
hug.fit(X_tr, y_tr)
print(hug.model_summary())

pattern_table = hug.get_pattern_info().merge(
    hug.feature_importances()[["pattern", "coefficient", "abs_coefficient"]],
    on="pattern",
    how="left",
).sort_values("abs_coefficient", ascending=False)
pattern_table.head(12)


# In[8]:


fig, ax = plt.subplots(figsize=(9.5, 5))
plot_df = pattern_table.head(12).iloc[::-1]
ax.barh(plot_df["pattern"], plot_df["coefficient"])
ax.axvline(0, linewidth=1)
ax.set_title("HUG-IML top pattern contributions on healthcare diagnostic data")
ax.set_xlabel("Coefficient")
ax.set_ylabel("")
plt.tight_layout()
plt.show()


# The benchmark separates predictive performance from interpretability review. HUG-IML can be evaluated with the same fold discipline as other classifiers while also exposing compact, auditable pattern tables after fitting.

# In[ ]:





# ## 6. Hyperparameter tuning comparison
# 
# This section runs a focused hyperparameter analysis on one dataset using the same fold discipline for XGBoost, LightGBM, and HUG-IML. The tables report the best parameter setting for each model and compare cross-validated performance, fit time, prediction time, and total tuning time.

# In[9]:


from itertools import product
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

TUNING_DATASET = "Healthcare diagnostic"
TUNING_SPLITS = 5

X_tune_raw, y_tune_raw = datasets[TUNING_DATASET]
X_tune, y_tune = stratified_cap(X_tune_raw, y_tune_raw, max_rows=500)

print(f"Dataset: {TUNING_DATASET}")
print(f"Rows used: {len(X_tune):,}")
print(f"Features: {X_tune.shape[1]}")
print(f"Positive rate: {pd.Series(y_tune).mean():.3f}")


# In[10]:


xgb_grid = {
    "n_estimators": [40, 80],
    "max_depth": [2, 3],
    "learning_rate": [0.05],
    "subsample": [0.9],
    "colsample_bytree": [0.9],
}

lightgbm_grid = {
    "n_estimators": [40, 80],
    "num_leaves": [15, 31],
    "learning_rate": [0.05],
    "subsample": [0.9],
    "colsample_bytree": [0.9],
}

hugiml_grid = {
    "B": [-1],
    "adaptive_binning": [True],
    "L": [1, 2],
    "feature_mode": ["patterns_only", "original_plus_patterns"],
    "topK": [30, 50, 100],
    "G": [0.01],
}

grid_inventory = pd.DataFrame([
    {"model": "XGB", "grid_size": len(list(ParameterGrid(xgb_grid))), "grid": xgb_grid},
    {"model": "LightGBM", "grid_size": len(list(ParameterGrid(lightgbm_grid))), "grid": lightgbm_grid},
    {"model": "HUG-IML", "grid_size": len(list(ParameterGrid(hugiml_grid))), "grid": hugiml_grid},
])
grid_inventory


# In[11]:


def make_tuned_model(model_name, params, X_train):
    if model_name == "XGB":
        return XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        )
    if model_name == "LightGBM":
        return LGBMClassifier(
            **params,
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        )
    if model_name == "HUG-IML":
        cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        int_cols = X_train.select_dtypes(include=["int", "int32", "int64"]).columns.tolist()
        flt_cols = [c for c in X_train.columns if c not in cat_cols + int_cols]
        return HUGIMLClassifierNative(
            allCols=[int_cols, flt_cols, cat_cols],
            origColumns=X_train.columns.tolist(),
            n_jobs=1,
            **params,
        )
    raise ValueError(model_name)


def evaluate_parameter_grid(model_name, grid, X, y):
    cv = StratifiedKFold(n_splits=TUNING_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    tuning_start = time.perf_counter()

    for param_id, params in enumerate(ParameterGrid(grid), start=1):
        for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
            X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
            y_tr, y_te = y.iloc[tr].to_numpy(), y.iloc[te].to_numpy()

            if model_name in {"XGB", "LightGBM"}:
                X_tr_model, X_te_model = encode_for_baseline(X_tr, X_te)
            else:
                X_tr_model, X_te_model = X_tr, X_te

            model = make_tuned_model(model_name, params, X_tr)

            fit_start = time.perf_counter()
            model.fit(X_tr_model, y_tr)
            fit_seconds = time.perf_counter() - fit_start

            pred_start = time.perf_counter()
            score = model.predict_proba(X_te_model)[:, 1]
            predict_seconds = time.perf_counter() - pred_start

            pred = (score >= 0.5).astype(int)
            rows.append({
                "model": model_name,
                "param_id": param_id,
                "fold": fold,
                "params": params,
                "roc_auc": roc_auc_score(y_te, score),
                "balanced_accuracy": balanced_accuracy_score(y_te, pred),
                "f1": f1_score(y_te, pred, zero_division=0),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            })

    out = pd.DataFrame(rows)
    out["tuning_seconds_total"] = time.perf_counter() - tuning_start
    return out


tuning_frames = []
for model_name, grid in [("XGB", xgb_grid), ("LightGBM", lightgbm_grid), ("HUG-IML", hugiml_grid)]:
    print(f"Running {model_name} grid with {len(list(ParameterGrid(grid)))} parameter settings...")
    tuning_frames.append(evaluate_parameter_grid(model_name, grid, X_tune, y_tune))

tuning_results = pd.concat(tuning_frames, ignore_index=True)
tuning_results.head()


# In[12]:


param_summary = (
    tuning_results.groupby(["model", "param_id"], as_index=False)
    .agg(
        roc_auc_mean=("roc_auc", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        f1_mean=("f1", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"),
        predict_seconds_mean=("predict_seconds", "mean"),
        tuning_seconds_total=("tuning_seconds_total", "first"),
        params=("params", "first"),
    )
)

best_params = (
    param_summary.sort_values(["model", "roc_auc_mean", "balanced_accuracy_mean"], ascending=[True, False, False])
    .groupby("model", as_index=False)
    .head(1)
    .sort_values("roc_auc_mean", ascending=False)
)

best_params[[
    "model",
    "roc_auc_mean",
    "balanced_accuracy_mean",
    "f1_mean",
    "fit_seconds_mean",
    "predict_seconds_mean",
    "tuning_seconds_total",
    "params",
]].round(4)


# In[13]:


comparison = best_params[[
    "model",
    "roc_auc_mean",
    "balanced_accuracy_mean",
    "f1_mean",
    "fit_seconds_mean",
    "predict_seconds_mean",
    "tuning_seconds_total",
]].copy()
comparison = comparison.sort_values("roc_auc_mean", ascending=False)
comparison.round(4)


# In[14]:


fig, ax = plt.subplots(figsize=(8.5, 4.6))
plot_df = comparison.set_index("model")[["roc_auc_mean", "balanced_accuracy_mean", "f1_mean"]]
plot_df.plot(kind="bar", ax=ax)
ax.set_title(f"Tuned model performance on {TUNING_DATASET}")
ax.set_ylabel("Cross-validated mean score")
ax.set_xlabel("")
ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()


# In[15]:


fig, ax = plt.subplots(figsize=(8.5, 4.6))
time_df = comparison.set_index("model")[["fit_seconds_mean", "predict_seconds_mean"]]
time_df.plot(kind="bar", ax=ax)
ax.set_title(f"Mean fold fit and prediction time on {TUNING_DATASET}")
ax.set_ylabel("Seconds")
ax.set_xlabel("")
ax.legend(title="Timing", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()


# The tuning summary gives a compact view of the best cross-validated parameter setting for each model. ROC-AUC is used to select the best setting, with balanced accuracy, F1, fit time, prediction time, and total tuning time reported for comparison.

# %% [markdown]
## HUG-IML Modeling Special Cases
### Multiclass outcomes, imbalanced labels, high-cardinality categoricals, adaptive binning, and governed pattern pruning.

# %% [markdown]
#
# This notebook demonstrates practical modeling scenarios that often create implementation errors in tabular interpretable modeling. Each section uses bounded, local data and keeps transformations inside the training workflow or fits encoders on training data only.

# %%
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score

from hugiml import HUGIMLClassifierNative
from hugiml.multiclass import MulticlassHUGReport, make_imbalanced_pipeline, encode_high_cardinality, apply_encoding
from hugiml.adaptive import HUGIMLAdaptive
from hugiml.pruning import PatternEditor

RANDOM_STATE = 42
pd.set_option("display.max_colwidth", 110)

# %% [markdown]
# ## 1. Multiclass classification
#
# HUG-IML can expose class-specific pattern contributions. The iris dataset keeps the demonstration compact.

# %%
iris = load_iris(as_frame=True)
X_mc = iris.data.copy()
y_mc = pd.Series(iris.target, name="species")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_mc, y_mc, test_size=0.25, stratify=y_mc, random_state=RANDOM_STATE
)
clf_mc = HUGIMLClassifierNative(B=5, L=2, G=1e-3, topK=50)
clf_mc.fit(X_tr, y_tr)
preds = clf_mc.predict(X_te)
print(classification_report(y_te, preds, target_names=iris.target_names))

report = MulticlassHUGReport(clf_mc)
print(report.summary(top_n=5))

# %% [markdown]
# ## 2. Imbalanced binary data
#
# The balanced wrapper is compared with a plain HUG-IML fit using the same split.

# %%
X_imb_arr, y_imb = make_classification(
    n_samples=1800,
    n_features=15,
    n_informative=8,
    n_redundant=3,
    weights=[0.90, 0.10],
    class_sep=0.85,
    flip_y=0.02,
    random_state=RANDOM_STATE,
)
X_imb = pd.DataFrame(X_imb_arr, columns=[f"signal_{i:02d}" for i in range(X_imb_arr.shape[1])])
X_tr, X_te, y_tr, y_te = train_test_split(
    X_imb, y_imb, test_size=0.25, stratify=y_imb, random_state=RANDOM_STATE
)

plain = HUGIMLClassifierNative(B=6, L=2, G=5e-3, topK=80)
plain.fit(X_tr, y_tr)
plain_pred = plain.predict(X_te)
plain_score = plain.predict_proba(X_te)[:, 1]

balanced = make_imbalanced_pipeline(HUGIMLClassifierNative(B=6, L=2, G=5e-3, topK=80), strategy="class_weight")
balanced.fit(X_tr, y_tr)
bal_pred = balanced.predict(X_te)
bal_score = balanced.predict_proba(X_te)[:, 1]

imb_summary = pd.DataFrame([
    {"model": "Plain HUG-IML", "balanced_accuracy": balanced_accuracy_score(y_te, plain_pred), "roc_auc": roc_auc_score(y_te, plain_score)},
    {"model": "Balanced HUG-IML", "balanced_accuracy": balanced_accuracy_score(y_te, bal_pred), "roc_auc": roc_auc_score(y_te, bal_score)},
])
imb_summary

# %%
fig, ax = plt.subplots(figsize=(6.8, 3.8))
imb_summary.set_index("model")[["balanced_accuracy", "roc_auc"]].plot(kind="bar", ax=ax)
ax.set_ylim(0.45, 1.0)
ax.set_title("Imbalanced-label handling")
ax.set_xlabel("")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. High-cardinality categorical feature
#
# The city encoder is fitted only on the training split and then applied to the test split. This avoids target leakage.

# %%
rng = np.random.default_rng(RANDOM_STATE)
n = 1800
cities = [f"city_{i:04d}" for i in range(520)]
X_hc = pd.DataFrame({
    "city": rng.choice(cities, n),
    "age": rng.integers(18, 72, n),
    "income": rng.exponential(42000, n),
    "product": rng.choice(["A", "B", "C", "D", "E"], n),
})
city_suffix = X_hc["city"].str[-2:].astype(int)
y_hc = ((X_hc["income"] > 45000) & (X_hc["age"] > 36) | (city_suffix < 12)).astype(int).to_numpy()

X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_hc, y_hc, test_size=0.25, stratify=y_hc, random_state=RANDOM_STATE
)
X_tr_enc, enc_map = encode_high_cardinality(X_tr_raw, y_tr, threshold=12, method="target_mean")
X_te_enc = apply_encoding(X_te_raw, enc_map)

cat_cols = X_tr_enc.select_dtypes(include=["object", "category"]).columns.tolist()
int_cols = X_tr_enc.select_dtypes(include=["int", "int32", "int64"]).columns.tolist()
flt_cols = [c for c in X_tr_enc.columns if c not in cat_cols + int_cols]

hc = HUGIMLClassifierNative(
    allCols=[int_cols, flt_cols, cat_cols],
    origColumns=X_tr_enc.columns.tolist(),
    B=6,
    L=2,
    G=5e-3,
    topK=80,
)
hc.fit(X_tr_enc, y_tr)
hc_score = hc.predict_proba(X_te_enc)[:, 1]
print(f"Unique cities in raw data: {X_hc['city'].nunique():,}")
print(f"Encoded city map entries fitted on training data: {len(enc_map['city']):,}")
print(f"High-cardinality ROC-AUC: {roc_auc_score(y_te, hc_score):.3f}")
hc.get_pattern_info().merge(hc.feature_importances()[["pattern", "coefficient", "abs_coefficient"]], on="pattern").sort_values("abs_coefficient", ascending=False).head(10)

# %% [markdown]
# ## 4. Adaptive binning
#
# Adaptive binning lets the model choose feature-level bin counts from a candidate list.

# %%
adapt = HUGIMLAdaptive(b_candidates=[3, 5, 7, 10], L=2, G=5e-3)
adapt.fit(X_tr_enc, y_tr)
adapt_score = adapt.predict_proba(X_te_enc)[:, 1]
print(f"Adaptive HUG-IML ROC-AUC: {roc_auc_score(y_te, adapt_score):.3f}")
chosen_bins = pd.DataFrame(sorted(adapt.per_feature_b_.items()), columns=["feature", "chosen_B"])
chosen_bins

# %%
fig, ax = plt.subplots(figsize=(6.8, 3.6))
ax.bar(chosen_bins["feature"], chosen_bins["chosen_B"])
ax.set_title("Adaptive bin count by feature")
ax.set_ylabel("Chosen B")
ax.set_xlabel("")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Governed pattern pruning
#
# The editor creates an audit trail for low-support pattern removal, downstream refitting, and probability calibration.

# %%
editor = PatternEditor(hc, operator_name="model_risk_reviewer")
before = editor.diff()["n_original"]
editor.remove_low_support(min_support=0.02, reason="low support relative to review threshold")
after = editor.diff()["n_current"]
editor.refit(X_tr_enc, y_tr)
editor.calibrate(X_te_enc, y_te, method="isotonic")
pruned = editor.finalize()
pruned_score = pruned.predict_proba(X_te_enc)[:, 1]

print(f"Patterns before pruning: {before}")
print(f"Patterns after pruning: {after}")
print(f"Pruned calibrated ROC-AUC: {roc_auc_score(y_te, pruned_score):.3f}")
print(editor.audit_report())

# %% [markdown]
#
# These scenarios are manageable when data preparation is fold-safe, high-cardinality transformations are fitted only on training data, and pattern-editing decisions are retained as auditable model governance evidence.

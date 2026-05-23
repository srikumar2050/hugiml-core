# %% [markdown]
# # Special Cases: Multiclass, Imbalanced Data, High-Cardinality Categoricals
#
# Evaluation-safe usage note: `prepareXy` performs schema/type preparation only.
# Discretization, HUG pattern mining, and downstream classifier fitting occur
# inside `fit()` on the training data supplied to that call.

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, balanced_accuracy_score
from hugiml import HUGIMLClassifierNative
import sys; from pathlib import Path; sys.path.insert(0, str(Path("../").resolve()))

# %% [markdown]
# ## 1. Multiclass Classification (iris)

# %%
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X_mc, y_mc = iris.data, iris.target.values
print(f"Iris: {X_mc.shape}, classes={np.unique(y_mc)}")

X_tr, X_te, y_tr, y_te = train_test_split(X_mc, y_mc, test_size=0.25,
                                            stratify=y_mc, random_state=42)
clf_mc = HUGIMLClassifierNative(B=5, L=2, G=1e-3)
clf_mc.fit(X_tr, y_tr)
preds = clf_mc.predict(X_te)
print(classification_report(y_te, preds, target_names=iris.target_names))

# Per-class pattern importances
from hugiml.multiclass import MulticlassHUGReport
report = MulticlassHUGReport(clf_mc)
print(report.summary(top_n=5))

# %% [markdown]
# ## 2. Imbalanced Data (class_weight strategy)

# %%
from sklearn.datasets import make_classification
from hugiml.multiclass import make_imbalanced_pipeline

X_imb, y_imb = make_classification(
    n_samples=2000, n_features=15, n_informative=8,
    weights=[0.9, 0.1], random_state=42
)
X_imb = pd.DataFrame(X_imb, columns=[f"f{i}" for i in range(X_imb.shape[1])])
print(f"Imbalanced: {X_imb.shape}, pos_rate={y_imb.mean():.3f}")

X_tr, X_te, y_tr, y_te = train_test_split(X_imb, y_imb, test_size=0.25,
                                            stratify=y_imb, random_state=42)

# Plain HUG-IML
clf_plain = HUGIMLClassifierNative(B=6, L=2, G=5e-3)
clf_plain.fit(X_tr, y_tr)
bal_plain = balanced_accuracy_score(y_te, clf_plain.predict(X_te))

# With class_weight='balanced'
clf_proto = HUGIMLClassifierNative(B=6, L=2, G=5e-3)
clf_bal = make_imbalanced_pipeline(clf_proto, strategy="class_weight")
clf_bal.fit(X_tr, y_tr)
bal_weighted = balanced_accuracy_score(y_te, clf_bal.predict(X_te))

print(f"Plain HUG-IML  balanced_acc={bal_plain:.4f}")
print(f"class_weight   balanced_acc={bal_weighted:.4f}")

# %% [markdown]
# ## 3. High-Cardinality Categoricals

# %%
from hugiml.multiclass import encode_high_cardinality, apply_encoding

np.random.seed(42); n = 2000
cities = [f"city_{i:04d}" for i in range(500)]  # 500 unique cities
X_hc = pd.DataFrame({
    "city": np.random.choice(cities, n),
    "age":  np.random.randint(18, 70, n),
    "income": np.random.exponential(40000, n),
    "product": np.random.choice(["A","B","C","D","E"], n),
})
y_hc = ((X_hc["income"] > 40000) & (X_hc["age"] > 35)).astype(int).values

print(f"High-cardinality: {X_hc.shape}")
print(f"City cardinality: {X_hc['city'].nunique()}")

X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_hc, y_hc, test_size=0.25, stratify=y_hc, random_state=42)

# Encode HIGH-cardinality column (city) with target-mean
X_tr_enc, enc_map = encode_high_cardinality(
    X_tr_raw, y_tr, threshold=10, method="target_mean")
X_te_enc = apply_encoding(X_te_raw, enc_map)

print(f"\nAfter encoding, 'city' dtype: {X_tr_enc['city'].dtype}")
print(f"Encoding map has {len(enc_map['city'])} city keys (fitted on train only)")

# Fit HUG-IML on encoded data
cat_cols = X_tr_enc.select_dtypes(include=["object","category"]).columns.tolist()
int_cols = X_tr_enc.select_dtypes(include="int").columns.tolist()
flt_cols = [c for c in X_tr_enc.columns if c not in cat_cols + int_cols]

clf_hc = HUGIMLClassifierNative(
    allCols=[int_cols, flt_cols, cat_cols],
    origColumns=X_tr_enc.columns.tolist(),
    B=6, L=2, G=5e-3,
)
clf_hc.fit(X_tr_enc, y_tr)
proba = clf_hc.predict_proba(X_te_enc)[:, 1]
print(f"\nHUG-IML on high-card data: ROC-AUC = {roc_auc_score(y_te, proba):.4f}")
print(clf_hc.model_summary())

# %% [markdown]
# ## 4. Adaptive Binning

# %%
from hugiml.adaptive import HUGIMLAdaptive

clf_adapt = HUGIMLAdaptive(b_candidates=[3,5,7,10,15], L=2, G=5e-3)
clf_adapt.prepareXy(X_hc, y_hc)
clf_adapt.fit(X_tr_enc, y_tr)
proba_adapt = clf_adapt.predict_proba(X_te_enc)[:, 1]
print(f"Adaptive HUG-IML: ROC-AUC = {roc_auc_score(y_te, proba_adapt):.4f}")
print("Per-feature chosen B:")
for feat, b in sorted(clf_adapt.per_feature_b_.items()):
    print(f"  {feat:<20} B={b}")

# %% [markdown]
# ## 5. Pattern Pruning (regulated editing workflow)

# %%
from hugiml.pruning import PatternEditor

editor = PatternEditor(clf_hc, operator_name="data_scientist")
print(f"Before: {editor.diff()['n_original']} patterns")

# Remove patterns with support < 2%
editor.remove_low_support(min_support=0.02, reason="low support — unstable in production")
print(f"After low-support removal: {editor.diff()['n_current']} patterns")

# Refit downstream classifier
editor.refit(X_tr_enc, y_tr)

# Calibrate probabilities on a held-out calibration set
editor.calibrate(X_te_enc, y_te, method="isotonic")

# Export to final model
new_clf = editor.finalize()
print(editor.audit_report())

proba_pruned = new_clf.predict_proba(X_te_enc)[:, 1]
print(f"Pruned model: ROC-AUC = {roc_auc_score(y_te, proba_pruned):.4f}")

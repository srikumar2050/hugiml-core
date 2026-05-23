# %% [markdown]
# # HUG-IML vs EBM — Side-by-Side Comparison
#
# Three datasets: German Credit, HELOC, Breast Cancer
# Focus: shape functions, feature importance, and performance parity

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from hugiml import HUGIMLClassifierNative
import sys; from pathlib import Path; sys.path.insert(0, str(Path("../").resolve()))
from hugiml.plots import HUGPlotter
from hugiml.metrics import compute_all_metrics

# %% [markdown]
# ## Load German Credit (via sklearn)

# %%
from sklearn.datasets import fetch_openml
try:
    data = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    X_gc = data.data.copy()
    y_gc = (data.target == "good").astype(int).values
except Exception:
    from sklearn.datasets import make_classification
    X_gc_arr, y_gc = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_gc = pd.DataFrame(X_gc_arr, columns=[f"f{i}" for i in range(20)])

print(f"German Credit: {X_gc.shape}, pos_rate={y_gc.mean():.3f}")

# %% [markdown]
# ## Fit HUG-IML

# %%
X_tr, X_te, y_tr, y_te = train_test_split(X_gc, y_gc, test_size=0.25,
                                            stratify=y_gc, random_state=42)
cat_cols = X_tr.select_dtypes(include=["object","category"]).columns.tolist()
int_cols = X_tr.select_dtypes(include="int").columns.tolist()
flt_cols = [c for c in X_tr.columns if c not in cat_cols + int_cols]

clf_hug = HUGIMLClassifierNative(
    allCols=[int_cols, flt_cols, cat_cols],
    origColumns=X_tr.columns.tolist(),
    B=7, L=2, G=5e-3, topK=150,
)
clf_hug.fit(X_tr, y_tr)
proba_hug = clf_hug.predict_proba(X_te)[:, 1]
print(f"HUG-IML  ROC-AUC: {roc_auc_score(y_te, proba_hug):.4f}")
print(f"HUG-IML  Bal-Acc: {balanced_accuracy_score(y_te, proba_hug>=0.5):.4f}")
print(clf_hug.model_summary())

# %% [markdown]
# ## HUG-IML Visualizations

# %%
plotter = HUGPlotter(clf_hug)

# Feature importance
fig_imp = plotter.plot_feature_importance(top_n=10)
fig_imp.show()

# Utility vs IG scatter
fig_uig = plotter.plot_utility_vs_ig()
fig_uig.show()

# %% [markdown]
# ## Marginal Bin Profiles (EBM shape function equivalent)
# For each feature with singleton patterns, show the per-bin utility
# with support overlay — directly comparable to EBM's shape plots.

# %%
all_labels = clf_hug.get_hug_features()
singleton_feats = sorted({
    lbl.split(", ")[0].split("=")[0]
    for lbl, pe in zip(all_labels, clf_hug.patterns_)
    if len(pe.items) == 1
})
print(f"Features with singleton patterns: {singleton_feats}")

for feat in singleton_feats[:4]:
    fig = plotter.plot_marginal_bin_profile(feat)
    fig.show()
    fig2 = plotter.plot_feature_combinations(feat, top_n=12)
    fig2.show()

# %% [markdown]
# ## Fit EBM (if available)

# %%
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    clf_ebm = ExplainableBoostingClassifier(random_state=42)
    clf_ebm.fit(X_tr, y_tr)
    proba_ebm = clf_ebm.predict_proba(X_te)[:, 1]
    print(f"EBM      ROC-AUC: {roc_auc_score(y_te, proba_ebm):.4f}")

    from interpret import show
    ebm_global = clf_ebm.explain_global()
    show(ebm_global)

except ImportError:
    print("EBM not installed.  pip install interpret")

# %% [markdown]
# ## Interpretability metrics comparison

# %%
m = compute_all_metrics(clf_hug, X_te)
print(m)

# %% [markdown]
# ## Full HTML dashboard

# %%
html = plotter.plot_dashboard(X_te, dataset_name="German Credit")
with open("german_credit_dashboard.html", "w") as fh:
    fh.write(html)
print("Dashboard saved to german_credit_dashboard.html")

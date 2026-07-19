# %% [markdown]
# <div style="padding:18px 22px;border-radius:16px;>
# <h1 style="margin:0">HUG Pattern Explanation Walkthrough</h1>
# <p style="margin:8px 0 0 0;font-size:15px">A compact, practical introduction to what a HUG pattern is, how patterns are mined, and how fired patterns can be translated into sample-level explanations.</p>
# </div>

# %% [markdown]
#
# A **HUG pattern** is a human-readable condition over one or more discretized feature regions. In a fitted HUG-IML model, each pattern becomes an interpretable binary signal: it either fires for a sample or it does not.
#
# This walkthrough trains a small model, lists the highest-value patterns, selects one observation, identifies which patterns fired, and converts their learned contributions into a plain-English explanation.

# %%
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, classification_report

from hugiml import HUGIMLClassifier

pd.set_option("display.max_colwidth", 110)
RANDOM_STATE = 42

# %% [markdown]
# ## 1. Train on a small dataset
#
# The breast-cancer dataset is small enough to make every step inspectable. The target below is encoded as **1 = benign** and **0 = malignant**, matching the scikit-learn dataset convention.

# %%
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = pd.Series(data.target, name="benign")

# A bounded sample keeps the walkthrough fast while preserving class balance.
X_small, _, y_small, _ = train_test_split(
    X, y, train_size=240, stratify=y, random_state=RANDOM_STATE
)
X_train, X_test, y_train, y_test = train_test_split(
    X_small, y_small, test_size=0.25, stratify=y_small, random_state=RANDOM_STATE
)

clf = HUGIMLClassifier(B=5, L=2, G=1e-3, topK=60)
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
print(f"Training rows: {X_train.shape[0]:,}; test rows: {X_test.shape[0]:,}; features: {X_train.shape[1]:,}")
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred, target_names=["malignant", "benign"]))

# %% [markdown]
# ## 2. Show the top 10 patterns
#
# Patterns are ranked here by the absolute downstream coefficient. The coefficient is the direction and strength of the pattern after the pattern matrix is passed into the logistic downstream model.

# %%
pattern_info = clf.get_pattern_info().copy()
importance = clf.feature_importances().copy()

top_patterns = (
    pattern_info.merge(importance[["pattern", "coefficient", "abs_coefficient"]], on="pattern", how="left")
    .sort_values(["abs_coefficient", "utility"], ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top_patterns.insert(0, "rank", np.arange(1, len(top_patterns) + 1))
top_patterns[["rank", "pattern", "coefficient", "utility", "information_gain", "support"]]

# %%
fig, ax = plt.subplots(figsize=(9, 4.8))
plot_df = top_patterns.iloc[::-1]
ax.barh(plot_df["pattern"], plot_df["coefficient"])
ax.axvline(0, linewidth=1)
ax.set_title("Top HUG patterns by downstream contribution")
ax.set_xlabel("Logistic coefficient for benign class")
ax.set_ylabel("")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Pick one sample
#
# We choose the test observation closest to the decision boundary. This makes the explanation interesting because positive and negative signals both matter.

# %%
sample_pos = int(np.argmin(np.abs(proba - 0.5)))
sample_idx = X_test.index[sample_pos]
sample = X_test.iloc[[sample_pos]]
print(f"Selected test index: {sample_idx}")
print(f"Observed class: {data.target_names[int(y_test.iloc[sample_pos])]}")
print(f"Predicted probability of benign: {proba[sample_pos]:.3f}")
sample.T.rename(columns={sample_idx: "value"}).head(12)

# %% [markdown]
# ## 4. Show which patterns fired
#
# The model transforms the row into a sparse pattern matrix. A value of 1 means the row satisfies the pattern condition.

# %%
Z = clf.transform(sample)
fired_positions = Z.nonzero()[1]
labels = clf.get_hug_features()
coef_map = importance.set_index("pattern")["coefficient"].to_dict()
info_map = pattern_info.set_index("pattern")[["utility", "information_gain", "support"]].to_dict("index")

fired_rows = []
for j in fired_positions:
    label = labels[j]
    stats = info_map.get(label, {})
    fired_rows.append({
        "pattern": label,
        "coefficient": float(coef_map.get(label, np.nan)),
        "utility": float(stats.get("utility", np.nan)),
        "information_gain": float(stats.get("information_gain", np.nan)),
        "support": float(stats.get("support", np.nan)),
    })

fired = pd.DataFrame(fired_rows).sort_values("coefficient", ascending=False).reset_index(drop=True)
fired

# %%
fig, ax = plt.subplots(figsize=(8.5, max(3, 0.38 * len(fired))))
plot_df = fired.sort_values("coefficient")
ax.barh(plot_df["pattern"], plot_df["coefficient"])
ax.axvline(0, linewidth=1)
ax.set_title("Patterns fired for the selected sample")
ax.set_xlabel("Contribution direction: negative ← → positive benign evidence")
ax.set_ylabel("")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Convert pattern contributions into plain-English explanation
#
# The explanation below uses three ingredients: the pattern condition, whether it supports the benign or malignant side, and the magnitude of the coefficient.

# %%
def describe_pattern(label: str, coefficient: float) -> str:
    direction = "supports the benign prediction" if coefficient >= 0 else "pushes away from the benign prediction"
    strength = "strongly" if abs(coefficient) >= 1.0 else "moderately" if abs(coefficient) >= 0.5 else "mildly"
    clean = label.replace("=", " in ")
    return f"When {clean}, this pattern {strength} {direction}."

if fired.empty:
    print("No mined pattern fired for this sample. The downstream intercept dominates this prediction.")
else:
    top_supporting = fired.sort_values("coefficient", ascending=False).head(3)
    top_opposing = fired.sort_values("coefficient", ascending=True).head(3)
    print("Plain-English explanation for the selected row:\n")
    print(f"The model estimated a {proba[sample_pos]:.1%} probability of the benign class.")
    print("\nMain supporting signals:")
    for _, row in top_supporting.iterrows():
        print(f"- {describe_pattern(row['pattern'], row['coefficient'])}")
    print("\nMain opposing signals:")
    for _, row in top_opposing.iterrows():
        if row["coefficient"] < 0:
            print(f"- {describe_pattern(row['pattern'], row['coefficient'])}")

# %% [markdown]
#
# HUG patterns are useful because they provide a transparent bridge between tabular data and model behavior. Instead of asking reviewers to inspect opaque trees or latent features, HUG-IML exposes compact rules, support levels, information gain, utility, and row-level firing behavior.

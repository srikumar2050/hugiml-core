#!/usr/bin/env python
# coding: utf-8

# ## HUG-IML versus Explainable Boosting Machine
# ### Pattern inventory, bin-profile evidence, EBM shape functions, and local explanations on a common clinical classification task.

# This notebook compares two interpretable supervised models trained on the same data split:
# 
# - **HUG-IML** represents evidence as discrete interval patterns. A fired pattern contributes a learned coefficient to the prediction.
# - **Explainable Boosting Machine (EBM)** represents evidence as additive feature and interaction shape functions.
# 
# The review emphasizes not only predictive metrics, but also whether the two explanation systems tell compatible stories. HUG bin profiles are placed next to EBM shape functions so that thresholded pattern evidence can be reviewed against smooth additive effects.

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    brier_score_loss,
    RocCurveDisplay,
)
from sklearn.calibration import calibration_curve

from hugiml import HUGIMLClassifierNative
from interpret.glassbox import ExplainableBoostingClassifier

RANDOM_STATE = 42
pd.set_option("display.max_colwidth", 160)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

PALETTE = {
    "hug": "#0f766e",
    "ebm": "#1d4ed8",
    "neutral": "#334155",
    "muted": "#64748b",
    "accent": "#f59e0b",
    "risk": "#be123c",
    "surface": "#f8fafc",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": PALETTE["surface"],
    "axes.edgecolor": "#94a3b8",
    "axes.grid": True,
    "grid.alpha": 0.24,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "font.size": 9.5,
})


# ## 1. Dataset and common split
# 
# The target is encoded as `1 = benign`. Both models receive identical training rows and are evaluated on identical held-out rows.

# In[2]:


data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = pd.Series(data.target, name="benign")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

summary = pd.DataFrame({
    "partition": ["train", "test", "overall"],
    "rows": [len(X_train), len(X_test), len(X)],
    "positive_rate_benign": [y_train.mean(), y_test.mean(), y.mean()],
    "features": [X_train.shape[1], X_test.shape[1], X.shape[1]],
})
summary


# In[3]:


fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
class_counts = y.value_counts().sort_index().rename(index={0: "malignant", 1: "benign"})
axes[0].bar(class_counts.index, class_counts.values, color=[PALETTE["risk"], PALETTE["hug"]])
axes[0].set_title("Class balance")
axes[0].set_ylabel("Rows")
axes[0].tick_params(axis="x", rotation=0)

sample_features = ["mean radius", "mean texture", "worst concave points"]
X[sample_features].plot(kind="box", ax=axes[1], color={"boxes": PALETTE["neutral"], "medians": PALETTE["accent"]})
axes[1].set_title("Selected feature ranges")
axes[1].tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()


# ## 2. Train HUG-IML
# 
# HUG-IML learns interval-based patterns. Coefficients indicate how much a fired pattern moves the prediction toward or away from the benign class.

# In[4]:


hug = HUGIMLClassifierNative(B=8, L=2, G=2e-3, topK=120)
hug.fit(X_train, y_train)

hug_score = hug.predict_proba(X_test)[:, 1]
hug_pred = (hug_score >= 0.5).astype(int)

print(hug.model_summary())


# ## 3. Train Explainable Boosting Machine
# 
# The comparator is a real `interpret.glassbox.ExplainableBoostingClassifier`. The configuration is deterministic and keeps training compact while still allowing a small set of pairwise interactions.

# In[5]:


ebm = ExplainableBoostingClassifier(
    random_state=RANDOM_STATE,
    interactions=3,
    max_rounds=80,
    learning_rate=0.04,
    n_jobs=1,
)
ebm.fit(X_train, y_train)

ebm_score = ebm.predict_proba(X_test)[:, 1]
ebm_pred = (ebm_score >= 0.5).astype(int)

print(f"Trained EBM terms: {len(ebm.term_names_)}")
print("First terms:", ebm.term_names_[:8])


# ## 4. Predictive comparison
# 
# Metrics are computed on the same holdout partition. The compact display groups rank metrics and probability-quality metrics side by side.

# In[6]:


def score_row(name, proba, pred):
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, proba),
        "avg_precision": average_precision_score(y_test, proba),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "log_loss": log_loss(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
    }

perf = pd.DataFrame([
    score_row("HUG-IML", hug_score, hug_pred),
    score_row("EBM", ebm_score, ebm_pred),
])
perf


# In[7]:


fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
rank_metrics = ["roc_auc", "avg_precision", "balanced_accuracy"]
loss_metrics = ["log_loss", "brier_score"]
perf.set_index("model")[rank_metrics].plot(kind="bar", ax=axes[0], color=[PALETTE["hug"], PALETTE["ebm"], PALETTE["accent"]])
axes[0].set_ylim(0.85, 1.01)
axes[0].set_title("Rank and threshold metrics")
axes[0].set_xlabel("")
axes[0].set_ylabel("Higher is better")
axes[0].tick_params(axis="x", rotation=0)

perf.set_index("model")[loss_metrics].plot(kind="bar", ax=axes[1], color=[PALETTE["risk"], PALETTE["muted"]])
axes[1].set_title("Probability quality metrics")
axes[1].set_xlabel("")
axes[1].set_ylabel("Lower is better")
axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()


# In[8]:


cal_rows = []
for name, score in [("HUG-IML", hug_score), ("EBM", ebm_score)]:
    frac_pos, mean_pred = calibration_curve(y_test, score, n_bins=6, strategy="quantile")
    for idx in range(len(mean_pred)):
        cal_rows.append({"model": name, "quantile_bin": idx + 1, "mean_predicted": mean_pred[idx], "observed_rate": frac_pos[idx]})
cal_df = pd.DataFrame(cal_rows)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
RocCurveDisplay.from_predictions(y_test, hug_score, name="HUG-IML", ax=axes[0], color=PALETTE["hug"])
RocCurveDisplay.from_predictions(y_test, ebm_score, name="EBM", ax=axes[0], color=PALETTE["ebm"])
axes[0].set_title("ROC curves")

for name, df in cal_df.groupby("model"):
    color = PALETTE["hug"] if name == "HUG-IML" else PALETTE["ebm"]
    axes[1].plot(df["mean_predicted"], df["observed_rate"], marker="o", label=name, color=color)
axes[1].plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="ideal", color=PALETTE["muted"])
axes[1].set_title("Calibration by probability quantile")
axes[1].set_xlabel("Mean predicted probability")
axes[1].set_ylabel("Observed benign rate")
axes[1].legend()
plt.tight_layout()
plt.show()

cal_df


# ## 5. Global explanation inventories
# 
# The first inventory is pattern-based: HUG reports direct conditions and their coefficients. The second inventory is term-based: EBM reports average absolute additive contribution by feature or interaction term.

# In[9]:


def ebm_term_importance_table(ebm_model):
    rows = []
    for term_index, term in enumerate(ebm_model.term_names_):
        scores_arr = np.asarray(ebm_model.term_scores_[term_index], dtype=float).ravel()
        weights_arr = np.asarray(ebm_model.bin_weights_[term_index], dtype=float).ravel()
        n = min(scores_arr.size, weights_arr.size)
        if n == 0 or weights_arr[:n].sum() <= 0:
            importance = float(np.mean(np.abs(scores_arr))) if scores_arr.size else 0.0
        else:
            importance = float(np.average(np.abs(scores_arr[:n]), weights=weights_arr[:n]))
        rows.append({"term": term, "importance": importance})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)

pattern_table = hug.get_pattern_info().merge(
    hug.feature_importances()[["pattern", "coefficient", "abs_coefficient", "support"]].rename(columns={"support": "model_support"}),
    on="pattern",
    how="left",
).sort_values("abs_coefficient", ascending=False)

ebm_importance = ebm_term_importance_table(ebm)

display(pattern_table.head(12))
display(ebm_importance.head(12))


# In[10]:


fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
hug_plot = pattern_table.head(12).iloc[::-1]
ebm_plot = ebm_importance.head(12).iloc[::-1]
axes[0].barh(hug_plot["pattern"], hug_plot["coefficient"], color=np.where(hug_plot["coefficient"] >= 0, PALETTE["hug"], PALETTE["risk"]))
axes[0].axvline(0, linewidth=1, color=PALETTE["neutral"])
axes[0].set_title("HUG: highest-impact patterns")
axes[0].set_xlabel("Coefficient toward benign")
axes[0].set_ylabel("")

axes[1].barh(ebm_plot["term"], ebm_plot["importance"], color=PALETTE["ebm"])
axes[1].set_title("EBM: top terms")
axes[1].set_xlabel("Average absolute contribution")
axes[1].set_ylabel("")
plt.tight_layout()
plt.show()


# ## 6. HUG bin profiles versus EBM shape functions
# 
# This is the central interpretability comparison. For the same features, the left panels show **HUG bin profiles**: training rows are grouped into empirical quantile bins, and each bin summarizes the average HUG pattern contribution involving that feature. The right panels show the corresponding **EBM shape function** for the same feature. Positive values push toward benign; negative values push away from benign.

# In[11]:


def extract_feature_from_pattern(pattern, feature_names):
    matched = [name for name in feature_names if pattern.startswith(name + "=") or (name + "=") in pattern]
    return max(matched, key=len) if matched else None

hug_fi = hug.feature_importances().copy()
hug_fi["feature"] = hug_fi["pattern"].apply(lambda p: extract_feature_from_pattern(p, list(X.columns)))
hug_feature_importance = (
    hug_fi.dropna(subset=["feature"])
    .groupby("feature", as_index=False)["abs_coefficient"]
    .sum()
    .sort_values("abs_coefficient", ascending=False)
)

single_ebm_terms = [t for t in ebm_importance["term"] if t in X.columns]
common_features = [f for f in single_ebm_terms if f in set(hug_feature_importance["feature"])]
# Prefer features that matter to both systems.
feature_rank = pd.DataFrame({"feature": common_features}).merge(hug_feature_importance, on="feature", how="left")
feature_rank = feature_rank.merge(ebm_importance.rename(columns={"term": "feature", "importance": "ebm_importance"}), on="feature", how="left")
feature_rank["joint_rank_score"] = feature_rank["abs_coefficient"].rank(ascending=False) + feature_rank["ebm_importance"].rank(ascending=False)
features_for_shape_review = feature_rank.sort_values("joint_rank_score")["feature"].head(2).tolist()
features_for_shape_review


# In[12]:


def clean_number_label(value):
    """Format numeric axis labels without unnecessary scientific notation."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return str(value)
    abs_value = abs(numeric)
    if abs_value == 0:
        return "0"
    if abs_value >= 1000:
        text = f"{numeric:,.0f}" if np.isclose(numeric, round(numeric)) else f"{numeric:,.1f}"
    elif abs_value >= 100:
        text = f"{numeric:.1f}"
    elif abs_value >= 10:
        text = f"{numeric:.2f}"
    elif abs_value >= 1:
        text = f"{numeric:.2f}"
    elif abs_value >= 0.01:
        text = f"{numeric:.3f}"
    else:
        text = f"{numeric:.4f}"
    return text.rstrip("0").rstrip(".")

def clean_tick_label(value):
    """Clean scalar or interval labels returned by EBM without changing their meaning."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return clean_number_label(value)
    text = str(value)
    # Replace numeric tokens, including values such as 1.23e+04, with plain-number labels.
    return re.sub(
        r"(?<![A-Za-z])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
        lambda m: clean_number_label(m.group(0)),
        text,
    )

def clean_range_label(left, right):
    return f"{clean_number_label(left)}–{clean_number_label(right)}"

def hug_bin_profile(feature, n_bins=8):
    patterns_for_feature = hug_fi.loc[hug_fi["feature"] == feature, ["pattern", "coefficient"]]
    if patterns_for_feature.empty:
        raise ValueError(f"No HUG patterns found for {feature}")
    hug_feature_matrix = pd.DataFrame(
        hug.transform(X_train).toarray(),
        columns=np.array(hug.get_hug_features()),
        index=X_train.index,
    )
    available_patterns = [p for p in patterns_for_feature["pattern"] if p in hug_feature_matrix.columns]
    coef = patterns_for_feature.set_index("pattern").loc[available_patterns, "coefficient"]
    contrib = hug_feature_matrix[available_patterns].mul(coef, axis=1).sum(axis=1)
    frame = pd.DataFrame({
        "feature_value": X_train[feature],
        "hug_feature_contribution": contrib,
        "actual_benign": y_train,
    })
    frame["bin"] = pd.qcut(frame["feature_value"], q=min(n_bins, frame["feature_value"].nunique()), duplicates="drop")
    profile = frame.groupby("bin", observed=True).agg(
        bin_mid=("feature_value", "mean"),
        min_value=("feature_value", "min"),
        max_value=("feature_value", "max"),
        mean_hug_contribution=("hug_feature_contribution", "mean"),
        benign_rate=("actual_benign", "mean"),
        rows=("actual_benign", "size"),
    ).reset_index(drop=True)
    profile["bin_label"] = profile.apply(lambda r: clean_range_label(r["min_value"], r["max_value"]), axis=1)
    return profile

def ebm_shape_frame(feature):
    idx = ebm.term_names_.index(feature)
    data = ebm.explain_global().data(idx)
    names = list(data["names"])
    scores = np.asarray(data["scores"], dtype=float).ravel()
    n = min(len(names), len(scores))
    x = np.arange(n)
    return pd.DataFrame({"grid_index": x, "value": names[:n], "clean_value": [clean_tick_label(v) for v in names[:n]], "score": scores[:n]})

profiles = {feature: hug_bin_profile(feature) for feature in features_for_shape_review}
shapes = {feature: ebm_shape_frame(feature) for feature in features_for_shape_review}

for feature in features_for_shape_review:
    print(f"{feature}: HUG profile bins={len(profiles[feature])}, EBM grid points={len(shapes[feature])}")


# In[13]:


fig, axes = plt.subplots(len(features_for_shape_review), 2, figsize=(12.8, 3.8 * len(features_for_shape_review)))
if len(features_for_shape_review) == 1:
    axes = np.array([axes])

for row_idx, feature in enumerate(features_for_shape_review):
    profile = profiles[feature]
    shape = shapes[feature]
    x_profile = np.arange(len(profile))

    ax = axes[row_idx, 0]
    ax.plot(x_profile, profile["mean_hug_contribution"], marker="o", linewidth=1.8, color=PALETTE["hug"], label="mean HUG contribution")
    ax.axhline(0, linewidth=1, color=PALETTE["neutral"], alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(x_profile, profile["benign_rate"], marker="s", linestyle="--", linewidth=1.5, color=PALETTE["accent"], label="observed benign rate")
    ax.set_title(f"HUG bin profile: {feature}")
    ax.set_xlabel("Empirical feature bin")
    ax.set_ylabel("Pattern contribution")
    ax2.set_ylabel("Observed benign rate")
    ax.set_xticks(x_profile)
    ax.set_xticklabels(profile["bin_label"], rotation=18, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: clean_number_label(x)))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: clean_number_label(x)))
    ax.margins(x=0.03)
    ax2.set_ylim(-0.03, 1.03)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8, frameon=True)

    ax = axes[row_idx, 1]
    ax.plot(shape["grid_index"], shape["score"], linewidth=1.8, color=PALETTE["ebm"])
    ax.fill_between(shape["grid_index"], 0, shape["score"], alpha=0.15, color=PALETTE["ebm"])
    ax.axhline(0, linewidth=1, color=PALETTE["neutral"], alpha=0.85)
    ax.set_title(f"EBM shape function: {feature}")
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Additive contribution")
    if len(shape) > 8:
        ticks = np.unique(np.linspace(0, len(shape) - 1, 6, dtype=int))
    else:
        ticks = np.arange(len(shape))
    ax.set_xticks(ticks)
    ax.set_xticklabels(shape.loc[ticks, "clean_value"], rotation=18, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: clean_number_label(x)))
    ax.margins(x=0.02)

plt.tight_layout(pad=1.2, w_pad=2.4, h_pad=2.0)
plt.show()


# ### What this comparison adds
# 
# HUG profiles make the thresholded pattern representation visible at the feature level. EBM shape functions make the additive curve visible at the same feature level. Reviewing them side by side is useful because:
# 
# - a HUG jump near a clinically meaningful range should align with a comparable EBM shape change;
# - disagreement can reveal discretization sensitivity, nonlinear behavior, or interaction effects;
# - the observed benign-rate overlay helps distinguish model evidence from the empirical class profile.

# In[14]:


profile_summary_rows = []
for feature in features_for_shape_review:
    p = profiles[feature]
    s = shapes[feature]
    profile_summary_rows.append({
        "feature": feature,
        "hug_contribution_range": p["mean_hug_contribution"].max() - p["mean_hug_contribution"].min(),
        "ebm_shape_range": s["score"].max() - s["score"].min(),
        "observed_benign_rate_range": p["benign_rate"].max() - p["benign_rate"].min(),
        "hug_bins": len(p),
        "ebm_grid_points": len(s),
    })
pd.DataFrame(profile_summary_rows)


# ## 7. Same-sample explanation
# 
# The selected holdout observation is the case where the HUG and EBM probabilities differ most. That makes it a useful stress case for comparing explanation forms.

# In[15]:


comparison = pd.DataFrame({
    "actual_benign": y_test.to_numpy(),
    "hug_probability": hug_score,
    "ebm_probability": ebm_score,
}, index=X_test.index)
comparison["absolute_model_gap"] = (comparison["hug_probability"] - comparison["ebm_probability"]).abs()
selected_index = comparison.sort_values("absolute_model_gap", ascending=False).index[0]
selected_position = list(X_test.index).index(selected_index)

sample = X_test.loc[[selected_index]]
print(f"Selected holdout index: {selected_index}")
print(f"Actual class: {'benign' if int(y_test.loc[selected_index]) == 1 else 'malignant'}")
print(f"HUG probability of benign: {hug_score[selected_position]:.4f}")
print(f"EBM probability of benign: {ebm_score[selected_position]:.4f}")

sample.T.rename(columns={selected_index: "value"}).head(12)


# In[16]:


hug_features = np.array(hug.get_hug_features())
hug_design = hug.transform(sample).toarray()[0]
coef_lookup = hug.feature_importances().set_index("pattern")["coefficient"].to_dict()

fired_patterns = pd.DataFrame({"pattern": hug_features[hug_design > 0]})
fired_patterns["contribution"] = fired_patterns["pattern"].map(coef_lookup)
fired_patterns["direction"] = np.where(fired_patterns["contribution"] >= 0, "toward benign", "toward malignant")
fired_patterns["feature"] = fired_patterns["pattern"].apply(lambda p: extract_feature_from_pattern(p, list(X.columns)))
fired_patterns = fired_patterns.reindex(fired_patterns["contribution"].abs().sort_values(ascending=False).index)

local = ebm.explain_local(sample, pd.Series([y_test.loc[selected_index]], index=sample.index)).data(0)
ebm_local = pd.DataFrame({
    "term": local["names"],
    "value": local["values"],
    "contribution": local["scores"],
})
ebm_local["direction"] = np.where(ebm_local["contribution"] >= 0, "toward benign", "toward malignant")
ebm_local = ebm_local.reindex(ebm_local["contribution"].abs().sort_values(ascending=False).index)

display(fired_patterns.head(12))
display(ebm_local.head(12))


# In[17]:


fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
fp = fired_patterns.head(10).iloc[::-1]
el = ebm_local.head(10).iloc[::-1]
axes[0].barh(fp["pattern"], fp["contribution"], color=np.where(fp["contribution"] >= 0, PALETTE["hug"], PALETTE["risk"]))
axes[0].axvline(0, linewidth=1, color=PALETTE["neutral"])
axes[0].set_title("HUG fired-pattern contributions")
axes[0].set_xlabel("Coefficient contribution")
axes[0].set_ylabel("")

axes[1].barh(el["term"], el["contribution"], color=np.where(el["contribution"] >= 0, PALETTE["ebm"], PALETTE["risk"]))
axes[1].axvline(0, linewidth=1, color=PALETTE["neutral"])
axes[1].set_title("EBM local term contributions")
axes[1].set_xlabel("Additive contribution")
axes[1].set_ylabel("")
plt.tight_layout()
plt.show()


# ## 8. Local feature-level reconciliation
# 
# The table below aggregates the sample-level evidence by feature where a direct mapping is possible. HUG evidence is summed across fired patterns involving the feature. EBM evidence is the local additive contribution of the same feature term, excluding interaction terms.

# In[18]:


hug_local_feature = fired_patterns.dropna(subset=["feature"]).groupby("feature", as_index=False)["contribution"].sum().rename(columns={"contribution": "hug_local_contribution"})
ebm_local_feature = ebm_local[ebm_local["term"].isin(X.columns)][["term", "contribution"]].rename(columns={"term": "feature", "contribution": "ebm_local_contribution"})
local_recon = hug_local_feature.merge(ebm_local_feature, on="feature", how="outer").fillna(0.0)
local_recon["absolute_combined_evidence"] = local_recon["hug_local_contribution"].abs() + local_recon["ebm_local_contribution"].abs()
local_recon = local_recon.sort_values("absolute_combined_evidence", ascending=False)
local_recon.head(12)


# In[19]:


plot_recon = local_recon.head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(9.8, 4.8))
ypos = np.arange(len(plot_recon))
width = 0.36
ax.barh(ypos - width/2, plot_recon["hug_local_contribution"], height=width, label="HUG", color=PALETTE["hug"])
ax.barh(ypos + width/2, plot_recon["ebm_local_contribution"], height=width, label="EBM", color=PALETTE["ebm"])
ax.axvline(0, linewidth=1, color=PALETTE["neutral"])
ax.set_yticks(ypos)
ax.set_yticklabels(plot_recon["feature"])
ax.set_title("Same-sample evidence by feature")
ax.set_xlabel("Contribution toward benign")
ax.legend()
plt.tight_layout()
plt.show()


# ## 9. Plain-English explanation for the selected sample

# In[20]:


def phrase_hug(row):
    return f"{row['pattern']} ({row['direction']}, contribution {row['contribution']:+.3f})"

def phrase_ebm(row):
    val = row["value"] if row["value"] != "" else "interaction"
    return f"{row['term']} = {val} ({row['direction']}, contribution {row['contribution']:+.3f})"

hug_top = fired_patterns.head(5).apply(phrase_hug, axis=1).tolist()
ebm_top = ebm_local.head(5).apply(phrase_ebm, axis=1).tolist()

print("HUG explanation:")
print("The HUG model predicts this case by checking which learned interval patterns fire. The strongest fired signals are:")
for item in hug_top:
    print(" - " + item)
print()
print("EBM explanation:")
print("The EBM prediction is the intercept plus additive feature and interaction contributions. The strongest terms are:")
for item in ebm_top:
    print(" - " + item)


# ## 10. Governance interpretation
# 
# The direct comparison shows two complementary styles of interpretability:
# 
# - **HUG-IML** provides compact, rule-like pattern evidence. This is strong for audit inventories, threshold checks, and sample-level firing explanations.
# - **EBM** provides additive shape functions. This is strong for reviewing monotonicity, nonlinear feature effects, and term-by-term additive reasoning.
# - The bin-profile versus shape-function panels make the comparison richer than a metric-only review: they show whether HUG’s discrete intervals align with EBM’s learned effect curves.
# - Local reconciliation helps isolate disagreement. A gap can come from a fired HUG threshold, an EBM univariate curve, or an EBM interaction term.

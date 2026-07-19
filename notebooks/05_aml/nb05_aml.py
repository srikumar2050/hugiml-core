import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix,
    precision_recall_fscore_support, accuracy_score,
)
warnings.filterwarnings("ignore")

from hugiml import HUGIMLClassifier
from hugiml.calibration import evaluate_calibration
from hugiml.metrics import compute_all_metrics
from hugiml.pruning import PatternEditor
from hugiml.governance import generate_model_card

RANDOM_STATE = 42
OUTDIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(OUTDIR, "nb05_aml_data.csv")
METADATA_FILE = os.path.join(OUTDIR, "nb05_aml_metadata.csv")
TARGET = "suspicious"

# AML typology labels — map pattern keywords to FinCEN typology names
TYPOLOGY_MAP = {
    "structuring_score": "Structuring / CTR Evasion",
    "amount":            "Structuring / CTR Evasion",
    "velocity_24h":      "Velocity Abuse / Rapid Movement",
    "account_age_days":  "New-Account Exploitation",
    "num_previous_transactions": "Layering / Sparse History",
    "hour_of_day":       "Off-Hours Activity",
    "country":           "Geographic Risk / Offshore Routing",
    "high_risk_country": "Offshore Jurisdiction Flag",
    "avg_monthly_balance": "Balance Anomaly",
    "customer_age":      "Demographic Profiling (review required)",
    "transaction_type":  "Transaction-Type Risk",
    "customer_type":     "Customer Segment Risk",
}

THEME = dict(
    bg="#fbf7ef", panel="#fffdf8", text="#23312a",
    muted="#6b5f4b", accent="#9a6a2f", accent2="#2f6b54",
    accent3="#7a2f2f", grid="#e5dccb",
)
plt.rcParams.update({
    "figure.facecolor": THEME["bg"],
    "axes.facecolor": THEME["panel"],
    "axes.edgecolor": THEME["grid"],
    "axes.labelcolor": THEME["text"],
    "xtick.color": THEME["muted"],
    "ytick.color": THEME["muted"],
    "text.color": THEME["text"],
    "font.size": 10,
})

def _no_spine(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)

print("hugiml-core", __import__("hugiml").__version__)

# ---------------------------------------------------------------------------
# Part 1 – Data loading and quality review
# ---------------------------------------------------------------------------
print("\n--- Part 1: Data Loading ---")

df = pd.read_csv(DATA_FILE)
metadata = pd.read_csv(METADATA_FILE)
df[TARGET] = df[TARGET].astype(int)

n_rows, n_cols = df.shape
feature_count = n_cols - 1
num_feats  = df.drop(columns=[TARGET]).select_dtypes(include=[np.number]).columns.tolist()
cat_feats  = [c for c in df.drop(columns=[TARGET]).columns if c not in num_feats]
target_rate = df[TARGET].mean()

print(f"Rows: {n_rows:,}  |  Features: {feature_count}  "
      f"({len(num_feats)} numeric, {len(cat_feats)} categorical)")
print(f"Suspicious rate: {target_rate:.2%}  — realistic for deployed AML systems")
print(f"Missing cells:   {int(df.isna().sum().sum())}")
print(f"Duplicate rows:  {int(df.duplicated().sum())}")

# ---------------------------------------------------------------------------
# Part 2 – Splits (60 / 20 / 20)
# ---------------------------------------------------------------------------
print("\n--- Part 2: Splits ---")
print("""NOTE: AML models are typically validated on a forward-looking window
(new accounts / new typologies). Without explicit timestamps here we use a
stratified random split with a separate calibration holdout for isotonic
recalibration of probabilities used in SAR filing threshold analysis.""")

X = df.drop(columns=[TARGET])
y = df[TARGET]

clf_prep = HUGIMLClassifier(B=10, L=1, G=5e-4, topK=100)
X_enc, y_enc = clf_prep.prepareXy(X, y)

X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_enc, y_enc, test_size=0.40, stratify=y_enc, random_state=RANDOM_STATE
)
X_cal, X_te, y_cal, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE
)
print(f"Train: {len(X_tr):,}  |  Calibration: {len(X_cal):,}  |  Test: {len(X_te):,}")
print(f"Test positives: {y_te.sum()} ({y_te.mean():.2%})")

# ---------------------------------------------------------------------------
# Part 3 – feature_mode comparison
# ---------------------------------------------------------------------------
print("\n--- Part 3: feature_mode Comparison ---")

mode_results = {}
for mode in ["patterns_only", "original_plus_patterns"]:
    c = HUGIMLClassifier(B=10, L=1, G=5e-4, topK=100, feature_mode=mode)
    Xe, ye = c.prepareXy(X, y)
    Xtr2, Xtmp2, ytr2, ytmp2 = train_test_split(Xe, ye, test_size=0.40,
                                                   stratify=ye, random_state=RANDOM_STATE)
    Xcal2, Xte2, ycal2, yte2 = train_test_split(Xtmp2, ytmp2, test_size=0.50,
                                                   stratify=ytmp2, random_state=RANDOM_STATE)
    c.fit(Xtr2, ytr2)
    p2 = c.predict_proba(Xte2)[:, 1]
    mode_results[mode] = dict(
        clf=c, Xtr=Xtr2, Xcal=Xcal2, Xte=Xte2,
        ytr=ytr2, ycal=ycal2, yte=yte2, proba=p2,
        auc=roc_auc_score(yte2, p2),
        ap=average_precision_score(yte2, p2),
    )
    print(f"  {mode:26s}: {len(c.get_hug_features()):3d} patterns | "
          f"AUC={mode_results[mode]['auc']:.4f} | AP={mode_results[mode]['ap']:.4f}")

# Work with patterns_only for audit trail clarity
R = mode_results["patterns_only"]
clf   = R["clf"];    X_tr  = R["Xtr"];  X_cal = R["Xcal"]; X_te  = R["Xte"]
y_tr  = R["ytr"];   y_cal = R["ycal"]; y_te  = R["yte"];   y_score = R["proba"]
auc   = R["auc"];   ap    = R["ap"]

# Threshold — Youden index
fpr_c, tpr_c, thr_c = roc_curve(y_te, y_score)
prec_c, rec_c, thr_pr = precision_recall_curve(y_te, y_score)
youden_idx = int(np.argmax(tpr_c - fpr_c))
op_threshold = float(thr_c[youden_idx]) if np.isfinite(thr_c[youden_idx]) else 0.50

y_pred = (y_score >= op_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
prec_op, rec_op, f1_op, _ = precision_recall_fscore_support(
    y_te, y_pred, average="binary", zero_division=0
)

# ---------------------------------------------------------------------------
# Part 4 – Calibration
# ---------------------------------------------------------------------------
print("\n--- Part 4: Calibration ---")

cal_pre = evaluate_calibration(np.asarray(y_te), y_score)
print(f"[Pre-calibration]  ECE={cal_pre.ece:.4f}  MCE={cal_pre.mce:.4f}  "
      f"Brier={cal_pre.brier_score:.4f}")

# ---------------------------------------------------------------------------
# Part 5 – Interpretability metrics
# ---------------------------------------------------------------------------
print("\n--- Part 5: Interpretability Metrics ---")

interp = compute_all_metrics(clf, X_te)
print(f"n_patterns          : {interp.n_patterns}")
print(f"avg_pattern_length  : {interp.avg_pattern_length:.2f}")
print(f"coverage            : {interp.coverage:.2%}")
print(f"mean_active/sample  : {interp.mean_active_patterns:.2f}")
print(f"explanation_sparsity: {interp.explanation_sparsity:.4f}")

# ---------------------------------------------------------------------------
# Part 6 – Typology mapping and pattern inventory
# ---------------------------------------------------------------------------
print("\n--- Part 6: Typology Mapping ---")

importances = clf.feature_importances().copy()

def _typology(pattern_str: str) -> str:
    for kw, typo in TYPOLOGY_MAP.items():
        if kw in pattern_str:
            return typo
    return "Other"

importances["typology"] = importances["pattern"].apply(_typology)
importances["direction"] = np.where(importances["coefficient"] >= 0,
                                     "risk-increasing", "risk-decreasing")

top15 = importances.nlargest(15, "abs_coefficient")
print("Top 15 patterns with typology:")
print("=" * 90)
for _, row in top15.iterrows():
    arrow = "▲" if row["coefficient"] > 0 else "▼"
    print(f"  {arrow} {row['pattern']:42s} {row['coefficient']:+.4f}  "
          f"sup={row['support']:.1%}  [{row['typology']}]")
print("=" * 90)

# Count patterns per typology
typo_counts = importances.groupby("typology").size().sort_values(ascending=False)
print("\nPatterns per typology:")
print(typo_counts)

# ---------------------------------------------------------------------------
# Part 7 – Pattern pruning (BSA / demographic review)
# ---------------------------------------------------------------------------
print("\n--- Part 7: Pattern Pruning (Regulatory Review) ---")
print("""
BSA / FinCEN guidance discourages demographic profiling as a primary
suspicious activity indicator. Age-based patterns (customer_age) that
have weak individual signal and low support are reviewed and optionally
removed to reduce disparate-impact exposure.

NOTE: Geographic-risk patterns (country, high_risk_country) are RETAINED.
These correspond directly to FinCEN high-risk jurisdiction flags and are
a required consideration under OFAC / FATF guidance.
""")

editor = PatternEditor(clf, operator_name="aml-compliance-team")
pats_df = editor.list_patterns()
print(f"Patterns before review: {len(pats_df)}")

# Identify weak demographic (age) patterns — low abs coefficient + low support
age_pats = pats_df[
    pats_df["pattern"].str.contains("customer_age") &
    (pats_df["support"] < 0.11) &
    (pats_df["coefficient"].abs() < 0.20)
]
print(f"Weak customer_age patterns identified for removal: {len(age_pats)}")
for _, row in age_pats.iterrows():
    print(f"  idx={row['idx']}  {row['pattern']}  coef={row['coefficient']:+.4f}  sup={row['support']:.1%}")

if len(age_pats):
    editor.remove(
        age_pats["idx"].tolist(),
        reason=(
            "Low-signal demographic age patterns — weak evidence "
            "and low support; removed to reduce BSA disparate-impact exposure"
        ),
    )

# Refit + calibrate
editor.refit(X_tr, y_tr)
editor.calibrate(X_cal, y_cal, method="isotonic")
clf_pruned = editor.finalize()

proba_pruned = clf_pruned.predict_proba(X_te)[:, 1]
auc_pruned   = roc_auc_score(y_te, proba_pruned)
ap_pruned    = average_precision_score(y_te, proba_pruned)
cal_post     = evaluate_calibration(np.asarray(y_te), proba_pruned)

# New Youden threshold on pruned model
fpr_p, tpr_p, thr_pp = roc_curve(y_te, proba_pruned)
youden_p = int(np.argmax(tpr_p - fpr_p))
op_threshold_pruned = float(thr_pp[youden_p]) if np.isfinite(thr_pp[youden_p]) else 0.50

y_pred_pruned = (proba_pruned >= op_threshold_pruned).astype(int)
tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_te, y_pred_pruned).ravel()
prec_p, rec_p, f1_p, _ = precision_recall_fscore_support(
    y_te, y_pred_pruned, average="binary", zero_division=0
)

print(f"\nAfter pruning + isotonic recalibration:")
print(f"  Patterns : {len(clf_pruned.get_hug_features())}  (was {len(clf.get_hug_features())})")
print(f"  AUC      : {auc_pruned:.4f}  (was {auc:.4f})")
print(f"  AP       : {ap_pruned:.4f}  (was {ap:.4f})")
print(f"  ECE      : {cal_post.ece:.4f}  (was {cal_pre.ece:.4f})")
print(f"  Brier    : {cal_post.brier_score:.4f}  (was {cal_pre.brier_score:.4f})")

# Audit trail
audit_json = json.loads(editor.audit_report())
print(f"\nAudit: operator={audit_json['operator']}, "
      f"removed={audit_json['diff']['n_removed']}, "
      f"calibrated={audit_json['calibration']['applied']}")

# ---------------------------------------------------------------------------
# Part 8 – SAR filing threshold analysis
# ---------------------------------------------------------------------------
print("\n--- Part 8: SAR Filing Threshold Analysis ---")
print("""
AML models feed SAR (Suspicious Activity Report) filing queues.
Each flagged transaction requires investigator review time (~30 min/case).
Too-low a threshold floods the queue; too-high misses genuine SARs.
We model the operational cost at each threshold point.
""")

INVESTIGATOR_MINUTES_PER_CASE = 30

threshold_grid = np.linspace(0.05, 0.95, 37)
rows = []
for thr in threshold_grid:
    pred = (proba_pruned >= thr).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(np.asarray(y_te), pred, labels=[0,1]).ravel()
    n_flags    = int(pred.sum())
    workload_h = n_flags * INVESTIGATOR_MINUTES_PER_CASE / 60
    rows.append({
        "threshold":   thr,
        "n_flagged":   n_flags,
        "tp":          tp_t,
        "fp":          fp_t,
        "fn":          fn_t,
        "precision":   tp_t / max(tp_t + fp_t, 1),
        "recall":      tp_t / max(tp_t + fn_t, 1),
        "fpr":         fp_t / max(fp_t + tn_t, 1),
        "workload_h":  workload_h,
        "flag_rate":   pred.mean(),
    })

threshold_table = pd.DataFrame(rows)

# Find 80% recall threshold (regulatory minimum for high-risk programmes)
recall_80 = threshold_table[threshold_table["recall"] >= 0.80]
thr_80 = float(recall_80["threshold"].max()) if len(recall_80) else op_threshold_pruned

print(f"Youden-index threshold:         {op_threshold_pruned:.3f}")
print(f"80%-recall threshold:           {thr_80:.3f}")
row_80 = threshold_table[threshold_table["threshold"] <= thr_80].iloc[-1]
print(f"  Flags at 80% recall:          {int(row_80.n_flagged):,}")
print(f"  Precision at 80% recall:      {row_80.precision:.2%}")
print(f"  Estimated workload:           {row_80.workload_h:.1f} investigator-hours")
print(f"  FP per TP at 80% recall:      {row_80.fp / max(row_80.tp, 1):.1f}")

# ---------------------------------------------------------------------------
# Part 9 – Subgroup flag-rate audit
# ---------------------------------------------------------------------------
print("\n--- Part 9: Subgroup Flag-Rate Audit ---")

GROUP_COLS = ["country", "customer_type", "transaction_type", "high_risk_country"]
# Re-attach raw categorical columns to test rows
test_idx = X_te.index if hasattr(X_te, "index") else pd.Index(range(len(y_te)))
raw_test = X.loc[test_idx].copy()
audit_frame = raw_test.copy()
audit_frame["_actual"] = np.asarray(y_te).astype(int)
audit_frame["_score"]  = proba_pruned
audit_frame["_flag"]   = y_pred_pruned

sub_rows = []
for col in GROUP_COLS:
    if col not in audit_frame.columns:
        continue
    for level in audit_frame[col].value_counts().index[:12]:
        g = audit_frame[audit_frame[col].eq(level)]
        if len(g) < 20:
            continue
        yg = g["_actual"].to_numpy(); fg = g["_flag"].to_numpy()
        auc_g = roc_auc_score(yg, g["_score"]) if len(np.unique(yg)) == 2 else np.nan
        tn_g, fp_g, fn_g, tp_g = confusion_matrix(yg, fg, labels=[0,1]).ravel()
        sub_rows.append({
            "feature": col, "segment": str(level),
            "n": len(g),
            "base_rate": yg.mean(),
            "flag_rate": fg.mean(),
            "auc": auc_g,
            "precision": tp_g / max(tp_g + fp_g, 1),
            "recall": tp_g / max(tp_g + fn_g, 1),
            "fpr": fp_g / max(fp_g + tn_g, 1),
        })

subgroup_audit = pd.DataFrame(sub_rows)
print(subgroup_audit.sort_values(["feature","n"], ascending=[True,False]).round(4))

# ---------------------------------------------------------------------------
# Part 10 – Score distribution and review queue prioritisation
# ---------------------------------------------------------------------------
print("\n--- Part 10: Review Queue Prioritisation ---")

score_df = pd.DataFrame({"actual": np.asarray(y_te).astype(int), "score": proba_pruned})
score_df["decile"] = pd.qcut(score_df["score"].rank(method="first"), 10,
                              labels=range(1, 11)).astype(int)
decile = score_df.groupby("decile").agg(
    n=("actual","size"),
    event_rate=("actual","mean"),
    avg_score=("score","mean"),
    events=("actual","sum"),
).reset_index()
decile["event_capture_pct"] = decile["events"] / max(decile["events"].sum(), 1)
decile = decile.sort_values("decile", ascending=False)

base = float(score_df["actual"].mean())
top_d = float(decile[decile["decile"].eq(10)]["event_rate"].iloc[0])
print(f"Top decile event rate: {top_d:.2%}  ({top_d/base:.1f}× base rate)")
print(decile.round(4))

# ---------------------------------------------------------------------------
# Part 11 – Covariate drift detection (PSI)
# ---------------------------------------------------------------------------
print("\n--- Part 11: Covariate Drift Detection (PSI) ---")
print("Simulating a shift to higher-velocity / higher-amount transactions.")

def compute_psi(expected: pd.DataFrame, actual: pd.DataFrame, buckets=10) -> pd.DataFrame:
    rows = []
    common_cols = expected.select_dtypes(include=[np.number]).columns.intersection(actual.columns)
    for col in common_cols:
        edges = np.percentile(expected[col].dropna(), np.linspace(0,100,buckets+1))
        edges[0] = -np.inf; edges[-1] = np.inf
        ep = np.maximum(np.histogram(expected[col],bins=edges)[0]/len(expected),1e-6)
        ap = np.maximum(np.histogram(actual[col],  bins=edges)[0]/len(actual),  1e-6)
        psi = float(np.sum((ap-ep)*np.log(ap/ep)))
        rows.append({"feature":col,"psi":round(psi,4),
                     "status":"STABLE" if psi<0.10 else "WARNING" if psi<0.25 else "SHIFT"})
    return pd.DataFrame(rows).sort_values("psi",ascending=False)

rng_d = np.random.default_rng(77)
n_d = 500
X_raw_train = X[num_feats].iloc[:6000]
X_raw_drift = pd.DataFrame({
    "amount":                     rng_d.lognormal(9.5, 0.7, n_d),
    "velocity_24h":               rng_d.integers(10, 20, n_d),
    "account_age_days":           rng_d.uniform(0, 60, n_d),
    "structuring_score":          rng_d.uniform(0.6, 1.0, n_d),
    "avg_monthly_balance":        rng_d.lognormal(7.5, 1.2, n_d),
    "num_previous_transactions":  rng_d.integers(0, 5, n_d),
    "customer_age":               rng_d.integers(18, 35, n_d),
    "hour_of_day":                rng_d.integers(0, 5, n_d),
})
X_raw_drift = X_raw_drift[[c for c in X_raw_drift.columns if c in X_raw_train.columns]]

psi_df = compute_psi(X_raw_train, X_raw_drift)
print("PSI vs simulated high-velocity / structuring-heavy window:")
print(psi_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Part 12 – Visualisations
# ---------------------------------------------------------------------------
print("\n--- Part 12: Visualisations ---")

fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)
fig.patch.set_facecolor(THEME["bg"])

# 12a — ROC
ax = fig.add_subplot(gs[0, 0])
ax.plot(fpr_c, tpr_c, lw=2.5, color=THEME["accent"],  label=f"Baseline AUC={auc:.4f}")
ax.plot(fpr_p, tpr_p, lw=2.5, color=THEME["accent2"], ls="--",
        label=f"Pruned+Cal AUC={auc_pruned:.4f}")
ax.plot([0,1],[0,1], lw=1, color=THEME["muted"], ls=":")
ax.scatter([fpr_p[youden_p]], [tpr_p[youden_p]], s=60, color=THEME["accent2"], zorder=5)
ax.set_title("ROC Profile", fontweight="bold"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.legend(fontsize=8); ax.grid(color=THEME["grid"], alpha=0.7); _no_spine(ax)

# 12b — Precision-Recall
ax = fig.add_subplot(gs[0, 1])
ax.plot(rec_c, prec_c, lw=2.5, color=THEME["accent"],  label=f"Baseline AP={ap:.4f}")
prec_pp, rec_pp, _ = precision_recall_curve(y_te, proba_pruned)
ax.plot(rec_pp, prec_pp, lw=2.5, color=THEME["accent2"], ls="--",
        label=f"Pruned+Cal AP={ap_pruned:.4f}")
ax.axhline(float(y_te.mean()), lw=1, color=THEME["muted"], ls=":")
ax.set_title("Precision-Recall Profile", fontweight="bold")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.legend(fontsize=8); ax.grid(color=THEME["grid"], alpha=0.7); _no_spine(ax)

# 12c — Threshold trade-off
ax = fig.add_subplot(gs[0, 2])
ax.plot(threshold_table["threshold"], threshold_table["precision"],
        color=THEME["accent"],  marker=".", ms=4, label="Precision")
ax.plot(threshold_table["threshold"], threshold_table["recall"],
        color=THEME["accent2"], marker=".", ms=4, label="Recall")
ax.plot(threshold_table["threshold"], threshold_table["flag_rate"],
        color=THEME["accent3"], marker=".", ms=4, label="Flag rate")
ax.axvline(op_threshold_pruned, color=THEME["muted"], ls="--", lw=1.2, label="Youden")
ax.axvline(thr_80, color=THEME["accent2"], ls=":", lw=1.2, label="80% recall")
ax.set_title("SAR Threshold Trade-off", fontweight="bold")
ax.set_xlabel("Threshold"); ax.set_ylabel("Rate")
ax.legend(fontsize=7, ncol=2); ax.grid(color=THEME["grid"], alpha=0.7); _no_spine(ax)

# 12d — Top patterns (typology-coloured)
ax = fig.add_subplot(gs[1, :2])
top14 = top15.head(14).copy()
colors_14 = [THEME["accent2"] if v >= 0 else THEME["accent3"] for v in top14["coefficient"]]
ax.barh(top14["pattern"], top14["coefficient"], color=colors_14, alpha=0.85)
ax.invert_yaxis()
ax.set_title("Top Pattern Coefficients (green=risk-up, red=risk-down)", fontweight="bold")
ax.set_xlabel("Coefficient")
ax.grid(axis="x", color=THEME["grid"], alpha=0.7); _no_spine(ax)
ax.tick_params(axis="y", labelsize=8)

# 12e — Decile lift
ax = fig.add_subplot(gs[1, 2])
ax.bar(decile["decile"].astype(str), decile["event_rate"],
       color=THEME["accent2"], alpha=0.85)
ax.axhline(base, color=THEME["muted"], ls="--", lw=1.2, label=f"Base {base:.2%}")
ax.set_title("Observed Event Rate by Decile\n(10 = highest risk)", fontweight="bold")
ax.set_xlabel("Score decile"); ax.set_ylabel("Event rate")
ax.legend(fontsize=8); ax.grid(axis="y", color=THEME["grid"], alpha=0.7); _no_spine(ax)

# 12f — Calibration
ax = fig.add_subplot(gs[2, 0])
n_b = 8; bins = np.linspace(0, 1, n_b+1)
def _cal(ax, yt, prob, label, color, ls="-"):
    idx = np.clip(np.digitize(prob,bins)-1,0,n_b-1)
    cnt = np.bincount(idx,minlength=n_b)
    emp = np.where(cnt>0, np.bincount(idx,weights=yt.astype(float),minlength=n_b)/np.maximum(cnt,1), 0.0)
    prd = np.where(cnt>0, np.bincount(idx,weights=prob,minlength=n_b)/np.maximum(cnt,1), 0.0)
    ax.plot(prd[cnt>0], emp[cnt>0], marker="o", lw=2, ls=ls, color=color, label=label)
ax.plot([0,1],[0,1],"k--",lw=1.5,label="Perfect")
_cal(ax, np.asarray(y_te), y_score,      f"Pre-cal  (ECE={cal_pre.ece:.3f})",  THEME["accent"])
_cal(ax, np.asarray(y_te), proba_pruned, f"Post-cal (ECE={cal_post.ece:.3f})", THEME["accent2"], "--")
ax.set_title("Calibration Reliability", fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Empirical")
ax.legend(fontsize=8); ax.grid(color=THEME["grid"], alpha=0.7); _no_spine(ax)

# 12g — Subgroup flag rate
ax = fig.add_subplot(gs[2, 1])
if len(subgroup_audit):
    plot_sub = subgroup_audit.sort_values("flag_rate",ascending=False).head(12).copy()
    plot_sub["label"] = plot_sub["feature"] + "=" + plot_sub["segment"]
    ax.barh(plot_sub["label"], plot_sub["flag_rate"], color=THEME["accent"], alpha=0.8)
    ax.scatter(plot_sub["base_rate"], plot_sub["label"],
               color=THEME["accent2"], s=45, zorder=5, label="Base rate")
    ax.invert_yaxis()
    ax.set_title("Top Subgroup Flag Rates", fontweight="bold")
    ax.set_xlabel("Rate"); ax.legend(fontsize=8)
    ax.grid(axis="x", color=THEME["grid"], alpha=0.7); _no_spine(ax)
    ax.tick_params(axis="y", labelsize=7)

# 12h — PSI
ax = fig.add_subplot(gs[2, 2])
bar_colors = [THEME["accent3"] if p>0.25 else "#c0903b" if p>0.10 else THEME["accent2"]
              for p in psi_df["psi"]]
ax.barh(psi_df["feature"], psi_df["psi"], color=bar_colors, alpha=0.85)
ax.axvline(0.10, color="#c0903b", ls="--", lw=1.2, label="Warn")
ax.axvline(0.25, color=THEME["accent3"], ls="--", lw=1.2, label="Shift")
ax.invert_yaxis()
ax.set_title("Covariate Drift PSI\n(simulated high-velocity window)", fontweight="bold")
ax.set_xlabel("PSI"); ax.legend(fontsize=8)
ax.grid(axis="x", color=THEME["grid"], alpha=0.7); _no_spine(ax)
ax.tick_params(axis="y", labelsize=8)

fig_path = os.path.join(OUTDIR, "nb05_aml_report.png")
plt.savefig(fig_path, dpi=130, bbox_inches="tight", facecolor=THEME["bg"])
plt.close()
print(f"Visualisation saved to {fig_path}")

# ---------------------------------------------------------------------------
# Part 13 – Model governance
# ---------------------------------------------------------------------------
print("\n--- Part 13: Model Governance ---")

card = generate_model_card(
    clf_pruned,
    model_id="aml-txn-monitor-v1.0",
    intended_use=(
        "Real-time transaction scoring for AML/BSA suspicious activity detection. "
        "Output feeds SAR filing queue prioritisation."
    ),
    out_of_scope_use=(
        "Not validated for sanctions screening (OFAC SDN matching). "
        "Not for customer onboarding KYC risk scoring. "
        "Not for cross-border wire fraud detection without revalidation."
    ),
    training_data_description=(
        f"Synthetic AML dataset, {n_rows:,} transactions, {feature_count} features. "
        "5% positive (suspicious) rate — consistent with deployed system baselines. "
        "60/20/20 train/calibration/test split with isotonic recalibration."
    ),
    evaluation_data_description="Stratified 20% holdout; 100 suspicious transactions in test set.",
    performance_metrics={
        "AUC-ROC":            round(auc_pruned, 4),
        "Average_Precision":  round(ap_pruned, 4),
        "ECE":                round(cal_post.ece, 4),
        "BrierScore":         round(cal_post.brier_score, 4),
        "Recall_at_Youden":   round(float(rec_p), 4),
        "Precision_at_Youden": round(float(prec_p), 4),
    },
    limitations=[
        "Trained on synthetic data — production deployment requires validation "
        "on institution-specific transaction history with confirmed SAR dispositions.",
        "Weak customer_age patterns removed to reduce demographic profiling exposure; "
        "geographic-risk patterns retained per FinCEN / FATF guidance.",
        "Model does not replace human investigator judgment for SAR filing decisions.",
        "PSI monitoring required monthly — AML typologies evolve rapidly.",
    ],
    ethical_considerations=(
        "Low-signal age-based patterns removed per BSA disparate-impact guidance. "
        "Subgroup flag-rate audit required quarterly to detect differential treatment. "
        "All flagging decisions must be reviewable via pattern-level explanations. "
        "No model output constitutes a legal determination of criminal activity."
    ),
)

print(card.to_markdown())

# Export artifacts
prefix = os.path.join(OUTDIR, "nb05_aml")
card.save(f"{prefix}_model_card.json")
editor.save_audit_report(f"{prefix}_audit_trail.json")
importances.to_csv(f"{prefix}_pattern_inventory.csv", index=False)
threshold_table.to_csv(f"{prefix}_threshold_grid.csv", index=False)
subgroup_audit.to_csv(f"{prefix}_subgroup_audit.csv", index=False)
psi_df.to_csv(f"{prefix}_psi_report.csv", index=False)

# Run metadata
acc_pruned = accuracy_score(y_te, y_pred_pruned)
run_meta = pd.DataFrame([{
    "notebook": "nb05_aml",
    "model_id": "aml-txn-monitor-v1.0",
    "hugiml_version": __import__("hugiml").__version__,
    "dataset": "nb05_aml_data.csv",
    "n_samples": n_rows,
    "n_features": feature_count,
    "target_rate": round(target_rate, 4),
    "train_size": len(X_tr),
    "cal_size": len(X_cal),
    "test_size": len(X_te),
    "n_patterns_raw": len(clf.get_hug_features()),
    "n_patterns_pruned": len(clf_pruned.get_hug_features()),
    "auc_baseline": round(auc, 4),
    "auc_pruned": round(auc_pruned, 4),
    "ap_baseline": round(ap, 4),
    "ap_pruned": round(ap_pruned, 4),
    "ece_pre_cal": round(cal_pre.ece, 4),
    "ece_post_cal": round(cal_post.ece, 4),
    "brier_pre_cal": round(cal_pre.brier_score, 4),
    "brier_post_cal": round(cal_post.brier_score, 4),
    "youden_threshold": round(op_threshold_pruned, 4),
    "recall_80_threshold": round(thr_80, 4),
    "flags_at_80_recall": int(row_80.n_flagged),
    "workload_h_at_80_recall": round(row_80.workload_h, 1),
    "age_patterns_removed": len(age_pats),
    "calibration_method": "isotonic",
    "max_psi_feature": psi_df.iloc[0]["feature"],
    "max_psi_value": psi_df.iloc[0]["psi"],
}])
run_meta.to_csv(f"{prefix}_run_metadata.csv", index=False)

print(f"\nArtifacts saved:")
for suffix in ["_model_card.json","_audit_trail.json","_pattern_inventory.csv",
               "_threshold_grid.csv","_subgroup_audit.csv","_psi_report.csv","_run_metadata.csv"]:
    print(f"  nb05_aml{suffix}")

print("\n✓ nb05_aml.py completed successfully.")

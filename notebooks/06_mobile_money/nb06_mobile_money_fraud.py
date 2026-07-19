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
    precision_recall_fscore_support,
)
warnings.filterwarnings("ignore")

from hugiml import HUGIMLClassifier
from hugiml.calibration import evaluate_calibration
from hugiml.metrics import compute_all_metrics
from hugiml.pruning import PatternEditor
from hugiml.governance import generate_model_card

RANDOM_STATE   = 42
OUTDIR         = os.path.dirname(os.path.abspath(__file__))
DATA_FILE      = os.path.join(OUTDIR, "nb06_mobile_money_fraud_data.csv")
METADATA_FILE  = os.path.join(OUTDIR, "nb06_mobile_money_fraud_metadata.csv")
TARGET         = "isFraud"
MAX_MODEL_ROWS = 50_000          # Subsample to keep notebook reproducible

# Fraud typology labels for pattern annotation
TYPOLOGY_MAP = {
    "dest_to_amount_ratio":      "Balance Manipulation (dest unchanged / ratio)",
    "balance_change_dest":       "Balance Manipulation (dest unchanged / ratio)",
    "dest_unchanged":            "Balance Manipulation (dest unchanged / ratio)",
    "newbalanceOrig":            "Account Drainage (origin near-zero after txn)",
    "origin_near_zero":          "Account Drainage (origin near-zero after txn)",
    "origin_significant_drop":   "Account Drainage (origin near-zero after txn)",
    "balance_change_orig":       "Account Drainage (origin near-zero after txn)",
    "oldbalanceOrg":             "High-Value Origin Targeting",
    "amount":                    "High-Value Transaction",
    "high_value":                "High-Value Transaction",
    "medium_value":              "High-Value Transaction",
    "amount_to_balance_ratio":   "Account Impact Ratio",
    "type":                      "Transaction-Type Risk Profile",
    "is_cash_out":               "Transaction-Type Risk Profile",
    "is_transfer":               "Transaction-Type Risk Profile",
    "oldbalanceDest":            "Destination Account Pattern",
    "newbalanceDest":            "Destination Account Pattern",
}

THEME = dict(
    bg="#130c2a", panel="#211642", text="#f5f1ff",
    muted="#b9aee7", accent="#00d7ff", accent2="#ff4fd8",
    accent3="#ffd166", grid="#3b2a68",
)

def _no_spine(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)

plt.rcParams.update({
    "figure.facecolor": THEME["bg"], "axes.facecolor": THEME["panel"],
    "axes.edgecolor": THEME["grid"], "axes.labelcolor": THEME["text"],
    "xtick.color": THEME["muted"], "ytick.color": THEME["muted"],
    "text.color": THEME["text"], "font.size": 10,
})

print("hugiml-core", __import__("hugiml").__version__)

# ---------------------------------------------------------------------------
# Part 1 – Data loading and quality review
# ---------------------------------------------------------------------------
print("\n--- Part 1: Data Loading ---")

df        = pd.read_csv(DATA_FILE)
metadata  = pd.read_csv(METADATA_FILE)
df[TARGET] = df[TARGET].astype(int)

n_total        = len(df)
fraud_total    = int(df[TARGET].sum())
fraud_rate_all = df[TARGET].mean()
num_feats  = df.drop(columns=[TARGET]).select_dtypes(include=[np.number]).columns.tolist()
cat_feats  = [c for c in df.drop(columns=[TARGET]).columns if c not in num_feats]

print(f"Full dataset : {n_total:,} transactions  |  {len(num_feats)} numeric + {len(cat_feats)} categorical features")
print(f"Fraud count  : {fraud_total:,}  ({fraud_rate_all:.4%}) — extreme imbalance")
print(f"Missing cells: {int(df.isna().sum().sum())}  |  Duplicates: {int(df.duplicated().sum())}")
print("\nTransaction type breakdown:")
print(df["type"].value_counts().to_string())

# ---------------------------------------------------------------------------
# Part 2 – Subsampling and split strategy
# ---------------------------------------------------------------------------
print("\n--- Part 2: Subsampling and Splits ---")
print(f"""
NOTE ON CLASS IMBALANCE
Fraud rate is {fraud_rate_all:.4%} — one of the most extreme imbalance scenarios
in financial ML. Key operational choices:
  1. Subsample to {MAX_MODEL_ROWS:,} rows (preserving fraud rate) to keep training
     time practical while maintaining a representative model.
  2. Use Average Precision (AP) as the primary metric — ROC-AUC is misleading
     at very low base rates because the large number of TN inflates performance.
  3. Reserve a calibration holdout for isotonic recalibration. NOTE: with only
     ~13 positives in the calibration set, isotonic calibration has high variance;
     Platt scaling may generalise better in production.
  4. All threshold analysis uses Precision-Recall, not Youden index.
""")

if len(df) > MAX_MODEL_ROWS:
    model_df, _ = train_test_split(
        df, train_size=MAX_MODEL_ROWS, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    model_df = model_df.reset_index(drop=True)
else:
    model_df = df.copy().reset_index(drop=True)

X = model_df.drop(columns=[TARGET])
y = model_df[TARGET]
fraud_model = int(y.sum())
print(f"Modeling pop : {len(X):,}  |  Fraud: {fraud_model} ({y.mean():.4%})")

# 60 / 20 / 20  train / calibration / test
clf_prep = HUGIMLClassifier(B=10, L=1, G=1e-4, topK=100)
X_enc, y_enc = clf_prep.prepareXy(X, y)

X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_enc, y_enc, test_size=0.40, stratify=y_enc, random_state=RANDOM_STATE
)
X_cal, X_te, y_cal, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE
)
print(f"Train: {len(X_tr):,} (fraud: {y_tr.sum()})  |  Cal: {len(X_cal):,} (fraud: {y_cal.sum()})  |  Test: {len(X_te):,} (fraud: {y_te.sum()})")

# ---------------------------------------------------------------------------
# Part 3 – feature_mode comparison
# ---------------------------------------------------------------------------
print("\n--- Part 3: feature_mode Comparison ---")

mode_results = {}
for mode in ["patterns_only", "original_plus_patterns"]:
    c = HUGIMLClassifier(B=10, L=1, G=1e-4, topK=100, feature_mode=mode)
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

# Use patterns_only for clarity
R    = mode_results["patterns_only"]
clf  = R["clf"];    X_tr  = R["Xtr"];  X_cal = R["Xcal"]; X_te  = R["Xte"]
y_tr = R["ytr"];   y_cal = R["ycal"]; y_te  = R["yte"];  y_score = R["proba"]
auc  = R["auc"];   ap    = R["ap"]

# Thresholds from PR curve (primary) and ROC
fpr_c, tpr_c, thr_roc = roc_curve(y_te, y_score)
prec_c, rec_c, thr_pr = precision_recall_curve(y_te, y_score)

# F1-maximising threshold
f1_scores = np.where((prec_c + rec_c) > 0,
                     2 * prec_c * rec_c / (prec_c + rec_c), 0)
f1_idx        = int(np.argmax(f1_scores))
op_threshold  = float(thr_pr[min(f1_idx, len(thr_pr)-1)])

y_pred = (y_score >= op_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
prec_op, rec_op, f1_op, _ = precision_recall_fscore_support(
    y_te, y_pred, average="binary", zero_division=0
)
print(f"\nBaseline @ F1-max threshold {op_threshold:.3f}:")
print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}  F1={f1_op:.3f}  precision={prec_op:.2%}  recall={rec_op:.2%}")

# ---------------------------------------------------------------------------
# Part 4 – Calibration (pre-recalibration baseline)
# ---------------------------------------------------------------------------
print("\n--- Part 4: Calibration ---")

cal_pre = evaluate_calibration(np.asarray(y_te), y_score)
print(f"[Pre-calibration]  ECE={cal_pre.ece:.4f}  MCE={cal_pre.mce:.4f}  Brier={cal_pre.brier_score:.6f}")
print("NOTE: With only ~13 positives in test, bin-level calibration metrics have high variance.")

# ---------------------------------------------------------------------------
# Part 5 – Interpretability metrics
# ---------------------------------------------------------------------------
print("\n--- Part 5: Interpretability Metrics ---")

interp = compute_all_metrics(clf, X_te)
print(f"n_patterns           : {interp.n_patterns}")
print(f"avg_pattern_length   : {interp.avg_pattern_length:.2f}")
print(f"coverage             : {interp.coverage:.2%}")
print(f"mean_active/sample   : {interp.mean_active_patterns:.2f}")
print(f"explanation_sparsity : {interp.explanation_sparsity:.4f}")

# ---------------------------------------------------------------------------
# Part 6 – Fraud typology mapping
# ---------------------------------------------------------------------------
print("\n--- Part 6: Fraud Typology Mapping ---")

importances = clf.feature_importances().copy()

def _typology(pat: str) -> str:
    for kw, t in TYPOLOGY_MAP.items():
        if kw in pat:
            return t
    return "Other"

importances["typology"]  = importances["pattern"].apply(_typology)
importances["direction"] = np.where(importances["coefficient"] >= 0,
                                     "fraud-signal", "legitimacy-signal")

top15 = importances.nlargest(15, "abs_coefficient")
print("Top 15 patterns with fraud typology labels:")
print("=" * 95)
for _, row in top15.iterrows():
    arrow = "▲" if row["coefficient"] > 0 else "▼"
    print(f"  {arrow} {row['pattern']:44s} {row['coefficient']:+.4f}  "
          f"sup={row['support']:.1%}  [{row['typology']}]")
print("=" * 95)

print("\nPatterns per typology:")
print(importances.groupby("typology").size().sort_values(ascending=False))

# ---------------------------------------------------------------------------
# Part 7 – Pattern review and pruning
# ---------------------------------------------------------------------------
print("\n--- Part 7: Pattern Review and Pruning ---")
print("""
FFIEC / Reg E context: Mobile money fraud models must be explainable for
dispute resolution (Reg E §1005.11). Each flagged transaction needs a
specific, auditable reason — pattern-level explanations satisfy this.

Review criteria:
  • RETAIN high-magnitude balance-manipulation and account-drainage patterns
    (these map to specific fraud typologies and are defensible in disputes).
  • REVIEW transaction-type patterns: type=PAYMENT and type=CASH_IN are strong
    legitimacy signals — their removal would inflate false positives.
  • REMOVE very low-support noise patterns (<2% support) that are unstable
    across production windows.
  • No protected demographic attributes exist in this dataset.
""")

editor = PatternEditor(clf, operator_name="fraud-ops-review")
pats_df = editor.list_patterns()
print(f"Patterns before review: {len(pats_df)}")

# Remove low-support noise patterns
editor.remove_low_support(
    min_support=0.08,
    reason="Low-support patterns (<2%) are unstable across payment windows; "
           "remove to improve model reliability in production",
)
pats_after_noise = len(editor.list_patterns())
print(f"Patterns after low-support removal: {pats_after_noise}")

# Refit + calibrate
editor.refit(X_tr, y_tr)
editor.calibrate(X_cal, y_cal, method="isotonic")
clf_pruned = editor.finalize()

proba_pruned  = clf_pruned.predict_proba(X_te)[:, 1]
auc_pruned    = roc_auc_score(y_te, proba_pruned)
ap_pruned     = average_precision_score(y_te, proba_pruned)
cal_post      = evaluate_calibration(np.asarray(y_te), proba_pruned)

prec_pp, rec_pp, thr_pp2 = precision_recall_curve(y_te, proba_pruned)
f1_pp = np.where((prec_pp + rec_pp) > 0, 2*prec_pp*rec_pp/(prec_pp+rec_pp), 0)
f1p_idx      = int(np.argmax(f1_pp))
op_thr_p     = float(thr_pp2[min(f1p_idx, len(thr_pp2)-1)])

y_pred_p = (proba_pruned >= op_thr_p).astype(int)
tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_te, y_pred_p).ravel()
prec_p, rec_p, f1_p, _ = precision_recall_fscore_support(
    y_te, y_pred_p, average="binary", zero_division=0
)

print(f"\nAfter pruning + isotonic recalibration:")
print(f"  Patterns  : {len(clf_pruned.get_hug_features())} (was {len(clf.get_hug_features())})")
print(f"  AUC-ROC   : {auc_pruned:.4f}  (was {auc:.4f})")
print(f"  Avg Prec  : {ap_pruned:.4f}  (was {ap:.4f})")
print(f"  ECE       : {cal_post.ece:.4f}  (was {cal_pre.ece:.4f})")
print(f"  F1 @ op   : {f1_p:.3f}  (was {f1_op:.3f})")
print(f"  TP/FP/FN  : {tp_p}/{fp_p}/{fn_p}")

audit_js = json.loads(editor.audit_report())
print(f"\nAudit: removed={audit_js['diff']['n_removed']}, "
      f"calibrated={audit_js['calibration']['applied']}")

# ---------------------------------------------------------------------------
# Part 8 – Threshold analysis (Reg E dispute framing)
# ---------------------------------------------------------------------------
print("\n--- Part 8: Threshold Analysis ---")
print("""
Reg E requires investigation of all reported fraud disputes within 10 days.
Threshold selection must balance:
  • FALSE POSITIVES  → declined legitimate transactions (customer friction, Reg E §1005.6(b))
  • FALSE NEGATIVES  → undetected fraud (financial loss + Reg E liability)

We sweep the full precision-recall curve and identify operating points.
""")

thresh_rows = []
for thr in np.linspace(0.01, 0.99, 49):
    pred = (proba_pruned >= thr).astype(int)
    if pred.sum() == 0:
        continue
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(np.asarray(y_te), pred, labels=[0,1]).ravel()
    prec_t, rec_t, f1_t, _ = precision_recall_fscore_support(
        y_te, pred, average="binary", zero_division=0
    )
    thresh_rows.append({
        "threshold":  thr,
        "n_flagged":  int(pred.sum()),
        "tp": tp_t, "fp": fp_t, "fn": fn_t,
        "precision":  prec_t,
        "recall":     rec_t,
        "f1":         f1_t,
        "fpr":        fp_t / max(fp_t + tn_t, 1),
        "flag_rate":  pred.mean(),
    })

threshold_table = pd.DataFrame(thresh_rows)

# Print key operating points
for target_recall in [0.70, 0.80, 0.90]:
    rows_ge = threshold_table[threshold_table["recall"] >= target_recall]
    if len(rows_ge):
        row = rows_ge.iloc[-1]
        print(f"  {int(target_recall*100)}% recall → threshold={row.threshold:.3f}  "
              f"precision={row.precision:.2%}  flags={int(row.n_flagged)}  "
              f"FP/TP={row.fp/max(row.tp,1):.1f}")

# ---------------------------------------------------------------------------
# Part 9 – Subgroup flag-rate audit
# ---------------------------------------------------------------------------
print("\n--- Part 9: Subgroup Audit ---")

GROUP_COLS = ["type", "is_cash_out", "is_transfer", "dest_unchanged",
              "origin_near_zero", "high_value"]
test_idx = X_te.index if hasattr(X_te, "index") else pd.Index(range(len(y_te)))
raw_test = X.loc[test_idx].copy()
af = raw_test.copy()
af["_actual"] = np.asarray(y_te).astype(int)
af["_score"]  = proba_pruned
af["_flag"]   = y_pred_p

sub_rows = []
for col in GROUP_COLS:
    if col not in af.columns:
        continue
    for level in af[col].value_counts().index[:10]:
        g = af[af[col].eq(level)]
        if len(g) < 20:
            continue
        yg = g["_actual"].to_numpy(); fg = g["_flag"].to_numpy()
        auc_g = roc_auc_score(yg, g["_score"]) if len(np.unique(yg)) == 2 else np.nan
        tn_g, fp_g, fn_g, tp_g = confusion_matrix(yg, fg, labels=[0,1]).ravel()
        sub_rows.append({
            "feature": col, "segment": str(level), "n": len(g),
            "base_rate": yg.mean(), "flag_rate": fg.mean(), "auc": auc_g,
            "precision": tp_g / max(tp_g + fp_g, 1),
            "recall": tp_g / max(tp_g + fn_g, 1),
            "fpr": fp_g / max(fp_g + tn_g, 1),
        })

subgroup_audit = pd.DataFrame(sub_rows)
print(subgroup_audit.sort_values(["feature", "n"], ascending=[True, False]).round(4))

# ---------------------------------------------------------------------------
# Part 10 – Score decile lift
# ---------------------------------------------------------------------------
print("\n--- Part 10: Score Decile Lift ---")

score_df = pd.DataFrame({"actual": np.asarray(y_te).astype(int), "score": proba_pruned})
score_df["decile"] = pd.qcut(score_df["score"].rank(method="first"), 10,
                              labels=range(1, 11)).astype(int)
decile = score_df.groupby("decile").agg(
    n=("actual","size"),
    event_rate=("actual","mean"),
    avg_score=("score","mean"),
    events=("actual","sum"),
).reset_index()
decile["capture_pct"] = decile["events"] / max(decile["events"].sum(), 1)
decile = decile.sort_values("decile", ascending=False)

base_rate = float(score_df["actual"].mean())
top_rate  = float(decile[decile["decile"].eq(10)]["event_rate"].iloc[0])
lift_top  = top_rate / max(base_rate, 1e-12)
print(f"Top decile: {top_rate:.2%} event rate  ({lift_top:.1f}× base rate of {base_rate:.4%})")
print(decile.round(4))

# ---------------------------------------------------------------------------
# Part 11 – Covariate drift detection (PSI)
# ---------------------------------------------------------------------------
print("\n--- Part 11: Covariate Drift Detection (PSI) ---")
print("Simulating a shift to higher-value transactions with unusual balance patterns.")

def compute_psi(expected: pd.DataFrame, actual: pd.DataFrame, buckets: int = 10) -> pd.DataFrame:
    rows = []
    common = expected.select_dtypes(include=[np.number]).columns.intersection(actual.columns)
    for col in common:
        edges = np.percentile(expected[col].dropna(), np.linspace(0, 100, buckets+1))
        edges[0] = -np.inf; edges[-1] = np.inf
        ep = np.maximum(np.histogram(expected[col], bins=edges)[0] / len(expected), 1e-6)
        ap_ = np.maximum(np.histogram(actual[col],  bins=edges)[0] / len(actual),   1e-6)
        psi = float(np.sum((ap_ - ep) * np.log(ap_ / ep)))
        rows.append({"feature": col, "psi": round(psi, 4),
                     "status": "STABLE" if psi < 0.10 else "WARNING" if psi < 0.25 else "SHIFT"})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)

rng_d = np.random.default_rng(55)
n_d   = 2000
X_raw_train = X[num_feats].iloc[:30000]
# Simulated shift: elevated amounts, unusual balance changes (potential fraud wave)
X_raw_drift = pd.DataFrame({
    "amount":                  rng_d.lognormal(12.5, 1.5, n_d),
    "oldbalanceOrg":           rng_d.lognormal(12.0, 1.2, n_d),
    "oldbalanceDest":          rng_d.uniform(0, 100, n_d),         # near-zero dest
    "newbalanceOrig":          rng_d.uniform(0, 200, n_d),          # near-zero after txn
    "newbalanceDest":          rng_d.uniform(0, 100, n_d),
    "balance_change_orig":     -rng_d.lognormal(12.0, 1.2, n_d),
    "balance_change_dest":     rng_d.uniform(-50, 50, n_d),         # near-zero change
    "amount_to_balance_ratio": rng_d.uniform(0.8, 2.0, n_d),
    "dest_to_amount_ratio":    rng_d.uniform(-0.1, 0.1, n_d),
})
X_raw_drift = X_raw_drift[[c for c in X_raw_drift.columns
                            if c in X_raw_train.columns]]

psi_df = compute_psi(X_raw_train, X_raw_drift)
print("PSI vs simulated high-value / account-drainage shift window:")
print(psi_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Part 12 – Visualisations
# ---------------------------------------------------------------------------
print("\n--- Part 12: Visualisations ---")

fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.40)
fig.patch.set_facecolor(THEME["bg"])

# 12a — PR curve (primary metric for extreme imbalance)
ax = fig.add_subplot(gs[0, 0])
ax.plot(rec_c, prec_c, lw=2.5, color=THEME["accent"],
        label=f"Baseline AP={ap:.4f}")
ax.plot(rec_pp, prec_pp, lw=2.5, color=THEME["accent2"], ls="--",
        label=f"Pruned+Cal AP={ap_pruned:.4f}")
ax.axhline(float(y_te.mean()), lw=1, color=THEME["muted"], ls=":", label="Base rate")
ax.set_title("Precision-Recall (primary metric\nfor extreme class imbalance)", fontweight="bold")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.legend(fontsize=7); ax.grid(color=THEME["grid"], alpha=0.6); _no_spine(ax)

# 12b — ROC
ax = fig.add_subplot(gs[0, 1])
ax.plot(fpr_c, tpr_c, lw=2.5, color=THEME["accent"], label=f"Baseline AUC={auc:.4f}")
fpr_pp, tpr_pp, _ = roc_curve(y_te, proba_pruned)
ax.plot(fpr_pp, tpr_pp, lw=2.5, color=THEME["accent2"], ls="--",
        label=f"Pruned+Cal AUC={auc_pruned:.4f}")
ax.plot([0,1],[0,1], lw=1, color=THEME["muted"], ls=":")
ax.set_title("ROC Curve\n(note: inflated by extreme imbalance)", fontweight="bold")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.legend(fontsize=7); ax.grid(color=THEME["grid"], alpha=0.6); _no_spine(ax)

# 12c — Top patterns
ax = fig.add_subplot(gs[0, 2])
top12 = importances.nlargest(12, "abs_coefficient")
bar_c = [THEME["accent2"] if v > 0 else THEME["accent3"] for v in top12["coefficient"]]
ax.barh(range(len(top12)), top12["coefficient"], color=bar_c, alpha=0.85)
ax.set_yticks(range(len(top12)))
ax.set_yticklabels(top12["pattern"], fontsize=7)
ax.invert_yaxis()
ax.set_title("Top Pattern Coefficients\n(cyan=fraud signal, pink=legit signal)", fontweight="bold")
ax.set_xlabel("Coefficient")
ax.grid(axis="x", color=THEME["grid"], alpha=0.6); _no_spine(ax)

# 12d — Threshold trade-off
ax = fig.add_subplot(gs[1, 0])
ax.plot(threshold_table["threshold"], threshold_table["precision"],
        color=THEME["accent"], marker=".", ms=3, label="Precision")
ax.plot(threshold_table["threshold"], threshold_table["recall"],
        color=THEME["accent2"], marker=".", ms=3, label="Recall")
ax.plot(threshold_table["threshold"], threshold_table["f1"],
        color=THEME["accent3"], marker=".", ms=3, label="F1")
ax.axvline(op_thr_p, color=THEME["muted"], ls="--", lw=1.2, label="F1-max")
ax.set_title("Threshold Trade-off Grid\n(Reg E dispute framing)", fontweight="bold")
ax.set_xlabel("Threshold"); ax.set_ylabel("Rate")
ax.legend(fontsize=7, ncol=2); ax.grid(color=THEME["grid"], alpha=0.6); _no_spine(ax)

# 12e — Decile lift
ax = fig.add_subplot(gs[1, 1])
ax.bar(decile["decile"].astype(str), decile["event_rate"], color=THEME["accent2"], alpha=0.85)
ax.axhline(base_rate, color=THEME["muted"], ls="--", lw=1.2, label=f"Base {base_rate:.4%}")
ax.set_title(f"Event Rate by Decile\n(Top decile: {lift_top:.0f}× lift)", fontweight="bold")
ax.set_xlabel("Score decile (10=highest)"); ax.set_ylabel("Event rate")
ax.legend(fontsize=7); ax.grid(axis="y", color=THEME["grid"], alpha=0.6); _no_spine(ax)

# 12f — Calibration before/after
ax = fig.add_subplot(gs[1, 2])
n_b = 8; bins = np.linspace(0, 1, n_b + 1)
def _cal(ax, yt, prob, label, color, ls="-"):
    idx = np.clip(np.digitize(prob, bins)-1, 0, n_b-1)
    cnt = np.bincount(idx, minlength=n_b)
    emp = np.where(cnt>0, np.bincount(idx, weights=yt.astype(float), minlength=n_b)/np.maximum(cnt,1), 0.0)
    prd = np.where(cnt>0, np.bincount(idx, weights=prob, minlength=n_b)/np.maximum(cnt,1), 0.0)
    ax.plot(prd[cnt>0], emp[cnt>0], marker="o", lw=2, ls=ls, color=color, label=label)
ax.plot([0,1],[0,1],"w--",lw=1.5,label="Perfect")
_cal(ax, np.asarray(y_te), y_score,      f"Pre-cal  (ECE={cal_pre.ece:.3f})",  THEME["accent"])
_cal(ax, np.asarray(y_te), proba_pruned, f"Post-cal (ECE={cal_post.ece:.3f})", THEME["accent2"], "--")
ax.set_title("Calibration\n(high variance at low n+ is expected)", fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Empirical")
ax.legend(fontsize=7); ax.grid(color=THEME["grid"], alpha=0.6); _no_spine(ax)

# 12g — Subgroup flag rates
ax = fig.add_subplot(gs[2, 0:2])
if len(subgroup_audit):
    plot_sub = subgroup_audit.sort_values("flag_rate", ascending=False).head(12).copy()
    plot_sub["label"] = plot_sub["feature"] + "=" + plot_sub["segment"]
    ax.barh(plot_sub["label"], plot_sub["flag_rate"], color=THEME["accent"], alpha=0.85)
    ax.scatter(plot_sub["base_rate"], plot_sub["label"],
               color=THEME["accent3"], s=45, zorder=5, label="Base rate")
    ax.invert_yaxis()
    ax.set_title("Top Subgroup Flag Rates", fontweight="bold")
    ax.set_xlabel("Rate"); ax.legend(fontsize=7)
    ax.grid(axis="x", color=THEME["grid"], alpha=0.6); _no_spine(ax)
    ax.tick_params(axis="y", labelsize=8)

# 12h — PSI
ax = fig.add_subplot(gs[2, 2])
bar_c_psi = [THEME["accent3"] if p > 0.25 else "#c0903b" if p > 0.10 else THEME["accent2"]
             for p in psi_df["psi"]]
ax.barh(psi_df["feature"].str[:20], psi_df["psi"], color=bar_c_psi, alpha=0.85)
ax.axvline(0.10, color="#c0903b", ls="--", lw=1.2, label="Warn")
ax.axvline(0.25, color=THEME["accent3"], ls="--", lw=1.2, label="Shift")
ax.invert_yaxis()
ax.set_title("Covariate Drift PSI\n(simulated fraud-wave window)", fontweight="bold")
ax.set_xlabel("PSI"); ax.legend(fontsize=7)
ax.grid(axis="x", color=THEME["grid"], alpha=0.6); _no_spine(ax)
ax.tick_params(axis="y", labelsize=7)

fig_path = os.path.join(OUTDIR, "nb06_mobile_money_fraud_report.png")
plt.savefig(fig_path, dpi=130, bbox_inches="tight", facecolor=THEME["bg"])
plt.close()
print(f"Visualisation saved to {fig_path}")

# ---------------------------------------------------------------------------
# Part 13 – Model governance
# ---------------------------------------------------------------------------
print("\n--- Part 13: Model Governance ---")

acc_pruned = (tp_p + tn_p) / (tp_p + fp_p + tn_p + fn_p)

card = generate_model_card(
    clf_pruned,
    model_id="mobile-money-fraud-v1.0",
    intended_use=(
        "Real-time transaction scoring for mobile-money fraud detection. "
        "Output feeds real-time block/flag decisions and Reg E dispute triage."
    ),
    out_of_scope_use=(
        "Not validated for card-present fraud. "
        "Not for account-takeover detection without revalidation on authentication features. "
        "Not a replacement for Reg E error-resolution investigation."
    ),
    training_data_description=(
        f"PaySim-style synthetic mobile-money dataset, {len(model_df):,} transactions "
        f"({fraud_model} fraud, {y.mean():.4%} rate). 60/20/20 split."
    ),
    evaluation_data_description=f"Stratified 20% holdout; {int(y_te.sum())} fraud transactions.",
    performance_metrics={
        "AUC-ROC":           round(auc_pruned, 4),
        "AvgPrecision":      round(ap_pruned, 4),
        "ECE":               round(cal_post.ece, 4),
        "BrierScore":        round(cal_post.brier_score, 6),
        "F1_at_op":          round(float(f1_p), 4),
        "Precision_at_op":   round(float(prec_p), 4),
        "Recall_at_op":      round(float(rec_p), 4),
    },
    limitations=[
        "Trained on synthetic data — production deployment requires validation on "
        "real transaction history with confirmed fraud labels.",
        "Extreme class imbalance (0.13%) means calibration estimates have high variance "
        "with small test positives; Platt scaling may generalise better than isotonic.",
        "Model does not capture account-takeover pre-cursors (authentication signals "
        "not available in this dataset).",
        "Monthly PSI monitoring required — mobile-money fraud patterns evolve rapidly.",
    ],
    ethical_considerations=(
        "No protected demographic attributes in this dataset. "
        "Reg E requires specific, auditable reasons for declined/flagged transactions — "
        "pattern-level explanations satisfy this requirement. "
        "False positive rate monitoring by customer segment required quarterly. "
        "Dispute investigation process must remain human-led per Reg E §1005.11."
    ),
)

print(card.to_markdown())

# Export artifacts
prefix = os.path.join(OUTDIR, "nb06_mobile_money_fraud")
card.save(f"{prefix}_model_card.json")
editor.save_audit_report(f"{prefix}_audit_trail.json")
importances.to_csv(f"{prefix}_pattern_inventory.csv", index=False)
threshold_table.to_csv(f"{prefix}_threshold_grid.csv", index=False)
subgroup_audit.to_csv(f"{prefix}_subgroup_audit.csv", index=False)
psi_df.to_csv(f"{prefix}_psi_report.csv", index=False)

# Run metadata
run_meta = pd.DataFrame([{
    "notebook":            "nb06_mobile_money_fraud",
    "model_id":            "mobile-money-fraud-v1.0",
    "hugiml_version":      __import__("hugiml").__version__,
    "dataset":             "nb06_mobile_money_fraud_data.csv",
    "n_total":             n_total,
    "fraud_rate_total":    round(fraud_rate_all, 6),
    "n_model_pop":         len(model_df),
    "fraud_model_pop":     fraud_model,
    "train_size":          len(X_tr),
    "cal_size":            len(X_cal),
    "test_size":           len(X_te),
    "test_fraud":          int(y_te.sum()),
    "n_patterns_raw":      len(clf.get_hug_features()),
    "n_patterns_pruned":   len(clf_pruned.get_hug_features()),
    "auc_baseline":        round(auc, 4),
    "auc_pruned":          round(auc_pruned, 4),
    "ap_baseline":         round(ap, 4),
    "ap_pruned":           round(ap_pruned, 4),
    "ece_pre_cal":         round(cal_pre.ece, 6),
    "ece_post_cal":        round(cal_post.ece, 6),
    "f1_pruned":           round(float(f1_p), 4),
    "precision_pruned":    round(float(prec_p), 4),
    "recall_pruned":       round(float(rec_p), 4),
    "calibration_method":  "isotonic",
    "max_psi_feature":     psi_df.iloc[0]["feature"],
    "max_psi_value":       psi_df.iloc[0]["psi"],
}])
run_meta.to_csv(f"{prefix}_run_metadata.csv", index=False)

print("\n✓ nb06_mobile_money_fraud.py completed successfully.")

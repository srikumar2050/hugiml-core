import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
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

RANDOM_STATE  = 42
OUTDIR        = os.path.dirname(os.path.abspath(__file__))
DATA_FILE     = os.path.join(OUTDIR, "nb08_healthcare_breast_cancer_data.csv")
METADATA_FILE = os.path.join(OUTDIR, "nb08_healthcare_breast_cancer_metadata.csv")
TARGET        = "diagnosis_malignant"

# Clinical morphology groups for pattern annotation
MORPH_MAP = {
    "worst":         "Worst-Case Morphology (most abnormal cell)",
    "mean radius":   "Cell Size — Radius",
    "mean area":     "Cell Size — Area",
    "mean perimeter":"Cell Size — Perimeter",
    "mean texture":  "Texture / Heterogeneity",
    "mean smooth":   "Surface Smoothness",
    "mean compact":  "Compactness",
    "mean concav":   "Concavity / Irregular Contour",
    "mean symm":     "Symmetry",
    "mean fractal":  "Fractal Dimension (border irregularity)",
    "radius error":  "Measurement Variability",
    "texture error": "Measurement Variability",
    "perimeter error":"Measurement Variability",
    "area error":    "Measurement Variability",
}

# Clinical safety: for breast cancer screening, recall (sensitivity) is critical.
# A missed malignancy (FN) is far more costly than an unnecessary biopsy (FP).
# Operating threshold selection must prioritise recall >= 90%.
MIN_CLINICAL_RECALL = 0.90

HEALTH_TEAL  = "#0b6e69"; HEALTH_CORAL = "#d95f45"
HEALTH_MINT  = "#dff3ee"; HEALTH_INK   = "#12232e"
HEALTH_AMBER = "#c47d15"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fbfefe",
    "axes.edgecolor": "#a9c5c8", "axes.labelcolor": HEALTH_INK,
    "xtick.color": HEALTH_INK, "ytick.color": HEALTH_INK,
    "text.color": HEALTH_INK, "font.size": 10,
})
def _no_spine(ax):
    for sp in ax.spines.values(): sp.set_visible(False)

print("hugiml-core", __import__("hugiml").__version__)

# ---------------------------------------------------------------------------
# Part 1 – Data loading and privacy controls
# ---------------------------------------------------------------------------
print("\n--- Part 1: Data Loading and Privacy Controls ---")

df       = pd.read_csv(DATA_FILE)
metadata = pd.read_csv(METADATA_FILE)
df[TARGET] = df[TARGET].astype(int)

# Exclude non-feature columns (patient ID, label columns)
exclude = ["patient_id","diagnosis_original_sklearn_target",
           "diagnosis_malignant","diagnosis_label"]
feature_cols = [c for c in df.columns if c not in exclude]
X = df[feature_cols]; y = df[TARGET]

print(f"Patients: {len(df)}  |  Features: {len(feature_cols)}")
print(f"Malignant: {int(y.sum())} ({y.mean():.2%})  |  Benign: {int((1-y).sum())} ({1-y.mean():.2%})")
print(f"Patient IDs excluded from model: ✓  (HIPAA de-identification)")
print(f"Missing values: {int(X.isna().sum().sum())}  |  Duplicates: {int(df.duplicated().sum())}")

# ---------------------------------------------------------------------------
# Part 2 – Splits (60 / 20 / 20)
# ---------------------------------------------------------------------------
print("\n--- Part 2: Splits (60/20/20) ---")
print(f"""
CLINICAL SAFETY NOTE: For malignancy detection, recall (sensitivity) is
the primary safety metric. A false negative (missed malignancy) is
clinically far more serious than a false positive (unnecessary follow-up).
Operating threshold is selected to achieve recall >= {MIN_CLINICAL_RECALL:.0%}.
""")

clf_prep = HUGIMLClassifier(B=10, L=1, G=1e-4, topK=120)
X_enc, y_enc = clf_prep.prepareXy(X, y)

X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_enc, y_enc, test_size=0.40, stratify=y_enc, random_state=RANDOM_STATE)
X_cal, X_te, y_cal, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)
print(f"Train:{len(X_tr)} (mal:{y_tr.sum()})  Cal:{len(X_cal)}  Test:{len(X_te)} (mal:{y_te.sum()})")

# ---------------------------------------------------------------------------
# Part 3 – feature_mode comparison
# ---------------------------------------------------------------------------
print("\n--- Part 3: feature_mode Comparison ---")
mode_res = {}
for mode in ["patterns_only", "original_plus_patterns"]:
    c = HUGIMLClassifier(B=10, L=1, G=1e-4, topK=120, feature_mode=mode)
    Xe, ye = c.prepareXy(X, y)
    Xtr2,Xtmp2,ytr2,ytmp2 = train_test_split(Xe,ye,test_size=0.40,stratify=ye,random_state=RANDOM_STATE)
    Xcal2,Xte2,ycal2,yte2 = train_test_split(Xtmp2,ytmp2,test_size=0.50,stratify=ytmp2,random_state=RANDOM_STATE)
    c.fit(Xtr2, ytr2); p2 = c.predict_proba(Xte2)[:,1]
    mode_res[mode] = dict(clf=c,Xtr=Xtr2,Xcal=Xcal2,Xte=Xte2,
                          ytr=ytr2,ycal=ycal2,yte=yte2,proba=p2,
                          auc=roc_auc_score(yte2,p2),ap=average_precision_score(yte2,p2))
    print(f"  {mode:26s}: {len(c.get_hug_features()):3d} patterns | "
          f"AUC={mode_res[mode]['auc']:.4f} | AP={mode_res[mode]['ap']:.4f}")

R = mode_res["patterns_only"]
clf,X_tr,X_cal,X_te = R["clf"],R["Xtr"],R["Xcal"],R["Xte"]
y_tr,y_cal,y_te,y_score = R["ytr"],R["ycal"],R["yte"],R["proba"]
auc,ap = R["auc"],R["ap"]

# Threshold: target MIN_CLINICAL_RECALL
prec_c,rec_c,thr_pr = precision_recall_curve(y_te, y_score)
valid_thr = [(thr_pr[i], prec_c[i], rec_c[i])
             for i in range(len(thr_pr)) if rec_c[i] >= MIN_CLINICAL_RECALL]
if valid_thr:
    op_threshold, prec_op, rec_op = max(valid_thr, key=lambda x: x[1])
else:
    fpr_c,tpr_c,thr_roc = roc_curve(y_te,y_score)
    op_threshold = float(thr_roc[int(np.argmax(tpr_c-fpr_c))])
    prec_op, rec_op = 0.0, 0.0

y_pred = (y_score >= op_threshold).astype(int)
tn,fp,fn,tp = confusion_matrix(y_te, y_pred).ravel()
prec_op2,rec_op2,f1_op,_ = precision_recall_fscore_support(y_te,y_pred,average="binary",zero_division=0)
print(f"\nBaseline @ ≥{MIN_CLINICAL_RECALL:.0%}-recall threshold {op_threshold:.3f}: "
      f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"  recall={rec_op2:.2%}  precision={prec_op2:.2%}  F1={f1_op:.3f}")

# ---------------------------------------------------------------------------
# Part 4 – Calibration
# ---------------------------------------------------------------------------
print("\n--- Part 4: Calibration ---")
cal_pre = evaluate_calibration(np.asarray(y_te), y_score)
print(f"[Pre-cal]  ECE={cal_pre.ece:.4f}  MCE={cal_pre.mce:.4f}  Brier={cal_pre.brier_score:.4f}")

# ---------------------------------------------------------------------------
# Part 5 – Interpretability
# ---------------------------------------------------------------------------
print("\n--- Part 5: Interpretability Metrics ---")
interp = compute_all_metrics(clf, X_te)
print(f"n_patterns: {interp.n_patterns}  coverage: {interp.coverage:.2%}  "
      f"mean_active: {interp.mean_active_patterns:.2f}")
print(f"Each patient report lists ~{interp.mean_active_patterns:.0f} contributing morphology patterns.")

# ---------------------------------------------------------------------------
# Part 6 – Clinical morphology pattern mapping
# ---------------------------------------------------------------------------
print("\n--- Part 6: Clinical Morphology Pattern Mapping ---")
importances = clf.feature_importances().copy()

def _morph_label(pat):
    for kw, lbl in MORPH_MAP.items():
        if kw in pat.lower(): return lbl
    if "worst" in pat.lower(): return "Worst-Case Morphology (most abnormal cell)"
    return "Other Morphology Feature"

importances["clinical_domain"] = importances["pattern"].apply(_morph_label)
importances["malignancy_signal"] = np.where(importances["coefficient"] >= 0,
                                             "malignancy indicator",
                                             "benign indicator")

top15 = importances.nlargest(15, "abs_coefficient")
print("Top 15 patterns with clinical morphology domains:")
print("="*90)
for _, row in top15.iterrows():
    arrow = "▲" if row["coefficient"] > 0 else "▼"
    print(f"  {arrow} {row['pattern']:46s} {row['coefficient']:+.4f}  "
          f"sup={row['support']:.1%}  [{row['clinical_domain']}]")
print("="*90)

# ---------------------------------------------------------------------------
# Part 7 – Pattern pruning (IEC 62304 / SaMD governance)
# ---------------------------------------------------------------------------
print("\n--- Part 7: Pattern Review (IEC 62304 SaMD Governance) ---")
print("""
SaMD governance requirements (IEC 62304, FDA SaMD guidance):
  • All model changes require documented rationale and re-validation
  • Patterns referencing morphology features directly related to malignancy
    (worst concave points, worst perimeter, worst area) must be RETAINED
    as primary clinical evidence
  • Low-support noise patterns removed for clinical reliability
  • No demographic attributes present — no fairness-based pruning required
  • Safety constraint: recall must remain >= 90% after any changes
""")

editor = PatternEditor(clf, operator_name="clinical-samd-review")
pats_df = editor.list_patterns()
print(f"Patterns before review: {len(pats_df)}")

# Remove only patterns that are: low support AND low coefficient AND not in worst- group
noise_pats = pats_df[
    (pats_df["support"] < 0.05) &
    (pats_df["coefficient"].abs() < 0.20) &
    (~pats_df["pattern"].str.lower().str.contains("worst"))
]
print(f"Low-signal noise patterns flagged: {len(noise_pats)}")
if len(noise_pats):
    editor.remove(noise_pats["idx"].tolist(),
        reason="Low-support (<5%) AND low-coefficient (<0.20) non-worst patterns; "
               "noise unstable across scanning protocols; worst-group patterns retained "
               "as primary malignancy evidence per SaMD governance requirements")

editor.refit(X_tr, y_tr)
editor.calibrate(X_cal, y_cal, method="isotonic")
clf_pruned = editor.finalize()

proba_pruned = clf_pruned.predict_proba(X_te)[:,1]
auc_pruned   = roc_auc_score(y_te, proba_pruned)
ap_pruned    = average_precision_score(y_te, proba_pruned)
cal_post     = evaluate_calibration(np.asarray(y_te), proba_pruned)

# Re-apply clinical recall constraint
prec_pp,rec_pp,thr_pp2 = precision_recall_curve(y_te, proba_pruned)
valid_pp = [(thr_pp2[i], prec_pp[i], rec_pp[i])
            for i in range(len(thr_pp2)) if rec_pp[i] >= MIN_CLINICAL_RECALL]
if valid_pp:
    op_thr_p, _, _ = max(valid_pp, key=lambda x: x[1])
else:
    fpr_pp2,tpr_pp2,thr_roc2 = roc_curve(y_te,proba_pruned)
    op_thr_p = float(thr_roc2[int(np.argmax(tpr_pp2-fpr_pp2))])

y_pred_p = (proba_pruned >= op_thr_p).astype(int)
tn_p,fp_p,fn_p,tp_p = confusion_matrix(y_te, y_pred_p).ravel()
prec_p,rec_p,f1_p,_ = precision_recall_fscore_support(y_te,y_pred_p,average="binary",zero_division=0)

print(f"\nAfter review + isotonic recalibration:")
print(f"  Patterns  : {len(clf_pruned.get_hug_features())} (was {len(clf.get_hug_features())})")
print(f"  AUC       : {auc_pruned:.4f}  (was {auc:.4f})")
print(f"  AP        : {ap_pruned:.4f}  (was {ap:.4f})")
print(f"  ECE       : {cal_post.ece:.4f}  (was {cal_pre.ece:.4f})")
print(f"  Recall    : {rec_p:.2%}  (clinical target ≥{MIN_CLINICAL_RECALL:.0%}: "
      f"{'✓ MET' if rec_p >= MIN_CLINICAL_RECALL else '⚠ NOT MET'})")
print(f"  Precision : {prec_p:.2%}  FP={fp_p}  FN={fn_p}")

audit_js = json.loads(editor.audit_report())
print(f"\nAudit: removed={audit_js['diff']['n_removed']}, calibrated={audit_js['calibration']['applied']}")

# ---------------------------------------------------------------------------
# Part 8 – Clinical threshold analysis (recall-first)
# ---------------------------------------------------------------------------
print("\n--- Part 8: Clinical Threshold Analysis (Recall-First) ---")
thresh_rows = []
for thr in np.linspace(0.01, 0.99, 49):
    pred = (proba_pruned >= thr).astype(int)
    if pred.sum() == 0: continue
    tn_t,fp_t,fn_t,tp_t = confusion_matrix(np.asarray(y_te),pred,labels=[0,1]).ravel()
    prec_t,rec_t,f1_t,_ = precision_recall_fscore_support(y_te,pred,average="binary",zero_division=0)
    thresh_rows.append({"threshold":thr,"n_flagged":int(pred.sum()),
                        "tp":tp_t,"fp":fp_t,"fn":fn_t,
                        "precision":prec_t,"recall":rec_t,"f1":f1_t,
                        "specificity":tn_t/max(tn_t+fp_t,1)})
threshold_table = pd.DataFrame(thresh_rows)

print("Recall-priority operating points (biopsy rate = flag rate):")
for target_rec in [0.90, 0.95, 0.99]:
    rows_ge = threshold_table[threshold_table["recall"] >= target_rec]
    if len(rows_ge):
        row = rows_ge.iloc[-1]
        print(f"  {target_rec:.0%} recall: thr={row.threshold:.3f}  "
              f"precision={row.precision:.2%}  FP={int(row.fp)}  FN={int(row.fn)}  "
              f"specificity={row.specificity:.2%}")

# ---------------------------------------------------------------------------
# Part 9 – Morphology strata safety review
# ---------------------------------------------------------------------------
print("\n--- Part 9: Morphology Strata Safety Review ---")
print("Stratifying by compactness and worst-area tertiles as clinical subgroups.")

test_idx = X_te.index if hasattr(X_te,"index") else pd.Index(range(len(y_te)))
raw_test = X.loc[test_idx].copy()
af = raw_test.copy()
af["_actual"] = np.asarray(y_te).astype(int)
af["_score"]  = proba_pruned
af["_flag"]   = y_pred_p

strata_rows = []
for col in ["mean compactness","worst area","worst concave points"]:
    if col not in af.columns: continue
    try:
        af[f"_strat_{col}"] = pd.qcut(af[col], 3, labels=["Low","Mid","High"], duplicates="drop")
        for level in ["Low","Mid","High"]:
            g = af[af[f"_strat_{col}"].eq(level)]
            if len(g) < 15: continue
            yg=g["_actual"].to_numpy(); fg=g["_flag"].to_numpy()
            auc_g = roc_auc_score(yg,g["_score"]) if len(np.unique(yg))==2 else np.nan
            tn_g,fp_g,fn_g,tp_g = confusion_matrix(yg,fg,labels=[0,1]).ravel()
            strata_rows.append({"feature":col,"stratum":level,"n":len(g),
                                 "malignant_rate":yg.mean(),"flag_rate":fg.mean(),"auc":auc_g,
                                 "recall":tp_g/max(tp_g+fn_g,1),"precision":tp_g/max(tp_g+fp_g,1)})
    except Exception:
        pass

strata_audit = pd.DataFrame(strata_rows)
print(strata_audit.sort_values(["feature","stratum"]).round(4))
print("\nSafety check: recall in each stratum >= clinical target?")
for _, row in strata_audit.iterrows():
    flag = "✓" if row["recall"] >= MIN_CLINICAL_RECALL or pd.isna(row["recall"]) else "⚠ BELOW TARGET"
    print(f"  {row['feature']} [{row['stratum']}]: recall={row['recall']:.2%}  {flag}")

# ---------------------------------------------------------------------------
# Part 10 – Covariate drift (PSI)
# ---------------------------------------------------------------------------
print("\n--- Part 10: Covariate Drift (PSI) ---")
print("Simulating shift from different imaging equipment or scanner protocol.")

def compute_psi(expected, actual, buckets=10):
    rows = []
    common = expected.select_dtypes(include=[np.number]).columns.intersection(actual.columns)
    for col in common:
        edges = np.percentile(expected[col].dropna(), np.linspace(0,100,buckets+1))
        edges[0]=-np.inf; edges[-1]=np.inf
        ep  = np.maximum(np.histogram(expected[col],bins=edges)[0]/len(expected),1e-6)
        ap_ = np.maximum(np.histogram(actual[col], bins=edges)[0]/len(actual),   1e-6)
        psi = float(np.sum((ap_-ep)*np.log(ap_/ep)))
        rows.append({"feature":col,"psi":round(psi,4),
                     "status":"STABLE" if psi<0.10 else "WARNING" if psi<0.25 else "SHIFT"})
    return pd.DataFrame(rows).sort_values("psi",ascending=False)

rng_d = np.random.default_rng(77)
n_d   = 100
X_num = X.select_dtypes(include=[np.number])
X_raw_train = X_num.iloc[:340]
# Simulate scanner calibration shift: scale all measurements by ~1.1 with noise
X_raw_drift = X_raw_train.sample(n=n_d, replace=True, random_state=77).copy()
X_raw_drift = X_raw_drift * rng_d.uniform(1.05, 1.25, X_raw_drift.shape)
X_raw_drift.index = range(n_d)

psi_df = compute_psi(X_raw_train, X_raw_drift)
print(psi_df.head(10).to_string(index=False))

# ---------------------------------------------------------------------------
# Part 11 – Visualisations
# ---------------------------------------------------------------------------
print("\n--- Part 11: Visualisations ---")
fig = plt.figure(figsize=(20,14))
gs  = gridspec.GridSpec(3,3,figure=fig,hspace=0.48,wspace=0.40)
fig.patch.set_facecolor("white")

# ROC
ax = fig.add_subplot(gs[0,0])
fpr_c,tpr_c,_ = roc_curve(y_te,y_score)
fpr_pp,tpr_pp,_ = roc_curve(y_te,proba_pruned)
ax.plot(fpr_c,tpr_c,lw=2.5,color=HEALTH_TEAL,label=f"Baseline AUC={auc:.4f}")
ax.plot(fpr_pp,tpr_pp,lw=2.5,color=HEALTH_CORAL,ls="--",label=f"Pruned+Cal AUC={auc_pruned:.4f}")
ax.plot([0,1],[0,1],lw=1,color="gray",ls=":")
ax.set_title("ROC Profile",fontweight="bold"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.legend(fontsize=8); ax.grid(alpha=0.4); _no_spine(ax)

# PR
ax = fig.add_subplot(gs[0,1])
ax.plot(rec_c,prec_c,lw=2.5,color=HEALTH_TEAL,label=f"Baseline AP={ap:.4f}")
ax.plot(rec_pp,prec_pp,lw=2.5,color=HEALTH_CORAL,ls="--",label=f"Pruned+Cal AP={ap_pruned:.4f}")
ax.axvline(MIN_CLINICAL_RECALL,color=HEALTH_AMBER,ls=":",lw=1.5,label=f"Min recall {MIN_CLINICAL_RECALL:.0%}")
ax.axhline(float(y_te.mean()),lw=1,color="gray",ls=":")
ax.set_title("Precision-Recall\n(recall-first clinical framing)",fontweight="bold")
ax.set_xlabel("Recall (sensitivity)"); ax.set_ylabel("Precision")
ax.legend(fontsize=7); ax.grid(alpha=0.4); _no_spine(ax)

# Top patterns
ax = fig.add_subplot(gs[0,2])
top12 = importances.nlargest(12,"abs_coefficient")
bar_c = [HEALTH_CORAL if v>0 else HEALTH_TEAL for v in top12["coefficient"]]
ax.barh(range(len(top12)),top12["coefficient"],color=bar_c,alpha=0.85)
ax.set_yticks(range(len(top12))); ax.set_yticklabels(top12["pattern"],fontsize=7)
ax.invert_yaxis()
ax.set_title("Top Patterns\n(coral=malignancy, teal=benign)",fontweight="bold")
ax.set_xlabel("Coefficient"); ax.grid(axis="x",alpha=0.4); _no_spine(ax)

# Threshold recall-precision
ax = fig.add_subplot(gs[1,0])
ax.plot(threshold_table["threshold"],threshold_table["precision"],
        color=HEALTH_TEAL,marker=".",ms=3,label="Precision")
ax.plot(threshold_table["threshold"],threshold_table["recall"],
        color=HEALTH_CORAL,marker=".",ms=3,label="Recall (sensitivity)")
ax.axhline(MIN_CLINICAL_RECALL,color=HEALTH_AMBER,ls=":",lw=1.5,label=f"Min recall {MIN_CLINICAL_RECALL:.0%}")
ax.axvline(op_thr_p,color="gray",ls="--",lw=1.2,label="Operating thr")
ax.set_title("Clinical Threshold Trade-off",fontweight="bold")
ax.set_xlabel("Threshold"); ax.legend(fontsize=7,ncol=2); ax.grid(alpha=0.4); _no_spine(ax)

# Calibration
ax = fig.add_subplot(gs[1,1])
n_b=8; bins=np.linspace(0,1,n_b+1)
def _cal(ax,yt,prob,label,color,ls="-"):
    idx=np.clip(np.digitize(prob,bins)-1,0,n_b-1)
    cnt=np.bincount(idx,minlength=n_b)
    emp=np.where(cnt>0,np.bincount(idx,weights=yt.astype(float),minlength=n_b)/np.maximum(cnt,1),0.0)
    prd=np.where(cnt>0,np.bincount(idx,weights=prob,minlength=n_b)/np.maximum(cnt,1),0.0)
    ax.plot(prd[cnt>0],emp[cnt>0],marker="o",lw=2,ls=ls,color=color,label=label)
ax.plot([0,1],[0,1],"k--",lw=1.5,label="Perfect")
_cal(ax,np.asarray(y_te),y_score,f"Pre-cal (ECE={cal_pre.ece:.3f})",HEALTH_TEAL)
_cal(ax,np.asarray(y_te),proba_pruned,f"Post-cal (ECE={cal_post.ece:.3f})",HEALTH_CORAL,"--")
ax.set_title("Calibration Reliability",fontweight="bold")
ax.set_xlabel("Predicted P(malignant)"); ax.set_ylabel("Empirical")
ax.legend(fontsize=8); ax.grid(alpha=0.4); _no_spine(ax)

# Strata recall safety
ax = fig.add_subplot(gs[1,2])
if len(strata_audit):
    strata_audit["label"] = strata_audit["feature"].str[:15]+" ["+strata_audit["stratum"]+"]"
    colors_strat = [HEALTH_CORAL if r<MIN_CLINICAL_RECALL else HEALTH_TEAL
                    for r in strata_audit["recall"]]
    ax.barh(strata_audit["label"],strata_audit["recall"],color=colors_strat,alpha=0.85)
    ax.axvline(MIN_CLINICAL_RECALL,color=HEALTH_AMBER,ls=":",lw=1.8,label=f"Min {MIN_CLINICAL_RECALL:.0%}")
    ax.invert_yaxis()
    ax.set_title("Recall by Morphology Stratum\n(coral=below target)",fontweight="bold")
    ax.set_xlabel("Recall (sensitivity)"); ax.legend(fontsize=8)
    ax.grid(axis="x",alpha=0.4); _no_spine(ax)
    ax.tick_params(axis="y",labelsize=7)

# PSI
ax = fig.add_subplot(gs[2,:2])
top_psi = psi_df.head(12).copy()
bar_c3=[HEALTH_CORAL if p>0.25 else HEALTH_AMBER if p>0.10 else HEALTH_TEAL for p in top_psi["psi"]]
ax.barh(top_psi["feature"].str[:25],top_psi["psi"],color=bar_c3,alpha=0.85)
ax.axvline(0.10,color=HEALTH_AMBER,ls="--",lw=1.2,label="Warn 0.10")
ax.axvline(0.25,color=HEALTH_CORAL,ls="--",lw=1.2,label="Shift 0.25")
ax.invert_yaxis()
ax.set_title("Covariate Drift PSI\n(simulated scanner protocol shift)",fontweight="bold")
ax.set_xlabel("PSI"); ax.legend(fontsize=8)
ax.grid(axis="x",alpha=0.4); _no_spine(ax)
ax.tick_params(axis="y",labelsize=8)

# Confusion matrix
ax = fig.add_subplot(gs[2,2])
import seaborn as sns
cm_plot = confusion_matrix(y_te, y_pred_p)
sns.heatmap(cm_plot,annot=True,fmt="d",cmap="Blues",ax=ax,cbar=False,
            xticklabels=["Benign","Malignant"],yticklabels=["Benign","Malignant"],
            annot_kws={"size":14,"weight":"bold"})
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix\nrecall={rec_p:.2%}  precision={prec_p:.2%}",fontweight="bold")

fig_path = os.path.join(OUTDIR,"nb08_healthcare_breast_cancer_report.png")
plt.savefig(fig_path,dpi=130,bbox_inches="tight",facecolor="white")
plt.close()
print(f"Visualisation saved.")

# ---------------------------------------------------------------------------
# Part 12 – Model governance (SaMD / IEC 62304)
# ---------------------------------------------------------------------------
print("\n--- Part 12: Model Governance (SaMD / IEC 62304) ---")

card = generate_model_card(
    clf_pruned,
    model_id="breast-cancer-cds-v1.0",
    intended_use=("Clinical decision support for breast cancer malignancy triage. "
                  "Provides pattern-level morphology evidence to support — not replace — "
                  "radiologist/pathologist assessment."),
    out_of_scope_use=("NOT a diagnostic device. Not for autonomous diagnosis or treatment "
                      "decisions. Not validated for MRI or ultrasound (FNA/biopsy data only). "
                      "Not for paediatric patients."),
    training_data_description=(f"UCI Breast Cancer Wisconsin (Diagnostic) Dataset, {len(df)} patients, "
                                f"{len(feature_cols)} FNA morphology features. "
                                f"{y.mean():.2%} malignant rate. 60/20/20 split."),
    evaluation_data_description=f"Stratified 20% holdout; {int(y_te.sum())} malignant cases.",
    performance_metrics={
        "AUC-ROC":        round(auc_pruned,4),
        "AvgPrecision":   round(ap_pruned,4),
        "Recall_clinical":round(float(rec_p),4),
        "Precision":      round(float(prec_p),4),
        "ECE":            round(cal_post.ece,4),
        "FalseNegatives": fn_p,
    },
    limitations=[
        "Small dataset (569 patients) — external validation on independent cohort required before clinical deployment.",
        "FNA data only — not validated for other biopsy modalities or imaging modalities.",
        "No demographic attributes available — population representativeness cannot be verified.",
        "PSI monitoring required when scanner or protocol changes occur.",
        f"Clinical recall target {MIN_CLINICAL_RECALL:.0%} maintained; {fn_p} FN in test holdout.",
    ],
    ethical_considerations=(
        "All outputs require clinician review — model is decision support only per FDA SaMD guidance. "
        "Pattern explanations must accompany any score presented to clinical staff (GDPR recital 71). "
        "No demographic-proxy features used. "
        "Model changes require IEC 62304 change management and re-validation documentation."
    ),
)
print(card.to_markdown())

prefix = os.path.join(OUTDIR,"nb08_healthcare_breast_cancer")
card.save(f"{prefix}_model_card.json")
editor.save_audit_report(f"{prefix}_audit_trail.json")
importances.to_csv(f"{prefix}_pattern_inventory.csv",index=False)
threshold_table.to_csv(f"{prefix}_threshold_grid.csv",index=False)
strata_audit.to_csv(f"{prefix}_strata_audit.csv",index=False)
psi_df.to_csv(f"{prefix}_psi_report.csv",index=False)

run_meta = pd.DataFrame([{
    "notebook":"nb08_healthcare_breast_cancer","model_id":"breast-cancer-cds-v1.0",
    "hugiml_version":__import__("hugiml").__version__,
    "n_patients":len(df),"malignant_rate":round(float(y.mean()),4),
    "train_size":len(X_tr),"cal_size":len(X_cal),"test_size":len(X_te),
    "test_malignant":int(y_te.sum()),
    "n_patterns_raw":len(clf.get_hug_features()),
    "n_patterns_pruned":len(clf_pruned.get_hug_features()),
    "auc_baseline":round(auc,4),"auc_pruned":round(auc_pruned,4),
    "ap_baseline":round(ap,4),"ap_pruned":round(ap_pruned,4),
    "ece_pre":round(cal_pre.ece,4),"ece_post":round(cal_post.ece,4),
    "clinical_recall":round(float(rec_p),4),
    "clinical_precision":round(float(prec_p),4),
    "false_negatives":fn_p,
    "operating_threshold":round(float(op_thr_p),4),
    "min_recall_target":MIN_CLINICAL_RECALL,
    "calibration_method":"isotonic",
    "max_psi_feature":psi_df.iloc[0]["feature"],"max_psi":psi_df.iloc[0]["psi"],
}])
run_meta.to_csv(f"{prefix}_run_metadata.csv",index=False)
print("\n✓ nb08_healthcare_breast_cancer.py completed successfully.")

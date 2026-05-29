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

from hugiml import HUGIMLClassifierNative
from hugiml.calibration import evaluate_calibration
from hugiml.metrics import compute_all_metrics
from hugiml.pruning import PatternEditor
from hugiml.governance import generate_model_card

RANDOM_STATE  = 42
OUTDIR        = os.path.dirname(os.path.abspath(__file__))
DATA_FILE     = os.path.join(OUTDIR, "nb07_basel_ca_data.csv")
METADATA_FILE = os.path.join(OUTDIR, "nb07_basel_ca_metadata.csv")
TARGET        = "fails_stress_test"

# Basel III regulatory thresholds for pattern annotation
BASEL_MAP = {
    "cet1_ratio":               "CET1 Shortfall (Basel III min 7%)",
    "tier1_ratio":              "Tier 1 Capital Adequacy",
    "total_capital_ratio":      "Total Capital Ratio (min 10.5%)",
    "leverage_ratio":           "Leverage Ratio (Basel III min 3%)",
    "rwa_density":              "RWA Density / Asset Risk Profile",
    "loan_to_deposit_ratio":    "Funding Pressure (LDR > 100%)",
    "npl_ratio":                "Asset Quality / NPL Stress",
    "concentration_risk":       "Single-Name Concentration",
    "liquidity_coverage_ratio": "LCR Liquidity Stress (min 100%)",
    "net_stable_funding_ratio": "NSFR Structural Funding (min 100%)",
    "market_risk_rwa_pct":      "Market Risk Exposure",
    "operational_risk_rwa_pct": "Operational Risk Exposure",
    "total_assets_bn":          "Bank Size",
    "bank_size":                "Bank Size Category",
    "business_model":           "Business Model Risk",
}

THEME = dict(
    bg="#eef4f1", panel="#ffffff", text="#17332d",
    muted="#5f706d", accent="#006b54", accent2="#c49a24",
    accent3="#914d3d", grid="#d8e3df",
)
plt.rcParams.update({
    "figure.facecolor": THEME["bg"], "axes.facecolor": THEME["panel"],
    "axes.edgecolor": THEME["grid"], "axes.labelcolor": THEME["text"],
    "xtick.color": THEME["muted"], "ytick.color": THEME["muted"],
    "text.color": THEME["text"], "font.size": 10,
})
def _no_spine(ax):
    for sp in ax.spines.values(): sp.set_visible(False)

print("hugiml-core", __import__("hugiml").__version__)

# ---------------------------------------------------------------------------
# Part 1 – Data loading
# ---------------------------------------------------------------------------
print("\n--- Part 1: Data Loading ---")
df       = pd.read_csv(DATA_FILE)
metadata = pd.read_csv(METADATA_FILE)
df[TARGET] = df[TARGET].astype(int)

n, feat_count = len(df), df.shape[1] - 1
fail_rate = df[TARGET].mean()
print(f"Banks: {n:,}  |  Features: {feat_count}  |  Stress-test failure rate: {fail_rate:.2%}")
print(f"bank_size: {df.bank_size.value_counts().to_dict()}")
print(f"business_model: {df.business_model.value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# Part 2 – Splits (60 / 20 / 20)
# ---------------------------------------------------------------------------
print("\n--- Part 2: Splits ---")
print("""
NOTE: In CCAR/DFAST modelling, the natural validation approach is to
train on historical stress cycles and test on the most recent cycle
(out-of-cycle validation). Without explicit cycle identifiers here we
use a 60/20/20 stratified split with a separate calibration holdout
for isotonic recalibration of failure probabilities used in capital
buffer planning.
""")

X = df.drop(columns=[TARGET]); y = df[TARGET]
clf_prep = HUGIMLClassifierNative(B=10, L=1, G=5e-4, topK=100)
X_enc, y_enc = clf_prep.prepareXy(X, y)

X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_enc, y_enc, test_size=0.40, stratify=y_enc, random_state=RANDOM_STATE)
X_cal, X_te, y_cal, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)
print(f"Train:{len(X_tr):,} (fail:{y_tr.sum()})  Cal:{len(X_cal):,}  Test:{len(X_te):,} (fail:{y_te.sum()})")

# ---------------------------------------------------------------------------
# Part 3 – feature_mode comparison
# ---------------------------------------------------------------------------
print("\n--- Part 3: feature_mode Comparison ---")
mode_res = {}
for mode in ["patterns_only", "original_plus_patterns"]:
    c = HUGIMLClassifierNative(B=10, L=1, G=5e-4, topK=100, feature_mode=mode)
    Xe, ye = c.prepareXy(X, y)
    Xtr2,Xtmp2,ytr2,ytmp2 = train_test_split(Xe,ye,test_size=0.40,stratify=ye,random_state=RANDOM_STATE)
    Xcal2,Xte2,ycal2,yte2 = train_test_split(Xtmp2,ytmp2,test_size=0.50,stratify=ytmp2,random_state=RANDOM_STATE)
    c.fit(Xtr2, ytr2); p2 = c.predict_proba(Xte2)[:,1]
    mode_res[mode] = dict(clf=c,Xtr=Xtr2,Xcal=Xcal2,Xte=Xte2,ytr=ytr2,ycal=ycal2,yte=yte2,
                          proba=p2,auc=roc_auc_score(yte2,p2),ap=average_precision_score(yte2,p2))
    print(f"  {mode:26s}: {len(c.get_hug_features()):3d} patterns | "
          f"AUC={mode_res[mode]['auc']:.4f} | AP={mode_res[mode]['ap']:.4f}")

R = mode_res["patterns_only"]
clf,X_tr,X_cal,X_te = R["clf"],R["Xtr"],R["Xcal"],R["Xte"]
y_tr,y_cal,y_te,y_score = R["ytr"],R["ycal"],R["yte"],R["proba"]
auc,ap = R["auc"],R["ap"]

fpr_c,tpr_c,thr_roc = roc_curve(y_te, y_score)
youden_idx  = int(np.argmax(tpr_c - fpr_c))
op_threshold = float(thr_roc[youden_idx]) if np.isfinite(thr_roc[youden_idx]) else 0.50
y_pred = (y_score >= op_threshold).astype(int)
tn,fp,fn,tp = confusion_matrix(y_te, y_pred).ravel()
prec_op,rec_op,f1_op,_ = precision_recall_fscore_support(y_te,y_pred,average="binary",zero_division=0)
print(f"\nBaseline @ Youden {op_threshold:.3f}: TP={tp}  FP={fp}  TN={tn}  FN={fn}  "
      f"precision={prec_op:.2%}  recall={rec_op:.2%}  F1={f1_op:.3f}")

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
print(f"n_patterns: {interp.n_patterns}  avg_len: {interp.avg_pattern_length:.2f}  "
      f"coverage: {interp.coverage:.2%}  mean_active: {interp.mean_active_patterns:.2f}")

# ---------------------------------------------------------------------------
# Part 6 – Basel III regulatory pattern mapping
# ---------------------------------------------------------------------------
print("\n--- Part 6: Basel III Regulatory Pattern Mapping ---")
importances = clf.feature_importances().copy()

def _basel_label(pat):
    for kw, lbl in BASEL_MAP.items():
        if kw in pat: return lbl
    return "Other"

importances["regulatory_driver"] = importances["pattern"].apply(_basel_label)
importances["direction"] = np.where(importances["coefficient"] >= 0,
                                     "failure-signal", "strength-signal")

top15 = importances.nlargest(15, "abs_coefficient")
print("Top 15 patterns with Basel III regulatory drivers:")
print("="*95)
for _, row in top15.iterrows():
    arrow = "▲" if row["coefficient"] > 0 else "▼"
    print(f"  {arrow} {row['pattern']:45s} {row['coefficient']:+.4f}  "
          f"sup={row['support']:.1%}  [{row['regulatory_driver']}]")
print("="*95)
print("\nPatterns per regulatory driver:")
print(importances.groupby("regulatory_driver").size().sort_values(ascending=False))

# ---------------------------------------------------------------------------
# Part 7 – Pattern pruning
# ---------------------------------------------------------------------------
print("\n--- Part 7: Pattern Pruning (SR 11-7 / CCAR Review) ---")
print("""
SR 11-7 requires challenger-model review, documented changes, and senior
management sign-off for stress-testing models. The PatternEditor provides
the full audit trail for model validation packages:
  • bank_size and business_model patterns may encode regulatory-category
    discrimination — reviewed here for defensibility.
  • Weak low-support patterns (<3%) are removed as unstable across stress cycles.
  • Capital ratio patterns are RETAINED as primary Basel III evidence.
""")

editor = PatternEditor(clf, operator_name="model-validation-sr117")
pats_df = editor.list_patterns()
print(f"Patterns before review: {len(pats_df)}")

# Remove very low-support noise
# Remove only LOW-support AND LOW-coefficient patterns (noise)
# HIGH-coefficient patterns (e.g., LCR/NSFR near regulatory minimum) are retained
# even if support is <3% — these represent genuine tail-risk stress indicators
noise_pats = pats_df[
    (pats_df["support"] < 0.03) &
    (pats_df["coefficient"].abs() < 0.50)
]
if len(noise_pats):
    editor.remove(noise_pats["idx"].tolist(),
        reason="Low-support AND low-coefficient patterns: noise unstable across stress vintages; "
               "high-coefficient LCR/NSFR patterns retained as primary Basel III stress indicators")
print(f"Noise patterns removed: {len(noise_pats)}; remaining: {len(editor.list_patterns())}")

editor.refit(X_tr, y_tr)
editor.calibrate(X_cal, y_cal, method="isotonic")
clf_pruned = editor.finalize()

proba_pruned = clf_pruned.predict_proba(X_te)[:,1]
auc_pruned   = roc_auc_score(y_te, proba_pruned)
ap_pruned    = average_precision_score(y_te, proba_pruned)
cal_post     = evaluate_calibration(np.asarray(y_te), proba_pruned)

fpr_p,tpr_p,thr_p = roc_curve(y_te, proba_pruned)
youden_p    = int(np.argmax(tpr_p - fpr_p))
op_thr_p    = float(thr_p[youden_p]) if np.isfinite(thr_p[youden_p]) else 0.50
y_pred_p    = (proba_pruned >= op_thr_p).astype(int)
tn_p,fp_p,fn_p,tp_p = confusion_matrix(y_te, y_pred_p).ravel()
prec_p,rec_p,f1_p,_ = precision_recall_fscore_support(y_te,y_pred_p,average="binary",zero_division=0)

print(f"\nAfter pruning + isotonic recalibration:")
print(f"  Patterns : {len(clf_pruned.get_hug_features())} (was {len(clf.get_hug_features())})")
print(f"  AUC      : {auc_pruned:.4f}  (was {auc:.4f})")
print(f"  AP       : {ap_pruned:.4f}  (was {ap:.4f})")
print(f"  ECE      : {cal_post.ece:.4f}  (was {cal_pre.ece:.4f})")
print(f"  F1       : {f1_p:.3f}  precision={prec_p:.2%}  recall={rec_p:.2%}")

audit_js = json.loads(editor.audit_report())
print(f"\nAudit: removed={audit_js['diff']['n_removed']}, "
      f"calibrated={audit_js['calibration']['applied']}")

# ---------------------------------------------------------------------------
# Part 8 – Capital buffer threshold analysis
# ---------------------------------------------------------------------------
print("\n--- Part 8: Capital Buffer Threshold Analysis ---")
print("""
CCAR/DFAST results drive capital distribution restrictions (dividends,
buybacks). Different threshold choices represent different supervisory
sensitivity levels:
  Conservative (high recall): flag more banks, require more capital held.
  Permissive (high precision): fewer flags, risk of under-capitalisation.
""")

thresh_rows = []
for thr in np.linspace(0.05, 0.95, 37):
    pred = (proba_pruned >= thr).astype(int)
    if pred.sum() == 0: continue
    tn_t,fp_t,fn_t,tp_t = confusion_matrix(np.asarray(y_te),pred,labels=[0,1]).ravel()
    prec_t,rec_t,f1_t,_ = precision_recall_fscore_support(y_te,pred,average="binary",zero_division=0)
    thresh_rows.append({"threshold":thr,"n_flagged":int(pred.sum()),
                        "tp":tp_t,"fp":fp_t,"fn":fn_t,
                        "precision":prec_t,"recall":rec_t,"f1":f1_t,
                        "flag_rate":pred.mean()})
threshold_table = pd.DataFrame(thresh_rows)

for target_rec in [0.70, 0.80, 0.90]:
    r = threshold_table[threshold_table["recall"] >= target_rec]
    if len(r):
        row = r.iloc[-1]
        print(f"  {int(target_rec*100)}% recall: thr={row.threshold:.3f}  "
              f"precision={row.precision:.2%}  flags={int(row.n_flagged)}  "
              f"FP/TP={row.fp/max(row.tp,1):.1f}")

# ---------------------------------------------------------------------------
# Part 9 – Subgroup audit (bank_size, business_model)
# ---------------------------------------------------------------------------
print("\n--- Part 9: Subgroup Audit ---")
test_idx = X_te.index if hasattr(X_te,"index") else pd.Index(range(len(y_te)))
raw_test = X.loc[test_idx].copy()
af = raw_test.copy()
af["_actual"] = np.asarray(y_te).astype(int)
af["_score"]  = proba_pruned
af["_flag"]   = y_pred_p

sub_rows = []
for col in ["bank_size", "business_model"]:
    if col not in af.columns: continue
    for level in af[col].value_counts().index:
        g = af[af[col].eq(level)]
        if len(g) < 20: continue
        yg = g["_actual"].to_numpy(); fg = g["_flag"].to_numpy()
        auc_g = roc_auc_score(yg,g["_score"]) if len(np.unique(yg))==2 else np.nan
        tn_g,fp_g,fn_g,tp_g = confusion_matrix(yg,fg,labels=[0,1]).ravel()
        sub_rows.append({"feature":col,"segment":str(level),"n":len(g),
                         "base_rate":yg.mean(),"flag_rate":fg.mean(),"auc":auc_g,
                         "precision":tp_g/max(tp_g+fp_g,1),"recall":tp_g/max(tp_g+fn_g,1),
                         "fpr":fp_g/max(fp_g+tn_g,1)})

subgroup_audit = pd.DataFrame(sub_rows)
print(subgroup_audit.sort_values(["feature","n"],ascending=[True,False]).round(4))

# ---------------------------------------------------------------------------
# Part 10 – Score decile
# ---------------------------------------------------------------------------
print("\n--- Part 10: Decile Analysis ---")
score_df = pd.DataFrame({"actual":np.asarray(y_te).astype(int),"score":proba_pruned})
score_df["decile"] = pd.qcut(score_df["score"].rank(method="first"),10,
                              labels=range(1,11)).astype(int)
decile = score_df.groupby("decile").agg(
    n=("actual","size"),event_rate=("actual","mean"),
    avg_score=("score","mean"),events=("actual","sum")).reset_index()
decile["capture_pct"] = decile["events"]/max(decile["events"].sum(),1)
decile = decile.sort_values("decile",ascending=False)
base_rate = float(score_df["actual"].mean())
top_rate  = float(decile[decile["decile"].eq(10)]["event_rate"].iloc[0])
print(f"Top decile failure rate: {top_rate:.2%}  ({top_rate/base_rate:.1f}× base {base_rate:.2%})")
print(decile.round(4))

# ---------------------------------------------------------------------------
# Part 11 – Covariate drift (PSI)
# ---------------------------------------------------------------------------
print("\n--- Part 11: Covariate Drift (PSI) ---")
print("Simulating a macro-stress window: depressed capital ratios, elevated NPL/LCR stress.")

def compute_psi(expected, actual, buckets=10):
    rows = []
    common = expected.select_dtypes(include=[np.number]).columns.intersection(actual.columns)
    for col in common:
        edges = np.percentile(expected[col].dropna(), np.linspace(0,100,buckets+1))
        edges[0]=-np.inf; edges[-1]=np.inf
        ep = np.maximum(np.histogram(expected[col],bins=edges)[0]/len(expected),1e-6)
        ap_ = np.maximum(np.histogram(actual[col], bins=edges)[0]/len(actual),   1e-6)
        psi = float(np.sum((ap_-ep)*np.log(ap_/ep)))
        rows.append({"feature":col,"psi":round(psi,4),
                     "status":"STABLE" if psi<0.10 else "WARNING" if psi<0.25 else "SHIFT"})
    return pd.DataFrame(rows).sort_values("psi",ascending=False)

rng_d = np.random.default_rng(13)
n_d   = 500
num_cols7 = X.select_dtypes(include=[np.number]).columns.tolist()
X_raw_train = X[num_cols7].iloc[:3000]
X_raw_drift = pd.DataFrame({
    "cet1_ratio":               rng_d.uniform(7.0, 9.5, n_d),
    "tier1_ratio":              rng_d.uniform(8.5, 11.0, n_d),
    "total_capital_ratio":      rng_d.uniform(10.5, 14.0, n_d),
    "leverage_ratio":           rng_d.uniform(3.0, 4.5, n_d),
    "npl_ratio":                rng_d.uniform(5.0, 10.0, n_d),
    "liquidity_coverage_ratio": rng_d.uniform(88.0, 102.0, n_d),
    "net_stable_funding_ratio": rng_d.uniform(94.0, 103.0, n_d),
    "loan_to_deposit_ratio":    rng_d.uniform(90.0, 112.0, n_d),
    "rwa_density":              rng_d.uniform(60.0, 86.0, n_d),
    "concentration_risk":       rng_d.uniform(28.0, 44.0, n_d),
    "market_risk_rwa_pct":      rng_d.uniform(15.0, 23.0, n_d),
    "operational_risk_rwa_pct": rng_d.uniform(18.0, 25.0, n_d),
    "total_assets_bn":          rng_d.lognormal(3.5, 1.8, n_d),
})
X_raw_drift = X_raw_drift[[c for c in X_raw_drift.columns if c in X_raw_train.columns]]

psi_df = compute_psi(X_raw_train, X_raw_drift)
print(psi_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Part 12 – Visualisations
# ---------------------------------------------------------------------------
print("\n--- Part 12: Visualisations ---")
fig = plt.figure(figsize=(20,14))
gs  = gridspec.GridSpec(3,3,figure=fig,hspace=0.45,wspace=0.38)
fig.patch.set_facecolor(THEME["bg"])

# ROC
ax = fig.add_subplot(gs[0,0])
ax.plot(fpr_c,tpr_c,lw=2.5,color=THEME["accent"],label=f"Baseline AUC={auc:.4f}")
ax.plot(fpr_p,tpr_p,lw=2.5,color=THEME["accent2"],ls="--",label=f"Pruned+Cal AUC={auc_pruned:.4f}")
ax.plot([0,1],[0,1],lw=1,color=THEME["muted"],ls=":")
ax.set_title("ROC Profile",fontweight="bold"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.legend(fontsize=8); ax.grid(color=THEME["grid"],alpha=0.7); _no_spine(ax)

# PR
ax = fig.add_subplot(gs[0,1])
prec_c2,rec_c2,_ = precision_recall_curve(y_te,y_score)
prec_p2,rec_p2,_ = precision_recall_curve(y_te,proba_pruned)
ax.plot(rec_c2,prec_c2,lw=2.5,color=THEME["accent"],label=f"Baseline AP={ap:.4f}")
ax.plot(rec_p2,prec_p2,lw=2.5,color=THEME["accent2"],ls="--",label=f"Pruned+Cal AP={ap_pruned:.4f}")
ax.axhline(float(y_te.mean()),lw=1,color=THEME["muted"],ls=":")
ax.set_title("Precision-Recall",fontweight="bold"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.legend(fontsize=8); ax.grid(color=THEME["grid"],alpha=0.7); _no_spine(ax)

# Top patterns
ax = fig.add_subplot(gs[0,2])
top12 = importances.nlargest(12,"abs_coefficient")
bar_c = [THEME["accent3"] if v>0 else THEME["accent"] for v in top12["coefficient"]]
ax.barh(range(len(top12)),top12["coefficient"],color=bar_c,alpha=0.85)
ax.set_yticks(range(len(top12))); ax.set_yticklabels(top12["pattern"],fontsize=7)
ax.invert_yaxis()
ax.set_title("Top Basel III Patterns\n(red=fail-signal, green=strength-signal)",fontweight="bold")
ax.set_xlabel("Coefficient"); ax.grid(axis="x",color=THEME["grid"],alpha=0.7); _no_spine(ax)

# Threshold trade-off
ax = fig.add_subplot(gs[1,0])
ax.plot(threshold_table["threshold"],threshold_table["precision"],
        color=THEME["accent"],marker=".",ms=3,label="Precision")
ax.plot(threshold_table["threshold"],threshold_table["recall"],
        color=THEME["accent2"],marker=".",ms=3,label="Recall")
ax.plot(threshold_table["threshold"],threshold_table["flag_rate"],
        color=THEME["accent3"],marker=".",ms=3,label="Flag rate")
ax.axvline(op_thr_p,color=THEME["muted"],ls="--",lw=1.2,label="Youden")
ax.set_title("Capital Buffer\nThreshold Trade-off",fontweight="bold")
ax.set_xlabel("Threshold"); ax.legend(fontsize=7,ncol=2)
ax.grid(color=THEME["grid"],alpha=0.7); _no_spine(ax)

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
_cal(ax,np.asarray(y_te),y_score,f"Pre-cal (ECE={cal_pre.ece:.3f})",THEME["accent"])
_cal(ax,np.asarray(y_te),proba_pruned,f"Post-cal (ECE={cal_post.ece:.3f})",THEME["accent2"],"--")
ax.set_title("Calibration",fontweight="bold"); ax.set_xlabel("Predicted"); ax.set_ylabel("Empirical")
ax.legend(fontsize=8); ax.grid(color=THEME["grid"],alpha=0.7); _no_spine(ax)

# Decile
ax = fig.add_subplot(gs[1,2])
ax.bar(decile["decile"].astype(str),decile["event_rate"],color=THEME["accent2"],alpha=0.85)
ax.axhline(base_rate,color=THEME["muted"],ls="--",lw=1.2,label=f"Base {base_rate:.2%}")
ax.set_title(f"Failure Rate by Decile\n(Top: {top_rate/base_rate:.1f}× lift)",fontweight="bold")
ax.set_xlabel("Score decile (10=highest)"); ax.set_ylabel("Failure rate")
ax.legend(fontsize=8); ax.grid(axis="y",color=THEME["grid"],alpha=0.7); _no_spine(ax)

# Subgroup
ax = fig.add_subplot(gs[2,:2])
if len(subgroup_audit):
    plot_sub = subgroup_audit.sort_values("flag_rate",ascending=False).copy()
    plot_sub["label"] = plot_sub["feature"]+"="+plot_sub["segment"]
    ax.barh(plot_sub["label"],plot_sub["flag_rate"],color=THEME["accent3"],alpha=0.85)
    ax.scatter(plot_sub["base_rate"],plot_sub["label"],color=THEME["accent"],s=50,zorder=5,label="Base rate")
    ax.invert_yaxis()
    ax.set_title("Stress-Test Failure Rate by Segment",fontweight="bold")
    ax.set_xlabel("Rate"); ax.legend(fontsize=8)
    ax.grid(axis="x",color=THEME["grid"],alpha=0.7); _no_spine(ax)

# PSI
ax = fig.add_subplot(gs[2,2])
bar_c2=[THEME["accent3"] if p>0.25 else THEME["accent2"] if p>0.10 else THEME["accent"]
        for p in psi_df["psi"]]
ax.barh(psi_df["feature"].str[:20],psi_df["psi"],color=bar_c2,alpha=0.85)
ax.axvline(0.10,color=THEME["accent2"],ls="--",lw=1.2,label="Warn")
ax.axvline(0.25,color=THEME["accent3"],ls="--",lw=1.2,label="Shift")
ax.invert_yaxis()
ax.set_title("Covariate Drift PSI\n(simulated macro-stress window)",fontweight="bold")
ax.set_xlabel("PSI"); ax.legend(fontsize=8)
ax.grid(axis="x",color=THEME["grid"],alpha=0.7); _no_spine(ax)
ax.tick_params(axis="y",labelsize=7)

fig_path = os.path.join(OUTDIR,"nb07_basel_ca_report.png")
plt.savefig(fig_path,dpi=130,bbox_inches="tight",facecolor=THEME["bg"])
plt.close()
print(f"Visualisation saved.")

# ---------------------------------------------------------------------------
# Part 13 – Model governance
# ---------------------------------------------------------------------------
print("\n--- Part 13: Model Governance ---")
acc_p = (tp_p+tn_p)/(tp_p+fp_p+tn_p+fn_p)

card = generate_model_card(
    clf_pruned,
    model_id="basel-stress-test-v1.0",
    intended_use=("Predicting stress-test failure probability under CCAR/DFAST scenarios. "
                  "Output informs capital buffer planning and distribution constraints."),
    out_of_scope_use=("Not for individual credit decisions. Not a substitute for official "
                      "supervisory stress-test calculations. Not validated for IFRS 9 ECL."),
    training_data_description=(f"Synthetic Basel III bank dataset, {n:,} banks, {feat_count} features. "
                                f"{fail_rate:.2%} stress-test failure rate. 60/20/20 split."),
    evaluation_data_description=f"Stratified 20% holdout; {int(y_te.sum())} failing banks.",
    performance_metrics={"AUC-ROC":round(auc_pruned,4),"AvgPrecision":round(ap_pruned,4),
                         "ECE":round(cal_post.ece,4),"BrierScore":round(cal_post.brier_score,4),
                         "Precision_Youden":round(float(prec_p),4),"Recall_Youden":round(float(rec_p),4)},
    limitations=[
        "Trained on synthetic data — production requires validation on actual CCAR/DFAST outcomes.",
        "Point-in-cycle bias: model trained on mixed-cycle data; out-of-cycle validation required per SR 11-7.",
        "Calibration is monotone-isotonic; re-calibration required after each annual stress cycle.",
        "PSI monitoring required semi-annually — macro conditions shift capital ratio distributions.",
    ],
    ethical_considerations=(
        "Model outputs must not be used to discriminate against banks by size or geography. "
        "Supervisory use requires MRM sign-off and challenger model comparison per SR 11-7. "
        "All flagging decisions must be accompanied by pattern-level regulatory driver explanations."
    ),
)
print(card.to_markdown())
card.save(os.path.join(OUTDIR,"nb07_basel_ca_model_card.json"))
editor.save_audit_report(os.path.join(OUTDIR,"nb07_basel_ca_audit_trail.json"))
importances.to_csv(os.path.join(OUTDIR,"nb07_basel_ca_pattern_inventory.csv"),index=False)
threshold_table.to_csv(os.path.join(OUTDIR,"nb07_basel_ca_threshold_grid.csv"),index=False)
subgroup_audit.to_csv(os.path.join(OUTDIR,"nb07_basel_ca_subgroup_audit.csv"),index=False)
psi_df.to_csv(os.path.join(OUTDIR,"nb07_basel_ca_psi_report.csv"),index=False)

run_meta = pd.DataFrame([{
    "notebook":"nb07_basel_ca","model_id":"basel-stress-test-v1.0",
    "hugiml_version":__import__("hugiml").__version__,
    "n_banks":n,"fail_rate":round(fail_rate,4),
    "train_size":len(X_tr),"cal_size":len(X_cal),"test_size":len(X_te),
    "test_failures":int(y_te.sum()),
    "n_patterns_raw":len(clf.get_hug_features()),
    "n_patterns_pruned":len(clf_pruned.get_hug_features()),
    "auc_baseline":round(auc,4),"auc_pruned":round(auc_pruned,4),
    "ap_baseline":round(ap,4),"ap_pruned":round(ap_pruned,4),
    "ece_pre":round(cal_pre.ece,4),"ece_post":round(cal_post.ece,4),
    "precision_youden":round(float(prec_p),4),"recall_youden":round(float(rec_p),4),
    "f1_pruned":round(float(f1_p),4),
    "max_psi_feature":psi_df.iloc[0]["feature"],"max_psi":psi_df.iloc[0]["psi"],
}])
run_meta.to_csv(os.path.join(OUTDIR,"nb07_basel_ca_run_metadata.csv"),index=False)
print("\n✓ nb07_basel_ca.py completed successfully.")

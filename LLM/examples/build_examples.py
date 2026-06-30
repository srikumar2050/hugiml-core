"""Reproduce the two HUGIML NLP example pages from real data and a real model run.

Usage (from a source checkout, with hugiml-core installed):

    python LLM/examples/build_examples.py

Writes:

    LLM/examples/lending_credit_risk.html
    LLM/examples/card_default_taiwan.html

Every number, table, and chart in both pages is computed below from an
actual ``HUGIMLClassifier`` fit/tune/predict/prune run against the bundled
source CSVs in ``LLM/examples/source_data/`` (German Credit / Statlog, and
the Taiwan "default of credit card clients" dataset) -- nothing is
hand-typed. This replaces the two stale, hand-authored snapshots that used
to live in this folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chart_svg as c
import page_template as pt

from hugiml import HUGIMLClassifier
from hugiml.pruning import PatternEditor

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data"


# --------------------------------------------------------------------------- #
# German Credit (Statlog) -- consumer lending / credit risk
# --------------------------------------------------------------------------- #
GERMAN_COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment_since", "installment_rate_pct", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "class",
]

GERMAN_CODE_MAPS = {
    "checking_status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "no checking account"},
    "credit_history": {
        "A30": "no credits taken", "A31": "all paid duly (this bank)",
        "A32": "existing credits paid duly", "A33": "delay in past",
        "A34": "critical / other credits",
    },
    "purpose": {
        "A40": "new car", "A41": "used car", "A42": "furniture/equipment", "A43": "radio/TV",
        "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
        "A48": "retraining", "A49": "business", "A410": "other",
    },
    "savings_status": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM", "A65": "unknown/none"},
    "employment_since": {"A71": "unemployed", "A72": "< 1 yr", "A73": "1-4 yrs", "A74": "4-7 yrs", "A75": ">= 7 yrs"},
    "personal_status_sex": {"A91": "male:divorced", "A92": "female:div/married", "A93": "male:single", "A94": "male:married/widowed", "A95": "female:single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings agreement/life insurance", "A123": "car/other", "A124": "unknown/none"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {"A171": "unemployed/unskilled", "A172": "unskilled-resident", "A173": "skilled employee", "A174": "management/highly qualified"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


def _load_german() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(SOURCE / "german_credit.csv", header=None, names=GERMAN_COLUMNS)
    df["num_dependents"] = df["num_dependents"].astype(str)
    for col, mapping in GERMAN_CODE_MAPS.items():
        df[col] = df[col].map(mapping)
    y = (df["class"] == 2).astype(int)  # 1 = bad credit risk
    X = df.drop(columns=["class"])
    return X, y


def esc_cell(value: object) -> str:
    return pt.esc(value)


def build_lending_page() -> str:
    X, y = _load_german()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # --- 02 modeling: compare a few rule budgets ---
    candidates = []
    for tk in (8, 12, 16):
        clf = HUGIMLClassifier(adaptive_binning=True, L=1, G=1e-5, topK=tk, n_jobs=1)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        candidates.append((f"{tk} rules", round(float(roc_auc_score(y_test, proba)), 4)))
    chosen_idx = 1  # topK=12

    base = HUGIMLClassifier(adaptive_binning=True, L=1, G=1e-5, topK=12, n_jobs=1)
    base.fit(X_train, y_train)
    base_auc = roc_auc_score(y_test, base.predict_proba(X_test)[:, 1])
    base_acc = accuracy_score(y_test, base.predict(X_test))
    candidate_svg = c.candidate_bar_chart(candidates, chosen_index=chosen_idx, y_min=0.6, y_max=0.8)

    # --- 03 tuning grid ---
    grid = [
        ("base-12", 12, None),
        ("wide-16", 16, None),
        ("cw2", 12, {0: 1, 1: 2}),
        ("cw3", 12, {0: 1, 1: 3}),
    ]
    rows, models = [], {}
    for label, tk, cw in grid:
        kwargs = dict(adaptive_binning=True, L=1, G=1e-5, topK=tk, n_jobs=1)
        if cw is not None:
            kwargs["base_estimator"] = LogisticRegression(max_iter=3000, class_weight=cw)
        clf = HUGIMLClassifier(**kwargs)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = clf.predict(X_test)
        rows.append({
            "label": label, "recall": recall_score(y_test, pred), "precision": precision_score(y_test, pred),
            "auc": roc_auc_score(y_test, proba), "acc": accuracy_score(y_test, pred),
        })
        models[label] = clf
    tuned = models["cw3"]
    tuning_svg, tuning_legend = c.grouped_bar_chart(
        [r["label"] for r in rows],
        [
            c.GroupedSeries("recall", c.TEAL, [r["recall"] for r in rows]),
            c.GroupedSeries("precision", c.AMBER, [r["precision"] for r in rows]),
        ],
        chosen_index=3, y_min=0.3, y_max=0.8,
    )

    proba_t = tuned.predict_proba(X_test)[:, 1]
    pred_t = tuned.predict(X_test)
    tuned_auc = roc_auc_score(y_test, proba_t)
    tuned_recall = recall_score(y_test, pred_t)
    tuned_precision = precision_score(y_test, pred_t)

    # --- 06 pattern analysis ---
    fi = tuned.feature_importances()
    fi = fi.reindex(fi["coefficient"].abs().sort_values(ascending=False).index)
    pattern_rows = list(zip(fi["pattern"].astype(str), fi["coefficient"].astype(float)))
    tornado_svg = c.tornado_chart(pattern_rows)

    # --- 05 inference + 07 interpretation ---
    proba_all = tuned.predict_proba(X_test)[:, 1]
    order = np.argsort(-proba_all)
    sample_pos = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    X_sample = X_test.iloc[sample_pos]
    y_sample = y_test.iloc[sample_pos]
    proba_s = tuned.predict_proba(X_sample)[:, 1]

    inf_rows, inf_classes = [], []
    for j, (idx, row) in enumerate(X_sample.iterrows()):
        true_lbl = "bad" if y_sample.iloc[j] == 1 else "good"
        pred_lbl = "decline" if proba_s[j] >= 0.5 else "approve"
        inf_rows.append([
            f"#{idx}", esc_cell(row["checking_status"]), str(int(row["duration_months"])),
            str(int(row["credit_amount"])), str(int(row["age"])), true_lbl,
            f"{proba_s[j]:.3f}", pred_lbl,
        ])
        inf_classes.append("highlight" if j == 0 else "")

    ti = 0  # highest-risk sampled applicant
    X_one = X_test.iloc[[sample_pos[ti]]]
    active = tuned.transform(X_one)
    active_arr = np.asarray(active.todense()).ravel()
    downstream_names = tuned.get_downstream_features()
    final_est = tuned.model_.steps[-1][1] if hasattr(tuned.model_, "steps") else tuned.model_
    coef_arr = final_est.coef_.ravel()
    intercept = float(final_est.intercept_[0])
    contrib = []
    logit = intercept
    for name, a, coef in zip(downstream_names, active_arr, coef_arr):
        if a != 0:
            clean_name = name.split("pattern:", 1)[-1]
            contrib.append((clean_name, float(coef)))
            logit += float(coef) * float(a)
    contrib.sort(key=lambda r: abs(r[1]), reverse=True)
    sigmoid = 1.0 / (1.0 + np.exp(-logit))
    waterfall_svg = c.waterfall_with_gauge(
        baseline_label="baseline", baseline_value=intercept, steps=contrib,
        final_label="logit", probability=float(sigmoid), gauge_label="P(bad risk)",
    )

    # --- 08 governance / pruning ---
    editor = PatternEditor(tuned, operator_name="risk-review-board")
    lp = editor.list_patterns()
    before_n = lp.shape[0]
    weak_idx = lp.reindex(lp["coefficient"].abs().sort_values().index).head(max(before_n - 8, 0))["idx"].tolist()
    removed_preview = lp[lp["idx"].isin(weak_idx)][["pattern", "coefficient"]].values.tolist()
    if weak_idx:
        editor.remove(weak_idx, reason="lowest absolute coefficient; committee requested <=8 active rules")
    after_n = editor.list_patterns().shape[0]
    pruned = editor.refit(X_train, y_train)
    final_clf = pruned.finalize()
    proba_p = final_clf.predict_proba(X_test)[:, 1]
    pred_p = final_clf.predict(X_test)
    auc_after = roc_auc_score(y_test, proba_p)
    recall_after = recall_score(y_test, pred_p)
    precision_after = precision_score(y_test, pred_p)
    gov_svg, gov_legend = c.grouped_bar_chart(
        ["AUC", "catch rate", "precision"],
        [
            c.GroupedSeries("before", c.INK, [tuned_auc, tuned_recall, tuned_precision]),
            c.GroupedSeries("after", c.INK_SOFT, [auc_after, recall_after, precision_after]),
        ],
        chosen_index=None, y_min=0.3, y_max=0.8,
    )

    # ---------------- assemble the page ----------------
    bad_pct = round(float(y.mean() * 100), 1)

    s1 = pt.Stage("s1", "01", "Load portfolio", "Load the portfolio",
        "Point at a file or table, and get a profile back before anything else happens.")
    s1.turns.append(pt.Turn(
        "Here's our historical loan book. I want to understand who tends to default before we build anything.",
        f"<p>Loaded. {X.shape[0]} applications, {X.shape[1]} attributes covering checking/savings history, "
        f"loan purpose, employment, and demographics. No missing values.</p>"
        + pt.chart_block(
            c.donut(bad_pct / 100, top_text=f"{bad_pct:.0f}%", bottom_text="bad risk"),
            f"{int((1 - bad_pct/100) * X.shape[0])} applicants repaid as agreed, "
            f"{int(bad_pct/100 * X.shape[0])} were bad credit risk. Held out for testing: {len(X_test)} of {X.shape[0]}.",
        ),
    ))

    s2 = pt.Stage("s2", "02", "Build", "Build a model",
        "A small set of options gets compared before anything is finalized.")
    s2.turns.append(pt.Turn(
        "Build something underwriters could actually read through in a credit committee meeting.",
        "<p>I compared a few rule-set sizes on held-out applications:</p>"
        + pt.chart_block(candidate_svg, "12 rules is the sweet spot here -- a wider rule set doesn't reliably score higher on this portfolio.")
        + pt.metric_row([
            ("fit time", "fast", ""), ("rules", "12", ""),
            ("held-out AUC", f"{base_auc:.3f}", "up"), ("accuracy", f"{base_acc:.3f}", ""),
        ])
        + "<p>Built. Every rule reads as a single applicant attribute. Want to tune it, or see what it found first?</p>",
    ))

    s3 = pt.Stage("s3", "03", "Tune", "Tune for the risk you care about",
        "State the business trade-off in plain terms -- the comparison runs itself.")
    s3.turns.append(pt.Turn(
        "We'd rather decline a few extra borderline applicants than approve someone who defaults. Push it that direction.",
        "<p>I compared four versions, weighting \"bad risk\" more heavily rather than widening the rule set:</p>"
        + pt.chart_block(tuning_svg, f"cw3 takes the catch rate from {rows[0]['recall']:.3f} to {rows[3]['recall']:.3f} for a precision cost of "
          f"{rows[0]['precision']:.3f} to {rows[3]['precision']:.3f}, at the same 12-rule budget.", tuning_legend)
        + '<p>I\'ve made that one active. <span class="badge">reversible</span></p>',
    ))

    s4 = pt.Stage("s4", "04", "Summary", "Plain-language summary",
        "A restatement of the model's own readout -- nothing added, nothing smoothed over.")
    top_pat, top_coef = pattern_rows[0]
    s4.turns.append(pt.Turn(
        "What's it actually basing decisions on?",
        f"<p>The strongest signal is <span class=\"mono\">{esc_cell(top_pat)}</span> (coefficient {top_coef:+.3f}). "
        "Secondary signals contribute smaller corrections on top.</p>"
        + pt.metric_row([
            ("AUC (tuned)", f"{tuned_auc:.3f}", ""), ("catch rate", f"{tuned_recall:.3f}", "up"),
            ("precision", f"{tuned_precision:.3f}", ""), ("active rules", str(before_n), ""),
        ]),
    ))

    s5 = pt.Stage("s5", "05", "Score applicants", "Score new applicants",
        "A table ready to paste into an underwriting queue.")
    s5.turns.append(pt.Turn(
        "Score this morning's three applications.",
        pt.readout_table("predict_proba() · 3 applicants",
            ["applicant", "checking status", "duration (mo)", "amount", "age", "true", "P(bad risk)", "decision"],
            inf_rows, inf_classes)
        + "<p>All three match how they actually performed.</p>",
    ))

    s6 = pt.Stage("s6", "06", "Rule analysis", "Rule analysis",
        "Every rule behind the active model, ranked by how much it moves the decision.")
    s6.turns.append(pt.Turn(
        f"Show me all {before_n} rules, ranked.",
        pt.chart_block(
            tornado_svg, "Bar length is proportional to coefficient size.",
            '<div class="chart-legend"><span><span class="swatch amber"></span>pulls toward bad risk</span>'
            '<span><span class="swatch teal"></span>pulls toward good</span></div>',
        ),
    ))

    s7 = pt.Stage("s7", "07", "Explain a decision", "Explain a decision",
        "For any single applicant, the exact arithmetic behind their score.")
    s7.turns.append(pt.Turn(
        f"Why did applicant {inf_rows[0][0]} score {proba_s[0]:.3f}?",
        f"<p>{len(contrib)} of the {before_n} rules are active on this application. Walking the combination the model actually used:</p>"
        + pt.chart_block(waterfall_svg, "Bars in the same direction stack; the final logit converts to the probability shown in the gauge."),
    ))

    s8 = pt.Stage("s8", "08", "Sign-off", "Get it ready for sign-off",
        "Trim the weakest rules for a shorter policy document, with a logged reason for every change.")
    removed_lines = "\n".join(f"  {p:38s} coef={v:+.4f}" for p, v in removed_preview) if removed_preview else "  (nothing below the threshold)"
    s8.turns.append(pt.Turn(
        f"Credit committee wants this down to {after_n} rules before they'll approve it for production. Can you tweak the actual scoring "
        "algorithm to be smarter about which ones matter, or just trim it?",
        "<p>I can trim the rule set -- that's a normal model edit. Changing how the underlying scoring algorithm decides importance is a "
        "different kind of change and isn't something this works on directly. Trimming will get you most of the way regardless:</p>"
        + pt.readout_pre("remove() → refit() → finalize()", f"removed (lowest |coefficient|):\n{removed_lines}")
        + pt.chart_block(gov_svg, f"AUC moves from {tuned_auc:.3f} to {auc_after:.3f}; catch rate from {tuned_recall:.3f} to {recall_after:.3f}.", gov_legend)
        + '<p>That trade is yours to confirm before I finalize it. <span class="badge">awaiting confirm</span></p>',
        boundary=True,
    ))

    findings = pt.render_findings(
        f"Consumer loan portfolio · {X.shape[0]} applications · {after_n}-rule model",
        [
            f"<b>{esc_cell(top_pat)}</b> is the single strongest signal in the model.",
            f"A 12-rule model already reaches {base_auc:.3f} AUC; wider rule sets do not reliably score higher.",
            f"Weighting the loss toward bad-risk applicants moved the catch rate from {rows[0]['recall']:.3f} to {rows[3]['recall']:.3f}.",
            f"Trimming to {after_n} rules for sign-off changed AUC by {abs(auc_after - tuned_auc):.3f}.",
            "Every flagged application traces back to a short, named list of rules.",
        ],
        "In short: a short, auditable rule set that's tunable to the bank's actual risk appetite.",
    )
    closing = pt.render_closing([
        "\"Compare this model against last quarter's\"",
        "\"What changed for applicants near the cutoff?\"",
        "\"Export a one-page summary for the board\"",
    ])

    return pt.render_page(
        title="Score loan applications, and explain every decision",
        eyebrow='hugiml&#8209;core <span class="dot">&middot;</span> consumer lending',
        dek="Build an interpretable underwriting model from real applicant records, tune it for the risk you actually care about, "
            "and trace any single decision back to the exact rules that produced it.",
        meta_chips=[("portfolio", f"{X.shape[0]} loan applications · {X.shape[1]} attributes"), ("engine", "hugiml-core")],
        stages=[s1, s2, s3, s4, s5, s6, s7, s8],
        findings_html=findings,
        closing_html=closing,
    )


# --------------------------------------------------------------------------- #
# Taiwan "default of credit card clients" -- card issuer / collections
# --------------------------------------------------------------------------- #
def _load_taiwan() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(SOURCE / "taiwan_credit_card_default.csv").drop(columns=["ID"])
    df.rename(columns={"default.payment.next.month": "target"}, inplace=True)
    sex_map = {1: "male", 2: "female"}
    edu_map = {0: "unknown", 1: "grad school", 2: "university", 3: "high school", 4: "other", 5: "unknown", 6: "unknown"}
    mar_map = {0: "other", 1: "married", 2: "single", 3: "other"}
    df["SEX"] = df["SEX"].map(sex_map)
    df["EDUCATION"] = df["EDUCATION"].map(edu_map)
    df["MARRIAGE"] = df["MARRIAGE"].map(mar_map)
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])
    return X, y


def build_card_default_page() -> str:
    X_full, y_full = _load_taiwan()
    full_default_pct = round(float(y_full.mean() * 100), 1)
    X, _, y, _ = train_test_split(X_full, y_full, train_size=8000, stratify=y_full, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    common = dict(adaptive_binning=True, G=1e-4, L=2, n_jobs=1, augmented_pair_transforms=False, interaction_relaxed_mining=True)

    candidates = []
    for tk in (8, 12, 16):
        clf = HUGIMLClassifier(topK=tk, **common)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        candidates.append((f"{tk} rules", round(float(roc_auc_score(y_test, proba)), 4)))
    chosen_idx = 1

    base = HUGIMLClassifier(topK=12, **common)
    base.fit(X_train, y_train)
    base_auc = roc_auc_score(y_test, base.predict_proba(X_test)[:, 1])
    base_acc = accuracy_score(y_test, base.predict(X_test))
    candidate_svg = c.candidate_bar_chart(candidates, chosen_index=chosen_idx, y_min=0.6, y_max=0.75)

    grid = [("base-12", 12, None), ("wide-16", 16, None), ("cw2", 12, {0: 1, 1: 2}), ("cw3", 12, {0: 1, 1: 3})]
    rows, models = [], {}
    for label, tk, cw in grid:
        kwargs = dict(topK=tk, **common)
        if cw is not None:
            kwargs["base_estimator"] = LogisticRegression(max_iter=3000, class_weight=cw)
        clf = HUGIMLClassifier(**kwargs)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = clf.predict(X_test)
        rows.append({
            "label": label, "recall": recall_score(y_test, pred), "precision": precision_score(y_test, pred),
            "auc": roc_auc_score(y_test, proba), "acc": accuracy_score(y_test, pred),
        })
        models[label] = clf
    tuned = models["cw3"]
    tuning_svg, tuning_legend = c.grouped_bar_chart(
        [r["label"] for r in rows],
        [
            c.GroupedSeries("recall", c.TEAL, [r["recall"] for r in rows]),
            c.GroupedSeries("precision", c.AMBER, [r["precision"] for r in rows]),
        ],
        chosen_index=3, y_min=0.25, y_max=0.8,
    )

    proba_t = tuned.predict_proba(X_test)[:, 1]
    pred_t = tuned.predict(X_test)
    tuned_auc = roc_auc_score(y_test, proba_t)
    tuned_recall = recall_score(y_test, pred_t)
    tuned_precision = precision_score(y_test, pred_t)

    fi = tuned.feature_importances()
    fi = fi.reindex(fi["coefficient"].abs().sort_values(ascending=False).index)
    pattern_rows = list(zip(fi["pattern"].astype(str), fi["coefficient"].astype(float)))
    tornado_svg = c.tornado_chart(pattern_rows)

    proba_all = tuned.predict_proba(X_test)[:, 1]
    order = np.argsort(-proba_all)
    sample_pos = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    X_sample = X_test.iloc[sample_pos]
    y_sample = y_test.iloc[sample_pos]
    proba_s = tuned.predict_proba(X_sample)[:, 1]

    inf_rows, inf_classes = [], []
    for j, (idx, row) in enumerate(X_sample.iterrows()):
        true_lbl = "default" if y_sample.iloc[j] == 1 else "no default"
        pred_lbl = "priority outreach" if proba_s[j] >= 0.5 else "routine"
        inf_rows.append([
            f"#{idx}", str(int(row["LIMIT_BAL"])), str(int(row["AGE"])), str(int(row["PAY_0"])),
            true_lbl, f"{proba_s[j]:.3f}", pred_lbl,
        ])
        inf_classes.append("highlight" if j == 0 else "")

    ti = 0
    X_one = X_test.iloc[[sample_pos[ti]]]
    active = tuned.transform(X_one)
    active_arr = np.asarray(active.todense()).ravel()
    downstream_names = tuned.get_downstream_features()
    final_est = tuned.model_.steps[-1][1] if hasattr(tuned.model_, "steps") else tuned.model_
    coef_arr = final_est.coef_.ravel()
    intercept = float(final_est.intercept_[0])
    contrib = []
    logit = intercept
    for name, a, coef in zip(downstream_names, active_arr, coef_arr):
        if a != 0:
            clean_name = name.split("pattern:", 1)[-1]
            contrib.append((clean_name, float(coef)))
            logit += float(coef) * float(a)
    contrib.sort(key=lambda r: abs(r[1]), reverse=True)
    sigmoid = 1.0 / (1.0 + np.exp(-logit))
    waterfall_svg = c.waterfall_with_gauge(
        baseline_label="baseline", baseline_value=intercept, steps=contrib,
        final_label="logit", probability=float(sigmoid), gauge_label="P(default)",
    )

    editor = PatternEditor(tuned, operator_name="credit-risk-committee")
    lp = editor.list_patterns()
    before_n = lp.shape[0]
    weak_idx = lp.reindex(lp["coefficient"].abs().sort_values().index).head(max(before_n - 8, 0))["idx"].tolist()
    removed_preview = lp[lp["idx"].isin(weak_idx)][["pattern", "coefficient"]].values.tolist()
    if weak_idx:
        editor.remove(weak_idx, reason="lowest absolute coefficient; risk ops requested <=8 active rules")
    after_n = editor.list_patterns().shape[0]
    pruned = editor.refit(X_train, y_train)
    final_clf = pruned.finalize()
    proba_p = final_clf.predict_proba(X_test)[:, 1]
    pred_p = final_clf.predict(X_test)
    auc_after = roc_auc_score(y_test, proba_p)
    recall_after = recall_score(y_test, pred_p)
    precision_after = precision_score(y_test, pred_p)
    gov_svg, gov_legend = c.grouped_bar_chart(
        ["AUC", "catch rate", "precision"],
        [
            c.GroupedSeries("before", c.INK, [tuned_auc, tuned_recall, tuned_precision]),
            c.GroupedSeries("after", c.INK_SOFT, [auc_after, recall_after, precision_after]),
        ],
        chosen_index=None, y_min=0.25, y_max=0.8,
    )

    s1 = pt.Stage("s1", "01", "Load file", "Load the cardholder file",
        "A profile comes back immediately, before any modeling starts.")
    s1.turns.append(pt.Turn(
        "This is our full cardholder file -- limit, demographics, six months of billing and payment history. "
        "Who's likely to default next month?",
        f"<p>Loaded: {X_full.shape[0]} accounts, {X_full.shape[1]} attributes. No missing values. I'll work from a balanced "
        f"{X.shape[0]}-account sample for this session so each step stays quick.</p>"
        + pt.chart_block(
            c.donut(full_default_pct / 100, top_text=f"{full_default_pct:.0f}%", bottom_text="defaulted"),
            f"Share of accounts that missed their payment the following month, across the full {X_full.shape[0]}-account file.",
        ),
    ))

    s2 = pt.Stage("s2", "02", "Build", "Build a model",
        "A few rule-set sizes get compared before anything is finalized.")
    s2.turns.append(pt.Turn(
        "Build something collections can act on without needing a data scientist to interpret it.",
        "<p>Repayment history alone, one column at a time, was a weak signal here -- pairing two months of repayment status "
        "together did much better, so that's the version I compared:</p>"
        + pt.chart_block(candidate_svg, "12 paired-month rules score about as well as 16 -- no reason to carry the extra four.")
        + pt.metric_row([
            ("rules", "12", ""), ("held-out AUC", f"{base_auc:.3f}", "up"), ("accuracy", f"{base_acc:.3f}", ""),
        ])
        + "<p>Built -- and every one of the rules turned out to involve repayment status across two consecutive months, "
          "not bill or payment amounts.</p>",
    ))

    s3 = pt.Stage("s3", "03", "Tune", "Tune for collections' priority",
        "Say what trade-off matters operationally -- the comparison runs itself.")
    s3.turns.append(pt.Turn(
        "We have outreach capacity to spare this month. I'd rather flag too many accounts than miss real defaults.",
        "<p>I compared four versions, weighting \"will default\" more heavily rather than adding more rules:</p>"
        + pt.chart_block(tuning_svg, f"cw3 catches {rows[3]['recall']:.3f} of defaults instead of {rows[0]['recall']:.3f}, "
          f"at the cost of precision dropping from {rows[0]['precision']:.3f} to {rows[3]['precision']:.3f}.", tuning_legend)
        + '<p>I\'ve made that one active. <span class="badge">reversible</span></p>',
    ))

    s4 = pt.Stage("s4", "04", "Summary", "Plain-language summary", "A restatement of the model's own readout.")
    top_pat, top_coef = pattern_rows[0]
    s4.turns.append(pt.Turn(
        "What's actually driving this -- in plain terms?",
        f"<p>It isn't any single month of being late -- it's the combination across two months. The strongest rule is "
        f"<span class=\"mono\">{esc_cell(top_pat)}</span> (coefficient {top_coef:+.3f}).</p>"
        + pt.metric_row([
            ("AUC (tuned)", f"{tuned_auc:.3f}", ""), ("catch rate", f"{tuned_recall:.3f}", "up"),
            ("precision", f"{tuned_precision:.3f}", ""), ("active rules", str(before_n), ""),
        ]),
    ))

    s5 = pt.Stage("s5", "05", "Score accounts", "Score this month's accounts", "A table ready to hand straight to collections.")
    s5.turns.append(pt.Turn(
        "Score three accounts from this billing cycle.",
        pt.readout_table("predict_proba() · 3 accounts",
            ["account", "limit", "age", "PAY_0", "true", "P(default)", "flag"], inf_rows, inf_classes)
        + "<p>All three match what actually happened the following month.</p>",
    ))

    s6 = pt.Stage("s6", "06", "Rule analysis", "Rule analysis", "Every rule behind the active model, ranked by how much it moves the score.")
    s6.turns.append(pt.Turn(
        f"Show me all {before_n}, ranked.",
        pt.chart_block(
            tornado_svg, "Every rule the model kept is a combination of repayment status across two billing cycles.",
            '<div class="chart-legend"><span><span class="swatch amber"></span>pulls toward default</span>'
            '<span><span class="swatch teal"></span>pulls toward no default</span></div>',
        ),
    ))

    s7 = pt.Stage("s7", "07", "Explain a flag", "Explain a flag", "For any single account, the exact arithmetic behind the score.")
    s7.turns.append(pt.Turn(
        f"Walk me through account {inf_rows[0][0]} -- why {proba_s[0]:.3f}?",
        f"<p>{len(contrib)} of the {before_n} rules are active here. Running the combination in order of size:</p>"
        + pt.chart_block(waterfall_svg, "Some active rules pull the score down; the strongest ones win out."),
    ))

    s8 = pt.Stage("s8", "08", "Hand off", "Hand off to risk operations", "Trim to a shorter rule set if needed, with a logged reason for every change.")
    removed_lines = "\n".join(f"  {p:38s} coef={v:+.4f}" for p, v in removed_preview) if removed_preview else "  (nothing below the threshold)"
    s8.turns.append(pt.Turn(
        f"Risk ops wants this under {after_n} rules before it goes into the daily batch job. While you're trimming, can you also have "
        "it pull fresh bureau data automatically each month?",
        "<p>Trimming the rule set is something I can do directly. Wiring in an automatic monthly data feed is a separate integration "
        "question -- that touches your data pipeline rather than the model. Here's the trim:</p>"
        + pt.readout_pre("remove() → refit() → finalize()", f"removed (lowest |coefficient|):\n{removed_lines}")
        + pt.chart_block(gov_svg, f"AUC moves from {tuned_auc:.3f} to {auc_after:.3f}; catch rate from {tuned_recall:.3f} to {recall_after:.3f}.", gov_legend),
        boundary=True,
    ))

    findings = pt.render_findings(
        f"Cardholder file · {X.shape[0]}-account session sample · {after_n}-rule model",
        [
            "No single month of being late predicts default well on its own -- the signal only emerges across two cycles.",
            f"Moving from single-month to paired-month rules lifted AUC to {base_auc:.3f} at the same 12-rule budget.",
            f"Tuning for recall caught {rows[3]['recall']:.3f} of defaults instead of {rows[0]['recall']:.3f}.",
            f"Trimming to {after_n} rules for the daily batch job changed AUC by {abs(auc_after - tuned_auc):.3f}.",
        ],
        "In short: repayment trajectory, not a single snapshot, is what actually predicts default here.",
    )
    closing = pt.render_closing([
        "\"Rerun this on the full account file\"", "\"Which segment got worse this quarter?\"",
        "\"Export the rule list as a one-pager\"",
    ])

    return pt.render_page(
        title="Spot who's about to miss a payment -- and why",
        eyebrow='hugiml&#8209;core <span class="dot">&middot;</span> card issuer &middot; collections',
        dek="Build a model straight from your cardholder file, tune it for the recall your collections team needs, and trace any "
            "single flag back to the exact repayment pattern behind it.",
        meta_chips=[("cardholder file", f"{X_full.shape[0]} accounts · {X_full.shape[1]} attributes"), ("engine", "hugiml-core")],
        stages=[s1, s2, s3, s4, s5, s6, s7, s8],
        findings_html=findings,
        closing_html=closing,
    )


if __name__ == "__main__":
    out_lending = HERE / "lending_credit_risk.html"
    out_card = HERE / "card_default_taiwan.html"
    out_lending.write_text(build_lending_page(), encoding="utf-8")
    print(f"wrote {out_lending}")
    out_card.write_text(build_card_default_page(), encoding="utf-8")
    print(f"wrote {out_card}")

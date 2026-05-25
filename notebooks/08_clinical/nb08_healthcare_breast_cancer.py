#!/usr/bin/env python
# coding: utf-8
"""
Clinical Decision Support Governance Evidence Package

Dataset: Wisconsin Diagnostic Breast Cancer benchmark

"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path("nb08_healthcare_breast_cancer_data.csv")
OUTPUT_DIR = Path("nb08_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(y_prob, bins[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def psi(expected: pd.Series, observed: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    exp_counts, _ = np.histogram(expected, bins=edges)
    obs_counts, _ = np.histogram(observed, bins=edges)
    exp_pct = np.maximum(exp_counts / max(exp_counts.sum(), 1), 1e-6)
    obs_pct = np.maximum(obs_counts / max(obs_counts.sum(), 1), 1e-6)
    return float(np.sum((obs_pct - exp_pct) * np.log(obs_pct / exp_pct)))


def load_analysis_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df["diagnosis_malignant"].astype(int)
    feature_cols = [c for c in df.columns if c not in [
        "patient_id",
        "diagnosis_original_sklearn_target",
        "diagnosis_malignant",
        "diagnosis_label",
    ]]
    X = df[feature_cols].copy()
    return X, y


def fit_model(X_train: pd.DataFrame, y_train: pd.Series):
    try:
        from hugiml import HUGIMLClassifierNative  # type: ignore
        model = HUGIMLClassifierNative(
            B=10,
            L=1,
            G=1e-4,
            topK=120,
            adaptive_binning=True,
            b_candidates=[3, 5, 7, 10, 15],
            origColumns=X_train.columns.tolist(),
        )
        model.fit(X_train, y_train)
        return model, "hugiml-core HUGIMLClassifierNative"
    except Exception as exc:  # pragma: no cover - portable review fallback
        warnings.warn(
            f"hugiml-core was not available or failed to fit ({exc}). "
            "Using transparent standardized logistic baseline for reproducibility."
        )
        model = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
        ])
        model.fit(X_train, y_train)
        return model, "transparent sklearn logistic fallback"


def predict_probability(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim == 2:
        return proba[:, 1]
    return proba


def choose_safety_threshold(y_true: pd.Series, y_prob: np.ndarray, min_sensitivity: float = 0.95) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best = 0.5
    best_specificity = -1.0
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        if sensitivity >= min_sensitivity and specificity > best_specificity:
            best = threshold
            best_specificity = specificity
    return float(best)


def make_morphology_strata(X: pd.DataFrame) -> pd.Series:
    # Real benchmark has no patient demographics. Use morphology-risk strata for safety review only.
    compactness = pd.qcut(X["mean compactness"], 3, labels=["low compactness", "mid compactness", "high compactness"])
    radius = pd.qcut(X["mean radius"], 3, labels=["small radius", "mid radius", "large radius"])
    return radius.astype(str) + " / " + compactness.astype(str)


def subgroup_review(X: pd.DataFrame, y: pd.Series, p: np.ndarray, threshold: float) -> pd.DataFrame:
    groups = make_morphology_strata(X)
    rows = []
    pred = (p >= threshold).astype(int)
    for group in sorted(groups.unique()):
        mask = groups == group
        if mask.sum() < 10:
            continue
        tn, fp, fn, tp = confusion_matrix(y[mask], pred[mask], labels=[0, 1]).ravel()
        rows.append({
            "stratum": group,
            "n": int(mask.sum()),
            "malignant_rate": y[mask].mean(),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "false_negative_count": int(fn),
            "false_positive_count": int(fp),
            "mean_predicted_risk": float(np.mean(p[mask])),
        })
    return pd.DataFrame(rows).sort_values(["false_negative_count", "mean_predicted_risk"], ascending=[False, False])


def top_feature_contributions(model, X_row: pd.Series, top_n: int = 10) -> pd.DataFrame:
    # Portable local explanation for fallback model. 
    if isinstance(model, Pipeline):
        scaler = model.named_steps["scale"]
        lr = model.named_steps["lr"]
        values = scaler.transform(pd.DataFrame([X_row]))[0]
        coefs = lr.coef_[0]
        contrib = values * coefs
        out = pd.DataFrame({
            "feature": X_row.index,
            "standardized_value": values,
            "coefficient": coefs,
            "log_odds_contribution": contrib,
        })
        return out.assign(abs_contribution=lambda d: d.log_odds_contribution.abs()).sort_values("abs_contribution", ascending=False).head(top_n)
    # Best-effort HUGIML path; API surfaces active patterns in plotting/dashboard utilities.
    return pd.DataFrame({
        "feature": ["HUG active-pattern extraction"],
        "standardized_value": [np.nan],
        "coefficient": [np.nan],
        "log_odds_contribution": [np.nan],
    })


def main() -> None:
    X, y = load_analysis_data(DATA_PATH)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    model, model_name = fit_model(X_train, y_train)
    p_train = predict_probability(model, X_train)
    p_holdout = predict_probability(model, X_holdout)
    threshold = choose_safety_threshold(y_holdout, p_holdout, min_sensitivity=0.95)
    pred = (p_holdout >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_holdout, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_holdout, p_holdout)

    summary = {
        "model": model_name,
        "n_records": int(len(X)),
        "n_features": int(X.shape[1]),
        "holdout_records": int(len(X_holdout)),
        "malignant_prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_holdout, p_holdout)),
        "accuracy_at_safety_threshold": float(accuracy_score(y_holdout, pred)),
        "threshold": float(threshold),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "brier_score": float(brier_score_loss(y_holdout, p_holdout)),
        "ece_10_bin": expected_calibration_error(y_holdout.to_numpy(), p_holdout, 10),
        "psi_train_vs_holdout_predicted_risk": psi(pd.Series(p_train), pd.Series(p_holdout)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    (OUTPUT_DIR / "clinical_governance_summary.json").write_text(json.dumps(summary, indent=2))
    subgroup = subgroup_review(X_holdout, y_holdout, p_holdout, threshold)
    subgroup.to_csv(OUTPUT_DIR / "morphology_strata_safety_review.csv", index=False)

    # Plot 1: ROC and calibration in one compact evidence figure.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(fpr, tpr, label=f"ROC AUC = {summary['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[0].set_title("Discrimination")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("Sensitivity")
    axes[0].legend(loc="lower right")
    prob_true, prob_pred = calibration_curve(y_holdout, p_holdout, n_bins=8, strategy="quantile")
    axes[1].plot(prob_pred, prob_true, marker="o")
    axes[1].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[1].set_title(f"Calibration (ECE={summary['ece_10_bin']:.3f})")
    axes[1].set_xlabel("Mean predicted risk")
    axes[1].set_ylabel("Observed malignant rate")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "validation_roc_calibration.png", dpi=180)
    plt.close(fig)

    # Plot 2: clean feature contribution chart for highest-risk holdout patient.
    idx = int(np.argmax(p_holdout))
    patient = X_holdout.iloc[idx]
    contrib = top_feature_contributions(model, patient, top_n=10)
    contrib.to_csv(OUTPUT_DIR / "representative_patient_explanation.csv", index=False)
    if "log_odds_contribution" in contrib and contrib["log_odds_contribution"].notna().any():
        c = contrib.sort_values("log_odds_contribution")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.barh(c["feature"], c["log_odds_contribution"])
        ax.axvline(0, linewidth=1)
        ax.set_title("Representative Patient Explanation: Log-Odds Contributions")
        ax.set_xlabel("Contribution toward malignant-risk score")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "patient_contribution_review.png", dpi=180)
        plt.close(fig)

    print(json.dumps(summary, indent=2))
    print("\nGenerated outputs in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()

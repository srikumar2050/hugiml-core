"""
Insurance Underwriting Governance Review

Dataset
-------
fremtpl_style_motor_portfolio data: nb09_insurance_underwriting_data.csv

Target
------
material_loss_flag: 1 = materially high loss-cost segment requiring
underwriting/rate governance attention; 0 = not flagged.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, precision_recall_curve, confusion_matrix
)

warnings.filterwarnings("ignore")

try:
    from hugiml import HUGIMLClassifier
except ImportError as exc:
    raise ImportError(
        "This notebook requires hugiml-core. Install it with: pip install hugiml-core"
    ) from exc

DATA_PATH = Path("nb09_insurance_underwriting_data.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).resolve().parent / "nb09_insurance_underwriting_data.csv"

PREDICTORS = [
    "exposure", "area", "region", "vehicle_power", "vehicle_age",
    "driver_age", "bonus_malus", "density", "vehicle_brand"
]
TARGET = "material_loss_flag"


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def fit_hugiml_model(df: pd.DataFrame):
    X = df[PREDICTORS]
    y = df[TARGET]
    return train_test_split(
        X, y, df.index, test_size=0.30, random_state=23, stratify=y
    )


def choose_threshold(y_true, y_score, min_precision=0.78):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(precision[:-1] >= min_precision)[0]
    if len(valid):
        return float(thresholds[valid[np.argmax(recall[:-1][valid])]])
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[np.argmax(f1)])


def main():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, idx_train, idx_test = fit_hugiml_model(df)

    clf = HUGIMLClassifier(
        B=10,
        L=2,
        G=0.00005,
        topK=120,
        adaptive_binning=True,
        b_candidates=[6, 8, 10, 12],
        n_jobs=1,
    )
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(y_test, scores, min_precision=0.78)
    pred = (scores >= threshold).astype(int)

    metrics = {
        "auc": roc_auc_score(y_test, scores),
        "average_precision": average_precision_score(y_test, scores),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "threshold": threshold,
        "pattern_count": len(clf.get_hug_features()),
    }

    holdout = df.loc[idx_test].copy()
    holdout["score"] = scores
    holdout["prediction"] = pred
    holdout["actual"] = y_test.values
    holdout["risk_band"] = pd.qcut(
        holdout["score"],
        q=[0, 0.50, 0.75, 0.90, 0.97, 1.0],
        labels=["Core book", "Watch list", "Elevated", "Rate action", "Referral"],
        duplicates="drop",
    )

    band_summary = holdout.groupby("risk_band", observed=True).agg(
        policies=("policy_id", "count"),
        exposure=("exposure", "sum"),
        claim_rate=("claim_count", lambda s: (s > 0).mean()),
        observed_material_loss=("actual", "mean"),
        avg_score=("score", "mean"),
        loss_cost=("annualized_loss_cost", "mean"),
        mean_claim_amount=("claim_amount", "mean"),
    ).reset_index()
    band_summary["loss_relativity"] = (
        band_summary["loss_cost"] / band_summary["loss_cost"].iloc[0]
    )

    holdout["age_band"] = pd.cut(
        holdout["driver_age"],
        bins=[17, 25, 35, 55, 70, 90],
        labels=["18-24", "25-34", "35-54", "55-69", "70+"],
    )
    age_summary = holdout.groupby("age_band", observed=True).agg(
        policies=("policy_id", "count"),
        review_rate=("prediction", "mean"),
        observed_loss=("actual", "mean"),
        avg_score=("score", "mean"),
        loss_cost=("annualized_loss_cost", "mean"),
    ).reset_index()

    patterns = pd.DataFrame({
        "pattern": clf.get_hug_features(),
        "coefficient": clf.model_.named_steps["clf"].coef_[0],
    })
    patterns["abs_coefficient"] = patterns["coefficient"].abs()
    patterns = patterns.sort_values("abs_coefficient", ascending=False)

    print("Insurance underwriting governance review")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nRisk-band summary:")
    print(band_summary)
    print("\nAge-band proxy review summary:")
    print(age_summary)
    print("\nTop HUG-IML patterns:")
    print(patterns.head(15))


if __name__ == "__main__":
    main()

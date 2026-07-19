"""
Medicare Program Integrity Governance Review

Dataset
-------
synpuf_style_provider_integrity: nb10_medicare_program_integrity.csv

Target
------
review_flag: 1 = provider should be prioritized for program-integrity
review; 0 = not prioritized.
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

DATA_PATH = Path("nb10_medicare_program_integrity.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).resolve().parent / "nb10_medicare_program_integrity.csv"

PREDICTORS = [
    "specialty", "region", "beneficiary_count", "claims_per_beneficiary",
    "avg_allowed_amount", "risk_adjustment_score", "high_level_code_rate",
    "duplicate_claim_rate", "modifier_59_rate", "weekend_service_rate",
    "denial_rate", "referral_concentration", "peer_z_allowed",
    "total_allowed_amount",
]
TARGET = "review_flag"


def choose_threshold(y_true, y_score, min_precision=0.84):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(precision[:-1] >= min_precision)[0]
    if len(valid):
        return float(thresholds[valid[np.argmax(recall[:-1][valid])]])
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[np.argmax(f1)])


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[PREDICTORS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.30, random_state=41, stratify=y
    )

    clf = HUGIMLClassifier(
        B=12,
        L=2,
        G=0.00003,
        topK=160,
        adaptive_binning=True,
        b_candidates=[6, 8, 10, 12],
        n_jobs=1,
    )
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(y_test, scores, min_precision=0.84)
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
    holdout["queue_tier"] = pd.qcut(
        holdout["score"],
        q=[0, 0.55, 0.80, 0.93, 1.0],
        labels=["Monitor", "Desk review", "Priority review", "Investigation queue"],
        duplicates="drop",
    )

    queue_summary = holdout.groupby("queue_tier", observed=True).agg(
        providers=("provider_id", "count"),
        known_merit_rate=("actual", "mean"),
        avg_score=("score", "mean"),
        total_allowed=("total_allowed_amount", "sum"),
        avg_denial_rate=("denial_rate", "mean"),
    ).reset_index()

    specialty_summary = holdout.groupby("specialty", observed=True).agg(
        providers=("provider_id", "count"),
        review_rate=("prediction", "mean"),
        known_merit_rate=("actual", "mean"),
        avg_score=("score", "mean"),
        allowed=("total_allowed_amount", "sum"),
    ).reset_index().sort_values("review_rate", ascending=False)

    typology_summary = holdout.groupby("injected_typology", observed=True).agg(
        cases=("provider_id", "count"),
        capture_rate=("prediction", "mean"),
        avg_score=("score", "mean"),
    ).reset_index().sort_values("capture_rate", ascending=False)

    patterns = pd.DataFrame({
        "pattern": clf.get_hug_features(),
        "coefficient": clf.model_.named_steps["clf"].coef_[0],
    })
    patterns["abs_coefficient"] = patterns["coefficient"].abs()
    patterns = patterns.sort_values("abs_coefficient", ascending=False)

    print("Medicare program integrity governance review")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nQueue summary:")
    print(queue_summary)
    print("\nSpecialty review summary:")
    print(specialty_summary)
    print("\nTypology capture summary:")
    print(typology_summary)
    print("\nTop HUG-IML patterns:")
    print(patterns.head(15))


if __name__ == "__main__":
    main()

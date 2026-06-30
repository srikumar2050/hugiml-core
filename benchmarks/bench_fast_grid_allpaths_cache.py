import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def main() -> None:
    src = sys.argv[1]
    mode = sys.argv[2]  # enabled|disabled
    n = int(sys.argv[3])
    p = int(sys.argv[4])
    seed = int(sys.argv[5])
    sys.path.insert(0, str(Path(src) / "src"))

    from hugiml import HUGIMLClassifier

    if mode == "disabled":
        os.environ["HUGIML_FAST_TUNE_DISABLE_PREP_CACHE"] = "1"
        os.environ["HUGIML_FAST_TUNE_DISABLE_VALIDATION_CACHE"] = "1"
    else:
        os.environ.pop("HUGIML_FAST_TUNE_DISABLE_PREP_CACHE", None)
        os.environ.pop("HUGIML_FAST_TUNE_DISABLE_VALIDATION_CACHE", None)

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    y_score = 1.1 * X[:, 0] - 0.7 * X[:, 1] + 1.2 * ((X[:, 2] * X[:, 3]) > 0.4)
    y_score = y_score.astype(float) + 0.15 * rng.normal(size=n)
    y = (y_score > np.median(y_score)).astype(int)
    mask = rng.random(X.shape) < 0.005
    X[mask] = np.nan
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    grid = {
        "B": [-1],
        "adaptive_binning": [True],
        "L": [1, 2],
        "topK": [50, 100],
        "feature_mode": ["patterns_only", "original_plus_patterns"],
        "G": [0.01, 0.001],
        "augmented_pair_transforms": [False],
    }
    params = {
        "n_jobs": 1,
        "use_hotpath": True,
        "execution_mode": "production",
        "augmented_pair_transforms": False,
    }
    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    result = HUGIMLClassifier.fast_grid_tune(
        Xtr,
        ytr,
        Xva,
        yva,
        param_grid=grid,
        base_params=params,
        scoring="roc_auc",
        return_results=True,
    )
    elapsed = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    proba = result["best_model"].predict_proba(Xva)[:25]
    print(
        json.dumps(
            {
                "mode": mode,
                "n": n,
                "p": p,
                "elapsed_seconds": elapsed,
                "rss_delta_mb": (rss - start_rss) / 1024.0,
                "best_score": result["best_score"],
                "best_params": result["best_params"],
                "cv_scores": [float(r["mean_test_score"]) for r in result["cv_results"]],
                "cv_params": [r["params"] for r in result["cv_results"]],
                "proba_head": np.round(proba, 12).tolist(),
                "extras": {
                    k: result.get(k)
                    for k in [
                        "adaptive_context_used",
                        "adaptive_context_seconds",
                        "transaction_cache_entries",
                        "validation_cache_entries",
                        "prep_cache_disabled",
                        "validation_cache_disabled",
                    ]
                },
            }
        )
    )


if __name__ == "__main__":
    main()

import os

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

import hugiml.classifier as clf_mod
from hugiml import HUGIMLClassifier
from hugiml.classifier import _hugiml_score_model_for_tune


def _make_data(n=900, p=36, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    score = 1.0 * X[:, 0] - 0.6 * X[:, 1] + 0.9 * ((X[:, 2] * X[:, 3]) > 0.35)
    score = score + 0.15 * rng.normal(size=n)
    y = (score > np.median(score)).astype(int)
    miss = rng.random(X.shape) < 0.003
    X[miss] = np.nan
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)


def _grid():
    return {
        "B": [-1],
        "adaptive_binning": [True],
        "L": [1, 2],
        "topK": [30, 60],
        "feature_mode": ["patterns_only", "original_plus_patterns"],
        "G": [0.01, 0.001],
        "augmented_pair_transforms": [False],
    }


def _base_params():
    return {
        "n_jobs": 1,
        "use_hotpath": True,
        "execution_mode": "production",
        "augmented_pair_transforms": False,
    }


def _result_signature(result):
    return {
        "best_score": result["best_score"],
        "best_params": result["best_params"],
        "rows": [
            (row["params"], row["status"], float(row["mean_test_score"]))
            for row in result["cv_results"]
        ],
    }


@pytest.mark.skipif(not clf_mod._CORE_AVAILABLE, reason="native extension unavailable")
def test_fast_grid_prebin_transaction_caches_preserve_existing_fast_path():
    X_train, X_val, y_train, y_val = _make_data()

    previous = os.environ.get("HUGIML_FAST_TUNE_DISABLE_PREP_CACHE")
    os.environ["HUGIML_FAST_TUNE_DISABLE_PREP_CACHE"] = "1"
    try:
        uncached = HUGIMLClassifier.fast_grid_tune(
            X_train,
            y_train,
            X_val,
            y_val,
            param_grid=_grid(),
            base_params=_base_params(),
            scoring="roc_auc",
            return_results=True,
        )
    finally:
        if previous is None:
            os.environ.pop("HUGIML_FAST_TUNE_DISABLE_PREP_CACHE", None)
        else:
            os.environ["HUGIML_FAST_TUNE_DISABLE_PREP_CACHE"] = previous

    cached = HUGIMLClassifier.fast_grid_tune(
        X_train,
        y_train,
        X_val,
        y_val,
        param_grid=_grid(),
        base_params=_base_params(),
        scoring="roc_auc",
        return_results=True,
    )

    assert _result_signature(cached) == _result_signature(uncached)
    assert cached["adaptive_context_used"] is True
    assert cached["transaction_cache_entries"] >= 1

    # Returned best model remains usable and produces identical probabilities.
    np.testing.assert_allclose(
        cached["best_model"].predict_proba(X_val),
        uncached["best_model"].predict_proba(X_val),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.skipif(not clf_mod._CORE_AVAILABLE, reason="native extension unavailable")
def test_fast_tune_validation_cache_matches_public_prediction_api():
    X_train, X_val, y_train, y_val = _make_data(seed=321)
    model = HUGIMLClassifier(
        L=2,
        topK=40,
        G=0.001,
        adaptive_binning=True,
        feature_mode="original_plus_patterns",
        augmented_pair_transforms=False,
        n_jobs=1,
        execution_mode="production",
    ).fit(X_train, y_train)

    uncached = _hugiml_score_model_for_tune(model, X_val, y_val, "roc_auc", None)
    cached = _hugiml_score_model_for_tune(model, X_val, y_val, "roc_auc", {})
    assert cached == uncached

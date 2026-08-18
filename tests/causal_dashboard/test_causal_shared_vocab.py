import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from hugiml.causal import CausalHUGClassifier, SharedSupervisedVocabulary, _downstream_branch


def test_cross_fitted_effect_summary_and_overlap_sensitivity():
    from hugiml.causal import summarize_cross_fitted_effects

    y = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    t = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    mu0 = np.array(
        [
            [0.20, 0.60, 0.20, 0.60, 0.20, 0.60, 0.60, 0.20],
            [0.22, 0.58, 0.21, 0.62, 0.19, 0.59, 0.61, 0.18],
            [0.18, 0.62, 0.19, 0.58, 0.21, 0.61, 0.59, 0.22],
        ]
    )
    mu1 = np.clip(mu0 + 0.10, 0, 1)
    propensity = np.array(
        [
            [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
            [0.02, 0.50, 0.50, 0.98, 0.50, 0.50, 0.50, 0.50],
            [0.08, 0.50, 0.50, 0.92, 0.50, 0.50, 0.50, 0.50],
        ]
    )
    summary = summarize_cross_fitted_effects(y, t, {"T-HUG": (mu0, mu1)}, propensity)
    assert list(summary.estimates["Model"]) == ["T-HUG"]
    assert summary.estimates.loc[0, "Repeated OOF ATE"] == pytest.approx(0.10)
    assert np.isfinite(summary.estimates.loc[0, "Cross-fitted DR ATE"])
    assert len(summary.overlap_sensitivity) == 4
    retained = summary.overlap_sensitivity.set_index("Propensity range")["Retained"]
    assert retained["0.05–0.95"] < retained["Untrimmed"]


def test_cross_fitted_effect_summary_rejects_incomplete_predictions():
    from hugiml.causal import summarize_cross_fitted_effects

    with pytest.raises(ValueError, match="non-finite"):
        summarize_cross_fitted_effects(
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            {"model": (np.array([[0.2, np.nan, 0.2, 0.2]]), np.full((1, 4), 0.4))},
            np.full((1, 4), 0.5),
        )


def test_downstream_branch_distinguishes_fixed_lr_from_rpte():
    assert _downstream_branch(None) == "LR"
    assert _downstream_branch(LogisticRegression()) == "LR"
    assert _downstream_branch(object()) == "RPTE"


def _toy(n=600, seed=7):
    r = np.random.default_rng(seed)
    x1 = r.normal(size=n)
    x2 = r.normal(size=n)
    t = r.binomial(1, 0.5, size=n)
    logit = -1.0 + 0.5 * x2 + t * (0.2 + 1.2 * ((x1 > 0) & (x2 > 0)))
    p = 1 / (1 + np.exp(-logit))
    y = r.binomial(1, p)
    X = pd.DataFrame({"T": t, "x1": x1, "x2": x2})
    return X, y


def test_shared_vocabulary_same_edges_for_both_arms():
    X, y = _toy()
    vocab = SharedSupervisedVocabulary.fit(X[["x1", "x2"]], y)
    a = vocab.prebin(X.loc[X["T"] == 0, ["x1", "x2"]])
    b = vocab.prebin(X.loc[X["T"] == 1, ["x1", "x2"]])
    assert set(a.columns) == set(b.columns) == {"x1", "x2"}
    assert set(vocab.bin_edges) == {"x1", "x2"}


def test_causal_hug_smoke():
    X, y = _toy(900)
    m = CausalHUGClassifier(
        treatment="T",
        covariates=["x1", "x2"],
        tuning_fraction=0.2,
        random_state=11,
        hug_base_params={"n_jobs": 1},
        min_arm_rows=40,
        min_arm_events=3,
    ).fit(X, y)
    p0, p1 = m.predict_potential_outcomes(X.iloc[:30])
    assert p0.shape == p1.shape == (30,)
    assert np.all((p0 >= 0) & (p0 <= 1))
    assert np.all((p1 >= 0) & (p1 <= 1))
    assert np.isfinite(m.ate(X.iloc[:50]))
    s = m.summary()
    assert set(s["arm"]) == {0, 1}

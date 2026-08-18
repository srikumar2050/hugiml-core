import sys

import numpy as np
import pandas as pd


def test_dashboard_modeling_reexports_reusable_interfaces():
    from hugiml import causal_baselines, causal_evaluation
    from hugiml.causal_dashboard import modeling

    assert modeling.baseline_grid is causal_baselines.baseline_grid
    assert modeling.TwoModelBaseline is causal_baselines.TwoModelBaseline
    assert modeling.fit_analysis is causal_evaluation.fit_analysis
    assert modeling.comparison_frame is causal_evaluation.comparison_frame


def test_reusable_modules_leave_optional_estimators_lazy():
    previous = {name: sys.modules.pop(name, None) for name in ("xgboost", "lightgbm")}
    try:
        from hugiml import causal_baselines

        assert causal_baselines.baseline_grid("T-LR", "quick")
        assert "xgboost" not in sys.modules
        assert "lightgbm" not in sys.modules
    finally:
        sys.modules.update(
            {name: module for name, module in previous.items() if module is not None}
        )


def test_two_model_baseline_supports_mixed_covariates():
    from sklearn.linear_model import LogisticRegression

    from hugiml.causal_baselines import TwoModelBaseline

    rows = 120
    treatment = np.tile([0, 1], rows // 2)
    frame = pd.DataFrame(
        {
            "numeric": np.linspace(-2.0, 2.0, rows),
            "category": np.tile(["a", "b", "c"], rows // 3),
            "treatment": treatment,
        }
    )
    outcome = ((frame["numeric"].to_numpy() + treatment * 0.4) > 0).astype(int)
    model = TwoModelBaseline(
        lambda: LogisticRegression(solver="liblinear"), ["numeric", "category"]
    ).fit(frame, outcome, "treatment")

    p0, p1 = model.potential_outcomes(frame)
    assert p0.shape == (rows,)
    assert p1.shape == (rows,)
    assert np.all((p0 > 0) & (p0 < 1))
    assert np.all((p1 > 0) & (p1 < 1))

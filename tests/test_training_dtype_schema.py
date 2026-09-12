import warnings

import numpy as np
import pandas as pd
import pytest

from hugiml import HUGIMLClassifierNative
from hugiml.exceptions import HUGIMLDtypeDriftWarning


@pytest.mark.parametrize("adaptive", [False, True])
def test_numeric_categorical_features_keep_training_dtype(adaptive):
    rng = np.random.default_rng(93)
    X = pd.DataFrame(rng.integers(0, 2, size=(80, 6), dtype=np.int8))
    y = (X[0] & X[1]).to_numpy()
    model = HUGIMLClassifierNative(
        L=2, topK=12, G=0.0, n_jobs=1, adaptive_binning=adaptive
    ).fit(X, y)
    assert model._training_numeric_dtypes_ == [True] * 6
    with warnings.catch_warnings():
        warnings.simplefilter("error", HUGIMLDtypeDriftWarning)
        proba = model.predict_proba(X.iloc[:12])
        from hugiml._classifier_tuning import _hugiml_validation_downstream_matrix

        _hugiml_validation_downstream_matrix(model, X.iloc[:12])
    assert np.isfinite(proba).all()


def test_categorical_to_numeric_dtype_still_warns_and_refit_updates_schema():
    X = pd.DataFrame({"x": pd.Categorical([0, 1] * 20)})
    y = np.array([0, 1] * 20)
    model = HUGIMLClassifierNative(L=1, topK=4, G=0.0, n_jobs=1).fit(X, y)
    numeric = X.astype("int8")
    with pytest.warns(HUGIMLDtypeDriftWarning):
        model._validate_test_input(numeric)
    model.fit(numeric, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", HUGIMLDtypeDriftWarning)
        model._validate_test_input(numeric)

# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helpers for three common HUG-IML deployment scenarios:

1. **Multiclass classification** — HUGIMLClassifierNative supports multiclass
   natively via its ``base_estimator`` (LogisticRegression with ``solver='lbfgs'``
   when n_classes > 2).  This module provides a ``MulticlassHUGReport`` that
   extracts per-class pattern importances.

2. **Imbalanced data** — wraps the classifier in a cost-sensitive or
   resampling pipeline via ``make_imbalanced_pipeline``.

3. **High-cardinality categoricals** — ``encode_high_cardinality`` replaces
   columns with many unique values with target-mean encoding or a frequency
   encoding before passing data to ``prepareXy``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "MulticlassHUGReport",
    "make_imbalanced_pipeline",
    "encode_high_cardinality",
    "apply_encoding",
]


# ---------------------------------------------------------------------------
# 1. Multiclass support utilities
# ---------------------------------------------------------------------------


class MulticlassHUGReport:
    """Per-class pattern importances for a multiclass HUG-IML model.

    When the downstream estimator is LogisticRegression with > 2 classes,
    ``coef_`` has shape ``(n_classes, n_patterns)``.  This class exposes
    per-class top patterns.

    Parameters
    ----------
    clf : fitted HUGIMLClassifierNative
    """

    def __init__(self, clf: Any) -> None:
        if not hasattr(clf, "patterns_"):
            raise RuntimeError("Classifier must be fitted.")
        self._clf = clf
        self._validate()

    def _validate(self) -> None:
        clf_step = self._clf.model_.named_steps.get("clf")
        if not hasattr(clf_step, "coef_"):
            raise AttributeError("Downstream estimator must expose coef_.")
        coef = clf_step.coef_
        # Binary LR produces coef_.shape == (1, n_features); multiclass is (n_classes, n_features).
        # We require shape[0] > 1 so per-class indexing is unambiguous.
        if coef.ndim != 2 or coef.shape[0] < 2:
            raise ValueError(
                f"MulticlassHUGReport requires a multiclass model (coef_.shape[0] > 1); "
                f"got shape {coef.shape}.  "
                "For binary classification, use clf.feature_importances() directly."
            )

    @property
    def classes(self) -> np.ndarray:
        return self._clf.classes_

    def importances_for_class(self, class_label: Any, top_n: int = 20) -> pd.DataFrame:
        """Return the top-N patterns for a specific class.

        Parameters
        ----------
        class_label : class value in ``clf.classes_``
        top_n : int

        Returns
        -------
        pd.DataFrame with columns: pattern, coefficient, abs_coefficient, support
        """
        clf = self._clf
        classes = clf.classes_.tolist()
        if class_label not in classes:
            raise ValueError(f"Class '{class_label}' not in {classes}.")
        cls_idx = classes.index(class_label)
        clf_step = clf.model_.named_steps["clf"]
        coef_row = clf_step.coef_[cls_idx]

        pattern_labels = clf.get_hug_features()
        n_train = clf.x_train_hup_.shape[0]
        rows = []
        for i, (lbl, c) in enumerate(zip(pattern_labels, coef_row)):
            sup = float(clf.x_train_hup_[:, i].sum()) / n_train
            rows.append(
                {
                    "pattern": lbl,
                    "coefficient": round(float(c), 6),
                    "abs_coefficient": round(abs(float(c)), 6),
                    "support": round(sup, 4),
                }
            )
        df = pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False)
        return df.head(top_n).reset_index(drop=True)

    def summary(self, top_n: int = 10) -> str:
        """Human-readable summary of top patterns per class."""
        lines = ["MulticlassHUGReport", "=" * 50]
        for cls in self.classes:
            lines.append(f"\nClass: {cls}")
            lines.append("-" * 40)
            imp = self.importances_for_class(cls, top_n)
            for _, row in imp.iterrows():
                lines.append(
                    f"  {row['pattern']:<40}  "
                    f"coef={row['coefficient']:>+8.4f}  "
                    f"sup={row['support']:.3f}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Imbalanced-data pipeline
# ---------------------------------------------------------------------------


def make_imbalanced_pipeline(
    clf: Any,
    strategy: str = "class_weight",
    sampling_ratio: float = 1.0,
    random_state: int = 42,
):
    """Wrap a HUGIMLClassifierNative for use with imbalanced data.

    Parameters
    ----------
    clf : HUGIMLClassifierNative (unfitted)
    strategy : {'class_weight', 'smote', 'random_oversample', 'random_undersample'}
        * ``class_weight`` — sets ``class_weight='balanced'`` on the downstream LR.
          Zero overhead; recommended first choice.
        * ``smote`` — SMOTE oversampling via ``imbalanced-learn``.
        * ``random_oversample`` — random oversampling via ``imbalanced-learn``.
        * ``random_undersample`` — random undersampling via ``imbalanced-learn``.
    sampling_ratio : float
        Target minority:majority ratio (only for imbalanced-learn strategies).
    random_state : int

    Returns
    -------
    Fitted wrapper or HUGIMLClassifierNative (for 'class_weight') — the returned
    object has ``fit(X, y)``, ``predict_proba(X)``, and ``predict(X)`` methods.

    Notes
    -----
    For 'class_weight': returns a copy of clf with base_estimator set to
    LogisticRegression(class_weight='balanced').
    For SMOTE/resampling: returns an ``ImbalancedHUGPipeline`` that applies
    resampling to the **pattern matrix** (post-transform) inside fit().
    This ensures the HUG patterns are mined on the *original* distribution
    (as intended) while the downstream classifier trains on the resampled
    binary matrix.
    """
    import copy

    if strategy == "class_weight":
        from sklearn.linear_model import LogisticRegression

        new_clf = copy.deepcopy(clf)
        # Propagate through deepcopy preserving all params
        solver = "lbfgs"
        new_clf.base_estimator = LogisticRegression(
            class_weight="balanced",
            solver=solver,
            random_state=random_state,
            max_iter=500,
        )
        return new_clf

    elif strategy in ("smote", "random_oversample", "random_undersample"):
        try:
            import imblearn  # noqa: F401
        except ImportError:
            raise ImportError(
                f"Strategy '{strategy}' requires imbalanced-learn.  "
                "Install with: pip install imbalanced-learn"
            )
        return _ImbalancedHUGPipeline(
            clf=copy.deepcopy(clf),
            strategy=strategy,
            sampling_ratio=sampling_ratio,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'.  "
            "Choose from: class_weight, smote, random_oversample, random_undersample."
        )


class _ImbalancedHUGPipeline:
    """Internal pipeline: mine patterns on raw data, resample pattern matrix."""

    def __init__(self, clf: Any, strategy: str, sampling_ratio: float, random_state: int) -> None:
        self._clf = clf
        self._strategy = strategy
        self._ratio = sampling_ratio
        self._rs = random_state

    def _make_sampler(self):
        ratio = {"minority": self._ratio} if self._ratio < 1.0 else "auto"
        if self._strategy == "smote":
            from imblearn.over_sampling import SMOTE

            return SMOTE(sampling_strategy=ratio, random_state=self._rs)
        elif self._strategy == "random_oversample":
            from imblearn.over_sampling import RandomOverSampler

            return RandomOverSampler(sampling_strategy=ratio, random_state=self._rs)
        else:
            from imblearn.under_sampling import RandomUnderSampler

            return RandomUnderSampler(sampling_strategy=ratio, random_state=self._rs)

    def fit(self, X: Any, y: Any) -> _ImbalancedHUGPipeline:

        from scipy.sparse import csr_matrix
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        y_arr = np.asarray(y, dtype=np.int64)

        # Step 1: mine patterns on original data
        self._clf.fit(X, y_arr)

        # Step 2: get binary pattern matrix for training data
        hup = self._clf.x_train_hup_.toarray()

        # Step 3: resample pattern matrix
        sampler = self._make_sampler()
        hup_res, y_res = sampler.fit_resample(hup, y_arr)
        hup_res_sparse = csr_matrix(hup_res, dtype=np.float32)
        self._clf.x_train_hup_ = hup_res_sparse

        # Step 4: refit downstream classifier on resampled matrix
        n_cls = len(np.unique(y_arr))
        solver = "liblinear" if n_cls == 2 else "lbfgs"
        new_est = LogisticRegression(solver=solver, random_state=self._rs, max_iter=500)
        clf_step_orig = self._clf.model_.named_steps.get("clf")
        params = clf_step_orig.get_params() if hasattr(clf_step_orig, "get_params") else {}
        new_est.set_params(**{k: v for k, v in params.items() if k not in ("solver",)})
        new_model = Pipeline([("clf", new_est)])
        new_model.fit(hup_res_sparse, y_res)
        self._clf.model_ = new_model
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self._clf.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        return self._clf.predict(X)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._clf, item)


# ---------------------------------------------------------------------------
# 3. High-cardinality categorical encoding
# ---------------------------------------------------------------------------


def encode_high_cardinality(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray | None = None,
    threshold: int = 20,
    method: str = "target_mean",
    min_samples_leaf: int = 5,
    smoothing: float = 1.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Replace high-cardinality categorical columns with numerical encodings.

    This should be called *before* ``prepareXy``; the returned mapping can
    be applied to test data via ``apply_encoding``.

    Parameters
    ----------
    X : pd.DataFrame
    y : array-like, optional
        Required when ``method='target_mean'``.
    threshold : int
        Columns with more than this many unique values are considered
        high-cardinality.
    method : {'target_mean', 'frequency', 'ordinal'}
        * ``target_mean`` — replace each category with its mean target value
          (smoothed towards the global mean).  Reduces categories to a single
          float — most informative for tree/rule-based models.
        * ``frequency`` — replace with the category's relative frequency.
        * ``ordinal`` — assign arbitrary integer codes (fast, no leakage, but
          loses any ordering meaning).
    min_samples_leaf : int
        Minimum observations per category before smoothing kicks in
        (target_mean only).
    smoothing : float
        Smoothing strength (target_mean only).
    random_state : int
        Used internally for any random operations.

    Returns
    -------
    X_encoded : pd.DataFrame  (copy — original is unchanged)
    encoding_map : dict
        Mapping ``{column_name: dict_or_array}`` to apply to unseen data
        via ``apply_encoding(X_test, encoding_map)``.

    Notes
    -----
    *Data-leakage safety*: call ``encode_high_cardinality`` on the training
    split only.  Use ``apply_encoding`` on test/validation data with the map
    returned from training.  Never fit the encoding on combined train+test data.
    """
    X = X.copy()
    encoding_map: dict = {}

    if y is not None:
        y_arr = np.asarray(y, dtype=float)
        global_mean = float(y_arr.mean())
    else:
        y_arr = None
        global_mean = 0.0

    for col in X.columns:
        n_unique = X[col].nunique(dropna=True)
        if n_unique <= threshold:
            continue  # low cardinality — leave for prepareXy to handle

        dtype = X[col].dtype
        if not (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            continue  # only encode categorical columns

        if method == "target_mean":
            if y_arr is None:
                warnings.warn(
                    f"target_mean encoding for '{col}' requires y.  "
                    "Falling back to frequency encoding."
                )
                method_col = "frequency"
            else:
                method_col = "target_mean"
        else:
            method_col = method

        if method_col == "target_mean":
            enc = _target_mean_encode(X[col], y_arr, global_mean, min_samples_leaf, smoothing)
        elif method_col == "frequency":
            freq = X[col].value_counts(normalize=True)
            enc = freq.to_dict()
        else:  # ordinal
            cats = sorted(X[col].dropna().unique().tolist())
            enc = {c: i for i, c in enumerate(cats)}

        encoding_map[col] = enc
        X[col] = X[col].map(enc).fillna(global_mean if method == "target_mean" else 0)

    return X, encoding_map


def apply_encoding(X: pd.DataFrame, encoding_map: dict, fill_value: float = 0.0) -> pd.DataFrame:
    """Apply an encoding map (produced by ``encode_high_cardinality``) to new data.

    Parameters
    ----------
    X : pd.DataFrame
    encoding_map : dict  (from ``encode_high_cardinality``)
    fill_value : float
        Value for unseen categories.

    Returns
    -------
    pd.DataFrame  (copy)
    """
    X = X.copy()
    for col, enc in encoding_map.items():
        if col not in X.columns:
            continue
        if isinstance(enc, dict):
            X[col] = X[col].map(enc).fillna(fill_value)
        elif hasattr(enc, "__getitem__"):
            X[col] = X[col].map(lambda v, e=enc: e.get(v, fill_value))
    return X


# ---------------------------------------------------------------------------
# Internal helper: smoothed target-mean encoding
# ---------------------------------------------------------------------------


def _target_mean_encode(
    series: pd.Series,
    y: np.ndarray,
    global_mean: float,
    min_samples_leaf: int,
    smoothing: float,
) -> dict:
    """Smoothed target-mean encoding (Micci-Barreca, 2001)."""
    df = pd.DataFrame({"cat": series.values, "y": y})
    agg = df.groupby("cat")["y"].agg(["mean", "count"])
    smoother = 1.0 / (1.0 + np.exp(-(agg["count"] - min_samples_leaf) / smoothing))
    smoothed_mean = smoother * agg["mean"] + (1 - smoother) * global_mean
    return smoothed_mean.to_dict()

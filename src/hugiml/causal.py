"""Causal treatment-effect estimation with HUGIML.

The binary-treatment T-learner uses one shared supervised adaptive vocabulary
(numeric cutpoints and feature semantics) with independently weighted, mined,
and fitted HUG outcome models for the control and treatment arms.

Design
------
1. Use pooled pre-treatment covariates Z and outcome y to select the ordinary
   HUG adaptive numeric vocabulary once.
2. Freeze that vocabulary for both treatment arms.
3. Within each arm, let HUG recompute item utilities, support, pattern mining,
   AUC-grid model selection, and LR/RPTE fitting normally.
4. Estimate potential-outcome probabilities m0(z), m1(z) and treatment effect
   tau(z) = m1(z) - m0(z).

Causal validity still requires the usual identification assumptions.  In an
observational study, the supplied covariates must be an appropriate adjustment
set (plus any pre-treatment precision/effect-modifier variables the analyst
intends to include).  This overlay does not discover a DAG or remove hidden
confounding.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from hugiml import HUGIMLClassifier


@dataclasses.dataclass(frozen=True)
class CrossFittedEffectSummary:
    """Repeated cross-fitted outcome-regression and AIPW effect summaries."""

    estimates: pd.DataFrame
    overlap_sensitivity: pd.DataFrame


def _repeat_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, math.nan, math.nan
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, math.nan, math.nan, math.nan
    sd = float(np.std(values, ddof=1))
    critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
    }.get(len(values) - 1, 1.96)
    half_width = critical * sd / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width, sd


def summarize_cross_fitted_effects(
    y: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    outcome_predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    propensity: np.ndarray,
    *,
    trim_thresholds: Sequence[float] = (0.0, 0.01, 0.05, 0.10),
    propensity_clip: float = 0.01,
) -> CrossFittedEffectSummary:
    """Calculate repeated out-of-fold potential-outcome and AIPW estimates.

    Prediction and propensity arrays have shape ``(n_repeats, n_rows)``.
    Confidence intervals measure variation across complete cross-fitting runs.
    """
    y_arr = np.asarray(y, dtype=float)
    t_arr = np.asarray(treatment, dtype=int)
    e = np.asarray(propensity, dtype=float)
    if y_arr.ndim != 1 or len(y_arr) == 0 or len(t_arr) != len(y_arr):
        raise ValueError(
            "y and treatment must be non-empty one-dimensional arrays of equal length."
        )
    if set(np.unique(t_arr)) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")
    if e.ndim != 2 or e.shape[1] != len(y_arr) or not np.isfinite(e).all():
        raise ValueError("propensity must be a finite (n_repeats, n_rows) array.")
    if not 0 < float(propensity_clip) < 0.5:
        raise ValueError("propensity_clip must be between 0 and 0.5.")
    e = np.clip(e, propensity_clip, 1.0 - propensity_clip)
    estimate_rows, overlap_rows = [], []
    for model_name, pair in outcome_predictions.items():
        mu0, mu1 = np.asarray(pair[0], dtype=float), np.asarray(pair[1], dtype=float)
        if mu0.shape != e.shape or mu1.shape != e.shape:
            raise ValueError(f"{model_name} predictions must match propensity shape {e.shape}.")
        if not np.isfinite(mu0).all() or not np.isfinite(mu1).all():
            raise ValueError(f"{model_name} predictions contain non-finite values.")
        mu0, mu1 = np.clip(mu0, 1e-6, 1 - 1e-6), np.clip(mu1, 1e-6, 1 - 1e-6)
        plug_in = np.mean(mu1 - mu0, axis=1)
        dr = (
            (mu1 - mu0)
            + t_arr[None, :] * (y_arr[None, :] - mu1) / e
            - (1 - t_arr[None, :]) * (y_arr[None, :] - mu0) / (1 - e)
        )
        dr_estimates = np.mean(dr, axis=1)
        plug_mean, plug_low, plug_high, plug_sd = _repeat_interval(plug_in)
        dr_mean, dr_low, dr_high, dr_sd = _repeat_interval(dr_estimates)
        estimate_rows.append(
            {
                "Model": model_name,
                "Repeated OOF ATE": plug_mean,
                "ATE 95% CI low": plug_low,
                "ATE 95% CI high": plug_high,
                "ATE repeat SD": plug_sd,
                "Cross-fitted DR ATE": dr_mean,
                "DR 95% CI low": dr_low,
                "DR 95% CI high": dr_high,
                "DR repeat SD": dr_sd,
                "Repeats": int(e.shape[0]),
            }
        )
        for threshold in trim_thresholds:
            threshold = float(threshold)
            if not 0 <= threshold < 0.5:
                raise ValueError("trim thresholds must be between 0 and 0.5.")
            per_repeat, retained = [], []
            for repeat in range(e.shape[0]):
                mask = (e[repeat] >= threshold) & (e[repeat] <= 1.0 - threshold)
                retained.append(float(np.mean(mask)))
                per_repeat.append(float(np.mean(dr[repeat, mask])) if np.any(mask) else math.nan)
            mean, low, high, sd = _repeat_interval(np.asarray(per_repeat))
            overlap_rows.append(
                {
                    "Model": model_name,
                    "Propensity range": "Untrimmed"
                    if threshold == 0
                    else f"{threshold:.2f}–{1 - threshold:.2f}",
                    "Retained": float(np.mean(retained)),
                    "Cross-fitted DR ATE": mean,
                    "95% CI low": low,
                    "95% CI high": high,
                    "Repeat SD": sd,
                }
            )
    return CrossFittedEffectSummary(pd.DataFrame(estimate_rows), pd.DataFrame(overlap_rows))


@dataclasses.dataclass(frozen=True)
class SharedSupervisedVocabulary:
    """Frozen adaptive HUG vocabulary shared by the two treatment arms.

    The vocabulary stores representation semantics only.  Outcome-dependent
    item weights/utilities, supports, mined patterns, LR coefficients, and RPTE
    structures are deliberately *not* shared.
    """

    feature_names: tuple[str, ...]
    bin_edges: dict[str, np.ndarray]
    attrs: dict[str, Any]
    builder_params: dict[str, Any]

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        y: Sequence[int] | np.ndarray,
        *,
        builder_params: Mapping[str, Any] | None = None,
    ) -> SharedSupervisedVocabulary:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SharedSupervisedVocabulary.fit currently requires a pandas DataFrame.")
        params = dict(builder_params or {})
        params.setdefault("adaptive_binning", True)
        params.setdefault("feature_mode", "original_plus_patterns")
        params.setdefault("convert_binary_to_categorical", False)
        # Vocabulary selection is independent of G/L/topK under ordinary
        # adaptive binning.  L=1 avoids pair-aware adaptive paths if callers
        # supplied unusual defaults.
        params.setdefault("L", 1)
        params.setdefault("G", 0.001)
        params.setdefault("topK", 50)
        params.setdefault("execution_mode", "production")

        builder = HUGIMLClassifier(**params)
        builder._validate_params()
        builder._resolve_col_meta(X)
        y_arr = builder._safe_cast_y(y)
        if len(y_arr) != len(X):
            raise ValueError("X and y have different row counts.")
        # Select adaptive bins with the classifier's configured binning routine.
        builder._apply_adaptive_binning_cpp(X, y_arr)

        attr_names = (
            "cat_cols_mask_",
            "is_int_mask_",
            "feature_names_in_",
            "binary_categorical_cols_",
            "_bin_edges_",
            "_missing_col_edges_",
            "_adaptive_code_label_map_",
            "_adaptive_precoded_features_",
            "per_feature_b_",
            "ig_scores_",
        )
        attrs: dict[str, Any] = {}
        for name in attr_names:
            if hasattr(builder, name):
                attrs[name] = _deepish_copy(getattr(builder, name))
        edges = {
            str(k): np.asarray(v, dtype=float).copy()
            for k, v in getattr(builder, "_bin_edges_", {}).items()
        }
        return cls(
            feature_names=tuple(str(c) for c in X.columns),
            bin_edges=edges,
            attrs=attrs,
            builder_params=params,
        )

    def prebin(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the frozen numeric cutpoints using HUG's integer-code semantics."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Shared vocabulary application currently requires a pandas DataFrame.")
        missing = [c for c in self.feature_names if c not in X.columns]
        extra = [c for c in X.columns if str(c) not in set(self.feature_names)]
        if missing:
            raise ValueError(f"Missing vocabulary columns: {missing}")
        # Enforce the fitted schema and stable feature lineage across arms.
        if extra:
            raise ValueError(f"Unexpected columns not present in shared vocabulary: {extra}")
        Xp = X.loc[:, list(self.feature_names)].copy()
        for name, edges in self.bin_edges.items():
            if name not in Xp.columns:
                continue
            col = pd.to_numeric(Xp[name], errors="coerce").to_numpy(dtype=float)
            n_bins = max(1, len(edges) - 1)
            codes = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1).astype(float)
            codes[~np.isfinite(col)] = np.nan
            Xp[name] = codes
        return Xp

    def context(self, X: pd.DataFrame) -> dict[str, Any]:
        attrs = {k: _deepish_copy(v) for k, v in self.attrs.items()}
        # Apply the canonical frozen edges and code-label mappings.
        attrs["_bin_edges_"] = {k: v.copy() for k, v in self.bin_edges.items()}
        attrs["per_feature_b_"] = {k: len(v) - 1 for k, v in self.bin_edges.items()}
        attrs["_adaptive_precoded_features_"] = set(self.bin_edges)
        attrs["_adaptive_code_label_map_"] = _make_code_label_map(self.bin_edges)
        return {"X_pre": self.prebin(X), "attrs": attrs}

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "bin_edges": {k: v.tolist() for k, v in self.bin_edges.items()},
            "builder_params": _json_safe(self.builder_params),
        }


class _ProbabilityCalibrator:
    def __init__(self, kind: str):
        kind = str(kind).lower()
        if kind not in {"sigmoid", "isotonic"}:
            raise ValueError("calibration must be None, 'sigmoid', or 'isotonic'.")
        self.kind = kind
        self.model: Any = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> _ProbabilityCalibrator:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(y, dtype=int)
        if np.unique(y).size < 2:
            self.model = None
            return self
        if self.kind == "sigmoid":
            logits = np.log(p / (1 - p)).reshape(-1, 1)
            self.model = LogisticRegression(solver="lbfgs").fit(logits, y)
        else:
            self.model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        if self.model is None:
            return p
        if self.kind == "sigmoid":
            logits = np.log(p / (1 - p)).reshape(-1, 1)
            return self.model.predict_proba(logits)[:, 1]
        return np.asarray(self.model.predict(p), dtype=float)


def make_frozen_vocabulary_hug_class(vocabulary: SharedSupervisedVocabulary):
    """Create a HUGIMLClassifier subclass whose adaptive vocabulary is frozen.

    A per-vocabulary dynamic subclass avoids process-global class replacement.  The
    normal HUG fast-grid tuner may instantiate as many candidate models as it
    likes; every candidate sees the same cutpoints while retaining ordinary
    group-level weighting, mining, and fitting.
    """

    class FrozenVocabularyHUGIMLClassifier(HUGIMLClassifier):
        _shared_supervised_vocabulary = vocabulary

        def _install_shared_context(self, X: pd.DataFrame) -> dict[str, Any]:
            ctx = self._shared_supervised_vocabulary.context(X)
            for name, value in ctx["attrs"].items():
                setattr(self, name, _deepish_copy(value))
            return ctx

        def _apply_adaptive_binning_cpp(self, X_train: Any, y_arr: np.ndarray) -> Any:
            del y_arr  # weighting/mining still uses y later; bin selection is frozen.
            return self._install_shared_context(X_train)["X_pre"]

        def _apply_adaptive_binning(self, X_train: Any, y_arr: np.ndarray) -> Any:
            del y_arr
            return self._install_shared_context(X_train)["X_pre"]

        def fit(self, X: Any, y: Any):
            # Supply the pre-binning context so every candidate uses the frozen
            # vocabulary during transaction construction and pattern mining.
            ctx = self._install_shared_context(X)
            self._fast_tune_adaptive_context = ctx
            try:
                return super().fit(X, y)
            finally:
                self.__dict__.pop("_fast_tune_adaptive_context", None)

    FrozenVocabularyHUGIMLClassifier.__name__ = "FrozenVocabularyHUGIMLClassifier"
    return FrozenVocabularyHUGIMLClassifier


class CausalHUGClassifier:
    """Binary-treatment causal T-learner using a shared supervised HUG vocabulary.

    Parameters
    ----------
    treatment:
        Name of the binary treatment column in X.
    covariates:
        Pre-treatment columns supplied to the two outcome models.  If None, all
        columns except treatment are used.  For observational data these should
        contain a valid adjustment set; this class does not identify one.
    param_grid:
        HUG grid used within each arm.  Defaults to ``performance_ho`` so LR vs
        RPTE is selected by ROC-AUC without a user-facing branch choice.
    tuning_fraction:
        Pooled holdout fraction used for grid selection.  The shared supervised
        vocabulary used during tuning is fit on the pooled *tuning-train* rows
        only, avoiding outcome leakage into validation AUC.
    calibration:
        None, ``sigmoid``, or ``isotonic``. Calibration is fitted separately and
        uses a separate pooled calibration holdout.
    calibration_fraction:
        Fraction reserved for calibration when calibration is enabled.
    """

    def __init__(
        self,
        *,
        treatment: str,
        covariates: Sequence[str] | None = None,
        param_grid: str | dict[str, list] = "performance_ho",
        scoring: str = "roc_auc",
        tuning_fraction: float = 0.20,
        calibration: str | None = None,
        calibration_fraction: float = 0.15,
        random_state: int = 42,
        hug_base_params: Mapping[str, Any] | None = None,
        min_arm_rows: int = 100,
        min_arm_events: int = 5,
    ):
        self.treatment = treatment
        self.covariates = None if covariates is None else list(covariates)
        self.param_grid = param_grid
        self.scoring = scoring
        self.tuning_fraction = float(tuning_fraction)
        self.calibration = None if calibration is None else str(calibration).lower()
        self.calibration_fraction = float(calibration_fraction)
        self.random_state = int(random_state)
        self.hug_base_params = dict(hug_base_params or {})
        self.min_arm_rows = int(min_arm_rows)
        self.min_arm_events = int(min_arm_events)

    def fit(self, X: pd.DataFrame, y: Sequence[int] | np.ndarray) -> CausalHUGClassifier:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CausalHUGClassifier.fit requires a pandas DataFrame.")
        if self.treatment not in X.columns:
            raise ValueError(f"Treatment column {self.treatment!r} is not present in X.")
        y_arr = np.asarray(y, dtype=int)
        if len(y_arr) != len(X):
            raise ValueError("X and y have different row counts.")
        t = np.asarray(X[self.treatment], dtype=int)
        if set(np.unique(t)) - {0, 1}:
            raise ValueError("Treatment must be binary and encoded as 0/1.")
        if np.unique(t).size < 2:
            raise ValueError("Both treatment arms are required.")
        if np.unique(y_arr).size < 2:
            raise ValueError("Binary outcome requires both classes in the pooled training data.")

        covariates = self.covariates or [c for c in X.columns if c != self.treatment]
        if not covariates:
            raise ValueError("At least one pre-treatment covariate is required for HUG fitting.")
        if self.treatment in covariates:
            raise ValueError("Treatment must not appear in covariates for the T-HUG design.")
        self.covariates_ = list(covariates)
        Z = X.loc[:, self.covariates_].copy()

        idx_all = np.arange(len(X))
        idx_dev, idx_cal = idx_all, np.array([], dtype=int)
        if self.calibration is not None:
            if self.calibration not in {"sigmoid", "isotonic"}:
                raise ValueError("calibration must be None, 'sigmoid', or 'isotonic'.")
            idx_dev, idx_cal = _joint_stratified_split(
                idx_all, t, y_arr, self.calibration_fraction, self.random_state + 911
            )

        # Leakage-safe grid selection: common vocabulary is learned only from
        # the pooled tune-train rows, never from validation outcomes.
        dev_t, dev_y = t[idx_dev], y_arr[idx_dev]
        idx_fit_rel, idx_val_rel = _joint_stratified_split(
            np.arange(len(idx_dev)), dev_t, dev_y, self.tuning_fraction, self.random_state
        )
        idx_fit = idx_dev[idx_fit_rel]
        idx_val = idx_dev[idx_val_rel]

        tune_vocab = SharedSupervisedVocabulary.fit(
            Z.iloc[idx_fit].reset_index(drop=True),
            y_arr[idx_fit],
            builder_params=self._vocab_builder_params(),
        )
        TuneHUG = make_frozen_vocabulary_hug_class(tune_vocab)

        self.tuning_results_: dict[int, dict[str, Any]] = {}
        self.best_params_: dict[int, dict[str, Any]] = {}
        self.best_scores_: dict[int, float] = {}
        for arm in (0, 1):
            tr = idx_fit[t[idx_fit] == arm]
            va = idx_val[t[idx_val] == arm]
            self._check_arm(y_arr, tr, arm, stage="tuning-train")
            can_auc = len(va) >= 2 and np.unique(y_arr[va]).size == 2
            if can_auc:
                result = TuneHUG.fast_grid_tune(
                    Z.iloc[tr].reset_index(drop=True),
                    y_arr[tr],
                    Z.iloc[va].reset_index(drop=True),
                    y_arr[va],
                    param_grid=self.param_grid,
                    base_params=self._arm_base_params(),
                    scoring=self.scoring,
                    refit_full=False,
                    return_results=True,
                )
                self.tuning_results_[arm] = result
                self.best_params_[arm] = copy.deepcopy(result["best_params"])
                self.best_scores_[arm] = float(result["best_score"])
            else:
                params = _first_lr_candidate(self.param_grid)
                self.tuning_results_[arm] = {
                    "method": "fallback_no_valid_auc",
                    "best_params": copy.deepcopy(params),
                    "best_score": math.nan,
                    "cv_results": [],
                }
                self.best_params_[arm] = copy.deepcopy(params)
                self.best_scores_[arm] = math.nan

        # Final vocabulary uses all development labels; both final arm models
        # then recompute their own utilities/patterns under exactly these bins.
        self.shared_vocabulary_ = SharedSupervisedVocabulary.fit(
            Z.iloc[idx_dev].reset_index(drop=True),
            y_arr[idx_dev],
            builder_params=self._vocab_builder_params(),
        )
        FinalHUG = make_frozen_vocabulary_hug_class(self.shared_vocabulary_)
        self.models_: dict[int, HUGIMLClassifier] = {}
        for arm in (0, 1):
            arm_idx = idx_dev[t[idx_dev] == arm]
            self._check_arm(y_arr, arm_idx, arm, stage="final-fit")
            self.models_[arm] = _fit_exact_candidate(
                FinalHUG,
                Z.iloc[arm_idx].reset_index(drop=True),
                y_arr[arm_idx],
                self.best_params_[arm],
                self._arm_base_params(final=True),
            )

        self.calibrators_: dict[int, _ProbabilityCalibrator | None] = {0: None, 1: None}
        if self.calibration is not None and len(idx_cal):
            for arm in (0, 1):
                ca = idx_cal[t[idx_cal] == arm]
                if len(ca) == 0:
                    continue
                p = self.models_[arm].predict_proba(Z.iloc[ca])[:, 1]
                self.calibrators_[arm] = _ProbabilityCalibrator(self.calibration).fit(p, y_arr[ca])

        self.arm_diagnostics_ = self._build_diagnostics(t, y_arr, idx_dev, idx_cal)
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def _vocab_builder_params(self) -> dict[str, Any]:
        allowed = {
            "allCols",
            "origColumns",
            "b_candidates",
            "min_marginal_gain_ratio",
            "adaptive_binning_sample_frac",
            "adaptive_binning_sample_random_state",
            "n_jobs",
            "use_hotpath",
            "convert_binary_to_categorical",
        }
        params = {k: copy.deepcopy(v) for k, v in self.hug_base_params.items() if k in allowed}
        params["adaptive_binning"] = True
        params.setdefault("convert_binary_to_categorical", False)
        return params

    def _arm_base_params(self, final: bool = False) -> dict[str, Any]:
        params = copy.deepcopy(self.hug_base_params)
        params["adaptive_binning"] = True
        params.setdefault("convert_binary_to_categorical", False)
        params.setdefault("execution_mode", "audit" if final else "production")
        return params

    def _check_arm(self, y: np.ndarray, idx: np.ndarray, arm: int, *, stage: str) -> None:
        n = len(idx)
        events = int(np.sum(y[idx] == 1)) if n else 0
        non_events = n - events
        if n < self.min_arm_rows:
            raise ValueError(
                f"Treatment arm {arm} has only {n} rows in {stage}; minimum is {self.min_arm_rows}."
            )
        if min(events, non_events) < self.min_arm_events:
            raise ValueError(
                f"Treatment arm {arm} has events/non-events={events}/{non_events} in {stage}; "
                f"minimum per outcome class is {self.min_arm_events}."
            )

    def _Z(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "models_"):
            raise RuntimeError("CausalHUGClassifier is not fitted.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Prediction requires a pandas DataFrame.")
        return X.loc[:, self.covariates_].copy()

    def predict_potential_outcomes(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Z = self._Z(X)
        out = []
        for arm in (0, 1):
            p = self.models_[arm].predict_proba(Z)[:, 1]
            cal = self.calibrators_.get(arm)
            if cal is not None:
                p = cal.predict(p)
            out.append(np.asarray(p, dtype=float))
        return out[0], out[1]

    def effect(self, X: pd.DataFrame) -> np.ndarray:
        p0, p1 = self.predict_potential_outcomes(X)
        return p1 - p0

    def cate(self, X: pd.DataFrame) -> np.ndarray:
        return self.effect(X)

    def ate(self, X: pd.DataFrame) -> float:
        return float(np.mean(self.effect(X)))

    @property
    def control_model_(self) -> HUGIMLClassifier:
        return self.models_[0]

    @property
    def treatment_model_(self) -> HUGIMLClassifier:
        return self.models_[1]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([self.arm_diagnostics_[0], self.arm_diagnostics_[1]])

    def pattern_comparison(self) -> pd.DataFrame:
        """Outer-join arm pattern tables using canonical displayed pattern text."""
        frames = []
        for arm in (0, 1):
            model = self.models_[arm]
            try:
                info = model.get_pattern_info()
                if not isinstance(info, pd.DataFrame):
                    info = pd.DataFrame(info)
            except Exception:
                info = pd.DataFrame()
            if info.empty:
                continue
            info = info.copy()
            pattern_col = "pattern" if "pattern" in info.columns else info.columns[0]
            info["canonical_pattern"] = info[pattern_col].astype(str)
            info = info.add_prefix(f"arm{arm}_")
            info = info.rename(columns={f"arm{arm}_canonical_pattern": "canonical_pattern"})
            frames.append(info)
        if not frames:
            return pd.DataFrame(columns=["canonical_pattern"])
        if len(frames) == 1:
            return frames[0]
        return frames[0].merge(frames[1], on="canonical_pattern", how="outer")

    def vocabulary_frame(self) -> pd.DataFrame:
        rows = []
        for feat, edges in self.shared_vocabulary_.bin_edges.items():
            for k in range(len(edges) - 1):
                rows.append({"feature": feat, "bin": k, "left": edges[k], "right": edges[k + 1]})
        return pd.DataFrame(rows)

    def _build_diagnostics(self, t, y, idx_dev, idx_cal):
        d = {}
        for arm in (0, 1):
            dev = idx_dev[t[idx_dev] == arm]
            model = self.models_[arm]
            params = self.best_params_[arm]
            base_est = params.get("base_estimator")
            branch = _downstream_branch(base_est)
            d[arm] = {
                "arm": arm,
                "n_fit": int(len(dev)),
                "events_fit": int(np.sum(y[dev] == 1)),
                "event_rate_fit": float(np.mean(y[dev])) if len(dev) else math.nan,
                "grid_branch": branch,
                "grid_auc": self.best_scores_.get(arm, math.nan),
                "n_patterns": int(len(getattr(model, "patterns_", []))),
                "L": params.get("L"),
                "G": params.get("G"),
                "topK": params.get("topK"),
                "n_calibration": int(np.sum(t[idx_cal] == arm)) if len(idx_cal) else 0,
                "calibration": self.calibration,
            }
        return d


def _fit_exact_candidate(cls, X, y, best_params, base_params):
    params = copy.deepcopy(base_params)
    params.update(copy.deepcopy(best_params))
    # Retain the requested execution mode for fitted causal arm models so
    # region and RPTE evidence is available in audit mode.
    if "execution_mode" in base_params:
        params["execution_mode"] = base_params["execution_mode"]
    lr_C = float(params.pop("lr_C", 1.0))
    model = cls(**params)
    model._benchmark_lr_C = lr_C
    return model.fit(X, y)


def _downstream_branch(base_estimator: Any) -> str:
    """Return the supported downstream family for causal audit summaries."""
    return (
        "LR" if base_estimator is None or isinstance(base_estimator, LogisticRegression) else "RPTE"
    )


def _first_lr_candidate(param_grid: str | dict[str, list]) -> dict[str, Any]:
    grid = (
        HUGIMLClassifier.default_param_grid(param_grid)
        if isinstance(param_grid, str)
        else copy.deepcopy(param_grid)
    )
    candidate: dict[str, Any] = {}
    for key, vals in grid.items():
        vals = list(vals)
        if key == "base_estimator":
            candidate[key] = None if None in vals else vals[0]
        else:
            candidate[key] = vals[0]
    return candidate


def _joint_stratified_split(idx, t, y, test_fraction, random_state):
    if not (0 < float(test_fraction) < 0.5):
        raise ValueError("holdout fractions must be between 0 and 0.5")
    idx = np.asarray(idx, dtype=int)
    labels = np.asarray(t, dtype=int) * 2 + np.asarray(y, dtype=int)
    try:
        a, b = train_test_split(
            idx, test_size=float(test_fraction), random_state=int(random_state), stratify=labels
        )
    except ValueError:
        a, b = train_test_split(
            idx,
            test_size=float(test_fraction),
            random_state=int(random_state),
            stratify=np.asarray(t, dtype=int),
        )
    return np.asarray(a, dtype=int), np.asarray(b, dtype=int)


def _make_code_label_map(bin_edges: Mapping[str, np.ndarray]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, edges in bin_edges.items():
        for k in range(len(edges) - 1):
            out[f"{name}=[{float(k):.3f},{float(k + 1):.3f}]"] = (
                f"{name}=[{edges[k]:.4g},{edges[k + 1]:.4g})"
            )
    return out


def _deepish_copy(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, dict):
        return {k: _deepish_copy(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_deepish_copy(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_deepish_copy(x) for x in v)
    if isinstance(v, set):
        return set(v)
    try:
        return copy.deepcopy(v)
    except Exception:
        return v


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return repr(obj)


__all__ = [
    "CausalHUGClassifier",
    "CrossFittedEffectSummary",
    "SharedSupervisedVocabulary",
    "make_frozen_vocabulary_hug_class",
    "summarize_cross_fitted_effects",
]

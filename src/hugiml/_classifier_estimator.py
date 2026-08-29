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

"""Estimator protocol, lifecycle state, and model persistence."""

from __future__ import annotations

import copy
import os
import threading
from typing import Any

import numpy as np

from hugiml._classifier_support import _PRESETS, HUGIMLClassifier, logger
from hugiml.exceptions import HUGIMLParamError, HUGIMLVersionError
from hugiml.hyperparameter_configs import get_hugiml_grid
from hugiml.serialization import MIN_SCHEMA_VERSION, MODEL_SCHEMA_VERSION
from hugiml.serialization import load_model as _load_model
from hugiml.serialization import save_model as _save_model


class _EstimatorMixin:
    """Implement the estimator protocol and lifecycle operations."""

    def _is_production_mode(self) -> bool:
        """Return True when audit/governance-heavy artifacts are not retained."""
        return getattr(self, "execution_mode", "audit") == "production"

    def _audit_artifact_message(self, artifact: str) -> str:
        return (
            f"{artifact} is not available because this model was fitted or loaded with "
            "execution_mode='production'. Refit the model with execution_mode='audit' "
            "or load a model file that was originally saved from audit mode for complete "
            "traceability and audit/governance artifacts."
        )

    def _require_audit_artifact(self, artifact: str, *required_attrs: str) -> None:
        """Raise a clear error when an audit/governance artifact is unavailable.

        Passing no required attributes is treated as an unconditional audit-only
        guard in production mode.  When required attributes are supplied, their
        presence is checked in every execution mode: production receives the
        governance-oriented retention message, while audit mode receives a
        fitted-state/corrupt-state message.  This keeps callers such as
        ``get_pattern_info()`` and drift helpers from falling through to an
        ``AttributeError`` if an expected audit artifact is absent.
        """
        is_prod = self._is_production_mode()
        if not required_attrs:
            if is_prod:
                raise RuntimeError(self._audit_artifact_message(artifact))
            return
        missing = [
            attr
            for attr in required_attrs
            if (not hasattr(self, attr)) or getattr(self, attr, None) is None
        ]
        if missing:
            if is_prod:
                raise RuntimeError(self._audit_artifact_message(artifact))
            raise RuntimeError(
                f"{artifact} is unavailable because required fitted artifact(s) "
                f"are missing: {', '.join(missing)}. Refit the model or reload a "
                "complete audit-mode model file."
            )

    def _apply_execution_mode_retention(self) -> None:
        """Drop audit/governance-heavy training artifacts in production mode.

        Prediction-critical state is retained: td_, patterns_, model_, bin/scaler
        metadata, selected downstream names/masks, augmented-pair transform
        metadata, and privacy-safe aggregate downstream metadata that was
        already cached before retention. Training matrices, drift baselines, and
        native-only transient score caches are audit/governance artifacts and are
        intentionally omitted in production mode.
        """
        if not self._is_production_mode():
            return
        x_hup = getattr(self, "x_train_hup_", None)
        if x_hup is not None:
            self._training_pattern_matrix_shape_ = tuple(int(v) for v in x_hup.shape)
            self._training_pattern_matrix_nnz_ = int(getattr(x_hup, "nnz", 0))
        x_down = getattr(self, "x_train_downstream_", None)
        if x_down is not None:
            self._training_downstream_matrix_shape_ = tuple(int(v) for v in x_down.shape)
            if hasattr(x_down, "nnz"):
                self._training_downstream_matrix_nnz_ = int(x_down.nnz)
            elif hasattr(x_down, "shape"):
                self._training_downstream_matrix_nnz_ = int(np.count_nonzero(x_down))
            else:
                self._training_downstream_matrix_nnz_ = 0
        for attr in (
            "x_train_hup_",
            "x_train_downstream_",
            # _downstream_* aggregate metadata is retained: it is aligned to
            # downstream feature names and lets feature_importances() report a
            # stable schema after production retention without needing training
            # matrices.
            "_native_original_feature_scores_downstream_",
            "_drift_det",
        ):
            self.__dict__.pop(attr, None)

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> HUGIMLClassifier:
        """Create a classifier from a named configuration preset.

        Parameters
        ----------
        name : {'quick', 'balanced', 'thorough'}
            quick     — B=5, L=1, G=1e-2, topK=50
            balanced  — B=7, L=1, G=5e-3, topK=-1
            thorough  — B=-1, L=2, G=1e-4, topK=-1

        Returns
        -------
        HUGIMLClassifier
        """
        if name not in _PRESETS:
            raise HUGIMLParamError(f"Unknown preset '{name}'.  Available: {list(_PRESETS)}")
        params = {**_PRESETS[name], **overrides}
        return cls(**params)

    @classmethod
    def default_param_grid(cls, name: str | None = None) -> dict[str, list]:
        """Return a named validation grid for compact HUGIML tuning.

        ``name`` selects ``performance_ho``, ``performance``,
        ``interpretability_ho``, or ``interpretability``. When omitted,
        ``performance_ho`` is returned. Grid definitions live in
        :mod:`hugiml.hyperparameter_configs` and are copied before return so
        callers can narrow candidate values locally without mutating the
        shared definitions.

        The two performance grids use ``feature_mode='original_plus_patterns'``;
        ``performance_ho`` additionally searches the adaptive RPTE downstream
        branch. The two interpretability grids use
        ``feature_mode='patterns_only'``,
        ``interaction_relaxed_mining=True``, and
        ``augmented_pair_transforms=False``;
        ``interpretability_ho`` additionally searches sequential RPTE while
        explicitly keeping lookahead inactive.
        """
        return get_hugiml_grid(name)

    def __repr__(self) -> str:
        fitted = hasattr(self, "patterns_")
        status = f", {len(self.patterns_)} patterns" if fitted else ", not fitted"
        adap = ", adaptive" if self.adaptive_binning else ""
        mode = f", feature_mode={self.feature_mode}"
        aug = f", augmented_pair_transforms={self.augmented_pair_transforms}"
        exec_mode = f", execution_mode={self.execution_mode}"
        return f"HUGIMLClassifier(B={self.B}, L={self.L}, G={self.G}{adap}{mode}{aug}{exec_mode}{status})"

    def get_params(self, deep: bool = True) -> dict:
        """Return constructor parameters as a dict (sklearn protocol)."""
        return dict(
            allCols=self.allCols,
            origColumns=self.origColumns,
            B=self.B,
            L=self.L,
            G=self.G,
            topK=self.topK,
            base_estimator=(copy.deepcopy(self.base_estimator) if deep else self.base_estimator),
            lr_solver=self.lr_solver,
            n_jobs=self.n_jobs,
            max_predict_ms=self.max_predict_ms,
            max_fit_seconds=self.max_fit_seconds,
            max_mining_seconds=self.max_mining_seconds,
            mining_degradation_policy=self.mining_degradation_policy,
            verbose=self.verbose,
            adaptive_binning=self.adaptive_binning,
            b_candidates=self.b_candidates,
            min_marginal_gain_ratio=self.min_marginal_gain_ratio,
            adaptive_binning_sample_frac=self.adaptive_binning_sample_frac,
            adaptive_binning_sample_random_state=self.adaptive_binning_sample_random_state,
            convert_binary_to_categorical=self.convert_binary_to_categorical,
            feature_mode=self.feature_mode,
            lr_source_policy=self.lr_source_policy,
            use_hotpath=self.use_hotpath,
            augmented_pair_transforms=self.augmented_pair_transforms,
            augmented_pair_mode=self.augmented_pair_mode,
            ii_partner_size=self.ii_partner_size,
            aug_feature_size=self.aug_feature_size,
            max_pair_features=self.max_pair_features,
            augmented_pair_max_features=self.augmented_pair_max_features,
            topk_budget_strict=self.topk_budget_strict,
            dense_downstream_max_width=self.dense_downstream_max_width,
            execution_mode=self.execution_mode,
            interaction_relaxed_mining=self.interaction_relaxed_mining,
            interaction_relaxed_feature_size=self.interaction_relaxed_feature_size,
        )

    def set_params(self, **params: Any) -> HUGIMLClassifier:
        """Set constructor parameters in-place and return self (sklearn protocol)."""
        # Defer validation until fit(), matching sklearn estimator conventions.
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _more_tags(self) -> dict:
        return {
            "requires_y": True,
            "binary_only": False,
            "poor_score": False,
            "X_types": ["2darray", "dataframe"],
            "allow_nan": False,
        }

    def __sklearn_tags__(self) -> Any:
        """Declare sklearn 1.6+ Tags, including TransformerTags for transform().

        ``__sklearn_tags__`` was introduced in sklearn 1.6.  Base classes on
        older installations do not implement it, so ``super().__sklearn_tags__()``
        raises ``AttributeError``.  Guard that call and return ``None`` when the
        parent chain does not support the protocol — callers must handle ``None``.
        """
        try:
            tags = super().__sklearn_tags__()
        except AttributeError:
            logger.debug(
                "super().__sklearn_tags__() raised AttributeError; "
                "sklearn base classes do not implement the tag protocol "
                "(expected sklearn >= 1.6).",
            )
            return None
        try:
            from sklearn.utils._tags import TransformerTags

            tags.transformer_tags = TransformerTags()
        except ImportError:
            logger.debug(
                "sklearn.utils._tags.TransformerTags not available; "
                "TransformerTags will not be declared.",
                exc_info=True,
            )
        return tags

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_fit_lock", None)
        state.pop("_native_augmented_pair_cache_", None)
        state.pop("_native_augmented_pair_score_topK_", None)
        # Remove instance-level methods set by instrument_classifier():\n        # these closures are not picklable.
        state.pop("predict_proba", None)
        state.pop("predict", None)
        state["_schema_version_"] = MODEL_SCHEMA_VERSION
        if "patterns_" in state and state["patterns_"]:
            state["patterns_"] = [
                {"utility": pe.utility, "items": list(pe.items), "ig": pe.ig}
                for pe in state["patterns_"]
            ]
            state["_patterns_pickled_"] = True
        # serialize raw_patterns_ (also holds PatternEntry objects) ──
        # __getstate__ must convert patterns_ and raw_patterns_ consistently as
        # native PatternEntry objects, which are not picklable/deepcopyable.
        # Mirror the same dict-serialisation used for patterns_.
        if "raw_patterns_" in state and state["raw_patterns_"]:
            try:
                state["raw_patterns_"] = [
                    {"utility": pe.utility, "items": list(pe.items), "ig": pe.ig}
                    for pe in state["raw_patterns_"]
                ]
                state["_raw_patterns_pickled_"] = True
            except (AttributeError, TypeError):
                # Already serialized (dicts) or empty — leave as-is
                pass
        return state

    def __setstate__(self, state: dict) -> None:
        schema_ver = state.pop("_schema_version_", 1)
        if schema_ver < MIN_SCHEMA_VERSION:
            raise HUGIMLVersionError(
                f"Model schema version {schema_ver} is too old.  "
                f"Minimum supported: {MIN_SCHEMA_VERSION}.  Re-fit the model."
            )

        if state.pop("_patterns_pickled_", False):

            class _PE:
                __slots__ = ("utility", "items", "ig")

                def __init__(self, d: dict) -> None:
                    self.utility = d["utility"]
                    self.items = d["items"]
                    self.ig = d["ig"]

            state["patterns_"] = [_PE(d) for d in state["patterns_"]]

        # restore raw_patterns_ from its serialized dict form ──────
        if state.pop("_raw_patterns_pickled_", False):

            class _PE2:
                __slots__ = ("utility", "items", "ig")

                def __init__(self, d: dict) -> None:
                    self.utility = d["utility"]
                    self.items = d["items"]
                    self.ig = d["ig"]

            if "raw_patterns_" in state and state["raw_patterns_"]:
                state["raw_patterns_"] = [_PE2(d) for d in state["raw_patterns_"]]

        self.__dict__.update(state)
        # Drop unsupported multi-round attributes when loading serialized estimators.
        for _attr in (
            "n_rounds",
            "g_decay_factor",
            "pattern_selection",
            "transaction_weighting",
            "_boosting_round_tds_",
            "_boosting_round_pats_",
        ):
            self.__dict__.pop(_attr, None)
        self._fit_lock = threading.RLock()

        # ── Backward compatibility ───────────────────────────────────────
        # Models saved with v1.1.13 do not have adaptive-sampling state in
        # their pickle payload. Fill in missing attributes so deserialized
        # models keep their established behavior after restore.
        if not hasattr(self, "adaptive_binning"):
            self.adaptive_binning = False
        if not hasattr(self, "adaptive_binning_sample_frac"):
            self.adaptive_binning_sample_frac = False
        if not hasattr(self, "adaptive_binning_sample_random_state"):
            self.adaptive_binning_sample_random_state = 42
        if not hasattr(self, "b_candidates"):
            self.b_candidates = None
        if not hasattr(self, "min_marginal_gain_ratio"):
            self.min_marginal_gain_ratio = 0.02
        if not hasattr(self, "use_hotpath"):
            self.use_hotpath = True
        if not hasattr(self, "augmented_pair_transforms"):
            self.augmented_pair_transforms = True
        if not hasattr(self, "topk_budget_strict"):
            self.topk_budget_strict = False
        if not hasattr(self, "dense_downstream_max_width"):
            self.dense_downstream_max_width = 200
        if not hasattr(self, "execution_mode"):
            self.execution_mode = "audit"
        if not hasattr(self, "lr_solver"):
            self.lr_solver = "auto"
        if not hasattr(self, "max_mining_seconds"):
            # v1.1.x compatibility: older pickles only know max_fit_seconds.
            self.max_mining_seconds = None
        if not hasattr(self, "mining_degradation_policy"):
            self.mining_degradation_policy = "allow"
        if not hasattr(self, "mining_audit_log_"):
            self.mining_audit_log_ = []
        if not hasattr(self, "augmented_pair_mode"):
            self.augmented_pair_mode = "interaction_information"
        if not hasattr(self, "ii_partner_size"):
            self.ii_partner_size = None
        if not hasattr(self, "aug_feature_size"):
            self.aug_feature_size = 10
        if not hasattr(self, "max_pair_features"):
            self.max_pair_features = 10
        if not hasattr(self, "augmented_pair_max_features"):
            self.augmented_pair_max_features = None
        if not hasattr(self, "augmented_pair_transforms_"):
            self.augmented_pair_transforms_ = []
        if not hasattr(self, "augmented_pair_selected_features_"):
            self.augmented_pair_selected_features_ = []
        if not hasattr(self, "_original_feature_mask_downstream_"):
            self._original_feature_mask_downstream_ = None
        if not hasattr(self, "_original_selected_feature_names_downstream_"):
            self._original_selected_feature_names_downstream_ = None
        if not hasattr(self, "binary_categorical_cols_"):
            self.binary_categorical_cols_ = []
        if not hasattr(self, "_strict_topk_applied_during_construction_"):
            self._strict_topk_applied_during_construction_ = False
        if not hasattr(self, "_missing_col_edges_"):
            self._missing_col_edges_ = {}
        if not hasattr(self, "_adaptive_code_label_map_"):
            self._adaptive_code_label_map_ = {}
        if not hasattr(self, "fallback_active_"):
            self.fallback_active_ = False
        if getattr(self, "fallback_active_", False) and not hasattr(self, "fallback_strategy_"):
            self.fallback_strategy_ = "constant_prior"
        # Rebuild the code→label map from stored bin edges whenever it's absent
        # or empty but adaptive bin edges are present.  This handles save/load via
        # both pickle and the custom .hugiml format (serialization.py).
        if (
            self.adaptive_binning
            and not self._adaptive_code_label_map_
            and getattr(self, "_bin_edges_", None)
        ):
            self._rebuild_adaptive_code_label_map()
        # ──────────────────────────────────────────────────────────────────

        if hasattr(self, "td_") and self.td_ is not None:
            td = self.td_
            self._native_available_ = not (hasattr(td, "_td") and td._td is None)
        else:
            self._native_available_ = False

    def save_model(self, path: str | os.PathLike) -> None:
        """Persist the fitted model to a binary file with schema versioning.

        Parameters
        ----------
        path : str or Path

        Raises
        ------
        HUGIMLSerializationError
        """
        _save_model(self, path)

    @classmethod
    def load_model(cls, path: str | os.PathLike) -> HUGIMLClassifier:
        """Load a model saved with :meth:`save_model`.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        HUGIMLClassifier

        Raises
        ------
        HUGIMLVersionError, HUGIMLSerializationError
        """
        return _load_model(path, expected_type=cls)  # type: ignore[no-any-return]

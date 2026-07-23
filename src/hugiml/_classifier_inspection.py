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

"""Pattern inventories, model composition, importance, and summaries."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.utils.validation import check_is_fitted

from hugiml.exceptions import HUGIMLWarning


class _InspectionMixin:
    """Implement fitted-model inspection and visualization operations."""

    def get_hug_features(self) -> list[str]:
        """Return a human-readable label for each mined HUG pattern.

        Singleton patterns use the format ``feature=[lo,hi)`` for adaptive
        numerical columns (e.g. ``age=[35,50)``) and ``feature=value`` for
        categorical columns (e.g. ``gender=F``).  Compound patterns (L > 1)
        are comma-separated, e.g. ``age=[35,50), gender=F``.

        When ``adaptive_binning=True`` and the integer-code path was used,
        C++ stores bin labels as ``feature=[k,k+1]`` (integer range).  These
        are transparently remapped to the original-scale ``[lo,hi)`` labels
        via ``_adaptive_code_label_map_`` so that the output is identical in
        appearance to the string-path output.

        Production mode
        ---------------
        This method remains available in ``execution_mode='production'`` because
        it only needs retained pattern labels.  ``get_pattern_info()`` is
        intentionally audit-only because it additionally needs the retained
        training pattern matrix to compute support.

        Returns
        -------
        list of str
        """
        check_is_fitted(self)
        item_map = self.td_.item_map
        # _adaptive_code_label_map_ is set by _apply_adaptive_binning when the
        # integer-code path is used.  Empty dict (default) → no remapping needed.
        label_remap = getattr(self, "_adaptive_code_label_map_", {})

        def _resolve_item(item_id: int) -> str:
            raw_label = item_map.get(item_id, str(item_id))
            return label_remap.get(raw_label, raw_label)

        return [", ".join(_resolve_item(it) for it in pe.items) for pe in self.patterns_]

    @staticmethod
    def _feature_name_from_item_label(label: str) -> str:
        """Return the source feature name from a rendered transaction item."""
        text = str(label or "")
        if "=" not in text:
            return text.strip()
        return text.split("=", 1)[0].strip()

    def _survivor_audit_lookup(self) -> dict[str, dict[str, object]]:
        """Return interaction-relaxed survivor metadata keyed by feature name."""
        if not bool(getattr(self, "interaction_relaxed_mining", False)):
            return {}
        rows = list(getattr(self, "interaction_relaxed_mining_survivors_", []) or [])
        lookup: dict[str, dict[str, object]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            lookup[name] = dict(row)
        return lookup

    def _pattern_source_features(self, pattern_entry: Any) -> list[str]:
        """Return source feature names used by a native pattern entry."""
        item_map = getattr(getattr(self, "td_", None), "item_map", {}) or {}
        label_remap = getattr(self, "_adaptive_code_label_map_", {}) or {}
        names: list[str] = []
        seen: set[str] = set()
        for item_id in getattr(pattern_entry, "items", []) or []:
            raw_label = item_map.get(int(item_id), str(item_id))
            label = label_remap.get(raw_label, raw_label)
            feature_name = self._feature_name_from_item_label(str(label))
            if feature_name and feature_name not in seen:
                seen.add(feature_name)
                names.append(feature_name)
        return names

    @staticmethod
    def _finite_float_or_nan(value: Any) -> float:
        """Convert a scalar to float, using NaN for missing/non-finite values."""
        try:
            out = float(value)
        except Exception:
            return float("nan")
        return out if np.isfinite(out) else float("nan")

    def _pattern_survivor_audit(self, pattern_entry: Any) -> dict[str, object]:
        """Return survivor-led provenance for one mined pattern."""
        survivor_lookup = self._survivor_audit_lookup()
        source_features = self._pattern_source_features(pattern_entry)
        survivor_features = [name for name in source_features if name in survivor_lookup]
        survivor_rows = [survivor_lookup[name] for name in survivor_features]
        marginal_igs = [
            self._finite_float_or_nan(row.get("marginal_ig", np.nan)) for row in survivor_rows
        ]
        interaction_scores = [
            self._finite_float_or_nan(row.get("interaction_score", row.get("score", np.nan)))
            for row in survivor_rows
        ]
        best_partners: list[str] = []
        for row in survivor_rows:
            partner = str(row.get("best_partner", "")).strip()
            if partner and partner not in best_partners:
                best_partners.append(partner)
        finite_marginal = [v for v in marginal_igs if np.isfinite(v)]
        finite_interaction = [v for v in interaction_scores if np.isfinite(v)]
        survivor_led = bool(survivor_features)
        return {
            "pattern_origin": "interaction_relaxed" if survivor_led else "standard",
            "survivor_led": survivor_led,
            "survivor_features": survivor_features,
            "survivor_feature_count": len(survivor_features),
            "survivor_min_marginal_ig": min(finite_marginal) if finite_marginal else np.nan,
            "survivor_max_interaction_score": max(finite_interaction)
            if finite_interaction
            else np.nan,
            "survivor_best_partners": best_partners,
        }

    def _pattern_audit_lookup_by_label(self) -> dict[str, dict[str, object]]:
        """Return survivor-led metadata keyed by raw and namespaced labels."""
        labels = self.get_hug_features()
        lookup: dict[str, dict[str, object]] = {}
        for label, pe in zip(labels, getattr(self, "patterns_", []) or []):
            audit = self._pattern_survivor_audit(pe)
            lookup[label] = audit
            lookup[f"pattern:{label}"] = audit
        return lookup

    def _pattern_origin_counts_for_downstream_features(
        self, features: list[str] | None = None
    ) -> dict[str, int]:
        """Return origin counts for downstream pattern features."""
        names = list(features or self._get_downstream_feature_names())
        lookup = self._pattern_audit_lookup_by_label()
        counts = {"standard": 0, "interaction_relaxed": 0, "survivor_led": 0}
        for feat in names:
            if self._downstream_feature_type(str(feat)) != "pattern":
                continue
            display_name = self._downstream_feature_display_name(str(feat))
            audit = lookup.get(str(feat), lookup.get(display_name, {}))
            origin = str(audit.get("pattern_origin", "standard"))
            if origin not in counts:
                counts[origin] = 0
            counts[origin] += 1
            if bool(audit.get("survivor_led", False)):
                counts["survivor_led"] += 1
        counts["total_pattern_features"] = sum(
            1 for feat in names if self._downstream_feature_type(str(feat)) == "pattern"
        )
        return counts

    def get_transformed_shape(self) -> tuple[int, int]:
        """Return (n_samples, n_patterns) for the training pattern matrix.

        In production mode the matrix itself is not retained, but its shape is
        persisted as lightweight diagnostic metadata.
        """
        check_is_fitted(self)
        if hasattr(self, "x_train_hup_"):
            shape = self.x_train_hup_.shape
            return int(shape[0]), int(shape[1])
        cached = getattr(self, "_training_pattern_matrix_shape_", None)
        if cached is not None:
            return int(cached[0]), int(cached[1])
        raise RuntimeError(self._audit_artifact_message("Training pattern matrix shape"))

    def get_pattern_info(self) -> pd.DataFrame:
        """Summary DataFrame with one row per mined HUG pattern.

        Columns include pattern, utility, information_gain, support, and
        survivor-led provenance when interaction-relaxed mining is enabled.

        This is an audit/governance table.  Unlike ``get_hug_features()``, it
        requires the retained training pattern matrix to compute support and
        therefore raises a clear error in ``execution_mode='production'``.
        """
        check_is_fitted(self)
        self._require_audit_artifact("Pattern support and pattern-info audit table", "x_train_hup_")
        n_train = self.x_train_hup_.shape[0]
        features = self.get_hug_features()
        records: list[dict[str, object]] = []
        for i, pe in enumerate(self.patterns_):
            support = float(self.x_train_hup_[:, i].sum()) / n_train
            audit = self._pattern_survivor_audit(pe)
            records.append(
                {
                    "pattern": features[i],
                    "utility": round(pe.utility, 6),
                    "information_gain": round(pe.ig, 6),
                    "support": round(support, 4),
                    **audit,
                }
            )
        return pd.DataFrame(records)

    def _downstream_feature_display_name(self, name: str) -> str:
        """Return a compact display label for a downstream feature name."""
        for prefix in ("orig:", "pattern:", "augmented_pair:"):
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    def _downstream_feature_type(self, name: str) -> str:
        """Classify a downstream feature name by its explicit namespace."""
        if name.startswith("orig:"):
            return "original"
        if name.startswith("augmented_pair:"):
            return "augmented_pair"
        return "pattern"

    def _pattern_support_lookup(self) -> dict[str, float]:
        """Return training support by both raw and namespaced pattern label.

        Production-mode models intentionally drop the training HUG matrix.
        ``feature_importances()`` should prefer cached support metadata when it
        exists; this lookup is only a best-effort recomputation path for audit
        models or legacy objects that still retain ``x_train_hup_``.
        """
        if not hasattr(self, "x_train_hup_"):
            return {}
        n_train = int(self.x_train_hup_.shape[0])
        if n_train <= 0:
            return {}
        labels = self.get_hug_features()
        support = np.asarray(self.x_train_hup_.sum(axis=0)).ravel() / max(n_train, 1)
        lookup: dict[str, float] = {}
        for label, value in zip(labels, support):
            val = float(value)
            lookup[label] = val
            lookup[f"pattern:{label}"] = val
        return lookup

    def get_downstream_features(self) -> list[str]:
        """Return names aligned with the downstream estimator input columns.

        The returned names include a namespace prefix so feature provenance is
        explicit: ``orig:`` for original features, ``pattern:`` for mined HUG
        patterns, and ``augmented_pair:`` for augmented pair transforms.  When
        ``topk_budget_strict=True``, the returned list is already filtered to
        the columns retained by the fitted strict TopK mask.
        """
        check_is_fitted(self)
        return list(self._get_downstream_feature_names())

    def _downstream_feature_counts(self) -> dict[str, int]:
        """Return counts by downstream feature family for the fitted estimator."""
        names = list(
            getattr(self, "_downstream_feature_names_", []) or self._get_downstream_feature_names()
        )
        counts = {
            "original": sum(1 for name in names if str(name).startswith("orig:")),
            "pattern": sum(1 for name in names if str(name).startswith("pattern:")),
            "augmented_pair": sum(1 for name in names if str(name).startswith("augmented_pair:")),
        }
        counts["total"] = len(names)
        return counts

    def get_model_composition(self) -> dict[str, Any]:
        """Return downstream feature composition and relevant fitted configuration.

        The composition describes the actual feature families entering the
        downstream estimator after feature-mode construction and optional
        strict TopK filtering.
        """
        check_is_fitted(self)
        counts = self._downstream_feature_counts()
        aug_config = dict(getattr(self, "augmented_pair_config_", {}) or {})
        return {
            "feature_mode": getattr(self, "feature_mode", "patterns_only"),
            "topK": getattr(self, "topK", None),
            "topk_budget_strict": bool(getattr(self, "topk_budget_strict", False)),
            "augmented_pair_transforms_enabled": bool(
                getattr(self, "augmented_pair_transforms", False)
            ),
            "augmented_pair_config": aug_config,
            "n_input_features": int(getattr(self, "n_features_in_", 0)),
            "n_patterns_mined": int(len(getattr(self, "patterns_", []))),
            "n_downstream_features": counts["total"],
            "downstream_feature_counts": counts,
            "pattern_origin_counts": self._pattern_origin_counts_for_downstream_features(),
        }

    def _cache_downstream_feature_metadata(self) -> None:
        """Cache metadata aligned with the fitted downstream feature matrix.

        This method must run before production retention because production
        mode drops the training matrices.  It is intentionally best-effort:
        pattern support can be unavailable for cached/tuned candidates, but
        non-missing-rate and variance can still be computed from the fitted
        downstream matrix before it is discarded.
        """
        features = self._get_downstream_feature_names()
        self._downstream_feature_names_ = list(features)
        n_features = len(features)

        self._downstream_pattern_support_ = np.full(n_features, np.nan, dtype=np.float64)
        self._downstream_pattern_origin_ = np.full(n_features, "", dtype=object)
        self._downstream_survivor_led_ = np.zeros(n_features, dtype=np.bool_)
        self._downstream_survivor_features_ = [list() for _ in range(n_features)]
        self._downstream_survivor_feature_count_ = np.zeros(n_features, dtype=np.int64)
        self._downstream_survivor_min_marginal_ig_ = np.full(n_features, np.nan, dtype=np.float64)
        self._downstream_survivor_max_interaction_score_ = np.full(
            n_features, np.nan, dtype=np.float64
        )
        self._downstream_survivor_best_partners_ = [list() for _ in range(n_features)]
        try:
            support_lookup = self._pattern_support_lookup()
        except Exception:
            support_lookup = {}
        try:
            audit_lookup = self._pattern_audit_lookup_by_label()
        except Exception:
            audit_lookup = {}
        for idx, feat in enumerate(features):
            if self._downstream_feature_type(feat) == "pattern":
                display_name = self._downstream_feature_display_name(feat)
                self._downstream_pattern_support_[idx] = support_lookup.get(
                    feat, support_lookup.get(display_name, np.nan)
                )
                audit = audit_lookup.get(feat, audit_lookup.get(display_name, {}))
                self._downstream_pattern_origin_[idx] = str(audit.get("pattern_origin", "standard"))
                self._downstream_survivor_led_[idx] = bool(audit.get("survivor_led", False))
                survivor_features = list(audit.get("survivor_features", []) or [])
                self._downstream_survivor_features_[idx] = survivor_features
                self._downstream_survivor_feature_count_[idx] = int(
                    audit.get("survivor_feature_count", len(survivor_features))
                )
                self._downstream_survivor_min_marginal_ig_[idx] = self._finite_float_or_nan(
                    audit.get("survivor_min_marginal_ig", np.nan)
                )
                self._downstream_survivor_max_interaction_score_[idx] = self._finite_float_or_nan(
                    audit.get("survivor_max_interaction_score", np.nan)
                )
                self._downstream_survivor_best_partners_[idx] = list(
                    audit.get("survivor_best_partners", []) or []
                )

        X_meta = getattr(self, "x_train_downstream_", None)
        if X_meta is not None and n_features == getattr(X_meta, "shape", (0, 0))[1]:
            X_arr = X_meta.toarray() if issparse(X_meta) else np.asarray(X_meta)
            finite_mask = np.isfinite(X_arr)
            self._downstream_non_missing_rate_ = finite_mask.mean(axis=0).astype(np.float64)
            self._downstream_variance_ = np.nanvar(
                np.where(finite_mask, X_arr, np.nan), axis=0
            ).astype(np.float64)
        else:
            self._downstream_non_missing_rate_ = np.full(n_features, np.nan, dtype=np.float64)
            self._downstream_variance_ = np.full(n_features, np.nan, dtype=np.float64)

    def feature_importances(self) -> pd.DataFrame:
        """Map downstream estimator coefficients to final feature names.

        Returns a DataFrame sorted by absolute coefficient magnitude.  Feature
        names are aligned to the downstream estimator after feature-mode and
        optional strict TopK filtering have been applied.  The ``feature_type``
        column distinguishes original features, mined HUG patterns, and
        augmented pair transforms.  ``pattern_support`` is populated only for
        mined HUG patterns; original and augmented-pair features use
        ``support_type='not_applicable'`` and ``pattern_support=NaN``.

        Raises
        ------
        AttributeError
            When the downstream estimator and its fitted one-vs-rest
            estimators do not expose linear coefficients.
        """
        check_is_fitted(self)
        if self._is_constant_prior_fallback_active():
            return pd.DataFrame(
                columns=[
                    "pattern",
                    "coefficient",
                    "abs_coefficient",
                    "pattern_support",
                    "support_type",
                    "feature_type",
                    "pattern_origin",
                    "survivor_led",
                ]
            )
        production_without_training_artifacts = self._is_production_mode() and not hasattr(
            self, "x_train_downstream_"
        )
        audit_note = (
            self._audit_artifact_message("Training matrices and drift-baseline audit artifacts")
            if production_without_training_artifacts
            else ""
        )
        if production_without_training_artifacts:
            warnings.warn(
                audit_note,
                HUGIMLWarning,
                stacklevel=2,
            )
        clf_step = self.model_.named_steps.get("clf")
        coefficient_blocks: list[np.ndarray] = []
        if hasattr(clf_step, "coef_"):
            coefficient_blocks.append(np.atleast_2d(np.asarray(clf_step.coef_, dtype=float)))
        else:
            for estimator in list(getattr(clf_step, "estimators_", []) or []):
                if hasattr(estimator, "coef_"):
                    coefficient_blocks.append(
                        np.atleast_2d(np.asarray(estimator.coef_, dtype=float))
                    )
        if not coefficient_blocks:
            raise AttributeError(
                "feature_importances requires a linear downstream estimator "
                "with fitted coefficients."
            )

        raw_coef = np.vstack(coefficient_blocks)
        coef = raw_coef.mean(axis=0) if raw_coef.shape[0] > 1 else raw_coef.ravel()
        features = self._get_downstream_feature_names()
        if len(features) != len(coef):
            raise RuntimeError(
                "Downstream feature names are not aligned with estimator coefficients: "
                f"{len(features)} names for {len(coef)} coefficients."
            )

        cached_pattern_support = getattr(self, "_downstream_pattern_support_", None)
        if cached_pattern_support is not None and len(cached_pattern_support) != len(features):
            cached_pattern_support = None
        support_lookup = self._pattern_support_lookup() if cached_pattern_support is None else {}
        cached_pattern_origin = getattr(self, "_downstream_pattern_origin_", None)
        cached_survivor_led = getattr(self, "_downstream_survivor_led_", None)
        cached_survivor_features = getattr(self, "_downstream_survivor_features_", None)
        cached_survivor_feature_count = getattr(self, "_downstream_survivor_feature_count_", None)
        cached_survivor_min_marginal_ig = getattr(
            self, "_downstream_survivor_min_marginal_ig_", None
        )
        cached_survivor_max_interaction_score = getattr(
            self, "_downstream_survivor_max_interaction_score_", None
        )
        cached_survivor_best_partners = getattr(self, "_downstream_survivor_best_partners_", None)
        has_cached_survivor_audit = all(
            value is not None and len(value) == len(features)
            for value in (
                cached_pattern_origin,
                cached_survivor_led,
                cached_survivor_features,
                cached_survivor_feature_count,
                cached_survivor_min_marginal_ig,
                cached_survivor_max_interaction_score,
                cached_survivor_best_partners,
            )
        )
        pattern_audit_lookup = (
            {} if has_cached_survivor_audit else self._pattern_audit_lookup_by_label()
        )
        strict_scores = getattr(self, "_strict_topk_feature_scores_", None)
        strict_score_lookup: dict[str, float] = {}
        if strict_scores is not None:
            full_names = getattr(self, "_downstream_feature_names_full_", None)
            if full_names is None or len(full_names) == 0:
                full_names = self._get_downstream_feature_names_full()
            strict_score_lookup = {
                name: float(score)
                for name, score in zip(full_names, np.asarray(strict_scores).ravel())
            }

        aug_lookup = {
            f"augmented_pair:{item.get('name')}": item
            for item in getattr(self, "augmented_pair_transforms_", [])
        }

        non_missing_rates = getattr(self, "_downstream_non_missing_rate_", None)
        variances = getattr(self, "_downstream_variance_", None)
        if (
            non_missing_rates is None
            or variances is None
            or len(non_missing_rates) != len(features)
            or len(variances) != len(features)
        ):
            X_meta = getattr(self, "x_train_downstream_", None)
            if X_meta is not None and len(features) == getattr(X_meta, "shape", (0, 0))[1]:
                X_arr = X_meta.toarray() if issparse(X_meta) else np.asarray(X_meta)
                finite_mask = np.isfinite(X_arr)
                non_missing_rates = finite_mask.mean(axis=0)
                variances = np.nanvar(np.where(finite_mask, X_arr, np.nan), axis=0)
            else:
                non_missing_rates = np.full(len(features), np.nan)
                variances = np.full(len(features), np.nan)

        rows: list[dict[str, object]] = []
        for idx, (feat, c) in enumerate(zip(features, coef)):
            feature_type = self._downstream_feature_type(feat)
            display_name = self._downstream_feature_display_name(feat)
            if feature_type == "pattern":
                if cached_pattern_support is not None:
                    pattern_support = float(cached_pattern_support[idx])
                else:
                    pattern_support = support_lookup.get(
                        feat, support_lookup.get(display_name, np.nan)
                    )
                support_type = "pattern_support"
                if has_cached_survivor_audit:
                    pattern_origin = str(cached_pattern_origin[idx]) or "standard"
                    survivor_led = bool(cached_survivor_led[idx])
                    survivor_features = list(cached_survivor_features[idx] or [])
                    survivor_feature_count = int(cached_survivor_feature_count[idx])
                    survivor_min_marginal_ig = self._finite_float_or_nan(
                        cached_survivor_min_marginal_ig[idx]
                    )
                    survivor_max_interaction_score = self._finite_float_or_nan(
                        cached_survivor_max_interaction_score[idx]
                    )
                    survivor_best_partners = list(cached_survivor_best_partners[idx] or [])
                else:
                    audit = pattern_audit_lookup.get(
                        feat, pattern_audit_lookup.get(display_name, {})
                    )
                    pattern_origin = str(audit.get("pattern_origin", "standard"))
                    survivor_led = bool(audit.get("survivor_led", False))
                    survivor_features = list(audit.get("survivor_features", []) or [])
                    survivor_feature_count = int(
                        audit.get("survivor_feature_count", len(survivor_features))
                    )
                    survivor_min_marginal_ig = self._finite_float_or_nan(
                        audit.get("survivor_min_marginal_ig", np.nan)
                    )
                    survivor_max_interaction_score = self._finite_float_or_nan(
                        audit.get("survivor_max_interaction_score", np.nan)
                    )
                    survivor_best_partners = list(audit.get("survivor_best_partners", []) or [])
            else:
                pattern_support = np.nan
                support_type = "not_applicable"
                pattern_origin = "not_applicable"
                survivor_led = False
                survivor_features = []
                survivor_feature_count = 0
                survivor_min_marginal_ig = np.nan
                survivor_max_interaction_score = np.nan
                survivor_best_partners = []

            support_value = (
                round(float(pattern_support), 4) if np.isfinite(pattern_support) else np.nan
            )
            aug_meta = aug_lookup.get(feat, {}) if feature_type == "augmented_pair" else {}
            std_mean = aug_meta.get("standardization_mean", np.nan)
            std_scale = aug_meta.get("standardization_scale", np.nan)
            std_scale_float = float(std_scale) if np.isfinite(std_scale) else np.nan
            coef_raw = (
                float(c) / std_scale_float
                if feature_type == "augmented_pair"
                and np.isfinite(std_scale_float)
                and std_scale_float != 0.0
                else np.nan
            )
            raw_formula = aug_meta.get("raw_formula", np.nan)
            if feature_type == "augmented_pair":
                aug_text = self._augmented_pair_effect_text(
                    raw_formula=str(raw_formula),
                    operation=str(aug_meta.get("operation", "")),
                    coefficient_raw_scale=coef_raw,
                    standardization_mean=float(std_mean) if np.isfinite(std_mean) else np.nan,
                    standardization_scale=std_scale_float,
                    source_observed_medians=aug_meta.get("source_observed_medians", np.nan),
                    pair_missing_policy=str(
                        aug_meta.get("pair_missing_policy", "reference_value_for_unavailable_pair")
                    ),
                    eligible_rate=float(aug_meta.get("eligible_rate", np.nan)),
                    missing_pair_rate=float(aug_meta.get("missing_pair_rate", np.nan)),
                )
                decision_direction = aug_text["decision_direction"]
                risk_increases_when = aug_text["risk_increases_when"]
                unit_effect_interpretation = aug_text["unit_effect_interpretation"]
                reference_raw_value_description = aug_text["reference_raw_value_description"]
                source_observed_medians_description = aug_text[
                    "source_observed_medians_description"
                ]
                pair_missing_policy_description = aug_text["pair_missing_policy_description"]
                raw_scale_note = aug_text["raw_scale_note"]
                raw_interpretation = aug_text["raw_interpretation"]
            else:
                decision_direction = np.nan
                risk_increases_when = np.nan
                unit_effect_interpretation = np.nan
                reference_raw_value_description = np.nan
                source_observed_medians_description = np.nan
                pair_missing_policy_description = np.nan
                raw_scale_note = np.nan
                raw_interpretation = np.nan
            rows.append(
                {
                    "pattern": display_name,
                    "feature": feat,
                    "display_name": display_name,
                    "feature_type": feature_type,
                    "pattern_origin": pattern_origin,
                    "survivor_led": survivor_led,
                    "survivor_features": survivor_features,
                    "survivor_feature_count": survivor_feature_count,
                    "survivor_min_marginal_ig": survivor_min_marginal_ig,
                    "survivor_max_interaction_score": survivor_max_interaction_score,
                    "survivor_best_partners": survivor_best_partners,
                    "coefficient": round(float(c), 6),
                    "abs_coefficient": round(abs(float(c)), 6),
                    "pattern_support": support_value,
                    "support": support_value,
                    "support_type": support_type,
                    "non_missing_rate": round(float(non_missing_rates[idx]), 6),
                    "variance": round(float(variances[idx]), 6),
                    "strict_topk_score": round(float(strict_score_lookup.get(feat, np.nan)), 6),
                    "standardization_mean": std_mean,
                    "standardization_scale": std_scale,
                    "operation": aug_meta.get("operation", np.nan),
                    "inputs": aug_meta.get("inputs", np.nan),
                    "raw_formula": raw_formula,
                    "standardized_formula": aug_meta.get("standardized_formula", np.nan),
                    "pair_missing_policy": aug_meta.get("pair_missing_policy", np.nan),
                    "eligible_count": aug_meta.get("eligible_count", np.nan),
                    "eligible_rate": aug_meta.get("eligible_rate", np.nan),
                    "missing_pair_rate": aug_meta.get("missing_pair_rate", np.nan),
                    "source_observed_medians": aug_meta.get("source_observed_medians", np.nan),
                    "transform_ig": aug_meta.get("transform_ig", np.nan),
                    "coefficient_standardized": round(float(c), 6)
                    if feature_type == "augmented_pair"
                    else np.nan,
                    "one_std_effect_on_log_odds": round(float(c), 6)
                    if feature_type == "augmented_pair"
                    else np.nan,
                    "coefficient_raw_scale": round(float(coef_raw), 12)
                    if np.isfinite(coef_raw)
                    else np.nan,
                    "one_raw_unit_effect_on_log_odds": round(float(coef_raw), 12)
                    if np.isfinite(coef_raw)
                    else np.nan,
                    "reference_raw_value": std_mean if feature_type == "augmented_pair" else np.nan,
                    "reference_raw_value_description": reference_raw_value_description,
                    "decision_direction": decision_direction,
                    "risk_increases_when": risk_increases_when,
                    "unit_effect_interpretation": unit_effect_interpretation,
                    "raw_scale_note": raw_scale_note,
                    "raw_interpretation": raw_interpretation,
                    "source_observed_medians_description": source_observed_medians_description,
                    "pair_missing_policy_description": pair_missing_policy_description,
                    "audit_note": (audit_note if production_without_training_artifacts else ""),
                }
            )
        result = (
            pd.DataFrame(rows)
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )
        if production_without_training_artifacts:
            result.attrs["audit_note"] = audit_note
        return pd.DataFrame(result)

    def plot_bin_profiles(self, figsize: tuple | None = None):
        """Bar chart of the chosen B per numerical feature (adaptive binning only).

        Colour encodes position in the candidate range:
        blue = coarse end, green = mid, amber/red = fine end.

        Returns
        -------
        (fig, ax)

        Raises
        ------
        RuntimeError
            When called on a non-adaptive or unfitted model.
        ImportError
            When matplotlib is not installed.
        """
        self._check_adaptive_fitted("plot_bin_profiles")
        self._require_mpl()
        import matplotlib.pyplot as plt

        feats = list(self.per_feature_b_.keys())
        bvals = [self.per_feature_b_[f] for f in feats]
        cands = self.b_candidates or [2, 15]
        lo, hi = min(cands), max(cands)

        colors = [
            "#2166ac"
            if b <= lo + (hi - lo) / 3
            else "#1a9641"
            if b <= lo + 2 * (hi - lo) / 3
            else "#d7191c"
            for b in bvals
        ]

        fig, ax = plt.subplots(figsize=figsize or (max(7, len(feats) * 0.5 + 2), 4))
        ax.bar(range(len(feats)), bvals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Chosen B_j", fontsize=10)
        ax.set_title(
            f"Adaptive binning — chosen B per feature  "
            f"(threshold={self.min_marginal_gain_ratio:.0%})",
            fontsize=11,
        )
        for i, b in enumerate(bvals):
            ax.text(i, b + 0.05, str(b), ha="center", fontsize=8)
        fig.tight_layout()
        return fig, ax

    def ig_heatmap(self, figsize: tuple | None = None):
        """Heatmap of IG score at every (feature, B) grid point (adaptive binning only).

        The chosen B per feature is highlighted with a bounding box.

        Returns
        -------
        (fig, ax)

        Raises
        ------
        RuntimeError
            When called on a non-adaptive or unfitted model, or when
            ``ig_scores_`` is empty.
        ImportError
            When matplotlib is not installed.
        """
        self._check_adaptive_fitted("ig_heatmap")
        if not getattr(self, "ig_scores_", None):
            raise RuntimeError("ig_scores_ is empty — call fit() first.")
        self._require_mpl()
        import matplotlib.pyplot as plt

        feats = sorted(self.ig_scores_)
        bs = sorted({b for sc in self.ig_scores_.values() for b in sc})
        grid = np.array([[self.ig_scores_[f].get(b, 0.0) for b in bs] for f in feats])

        fig, ax = plt.subplots(
            figsize=figsize or (max(6, len(bs) * 0.9), max(4, len(feats) * 0.45))
        )
        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(bs)))
        ax.set_xticklabels([str(b) for b in bs], fontsize=9)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=8)
        ax.set_xlabel("B candidates", fontsize=10)
        ax.set_title("IG score per (feature, B)  — box = chosen B", fontsize=11)
        for i, f in enumerate(feats):
            chosen = self.per_feature_b_.get(f)
            if chosen and chosen in bs:
                j = bs.index(chosen)
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2,
                    )
                )
        plt.colorbar(im, ax=ax, label="Information gain")
        fig.tight_layout()
        return fig, ax

    def _check_adaptive_fitted(self, method_name: str) -> None:
        """Raise a clear error when an adaptive-only method is called incorrectly."""
        check_is_fitted(self)
        if not getattr(self, "adaptive_binning", False):
            raise RuntimeError(
                f"{method_name}() is only available when adaptive_binning=True.  "
                f"Re-fit with HUGIMLClassifier(adaptive_binning=True, ...) "
                f"or use HUGIMLAdaptive."
            )
        if not getattr(self, "per_feature_b_", None):
            raise RuntimeError(f"{method_name}() requires per_feature_b_ — call fit() first.")

    @staticmethod
    def _require_mpl() -> None:
        """Raise ImportError when matplotlib is not installed."""
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise ImportError(
                "matplotlib is required for diagnostic plots. "
                "Install with:  pip install matplotlib  "
                "or:  pip install 'hugiml-core[plots]'"
            )

    def _summary_shape_text(self, matrix_attr: str, cached_shape_attr: str) -> str:
        """Return a stable summary shape for audit or production-retained models."""
        matrix = getattr(self, matrix_attr, None)
        if matrix is not None and hasattr(matrix, "shape"):
            return str(tuple(int(v) for v in matrix.shape))
        cached = getattr(self, cached_shape_attr, None)
        if cached is not None:
            return (
                f"{tuple(int(v) for v in cached)} (training matrix not retained in production mode)"
            )
        if self._is_production_mode():
            return "not retained in production mode"
        return "unavailable"

    def model_summary(self) -> str:
        """Human-readable model summary including top patterns."""
        check_is_fitted(self)
        composition = self.get_model_composition()
        counts = composition.get("downstream_feature_counts", {})
        lines = [
            "HUGIMLClassifier — Model Summary",
            "=" * 50,
            f"Config:       B={self.B}, L={self.L}, G={self.G}",
            f"Feature mode: {getattr(self, 'feature_mode', 'patterns_only')}",
            f"Training:     {self.fit_metadata_.n_samples} samples, "
            f"{self.fit_metadata_.n_features} features, "
            f"{self.fit_metadata_.n_classes} classes",
            f"Patterns:     {self.fit_metadata_.n_patterns} "
            f"({self.fit_metadata_.n_compound} compound)",
            f"Augmented pairs: {counts.get('augmented_pair', 0)} retained",
            f"Downstream composition: original={counts.get('original', 0)}, "
            f"patterns={counts.get('pattern', 0)}, "
            f"augmented_pair={counts.get('augmented_pair', 0)}, "
            f"total={counts.get('total', 0)}",
            f"Matrix:       {self._summary_shape_text('x_train_hup_', '_training_pattern_matrix_shape_')} "
            f"(density={self.fit_metadata_.matrix_density:.4f})",
            f"Downstream:   {self._summary_shape_text('x_train_downstream_', '_training_downstream_matrix_shape_')}",
            f"Fit time:     {self.fit_metadata_.total_fit_ms:.0f} ms",
            "",
            "Stage breakdown (ms):",
        ]
        for stage, ms in self.fit_metadata_.stage_times_ms.items():
            lines.append(f"  {stage:<25} {ms:>8.1f}")
        try:
            imp = self.feature_importances().head(10)
            has_non_pattern = bool((imp.get("feature_type", "pattern") != "pattern").any())
            has_augmented = bool((imp.get("feature_type", "pattern") == "augmented_pair").any())
            section = (
                "Top 10 downstream features by importance:"
                if has_non_pattern
                else "Top 10 patterns by importance:"
            )
            lines += ["", section]
            if has_augmented:
                lines.append(
                    "  (includes augmented pair transforms; use "
                    "explain_augmented_pair_effects() for raw-scale interpretation)"
                )
            for _, row in imp.iterrows():
                support_text = (
                    f"pattern_support={row['pattern_support']:.3f}"
                    if row.get("support_type") == "pattern_support"
                    else "pattern_support=n/a"
                )
                lines.append(
                    f"  [{row.get('feature_type', 'pattern')}] "
                    f"{row['pattern']:<40} "
                    f"coef={row['coefficient']:>+8.4f}  "
                    f"{support_text}"
                )
        except AttributeError:
            lines += ["", "Top downstream features by importance:"]
            lines.append("  (not available — non-LR downstream estimator)")

        # ── Adaptive binning section ────────────────────────────────────
        if getattr(self, "_missing_col_edges_", None):
            lines += [
                "",
                f"NaN handling: {len(self._missing_col_edges_)} numerical column(s) "
                f"pre-binned (NaN/Inf generates no transaction item at train or test time).",
            ]
        if self.adaptive_binning and getattr(self, "per_feature_b_", None):
            lines += ["", "Adaptive binning — chosen B per feature:"]
            for feat, b in sorted(self.per_feature_b_.items(), key=lambda kv: -kv[1]):
                edges = self._bin_edges_.get(feat, [])
                rng = f"  [{float(edges[0]):.4g}…{float(edges[-1]):.4g}]" if len(edges) >= 2 else ""
                lines.append(f"  {feat:<35} B={b:<3}{rng}")
        # ─────────────────────────────────────────────────────────────────

        return "\n".join(lines)

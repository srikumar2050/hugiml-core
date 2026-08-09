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

"""Pattern, RPTE, complexity, and augmented-pair interpretation APIs."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted


class _InterpretationMixin:
    """Implement model interpretation and audit views."""

    _PATTERN_ATOM_INTERVAL_RE = re.compile(
        r"^(?P<feature>.+?)=\[(?P<lower>[^,\]]+),\s*(?P<upper>[^)\]]+)\)$"
    )

    @classmethod
    def _parse_pattern_atom_label(cls, label: str) -> dict[str, Any] | None:
        """Parse ONE individual transaction-item label (e.g. "age=[35,50)"
        or "gender=F") into a structured atom.

        Deliberately operates on a single item's own label, never on a
        compound pattern's comma-joined display string ("income=[91.02,
        101.4), age=[60,87.2)") -- splitting THAT on every comma is unsafe,
        since the commas inside numeric interval bounds are not atom
        separators. Each pattern's individual item labels are already
        available separately (self.td_.item_map, indexed by the item ids
        in a PatternEntry.items list) with no compound-string parsing
        needed at all; see get_pattern_provenance(), the only caller.
        """
        text = str(label or "").strip()
        if not text:
            return None
        m = cls._PATTERN_ATOM_INTERVAL_RE.match(text)
        if m:
            feature = m.group("feature").strip()
            try:
                lower = float(m.group("lower"))
                upper = float(m.group("upper"))
            except ValueError:
                return {"feature": feature, "operator": "unparsed", "raw_label": text}
            return {
                "feature": feature,
                "operator": "interval",
                "lower": lower,
                "upper": upper,
                "lower_inclusive": True,
                "upper_inclusive": False,
            }
        if "=" in text:
            feature, value = text.split("=", 1)
            return {"feature": feature.strip(), "operator": "equals", "value": value.strip()}
        return {"feature": text, "operator": "present"}

    def get_pattern_provenance(self) -> dict[str, dict[str, Any]]:
        """Return raw-feature mappings for downstream HUG pattern columns.

        The method is available in both audit and production execution modes.
        It reads the retained pattern structures and transaction item labels;
        unlike :meth:`get_pattern_info`, it does not require the retained
        training pattern matrix.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping keyed by the full downstream pattern name. Each value
            contains pattern order, ordered raw feature names, and parsed atom
            records. A representative value has this shape::

                {
                    "name": "pattern:age=[35,50), smoker=1",
                    "family": "pattern",
                    "order": 2,
                    "raw_features": ["age", "smoker"],
                    "atoms": [
                        {
                            "feature": "age",
                            "operator": "interval",
                            "lower": 35.0,
                            "upper": 50.0,
                            "lower_inclusive": True,
                            "upper_inclusive": False,
                        },
                        {
                            "feature": "smoker",
                            "operator": "equals",
                            "value": "1",
                        },
                    ],
                }

        Notes
        -----
        Each atom is parsed from its individual item label. The method never
        splits the compound comma-joined display string, because numeric
        interval bounds can themselves contain commas.
        """
        check_is_fitted(self)
        try:
            names = list(self._get_downstream_feature_names())
            labels = self.get_hug_features()
            patterns = list(getattr(self, "patterns_", []) or [])
            item_map = getattr(getattr(self, "td_", None), "item_map", {}) or {}
            label_remap = getattr(self, "_adaptive_code_label_map_", {}) or {}
        except AttributeError:
            return {}
        if len(labels) != len(patterns):
            return {}
        pattern_by_label: dict[str, Any] = {}
        for label, pe in zip(labels, patterns):
            pattern_by_label[label] = pe
            pattern_by_label[f"pattern:{label}"] = pe

        provenance: dict[str, dict[str, Any]] = {}
        for name in names:
            try:
                if self._downstream_feature_type(str(name)) != "pattern":
                    continue
                display = self._downstream_feature_display_name(str(name))
            except AttributeError:
                continue
            pe = (
                pattern_by_label.get(str(name))
                or pattern_by_label.get(f"pattern:{display}")
                or pattern_by_label.get(display)
            )
            if pe is None:
                continue
            atoms: list[dict[str, Any]] = []
            raw_features: list[str] = []
            item_ids = list(getattr(pe, "items", []) or [])
            for item_id in item_ids:
                raw_label = item_map.get(int(item_id), str(item_id))
                label = label_remap.get(raw_label, raw_label)
                atom = self._parse_pattern_atom_label(str(label))
                if atom is None:
                    continue
                atoms.append(atom)
                feature = atom.get("feature")
                if feature and feature not in raw_features:
                    raw_features.append(str(feature))
            if raw_features:
                provenance[str(name)] = {
                    "name": str(name),
                    "family": "pattern",
                    "order": int(len(item_ids)),
                    "raw_features": raw_features,
                    "atoms": atoms,
                }
        return provenance

    def get_augmented_pair_transforms(self) -> list[dict[str, Any]]:
        """Return augmented pair transforms used by the downstream estimator.

        Each catalog entry includes the raw pair formula, source-feature IG
        source mappings, candidate coverage, unavailable-pair policy, and the
        standardization parameters used before the downstream estimator sees
        the feature.  Candidate IG is scored on rows where both source values
        are observed.  For selected features, rows where the pair value cannot
        be computed receive the pair feature's training reference value before
        standardization, yielding a neutral standardized value.
        """
        return [dict(item) for item in getattr(self, "augmented_pair_transforms_", [])]

    def rpte_rule_table(self, feature_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Return structured RPTE leaf and direct-source explanation rows.

        The downstream estimator is identified by a
        ``unified_rule_table()`` method, so this API does not require a hard
        dependency on a particular RPTE implementation. The fitted pipeline is
        unwrapped first. For a fitted ``OneVsRestClassifier``, rows from every
        binary sub-estimator are concatenated and retain their class field.

        RPTE rows cover both fitted leaf indicators and non-zero direct source
        terms. Direct source terms may be original columns, HUG patterns, or
        augmented pairs that were not selected in accepted tree splits.

        Parameters
        ----------
        feature_names : list[str] or None, default None
            Optional names aligned with the downstream estimator input. When
            omitted, fitted downstream feature names are used.

        Returns
        -------
        list[dict[str, Any]]
            Structured explanation rows. Returns an empty list when the fitted
            downstream estimator does not implement ``unified_rule_table()``
            (for example, the built-in logistic-regression branch).

        See Also
        --------
        rpte_rule_tree
            Ready-to-print decision-tree-style representation.
        """
        check_is_fitted(self, "model_")
        estimator = self.model_.named_steps.get("clf", self.model_)
        names = feature_names if feature_names is not None else self._get_downstream_feature_names()

        sub_estimators = []
        if hasattr(estimator, "estimators_") and hasattr(estimator, "classes_"):
            # OneVsRestClassifier (or a compatible multiclass wrapper):
            # one fitted binary sub-estimator per class.
            sub_estimators = list(getattr(estimator, "estimators_", []))
        elif hasattr(estimator, "unified_rule_table"):
            sub_estimators = [estimator]

        rows: list[dict[str, Any]] = []
        for sub in sub_estimators:
            if not hasattr(sub, "unified_rule_table"):
                continue
            rows.extend(sub.unified_rule_table(feature_names=names))
        return rows

    def rpte_representation_aliases(self) -> list[dict[str, Any]]:
        """Return mined pattern columns canonicalized to equivalent RPTE leaves.

        Each row identifies a direct pattern column omitted from the final
        logistic layout because an RPTE leaf represented the same atom
        conjunction and had identical fitted support. For one-versus-rest models,
        the governed class is attached to
        every row. Non-RPTE downstream estimators return an empty list.
        """
        check_is_fitted(self, "model_")
        estimator = self.model_.named_steps.get("clf", self.model_)
        aliases: list[dict[str, Any]] = []
        if hasattr(estimator, "estimators_") and hasattr(estimator, "classes_"):
            classes = list(getattr(estimator, "classes_", []))
            sub_estimators = list(getattr(estimator, "estimators_", []))
            labels = [classes[-1]] if len(sub_estimators) == 1 and len(classes) == 2 else classes
            for index, sub in enumerate(sub_estimators):
                method = getattr(sub, "representation_alias_table", None)
                if not callable(method):
                    continue
                class_label = labels[index] if index < len(labels) else index
                for row in method():
                    item = dict(row)
                    item["class"] = class_label
                    aliases.append(item)
            return aliases
        method = getattr(estimator, "representation_alias_table", None)
        if callable(method):
            aliases.extend(dict(row) for row in method())
        return aliases

    def get_complexity_report(
        self,
        *,
        X: Any | None = None,
        coefficient_tolerance: float = 1e-12,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """Return the fitted model's structural and inspection measures.

        Passing ``X`` adds the mean, sample standard deviation, and two-sided
        confidence interval for instance inspection units.
        """
        from hugiml.compute_complexity import get_complexity_report

        report = get_complexity_report(
            self,
            X=X,
            coefficient_tolerance=coefficient_tolerance,
            confidence_level=confidence_level,
        )
        if report is None:
            raise RuntimeError("Complexity is unavailable for this fitted estimator")
        return report

    def get_instance_inspection_units(
        self,
        X: Any,
        *,
        coefficient_tolerance: float = 1e-12,
    ) -> np.ndarray:
        """Return one expanded inspection count for every row in ``X``."""
        from hugiml.compute_complexity import get_instance_inspection_units

        values = get_instance_inspection_units(
            self,
            X,
            coefficient_tolerance=coefficient_tolerance,
        )
        if values is None:
            raise RuntimeError(
                "Instance inspection units are unavailable for this fitted estimator"
            )
        return values

    def get_complexity(
        self,
        mode: str | None = None,
        *,
        X: Any | None = None,
        coefficient_tolerance: float = 1e-12,
        confidence_level: float = 0.95,
    ) -> int | float:
        """Return one measure from the uniform package interface.

        Parameters
        ----------
        mode : {"model units", "model inspection units",
                "instance inspection units"} or None, default=None
            ``None`` returns model inspection units. Instance inspection units
            require ``X`` and return the arithmetic mean across its rows.
        X : array-like or DataFrame, optional
            Rows used for instance inspection units.
        coefficient_tolerance : float, default=1e-12
            Absolute threshold used to identify active fitted terms and
            non-zero row-specific transformed values.
        confidence_level : float, default=0.95
            Confidence level used when the instance summary is requested.
        """
        from hugiml.compute_complexity import get_complexity

        value = get_complexity(
            self,
            mode,
            X=X,
            coefficient_tolerance=coefficient_tolerance,
            confidence_level=confidence_level,
        )
        if value is None:
            raise RuntimeError("Complexity is unavailable for this fitted estimator")
        return value

    def rpte_rule_tree(
        self,
        feature_names: list[str] | None = None,
        *,
        condition_space: str = "raw",
        detail_level: str = "full",
        precision: int = 5,
        include_direct_terms: bool = True,
        include_generation_details: bool = False,
        class_label: Any | None = None,
        tree_index: int | None = None,
    ) -> str:
        """Return RPTE prediction evidence as ready-to-print flat trees.

        This is the readable companion to :meth:`rpte_rule_table`.  Shared
        condition prefixes are merged, one split is shown per indentation
        level, and final LR details are attached at terminal leaves.  Direct
        source terms are grouped by original, HUG-pattern, and augmented-pair
        families after the tree sections.

        ``condition_space`` accepts ``"raw"``, ``"downstream"``, or ``"both"``.
        ``detail_level`` accepts ``"compact"`` or ``"full"``.  The method
        returns an empty string when the fitted downstream estimator is not
        RPTE-based.

        Example::

            print(model.rpte_rule_tree())
        """
        from .rpte_interpretability import format_rpte_rule_tree

        rows = self.rpte_rule_table(feature_names=feature_names)
        return format_rpte_rule_tree(
            rows,
            condition_space=condition_space,
            detail_level=detail_level,
            precision=precision,
            include_direct_terms=include_direct_terms,
            include_generation_details=include_generation_details,
            class_label=class_label,
            tree_index=tree_index,
        )

    def get_original_feature_standardization(self) -> dict[str, dict[str, Any]]:
        """Standardization and missing-value-imputation parameters for
        every numeric original ("orig:") feature, keyed by raw feature
        name (not the "orig:" prefixed downstream name).

        This is what makes exact threshold rendering possible for original
        features (interpretability priority #4): a downstream split like
        "orig:age > -0.015" is meaningless to a domain reader on its own --
        HUGIML standardizes every original numeric column with its own
        (mean, scale) before any downstream estimator (including RPTE) ever
        sees it. Inverting is a single affine transform:
        raw_threshold = standardized_threshold * scale + mean.

        Also reports the training-median imputation value used for missing
        raw values in that column (interpretability priority #11): a row
        whose standardized value is very close to 0 could be a genuinely
        average observation OR a row where this feature was missing and
        got median-imputed before standardization -- these are NOT always
        distinguishable from the standardized value alone, which is
        exactly why the imputation value is reported explicitly here rather
        than left implicit.

        Returns {} if no original numeric columns were standardized (e.g.
        feature_mode="patterns_only", or every original column is
        categorical).
        """
        check_is_fitted(self)
        scaler = getattr(self, "_original_scaler_", None)
        cols = list(getattr(self, "_original_numeric_cols_", []) or [])
        if scaler is None or not cols:
            return {}
        means = np.asarray(getattr(scaler, "mean_", np.full(len(cols), np.nan)), dtype=float)
        scales = np.asarray(getattr(scaler, "scale_", np.full(len(cols), np.nan)), dtype=float)
        medians = getattr(self, "_original_numeric_medians_", None)
        out: dict[str, dict[str, Any]] = {}
        for i, name in enumerate(cols):
            median_value = float("nan")
            if medians is not None:
                try:
                    median_value = float(medians.get(name, float("nan")))
                except Exception:
                    median_value = float("nan")
            out[str(name)] = {
                "feature": str(name),
                "standardization_mean": float(means[i]) if i < len(means) else float("nan"),
                "standardization_scale": float(scales[i]) if i < len(scales) else float("nan"),
                "missing_value_policy": "median_imputation",
                "median_imputation_value": median_value,
            }
        return out

    def get_augmented_pair_standardization(self) -> pd.DataFrame:
        """Return standardization metadata for augmented pair features.

        The returned columns are aligned to ``get_augmented_pair_transforms()``
        and make the raw-to-estimator transformation explicit.
        """
        rows: list[dict[str, Any]] = []
        for item in self.get_augmented_pair_transforms():
            rows.append(
                {
                    "name": item.get("name"),
                    "operation": item.get("operation"),
                    "inputs": item.get("inputs"),
                    "raw_formula": item.get("raw_formula", item.get("formula")),
                    "standardization_mean": item.get("standardization_mean"),
                    "standardization_scale": item.get("standardization_scale"),
                    "standardized_formula": item.get("standardized_formula"),
                    "reference_raw_value": item.get("reference_raw_value"),
                    "pair_missing_policy": item.get("pair_missing_policy"),
                    "eligible_count": item.get("eligible_count"),
                    "eligible_rate": item.get("eligible_rate"),
                    "missing_pair_rate": item.get("missing_pair_rate"),
                    "source_observed_medians": item.get("source_observed_medians"),
                    "transform_ig": item.get("transform_ig"),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _format_source_observed_medians(source_observed_medians: Any) -> str:
        if not isinstance(source_observed_medians, dict) or not source_observed_medians:
            return "not available"
        parts: list[str] = []
        for key, value in source_observed_medians.items():
            try:
                parts.append(f"{key}={float(value):.6g}")
            except (TypeError, ValueError):
                parts.append(f"{key}={value}")
        return ", ".join(parts)

    @staticmethod
    def _augmented_pair_effect_text(
        *,
        raw_formula: str,
        operation: str,
        coefficient_raw_scale: float,
        standardization_mean: float,
        standardization_scale: float,
        source_observed_medians: Any,
        pair_missing_policy: str,
        eligible_rate: float,
        missing_pair_rate: float,
    ) -> dict[str, Any]:
        eligible_text = (
            f"Candidate scoring used rows where both source values were observed"
            f" (eligible_rate={eligible_rate:.3g})."
            if np.isfinite(eligible_rate)
            else "Candidate scoring used rows where both source values were observed."
        )
        missing_text = (
            f"training rows where the pair was unavailable: {missing_pair_rate:.3g}."
            if np.isfinite(missing_pair_rate)
            else "the unavailable-pair rate is not available."
        )
        reference_note = (
            f"The reference raw value {standardization_mean:.6g} is the training-cohort mean "
            f"of the observed {raw_formula} pair term after applying the selected pair operation. "
            "It is not a domain-specific baseline."
        )
        source_median_text = _InterpretationMixin._format_source_observed_medians(
            source_observed_medians
        )
        missing_policy_note = (
            "If a selected pair cannot be computed for a row because one or both source values are missing, "
            "the augmented-pair feature is set to its training reference raw value before standardization. "
            "That gives the pair term a neutral standardized value of 0 for that row. "
            "This policy applies only to continuous augmented-pair features; HUGIML pattern features keep "
            "their native missing-value handling. "
            f"For diagnostics, source feature medians observed in training were: {source_median_text}."
        )

        if not np.isfinite(coefficient_raw_scale):
            return {
                "decision_direction": "effect_not_available",
                "risk_increases_when": "not_available",
                "unit_effect_interpretation": "Raw-scale log-odds effect is not available for this downstream estimator.",
                "reference_raw_value_description": "training_cohort_mean_of_observed_raw_pair_value",
                "source_observed_medians_description": "per-source-feature observed medians for diagnostics only; not used to construct pair values",
                "pair_missing_policy_description": "unavailable pair values are set to the pair reference raw value before standardization",
                "raw_scale_note": f"{reference_note} {eligible_text}",
                "raw_interpretation": (
                    f"The downstream estimator uses ({raw_formula} - {standardization_mean:.6g}) "
                    f"/ {standardization_scale:.6g}. {reference_note} {eligible_text} "
                    f"For selected-feature construction, {missing_policy_note}"
                ),
            }

        if coefficient_raw_scale > 0:
            direction = "higher_raw_value_increases_score"
            direction_text = f"Higher {raw_formula} increases the model score."
        elif coefficient_raw_scale < 0:
            direction = "higher_raw_value_decreases_score"
            direction_text = f"Higher {raw_formula} decreases the model score."
        else:
            direction = "raw_value_has_zero_linear_effect"
            direction_text = f"Higher {raw_formula} does not change the linear model score."

        if operation == "absolute_difference":
            risk_when = (
                "absolute_difference_increases"
                if coefficient_raw_scale > 0
                else "absolute_difference_decreases"
                if coefficient_raw_scale < 0
                else "not_applicable"
            )
            unit_text = (
                f"Each +1 increase in the absolute difference term changes the log-odds by "
                f"{coefficient_raw_scale:.6g}."
            )
            raw_scale_note = (
                "The raw-unit effect is expressed on the absolute-difference scale. "
                + reference_note
            )
        elif operation == "product":
            risk_when = (
                "product_value_increases"
                if coefficient_raw_scale > 0
                else "product_value_decreases"
                if coefficient_raw_scale < 0
                else "not_applicable"
            )
            unit_text = (
                f"A +1 change in the product term changes the log-odds by "
                f"{coefficient_raw_scale:.6g}. For a product feature, changing one source "
                "variable does not have a constant marginal effect; it depends on the current "
                "value of the other source variable."
            )
            raw_scale_note = (
                "The raw-unit effect is expressed on the product-term scale, not as a constant "
                "one-unit effect of either individual source feature. " + reference_note
            )
        elif operation == "sum":
            risk_when = (
                "sum_value_increases"
                if coefficient_raw_scale > 0
                else "sum_value_decreases"
                if coefficient_raw_scale < 0
                else "not_applicable"
            )
            unit_text = (
                f"Each +1 increase in the sum term changes the log-odds by "
                f"{coefficient_raw_scale:.6g}. The same coefficient applies to a one-unit "
                "increase in either source feature while the other source feature is kept constant."
            )
            raw_scale_note = (
                "The raw-unit effect is expressed on the pair sum scale. " + reference_note
            )
        elif operation == "signed_difference":
            risk_when = (
                "left_minus_right_increases"
                if coefficient_raw_scale > 0
                else "left_minus_right_decreases"
                if coefficient_raw_scale < 0
                else "not_applicable"
            )
            unit_text = (
                f"Each +1 increase in the signed difference term changes the log-odds by "
                f"{coefficient_raw_scale:.6g}. Increasing the left source feature raises this "
                "term, while increasing the right source feature lowers it."
            )
            raw_scale_note = (
                "The raw-unit effect is expressed on the signed left-minus-right difference scale. "
                + reference_note
            )
        else:
            risk_when = (
                "raw_value_increases"
                if coefficient_raw_scale > 0
                else "raw_value_decreases"
                if coefficient_raw_scale < 0
                else "not_applicable"
            )
            unit_text = (
                f"Each +1 raw-unit increase changes the log-odds by {coefficient_raw_scale:.6g}."
            )
            raw_scale_note = reference_note

        return {
            "decision_direction": direction,
            "risk_increases_when": risk_when,
            "unit_effect_interpretation": unit_text,
            "reference_raw_value_description": "training_cohort_mean_of_observed_raw_pair_value",
            "source_observed_medians_description": "per-source-feature observed medians for diagnostics only; not used to construct pair values",
            "pair_missing_policy_description": "unavailable pair values are set to the pair reference raw value before standardization",
            "raw_scale_note": f"{raw_scale_note} {eligible_text}",
            "raw_interpretation": (
                f"{direction_text} {unit_text} The downstream estimator uses "
                f"({raw_formula} - {standardization_mean:.6g}) / {standardization_scale:.6g}. "
                f"{reference_note} {eligible_text} For selected-feature construction, {missing_policy_note} "
                f"Among training rows, {missing_text}"
            ),
        }

    def _augmented_pair_effect_rows(self) -> list[dict[str, Any]]:
        """Return augmented-pair effect rows in raw and standardized units."""
        check_is_fitted(self)
        try:
            imp = self.feature_importances()
            coef_lookup = dict(zip(imp["feature"], imp["coefficient"]))
        except AttributeError:
            coef_lookup = {}

        rows: list[dict[str, Any]] = []
        for item in self.get_augmented_pair_transforms():
            name = str(item.get("name"))
            feature = f"augmented_pair:{name}"
            coef_std = float(coef_lookup.get(feature, np.nan))
            mean = float(item.get("standardization_mean", np.nan))
            scale = float(item.get("standardization_scale", np.nan))
            scale_safe = scale if np.isfinite(scale) and scale != 0.0 else np.nan
            coef_raw = (
                coef_std / scale_safe
                if np.isfinite(coef_std) and np.isfinite(scale_safe)
                else np.nan
            )
            operation = str(item.get("operation", ""))
            raw_formula = str(item.get("raw_formula", item.get("formula", name)))

            text = self._augmented_pair_effect_text(
                raw_formula=raw_formula,
                operation=operation,
                coefficient_raw_scale=coef_raw,
                standardization_mean=mean,
                standardization_scale=scale,
                source_observed_medians=item.get("source_observed_medians"),
                pair_missing_policy=str(
                    item.get("pair_missing_policy", "reference_value_for_unavailable_pair")
                ),
                eligible_rate=float(item.get("eligible_rate", np.nan)),
                missing_pair_rate=float(item.get("missing_pair_rate", np.nan)),
            )

            rows.append(
                {
                    "feature": feature,
                    "name": name,
                    "operation": operation,
                    "inputs": item.get("inputs"),
                    "raw_formula": raw_formula,
                    "standardized_formula": item.get("standardized_formula"),
                    "standardization_mean": mean,
                    "standardization_scale": scale,
                    "reference_raw_value": mean,
                    "reference_raw_value_description": text["reference_raw_value_description"],
                    "coefficient_standardized": coef_std,
                    "one_std_effect_on_log_odds": coef_std,
                    "coefficient_raw_scale": coef_raw,
                    "one_raw_unit_effect_on_log_odds": coef_raw,
                    "decision_direction": text["decision_direction"],
                    "risk_increases_when": text["risk_increases_when"],
                    "unit_effect_interpretation": text["unit_effect_interpretation"],
                    "raw_scale_note": text["raw_scale_note"],
                    "raw_interpretation": text["raw_interpretation"],
                    "pair_missing_policy": item.get("pair_missing_policy"),
                    "pair_missing_policy_description": text["pair_missing_policy_description"],
                    "eligible_count": item.get("eligible_count"),
                    "eligible_rate": item.get("eligible_rate"),
                    "missing_pair_rate": item.get("missing_pair_rate"),
                    "source_observed_medians": item.get("source_observed_medians"),
                    "source_observed_medians_description": text[
                        "source_observed_medians_description"
                    ],
                    "transform_ig": item.get("transform_ig"),
                }
            )
        return rows

    def explain_augmented_pair_effects(self) -> pd.DataFrame:
        """Explain augmented-pair effects in standardized and raw units.

        The downstream estimator is fit on standardized augmented-pair values.
        This method converts each standardized coefficient back to the raw pair
        scale and states that the reference value is the training-cohort mean
        of the observed pair term, not a domain-specific baseline. Candidate
        scoring uses rows where both source values are observed. For selected
        features, rows where the pair cannot be computed receive the pair
        feature's training reference raw value before standardization, yielding
        a neutral standardized value for that pair term. HUGIML pattern
        features keep their native missing-value handling.

        For logistic-regression downstream models, coefficient columns are
        log-odds effects.  Product-term effects are expressed on the product
        scale; changing one individual input does not have a constant marginal
        effect because it depends on the current value of the other input.
        """
        return pd.DataFrame(self._augmented_pair_effect_rows())

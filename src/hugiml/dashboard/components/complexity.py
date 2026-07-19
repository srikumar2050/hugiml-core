"""Complexity and budget component."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from hugiml.compute_complexity import get_complexity, get_complexity_report
from hugiml.dashboard.display import dataframe_for_display


def _base_estimator_structure(model: Any) -> dict[str, Any] | None:
    """Return RPTE-specific structural settings when available."""
    estimator = getattr(model, "base_estimator", None)
    if estimator is None:
        return None
    # Unwrap a OneVsRestClassifier (or compatible multiclass wrapper) to its
    # own per-class RPTE template -- hyperparameters are shared across every
    # one-vs-rest sub-estimator, so any one of them (fitted or not) reports
    # the same configured complexity.
    target = getattr(estimator, "estimator", estimator)
    fitted_targets = list(getattr(estimator, "estimators_", []) or [])
    if fitted_targets:
        target = fitted_targets[0]
    info: dict[str, Any] = {"downstream_estimator_class": type(target).__name__}
    for attr in ("depth", "leaf_config", "enable_lookahead", "n_estimators", "rpte_learning_rate"):
        if hasattr(target, attr):
            info[attr] = getattr(target, attr)
    fe = getattr(target, "fe_", None)
    growth = getattr(fe, "growth_summary", None)
    if callable(growth):
        try:
            summary = growth()
            info["n_trees_grown"] = len(summary) if hasattr(summary, "__len__") else None
        except Exception:
            pass
    return info if len(info) > 1 else info


def render_complexity(model: Any = None, *args, **kwargs) -> None:
    st.markdown("### Complexity and Budget Evidence")
    if model is None:
        st.info("No model loaded.")
        return

    config = {
        "L": getattr(model, "L", None),
        "topK": getattr(model, "topK", None),
        "G": getattr(model, "G", None),
        "feature_mode": getattr(model, "feature_mode", None),
        "topk_budget_strict": getattr(model, "topk_budget_strict", None),
    }
    base_estimator = getattr(model, "base_estimator", None)
    config["downstream_estimator"] = (
        "RPTE (higher-order)" if base_estimator is not None else "Logistic regression (built-in)"
    )
    config = {k: v for k, v in config.items() if v is not None}

    inspection_X = kwargs.get("X")
    try:
        complexity_report = get_complexity_report(model, X=inspection_X)
        model_units = get_complexity(model, "model units")
        model_inspection_units = get_complexity(model, "model inspection units")
        instance_summary = (complexity_report or {}).get("instance_inspection_units", {})
        instance_mean = (
            instance_summary.get("mean") if instance_summary.get("available") else None
        )
        instance_ci_lower = (
            instance_summary.get("ci_lower") if instance_summary.get("available") else None
        )
        instance_ci_upper = (
            instance_summary.get("ci_upper") if instance_summary.get("available") else None
        )
        instance_confidence = (
            instance_summary.get("confidence_level")
            if instance_summary.get("available")
            else None
        )
    except Exception:
        complexity_report = None
        model_units = None
        model_inspection_units = None
        instance_mean = None
        instance_ci_lower = None
        instance_ci_upper = None
        instance_confidence = None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model units", model_units if model_units is not None else "N/A")
    c2.metric(
        "Model inspection units",
        model_inspection_units if model_inspection_units is not None else "N/A",
    )
    c3.metric(
        "Instance inspection units",
        "N/A" if instance_mean is None else f"{float(instance_mean):.2f}",
    )
    c4.metric("L", config.get("L", "N/A"))
    c5.metric("topK", config.get("topK", "N/A"))

    if (
        instance_ci_lower is not None
        and instance_ci_upper is not None
        and instance_confidence is not None
    ):
        st.caption(
            f"Instance inspection {100 * float(instance_confidence):.0f}% confidence "
            f"interval: {float(instance_ci_lower):.2f}–{float(instance_ci_upper):.2f}"
        )

    if complexity_report is not None:
        st.markdown("#### Complexity report")
        st.json(complexity_report)

    st.markdown("#### Explicit configuration")
    st.json(config)

    rpte_structure = _base_estimator_structure(model)
    if rpte_structure:
        st.markdown("#### RPTE downstream-estimator complexity")
        st.caption(
            "Additional complexity/budget dimensions specific to the RPTE downstream "
            "branch: each boosted tree's depth, leaf budget, and lookahead search mode "
            "directly control how many higher-order interactions the model can represent."
        )
        st.json(rpte_structure)

    rows = [
        {"regime": "patterns_only", "inspection_meaning": "Representation is limited to selected HUG pattern features."},
        {"regime": "original_plus_patterns", "inspection_meaning": "Original business features are retained alongside HUG pattern features."},
        {"regime": "original_plus_interactions", "inspection_meaning": "Original features plus multi-feature HUG interactions."},
        {"regime": "strict budget", "inspection_meaning": "Optional mode that caps the final representation for compactness-sensitive deployments."},
        {"regime": "base_estimator=RPTE", "inspection_meaning": "Downstream branch adds boosted-tree higher-order interactions on top of the mined representation; explained via rpte_rule_table(), not feature_importances()."},
    ]
    st.dataframe(dataframe_for_display(pd.DataFrame(rows)), width="stretch", hide_index=True)

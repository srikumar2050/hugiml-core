"""Configuration comparison component."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from hugiml.dashboard.display import dataframe_for_display
from hugiml.dashboard.runner import fit_hugiml_config
from hugiml.rpte_bounded_lookahead_leafwise import LEAF_CONFIGS as RPTE_LEAF_CONFIGS
from hugiml.rpte_bounded_lookahead_leafwise import LeafWiseBoundedLookaheadRPTEFeatureLR


def _base_estimator_label(model: Any) -> str:
    """Short, readable label for a fitted or candidate model's downstream
    estimator -- 'HUGIML LR' for the built-in logistic-regression branch,
    or the wrapped estimator's class name otherwise."""
    value = getattr(model, "base_estimator", None)
    if value is None:
        return "HUGIML LR"
    inner = getattr(value, "estimator", None)
    if inner is not None:
        return f"{type(inner).__name__} via {type(value).__name__}"
    return type(value).__name__


def _family_counts(model: Any, X: pd.DataFrame | None = None) -> dict[str, int]:
    if model is None:
        return {"original": 0, "pattern": 0, "augmented": 0, "total": 0}

    meta = getattr(model, "fit_metadata_", None)
    counts = getattr(meta, "downstream_feature_counts", {}) if meta is not None else {}
    if not isinstance(counts, dict):
        counts = {}

    original = int(counts.get("original", len(getattr(model, "feature_names_in_", [])) or (X.shape[1] if X is not None else 0)))
    pattern = int(counts.get("pattern", len(getattr(model, "patterns_", [])) or len(getattr(model, "raw_patterns_", []))))
    augmented = int(counts.get("augmented_pair", len(getattr(model, "augmented_pair_transforms_", []))))
    total = original + pattern + augmented
    return {"original": original, "pattern": pattern, "augmented": augmented, "total": total}




def _fit_time_ms(model: Any, result: Any = None) -> Any:
    """Return total fit time in milliseconds when fitted metadata exposes it."""
    for obj in (getattr(model, "fit_metadata_", None), getattr(result, "fit_metadata_", None), result):
        if obj is None:
            continue
        if isinstance(obj, dict):
            value = obj.get("total_fit_ms") or obj.get("fit_time_ms")
        else:
            value = getattr(obj, "total_fit_ms", None)
            if value is None:
                value = getattr(obj, "fit_time_ms", None)
        if value is not None:
            return value
    return None

def _result_row(label: str, result: Any, X: pd.DataFrame | None = None) -> dict:
    model = getattr(result, "best_estimator_", None)
    counts = _family_counts(model, X)
    params = getattr(result, "best_params_", {}) or {}
    return {
        "run": label,
        "status": getattr(result, "status_", "ok"),
        "validation_roc_auc": getattr(result, "best_score_", None),
        "L": params.get("L", getattr(model, "L", None) if model is not None else None),
        "topK": params.get("topK", getattr(model, "topK", None) if model is not None else None),
        "G": params.get("G", getattr(model, "G", None) if model is not None else None),
        "feature_mode": params.get("feature_mode", getattr(model, "feature_mode", None) if model is not None else None),
        "strict_budget": bool(params.get("topk_budget_strict", getattr(model, "topk_budget_strict", False) if model is not None else False)),
        "downstream_estimator": _base_estimator_label(model) if model is not None else "HUGIML LR",
        "original_features": counts["original"],
        "pattern_features": counts["pattern"],
        "augmented_features": counts["augmented"],
        "displayed_total": counts["total"],
        "total_fit_ms": _fit_time_ms(model, result),
        "diagnostic": getattr(result, "error_", None),
    }


def _explain_failure(result: Any) -> None:
    status = getattr(result, "status_", "ok")
    if status == "ok":
        return

    message = getattr(result, "error_", "") or ""
    st.error("Candidate configuration did not produce a fitted model.")
    if status == "no_patterns":
        st.warning(
            "No HUG patterns were mined for this candidate configuration. "
            "`original_plus_patterns` still requires at least one selected pattern. "
            "Try lowering `G`, increasing `topK`, using `L=1/2` appropriately, or using `patterns_only` only when patterns are expected."
        )
    else:
        st.warning("The candidate fit failed. See diagnostic details below.")

    with st.expander("Candidate failure diagnostic", expanded=True):
        st.code(message)


def render_config_comparison(ctx: dict, *args, **kwargs) -> None:
    st.subheader("Configuration Comparison")
    st.markdown(
        """
        <div class="hugiml-section-note">
          <p>This is a different HUGIML configuration run, not pruning. Strict budget is a supported
          configuration option and is passed explicitly into the candidate run. If a candidate mines
          zero patterns, the run is reported as invalid instead of crashing the dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    base_model = ctx["model"]
    X = ctx["X"]
    y = ctx["y"]

    with st.container(border=True):
        st.markdown("#### Candidate HUGIML configuration")
        c1, c2, c3 = st.columns(3)
        L = int(c1.selectbox("L", [1, 2], index=1 if getattr(base_model, "L", 2) == 2 else 0))
        topK = int(c2.number_input("topK", min_value=1, max_value=500, value=int(getattr(base_model, "topK", 50) or 50), step=5))
        G = float(c3.selectbox("G", [1e-1, 1e-2, 1e-3, 1e-4], index=1))

        c4, c5 = st.columns([0.7, 0.3])
        feature_mode = c4.selectbox(
            "feature_mode",
            ["patterns_only", "original_plus_patterns", "original_plus_interactions"],
            index=1,
        )
        strict = c5.checkbox(
            "Strict budget",
            value=bool(getattr(base_model, "topk_budget_strict", False)),
            help="Supported HUGIML setting. When enabled, the final representation is constrained to the configured topK budget where applicable.",
        )

        st.markdown("###### Downstream estimator")
        downstream_choice = st.radio(
            "Downstream estimator",
            ["HUGIML logistic regression", "RPTE (bounded-lookahead trees)"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            help="RPTE (leaf-wise bounded-lookahead trees + logistic regression) searches "
                 "higher-order feature interactions the logistic branch can't represent directly.",
            key="cc_downstream_estimator",
        )
        base_estimator = None
        if downstream_choice.startswith("RPTE"):
            r1, r2, r3 = st.columns(3)
            leaf_config_options = sorted(RPTE_LEAF_CONFIGS)
            rpte_leaf_config = r1.selectbox(
                "Leaf config", leaf_config_options,
                index=leaf_config_options.index("3xD") if "3xD" in leaf_config_options else 0,
                key="cc_rpte_leaf_config",
            )
            rpte_depth = int(r2.number_input("Depth", value=4, min_value=1, max_value=10, step=1, key="cc_rpte_depth"))
            rpte_n_estimators = int(r3.number_input("n_estimators", value=10, min_value=1, max_value=100, step=1, key="cc_rpte_n_estimators"))
            base_estimator = LeafWiseBoundedLookaheadRPTEFeatureLR(
                leaf_config=rpte_leaf_config, depth=rpte_depth, n_estimators=rpte_n_estimators,
                enable_lookahead="adaptive",
            )

        params = {
            "adaptive_binning": True,
            "B": -1,
            "L": L,
            "topK": topK,
            "G": G,
            "feature_mode": feature_mode,
            "topk_budget_strict": bool(strict),
            "base_estimator": base_estimator,
        }

        run = st.button("Fit candidate configuration", type="primary", width="stretch")

    if run:
        with st.spinner("Fitting candidate configuration..."):
            result = fit_hugiml_config(
                X,
                y,
                params=params,
                cv=int(ctx.get("cv", 3)),
                scoring="roc_auc",
                random_state=int(ctx.get("random_state", 2026)),
                raise_on_error=False,
            )
            st.session_state["hugiml_config_compare_result"] = result

    candidate = st.session_state.get("hugiml_config_compare_result")
    if candidate is None:
        st.info("Fit a candidate configuration to compare it with the current selected model.")
        return

    _explain_failure(candidate)

    rows = [
        _result_row("Current selected model", ctx["result"], X),
        _result_row("Candidate configuration", candidate, X),
    ]
    df = pd.DataFrame(rows)

    if "validation_roc_auc" in df.columns and df["validation_roc_auc"].notna().all():
        base_score = float(df.loc[0, "validation_roc_auc"])
        cand_score = float(df.loc[1, "validation_roc_auc"])
        df["validation_roc_auc_delta_vs_current"] = [0.0, cand_score - base_score]

    st.markdown("#### Comparison")
    st.dataframe(dataframe_for_display(df), width="stretch", hide_index=True)

    candidate_rows = pd.DataFrame(getattr(candidate, "results_", []))
    if not candidate_rows.empty:
        st.markdown("#### Candidate fold / diagnostic evidence")
        st.dataframe(candidate_rows, width="stretch", hide_index=True)

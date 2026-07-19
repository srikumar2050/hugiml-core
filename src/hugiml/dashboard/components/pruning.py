"""Representation pruning analysis component."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from hugiml.dashboard.components.patterns import _get_rpte_feature_flow_audit
from hugiml.dashboard.components.rpte_governance import (
    rpte_direct_source_terms_frame,
    rpte_model_comparison_row,
    rpte_raw_input_lineage_frame,
)
from hugiml.dashboard.display import dataframe_for_display
from hugiml.dashboard.runner import fit_feature_pruned_hugiml, fit_representation_pruned_downstream


def _safe_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except Exception:
        return []


def _infer_source_features_from_label(label: str) -> list[str]:
    import re

    s = str(label)
    matches = re.findall(r"(?:^|,\s*)([^=,\[\]]+?)\s*=", s)
    return [m.strip() for m in matches if m.strip()]


def _selected_original_features(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    names = _safe_list(getattr(model, "feature_names_in_", None)) or list(X.columns)
    names = [str(c) for c in names if str(c) in set(map(str, X.columns))]
    return pd.DataFrame({"feature": names, "family": "Original features"})


def _fitted_downstream_names(model: Any) -> list[str]:
    getter = getattr(model, "get_downstream_features", None)
    if callable(getter):
        try:
            return [str(value) for value in list(getter() or [])]
        except Exception:
            pass
    return [
        str(value)
        for value in _safe_list(getattr(model, "_downstream_feature_names_", None))
    ]


def _family_name_lookup(model: Any, prefix: str) -> tuple[dict[str, str], bool]:
    names = _fitted_downstream_names(model)
    selected = [name for name in names if name.startswith(prefix)]
    lookup: dict[str, str] = {}
    for name in selected:
        display = name[len(prefix) :]
        lookup[name] = name
        lookup[display] = name
    return lookup, bool(names)


def _pattern_inventory(model: Any) -> pd.DataFrame:
    lookup, names_available = _family_name_lookup(model, "pattern:")
    try:
        if hasattr(model, "get_pattern_info"):
            info = model.get_pattern_info()
            df = info if isinstance(info, pd.DataFrame) else pd.DataFrame(info)
            if not df.empty:
                label_col = next((c for c in df.columns if str(c).lower() in {"pattern", "label", "name", "feature"}), df.columns[0])
                out = df.copy()
                out["_label"] = out[label_col].astype(str)
                out["_source_features"] = out["_label"].map(lambda s: ", ".join(_infer_source_features_from_label(s)))
                out["_repr_column"] = out["_label"].map(lookup)
                if not names_available:
                    out["_repr_column"] = out["_label"]
                else:
                    out = out.loc[out["_repr_column"].notna()].copy()
                return out
    except Exception:
        pass

    labels = None
    get_hug_features = getattr(model, "get_hug_features", None)
    if callable(get_hug_features):
        try:
            labels = get_hug_features()
        except Exception:
            labels = None
    if labels is None:
        for attr in ("pattern_labels_", "_pattern_labels_", "patterns_", "raw_patterns_"):
            try:
                labels = getattr(model, attr, None)
                if labels is not None:
                    break
            except Exception:
                labels = None

    rows = []
    for i, label in enumerate(_safe_list(labels), start=1):
        text = str(label)
        representation = lookup.get(text)
        if names_available and representation is None:
            continue
        rows.append({
            "rank": i,
            "_label": text,
            "_source_features": ", ".join(_infer_source_features_from_label(text)),
            "_repr_column": representation or text,
        })
    return pd.DataFrame(rows)


def _selected_augmented_features(model: Any) -> pd.DataFrame:
    """Return augmented features aligned to fitted downstream column names."""
    lookup, names_available = _family_name_lookup(model, "augmented_pair:")
    rows = []
    represented: set[str] = set()

    for i, spec in enumerate(_safe_list(getattr(model, "augmented_pair_transforms_", None)), start=1):
        if isinstance(spec, dict):
            name = str(spec.get("name", f"augmented_{i}"))
            representation = lookup.get(name)
            if names_available and representation is None:
                continue
            representation = representation or name
            represented.add(representation)
            rows.append({
                "rank": len(rows) + 1,
                "feature": name,
                "family": "Augmented features",
                "operation": str(spec.get("operation", "")),
                "source_features": ", ".join(map(str, spec.get("inputs", []))),
                "repr_column": representation,
            })

    if not rows:
        for attr in ("selected_augmented_features_", "augmented_feature_names_", "augmented_columns_", "augmented_pair_feature_names_"):
            values = _safe_list(getattr(model, attr, None))
            if values:
                for name_value in values:
                    name = str(name_value)
                    representation = lookup.get(name)
                    if names_available and representation is None:
                        continue
                    representation = representation or name
                    represented.add(representation)
                    rows.append({
                        "rank": len(rows) + 1,
                        "feature": name,
                        "family": "Augmented features",
                        "operation": "",
                        "source_features": "",
                        "repr_column": representation,
                    })
                break

    for representation in sorted(set(lookup.values())):
        if representation in represented:
            continue
        rows.append({
            "rank": len(rows) + 1,
            "feature": representation[len("augmented_pair:") :],
            "family": "Augmented features",
            "operation": "",
            "source_features": "",
            "repr_column": representation,
        })

    return pd.DataFrame(rows)


def _downstream_columns(model: Any, X: pd.DataFrame) -> list[str]:
    candidates = ["transform_downstream", "transform_features", "transform", "_transform_downstream", "_transform_features"]
    for name in candidates:
        fn = getattr(model, name, None)
        if fn is None:
            continue
        try:
            Z = fn(X.head(3))
            if hasattr(Z, "toarray"):
                Z = Z.toarray()
            if isinstance(Z, pd.DataFrame):
                return list(map(str, Z.columns))
            arr = np.asarray(Z)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return [f"repr_{i+1}" for i in range(arr.shape[1])]
        except Exception:
            continue
    return []


def _filter_selected_by_downstream(df: pd.DataFrame, model: Any, X: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Keep only rows whose label/column appears in the final downstream representation when columns are exposed.

    If downstream column names are not exposed, do not invent filtering. Return the fitted
    inventory as exposed by the model, which should already be selected patterns/transforms.
    """
    if df.empty:
        return df
    cols = _downstream_columns(model, X)
    if not cols:
        return df
    colset = set(cols)
    mask = df[label_col].astype(str).map(lambda s: s in colset)
    if mask.any():
        return df.loc[mask].copy()
    return df


def _base_summary_row(ctx: dict) -> dict:
    model = ctx["model"]
    X = ctx["X"]
    meta = getattr(model, "fit_metadata_", None)
    counts = getattr(meta, "downstream_feature_counts", {}) if meta is not None else {}
    if not isinstance(counts, dict):
        counts = {}
    original = int(counts.get("original", len(getattr(model, "feature_names_in_", [])) or X.shape[1]))
    pattern = int(counts.get("pattern", len(getattr(model, "patterns_", [])) or len(getattr(model, "raw_patterns_", []))))
    augmented = int(counts.get("augmented_pair", len(getattr(model, "augmented_pair_transforms_", []))))
    return {
        "run": "Current selected model",
        "cv_score": getattr(ctx["result"], "best_score_", None),
        "input_features": int(X.shape[1]),
        "removed_items": 0,
        "original_features": original,
        "pattern_features": pattern,
        "augmented_features": augmented,
        "displayed_total": original + pattern + augmented,
    }


def _result_summary_row(label: str, score: float | None, input_features: int, removed_count: int, original: int, pattern: int, augmented: int) -> dict:
    return {
        "run": label,
        "cv_score": score,
        "input_features": int(input_features),
        "removed_items": int(removed_count),
        "original_features": int(original),
        "pattern_features": int(pattern),
        "augmented_features": int(augmented),
        "displayed_total": int(original + pattern + augmented),
    }


def _render_standard_pruning_analysis(ctx: dict, *args, **kwargs) -> None:
    st.subheader("Representation Pruning")
    st.markdown(
        """
        <div class="hugiml-section-note">
          <p>Select one HUGIML representation family to prune. The selector below shows only the
          chosen family: original features, HUG patterns, or augmented features. The lists are based
          on the fitted model's selected representation metadata, not all possible candidates.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    X = ctx["X"]
    y = ctx["y"]
    model = ctx["model"]

    family = st.radio(
        "Representation family to prune",
        ["Original features", "HUG patterns", "Augmented features"],
        horizontal=True,
    )

    base_row = _base_summary_row(ctx)

    if family == "Original features":
        original_df = _selected_original_features(model, X)
        st.markdown("#### Selected original features")
        st.dataframe(original_df, width="stretch", hide_index=True)

        selected = st.multiselect(
            "Original features to remove and rerun",
            options=original_df["feature"].astype(str).tolist(),
            default=[],
        )

        if not selected:
            st.info("Select one or more original features to prune.")
            return

        if len(selected) >= X.shape[1]:
            st.error("Cannot remove all original input features.")
            return

        run = st.button("Run original-feature pruning", type="primary", width="stretch")
        if run:
            with st.spinner("Rerunning HUGIML without selected original features..."):
                result, X_pruned = fit_feature_pruned_hugiml(
                    X,
                    y,
                    base_model=model,
                    remove_features=selected,
                    cv=int(ctx.get("cv", 3)),
                    scoring="roc_auc",
                    random_state=int(ctx.get("random_state", 2026)),
                )
                st.session_state["hugiml_prune_result"] = ("Original features", result, X_pruned, selected)

    elif family == "HUG patterns":
        pattern_df = _pattern_inventory(model)
        if pattern_df.empty:
            st.info("The fitted model did not expose selected HUG pattern metadata.")
            return
        pattern_df = _filter_selected_by_downstream(pattern_df, model, X, "_repr_column")

        st.markdown("#### Selected HUG patterns")
        display_cols = [
            c
            for c in [
                "rank",
                "_label",
                "_source_features",
                "pattern_origin",
                "survivor_led",
                "survivor_features",
                "survivor_feature_count",
            ]
            if c in pattern_df.columns
        ]
        st.dataframe(
            pattern_df[display_cols].rename(
                columns={"_label": "pattern", "_source_features": "source_features"}
            ),
            width="stretch",
            hide_index=True,
        )

        selected = st.multiselect(
            "HUG patterns to remove from downstream representation",
            options=pattern_df["_repr_column"].astype(str).tolist(),
            default=[],
        )

        if not selected:
            st.info("Select one or more selected HUG patterns to prune.")
            return

        run = st.button("Run HUG-pattern representation pruning", type="primary", width="stretch")
        if run:
            with st.spinner("Refitting downstream estimator without selected HUG pattern columns..."):
                result, Z_pruned = fit_representation_pruned_downstream(
                    model,
                    X,
                    y,
                    remove_columns=selected,
                    family="HUG patterns",
                    cv=int(ctx.get("cv", 3)),
                    scoring="roc_auc",
                    random_state=int(ctx.get("random_state", 2026)),
                )
                st.session_state["hugiml_prune_result"] = ("HUG patterns", result, Z_pruned, selected)

    else:
        aug_df = _selected_augmented_features(model)
        if aug_df.empty:
            st.info("No selected augmented features are recorded in the fitted model.")
            return
        aug_df = _filter_selected_by_downstream(aug_df, model, X, "repr_column")

        st.markdown("#### Selected augmented features")
        st.dataframe(aug_df, width="stretch", hide_index=True)

        selected = st.multiselect(
            "Augmented features to remove from downstream representation",
            options=aug_df["repr_column"].astype(str).tolist(),
            default=[],
        )

        if not selected:
            st.info("Select one or more selected augmented features to prune.")
            return

        run = st.button("Run augmented-feature representation pruning", type="primary", width="stretch")
        if run:
            with st.spinner("Refitting downstream estimator without selected augmented-feature columns..."):
                result, Z_pruned = fit_representation_pruned_downstream(
                    model,
                    X,
                    y,
                    remove_columns=selected,
                    family="Augmented features",
                    cv=int(ctx.get("cv", 3)),
                    scoring="roc_auc",
                    random_state=int(ctx.get("random_state", 2026)),
                )
                st.session_state["hugiml_prune_result"] = ("Augmented features", result, Z_pruned, selected)

    stored = st.session_state.get("hugiml_prune_result")
    if stored is None:
        return

    stored_family, result, pruned_frame, removed = stored

    if hasattr(result, "best_estimator_"):
        # Source-feature pruning reran HUGIML.
        if getattr(result, "best_estimator_", None) is None:
            st.error("The pruned HUGIML rerun did not produce a fitted model.")
            if getattr(result, "status_", "") == "no_patterns":
                st.warning(
                    "The pruned rerun mined zero HUG patterns. Try pruning fewer original features, "
                    "lowering G in Configuration Comparison, or using a configuration that produces patterns."
                )
            with st.expander("Pruned-run diagnostic", expanded=True):
                st.code(getattr(result, "error_", "No diagnostic available."))
            return
        pruned_model = result.best_estimator_
        meta = getattr(pruned_model, "fit_metadata_", None)
        counts = getattr(meta, "downstream_feature_counts", {}) if meta is not None else {}
        if not isinstance(counts, dict):
            counts = {}
        original = int(counts.get("original", len(getattr(pruned_model, "feature_names_in_", [])) or pruned_frame.shape[1]))
        pattern = int(counts.get("pattern", len(getattr(pruned_model, "patterns_", [])) or len(getattr(pruned_model, "raw_patterns_", []))))
        augmented = int(counts.get("augmented_pair", len(getattr(pruned_model, "augmented_pair_transforms_", []))))
        score = getattr(result, "best_score_", None)
        label = "Original-feature pruned rerun"
    else:
        # Representation-column pruning refit downstream estimator.
        original = base_row["original_features"]
        pattern = base_row["pattern_features"]
        augmented = base_row["augmented_features"]
        if stored_family == "HUG patterns":
            pattern = max(0, pattern - len(removed))
        elif stored_family == "Augmented features":
            augmented = max(0, augmented - len(removed))
        score = getattr(result, "score", None)
        label = f"{stored_family} representation-pruned downstream"

    pruned_row = _result_summary_row(
        label=label,
        score=score,
        input_features=pruned_frame.shape[1] if stored_family == "Original features" else X.shape[1],
        removed_count=len(removed),
        original=original,
        pattern=pattern,
        augmented=augmented,
    )

    df = pd.DataFrame([base_row, pruned_row])
    if df["cv_score"].notna().all():
        df["score_delta_vs_current"] = [0.0, float(df.loc[1, "cv_score"]) - float(df.loc[0, "cv_score"])]

    base_total = max(int(base_row["displayed_total"]), 1)
    df["displayed_representation_reduction_pct"] = [
        0.0,
        round((base_total - int(pruned_row["displayed_total"])) / base_total * 100.0, 3),
    ]

    st.markdown("#### Pruning comparison")
    if "cv_score" in df.columns and df["cv_score"].notna().sum() >= 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        _score_df = df[["run", "cv_score"]].dropna(subset=["cv_score"]).copy()
        if len(_score_df) >= 1:
            _fig_p, _ax_p = _plt.subplots(figsize=(7, max(1.8, len(_score_df) * 0.55)))
            _colors_p = ["#AFA9EC" if i == 0 else "#534AB7" for i in range(len(_score_df))]
            _ax_p.barh(_score_df["run"].astype(str), _score_df["cv_score"].astype(float), color=_colors_p)
            _ax_p.set_xlabel("CV ROC-AUC")
            _ax_p.set_title("CV score — current vs pruned")
            _ax_p.set_xlim(max(0, float(_score_df["cv_score"].min()) - 0.05), 1.0)
            _ax_p.invert_yaxis()
            _fig_p.tight_layout()
            st.pyplot(_fig_p)
            _plt.close(_fig_p)
    st.dataframe(dataframe_for_display(df), width="stretch", hide_index=True)

    st.markdown("#### Removed representation items")
    st.dataframe(dataframe_for_display(pd.DataFrame({"removed_item": removed, "family": stored_family})), width="stretch", hide_index=True)



def _render_rpte_rebuild_analysis(ctx: dict) -> None:
    """Render RPTE-aware final-term review and full-pipeline sensitivity controls."""
    X = ctx["X"]
    y = ctx["y"]
    model = ctx["model"]
    flow = _get_rpte_feature_flow_audit(model)
    representation = flow.get("final_representation")
    has_leaf_block = representation in {
        "rpte_leaf_rules",
        "rpte_leaves_plus_direct_terms",
    }

    st.subheader(
        "RPTE Representation Review & Rebuild"
        if has_leaf_block
        else "RPTE Source Fallback Review & Rebuild"
    )
    if has_leaf_block:
        st.markdown(
            """
            <div class="hugiml-section-note">
              <p>The final LR contains RPTE leaf indicators. Source columns not used in accepted
              splits are appended directly after the leaf block. The direct terms below are filtered
              to direct original features, HUG patterns, and augmented pairs only; split-used columns are
              represented through leaf indicators and are not repeated.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "Removing an individual direct term with a generic source-matrix LR refit is not offered, "
            "because that operation would discard the fitted RPTE leaf block. Raw-input exclusion below "
            "reruns mining, pair generation, RPTE construction, leaf encoding, direct-term selection, "
            "and the final LR together."
        )
    else:
        st.markdown(
            """
            <div class="hugiml-section-note">
              <p>This fitted object uses the explicit HUGIML source fallback because RPTE formed no
              valid tree. The displayed coefficients are direct source terms rather than RPTE leaves.
              The supported sensitivity action remains a complete pipeline rebuild.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    direct = rpte_direct_source_terms_frame(model, include_zero=True)
    if has_leaf_block:
        st.markdown("#### Direct source terms in final LR")
        tabs = st.tabs(["Direct original features", "Direct HUG patterns", "Direct augmented pairs"])
        family_specs = [
            ("original", "original feature"),
            ("pattern", "HUG pattern"),
            ("augmented_pair", "augmented pair"),
        ]
        for tab, (family, label) in zip(tabs, family_specs):
            with tab:
                family_df = direct.loc[direct.get("family", pd.Series(index=direct.index, dtype=str)).eq(family)].copy()
                if family_df.empty:
                    st.info(f"No direct {label} terms enter the final LR.")
                    continue
                cols = [
                    "class", "display_name", "source_column", "raw_sources",
                    "coefficient", "odds_multiplier", "nonzero",
                ]
                st.dataframe(
                    dataframe_for_display(family_df[[c for c in cols if c in family_df.columns]]),
                    width="stretch",
                    hide_index=True,
                )

    lineage = rpte_raw_input_lineage_frame(model, X)
    st.markdown("#### Raw-input lineage before rebuild")
    st.caption(
        "Each raw input is traced to source columns used in RPTE splits, direct source terms "
        "carried directly into LR, and leaf rules that reference it."
    )
    if lineage.empty:
        st.info("Raw-input lineage is not available for this fitted model.")
        return
    st.dataframe(dataframe_for_display(lineage), width="stretch", hide_index=True)

    selected = st.multiselect(
        "Raw model inputs to exclude from the full rebuild",
        options=lineage["raw_input"].astype(str).tolist(),
        default=[],
        key="hugiml_rpte_raw_input_exclusions",
        help=(
            "Exclusion reruns preprocessing, HUG pattern mining, augmented-pair selection, "
            "RPTE tree construction, leaf encoding, and the final logistic regression."
        ),
    )
    if not selected:
        st.caption("Select one or more raw model inputs to run a full RPTE rebuild sensitivity test.")
    elif len(selected) >= X.shape[1]:
        st.error("Cannot exclude all raw model inputs.")
    else:
        impact = lineage.loc[lineage["raw_input"].astype(str).isin(set(map(str, selected)))].copy()
        st.markdown("#### Selected-input dependency impact")
        st.dataframe(dataframe_for_display(impact), width="stretch", hide_index=True)
        run = st.button(
            "Run full RPTE rebuild without selected raw inputs",
            type="primary",
            width="stretch",
            key="hugiml_run_rpte_raw_input_rebuild",
        )
        if run:
            with st.spinner(
                "Rebuilding preprocessing, HUGIML source representation, RPTE trees, leaves, and final LR..."
            ):
                result, X_pruned = fit_feature_pruned_hugiml(
                    X,
                    y,
                    base_model=model,
                    remove_features=selected,
                    cv=int(ctx.get("cv", 3)),
                    scoring="roc_auc",
                    random_state=int(ctx.get("random_state", 2026)),
                )
                st.session_state["hugiml_rpte_rebuild_result"] = {
                    "result": result,
                    "X": X_pruned,
                    "removed": list(selected),
                    "impact": impact,
                }

    stored = st.session_state.get("hugiml_rpte_rebuild_result")
    if not isinstance(stored, dict):
        return
    result = stored.get("result")
    X_pruned = stored.get("X")
    removed = list(stored.get("removed") or [])
    impact = stored.get("impact")
    if result is None or X_pruned is None:
        return
    pruned_model = getattr(result, "best_estimator_", None)
    if pruned_model is None:
        st.error("The RPTE rebuild did not produce a fitted model.")
        if getattr(result, "status_", "") == "no_patterns":
            st.warning(
                "The reduced raw-input set mined no usable HUG patterns. Reduce the exclusion set "
                "or compare a different G/topK/feature_mode configuration."
            )
        with st.expander("Rebuild diagnostic", expanded=True):
            st.code(getattr(result, "error_", "No diagnostic available."))
        return

    base_row = rpte_model_comparison_row(
        "Current promoted RPTE model",
        model,
        getattr(ctx.get("result"), "best_score_", None),
        X,
    )
    rebuilt_row = rpte_model_comparison_row(
        "Full rebuild after raw-input exclusion",
        pruned_model,
        getattr(result, "best_score_", None),
        X_pruned,
    )
    comparison = pd.DataFrame([base_row, rebuilt_row])
    if comparison["validation_roc_auc"].notna().all():
        comparison["validation_roc_auc_delta_vs_current"] = [
            0.0,
            float(comparison.loc[1, "validation_roc_auc"])
            - float(comparison.loc[0, "validation_roc_auc"]),
        ]
    comparison["raw_input_delta"] = [
        0,
        int(comparison.loc[1, "raw_input_features"])
        - int(comparison.loc[0, "raw_input_features"]),
    ]
    comparison["rpte_source_column_delta"] = [
        0,
        int(comparison.loc[1, "rpte_source_total"])
        - int(comparison.loc[0, "rpte_source_total"]),
    ]
    comparison["rpte_leaf_term_delta"] = [
        0,
        int(comparison.loc[1, "rpte_leaf_terms"])
        - int(comparison.loc[0, "rpte_leaf_terms"]),
    ]
    comparison["rpte_direct_source_term_delta"] = [
        0,
        int(comparison.loc[1, "rpte_direct_source_terms"])
        - int(comparison.loc[0, "rpte_direct_source_terms"]),
    ]

    st.markdown("#### Full-rebuild comparison")
    st.caption(
        "Source-column, leaf-term, and direct-source term counts are shown separately. "
        "A smaller source matrix can change both the reconstructed trees and which source columns "
        "are not selected in accepted splits and enter the final LR directly."
    )
    st.dataframe(dataframe_for_display(comparison), width="stretch", hide_index=True)

    st.markdown("#### Excluded raw inputs")
    removed_df = pd.DataFrame({"raw_input_excluded": removed})
    if isinstance(impact, pd.DataFrame) and not impact.empty:
        removed_df = removed_df.merge(
            impact,
            left_on="raw_input_excluded",
            right_on="raw_input",
            how="left",
        ).drop(columns=["raw_input"], errors="ignore")
    st.dataframe(dataframe_for_display(removed_df), width="stretch", hide_index=True)


def render_pruning_analysis(ctx: dict, *args, **kwargs) -> None:
    """Render model-family-valid pruning/rebuild controls."""
    flow = _get_rpte_feature_flow_audit(ctx.get("model"))
    if flow:
        _render_rpte_rebuild_analysis(ctx)
        return
    _render_standard_pruning_analysis(ctx, *args, **kwargs)

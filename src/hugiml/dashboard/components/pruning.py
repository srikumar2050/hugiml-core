"""Representation pruning analysis component."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

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


def _pattern_inventory(model: Any) -> pd.DataFrame:
    try:
        if hasattr(model, "get_pattern_info"):
            info = model.get_pattern_info()
            df = info if isinstance(info, pd.DataFrame) else pd.DataFrame(info)
            if not df.empty:
                label_col = next((c for c in df.columns if str(c).lower() in {"pattern", "label", "name", "feature"}), df.columns[0])
                out = df.copy()
                out["_label"] = out[label_col].astype(str)
                out["_source_features"] = out["_label"].map(lambda s: ", ".join(_infer_source_features_from_label(s)))
                out["_repr_column"] = out["_label"]
                return out
    except Exception:
        pass

    labels = None
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
        rows.append({
            "rank": i,
            "_label": text,
            "_source_features": ", ".join(_infer_source_features_from_label(text)),
            "_repr_column": text,
        })
    return pd.DataFrame(rows)


def _selected_augmented_features(model: Any) -> pd.DataFrame:
    """Return only augmented features actually recorded/selected by the fitted model."""
    rows = []

    # Preferred source: selected augmented pair transforms recorded by fit.
    for i, spec in enumerate(_safe_list(getattr(model, "augmented_pair_transforms_", None)), start=1):
        if isinstance(spec, dict):
            name = str(spec.get("name", f"augmented_{i}"))
            rows.append({
                "rank": i,
                "feature": name,
                "family": "Augmented features",
                "operation": str(spec.get("operation", "")),
                "source_features": ", ".join(map(str, spec.get("inputs", []))),
                "repr_column": name,
            })

    # Secondary sources used by different versions.
    if not rows:
        for attr in ("selected_augmented_features_", "augmented_feature_names_", "augmented_columns_", "augmented_pair_feature_names_"):
            values = _safe_list(getattr(model, attr, None))
            if values:
                for i, name in enumerate(values, start=1):
                    rows.append({
                        "rank": i,
                        "feature": str(name),
                        "family": "Augmented features",
                        "operation": "",
                        "source_features": "",
                        "repr_column": str(name),
                    })
                break

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


def render_pruning_analysis(ctx: dict, *args, **kwargs) -> None:
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
        display_cols = [c for c in ["rank", "_label", "_source_features"] if c in pattern_df.columns]
        st.dataframe(pattern_df[display_cols].rename(columns={"_label": "pattern", "_source_features": "source_features"}), width="stretch", hide_index=True)

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
    st.dataframe(dataframe_for_display(df), width="stretch", hide_index=True)

    st.markdown("#### Removed representation items")
    st.dataframe(dataframe_for_display(pd.DataFrame({"removed_item": removed, "family": stored_family})), width="stretch", hide_index=True)

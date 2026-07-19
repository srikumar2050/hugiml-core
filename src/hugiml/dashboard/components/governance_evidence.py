"""High-value governance evidence panels for the HUGIML dashboard.

These panels surface model evidence that is already produced by fitted HUGIML
models, without adding new model APIs. Every renderer is deliberately guarded so
older model objects or non-HUGIML baselines degrade to an explanatory info banner
instead of failing the dashboard.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from hugiml.dashboard.components.patterns import (
    _get_rpte_feature_flow_audit,
    _get_rpte_final_term_rows,
    _rpte_rules_to_frame,
    render_rpte_flat_tree_view,
)
from hugiml.dashboard.display import dataframe_for_display

try:
    from hugiml.plots import HUGPlotter
except Exception:  # pragma: no cover - optional Plotly dependency/model API
    HUGPlotter = None  # type: ignore[assignment]


def _safe_dataframe(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value)
    except Exception:
        return pd.DataFrame()




def _is_enabled_flag(value: Any) -> bool:
    """Safely coerce model flags without triggering NumPy ambiguous truth errors."""
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        if value.size == 1:
            return bool(value.reshape(-1)[0])
        return bool(np.any(value))
    try:
        return bool(value)
    except ValueError:
        try:
            return bool(np.any(value))
        except Exception:
            return False


def _mapping_from_pairs_or_sequence(value: Any, feature_names: list[str] | None = None) -> dict[Any, Any]:
    """Convert common metadata shapes to a mapping without relying on ambiguous truthiness."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, pd.DataFrame):
        if value.shape[1] >= 2:
            return dict(zip(value.iloc[:, 0], value.iloc[:, 1]))
        return {}
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        return {}
    if feature_names and len(feature_names) == len(arr):
        return dict(zip(feature_names, arr.tolist()))
    try:
        return dict(value)
    except Exception:
        return {}


def _extract_figure(value: Any) -> Any | None:
    """Return a matplotlib Figure from common HUGIML plotting return shapes."""
    if value is None:
        return None
    # HUGIML plotting helpers commonly return (fig, ax). Some matplotlib calls
    # may return just the figure or nested lists/tuples when multiple axes exist.
    if hasattr(value, "savefig"):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            fig = _extract_figure(item)
            if fig is not None:
                return fig
    return None


def _render_model_figure(model: Any, method_name: str, title: str) -> bool:
    method = getattr(model, method_name, None)
    if not callable(method):
        st.info(f"{title} is not available for this installed model version.")
        return False
    try:
        fig = _extract_figure(method())
    except Exception as exc:
        st.info(f"{title} could not be rendered: {exc}")
        return False
    if fig is None:
        st.info(f"{title} did not return a matplotlib figure.")
        return False
    st.pyplot(fig)
    return True


def adaptive_binning_table(model: Any) -> pd.DataFrame:
    """Return long-form adaptive-binning evidence from per-feature B and IG scores."""
    chosen = getattr(model, "per_feature_b_", None)
    ig_scores = getattr(model, "ig_scores_", None)
    feature_names = None
    try:
        names = getattr(model, "feature_names_in_", None)
        feature_names = [str(x) for x in list(names)] if names is not None else None
    except Exception:
        feature_names = None

    chosen_map = _mapping_from_pairs_or_sequence(chosen, feature_names)
    scores_map = _mapping_from_pairs_or_sequence(ig_scores, feature_names)
    features = sorted(set(chosen_map) | set(scores_map), key=str)

    rows: list[dict[str, Any]] = []
    for feature in features:
        chosen_b = chosen_map.get(feature, np.nan)
        curve = scores_map.get(feature, {})
        if isinstance(curve, dict) and len(curve) > 0:
            for b_candidate, score in curve.items():
                rows.append({
                    "feature": str(feature),
                    "chosen_B": chosen_b,
                    "B_candidate": b_candidate,
                    "ig_score": score,
                    "is_chosen_B": bool(str(b_candidate) == str(chosen_b)),
                })
        else:
            rows.append({
                "feature": str(feature),
                "chosen_B": chosen_b,
                "B_candidate": np.nan,
                "ig_score": np.nan,
                "is_chosen_B": False,
            })
    return pd.DataFrame(rows)






def _available_profile_features(model: Any) -> list[str]:
    """Return features that have singleton pattern/effect evidence."""
    features: list[str] = []
    try:
        fi = _safe_dataframe(model.feature_importances()) if callable(getattr(model, "feature_importances", None)) else pd.DataFrame()
    except Exception:
        fi = pd.DataFrame()
    if not fi.empty:
        label_col = _first_existing_column(fi, ["display_name", "pattern", "feature"])
        type_col = _first_existing_column(fi, ["feature_type", "type"])
        if label_col:
            rows = fi
            if type_col:
                rows = rows[rows[type_col].astype(str).str.lower().eq("pattern")]
            for label in rows[label_col].astype(str).tolist():
                parts = _split_pattern_assignments(label)
                if len(parts) != 1 or "=" not in parts[0]:
                    continue
                feature = parts[0].split("=", 1)[0].strip()
                if feature and feature not in features:
                    features.append(feature)
    if features:
        return features

    # Fallback: model schema + singleton HUG labels for older/deserialized models.
    names = getattr(model, "feature_names_in_", None)
    try:
        if names is not None:
            features = [str(x) for x in list(names)]
    except Exception:
        features = []
    try:
        labels = model.get_hug_features() if callable(getattr(model, "get_hug_features", None)) else []
        for label in labels:
            text = str(label)
            parts = _split_pattern_assignments(text)
            if len(parts) != 1 or "=" not in parts[0]:
                continue
            feature = parts[0].split("=", 1)[0].strip()
            if feature and feature not in features:
                features.append(feature)
    except Exception:
        pass
    return features


def _first_existing_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _split_pattern_assignments(label: str) -> list[str]:
    """Split a HUG pattern label into top-level feature assignments.

    Numeric interval labels contain commas, e.g. ``income=[38500,55000)``, so a
    plain string split would incorrectly treat a singleton bin as a compound
    pattern. This parser splits only on commas outside brackets.
    """
    text = str(label)
    parts: list[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(text):
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            part = text[start:i].strip()
            if part:
                parts.append(part)
            start = i + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_interval_lower(label: str) -> float | None:
    value = str(label).strip()
    if "=" in value:
        value = value.split("=", 1)[1].strip()
    if not value.startswith("["):
        return None
    try:
        return float(value[1:].split(",", 1)[0].strip())
    except Exception:
        return None


def _format_edge(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if np.isfinite(val):
        return f"{val:.6g}"
    return str(value)


def _bin_label_from_edges(lo: float, hi: float, is_last: bool = False) -> str:
    close = "]" if is_last else ")"
    return f"[{_format_edge(lo)}, {_format_edge(hi)}{close}"


def _support_from_X(X: Any, feature: str, lo: float, hi: float, is_last: bool = False) -> float:
    if X is None:
        return np.nan
    try:
        series = X[feature] if isinstance(X, pd.DataFrame) and feature in X.columns else None
        if series is None:
            return np.nan
        vals = pd.to_numeric(series, errors="coerce")
        mask = vals.ge(lo) & (vals.le(hi) if is_last else vals.lt(hi))
        valid = vals.notna()
        denom = int(valid.sum())
        return float(mask[valid].mean()) if denom else np.nan
    except Exception:
        return np.nan


def _bin_edges_for_feature(model: Any, feature: str) -> np.ndarray | None:
    for attr in ("_bin_edges_", "bin_edges_"):
        edges = getattr(model, attr, None)
        if isinstance(edges, dict) and feature in edges:
            arr = np.asarray(edges[feature], dtype=float)
            return arr if arr.size >= 2 else None
    return None


def feature_shape_frame(model: Any, feature: str, X: Any = None) -> pd.DataFrame:
    """Build an EBM-like per-feature shape table from singleton HUG coefficients.

    The y-value is the downstream LR coefficient for a singleton pattern bin,
    i.e. a log-odds contribution. Compound patterns are excluded because their
    coefficient cannot be assigned to one feature unambiguously. Bins that exist
    in ``_bin_edges_`` but have no mined singleton pattern are retained with a
    zero coefficient.
    """
    if model is None or not callable(getattr(model, "feature_importances", None)):
        return pd.DataFrame()
    try:
        fi = _safe_dataframe(model.feature_importances())
    except Exception:
        return pd.DataFrame()
    if fi.empty:
        return pd.DataFrame()

    label_col = _first_existing_column(fi, ["display_name", "pattern", "feature"])
    coef_col = _first_existing_column(fi, ["coefficient", "coef", "log_odds", "importance"])
    support_col = _first_existing_column(fi, ["pattern_support", "support"])
    type_col = _first_existing_column(fi, ["feature_type", "type"])
    if not label_col or not coef_col:
        return pd.DataFrame()

    rows_by_lower: dict[float, dict[str, Any]] = {}
    fallback_rows: list[dict[str, Any]] = []
    filtered = fi.copy()
    if type_col:
        filtered = filtered[filtered[type_col].astype(str).str.lower().eq("pattern")]
    for _, row in filtered.iterrows():
        label = str(row.get(label_col, ""))
        parts = _split_pattern_assignments(label)
        if len(parts) != 1 or not parts[0].startswith(f"{feature}="):
            continue
        bin_label = parts[0].split("=", 1)[1].strip()
        try:
            coef = float(row.get(coef_col, 0.0))
        except Exception:
            coef = 0.0
        try:
            support = float(row.get(support_col, np.nan)) if support_col else np.nan
        except Exception:
            support = np.nan
        lower = _parse_interval_lower(parts[0])
        item = {
            "feature": feature,
            "bin": bin_label,
            "log_odds_contribution": coef,
            "coefficient": coef,
            "support": support,
            "source": "singleton_pattern",
            "pattern": label,
            "is_mined_singleton": True,
        }
        if lower is None:
            fallback_rows.append(item)
        else:
            rows_by_lower[float(lower)] = item

    edges = _bin_edges_for_feature(model, feature)
    if edges is not None and len(edges) >= 2:
        out: list[dict[str, Any]] = []
        lower_keys = list(rows_by_lower.keys())
        for i in range(len(edges) - 1):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            is_last = i == len(edges) - 2
            match_key = None
            for key in lower_keys:
                if np.isfinite(key) and abs(key - lo) <= max(1e-9, 0.02 * max(abs(lo), 1.0)):
                    match_key = key
                    break
            if match_key is not None:
                item = dict(rows_by_lower[match_key])
                if pd.isna(item.get("support", np.nan)):
                    item["support"] = _support_from_X(X, feature, lo, hi, is_last)
                item["lower"] = lo
                item["upper"] = hi
                out.append(item)
            else:
                out.append({
                    "feature": feature,
                    "bin": _bin_label_from_edges(lo, hi, is_last=is_last),
                    "log_odds_contribution": 0.0,
                    "coefficient": 0.0,
                    "support": _support_from_X(X, feature, lo, hi, is_last),
                    "source": "no_singleton_pattern",
                    "pattern": "",
                    "is_mined_singleton": False,
                    "lower": lo,
                    "upper": hi,
                })
        return pd.DataFrame(out)

    rows = list(rows_by_lower.values()) + fallback_rows
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if rows_by_lower:
        df["_lower_sort"] = df["bin"].map(lambda v: _parse_interval_lower(str(v)))
        df = df.sort_values("_lower_sort", na_position="last").drop(columns=["_lower_sort"])
    return df.reset_index(drop=True)


def plot_feature_shape(model: Any, X: Any, feature: str) -> Any | None:
    """Plot an EBM-like HUGIML feature shape using singleton log-odds coefficients."""
    df = feature_shape_frame(model, feature, X)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(7.0, min(14.0, len(df) * 0.9)), 4.2))
    x = np.arange(len(df))
    vals = pd.to_numeric(df["log_odds_contribution"], errors="coerce").fillna(0.0).to_numpy()
    ax.bar(x, vals)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["bin"].astype(str).tolist(), rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Log-odds contribution")
    ax.set_xlabel(str(feature))
    ax.set_title(f"Feature shape · {feature}")
    support = pd.to_numeric(df.get("support", pd.Series(dtype=float)), errors="coerce")
    if support.notna().any():
        ax2 = ax.twinx()
        support_vals = support.to_numpy(dtype=float)
        if np.nanmax(support_vals) <= 1.0:
            support_vals = support_vals * 100.0
        ax2.plot(x, support_vals, marker="o", linestyle="--", linewidth=1.2)
        ax2.set_ylabel("Support %")
    fig.tight_layout()
    return fig


# Backward-compatible public alias for earlier dashboard/tests/downstream code.
def plot_feature_bin_profile(model: Any, X: Any, feature: str) -> Any | None:
    return plot_feature_shape(model, X, feature)


def render_feature_effect_profiles(model: Any = None, X: Any = None, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Render EBM-style HUGIML 1-D shape profiles in Representation Audit.

    This uses singleton pattern coefficients from ``feature_importances()``. A
    singleton pattern such as ``income=[38500, 55000)`` is a binary item; its
    downstream logistic-regression coefficient is the marginal log-odds
    contribution of being in that bin, which is the HUGIML analogue of an EBM
    shape value. Compound patterns are excluded by design.
    """
    st.markdown("### Feature Effect Profiles")
    st.caption(
        "EBM-style one-dimensional HUGIML shape profiles: x = feature bin/range, "
        "y = singleton-pattern log-odds contribution. Compound patterns are excluded."
    )

    if model is None:
        st.info("No fitted HUGIML model is available for feature effect profiles.")
        return pd.DataFrame()
    if not callable(getattr(model, "feature_importances", None)):
        st.info("feature_importances() is not available on this model, so log-odds feature profiles cannot be computed.")
        return pd.DataFrame()

    features = _available_profile_features(model)
    if not features:
        st.info("No singleton HUG pattern labels are available for feature effect profiles.")
        return pd.DataFrame()

    feature = st.selectbox(
        "Feature",
        features,
        index=0,
        key="feature_effect_profile_feature",
        help="Select a feature to view singleton-bin log-odds contributions, analogous to an EBM 1-D shape function.",
    )
    df = feature_shape_frame(model, str(feature), X)
    if df.empty:
        st.info(
            f"No singleton pattern coefficients are available for '{feature}'. "
            "Try reducing G, increasing topK, or using a fitted HUGIML model with feature_importances()."
        )
        return df
    fig = plot_feature_shape(model, X, str(feature))
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)
    st.dataframe(dataframe_for_display(df), width="stretch", hide_index=True)
    return df

def render_adaptive_binning_evidence(model: Any = None, X: Any = None, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Render bin-profile/IG evidence for adaptive binning decisions."""
    st.markdown("### Adaptive Binning Evidence")
    st.caption(
        "Shows the selected per-feature bin count and information-gain evidence used to make "
        "adaptive discretisation auditable."
    )

    if model is None:
        st.info("No fitted model is available for adaptive-binning evidence.")
        return pd.DataFrame()

    if not _is_enabled_flag(getattr(model, "adaptive_binning", False)):
        st.info("Adaptive binning is off for this model, so no per-feature B evidence is available.")
        return pd.DataFrame()

    fig_tab, table_tab = st.tabs(["Figures", "IG / selected-B table"])
    with fig_tab:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Chosen B per feature")
            _render_model_figure(model, "plot_bin_profiles", "Chosen B per feature")
        with c2:
            st.markdown("#### IG heatmap")
            _render_model_figure(model, "ig_heatmap", "IG heatmap")

    df = adaptive_binning_table(model)
    with table_tab:
        if df.empty:
            st.info("per_feature_b_ / ig_scores_ metadata is not available on this model.")
        else:
            c1, c2 = st.columns([0.7, 0.3])
            query = c1.text_input("Search adaptive-binning evidence", value="", key="adaptive_binning_evidence_search")
            sort_by_b = c2.toggle("Sort by B", value=True, key="adaptive_binning_evidence_sort_b")
            show_df = df.copy()
            if query:
                mask = show_df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
                show_df = show_df.loc[mask]
            sort_cols = ["chosen_B", "feature", "B_candidate"] if sort_by_b else ["feature", "B_candidate"]
            show_df = show_df.sort_values([c for c in sort_cols if c in show_df.columns], kind="mergesort")
            st.dataframe(dataframe_for_display(show_df), width="stretch", hide_index=True)
    return df


def survivor_led_patterns_frame(model: Any) -> pd.DataFrame:
    """Return mined-pattern audit rows admitted through relaxed interaction evidence."""
    if model is None:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    method = getattr(model, "get_pattern_info", None)
    if callable(method):
        try:
            frames.append(_safe_dataframe(method()))
        except Exception:
            pass
    if not frames or all(frame.empty for frame in frames):
        method = getattr(model, "feature_importances", None)
        if callable(method):
            try:
                fi = _safe_dataframe(method())
                if not fi.empty:
                    type_col = _first_existing_column(fi, ["feature_type", "type"])
                    if type_col:
                        fi = fi[fi[type_col].astype(str).str.lower().eq("pattern")]
                    frames.append(fi)
            except Exception:
                pass
    for df in frames:
        if df.empty or "survivor_led" not in df.columns:
            continue
        mask = df["survivor_led"].map(_is_enabled_flag)
        if not mask.any():
            continue
        out = df.loc[mask].copy()
        preferred = [
            "pattern",
            "display_name",
            "feature",
            "pattern_origin",
            "survivor_led",
            "survivor_features",
            "survivor_feature_count",
            "survivor_min_marginal_ig",
            "survivor_max_interaction_score",
            "survivor_best_partners",
            "information_gain",
            "support",
            "coefficient",
            "abs_coefficient",
        ]
        columns = [c for c in preferred if c in out.columns] + [
            c for c in out.columns if c not in preferred
        ]
        return out[columns].reset_index(drop=True)
    return pd.DataFrame()


def render_survivor_led_pattern_audit(model: Any = None, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Render audit evidence for patterns surfaced by interaction-relaxed mining."""
    st.markdown("### Survivor-Led Pattern Audit")
    st.caption(
        "Shows ordinary HUG pattern rows whose source features were admitted through "
        "interaction-relaxed mining despite weak standalone marginal evidence."
    )

    if model is None:
        st.info("No fitted model is available for survivor-led pattern audit.")
        return pd.DataFrame()
    df = survivor_led_patterns_frame(model)
    if df.empty:
        st.info("No survivor-led pattern rows are recorded for this model.")
        return df

    query = st.text_input(
        "Search survivor-led patterns",
        value="",
        key="survivor_led_pattern_search",
    )
    show_df = df.copy()
    if query:
        mask = show_df.astype(str).apply(
            lambda col: col.str.contains(query, case=False, na=False)
        ).any(axis=1)
        show_df = show_df.loc[mask]
    st.metric("Survivor-led patterns", f"{len(show_df):,}")
    st.dataframe(dataframe_for_display(show_df), width="stretch", hide_index=True)
    return df


def rpte_rule_evidence_frame(model: Any) -> pd.DataFrame:
    """Return the complete fitted RPTE final-LR representation.

    The frame contains RPTE leaf indicators and every direct HUGIML source term
    source column carried directly into the final LR. Zero-valued direct
    coefficients are retained so the audit remains aligned with the fitted
    coefficient vector.
    """
    rows = _get_rpte_final_term_rows(model, include_zero_direct=True)
    return _rpte_rules_to_frame(rows) if rows else pd.DataFrame()


def render_rpte_rule_evidence(model: Any = None, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Render leaf-tree and direct-source evidence separately."""
    st.markdown("### RPTE Final LR Evidence")
    st.caption(
        "The final logistic regression contains RPTE leaf indicators followed by HUGIML "
        "source columns not used in accepted tree splits. Direct source terms are grouped "
        "as original features, HUG patterns, and augmented pairs."
    )

    if model is None:
        st.info("No fitted model is available for RPTE evidence.")
        return pd.DataFrame()

    rows = _get_rpte_final_term_rows(model, include_zero_direct=True)
    df = _rpte_rules_to_frame(rows) if rows else pd.DataFrame()
    if df.empty:
        st.info("No RPTE final-LR evidence is recorded for this fitted model.")
        return df

    flow = _get_rpte_feature_flow_audit(model)
    leaf_df = df.loc[df.get("is_leaf_term", False)].copy() if "is_leaf_term" in df else pd.DataFrame()
    direct_flag = df.get("is_direct_source_term", False)
    direct_df = df.loc[direct_flag].copy() if not df.empty else pd.DataFrame()

    metrics = st.columns(5)
    metrics[0].metric("Final LR terms", f"{int(flow.get('final_term_count', len(df)) or len(df)):,}")
    metrics[1].metric("Leaf terms", f"{len(leaf_df):,}")
    family_counts = direct_df.get("source_family", pd.Series(dtype=str)).value_counts()
    metrics[2].metric("Direct original features", f"{int(family_counts.get('original', 0)):,}")
    metrics[3].metric("Direct HUG patterns", f"{int(family_counts.get('pattern', 0)):,}")
    metrics[4].metric("Direct augmented pairs", f"{int(family_counts.get('augmented_pair', 0)):,}")

    if not leaf_df.empty:
        st.markdown("#### RPTE leaf trees")
        query = st.text_input("Search RPTE leaf rules", value="", key="rpte_rule_evidence_search")
        show_leaf = leaf_df.copy()
        if query:
            mask = show_leaf.astype(str).apply(
                lambda col: col.str.contains(query, case=False, na=False)
            ).any(axis=1)
            show_leaf = show_leaf.loc[mask]
        render_rpte_flat_tree_view(rows, show_leaf, key_prefix="governance_rpte_flat_tree")
        leaf_columns = [
            "class", "tree", "leaf", "effect", "coefficient", "odds_multiplier",
            "support_rate", "support_count", "n_conditions", "raw_sources", "backend",
        ]
        st.markdown("##### Leaf coefficient table")
        st.dataframe(
            dataframe_for_display(show_leaf[[c for c in leaf_columns if c in show_leaf.columns]]),
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Direct source terms carried directly into LR")
    st.caption(
        "Only fitted source columns identified as direct because they were not selected by accepted RPTE splits appear below. "
        "Columns used in tree splits remain represented through leaf indicators and are not repeated."
    )
    tabs = st.tabs(["Direct original features", "Direct HUG patterns", "Direct augmented pairs"])
    families = ["original", "pattern", "augmented_pair"]
    for tab, family in zip(tabs, families):
        with tab:
            family_df = direct_df.loc[direct_df.get("source_family", pd.Series(index=direct_df.index, dtype=str)).eq(family)].copy()
            if family_df.empty:
                st.info("No direct terms from this source family enter the final LR.")
                continue
            display_cols = [
                "class", "source_display_name", "source_column", "raw_sources",
                "effect", "coefficient", "odds_multiplier", "backend",
            ]
            st.dataframe(
                dataframe_for_display(family_df[[c for c in display_cols if c in family_df.columns]]),
                width="stretch",
                hide_index=True,
            )
    return df


def augmented_pair_effects_frame(model: Any) -> pd.DataFrame:
    method = getattr(model, "explain_augmented_pair_effects", None)
    if not callable(method):
        return pd.DataFrame()
    try:
        df = _safe_dataframe(method())
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if "coefficient_raw_scale" in df.columns:
        coef = pd.to_numeric(df["coefficient_raw_scale"], errors="coerce").abs()
        if coef.notna().any():
            threshold = float(coef.quantile(0.90))
            df["large_raw_effect"] = coef >= threshold
    return df


def render_augmented_pair_traceability(model: Any = None, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Render audit traceability for augmented pair features."""
    st.markdown("### Augmented Pair Effect Traceability")
    st.caption(
        "Surfaces raw formulas, raw-scale coefficients, missingness/eligibility rates, "
        "and pre-built interpretation narratives for augmented pair features."
    )

    if model is None:
        st.info("No fitted model is available for augmented pair traceability.")
        return pd.DataFrame()

    df = augmented_pair_effects_frame(model)
    if df.empty:
        st.info("Augmented pair effect explanations are not available for this model/version.")
        return df

    if "transform_ig" in df.columns:
        _ig_df = df[["feature", "transform_ig"]].copy()
        _ig_df["transform_ig"] = pd.to_numeric(_ig_df["transform_ig"], errors="coerce")
        _ig_df = _ig_df.dropna(subset=["transform_ig"]).sort_values("transform_ig", ascending=True)
        if not _ig_df.empty:
            _fig2, _ax2 = plt.subplots(figsize=(7, max(2.0, len(_ig_df) * 0.38)))
            _ax2.barh(_ig_df["feature"].astype(str), _ig_df["transform_ig"], color="#1D9E75")
            _ax2.set_xlabel("Transform IG (information gain)")
            _ax2.set_title("Augmented pair selection — IG per pair")
            _ax2.invert_yaxis()
            _fig2.tight_layout()
            st.pyplot(_fig2)
            plt.close(_fig2)

    query = st.text_input("Search augmented pair effects", value="", key="augmented_pair_effect_search")
    show_df = df.copy()
    if query:
        mask = show_df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        show_df = show_df.loc[mask]

    preferred = [
        "feature",
        "raw_formula",
        "formula",
        "standardization_mean",
        "standardization_scale",
        "coefficient_raw_scale",
        "risk_increases_when",
        "eligible_rate",
        "missing_pair_rate",
        "large_raw_effect",
        "raw_interpretation",
    ]
    ordered = [c for c in preferred if c in show_df.columns] + [c for c in show_df.columns if c not in preferred]
    st.dataframe(dataframe_for_display(show_df[ordered]), width="stretch", hide_index=True)

    text_col = next((c for c in ("raw_interpretation", "interpretation", "plain_english") if c in df.columns), None)
    if text_col:
        with st.expander("Plain-English augmented pair interpretations", expanded=False):
            for i, row in df.head(50).iterrows():
                label = row.get("feature", row.get("raw_formula", row.get("formula", f"Augmented pair {i + 1}")))
                st.markdown(f"**{label}**")
                st.write(str(row.get(text_col, "")))
    return df

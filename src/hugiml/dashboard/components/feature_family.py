"""Feature-family audit component."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _as_list(value: Any) -> list:
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


def original_feature_audit(
    model: Any,
    X: pd.DataFrame | None = None,
    sensitive_columns: list[str] | None = None,
    excluded_columns: list[str] | None = None,
    id_column: str | None = None,
) -> pd.DataFrame:
    sensitive = set(map(str, _as_list(sensitive_columns)))
    excluded = set(map(str, _as_list(excluded_columns)))
    id_col = str(id_column) if id_column else None
    names = _as_list(getattr(model, "feature_names_in_", None)) or _as_list(getattr(model, "origColumns", None))
    if not names and X is not None:
        names = list(X.columns)

    rows = []
    for name in map(str, names):
        used = name not in excluded and name != id_col
        missing_pct = np.nan
        dtype = ""
        if X is not None and name in X.columns:
            missing_pct = float(pd.isna(X[name]).mean() * 100.0)
            dtype = str(X[name].dtype)
        role = "ID" if name == id_col else ("Sensitive/review" if name in sensitive else ("Excluded" if name in excluded else "Model feature"))
        risk = "Review" if name in sensitive and used else ("Excluded" if not used else "OK")
        rows.append({
            "feature": name,
            "family": "Original",
            "dtype": dtype,
            "used_in_model": bool(used),
            "role": role,
            "missing_pct": missing_pct,
            "status": risk,
        })
    return pd.DataFrame(rows)


def pattern_feature_audit(model: Any, sensitive_columns: list[str] | None = None) -> pd.DataFrame:
    sensitive = set(map(str, _as_list(sensitive_columns)))
    rows = []

    try:
        if hasattr(model, "get_pattern_info"):
            info = model.get_pattern_info()
            if isinstance(info, pd.DataFrame) and not info.empty:
                df = info.copy()
                label_col = next((c for c in df.columns if str(c).lower() in {"pattern", "label", "feature", "name"}), None)
                if label_col:
                    for i, row in df.iterrows():
                        label = str(row[label_col])
                        src = _infer_source_features_from_label(label)
                        rows.append({
                            "rank": int(i) + 1,
                            "pattern": label,
                            "source_features": ", ".join(src),
                            "order": len(src) if src else np.nan,
                            "status": "Review" if set(src).intersection(sensitive) else "OK",
                        })
                    return pd.DataFrame(rows)
    except Exception:
        pass

    labels = []
    for attr in ("pattern_labels_", "_pattern_labels_", "patterns_", "raw_patterns_"):
        try:
            labels = getattr(model, attr, None)
            if labels is not None:
                break
        except Exception:
            labels = None

    for i, label in enumerate(_as_list(labels), start=1):
        label = str(label)
        src = _infer_source_features_from_label(label)
        rows.append({
            "rank": i,
            "pattern": label,
            "source_features": ", ".join(src),
            "order": len(src) if src else np.nan,
            "status": "Review" if set(src).intersection(sensitive) else "OK",
        })
    return pd.DataFrame(rows)


def augmented_feature_audit(model: Any, sensitive_columns: list[str] | None = None) -> pd.DataFrame:
    sensitive = set(map(str, _as_list(sensitive_columns)))
    specs = _as_list(getattr(model, "augmented_pair_transforms_", []))
    rows = []
    for i, spec in enumerate(specs, start=1):
        if not isinstance(spec, dict):
            continue
        inputs = [str(x) for x in spec.get("inputs", [])]
        rows.append({
            "rank": i,
            "name": str(spec.get("name", f"augmented_{i}")),
            "operation": str(spec.get("operation", "")),
            "source_features": ", ".join(inputs),
            "formula": str(spec.get("formula", spec.get("raw_formula", ""))),
            "missing_policy": str(spec.get("pair_missing_policy", "")),
            "eligible_rate": spec.get("eligible_rate", np.nan),
            "missing_pair_rate": spec.get("missing_pair_rate", np.nan),
            "transform_ig": spec.get("transform_ig", np.nan),
            "status": "Review" if set(inputs).intersection(sensitive) else "OK",
        })
    return pd.DataFrame(rows)


def feature_family_summary(model: Any, X: pd.DataFrame | None = None) -> pd.DataFrame:
    meta = getattr(model, "fit_metadata_", None)
    counts = getattr(meta, "downstream_feature_counts", {}) if meta is not None else {}
    if not isinstance(counts, dict):
        counts = {}

    p = len(_as_list(getattr(model, "feature_names_in_", None))) or (X.shape[1] if X is not None else 0)
    n_patterns = len(_as_list(getattr(model, "patterns_", []))) or len(_as_list(getattr(model, "raw_patterns_", [])))
    n_aug = len(_as_list(getattr(model, "augmented_pair_transforms_", [])))

    rows = [
        {"feature_family": "Original features", "count": int(counts.get("original", p)), "evidence": "Raw input columns after exclusions"},
        {"feature_family": "HUG pattern features", "count": int(counts.get("pattern", n_patterns)), "evidence": "Human-readable mined patterns"},
        {"feature_family": "Augmented/generated features", "count": int(counts.get("augmented_pair", n_aug)), "evidence": "Generated transforms with source-feature provenance"},
    ]
    total = sum(int(r["count"]) for r in rows)
    rows.append({"feature_family": "Displayed total", "count": total, "evidence": "Sum of displayed family counts"})
    return pd.DataFrame(rows)


def render_feature_family_audit(
    model: Any,
    X: pd.DataFrame | None = None,
    sensitive_columns: list[str] | None = None,
    excluded_columns: list[str] | None = None,
    id_column: str | None = None,
) -> dict[str, pd.DataFrame]:
    st.markdown("### Feature Family Audit")
    st.caption("Audits the downstream representation using explicit feature-family evidence, not opaque scores.")

    summary = feature_family_summary(model, X)
    originals = original_feature_audit(model, X, sensitive_columns, excluded_columns, id_column)
    patterns = pattern_feature_audit(model, sensitive_columns)
    augmented = augmented_feature_audit(model, sensitive_columns)

    cols = st.columns(4)
    def _count(name: str) -> int:
        s = summary.loc[summary["feature_family"].eq(name), "count"]
        return int(s.iloc[0]) if len(s) else 0

    cols[0].metric("Original features", _count("Original features"))
    cols[1].metric("HUG patterns", _count("HUG pattern features"))
    cols[2].metric("Augmented features", _count("Augmented/generated features"))
    review_flags = int((originals.get("status", pd.Series(dtype=str)).eq("Review")).sum()) + int((patterns.get("status", pd.Series(dtype=str)).eq("Review")).sum()) + int((augmented.get("status", pd.Series(dtype=str)).eq("Review")).sum())
    cols[3].metric("Review flags", review_flags)

    st.dataframe(summary, width="stretch", hide_index=True)

    # Signed coefficient waterfall from feature_importances() when available
    try:
        _fi = model.feature_importances() if hasattr(model, "feature_importances") and callable(getattr(model, "feature_importances")) else None
    except Exception:
        _fi = None
    if _fi is not None and not _fi.empty and "coefficient" in _fi.columns:
        _wf = _fi.copy()
        _wf["coefficient"] = pd.to_numeric(_wf["coefficient"], errors="coerce")
        _wf = _wf.dropna(subset=["coefficient"]).head(25).sort_values("coefficient")
        if not _wf.empty:
            st.markdown("#### Feature coefficient waterfall (signed)")
            st.caption(
                "Positive coefficients (right) increase the model score; negative (left) decrease it. "
                "Color encodes the feature family: original, HUG pattern, or augmented/generated."
            )
            _family_colors = {
                "original": "#378ADD",
                "pattern": "#534AB7",
                "augmented_pair": "#1D9E75",
                "unknown": "#888780",
            }
            _family_labels = {
                "original": "Original",
                "pattern": "HUG pattern",
                "augmented_pair": "Augmented/generated",
                "unknown": "Unknown",
            }

            def _normalise_feature_family(value: Any, label: str = "") -> str:
                raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
                if raw in {"original", "original_feature", "original_features", "raw", "input", "input_feature"}:
                    return "original"
                if raw in {"pattern", "patterns", "hug_pattern", "hug_patterns", "pattern_feature", "pattern_features"}:
                    return "pattern"
                if raw in {
                    "augmented",
                    "augmented_pair",
                    "augmented_pair_feature",
                    "augmented_pair_features",
                    "augmented_generated",
                    "generated",
                    "generated_feature",
                    "interaction",
                    "interaction_feature",
                }:
                    return "augmented_pair"
                text = str(label or value or "").lower()
                if "augmented_pair" in text or text.startswith("augmented") or "*" in text or " abs(" in text or text.startswith("abs("):
                    return "augmented_pair"
                if "[" in text and "]" in text or "=" in text:
                    return "pattern"
                return "unknown"

            _type_col = "feature_type" if "feature_type" in _wf.columns else None
            _name_col = "display_name" if "display_name" in _wf.columns else "pattern" if "pattern" in _wf.columns else "feature"
            _wf["_feature_family"] = [
                _normalise_feature_family(_wf[_type_col].iloc[i] if _type_col else "", str(_wf[_name_col].iloc[i]))
                for i in range(len(_wf))
            ]
            _colors_wf = [_family_colors.get(str(v), _family_colors["unknown"]) for v in _wf["_feature_family"]]

            _fig_wf, _ax_wf = plt.subplots(figsize=(7, max(3.0, len(_wf) * 0.32)))
            _ax_wf.barh(_wf[_name_col].astype(str), _wf["coefficient"], color=_colors_wf)
            _ax_wf.axvline(0, color="#888780", linewidth=0.8)
            _ax_wf.set_xlabel("Coefficient (log-odds)")
            _ax_wf.set_title("Signed feature coefficients")
            _ax_wf.invert_yaxis()
            _present_families = [fam for fam in ("original", "pattern", "augmented_pair", "unknown") if fam in set(_wf["_feature_family"])]
            if _present_families:
                _handles = [
                    Patch(facecolor=_family_colors[fam], edgecolor="none", label=_family_labels[fam])
                    for fam in _present_families
                ]
                _ax_wf.legend(handles=_handles, fontsize=8, loc="lower right", title="Feature family")
            _fig_wf.tight_layout()
            st.pyplot(_fig_wf)
            plt.close(_fig_wf)

    tab1, tab2, tab3 = st.tabs(["Originals", "HUG patterns", "Augmented/generated"])
    with tab1:
        st.dataframe(originals, width="stretch", hide_index=True)
    with tab2:
        if patterns.empty:
            st.info("Pattern metadata is not available for this model/version.")
        else:
            st.dataframe(patterns, width="stretch", hide_index=True)
    with tab3:
        if augmented.empty:
            st.info("No augmented/generated feature metadata was found for this model.")
        else:
            st.dataframe(augmented, width="stretch", hide_index=True)

    return {"summary": summary, "originals": originals, "patterns": patterns, "augmented": augmented}

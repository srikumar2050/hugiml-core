"""Executive overview component."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from hugiml.dashboard.display import dataframe_for_display
from hugiml.serialization import generate_sbom


def _metric_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _meta_get(meta: Any, name: str, default: Any = None) -> Any:
    if meta is None:
        return default
    if isinstance(meta, dict):
        return meta.get(name, default)
    return getattr(meta, name, default)


def _has_evidence_rows(value: Any) -> bool:
    """Return whether a result artifact contains at least one row/item.

    Pandas objects intentionally reject scalar truth testing because ``bool(df)``
    is ambiguous. Promotion preserves the original tuning result, whose
    ``results_`` can be a DataFrame, list of rows, NumPy array, mapping, or a
    custom table-like object. Inspect shape/length explicitly instead of ever
    evaluating the artifact itself in a boolean context.
    """
    if value is None:
        return False
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return not value.empty

    empty = getattr(value, "empty", None)
    if isinstance(empty, bool):
        return not empty

    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and shape:
        try:
            return int(shape[0]) > 0
        except (TypeError, ValueError, OverflowError):
            pass

    try:
        return len(value) > 0
    except (TypeError, AttributeError):
        # A non-container scalar/custom evidence object is still an available
        # artifact; the renderer can decide how much detail it exposes.
        return True


def fit_metadata_frame(model: Any) -> pd.DataFrame:
    """Flatten model.fit_metadata_ into dashboard rows."""
    fit_meta = getattr(model, "fit_metadata_", None)
    if fit_meta is None:
        return pd.DataFrame()
    fields = [
        "n_patterns",
        "total_fit_ms",
        "stage_times_ms",
        "matrix_density",
        "degraded",
        "execution_mode",
        "memory_usage_mb",
        "peak_memory_mb",
        "downstream_feature_counts",
        "n_original_features",
        "n_pattern_features",
        "n_augmented_features",
        "fit_status",
    ]
    rows = []
    for field in fields:
        value = _meta_get(fit_meta, field, None)
        if value is not None:
            rows.append({"field": field, "value": value})
    if not rows and hasattr(fit_meta, "__dict__"):
        rows = [{"field": k, "value": v} for k, v in vars(fit_meta).items() if not k.startswith("_")]
    return pd.DataFrame(rows)


def evidence_status_frame(model: Any = None, result: Any = None, X: Any = None, y: Any = None) -> pd.DataFrame:
    """Return evidence status rows derived from actual loaded artifacts."""
    def status(ok: bool) -> str:
        return "Available" if ok else "Missing"

    rows = [
        {
            "area": "Validation evidence",
            "status": status(result is not None and _has_evidence_rows(getattr(result, "results_", None))),
            "evidence": "CV/tuning result table and label-aware diagnostics when y is loaded",
        },
        {
            "area": "Representation audit",
            "status": status(model is not None and (hasattr(model, "feature_names_in_") or hasattr(model, "patterns_") or hasattr(model, "augmented_pair_transforms_"))),
            "evidence": "Original, pattern, augmented, adaptive-binning, and pair-effect artifacts",
        },
        {
            "area": "Pattern inventory",
            "status": status(model is not None and (hasattr(model, "get_pattern_info") or hasattr(model, "patterns_") or hasattr(model, "raw_patterns_"))),
            "evidence": "Human-readable HUG pattern table and population-coverage audit",
        },
        {
            "area": "Case review",
            "status": status(model is not None and X is not None and hasattr(model, "predict_proba")),
            "evidence": "Prediction and case input traceability",
        },
        {
            "area": "Data quality",
            "status": status(X is not None),
            "evidence": "Feature missingness, model missing-value edges, binary categorical routing",
        },
        {
            "area": "Monitoring",
            "status": status(model is not None and X is not None),
            "evidence": "Prediction distribution, drift review, CV monitoring when y and API are available",
        },
    ]
    return pd.DataFrame(rows)


def render_overview(model: Any = None, result: Any = None, roles: dict | None = None, meta: dict | None = None, X: Any = None, y: Any = None) -> None:
    roles = roles or {}
    meta = meta or {}

    st.subheader("Executive Overview")
    st.markdown(
        """
        <div class="hugiml-section-note">
          <p>Concise governance summary of model configuration, column roles, fit metadata, and evidence available for review.
          No opaque interpretability score is used; auditability is represented through explicit artifacts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_score = getattr(result, "best_score_", None) if result is not None else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows reviewed", _metric_value(meta.get("n_rows")))
    c2.metric("Model features", _metric_value(meta.get("n_features")))
    c3.metric("Best CV ROC-AUC", _metric_value(best_score))
    positive_rate = meta.get("positive_rate")
    c4.metric("Positive rate", f"{positive_rate:.2%}" if isinstance(positive_rate, (int, float)) else "N/A")

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("#### Evidence status")
        st.dataframe(dataframe_for_display(evidence_status_frame(model, result, X, y)), width="stretch", hide_index=True)

        fit_df = fit_metadata_frame(model)
        with st.expander("Fit metadata", expanded=not fit_df.empty):
            if fit_df.empty:
                st.info("fit_metadata_ is not available on this model.")
            else:
                fit_meta = getattr(model, "fit_metadata_", None)
                stage_times = _meta_get(fit_meta, "stage_times_ms", None)
                if isinstance(stage_times, dict) and stage_times:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as _plt
                    _stages = list(stage_times.keys())
                    _times = [float(stage_times[s]) for s in _stages]
                    _total = sum(_times) or 1.0
                    _fig, _ax = _plt.subplots(figsize=(7, max(2.0, len(_stages) * 0.38)))
                    _colors = ["#534AB7" if t == max(_times) else "#AFA9EC" for t in _times]
                    _ax.barh(_stages, _times, color=_colors)
                    for i, (s, t) in enumerate(zip(_stages, _times)):
                        _ax.text(t + _total * 0.01, i, f"{t:.0f} ms ({t / _total:.0%})", va="center", fontsize=8)
                    _ax.set_xlabel("Wall-clock time (ms)")
                    _ax.set_title("Fit stage timing")
                    _ax.invert_yaxis()
                    _fig.tight_layout()
                    st.pyplot(_fig)
                    _plt.close(_fig)
                st.dataframe(dataframe_for_display(fit_df), width="stretch", hide_index=True)

        with st.expander("Software Bill of Materials (SBOM)", expanded=False):
            try:
                sbom = generate_sbom()
                st.json(sbom)
                st.download_button(
                    "Download SBOM JSON",
                    data=json.dumps(sbom, indent=2),
                    file_name="hugiml_sbom.json",
                    mime="application/json",
                )
            except Exception as exc:
                st.info(f"SBOM could not be generated in this environment: {exc}")

    with right:
        st.markdown("#### Column roles")
        role_rows = [
            {"role": "Target", "columns": roles.get("target") or "N/A"},
            {"role": "ID / case lookup", "columns": roles.get("id_column") or "Generated case IDs"},
            {"role": "Excluded from modeling", "columns": ", ".join(roles.get("excluded_columns", [])) or "None"},
            {"role": "Sensitive/proxy review", "columns": ", ".join(roles.get("sensitive_columns", [])) or "None"},
        ]
        st.dataframe(dataframe_for_display(pd.DataFrame(role_rows)), width="stretch", hide_index=True)

        st.markdown("#### Selected HUGIML configuration")
        config_rows = [
            {"parameter": "L", "value": getattr(model, "L", "N/A"), "meaning": "Maximum pattern order used by the selected model."},
            {"parameter": "topK", "value": getattr(model, "topK", "N/A"), "meaning": "Selected pattern budget."},
            {"parameter": "G", "value": getattr(model, "G", "N/A"), "meaning": "Mining threshold used by the selected model."},
            {"parameter": "feature_mode", "value": str(getattr(model, "feature_mode", "N/A")), "meaning": "Final representation regime. Shown as text to avoid truncation."},
        ]
        st.dataframe(
            dataframe_for_display(pd.DataFrame(config_rows)),
            width="stretch",
            hide_index=True,
            column_config={
                "parameter": st.column_config.TextColumn("Parameter", width="small"),
                "value": st.column_config.TextColumn("Value", width="medium"),
                "meaning": st.column_config.TextColumn("Meaning", width="large"),
            },
        )

"""Monitoring and distribution review component.

This component intentionally avoids calling model-level drift APIs blindly on the
same dataframe. Some APIs require a training reference and a current production
batch; calling them with only one dataset can produce misleading "all features
drifted" output. The dashboard therefore shows transparent, data-derived
reference/current distribution evidence and keeps optional model API output out
of the primary path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from hugiml.dashboard.display import dataframe_for_display


def _prediction_summary(model: Any, X: pd.DataFrame | None) -> tuple[pd.DataFrame, np.ndarray | None]:
    if model is None or X is None:
        return pd.DataFrame(), None
    try:
        proba = model.predict_proba(X)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] > 1:
            p = np.asarray(proba[:, 1], dtype=float)
        else:
            p = np.asarray(proba).ravel().astype(float)
    except Exception:
        return pd.DataFrame(), None

    s = pd.Series(p, name="predicted_probability")
    df = pd.DataFrame({
        "metric": [
            "n_cases",
            "mean_probability",
            "median_probability",
            "std_probability",
            "min_probability",
            "max_probability",
            "share_ge_0_50",
            "share_ge_0_80",
        ],
        "value": [
            int(len(s)),
            float(s.mean()),
            float(s.median()),
            float(s.std(ddof=0)),
            float(s.min()),
            float(s.max()),
            float((s >= 0.50).mean()),
            float((s >= 0.80).mean()),
        ],
    })
    return df, p


def _probability_band_table(p: np.ndarray | None) -> pd.DataFrame:
    if p is None or len(p) == 0:
        return pd.DataFrame()
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
    labels = ["[0.0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0]"]
    bands = pd.cut(p, bins=bins, labels=labels, include_lowest=True, right=False)
    vc = pd.Series(bands).value_counts(sort=False)
    return pd.DataFrame({
        "probability_band": vc.index.astype(str),
        "n_cases": vc.values.astype(int),
        "share": (vc.values / len(p)).round(6),
    })


def _missingness_summary(X: pd.DataFrame | None) -> pd.DataFrame:
    if X is None:
        return pd.DataFrame()
    rows = []
    for col in X.columns:
        miss = pd.isna(X[col])
        rows.append({
            "feature": str(col),
            "missing_pct": float(miss.mean() * 100.0),
            "n_missing": int(miss.sum()),
            "dtype": str(X[col].dtype),
        })
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


def _split_reference_current(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a transparent review split when no production baseline is supplied.

    This is not a substitute for real production drift monitoring. It is a
    within-dataset reference/current distribution review for dashboard inspection.
    """
    n = len(X)
    if n < 20:
        return X.copy(), X.iloc[0:0].copy()
    cut = max(1, int(n * 0.70))
    cut = min(cut, n - 1)
    return X.iloc[:cut].copy(), X.iloc[cut:].copy()


def _psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float | None:
    ref_num = pd.to_numeric(ref, errors="coerce").dropna()
    cur_num = pd.to_numeric(cur, errors="coerce").dropna()
    if len(ref_num) < 10 or len(cur_num) < 10:
        return None

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.nanquantile(ref_num, quantiles))
    if len(edges) < 3:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_bins = pd.cut(ref_num, edges, include_lowest=True).value_counts(sort=False)
    cur_bins = pd.cut(cur_num, edges, include_lowest=True).value_counts(sort=False)

    ref_pct = ref_bins / max(ref_bins.sum(), 1)
    cur_pct = cur_bins / max(cur_bins.sum(), 1)

    eps = 1e-6
    ref_pct = ref_pct.clip(lower=eps)
    cur_pct = cur_pct.clip(lower=eps)
    return float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())


def _ks_stat(ref: pd.Series, cur: pd.Series) -> float | None:
    ref_num = pd.to_numeric(ref, errors="coerce").dropna().to_numpy()
    cur_num = pd.to_numeric(cur, errors="coerce").dropna().to_numpy()
    if len(ref_num) < 10 or len(cur_num) < 10:
        return None
    ref_sorted = np.sort(ref_num)
    cur_sorted = np.sort(cur_num)
    values = np.sort(np.unique(np.concatenate([ref_sorted, cur_sorted])))
    if len(values) == 0:
        return None
    ref_cdf = np.searchsorted(ref_sorted, values, side="right") / len(ref_sorted)
    cur_cdf = np.searchsorted(cur_sorted, values, side="right") / len(cur_sorted)
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def _categorical_tvd(ref: pd.Series, cur: pd.Series) -> float | None:
    ref_s = ref.astype("string").fillna("<missing>")
    cur_s = cur.astype("string").fillna("<missing>")
    if len(ref_s) == 0 or len(cur_s) == 0:
        return None
    cats = sorted(set(ref_s.unique()).union(set(cur_s.unique())))
    if len(cats) == 0:
        return None
    ref_pct = ref_s.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    cur_pct = cur_s.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    return float(0.5 * np.abs(ref_pct - cur_pct).sum())


def _distribution_review(X: pd.DataFrame | None) -> pd.DataFrame:
    if X is None or X.empty:
        return pd.DataFrame()

    ref, cur = _split_reference_current(X)
    if cur.empty:
        return pd.DataFrame([{
            "feature": "<dataset>",
            "review_type": "not_enough_rows",
            "reference_n": len(ref),
            "current_n": len(cur),
            "metric": "",
            "value": "",
            "interpretation": "Need at least 20 rows to create a reference/current review split.",
        }])

    rows = []
    for col in X.columns:
        s_ref = ref[col]
        s_cur = cur[col]
        numeric = pd.api.types.is_numeric_dtype(X[col])

        if numeric:
            psi = _psi(s_ref, s_cur)
            ks = _ks_stat(s_ref, s_cur)
            rows.append({
                "feature": str(col),
                "review_type": "numeric_distribution",
                "reference_n": int(s_ref.notna().sum()),
                "current_n": int(s_cur.notna().sum()),
                "metric": "psi",
                "value": "" if psi is None else round(float(psi), 6),
                "secondary_metric": "ks",
                "secondary_value": "" if ks is None else round(float(ks), 6),
                "interpretation": "Distribution metric only; not a production drift decision without an external baseline.",
            })
        else:
            tvd = _categorical_tvd(s_ref, s_cur)
            rows.append({
                "feature": str(col),
                "review_type": "categorical_distribution",
                "reference_n": int(s_ref.notna().sum()),
                "current_n": int(s_cur.notna().sum()),
                "metric": "total_variation_distance",
                "value": "" if tvd is None else round(float(tvd), 6),
                "secondary_metric": "",
                "secondary_value": "",
                "interpretation": "Distribution metric only; not a production drift decision without an external baseline.",
            })

    return pd.DataFrame(rows)


def _pattern_activation_summary(model: Any, X: pd.DataFrame | None) -> pd.DataFrame:
    if model is None or X is None:
        return pd.DataFrame()

    transform_candidates = [
        "transform_patterns",
        "transform_pattern_features",
        "_transform_patterns",
        "_transform_pattern_features",
        "transform",
    ]

    for name in transform_candidates:
        fn = getattr(model, name, None)
        if fn is None:
            continue
        try:
            Z = fn(X)
            if hasattr(Z, "toarray"):
                Z = Z.toarray()
            if isinstance(Z, pd.DataFrame):
                arr = Z.to_numpy()
                labels = list(map(str, Z.columns))
            else:
                arr = np.asarray(Z)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                labels = [f"pattern_{i+1}" for i in range(arr.shape[1])]

            if arr.size == 0:
                return pd.DataFrame()

            activation_rate = np.asarray(arr.astype(bool).mean(axis=0)).ravel()
            rows = []
            for i, rate in enumerate(activation_rate):
                rows.append({
                    "rank": i + 1,
                    "pattern": labels[i] if i < len(labels) else f"pattern_{i+1}",
                    "activation_rate": float(rate),
                    "n_active": int(np.asarray(arr[:, i]).astype(bool).sum()),
                })
            return pd.DataFrame(rows).sort_values("activation_rate", ascending=False).head(25)
        except Exception:
            continue

    return pd.DataFrame()


def render_drift(model: Any = None, X: pd.DataFrame | None = None, y: Any = None, *args, **kwargs) -> None:
    st.subheader("Monitoring")
    st.markdown(
        """
        <div class="hugiml-section-note">
          <p>This page reports transparent monitoring evidence from the loaded data and fitted model.
          It does not call drift APIs without a proper external reference/current baseline, because that
          can incorrectly label many or all features as drifted.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pred_summary, p = _prediction_summary(model, X)
    bands = _probability_band_table(p)
    miss = _missingness_summary(X)
    dist = _distribution_review(X)
    pattern_summary = _pattern_activation_summary(model, X)

    c1, c2, c3, c4 = st.columns(4)
    if not pred_summary.empty:
        mean_prob = pred_summary.loc[pred_summary["metric"].eq("mean_probability"), "value"].iloc[0]
        high_share = pred_summary.loc[pred_summary["metric"].eq("share_ge_0_80"), "value"].iloc[0]
        c1.metric("Cases scored", f"{len(X):,}" if X is not None else "N/A")
        c2.metric("Mean probability", f"{mean_prob:.4f}")
        c3.metric("Share probability ≥ 0.80", f"{high_share:.2%}")
    else:
        c1.metric("Cases scored", f"{len(X):,}" if X is not None else "N/A")
        c2.metric("Mean probability", "N/A")
        c3.metric("Share probability ≥ 0.80", "N/A")

    if not miss.empty:
        c4.metric("Max missingness", f"{miss['missing_pct'].max():.2f}%")
    else:
        c4.metric("Max missingness", "N/A")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Prediction distribution",
        "Missingness",
        "Distribution review",
        "Pattern activation",
        "CV Monitoring",
    ])

    with tab1:
        st.markdown("#### Prediction summary")
        if pred_summary.empty:
            st.info("Prediction probabilities are not available for this model/data combination.")
        else:
            st.dataframe(dataframe_for_display(pred_summary), width="stretch", hide_index=True)
            st.markdown("#### Probability bands")
            st.dataframe(dataframe_for_display(bands), width="stretch", hide_index=True)

    with tab2:
        st.markdown("#### Feature-level missingness")
        if miss.empty:
            st.info("Missingness summary is not available.")
        else:
            st.dataframe(dataframe_for_display(miss), width="stretch", hide_index=True)

    with tab3:
        st.markdown("#### Reference/current distribution review")
        st.caption(
            "Reference = first 70% of loaded rows; current = remaining 30%. "
            "This is a dashboard review split only, not a production drift decision."
        )
        if dist.empty:
            st.info("Distribution review is not available.")
        else:
            numeric_dist = dist.loc[dist["review_type"].astype(str).str.contains("numeric", case=False, na=False)].copy()
            if not numeric_dist.empty and "value" in numeric_dist.columns:
                psi_df = numeric_dist[["feature", "value"]].copy()
                psi_df["value"] = pd.to_numeric(psi_df["value"], errors="coerce")
                psi_df = psi_df.dropna(subset=["value"]).sort_values("value", ascending=False).reset_index(drop=True)
                if not psi_df.empty:
                    st.markdown("#### PSI by feature")
                    st.caption("Green < 0.10 (stable) · Amber 0.10–0.25 (minor shift) · Red > 0.25 (major shift)")
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as _plt
                    _fig, _ax = _plt.subplots(figsize=(7, max(2.5, len(psi_df) * 0.38)))
                    _colors = ["#E24B4A" if v > 0.25 else "#EF9F27" if v > 0.10 else "#1D9E75" for v in psi_df["value"]]
                    _ax.barh(psi_df["feature"].astype(str), psi_df["value"], color=_colors)
                    _ax.axvline(0.10, linestyle="--", linewidth=0.8, color="#EF9F27", alpha=0.7)
                    _ax.axvline(0.25, linestyle="--", linewidth=0.8, color="#E24B4A", alpha=0.7)
                    _ax.set_xlabel("PSI")
                    _ax.set_title("Population Stability Index per feature")
                    _ax.invert_yaxis()
                    _fig.tight_layout()
                    st.pyplot(_fig)
                    _plt.close(_fig)
            st.dataframe(dataframe_for_display(dist), width="stretch", hide_index=True)

    with tab4:
        st.markdown("#### Pattern activation summary")
        if pattern_summary.empty:
            st.info(
                "Pattern activation transform is not exposed by this installed model version. "
                "The Pattern Inventory page still provides global pattern evidence."
            )
        else:
            st.dataframe(dataframe_for_display(pattern_summary), width="stretch", hide_index=True)


    with tab5:
        render_cv_monitoring(model, X, y, cv=5)


def _cv_monitoring_cache_key(model: Any, X: pd.DataFrame | None, y: Any, cv: int) -> str:
    n_rows = len(X) if X is not None else 0
    n_targets = len(y) if y is not None and hasattr(y, "__len__") else 0
    return f"cv_monitoring::{id(model)}::{n_rows}::{n_targets}::{cv}"


def _safe_cv_monitoring_report(model: Any, X: pd.DataFrame | None, y: Any, cv: int = 5) -> tuple[dict[str, Any] | None, str | None]:
    if model is None or X is None or y is None:
        return None, "A fitted model, X, and y are required for cross-validation monitoring."
    method = getattr(model, "cross_validate_monitored", None)
    if not callable(method):
        return None, "cross_validate_monitored() is not available on this model/version."
    try:
        report = method(X, y, cv=cv)
    except Exception as exc:
        return None, f"cross_validate_monitored() could not be computed: {exc}"
    if not isinstance(report, dict):
        return None, "cross_validate_monitored() returned an unsupported report shape."
    return report, None


def _cv_scores_frame(report: dict[str, Any]) -> pd.DataFrame:
    scores = report.get("test_scores", [])
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(scores, start=1):
        if isinstance(item, dict):
            auc = item.get("auc", item.get("roc_auc", item.get("score", np.nan)))
            row = {"fold": i, **item, "auc": auc}
        else:
            row = {"fold": i, "auc": item}
        rows.append(row)
    return pd.DataFrame(rows)


def _cv_drift_frame(report: dict[str, Any]) -> pd.DataFrame:
    drift = report.get("fold_drift", [])
    rows: list[dict[str, Any]] = []
    if isinstance(drift, dict):
        iterator = drift.items()
    else:
        iterator = enumerate(drift, start=1)
    for fold_key, fold_value in iterator:
        fold = int(fold_key) if isinstance(fold_key, (int, np.integer)) else fold_key
        if isinstance(fold_value, pd.DataFrame):
            df = fold_value.copy()
            if "feature" not in df.columns:
                df = df.reset_index().rename(columns={"index": "feature"})
            for _, row in df.iterrows():
                psi = row.get("psi", row.get("PSI", np.nan))
                rows.append({"fold": fold, "feature": row.get("feature"), "psi": psi})
        elif isinstance(fold_value, dict):
            for feature, value in fold_value.items():
                psi = value.get("psi", value.get("PSI", np.nan)) if isinstance(value, dict) else value
                rows.append({"fold": fold, "feature": feature, "psi": psi})
    return pd.DataFrame(rows)


def _cv_metadata_frame(report: dict[str, Any]) -> pd.DataFrame:
    metadata = report.get("fold_metadata", [])
    rows: list[dict[str, Any]] = []
    if isinstance(metadata, dict):
        iterator = metadata.items()
    else:
        iterator = enumerate(metadata, start=1)
    fit_times = report.get("fit_times_ms", [])
    for fold_key, value in iterator:
        fold = int(fold_key) if isinstance(fold_key, (int, np.integer)) else fold_key
        row = {"fold": fold}
        if isinstance(value, dict):
            row.update(value)
        else:
            row["metadata"] = value
        idx = int(fold) - 1 if isinstance(fold, int) else None
        if idx is not None and idx < len(fit_times) and "fit_time_ms" not in row:
            row["fit_time_ms"] = fit_times[idx]
        rows.append(row)
    if not rows and fit_times:
        rows = [{"fold": i + 1, "fit_time_ms": t} for i, t in enumerate(fit_times)]
    return pd.DataFrame(rows)


def render_cv_monitoring(model: Any = None, X: pd.DataFrame | None = None, y: Any = None, cv: int = 5) -> dict[str, Any] | None:
    """Render cross-validated monitoring stability evidence."""
    st.markdown("#### Cross-Validation Monitoring")
    st.caption(
        "Evaluates whether prediction behaviour, fold-level PSI, fit timing, and pattern counts remain stable across CV folds."
    )

    cache_key = _cv_monitoring_cache_key(model, X, y, cv)
    if cache_key not in st.session_state:
        report, error = _safe_cv_monitoring_report(model, X, y, cv=cv)
        st.session_state[cache_key] = {"report": report, "error": error}
    cached = st.session_state.get(cache_key, {})
    report = cached.get("report")
    error = cached.get("error")
    if error:
        st.info(error)
        return None
    if not isinstance(report, dict):
        st.info("Cross-validation monitoring report is not available.")
        return None

    scores = _cv_scores_frame(report)
    drift = _cv_drift_frame(report)
    metadata = _cv_metadata_frame(report)

    c1, c2, c3 = st.columns(3)
    if not scores.empty and "auc" in scores.columns:
        auc = pd.to_numeric(scores["auc"], errors="coerce")
        c1.metric("Mean CV AUC", f"{auc.mean():.4f}" if auc.notna().any() else "N/A")
    else:
        c1.metric("Mean CV AUC", "N/A")
    if not drift.empty and "psi" in drift.columns:
        psi = pd.to_numeric(drift["psi"], errors="coerce")
        c2.metric("Max fold PSI", f"{psi.max():.4f}" if psi.notna().any() else "N/A")
    else:
        c2.metric("Max fold PSI", "N/A")
    pattern_col = next((c for c in ("pattern_count", "n_patterns", "patterns") if c in metadata.columns), None)
    if pattern_col:
        pc = pd.to_numeric(metadata[pattern_col], errors="coerce")
        c3.metric("Pattern count range", f"{int(pc.min())}–{int(pc.max())}" if pc.notna().any() else "N/A")
    else:
        c3.metric("Pattern count range", "N/A")

    tab1, tab2, tab3 = st.tabs(["Fold AUC", "Fold PSI", "Fit / pattern stability"])
    with tab1:
        if scores.empty:
            st.info("Fold test scores are not available in the CV monitoring report.")
        else:
            chart_df = scores[["fold", "auc"]].copy() if "auc" in scores.columns else scores.copy()
            if "auc" in chart_df.columns:
                st.bar_chart(chart_df.set_index("fold")["auc"])
            st.dataframe(dataframe_for_display(scores), width="stretch", hide_index=True)
    with tab2:
        if drift.empty:
            st.info("Fold PSI drift output is not available in the CV monitoring report.")
        else:
            heat = drift.pivot_table(index="feature", columns="fold", values="psi", aggfunc="mean")
            st.dataframe(dataframe_for_display(heat.reset_index()), width="stretch", hide_index=True)
            with st.expander("Long-form fold PSI table", expanded=False):
                st.dataframe(dataframe_for_display(drift), width="stretch", hide_index=True)
    with tab3:
        if metadata.empty:
            st.info("Fold metadata is not available in the CV monitoring report.")
        else:
            if pattern_col:
                st.line_chart(metadata.set_index("fold")[[pattern_col]])
            st.dataframe(dataframe_for_display(metadata), width="stretch", hide_index=True)
    return report

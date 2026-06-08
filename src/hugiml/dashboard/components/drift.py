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


def render_drift(model: Any = None, X: pd.DataFrame | None = None, *args, **kwargs) -> None:
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

    tab1, tab2, tab3, tab4 = st.tabs([
        "Prediction distribution",
        "Missingness",
        "Distribution review",
        "Pattern activation",
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

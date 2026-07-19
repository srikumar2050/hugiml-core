"""Performance comparison component."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hugiml.dashboard.display import dataframe_for_display


def _as_1d_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim > 1:
        arr = arr.ravel()
    return arr


def _positive_probabilities(model: Any, X: pd.DataFrame | None) -> np.ndarray | None:
    if model is None or X is None:
        return None
    try:
        proba = model.predict_proba(X)
    except Exception:
        return None
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr[:, 1]
    return arr.ravel()


def _binary_y(y: Any) -> np.ndarray | None:
    if y is None:
        return None
    arr = _as_1d_array(y)
    if arr.size == 0:
        return None
    try:
        numeric = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
        finite = numeric[np.isfinite(numeric)]
        unique = sorted(set(finite.tolist()))
        if set(unique).issubset({0.0, 1.0}) and len(unique) <= 2:
            return numeric.astype(int)
    except Exception:
        pass
    s = pd.Series(arr).astype(str)
    values = sorted(s.dropna().unique().tolist())
    if len(values) != 2:
        return None
    positive = values[-1]
    return (s == positive).astype(int).to_numpy()


def _auc_from_scores(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    # Average tied ranks.
    series = pd.DataFrame({"score": score, "rank": ranks})
    avg_ranks = series.groupby("score", sort=False)["rank"].transform("mean").to_numpy()
    sum_pos_ranks = float(avg_ranks[y_true == 1].sum())
    return (sum_pos_ranks - pos * (pos + 1) / 2.0) / float(pos * neg)


def _threshold_sweep_frame(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    thresholds = np.round(np.linspace(0.0, 1.0, 21), 2)
    rows: list[dict[str, Any]] = []
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        accuracy = (tp + tn) / len(y_true) if len(y_true) else np.nan
        rows.append({
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "accuracy": accuracy,
            "predicted_positive_rate": float(pred.mean()) if len(pred) else np.nan,
        })
    return pd.DataFrame(rows)


def score_separation_frames(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    """Build histogram data for class-separated score distributions."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, 21)
    pos_counts, _ = np.histogram(p[y == 1], bins=bins)
    neg_counts, _ = np.histogram(p[y == 0], bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    sep_df = pd.DataFrame({
        "score_bin": [f"{bins[i]:.2f}" for i in range(len(centers))],
        "y=1 (positive)": pos_counts.astype(int),
        "y=0 (negative)": neg_counts.astype(int),
    })
    # Calibration: 10 equal-width bins
    cal_bins = np.linspace(0.0, 1.0, 11)
    rows = []
    for i in range(len(cal_bins) - 1):
        mask = (p >= cal_bins[i]) & (p < cal_bins[i + 1])
        if mask.sum() >= 3:
            rows.append({
                "predicted": float(np.mean(p[mask])),
                "actual": float(np.mean(y[mask])),
                "n": int(mask.sum()),
            })
    cal_df = pd.DataFrame(rows)
    return {"separation": sep_df, "calibration": cal_df}


def _diagnostic_inputs(
    model: Any = None,
    X: pd.DataFrame | None = None,
    y: Any = None,
    evaluation: dict[str, Any] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Resolve diagnostics from promoted evidence before considering a refit.

    A promoted Workbench run carries stitched out-of-fold predictions.  Those
    arrays are the evaluation contract.  Re-running ``predict_proba`` on the
    final full-data refit would produce training diagnostics and is therefore
    deliberately only the fallback for non-promoted contexts.
    """
    if isinstance(evaluation, dict):
        try:
            y_true = _binary_y(evaluation.get("y_true"))
            proba = np.asarray(evaluation.get("y_proba"), dtype=float).reshape(-1)
            pred = np.asarray(evaluation.get("y_pred"), dtype=int).reshape(-1)
        except Exception:
            y_true, proba, pred = None, np.asarray([]), np.asarray([])
        if y_true is not None:
            n = min(len(y_true), len(proba), len(pred))
            if n > 0:
                return y_true[:n], np.clip(proba[:n], 0.0, 1.0), pred[:n]

    y_true = _binary_y(y)
    proba = _positive_probabilities(model, X)
    if y_true is None or proba is None:
        return None, None, None
    n = min(len(y_true), len(proba))
    if n == 0:
        return None, None, None
    y_true = y_true[:n]
    proba = np.clip(np.asarray(proba[:n], dtype=float), 0.0, 1.0)
    return y_true, proba, (proba >= 0.5).astype(int)


def performance_diagnostic_frames(
    model: Any = None,
    X: pd.DataFrame | None = None,
    y: Any = None,
    evaluation: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build validation diagnostics from one explicitly identified evidence set."""
    y_true, proba, supplied_pred = _diagnostic_inputs(model, X, y, evaluation)
    empty = {
        "metrics": pd.DataFrame(),
        "confusion": pd.DataFrame(),
        "thresholds": pd.DataFrame(),
        "curves": pd.DataFrame(),
        "evaluation": pd.DataFrame(),
    }
    if y_true is None or proba is None or supplied_pred is None:
        return empty

    pred_50 = (proba >= 0.5).astype(int)
    pred = np.asarray(supplied_pred, dtype=int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision + recall) and (precision + recall) else np.nan
    brier = float(np.mean((proba - y_true) ** 2))
    auc = _auc_from_scores(y_true, proba)
    thresholds = _threshold_sweep_frame(y_true, proba)
    metrics = pd.DataFrame([
        {"metric": "roc_auc", "value": auc},
        {"metric": "brier_score", "value": brier},
        {"metric": "accuracy", "value": float((pred == y_true).mean())},
        {"metric": "precision", "value": precision},
        {"metric": "recall", "value": recall},
        {"metric": "f1", "value": f1},
        {"metric": "accuracy_at_0_50", "value": float((pred_50 == y_true).mean())},
        {"metric": "positive_rate", "value": float(y_true.mean())},
        {"metric": "predicted_positive_rate", "value": float(pred.mean())},
        {"metric": "predicted_positive_rate_at_0_50", "value": float(pred_50.mean())},
    ])
    confusion = pd.DataFrame([
        {"actual": 0, "predicted": 0, "count": tn},
        {"actual": 0, "predicted": 1, "count": fp},
        {"actual": 1, "predicted": 0, "count": fn},
        {"actual": 1, "predicted": 1, "count": tp},
    ])
    curves = thresholds[["threshold", "recall", "precision", "specificity", "predicted_positive_rate"]].copy()
    evaluation_row = {
        "source": evaluation.get("source") if isinstance(evaluation, dict) else "fitted_model_rescore",
        "scope": evaluation.get("scope") if isinstance(evaluation, dict) else "full_data_refit",
        "run_id": evaluation.get("run_id") if isinstance(evaluation, dict) else None,
        "cv": evaluation.get("cv") if isinstance(evaluation, dict) else None,
        "random_state": evaluation.get("random_state") if isinstance(evaluation, dict) else None,
        "n_rows": int(len(y_true)),
    }
    return {
        "metrics": metrics,
        "confusion": confusion,
        "thresholds": thresholds,
        "curves": curves,
        "evaluation": pd.DataFrame([evaluation_row]),
    }



def normalize_validation_results(df: pd.DataFrame) -> pd.DataFrame:
    """Give validation result columns audit-friendly names.

    The training/tuning result object historically exposed a generic ``score``
    column. In the dashboard this is not meaningful to auditors, so display it
    as ``validation_roc_auc`` when no more explicit ROC-AUC column exists.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    if "score" in out.columns and not any(c in out.columns for c in ("validation_roc_auc", "cv_roc_auc", "roc_auc")):
        out = out.rename(columns={"score": "validation_roc_auc"})
    return out


def render_performance(
    results: Any = None,
    model: Any = None,
    X: pd.DataFrame | None = None,
    y: Any = None,
    evaluation: dict[str, Any] | None = None,
    *args,
    **kwargs,
) -> None:
    st.subheader("Validation Evidence")
    st.markdown(
        """
        <div class="hugiml-section-note">
          <p>Shows HUGIML tuning/CV evidence and one consistent diagnostic evidence set. For promoted
          Workbench runs, Governance uses the exact stitched out-of-fold predictions carried by the
          promotion; it does not replace them with full-data refit scores.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    diagnostics = performance_diagnostic_frames(model, X, y, evaluation=evaluation)
    diag_available = not diagnostics["metrics"].empty

    if results is None:
        df = pd.DataFrame()
    else:
        try:
            df = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
        except Exception as exc:
            st.warning(f"Could not convert results to a table: {exc}")
            st.write(results)
            df = pd.DataFrame()

    df = normalize_validation_results(df)

    promoted_metrics = evaluation.get("metrics", {}) if isinstance(evaluation, dict) else {}
    promoted_cv = promoted_metrics.get("cv_roc_auc") if isinstance(promoted_metrics, dict) else None
    table_cv = None
    if not df.empty:
        if "mean_test_score" in df.columns:
            values = pd.to_numeric(df["mean_test_score"], errors="coerce")
            if values.notna().any():
                table_cv = float(values.max())
        elif "cv_roc_auc" in df.columns:
            values = pd.to_numeric(df["cv_roc_auc"], errors="coerce")
            if values.notna().any():
                table_cv = float(values.max())
        elif "validation_roc_auc" in df.columns:
            values = pd.to_numeric(df["validation_roc_auc"], errors="coerce")
            if values.notna().any():
                # SimpleTuneResult stores one row per fold, so best_score_ is
                # the mean rather than the largest individual fold.
                table_cv = float(values.mean()) if "fold" in df.columns else float(values.max())
        elif "score" in df.columns:
            values = pd.to_numeric(df["score"], errors="coerce")
            if values.notna().any():
                table_cv = float(values.mean()) if "fold" in df.columns else float(values.max())
    if promoted_cv is not None and table_cv is not None and not np.isclose(
        float(promoted_cv), float(table_cv), rtol=1e-8, atol=1e-10
    ):
        st.error(
            "The promoted Workbench CV score "
            f"({float(promoted_cv):.6g}) differs from the Governance result table ({float(table_cv):.6g})."
        )

    if df.empty and not diag_available:
        st.info("No performance result table or label-aware diagnostics are available.")
        return

    tab1, tab2, tab3 = st.tabs(["CV / tuning", "Validation diagnostics", "Threshold sweep"])
    with tab1:
        if df.empty:
            st.info("No performance rows were returned.")
        else:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            score_cols = [c for c in numeric_cols if "score" in str(c).lower() or "auc" in str(c).lower()]
            score_col = score_cols[0] if score_cols else None
            best_idx = None
            if score_col:
                score_values = pd.to_numeric(df[score_col], errors="coerce")
                fold_rows = "fold" in df.columns and score_col in {"validation_roc_auc", "score"}
                if fold_rows:
                    summary_value = float(score_values.mean())
                    summary_label = "Mean validation ROC-AUC"
                else:
                    best_idx = score_values.idxmax()
                    summary_value = float(score_values.loc[best_idx])
                    summary_label = "Best validation ROC-AUC"
                c1, c2, c3 = st.columns(3)
                c1.metric(summary_label, f"{summary_value:.4f}")
                c2.metric("Rows compared", f"{len(df):,}")
                c3.metric("Validation metric", str(score_col))
            with st.container(border=True):
                st.markdown("#### CV / tuning result table")
                st.dataframe(dataframe_for_display(df), width="stretch", hide_index=True)
            if score_col and best_idx is not None:
                with st.container(border=True):
                    st.markdown("#### Best row by validation metric")
                    st.dataframe(df.loc[[best_idx]], width="stretch", hide_index=True)

    with tab2:
        if not diag_available:
            st.info("Validation diagnostics require a fitted model, X, and binary y.")
        else:
            metrics = diagnostics["metrics"]
            metric_map = dict(zip(metrics["metric"], metrics["value"]))
            scope = str(evaluation.get("scope") or "unknown") if isinstance(evaluation, dict) else "full_data_refit"
            is_oof = scope == "out_of_fold"
            if is_oof:
                st.info(
                    "These diagnostics use the exact stitched out-of-fold predictions promoted from Workbench. "
                    "The fitted full-data model is used for rules and explanations only, not to replace validation numbers."
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean-fold CV ROC-AUC", f"{float(promoted_cv):.4f}" if promoted_cv is not None else "N/A")
                c2.metric("Pooled OOF ROC-AUC", f"{metric_map.get('roc_auc', np.nan):.4f}")
                c3.metric("OOF F1", f"{metric_map.get('f1', np.nan):.2%}")
                c4.metric("OOF accuracy", f"{metric_map.get('accuracy', np.nan):.2%}")
                st.markdown("#### Promoted out-of-fold confusion matrix")
            else:
                st.warning(
                    "No promoted out-of-fold evidence was available. These diagnostics were computed from the "
                    "full-data refit and must be treated as training diagnostics."
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Refit ROC-AUC", f"{metric_map.get('roc_auc', np.nan):.4f}")
                c2.metric("Brier score", f"{metric_map.get('brier_score', np.nan):.4f}")
                c3.metric("Refit accuracy", f"{metric_map.get('accuracy', np.nan):.2%}")
                st.markdown("#### Full-data refit confusion matrix")
            st.dataframe(dataframe_for_display(diagnostics["confusion"]), width="stretch", hide_index=True)

            # Score separation histogram from the same diagnostic evidence.
            _y_true_sep, _proba_sep, _ = _diagnostic_inputs(model, X, y, evaluation)
            if _y_true_sep is not None and _proba_sep is not None:
                _sep_frames = score_separation_frames(_y_true_sep, _proba_sep)
                st.markdown("#### Score distribution by class")
                st.caption("Overlap between distributions indicates cases where the model is uncertain.")
                _sep_df = _sep_frames["separation"]
                if not _sep_df.empty:
                    _fig_sep, _ax_sep = plt.subplots(figsize=(7, 3.5))
                    _x = np.arange(len(_sep_df))
                    _w = 0.38
                    _ax_sep.bar(_x - _w/2, _sep_df["y=1 (positive)"], _w, label="y=1 (positive)", color="#1D9E75", alpha=0.85)
                    _ax_sep.bar(_x + _w/2, _sep_df["y=0 (negative)"], _w, label="y=0 (negative)", color="#E24B4A", alpha=0.85)
                    _ax_sep.set_xticks(_x[::2])
                    _ax_sep.set_xticklabels(_sep_df["score_bin"].iloc[::2], rotation=45, ha="right", fontsize=8)
                    _ax_sep.set_xlabel("Predicted probability")
                    _ax_sep.set_ylabel("Row count")
                    _ax_sep.legend(fontsize=9)
                    _fig_sep.tight_layout()
                    st.pyplot(_fig_sep)
                    plt.close(_fig_sep)

                # Calibration plot
                st.markdown("#### Calibration plot")
                st.caption("Points on the diagonal y = x indicate perfect calibration.")
                _cal_df = _sep_frames["calibration"]
                if not _cal_df.empty:
                    _fig_cal, _ax_cal = plt.subplots(figsize=(5, 4))
                    _ax_cal.plot([0, 1], [0, 1], linestyle="--", color="#888780", linewidth=0.8, label="Perfect calibration")
                    _ax_cal.scatter(_cal_df["predicted"], _cal_df["actual"], s=_cal_df["n"].clip(5, 200), color="#534AB7", alpha=0.85, label="Model bins")
                    _ax_cal.set_xlabel("Mean predicted probability")
                    _ax_cal.set_ylabel("Observed positive rate")
                    _ax_cal.set_xlim(0, 1)
                    _ax_cal.set_ylim(0, 1)
                    _ax_cal.legend(fontsize=9)
                    _fig_cal.tight_layout()
                    st.pyplot(_fig_cal)
                    plt.close(_fig_cal)

            st.markdown("#### ROC / PR curve points")
            st.line_chart(diagnostics["curves"].set_index("threshold")[["recall", "precision", "specificity"]])
            st.dataframe(dataframe_for_display(metrics), width="stretch", hide_index=True)

    with tab3:
        if diagnostics["thresholds"].empty:
            st.info("Threshold sweep requires prediction probabilities and binary y.")
        else:
            st.dataframe(dataframe_for_display(diagnostics["thresholds"]), width="stretch", hide_index=True)

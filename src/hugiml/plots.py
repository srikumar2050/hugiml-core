# Copyright 2026 Srikumar Krishnamoorthy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
HUG-IML first-class visualizations using Plotly.

Public API
----------
    from hugiml.plots import HUGPlotter

    plotter = HUGPlotter(clf)
    fig = plotter.plot_marginal_bin_profile("age", X)   # EBM shape-function equivalent
    fig = plotter.plot_feature_combinations("age")       # compound patterns for one feature
    fig = plotter.plot_feature_importance(top_n=15)
    fig = plotter.plot_utility_vs_ig()                   # scatter: utility × IG × support
    fig = plotter.plot_top_patterns(top_n=20)
    fig = plotter.plot_feature_coverage()
    fig = plotter.plot_pattern_lengths()
    fig = plotter.plot_support_distribution()
    fig = plotter.plot_active_patterns(X, sample_idx=0) # local explanation
    fig = plotter.plot_dashboard(X)                      # full multi-panel HTML
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _PLOTLY = True
except ImportError:
    _PLOTLY = False


__all__ = [
    "HUGPlotter",
]

# ---------------------------------------------------------------------------
# Design tokens (matching the governance dashboard)
# ---------------------------------------------------------------------------

_T = {
    "bg": "#0d1117",
    "panel": "#161b22",
    "border": "#30363d",
    "grid": "#21262d",
    "text": "#e6edf3",
    "muted": "#8b949e",
    "a1": "#58a6ff",  # blue  – utility / main bars
    "a2": "#3fb950",  # green – combinations +1
    "a3": "#f78166",  # red   – combinations +3 / negative
    "a4": "#d2a8ff",  # purple– support dist histogram
    "a5": "#ffa657",  # orange– support line overlay
    "font_b": "JetBrains Mono, monospace",
    "font_t": "Syne, sans-serif",
}

# Plotly 6 rejects duplicate kwargs in update_layout(**_LAYOUT_BASE, margin=...).
# margin and legend are excluded here; each chart sets them in a separate call.
_LAYOUT_BASE = dict(
    paper_bgcolor=_T["bg"],
    plot_bgcolor=_T["panel"],
    font=dict(color=_T["text"], family=_T["font_b"], size=11),
    xaxis=dict(tickfont=dict(color=_T["muted"]), gridcolor=_T["grid"], linecolor=_T["border"]),
    yaxis=dict(tickfont=dict(color=_T["muted"]), gridcolor=_T["grid"], linecolor=_T["border"]),
)
_LAYOUT_MARGIN = dict(l=44, r=20, t=50, b=40)
_LAYOUT_LEGEND = dict(
    font=dict(color=_T["muted"]), bgcolor="rgba(0,0,0,0)", bordercolor=_T["border"]
)


def _title(text: str) -> dict:
    return dict(text=text, font=dict(color=_T["a1"], size=13, family=_T["font_t"]))


def _require_plotly() -> None:
    if not _PLOTLY:
        raise ImportError("plotly is required for HUGPlotter. Install with: pip install plotly")


# ---------------------------------------------------------------------------
# HUGPlotter
# ---------------------------------------------------------------------------


def _parse_bin_lower(bin_label: str) -> float | None:
    """Parse the lower bound from a bin label like '[7.69,11.41]' or '[11.8,17.5)'.

    Returns None for non-numeric labels (categorical values without brackets).
    """
    if bin_label.startswith("["):
        try:
            return float(bin_label[1:].split(",")[0])
        except (ValueError, IndexError):
            pass
    return None


def _get_bin_edges(clf, feature_name: str):
    """Return real-value bin edges for *feature_name*, or None if unavailable.

    Works for both HUGIMLAdaptive (uses _bin_edges_) and native
    HUGIMLClassifierNative (reconstructs from td_._cpp_all_edges).
    """
    # HUGIMLAdaptive path — edges stored directly
    if hasattr(clf, "_bin_edges_") and feature_name in clf._bin_edges_:
        return np.array(clf._bin_edges_[feature_name])

    # Native clf path — reconstruct from normalised C++ edges
    feature_names = getattr(clf, "feature_names_in_", None)
    if feature_names is None:
        return None
    try:
        j = list(feature_names).index(feature_name)
    except ValueError:
        return None
    td = getattr(clf, "td_", None)
    if td is None:
        return None
    all_edges = getattr(td, "_cpp_all_edges", None)
    if all_edges is None or j >= len(all_edges):
        return None
    edges_norm = np.array(all_edges[j])
    if len(edges_norm) < 2:
        return None
    col_min_arr = getattr(td, "_cpp_col_min", None)
    col_range_arr = getattr(td, "_cpp_col_range", None)
    if col_min_arr is None or col_range_arr is None:
        return None
    col_min = float(col_min_arr[j])
    col_range = float(col_range_arr[j])
    return edges_norm * col_range + col_min


class HUGPlotter:
    """Unified Plotly-based visualization interface for a fitted HUGIMLClassifierNative.

    Parameters
    ----------
    clf : fitted HUGIMLClassifierNative
    height_default : int
        Default figure height.
    """

    def __init__(self, clf: Any, height_default: int = 380) -> None:
        _require_plotly()
        self._clf = clf
        self._h = height_default
        self._validate_fitted()

    # ------------------------------------------------------------------
    # 1. Marginal Bin Profile  (= EBM per-feature shape function)
    # ------------------------------------------------------------------

    def plot_marginal_bin_profile(
        self,
        feature_name: str,
        X: Any | None = None,
        height: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """1-D HUG profile — EBM shape function equivalent.

        For a given feature, shows every singleton pattern bin as a bar
        (x = bin label, y = utility, colour = information gain).
        An orange dotted line overlays the training support fraction on the
        right y-axis, mirroring the dashboard's "Marginal Bin Profile" card.

        Parameters
        ----------
        feature_name : str
        X : ignored (support uses training data stored in clf.x_train_hup_)
        height : int, optional
        title : str, optional

        Returns
        -------
        plotly.graph_objects.Figure
        """
        clf = self._clf
        all_labels = clf.get_hug_features()
        n_train = clf.x_train_hup_.shape[0]

        # ── Step 1: collect all singleton patterns for this feature ─────────
        # Key: lower-bound float → (bin_label_str, util, ig, sup)
        # Using lower bound as key lets us match patterns to edge-derived bins
        # regardless of minor float formatting differences between the two.
        mined: dict[float, tuple[str, float, float, float]] = {}
        for idx, (label, pe) in enumerate(zip(all_labels, clf.patterns_)):
            if len(pe.items) != 1:
                continue
            parts = label.split(", ")
            if len(parts) == 1 and label.startswith(feature_name + "="):
                bin_label = label[len(feature_name) + 1 :]
                lo = _parse_bin_lower(bin_label)
                util = float(pe.utility)
                ig = float(pe.ig)
                sup = float(clf.x_train_hup_[:, idx].sum()) / n_train * 100
                key = lo if lo is not None else float("inf")
                mined[key] = (bin_label, util, ig, sup)

        # ── Step 2: build the dense ordered bin sequence ────────────────────
        # Try to get ALL bin edges so empty bins (utility ≈ 0, not mined) are
        # shown as zero-height bars rather than silently dropped.
        real_edges = _get_bin_edges(clf, feature_name)

        if real_edges is not None and len(real_edges) >= 2:
            # Reconstruct every bin from edges; fill missing ones with zeros.
            bins_x, bins_util, bins_ig, bins_sup, bins_mined = [], [], [], [], []
            for i in range(len(real_edges) - 1):
                lo_val = float(real_edges[i])
                hi_val = float(real_edges[i + 1])
                # Find the closest mined bin by lower bound (1% tolerance)
                match = None
                for key, data in mined.items():
                    if key != float("inf") and abs(key - lo_val) / (abs(lo_val) + 1e-9) < 0.02:
                        match = data
                        break
                if match:
                    bins_x.append(match[0])
                    bins_util.append(match[1])
                    bins_ig.append(match[2])
                    bins_sup.append(match[3])
                    bins_mined.append(True)
                else:
                    # Construct a readable label for the un-mined bin
                    bins_x.append(f"[{lo_val:.4g},{hi_val:.4g}]")
                    bins_util.append(0.0)
                    bins_ig.append(0.0)
                    bins_sup.append(0.0)
                    bins_mined.append(False)
        else:
            # Fallback: only mined bins, sorted left-to-right by lower bound
            if not mined:
                fig = go.Figure()
                fig.update_layout(
                    **_LAYOUT_BASE,
                    title=_title(title or f"Marginal Bin Profile · {feature_name}"),
                    height=height or self._h,
                )
                fig.add_annotation(
                    text=f"No singleton patterns for '{feature_name}'. "
                    "Try reducing G or increasing topK.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color=_T["muted"]),
                )
                return fig
            sorted_items = sorted(mined.items(), key=lambda kv: kv[0])
            bins_x = [v[0] for _, v in sorted_items]
            bins_util = [v[1] for _, v in sorted_items]
            bins_ig = [v[2] for _, v in sorted_items]
            bins_sup = [v[3] for _, v in sorted_items]
            bins_mined = [True] * len(bins_x)

        n_mined = sum(bins_mined)
        n_total = len(bins_x)
        subtitle = (
            f"{n_mined}/{n_total} bins have patterns"
            if n_mined < n_total
            else f"All {n_total} bins have patterns"
        )

        # ── Step 3: colours — IG intensity for mined bins, gray for empty ───
        max_ig = max(bins_ig) if any(bins_ig) else 1
        colors = []
        for ig, is_m in zip(bins_ig, bins_mined):
            if is_m:
                alpha = 0.4 + 0.6 * ig / (max_ig + 1e-9)
                colors.append(f"rgba(88,166,255,{alpha:.2f})")
            else:
                colors.append("rgba(100,100,100,0.25)")  # grey — no pattern mined

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=bins_x,
                y=bins_util,
                name="Utility",
                marker=dict(color=colors, line=dict(color=_T["border"], width=0.5)),
                hovertemplate="<b>%{x}</b><br>Utility: %{y:.4f}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=bins_x,
                y=bins_sup,
                name="Support %",
                mode="lines+markers",
                line=dict(color=_T["a5"], dash="dot", width=2.5),
                marker=dict(color=_T["a5"], size=7),
                hovertemplate="Support: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            **_LAYOUT_BASE,
            title=_title(title or f"Marginal Bin Profile · {feature_name}"),
            height=height or self._h,
            annotations=[
                dict(
                    text=subtitle,
                    xref="paper",
                    yref="paper",
                    x=0.99,
                    y=0.99,
                    showarrow=False,
                    xanchor="right",
                    font=dict(size=10, color=_T["muted"]),
                )
            ],
        )
        fig.update_layout(
            legend=dict(
                x=0.01,
                y=0.99,
                font=dict(color=_T["muted"]),
                bgcolor="rgba(0,0,0,0)",
                bordercolor=_T["border"],
            )
        )
        fig.update_xaxes(
            title_text="Bin Range",
            tickangle=-35,
            tickfont=dict(size=9, color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        fig.update_yaxes(
            title_text="Utility",
            secondary_y=False,
            title_font=dict(color=_T["a1"]),
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        fig.update_yaxes(
            title_text="Support %",
            secondary_y=True,
            title_font=dict(color=_T["a5"]),
            showgrid=False,
            tickfont=dict(color=_T["a5"]),
        )
        return fig

    # ------------------------------------------------------------------
    # 2. Feature Combinations
    # ------------------------------------------------------------------

    def plot_feature_combinations(
        self,
        feature_name: str,
        top_n: int = 25,
        height: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Compound patterns that include a specific feature.

        Each bar = one compound pattern; bars coloured by the number of
        extra features (+1 = green, +2 = orange, +3 = red), matching the
        dashboard's "Feature Combinations" card.

        Parameters
        ----------
        feature_name : str
        top_n : int
        height : int, optional
        title : str, optional

        Returns
        -------
        go.Figure
        """
        clf = self._clf
        all_labels = clf.get_hug_features()
        n_train = clf.x_train_hup_.shape[0]

        # Collect compound patterns for this feature.
        # Split compound labels ("feat1=[lo,hi], feat2=val") and check each
        # part with startswith to avoid substring matches like "age" in "mortgage_age=...".
        rows = []
        for idx, (label, pe) in enumerate(zip(all_labels, clf.patterns_)):
            if len(pe.items) < 2:
                continue
            parts = [p.strip() for p in label.split(", ")]
            if not any(p.startswith(feature_name + "=") for p in parts):
                continue
            util = float(pe.utility)
            sup = float(clf.x_train_hup_[:, idx].sum()) / n_train * 100
            extra = len(pe.items) - 1
            rows.append({"label": label, "util": util, "extra": extra, "sup": sup})

        fig = go.Figure()
        if not rows:
            fig.update_layout(
                **_LAYOUT_BASE,
                title=_title(title or f"Feature Combinations · {feature_name}"),
                height=height or self._h,
            )
            fig.add_annotation(
                text=f"No compound patterns for '{feature_name}'",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color=_T["muted"]),
            )
            return fig

        df = pd.DataFrame(rows).sort_values("util", ascending=False).head(top_n)[::-1]

        # Colour by extra-feature count
        color_map = {1: _T["a2"], 2: _T["a5"], 3: _T["a3"]}
        colors = [color_map.get(e, _T["a4"]) for e in df["extra"]]

        fig.add_trace(
            go.Bar(
                x=df["util"],
                y=[
                    f"{r['label'][:45]}…" if len(r["label"]) > 45 else r["label"]
                    for _, r in df.iterrows()
                ],
                orientation="h",
                marker=dict(color=colors, line=dict(color=_T["border"], width=0.3)),
                text=[f"supp={r['sup']:.1f}%" for _, r in df.iterrows()],
                textfont=dict(color=_T["muted"], size=8),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Utility: %{x:.4f}<extra></extra>",
            )
        )

        # Legend annotation
        fig.add_annotation(
            text=(
                f"<span style='color:{_T['a2']}'>■</span> +1  "
                f"<span style='color:{_T['a5']}'>■</span> +2  "
                f"<span style='color:{_T['a3']}'>■</span> +3 extra features"
            ),
            x=1.01,
            xanchor="right",
            xref="paper",
            y=1.06,
            yref="paper",
            showarrow=False,
            font=dict(color=_T["muted"], size=9),
        )

        fig.update_layout(
            **_LAYOUT_BASE,
            title=_title(title or f"Feature Combinations · {feature_name}"),
            height=max(320, len(df) * 22 + 80) if height is None else height,
        )
        fig.update_xaxes(
            title_text="Utility",
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        fig.update_yaxes(
            tickfont=dict(size=9, color=_T["muted"]),
            autorange="reversed",
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        return fig

    # ------------------------------------------------------------------
    # 3. Feature Importance
    # ------------------------------------------------------------------

    def plot_feature_importance(
        self,
        top_n: int = 15,
        height: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Feature importance: mean utility per feature, coloured by mean IG.

        Matches the "Feature Importance" card in the governance dashboard.

        Parameters
        ----------
        top_n : int
        height : int, optional
        title : str, optional

        Returns
        -------
        go.Figure
        """
        clf = self._clf
        all_labels = clf.get_hug_features()

        feat_stats: dict[str, dict] = {}
        for idx, (label, pe) in enumerate(zip(all_labels, clf.patterns_)):
            parts = label.split(", ")
            for part in parts:
                if "=" in part:
                    fname = part.split("=")[0]
                    entry = feat_stats.setdefault(fname, {"utils": [], "igs": [], "n": 0})
                    entry["utils"].append(float(pe.utility))
                    entry["igs"].append(float(pe.ig))
                    entry["n"] += 1

        rows = []
        for fname, v in feat_stats.items():
            rows.append(
                {
                    "feature": fname,
                    "mean_util": float(np.mean(v["utils"])),
                    "mean_ig": float(np.mean(v["igs"])),
                    "n": v["n"],
                }
            )

        df = pd.DataFrame(rows).sort_values("mean_util", ascending=False).head(top_n)[::-1]

        fig = go.Figure(
            go.Bar(
                x=df["mean_util"],
                y=df["feature"],
                orientation="h",
                marker=dict(
                    color=df["mean_ig"],
                    colorscale=[[0, "#30123b"], [0.5, "#1bcfd4"], [1, "#7a0402"]],
                    showscale=True,
                    colorbar=dict(
                        thickness=11,
                        tickfont=dict(color=_T["muted"]),
                        title=dict(font=dict(color=_T["muted"]), text="Mean IG"),
                    ),
                    line=dict(color=_T["border"], width=0.3),
                ),
                text=[f"n={r['n']}" for _, r in df.iterrows()],
                textfont=dict(color=_T["muted"], size=9),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Mean U: %{x:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            **_LAYOUT_BASE,
            title=_title(title or "Feature Importance"),
            height=height or max(280, len(df) * 28 + 80),
            margin=dict(l=44, r=20, t=76, b=40),
        )
        fig.update_xaxes(
            title_text="Mean Utility",
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        fig.update_yaxes(
            autorange="reversed",
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        return fig

    # ------------------------------------------------------------------
    # 4. Utility vs Information Gain scatter
    # ------------------------------------------------------------------

    def plot_utility_vs_ig(
        self,
        feature_filter: str | None = None,
        height: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Scatter: utility (x) × information gain (y), coloured by support.

        Matches the "Utility vs Info Gain" card in the governance dashboard.
        Optionally filter to patterns containing one feature.

        Parameters
        ----------
        feature_filter : str, optional
            If given, highlight only patterns for this feature.
        height : int, optional
        title : str, optional

        Returns
        -------
        go.Figure
        """
        clf = self._clf
        all_labels = clf.get_hug_features()
        n_train = clf.x_train_hup_.shape[0]

        x_util, y_ig, z_sup, texts = [], [], [], []
        for idx, (label, pe) in enumerate(zip(all_labels, clf.patterns_)):
            if feature_filter and feature_filter + "=" not in label:
                continue
            x_util.append(float(pe.utility))
            y_ig.append(float(pe.ig))
            z_sup.append(float(clf.x_train_hup_[:, idx].sum()) / n_train)
            texts.append(label)

        fig = go.Figure(
            go.Scatter(
                x=x_util,
                y=y_ig,
                mode="markers",
                text=texts,
                hovertemplate="<b>%{text}</b><br>U=%{x:.4f} IG=%{y:.4f}<extra></extra>",
                marker=dict(
                    color=z_sup,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        thickness=11,
                        tickfont=dict(color=_T["muted"]),
                        title=dict(font=dict(color=_T["muted"]), text="Support"),
                    ),
                    line=dict(color=_T["border"], width=0.4),
                    opacity=0.85,
                    size=9,
                ),
            )
        )

        fig.update_layout(
            **_LAYOUT_BASE,
            title=_title(
                title
                or ("Utility vs Info Gain" + (f" · {feature_filter}" if feature_filter else ""))
            ),
            height=height or 430,
            margin=dict(l=44, r=20, t=82, b=40),
        )
        fig.update_xaxes(
            title_text="Utility",
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        fig.update_yaxes(
            title_text="Information Gain",
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        return fig

    # ------------------------------------------------------------------
    # 5. Top Patterns
    # ------------------------------------------------------------------

    def plot_top_patterns(
        self,
        top_n: int = 20,
        height: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Horizontal bar chart of top-N patterns by utility, coloured by IG.

        Matches the "Top Patterns" card in the governance dashboard.

        Parameters
        ----------
        top_n : int
        height : int, optional
        title : str, optional

        Returns
        -------
        go.Figure
        """
        clf = self._clf
        all_labels = clf.get_hug_features()
        n_train = clf.x_train_hup_.shape[0]

        rows = []
        for idx, (label, pe) in enumerate(zip(all_labels, clf.patterns_)):
            sup = float(clf.x_train_hup_[:, idx].sum()) / n_train
            rows.append({"label": label, "util": float(pe.utility), "ig": float(pe.ig), "sup": sup})

        df = pd.DataFrame(rows).sort_values("util", ascending=False).head(top_n)[::-1]

        short = [
            f"{r['label'][:40]}…" if len(r["label"]) > 40 else r["label"] for _, r in df.iterrows()
        ]

        fig = go.Figure(
            go.Bar(
                x=df["util"],
                y=short,
                orientation="h",
                marker=dict(
                    color=df["ig"],
                    colorscale=[[0, "#0d0887"], [0.5, "#bd3786"], [1, "#f0f921"]],
                    showscale=True,
                    colorbar=dict(
                        thickness=11,
                        tickfont=dict(color=_T["muted"]),
                        title=dict(font=dict(color=_T["muted"]), text="IG"),
                    ),
                    line=dict(color=_T["border"], width=0.3),
                ),
                text=[f"s={r['sup']:.3f}" for _, r in df.iterrows()],
                textfont=dict(color=_T["muted"], size=9),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Utility: %{x:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            **_LAYOUT_BASE,
            title=_title(title or "Top Patterns"),
            height=height or max(280, len(df) * 24 + 80),
            margin=dict(l=44, r=20, t=76, b=40),
        )
        fig.update_xaxes(
            title_text="Utility",
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        fig.update_yaxes(
            autorange="reversed",
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        return fig

    # ------------------------------------------------------------------
    # 6. Feature Coverage
    # ------------------------------------------------------------------

    def plot_feature_coverage(
        self,
        top_n: int = 15,
        height: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Horizontal bar: how many patterns reference each feature.

        Matches the "Feature Coverage" card in the governance dashboard.
        """
        clf = self._clf
        all_labels = clf.get_hug_features()

        from collections import Counter

        feat_counts: Counter = Counter()
        for label in all_labels:
            for part in label.split(", "):
                if "=" in part:
                    feat_counts[part.split("=")[0]] += 1

        df = pd.DataFrame(feat_counts.most_common(top_n), columns=["feature", "count"]).iloc[::-1]

        fig = go.Figure(
            go.Bar(
                x=df["count"],
                y=df["feature"],
                orientation="h",
                marker=dict(color=_T["a2"], line=dict(color=_T["border"], width=0.3)),
                text=df["count"].astype(str),
                textposition="auto",
            )
        )

        fig.update_layout(
            **_LAYOUT_BASE,
            title=_title(title or "Feature Coverage"),
            height=height or max(280, len(df) * 26 + 80),
            margin=dict(l=44, r=20, t=76, b=40),
        )
        fig.update_xaxes(
            title_text="# Patterns",
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        fig.update_yaxes(
            autorange="reversed",
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        return fig

    # ------------------------------------------------------------------
    # 7. Pattern Lengths
    # ------------------------------------------------------------------

    def plot_pattern_lengths(
        self,
        height: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Bar chart of pattern length distribution.

        Matches the "Pattern Lengths" card in the governance dashboard.
        """
        from collections import Counter

        lengths = Counter(len(pe.items) for pe in self._clf.patterns_)
        xs = sorted(lengths.keys())
        colors = [_T["a1"] if i == 0 else _T["a2"] for i in range(len(xs))]

        fig = go.Figure(
            go.Bar(
                x=[f"Length {x}" for x in xs],
                y=[lengths[x] for x in xs],
                marker=dict(color=colors[: len(xs)]),
                text=[str(lengths[x]) for x in xs],
                textposition="auto",
            )
        )

        fig.update_layout(
            **_LAYOUT_BASE,
            title=_title(title or "Pattern Lengths"),
            height=height or 280,
        )
        fig.update_xaxes(
            title_text="Pattern Length",
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        fig.update_yaxes(
            title_text="Count",
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        return fig

    # ------------------------------------------------------------------
    # 8. Support Distribution
    # ------------------------------------------------------------------

    def plot_support_distribution(
        self,
        height: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Histogram of pattern support values.

        Matches the "Support Distribution" card in the governance dashboard.
        """
        n_train = self._clf.x_train_hup_.shape[0]
        supports = [
            float(self._clf.x_train_hup_[:, i].sum()) / n_train
            for i in range(len(self._clf.patterns_))
        ]

        fig = go.Figure(
            go.Histogram(
                x=supports,
                nbinsx=25,
                marker=dict(color=_T["a4"], line=dict(color=_T["border"], width=0.3)),
            )
        )

        fig.update_layout(
            **_LAYOUT_BASE,
            title=_title(title or "Support Distribution"),
            height=height or 280,
        )
        fig.update_xaxes(
            title_text="Support",
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        fig.update_yaxes(
            title_text="Count",
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        return fig

    # ------------------------------------------------------------------
    # 9. Active-pattern local explanation
    # ------------------------------------------------------------------

    def plot_active_patterns(
        self,
        X: Any,
        sample_idx: int = 0,
        max_patterns: int = 20,
        height: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Local explanation: active HUG patterns for a single sample.

        Shows active patterns sorted by |coefficient|, coloured blue (positive)
        or red (negative).

        Parameters
        ----------
        X : array-like or DataFrame
        sample_idx : int
        max_patterns : int
        height : int, optional
        title : str, optional

        Returns
        -------
        go.Figure
        """
        clf = self._clf
        is_df = isinstance(X, pd.DataFrame)
        row = X.iloc[[sample_idx]] if is_df else X[sample_idx : sample_idx + 1]
        hup = clf.transform(row)
        active_cols = hup[0].nonzero()[1].tolist()

        all_labels = clf.get_hug_features()
        try:
            imp = clf.feature_importances()
            coef_map = dict(zip(imp["pattern"], imp["coefficient"]))
        except Exception:
            coef_map = {}

        records = []
        for col_idx in active_cols:
            if col_idx < len(all_labels):
                lbl = all_labels[col_idx]
                records.append({"label": lbl, "coef": coef_map.get(lbl, 0.0)})

        records.sort(key=lambda r: abs(r["coef"]), reverse=True)
        records = records[:max_patterns][::-1]

        fig = go.Figure()
        if not records:
            fig.update_layout(
                **_LAYOUT_BASE,
                title=_title(f"Active Patterns — sample #{sample_idx}"),
                height=height or self._h,
            )
            fig.add_annotation(
                text="No active patterns for this sample.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color=_T["muted"]),
            )
            return fig

        coefs = [r["coef"] for r in records]
        colors = [_T["a1"] if c >= 0 else _T["a3"] for c in coefs]
        labels = [f"{r['label'][:50]}…" if len(r["label"]) > 50 else r["label"] for r in records]

        fig.add_trace(
            go.Bar(
                x=coefs,
                y=labels,
                orientation="h",
                marker=dict(color=colors, line=dict(color=_T["border"], width=0.3)),
                hovertemplate="<b>%{y}</b><br>Coef: %{x:.4f}<extra></extra>",
            )
        )

        n_active = len(active_cols)
        fig.update_layout(
            **_LAYOUT_BASE,
            title=_title(title or f"Active Patterns — sample #{sample_idx} ({n_active} active)"),
            height=height or max(280, len(records) * 24 + 80),
        )
        fig.update_xaxes(
            title_text="Coefficient",
            tickfont=dict(color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        fig.update_yaxes(
            tickfont=dict(size=9, color=_T["muted"]), gridcolor=_T["grid"], linecolor=_T["border"]
        )
        return fig

    # ------------------------------------------------------------------
    # 10. Radar / performance summary
    # ------------------------------------------------------------------

    def plot_performance_radar(
        self,
        metrics: dict,
        dataset_name: str = "Dataset",
        height: int | None = None,
    ) -> go.Figure:
        """Radar / spider chart of classification performance metrics.

        Matches the "Performance" card in the governance dashboard.

        Parameters
        ----------
        metrics : dict
            Keys: 'accuracy', 'balanced_accuracy', 'roc_auc', 'f1'
            Values: floats in [0, 1].
        dataset_name : str
        height : int, optional

        Returns
        -------
        go.Figure
        """
        cats = ["Accuracy", "Bal Acc", "ROC AUC", "F1"]
        vals = [
            metrics.get("accuracy", 0),
            metrics.get("balanced_accuracy", 0),
            metrics.get("roc_auc", 0),
            metrics.get("f1", 0),
        ]
        # Close the polygon
        cats = cats + [cats[0]]
        vals = vals + [vals[0]]

        fig = go.Figure(
            go.Scatterpolar(
                r=vals,
                theta=cats,
                fill="toself",
                fillcolor="rgba(88,166,255,.12)",
                line=dict(color=_T["a1"], width=2),
            )
        )

        fig.update_layout(
            **_LAYOUT_BASE,
            title=_title(f"{dataset_name}\nPerformance"),
            height=height or 300,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor=_T["grid"],
                    tickfont=dict(color=_T["muted"]),
                ),
                angularaxis=dict(gridcolor=_T["grid"], tickfont=dict(color=_T["text"])),
                bgcolor=_T["panel"],
            ),
        )
        return fig

    # ------------------------------------------------------------------
    # 11. 2-D HUG profile heatmap
    # ------------------------------------------------------------------

    def plot_2d_profile(
        self,
        feature_a: str,
        feature_b: str,
        height: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """2-D HUG profile heatmap for compound patterns involving two features.

        Parameters
        ----------
        feature_a, feature_b : str
        height : int, optional
        title : str, optional

        Returns
        -------
        go.Figure
        """
        clf = self._clf
        all_labels = clf.get_hug_features()

        def _bin(label: str, feat: str) -> str:
            for p in label.split(", "):
                if p.startswith(feat + "="):
                    return p[len(feat) + 1 :]
            return "?"

        compound = [
            (lbl, pe)
            for lbl, pe in zip(all_labels, clf.patterns_)
            if feature_a + "=" in lbl and feature_b + "=" in lbl
        ]

        fig = go.Figure()
        if not compound:
            fig.update_layout(
                **_LAYOUT_BASE,
                title=_title(title or f"2-D HUG Profile · {feature_a} × {feature_b}"),
                height=height or self._h,
            )
            fig.add_annotation(
                text=f"No compound patterns for '{feature_a}' × '{feature_b}'.\nTry L≥2 and lower G.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color=_T["muted"]),
            )
            return fig

        bins_a = sorted(set(_bin(lbl, feature_a) for lbl, _ in compound))
        bins_b = sorted(set(_bin(lbl, feature_b) for lbl, _ in compound))
        idx_a = {b: i for i, b in enumerate(bins_a)}
        idx_b = {b: i for i, b in enumerate(bins_b)}
        grid = np.zeros((len(bins_a), len(bins_b)))
        for lbl, pe in compound:
            ia = idx_a.get(_bin(lbl, feature_a))
            ib = idx_b.get(_bin(lbl, feature_b))
            if ia is not None and ib is not None:
                grid[ia, ib] += float(pe.utility)

        fig.add_trace(
            go.Heatmap(
                z=grid,
                x=bins_b,
                y=bins_a,
                colorscale="RdBu_r",
                zmid=0,
                colorbar=dict(
                    tickfont=dict(color=_T["muted"]),
                    title=dict(font=dict(color=_T["muted"]), text="aggregate utility"),
                ),
            )
        )

        fig.update_layout(
            **_LAYOUT_BASE,
            title=_title(title or f"2-D HUG Profile · {feature_a} × {feature_b}"),
            height=height or self._h,
        )
        fig.update_xaxes(
            title_text=feature_b,
            tickangle=-35,
            tickfont=dict(size=9, color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        fig.update_yaxes(
            title_text=feature_a,
            tickfont=dict(size=9, color=_T["muted"]),
            gridcolor=_T["grid"],
            linecolor=_T["border"],
        )
        return fig

    # ------------------------------------------------------------------
    # 12. Full HTML dashboard
    # ------------------------------------------------------------------

    def plot_dashboard(
        self,
        X: Any,
        dataset_name: str = "Dataset",
        feature_names_for_profile: list[str] | None = None,
        output_path: str | None = None,
    ) -> str:
        """Generate a self-contained multi-panel HTML dashboard.

        Produces performance overview, feature importance, utility-vs-IG,
        top patterns, pattern lengths, support distribution, feature coverage,
        and per-feature marginal bin profiles.

        Parameters
        ----------
        X : array-like or DataFrame
            Used for active-pattern coverage check.
        dataset_name : str
        feature_names_for_profile : list of str, optional
            Which features to include marginal bin profiles for.
            Defaults to all features that have singleton patterns.
        output_path : str, optional
            If given, writes the HTML to this path.

        Returns
        -------
        str  (HTML string)
        """
        clf = self._clf
        all_labels = clf.get_hug_features()

        # Find features with singleton patterns
        singleton_feats = sorted(
            {
                lbl.split(", ")[0].split("=")[0]
                for lbl, pe in zip(all_labels, clf.patterns_)
                if len(pe.items) == 1
            }
        )
        if feature_names_for_profile is None:
            feature_names_for_profile = singleton_feats[:12]  # limit for dashboard

        # Build all figures
        figs = [
            self.plot_feature_importance(top_n=10),
            self.plot_top_patterns(top_n=15),
            self.plot_utility_vs_ig(),
            self.plot_feature_coverage(top_n=10),
            self.plot_pattern_lengths(),
            self.plot_support_distribution(),
        ]

        # Per-feature bin profiles
        for fname in feature_names_for_profile:
            figs.append(self.plot_marginal_bin_profile(fname))
            figs.append(self.plot_feature_combinations(fname, top_n=15))

        # Assemble HTML — use plotly's own bundled JS so the file is self-contained
        # and works without a network connection (no CDN pin to go stale).
        import plotly.io as pio

        html_parts = [
            "<!DOCTYPE html><html><head>",
            "<meta charset='utf-8'>",
            f"<title>HUG-IML Dashboard — {dataset_name}</title>",
            "<style>",
            f"body{{background:{_T['bg']};color:{_T['text']};font-family:{_T['font_b']};margin:0;padding:16px;}}",
            f"h1{{font-family:{_T['font_t']};color:{_T['a1']};}}",
            ".grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;}}",
            ".card{{background:{panel};border:1px solid {border};border-radius:8px;overflow:hidden;}}".format(
                panel=_T["panel"], border=_T["border"]
            ),
            "</style></head><body>",
            f"<h1>HUG-IML Governance Dashboard — {dataset_name}</h1>",
            "<div class='grid-2'>",
        ]

        for i, fig in enumerate(figs):
            fig_html = pio.to_html(fig, full_html=False, include_plotlyjs=(i == 0))
            html_parts.append(f"<div class='card'>{fig_html}</div>")
            if (i + 1) % 2 == 0 and i + 1 < len(figs):
                html_parts.append("</div><div class='grid-2'>")

        html_parts += ["</div></body></html>"]
        html = "\n".join(html_parts)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(html)

        return html

    # ------------------------------------------------------------------

    def _validate_fitted(self) -> None:
        if not hasattr(self._clf, "patterns_"):
            raise RuntimeError("Classifier must be fitted before creating a HUGPlotter.")

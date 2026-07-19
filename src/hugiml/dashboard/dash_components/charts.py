"""Chart utilities: Plotly helpers and matplotlib→base64 for the Dash dashboard."""

from __future__ import annotations

import base64
import io

import numpy as np
import plotly.graph_objects as go

_P = {
    "blue": "#2563eb",
    "cyan": "#06b6d4",
    "purple": "#534AB7",
    "green": "#1D9E75",
    "red": "#E24B4A",
    "amber": "#EF9F27",
    "grey": "#888780",
    "lav": "#AFA9EC",
    "teal": "#0d9488",
}
FAMILY_COLOURS = {
    "original": "#2563eb",
    "pattern": "#534AB7",
    "augmented_pair": "#1D9E75",
    "rpte_rule": "#C77D2E",
    "unknown": "#888780",
}
_BL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12),
    margin=dict(t=36, b=36, l=60, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
)


def _base(fig, h=340):
    fig.update_layout(**_BL, height=h)
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.10)", zerolinecolor="rgba(128,128,128,0.16)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.10)", zerolinecolor="rgba(128,128,128,0.16)")
    return fig


def fig_to_uri(fig):
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    except Exception:
        fig.savefig(buf, format="png", dpi=110)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def empty_fig(msg="No data"):
    f = go.Figure()
    f.add_annotation(
        text=msg,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=13, color="#888"),
    )
    f.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=140,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return f


def bar_h(vals, labs, title="", color=None, xlabel="", h=340, vlines=None):
    cs = color if isinstance(color, list) else [color or _P["blue"]] * len(vals)
    f = go.Figure(
        go.Bar(
            x=list(vals),
            y=[str(label) for label in labs],
            orientation="h",
            marker_color=cs,
            hovertemplate="%{y}: %{x:.4g}<extra></extra>",
        )
    )
    if vlines:
        for vx, vc in vlines:
            f.add_vline(x=vx, line_dash="dash", line_color=vc, line_width=1.2, opacity=0.7)
    f.update_yaxes(autorange="reversed")
    f.update_layout(title=dict(text=title, font=dict(size=13)), xaxis_title=xlabel)
    return _base(f, max(h, len(vals) * 28 + 60))


def bar_v(x, y, title="", color=None, xlabel="", ylabel="", h=260):
    f = go.Figure(
        go.Bar(
            x=list(x),
            y=list(y),
            marker_color=color or _P["blue"],
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    f.update_layout(
        title=dict(text=title, font=dict(size=13)), xaxis_title=xlabel, yaxis_title=ylabel
    )
    return _base(f, h)


def line_chart(df, xcol, ycols, title="", h=300, markers=True):
    f = go.Figure()
    cs = [_P["blue"], _P["cyan"], _P["purple"], _P["green"], _P["amber"]]
    for i, c in enumerate(ycols):
        if c not in df.columns:
            continue
        f.add_trace(
            go.Scatter(
                x=df[xcol],
                y=df[c],
                name=c,
                mode="lines+markers" if markers else "lines",
                line=dict(color=cs[i % len(cs)], width=2),
                marker=dict(size=6),
            )
        )
    f.update_layout(title=dict(text=title, font=dict(size=13)))
    return _base(f, h)


def separation_hist(proba, y_true, h=280):
    p = np.clip(np.asarray(proba, dtype=float), 0, 1)
    y = np.asarray(y_true, dtype=int)
    bins = np.linspace(0, 1, 21)
    pos, _ = np.histogram(p[y == 1], bins=bins)
    neg, _ = np.histogram(p[y == 0], bins=bins)
    c = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    f = go.Figure()
    f.add_trace(go.Bar(x=c, y=neg.tolist(), name="y=0", marker_color=_P["red"], opacity=0.75))
    f.add_trace(go.Bar(x=c, y=pos.tolist(), name="y=1", marker_color=_P["green"], opacity=0.75))
    f.update_layout(
        barmode="overlay",
        xaxis_title="Predicted probability",
        yaxis_title="Count",
        title=dict(text="Score distribution by class", font=dict(size=13)),
    )
    return _base(f, h)


def calibration_chart(predicted, actual, n=None, h=280):
    f = go.Figure()
    f.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color=_P["grey"], width=1),
            name="Perfect",
        )
    )
    sz = [min(max(int(v), 6), 22) for v in n] if n else [9] * len(list(predicted))
    f.add_trace(
        go.Scatter(
            x=list(predicted),
            y=list(actual),
            mode="markers",
            marker=dict(size=sz, color=_P["purple"], opacity=0.85),
            name="Model",
        )
    )
    f.update_layout(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        xaxis_title="Mean predicted prob.",
        yaxis_title="Observed rate",
        title=dict(text="Calibration", font=dict(size=13)),
    )
    return _base(f, h)


def psi_bar(features, psi_values, h=None):
    cs = [_P["red"] if v > 0.25 else _P["amber"] if v > 0.10 else _P["green"] for v in psi_values]
    return bar_h(
        psi_values,
        features,
        title="PSI by feature",
        color=cs,
        xlabel="PSI",
        h=h or max(220, len(features) * 28 + 60),
        vlines=[(0.10, _P["amber"]), (0.25, _P["red"])],
    )


def coef_waterfall(names, coefs, family_colors=None, title="Signed feature coefficients", h=None):
    cs = family_colors or [_P["blue"]] * len(names)
    return bar_h(
        coefs,
        names,
        title=title,
        color=cs,
        xlabel="Coefficient (log-odds)",
        h=h or max(220, len(names) * 28 + 60),
        vlines=[(0, _P["grey"])],
    )


def timing_bar(stages, times_ms, h=None):
    mx = max(times_ms) if times_ms else 1
    cs = [_P["purple"] if t == mx else _P["lav"] for t in times_ms]
    return bar_h(
        times_ms,
        stages,
        title="Fit stage timing",
        color=cs,
        xlabel="ms",
        h=h or max(180, len(stages) * 28 + 60),
    )


def group_rates(labels, rates, max_rate, h=None):
    cs = [
        _P["red"] if max_rate > 0 and abs(r - max_rate) / max_rate > 0.3 else _P["blue"]
        for r in rates
    ]
    return bar_h(
        rates,
        labels,
        title="Predicted positive rate by group",
        color=cs,
        xlabel="Rate",
        h=h or max(200, len(labels) * 28 + 60),
    )


def co_miss_bar(pair_labels, r_vals, h=None):
    cs = [_P["red"] if r > 0.5 else _P["amber"] for r in r_vals]
    return bar_h(
        r_vals,
        pair_labels,
        title="Co-missingness pairs",
        color=cs,
        xlabel="Correlation (r)",
        vlines=[(0.3, _P["grey"])],
        h=h or max(180, len(pair_labels) * 28 + 60),
    )


def coverage_line(x, y, title="Pattern accumulation curve", h=280):
    f = go.Figure()
    f.add_trace(
        go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines",
            fill="tozeroy",
            line=dict(color=_P["purple"], width=2),
            fillcolor="rgba(83,74,183,0.12)",
        )
    )
    f.update_yaxes(tickformat=".0%", range=[0, 1.05])
    f.update_layout(
        title=dict(text=title, font=dict(size=13)), xaxis_title="Patterns", yaxis_title="Coverage"
    )
    return _base(f, h)


def roc_pr_curves(thresholds, recall, precision, specificity, h=300):
    f = go.Figure()
    f.add_trace(
        go.Scatter(x=thresholds, y=recall, name="Recall", line=dict(color=_P["blue"], width=2))
    )
    f.add_trace(
        go.Scatter(
            x=thresholds, y=precision, name="Precision", line=dict(color=_P["green"], width=2)
        )
    )
    f.add_trace(
        go.Scatter(
            x=thresholds, y=specificity, name="Specificity", line=dict(color=_P["purple"], width=2)
        )
    )
    f.update_layout(
        xaxis_title="Threshold",
        yaxis_title="Score",
        title=dict(text="Threshold sweep", font=dict(size=13)),
    )
    return _base(f, h)

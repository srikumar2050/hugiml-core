"""Reusable SVG chart builders for the HUGIML NLP example pages.

These are plain functions over numeric data -- no Jinja, no external chart
library -- so the same layout math that was hand-verified for the original
lending/card-default mockups (linear bar scaling, rotated waterfall labels
spaced wide enough not to collide, donut-gauge dasharray from a fraction) is
now driven by real numbers instead of being re-derived by hand each time.

Colors are the fixed palette used across the example pages:
    ink    = #1C2420   amber = #B5651D   teal = #2C6E63
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from dataclasses import dataclass

INK = "#1C2420"
INK_SOFT = "#5B6358"
RULE = "#C7CFC0"
TEAL = "#2C6E63"
AMBER = "#B5651D"


def _esc(value: object) -> str:
    return html.escape(str(value))


# --------------------------------------------------------------------------- #
# Candidate comparison bar chart (e.g. "8 / 12 / 16 rules" AUC comparison)
# --------------------------------------------------------------------------- #
def candidate_bar_chart(
    candidates: Sequence[tuple[str, float]],
    *,
    chosen_index: int,
    y_min: float,
    y_max: float,
    value_fmt: str = "{:.3f}",
    width: int = 360,
    height: int = 150,
) -> str:
    n = len(candidates)
    bar_w = 56
    gap = (width - 60 - n * bar_w) / max(n - 1, 1) if n > 1 else 0
    baseline_y = height - 10
    plot_h = height - 30
    bars = []
    for i, (label, value) in enumerate(candidates):
        x = 60 + i * (bar_w + gap)
        frac = (value - y_min) / (y_max - y_min) if y_max > y_min else 0.0
        frac = min(max(frac, 0.0), 1.0)
        bar_h = frac * plot_h
        top = baseline_y - bar_h
        chosen = i == chosen_index
        color = AMBER if chosen else TEAL
        weight = ' font-weight="600"' if chosen else ""
        fill_color = INK if chosen else INK_SOFT
        bars.append(
            f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}"/>'
            f'<text x="{x + bar_w / 2:.1f}" y="{max(top - 6, 12):.1f}" font-size="11" '
            f'fill="{INK}" text-anchor="middle"{weight}>{_esc(value_fmt.format(value))}</text>'
            f'<text x="{x + bar_w / 2:.1f}" y="{height + 4:.1f}" font-size="10.5" fill="{fill_color}" '
            f'text-anchor="middle"{weight}>{_esc(label)}</text>'
        )
    grid = (
        f'<line x1="30" y1="{baseline_y}" x2="{width - 20}" y2="{baseline_y}" stroke="{INK_SOFT}"/>'
    )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height + 20}" role="img" '
        f'aria-label="Candidate comparison">{grid}{"".join(bars)}</svg>'
    )


# --------------------------------------------------------------------------- #
# Grouped bar chart (e.g. recall/precision per tuning candidate, before/after)
# --------------------------------------------------------------------------- #
@dataclass
class GroupedSeries:
    name: str
    color: str
    values: Sequence[float]


def grouped_bar_chart(
    group_labels: Sequence[str],
    series: Sequence[GroupedSeries],
    *,
    chosen_index: int | None,
    y_min: float,
    y_max: float,
    width: int = 400,
    height: int = 200,
) -> tuple[str, str]:
    """Returns (svg, legend_html)."""

    n = len(group_labels)
    bar_w = 20
    inner_gap = 2
    group_w = len(series) * bar_w + (len(series) - 1) * inner_gap
    group_gap = (width - 70 - n * group_w) / max(n - 1, 1) if n > 1 else 0
    baseline_y = height - 20
    plot_h = height - 40
    bars = []
    for gi, label in enumerate(group_labels):
        gx = 50 + gi * (group_w + group_gap)
        chosen = chosen_index is not None and gi == chosen_index
        stroke = f' stroke="{INK}" stroke-width="1.5"' if chosen else ""
        weight = ' font-weight="600"' if chosen else ""
        for si, s in enumerate(series):
            value = s.values[gi]
            frac = (value - y_min) / (y_max - y_min) if y_max > y_min else 0.0
            frac = min(max(frac, 0.0), 1.0)
            bar_h = frac * plot_h
            x = gx + si * (bar_w + inner_gap)
            top = baseline_y - bar_h
            bars.append(
                f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_w}" height="{bar_h:.1f}" '
                f'fill="{s.color}"{stroke}/>'
            )
        label_color = INK if chosen else INK_SOFT
        bars.append(
            f'<text x="{gx + group_w / 2:.1f}" y="{height - 2}" font-size="10" '
            f'fill="{label_color}" text-anchor="middle"{weight}>{_esc(label)}</text>'
        )
    grid = f'<line x1="30" y1="{baseline_y}" x2="{width - 20}" y2="{baseline_y}" stroke="{INK_SOFT}"/>'
    svg = (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Grouped comparison">{grid}{"".join(bars)}</svg>'
    )
    legend = '<div class="chart-legend">' + "".join(
        f'<span><span class="swatch" style="background:{s.color}"></span>{_esc(s.name)}</span>'
        for s in series
    ) + "</div>"
    return svg, legend


# --------------------------------------------------------------------------- #
# Tornado / diverging bar chart (pattern coefficients)
# --------------------------------------------------------------------------- #
def tornado_chart(rows: Sequence[tuple[str, float]], *, width: int = 440) -> str:
    if not rows:
        return ""
    max_abs = max(abs(v) for _, v in rows) or 1.0
    half = (width - 240) / 2
    zero_x = 230
    row_h = 22
    height = 6 + row_h * len(rows) + 16
    parts = [
        f'<line x1="{zero_x}" y1="6" x2="{zero_x}" y2="{height - 16}" stroke="{INK_SOFT}"/>',
        f'<text x="{zero_x}" y="{height - 4}" font-size="10" fill="{INK_SOFT}" text-anchor="middle">0</text>',
    ]
    for i, (label, value) in enumerate(rows):
        y = 8 + i * row_h
        w = abs(value) / max_abs * half
        color = AMBER if value >= 0 else TEAL
        x = zero_x if value >= 0 else zero_x - w
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="14" fill="{color}"/>')
        parts.append(f'<text x="6" y="{y + 10}" font-size="10" fill="{INK}">{_esc(label)}</text>')
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Pattern coefficients">{"".join(parts)}</svg>'
    )


# --------------------------------------------------------------------------- #
# Waterfall chart + probability gauge (per-instance interpretation)
# --------------------------------------------------------------------------- #
def waterfall_with_gauge(
    *,
    baseline_label: str,
    baseline_value: float,
    steps: Sequence[tuple[str, float]],
    final_label: str,
    probability: float,
    gauge_label: str,
    width: int | None = None,
    pitch: int = 46,
    bar_width: int = 26,
) -> str:
    """``steps`` are (label, delta) pairs applied in order after the baseline."""

    n_cols = 1 + len(steps) + 1  # baseline + deltas + total
    width = width or max(360, 70 + n_cols * pitch + 140)
    start_x = 30
    top_margin = 20
    plot_h = 150

    values = [baseline_value]
    running = baseline_value
    for _, delta in steps:
        running += delta
        values.append(running)
    final_value = running

    v_min = min(values + [0.0])
    v_max = max(values + [0.0])
    scale = plot_h / (v_max - v_min) if v_max > v_min else 1.0

    def y_of(v: float) -> float:
        return top_margin + (v_max - v) * scale

    zero_y = y_of(0.0)
    label_anchor_y = top_margin + plot_h + 26
    height = label_anchor_y + 70

    parts = [
        f'<line x1="20" y1="{top_margin}" x2="20" y2="{top_margin + plot_h}" stroke="{RULE}"/>',
        f'<line x1="20" y1="{zero_y:.1f}" x2="{start_x + n_cols * pitch}" y2="{zero_y:.1f}" '
        f'stroke="{RULE}" stroke-dasharray="2 3"/>',
        f'<text x="4" y="{zero_y + 3:.1f}" font-size="9" fill="{INK_SOFT}">0</text>',
    ]

    def add_bar(cx: float, start_v: float, end_v: float, label: str, value_text: str | None, color: str) -> None:
        y0, y1 = y_of(start_v), y_of(end_v)
        top, bot = min(y0, y1), max(y0, y1)
        x = cx - bar_width / 2
        parts.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_width}" height="{max(bot - top, 0.6):.1f}" fill="{color}"/>')
        if value_text:
            text_y = top - 4 if top > top_margin + 10 else bot + 12
            parts.append(
                f'<text x="{cx:.1f}" y="{text_y:.1f}" font-size="8.5" fill="{INK}" text-anchor="middle">{_esc(value_text)}</text>'
            )
        parts.append(
            f'<text x="{cx:.1f}" y="{label_anchor_y:.1f}" font-size="8.5" fill="{INK_SOFT}" '
            f'text-anchor="end" transform="rotate(-45 {cx:.1f} {label_anchor_y:.1f})">{_esc(label)}</text>'
        )

    col = 0
    cx = start_x + bar_width / 2 + col * pitch
    add_bar(cx, 0.0, baseline_value, baseline_label, f"{baseline_value:+.3f}", AMBER if baseline_value >= 0 else TEAL)
    running = baseline_value
    for label, delta in steps:
        col += 1
        cx = start_x + bar_width / 2 + col * pitch
        new_running = running + delta
        color = AMBER if delta >= 0 else TEAL
        add_bar(cx, running, new_running, label, f"{delta:+.3f}", color)
        running = new_running
    col += 1
    cx = start_x + bar_width / 2 + col * pitch + 10
    total_w = bar_width + 10
    y0, y1 = y_of(0.0), y_of(final_value)
    top, bot = min(y0, y1), max(y0, y1)
    parts.append(f'<rect x="{cx - total_w / 2:.1f}" y="{top:.1f}" width="{total_w}" height="{max(bot - top, 0.6):.1f}" fill="{INK}"/>')
    mid_y = (top + bot) / 2 + 3
    parts.append(f'<text x="{cx:.1f}" y="{mid_y:.1f}" font-size="9.5" fill="#ECEFE6" text-anchor="middle" font-weight="600">{final_value:+.3f}</text>')
    parts.append(
        f'<text x="{cx:.1f}" y="{label_anchor_y:.1f}" font-size="9.5" fill="{INK}" font-weight="600" '
        f'text-anchor="end" transform="rotate(-45 {cx:.1f} {label_anchor_y:.1f})">{_esc(final_label)}</text>'
    )

    gauge_cx = start_x + bar_width / 2 + (col + 2) * pitch + 30
    gauge_cy = top_margin + plot_h / 2
    r = 46
    circumference = 2 * math.pi * r
    filled = max(min(probability, 1.0), 0.0) * circumference
    remainder = circumference - filled
    pct_text = f"{probability * 100:.1f}%"
    parts.append(
        f'<g transform="translate({gauge_cx:.1f},{gauge_cy:.1f})">'
        f'<circle r="{r}" fill="none" stroke="{RULE}" stroke-width="10"/>'
        f'<circle r="{r}" fill="none" stroke="{AMBER}" stroke-width="10" '
        f'stroke-dasharray="{filled:.1f} {remainder:.1f}" transform="rotate(-90)"/>'
        f'<text y="-2" font-size="17" fill="{INK}" text-anchor="middle" font-weight="600">{pct_text}</text>'
        f'<text y="16" font-size="9.5" fill="{INK_SOFT}" text-anchor="middle">{_esc(gauge_label)}</text>'
        f"</g>"
    )

    full_width = gauge_cx + r + 20
    return (
        f'<svg class="chart" viewBox="0 0 {full_width:.0f} {height:.0f}" role="img" '
        f'aria-label="Score breakdown">{"".join(parts)}</svg>'
    )


# --------------------------------------------------------------------------- #
# Donut / gauge (class balance, simple fraction displays)
# --------------------------------------------------------------------------- #
def donut(
    fraction: float,
    *,
    top_text: str,
    bottom_text: str,
    cx: int = 60,
    cy: int = 65,
    r: int = 50,
) -> str:
    circumference = 2 * math.pi * r
    filled = max(min(fraction, 1.0), 0.0) * circumference
    remainder = circumference - filled
    return (
        f'<g transform="translate({cx},{cy})">'
        f'<circle r="{r}" fill="none" stroke="{RULE}" stroke-width="18"/>'
        f'<circle r="{r}" fill="none" stroke="{AMBER}" stroke-width="18" '
        f'stroke-dasharray="{filled:.1f} {remainder:.1f}" transform="rotate(-90)"/>'
        f'<text y="-4" font-size="20" font-weight="600" fill="{INK}" text-anchor="middle">{_esc(top_text)}</text>'
        f'<text y="14" font-size="9.5" fill="{INK_SOFT}" text-anchor="middle">{_esc(bottom_text)}</text>'
        f"</g>"
    )

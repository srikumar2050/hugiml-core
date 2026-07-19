"""DataTable factory for the Dash dashboard."""

from __future__ import annotations

import numpy as np
from dash import dash_table

from hugiml.dashboard.display import dataframe_for_display


def df_to_records(df):
    if df is None or df.empty:
        return [], []
    safe = dataframe_for_display(df).fillna("")
    recs = []
    for row in safe.itertuples(index=False):
        rec = {}
        for col, val in zip(safe.columns, row):
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                rec[col] = ""
            elif isinstance(val, (np.integer,)):
                rec[col] = int(val)
            elif isinstance(val, (np.floating,)):
                rec[col] = float(val)
            elif isinstance(val, (np.bool_,)):
                rec[col] = bool(val)
            else:
                rec[col] = val
        recs.append(rec)
    cols = [{"name": str(c), "id": str(c)} for c in safe.columns]
    return recs, cols


def make_table(df, tid="table", page_size=25, height="400px", sortable=True, filterable=False):
    if df is None or df.empty:
        data, columns = [], []
    else:
        data, columns = df_to_records(df.head(2000))
    kw = dict(
        id=tid,
        data=data,
        columns=columns,
        page_size=page_size,
        sort_action="native" if sortable else "none",
        filter_action="native" if filterable else "none",
        style_table={
            "overflowX": "auto",
            "overflowY": "auto",
            "borderRadius": "8px",
            "border": "1px solid var(--bs)",
            "backgroundColor": "var(--sf)",
        },
        style_header={
            "backgroundColor": "var(--cb)",
            "color": "var(--tx)",
            "fontWeight": "700",
            "fontSize": ".73rem",
            "letterSpacing": ".06em",
            "textTransform": "uppercase",
            "border": "1px solid var(--bs)",
            "whiteSpace": "normal",
        },
        style_cell={
            "fontFamily": "Inter, sans-serif",
            "fontSize": ".82rem",
            "padding": "7px 10px",
            "border": "1px solid var(--bs)",
            "textAlign": "left",
            "backgroundColor": "var(--sf)",
            "color": "var(--tx)",
            "maxWidth": "320px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "var(--sa)"},
            {
                "if": {"state": "selected"},
                "backgroundColor": "var(--cb)",
                "border": "1px solid var(--bd)",
            },
        ],
        tooltip_delay=0,
        tooltip_duration=None,
    )
    if height:
        kw["style_table"]["maxHeight"] = height
        kw["fixed_rows"] = {"headers": True}
    return dash_table.DataTable(**kw)

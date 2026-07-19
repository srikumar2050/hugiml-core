"""Shared UI primitives for all governance pages."""

from dash import html


def mc(label, value):
    v = str(value) if value is not None else "N/A"
    return html.Div(
        [html.Div(label, className="mc-l"), html.Div(v, className="mc-v")], className="mc"
    )


def sn(text):
    return html.Div([html.P(text)], className="sn")


def info(text):
    return html.Div(text, className="info-b mt-2 mb-2")


def warn(text):
    return html.Div(text, className="warn-b mt-2 mb-2")


def err(text):
    return html.Div(text, className="err-b mt-2 mb-2")

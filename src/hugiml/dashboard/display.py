"""Display helpers for Streamlit dashboard tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_str(v: Any) -> str:
    """Stringify a cell value without crashing on numpy arrays or ambiguous NA checks."""
    try:
        result = pd.isna(v)
        if isinstance(result, (bool, np.bool_)) and result:
            return ""
    except Exception:
        pass
    if isinstance(v, np.ndarray):
        return str(v.tolist())
    return str(v)


def dataframe_for_display(df: Any, stringify_mixed_object: bool = True) -> pd.DataFrame:
    """Return a Streamlit/Arrow-safe dataframe for display.

    Streamlit serializes dataframes through Arrow. Object columns that contain
    mixed Python types, such as integers/floats plus strings like
    ``original_plus_patterns``, can trigger ArrowInvalid because Arrow attempts
    to coerce the column to a numeric type. Audit/config tables often have this
    shape by design.

    This helper preserves numeric columns when they are truly numeric, and
    stringifies only object columns that contain mixed Python scalar types.
    Cells containing numpy arrays are safely converted to their list repr so that
    ``pd.isna()`` is never called on a multi-element array (which raises ValueError).
    """
    out = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    out = out.copy()

    # Streamlit/Arrow warns when dataframe column labels have mixed Python
    # types because they are coerced to strings for transport. These are
    # display-only frames, so normalize labels explicitly and deterministically.
    out.columns = [str(c) for c in out.columns]

    if not stringify_mixed_object:
        return out

    for col in out.columns:
        s = out[col]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            non_null = s.dropna()
            type_names = {type(v).__name__ for v in non_null.head(200).tolist()}
            # If object column is mixed, or contains any non-basic objects, make it text.
            if len(type_names) > 1 or any(
                t not in {"str", "int", "float", "bool", "int64", "float64", "bool_"}
                for t in type_names
            ):
                out[col] = s.map(_safe_str)
            elif type_names and next(iter(type_names)) == "str":
                out[col] = s.astype("string")
            # Single numeric-looking object columns are left alone only if pandas can safely convert.
            elif type_names:
                try:
                    converted = pd.to_numeric(s, errors="raise")
                    out[col] = converted
                except Exception:
                    out[col] = s.map(_safe_str)

    return out

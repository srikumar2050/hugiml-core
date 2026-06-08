"""Data loading and column-role helpers for the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_table(path: str | Path, sep: str | None = None, sheet_name: str | int | None = None) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, sep=sep or ",")
    if suffix == ".tsv":
        return pd.read_csv(path, sep=sep or "\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0 if sheet_name is None else sheet_name)
    raise ValueError(f"Unsupported data format: {suffix}")


def ensure_case_id(df: pd.DataFrame, id_column: str | None = None) -> tuple[pd.Series, str]:
    if id_column and id_column in df.columns:
        return df[id_column].astype(str), id_column
    generated = pd.Series([f"CASE-{i+1:06d}" for i in range(len(df))], name="_hugiml_case_id")
    return generated, "_hugiml_case_id"


def prepare_model_frame(
    df: pd.DataFrame,
    target: str,
    id_column: str | None = None,
    excluded_columns: list[str] | None = None,
    positive_label: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series, dict[str, Any]]:
    excluded_columns = excluded_columns or []
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found.")

    case_id, case_id_name = ensure_case_id(df, id_column)
    y_raw = df[target]
    valid_y = ~pd.isna(y_raw)
    df = df.loc[valid_y].reset_index(drop=True)
    case_id = case_id.loc[valid_y].reset_index(drop=True)
    y_raw = y_raw.loc[valid_y].reset_index(drop=True)

    drop_cols = [target]
    if id_column and id_column in df.columns:
        drop_cols.append(id_column)
    drop_cols.extend([c for c in excluded_columns if c in df.columns])
    X = df.drop(columns=list(dict.fromkeys(drop_cols))).copy()
    X.columns = [str(c) for c in X.columns]

    classes = list(pd.unique(y_raw))
    if len(classes) != 2:
        raise ValueError(f"Expected binary target; found {len(classes)} classes: {classes}")

    if positive_label is not None:
        y = (y_raw.astype(str).to_numpy() == str(positive_label)).astype(int)
        class_mapping = {str(positive_label): 1, "other": 0}
    else:
        enc = LabelEncoder()
        y = enc.fit_transform(y_raw.to_numpy()).astype(int)
        class_mapping = {str(cls): int(code) for code, cls in enumerate(enc.classes_)}

    meta = {
        "case_id_name": case_id_name,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_mapping": class_mapping,
        "positive_rate": float(np.mean(y)),
        "model_features": list(X.columns),
        "excluded_columns": excluded_columns,
        "id_column": id_column,
    }
    return X, y, case_id, meta

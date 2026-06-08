"""Lightweight dashboard schemas kept for compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnRoles:
    target: str | None = None
    id_column: str | None = None
    excluded_columns: list[str] = field(default_factory=list)
    sensitive_columns: list[str] = field(default_factory=list)


@dataclass
class DashboardState:
    mode: str = "demo"
    model: Any | None = None
    X: Any | None = None
    y: Any | None = None
    roles: ColumnRoles = field(default_factory=ColumnRoles)

"""HUGIML Causal Investigation Studio."""

from __future__ import annotations


def launch_dashboard() -> None:
    from hugiml.causal_dashboard.launcher import main

    main()


__all__ = ["launch_dashboard"]

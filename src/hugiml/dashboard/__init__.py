"""HUGIML Governance Studio dashboard."""

from __future__ import annotations


def launch_dashboard() -> None:
    from hugiml.dashboard.app import main

    main()

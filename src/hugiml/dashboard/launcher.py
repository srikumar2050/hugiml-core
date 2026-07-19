"""HUGIML Dashboard launcher for Dash or the lightweight Streamlit interface."""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(
        prog="hugiml-dashboard",
        description="HUGIML Governance Studio",
        add_help=True,
    )
    parser.add_argument(
        "--ui",
        choices=["dash", "light"],
        default="dash",
        help="'dash' (default) or 'light' (Streamlit)",
    )
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--no-open", action="store_true", default=False)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=2026)
    args, remaining = parser.parse_known_args()
    args._remaining = remaining
    return args


def _browser_url(host: str, port: int) -> str:
    display_host = "localhost" if host in {"127.0.0.1", "0.0.0.0", "::"} else host
    return f"http://{display_host}:{port}/"


def _open_browser_when_ready(host: str, port: int, url: str) -> None:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    for _ in range(80):
        try:
            with socket.create_connection((connect_host, port), timeout=0.25):
                break
        except OSError:
            time.sleep(0.20)
    if sys.platform.startswith("win"):
        try:
            os.startfile(url)  # type: ignore[attr-defined]
            return
        except OSError:
            pass
    try:
        webbrowser.open_new_tab(url)
    except Exception:
        pass


def main():
    args = _parse_args()
    if os.environ.get("HUGIML_UI", "").strip().lower() in {"streamlit", "light"}:
        args.ui = "light"

    if args.ui == "light":
        from streamlit.web import cli

        app_path = str(Path(__file__).resolve().parent / "app.py")
        dashboard_args = [
            f"--cv={args.cv}",
            f"--random-state={args.random_state}",
            *args._remaining,
        ]
        sys.argv = ["streamlit", "run", app_path]
        if dashboard_args:
            sys.argv += ["--", *dashboard_args]
        cli.main()
        return

    from hugiml.dashboard.dash_app import create_app

    app = create_app(args.debug)
    url = _browser_url(args.host, args.port)
    print(f"\n  HUGIML Governance Studio (Dash)\n  {url}\n  Ctrl+C to quit.\n")
    if not args.no_open:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(args.host, args.port, url),
            daemon=True,
        ).start()
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()

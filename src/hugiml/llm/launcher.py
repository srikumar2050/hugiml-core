"""Launcher for the Dash or lightweight Streamlit LLM Assistant."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


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


def launch(
    *,
    ui: str = "dash",
    host: str = "127.0.0.1",
    port: int | None = None,
    debug: bool = False,
    no_open: bool = False,
    headless: bool = False,
) -> int:
    """Launch the selected LLM Assistant interface."""

    selected = str(ui or "dash").strip().lower()
    if selected in {"light", "streamlit"}:
        from streamlit.web import cli

        app_path = str(Path(__file__).resolve().parent / "ui_app.py")
        streamlit_port = int(port or 8501)
        sys.argv = [
            "streamlit",
            "run",
            app_path,
            "--server.address",
            host,
            "--server.port",
            str(streamlit_port),
        ]
        if headless or no_open:
            sys.argv.extend(["--server.headless", "true"])
        cli.main()
        return 0

    from .dash_app import create_app

    dash_port = int(port or 8051)
    app = create_app(debug=debug)
    url = _browser_url(host, dash_port)
    print(f"\n  HUGIML LLM Assistant (Dash)\n  {url}\n  Ctrl+C to quit.\n")
    if not no_open:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(host, dash_port, url),
            daemon=True,
        ).start()
    app.run(host=host, port=dash_port, debug=debug, use_reloader=False)
    return 0

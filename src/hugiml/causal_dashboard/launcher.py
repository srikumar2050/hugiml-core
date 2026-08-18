"""Console launcher for ``hugiml-causal-dashboard``."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser


def _browser_url(host: str, port: int) -> str:
    display = "localhost" if host in {"127.0.0.1", "0.0.0.0", "::"} else host
    return f"http://{display}:{port}/"


def _open_when_ready(host: str, port: int, url: str) -> None:
    connect = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    for _ in range(80):
        try:
            with socket.create_connection((connect, port), timeout=0.25):
                break
        except OSError:
            time.sleep(0.2)
    if sys.platform.startswith("win"):
        try:
            os.startfile(url)
            return  # type: ignore[attr-defined]
        except OSError:
            pass
    try:
        webbrowser.open_new_tab(url)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hugiml-causal-dashboard", description="HUGIML Causal Investigation Studio"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8052)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args(argv)
    try:
        from .dash_app import create_app
    except ModuleNotFoundError as exc:
        if exc.name in {"dash", "dash_bootstrap_components"}:
            p.error(
                'Optional dashboard dependencies are missing. Install with: pip install "hugiml-core[causal-dashboard]"'
            )
        raise
    app = create_app(args.debug)
    url = _browser_url(args.host, args.port)
    print(f"\n  HUGIML Causal Investigation Studio\n  {url}\n  Ctrl+C to quit.\n")
    if not args.no_open:
        threading.Thread(
            target=_open_when_ready, args=(args.host, args.port, url), daemon=True
        ).start()
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
app.py
======

Application bootstrap for the Transaction Authentication Dashboard.

Responsibilities
----------------
1. Start the existing FastAPI backend.
2. Build the NiceGUI dashboard.
3. Launch the application.

No authentication logic lives here.
No business logic lives here.
No model loading lives here.
"""

from __future__ import annotations

import threading
import webbrowser
from time import sleep

import uvicorn
from nicegui import ui

from api.server import app as fastapi_app
from dashboard import AuthDashboard


BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8080


def start_backend() -> None:
    """Run the existing FastAPI authentication server."""
    uvicorn.run(
        fastapi_app,
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        log_level="info",
    )


def open_browser() -> None:
    """Open the dashboard once it is available."""
    sleep(2)
    webbrowser.open(f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")


def main() -> None:
    """Application entrypoint."""

    # Start FastAPI in the background
    backend_thread = threading.Thread(
        target=start_backend,
        daemon=True,
    )
    backend_thread.start()

    # Build NiceGUI dashboard
    dashboard = AuthDashboard(
        backend_url=f"http://{BACKEND_HOST}:{BACKEND_PORT}"
    )
    dashboard.build()

    # Open browser automatically
    threading.Thread(
        target=open_browser,
        daemon=True,
    ).start()

    # Launch NiceGUI
    ui.run(
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        title="Vehicle Authentication Command Center",
        dark=True,
        reload=False,
        show=False,
    )


if __name__ == "__main__":
    main()
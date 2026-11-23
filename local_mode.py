"""Local-mode detection for the website (home.py), shared by Pages/app_module7.py and run_local.py.

Local mode is enabled ONLY by ``python run_local.py``, which sets CSC8830_LOCAL_MODE=1 in the environment of the website process it
starts.  Nobody can set that variable on the public Streamlit Community Cloud deployment except its owner, and the visitor application
URL is never taken from the environment: it is always http://127.0.0.1:<port> with a port number (an invalid value falls back to 8640),
so a setting can never produce a link to another host.
"""
import os
import socket

LOCAL_MODE_ENV = "CSC8830_LOCAL_MODE"
VISITOR_PORT_ENV = "CSC8830_VISITOR_PORT"
LOOPBACK = "127.0.0.1"
DEFAULT_WEBSITE_PORT = 8501
DEFAULT_VISITOR_PORT = 8640


def is_local_mode():
    """True only when the launcher set CSC8830_LOCAL_MODE to exactly "1"."""
    return os.environ.get(LOCAL_MODE_ENV) == "1"


def visitor_port():
    raw = os.environ.get(VISITOR_PORT_ENV, "")
    return int(raw) if raw.isdigit() and 1024 <= int(raw) <= 65535 else DEFAULT_VISITOR_PORT


def visitor_url():
    return f"http://{LOOPBACK}:{visitor_port()}"


def port_in_use(port, host=LOOPBACK, timeout=0.5):
    """True when something accepts TCP connections on host:port."""
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def visitor_app_running():
    """The visitor application's server answers on the loopback port (checked from the website's own process, on the visitor's computer)."""
    return port_in_use(visitor_port())

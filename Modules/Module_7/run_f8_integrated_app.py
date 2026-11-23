"""Start the LOCAL F8.18 integrated calibration experiment (isolated; not linked from any page; nothing is deployed).

    F8_PASSCODE=<at least 8 characters> python Modules/Module_7/run_f8_integrated_app.py [port] [--enable-full-calibration --max-tree-rss-mb=<MB>]    ->  http://127.0.0.1:8640 (default)

The passcode is read from the environment of this process and is never printed.  Full calibration stays DISABLED unless ``--enable-full-calibration`` is given
(it only sets F8_ENABLE_FULL_CALIBRATION=1 for the child server).  Binds to 127.0.0.1 only; dark theme with the Module 7 red accent; no usage statistics.
Room video processing is not available in this phase.
F8.20a: ``--enable-full-calibration`` REQUIRES ``--max-tree-rss-mb=<256..65536>`` (use the ``=`` form).  The limit applies to the app process + the calibration worker and its children
(resident memory, measured with ``ps`` on macOS); the worker is stopped when it is exceeded, and a full run is refused when the limit cannot be measured (F8_REQUIRE_MEMORY_GUARD=1 is set).
There is deliberately no default: see f8_chunk_poc/README.md, "Choosing a memory limit on a Mac"."""
import os
import socket
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent / "visitor_workflow" / "app_f8_integrated.py"
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
RSS_RANGE = (256.0, 65536.0)


def parse_rss_limit(argv):
    """-> (limit_mb | None, problem | None).  Only the ``--max-tree-rss-mb=<number>`` form; a value outside RSS_RANGE or not a finite number is a problem."""
    vals = [a.split("=", 1)[1] for a in argv if a.startswith("--max-tree-rss-mb=")]
    if not vals:
        return None, None
    try:
        v = float(vals[-1])
    except ValueError:
        return None, "--max-tree-rss-mb must be a number of megabytes"
    if not (v == v and RSS_RANGE[0] <= v <= RSS_RANGE[1]):
        return None, f"--max-tree-rss-mb must be between {RSS_RANGE[0]:g} and {RSS_RANGE[1]:g}"
    return v, None
PORT = int(ARGS[0]) if ARGS else 8640

if __name__ == "__main__":
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0: raise SystemExit(f"port {PORT} is already in use; pass another port: python {Path(__file__).name} <port>")
    env = dict(os.environ)
    limit, problem = parse_rss_limit(sys.argv[1:])
    for k in ("F8_MAX_TREE_RSS_MB", "F8_REQUIRE_MEMORY_GUARD"):
        env.pop(k, None)                                                    # only the flags below decide; an inherited value never enables or weakens the guard
    if problem:
        raise SystemExit(problem)
    if "--enable-full-calibration" in sys.argv:
        if limit is None:
            raise SystemExit("--enable-full-calibration requires an explicit memory limit, e.g. --max-tree-rss-mb=3000 (see f8_chunk_poc/README.md, 'Choosing a memory limit on a Mac')")
        env["F8_ENABLE_FULL_CALIBRATION"] = "1"; env["F8_MAX_TREE_RSS_MB"] = f"{limit:g}"; env["F8_REQUIRE_MEMORY_GUARD"] = "1"
    else:
        env.pop("F8_ENABLE_FULL_CALIBRATION", None)                          # an inherited value must not silently enable full calibration
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP), "--server.address", "127.0.0.1", "--server.port", str(PORT), "--server.headless", "true", "--server.maxUploadSize", "10",
           "--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal", "--theme.base", "dark", "--theme.primaryColor", "#C9363E"]
    print(f"F8.18 integrated calibration experiment (starting): http://127.0.0.1:{PORT}  ·  full calibration {'ENABLED (memory limit ' + env['F8_MAX_TREE_RSS_MB'] + ' MB)' if env.get('F8_ENABLE_FULL_CALIBRATION') == '1' else 'disabled'}"
          f"  ·  passcode {'set' if len(env.get('F8_PASSCODE', '').strip()) >= 8 else 'NOT set (the app will stay locked)'}")
    raise SystemExit(subprocess.call(cmd, env=env))

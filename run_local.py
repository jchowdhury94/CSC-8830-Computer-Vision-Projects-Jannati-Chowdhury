"""Start the whole local experience with one command:

    python run_local.py [--website-port 8501] [--visitor-port 8640]

It starts two separate Streamlit applications, both bound to 127.0.0.1 only:

  * the main website (home.py)                                    -> http://127.0.0.1:8501
  * the visitor measurement application                           -> http://127.0.0.1:8640
    (Modules/Module_7/room_measurement_question_3/run_f8_integrated_app.py, started unchanged with its port argument)

Local mode is switched on only here, through CSC8830_LOCAL_MODE=1 in the website process's own environment (see local_mode.py); this
is what enables the "Run Your Experiment (Local Only)" card, which opens the visitor application in the SAME tab; the visitor application's Back and Home links return to the website the same way.  Ctrl+C (or SIGTERM) stops both
applications, including the streamlit processes the visitor launcher starts.  If a port is busy or an application fails to start,
nothing is left running and the reason is printed.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import local_mode

ROOT = Path(__file__).resolve().parent
HOME = ROOT / "home.py"
VISITOR_LAUNCHER = ROOT / "Modules" / "Module_7" / "room_measurement_question_3" / "run_f8_integrated_app.py"
STARTUP_TIMEOUT_S = 120.0
WINDOWS = os.name == "nt"


def parse_args(argv):
    p = argparse.ArgumentParser(prog="run_local.py", description="Start the website and the visitor application locally (127.0.0.1 only).")
    p.add_argument("--website-port", type=int, default=local_mode.DEFAULT_WEBSITE_PORT)
    p.add_argument("--visitor-port", type=int, default=local_mode.DEFAULT_VISITOR_PORT)
    args = p.parse_args(argv)
    for name in ("website_port", "visitor_port"):
        if not 1024 <= getattr(args, name) <= 65535:
            p.error(f"--{name.replace('_', '-')} must be between 1024 and 65535")
    if args.website_port == args.visitor_port:
        p.error("the website and the visitor application need different ports")
    return args


def check_ports(website_port, visitor_port):
    """-> list of problem messages (empty when both ports are free)."""
    problems = []
    for label, port in (("website", website_port), ("visitor application", visitor_port)):
        if local_mode.port_in_use(port):
            problems.append(f"Port {port} ({label}) is already in use. Close the program using it, or choose another port, e.g. "
                            f"python run_local.py --website-port {website_port + 1} --visitor-port {visitor_port + 1}")
    return problems


def website_command(port):
    return [sys.executable, "-m", "streamlit", "run", str(HOME), "--server.address", local_mode.LOOPBACK, "--server.port", str(port),
            "--server.headless", "true", "--browser.gatherUsageStats", "false"]


def visitor_command(port):
    return [sys.executable, str(VISITOR_LAUNCHER), str(port)]


def child_env(website_port, visitor_port):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CSC8830_")}   # an inherited value never decides the mode
    env[local_mode.LOCAL_MODE_ENV] = "1"
    env[local_mode.VISITOR_PORT_ENV] = str(visitor_port)
    env[local_mode.WEBSITE_PORT_ENV] = str(website_port)   # the visitor application links back to the website with it
    return env


def start(cmd, env):
    """Start a child in its own process group so that it and everything it starts can be stopped together."""
    if WINDOWS:
        return subprocess.Popen(cmd, env=env, cwd=ROOT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    return subprocess.Popen(cmd, env=env, cwd=ROOT, start_new_session=True)


def stop(proc, grace=8.0):
    """Stop a child and its descendants: SIGTERM to the process group, SIGKILL if it is still alive after `grace` seconds."""
    if proc is None:
        return
    try:
        if proc.poll() is None:
            if WINDOWS:
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(grace)
            except subprocess.TimeoutExpired:
                if not WINDOWS:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(grace)
        elif not WINDOWS:
            os.killpg(proc.pid, signal.SIGTERM)       # the launcher exited; make sure its streamlit child is gone too
    except (ProcessLookupError, PermissionError):
        pass


def wait_until_up(name, port, proc, timeout=STARTUP_TIMEOUT_S):
    """-> None when the server accepts connections, otherwise a message explaining why not."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            return f"The {name} stopped during start-up (exit code {code}); see the messages above."
        if local_mode.port_in_use(port):
            return None
        time.sleep(0.3)
    return f"The {name} did not start listening on port {port} within {timeout:.0f} seconds."


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    for path in (HOME, VISITOR_LAUNCHER):
        if not path.is_file():
            print(f"Missing file: {path}. Run this command from a complete copy of the repository.", file=sys.stderr)
            return 1
    problems = check_ports(args.website_port, args.visitor_port)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    env = child_env(args.website_port, args.visitor_port)
    visitor = website = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        print("Starting the visitor application ...", flush=True)
        visitor = start(visitor_command(args.visitor_port), env)
        problem = wait_until_up("visitor application", args.visitor_port, visitor)
        if problem is None:
            print("Starting the website ...", flush=True)
            website = start(website_command(args.website_port), env)
            problem = wait_until_up("website", args.website_port, website)
        if problem:
            print(problem, file=sys.stderr)
            return 1
        print(f"\nRunning locally (127.0.0.1 only):\n  Website:             http://{local_mode.LOOPBACK}:{args.website_port}\n"
              f"  Visitor application: http://{local_mode.LOOPBACK}:{args.visitor_port}\n"
              "Open the website, go to Module 7 > Run Your Own Experiment > Room Measurement Using Stereo Videos and press "
              "'Run Your Experiment (Local Only)'.\nPress Ctrl+C to stop both.", flush=True)
        while True:
            for name, proc in (("visitor application", visitor), ("website", website)):
                if proc.poll() is not None:
                    print(f"The {name} stopped unexpectedly (exit code {proc.returncode}); stopping everything.", file=sys.stderr)
                    return 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping ...", flush=True)
        return 0
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        stop(website)
        stop(visitor)


if __name__ == "__main__":
    raise SystemExit(main())

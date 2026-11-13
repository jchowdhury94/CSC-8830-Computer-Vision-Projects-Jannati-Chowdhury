"""Real-browser end-to-end test of the F8 PoC against a RUNNING local server (Playwright + Chromium).

    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 run_e2e_local.py <file> <out_json> [url] [server_pid] [--cancel-at N] [--timeout S]

Selects <file> in the component's file input exactly as a visitor would, waits for the result, compares the SHA-256 shown by the app with
the file's own (set F8_E2E_PASSCODE in the environment when the app is passcode-protected), and samples the server's RSS (via ps, if a pid is given). Writes a JSON record (no file content, no file name).
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = {sys.argv[i]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
args = [a for a in args if a not in opts.values()]
FILE, OUT = Path(args[0]), Path(args[1]); URL = args[2] if len(args) > 2 else "http://127.0.0.1:8741"; PID = int(args[3]) if len(args) > 3 else None
TIMEOUT = float(opts.get("--timeout", 900)); CANCEL_AT = int(opts["--cancel-at"]) if "--cancel-at" in opts else None
rss, stop = [], threading.Event()


def poll():
    while not stop.is_set() and PID:
        try:
            rss.append(int(subprocess.run(["ps", "-o", "rss=", "-p", str(PID)], capture_output=True, text=True).stdout.strip()) / 1024)
        except Exception:
            pass
        time.sleep(0.25)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


expected = sha_file(FILE); size = FILE.stat().st_size; result = dict(file_mb=round(size / 1048576, 2), expected_sha_prefix=expected[:12])
threading.Thread(target=poll, daemon=True).start()
with sync_playwright() as p:
    exe = os.environ.get("F8_CHROMIUM") or next(iter(sorted(glob.glob(os.path.expanduser("~/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-mac-arm64/chrome-headless-shell")))), None)
    try:
        b = p.chromium.launch()
    except Exception:
        b = p.chromium.launch(executable_path=exe)         # fall back to whatever Chromium build is already installed (no download)
    pg = b.new_page(viewport=dict(width=1100, height=1000)); errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None); pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL)
    code = os.environ.get("F8_E2E_PASSCODE")                    # the access gate: read from the environment, never written to the result file
    if code:
        pg.get_by_label("Passcode").fill(code); pg.get_by_role("button", name="Unlock").click()
    pg.wait_for_selector("iframe", timeout=60000)
    fr = pg.frame_locator("iframe").first; fr.locator("#file").wait_for(timeout=30000)
    time.sleep(1.0); rss_before = max(rss) if rss else None
    t0 = time.time(); fr.locator("#file").set_input_files(str(FILE))
    cancelled = False
    while time.time() - t0 < TIMEOUT:
        st = fr.locator("#status").inner_text()
        if CANCEL_AT is not None and not cancelled and "Uploading" in st:
            n = int(st.split("…")[1].strip().split()[0])
            if n >= CANCEL_AT:
                fr.locator("#cancel").click(); cancelled = True
        if st.startswith("Uploaded and verified") or st.startswith("Upload cancelled") or fr.locator("#retry").is_visible():
            break
        time.sleep(0.1)
    result.update(wall_s=round(time.time() - t0, 2), final_status=fr.locator("#status").inner_text(), stats_line=fr.locator("#stats").inner_text())
    if "verified" in result["final_status"]:
        result["sha_matches"] = expected[:12] in result["final_status"]
        pg.get_by_text("Measurements").first.wait_for(timeout=60000); time.sleep(1.0)
    result["page_has_measurements"] = pg.get_by_text("Measurements").count() > 0
    result["console_errors"] = errs[:10]
    pg.screenshot(path=str(OUT.with_suffix(".png")), full_page=True)
    result["cancelled_by_test"] = cancelled
    if cancelled:
        time.sleep(1.0)
    b.close()
stop.set(); result.update(server_rss_before_mb=None if rss_before is None else round(rss_before, 1), server_rss_peak_mb=round(max(rss), 1) if rss else None)
OUT.write_text(json.dumps(result, indent=1)); print(json.dumps(result, indent=1))

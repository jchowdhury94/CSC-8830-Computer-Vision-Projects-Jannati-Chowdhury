"""Real-browser end-to-end test of the dual-video calibration page against a RUNNING local F8 server (Playwright + Chromium). SYNTHETIC videos only.

    F8_E2E_PASSCODE=<passcode> python3 run_e2e_dual.py <left.mp4> <right.mp4> <other_left.mp4> <out.json> [url] [server_pid]

Steps (each recorded): log in -> choose 'Dual-video calibration' -> upload LEFT -> try the SAME file as RIGHT (must be refused) -> upload RIGHT ->
replace LEFT with a different file (RIGHT must stay) -> put the original LEFT back -> Run benchmark (default) -> wait for the result -> check both videos were KEPT.
F8_E2E_MODE=full instead ticks the confirmation and presses 'Run full calibration' (the server must run with F8_ENABLE_FULL_CALIBRATION=1) and checks both videos were deleted.
The passcode comes from the environment and is never written to the result."""
import glob
import hashlib
import json
import os
import re
import sys
import threading
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

LEFT, RIGHT, OTHER, OUT = sys.argv[1:5]; URL = sys.argv[5] if len(sys.argv) > 5 else "http://127.0.0.1:8741"; PID = int(sys.argv[6]) if len(sys.argv) > 6 else None
FULL = os.environ.get("F8_E2E_MODE") == "full"
steps, rss, stop = [], [], threading.Event()


def poll():
    while not stop.is_set() and PID:
        try: rss.append(int(subprocess.run(["ps", "-o", "rss=", "-p", str(PID)], capture_output=True, text=True).stdout.strip()) / 1024)
        except Exception: pass
        time.sleep(0.5)


def step(name, ok, **kw):
    steps.append(dict(step=name, ok=bool(ok), **kw)); print(("PASS " if ok else "FAIL ") + name, kw if kw else "", flush=True)


def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


threading.Thread(target=poll, daemon=True).start()
with sync_playwright() as p:
    exe = next(iter(sorted(glob.glob(os.path.expanduser("~/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-mac-arm64/chrome-headless-shell")))), None)
    try: b = p.chromium.launch()
    except Exception: b = p.chromium.launch(executable_path=exe)
    pg = b.new_page(viewport=dict(width=1250, height=1500)); errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None); pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL); pg.get_by_label("Passcode").fill(os.environ["F8_E2E_PASSCODE"]); pg.get_by_role("button", name="Unlock").click()
    pg.get_by_text("Dual-video calibration").first.click(); pg.wait_for_selector("iframe", timeout=60000); pg.wait_for_function("document.querySelectorAll('iframe').length === 2", timeout=30000)
    frames = [pg.frame_locator("iframe").nth(0), pg.frame_locator("iframe").nth(1)]
    for f in frames: f.locator("#file").wait_for(timeout=30000)
    step("dual page shows a LEFT and a RIGHT uploader", pg.locator("iframe").count() == 2 and frames[0].locator("#label").inner_text().startswith("LEFT") and frames[1].locator("#label").inner_text().startswith("RIGHT"))

    def upload(i, path, expect_ok=True, timeout=120):
        f = frames[i]; f.locator("#file").set_input_files(str(path)); t = time.time()
        while time.time() - t < timeout:
            s = f.locator("#status").inner_text()
            if s.startswith("Uploaded and verified"): return True, s
            if f.locator("#retry").is_visible(): return False, s
            time.sleep(0.1)
        return False, "timeout"

    def caption(slot): 
        m = re.search(rf"{slot}: [^\n]*", pg.inner_text("body")); return m.group(0) if m else ""

    ok, s = upload(0, LEFT); pg.wait_for_timeout(1500)
    step("LEFT uploaded and verified", ok and sha(LEFT)[:12] in s and "✅" in caption("LEFT"), caption=caption("LEFT")[:80])
    ok, s = upload(1, LEFT); pg.wait_for_timeout(1000)
    step("the SAME file as RIGHT is refused", (not ok) and "same file" in s, message=s[:110])
    step("LEFT is still there after the refusal", "✅" in caption("LEFT"))
    ok, s = upload(1, RIGHT); pg.wait_for_timeout(1500)
    step("RIGHT uploaded and verified (LEFT kept)", ok and sha(RIGHT)[:12] in s and "✅" in caption("LEFT") and "✅" in caption("RIGHT"))
    left_before = caption("LEFT"); right_before = caption("RIGHT")
    ok, s = upload(0, OTHER); pg.wait_for_timeout(1500)
    step("replacing LEFT keeps RIGHT", ok and caption("RIGHT") == right_before and caption("LEFT") != left_before and sha(OTHER)[:12] in caption("LEFT"), left=caption("LEFT")[:70])
    ok, s = upload(0, LEFT); pg.wait_for_timeout(1500)
    step("original LEFT restored, RIGHT unchanged", ok and caption("LEFT") == left_before and caption("RIGHT") == right_before)
    if FULL:
        pg.get_by_text("I understand a full calibration").click(); run = pg.get_by_role("button", name="Run full calibration")
    else:
        run = pg.get_by_role("button", name="Run benchmark")
    t1 = time.time()
    while not run.is_enabled() and time.time() - t1 < 10: time.sleep(0.2)                       # the checkbox needs a rerun before the button unlocks
    step("Run is enabled with both videos", run.is_enabled())
    if not FULL: step("no full-calibration button unless the server enables it", pg.get_by_role("button", name="Run full calibration").count() == 0)
    t0 = time.time(); run.click(); done = None; stages = set(); ok_text = "Calibration finished" if FULL else "Benchmark completed"
    while time.time() - t0 < 300:
        txt = pg.inner_text("body")
        for st in ("validate", "verify", "detect left", "detect right", "solve"):
            if st in txt: stages.add(st)
        if ok_text in txt or re.search(r"\n(Benchmark )?(rejected|failed|timeout|cancelled|worker_crashed|memory_limit)", txt): done = txt; break
        time.sleep(0.5)
    wall = round(time.time() - t0, 1); done = done or pg.inner_text("body")
    step("run finished", ok_text in done, wall_s=wall, stages_seen=sorted(stages), headline=(re.search(rf"{ok_text}[^\n]*", done) or [""])[0][:90])
    if FULL:
        step("both videos were deleted afterwards", "Cleanup: videos deleted = True" in done and "no video yet" in caption("LEFT") + caption("RIGHT") + " no video yet", cleanup=(re.search(r"Cleanup:[^\n]*", done) or [""])[0])
    else:
        step("benchmark table shows frames, detections and integrity", all(k in done for k in ("Frames processed", "checkerboard detections", "SHA-256 verified before use", "Container memory peak")))
        step("both videos were KEPT after the benchmark", "Cleanup: videos deleted = False" in done and "✅" in caption("LEFT") and "✅" in caption("RIGHT") and "in use by the calibration" not in caption("LEFT"), cleanup=(re.search(r"Cleanup:[^\n]*", done) or [""])[0])
    step("no console errors", not errs, errors=errs[:3])
    pg.screenshot(path=str(Path(OUT).with_suffix(".png")), full_page=True); b.close()
stop.set()
res = dict(steps=steps, all_passed=all(s["ok"] for s in steps), server_rss_peak_mb=round(max(rss), 1) if rss else None, server_rss_min_mb=round(min(rss), 1) if rss else None)
Path(OUT).write_text(json.dumps(res, indent=1)); print("ALL PASSED" if res["all_passed"] else "SOME STEPS FAILED", "| server (Streamlit) RSS min/peak MB:", res["server_rss_min_mb"], res["server_rss_peak_mb"])
sys.exit(0 if res["all_passed"] else 1)

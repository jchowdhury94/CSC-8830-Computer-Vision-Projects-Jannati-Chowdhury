"""Stage F6 real-browser walk-through (LOCAL).  Drives the running F6 app (default http://127.0.0.1:8630) through every step with real clicks, using the
original-room fixture (read-only paths).  Needs Playwright:  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 browser_walkthrough_f6.py <out_dir> [url]

The 'visitor decisions' below were made by LOOKING at the candidate overlays (see the F6 report); they are not read from any Phase 1 file by this script.
Nothing is bypassed: every confirmation is a click on 'Confirm this boundary'; the manual correction is two real clicks on the image + 'Refine boundary'."""
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1]); URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8630"; SH = OUT / "screenshots"; SH.mkdir(parents=True, exist_ok=True)
W, H = 2160, 3840
# visitor decisions per boundary number (1..6): ("pick", rank) or ("manual", [[x, y], [x, y]])  (manual points are full-resolution distorted-frame pixels)
DECISIONS = {1: ("manual", [[509, 450], [510, 1450]]), 2: ("pick", 1), 3: ("pick", 1), 4: ("pick", 3), 5: ("pick", 3), 6: ("pick", 2)}
log = []; rss = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)


def server_pid():
    out = subprocess.run(["lsof", "-tiTCP:" + URL.rsplit(":", 1)[1], "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split(); return int(out[0]) if out else None


def poll_rss(pid, stop):
    while not stop.is_set():
        try: rss.append((time.time(), int(subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()) / 1024))
        except Exception: pass
        time.sleep(0.5)


with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True); pg = b.new_page(viewport={"width": 1500, "height": 1500}); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e))); pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    stop = threading.Event(); pid = server_pid(); th = threading.Thread(target=poll_rss, args=(pid, stop), daemon=True); th.start(); say("server pid", pid)

    def idle(t=0.8, timeout=300):
        pg.wait_for_timeout(int(t * 1000)); t0 = time.time()
        while time.time() - t0 < timeout:
            if pg.locator('[data-testid="stStatusWidget"]').count() == 0: break
            pg.wait_for_timeout(300)
        pg.wait_for_timeout(500)

    def shot(name, full=True): pg.screenshot(path=str(SH / f"{name}.png"), full_page=full); say("shot", name)
    def btn(name, exact=True): return pg.get_by_role("button", name=name, exact=exact).first
    def expand(title): pg.get_by_text(title).first.click(); pg.wait_for_timeout(400)
    def frame_img():
        im = pg.frame_locator('iframe[title*="streamlit_image_coordinates"]').first.locator("img").first; im.wait_for(timeout=30000); return im
    def click_full(x, y):
        im = frame_img(); bb = im.bounding_box(); u = (x + 0.5) / W * bb["width"]; v = (y + 0.5) / H * bb["height"]; im.click(position={"x": u, "y": v}); idle(1.2)
    def goto_boundary(i): btn(re.compile(rf"^(✓ )?{i}$"), exact=False).click(); idle()
    def goto_set(i): btn(f"Frame set {i}").click(); idle()

    pg.goto(URL); pg.get_by_text("Measure a Room").first.wait_for(timeout=60000); idle(); shot("01_calibration")
    expand("Local test fixture (original-room calibration)"); btn("Use original-room calibration").click(); idle(2)
    pg.get_by_text(re.compile("Calibration accepted")).first.wait_for(timeout=60000); shot("02_calibration_accepted"); btn("Continue").click(); idle(); shot("03_upload_empty")
    t_sel = time.time(); expand("Local test fixture (original-room recordings)"); btn("Use original-room recordings").click()
    pg.get_by_text(re.compile(r"frame sets selected and boundaries suggested")).first.wait_for(timeout=300000); idle(); say("videos+frame selection+suggestions seconds:", round(time.time() - t_sel, 1))
    expand("Frame details and diagnostics"); shot("04_frames_selected"); btn("Review boundaries").click(); idle(2)
    pg.get_by_text("3 · Review the room boundaries").first.wait_for(timeout=60000); shot("05_review_first_proposal")
    conf_log = []
    for si in (1, 2):
        goto_set(si)
        for i in range(1, 7):
            goto_boundary(i); kind, arg = DECISIONS[i]
            if kind == "pick":
                if arg != 1 or (si, i) == (1, 4): pass
                pg.get_by_text(re.compile(rf"^Suggestion {arg}\b")).first.click(); idle(1.5)
                if si == 1 and i == 4: shot("06_candidate_switched_warning")
            else:
                btn("Correct manually").click(); idle(1.5)
                if si == 1: shot("07a_manual_panel")
                click_full(*arg[0]); click_full(*arg[1]); btn("Refine boundary").click(); idle(2)
                if si == 1: shot("07b_manual_refined")
            ok = btn("Confirm this boundary"); en = ok.is_enabled(); conf_log.append(dict(set=si, boundary=i, decision=kind, arg=arg, confirm_enabled=en))
            say("set", si, "boundary", i, kind, arg, "confirm enabled:", en)
            if not en: shot(f"blocked_set{si}_b{i}"); raise SystemExit(f"cannot confirm set {si} boundary {i}")
            ok.click(); idle(1.5)
            if si == 1 and i in (2, 3): shot(f"08_confirmed_set{si}_b{i}")
    shot("09_all_confirmed")
    say("view results enabled:", btn("View results").is_enabled()); btn("View results").click(); idle(); shot("10_results_before")
    t_m = time.time(); btn("Estimate room width and height").click()
    pg.get_by_text("Estimated width").first.wait_for(timeout=600000); idle(); say("measurement seconds:", round(time.time() - t_m, 1)); shot("11_results_estimate")
    pg.get_by_label("Actual width").fill("2900"); pg.get_by_label("Actual width").press("Enter"); idle(1.5); pg.get_by_label("Actual height").fill("2400"); pg.get_by_label("Actual height").press("Enter"); idle(2)
    shot("12_results_comparison"); txt = pg.inner_text("body"); say("BODY (results):", re.sub(r"\s+", " ", txt)[:1800])
    stop.set(); json.dump(dict(log=log, console_errors=errs, decisions=conf_log, rss_peak_mb=max(r[1] for r in rss) if rss else None, rss_samples=len(rss)), open(OUT / "walkthrough_log.json", "w"), indent=1)
    say("console errors:", errs[:5], "| peak server RSS MB:", max(r[1] for r in rss) if rss else None); b.close()

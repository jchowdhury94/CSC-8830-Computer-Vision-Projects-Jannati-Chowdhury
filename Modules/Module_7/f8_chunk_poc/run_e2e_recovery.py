"""F8.11 real-browser test of recovery-code privacy, copy, refresh/new-session recovery and the JSON download, against a RUNNING LOCAL F8 server
(Playwright + Chromium). SYNTHETIC videos only; the server must run locally with F8_ENABLE_FULL_CALIBRATION=1 (this runs a calibration of the synthetic pair).

    F8_E2E_PASSCODE=<passcode> python3 run_e2e_recovery.py <left.mp4> <right.mp4> <out.json> <server_log> [url]

Steps: log in -> upload LEFT/RIGHT -> run -> the recovery code is shown, the Copy button puts it on the clipboard, the URL never contains it ->
REFRESH the page mid-run (new session) -> recover by pasting the code in a NEW session -> download the JSON -> check it holds the calibration results and
no private paths or video bytes -> a wrong code finds nothing -> the code is in no request URL and not in the server log. The passcode is never recorded."""
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

LEFT, RIGHT, OUT, LOG = sys.argv[1:5]; URL = sys.argv[5] if len(sys.argv) > 5 else "http://127.0.0.1:8742"; PASS = os.environ["F8_E2E_PASSCODE"]
steps = []


def step(name, ok, **kw):
    steps.append(dict(step=name, ok=bool(ok), **kw)); print(("PASS " if ok else "FAIL ") + name, kw if kw else "", flush=True)


def login(pg):
    pg.goto(URL); pg.get_by_label("Passcode").fill(PASS); pg.get_by_role("button", name="Unlock").click()
    pg.get_by_text("Dual-video calibration").first.click()


def upload(pg, i, path, timeout=120):
    f = pg.frame_locator("iframe").nth(i); f.locator("#file").wait_for(timeout=30000); f.locator("#file").set_input_files(str(path)); t = time.time()
    while time.time() - t < timeout:
        if f.locator("#status").inner_text().startswith("Uploaded and verified"): return True
        time.sleep(0.1)
    return False


with sync_playwright() as p:
    exe = next(iter(sorted(glob.glob(os.path.expanduser("~/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-mac-arm64/chrome-headless-shell")))), None)
    try: b = p.chromium.launch()
    except Exception: b = p.chromium.launch(executable_path=exe)
    ctx = b.new_context(viewport=dict(width=1250, height=1700), accept_downloads=True, permissions=["clipboard-read", "clipboard-write"], base_url=URL)
    urls = []; ctx.on("request", lambda r: urls.append(r.url)); errs = []
    pg = ctx.new_page(); pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None); pg.on("pageerror", lambda e: errs.append(str(e)))
    login(pg); pg.wait_for_function("document.querySelectorAll('iframe').length === 2", timeout=60000)
    step("uploads LEFT and RIGHT", upload(pg, 0, LEFT) and (pg.wait_for_timeout(1500) or True) and upload(pg, 1, RIGHT))
    pg.wait_for_timeout(1500)
    pg.get_by_text("I understand a full calibration").click(); run = pg.get_by_role("button", name="Run full calibration")
    t1 = time.time()
    while not run.is_enabled() and time.time() - t1 < 10: time.sleep(0.2)
    run.click()
    pg.locator('[data-testid="stCode"]').first.wait_for(timeout=60000)
    code = pg.locator('[data-testid="stCode"]').first.inner_text().strip()
    step("a recovery code is displayed when the job starts", re.fullmatch(r"[A-Za-z0-9_-]{16,64}", code) is not None, code_length=len(code))
    step("the page URL has no query string and no code", urlparse(pg.url).query == "" and code not in pg.url, url_query=urlparse(pg.url).query)
    copy = pg.frame_locator("iframe").last.get_by_role("button", name="Copy Recovery Code")
    step("a Copy Recovery Code button exists", copy.count() == 1)
    copy.click(); pg.wait_for_timeout(500)
    try: clip = pg.evaluate("navigator.clipboard.readText()")
    except Exception as e: clip = f"ERR {type(e).__name__}"
    step("the Copy button puts exactly the code on the clipboard", clip == code, clipboard_matches=clip == code)

    # ---- browser refresh in the middle of the job: a new Streamlit session; the job must keep running
    pg.reload(); pg.get_by_label("Passcode").wait_for(timeout=30000)
    step("after a refresh the URL still has no code and the old session is gone (passcode asked again)", urlparse(pg.url).query == "" and code not in pg.url)
    pg.get_by_label("Passcode").fill(PASS); pg.get_by_role("button", name="Unlock").click(); pg.get_by_text("Dual-video calibration").first.click()
    pg.get_by_text("Recover a previous job").first.wait_for(timeout=30000); pg.get_by_text("Recover a previous job").first.click()
    box = pg.get_by_label("Recovery code"); box.wait_for(timeout=15000)
    step("the recovery field is masked (password type)", box.get_attribute("type") == "password")
    box.fill("definitely-not-a-real-code-12345"); box.press("Enter")
    pg.get_by_text("Nothing found for that code").wait_for(timeout=20000); step("a wrong code finds nothing", True)
    box.fill(code); box.press("Enter")
    seen = None; t = time.time()
    while time.time() - t < 120 and seen is None:
        txt = pg.inner_text("body")
        if "Job still running on the server" in txt: seen = "running"; break
        if "Saved result found" in txt: seen = "saved"; break
        time.sleep(0.3)
    step("a NEW session recovers the job by code (it was not cancelled by the refresh)", seen in ("running", "saved"), first_seen=seen)
    pg.get_by_text("Saved result found").wait_for(timeout=180000)
    txt = pg.inner_text("body"); step("the saved result reports a completed calibration", "outcome completed" in txt and "quality GOOD" in txt, line=(re.search(r"Saved result found[^\n]*", txt) or [""])[0][:140])
    with pg.expect_download(timeout=30000) as dl: pg.get_by_role("button", name="Download the result (JSON)").click()
    path = dl.value.path(); raw = Path(path).read_bytes(); doc = json.loads(raw); text = raw.decode("utf-8")
    cams = doc["cameras"]
    step("the download is JSON with the calibration results", doc["schema"] in ("f8.calibration_result/1", "f8.calibration_result/2") and doc["outcome"] == "completed" and doc["quality"]["level"] == "GOOD" and doc["quality"]["stereo_rms_px"] is not None
         and cams["left"]["K"] and cams["right"]["K"] and cams["left"]["distortion"] and doc["extrinsics"]["R"] and doc["extrinsics"]["T"] and doc["extrinsics"]["baseline_mm"]
         and doc["selected_poses"]["count"] and doc["synchronization"]["sync"] and doc["runtime"]["supervisor_total_s"] and doc["measurements"]["stages"],
         stereo_rms=doc["quality"]["stereo_rms_px"], baseline_mm=doc["extrinsics"]["baseline_mm"], poses=doc["selected_poses"]["count"], bytes=len(raw))
    leaks = [x for x in ("/private", "/var/folders", "/tmp", "/Users/", "f8_job_", "f8_store", ".mp4", ".mov", "ftyp", "mdat", code, PASS) if x in text]
    step("the download contains no private paths, video bytes, recovery code or passcode", not leaks and len(raw) < 1_000_000 and "corners" not in text, leaks=leaks)
    step("videos were deleted after the saved result (success cleanup)", doc["cleanup"]["videos_deleted"] is True)
    # ---- a second, independent browser context (another person / device) can also recover with the code
    ctx2 = b.new_context(accept_downloads=True); pg2 = ctx2.new_page(); login(pg2)
    pg2.get_by_text("Recover a previous job").first.wait_for(timeout=30000); pg2.get_by_text("Recover a previous job").first.click()
    pg2.get_by_label("Recovery code").fill(code); pg2.get_by_label("Recovery code").press("Enter")
    pg2.get_by_text("Saved result found").wait_for(timeout=30000); step("a separate browser context recovers the result with the code", True)
    step("no request URL ever contained the code", not any(code in u for u in urls), requests=len(urls))
    step("no console errors", not [e for e in errs if "favicon" not in e], errors=errs[:3])
    pg.screenshot(path=str(Path(OUT).with_suffix(".png")), full_page=True); b.close()
log = Path(LOG).read_text(errors="ignore") if Path(LOG).exists() else ""
step("the recovery code and passcode are not in the server log", code not in log and PASS not in log, log_bytes=len(log))
res = dict(steps=steps, all_passed=all(s["ok"] for s in steps)); Path(OUT).write_text(json.dumps(res, indent=1))
print("ALL PASSED" if res["all_passed"] else "SOME STEPS FAILED"); sys.exit(0 if res["all_passed"] else 1)

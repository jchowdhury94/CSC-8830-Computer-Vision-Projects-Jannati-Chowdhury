"""Real-browser check of the access gate against a RUNNING server (Playwright). Prints one line per check; exit code 1 if any check fails.

    F8_E2E_PASSCODE=<the server's passcode> python3 run_e2e_gate.py <url> [--expect-locked]

--expect-locked : the server was started WITHOUT a passcode; the page must show no form and no upload component.
Otherwise       : the page must show only the form; a wrong passcode must be refused; the right one (from F8_E2E_PASSCODE) must reveal the uploader.
The passcode is read from the environment and never printed or saved.
"""
import glob
import os
import sys

from playwright.sync_api import sync_playwright

url = sys.argv[1]; locked = "--expect-locked" in sys.argv; code = os.environ.get("F8_E2E_PASSCODE", ""); ok = True


def check(name, cond):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)


with sync_playwright() as p:
    exe = next(iter(sorted(glob.glob(os.path.expanduser("~/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-mac-arm64/chrome-headless-shell")))), None)
    try: b = p.chromium.launch()
    except Exception: b = p.chromium.launch(executable_path=exe)
    pg = b.new_page(); pg.goto(url); pg.wait_for_selector("h1", timeout=60000); pg.wait_for_timeout(2500)
    if locked:
        check("shows the 'locked' message", pg.get_by_text("locked").count() > 0)
        check("no passcode form", pg.get_by_label("Passcode").count() == 0)
        check("no upload component (no iframe)", pg.locator("iframe").count() == 0)
    else:
        check("login form is shown", pg.get_by_label("Passcode").count() == 1)
        check("passcode field is a password field", pg.get_by_label("Passcode").get_attribute("type") == "password")
        check("no upload component before login", pg.locator("iframe").count() == 0)
        check("no 'Choose File' control before login", pg.get_by_text("Choose File").count() == 0)
        pg.get_by_label("Passcode").fill("definitely-wrong-passcode"); pg.get_by_role("button", name="Unlock").click(); pg.wait_for_timeout(2000)
        check("wrong passcode is refused", pg.get_by_text("Incorrect passcode").count() > 0)
        check("still no upload component after a wrong passcode", pg.locator("iframe").count() == 0)
        pg.get_by_label("Passcode").fill(code); pg.get_by_role("button", name="Unlock").click(); pg.wait_for_selector("iframe", timeout=30000)
        check("correct passcode reveals the upload component", pg.locator("iframe").count() == 1)
        check("login form is gone after login", pg.get_by_label("Passcode").count() == 0)
        pg.reload(); pg.wait_for_selector("h1", timeout=60000); pg.wait_for_timeout(2500)
        check("a new browser session (reload) must authenticate again", pg.locator("iframe").count() == 0 and pg.get_by_label("Passcode").count() == 1)
    b.close()
sys.exit(0 if ok else 1)

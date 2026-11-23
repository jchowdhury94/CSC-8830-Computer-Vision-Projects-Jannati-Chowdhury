"""Start the LOCAL Stage F1 annotation prototype (isolated; not linked from any existing page).

    python Modules/Module_7/run_stage_f1_annotation_ui.py          ->  http://127.0.0.1:8620

Binds to 127.0.0.1 only, dark theme with the Module 7 red accent, no usage statistics.  Needs the local frames from annotation_ui/prepare_local_frames.py."""
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent / "annotation_ui" / "app_annotate.py"
PORT = "8620"

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP), "--server.address", "127.0.0.1", "--server.port", PORT, "--server.headless", "true", "--browser.gatherUsageStats", "false",
           "--client.toolbarMode", "minimal", "--theme.base", "dark", "--theme.primaryColor", "#C9363E"]
    print(f"Stage F1 annotation prototype: http://127.0.0.1:{PORT}")
    raise SystemExit(subprocess.call(cmd))

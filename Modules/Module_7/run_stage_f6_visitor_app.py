"""Start the LOCAL Stage F6 visitor workflow (isolated; not linked from any existing page; nothing is deployed).

    python Modules/Module_7/run_stage_f6_visitor_app.py [port]        ->  http://127.0.0.1:8630  (default)

Home / Back (top row) open the main app's Home and /module-7; set F6_MAIN_APP_URL (default http://localhost:8501, i.e. `streamlit run home.py`).
Binds to 127.0.0.1 only, dark theme with the Module 7 red accent, no usage statistics.  Classical computer vision only.
The upload cap stays at Streamlit's 200 MB default, matching the Stage B video limit (it is not loosened)."""
import socket
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent / "visitor_workflow" / "app_f6.py"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8630

if __name__ == "__main__":
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0: raise SystemExit(f"port {PORT} is already in use; pass another port: python {Path(__file__).name} <port>")
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP), "--server.address", "127.0.0.1", "--server.port", str(PORT), "--server.headless", "true", "--server.maxUploadSize", "200",
           "--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal", "--theme.base", "dark", "--theme.primaryColor", "#C9363E"]
    print(f"Stage F6 visitor workflow: http://127.0.0.1:{PORT}")
    raise SystemExit(subprocess.call(cmd))

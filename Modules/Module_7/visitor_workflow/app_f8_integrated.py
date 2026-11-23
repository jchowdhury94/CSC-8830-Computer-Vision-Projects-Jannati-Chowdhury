"""Stage F8.18: LOCAL integrated visitor calibration experiment (F8 chunked calibration -> review -> accept -> F6 hand-off).

Run:   python Modules/Module_7/run_f8_integrated_app.py          (binds to 127.0.0.1 only)
ISOLATED: not linked from Pages/app_module7.py, does not replace app_f6.py or the F8 proof of concept (f8_chunk_poc/app.py), and uses only ``f8i_`` session keys.
Room VIDEO UPLOAD + validation exist (F8.31A); room processing / reconstruction is NOT part of this phase and is not reachable from here.
Access: open. The app starts directly on its home page (no password or login screen); the upload limits, the resource budget, per-session isolation and the cleanup sweep still apply."""
import logging
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
for _p in (str(MODULE), str(HERE), str(MODULE / "f8_chunk_poc"), str(MODULE.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st                                                            # noqa: E402

st.set_page_config(page_title="Measure a Room — calibration experiment", layout="centered")
logging.getLogger("f8_chunk_store").setLevel(logging.INFO)
ss = st.session_state


@st.cache_resource
def get_service():
    """One service per server process (uploads are owned per session token).  Imported lazily on the first run of the app."""
    from calibration_runner import RunnerLimits
    from chunk_store import ChunkStore, env_limits
    from f8_visitor_service import VisitorCalibrationService
    from result_store import ResultStore
    from room_videos import ALL_ROOM_SLOTS, room_limits
    store, results = ChunkStore(env_limits()), ResultStore()
    room_store = ChunkStore(room_limits(), slots=ALL_ROOM_SLOTS, root_prefix="f8_room_videos_")   # F8.31A: separate bounded temp dir, never shared with calibration uploads

    def sweeper():
        n = 0
        while True:
            time.sleep(30); n += 1
            try:
                store.sweep(); room_store.sweep()
                if n % 10 == 0:
                    results.sweep()
            except Exception:
                pass

    threading.Thread(target=sweeper, daemon=True, name="f8i-sweeper").start()
    return VisitorCalibrationService(store, results, RunnerLimits.from_env(), room_store=room_store)


def main():
    import f8_visitor_pages as P
    P.render(get_service(), P.session_of(ss), ss)


main()

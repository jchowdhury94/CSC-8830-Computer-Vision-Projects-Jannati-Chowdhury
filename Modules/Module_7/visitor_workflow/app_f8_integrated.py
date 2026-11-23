"""Stage F8.18: LOCAL integrated visitor calibration experiment (F8 chunked calibration -> review -> accept -> F6 hand-off).

Run:   python Modules/Module_7/run_f8_integrated_app.py          (binds to 127.0.0.1 only)
ISOLATED: not linked from Pages/app_module7.py, does not replace app_f6.py or the F8 proof of concept (f8_chunk_poc/app.py), and uses only ``f8i_`` session keys.
Room VIDEO UPLOAD + validation exist (F8.31A); room processing / reconstruction is NOT part of this phase and is not reachable from here.
Access: the same fail-closed passcode gate as the F8 app (F8_PASSCODE, at least 8 characters). Nothing below the gate (store, upload component, service) exists until it is passed."""
import logging
import os
import secrets
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

from auth import AttemptLimiter, load_passcode                                    # noqa: E402

st.set_page_config(page_title="Measure a Room — calibration experiment", layout="centered")
logging.getLogger("f8_chunk_store").setLevel(logging.INFO)
ss = st.session_state


@st.cache_resource
def get_limiter():
    return AttemptLimiter()


def _gate() -> bool:
    """FAIL CLOSED.  Returns True only when this session has entered the passcode."""
    ss.setdefault("f8i_gate_id", secrets.token_urlsafe(24))
    expected, _problem = load_passcode(lambda name: st.secrets[name], os.environ)
    if expected is None:
        st.markdown("<h1 style='text-align:center'>Measure a Room</h1>", unsafe_allow_html=True)
        st.error("This test app is locked: it has not been configured with an access passcode. The administrator must set F8_PASSCODE (environment variable or Streamlit secrets, at least 8 characters).")
        return False
    if ss.get("f8i_authed"):
        return True
    st.markdown("<h1 style='text-align:center'>Measure a Room</h1>", unsafe_allow_html=True)
    with st.form("f8i_gate", clear_on_submit=True):
        typed = st.text_input("Passcode", type="password")
        submitted = st.form_submit_button("Unlock")
    if submitted:
        granted, note = get_limiter().attempt(ss["f8i_gate_id"], typed, expected)
        if granted:
            ss["f8i_authed"] = True
            st.rerun()
        st.error(note)
    return False


@st.cache_resource
def get_service():
    """One service per server process (uploads are owned per session token).  Imported lazily so nothing of F8 exists before the gate is passed."""
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
    if not _gate():
        st.stop()
    import f8_visitor_pages as P
    P.render(get_service(), P.session_of(ss), ss)


main()

"""F8.31C-2C-1: ISOLATED, passcode-gated Cloud resource diagnostic (a separate Streamlit app; NOT linked from home.py, Pages/ or any visitor app).

Entry point for a separate Community Cloud app:  Modules/Module_7/f8_chunk_poc/cloud_diagnostic_app.py  (it picks up the requirements.txt next to it).
Access: the SAME fail-closed passcode mechanism as the other F8 apps (``auth.py``; F8_PASSCODE, at least 8 characters, throttled).  Nothing is measured and nothing is
shown until the passcode is accepted AND the operator presses the button: no automatic run, no background loop, no route a bare URL can reach, nothing persisted.
It imports neither Method J, calibration, reconstruction, the upload store nor any visitor module; ROOM_PROCESSING_ENABLED is not imported and not involved.
"""
import json
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st                                              # noqa: E402

from auth import AttemptLimiter, load_passcode                      # noqa: E402

st.set_page_config(page_title="Resource diagnostic", layout="centered")
ss = st.session_state


@st.cache_resource
def _limiter():
    return AttemptLimiter()


ss.setdefault("diag_owner", secrets.token_urlsafe(24))
expected, _problem = load_passcode(lambda name: st.secrets[name], os.environ)
if expected is None:                                               # FAIL CLOSED
    st.error("This diagnostic is locked: no access passcode is configured.")
    st.stop()
if not ss.get("diag_authed"):
    with st.form("diag_gate", clear_on_submit=True):
        typed = st.text_input("Passcode", type="password")
        submitted = st.form_submit_button("Unlock")
    if submitted:
        granted, note = _limiter().attempt(ss["diag_owner"], typed, expected)
        if granted:
            ss["diag_authed"] = True
            st.rerun()
        st.error(note)
    st.stop()

import cloud_resource_snapshot as CS                                # noqa: E402  (imported only after the gate)

st.title("Resource diagnostic")
st.caption("One read-only snapshot per button press. Nothing is stored, no video or visitor data is read, and no calibration or reconstruction is started.")
probe = st.checkbox("Also time a trivial child process (python -I -c pass) and test process-group cleanup", value=False)
if st.button("Take snapshot"):
    snap = CS.take_snapshot(probe_processes=probe)
    st.json(snap, expanded=2)
    st.caption("Copy the JSON above by hand if needed; the app keeps no copy.")

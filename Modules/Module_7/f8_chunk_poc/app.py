"""Stage F8: chunked-upload proof of concept (ISOLATED experiment: not part of Module 7, not linked from any page).

Run locally:   streamlit run Modules/Module_7/f8_chunk_poc/app.py
The browser slices the chosen file into ~2 MB chunks and sends them one at a time through a custom component; Python appends them to a
temporary file on disk (no file bytes in RAM or session state) and verifies SHA-256. Then OpenCV inspects the assembled file.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path

import streamlit as st

from auth import AttemptLimiter, load_passcode
from chunk_store import MiB, ChunkStore, UploadError, env_limits
from component import chunk_uploader
from perf import current_rss_mb, peak_rss_mb
from protocol import handle
from video_check import inspect_video

st.set_page_config(page_title="F8 chunked upload PoC", layout="centered")
logging.getLogger("f8_chunk_store").setLevel(logging.INFO)


@st.cache_resource
def get_store():
    """One store per server process (shared by sessions; ownership is enforced per session token). A daemon thread expires stale uploads."""
    store = ChunkStore(env_limits())
    start = dict(rss_mb=current_rss_mb(), peak_mb=peak_rss_mb(), t=time.time())

    def sweeper():
        while True:
            time.sleep(30)
            try:
                store.sweep()
            except Exception:
                pass

    threading.Thread(target=sweeper, daemon=True, name="f8-sweeper").start()
    return store, start


@st.cache_resource
def get_limiter():
    return AttemptLimiter()


ss = st.session_state
ss.setdefault("owner", secrets.token_urlsafe(24))          # unguessable per-session token; the only credential for this session's uploads

st.title("F8 · Chunked upload proof of concept")
st.caption("Isolated test: select a large video; it is sent in small pieces and assembled in a private temporary file. Nothing is stored permanently.")

# ---- access gate (F8.1). FAIL CLOSED: nothing below this block (store, upload component, message handling) exists until the passcode is accepted.
expected, problem = load_passcode(lambda name: st.secrets[name], os.environ)
if expected is None:
    st.error("This test app is locked: it has not been configured with an access passcode. The administrator must set F8_PASSCODE "
             "(Streamlit secrets or environment variable, at least 8 characters).")
    st.stop()
if not ss.get("authed"):
    with st.form("f8_gate", clear_on_submit=True):          # the typed value is only read inside the submit run, then the field is cleared
        typed = st.text_input("Passcode", type="password")
        submitted = st.form_submit_button("Unlock")
    if submitted:
        granted, note = get_limiter().attempt(ss["owner"], typed, expected)
        if granted:
            ss["authed"] = True
            st.rerun()
        st.error(note)
    st.stop()

store, baseline = get_store()
mode = st.radio("Test", ["Single upload", "Dual-video calibration"], horizontal=True, key="f8_mode", label_visibility="collapsed")
if mode == "Dual-video calibration":                        # F8.5: two slots, a lease, and the F7 functions in a worker process
    import dual_page
    dual_page.render(store, ss, store.limits)
    st.stop()
ss.setdefault("proto", {})                                 # last message id + ack (a few hundred bytes)
ss.setdefault("epoch", 0)
ss["runs"] = ss.get("runs", 0) + 1                         # script executions in this session (reruns)
ss.setdefault("video", {})
ss.setdefault("saved", set())

# 1) read the browser's newest message BEFORE drawing the component, so the ack travels back in the same run.
msg = ss.get("f8_uploader")
ack = handle(store, ss["owner"], ss["proto"], msg)
lim = store.limits
chunk_uploader(lim.default_chunk_bytes, lim.max_file_bytes, ack, ss["epoch"], key="f8_uploader")

with st.expander("Limits", expanded=False):
    cap = store.capacity()
    st.write(f"Chunk {lim.default_chunk_bytes / MiB:g} MB · max file {lim.max_file_bytes / MiB:.0f} MB · max temporary disk {lim.max_disk_bytes / MiB:.0f} MB · "
             f"max active uploads {lim.max_active_uploads} · idle timeout {lim.idle_timeout_s:.0f} s · max upload time {lim.max_upload_s:.0f} s")
    st.write(f"Disk: {cap['free_bytes'] / MiB:.0f} MB free · {cap['pending_bytes'] / MiB:.0f} MB promised to running uploads · "
             f"{cap['margin_bytes'] / MiB:.0f} MB safety margin → **{cap['usable_bytes'] / MiB:.0f} MB usable** for a new upload")

if ack and not ack.get("ok"):
    st.warning(ack.get("message") or ack.get("error"))

uid = ack.get("upload_id") if ack else None
if ack and ack.get("ok") and uid and ack.get("state") == "uploading":
    st.progress(min(1.0, ack["bytes_received"] / max(ack["size"], 1)), text=f"Received {ack['next_index']} / {ack['n_chunks']} chunks ({ack['bytes_received'] / MiB:.1f} MB)")

if ack and ack.get("ok") and uid and ack.get("state") == "complete":
    owner = ss["owner"]
    try:
        m = store.metrics(owner, uid)
        path = store.path_of(owner, uid)
    except UploadError as e:
        st.info(e.message); m = path = None
    if m:
        st.success(f"Assembled and verified. SHA-256 `{m['sha256'][:16]}…`  ·  {m['size_bytes'] / MiB:.1f} MB in {m['n_chunks']} chunks")
        if uid not in ss["video"]:                         # read-only inspection, once per upload
            with st.spinner("Inspecting the video with OpenCV…"):
                ss["video"][uid] = inspect_video(path)
        v = ss["video"][uid]
        st.subheader("Video inspection (OpenCV, sandboxed)")
        for e in v["errors"]:
            st.error(e["message"])
        if v["opened"] and v["decodable"]:
            c = st.columns(4)
            c[0].metric("Resolution", f"{v['width']}×{v['height']}" if v["width"] else "-"); c[1].metric("FPS", f"{v['fps']:.2f}" if v["fps"] else "-")
            c[2].metric("Frames", v["frame_count"] if v["frame_count"] else "-"); c[3].metric("Duration", f"{v['duration_s']:.1f} s" if v["duration_s"] else "-")
            st.write(f"Container: `{v['container']}` · codec: `{v['codec']}` · backend: `{v['backend']}` · decoded {v['frames_read']} sample frame(s) in "
                     f"{v['sample_decode_ms']} ms ({v['sample_fps']} fps) · inspection took {v['worker_s']} s")
        for issue in v["issues"][len(v["errors"]):]:
            st.warning(issue)

        cs = m["client_stats"] or {}
        rss_now, rss_peak = current_rss_mb(), peak_rss_mb()
        report = dict(
            upload=dict(size_mb=round(m["size_bytes"] / MiB, 2), chunk_mb=round(m["chunk_size"] / MiB, 3), chunks=m["n_chunks"], duplicates=m["duplicates"], rejected=m["rejected"],
                        browser_upload_s=cs.get("upload_s"), browser_mb_per_s=cs.get("mb_per_s"), server_upload_s=m["server_upload_s"], server_mb_per_s=m["server_mb_per_s"],
                        resumed_from_chunk=cs.get("resumed_from_chunk")),
            ack_latency_ms=dict(browser_mean=cs.get("ack_ms_mean"), browser_p50=cs.get("ack_ms_p50"), browser_p95=cs.get("ack_ms_p95"), browser_max=cs.get("ack_ms_max"),
                                server_handling=m["server_chunk_ms"]),
            reruns=dict(session_script_runs=ss["runs"], messages_processed=ss["proto"].get("processed")),
            memory=dict(rss_now_mb=None if rss_now is None else round(rss_now, 1), rss_peak_mb=None if rss_peak is None else round(rss_peak, 1),
                        rss_at_start_mb=None if baseline["rss_mb"] is None else round(baseline["rss_mb"], 1),
                        rss_peak_growth_mb=None if rss_peak is None or baseline["peak_mb"] is None else round(rss_peak - baseline["peak_mb"], 1),
                        browser_hash_ms_total=cs.get("hash_ms_total")),
            disk=dict(temp_usage_mb=round(store.disk_usage_bytes() / MiB, 2), reserved_mb=round(store.reserved_bytes() / MiB, 2)),
            video=dict(ok=v["ok"], container=v["container"], stage=v["stage"], error_codes=[e["code"] for e in v["errors"]], opened=v["opened"], decodable=v["decodable"],
                       width=v["width"], height=v["height"], fps=v["fps"], frame_count=v["frame_count"], duration_s=v["duration_s"], codec=v["codec"], inspection_s=v["worker_s"]),
            limits=dict(max_file_mb=lim.max_file_bytes / MiB, max_disk_mb=lim.max_disk_bytes / MiB, max_active=lim.max_active_uploads, usable_disk_mb=round(store.capacity()["usable_bytes"] / MiB)))
        st.subheader("Measurements")
        st.json(report)
        out_dir = os.environ.get("F8_METRICS_DIR")
        if out_dir and uid not in ss["saved"]:                 # optional local record: counters and timings only, never file name or content
            try:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                (Path(out_dir) / f"upload_{int(time.time())}_{uid[:6]}.json").write_text(json.dumps(report, indent=1, default=str)); ss["saved"].add(uid)
            except OSError:
                pass
        if st.button("Delete the uploaded file and start over"):
            try:
                store.release(owner, uid)
            except UploadError:
                pass
            ss["video"].pop(uid, None); ss["epoch"] += 1; ss["proto"] = {}
            st.rerun()

st.divider()
if st.button("Lock this session (deletes your uploads)"):
    store.release_owner(ss["owner"]); ss["video"].clear(); ss["proto"] = {}; ss["epoch"] += 1; ss["authed"] = False
    st.rerun()
st.caption(f"Server: {store.active_count()} active upload(s) · {store.disk_usage_bytes() / MiB:.1f} MB temporary disk in use · script runs this session: {ss['runs']}")

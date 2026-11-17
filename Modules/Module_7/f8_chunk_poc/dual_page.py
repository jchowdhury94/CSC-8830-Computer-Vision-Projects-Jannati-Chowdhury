"""Stage F8.5: local test page for the dual-video calibration (Streamlit drawing only; all rules live in chunk_store / calibration_runner).

Reached through the same passcode gate as the rest of the F8 app. Not part of Module 7. Two upload slots (LEFT, RIGHT), one Run button, live progress,
Cancel, and the result with stage-by-stage resource measurements."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st

from calibration_runner import CalibrationJobRunner, RunnerLimits, active_runner, cancel_by_code
from chunk_store import MiB, SLOTS, UploadError
import component
from protocol import handle
from result_store import ResultStore, job_id, new_code, valid_code

LIMITS = RunnerLimits.from_env()                                       # F8_CALIB_TIMEOUT_S, F8_BENCH_*, F8_ENABLE_FULL_CALIBRATION


@st.cache_resource
def get_results():
    """One result store per server process (server-local, ephemeral disk). A daemon thread deletes results older than their time-to-live."""
    import threading
    rs = ResultStore()

    def sweeper():
        while True:
            time.sleep(300)
            try:
                rs.sweep()
            except Exception:
                pass
    threading.Thread(target=sweeper, daemon=True, name="f8-result-sweeper").start()
    return rs


def _start_job(store, results, ss, owner, settings, mode, bench_n=None):
    """Create the recovery code BEFORE the job starts, so the person can note it (and the URL keeps it) while the job runs."""
    code = new_code()
    store.set_recovery(owner, code)
    ss["runner"] = CalibrationJobRunner(store, owner, settings, LIMITS, mode=mode, benchmark_frames=bench_n, results=results, job_code=code)
    ss["runner"].start(); ss["dual_saved"] = False; ss["job_code"] = code                  # the code lives only in this session's memory and on the page; never in the URL or a log


def _show_code(code, key):
    """Show the recovery code to this session only (page text, not the URL) with a Copy button. The code is a secret: it is never logged or put in a query parameter."""
    import streamlit.components.v1 as components
    st.code(code, language=None)
    components.html(
        "<button id='b' style='font:14px system-ui;padding:6px 12px;border-radius:6px;border:1px solid #888;cursor:pointer'>Copy Recovery Code</button> <span id='m' style='font:13px system-ui'></span>"
        "<script>const c=" + json.dumps(code) + ";const m=document.getElementById('m');"
        "function ok(){m.textContent='Copied.'}function bad(){const t=document.createElement('textarea');t.value=c;document.body.appendChild(t);t.select();"
        "try{document.execCommand('copy')?ok():m.textContent='Select the code above and copy it.'}catch(e){m.textContent='Select the code above and copy it.'}t.remove()}"
        "document.getElementById('b').onclick=()=>{(navigator.clipboard&&navigator.clipboard.writeText?navigator.clipboard.writeText(c).then(ok,bad):bad())};</script>", height=44)


def _download(results, code, label="Download the result (JSON)"):
    data = results.result_bytes(code)
    if data is None:
        st.warning("The result file is not available on the server (it may have been deleted or the server restarted).")
        return None
    st.download_button(label, data, file_name=f"f8_result_{job_id(code)}.json", mime="application/json", key=f"dl_{job_id(code)}")
    return data


def _recovery(store, results, ss, owner):
    """Find a job from ANOTHER session (refresh, new tab, disconnect) by its recovery code: live status, the saved result, or the retained videos."""
    if "job" in st.query_params:                                       # F8.11: an old ?job= link is ignored and removed; the code is only ever typed or pasted
        del st.query_params["job"]
    with st.expander("Recover a previous job (after a refresh or disconnect)"):
        st.caption("Results are saved on the server, not in this browser session, and are found with the recovery code shown when the job started (copy it somewhere private: it is NOT stored in the page address). "
                   "Server storage is **temporary**: an app restart, redeploy or reboot erases it, and results are deleted after about a day. Download the result to keep it.")
        code = st.text_input("Recovery code", key="dual_rec_code", type="password").strip()
        if not code:
            return
        if not valid_code(code):
            st.warning("That does not look like a recovery code.")
            return
        live = active_runner(code)
        if live is not None:
            s = live.status()
            st.info(f"Job still running on the server: {s['stage'].replace('_', ' ')} · {s['elapsed_s']:.0f} s · {s['fraction']:.0%}")
            if st.button("Cancel that job", key="dual_rec_cancel"):
                cancel_by_code(code)
            time.sleep(2.0); st.rerun()
            return
        res = results.load_result(code)
        if res is not None:
            q = res.get("quality") or {}
            st.success(f"Saved result found: outcome **{res.get('outcome')}** · quality **{q.get('level')}** · stereo RMS {q.get('stereo_rms_px')} px · "
                       f"baseline {((res.get('extrinsics') or {}).get('baseline_mm'))} mm")
            _download(results, code)
        else:
            stt = results.read_status(code)
            if stt:
                st.warning(f"No saved result. The job last reported “{stt.get('state')} / {stt.get('stage')}” {stt.get('age_s')} s ago; if it is not running now, the server was probably restarted "
                           "or the job was interrupted.")
            else:
                st.info("Nothing found for that code: it expired, or the server restarted (its local disk is temporary).")
        if st.button("Reclaim my retained videos for a retry", key="dual_rec_adopt"):
            try:
                n = store.adopt(code, owner)["adopted"]
                st.success(f"{n} video(s) are now attached to this session." if n else "These videos already belong to this session.")
                st.rerun()
            except UploadError as e:
                st.error(e.message)


def _slot_line(info, slot):
    if not info:
        return f"{slot.upper()}: no video yet"
    if info["state"] == "complete":
        return f"{slot.upper()}: ✅ {info['size'] / MiB:.1f} MB verified · SHA-256 {info['sha256'][:12]}…" + (" · in use by the calibration" if info["leased"] else "")
    return f"{slot.upper()}: uploading {info['next_index']}/{info['n_chunks']} chunks"


def render(store, ss, lim):
    results = get_results()
    owner = ss["owner"]; ss.setdefault("dual_epoch", 0); ss.setdefault("runner", None); ss.setdefault("dual_saved", False)
    st.subheader("Dual-video stereo calibration (local test)")
    st.caption("Upload the LEFT and RIGHT calibration videos one after the other. Both are kept until the calibration has finished; the camera is decided by the slot, "
               "never by the file name. The existing F7 calibration functions run in a separate worker process. The **benchmark** only measures checkerboard detection "
               "on the first frames of each video (nothing is solved) and keeps your uploads so you can decide on a full run afterwards.")
    runner = ss["runner"]; running = runner is not None and not runner.done
    acks = {}
    for slot in SLOTS:
        proto = ss.setdefault(f"dual_proto_{slot}", {})
        acks[slot] = handle(store, owner, proto, ss.get(f"f8_up_{slot}"))
    cols = st.columns(2)
    for col, slot in zip(cols, SLOTS):
        with col:
            component.chunk_uploader(lim.default_chunk_bytes, lim.max_file_bytes, acks[slot], ss["dual_epoch"], key=f"f8_up_{slot}", slot=slot, label=f"{slot.upper()} camera video")
            if acks[slot] and not acks[slot].get("ok"):
                st.warning(acks[slot].get("message") or acks[slot].get("error"))
    job = store.job_status(owner)
    for slot in SLOTS:
        st.caption(_slot_line(job["slots"].get(slot), slot))

    if st.button("Lock this session (cancels any calibration, deletes uploads)"):
        store.release_owner(owner); ss["runner"] = None; ss["dual_epoch"] += 1; ss["authed"] = False; ss.pop("job_code", None)
        for slot in SLOTS:
            ss[f"dual_proto_{slot}"] = {}
        st.rerun()

    with st.expander("Checkerboard and options", expanded=runner is None):
        c = st.columns(4)
        cols_n = c[0].number_input("Inner corners — columns", 2, 40, 8, key="dual_cols"); rows_n = c[1].number_input("Inner corners — rows", 2, 40, 6, key="dual_rows")
        sq = c[2].number_input("Square size (mm)", 1.0, 500.0, 20.0, key="dual_sq"); base = c[3].number_input("Measured baseline (mm, 0 = none)", 0.0, 2000.0, 0.0, key="dual_base")
        bench_n = st.number_input("Benchmark: frames per video", 30, LIMITS.benchmark_frames_cap, LIMITS.benchmark_frames, step=30, key="dual_bench_n")
        st.caption(f"Safety limits for these recordings: {LIMITS.min_duration_s:g}–{LIMITS.max_duration_s:g} s each, same resolution and frame rate, ≤ {LIMITS.max_frames} frames. "
                   f"Benchmark timeout {LIMITS.benchmark_timeout_s:g} s (≤ {LIMITS.benchmark_frames_cap} frames per video) · full-calibration timeout {LIMITS.timeout_s:g} s "
                   f"({'enabled' if LIMITS.allow_full else 'DISABLED on this server'}). (F7's 200 MB / 120 s visitor-upload limits do not apply.)")

    both = all(job["slots"].get(s, {}).get("state") == "complete" for s in SLOTS)
    settings = dict(cols=int(cols_n), rows=int(rows_n), square_mm=float(sq), baseline_mm=float(base) or None)
    blocked = not both or running or runner is not None
    c = st.columns([1, 1, 1, 2])
    if c[0].button("Run benchmark", type="primary", disabled=blocked):
        _start_job(store, results, ss, owner, settings, "benchmark", int(bench_n))
        st.rerun()
    if LIMITS.allow_full:
        sure = c[3].checkbox("I understand a full calibration can take up to an hour. Both videos are deleted only after the result is saved; after a failure they are kept "
                             f"for {store.limits.retry_retention_s / 60:.0f} min so you can retry.", key="dual_full_ok")
        if c[1].button("Run full calibration", disabled=blocked or not sure):
            _start_job(store, results, ss, owner, settings, "calibrate")
            st.rerun()
    else:
        c[3].caption("Full calibration is disabled on this server.")
    if c[2].button("Cancel", disabled=not running):
        runner.cancel()

    runner = ss["runner"]
    if runner is None:
        _recovery(store, results, ss, owner)
    if runner is not None:
        s = runner.status()
        if not runner.done:
            st.progress(min(s["fraction"], 0.99), text=f"{s['stage'].replace('_', ' ')} · {s['elapsed_s']:.0f} s")
            m = s.get("monitor") or {}
            st.caption(f"parent RSS {m.get('parent_rss_mb')} MB · worker RSS {m.get('worker_rss_mb')} MB · temp disk {m.get('temp_disk_mb')} MB")
            st.info("Recovery code: copy it now and keep it private. If the page reloads or disconnects, paste it under “Recover a previous job”.")
            _show_code(runner.job_code, "run")
            time.sleep(1.0); st.rerun()
        else:
            _result(runner, ss, results)
            kept = (runner.result or {}).get("cleanup", {}).get("videos_kept_for_next_step")
            if st.button("Clear result" if kept else "Start over"):
                ss["runner"] = None; ss.pop("job_code", None)
                if not kept:                                            # the videos are gone: reset the uploaders too
                    ss["dual_epoch"] += 1
                    for slot in SLOTS:
                        ss[f"dual_proto_{slot}"] = {}
                st.rerun()


def _benchmark_result(res):
    b = res.get("benchmark_report") or {}
    out = res.get("outcome")
    (st.success if out == "completed" else st.error)(f"Benchmark {out}" + ("" if out == "completed" else f": {res.get('error') or ''}"))
    cpu, mem, disk, lim, integ = (b.get(k) or {} for k in ("cpu", "memory", "disk", "limits", "integrity"))
    est = " (estimate from the last progress report)" if b.get("frames_processed_is_estimate") else ""
    st.table([{"Measure": "Frames processed" + est, "Value": b.get("frames_processed")},
              {"Measure": "Detection attempts / checkerboard detections", "Value": f"{b.get('detection_attempts')} / {b.get('checkerboard_detections')}"},
              {"Measure": "Wall-clock (worker, total)", "Value": b.get("wall_clock_s")},
              {"Measure": "Frames per second", "Value": b.get("frames_per_s")},
              {"Measure": "CPU seconds (worker) · average cores · limit", "Value": f"{cpu.get('worker_cpu_s')} · {cpu.get('worker_avg_cores')} · {cpu.get('cpu_limit_cores')}"},
              {"Measure": "Throttled periods / seconds", "Value": f"{cpu.get('throttled_periods')} / {cpu.get('throttled_s')}" if cpu.get("throttling_measurable") else "not measurable here"},
              {"Measure": "Peak RSS: worker · worker tree · app+worker", "Value": f"{mem.get('worker_self_peak_rss_mb')} · {mem.get('worker_tree_peak_rss_mb')} · {mem.get('combined_peak_rss_mb')} MB"},
              {"Measure": "Container memory peak · limit", "Value": f"{mem.get('container_peak_mb')} · {mem.get('container_limit_mb')} MB" if mem.get("container_measurable") else "not measurable here"},
              {"Measure": "Temporary disk peak · memory-backed?", "Value": f"{disk.get('temp_disk_peak_mb')} MB · {disk.get('temp_dir_memory_backed')}"},
              {"Measure": "Timed out / cancelled / memory guard", "Value": f"{lim.get('timed_out')} / {lim.get('cancelled')} / {lim.get('memory_guard_triggered')}"},
              {"Measure": "SHA-256 verified before use · files unchanged after", "Value": f"{integ.get('sha256_verified_before_use')} · {integ.get('videos_unchanged_after')}"}], hide_index=True)
    if b.get("projection"):
        p = b["projection"]
        st.info(f"Estimated full-length detection time ≈ {p['estimated_full_detection_s']:.0f} s for {p['full_length_frames']} frames. {p['note']}")
    st.json(b, expanded=False)


def _result(runner, ss, results):
    res = runner.result or {}
    out = res.get("outcome")
    if res.get("mode") == "benchmark":
        _benchmark_result(res)
    elif out == "completed":
        a = res.get("assessment") or {}
        st.success(f"Calibration finished: {a.get('headline')} ({a.get('level')})")
        m = res.get("model") or {}
        mt = (a.get("metrics") or {})
        st.table([{"Check": "Board poses used", "Result": f"{mt.get('pairs_used')} (discarded {mt.get('pairs_discarded')})"},
                  {"Check": "Stereo reprojection error", "Result": f"{mt.get('rms_stereo_px'):.3f} px" if mt.get("rms_stereo_px") is not None else "—"},
                  {"Check": "Calibrated baseline", "Result": f"{m.get('baseline_mm'):.2f} mm" if m.get("baseline_mm") else "—"},
                  {"Check": "Time offset (frames)", "Result": str(mt.get("time_offset_frames"))}])
        for p in a.get("concerns", []):
            st.markdown(f"⚠️ {p}")
    else:
        st.error(f"{out}: {res.get('error') or 'The calibration did not complete.'}")
    cl = res.get("cleanup") or {}
    ps = res.get("persistence") or {}
    if ps.get("saved"):
        st.success(f"Result saved on the server ({ps.get('bytes')} bytes, SHA-256 {str(ps.get('sha256'))[:12]}…). Download it now: the server's disk is temporary.")
        _download(results, runner.job_code)
    else:
        st.error(f"The result could NOT be saved on the server ({ps.get('error')}). The videos were kept so you can retry.")
    st.caption("Recovery code (private; keep it to download this result again from another session):"); _show_code(runner.job_code, "done")
    if res.get("mode") != "benchmark" and cl.get("videos_kept_for_next_step"):
        st.warning(f"Your videos were kept for {((cl.get('kept_for_retry_s') or 0) / 60):.0f} min so you can retry without uploading again. “Lock this session” deletes them now.")
    st.caption(f"Cleanup: videos deleted = {cl.get('videos_deleted')} · job folder removed = {cl.get('job_dir_removed')}")
    r = res.get("resources") or {}
    if r:
        st.markdown("**Stage-by-stage measurements**")
        st.dataframe([{k: v for k, v in s.items() if k in ("stage", "wall_s", "parent_rss_max_mb", "worker_rss_max_mb", "tree_rss_max_mb", "tree_pss_max_mb", "cgroup_memory_max_mb",
                                                          "cpu_s", "throttled_periods", "temp_disk_max_mb")} for s in r.get("stages", [])], hide_index=True)
        st.json({"peak": r.get("peak"), "cgroup": r.get("cgroup"), "available": r.get("available")}, expanded=False)
    with st.expander("Environment and validation"):
        st.json({"environment": res.get("environment"), "validation": res.get("validation"), "worker_env": res.get("env"), "f7": res.get("f7")}, expanded=False)
    d = os.environ.get("F8_METRICS_DIR")
    if d and not ss.get("dual_saved"):                                  # optional local record (no detections, no names)
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
            slim = {k: v for k, v in res.items() if k != "detections"}
            (Path(d) / f"calibration_{int(time.time())}.json").write_text(json.dumps(slim, indent=1, default=str)); ss["dual_saved"] = True
        except OSError:
            pass

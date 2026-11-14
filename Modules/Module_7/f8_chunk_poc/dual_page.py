"""Stage F8.5: local test page for the dual-video calibration (Streamlit drawing only; all rules live in chunk_store / calibration_runner).

Reached through the same passcode gate as the rest of the F8 app. Not part of Module 7. Two upload slots (LEFT, RIGHT), one Run button, live progress,
Cancel, and the result with stage-by-stage resource measurements."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st

from calibration_runner import CalibrationJobRunner, RunnerLimits
from chunk_store import MiB, SLOTS, UploadError
import component
from protocol import handle

LIMITS = RunnerLimits.from_env()                                       # F8_CALIB_TIMEOUT_S, F8_BENCH_*, F8_ENABLE_FULL_CALIBRATION


def _slot_line(info, slot):
    if not info:
        return f"{slot.upper()}: no video yet"
    if info["state"] == "complete":
        return f"{slot.upper()}: ✅ {info['size'] / MiB:.1f} MB verified · SHA-256 {info['sha256'][:12]}…" + (" · in use by the calibration" if info["leased"] else "")
    return f"{slot.upper()}: uploading {info['next_index']}/{info['n_chunks']} chunks"


def render(store, ss, lim):
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
        store.release_owner(owner); ss["runner"] = None; ss["dual_epoch"] += 1; ss["authed"] = False
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
        ss["runner"] = CalibrationJobRunner(store, owner, settings, LIMITS, mode="benchmark", benchmark_frames=int(bench_n))
        ss["runner"].start(); ss["dual_saved"] = False
        st.rerun()
    if LIMITS.allow_full:
        sure = c[3].checkbox("I understand a full calibration can take many minutes and deletes both videos afterwards", key="dual_full_ok")
        if c[1].button("Run full calibration", disabled=blocked or not sure):
            ss["runner"] = CalibrationJobRunner(store, owner, settings, LIMITS, mode="calibrate")
            ss["runner"].start(); ss["dual_saved"] = False
            st.rerun()
    else:
        c[3].caption("Full calibration is disabled on this server.")
    if c[2].button("Cancel", disabled=not running):
        runner.cancel()

    runner = ss["runner"]
    if runner is not None:
        s = runner.status()
        if not runner.done:
            st.progress(min(s["fraction"], 0.99), text=f"{s['stage'].replace('_', ' ')} · {s['elapsed_s']:.0f} s")
            m = s.get("monitor") or {}
            st.caption(f"parent RSS {m.get('parent_rss_mb')} MB · worker RSS {m.get('worker_rss_mb')} MB · temp disk {m.get('temp_disk_mb')} MB")
            time.sleep(1.0); st.rerun()
        else:
            _result(runner, ss)
            kept = (runner.result or {}).get("cleanup", {}).get("videos_kept_for_next_step")
            if st.button("Clear result" if kept else "Start over"):
                ss["runner"] = None
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


def _result(runner, ss):
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

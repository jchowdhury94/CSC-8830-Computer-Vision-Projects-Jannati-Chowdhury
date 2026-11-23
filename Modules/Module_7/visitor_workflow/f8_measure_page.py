"""Measurement page (Streamlit drawing only): a short readiness summary, ONE gated action (Calculate Wall Dimensions) and a results-only page.

Every rule lives elsewhere: readiness and the gate in ``f8_measurement`` (the backend's own input check), numbers and wording in ``f8_result_view`` / ``visitor_messages``.
This module imports neither the backend, nor Method J, nor the adapter.  The only call that can start a measurement is ``svc.measurement.start`` inside ``_start`` below, which runs
as the click callback of the one Calculate button (so it executes once per click) and which refuses while ``ROOM_PROCESSING_ENABLED`` is False.  There is no other button, route,
query parameter, environment switch or callback.

The results page is organised around ``view["dimensions"]`` (a list of measurements): it draws whatever the view model yields, so a further measurement can be added in
``f8_result_view.MEASUREMENTS`` without redesigning the page.  Nothing about a measurement that does not exist is drawn.
"""
from __future__ import annotations

import json
import time

import streamlit as st

import f8_measurement as M
import f8_result_view as RVW
import f8_ui_text as UT
import f8_walls as W
import room_videos as RV

ACCENT = "#C9363E"
TONE_COLOR = {"good": "#2E9E5B", "warn": "#E0A526", "bad": "#C9363E", "info": "#4C8DD6", "neutral": "#9AA1AC"}
CSS = f"""<style>
.f8m-card {{ border: 1px solid #2A2E37; border-left: 5px solid {ACCENT}; border-radius: 10px; padding: 1rem 1.2rem; background: #14171C; height: 100%; }}
.f8m-label {{ color: #B8BDC7; font-size: 0.95rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }}
.f8m-value {{ font-size: 3rem; font-weight: 800; line-height: 1.05; color: #FFFFFF; }}
.f8m-sub {{ color: #B8BDC7; font-size: 1.1rem; }}
.f8m-title {{ font-size: 2rem; font-weight: 800; margin: 0.2rem 0 0.8rem; }}
.f8m-note {{ color: #B8BDC7; margin: 0.9rem 0 0.6rem; }}
.f8m-warn {{ border-left: 4px solid #E0A526; padding: 0.4rem 0.9rem; margin: 0.4rem 0; background: #1B1A14; border-radius: 6px; }}
/* Download Results: a compact, prominent red button (fits its label; slightly taller than a default button) */
.st-key-f8i_m_dl button {{ min-height: 3rem; padding: 0.4rem 1.4rem; background-color: {ACCENT}; border: 2px solid {ACCENT}; color: #FFFFFF; border-radius: 10px; }}
.st-key-f8i_m_dl button p {{ font-size: 1.1rem; font-weight: 700; color: #FFFFFF; }}
.st-key-f8i_m_dl button:hover {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
.st-key-f8i_m_second button {{ min-height: 3.4rem; padding: 0.5rem 1.8rem; background-color: {ACCENT}; border: 2px solid {ACCENT}; color: #FFFFFF; border-radius: 10px; }}
.st-key-f8i_m_second button p {{ font-size: 1.2rem; font-weight: 800; color: #FFFFFF; }}
.st-key-f8i_m_second button:hover {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
.f8m-src {{ color: #9AA1AC; font-size: 0.9rem; margin-top: 0.3rem; }}
.st-key-f8i_m_actions button {{ min-height: 3rem; }}
.st-key-f8i_m_actions > div {{ flex: 0 0 auto; width: auto; }}
</style>"""
STEP_TEXT = {"home": "calibration", "ready": "calibration", "room": "room videos", "frames": "frames and boundaries"}
STEP_NAMES = RVW.STEP_LABEL
CALCULATE = "Calculate Wall Dimensions"
RECALCULATE = "Recalculate Wall Dimensions"
CALCULATE_B = "Calculate Room Length"
RECALCULATE_B = "Recalculate Room Length"
CHANGED_NOTE = "Your inputs changed, so the previous result was cleared. Calculate again to use the updated inputs."


def _go(sess, step):
    try:
        W.go(sess, st.session_state, step)
    except Exception as e:                                    # ServiceError: shown on the next render, never a traceback
        st.session_state["f8i_msg"] = str(e)


def _badge(label, tone):
    st.markdown(f"<span class='f8i-badge' style='background:{TONE_COLOR[tone]}'>{label}</span>", unsafe_allow_html=True)


def _table(rows):
    if rows:
        st.table([{"Item": a, "Value": b} for a, b in rows])


# ------------------------------------------------------------------ readiness (short)
def _technical_inputs(rd, ctx):
    """Developer-level detail about the inputs (calibration id, hashes, warnings, per-item state): ONE collapsed section."""
    s = rd.summary
    if s:
        rows = [("Calibration id", str(s.get("calibration_id", "—"))), ("Quality", str(s.get("quality", "—"))),
                ("Camera ordering", " → ".join(str(c).title() for c in s.get("camera_order", [])) or "—"),
                ("Calibration image size", " × ".join(str(v) for v in s.get("calibration_size_wh", [])) + " px"),
                ("Baseline", f"{s['baseline_mm']:g} mm ({UT.baseline_phrase((ctx or {}).get('baseline_source'))})" if "baseline_mm" in s else "—")]
        for slot in (RV.ROOM_UPLOAD_SLOTS_3 if s.get("n_viewpoints") == 3 else RV.ROOM_UPLOAD_SLOTS):
            v = (s.get("videos") or {}).get(slot)
            rows.append((RV.SLOT_INFO[slot][2], "missing or expired" if v is None else f"{v['size_wh'][0]} × {v['size_wh'][1]} px · {v['frames']} frames"))
        rows.append(("Frame sets usable", f"{s.get('frame_sets_usable', 0)} of {s.get('n_frame_sets', 0)}"))
        if s.get("n_viewpoints") == 3:
            rows.append(("Method", "three-viewpoint Wall B (fast matcher by default)"))
        _table(rows)
        vids = s.get("videos") or {}
        if any(vids.values()):
            st.table([{"Video": RV.SLOT_INFO[k][2], "SHA-256": v["sha256"]} for k, v in vids.items() if v])
    for m in rd.warnings:
        st.caption(m)


def _readiness(sess, rd, ctx=None):
    """One short summary line; unmet requirements (real problems) are listed with a button to the step that fixes them."""
    s = rd.summary or {}
    if rd.ready:
        n = int(s.get("n_frame_sets", 0))
        if s.get("n_viewpoints") == 3:
            st.success(f"Ready to calculate · calibration accepted · 6 videos (3 viewpoints) · {s.get('frame_sets_usable', 0)} usable frame sets · {s.get('verticals_marked', 0)} wall corner lines")
        else:
            st.success(f"Ready to calculate · calibration accepted · 4 videos · {n} frame sets · {s.get('seeds_marked', 0)} wall/ceiling boundaries")
    else:
        for item in rd.items:
            if item.ok:
                continue
            st.markdown(UT.line("bad", f"**{item.label}** · {item.detail}"))
            for p in item.problems:
                st.markdown(f"&nbsp;&nbsp;&nbsp;{p}")
            if item.step:
                st.button(f"Fix this: go to the {STEP_TEXT.get(item.step, item.step)}", key=f"f8i_m_fix_{item.key}", on_click=_go, args=(sess, item.step))
        status_text = {"current": "", "incomplete": "Some inputs are missing.", "expired": "An uploaded video has expired and must be uploaded again.",
                       "replaced": "A video was replaced after the frames were chosen; choose the frames again.", "calibration_changed": "The calibration changed after the frames were chosen.",
                       "not_accepted": "No calibration is accepted."}[rd.input_status]
        if status_text:
            st.info(status_text)
    with st.expander("Technical Details"):
        _technical_inputs(rd, ctx)


# ------------------------------------------------------------------ action
def _start(svc, sess):
    """Click callback of the ONE calculate button: runs once per click (a rerun never starts anything).  A refusal is shown on the next render."""
    try:
        svc.measurement.start(sess)
    except M.MeasurementError as e:
        st.session_state["f8i_m_err"] = str(e)
    else:
        st.session_state.pop("f8i_m_recalc", None)
        st.session_state.pop("f8i_m_err", None)


def _action(svc, sess, rd):
    if st.session_state.get("f8i_m_err"):
        st.error(st.session_state.pop("f8i_m_err"))
    second = sess.active_wall == "B"
    label = (RECALCULATE_B if second else RECALCULATE) if st.session_state.get("f8i_m_recalc") else (CALCULATE_B if second else CALCULATE)
    if not rd.gate_open:
        st.info(RVW.GATE_MESSAGE)
        st.button(label, key="f8i_m_start_off", disabled=True, type="primary")                    # no handler: the gate is a code constant, not a UI state
        return
    st.button(label, key="f8i_m_start", type="primary", disabled=not rd.ready or sess.measurement.running, on_click=_start, args=(svc, sess))


def _running(svc, sess):
    job = sess.measurement.job
    label, _, sentence = RVW.STATE_TEXT[RVW.PROCESSING if job.stage == "processing" else RVW.VALIDATING]
    _badge(label, "info")
    st.caption(f"{sentence} · {job.elapsed():.0f} s elapsed")
    with st.spinner(f"{job.stage.capitalize()}…"):
        if st.button("Cancel the measurement", key="f8i_m_cancel"):
            svc.measurement.cancel(sess)
        if st.session_state.get("f8i_m_autorefresh", True):
            time.sleep(1.0)
    if st.session_state.get("f8i_m_autorefresh", True):
        st.rerun()


# ------------------------------------------------------------------ results
def _metric(col, label, dim, src=None):
    with col:
        if dim is None:
            st.markdown(f"<div class='f8m-card'><div class='f8m-label'>{label}</div><div class='f8m-value'>—</div><div class='f8m-sub'>not available</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='f8m-card'><div class='f8m-label'>{label}</div><div class='f8m-value'>{dim['cm_text']}</div><div class='f8m-sub'>{dim['m_text']}</div>{f"<div class='f8m-src'>{src}</div>" if src else ""}</div>", unsafe_allow_html=True)


def _ask_new(): st.session_state["f8i_m_new_ask"] = True
def _cancel_new(): st.session_state.pop("f8i_m_new_ask", None)


def _do_new(svc, sess):
    st.session_state.pop("f8i_m_new_ask", None)
    sess.reset(svc.store, svc.room_store)                         # the same cleanup path as Start over: deletes this session's uploads, frames, boundaries and result


def _render_three_view(t):
    """The quality of a three-viewpoint Wall B result, shown in the open: the headline, every frame set's classification, every warning and failure with its corrective action, the indicators and
    the owner's limitations statement.  Nothing is hidden behind a collapsed section (warnings accompany the number)."""
    if t is None:
        return
    st.markdown(f"**{t['headline']}** · 3 viewpoints · {t['frames_usable']} of {t['n_frame_sets']} frame sets used")
    st.caption(t["method"])
    if t["excluded"]:
        st.markdown(UT.line("warn", "Frame set(s) " + ", ".join(map(str, t["excluded"])) + " were not used because their four vertical corners were not all marked."))
    for f in t["frames"]:
        head = f"**Frame set {f['frame_set']}** · " + {"OK": "no quality warning", "WARNINGS": "usable, with warnings", "FAILED": "not usable"}.get(f["status"], f["status"]) + ("" if f["length"] is None else f" · {f['length']['cm_text_precise']}")
        st.markdown(UT.line({"OK": "ok", "WARNINGS": "warn", "FAILED": "bad"}.get(f["status"], "warn"), head))
        for x in f["hard_failures"]:
            st.markdown("&nbsp;&nbsp;&nbsp;" + UT.line("bad", f"{x['meaning'] or x['message']} {x['next_step'] or ''}".strip()))
        for x in f["warnings"]:
            st.markdown("&nbsp;&nbsp;&nbsp;" + UT.line("warn", f"{x['meaning'] or x['message']} ({x['message']}) {x['next_step'] or ''}".strip()))
    for x in t["aggregate_warnings"]:
        st.markdown(UT.line("warn", f"{x['meaning'] or x['message']} {x['message']} {x['next_step'] or ''}".strip()))
    if t["message"] and not t["ok"]:
        st.markdown(UT.line("bad", t["message"]))
    if t["indicators"]:
        with st.expander("Quality indicators"):
            _table(t["indicators"])
            st.caption(t["quality_note"])
    if t["limitations"]:
        st.markdown(f"<div class='f8m-note'><b>Measurement Limitations</b><br>{t['limitations']}</div>", unsafe_allow_html=True)


def _per_set_table(v):
    rows = []
    for s in v["per_set"]:
        row = {"Frame set": s["frame_set"]}
        for m in v["dimensions"]:
            d = (s.get("dims") or {}).get(m["key"])
            row[m["label"].replace("Estimated ", "")] = "—" if d is None else d["cm_text_precise"]
        if s["status"] != "reconstructed":
            row["Note"] = s.get("failure") or "Not used"
        rows.append(row)
    return rows


def _second_wall(sess):
    try:
        W.start_second_wall(sess, st.session_state)
    except Exception as e:
        st.session_state["f8i_msg"] = str(e)


def _view_wall(sess, wall):
    try:
        W.switch_wall(sess, st.session_state, wall, "measure")
    except Exception as e:
        st.session_state["f8i_msg"] = str(e)


def _change_wall(sess, wall):
    try:
        W.switch_wall(sess, st.session_state, wall, "frames")
    except Exception as e:
        st.session_state["f8i_msg"] = str(e)


def _actions(svc, sess, doc, *, second_wall=False):
    if second_wall:
        with st.container(key="f8i_m_second_box"):
            st.caption(W.SECOND_WALL_HINT)
            st.button(W.SECOND_WALL_ACTION, key="f8i_m_second", icon=":material/arrow_forward:", on_click=_second_wall, args=(sess,), width="content")
    with st.container(horizontal=True, vertical_alignment="center", key="f8i_m_actions"):
        if doc is not None:
            st.download_button("Download Results", json.dumps(doc, indent=1).encode(), file_name="room_measurement_summary.json", mime="application/json", key="f8i_m_dl",
                               type="primary", icon=":material/download:", width="content", help="Numbers only. No video or frames are included.")
        st.button("Change Boundaries", key="f8i_m_change", icon=":material/edit:", on_click=_go, args=(sess, "frames"), help="Return to the boundary editor. Your uploaded videos are kept.")
        st.button("New Measurement", key="f8i_m_new", icon=":material/add:", on_click=_ask_new)
    if st.session_state.get("f8i_m_new_ask"):
        st.warning("Start a new measurement? This deletes the calibration, uploaded videos, frames, boundaries and result of this session. Download your results first if you want to keep them.")
        c = st.columns(2)
        c[0].button("Yes, start a new measurement", key="f8i_m_new_yes", type="primary", width="stretch", on_click=_do_new, args=(svc, sess))
        c[1].button("Cancel", key="f8i_m_new_no", width="stretch", on_click=_cancel_new)


def _resource_use(v):
    """Read-only 'Resource Use' section of Technical Details (the numbers of this calculation, from the result's own record; nothing is measured here)."""
    rows = v.get("resources")
    if rows:
        st.markdown("**Resource Use**")
        _table(rows)


def _technical_result(v):
    with st.expander("Technical Details"):
        _technical_body(v)


def _technical_body(v):
    if True:
        for w in v["warnings"]:
            st.markdown(f"<div class='f8m-warn'><b>{w['meaning']}</b><br>Why it matters: {w['reliability']}<br>What you can do: {w['action']}<br><small>{w['code']}</small></div>", unsafe_allow_html=True)
        if v["notice"]:
            st.caption(v["notice"])
        if v["seconds"] is not None:
            st.caption(f"Processing time: {v['seconds']:.1f} s")
        _table(v["pose"] and [("Pose · " + a, b) for a, b in v["pose"]])
        _table(v["registration"] and [("Registration · " + a, b) for a, b in v["registration"]])
        _table(v["provenance"])
        _table(v.get("technical"))
        _resource_use(v)
        st.caption(v["disclaimer"])


def render_wall_a_done(svc, sess, result):
    """After Wall A alone: NOT a final page.  Step 1 of 2 is done; the one prominent action is to continue to Wall B."""
    src = (svc.calibration_context(sess) or {}).get("baseline_source")
    v = RVW.build_view(result, src)
    st.markdown(f"<div class='f8m-title'>{W.WALL_A_DONE_TITLE}</div>", unsafe_allow_html=True)
    cols = st.columns(2, gap="medium")
    _metric(cols[0], "Room Width", v["width"])
    _metric(cols[1], "Room Height", v["height"])
    st.markdown(f"<div class='f8m-note'>{W.WALL_A_DONE_TEXT}</div>", unsafe_allow_html=True)
    with st.container(key="f8i_m_second"):
        st.button(W.SECOND_WALL_ACTION, key="f8i_m_second_btn", icon=":material/arrow_forward:", type="primary", on_click=_second_wall, args=(sess,), width="stretch")
    with st.expander("Wall A Details"):
        st.caption(v["estimate_note"])
        if v["warnings"] or v["notice"]:
            st.caption("Some automatic checks raised notes. They are listed below; they do not stop the estimate from being shown.")
        if v["partial"]:
            st.markdown(UT.line("warn", "Only one dimension could be computed."))
        if v["per_set"]:
            st.table(_per_set_table(v))
        _technical_body(v)
        doc = RVW.summary_document(result, src)
        if doc is not None:
            st.download_button("Download Wall A Results", json.dumps(doc, indent=1).encode(), file_name="room_measurement_wall_a.json", mime="application/json", key="f8i_m_dl_a",
                               icon=":material/download:", width="content", help="Numbers only. No video or frames are included.")
    with st.container(horizontal=True, vertical_alignment="center", key="f8i_m_actions"):
        st.button("Change Boundaries", key="f8i_m_change", icon=":material/edit:", on_click=_go, args=(sess, "frames"), help="Return to the boundary editor. Your uploaded videos are kept.")
        st.button("New Measurement", key="f8i_m_new", icon=":material/add:", on_click=_ask_new)
    if st.session_state.get("f8i_m_new_ask"):
        st.warning("Start a new measurement? This deletes the calibration, uploaded videos, frames, boundaries and result of this session.")
        c = st.columns(2)
        c[0].button("Yes, start a new measurement", key="f8i_m_new_yes", type="primary", width="stretch", on_click=_do_new, args=(svc, sess))
        c[1].button("Cancel", key="f8i_m_new_no", width="stretch", on_click=_cancel_new)


def render_result(svc, sess, result, *, second_wall=False):
    src = (svc.calibration_context(sess) or {}).get("baseline_source")
    v = RVW.build_view(result, src)
    if not v["accepted"]:
        return _render_failure(svc, sess, v)
    st.markdown(f"<div class='f8m-title'>{v['title']}</div>", unsafe_allow_html=True)
    dims = v["dimensions"]
    cols = st.columns(max(len(dims), 1), gap="medium")
    for col, m in zip(cols, dims):
        _metric(col, m["label"], m["value"])
    if v["three_view"] is not None:
        _render_three_view(v["three_view"])
    else:
        if v["partial"]:
            st.markdown(UT.line("warn", "Only one dimension could be computed."))
        st.markdown(f"<div class='f8m-note'>{v['estimate_note']}</div>", unsafe_allow_html=True)
        if v["warnings"] or v["notice"]:
            st.caption("Some automatic checks raised notes. They are listed under Technical Details.")
    if v["per_set"]:
        with st.expander("Measurements per Frame Set"):
            st.table(_per_set_table(v))
    _actions(svc, sess, RVW.summary_document(result, src), second_wall=second_wall)
    _technical_result(v)


def _render_failure(svc, sess, v):
    _badge(v["label"], v["tone"])
    st.caption(v["sentence"])
    f = v["failure"]
    st.error(f"{f['what']}" + (f" {f['next_step']}" if f["next_step"] else ""))
    st.markdown("No dimension is reported, because nothing was accepted as a measurement.")
    if v.get("three_view") is not None:
        _render_three_view(v["three_view"])                              # the frame-level failures are shown, never hidden
    fb = f.get("feedback")
    if fb:
        st.caption(f"Affected: {fb['affected']} · Suggested: {fb['action'] if fb['action'] else '—'} · Record again: {fb['record_again']}.")
    r = f["retry"]
    if r["kind"] == "fix":
        for n, a in enumerate(f["actions"] or [r]):                       # the typed code decides the destination(s); the displayed advice is not parsed
            st.button(a["label"], key=f"f8i_m_retry_fix_{n}", type="primary" if n == 0 else "secondary", on_click=_go, args=(sess, a["step"]))
    elif not svc.measurement.gate_open(sess):
        st.caption("Measuring is not available on this server at the moment, so there is nothing to retry yet.")
    else:
        st.button(r["label"], key="f8i_m_retry", type="primary", disabled=sess.measurement.running, on_click=_start, args=(svc, sess))
    with st.expander("Technical Details"):
        st.table([{"Code": t["code"], "Class": t["failure_class"], "Stage": t["stage"], "Message": t["message"]} for t in f["technical"]])
        _table(v["provenance"])
        _table(v.get("technical"))
        _resource_use(v)
    st.caption(v["disclaimer"])


# ------------------------------------------------------------------ two walls
def _accepted(r) -> bool:
    return r is not None and RVW.result_state(r) in (RVW.DONE_OK, RVW.DONE_WARN)


def _wall_result(svc, sess, wall):
    """The CURRENT result of ``wall`` (None when it has none or its inputs changed).  Reading the other wall swaps its data in and back; no widget state is touched."""
    if wall == sess.active_wall:
        return svc.measurement.current(sess)
    with sess.on_wall(wall):
        return svc.measurement.current(sess)


def _wall_a_strip(svc, sess, src):
    """Shown on the second wall's page: the first wall's saved result (never lost by anything that happens to Wall B) and its download."""
    ra = _wall_result(svc, sess, "A")
    if not _accepted(ra):
        return
    va = RVW.build_view(ra, src)
    w, h = va["width"], va["height"]
    st.success("Wall A measured · " + " · ".join(f"{lab} {d['cm_text']}" for lab, d in (("Room Width", w), ("Room Height", h)) if d))
    with st.container(horizontal=True, vertical_alignment="center", key="f8i_m_actions_a"):
        doc = RVW.summary_document(ra, src)
        if doc is not None:
            st.download_button("Download Wall A Results", json.dumps(doc, indent=1).encode(), file_name="room_measurement_wall_a.json", mime="application/json", key="f8i_m_dl_a",
                               icon=":material/download:", width="content", help="Numbers only. No video or frames are included.")
        st.button("View Wall A", key="f8i_m_view_a", on_click=_view_wall, args=(sess, "A"))


def render_room(svc, sess, ra, rb):
    src = (svc.calibration_context(sess) or {}).get("baseline_source")
    rv = RVW.room_view(ra, rb, src)
    st.markdown(f"<div class='f8m-title'>{rv['title']}</div>", unsafe_allow_html=True)
    dims = rv["dimensions"]
    cols = st.columns(max(len(dims), 1), gap="medium")
    for col, m in zip(cols, dims):
        _metric(col, m["label"], m["value"], f"From Wall {m['wall']}")
    st.markdown(f"<div class='f8m-note'><b>{rv['limitations_title']}</b><br>{rv['limitations']}</div>", unsafe_allow_html=True)
    if rv.get("wall_b_three_view") is not None:                                   # Wall B used three viewpoints: its approximate status and warnings are shown with the room dimensions
        st.markdown("<div class='f8m-label'>Wall B (room length)</div>", unsafe_allow_html=True)
        _render_three_view(dict(rv["wall_b_three_view"], limitations=""))
    with st.expander("Measurements per Wall"):
        st.table([{"Measurement": d["label"], "From": f"Wall {d['wall']}", "Value": d["value"]["cm_text_precise"]} for d in dims])     # Wall B's own height is never listed as a room dimension
        st.caption("Room Width and Room Height come from Wall A; Room Length is the horizontal measurement of Wall B. Nothing is averaged.")
    doc = RVW.room_summary_document(ra, rb, src)
    with st.container(horizontal=True, vertical_alignment="center", key="f8i_m_actions"):
        if doc is not None:
            st.download_button("Download Results", json.dumps(doc, indent=1).encode(), file_name="room_measurement_summary.json", mime="application/json", key="f8i_m_dl",
                               type="primary", icon=":material/download:", width="content", help="Numbers only. No video or frames are included.")
        st.button("Change Wall A Boundaries", key="f8i_m_change_a", icon=":material/edit:", on_click=_change_wall, args=(sess, "A"))
        st.button("Change Wall B Boundaries", key="f8i_m_change_b", icon=":material/edit:", on_click=_change_wall, args=(sess, "B"))
        st.button("New Measurement", key="f8i_m_new", icon=":material/add:", on_click=_ask_new)
    if st.session_state.get("f8i_m_new_ask"):
        st.warning("Start a new measurement? This deletes the calibration, uploaded videos, frames, boundaries and results of both walls. Download your results first if you want to keep them.")
        c = st.columns(2)
        c[0].button("Yes, start a new measurement", key="f8i_m_new_yes", type="primary", width="stretch", on_click=_do_new, args=(svc, sess))
        c[1].button("Cancel", key="f8i_m_new_no", width="stretch", on_click=_cancel_new)
    with st.expander("Technical Details"):
        for name, v in (("Wall A (room width and height)", rv["wall_a"]), ("Wall B (room length)", rv["wall_b"])):
            st.markdown(f"**{name}**")
            _technical_body(v)


# ------------------------------------------------------------------ the page
def render_measure(svc, sess):
    st.html(CSS)
    rec = sess.measurement
    if svc.calibration_context(sess) is None:
        st.warning("There is no accepted calibration in this session.")
        st.button("Go to the calibration", key="f8i_ms_cal", on_click=_go, args=(sess, "home"))
        return
    if rec.running:
        _running(svc, sess)
        return
    result = svc.measurement.current(sess)                # may discard a stale result (and add a note), so it runs before the notes are shown
    if rec.notes:
        st.session_state["f8i_m_recalc"] = True
    for n in list(rec.notes):
        st.info(CHANGED_NOTE if "changed" in n or "discarded" in n else n)
    rec.notes.clear()
    second = sess.active_wall == "B"
    other = _wall_result(svc, sess, "A" if second else "B")
    if _accepted(result) and _accepted(other):
        render_room(svc, sess, *((other, result) if second else (result, other)))          # both walls measured: the room page
        return
    if second:
        src = (svc.calibration_context(sess) or {}).get("baseline_source")
        if _accepted(result):                                                              # Wall B is done but Wall A has no current result
            st.info("Wall B was measured. Calculate Wall A again (its boundaries or videos changed) to see the room dimensions.")
            st.button("View Wall A", key="f8i_m_view_a", on_click=_view_wall, args=(sess, "A"))
            return
        _wall_a_strip(svc, sess, src)
    elif _accepted(result):
        render_wall_a_done(svc, sess, result)                    # Step 1 of 2 only: not a final page
        return
    if result is not None:
        render_result(svc, sess, result)                  # a failed measurement: the actual reason and the way to fix it
        return
    rd = svc.measurement.readiness(sess)
    st.markdown(f"<div class='f8m-title'>{'Calculate Room Length' if second else 'Calculate Wall Dimensions'}</div>", unsafe_allow_html=True)
    _readiness(sess, rd, svc.calibration_context(sess))
    _action(svc, sess, rd)

"""Stage F8.18 pages: the LOCAL visitor calibration integration (Streamlit drawing only; every rule lives in ``f8_visitor_service``).

Session-state namespace: every key this module creates starts with ``f8i_`` (the F6 app uses ``f6*`` / ``f7*``, the F8 proof of concept ``f8_`` / ``dual_``).
Nothing here starts a job on a rerun: a calibration starts only inside the ``Start calibration`` button handler.
Room VIDEO UPLOAD (F8.31A), frames and boundaries (F8.31B), readiness and results (F8.31C-2A) exist; the measurement action is gated per wall (WALL_A_PROCESSING_ENABLED, WALL_B_PROCESSING_ENABLED)."""
from __future__ import annotations

import json
import time

import streamlit as st

import component
import f8_visitor_service as S
import room_videos as RV
import f8_frames_page as FP
import f8_measure_page as MP
import f8_nav as NAV
import f8_walls as W
import f8_viewpoints as VPC
import f8_guided_steps as GS
import f8_guided_text as GT
import f8_ui_text as UT
import quality_feedback as QF
from chunk_store import MiB, SLOTS
from protocol import handle

ACCENT = "#C9363E"
GOOD_C, WARN_C, BAD_C, NEUTRAL_C = "#2E9E5B", "#E0A526", "#C9363E", "#9AA1AC"
LEVEL_STYLE = {S.GOOD: ("Good", GOOD_C), S.QUESTIONABLE: ("Questionable but usable", WARN_C), S.FAILED: ("Failed", BAD_C), S.NOT_ASSESSED: ("Quality not assessed", NEUTRAL_C),
               S.INVALID: ("Cannot be used", BAD_C)}
NOT_ASSESSED_TEXT = "This file has no quality information, so how good the calibration is could not be judged. Only its numbers were checked, and they are valid."
KIND_TEXT = UT.KIND_TEXT                         # F8.31D-2: plain names for the saved-calibration kinds
BASELINE_TEXT = UT.BASELINE_TEXT
CSS = f"""<style>
.f8i-title {{ text-align: center; font-size: 2.1rem; font-weight: 800; margin: 0.2rem 0 0.1rem; line-height: 1.15; }}
.f8i-sub {{ text-align: center; color: #B8BDC7; margin: 0 0 1rem; }}
.f8i-progress {{ max-width: 800px; margin: 0.6rem auto 1.4rem; }}
.f8i-progress-label {{ color: #B8BDC7; font-size: 1rem; font-weight: 600; margin-bottom: 0.4rem; text-align: center; }}
.f8i-flow {{ display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 0.7rem 1rem; margin: 0.4rem auto 1rem; font-weight: 800; }}
.f8i-flow .f8i-fl {{ padding: 0.7rem 1.7rem; min-height: 3rem; display: inline-flex; align-items: center; border-radius: 999px; border: 2px solid #2A2E37; color: #9AA1AC; background: #14171C; font-size: 1.3rem; line-height: 1.2; }}
.f8i-flow .f8i-fl.on {{ border-color: {ACCENT}; color: #FFFFFF; background: {ACCENT}; }}
.f8i-flow .f8i-fl.done {{ border-color: #2E9E5B; color: #FFFFFF; background: transparent; }}
.f8i-flow .f8i-arrow {{ color: #9AA1AC; font-size: 1.9rem; line-height: 1; }}
.f8i-wallstep {{ text-align: center; font-size: 1.3rem; font-weight: 700; margin: 0 0 0.5rem; }}
@media (max-width: 640px) {{
    .f8i-flow {{ gap: 0.55rem 0.7rem; }}
    .f8i-flow .f8i-fl {{ padding: 0.6rem 1.2rem; font-size: 1.15rem; min-height: 2.8rem; }}
    .f8i-flow .f8i-arrow {{ font-size: 1.6rem; }}
    .f8i-wallstep {{ font-size: 1.15rem; }}
}}
.f8i-bar {{ height: 6px; border-radius: 999px; background: #2A2E37; overflow: hidden; }}
.f8i-fill {{ height: 100%; background: {ACCENT}; border-radius: 999px; }}
.f8i-badge {{ display: inline-block; padding: 0.15rem 0.7rem; border-radius: 999px; font-weight: 700; font-size: 0.9rem; color: #fff; }}
.f8i-big {{ font-size: 2.4rem; font-weight: 800; line-height: 1.1; }}
button[kind="primary"] {{ background-color: {ACCENT}; border-color: {ACCENT}; color: #FFFFFF; }}
button[kind="primary"]:hover {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
button[kind="primary"]:disabled {{ background-color: rgba(201, 54, 62, 0.35); border-color: transparent; color: rgba(255, 255, 255, 0.6); }}
.stMainBlockContainer, .block-container {{ max-width: 900px !important; margin: 0 auto; }}
:root {{ color-scheme: dark; }}
.f8i-sub {{ margin-bottom: 0.4rem; }}
.f8i-pagetitle {{ font-size: 1.35rem; font-weight: 700; margin: 0.2rem 0 0.4rem; }}
.f8i-limitnote {{ font-size: 0.9rem; line-height: 1.45; opacity: 0.85; margin: 0.4rem 0 0.6rem; padding: 0.5rem 0.75rem; border-left: 3px solid {ACCENT}; background: rgba(201, 54, 62, 0.08); border-radius: 0 6px 6px 0; }}
/* F8.31D-3B guided steps: locked steps stay visible but dimmed; no interactive element exists inside them. */
/* F8.31E: the status line of an upload card keeps the remove (×) control at its right edge, also on narrow screens */
[class*="st-key-f8i_cardrow_"] > div:first-child {{ flex: 1 1 auto; min-width: 0; overflow-wrap: anywhere; }}
[class*="st-key-f8i_cardrow_"] > div:not(:first-child) {{ flex: 0 0 auto; }}
/* F8.31F: remove-video control: a 34 px red-outlined circle with a white x, at the top-right of the card row (always one line, also at 390 px) */
[class*="st-key-f8i_room_rm_p"] {{ width: 34px; flex: 0 0 34px; }}
[class*="st-key-f8i_room_rm_p"] button {{ width: 34px; height: 34px; min-height: 34px; min-width: 34px; padding: 0; border-radius: 50%; border: 2px solid {ACCENT}; background: transparent; color: #FFFFFF; font-size: 1.3rem; font-weight: 700; line-height: 1; display: flex; align-items: center; justify-content: center; }}
[class*="st-key-f8i_room_rm_p"] button:hover, [class*="st-key-f8i_room_rm_p"] button:focus-visible {{ background: {ACCENT}; border-color: {ACCENT}; color: #FFFFFF; }}
[class*="st-key-f8i_room_rm_p"] button p {{ margin: 0; line-height: 1; font-size: 1.7rem; font-weight: 400; }}
/* F8.31G: zoom toolbar above the top-right corner of the boundary image (one row, compact, also at 390 px) */
.st-key-f8i_bd_toolbar {{ gap: 0.3rem; }}
.st-key-f8i_bd_toolbar [data-testid="stElementContainer"] {{ width: auto; flex: 0 0 auto; }}
.st-key-f8i_bd_toolbar button {{ min-width: 40px; min-height: 40px; padding: 0 0.6rem; }}
.st-key-f8i_bd_toolbar button p {{ font-size: 1.15rem; line-height: 1; }}
[class*="st-key-f8i_fixrow_"] > div:first-child {{ flex: 1 1 14rem; min-width: 0; }}
/* F8.31D-5: the keyed step BOX is stable (a key that changed with the state re-created the uploader iframes inside it and abandoned an upload in flight);
   only the small heading wrapper inside it carries the state, and the box is styled from it with :has(). */
[class*="st-key-f8i_stepbox_"]:has([class*="st-key-f8i_stmark_"][class*="_locked"]) {{ opacity: 0.5; }}
[class*="st-key-f8i_stepbox_"]:has([class*="st-key-f8i_stmark_"][class*="_active"]) {{ border-color: {ACCENT} !important; }}
[class*="st-key-f8i_stepbox_"]:has([class*="st-key-f8i_stmark_"][class*="_done"]) {{ border-color: rgba(46, 158, 91, 0.6) !important; }}
/* F8.31D-3A landing page: the same dark/red activity-selection pattern as the Module 2/3/4/6 homepages (large title, subtitle, two equal ~110px red buttons that stack on narrow screens). */
.st-key-f8i_home_layout {{ gap: 0; padding-top: 1rem; }}
.st-key-f8i_home_title, .st-key-f8i_home_subtitle, .st-key-f8i_home_nav {{ gap: 0; flex-shrink: 0; }}
.st-key-f8i_home_title [data-testid="stMarkdownContainer"], .st-key-f8i_home_subtitle [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
.st-key-f8i_home_subtitle {{ padding-top: 44px; }}
.st-key-f8i_home_nav {{ padding-top: 64px; flex-direction: row; flex-wrap: wrap; gap: 24px; }}
.st-key-f8i_home_title h1 {{ font-size: clamp(2.2rem, 11vw, 72px); font-weight: 800; color: inherit; line-height: 1.1; text-align: center; padding: 0; margin: 0; overflow-wrap: anywhere; }}
.st-key-f8i_home_subtitle h2 {{ font-size: clamp(1.1rem, 4.6vw, 28px); font-weight: 600; color: inherit; opacity: 0.85; line-height: 1.2; text-align: center; padding: 0; margin: 0; }}
.st-key-f8i_pick_new, .st-key-f8i_pick_import {{ flex: 1 1 280px; width: auto; min-width: 0; max-width: 100%; }}
.st-key-f8i_pick_new [data-testid="stButton"], .st-key-f8i_pick_import [data-testid="stButton"] {{ width: 100%; }}
.st-key-f8i_pick_new button, .st-key-f8i_pick_import button {{
    width: 100%; height: 110px; min-height: 110px; padding: 0.75rem 1.25rem; border-radius: 12px; background-color: {ACCENT}; border: 2px solid {ACCENT}; color: #FFFFFF;
    display: flex; align-items: center; justify-content: center; text-align: center; }}
.st-key-f8i_pick_new button p, .st-key-f8i_pick_import button p {{ font-size: 21px; font-weight: 700; line-height: 1.3; text-align: center; text-wrap: balance; color: #FFFFFF; }}
.st-key-f8i_pick_new button:hover, .st-key-f8i_pick_import button:hover {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
.st-key-f8i_pick_new button:focus-visible, .st-key-f8i_pick_import button:focus-visible {{ outline: 3px solid #FFFFFF; outline-offset: 2px; box-shadow: 0 0 0 6px {ACCENT}; }}
.st-key-f8i_pick_new button:active, .st-key-f8i_pick_import button:active {{ background-color: #8E2229; border-color: #8E2229; }}
@media (max-width: 640px) {{
    .st-key-f8i_home_nav {{ flex-direction: column; flex-wrap: nowrap; gap: 16px; padding-top: 40px; }}
    .st-key-f8i_pick_new, .st-key-f8i_pick_import {{ flex: 0 0 auto; width: 100%; }}
}}
</style>"""
PROGRESS = {"home": (0.05, "How would you like to calibrate your cameras?"), "setup": (0.35, "New calibration"), "import": (0.35, "Use a saved calibration"), "review": (0.7, "Review the calibration"),
            "ready": (0.8, "Calibration accepted"), "room": (0.85, "Upload your four room videos"), "frames": (0.92, "Choose frames and mark the wall boundaries"), "measure": (0.95, "Room measurement")}
# The wall pages show the two-step experiment instead (the bar only reaches 100 % when BOTH walls are measured; one finished wall is never "complete").
PROGRESS_WALL = {"A": {"room": (0.82, "Upload videos"), "frames": (0.86, "Frames and boundaries"), "measure": (0.9, "Calculate")},
                 "B": {"room": (0.92, "Upload videos"), "frames": (0.94, "Frames and boundaries"), "measure": (0.96, "Calculate")}}


# ------------------------------------------------------------------ state helpers
def session_of(ss) -> S.VisitorSession:
    if "f8i_session" not in ss:
        ss["f8i_session"] = S.VisitorSession()
    return ss["f8i_session"]


def clear_namespace(ss):
    """Remove every ``f8i_`` key (the session object, widget state, messages).  Used by Lock: nothing of this experiment stays in the browser session."""
    for k in [k for k in list(ss) if str(k).startswith("f8i_")]:
        del ss[k]


def _go(sess, step):
    try:
        W.go(sess, st.session_state, step)
    except S.ServiceError as e:
        st.session_state["f8i_msg"] = str(e)


def _wall_ok(sess, wall) -> bool:
    try:
        if wall == sess.active_wall:
            r = sess.measurement.result
        else:
            with sess.on_wall(wall):
                r = sess.measurement.result
        return r is not None and MP.RVW.result_state(r) in (MP.RVW.DONE_OK, MP.RVW.DONE_WARN)
    except Exception:
        return False


def _measurement_complete(sess) -> bool:
    """The whole room experiment is complete only when BOTH walls hold an accepted result."""
    return _wall_ok(sess, "A") and _wall_ok(sess, "B")


def _flow_html(sess, a_done, both) -> str:
    wall = sess.active_wall
    cls = lambda on, done: "f8i-fl " + ("done" if done else ("on" if on else ""))          # noqa: E731
    return ("<div class='f8i-flow' aria-label='Wall A, then Wall B, then the room results'>"
            f"<span class='{cls(wall == 'A' and not both, a_done)}'>Wall A</span><span class='f8i-arrow'>→</span>"
            f"<span class='{cls(wall == 'B' and not both, both)}'>Wall B</span><span class='f8i-arrow'>→</span>"
            f"<span class='{cls(both, False)}'>{'Room Measurement Complete' if both else 'Room Results'}</span></div>")


def _header(sess, ss):
    both = _measurement_complete(sess)
    a_done = _wall_ok(sess, "A")
    on_wall = sess.step in W.WALL_STEPS
    if on_wall:
        frac, sub = PROGRESS_WALL[sess.active_wall][sess.step]
        label = f"{'Wall A' if sess.active_wall == 'A' else 'Wall B'} · {sub}"
        if sess.step == "measure" and both:
            frac, label = 1.0, "Room Measurement Complete"
        elif sess.step == "measure" and sess.active_wall == "A" and a_done:
            label = "Wall A measured · Step 1 of 2"
    else:
        frac, label = UT.progress_for(sess.step, PROGRESS, measurement_complete=False)
    if sess.step == "frames":
        FP._apply_pending_mode(ss)                 # the frames sub-step must be final BEFORE the scroll key is computed (a button may have requested a new sub-step)
    NAV.scroll_to_top_if_new_view(ss, (sess.step, FP.current_mode(ss, sess.frames) if sess.step == "frames" and sess.n_viewpoints != 3 else None) + (() if sess.active_wall == "A" else (sess.active_wall,)))
    NAV.render_nav(sess, label if sess.step != "home" else "Choose how to calibrate")
    if sess.step == "home":
        return                                      # F8.31D-3A: the selection page draws its own heading and has no progress bar
    st.markdown("<h1 class='f8i-title'>Measure a Room</h1>", unsafe_allow_html=True)
    if on_wall:
        st.markdown(_flow_html(sess, a_done, both), unsafe_allow_html=True)
        if not (sess.step == "measure" and both):
            st.markdown(f"<div class='f8i-wallstep'>{W.WALL_STEP[sess.active_wall]}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='f8i-sub'>Calibrate your two cameras, then measure two walls</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='f8i-progress'><div class='f8i-progress-label'>{label}</div><div class='f8i-bar' role='progressbar' aria-valuenow='{frac * 100:.0f}' aria-valuemin='0' aria-valuemax='100'>"
                f"<div class='f8i-fill' style='width:{frac * 100:.0f}%'></div></div></div>", unsafe_allow_html=True)


def quality_phrase(level) -> str:
    """'Quality: Good' / 'Quality not assessed' (the not-assessed label already starts with the word, so it is never doubled)."""
    label = LEVEL_STYLE.get(level, (str(level),))[0]
    return label if label.startswith("Quality") else f"Quality: {label}"


def _badge(level):
    text, color = LEVEL_STYLE[level]
    st.markdown(f"<span class='f8i-badge' style='background:{color}'>{text}</span>", unsafe_allow_html=True)


def _show_code(code):
    st.code(code, language=None)                 # the recovery code is a secret: page text only, never the URL or a log
    st.caption("Keep this code private. It is NOT stored in the page address.")


# ------------------------------------------------------------------ recovery (shared)
def _recovery(svc, sess):
    with st.expander("I have a recovery code (after a refresh or disconnect)"):
        st.caption("A finished calibration is saved on the server and found with the private recovery code shown when it started. Server storage is **temporary**: a restart erases it "
                   "and results are deleted after about a day. Download the result to keep it.")
        code = st.text_input("Recovery code", type="password", key="f8i_rec_code").strip()
        if not code:
            return
        info = svc.recover(code)
        if info["state"] == "running":
            st.info(f"That calibration is still running on the server: {info['stage']} · {info['elapsed_s']:.0f} s · {info['fraction']:.0%}")
            c = st.columns(2)
            c[0].button("Check again", key="f8i_rec_again")
            if c[1].button("Cancel that calibration", key="f8i_rec_cancel"):
                svc.cancel_recovered(code)
        elif info["state"] == "result":
            rev = info["review"]
            st.success(f"Saved result found: {LEVEL_STYLE[rev.level][0]}.")
            if st.button("Review this result", key="f8i_rec_use", type="primary"):
                try:
                    svc.use_recovered(sess, code); st.rerun()
                except S.ServiceError as e:
                    st.error(str(e))
        else:
            (st.warning if info["state"] != "invalid" else st.error)(info.get("message", "Nothing found."))
        if info["state"] in ("interrupted", "not_found", "result") and st.button("Reclaim my retained videos for a retry", key="f8i_rec_adopt"):
            try:
                n = svc.reclaim_videos(sess, code)
                st.success(f"{n} video(s) are now attached to this session." if n else "These videos already belong to this session.")
            except S.ServiceError as e:
                st.error(str(e))


REPLACE_NOTE = ("Starting or importing a different calibration **replaces** the accepted one and discards the room-video checks, frame selections, boundaries and any result that depended on it. "
                "Going back, or viewing the accepted calibration, does not.")


# ------------------------------------------------------------------ home
def page_home(svc, sess):
    with st.container(key="f8i_home_layout"):
        with st.container(key="f8i_home_title"):
            st.title("Camera Calibration", anchor=False)
        with st.container(key="f8i_home_subtitle"):
            st.header("Choose how you'd like to get started.", anchor=False)
        with st.container(key="f8i_home_nav"):
            st.button("Calibrate My Cameras", key="f8i_pick_new", on_click=_go, args=(sess, "setup"), width="stretch")
            st.button("Upload Saved Calibration", key="f8i_pick_import", on_click=_go, args=(sess, "import"), width="stretch")
    if sess.accepted is not None:
        st.success("A calibration is already accepted in this session.")
        st.button("Show it", key="f8i_show_ready", on_click=_go, args=(sess, "ready"))


# ------------------------------------------------------------------ Option A: setup, upload, run
def _slot_line(info, slot):
    if not info:
        return f"{slot.upper()}: no video yet"
    if info["state"] == "complete":
        return f"{slot.upper()}: {info['size'] / MiB:.1f} MB received and verified" + (" · in use by the calibration" if info["leased"] else "")
    return f"{slot.upper()}: uploading {info['next_index']}/{info['n_chunks']} chunks"


DEF = GS.DEFAULT_SETTINGS


def _facts(svc, sess):
    """The backend facts the step states are derived from (recomputed on every rerun; nothing is cached)."""
    r = sess.runner
    rev = None
    if r is not None and r.done:
        try:
            rev = svc.finished_review(sess)
        except S.ServiceError:
            rev = None
    return GS.Facts(uploads_valid=(r is None and svc.slots_ready(sess)), result_ready=bool(rev is not None and rev.usable), accepted=sess.accepted is not None, calibration_started=r is not None), rev


def _confirm(sess, n):
    sess.guide.set_confirmed(n, st.session_state.get(f"f8i_gc_{n}", False), frozen=sess.runner is not None)


def _setting(sess, name, key, step):
    """A measurement was edited: store it (one value, on the session) and clear that step's confirmation and every later one, so nothing stays 'done' on top of an old value."""
    new = st.session_state.get(key)
    if new != sess.guide.settings.get(name):
        sess.guide.set_setting(name, new)
        sess.guide.set_confirmed(step, False)


def _step_head(n, state, reason=None):
    st.subheader(f"Step {n} · {GS.TITLES[n]}", anchor=False)
    if state == "locked":
        st.caption(f":material/lock: {reason}")
    elif state == "done":
        st.caption(":green[:material/check_circle: Completed]")
    else:
        st.caption(":red[:material/arrow_forward: Current step]")


def _instructions(n):
    with st.expander("View instructions", key=f"f8i_ins_{n}"):
        st.markdown(GT.INSTRUCTIONS[n])


def _process_uploads(svc, sess):
    """Run the upload protocol for both slots (only while step 4 is open and no job exists).  Done BEFORE the step states are computed so a finished upload unlocks step 5 in the same run."""
    if sess.runner is not None or not sess.guide.confirmed[3]:
        return {}
    ss = st.session_state
    return {slot: handle(svc.store, sess.owner, sess.protos[slot], ss.get(f"f8i_up_{slot}")) for slot in SLOTS}


def _step4(svc, sess, acks):
    _instructions(4)
    store, lim = svc.store, svc.store.limits
    st.caption(UT.retention_short(lim))
    if sess.runner is not None:
        job = store.job_status(sess.owner)
        for slot in SLOTS:
            info = job["slots"].get(slot)
            st.caption(_slot_line(info, slot) if info else f"{slot.upper()}: received; used by the calibration")
        return
    cols = st.columns(2)
    for col, slot in zip(cols, SLOTS):
        with col:
            component.chunk_uploader(lim.default_chunk_bytes, lim.max_file_bytes, acks.get(slot), sess.epoch, key=f"f8i_up_{slot}", slot=slot, label=f"{slot.upper()} camera video", theme="dark")
            if acks.get(slot) and not acks[slot].get("ok"):
                st.warning(acks[slot].get("message") or acks[slot].get("error"))
    job = store.job_status(sess.owner)
    for slot in SLOTS:
        st.caption(_slot_line(job["slots"].get(slot), slot))


def _num(sess, label, key, name, kind, lo, hi, step, owner_step, help=None, disabled=False):
    v = sess.guide.settings[name]
    return st.number_input(label, min_value=lo, max_value=hi, value=(int(v) if kind == "int" else float(v)), step=step, key=key, on_change=_setting, args=(sess, name, key, owner_step),
                           help=help, disabled=disabled)


def _setup_problem(g, which):
    """Validation by the existing ``parse_setup`` (no rule is repeated here).  which: 'baseline' (step 1) or 'board' (step 2).  -> (CalibrationSetup | None, message | None)"""
    s = g.settings
    try:
        if which == "baseline":
            return S.parse_setup(DEF["cols"], DEF["rows"], DEF["square"], s["baseline"] or None), None
        return S.parse_setup(s["cols"], s["rows"], s["square"], None), None
    except S.ServiceError as e:
        return None, str(e)


def _step1_inputs(sess, frozen):
    st.markdown("**Stereo baseline** is the distance between the centres of the two camera lenses. It is **not** the distance from the cameras to the object or wall you will measure later.")
    _num(sess, "Physical stereo baseline (mm), if measured", "f8i_base", "baseline", "float", 0.0, 2000.0, 1.0, 1, disabled=frozen,
         help="Measured with a ruler between the two lens centres. 0 = not measured: the distance estimated by the calibration is used.")
    _, problem = _setup_problem(sess.guide, "baseline")
    if problem:
        st.error(problem)
    return problem is None


def _step2_inputs(sess, frozen):
    c = st.columns(3)
    with c[0]:
        _num(sess, "Checkerboard inner corners — columns", "f8i_cols", "cols", "int", 2, 40, 1, 2, disabled=frozen)
    with c[1]:
        _num(sess, "Checkerboard inner corners — rows", "f8i_rows", "rows", "int", 2, 40, 1, 2, disabled=frozen)
    with c[2]:
        _num(sess, "Checkerboard square size (mm)", "f8i_sq", "square", "float", 1.0, 500.0, 0.5, 2, disabled=frozen)
    st.caption("Count the *inner* corners (where four squares meet): a board with 9 × 7 squares has 8 × 6 inner corners.")
    _, problem = _setup_problem(sess.guide, "board")
    if problem:
        st.error(problem)
    return problem is None


def _step5(svc, sess, rev):
    ss = st.session_state
    _instructions(5)
    r = sess.runner
    if r is not None and not r.done:
        _running(svc, sess)
        return
    if r is not None:
        (st.success if rev is not None and rev.usable else st.error)(f"The calibration has finished: {LEVEL_STYLE[rev.level][0] if rev is not None else 'no result'}.")
        if rev is not None and not rev.usable:
            if rev.headline:
                st.markdown(f"**{rev.headline}**")
            for p in rev.problems:
                st.markdown(UT.line("bad", p))
        st.caption("Recovery code (private; keep it to find this result again from another session):")
        _show_code(sess.job_code)
        if svc.retry_possible(sess):
            st.warning("Your videos were kept for a short time so you can retry without uploading them again. “Start over” deletes them now.")
        st.button("Retry" if svc.retry_possible(sess) else "Start a new calibration", key="f8i_clear", on_click=svc.clear_result, args=(sess,))
        return
    if sess.accepted is not None:
        st.warning("A calibration is already accepted. " + REPLACE_NOTE)
    g = sess.guide.settings
    st.markdown("**Your settings** (entered in Steps 1 and 2; change them there):")
    st.table([{"Setting": "Checkerboard inner corners (columns × rows)", "Value": f"{int(g['cols'])} × {int(g['rows'])}"},
              {"Setting": "Checkerboard square size", "Value": f"{g['square']:g} mm"},
              {"Setting": "Physical stereo baseline", "Value": f"{g['baseline']:g} mm" if g["baseline"] else "Not measured (the calibrated distance is used)"}])
    try:
        setup = S.parse_setup(g["cols"], g["rows"], g["square"], g["baseline"] or None); problem = None
    except S.ServiceError as e:
        setup, problem = None, str(e)
    if problem:
        st.error(problem)
    both = svc.slots_ready(sess)
    sure = st.checkbox("I understand a full calibration can take up to an hour. Both videos are deleted only after the result is saved.", key="f8i_sure")
    if not svc.full_calibration_available:
        st.warning("Calibrating from videos is switched off on this server, so **Start calibration** is disabled here. Steps 1–4 and your settings can still be completed. "
                   "To continue, go back to the first page and choose **Upload Saved Calibration**.")
    if st.button("Start calibration", type="primary", key="f8i_start", disabled=not (both and setup and svc.full_calibration_available and sure)):
        try:
            svc.start_calibration(sess, setup)             # ONLY here: a rerun never starts a job
        except S.ServiceError as e:
            st.error(str(e))
        else:
            st.rerun()


def _step6(svc, sess, rev):
    _instructions(6)
    _badge(rev.level)
    if rev.headline:
        st.markdown(f"**{rev.headline}**")
    if sess.accepted is not None:
        st.success("You accepted this calibration.")
        st.button("Show the accepted calibration", key="f8i_s6_ready", on_click=_go, args=(sess, "ready"))
    else:
        st.button("Review the calibration", type="primary", key="f8i_to_review", on_click=_go, args=(sess, "review"))


def page_setup(svc, sess):
    g, r = sess.guide, sess.runner
    frozen = r is not None
    acks = _process_uploads(svc, sess)
    facts, rev = _facts(svc, sess)
    st.markdown("<div class='f8i-pagetitle'>Calibrate My Cameras</div>", unsafe_allow_html=True)
    st.caption("Follow the six steps in order. Each step opens when the one before it is complete.")
    for n in range(1, GS.N_STEPS + 1):
        state = g.state(n, facts)
        with st.container(border=True, key=f"f8i_stepbox_{n}"):
            with st.container(key=f"f8i_stmark_{n}_{state}"):          # only this heading wrapper changes with the state; the widgets below keep their identity
                _step_head(n, state, g.locked_reason(n))
            if state == "locked":
                continue                                            # nothing interactive exists inside a locked step
            if n in GS.CONFIRM_STEPS:
                _instructions(n)
                valid = _step1_inputs(sess, frozen) if n == 1 else _step2_inputs(sess, frozen) if n == 2 else True
                st.session_state[f"f8i_gc_{n}"] = g.confirmed[n]          # the session object is the single source of truth (set BEFORE the widget exists, so no widget-state error)
                st.checkbox(GS.CONFIRM_LABELS[n], key=f"f8i_gc_{n}", on_change=_confirm, args=(sess, n), disabled=frozen or not valid,
                            help="Locked while a calibration exists. Use Retry / Start a new calibration (Step 5) to change it." if frozen else None)
            elif n == 4:
                _step4(svc, sess, acks)
            elif n == 5:
                _step5(svc, sess, rev)
            else:
                _step6(svc, sess, rev)
    if r is not None and not r.done:
        time.sleep(1.0)
        st.rerun()
    _recovery(svc, sess)


def _running(svc, sess):
    r = sess.runner; s = r.status()
    st.progress(min(s["fraction"], 0.99), text=f"{s['stage'].replace('_', ' ')} · {s['elapsed_s']:.0f} s")
    st.info("Recovery code: copy it now and keep it private. If the page reloads or disconnects, paste it under “I have a recovery code”.")
    _show_code(sess.job_code)
    if st.button("Cancel the calibration", key="f8i_cancel"):
        svc.cancel(sess)


# ------------------------------------------------------------------ Option B: import
def page_import(svc, sess):
    st.markdown("<div class='f8i-pagetitle'>Upload Saved Calibration</div>", unsafe_allow_html=True)
    with st.expander("View Instructions", expanded=True):
        st.markdown(
            "**Supported files.** A calibration saved earlier as a **JSON** file: the result downloaded from this app, a calibration contract, or a file from an earlier version. "
            "It holds each camera's intrinsics and lens distortion and the distance between the two cameras. It does **not** calibrate cameras.\n\n"
            "**Upload.** Choose the file below. If it does not say whether the cameras were calibrated from video or photos, choose that; you can also enter the distance between the cameras.\n\n"
            "**Quality review.** Press **Check this file**. You will see a quality rating, any concerns and the numbers that would be used. A file with no quality information is flagged.\n\n"
            "**Acceptance.** Nothing is used until you review it and press **Accept this calibration**.")
    if sess.accepted is not None:
        st.warning("A calibration is already accepted. " + REPLACE_NOTE)
    up = st.file_uploader("Calibration file (JSON)", type=["json"], key="f8i_cal_file")
    c = st.columns(2)
    mode = c[0].radio("Both cameras were calibrated from", ["video", "photo"], index=None, horizontal=True, key="f8i_cal_mode",
                      help="Only needed for files that do not state it (Phase 1 files, some contracts). It is never guessed.")
    base = c[1].number_input("Distance between the two cameras (mm)", min_value=0.0, value=0.0, step=1.0, key="f8i_cal_base",
                             help="For a result saved from this app this overrides the stored baseline; for other files it is only used when the file has none. 0 = not entered. A distance you enter is labelled as entered by you.")
    if st.button("Check this file", type="primary", disabled=up is None, key="f8i_cal_go"):
        sess_data = up.getvalue()
        svc.import_saved(sess, sess_data, capture_mode=mode, baseline_mm=base or None)
        st.rerun()


# ------------------------------------------------------------------ review + accept
def page_review(svc, sess):
    rev = sess.review
    if rev is None:
        st.warning("There is nothing to review yet."); return
    st.caption(KIND_TEXT.get(rev.kind, rev.kind))
    _badge(rev.level)
    if rev.level == S.NOT_ASSESSED:
        # One explanation instead of four: the headline, the "no quality assessment" warning, the generic concern and the checkbox all said the same thing.
        st.markdown(NOT_ASSESSED_TEXT)
        concerns = [c for c in rev.concerns if c != S.NOT_ASSESSED_CONCERN]            # a genuinely different concern is still shown
    else:
        if rev.headline:
            st.markdown(f"**{rev.headline}**")
        if rev.quality_assessed_by_f8:
            st.caption("The calibration's own quality checks were applied.")
        concerns = list(rev.concerns)
    for p in rev.problems:
        st.markdown(UT.line("bad", p))
    for cn in concerns:
        st.markdown(UT.line("warn", cn))
    for w in rev.f6_warnings:
        st.markdown(UT.line("warn", w))
    if rev.usable:
        rows = [{"Item": "Baseline that will be used", "Value": f"{rev.baseline_used_mm:g} mm ({UT.baseline_phrase(rev.baseline_source)})"}]
        if rev.calibrated_baseline_mm is not None and rev.baseline_source == "measured":
            rows.append({"Item": "Baseline estimated by the calibration (for comparison)", "Value": f"{rev.calibrated_baseline_mm:.1f} mm"})
        if rev.stereo_rms_px is not None:
            rows.append({"Item": "Stereo reprojection error", "Value": f"{rev.stereo_rms_px:.3f} px"})
        rows.append({"Item": "Image size", "Value": f"{rev.image_size_wh[0]} × {rev.image_size_wh[1]} px"})
        st.table(rows)
    data = svc.result_bytes(sess)
    if data is not None and rev.kind in ("f8_job", "f8_recovered"):
        st.download_button("Download the calibration result (JSON)", data, file_name=f"f8_result_{rev.job_id or 'job'}.json", mime="application/json", key="f8i_dl_result")
    ack = False
    if rev.requires_acknowledgement:
        label = ("I have read the notes above and want to use this calibration anyway." if rev.level == S.QUESTIONABLE
                 else "I understand and want to use this calibration.")
        ack = st.checkbox(label, key=f"f8i_ack_{id(rev)}")
    if rev.level == S.FAILED:
        st.error("A failed calibration cannot be accepted.")
    elif rev.level == S.INVALID and not rev.problems:
        st.error("This calibration cannot be used.")
    if st.button("Accept this calibration", type="primary", key="f8i_accept", disabled=not rev.can_accept(ack)):
        try:
            res = svc.accept(sess, acknowledged=ack)
        except S.ServiceError as e:
            st.error(str(e))
        else:
            if res.ok:
                st.rerun()
            st.error("This calibration was rejected by the measurement checks."); [st.markdown(UT.line("bad", m)) for m in res.problems]


# ------------------------------------------------------------------ ready
def page_ready(svc, sess):
    a = sess.accepted
    if a is None:
        st.warning("No calibration has been accepted yet."); return
    st.success("Calibration accepted. You are ready for room measurement.")
    st.markdown(f"**{UT.baseline_line(a.baseline_used_mm, (a.provenance or {}).get('baseline_source'))}** · **Image size:** {a.image_size_wh[0]}×{a.image_size_wh[1]} px · **{quality_phrase(a.level)}**")
    st.caption("The calibration passed the measurement checks. You can go back to review it; doing so does not discard anything.")
    _cf = [x for x in QF.calibration_feedback(a.level, getattr(a, "concerns", ()) or ()) if x.state == QF.NEEDS]
    if _cf:
        with st.expander("How to improve this calibration"):
            for x in _cf:
                st.markdown(f"**{x.state_label}.** {x.what} {x.action}")
    st.caption(REPLACE_NOTE)
    data = svc.contract_json(sess)
    if data:
        st.download_button("Download this calibration (JSON)", data, file_name="calibration_contract.json", mime="application/json", key="f8i_dl_contract",
                           help="Calibration numbers only. You can import it again with “Use a saved calibration”.")
    st.divider()
    st.button("Continue to Room Measurement", type="primary", key="f8i_to_room", on_click=_go, args=(sess, "room"))


# ------------------------------------------------------------------ room videos (F8.31A: upload + validation ONLY)
STATUS_STYLE = {RV.OK: (UT.ICON["ok"], "Accepted"), RV.WARN: (UT.ICON["warn"], "Accepted, with recommendations"), RV.ERROR: (UT.ICON["bad"], "Rejected"),
                RV.MISSING: (UT.ICON["todo"], "No video yet"), RV.PENDING: (UT.ICON["wait"], "Uploading / checking")}

def _slot_facts(r):
    f = r.facts
    if not f.get("width"):
        return ""
    return f"{f['width']} × {f['height']} px · {f['duration_s']:.1f} s · {f['fps']:g} fps"


def _remove_slot(svc, sess, slot):
    """× pressed.  Significant downstream work -> ask first (the pending slot is kept per session in session state); otherwise delete at once."""
    if svc.removal_impact(sess, slot):
        st.session_state["f8i_rm_pending"] = slot
        return
    _do_remove(svc, sess, slot)


def _do_remove(svc, sess, slot):
    st.session_state.pop("f8i_rm_pending", None)
    try:
        svc.remove_room_video(sess, slot)
    except S.ServiceError as e:
        st.session_state["f8i_msg"] = str(e)


def _room_slot_card(svc, sess, ss, slot, acks, report):
    r = report.slots[slot]; icon, text = STATUS_STYLE[r.status]
    with st.container(border=True):
        lim = svc.room_store.limits
        # The uploader stays the FIRST element of the card with a fixed key, so removing / replacing a video never re-creates it (F8.31D-5).  Only its epoch changes to clear its own display.
        component.chunk_uploader(lim.default_chunk_bytes, lim.max_file_bytes, acks[slot], sess.epoch * 1000 + sess.room_slot_epoch.get(slot, 0), key=f"f8i_room_up_{slot}{W.suffix(sess)}", slot=slot, label=r.label, theme="dark", hide_idle_hint=True)
        if acks[slot] and not acks[slot].get("ok") and not r.errors:         # a rejected file is explained once, below, not twice
            st.warning(acks[slot].get("message") or acks[slot].get("error"))
        has_file = bool(r.facts.get("name"))
        with st.container(horizontal=True, wrap=False, vertical_alignment="center", key=f"f8i_cardrow_{slot}{W.suffix(sess)}"):
            st.markdown(f"{icon} **{text}**" + (f" · {r.facts['name']}" if has_file else ""))
            if has_file:
                st.button("×", key=f"f8i_room_rm_{slot}{W.suffix(sess)}", type="tertiary", help="Remove video", on_click=_remove_slot, args=(svc, sess, slot))
        if ss.get("f8i_rm_pending") == slot:
            what = "; ".join(svc.removal_impact(sess, slot)) or "work that depended on it"
            st.warning(f"Removing this video will also discard {what}. This cannot be undone.")
            c = st.columns(2)
            c[0].button("Remove and discard", key=f"f8i_room_rm_yes_{slot}{W.suffix(sess)}", type="primary", width="stretch", on_click=_do_remove, args=(svc, sess, slot))
            c[1].button("Keep it", key=f"f8i_room_rm_no_{slot}{W.suffix(sess)}", width="stretch", on_click=lambda: st.session_state.pop("f8i_rm_pending", None))
        if _slot_facts(r):
            st.caption(_slot_facts(r))
        for m in r.errors:
            st.markdown(UT.line("bad", m))
        for m in r.warnings:
            st.markdown(UT.line("warn", m))


def _room_retention(lim):
    """The upload-retention facts for this page (same numbers as everywhere, derived from the real limits); the last line of ``UT.retention_lines`` names buttons this page does not have."""
    return UT.retention_lines(lim)[:3]


def _room_step(n, state, wall="A", n_viewpoints=2):
    st.subheader(f"Preparation {n} · {GS.ROOM_TITLES[n]}" if n < 3 else f"Upload Your {VPC.count_word(n_viewpoints).title()} Wall {wall} Videos", anchor=False)
    if state == "locked":
        st.caption(f":material/lock: {GS.ROOM_LOCK_REASONS[n]}")
    elif state == "done":
        st.caption(":green[:material/check_circle: Completed]")
    else:
        st.caption(":red[:material/arrow_forward: Current step]")


def _room_confirm(sess, n):
    sess.guide.room.set_confirmed(n, st.session_state.get(f"f8i_rc_{n}", False))


def _vp_apply(svc, sess, n):
    """Apply a viewpoint-count change for the ACTIVE wall (the service discards only that wall's incompatible work)."""
    st.session_state.pop("f8i_vp_pending", None)
    try:
        notes = svc.set_viewpoint_count(sess, sess.active_wall, n)
    except S.ServiceError as e:
        st.session_state["f8i_msg"] = str(e)
    else:
        if notes:
            st.session_state["f8i_msg"] = notes[0]


def _vp_has_work(svc, sess) -> bool:
    uploads = svc.room_store.job_status(sess.room_owner)["slots"] if svc.room_store is not None else {}
    return bool(uploads or sess.frames is not None or sess.measurement.result is not None)


def _vp_changed(svc, sess):
    """The radio changed.  Work that would be discarded -> ask first (the radio is put back until the visitor confirms); nothing to discard -> apply at once."""
    key = f"f8i_nvp{W.suffix(sess)}"; new, cur = st.session_state.get(key), sess.n_viewpoints
    if new == cur or new not in VPC.VALID_COUNTS:
        return
    st.session_state[key] = cur
    if _vp_has_work(svc, sess):
        st.session_state["f8i_vp_pending"] = new
    else:
        _vp_apply(svc, sess, new)


def _viewpoint_selector(svc, sess, ss):
    """Number of stereo viewpoints of THIS wall.  Wall A's three-viewpoint option is shown, disabled and labelled not yet validated."""
    wall, key = sess.active_wall, f"f8i_nvp{W.suffix(sess)}"
    allowed3 = VPC.allowed(wall, 3)
    ss[key] = sess.n_viewpoints                                              # the session object is the single source of truth
    st.radio("Number of stereo viewpoints for this wall", VPC.VALID_COUNTS, key=key, horizontal=True, disabled=not allowed3,
             format_func=lambda n: VPC.OPTION_LABEL[n] + ("" if VPC.allowed(wall, n) else f" — {VPC.NOT_VALIDATED}"), on_change=_vp_changed, args=(svc, sess))
    if not allowed3:
        st.caption(f":material/info: 3 viewpoints — **{VPC.NOT_VALIDATED}** for Wall A.")
        with st.expander("Why is this not available for Wall A?"):
            st.markdown(VPC.WALL_A_THREE_MISSING)
    else:
        st.caption(VPC.WALL_B_THREE_HELP)
    pend = ss.get("f8i_vp_pending")
    if pend in VPC.VALID_COUNTS and pend != sess.n_viewpoints:
        st.warning(f"Switch Wall {wall} to {pend} viewpoints? {VPC.CHANGE_WARNING}")
        c = st.columns(2)
        c[0].button("Switch and discard", key=f"f8i_vp_yes{W.suffix(sess)}", type="primary", width="stretch", on_click=_vp_apply, args=(svc, sess, pend))
        c[1].button("Keep the current choice", key=f"f8i_vp_no{W.suffix(sess)}", width="stretch", on_click=lambda: st.session_state.pop("f8i_vp_pending", None))


def page_room(svc, sess):
    ss = st.session_state
    ctx = svc.calibration_context(sess)
    if ctx is None:
        st.warning("There is no accepted calibration in this session, so room videos cannot be uploaded. Calibrate (or import a calibration) and accept it first.")
        st.button("Go to the calibration", key="f8i_room_back0", on_click=_go, args=(sess, "home"))
        return
    if svc.room_store is None:
        st.error("Room video upload is not available on this server."); return
    rg, lim = sess.guide.room, svc.room_store.limits
    # Uploads are handled only while step 3 is open; the readiness is recomputed from the backend on EVERY run (never from a stored flag), so an expired or replaced upload locks it again.
    step3_open = rg.unlocked(3)
    sfx = W.suffix(sess)
    nvp = sess.n_viewpoints; slots = RV.slots_for(nvp); nv = len(slots)
    acks = {slot: (handle(svc.room_store, sess.room_owner, sess.room_protos[slot], ss.get(f"f8i_room_up_{slot}{sfx}")) if step3_open else None) for slot in slots}
    report = svc.room_report(sess)
    ready = rg.can_continue(report.ready)
    q = ctx["quality"]
    second = sess.active_wall == "B"
    st.markdown(f"<div class='f8i-pagetitle'>{W.WALL_HEADING[sess.active_wall]}</div>", unsafe_allow_html=True)
    st.markdown(W.WALL_UPLOAD_HINT[sess.active_wall] if nvp == 2 else W.WALL_UPLOAD_HINT_3[sess.active_wall])
    if second:
        st.info("Wall A's results and the calibration are kept. You do not need to repeat Wall A.")
    st.success(f"Accepted calibration · {UT.baseline_line(ctx['baseline_mm'], ctx.get('baseline_source'))} · Image {ctx['image_size_wh'][0]}×{ctx['image_size_wh'][1]} px · {quality_phrase(q)}")
    if not second:
        st.caption("Follow the three parts in order. Each part opens when the one before it is complete.")
    for n in range(1, GS.ROOM_N_STEPS + 1):
        if second and n != 3:
            continue                                                # the second wall has only the upload step (same camera rig, same calibration)
        state = rg.state(n, report.ready)
        with st.container(border=True, key=f"f8i_stepbox_r{n}{sfx}"):
            with st.container(key=f"f8i_stmark_r{n}{sfx}_{state}"):         # F8.31D-5: only the heading wrapper carries the state, so the four uploaders are never re-created mid-upload
                _room_step(n, state, sess.active_wall, nvp)
            if state == "locked":
                continue                                            # nothing interactive exists inside a locked step
            with st.expander("View Instructions", key=f"f8i_rins_{n}{sfx}"):
                st.markdown(GT.ROOM_INSTRUCTIONS[n])
                if n == 1:
                    st.markdown(f"<div class='f8i-limitnote'>{GT.ROOM_LIMITATION_NOTE}</div>", unsafe_allow_html=True)
            if n in GS.ROOM_CONFIRM_STEPS:
                ss[f"f8i_rc_{n}"] = rg.confirmed[n]                  # the session object is the single source of truth (set BEFORE the widget exists)
                st.checkbox(GS.ROOM_CONFIRM_LABELS[n], key=f"f8i_rc_{n}", on_change=_room_confirm, args=(sess, n))
                continue
            _viewpoint_selector(svc, sess, ss)
            st.caption(f"{UT.retention_short(lim)} Use × on a video to remove it.")
            with st.container(border=True, key=f"f8i_cloud{sfx}"):              # the Cloud-resources notice: limits come from the enforcing objects (the room store, F6 limits, the measurement service)
                pol = svc.measurement.resource_policy()
                st.markdown("\n".join(UT.cloud_limits_notice(
                    wall=sess.active_wall, n_videos=nv, n_viewpoints=nvp, per_file_mb=lim.max_file_bytes / MiB, store_total_mb=lim.max_disk_bytes / MiB,
                    min_duration_s=RV.F6_LIMITS.min_duration_s, max_duration_s=RV.F6_LIMITS.max_duration_s, timeout_s=pol["timeout_s"], cpu_limit_s=pol["cpu_limit_s"],
                    min_memory_mb_2=pol["min_memory_mb_2"], min_memory_mb_3=pol["min_memory_mb_3"])))
            with st.container(border=True, key=f"f8i_reqs{sfx}"):               # Phase D.3: the verified recording and upload requirements, beside the uploaders (numbers come from the enforcing objects)
                st.markdown("\n".join(UT.upload_requirements(
                    n_videos=nv, viewpoints=[(vp, VPC.ROLE[vp] if nvp == 3 else None) for vp in range(1, nvp + 1)], per_file_mb=lim.max_file_bytes / MiB, store_total_mb=lim.max_disk_bytes / MiB,
                    min_duration_s=RV.F6_LIMITS.min_duration_s, max_duration_s=RV.F6_LIMITS.max_duration_s, min_fps=RV.F6_LIMITS.min_fps, max_fps=RV.F6_LIMITS.max_fps,
                    fps_tolerance=RV.FPS_MATCH_TOLERANCE, validated_wh=RV.VALIDATED_FRAME_SIZES[0])))
            for vp in range(1, nvp + 1):
                st.markdown(f"#### {RV.VIEWPOINT_HEADINGS[vp]}" + (f" · {VPC.ROLE[vp]}" if nvp == 3 else ""))
                cols = st.columns(2, gap="medium")
                for col, cam in zip(cols, ("left", "right")):
                    with col:
                        _room_slot_card(svc, sess, ss, f"p{vp}_{cam}", acks, report)
            st.progress(report.n_valid / nv, text=f"{report.n_valid} of {nv} videos accepted" + (f" · {report.n_rejected} rejected (does not count)" if report.n_rejected else ""))
            for m in report.set_errors:
                st.markdown(UT.line("bad", m))
            for m in report.set_warnings:
                st.markdown(UT.line("warn", m))
            _vf = [x for x in QF.video_feedback(report) if x.state in (QF.FAILED, QF.NEEDS)]
            if _vf:
                with st.expander("Suggestions"):
                    for x in _vf:
                        st.markdown(f"**{x.state_label} · {x.affected}.** {x.what} {x.action} Record again: {x.record_again}.")
            if report.missing:
                st.markdown(UT.line("todo", "Still needed: " + "; ".join(report.missing) + "."))
            if report.pending:
                st.markdown(UT.line("wait", "Waiting for: " + "; ".join(report.pending) + "."))
            if report.ready:
                st.success(f"All {VPC.count_word(nvp)} videos pass the required checks." + (" Recommendations above do not block you." if report.all_warnings else ""))
    if not ready:
        st.info(f"Continue becomes available when all {VPC.count_word(nvp)} videos are uploaded and accepted.")
    if st.button("Continue to Frame Selection", type="primary", key="f8i_to_frames", disabled=not ready):
        try:
            svc.go_frames(sess)
        except S.ServiceError as e:
            st.error(str(e))
        else:
            st.rerun()


def page_frames(svc, sess):
    """F8.31B: frame selection and wall-boundary annotation (see ``f8_frames_page``)."""
    if svc.calibration_context(sess) is None or not sess.room_ready:
        svc.room_report(sess)
        if svc.calibration_context(sess) is None or not sess.room_ready:
            st.warning(f"Upload and validate the {VPC.count_word(sess.n_viewpoints)} room videos first.")
            if svc.calibration_context(sess) is not None:
                st.button("Go to the room videos", key="f8i_frames_back0", on_click=_go, args=(sess, "room"))
            return
    if sess.n_viewpoints == 3:
        import f8_three_view_page as TVP                      # three-viewpoint Wall B: its own frame sets and seven-line boundary workflow
        TVP.render_frames_3(svc, sess)
        return
    FP.render_frames(svc, sess)


def page_measure(svc, sess):
    """F8.31C-2A: measurement readiness, the gated measurement action and the results (see ``f8_measure_page``)."""
    MP.render_measure(svc, sess)


PAGES = dict(home=page_home, setup=page_setup, **{"import": page_import}, review=page_review, ready=page_ready, room=page_room, frames=page_frames, measure=page_measure)


JOB_KINDS = ("f8_job", "f8_recovered")           # reviews that come from a calibration made in this app; every other kind is a saved calibration file


def in_import_workflow(sess) -> bool:
    """F8.31D: True on the pages of the Upload Saved Calibration workflow: the import page, the review of an uploaded file, and the accepted-calibration page of an uploaded file.
    Those pages show no Start over / Lock buttons; the reset and lock logic itself (``VisitorSession.reset`` / ``clear_namespace``) is unchanged and still used by every other page."""
    if sess.step == "import":
        return True
    if sess.step == "review":
        return sess.review is not None and sess.review.kind not in JOB_KINDS
    if sess.step == "ready":
        kind = (getattr(sess.accepted, "provenance", None) or {}).get("kind")
        return kind is not None and kind not in JOB_KINDS
    return False


def _session_options(svc, sess, ss):
    with st.expander("Session Options"):
        if st.button("Lock this session", key="f8i_lock", help="Cancels any calibration, deletes your uploads and starts a fresh session."):
            sess.reset(svc.store, svc.room_store); clear_namespace(ss); st.rerun()


def render(svc, sess, ss):
    """The whole experiment."""
    st.html(CSS)
    _header(sess, ss)
    if ss.get("f8i_msg"):
        st.warning(ss.pop("f8i_msg"))
    PAGES[sess.step](svc, sess)
    if sess.step == "measure":
        _session_options(svc, sess, ss)             # no Start over here: New Measurement (results page) is the visible reset; locking sits in a small expander
        return
    if sess.step in ("home", "room", "frames") or in_import_workflow(sess):
        return                                      # F8.31F: the boundary page has no Start over / Lock either.  The selection page and the room-upload page show only the heading, the two buttons and the navigation; reset/lock stay on the workflow pages
    st.divider()
    c = st.columns(2)
    if c[0].button("Start over (discard this session's data)", key="f8i_reset", width="stretch"):
        sess.reset(svc.store, svc.room_store); st.rerun()
    if c[1].button("Lock this session", key="f8i_lock", width="stretch", help="Cancels any calibration, deletes your uploads and starts a fresh session."):
        sess.reset(svc.store, svc.room_store); clear_namespace(ss); st.rerun()

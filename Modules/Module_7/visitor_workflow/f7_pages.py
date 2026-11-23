"""Stage F7 pages: the guided calibration wizard, ONE STEP AT A TIME (Streamlit drawing only; all rules live in f7_calibration.py).

Only the current step's content is rendered.  Entered values live in the CalibrationWizard object (not in widget state), so they survive Back / Continue.
"""
import json
import os
import time

import streamlit as st

import f7_calibration as K

GOOD_C, WARN_C, BAD_C = "#2E9E5B", "#E0A526", "#C9363E"
LEVEL_STYLE = {K.GOOD: ("Good", GOOD_C), K.QUESTIONABLE: ("Questionable but usable", WARN_C), K.FAILED: ("Failed", BAD_C)}
# Visitor-facing headings of the six internal steps (the core's K.STEPS names are unchanged).
TITLES = ("Prepare Your Cameras", "Enter Camera Details", "Upload Calibration Videos", "Calibrating Your Cameras", "Review Calibration Results", "Accept Calibration")
CSS = """<style>
.f7-lead { color: #B8BDC7; font-size: 1.05rem; margin: 0 0 1.2rem; }
.f7-part { color: #9AA1AC; font-size: 0.9rem; margin: -0.4rem 0 0.8rem; }
.f7-item { display: flex; gap: 0.9rem; align-items: flex-start; margin: 0.9rem 0; }
.f7-num { flex: 0 0 2rem; height: 2rem; border-radius: 50%; background: rgba(201, 54, 62, 0.18); color: #C9363E; font-weight: 800; display: flex; align-items: center; justify-content: center; }
.f7-item b { display: block; font-size: 1.05rem; }
.f7-item span { color: #B8BDC7; }
.f7-gap { height: 0.6rem; }
</style>"""
STEP_ITEMS = (("🎥", "Set Up Your Cameras", "Mount two cameras side by side and keep them fixed in position."),
              ("🏁", "Prepare a Checkerboard", "Print a flat checkerboard and measure its square size."),
              ("🎬", "Record Calibration Videos", "Record the checkerboard with both cameras simultaneously, varying its position and angle."))
DETAILS = (
    "**Same setup as the room measurement.** Use the same two cameras, mounted the same way, with the same lens, zoom and recording mode for calibration and for filming the room. Do not change them afterwards.",
    "**Checkerboard.** Print it on a flat, rigid surface. Count its *inner corners* (where four squares meet; a board with 9 × 7 squares has 8 × 6 inner corners) and measure the physical size of one square with a ruler.",
    "**Camera baseline.** Optionally measure the physical distance between the two lens centres with a ruler. If you enter it, it is used for room measurement.",
    "**Record simultaneously.** Start both cameras, then move the board. Each camera records its own video.",
    "**Diverse positions.** Hold the board still for about 2 seconds in each pose, then move on. Aim for 20 or more different poses: vary the distance, tilt the board, and place it near the edges and corners of the picture.",
    "**Visible in both cameras.** Keep the whole board visible to both cameras in every pose.",
    "**Video limits.** Each video must be between 10 and 120 seconds and under 200 MB, with the same resolution and orientation as your room videos.",
)


def wizard() -> K.CalibrationWizard:
    ss = st.session_state
    if "f7" not in ss: ss["f7"] = K.CalibrationWizard()
    return ss["f7"]


def close_wizard():
    w = st.session_state.get("f7")
    if w is not None: w.close()
    for k in [k for k in st.session_state if str(k).startswith("f7")]: del st.session_state[k]


def _heading(w):
    st.markdown(f"## {'Calibrate Your Cameras' if w.step == 0 else TITLES[w.step]}")
    st.markdown(f"<div class='f7-part'>Calibration part {w.step + 1} of {len(K.STEPS)} · {TITLES[w.step]}</div>", unsafe_allow_html=True)


def _nav(w, continue_label="Continue", show_continue=True):
    blockers = w.blockers() if show_continue else []
    if blockers and show_continue:
        st.markdown("<div class='f7-gap'></div>", unsafe_allow_html=True)
        for b in blockers[:4]: st.caption(f"• {b}")
    c = st.columns([1, 2, 1])
    c[0].button("Back", key=f"f7_back_{w.step}", on_click=w.back, disabled=w.step == 0, width="stretch")
    if show_continue: c[2].button(continue_label, key=f"f7_next_{w.step}", type="primary", on_click=w.next, disabled=bool(blockers), width="stretch")


def _fixtures_visible() -> bool:
    return os.environ.get("F6_SHOW_FIXTURES") == "1"


# ------------------------------------------------------------------ the six steps
def step_instructions(w, on_use_import):
    _heading(w)
    st.markdown("<div class='f7-lead'>Before measuring your room, calibrate your two cameras using a checkerboard.</div>", unsafe_allow_html=True)
    st.markdown("".join(f"<div class='f7-item'><div class='f7-num'>{i}</div><div><b>{icon} {t}</b><span>{d}</span></div></div>" for i, (icon, t, d) in enumerate(STEP_ITEMS, 1)), unsafe_allow_html=True)
    with st.expander("Detailed Recording Instructions"):
        for d in DETAILS: st.markdown(f"- {d}")
        st.caption("Full checklist: " + " ".join(K.INSTRUCTIONS))
    _nav(w)
    with st.expander("Already Have a Calibration File?"):
        st.caption("If you already calibrated this exact camera pair, you can import the saved calibration instead of calibrating again.")
        st.button("Import a calibration file", key="f7_use_import", on_click=on_use_import)


def _sync(field, key, cast=None):
    def run():
        v = st.session_state.get(key)
        if cast is not None and v is not None:
            try: v = cast(v)
            except (TypeError, ValueError): v = None
        wizard().set_info(**{field: v})
    return run


def step_info(w):
    _heading(w)
    st.markdown("<div class='f7-lead'>Tell us which cameras you are using and describe your checkerboard.</div>", unsafe_allow_html=True)
    i = w.info
    c = st.columns(2)
    c[0].text_input("Left camera", value=i["left_camera"], key="f7_left_camera", on_change=_sync("left_camera", "f7_left_camera"), placeholder="e.g. iPhone 15 Pro", max_chars=60)
    c[1].text_input("Right camera", value=i["right_camera"], key="f7_right_camera", on_change=_sync("right_camera", "f7_right_camera"), placeholder="e.g. iPhone 17 Pro Max", max_chars=60)
    st.markdown("**Checkerboard** — count the *inner corners* (where four squares meet). A board with 9 × 7 squares has 8 × 6 inner corners.")
    c = st.columns(3)
    c[0].number_input("Inner corners — columns", value=i["cols"], min_value=0, max_value=40, step=1, key="f7_cols", on_change=_sync("cols", "f7_cols", int), placeholder="e.g. 8")
    c[1].number_input("Inner corners — rows", value=i["rows"], min_value=0, max_value=40, step=1, key="f7_rows", on_change=_sync("rows", "f7_rows", int), placeholder="e.g. 6")
    c[2].number_input("Square size (mm)", value=i["square_mm"], min_value=0.0, max_value=500.0, step=0.5, key="f7_square", on_change=_sync("square_mm", "f7_square", float), placeholder="e.g. 20")
    st.number_input("Measured distance between the two cameras (mm) — optional", value=i["baseline_mm"], min_value=0.0, max_value=2000.0, step=1.0, key="f7_baseline",
                    on_change=_sync("baseline_mm", "f7_baseline", float), placeholder="leave empty to use the calibrated value",
                    help="Measure between the two lens centres with a ruler. If you enter it, it is used as the baseline for room measurement.")
    _nav(w)


def _video_card(w, side):
    v = w.videos[side]; r = None if v is None else v["report"]
    if v is None: return
    ok = not r.errors
    st.markdown(f"{'✅' if ok else '⛔'} **{v['name']}** · {v['size_mb']:.0f} MB" + (f" · {r.width}×{r.height} px · {r.duration_s:.0f} s · {r.fps:.0f} fps" if r.width else ""))
    for e in r.errors: st.markdown(f"⛔ {e.message}")
    for x in r.warnings: st.markdown(f"⚠️ {x.message}")
    st.button(f"Remove {side} video", key=f"f7_rm_{side}", on_click=w.remove_video, args=(side,))


def step_videos(w):
    _heading(w)
    st.markdown("<div class='f7-lead'>Upload the two videos filmed at the same time, one per camera, showing the board in many poses.</div>", unsafe_allow_html=True)
    st.caption("Same resolution · 10–120 s · under 200 MB each")
    ss = st.session_state; c = st.columns(2)
    for col, side in zip(c, ("left", "right")):
        with col:
            st.markdown(f"**{side.title()} camera video**" + (f" — {w.info[side + '_camera']}" if w.info[side + "_camera"] else ""))
            up = st.file_uploader(f"{side.title()} calibration video", type=["mov", "mp4", "m4v"], key=f"f7_up_{side}", label_visibility="collapsed")
            if up is not None and ss.get(f"f7_upid_{side}") != up.file_id:
                with st.spinner("Checking the video…"): w.set_video(side, up.getvalue(), up.name)
                ss[f"f7_upid_{side}"] = up.file_id
            _video_card(w, side)
    for x in w.video_warnings(): st.warning(x)
    _nav(w)


def step_process(w):
    _heading(w)
    stt = w.processing_status(); running = stt["state"] == "running"
    st.markdown("<div class='f7-lead'>We find the checkerboard in both videos and calibrate the cameras. This takes about a minute.</div>", unsafe_allow_html=True)
    st.button("Start calibration" if w.result is None else "Run again", key="f7_start", type="primary", on_click=w.start_processing, disabled=running)
    if running:
        st.progress(min(stt["progress"], 0.99), text=stt["message"]); time.sleep(0.8); st.rerun()
    elif w.result is not None:
        r = w.result
        if r.get("ok"): st.success(f"Calibration finished: {r['pairs_used']} board poses used ({r['pairs_discarded']} discarded) in {r['runtime_s']:.0f} s. Review the quality on the next step.")
        else: st.error(r["error"]); st.caption("You can continue to see the assessment, or go Back to upload different videos.")
    _nav(w, show_continue=True)


def _metrics_rows(m, keys=None):
    f = lambda v, u="": "—" if v is None else f"{v:.2f}{u}"
    rows = [("pairs", "Board poses used", f"{m['pairs_used']} (discarded: {m['pairs_discarded']})"), ("left", "Left camera reprojection error", f(m["rms_left_px"], " px")), ("right", "Right camera reprojection error", f(m["rms_right_px"], " px")),
            ("stereo", "Stereo reprojection error", f(m["rms_stereo_px"], " px")), ("epi", "Median epipolar error", f(m["epipolar_median_px"], " px")), ("cal", "Camera separation (calibrated)", f(m["calibrated_baseline_mm"], " mm")),
            ("meas", "Camera separation (you measured)", f(m["measured_baseline_mm"], " mm")), ("used", "Camera separation used for measuring", f(m["baseline_used_mm"], " mm")), ("scale", "Checkerboard scale check", f(m["board_square_scale"]) + " (1.00 is ideal)")]
    return [{"Check": b, "Result": c} for k, b, c in rows if keys is None or k in keys]


def _metrics_table(m):
    st.table(_metrics_rows(m))


def _summary(a):
    text, color = LEVEL_STYLE[a["level"]]
    st.markdown(f"<span class='f6-badge' style='background:{color}'>{text}</span> &nbsp; **{a['headline']}**", unsafe_allow_html=True)


def step_assessment(w):
    _heading(w)
    st.markdown("<div class='f7-lead'>This is how well the calibration fits. A failed calibration cannot be used.</div>", unsafe_allow_html=True)
    a = w.assessment(); _summary(a)
    for p in a["problems"]: st.markdown(f"⛔ {p}")
    for c in a["concerns"]: st.markdown(f"⚠️ {c}")
    if a["metrics"]: st.table(_metrics_rows(a["metrics"], ("pairs", "left", "right", "stereo", "used")))
    st.caption(a["advice"])
    with st.expander("All quality measurements and technical details"):
        if a["metrics"]: _metrics_table(a["metrics"])
        st.json(json.loads(json.dumps({k: v for k, v in (w.result or {}).items() if k not in ("model",)}, default=str)))
    if a["level"] == K.FAILED: st.button("Discard & Recalibrate", key="f7_disc5", type="primary", on_click=w.discard)
    _nav(w)


def step_decision(w, on_accept):
    _heading(w)
    a = w.assessment(); _summary(a)
    if a["level"] == K.FAILED: st.error("A failed calibration cannot be used.")
    if a["level"] == K.QUESTIONABLE:
        for c in a["concerns"]: st.markdown(f"⚠️ {c}")
        st.checkbox("I understand these concerns and want to use this calibration anyway.", value=w.acknowledged, key="f7_ack", on_change=lambda: w.acknowledge(st.session_state.get("f7_ack", False)))
    elif a["level"] == K.GOOD: st.markdown("No concerns were found. Nothing is applied until you press Accept & Continue.")
    if a["metrics"]:
        with st.expander("Quality measurements"): _metrics_table(a["metrics"])
    st.markdown("<div class='f7-gap'></div>", unsafe_allow_html=True)
    c = st.columns([1, 1.4, 1.4])
    c[0].button("Back", key="f7_back_6", on_click=w.back, width="stretch")
    c[1].button("Discard & Recalibrate", key="f7_disc6", on_click=w.discard, width="stretch")
    c[2].button("Accept & Continue", key="f7_accept", type="primary", on_click=on_accept, args=(w,), disabled=not w.can_accept(), width="stretch")


def render(on_accept, on_use_import):
    """Draw the wizard's CURRENT step only."""
    w = wizard(); st.html(CSS)
    [lambda: step_instructions(w, on_use_import), lambda: step_info(w), lambda: step_videos(w), lambda: step_process(w), lambda: step_assessment(w), lambda: step_decision(w, on_accept)][w.step]()

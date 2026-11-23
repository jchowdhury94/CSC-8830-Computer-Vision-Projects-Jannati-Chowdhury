"""Stage F1: "Annotate Wall Boundaries" -- LOCAL, ISOLATED annotation prototype.

Run it with  python Modules/Module_7/run_stage_f1_annotation_ui.py  (http://127.0.0.1:8620).  It is NOT linked from Pages/app_module7.py, the Module 7
landing page or any existing navigation, and it does not measure the room.  Frames come from the local Stage E3 test fixtures (original resolution).

All annotation logic (Stage D refinement, Stage E2 identity / digest / confirmation rules, zoom / pan mapping) lives in core.py, which does not import
Streamlit.  This file only draws widgets and keeps presentation state in st.session_state.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
ROOT = MODULE.parents[1]
for p in (str(MODULE), str(HERE), str(ROOT)):
    if p not in sys.path: sys.path.insert(0, p)

import streamlit as st                                            # noqa: E402
from PIL import Image                                              # noqa: E402
from streamlit_image_coordinates import streamlit_image_coordinates  # noqa: E402

import core                                                        # noqa: E402
import fixture                                                     # noqa: E402
import home_navigation                                             # noqa: E402  (shared Back-button style only)

ACCENT = "#C9363E"
LABEL = {"corner": "wall corner (vertical edge)", "ceilF": "ceiling / front-wall edge (near horizontal)"}
STATUS_TEXT = {"READY": ("Ready", "#2E9E5B"), "CONFIRMED": ("Confirmed", "#2E9E5B"), "NEEDS_CONFIRMATION": ("Needs your confirmation", "#E0A526"), "BLOCKED": ("Cannot be used", ACCENT),
               "NOT_REFINED": ("Not refined yet", "#9AA1AC"), "INCOMPLETE": ("Select two points", "#9AA1AC")}
CSS = f"""<style>
.f1-title {{ text-align: center; font-size: 2.6rem; font-weight: 800; margin: 0.2rem 0 0.2rem; line-height: 1.15; }}
.f1-sub {{ text-align: center; color: #9AA1AC; font-size: 1rem; margin-bottom: 0.6rem; }}
.f1-badge {{ display: inline-block; padding: 0.15rem 0.7rem; border-radius: 999px; font-weight: 700; font-size: 0.9rem; color: #fff; }}
.f1-legend span {{ display: inline-block; width: 1.6rem; height: 0.28rem; vertical-align: middle; margin-right: 0.35rem; border-radius: 2px; }}
button[kind="primary"] {{ background-color: {ACCENT}; border-color: {ACCENT}; color: #FFFFFF; }}
button[kind="primary"]:hover {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
button[kind="primary"]:disabled {{ background-color: rgba(201, 54, 62, 0.35); border-color: transparent; color: rgba(255, 255, 255, 0.6); }}
.f1-small button {{ min-height: 2.1rem; padding: 0.1rem 0.5rem; }}
</style>"""


# ------------------------------------------------------------------ session
def _boot():
    ss = st.session_state
    if "f1" in ss: return ss["f1"]
    if not fixture.available():
        st.error("Local test frames are missing. Run  python Modules/Module_7/annotation_ui/prepare_local_frames.py  once (it reads the original videos read-only).")
        st.stop()
    sess = core.AnnotationSession(fixture.load_frames(), fixture.load_calibration())
    ss["f1"] = dict(session=sess, views={}, last_click={}, msg="", redraw=0); return ss["f1"]


def _new_job():
    """Start a fresh annotation job: new job id, no points, no confirmations, no leftover widget state."""
    for k in [k for k in st.session_state if str(k).startswith("f1_")]: del st.session_state[k]
    del st.session_state["f1"]


def _vs(f1):
    s = f1["session"]; return f1["views"].get((s.set_index, s.line), core.ViewState())


def _set_vs(f1, vs):
    s = f1["session"]; f1["views"][(s.set_index, s.line)] = vs; f1["redraw"] += 1; f1["msg"] = ""


# ------------------------------------------------------------------ callbacks (mutate state before the rerun)
def cb(fn, *a):
    def run():
        f1 = st.session_state["f1"]; fn(f1, *a)
    return run


def do_zoom(f1, factor):
    s = f1["session"]; full = s.record().image_wh; _set_vs(f1, core.zoom_by(full, _vs(f1), factor))


def do_pan(f1, dx, dy):
    s = f1["session"]; _set_vs(f1, core.pan_by(s.record().image_wh, _vs(f1), dx, dy))


def do_reset_view(f1): _set_vs(f1, core.ViewState())


def do_center(f1, i):
    s = f1["session"]; pts = s.state().points
    if i < len(pts): _set_vs(f1, core.center_on(s.record().image_wh, _vs(f1), pts[i]))


def do_edit(f1, i): f1["session"].begin_edit(i); f1["redraw"] += 1; f1["msg"] = f"Click the new position of point {i + 1}."


def do_clear(f1): f1["session"].clear(); f1["redraw"] += 1; f1["msg"] = "Points cleared."


def do_submit(f1):
    try: f1["session"].submit(); f1["msg"] = ""
    except (ValueError, core.FrameIntegrityError) as e: f1["msg"] = str(e)
    f1["redraw"] += 1


def do_confirm(f1):
    ok = f1["session"].confirm(); f1["msg"] = "Boundary confirmed." if ok else "This boundary could not be confirmed."; f1["redraw"] += 1


def do_reject(f1): f1["session"].reject(); f1["redraw"] += 1; f1["msg"] = "Refinement discarded - move a point or submit again."


def do_prev(f1): f1["session"].prev(); f1["msg"] = ""; f1["redraw"] += 1


def do_next(f1): f1["session"].next(); f1["msg"] = ""; f1["redraw"] += 1


def do_goto(f1, i): f1["session"].goto(i); f1["msg"] = ""; f1["redraw"] += 1


def do_set(f1):
    f1["session"].select_set(st.session_state["f1_set_select"]); f1["msg"] = ""; f1["redraw"] += 1


# ------------------------------------------------------------------ page
def render_progress(f1):
    s = f1["session"]; k, n = s.progress()
    st.markdown(f"<div class='f1-sub'>Boundary {k} of {n} &nbsp;·&nbsp; frame set {s.set_index}</div>", unsafe_allow_html=True)
    cols = st.columns(n, gap="small")
    for i, (col, ln) in enumerate(zip(cols, s.lines)):
        label = f"{'✓ ' if s.state(ln).accepted else ''}{ln}"
        with col: st.button(label, key=f"f1_goto_{s.job_id}_{s.set_index}_{i}", type="primary" if i == s.position else "secondary", disabled=not s.reachable(i), on_click=cb(do_goto, i), width="stretch")


def render_instructions():
    with st.expander("How to annotate"):
        st.markdown("1. Zoom in on the boundary named above and pan until a long piece of it is visible.\n"
                    "2. Click two points on the visible edge, as far apart as the edge allows. The hidden corner does **not** need to be clicked.\n"
                    "3. Press **Refine boundary**. The amber dashed line is your clicked line; the green line is the refined boundary.\n"
                    "4. Check by eye that the green line follows the wall boundary you meant. Confirm it, or correct the points.\n\n"
                    "Corner = the vertical edge where two walls meet. Ceiling edge = where the ceiling meets the front wall. Door frames, the keyboard and the whiteboard are not wall boundaries.")


def render_view_controls(f1):
    s = f1["session"]; vs = core.clamp_state(s.record().image_wh, _vs(f1)); c = st.columns([1, 1, 1, 1, 1, 1, 1.4], gap="small")
    c[0].button("−", key=f"f1_zout_{s.job_id}", on_click=cb(do_zoom, 0.5), disabled=vs.zoom <= core.ZOOM_MIN, help="Zoom out", width="stretch")
    c[1].button("+", key=f"f1_zin_{s.job_id}", on_click=cb(do_zoom, 2.0), disabled=vs.zoom >= core.ZOOM_MAX, help="Zoom in", width="stretch")
    c[2].button("←", key=f"f1_pl_{s.job_id}", on_click=cb(do_pan, -0.25, 0), disabled=vs.zoom <= 1, help="Pan left", width="stretch")
    c[3].button("→", key=f"f1_pr_{s.job_id}", on_click=cb(do_pan, 0.25, 0), disabled=vs.zoom <= 1, help="Pan right", width="stretch")
    c[4].button("↑", key=f"f1_pu_{s.job_id}", on_click=cb(do_pan, 0, -0.25), disabled=vs.zoom <= 1, help="Pan up", width="stretch")
    c[5].button("↓", key=f"f1_pd_{s.job_id}", on_click=cb(do_pan, 0, 0.25), disabled=vs.zoom <= 1, help="Pan down", width="stretch")
    c[6].button("Reset view", key=f"f1_rv_{s.job_id}", on_click=cb(do_reset_view), width="stretch")
    st.caption(f"Zoom ×{vs.zoom:g} · view {core.make_view(s.record().image_wh, vs).crop} of {s.record().image_wh[0]}×{s.record().image_wh[1]} px")


def render_image(f1):
    s = f1["session"]; rec = s.record(); stt = s.state(); vs = _vs(f1)
    try: gray = s.frame()
    except core.FrameIntegrityError as e:
        st.error(f"{e}. Nothing was substituted."); return
    view = core.make_view(rec.image_wh, vs); a = core.association(s.line); K, D = (s.cal.K_left, s.cal.D_left) if a["camera"] == "left" else (s.cal.K_right, s.cal.D_right); r = stt.result
    clicked = core.line_polyline(r["initial_line"], K, D) if r and r.get("initial_line") else None
    refined = core.line_polyline(r["refined_line"], K, D) if r and r.get("refined_line") else None
    img = core.render_view(gray, view, stt.points, clicked, refined, stt.edit_index)
    key = f"f1_img_{s.job_id}_{s.set_index}_{s.line}_{stt.revision}_{f1['redraw']}"
    click = streamlit_image_coordinates(Image.fromarray(img), key=key, width=img.shape[1], cursor="crosshair", png_compression_level=1)
    if click and f1["last_click"].get(key) != click.get("unix_time"):
        f1["last_click"][key] = click.get("unix_time"); f1["msg"] = s.click(*view.click_to_full(click)); f1["redraw"] += 1; st.rerun()
    st.markdown("<div class='f1-legend'><span style='background:#FFBE00'></span>your clicked line &nbsp;&nbsp; <span style='background:#28FF6E'></span>refined boundary &nbsp;&nbsp; (clicks are full-resolution frame pixels)</div>", unsafe_allow_html=True)


def render_side(f1):
    s = f1["session"]; rec = s.record(); stt = s.state(); a = core.association(s.line)
    st.markdown(f"**{rec.label}**  \nRequired boundary: **{s.line}** - {LABEL[a['boundary_type']]}  \nImage {rec.image_wh[0]}×{rec.image_wh[1]} px · fingerprint `{rec.frame_fingerprint[:12]}…`")
    st.markdown("**Points** (full-resolution pixels)")
    for i in range(2):
        c = st.columns([3, 1.4], gap="small")
        txt = f"{i + 1}: ({stt.points[i][0]:.1f}, {stt.points[i][1]:.1f})" if i < len(stt.points) else f"{i + 1}: not set"
        c[0].markdown(txt + ("  ← click to move" if stt.edit_index == i else ""))
        c[1].button("Move", key=f"f1_edit_{s.job_id}_{s.set_index}_{s.line}_{i}", on_click=cb(do_edit, i), disabled=i >= len(stt.points), width="stretch")
    cc = st.columns(2, gap="small")
    cc[0].button("Center on 1", key=f"f1_c0_{s.job_id}", on_click=cb(do_center, 0), disabled=len(stt.points) < 1, width="stretch")
    cc[1].button("Center on 2", key=f"f1_c1_{s.job_id}", on_click=cb(do_center, 1), disabled=len(stt.points) < 2, width="stretch")
    c2 = st.columns(2, gap="small")
    c2[0].button("Clear points", key=f"f1_clear_{s.job_id}_{s.set_index}_{s.line}", on_click=cb(do_clear), disabled=not stt.points, width="stretch")
    c2[1].button("Refine boundary", key=f"f1_submit_{s.job_id}_{s.set_index}_{s.line}", type="primary", on_click=cb(do_submit), disabled=not s.can_submit(), width="stretch")
    if f1["msg"]: st.info(f1["msg"])
    render_result(f1)


def render_result(f1):
    s = f1["session"]; stt = s.state(); summary = s.state_summary(stt)
    text, color = STATUS_TEXT[summary]; st.markdown(f"<span class='f1-badge' style='background:{color}'>{text}</span>", unsafe_allow_html=True)
    r = stt.result
    if not r: return
    msgs = {i["code"]: i["message"] for i in r["issues"]}
    for code in r["failure_codes"] + r["warning_codes"]:
        icon = "⛔" if code in r["failure_codes"] else "⚠️"; st.markdown(f"{icon} {core.explain(code, msgs.get(code, ''))}")
    if not r["identity_ok"]:
        for reason in r["identity_reasons"]: st.markdown(f"⛔ {reason}")
    if r["decision"] == "ALLOWED": st.success("No problems found. You can continue - still check the green line by eye.")
    elif r["decision"] == "ALLOWED_CONFIRMED": st.success("You confirmed this boundary.")
    elif r["decision"] == "BLOCKED_UNCONFIRMED_WARNING": st.warning("Confirm only if the green line follows the wall boundary you meant; otherwise correct the points.")
    else: st.error("This boundary cannot be confirmed or used. Correct the points and refine again.")
    st.caption(core.FAIL_NOTE)
    c = st.columns(2, gap="small")
    c[0].button("Confirm boundary", key=f"f1_confirm_{s.job_id}_{s.set_index}_{s.line}_{stt.revision}", type="primary", on_click=cb(do_confirm), disabled=not s.can_confirm(), width="stretch")
    c[1].button("Reject & correct", key=f"f1_reject_{s.job_id}_{s.set_index}_{s.line}_{stt.revision}", on_click=cb(do_reject), width="stretch")
    st.button("Retry refinement", key=f"f1_retry_{s.job_id}_{s.set_index}_{s.line}_{stt.revision}", on_click=cb(do_submit), width="stretch")
    with st.expander("Diagnostic details (backend output)"):
        st.json(dict(decision=r["decision"], stage_d_status=r["status"], warning_codes=r["warning_codes"], failure_codes=r["failure_codes"], issues=r["issues"], identity_ok=r["identity_ok"], diagnostics=r["diagnostics"],
                     annotation_id=r["annotation_id"], annotation_digest=r["annotation_digest"], refined_line_digest=r["refined_line_digest"], frame_fingerprint=r["frame_fingerprint"], confirmation_valid=r["confirmation_valid"],
                     not_evaluated_checks=r["checks_not_evaluated"], note="annotation stereo consistency is NOT_EVALUATED here: no pose exists in this prototype"))


def render_nav(f1):
    s = f1["session"]; k, n = s.progress(); done = s.completed()
    c = st.columns([1, 1, 2], gap="small")
    with c[0]: home_navigation.back_button(f"f1_back_{s.job_id}", cb(do_prev))
    c[1].button("Continue", key=f"f1_next_{s.job_id}_{s.set_index}_{s.line}", type="primary", on_click=cb(do_next), disabled=not (s.state().accepted and k < n), width="stretch")
    c[2].markdown(f"Completed in set {s.set_index}: **{len(done)} / {n}**" + ("  \n(all required boundaries annotated)" if len(done) == n else ""))
    with st.expander("Completed annotations and export"):
        exp = s.export(); st.caption("Exact Stage D payloads and Stage E2 confirmation records of accepted boundaries. No measurement has been run.")
        st.download_button("Download annotations (JSON)", data=json.dumps(exp, indent=1), file_name=f"annotations_{s.job_id}.json", mime="application/json", disabled=not exp["annotations"], key=f"f1_dl_{s.job_id}")
        st.json({"accepted": [dict(set=a["set_index"], line=a["line_name"], id=a["annotation_id"]) for a in exp["annotations"]], "confirmations": len(exp["confirmations"])})
        st.button("Start a new annotation job", key=f"f1_newjob_{s.job_id}", on_click=_new_job)


def main():
    st.set_page_config(page_title="Annotate Wall Boundaries", layout="wide")
    st.html(home_navigation._CSS); st.html(CSS)
    f1 = _boot(); s = f1["session"]
    st.markdown("<h1 class='f1-title'>Annotate Wall Boundaries</h1>", unsafe_allow_html=True)
    if len(s.sets) > 1:
        c = st.columns([2, 1, 2])
        with c[1]: st.selectbox("Frame set", s.sets, index=s.sets.index(s.set_index), key="f1_set_select", on_change=cb(do_set), format_func=lambda v: f"Set {v}")
    render_progress(f1); render_instructions()
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        render_view_controls(f1); render_image(f1)
    with right: render_side(f1)
    render_nav(f1)


main()

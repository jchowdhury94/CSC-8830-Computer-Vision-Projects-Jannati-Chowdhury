"""Three-viewpoint Wall B page: frame sets (1) and the seven-line boundary marking (2), then review (3).  Streamlit drawing only; every rule lives in ``three_view_workflow``.

It reuses the existing annotation components: the crop-based zoom / pan and the exactly invertible click -> original-pixel mapping of ``annotation_ui.core`` (``f1``), the same two-click
image component, and the shared sandboxed preview / extraction worker.  Session-state keys all start with ``f8i_tv_`` (parked per wall by ``f8_walls``).  A click is converted back to the
ORIGINAL full-resolution, lens-distorted frame before anything is stored; previews are reduced copies for display only.  Nothing here matches features or measures.
"""
from __future__ import annotations

import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

import f8_ui_text as UT
import f8_visitor_service as S
import f8_walls as WL
import room_frames as RF
import room_videos as RV
import three_view_workflow as TVW

f1 = S.F.f1core
STATUS_ICON = {"marked": "ok", "unavailable": "warn", "partial": "edit", "empty": "todo", "stale": "bad"}
STATUS_TEXT = {"marked": "marked", "unavailable": "not visible", "partial": "one point", "empty": "not marked", "stale": "mark again"}
INTRO = ("Three viewpoints (left end, middle, right end of the wall) are used together. Choose the frames, then mark the wall corners and the ceiling line. "
         "Keep each camera pair still while you record: the same frame number is used in all six videos.")
GUIDE = {
    "vertical": ("Click two points on the vertical edge where the two walls meet (the wall corner): one near the top and one near the bottom of the part you can see, far apart. "
                 "If the lower part is hidden by furniture, mark the visible part of the edge."),
    "horizontal": ("Click two points along the line where the wall meets the ceiling. This line is optional (it is not used for the length). "
                   "If you cannot see it, choose “Not visible in this frame”."),
}
END_TEXT = {1: "the LEFT end of the wall", 3: "the RIGHT end of the wall"}


def _flash(msg):
    st.session_state["f8i_tv_msg"] = msg


def _try(fn, *a, **k):
    try:
        return fn(*a, **k)
    except RF.WorkflowError as e:
        _flash(str(e))


def _go_safe(sess, step):
    try:
        sess.go(step)
    except S.ServiceError as e:
        st.session_state["f8i_msg"] = str(e)


def render_frames_3(svc, sess):
    ss = st.session_state
    stt = svc.frame_state(sess)
    top = st.container(key="f8i_tv_top")
    with top:
        for n in sess.frame_notes:
            st.warning(n)
    sess.frame_notes.clear()
    if stt["state"] == "no_calibration":
        st.warning("There is no accepted calibration in this session. Go back and accept a calibration first."); return
    if stt["state"] in ("expired", "not_ready"):
        st.error("Your room videos are no longer usable here." if stt["state"] == "expired" else "Your room videos are not all valid.")
        for p in stt["problems"]:
            st.markdown(UT.line("bad", p))
        st.button("Upload the videos again", key="f8i_tv_reup", type="primary", on_click=lambda: _go_safe(sess, "room")); return
    wf = stt["workflow"]; ctx = svc.calibration_context(sess)
    with top:
        if ss.get("f8i_tv_msg"):
            st.warning(ss.pop("f8i_tv_msg"))
        st.success(f"Accepted calibration · {UT.baseline_line(ctx['baseline_mm'], ctx.get('baseline_source'))} · Image {ctx['image_size_wh'][0]}×{ctx['image_size_wh'][1]} px · every selection stays tied to it.")
        st.markdown(INTRO)
    with st.container(border=True, key="f8i_tv_box1"):
        _step1(svc, sess, wf)
    if not wf.confirmed:
        st.caption(":material/lock: Boundaries open after you confirm the frame selection."); return
    with st.container(border=True, key="f8i_tv_box2"):
        _step2(svc, sess, wf)
    with st.container(border=True, key="f8i_tv_box3"):
        _step3(svc, sess, wf)


# ------------------------------------------------------------------ 1 · frames
def _set_frame(wf, i):
    _try(wf.set_frame, i, st.session_state.get(f"f8i_tv_fr{i}"))


def _set_count(wf):
    n = st.session_state.get("f8i_tv_nsets")
    if n != wf.n_sets:
        _try(wf.auto_select, n); st.session_state["f8i_tv_idx"] = 0


def _step1(svc, sess, wf):
    ss = st.session_state
    st.subheader("1 · Choose the frames", anchor=False)
    st.caption("Suggested frames are spread across the videos. Check that every preview shows the wall end you will mark. You can change any frame number.")
    last = wf.shortest() - 1
    ss["f8i_tv_nsets"] = wf.n_sets
    st.selectbox("Number of frame sets", list(range(TVW.MIN_SETS, TVW.MAX_SETS + 1)), key="f8i_tv_nsets", on_change=_set_count, args=(wf,),
                 help="Changing this chooses new frames and discards any boundaries you marked.")
    for i in range(wf.n_sets):
        with st.container(border=True, key=f"f8i_tv_set{i}"):
            ss[f"f8i_tv_fr{i}"] = int(wf.sets[i][TVW.SLOTS[0]]) if len({wf.sets[i][s] for s in TVW.SLOTS}) == 1 else ss.get(f"f8i_tv_fr{i}", 0)
            c = st.columns([1, 3])
            c[0].number_input(f"Frame set {i + 1} · frame number", min_value=0, max_value=last, step=1, key=f"f8i_tv_fr{i}", on_change=_set_frame, args=(wf, i))
            if len({wf.sets[i][s] for s in TVW.SLOTS}) > 1:
                c[1].caption("This set uses different frame numbers in different videos.")
            with st.spinner("Preparing previews…"):
                try:
                    pv = wf.preview_set(i)
                except RF.WorkflowError as e:
                    st.error(str(e)); continue
            cols = st.columns(3, gap="small")
            for k, v in enumerate(TVW.VIEWPOINTS):
                with cols[k]:
                    for side in ("left", "right"):
                        slot = f"p{v}_{side}"
                        st.image(pv[slot]["path"], caption=f"Viewpoint {v} · {side} camera · frame {wf.sets[i][slot]}", width="stretch")
    for m in wf.duplicate_frames():
        st.markdown(UT.line("warn", m))
    with st.expander("Use a different frame number for one video (advanced)"):
        st.caption("Normally the same number is used in all six videos. Change one only if its left/right recordings are offset in time. The boundaries of that frame set are discarded.")
        sel = st.selectbox("Frame set", list(range(1, wf.n_sets + 1)), key="f8i_tv_adv_set")
        for s in TVW.SLOTS:
            ss[f"f8i_tv_adv_{s}_{sel}"] = int(wf.sets[sel - 1][s])
            st.number_input(RV.SLOT_INFO[s][2], min_value=0, max_value=wf.source(s).frame_count - 1, step=1, key=f"f8i_tv_adv_{s}_{sel}",
                            on_change=lambda s=s, sel=sel: _try(wf.set_slot_frame, sel - 1, s, st.session_state.get(f"f8i_tv_adv_{s}_{sel}")))
    if wf.confirmed:
        st.markdown(UT.line("ok", "Frames confirmed and read at full quality. Changing a frame number asks you to confirm again."))
    elif st.button("Confirm these frames", key="f8i_tv_confirm", type="primary"):
        with st.spinner("Reading the six frames of every set at full quality…"):
            _try(wf.extract_frames)
        st.rerun()


# ------------------------------------------------------------------ 2 · boundaries
def _toolbar(views, vkey, full_wh, vs):
    def zoom(f): views[vkey] = f1.zoom_by(full_wh, vs, f)
    def pan(dx, dy): views[vkey] = f1.pan_by(full_wh, vs, dx, dy)
    with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", wrap=False, gap="xsmall", key="f8i_tv_toolbar"):
        if vs.zoom > 1:
            for j, (lab, dx, dy, tip) in enumerate((("←", -0.25, 0, "Pan left"), ("→", 0.25, 0, "Pan right"), ("↑", 0, -0.25, "Pan up"), ("↓", 0, 0.25, "Pan down"))):
                st.button(lab, key=f"f8i_tv_pn{j}", on_click=pan, args=(dx, dy), help=tip)
        st.button("−", key="f8i_tv_zo", on_click=zoom, args=(0.5,), disabled=vs.zoom <= f1.ZOOM_MIN, help="Zoom out")
        st.button("+", key="f8i_tv_zi", on_click=zoom, args=(2.0,), disabled=vs.zoom >= f1.ZOOM_MAX, help="Zoom in")
        st.button("Fit", key="f8i_tv_rv", on_click=lambda: views.pop(vkey, None), help="Fit the whole image")


def _walk(wf):
    return [(i, n) for i in range(wf.n_sets) for n in TVW.LINE_NAMES]


def _step2(svc, sess, wf):
    ss = st.session_state
    st.subheader("2 · Mark the wall boundaries", anchor=False)
    walk = _walk(wf); total = len(walk); idx = min(max(int(ss.get("f8i_tv_idx", 0)), 0), total - 1); ss["f8i_tv_idx"] = idx
    i, line = walk[idx]; v, cam, ori, req, bd, role = TVW.LINE_INFO[line]; slot = TVW.slot_of(line)
    st.markdown(f"**{idx + 1} of {total}** · Frame set {i + 1} · {TVW.label(line)}" + ("" if req else " · optional"))
    st.caption(GUIDE[ori] + (f" This is the corner at {END_TEXT[v]}." if ori == "vertical" else ""))
    state = wf.state(i, line); status = wf.line_status(i, line)
    W, H = (int(x) for x in wf.cal.image_size)
    try:
        gray = wf.frame_gray(i, slot)
    except RF.WorkflowError as e:
        st.error(str(e)); return
    views = ss.setdefault("f8i_tv_view", {}); vkey = (i, slot, wf.full[(i, slot)]["fingerprint"]); vs = views.get(vkey, f1.ViewState())
    view = f1.make_view((W, H), vs, max_wh=(836, 900))
    _toolbar(views, vkey, (W, H), vs)
    img = f1.render_view(gray, view, state.points, manual_pts=state.points if len(state.points) == 2 else None)
    if state.unavailable:
        st.image(img, width="stretch"); st.markdown(UT.line("warn", "Marked as not visible in this frame." + (" This frame set cannot be used for the length." if req else "")))
        click = None
    else:
        click = streamlit_image_coordinates(Image.fromarray(img), key=f"f8i_tv_img_{i}_{line}_{state.revision}_{vs.zoom}_{vs.cx}_{vs.cy}{WL.suffix(sess)}", width="stretch", cursor="crosshair", png_compression_level=1)
    lc = ss.setdefault("f8i_tv_last", {}); ck = (i, line)
    if click and lc.get(ck) != (click.get("unix_time"), state.revision):
        lc[ck] = (click.get("unix_time"), state.revision + 1)
        x, y = view.click_to_full(click); _try(wf.click, i, line, x, y); st.rerun()
    st.caption("Click the first point, then the second point. Zoom in with + for precision; the points are stored on the full-quality frame.")
    if len(state.points) == 1:
        st.caption("Now click the second point.")
    if status == "marked":
        st.markdown(UT.line("ok", "Line marked."))
    b = st.columns(4)
    if b[0].button("Undo last point", key=f"f8i_tv_undo_{idx}", disabled=not state.points, width="stretch"):
        wf.undo(i, line); st.rerun()
    if b[1].button("Reset line", key=f"f8i_tv_reset_{idx}", disabled=not (state.points or state.unavailable), width="stretch"):
        wf.reset(i, line); st.rerun()
    if state.unavailable:
        if b[2].button("It is visible after all", key=f"f8i_tv_vis_{idx}", width="stretch"):
            wf.set_unavailable(i, line, False); st.rerun()
    elif b[2].button("Not visible in this frame", key=f"f8i_tv_na_{idx}", width="stretch", help="Use this for a boundary that is hidden or outside the picture."):
        wf.set_unavailable(i, line, True); st.rerun()
    if i > 0 and b[3].button("Copy from frame set 1", key=f"f8i_tv_copy_{idx}", disabled=wf.line_status(0, line) != "marked", width="stretch",
                             help="Start from the line you marked in frame set 1. Check it: the cameras may have moved between the frames."):
        _try(wf.copy_points, 0, i, line); st.rerun()
    nav = st.columns(2)
    if nav[0].button("Back", key="f8i_tv_back", disabled=idx == 0, width="stretch"):
        ss["f8i_tv_idx"] = idx - 1; st.rerun()
    done = status in ("marked", "unavailable")
    if nav[1].button("Next", key="f8i_tv_next", type="primary", disabled=idx == total - 1 or not done, width="stretch"):
        ss["f8i_tv_idx"] = idx + 1; st.rerun()
    with st.expander("Technical Details"):
        st.markdown(f"Frame set {i + 1} · `{slot}` frame {wf.sets[i][slot]} · {ori} · points (original {W}×{H} pixels): " + (", ".join(f"({p[0]:.0f}, {p[1]:.0f})" for p in state.points) or "none"))
        if v == 2:
            st.caption("The middle viewpoint's ceiling line is stored but not used by the reconstruction (it has not been validated as a geometric constraint).")


# ------------------------------------------------------------------ 3 · review
def _jump(idx):
    st.session_state["f8i_tv_idx"] = idx


def _step3(svc, sess, wf):
    st.subheader("3 · Review and continue", anchor=False)
    rd = wf.readiness()
    for i in range(wf.n_sets):
        usable = i in rd["usable_sets"]
        st.markdown(UT.line("ok" if usable else "warn", f"**Frame set {i + 1}** · " + ("usable" if usable else "cannot be used yet")))
        with st.expander(f"Frame set {i + 1} boundaries"):
            for n in TVW.LINE_NAMES:
                s = wf.line_status(i, n)
                c = st.columns([5, 2])
                c[0].markdown(UT.line(STATUS_ICON[s], f"{TVW.label(n)} — {STATUS_TEXT[s]}" + ("" if TVW.LINE_INFO[n][3] else " (optional)")))
                c[1].button("Change", key=f"f8i_tv_chg_{i}_{n}", on_click=_jump, args=(i * len(TVW.LINE_NAMES) + TVW.LINE_NAMES.index(n),))
    for m in rd["missing"]:
        st.markdown(UT.line("todo", m))
    for m in rd["notes"]:
        st.markdown(UT.line("warn", m))
    if rd["ready"]:
        st.success(f"{len(rd['usable_sets'])} of {wf.n_sets} frame sets are complete." + (" The others are excluded and will be reported." if len(rd["usable_sets"]) < wf.n_sets else ""))
    st.button("Continue to Room Measurement", key="f8i_tv_continue", type="primary", disabled=not rd["ready"], on_click=_continue, args=(svc, sess))


def _continue(svc, sess):
    try:
        svc.go_measure(sess)
    except S.ServiceError as e:
        st.session_state["f8i_msg"] = str(e)

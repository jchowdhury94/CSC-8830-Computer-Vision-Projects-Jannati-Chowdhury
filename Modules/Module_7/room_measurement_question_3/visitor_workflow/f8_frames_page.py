"""Stage F8.31D-6 page: automatic frames, six wall boundaries marked once, refinement in every frame set (Streamlit drawing only; every rule lives in ``room_frames`` / ``f8_visitor_service``).

Three numbered step cards with sequential unlocking:  1 Select frames automatically  ->  2 Mark six wall boundaries (once)  ->  3 Review and continue.
This reproduces the ORIGINAL experiment: the frames come from the documented rule, and each boundary is ONE seed (two clicks) reused for all five frame sets and refined by the unchanged
Method J line functions.  Session-state keys all start with ``f8i_``.  Previews are reduced copies for DISPLAY; clicks are always converted back to the original full-resolution frame
(``f1core.View.click_to_full``) before anything is stored.  Nothing here runs feature matching, pose estimation or a measurement."""
from __future__ import annotations

import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

import f8_ui_text as UT
import f8_walls as WL
import f8_visitor_service as S
import room_frames as RF
import room_videos as RV
from room_measurement import visitor_annotations as V

F = S.F
f1 = F.f1core
MODES = ("1 · Select frames automatically", "2 · Mark wall boundaries", "3 · Review and continue")          # the three step cards (the session key f8i_fs_mode records the current one)
LINE_TEXT = {"corner": UT.LINE_HELP["corner"], "ceilF": UT.LINE_HELP["ceilF"]}
LINE_STATUS_TEXT = {"ok": "Refined", "ok_warning": "Refined (check the line)", "failed": "Could not be refined", "not_run": "Not run"}
STILL_RIG_TEXT = "Keep each camera pair still while you record: one boundary mark is reused in all five frame sets."
LIMITS_TEXT = "Frames are chosen automatically. You confirm six wall boundaries once; the app refines them in all five frame sets."


def boundary_svg(kind: str, camera: str) -> str:
    """A small schematic that shows WHICH physical edge is meant (drawn in red).  A diagram of the idea, not of the visitor's room."""
    wall = "#2A2E37"; edge = "#C9363E"; line = "#9AA1AC"
    if kind == "corner":
        body = (f"<polygon points='20,30 90,45 90,150 20,160' fill='{wall}' stroke='{line}'/><polygon points='90,45 170,28 170,162 90,150' fill='#20242C' stroke='{line}'/>"
                f"<line x1='90' y1='45' x2='90' y2='150' stroke='{edge}' stroke-width='5'/><text x='98' y='100' fill='{edge}' font-size='11'>wall corner</text>")
    else:
        body = (f"<polygon points='20,60 170,60 170,165 20,165' fill='{wall}' stroke='{line}'/><polygon points='20,60 40,22 150,22 170,60' fill='#14171C' stroke='{line}'/>"
                f"<line x1='20' y1='60' x2='170' y2='60' stroke='{edge}' stroke-width='5'/><text x='60' y='52' fill='{edge}' font-size='11'>ceiling / wall</text>")
    return (f"<svg viewBox='0 0 190 175' width='190' role='img' aria-label='Diagram: the {UT._LINE_KIND[kind].lower()} to mark is drawn in red' xmlns='http://www.w3.org/2000/svg'>{body}</svg>")


def _flash(msg):
    st.session_state["f8i_fs_msg"] = msg


def _try(fn, *a, **k):
    try:
        return fn(*a, **k)
    except RF.WorkflowError as e:
        _flash(str(e))
    except S.ServiceError as e:
        _flash(str(e))
    return None


def _mode_default():
    return st.session_state.setdefault("f8i_fs_mode", MODES[0])


def _apply_pending_mode(ss):
    """A one-shot request (``f8i_fs_goto``) to record which step is current; consumed at the top of the next run.  Unknown values are ignored."""
    goto = ss.pop("f8i_fs_goto", None)
    if goto in MODES:
        ss["f8i_fs_mode"] = goto
    ss.setdefault("f8i_fs_mode", MODES[0])


def step_states(wf) -> dict:
    """Pure: ``{1|2|3: 'locked'|'active'|'done'}``.  Derived from the workflow on every run (nothing is cached), so an invalidated input re-locks the later steps at once."""
    s1 = bool(wf.frames_ready())
    s2 = s1 and wf.seeds_complete() and wf.seeds_applied()
    return {1: "done" if s1 else "active", 2: "locked" if not s1 else "done" if s2 else "active", 3: "locked" if not s2 else "active"}


def current_mode(ss, wf) -> str:
    """The step card the visitor is working on, as a pure function of the workflow (used for the scroll-to-top key BEFORE the page draws, and by the page itself, so they cannot disagree).
    Frame selection is automatic, so it only counts as the current step while it has FAILED; otherwise the visitor is at step 2 until the boundaries were applied, then at step 3."""
    if wf is None or getattr(wf, "closed", False):
        return MODES[0]
    if not wf.frames_ready():
        err = (ss.get("f8i_fs_auto") or {}).get(id(wf))
        return MODES[0] if err not in (None, "ok") else MODES[1]
    return MODES[2] if wf.seeds_complete() and wf.seeds_applied() else MODES[1]


def render_frames(svc, sess):
    ss = st.session_state
    _apply_pending_mode(ss)
    stt = svc.frame_state(sess)
    top = st.container(key="f8i_fs_top")               # ONE fixed slot for every message above the steps, so a changing number of notices never shifts the step cards
    with top:
        for n in sess.frame_notes:
            st.warning(n)
    sess.frame_notes.clear()
    if stt["state"] == "no_calibration":
        st.warning("There is no accepted calibration in this session. Go back and accept a calibration first.")
        return
    if stt["state"] in ("expired", "not_ready"):
        st.error("Your room videos are no longer usable here." if stt["state"] == "expired" else "Your room videos are not all valid.")
        for p in stt["problems"]:
            st.markdown(UT.line("bad", p))
        if stt["state"] == "expired":
            st.info(UT.retention_short(svc.room_store.limits) + " Go back and upload the missing videos again; selections that depended on an expired video were discarded, the others are kept.")
        st.button("Upload the videos again", key="f8i_fs_reup", type="primary", on_click=lambda: _go_safe(sess, "room"))
        return
    wf = stt["workflow"]
    ctx = svc.calibration_context(sess)
    _auto_frames(wf)                                    # runs once per workflow; a failure is remembered and explained, never retried in a loop
    with top:
        if ss.get("f8i_fs_msg"):
            st.warning(ss.pop("f8i_fs_msg"))
        st.success(f"Accepted calibration · {UT.baseline_line(ctx['baseline_mm'], ctx.get('baseline_source'))} · Image {ctx['image_size_wh'][0]}×{ctx['image_size_wh'][1]} px · "
                   "every selection below stays tied to it.")
        st.markdown(LIMITS_TEXT)
    states = step_states(wf)
    ss["f8i_fs_mode"] = current_mode(ss, wf)
    for n in (1, 2, 3):
        state = states[n]
        with st.container(border=True, key=f"f8i_stepbox_f{n}"):                       # a stable box; only the heading wrapper carries the state
            with st.container(key=f"f8i_stmark_f{n}_{state}"):
                _fs_head(n, state)
            if state == "locked":
                continue
            if n == 1:
                _step1(svc, sess, wf)
            elif n == 2:
                _step2(svc, sess, wf)
            else:
                _step3(svc, sess, wf)


def _go_safe(sess, step):
    try:
        sess.go(step)
    except S.ServiceError as e:
        st.session_state["f8i_msg"] = str(e)


FS_TITLES = {1: "Select Frames Automatically", 2: "Confirm Wall Boundaries", 3: "Review and Continue"}
FS_LOCK = {2: "Locked. The frames are selected first.", 3: "Locked. Confirm all six wall boundaries first."}


def _fs_head(n, state):
    st.subheader(f"Part {n} · {FS_TITLES[n]}", anchor=False)
    if state == "locked":
        st.caption(f":material/lock: {FS_LOCK[n]}")
    elif state == "done":
        st.caption(":green[:material/check_circle: Completed]")
    else:
        st.caption(":red[:material/arrow_forward: Current step]")


# ------------------------------------------------------------------ 1 · frames (automatic)
def _auto_frames(wf):
    """Run the original frame rule once for this workflow.  ``f8i_fs_auto`` maps id(workflow) -> 'ok' | error text, so a rerun never repeats the (video-reading) work and a
    failure is shown with its reason until the visitor presses Try again."""
    ss = st.session_state; rec = ss.setdefault("f8i_fs_auto", {}); key = id(wf)
    if wf.frames_ready() or key in rec or wf.pairs:
        return
    with st.spinner("Choosing five sharp frames from each video…"):
        try:
            wf.auto_select(); rec[key] = "ok"
        except (RF.WorkflowError, S.ServiceError) as e:
            rec[key] = str(e)
            for k in [k for k in wf.pairs]:
                wf.drop_pair(*k)                       # nothing half-chosen is kept: no index was invented


def _step1(svc, sess, wf):
    ss = st.session_state; rec = ss.get("f8i_fs_auto", {}); err = rec.get(id(wf))
    if err not in (None, "ok") and not wf.frames_ready():
        st.markdown(UT.line("bad", "The frames could not be selected automatically."))
        st.markdown(f"**Reason:** {err}")
        st.markdown("**What to do:** check that all four videos are valid and long enough (at least five frames each), and that they are recorded at the calibrated size. You can try again, "
                    "or switch on **Change Frames** below to choose frames yourself.")
        if st.button("Try again", key="f8i_fs_retry", type="primary"):
            rec.pop(id(wf), None); st.rerun()
    elif wf.all_pairs_confirmed() and not wf.frames_ready():
        with st.spinner("Reading the selected frames at full quality…"):
            _try(wf.extract_confirmed)
        st.rerun()
    elif wf.frames_ready():
        auto = all(wf.pairs[(vp, i)].get("auto") for vp in RF.VIEWPOINTS for i in range(wf.n_sets))
        st.markdown(UT.line("ok", f"**{wf.n_sets} frame sets** were selected " + ("automatically" if auto else "(some of them by you)") + ": each of the four videos contributed its own sharp frames, "
                            "chosen independently, as in the original experiment."))
        for m in wf.duplicate_frames():
            st.markdown(UT.line("warn", m))
    if wf.pairs and st.toggle("View Selected Frames", key="f8i_fs_view"):
        _view_frames(wf)
    if wf.pairs and st.toggle("Change Frames", key="f8i_fs_manual", help="Choose different frames yourself. Changing a frame discards the boundary results that depended on it."):
        _manual(wf)


def _view_frames(wf):
    for vp in RF.VIEWPOINTS:
        st.markdown(f"#### {RV.VIEWPOINT_TITLES[vp]}")
        for i in range(wf.n_sets):
            p = wf.pairs.get((vp, i))
            if p is None:
                continue
            try:
                imgs = {side: wf.preview(f"p{vp}_{side}", p[side]) for side in RF.SIDES}
            except RF.WorkflowError as e:
                st.error(str(e)); return
            cols = st.columns(2, gap="small")
            for col, side in zip(cols, RF.SIDES):
                with col:
                    st.image(imgs[side]["path"], caption=f"Viewpoint {vp} · {side.title()} camera · frame set {i + 1} · video frame {p[side]}", width="stretch")


def _manual(wf):
    """The earlier technical controls: frame numbers, offsets, the frame-set table.  Hidden unless the visitor switches on Change Frames."""
    ss = st.session_state
    st.caption("Frame numbers start at 0 and count frames exactly as decoded. Each camera's frame is chosen on its own; the offset only shows how far apart the two frame numbers are.")
    vp = st.radio("Viewpoint", RF.VIEWPOINTS, format_func=lambda v: RV.VIEWPOINT_TITLES[v], key="f8i_fs_vp", horizontal=True)
    rows = []
    for i in range(wf.n_sets):
        info = wf.pair_info(vp, i)
        rows.append({"Set": i + 1, "Left frame": "—" if info is None else info["left"], "Right frame": "—" if info is None else info["right"],
                     "Offset (frames)": "—" if info is None else f"{info['offset_frames']:+d}", "≈ seconds": "—" if info is None else f"{info['offset_seconds']:+.2f}",
                     "Status": "not chosen" if info is None else "accepted" if info["confirmed"] else "to confirm" if info["previews_viewed"] else "preview it"})
    st.table(rows)
    i = int(st.select_slider("Edit frame set", options=list(range(1, wf.n_sets + 1)), key=f"f8i_fs_set{vp}")) - 1
    _pair_editor(wf, vp, i)


def _pair_editor(wf, vp, i):
    L_, R_ = wf.source(f"p{vp}_left"), wf.source(f"p{vp}_right"); cur = wf.pairs.get((vp, i)); k = f"f8i_fs_{vp}_{i}_{cur['left'] if cur else 'n'}_{cur['right'] if cur else 'n'}"
    with st.container(border=True):
        st.markdown(f"**Viewpoint {vp} · frame set {i + 1}**")
        c = st.columns(2)
        left = c[0].number_input(f"Left camera frame (0–{L_.frame_count - 1})", 0, L_.frame_count - 1, cur["left"] if cur else L_.frame_count // 2, 1, key=k + "_l")
        right = c[1].number_input(f"Right camera frame (0–{R_.frame_count - 1})", 0, R_.frame_count - 1, cur["right"] if cur else R_.frame_count // 2, 1, key=k + "_r")
        st.caption(f"Offset right − left: {int(right) - int(left):+d} frames (≈ {RF.offset_seconds(int(left), int(right), L_.fps, R_.fps):+.2f} s at {L_.fps:g} / {R_.fps:g} fps).")
        b = st.columns(4)
        if b[0].button("Use these frames", key=k + "_set", help="Create or replace this pair. A replaced pair must be confirmed again."):
            _try(wf.set_pair, vp, i, int(left), int(right)); st.rerun()
        if b[1].button("◀ −1 left", key=k + "_m1"):
            _try(wf.set_pair, vp, i, int(left) - 1, int(right)); st.rerun()
        if b[2].button("+1 left ▶", key=k + "_p1"):
            _try(wf.set_pair, vp, i, int(left) + 1, int(right)); st.rerun()
        off = b[3].number_input("Keep offset", -R_.frame_count, R_.frame_count, int(right) - int(left), 1, key=k + "_off", help="Move the left frame and pick the right frame at this offset.")
        e = st.columns(3)
        if e[0].button("Apply offset to the left frame", key=k + "_applyoff"):
            _try(wf.set_pair_by_offset, vp, i, int(left), int(off)); st.rerun()
        cur = wf.pairs.get((vp, i))
        if cur is not None:
            if e[1].button("Preview both frames", key=k + "_prev"):
                with st.spinner("Reading the two frames…"):
                    _try(wf.preview, f"p{vp}_left", cur["left"]); _try(wf.preview, f"p{vp}_right", cur["right"])
                st.rerun()
            if wf.previews_viewed(vp, i):
                pv = st.columns(2)
                for col, side in zip(pv, RF.SIDES):
                    p = wf.preview(f"p{vp}_{side}", cur[side])
                    with col:
                        st.image(p["path"], caption=f"{side.title()} camera · frame {cur[side]} · original {p['full_wh'][0]}×{p['full_wh'][1]} px (shown reduced)", width="stretch")
                st.caption("Do the two images show the same scene from the two cameras, with nothing moved? Look at the wall edge, the floor and any objects.")
            if cur["confirmed"]:
                if e[2].button("Un-confirm", key=k + "_unc"):
                    wf.unconfirm_pair(vp, i); st.rerun()
            elif e[2].button("Confirm this pair", key=k + "_conf", type="primary", disabled=not wf.previews_viewed(vp, i), help="Available after both previews are shown."):
                _try(wf.confirm_pair, vp, i); st.rerun()




# ------------------------------------------------------------------ 2 · six boundaries, one screen each (F8.31F)
BOUNDARY_ORDER = tuple(RF.REQUIRED_LINES)          # viewpoint 1: left corner, right corner, ceiling edge; then viewpoint 2
INSTRUCTION = {"corner": "Click two points on the tall vertical line where the two walls meet.", "ceilF": "Click two points on the line where the front wall meets the ceiling."}
LEGEND_TEXT = "Amber dashed line: automatic suggestion, not yet confirmed. Red line: your boundary. Green line: refined result (shown only after refinement)."
SUGGEST_TEXT = "The amber dashed line is a suggestion. Check that it follows the edge. It is used only if you press Looks Correct."


def _guide(line):
    """F8.31H: heading, instruction and a small red-edge example for the ACTIVE boundary, directly above the image.  Explains the direction; nothing snaps or constrains the clicks."""
    g = UT.boundary_guide(line); kind = line.split("_")[1]
    c = st.columns([3, 1], vertical_alignment="center")
    with c[0]:
        st.markdown(f"**{g['heading']}**")
        st.markdown(g["body"])
        if g["note"]:
            st.caption(g["note"])
    with c[1]:
        st.markdown(boundary_svg(kind, ""), unsafe_allow_html=True)


def _suggestions(wf):
    """The optional F2 suggestions for this workflow's set-0 frames, computed once and re-computed if a set-0 frame changed.  Never raises (F2 failure = no suggestions)."""
    import boundary_suggestions as BS
    ss = st.session_state; store = ss.setdefault("f8i_bd_sugg", {}); key = BS.fingerprints(wf); cur = store.get(id(wf))
    if cur is None or cur["key"] != key:
        with st.spinner("Looking for suggested boundaries on your frames…"):
            cur = BS.suggest_for_workflow(wf)
        store.clear(); store[id(wf)] = cur
    return cur["by_line"]


def completion_text(srcs) -> str:
    """Step 2 reports ONLY what happened: the boundaries were marked or confirmed.  Whether refinement worked is reported by step 3, from the backend result."""
    manual = all((srcs or {}).get(n) == "visitor" for n in BOUNDARY_ORDER)
    return f"All six wall boundaries have been {'marked' if manual else 'confirmed'}. Review the refinement results below."


def _correct_boundary(wf, line):
    """Step 3 -> the exact boundary.  Its points are kept (so they can be seen and replaced) and only ITS refinement results are dropped; the other five seeds stay."""
    ss = st.session_state
    try:
        wf.reopen_boundary(line)
    except RF.WorkflowError as e:
        _flash(str(e)); return
    ss["f8i_bd_idx"] = BOUNDARY_ORDER.index(line)


def _toolbar(views, vkey, full_wh, vs):
    """Compact zoom toolbar directly above the image's top-right corner (no expander).  Callbacks only change the zoom / pan state; they never touch a stored point."""
    def zoom(f):
        views[vkey] = f1.zoom_by(full_wh, vs, f)

    def pan(dx, dy):
        views[vkey] = f1.pan_by(full_wh, vs, dx, dy)

    with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", wrap=False, gap="xsmall", key="f8i_bd_toolbar"):
        if vs.zoom > 1:
            for j, (lab, dx, dy, tip) in enumerate((("←", -0.25, 0, "Pan left"), ("→", 0.25, 0, "Pan right"), ("↑", 0, -0.25, "Pan up"), ("↓", 0, 0.25, "Pan down"))):
                st.button(lab, key=f"f8i_bd_pn{j}", on_click=pan, args=(dx, dy), help=tip)
        st.button("−", key="f8i_bd_zo", on_click=zoom, args=(0.5,), disabled=vs.zoom <= f1.ZOOM_MIN, help="Zoom out")
        st.button("+", key="f8i_bd_zi", on_click=zoom, args=(2.0,), disabled=vs.zoom >= f1.ZOOM_MAX, help="Zoom in")
        st.button("Fit", key="f8i_bd_rv", on_click=lambda: views.pop(vkey, None), help="Fit the whole image")


def _step2(svc, sess, wf):
    ss = st.session_state
    if wf.seeds_applied():
        st.markdown(UT.line("ok", completion_text(ss.get("f8i_bd_src"))))
        if st.button("Change a boundary", key="f8i_bd_reopen"):
            bad = [k for k, n in enumerate(BOUNDARY_ORDER) if any(wf.line_report(i, n)["status"] == "failed" for i in range(wf.n_sets))]
            for k in bad:                                       # a boundary that could not be refined is marked again by hand (the same suggestion would fail the same way)
                wf.seed_clear(BOUNDARY_ORDER[k]); ss.setdefault("f8i_bd_mode", {})[BOUNDARY_ORDER[k]] = "marking"; ss.setdefault("f8i_bd_src", {}).pop(BOUNDARY_ORDER[k], None)
            wf.seeds_reopen(); ss["f8i_bd_idx"] = bad[0] if bad else 0; st.rerun()
        return
    sugg = _suggestions(wf)
    n = len(BOUNDARY_ORDER); idx = min(max(int(ss.get("f8i_bd_idx", 0)), 0), n - 1); ss["f8i_bd_idx"] = idx
    line = BOUNDARY_ORDER[idx]; vk = line[:2]; placement, camera = V.VIEW_PLACEMENT[vk]; kind = line.split("_")[1]
    modes = ss.setdefault("f8i_bd_mode", {}); srcs = ss.setdefault("f8i_bd_src", {}); sg = sugg[line]
    pts = wf.seed_points(line); confirmed = len(pts) == 2
    if confirmed and srcs.get(line) is None:
        srcs[line] = "visitor"
    if not pts:
        srcs.pop(line, None)
    st.markdown(f"**{idx + 1} of {n}** · {UT.boundary_label(line)}")
    if idx == 0:
        st.caption(STILL_RIG_TEXT)
    _guide(line)
    i = RF.SEED_SET; state = wf.state(i, line)
    views = ss.setdefault("f8i_an_view", {}); W, H = (int(v) for v in wf.cal.image_size)
    vkey = (i, line, wf.full[(i, vk)]["fingerprint"] if (i, vk) in wf.full else None)           # a different camera frame starts at Fit
    vs = views.get(vkey, f1.ViewState())
    view = f1.make_view((W, H), vs, max_wh=(836, 900))
    try:
        gray = wf.frame_gray(i, vk)
    except RF.WorkflowError as e:
        st.error(str(e)); return
    K, D = ((wf.cal.K_left, wf.cal.D_left) if camera == "left" else (wf.cal.K_right, wf.cal.D_right))
    show_sug = sg["available"] and not confirmed and modes.get(line) != "marking"                  # amber dashed: an unconfirmed suggestion only
    sug_poly = f1.line_polyline(sg["line_undistorted"], K, D) if show_sug else None
    res = state.result
    ref_poly = f1.line_polyline(res.refined_line, K, D) if (state.accepted and res is not None and getattr(res, "refined_line", None)) else None      # green: ONLY a real accepted refinement
    _toolbar(views, vkey, (W, H), vs)
    img = f1.render_view(gray, view, state.points, sug_poly, ref_poly, state.edit_index, manual_pts=state.points if confirmed else None)    # red: the two SAVED points joined
    click = None
    if confirmed or modes.get(line) == "marking" or not sg["available"]:
        click = streamlit_image_coordinates(Image.fromarray(img), key=f"f8i_bd_img_{line}_{state.revision}_{vs.zoom}_{vs.cx}_{vs.cy}{WL.suffix(sess)}", width="stretch", cursor="crosshair", png_compression_level=1)         # "stretch": the image always fits its column (a fixed 720 px image was clipped at 390 px); clicks are rescaled by click_to_full from the width the browser reports
    else:
        st.image(img, width="stretch")                 # a suggestion is only looked at here: clicks start only after Correct Line
    st.caption(LEGEND_TEXT)
    lc = ss.setdefault("f8i_an_last", {}); ck = (i, line)
    if click and lc.get(ck) != (click.get("unix_time"), state.revision):
        lc[ck] = (click.get("unix_time"), state.revision + 1); srcs.pop(line, None); _try(wf.seed_click, line, *view.click_to_full(click)); st.rerun()          # no banner: a message above the picture would shift it between the two clicks
    if confirmed:
        st.markdown(UT.line("ok", "You confirmed the suggested line." if srcs.get(line) == "suggestion" else "Line marked."))
        if st.button("Correct Line", key=f"f8i_bd_redo_{idx}", help="Remove these two points and mark the line again."):
            wf.seed_clear(line); srcs.pop(line, None); modes[line] = "marking"; st.rerun()
    elif sg["available"] and modes.get(line) != "marking":
        st.caption(SUGGEST_TEXT)
        a = st.columns(2)
        if a[0].button("Looks Correct", key=f"f8i_bd_ok_{idx}", type="primary", width="stretch"):
            _try(wf.seed_set, line, sg["points_distorted_full"]); srcs[line] = "suggestion"; st.rerun()
        if a[1].button("Correct Line", key=f"f8i_bd_fix_{idx}", width="stretch"):
            modes[line] = "marking"; st.rerun()
    else:
        st.caption("No suggestion was found for this line. Mark it yourself." if not sg["available"] else "Click the first point, then the second point.")
        if len(pts) == 1:
            st.caption("Now click the second point.")
        if pts and st.button("Clear points", key=f"f8i_bd_clr_{idx}"):
            wf.seed_clear(line); st.rerun()
    if ss.get("f8i_fs_msg"):
        ss.pop("f8i_fs_msg", None)
    nav = st.columns(2)
    if nav[0].button("Back", key="f8i_bd_back", disabled=idx == 0, width="stretch"):
        ss["f8i_bd_idx"] = idx - 1; st.rerun()
    last = idx == n - 1
    if nav[1].button("Finish" if last else "Next", key="f8i_bd_next", type="primary", disabled=not confirmed, width="stretch"):
        if last:
            with st.spinner("Refining the six boundaries in all five frame sets…"):
                _try(wf.apply_seeds)
        else:
            ss["f8i_bd_idx"] = idx + 1
        st.rerun()
    if wf.seeds_complete() and not last:
        if st.button("Refine All Boundaries Now", key="f8i_bd_now", width="stretch", help="All six boundaries are marked. You can refine them now without going through the others again."):
            with st.spinner("Refining the six boundaries in all five frame sets…"):
                _try(wf.apply_seeds)
            st.rerun()
    with st.expander("Technical Details"):
        st.markdown(f"Boundary `{line}` · frame set 1 · video frame {wf.full[(i, vk)]['index']} · points: " + (", ".join(f"({p[0]:.0f}, {p[1]:.0f})" for p in pts) or "none"))
        if sg["available"] or sg["reasons"]:
            st.markdown(f"Automatic suggestion (Stage F2, classical): **{sg['status']}**" + ("" if not sg["available"] else f" · visible length {sg['f2']['visible_length_px']:.0f} px · edge type {sg['f2']['confuser_class']} · support {sg['f2']['support_fraction']:.2f}"))
            for m in sg["reasons"][:6]:
                st.markdown(f"- {m}")
        st.caption("A top-ranked suggestion is not verified. The line you confirm is refined by the original edge search; a good fit does not prove it is the intended edge.")


# ------------------------------------------------------------------ 3 · review (STRICT, like the original experiment)
def _fix_row(wf, f, warn=False):
    with st.container(horizontal=True, vertical_alignment="center", gap="small", key=f"f8i_fixrow_{f['line']}"):
        st.markdown(UT.line("warn" if warn else "bad", UT.boundary_label(f["line"])))
        st.button("Correct Boundary" if not warn else "Review Boundary", key=f"f8i_fix_{f['line']}", on_click=_correct_boundary, args=(wf, f["line"]), help="Go straight to this boundary. The other boundaries are kept.")


def _group_lines(msgs):
    """Raw diagnostics grouped for the Technical Details section: identical (code, message) pairs are listed once with the frame sets they occurred in."""
    return [f"[{c}] {m} (frame sets {', '.join(map(str, sets))})" for (c, m), sets in msgs.items()]


_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _step3(svc, sess, wf):
    import quality_feedback as QF
    import wall_b_fallback as FB
    if sess.active_wall == "B":                          # Wall B only: a boundary the automatic edge search could not fit uses the visitor's own two-point line (Wall A is never touched)
        FB.apply_manual_fallback(wf)
    manual = FB.fallback_report(wf) if sess.active_wall == "B" else []
    rd = wf.readiness()
    usable = wf.usable_sets()
    items = QF.boundary_feedback(wf)
    state = QF.overall(items)
    fails = QF.boundary_failures(wf); warns = QF.boundary_warnings(wf)
    if fails:
        st.error("**Boundary refinement failed**\n\n" + QF.failure_headline(fails))
        for text in QF.failure_guidance(fails):
            st.markdown(f"- {text}")
        if len(usable) < wf.n_sets:
            st.markdown(f"- {QF.FRAME_SET_GUIDANCE}")
        st.caption(QF.RIG_HINT)
        for f in fails:
            _fix_row(wf, f)
    elif len(usable) == wf.n_sets:
        st.success(f"Boundary refinement completed. All {_WORDS.get(wf.n_sets, wf.n_sets)} frame sets were processed.")
    else:
        st.error(f"**Boundary refinement failed**\n\n{QF.FRAME_SET_GUIDANCE}")
    if manual:
        st.info("The automatic edge search could not fit " + ", ".join(QF.boundary_place(m["line"]) for m in manual) + " (for example because the edge is partly hidden), so the line you marked is used there. "
                "This is less precise, but it still gives an approximate estimate.")
    if warns:
        st.info("Some boundaries may benefit from visual review.")
        for f in warns:                                     # each affected boundary once (the frame sets and codes are in Technical Details)
            _fix_row(wf, f, warn=True)
    with st.expander("Technical Details"):
        grid = [dict({"Boundary": UT.boundary_row_label(n)}, **{f"Set {i + 1}": LINE_STATUS_TEXT[wf.line_report(i, n)["status"]] for i in range(wf.n_sets)}) for n in RF.REQUIRED_LINES]
        st.table(grid)
        for f in fails:
            st.markdown(f"**{f['place']}** · `{f['line']}` · failed in frame sets {', '.join(map(str, f['failed_sets']))} · codes: {', '.join(f'`{c}`' for c in f['codes'])}")
            for t in _group_lines(f["messages"]):
                st.caption(t)
        for f in warns:
            st.markdown(f"**{f['place']}** · `{f['line']}` · warnings in frame sets {', '.join(map(str, f['warned_sets']))} · codes: {', '.join(f'`{c}`' for c in f['codes'])}")
            for t in _group_lines(f["messages"]):
                st.caption(t)
        for m in manual:
            st.markdown(f"**Manual boundary used** · `{m['line']}` · frame sets {', '.join(map(str, m['sets']))} · code `{FB.FALLBACK_CODE}` (automatic refinement failed; the unrefined two-point line was used)")
        for m in rd.notes:
            st.caption(m)
        st.caption(QF.NO_ACCURACY_CLAIM)
    if rd.problems:
        for m in rd.problems:
            st.markdown(UT.line("bad", m))
    st.button("Continue", type="primary", key="f8i_to_measure", disabled=not rd.ready, on_click=_continue, args=(svc, sess))


def _continue(svc, sess):
    try:
        svc.go_measure(sess)
    except S.ServiceError as e:
        st.session_state["f8i_msg"] = str(e)


def _need_extracted(wf):
    return wf.frames_ready()



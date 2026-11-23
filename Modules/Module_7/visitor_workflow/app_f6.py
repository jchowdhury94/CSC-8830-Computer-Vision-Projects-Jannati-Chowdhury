"""Stage F6: "Measure a Room" -- LOCAL, ISOLATED, classical-CV visitor workflow (calibration -> videos -> boundary review -> results).

Run it with  python Modules/Module_7/run_stage_f6_visitor_app.py  (http://127.0.0.1:8630).  It is NOT linked from Pages/app_module7.py or any existing navigation.
All logic lives in f6_core.py (no Streamlit); this file only draws widgets and keeps presentation state.  Classical computer vision only: no ML / DL anywhere.
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

import f6_core as F                                                # noqa: E402
import f7_calibration as K7                                        # noqa: E402
import f7_pages                                                    # noqa: E402
import page_nav
import home_navigation                                             # noqa: E402  (shared Back-button style only)
from room_measurement import calibration as C                      # noqa: E402  (only to locate the optional local test fixture)
from room_measurement import visitor_annotations as V               # noqa: E402

f1 = F.f1core
ACCENT = "#C9363E"
FIX_VIDEO_DIR = MODULE / "data" / "room_measurement" / "calibration_videos"
FIX_VIDEOS = dict(placement_1_left="left_air_wall_left.MOV", placement_1_right="right_pro_wall_left.MOV", placement_2_left="left_air_wall_right.MOV", placement_2_right="right_pro_wall_right.MOV")
SLOTS = (("placement_1_left", "Placement 1 · LEFT camera video"), ("placement_1_right", "Placement 1 · RIGHT camera video"),
         ("placement_2_left", "Placement 2 · LEFT camera video"), ("placement_2_right", "Placement 2 · RIGHT camera video"))
BADGE = {"CONFIRMED": ("Confirmed", "#2E9E5B"), "PROPOSED": ("Suggested — not confirmed yet", "#E0A526"), "NEEDS_CONFIRMATION": ("Check the warning, then confirm", "#E0A526"),
         "BLOCKED": ("This line cannot be used", ACCENT), "NO_SUGGESTION": ("No suggestion", "#9AA1AC")}
CSS = f"""<style>
.f6-title {{ text-align: center; font-size: 2.6rem; font-weight: 800; margin: 0.2rem 0 0.1rem; line-height: 1.15; }}
.f6-progress {{ max-width: 800px; margin: 0.6rem auto 1.4rem; }}
.f6-progress-label {{ color: #B8BDC7; font-size: 1rem; font-weight: 600; margin-bottom: 0.4rem; text-align: center; }}
.f6-bar {{ height: 6px; border-radius: 999px; background: #2A2E37; overflow: hidden; }}
.f6-fill {{ height: 100%; background: {ACCENT}; border-radius: 999px; }}
.f6-badge {{ display: inline-block; padding: 0.15rem 0.7rem; border-radius: 999px; font-weight: 700; font-size: 0.9rem; color: #fff; }}
.f6-big {{ font-size: 2.4rem; font-weight: 800; line-height: 1.1; }}
[class*="st-key-navback_"] button {{ background: transparent !important; border: 1px solid rgba(255, 255, 255, 0.55) !important; }}
[class*="st-key-navback_"] button p {{ color: #FFFFFF !important; }}
button[kind="primary"] {{ background-color: {ACCENT}; border-color: {ACCENT}; color: #FFFFFF; }}
button[kind="primary"]:hover {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
button[kind="primary"]:disabled {{ background-color: rgba(201, 54, 62, 0.35); border-color: transparent; color: rgba(255, 255, 255, 0.6); }}
</style>"""


# ------------------------------------------------------------------ session plumbing
def _sess() -> F.F6Session:
    ss = st.session_state
    if "f6" not in ss: ss["f6"] = F.F6Session()
    return ss["f6"]


def _start_over():
    ss = st.session_state
    if "f6" in ss: ss["f6"].close()
    f7_pages.close_wizard()
    for k in [k for k in ss if str(k).startswith("f6")]: del ss[k]


def _go(step):
    st.session_state["f6_step"] = step


def _nav_back(step):
    if step > 0: home_navigation.back_button(f"f6_back_{step}", lambda: _go(step - 1))


def _problems(r):
    """Plain list of the blocking messages of a step result."""
    out = []
    for f in r.get("failures", []): out.append(f"{f.get('message', f.get('code'))}")
    return out


def render_header(step):
    st.markdown("<h1 class='f6-title'>Measure a Room</h1>", unsafe_allow_html=True)
    n = len(F.STEPS); frac = step / n
    if step == 0 and not st.session_state.get("f7_mode") == "import": frac += (st.session_state["f7"].step / len(K7.STEPS) / n) if "f7" in st.session_state else 0
    st.markdown(f"<div class='f6-progress'><div class='f6-progress-label'>Step {step + 1} of {n} — {F.STEPS[step] if step else 'Camera Calibration'}</div>"
                f"<div class='f6-bar'><div class='f6-fill' style='width:{max(frac, 0.03) * 100:.0f}%'></div></div></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ step 1: calibration
def _accept_calibration(w):
    """Explicit 'Accept calibration': Stage A validates the wizard's contract; only then does the workflow move on to the room-video upload."""
    ss = st.session_state
    try: contract = w.accept()
    except K7.WizardError as e: ss["f7_err"] = str(e); return
    r = _sess().import_calibration(contract); ss["f6_cal_result"] = r
    if r.get("can_proceed"): _sess().calibration_provenance = w.provenance(); ss.pop("f7_err", None); ss["f6_step"] = 1
    else: w.decision = None; w.contract = None; ss["f7_err"] = "The measurement backend rejected this calibration: " + "; ".join(_problems(r))


def page_calibration():
    ss = st.session_state
    if ss.get("f7_mode") == "import": return page_calibration_import()
    f7_pages.render(_accept_calibration, lambda: ss.__setitem__("f7_mode", "import"))
    if ss.get("f7_err"): st.error(ss["f7_err"])


def page_calibration_import():
    sess = _sess(); ss = st.session_state; st.markdown("## Import a Calibration File")
    st.button("Back", key="f7_back_to_wizard", on_click=lambda: ss.__setitem__("f7_mode", "wizard"))
    st.info("This step **imports** calibration that was computed earlier: each camera's intrinsics and lens distortion, and the physical distance between the two cameras. "
            "It does **not** calibrate cameras; use the guided calibration for that.")
    up = st.file_uploader("Calibration file (JSON)", type=["json"], key="f6_cal_file")
    c = st.columns(2)
    mode = c[0].radio("Both cameras were calibrated from", ["video", "photo"], index=None, horizontal=True, key="f6_cal_mode", help="Only needed when the file does not state it. It is never guessed.")
    base = c[1].number_input("Distance between the two cameras (mm)", min_value=0.0, value=0.0, step=1.0, key="f6_cal_base", help="Only needed when the file does not contain it. 0 = not entered.")
    if st.button("Use this calibration", type="primary", disabled=up is None, key="f6_cal_go"):
        cal, probs = F.parse_calibration_json(up.getvalue(), capture_mode=mode, baseline_mm=base or None)
        ss["f6_cal_result"] = dict(failures=[dict(message=p) for p in probs]) if probs else sess.import_calibration(cal)
    if f7_pages._fixtures_visible():                  # local test aid only: hidden from normal visitors (set F6_SHOW_FIXTURES=1)
        with st.expander("Local test fixture (original-room calibration)"):
            st.caption("Test data only: the saved Phase 1 Recipe A calibration, declared as video-mode with the measured 225 mm baseline (the same declaration the F1 / E2 tests use).")
            if st.button("Use original-room calibration", disabled=not Path(C.DEFAULT_CALIBRATION_PATH).exists(), key="f6_cal_fix"):
                cal, probs = F.parse_calibration_json(Path(C.DEFAULT_CALIBRATION_PATH).read_bytes(), capture_mode="video", baseline_mm=225.0)
                ss["f6_cal_result"] = sess.import_calibration(cal)
    r = ss.get("f6_cal_result")
    if r:
        if r.get("can_proceed"):
            cal = r["calibration"]; st.success(f"Calibration accepted · image {cal['image_size_wh'][0]}×{cal['image_size_wh'][1]} px · baseline {cal['baseline_mm']:g} mm")
            for w in r.get("warnings", []): st.warning(w["message"])
        else:
            st.error("This calibration cannot be used."); [st.markdown(f"⛔ {m}") for m in _problems(r)]
        with st.expander("Technical details"): st.json(r)
    st.button("Continue", type="primary", disabled=not (r and r.get("can_proceed")), on_click=_go, args=(1,), key="f6_cal_next")


# ------------------------------------------------------------------ step 2: videos + automatic frame selection
def _run_videos(videos):
    sess = _sess(); ss = st.session_state
    with st.spinner("Checking the videos…"): r = sess.submit_videos(videos)
    ss["f6_vid_result"] = r; ss.pop("f6_sel_result", None)
    if not r["can_proceed"]: return
    with st.spinner("Selecting frames and suggesting boundaries (about 20–30 s)…"): ss["f6_sel_result"] = sess.select_frames()
    ss["f6_set"] = 0; ss["f6_cur"] = 0


def page_upload():
    sess = _sess(); ss = st.session_state; _nav_back(1); st.markdown("### 2 · Upload your four videos")
    st.info("Each placement needs **two videos of the same scene, filmed at the same time**: one from the LEFT camera and one from the RIGHT camera. "
            "Keep both cameras fixed relative to each other. Videos must be 2160 × 3840 px portrait, as in the calibration.")
    cols = st.columns(2); ups = {}
    for i, (key, label) in enumerate(SLOTS):
        with cols[i // 2]: ups[key] = st.file_uploader(label, type=["mov", "mp4", "m4v"], key=f"f6_up_{key}")
    with st.expander("Advanced"):
        n = st.radio("Frame sets to use", [2, 3], horizontal=True, key="f6_nsets", help="More sets means more boundaries to confirm (6 per set).")
        sess.set_frame_set_count(n)
    if st.button("Check videos and select frames", type="primary", disabled=not all(ups.values()), key="f6_up_go"):
        _run_videos({k: u.getvalue() for k, u in ups.items()})
    fix_ok = all((FIX_VIDEO_DIR / f).exists() for f in FIX_VIDEOS.values())
    with st.expander("Local test fixture (original-room recordings)"):
        st.caption("Uses the original recordings in place, read-only. Nothing is copied.")
        if st.button("Use original-room recordings", disabled=not fix_ok, key="f6_up_fix"): _run_videos({k: str(FIX_VIDEO_DIR / f) for k, f in FIX_VIDEOS.items()})
    vr, sr = ss.get("f6_vid_result"), ss.get("f6_sel_result")
    if vr and not vr["can_proceed"]:
        st.error("These videos cannot be used."); [st.markdown(f"⛔ {m}") for m in _problems(vr)]
    if sr and not sr["can_proceed"]:
        st.error("Frames could not be selected."); [st.markdown(f"⛔ {m}") for m in _problems(sr)]
    if sr and sr["can_proceed"]:
        sel = sr.get("selection_check") or {}; val = sel.get("value") or {}
        st.success(f"{sess.job.n_sets} frame sets selected and boundaries suggested.")
        st.markdown(f"**Stereo matching:** {val.get('min_stereo_verified', '—')} verified matches at least per pair · {val.get('min_four_view_tracks', '—')} cross-view tracks at least (counted on reduced-resolution previews)")
        for w in sr.get("warnings", []): st.warning(w["message"])
        with st.expander("Frame details and diagnostics"):
            st.table([dict(set=i + 1, **{vk: sess.frame_record(i, vk)["frame_index"] for vk in ("AL", "PL", "AR", "PR")}) for i in range(sess.job.n_sets)])
            st.json(dict(selection_check=sel, frame_selection=sr.get("frame_selection"), video_diagnostics=(vr or {}).get("video_diagnostics"), detection=sess.detection_meta))
    st.button("Review boundaries", type="primary", disabled=not (sr and sr["can_proceed"] and sess.reviews), on_click=_go, args=(2,), key="f6_up_next")


# ------------------------------------------------------------------ step 3: boundary review
def _mviews(): return st.session_state.setdefault("f6_mviews", {})
def _mv(si, line): return _mviews().get((si, line), f1.ViewState())
def _bump(): st.session_state["f6_redraw"] = st.session_state.get("f6_redraw", 0) + 1


def cb_set(si): st.session_state["f6_set"] = si; st.session_state["f6_cur"] = 0
def cb_cur(k): st.session_state["f6_cur"] = k
def cb_pick(si, line): sess = _sess(); sess.select_candidate(si, line, st.session_state[f"f6_cand_{sess.session_id}_{si}_{line}"]); _bump()
def cb_confirm(si, line): ok = _sess().confirm(si, line); st.session_state["f6_msg"] = "" if ok else "This boundary could not be confirmed."; _bump()
def cb_reject(si, line): _sess().reject(si, line); _bump()
def cb_manual(si, line): _sess().use_manual(si, line); _sess().manual_focus(si, line); _bump()
def cb_auto(si, line): _sess().select_candidate(si, line, 1); _bump()


def cb_zoom(si, line, factor): full = tuple(_sess().job.cal.image_size); _mviews()[(si, line)] = f1.zoom_by(full, _mv(si, line), factor); _bump()
def cb_pan(si, line, dx, dy): full = tuple(_sess().job.cal.image_size); _mviews()[(si, line)] = f1.pan_by(full, _mv(si, line), dx, dy); _bump()
def cb_reset(si, line): _mviews()[(si, line)] = f1.ViewState(); _bump()
def cb_clear(si, line): _sess().manual_focus(si, line).clear(); _bump()
def cb_edit(si, line, i): _sess().manual_focus(si, line).begin_edit(i); _bump()


def cb_refine(si, line):
    try: _sess().manual_focus(si, line).submit(); st.session_state["f6_msg"] = ""
    except (ValueError, f1.FrameIntegrityError) as e: st.session_state["f6_msg"] = str(e)
    _bump()


def _explain(res):
    msgs = {i["code"]: i["message"] for i in res["issues"]}
    for code in res["failure_codes"] + res["warning_codes"]: st.markdown(f"{'⛔' if code in res['failure_codes'] else '⚠️'} {f1.explain(code, msgs.get(code, ''))}")
    for reason in ([] if res["identity_ok"] else res["identity_reasons"]): st.markdown(f"⛔ {reason}")


def _candidate_label(c):
    return f"Suggestion {c['rank']}" + (" (best guess)" if c["rank"] == 1 else "") + f" — {F.CONFUSER_TEXT.get(c['confuser_class'], c['confuser_class'])}, {c['visible_length_px']:.0f} px long"


def render_manual_image(sess, si, line):
    s = sess.manual_focus(si, line); stt = s.state(); rec = s.record(); vs = f1.clamp_state(rec.image_wh, _mv(si, line)); view = f1.make_view(rec.image_wh, vs)
    try: gray = s.frame()
    except f1.FrameIntegrityError as e: st.error(f"{e}. Nothing was substituted."); return
    a = f1.association(line); K, D = (sess.job.cal.K_left, sess.job.cal.D_left) if a["camera"] == "left" else (sess.job.cal.K_right, sess.job.cal.D_right); r = stt.result
    clicked = f1.line_polyline(r["initial_line"], K, D) if r and r.get("initial_line") else None; refined = f1.line_polyline(r["refined_line"], K, D) if r and r.get("refined_line") else None
    img = f1.render_view(gray, view, stt.points, clicked, refined, stt.edit_index); ss = st.session_state
    key = f"f6_mimg_{sess.session_id}_{si}_{line}_{stt.revision}_{ss.get('f6_redraw', 0)}"
    click = streamlit_image_coordinates(Image.fromarray(img), key=key, width=img.shape[1], cursor="crosshair", png_compression_level=1)
    lc = ss.setdefault("f6_last_click", {})
    if click and lc.get(key) != click.get("unix_time"):
        lc[key] = click.get("unix_time"); ss["f6_msg"] = s.click(*view.click_to_full(click)); _bump(); st.rerun()
    st.caption("Amber dashed = your clicked line · green = refined boundary · zoom ×%g" % vs.zoom)


def render_manual_panel(sess, si, line):
    s = sess.manual_focus(si, line); stt = s.state(); vs = f1.clamp_state(s.record().image_wh, _mv(si, line)); k = (sess.session_id, si, line)
    st.markdown("**Mark the boundary yourself.** Zoom in, then click two points on the visible edge, as far apart as it allows.")
    c = st.columns(6)
    c[0].button("−", key=f"f6_zo_{k}", on_click=cb_zoom, args=(si, line, 0.5), disabled=vs.zoom <= f1.ZOOM_MIN, width="stretch")
    c[1].button("+", key=f"f6_zi_{k}", on_click=cb_zoom, args=(si, line, 2.0), disabled=vs.zoom >= f1.ZOOM_MAX, width="stretch")
    for j, (lab, dx, dy) in enumerate((("←", -.25, 0), ("→", .25, 0), ("↑", 0, -.25), ("↓", 0, .25))):
        c[2 + j].button(lab, key=f"f6_pn{j}_{k}", on_click=cb_pan, args=(si, line, dx, dy), disabled=vs.zoom <= 1, width="stretch")
    st.button("Reset view", key=f"f6_rv_{k}", on_click=cb_reset, args=(si, line), width="stretch")
    for i in range(2):
        cc = st.columns([3, 1.4]); cc[0].markdown(f"Point {i + 1}: " + (f"({stt.points[i][0]:.0f}, {stt.points[i][1]:.0f})" if i < len(stt.points) else "not set") + ("  ← click to move" if stt.edit_index == i else ""))
        cc[1].button("Move", key=f"f6_mv{i}_{k}_{stt.revision}", on_click=cb_edit, args=(si, line, i), disabled=i >= len(stt.points), width="stretch")
    cc = st.columns(2)
    cc[0].button("Clear points", key=f"f6_cl_{k}", on_click=cb_clear, args=(si, line), disabled=not stt.points, width="stretch")
    cc[1].button("Refine boundary", key=f"f6_rf_{k}", type="primary", on_click=cb_refine, args=(si, line), disabled=not s.can_submit(), width="stretch")
    if st.session_state.get("f6_msg"): st.info(st.session_state["f6_msg"])
    st.button("Use the automatic suggestions again", key=f"f6_au_{k}", on_click=cb_auto, args=(si, line), disabled=not sess.candidates(si, line))


def page_review():
    sess = _sess(); ss = st.session_state; _nav_back(2); n_sets = sess.job.n_sets
    if not sess.reviews: st.warning("No boundaries to review yet. Go back and select frames."); return
    si = ss.setdefault("f6_set", 0); k = ss.setdefault("f6_cur", 0); line = F.BOUNDARY_ORDER[k]; rv = sess.review(si, line); prog = sess.progress()
    st.markdown("### 3 · Review the room boundaries")
    st.progress(prog["confirmed"] / max(prog["total"], 1), text=f"{prog['confirmed']} of {prog['total']} boundaries confirmed — the line is only a suggestion until you confirm it")
    if n_sets > 1:
        cs = st.columns(n_sets + 2)
        for i in range(n_sets): cs[i + 1].button(f"Frame set {i + 1}", key=f"f6_setbtn_{i}", type="primary" if i == si else "secondary", on_click=cb_set, args=(i,), width="stretch")
    cols = st.columns(6)
    for i, ln in enumerate(F.BOUNDARY_ORDER):
        cols[i].button(("✓ " if sess.is_confirmed(si, ln) else "") + f"{i + 1}", key=f"f6_bn_{si}_{i}", type="primary" if i == k else "secondary", on_click=cb_cur, args=(i,), width="stretch", help=F.label_of(ln))
    left, right = st.columns([1, 1.15], gap="large"); res = None
    try: res = sess.evaluate(si, line)
    except f1.FrameIntegrityError as e: st.error(f"{e}. Nothing was substituted.")
    with left:
        try:
            if rv.source == F.SOURCE_MANUAL: render_manual_image(sess, si, line)
            else: st.image(F.overlay_image(sess, si, line), width=560); st.caption("Amber dashed = suggested line · green = refined boundary")
        except f1.FrameIntegrityError as e: st.error(f"{e}. Nothing was substituted.")
    with right:
        rec = sess.frame_record(si, rv.view_key); det = sess.detection(si, line); status = sess.status(si, line); text, color = BADGE[status]
        st.markdown(f"## {F.label_of(line)}"); st.caption(f"{F.camera_text(line)} · frame set {si + 1} · video frame {rec['frame_index']}")
        st.markdown(F.INSTRUCTIONS[f1.association(line)["boundary_type"]])
        if rv.source == F.SOURCE_CANDIDATE:
            if det is not None: st.markdown(f"_{F.DETECTION_TEXT[det.status]}_")
            cands = sess.candidates(si, line)
            if cands:
                st.radio("Which line is the wall boundary?", [c["rank"] for c in cands], index=None if rv.rank is None else rv.rank - 1, key=f"f6_cand_{sess.session_id}_{si}_{line}",
                         format_func=lambda r: _candidate_label(cands[r - 1]), on_change=cb_pick, args=(si, line))
        else: render_manual_panel(sess, si, line)
        st.markdown(f"<span class='f6-badge' style='background:{color}'>{text}</span>", unsafe_allow_html=True)
        if res: _explain(res)
        if res and res["decision"] == "ALLOWED" and status != "CONFIRMED": st.success("No problems found. Still check the line by eye before confirming.")
        if res and res["decision"].startswith("BLOCKED") and res["decision"] != "BLOCKED_UNCONFIRMED_WARNING": st.error("This line cannot be used. Choose another suggestion or mark it yourself.")
        can = bool(res) and res["decision"] in ("ALLOWED", "BLOCKED_UNCONFIRMED_WARNING") and status != "CONFIRMED"
        b = st.columns(2)
        b[0].button("Confirm this boundary", key=f"f6_ok_{sess.session_id}_{si}_{line}_{rv.revision}", type="primary", on_click=cb_confirm, args=(si, line), disabled=not can, width="stretch")
        if rv.source == F.SOURCE_CANDIDATE: b[1].button("Correct manually", key=f"f6_man_{sess.session_id}_{si}_{line}", on_click=cb_manual, args=(si, line), width="stretch")
        else: b[1].button("Discard and start over", key=f"f6_rj_{sess.session_id}_{si}_{line}", on_click=cb_reject, args=(si, line), width="stretch")
        if ss.get("f6_msg") and rv.source == F.SOURCE_CANDIDATE: st.info(ss["f6_msg"])
        n = st.columns(2)
        n[0].button("Previous", key=f"f6_prev_{si}_{k}", on_click=cb_cur, args=(max(k - 1, 0),), disabled=k == 0, width="stretch")
        n[1].button("Next boundary", key=f"f6_next_{si}_{k}", on_click=cb_cur, args=(min(k + 1, 5),), disabled=k == 5, width="stretch")
        with st.expander("Technical details"):
            st.json(dict(boundary_id=line, set_index=si, camera=f1.association(line)["camera"], placement=f1.association(line)["placement"], frame_index=rec["frame_index"], frame_fingerprint=rec["frame_fingerprint"],
                         source=rv.source, selected_rank=rv.rank, f2_status=None if det is None else det.status, f2_reasons=[] if det is None else det.reasons, stage_d=res, acceptance=rv.acceptance))
            if det is not None and det.candidates:
                st.dataframe([dict(rank=c["rank"], rank_score=round(c["rank_score"], 3), confuser=c["confuser_class"], length_px=round(c["visible_length_px"]), support=round(c["support_fraction"], 2), notes="; ".join(c["notes"])) for c in det.candidates], hide_index=True)
    done = prog["all_confirmed"]
    st.button("View results", type="primary", disabled=not done, on_click=_go, args=(3,), key="f6_rev_next", help=None if done else "Confirm all boundaries in every frame set first.")


# ------------------------------------------------------------------ step 4: results
def page_results():
    sess = _sess(); ss = st.session_state; _nav_back(3); st.markdown("### 4 · Room size")
    m = sess.measurement; dims = sess.dimensions()
    if dims is None:
        st.markdown(f"All **{sess.progress()['confirmed']}** boundaries are confirmed. The room is measured from your confirmed boundaries with the same stereo geometry as the original experiment. "
                    "This takes about a minute and up to ~2.5 GB of memory.")
        if st.button("Estimate room width and height", type="primary", disabled=not sess.progress()["all_confirmed"], key="f6_measure"):
            try:
                with st.spinner("Estimating pose, reconstructing the floor and measuring…"): sess.measure()
            except F.F6Error as e: ss["f6_err"] = str(e)
            st.rerun()
        if ss.get("f6_err"): st.error(ss["f6_err"])
        pend = sess.pending_set_confirmations()
        for q in pend:
            st.warning(q["message"])
            st.button("I have checked this — continue", key=f"f6_setconf_{q['line_name']}_{q['set_index']}", on_click=lambda q=q: sess.confirm_set_issue(q))
        if m and m.get("step") == "measure" and not m.get("can_proceed"):
            st.error("The room could not be measured. No dimensions are shown, because none can be trusted."); [st.markdown(f"⛔ {x}") for x in _problems(m)]
        if m and m.get("step") != "measure" and _problems(m): [st.markdown(f"⛔ {x}") for x in _problems(m)]
        if m:
            with st.expander("Technical details"): st.json(m)
        return
    unit = st.radio("Show results in", F.DISPLAY_UNITS, horizontal=True, key="f6_unit")
    c = st.columns(2)
    c[0].markdown(f"**Estimated width**<div class='f6-big'>{F.format_length(dims['width_mm'], unit)}</div>", unsafe_allow_html=True)
    c[1].markdown(f"**Estimated height**<div class='f6-big'>{F.format_length(dims['height_mm'], unit)}</div>", unsafe_allow_html=True)
    for w in m.get("warnings", []): st.warning(w["message"])
    st.caption(("Internal consistency checks passed" if m.get("overall_status") == "PASS" else "Internal consistency checks finished with the warnings above")
               + "; they do not prove the dimensions are accurate. Compare with a tape measure below if you can.")
    st.markdown("#### Compare with the actual room (optional)")
    f = st.columns([1, 1, 1])
    wv = f[0].text_input("Actual width", key="f6_aw", placeholder="optional"); hv = f[1].text_input("Actual height", key="f6_ah", placeholder="optional")
    au = f[2].selectbox("Unit", list(F.UNIT_TO_MM), index=list(F.UNIT_TO_MM).index("mm"), key="f6_au")
    try: sess.set_actual_dimensions(wv, hv, au); err = None
    except ValueError as e: err = str(e)
    if err: st.error(f"Actual dimensions: {err}")
    rep = sess.validation() if err is None else None
    if rep and sess.actual.any:
        rows = []
        for k in ("width", "height"):
            r = rep[k]; cmpd = r["comparison"]
            rows.append({"": k.title(), "Estimated": F.format_length(r["estimated_mm"], unit), "Actual": "—" if cmpd is None else F.format_length(r["actual_mm"], unit),
                         "Absolute error": "—" if cmpd is None else F.format_length(cmpd["absolute_error_mm"], unit), "Percentage error": "—" if cmpd is None else f"{cmpd['absolute_percent_error']:.2f} %"})
        st.table(rows)
        with st.expander("Signed difference (estimate − actual)"):
            for k in ("width", "height"):
                cmpd = rep[k]["comparison"]
                if cmpd: st.markdown(f"{k.title()}: {cmpd['signed_error_mm']:+.1f} mm ({cmpd['signed_percent_error']:+.2f} %)")
    elif not sess.actual.any: st.caption("No actual dimensions entered — no error is shown.")
    with st.expander("Technical details"):
        st.json({k: m.get(k) for k in ("overall_status", "dimensions", "pose_diagnostics", "pose_stability", "quality_checks", "registration", "annotation_review", "timings_s", "peak_rss_mb")})
    st.download_button("Download results (JSON)", data=json.dumps(sess.summary_for_export(), indent=1), file_name=f"room_measurement_{sess.session_id}.json", mime="application/json", key="f6_dl")
    st.button("Start over", on_click=_start_over, key="f6_restart")


def render_workflow():
    """The workflow itself, with NO page-level navigation: a host page (Pages/app_module7.py, which draws nav_row) calls this directly."""
    st.html(home_navigation._CSS); st.html(CSS)
    ss = st.session_state; step = ss.setdefault("f6_step", 0)
    if step == 0: st.html('<style>.stMainBlockContainer, .block-container { max-width: 820px !important; margin: 0 auto; }</style>')   # centered, readable column for the calibration screens
    render_header(step)
    [page_calibration, page_upload, page_review, page_results][step]()
    if step > 0 or "f6" in ss: st.divider(); st.button("Start over (discard this session's data)", key="f6_reset_bottom", on_click=_start_over)


def main():
    """Standalone entry: Module 7's Home / Back row on top, then the workflow."""
    st.set_page_config(page_title="Measure a Room", layout="wide")
    page_nav.standalone_nav()
    render_workflow()


main()

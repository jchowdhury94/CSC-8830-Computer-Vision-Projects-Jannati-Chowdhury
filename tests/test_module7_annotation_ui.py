"""Phase 2B Stage F1: local annotation prototype (Modules/Module_7/annotation_ui).  The pure core is tested without Streamlit; the page is exercised with
Streamlit's AppTest, with the click component replaced by a scripted stand-in (a real browser was not available -- see the Stage F1 report).
Backend-dependent tests use the original-resolution local frames and are skipped when they have not been prepared."""
import ast
import copy
import dataclasses
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7"
UI = MODULE / "annotation_ui"
for p in (str(MODULE), str(UI), str(ROOT)):
    if p not in sys.path: sys.path.insert(0, p)

import core  # noqa: E402
import fixture  # noqa: E402
from room_measurement import visitor_annotations as V, visitor_measurement as VM  # noqa: E402

HAVE_FRAMES = fixture.available()
needs_frames = unittest.skipUnless(HAVE_FRAMES, "local original-resolution frames not prepared (annotation_ui/prepare_local_frames.py)")
WH = (2160, 3840)
# Independent clicks (read visually from the frames in Stage E3; not derived from any Phase 1 line)
CLICKS = {"AL_corner": [[504, 800], [507, 2250]], "PL_corner": [[671, 700], [697, 2400]], "PL_ceilF": [[800, 436], [1500, 238]], "AR_corner": [[1736, 650], [1713, 2200]], "PR_corner": [[1768, 600], [1802, 2200]], "PR_ceilF": [[300, 207], [1600, 236]]}
OFF_CLICKS = {"PR_corner": [[1740, 600], [1774, 2200]]}      # ~28 px beside the corner: Stage D refuses it (REFINEMENT_AT_WINDOW_LIMIT)

_STATE = {}


def shared():
    if not _STATE: _STATE.update(frames=fixture.load_frames(), cal=fixture.load_calibration())
    return _STATE["frames"], _STATE["cal"]


def new_session(**kw):
    f, c = shared(); return core.AnnotationSession(f, c, **kw)


def place(sess, line, pts=None):
    sess.goto(sess.lines.index(line))
    for p in (pts or CLICKS[line]): sess.click(*p)


class Viewport(unittest.TestCase):
    STATES = [core.ViewState(1.0), core.ViewState(2.0, 700, 900), core.ViewState(8.0, 1800, 3000), core.ViewState(32.0, 5, 5), core.ViewState(4.0, 2159, 3839), core.ViewState(1.5, 1080, 1920)]

    def test_zoom_one_shows_the_whole_frame_with_the_right_aspect(self):
        v = core.make_view(WH); self.assertEqual(v.crop, (0, 0, 2160, 3840)); self.assertLessEqual(v.shown_wh[0], 720); self.assertLessEqual(v.shown_wh[1], 900)
        self.assertAlmostEqual(v.shown_wh[0] / v.shown_wh[1], WH[0] / WH[1], delta=0.003); self.assertAlmostEqual(v.sx, v.sy, delta=0.002); self.assertEqual(v.canvas_wh, v.shown_wh)

    def test_view_full_roundtrip_is_exact_for_every_zoom_and_pan(self):
        rs = np.random.RandomState(0)
        for st in self.STATES:
            v = core.make_view(WH, st)
            for _ in range(20):
                x, y = rs.uniform(v.crop[0], v.crop[2] - 1), rs.uniform(v.crop[1], v.crop[3] - 1); u, w = v.to_view(x, y); bx, by = v.to_full(u, w)
                self.assertAlmostEqual(float(bx), x, places=9); self.assertAlmostEqual(float(by), y, places=9)

    def test_browser_scaled_component_click_is_rescaled(self):
        v = core.make_view(WH, core.ViewState(4.0, 1000, 2000)); w, h = v.shown_wh
        for scale in (1.0, 0.8, 0.5):
            cw, ch = int(round(w * scale)), int(round(h * scale)); x, y = cw // 3, ch // 2; got = v.click_to_full(dict(x=x, y=y, width=cw, height=ch)); ref = v.to_full((x + 0.5) * w / cw, (y + 0.5) * h / ch)
            self.assertAlmostEqual(got[0], float(ref[0]), places=9); self.assertAlmostEqual(got[1], float(ref[1]), places=9)
        mid = core.make_view(WH).click_to_full(dict(x=253, y=450, width=506, height=900)); self.assertAlmostEqual(mid[0], 1080, delta=6); self.assertAlmostEqual(mid[1], 1920, delta=6)

    def test_zoom_mapping_scale_and_centre(self):
        st = core.ViewState(1.0)
        for f in (2.0, 2.0, 2.0):
            st = core.zoom_by(WH, st, f)
        self.assertEqual(st.zoom, 8.0); v = core.make_view(WH, st); base = v.canvas_wh[0] / WH[0]; self.assertAlmostEqual(v.sx, base * 8.0, delta=0.02 * base * 8)
        c = v.click_to_full(dict(x=v.shown_wh[0] // 2, y=v.shown_wh[1] // 2, width=v.shown_wh[0], height=v.shown_wh[1])); self.assertAlmostEqual(c[0], st.cx, delta=1.5); self.assertAlmostEqual(c[1], st.cy, delta=1.5)
        self.assertEqual(core.zoom_by(WH, core.ViewState(core.ZOOM_MAX), 2.0).zoom, core.ZOOM_MAX); self.assertEqual(core.zoom_by(WH, core.ViewState(1.0), 0.5).zoom, 1.0)

    def test_zoom_about_an_anchor_keeps_the_anchor_on_screen(self):
        st = core.ViewState(2.0, 1080, 1920); anchor = (900.0, 1500.0); v0 = core.make_view(WH, st); u0 = v0.to_view(*anchor); st2 = core.zoom_by(WH, st, 2.0, anchor); v1 = core.make_view(WH, st2); u1 = v1.to_view(*anchor)
        self.assertLess(abs(float(u0[0]) - float(u1[0])), 3.0); self.assertLess(abs(float(u0[1]) - float(u1[1])), 3.0)

    def test_pan_moves_the_view_not_the_image_or_zoom(self):
        st = core.ViewState(4.0, 1080, 1920); p = core.pan_by(WH, st, 0.25, -0.25); self.assertEqual(p.zoom, 4.0)
        self.assertAlmostEqual(p.cx - st.cx, 0.25 * WH[0] / 4.0, places=6); self.assertAlmostEqual(p.cy - st.cy, -0.25 * WH[1] / 4.0, places=6)
        v0, v1 = core.make_view(WH, st), core.make_view(WH, p); click = dict(x=100, y=200, width=v0.shown_wh[0], height=v0.shown_wh[1]); a, b = v0.click_to_full(click), v1.click_to_full(click)
        self.assertAlmostEqual(b[0] - a[0], p.cx - st.cx, delta=1.0); self.assertAlmostEqual(b[1] - a[1], p.cy - st.cy, delta=1.0)
        edge = core.pan_by(WH, st, 50, 50); self.assertLessEqual(edge.cx, WH[0] - WH[0] / 8 + 1e-6); self.assertLessEqual(edge.cy, WH[1] - WH[1] / 8 + 1e-6)
        self.assertEqual(core.pan_by(WH, core.ViewState(1.0), 0.5, 0.5), core.clamp_state(WH, core.ViewState(1.0)))      # nothing to pan at zoom 1

    def test_a_point_stays_at_the_same_full_resolution_position_under_zoom_and_pan(self):
        pt = (1100.0, 2000.0)
        for st in (core.ViewState(4.0, 1100, 2000), core.pan_by(WH, core.ViewState(4.0, 1100, 2000), 0.1, 0.05), core.ViewState(16.0, 1100, 2000)):
            v = core.make_view(WH, st)
            if v.contains(*pt): u, w = v.to_view(*pt); bx, by = v.to_full(u, w); self.assertAlmostEqual(float(bx), pt[0], places=9)
        self.assertFalse(core.make_view(WH, core.ViewState(16.0, 200, 200)).contains(*pt))

    def test_full_resolution_recovery_from_a_rendered_marker_position(self):
        img = np.full((WH[1], WH[0]), 60, np.uint8); X, Y = 1234, 2345; img[Y - 1:Y + 2, X - 1:X + 2] = 255
        for z in (4.0, 8.0, 16.0):
            v = core.make_view(WH, core.ViewState(z, X + 6, Y - 4)); out = core.render_view(img, v); g = out[..., 0].astype(float); ys, xs = np.nonzero(g > 150)
            self.assertGreater(len(xs), 4); cx, cy = float((xs * g[ys, xs]).sum() / g[ys, xs].sum()), float((ys * g[ys, xs]).sum() / g[ys, xs].sum())
            fx, fy = v.to_full(cx + 0.5, cy + 0.5); self.assertAlmostEqual(float(fx), X, delta=0.3); self.assertAlmostEqual(float(fy), Y, delta=0.3)

    def test_rendering_never_modifies_or_resizes_the_source(self):
        img = np.random.RandomState(1).randint(0, 255, (3840, 2160), np.uint8); before = img.copy(); ro = img.flags.writeable; img.setflags(write=False)
        core.render_view(img, core.make_view(WH, core.ViewState(4.0, 900, 900)), points=[[900, 900], [1000, 1000]]); core.render_view(img, core.make_view(WH)); img.setflags(write=ro)
        self.assertTrue(np.array_equal(img, before)); self.assertEqual(img.shape, (3840, 2160))

    def test_markers_are_drawn_at_the_point_positions_after_zoom_and_pan(self):
        img = np.full((WH[1], WH[0]), 90, np.uint8); pts = [[1000.0, 1500.0], [1060.0, 1700.0]]
        for st in (core.ViewState(1.0), core.ViewState(4.0, 1030, 1600), core.pan_by(WH, core.ViewState(4.0, 1030, 1600), 0.05, 0.0)):
            v = core.make_view(WH, st); out = core.render_view(img, v, points=pts)
            for p in pts:
                if v.contains(*p):
                    u, w = (float(c) for c in v.to_view(*p)); ring = out[int(round(w)), int(round(u + 9))]; self.assertGreater(int(ring.min()), 180, (st, p))     # the white marker ring, 9 px right of the centre
        none = core.render_view(img, core.make_view(WH, core.ViewState(32.0, 200, 200)), points=pts); self.assertLess(int(none.max()), 120)                 # outside the crop: nothing drawn

    def test_editing_point_is_highlighted_differently(self):
        img = np.full((WH[1], WH[0]), 90, np.uint8); v = core.make_view(WH, core.ViewState(4.0, 1000, 1500)); a = core.render_view(img, v, points=[[1000.0, 1500.0]]); b = core.render_view(img, v, points=[[1000.0, 1500.0]], edit_index=0)
        self.assertFalse(np.array_equal(a, b))


class BoundariesAndAssociation(unittest.TestCase):
    def test_boundary_order_is_the_verified_phase1_mapping_with_top_variant_0(self):
        self.assertEqual([b[2] for b in core.BOUNDARY_ORDER], ["AL_corner", "PL_corner", "PL_ceilF", "AR_corner", "PR_corner", "PR_ceilF"]); self.assertEqual(VM.TOP_VARIANT, 0)
        self.assertTrue(all("ceilS" not in b[2] for b in core.BOUNDARY_ORDER))

    def test_association_comes_from_the_backend_tables(self):
        exp = {"AL_corner": ("placement_1", "left"), "PL_corner": ("placement_1", "right"), "PL_ceilF": ("placement_1", "right"), "AR_corner": ("placement_2", "left"), "PR_corner": ("placement_2", "right"), "PR_ceilF": ("placement_2", "right")}
        for ln, (pl, cam) in exp.items(): a = core.association(ln); self.assertEqual((a["placement"], a["camera"]), (pl, cam)); self.assertEqual(a["view_key"], ln[:2]); self.assertIn(a["boundary_type"], V.CAMERA_BOUNDARIES[cam])
        for bad in ("AL_ceilF", "PL_ceilS", "AR_ceilF", "XX_corner"):
            with self.assertRaises(ValueError): core.association(bad)

    @needs_frames
    def test_each_boundary_is_shown_on_its_own_camera_frame(self):
        s = new_session()
        for si in s.sets:
            s.select_set(si)
            for ln in s.lines:
                r = s.record(ln); self.assertEqual((r.set_index, r.view_key), (si, ln[:2])); a = core.association(ln); self.assertEqual((r.placement, r.camera), (a["placement"], a["camera"])); self.assertEqual(r.image_wh, WH)

    @needs_frames
    def test_mismatched_frame_records_are_refused(self):
        f, c = shared(); bad = dict(f); bad[(0, "PL")], bad[(0, "AL")] = f[(0, "AL")], f[(0, "PL")]
        with self.assertRaises(ValueError): core.AnnotationSession(bad, c)
        bad = dict(f); del bad[(1, "PR")]
        with self.assertRaises(ValueError): core.AnnotationSession(bad, c)

    @needs_frames
    def test_frames_are_the_exact_fingerprinted_frames_and_a_changed_file_is_refused(self):
        f, c = shared(); s = core.AnnotationSession(f, c); g = s.frame("AL_corner"); self.assertEqual(V.frame_fingerprint(g), f[(0, "AL")].frame_fingerprint); self.assertEqual(g.shape, (3840, 2160))
        fake = dict(f); fake[(0, "AL")] = dataclasses.replace(f[(0, "AL")], frame_fingerprint="0" * 64); s2 = core.AnnotationSession(fake, c)
        with self.assertRaises(core.FrameIntegrityError): s2.frame("AL_corner")


@needs_frames
class PointCollection(unittest.TestCase):
    def test_exactly_two_points_then_correction_and_clear(self):
        s = new_session(); self.assertFalse(s.can_submit()); self.assertIn("placed", s.click(500, 800)); self.assertFalse(s.can_submit()); self.assertIn("placed", s.click(510, 2200)); self.assertTrue(s.can_submit())
        self.assertIn("choose which point", s.click(600, 900)); self.assertEqual(len(s.state().points), 2); s.begin_edit(1); self.assertIn("point 2 moved", s.click(512, 2210)); self.assertEqual(s.state().points[1], [512.0, 2210.0]); self.assertEqual(s.state().points[0], [500.0, 800.0])
        self.assertIsNone(s.state().edit_index); s.clear(); self.assertEqual(s.state().points, []); self.assertFalse(s.can_submit())
        with self.assertRaises(ValueError): s.begin_edit(0)

    def test_submission_needs_two_points_and_clicks_outside_are_ignored(self):
        s = new_session()
        with self.assertRaises(ValueError): s.submit()
        s.click(500, 800)
        with self.assertRaises(ValueError): s.submit()
        self.assertIn("outside", s.click(-50, 10)); self.assertIn("outside", s.click(3000, 10)); self.assertIn("outside", s.click(float("nan"), 10)); self.assertEqual(len(s.state().points), 1)

    def test_point_change_bumps_the_revision(self):
        s = new_session(); r0 = s.state().revision; s.click(500, 800); s.click(510, 2200); self.assertEqual(s.state().revision, r0 + 2); s.begin_edit(0); s.click(505, 810); self.assertEqual(s.state().revision, r0 + 3)


@needs_frames
class RefinementAndConfirmation(unittest.TestCase):
    def test_payload_is_the_stage_d_annotation_contract(self):
        s = new_session(); place(s, "PL_corner"); r = s.submit(); raw = s.state().raw; rec = s.record("PL_corner")
        self.assertEqual((raw["placement"], raw["camera"], raw["frame_index"], raw["boundary_type"], raw["coordinate_space"]), ("placement_1", "right", rec.frame_index, "corner", "distorted_full")); self.assertEqual(raw["image"], dict(width=2160, height=3840))
        self.assertEqual(raw["frame_fingerprint"], rec.frame_fingerprint); self.assertEqual(raw["line_name"], "PL_corner"); self.assertIn(s.job_id, raw["annotation_id"]); self.assertTrue(raw["annotation_id"].endswith(f":r{s.state().revision}"))
        parsed, issues = V.parse_annotation(raw); self.assertEqual(issues, []); self.assertEqual(VM.annotation_contract_problems(raw), []); self.assertEqual(raw["points"], [[671.0, 700.0], [697.0, 2400.0]]); self.assertEqual(r["annotation_digest"], VM.annotation_digest(raw))

    def test_pass_proceeds_without_confirmation(self):
        s = new_session(); place(s, "AL_corner"); r = s.submit()
        self.assertEqual((r["status"], r["decision"]), ("PASS", "ALLOWED")); self.assertTrue(s.state().accepted); self.assertFalse(s.can_confirm()); self.assertEqual(s.state_summary(), "READY"); self.assertTrue(s.accept()); self.assertFalse(s.confirm())
        self.assertEqual(s.completed(), ["AL_corner"]); self.assertEqual(s.export()["confirmations"], [])

    def test_refined_line_overlay_follows_the_clicked_edge(self):
        s = new_session(); place(s, "PL_corner"); r = s.submit(); cal = s.cal; poly = core.line_polyline(r["refined_line"], cal.K_right, cal.D_right); clicked = core.line_polyline(r["initial_line"], cal.K_right, cal.D_right)
        pts = np.array(s.state().points); d = np.abs(np.interp(pts[:, 1], poly[:, 1], poly[:, 0]) - pts[:, 0]); self.assertLess(d.max(), 20); self.assertGreater(len(poly), 10); self.assertFalse(np.array_equal(poly, clicked))
        view = core.make_view(WH); base = core.render_view(s.frame(), view, s.state().points); over = core.render_view(s.frame(), view, s.state().points, clicked, poly)
        diff = np.abs(over.astype(int) - base.astype(int)).sum(2) > 60; self.assertGreater(int(diff.sum()), 200); g = (over[..., 1] > 200) & (over[..., 0] < 90) & (over[..., 2] < 150); self.assertGreater(int(g.sum()), 200)     # green refined line present
        us, vs_ = view.to_view(poly[:, 0], poly[:, 1]); ys, xs = np.nonzero(g); mid = len(poly) // 2; near = np.abs(ys - vs_[mid]) < 6; self.assertTrue(near.any()); self.assertLess(np.abs(xs[near] - us[mid]).min(), 4)   # drawn where the polyline is

    def test_warning_requires_explicit_confirmation_bound_to_the_exact_annotation(self):
        s = new_session(); place(s, "PR_corner"); r = s.submit(); st = s.state()
        self.assertEqual((r["status"], r["decision"], r["warning_codes"]), ("WARNING", "BLOCKED_UNCONFIRMED_WARNING", ["COMPETING_EDGE_NEARBY"])); self.assertFalse(st.accepted); self.assertFalse(s.accept()); self.assertTrue(s.can_confirm()); self.assertEqual(s.state_summary(), "NEEDS_CONFIRMATION")
        self.assertEqual(s.export()["annotations"], []); req = r["confirmation_request"]; self.assertEqual(req["annotation_digest"], r["annotation_digest"]); self.assertEqual(req["refined_line_digest"], r["refined_line_digest"]); self.assertEqual(req["frame_fingerprint"], s.record("PR_corner").frame_fingerprint)
        self.assertTrue(s.confirm()); c = st.confirmation; self.assertIsInstance(c, VM.Confirmation); self.assertTrue(c.confirmed); self.assertEqual((c.annotation_digest, c.refined_line_digest, c.frame_fingerprint, c.confirmed_codes), (r["annotation_digest"], r["refined_line_digest"], req["frame_fingerprint"], ("COMPETING_EDGE_NEARBY",)))
        self.assertEqual((st.result["decision"], s.state_summary(), st.accepted), ("ALLOWED_CONFIRMED", "CONFIRMED", True)); e = s.export(); self.assertEqual(len(e["confirmations"]), 1); self.assertEqual(VM.parse_confirmation(e["confirmations"][0])[1], [])

    def test_fail_blocks_and_can_never_be_confirmed(self):
        s = new_session(); place(s, "PR_corner", OFF_CLICKS["PR_corner"]); r = s.submit()
        self.assertEqual((r["status"], r["decision"]), ("FAIL", "BLOCKED_FAIL")); self.assertIn("REFINEMENT_AT_WINDOW_LIMIT", r["failure_codes"]); self.assertFalse(s.can_confirm()); self.assertFalse(s.confirm()); self.assertFalse(s.accept()); self.assertEqual(s.state_summary(), "BLOCKED")
        st = s.state(); st.confirmation = VM.Confirmation("annotation", 0, "PR_corner", r["annotation_id"], s.record("PR_corner").frame_index, s.record("PR_corner").frame_fingerprint, r["annotation_digest"], r["refined_line_digest"] or "", (), True, "forged")
        s._regate(st); self.assertEqual(st.result["decision"], "BLOCKED_FAIL"); self.assertFalse(st.accepted); self.assertEqual(s.export()["annotations"], [])

    def test_changing_a_point_invalidates_the_confirmation(self):
        s = new_session(); place(s, "PR_corner"); s.submit(); s.confirm(); st = s.state(); self.assertTrue(st.accepted); rev = st.revision
        s.begin_edit(1); s.click(1803, 2205)
        self.assertEqual((st.revision, st.accepted, st.confirmation, st.result, st.raw), (rev + 1, False, None, None, None)); self.assertEqual(s.state_summary(), "NOT_REFINED"); self.assertEqual(s.export()["annotations"], [])
        r = s.submit(); self.assertEqual(r["decision"], "BLOCKED_UNCONFIRMED_WARNING"); self.assertFalse(r["confirmation_valid"] or False)

    def test_a_changed_refined_line_or_annotation_digest_invalidates_a_stored_confirmation(self):
        s = new_session(); place(s, "PR_corner"); s.submit(); s.confirm(); st = s.state(); good = st.confirmation
        for bad in (dataclasses.replace(good, refined_line_digest="0" * 64), dataclasses.replace(good, annotation_digest="0" * 64), dataclasses.replace(good, frame_fingerprint="f" * 64), dataclasses.replace(good, confirmed_codes=("WEAK_EDGE",)), dataclasses.replace(good, confirmed=False)):
            st.confirmation = bad; s._regate(st); self.assertEqual(st.result["decision"], "BLOCKED_UNCONFIRMED_WARNING", bad); self.assertFalse(st.accepted)
        st.confirmation = good; s._regate(st); self.assertTrue(st.accepted)

    def test_reject_keeps_points_but_drops_refinement_and_confirmation(self):
        s = new_session(); place(s, "PR_corner"); s.submit(); s.confirm(); s.reject(); st = s.state()
        self.assertEqual(len(st.points), 2); self.assertIsNone(st.result); self.assertIsNone(st.confirmation); self.assertFalse(st.accepted); self.assertTrue(s.can_submit())

    def test_retry_with_unchanged_points_is_deterministic(self):
        s = new_session(); place(s, "AL_corner"); a = s.submit(); b = s.submit(); self.assertEqual((a["annotation_digest"], a["refined_line_digest"]), (b["annotation_digest"], b["refined_line_digest"]))

    def test_codes_are_explained_in_plain_language_and_kept_in_diagnostics(self):
        for code in ("COMPETING_EDGE_NEARBY", "WEAK_EDGE", "REFINEMENT_FAILED"): self.assertGreater(len(core.explain(code)), 30); self.assertNotIn(code, core.explain(code))
        self.assertEqual(core.explain("SOMETHING_NEW", "backend text"), "backend text"); self.assertEqual(core.explain("SOMETHING_NEW"), "SOMETHING_NEW")
        s = new_session(); place(s, "PR_corner"); r = s.submit(); self.assertEqual(r["warning_codes"], ["COMPETING_EDGE_NEARBY"]); self.assertTrue(r["issues"]); self.assertEqual(r["checks_not_evaluated"], ["stereo_consistency"])      # NOT_EVALUATED is kept, not turned into PASS


@needs_frames
class NavigationAndIsolation(unittest.TestCase):
    def test_navigation_progress_and_reachability(self):
        s = new_session(); self.assertEqual(s.progress(), (1, 6)); self.assertTrue(s.reachable(0)); self.assertFalse(s.reachable(1)); s.prev(); self.assertEqual(s.progress(), (1, 6)); s.next(); self.assertEqual(s.progress(), (2, 6))
        s.goto(99); self.assertEqual(s.progress(), (6, 6)); s.goto(-5); self.assertEqual(s.position, 0); place(s, "AL_corner"); s.submit(); self.assertTrue(s.reachable(1)); self.assertFalse(s.reachable(2))

    def test_completed_work_survives_navigation_and_is_not_confused_between_boundaries(self):
        s = new_session(); place(s, "AL_corner"); s.submit(); place(s, "PL_corner"); s.submit(); s.goto(0); self.assertEqual(s.state().points, [[504.0, 800.0], [507.0, 2250.0]]); self.assertTrue(s.state().accepted)
        s.goto(1); self.assertEqual(s.state().points, [[671.0, 700.0], [697.0, 2400.0]]); s.goto(2); self.assertEqual(s.state().points, []); self.assertFalse(s.state().accepted); self.assertEqual(s.completed(), ["AL_corner", "PL_corner"])
        s.goto(0); s.begin_edit(0); s.click(506, 810); s.goto(1); self.assertTrue(s.state().accepted); s.goto(0); self.assertFalse(s.state().accepted)       # changing one boundary never touches another

    def test_two_sessions_and_two_frame_sets_do_not_leak(self):
        a, b = new_session(), new_session(); self.assertNotEqual(a.job_id, b.job_id); place(a, "AL_corner"); self.assertEqual(b.state("AL_corner").points, []); a.submit(); self.assertIsNone(b.state("AL_corner").result)
        a.select_set(1); self.assertEqual(a.position, 0); self.assertEqual(a.state().points, []); self.assertEqual(a.record().set_index, 1); place(a, "AL_corner", [[503, 800], [509, 2200]]); a.submit(); self.assertEqual(a.state(set_index=0, line="AL_corner").points, [[504.0, 800.0], [507.0, 2250.0]])
        self.assertNotEqual(a.state(set_index=0, line="AL_corner").raw["annotation_id"], a.state(set_index=1, line="AL_corner").raw["annotation_id"]); self.assertEqual(len(a.export()["annotations"]), 2); self.assertEqual(b.export()["annotations"], [])
        self.assertEqual({r["set_index"] for r in a.export()["annotations"]}, {0, 1}); self.assertEqual(a.export()["job_id"], a.job_id)
        with self.assertRaises(ValueError): a.select_set(7)


class IsolationFromTheRestOfTheProject(unittest.TestCase):
    def test_core_has_no_streamlit_and_no_own_line_fitting(self):
        src = (UI / "core.py").read_text(); tree = ast.parse(src); imp = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertNotIn("streamlit", imp); names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}; self.assertFalse({"lines", "geometry", "measure", "pose", "features", "validation"} & names)
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)} | {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertFalse(calls & {"fit_line", "fit_lines", "edge_points", "polyfit", "lstsq", "HoughLines", "HoughLinesP", "Canny", "fitLine"}, calls)       # no line fitting of its own
        for bad in ("GROUND_TRUTH", "wall_boundary_annotations", "reference_consensus"): self.assertNotIn(bad, src, bad)
        self.assertIn("VM.evaluate_set", src); self.assertIn("VM.gate_set", src); self.assertIn("VM.Confirmation", src)                                   # the backend does the work

    def test_the_prototype_is_not_linked_from_existing_pages(self):
        for f in (ROOT / "Pages").glob("*.py"): self.assertNotIn("annotation_ui", f.read_text(), f.name)
        for f in (ROOT / "home.py", ROOT / "home_navigation.py"): self.assertNotIn("annotation_ui", f.read_text())
        self.assertNotIn("app_module7", (UI / "app_annotate.py").read_text().replace("Pages/app_module7.py", ""))

    def test_no_tracked_file_was_modified(self):
        self.assertEqual(subprocess.run(["git", "diff", "--name-only"], cwd=ROOT, capture_output=True, text=True).stdout.strip(), "")

    def test_the_launcher_is_local_only(self):
        src = (MODULE / "run_stage_f1_annotation_ui.py").read_text(); self.assertIn("127.0.0.1", src); self.assertIn("8620", src); self.assertNotIn("0.0.0.0", src)

    def test_no_f1_outputs_under_the_deployable_results_directory(self):
        out = MODULE / "outputs" / "room_measurement"; self.assertEqual([p for p in out.rglob("*") if "f1" in p.name.lower() and p.is_file()], [])


try:
    from streamlit.testing.v1 import AppTest
except Exception:                                                       # pragma: no cover
    AppTest = None


@needs_frames
@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class PageRerunBehaviour(unittest.TestCase):
    """The page with the click component replaced by a scripted stand-in that clicks FULL-RESOLUTION targets (converted with the page's own view)."""
    def make(self, queue):
        def fake(image, key=None, width=None, **kw):
            import streamlit as st
            if not queue: return None
            f1 = st.session_state["f1"]; s = f1["session"]; view = core.make_view(s.record().image_wh, f1["views"].get((s.set_index, s.line), core.ViewState())); x, y = queue.pop(0)
            u, v = view.to_view(x, y); assert 0 <= u < image.size[0] and 0 <= v < image.size[1], "target not inside the visible crop"
            return dict(x=int(u), y=int(v), width=image.size[0], height=image.size[1], unix_time=1000 + len(f1["last_click"]))
        patcher = mock.patch("streamlit_image_coordinates.streamlit_image_coordinates", fake); patcher.start(); self.addCleanup(patcher.stop)
        at = AppTest.from_file(str(UI / "app_annotate.py"), default_timeout=90); at.run(); return at

    def btn(self, at, label): return [b for b in at.button if b.label == label][0]

    def near(self, pts, targets, tol=5.0):
        """At zoom 1 one displayed pixel is ~4.3 full-resolution pixels, so a click lands within that of the target (zoom in for precision)."""
        self.assertEqual(len(pts), len(targets))
        for p, t in zip(pts, targets): self.assertLessEqual(max(abs(p[0] - t[0]), abs(p[1] - t[1])), tol, (p, t))

    def test_initial_render(self):
        at = self.make([]); self.assertFalse(at.exception); text = " ".join(m.value for m in at.markdown)
        self.assertIn("Annotate Wall Boundaries", text); self.assertIn("Boundary 1 of 6", text); self.assertIn("AL_corner", text); self.assertIn("placement 1", text); self.assertIn("frame 133", text); self.assertTrue(self.btn(at, "Refine boundary").disabled)
        self.assertTrue(self.btn(at, "Continue").disabled); self.assertTrue(self.btn(at, "PL_corner").disabled); self.assertEqual([e.label for e in at.expander][0], "How to annotate"); self.assertEqual(at.session_state["f1"]["session"].set_index, 0)

    def test_rerun_preserves_points_and_state(self):
        at = self.make([(504, 800), (507, 2250)]); st = at.session_state["f1"]["session"].state(); self.near(st.points, [[504, 800], [507, 2250]]); first = copy.deepcopy(st.points)
        sid = id(at.session_state["f1"]["session"]); at.run(); at.run(); self.assertEqual(id(at.session_state["f1"]["session"]), sid); self.assertEqual(at.session_state["f1"]["session"].state().points, first); self.assertFalse(self.btn(at, "Refine boundary").disabled)

    def test_full_annotation_flow_for_a_passing_boundary(self):
        at = self.make([(504, 800), (507, 2250)]); self.btn(at, "Refine boundary").click().run(); s = at.session_state["f1"]["session"]; first = copy.deepcopy(s.state().points); text = " ".join(m.value for m in at.markdown)
        self.assertEqual(s.state().result["decision"], "ALLOWED"); self.assertIn("Ready", text); self.assertFalse(self.btn(at, "Continue").disabled); self.assertTrue(self.btn(at, "Confirm boundary").disabled)
        self.btn(at, "Continue").click().run(); self.assertEqual(s.progress(), (2, 6)); self.assertIn("Boundary 2 of 6", " ".join(m.value for m in at.markdown)); self.assertTrue(self.btn(at, "Continue").disabled)
        self.assertFalse([b for b in at.button if b.label.endswith("AL_corner")][0].disabled)      # earlier boundary can be reviewed
        [b for b in at.button if b.label.endswith("AL_corner")][0].click().run(); self.assertEqual(s.progress(), (1, 6)); self.assertEqual(s.state().points, first); self.assertTrue(s.state().accepted)

    def test_warning_needs_the_explicit_confirm_button_and_a_point_change_resets_it(self):
        at = self.make([]); s = at.session_state["f1"]["session"]; s.goto(4); [s.click(*p) for p in CLICKS["PR_corner"]]; at.run(); self.btn(at, "Refine boundary").click().run()
        self.assertEqual(s.state().result["decision"], "BLOCKED_UNCONFIRMED_WARNING"); text = " ".join(m.value for m in at.markdown) + " ".join(w.value for w in at.warning); self.assertIn("Another nearby edge was detected", text); self.assertIn("Needs your confirmation", text)
        self.assertFalse(self.btn(at, "Confirm boundary").disabled); self.assertTrue(self.btn(at, "Continue").disabled or s.progress()[0] == 6); self.assertFalse(s.state().accepted)
        self.btn(at, "Confirm boundary").click().run(); self.assertTrue(s.state().accepted); self.assertIn("confirmed", " ".join(x.value for x in at.success).lower()); self.assertTrue(self.btn(at, "Confirm boundary").disabled)
        self.btn(at, "Move").click().run(); self.assertEqual(s.state().edit_index, 0); s.begin_edit(1); s.click(1803, 2205); at.run()
        self.assertFalse(s.state().accepted); self.assertIsNone(s.state().confirmation); self.assertIsNone(s.state().result); self.assertFalse([b for b in at.button if b.label == "Confirm boundary"])      # nothing left to confirm

    def test_fail_shows_no_usable_confirmation(self):
        at = self.make([]); s = at.session_state["f1"]["session"]; s.goto(4); [s.click(*p) for p in OFF_CLICKS["PR_corner"]]; at.run(); self.btn(at, "Refine boundary").click().run()
        self.assertEqual(s.state().result["decision"], "BLOCKED_FAIL"); self.assertTrue(self.btn(at, "Confirm boundary").disabled); self.assertIn("cannot be confirmed", " ".join(e.value for e in at.error)); self.assertIn("limit of the search area", " ".join(m.value for m in at.markdown))
        self.assertTrue(self.btn(at, "Continue").disabled or s.progress()[0] == 6)

    def test_zoom_and_pan_buttons_preserve_points_and_reset_restores_the_view(self):
        at = self.make([(504, 800), (507, 2250)]); s = at.session_state["f1"]["session"]; pts = copy.deepcopy(s.state().points); f1 = at.session_state["f1"]
        self.btn(at, "+").click().run(); self.assertEqual(f1["views"][(0, "AL_corner")].zoom, 2.0); self.assertEqual(s.state().points, pts)
        self.btn(at, "→").click().run(); self.btn(at, "↓").click().run(); vs = f1["views"][(0, "AL_corner")]; self.assertEqual(vs.zoom, 2.0); self.assertEqual(s.state().points, pts); self.assertIn("Zoom ×2", " ".join(c.value for c in at.caption))
        self.btn(at, "Reset view").click().run(); self.assertEqual(f1["views"][(0, "AL_corner")], core.ViewState()); self.assertEqual(s.state().points, pts)

    def test_a_new_job_starts_clean(self):
        at = self.make([(504, 800), (507, 2250)]); old = at.session_state["f1"]["session"].job_id; self.btn(at, "Start a new annotation job").click().run(); new = at.session_state["f1"]["session"]
        self.assertNotEqual(new.job_id, old); self.assertEqual(new.state().points, []); self.assertEqual(new.position, 0)


class SavedArtifactsAndFrames(unittest.TestCase):
    @needs_frames
    def test_local_frames_are_original_resolution_and_fingerprint_verified(self):
        f = fixture.load_frames(); self.assertEqual(len(f), 8)
        for r in f.values(): self.assertEqual(r.image_wh, WH); self.assertEqual(V.frame_fingerprint(core.load_frame(r)), r.frame_fingerprint)
        e3 = json.loads((MODULE / "outputs" / "stage_e3_annotation_frames" / "frames_index.json").read_text())["frames"]
        for x in e3: self.assertEqual(f[(x["set_index"], x["view_key"])].frame_fingerprint, x["frame_fingerprint"])        # same frames as the Stage E3 export

    @needs_frames
    def test_calibration_is_the_stage_a_adapter_output(self):
        c = fixture.load_calibration(); self.assertEqual(tuple(c.image_size), WH); self.assertEqual(c.baseline_mm, 225.0)


if __name__ == "__main__":
    unittest.main()

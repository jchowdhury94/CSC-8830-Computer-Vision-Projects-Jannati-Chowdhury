"""Phase 2B Stage D: visitor wall-boundary annotation backend (room_measurement/visitor_annotations.py).
Synthetic images only (an analytic scene rendered through a known lens-distortion model); no real video is decoded here."""
import ast
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7"
sys.path.insert(0, str(MODULE))

from room_measurement import calibration as C, geometry as G, measure as M, visitor_annotations as V  # noqa: E402
from room_measurement.visitor_annotations import Code, Status  # noqa: E402

W, H = 400, 300
K = np.array([[500.0, 0, W / 2], [0, 500.0, H / 2], [0, 0, 1]])
D0 = np.zeros(5)
D1 = np.array([-0.25, 0.08, 0.001, -0.001, 0.0])      # strong barrel distortion
EDGE_X, CEIL_Y = 200, 100


def cal(D=D0):
    return C.StereoCalibration("synthetic", K, D, K.copy(), D, (W, H), 225.0)


def render(D=D0, edge_x=EDGE_X, ceil_y=CEIL_Y, second_edge=None, noise=0.0, seed=0):
    """Analytic room: ceiling (y<ceil_y) bright; wall dark left / brighter right of a vertical edge.  Defined in UNDISTORTED coordinates and
    sampled at every DISTORTED pixel, so the stored frame is lens-distorted exactly as ``D`` says."""
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    u = cv2.undistortPoints(np.c_[gx.ravel(), gy.ravel()].reshape(-1, 1, 2), K, D, P=K, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)).reshape(H, W, 2)
    xu, yu = u[..., 0], u[..., 1]
    if second_edge is None: wall = np.where(xu < edge_x, 80.0, 180.0)
    else: wall = np.where(xu < second_edge, 60.0, np.where(xu < edge_x, 220.0, 250.0))
    im = np.where(yu < ceil_y, 235.0, wall) + np.random.RandomState(seed).randn(H, W) * noise
    return np.clip(cv2.GaussianBlur(im, (0, 0), 1.0), 0, 255).astype(np.uint8)


def ann(line="AL_corner", pts=((200, 150), (200, 270)), space="undistorted_full", set_index=0, frame_index=7, display=None, **kw):
    pl, cam = V.VIEW_PLACEMENT[line[:2]]; bt = line.split("_")[1]
    return V.build_annotation(set_index=set_index, placement=pl, camera=cam, frame_index=frame_index, boundary_type=bt, image_wh=(W, H), coordinate_space=space, points=pts, display=display, **kw)


def ctx(frame=None, D=D0):
    return V.CameraContext(render(D) if frame is None else frame, K, D)


def full_set(D=D0, shift=0.0):
    """Annotations for the six required Phase 1 lines (+ceilS omitted); undistorted clicks on the true edges."""
    return [ann("AL_corner", ((EDGE_X + shift, 150), (EDGE_X + shift, 270))), ann("PL_corner", ((EDGE_X + shift, 150), (EDGE_X + shift, 270))), ann("PL_ceilF", ((60, CEIL_Y + shift), (340, CEIL_Y + shift))),
            ann("AR_corner", ((EDGE_X + shift, 150), (EDGE_X + shift, 270))), ann("PR_corner", ((EDGE_X + shift, 150), (EDGE_X + shift, 270))), ann("PR_ceilF", ((60, CEIL_Y + shift), (340, CEIL_Y + shift)))]


def frames4(D=D0, **kw):
    f = render(D, **kw); return dict(AL=f, PL=f, AR=f, PR=f)


class ParsingAndValidation(unittest.TestCase):
    def test_valid_two_point_annotation_passes_and_refines(self):
        r = V.process_annotation(ann(pts=((203, 152), (197, 268))), ctx(), cal())          # sloppy clicks, ~3-4 px off the edge
        self.assertEqual(r.status, Status.PASS, r.issues); self.assertTrue(r.downstream_permitted)
        self.assertLess(abs(r.downstream_line["a"] * 210 + r.downstream_line["b"] - (EDGE_X - 0.5)), 0.6)
        self.assertEqual(r.refined_line["kind"], "v"); self.assertGreater(r.diagnostics["n_edge_points"], 15); self.assertGreater(r.diagnostics["displacement_max_px"], 1.0)

    def test_missing_fields(self):
        a = ann(); del a["frame_index"]; del a["points"]
        r = V.process_annotation(a, ctx(), cal()); self.assertEqual(r.status, Status.FAIL); self.assertIn(Code.MISSING_FIELD, r.failure_codes); self.assertFalse(r.downstream_permitted)
        self.assertIsNone(r.downstream_line)

    def test_invalid_coordinates(self):
        for pts, code in ((((np.nan, 150), (200, 270)), Code.NON_FINITE), (((200, 150), (np.inf, 270)), Code.NON_FINITE), (((200, 150), (200, 400)), Code.OUT_OF_BOUNDS),
                          (((-1, 150), (200, 270)), Code.OUT_OF_BOUNDS), ((("a", 150), (200, 270)), Code.BAD_POINTS), (((200, 150),), Code.BAD_POINTS)):
            a = ann(); a["points"] = [list(p) for p in pts]
            r = V.process_annotation(a, ctx(), cal()); self.assertEqual(r.status, Status.FAIL, pts); self.assertIn(code, r.failure_codes, (pts, r.failure_codes)); self.assertIsNone(r.downstream_line)

    def test_bad_enums_and_spaces(self):
        for key, val, code in (("coordinate_space", "pixels", Code.BAD_SPACE), ("boundary_type", "floor", Code.BAD_BOUNDARY), ("camera", "middle", Code.BAD_VALUE), ("placement", "placement_3", Code.BAD_VALUE),
                               ("frame_index", -1, Code.BAD_VALUE), ("frame_index", True, Code.BAD_VALUE)):
            a = ann(); a[key] = val; r = V.process_annotation(a, ctx(), cal()); self.assertIn(code, r.failure_codes, (key, r.failure_codes))

    def test_duplicate_endpoints(self):
        r = V.process_annotation(ann(pts=((200, 150), (200.4, 150.3))), ctx(), cal()); self.assertIn(Code.DUPLICATE_ENDPOINTS, r.failure_codes)

    def test_very_short_line(self):
        r = V.process_annotation(ann(pts=((200, 150), (200, 200))), ctx(), cal()); self.assertIn(Code.TOO_SHORT, r.failure_codes); self.assertIsNone(r.downstream_line)
        r = V.process_annotation(ann(pts=((200, 150), (200, 200))), ctx(), cal(), V.AnnotationLimits(min_length_px=40)); self.assertNotIn(Code.TOO_SHORT, r.failure_codes)   # configurable

    def test_wrong_frame_and_set_identifiers(self):
        r = V.process_annotation(ann(frame_index=7), ctx(), cal(), expected_frame_index=8); self.assertIn(Code.WRONG_FRAME, r.failure_codes)
        r = V.process_annotation(ann(set_index=1), ctx(), cal(), expected_set_index=0); self.assertIn(Code.WRONG_SET, r.failure_codes)
        r = V.process_annotation(ann(frame_fingerprint="0" * 64), ctx(), cal()); self.assertIn(Code.FINGERPRINT, r.failure_codes)
        f = render(); r = V.process_annotation(ann(frame_fingerprint=V.frame_fingerprint(f)), ctx(f), cal()); self.assertEqual(r.status, Status.PASS)
        r = V.process_annotation(ann(), None, cal()); self.assertIn(Code.NO_FRAME, r.failure_codes)

    def test_wrong_camera_identifiers(self):
        a = ann("AL_corner"); a["line_name"] = "PL_corner"; self.assertIn(Code.LINE_NAME, V.process_annotation(a, ctx(), cal()).failure_codes)
        a = ann("AL_corner"); a["camera"] = "right"; self.assertIn(Code.LINE_NAME, V.process_annotation(a, ctx(), cal()).failure_codes)
        a = V.build_annotation(set_index=0, placement="placement_1", camera="left", frame_index=7, boundary_type="ceilF", image_wh=(W, H), coordinate_space="undistorted_full", points=((60, 100), (340, 100)))
        self.assertIn(Code.BAD_CAMERA_BOUNDARY, V.process_annotation(a, ctx(), cal()).failure_codes)       # Air cannot see the ceiling

    def test_image_and_calibration_size_checks(self):
        a = ann(); a["image"] = dict(width=W // 2, height=H // 2); self.assertIn(Code.SIZE, V.process_annotation(a, ctx(), cal()).failure_codes)
        small = C.StereoCalibration("s", K, D0, K.copy(), D0, (W // 2, H // 2), 225.0)
        self.assertIn(Code.CAL_SIZE, V.process_annotation(ann(), ctx(), small).failure_codes)       # intrinsics are never rescaled

    def test_unreasonable_orientation(self):
        r = V.process_annotation(ann(pts=((100, 150), (300, 250))), ctx(), cal()); self.assertIn(Code.ORIENTATION, r.failure_codes)           # corner at 27 deg off vertical... horizontal-ish
        r = V.process_annotation(ann("PL_ceilF", pts=((200, 40), (230, 280))), ctx(), cal()); self.assertIn(Code.ORIENTATION, r.failure_codes)

    def test_input_is_not_mutated(self):
        a = ann(); before = copy.deepcopy(a); f = render(); fb = f.copy(); V.process_annotation(a, ctx(f), cal()); self.assertEqual(a, before); self.assertTrue(np.array_equal(f, fb))


class CoordinateTransforms(unittest.TestCase):
    def test_distorted_to_undistorted_roundtrip(self):
        pts = np.array([[20.0, 30.0], [200, 150], [390, 280], [50, 250]]); d = V.distort_pixels(pts, K, D1)
        self.assertGreater(np.abs(d - pts).max(), 5)                                   # the distortion is real
        self.assertLess(np.abs(G.undistort_pixels(d, K, D1) - pts).max(), 0.02)

    def test_distorted_clicks_end_to_end(self):
        """Visitor clicks on the DISTORTED frame at two points of the (curved) edge; the refined line must be the straight undistorted edge."""
        ex = 300; p_und = np.array([[ex, 160.0], [ex, 270.0]]); p_dist = V.distort_pixels(p_und, K, D1) + [2.5, -1.5]
        self.assertGreater(abs(p_dist[0, 0] - p_dist[1, 0]), 0.5)                      # curved edge: clicks are not vertically aligned
        r = V.process_annotation(ann(pts=p_dist, space="distorted_full"), V.CameraContext(render(D1, edge_x=ex), K, D1), cal(D1))
        self.assertEqual(r.status, Status.PASS, r.issues); l = r.downstream_line
        for y in (100, 150, 270): self.assertLess(abs(l["a"] * y + l["b"] - (ex - 0.5)), 1.0)      # true edge lies between pixels 299 and 300; <1 px vs >5 px of distortion
        self.assertEqual(r.transforms["distorted_to_undistorted"], "applied"); self.assertFalse(r.transforms["intrinsics_rescaled"])

    def test_same_clicks_treated_as_undistorted_would_be_wrong(self):
        p_dist = V.distort_pixels(np.array([[300.0, 160.0], [300.0, 270.0]]), K, D1); c1 = V.CameraContext(render(D1, edge_x=300), K, D1)
        a_ok = V.process_annotation(ann(pts=p_dist, space="distorted_full"), c1, cal(D1))
        a_bad = V.process_annotation(ann(pts=p_dist, space="undistorted_full"), c1, cal(D1))      # wrong declared space: different initial line
        self.assertNotEqual(np.round(a_ok.initial_line["a"], 4), np.round(a_bad.initial_line["a"], 4))

    def test_letterbox_mapping_known_values(self):
        m = V.DisplayMapping.letterbox((2160, 3840), (1000, 800))          # portrait frame in a landscape canvas: scale 800/3840, pad left/right
        s = 800 / 3840; self.assertAlmostEqual(m.scale_x, s); self.assertAlmostEqual(m.scale_y, s); self.assertAlmostEqual(m.image_h, 800); self.assertAlmostEqual(m.image_x, (1000 - 2160 * s) / 2)
        self.assertEqual(m.validate(), [])
        full = m.to_full([[m.image_x, 0], [m.image_x + m.image_w, 800], [500, 400]])
        self.assertAlmostEqual(full[0, 0], -0.5); self.assertAlmostEqual(full[1, 0], 2160 - 0.5); self.assertAlmostEqual(full[1, 1], 3840 - 0.5)
        self.assertAlmostEqual(full[2, 0], 1080 - 0.5); self.assertAlmostEqual(full[2, 1], 1920 - 0.5)       # canvas centre = frame centre
        self.assertTrue(np.allclose(m.from_full(m.to_full([[333.3, 444.4]])), [[333.3, 444.4]]))

    def test_scaling_without_padding_and_naive_mapping_is_wrong(self):
        m = V.DisplayMapping.letterbox((W, H), (W // 2, H // 2)); self.assertAlmostEqual(m.image_x, 0); self.assertAlmostEqual(m.scale_x, 0.5)
        self.assertTrue(np.allclose(m.to_full([[100, 75]]), [[199.5, 149.5]]))
        pad = V.DisplayMapping.letterbox((W, H), (800, 800))              # padded: naive 'divide by scale' ignores the offset
        click = np.array([[pad.image_x + 100, pad.image_y + 50]]); naive = click / pad.scale_x
        self.assertGreater(np.abs(naive - pad.to_full(click)).max(), 50); self.assertTrue(np.allclose(pad.to_full(click), [[100 / pad.scale_x - 0.5, 50 / pad.scale_y - 0.5]]))

    def test_padding_click_and_stretched_display_are_rejected(self):
        pad = V.DisplayMapping.letterbox((W, H), (800, 800)); self.assertGreaterEqual(pad.image_y, 100)
        r = V.process_annotation(ann(pts=((400, 5), (400, 600)), space="preview_display", display=pad), ctx(), cal()); self.assertIn(Code.PADDING, r.failure_codes)
        stretched = V.DisplayMapping(W, H, 400, 600, 0, 0, 400, 600)           # 1:1 in x, 2:1 in y
        r = V.process_annotation(ann(pts=((200, 300), (200, 500)), space="preview_display", display=stretched), ctx(), cal()); self.assertIn(Code.ASPECT, r.failure_codes)
        r = V.process_annotation(ann(pts=((1, 1), (2, 2)), space="preview_display"), ctx(), cal()); self.assertIn(Code.MISSING_FIELD, r.failure_codes)     # no display mapping

    def test_preview_clicks_with_padding_and_scaling_end_to_end(self):
        d = V.DisplayMapping.letterbox((W, H), (640, 640)); p_full = np.array([[EDGE_X + 2.0, 160.0], [EDGE_X + 2.0, 270.0]])      # 2 px click error
        r = V.process_annotation(ann(pts=d.from_full(p_full), space="preview_display", display=d), ctx(), cal())
        self.assertEqual(r.status, Status.PASS, r.issues); self.assertTrue(np.allclose(r.transforms["points_distorted_full"], p_full, atol=1e-6))
        self.assertLess(abs(r.downstream_line["a"] * 200 + r.downstream_line["b"] - EDGE_X), 0.5)
        self.assertLess(r.diagnostics["displacement_max_px"], 3.5)                       # a mapping error of the padding would be tens of px

    def test_preview_plus_distortion_chain(self):
        d = V.DisplayMapping.letterbox((W, H), (500, 700)); p_dist = V.distort_pixels(np.array([[300.0, 160.0], [300.0, 270.0]]), K, D1)
        r = V.process_annotation(ann(pts=d.from_full(p_dist), space="preview_display", display=d), V.CameraContext(render(D1, edge_x=300), K, D1), cal(D1))
        self.assertEqual(r.status, Status.PASS, r.issues); self.assertLess(abs(r.downstream_line["a"] * 220 + r.downstream_line["b"] - 300), 0.5)


class Refinement(unittest.TestCase):
    def test_success_diagnostics(self):
        r = V.process_annotation(ann(pts=((198, 150), (202, 270))), ctx(), cal()); g = r.diagnostics
        for k in ("n_edge_points", "inlier_fraction", "fit_rms_px", "displacement_max_px", "displacement_mean_px", "angle_change_deg", "coverage_fraction", "edge_strength"): self.assertIn(k, g)
        self.assertEqual(r.status, Status.PASS); self.assertLess(g["fit_rms_px"], 0.5); self.assertGreater(g["inlier_fraction"], 0.9); self.assertEqual(g["pass_used"], "pass2")
        self.assertIn("angle_deg", r.refined_line); self.assertIn("angle_deg", r.initial_line); self.assertEqual(g["polarity_source"], "inferred"); self.assertEqual(g["polarity"], 1)

    def test_failure_on_blank_image_is_explicit_not_substituted(self):
        r = V.process_annotation(ann(), ctx(np.full((H, W), 90, np.uint8)), cal())
        self.assertEqual(r.status, Status.FAIL); self.assertIn(Code.REFINE_FAILED, r.failure_codes); self.assertIsNone(r.refined_line); self.assertIsNone(r.downstream_line)
        self.assertIsNotNone(r.initial_line); self.assertFalse(r.downstream_permitted)       # the unrefined click is never passed on

    def test_edge_outside_window_does_not_grab_unrelated_edge(self):
        r = V.process_annotation(ann(pts=((130, 150), (130, 270))), ctx(), cal())          # the real edge is 70 px away
        self.assertEqual(r.status, Status.FAIL); self.assertIsNone(r.downstream_line)

    def test_wrong_edge_competing_edge_is_flagged(self):
        f = render(second_edge=150)                                                        # strong edge at 150, weak one at 200
        r = V.process_annotation(ann(pts=((201, 150), (199, 270))), ctx(f), cal())
        self.assertAlmostEqual(r.downstream_line["a"] * 200 + r.downstream_line["b"], 200, delta=0.7)       # fit is excellent...
        self.assertLess(r.diagnostics["fit_rms_px"], 0.6)
        self.assertEqual(r.status, Status.WARNING); self.assertIn(Code.COMPETING, r.warning_codes)             # ...yet it is flagged: RMS alone cannot prove the physical edge
        self.assertGreater(r.diagnostics["competitor_ratio"], 1.0)

    def test_click_between_two_edges_is_not_pass(self):
        f = render(second_edge=150); r = V.process_annotation(ann(pts=((175, 150), (175, 270))), ctx(f), cal())
        self.assertNotEqual(r.status, Status.PASS)

    def test_edge_at_window_limit_fails(self):
        r = V.process_annotation(ann(pts=((181, 150), (181, 270))), ctx(), cal()); self.assertEqual(r.status, Status.FAIL, (r.issues, r.diagnostics)); self.assertIn(Code.AT_LIMIT, r.failure_codes)

    def test_wrong_visitor_polarity_does_not_find_the_edge(self):
        r = V.process_annotation(ann(edge_polarity=-1), ctx(), cal()); self.assertEqual(r.status, Status.FAIL)
        r = V.process_annotation(ann(edge_polarity=1), ctx(), cal()); self.assertEqual(r.status, Status.PASS); self.assertEqual(r.diagnostics["polarity_source"], "visitor")

    def test_search_window_leaving_image_is_clipped_not_wrapped(self):
        f = render(edge_x=14); r = V.process_annotation(ann(pts=((14, 150), (14, 270))), ctx(f), cal(), V.AnnotationLimits(search_half_window_px=20))
        self.assertIn(Code.WINDOW, r.failure_codes)
        self.assertNotEqual(r.status, Status.PASS)                                         # window [-6, 34] leaves the image -> never silently wrapped

    def test_noisy_image(self):
        f = render(noise=6.0, seed=3); r = V.process_annotation(ann(pts=((204, 150), (196, 270))), ctx(f), cal())
        self.assertIn(r.status, (Status.PASS, Status.WARNING)); self.assertLess(abs(r.downstream_line["a"] * 210 + r.downstream_line["b"] - EDGE_X), 1.0)

    def test_horizontal_ceiling_line(self):
        r = V.process_annotation(ann("PL_ceilF", pts=((60, 103), (340, 97))), ctx(), cal()); self.assertEqual(r.status, Status.PASS, r.issues); self.assertEqual(r.refined_line["kind"], "h")
        self.assertLess(abs(r.downstream_line["a"] * 200 + r.downstream_line["b"] - CEIL_Y), 0.7); self.assertEqual(r.diagnostics["polarity"], -1)


class AnnotationSets(unittest.TestCase):
    def test_complete_set_passes_and_gives_phase1_lines(self):
        s = V.process_annotation_set(full_set(), frames4(), cal(), set_index=0, expected_frame_indices=dict(AL=7, PL=7, AR=7, PR=7))
        self.assertEqual(s.status, Status.PASS, s.issues); self.assertTrue(s.downstream_use_permitted); lines = s.lines_for_measurement()
        self.assertEqual(set(lines), set(V.REQUIRED_LINES)); self.assertTrue(s.completeness["complete_for_variant0"]); self.assertFalse(s.completeness["complete_all_phase1_lines"])
        for ln in lines.values(): self.assertTrue(all(k in ln for k in ("kind", "a", "b", "n", "N", "rms", "span", "pts")))      # Phase 1 fitted-line format
        for v, (cl, cp, ce, _) in M.VIEW_LINES.items():                                    # the unchanged Phase 1 line algebra accepts them
            self.assertTrue(np.isfinite(G.line_intersection(lines[cp], lines[ce])).all()); self.assertTrue(np.isfinite(G.line_to_h(lines[cl])).all())
        self.assertEqual(s.stereo_consistency["V1"]["status"], "NOT_EVALUATED")             # no pose supplied: never PASS

    def test_missing_required_boundaries_fail(self):
        a = [x for x in full_set() if x["line_name"] != "PR_ceilF"]
        s = V.process_annotation_set(a, frames4(), cal()); self.assertEqual(s.status, Status.FAIL); self.assertFalse(s.downstream_use_permitted); self.assertIn(Code.MISSING_REQUIRED, s.failure_codes)
        self.assertIsNone(s.line_set); self.assertEqual(s.completeness["missing"], ["PR_ceilF"])
        with self.assertRaises(V.AnnotationError): s.lines_for_measurement()
        self.assertEqual(V.process_annotation_set([], frames4(), cal()).status, Status.FAIL)

    def test_one_invalid_required_annotation_blocks_downstream(self):
        a = full_set(); a[1]["points"] = [[200, 150], [200, 152]]
        s = V.process_annotation_set(a, frames4(), cal()); self.assertFalse(s.downstream_use_permitted); self.assertIn(Code.REQUIRED_INVALID, s.failure_codes); self.assertIn(Code.TOO_SHORT, s.failure_codes)

    def test_duplicate_annotation_is_not_silently_resolved(self):
        a = full_set() + [ann("AL_corner", ((201, 150), (201, 270)))]
        s = V.process_annotation_set(a, frames4(), cal()); self.assertFalse(s.downstream_use_permitted); self.assertIn(Code.DUPLICATE_ANNOTATION, s.failure_codes)

    def test_optional_ceilS_is_optional_unless_requested(self):
        a = full_set(); self.assertTrue(V.process_annotation_set(a, frames4(), cal()).downstream_use_permitted)
        s = V.process_annotation_set(a, frames4(), cal(), V.AnnotationLimits(require_ceiling_side=True)); self.assertFalse(s.downstream_use_permitted)
        self.assertIn(Code.MISSING_REQUIRED, s.failure_codes)

    def test_wrong_frame_in_set(self):
        s = V.process_annotation_set(full_set(), frames4(), cal(), set_index=0, expected_frame_indices=dict(AL=7, PL=8, AR=7, PR=7))
        self.assertFalse(s.downstream_use_permitted); self.assertIn(Code.WRONG_FRAME, s.failure_codes)

    def test_occluded_corner_is_inferred_not_required_visible(self):
        """Only a 90 px stretch of each corner edge is annotated (a sofa hides the rest, so the corner/ceiling junction is NOT annotated either)."""
        a = [x for x in full_set() if "corner" not in x["line_name"]] + [ann(n, ((EDGE_X, 125), (EDGE_X, 215))) for n in ("AL_corner", "PL_corner", "AR_corner", "PR_corner")]
        s = V.process_annotation_set(a, frames4(), cal(), V.AnnotationLimits(min_length_px=80)); self.assertTrue(s.downstream_use_permitted, (s.issues, s.failure_codes))
        row = s.reconstruction["V1"]; self.assertFalse(row["intersection_on_annotated_segment"]); self.assertGreater(row["extrapolation_px"], 20)    # the corner itself lies outside what was annotated
        self.assertTrue(np.allclose(row["intersection_px"], [EDGE_X - 0.5, CEIL_Y - 0.5], atol=1.0))       # but is inferred correctly by the existing line algebra
        s2 = V.process_annotation_set(a, frames4(), cal(), V.AnnotationLimits(min_length_px=80, max_extrapolation_ratio=0.1)); self.assertEqual(s2.status, Status.WARNING); self.assertIn(Code.EXTRAPOLATION, [i.code for i in s2.issues])

    def test_unreconstructible_geometry_fails(self):
        a = [x for x in full_set() if x["line_name"] != "PL_ceilF"] + [ann("PL_ceilF", ((60, 100), (340, 100)))]
        a = [x for x in a if x["line_name"] != "PL_corner"] + [ann("PL_corner", ((100, 150), (130, 270)))]      # corner almost 15 deg off vertical OK for the click ...
        s = V.process_annotation_set(a, frames4(), cal()); self.assertFalse(s.downstream_use_permitted)                  # ... but no edge there -> refinement fails
        self.assertIn(Code.REQUIRED_INVALID, s.failure_codes)

    def test_stereo_consistency_with_pose(self):
        R, T = np.eye(3), np.array([-225.0, 0, 0]); xl, xr = 200 + 500 * 500 / 3000, 200 + 500 * 275 / 3000           # a vertical 3D edge at x=500 mm, z=3000 mm
        ll = dict(kind="v", a=0.0, b=xl, span=(140.0, 280.0)); lr = dict(kind="v", a=0.0, b=xr, span=(140.0, 280.0))
        st, issues, info = V.stereo_consistency(cal(), R, T, ll, lr); self.assertEqual(st, Status.PASS, issues); self.assertAlmostEqual(info["depth_left_mm"], 3000, delta=1)
        st, issues, info = V.stereo_consistency(cal(), R, T, lr, ll); self.assertEqual(st, Status.FAIL); self.assertEqual(issues[0].code, Code.STEREO_DEPTH)          # swapped: behind the cameras
        lp = dict(ll, b=xl + 0.2); st, issues, _ = V.stereo_consistency(cal(), R, T, lp, dict(lr, b=xr + 0.1)); self.assertNotEqual(st, Status.FAIL)
        st, issues, _ = V.stereo_consistency(cal(), R, T, ll, dict(lr, span=(10.0, 20.0))); self.assertEqual(st, Status.WARNING); self.assertEqual(issues[0].code, Code.STEREO_OVERLAP)   # disjoint visible parts
        st, issues, _ = V.stereo_consistency(cal(), R, T, ll, dict(lr, span=(150.0, 230.0))); self.assertEqual(st, Status.PASS)                                                           # partial overlap is fine
        st, issues, _ = V.stereo_consistency(cal(), R, T, ll, dict(lr, b=xl)); self.assertNotEqual(st, Status.PASS)             # identical slopes + this baseline -> parallel planes, no common 3D line

    def test_stereo_in_a_set_with_pose(self):
        xl, xr = 200 + 500 * 500 / 3000, 200 + 500 * 275 / 3000
        fl, fr = render(edge_x=xl), render(edge_x=xr); frames = dict(AL=fl, PL=fr, AR=fl, PR=fr)
        a = [ann("AL_corner", ((xl, 150), (xl, 270))), ann("PL_corner", ((xr, 150), (xr, 270))), ann("PL_ceilF", ((60, 100), (340, 100))), ann("AR_corner", ((xl, 150), (xl, 270))),
             ann("PR_corner", ((xr, 150), (xr, 270))), ann("PR_ceilF", ((60, 100), (340, 100)))]
        s = V.process_annotation_set(a, frames, cal(), pose=(np.eye(3), np.array([-225.0, 0, 0])))
        self.assertEqual(s.stereo_consistency["V1"]["status"], "PASS", s.stereo_consistency); self.assertIn("physical wall edge", s.stereo_consistency["V1"]["note"])
        self.assertEqual({r.line_name: r.stereo_consistency for r in s.annotations}["PL_ceilF"], Status.NOT_EVALUATED)          # single-camera boundary: not evaluable
        bad = V.process_annotation_set(a, dict(AL=fr, PL=fl, AR=fl, PR=fr), cal(), pose=(np.eye(3), np.array([-225.0, 0, 0])))      # left/right frames swapped in V1
        self.assertNotEqual(bad.stereo_consistency["V1"]["status"], "PASS")

    def test_nothing_returned_downstream_unless_permitted(self):
        s = V.process_annotation_set(full_set(), frames4(edge_x=EDGE_X + 60), cal())            # clicks are 60 px away from every real edge
        self.assertFalse(s.downstream_use_permitted); self.assertIsNone(s.line_set); self.assertTrue(all(a.downstream_line is None for a in s.annotations if a.status is Status.FAIL))


class Serialization(unittest.TestCase):
    def test_result_is_json_serialisable_and_roundtrips(self):
        d = copy.deepcopy(full_set()); d[0]["points"] = [[199.0, 151], [201, 269]]
        s = V.process_annotation_set(dict(schema=V.SCHEMA, annotations=d), frames4(), cal(), pose=None); js = json.dumps(s.to_dict()); back = json.loads(js)
        self.assertEqual(back["status"], "PASS"); self.assertTrue(back["downstream_use_permitted"]); self.assertEqual(len(back["annotations"]), 6)
        a0 = back["annotations"][0]
        for k in ("metadata", "points_original", "transforms", "initial_line", "refined_line", "diagnostics", "issues", "status", "downstream_permitted", "failure_codes"): self.assertIn(k, a0)
        self.assertNotIn("pts", a0["refined_line"]); self.assertIn(back["stereo_consistency"]["V1"]["status"], ("NOT_EVALUATED",))
        self.assertEqual(set(json.loads(json.dumps({s.value: 1 for s in Status}))), {"PASS", "WARNING", "FAIL", "NOT_EVALUATED"})

    def test_failed_result_serialises_with_codes(self):
        r = V.process_annotation(ann(pts=((np.nan, 1), (2, 3))), ctx(), cal()); b = json.loads(json.dumps(r.to_dict(), allow_nan=False)); self.assertEqual(b["status"], "FAIL"); self.assertIn(Code.NON_FINITE, b["failure_codes"])

    def test_unsupported_schema(self):
        s = V.process_annotation_set(dict(schema="other.v9", annotations=full_set()), frames4(), cal()); self.assertFalse(s.downstream_use_permitted); self.assertIn(Code.SCHEMA, s.failure_codes)


class ReadOnlySafety(unittest.TestCase):
    SRC = MODULE / "room_measurement" / "visitor_annotations.py"

    def test_module_has_no_file_video_or_ground_truth_access(self):
        tree = ast.parse(self.SRC.read_text()); names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom): names |= {a.name for a in n.names} | {n.module or ""}
            if isinstance(n, ast.Import): names |= {a.name for a in n.names}
        self.assertNotIn("validation", names); self.assertNotIn("frames", names); self.assertNotIn("pipeline_validation", names)
        text = self.SRC.read_text()
        for bad in ("VideoCapture", "write_text", "write_bytes", "open(", "os.remove", "shutil", "2900", "2400", "ground_truth_dimension"): self.assertNotIn(bad, text, bad)

    def test_processing_does_not_touch_protected_files(self):
        protected = [MODULE / "room_measurement" / "config" / "wall_boundary_annotations.json", MODULE / "room_measurement" / "config" / "frame_manifest.json", MODULE / "calibration" / "room_video" / "room_video_recipe_a_calibration.json"]
        before = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in protected if p.exists()}
        V.process_annotation_set(full_set(), frames4(), cal()); after = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in before}
        self.assertEqual(before, after)

    def test_phase1_names_come_from_the_phase1_module(self):
        self.assertEqual(set(V.ALL_PHASE1_LINES), set(json.loads((MODULE / "room_measurement" / "config" / "wall_boundary_annotations.json").read_text())["lines"]))
        self.assertEqual(set(V.REQUIRED_LINES) | set(V.OPTIONAL_LINES), set(V.ALL_PHASE1_LINES))


if __name__ == "__main__":
    unittest.main()

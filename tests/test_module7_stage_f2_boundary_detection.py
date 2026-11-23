"""Stage F2 focused tests: automatic room-boundary detection prototype (synthetic scenes + optional original-room regression)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "Modules" / "Module_7"
sys.path.insert(0, str(MODULE))

from room_measurement import calibration as C, visitor_annotations as V, visitor_measurement as VM     # noqa: E402
from room_measurement import stage_f2_boundary_detection as F                                         # noqa: E402

W, H = 1080, 1920
LOCAL = MODULE / "outputs" / "stage_f1_local_frames"


def make_cal():
    K = np.array([[1500.0, 0, W / 2], [0, 1500.0, H / 2], [0, 0, 1]]); D = np.zeros(5)
    return C.StereoCalibration(name="synthetic", K_left=K, D_left=D, K_right=K.copy(), D_right=D.copy(), image_size=(W, H), baseline_mm=225.0)


def noise(shape, seed=0, s=0.6): return np.random.default_rng(seed).normal(0, s, shape).astype(np.float32)


def scene(corner_x=400, casing=None, ceiling_y=None, corner_step=9, occlude=None, seed=0):
    """Wall scene: darker left wall / lighter right wall (a faint corner step), optional bright door-casing strip, optional ceiling edge."""
    g = np.full((H, W), 122.0, np.float32); g[:, corner_x:] += corner_step
    yy = np.arange(H, dtype=np.float32)[:, None]; g += (yy - H / 2) * 0.004            # mild shading
    if casing is not None: g[:, casing:casing + 50] = 215.0; g[:, casing + 22:casing + 24] = 170.0     # bright strip with a faint gap line (casing + door)
    if ceiling_y is not None: g[:ceiling_y, :] -= 16.0
    g[1300:1550, 800:1050] = 232.0; g[1600:1850, 800:1050] = 25.0           # bright + dark objects far from the corner give the frame a realistic dynamic range
    if occlude is not None:
        x0, x1, y0, y1 = occlude; g[y0:y1, x0:x1] = 25.0
    g += noise(g.shape, seed); return np.clip(g, 0, 255).astype(np.uint8)


def det(gray, vk="PL"):
    cal = make_cal(); K, D = (cal.K_left, cal.D_left) if vk[0] == "A" else (cal.K_right, cal.D_right)
    p = F.prepare(gray, K, D, vk); return p, F.detect_view_boundaries(p, F.PLACEMENT_OF[vk])


def x_at(c, row_full):
    l = c["line"]; return l["a"] * row_full + l["b"]


# ------------------------------------------------------------------ candidate generation
def test_candidate_generation_finds_faint_corner_step():
    p, cands = det(scene(), "AL"); pool = cands["corner"]
    assert pool, "no candidates"
    best = pool[0]; line = best.full_line(*p.s.shape)
    assert abs(line["a"] * 1000 + line["b"] - 400) < 8 and best.length_px_full > 800


def test_candidates_are_bounded_and_have_visible_segments():
    p, cands = det(scene(), "AL"); H_, W_ = p.s.shape
    assert 0 < len(cands["corner"]) <= F.DEFAULT_LIMITS.pool
    for c in cands["corner"]:
        d = c.full_line(H_, W_); lo, hi = d["span"]; assert 0 <= lo < hi <= H and c.length_px_full >= F.DEFAULT_LIMITS.min_length_px_candidate


# ------------------------------------------------------------------ ranking, wrong-edge rejection
def test_band_like_casing_is_not_ranked_above_isolated_corner_step():
    g = scene(corner_x=330, casing=640, seed=1); p, cands = det(g, "AL"); H_, W_ = p.s.shape
    top = cands["corner"][0]; x = top.full_line(H_, W_); xm = x["a"] * 1000 + x["b"]
    assert abs(xm - 330) < 10, f"selected x={xm}"                  # the wall-plane step, not the bright casing edges (x=640 / 690)
    casing = [c for c in cands["corner"] if abs(c.full_line(H_, W_)["a"] * 1000 + c.full_line(H_, W_)["b"] - 640) < 12]
    assert casing and all(c.rank_score < top.rank_score for c in casing)
    assert any(c.confuser != "ISOLATED" or c.flags["contrast_factor"] < 1 for c in casing)    # flagged as band / high-contrast


def test_wrong_edge_only_scene_is_never_detected_as_corner():
    """The wall corner is absent; only a casing strip exists.  The strong casing edge must not become a DETECTED corner."""
    g = scene(corner_step=0, casing=640, seed=2); cal = make_cal()
    res = F.detect_set({"AL": g, "PL": g, "AR": g, "PR": g}, cal, set_index=0)
    for bid in ("AL_corner", "PL_corner", "AR_corner", "PR_corner"): assert res["boundaries"][bid].status != "DETECTED", bid


def test_contrast_normalisation_keeps_decision_under_low_contrast():
    g = scene(corner_x=330, casing=640, seed=3); low = np.clip((g.astype(np.float32) - 128) * 0.4 + 128, 0, 255).astype(np.uint8)
    for im in (g, low):
        p, cands = det(im, "AL"); top = cands["corner"][0]; l = top.full_line(*p.s.shape); assert abs(l["a"] * 1000 + l["b"] - 330) < 12


# ------------------------------------------------------------------ occlusion
def test_partially_occluded_corner_returns_visible_segment_only():
    g = scene(occlude=(250, 560, 1100, 1900), seed=4); p, cands = det(g, "AL"); H_, W_ = p.s.shape
    top = cands["corner"][0]; d = top.full_line(H_, W_); lo, hi = d["span"]
    assert abs(d["a"] * 800 + d["b"] - 400) < 10 and hi <= 1100 + 40         # ends at the occluder: no invented continuation
    assert top.length_px_full < 1200


def test_status_never_detected_when_visible_length_is_short():
    g = scene(occlude=(200, 600, 380, 1900), seed=5)           # only ~300 px of the corner left (below top rows) -> short
    res = F.detect_set({"AL": g}, make_cal(), set_index=0); assert res["boundaries"]["AL_corner"].status in ("UNCERTAIN", "NOT_DETECTED")


# ------------------------------------------------------------------ missing boundaries / failure handling
def test_blank_frame_gives_not_detected_with_reason():
    blank = np.full((H, W), 128, np.uint8) + (noise((H, W), 9, 0.3) > 5).astype(np.uint8)
    res = F.detect_set({"AL": blank}, make_cal(), set_index=0); br = res["boundaries"]["AL_corner"]
    assert br.status == "NOT_DETECTED" and br.failure_reason and br.selected_rank is None


def test_air_corner_is_never_detected_from_monocular_evidence_alone():
    res = F.detect_set({"AL": scene(seed=6)}, make_cal(), set_index=0); br = res["boundaries"]["AL_corner"]
    assert br.status == "UNCERTAIN" and any("Air camera" in r for r in br.reasons) and br.candidates


def test_missing_views_are_reported_not_invented():
    res = F.detect_set({"AL": scene(seed=7)}, make_cal(), set_index=0)
    assert set(res["missing_views"]) == {"PL_corner", "PL_ceilF", "AR_corner", "PR_corner", "PR_ceilF"} and "PL_corner" not in res["boundaries"]


def test_ceilF_without_corner_vertex_is_not_detected():
    """A lone long horizontal edge (door-head casing / shelf / light) is not enough: ceilF needs vertex corroboration with the corner."""
    g = scene(ceiling_y=300, seed=8); res = F.detect_set({"PL": g}, make_cal(), set_index=0); br = res["boundaries"]["PL_ceilF"]
    assert br.status != "DETECTED" and any("vertex" in r for r in br.reasons)


def test_no_status_other_than_the_three_descriptive_ones_and_no_probabilities():
    res = F.detect_set({"AL": scene(seed=10), "PL": scene(seed=11)}, make_cal(), set_index=0)
    for br in res["boundaries"].values():
        assert br.status in F.STATUSES; d = br.to_dict(); assert "confidence" not in json.dumps(d).lower() and "probability" not in json.dumps(d).lower()


# ------------------------------------------------------------------ frame identity, coordinate mapping, Stage D, E2 compatibility
def _result(vk="AL", seed=12, fp="abc", frame_index=133):
    g = scene(corner_x=330, seed=seed); res = F.detect_set({vk: g}, make_cal(), set_index=0, frame_indices={vk: frame_index}, fingerprints={vk: V.frame_fingerprint(g)}); return g, res["boundaries"][f"{vk}_corner"]


def test_frame_identity_is_carried_unchanged_into_the_stage_d_annotation():
    g, br = _result(); a = F.candidate_to_annotation(br)
    assert a["frame_fingerprint"] == V.frame_fingerprint(g) and a["frame_index"] == 133 and a["set_index"] == 0 and a["line_name"] == "placement_1_left_corner" or a["line_name"] == V.line_name("placement_1", "left", "corner")
    assert a["source"] == "stage_f2_automatic_detection" and a["image"] == dict(width=W, height=H)


def test_coordinate_mapping_points_lie_on_candidate_in_undistorted_full():
    g, br = _result(); a = F.candidate_to_annotation(br); assert a["coordinate_space"] == "undistorted_full"
    l = br.candidates[0]["line"]
    for x, y in a["points"]: assert abs(l["a"] * y + l["b"] - x) < 1e-6 and 0 <= x < W and 0 <= y < H


def test_stage_d_refines_the_detected_candidate_without_a_second_fitter():
    g, br = _result(); cal = make_cal(); ctx = V.build_contexts({"AL": g}, cal)["AL"]
    r = F.run_stage_d(br, ctx, cal, expected_set_index=0, expected_frame_index=133)
    assert r.refined_line is not None and r.status is not V.Status.FAIL
    assert abs(r.refined_line["a"] * 1000 + r.refined_line["b"] - 330) < 3
    assert F.V.process_annotation is V.process_annotation               # the Stage D function itself is used


def test_stage_d_rejects_wrong_frame_or_fingerprint_instead_of_substituting():
    g, br = _result(); cal = make_cal(); ctx = V.build_contexts({"AL": g}, cal)["AL"]
    bad = F.candidate_to_annotation(br); bad["frame_fingerprint"] = "0" * 64; r = V.process_annotation(bad, ctx, cal, expected_set_index=0, expected_frame_index=133)
    assert r.status is V.Status.FAIL and r.downstream_line is None
    r2 = V.process_annotation(F.candidate_to_annotation(br), ctx, cal, expected_set_index=0, expected_frame_index=999); assert r2.status is V.Status.FAIL


def test_refinement_failure_is_reported_not_hidden():
    g, br = _result(); cal = make_cal(); flat = np.full((H, W), 128, np.uint8); ctx = V.build_contexts({"AL": flat}, cal)["AL"]
    a = F.candidate_to_annotation(br); a.pop("frame_fingerprint", None); r = V.process_annotation(a, ctx, cal, expected_set_index=0, expected_frame_index=133)
    assert r.status is V.Status.FAIL and r.downstream_line is None


def test_candidate_to_annotation_refuses_not_detected():
    blank = np.full((H, W), 128, np.uint8); br = F.detect_set({"AL": blank}, make_cal(), set_index=0)["boundaries"]["AL_corner"]
    with pytest.raises(ValueError): F.candidate_to_annotation(br)


def test_annotation_meets_the_stage_e2_contract_but_e2_still_gates_warnings():
    g, br = _result(); a = F.candidate_to_annotation(br); assert VM.annotation_contract_problems(a) == []
    assert "allow_warnings_downstream" in V.AnnotationLimits.__dataclass_fields__
    # the E2 gate has no notion of an automatic source: a WARNING line without an explicit Confirmation stays blocked
    src = Path(VM.__file__).read_text(); assert "BLOCKED_UNCONFIRMED_WARNING" in src and "stage_f2" not in src


def test_f2_module_does_not_import_ground_truth_or_original_annotations():
    src = Path(F.__file__).read_text()
    for forbidden in ("wall_boundary_annotations", "validation", "expected_model_b", "independent_annotation_fixture", "stage_c_integration", "stage_e1_integration"): assert forbidden not in src.split('"""', 2)[2], forbidden


# ------------------------------------------------------------------ optional original-room regression (skipped when the local frames are absent)
@pytest.mark.skipif(not (LOCAL / "frames_full_index.json").exists(), reason="local Stage F1 frames not present")
def test_original_room_pl_corner_and_ceilF_detected_near_reference():
    idx = json.loads((LOCAL / "frames_full_index.json").read_text())["frames"]; cal = C.load_calibration()
    fr = {f["view_key"]: cv2.imread(str(LOCAL / f["file"]), cv2.IMREAD_GRAYSCALE) for f in idx if f["set_index"] == 0}
    res = F.detect_set(fr, cal, set_index=0)
    pl = res["boundaries"]["PL_corner"]; assert pl.status == "DETECTED"; assert abs(x_at(pl.candidates[0], 1500) - 684) < 20
    cf = res["boundaries"]["PL_ceilF"]; assert cf.status == "DETECTED"
    assert res["boundaries"]["AL_corner"].status == "UNCERTAIN"              # Air corner is capped without stereo corroboration

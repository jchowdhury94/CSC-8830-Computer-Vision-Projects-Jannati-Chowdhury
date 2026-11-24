"""Quality classification of the three-viewpoint reconstruction: boundaries, hard-vs-warning separation, aggregation.  Fast; metrics are hand-built, so no limit is tied to any recording."""
import copy
import math

import cv2
import numpy as np
import pytest

from room_measurement import three_view_quality as Q, three_view_wall_b as T
from room_measurement.tests import synthetic_three_view as S

TH = Q.DEFAULT_THRESHOLDS


def pose(**kw):
    d = dict(scale_px=0.35, inlier_fraction=0.9, positive_depth_fraction=0.999, n_matches=2800, converged=True, degenerate=False, fallback_used=False)
    d.update(kw); return d


def reg(**kw):
    d = dict(n_inliers=150, n_pairs=155, inlier_fraction=0.97, rms_mm=8.0, rot_deg=21.0, coverage_a=dict(cells=5), coverage_b=dict(cells=5), spread=dict(sd_mm=[300.0, 130.0, 85.0], planarity=0.28))
    d.update(kw); return d


def clean(length=3800.0):
    return dict(frame=60, length_mm=length, metrics=dict(
        pose={1: pose(), 2: pose(), 3: pose()}, reg_v1_v2=reg(), reg_v3_v2=reg(),
        floor={v: dict(floor_inl20=0.99, n_floor=2000) for v in (1, 2, 3)},
        floor_consistency=dict(ang_v1_v2_deg=0.3, ang_v3_v2_deg=0.3, ang_v1_v3_deg=0.6),
        corner=dict(vertical_edge_vs_floor_normal_deg=dict(v1=0.8, v3=0.9), b1_off_v2_floor_mm=5.0, b3_off_v2_floor_mm=-3.0, b1_off_v3_floor_mm=10.0, b3_off_v1_floor_mm=4.0, length_vertical_component_mm=20.0)))


def codes(q, sev): return {f["code"] for f in q["warnings" if sev == "w" else "hard_failures"]}


def test_clean_frame_has_no_findings():
    q = Q.classify_frame(clean()); assert q["status"] == Q.OK and not q["warnings"] and not q["hard_failures"]


def setter(path):
    def f(r, v):
        o = r["metrics"]
        for k in path[:-1]: o = o[k]
        o[path[-1]] = v
    return f


# (name, setter, upper?, hard, warn, code)
CASES = [
    ("pose scale", setter(("pose", 1, "scale_px")), True, 1.0, 0.6, "POSE_RESIDUAL"),
    ("pose inlier fraction", setter(("pose", 2, "inlier_fraction")), False, 0.5, 0.75, "POSE_INLIERS"),
    ("positive depth", setter(("pose", 3, "positive_depth_fraction")), False, 0.95, 0.99, "POSE_DEPTH"),
    ("pose matches", setter(("pose", 1, "n_matches")), False, 300, 1000, "POSE_FEW_MATCHES"),
    ("reg inliers", setter(("reg_v1_v2", "n_inliers")), False, 20, 80, "REG_FEW_INLIERS"),
    ("reg inlier fraction", setter(("reg_v3_v2", "inlier_fraction")), False, 0.5, 0.8, "REG_INLIER_FRACTION"),
    ("reg rms", setter(("reg_v1_v2", "rms_mm")), True, 30.0, 20.0, "REG_RMS"),
    ("floor inliers", setter(("floor", 2, "floor_inl20")), False, 0.5, 0.9, "FLOOR_FIT"),
    ("floor points", setter(("floor", 3, "n_floor")), False, 200, 500, "FLOOR_FEW_POINTS"),
    ("floor angle v1-v2", setter(("floor_consistency", "ang_v1_v2_deg")), True, 3.0, 1.0, "FLOOR_ANGLE"),
    ("floor angle v3-v2", setter(("floor_consistency", "ang_v3_v2_deg")), True, 3.0, 1.0, "FLOOR_ANGLE"),
    ("floor angle v1-v3", setter(("floor_consistency", "ang_v1_v3_deg")), True, 4.0, 2.0, "FLOOR_ANGLE"),
    ("corner direction v3", setter(("corner", "vertical_edge_vs_floor_normal_deg", "v3")), True, 10.0, 3.0, "CORNER_DIRECTION"),
    ("corner off floor", setter(("corner", "b1_off_v3_floor_mm")), True, 150.0, 40.0, "CORNER_OFF_FLOOR"),
]


@pytest.mark.parametrize("name,put,upper,hard,warn,code", CASES, ids=[c[0] for c in CASES])
def test_boundaries_and_severity(name, put, upper, hard, warn, code):
    eps = 1e-6 * max(1.0, abs(hard)); sgn = 1 if upper else -1
    def classify(v):
        r = clean(); put(r, v); return Q.classify_frame(r)
    q = classify(warn); assert code not in codes(q, "w") | codes(q, "h"), "a value exactly at the warning limit must not fire"
    q = classify(warn + sgn * eps); assert code in codes(q, "w") and code not in codes(q, "h") and q["status"] == Q.WARNINGS
    q = classify(hard); assert code in codes(q, "w") and code not in codes(q, "h"), "exactly at the hard limit is still only a warning"
    q = classify(hard + sgn * eps); assert code in codes(q, "h") and q["status"] == Q.FAILED
    q = classify(float("nan")); assert code in codes(q, "h"), "a non-finite metric is a hard failure"


def test_every_finding_carries_scope_message_and_corrective_action():
    r = clean(); r["metrics"]["corner"]["vertical_edge_vs_floor_normal_deg"]["v3"] = 5.29
    f = Q.classify_frame(r)["warnings"][0]
    assert f["scope"] == "V3" and f["limit"] == 3.0 and "5.29" in f["message"] and f["action"]


def test_pose_degenerate_or_unconverged_is_a_hard_failure():
    for k, v in (("degenerate", True), ("converged", False)):
        r = clean(); r["metrics"]["pose"][2][k] = v; q = Q.classify_frame(r)
        assert "POSE_DEGENERATE" in codes(q, "h") and q["status"] == Q.FAILED
    r = clean(); r["metrics"]["pose"][1]["fallback_used"] = True; assert "POSE_FALLBACK_START" in codes(Q.classify_frame(r), "w")


def test_collinear_registration_inliers_fail_but_planar_ones_only_warn():
    r = clean(); r["metrics"]["reg_v1_v2"]["spread"]["sd_mm"] = [400.0, 5.0, 1.0]
    assert "REG_COLLINEAR" in codes(Q.classify_frame(r), "h")
    r = clean(); r["metrics"]["reg_v1_v2"]["spread"]["sd_mm"] = [400.0, 200.0, 5.0]
    q = Q.classify_frame(r); assert "REG_PLANAR" in codes(q, "w") and not q["hard_failures"]


def test_spatial_coverage_is_warning_only_even_when_extreme():
    for cells in (2, 1, 0):
        r = clean(); r["metrics"]["reg_v1_v2"]["coverage_a"]["cells"] = cells; q = Q.classify_frame(r)
        assert "REG_COVERAGE" in codes(q, "w") and not q["hard_failures"] and q["status"] == Q.WARNINGS
    r = clean(); r["metrics"]["reg_v1_v2"]["coverage_a"]["cells"] = 3; assert not Q.classify_frame(r)["warnings"]


def test_length_range_is_a_configurable_hard_check():
    assert "LENGTH_OUT_OF_RANGE" in codes(Q.classify_frame(clean(400.0)), "h")
    assert "LENGTH_OUT_OF_RANGE" in codes(Q.classify_frame(clean(31000.0)), "h")
    for edge in (500.0, 30000.0):                                              # the limits themselves are inside the range
        r = clean(edge); r["metrics"]["corner"]["length_vertical_component_mm"] = 0.001 * edge; assert Q.classify_frame(r)["status"] == Q.OK
    off = Q.QualityThresholds(length_range_mm=None); r = clean(400.0); r["metrics"]["corner"]["length_vertical_component_mm"] = 0.4; assert Q.classify_frame(r, off)["status"] == Q.OK
    custom = Q.QualityThresholds(length_range_mm=(2000.0, 5000.0)); assert "LENGTH_OUT_OF_RANGE" in codes(Q.classify_frame(clean(1500.0), custom), "h")
    for bad in (float("nan"), 0.0, -5.0, None): assert "LENGTH_INVALID" in codes(Q.classify_frame(clean(bad)), "h")


def test_corner_to_corner_vector_must_be_horizontal():
    r = clean(1000.0); r["metrics"]["corner"]["length_vertical_component_mm"] = 40.0; assert "LENGTH_NOT_HORIZONTAL" in codes(Q.classify_frame(r), "w")
    r["metrics"]["corner"]["length_vertical_component_mm"] = 110.0; assert "LENGTH_NOT_HORIZONTAL" in codes(Q.classify_frame(r), "h")


# ------------------------------------------------------------------ aggregation
def classified(length, **mut):
    r = clean(length)
    for k, v in mut.items(): r["metrics"]["floor_consistency"][k] = v
    r["quality"] = Q.classify_frame(r); return r


def test_aggregate_without_findings_is_measured_and_a_plain_mean():
    a = Q.aggregate([classified(3790.0), classified(3800.0), classified(3810.0)])
    assert a["status"] == Q.MEASURED and a["length_mm"] == pytest.approx(3800.0) and a["frames_usable"] == 3 and not a["frame_warnings"]


def test_warnings_are_kept_with_a_numerical_result():
    a = Q.aggregate([classified(3790.0, ang_v1_v3_deg=2.5), classified(3800.0), classified(3810.0)])
    assert a["status"] == Q.MEASURED_WITH_WARNINGS and a["length_mm"] == pytest.approx(3800.0) and a["frame_warnings"][0]["frame"] == 60 and a["frame_warnings"][0]["code"] == "FLOOR_ANGLE"


def test_a_failed_frame_is_excluded_and_reported_when_two_frames_remain():
    bad = classified(5000.0, ang_v1_v2_deg=9.0); assert bad["quality"]["status"] == Q.FAILED
    a = Q.aggregate([classified(3790.0), bad, classified(3810.0)])
    assert a["status"] == Q.MEASURED_WITH_WARNINGS and a["length_mm"] == pytest.approx(3800.0) and a["frames_failed"] == [60] and a["frame_failures"]


def test_fewer_than_two_usable_frames_is_insufficient_evidence_with_no_length():
    bad = lambda: classified(5000.0, ang_v1_v2_deg=9.0)
    a = Q.aggregate([classified(3790.0), bad(), bad()])
    assert a["status"] == Q.INSUFFICIENT_EVIDENCE and a["length_mm"] is None and a["length_spread_mm"] is None and "at least 2" in a["message"]
    assert Q.aggregate([classified(3800.0)])["status"] == Q.INSUFFICIENT_EVIDENCE
    assert Q.aggregate([classified(3800.0)], Q.QualityThresholds(min_usable_frames=1))["status"] == Q.MEASURED


def test_frame_spread_is_a_warning_never_a_failure():
    a = Q.aggregate([classified(3600.0), classified(3800.0), classified(3900.0)])         # range 300 mm = 7.9 %
    assert a["status"] == Q.MEASURED_WITH_WARNINGS and a["length_mm"] == pytest.approx(np.mean([3600.0, 3800.0, 3900.0])) and a["aggregate_warnings"][0]["code"] == "FRAME_SPREAD"
    assert not Q.aggregate([classified(3790.0), classified(3810.0), classified(3800.0)])["aggregate_warnings"]      # 0.5 %


# ------------------------------------------------------------------ the reconstruction wiring (synthetic scene)
def three_frame_scene(bad_frames=()):
    cal, ms, doc, L = S.build()
    extra = [dict(a, frame_index=f) for a in doc["annotations"] for f in (61, 62)]
    doc["annotations"] += extra; by = {}
    for f in (60, 61, 62):
        by[f] = copy.deepcopy(ms)
        if f in bad_frames: by[f][(1, 2)]["MX"] = by[f][(1, 2)]["MX"][:3]
    return cal, by, T.parse_annotations(doc, cal.image_size), L


def test_reconstruct_classifies_each_frame_and_aggregates():
    cal, by, ann, L = three_frame_scene(); o = T.reconstruct(cal, by, ann)
    assert [r["frame"] for r in o["per_frame"]] == [60, 61, 62] and all("quality" in r for r in o["per_frame"])
    assert not any(r["quality"]["hard_failures"] for r in o["per_frame"]), [r["quality"]["hard_failures"] for r in o["per_frame"]]
    assert o["length_mm"] == pytest.approx(L, rel=0.01) and o["quality"]["status"] in (Q.MEASURED, Q.MEASURED_WITH_WARNINGS)


def test_a_frame_that_cannot_be_registered_is_reported_not_raised():
    cal, by, ann, L = three_frame_scene(bad_frames=(62,)); o = T.reconstruct(cal, by, ann)
    bad = o["per_frame"][2]; assert bad["quality"]["status"] == Q.FAILED and bad["length_mm"] is None and bad["quality"]["hard_failures"][0]["code"] == "REG_PAIRS"
    assert o["quality"]["frames_usable"] == 2 and o["length_mm"] == pytest.approx(L, rel=0.01) and o["quality"]["status"] == Q.MEASURED_WITH_WARNINGS


def test_two_unusable_frames_give_no_length():
    cal, by, ann, _ = three_frame_scene(bad_frames=(61, 62)); o = T.reconstruct(cal, by, ann)
    assert o["quality"]["status"] == Q.INSUFFICIENT_EVIDENCE and o["length_mm"] is None


def test_missing_annotations_still_raise_instead_of_hiding_behind_a_frame_failure():
    cal, by, ann, _ = three_frame_scene(); by[99] = by[60]
    with pytest.raises(T.ThreeViewError) as e: T.reconstruct(cal, by, ann)
    assert e.value.code == "ANNOTATION_MISSING"


# ------------------------------------------------------------------ the match-sharing optimisation changes nothing
def _textured(seed, shift):
    rng = np.random.default_rng(seed); base = cv2.GaussianBlur(rng.integers(0, 255, (420, 520), dtype=np.uint8), (0, 0), 2.2)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX); return np.ascontiguousarray(np.roll(base, shift, axis=1)[10:410, 10:510])


def test_shared_detection_and_subset_mutual_check_reproduce_features_match_set_exactly():
    grays = {(v, s): _textured(7, sh) for v, sh0 in ((1, 0), (2, 6), (3, 12)) for s, sh in (("L", sh0), ("R", sh0 + 3))}
    ref = T.compute_match_sets(grays, 0, T.ThreeViewConfig(nfeatures=1500, contrast=0.02))
    assert len(ref[(1, 2)]["M1"]) > 20 and len(ref[(1, 2)]["MX"]) > 20          # the test images do produce matches
    off = dict(share_detections=False, detect_only_sharing=False, subset_mutual_check=False)
    for kw in (dict(share_detections=True), dict(share_detections=True, detect_only_sharing=True), dict(share_detections=True, subset_mutual_check=True), dict(subset_mutual_check=True), {}):
        cfg = T.ThreeViewConfig(nfeatures=1500, contrast=0.02, **{**off, **kw}) if kw else T.ThreeViewConfig(nfeatures=1500, contrast=0.02); got = T.compute_match_sets_shared(grays, 0, cfg)
        for pair in ref:
            for k in ref[pair]: assert np.array_equal(ref[pair][k], got[pair][k]), (kw, pair, k)


def test_fast_matching_is_the_default_and_the_original_matcher_stays_available():
    d = T.ThreeViewConfig(); assert d.share_detections and d.subset_mutual_check and d.uses_fast_matcher()
    o = T.ThreeViewConfig.original_matcher(nfeatures=1500); assert not o.uses_fast_matcher() and o.nfeatures == 1500

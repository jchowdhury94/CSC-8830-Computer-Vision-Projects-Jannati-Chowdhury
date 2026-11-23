"""F8.31B: visitor frame selection and wall-boundary annotations.  SYNTHETIC fixtures only: tiny generated clips of a drawn wall corner / ceiling edge.
No original recording is opened or stat-ed, nothing is reconstructed, Method J is never run."""
import copy
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import pytest

import f8_visitor_service as S
import frame_worker as FW
import room_frames as RF
import room_videos as RV
from calibration_runner import RunnerLimits
from chunk_store import ChunkStore, Limits, MiB
from result_store import ResultStore
from room_measurement import visitor_annotations as V
from room_measurement.visitor_measurement import JobLimits
from test_f8_31a_room_videos import SLOTS, good_info, put
from test_f8_visitor_service import make_doc
from video_check import inspect_video

W, H, N_FRAMES, FPS = 1920, 1080, 36, 30.0
PEAK = {"p1_left": 14, "p1_right": 18, "p2_left": 12, "p2_right": 22}           # the sharpest frame of each synthetic video
HERE = Path(__file__).resolve().parents[1]


# ================================================================ synthetic clips: a drawn wall corner (vertical edge) and ceiling edge, blur varies per frame
def corner_x(y): return 900.0 + 0.03 * (y - 540.0)
def ceil_y(x): return 250.0 + 0.02 * (x - 960.0)


def scene(shift):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = np.clip(0.5 - (yy - ceil_y(xx)), 0, 1); b = np.clip(0.5 + (xx - corner_x(yy)), 0, 1)
    g = a * 230 + (1 - a) * (170 * (1 - b) + 60 * b) + shift
    return np.clip(g, 0, 255).astype(np.uint8)


def make_clip(path, slot, k):
    base = scene(k); w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N_FRAMES):
        d = abs(i - PEAK[slot]); g = base if d == 0 else cv2.GaussianBlur(base, (0, 0), 0.6 + 0.25 * d)
        w.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    w.release()
    return Path(path)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips"); out = {s: make_clip(d / f"{s}.mp4", s, k) for k, s in enumerate(SLOTS)}
    return out, {p.stat().st_size: dict(inspect_video(p), size=p.stat().st_size) for p in out.values()}


def sequential_gray(path, index):
    """Reference decode with the Phase 1 convention (sequential read, BGR2GRAY)."""
    cap = cv2.VideoCapture(str(path)); pos = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if pos == index:
            cap.release(); return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        pos += 1
    raise AssertionError("frame not found")


@pytest.fixture
def rstore(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB, reserve_sibling=False, max_active_uploads=4), root=tmp_path / "rooms", slots=SLOTS); yield s; s.close()


@pytest.fixture
def cstore(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB), root=tmp_path / "chunks"); yield s; s.close()


@pytest.fixture
def svc(cstore, rstore, tmp_path, clips):
    table = clips[1]
    sv = S.VisitorCalibrationService(cstore, ResultStore(tmp_path / "results"), RunnerLimits(), room_store=rstore, inspector=lambda p: table[Path(p).stat().st_size],
                                     frame_workspace_root=tmp_path)
    sv._test_sessions = []
    yield sv
    for sess in sv._test_sessions:                                                   # F6 jobs hold a class-level slot until closed: never leak one between tests
        sess.reset(cstore, rstore)


def accepted(svc, doc=None):
    s = S.VisitorSession(); getattr(svc, "_test_sessions", []).append(s); s.review = S.review_f8_document(doc or make_doc("GOOD")[1]); svc.accept(s); return s


def upload_all(svc, sess, clips, only=None):
    for slot, p in clips[0].items():
        if only is None or slot in only:
            put(svc.room_store, sess.owner, slot, p.read_bytes(), name=p.name)


@pytest.fixture
def ready(svc, clips):
    s = accepted(svc); upload_all(svc, s, clips); st = svc.frame_state(s)
    assert st["state"] == "ok", st
    return s, st["workflow"]


PAIRS = {1: [(10, 14), (20, 27)], 2: [(8, 6), (24, 30)]}          # (left, right) per set: differing offsets, as in the original data


def choose_and_confirm(wf, sets=2, pairs=PAIRS):
    wf.set_n_sets(sets)
    for vp in RF.VIEWPOINTS:
        for i in range(sets):
            l, r = pairs[vp][i]; wf.set_pair(vp, i, l, r); wf.preview(f"p{vp}_left", l); wf.preview(f"p{vp}_right", r); wf.confirm_pair(vp, i)
    return wf.extract_confirmed()


LINE_POINTS = {"corner": [(corner_x(350), 350.0), (corner_x(950), 950.0)], "ceilF": [(1050.0, ceil_y(1050)), (1650.0, ceil_y(1650))]}


def annotate_all(wf, sets=2):
    for i in range(sets):
        for line in RF.REQUIRED_LINES:
            wf.set_points(i, line, LINE_POINTS[line.split("_")[1]]); res = wf.refine(i, line)
            assert res.status is not V.Status.FAIL, (i, line, res.failure_codes, [m.message for m in res.issues])
            wf.accept(i, line, acknowledged=True)


@pytest.fixture
def complete(ready):
    s, wf = ready; choose_and_confirm(wf); annotate_all(wf); assert wf.readiness(deep=True).ready, wf.readiness().to_dict(); return s, wf


# ================================================================ audit of the original input representation
def test_required_lines_are_six_not_eight_and_ceils_is_not_requested():
    assert RF.REQUIRED_LINES == ("AL_corner", "PL_corner", "PL_ceilF", "AR_corner", "PR_corner", "PR_ceilF")
    assert set(RF.NOT_REQUESTED_LINES) == {"PL_ceilS", "PR_ceilS"} and not set(RF.NOT_REQUESTED_LINES) & set(RF.REQUIRED_LINES)
    from room_measurement import visitor_method_j_adapter as A
    assert tuple(A.REQUIRED_LINES) == RF.REQUIRED_LINES                    # exactly what the adapter reads (measure_wall, top_variant 0)
    src = (HERE.parent / "room_measurement" / "measure.py").read_text()
    assert "top_variant == 0" in src and "lines[cs]" in src                 # ceilS is read only by the other variant


def test_original_manifest_pairs_are_independent_per_video():
    """The shipped manifest is configuration, not a video: five sets, independent index per video, right-minus-left differs from set to set."""
    m = json.loads((HERE.parent / "room_measurement" / "config" / "frame_manifest.json").read_text())
    a, p = m["videos"]["AL"]["frame_indices_per_set"], m["videos"]["PL"]["frame_indices_per_set"]
    assert len(a) == RF.DEFAULT_SETS == 5 and len({q - r for q, r in zip(p, a)}) > 1


# ================================================================ 1. extraction from all four slots (full resolution, nothing resized)
def test_extract_all_four_slots_is_byte_identical_to_sequential_decoding(ready, clips):
    _, wf = ready; assert choose_and_confirm(wf) == 8
    for vp in RF.VIEWPOINTS:
        for i, (l, r) in enumerate(PAIRS[vp]):
            for side, idx in (("left", l), ("right", r)):
                slot, vk = f"p{vp}_{side}", RF.VIEW_KEY[(vp, side)]; g = wf.frame_gray(i, vk)
                assert g.shape == (H, W) and g.dtype == np.uint8                                  # original size, orientation and sample type
                assert np.array_equal(g, sequential_gray(clips[0][slot], idx)), (slot, idx)        # not resized, cropped, rotated or resampled
                assert wf.full[(i, vk)]["index"] == idx and wf.full[(i, vk)]["wh"] == (W, H)


def test_wrong_resolution_frame_is_refused_not_resized(ready, tmp_path):
    _, wf = ready; wf.cal = type("C", (), dict(image_size=(W // 2, H // 2), K_left=wf.cal.K_left, D_left=wf.cal.D_left, K_right=wf.cal.K_right, D_right=wf.cal.D_right, baseline_mm=226.0))()
    wf.set_n_sets(2); wf.set_pair(1, 0, 10, 14); wf.preview("p1_left", 10); wf.preview("p1_right", 14); wf.confirm_pair(1, 0)
    with pytest.raises(RF.WorkflowError, match="not rescaled"):
        wf.extract_confirmed()
    assert not wf.full


def test_extraction_only_happens_for_confirmed_pairs(ready):
    _, wf = ready; wf.set_pair(1, 0, 10, 14)
    assert wf.extract_confirmed() == 0 and not wf.full


# ================================================================ 2. stereo frame offsets
def test_offsets_are_explicit_frames_and_nominal_seconds():
    assert RF.offset_frames(247, 292) == 45 and RF.offset_frames(30, 20) == -10
    assert RF.offset_seconds(30, 33, 30.0, 30.0) == pytest.approx(0.1)
    assert RF.offset_seconds(30, 60, 30.0, 60.0) == pytest.approx(0.0)           # each index uses ITS OWN video's frame rate


def test_right_index_from_offset_never_clamps():
    assert RF.right_index_from_offset(10, 4, 36) == 14 and RF.right_index_from_offset(10, -10, 36) == 0
    for left, off in ((10, 40), (3, -4), (35, 1)):
        with pytest.raises(RF.OffsetError):
            RF.right_index_from_offset(left, off, 36)


def test_pair_by_offset_and_pair_info(ready):
    _, wf = ready; wf.set_n_sets(2)
    wf.set_pair_by_offset(1, 0, 10, 4); info = wf.pair_info(1, 0)
    assert (info["left"], info["right"], info["offset_frames"]) == (10, 14, 4) and info["offset_seconds"] == pytest.approx(4 / FPS) and not info["confirmed"]
    with pytest.raises(RF.OffsetError):
        wf.set_pair_by_offset(1, 0, 30, 20)
    assert wf.pair_info(1, 0)["right"] == 14                                      # a refused offset changes nothing
    wf.set_pair(1, 1, 20, 24)
    assert wf.viewpoint_offset(1) == 4
    wf.set_pair(1, 1, 20, 27)
    assert wf.viewpoint_offset(1) is None                                         # as in the original data: offsets differ from set to set


# ================================================================ 3. pair selection, confirmation and replacement
def test_confirmation_requires_both_previews(ready):
    _, wf = ready; wf.set_n_sets(2); wf.set_pair(1, 0, 10, 14)
    with pytest.raises(RF.WorkflowError, match="Preview both"):
        wf.confirm_pair(1, 0)
    wf.preview("p1_left", 10)
    with pytest.raises(RF.WorkflowError, match="Preview both"):
        wf.confirm_pair(1, 0)
    wf.preview("p1_right", 14); wf.confirm_pair(1, 0); assert wf.pair_info(1, 0)["confirmed"]


def test_out_of_range_and_bad_frame_numbers_are_refused(ready):
    _, wf = ready; wf.set_n_sets(2)
    for bad in (-1, N_FRAMES, 10_000):
        with pytest.raises(RF.WorkflowError, match="does not exist"):
            wf.set_pair(1, 0, bad, 5)
    for bad in (True, 1.5, "3", None):
        with pytest.raises(RF.WorkflowError):
            wf.set_pair(1, 0, bad, 5)
    with pytest.raises(RF.WorkflowError):
        wf.set_pair(1, 9, 1, 1)


def test_replacing_a_frame_drops_only_what_depended_on_it(ready):
    _, wf = ready; choose_and_confirm(wf); annotate_all(wf)
    keep_right = dict(wf.full[(0, "PL")]); keep_other = dict(wf.full[(1, "AL")])
    wf.set_pair(1, 0, 11, 14)                                                      # only the LEFT frame of viewpoint 1, set 0 changes
    assert not wf.pair_info(1, 0)["confirmed"] and (0, "AL") not in wf.full
    assert wf.full[(0, "PL")] == keep_right and wf.full[(1, "AL")] == keep_other   # the right frame and other sets are untouched
    assert (0, "AL_corner") not in wf.ann and (0, "PL_corner") in wf.ann and (0, "PL_ceilF") in wf.ann and (1, "AL_corner") not in wf.ann   # F8.31D-6: set 0 carries the seed, so replacing its frame also drops the sets derived from it
    assert not wf.readiness().ready and Path(keep_right["path"]).is_file()         # sharing a job directory must not delete the surviving frames


def test_n_sets_default_is_the_original_five_and_is_bounded(ready):
    _, wf = ready
    assert wf.n_sets == 5 == RF.DEFAULT_SETS
    for bad in (1, 9):
        with pytest.raises(RF.WorkflowError):
            wf.set_n_sets(bad)
    wf.set_n_sets(3); assert wf.n_sets == 3


def test_original_rule_indices():
    s = [1.0] * 100; s[40] = 9.0; s[22] = 5.0; s[58] = 5.0; s[81] = 5.0; s[19] = 7.0
    idx = RF.original_rule_indices(s, 5)
    assert idx[0] == 40 and idx[1] == 19
    assert len(idx) == 5 and len(set(idx)) == 5                                    # never the same frame twice
    for k, i in enumerate(idx[1:], start=1):
        assert abs(i - round(k / 5 * 100)) <= RF.ORIGINAL_WINDOW_FRAMES
    with pytest.raises(RF.WorkflowError):
        RF.original_rule_indices([1.0, 2.0], 5)


def test_propose_original_rule_with_the_sandboxed_scan(ready):
    _, wf = ready; wf.set_n_sets(3)
    sharp = wf.scan_sharpness("p1_left"); assert len(sharp) == N_FRAMES
    assert int(np.argmax(sharp)) == PEAK["p1_left"]
    pairs = wf.propose_original_rule(1)
    assert pairs[0]["left"] == PEAK["p1_left"] and pairs[0]["right"] == PEAK["p1_right"]        # set 0 = the sharpest frame of EACH video
    assert all(not p["confirmed"] for p in pairs) and len({p["left"] for p in pairs}) == 3
    assert not list(Path(wf.ws.path).glob("job_*"))                                  # the scan left no output behind


# ================================================================ 4. coordinates
def test_preview_to_full_round_trip_and_known_points():
    fw, pw = (1920, 1080), (900, 506)
    full = np.array([[0.0, 0.0], [959.5, 540.0], [1919.0, 1079.0], [123.25, 877.75]])
    assert np.allclose(RF.preview_to_full(RF.full_to_preview(full, fw, pw), fw, pw), full, atol=1e-9)
    assert np.allclose(RF.preview_to_full([[450.0, 253.0]], fw, pw), [[450.0 * 1920 / 900 - 0.5, 253.0 * 1080 / 506 - 0.5]], atol=1e-9)
    assert np.allclose(RF.preview_to_full([[0.0, 0.0]], fw, pw), [[-0.5, -0.5]])        # edge-aligned preview origin = the corner of the first pixel


def test_a_stretched_preview_is_refused():
    with pytest.raises(RF.WorkflowError, match="uniform"):
        RF.preview_to_full([[10.0, 10.0]], (1920, 1080), (900, 600))


def test_worker_preview_is_a_uniform_reduction_and_keeps_the_original_size(ready):
    _, wf = ready; p = wf.preview("p1_left", 10)
    assert p["full_wh"] == (W, H) and max(p["wh"]) <= wf.job_limits.preview_long_side and not RF.preview_mapping(p["full_wh"], p["wh"]).validate()
    g = cv2.imread(p["path"], cv2.IMREAD_GRAYSCALE); assert g.shape == (p["wh"][1], p["wh"][0])
    assert wf.source("p1_left").wh == (W, H)                                       # the original frame size is unchanged by previewing


def test_annotation_points_are_stored_in_original_coordinates(ready):
    _, wf = ready; choose_and_confirm(wf); p = wf.preview("p1_left", 10)
    click_on_preview = RF.full_to_preview([corner_x(350), 350.0], (W, H), p["wh"])[0]
    full_pt = RF.preview_to_full([click_on_preview], (W, H), p["wh"])[0]
    wf.click(0, "AL_corner", *full_pt); wf.click(0, "AL_corner", *RF.preview_to_full(RF.full_to_preview([corner_x(950), 950.0], (W, H), p["wh"]), (W, H), p["wh"])[0])
    pts = wf.state(0, "AL_corner").points
    assert np.allclose(pts, [[corner_x(350), 350.0], [corner_x(950), 950.0]], atol=1e-6) and all(0 <= x <= W - 1 and 0 <= y <= H - 1 for x, y in pts)
    res = wf.refine(0, "AL_corner")
    assert res.metadata["coordinate_space"] == "distorted_full" and res.metadata["image_wh"] == [W, H] and res.points_original == pts


# ================================================================ 5. annotations: create, edit, persist
def test_annotation_create_edit_refine_accept(ready):
    _, wf = ready; choose_and_confirm(wf)
    st = wf.state(0, "PL_ceilF")
    assert wf.annotation_status(0, "PL_ceilF") == "not_started"
    wf.click(0, "PL_ceilF", 1050.0, ceil_y(1050) + 4); assert wf.annotation_status(0, "PL_ceilF") == "placing_points"
    wf.click(0, "PL_ceilF", 1650.0, ceil_y(1650) - 3); assert wf.annotation_status(0, "PL_ceilF") == "points_placed"
    wf.begin_edit(0, "PL_ceilF", 0); wf.click(0, "PL_ceilF", 1050.0, ceil_y(1050)); assert st.points[0] == [1050.0, ceil_y(1050)] and st.edit_index is None
    wf.click(0, "PL_ceilF", 1640.0, ceil_y(1640))                                   # a third click moves the NEAREST point, it never adds one
    assert len(st.points) == 2 and st.points[1][0] == 1640.0
    res = wf.refine(0, "PL_ceilF"); assert res.status is not V.Status.FAIL and wf.annotation_status(0, "PL_ceilF") == "needs_review"
    line = res.refined_line; assert abs(line["a"] - 0.02) < 5e-3                    # the refined line follows the drawn edge (the unchanged Method J fit)
    wf.accept(0, "PL_ceilF", acknowledged=True); assert wf.annotation_status(0, "PL_ceilF") == "accepted"
    wf.click(0, "PL_ceilF", 1500.0, ceil_y(1500))                                   # editing after acceptance invalidates it
    assert not st.accepted and st.result is None and wf.annotation_status(0, "PL_ceilF") == "points_placed" 


def test_a_boundary_not_needed_by_method_j_is_refused(ready):
    _, wf = ready
    for bad in ("PL_ceilS", "PR_ceilS", "AL_ceilF", "nonsense"):
        with pytest.raises(RF.WorkflowError, match="not needed"):
            wf.state(0, bad)


def test_points_outside_the_frame_and_bad_numbers_are_refused(ready):
    _, wf = ready; choose_and_confirm(wf)
    for bad in ((-1.0, 5.0), (W, 5.0), (5.0, H), (float("nan"), 1.0), (float("inf"), 1.0)):
        with pytest.raises(RF.WorkflowError):
            wf.click(0, "AL_corner", *bad)
    assert wf.state(0, "AL_corner").points == []


def test_accept_rules(ready):
    _, wf = ready; choose_and_confirm(wf)
    with pytest.raises(RF.WorkflowError, match="Refine"):
        wf.accept(0, "AL_corner")
    wf.set_points(0, "AL_corner", [(100.0, 100.0), (101.0, 101.0)])               # too short: Stage D FAILs it
    assert wf.refine(0, "AL_corner").status is V.Status.FAIL
    with pytest.raises(RF.WorkflowError, match="cannot be used"):
        wf.accept(0, "AL_corner", acknowledged=True)
    assert wf.annotation_status(0, "AL_corner") == "refine_failed"


def test_refine_needs_a_confirmed_pair(ready):
    _, wf = ready; wf.set_points(0, "AL_corner", LINE_POINTS["corner"])
    with pytest.raises(RF.WorkflowError, match="Confirm this set"):
        wf.refine(0, "AL_corner")


def test_copy_and_propagate_seed_produce_proposals_never_acceptances(ready):
    _, wf = ready; choose_and_confirm(wf)
    wf.set_points(0, "AL_corner", LINE_POINTS["corner"]); wf.refine(0, "AL_corner"); wf.accept(0, "AL_corner", acknowledged=True)
    out = wf.propagate("AL_corner", 0)
    assert set(out) == {1} and out[1] in ("PASS", "WARNING") and wf.annotation_status(1, "AL_corner") == "needs_review"
    assert wf.annotation_status(0, "AL_corner") == "accepted"
    with pytest.raises(RF.WorkflowError):
        wf.copy_points(1, 0, "PL_corner")                                           # nothing to copy from


def test_annotations_persist_across_frame_state_calls(svc, complete):
    s, wf = complete
    for _ in range(3):
        st = svc.frame_state(s); assert st["state"] == "ok" and st["workflow"] is wf
    assert wf.readiness(deep=True).ready and all(wf.annotation_status(i, n) == "accepted" for i in range(2) for n in RF.REQUIRED_LINES)
    exp = wf.export_state(); assert exp["ready"] and len(exp["annotations"]) == 12 and exp["coordinate_space"] == "distorted_full"
    json.dumps(exp)                                                                  # JSON-safe: no arrays, no paths
    assert str(wf.ws.path) not in json.dumps(exp)


# ================================================================ 6. schema compatibility with the Method J adapter
def test_lines_match_what_the_adapter_and_method_j_consume(complete):
    from room_measurement import lines as L, visitor_method_j_adapter as A, visitor_method_j_adapter_synthetic as SYN
    s, wf = complete; sets = wf.lines_for_adapter()
    assert len(sets) == 2
    fit_keys = {"kind", "a", "b", "n", "N", "rms", "span", "pts"}                      # exactly what lines.fit_line returns
    ref = L.fit_line(np.c_[np.linspace(10, 500, 40), 0.1 * np.linspace(10, 500, 40) + 3], "h")
    assert set(ref) == fit_keys
    for ls in sets:
        assert set(ls) == set(RF.REQUIRED_LINES)
        for n, l in ls.items():
            assert set(l) == fit_keys and l["kind"] == ("v" if n.endswith("corner") else "h") and np.asarray(l["pts"]).shape[1] == 2 and l["n"] >= 8
    ctx = A.VisitorCalibrationContext.from_params(dict(K_left=wf.contract["K1"], D_left=wf.contract["D1"], K_right=wf.contract["K2"], D_right=wf.contract["D2"], baseline_mm=wf.cal.baseline_mm),
                                                  wf.cal.image_size)
    scene_inputs, _ = SYN.make_scene(n_sets=2, n_pts=900)
    inputs = A.ReconstructionInputs(tuple(scene_inputs.match_sets), tuple(sets), ctx)          # the adapter's own structural check accepts our line sets
    assert inputs.problems(ctx) == []
    sets[0].pop("AR_corner"); bad = A.ReconstructionInputs(tuple(scene_inputs.match_sets), tuple(sets), ctx)
    assert any("AR_corner" in p["message"] for p in bad.problems(ctx))


def test_lines_for_adapter_refuses_incomplete_work(ready):
    _, wf = ready; choose_and_confirm(wf)
    with pytest.raises(RF.WorkflowError, match="not complete"):
        wf.lines_for_adapter()


def test_annotation_payload_passes_the_stage_d_schema(complete):
    _, wf = complete; st = wf.state(1, "PR_ceilF"); f = wf.full[(1, "PR")]
    parsed, issues = V.parse_annotation(RF.res_raw(st, 1, "PR_ceilF", f, wf.cal))
    assert parsed is not None and not [i for i in issues if i.status is V.Status.FAIL] and parsed.line_name == "PR_ceilF" and parsed.view_key == "PR"


# ================================================================ 7. calibration ownership
def test_selections_stay_bound_to_the_accepted_calibration(complete):
    s, wf = complete; rec = wf.calibration_record(); c = s.contract
    assert rec["K_left"] == c["K1"] and rec["K_right"] == c["K2"] and rec["D_left"] == c["D1"] and rec["D_right"] == c["D2"]
    assert rec["image_size_wh"] == [W, H] == c["image_size_wh"] and rec["baseline_mm"] == 226.0 == s.accepted.baseline_used_mm            # the visitor's measured value
    assert rec["camera_order"] == ["left", "right"] and "k1, k2, p1, p2, k3" in rec["distortion_model"] and rec["calibration_id"] == RF.calibration_id(c)
    assert rec["capture_mode"] == dict(left="video", right="video")
    assert rec["baseline_mm"] != 225.0                                              # never the owner's baseline


def test_no_owner_or_recipe_a_value_is_a_default():
    for f in (HERE / "room_frames.py", HERE / "frame_worker.py"):
        src = f.read_text()
        assert not re.search(r"\b225(\.0)?\b", src) and "room_video_recipe_a" not in src and "DEFAULT_CALIBRATION_PATH" not in src and "load_calibration" not in src


def test_the_calibration_id_changes_with_every_calibration_number():
    base = make_doc("GOOD")[1]["contract"]; ids = {RF.calibration_id(base)}
    for edit in (("baseline_mm", 227.0), ("D1", [0.01, 0, 0, 0, 0]), ("K2", [[1001.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]]), ("image_size_wh", [1280, 720])):
        c = copy.deepcopy(base); c[edit[0]] = edit[1]; ids.add(RF.calibration_id(c))
    assert len(ids) == 5 and RF.calibration_id(copy.deepcopy(base)) == RF.calibration_id(base)


def test_another_session_cannot_use_or_resync_this_workflow(svc, complete, clips):
    s, wf = complete; other = S.VisitorSession()
    with pytest.raises(RF.WorkflowError, match="another session"):
        wf.assert_owner(other.owner)
    with pytest.raises(RF.WorkflowError, match="another session"):
        wf.sync(other.owner, s.contract, s.f6.job.cal, svc.room_sources(s))


# ================================================================ 8-9. invalidation
def test_replacing_a_video_invalidates_only_that_viewpoint(svc, complete, clips):
    s, wf = complete; full2 = dict(wf.full[(0, "AR")])
    p = clips[0]["p1_left"]; data = bytearray(p.read_bytes()); data[-1] ^= 0xFF; data[-2] ^= 0x55                      # a different file, same container
    put(svc.room_store, s.owner, "p1_left", bytes(data), name="replacement.mp4")                                         # a replacement upload to the same slot
    svc._inspector = lambda path, t=svc._inspector: (t(path) if Path(path).stat().st_size in clips[1] else dict(clips[1][p.stat().st_size], size=Path(path).stat().st_size))
    st = svc.frame_state(s); wf2 = st["workflow"]
    assert wf2 is wf and not any(k[0] == 1 for k in wf.pairs) and not any(k[1] in ("AL", "PL") for k in wf.full) and not any(k[1][:2] in ("AL", "PL") for k in wf.ann)
    assert all(wf.pairs[(2, i)]["confirmed"] for i in range(2)) and wf.full[(0, "AR")] == full2 and (0, "AR_corner") in wf.ann and (1, "PR_ceilF") in wf.ann   # viewpoint 2 survives
    assert any("replaced" in n for n in s.frame_notes) and not wf.readiness().ready and wf.readiness().missing


def test_a_new_calibration_invalidates_everything(svc, complete):
    s, wf = complete; ws_path = Path(wf.ws.path)
    assert ws_path.is_dir()
    other = make_doc("GOOD")[1]; other["extrinsics"]["baseline_mm"] = 231.0; other["contract"]["baseline_mm"] = 231.0
    s.forget_acceptance()                                                           # what every new calibration attempt does
    assert s.frames is None and not ws_path.exists() and any("calibration changed" in n for n in s.frame_notes)
    s.review = S.review_f8_document(other); svc.accept(s)
    st = svc.frame_state(s); assert st["state"] == "ok" and st["workflow"] is not wf and not st["workflow"].pairs and st["workflow"].calibration_record()["baseline_mm"] == 231.0


def test_sync_with_a_changed_calibration_discards_all_work(svc, complete):
    s, wf = complete; c2 = copy.deepcopy(s.contract); c2["baseline_mm"] = 230.0
    cal2 = type("C", (), dict(image_size=(W, H), K_left=wf.cal.K_left, D_left=wf.cal.D_left, K_right=wf.cal.K_right, D_right=wf.cal.D_right, baseline_mm=230.0))()
    notes = wf.sync(s.owner, c2, cal2, svc.room_sources(s))
    assert notes and "calibration" in notes[0] and not wf.pairs and not wf.full and not wf.ann and wf.calibration_record()["baseline_mm"] == 230.0
    assert not list(Path(wf.ws.path).glob("job_*"))                                   # frames of the old calibration are gone from disk


def test_removing_the_accepted_calibration_closes_the_workflow(svc, complete):
    s, wf = complete; s.f6.close()
    assert svc.frame_state(s)["state"] == "no_calibration" and s.frames is None and wf.closed


# ================================================================ 10. expired uploads
def test_expired_upload_is_detected_and_no_stale_path_is_used(svc, complete, clips):
    s, wf = complete; stale = wf.source("p2_right").path
    base = svc.room_store._clock(); svc.room_store._clock = lambda: base + 3600; svc.room_store.sweep()                  # retention is 15 minutes of inactivity
    assert not Path(stale).exists()
    st = svc.frame_state(s)
    assert st["state"] == "expired" and any("Right camera" in p or "Left camera" in p for p in st["problems"]) and not s.room_ready
    assert not wf.readiness().ready and "available" in " ".join(wf.readiness().missing)
    with pytest.raises(RF.WorkflowError, match="expired|available"):
        wf.preview("p2_right", 3)                                                   # a new job on an expired source is refused before any worker starts
    assert any("no longer available" in n for n in s.frame_notes)
    with pytest.raises(S.ServiceError):
        svc.go_measure(s)


def test_a_missing_file_is_not_decoded(tmp_path):
    ws = FW.FrameWorkspace(tmp_path); lim = FW.FrameJobLimits(max_file_bytes=10 * MiB)
    try:
        with pytest.raises(FW.FrameJobError) as e:
            FW.run_frame_job(ws, tmp_path / "gone.mp4", lim, preview_indices=[0])
        assert e.value.code == "video_missing" and not list(ws.path.glob("job_*"))
    finally:
        ws.close()


def test_activity_keeps_uploads_alive_but_never_revives_expired_ones(svc, ready):
    s, wf = ready; store = svc.room_store; now = [store._clock()]; store._clock = lambda: now[0]
    now[0] += 600; assert store.touch_owner(s.owner) == 4; now[0] += 600; store.sweep()
    assert all(v is not None for v in svc.room_sources(s).values())                  # 20 minutes after the upload but only 10 after the last interaction
    now[0] += 1000; store.sweep(); assert store.touch_owner(s.owner) == 0 and all(v is None for v in svc.room_sources(s).values())


# ================================================================ 11. session isolation
def test_two_sessions_never_share_workspaces_or_frames(svc, clips):
    a, b = accepted(svc), accepted(svc); upload_all(svc, a, clips); upload_all(svc, b, clips)
    wa, wb = svc.frame_state(a)["workflow"], svc.frame_state(b)["workflow"]
    assert wa is not wb and wa.ws.path != wb.ws.path and wa.binding.owner != wb.binding.owner
    choose_and_confirm(wa); choose_and_confirm(wb)
    paths_a = {v["path"] for v in wa.full.values()}; paths_b = {v["path"] for v in wb.full.values()}
    assert paths_a and paths_b and not paths_a & paths_b and all(Path(wa.ws.path) in Path(p).parents for p in paths_a)
    assert svc.room_sources(a)["p1_left"].path != svc.room_sources(b)["p1_left"].path
    a.reset(svc.store, svc.room_store)                                               # Start over for A only
    assert wa.closed and not Path(wa.ws.path).exists() and not wb.closed and all(Path(p).is_file() for p in paths_b)
    assert all(v is not None for v in svc.room_sources(b).values())


def test_lock_removes_this_visitors_frames_and_nothing_else(svc, complete, clips):
    s, wf = complete; other = accepted(svc); upload_all(svc, other, clips); wo = svc.frame_state(other)["workflow"]
    s.reset(svc.store, svc.room_store)
    assert s.frames is None and not Path(wf.ws.path).exists() and not wo.closed and Path(wo.ws.path).is_dir()


# ================================================================ 12. limit propagation (large videos)
@pytest.fixture
def env600(monkeypatch):
    monkeypatch.setenv("F8_ROOM_MAX_FILE_MB", "600")                                 # set BEFORE the session accepts its calibration (fixture order)


def test_the_room_limit_reaches_every_frame_job(env600, svc, ready):
    s, wf = ready; lim = svc.frame_job_limits()
    assert lim.video.max_file_bytes == 600 * MiB == RV.room_video_limits().max_file_bytes > RV.VV.DEFAULT_VIDEO_LIMITS.max_file_bytes == 200 * MiB
    assert FW.FrameJobLimits.from_job_limits(lim).max_file_bytes == 600 * MiB
    assert RV.room_video_limits().max_file_bytes == 600 * MiB and s.f6_session().job.limits.video.max_file_bytes == RV.room_video_limits().max_file_bytes
    assert JobLimits().video.max_file_bytes == 200 * MiB                           # the global default is untouched
    changed = {f.name for f in fields(JobLimits) if getattr(JobLimits(video=RV.room_video_limits()), f.name) != getattr(JobLimits(), f.name)}
    assert changed == {"video"}


def test_a_300_mb_video_passes_the_size_check_and_one_above_the_limit_never_reaches_the_worker(tmp_path):
    ws = FW.FrameWorkspace(tmp_path); spawned = []
    lim = FW.FrameJobLimits.from_job_limits(JobLimits(video=RV.room_video_limits()))
    big = tmp_path / "big.mp4"
    try:
        with open(big, "wb") as fh:
            fh.truncate(300 * MiB)                                                   # sparse placeholder: no real data
        with pytest.raises(FW.FrameJobError) as e:
            FW.run_frame_job(ws, big, lim, preview_indices=[0], worker_cmd=[sys.executable, "-c", "print('{}')"])
        assert e.value.code != "file_too_large"                                      # F6's 200 MB default would have refused this; the room limit does not
        tiny = FW.FrameJobLimits(max_file_bytes=100 * MiB)
        with pytest.raises(FW.FrameJobError) as e:
            FW.run_frame_job(ws, big, tiny, preview_indices=[0], worker_cmd=[sys.executable, "-c", "open(%r,'w').write('spawned')" % str(tmp_path / "spawned")])
        assert e.value.code == "file_too_large" and not (tmp_path / "spawned").exists()
    finally:
        ws.close()


def test_the_worker_enforces_frame_pixel_and_index_limits(clips, tmp_path):
    ws = FW.FrameWorkspace(tmp_path); p = next(iter(clips[0].values()))
    try:
        with pytest.raises(FW.FrameJobError) as e:
            FW.run_frame_job(ws, p, FW.FrameJobLimits(max_file_bytes=64 * MiB, max_frame_pixels=1_000_000), preview_indices=[0])
        assert e.value.code == "dimensions" and not list(ws.path.glob("job_*"))
        with pytest.raises(FW.FrameJobError) as e:
            FW.run_frame_job(ws, p, FW.FrameJobLimits(max_file_bytes=64 * MiB, max_indices=2), full_indices=[0, 1, 2])
        assert e.value.code == "too_many_frames"
        with pytest.raises(FW.FrameJobError) as e:
            FW.run_frame_job(ws, p, FW.FrameJobLimits(max_file_bytes=64 * MiB), preview_indices=[N_FRAMES + 50])
        assert e.value.code == "frame_not_decodable" and not list(ws.path.glob("job_*"))
        with pytest.raises(FW.FrameJobError) as e:
            FW.run_frame_job(ws, p, FW.FrameJobLimits(max_file_bytes=64 * MiB, max_scan_frames=5), preview_indices=[10])
        assert e.value.code == "frame_not_decodable"                                  # decoding is bounded in frames too
    finally:
        ws.close()


# ================================================================ 13. worker timeout and cleanup
def test_a_hung_worker_is_killed_and_leaves_nothing(tmp_path):
    ws = FW.FrameWorkspace(tmp_path); v = tmp_path / "v.mp4"; v.write_bytes(b"x" * 100)
    marker = tmp_path / "alive"
    code = "import time,os,sys;open(%r,'w').write(str(os.getpid()));time.sleep(60)" % str(marker)
    t0 = time.perf_counter()
    try:
        with pytest.raises(FW.FrameJobError) as e:
            FW.run_frame_job(ws, v, FW.FrameJobLimits(max_file_bytes=MiB, timeout_s=1.0), preview_indices=[0], worker_cmd=[sys.executable, "-c", code])
        assert e.value.code == "timeout" and time.perf_counter() - t0 < 15 and not list(ws.path.glob("job_*"))
        pid = int(marker.read_text())
        with pytest.raises(ProcessLookupError):
            for _ in range(50):
                os.kill(pid, 0); time.sleep(0.1)                                      # the process group was killed
            os.kill(pid, 0)
    finally:
        ws.close()


def test_a_crashing_worker_and_garbage_output_are_visitor_safe_and_clean(tmp_path):
    ws = FW.FrameWorkspace(tmp_path); v = tmp_path / "v.mp4"; v.write_bytes(b"x" * 100); lim = FW.FrameJobLimits(max_file_bytes=MiB)
    try:
        for cmd, code in (([sys.executable, "-c", "import sys;sys.exit(3)"], "worker_failed"), ([sys.executable, "-c", "print('not json')"], "worker_failed"), ([sys.executable, "-c", "print('{}')"], "not_decodable")):
            with pytest.raises(FW.FrameJobError) as e:
                FW.run_frame_job(ws, v, lim, preview_indices=[0], worker_cmd=cmd)
            assert e.value.code == code and str(tmp_path) not in e.value.message and not list(ws.path.glob("job_*"))
    finally:
        ws.close()


def test_a_worker_that_writes_outside_its_job_directory_is_rejected(tmp_path):
    ws = FW.FrameWorkspace(tmp_path); v = tmp_path / "v.mp4"; v.write_bytes(b"x" * 100); outside = tmp_path / "elsewhere.png"; outside.write_bytes(b"png")
    code = "import json;print(json.dumps(dict(opened=True,decoded_frames=1,written={'prev_0':dict(path=%r,wh=[1,1],full_wh=[1,1])})))" % str(outside)
    try:
        with pytest.raises(FW.FrameJobError):
            FW.run_frame_job(ws, v, FW.FrameJobLimits(max_file_bytes=MiB), preview_indices=[0], worker_cmd=[sys.executable, "-c", code])
        assert not list(ws.path.glob("job_*")) and outside.exists()                     # and it is not deleted either: only job directories inside the workspace are
    finally:
        ws.close()


def test_the_worker_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv("F8_PASSCODE", "super-secret-passcode"); assert "F8_PASSCODE" not in FW._scrubbed_env()


def test_workspace_is_private_and_closes_cleanly(tmp_path):
    ws = FW.FrameWorkspace(tmp_path); d1, d2 = ws.new_job_dir(), ws.new_job_dir()
    assert d1 != d2 and oct(ws.path.stat().st_mode & 0o777) == "0o700"                   # no shared mutable output path
    (d1 / "f").write_bytes(b"1" * 10); assert ws.size_bytes() == 10
    ws.close(); ws.close(); assert not ws.path.exists()
    with pytest.raises(FW.FrameJobError):
        ws.new_job_dir()


def test_close_removes_every_preview_and_frame(complete):
    s, wf = complete; root = Path(wf.ws.path); assert any(root.rglob("*.png"))
    wf.close(); assert not root.exists() and wf.closed
    with pytest.raises(RF.WorkflowError):
        wf.assert_owner(s.owner)


def test_temporary_storage_is_bounded_by_the_extracted_frames(complete):
    s, wf = complete; used = wf.ws.size_bytes()
    assert 0 < used < 8 * 4 * W * H + 40 * MiB                                           # lossless but compressed: never more than raw frames plus previews
    n_before = len(list(Path(wf.ws.path).glob("job_*")))
    for _ in range(3):
        wf.extract_confirmed()                                                            # re-running extracts nothing and adds no directory
    assert len(list(Path(wf.ws.path).glob("job_*"))) == n_before


# ================================================================ 14. navigation
def test_navigation_ready_room_frames_measure(svc, complete):
    s, wf = complete; svc.room_report(s); s.step = "room"
    svc.go_frames(s); assert s.step == "frames"
    s.go("room"); assert s.step == "room"
    svc.go_frames(s); svc.go_measure(s); assert s.step == "measure"
    s.go("frames"); assert s.step == "frames" and wf.readiness().ready                 # going back keeps the work


def test_measure_is_unreachable_until_everything_is_marked(svc, ready):
    s, wf = ready; svc.room_report(s)
    with pytest.raises(S.ServiceError, match="Select the frames"):
        s.go("measure")
    with pytest.raises(S.ServiceError):
        svc.go_measure(s)
    choose_and_confirm(wf)
    with pytest.raises(S.ServiceError):
        svc.go_measure(s)                                                                # frames confirmed, boundaries still missing
    annotate_all(wf)
    svc.go_measure(s); assert s.step == "measure"


def test_a_stale_measure_step_cannot_survive_an_invalidating_change(svc, complete, clips):
    s, wf = complete; svc.go_measure(s)
    wf.set_pair(1, 0, 11, 14)
    with pytest.raises(S.ServiceError):
        s.go("measure")
    with pytest.raises(S.ServiceError):
        svc.go_measure(s)


def test_unknown_step_and_missing_calibration_are_refused():
    s = S.VisitorSession()
    for step in ("frames", "measure"):
        with pytest.raises(S.ServiceError):
            s.go(step)
    with pytest.raises(S.ServiceError):
        s.go("results")


# ================================================================ 15. reconstruction stays disabled
def test_room_processing_is_disabled_and_nothing_runs_method_j(svc, complete):
    s, wf = complete
    assert S.ROOM_PROCESSING_ENABLED is False and svc.room_processing_enabled() is False
    svc.go_measure(s)
    assert S.ROOM_PROCESSING_ENABLED is False
    for f in (HERE / "room_frames.py", HERE / "frame_worker.py"):
        src = f.read_text()
        for banned in ("visitor_method_j_adapter", "import pose", "pose.", "pipeline", "features", "visitor_uncertainty", "estimate_pose", "measure_all_sets", "run_estimation"):
            assert not re.search(rf"(^|\s)(import|from)\s+\S*{re.escape(banned)}", src, re.M) and (banned not in ("estimate_pose", "measure_all_sets", "run_estimation") or banned not in src), (f.name, banned)
    for attr in ("measure", "run_measurement", "reconstruct", "run_method_j", "evaluate"):
        assert not hasattr(svc, attr) and not hasattr(wf, attr)


def test_original_sources_are_untouched():
    import subprocess
    root = HERE.parents[2]
    tracked = [f"Modules/Module_7/room_measurement/{n}.py" for n in ("pose", "measure", "geometry", "lines", "features", "pipeline", "calibration")]
    r = subprocess.run(["git", "status", "--porcelain", "--"] + tracked, cwd=root, capture_output=True, text=True)
    assert r.stdout.strip() == ""                                                      # the original Method J and calibration sources are tracked and identical to HEAD


def test_no_original_recording_is_referenced():
    for f in (HERE / "room_frames.py", HERE / "frame_worker.py"):
        src = f.read_text()
        assert not re.search(r"\.MOV\b", src) and "DEFAULT_VIDEO_DIR" not in src and "data/room_measurement" not in src
        assert "load_frame_sets" not in src and "verify_videos" not in src

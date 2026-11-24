"""EXPENSIVE regression of the PRODUCTION three-viewpoint path on the protected Wall B recordings (several minutes, about 2.6 GB).  Opt in with ``M7_REAL_VIDEO_TESTS=1``; skipped when the data are absent.

Path under test (everything except the Streamlit page and the upload store): sandboxed frame extraction (``frame_worker``, sequential decode) -> the seven-line annotation workflow
(the owner's annotation backup is replayed click by click) -> ``three_view_runner.run`` (input checks, staged frames, supervised sandboxed worker, quality classification, result assembly).
The recordings, the annotation backup and the Recipe A calibration file are only READ.  Reference values: frame 60 = 3790.889 mm, 150 = 3811.971 mm, 240 = 3803.973 mm (regression
references from the offline experiment, never inputs); tolerance +-5 mm.  The tape-measured length is not used anywhere in this file."""
import json
import os
from pathlib import Path

import cv2
import pytest

import room_frames as RF
import room_reconstruction as RR
import room_videos as RV
import three_view_runner as TVRUN
import three_view_workflow as TVW
from room_measurement import three_view_wall_b as T
from room_measurement.visitor_measurement import JobLimits

MODULE = Path(__file__).resolve().parents[2]
DATA = MODULE / "data" / "room_measurement" / "wall_b_three_viewpoints"
ANN = DATA / "wall_b_three_viewpoint_annotations_backup_20261120.json"
CALIB = MODULE / "calibration" / "room_video" / "room_video_recipe_a_calibration.json"
VIDEO = {f"p{v}_{side}": DATA / (f"left_air_2_{v}.MOV" if side == "left" else f"right_pro_2_{v}.MOV") for v in (1, 2, 3) for side in ("left", "right")}
FRAMES = (60, 150, 240); REFERENCE_MM = {0: 3790.889, 1: 3811.971, 2: 3803.973}; TOLERANCE_MM = 5.0

pytestmark = [pytest.mark.real_video, pytest.mark.skipif(os.environ.get("M7_REAL_VIDEO_TESTS") != "1", reason="expensive: set M7_REAL_VIDEO_TESTS=1"),
              pytest.mark.skipif(not (ANN.exists() and CALIB.exists() and all(p.exists() for p in VIDEO.values())), reason="protected Wall B recordings are not present")]


@pytest.fixture(scope="module")
def production_result(tmp_path_factory):
    import resource_budget as RB
    tmp = tmp_path_factory.mktemp("real3v")
    rec = json.loads(CALIB.read_text())
    contract = dict(K1=rec["left"]["K"], D1=rec["left"]["distortion_k1_k2_p1_p2_k3"], K2=rec["right"]["K"], D2=rec["right"]["distortion_k1_k2_p1_p2_k3"], image_size_wh=rec["image_size_wh"],
                    baseline_mm=rec["physical_baseline_mm"], capture_mode=dict(left="video", right="video"))
    cal = T.calibration_from_contract(contract); sources = {}
    for slot, p in VIDEO.items():
        cap = cv2.VideoCapture(str(p)); n, fps, wh = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))); cap.release()
        sources[slot] = RF.SourceVideo(slot, str(p), f"upload-{slot}", RR.sha256_stream(p), p.stat().st_size, n, fps, wh)
    limits = RR.ReconstructionLimits(); owner = "real-data-regression-owner"
    wf = TVW.ThreeViewWorkflow(owner, contract, cal, sources, JobLimits(video=RV.room_video_limits()), workspace_root=tmp)
    for i, f in enumerate(FRAMES):
        wf.set_frame(i, f)
    assert wf.extract_frames() == 18 and wf.confirmed
    doc = json.loads(ANN.read_text()); order = {f: i for i, f in enumerate(FRAMES)}
    for a in doc["annotations"]:                                                   # the owner's 21 annotations, replayed as clicks on the exact extracted frames
        line = f"V{a['viewpoint'][1:]}_{a['camera']}_{a['orientation']}"
        for x, y in a["points"]:
            wf.click(order[a["frame_index"]], line, x, y)
    assert wf.readiness()["ready"] and wf.usable_sets() == [0, 1, 2]
    coord = RB.ResourceCoordinator(root=tmp / "rb", cfg=RB.BudgetConfig(ceiling_mb=65536.0), container=dict(limit_mb=None, reliable=False, reason="test isolation"))
    result = TVRUN.run(RR, wf, owner, contract, cal, sources, limits, workspace_root=tmp, coordinator=coord)
    yield result, limits
    wf.close()


def test_production_path_reproduces_the_offline_frame_lengths(production_result):
    r, _ = production_result
    assert r["status"] in ("SUCCESS", "QUESTIONABLE"), r["failures"]
    got = {f["frame_set"]: f for f in r["three_view"]["frames"]}
    for k, ref in REFERENCE_MM.items():
        print(f"frame set {k + 1}: production {got[k]['length_mm']:.3f} mm, offline reference {ref:.3f} mm, difference {got[k]['length_mm'] - ref:+.3f} mm")
        assert abs(got[k]["length_mm"] - ref) <= TOLERANCE_MM, (k, got[k]["length_mm"], ref)
    print(f"mean: production {r['width_mm']:.3f} mm, reference {sum(REFERENCE_MM.values()) / 3:.3f} mm")
    assert abs(r["width_mm"] - sum(REFERENCE_MM.values()) / 3) <= TOLERANCE_MM and r["height_mm"] is None


def test_production_path_classifies_the_recordings_as_approximate_with_warnings_and_no_failure(production_result):
    r, _ = production_result; tv = r["three_view"]
    assert r["status"] == "QUESTIONABLE" and tv["frames_usable"] == 3 and not tv["frames_failed"] and not any(f["hard_failures"] for f in tv["frames"])
    assert {w["code"] for w in r["warnings"]} <= {"CORNER_DIRECTION", "CORNER_OFF_FLOOR", "FLOOR_ANGLE"} and r["warnings"]
    assert tv["matcher"] == "fast" and r["provenance"]["calibration"]["baseline_mm"] == 225.0


def test_production_path_stays_within_the_unchanged_time_and_memory_limits(production_result):
    r, limits = production_result; w = r["processing"]["worker"]
    assert w["exit_code"] == 0 and w["wall_seconds"] < limits.timeout_s and w["limit_mb"] <= limits.max_tree_rss_mb and (w["tree_peak_rss_mb"] or 0) < w["limit_mb"]
    print(f"worker wall {w['wall_seconds']:.0f} s, tree peak RSS {w['tree_peak_rss_mb']:.0f} MB, self peak {r['processing']['worker_self_peak_rss_mb']:.0f} MB (limits: {limits.timeout_s:g} s, {limits.max_tree_rss_mb:g} MB)")

"""EXPENSIVE regression on the protected Wall B recordings (about 2 minutes, 2.5 GB).  Opt in with ``M7_REAL_VIDEO_TESTS=1``; skipped when the data are absent.

The recordings, the annotation backup and the Recipe A calibration file are only READ.  The reference values are the offline experiment's results (regression references, never inputs):
frame 60: 3791 mm, frame 150: 3812 mm, frame 240: 3804 mm.  The tape-measured length (3937 mm) is deliberately not used anywhere in this file.
"""
import json
import os
from pathlib import Path

import pytest

from room_measurement import three_view_wall_b as T

MODULE = Path(__file__).resolve().parents[2]
DATA = MODULE / "data" / "room_measurement" / "wall_b_three_viewpoints"
ANN = DATA / "wall_b_three_viewpoint_annotations_backup_20261120.json"
CALIB = MODULE / "calibration" / "room_video" / "room_video_recipe_a_calibration.json"
VIDEOS = {(v, s): DATA / (f"left_air_2_{v}.MOV" if s == "L" else f"right_pro_2_{v}.MOV") for v in (1, 2, 3) for s in "LR"}
REFERENCE_MM = {60: 3791.0, 150: 3812.0, 240: 3804.0}
TOLERANCE_MM = 5.0

pytestmark = [pytest.mark.real_video,
              pytest.mark.skipif(os.environ.get("M7_REAL_VIDEO_TESTS") != "1", reason="expensive: set M7_REAL_VIDEO_TESTS=1"),
              pytest.mark.skipif(not (ANN.exists() and CALIB.exists() and all(p.exists() for p in VIDEOS.values())), reason="protected Wall B recordings are not present")]


def _inputs():
    rec = json.loads(CALIB.read_text())            # the accepted calibration, presented as a contract (K1/D1 = Air, K2/D2 = Pro): the module never opens this file itself
    cal = T.calibration_from_contract(dict(K1=rec["left"]["K"], D1=rec["left"]["distortion_k1_k2_p1_p2_k3"], K2=rec["right"]["K"], D2=rec["right"]["distortion_k1_k2_p1_p2_k3"],
                                           image_size_wh=rec["image_size_wh"], baseline_mm=rec["physical_baseline_mm"]))
    return cal, T.parse_annotations(json.loads(ANN.read_text()), cal.image_size)


@pytest.fixture(scope="module")
def result():
    cal, ann = _inputs()
    return T.measure_from_videos(cal, VIDEOS, sorted(REFERENCE_MM), ann)


def test_each_frame_reproduces_the_offline_experiment(result):
    for r in result["per_frame"]:
        assert abs(r["length_mm"] - REFERENCE_MM[r["frame"]]) <= TOLERANCE_MM, (r["frame"], r["length_mm"])


def test_the_estimate_is_the_mean_of_the_frames(result):
    assert abs(result["length_mm"] - sum(REFERENCE_MM.values()) / 3) <= TOLERANCE_MM


def test_no_frame_has_a_hard_failure_and_warnings_are_reported_with_the_number(result):
    """Provisional checks: the recordings are usable but not clean (corner-direction, corner/floor and floor-angle warnings), so the result is approximate-with-warnings, never plain MEASURED."""
    q = result["quality"]
    assert all(not r["quality"]["hard_failures"] for r in result["per_frame"]) and q["frames_usable"] == 3
    assert q["status"] == "MEASURED_WITH_WARNINGS" and q["frame_warnings"]
    assert {w["code"] for w in q["frame_warnings"]} <= {"CORNER_DIRECTION", "CORNER_OFF_FLOOR", "FLOOR_ANGLE"}


def test_the_original_matcher_gives_identical_lengths_to_the_fast_default(result):
    """``result`` used the default (fast) matcher; the original ``features.match_set`` path must give exactly the same lengths."""
    cal, ann = _inputs()
    orig = T.measure_from_videos(cal, VIDEOS, sorted(REFERENCE_MM), ann, T.ThreeViewConfig.original_matcher())
    assert [r["length_mm"] for r in orig["per_frame"]] == [r["length_mm"] for r in result["per_frame"]]       # exact: the match sets are identical
    assert orig["length_mm"] == result["length_mm"]


def test_frame_sets_with_a_different_key_than_the_frame_number_reproduce_the_same_lengths(result):
    """The visitor workflow keys annotations by frame-set number, not by frame number: the lengths must not depend on that."""
    cal, ann = _inputs(); doc = json.loads(ANN.read_text()); order = {60: 0, 150: 1, 240: 2}
    for a in doc["annotations"]: a["frame_index"] = order[a["frame_index"]]
    sets = {order[f]: {(v, s): f for v in (1, 2, 3) for s in "LR"} for f in REFERENCE_MM}
    out = T.measure_frame_sets(cal, VIDEOS, sets, T.parse_annotations(doc, cal.image_size))
    assert [r["length_mm"] for r in out["per_frame"]] == [r["length_mm"] for r in result["per_frame"]]

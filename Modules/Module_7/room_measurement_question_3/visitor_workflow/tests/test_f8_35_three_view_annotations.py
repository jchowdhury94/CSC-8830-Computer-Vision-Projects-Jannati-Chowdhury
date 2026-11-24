"""Step 2 of the three-viewpoint Wall B integration: frame sets and the seven-line annotation workflow.  SYNTHETIC clips; no reconstruction."""
import copy
import dataclasses

import numpy as np
import pytest

import f8_viewpoints as VPC
import f8_visitor_service as S
import room_frames as RF
import room_videos as RV
import three_view_workflow as TVW
from room_measurement import three_view_wall_b as T
from test_f8_31b_frames import W, H, accepted, sequential_gray
from test_f8_31b_pages import button, has_button, page_app, texts
from test_f8_35_viewpoint_state import clips6, svc6, upload       # noqa: F401  (fixtures)

f1 = S.F.f1core
V1V, V1P, V1H, V2H, V3V, V3P, V3H = TVW.LINE_NAMES
LEFT_EDGE = [[900.0, 200.0], [930.0, 900.0]]
RIGHT_EDGE = [[1000.0, 150.0], [1040.0, 950.0]]
CEIL = [[100.0, 250.0], [1500.0, 280.0]]


@pytest.fixture
def b3(svc6, clips6):
    """A session whose Wall B uses three viewpoints with six synthetic videos, and its three-viewpoint workflow."""
    s = accepted(svc6); svc6.set_viewpoint_count(s, "B", 3); s.switch_wall("B"); upload(svc6, s, clips6, RV.ROOM_UPLOAD_SLOTS_3)
    st = svc6.frame_state(s); assert st["state"] == "ok", st
    return s, st["workflow"]


def mark(wf, i, line, pts=None):
    ori = TVW.LINE_INFO[line][2]; pts = pts or (CEIL if ori == "horizontal" else LEFT_EDGE if TVW.LINE_INFO[line][1] == "left" else RIGHT_EDGE)
    for x, y in pts:
        wf.click(i, line, x, y)


def mark_set(wf, i, horizontals=True):
    for n in TVW.LINE_NAMES:
        if TVW.LINE_INFO[n][2] == "vertical" or horizontals:
            mark(wf, i, n)


# ------------------------------------------------------------------ schema
def test_the_seven_lines_follow_the_validated_annotation_scheme():
    assert len(TVW.LINE_NAMES) == 7 and TVW.REQUIRED_LINES == (V1V, V1P, V3V, V3P)
    by_vp = {v: [n for n in TVW.LINE_NAMES if TVW.LINE_INFO[n][0] == v] for v in (1, 2, 3)}
    assert len(by_vp[1]) == 3 and len(by_vp[3]) == 3 and by_vp[2] == [V2H]                      # V2: the ceiling line only, no vertical
    assert [TVW.LINE_INFO[n][2] for n in by_vp[2]] == ["horizontal"]
    for v in (1, 3):
        assert sorted(TVW.LINE_INFO[n][1:3] for n in by_vp[v]) == [("left", "vertical"), ("right", "horizontal"), ("right", "vertical")]     # horizontals only from the right (Pro) camera
    assert {TVW.LINE_INFO[n][5] for n in (V1V, V1P)} == {"left_wall_corner"} and {TVW.LINE_INFO[n][5] for n in (V3V, V3P)} == {"right_wall_corner"}
    assert all(not TVW.LINE_INFO[n][3] for n in (V1H, V2H, V3H))                                  # horizontals are optional: not used by the reconstruction


# ------------------------------------------------------------------ frame sets
def test_a_three_viewpoint_session_gets_the_three_viewpoint_workflow_and_two_viewpoints_keep_the_original(svc6, clips6, b3):
    s, wf = b3
    assert isinstance(wf, TVW.ThreeViewWorkflow) and wf.n_sets == 3 and not wf.confirmed
    idx = [wf.sets[i][TVW.SLOTS[0]] for i in range(3)]
    assert idx == sorted(idx) and len(set(idx)) == 3 and all(0 <= x < wf.shortest() for x in idx)
    assert all(len({wf.sets[i][sl] for sl in TVW.SLOTS}) == 1 for i in range(3))                  # the same number in all six videos (the validated form)
    sa = accepted(svc6); upload(svc6, sa, clips6, RV.ROOM_UPLOAD_SLOTS)
    assert type(svc6.frame_state(sa)["workflow"]) is RF.RoomFrameWorkflow


def test_extraction_reads_the_exact_frames_and_confirms(b3):
    s, wf = b3
    assert wf.extract_frames() == 18 and wf.confirmed
    i, slot = 1, "p2_right"; ix = wf.sets[i][slot]
    assert np.array_equal(wf.frame_gray(i, slot), sequential_gray(wf.source(slot).path, ix))      # the Phase 1 decode convention, bit for bit
    assert wf.extract_frames() == 0                                                                # nothing is decoded twice


def test_extraction_refuses_a_frame_that_does_not_match_the_calibration_size(b3):
    s, wf = b3; wf.cal = dataclasses.replace(wf.cal, image_size=(1000, 500))
    with pytest.raises(TVW.ThreeViewError, match="calibration"):
        wf.extract_frames()
    assert not wf.confirmed


def test_changing_one_frame_set_discards_only_its_work(b3):
    s, wf = b3; wf.extract_frames(); mark_set(wf, 0); mark_set(wf, 1)
    wf.set_frame(1, wf.sets[1][TVW.SLOTS[0]] + 1)
    assert not wf.confirmed and wf.line_status(1, V1V) == "empty" and wf.line_status(0, V1V) == "marked"
    wf.extract_frames(); assert wf.confirmed and wf.line_status(0, V1V) == "marked"                 # set 1 kept its extracted frames and lines
    with pytest.raises(RF.WorkflowError, match="does not exist"):
        wf.set_frame(0, 10_000)
    with pytest.raises(TVW.ThreeViewError):
        wf.auto_select(9)


def test_a_frame_used_twice_is_reported_as_correlated(b3):
    s, wf = b3; wf.set_frame(0, 10); wf.set_frame(1, 10)
    assert any("correlated" in m for m in wf.duplicate_frames())


# ------------------------------------------------------------------ marking lines
def test_two_clicks_make_a_line_a_third_is_refused_and_undo_reset_work(b3):
    s, wf = b3
    with pytest.raises(TVW.ThreeViewError, match="Confirm the frame"):
        wf.click(0, V1V, 10, 10)                                                                    # nothing can be marked before the frames are extracted
    wf.extract_frames()
    assert wf.click(0, V1V, 900, 200) == "first" and wf.line_status(0, V1V) == "partial"
    with pytest.raises(TVW.ThreeViewError, match="too close"):
        wf.click(0, V1V, 905, 205)
    assert wf.click(0, V1V, 930, 900) == "second" and wf.line_status(0, V1V) == "marked"
    with pytest.raises(TVW.ThreeViewError, match="two points"):
        wf.click(0, V1V, 5, 5)
    wf.undo(0, V1V); assert wf.state(0, V1V).points == [[900.0, 200.0]] and wf.line_status(0, V1V) == "partial"
    wf.reset(0, V1V); assert wf.line_status(0, V1V) == "empty"
    for bad in ((-1, 5), (5, -1), (W + 1, 5), (5, H + 1), (float("nan"), 5)):
        with pytest.raises(TVW.ThreeViewError, match="outside"):
            wf.click(0, V1V, *bad)


def test_points_on_the_frame_edge_are_accepted(b3):
    s, wf = b3; wf.extract_frames(); wf.click(0, V1V, 0, 0); wf.click(0, V1V, W, H); assert wf.line_status(0, V1V) == "marked"


def test_partially_hidden_corner_can_be_marked_on_its_visible_part(b3):
    s, wf = b3; wf.extract_frames(); mark(wf, 0, V3V, [[1500.0, 100.0], [1520.0, 300.0]])        # only the top 200 px of the edge visible
    assert wf.line_status(0, V3V) == "marked"


def test_unavailable_boundaries_and_their_effect_on_usability(b3):
    s, wf = b3; wf.extract_frames(); mark_set(wf, 0, horizontals=False)
    assert wf.set_usable(0)                                                                         # the horizontals are optional
    wf.set_unavailable(0, V1H); assert wf.line_status(0, V1H) == "unavailable" and wf.set_usable(0)
    with pytest.raises(TVW.ThreeViewError, match="unavailable"):
        wf.click(0, V1H, 10, 10)
    wf.set_unavailable(0, V3V); assert not wf.set_usable(0) and "marked as unavailable" in " ".join(wf.set_problems(0))     # a missing vertical corner excludes the frame set
    wf.set_unavailable(0, V3V, False); assert wf.line_status(0, V3V) == "empty"


def test_copy_from_another_frame_set_needs_a_marked_line(b3):
    s, wf = b3; wf.extract_frames()
    with pytest.raises(TVW.ThreeViewError, match="no marked line"):
        wf.copy_points(0, 1, V1V)
    mark(wf, 0, V1V); wf.copy_points(0, 1, V1V); assert wf.state(1, V1V).points == wf.state(0, V1V).points and wf.line_status(1, V1V) == "marked"


# ------------------------------------------------------------------ coordinate mapping
@pytest.mark.parametrize("zoom,center", [(1.0, None), (4.0, (700.0, 400.0)), (8.0, (1500.0, 900.0)), (32.0, (20.0, 20.0))])
def test_clicks_in_a_zoomed_panned_view_are_stored_in_original_frame_pixels(b3, zoom, center):
    s, wf = b3; wf.extract_frames()
    vs = f1.zoom_by((W, H), f1.ViewState(), zoom) if center is None else f1.clamp_state((W, H), f1.ViewState(zoom, *center))
    view = f1.make_view((W, H), vs, max_wh=(836, 900)); shown_w, shown_h = view.shown_wh
    for fx, fy in ((0.25, 0.3), (0.75, 0.8)):
        click = dict(x=int(fx * shown_w), y=int(fy * shown_h), width=shown_w, height=shown_h)           # the component reports offsets inside the image as the browser shows it
        x, y = view.click_to_full(click)
        X0, Y0, X1, Y1 = view.crop; assert X0 - 1 <= x <= X1 and Y0 - 1 <= y <= Y1                      # inside the visible crop of the ORIGINAL frame
        assert abs(x - (X0 + fx * (X1 - X0))) <= 1.0 + 1.5 / view.sx and abs(y - (Y0 + fy * (Y1 - Y0))) <= 1.0 + 1.5 / view.sy      # within one ORIGINAL pixel (pixel-centre convention) plus one displayed pixel
    # the shrunk-component case: the browser shows the image at half size, the reported offsets are in those units
    click = dict(x=shown_w // 4, y=shown_h // 4, width=shown_w // 2, height=shown_h // 2); x2, y2 = view.click_to_full(click)
    assert abs(x2 - (view.crop[0] + 0.5 * (view.crop[2] - view.crop[0]))) <= 1.0 + 2.0 / view.sx
    wf.click(0, V1V, x, y); assert wf.state(0, V1V).points[0] == [pytest.approx(float(x)), pytest.approx(float(y))]


# ------------------------------------------------------------------ readiness and hand-off
def test_readiness_needs_confirmed_frames_and_two_usable_frame_sets(b3):
    s, wf = b3; assert not wf.readiness()["ready"] and "Confirm the frame selection." in wf.readiness()["missing"]
    wf.extract_frames(); mark_set(wf, 0, horizontals=False)
    rd = wf.readiness(); assert not rd["ready"] and rd["usable_sets"] == [0] and "at least 2" in " ".join(rd["missing"])
    mark(wf, 1, V1V); rd = wf.readiness(); assert not rd["ready"] and any("Frame set 2 cannot be used yet" in m for m in rd["notes"])      # an incomplete set is explained
    for n in (V1P, V3V, V3P): mark(wf, 1, n)
    rd = wf.readiness(); assert rd["ready"] and rd["usable_sets"] == [0, 1] and wf.frame_sets_for_run().keys() == {0, 1}
    assert not [m for m in rd["notes"] if "cannot be used" in m]
    fr = wf.frame_sets_for_run()[0]; assert set(fr) == {(v, sd) for v in (1, 2, 3) for sd in "LR"} and len(set(fr.values())) == 1


def test_annotation_document_matches_the_schema_and_feeds_the_reconstruction_parser(b3):
    s, wf = b3; wf.extract_frames(); mark_set(wf, 0); mark_set(wf, 1, horizontals=False); wf.set_unavailable(1, V1H); mark_set(wf, 2, horizontals=False); wf.set_unavailable(2, V1V)
    doc = wf.annotation_document()
    assert doc["schema"] == "wall_b_three_viewpoint_annotations.v1" and set(doc["frame_sets"]) == {"0", "1"}                  # set 3 lacks a vertical corner: excluded, not silently half-used
    a0 = [a for a in doc["annotations"] if a["frame_index"] == 0]
    assert len(a0) == 7 and {(a["viewpoint"], a["camera"], a["orientation"]) for a in a0 if a["viewpoint"] == "V2"} == {("V2", "right", "horizontal")}
    assert all(a["video_frame"] == wf.sets[0][a["video"]] for a in a0) and len([a for a in doc["annotations"] if a["frame_index"] == 1]) == 4
    ann = wf.annotations(); T.check_annotations_complete(ann, [0, 1])
    assert ann.get(1, 0, "left", "vertical").tolist() == LEFT_EDGE and ann.get(3, 1, "right", "vertical").tolist() == RIGHT_EDGE and ann.image_wh == (W, H)
    assert ann.has(2, 0, "right", "horizontal") and not ann.has(2, 0, "left", "vertical")
    with pytest.raises(T.ThreeViewError):
        T.check_annotations_complete(ann, [2])


# ------------------------------------------------------------------ invalidation: nothing from other videos / counts is reused
def test_replacing_a_video_discards_only_that_viewpoints_work(svc6, clips6, b3):
    s, wf = b3; wf.extract_frames(); mark_set(wf, 0); mark_set(wf, 1)
    svc6.remove_room_video(s, "p2_left"); upload(svc6, s, clips6, ["p2_left"])                  # the same bytes again, but a NEW upload
    wf2 = svc6.frame_state(s)["workflow"]
    assert wf2 is wf and any("Viewpoint 2" in n for n in s.frame_notes)
    assert wf.line_status(0, V1V) == "marked" and wf.line_status(1, V3V) == "marked"            # viewpoints 1 and 3 keep their lines
    assert wf.line_status(0, V2H) == "empty" and not wf.confirmed and not any(k[1].startswith("p2_") for k in wf.full)


def test_a_new_calibration_discards_all_frames_and_lines(svc6, b3):
    s, wf = b3; wf.extract_frames(); mark_set(wf, 0)
    s.forget_acceptance(); assert s.frames is None or s.frames.closed
    assert wf.closed and not wf.ann


def test_changing_the_viewpoint_count_never_reuses_old_work(svc6, clips6, b3):
    s, wf = b3; wf.extract_frames(); mark_set(wf, 0)
    svc6.set_viewpoint_count(s, "B", 2); assert wf.closed and s.frames is None and any("number of viewpoints" in n for n in s.frame_notes)
    upload(svc6, s, clips6, RV.ROOM_UPLOAD_SLOTS)
    wf2 = svc6.frame_state(s)["workflow"]; assert type(wf2) is RF.RoomFrameWorkflow and not wf2.ann
    svc6.set_viewpoint_count(s, "B", 3); upload(svc6, s, clips6, RV.ROOM_UPLOAD_SLOTS_3)
    wf3 = svc6.frame_state(s)["workflow"]; assert isinstance(wf3, TVW.ThreeViewWorkflow) and wf3 is not wf and not wf3.ann and not wf3.confirmed


def test_removal_impact_names_what_would_be_lost(svc6, b3):
    s, wf = b3; assert svc6.removal_impact(s, "p1_left") == []
    wf.extract_frames(); mark_set(wf, 0)
    impact = " ".join(svc6.removal_impact(s, "p1_left")); assert "extracted frames of Viewpoint 1" in impact and "marked boundaries of Viewpoint 1" in impact
    assert "Viewpoint 2" in " ".join(svc6.removal_impact(s, "p2_right"))


def test_wall_a_workflow_is_untouched_by_wall_b_three_viewpoint_work(svc6, clips6, b3):
    s, wf = b3; wf.extract_frames(); mark_set(wf, 0)
    s.switch_wall("A"); assert s.frames is None and s.n_viewpoints == 2
    upload(svc6, s, clips6, RV.ROOM_UPLOAD_SLOTS); wa = svc6.frame_state(s)["workflow"]; assert type(wa) is RF.RoomFrameWorkflow
    s.switch_wall("B"); assert s.frames is wf and wf.line_status(0, V1V) == "marked"


# ------------------------------------------------------------------ the page
def test_page_walks_through_frames_then_boundaries_then_review(svc6, b3):
    s, wf = b3; s.step = "frames"
    at = page_app(svc6, s); at.run(); assert not at.exception, at.exception
    assert "Choose the frames" in " ".join(h.value for h in at.subheader) and has_button(at, "Confirm these frames") and "Boundaries open after you confirm" in texts(at)
    button(at, "Confirm these frames").click().run(); assert not at.exception, at.exception and wf.confirmed
    sub = " ".join(h.value for h in at.subheader); assert "Mark the wall boundaries" in sub and "Review and continue" in sub
    assert "1 of 21" in texts(at) and "Viewpoint 1 (left end)" in texts(at) and button(at, "Next").disabled and button(at, "Back").disabled
    assert button(at, "Continue to Room Measurement").disabled
    wf.click(0, V1V, *LEFT_EDGE[0]); at.run(); assert "Now click the second point" in texts(at) and not button(at, "Undo last point").disabled
    wf.click(0, V1V, *LEFT_EDGE[1]); at.run(); assert "Line marked" in texts(at) and not button(at, "Next").disabled
    button(at, "Next").click().run(); assert "2 of 21" in texts(at) and "right camera · vertical" in texts(at)
    button(at, "Not visible in this frame").click().run(); assert wf.line_status(0, V1P) == "unavailable" and not button(at, "Next").disabled


def test_page_continue_opens_only_with_two_usable_frame_sets(svc6, b3):
    s, wf = b3; wf.extract_frames(); mark_set(wf, 0, horizontals=False); s.step = "frames"
    at = page_app(svc6, s); at.run(); assert not at.exception, at.exception
    assert button(at, "Continue to Room Measurement").disabled and "at least 2" in texts(at)
    mark_set(wf, 1, horizontals=False); at.run(); assert not button(at, "Continue to Room Measurement").disabled and "2 of 3 frame sets are complete" in texts(at)
    button(at, "Continue to Room Measurement").click().run(); assert s.step == "measure", at.exception

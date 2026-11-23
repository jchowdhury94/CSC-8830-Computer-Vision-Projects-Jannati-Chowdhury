"""Step 3 of the three-viewpoint Wall B integration: the result, quality warnings and failures, the final room dimensions, and the gated measurement service.  INJECTED worker output / fake runners only
(the real worker has its own test below); the original recordings are never touched here."""
import json

import pytest

import f8_measurement as M
import f8_result_view as RVW
import f8_visitor_service as S
import room_videos as RV
import three_view_result as TVR
from room_measurement import three_view_quality as Q
from room_measurement.tests.test_three_view_quality import classified
from test_f8_31c2a_measurement_interface import failed_result, ok_result, run_to_completion, use_measurement
from test_f8_35_three_view_annotations import b3, mark, mark_set     # noqa: F401  (fixtures + helpers)
from test_f8_35_viewpoint_state import clips6, svc6, upload          # noqa: F401  (fixtures)
from test_f8_31b_frames import accepted

META = dict(calibration=dict(calibration_id="cal123", baseline_mm=226.0, image_size_wh=[1920, 1080]), videos={"p1_left": dict(sha256="ab" * 32, frame_count=36)},
            frame_sets={"0": {"p1_left": 5}, "1": {"p1_left": 15}, "2": {"p1_left": 25}}, source_sets=[0, 1, 2], excluded_sets=[])


def worker_doc(frames, *, matcher="fast", failures=()):
    """The worker's document for classified frames ``[(length, mutation)]`` (a JSON round trip, as the real worker writes it)."""
    per = []
    for k, (length, mut) in enumerate(frames):
        r = classified(length, **(mut or {})); r["frame"] = k; per.append(r)
    q = Q.aggregate(per)
    doc = dict(schema="f8.three_view_worker/1", failures=list(failures), result=None if failures else dict(per_frame=per, quality=q, length_mm=q["length_mm"], thresholds_note="Provisional engineering checks."),
               timings=dict(total=1.0), peak_rss_mb=1234.0, matcher=matcher)
    return json.loads(json.dumps(doc))


CLEAN = [(3790.0, None), (3800.0, None), (3810.0, None)]
BAD = dict(ang_v1_v2_deg=9.0)                                     # a hard failure (floor planes cannot be the same floor)


# ------------------------------------------------------------------ the result
def test_clean_measurement_is_a_success_with_the_length_stored_as_wall_b_width_and_no_height():
    r = TVR.assemble(META, worker_out=worker_doc(CLEAN))
    assert r["schema"] == RVW.SCHEMA_RESULT and r["method"] == "three_viewpoint_wall_b" and r["status"] == "SUCCESS" and r["dimensions_accepted"] and not r["failures"] and not r["warnings"]
    assert r["width_mm"] == pytest.approx(3800.0) and r["length_mm"] == r["width_mm"] and r["height_mm"] is None and r["ground_truth_used"] is False
    assert RVW.result_state(r) == RVW.DONE_OK
    v = RVW.build_view(r); assert [d["key"] for d in v["dimensions"]] == ["room_length"] and v["dimensions"][0]["label"] == "Estimated Room Length" and v["dimensions"][0]["value"]["cm_text"] == "380 cm"
    assert v["height"] is None and v["three_view"]["approximate"] and v["three_view"]["headline"] == "Approximate length" and v["three_view"]["limitations"] == TVR.LIMITATIONS


def test_warnings_make_it_questionable_and_travel_with_the_number():
    r = TVR.assemble(META, worker_out=worker_doc([(3790.0, dict(ang_v1_v3_deg=2.5)), (3800.0, None), (3810.0, dict(ang_v1_v3_deg=2.6))]))
    assert r["status"] == "QUESTIONABLE" and r["dimensions_accepted"] and RVW.result_state(r) == RVW.DONE_WARN
    assert [w["code"] for w in r["warnings"]] == ["FLOOR_ANGLE"] and r["warnings"][0]["frames"] == [0, 2]                  # one entry per code, with the frame sets it applies to
    v = RVW.build_view(r)["three_view"]
    assert v["headline"] == "Approximate length — 1 quality warning" and [f["status"] for f in v["frames"]] == ["WARNINGS", "OK", "WARNINGS"]
    assert v["frames"][0]["warnings"][0]["code"] == "FLOOR_ANGLE" and v["frames"][0]["warnings"][0]["next_step"] and v["frames"][0]["length"]["cm_text"] == "379 cm"
    assert any(row[0].startswith("Floor agreement") for row in v["indicators"])


def test_a_failed_frame_is_excluded_and_reported_while_two_usable_frames_give_a_length():
    r = TVR.assemble(META, worker_out=worker_doc([(3790.0, None), (5000.0, BAD), (3810.0, None)]))
    assert r["status"] == "QUESTIONABLE" and r["width_mm"] == pytest.approx(3800.0) and {w["code"] for w in r["warnings"]} == {"FLOOR_ANGLE", "SOME_FRAME_SETS_FAILED"} or {w["code"] for w in r["warnings"]} == {"SOME_FRAME_SETS_FAILED"}
    v = RVW.build_view(r)["three_view"]
    assert v["frames_failed"] == [2] and v["frames"][1]["status"] == "FAILED" and v["frames"][1]["hard_failures"][0]["code"] == "FLOOR_ANGLE" and v["frames"][1]["length"] is None
    assert [s["status"] for s in r["per_frame_set"]] == ["reconstructed", "failed", "reconstructed"]


def test_fewer_than_two_usable_frames_is_a_failure_with_no_length_at_all():
    r = TVR.assemble(META, worker_out=worker_doc([(3790.0, None), (5000.0, BAD), (5100.0, BAD)]))
    assert r["status"] == "FAILED" and not r["dimensions_accepted"] and r["width_mm"] is None and r["length_mm"] is None and all(s["width_mm"] is None for s in r["per_frame_set"])
    assert r["failures"][0]["code"] == "THREE_VIEW_INSUFFICIENT_EVIDENCE" and r["failures"][0]["visitor_message"]
    assert RVW.result_state(r) == RVW.DONE_FAILED
    v = RVW.build_view(r); assert v["dimensions"] == [] and v["width"] is None and v["failure"]["code"] == "THREE_VIEW_INSUFFICIENT_EVIDENCE" and v["failure"]["retry"]["step"] == "frames"
    t = v["three_view"]; assert not t["ok"] and t["length"] is None and all(f["length"] is None for f in t["frames"]) and [f["status"] for f in t["frames"]] == ["OK", "FAILED", "FAILED"]
    assert t["frames"][1]["hard_failures"] and "at least 2" in t["message"]                                              # the frame-level reasons are kept for display
    assert "3790" not in json.dumps({k: r[k] for k in ("width_mm", "length_mm", "per_frame_set")})                       # not even the one good frame's number leaks as a result


def test_excluded_frame_sets_are_reported_and_make_the_result_approximate():
    meta = dict(META, excluded_sets=[3])
    r = TVR.assemble(meta, worker_out=worker_doc(CLEAN))
    assert r["status"] == "QUESTIONABLE" and "THREE_VIEW_SETS_EXCLUDED" in [w["code"] for w in r["warnings"]] and RVW.build_view(r)["three_view"]["excluded"] == [4]


@pytest.mark.parametrize("worker_code,code,cls", [("RESOLUTION_MISMATCH", "CALIBRATION_VIDEO_SIZE_MISMATCH", "visitor_input"), ("ANNOTATION_MISSING", "ANNOTATION_MISSING", "visitor_input"),
                                                  ("FRAME_STALE", "FRAME_STALE", "visitor_input"), ("SOMETHING_ELSE", "SOMETHING_ELSE", "method_j")])
def test_structural_worker_failures_map_to_the_existing_typed_codes(worker_code, code, cls):
    r = TVR.assemble(META, worker_out=dict(schema="f8.three_view_worker/1", failures=[dict(code=worker_code, message="technical /secret/path", stage="frames")], result=None))
    assert r["status"] == "FAILED" and r["failures"][0]["code"] == code and r["failures"][0]["failure_class"] == cls and r["width_mm"] is None
    assert "/secret/path" not in RVW.build_view(r)["failure"]["what"]


def test_every_quality_code_has_visitor_wording():
    import visitor_messages as VM
    from room_measurement import three_view_quality as Q2
    codes = {"POSE_DEGENERATE", "POSE_RESIDUAL", "POSE_INLIERS", "POSE_DEPTH", "POSE_FEW_MATCHES", "POSE_FALLBACK_START", "REG_FEW_INLIERS", "REG_INLIER_FRACTION", "REG_RMS", "REG_COLLINEAR", "REG_PLANAR",
             "REG_COVERAGE", "REG_PAIRS", "FLOOR_FIT", "FLOOR_FEW_POINTS", "FLOOR_ANGLE", "CORNER_DIRECTION", "CORNER_OFF_FLOOR", "LENGTH_NOT_HORIZONTAL", "LENGTH_OUT_OF_RANGE", "LENGTH_INVALID",
             "RECONSTRUCTION_FAILED", "FRAME_SPREAD"}
    for c in codes:
        assert VM.failure_parts(c)["known"] or VM.warning_parts(c)["known"], c
    assert all(VM.failure_parts(c)["known"] for c in ("THREE_VIEW_INSUFFICIENT_EVIDENCE", "POSE_DEGENERATE", "REG_PAIRS", "RECONSTRUCTION_FAILED"))
    assert all(VM.warning_parts(c)["known"] for c in ("REG_COVERAGE", "FRAME_SPREAD", "THREE_VIEW_SETS_EXCLUDED", "CORNER_DIRECTION"))
    assert VM.warning_parts("SOME_FRAME_SETS_FAILED")["meaning"] == "Some frame sets could not be used; the result comes from the remaining sets."           # existing wording untouched


# ------------------------------------------------------------------ Wall A / Wall B result isolation and the final room dimensions
A_RES = ok_result(width=2850.0, height=2358.0)


def test_final_room_combines_wall_a_width_and_height_with_wall_b_three_viewpoint_length():
    b = TVR.assemble(META, worker_out=worker_doc([(3790.889, None), (3811.971, None), (3803.973, dict(ang_v1_v3_deg=2.5))]))
    rv = RVW.room_view(A_RES, b); d = {x["key"]: x for x in rv["dimensions"]}
    assert rv["complete"] and d["room_width"]["value"]["mm"] == 2850.0 and d["room_height"]["value"]["mm"] == 2358.0 and d["room_length"]["value"]["mm"] == pytest.approx(3802.2777, abs=1e-3)
    assert (d["room_width"]["wall"], d["room_length"]["wall"], d["room_height"]["wall"]) == ("A", "B", "A") and d["room_length"]["source"] == "Wall B · three-viewpoint length"
    assert rv["limitations"] == TVR.LIMITATIONS and rv["wall_b_three_view"]["approximate"] and rv["wall_b_three_view"]["frames"][2]["warnings"]
    assert d["room_width"]["source"] == "Wall A · horizontal width" and d["room_height"]["source"] == "Wall A · height"                               # Wall A's two-viewpoint entries are unchanged


def test_wall_b_height_is_never_used_for_the_room_height():
    b = TVR.assemble(META, worker_out=worker_doc(CLEAN)); b["height_mm"] = 9999.0                         # even a stray height on Wall B's result must not leak
    d = {x["key"]: x for x in RVW.room_view(A_RES, b)["dimensions"]}
    assert d["room_height"]["value"]["mm"] == 2358.0
    assert RVW.room_view(A_RES, ok_result(width=3400.0, height=2400.0))["limitations"] == RVW.LIMITATIONS_TEXT             # a two-viewpoint Wall B keeps the original statement
    assert RVW.room_view(A_RES, ok_result(width=3400.0, height=2400.0))["wall_b_three_view"] is None


def test_the_room_is_incomplete_when_wall_b_has_no_usable_result():
    b = TVR.assemble(META, worker_out=worker_doc([(3790.0, None), (5000.0, BAD), (5100.0, BAD)]))
    rv = RVW.room_view(A_RES, b); assert not rv["complete"] and rv["dimensions"] == [] and RVW.room_summary_document(A_RES, b) is None


def test_downloadable_summaries_carry_method_quality_and_limitations_but_no_ground_truth():
    b = TVR.assemble(META, worker_out=worker_doc([(3790.889, None), (3811.971, None), (3803.973, dict(ang_v1_v3_deg=2.5))]))
    doc = RVW.room_summary_document(A_RES, b); text = json.dumps(doc)
    assert doc["dimensions"]["room_length"]["cm"] == pytest.approx(380.23, abs=0.01) and doc["note"] == TVR.LIMITATIONS
    wb = doc["walls"]["wall_b"]; assert wb["method"] == "three_viewpoint_wall_b" and wb["approximate"] and wb["three_view"]["per_frame"][2]["warnings"] == ["FLOOR_ANGLE"] and wb["limitations"] == TVR.LIMITATIONS
    assert "393.7" not in text and "3937" not in text and "3.43" not in text and wb["provenance"]["ground_truth_used"] is False
    assert doc["walls"]["wall_a"].get("method") is None                                                       # Wall A's summary is exactly the two-viewpoint document


# ------------------------------------------------------------------ the measurement service
def test_the_gate_stays_closed_for_a_three_viewpoint_wall(svc6, b3, monkeypatch):
    monkeypatch.setattr(S, "WALL_B_PROCESSING_ENABLED", False)                                          # closed Wall B gate
    s, wf = b3; assert S.ROOM_PROCESSING_ENABLED is False
    with pytest.raises(M.MeasurementError, match="not yet available"):
        svc6.measurement.start(s)
    assert s.measurement.job is None and s.measurement.result is None


def test_three_viewpoint_readiness_follows_the_workflow_and_binds_to_the_exact_inputs(svc6, b3):
    s, wf = b3; ms = use_measurement(svc6, runner=lambda ctx: None)
    rd = ms.readiness(s); assert not rd.ready and any(not i.ok for i in rd.items) and rd.summary["n_viewpoints"] == 3 and rd.input_status == "incomplete"
    wf.extract_frames(); mark_set(wf, 0, horizontals=False); rd = ms.readiness(s); assert not rd.ready and "two frame sets" in " ".join(" ".join(i.problems) for i in rd.items) and "six boundaries" not in " ".join(" ".join(i.problems) for i in rd.items)
    mark_set(wf, 1, horizontals=False); rd = ms.readiness(s)
    assert rd.ready and [i.key for i in rd.items] == ["calibration", "videos", "frames", "annotations"] and rd.summary["frame_sets_usable"] == 2 and rd.summary["videos"]["p3_right"]["frames"] > 0 and rd.binding["method"] == "three_viewpoint_wall_b"
    key = rd.binding["key"]; wf.reset(1, "V1_left_vertical"); mark(wf, 1, "V1_left_vertical", [[901.0, 200.0], [931.0, 900.0]])
    assert ms.readiness(s).binding["key"] != key                                                              # moving one boundary point changes what a result belongs to


def test_a_result_is_stored_bound_and_discarded_when_a_boundary_changes(svc6, b3):
    s, wf = b3; wf.extract_frames(); mark_set(wf, 0, horizontals=False); mark_set(wf, 1, horizontals=False)
    result = TVR.assemble(META, worker_out=worker_doc(CLEAN)); use_measurement(svc6, runner=lambda ctx: result)
    run_to_completion(svc6.measurement, s)
    assert s.measurement.result is result and svc6.measurement.current(s) is result
    with pytest.raises(M.MeasurementError, match="already measured"):
        svc6.measurement.start(s)
    wf.reset(0, "V3_right_vertical"); assert svc6.measurement.current(s) is None and any("changed" in n for n in s.measurement.notes)


def test_wall_a_result_is_untouched_by_wall_b_three_viewpoint_processing(svc6, clips6, b3):
    s, wf = b3; s.switch_wall("A"); upload(svc6, s, clips6, RV.ROOM_UPLOAD_SLOTS); s.measurement.result = A_RES; a_binding = {"key": "kept"}; s.measurement.binding = a_binding
    s.switch_wall("B"); wf.extract_frames(); mark_set(wf, 0, horizontals=False); mark_set(wf, 1, horizontals=False)
    use_measurement(svc6, runner=lambda ctx: TVR.assemble(META, worker_out=worker_doc(CLEAN))); run_to_completion(svc6.measurement, s)
    s.switch_wall("A"); assert s.measurement.result is A_RES and s.measurement.binding is a_binding and s.n_viewpoints == 2
    s.switch_wall("B"); assert TVR.is_three_view(s.measurement.result)


def test_the_default_runner_is_the_three_viewpoint_runner_only_for_a_three_viewpoint_workflow(svc6, b3, monkeypatch):
    import three_view_runner as TVRUN
    s, wf = b3; seen = []
    monkeypatch.setattr(TVRUN, "run", lambda *a, **k: seen.append(a[1]) or TVR.failed(META, [dict(code="WORKER_CRASHED", failure_class="worker_resource", stage="worker", message="x", visitor_message="x")]))
    ctx = M.RunContext("j", wf, s.room_owner, s.contract, None, {}, None, None, None, None)
    assert M.MeasurementService._default_runner(ctx)["status"] == "FAILED" and seen == [wf]


# the same per-code guarantee the two-viewpoint failure codes have (test_f8_31c2a: every defined failure message is presented without dimensions)
@pytest.mark.parametrize("code", sorted(__import__("visitor_messages").THREE_VIEW_FAILURE_TEXT))
def test_every_three_viewpoint_failure_message_is_presented_without_dimensions(code):
    import re
    import visitor_messages as VM
    r = failed_result(code); r["width_mm"] = 1000.0
    v = RVW.build_view(r); p = VM.failure_parts(code)
    assert p["known"] and v["accepted"] is False and v["width"] is None and v["height"] is None and v["per_set"] == [] and v["warnings"] == []
    assert v["failure"]["what"] == p["what"] and v["failure"]["next_step"] == p["next_step"] and v["failure"]["retry"]["label"]
    assert v["state"] in (RVW.DONE_FAILED, RVW.BUSY, RVW.NO_MEMORY, RVW.TIMEOUT, RVW.INPUTS_CHANGED) and RVW.summary_document(r)["width"] is None and v["failure"]["technical"][0]["code"] == code
    if re.search(r"re-?record|record again|record the room|record the three|upload .*again", p["next_step"], re.I):
        assert "room" in [a["step"] for a in v["failure"]["actions"]], code                 # advice to record again always offers the room-video step

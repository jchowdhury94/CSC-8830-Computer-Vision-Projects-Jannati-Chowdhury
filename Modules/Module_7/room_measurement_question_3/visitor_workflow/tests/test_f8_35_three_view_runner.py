"""Step 3: the sandboxed three-viewpoint run -- input checks, supervision (shared with the two-viewpoint worker) and result assembly.  SYNTHETIC clips; the real worker program runs on them (a
reconstruction of a made-up, unrealistic scene is expected to be REJECTED by the quality checks: the point is a typed, complete result and a clean sandbox).  No protected recording is read here."""
import dataclasses
import sys
import threading

import pytest

import room_frames as RF
import room_reconstruction as RR
import room_videos as RV
import three_view_result as TVR
import three_view_runner as TVRUN
import three_view_workflow as TVW
from test_f8_35_three_view_annotations import b3, mark_set     # noqa: F401  (fixtures + helper)
from test_f8_35_viewpoint_state import clips6, svc6, upload     # noqa: F401  (fixtures)
from test_f8_31b_frames import accepted


@pytest.fixture
def ready3(svc6, b3):
    s, wf = b3; wf.extract_frames(); mark_set(wf, 0, horizontals=False); mark_set(wf, 1, horizontals=False)
    return s, wf, dict(contract=s.contract, cal=s.f6.job.cal, sources=svc6.room_sources(s))


def run(s, wf, c, tmp_path, **kw):
    return TVRUN.run(RR, wf, s.room_owner, c["contract"], c["cal"], c["sources"], kw.pop("limits", None), workspace_root=tmp_path, **kw)


def codes(r): return [f["code"] for f in r["failures"]]


# ------------------------------------------------------------------ the real worker on synthetic clips
def test_the_real_worker_returns_a_typed_complete_result_and_leaves_nothing_behind(ready3, tmp_path):
    s, wf, c = ready3; r = run(s, wf, c, tmp_path)
    assert r["schema"] == RR.SCHEMA_RESULT and r["method"] == "three_viewpoint_wall_b" and r["status"] in ("SUCCESS", "QUESTIONABLE", "FAILED") and r["ground_truth_used"] is False
    if r["status"] == "FAILED":
        assert r["width_mm"] is None and r["failures"] and r["failures"][0]["code"] and not r["dimensions_accepted"]
    else:
        assert r["dimensions_accepted"] and r["three_view"]["frames_usable"] >= 2
    assert r["height_mm"] is None and r["processing"]["worker"]["exit_code"] == 0 and r["processing"]["stage_seconds"] is not None
    assert r["provenance"]["frame_indices"].keys() == {"0", "1"} and r["provenance"]["calibration"]["baseline_mm"] == float(s.accepted.baseline_used_mm)    # the accepted baseline, not a constant
    assert list(tmp_path.glob("f8_recon_*")) == []                                                    # the job directory was deleted


def test_the_worker_reads_the_visitors_own_staged_frames_and_the_configuration(ready3, tmp_path):
    s, wf, c = ready3; seen = {}
    def cmd(job):
        import json
        tv = json.loads((job / "three_view.json").read_text()); req = json.loads((job / "request.json").read_text())
        seen.update(keys=tv["keys"], config=tv["config"], frames=sorted(req["frames"][0]), n_png=len(list((job / "frames").glob("*.png"))), baseline=req["calibration"]["baseline_mm"],
                    ann_schema=tv["annotations"]["schema"], limits=req["limits"])
        return [sys.executable, "-c", "import sys, json, pathlib; p = pathlib.Path(sys.argv[1]); (p / 'result.json').write_text(json.dumps({'schema': 'x'}))", str(job)]
    r = run(s, wf, c, tmp_path, worker_cmd=cmd)
    assert seen["keys"] == [0, 1] and seen["frames"] == sorted(TVW.SLOTS) and seen["n_png"] == 12 and seen["ann_schema"] == "wall_b_three_viewpoint_annotations.v1"
    assert seen["config"]["share_detections"] is True and seen["config"]["subset_mutual_check"] is True and "thresholds" not in seen["config"]             # the fast matcher is the default
    assert seen["baseline"] == float(s.accepted.baseline_used_mm) and seen["limits"]["max_rss_mb"] == 3072.0                                                    # the unchanged 3072 MB request
    assert "WORKER_OUTPUT_INVALID" in codes(r) and r["status"] == "FAILED"                                                                                      # a worker document of the wrong shape is refused


def test_the_original_matcher_can_be_selected(ready3, tmp_path):
    from room_measurement import three_view_wall_b as T
    s, wf, c = ready3; seen = {}
    def cmd(job):
        import json
        seen.update(json.loads((job / "three_view.json").read_text())["config"]); return [sys.executable, "-c", "pass"]
    run(s, wf, c, tmp_path, worker_cmd=cmd, config=T.ThreeViewConfig.original_matcher())
    assert seen["share_detections"] is False and seen["subset_mutual_check"] is False


# ------------------------------------------------------------------ supervision is the shared one
@pytest.mark.parametrize("program,limits,code", [
    ("import sys; sys.exit(3)", {}, "WORKER_CRASHED"),
    ("import time; time.sleep(30)", dict(timeout_s=1.5), "WORKER_TIMEOUT"),
])
def test_worker_failures_are_typed_and_leave_no_process_or_directory(ready3, tmp_path, program, limits, code):
    s, wf, c = ready3
    r = run(s, wf, c, tmp_path, limits=RR.ReconstructionLimits(**limits), worker_cmd=lambda job: [sys.executable, "-c", program])
    assert r["status"] == "FAILED" and codes(r) == [code] and r["failures"][0]["failure_class"] == "worker_resource" and r["width_mm"] is None
    assert list(tmp_path.glob("f8_recon_*")) == []


def test_cancelling_stops_the_worker(ready3, tmp_path):
    s, wf, c = ready3; ev = threading.Event(); threading.Timer(0.7, ev.set).start()
    r = run(s, wf, c, tmp_path, cancel=ev, worker_cmd=lambda job: [sys.executable, "-c", "import time; time.sleep(30)"])
    assert codes(r) == ["WORKER_CANCELLED"]


def test_the_memory_budget_is_consulted_and_a_refusal_is_typed(ready3, tmp_path):
    import resource_budget as RB
    s, wf, c = ready3; coord = RB.ResourceCoordinator(root=tmp_path / "rb", cfg=RB.BudgetConfig(ceiling_mb=512.0), container=dict(limit_mb=None, reliable=False, reason="test"))
    r = run(s, wf, c, tmp_path, coordinator=coord, worker_cmd=lambda job: [sys.executable, "-c", "pass"])
    assert codes(r) == ["INSUFFICIENT_MEMORY"] and r["failures"][0]["failure_class"] == "worker_resource"                                                      # 512 MB cannot grant the unchanged 1024 MB minimum


# ------------------------------------------------------------------ the input checks
def test_prepare_accepts_a_complete_session_and_names_excluded_sets(ready3):
    s, wf, c = ready3; p = TVRUN.prepare(RR, wf, s.room_owner, c["contract"], c["cal"], c["sources"], RR.ReconstructionLimits(verify_video_hashes=False))
    assert p.request is not None and not p.failures and p.keys == (0, 1) and p.meta["excluded_sets"] == [2] and set(p.request.frames[0]) == set(TVW.SLOTS) and p.annotations["schema"].startswith("wall_b")


@pytest.mark.parametrize("breakit,code,words", [
    (lambda s, wf, c: wf.reset(1, "V1_left_vertical"), "TOO_FEW_USABLE_SETS", "two frame sets"),
    (lambda s, wf, c: setattr(wf, "confirmed", False), "FRAMES_NOT_EXTRACTED", "confirm"),
    (lambda s, wf, c: c["sources"].update(p3_right=None), "UPLOAD_EXPIRED", "six room videos"),
    (lambda s, wf, c: c.update(contract=dict(c["contract"], baseline_mm=c["contract"]["baseline_mm"] + 1)), "CALIBRATION_INCONSISTENT", ""),
])
def test_prepare_refuses_missing_or_stale_inputs_with_this_methods_wording(ready3, breakit, code, words):
    s, wf, c = ready3; breakit(s, wf, c)
    p = TVRUN.prepare(RR, wf, s.room_owner, c["contract"], c["cal"], c["sources"], RR.ReconstructionLimits(verify_video_hashes=False))
    assert p.request is None and code in [f["code"] for f in p.failures]
    msg = " ".join(f["visitor_message"] for f in p.failures if f["code"] == code).lower(); assert words.lower() in msg and "four videos" not in msg and "five frame sets" not in msg


def test_prepare_detects_a_replaced_video_and_another_sessions_workflow(svc6, clips6, ready3):
    s, wf, c = ready3
    svc6.remove_room_video(s, "p2_left"); upload(svc6, s, clips6, ["p2_left"])                         # the same bytes, a new upload
    src = svc6.room_sources(s); p = TVRUN.prepare(RR, wf, s.room_owner, c["contract"], c["cal"], src, RR.ReconstructionLimits(verify_video_hashes=False))
    assert "VIDEO_REPLACED" in [f["code"] for f in p.failures]
    other = TVRUN.prepare(RR, wf, "another-session", c["contract"], c["cal"], src); assert [f["code"] for f in other.failures] == ["OWNERSHIP_MISMATCH"]


def test_prepare_refuses_a_two_viewpoint_workflow(svc6, clips6):
    s = accepted(svc6); upload(svc6, s, clips6, RV.ROOM_UPLOAD_SLOTS); wf = svc6.frame_state(s)["workflow"]; assert type(wf) is RF.RoomFrameWorkflow
    p = TVRUN.prepare(RR, wf, s.room_owner, s.contract, s.f6.job.cal, svc6.room_sources(s)); assert [f["code"] for f in p.failures] == ["REQUEST_INVALID"]


def test_staging_uses_only_the_visitors_extracted_frames_never_a_video(ready3):
    s, wf, c = ready3; p = TVRUN.prepare(RR, wf, s.room_owner, c["contract"], c["cal"], c["sources"], RR.ReconstructionLimits(verify_video_hashes=False))
    paths = {rec["path"] for fs in p.request.frames for rec in fs.values()}
    assert paths and all(x.endswith(".png") and str(wf.ws.path) in x for x in paths) and not any(x.endswith((".mp4", ".mov", ".MOV")) for x in paths)

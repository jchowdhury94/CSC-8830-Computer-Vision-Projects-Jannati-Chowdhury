"""Phase D.3 safety improvements: (1) admission for the frame-extraction and upload-inspection workers, (2) three-viewpoint worker admission and the explicit CPU limit.
Tiny synthetic clips and fake workers only.  No SIFT, no matching, no real recording, no full-resolution work.  The processing gate stays closed."""
import json
import math
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import types
import uuid
from pathlib import Path

import pytest

import f8_visitor_service as S
import frame_worker as FW
import resource_budget as RB
import room_frames as RF
import room_reconstruction as RR
import three_view_runner as TVRUN
import three_view_worker as TVW
from calibration_runner import RunnerLimits
from chunk_store import MiB
from result_store import ResultStore
from room_measurement.visitor_measurement import JobLimits
from test_f8_31a_room_videos import SLOTS, _paths, accepted, cstore, good_info, make_video, put, rstore                  # noqa: F401  (fixtures and helpers)


NO_CONTAINER = dict(limit_mb=None, reliable=False, reason="test")


def table(parent_mb=200.0):
    t = {os.getpid(): dict(ppid=1, rss_mb=parent_mb, cpu_s=0.0)}
    return lambda: dict(t)


def coord(tmp_path, *, total=4096.0, parent=200.0, name="rb", reserve_floor=256.0):
    """A coordinator on a private directory with a synthetic budget (the same recipe the budget tests use)."""
    return RB.ResourceCoordinator(root=tmp_path / name, cfg=RB.BudgetConfig(ceiling_mb=total, reserve_floor_mb=reserve_floor), snapshot=table(parent), container=NO_CONTAINER)
import room_videos as RV


@pytest.fixture
def budget(tmp_path):
    """A private budget used as THE application-wide coordinator for the test (and restored afterwards)."""
    c = coord(tmp_path, total=6000.0, parent=200.0, name="budget")
    RB.set_app_coordinator(c)
    yield c
    RB.set_app_coordinator(None)


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "frames").mkdir()
    w = FW.FrameWorkspace(tmp_path / "frames"); yield w; w.close()


def limits(**kw):
    return FW.FrameJobLimits(max_file_bytes=64 * MiB, **kw)


def job_dirs(ws):
    """Everything a job left in the workspace.  Only the application's own owner marker (a regular file written by FrameWorkspace itself, see test_f8_workspace_cleanup.py) is not job output."""
    return [p for p in ws.path.iterdir() if p.name != FW.OWNER_FILE]


# ================================================================ 1a. frame extraction takes a slot, releases it, and is refused next to a heavy job
def test_a_frame_job_holds_a_frames_slot_while_it_runs_and_releases_it_after(tmp_path, ws, budget, monkeypatch):
    clip = make_video(tmp_path / "c.mp4"); started, release, seen = threading.Event(), threading.Event(), []
    real = FW._spawn

    def slow(*a, **k):
        seen.append([r["kind"] for r in budget.state()["reservations"]]); started.set(); release.wait(20); return real(*a, **k)
    monkeypatch.setattr(FW, "_spawn", slow); out = {}
    t = threading.Thread(target=lambda: out.update(FW.run_frame_job(ws, clip, limits(), preview_indices=[0])))
    t.start(); assert started.wait(20)
    assert seen == [["frames"]]                                                                  # the slot is held while the worker runs
    release.set(); t.join(30)
    assert out["written"] and budget.state()["reservations"] == []                               # and released afterwards


def test_a_frame_job_is_refused_before_anything_starts_while_a_reconstruction_runs(tmp_path, ws, budget, monkeypatch):
    clip = make_video(tmp_path / "c.mp4")
    monkeypatch.setattr(FW, "_spawn", lambda *a, **k: pytest.fail("a decoder was started next to a reconstruction"))
    h = budget.reserve("reconstruction", "o", 1500.0, 100.0); assert h.ok
    with pytest.raises(FW.FrameJobError) as e:
        FW.run_frame_job(ws, clip, limits(), preview_indices=[0], wait_s=0.0)
    assert e.value.code == "busy" and "try again in a minute" in e.value.message
    assert job_dirs(ws) == []                                                                    # no job directory was even created
    h.reservation.release()
    assert budget.state()["reservations"] == []


def test_the_default_is_the_application_wide_coordinator_so_admission_cannot_be_skipped(tmp_path, ws, budget, monkeypatch):
    clip = make_video(tmp_path / "c.mp4")
    h = budget.reserve("calibration", "o", 1500.0, 100.0); assert h.ok
    with pytest.raises(FW.FrameJobError) as e:                                                   # no coordinator argument: the app-wide one still applies
        FW.run_frame_job(ws, clip, limits(), preview_indices=[0], wait_s=0.0)
    assert e.value.code == "busy"
    h.reservation.release()


def test_failures_timeouts_and_crashes_release_the_slot_and_leave_no_output(tmp_path, ws, budget):
    clip = make_video(tmp_path / "c.mp4"); marker = "MK" + uuid.uuid4().hex[:10]
    crash = [sys.executable, "-c", "import sys; sys.exit(3)"]
    hang = [sys.executable, "-c", f"import time; time.sleep(300)  # {marker}"]
    with pytest.raises(FW.FrameJobError) as e:
        FW.run_frame_job(ws, clip, limits(), preview_indices=[0], worker_cmd=crash)
    assert e.value.code == "worker_failed" and budget.state()["reservations"] == [] and job_dirs(ws) == []
    with pytest.raises(FW.FrameJobError) as e:
        FW.run_frame_job(ws, clip, limits(timeout_s=1.0), preview_indices=[0], worker_cmd=hang)
    assert e.value.code == "timeout" and budget.state()["reservations"] == [] and job_dirs(ws) == []
    time.sleep(0.3); assert subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout.split() == []      # the timed-out worker's process group is gone
    with pytest.raises(FW.FrameJobError) as e:
        FW.run_frame_job(ws, tmp_path / "missing.mp4", limits(), preview_indices=[0])
    assert e.value.code == "video_missing" and budget.state()["reservations"] == []              # a bad request holds no slot afterwards
    ok = FW.run_frame_job(ws, clip, limits(), preview_indices=[0])
    assert ok["written"] and budget.state()["reservations"] == []                                # and a good job still works after all of that


def test_two_frame_jobs_run_together_and_a_third_is_refused(tmp_path, ws, budget, monkeypatch):
    clip = make_video(tmp_path / "c.mp4"); gate, release, running = threading.Barrier(3), threading.Event(), []
    real = FW._spawn

    def slow(*a, **k):
        running.append(1); release.wait(20); return real(*a, **k)
    monkeypatch.setattr(FW, "_spawn", slow); results = []

    def go():
        try:
            results.append(FW.run_frame_job(ws, clip, limits(), preview_indices=[0], wait_s=0.0))
        except FW.FrameJobError as e:
            results.append(e)
    a = [threading.Thread(target=go) for _ in range(2)]; [t.start() for t in a]
    deadline = time.time() + 20
    while len(running) < 2 and time.time() < deadline:
        time.sleep(0.05)
    assert len(running) == 2
    third = None
    try:
        FW.run_frame_job(ws, clip, limits(), preview_indices=[0], wait_s=0.0)
    except FW.FrameJobError as e:
        third = e
    assert third is not None and third.code == "busy"
    release.set(); [t.join(30) for t in a]
    assert sum(isinstance(r, dict) for r in results) == 2 and budget.state()["reservations"] == []


def test_the_workflow_turns_a_busy_refusal_into_a_visitor_message(tmp_path, ws, budget):
    wf = RF.RoomFrameWorkflow.__new__(RF.RoomFrameWorkflow)
    wf.source = lambda slot: types.SimpleNamespace(path=str(tmp_path / "x.mp4")); wf.ws = ws; wf.job_limits = limits(); wf.worker_cmd = None; wf.log = []
    h = budget.reserve("reconstruction", "o", 1500.0, 100.0); assert h.ok
    with pytest.raises(RF.WorkflowError) as e:
        wf._job("preview", "p1_left", preview_indices=[0])
    assert "busy" in str(e.value).lower() and wf.log[-1]["error"] == "busy"
    h.reservation.release()


# ================================================================ 1b. upload inspection: gated, and a busy refusal is never a verdict on the video
@pytest.fixture
def real_inspect_svc(cstore, rstore, tmp_path, budget, monkeypatch):
    """A service WITHOUT an injected inspector (so the default, gated path is used) whose real inspector is replaced by a spy."""
    import video_check
    calls = []

    def spy(path):
        calls.append([r["kind"] for r in budget.state()["reservations"]])
        return dict(good_info(1920, 1080), size=Path(path).stat().st_size)
    monkeypatch.setattr(video_check, "inspect_video", spy)
    svc = S.VisitorCalibrationService(cstore, ResultStore(tmp_path / "results"), RunnerLimits(), room_store=rstore)
    return svc, calls


def test_the_default_inspection_takes_a_validation_slot_and_releases_it(real_inspect_svc, rstore, tmp_path, budget):
    svc, calls = real_inspect_svc
    s = accepted(svc); put(rstore, s.owner, "p1_left", make_video(tmp_path / "a.mp4").read_bytes())
    up = svc.room_uploads(s)
    assert calls == [["validation"]] and up["p1_left"]["inspection"] is not None                 # inspected while holding a validation slot
    assert budget.state()["reservations"] == []                                                  # released
    svc.room_uploads(s); assert len(calls) == 1                                                  # a real verdict is cached (inspected once), as before


def test_a_busy_server_leaves_the_slot_pending_and_uncached_and_it_is_checked_later(real_inspect_svc, rstore, tmp_path, budget, monkeypatch):
    svc, calls = real_inspect_svc
    real_acquire = RB.acquire_light
    monkeypatch.setattr(RB, "acquire_light", lambda kind, owner="", coordinator=None, wait_s=0.0, poll_s=0.2: real_acquire(kind, owner, coordinator, 0.0, poll_s))       # no 3 s wait in the test
    s = accepted(svc); put(rstore, s.owner, "p1_left", make_video(tmp_path / "a.mp4").read_bytes())
    h = budget.reserve("reconstruction", "other", 1500.0, 100.0); assert h.ok
    up = svc.room_uploads(s)
    assert calls == [] and up["p1_left"]["inspection"] is None and s.room_inspect == {}          # not inspected, no verdict, nothing cached
    rep = svc.room_report(s); assert rep.slots["p1_left"].status == RV.PENDING and not rep.slots["p1_left"].errors      # shown as "being checked", never as an error
    h.reservation.release()
    up = svc.room_uploads(s)                                                                     # the next page run checks it for real
    assert len(calls) == 1 and up["p1_left"]["inspection"] is not None and len(s.room_inspect) == 1


def test_an_injected_inspector_is_not_gated(cstore, rstore, tmp_path, budget):
    svc = S.VisitorCalibrationService(cstore, ResultStore(tmp_path / "results"), RunnerLimits(), room_store=rstore, inspector=lambda p: good_info(1920, 1080))
    h = budget.reserve("reconstruction", "o", 1500.0, 100.0); assert h.ok
    s = accepted(svc); put(rstore, s.owner, "p1_left", make_video(tmp_path / "a.mp4").read_bytes())
    assert svc.room_uploads(s)["p1_left"]["inspection"] is not None                              # tests keep their seam: nothing changes for them
    h.reservation.release()


# ================================================================ 2a. the three-viewpoint admission minimum
def test_the_three_view_minimum_is_the_measured_cloud_peak_and_only_raises_a_limit():
    assert TVRUN.THREE_VIEW_MIN_TREE_MB == 2450.0                                                # 2418.1 MB measured, rounded up
    base = RR.ReconstructionLimits()
    assert base.min_tree_rss_mb == 1024.0                                                        # the shared default (Wall A, two-viewpoint Wall B) is unchanged
    tv = TVRUN.three_view_limits(RR, base)
    assert tv.min_tree_rss_mb == 2450.0 and tv.timeout_s == 300.0 and tv.cpu_limit_s == 1800.0 and tv.max_tree_rss_mb == base.max_tree_rss_mb     # wall timeout and CPU limit untouched
    higher = RR.ReconstructionLimits(min_tree_rss_mb=3000.0)
    assert TVRUN.three_view_limits(RR, higher).min_tree_rss_mb == 3000.0                          # never lowered
    assert TVRUN.three_view_limits(RR, None).min_tree_rss_mb == 2450.0                            # the default object is used when none is given


@pytest.mark.parametrize("value,expected", [("3000", 3000.0), ("1024", 1024.0), ("8192", 8192.0), ("500", 2450.0), ("9000", 2450.0), ("x", 2450.0), ("", 2450.0), ("nan", 2450.0)])
def test_the_three_view_minimum_is_configurable_and_invalid_values_are_ignored(value, expected):
    assert TVRUN.three_view_min_mb({"F8_THREE_VIEW_MIN_MB": value}) == expected
    assert TVRUN.three_view_min_mb({}) == 2450.0


def test_run_applies_the_three_view_minimum_even_when_the_service_passes_its_own_limits(monkeypatch):
    """``f8_measurement`` hands the runner its own (default) limits object; the minimum must still be raised inside ``TVRUN.run`` for three viewpoints, and for them only."""
    captured = {}
    monkeypatch.setattr(TVRUN, "prepare", lambda RR_, wf, owner, contract, cal, sources, limits=None, min_usable=None: TVRUN.Prepared(TVRUN.StagedRequest("owner", {"x": 1}, ()), [], {}, {"annotations": 1}, (0, 1)))

    def fake_run_worker(request, limits, **kw):
        captured["limits"] = limits
        return None, RR.failure(RR.Code.INSUFFICIENT_MEMORY, "not enough", stage="admission", minimum_mb=limits.min_tree_rss_mb, available_mb=900.0), dict(budget={})
    monkeypatch.setattr(RR, "_run_worker", fake_run_worker)
    service_limits = RR.ReconstructionLimits()
    res = TVRUN.run(RR, object(), "owner", {}, object(), {}, service_limits)
    assert captured["limits"].min_tree_rss_mb == 2450.0 and service_limits.min_tree_rss_mb == 1024.0         # raised for this job; the service's own object is untouched
    assert captured["limits"].timeout_s == 300.0 and captured["limits"].cpu_limit_s == 1800.0
    failure = res["failures"][0] if "failures" in res else res["result"]["failures"][0]
    assert failure["code"] == "INSUFFICIENT_MEMORY" and "2450 MB are needed" in failure["visitor_message"]


def test_a_memory_ceiling_below_the_three_view_minimum_is_refused_before_any_reservation(monkeypatch):
    """Fail fast: ``max_tree_rss_mb`` below the 2450 MB minimum used to be admitted with a smaller grant (``min(min, max)``) and then stopped mid-run."""
    calls = []
    monkeypatch.setattr(TVRUN, "prepare", lambda *a, **k: calls.append("prepare"))
    monkeypatch.setattr(RR, "_run_worker", lambda *a, **k: calls.append("run_worker"))
    res = TVRUN.run(RR, object(), "owner", {}, object(), {}, RR.ReconstructionLimits(max_tree_rss_mb=2000.0))
    failure = res["failures"][0] if "failures" in res else res["result"]["failures"][0]
    assert failure["code"] == "INSUFFICIENT_MEMORY" and failure["stage"] == "admission" and failure["minimum_mb"] == 2450.0 and failure["ceiling_mb"] == 2000.0
    assert "not a problem with your videos" in failure["visitor_message"] and calls == []           # nothing prepared, nothing reserved


def test_the_default_limits_get_past_the_three_view_ceiling_check(monkeypatch):
    base = RR.ReconstructionLimits()
    assert base.max_tree_rss_mb == 3072.0 and TVRUN.three_view_limits(RR, base).max_tree_rss_mb >= TVRUN.three_view_limits(RR, base).min_tree_rss_mb   # the default is admitted
    calls = []
    monkeypatch.setattr(TVRUN, "prepare", lambda *a, **k: calls.append("prepare") or (_ for _ in ()).throw(RuntimeError("stop")))
    TVRUN.run(RR, object(), "owner", {}, object(), {}, base); assert calls == ["prepare"]            # the default limits get past the new check


def staged():
    return TVRUN.StagedRequest("owner", {"calibration_id": "x"}, ())


def worker_cmd(tmp_path, body):
    script = tmp_path / f"w{uuid.uuid4().hex[:6]}.py"; script.write_text(textwrap.dedent(body))
    return lambda job: [sys.executable, str(script), str(job)]


def test_insufficient_memory_is_refused_before_anything_starts_with_a_clear_message(tmp_path):
    c = coord(tmp_path, total=2600.0, parent=300.0)                                              # usable 2340 - parent 300 = about 2040 MB free: below 2450
    root = tmp_path / "wsroot"; root.mkdir()
    out, fail, info = RR._run_worker(staged(), TVRUN.three_view_limits(RR), worker_cmd=worker_cmd(tmp_path, "raise SystemExit('must not run')"), workspace_root=root, coordinator=c)
    assert out is None and fail["code"] == "INSUFFICIENT_MEMORY" and fail["stage"] == "admission" and fail["minimum_mb"] == 2450.0
    assert list(root.iterdir()) == [] and c.state()["reservations"] == []                         # nothing staged, nothing reserved
    fail = TVRUN._resource_wording(fail, info)
    assert "three viewpoints" in fail["visitor_message"] and "2450 MB are needed" in fail["visitor_message"] and "not a problem with your videos" in fail["visitor_message"]
    assert fail["code"] == "INSUFFICIENT_MEMORY" and "available under the memory budget" in fail["message"]      # the typed code and the technical message are unchanged


def test_unavailable_cgroup_information_fails_closed_and_says_why(tmp_path):
    c = RB.ResourceCoordinator(root=tmp_path / "rb", cfg=RB.BudgetConfig(), snapshot=table(250.0), container=NO_CONTAINER)    # no cgroup, no operator ceiling
    out, fail, info = RR._run_worker(staged(), TVRUN.three_view_limits(RR), worker_cmd=worker_cmd(tmp_path, "raise SystemExit('must not run')"), coordinator=c)
    assert fail["code"] == "INSUFFICIENT_MEMORY" and "could not be determined" in fail["message"] and "conservative fallback" in fail["message"]
    assert "could not determine its own memory size" in TVRUN._resource_wording(fail, info)["visitor_message"]
    # an operator who states the container size explicitly is trusted, exactly as before
    c2 = RB.ResourceCoordinator(root=tmp_path / "rb2", cfg=RB.BudgetConfig(ceiling_mb=3072.0), snapshot=table(250.0), container=NO_CONTAINER)
    ok = worker_cmd(tmp_path, "import json, sys; from pathlib import Path; (Path(sys.argv[1]) / 'result.json').write_text(json.dumps(dict(schema='x')))")
    out, fail, info = RR._run_worker(staged(), TVRUN.three_view_limits(RR), worker_cmd=ok, coordinator=c2)
    assert fail is None and out == dict(schema="x") and info["limit_mb"] >= 2450.0 and c2.state()["reservations"] == []


def test_a_busy_server_gets_wording_that_names_the_real_cause(tmp_path):
    c = coord(tmp_path, total=6000.0)
    l = RB.acquire_light("frames", "o", c)
    out, fail, info = RR._run_worker(staged(), TVRUN.three_view_limits(RR), worker_cmd=worker_cmd(tmp_path, "raise SystemExit('must not run')"), coordinator=c)
    assert fail["code"] == "WORKER_BUSY" and fail["busy_with"] == "light" and "frame-extraction or video-check" in fail["message"]
    assert "another visitor" in TVRUN._resource_wording(fail, info)["visitor_message"]
    l.release()
    assert RR._run_worker(staged(), TVRUN.three_view_limits(RR), worker_cmd=worker_cmd(tmp_path, "import json, sys; from pathlib import Path; (Path(sys.argv[1]) / 'result.json').write_text('{}')"), coordinator=c)[1] is None


def test_the_reservation_and_the_workspace_are_cleaned_up_after_a_crash_and_a_cpu_kill(tmp_path):
    c = coord(tmp_path, total=6000.0); root = tmp_path / "wsroot"; root.mkdir(); lim = TVRUN.three_view_limits(RR)
    crash = worker_cmd(tmp_path, "import sys; sys.exit(7)")
    out, fail, _ = RR._run_worker(staged(), lim, worker_cmd=crash, workspace_root=root, coordinator=c)
    assert fail["code"] == "WORKER_CRASHED" and list(root.iterdir()) == [] and c.state()["reservations"] == []
    cpu = worker_cmd(tmp_path, "import os, signal; os.kill(os.getpid(), signal.SIGXCPU)")
    out, fail, _ = RR._run_worker(staged(), lim, worker_cmd=cpu, workspace_root=root, coordinator=c)
    assert fail["code"] == "WORKER_CRASHED" and "1800 CPU-second limit" in fail["message"] and fail["cpu_limit_s"] == 1800.0     # named, not an unexplained crash
    assert list(root.iterdir()) == [] and c.state()["reservations"] == []


# ================================================================ 2b. the explicit CPU limit
def test_the_cpu_limit_defaults_to_1800_and_is_configurable_with_validation():
    assert RR.ReconstructionLimits().cpu_limit_s == 1800.0 and RR.DEFAULT_CPU_LIMIT_S == 1800.0
    assert RR.default_cpu_limit_s({"F8_RECON_CPU_S": "2400"}) == 2400.0 and RR.default_cpu_limit_s({}) == 1800.0
    for bad in ("0", "59", "7201", "-5", "abc", "nan", "inf", ""):
        assert RR.default_cpu_limit_s({"F8_RECON_CPU_S": bad}) == 1800.0                           # invalid values are ignored, never clamped
    assert RR.ReconstructionLimits(cpu_limit_s=900.0).cpu_limit_s == 900.0
    assert RR.ReconstructionLimits().timeout_s == 300.0                                          # the wall timeout is unchanged


def test_the_supervisor_writes_the_cpu_limit_into_the_workers_request(tmp_path):
    c = coord(tmp_path, total=6000.0); seen = tmp_path / "seen.json"
    cmd = worker_cmd(tmp_path, f"""
        import json, shutil, sys
        from pathlib import Path
        job = Path(sys.argv[1]); shutil.copy(job / "request.json", {str(seen)!r}); (job / "result.json").write_text("{{}}")
    """)
    assert RR._run_worker(staged(), RR.ReconstructionLimits(), worker_cmd=cmd, coordinator=c)[1] is None
    assert json.loads(seen.read_text())["limits"]["cpu_limit_s"] == 1800.0
    assert RR._run_worker(staged(), RR.ReconstructionLimits(cpu_limit_s=900.0), worker_cmd=cmd, coordinator=c)[1] is None
    lims = json.loads(seen.read_text())["limits"]
    assert lims["cpu_limit_s"] == 900.0 and {"max_rss_mb", "watchdog_s", "exit_code"} <= set(lims)   # the existing worker limits are all still there


def test_the_worker_applies_the_limit_it_was_given_and_never_a_bad_one():
    assert TVW.cpu_limit_from({"cpu_limit_s": 900.0}, {}) == 900
    assert TVW.cpu_limit_from({"cpu_limit_s": 900.0}, {"F8_RECON_CPU_S": "2400"}) == 900          # the supervisor's value wins
    assert TVW.cpu_limit_from({}, {"F8_RECON_CPU_S": "2400"}) == 2400                              # hand-run worker: the environment still works
    assert TVW.cpu_limit_from(None, {}) == 1800
    for bad in (0, 59, 7201, -1, "x", float("nan"), float("inf"), None):
        assert TVW.cpu_limit_from({"cpu_limit_s": bad}, {}) == 1800, bad                         # out-of-range / non-finite: ignored, never clamped


def test_the_worker_reads_the_limit_before_it_starts_work():
    src = (Path(TVW.__file__)).read_text()
    assert src.index("cpu = cpu_limit_from(limits)") < src.index("out = run_job(job)")           # applied before any processing
    assert "1800" in src and "7200" in src


# ================================================================ 3. what stays exactly as it was
def test_wall_a_and_two_viewpoint_defaults_are_untouched():
    base = RR.ReconstructionLimits()
    assert (base.timeout_s, base.max_tree_rss_mb, base.min_tree_rss_mb, base.container_fraction) == (300.0, 3072.0, 1024.0, 0.90)
    c = RB.ResourceCoordinator(root=None, cfg=RB.BudgetConfig(ceiling_mb=3072.0), snapshot=table(200.0), container=NO_CONTAINER)
    a = c.reserve("reconstruction", "o", base.max_tree_rss_mb, min(base.min_tree_rss_mb, base.max_tree_rss_mb))
    assert a.ok and 1024.0 <= a.granted_mb <= 3072.0                                              # the two-viewpoint minimum still admits a job the three-viewpoint minimum would refuse
    assert TVRUN.three_view_limits(RR, base).min_tree_rss_mb == 2450.0 and base.min_tree_rss_mb == 1024.0       # (frozen: the original object is not modified)


def test_the_production_gate_stays_closed():
    assert S.ROOM_PROCESSING_ENABLED is False

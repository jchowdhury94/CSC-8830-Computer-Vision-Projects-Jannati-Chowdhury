"""F8.31C-1C: visitor wording, advisory explanations, the shared memory budget inside the reconstruction backend, upload retention during a job, visitor isolation and the closed gate.

SYNTHETIC only: fake workers, fake cgroup readings, tiny byte payloads as 'videos'.  No original recording, no real reconstruction (that is covered by the F8.31C-1 file)."""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

import f8_visitor_service as FS
import resource_budget as RB
import room_frames as RF
import room_reconstruction as RR
import room_videos as RV
import synthetic_room_render as SR
import visitor_messages as VM
from chunk_store import ChunkStore, Limits, MiB, UploadError
from room_measurement import visitor_method_j_adapter as A
from test_f8_31c1_reconstruction import call_params

HERE = Path(__file__).resolve().parents[1]
FAKE_PY = lambda code: (lambda job: [sys.executable, "-c", code])   # noqa: E731
NO_CONTAINER = dict(limit_mb=None, reliable=False, reason="test")
WORKER_OK = {"schema": "f8.reconstruction_worker/1", "failures": [], "sets": [], "timings": {}, "peak_rss_mb": 120.0,
             "adapter": dict(outcome="SUCCESS", failures=[], warnings=[], notes=[], width_mm=3000.0, height_mm=2400.0, diagnostics={})}


def coord(tmp_path, total=6000.0, parent=200.0, name="rb2", container=NO_CONTAINER):
    snap = lambda: {os.getpid(): dict(ppid=1, rss_mb=parent, cpu_s=0.0)}               # noqa: E731
    return RB.ResourceCoordinator(root=tmp_path / name, cfg=RB.BudgetConfig(ceiling_mb=total), snapshot=snap, container=container)


@pytest.fixture
def request_(tmp_path):
    """A minimal request (frames are placeholder files: the fake workers never decode them)."""
    rows, table = [], []
    row, tb = {}, {}
    for k in RF.VIEW_KEYS:
        p = tmp_path / f"{k}.png"; p.write_bytes(b"png-placeholder"); row[k] = dict(path=str(p), index=1, fingerprint="f" * 64); tb[k] = 1
    return RR.ReconstructionRequest("owner-digest", call_params(SR.make_context()), (row,), ({},), {}, (), (tb,))


def run(req, tmp_path, code, *, coordinator=None, **lim):
    root = tmp_path / "ws"; root.mkdir(exist_ok=True)
    lim.setdefault("poll_s", 0.05); lim.setdefault("kill_grace_s", 1.0)
    r = RR.run_request(req, RR.ReconstructionLimits(**lim), worker_cmd=FAKE_PY(code) if isinstance(code, str) else code, workspace_root=root, coordinator=coordinator)
    assert not list(root.iterdir()), "the job directory must be removed"
    return r


def writes(doc):
    return lambda job: [sys.executable, "-c", f"import pathlib\npathlib.Path({str(job)!r}, 'result.json').write_text({json.dumps(doc)!r})"]


# ================================================================ failure wording
def all_defined_codes():
    out = {c.value for c in RR.Code} | {e.value for e in A.FailureReason} | {"width_not_finite", "width_not_positive", "height_not_finite", "height_not_positive"}
    return out


def test_every_defined_failure_code_has_visitor_wording():
    missing = sorted(all_defined_codes() - set(VM.FAILURE_TEXT))
    assert missing == [], missing
    for code in all_defined_codes():
        p = VM.failure_parts(code); assert p["known"] and p["what"].strip() and p["text"].startswith(p["what"]) and RR.VISITOR_MESSAGES[code] == p["text"]


def test_the_fifteen_codes_that_used_to_fall_back_to_the_technical_message_are_now_worded():
    fifteen = ["INPUT_NO_FRAME_SETS", "INPUT_IMAGE_SIZE_MISMATCH", "INPUT_INVALID_CALIBRATION", "LINE_RETARGET_FAILED", "BASELINE_CHANGED_UNLABELLED", "DATASET_BUILD_FAILED", "POSE_SOLVER_EXCEPTION",
               "POSE_NON_FINITE", "TRIANGULATION_FAILED", "FLOOR_PLANE_FAILED", "REGISTRATION_FAILED", "WALL_RECONSTRUCTION_FAILED", "GEOMETRY_FAILED", "WIDTH_UNAVAILABLE", "HEIGHT_UNAVAILABLE"]
    for c in fifteen:
        f = RR.failure(c, "ValueError: tensor(0x7f) at /tmp/job/frames/s0_AL.png", stage="geometry")
        assert f["visitor_message"] == VM.failure_text(c) and "ValueError" not in f["visitor_message"] and "/tmp" not in f["visitor_message"] and f["message"].startswith("ValueError") and f["code"] == c


def test_wording_is_plain_has_no_internals_and_never_sounds_like_a_result():
    leak = re.compile(r"\w+Error\b|Exception|Traceback|\.py\b|/tmp|0x[0-9a-f]|NaN|\bnan\b|\binf\b|ndarray|numpy|cv2|RANSAC|SIFT|inlier|px\b|mm\b|stderr|exit code")
    for code, (what, nxt) in VM.FAILURE_TEXT.items():
        t = f"{what} {nxt}"
        assert not leak.search(t), (code, t)
        assert what.endswith(".") and (nxt == "" or nxt.endswith(".")), code
        assert not re.search(r"\bsuccess|\bmeasured successfully|\baccepted result|\bcompleted\b", t, re.I), (code, t)
        assert nxt or code in ("WORKER_CANCELLED",), f"{code} should offer a next step"


def test_the_requested_situations_say_what_failed_and_what_to_do():
    t = VM.failure_text
    assert "texture" in t("FEATURE_MATCHING_FAILED") and "Re-record" in t("FEATURE_MATCHING_FAILED")
    assert "floor" in t("FLOOR_PLANE_FAILED") and "clearly visible" in t("FLOOR_PLANE_FAILED")
    assert "viewpoints could not be joined" in t("REGISTRATION_FAILED") and "shared background" in t("REGISTRATION_FAILED")
    assert "wall edges" in t("WALL_RECONSTRUCTION_FAILED") and "Mark the boundaries again" in t("WALL_RECONSTRUCTION_FAILED")
    for c in ("CALIBRATION_CHANGED", "CALIBRATION_INCONSISTENT", "CALIBRATION_VIDEO_SIZE_MISMATCH", "INPUT_INVALID_CALIBRATION"):
        assert "alibrat" in t(c) or "calibrated" in t(c), c
    assert "expired" in t("UPLOAD_EXPIRED") and "Upload the four room videos again" in t("UPLOAD_EXPIRED")
    for c in ("ANNOTATION_MISSING", "ANNOTATION_STALE", "ANNOTATION_INVALID", "INPUT_LINES_MISSING"):
        assert "boundar" in t(c) or "line" in t(c), c
    assert "too long" in t("WORKER_TIMEOUT") and "memory" in t("WORKER_MEMORY_LIMIT") and "Another measurement is running" in t("WORKER_BUSY")
    assert "free memory" in t("INSUFFICIENT_MEMORY") and t("INSUFFICIENT_MEMORY") != t("WORKER_BUSY")


def test_unknown_codes_get_generic_wording_never_the_technical_message():
    p = VM.failure_parts("SOMETHING_NEW"); assert not p["known"] and p["what"] == VM.GENERIC_FAILURE[0] and "Nothing was concluded" in p["text"]
    f = RR.failure("SOMETHING_NEW", "RuntimeError: secret path /home/x/y", stage="x")
    assert "RuntimeError" not in f["visitor_message"] and "/home" not in f["visitor_message"] and f["code"] == "SOMETHING_NEW" and "RuntimeError" in f["message"]
    assert VM.failure_parts(None)["known"] is False and VM.failure_parts(12345)["text"]


def test_a_method_j_failure_is_never_reported_as_a_measurement(request_, tmp_path):
    doc = dict(WORKER_OK, adapter=dict(outcome="FAILED", failures=[dict(code="FLOOR_PLANE_FAILED", message="LinAlgError: singular", stage="geometry")], warnings=[], notes=[], width_mm=3000.0, height_mm=2400.0,
                                       diagnostics=dict(per_set=[dict(frame_set=0, width_mm=2999.0, height_mm=2400.0)], set_failures=[dict(frame_set=0, code="FLOOR_PLANE_FAILED")])))
    r = run(request_, tmp_path, writes(doc), coordinator=coord(tmp_path))
    f = r["failures"][0]
    assert r["status"] == "FAILED" and r["width_mm"] is None and r["height_mm"] is None and r["dimensions_accepted"] is False and f["failure_class"] == "method_j"
    assert f["visitor_message"] == VM.failure_text("FLOOR_PLANE_FAILED") and f["message"] == "LinAlgError: singular" and r["visitor_notice"] == f["visitor_message"]
    assert "2999" not in json.dumps(r["per_frame_set"]) and "per_set" not in r["diagnostics"]


# ================================================================ advisory warnings
def test_all_seven_advisory_flags_are_explained_in_three_parts():
    flags = {q.value for q in A.QualityFlag}; assert len(flags) == 7 and flags == set(VM.WARNING_TEXT)
    for c in flags:
        p = VM.warning_parts(c)
        assert p["known"] and all(p[k].strip().endswith(".") for k in ("meaning", "reliability", "action")) and isinstance(p["visitor_can_act"], bool)
        assert p["meaning"] != p["reliability"] != p["action"]
    assert all(VM.warning_parts(c)["visitor_can_act"] for c in flags)


def test_warning_wording_claims_no_accuracy_and_no_confidence_interval():
    bad = re.compile(r"confidence interval|guarantee|\bcertainty|\baccurate|\bexact\b|\bcorrect\b|\bverified\b|within \d|±|%|error of", re.I)
    for c in list(VM.WARNING_TEXT) + ["UNKNOWN_FLAG"]:
        assert not bad.search(VM.warning_parts(c)["text"]), c
    assert "no confidence interval" in VM.NO_CLAIMS and "no guaranteed accuracy" in VM.NO_CLAIMS and "no accuracy is claimed" in VM.QUESTIONABLE_SUMMARY
    u = VM.warning_parts("NOT_A_FLAG"); assert not u["known"] and not u["visitor_can_act"] and u["text"]


def test_warnings_are_attached_without_changing_numbers_status_or_decisions(request_, tmp_path):
    flags = [dict(code=q.value, message=f"technical {q.value}", stage="x") for q in A.QualityFlag]
    plain = run(request_, tmp_path, writes(WORKER_OK), coordinator=coord(tmp_path))
    warned = run(request_, tmp_path, writes(dict(WORKER_OK, adapter=dict(WORKER_OK["adapter"], outcome="QUESTIONABLE", warnings=flags))), coordinator=coord(tmp_path))
    assert plain["status"] == "SUCCESS" and warned["status"] == "QUESTIONABLE" and not warned["failures"]
    assert (warned["width_mm"], warned["height_mm"]) == (plain["width_mm"], plain["height_mm"]) == (3000.0, 2400.0) and warned["dimensions_accepted"] is True
    assert len(warned["warnings"]) == 7 and all(w["advisory"] and w["visitor_explanation"]["known"] and w["message"].startswith("technical") for w in warned["warnings"])
    assert warned["visitor_notice"].startswith(VM.QUESTIONABLE_SUMMARY) and plain["visitor_notice"] == VM.NO_CLAIMS and plain["uncertainty"]["status"] == "not_computed"
    assert "confidence" not in json.dumps({k: v for k, v in warned.items() if k not in ("visitor_notice", "uncertainty", "warnings")}).lower().replace("no confidence", "")


def test_a_warning_can_never_turn_a_method_j_failure_into_success(request_, tmp_path):
    doc = dict(WORKER_OK, adapter=dict(WORKER_OK["adapter"], outcome="FAILED", failures=[dict(code="NO_USABLE_FRAME_SETS", message="m", stage="geometry")],
                                       warnings=[dict(code="DIMENSION_PARTIAL", message="m", stage="dimensions")]))
    r = run(request_, tmp_path, writes(doc), coordinator=coord(tmp_path))
    assert r["status"] == "FAILED" and r["width_mm"] is None and not r["dimensions_accepted"]


# ================================================================ shared memory budget inside the backend
def test_busy_and_insufficient_memory_are_distinct_typed_outcomes(request_, tmp_path):
    c = coord(tmp_path); held = c.reserve("reconstruction", "other", 800.0, 100.0); assert held.ok
    busy = run(request_, tmp_path, "pass", coordinator=c)
    assert busy["failures"][0]["code"] == "WORKER_BUSY" and busy["failure_class"] == "worker_resource" and busy["status"] == "FAILED" and busy["width_mm"] is None
    held.reservation.release()
    tiny = coord(tmp_path, total=900.0, name="rb3")
    low = run(request_, tmp_path, "pass", coordinator=tiny)
    f = low["failures"][0]
    assert f["code"] == "INSUFFICIENT_MEMORY" and f["failure_class"] == "worker_resource" and f["visitor_message"] == VM.failure_text("INSUFFICIENT_MEMORY") and f["visitor_message"] != busy["failures"][0]["visitor_message"]
    assert low["processing"]["worker"]["budget"]["outcome"] == RB.INSUFFICIENT_MEMORY and low["processing"]["worker"]["budget"]["granted_mb"] is None and tiny.state()["reservations"] == []


def test_calibration_and_reconstruction_share_one_budget(request_, tmp_path):
    c = coord(tmp_path, total=3000.0)                                                  # usable about 2500 for workers
    cal = c.reserve("calibration", "visitor-1", 2200.0, 1024.0); assert cal.ok
    r = run(request_, tmp_path, "pass", coordinator=c)                                  # a reconstruction cannot take memory the calibration already holds
    assert r["failures"][0]["code"] == "INSUFFICIENT_MEMORY" and r["processing"]["worker"]["budget"]["reserved_mb_before"] == pytest.approx(2200.0)
    cal.reservation.release()
    r = run(request_, tmp_path, writes(WORKER_OK), coordinator=c); assert r["status"] == "SUCCESS" and c.state()["reservations"] == []


def test_the_enforced_ceiling_is_the_grant_not_the_requested_limit(request_, tmp_path):
    c = coord(tmp_path, total=3000.0, parent=200.0)                                     # grant = 3000 - 300 - 200 = 2500 (< the 3072 default request)
    r = run(request_, tmp_path, writes(WORKER_OK), coordinator=c)
    w = r["processing"]["worker"]
    assert abs(w["limit_mb"] - 2500.0) < 1 and w["limit_mb"] < w["requested_limit_mb"] == 3072.0 and w["budget"]["granted_mb"] == w["limit_mb"] and w["budget"]["source"].startswith("configured_ceiling")
    assert w["budget"]["limitation"] and w["enforcement"]["hard_os_limit"] is False and w["limit_mb"] <= 3000.0


def test_a_worker_above_the_grant_is_killed_with_its_children_and_the_reservation_released(request_, tmp_path):
    c = coord(tmp_path, total=1800.0, parent=100.0)                                     # usable 1800-256(10% = 180 -> floor 256)-100 = ~1444
    pids = tmp_path / "pids"
    code = (f"import subprocess, sys, os, time\nc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\nopen({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{c.pid}}')\n"
            "b = bytearray(1700 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096]); time.sleep(120)")
    r = run(request_, tmp_path, code, coordinator=c, timeout_s=60)
    f = r["failures"][0]
    assert f["code"] == "WORKER_MEMORY_LIMIT" and r["status"] == "FAILED" and r["processing"]["worker"]["tree_peak_rss_mb"] > r["processing"]["worker"]["limit_mb"]
    wp, cp = (int(x) for x in pids.read_text().split())
    alive = lambda p: subprocess.run(["ps", "-o", "stat=", "-p", str(p)], capture_output=True, text=True).stdout.strip() not in ("",) and "Z" not in subprocess.run(["ps", "-o", "stat=", "-p", str(p)], capture_output=True, text=True).stdout   # noqa: E731
    t = time.time()
    while (alive(wp) or alive(cp)) and time.time() - t < 10:
        time.sleep(0.1)
    assert not alive(wp) and not alive(cp) and c.state()["reservations"] == []


def test_reservations_are_released_after_timeout_crash_cancel_staging_failure_and_bad_output(request_, tmp_path):
    import threading
    c = coord(tmp_path)
    assert run(request_, tmp_path, "import time; time.sleep(60)", coordinator=c, timeout_s=1.0)["failures"][0]["code"] == "WORKER_TIMEOUT" and c.state()["reservations"] == []
    assert run(request_, tmp_path, "import os; os._exit(5)", coordinator=c)["failures"][0]["code"] == "WORKER_CRASHED" and c.state()["reservations"] == []
    assert run(request_, tmp_path, "pass", coordinator=c, max_job_bytes=3)["failures"][0]["code"] == "JOB_TOO_LARGE" and c.state()["reservations"] == []         # staging failed after admission
    assert run(request_, tmp_path, writes({"schema": "wrong"}), coordinator=c)["failures"][0]["code"] == "WORKER_OUTPUT_INVALID" and c.state()["reservations"] == []
    cancel = threading.Event(); out = {}
    th = threading.Thread(target=lambda: out.update(r=RR.run_request(request_, RR.ReconstructionLimits(poll_s=0.05, kill_grace_s=1.0), worker_cmd=FAKE_PY("import time; time.sleep(60)"),
                                                                       workspace_root=tmp_path, cancel=cancel, coordinator=c)))
    th.start(); t = time.time()
    while not c.state()["reservations"] and time.time() - t < 10:
        time.sleep(0.02)
    assert len(c.state()["reservations"]) == 1; cancel.set(); th.join(30)
    assert out["r"]["failures"][0]["code"] == "WORKER_CANCELLED" and c.state()["reservations"] == []


def test_an_exception_inside_the_supervisor_releases_the_reservation(request_, tmp_path, monkeypatch):
    c = coord(tmp_path)
    monkeypatch.setattr(RR.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    r = RR.run_request(request_, RR.ReconstructionLimits(), workspace_root=tmp_path, coordinator=c)
    assert r["status"] == "FAILED" and r["failures"][0]["code"] == "INTERNAL_ERROR" and c.state()["reservations"] == []


def test_a_broken_coordinator_fails_closed(request_, tmp_path):
    class Broken:
        def reserve(self, *a, **k): raise OSError("gone")
    r = run(request_, tmp_path, "pass", coordinator=Broken())
    assert r["failures"][0]["code"] == "INSUFFICIENT_MEMORY" and r["status"] == "FAILED"


def test_the_worker_peak_found_after_the_run_voids_the_result(request_, tmp_path):
    c = coord(tmp_path, total=3000.0)                                                   # grant about 2500
    r = run(request_, tmp_path, writes(dict(WORKER_OK, peak_rss_mb=2900.0)), coordinator=c)
    assert r["status"] == "FAILED" and r["width_mm"] is None and r["failures"][0]["code"] == "WORKER_MEMORY_LIMIT" and "after the run" in r["failures"][0]["message"]
    ok = run(request_, tmp_path, writes(dict(WORKER_OK, peak_rss_mb=900.0)), coordinator=c)
    assert ok["status"] == "SUCCESS" and ok["processing"]["worker"]["worker_self_peak_rss_mb"] == 900.0


def test_container_memory_pressure_stops_the_worker_when_the_cgroup_is_readable(request_, tmp_path, monkeypatch):
    c = coord(tmp_path, total=9000.0, container=dict(limit_mb=4000.0, reliable=True, source="cgroup_v2", reason="test"))
    monkeypatch.setattr(RR, "read_cgroup", lambda *a, **k: dict(anon_mb=3700.0, shmem_mb=0.0, memory_current_mb=3800.0, file_mb=100.0))
    r = run(request_, tmp_path, "import time; time.sleep(60)", coordinator=c, timeout_s=60)
    f = r["failures"][0]
    assert f["code"] == "WORKER_MEMORY_LIMIT" and "container memory" in f["message"] and r["processing"]["worker"]["container_guard"]["peak_used_mb"] == 3700.0 and c.state()["reservations"] == []
    monkeypatch.setattr(RR, "read_cgroup", lambda *a, **k: dict(anon_mb=500.0))
    ok = run(request_, tmp_path, writes(WORKER_OK), coordinator=c); assert ok["status"] == "SUCCESS"                   # below 90%: no interference with a legitimate run


def test_polling_gets_faster_close_to_the_ceiling(request_, tmp_path):
    c = coord(tmp_path, total=9000.0)
    body = "import time\nb = bytearray(700 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096])\ntime.sleep(2.0)"
    slow = run(request_, tmp_path, body, coordinator=c, max_tree_rss_mb=1000.0, poll_s=0.5, fast_poll_s=0.5)["processing"]["worker"]["rss_checks"]
    fast = run(request_, tmp_path, body, coordinator=c, max_tree_rss_mb=1000.0, poll_s=0.5, fast_poll_s=0.02)["processing"]["worker"]["rss_checks"]       # 700 MB > 60% of 1000 MB
    assert fast > 3 * slow and fast >= 20


def test_the_worker_watchdog_stops_a_process_above_its_ceiling_even_when_the_parent_polls_slowly(request_, tmp_path):
    """The REAL worker: the parent polls every 30 s (so it cannot be the one that acts); the worker's own peak-RSS watchdog stops it at the granted 20 MB."""
    c = coord(tmp_path)
    r = RR.run_request(request_, RR.ReconstructionLimits(max_tree_rss_mb=20.0, min_tree_rss_mb=20.0, poll_s=30.0, fast_poll_s=30.0, timeout_s=120, kill_grace_s=1.0), workspace_root=tmp_path, coordinator=c)
    f = r["failures"][0]
    assert f["code"] == "WORKER_MEMORY_LIMIT" and "watchdog" in f["message"] and r["status"] == "FAILED" and r["width_mm"] is None and r["processing"]["worker"]["exit_code"] == RR.WATCHDOG_EXIT_CODE
    assert r["processing"]["worker"]["rss_checks"] == 0 and c.state()["reservations"] == []


def test_the_watchdog_does_not_interfere_with_a_legitimate_worker(tmp_path):
    import reconstruction_worker as RW
    job = tmp_path / "job"; job.mkdir()
    (job / "request.json").write_text(json.dumps(dict(calibration=call_params(SR.make_context()), frames=[], line_sets=[], limits=dict(max_rss_mb=4096.0, watchdog_s=0.01, exit_code=86))))
    p = subprocess.run([sys.executable, "-I", str(HERE / "reconstruction_worker.py"), str(job)], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and (job / "result.json").is_file()                        # no frames: the worker reports its own typed outcome, and the watchdog never fired
    assert RW._start_watchdog({}) is None and RW._start_watchdog({"max_rss_mb": "x"}) is None


# ================================================================ upload retention during a job
class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def make_store(tmp_path, clock, **kw):
    lim = dict(max_file_bytes=64 * MiB, complete_retention_s=100.0, job_max_s=10_000.0, lease_heartbeat_timeout_s=30.0, lease_max_s=600.0, idle_timeout_s=50.0)
    lim.update(kw)
    return ChunkStore(Limits(**lim), root=tmp_path / "room", clock=clock, slots=RV.ROOM_UPLOAD_SLOTS, root_prefix="f8_test_room_")


def upload(store, owner, slot, payload=None):
    data = payload or (slot + owner).encode() * 1000; sha = hashlib.sha256(data).hexdigest()
    uid = store.begin(owner, "clip.mov", len(data), 64 * 1024, slot + sha[:6], slot=slot)["upload_id"]
    for i in range(0, len(data), 64 * 1024):
        c = data[i:i + 64 * 1024]; store.put_chunk(owner, uid, i // (64 * 1024), c, hashlib.sha256(c).hexdigest())
    store.finish(owner, uid, sha); return uid, sha


def upload_all(store, owner, **kw):
    return {s: upload(store, owner, s, **kw) for s in RV.ROOM_UPLOAD_SLOTS}


def expected(ups): return {s: (u, h) for s, (u, h) in ups.items()}


def sources(store, owner, ups):
    return {s: RF.SourceVideo(s, str(store.path_of(owner, u)), u, h, 10, 100, 30.0, (2160, 3840)) for s, (u, h) in ups.items()}


@pytest.fixture
def clk(): return Clock()


@pytest.fixture
def rs(tmp_path, clk):
    s = make_store(tmp_path, clk); yield s; s.close()


def test_a_job_lease_keeps_uploads_past_the_retention_window_and_release_restores_normal_expiry(rs, clk):
    A_ = "owner-A" + "a" * 20; ups = upload_all(rs, A_); lease = rs.acquire_job_lease(A_, expected(ups), 500.0)
    for _ in range(20):                                                                   # 400 s: far past complete_retention_s (100) and the 30 s heartbeat timeout, renewed every 20 s
        clk.t += 20; rs.heartbeat(A_, lease["lease_id"]); assert rs.sweep() == []
    assert all(rs.path_of(A_, u).is_file() for u, _ in ups.values())                      # still there: the active job kept them
    rs.release_job_lease(A_, lease["lease_id"])
    assert rs.sweep() == [] and all(rs.path_of(A_, u).is_file() for u, _ in ups.values())  # retention restarts from the release: not deleted at once ...
    clk.t += 101
    assert len(rs.sweep()) == 4 and rs.job_status(A_)["slots"] == {}                      # ... but not kept forever either: abandoned uploads are still cleaned up


def test_without_a_lease_the_same_uploads_do_expire(rs, clk):
    A_ = "owner-A" + "a" * 20; upload_all(rs, A_); clk.t += 150
    assert len(rs.sweep()) == 4


def test_leased_uploads_cannot_be_deleted_or_replaced_meanwhile(rs, clk):
    A_ = "owner-A" + "a" * 20; ups = upload_all(rs, A_); lease = rs.acquire_job_lease(A_, expected(ups), 500.0)
    with pytest.raises(UploadError) as e:
        rs.cancel(A_, ups["p1_left"][0])
    assert e.value.code == "in_use"
    with pytest.raises(UploadError) as e:
        rs.begin(A_, "other.mov", 1000, 64 * 1024, "zz", slot="p1_left")
    assert e.value.code == "in_use" and rs.release_owner("someone-else") == 0
    rs.release_job_lease(A_, lease["lease_id"]); rs.cancel(A_, ups["p1_left"][0])         # after release the visitor may delete again


def test_expired_missing_and_replaced_uploads_are_never_protected(rs, clk, tmp_path):
    A_ = "owner-A" + "a" * 20; ups = upload_all(rs, A_)
    clk.t += 150; rs.sweep()                                                              # expired and deleted
    with pytest.raises(UploadError) as e:
        rs.acquire_job_lease(A_, expected(ups), 500.0)
    assert e.value.code == "not_found"
    ret = RR.StoreRetention(rs, A_); f = ret.acquire({s: None for s in RV.ROOM_UPLOAD_SLOTS}, 500.0)
    assert f["code"] == "UPLOAD_EXPIRED" and f["failure_class"] == "visitor_input" and f["visitor_message"] == VM.failure_text("UPLOAD_EXPIRED")
    ups2 = upload_all(rs, A_)                                                             # re-uploaded: new ids; the OLD selection must not become valid again
    old = {s: RF.SourceVideo(s, "x", ups[s][0], ups[s][1], 10, 100, 30.0, (2160, 3840)) for s in ups}
    f = RR.StoreRetention(rs, A_).acquire(old, 500.0)
    assert f["code"] == "VIDEO_REPLACED" and rs.job_status(A_)["slots"]["p1_left"]["leased"] is False
    same_id_other_content = {s: RF.SourceVideo(s, "x", ups2[s][0], "0" * 64, 10, 100, 30.0, (2160, 3840)) for s in ups2}      # same upload id, different recorded fingerprint
    assert RR.StoreRetention(rs, A_).acquire(same_id_other_content, 500.0)["code"] == "VIDEO_REPLACED"
    ok = RR.StoreRetention(rs, A_); assert ok.acquire(sources(rs, A_, ups2), 500.0) is None; ok.release()


def test_a_second_job_on_the_same_uploads_is_refused_and_ownership_is_enforced(rs, clk):
    A_, B_ = "owner-A" + "a" * 20, "owner-B" + "b" * 20; ups = upload_all(rs, A_); lease = rs.acquire_job_lease(A_, expected(ups), 500.0)
    with pytest.raises(UploadError) as e:
        rs.acquire_job_lease(A_, expected(ups), 500.0)
    assert e.value.code == "job_running"
    with pytest.raises(UploadError) as e:
        rs.acquire_job_lease(B_, expected(ups), 500.0)                                    # another visitor cannot protect (or reach) A's uploads
    assert e.value.code == "not_found"
    for fn in (lambda: rs.heartbeat(B_, lease["lease_id"]), lambda: rs.release_job_lease(B_, lease["lease_id"])):
        with pytest.raises(UploadError) as e:
            fn()
        assert e.value.code == "not_found"
    assert RR.StoreRetention(rs, B_).acquire(sources(rs, A_, ups), 500.0)["code"] in ("UPLOAD_EXPIRED",)


def test_concurrent_visitors_are_isolated(rs, clk):
    A_, B_ = "owner-A" + "a" * 20, "owner-B" + "b" * 20
    ua = upload_all(rs, A_); ub = upload_all(rs, B_)
    la = rs.acquire_job_lease(A_, expected(ua), 500.0)
    clk.t += 120
    for _ in range(3):
        clk.t += 1; rs.heartbeat(A_, la["lease_id"])
    gone = rs.sweep()
    assert set(gone) == {u for u, _ in ub.values()} and all(rs.path_of(A_, u).is_file() for u, _ in ua.values())      # A's lease protects only A; B's uploads expire as usual
    with pytest.raises(UploadError):
        rs.acquire_job_lease(B_, expected(ub), 500.0)                                     # B's expired uploads cannot be leased
    assert rs.job_status(B_)["slots"] == {} and set(rs.job_status(A_)["slots"]) == set(RV.ROOM_UPLOAD_SLOTS)


def test_a_crashed_job_cannot_protect_uploads_forever(rs, clk):
    A_ = "owner-A" + "a" * 20; ups = upload_all(rs, A_); lease = rs.acquire_job_lease(A_, expected(ups), 500.0)
    clk.t += 31                                                                           # no heartbeat for longer than lease_heartbeat_timeout_s: the supervisor died
    assert rs.sweep() == [] and rs._leases[lease["lease_id"]].state == "expired"          # protection ended; the uploads were NOT deleted by the lease
    assert all(u.leased_by is None for u in rs._uploads.values()) and rs.job_status(A_)["lease"] is None
    with pytest.raises(UploadError) as e:
        rs.heartbeat(A_, lease["lease_id"])
    assert e.value.code == "lease_expired"
    clk.t += 101
    assert len(rs.sweep()) == 4                                                           # ordinary retention resumed from the lease end
    # the hard bound: even a supervisor that heartbeats forever is cut off at max_s
    ups = upload_all(rs, A_); lease = rs.acquire_job_lease(A_, expected(ups), 90.0)
    for _ in range(5):
        clk.t += 20
        try:
            rs.heartbeat(A_, lease["lease_id"])
        except UploadError:
            break
    rs.sweep(); assert rs._leases[lease["lease_id"]].state == "expired" and all(u.leased_by is None for u in rs._uploads.values())


def test_locking_the_session_ends_the_lease_and_tells_the_job_to_stop(rs, clk):
    A_ = "owner-A" + "a" * 20; ups = upload_all(rs, A_); ret = RR.StoreRetention(rs, A_)
    assert ret.acquire(sources(rs, A_, ups), 500.0) is None and ret.heartbeat() is False
    rs.release_owner(A_)
    assert rs.job_status(A_)["slots"] == {} and ret.heartbeat() is True                   # the session lock deleted the uploads; the supervisor learns it must stop
    ret.release(); assert ret.lease_id is None


def test_job_leases_do_not_disturb_the_calibration_lease_view(rs, clk):
    A_ = "owner-A" + "a" * 20; ups = upload_all(rs, A_); rs.acquire_job_lease(A_, expected(ups), 500.0)
    assert rs.job_status(A_)["lease"] is None and all(v["leased"] for v in rs.job_status(A_)["slots"].values())


def test_retention_is_released_when_the_reconstruction_ends_early_or_late(rs, clk, request_, tmp_path):
    A_ = "owner-A" + "a" * 20; ups = upload_all(rs, A_); ret = RR.StoreRetention(rs, A_)
    r = RR.run_visitor_reconstruction(None, A_, None, None, sources(rs, A_, ups), RR.ReconstructionLimits(), retention=ret, coordinator=coord(tmp_path))     # prepare cannot run: still released
    assert r["status"] == "FAILED" and ret.lease_id is None and all(u.leased_by is None for u in rs._uploads.values()) and set(rs.job_status(A_)["slots"]) == set(RV.ROOM_UPLOAD_SLOTS)
    ret2 = RR.StoreRetention(rs, A_); c = coord(tmp_path, name="rb5")
    r = RR.run_request(request_, RR.ReconstructionLimits(poll_s=0.05, heartbeat_s=0.05, kill_grace_s=1.0), worker_cmd=FAKE_PY("import time; time.sleep(60)"), workspace_root=tmp_path, coordinator=c, retention=ret2)
    assert r["failures"][0]["code"] == "WORKER_CANCELLED" and "lease" in r["failures"][0]["message"] and c.state()["reservations"] == []        # an un-acquired / lost lease stops the job
    f = RR.run_visitor_reconstruction(None, A_, None, None, {s: None for s in RV.ROOM_UPLOAD_SLOTS}, RR.ReconstructionLimits(), retention=RR.StoreRetention(rs, A_))
    assert f["failures"][0]["code"] == "UPLOAD_EXPIRED" and f["status"] == "FAILED" and f["width_mm"] is None


def test_heartbeats_renew_the_lease_while_the_worker_runs(rs, clk, request_, tmp_path):
    A_ = "owner-A" + "a" * 20; ups = upload_all(rs, A_); ret = RR.StoreRetention(rs, A_); assert ret.acquire(sources(rs, A_, ups), 500.0) is None
    beats = []
    ret.heartbeat = lambda: (beats.append(1), False)[1]
    r = RR.run_request(request_, RR.ReconstructionLimits(poll_s=0.05, heartbeat_s=0.1, kill_grace_s=1.0), worker_cmd=FAKE_PY("import time; time.sleep(1.0)"), workspace_root=tmp_path,
                       coordinator=coord(tmp_path), retention=ret)
    assert len(beats) >= 4 and r["failures"][0]["code"] == "WORKER_CRASHED"
    ret.release()


# ================================================================ resolution policy and the closed gate
def test_resolution_policy_reports_validated_and_unvalidated_sizes_separately_and_nothing_is_resized():
    p = RR.resolution_policy()
    assert p["initial_deployment_target_wh"] == [2160, 3840] and p["target_is_a_guarantee"] is False and p["validated_on_real_recordings"] == [[2160, 3840]]
    assert p["exercised_with_synthetic_renders_only"] == [[1080, 1920]] and "unvalidated" in p["other_sizes"] and "never a rejection" in p["other_sizes"]
    assert p["resized_cropped_or_resampled"] is False and p["intrinsics_rescaled"] is False and p["pixel_unit_thresholds_rescaled"] is False
    for f in ("room_reconstruction.py", "reconstruction_worker.py"):
        src = (HERE / f).read_text(); assert "cv2.resize" not in src and "INTER_" not in src and "pyrDown" not in src, f
    assert not hasattr(RR.ReconstructionLimits(), "allowed_sizes") and "VALIDATED_FRAME_SIZES" not in (HERE / "room_reconstruction.py").read_text().replace("RV.VALIDATED_FRAME_SIZES", "")


def test_public_reconstruction_gate_stays_closed_and_ignores_the_environment(monkeypatch):
    monkeypatch.setenv("ROOM_PROCESSING_ENABLED", "1"); monkeypatch.setenv("F8_ROOM_PROCESSING_ENABLED", "true")
    assert FS.ROOM_PROCESSING_ENABLED is False
    src = (HERE / "f8_visitor_service.py").read_text()
    assert re.search(r"^ROOM_PROCESSING_ENABLED = False\b", src, re.M) and "environ" not in "".join(l for l in src.splitlines() if "ROOM_PROCESSING_ENABLED" in l)
    for name in ("resource_budget", "visitor_messages", "room_reconstruction", "StoreRetention"):                       # the service does not use any of the new machinery
        assert name not in src, name


def test_the_new_modules_import_no_machine_learning_and_no_original_data_files():
    for p in (HERE / "visitor_messages.py", HERE.parent / "f8_chunk_poc" / "resource_budget.py"):
        t = p.read_text()
        assert not re.search(r"^\s*(import|from)\s+(torch|tensorflow|sklearn|scipy|onnx|transformers|cv2)\b", t, re.M), p.name
        assert not re.search(r"recipe_a|load_calibration|DEFAULT_CALIBRATION_PATH|ground_truth|outputs/", t, re.I), p.name

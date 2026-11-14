"""F8.5: path-based dual-video calibration runner (worker process, lease, validation, timeout, cancel, memory guard, cleanup).

Real runs use SYNTHETIC checkerboard videos with known ground truth and the UNMODIFIED F7 module; failure paths use fake workers that still take the
same shared file locks as the real worker. No protected or real video is read."""
import hashlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

import filelock
from calibration_runner import DEFAULT_F7_DIR, CalibrationJobRunner, RunnerLimits, validate_videos
from chunk_store import ChunkStore, Limits, MiB, UploadError
from make_synthetic import make_video
from synthetic_stereo import make_stereo_pair

A, B = "owner-A-" + "a" * 16, "owner-B-" + "b" * 16
SETTINGS = dict(cols=8, rows=6, square_mm=20.0)
F7_FILE = DEFAULT_F7_DIR / "f7_calibration.py"
pytestmark = pytest.mark.skipif(not F7_FILE.is_file(), reason="the F7 module is not on this machine (untracked local file)")
HERE = str(Path(filelock.__file__).parent)
PRE = (f"import json,sys,os,time,subprocess; sys.path.insert(0,{HERE!r}); import filelock; spec=json.load(open(sys.argv[1])); "
       "locks=filelock.acquire_shared([spec['left_path'],spec['right_path']])\n")


def fake(body):
    return lambda spec, jd: [sys.executable, "-I", "-c", PRE + body, str(spec)]


OK_RESULT = "json.dump({'ok': True, 'assessment': {'level': 'GOOD'}, 'leak': os.environ.get('F8_PASSCODE', 'none')}, open(os.path.join(spec['out_dir'], 'result.json'), 'w'))"


def sha(b): return hashlib.sha256(b).hexdigest()


def files(store): return sorted(p.name for p in Path(store.root).rglob("*") if p.is_file())


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        return "Z" not in os.popen(f"ps -o stat= -p {pid}").read()                                     # a zombie is not alive
    except OSError:
        return True


@pytest.fixture(scope="module")
def syn(tmp_path_factory): return make_stereo_pair(tmp_path_factory.mktemp("syn"))


@pytest.fixture
def store(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB, lease_heartbeat_timeout_s=60), root=tmp_path / "store"); yield s; s.close()


def put(store, owner, slot, path):
    data = Path(path).read_bytes(); cs = MiB
    uid = store.begin(owner, "clip.mov", len(data), cs, slot + sha(data)[:6], slot=slot)["upload_id"]
    for i in range(0, len(data), cs):
        c = data[i:i + cs]; store.put_chunk(owner, uid, i // cs, c, sha(c))
    store.finish(owner, uid, sha(data)); return uid


def put_pair(store, left, right, owner=A):
    return put(store, owner, "left", left), put(store, owner, "right", right)


def run(store, owner=A, limits=RunnerLimits(allow_full=True, heartbeat_s=1.0), timeout=240, **kw):
    r = CalibrationJobRunner(store, owner, SETTINGS, limits, monitor_interval=0.25, **kw); r.start(); r.join(timeout)
    assert r.done, r.status()
    return r


def wait_for(cond, secs=20):
    t = time.time()
    while time.time() - t < secs:
        if cond(): return True
        time.sleep(0.05)
    return False


def assert_clean(store, owner=A):
    assert files(store) == [] and store.job_status(owner) == dict(slots={}, lease=None)


# ---------------------------------------------------------------- the real thing, on synthetic data with ground truth
@pytest.fixture(scope="module")
def real_run(syn, tmp_path_factory):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB), root=tmp_path_factory.mktemp("rr") / "store")
    put_pair(s, syn["left"], syn["right"])
    r = CalibrationJobRunner(s, A, SETTINGS, RunnerLimits(allow_full=True, heartbeat_s=1.0), monitor_interval=0.25)
    seen = []; r.start()
    while not r.done:
        st = r.status(); seen.append((st["state"], st["stage"], st["fraction"])); time.sleep(0.2)
    r.join(30)
    yield dict(runner=r, store=s, seen=seen, files_after=files(s), job=s.job_status(A))
    s.close()


def test_full_run_calibrates_the_synthetic_rig(real_run, syn):
    res = real_run["runner"].result
    assert res["outcome"] == "completed" and res["ok"] and res["assessment"]["level"] == "GOOD" and res["calibration"]["pairs_used"] >= 8
    K1, K2, T = res["model"]["K1"], res["model"]["K2"], res["model"]["T"]
    assert K1[0][0] == pytest.approx(syn["K"][0, 0], rel=0.01) and K2[1][1] == pytest.approx(syn["K"][1, 1], rel=0.01)
    assert K1[0][2] == pytest.approx(syn["K"][0, 2], abs=3) and K1[1][2] == pytest.approx(syn["K"][1, 2], abs=3)
    assert res["model"]["baseline_mm"] == pytest.approx(syn["baseline_mm"], rel=0.01) and res["model"]["stereo_rms"] < 0.5
    assert res["calibration"]["sync"]["offset_frames"] == syn["expected_offset"] and res["calibration"]["sync"]["confident"]
    assert res["contract"]["baseline_mm"] == pytest.approx(syn["baseline_mm"], rel=0.01) and res["contract"]["image_size_wh"] == list(syn["size"])


def test_result_is_serialisable_bounded_and_leaks_no_paths(real_run):
    res = real_run["runner"].result; text = json.dumps(res)
    assert json.loads(text)["outcome"] == "completed" and len(text) < 8 * MiB
    assert str(real_run["store"].root) not in text and "f8_job_" not in text and "/private/" not in text.replace("/private/tmp", "")
    for key in ("calibration", "model", "assessment", "contract", "resources", "environment", "validation", "stage_timings", "f7", "env", "cleanup", "detections"):
        assert key in res, key
    assert res["detections"]["left"]["detections"] and res["detections"]["left"]["motion"]


def test_left_is_processed_before_right_and_progress_is_ordered(real_run):
    stages = [s["stage"] for s in real_run["runner"].result["resources"]["stages"]]
    assert stages.index("detect_left") < stages.index("detect_right")
    assert "solve" not in stages or stages.index("detect_right") < stages.index("solve")                # the synthetic solve (~0.13 s) can finish between 0.25 s polls; the worker's stage_timings below prove it ran
    t = real_run["runner"].result["stage_timings"]; assert set(t) >= {"detect_left", "detect_right", "solve", "assess"} and t["detect_left"]["frames"] == t["detect_right"]["frames"]
    seen = [x for x in real_run["seen"] if x[0] == "running"]
    order = [x[1] for x in seen]; first = lambda n: order.index(n) if n in order else None
    assert first("detect_left") is not None and (first("detect_right") is None or first("detect_left") < first("detect_right"))
    fr = [x[2] for x in seen if x[2] > 0]; assert fr == sorted(fr)


def test_both_videos_are_deleted_and_the_lease_released_after_success(real_run):
    res = real_run["runner"].result
    assert res["cleanup"] == dict(lease_released=True, videos_deleted=True, job_dir_removed=True)
    assert real_run["files_after"] == [] and real_run["job"] == dict(slots={}, lease=None)
    assert not alive(res["env"]["pid"])                                                                  # the worker is gone


def test_resource_measurements_are_recorded_stage_by_stage(real_run):
    r = real_run["runner"].result["resources"]; names = [s["stage"] for s in r["stages"]]
    assert {"validate", "detect_left", "detect_right"} <= set(names)                                     # "solve" is too short on synthetic data to be sampled reliably
    det = next(s for s in r["stages"] if s["stage"] == "detect_left")
    assert det["wall_s"] > 0 and det["worker_rss_max_mb"] and det["worker_rss_max_mb"] > 30 and det["parent_rss_max_mb"] and det["cpu_s"] and det["cpu_s"] > 0
    assert det["temp_disk_max_mb"] and det["temp_disk_max_mb"] > 5                                       # both ~4.6 MB videos were on disk
    assert r["peak"]["worker_rss_mb"] and r["peak"]["tree_rss_mb"] >= r["peak"]["worker_rss_mb"] and r["total_wall_s"] > 0
    assert set(r["available"]) == {"pss", "cgroup"} and len(json.dumps(r)) < 20000                       # bounded
    env = real_run["runner"].result["environment"]; assert env["cpu_count"] and "cgroup" in env and "temp_dir_mount" in env
    assert real_run["runner"].result["env"]["cv2"] and real_run["runner"].result["env"]["threads"] >= 1


def test_f7_is_used_unmodified_and_its_identity_is_recorded(real_run):
    f7 = real_run["runner"].result["f7"]
    assert f7["f7_calibration"]["sha256"] == hashlib.sha256(F7_FILE.read_bytes()).hexdigest() and f7["stride"] == 5 and f7["max_pairs"] == 30
    for k in ("visitor_calibration", "visitor_video_validation", "user_calibration"): assert len(f7[k]["sha256"]) == 64


def test_runner_results_equal_running_the_f7_functions_directly(real_run, syn):
    """The runner/worker must not change the calibration: same corners, same model, same quality, to numerical precision.

    Tolerance 1e-6 (was 1e-9): OpenCV's multithreaded stereo solver is not bit-reproducible -- repeated calls on IDENTICAL input in ONE process differ by up to ~1e-8
    (measured; with cv2.setNumThreads(1) they are identical). 1e-6 mm/rad is still ~1000x below anything physically meaningful and far tighter than any real change."""
    sys.path.insert(0, str(DEFAULT_F7_DIR))
    import f7_calibration as K
    L, R = K.extract_detections(syn["left"], (8, 6)), K.extract_detections(syn["right"], (8, 6))
    direct = K.calibrate_from_detections(L, R, SETTINGS); a = K.assess(direct)
    res = real_run["runner"].result; m = direct["model"]
    import numpy as np
    for k in ("K1", "K2", "D1", "D2", "R", "T"):
        assert np.allclose(np.asarray(res["model"][k], float), np.asarray(m[k], float), rtol=1e-6, atol=1e-6), k
    assert res["model"]["stereo_rms"] == pytest.approx(m["stereo_rms"], abs=1e-6) and res["model"]["baseline_mm"] == pytest.approx(m["baseline_mm"], abs=1e-6)
    assert res["calibration"]["pair_frames"] == [list(p) for p in direct["pair_frames"]] and res["assessment"]["level"] == a["level"]
    got = {d["frame"]: d["corners"] for d in res["detections"]["left"]["detections"]}
    assert set(got) == {d["frame"] for d in L["detections"]} and all(np.allclose(got[d["frame"]], d["corners"], atol=1e-3) for d in L["detections"])


def test_an_offset_between_the_cameras_is_found(tmp_path, store):
    gt = make_stereo_pair(tmp_path / "off", right_starts_later_by=7)
    put_pair(store, gt["left"], gt["right"]); r = run(store)
    assert r.result["outcome"] == "completed" and r.result["calibration"]["sync"]["offset_frames"] == -7 and r.result["model"]["baseline_mm"] == pytest.approx(100, rel=0.01)
    assert_clean(store)


# ---------------------------------------------------------------- explicit validation (instead of the F7 visitor limits), nothing silently skipped
def no_spawn(spec, jd): raise AssertionError("the worker must not be started for rejected videos")


def test_short_video_is_rejected_with_a_clear_reason_and_cleaned_up(tmp_path, store, syn):
    short = make_video(tmp_path / "short.mp4", seconds=2, w=1280, h=720, fps=30)
    put_pair(store, short, syn["right"]); r = run(store, worker_cmd=no_spawn)
    assert r.result["outcome"] == "rejected" and "left_too_short" in [e["code"] for e in r.result["validation"]["errors"]] and "shorter than" in r.result["error"]
    assert r.result["validation"]["limits_applied"]["min_duration_s"] == 10.0 and r.result["validation"]["limits_applied"]["max_duration_s"] == 900.0
    assert_clean(store)


def test_resolution_and_frame_rate_mismatch_are_rejected(tmp_path, store, syn):
    other = make_video(tmp_path / "small.mp4", seconds=12, w=320, h=180, fps=25)
    put_pair(store, syn["left"], other); r = run(store, worker_cmd=no_spawn)
    codes = [e["code"] for e in r.result["validation"]["errors"]]
    assert r.result["outcome"] == "rejected" and "size_mismatch" in codes and "fps_mismatch" in codes
    assert_clean(store)


def test_too_long_and_too_many_frames_use_the_configured_limits(store, syn):
    put_pair(store, syn["left"], syn["right"])
    r = run(store, limits=RunnerLimits(allow_full=True, max_duration_s=5, max_frames=100), worker_cmd=no_spawn)
    codes = [e["code"] for e in r.result["validation"]["errors"]]
    assert {"left_too_long", "right_too_long", "left_too_many_frames", "right_too_many_frames"} <= set(codes) and r.result["outcome"] == "rejected"
    assert_clean(store)


def test_non_video_files_are_rejected_by_the_sandboxed_inspection(tmp_path, store):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"; a.write_bytes(os.urandom(5000)); b.write_bytes(os.urandom(5001))
    put_pair(store, a, b); r = run(store, worker_cmd=no_spawn)
    assert {"left_unsupported_signature", "right_unsupported_signature"} <= {e["code"] for e in r.result["validation"]["errors"]}
    assert_clean(store)


def test_validate_videos_reports_every_check(syn):
    rep = validate_videos(syn["left"], syn["right"], RunnerLimits(allow_full=True))
    assert rep["ok"] and rep["errors"] == [] and set(rep["videos"]) == {"left", "right"} and rep["videos"]["left"]["width"] == 1280 and rep["videos"]["left"]["duration_s"] > 20
    assert rep["limits_applied"]["video_safety"]["max_pixels"] == 40_000_000


def test_missing_second_video_is_refused_and_the_first_is_kept(store, syn):
    put(store, A, "left", syn["left"]); r = run(store, worker_cmd=no_spawn)
    assert r.result["outcome"] == "rejected" and "RIGHT" in r.result["error"] and store.job_status(A)["slots"]["left"]["state"] == "complete"


def test_another_session_cannot_run_this_sessions_videos(store, syn):
    put_pair(store, syn["left"], syn["right"], A); r = run(store, owner=B, worker_cmd=no_spawn)
    assert r.result["outcome"] == "rejected" and set(store.job_status(A)["slots"]) == {"left", "right"} and files(store) != []


def test_f7_module_not_found_is_reported_not_silenced(tmp_path, store, syn):
    put_pair(store, syn["left"], syn["right"]); r = run(store, f7_dir=tmp_path, worker_cmd=no_spawn)
    assert r.result["outcome"] == "rejected" and "F7 calibration module was not found" in r.result["error"]
    assert_clean(store)


def test_only_one_calibration_at_a_time(store, syn):
    put_pair(store, syn["left"], syn["right"], A); put_pair(store, syn["left"], syn["right"], B)
    r1 = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(allow_full=True, heartbeat_s=0.5), worker_cmd=fake("time.sleep(4); " + OK_RESULT), monitor_interval=0.25); r1.start()
    assert wait_for(lambda: r1.status()["state"] == "running")
    r2 = CalibrationJobRunner(store, B, SETTINGS, RunnerLimits(allow_full=True), worker_cmd=no_spawn); r2.start(); r2.join(20)
    assert r2.result["outcome"] == "rejected" and "Another calibration" in r2.result["error"]
    r1.join(30); assert r1.result["outcome"] == "completed"


# ---------------------------------------------------------------- timeout, termination, cancellation: files are never deleted under a live worker
def test_timeout_stops_the_worker_and_its_children_then_cleans_up(tmp_path, store, syn):
    put_pair(store, syn["left"], syn["right"]); pidfile = tmp_path / "pids.txt"
    body = (f"c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); open({str(pidfile)!r}, 'w').write(f'{{os.getpid()}} {{c.pid}}'); time.sleep(120)")
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(allow_full=True, timeout_s=2.5, heartbeat_s=0.5, kill_grace_s=1.0), worker_cmd=fake(body), monitor_interval=0.25); r.start()
    assert wait_for(lambda: pidfile.exists() and r.status()["state"] == "running")
    with pytest.raises(UploadError) as e: store.release_lease(A, r._lease, "x")                          # a live worker holds the files: release is refused ...
    assert e.value.code == "in_use" and len(files(store)) == 2                                           # ... and nothing was deleted
    r.join(30)
    assert r.result["outcome"] == "timeout" and "2.5" in r.result["error"] and r.result["cleanup"]["videos_deleted"]
    for pid in map(int, pidfile.read_text().split()): assert wait_for(lambda p=pid: not alive(p), 10), pid
    assert_clean(store)


def test_cancel_terminates_the_worker_and_cleans_up(tmp_path, store, syn):
    put_pair(store, syn["left"], syn["right"]); pidfile = tmp_path / "pid.txt"
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(allow_full=True, heartbeat_s=0.5, kill_grace_s=1.0), monitor_interval=0.25,
                             worker_cmd=fake(f"open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(120)")); r.start()
    assert wait_for(lambda: pidfile.exists() and r.status()["state"] == "running")
    t0 = time.time(); r.cancel(); r.join(30)
    assert r.result["outcome"] == "cancelled" and time.time() - t0 < 10 and not alive(int(pidfile.read_text()))
    assert_clean(store)


def test_cancel_during_a_real_calibration(store, syn):
    put_pair(store, syn["left"], syn["right"])
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(allow_full=True, heartbeat_s=0.5, kill_grace_s=2.0), monitor_interval=0.25); r.start()
    assert wait_for(lambda: r.status()["stage"].startswith("detect"), 60)
    r.cancel(); r.join(60)
    assert r.result["outcome"] == "cancelled" and r._proc.poll() is not None
    assert_clean(store)


@pytest.mark.parametrize("body,expect", [("sys.exit(3)", "exit code 3"), ("os.kill(os.getpid(), signal.SIGKILL)", "stopped unexpectedly"),
                                         ("open(os.path.join(spec['out_dir'], 'result.json'), 'w').write('not json at all')", "stopped unexpectedly"),
                                         ("raise RuntimeError('boom')", "stopped unexpectedly")])
def test_worker_crash_or_garbage_result_is_reported_and_cleaned(store, syn, body, expect):
    put_pair(store, syn["left"], syn["right"])
    r = run(store, limits=RunnerLimits(allow_full=True, heartbeat_s=0.5), worker_cmd=fake("import signal\n" + body), timeout=60)
    assert r.result["outcome"] == "worker_crashed" and expect in r.result["error"] and r.result["cleanup"]["videos_deleted"]
    assert_clean(store)


def test_oversized_result_file_is_ignored(store, syn):
    put_pair(store, syn["left"], syn["right"])
    r = run(store, limits=RunnerLimits(allow_full=True, heartbeat_s=0.5, result_max_bytes=200), worker_cmd=fake("json.dump({'ok': True, 'pad': 'x' * 5000}, open(os.path.join(spec['out_dir'], 'result.json'), 'w'))"))
    assert r.result["outcome"] == "worker_crashed"; assert_clean(store)


def test_worker_reported_failure_is_failed_not_completed(store, syn):
    put_pair(store, syn["left"], syn["right"])
    r = run(store, worker_cmd=fake("json.dump({'ok': False, 'error': 'Only 2 usable board poses were visible'}, open(os.path.join(spec['out_dir'], 'result.json'), 'w'))"))
    assert r.result["outcome"] == "failed" and "usable board poses" in r.result["error"]; assert_clean(store)


def test_a_failed_quality_level_is_not_completed(store, syn):
    put_pair(store, syn["left"], syn["right"])
    r = run(store, worker_cmd=fake("json.dump({'ok': True, 'assessment': {'level': 'FAILED'}}, open(os.path.join(spec['out_dir'], 'result.json'), 'w'))"))
    assert r.result["outcome"] == "failed"; assert_clean(store)


def test_memory_guard_kills_a_runaway_worker(store, syn):
    put_pair(store, syn["left"], syn["right"])
    body = "b = bytearray(450 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096]); time.sleep(120)"
    t0 = time.time(); r = run(store, limits=RunnerLimits(allow_full=True, heartbeat_s=0.5, max_tree_rss_mb=300, kill_grace_s=1.0), worker_cmd=fake(body), timeout=60)
    assert r.result["outcome"] == "memory_limit" and "memory guard" in r.result["error"] and time.time() - t0 < 40
    assert r.result["resources"]["peak"]["worker_rss_mb"] > 250 and r._proc.poll() is not None
    assert_clean(store)


# ---------------------------------------------------------------- lease heartbeat and expiry while a job runs
def test_heartbeats_keep_a_short_lease_alive_for_the_whole_job(tmp_path, syn):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB, lease_heartbeat_timeout_s=2.0), root=tmp_path / "s"); stop = threading.Event()
    sweeper = threading.Thread(target=lambda: [s.sweep() or time.sleep(0.2) for _ in iter(stop.is_set, True)], daemon=True); sweeper.start()
    try:
        put_pair(s, syn["left"], syn["right"])
        r = CalibrationJobRunner(s, A, SETTINGS, RunnerLimits(allow_full=True, heartbeat_s=0.4), worker_cmd=fake("time.sleep(5.5); " + OK_RESULT), monitor_interval=0.25); r.start(); r.join(60)
        assert r.result["outcome"] == "completed"                                                         # 5.5 s of work under a 2 s lease timeout: only the heartbeat kept it
    finally:
        stop.set(); s.close()


def test_a_silent_supervisor_loses_the_lease_and_the_worker_is_stopped(tmp_path, syn):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB, lease_heartbeat_timeout_s=1.5), root=tmp_path / "s"); stop = threading.Event()
    sweeper = threading.Thread(target=lambda: [s.sweep() or time.sleep(0.2) for _ in iter(stop.is_set, True)], daemon=True); sweeper.start()
    try:
        put_pair(s, syn["left"], syn["right"]); pidfile = tmp_path / "pid.txt"
        r = CalibrationJobRunner(s, A, SETTINGS, RunnerLimits(allow_full=True, heartbeat_s=3600, kill_grace_s=1.0), monitor_interval=0.25,
                                 worker_cmd=fake(f"open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(120)")); r.start(); r.join(60)
        assert r.result["outcome"] in ("lease_expired", "worker_crashed") and not alive(int(pidfile.read_text()))
        assert wait_for(lambda: files(s) == [], 10)
    finally:
        stop.set(); s.close()


def test_secrets_never_reach_the_worker_or_the_result(store, syn, monkeypatch):
    monkeypatch.setenv("F8_PASSCODE", "super-secret-passcode-xyz")
    put_pair(store, syn["left"], syn["right"]); r = run(store, worker_cmd=fake(OK_RESULT))
    assert r.result["leak"] == "none" and "super-secret" not in json.dumps(r.result)


def test_a_runner_can_only_be_started_once(store, syn):
    put_pair(store, syn["left"], syn["right"])
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(allow_full=True, heartbeat_s=0.5), worker_cmd=fake(OK_RESULT)); r.start(); r.join(30)
    from calibration_runner import RunnerError
    with pytest.raises(RunnerError): r.start()

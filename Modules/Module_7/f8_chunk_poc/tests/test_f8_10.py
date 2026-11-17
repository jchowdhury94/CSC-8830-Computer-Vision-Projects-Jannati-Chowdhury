"""F8.10: result persistence, video retention and retry, 5400 s timeout / lease, memory-guard input handling, recovery from another session.

SYNTHETIC data only (no protected or real video). Failure paths use fake workers that take the same shared file locks as the real worker; one real
end-to-end run (synthetic stereo pair, unmodified F7 module) proves the saved document carries the real numbers."""
import json
import os
import stat
import threading
import time
from pathlib import Path

import pytest

import result_store as RS
from calibration_runner import (DEFAULT_F7_DIR, CalibrationJobRunner, RunnerLimits, active_runner, cancel_by_code, guard_memory, lease_margin_s)
from chunk_store import ChunkStore, Limits, MiB, UploadError, env_limits
from result_store import ResultError, ResultStore, build_result_document, new_code, valid_code, write_atomic
from synthetic_stereo import make_stereo_pair
from test_runner import A, B, SETTINGS, alive, assert_retained, fake, files, put_pair, wait_for

F7_FILE = DEFAULT_F7_DIR / "f7_calibration.py"
needs_f7 = pytest.mark.skipif(not F7_FILE.is_file(), reason="the F7 module is not on this machine (untracked local file)")

FULL = ("json.dump({'ok': True, 'assessment': {'level': 'GOOD', 'headline': 'Calibration looks good', 'problems': [], 'concerns': [], "
        "'metrics': {'rms_stereo_px': 0.79, 'measured_baseline_mm': 226.0}}, 'model': {'K1': [[3000,0,1000],[0,3000,1900],[0,0,1]], 'D1': [[0.1,0,0,0,0]], "
        "'K2': [[2900,0,1000],[0,2900,1900],[0,0,1]], 'D2': [[0.2,0,0,0,0]], 'R': [[1,0,0],[0,1,0],[0,0,1]], 'T': [[-227.1],[0],[0]], 'baseline_mm': 227.13, "
        "'stereo_rms': 0.79, 'rms_left': 0.47, 'rms_right': 0.48, 'image_size_left': [2160, 3840], 'image_size_right': [2160, 3840]}, "
        "'calibration': {'pairs_used': 15, 'pairs_discarded': 0, 'pair_frames': [[250, 330], [1480, 1560]], 'sync': {'offset_frames': 80, 'confident': True}, "
        "'pairing': {'n_selected': 15}, 'fps': [30.0, 30.0], 'detections': {'left': 845, 'right': 876}}, "
        "'detections': {'left': {'detections': [{'frame': 1, 'corners': [[1.0, 2.0]]}]}}, 'peak_rss_mb': 700.0, 'stage_timings': {'detect_left': {'wall_s': 1.0}}}, "
        "open(os.path.join(spec['out_dir'], 'result.json'), 'w'))")
CRASH = "sys.exit(3)"
SLEEP = "time.sleep(120)"


@pytest.fixture
def results(tmp_path):
    return ResultStore(tmp_path / "res")


@pytest.fixture
def store(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB, lease_heartbeat_timeout_s=60), root=tmp_path / "store"); yield s; s.close()


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return make_stereo_pair(tmp_path_factory.mktemp("syn810"))


LIM = RunnerLimits(allow_full=True, heartbeat_s=0.5, kill_grace_s=1.0)


def runner(store, results, body, owner=A, limits=LIM, **kw):
    return CalibrationJobRunner(store, owner, SETTINGS, limits, worker_cmd=fake(body), monitor_interval=0.25, results=results, **kw)


def run(store, results, body, owner=A, limits=LIM, timeout=60, **kw):
    r = runner(store, results, body, owner, limits, **kw); r.start(); r.join(timeout)
    assert r.done, r.status()
    return r


def fake_cgroup(root, limit_mb, anon_mb, file_mb=0.0, shmem_mb=0.0, current_mb=None):
    root.mkdir(parents=True, exist_ok=True)
    cur = current_mb if current_mb is not None else anon_mb + file_mb + shmem_mb
    (root / "memory.current").write_text(str(int(cur * MiB))); (root / "memory.peak").write_text(str(int(cur * MiB)))
    (root / "memory.max").write_text(str(int(limit_mb * MiB)) if limit_mb is not None else "max")
    (root / "memory.stat").write_text(f"anon {int(anon_mb * MiB)}\nfile {int((file_mb + shmem_mb) * MiB)}\nshmem {int(shmem_mb * MiB)}\n")
    return str(root)


# ================================================================== 1. result document and atomic JSON writing
def test_codes_are_long_random_and_only_a_hash_names_the_folder(results):
    c1, c2 = new_code(), new_code()
    assert c1 != c2 and valid_code(c1) and len(c1) >= 20 and not valid_code("short") and not valid_code("../" * 8) and not valid_code(None) and not valid_code("a" * 70)
    results.save_result(c1, dict(x=1))
    assert [p.name for p in results.root.iterdir()] and c1 not in " ".join(str(p) for p in results.root.rglob("*"))
    with pytest.raises(ResultError):
        results.save_result("../../etc/passwd", {})
    assert results.result_bytes("nonexistent-code-0123456789") is None and results.load_result("nonexistent-code-0123456789") is None


def test_roundtrip_nan_sanitised_private_permissions(results):
    code = new_code(); info = results.save_result(code, dict(a=float("nan"), b=float("inf"), c=[1, 2.5], d=dict(e="x")))
    got = results.load_result(code)
    assert got == dict(a=None, b=None, c=[1, 2.5], d=dict(e="x")) and info["bytes"] == len(results.result_bytes(code))
    f = next(results.root.rglob("result.json"))
    assert stat.S_IMODE(f.stat().st_mode) == 0o600 and stat.S_IMODE(f.parent.stat().st_mode) == 0o700 and stat.S_IMODE(results.root.stat().st_mode) == 0o700


def test_write_is_atomic_a_failed_replace_leaves_the_old_result_and_no_temp_file(results, monkeypatch):
    code = new_code(); results.save_result(code, dict(version="old"))
    def boom(*a, **k): raise OSError("disk exploded between write and rename")
    monkeypatch.setattr(RS.os, "replace", boom)
    with pytest.raises(ResultError):
        results.save_result(code, dict(version="new"))
    monkeypatch.undo()
    assert results.load_result(code) == dict(version="old")                                      # the reader never sees a half-written or new-but-partial file
    assert not [p for p in results.root.rglob("*") if p.name.endswith(".tmp")]


def test_write_atomic_never_exposes_a_partial_file(tmp_path):
    target = tmp_path / "r.json"; write_atomic(target, b"A" * 1000); seen = set(); stop = threading.Event()
    def reader():
        while not stop.is_set():
            try:
                seen.add(len(target.read_bytes()))
            except OSError:
                pass
    t = threading.Thread(target=reader); t.start()
    for i in range(200):
        write_atomic(target, bytes([65 + i % 5]) * (1000 if i % 2 else 4_000_000))
    stop.set(); t.join()
    assert seen <= {1000, 4_000_000}, seen                                                      # only complete versions were ever visible


def test_oversize_documents_are_refused_and_nothing_is_written(tmp_path):
    rs = ResultStore(tmp_path / "r", max_bytes=1000); code = new_code()
    with pytest.raises(ResultError):
        rs.save_result(code, dict(pad="x" * 5000))
    assert rs.result_bytes(code) is None


def test_expired_and_excess_results_are_swept(tmp_path):
    now = [1000.0]; rs = ResultStore(tmp_path / "r", ttl_s=100, max_results=2, clock=lambda: now[0]); codes = [new_code() for _ in range(4)]
    for c in codes:
        rs.save_result(c, dict(i=1)); os.utime(next((rs.root).glob("job_*/result.json")).parent, None)
    for c, age in zip(codes, (500, 400, 300, 200)):                                              # make the folders look old, newest last
        for p in rs._dir(c).iterdir(): os.utime(p, (1000 - age, 1000 - age))
    now[0] = 1000 + 0
    n = rs.sweep()                                                                              # all older than ttl=100 -> all go
    assert n == 4 and rs.result_bytes(codes[-1]) is None
    for c in codes: rs.save_result(c, dict(i=1))
    for c, age in zip(codes, (50, 40, 30, 20)):
        for p in rs._dir(c).iterdir(): os.utime(p, (1000 - age, 1000 - age))
    assert rs.sweep() == 2 and rs.result_bytes(codes[0]) is None and rs.result_bytes(codes[3]) is not None      # only the newest two survive; none is older than the ttl


def test_document_has_every_required_section_and_no_paths_or_detections():
    res = dict(ok=True, mode="calibrate", outcome="completed", assessment=dict(level="GOOD", headline="h", problems=[], concerns=[], metrics=dict(rms_stereo_px=0.79)),
               model=dict(K1=[[1]], D1=[[0]], K2=[[2]], D2=[[0]], R=[[1]], T=[[1]], baseline_mm=227.1, stereo_rms=0.79, rms_left=0.4, rms_right=0.5),
               calibration=dict(pairs_used=15, pair_frames=[[1, 2]], sync=dict(offset_frames=80), pairing=dict(n_selected=15)), detections=dict(left=dict(detections=[dict(corners=[[1, 2]])])),
               resources=dict(peak=dict(cgroup_peak_mb=2443.0), cgroup=dict(memory_max_mb=3072.0)), validation=dict(videos=dict(left=dict(width=2160, path="/secret/left.mov"))),
               total_wall_s=12.0, peak_rss_mb=700.0)
    doc = build_result_document(res, code=new_code(), settings=SETTINGS)
    for k in ("quality", "cameras", "extrinsics", "selected_poses", "synchronization", "runtime", "measurements", "videos", "settings", "schema"):
        assert k in doc, k
    assert doc["quality"]["level"] == "GOOD" and doc["quality"]["stereo_rms_px"] == 0.79 and doc["extrinsics"]["baseline_mm"] == 227.1
    assert doc["cameras"]["left"]["K"] == [[1]] and doc["cameras"]["right"]["distortion"] == [[0]] and doc["selected_poses"]["frame_pairs_left_right"] == [[1, 2]]
    assert doc["synchronization"]["sync"]["offset_frames"] == 80 and doc["measurements"]["peak"]["cgroup_peak_mb"] == 2443.0
    text = json.dumps(doc)
    assert "/secret" not in text and "corners" not in text and "detections\"" not in text.replace('"detections_found"', "")


# ================================================================== 2. a completed run saves first, then deletes
def test_result_is_saved_before_the_videos_are_deleted_and_is_downloadable(store, results, syn):
    put_pair(store, syn["left"], syn["right"]); seen = {}
    real_release = store.release_lease
    def spy(owner, lease_id, outcome="done", delete_files=True, retry_keep=False):
        r = active_runner_code[0]
        seen["saved_before_release"] = results.result_bytes(r) is not None and len(files(store)) == 2        # file on disk, videos still there
        seen["delete"] = delete_files
        return real_release(owner, lease_id, outcome, delete_files, retry_keep)
    active_runner_code = [new_code()]; store.release_lease = spy
    r = run(store, results, FULL, job_code=active_runner_code[0])
    assert r.result["outcome"] == "completed" and seen == dict(saved_before_release=True, delete=True)
    assert r.result["cleanup"]["videos_deleted"] and files(store) == [] and r.result["persistence"]["saved"]
    data = results.result_bytes(r.job_code)                                                         # what the download button serves
    doc = json.loads(data)
    assert doc["schema"] == RS.SCHEMA and doc["quality"]["level"] == "GOOD" and doc["quality"]["stereo_rms_px"] == 0.79 and doc["extrinsics"]["baseline_mm"] == 227.13
    assert doc["selected_poses"]["count"] == 15 and doc["synchronization"]["sync"]["offset_frames"] == 80 and doc["runtime"]["supervisor_total_s"] > 0
    assert doc["measurements"]["stages"] and doc["measurements"]["peak"] is not None and "memory_guard" in doc["measurements"] and doc["runtime"]["timeout"]["effective_s"] > 0
    assert doc["cleanup"]["videos_deleted"] is True                                                 # the second atomic write carries the final clean-up facts
    assert str(store.root) not in data.decode() and len(data) < 1 * MiB and "corners" not in data.decode()


@needs_f7
def test_real_synthetic_calibration_is_saved_with_the_real_numbers(store, results, syn):
    put_pair(store, syn["left"], syn["right"])
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(allow_full=True, heartbeat_s=1.0), monitor_interval=0.25, results=results); r.start(); r.join(240)
    assert r.done and r.result["outcome"] == "completed"
    doc = results.load_result(r.job_code)
    m = r.result["model"]
    assert doc["quality"]["level"] == "GOOD" and doc["cameras"]["left"]["K"] == m["K1"] and doc["cameras"]["right"]["distortion"] == m["D2"]
    assert doc["extrinsics"]["R"] == m["R"] and doc["extrinsics"]["T"] == m["T"] and doc["extrinsics"]["baseline_mm"] == m["baseline_mm"] == pytest.approx(syn["baseline_mm"], rel=0.01)
    assert doc["selected_poses"]["count"] >= 8 and doc["selected_poses"]["frame_pairs_left_right"] and doc["synchronization"]["sync"]["offset_frames"] == syn["expected_offset"]
    assert doc["runtime"]["total_wall_s"] > 0 and doc["measurements"]["stages"] and files(store) == []


def test_a_result_that_cannot_be_saved_keeps_the_videos(store, tmp_path, syn):
    class Broken(ResultStore):
        def save_result(self, code, doc): raise ResultError("disk full")
    put_pair(store, syn["left"], syn["right"])
    r = run(store, Broken(tmp_path / "b"), FULL)
    assert r.result["outcome"] == "completed" and r.result["persistence"]["saved"] is False and "disk full" in r.result["persistence"]["error"]
    assert not r.result["cleanup"]["videos_deleted"] and len(files(store)) == 2                    # nothing is deleted unless the result is safe


def test_every_failure_also_saves_a_small_result_with_the_reason(store, results, syn):
    put_pair(store, syn["left"], syn["right"]); r = run(store, results, CRASH)
    doc = results.load_result(r.job_code)
    assert doc["outcome"] == "worker_crashed" and "exit code 3" in doc["error"] and doc["quality"]["level"] is None and doc["cleanup"]["videos_kept_for_next_step"]


# ================================================================== 3. session interruption: the job and its result do not depend on a session
def test_job_status_cancel_and_result_are_reachable_without_the_starting_session(store, results, syn):
    put_pair(store, syn["left"], syn["right"]); code = new_code(); store.set_recovery(A, code)
    r = runner(store, results, "time.sleep(4); " + FULL, job_code=code); r.start()                  # "session 1" starts the job ...
    assert wait_for(lambda: r.status()["state"] == "running")
    del_ref = r; r = None                                                                           # ... and its page object is gone (refresh / disconnect)
    live = active_runner(code)
    assert live is del_ref                                                                          # a NEW session finds the live job by code
    assert wait_for(lambda: (results.read_status(code) or {}).get("state") == "running", 10)
    st = results.read_status(code); assert st["age_s"] < 5 and st["mode"] == "calibrate" and st["timeout_s"] > 0
    del_ref.join(60)
    assert active_runner(code) is None and results.load_result(code)["outcome"] == "completed"       # result recovered by code only
    assert results.read_status(code)["state"] == "done" and results.read_status(code)["result_saved"] is True


def test_another_session_can_cancel_a_running_job_by_code(store, results, syn):
    put_pair(store, syn["left"], syn["right"]); r = runner(store, results, SLEEP); r.start()
    assert wait_for(lambda: r.status()["state"] == "running")
    assert cancel_by_code("not-a-live-code-0123456789") is False and cancel_by_code(r.job_code) is True
    r.join(30); assert r.result["outcome"] == "cancelled"
    assert_retained(store)                                                                          # cancellation keeps the videos for a retry


def test_a_stale_status_is_visible_so_a_restarted_server_is_recognisable(results):
    now = [100.0]; rs = ResultStore(results.root, clock=lambda: now[0]); c = new_code()
    rs.write_status(c, dict(state="running", stage="detect_left")); now[0] = 400.0
    assert rs.read_status(c)["age_s"] == 300.0 and rs.read_status(new_code()) is None               # no result, no status: "expired or server restarted"


# ================================================================== 4. failure and retry with retained videos
def test_timeout_keeps_the_videos_and_a_retry_with_the_same_files_succeeds(store, results, syn):
    put_pair(store, syn["left"], syn["right"])
    r1 = run(store, results, SLEEP, limits=RunnerLimits(allow_full=True, timeout_s=2.0, heartbeat_s=0.5, kill_grace_s=1.0))
    assert r1.result["outcome"] == "timeout" and not r1.result["cleanup"]["videos_deleted"] and results.load_result(r1.job_code)["outcome"] == "timeout"
    assert_retained(store)
    r2 = run(store, results, FULL)                                                                  # retry: no re-upload
    assert r2.result["outcome"] == "completed" and r2.result["cleanup"]["videos_deleted"] and files(store) == []


def test_recoverable_worker_failure_keeps_videos_and_retries_are_bounded(store, results, syn):
    put_pair(store, syn["left"], syn["right"])
    for n in range(3):
        r = run(store, results, CRASH); assert r.result["outcome"] == "worker_crashed"; assert_retained(store)
        assert all(u.attempts == n + 1 for u in store._uploads.values())
    r4 = run(store, results, FULL)                                                                  # the same upload is not retried forever
    assert r4.result["outcome"] == "rejected" and "failed 3 times" in r4.result["error"] and len(files(store)) == 2


def test_memory_guard_stop_allows_one_controlled_retry_then_blocks(store, results, syn, tmp_path):
    put_pair(store, syn["left"], syn["right"])
    hot = fake_cgroup(tmp_path / "hot", 1000, anon_mb=950)                                          # over 90 % of the limit: anonymous memory
    r = run(store, results, SLEEP, limits=RunnerLimits(allow_full=True, heartbeat_s=0.5, kill_grace_s=1.0, start_headroom_fraction=1.0), cgroup_root=hot)
    assert r.result["outcome"] == "memory_limit" and "memory guard" in r.result["error"] and r.result["memory_guard"]["active"] is True
    assert_retained(store); assert all(u.memory_kills == 1 for u in store._uploads.values())
    ok = run(store, results, FULL, cgroup_root=fake_cgroup(tmp_path / "ok", 1000, anon_mb=300))        # controlled retry once memory has recovered: completes
    assert ok.result["outcome"] == "completed" and files(store) == []
    # a second pair that gets stopped twice is blocked
    put_pair(store, syn["left"], syn["right"], B)
    for _ in range(2):
        rr = run(store, results, SLEEP, owner=B, limits=RunnerLimits(allow_full=True, heartbeat_s=0.5, kill_grace_s=1.0, start_headroom_fraction=1.0), cgroup_root=hot)
        assert rr.result["outcome"] == "memory_limit"
    blocked = run(store, results, FULL, owner=B, cgroup_root=fake_cgroup(tmp_path / "ok2", 1000, anon_mb=300))
    assert blocked.result["outcome"] == "rejected" and "memory guard stopped this calibration twice" in blocked.result["error"] and len(files(store)) == 2


def test_a_retry_does_not_start_while_memory_has_not_recovered(store, results, syn, tmp_path):
    put_pair(store, syn["left"], syn["right"])
    r = run(store, results, FULL, cgroup_root=fake_cgroup(tmp_path / "busy", 1000, anon_mb=700))      # 70 % > the 60 % start headroom
    assert r.result["outcome"] == "rejected" and "not started until it drops below 60%" in r.result["error"]
    assert len(files(store)) == 2 and all(u.attempts == 0 for u in store._uploads.values())            # nothing was run, nothing counted, nothing deleted


def test_validation_rejection_of_unusable_videos_still_frees_the_disk(store, results, tmp_path):
    bad, bad2 = tmp_path / "bad.bin", tmp_path / "bad2.bin"; bad.write_bytes(os.urandom(300_000)); bad2.write_bytes(os.urandom(300_000))
    put_pair(store, bad, bad2)
    r = run(store, results, FULL)
    assert r.result["outcome"] == "rejected" and files(store) == []


# ================================================================== 5. abandonment, lock, ownership
def clocked_store(tmp_path, **lim):
    t = [1000.0]
    return ChunkStore(Limits(max_file_bytes=64 * MiB, lease_heartbeat_timeout_s=60, retry_retention_s=600, job_max_s=1000, **lim), root=tmp_path / "cs", clock=lambda: t[0]), t


def test_retained_videos_expire_automatically_and_activity_does_not_extend_them(tmp_path, results, syn):
    s, t = clocked_store(tmp_path)
    try:
        put_pair(s, syn["left"], syn["right"])
        r = runner(s, results, "time.sleep(2); sys.exit(3)"); r.start(); assert wait_for(lambda: r.status()["state"] == "running")
        t[0] += 1500; r.join(30)                                                                      # the pair is older than job_max_s (1000 s) when the long run fails
        assert r.result["outcome"] == "worker_crashed"
        s.sweep(); assert len(files(s)) == 2                                                         # NOT swept at once: the retry window overrides the old age limit
        t[0] += 599; s.sweep(); assert len(files(s)) == 2
        t[0] += 2; s.sweep(); assert files(s) == []                                                  # 600 s window over: gone
    finally:
        s.close()


def test_a_new_lease_clears_the_retry_deadline_and_a_success_deletes(tmp_path, results, syn):
    s, t = clocked_store(tmp_path)
    try:
        put_pair(s, syn["left"], syn["right"]); run(s, results, CRASH)
        assert all(u.retained_until for u in s._uploads.values())
        r = run(s, results, FULL); assert r.result["outcome"] == "completed" and files(s) == []
    finally:
        s.close()


def test_session_lock_deletes_retained_videos_and_forgets_the_recovery_code(store, results, syn):
    put_pair(store, syn["left"], syn["right"]); code = new_code(); store.set_recovery(A, code)
    run(store, results, CRASH, job_code=code); assert_retained(store)
    assert store.release_owner(A) == 2 and files(store) == [] and store.job_status(A) == dict(slots={}, lease=None)
    with pytest.raises(UploadError) as e: store.adopt(code, B)
    assert e.value.code == "not_found"


def test_lock_during_a_run_stops_the_worker_and_deletes_everything(store, results, syn, tmp_path):
    put_pair(store, syn["left"], syn["right"]); pidfile = tmp_path / "pid"
    r = runner(store, results, f"open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(120)"); r.start()
    assert wait_for(lambda: pidfile.exists() and r.status()["state"] == "running")
    store.release_owner(A); r.join(30)
    assert wait_for(lambda: not alive(int(pidfile.read_text())), 10) and wait_for(lambda: files(store) == [], 10)       # explicit lock deletes, no retry window


def test_abandoned_lease_deletes_even_with_the_retry_policy(tmp_path, results, syn):
    s, t = clocked_store(tmp_path)
    try:
        put_pair(s, syn["left"], syn["right"]); lease = s.acquire_lease(A)
        t[0] += 3600; s.sweep()                                                                      # heartbeat silent for an hour: the store ends the job
        assert files(s) == []
    finally:
        s.close()


def test_other_sessions_cannot_read_run_or_adopt_without_the_code(store, results, syn):
    put_pair(store, syn["left"], syn["right"], A); code = new_code(); store.set_recovery(A, code)
    r = run(store, results, CRASH, job_code=code); assert_retained(store)
    rb = run(store, results, FULL, owner=B); assert rb.result["outcome"] == "rejected" and len(files(store)) == 2        # B cannot run A's retained videos
    for bad in (new_code(), "short", None, "x" * 70):
        with pytest.raises(UploadError): store.adopt(bad, B)
    assert set(store.job_status(B)["slots"]) == set() and set(store.job_status(A)["slots"]) == {"left", "right"}
    assert results.result_bytes(new_code()) is None                                                 # results are found by code only; there is no listing
    assert store.adopt(code, B)["adopted"] == 2                                                      # the code holder can take the retained videos over ...
    assert set(store.job_status(B)["slots"]) == {"left", "right"} and store.job_status(A)["slots"] == {}   # ... and A no longer holds them
    rr = run(store, results, FULL, owner=B); assert rr.result["outcome"] == "completed" and files(store) == []


def test_adopt_is_refused_while_the_job_runs_or_when_the_new_session_has_videos(store, results, syn):
    put_pair(store, syn["left"], syn["right"], A); code = new_code(); store.set_recovery(A, code)
    r = runner(store, results, SLEEP, job_code=code); r.start(); assert wait_for(lambda: r.status()["state"] == "running")
    with pytest.raises(UploadError) as e: store.adopt(code, B)
    assert e.value.code == "job_running"
    r.cancel(); r.join(30); assert_retained(store)
    from test_runner import put as put1
    put1(store, B, "left", syn["left"])
    with pytest.raises(UploadError) as e2: store.adopt(code, B)
    assert e2.value.code == "owner_busy" and set(store.job_status(A)["slots"]) == {"left", "right"}


# ================================================================== 6. timeout and lease: 5400 s
def test_5400_second_timeout_is_supported_and_the_lease_covers_it():
    lim = RunnerLimits.from_env({"F8_CALIB_TIMEOUT_S": "5400", "F8_ENABLE_FULL_CALIBRATION": "1"})
    st = Limits()                                                                                    # production default lease: 7200 s
    margin = lease_margin_s(lim)
    assert lim.timeout_s == 5400.0 and lim.allow_full and st.lease_max_s - margin >= 5400.0 and margin >= 2 * lim.inspect_timeout_s + lim.kill_grace_s
    assert max(1.0, min(lim.timeout_s, st.lease_max_s - margin)) == 5400.0 and st.lease_heartbeat_timeout_s < 5400.0       # the worker's timeout fits inside the lease


def test_runner_reports_the_effective_timeout_and_clips_to_the_lease(store, results):
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(timeout_s=5400.0, allow_full=True), results=results)
    assert r.timeout_s == 5400.0 and r.timeout_info == dict(requested_s=5400.0, effective_s=5400.0, lease_max_s=7200.0, lease_margin_s=r.lease_margin_s, clipped_by_lease=False)
    small = ChunkStore(Limits(lease_max_s=3000.0), root=store.root.parent / "small")
    try:
        c = CalibrationJobRunner(small, A, SETTINGS, RunnerLimits(timeout_s=5400.0, allow_full=True), results=results)
        assert c.timeout_s == 3000.0 - c.lease_margin_s and c.timeout_info["clipped_by_lease"] is True       # a short lease can never be outlived by the worker
    finally:
        small.close()


def test_lease_max_is_configurable_but_bounded(monkeypatch):
    for v, want in (("9000", 9000.0), ("10", 600.0), ("999999", 14400.0), ("junk", 7200.0)):
        monkeypatch.setenv("F8_LEASE_MAX_S", v); assert env_limits().lease_max_s == want
    monkeypatch.setenv("F8_RETRY_RETENTION_S", "900"); assert env_limits().retry_retention_s == 900.0


def test_full_calibration_stays_off_by_default_and_server_side_gated(store, results, syn, monkeypatch):
    monkeypatch.delenv("F8_ENABLE_FULL_CALIBRATION", raising=False)
    assert RunnerLimits().allow_full is False and RunnerLimits.from_env({"F8_CALIB_TIMEOUT_S": "5400"}).allow_full is False
    assert RunnerLimits.from_env({"F8_ENABLE_FULL_CALIBRATION": "true"}).allow_full is False
    put_pair(store, syn["left"], syn["right"])
    r = run(store, results, FULL, limits=RunnerLimits(heartbeat_s=0.5))                              # allow_full defaults to False -> the server refuses
    assert r.result["outcome"] == "rejected" and "disabled" in r.result["error"] and len(files(store)) == 2 and all(u.attempts == 0 for u in store._uploads.values())


def test_heartbeat_keeps_the_lease_alive_far_beyond_the_heartbeat_timeout_but_never_past_the_hard_limit(tmp_path, syn):
    s, t = clocked_store(tmp_path); put_pair(s, syn["left"], syn["right"]); lease = s.acquire_lease(A)
    try:
        for _ in range(40):                                                                          # 40 x 100 s = 4000 s with heartbeats every 100 s (timeout 60 s would be missed only if silent)
            t[0] += 50; s.heartbeat(A, lease["lease_id"]); s.sweep()
        assert len(files(s)) == 2 and s.job_status(A)["lease"] is not None
        t[0] = 1000 + s.limits.lease_max_s + 1
        with pytest.raises(UploadError): s.heartbeat(A, lease["lease_id"])                           # the hard deadline ends it regardless
    finally:
        s.close()


# ================================================================== 7. memory-guard inputs: missing or invalid measurements never kill, never crash
BAD = [None, float("nan"), float("inf"), float("-inf"), -5.0, "garbage", True, [], {}]


@pytest.mark.parametrize("bad", BAD)
def test_guard_memory_ignores_invalid_values(bad):
    g = guard_memory(dict(memory_max_mb=bad, memory_current_mb=bad, anon_mb=bad, file_mb=bad, shmem_mb=bad))
    assert g["limit_mb"] is None and g["used_mb"] is None and g["basis"] == "unknown"
    g2 = guard_memory(dict(memory_max_mb=1000.0, anon_mb=bad, memory_current_mb=400.0, file_mb=bad, shmem_mb=bad))
    assert g2["limit_mb"] == 1000.0 and g2["used_mb"] == 400.0 and "includes page cache" in g2["basis"]       # falls back conservatively, never to a bad number


def test_guard_memory_basis_priority_and_cloud_like_numbers():
    assert guard_memory(dict(memory_max_mb=3072.0, anon_mb=900.0, shmem_mb=0.0, file_mb=1500.0, memory_current_mb=2443.0))["used_mb"] == 900.0       # page cache not counted
    assert guard_memory(dict(memory_max_mb=3072.0, anon_mb=None, file_mb=1500.0, memory_current_mb=2443.0))["used_mb"] == pytest.approx(943.0)       # current minus cache
    assert guard_memory(dict(memory_max_mb=3072.0, anon_mb=300.0, shmem_mb=650.0))["used_mb"] == 950.0                                            # memory-backed /tmp counts
    assert guard_memory(dict(memory_max_mb=0.0, anon_mb=100.0))["limit_mb"] is None and guard_memory("not a dict")["basis"] == "unknown"
    assert guard_memory(dict(memory_max_mb=100.0, memory_current_mb=50.0, file_mb=80.0))["basis"] == "current (includes page cache)"                 # file > current is inconsistent: do not subtract


@pytest.mark.parametrize("bad", BAD)
def test_guard_never_kills_on_bad_numbers(store, results, bad):
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(allow_full=True, max_cgroup_fraction=0.9), results=results)
    import calibration_runner as CR
    mp = pytest.MonkeyPatch(); mp.setattr(CR, "read_cgroup", lambda *a, **k: dict(memory_max_mb=bad, memory_current_mb=bad, anon_mb=bad, file_mb=bad, shmem_mb=bad))
    try:
        assert r._guard({"combined_rss_mb": bad}) is None and r._guard_info["active"] is False
    finally:
        mp.undo()


def test_guard_survives_an_unreadable_cgroup_and_bad_fraction(store, results, tmp_path):
    import calibration_runner as CR
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(allow_full=True, max_cgroup_fraction=-3.0), results=results, cgroup_root=str(tmp_path / "nothing-here"))
    assert r._guard({}) is None and r._guard_operative() is False
    mp = pytest.MonkeyPatch(); mp.setattr(CR, "read_cgroup", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    try:
        assert r._read_cg() == {} and r._guard({}) is None
    finally:
        mp.undo()
    r2 = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(allow_full=True, max_cgroup_fraction=-3.0), results=results, cgroup_root=fake_cgroup(tmp_path / "c", 1000, anon_mb=950))
    assert "memory guard" in r2._guard({})                                                           # a nonsense fraction falls back to the 90 % default


def test_guard_ignores_a_big_page_cache_and_matches_the_cloud_benchmark_shape(store, results, tmp_path):
    root = fake_cgroup(tmp_path / "cloud", 3072, anon_mb=900, file_mb=1540)                          # ~2443 MB current like the Cloud benchmark, of which 1540 MB cache
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(allow_full=True), results=results, cgroup_root=root)
    assert r._guard({"combined_rss_mb": 892.0}) is None and r._guard_info["active"] and r._guard_info["threshold_mb"] == pytest.approx(0.9 * 3072)
    hot = fake_cgroup(tmp_path / "hot", 3072, anon_mb=2800, file_mb=100)
    assert "memory guard" in CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(allow_full=True), results=results, cgroup_root=hot)._guard({})


def test_require_memory_guard_refuses_a_full_run_that_the_guard_could_not_protect(store, results, syn, tmp_path):
    put_pair(store, syn["left"], syn["right"]); lim = RunnerLimits(allow_full=True, heartbeat_s=0.5, require_memory_guard=True)
    r = run(store, results, FULL, limits=lim, cgroup_root=str(tmp_path / "no-cgroup"))
    assert r.result["outcome"] == "rejected" and "memory guard cannot act" in r.result["error"] and len(files(store)) == 2 and all(u.attempts == 0 for u in store._uploads.values())
    r2 = run(store, results, FULL, limits=lim, cgroup_root=fake_cgroup(tmp_path / "ok", 3072, anon_mb=500))
    assert r2.result["outcome"] == "completed"
    assert RunnerLimits.from_env({"F8_REQUIRE_MEMORY_GUARD": "1"}).require_memory_guard and not RunnerLimits.from_env({}).require_memory_guard


# ================================================================== 8. the page
def test_page_offers_download_and_recovery_and_sets_no_permanence_claim():
    src = (Path(__file__).resolve().parent.parent / "dual_page.py").read_text()
    assert "st.download_button" in src and "_recovery(" in src and "results.result_bytes" in src and "store.adopt(" in src and "cancel_by_code" in src
    assert "temporary" in src and "erases it" in src and "Download the result" in src                  # states that storage is NOT permanent
    assert "release_owner" in src


# ================================================================== F8.11: the recovery code is private (never in a URL or a log)
def test_page_never_writes_the_code_to_the_url():
    import re
    src = (Path(__file__).resolve().parent.parent / "dual_page.py").read_text()
    uses = re.findall(r"[^\n]*query_params[^\n]*", src)
    assert len(uses) == 2 and all("del st.query_params" in u or '"job" in st.query_params' in u or u.strip().startswith("#") for u in uses), uses     # only the strip-an-old-link code
    assert "st.query_params[" not in src.replace('del st.query_params["job"]', "") and "query_params.set" not in src and "?job" not in src.replace("old `?job=`", "").replace("?job= link", "")
    assert "Copy Recovery Code" in src and "clipboard" in src and 'type="password"' in src               # copy button, manual entry, masked field


def test_the_code_is_not_logged_or_written_to_disk_in_plaintext(store, results, syn, caplog, tmp_path):
    import logging
    caplog.set_level(logging.DEBUG)
    put_pair(store, syn["left"], syn["right"]); code = new_code(); store.set_recovery(A, code)
    r = run(store, results, FULL, job_code=code)
    assert code not in caplog.text and code not in repr(r.status()) and code not in json.dumps(r.result)
    for root in (results.root, store.root):                                                           # not in any file name or file content on the results/upload disks
        for p in Path(root).rglob("*"):
            assert code not in str(p)
            if p.is_file() and p.stat().st_size < 5 * MiB and p.name.endswith((".json", ".txt")):
                assert code not in p.read_text(errors="ignore"), p
    assert RS.job_id(code) in results.result_bytes(code).decode() and code not in results.result_bytes(code).decode()    # only the public hash prefix is in the download


def test_chunk_store_never_logs_the_code_on_adopt(store, results, syn, caplog):
    import logging
    caplog.set_level(logging.DEBUG); put_pair(store, syn["left"], syn["right"]); code = new_code(); store.set_recovery(A, code)
    store.adopt(code, B)
    with pytest.raises(UploadError): store.adopt(new_code(), B)
    assert code not in caplog.text and store._recovery and code not in " ".join(store._recovery)         # only a hash is kept in memory


# ================================================================== F8.11: Cloud settings do not switch full calibration on
def test_timeout_and_memory_guard_settings_do_not_enable_full_calibration():
    env = {"F8_CALIB_TIMEOUT_S": "5400", "F8_REQUIRE_MEMORY_GUARD": "1", "F8_LEASE_MAX_S": "7200", "F8_BENCH_FRAMES": "2000", "F8_BENCH_TIMEOUT_S": "900", "F8_RETRY_RETENTION_S": "1800"}
    lim = RunnerLimits.from_env(env)
    assert lim.timeout_s == 5400.0 and lim.require_memory_guard is True and lim.benchmark_frames == 2000 and lim.benchmark_timeout_s == 900.0
    assert lim.allow_full is False                                                                    # only F8_ENABLE_FULL_CALIBRATION=1 does that
    for v in ("", "0", "true", "yes", "1 ", " 1", "11"):
        assert RunnerLimits.from_env(dict(env, F8_ENABLE_FULL_CALIBRATION=v)).allow_full is (v.strip() == "1")
    assert RunnerLimits.from_env({}).allow_full is False


def test_the_page_shows_no_full_run_button_with_those_settings_alone(monkeypatch):
    import importlib, sys, types
    monkeypatch.setenv("F8_CALIB_TIMEOUT_S", "5400"); monkeypatch.setenv("F8_REQUIRE_MEMORY_GUARD", "1"); monkeypatch.delenv("F8_ENABLE_FULL_CALIBRATION", raising=False)
    pytest.importorskip("streamlit")
    sys.modules.pop("dual_page", None); dp = importlib.import_module("dual_page")
    assert dp.LIMITS.allow_full is False and dp.LIMITS.timeout_s == 5400.0 and dp.LIMITS.require_memory_guard is True
    sys.modules.pop("dual_page", None)

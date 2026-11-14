"""F8.7: the bounded Cloud calibration benchmark, the configurable timeouts and the full-run opt-in.

Everything runs on SYNTHETIC checkerboard videos (no real or protected video is read) and the UNMODIFIED F7 module. Failure paths use fake workers that take
the same shared file locks as the real worker."""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from calibration_runner import DEFAULT_F7_DIR, WORKER, CalibrationJobRunner, RunnerError, RunnerLimits
from chunk_store import ChunkStore, Limits, MiB
from synthetic_stereo import make_stereo_pair
from test_runner import A, B, PRE, SETTINGS, alive, assert_clean, fake, files, put_pair, sha, wait_for  # noqa: F401  (shared helpers; also re-collects nothing)

HERE = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (DEFAULT_F7_DIR / "f7_calibration.py").is_file(), reason="the F7 module is not on this machine (untracked local file)")


@pytest.fixture(scope="module")
def syn(tmp_path_factory): return make_stereo_pair(tmp_path_factory.mktemp("synb"))


@pytest.fixture
def store(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB, lease_heartbeat_timeout_s=60), root=tmp_path / "store"); yield s; s.close()


def bench(store, owner=A, limits=None, timeout=240, **kw):
    r = CalibrationJobRunner(store, owner, SETTINGS, limits or RunnerLimits(heartbeat_s=1.0, benchmark_frames=300), monitor_interval=0.25, mode="benchmark", **kw)
    r.start(); r.join(timeout); assert r.done, r.status()
    return r


def fingerprint(store):
    """(name, size, mtime_ns, inode, sha256) of every stored video."""
    out = []
    for p in sorted(Path(store.root).rglob("data.bin")):
        st = p.stat(); out.append((p.parent.name, st.st_size, st.st_mtime_ns, st.st_ino, hashlib.sha256(p.read_bytes()).hexdigest()))
    return out


@pytest.fixture(scope="module")
def done(syn, tmp_path_factory):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB), root=tmp_path_factory.mktemp("br") / "store"); put_pair(s, syn["left"], syn["right"])
    before = fingerprint(s); src_before = {k: hashlib.sha256(Path(syn[k]).read_bytes()).hexdigest() for k in ("left", "right")}
    r = CalibrationJobRunner(s, A, SETTINGS, RunnerLimits(heartbeat_s=1.0, benchmark_frames=300), monitor_interval=0.25, mode="benchmark"); r.start(); r.join(240)
    yield dict(r=r, store=s, before=before, after=fingerprint(s), res=r.result, rep=r.result["benchmark_report"], syn=syn, src_before=src_before)
    s.close()


# ---------------------------------------------------------------- the report
def test_report_has_everything_the_cloud_test_needs(done):
    rep, res = done["rep"], done["res"]
    assert res["outcome"] == "completed" and res["mode"] == "benchmark" and rep["completed"]
    assert rep["frames_processed"] == 600 and rep["frames_requested_per_video"] == 300 and not rep["frames_processed_is_estimate"]
    assert rep["detection_attempts"] == 120 and rep["checkerboard_detections"] > 0 and 0 < rep["detection_rate"] <= 1                    # STRIDE 5 -> one attempt per 5 frames
    assert set(rep["per_video"]) == {"left", "right"} and all(v["frames_processed"] == 300 and v["wall_s"] > 0 for v in rep["per_video"].values())
    assert rep["wall_clock_s"] > 0 and rep["frames_per_s"] > 0
    assert rep["cpu"]["worker_cpu_s"] > 0 and rep["cpu"]["host_cpu_count"] and "throttled_periods" in rep["cpu"] and "throttling_measurable" in rep["cpu"] and "container_cpu_measurable" in rep["cpu"]
    assert rep["memory"]["worker_self_peak_rss_mb"] > 0 and rep["memory"]["worker_tree_peak_rss_mb"] > 0 and rep["memory"]["combined_peak_rss_mb"] > 0
    assert {"container_peak_mb", "container_limit_mb", "container_measurable", "oom_kills"} <= set(rep["memory"])
    assert rep["disk"]["temp_disk_peak_mb"] > 0 and rep["disk"]["temp_disk_free_min_mb"] > 0
    assert rep["limits"] == dict(rep["limits"], benchmark_timeout_s=300.0, timed_out=False, cancelled=False, memory_guard_triggered=False)
    assert rep["projection"]["full_length_frames"] > 0 and "estimate" in rep["projection"]["note"]
    json.dumps(res, default=str)                                                                                                           # serialisable
    assert len(json.dumps(res, default=str)) < 100_000                                                                                     # bounded


def test_the_benchmark_only_detects_it_does_not_calibrate_or_dump_detections(done):
    res = done["res"]
    for k in ("model", "assessment", "contract", "calibration", "detections"):
        assert k not in res or res[k] is None, k
    assert set(res["stage_timings"]) == {"verify", "detect_left", "detect_right", "verify_after"}                                          # no solve / assess stage ran


def test_f7_is_used_unmodified_and_matches_a_direct_call(done, syn):
    sys.path.insert(0, str(DEFAULT_F7_DIR))
    import f7_calibration as K
    d = K.extract_detections(syn["left"], (8, 6), K.STRIDE, None, max_frames=300)
    left = done["rep"]["per_video"]["left"]
    assert (left["frames_processed"], left["detections"]) == (d["n_frames"], len(d["detections"]))                                         # same function, same numbers
    assert done["res"]["f7"]["stride"] == K.STRIDE and done["res"]["f7"]["f7_calibration"]["sha256"] == hashlib.sha256((DEFAULT_F7_DIR / "f7_calibration.py").read_bytes()).hexdigest()


# ---------------------------------------------------------------- the videos are never modified or deleted
def test_stored_videos_are_byte_identical_and_still_present_afterwards(done):
    assert done["before"] and done["before"] == done["after"]                                                                             # size, mtime, inode AND content hash
    assert done["rep"]["integrity"] == dict(sha256_verified_before_use=True, videos_unchanged_after=True)
    assert done["res"]["cleanup"]["videos_deleted"] is False and done["res"]["cleanup"]["videos_kept_for_next_step"] is True
    assert done["store"].job_status(A)["lease"] is None and all(s["leased"] is False for s in done["store"].job_status(A)["slots"].values())   # lease released, files free again


def test_the_source_files_outside_the_store_are_untouched(done):
    for k in ("left", "right"):
        assert Path(done["syn"][k]).is_file() and hashlib.sha256(Path(done["syn"][k]).read_bytes()).hexdigest() == done["src_before"][k]


def test_the_worker_refuses_a_video_outside_the_upload_store(tmp_path, syn):
    """Pointing the worker at an 'original' that is not in the store (as a mistaken path or a symlink could) must fail before anything is opened."""
    out = tmp_path / "job"; out.mkdir(); root = tmp_path / "store"; root.mkdir()
    link = root / "sneaky.mp4"; link.symlink_to(syn["left"])                                                                               # a symlink inside the store pointing out of it
    before = {k: hashlib.sha256(Path(syn[k]).read_bytes()).hexdigest() for k in ("left", "right")}
    for left in (syn["left"], str(link)):
        spec = dict(left_path=str(left), right_path=syn["right"], cols=8, rows=6, square_mm=20.0, baseline_mm=None, f7_dir=str(DEFAULT_F7_DIR), out_dir=str(out), parent_pid=None,
                    mode="benchmark", max_frames=30, allowed_root=str(root), expected_sha256={})
        (out / "spec.json").write_text(json.dumps(spec))
        rc = subprocess.run([sys.executable, "-I", str(WORKER), str(out / "spec.json")], capture_output=True, timeout=120).returncode
        res = json.loads((out / "result.json").read_text())
        assert rc == 3 and not res["ok"] and "outside the upload store" in res["error"] and "benchmark" not in res
    assert {k: hashlib.sha256(Path(syn[k]).read_bytes()).hexdigest() for k in ("left", "right")} == before


def test_the_worker_source_has_no_way_to_delete_or_rewrite_a_video():
    src = (HERE / "calibration_worker.py").read_text()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())                                                                          # ignore comments
    for bad in ("unlink", "rmtree", "os.remove", "shutil", "os.rename", "truncate", "os.replace(left", "os.replace(right", "chmod", "os.utime"):
        assert bad not in code, bad
    opens = re.findall(r"open\(([^)]*)\)", code)
    assert all(('"rb"' in o) or ('"w"' in o and "tmp" in o) or "read_text" in o for o in opens), opens                                      # videos only 'rb'; the one write is the job's own tmp json
    assert 'open(tmp, "w")' in code and code.count('"w"') == 1


def test_a_modified_stored_file_is_caught_by_the_sha256_check_and_not_touched_further(store, syn):
    put_pair(store, syn["left"], syn["right"])
    victim = sorted(Path(store.root).rglob("data.bin"))[0]; raw = bytearray(victim.read_bytes()); raw[len(raw) // 2] ^= 0xFF; victim.write_bytes(bytes(raw))      # same size, one flipped byte
    snap = fingerprint(store)
    r = bench(store)
    assert r.result["outcome"] == "failed" and "SHA-256" in r.result["error"] and "benchmark" not in r.result
    assert fingerprint(store) == snap and len(files(store)) == 2                                                                          # still there, still exactly as found
    rep = r.result["benchmark_report"]                                                                                                    # a report is still produced, and it does not claim success
    assert rep["completed"] is False and rep["integrity"]["sha256_verified_before_use"] is False and rep["integrity"]["videos_unchanged_after"] is None and rep["frames_processed"] is None
    assert store.job_status(A)["lease"] is None


# ---------------------------------------------------------------- bounded
def test_frames_are_hard_capped_whatever_is_requested(store, syn):
    put_pair(store, syn["left"], syn["right"])
    r = bench(store, limits=RunnerLimits(heartbeat_s=1.0, benchmark_frames=90, benchmark_frames_cap=90), benchmark_frames=10 ** 9)
    assert r.bench_frames == 90 and r.result["benchmark_report"]["frames_processed"] == 180
    r2 = bench(store, benchmark_frames=0)
    assert r2.bench_frames == 300 and r2.result["benchmark_report"]["frames_processed"] == 600                                           # 0/None -> the configured default
    r3 = bench(store, limits=RunnerLimits(heartbeat_s=1.0, benchmark_frames=300), benchmark_frames=-5)
    assert r3.bench_frames == 1


def test_a_benchmark_never_reads_more_frames_than_the_video_has(store, syn):
    put_pair(store, syn["left"], syn["right"])
    r = bench(store, limits=RunnerLimits(heartbeat_s=1.0, benchmark_frames=3000))
    assert r.result["benchmark_report"]["per_video"]["left"]["frames_processed"] == r.result["benchmark_report"]["per_video"]["left"]["video_frame_count"]


def test_unknown_mode_is_refused(store):
    with pytest.raises(RunnerError): CalibrationJobRunner(store, A, SETTINGS, mode="solve-everything")


# ---------------------------------------------------------------- timeout / cancel
def test_benchmark_timeout_is_reported_with_progress_and_cleans_up_the_process(tmp_path, store, syn):
    put_pair(store, syn["left"], syn["right"]); pidfile = tmp_path / "pid.txt"
    body = (f"open({str(pidfile)!r}, 'w').write(str(os.getpid())); json.dump(dict(stage='detect_left', fraction=0.3, frames=77), open(os.path.join(spec['out_dir'], 'progress.json'), 'w')); time.sleep(120)")
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(benchmark_timeout_s=2.5, heartbeat_s=0.5, kill_grace_s=1.0), worker_cmd=fake(body), monitor_interval=0.25, mode="benchmark"); r.start(); r.join(30)
    rep = r.result["benchmark_report"]
    assert r.result["outcome"] == "timeout" and rep["limits"]["timed_out"] is True and rep["limits"]["benchmark_timeout_s"] == 2.5 and not rep["completed"]
    assert rep["frames_processed"] == 77 and rep["frames_processed_is_estimate"] is True and rep["projection"] is None
    assert wait_for(lambda: not alive(int(pidfile.read_text())), 10)
    assert len(files(store)) == 2 and store.job_status(A)["lease"] is None                                                                # uploads kept, lease released


def test_benchmark_cancel_is_reported(tmp_path, store, syn):
    put_pair(store, syn["left"], syn["right"]); pidfile = tmp_path / "pid.txt"
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(heartbeat_s=0.5, kill_grace_s=1.0), monitor_interval=0.25, mode="benchmark",
                             worker_cmd=fake(f"open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(120)")); r.start()
    assert wait_for(lambda: pidfile.exists() and r.status()["state"] == "running")
    r.cancel(); r.join(30)
    lim = r.result["benchmark_report"]["limits"]
    assert r.result["outcome"] == "cancelled" and lim["cancelled"] is True and lim["timed_out"] is False and not alive(int(pidfile.read_text()))
    assert len(files(store)) == 2


def test_a_crashed_benchmark_worker_is_reported_and_keeps_the_uploads(store, syn):
    put_pair(store, syn["left"], syn["right"])
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(heartbeat_s=0.5), monitor_interval=0.25, mode="benchmark", worker_cmd=fake("os._exit(9)")); r.start(); r.join(30)
    assert r.result["outcome"] == "worker_crashed" and len(files(store)) == 2 and store.job_status(A)["lease"] is None


# ---------------------------------------------------------------- full-calibration timeout and opt-in
def test_full_timeout_is_configurable_and_cannot_outlive_the_lease(store):
    assert CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(timeout_s=900.0)).timeout_s == 900.0
    assert CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(timeout_s=10 ** 6)).timeout_s == store.limits.lease_max_s - 120.0       # clamped below the store's hard lease limit
    assert CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(timeout_s=900.0), mode="benchmark").timeout_s == 300.0                    # a benchmark has its own short one


def test_limits_from_env_parse_clamp_and_ignore_garbage():
    d = RunnerLimits()
    assert RunnerLimits.from_env({}) == RunnerLimits(allow_full=False)                                                                     # production default: full runs OFF
    e = RunnerLimits.from_env({"F8_CALIB_TIMEOUT_S": "1800", "F8_BENCH_TIMEOUT_S": "120", "F8_BENCH_FRAMES": "450", "F8_ENABLE_FULL_CALIBRATION": "1"})
    assert (e.timeout_s, e.benchmark_timeout_s, e.benchmark_frames, e.allow_full) == (1800.0, 120.0, 450, True)
    c = RunnerLimits.from_env({"F8_CALIB_TIMEOUT_S": "1", "F8_BENCH_TIMEOUT_S": "99999", "F8_BENCH_FRAMES": "99999999"})
    assert (c.timeout_s, c.benchmark_timeout_s, c.benchmark_frames) == (60.0, 900.0, d.benchmark_frames_cap)
    for bad in ("abc", "", "nan", "inf", "-5"):
        g = RunnerLimits.from_env({"F8_CALIB_TIMEOUT_S": bad, "F8_BENCH_TIMEOUT_S": bad, "F8_BENCH_FRAMES": bad})
        assert d.min_duration_s and 60.0 <= g.timeout_s <= 14400.0 and 10.0 <= g.benchmark_timeout_s <= 900.0 and 30 <= g.benchmark_frames <= d.benchmark_frames_cap
    for v in ("true", "yes", "0", "2", " "):
        assert RunnerLimits.from_env({"F8_ENABLE_FULL_CALIBRATION": v}).allow_full is False                                                # only the exact value "1"


def test_a_full_calibration_is_refused_when_not_enabled_and_starts_nothing(store, syn):
    put_pair(store, syn["left"], syn["right"]); snap = fingerprint(store)
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(allow_full=False), monitor_interval=0.25, worker_cmd=lambda *a: pytest.fail("a worker was started"))
    r.start(); r.join(30)
    assert r.result["outcome"] == "rejected" and "disabled" in r.result["error"] and "F8_ENABLE_FULL_CALIBRATION" in r.result["error"]
    assert fingerprint(store) == snap and store.job_status(A)["lease"] is None                                                            # never even leased or touched the videos


def test_the_page_only_starts_a_full_run_from_an_explicit_gated_button():
    src = (HERE / "dual_page.py").read_text()
    sites = re.findall(r"CalibrationJobRunner\([^\n]*", src)
    assert len(sites) == 2 and sum('mode="calibrate"' in s for s in sites) == 1 and sum('mode="benchmark"' in s for s in sites) == 1
    full = src.index('mode="calibrate"')
    gate = src.rindex("if LIMITS.allow_full:", 0, full); btn = src.rindex("button(", 0, full)
    assert gate < btn < full and "disabled=blocked or not sure" in src[btn:full + 120]                                                      # opt-in setting AND a ticked confirmation AND a click
    assert "Run full calibration" not in src[:gate]                                                                                       # no other full-run trigger


# ---------------------------------------------------------------- ownership and cleanup survive benchmark mode
def test_another_session_cannot_benchmark_these_videos(store, syn):
    put_pair(store, syn["left"], syn["right"]); snap = fingerprint(store)
    r = bench(store, owner=B)
    assert r.result["outcome"] == "rejected" and fingerprint(store) == snap


def test_only_one_job_at_a_time_and_a_second_benchmark_works_after_the_first(store, syn):
    put_pair(store, syn["left"], syn["right"])
    first = bench(store); second = bench(store)
    assert first.result["outcome"] == second.result["outcome"] == "completed" and len(files(store)) == 2


def test_kept_uploads_are_deleted_by_lock_or_by_the_retention_sweep(tmp_path, syn):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB, complete_retention_s=1.0), root=tmp_path / "s1"); put_pair(s, syn["left"], syn["right"])
    assert bench(s).result["outcome"] == "completed" and len(files(s)) == 2
    time.sleep(1.3); s.sweep(); assert_clean(s, A); s.close()                                                                             # retention still applies after a benchmark
    s = ChunkStore(Limits(max_file_bytes=64 * MiB), root=tmp_path / "s2"); put_pair(s, syn["left"], syn["right"])
    assert bench(s).result["outcome"] == "completed"
    s.release_owner(A); assert_clean(s, A); s.close()                                                                                     # 'Lock this session' deletes them


def test_a_full_run_after_a_benchmark_still_deletes_both_videos(store, syn):
    put_pair(store, syn["left"], syn["right"])
    assert bench(store).result["outcome"] == "completed"
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(heartbeat_s=1.0, allow_full=True), monitor_interval=0.25); r.start(); r.join(240)
    assert r.result["outcome"] == "completed" and r.result["cleanup"]["videos_deleted"] is True and r.result.get("benchmark_report") is None
    assert_clean(store)


# ---------------------------------------------------------------- memory guard sees a memory-backed /tmp
def test_memory_guard_counts_shared_memory_but_not_page_cache(store, monkeypatch):
    import calibration_runner as CR
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits())                                                                          # fraction 0.90
    cg = lambda **kw: (lambda: dict(dict(version=2, memory_max_mb=1000.0, memory_current_mb=None, anon_mb=None, shmem_mb=None), **kw))
    monkeypatch.setattr(CR, "read_cgroup", cg(anon_mb=300.0, shmem_mb=650.0)); assert "memory guard" in r._guard({})                     # tmpfs-held uploads + worker heap exceed 90 %
    monkeypatch.setattr(CR, "read_cgroup", cg(anon_mb=300.0, shmem_mb=100.0)); assert r._guard({}) is None
    monkeypatch.setattr(CR, "read_cgroup", cg(anon_mb=300.0, shmem_mb=None, memory_current_mb=990.0)); assert r._guard({}) is None      # reclaimable page cache alone never trips it
    monkeypatch.setattr(CR, "read_cgroup", cg(memory_current_mb=950.0)); assert "memory guard" in r._guard({})                           # no anon breakdown: fall back to total
    monkeypatch.setattr(CR, "read_cgroup", lambda: dict(version=None, memory_max_mb=None, anon_mb=None, shmem_mb=None, memory_current_mb=None)); assert r._guard({}) is None   # no cgroup: no guess


def test_full_calibration_is_off_by_default_even_for_a_directly_constructed_runner(store, syn):
    assert RunnerLimits().allow_full is False and RunnerLimits.from_env({}).allow_full is False
    put_pair(store, syn["left"], syn["right"]); snap = fingerprint(store)
    r = CalibrationJobRunner(store, A, SETTINGS, monitor_interval=0.25, worker_cmd=lambda *a: pytest.fail("a worker was started")); r.start(); r.join(30)   # default limits, default mode
    assert r.result["outcome"] == "rejected" and "disabled" in r.result["error"] and fingerprint(store) == snap

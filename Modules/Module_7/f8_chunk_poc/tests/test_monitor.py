"""F8.5: resource monitoring. The Linux code paths (/proc, smaps_rollup, cgroup v1/v2, mounts) are tested against fake trees so they are verified on any machine."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import monitor as M
from monitor import MiB, ResourceMonitor

PAGE = os.sysconf("SC_PAGE_SIZE")


def write(p, text):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)


def fake_proc(root, procs):
    """procs: {pid: (ppid, rss_mb, cpu_ticks, pss_kb or None)}  -> a fake /proc"""
    (Path(root) / "self").mkdir(parents=True, exist_ok=True)
    for pid, (ppid, rss_mb, ticks, pss_kb) in procs.items():
        write(f"{root}/{pid}/stat", f"{pid} (my proc (x)) S {ppid} 1 1 0 -1 4194304 100 0 0 0 {ticks} {ticks // 2} 0 0 20 0 1 0 100 1000 {int(rss_mb * MiB / PAGE)}\n")
        write(f"{root}/{pid}/statm", f"1000 {int(rss_mb * MiB / PAGE)} 10 5 0 100 0\n")
        if pss_kb is not None:
            write(f"{root}/{pid}/smaps_rollup", f"00400000-ffffffff ---p 00000000 00:00 0 [rollup]\nRss: {pss_kb * 2} kB\nPss: {pss_kb} kB\nPss_Anon: 1 kB\n")


# ---------------------------------------------------------------- /proc
def test_process_table_and_tree_from_a_fake_proc(tmp_path):
    fake_proc(tmp_path, {100: (1, 200.0, 500, 150_000), 200: (100, 400.0, 3000, 300_000), 201: (200, 100.0, 800, 60_000), 300: (1, 50.0, 10, 5_000)})
    t = M.snapshot_processes(str(tmp_path))
    assert set(t) == {100, 200, 201, 300} and abs(t[200]["rss_mb"] - 400.0) < 0.01 and t[201]["ppid"] == 200
    assert abs(t[200]["cpu_s"] - (3000 + 1500) / M._CLK) < 1e-9                                    # utime + stime, command name with spaces/parentheses parsed
    assert sorted(M.tree_pids(t, 200)) == [200, 201] and sorted(M.tree_pids(t, 100)) == [100, 200, 201] and M.tree_pids(t, 999) == [999]


def test_pss_is_read_per_process_and_missing_is_none(tmp_path):
    fake_proc(tmp_path, {200: (100, 400.0, 1, 300_000), 201: (200, 10.0, 1, None)})
    assert abs(M.read_pss_mb(200, str(tmp_path)) - 300_000 / 1024) < 1e-6
    assert M.read_pss_mb(201, str(tmp_path)) is None and M.read_pss_mb(555, str(tmp_path)) is None


def test_unreadable_proc_is_empty_not_an_error(tmp_path):
    assert isinstance(M.snapshot_processes(str(tmp_path / "nope")), dict)                               # no such /proc: falls back to ps (or {}); never raises


# ---------------------------------------------------------------- cgroup
def test_cgroup_v2(tmp_path):
    write(tmp_path / "memory.current", f"{800 * MiB}\n"); write(tmp_path / "memory.peak", f"{1200 * MiB}\n"); write(tmp_path / "memory.max", f"{2700 * MiB}\n")
    write(tmp_path / "memory.stat", f"anon {500 * MiB}\nfile {250 * MiB}\nshmem {10 * MiB}\nkernel 5\n")
    write(tmp_path / "cpu.stat", "usage_usec 4500000\nuser_usec 4000000\nnr_periods 100\nnr_throttled 7\nthrottled_usec 2500000\n")
    write(tmp_path / "cpu.max", "200000 100000\n"); write(tmp_path / "memory.events", "low 0\nhigh 0\nmax 3\noom 1\noom_kill 1\n")
    c = M.read_cgroup(str(tmp_path))
    assert c["version"] == 2 and c["memory_current_mb"] == 800 and c["memory_peak_mb"] == 1200 and c["memory_max_mb"] == 2700
    assert (c["anon_mb"], c["file_mb"], c["shmem_mb"]) == (500, 250, 10) and c["cpu_usage_s"] == 4.5 and c["nr_throttled"] == 7 and c["throttled_s"] == 2.5
    assert c["cpu_limit_cores"] == 2.0 and c["oom_events"] == 1


def test_cgroup_v2_unlimited(tmp_path):
    write(tmp_path / "memory.current", f"{100 * MiB}\n"); write(tmp_path / "memory.max", "max\n"); write(tmp_path / "cpu.max", "max 100000\n")
    c = M.read_cgroup(str(tmp_path)); assert c["memory_max_mb"] is None and c["cpu_limit_cores"] is None and c["memory_peak_mb"] is None


def test_cgroup_v1(tmp_path):
    write(tmp_path / "memory/memory.usage_in_bytes", f"{600 * MiB}\n"); write(tmp_path / "memory/memory.max_usage_in_bytes", f"{900 * MiB}\n")
    write(tmp_path / "memory/memory.limit_in_bytes", f"{1500 * MiB}\n"); write(tmp_path / "memory/memory.stat", f"cache {100 * MiB}\nrss {400 * MiB}\n")
    write(tmp_path / "cpu/cpu.stat", "nr_periods 50\nnr_throttled 4\nthrottled_time 3000000000\n"); write(tmp_path / "cpu/cpu.cfs_quota_us", "150000\n"); write(tmp_path / "cpu/cpu.cfs_period_us", "100000\n")
    c = M.read_cgroup(str(tmp_path))
    assert c["version"] == 1 and c["memory_current_mb"] == 600 and c["memory_peak_mb"] == 900 and c["memory_max_mb"] == 1500 and c["anon_mb"] == 400 and c["file_mb"] == 100
    assert c["nr_throttled"] == 4 and c["throttled_s"] == 3.0 and c["cpu_limit_cores"] == 1.5


def test_v1_unlimited_memory_is_none(tmp_path):
    write(tmp_path / "memory/memory.usage_in_bytes", "1000\n"); write(tmp_path / "memory/memory.limit_in_bytes", "9223372036854771712\n")
    assert M.read_cgroup(str(tmp_path))["memory_max_mb"] is None


def test_no_cgroup_is_graceful(tmp_path):
    c = M.read_cgroup(str(tmp_path / "nothing")); assert c["version"] is None and all(v is None for k, v in c.items() if k != "version")


# ---------------------------------------------------------------- mounts (is the temp dir memory-backed?)
def test_mount_info_detects_tmpfs_and_disk(tmp_path):
    base = os.path.realpath(tmp_path)                                                                  # macOS resolves /var -> /private/var; mounts list real paths
    mounts = tmp_path / "mounts"
    write(mounts, f"overlay {base} overlay rw,relatime 0 0\ntmpfs {base}/ram tmpfs rw,nosuid,size=65536k 0 0\n/dev/sda1 {base}/data ext4 rw 0 0\n")
    r = M.mount_info(f"{base}/ram/x/y", str(mounts)); assert r["fstype"] == "tmpfs" and r["memory_backed"] is True
    r = M.mount_info(f"{base}/other/thing", str(mounts)); assert r["fstype"] == "overlay" and r["memory_backed"] is False
    assert M.mount_info(f"{base}/data/v", str(mounts))["fstype"] == "ext4" and M.mount_info(f"{base}/ram", str(tmp_path / "missing")) == {}


def test_probe_environment_has_no_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("F8_PASSCODE", "super-secret-passcode"); monkeypatch.setenv("SOME_TOKEN", "tok-123")
    p = M.probe_environment(tmp_path)
    text = json.dumps(p)
    assert "super-secret" not in text and "tok-123" not in text and {"platform", "python", "cpu_count", "cgroup", "temp_dir_mount", "temp_disk"} <= set(p)


# ---------------------------------------------------------------- aggregation (deterministic: fake clock, fake process table)
class FakeClock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def test_stage_by_stage_aggregates_with_fake_inputs(tmp_path):
    clk = FakeClock(); n = {"i": 0}
    cg = tmp_path / "cg"; write(cg / "memory.current", f"{500 * MiB}\n"); write(cg / "memory.stat", f"anon {300 * MiB}\nfile {100 * MiB}\n")
    write(cg / "cpu.stat", "usage_usec 0\nnr_throttled 2\nthrottled_usec 0\n")
    proc = tmp_path / "proc"; fake_proc(proc, {10: (1, 100.0, 0, 90_000), 20: (10, 300.0, 0, 250_000), 21: (20, 50.0, 0, 40_000)})

    def snap():
        n["i"] += 1
        return {10: dict(ppid=1, rss_mb=100.0 + n["i"], cpu_s=0.0), 20: dict(ppid=10, rss_mb=300.0 + 10 * n["i"], cpu_s=n["i"] * 1.0), 21: dict(ppid=20, rss_mb=50.0, cpu_s=n["i"] * 0.5)}

    disk = {"used": 10.0}
    mon = ResourceMonitor(worker_pid=lambda: 20, disk=lambda: (disk["used"], 5000.0), parent_pid=10, proc_root=str(proc), cgroup_root=str(cg), clock=clk, snapshot=snap, sample_on_mark=False)
    mon.mark("detect_left")
    for _ in range(3):
        clk.t += 1; disk["used"] += 100; mon.sample_once()
    write(cg / "cpu.stat", "usage_usec 0\nnr_throttled 6\nthrottled_usec 0\n")
    mon.mark("solve")
    clk.t += 2; mon.sample_once()
    s = mon.stop()
    d, v = s["stages"][0], s["stages"][1]
    assert d["stage"] == "detect_left" and d["wall_s"] == 3.0 and d["samples"] == 3
    assert d["worker_rss_max_mb"] == pytest.approx((300.0 + 10 * 3) + 50.0)                                                                   # third sample: 330 + 50
    assert d["parent_rss_max_mb"] == pytest.approx(103.0) and d["tree_rss_max_mb"] == pytest.approx(103.0 + 380.0)
    assert d["tree_pss_max_mb"] == pytest.approx((90_000 + 250_000 + 40_000) / 1024, abs=0.1)                                                # parent + worker tree PSS, shared pages not double-counted
    assert d["cgroup_memory_max_mb"] == 500 and d["temp_disk_max_mb"] == 310.0 and d["disk_free_min_mb"] == 5000.0
    assert d["cpu_s"] == pytest.approx((3 * 1.0 + 3 * 0.5) - (1 * 1.0 + 1 * 0.5))                                                             # worker-tree CPU delta inside the stage
    assert v["stage"] == "solve" and v["throttled_periods"] == 0 and s["peak"]["worker_rss_mb"] >= d["worker_rss_max_mb"] and s["available"]["pss"] and s["available"]["cgroup"]


def test_unavailable_metrics_are_none_not_zero(tmp_path):
    mon = ResourceMonitor(worker_pid=lambda: None, interval=0.05, proc_root=str(tmp_path / "noproc"), cgroup_root=str(tmp_path / "nocg"), snapshot=lambda: {})
    mon.mark("x"); mon.sample_once(); s = mon.stop()
    st = s["stages"][0]
    assert st["worker_rss_max_mb"] is None and st["tree_pss_max_mb"] is None and st["cgroup_memory_max_mb"] is None and st["cpu_s"] is None and not s["available"]["pss"] and not s["available"]["cgroup"]


def test_samples_and_timeline_are_bounded():
    mon = ResourceMonitor(max_samples=20, snapshot=lambda: {}, cgroup_root="/nonexistent")
    for _ in range(1000): mon.sample_once()
    assert len(mon.samples) <= 20 and mon._thin > 1 and len(mon.timeline(8)) == 8 and len(mon.timeline(500)) <= 20


def test_a_failing_sampler_never_breaks_the_job():
    def boom(): raise RuntimeError("proc vanished")
    mon = ResourceMonitor(snapshot=boom, interval=0.05).start(); time.sleep(0.2); s = mon.stop()
    assert isinstance(s, dict)


# ---------------------------------------------------------------- real processes (any OS)
def test_real_child_memory_is_measured():
    code = "import time; b = bytearray(300 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096]); print('READY', flush=True); time.sleep(30)"
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "READY"
        mon = ResourceMonitor(worker_pid=lambda: p.pid, interval=0.1); mon.mark("alloc"); mon.sample_once(); time.sleep(0.3); mon.sample_once()
        s = mon.stop()
        assert s["stages"][0]["worker_rss_max_mb"] > 250 and s["peak"]["parent_rss_mb"] is not None                                   # a 300 MB allocation is visible in the worker, not hidden in the parent
        assert s["peak"]["tree_rss_mb"] >= s["peak"]["worker_rss_mb"]
    finally:
        p.kill(); p.wait()


def test_real_cpu_seconds_are_measured():
    p = subprocess.Popen([sys.executable, "-c", "import time\nt=time.time()\nwhile time.time()-t < 3: sum(range(10000))"])
    try:
        mon = ResourceMonitor(worker_pid=lambda: p.pid, interval=0.1); mon.mark("burn"); mon.sample_once(); time.sleep(1.5); mon.sample_once()
        assert mon.stop()["stages"][0]["cpu_s"] > 0.5
    finally:
        p.kill(); p.wait()


def test_temp_disk_usage_is_reported(tmp_path):
    (tmp_path / "f").write_bytes(b"x" * (3 * MiB))
    used = lambda: (sum(f.stat().st_size for f in tmp_path.iterdir()) / MiB, 1.0)
    mon = ResourceMonitor(disk=used, snapshot=lambda: {}, cgroup_root="/nonexistent"); mon.mark("d"); mon.sample_once(); st = mon.stop()["stages"][0]
    assert st["temp_disk_max_mb"] == pytest.approx(3.0)


def test_every_stage_gets_a_start_and_an_end_sample_even_when_shorter_than_the_interval():
    """With a 1 s interval, a 0.2 s stage used to get no sample at all (and a CPU delta of 0 or None)."""
    clk = FakeClock(); n = {"i": 0}

    def snap():
        n["i"] += 1
        return {1: dict(ppid=0, rss_mb=100.0, cpu_s=0.0), 2: dict(ppid=1, rss_mb=200.0 + n["i"], cpu_s=n["i"] * 0.5)}

    mon = ResourceMonitor(worker_pid=lambda: 2, parent_pid=1, interval=1.0, snapshot=snap, cgroup_root="/nonexistent", clock=clk)             # no background thread started
    for name in ("short_a", "short_b"):
        mon.mark(name); clk.t += 0.2
    s = mon.stop()
    for st in s["stages"][:2]:
        assert st["samples"] >= 2 and st["worker_rss_max_mb"] is not None and st["cpu_s"] is not None and st["cpu_s"] > 0, st


# ---------------------------------------------------------------- F8.7: Linux container layouts and CPU / throttling summary
def test_cgroup_v2_nested_path_is_followed_when_the_mount_holds_the_host_hierarchy(tmp_path):
    root, proc = tmp_path / "cgroup", tmp_path / "proc"
    write(proc / "self" / "cgroup", "0::/kubepods/pod123/ctr456\n")
    d = root / "kubepods" / "pod123" / "ctr456"
    write(d / "memory.current", f"{700 * MiB}\n"); write(d / "memory.max", f"{2700 * MiB}\n"); write(d / "cpu.stat", "usage_usec 1000000\nnr_periods 10\nnr_throttled 1\nthrottled_usec 5000\n")
    write(root / "cpu.stat", "usage_usec 99\n")                                                                                   # the host's own file must NOT be mistaken for ours
    c = M.read_cgroup(str(root), str(proc))
    assert c["version"] == 2 and c["memory_current_mb"] == 700 and c["memory_max_mb"] == 2700 and c["nr_throttled"] == 1


def test_cgroup_direct_root_wins_and_dotdot_or_garbage_paths_are_ignored(tmp_path):
    root, proc = tmp_path / "cgroup", tmp_path / "proc"; write(root / "memory.current", f"{10 * MiB}\n"); write(proc / "self" / "cgroup", "0::/../../etc\n")
    assert M.read_cgroup(str(root), str(proc))["memory_current_mb"] == 10                                                          # own namespace root has the files: used as-is
    root2 = tmp_path / "cg2"; write(root2 / "x" / "memory.current", "1\n"); write(proc / "self" / "cgroup", "0::/../x\n")
    assert M.read_cgroup(str(root2), str(proc))["version"] is None                                                                  # '..' is never followed
    for junk in ("", "garbage", "1:cpu:/a\n", "0::\n", "0::/missing\n"):
        write(proc / "self" / "cgroup", junk); assert M.read_cgroup(str(root2), str(proc))["version"] is None


def _cg(root, usage, periods, thr, thr_us, cur=500):
    write(root / "memory.current", f"{cur * MiB}\n"); write(root / "memory.max", f"{2000 * MiB}\n"); write(root / "cpu.max", "200000 100000\n")
    write(root / "cpu.stat", f"usage_usec {int(usage * 1e6)}\nnr_periods {periods}\nnr_throttled {thr}\nthrottled_usec {int(thr_us)}\n")


def test_cpu_usage_and_throttling_are_reported_as_deltas_over_the_monitored_period(tmp_path):
    clk = FakeClock(); cg = tmp_path / "cg"; _cg(cg, 100.0, 1000, 40, 8_000_000)                                                    # counters are cumulative: a non-zero start must be subtracted
    mon = ResourceMonitor(worker_pid=lambda: None, parent_pid=os.getpid(), proc_root=str(tmp_path / "noproc"), cgroup_root=str(cg), clock=clk, snapshot=lambda: {}, sample_on_mark=False)
    mon.mark("detect_left"); clk.t += 10; _cg(cg, 115.0, 1100, 70, 9_500_000); mon.sample_once()
    s = mon.stop(); c = s["cpu"]
    assert c["cgroup_cpu_s"] == 15.0 and c["cgroup_avg_cores"] == 1.5 and c["cpu_limit_cores"] == 2.0
    assert (c["periods"], c["throttled_periods"], c["throttled_s"], c["throttled_fraction"]) == (100, 30, 1.5, 0.3)
    assert s["cgroup"]["nr_throttled_total"] == 30 and c["throttling_measurable"] and c["cgroup_cpu_measurable"] and s["cgroup"]["memory_max_mb"] == 2000


def test_worker_tree_cpu_comes_from_the_process_table(tmp_path):
    clk = FakeClock(); n = {"i": 0}

    def snap():
        n["i"] += 1
        return {1: dict(ppid=0, rss_mb=10.0, cpu_s=0.0), 2: dict(ppid=1, rss_mb=50.0, cpu_s=n["i"] * 2.0), 3: dict(ppid=2, rss_mb=5.0, cpu_s=n["i"] * 1.0)}
    mon = ResourceMonitor(worker_pid=lambda: 2, parent_pid=1, proc_root=str(tmp_path / "noproc"), cgroup_root=str(tmp_path / "nocg"), clock=clk, snapshot=snap, sample_on_mark=False)
    mon.mark("detect_left")
    for _ in range(4): clk.t += 1; mon.sample_once()
    c = mon.stop()["cpu"]
    assert c["worker_tree_cpu_s"] > 0 and c["worker_tree_avg_cores"] > 0                                                              # child (pid 3) is counted with the worker (pid 2)


def test_nothing_measurable_gives_none_everywhere_and_never_raises(tmp_path):
    """A platform with no /proc, no cgroup and no disk callback (e.g. a locked-down sandbox): every metric is None / False, nothing is guessed."""
    mon = ResourceMonitor(worker_pid=lambda: 4242, parent_pid=999_999, proc_root=str(tmp_path / "x"), cgroup_root=str(tmp_path / "y"), snapshot=lambda: {})
    mon.mark("detect_left"); s = mon.stop()
    assert s["available"] == dict(pss=False, cgroup=False) and not s["cpu"]["cgroup_cpu_measurable"] and not s["cpu"]["throttling_measurable"]
    assert s["cpu"]["cgroup_cpu_s"] is None and s["cpu"]["throttled_periods"] is None and s["cpu"]["throttled_fraction"] is None and s["cpu"]["worker_tree_cpu_s"] is None
    assert s["peak"]["cgroup_memory_mb"] is None and s["peak"]["worker_rss_mb"] is None and s["cgroup"]["memory_max_mb"] is None and s["cgroup"]["nr_throttled_total"] is None


def test_unreadable_or_corrupt_proc_and_cgroup_files_do_not_break_sampling(tmp_path):
    proc, cg = tmp_path / "proc", tmp_path / "cg"; fake_proc(proc, {5: (1, 20.0, 10, None)})                                           # smaps_rollup missing (older kernel / hidepid)
    write(proc / "6" / "stat", "6 (broken"); write(proc / "7" / "stat", ""); write(proc / "7" / "statm", "x y z")
    write(cg / "memory.current", "not a number\n"); write(cg / "cpu.stat", "usage_usec\nnr_throttled abc\n"); write(cg / "memory.max", "max\n")
    mon = ResourceMonitor(worker_pid=lambda: 5, parent_pid=5, proc_root=str(proc), cgroup_root=str(cg)); mon.mark("a"); s = mon.stop()
    assert s["stages"][0]["worker_rss_max_mb"] == pytest.approx(20.0, abs=0.1) and s["peak"]["tree_pss_mb"] is None and s["peak"]["cgroup_memory_mb"] is None


def test_mount_info_flags_a_memory_backed_tmp_so_the_report_can_say_disk_equals_ram(tmp_path):
    base = os.path.realpath(tmp_path); mounts = tmp_path / "mounts"
    mounts.write_text(f"overlay {base} overlay rw 0 0\ntmpfs {base}/ram tmpfs rw,nosuid 0 0\n")
    assert M.mount_info(f"{base}/ram/job", str(mounts))["memory_backed"] is True and M.mount_info(f"{base}/disk/job", str(mounts))["memory_backed"] is False

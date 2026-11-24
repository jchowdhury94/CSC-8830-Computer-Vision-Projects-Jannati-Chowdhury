"""F8.31C-1C: container-aware memory budget, shared worker coordination, calibration-runner admission.  Synthetic cgroup trees and fake workers only; nothing real is read."""
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import resource_budget as RB
from calibration_runner import CalibrationJobRunner, RunnerLimits
from chunk_store import ChunkStore, Limits, MiB
from test_runner import A, OK_RESULT, SETTINGS, alive, assert_retained, fake, files, put_pair, run, syn, wait_for       # noqa: F401  (fixtures and helpers)

NO_CONTAINER = dict(limit_mb=None, reliable=False, reason="test")


def table(parent_mb=200.0, pid=None, extra=None):
    t = {pid or os.getpid(): dict(ppid=1, rss_mb=parent_mb, cpu_s=0.0)}
    t.update(extra or {})
    return lambda: dict(t)


def coord(tmp_path, *, total=4096.0, parent=200.0, name="rb", reserve_floor=256.0, **kw):
    return RB.ResourceCoordinator(root=tmp_path / name, cfg=RB.BudgetConfig(ceiling_mb=total, reserve_floor_mb=reserve_floor), snapshot=table(parent), container=NO_CONTAINER, **kw)


def w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(str(text))


# ================================================================ container limit discovery (cgroup v2 / v1, missing, unlimited, garbage)
def v2(root, limit="1073741824", cur="104857600", anon="52428800"):
    w(root / "memory.current", cur); w(root / "memory.max", limit); w(root / "memory.stat", f"anon {anon}\nfile 1000\n")


def test_cgroup_v2_limit_is_discovered(tmp_path):
    v2(tmp_path / "cg"); d = RB.container_memory_limit(str(tmp_path / "cg"), str(tmp_path / "proc"))
    assert d["limit_mb"] == 1024.0 and d["version"] == 2 and d["reliable"] and d["source"] == "cgroup_v2"


def test_cgroup_v2_max_means_unlimited_and_is_not_a_limit(tmp_path):
    v2(tmp_path / "cg", limit="max"); d = RB.container_memory_limit(str(tmp_path / "cg"), str(tmp_path / "proc"))
    assert d["limit_mb"] is None and not d["reliable"] and "unlimited" in d["reason"]


def test_cgroup_v2_the_smallest_ancestor_limit_applies(tmp_path):
    root = tmp_path / "cg"; child = root / "app" / "worker"
    w(root / "app" / "memory.max", str(512 * MiB)); v2(child, limit=str(2048 * MiB))                      # like a real cgroup mount, the root itself has no memory files
    w(tmp_path / "proc" / "self" / "cgroup", "0::/app/worker\n")
    d = RB.container_memory_limit(str(root), str(tmp_path / "proc"))
    assert d["limit_mb"] == 512.0, d                                                  # the parent's 512 MB binds the child's 2048 MB


def test_cgroup_v1_limit_unlimited_sentinel_and_hierarchical(tmp_path):
    def v1(root, limit, hier=None):
        m = root / "memory"; w(m / "memory.usage_in_bytes", "1000"); w(m / "memory.max_usage_in_bytes", "2000"); w(m / "memory.limit_in_bytes", limit)
        w(m / "memory.stat", "rss 10\ncache 20\n" + (f"hierarchical_memory_limit {hier}\n" if hier else ""))
    v1(tmp_path / "a", str(768 * MiB)); d = RB.container_memory_limit(str(tmp_path / "a"), str(tmp_path / "p"))
    assert d["limit_mb"] == 768.0 and d["version"] == 1
    v1(tmp_path / "b", "9223372036854771712"); d = RB.container_memory_limit(str(tmp_path / "b"), str(tmp_path / "p"))
    assert d["limit_mb"] is None and not d["reliable"]                                  # the kernel's 'unlimited' sentinel
    v1(tmp_path / "c", "9223372036854771712", hier=str(300 * MiB)); d = RB.container_memory_limit(str(tmp_path / "c"), str(tmp_path / "p"))
    assert d["limit_mb"] == 300.0                                                       # the hierarchical limit is the effective one


@pytest.mark.parametrize("junk", ["", "abc", "0", "-5", "nan"])
def test_missing_or_garbage_limits_are_unknown_never_guessed(tmp_path, junk):
    v2(tmp_path / "cg", limit=junk); d = RB.container_memory_limit(str(tmp_path / "cg"), str(tmp_path / "proc"))
    assert d["limit_mb"] is None and not d["reliable"]
    d = RB.container_memory_limit(str(tmp_path / "nothing-here"), str(tmp_path / "proc"))
    assert d["limit_mb"] is None and d["version"] is None and "no cgroup" in d["reason"]


# ================================================================ budget computation: fallback, configured ceiling, never above the ceiling
def test_conservative_fallback_is_documented_and_reported():
    b = RB.compute_budget(RB.BudgetConfig(), NO_CONTAINER)
    assert b["source"] == "conservative_fallback" and b["total_mb"] == RB.FALLBACK_TOTAL_MB == 2048.0 and b["limitation"] and "could not be determined" in b["limitation"]
    assert b["usable_mb"] == pytest.approx(b["total_mb"] - b["reserve_mb"]) and b["reserve_mb"] >= RB.RESERVE_FLOOR_MB


def test_container_limit_is_used_and_a_lower_configured_ceiling_wins():
    c = dict(limit_mb=1000.0, reliable=True, source="cgroup_v2", reason="x")
    b = RB.compute_budget(RB.BudgetConfig(), c); assert b["total_mb"] == 1000.0 and b["source"] == "cgroup_v2" and b["limitation"] is None
    b = RB.compute_budget(RB.BudgetConfig(ceiling_mb=600.0), c); assert b["total_mb"] == 600.0 and "below the container limit" in b["source"]
    b = RB.compute_budget(RB.BudgetConfig(ceiling_mb=9000.0), c); assert b["total_mb"] == 1000.0                 # a configured ceiling never raises a real limit
    b = RB.compute_budget(RB.BudgetConfig(ceiling_mb=5000.0), NO_CONTAINER); assert b["total_mb"] == 5000.0 and "operator-asserted" in b["source"] and b["limitation"]


def test_configured_reserve_is_honoured_and_env_values_are_validated():
    assert RB.BudgetConfig.from_env({}).ceiling_mb is None
    c = RB.BudgetConfig.from_env({"F8_APP_MEMORY_CEILING_MB": "3000", "F8_MEMORY_RESERVE_MB": "500"}); assert c.ceiling_mb == 3000.0 and c.reserve_floor_mb == 500.0
    c = RB.BudgetConfig.from_env({"F8_APP_MEMORY_CEILING_MB": "10", "F8_MEMORY_RESERVE_MB": "nan"}); assert c.ceiling_mb is None and c.reserve_floor_mb == RB.RESERVE_FLOOR_MB   # invalid: ignored, not clamped
    assert RB.compute_budget(RB.BudgetConfig(ceiling_mb=2000.0, reserve_floor_mb=500.0), NO_CONTAINER)["usable_mb"] == 1500.0


# ================================================================ accounting: parent, workers, reserve
def test_parent_memory_and_safety_reserve_are_subtracted(tmp_path):
    c = coord(tmp_path, total=4096.0, parent=600.0); st = c.state()
    assert st["parent_rss_mb"] == 600.0 and st["parent_rss_measured"] and st["budget"]["reserve_mb"] == 409.6          # 10% beats the 256 MB floor
    assert st["available_mb"] == pytest.approx(4096 - 409.6 - 600, abs=0.2)


def test_unreadable_parent_rss_assumes_a_cautious_value(tmp_path):
    c = RB.ResourceCoordinator(root=tmp_path / "rb", cfg=RB.BudgetConfig(ceiling_mb=4096.0), snapshot=lambda: {}, container=NO_CONTAINER); st = c.state()
    assert not st["parent_rss_measured"] and st["parent_rss_mb"] == RB.ASSUMED_PARENT_MB


def test_grant_is_capped_by_what_is_available_and_by_the_request(tmp_path):
    c = coord(tmp_path, total=4096.0, parent=200.0)
    a = c.reserve("reconstruction", "o", 100000.0, 512.0); assert a.ok and a.granted_mb == pytest.approx(4096 - 409.6 - 200, abs=0.2) and a.granted_mb <= 4096.0   # never above the ceiling
    a.reservation.release()
    a = c.reserve("reconstruction", "o", 1000.0, 512.0); assert a.ok and a.granted_mb == 1000.0
    a.reservation.release()


def test_other_processes_parent_memory_is_counted(tmp_path):
    other = os.getpid() + 7777; c = RB.ResourceCoordinator(root=tmp_path / "rb", cfg=RB.BudgetConfig(ceiling_mb=4096.0), container=NO_CONTAINER,
                                                           snapshot=table(200.0, extra={other: dict(ppid=1, rss_mb=300.0, cpu_s=0)}))
    a = c.reserve("calibration", "o", 500.0, 100.0)
    assert a.ok
    # simulate a second app process holding a reservation: write a record owned by another pid, lock it from a different descriptor
    import fcntl
    p = c.root / "calibration-x.res"; fh = open(p, "w+b"); fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    fh.write(b'{"kind": "reconstruction", "granted_mb": 100.0, "pid": %d, "owner": "z"}' % other); fh.flush()
    try:
        st = c.state(); assert st["other_process_parents_mb"] == 300.0 and any(r["pid"] == other for r in st["reservations"])
    finally:
        fh.close(); p.unlink()
    a.reservation.release()


def test_live_container_usage_can_only_lower_the_grant(tmp_path):
    root = tmp_path / "cg"; v2(root, limit=str(2048 * MiB), anon=str(1500 * MiB), cur=str(1600 * MiB))
    c = RB.ResourceCoordinator(root=tmp_path / "rb", cfg=RB.BudgetConfig(), cgroup_root=str(root), proc_root=str(tmp_path / "proc"), snapshot=table(100.0))
    st = c.state()                                                                      # accounting says ~1700 free; the container itself reports 1500 MB already used
    assert st["budget"]["source"] == "cgroup_v2" and st["live_container_used_mb"] == 1500.0 and st["available_mb"] == pytest.approx(2048 - 256 - 1500, abs=1) and st["available_mb"] < st["accounting_available_mb"]


# ================================================================ one reconstruction, one calibration; BUSY vs INSUFFICIENT; shared budget
def test_one_job_per_kind_and_distinct_refusals(tmp_path):
    c = coord(tmp_path, total=4096.0)
    r1 = c.reserve("reconstruction", "A", 1000.0, 500.0); assert r1.ok
    busy = c.reserve("reconstruction", "B", 1000.0, 500.0); assert busy.outcome == RB.BUSY and busy.reservation is None and busy.granted_mb is None
    r2 = c.reserve("calibration", "B", 1000.0, 500.0); assert r2.ok                                  # a different kind may run if the SHARED budget has room
    big = RB.ResourceCoordinator(root=tmp_path / "rb", cfg=RB.BudgetConfig(ceiling_mb=4096.0), snapshot=table(200.0), container=NO_CONTAINER)     # same directory = same shared state
    assert big.reserve("calibration", "C", 1.0, 1.0).outcome == RB.BUSY                              # the second coordinator object sees the first one's reservations
    r1.reservation.release(); r2.reservation.release()
    assert c.state()["reservations"] == []


def test_calibration_and_reconstruction_cannot_both_reserve_the_same_memory(tmp_path):
    c = coord(tmp_path, total=3000.0, parent=200.0)                                                  # usable 3000-300-200 = 2500 for workers
    first = c.reserve("reconstruction", "A", 100000.0, 500.0); assert first.ok and first.granted_mb == pytest.approx(2500, abs=1)
    second = c.reserve("calibration", "B", 1024.0, 1024.0)
    assert second.outcome == RB.INSUFFICIENT_MEMORY and "needs at least 1024" in second.detail["reason"]      # nothing left, and it is NOT reported as busy
    first.reservation.release()
    third = c.reserve("calibration", "B", 1024.0, 1024.0); assert third.ok
    third.reservation.release()
    both = c.reserve("reconstruction", "A", 1200.0, 500.0); both2 = c.reserve("calibration", "B", 1200.0, 500.0)
    assert both.ok and both2.ok and both.granted_mb + both2.granted_mb <= 3000 - 300 - 200 + 0.5      # together they never exceed what the budget allows
    both.reservation.release(); both2.reservation.release()


def test_sum_of_grants_never_exceeds_usable_whatever_the_sequence(tmp_path):
    import random
    rnd = random.Random(1); c = coord(tmp_path, total=3000.0, parent=250.0); held = []
    for _ in range(300):
        if held and rnd.random() < 0.45:
            held.pop(rnd.randrange(len(held))).release()
        else:
            a = c.reserve(rnd.choice(RB.KINDS), "o", rnd.uniform(100, 5000), rnd.uniform(1, 800))
            if a.ok:
                held.append(a.reservation)
        st = c.state(); assert st["reserved_mb"] <= st["budget"]["usable_mb"] + 0.5 and st["budget"]["usable_mb"] <= st["budget"]["total_mb"] <= 3000.0
    for r in held:
        r.release()
    assert c.state()["reservations"] == []


def test_refusals_return_immediately_and_nothing_is_queued(tmp_path):
    c = coord(tmp_path); held = c.reserve("reconstruction", "A", 1000.0, 500.0); assert held.ok
    t = time.monotonic(); outs = [c.reserve("reconstruction", f"v{i}", 1000.0, 500.0).outcome for i in range(50)]
    assert time.monotonic() - t < 5 and set(outs) == {RB.BUSY}                                       # 50 further visitors: 50 immediate refusals, no waiting list
    assert len(c.state()["reservations"]) == 1
    held.reservation.release()


def test_reservation_is_released_by_an_exception_a_with_block_and_a_double_release(tmp_path):
    c = coord(tmp_path)
    with pytest.raises(RuntimeError):
        with c.reserve("calibration", "o", 800.0, 400.0).reservation:
            assert len(c.state()["reservations"]) == 1; raise RuntimeError("worker supervisor blew up")
    assert c.state()["reservations"] == [] and list(c.root.glob("*.res")) == []
    r = c.reserve("calibration", "o", 800.0, 400.0).reservation; r.release(); r.release()             # idempotent
    a = c.reserve("calibration", "o", 800.0, 400.0); a.reservation = None; del a                         # a leaked, garbage-collected reservation releases itself
    import gc; gc.collect(); assert c.state()["reservations"] == []


def test_state_never_exposes_an_owner_and_unknown_kinds_are_refused(tmp_path):
    c = coord(tmp_path); a = c.reserve("calibration", "secret-owner-token-123", 800.0, 400.0)
    assert "secret-owner-token-123" not in repr(c.state()) and "secret-owner-token-123" not in (c.root / next(n for n in os.listdir(c.root) if n.endswith(".res"))).read_text()
    with pytest.raises(ValueError):
        c.reserve("bogus", "o", 1.0, 1.0)
    a.reservation.release()


def test_threads_cannot_both_take_the_single_slot(tmp_path):
    c = coord(tmp_path); outs, gate = [], threading.Barrier(8)

    def go(i):
        gate.wait(); a = c.reserve("reconstruction", f"v{i}", 500.0, 100.0); outs.append(a)
    ts = [threading.Thread(target=go, args=(i,)) for i in range(8)]; [t.start() for t in ts]; [t.join() for t in ts]
    assert sum(a.ok for a in outs) == 1 and sum(a.outcome == RB.BUSY for a in outs) == 7
    [a.reservation.release() for a in outs if a.ok]


# ================================================================ a crashed / killed holder never leaves a permanent reservation (real second process)
CHILD = textwrap.dedent("""
    import os, sys, time
    sys.path.insert(0, sys.argv[1])
    import resource_budget as RB
    c = RB.ResourceCoordinator(root=sys.argv[2], cfg=RB.BudgetConfig(ceiling_mb=4096.0), container=dict(limit_mb=None, reliable=False, reason="t"))
    a = c.reserve("reconstruction", "child", 1000.0, 100.0)
    open(sys.argv[3], "w").write(a.outcome); time.sleep(300)
""")


def test_a_killed_process_leaves_no_permanent_reservation_and_a_live_one_blocks_others(tmp_path):
    root, flag = tmp_path / "shared", tmp_path / "flag"
    p = subprocess.Popen([sys.executable, "-c", CHILD, str(Path(RB.__file__).parent), str(root), str(flag)])
    try:
        assert wait_for(flag.exists, 20) and flag.read_text() == RB.GRANTED
        mine = RB.ResourceCoordinator(root=root, cfg=RB.BudgetConfig(ceiling_mb=4096.0), snapshot=table(200.0), container=NO_CONTAINER)
        assert mine.reserve("reconstruction", "me", 500.0, 100.0).outcome == RB.BUSY                  # cross-PROCESS: the child's reservation is seen
        assert len(list(root.glob("*.res"))) == 1
        p.kill(); p.wait(10)
        a = mine.reserve("reconstruction", "me", 500.0, 100.0)                                       # the kernel dropped the dead holder's lock
        assert a.ok and len(list(root.glob("*.res"))) == 1                                           # the orphan file was removed, ours is the only one
        a.reservation.release()
    finally:
        if p.poll() is None:
            p.kill()


def test_an_unsafe_coordination_directory_is_not_used(tmp_path):
    target = tmp_path / "real"; target.mkdir(); link = tmp_path / "link"; link.symlink_to(target)
    c = RB.ResourceCoordinator(root=link, cfg=RB.BudgetConfig(ceiling_mb=4096.0), snapshot=table(), container=NO_CONTAINER)
    assert c.root is None and not c.cross_process                                                    # a planted symlink is refused; in-process coordination still works and says so
    a = c.reserve("calibration", "o", 500.0, 100.0); assert a.ok and c.reserve("calibration", "o", 500.0, 100.0).outcome == RB.BUSY; a.reservation.release()
    assert list(target.iterdir()) == []


# ================================================================ the calibration runner asks the shared budget first
def test_runner_refuses_busy_and_insufficient_memory_distinctly_and_touches_no_upload(tmp_path, syn):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB), root=tmp_path / "store")
    try:
        put_pair(s, syn["left"], syn["right"])
        c = coord(tmp_path, total=4096.0)
        held = c.reserve("calibration", "other-visitor", 500.0, 100.0)
        r = run(s, coordinator=c, worker_cmd=fake(OK_RESULT))
        assert r.result["outcome"] == "rejected" and "Another calibration is running" in r.result["error"] and r.result["resource_budget"]["outcome"] == RB.BUSY
        assert len(files(s)) == 2 and s.job_status(A)["lease"] is None                                # videos untouched, no lease taken
        held.reservation.release()
        tiny = coord(tmp_path, total=900.0, name="rb2")                                              # usable is below the 1024 MB minimum
        r = run(s, coordinator=tiny, worker_cmd=fake(OK_RESULT))
        assert r.result["outcome"] == "rejected" and "not have enough free memory" in r.result["error"] and r.result["resource_budget"]["outcome"] == RB.INSUFFICIENT_MEMORY
        assert r.result["resource_budget"]["source"] == "configured_ceiling (operator-asserted; the container limit could not be read: test)" and len(files(s)) == 2
    finally:
        s.close()


def test_runner_reserves_while_running_releases_after_success_crash_timeout_and_guard_kill(tmp_path, syn):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB, lease_heartbeat_timeout_s=60), root=tmp_path / "store")
    try:
        c = coord(tmp_path, total=8192.0)
        put_pair(s, syn["left"], syn["right"]); seen = []
        r = CalibrationJobRunner(s, A, SETTINGS, RunnerLimits(allow_full=True, heartbeat_s=0.5), worker_cmd=fake("time.sleep(1.5); " + OK_RESULT), monitor_interval=0.25, coordinator=c); r.start()
        assert wait_for(lambda: c.state()["reservations"], 10); seen = c.state()["reservations"]; r.join(60)
        assert seen[0]["kind"] == "calibration" and r.result["outcome"] == "completed" and c.state()["reservations"] == []
        assert r.result["resource_budget"]["granted_mb"] == seen[0]["granted_mb"] <= 2048.0 and r.result["memory_guard"]["process_tree"]["limit_mb"] == seen[0]["granted_mb"]
        for body, kw, want in (("os._exit(9)", {}, "worker_crashed"), ("time.sleep(120)", dict(timeout_s=2.0, kill_grace_s=1.0), "timeout")):
            put_pair(s, syn["left"], syn["right"])
            r = run(s, limits=RunnerLimits(allow_full=True, heartbeat_s=0.5, **kw), worker_cmd=fake(body), coordinator=c)
            assert r.result["outcome"] == want and c.state()["reservations"] == []
        put_pair(s, syn["left"], syn["right"])                                                       # a cancelled run (the supervisor ends it) also releases
        r = CalibrationJobRunner(s, A, SETTINGS, RunnerLimits(allow_full=True, heartbeat_s=0.5, kill_grace_s=1.0), worker_cmd=fake("time.sleep(120)"), monitor_interval=0.25, coordinator=c); r.start()
        assert wait_for(lambda: c.state()["reservations"], 10); r.cancel(); r.join(60)
        assert r.result["outcome"] == "cancelled" and c.state()["reservations"] == []
    finally:
        s.close()


def test_runner_never_leaves_a_reservation_when_the_supervisor_itself_fails(tmp_path, syn, monkeypatch):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB), root=tmp_path / "store")
    try:
        c = coord(tmp_path, total=8192.0); put_pair(s, syn["left"], syn["right"])
        monkeypatch.setattr("calibration_runner.validate_videos", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("supervisor bug")))
        r = run(s, coordinator=c, worker_cmd=fake(OK_RESULT))
        assert r.result["outcome"] == "worker_crashed" and c.state()["reservations"] == [] and r._reservation is None
    finally:
        s.close()


def test_a_broken_coordinator_is_a_refusal_never_an_unbudgeted_run(tmp_path, syn):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB), root=tmp_path / "store")
    try:
        class Broken:
            def reserve(self, *a, **k): raise OSError("disk gone")
        put_pair(s, syn["left"], syn["right"]); r = run(s, coordinator=Broken(), worker_cmd=fake(OK_RESULT))
        assert r.result["outcome"] == "rejected" and "could not be consulted" in r.result["error"] and len(files(s)) == 2
    finally:
        s.close()


def test_the_granted_ceiling_is_what_the_runner_enforces(tmp_path, syn):
    """The tree limit is the grant: a tight budget (not RunnerLimits.max_tree_rss_mb) stops a worker that exceeds it, and the parent memory is part of what the grant left out."""
    s = ChunkStore(Limits(max_file_bytes=64 * MiB), root=tmp_path / "store")
    try:
        from monitor import snapshot_processes
        parent = snapshot_processes()[os.getpid()]["rss_mb"]
        total = parent + 256 + 1100 + 150                                                           # leaves a grant of roughly 1100 MB beyond the parent; floor is 1024
        c = RB.ResourceCoordinator(root=tmp_path / "rb", cfg=RB.BudgetConfig(ceiling_mb=total, reserve_floor_mb=256.0, reserve_fraction=0.0), container=NO_CONTAINER)
        put_pair(s, syn["left"], syn["right"])
        body = "b = bytearray(1600 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096]); time.sleep(120)"
        r = run(s, limits=RunnerLimits(allow_full=True, heartbeat_s=0.5, kill_grace_s=1.0), worker_cmd=fake(body), timeout=90, coordinator=c)
        assert r.result["outcome"] == "memory_limit" and "memory guard" in r.result["error"] and c.state()["reservations"] == []
        assert r.result["memory_guard"]["process_tree"]["limit_mb"] == r.result["resource_budget"]["granted_mb"] < 1400
    finally:
        s.close()

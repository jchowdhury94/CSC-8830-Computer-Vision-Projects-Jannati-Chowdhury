"""Phase D.3: container-wide admission for the frame-extraction and upload-inspection workers (the 'light' kinds), and their exclusion against calibration / reconstruction.
Synthetic budgets, fake holders and real second processes only; no video is decoded and no benchmark runs."""
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import resource_budget as RB
from test_f8_31c1c_resource_budget import NO_CONTAINER, coord, table                       # noqa: F401  (helpers)
from test_runner import wait_for                                                         # noqa: F401


def light(c, kind="frames", wait_s=0.0):
    return RB.acquire_light(kind, "visitor", c, wait_s=wait_s)


# ================================================================ the new kinds and the two-way exclusion
def test_light_kinds_are_known_and_a_light_job_is_admitted_on_an_idle_server(tmp_path):
    c = coord(tmp_path)
    assert set(RB.LIGHT_KINDS) == {"frames", "validation"} and set(RB.KINDS) == {"calibration", "reconstruction", "frames", "validation"}
    for kind in RB.LIGHT_KINDS:
        a = c.reserve(kind, "o", *RB.LIGHT_CLAIMS_MB[kind]); assert a.ok and a.granted_mb >= RB.LIGHT_CLAIMS_MB[kind][1]; a.reservation.release()
    with pytest.raises(ValueError):
        RB.acquire_light("reconstruction", "o", c)                                             # only light kinds may use the helper


@pytest.mark.parametrize("heavy", RB.HEAVY_KINDS)
def test_a_running_heavy_job_refuses_every_light_job_and_vice_versa(tmp_path, heavy):
    c = coord(tmp_path)
    h = c.reserve(heavy, "o", 1000.0, 100.0); assert h.ok
    for kind in RB.LIGHT_KINDS:
        a = c.reserve(kind, "o", *RB.LIGHT_CLAIMS_MB[kind])
        assert a.outcome == RB.BUSY and a.detail["busy_with"] == "heavy" and "calibration or measurement" in a.detail["reason"]
        with pytest.raises(RB.ResourceBusy) as e:
            light(c, kind)
        assert e.value.busy_with == "heavy" and "try again in a minute" in e.value.message and e.value.kind == kind
    h.reservation.release()
    l = light(c)                                                                                # now the other direction: a light job blocks a heavy one
    for kind in RB.HEAVY_KINDS:
        a = c.reserve(kind, "o", 1000.0, 100.0)
        assert a.outcome == RB.BUSY and a.detail["busy_with"] == "light" and "frame-extraction or video-check" in a.detail["reason"]
    l.release()
    again = c.reserve(heavy, "o", 1000.0, 100.0); assert again.ok; again.reservation.release()   # and everything is admitted again once the light job is gone


def test_a_holder_that_cannot_be_read_counts_as_heavy_and_blocks_light_jobs(tmp_path):
    c = coord(tmp_path)
    h = c.reserve("reconstruction", "o", 1000.0, 100.0); assert h.ok
    fh = h.reservation._fh; fh.seek(0); fh.truncate(); fh.write(b"not json"); fh.flush()       # a live holder whose record is unreadable
    a = c.reserve("frames", "o", *RB.LIGHT_CLAIMS_MB["frames"])
    assert a.outcome == RB.BUSY and a.detail["busy_with"] == "heavy"                           # fail closed
    h.reservation.release()


def test_heavy_kinds_keep_their_existing_rules(tmp_path):
    c = coord(tmp_path)
    a = c.reserve("calibration", "o", 500.0, 100.0); b = c.reserve("calibration", "o", 500.0, 100.0)
    assert a.ok and b.outcome == RB.BUSY and b.detail["busy_with"] == "same_kind"              # one calibration at a time, as before
    r = c.reserve("reconstruction", "o", 500.0, 100.0); assert r.ok                            # a calibration and a reconstruction still share the budget
    a.reservation.release(); r.reservation.release()


# ================================================================ light jobs: bounded in number, not serialised needlessly
def test_two_light_jobs_run_together_and_a_third_is_refused_until_one_finishes(tmp_path):
    c = coord(tmp_path)
    a, b = light(c, "frames"), light(c, "validation")                                           # frames and validation share one limit
    with pytest.raises(RB.ResourceBusy) as e:
        light(c, "frames")
    assert e.value.busy_with == "light_limit" and "limit 2" in e.value.reason
    a.release(); third = light(c, "frames"); third.release(); b.release()


def test_the_light_job_limit_is_configurable_and_invalid_values_are_ignored():
    assert RB.BudgetConfig.from_env({"F8_MAX_LIGHT_JOBS": "3"}).max_light_jobs == 3
    for bad in ("0", "9", "-1", "x", "2.5", ""):
        assert RB.BudgetConfig.from_env({"F8_MAX_LIGHT_JOBS": bad}).max_light_jobs == RB.MAX_LIGHT_DEFAULT
    assert RB.BudgetConfig.from_env({}).max_light_jobs == 2


def test_a_server_with_too_little_memory_refuses_a_light_job_with_the_budget_reason(tmp_path):
    c = coord(tmp_path, total=900.0, reserve_floor=200.0, parent=300.0)                         # usable 700 - parent 300 = 400 MB < the 512 MB minimum of a frame job
    with pytest.raises(RB.ResourceBusy) as e:
        light(c, "frames")
    assert e.value.outcome == RB.INSUFFICIENT_MEMORY and "needs at least 512 MB" in e.value.reason


def test_an_unreadable_container_limit_is_explained_in_the_refusal(tmp_path):
    c = RB.ResourceCoordinator(root=tmp_path / "rb", cfg=RB.BudgetConfig(), snapshot=table(1500.0), container=NO_CONTAINER)      # no cgroup, no operator ceiling: the 2048 MB fallback
    a = c.reserve("reconstruction", "o", 3072.0, 2450.0)
    assert a.outcome == RB.INSUFFICIENT_MEMORY and "conservative fallback" in a.detail["reason"] and "could not be determined" in a.detail["reason"]


# ================================================================ concurrency: the admission lock closes every race
def test_threads_cannot_exceed_the_light_job_limit(tmp_path):
    c = coord(tmp_path); got, refused, gate = [], [], threading.Barrier(10)

    def go(i):
        gate.wait()
        try:
            got.append(light(c, "frames" if i % 2 else "validation"))
        except RB.ResourceBusy as e:
            refused.append(e)
    ts = [threading.Thread(target=go, args=(i,)) for i in range(10)]; [t.start() for t in ts]; [t.join() for t in ts]
    assert len(got) == 2 and len(refused) == 8 and all(e.busy_with == "light_limit" for e in refused)
    [r.release() for r in got]


def test_threads_cannot_start_a_light_job_and_a_heavy_job_together(tmp_path):
    c = coord(tmp_path); outcomes, gate = [], threading.Barrier(2)

    def heavy():
        gate.wait(); a = c.reserve("reconstruction", "o", 1000.0, 100.0); outcomes.append(("heavy", a.ok, a.reservation))

    def lite():
        gate.wait()
        try:
            outcomes.append(("light", True, light(c)))
        except RB.ResourceBusy:
            outcomes.append(("light", False, None))
    for _ in range(40):                                                                         # repeat: a race would show up as both succeeding at least once
        outcomes.clear(); gate.reset(); ts = [threading.Thread(target=heavy), threading.Thread(target=lite)]; [t.start() for t in ts]; [t.join() for t in ts]
        assert sum(ok for _, ok, _ in outcomes) == 1, outcomes
        [r.release() for _, ok, r in outcomes if ok]


def test_a_short_wait_lets_a_quick_job_finish_but_never_hangs_behind_a_long_one(tmp_path):
    c = coord(tmp_path)
    a, b = light(c, "frames"), light(c, "frames")
    threading.Timer(0.3, a.release).start()
    t0 = time.monotonic(); got = light(c, "frames", wait_s=3.0); waited = time.monotonic() - t0
    assert 0.2 < waited < 2.0; got.release(); b.release()                                       # waited for the quick job, then went ahead
    h = c.reserve("reconstruction", "o", 1000.0, 100.0)
    t0 = time.monotonic()
    with pytest.raises(RB.ResourceBusy):
        light(c, "frames", wait_s=0.5)
    assert 0.4 < time.monotonic() - t0 < 2.0                                                    # a long heavy job: refused after the short wait, no deadlock
    h.reservation.release()


# ================================================================ failure, timeout, cancel and a broken coordinator all release or refuse safely
def test_the_reservation_is_released_on_success_failure_and_cancel(tmp_path):
    c = coord(tmp_path)
    with light(c, "frames"):
        pass
    for exc in (RuntimeError("worker failed"), TimeoutError("decoder timed out"), KeyboardInterrupt(), SystemExit(1)):
        with pytest.raises(type(exc)):
            with light(c, "frames"):
                raise exc
    assert c.state()["reservations"] == [] and len(list(c.root.glob("*.res"))) == 0            # nothing was left behind
    a, b = light(c, "frames"), light(c, "frames"); a.release(); a.release(); b.release()        # release is idempotent
    assert c.state()["reservations"] == []


def test_a_garbage_collected_reservation_releases_itself(tmp_path):
    c = coord(tmp_path); r = light(c, "frames"); del r
    import gc; gc.collect()
    assert c.state()["reservations"] == []


def test_a_broken_coordinator_is_a_refusal_not_an_unbudgeted_run():
    class Broken:
        def reserve(self, *a, **k):
            raise OSError("temp dir unusable")
    with pytest.raises(RB.ResourceBusy) as e:
        RB.acquire_light("frames", "o", Broken())
    assert e.value.outcome == "UNAVAILABLE" and "could not be consulted" in e.value.reason


# ================================================================ a killed holder never leaves a stale lock (real second process)
CHILD = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, sys.argv[1])
    import resource_budget as RB
    c = RB.ResourceCoordinator(root=sys.argv[2], cfg=RB.BudgetConfig(ceiling_mb=4096.0), container=dict(limit_mb=None, reliable=False, reason="t"))
    a = c.reserve(sys.argv[4], "child", 800.0, 100.0)
    open(sys.argv[3], "w").write(a.outcome); time.sleep(300)
""")


@pytest.mark.parametrize("child_kind", ["reconstruction", "frames"])
def test_another_process_is_seen_and_its_death_releases_the_lock(tmp_path, child_kind):
    root, flag = tmp_path / "shared", tmp_path / "flag"
    p = subprocess.Popen([sys.executable, "-c", CHILD, str(Path(RB.__file__).parent), str(root), str(flag), child_kind])
    try:
        assert wait_for(flag.exists, 20) and flag.read_text() == RB.GRANTED
        mine = RB.ResourceCoordinator(root=root, cfg=RB.BudgetConfig(ceiling_mb=4096.0), snapshot=table(200.0), container=NO_CONTAINER)
        if child_kind == "reconstruction":
            with pytest.raises(RB.ResourceBusy) as e:
                RB.acquire_light("frames", "me", mine)                                          # a heavy job in ANOTHER process blocks the decoder here
            assert e.value.busy_with == "heavy"
        else:
            assert mine.reserve("reconstruction", "me", 500.0, 100.0).detail["busy_with"] == "light"   # and a decoder in another process blocks a big job here
        assert len(list(root.glob("*.res"))) == 1
        p.kill(); p.wait(10)                                                                    # the holder is killed: the kernel drops its lock
        r = RB.acquire_light("frames", "me", mine)
        assert len(list(root.glob("*.res"))) == 1                                               # the orphan record was removed; ours is the only one
        r.release()
    finally:
        if p.poll() is None:
            p.kill()

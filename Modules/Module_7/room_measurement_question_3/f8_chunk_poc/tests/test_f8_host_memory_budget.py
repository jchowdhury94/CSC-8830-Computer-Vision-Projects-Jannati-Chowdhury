"""The memory budget on a computer with NO container limit (a Mac, Windows, or an unlimited Linux host).  Synthetic cgroup trees, sample ``vm_stat`` / ``/proc/meminfo`` text and
injected memory figures only: nothing real is read and no worker is started.  What must hold:

* when it is CERTAIN that no container limit applies, the computer's physical memory is the budget and the memory free right now is re-checked at every admission;
* when the limit merely could not be read (error, garbage value, an unknown Linux layout) nothing is assumed: the conservative fallback / the operator's ceiling apply exactly as before;
* a real container limit always wins and the host reading is ignored; an operator ceiling can only lower the budget."""
import sys
from pathlib import Path

import pytest

import resource_budget as RB

MiB = RB.MiB
FREE_HOST = dict(total_mb=32768.0, available_mb=12000.0, source="test")
HOST_BOUND = dict(limit_mb=None, reliable=False, reason="no limit", host_bound=True)


def w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(str(text))


# ------------------------------------------------------------------ parsing the operating systems' own reports
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   100000.
Pages active:                                 600000.
Pages inactive:                               200000.
Pages speculative:                              1000.
Pages purgeable:                                9000.
Pages wired down:                             235000.
"""


def test_vm_stat_counts_free_inactive_and_speculative_pages_only():
    assert RB.parse_vm_stat(VM_STAT) == pytest.approx((100000 + 200000 + 1000) * 16384 / MiB)        # purgeable pages are deliberately not added


@pytest.mark.parametrize("junk", ["", "nonsense", "Pages free: 5.", None])
def test_vm_stat_garbage_is_unknown(junk):
    assert RB.parse_vm_stat(junk) is None


def test_meminfo_gives_total_and_available():
    total, avail = RB.parse_meminfo("MemTotal:       16384000 kB\nMemFree:  100 kB\nMemAvailable:    8192000 kB\n")
    assert total == pytest.approx(16384000 / 1024) and avail == pytest.approx(8192000 / 1024)
    assert RB.parse_meminfo("") == (None, None) and RB.parse_meminfo("MemTotal: x kB") == (None, None)


def test_host_memory_never_reports_more_free_than_total(monkeypatch):
    monkeypatch.setattr(RB, "_host_total_mb", lambda: 8000.0); monkeypatch.setattr(RB, "_host_available_mb", lambda: 9000.0)
    RB._host_cache[:] = [0.0, None]
    assert RB.host_memory_mb()["available_mb"] == 8000.0
    RB._host_cache[:] = [0.0, None]


# ------------------------------------------------------------------ when is the host the bound?  (only when it is certain)
def v2(root, limit="1073741824"):
    w(root / "memory.current", "104857600"); w(root / "memory.max", limit); w(root / "memory.stat", "anon 52428800\nfile 1000\n")


def test_platforms_without_cgroups_are_host_bound_but_unknown_linux_is_not(tmp_path, monkeypatch):
    for plat, expected in (("darwin", True), ("win32", True), ("linux", False)):
        monkeypatch.setattr(sys, "platform", plat)
        d = RB.container_memory_limit(str(tmp_path / "nothing-here"), str(tmp_path / "proc"))
        assert d["limit_mb"] is None and d["host_bound"] is expected, (plat, d)


def test_an_explicitly_unlimited_cgroup_is_host_bound_but_a_garbage_or_missing_limit_is_not(tmp_path):
    v2(tmp_path / "unlimited", limit="max")
    assert RB.container_memory_limit(str(tmp_path / "unlimited"), str(tmp_path / "proc"))["host_bound"] is True
    for i, junk in enumerate(["", "abc", "0", "-5", "nan"]):
        v2(tmp_path / f"junk{i}", limit=junk)
        d = RB.container_memory_limit(str(tmp_path / f"junk{i}"), str(tmp_path / "proc"))
        assert d["limit_mb"] is None and d["host_bound"] is False, (junk, d)                # an unreadable limit is "unknown", never "unlimited"
    v2(tmp_path / "limited", limit=str(512 * MiB))
    d = RB.container_memory_limit(str(tmp_path / "limited"), str(tmp_path / "proc"))
    assert d["limit_mb"] == 512.0 and d["host_bound"] is False


def test_cgroup_v1_sentinel_is_unlimited_and_host_bound(tmp_path):
    m = tmp_path / "memory"
    w(m / "memory.usage_in_bytes", "1000"); w(m / "memory.max_usage_in_bytes", "2000"); w(m / "memory.limit_in_bytes", "9223372036854771712"); w(m / "memory.stat", "rss 10\ncache 20\n")
    d = RB.container_memory_limit(str(tmp_path), str(tmp_path / "p"))
    assert d["limit_mb"] is None and d["version"] == 1 and d["host_bound"] is True


# ------------------------------------------------------------------ the budget
def test_host_memory_is_the_total_when_no_limit_applies():
    b = RB.compute_budget(RB.BudgetConfig(), HOST_BOUND, FREE_HOST)
    assert b["source"] == "host_physical_memory" and b["total_mb"] == 32768.0
    assert b["usable_mb"] == pytest.approx(b["total_mb"] - b["reserve_mb"]) and b["reserve_mb"] == pytest.approx(0.1 * 32768.0)
    assert "checked again at every start" in b["limitation"]


def test_an_operator_ceiling_can_only_lower_the_host_budget():
    b = RB.compute_budget(RB.BudgetConfig(ceiling_mb=4096.0), HOST_BOUND, FREE_HOST)
    assert b["total_mb"] == 4096.0 and "below this computer" in b["source"]
    b = RB.compute_budget(RB.BudgetConfig(ceiling_mb=999999.0), HOST_BOUND, FREE_HOST)
    assert b["total_mb"] == 32768.0 and b["source"] == "host_physical_memory"                 # an operator cannot assert more memory than the computer has


def test_nothing_is_assumed_when_the_limit_is_unknown_or_the_host_cannot_be_read():
    unknown = dict(limit_mb=None, reliable=False, reason="cgroup not readable", host_bound=False)
    for host in (FREE_HOST, None):
        b = RB.compute_budget(RB.BudgetConfig(), unknown, host)
        assert b["source"] == "conservative_fallback" and b["total_mb"] == RB.FALLBACK_TOTAL_MB == 2048.0       # the documented fallback, exactly as before
    assert RB.compute_budget(RB.BudgetConfig(), HOST_BOUND, None)["source"] == "conservative_fallback"           # host-bound, but its memory could not be read
    assert RB.compute_budget(RB.BudgetConfig(), HOST_BOUND, dict(total_mb=None, available_mb=None))["source"] == "conservative_fallback"
    b = RB.compute_budget(RB.BudgetConfig(ceiling_mb=5000.0), unknown, FREE_HOST)
    assert b["total_mb"] == 5000.0 and "operator-asserted" in b["source"]                       # an operator's statement still works where the host is not certain


def test_a_real_container_limit_always_wins_over_the_host_figure():
    c = dict(limit_mb=1000.0, reliable=True, source="cgroup_v2", reason="x", host_bound=False)
    b = RB.compute_budget(RB.BudgetConfig(), c, FREE_HOST)
    assert b["total_mb"] == 1000.0 and b["source"] == "cgroup_v2" and b["limitation"] is None
    c["host_bound"] = True                                                                      # even a (wrong) flag cannot override a readable limit
    assert RB.compute_budget(RB.BudgetConfig(), c, FREE_HOST)["total_mb"] == 1000.0


# ------------------------------------------------------------------ admission: the free memory is re-checked at every start
def coord(tmp_path, host, parent_mb=400.0, cfg=None, container=HOST_BOUND):
    snap = lambda: {RB.os.getpid(): dict(ppid=1, rss_mb=parent_mb, cpu_s=0.0)}
    return RB.ResourceCoordinator(root=tmp_path / "rb", cfg=cfg or RB.BudgetConfig(), snapshot=snap, container=container, host_memory=lambda: dict(host))


def test_wall_a_style_request_is_admitted_on_a_roomy_computer(tmp_path):
    c = coord(tmp_path, dict(total_mb=32768.0, available_mb=12000.0))
    a = c.reserve("reconstruction", "o", 3072.0, 1024.0)
    assert a.ok and a.granted_mb == 3072.0 and c.state()["budget"]["source"] == "host_physical_memory"
    a.reservation.release()


def test_a_computer_that_is_nearly_full_is_refused_with_a_clear_reason_even_though_it_has_a_lot_of_memory(tmp_path):
    c = coord(tmp_path, dict(total_mb=32768.0, available_mb=2000.0))                       # 2 GB free of 32 GB: the 10 % reserve (3.3 GB) is already gone
    a = c.reserve("reconstruction", "o", 3072.0, 1024.0)
    assert a.outcome == RB.INSUFFICIENT_MEMORY and "this job needs at least 1024 MB" in a.detail["reason"]
    assert c.state()["live_host_available_mb"] < 0 and c.state()["accounting_available_mb"] > 20000        # the accounting alone would have admitted it: the live check is what refuses


def test_a_small_computer_gets_a_smaller_grant_but_never_more_than_it_has(tmp_path):
    c = coord(tmp_path, dict(total_mb=8192.0, available_mb=3500.0), parent_mb=300.0)
    a = c.reserve("reconstruction", "o", 3072.0, 1024.0)
    live = 3500.0 - 0.1 * 8192.0
    assert a.ok and a.granted_mb == pytest.approx(live, abs=1.0) and a.granted_mb < 3072.0 and a.granted_mb <= 8192.0
    a.reservation.release()
    b = c.reserve("reconstruction", "o", 3072.0, 3000.0)                                    # a job whose minimum is above what is really free is refused
    assert b.outcome == RB.INSUFFICIENT_MEMORY


def test_live_figure_unreadable_falls_back_to_accounting_only_never_crashes(tmp_path):
    c = coord(tmp_path, dict(total_mb=16384.0, available_mb=None))
    s = c.state()
    assert s["live_host_available_mb"] is None and s["available_mb"] == pytest.approx(s["accounting_available_mb"])


def test_a_failing_host_reader_degrades_to_the_documented_fallback(tmp_path):
    def boom():
        raise OSError("vm_stat missing")
    c = RB.ResourceCoordinator(root=tmp_path / "rb", cfg=RB.BudgetConfig(), snapshot=lambda: {}, container=HOST_BOUND, host_memory=boom)
    b = c.state()["budget"]
    assert b["source"] == "conservative_fallback" and b["total_mb"] == 2048.0


def test_container_budgets_are_untouched_by_the_host_logic(tmp_path):
    c = RB.ResourceCoordinator(root=tmp_path / "rb", cfg=RB.BudgetConfig(), snapshot=lambda: {RB.os.getpid(): dict(ppid=1, rss_mb=200.0, cpu_s=0.0)},
                               container=dict(limit_mb=2048.0, reliable=True, source="cgroup_v2", reason="x", host_bound=False), host_memory=lambda: (_ for _ in ()).throw(AssertionError("host must not be read")))
    s = c.state()
    assert s["budget"]["total_mb"] == 2048.0 and s["budget"]["source"] == "cgroup_v2" and s["live_host_available_mb"] is None


def test_this_computer_can_be_read_without_error():
    d = RB.host_memory_mb()
    assert set(d) == {"total_mb", "available_mb", "source"}
    assert d["total_mb"] is None or d["total_mb"] > 0
    assert d["available_mb"] is None or 0 <= d["available_mb"] <= d["total_mb"]

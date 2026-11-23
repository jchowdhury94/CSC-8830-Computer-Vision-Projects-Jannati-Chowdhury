"""Cloud test preparation: the read-only Resource Use rows, the sampled worker CPU time and the Cloud-resources notice.
Synthetic stores and fake workers only: no recording is opened, no algorithm is touched, no limit is changed.  The notice must show exactly the numbers the backend enforces."""
import sys
import threading
import time

import pytest

import f8_result_view as RVW
import f8_ui_text as UT
import room_reconstruction as RR
import three_view_runner as TVRUN
from room_videos import F6_LIMITS
from chunk_store import MiB
from test_f8_31c1c_messages_budget_retention import coord, request_, run, WORKER_OK, writes, FAKE_PY   # noqa: F401  (fixtures and helpers)
from test_f8_d3_upload_requirements import open_room_page                                            # noqa: F401
from test_f8_31a_room_videos import _paths                                                          # noqa: F401  (autouse: the page harness needs it)
from test_f8_31b_frames import clips, cstore, complete, ready, rstore, svc                          # noqa: F401  (fixtures)
from test_f8_31b_pages import page_app, texts                                                       # noqa: F401
from test_f8_31c2a_measurement_interface import finished, ok_result                                 # noqa: F401

BURN = "import time\nt=time.process_time()\nwhile time.process_time()-t<1.0: pass\n"


# ------------------------------------------------------------------ worker CPU time
def test_worker_cpu_time_is_sampled_from_the_workers_own_process_tree(request_, tmp_path):
    r = run(request_, tmp_path, BURN, coordinator=coord(tmp_path), timeout_s=60.0)
    w = r["processing"]["worker"]
    assert w["tree_cpu_s"] is not None and 0.5 <= w["tree_cpu_s"] <= 5.0, w["tree_cpu_s"]          # about one CPU-second of work, never more than the wall time allows
    assert w["cpu_limit_s"] == RR.ReconstructionLimits().cpu_limit_s                                 # the limit that is configured is reported next to it


def test_cpu_of_the_parent_or_other_threads_is_not_attributed_to_the_worker(request_, tmp_path):
    stop = threading.Event()

    def spin():                                                                                      # the supervising process itself is busy while the worker only sleeps
        while not stop.is_set():
            pass
    th = threading.Thread(target=spin, daemon=True); th.start()
    try:
        r = run(request_, tmp_path, "import time; time.sleep(1.0)", coordinator=coord(tmp_path), timeout_s=60.0)
    finally:
        stop.set(); th.join()
    cpu = r["processing"]["worker"]["tree_cpu_s"]
    assert cpu is None or cpu < 0.3, cpu


def test_cpu_time_is_still_reported_when_the_worker_is_stopped_by_the_timeout(request_, tmp_path):
    r = run(request_, tmp_path, "while True: pass", coordinator=coord(tmp_path), timeout_s=1.5)
    w = r["processing"]["worker"]
    assert r["status"] == "FAILED" and r["failures"][0]["code"] == "WORKER_TIMEOUT"
    assert w["tree_cpu_s"] is not None and w["tree_cpu_s"] > 0.3 and w["wall_seconds"] is not None


# ------------------------------------------------------------------ Resource Use rows
FULL = dict(seconds=61.234, worker_self_peak_rss_mb=2161.4,
            worker=dict(budget=dict(outcome="GRANTED", available_mb_before=2600.0, granted_mb=2700.0), limit_mb=2700.0, requested_limit_mb=3072.0, tree_peak_rss_mb=2170.2,
                        container_guard=dict(limit_mb=3072.0, peak_used_mb=2600.5), wall_seconds=59.04, timeout_s=300.0, tree_cpu_s=410.5, cpu_limit_s=1800.0, exit_code=0))


def rows(proc):
    return dict(RVW.resource_rows(proc))


def test_resource_rows_show_the_recorded_values():
    d = rows(FULL)
    assert d["Admission"] == "Admitted" and d["Memory free before start"] == "2,600 MB" and d["Memory granted to the worker"] == "2,700 MB (requested 3,072 MB)"
    assert d["Worker peak memory (process tree, sampled)"] == "2,170 MB" and d["Worker peak memory (its own high-water mark)"] == "2,161 MB"
    assert d["Container memory in use, peak"] == "2,600 MB (container limit 3,072 MB)" and d["Worker wall-clock time"] == "59.0 s (limit 300 s)"
    assert d["Worker CPU time (sampled, all threads)"] == "410.5 s (limit 1,800 s)" and d["Total processing time"] == "61.2 s" and d["Worker exit status"] == "0"


@pytest.mark.parametrize("proc", [None, {}, {"worker": None}, {"worker": {"budget": "x", "container_guard": 3, "tree_peak_rss_mb": float("nan"), "wall_seconds": True, "exit_code": "0"}}, "garbage"])
def test_missing_or_malformed_fields_read_not_available_and_never_raise(proc):
    d = rows(proc)
    assert d["Admission"] == "not available" and d["Worker CPU time (sampled, all threads)"] == "not available" and d["Worker peak memory (process tree, sampled)"] == "not available"
    assert d["Worker wall-clock time"] == "not available" and d["Worker exit status"] == "not available"


def test_a_refused_admission_is_named_and_exposes_only_numbers_and_fixed_words():
    d = rows(dict(worker=dict(budget=dict(outcome="INSUFFICIENT_MEMORY", available_mb_before=1500.0, granted_mb=None, owner="SECRET", path="/private/x"), tmp="/private/y")))
    assert d["Admission"] == "Refused: not enough free memory" and d["Memory free before start"] == "1,500 MB" and d["Memory granted to the worker"] == "not available"
    text = " ".join(f"{a} {b}" for a, b in RVW.resource_rows(dict(worker=dict(budget=dict(owner="SECRET", path="/private/x")), job_dir="/private/z", sources={"p1": "/v.mov"})))
    assert "SECRET" not in text and "/private" not in text and ".mov" not in text


def test_the_result_view_and_the_page_carry_the_resource_use_section(svc, complete):
    s, wf = complete
    res = ok_result(); res["processing"] = FULL
    finished(svc, s, res)
    assert dict(RVW.build_view(res)["resources"])["Admission"] == "Admitted"
    at = page_app(svc, s); at.run(); assert not at.exception, at.exception
    assert "Resource Use" in texts(at)
    cells = " ".join(str(v) for tb in at.table for v in tb.value.values.flatten()) if at.table else ""
    assert "Worker CPU time (sampled, all threads)" in cells and "410.5 s (limit 1,800 s)" in cells


# ------------------------------------------------------------------ the Cloud-resources notice
def notice_text(at):
    return "\n".join(m.value for m in at.markdown)


def test_both_walls_show_the_notice_with_the_numbers_the_backend_enforces(svc):
    for nvp, wall, word, count in ((2, "A", "four", "2"), (3, "B", "six", "3")):
        at = open_room_page(nvp, 300, 1000)
        t = notice_text(at)
        lim = RR.ReconstructionLimits(); min3 = TVRUN.three_view_limits(RR, lim).min_tree_rss_mb
        assert "**This application runs on limited Streamlit Cloud resources.**" in t
        assert f"**Wall {wall}** needs {word} videos from {count} stereo viewpoints." in t
        assert "up to 300 MB" in t and "1,000 MB in total, shared by all visitors and by both walls" in t                         # the store's own limits (this page's store is 300 / 1000)
        assert f"{F6_LIMITS.min_duration_s:g}–{F6_LIMITS.max_duration_s:g} seconds" in t and UT.ACCEPTED_FORMATS in t
        assert f"stopped after {lim.timeout_s:g} seconds of run time or {lim.cpu_limit_s:g} seconds of CPU time" in t
        assert f"at least {min(lim.min_tree_rss_mb, lim.max_tree_rss_mb):,.0f} MB for two viewpoints and {min3:,.0f} MB for three" in t
        assert "needs more processing time and memory than Wall A" in t and "does not guarantee a successful measurement" in t and "not guaranteed to be enough" in t


def test_the_notice_values_are_the_enforced_defaults(svc):
    pol = svc.measurement.resource_policy(); lim = RR.ReconstructionLimits()
    assert pol == dict(timeout_s=300.0, cpu_limit_s=1800.0, min_memory_mb_2=1024.0, min_memory_mb_3=TVRUN.three_view_min_mb())
    assert (pol["timeout_s"], pol["cpu_limit_s"]) == (lim.timeout_s, lim.cpu_limit_s) and pol["min_memory_mb_3"] == 2450.0
    assert TVRUN.three_view_limits(RR, lim).timeout_s == lim.timeout_s and TVRUN.three_view_limits(RR, lim).cpu_limit_s == lim.cpu_limit_s      # three viewpoints change neither limit


def test_no_resource_limit_was_changed():
    lim = RR.ReconstructionLimits()
    assert (lim.timeout_s, lim.cpu_limit_s, lim.max_tree_rss_mb, lim.min_tree_rss_mb, lim.container_fraction) == (300.0, 1800.0, 3072.0, 1024.0, 0.90)
    assert TVRUN.THREE_VIEW_MIN_TREE_MB == 2450.0

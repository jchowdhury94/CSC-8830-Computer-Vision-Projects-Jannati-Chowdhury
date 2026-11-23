"""F8.31C-2C-1: Cloud resource snapshot.  Synthetic cgroup / mount trees and an injected coordinator only; no worker, no video, no stress."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import cloud_resource_snapshot as CS
import resource_budget as RB

HERE = Path(__file__).resolve().parent.parent
NO_CONTAINER = dict(limit_mb=None, reliable=False, reason="test")


def w(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def v2(root, *, max_="2147483648", cpu="200000 100000", current=str(500 * 1024 * 1024)):
    w(root / "memory.current", current)
    w(root / "memory.max", max_)
    w(root / "memory.stat", f"anon {300 * 1024 * 1024}\nfile {100 * 1024 * 1024}\nshmem 0\n")
    w(root / "cpu.max", cpu)
    w(root / "cpu.stat", "usage_usec 5000000\nnr_periods 10\nnr_throttled 2\nthrottled_usec 1000\n")
    return str(root)


def coord(tmp_path, container=None, ceiling=None):
    return RB.ResourceCoordinator(root=tmp_path / "rb", cfg=RB.BudgetConfig(ceiling_mb=ceiling), container=container or NO_CONTAINER,
                                  snapshot=lambda: {os.getpid(): dict(ppid=1, rss_mb=300.0, cpu_s=0.0)})


def snap(tmp_path, cg=None, **kw):
    cg = cg or str(tmp_path / "no_cgroup")
    return CS.take_snapshot(cgroup_root=cg, proc_root=str(tmp_path / "no_proc"), mounts=str(tmp_path / "no_mounts"), tmp_dir=tmp_path,
                            coordinator=kw.pop("coordinator", None) or coord(tmp_path), **kw)


def leaves(o, path=""):
    if isinstance(o, dict):
        if "kind" in o and "value" in o:
            yield path, o
        for k, v in o.items():
            yield from leaves(v, f"{path}.{k}")


# 1 -------------------------------------------------------------------------------------------------------------------------------
def test_snapshot_is_json_safe_and_tagged(tmp_path):
    s = snap(tmp_path, v2(tmp_path / "cg"))
    json.dumps(s, allow_nan=False)
    assert s["schema"] == CS.SCHEMA
    ls = list(leaves(s))
    assert ls and all(x["kind"] in ("measured", "configured", "estimated", "unknown") for _, x in ls)
    assert all("basis" in x for _, x in ls if x["kind"] == "estimated")
    assert all("reason" in x and x["value"] is None for _, x in ls if x["kind"] == "unknown")


# 2 -------------------------------------------------------------------------------------------------------------------------------
def test_missing_cgroup_is_unknown_not_guessed(tmp_path):
    s = snap(tmp_path)
    m = s["memory"]
    assert m["container_limit_mb"]["kind"] == "unknown" and m["cgroup_version"]["kind"] == "unknown"
    assert m["memory_limit_unknown"] is True and m["container_used_mb"]["kind"] == "unknown"
    assert s["cpu"]["cpu_quota_cores"]["kind"] == "unknown"
    assert m["budget_total_mb"]["value"] == RB.FALLBACK_TOTAL_MB and m["budget_total_mb"]["kind"] == "configured"      # fallback is labelled configured, not measured
    assert m["budget_limitation"]


# 3 -------------------------------------------------------------------------------------------------------------------------------
def test_unlimited_cgroup_is_reported_as_unlimited(tmp_path):
    s = snap(tmp_path, v2(tmp_path / "cg", max_="max", cpu="max 100000"))
    m = s["memory"]
    assert m["cgroup_version"]["value"] == 2 and m["limit_status"]["value"] == "unlimited"
    assert m["container_limit_mb"]["kind"] == "unknown" and "unlimited" in m["container_limit_mb"]["reason"]
    assert s["cpu"]["cpu_quota_cores"]["kind"] == "unknown" and "unlimited" in s["cpu"]["cpu_quota_cores"]["reason"]


def test_limited_cgroup_is_measured(tmp_path):
    s = snap(tmp_path, v2(tmp_path / "cg"))
    m = s["memory"]
    assert m["container_limit_mb"]["value"] == pytest.approx(2048.0) and m["container_limit_mb"]["kind"] == "measured"
    assert m["limit_status"]["value"] == "limited" and m["memory_limit_unknown"] is False
    assert m["container_used_mb"]["value"] == pytest.approx(300.0)
    assert m["safety_reserve_mb"]["value"] == pytest.approx(256.0)
    assert m["budget_usable_mb"]["value"] == pytest.approx(2048.0 - 256.0)


def test_configured_ceiling_shown_as_configured(tmp_path):
    s = snap(tmp_path, env={"F8_APP_MEMORY_CEILING_MB": "4096"})
    assert s["memory"]["memory_ceiling_configured_mb"] == dict(kind="configured", value=4096.0)


# 4 -------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [("200000 100000", 2.0), ("50000 100000", 0.5), ("max 100000", None), ("0 100000", None), ("-1 100000", None),
                                           ("garbage", None), ("", None), ("100 0", None), ("nan 100", None)])
def test_cpu_max_parsing(text, expected):
    got = CS.parse_cpu_max(text)
    assert (got is None) if expected is None else got == pytest.approx(expected)


@pytest.mark.parametrize("q,p,expected", [("150000", "100000", 1.5), ("-1", "100000", None), ("x", "1", None), ("100", "0", None)])
def test_cfs_parsing(q, p, expected):
    got = CS.parse_cfs(q, p)
    assert (got is None) if expected is None else got == pytest.approx(expected)


def test_cpu_section_uses_quota(tmp_path):
    s = snap(tmp_path, v2(tmp_path / "cg", cpu="150000 100000"))
    assert s["cpu"]["cpu_quota_cores"]["value"] == pytest.approx(1.5)
    assert s["cpu"]["effective_cpus"]["value"] <= 1.5 and s["cpu"]["effective_cpus"]["kind"] == "estimated"


# 5 / 6 ---------------------------------------------------------------------------------------------------------------------------
def test_disk_space_and_unknown_filesystem(tmp_path):
    s = snap(tmp_path)["disk"]
    assert s["temp_free_mb"]["kind"] == "measured" and s["temp_free_mb"]["value"] > 0
    assert s["temp_filesystem"]["kind"] == "unknown" and s["temp_memory_backed"]["kind"] == "unknown"
    assert s["scratch_estimate_mb"]["kind"] == "estimated" and "LOCAL" in s["scratch_estimate_mb"]["basis"]


def test_known_filesystem_and_memory_backed_flag(tmp_path):
    mounts = tmp_path / "mounts"
    mounts.write_text(f"tmpfs {os.path.realpath(tmp_path)} tmpfs rw,nosuid 0 0\n")
    d = CS.disk_section(tmp_path, str(mounts))
    assert d["temp_filesystem"]["value"] == "tmpfs" and d["temp_memory_backed"]["value"] is True


def test_upload_usage_is_aggregate_only(tmp_path):
    a = tmp_path / "f8_room_videos_abc"
    a.mkdir()
    (a / "secret_name.MOV").write_bytes(b"x" * 1_048_576)
    (tmp_path / "other_dir").mkdir()
    (tmp_path / "other_dir" / "f").write_bytes(b"y" * 5_000_000)
    d = CS.disk_section(tmp_path, str(tmp_path / "nomounts"))
    u = d["visitor_upload_storage_mb"]
    assert u["value"] == pytest.approx(1.0, abs=0.01) and u["directories"] == 1
    assert "secret_name" not in json.dumps(d)


# 7 -------------------------------------------------------------------------------------------------------------------------------
def test_no_secrets_or_private_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("F8_PASSCODE", "hunter2-super-secret")
    monkeypatch.setenv("MY_PRIVATE_TOKEN", "tok-123456")
    (tmp_path / "f8_room_videos_zz").mkdir()
    (tmp_path / "f8_room_videos_zz" / "left_air_wall_left.MOV").write_bytes(b"0")
    text = json.dumps(snap(tmp_path, v2(tmp_path / "cg")))
    for bad in ("hunter2", "tok-123456", str(tmp_path), "left_air_wall", "f8_room_videos_zz", "/Users/", os.path.expanduser("~")):
        assert bad not in text
    assert "F8_PASSCODE" not in text and "MY_PRIVATE" not in text and "HOME" not in text


def test_scrub_redacts_paths_and_nan():
    out = CS.scrub({"a": "/tmp/f8_chunk_poc_x/file.json", "b": "ok /sys/fs/cgroup fine", "c": float("nan"), "d": ["/Users/me/x"]})
    assert out["a"] == "[redacted]" and out["b"].startswith("ok") and out["c"] is None and out["d"] == ["[redacted]"]


# 8 / 9 ---------------------------------------------------------------------------------------------------------------------------
def test_module_has_no_recording_or_worker_dependencies():
    src = (HERE / "cloud_resource_snapshot.py").read_text()
    imports = set(re.findall(r"^(?:from|import)\s+([A-Za-z_][\w.]*)", src, re.M))
    forbidden = {"calibration_worker", "calibration_runner", "reconstruction_worker", "room_reconstruction", "room_videos", "chunk_store", "visitor_method_j_adapter",
                 "f8_visitor_service", "streamlit", "threading"}
    assert not (imports & forbidden), imports & forbidden
    assert "VideoCapture" not in src and "ROOM_PROCESSING_ENABLED" not in src
    assert "while True" not in src and not re.search(r"(?<![A-Za-z_.])open\(", src)        # no loop, no file written


def test_probe_off_by_default_starts_no_process(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a process was started")
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    s = snap(tmp_path)
    assert s["worker_environment"]["worker_startup_s"]["kind"] == "unknown"
    assert s["worker_environment"]["process_group_cleanup"]["reason"] == "process probe not requested"


def test_probe_command_is_a_trivial_interpreter(tmp_path, monkeypatch):
    seen = []
    real_run, real_popen = subprocess.run, subprocess.Popen
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: (seen.append(cmd), real_run(cmd, *a, **k))[1])
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, *a, **k: (seen.append(cmd), real_popen(cmd, *a, **k))[1])
    s = snap(tmp_path, probe_processes=True)
    seen = [c for c in seen if c[0] == sys.executable]            # (macOS process listing uses ps; only interpreter launches are ours)
    assert seen and all(c[1:3] == ["-I", "-c"] and ("pass" in c[3] or "sleep" in c[3]) for c in seen)
    we = s["worker_environment"]
    assert we["process_creation"]["value"] is True and we["worker_startup_s"]["kind"] == "measured"
    assert we["process_group_cleanup"]["value"] is True


def test_snapshot_does_not_allocate_much(tmp_path):
    import tracemalloc
    tracemalloc.start()
    snap(tmp_path, v2(tmp_path / "cg"))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 32 * 1024 * 1024


def test_no_original_recordings_opened(tmp_path, monkeypatch):
    opened = []
    real = open
    import builtins
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real(f, *a, **k))[1])
    snap(tmp_path, v2(tmp_path / "cg"))
    assert not [p for p in opened if re.search(r"\.(MOV|mov|mp4|avi)$|room_video|calibration/|wall_boundary", p)]


# 10 ------------------------------------------------------------------------------------------------------------------------------
def test_public_reconstruction_gate_unchanged(monkeypatch):
    monkeypatch.setenv("ROOM_PROCESSING_ENABLED", "1")
    text = (HERE.parent / "visitor_workflow" / "f8_visitor_service.py").read_text()
    assert re.search(r"^ROOM_PROCESSING_ENABLED = False\b", text, re.M)
    assert "ROOM_PROCESSING_ENABLED" not in (HERE / "cloud_diagnostic_app.py").read_text().replace("ROOM_PROCESSING_ENABLED is not", "")


def test_diagnostic_app_is_gated_and_isolated():
    import ast
    full = (HERE / "cloud_diagnostic_app.py").read_text()
    tree = ast.parse(full)
    tree.body = tree.body[1:]                                                                      # drop the module docstring
    src = ast.unparse(tree)
    assert full.index("st.stop()") < full.index("import cloud_resource_snapshot")
    assert "load_passcode" in src and "AttemptLimiter" in src
    assert src.count("take_snapshot") == 1 and "if st.button" in src                              # only on an explicit press
    for banned in ("room_reconstruction", "visitor_workflow", "chunk_store", "calibration", "st.cache_data", "write("):
        assert banned not in src.replace("calibration or reconstruction", "")
    assert "while" not in src and "threading" not in src and "open(" not in src
    for p in ("home.py", "home_navigation.py", "Pages/app_module7.py"):
        assert "cloud_diagnostic_app" not in (HERE.parents[2] / p).read_text()                     # not linked from the main app


def test_budget_functions_are_reused_not_duplicated(tmp_path):
    s = snap(tmp_path, v2(tmp_path / "cg"), coordinator=coord(tmp_path, container=dict(limit_mb=2048.0, reliable=True, source="cgroup_v2", reason="x")))
    expected = coord(tmp_path, container=dict(limit_mb=2048.0, reliable=True, source="cgroup_v2", reason="x")).state()["available_mb"]
    assert s["memory"]["available_for_new_worker_mb"]["value"] == expected

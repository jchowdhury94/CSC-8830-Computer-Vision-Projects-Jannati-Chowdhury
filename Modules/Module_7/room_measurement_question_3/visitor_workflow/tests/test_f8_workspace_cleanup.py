"""Lifecycle of the temporary frame workspaces (``f8_frames_*``): removed on close, when the owning object is garbage-collected, and at process exit; workspaces left by a process that no
longer exists are swept at the next start.  Everything is confined to ``tmp_path``.  The essential safety property: ONLY a directory that carries this application's owner marker
(and whose recorded name and creating process match) is ever removed: a folder made before the marker existed, another program's folder, a link, a young workspace, or one whose
creator is still alive is never touched."""
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import frame_worker as FW

HERE = Path(__file__).resolve().parents[1]


def dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"]); p.wait()
    return p.pid


def make_marked(root, name, pid, created=None, app=FW.OWNER_APP, marker_name=None, with_file=True):
    d = Path(root) / name; d.mkdir(parents=True)
    (d / "frame.png").write_bytes(b"x" * 100)
    if with_file:
        (d / FW.OWNER_FILE).write_text(json.dumps(dict(app=app, name=marker_name or name, pid=pid, created=time.time() - 3600 if created is None else created)))
    return d


# ------------------------------------------------------------------ normal lifecycle
def test_workspace_has_an_owner_marker_and_close_removes_everything(tmp_path):
    ws = FW.FrameWorkspace(tmp_path)
    rec = FW._owner_record(ws.path)
    assert rec and rec["app"] == FW.OWNER_APP and rec["pid"] == os.getpid() and rec["name"] == ws.path.name
    assert ws.size_bytes() == 0                                           # the marker is not counted as visitor data
    (ws.new_job_dir() / "f.png").write_bytes(b"1234")
    assert ws.size_bytes() == 4
    ws.close(); ws.close()                                                # safe twice
    assert ws.closed and not ws.path.exists()


def test_a_closed_workspace_leaves_no_pending_finalizer(tmp_path):
    ws = FW.FrameWorkspace(tmp_path); p = ws.path
    ws.close()
    p.mkdir()                                                             # somebody else re-creates that path: a later garbage collection must not touch it
    (p / "mine.txt").write_text("keep")
    del ws; gc.collect()
    assert (p / "mine.txt").read_text() == "keep"


def test_dropping_the_owner_without_closing_removes_the_workspace(tmp_path):
    ws = FW.FrameWorkspace(tmp_path); p = ws.path
    (ws.new_job_dir() / "f.png").write_bytes(b"x" * 1000)
    assert p.is_dir()
    del ws; gc.collect()                                                  # the visitor's session object was dropped without Start over
    assert not p.exists()


def test_the_finalizer_refuses_a_directory_without_our_marker(tmp_path):
    ws = FW.FrameWorkspace(tmp_path); p = ws.path
    (p / FW.OWNER_FILE).unlink()                                          # the marker is gone: the directory can no longer be positively identified
    (p / "evidence.txt").write_text("keep me")
    del ws; gc.collect()
    assert (p / "evidence.txt").read_text() == "keep me"


# ------------------------------------------------------------------ process exit and crashes
def child_script(root, mode):
    return (f"import sys; sys.path.insert(0, {str(HERE)!r}); import os, signal, frame_worker as FW\n"
            f"ws = FW.FrameWorkspace({str(root)!r}); (ws.new_job_dir() / 'f.png').write_bytes(b'x' * 500); print(ws.path, flush=True)\n"
            + ("os.kill(os.getpid(), signal.SIGKILL)\n" if mode == "kill" else ""))


def test_a_normal_process_exit_removes_workspaces_that_were_never_closed(tmp_path):
    out = subprocess.run([sys.executable, "-c", child_script(tmp_path, "exit")], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert not Path(out.stdout.strip()).exists()


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL")
def test_a_killed_process_leaves_a_marked_workspace_and_the_next_start_sweeps_it(tmp_path):
    p = subprocess.run([sys.executable, "-c", child_script(tmp_path, "kill")], capture_output=True, text=True, timeout=60)
    left = Path(p.stdout.strip())
    assert p.returncode != 0 and left.is_dir() and FW._owner_record(left)["pid"] not in (None, os.getpid())     # nothing ran at exit: it is an orphan
    assert FW.sweep_orphaned_workspaces(tmp_path, min_age_s=0.0) == [left.name]
    assert not left.exists()


# ------------------------------------------------------------------ the sweep removes only what it can positively identify
def test_the_sweep_removes_a_dead_owners_marked_workspace_and_nothing_else(tmp_path):
    gone = make_marked(tmp_path, "f8_frames_dead0001", dead_pid())
    unmarked = make_marked(tmp_path, "f8_frames_old00002", dead_pid(), with_file=False)                  # made before the marker existed
    wrong_app = make_marked(tmp_path, "f8_frames_other003", dead_pid(), app="someone-elses-tool")
    copied_marker = make_marked(tmp_path, "f8_frames_copy0004", dead_pid(), marker_name="f8_frames_dead0001")
    young = make_marked(tmp_path, "f8_frames_young005", dead_pid(), created=time.time())                  # just created: maybe its owner is still starting
    alive = make_marked(tmp_path, "f8_frames_alive006", os.getppid())                                    # its creator (this test's parent process) is alive
    mine = make_marked(tmp_path, "f8_frames_mine0007", os.getpid())                                      # created by the current process
    other_prefix = make_marked(tmp_path, "something_else", dead_pid())
    (tmp_path / "f8_frames_afile").write_text("not a directory")
    assert FW.sweep_orphaned_workspaces(tmp_path) == ["f8_frames_dead0001"]
    assert not gone.exists()
    for keep in (unmarked, wrong_app, copied_marker, young, alive, mine, other_prefix):
        assert keep.is_dir() and (keep / "frame.png").exists(), keep.name
    assert (tmp_path / "f8_frames_afile").read_text() == "not a directory"


@pytest.mark.skipif(os.name != "posix", reason="symbolic links")
def test_a_link_named_like_a_workspace_is_never_followed_or_removed(tmp_path):
    target = tmp_path / "precious"; target.mkdir(); (target / "calibration.json").write_text("{}")
    (target / FW.OWNER_FILE).write_text(json.dumps(dict(app=FW.OWNER_APP, name="f8_frames_link0001", pid=dead_pid(), created=time.time() - 3600)))   # even a forged marker behind the link
    link = tmp_path / "f8_frames_link0001"; link.symlink_to(target, target_is_directory=True)
    assert FW.sweep_orphaned_workspaces(tmp_path) == []
    assert (target / "calibration.json").read_text() == "{}" and link.is_symlink()
    assert FW._remove_owned(link) is False and (target / "calibration.json").exists()


def test_a_pid_that_cannot_be_judged_counts_as_alive():
    assert FW._pid_alive(os.getpid()) and FW._pid_alive(0) and FW._pid_alive(-3)
    assert FW._pid_alive(dead_pid()) is False or os.name != "posix"


def test_the_first_workspace_of_a_process_sweeps_its_folder_once(tmp_path):
    orphan = make_marked(tmp_path, "f8_frames_orph0001", dead_pid())
    FW._swept_roots.discard((str(tmp_path), "f8_frames_"))
    ws = FW.FrameWorkspace(tmp_path)
    assert not orphan.exists() and ws.path.is_dir()
    orphan2 = make_marked(tmp_path, "f8_frames_orph0002", dead_pid())
    ws2 = FW.FrameWorkspace(tmp_path)                                     # not again for the same folder in the same process
    assert orphan2.is_dir()
    ws.close(); ws2.close()


def test_it_never_touches_files_outside_its_own_workspaces(tmp_path):
    for n in ("calibration.json", "recording.MOV", "annotations_backup.json", "results.json"):
        (tmp_path / n).write_text(n)
    make_marked(tmp_path, "f8_frames_dead0009", dead_pid())
    FW.sweep_orphaned_workspaces(tmp_path)
    ws = FW.FrameWorkspace(tmp_path); ws.close()
    for n in ("calibration.json", "recording.MOV", "annotations_backup.json", "results.json"):
        assert (tmp_path / n).read_text() == n, n                         # untouched
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("f8_frames_")]       # and no workspace is left behind

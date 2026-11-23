"""F8.5: LEFT/RIGHT slots, combined reservations, leases, heartbeats, expiry and race-free cleanup (store level; synthetic bytes; fake clock)."""
import hashlib
import os
import subprocess
import sys
import time
from collections import namedtuple
from pathlib import Path

import pytest

import filelock
from chunk_store import ChunkStore, Limits, MiB, UploadError
from protocol import handle

CS = 64 * 1024
A, B = "owner-A-" + "a" * 16, "owner-B-" + "b" * 16
GiB = 1024 * MiB
Usage = namedtuple("Usage", "total used free")


def sha(b): return hashlib.sha256(b).hexdigest()
def code(e): return e.value.code


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


@pytest.fixture
def clock(): return Clock()


def make(tmp_path, clock, **lim):
    base = dict(max_file_bytes=64 * MiB, max_disk_bytes=10 * GiB, min_free_bytes=1, min_free_fraction=0.0, idle_timeout_s=60, max_upload_s=600, complete_retention_s=100,
                job_max_s=1000, lease_heartbeat_timeout_s=30, lease_max_s=500, max_active_leases=1)
    base.update(lim)
    return ChunkStore(Limits(**base), root=tmp_path / "root", clock=clock)


@pytest.fixture
def store(tmp_path, clock):
    s = make(tmp_path, clock); yield s; s.close()


def upload(store, owner, slot, data, fp=None, cs=CS):
    uid = store.begin(owner, "clip.mov", len(data), cs, fp or slot, slot=slot)["upload_id"]
    for i in range(0, len(data), cs):
        c = data[i:i + cs]; store.put_chunk(owner, uid, i // cs, c, sha(c))
    store.finish(owner, uid, sha(data)); return uid


def both(store, owner=A, n=CS * 3 + 5):
    L, R = os.urandom(n), os.urandom(n + 11)
    return upload(store, owner, "left", L), upload(store, owner, "right", R), L, R


def files(store): return sorted(p.name for p in Path(store.root).rglob("*") if p.is_file())


# ---------------------------------------------------------------- slots
def test_sequential_left_then_right_both_retained(store):
    lid, rid, L, R = both(store)
    js = store.job_status(A)
    assert set(js["slots"]) == {"left", "right"} and all(v["state"] == "complete" for v in js["slots"].values())
    assert store.path_of(A, lid).read_bytes() == L and store.path_of(A, rid).read_bytes() == R          # camera identity = slot, content intact
    assert js["slots"]["left"]["sha256"] == sha(L) and js["slots"]["right"]["sha256"] == sha(R)
    assert store.status(A, lid)["slot"] == "left" and store.status(A, rid)["slot"] == "right"


def test_replacing_left_keeps_right_and_vice_versa(store):
    lid, rid, L, R = both(store)
    L2 = os.urandom(CS * 2); lid2 = upload(store, A, "left", L2, fp="left-2")
    assert lid2 != lid and store.path_of(A, rid).read_bytes() == R and store.path_of(A, lid2).read_bytes() == L2
    with pytest.raises(UploadError): store.status(A, lid)                                                 # the old LEFT is gone
    R2 = os.urandom(CS * 2 + 3); rid2 = upload(store, A, "right", R2, fp="right-2")
    assert store.path_of(A, lid2).read_bytes() == L2 and store.path_of(A, rid2).read_bytes() == R2
    with pytest.raises(UploadError): store.status(A, rid)
    assert len([f for f in files(store)]) == 2


def test_invalid_slot_rejected_and_slot_mode_does_not_touch_other_owner(store):
    for bad in ("LEFT", "middle", "", 1, ["left"]):
        with pytest.raises(UploadError) as e: store.begin(A, "x", 1000, CS, "f", slot=bad)
        assert code(e) == "bad_request"
    lidA = upload(store, A, "left", os.urandom(CS))
    upload(store, B, "left", os.urandom(CS))                                                              # same slot name, different session
    assert store.status(A, lidA)["state"] == "complete"


def test_identical_files_in_both_slots_rejected(store):
    data = os.urandom(CS * 2); lid = upload(store, A, "left", data)
    uid = store.begin(A, "again.mov", len(data), CS, "r", slot="right")["upload_id"]
    for i in range(0, len(data), CS): store.put_chunk(A, uid, i // CS, data[i:i + CS], sha(data[i:i + CS]))
    with pytest.raises(UploadError) as e: store.finish(A, uid, sha(data))
    assert code(e) == "duplicate_file" and "same file" in e.value.message and "LEFT" in e.value.message
    assert store.status(A, lid)["state"] == "complete" and "right" not in store.job_status(A)["slots"]     # LEFT untouched, the duplicate removed
    assert len(files(store)) == 1
    upload(store, A, "right", os.urandom(CS * 2))                                                         # a different file is accepted
    assert set(store.job_status(A)["slots"]) == {"left", "right"}


def test_same_content_in_different_sessions_is_not_a_duplicate(store):
    data = os.urandom(CS); upload(store, A, "left", data); upload(store, B, "right", data)
    assert store.job_status(A)["slots"]["left"]["sha256"] == store.job_status(B)["slots"]["right"]["sha256"]


def test_one_active_upload_at_a_time_across_slots(store):
    store.begin(A, "l", CS * 3, CS, "l", slot="left")
    with pytest.raises(UploadError) as e: store.begin(A, "r", CS * 3, CS, "r", slot="right")
    assert code(e) == "busy"
    with pytest.raises(UploadError) as e: store.begin(B, "r", CS * 3, CS, "r", slot="right")
    assert code(e) == "busy"


def test_resume_is_per_slot(store):
    d = os.urandom(CS * 4); uid = store.begin(A, "l", len(d), CS, "same", slot="left")["upload_id"]
    store.put_chunk(A, uid, 0, d[:CS], sha(d[:CS]))
    again = store.begin(A, "l", len(d), CS, "same", slot="left")
    assert again["resumed"] and again["upload_id"] == uid and again["next_index"] == 1 and again["slot"] == "left"


# ---------------------------------------------------------------- combined disk reservation
def test_first_slot_reserves_room_for_the_second_video(tmp_path, clock):
    free = {"v": 2 * GiB}
    s = ChunkStore(Limits(max_file_bytes=2 * GiB, max_disk_bytes=10 * GiB, min_free_bytes=512 * MiB, min_free_fraction=0.0, max_active_uploads=1), root=tmp_path / "r", clock=clock,
                   disk_usage=lambda p: Usage(100 * GiB, 0, free["v"]))
    with pytest.raises(UploadError) as e: s.begin(A, "l", 800 * MiB, CS, "l", slot="left")            # 800 + 800 reserved > 2 GiB - 512 MiB
    assert code(e) == "insufficient_disk" and "both videos" in e.value.message
    ok = s.begin(A, "l", 700 * MiB, CS, "l", slot="left")                                             # 700 + 700 = 1400 <= 1536
    assert ok["state"] == "uploading"
    s.close()


def test_second_slot_does_not_reserve_a_third_video(tmp_path, clock):
    s = make(tmp_path, clock, max_disk_bytes=3 * MiB, max_file_bytes=2 * MiB)
    lid = s.begin(A, "l", 1 * MiB, CS, "l", slot="left")["upload_id"]
    d = os.urandom(1 * MiB)
    for i in range(0, len(d), CS): s.put_chunk(A, lid, i // CS, d[i:i + CS], sha(d[i:i + CS]))
    s.finish(A, lid, sha(d))
    assert s.begin(A, "r", 2 * MiB, CS, "r", slot="right")["state"] == "uploading"                   # 1 + 2 = 3 MiB fits the cap exactly: no extra reserve
    s.close()


def test_combined_logical_cap_counts_both_videos(tmp_path, clock):
    s = make(tmp_path, clock, max_disk_bytes=3 * MiB, max_file_bytes=2 * MiB)
    with pytest.raises(UploadError) as e: s.begin(A, "l", 2 * MiB, CS, "l", slot="left")             # 2 + 2 > 3 MiB
    assert code(e) == "disk_limit" and "both videos" in e.value.message
    s.close()


def test_reserve_sibling_can_be_disabled(tmp_path, clock):
    s = make(tmp_path, clock, max_disk_bytes=3 * MiB, max_file_bytes=2 * MiB, reserve_sibling=False)
    assert s.begin(A, "l", 2 * MiB, CS, "l", slot="left")["state"] == "uploading"
    s.close()


# ---------------------------------------------------------------- session isolation
def test_other_session_cannot_use_a_job(store):
    both(store); lease = store.acquire_lease(A)
    for call in (lambda: store.heartbeat(B, lease["lease_id"]), lambda: store.request_cancel(B, lease["lease_id"]), lambda: store.release_lease(B, lease["lease_id"]),
                 lambda: store.attach_process(B, lease["lease_id"], 12345)):
        with pytest.raises(UploadError) as e: call()
        assert code(e) == "not_found"
    assert store.job_status(B) == dict(slots={}, lease=None)
    with pytest.raises(UploadError) as e: store.acquire_lease(B)
    assert code(e) == "job_incomplete"


# ---------------------------------------------------------------- lease
def test_lease_needs_both_videos_and_returns_verified_paths(store):
    upload(store, A, "left", os.urandom(CS))
    with pytest.raises(UploadError) as e: store.acquire_lease(A)
    assert code(e) == "job_incomplete" and "RIGHT" in e.value.message
    lid, rid, L, R = (upload(store, A, "right", os.urandom(CS * 2)), None, None, None)
    lease = store.acquire_lease(A)
    assert Path(lease["left"]["path"]).is_file() and Path(lease["right"]["path"]).is_file() and lease["left"]["sha256"] != lease["right"]["sha256"]
    assert store.job_status(A)["lease"]["state"] == "active" and all(s["leased"] for s in store.job_status(A)["slots"].values())
    with pytest.raises(UploadError) as e: store.acquire_lease(A)
    assert code(e) == "job_running"


def test_only_one_lease_at_a_time(store):
    both(store, A); both(store, B)
    store.acquire_lease(A)
    with pytest.raises(UploadError) as e: store.acquire_lease(B)
    assert code(e) == "busy"


def test_stale_reference_checks_before_a_lease(store):
    lid, rid, L, R = both(store)
    p = store.path_of(A, rid); p.write_bytes(p.read_bytes()[:-5])                                      # file changed on disk after verification
    with pytest.raises(UploadError) as e: store.acquire_lease(A)
    assert code(e) == "job_incomplete" and "changed" in e.value.message
    p.unlink()
    with pytest.raises(UploadError) as e: store.acquire_lease(A)
    assert code(e) == "job_incomplete" and "no longer available" in e.value.message


def test_leased_files_survive_sweeps_cancel_and_replacement(store, clock):
    lid, rid, L, R = both(store); lease = store.acquire_lease(A)
    for _ in range(8):                                                                                 # 200 s of heartbeats: twice the 100 s retention, inside the hard maximum
        clock.t += 25; store.heartbeat(A, lease["lease_id"]); assert store.sweep() == []
    assert len(files(store)) == 2
    with pytest.raises(UploadError) as e: store.cancel(A, lid)
    assert code(e) == "in_use"
    with pytest.raises(UploadError) as e: store.begin(A, "new", CS, CS, "n", slot="right")
    assert code(e) == "in_use"
    with pytest.raises(UploadError) as e: store.begin(A, "new", CS, CS, "n")                           # single-file mode would discard them too
    assert code(e) == "in_use"
    assert store.path_of(A, lid).read_bytes() == L and store.path_of(A, rid).read_bytes() == R


def test_heartbeat_extends_and_silence_expires(store, clock):
    both(store); lease = store.acquire_lease(A); lid = lease["lease_id"]
    for _ in range(4):                                                                                 # 4 x 25 s = 100 s, each within the 30 s heartbeat timeout
        clock.t += 25; assert store.heartbeat(A, lid)["ok"]; assert store.sweep() == []
    assert len(files(store)) == 2
    clock.t += 31                                                                                      # the supervisor went silent: abandoned
    gone = store.sweep()
    assert len(gone) == 2 and files(store) == []
    with pytest.raises(UploadError) as e: store.heartbeat(A, lid)
    assert code(e) == "lease_expired"
    assert store.job_status(A) == dict(slots={}, lease=None)


def test_hard_maximum_lease_duration_even_with_heartbeats(store, clock):
    both(store); lid = store.acquire_lease(A)["lease_id"]
    for _ in range(19):
        clock.t += 25; store.heartbeat(A, lid)                                                         # 475 s: still inside the 500 s hard maximum
    assert store.sweep() == [] and len(files(store)) == 2
    clock.t += 26; store.sweep()                                                                       # 501 s: the hard maximum has passed
    assert files(store) == []
    with pytest.raises(UploadError) as e: store.heartbeat(A, lid)
    assert code(e) == "lease_expired"


def test_release_cleans_up_for_every_outcome(store):
    for outcome in ("completed", "failed", "cancelled", "timeout"):
        both(store); lease = store.acquire_lease(A)
        r = store.release_lease(A, lease["lease_id"], outcome)
        assert r["ok"] and r["outcome"] == outcome and files(store) == [] and store.job_status(A) == dict(slots={}, lease=None)
        assert store.release_lease(A, lease["lease_id"], outcome)["already"] is True                   # idempotent


def test_release_can_keep_the_files(store, clock):
    both(store); lease = store.acquire_lease(A)
    store.release_lease(A, lease["lease_id"], "failed", delete_files=False)
    assert len(files(store)) == 2 and not any(s["leased"] for s in store.job_status(A)["slots"].values())
    clock.t += 99
    assert store.sweep() == []                                                                          # retention restarts from the end of the job
    clock.t += 5; assert len(store.sweep()) == 2


# ---------------------------------------------------------------- retention of an unleased job
def test_left_survives_while_right_is_still_uploading(store, clock):
    lid = upload(store, A, "left", os.urandom(CS))
    d = os.urandom(CS * 6); rid = store.begin(A, "r", len(d), CS, "r", slot="right")["upload_id"]
    for i in range(6):
        clock.t += 50                                                                                  # 300 s in total: three times the 100 s retention
        store.put_chunk(A, rid, i, d[i * CS:(i + 1) * CS], sha(d[i * CS:(i + 1) * CS])); store.sweep()
    assert store.status(A, lid)["state"] == "complete"
    store.finish(A, rid, sha(d)); assert set(store.job_status(A)["slots"]) == {"left", "right"}


def test_abandoned_job_expires_without_a_lease(store, clock):
    both(store)
    clock.t += 101
    assert len(store.sweep()) == 2 and files(store) == []


def test_job_max_age_applies_to_unleased_slot_uploads(tmp_path, clock):
    s = make(tmp_path, clock, complete_retention_s=10_000, job_max_s=300)
    both(s); clock.t += 200; s.job_status(A); s.sweep(); assert len(files(s)) == 2
    clock.t += 101; assert len(s.sweep()) == 2
    s.close()


# ---------------------------------------------------------------- cleanup cannot race with a reader
def test_release_is_refused_while_a_reader_has_the_file(store):
    both(store); lease = store.acquire_lease(A)
    held = filelock.acquire_shared([lease["left"]["path"], lease["right"]["path"]])                   # what the worker does
    with pytest.raises(UploadError) as e: store.release_lease(A, lease["lease_id"], "completed")
    assert code(e) == "in_use" and len(files(store)) == 2 and store.job_status(A)["lease"]["state"] == "active"
    for h in held: h.close()
    assert store.release_lease(A, lease["lease_id"], "completed")["ok"] and files(store) == []


def test_expiry_waits_for_the_reader_then_deletes(store, clock):
    both(store); lease = store.acquire_lease(A)
    held = filelock.acquire_shared([lease["left"]["path"]])
    clock.t += 31
    assert store.sweep() == [] and len(files(store)) == 2                                              # expired, but the reader still has it: nothing deleted
    assert store.job_status(A)["lease"]["state"] == "expiring"
    for h in held: h.close()
    assert len(store.sweep()) == 2 and files(store) == []


def test_unleased_file_with_a_reader_cannot_be_replaced_or_swept(store, clock):
    lid, rid, L, R = both(store)
    held = filelock.acquire_shared([str(store.path_of(A, lid))])
    with pytest.raises(UploadError) as e: store.cancel(A, lid)
    assert code(e) == "in_use"
    with pytest.raises(UploadError) as e: store.begin(A, "n", CS, CS, "n", slot="left")
    assert code(e) == "in_use"
    clock.t += 500; store.sweep()
    assert store.path_of(A, lid).read_bytes() == L                                                     # the sweeper skipped it
    for h in held: h.close()
    store.cancel(A, lid)


def test_a_dead_reader_never_blocks_cleanup(store, tmp_path):
    lid, rid, L, R = both(store); path = str(store.path_of(A, lid))
    code_ = ("import sys,time; sys.path.insert(0, %r); import filelock; h = filelock.acquire_shared([sys.argv[1]]); print('READY', flush=True); time.sleep(120)" % str(Path(filelock.__file__).parent))
    p = subprocess.Popen([sys.executable, "-c", code_, path], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "READY"
    with pytest.raises(UploadError) as e: store.cancel(A, lid)
    assert code(e) == "in_use"                                                                         # a live reader in ANOTHER PROCESS blocks deletion
    p.kill(); p.wait()                                                                                 # the kernel drops a killed process's locks
    assert store.cancel(A, lid)["ok"]


# ---------------------------------------------------------------- workers are stopped before their files go
def _sleeper():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)


def test_expiring_lease_kills_the_registered_worker_group(store, clock):
    both(store); lease = store.acquire_lease(A); p = _sleeper()
    try:
        store.attach_process(A, lease["lease_id"], p.pid)
        clock.t += 31; store.sweep()
        for _ in range(60):
            if p.poll() is not None: break
            time.sleep(0.05)
        assert p.poll() is not None                                                                    # killed
        for _ in range(20):
            store.sweep()
            if not files(store): break
            time.sleep(0.05)
        assert files(store) == []
    finally:
        if p.poll() is None: p.kill()
        p.wait()


def test_the_store_never_signals_its_own_process_group(store, clock):
    both(store); lease = store.acquire_lease(A)
    store.attach_process(A, lease["lease_id"], os.getpid())                                           # would be suicidal: must be ignored
    clock.t += 31; store.sweep()
    assert files(store) == []                                                                          # and we are still here


def test_attach_process_validates_pid(store):
    both(store); lid = store.acquire_lease(A)["lease_id"]
    for bad in (None, 0, 1, -5, "12", True, 1.5):
        store.attach_process(A, lid, bad)
        assert store._leases[lid].worker_pid is None


def test_release_owner_cancels_a_running_job_and_removes_files(store):
    both(store); lease = store.acquire_lease(A); p = _sleeper()
    try:
        store.attach_process(A, lease["lease_id"], p.pid)
        deadline = time.time() + 5
        while time.time() < deadline and files(store):
            store.release_owner(A); time.sleep(0.05)
        assert files(store) == [] and p.wait(timeout=5) is not None
    finally:
        if p.poll() is None: p.kill()
        p.wait()


def test_close_kills_workers_and_removes_everything(clock):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB, min_free_bytes=1, min_free_fraction=0.0), clock=clock, purge_orphans=False)       # owns its root
    both(s); lease = s.acquire_lease(A); p = _sleeper(); root = s.root
    try:
        s.attach_process(A, lease["lease_id"], p.pid); s.close()
        assert p.wait(timeout=5) is not None and not root.exists()
    finally:
        if p.poll() is None: p.kill()
        p.wait()


# ---------------------------------------------------------------- protocol
def test_protocol_carries_the_slot(store):
    state = {}
    a = handle(store, A, state, dict(type="init", msg_id="1", name="x.mov", size=CS, chunk_size=CS, fingerprint="f", slot="right"))
    assert a["ok"] and a["slot"] == "right"
    b = handle(store, A, state, dict(type="init", msg_id="2", name="x.mov", size=CS, chunk_size=CS, fingerprint="g", slot="../../etc"))
    assert b["ok"] is False and b["error"] == "bad_request"

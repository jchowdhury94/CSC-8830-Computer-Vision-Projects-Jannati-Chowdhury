"""F8.1: disk-space protection (real free space, safety margin, promised bytes, concurrency, failure while writing)."""
import errno
import hashlib
import os
import threading
from collections import namedtuple
from pathlib import Path

import pytest

import chunk_store
from chunk_store import ChunkStore, Limits, MiB, UploadError

GiB = 1024 * MiB
CS = 64 * 1024
Usage = namedtuple("Usage", "total used free")
O = lambda n: f"owner-{n:02d}-" + "x" * 16


def sha(b): return hashlib.sha256(b).hexdigest()
def code(e): return e.value.code


def make(tmp_path, free, total=100 * GiB, **lim):
    state = dict(free=free, total=total)
    lim = dict(dict(max_file_bytes=50 * GiB, max_disk_bytes=500 * GiB, min_free_bytes=512 * MiB, min_free_fraction=0.0, max_active_uploads=1), **lim)
    s = ChunkStore(Limits(**lim), root=tmp_path / "root", disk_usage=lambda p: Usage(state["total"], state["total"] - state["free"], state["free"]))
    return s, state


def test_insufficient_free_space_is_rejected_with_a_clear_message(tmp_path):
    s, _ = make(tmp_path, free=700 * MiB)
    with pytest.raises(UploadError) as e: s.begin(O(1), "v.mov", 300 * MiB, CS, "f")        # 700 - 512 margin = 188 MB usable
    assert code(e) == "insufficient_disk" and "300 MB" in e.value.message and "188 MB usable" in e.value.message
    assert s.begin(O(1), "v.mov", 188 * MiB, CS, "f")["state"] == "uploading"
    s.close()


def test_margin_is_the_larger_of_fixed_and_fraction(tmp_path):
    s, _ = make(tmp_path, free=8 * GiB, total=100 * GiB, min_free_fraction=0.05)             # 5% of 100 GiB = 5 GiB > 512 MiB: 3 GiB usable
    assert s.capacity()["margin_bytes"] == 5 * GiB and s.capacity()["usable_bytes"] == 3 * GiB
    with pytest.raises(UploadError) as e: s.begin(O(1), "v", 3 * GiB + 1, CS, "f")
    assert code(e) == "insufficient_disk"
    s.begin(O(1), "v", 3 * GiB, CS, "f"); s.close()


def test_no_free_space_at_all_and_unreadable_disk_fail_closed(tmp_path):
    s, st = make(tmp_path, free=0)
    with pytest.raises(UploadError) as e: s.begin(O(1), "v", 1000, CS, "f")
    assert code(e) == "insufficient_disk" and s.capacity()["usable_bytes"] == 0
    s.close()
    def broken(path): raise OSError("statvfs failed")
    s2 = ChunkStore(Limits(), root=tmp_path / "r2", disk_usage=broken)
    with pytest.raises(UploadError) as e: s2.begin(O(1), "v", 1000, CS, "f")
    assert code(e) == "insufficient_disk"                                                    # cannot read the disk: refuse, do not guess
    s2.close()


def test_promised_bytes_of_running_uploads_are_subtracted(tmp_path):
    s, _ = make(tmp_path, free=10 * GiB, max_active_uploads=5)
    s.begin(O(1), "a", 3 * GiB, CS, "a")                                                     # promised, nothing written yet
    cap = s.capacity(); assert cap["pending_bytes"] == 3 * GiB and cap["usable_bytes"] == 10 * GiB - 3 * GiB - 512 * MiB
    with pytest.raises(UploadError) as e: s.begin(O(2), "b", 7 * GiB, CS, "b")               # 7 GiB > 6.5 GiB usable
    assert code(e) == "insufficient_disk"
    s.begin(O(2), "b", 6 * GiB, CS, "b")
    assert s.capacity()["pending_bytes"] == 9 * GiB and s.capacity()["usable_bytes"] == 512 * MiB       # 10 - 9 - 0.5 GiB margin
    s.close()


def test_written_bytes_are_not_counted_twice(tmp_path):
    """Free space shrinks as chunks are written while 'promised' shrinks by the same amount: usable space must stay the same."""
    holder = {}
    s = ChunkStore(Limits(max_file_bytes=10 * GiB, max_disk_bytes=500 * GiB, min_free_bytes=64 * MiB, min_free_fraction=0.0, chunk_floor_bytes=0), root=tmp_path / "root",
                   disk_usage=lambda p: Usage(100 * GiB, 0, 50 * GiB - holder["s"].disk_usage_bytes()))
    holder["s"] = s
    size = 40 * CS; uid = s.begin(O(1), "v", size, CS, "f")["upload_id"]; before = s.capacity()["usable_bytes"]
    for i in range(20):
        c = os.urandom(CS); s.put_chunk(O(1), uid, i, c, sha(c))
    after = s.capacity(); assert after["usable_bytes"] == before and after["pending_bytes"] == 20 * CS
    s.close()


def test_logical_reservation_cap_still_applies_on_top_of_real_space(tmp_path):
    s, _ = make(tmp_path, free=90 * GiB, max_disk_bytes=3 * GiB, max_active_uploads=5)
    s.begin(O(1), "a", 2 * GiB, CS, "a")
    with pytest.raises(UploadError) as e: s.begin(O(2), "b", 2 * GiB, CS, "b")
    assert code(e) == "disk_limit"                                                           # plenty of real space, but the configured cap wins
    s.close()


def test_concurrent_sessions_cannot_bypass_the_limit(tmp_path):
    s, _ = make(tmp_path, free=3 * GiB, max_active_uploads=16)                              # usable = 3 GiB - 512 MiB = 2.5 GiB -> exactly two 1 GiB uploads
    barrier = threading.Barrier(12); results = []

    def attempt(n):
        barrier.wait()
        try: s.begin(O(n), "v", 1 * GiB, CS, f"f{n}"); results.append("ok")
        except UploadError as e: results.append(e.code)

    ts = [threading.Thread(target=attempt, args=(n,)) for n in range(12)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert results.count("ok") == 2 and results.count("insufficient_disk") == 10 and s.reserved_bytes() == 2 * GiB
    s.close()


def test_one_active_upload_at_a_time_is_kept(tmp_path):
    s, _ = make(tmp_path, free=50 * GiB)
    s.begin(O(1), "a", 1000, CS, "a")
    with pytest.raises(UploadError) as e: s.begin(O(2), "b", 1000, CS, "b")
    assert code(e) == "busy"
    s.close()


def test_space_is_released_when_an_upload_is_cancelled_or_fails(tmp_path):
    s, st = make(tmp_path, free=1 * GiB)
    uid = s.begin(O(1), "a", 400 * MiB, CS, "a")["upload_id"]
    assert s.capacity()["usable_bytes"] == 1 * GiB - 400 * MiB - 512 * MiB
    s.cancel(O(1), uid); assert s.capacity()["usable_bytes"] == 1 * GiB - 512 * MiB and s.reserved_bytes() == 0
    s.close()


def test_disk_filling_up_mid_upload_stops_cleanly(tmp_path):
    s, st = make(tmp_path, free=50 * GiB, chunk_floor_bytes=64 * MiB)
    data = [os.urandom(CS) for _ in range(4)]; uid = s.begin(O(1), "v", CS * 4, CS, "f")["upload_id"]
    s.put_chunk(O(1), uid, 0, data[0], sha(data[0]))
    st["free"] = 64 * MiB + CS - 1                                                           # another process filled the disk
    with pytest.raises(UploadError) as e: s.put_chunk(O(1), uid, 1, data[1], sha(data[1]))
    assert code(e) == "disk_full" and "ran out" in e.value.message
    assert not any(Path(s.root).glob("*/*"))                                                 # the partial file was removed
    st["free"] = 50 * GiB; s.sweep()
    assert s.reserved_bytes() == 0 and s.active_count() == 0
    s.close()


def test_enospc_during_write_is_reported_and_cleaned(tmp_path, monkeypatch):
    s, _ = make(tmp_path, free=50 * GiB)
    d = os.urandom(CS); uid = s.begin(O(1), "v", CS, CS, "f")["upload_id"]
    real_open = open
    def fake_open(path, mode="r", *a, **k):
        if str(path).endswith("data.part") and "a" in mode: raise OSError(errno.ENOSPC, "No space left on device")
        return real_open(path, mode, *a, **k)
    monkeypatch.setattr(chunk_store, "open", fake_open, raising=False)
    with pytest.raises(UploadError) as e: s.put_chunk(O(1), uid, 0, d, sha(d))
    assert code(e) == "disk_full"
    monkeypatch.undo(); s.sweep(); assert not any(Path(s.root).glob("*/*")); s.close()


def test_other_write_errors_are_internal_not_leaky(tmp_path, monkeypatch):
    s, _ = make(tmp_path, free=50 * GiB)
    d = os.urandom(CS); uid = s.begin(O(1), "v", CS, CS, "f")["upload_id"]
    monkeypatch.setattr(chunk_store, "open", lambda *a, **k: (_ for _ in ()).throw(PermissionError(errno.EACCES, "/secret/path denied")), raising=False)
    with pytest.raises(UploadError) as e: s.put_chunk(O(1), uid, 0, d, sha(d))
    assert code(e) == "internal" and "/secret" not in e.value.message
    monkeypatch.undo(); s.close()


def test_release_owner_removes_only_that_sessions_uploads(tmp_path):
    s, _ = make(tmp_path, free=50 * GiB, max_active_uploads=5)
    a = s.begin(O(1), "a", 1000, CS, "a")["upload_id"]; b = s.begin(O(2), "b", 1000, CS, "b")["upload_id"]
    assert s.release_owner(O(1)) == 1 and s.release_owner(O(1)) == 0
    with pytest.raises(UploadError): s.status(O(1), a)
    assert s.status(O(2), b)["upload_id"] == b
    s.close()


def test_real_disk_default_still_accepts_a_normal_upload(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=4 * MiB), root=tmp_path / "root")                    # the real shutil.disk_usage
    data = os.urandom(CS * 3 + 5); uid = s.begin(O(1), "v", len(data), CS, "f")["upload_id"]
    for i in range(0, len(data), CS):
        c = data[i:i + CS]; s.put_chunk(O(1), uid, i // CS, c, sha(c))
    assert s.finish(O(1), uid, sha(data))["state"] == "complete" and s.capacity()["usable_bytes"] > 0
    s.close()

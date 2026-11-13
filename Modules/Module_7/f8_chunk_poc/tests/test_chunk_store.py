"""Isolated tests for the F8 chunk store, message protocol and video check. Synthetic data only; no existing fixture is used."""
import base64
import hashlib
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from chunk_store import ChunkStore, Limits, UploadError, safe_display_name
from protocol import handle
from video_check import inspect_video

CS = 64 * 1024          # chunk size used in tests (the minimum allowed)
OWNER, OTHER = "owner-token-aaaaaaaaaaaaaaaa", "other-token-bbbbbbbbbbbbbbbb"


def sha(b): return hashlib.sha256(b).hexdigest()
def chunks(data, cs=CS): return [data[i:i + cs] for i in range(0, len(data), cs)]


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


@pytest.fixture
def clock(): return Clock()


@pytest.fixture
def store(tmp_path, clock):
    s = ChunkStore(Limits(max_file_bytes=2 * 1024 * 1024, max_disk_bytes=3 * 1024 * 1024, idle_timeout_s=60, max_upload_s=600, complete_retention_s=120),
                   root=tmp_path / "root", clock=clock)
    yield s
    s.close()


def begin(store, data, owner=OWNER, fp="f", cs=CS):
    return store.begin(owner, "clip.mov", len(data), cs, fp)


def send_all(store, uid, data, owner=OWNER, cs=CS, start=0):
    for i, c in enumerate(chunks(data, cs)):
        if i >= start:
            store.put_chunk(owner, uid, i, c, sha(c))


def code(excinfo): return excinfo.value.code


# ---------------------------------------------------------------- ordering, assembly, SHA-256
def test_in_order_upload_assembles_identical_file(store):
    data = os.urandom(CS * 5 + 123)
    uid = begin(store, data)["upload_id"]
    send_all(store, uid, data)
    ack = store.finish(OWNER, uid, sha(data))
    assert ack["state"] == "complete" and ack["sha256"] == sha(data)
    p = store.path_of(OWNER, uid)
    assert p.read_bytes() == data
    assert stat.S_IMODE(p.stat().st_mode) == 0o600 and stat.S_IMODE(p.parent.stat().st_mode) == 0o700


def test_single_chunk_and_exact_multiple(store):
    for n in (1000, CS * 3):
        data = os.urandom(n)
        uid = begin(store, data, fp=str(n))["upload_id"]; send_all(store, uid, data)
        assert store.finish(OWNER, uid, sha(data))["state"] == "complete"
        assert store.path_of(OWNER, uid).read_bytes() == data


def test_out_of_order_chunk_rejected_and_state_unchanged(store):
    data = os.urandom(CS * 4); cs = chunks(data); uid = begin(store, data)["upload_id"]
    store.put_chunk(OWNER, uid, 0, cs[0], sha(cs[0]))
    with pytest.raises(UploadError) as e: store.put_chunk(OWNER, uid, 2, cs[2], sha(cs[2]))
    assert code(e) == "out_of_order" and e.value.extra["expected"] == 1
    assert store.status(OWNER, uid)["next_index"] == 1
    send_all(store, uid, data, start=1)                    # the sender recovers from the reported index
    assert store.finish(OWNER, uid, sha(data))["state"] == "complete"


def test_missing_chunk_blocks_finish(store):
    data = os.urandom(CS * 3); cs = chunks(data); uid = begin(store, data)["upload_id"]
    store.put_chunk(OWNER, uid, 0, cs[0], sha(cs[0]))
    with pytest.raises(UploadError) as e: store.finish(OWNER, uid, sha(data))
    assert code(e) == "incomplete" and e.value.extra["expected"] == 1


def test_index_out_of_range_and_bad_types(store):
    data = os.urandom(CS * 2); uid = begin(store, data)["upload_id"]; c = chunks(data)[0]
    for bad in (-1, 2, 99, "0", None, 1.0, True):
        with pytest.raises(UploadError) as e: store.put_chunk(OWNER, uid, bad, c, sha(c))
        assert code(e) in ("bad_index",)


# ---------------------------------------------------------------- duplicates
def test_duplicate_identical_chunk_is_idempotent(store):
    data = os.urandom(CS * 3); cs = chunks(data); uid = begin(store, data)["upload_id"]
    store.put_chunk(OWNER, uid, 0, cs[0], sha(cs[0])); store.put_chunk(OWNER, uid, 1, cs[1], sha(cs[1]))
    size_before = store.disk_usage_bytes()
    ack = store.put_chunk(OWNER, uid, 0, cs[0], sha(cs[0]))
    assert ack["duplicate"] is True and ack["next_index"] == 2 and store.disk_usage_bytes() == size_before
    send_all(store, uid, data, start=2)
    assert store.finish(OWNER, uid, sha(data))["state"] == "complete"
    assert store.path_of(OWNER, uid).read_bytes() == data      # the duplicate wrote nothing


def test_duplicate_with_different_content_rejected(store):
    data = os.urandom(CS * 2); cs = chunks(data); uid = begin(store, data)["upload_id"]
    store.put_chunk(OWNER, uid, 0, cs[0], sha(cs[0]))
    other = os.urandom(CS)
    with pytest.raises(UploadError) as e: store.put_chunk(OWNER, uid, 0, other, sha(other))
    assert code(e) == "duplicate_mismatch"


# ---------------------------------------------------------------- corruption
def test_corrupted_chunk_rejected_then_resent(store):
    data = os.urandom(CS * 2); cs = chunks(data); uid = begin(store, data)["upload_id"]
    bad = bytearray(cs[0]); bad[10] ^= 0xFF
    with pytest.raises(UploadError) as e: store.put_chunk(OWNER, uid, 0, bytes(bad), sha(cs[0]))
    assert code(e) == "corrupt_chunk" and store.status(OWNER, uid)["next_index"] == 0
    assert store.disk_usage_bytes() == 0
    send_all(store, uid, data)
    assert store.finish(OWNER, uid, sha(data))["state"] == "complete"


def test_wrong_chunk_length_rejected(store):
    data = os.urandom(CS * 2); uid = begin(store, data)["upload_id"]; short = chunks(data)[0][:-1]
    with pytest.raises(UploadError) as e: store.put_chunk(OWNER, uid, 0, short, sha(short))
    assert code(e) == "bad_length"
    big = os.urandom(CS + 1)
    with pytest.raises(UploadError) as e: store.put_chunk(OWNER, uid, 0, big, sha(big))
    assert code(e) == "bad_length"


def test_malformed_digest_rejected(store):
    data = os.urandom(CS); uid = begin(store, data)["upload_id"]
    for d in (None, "", "xyz", "A" * 64, 5):
        with pytest.raises(UploadError) as e: store.put_chunk(OWNER, uid, 0, data, d)
        assert code(e) == "bad_digest"


def test_wrong_file_hash_fails_and_deletes(store):
    data = os.urandom(CS * 2); uid = begin(store, data)["upload_id"]; send_all(store, uid, data)
    with pytest.raises(UploadError) as e: store.finish(OWNER, uid, sha(b"something else"))
    assert code(e) == "hash_mismatch" and store.disk_usage_bytes() == 0
    with pytest.raises(UploadError): store.path_of(OWNER, uid)


def test_disk_tampering_detected_by_reread(store):
    data = os.urandom(CS * 2); uid = begin(store, data)["upload_id"]; send_all(store, uid, data)
    part = next(Path(store.root).glob("*/data.part"))
    with open(part, "r+b") as fh: fh.seek(5); fh.write(b"\x00\x00\x00")
    with pytest.raises(UploadError) as e: store.finish(OWNER, uid, sha(data))
    assert code(e) == "hash_mismatch"


# ---------------------------------------------------------------- interruption / resume / timeouts
def test_interrupted_upload_resumes_from_server_position(store):
    data = os.urandom(CS * 6 + 7); cs = chunks(data)
    a = begin(store, data, fp="same"); uid = a["upload_id"]
    for i in range(3): store.put_chunk(OWNER, uid, i, cs[i], sha(cs[i]))
    again = begin(store, data, fp="same")                  # the browser reconnects and re-initialises the same file
    assert again["resumed"] is True and again["upload_id"] == uid and again["next_index"] == 3
    send_all(store, uid, data, start=3)
    assert store.finish(OWNER, uid, sha(data))["state"] == "complete"
    assert store.path_of(OWNER, uid).read_bytes() == data


def test_different_file_does_not_resume_and_discards_old(store):
    d1, d2 = os.urandom(CS * 2), os.urandom(CS * 3)
    u1 = begin(store, d1, fp="a")["upload_id"]; send_all(store, u1, d1)
    u2 = begin(store, d2, fp="b")["upload_id"]
    assert u2 != u1
    with pytest.raises(UploadError): store.status(OWNER, u1)
    assert len(list(Path(store.root).iterdir())) == 1


def test_idle_timeout_removes_upload_and_files(store, clock):
    data = os.urandom(CS * 2); uid = begin(store, data)["upload_id"]; c = chunks(data)[0]
    store.put_chunk(OWNER, uid, 0, c, sha(c))
    clock.t += 61
    assert store.sweep() == [uid] and not any(Path(store.root).iterdir())
    with pytest.raises(UploadError) as e: store.put_chunk(OWNER, uid, 1, chunks(data)[1], sha(chunks(data)[1]))
    assert code(e) == "not_found"


def test_chunk_after_timeout_without_sweep_is_refused_and_cleaned(store, clock):
    data = os.urandom(CS * 2); cs = chunks(data); uid = begin(store, data)["upload_id"]
    store.put_chunk(OWNER, uid, 0, cs[0], sha(cs[0])); clock.t += 61
    with pytest.raises(UploadError) as e: store.put_chunk(OWNER, uid, 1, cs[1], sha(cs[1]))
    assert code(e) == "timeout" and not any(Path(store.root).iterdir())


def test_max_upload_duration(store, clock):
    data = os.urandom(CS * 3); cs = chunks(data); uid = begin(store, data)["upload_id"]
    for step in range(3):
        clock.t += 50                                       # never idle >60 s, but total exceeds 600 s
        if step < 2: store.put_chunk(OWNER, uid, step, cs[step], sha(cs[step]))
    clock.t += 520
    assert store.sweep() == [uid]


def test_assembled_file_retention(store, clock):
    data = os.urandom(CS); uid = begin(store, data)["upload_id"]; send_all(store, uid, data); store.finish(OWNER, uid, sha(data))
    clock.t += 119; assert store.sweep() == []
    clock.t += 2; assert store.sweep() == [uid] and not any(Path(store.root).iterdir())


# ---------------------------------------------------------------- limits
def test_file_size_limits(store):
    for bad in (0, -5, 2 * 1024 * 1024 + 1, 1.5, "10", None, True):
        with pytest.raises(UploadError) as e: store.begin(OWNER, "x", bad, CS, "")
        assert code(e) in ("bad_size", "file_too_large")
    assert store.begin(OWNER, "x", 2 * 1024 * 1024, CS, "")["n_chunks"] == 32


def test_chunk_size_limits(store):
    for bad in (0, CS - 1, 8 * 1024 * 1024 + 1, "2", None):
        with pytest.raises(UploadError) as e: store.begin(OWNER, "x", 1000, bad, "")
        assert code(e) == "bad_chunk_size"


def test_active_upload_limit_one_visitor_at_a_time(store):
    begin(store, os.urandom(1000), owner=OWNER)
    with pytest.raises(UploadError) as e: begin(store, os.urandom(1000), owner=OTHER)
    assert code(e) == "busy"


def test_stalled_upload_is_preempted_but_active_one_is_not(store, clock):
    d = os.urandom(CS * 2); cs = chunks(d); u1 = begin(store, d, owner=OWNER)["upload_id"]
    store.put_chunk(OWNER, u1, 0, cs[0], sha(cs[0]))
    clock.t += 10
    with pytest.raises(UploadError) as e: begin(store, d, owner=OTHER)               # still active 10 s ago: protected
    assert code(e) == "busy" and store.status(OWNER, u1)["next_index"] == 1
    clock.t += 11                                                                     # 21 s of silence: the visitor refreshed or left
    u2 = begin(store, d, owner=OTHER)["upload_id"]
    assert u2 != u1 and len(list(Path(store.root).iterdir())) == 1
    with pytest.raises(UploadError) as e: store.status(OWNER, u1)
    assert code(e) == "not_found"


def test_disk_reservation_limit(tmp_path, clock):
    s = ChunkStore(Limits(max_file_bytes=2 * 1024 * 1024, max_disk_bytes=3 * 1024 * 1024, max_active_uploads=5), root=tmp_path / "r", clock=clock)
    s.begin("a" * 20, "x", 2 * 1024 * 1024, CS, "")
    with pytest.raises(UploadError) as e: s.begin("b" * 20, "x", 2 * 1024 * 1024, CS, "")
    assert code(e) == "disk_limit"
    s.begin("b" * 20, "x", 1024 * 1024, CS, "")            # 2 + 1 = 3 MB fits exactly
    s.close()


def test_reserved_bytes_returned_after_cancel(store):
    uid = begin(store, os.urandom(1000))["upload_id"]
    assert store.reserved_bytes() == 1000
    store.cancel(OWNER, uid)
    assert store.reserved_bytes() == 0 and store.active_count() == 0


# ---------------------------------------------------------------- security
def test_other_owner_cannot_touch_upload(store):
    data = os.urandom(CS * 2); cs = chunks(data); uid = begin(store, data)["upload_id"]
    store.put_chunk(OWNER, uid, 0, cs[0], sha(cs[0]))
    for call in (lambda: store.put_chunk(OTHER, uid, 1, cs[1], sha(cs[1])), lambda: store.finish(OTHER, uid, sha(data)), lambda: store.cancel(OTHER, uid),
                 lambda: store.status(OTHER, uid), lambda: store.path_of(OTHER, uid), lambda: store.metrics(OTHER, uid)):
        with pytest.raises(UploadError) as e: call()
        assert code(e) == "not_found"                       # same answer as for a non-existent id
    assert store.status(OWNER, uid)["next_index"] == 1


def test_path_traversal_attempts(store, tmp_path):
    uid = store.begin(OWNER, "../../../../etc/passwd", 1000, CS, "")["upload_id"]
    assert safe_display_name("../../../../etc/passwd") == "passwd" and safe_display_name("C:\\x\\..\\y.mov") == "y.mov"
    assert safe_display_name("a\x00b\n.mov") == "ab.mov" and safe_display_name("") == "video" and len(safe_display_name("a" * 500)) == 80
    files = [p for p in Path(store.root).rglob("*") if p.is_file()]
    assert files and all(p.name in ("data.part", "data.bin") and Path(store.root) in p.parents for p in files)
    for evil in ("../" + uid, "..", "../..", "/etc/passwd", uid + "/../x", "", None, 5, "a" * 100):
        with pytest.raises(UploadError) as e: store.status(OWNER, evil)
        assert code(e) == "not_found"
    assert not (tmp_path / "etc").exists()


def test_root_is_private_and_upload_ids_are_unguessable(store):
    assert stat.S_IMODE(Path(store.root).stat().st_mode) == 0o700
    ids = set()
    for i in range(30):
        uid = store.begin(OWNER, "x", 100, CS, f"fp{i}")["upload_id"]; ids.add(uid)
    assert len(ids) == 30 and all(len(u) >= 24 for u in ids)


def test_same_session_cannot_hold_more_than_one_upload(store):
    for i in range(5): store.begin(OWNER, "x", 5000, CS, f"fp{i}")
    assert len(list(Path(store.root).iterdir())) == 1


# ---------------------------------------------------------------- cleanup
def test_cancel_removes_files(store):
    data = os.urandom(CS * 2); uid = begin(store, data)["upload_id"]; send_all(store, uid, data)
    assert store.disk_usage_bytes() == len(data)
    store.cancel(OWNER, uid)
    assert store.disk_usage_bytes() == 0 and not any(Path(store.root).iterdir())


def test_release_after_complete_removes_file(store):
    data = os.urandom(CS); uid = begin(store, data)["upload_id"]; send_all(store, uid, data); store.finish(OWNER, uid, sha(data))
    store.release(OWNER, uid)
    assert not any(Path(store.root).iterdir())


def test_close_removes_everything_including_owned_root():
    s = ChunkStore(Limits(), purge_orphans=False)
    root = s.root; uid = s.begin(OWNER, "x", 1000, CS, "")["upload_id"]
    assert root.exists(); s.close()
    assert not root.exists()


def test_stale_foreign_dirs_are_not_touched_but_old_prefixed_roots_are_purged(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    stale = tmp_path / "f8_chunk_poc_old"; stale.mkdir(); (stale / "x").write_text("1")
    fresh = tmp_path / "f8_chunk_poc_fresh"; fresh.mkdir()
    other = tmp_path / "unrelated_dir"; other.mkdir(); (other / "keep").write_text("1")
    old = 1.0
    os.utime(stale, (old, old))
    s = ChunkStore(Limits())
    try:
        assert not stale.exists() and fresh.exists() and (other / "keep").exists()
    finally:
        s.close()


# ---------------------------------------------------------------- protocol layer
def b64(b): return base64.b64encode(b).decode()


def run(store, state, msg): return handle(store, OWNER, state, msg)


def test_protocol_full_flow_and_cached_ack_for_reruns(store):
    data = os.urandom(CS * 3 + 5); state = {}
    a = run(store, state, dict(type="init", msg_id="c-1", name="v.mov", size=len(data), chunk_size=CS, fingerprint="f")); assert a["ok"]
    uid = a["upload_id"]
    for i, c in enumerate(chunks(data)):
        a = run(store, state, dict(type="chunk", msg_id=f"c-{i + 2}", upload_id=uid, index=i, sha256=sha(c), data=b64(c))); assert a["ok"] and a["msg_id"] == f"c-{i + 2}"
    n_processed = state["processed"]
    again = run(store, state, dict(type="chunk", msg_id=f"c-{len(chunks(data)) + 1}", upload_id=uid, index=3, sha256="0" * 64, data="!!"))
    assert again == state["last_ack"] and state["processed"] == n_processed       # rerun re-delivery: no side effect
    fin = run(store, state, dict(type="finish", msg_id="c-99", upload_id=uid, sha256=sha(data), client_stats=dict(upload_s=1.0)))
    assert fin["ok"] and fin["state"] == "complete" and store.path_of(OWNER, uid).read_bytes() == data
    assert store.metrics(OWNER, uid)["client_stats"] == dict(upload_s=1.0)


def test_protocol_rejects_malformed_messages(store):
    state = {}
    assert run(store, state, None) is None and run(store, state, "x") is None and run(store, state, {"type": "init"}) is None   # no msg_id: ignored
    for i, bad in enumerate([dict(type="nope"), dict(type="init", name="x", size="big", chunk_size=CS), dict(type="chunk", upload_id="x" * 20, index=0, sha256="0" * 64, data=5)]):
        a = run(store, state, dict(bad, msg_id=f"m{i}")); assert a["ok"] is False and a["error"] in ("bad_request", "bad_size", "not_found")
    uid = run(store, state, dict(type="init", msg_id="i", name="x", size=CS, chunk_size=CS, fingerprint="f"))["upload_id"]
    a = run(store, state, dict(type="chunk", msg_id="b1", upload_id=uid, index=0, sha256="0" * 64, data="@@@not base64@@@")); assert a["error"] == "corrupt_chunk"
    huge = "A" * (8 * 1024 * 1024 * 4 // 3 + 100)
    a = run(store, state, dict(type="chunk", msg_id="b2", upload_id=uid, index=0, sha256="0" * 64, data=huge)); assert a["error"] == "bad_length"


def test_protocol_internal_errors_do_not_leak(store, monkeypatch):
    def boom(*a, **k): raise RuntimeError("secret path /tmp/xyz")
    monkeypatch.setattr(store, "begin", boom)
    a = run(store, {}, dict(type="init", msg_id="z", name="x", size=10, chunk_size=CS))
    assert a["ok"] is False and a["error"] == "internal" and "secret" not in a["message"]


# ---------------------------------------------------------------- memory behaviour (fresh interpreter so RSS is clean)
def test_memory_stays_bounded_for_a_256mb_upload(tmp_path):
    script = textwrap.dedent(f"""
        import hashlib, os, sys, resource
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from chunk_store import ChunkStore, Limits, MiB
        from protocol import handle
        import base64, pickle
        def rss():
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return r / MiB if sys.platform == 'darwin' else r / 1024
        s = ChunkStore(Limits(max_file_bytes=1024*MiB), root={str(tmp_path / 'm')!r})
        size, cs = 256 * MiB, 2 * MiB
        state = {{}}
        owner = 'o' * 24
        uid = handle(s, owner, state, dict(type='init', msg_id='i', name='x', size=size, chunk_size=cs, fingerprint='f'))['upload_id']
        # 8 distinct pre-built chunks are cycled so the harness itself allocates nothing per chunk: this measures the SERVER path.
        blocks = [os.urandom(cs) for _ in range(8)]
        enc = [(hashlib.sha256(b).hexdigest(), base64.b64encode(b).decode()) for b in blocks]
        base = rss()
        full = hashlib.sha256()
        for i in range(size // cs):
            k = i % 8
            full.update(blocks[k])
            a = handle(s, owner, state, dict(type='chunk', msg_id=f'c{{i}}', upload_id=uid, index=i, sha256=enc[k][0], data=enc[k][1]))
            assert a['ok'], a
        fin = handle(s, owner, state, dict(type='finish', msg_id='f', upload_id=uid, sha256=full.hexdigest()))
        assert fin['ok'], fin
        growth = rss() - base
        held = len(pickle.dumps((state, [(u.chunk_digests, u.server_ms) for u in s._uploads.values()])))
        print('GROWTH_MB', round(growth, 1), 'HELD_BYTES', held)
        assert os.path.getsize(s.path_of(owner, uid)) == size
        s.close()
    """)
    r = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=300)
    print(r.stdout)
    assert r.returncode == 0, r.stdout + r.stderr
    growth = float(r.stdout.split("GROWTH_MB")[1].split()[0]); held = int(r.stdout.split("HELD_BYTES")[1].split()[0])
    assert growth < 60, f"peak RSS grew {growth} MB for a 256 MB upload"          # a whole-file copy would add >= 256 MB
    assert held < 200_000, f"session/registry metadata is {held} bytes"            # digests + timings only, not file data


# ---------------------------------------------------------------- video inspection
def test_video_check_on_synthetic_video(tmp_path):
    from make_synthetic import make_video
    p = make_video(tmp_path / "v.mp4", seconds=2, w=320, h=180, fps=25)
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    r = inspect_video(p, max_frames=5)
    assert r["opened"] and r["decodable"] and (r["width"], r["height"]) == (320, 180) and r["frames_read"] == 5
    assert abs(r["fps"] - 25) < 0.5 and r["frame_count"] in range(49, 52) and r["duration_s"] and r["codec"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before                    # read-only


def test_video_check_rejects_non_video_and_missing(tmp_path):
    p = tmp_path / "junk.bin"; p.write_bytes(os.urandom(5000))
    r = inspect_video(p); assert not r["opened"] and not r["decodable"] and r["issues"]
    assert not inspect_video(tmp_path / "missing.mp4")["opened"]


def test_end_to_end_chunked_video_is_openable(store, tmp_path):
    from make_synthetic import make_video
    p = make_video(tmp_path / "src.mp4", seconds=2, w=320, h=180, fps=25); data = p.read_bytes()
    uid = begin(store, data)["upload_id"]; send_all(store, uid, data)
    store.finish(OWNER, uid, sha(data))
    r = inspect_video(store.path_of(OWNER, uid))
    assert r["opened"] and r["decodable"] and (r["width"], r["height"]) == (320, 180)

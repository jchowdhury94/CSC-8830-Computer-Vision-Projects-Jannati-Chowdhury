"""Stage F8 PoC: server-side chunk store (NO Streamlit import, so it is unit-testable).

Chunks are appended straight to a file on disk; nothing but small metadata (counters, one 32-byte digest per chunk) is kept in RAM.

Security model
  * The browser never supplies a path. The destination is  <root>/<upload_id>/data.part , where upload_id is generated here.
  * The visitor-supplied file name is only sanitised for DISPLAY; it never reaches the file system.
  * Every call takes an ``owner`` token (one per Streamlit session). A wrong owner and an unknown id are indistinguishable (``not_found``).
  * The root directory is private (0o700), files are 0o600, nothing here serves files over HTTP.
  * Hard limits: file size, total reserved disk, simultaneously active uploads, chunk-size range, idle timeout, total upload duration.
"""
from __future__ import annotations

import atexit
import errno
import hashlib
import logging
import os
import re
import secrets
import shutil
import signal
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import filelock

log = logging.getLogger("f8_chunk_store")          # logs ids and sizes only: never file names or contents

MiB = 1024 * 1024
ROOT_PREFIX = "f8_chunk_poc_"
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class UploadError(Exception):
    """``code`` is a stable machine-readable string; ``message`` is safe to show to the visitor."""

    def __init__(self, code: str, message: str, **extra):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra

    def to_dict(self):
        return dict(ok=False, error=self.code, message=self.message, **self.extra)


@dataclass(frozen=True)
class Limits:
    max_file_bytes: int = 1024 * MiB          # 1 GiB: room for the 800 MB target
    max_disk_bytes: int = 3 * 1024 * MiB      # all uploads together (reserved at begin)
    max_active_uploads: int = 1               # simultaneously UPLOADING (one visitor at a time)
    min_free_bytes: int = 512 * MiB           # disk space that must stay free after every accepted upload ...
    min_free_fraction: float = 0.05           # ... or this fraction of the disk, whichever is larger
    chunk_floor_bytes: int = 64 * MiB         # a chunk is refused if writing it would leave less than this free
    min_chunk_bytes: int = 64 * 1024
    max_chunk_bytes: int = 8 * MiB
    default_chunk_bytes: int = 2 * MiB
    preempt_idle_s: float = 20.0              # another session may take over an upload that has been silent this long (e.g. the visitor refreshed the page)
    idle_timeout_s: float = 300.0             # no chunk for this long -> upload expires and is deleted
    max_upload_s: float = 3600.0              # wall-clock cap for one upload
    complete_retention_s: float = 900.0       # an assembled file is deleted after this unless released earlier (no session-end hook exists)
    job_max_s: float = 7200.0                 # a LEFT/RIGHT slot upload is never kept longer than this without a lease
    max_active_leases: int = 1                # one calibration job at a time
    lease_heartbeat_timeout_s: float = 120.0  # a lease whose heartbeat stops this long is abandoned and expires
    lease_max_s: float = 7200.0               # hard maximum lifetime of any lease, heartbeat or not
    reserve_sibling: bool = True              # the first slot upload also reserves room for the other camera's video (assumed the same size)


def env_limits() -> Limits:
    """Limits from F8_* environment variables (sizes in MB, times in seconds); defaults otherwise. Invalid values are ignored."""
    d = Limits()

    def num(name, default, scale=1):
        try:
            v = float(os.environ[name]) * scale
            return type(default)(v) if v > 0 else default
        except (KeyError, ValueError):
            return default

    return Limits(max_file_bytes=num("F8_MAX_FILE_MB", d.max_file_bytes, MiB), max_disk_bytes=num("F8_MAX_DISK_MB", d.max_disk_bytes, MiB),
                  max_active_uploads=num("F8_MAX_ACTIVE", d.max_active_uploads), min_free_bytes=num("F8_MIN_FREE_MB", d.min_free_bytes, MiB), idle_timeout_s=num("F8_IDLE_TIMEOUT_S", d.idle_timeout_s),
                  max_upload_s=num("F8_MAX_UPLOAD_S", d.max_upload_s), complete_retention_s=num("F8_RETENTION_S", d.complete_retention_s))


def safe_display_name(name) -> str:
    """For display only. Never used to build a path."""
    s = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    s = "".join(ch for ch in s if ch.isprintable())
    return s.strip()[:80] or "video"


UPLOADING, COMPLETE, FAILED = "uploading", "complete", "failed"
SLOTS = ("left", "right")                                           # camera identity comes from the slot, never from a file name


@dataclass
class Upload:
    upload_id: str
    owner: str
    display_name: str
    size: int
    chunk_size: int
    n_chunks: int
    fingerprint: str
    directory: Path
    path: Path
    t_created: float
    t_activity: float
    state: str = UPLOADING
    next_index: int = 0
    bytes_received: int = 0
    hasher: object = field(default_factory=hashlib.sha256)        # running SHA-256 (internal state is ~100 bytes)
    chunk_digests: list = field(default_factory=list)             # 32 bytes per chunk (2 MiB chunk -> 800 MB is ~13 KB)
    server_ms: list = field(default_factory=list)                 # per-chunk handling time
    t_first_chunk: float | None = None
    t_complete: float | None = None
    sha256: str | None = None
    client_stats: dict | None = None
    duplicates: int = 0
    rejected: dict = field(default_factory=dict)
    slot: str | None = None                                       # 'left' | 'right' | None (single-file mode)
    leased_by: str | None = None                                  # lease id while a calibration job protects this file


@dataclass
class Lease:
    """A calibration job's claim on BOTH of a session's videos. While it is active nothing may delete them."""
    lease_id: str
    owner: str
    upload_ids: dict                                              # slot -> upload id
    t_created: float
    t_heartbeat: float
    hard_deadline: float
    state: str = "active"                                         # active | expiring | released | expired
    cancel_requested: bool = False
    worker_pid: int | None = None
    outcome: str | None = None


class ChunkStore:
    def __init__(self, limits: Limits | None = None, root: str | os.PathLike | None = None, clock=time.monotonic, purge_orphans: bool = True,
                 disk_usage=shutil.disk_usage):
        self.limits = limits or Limits()
        self._clock = clock
        self._disk_usage = disk_usage                                 # injectable for tests
        self._lock = threading.RLock()
        self._uploads: dict[str, Upload] = {}
        self._leases: dict[str, Lease] = {}
        if root is None:
            if purge_orphans:
                self._purge_stale_roots()
            root = tempfile.mkdtemp(prefix=ROOT_PREFIX)               # private by construction (0o700)
            self._owns_root = True
        else:
            Path(root).mkdir(parents=True, exist_ok=True)
            self._owns_root = False
        self.root = Path(root)
        os.chmod(self.root, 0o700)
        self._closed = False
        atexit.register(self.close)

    # ------------------------------------------------------------ housekeeping
    def _purge_stale_roots(self, max_age_s: float | None = None):
        """Remove roots left by a previous crashed process (matching prefix, older than 2x retention). Never touches anything else."""
        age = 2 * self.limits.complete_retention_s if max_age_s is None else max_age_s
        tmp = Path(tempfile.gettempdir())
        try:
            for p in tmp.glob(ROOT_PREFIX + "*"):
                try:
                    if p.is_dir() and not p.is_symlink() and time.time() - p.stat().st_mtime > age:
                        shutil.rmtree(p, ignore_errors=True)
                except OSError:
                    pass
        except OSError:
            pass

    def close(self):
        """Delete every upload and, if this store created it, the root directory. Running workers are killed first."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for lease in list(self._leases.values()):
                if lease.state in ("active", "expiring"):
                    lease.cancel_requested = True
                    self._kill_worker(lease)
            for u in list(self._uploads.values()):
                self._delete(u, force=True)
            if self._owns_root:
                shutil.rmtree(self.root, ignore_errors=True)

    def _delete(self, u: Upload, force: bool = False):
        """Remove an upload's files. Refuses (UploadError 'in_use') while a lease protects it or any process holds the file open for reading."""
        held = None
        if not force:
            if u.leased_by is not None:
                raise UploadError("in_use", "These videos are in use by a running calibration. Cancel the calibration first.")
            if u.state == COMPLETE:                                   # only completed files are ever read by workers
                held = filelock.try_exclusive(u.path)
                if held is None:
                    raise UploadError("in_use", "This video is still being read by a running process.")
        try:
            shutil.rmtree(u.directory, ignore_errors=True)
            self._uploads.pop(u.upload_id, None)
        finally:
            if held is not None:
                held.close()

    def _kill_worker(self, lease: Lease):
        """SIGKILL the registered worker's process group (it was started as its own group leader). Never touches our own group."""
        pid = lease.worker_pid
        if not pid or pid == os.getpid():
            return
        try:
            if os.getpgid(pid) == pid and pid != os.getpgrp():
                os.killpg(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def _try_release(self, lease: Lease, outcome: str, delete_files: bool) -> bool:
        """Unprotect (and normally delete) the lease's files. Returns False, changing nothing, if any reader still holds a file."""
        ups = [self._uploads[i] for i in lease.upload_ids.values() if i in self._uploads]
        locks = []
        for u in ups:                                                 # all-or-nothing: take every exclusive lock first
            h = filelock.try_exclusive(u.path)
            if h is None:
                for x in locks:
                    x.close()
                return False
            locks.append(h)
        try:
            now = self._clock()
            for u in ups:
                u.leased_by = None
                u.t_activity = now                                    # retention restarts from the end of the job
            lease.state = "released" if outcome != "expired" else "expired"
            lease.outcome = outcome; lease.worker_pid = None
            if delete_files:
                for u in ups:
                    self._delete(u, force=True)                       # we hold the exclusive lock: no reader can be mid-read
        finally:
            for x in locks:
                x.close()
        return True

    def _sweep_leases(self, now: float) -> list:
        gone = []
        for lease in list(self._leases.values()):
            if lease.state not in ("active", "expiring"):
                if now - lease.t_heartbeat > 3600 and lease.state in ("released", "expired"):
                    self._leases.pop(lease.lease_id, None)            # forget finished leases after an hour (bounded memory)
                continue
            if lease.state == "active" and not (now - lease.t_heartbeat > self.limits.lease_heartbeat_timeout_s or now > lease.hard_deadline):
                continue
            lease.state = "expiring"; lease.cancel_requested = True   # abandoned or over-long: stop the worker, then delete once it is really gone
            self._kill_worker(lease)
            ids = [i for i in lease.upload_ids.values() if i in self._uploads]
            if self._try_release(lease, "expired", True):
                gone += ids
        return gone

    def sweep(self) -> list:
        """Delete expired uploads and leases (idle / over-long / abandoned / retained too long). Returns the ids removed.
        Files protected by an active lease, or still open in a reader, are never deleted here."""
        now = self._clock(); gone = []
        with self._lock:
            gone += self._sweep_leases(now)
            for u in list(self._uploads.values()):
                if u.leased_by is not None or u.upload_id not in self._uploads:
                    continue
                if u.state == UPLOADING and (now - u.t_activity > self.limits.idle_timeout_s or now - u.t_created > self.limits.max_upload_s):
                    expired = True
                elif u.state == COMPLETE:
                    if u.slot:                                        # a job keeps its files while the session is active (e.g. uploading the other video)
                        anchor = max([u.t_complete or u.t_created] + [o.t_activity for o in self._uploads.values() if o.owner == u.owner])
                        expired = now - anchor > self.limits.complete_retention_s or now - u.t_created > self.limits.job_max_s
                    else:
                        expired = now - (u.t_complete or u.t_created) > self.limits.complete_retention_s
                else:
                    expired = u.state == FAILED
                if expired:
                    try:
                        self._delete(u)
                        gone.append(u.upload_id)
                    except UploadError:
                        log.info("sweep skipped an upload that is in use")
        if gone:
            log.info("swept %d expired upload(s)", len(gone))
        return gone

    # ------------------------------------------------------------ lookup (owner-checked)
    def _get(self, owner: str, upload_id) -> Upload:
        u = self._uploads.get(upload_id) if isinstance(upload_id, str) and _ID_RE.match(upload_id) else None
        if u is None or not secrets.compare_digest(u.owner.encode(), str(owner).encode()):
            raise UploadError("not_found", "This upload does not exist (it may have expired). Please select the file again.")
        return u

    def reserved_bytes(self) -> int:
        return sum(u.size for u in self._uploads.values())

    def disk_usage_bytes(self) -> int:
        """Actual bytes currently on disk under the root."""
        total = 0
        for dirpath, _, files in os.walk(self.root):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        return total

    def _free_bytes(self) -> int:
        """Real free space of the volume holding the root. Fails CLOSED: 0 if it cannot be read."""
        try:
            return int(self._disk_usage(self.root).free)
        except (OSError, ValueError, AttributeError):
            return 0

    def _margin_bytes(self) -> int:
        try:
            total = int(self._disk_usage(self.root).total)
        except (OSError, ValueError, AttributeError):
            total = 0
        return max(self.limits.min_free_bytes, int(self.limits.min_free_fraction * total))

    def _pending_bytes(self) -> int:
        """Bytes accepted uploads still have to write (declared size minus what they have already put on disk)."""
        return sum(max(u.size - u.bytes_received, 0) for u in self._uploads.values() if u.state == UPLOADING)

    def capacity(self) -> dict:
        """What a new upload could use right now: free space, minus bytes already promised to running uploads, minus the safety margin."""
        with self._lock:
            free, pending, margin = self._free_bytes(), self._pending_bytes(), self._margin_bytes()
            return dict(free_bytes=free, pending_bytes=pending, margin_bytes=margin, usable_bytes=max(free - pending - margin, 0),
                        reserved_bytes=self.reserved_bytes(), logical_remaining_bytes=max(self.limits.max_disk_bytes - self.reserved_bytes(), 0))

    def active_count(self) -> int:
        return sum(1 for u in self._uploads.values() if u.state == UPLOADING)

    # ------------------------------------------------------------ protocol steps
    def begin(self, owner: str, name, size, chunk_size, fingerprint="", slot=None) -> dict:
        lim = self.limits
        if not isinstance(owner, str) or not owner:
            raise UploadError("bad_request", "Missing session.")
        if slot is not None and slot not in SLOTS:
            raise UploadError("bad_request", "Unknown video slot.")
        if not (isinstance(size, int) and not isinstance(size, bool)) or size <= 0:
            raise UploadError("bad_size", "The file is empty or its size is invalid.")
        if size > lim.max_file_bytes:
            raise UploadError("file_too_large", f"The file is {size / MiB:.0f} MB; the limit is {lim.max_file_bytes / MiB:.0f} MB.")
        if not (isinstance(chunk_size, int) and not isinstance(chunk_size, bool)) or not lim.min_chunk_bytes <= chunk_size <= lim.max_chunk_bytes:
            raise UploadError("bad_chunk_size", f"Chunk size must be between {lim.min_chunk_bytes} and {lim.max_chunk_bytes} bytes.")
        fingerprint = str(fingerprint)[:200]
        with self._lock:
            self.sweep()
            mine = [u for u in self._uploads.values() if u.owner == owner]
            for u in mine:                                            # resume: same slot, same file, same chunking, still uploading
                if u.state == UPLOADING and u.slot == slot and u.fingerprint == fingerprint and u.size == size and u.chunk_size == chunk_size and fingerprint:
                    u.t_activity = self._clock()
                    return self._ack(u, resumed=True)
            # single-file mode: a session keeps one upload (starting another discards the old one). Slot mode: replacing a slot touches ONLY that slot.
            victims = [u for u in mine if slot is None or u.slot == slot or u.slot is None]
            if any(u.leased_by for u in victims):
                raise UploadError("in_use", "These videos are in use by a running calibration. Cancel the calibration before replacing a video.")
            for u in victims:
                self._delete(u)
            if self.active_count() >= lim.max_active_uploads:          # take over stalled uploads of other sessions (a refreshed page is a NEW session)
                for u in [u for u in self._uploads.values() if u.state == UPLOADING and self._clock() - u.t_activity > lim.preempt_idle_s]:
                    self._delete(u)
            if self.active_count() >= lim.max_active_uploads:
                raise UploadError("busy", "Another upload is in progress. Please try again in a few minutes.")
            has_sibling = slot is not None and any(u.owner == owner and u.slot and u.slot != slot for u in self._uploads.values())
            extra = size if (slot is not None and lim.reserve_sibling and not has_sibling) else 0     # room for the other camera's video too
            both = " for both videos" if extra else ""
            if self.reserved_bytes() + size + extra > lim.max_disk_bytes:
                raise UploadError("disk_limit", f"The server's temporary storage limit would be exceeded{both}. Please try again later.")
            usable = self.capacity()["usable_bytes"]                   # REAL free space; decided under the same lock for every session
            if size + extra > usable:
                raise UploadError("insufficient_disk", f"The server does not have enough free temporary storage{both} for a {size / MiB:.0f} MB file right now "
                                                       f"({usable / MiB:.0f} MB usable). Try a smaller file or try again later.")
            uid = secrets.token_urlsafe(18)
            d = self.root / uid
            d.mkdir(mode=0o700)
            path = d / "data.part"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.close(fd)
            now = self._clock()
            u = Upload(uid, owner, safe_display_name(name), size, chunk_size, -(-size // chunk_size), fingerprint, d, path, now, now, slot=slot)
            self._uploads[uid] = u
            log.info("upload %s started: %d bytes, %d chunks, slot=%s", uid[:6], size, u.n_chunks, slot)
            return self._ack(u)

    def expected_len(self, u: Upload, index: int) -> int:
        return u.chunk_size if index < u.n_chunks - 1 else u.size - (u.n_chunks - 1) * u.chunk_size

    def put_chunk(self, owner: str, upload_id: str, index, data: bytes, digest_hex) -> dict:
        t0 = time.perf_counter()
        with self._lock:
            u = self._get(owner, upload_id)
            if u.state != UPLOADING:
                raise UploadError("bad_state", "This upload is not accepting chunks.")
            now = self._clock()
            if now - u.t_activity > self.limits.idle_timeout_s or now - u.t_created > self.limits.max_upload_s:
                self._delete(u)
                raise UploadError("timeout", "The upload timed out. Please select the file again.")
            if not (isinstance(index, int) and not isinstance(index, bool)) or not 0 <= index < u.n_chunks:
                self._reject(u, "bad_index"); raise UploadError("bad_index", f"Chunk index out of range (0..{u.n_chunks - 1}).")
            if not isinstance(data, (bytes, bytearray)) or len(data) > self.limits.max_chunk_bytes:
                self._reject(u, "bad_length"); raise UploadError("bad_length", "Invalid chunk payload.")
            if not isinstance(digest_hex, str) or not _HEX_RE.match(digest_hex):
                self._reject(u, "bad_digest"); raise UploadError("bad_digest", "Missing or malformed chunk checksum.")
            if index < u.next_index:                                  # duplicate: idempotent only if it is the SAME data
                if hashlib.sha256(data).hexdigest() == u.chunk_digests[index] == digest_hex:
                    u.duplicates += 1; u.t_activity = now
                    return self._ack(u, duplicate=True)
                self._reject(u, "duplicate_mismatch")
                raise UploadError("duplicate_mismatch", "A chunk with this index was already received with different content.", expected=u.next_index)
            if index > u.next_index:
                self._reject(u, "out_of_order")
                raise UploadError("out_of_order", f"Chunk {u.next_index} is missing; received {index}.", expected=u.next_index)
            if len(data) != self.expected_len(u, index):
                self._reject(u, "bad_length")
                raise UploadError("bad_length", f"Chunk {index} has {len(data)} bytes; expected {self.expected_len(u, index)}.", expected=u.next_index)
            actual = hashlib.sha256(data).hexdigest()
            if actual != digest_hex:
                self._reject(u, "corrupt_chunk")
                raise UploadError("corrupt_chunk", f"Chunk {index} failed its checksum; it can be sent again.", expected=u.next_index)
            offset = index * u.chunk_size
            if self._free_bytes() - len(data) < self.limits.chunk_floor_bytes:      # the disk filled up (another process, or a long upload): stop before ENOSPC
                self._fail(u); raise UploadError("disk_full", "The server ran out of temporary storage. Please try again later.")
            try:
                with open(u.path, "ab") as fh:                        # strictly sequential append; position is verified
                    if fh.tell() != offset:
                        self._fail(u); raise UploadError("internal", "Temporary file out of sync; please restart the upload.")
                    fh.write(data)
            except OSError as e:
                self._fail(u)
                raise UploadError("disk_full" if getattr(e, "errno", None) in (errno.ENOSPC, errno.EDQUOT) else "internal",
                                  "The server could not write the file (storage problem). Please try again later.")
            u.hasher.update(data); u.chunk_digests.append(actual)
            u.next_index += 1; u.bytes_received += len(data); u.t_activity = now
            if u.t_first_chunk is None:
                u.t_first_chunk = now
            u.server_ms.append((time.perf_counter() - t0) * 1000.0)
            return self._ack(u)

    def finish(self, owner: str, upload_id: str, file_sha256, client_stats=None) -> dict:
        with self._lock:
            u = self._get(owner, upload_id)
            if u.state == COMPLETE and u.sha256 == file_sha256:        # idempotent
                return self._ack(u)
            if u.state != UPLOADING:
                raise UploadError("bad_state", "This upload cannot be finished.")
            if u.next_index != u.n_chunks:
                raise UploadError("incomplete", f"{u.n_chunks - u.next_index} chunk(s) are missing.", expected=u.next_index)
            if not isinstance(file_sha256, str) or not _HEX_RE.match(file_sha256):
                raise UploadError("bad_digest", "Missing or malformed file checksum.")
            on_disk = u.path.stat().st_size
            if on_disk != u.size:
                self._fail(u); raise UploadError("size_mismatch", f"Assembled file has {on_disk} bytes; expected {u.size}.")
            h = hashlib.sha256()                                      # re-read from DISK in 1 MiB blocks (bounded memory)
            with open(u.path, "rb") as fh:
                for block in iter(lambda: fh.read(MiB), b""):
                    h.update(block)
            disk_hex, running_hex = h.hexdigest(), u.hasher.hexdigest()
            if not (disk_hex == running_hex == file_sha256):
                self._fail(u)
                raise UploadError("hash_mismatch", "The assembled file does not match the checksum of the original file. Please upload again.",
                                  server_disk=disk_hex[:12], server_running=running_hex[:12], client=file_sha256[:12])
            if u.slot:                                                # the two cameras must not receive the same recording
                other = next((o for o in self._uploads.values() if o.owner == owner and o.slot and o.slot != u.slot and o.state == COMPLETE), None)
                if other is not None and other.sha256 == disk_hex:
                    self._fail(u)
                    raise UploadError("duplicate_file", f"The {u.slot.upper()} and {other.slot.upper()} videos are the same file. "
                                                        "Each camera must have its own recording. Choose the correct file.", other_slot=other.slot)
            u.sha256, u.state, u.t_complete = disk_hex, COMPLETE, self._clock()
            u.t_activity = u.t_complete
            u.client_stats = client_stats if isinstance(client_stats, dict) else None
            os.chmod(u.path, 0o600)
            final = u.directory / "data.bin"
            os.replace(u.path, final); u.path = final                 # name carries no visitor input
            log.info("upload %s complete (%d bytes)", u.upload_id[:6], u.size)
            return self._ack(u)

    def cancel(self, owner: str, upload_id: str) -> dict:
        with self._lock:
            u = self._get(owner, upload_id)
            self._delete(u)
            return dict(ok=True, cancelled=True, upload_id=upload_id)

    def release_owner(self, owner: str) -> int:
        """Delete every upload that belongs to this session (used when the session locks itself). A running calibration is cancelled first
        (its worker is killed); files a process still holds are left for the next sweep. Returns how many were removed."""
        with self._lock:
            for lease in [l for l in self._leases.values() if l.owner == owner and l.state in ("active", "expiring")]:
                lease.cancel_requested = True; lease.state = "expiring"
                self._kill_worker(lease)
                self._try_release(lease, "cancelled", True)
            mine = [u for u in self._uploads.values() if u.owner == owner]
            removed = 0
            for u in mine:
                try:
                    self._delete(u); removed += 1
                except UploadError:
                    pass                                              # still in use: the sweeper retries
            return removed

    release = cancel                                                   # delete an assembled file when the visitor is done with it

    def path_of(self, owner: str, upload_id: str) -> Path:
        """Path of a COMPLETE upload, for server-side processing only (read it; never write)."""
        with self._lock:
            u = self._get(owner, upload_id)
            if u.state != COMPLETE:
                raise UploadError("bad_state", "The upload is not complete.")
            return u.path

    def status(self, owner: str, upload_id: str) -> dict:
        with self._lock:
            return self._ack(self._get(owner, upload_id))

    # ------------------------------------------------------------ calibration job: slots and lease
    def job_status(self, owner: str) -> dict:
        """What this session holds: per slot the upload state, plus its lease. Safe to call on every rerun."""
        with self._lock:
            out = dict(slots={}, lease=None)
            for u in self._uploads.values():
                if u.owner == owner and u.slot and u.state != FAILED:      # a failed upload has no file; it is only waiting to be swept
                    out["slots"][u.slot] = dict(upload_id=u.upload_id, state=u.state, size=u.size, bytes_received=u.bytes_received, n_chunks=u.n_chunks,
                                                next_index=u.next_index, sha256=u.sha256, leased=u.leased_by is not None, name=u.display_name)
            for l in self._leases.values():
                if l.owner == owner and l.state in ("active", "expiring"):
                    now = self._clock()
                    out["lease"] = dict(lease_id=l.lease_id, state=l.state, cancel_requested=l.cancel_requested,
                                        heartbeat_age_s=round(now - l.t_heartbeat, 1), hard_remaining_s=round(l.hard_deadline - now, 1))
            return out

    def _lease(self, owner: str, lease_id) -> Lease:
        l = self._leases.get(lease_id) if isinstance(lease_id, str) else None
        if l is None or not secrets.compare_digest(l.owner.encode(), str(owner).encode()):
            raise UploadError("not_found", "This calibration job does not exist (it may have expired).")
        return l

    def acquire_lease(self, owner: str) -> dict:
        """Claim BOTH videos for a calibration job. Needs a completed LEFT and RIGHT upload of this session, different files, intact on disk.
        Returns the lease id and the two verified paths (valid only while the lease is active)."""
        with self._lock:
            self.sweep()
            mine = {u.slot: u for u in self._uploads.values() if u.owner == owner and u.slot}
            missing = [s for s in SLOTS if s not in mine or mine[s].state != COMPLETE]
            if missing:
                raise UploadError("job_incomplete", "Both the LEFT and the RIGHT video must be uploaded and verified first (missing: " + ", ".join(m.upper() for m in missing) + ").")
            if any(u.leased_by for u in mine.values()):
                raise UploadError("job_running", "A calibration is already running for these videos.")
            if sum(1 for l in self._leases.values() if l.state in ("active", "expiring")) >= self.limits.max_active_leases:
                raise UploadError("busy", "Another calibration is running. Please try again when it has finished.")
            for s, u in mine.items():                                  # stale-reference check: the file must still be exactly what was verified
                try:
                    on_disk = u.path.stat().st_size
                except OSError:
                    raise UploadError("job_incomplete", f"The {s.upper()} video file is no longer available. Please upload it again.")
                if on_disk != u.size:
                    raise UploadError("job_incomplete", f"The {s.upper()} video file changed on disk. Please upload it again.")
            if mine["left"].sha256 == mine["right"].sha256:
                raise UploadError("duplicate_file", "The LEFT and RIGHT videos are the same file. Each camera must have its own recording.")
            now = self._clock()
            lease = Lease(secrets.token_urlsafe(18), owner, {s: u.upload_id for s, u in mine.items()}, now, now, now + self.limits.lease_max_s)
            self._leases[lease.lease_id] = lease
            for u in mine.values():
                u.leased_by = lease.lease_id; u.t_activity = now
            info = {s: dict(path=str(u.path), sha256=u.sha256, size=u.size, upload_id=u.upload_id) for s, u in mine.items()}
            log.info("lease %s acquired", lease.lease_id[:6])
            return dict(ok=True, lease_id=lease.lease_id, left=info["left"], right=info["right"], heartbeat_timeout_s=self.limits.lease_heartbeat_timeout_s,
                        max_s=self.limits.lease_max_s)

    def heartbeat(self, owner: str, lease_id: str) -> dict:
        """Extend an active lease. Raises 'lease_expired' once it is no longer active. ``cancel_requested`` tells the supervisor to stop the worker."""
        with self._lock:
            l = self._lease(owner, lease_id)
            now = self._clock()
            if l.state != "active" or now > l.hard_deadline:
                raise UploadError("lease_expired", "This calibration job has expired.")
            l.t_heartbeat = now
            return dict(ok=True, cancel_requested=l.cancel_requested, hard_remaining_s=l.hard_deadline - now)

    def attach_process(self, owner: str, lease_id: str, pid) -> None:
        """Register (or clear, with None) the worker so an expiring lease can stop it before deleting its files."""
        with self._lock:
            l = self._lease(owner, lease_id)
            l.worker_pid = int(pid) if isinstance(pid, int) and not isinstance(pid, bool) and pid > 1 else None

    def request_cancel(self, owner: str, lease_id: str) -> dict:
        with self._lock:
            l = self._lease(owner, lease_id)
            l.cancel_requested = True
            return dict(ok=True)

    def release_lease(self, owner: str, lease_id: str, outcome: str = "done", delete_files: bool = True) -> dict:
        """End a job (success, failure or cancellation) and normally delete both videos. Refuses ('in_use') while the worker still has a file open."""
        with self._lock:
            l = self._lease(owner, lease_id)
            if l.state in ("released", "expired"):
                return dict(ok=True, already=True, outcome=l.outcome)
            if not self._try_release(l, str(outcome)[:24], bool(delete_files)):
                raise UploadError("in_use", "The calibration worker is still running; stop it before releasing the videos.")
            return dict(ok=True, outcome=l.outcome, deleted=bool(delete_files))

    # ------------------------------------------------------------ internals
    def _reject(self, u: Upload, code: str):
        u.rejected[code] = u.rejected.get(code, 0) + 1

    def _fail(self, u: Upload):
        u.state = FAILED
        shutil.rmtree(u.directory, ignore_errors=True)                 # drop the bad file now; the registry entry is swept

    def _ack(self, u: Upload, **extra) -> dict:
        return dict(ok=True, upload_id=u.upload_id, state=u.state, next_index=u.next_index, n_chunks=u.n_chunks, size=u.size,
                    chunk_size=u.chunk_size, bytes_received=u.bytes_received, sha256=u.sha256, slot=u.slot, **extra)

    def metrics(self, owner: str, upload_id: str) -> dict:
        with self._lock:
            u = self._get(owner, upload_id)
            ms = sorted(u.server_ms)
            dur = (u.t_complete - u.t_first_chunk) if (u.t_complete and u.t_first_chunk) else None
            return dict(upload_id_prefix=u.upload_id[:6], size_bytes=u.size, chunk_size=u.chunk_size, n_chunks=u.n_chunks, chunks_received=u.next_index,
                        duplicates=u.duplicates, rejected=dict(u.rejected), state=u.state,
                        server_upload_s=dur, server_mb_per_s=(u.size / MiB / dur) if dur else None,
                        server_chunk_ms=dict(mean=(sum(ms) / len(ms)) if ms else None, p95=ms[int(0.95 * (len(ms) - 1))] if ms else None, max=ms[-1] if ms else None),
                        client_stats=u.client_stats, sha256=u.sha256, slot=u.slot)

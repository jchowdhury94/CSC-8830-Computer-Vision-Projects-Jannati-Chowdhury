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
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

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


class ChunkStore:
    def __init__(self, limits: Limits | None = None, root: str | os.PathLike | None = None, clock=time.monotonic, purge_orphans: bool = True,
                 disk_usage=shutil.disk_usage):
        self.limits = limits or Limits()
        self._clock = clock
        self._disk_usage = disk_usage                                 # injectable for tests
        self._lock = threading.RLock()
        self._uploads: dict[str, Upload] = {}
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
        """Delete every upload and, if this store created it, the root directory."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for u in list(self._uploads.values()):
                self._delete(u)
            if self._owns_root:
                shutil.rmtree(self.root, ignore_errors=True)

    def _delete(self, u: Upload):
        shutil.rmtree(u.directory, ignore_errors=True)
        self._uploads.pop(u.upload_id, None)

    def sweep(self) -> list:
        """Delete expired uploads (idle / over-long / retained too long). Returns the ids removed."""
        now = self._clock(); gone = []
        with self._lock:
            for u in list(self._uploads.values()):
                if u.state == UPLOADING and (now - u.t_activity > self.limits.idle_timeout_s or now - u.t_created > self.limits.max_upload_s):
                    gone.append(u.upload_id)
                elif u.state == COMPLETE and now - (u.t_complete or u.t_created) > self.limits.complete_retention_s:
                    gone.append(u.upload_id)
                elif u.state == FAILED:
                    gone.append(u.upload_id)
            for uid in gone:
                self._delete(self._uploads[uid])
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
    def begin(self, owner: str, name, size, chunk_size, fingerprint="") -> dict:
        lim = self.limits
        if not isinstance(owner, str) or not owner:
            raise UploadError("bad_request", "Missing session.")
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
            for u in mine:                                            # resume: same file, same chunking, still uploading
                if u.state == UPLOADING and u.fingerprint == fingerprint and u.size == size and u.chunk_size == chunk_size and fingerprint:
                    u.t_activity = self._clock()
                    return self._ack(u, resumed=True)
            for u in mine:                                            # a session keeps at most one upload: starting another discards the old one
                self._delete(u)
            if self.active_count() >= lim.max_active_uploads:          # take over stalled uploads of other sessions (a refreshed page is a NEW session)
                for u in [u for u in self._uploads.values() if u.state == UPLOADING and self._clock() - u.t_activity > lim.preempt_idle_s]:
                    self._delete(u)
            if self.active_count() >= lim.max_active_uploads:
                raise UploadError("busy", "Another upload is in progress. Please try again in a few minutes.")
            if self.reserved_bytes() + size > lim.max_disk_bytes:
                raise UploadError("disk_limit", "The server's temporary storage limit would be exceeded. Please try again later.")
            usable = self.capacity()["usable_bytes"]                   # REAL free space; decided under the same lock for every session
            if size > usable:
                raise UploadError("insufficient_disk", f"The server does not have enough free temporary storage for a {size / MiB:.0f} MB file right now "
                                                       f"({usable / MiB:.0f} MB usable). Try a smaller file or try again later.")
            uid = secrets.token_urlsafe(18)
            d = self.root / uid
            d.mkdir(mode=0o700)
            path = d / "data.part"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.close(fd)
            now = self._clock()
            u = Upload(uid, owner, safe_display_name(name), size, chunk_size, -(-size // chunk_size), fingerprint, d, path, now, now)
            self._uploads[uid] = u
            log.info("upload %s started: %d bytes, %d chunks", uid[:6], size, u.n_chunks)
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
        """Delete every upload that belongs to this session (used when the session locks itself). Returns how many were removed."""
        with self._lock:
            mine = [u for u in self._uploads.values() if u.owner == owner]
            for u in mine:
                self._delete(u)
            return len(mine)

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

    # ------------------------------------------------------------ internals
    def _reject(self, u: Upload, code: str):
        u.rejected[code] = u.rejected.get(code, 0) + 1

    def _fail(self, u: Upload):
        u.state = FAILED
        shutil.rmtree(u.directory, ignore_errors=True)                 # drop the bad file now; the registry entry is swept

    def _ack(self, u: Upload, **extra) -> dict:
        return dict(ok=True, upload_id=u.upload_id, state=u.state, next_index=u.next_index, n_chunks=u.n_chunks, size=u.size,
                    chunk_size=u.chunk_size, bytes_received=u.bytes_received, sha256=u.sha256, **extra)

    def metrics(self, owner: str, upload_id: str) -> dict:
        with self._lock:
            u = self._get(owner, upload_id)
            ms = sorted(u.server_ms)
            dur = (u.t_complete - u.t_first_chunk) if (u.t_complete and u.t_first_chunk) else None
            return dict(upload_id_prefix=u.upload_id[:6], size_bytes=u.size, chunk_size=u.chunk_size, n_chunks=u.n_chunks, chunks_received=u.next_index,
                        duplicates=u.duplicates, rejected=dict(u.rejected), state=u.state,
                        server_upload_s=dur, server_mb_per_s=(u.size / MiB / dur) if dur else None,
                        server_chunk_ms=dict(mean=(sum(ms) / len(ms)) if ms else None, p95=ms[int(0.95 * (len(ms) - 1))] if ms else None, max=ms[-1] if ms else None),
                        client_stats=u.client_stats, sha256=u.sha256)

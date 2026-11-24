"""Stage F8.1: SAFE, read-only inspection of an uploaded video (classical OpenCV decode; no calibration; the file is never modified).

The upload is untrusted. Defence in depth, in this order:
  1. SIGNATURE   the first bytes must be a supported container (MP4/MOV/M4V, Matroska/WebM, AVI); the extension is never used.
  2. HEADER      for MP4/MOV the movie header is parsed in pure Python (bounded reads, no OpenCV): the declared frame size is checked against the
                 limits BEFORE any decoder sees the file. Metadata is not trusted blindly: step 3 checks it again.
  3. WORKER      OpenCV runs in a SEPARATE PROCESS (isolated interpreter, scrubbed environment, own process group) with a wall-clock timeout,
                 CPU-time and (Linux) address-space caps. The worker re-checks the size OpenCV reports before reading a frame and the size of the
                 first decoded frame. On timeout the whole process group is killed.
The parent process never imports OpenCV, so a hostile file cannot crash or stall the Streamlit server.
"""
from __future__ import annotations

import json
import os
import signal
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_SAMPLE_FRAMES = 10
MiB = 1024 * 1024


@dataclass(frozen=True)
class VideoSafetyLimits:
    max_side_px: int = 8192                  # 8K DCI is 8192 px wide; iPhone 4K is 3840
    max_pixels: int = 40_000_000             # same per-frame cap as the main app's Stage B (max_frame_pixels)
    timeout_s: float = 20.0                  # wall clock for the whole worker
    worker_extra_memory_mb: int = 2048       # Linux: address space allowed on top of what the worker already uses after importing OpenCV
    max_moov_bytes: int = 64 * MiB           # largest movie header that will be read into memory
    max_boxes: int = 20000                   # total MP4 boxes visited (guards against crafted nesting)


DEFAULT_LIMITS = VideoSafetyLimits()

# image/audio-only ISO-BMFF brands: same 'ftyp' signature, but not a video
NON_VIDEO_BRANDS = {b"heic", b"heix", b"hevx", b"heim", b"heis", b"mif1", b"msf1", b"avif", b"avis", b"crx ", b"M4A ", b"M4B ", b"M4P "}
LEGACY_QT_ATOMS = {b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}
HEVC_CODES = {"hvc1", "hev1", "dvh1", "dvhe"}
VIDEO_CODES_NOTE = {"hvc1": "HEVC/H.265", "hev1": "HEVC/H.265", "dvh1": "Dolby Vision HEVC", "dvhe": "Dolby Vision HEVC"}


class HeaderError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


def _err(code, message): return dict(code=code, message=message)


# ---------------------------------------------------------------- 1. signature
def sniff_container(head: bytes):
    """Returns (container, detail) for a supported signature, else (None, reason). Looks only at the first bytes."""
    if len(head) < 12:
        return None, "The file is too short to be a video."
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in NON_VIDEO_BRANDS:
            return None, f"This is an ISO media file with brand '{brand.decode('latin-1')}' (an image or audio file), not a video."
        return ("quicktime" if brand == b"qt  " else "mp4"), brand.decode("latin-1")
    size = struct.unpack(">I", head[:4])[0]
    if head[4:8] in LEGACY_QT_ATOMS and (size == 0 or size == 1 or size >= 8):
        return "quicktime", head[4:8].decode("latin-1")
    if head[:4] == b"\x1aE\xdf\xa3":
        return "matroska", "ebml"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "avi", "riff"
    return None, "Unsupported file type: the file does not start with an MP4/MOV, Matroska/WebM or AVI signature."


# ---------------------------------------------------------------- 2. MP4/MOV header (pure Python, bounded)
class _Walk:
    def __init__(self, limits): self.n = 0; self.limits = limits

    def children(self, buf, start, end, depth=0):
        if depth > 8:
            raise HeaderError("bad_header", "The movie header is nested too deeply.")
        pos = start
        while pos + 8 <= end:
            self.n += 1
            if self.n > self.limits.max_boxes:
                raise HeaderError("bad_header", "The movie header has too many boxes.")
            size, typ = struct.unpack_from(">I4s", buf, pos); hdr = 8
            if size == 1:
                if pos + 16 > end: raise HeaderError("bad_header", "Truncated box header.")
                size = struct.unpack_from(">Q", buf, pos + 8)[0]; hdr = 16
            elif size == 0:
                size = end - pos
            if size < hdr or pos + size > end:
                raise HeaderError("bad_header", "A box in the movie header has an invalid size.")
            yield typ, pos + hdr, pos + size
            pos += size


def _find(walk, buf, start, end, typ):
    return [(s, e) for t, s, e in walk.children(buf, start, end) if t == typ]


def parse_iso_video_header(path, limits: VideoSafetyLimits = DEFAULT_LIMITS) -> dict:
    """Declared size / codec of the largest video track of an MP4/MOV, using bounded reads. Raises HeaderError."""
    file_size = os.path.getsize(path); pos = 0; n = 0; moov = None
    with open(path, "rb") as fh:
        while pos + 8 <= file_size and n < 4096:
            n += 1; fh.seek(pos); h = fh.read(16)
            if len(h) < 8: break
            size, typ = struct.unpack(">I4s", h[:8]); hdr = 8
            if size == 1:
                if len(h) < 16: break
                size = struct.unpack(">Q", h[8:16])[0]; hdr = 16
            elif size == 0:
                size = file_size - pos
            if size < hdr: raise HeaderError("bad_header", "The file has a malformed top-level box.")
            if typ == b"moov":
                if size - hdr > limits.max_moov_bytes: raise HeaderError("bad_header", "The movie header is unreasonably large.")
                if pos + size > file_size: raise HeaderError("bad_header", "The movie header is truncated (the upload may be incomplete).")
                fh.seek(pos + hdr); moov = fh.read(size - hdr); break
            pos += size
    if moov is None:
        raise HeaderError("no_movie_header", "No movie header (moov) was found, so the video track cannot be described.")
    walk = _Walk(limits); best = None
    try:
        for ts, te in _find(walk, moov, 0, len(moov), b"trak"):
            kids = {t: (s, e) for t, s, e in walk.children(moov, ts, te)}
            if b"mdia" not in kids: continue
            ms, me = kids[b"mdia"]; mk = {t: (s, e) for t, s, e in walk.children(moov, ms, me)}
            if b"hdlr" not in mk or b"minf" not in mk: continue
            hs = mk[b"hdlr"][0]
            if moov[hs + 8:hs + 12] != b"vide": continue
            fs, fe = mk[b"minf"]; stbl = next(((s, e) for t, s, e in walk.children(moov, fs, fe) if t == b"stbl"), None)
            if stbl is None: continue
            stsd = next(((s, e) for t, s, e in walk.children(moov, *stbl) if t == b"stsd"), None)
            if stsd is None: continue
            es = stsd[0] + 8                                                   # after version/flags + entry_count
            if es + 36 > stsd[1]: raise HeaderError("bad_header", "The video sample description is truncated.")
            codec = moov[es + 4:es + 8].decode("latin-1")
            w, h2 = struct.unpack_from(">HH", moov, es + 32)
            tw = th = 0
            if b"tkhd" in kids:
                s = kids[b"tkhd"][0]; ver = moov[s]; off = s + (88 if ver == 1 else 76)
                if off + 8 <= kids[b"tkhd"][1]:
                    tw, th = (v >> 16 for v in struct.unpack_from(">II", moov, off))
            cand = dict(width=max(w, tw), height=max(h2, th), stsd_wh=(w, h2), tkhd_wh=(tw, th), codec=codec if codec.isprintable() else None)
            if best is None or cand["width"] * cand["height"] > best["width"] * best["height"]: best = cand
    except (struct.error, IndexError) as e:
        raise HeaderError("bad_header", "The movie header could not be parsed.") from e
    if best is None:
        raise HeaderError("no_video_track", "The file contains no video track.")
    if best["width"] <= 0 or best["height"] <= 0:
        raise HeaderError("bad_dimensions", "The video header declares a zero frame size.")
    return best


def _check_dims(w, h, limits, source):
    if w > limits.max_side_px or h > limits.max_side_px:
        return _err("dimensions_too_large", f"The video's frame size ({w}×{h}, from the {source}) exceeds the {limits.max_side_px} px side limit.")
    if w * h > limits.max_pixels:
        return _err("pixels_too_large", f"The video's frame size ({w}×{h} = {w * h / 1e6:.1f} megapixels, from the {source}) exceeds the {limits.max_pixels / 1e6:.0f} megapixel limit.")
    return None


# ---------------------------------------------------------------- 3. worker process
def _fourcc(code):
    try:
        c = int(code); s = "".join(chr((c >> (8 * i)) & 0xFF) for i in range(4))
        return s if s.strip("\x00") and s.isascii() and s.isprintable() else None
    except (TypeError, ValueError):
        return None


def _worker_main(path, limits_json, max_frames):
    """Runs in the subprocess. Prints exactly one JSON line. Never writes files."""
    lim = VideoSafetyLimits(**json.loads(limits_json)); out = dict(opened=False)
    try:
        import resource
        cpu = int(lim.timeout_s) + 5; resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu)); resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        resource = None
    import cv2
    if resource is not None and sys.platform.startswith("linux"):           # address-space cap = what is mapped after the import + headroom
        try:
            with open("/proc/self/status") as fh:
                vm = next(int(l.split()[1]) for l in fh if l.startswith("VmSize:")) * 1024
            cap = vm + lim.worker_extra_memory_mb * MiB; resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except Exception:
            pass
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            print(json.dumps(out)); return 0
        out["opened"] = True
        try: out["backend"] = str(cap.getBackendName())[:20]
        except Exception: pass
        pw, ph = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        out["props_wh"] = [pw, ph]
        if pw > lim.max_side_px or ph > lim.max_side_px or pw * ph > lim.max_pixels:    # BEFORE the first frame is decoded
            out["rejected"] = "props_dimensions"; print(json.dumps(out)); return 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out["fps"] = fps if 0 < fps < 10000 else None; out["frame_count"] = n if 0 < n < 10**9 else None; out["codec"] = _fourcc(cap.get(cv2.CAP_PROP_FOURCC))
        t0 = time.perf_counter(); read = 0; shape = None
        for _ in range(max(1, int(max_frames))):
            ok, frame = cap.read()
            if not ok or frame is None: break
            if shape is None:
                shape = frame.shape
                if shape[1] > lim.max_side_px or shape[0] > lim.max_side_px or shape[0] * shape[1] > lim.max_pixels:
                    out["rejected"] = "decoded_dimensions"; out["decoded_wh"] = [int(shape[1]), int(shape[0])]; print(json.dumps(out)); return 0
            read += 1
        dt = time.perf_counter() - t0
        out["frames_read"] = read; out["sample_decode_ms"] = round(dt * 1000.0, 1)
        if shape is not None:
            out["decoded_wh"] = [int(shape[1]), int(shape[0])]; out["sample_fps"] = round(read / dt, 1) if dt > 0 else None
        print(json.dumps(out)); return 0
    finally:
        cap.release()


# ---------------------------------------------------------------- public API
_ENV_KEEP = ("PATH", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH")


def _scrubbed_env():
    """The worker needs nothing secret (e.g. the passcode): pass only a few harmless variables."""
    return {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}


def _empty():
    return dict(ok=False, opened=False, decodable=False, width=None, height=None, fps=None, frame_count=None, duration_s=None, codec=None, backend=None,
                frames_read=0, sample_decode_ms=None, sample_fps=None, container=None, header=None, stage=None, worker_s=None, errors=[], issues=[])


def _finish(out):
    out["ok"] = bool(out["decodable"]) and not out["errors"]
    out["issues"] = [e["message"] for e in out["errors"]] + out["issues"]       # errors first, then warnings (kept for older callers)
    return out


def _run_worker(path, max_frames, limits, worker_cmd):
    cmd = list(worker_cmd) if worker_cmd else [sys.executable, "-I", str(Path(__file__).resolve()), "--worker", str(path), json.dumps(asdict(limits)), str(int(max_frames))]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_scrubbed_env(), start_new_session=True)
    try:
        stdout, _ = proc.communicate(timeout=limits.timeout_s)
        return proc.returncode, stdout, time.perf_counter() - t0, False
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid, signal.SIGKILL)                              # the worker and anything it started
        except (ProcessLookupError, PermissionError): proc.kill()
        try: proc.communicate(timeout=5)                                       # reap
        except Exception: pass
        return None, b"", time.perf_counter() - t0, True


def inspect_video(path, max_frames: int = MAX_SAMPLE_FRAMES, limits: VideoSafetyLimits = DEFAULT_LIMITS, worker_cmd=None) -> dict:
    """Validate and inspect ``path``. Never raises for bad input. ``worker_cmd`` replaces the worker command (test hook only).
    Result keys: ok, opened, decodable, width, height, fps, frame_count, duration_s, codec, frames_read, errors [{code, message}], issues (messages), stage."""
    out = _empty(); path = Path(path)
    try:
        if not path.is_file() or path.is_symlink():
            out["errors"].append(_err("not_a_file", "The file does not exist.")); out["stage"] = "signature"; return _finish(out)
        with open(path, "rb") as fh:
            head = fh.read(64)
    except OSError:
        out["errors"].append(_err("unreadable", "The file could not be read.")); out["stage"] = "signature"; return _finish(out)

    container, detail = sniff_container(head); out["stage"] = "signature"
    if container is None:
        out["errors"].append(_err("unsupported_signature", detail)); return _finish(out)
    out["container"] = container

    hdr = None
    if container in ("mp4", "quicktime"):
        out["stage"] = "header"
        try:
            hdr = parse_iso_video_header(path, limits)
        except HeaderError as e:
            out["errors"].append(_err(e.code, e.message)); return _finish(out)
        except OSError:
            out["errors"].append(_err("unreadable", "The file could not be read.")); return _finish(out)
        out["header"] = dict(width=hdr["width"], height=hdr["height"], codec=hdr["codec"])
        bad = _check_dims(hdr["width"], hdr["height"], limits, "movie header")
        if bad:
            out["errors"].append(bad); return _finish(out)                    # rejected BEFORE any decoder runs
        if hdr["codec"] in VIDEO_CODES_NOTE:
            out["issues"].append(f"{VIDEO_CODES_NOTE[hdr['codec']]} video: it is accepted, but decoding depends on the OpenCV build of the host.")

    out["stage"] = "worker"
    try:
        rc, stdout, dt, timed_out = _run_worker(path, max_frames, limits, worker_cmd)
    except OSError:
        out["errors"].append(_err("worker_failed", "The inspection process could not be started.")); return _finish(out)
    out["worker_s"] = round(dt, 2)
    if timed_out:
        out["errors"].append(_err("timeout", f"Inspecting the video took longer than {limits.timeout_s:g} s and was stopped.")); return _finish(out)
    if rc != 0:
        out["errors"].append(_err("worker_failed", "The video decoder stopped unexpectedly while inspecting this file (it may be corrupt).")); return _finish(out)
    try:
        if len(stdout) > 65536: raise ValueError
        w = json.loads(stdout.decode("utf-8", "replace").strip().splitlines()[-1])
        if not isinstance(w, dict): raise ValueError
    except (ValueError, IndexError):
        out["errors"].append(_err("worker_failed", "The inspection process returned an unreadable result.")); return _finish(out)

    out["backend"] = w.get("backend") if isinstance(w.get("backend"), str) else None
    if w.get("rejected") in ("props_dimensions", "decoded_dimensions"):
        wh = w.get("decoded_wh") or w.get("props_wh") or [0, 0]
        out["errors"].append(_check_dims(int(wh[0]), int(wh[1]), limits, "decoder") or _err("dimensions_too_large", "The decoded frame size exceeds the limits."))
        return _finish(out)
    if not w.get("opened"):
        hevc = hdr is not None and hdr["codec"] in HEVC_CODES
        out["errors"].append(_err("cannot_open", "OpenCV could not open this file" + (" (it is HEVC/H.265, which this OpenCV build cannot decode; re-encode to H.264)." if hevc else " (corrupt, or a codec this OpenCV build cannot decode).")))
        return _finish(out)
    out["opened"] = True
    num = lambda k, lo, hi: (float(w[k]) if isinstance(w.get(k), (int, float)) and not isinstance(w.get(k), bool) and lo <= w[k] <= hi else None)
    out["fps"] = num("fps", 0.001, 10000); fc = num("frame_count", 1, 10 ** 9); out["frame_count"] = int(fc) if fc else None
    out["codec"] = w["codec"] if isinstance(w.get("codec"), str) and len(w["codec"]) <= 4 and w["codec"].isascii() and w["codec"].isprintable() else None
    out["frames_read"] = int(w["frames_read"]) if isinstance(w.get("frames_read"), int) and 0 <= w["frames_read"] <= max(1, max_frames) else 0
    out["sample_decode_ms"] = num("sample_decode_ms", 0, 1e9); out["sample_fps"] = num("sample_fps", 0, 1e9)
    dwh = w.get("decoded_wh")
    if out["frames_read"] and isinstance(dwh, list) and len(dwh) == 2 and all(isinstance(v, int) and v > 0 for v in dwh):
        bad = _check_dims(dwh[0], dwh[1], limits, "decoded frame")
        if bad:
            out["errors"].append(bad); return _finish(out)
        out["width"], out["height"], out["decodable"] = dwh[0], dwh[1], True
        if hdr and sorted(dwh) != sorted([hdr["width"], hdr["height"]]):         # equal or rotated is normal; anything else is suspicious
            out["issues"].append(f"The decoded frame size ({dwh[0]}×{dwh[1]}) differs from the size in the movie header ({hdr['width']}×{hdr['height']}).")
    else:
        out["errors"].append(_err("no_frame", "The file opened but no frame could be decoded."))
    if out["fps"] and out["frame_count"]:
        out["duration_s"] = round(out["frame_count"] / out["fps"], 2)
    if out["frame_count"] is None:
        out["issues"].append("The container does not report a frame count.")
    if hdr is None and out["codec"] in HEVC_CODES:
        out["issues"].append("HEVC/H.265 video: it decoded here, but decoder support depends on the OpenCV build of the host.")
    return _finish(out)


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        sys.exit(_worker_main(sys.argv[2], sys.argv[3], int(sys.argv[4])))
    print("This module is imported by app.py; the worker mode is internal.", file=sys.stderr); sys.exit(2)

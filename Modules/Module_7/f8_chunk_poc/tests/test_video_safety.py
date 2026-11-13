"""F8.1: safe video inspection (signature, header dimensions, subprocess timeout/failure). Synthetic files only; no existing fixture is used."""
import os
import shutil
import struct
import sys
import time

import pytest

import video_check as vc
from video_check import VideoSafetyLimits, inspect_video, parse_iso_video_header, sniff_container
from make_synthetic import make_video


def codes(r): return [e["code"] for e in r["errors"]]


@pytest.fixture(scope="module")
def good_mp4(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("vid") / "good.mp4", seconds=2, w=320, h=180, fps=25)


@pytest.fixture
def no_worker(monkeypatch):
    """Fails the test if the decoder process is started: used to prove a file was rejected BEFORE any decoding."""
    def boom(*a, **k): raise AssertionError("the decoder worker must not be started for this file")
    monkeypatch.setattr(vc, "_run_worker", boom)


def patch_dims(src, dst, stsd_wh=None, tkhd_wh=None):
    """Copy ``src`` and overwrite the declared frame size in the sample description (stsd) and/or track header (tkhd)."""
    b = bytearray(open(src, "rb").read())
    if stsd_wh:
        t = b.index(b"stsd"); struct.pack_into(">HH", b, t + 44, *[min(v, 65535) for v in stsd_wh])
    if tkhd_wh:
        t = b.index(b"tkhd"); s = t + 4; off = s + (88 if b[s] == 1 else 76); struct.pack_into(">II", b, off, tkhd_wh[0] << 16, tkhd_wh[1] << 16)
    open(dst, "wb").write(bytes(b)); return dst


# ---------------------------------------------------------------- signatures
@pytest.mark.parametrize("name,payload", [
    ("random.mp4", os.urandom(4096)),                                             # right extension, wrong content
    ("image.mp4", b"\x89PNG\r\n\x1a\n" + os.urandom(2000)),
    ("photo.mov", b"\xff\xd8\xff\xe0" + os.urandom(2000)),
    ("zip.mp4", b"PK\x03\x04" + os.urandom(2000)),
    ("text.mp4", b"hello world, this is not a video at all" * 20),
    ("pdf.mov", b"%PDF-1.7\n" + os.urandom(2000)),
    ("heic.mov", struct.pack(">I", 24) + b"ftypheic" + b"\0" * 12 + os.urandom(100)),   # ISO media, but an IMAGE brand
    ("audio.mp4", struct.pack(">I", 24) + b"ftypM4A " + b"\0" * 12 + os.urandom(100)),
    ("short.mp4", b"abc"),
    ("empty.mp4", b""),
])
def test_unsupported_signatures_rejected_before_decoding(tmp_path, no_worker, name, payload):
    p = tmp_path / name; p.write_bytes(payload)
    r = inspect_video(p)
    assert not r["ok"] and not r["opened"] and not r["decodable"] and r["stage"] == "signature"
    assert codes(r) == ["unsupported_signature"] and r["issues"] and r["errors"][0]["message"]


def test_missing_and_directory_and_symlink(tmp_path, good_mp4, no_worker):
    assert codes(inspect_video(tmp_path / "nope.mp4")) == ["not_a_file"]
    assert codes(inspect_video(tmp_path)) == ["not_a_file"]
    link = tmp_path / "link.mp4"; os.symlink(good_mp4, link)
    assert codes(inspect_video(link)) == ["not_a_file"]                          # an upload path must never be a symlink


def test_extension_is_never_used(tmp_path, good_mp4):
    renamed = tmp_path / "video_without_extension.txt"; shutil.copy(good_mp4, renamed)
    assert inspect_video(renamed)["ok"]                                          # valid content, odd name: accepted
    assert sniff_container(b"RIFF\0\0\0\0AVI LIST" + b"\0" * 8)[0] == "avi"
    assert sniff_container(b"\x1aE\xdf\xa3" + b"\0" * 12)[0] == "matroska"
    assert sniff_container(struct.pack(">I", 20) + b"ftypqt  " + b"\0" * 8)[0] == "quicktime"
    assert sniff_container(struct.pack(">I", 20) + b"ftypmp42" + b"\0" * 8)[0] == "mp4"
    assert sniff_container(struct.pack(">I", 40) + b"wide" + b"\0" * 12)[0] == "quicktime"   # old QuickTime files without 'ftyp'


# ---------------------------------------------------------------- valid video (incl. iPhone-style MOV)
def test_valid_video_inspection(good_mp4):
    r = inspect_video(good_mp4, max_frames=5)
    assert r["ok"] and r["opened"] and r["decodable"] and r["stage"] == "worker" and not r["errors"]
    assert (r["width"], r["height"]) == (320, 180) and r["frames_read"] == 5 and abs(r["fps"] - 25) < 0.5
    assert r["frame_count"] in range(49, 52) and r["duration_s"] and r["codec"] and r["container"] == "mp4" and r["worker_s"] is not None
    assert r["header"]["width"] == 320 and r["header"]["height"] == 180


def test_iphone_style_mov_brand_and_extension_still_accepted(good_mp4, tmp_path):
    b = bytearray(open(good_mp4, "rb").read()); assert b[4:8] == b"ftyp"
    b[8:12] = b"qt  "                                                            # what iPhone MOV files declare as their major brand
    p = tmp_path / "IMG_0001.MOV"; p.write_bytes(bytes(b))
    r = inspect_video(p)
    assert r["ok"] and r["container"] == "quicktime" and (r["width"], r["height"]) == (320, 180)


def test_hevc_header_is_accepted_with_a_note_not_rejected(good_mp4, tmp_path, monkeypatch):
    """The decoder may or may not support HEVC on the host; the validator must not reject HEVC itself."""
    b = bytearray(open(good_mp4, "rb").read()); t = b.index(b"stsd"); b[t + 16:t + 20] = b"hvc1"
    p = tmp_path / "hevc.mov"; p.write_bytes(bytes(b))
    hdr = parse_iso_video_header(p); assert hdr["codec"] == "hvc1"
    r = inspect_video(p)                                                         # whichever way OpenCV answers, there is no validator error
    assert not [c for c in codes(r) if c in ("unsupported_signature", "bad_header", "dimensions_too_large", "pixels_too_large")]
    assert any("HEVC" in m for m in r["issues"])


def test_rotated_portrait_header_is_not_flagged(good_mp4, tmp_path):
    p = patch_dims(good_mp4, tmp_path / "portrait.mp4", stsd_wh=(180, 320), tkhd_wh=(180, 320))   # header swapped, decoded landscape: a rotated phone clip
    r = inspect_video(p)
    assert r["ok"] and not [m for m in r["issues"] if "differs" in m]


# ---------------------------------------------------------------- dimensions
def test_excessive_dimensions_rejected_before_decoding(good_mp4, tmp_path, no_worker):
    for which, kw in (("stsd", dict(stsd_wh=(60000, 60000))), ("tkhd", dict(tkhd_wh=(60000, 60000))), ("both", dict(stsd_wh=(20000, 100), tkhd_wh=(20000, 100)))):
        r = inspect_video(patch_dims(good_mp4, tmp_path / f"huge_{which}.mp4", **kw))
        assert codes(r) == ["dimensions_too_large"] and r["stage"] == "header" and not r["opened"], which


def test_pixel_budget_enforced_even_when_each_side_is_allowed(good_mp4, tmp_path, no_worker):
    r = inspect_video(patch_dims(good_mp4, tmp_path / "px.mp4", stsd_wh=(8000, 8000), tkhd_wh=(8000, 8000)))   # 64 MP > 40 MP, sides < 8192
    assert codes(r) == ["pixels_too_large"] and "megapixel" in r["errors"][0]["message"]


def test_a_lying_small_stsd_cannot_hide_a_large_tkhd(good_mp4, tmp_path, no_worker):
    r = inspect_video(patch_dims(good_mp4, tmp_path / "lie.mp4", stsd_wh=(320, 180), tkhd_wh=(30000, 30000)))
    assert codes(r) == ["dimensions_too_large"]                                  # the larger of the declared sizes is what is checked


def test_largest_allowed_sizes_pass_the_header_check(good_mp4, tmp_path):
    for wh in ((8192, 4320), (3840, 2160), (7680, 4320)):
        p = patch_dims(good_mp4, tmp_path / f"ok_{wh[0]}.mp4", stsd_wh=wh, tkhd_wh=wh)
        h = parse_iso_video_header(p); assert vc._check_dims(h["width"], h["height"], vc.DEFAULT_LIMITS, "x") is None
    r = inspect_video(patch_dims(good_mp4, tmp_path / "ok4k.mp4", stsd_wh=(3840, 2160), tkhd_wh=(3840, 2160)))
    assert r["ok"] and any("differs" in m for m in r["issues"])                  # header says 4K, the real frames are 320x180: reported, not trusted


def test_metadata_is_rechecked_against_the_decoder(good_mp4, tmp_path, monkeypatch):
    """Header says small, but the decoder reports/decodes something bigger than the limit: the worker (stage 'worker') rejects it."""
    p = patch_dims(good_mp4, tmp_path / "small_claim.mp4", stsd_wh=(100, 100), tkhd_wh=(100, 100))
    r = inspect_video(p, limits=VideoSafetyLimits(max_side_px=200, max_pixels=40000))
    assert not r["ok"] and r["stage"] == "worker" and codes(r)[0] in ("dimensions_too_large", "pixels_too_large")
    assert r["frames_read"] == 0 or not r["decodable"]


def test_worker_checks_properties_before_reading_a_frame(good_mp4, monkeypatch):
    monkeypatch.setattr(vc, "parse_iso_video_header", lambda p, l=None: dict(width=100, height=100, codec="mp4v"))   # skip the pure-Python check
    r = inspect_video(good_mp4, limits=VideoSafetyLimits(max_side_px=200, max_pixels=40000))
    assert codes(r) and r["stage"] == "worker" and not r["decodable"] and r["frames_read"] == 0


# ---------------------------------------------------------------- malformed headers
def test_truncated_file_without_movie_header(good_mp4, tmp_path, no_worker):
    b = open(good_mp4, "rb").read(); p = tmp_path / "trunc.mp4"; p.write_bytes(b[: len(b) // 2])
    r = inspect_video(p); assert codes(r)[0] in ("no_movie_header", "bad_header") and r["stage"] == "header"


def test_crafted_boxes_do_not_crash_the_parser(tmp_path, no_worker):
    ftyp = struct.pack(">I", 20) + b"ftypisom" + b"\0\0\0\0"
    cases = {
        "garbage_moov": ftyp + struct.pack(">I", 8 + 300) + b"moov" + os.urandom(300),
        "huge_moov_size": ftyp + struct.pack(">I", 0x7FFFFFF0) + b"moov" + b"\0" * 64,
        "zero_size_box": ftyp + struct.pack(">I", 0) + b"free" + b"\0" * 64,
        "box_size_below_header": ftyp + struct.pack(">I", 4) + b"moov" + b"\0" * 64,
        "no_tracks": ftyp + struct.pack(">I", 8 + 8) + b"moov" + struct.pack(">I", 8) + b"free",
        "deep_nesting": ftyp + b"".join(struct.pack(">I", 8 + 8 * (60 - i)) + b"trak" for i in range(40)) + b"\0" * 64,
    }
    for name, payload in cases.items():
        p = tmp_path / f"{name}.mp4"; p.write_bytes(payload)
        r = inspect_video(p)
        assert not r["ok"] and r["errors"] and codes(r)[0] in ("bad_header", "no_movie_header", "no_video_track", "bad_dimensions"), (name, codes(r))


def test_header_parser_reads_only_a_bounded_amount(good_mp4, monkeypatch):
    big = vc.VideoSafetyLimits(max_moov_bytes=100)                                # a movie header bigger than this is never read into memory
    with pytest.raises(vc.HeaderError) as e: parse_iso_video_header(good_mp4, big)
    assert e.value.code == "bad_header"


# ---------------------------------------------------------------- subprocess: timeout, failures, isolation
def py(code, *args): return [sys.executable, "-c", code, *map(str, args)]


def test_decode_timeout_terminates_the_worker_and_its_children(good_mp4, tmp_path):
    pidfile = tmp_path / "pids.txt"
    code = ("import os,subprocess,sys,time\n"
            "c = subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])\n"
            "open(sys.argv[1],'w').write(f'{os.getpid()} {c.pid}')\n"
            "time.sleep(120)\n")
    t0 = time.time()
    r = inspect_video(good_mp4, limits=VideoSafetyLimits(timeout_s=1.5), worker_cmd=py(code, pidfile))
    assert codes(r) == ["timeout"] and not r["ok"] and r["stage"] == "worker" and time.time() - t0 < 12
    assert "1.5" in r["errors"][0]["message"]
    for pid in map(int, pidfile.read_text().split()):
        for _ in range(50):
            try: os.kill(pid, 0)
            except ProcessLookupError: break
            time.sleep(0.1)
        else: pytest.fail(f"process {pid} survived the timeout")


@pytest.mark.parametrize("code", ["import sys; sys.exit(3)", "import os,signal; os.kill(os.getpid(), signal.SIGSEGV)", "import os,signal; os.kill(os.getpid(), signal.SIGKILL)",
                                  "raise RuntimeError('boom')"])
def test_worker_crash_or_error_exit_is_reported_cleanly(good_mp4, code):
    r = inspect_video(good_mp4, worker_cmd=py(code))
    assert codes(r) == ["worker_failed"] and not r["ok"] and "stopped unexpectedly" in r["errors"][0]["message"]


@pytest.mark.parametrize("out", [b"", b"not json", b"[1,2,3]", b"{", b"null", b"\x00\xff\xfe"])
def test_worker_garbage_output_is_reported_cleanly(good_mp4, out):
    r = inspect_video(good_mp4, worker_cmd=py("import sys; sys.stdout.buffer.write(bytes.fromhex(sys.argv[1]))", out.hex()))
    assert codes(r) == ["worker_failed"] and "unreadable" in r["errors"][0]["message"]


def test_worker_output_flood_is_rejected(good_mp4):
    r = inspect_video(good_mp4, worker_cmd=py("print('{\"opened\": true, \"pad\": \"' + 'A' * 200000 + '\"}')"))
    assert codes(r) == ["worker_failed"]


def test_worker_that_cannot_start(good_mp4):
    r = inspect_video(good_mp4, worker_cmd=["/nonexistent/definitely/not/here"])
    assert codes(r) == ["worker_failed"] and "could not be started" in r["errors"][0]["message"]


def test_worker_results_are_sanitised(good_mp4):
    evil = ('import json; print(json.dumps(dict(opened=True, codec="X"*500, fps=-5, frame_count=10**15, frames_read=10**9, sample_fps="x", '
            'decoded_wh=[320,180], sample_decode_ms=True, backend=["x"])))')
    r = inspect_video(good_mp4, worker_cmd=py(evil))
    assert r["codec"] is None and r["fps"] is None and r["frame_count"] is None and r["frames_read"] == 0 and r["sample_decode_ms"] is None
    assert codes(r) == ["no_frame"] and r["backend"] is None and not r["ok"]      # a result that does not add up is never trusted


def test_worker_reports_opened_without_frames(good_mp4):
    r = inspect_video(good_mp4, worker_cmd=py("import json; print(json.dumps(dict(opened=True, frames_read=0)))"))
    assert codes(r) == ["no_frame"] and r["opened"] and not r["decodable"]


def test_worker_cannot_open_gives_a_clear_message(good_mp4):
    r = inspect_video(good_mp4, worker_cmd=py("import json; print(json.dumps(dict(opened=False)))"))
    assert codes(r) == ["cannot_open"] and "OpenCV could not open" in r["errors"][0]["message"]


def test_decoded_frame_over_limit_reported_by_worker_is_rejected(good_mp4):
    r = inspect_video(good_mp4, worker_cmd=py("import json; print(json.dumps(dict(opened=True, rejected='decoded_dimensions', decoded_wh=[30000, 30000])))"))
    assert codes(r) == ["dimensions_too_large"] and r["stage"] == "worker"
    r = inspect_video(good_mp4, worker_cmd=py("import json; print(json.dumps(dict(opened=True, frames_read=1, decoded_wh=[9000, 9000])))"))   # a worker that forgot to check
    assert codes(r) == ["dimensions_too_large"]                                   # the parent re-checks


def test_worker_does_not_inherit_secrets(good_mp4, monkeypatch):
    monkeypatch.setenv("F8_PASSCODE", "abcd-secret-1234"); monkeypatch.setenv("SOME_API_TOKEN", "tok")
    code = "import json,os; print(json.dumps(dict(opened=True, codec=('LEAK' if ('F8_PASSCODE' in os.environ or 'SOME_API_TOKEN' in os.environ) else 'safe'))))"
    assert inspect_video(good_mp4, worker_cmd=py(code))["codec"] == "safe"


def test_parent_process_never_imports_opencv():
    import subprocess
    out = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {os.path.dirname(vc.__file__)!r}); import video_check; print('cv2' in sys.modules)"],
                         capture_output=True, text=True).stdout.strip()
    assert out == "False"


def test_inspection_never_modifies_the_file(good_mp4):
    import hashlib
    before = hashlib.sha256(open(good_mp4, "rb").read()).hexdigest(); inspect_video(good_mp4)
    assert hashlib.sha256(open(good_mp4, "rb").read()).hexdigest() == before

"""Original wall videos: hashing, manifest, deterministic frame extraction by index.  Videos are only ever READ."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = MODULE_DIR / "data" / "room_measurement" / "calibration_videos"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "config" / "frame_manifest.json"
VIEW_KEYS = ("AL", "PL", "AR", "PR")     # Air view 1, Pro view 1, Air view 2, Pro view 2

def load_manifest(path=DEFAULT_MANIFEST):
    m = json.loads(Path(path).read_text())
    for k in VIEW_KEYS:
        if k not in m["videos"]: raise ValueError(f"manifest lacks video {k}")
    n = {len(m["videos"][k]["frame_indices_per_set"]) for k in VIEW_KEYS}
    if len(n) != 1: raise ValueError("every video needs the same number of frame sets")
    return m

def n_sets(manifest): return len(manifest["videos"]["AL"]["frame_indices_per_set"])

def frame_index_table(manifest):
    """[{AL: idx, PL: idx, AR: idx, PR: idx} per frame set]"""
    return [{k: int(manifest["videos"][k]["frame_indices_per_set"][s]) for k in VIEW_KEYS} for s in range(n_sets(manifest))]

def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()

def verify_videos(manifest, video_dir=DEFAULT_VIDEO_DIR):
    """Compare full SHA-256 of the four wall videos against the manifest.  Returns list of dict rows; never modifies a file."""
    rows = []
    for k in VIEW_KEYS:
        v = manifest["videos"][k]; p = Path(video_dir) / v["file"]
        actual = sha256_file(p) if p.exists() else None
        rows.append(dict(view=k, file=v["file"], expected_sha256=v["sha256"], actual_sha256=actual, ok=actual == v["sha256"]))
    return rows

def extract_frames(video_path, indices):
    """Sequentially decode (no seeking) and return {index: grayscale uint8 frame}.  Raises if a requested frame does not exist."""
    want = sorted(set(int(i) for i in indices)); cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise IOError(f"cannot open {video_path}")
    out = {}; pos = 0
    while pos <= want[-1]:
        ok, f = cap.read()
        if not ok: break
        if pos in want: out[pos] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        pos += 1
    cap.release()
    missing = [i for i in want if i not in out]
    if missing: raise IndexError(f"{video_path}: frames {missing} not decodable")
    return out

def load_frame_sets(manifest, video_dir=DEFAULT_VIDEO_DIR):
    """[{AL,PL,AR,PR: gray frame} per frame set], decoding each video once."""
    table = frame_index_table(manifest); W, H = manifest["resolution_wh"]; per_video = {}
    for k in VIEW_KEYS:
        per_video[k] = extract_frames(Path(video_dir) / manifest["videos"][k]["file"], [t[k] for t in table])
    sets = []
    for t in table:
        s = {k: per_video[k][t[k]] for k in VIEW_KEYS}
        for k, im in s.items():
            if im.shape != (H, W): raise ValueError(f"{k}: frame shape {im.shape} != manifest {(H, W)}")
        sets.append(s)
    return sets

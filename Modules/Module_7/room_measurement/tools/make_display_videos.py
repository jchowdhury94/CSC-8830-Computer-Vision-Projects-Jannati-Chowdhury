"""Create small browser-playable H.264 MP4 copies of the four wall videos for the Room Measurement results page.

    python Modules/Module_7/room_measurement/tools/make_display_videos.py [--out DIR]

The originals (HEVC .MOV, hash-checked against config/frame_manifest.json) are only READ.  Each copy keeps every frame, the 30 fps
timing and the upright orientation; it is downscaled to 720 x 1280 (portrait) and re-encoded with libx264 (yuv420p, +faststart).
Audio is dropped (the page is about the picture; this also keeps the files small).  Output: outputs/room_measurement/videos/
plus videos_manifest.json (sizes, frame counts, durations, source hashes).  Nothing is cropped, retimed, edited or synthesised.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg

MODULE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODULE_DIR))
from room_measurement import frames as Fr  # noqa: E402

OUT_DIR = MODULE_DIR / "outputs" / "room_measurement" / "videos"
CRF, WIDTH, HEIGHT = 25, 720, 1280


def count(path):
    cap = cv2.VideoCapture(str(path))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, (w, h)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_DIR))
    out = Path(ap.parse_args(argv).out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = Fr.load_manifest()
    bad = [r for r in Fr.verify_videos(manifest) if not r["ok"]]
    if bad:
        sys.exit(f"source videos do not match the manifest hashes: {bad}")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    rows = {}
    for key in Fr.VIEW_KEYS:
        src = Fr.DEFAULT_VIDEO_DIR / manifest["videos"][key]["file"]
        dst = out / (Path(src).stem + ".mp4")
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-map", "0:v:0", "-an",
                        "-vf", f"scale={WIDTH}:{HEIGHT}:flags=lanczos", "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "slow",
                        "-crf", str(CRF), "-pix_fmt", "yuv420p", "-profile:v", "main", "-movflags", "+faststart", str(dst)], check=True)
        n_src, fps_src, size_src = count(src)
        n_dst, fps_dst, size_dst = count(dst)
        if n_src != n_dst or abs(fps_src - fps_dst) > 0.05 or size_dst != (WIDTH, HEIGHT):
            sys.exit(f"{dst.name}: frames {n_src}->{n_dst}, fps {fps_src}->{fps_dst}, size {size_dst}; refusing to continue")
        rows[key] = dict(file=dst.name, source=Path(src).name, source_sha256=manifest["videos"][key]["sha256"], frames=n_dst, fps=fps_dst,
                         duration_s=round(n_dst / fps_dst, 3), size=list(size_dst), bytes=dst.stat().st_size,
                         source_size=list(size_src))
    (out / "videos_manifest.json").write_text(json.dumps(dict(
        schema="csc8830.module7.room_display_videos.v1", codec="H.264 (libx264) main, yuv420p, faststart", crf=CRF, audio="removed",
        videos=rows, total_bytes=sum(r["bytes"] for r in rows.values()),
        note="Re-encoded copies of the original recordings (all frames, same timing/orientation, downscaled); originals untouched."), indent=1) + "\n")
    print(json.dumps({k: (v["frames"], v["bytes"]) for k, v in rows.items()}, indent=1))


if __name__ == "__main__":
    main()

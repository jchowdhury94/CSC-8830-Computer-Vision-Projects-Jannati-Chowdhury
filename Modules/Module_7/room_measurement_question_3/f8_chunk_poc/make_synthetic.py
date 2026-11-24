"""Create synthetic test inputs in a directory of your choice (default: the system temp dir). Nothing real or protected is read.

  python make_synthetic.py <out_dir> bytes <MB>        random bytes (not a video) -- transfer/memory tests
  python make_synthetic.py <out_dir> video <seconds>   small valid mp4v video (moving shapes + noise)
"""
import os
import sys
from pathlib import Path


def make_bytes(path, mb):
    with open(path, "wb") as fh:
        for _ in range(mb):
            fh.write(os.urandom(1 << 20))
    return path


def make_video(path, seconds=3, w=640, h=360, fps=30, noise=0):
    import cv2
    import numpy as np
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open mp4v")
    rng = np.random.default_rng(0)
    for i in range(int(seconds * fps)):
        f = np.full((h, w, 3), 40, np.uint8)
        cv2.circle(f, (50 + (i * 7) % (w - 100), h // 2 + int(80 * np.sin(i / 10))), 30, (0, 200, 255), -1)
        cv2.putText(f, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if noise:
            f = cv2.add(f, rng.integers(0, noise, f.shape, dtype=np.uint8))
        vw.write(f)
    vw.release()
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); kind = sys.argv[2]; n = float(sys.argv[3])
    p = make_bytes(out / f"synthetic_{int(n)}MB.bin", int(n)) if kind == "bytes" else make_video(out / f"synthetic_{int(n)}s.mp4", n)
    print(p, p.stat().st_size)

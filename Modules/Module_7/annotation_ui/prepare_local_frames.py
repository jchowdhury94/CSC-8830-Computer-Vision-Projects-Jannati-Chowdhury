"""Stage F1: extract the 8 ORIGINAL-RESOLUTION frames behind the Stage E3 exported annotation frames into a local test-fixture folder.

    python annotation_ui/prepare_local_frames.py

Reads (never writes) the original videos, decodes only the selected frame indices (Phase 1 sequential decode), verifies each frame's SHA-256
fingerprint (the backend's method, visitor_annotations.frame_fingerprint) against the fingerprint recorded in the Stage E3 export, and writes
lossless grayscale PNGs + an index under outputs/stage_f1_local_frames/ (outside outputs/room_measurement, which has an output-size test).
Refuses to overwrite.  These are LOCAL TEST FIXTURES, not visitor uploads.
"""
import json
import sys
from pathlib import Path

import cv2

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from room_measurement import frames as Fr, visitor_annotations as V       # noqa: E402

E3_INDEX = MODULE / "outputs" / "stage_e3_annotation_frames" / "frames_index.json"
OUT = MODULE / "outputs" / "stage_f1_local_frames"


def main():
    idx_p = OUT / "frames_full_index.json"
    if idx_p.exists(): sys.exit(f"{OUT} already prepared; refusing to overwrite")
    OUT.mkdir(parents=True, exist_ok=True); e3 = json.loads(E3_INDEX.read_text())["frames"]; m = Fr.load_manifest(); rows = []
    for vk in Fr.VIEW_KEYS:
        want = {f["set_index"]: f for f in e3 if f["view_key"] == vk}; got = Fr.extract_frames(Fr.DEFAULT_VIDEO_DIR / m["videos"][vk]["file"], [f["frame_index"] for f in want.values()])
        for si, f in sorted(want.items()):
            g = got[f["frame_index"]]; fp = V.frame_fingerprint(g)
            if fp != f["frame_fingerprint"]: sys.exit(f"fingerprint mismatch for set {si} {vk}")
            name = f"set{si}_{vk}_f{f['frame_index']}.png"; cv2.imwrite(str(OUT / name), g, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            back = cv2.imread(str(OUT / name), cv2.IMREAD_GRAYSCALE)
            if V.frame_fingerprint(back) != fp: sys.exit(f"PNG round trip changed {name}")
            rows.append(dict(set_index=si, view_key=vk, frame_index=f["frame_index"], frame_fingerprint=fp, image=dict(width=g.shape[1], height=g.shape[0]), file=name))
    idx_p.write_text(json.dumps(dict(source="original videos, frame indices from the Stage E3 export; fingerprints verified", note="LOCAL TEST FIXTURES (not visitor uploads)", frames=rows), indent=1) + "\n")
    print("wrote", len(rows), "full-resolution frames")


if __name__ == "__main__":
    main()

"""Stage F1 LOCAL TEST FIXTURE loader (not a visitor upload path).

Frames: the original-resolution Stage E3 selected frames written by prepare_local_frames.py (fingerprints verified against the backend method).
Calibration: the Recipe A video calibration passed through the Stage A adapter with the measured 225 mm baseline, exactly as in the E2 regression;
it is test data for the prototype only, the UI core takes any calibration."""
import json
from pathlib import Path

from room_measurement import calibration as C, visitor_calibration as VC
from core import FrameRecord

MODULE = Path(__file__).resolve().parents[1]
FRAMES_DIR = MODULE / "outputs" / "stage_f1_local_frames"


def load_frames(frames_dir=FRAMES_DIR) -> dict:
    idx = json.loads((Path(frames_dir) / "frames_full_index.json").read_text())["frames"]
    return {(f["set_index"], f["view_key"]): FrameRecord(f["set_index"], f["view_key"], f["frame_index"], f["frame_fingerprint"], (f["image"]["width"], f["image"]["height"]), str(Path(frames_dir) / f["file"])) for f in idx}


def load_calibration():
    d = json.loads(Path(C.DEFAULT_CALIBRATION_PATH).read_text())
    model = dict(K1=d["left"]["K"], D1=d["left"]["distortion_k1_k2_p1_p2_k3"], K2=d["right"]["K"], D2=d["right"]["distortion_k1_k2_p1_p2_k3"], image_size_left=d["image_size_wh"], image_size_right=d["image_size_wh"],
                 camera=dict(left=dict(capture_mode="video"), right=dict(capture_mode="video")))
    ad = VC.adapt_visitor_calibration(model, 225.0, name="f1_local_fixture")
    return ad.require()


def available(frames_dir=FRAMES_DIR) -> bool:
    return (Path(frames_dir) / "frames_full_index.json").exists()

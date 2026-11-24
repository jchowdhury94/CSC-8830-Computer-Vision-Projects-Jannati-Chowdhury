"""Recipe A calibration artifact: load / save / validate.  JSON floats round-trip float64 exactly."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_PATH = MODULE_DIR / "calibration" / "room_video" / "room_video_recipe_a_calibration.json"
SCHEMA = "csc8830.module7.room_video_calibration.v1"

@dataclass(frozen=True)
class StereoCalibration:
    name: str
    K_left: np.ndarray            # 3x3, pixels, calibration resolution
    D_left: np.ndarray            # (5,) OpenCV k1,k2,p1,p2,k3
    K_right: np.ndarray
    D_right: np.ndarray
    image_size: tuple             # (width, height) px of the calibration / video frames
    baseline_mm: float            # PHYSICAL baseline used to fix the metric scale of the pose
    provenance: dict = field(default_factory=dict)
    legacy_extrinsics: dict | None = None   # provenance only; never used for the wall-video pose

    def with_intrinsics(self, K_left=None, K_right=None, D_left=None, D_right=None):
        return replace(self, K_left=self.K_left if K_left is None else K_left, K_right=self.K_right if K_right is None else K_right,
                       D_left=self.D_left if D_left is None else D_left, D_right=self.D_right if D_right is None else D_right)

def _check(K, D, label):
    K = np.asarray(K, np.float64); D = np.asarray(D, np.float64).reshape(-1)
    if K.shape != (3, 3): raise ValueError(f"{label}: intrinsic matrix must be 3x3, got {K.shape}")
    if D.shape != (5,): raise ValueError(f"{label}: expected 5 distortion coefficients, got {D.shape}")
    if not (np.isfinite(K).all() and np.isfinite(D).all()): raise ValueError(f"{label}: non-finite values")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]) or K[0, 1] != 0: raise ValueError(f"{label}: malformed intrinsic matrix")
    return K, D

def validate(cal: StereoCalibration) -> StereoCalibration:
    K1, D1 = _check(cal.K_left, cal.D_left, "left"); K2, D2 = _check(cal.K_right, cal.D_right, "right")
    w, h = cal.image_size
    if not (w > 0 and h > 0): raise ValueError("image_size must be positive")
    for K, lab in ((K1, "left"), (K2, "right")):
        if not (0 < K[0, 2] < w and 0 < K[1, 2] < h): raise ValueError(f"{lab}: principal point outside the {w}x{h} image")
    if not cal.baseline_mm > 0: raise ValueError("baseline must be positive")
    return replace(cal, K_left=K1, D_left=D1, K_right=K2, D_right=D2, image_size=(int(w), int(h)), baseline_mm=float(cal.baseline_mm))

def to_dict(cal: StereoCalibration) -> dict:
    d = dict(schema=SCHEMA, name=cal.name, image_size_wh=list(cal.image_size), physical_baseline_mm=cal.baseline_mm, units=dict(length="mm", image="px"),
             left=dict(camera="iPhone Air, 1x rear", K=np.asarray(cal.K_left).tolist(), distortion_k1_k2_p1_p2_k3=np.asarray(cal.D_left).reshape(-1).tolist()),
             right=dict(camera="iPhone 17 Pro Max, 1x rear", K=np.asarray(cal.K_right).tolist(), distortion_k1_k2_p1_p2_k3=np.asarray(cal.D_right).reshape(-1).tolist()),
             provenance=cal.provenance)
    if cal.legacy_extrinsics is not None:
        le = cal.legacy_extrinsics; d["legacy_extrinsics_NOT_USED"] = dict(R=np.asarray(le["R"]).tolist(), T_mm=np.asarray(le["T"]).reshape(-1).tolist(), calibrated_baseline_mm=le["baseline_mm"])
    return d

def from_dict(d: dict) -> StereoCalibration:
    if d.get("schema") != SCHEMA: raise ValueError(f"unsupported calibration schema {d.get('schema')!r}")
    le = d.get("legacy_extrinsics_NOT_USED")
    cal = StereoCalibration(name=d["name"], K_left=np.array(d["left"]["K"], np.float64), D_left=np.array(d["left"]["distortion_k1_k2_p1_p2_k3"], np.float64),
                            K_right=np.array(d["right"]["K"], np.float64), D_right=np.array(d["right"]["distortion_k1_k2_p1_p2_k3"], np.float64),
                            image_size=tuple(d["image_size_wh"]), baseline_mm=float(d["physical_baseline_mm"]), provenance=d.get("provenance", {}),
                            legacy_extrinsics=None if le is None else dict(R=np.array(le["R"]), T=np.array(le["T_mm"]).reshape(3, 1), baseline_mm=le["calibrated_baseline_mm"]))
    return validate(cal)

def load_calibration(path=DEFAULT_CALIBRATION_PATH) -> StereoCalibration:
    return from_dict(json.loads(Path(path).read_text()))

def save_calibration(cal: StereoCalibration, path=DEFAULT_CALIBRATION_PATH, overwrite=False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite: raise FileExistsError(f"{path} exists; refusing to overwrite a calibration artifact")
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(to_dict(validate(cal)), indent=1) + "\n"); return path

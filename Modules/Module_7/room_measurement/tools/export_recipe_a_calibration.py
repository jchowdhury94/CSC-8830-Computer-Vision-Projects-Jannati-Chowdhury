"""One-off export of the 'Recipe A' calibration from the exploratory scratch pickle into the repository artifact.

    python export_recipe_a_calibration.py /path/to/recipes.pkl  [--out FILE]   (refuses to overwrite)

Not used in normal execution: the pipeline reads the JSON artifact only.  The pickle path is an argument on purpose.
"""
import argparse, hashlib, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from room_measurement.calibration import DEFAULT_CALIBRATION_PATH, StereoCalibration, save_calibration  # noqa: E402

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("recipes_pkl"); ap.add_argument("--out", default=str(DEFAULT_CALIBRATION_PATH)); a = ap.parse_args()
    raw = Path(a.recipes_pkl).read_bytes(); A = pickle.loads(raw)["A"]
    prov = dict(
        recipe="Recipe A ('radial3', intrinsics fixed, stereo extrinsics from a separate stereoCalibrate pass)",
        distortion_model="OpenCV [k1, k2, p1, p2, k3] with p1 = p2 = 0 (radial3: k1,k2,k3)",
        source="exploratory scratch fit (checkerboard corners from the calibration videos left_air / right_pro and their repeat recordings)",
        pooled_corner_sets=dict(mono_left=int(A["n"][0]), mono_right=int(A["n"][1]), stereo_pairs=int(A["n"][2])),
        fit_quality_px=dict(rms_left=float(A["rms_l"]), rms_right=float(A["rms_r"]), rms_stereo=float(A["rms_s"])),
        source_pickle_sha256=hashlib.sha256(raw).hexdigest(),
        not_the_old_photo_calibration="Module 7 Model B (30 photo pairs, calibration/model_b) is NOT reused for the wall videos",
        extrinsics_note="legacy_extrinsics are the stereoCalibrate output of the calibration pass. They are stored for provenance ONLY: "
                        "the wall-video rig pose is re-estimated (Method J) and the legacy values are never read by the pipeline.")
    cal = StereoCalibration(name="room_video_recipe_a", K_left=A["K1"], D_left=A["D1"], K_right=A["K2"], D_right=A["D2"], image_size=(2160, 3840),
                            baseline_mm=225.0, provenance=prov, legacy_extrinsics=dict(R=A["R"], T=A["T"], baseline_mm=float(A["baseline"])))
    save_calibration(cal, a.out, overwrite=False); print("wrote", a.out)

if __name__ == "__main__":
    main()

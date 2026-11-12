"""Read-only loader for the Room Measurement results page (Module 7 Part 3, Phase 2A).

Reads ONLY the committed JSON outputs (estimation_result.json, validation.json) and lists the committed display images.  It does
no estimation: it imports neither the estimator, OpenCV nor ``validation`` (the one module that holds the ground truth), and it
never needs the original .MOV recordings.  Every number shown on the page comes from the JSON files.
"""
import json
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = MODULE_DIR / "outputs" / "room_measurement"
DISPLAY_DIR = RESULTS_DIR / "display"
ESTIMATION_FILE, VALIDATION_FILE = "estimation_result.json", "validation.json"
SCHEMA = "csc8830.module7.room_measurement_result.v1"

# (file name, tab, caption) -- the committed images, in display order
IMAGES = (
    ("view1_stereo_pair.jpg", "stereo", "Placement 1: left (iPhone Air) and right (iPhone Pro) camera frames."),
    ("view2_stereo_pair.jpg", "stereo", "Placement 2: left (iPhone Air) and right (iPhone Pro) camera frames."),
    ("view1_boundaries_corners.jpg", "corners", "Placement 1: observed edge points, fitted boundary lines and inferred corners."),
    ("view2_boundaries_corners.jpg", "corners", "Placement 2: observed edge points, fitted boundary lines and inferred corners."),
    ("overlapping_views.jpg", "overlap", "The two left-camera views overlap on the floor; green lines are a sample of the matched "
                                         "features used to register placement 1 into placement 2."),
)


VIDEO_DIR = RESULTS_DIR / "videos"
CALIBRATION_FILE = MODULE_DIR / "calibration" / "room_video" / "room_video_recipe_a_calibration.json"
# The checkerboard used for the room-video calibration is not stored in the calibration JSON; these are the capture-setup values
# (inner corners cols x rows, square size), the same board settings documented in the Module 7 README.
CHECKERBOARD_INNER_CORNERS = (8, 6)
CHECKERBOARD_SQUARE_MM = 20.0

# (group, file, label): H.264 display copies of the four wall recordings, in display order
VIDEOS = (
    ("view1", "left_air_wall_left.mp4", "iPhone Air (Left Camera)"),
    ("view1", "right_pro_wall_left.mp4", "iPhone 17 Pro Max (Right Camera)"),
    ("view2", "left_air_wall_right.mp4", "iPhone Air (Left Camera)"),
    ("view2", "right_pro_wall_right.mp4", "iPhone 17 Pro Max (Right Camera)"),
)
VIDEO_GROUPS = {"view1": "View 1 \u2014 Left Side of Wall", "view2": "View 2 \u2014 Right Side of Wall"}


class ResultsUnavailable(Exception):
    """The committed results are missing or unusable; the message is shown to the visitor."""


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ResultsUnavailable(f"Required results file not found: {Path(path).name}") from None
    except (OSError, ValueError) as error:
        raise ResultsUnavailable(f"Results file {Path(path).name} could not be read ({error}).") from None


def load_results(results_dir=RESULTS_DIR):
    """Return the numbers for the page, taken from the two committed JSON files.  Raises ResultsUnavailable otherwise."""
    results_dir = Path(results_dir)
    estimation, validation = _read(results_dir / ESTIMATION_FILE), _read(results_dir / VALIDATION_FILE)
    try:
        if estimation["schema"] != SCHEMA:
            raise ResultsUnavailable(f"{ESTIMATION_FILE} has an unexpected schema ({estimation['schema']!r}).")
        if estimation["ground_truth_used"] is not False:
            raise ResultsUnavailable(f"{ESTIMATION_FILE} does not certify that the estimate was made without ground truth.")
        total = estimation["uncertainty"]["total"]
        dims = {}
        for key in ("width", "height"):
            v = validation[f"{key}_mm"]
            est = estimation["summary"][f"{key}_mm"]
            if abs(v["estimate_mm"] - est) > 1e-6 or abs(total[key]["estimate_mm"] - est) > 1e-6:
                raise ResultsUnavailable(f"{VALIDATION_FILE} and {ESTIMATION_FILE} disagree about the {key} estimate.")
            dims[key] = dict(estimate_mm=est, actual_mm=v["ground_truth_mm"], error_mm=v["error_mm"], abs_error_mm=v["abs_error_mm"],
                             error_percent=v["error_percent"], interval_95_mm=tuple(total[key]["interval_95_mm"]),
                             sd_mm=total[key]["sd_mm"], actual_inside_interval=v["ground_truth_inside_interval"])
        unc = estimation["uncertainty"]
        pose = estimation["pose"]
        flags = pose["report"]["method_J"]
        return dict(
            width=dims["width"], height=dims["height"],
            components={name: dict(width_sd_mm=c["width_sd_mm"], height_sd_mm=c["height_sd_mm"])
                        for name, c in unc["components_1sigma"].items()},
            assumptions=unc["assumptions"],
            pose=dict(baseline_mm=pose["baseline_mm"], yaw_deg=pose["yaw_deg"], robust_scale_px=flags["robust_scale_px"],
                      inlier_fraction=flags["flags"]["inlier_fraction"], n_epipolar=flags["n_epipolar"], n_tracks=flags["n_tracks"]),
            n_frame_sets=estimation["summary"]["n_sets"], n_sets_ok=estimation["summary"]["n_sets_ok"],
            manual=estimation["manual_vs_automatic"]["manual"], automatic=estimation["manual_vs_automatic"]["automatic"],
            lens_height_check=validation.get("lens_height_check"),
            ground_truth_used_in_estimate=estimation["ground_truth_used"])
    except KeyError as missing:
        raise ResultsUnavailable(f"Results files are incomplete (missing field {missing}).") from None


def image_status(display_dir=DISPLAY_DIR):
    """[(path, tab, caption, exists)] for the committed display images."""
    display_dir = Path(display_dir)
    return [(display_dir / name, tab, caption, (display_dir / name).is_file()) for name, tab, caption in IMAGES]


def cm(mm):
    """Whole centimetres for display (the estimate is only good to about this level)."""
    return f"{mm / 10:.0f}"


def signed_cm(mm):
    return f"{mm / 10:+.1f}".replace("-", "−")


def video_status(video_dir=VIDEO_DIR):
    """[(path, group, label, exists)] for the committed display videos."""
    video_dir = Path(video_dir)
    return [(video_dir / name, group, label, (video_dir / name).is_file()) for group, name, label in VIDEOS]


def load_calibration(path=CALIBRATION_FILE):
    """Intrinsics, distortion and rig facts from the room-video calibration JSON.  The legacy stereo extrinsics stored in that file
    are deliberately not returned: they are provenance only and are not the pose used for the wall videos."""
    data = _read(path)
    try:
        side = {}
        for key in ("left", "right"):
            block = data[key]
            k1, k2, p1, p2, k3 = block["distortion_k1_k2_p1_p2_k3"]
            side[key] = dict(camera=block["camera"], K=block["K"], distortion=dict(k1=k1, k2=k2, p1=p1, p2=p2, k3=k3))
        width, height = data["image_size_wh"]
        return dict(left=side["left"], right=side["right"], width=width, height=height, baseline_mm=data["physical_baseline_mm"],
                    pattern=CHECKERBOARD_INNER_CORNERS, square_mm=CHECKERBOARD_SQUARE_MM, name=data["name"])
    except (KeyError, ValueError, TypeError) as missing:
        raise ResultsUnavailable(f"Calibration file {Path(path).name} is incomplete ({missing}).") from None

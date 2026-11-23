"""Fast, isolated tests of the three-viewpoint Wall B reconstruction (synthetic geometry only; no videos, no protected data)."""
import copy

import cv2
import numpy as np
import pytest

from room_measurement import geometry as G, three_view_wall_b as T
from room_measurement.tests import synthetic_three_view as S


@pytest.fixture(scope="module")
def scene():
    return S.build()


# ------------------------------------------------------------------ calibration contract
def test_contract_values_are_used_unchanged_and_baseline_is_not_hardcoded():
    for b in (225.0, 300.0, 412.5):
        c = S.contract(b); cal = T.calibration_from_contract(c)
        assert cal.baseline_mm == b and cal.image_size == tuple(c["image_size_wh"])
        assert np.array_equal(cal.K_left, np.array(c["K1"])) and np.array_equal(cal.D_right, np.array(c["D2"]))


@pytest.mark.parametrize("breakit", [lambda c: c.pop("K1"), lambda c: c.update(baseline_mm=-1), lambda c: c.update(D1=[0, 0]), lambda c: c.update(image_size_wh=[10, 10])])
def test_invalid_contract_is_refused(breakit):
    c = S.contract(); breakit(c)
    with pytest.raises(T.ThreeViewError) as e: T.calibration_from_contract(c)
    assert e.value.code == "CALIBRATION_INVALID"


def test_reconstruction_never_changes_the_calibration(scene):
    cal, ms, doc, _ = scene; before = [a.copy() for a in (cal.K_left, cal.D_left, cal.K_right, cal.D_right)] + [cal.baseline_mm, cal.image_size]
    T.reconstruct_frame(cal, ms, T.parse_annotations(doc, cal.image_size), 60)
    after = [cal.K_left, cal.D_left, cal.K_right, cal.D_right, cal.baseline_mm, cal.image_size]
    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b for a, b in zip(before, after))


# ------------------------------------------------------------------ resolution handling
def test_floor_row_threshold_scales_with_frame_height_and_is_2600_at_the_original_height():
    assert T.floor_y_min(T.calibration_from_contract(S.contract(225, f=3000, size=(2160, 3840)))) == 2600.0
    assert T.floor_y_min(T.calibration_from_contract(S.contract(225))) == pytest.approx(1300.0)


def test_frame_size_must_match_the_calibration(tmp_path):
    cal = T.calibration_from_contract(S.contract())
    p = tmp_path / "tiny.mp4"; w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    for _ in range(3): w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    with pytest.raises(T.ThreeViewError) as e: T.read_gray_frame(p, 0, cal)
    assert e.value.code == "RESOLUTION_MISMATCH"
    assert T.read_gray_frame(p, 0).shape == (48, 64)                        # without a calibration it is only decoded
    with pytest.raises(T.ThreeViewError) as e: T.read_gray_frame(tmp_path / "missing.mp4", 0)
    assert e.value.code == "VIDEO_UNREADABLE"


def test_annotation_frame_size_must_match_the_calibration(tmp_path):
    cal = T.calibration_from_contract(S.contract()); _, _, doc, _ = S.build()
    ann = T.Annotations(T.parse_annotations(doc).lines, (640, 480))          # annotations recorded on a different frame size
    with pytest.raises(T.ThreeViewError) as e: T.measure_from_videos(cal, {(v, s): tmp_path / "x.mp4" for v in (1, 2, 3) for s in "LR"}, [60], ann)
    assert e.value.code == "RESOLUTION_MISMATCH"


# ------------------------------------------------------------------ annotation schema
def test_annotation_schema_accepts_the_valid_document_and_carries_v2_horizontal(scene):
    cal, _, doc, _ = scene; ann = T.parse_annotations(doc, cal.image_size)
    assert ann.has(2, 60, "right", "horizontal") and not ann.has(2, 60, "left", "vertical")
    T.check_annotations_complete(ann, [60])                                    # V2 needs nothing for the length


@pytest.mark.parametrize("mutate,code", [
    (lambda d: d.update(schema="other"), "ANNOTATION_SCHEMA"),
    (lambda d: d["annotations"][0].update(points=[[5, 5], [5, 5]]), "ANNOTATION_MALFORMED"),
    (lambda d: d["annotations"][0].update(points=[[5, 5], [float("nan"), 9]]), "ANNOTATION_MALFORMED"),
    (lambda d: d["annotations"][0].update(points=[[5, 5], [5000, 900]]), "ANNOTATION_OUTSIDE_FRAME"),
    (lambda d: d["annotations"][0].update(viewpoint="V4"), "ANNOTATION_MALFORMED"),
    (lambda d: d["annotations"].append(copy.deepcopy(d["annotations"][0])), "ANNOTATION_DUPLICATE"),
])
def test_bad_annotations_are_refused(mutate, code):
    _, _, doc, _ = S.build(); mutate(doc)
    with pytest.raises(T.ThreeViewError) as e: T.parse_annotations(doc, S.SIZE)
    assert e.value.code == code


def test_missing_corner_line_is_named():
    _, _, doc, _ = S.build(); doc["annotations"] = [a for a in doc["annotations"] if not (a["viewpoint"] == "V3" and a["camera"] == "right")]
    ann = T.parse_annotations(doc, S.SIZE)
    with pytest.raises(T.ThreeViewError) as e: T.check_annotations_complete(ann, [60])
    assert e.value.code == "ANNOTATION_MISSING" and "V3 Pro vertical" in e.value.message


# ------------------------------------------------------------------ geometry: the registration chain against known truth
def test_noise_free_synthetic_scene_recovers_the_known_length(scene):
    cal, ms, doc, L = scene; r = T.reconstruct_frame(cal, ms, T.parse_annotations(doc, cal.image_size), 60)
    assert r["length_mm"] == pytest.approx(L, rel=0.01)                        # noise-free: the chain must be geometrically consistent to ~1 %
    m = r["metrics"]
    assert m["reg_v1_v2"]["n_inliers"] > 50 and m["reg_v3_v2"]["n_inliers"] > 50
    assert m["floor_consistency"]["ang_v1_v2_deg"] < 0.5 and m["floor_consistency"]["ang_v3_v2_deg"] < 0.5
    assert m["corner"]["length_vertical_component_mm"] < 0.02 * L


def test_corners_are_expressed_in_the_v2_frame_and_their_distance_is_the_length(scene):
    cal, ms, doc, _ = scene; r = T.reconstruct_frame(cal, ms, T.parse_annotations(doc, cal.image_size), 60)
    assert np.linalg.norm(np.array(r["corner_v3_in_v2"]) - np.array(r["corner_v1_in_v2"])) == pytest.approx(r["length_mm"])
    # the V1 corner (left end) lies to the left of the V3 corner (right end) in the V2 camera frame, and both are in front of V2
    assert r["corner_v1_in_v2"][0] < r["corner_v3_in_v2"][0] and r["corner_v1_in_v2"][2] > 0 and r["corner_v3_in_v2"][2] > 0


def test_registration_reproduces_a_known_rigid_transform():
    rng = np.random.default_rng(1); X = rng.uniform(-2000, 2000, (200, 3)); R = cv2.Rodrigues(np.array([0.1, -0.4, 0.2]))[0]; t = np.array([300.0, -50.0, 900.0])
    R0, t0, rms, ni, nt = G.rigid_ransac(X, X @ R.T + t, np.random.default_rng(3), 300, 40.0)
    assert np.allclose(R0, R, atol=1e-9) and np.allclose(t0, t, atol=1e-6) and ni == nt == 200 and rms < 1e-6


def test_metric_scale_follows_the_contract_baseline(scene):
    """The same images with a contract baseline 1.5x larger give a 1.5x longer wall: the scale comes from the contract, not from a constant."""
    _, ms, doc, L = scene
    cal2 = T.calibration_from_contract(S.contract(450.0)); r = T.reconstruct_frame(cal2, ms, T.parse_annotations(doc, cal2.image_size), 60)
    assert r["length_mm"] == pytest.approx(1.5 * L, rel=0.02)


def test_the_v2_horizontal_annotation_does_not_influence_the_result(scene):
    cal, ms, doc, _ = scene; a = T.reconstruct_frame(cal, ms, T.parse_annotations(doc, cal.image_size), 60)["length_mm"]
    d2 = copy.deepcopy(doc)
    for x in d2["annotations"]:
        if x["viewpoint"] == "V2": x["points"] = [[10.0, 900.0], [1000.0, 100.0]]
    d3 = copy.deepcopy(doc); d3["annotations"] = [x for x in d3["annotations"] if x["viewpoint"] != "V2"]
    assert T.reconstruct_frame(cal, ms, T.parse_annotations(d2, cal.image_size), 60)["length_mm"] == a
    assert T.reconstruct_frame(cal, ms, T.parse_annotations(d3, cal.image_size), 60)["length_mm"] == a


def test_reconstruction_is_deterministic(scene):
    cal, ms, doc, _ = scene; ann = T.parse_annotations(doc, cal.image_size)
    assert T.reconstruct_frame(cal, ms, ann, 60)["length_mm"] == T.reconstruct_frame(cal, ms, ann, 60)["length_mm"]


# ------------------------------------------------------------------ structural failures
def test_too_few_registration_pairs_is_a_structural_failure(scene):
    cal, ms, doc, _ = scene; bad = copy.deepcopy(ms); bad[(1, 2)]["MX"] = bad[(1, 2)]["MX"][:3]
    with pytest.raises(T.ThreeViewError) as e: T.reconstruct_frame(cal, bad, T.parse_annotations(doc, cal.image_size), 60)
    assert e.value.code == "REGISTRATION_PAIRS"


def test_missing_video_slot_is_named(scene):
    cal, _, doc, _ = scene
    with pytest.raises(T.ThreeViewError) as e: T.measure_from_videos(cal, {(1, "L"): "a.mov"}, [60], T.parse_annotations(doc, cal.image_size))
    assert e.value.code == "VIDEO_MISSING" and "V3 R" in e.value.message


def test_module_code_has_no_streamlit_calibration_file_or_225_mm_dependency():
    import ast
    tree = ast.parse(open(T.__file__).read()); names, consts, mods = set(), set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name): names.add(n.id)
        elif isinstance(n, ast.Attribute): names.add(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): consts.add(n.value)
        elif isinstance(n, ast.Import): mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom): mods.add(n.module or "")
    assert not any("streamlit" in m for m in mods)
    assert not names & {"load_calibration", "DEFAULT_CALIBRATION_PATH", "save_calibration"}
    assert 225 not in consts and 225.0 not in consts

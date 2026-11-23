"""F8.29: calibration-observation retention, schema /2, backward compatibility, limits, regression of calibration numbers.

SYNTHETIC corner data only. No video is opened or decoded (``extract_detections`` is replaced by a function that returns synthetic detections; the two
"video" files are dummy bytes), no real calibration job is started, nothing from the original experiment is read."""
import copy
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2]
for sub in ("", "visitor_workflow", "f8_chunk_poc"):
    sys.path.insert(0, str(MODULE / sub) if sub else str(MODULE))

import calibration_observations as CO                    # noqa: E402
import calibration_worker as W                           # noqa: E402
import f7_calibration as K                               # noqa: E402
import f8_calibration_bridge as B                        # noqa: E402
import result_store as RS                                # noqa: E402
from stereo_measure import calibration_io as cio, user_calibration as uc   # noqa: E402

PATTERN, SQ, SIZE = (8, 6), 25.0, (1280, 720)
F_PX, BASE = 1000.0, 100.0


# ------------------------------------------------------------------ synthetic stereo detections (no video)
def _rig_detections(n_poses=16, seed=3, flip_right=(2, 5, 9), noise=0.15):
    rng = np.random.default_rng(seed)
    Kc = np.array([[F_PX, 0, 640.0], [0, F_PX, 360.0], [0, 0, 1.0]])
    cols, rows = PATTERN
    obj = np.zeros((cols * rows, 3)); obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQ
    obj -= obj.mean(0)
    Rr, Tr = np.eye(3), np.array([-BASE, 0.0, 0.0])                       # right camera sits +BASE along x of the left camera
    left, right, frame, flipped = [], [], 0, []
    for k in range(n_poses):
        rv = rng.uniform(-0.45, 0.45, 3); tv = np.array([rng.uniform(-120, 120), rng.uniform(-70, 70), rng.uniform(550, 950)])
        pl, _ = cv2.projectPoints(obj, rv, tv, Kc, None)
        Rb, _ = cv2.Rodrigues(rv)
        rvr, _ = cv2.Rodrigues(Rr @ Rb); pr, _ = cv2.projectPoints(obj, rvr, Rr @ tv + Tr, Kc, None)
        pl, pr = pl.reshape(-1, 2) + rng.normal(0, noise, (len(obj), 2)), pr.reshape(-1, 2) + rng.normal(0, noise, (len(obj), 2))
        if k in flip_right:
            pr = pr[::-1].copy(); flipped.append(k)                       # the detector numbered the board the other way round
        for j in range(3):                                                # the board is held still for three consecutive samples
            left.append(dict(frame=frame, corners=pl + rng.normal(0, 0.03, pl.shape), square_px=60.0, sharp=100.0 + k * 3 + j))
            right.append(dict(frame=frame, corners=pr + rng.normal(0, 0.03, pr.shape), square_px=60.0, sharp=100.0 + k * 3 + j))
            frame += K.STRIDE
        frame += 20
    mk = lambda d: dict(size=SIZE, fps=30.0, n_frames=frame, motion=np.zeros(frame), detections=d)
    return mk(left), mk(right), flipped


@pytest.fixture(scope="module")
def rig():
    left, right, flipped = _rig_detections()
    settings = dict(cols=PATTERN[0], rows=PATTERN[1], square_mm=SQ)
    spy = {}
    orig = uc.calibrate_stereo

    def capture(l, r, *a, **k):
        spy["left"], spy["right"], spy["pair_ids"] = [np.array(x, copy=True) for x in l], [np.array(x, copy=True) for x in r], list(k.get("pair_ids") or [])
        return orig(l, r, *a, **k)
    uc.calibrate_stereo = capture
    try:
        res = K.calibrate_from_detections(left, right, settings)
    finally:
        uc.calibrate_stereo = orig
    assert res["ok"], res.get("error")
    return dict(left=left, right=right, settings=settings, res=res, spy=spy, flipped=flipped)


# ------------------------------------------------------------------ 1-3: what is retained
def test_selected_pairs_are_retained_with_their_frame_indices(rig):
    res, o = rig["res"], rig["res"]["observations"]
    assert len(o["left"]) == len(o["right"]) == res["pairs_used"] == len(res["pair_frames"]) >= uc.MIN_VALID_PAIRS
    by_frame = {d["frame"]: d["corners"] for d in rig["left"]["detections"]}
    for corners, (lf, _rf) in zip(o["left"], res["pair_frames"]):
        assert np.array_equal(corners, by_frame[lf])                      # the left corners are those of the selected frame


def test_post_ordering_right_corners_are_what_the_solver_received(rig):
    o, spy = rig["res"]["observations"], rig["spy"]
    assert len(spy["right"]) == len(o["right"])
    for a, b in zip(o["right"], spy["right"]):
        assert np.array_equal(np.asarray(a, np.float64).reshape(-1, 2), b)       # bit-exact the arrays calibrate_stereo got
    for a, b in zip(o["left"], spy["left"]):
        assert np.array_equal(np.asarray(a, np.float64).reshape(-1, 2), b)
    assert spy["pair_ids"] == o["pair_ids"]


def test_ordering_correction_is_retained_not_the_raw_detection(rig):
    res, o = rig["res"], rig["res"]["observations"]
    raw_right = {d["frame"]: d["corners"] for d in rig["right"]["detections"]}
    corrected = 0
    for rc, (_lf, rf) in zip(o["right"], res["pair_frames"]):
        if not np.array_equal(np.asarray(rc).reshape(-1, 2), raw_right[rf]):
            corrected += 1
            assert np.allclose(np.asarray(rc).reshape(-1, 2), raw_right[rf][::-1])      # it is the reordered raw detection
    assert corrected >= 1, "the synthetic flipped pairs should have been order-corrected"


def test_unused_detections_are_not_persisted(rig):
    res = rig["res"]
    total = len(rig["left"]["detections"]) + len(rig["right"]["detections"])
    assert total > 2 * res["pairs_used"]
    obs = CO.build_observations(pattern=PATTERN, square_mm=SQ, image_size_left=SIZE, image_size_right=SIZE, pair_ids=res["observations"]["pair_ids"],
                                pair_frames=res["pair_frames"], left_corners=res["observations"]["left"], right_corners=res["observations"]["right"],
                                params=dict(K1=res["model"]["K1"], D1=res["model"]["D1"], K2=res["model"]["K2"], D2=res["model"]["D2"], R=res["model"]["R"], T=res["model"]["T"]))
    assert obs["n_pairs"] == res["pairs_used"] and len(obs["pairs"]) == res["pairs_used"]
    text = json.dumps(obs)
    assert not any(k in text for k in ('"detections"', '"motion"', '"sharp"', '"square_px"'))
    assert obs["contains_image_data"] is False


# ------------------------------------------------------------------ 4: calibration numbers unchanged
def test_calibration_numbers_equal_the_pre_change_pipeline(rig):
    """Re-run the pre-F8.29 sequence directly (pair_detections -> resolve_pair_orderings -> calibrate_stereo) and compare every number."""
    left, right = rig["left"], rig["right"]
    pr = K.pair_detections(left, right, K.estimate_offset(left["motion"], right["motion"], left["fps"])["offset_frames"])
    pairs = [(a["corners"], b["corners"]) for a, b in pr["pairs"]]; ordr = uc.resolve_pair_orderings(pairs, PATTERN, max(SIZE))
    keep = [i for i, p in enumerate(ordr["pairs"]) if p["status"] == "ok"]
    ref = uc.calibrate_stereo([pairs[i][0] for i in keep], [ordr["pairs"][i]["right_corners"] for i in keep], PATTERN, SQ, SIZE, SIZE, pair_ids=[i + 1 for i in keep])
    new = rig["res"]["model"]
    # (a) the solver's INPUTS are bit-identical to the pre-change pipeline's
    for a, b in zip(rig["spy"]["left"], [pairs[i][0] for i in keep]): assert np.array_equal(a, np.asarray(b).reshape(-1, 2))
    for a, b in zip(rig["spy"]["right"], [ordr["pairs"][i]["right_corners"] for i in keep]): assert np.array_equal(a, np.asarray(b).reshape(-1, 2))
    assert rig["spy"]["pair_ids"] == [i + 1 for i in keep]
    # (b) its OUTPUTS agree to the solver's own run-to-run noise. Measured: two runs of the UNMODIFIED code (git HEAD) already differ by ~7e-8 in K/D/R/T
    #     because OpenCV's solver is not bit-reproducible between calls, so exact equality of outputs is not a meaningful check.
    tol = dict(rtol=1e-9, atol=1e-6)
    for k in ("K1", "D1", "K2", "D2", "R", "T", "E", "F"):
        assert np.allclose(ref[k], new[k], **tol), k
    for k in ("stereo_rms", "rms_left", "rms_right", "baseline_mm"):
        assert ref[k] == pytest.approx(new[k], rel=1e-9, abs=1e-6), k
    assert json.loads(json.dumps(ref["metrics"], default=str)).keys() == json.loads(json.dumps(new["metrics"], default=str)).keys()
    assert ref["metrics"]["epipolar_px"]["median"] == pytest.approx(new["metrics"]["epipolar_px"]["median"], rel=1e-9, abs=1e-6)
    assert ref["distortion_left"] == new["distortion_left"] and ref["distortion_right"] == new["distortion_right"]
    assert rig["res"]["pair_frames"] == [(pr["pairs"][i][0]["frame"], pr["pairs"][i][1]["frame"]) for i in keep]
    a = K.assess(rig["res"], None)
    assert a["level"] in (K.GOOD, K.QUESTIONABLE) and a["metrics"]["rms_stereo_px"] == pytest.approx(ref["stereo_rms"], rel=1e-9, abs=1e-6) and a["metrics"]["calibrated_baseline_mm"] == pytest.approx(ref["baseline_mm"], rel=1e-9, abs=1e-6)


# ------------------------------------------------------------------ run the REAL worker with synthetic detections (no video decoded)
@pytest.fixture(scope="module")
def worker_result(rig, tmp_path_factory, monkeypatch_module):
    d = tmp_path_factory.mktemp("job"); (d / "l.mov").write_bytes(b"not a video"); (d / "r.mov").write_bytes(b"not a video")
    dets = dict(L=rig["left"], R=rig["right"]); calls = []

    def fake_extract(path, pattern, stride=5, progress=None, max_frames=None):
        calls.append(Path(path).name); return dets["L"] if Path(path).name == "l.mov" else dets["R"]
    monkeypatch_module.setattr(K, "extract_detections", fake_extract)
    def no_video(*a, **k): raise AssertionError("a video must not be opened")
    monkeypatch_module.setattr(cv2, "VideoCapture", no_video)
    spec = dict(out_dir=str(d), left_path=str(d / "l.mov"), right_path=str(d / "r.mov"), f7_dir=str(MODULE / "visitor_workflow"), cols=PATTERN[0], rows=PATTERN[1],
                square_mm=SQ, baseline_mm=None, dump_detections=True)
    (d / "spec.json").write_text(json.dumps(spec))
    assert W.main(str(d / "spec.json")) == 0
    assert calls == ["l.mov", "r.mov"]
    return json.loads((d / "result.json").read_text())


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch(); yield mp; mp.undo()


def _doc(worker_result, code="abcdefghijklmnop0123456789"):
    res = dict(worker_result, mode="calibrate", outcome="completed")
    return RS.build_result_document(res, code=code, settings=dict(cols=PATTERN[0], rows=PATTERN[1], square_mm=SQ, baseline_mm=None))


def test_worker_emits_observations_and_not_the_raw_arrays_in_calibration(worker_result):
    assert worker_result["ok"] and worker_result["calibration_observations"]["schema"] == CO.OBS_SCHEMA
    assert "observations" not in worker_result["calibration"]
    assert worker_result["calibration_observations_status"]["available"] is True


def test_document_is_v2_validated_and_bound_to_its_calibration(worker_result):
    doc = _doc(worker_result)
    assert doc["schema"] == "f8.calibration_result/2" and doc["calibration_observations_status"]["available"] is True
    assert CO.validate_document_observations(doc)["n_pairs"] == worker_result["calibration"]["pairs_used"]
    assert doc["selected_poses"]["frame_pairs_left_right"] == [[p["left_frame"], p["right_frame"]] for p in doc["calibration_observations"]["pairs"]]
    blob = json.dumps(doc)
    assert "l.mov" not in blob and "r.mov" not in blob and "/job" not in blob and "abcdefghijklmnop0123456789" not in blob


# ------------------------------------------------------------------ 5: v2 round trip (store -> download bytes -> parse -> bridge)
def test_v2_round_trip_is_bit_exact_through_store_download_and_parse(worker_result, rig, tmp_path):
    store, code = RS.ResultStore(tmp_path / "r"), "abcdefghijklmnop0123456789"
    info = store.save_result(code, _doc(worker_result))
    data = store.result_bytes(code); assert len(data) == info["bytes"]
    parsed = CO.parse_result_document(data)                                # exactly what the download button serves
    L, R = CO.corner_arrays(parsed["observations"])
    o = rig["res"]["observations"]
    assert np.array_equal(L, np.asarray([np.asarray(x, np.float64).reshape(-1, 2) for x in o["left"]]))
    assert np.array_equal(R, np.asarray([np.asarray(x, np.float64).reshape(-1, 2) for x in o["right"]]))
    assert parsed["uncertainty"]["available"] is True
    h = B.parse_f8_result(data)                                           # the visitor workflow's importer accepts it and keeps the observations
    assert h.observations_available and h.observations()["n_pairs"] == parsed["observations"]["n_pairs"] and "no calibration uncertainty" in h.uncertainty_note
    assert h.can_accept(True) or h.can_accept(False) is False


def test_observations_do_not_change_the_f6_contract(worker_result):
    doc = _doc(worker_result); v2 = B.parse_f8_result(doc)
    stripped = copy.deepcopy(doc); stripped["calibration_observations"] = None
    assert B.parse_f8_result(stripped).level == v2.level
    c2 = {k: v for k, v in v2.accept(True).items()}
    assert c2 == B.parse_f8_result(stripped).accept(True)                  # observations are carried alongside, never part of the contract


# ------------------------------------------------------------------ 6-8: compatibility and missing observations
def _v1_doc():
    Kc = [[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]]
    return dict(schema="f8.calibration_result/1", job_id="x" * 12, mode="calibrate", outcome="completed", ok=True, settings=dict(cols=8, rows=6, square_mm=20, measured_baseline_mm=None),
                quality=dict(level="GOOD", headline="synthetic", problems=[], concerns=[], stereo_rms_px=0.7),
                cameras=dict(left=dict(K=Kc, image_size_wh=[1920, 1080]), right=dict(K=Kc, image_size_wh=[1920, 1080])), extrinsics=dict(baseline_mm=226.0),
                contract=dict(K1=Kc, D1=[0.0] * 5, K2=Kc, D2=[0.0] * 5, image_size_wh=[1920, 1080], capture_mode=dict(left="video", right="video"), baseline_mm=226.0))


def test_v1_json_still_loads_and_is_marked_uncertainty_unavailable():
    data = json.dumps(_v1_doc())
    p = CO.parse_result_document(data)
    assert p["observations"] is None and p["uncertainty"]["available"] is False and p["uncertainty"]["text"] == "calibration uncertainty unavailable"
    h = B.parse_f8_result(data)
    assert h.level == "GOOD" and h.observations_available is False and h.observations() is None and h.uncertainty_note == "calibration uncertainty unavailable"
    assert B.SCHEMA == "f8.calibration_result/1"


def test_v1_document_cannot_smuggle_observations():
    d = _v1_doc(); d["calibration_observations"] = {"schema": CO.OBS_SCHEMA}
    with pytest.raises(CO.ObservationError):
        CO.parse_result_document(json.dumps(d))
    with pytest.raises(B.HandoffError) as e:
        B.parse_f8_result(d)
    assert "INVALID_OBSERVATIONS" in e.value.codes


def test_unknown_result_schema_is_rejected():
    d = _v1_doc(); d["schema"] = "f8.calibration_result/3"
    with pytest.raises(CO.ObservationError):
        CO.parse_result_document(json.dumps(d))
    with pytest.raises(B.HandoffError):
        B.parse_f8_result(d)


def test_v2_without_observations_loads_as_unavailable(worker_result):
    wr = dict(worker_result, calibration_observations=None, calibration_observations_status=dict(available=False, reason="x"))
    doc = _doc(wr); assert doc["calibration_observations"] is None and doc["calibration_uncertainty"] == CO.UNAVAILABLE_TEXT
    p = CO.parse_result_document(json.dumps(doc))
    assert p["observations"] is None and p["uncertainty"]["available"] is False
    assert B.parse_f8_result(doc).observations_available is False


def test_a_result_with_no_observation_keys_at_all_is_unavailable_not_invented():
    doc = RS.build_result_document(dict(mode="calibrate", outcome="failed", ok=False, error="boom"), code="abcdefghijklmnop0123456789")
    assert doc["schema"] == "f8.calibration_result/2" and doc["calibration_observations"] is None and CO.uncertainty_status(doc)["available"] is False


def test_failed_assessment_keeps_no_observations(rig, tmp_path, monkeypatch):
    monkeypatch.setattr(K, "assess", lambda r, b=None: dict(level=K.FAILED, headline="x", problems=["bad"], concerns=[], metrics=None, advice=""))
    d = tmp_path; (d / "l.mov").write_bytes(b"x"); (d / "r.mov").write_bytes(b"x")
    monkeypatch.setattr(K, "extract_detections", lambda path, *a, **k: rig["left"] if Path(path).name == "l.mov" else rig["right"])
    (d / "s.json").write_text(json.dumps(dict(out_dir=str(d), left_path=str(d / "l.mov"), right_path=str(d / "r.mov"), f7_dir=str(MODULE / "visitor_workflow"), cols=8, rows=6, square_mm=SQ)))
    W.main(str(d / "s.json")); r = json.loads((d / "result.json").read_text())
    assert r["calibration_observations"] is None and "not retained" in r["calibration_observations_status"]["reason"]


def test_npz_export_and_import_are_unchanged_and_carry_no_observations(rig):
    model = rig["res"]["model"]
    data = cio.export_calibration(model)
    loaded = cio.load_calibration(data)
    assert np.array_equal(loaded["K1"], model["K1"]) and loaded["baseline_mm"] == pytest.approx(model["baseline_mm"], rel=1e-9)
    assert cio.FORMAT_VERSION == 1 and not any("observ" in k or "corner" in k for k in cio.SPEC)
    # the strict loader does not accept extra members, so the NPZ cannot (and must not) be extended silently
    import io
    f = dict(np.load(io.BytesIO(data), allow_pickle=False)); f["calibration_observations"] = np.zeros((1,))
    with pytest.raises(cio.CalibrationFileError):
        cio.validate_fields(f)


# ------------------------------------------------------------------ 9: malformed observations are rejected
def _good(worker_result):
    return _doc(worker_result)


def _mut(doc, fn):
    d = copy.deepcopy(doc); fn(d); return d


MUTATIONS = {
    "wrong obs schema": lambda d: d["calibration_observations"].update(schema="f8.calibration_observations/9"),
    "missing key": lambda d: d["calibration_observations"].pop("pairs"),
    "extra key": lambda d: d["calibration_observations"].update(extra=1),
    "pattern wrong": lambda d: d["calibration_observations"]["pattern"].update(cols=7),
    "pattern not int": lambda d: d["calibration_observations"]["pattern"].update(cols=8.0),
    "corner count": lambda d: d["calibration_observations"]["pairs"][0]["left_corners"].pop(),
    "corner shape": lambda d: d["calibration_observations"]["pairs"][0]["right_corners"].__setitem__(0, [1.0, 2.0, 3.0]),
    "nan": lambda d: d["calibration_observations"]["pairs"][0]["left_corners"].__setitem__(0, [float("nan"), 1.0]),
    "inf": lambda d: d["calibration_observations"]["pairs"][0]["left_corners"].__setitem__(0, [float("inf"), 1.0]),
    "string coord": lambda d: d["calibration_observations"]["pairs"][0]["left_corners"].__setitem__(0, ["1", "2"]),
    "outside image": lambda d: d["calibration_observations"]["pairs"][0]["left_corners"].__setitem__(0, [99999.0, 1.0]),
    "negative frame": lambda d: d["calibration_observations"]["pairs"][0].update(left_frame=-1),
    "bool frame": lambda d: d["calibration_observations"]["pairs"][0].update(right_frame=True),
    "float frame": lambda d: d["calibration_observations"]["pairs"][0].update(left_frame=3.5),
    "incomplete pair": lambda d: d["calibration_observations"]["pairs"][1].pop("right_corners"),
    "duplicate left frame": lambda d: d["calibration_observations"]["pairs"][1].update(left_frame=d["calibration_observations"]["pairs"][0]["left_frame"], pair_id=999),
    "bad square": lambda d: d["calibration_observations"].update(square_mm=-1.0),
    "n_pairs lie": lambda d: d["calibration_observations"].update(n_pairs=3),
    "image data flag": lambda d: d["calibration_observations"].update(contains_image_data=True),
    "edited corner (digest)": lambda d: d["calibration_observations"]["pairs"][0]["left_corners"][0].__setitem__(0, d["calibration_observations"]["pairs"][0]["left_corners"][0][0] + 0.001),
    "other calibration (K edited)": lambda d: d["cameras"]["left"]["K"][0].__setitem__(0, d["cameras"]["left"]["K"][0][0] + 1.0),
    "other calibration (T edited)": lambda d: d["extrinsics"]["T"][0].__setitem__(0, d["extrinsics"]["T"][0][0] + 1.0),
    "different frames recorded": lambda d: d["selected_poses"].update(frame_pairs_left_right=[[0, 0]] * len(d["selected_poses"]["frame_pairs_left_right"])),
    "settings square differs": lambda d: d["settings"].update(square_mm=30.0),
    "not an object": lambda d: d.update(calibration_observations=[1, 2, 3]),
}


@pytest.mark.parametrize("name", MUTATIONS)
def test_malformed_or_mismatched_observations_are_rejected(worker_result, name, tmp_path):
    bad = _mut(_good(worker_result), MUTATIONS[name])
    with pytest.raises(CO.ObservationError):
        CO.validate_document_observations(bad)
    with pytest.raises(CO.ObservationError):
        CO.parse_result_document(json.dumps(bad, allow_nan=True).replace("NaN", "null") if name in ("nan", "inf") else json.dumps(bad))
    if name not in ("nan", "inf"):                                         # the store refuses to write them (nothing partial is left behind)
        store = RS.ResultStore(tmp_path / "s")
        with pytest.raises(RS.ResultError):
            store.save_result("abcdefghijklmnop0123456789", bad)
        assert store.result_bytes("abcdefghijklmnop0123456789") is None


def test_nan_and_infinity_literals_in_a_file_are_rejected(worker_result):
    text = json.dumps(_good(worker_result)); i = text.index('"left_corners": [[') if '"left_corners": [[' in text else text.index('"left_corners":[[')
    for lit in ("NaN", "Infinity", "-Infinity"):
        bad = re.sub(r'(\"left_corners\": ?\[\[)[^,]+', lambda m: m.group(1) + lit, text, count=1)
        with pytest.raises(CO.ObservationError):
            CO.parse_result_document(bad)


def test_build_result_document_drops_invalid_observations_but_keeps_the_calibration(worker_result):
    wr = copy.deepcopy(worker_result); wr["calibration_observations"]["pairs"][0]["left_corners"][0] = [1.0]
    doc = _doc(wr)
    assert doc["calibration_observations"] is None and "rejected" in doc["calibration_observations_status"]["reason"]
    assert doc["cameras"]["left"]["K"] is not None and doc["contract"] is not None and CO.uncertainty_status(doc)["available"] is False


def test_garbage_files_are_rejected():
    for bad in (b"", b"[]", b"{", b"\xff\xfe", json.dumps({"schema": None}).encode()):
        with pytest.raises(CO.ObservationError):
            CO.parse_result_document(bad)


# ------------------------------------------------------------------ 10: size and pair-count limits
def _synthetic_obs(n_pairs, cols, rows, size=(4000, 3000), seed=0):
    rng = np.random.default_rng(seed); n = cols * rows
    pairs = [dict(pair_id=i + 1, left_frame=i * 5, right_frame=i * 5, left_corners=rng.uniform(1, 2900, (n, 2)).tolist(), right_corners=rng.uniform(1, 2900, (n, 2)).tolist()) for i in range(n_pairs)]
    o = dict(schema=CO.OBS_SCHEMA, pattern=dict(cols=cols, rows=rows), square_mm=20.0, image_size_left=list(size), image_size_right=list(size), n_pairs=n_pairs, corner_order="x",
             coordinate_space="x", precision="x", selection="x", contains_image_data=False, pairs=pairs, observations_sha256=CO.pairs_digest(pairs), calibration_sha256="0" * 64)
    return o


def test_pair_count_limit():
    CO.validate_observations(_synthetic_obs(CO.MAX_PAIRS, 8, 6))
    with pytest.raises(CO.ObservationError, match="number of pairs"):
        CO.validate_observations(_synthetic_obs(CO.MAX_PAIRS + 1, 8, 6))
    with pytest.raises(CO.ObservationError, match="number of pairs"):
        CO.validate_observations(_synthetic_obs(0, 8, 6))


def test_serialized_size_and_corner_count_limits():
    with pytest.raises(CO.ObservationError, match="limit"):
        CO.validate_observations(_synthetic_obs(CO.MAX_PAIRS, 20, 20))            # 64 pairs x 400 corners x full-precision floats > MAX_OBS_BYTES
    with pytest.raises(CO.ObservationError, match="corners"):
        CO.validate_observations(_synthetic_obs(2, 21, 20))                       # 420 corners > MAX_POINTS
    with pytest.raises(CO.ObservationError):
        CO.validate_observations(_synthetic_obs(2, 101, 3))                       # side > MAX_PATTERN_SIDE


def test_store_rejects_oversized_nested_arrays_before_writing(worker_result, tmp_path):
    big = _good(worker_result); o = _synthetic_obs(CO.MAX_PAIRS, 20, 20); big["calibration_observations"] = o
    store = RS.ResultStore(tmp_path / "big")
    with pytest.raises(RS.ResultError):
        store.save_result("abcdefghijklmnop0123456789", big)
    assert list((tmp_path / "big").glob("job_*/result.json")) == []
    small = RS.ResultStore(tmp_path / "small", max_bytes=10_000)                 # the store-wide document cap still applies
    with pytest.raises(RS.ResultError):
        small.save_result("abcdefghijklmnop0123456789", _good(worker_result))


def test_typical_overhead_is_bounded(worker_result, tmp_path):
    store = RS.ResultStore(tmp_path / "t"); doc = _good(worker_result)
    with_obs = store.save_result("abcdefghijklmnop0123456789", doc)["bytes"]
    without = store.save_result("bbcdefghijklmnop0123456789", dict(doc, calibration_observations=None))["bytes"]
    overhead = with_obs - without
    print(f"\nF8.29 overhead: {doc['calibration_observations']['n_pairs']} pairs x 48 corners: {overhead} bytes in the stored document "
          f"({len(json.dumps(doc['calibration_observations'], separators=(',', ':')))} compact); total document {with_obs} bytes of {store.max_bytes}")
    assert 0 < overhead < 1_000_000 and with_obs < store.max_bytes // 4


# ------------------------------------------------------------------ 11-13: no experiment constants, no video processing, gate unchanged
CHANGED = ["visitor_workflow/f7_calibration.py", "visitor_workflow/f8_calibration_bridge.py", "f8_chunk_poc/calibration_observations.py", "f8_chunk_poc/result_store.py", "f8_chunk_poc/calibration_worker.py"]


def test_no_experimental_constants_or_recipe_a_in_visitor_calibration():
    for rel in CHANGED:
        src = (MODULE / rel).read_text()
        code = "\n".join(l.split("#")[0] for l in src.splitlines())
        assert not re.search(r"\b22[56](\.0)?\b", code), rel                         # no 225 / 226 mm baseline
        for word in ("recipe_a", "Recipe-A", "Recipe A", "ground_truth", "frame_manifest", "method_j_experiment", "boundary_annotations"):
            assert word.lower() not in code.lower(), (rel, word)
    imports = " ".join(re.findall(r"^(?:import|from)\s+(\S+)", (MODULE / "f8_chunk_poc/calibration_observations.py").read_text(), re.M))
    assert not {"room_measurement", "cv2", "numpy"} & set(imports.split())          # top-level imports are standard library only


def test_observation_and_store_modules_cannot_open_or_decode_video():
    for rel in ("f8_chunk_poc/calibration_observations.py", "f8_chunk_poc/result_store.py", "visitor_workflow/f8_calibration_bridge.py"):
        src = (MODULE / rel).read_text()
        assert "cv2" not in src and "VideoCapture" not in src, rel


def test_room_processing_gate_is_unchanged(monkeypatch):
    import f8_visitor_service as S
    monkeypatch.setenv("ROOM_PROCESSING_ENABLED", "1"); monkeypatch.setenv("F8_ENABLE_ROOM_PROCESSING", "1")
    assert S.ROOM_PROCESSING_ENABLED is False
    assert re.search(r"^ROOM_PROCESSING_ENABLED = False\b", (MODULE / "visitor_workflow/f8_visitor_service.py").read_text(), re.M)


# ------------------------------------------------------------------ stale label (visitor_annotations.py)
def test_distortion_label_no_longer_claims_recipe_a():
    src = (MODULE / "room_measurement/visitor_annotations.py").read_text()
    assert "Recipe-A" not in src and "Recipe A" not in src
    line = next(l for l in src.splitlines() if "distortion_model=" in l)
    assert "calibration K, D supplied" in line and "cv2.undistortPoints" in line
    assert "ctx.K, ctx.D" in src and "undistort_pixels(dist, ctx.K, ctx.D)" in src            # the geometry still uses the supplied calibration (unchanged)

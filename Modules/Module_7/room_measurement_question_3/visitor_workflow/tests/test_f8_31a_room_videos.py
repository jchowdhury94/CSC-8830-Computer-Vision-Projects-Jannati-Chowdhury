"""F8.31A: four-video room upload + metadata validation.  SYNTHETIC fixtures only (tiny generated clips); no original recording is opened, nothing is measured."""
import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

import f8_visitor_service as S
import room_videos as RV
from calibration_runner import RunnerLimits
from chunk_store import ChunkStore, Limits, MiB, UploadError
from result_store import ResultStore
from test_f8_visitor_service import make_doc
from video_check import inspect_video

HERE = Path(__file__).resolve().parents[1]
SLOTS = RV.ROOM_UPLOAD_SLOTS
CAL = dict(image_size_wh=[160, 96], capture_mode=dict(left="video", right="video"))
CHUNK = 64 * 1024


def make_video(path, size=(160, 96), fps=30.0, n=40, seed=0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(seed)
    for i in range(n):
        f = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8); f[:, : (i * 3) % size[0]] = 90
        w.write(f)
    w.release()
    return Path(path)


def put(store, owner, slot, data: bytes, name="clip.mp4"):
    ack = store.begin(owner, name, len(data), CHUNK, fingerprint=f"{name}|{len(data)}|{hashlib.sha256(data).hexdigest()[:8]}", slot=slot)
    uid = ack["upload_id"]
    for i in range(0, len(data), CHUNK):
        b = data[i:i + CHUNK]; store.put_chunk(owner, uid, i // CHUNK, b, hashlib.sha256(b).hexdigest())
    return store.finish(owner, uid, hashlib.sha256(data).hexdigest())


@pytest.fixture(autouse=True)
def _paths(monkeypatch):
    import builtins
    monkeypatch.setattr(builtins, "HERE_S", str(HERE), raising=False)
    monkeypatch.syspath_prepend(str(HERE / "tests"))


@pytest.fixture
def rstore(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB, reserve_sibling=False, max_active_uploads=4), root=tmp_path / "rooms", slots=SLOTS); yield s; s.close()


@pytest.fixture
def cstore(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB), root=tmp_path / "chunks"); yield s; s.close()


def fake_inspector(table):
    """path -> inspection, keyed by file size (each synthetic clip has a distinct size)."""
    return lambda p: table[Path(p).stat().st_size]


def good_info(w=160, h=96, fps=30.0, n=40):
    return dict(ok=True, decodable=True, width=w, height=h, fps=fps, frame_count=n, duration_s=round(n / fps, 2), codec="FMP4", errors=[], issues=[])


def uploads_of(**kw):
    """slot -> upload dict with a given inspection."""
    return {s: dict(name=f"{s}.mp4", size_bytes=1000 + i, sha256=f"{i:064x}", state="complete", inspection=kw.get(s, good_info())) for i, s in enumerate(SLOTS) if kw.get(s, 1) is not None}


# ================================================================ 1. slot mapping
def test_slot_mapping_is_by_slot_not_by_name():
    assert RV.ROOM_UPLOAD_SLOTS == ("p1_left", "p1_right", "p2_left", "p2_right")
    assert {s: RV.SLOT_INFO[s][:2] for s in SLOTS} == {"p1_left": (1, "left"), "p1_right": (1, "right"), "p2_left": (2, "left"), "p2_right": (2, "right")}
    assert [RV.f6_key(s) for s in SLOTS] == ["placement_1_left", "placement_1_right", "placement_2_left", "placement_2_right"]
    assert RV.SLOT_INFO["p1_left"][2] == "Viewpoint 1 · Left camera video" and RV.SLOT_INFO["p2_right"][2] == "Viewpoint 2 · Right camera video"


def test_store_accepts_room_slots_and_calibration_store_does_not(rstore, cstore, tmp_path):
    v = make_video(tmp_path / "a.mp4").read_bytes()
    assert put(rstore, "own", "p2_left", v, name="right_camera_final.mp4")["state"] == "complete"      # the file NAME says "right"; the slot decides
    assert set(rstore.job_status("own")["slots"]) == {"p2_left"}
    with pytest.raises(UploadError):
        cstore.begin("own", "x.mp4", 10, CHUNK, slot="p1_left")
    with pytest.raises(UploadError):
        rstore.begin("own", "x.mp4", 10, CHUNK, slot="left")


# ================================================================ 2. missing videos / readiness
def test_empty_and_partial_sets_are_not_ready():
    rep = RV.assess_room_set({}, CAL)
    assert not rep.ready and len(rep.missing) == 4 and all(r.status == RV.MISSING for r in rep.slots.values())
    partial = uploads_of(p2_right=None)
    rep = RV.assess_room_set(partial, CAL)
    assert not rep.ready and rep.missing == ["Viewpoint 2 · Right camera video"]


def test_complete_valid_set_is_ready_and_has_unvalidated_label():
    rep = RV.assess_room_set(uploads_of(), CAL)
    assert rep.ready and all(r.status in (RV.OK, RV.WARN) for r in rep.slots.values())
    assert rep.unvalidated_resolution and any("UNVALIDATED" in w for w in rep.set_warnings)          # 160x96 is not a validated size; accepted but labelled
    rep2 = RV.assess_room_set(uploads_of(**{s: good_info(2160, 3840) for s in SLOTS}), dict(CAL, image_size_wh=[2160, 3840]))
    assert rep2.ready and not rep2.unvalidated_resolution


def test_uploading_file_is_pending_not_ready():
    u = uploads_of(); u["p1_left"]["state"] = "uploading"; u["p1_left"]["inspection"] = None
    rep = RV.assess_room_set(u, CAL)
    assert not rep.ready and rep.pending == ["Viewpoint 1 · Left camera video"]


# ================================================================ 3. duplicates
def test_duplicate_files_are_errors_for_both_slots():
    u = uploads_of(); u["p2_left"]["sha256"] = u["p1_left"]["sha256"]
    rep = RV.assess_room_set(u, CAL)
    assert not rep.ready and rep.slots["p1_left"].status == RV.ERROR and rep.slots["p2_left"].status == RV.ERROR
    assert "same file" in rep.slots["p1_left"].errors[0]


def test_store_refuses_the_same_file_in_two_slots(rstore, tmp_path):
    v = make_video(tmp_path / "a.mp4").read_bytes()
    put(rstore, "own", "p1_left", v)
    with pytest.raises(UploadError) as e:
        put(rstore, "own", "p2_right", v)
    assert e.value.code == "duplicate_file"


# ================================================================ 4. unsupported / 5. corrupt (real inspector, in its sandbox process)
def test_unsupported_format_text_file(tmp_path):
    p = tmp_path / "notes.mp4"; p.write_bytes(b"this is definitely not a video file at all, just text" * 10)     # an .mp4 NAME does not make it a video
    u = uploads_of(p1_left=inspect_video(p)); rep = RV.assess_room_set(u, CAL)
    assert rep.slots["p1_left"].status == RV.ERROR and "Unsupported" in rep.slots["p1_left"].errors[0] and not rep.ready


def test_corrupt_video_is_an_error(tmp_path):
    good = make_video(tmp_path / "g.mp4").read_bytes()
    p = tmp_path / "bad.mp4"; p.write_bytes(good[:200] + b"\x00" * 3000)
    rep = RV.assess_room_set(uploads_of(p1_left=inspect_video(p)), CAL)
    assert rep.slots["p1_left"].status == RV.ERROR and rep.slots["p1_left"].errors and not rep.ready


def test_real_valid_clip_passes_the_real_inspector(tmp_path):
    info = inspect_video(make_video(tmp_path / "ok.mp4"))
    rep = RV.assess_room_set(uploads_of(p1_left=info), CAL)
    assert rep.slots["p1_left"].status in (RV.OK, RV.WARN) and rep.slots["p1_left"].facts["width"] == 160 and rep.slots["p1_left"].facts["frame_count"] == 40


# ================================================================ 6/7. dimensions, frame rate, duration
def test_left_right_frame_size_mismatch_is_blocking_and_nothing_is_resized():
    rep = RV.assess_room_set(uploads_of(p1_right=good_info(w=192, h=108)), CAL)
    assert not rep.ready and any("different frame sizes" in m for m in rep.set_errors)
    assert rep.slots["p1_left"].facts["width"] == 160 and rep.slots["p1_right"].facts["width"] == 192          # facts untouched


def test_viewpoints_with_different_sizes_are_blocking():
    rep = RV.assess_room_set(uploads_of(p2_left=good_info(w=192, h=108), p2_right=good_info(w=192, h=108)), CAL)
    assert not rep.ready and any("do not all have the same frame size" in m for m in rep.set_errors)


def test_fps_mismatch_within_a_viewpoint_is_a_recommendation_only():
    rep = RV.assess_room_set(uploads_of(p1_right=good_info(fps=24.0)), CAL)
    assert rep.ready and any("different frame rates" in m for m in rep.set_warnings)
    rep = RV.assess_room_set(uploads_of(p1_right=good_info(fps=30.5)), CAL)                                  # within the existing F6 tolerance (1 fps): no warning
    assert not any("different frame rates" in m for m in rep.set_warnings)


def test_length_difference_alone_is_not_a_warning():
    rep = RV.assess_room_set(uploads_of(p1_right=good_info(n=300)), CAL)
    assert rep.ready and not any("length" in m for m in rep.set_warnings)


def test_f6_limits_are_reused_not_invented():
    assert RV.F6_LIMITS is RV.VV.DEFAULT_VIDEO_LIMITS and RV.FPS_MATCH_TOLERANCE == RV.VV.DEFAULT_VIDEO_LIMITS.max_fps_difference
    rep = RV.assess_room_set(uploads_of(p1_left=good_info(n=1)), CAL)
    assert rep.slots["p1_left"].status == RV.ERROR
    rep = RV.assess_room_set(uploads_of(p1_left=good_info(n=5000)), CAL)                                     # 166 s > 120 s
    assert rep.slots["p1_left"].status == RV.ERROR and "Duration" in rep.slots["p1_left"].errors[0]
    rep = RV.assess_room_set(uploads_of(p1_left=dict(good_info(), fps=None)), CAL)
    assert rep.slots["p1_left"].status == RV.ERROR


# ================================================================ 8. calibration / video mismatch
def test_resolution_mismatch_with_calibration_is_blocking():
    rep = RV.assess_room_set(uploads_of(**{s: good_info(192, 108) for s in SLOTS}), CAL)
    assert not rep.ready and all(r.status == RV.ERROR and "not rescaled" in r.errors[0] for r in rep.slots.values())


def test_orientation_mismatch_with_calibration_is_blocking():
    rep = RV.assess_room_set(uploads_of(**{s: good_info(96, 160) for s in SLOTS}), CAL)
    assert not rep.ready and all("portrait" in r.errors[0] and "Nothing is rotated" in r.errors[0] for r in rep.slots.values())


@pytest.mark.parametrize("modes", [dict(left="photo", right="photo"), dict(left="video"), {}])
def test_non_video_or_unknown_calibration_capture_mode_blocks(modes):
    rep = RV.assess_room_set(uploads_of(), dict(CAL, capture_mode=modes))
    assert not rep.ready and any("VIDEO" in m for m in rep.set_errors)


def test_no_calibration_means_nothing_is_ready():
    rep = RV.assess_room_set(uploads_of(), None)
    assert not rep.ready and "No accepted calibration" in rep.set_errors[0]


# ================================================================ service: storage, isolation, limits, cleanup, calibration context, navigation
@pytest.fixture
def svc(cstore, rstore, tmp_path):
    return S.VisitorCalibrationService(cstore, ResultStore(tmp_path / "results"), RunnerLimits(), room_store=rstore,
                                       inspector=lambda p: dict(good_info(1920, 1080), size=Path(p).stat().st_size))


def accepted(svc):
    s = S.VisitorSession(); s.review = S.review_f8_document(make_doc("GOOD")[1]); svc.accept(s); return s


def four_videos(tmp_path):
    return {s: make_video(tmp_path / f"{s}.mp4", seed=i + 1).read_bytes() for i, s in enumerate(SLOTS)}


def test_full_flow_with_continue_gate_and_calibration_context_preserved(svc, rstore, tmp_path):
    import copy
    s = accepted(svc); before = (copy.deepcopy(s.contract), s.accepted, s.f6)
    ctx = svc.calibration_context(s)
    assert ctx["image_size_wh"] == [1920, 1080] and ctx["baseline_mm"] == 226.0 and ctx["capture_mode"] == dict(left="video", right="video")
    vids = four_videos(tmp_path)
    for slot in SLOTS[:3]:
        put(rstore, s.owner, slot, vids[slot]); assert not svc.room_report(s).ready
    with pytest.raises(S.ServiceError):
        svc.go_frames(s)
    with pytest.raises(S.ServiceError):
        s.go("frames")
    put(rstore, s.owner, "p2_right", vids["p2_right"])
    rep = svc.room_report(s)
    assert rep.ready and s.room_ready
    svc.go_frames(s); assert s.step == "frames"
    s.go("room"); s.go("ready"); s.go("room")
    assert s.contract == before[0] and s.accepted is before[1] and s.f6 is before[2] and not s.f6.closed                         # untouched by uploading / navigating
    assert s.accepted.baseline_used_mm == 226.0 and s.accepted.image_size_wh == (1920, 1080)


def test_inspection_runs_once_per_upload_and_cache_holds_no_file_data(svc, rstore, tmp_path):
    calls = []; svc._inspector = lambda p: calls.append(p) or good_info(1920, 1080)
    s = accepted(svc); put(rstore, s.owner, "p1_left", make_video(tmp_path / "a.mp4").read_bytes())
    for _ in range(3):
        svc.room_report(s)
    assert len(calls) == 1 and all(isinstance(v, dict) and len(str(v)) < 2000 for v in s.room_inspect.values())
    put(rstore, s.owner, "p1_left", make_video(tmp_path / "b.mp4", seed=9, n=30).read_bytes())               # replacing the slot discards the old inspection
    svc.room_report(s); assert len(calls) == 2 and len(s.room_inspect) == 1


def test_new_calibration_attempt_invalidates_room_readiness(svc, rstore, tmp_path):
    s = accepted(svc)
    for slot, v in four_videos(tmp_path).items():
        put(rstore, s.owner, slot, v)
    assert svc.room_report(s).ready
    svc.import_saved(s, b"not json")                                                                           # forgets the acceptance
    assert not s.room_ready and svc.calibration_context(s) is None
    rep = svc.room_report(s); assert not rep.ready and "No accepted calibration" in rep.set_errors[0]
    with pytest.raises(S.ServiceError):
        s.go("room")                                                                                           # guided back: room page needs an accepted calibration


def test_visitor_storage_isolation(svc, rstore, tmp_path):
    a, b = accepted(svc), accepted(svc)
    v = make_video(tmp_path / "a.mp4").read_bytes()
    put(rstore, a.owner, "p1_left", v)
    assert set(svc.room_uploads(b)) == set() and set(svc.room_uploads(a)) == {"p1_left"}
    uid = rstore.job_status(a.owner)["slots"]["p1_left"]["upload_id"]
    with pytest.raises(UploadError):
        rstore.path_of(b.owner, uid)                                                                           # another visitor cannot reach the file
    with pytest.raises(UploadError):
        rstore.cancel(b.owner, uid)
    put(rstore, b.owner, "p1_left", v)                                                                         # the same bytes for another visitor are not a "duplicate"
    paths = {rstore.path_of(o.owner, rstore.job_status(o.owner)["slots"]["p1_left"]["upload_id"]) for o in (a, b)}
    assert len(paths) == 2 and all(str(p).startswith(str(rstore.root)) for p in paths) and rstore.root != svc.store.root
    assert not any(str(p).startswith(str(svc.store.root)) for p in paths)                                      # never in the calibration store


def test_upload_size_limit(tmp_path):
    s = ChunkStore(RV.room_limits(Limits(max_file_bytes=2 * MiB)), root=tmp_path / "r", slots=SLOTS)
    try:
        with pytest.raises(UploadError) as e:
            s.begin("o", "big.mp4", 3 * MiB, CHUNK, slot="p1_left")
        assert e.value.code == "file_too_large"
        with pytest.raises(UploadError):
            s.begin("o", "empty.mp4", 0, CHUNK, slot="p1_left")
    finally:
        s.close()


def test_room_limits_are_bounded_and_env_configurable(monkeypatch):
    base = RV.room_limits(Limits())
    assert base.reserve_sibling is False and base.max_disk_bytes == 4 * base.max_file_bytes
    monkeypatch.setenv("F8_ROOM_MAX_FILE_MB", "50")
    lim = RV.room_limits(Limits()); assert lim.max_file_bytes == 50 * MiB and lim.max_disk_bytes == 200 * MiB
    monkeypatch.setenv("F8_ROOM_MAX_FILE_MB", "junk"); assert RV.room_limits(Limits()).max_file_bytes == Limits().max_file_bytes


def test_temporary_files_are_cleaned_up(tmp_path):
    t = [0.0]
    s = ChunkStore(Limits(max_file_bytes=64 * MiB, reserve_sibling=False, complete_retention_s=100.0, idle_timeout_s=50.0), root=tmp_path / "r", slots=SLOTS, clock=lambda: t[0])
    try:
        v = make_video(tmp_path / "a.mp4").read_bytes()
        put(s, "o1", "p1_left", v)
        ack = s.begin("o2", "x.mp4", 200000, CHUNK, slot="p1_right"); half = b"\0" * CHUNK
        s.put_chunk("o2", ack["upload_id"], 0, half, hashlib.sha256(half).hexdigest())                          # an abandoned partial upload
        assert any(Path(s.root).rglob("data.*"))
        t[0] = 200.0; s.sweep()
        assert not any(Path(s.root).rglob("data.*")) and s.job_status("o1")["slots"] == {} and s.job_status("o2")["slots"] == {}
        put(s, "o3", "p2_left", v); assert s.release_owner("o3") == 1 and not any(Path(s.root).rglob("data.*"))
    finally:
        s.close()


def test_start_over_and_lock_delete_room_uploads_and_rotate_owner(svc, rstore, tmp_path):
    s = accepted(svc); old = s.owner
    for slot, v in four_videos(tmp_path).items():
        put(rstore, s.owner, slot, v)
    svc.room_report(s); assert s.room_ready and len(s.room_inspect) == 4 and any(Path(rstore.root).rglob("data.bin"))
    s.reset(svc.store, svc.room_store)
    assert not any(Path(rstore.root).rglob("data.bin")) and rstore.job_status(old)["slots"] == {} and s.owner != old
    assert s.room_ready is False and s.room_inspect == {} and s.step == "home" and s.accepted is None


def test_store_close_removes_the_temporary_root():
    s = ChunkStore(Limits(), slots=SLOTS, root_prefix="f8_room_videos_test_"); root = s.root
    assert root.name.startswith("f8_room_videos_test_") and root.exists()
    s.close(); assert not root.exists()


# ================================================================ 13. navigation (AppTest)
def test_room_page_without_calibration_guides_back():
    from streamlit.testing.v1 import AppTest
    import test_f8_26_room_navigation as T
    at = T._at(False).run(); assert not at.exception
    sess = at.session_state["f8i_session"]
    with pytest.raises(S.ServiceError):
        sess.go("room")
    sess.step = "room"; at.run()                                                                               # even if forced, the page refuses to offer uploads
    assert not at.exception and any("no accepted calibration" in w.value.lower() for w in at.warning)
    assert not any(b.key == "f8i_to_frames" for b in at.button)


def test_page_navigation_without_uploads_never_reaches_the_frames_page(svc):
    """F8.31B: the frames page is a real workflow now.  A forced ``room_ready`` flag is not trusted: the frames page re-validates through ``frame_state``, finds no uploads and offers only a way back."""
    import test_f8_26_room_navigation as T
    at = T._at(True).run()
    cont = next(b for b in at.button if b.key == "f8i_to_room"); cont.click(); at.run()
    sess = at.session_state["f8i_session"]; assert sess.step == "room" and not at.exception
    assert next(b for b in at.button if b.key == "f8i_to_frames").disabled                                    # missing uploads: Continue is disabled
    with pytest.raises(S.ServiceError):
        sess.go("frames")                                                                                     # and the session itself refuses
    sess.room_ready = True; sess.step = "frames"; at.run()                                                    # forced flag + no uploads
    assert not at.exception and any(("no longer usable" in e.value or "not all valid" in e.value) for e in at.error)
    assert not any(b.key == "f8i_to_frames" for b in at.button)
    assert not any(b.label.startswith(("Suggest frames", "Use these frames", "Start room measurement")) for b in at.button)  # no frame / reconstruction controls
    next(b for b in at.button if b.key == "f8i_nav_back").click(); at.run()
    assert sess.step == "room" and S.ROOM_PROCESSING_ENABLED is False


# ================================================================ 14. no original-data dependencies / 15. gate
def test_no_original_data_or_experiment_constants_in_the_new_code():
    for f in (HERE / "room_videos.py",):
        t = f.read_text()
        for bad in ("recipe_a", "Recipe A", "stage_c_integration", "stage_e1_integration", "validation_reference", ".MOV", "outputs/", "225", "226", "visitor_annotations"):
            assert bad not in t, (f.name, bad)
    pages = (HERE / "f8_visitor_pages.py").read_text()
    assert not re.search(r"\.MOV|/Users/|experiment_recordings", pages)


def test_validation_never_decodes_or_modifies_video_and_calls_no_f6_processing(svc, rstore, tmp_path, monkeypatch):
    import f6_core as F, subprocess
    for name in ("submit_videos", "select_frames", "measure", "triangulate", "detect_boundaries"):
        if hasattr(F.F6Session, name):
            monkeypatch.setattr(F.F6Session, name, lambda *a, **k: pytest.fail("F6 processing called"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("worker started"))
    s = accepted(svc); v = make_video(tmp_path / "a.mp4").read_bytes(); put(rstore, s.owner, "p1_left", v)
    svc.room_report(s); svc.room_report(s)
    assert rstore.path_of(s.owner, rstore.job_status(s.owner)["slots"]["p1_left"]["upload_id"]).read_bytes() == v      # byte-identical: nothing resized / re-encoded


def test_room_processing_gate_stays_disabled(svc, monkeypatch):
    monkeypatch.setenv("ROOM_PROCESSING_ENABLED", "1"); monkeypatch.setenv("F8_ENABLE_ROOM_PROCESSING", "1")
    assert S.ROOM_PROCESSING_ENABLED is False and svc.room_processing_enabled() is False
    src = (HERE / "f8_visitor_service.py").read_text()
    assert re.search(r"^ROOM_PROCESSING_ENABLED = False\b", src, re.M)
    assert "ROOM_PROCESSING_ENABLED" not in (HERE / "room_videos.py").read_text()                              # the gate lives in one place only
    assert "environ" not in src.split("ROOM_PROCESSING_ENABLED = False")[0].splitlines()[-1]

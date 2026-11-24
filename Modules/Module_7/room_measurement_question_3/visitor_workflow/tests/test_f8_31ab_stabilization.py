"""F8.31A-B: upload-size policy, calibration ownership and invalidation.  SYNTHETIC fixtures only (tiny generated clips, sparse placeholder files for size checks);
no original recording is opened and nothing is measured."""
import copy
import hashlib
from dataclasses import fields
from pathlib import Path

import pytest

import f8_visitor_service as S
import room_videos as RV
from calibration_runner import RunnerLimits
from chunk_store import ChunkStore, Limits, MiB, UploadError
from result_store import ResultStore
from test_f8_31a_room_videos import SLOTS, four_videos, good_info, make_video, put, uploads_of
from test_f8_visitor_service import make_doc

VV = RV.VV
GB_SIZE = 1024 * MiB


@pytest.fixture
def rstore(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB, reserve_sibling=False, max_active_uploads=4), root=tmp_path / "rooms", slots=SLOTS); yield s; s.close()


@pytest.fixture
def cstore(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB), root=tmp_path / "chunks"); yield s; s.close()


@pytest.fixture
def svc(cstore, rstore, tmp_path):
    return S.VisitorCalibrationService(cstore, ResultStore(tmp_path / "results"), RunnerLimits(), room_store=rstore,
                                       inspector=lambda p: dict(good_info(1920, 1080), size=Path(p).stat().st_size))


def accepted(svc, doc=None):
    s = S.VisitorSession(); s.review = S.review_f8_document((doc or make_doc("GOOD")[1])); svc.accept(s); return s


# ================================================================ size policy
def test_default_room_limit_is_1024_mb_and_f6_default_is_untouched(monkeypatch):
    monkeypatch.delenv("F8_ROOM_MAX_FILE_MB", raising=False)
    assert RV.room_file_limit_bytes() == 1024 * MiB
    assert VV.DEFAULT_VIDEO_LIMITS.max_file_bytes == 200 * MiB           # the F6 default is NOT changed: only the room path derives its own limits


def test_one_number_for_uploader_validation_and_f6_limits(monkeypatch):
    for mb in (None, "50", "300", "2048"):
        monkeypatch.delenv("F8_ROOM_MAX_FILE_MB", raising=False)
        if mb:
            monkeypatch.setenv("F8_ROOM_MAX_FILE_MB", mb)
        n = RV.room_file_limit_bytes()
        assert RV.room_limits(Limits()).max_file_bytes == n == RV.room_video_limits().max_file_bytes
        assert RV.room_limits(Limits()).max_disk_bytes == 4 * n


def test_room_f6_limits_differ_from_the_default_only_in_file_size(monkeypatch):
    monkeypatch.delenv("F8_ROOM_MAX_FILE_MB", raising=False)
    room, base = RV.room_video_limits(), VV.DEFAULT_VIDEO_LIMITS
    changed = {f.name for f in fields(base) if getattr(room, f.name) != getattr(base, f.name)}
    assert changed == {"max_file_bytes"}                                  # duration, frame size, fps, frame count: the existing F6 limits, unchanged


@pytest.mark.parametrize("value,expected_mb", [("0", 1024), ("-5", 1024), ("junk", 1024), ("nan", 1024), ("0.5", 1), ("999999", 2048), ("2048", 2048), ("300", 300)])
def test_override_is_clamped_and_bad_values_fall_back(monkeypatch, value, expected_mb):
    monkeypatch.setenv("F8_ROOM_MAX_FILE_MB", value)
    assert RV.room_file_limit_bytes() == int(expected_mb * MiB)
    assert RV.room_limits(Limits()).max_disk_bytes <= 4 * RV.ABSOLUTE_MAX_ROOM_FILE_MB * MiB          # total temporary storage stays bounded


def test_uploader_refuses_a_file_above_the_room_limit_before_any_byte_is_written(tmp_path, monkeypatch):
    monkeypatch.setenv("F8_ROOM_MAX_FILE_MB", "10")
    s = ChunkStore(RV.room_limits(Limits()), root=tmp_path / "r", slots=SLOTS)
    try:
        with pytest.raises(UploadError) as e:
            s.begin("o", "big.mp4", 10 * MiB + 1, 64 * 1024, slot="p1_left")
        assert e.value.code == "file_too_large" and not any(Path(s.root).rglob("data.*"))
    finally:
        s.close()


def test_a_file_between_200_and_1024_mb_is_no_longer_flagged_and_f6_agrees(tmp_path, monkeypatch):
    """What used to be a warning ('F6 will reject this later') is gone: the SAME limit is applied by F8 and by F6 (via room_video_limits)."""
    monkeypatch.delenv("F8_ROOM_MAX_FILE_MB", raising=False)
    big = 600 * MiB
    up = uploads_of(); up["p1_left"] = dict(up["p1_left"], size_bytes=big)
    rep = RV.assess_room_set(up, dict(image_size_wh=[160, 96], capture_mode=dict(left="video", right="video")))
    assert rep.slots["p1_left"].errors == [] and rep.slots["p1_left"].warnings == [] and rep.slots["p1_left"].status == RV.OK and rep.ready
    assert not any("MB" in m for m in rep.all_errors + rep.all_warnings)
    f = tmp_path / "sparse.mp4"                                           # sparse placeholder: no real disk use; not a video, only its SIZE is examined here
    with open(f, "wb") as fh:
        fh.truncate(big)
    codes_default = {c.code for c in VV.inspect_video(f, "left_video_1", VV.DEFAULT_VIDEO_LIMITS).checks}
    codes_room = {c.code for c in VV.inspect_video(f, "left_video_1", RV.room_video_limits()).checks}
    assert "FILE_TOO_LARGE" in codes_default and "FILE_TOO_LARGE" not in codes_room and "FILE_SIZE_OK" in codes_room


def test_a_file_above_the_room_limit_is_a_hard_error_not_ready(monkeypatch):
    monkeypatch.setenv("F8_ROOM_MAX_FILE_MB", "100")
    up = uploads_of(); up["p2_right"] = dict(up["p2_right"], size_bytes=101 * MiB)
    rep = RV.assess_room_set(up, dict(image_size_wh=[1920, 1080], capture_mode=dict(left="video", right="video")))
    assert rep.slots["p2_right"].status == RV.ERROR and any("100 MB" in m for m in rep.slots["p2_right"].errors) and not rep.ready


def test_downstream_f6_job_accepts_the_room_limits(monkeypatch):
    from room_measurement import visitor_measurement as VM
    monkeypatch.delenv("F8_ROOM_MAX_FILE_MB", raising=False)
    lim = VM.JobLimits(video=RV.room_video_limits())
    assert lim.video.max_file_bytes == GB_SIZE and lim.max_working_memory_mb == VM.JobLimits().max_working_memory_mb and lim.max_processing_s == VM.JobLimits().max_processing_s


def test_f6_never_loads_or_copies_whole_files():
    """The reason file size is an upload policy and not a decoder limit: no stage reads the whole file into memory or copies it."""
    root = Path(RV.__file__).resolve().parents[1]
    for p in (root / "room_measurement" / "visitor_video_validation.py", root / "room_measurement" / "visitor_frame_selection.py", root / "room_measurement" / "visitor_measurement.py",
              root / "visitor_workflow" / "f8_visitor_service.py", root / "visitor_workflow" / "room_videos.py"):
        src = p.read_text()
        for bad in ("read_bytes(", "shutil.copy", "copyfile(", "copyfileobj("):
            assert bad not in src, f"{p.name} uses {bad}"


# ================================================================ calibration ownership and invalidation
def _snapshot(s):
    return dict(contract=copy.deepcopy(s.contract), accepted={f.name: copy.deepcopy(getattr(s.accepted, f.name)) for f in fields(s.accepted)}, review=copy.deepcopy(getattr(s.review, "_contract", None)),
                provenance=copy.deepcopy(s.accepted.provenance))


def test_uploading_validating_replacing_and_navigating_never_mutate_the_calibration(svc, rstore, tmp_path):
    s = accepted(svc); before = _snapshot(s); c = s.contract
    assert c["baseline_mm"] == 226.0 and c["image_size_wh"] == [1920, 1080] and c["capture_mode"] == dict(left="video", right="video")
    assert c["K1"] and c["K2"] and c["D1"] is not None and c["D2"] is not None
    vids = four_videos(tmp_path)
    for slot, v in vids.items():
        put(rstore, s.owner, slot, v); svc.room_report(s)
    assert svc.room_report(s).ready
    put(rstore, s.owner, "p1_left", make_video(tmp_path / "again.mp4", seed=77, n=30).read_bytes())          # replace one video
    svc.room_report(s); svc.go_frames(s); s.go("room"); s.go("ready")
    assert _snapshot(s) == before and s.contract is c


def test_replacing_a_room_video_triggers_revalidation_against_the_calibration(rstore, cstore, tmp_path):
    calls = []
    table = {}

    def inspector(p):
        calls.append(p); return table.get(Path(p).stat().st_size, good_info(1920, 1080))
    svc = S.VisitorCalibrationService(cstore, ResultStore(tmp_path / "results"), RunnerLimits(), room_store=rstore, inspector=inspector)
    s = accepted(svc)
    for slot, v in four_videos(tmp_path).items():
        put(rstore, s.owner, slot, v)
    assert svc.room_report(s).ready and len(calls) == 4
    bad = make_video(tmp_path / "bad.mp4", seed=99, n=33).read_bytes(); table[len(bad)] = good_info(1080, 1920)      # the replacement is the wrong orientation
    put(rstore, s.owner, "p2_left", bad)
    rep = svc.room_report(s)
    assert len(calls) == 5 and not rep.ready and not s.room_ready and rep.slots["p2_left"].status == RV.ERROR and rep.slots["p1_left"].status in (RV.OK, RV.WARN)
    with pytest.raises(S.ServiceError):
        svc.go_frames(s)


def test_a_new_calibration_with_a_different_resolution_blocks_existing_videos(svc, rstore, tmp_path):
    s = accepted(svc)
    for slot, v in four_videos(tmp_path).items():
        put(rstore, s.owner, slot, v)
    assert svc.room_report(s).ready
    _, doc = make_doc("GOOD"); doc["contract"]["image_size_wh"] = [3840, 2160]
    for side in ("left", "right"):
        doc["cameras"][side]["image_size_wh"] = [3840, 2160]
    svc.import_saved(s, __import__("json").dumps(doc).encode())                                                # a NEW attempt: forgets the old acceptance
    assert not s.room_ready and s.accepted is None and svc.calibration_context(s) is None and not svc.room_report(s).ready
    assert len(rstore.job_status(s.owner)["slots"]) == 4                                                        # uploads stay until retention ...
    svc.accept(s)
    rep = svc.room_report(s)                                                                                    # ... and are re-validated against whichever calibration is accepted next
    assert not rep.ready and all(r.status == RV.ERROR and "not rescaled" in " ".join(r.errors) for r in rep.slots.values())


def test_expired_room_uploads_remove_readiness(cstore, tmp_path):
    t = [0.0]
    rs = ChunkStore(Limits(max_file_bytes=64 * MiB, reserve_sibling=False, max_active_uploads=4, complete_retention_s=100.0), root=tmp_path / "rooms", slots=SLOTS, clock=lambda: t[0])
    try:
        svc = S.VisitorCalibrationService(cstore, ResultStore(tmp_path / "results"), RunnerLimits(), room_store=rs, inspector=lambda p: dict(good_info(1920, 1080)))
        s = accepted(svc)
        for slot, v in four_videos(tmp_path).items():
            put(rs, s.owner, slot, v)
        assert svc.room_report(s).ready
        t[0] = 500.0; rs.sweep()
        rep = svc.room_report(s)
        assert not rep.ready and not s.room_ready and len(rep.missing) == 4 and s.accepted is not None          # videos gone, calibration still the visitor's
        with pytest.raises(S.ServiceError):
            svc.go_frames(s)
    finally:
        rs.close()


# ================================================================ slots and visitor isolation
def test_slot_assignment_follows_the_slot_not_the_file_name_or_upload_order(svc, rstore, tmp_path):
    s = accepted(svc)
    order = ("p2_right", "p1_left", "p2_left", "p1_right")                                                      # uploaded out of order, with misleading names
    datas = four_videos(tmp_path)
    for slot in order:
        put(rstore, s.owner, slot, datas[slot], name={"p1_left": "right.mp4", "p1_right": "left.mp4", "p2_left": "viewpoint1.mp4", "p2_right": "p1_left.mp4"}[slot])
    ups = svc.room_uploads(s)
    assert set(ups) == set(SLOTS) and {k: v["sha256"] for k, v in ups.items()} == {k: hashlib.sha256(d).hexdigest() for k, d in datas.items()}
    first = hashlib.sha256(datas["p1_left"]).hexdigest()
    put(rstore, s.owner, "p1_left", make_video(tmp_path / "n.mp4", seed=123, n=31).read_bytes())               # replacing one slot leaves the other three in place
    ups2 = svc.room_uploads(s)
    assert ups2["p1_left"]["sha256"] != first and all(ups2[k]["sha256"] == ups[k]["sha256"] for k in SLOTS if k != "p1_left")


def test_start_over_and_lock_clean_only_the_right_visitor(svc, rstore, tmp_path):
    a, b = accepted(svc), accepted(svc)
    for slot, v in four_videos(tmp_path).items():
        put(rstore, a.owner, slot, v)
    put(rstore, b.owner, "p1_left", make_video(tmp_path / "b.mp4", seed=5).read_bytes())
    svc.room_report(a); svc.room_report(b)
    a_old, b_old = a.owner, b.owner; b_before = copy.deepcopy(b.contract)
    a.reset(svc.store, svc.room_store)                                                                          # Start over (Lock uses the same reset)
    assert rstore.job_status(a_old)["slots"] == {} and a.owner != a_old
    assert set(rstore.job_status(b_old)["slots"]) == {"p1_left"} and b.owner == b_old and b.contract == b_before and b.accepted is not None
    assert set(svc.room_uploads(b)) == {"p1_left"} and svc.room_uploads(a) == {}

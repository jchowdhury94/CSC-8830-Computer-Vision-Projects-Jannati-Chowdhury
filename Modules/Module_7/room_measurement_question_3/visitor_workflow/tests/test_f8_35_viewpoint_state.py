"""Step 1 of the three-viewpoint Wall B integration: per-wall viewpoint count, six upload slots, isolation of the two walls, and the page.  SYNTHETIC clips and INJECTED fake results only."""
from pathlib import Path

import pytest

import f8_measurement as M
import f8_viewpoints as VPC
import f8_visitor_service as S
import room_videos as RV
from calibration_runner import RunnerLimits
from chunk_store import ChunkStore, Limits, MiB
from result_store import ResultStore
from test_f8_31a_room_videos import put
from test_f8_31b_frames import PEAK, accepted, make_clip
from test_f8_31b_pages import button, has_button, page_app, texts
from test_f8_31c2a_measurement_interface import ok_result, run_to_completion, use_measurement
from video_check import inspect_video


@pytest.fixture(scope="module")
def clips6(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips6"); out = {}
    for k, slot in enumerate(RV.ROOM_UPLOAD_SLOTS_3):
        out[slot] = make_clip(d / f"{slot}.mp4", slot if slot in PEAK else slot.replace("p3", "p1"), k)          # distinct scene per slot, so no two files are identical
    return out, {p.stat().st_size: dict(inspect_video(p), size=p.stat().st_size) for p in out.values()}


@pytest.fixture
def svc6(tmp_path, clips6):
    cs = ChunkStore(Limits(max_file_bytes=64 * MiB), root=tmp_path / "chunks")
    rs = ChunkStore(Limits(max_file_bytes=64 * MiB, reserve_sibling=False, max_active_uploads=12), root=tmp_path / "rooms", slots=RV.ALL_ROOM_SLOTS)
    table = clips6[1]
    sv = S.VisitorCalibrationService(cs, ResultStore(tmp_path / "results"), RunnerLimits(), room_store=rs, inspector=lambda p: table[Path(p).stat().st_size], frame_workspace_root=tmp_path)
    sv._test_sessions = []
    yield sv
    for sess in sv._test_sessions:
        sess.reset(cs, rs)
    cs.close(); rs.close()


def upload(svc, sess, clips6, slots):
    for s in slots:
        put(svc.room_store, sess.room_owner, s, clips6[0][s].read_bytes(), name=clips6[0][s].name)


# ------------------------------------------------------------------ defaults and compatibility
def test_two_viewpoints_is_the_default_everywhere_and_the_original_slots_are_unchanged(svc6):
    assert RV.ROOM_UPLOAD_SLOTS == ("p1_left", "p1_right", "p2_left", "p2_right")
    assert RV.slots_for(2) == RV.ROOM_UPLOAD_SLOTS and RV.slots_for(3) == RV.ALL_ROOM_SLOTS and len(RV.ALL_ROOM_SLOTS) == 6
    s = accepted(svc6)
    for w in ("A", "B"):
        s.switch_wall(w); assert s.n_viewpoints == 2
    s.switch_wall("A")
    with pytest.raises(ValueError):
        RV.slots_for(4)


def test_a_two_viewpoint_wall_ignores_a_stray_third_viewpoint_upload(svc6, clips6):
    s = accepted(svc6); upload(svc6, s, clips6, RV.ROOM_UPLOAD_SLOTS_3)
    assert set(svc6.room_uploads(s)) == set(RV.ROOM_UPLOAD_SLOTS) and set(svc6.room_sources(s)) == set(RV.ROOM_UPLOAD_SLOTS)
    rep = svc6.room_report(s); assert rep.ready and len(rep.slots) == 4


# ------------------------------------------------------------------ the viewpoint count rules
def test_wall_a_cannot_use_three_viewpoints_and_nothing_changes(svc6, clips6):
    s = accepted(svc6); upload(svc6, s, clips6, RV.ROOM_UPLOAD_SLOTS)
    before = dict(svc6.room_store.job_status(s.room_owner)["slots"])
    with pytest.raises(S.ServiceError, match="validated"):
        svc6.set_viewpoint_count(s, "A", 3)
    assert s.n_viewpoints == 2 and svc6.room_store.job_status(s.room_owner)["slots"] == before
    assert VPC.refusal("A", 3) and VPC.refusal("B", 3) is None and VPC.refusal("B", 4) and not VPC.allowed("A", 3) and VPC.allowed("B", 3) and VPC.allowed("A", 2)
    assert "height" in VPC.WALL_A_THREE_MISSING and "validated" in VPC.WALL_A_THREE_MISSING


def test_wall_b_three_viewpoints_needs_all_six_videos(svc6, clips6):
    s = accepted(svc6); svc6.set_viewpoint_count(s, "B", 3); s.switch_wall("B")
    assert s.n_viewpoints == 3 and s.active_wall == "B"
    upload(svc6, s, clips6, RV.ROOM_UPLOAD_SLOTS)
    rep = svc6.room_report(s); assert not rep.ready and not s.room_ready and len(rep.missing) == 2 and "Viewpoint 3" in " ".join(rep.missing)
    with pytest.raises(S.ServiceError, match="six"):
        svc6.go_frames(s)
    upload(svc6, s, clips6, ("p3_left", "p3_right"))
    rep = svc6.room_report(s); assert rep.ready and s.room_ready and len(rep.slots) == 6 and rep.n_valid == 6
    assert set(svc6.room_sources(s)) == set(RV.ROOM_UPLOAD_SLOTS_3) and all(v is not None for v in svc6.room_sources(s).values())


def test_identical_files_among_six_are_rejected_by_the_store(svc6, clips6):
    from chunk_store import UploadError
    s = accepted(svc6); svc6.set_viewpoint_count(s, "B", 3); s.switch_wall("B")
    upload(svc6, s, clips6, RV.ROOM_UPLOAD_SLOTS_3[:-1])
    with pytest.raises(UploadError) as e:
        put(svc6.room_store, s.room_owner, "p3_right", clips6[0]["p1_left"].read_bytes(), name="dup.mp4")
    assert e.value.code == "duplicate_file" and not svc6.room_report(s).ready


def test_viewpoint_count_is_kept_per_wall_through_navigation(svc6):
    s = accepted(svc6); svc6.set_viewpoint_count(s, "B", 3)
    for w in ("B", "A", "B", "A"):
        s.switch_wall(w); assert s.n_viewpoints == (3 if w == "B" else 2)
    s.switch_wall("B"); s.go("room"); assert s.n_viewpoints == 3
    s.switch_wall("A"); assert s.n_viewpoints == 2


def test_changing_wall_b_discards_only_wall_b_work_and_keeps_wall_a_and_the_calibration(svc6, clips6):
    s = accepted(svc6); upload(svc6, s, clips6, RV.ROOM_UPLOAD_SLOTS)                       # Wall A: four videos + a result
    use_measurement(svc6, runner=lambda ctx: ok_result(width=2850.0, height=2360.0)); s.measurement.result = ok_result(width=2850.0, height=2360.0)
    contract, acc = s.contract, s.accepted; a_uploads = dict(svc6.room_store.job_status(s.room_owner)["slots"]); a_result = s.measurement.result
    s.switch_wall("B"); upload(svc6, s, clips6, RV.ROOM_UPLOAD_SLOTS); b_owner = s.room_owner
    s.measurement.result = ok_result(width=3400.0, height=2400.0); s.switch_wall("A")
    notes = svc6.set_viewpoint_count(s, "B", 3)
    assert notes and "Wall B" in notes[0] and s.active_wall == "A"                           # the active wall is not changed by changing the other wall's setting
    assert svc6.room_store.job_status(s.room_owner)["slots"] == a_uploads and s.measurement.result is a_result and s.n_viewpoints == 2
    assert s.contract is contract and s.accepted is acc and s.f6 is not None
    s.switch_wall("B")
    assert s.n_viewpoints == 3 and svc6.room_store.job_status(b_owner)["slots"] == {} and s.measurement.result is None and not s.room_ready
    assert all(s.room_slot_epoch[k] > 500 for k in RV.ALL_ROOM_SLOTS)                        # every Wall B uploader widget resets
    assert svc6.set_viewpoint_count(s, "B", 3) == []                                         # choosing the same count again discards nothing


def test_switching_back_to_two_viewpoints_discards_the_six_video_set(svc6, clips6):
    s = accepted(svc6); svc6.set_viewpoint_count(s, "B", 3); s.switch_wall("B"); upload(svc6, s, clips6, RV.ROOM_UPLOAD_SLOTS_3)
    assert svc6.room_report(s).ready
    svc6.set_viewpoint_count(s, "B", 2)
    assert s.n_viewpoints == 2 and svc6.room_store.job_status(s.room_owner)["slots"] == {} and not svc6.room_report(s).ready and len(svc6.room_report(s).slots) == 4


def test_a_running_measurement_blocks_a_count_change(svc6):
    import threading
    s = accepted(svc6); s.switch_wall("B"); ev = threading.Event()
    ms = use_measurement(svc6, runner=lambda ctx: (ev.wait(10), ok_result())[1])
    s.measurement.job = M.MeasurementJob("j", {}, 0.0, threading.Event())
    with pytest.raises(S.ServiceError, match="running"):
        svc6.set_viewpoint_count(s, "B", 3)
    assert s.n_viewpoints == 2
    s.measurement.job.done = True; ev.set()


def test_start_over_and_calibration_change_behave(svc6):
    s = accepted(svc6); svc6.set_viewpoint_count(s, "B", 3)
    s.forget_acceptance(); s.switch_wall("B"); assert s.n_viewpoints == 3                      # a new calibration does not change the visitor's choice of method
    s.reset(svc6.store, svc6.room_store)
    for w in ("A", "B"):
        s.switch_wall(w); assert s.n_viewpoints == 2


# ------------------------------------------------------------------ the page
def test_page_shows_wall_a_option_disabled_and_labelled_not_yet_validated(svc6):
    s = accepted(svc6); s.guide.room.confirmed = {1: True, 2: True}; s.step = "room"
    at = page_app(svc6, s); at.run(); assert not at.exception, at.exception
    r = [x for x in at.radio if x.key == "f8i_nvp"]; assert r and r[0].disabled and r[0].value == 2
    assert any(VPC.NOT_VALIDATED in str(o) for o in r[0].options) and VPC.NOT_VALIDATED in texts(at) and "videos" in texts(at)
    assert "Four Wall A Videos" in " ".join(h.value for h in at.subheader)


def test_page_wall_b_switches_to_six_slots_without_asking_when_nothing_would_be_lost(svc6):
    s = accepted(svc6); s.switch_wall("B"); s.step = "room"
    at = page_app(svc6, s); at.run(); assert not at.exception, at.exception
    r = [x for x in at.radio if x.key == "f8i_nvp_b"][0]; assert not r.disabled and r.value == 2
    assert "Upload Your Four Wall B Videos" in " ".join(h.value for h in at.subheader)
    r.set_value(3).run(); assert not at.exception, at.exception
    assert s.n_viewpoints == 3 and [x for x in at.radio if x.key == "f8i_nvp_b"][0].value == 3
    assert "Upload Your Six Wall B Videos" in " ".join(h.value for h in at.subheader) and "Viewpoint 3 · right end" in texts(at) and "Viewpoint 2 · middle" in texts(at)


def test_page_asks_before_discarding_wall_b_work(svc6, clips6):
    s = accepted(svc6); s.switch_wall("B"); upload(svc6, s, clips6, RV.ROOM_UPLOAD_SLOTS); s.step = "room"
    at = page_app(svc6, s); at.run()
    [x for x in at.radio if x.key == "f8i_nvp_b"][0].set_value(3).run(); assert not at.exception, at.exception
    assert s.n_viewpoints == 2 and "Switch Wall B to 3 viewpoints?" in texts(at) and len(svc6.room_store.job_status(s.room_owner)["slots"]) == 4     # nothing is discarded yet
    button(at, "Keep the current choice").click().run(); assert s.n_viewpoints == 2 and not has_button(at, "Switch and discard")
    [x for x in at.radio if x.key == "f8i_nvp_b"][0].set_value(3).run(); button(at, "Switch and discard").click().run()
    assert s.n_viewpoints == 3 and svc6.room_store.job_status(s.room_owner)["slots"] == {}

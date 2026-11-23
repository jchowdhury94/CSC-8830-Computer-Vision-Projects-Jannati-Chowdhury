"""F8.31E: the × control on uploaded room videos, no retention dropdowns, and the simplified Suggest -> Review -> Accept frame selection.  SYNTHETIC clips only."""
from pathlib import Path

import pytest

import f8_frames_page as FP
import f8_visitor_pages as P
import f8_visitor_service as S
import room_frames as RF
import room_videos as RV
from chunk_store import UploadError
from test_f8_31b_frames import accepted, annotate_all, choose_and_confirm, clips, complete, cstore, ready, rstore, svc, upload_all   # noqa: F401  (fixtures)
from test_f8_31b_pages import button, has_button, page_app, texts


def _room_session(svc, clips):
    s = accepted(svc); s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True); upload_all(svc, s, clips); s.step = "room"
    return s


def _slots(svc, s):
    return set(svc.room_store.job_status(s.owner)["slots"])


# ------------------------------------------------------------------ A: removal (service)
def test_removal_deletes_only_that_video_and_keeps_calibration_and_the_others(svc, clips):
    s = _room_session(svc, clips); assert svc.room_report(s).ready and len(_slots(svc, s)) == 4
    path = svc.room_store.path_of(s.owner, svc.room_store.job_status(s.owner)["slots"]["p1_right"]["upload_id"])
    assert svc.remove_room_video(s, "p1_right") is True
    assert _slots(svc, s) == {"p1_left", "p2_left", "p2_right"} and not path.exists()
    rep = svc.room_report(s)
    assert rep.n_valid == 3 and not rep.ready and not s.room_ready
    assert s.accepted is not None and s.contract is not None and svc.calibration_context(s) is not None
    assert svc.remove_room_video(s, "p1_right") is False                         # nothing left to remove: harmless


def test_a_session_can_never_remove_or_see_another_sessions_video(svc, clips):
    a = _room_session(svc, clips); b = accepted(svc)
    uid = svc.room_store.job_status(a.owner)["slots"]["p1_left"]["upload_id"]
    with pytest.raises(UploadError):
        svc.room_store.cancel(b.owner, uid)                                       # the store's own owner check
    assert svc.remove_room_video(b, "p1_left") is False                           # B has nothing in that slot; A's file is untouched
    assert _slots(svc, a) == set(RV.ROOM_UPLOAD_SLOTS) and svc.room_store.path_of(a.owner, uid).is_file()
    with pytest.raises(S.ServiceError):
        svc.remove_room_video(a, "../../etc/passwd")


def test_removal_invalidates_only_the_dependent_viewpoint_and_the_result(complete, svc):
    s, wf = complete
    assert svc.removal_impact(s, "p1_left")                                       # significant work exists: the page must ask first
    s.measurement.result = object(); s.measurement.binding = object()
    svc.remove_room_video(s, "p1_left")
    assert s.measurement.result is None
    assert not any(v == 1 for (v, _) in wf.pairs) and any(v == 2 for (v, _) in wf.pairs)      # Viewpoint 1 discarded, Viewpoint 2 kept
    assert not wf.readiness().ready and not s.room_ready
    assert any("Viewpoint 1" in n for n in s.frame_notes)


def test_removal_impact_is_empty_without_downstream_work(svc, clips):
    s = _room_session(svc, clips)
    assert svc.removal_impact(s, "p2_left") == []


def test_removing_one_video_resets_only_that_slots_widget(svc, clips):
    s = _room_session(svc, clips); before = dict(s.room_slot_epoch)
    svc.remove_room_video(s, "p2_left")
    assert s.room_slot_epoch["p2_left"] == before["p2_left"] + 1 and all(s.room_slot_epoch[k] == before[k] for k in before if k != "p2_left")


# ------------------------------------------------------------------ A: removal (page)
def test_x_control_exists_only_for_uploaded_videos_and_updates_the_counter(svc, clips):
    s = accepted(svc); s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True); s.step = "room"
    at = page_app(svc, s); at.run(); assert not at.exception and not [b for b in at.button if b.label == "×"]
    upload_all(svc, s, clips, only=["p1_left", "p1_right", "p2_left", "p2_right"]); at = page_app(svc, s); at.run()
    xs = [b for b in at.button if b.label == "×"]; assert len(xs) == 4 and not button(at, "Continue to Frame Selection").disabled
    xs[0].click().run(); assert not at.exception
    assert len(_slots(svc, s)) == 3 and any("3 of 4" in str(p.proto.text) for p in at.get("progress")) and button(at, "Continue to Frame Selection").disabled
    assert len([b for b in at.button if b.label == "×"]) == 3 and s.accepted is not None


def test_x_asks_before_discarding_significant_work(complete, svc):
    s, wf = complete; s.step = "room"; s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True)
    at = page_app(svc, s); at.run(); assert not at.exception
    [b for b in at.button if b.label == "×"][0].click().run()
    assert len(_slots(svc, s)) == 4 and "cannot be undone" in texts(at) and "frame pairs" in texts(at)      # nothing deleted yet
    button(at, "Keep it").click().run(); assert len(_slots(svc, s)) == 4 and "cannot be undone" not in texts(at)
    [b for b in at.button if b.label == "×"][0].click().run(); button(at, "Remove and discard").click().run()
    assert not at.exception and len(_slots(svc, s)) == 3 and not any(v == 1 for (v, _) in wf.pairs)


def test_the_uploader_identity_is_stable_and_x_adds_nothing_before_it():
    src = Path(P.__file__).read_text()
    assert 'key=f"f8i_room_up_{slot}{W.suffix(sess)}"' in src and "def _room_slot_card" in src
    body = src[src.index("def _room_slot_card"):src.index("def _room_retention")]
    assert body.index("component.chunk_uploader") < body.index("f8i_room_rm_")                     # × comes AFTER the uploader inside the card
    assert "f8i_room_up_{slot}_" not in body and "epoch" in body                                  # the key never changes; only the epoch argument does


# ------------------------------------------------------------------ B: retention dropdowns
def test_no_retention_dropdown_anywhere_and_backend_policy_unchanged(svc, clips):
    for f in ("f8_visitor_pages.py", "f8_frames_page.py", "f8_measure_page.py", "f8_ui_text.py", "f8_guided_text.py"):
        assert "How long are my videos kept" not in (Path(P.__file__).parent / f).read_text(), f
    s = _room_session(svc, clips); at = page_app(svc, s); at.run()
    assert not any("How long" in str(e.label) for e in at.expander) and "15 minutes" in texts(at)        # concise line instead
    lim = svc.room_store.limits; assert lim.complete_retention_s == 900.0 and lim.idle_timeout_s == 300.0


# ------------------------------------------------------------------ C: suggest -> review -> accept
def _frames_app(svc, s):
    s.step = "frames"; at = page_app(svc, s); at.run(); assert not at.exception, at.exception; return at


def test_default_frame_page_is_automatic_and_hides_technical_controls(ready, svc):
    s, wf = ready; at = _frames_app(svc, s); t = texts(at)
    assert not has_button(at, "Suggest Frames") and [x.value for x in at.subheader] == ["Part 1 · Select Frames Automatically", "Part 2 · Confirm Wall Boundaries", "Part 3 · Review and Continue"]
    assert "5 frame sets" in t and "Locked. Confirm all six wall boundaries first." in t
    for hidden in ("Number of frame sets", "Edit frame set", "Use these frames", "Offset"):
        assert hidden not in t and not has_button(at, hidden)
    assert not at.select_slider and wf.frames_ready()


def test_view_selected_frames_is_optional_and_shows_all_twenty(ready, svc):
    s, wf = ready; at = _frames_app(svc, s)
    n0 = len(at.get("image"))                                                       # the single picture of the first boundary screen; the 20 previews are not drawn unless asked for
    assert n0 <= 1
    at.toggle(key="f8i_fs_view").set_value(True).run(); assert not at.exception, at.exception
    caps = " ".join(str(i.proto) for i in at.get("image"))
    assert len(at.get("image")) == n0 + 20 and "Viewpoint 1 · Left camera" in caps and "Viewpoint 2 · Right camera" in caps and "frame set 1" in caps


def test_change_frames_reveals_the_manual_controls_and_a_change_needs_reconfirming(ready, svc):
    s, wf = ready; at = _frames_app(svc, s)
    assert wf.all_pairs_confirmed() and not has_button(at, "Use these frames")
    at.toggle(key="f8i_fs_manual").set_value(True).run(); assert not at.exception
    assert has_button(at, "Use these frames") and at.select_slider
    cur = dict(wf.pairs[(1, 0)]); button(at, "+1 left").click().run()
    assert wf.pairs[(1, 0)]["left"] == min(cur["left"] + 1, wf.source("p1_left").frame_count - 1) and not wf.all_pairs_confirmed()
    assert has_button(at, "Confirm this pair") and not wf.frames_ready()


def test_re_suggesting_after_boundaries_were_marked_asks_first(complete, svc):
    s, wf = complete;
    s.step = "frames"; at = page_app(svc, s); at.session_state["f8i_fs_mode"] = FP.MODES[0]; at.run()
    if has_button(at, "Suggest Frames"):
        before = dict(wf.pairs); button(at, "Suggest Frames").click().run()
        assert "discards the wall boundaries" in texts(at) and wf.pairs == before and wf.full
        button(at, "Keep my frames").click().run(); assert wf.full


def test_selections_follow_video_and_calibration_changes(ready, svc, clips):
    s, wf = ready; at = _frames_app(svc, s)
    svc.remove_room_video(s, "p2_right")                                           # a video removed: only Viewpoint 2 is discarded
    assert all(v == 1 for (v, _) in wf.pairs) and wf.pairs and s.frames is wf
    upload_all(svc, s, clips, only=["p2_right"]); svc.frame_state(s)
    assert svc.room_report(s).ready
    svc.import_saved(s, b"not json")                                              # calibration replaced: everything goes
    assert s.frames is None and not s.room_ready and s.accepted is None


def test_back_and_home_keep_the_selected_frames(ready, svc):
    s, wf = ready; at = _frames_app(svc, s)
    button(at, "Go back").click().run(); assert s.frames is wf and wf.all_pairs_confirmed()
    s.go("home"); s.go("room"); s.go("frames")
    at = page_app(svc, s); at.run(); assert not at.exception and s.frames is wf and wf.frames_ready() and "Part 2 · Confirm Wall Boundaries" in [x.value for x in at.subheader]

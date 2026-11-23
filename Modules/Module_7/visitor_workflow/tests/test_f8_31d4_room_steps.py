"""F8.31D-4: calibration-page cleanup and the guided three-step room-video page.  SYNTHETIC fixtures only; no reconstruction, no calibration processing, no original data."""
import re

import pytest

import f8_guided_steps as GS
import f8_visitor_pages as P
import f8_visitor_service as S
from test_f8_31b_frames import clips, cstore, ready, rstore, svc, upload_all, accepted         # noqa: F401  (fixtures)
from test_f8_31b_pages import button, has_button, page_app, texts
from test_f8_31d2_ui_fixes import all_text, FORBIDDEN
from test_f8_visitor_service import make_doc


def _open_room(svc, s):
    s.step = "room"; at = page_app(svc, s); at.run(); assert not at.exception
    return at


# ---------------------------------------------------------------- pure step logic
def test_room_steps_sequential_and_withdrawal_relocks():
    r = GS.RoomSteps()
    assert [r.state(n, False) for n in (1, 2, 3)] == ["active", "locked", "locked"]
    assert not r.set_confirmed(2, True)                                    # out of order: refused
    assert r.set_confirmed(1, True) and r.state(2, False) == "active" and r.state(3, False) == "locked"
    assert r.set_confirmed(2, True) and r.state(3, False) == "active"
    assert r.state(3, True) == "done" and r.can_continue(True) and not r.can_continue(False)
    r.set_confirmed(1, False)                                              # withdrawing step 1 also withdraws step 2 and locks 3
    assert r.confirmed == {1: False, 2: False} and r.state(3, True) == "locked" and not r.can_continue(True)


def test_step3_done_comes_only_from_the_backend_fact():
    r = GS.RoomSteps(); r.set_confirmed(1, True); r.set_confirmed(2, True)
    assert r.state(3, False) == "active" and r.state(3, True) == "done"    # no stored flag: an expired upload makes it active again


# ---------------------------------------------------------------- calibration cleanup
def test_accepted_page_has_no_technical_details_and_keeps_the_rest(svc):
    s = S.VisitorSession(); s.review = S.review_f8_document(make_doc("GOOD")[1]); svc.accept(s); assert s.accepted.provenance
    s.step = "ready"; at = page_app(svc, s); at.run(); assert not at.exception
    assert not any("Technical" in str(e.label) for e in at.expander)
    t = all_text(at)
    assert "developers" not in t and "can_proceed" not in t
    assert "Calibration accepted" in t and "Baseline" in t and "Image size" in t and "Quality" in t
    assert has_button(at, "Continue to Room Measurement") and any("Download" in str(d.label) for d in at.get("download_button"))
    assert s.accepted.provenance                                           # provenance is preserved in the session


def _synthetic_saved_file(svc):
    s = S.VisitorSession(); s.review = S.review_f8_document(make_doc("GOOD")[1]); svc.accept(s)
    return svc.contract_json(s)


def test_not_assessed_review_is_not_repetitive_and_keeps_the_acknowledgement(svc):
    s = S.VisitorSession(); svc.import_saved(s, _synthetic_saved_file(svc)); s.step = "review"
    at = page_app(svc, s); at.run(); assert not at.exception
    md = " ".join(m.value for m in at.markdown); t = all_text(at)
    assert md.count("Quality not assessed") == 1                                     # the status badge
    assert len(re.findall(r"not assessed|no quality|quality information|quality assessment", md, re.I)) == 2   # the badge and ONE explanation
    assert "F8" not in t
    assert not at.warning                                                            # no stacked warning boxes
    box = next(c for c in at.checkbox if c.key.startswith("f8i_ack_"))
    assert button(at, "Accept this calibration").disabled                            # restrictions unchanged
    box.check().run(); assert not button(at, "Accept this calibration").disabled
    button(at, "Accept this calibration").click().run(); assert s.accepted is not None and s.step in ("review", "ready")


def test_a_distinct_concern_is_still_shown_on_a_not_assessed_review(svc):
    s = S.VisitorSession(); rev = svc.import_saved(s, _synthetic_saved_file(svc))
    import dataclasses
    s.review = dataclasses.replace(rev, concerns=rev.concerns + ("A genuinely different synthetic concern.",)); s.step = "review"
    at = page_app(svc, s); at.run(); t = all_text(at)
    assert "A genuinely different synthetic concern." in t and S.NOT_ASSESSED_CONCERN not in t


# ---------------------------------------------------------------- room page
def test_initial_state_only_step_1_is_open(svc, ready):
    s, wf = ready; s.step = "room"; svc.room_report(s)                               # four valid uploads already exist, but nothing is confirmed
    at = _open_room(svc, s)
    keys = [c.key for c in at.get("container") if str(getattr(c, "key", "")).startswith("f8i_step_r")] if False else None
    t = all_text(at)
    assert "Prepare Your Room and Cameras" in t and "Record Two Stereo Viewpoints" in t and "Upload Your Four Wall A Videos" in t
    assert "Locked. Confirm Step 1" in t and "Locked. Confirm Step 2" in t
    assert [c.label for c in at.checkbox] == [GS.ROOM_CONFIRM_LABELS[1]]
    assert button(at, "Continue to Frame Selection").disabled                         # stale / pre-existing uploads cannot satisfy readiness while step 3 is locked
    assert not has_button(at, "Start over") and not has_button(at, "Lock this session")
    assert [b.label for b in at.button][:2] == ["Go back", "Home"]


def test_step_by_step_unlock_and_relock_keeps_uploads(svc, ready):
    s, wf = ready; s.step = "room"; svc.room_report(s); uploads = svc.room_store.job_status(s.owner)["slots"].keys()
    at = _open_room(svc, s)
    at.checkbox(key="f8i_rc_1").check().run(); assert s.guide.room.confirmed == {1: True, 2: False}
    assert [c.key for c in at.checkbox] == ["f8i_rc_1", "f8i_rc_2"] and button(at, "Continue to Frame Selection").disabled
    at.checkbox(key="f8i_rc_2").check().run(); assert not at.exception and s.guide.room.confirmed == {1: True, 2: True}
    assert "4 of 4 videos accepted" in " ".join(str(p.proto.text) for p in at.get("progress"))
    assert not button(at, "Continue to Frame Selection").disabled
    at.checkbox(key="f8i_rc_1").uncheck().run()                                       # withdrawing step 1 relocks 2 and 3 ...
    assert s.guide.room.confirmed == {1: False, 2: False} and button(at, "Continue to Frame Selection").disabled
    assert not at.get("progress") and svc.room_store.job_status(s.owner)["slots"].keys() == uploads      # ... without deleting a valid upload
    at.checkbox(key="f8i_rc_1").check().run(); at.checkbox(key="f8i_rc_2").check().run()
    assert not button(at, "Continue to Frame Selection").disabled                      # the same uploads count again


def test_continue_goes_to_the_existing_frame_page(svc, ready):
    s, wf = ready; s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True); s.step = "room"
    at = _open_room(svc, s); button(at, "Continue to Frame Selection").click().run()
    assert not at.exception and s.step == "frames" and "Frames and boundaries" in texts(at)


def test_invalid_or_missing_uploads_do_not_count_and_continue_stays_disabled(svc, clips):
    s = accepted(svc); upload_all(svc, s, clips, only=["p1_left", "p1_right", "p2_left"])
    s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True)
    at = _open_room(svc, s)
    assert "3 of 4 videos accepted" in " ".join(str(p.proto.text) for p in at.get("progress")) or "2 of 4" in " ".join(str(p.proto.text) for p in at.get("progress"))
    assert button(at, "Continue to Frame Selection").disabled
    assert "Viewpoint 2" in all_text(at)


def test_expired_upload_invalidates_readiness(svc, ready):
    s, wf = ready; s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True); s.step = "room"
    at = _open_room(svc, s); assert not button(at, "Continue to Frame Selection").disabled
    svc.room_store.release_owner(s.owner)                                              # what expiry does: the uploads are gone
    at.run(); assert not at.exception and button(at, "Continue to Frame Selection").disabled and not s.room_ready
    assert "0 of 4 videos accepted" in " ".join(str(p.proto.text) for p in at.get("progress"))


def test_new_calibration_clears_room_confirmations_and_start_over_resets(svc, ready):
    s, wf = ready; s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True)
    s.forget_acceptance(); assert s.guide.room.confirmed == {1: False, 2: False}
    s.guide.room.set_confirmed(1, True); s.reset(svc.store, svc.room_store); assert s.guide.room.confirmed == {1: False, 2: False}


def test_room_page_wording_has_no_internal_terms(svc, ready):
    s, wf = ready; s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True); s.step = "room"
    t = all_text(_open_room(svc, s))
    for term in FORBIDDEN + ("adapter", "hash", "backend", "/tmp", "/var/", "Traceback"):
        assert term not in t, term
    assert "15 minutes" in t and S.ROOM_PROCESSING_ENABLED is False

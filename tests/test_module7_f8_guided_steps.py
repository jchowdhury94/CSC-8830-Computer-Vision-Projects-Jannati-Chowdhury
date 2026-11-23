"""F8.31D-3B: guided six-step "Calibrate My Cameras" page.  SYNTHETIC fixtures only (random bytes as videos, a fake calibration runner); no original recording, no real calibration."""
import pytest

from test_module7_f8_visitor_integration import (FakeRunner, J, _isolated, button, env, make_doc, page_app, run_job, texts, upload_both)    # noqa: F401,E402  (fixtures; also sets sys.path)
import calibration_runner as CR  # noqa: E402
import f8_guided_steps as GS  # noqa: E402
import f8_visitor_service as S  # noqa: E402

F = GS.Facts


def labels(at):
    return [c.label for c in at.checkbox]


def sub(at):
    return [s.value for s in at.subheader]


def setup_app(env, sess=None):
    sess = sess or S.VisitorSession(); sess.step = "setup"
    at = page_app(env, sess); at.run(); assert not at.exception
    return at, sess


def tick(at, n):
    at.checkbox(key=f"f8i_gc_{n}").check().run(); assert not at.exception


# ------------------------------------------------------------------ pure step logic
def test_only_step_1_is_unlocked_initially():
    g = GS.GuidedSteps(); f = F()
    assert [g.state(n, f) for n in range(1, 7)] == ["active", "locked", "locked", "locked", "locked", "locked"]
    assert g.current(f) == 1


def test_confirmations_unlock_in_order_and_cannot_be_skipped():
    g = GS.GuidedSteps(); f = F()
    assert g.set_confirmed(3, True) is False and not g.confirmed[3]                   # out of order: ignored
    g.set_confirmed(1, True); assert [g.state(n, f) for n in (1, 2, 3, 4)] == ["done", "active", "locked", "locked"]
    g.set_confirmed(2, True); g.set_confirmed(3, True)
    assert [g.state(n, f) for n in (3, 4, 5, 6)] == ["done", "active", "locked", "locked"]


def test_step_5_needs_valid_uploads_and_step_6_needs_a_result():
    g = GS.GuidedSteps()
    for n in (1, 2, 3):
        g.set_confirmed(n, True)
    assert g.state(5, F()) == "locked" and g.state(6, F()) == "locked"
    assert g.state(4, F(uploads_valid=True)) == "done" and g.state(5, F(uploads_valid=True)) == "active" and g.state(6, F(uploads_valid=True)) == "locked"
    done = F(uploads_valid=False, result_ready=True, calibration_started=True)
    assert g.state(5, done) == "done" and g.state(6, done) == "active"
    assert g.state(6, F(result_ready=True, accepted=True, calibration_started=True)) == "done"


def test_unchecking_an_earlier_confirmation_clears_only_later_ones():
    g = GS.GuidedSteps()
    for n in (1, 2, 3):
        g.set_confirmed(n, True)
    g.set_confirmed(2, False)
    assert g.confirmed == {1: True, 2: False, 3: False}
    g.set_confirmed(1, False); assert g.confirmed == {1: False, 2: False, 3: False}


def test_completion_never_outlives_its_input():
    g = GS.GuidedSteps()
    for n in (1, 2, 3):
        g.set_confirmed(n, True)
    assert g.complete(4, F(uploads_valid=True)) and not g.complete(4, F(uploads_valid=False))        # an expired / removed upload un-completes step 4 and locks step 5
    assert g.state(5, F(uploads_valid=False)) == "locked"
    g.set_confirmed(3, False)
    assert g.state(5, F(uploads_valid=True)) == "locked" and g.state(4, F(uploads_valid=True)) == "locked"


def test_frozen_confirmations_refuse_changes_once_a_job_exists():
    g = GS.GuidedSteps()
    g.set_confirmed(1, True)
    assert g.set_confirmed(1, False, frozen=True) is False and g.confirmed[1]


def test_session_isolation_and_reset():
    a, b = S.VisitorSession(), S.VisitorSession()
    a.guide.set_confirmed(1, True); a.guide.set_setting("cols", 9)
    assert not b.guide.confirmed[1] and b.guide.settings["cols"] == 8
    a.reset(); assert a.guide.confirmed == {1: False, 2: False, 3: False} and a.guide.settings == GS.DEFAULT_SETTINGS


# ------------------------------------------------------------------ the page
def test_initially_only_step_1_has_controls(env):
    at, sess = setup_app(env)
    assert sub(at) == [f"Step {n} · {GS.TITLES[n]}" for n in range(1, 7)]
    assert labels(at) == [GS.CONFIRM_LABELS[1]]                                       # no control exists inside any locked step
    assert [n.label for n in at.number_input] == ["Physical stereo baseline (mm), if measured"] and not [b for b in at.button if "Start calibration" in b.label]   # only step 1's input exists
    assert len([e for e in at.expander if e.label == "View instructions"]) == 1
    assert texts(at).count("Locked.") >= 5 and "For the administrator" not in texts(at)
    assert FakeRunner.instances == []


def test_steps_unlock_one_by_one(env):
    at, sess = setup_app(env)
    tick(at, 1); assert labels(at) == [GS.CONFIRM_LABELS[1], GS.CONFIRM_LABELS[2]] and sess.guide.confirmed[1]
    tick(at, 2); assert GS.CONFIRM_LABELS[3] in labels(at) and len(at.number_input) == 4          # baseline (step 1) + 3 board inputs (step 2)
    tick(at, 3)
    assert "Step 5" in " ".join(sub(at))                                               # step 4 open (uploaders), step 5 still locked: no uploads yet
    assert not [b for b in at.button if "Start calibration" in b.label] and "Locked. Upload both videos" in texts(at)


def test_valid_uploads_unlock_step_5_in_the_same_run_and_expired_uploads_relock_it(env):
    at, sess = setup_app(env)
    for n in (1, 2, 3):
        tick(at, n)
    upload_both(env, sess); at.run()
    assert not at.exception and button(at, "Start calibration").disabled and len(at.number_input) == 4          # step 5 open (confirmation still required); its settings are read-only
    assert "Locked. Available once your calibration" in texts(at)                      # step 6 stays locked
    env.store.release_owner(sess.owner); at.run()                                      # uploads gone / expired
    assert not [b for b in at.button if "Start calibration" in b.label] and "Locked. Upload both videos" in texts(at)


def test_one_invalid_upload_does_not_unlock_step_5(env):
    at, sess = setup_app(env)
    for n in (1, 2, 3):
        tick(at, n)
    import hashlib
    data = b"x" * 3000; uid = env.store.begin(sess.owner, "c.mov", len(data), 1024, "left", slot="left")["upload_id"]
    env.store.put_chunk(sess.owner, uid, 0, data[:1024], hashlib.sha256(data[:1024]).hexdigest())      # unfinished LEFT, no RIGHT
    at.run(); assert not [b for b in at.button if "Start calibration" in b.label] and not env.slots_ready(sess)


def test_changing_step_1_after_progress_relocks_dependents_but_keeps_uploads(env):
    at, sess = setup_app(env)
    for n in (1, 2, 3):
        tick(at, n)
    upload_both(env, sess); at.run(); assert len(at.number_input) == 4
    at.checkbox(key="f8i_gc_1").uncheck().run(); assert not at.exception
    assert sess.guide.confirmed == {1: False, 2: False, 3: False} and labels(at) == [GS.CONFIRM_LABELS[1]] and len(at.number_input) == 1
    assert env.slots_ready(sess)                                                       # uploads are valid backend facts: kept, the step simply stays locked
    for n in (1, 2, 3):
        assert not at.checkbox(key=f"f8i_gc_{n}").value if n == 1 else True
    tick(at, 1); assert not sess.guide.confirmed[2] and GS.CONFIRM_LABELS[3] not in labels(at)         # step 2 had to be confirmed again


def test_measurements_are_entered_in_steps_1_and_2_and_summarised_read_only_in_step_5(env):
    at, sess = setup_app(env)
    at.number_input(key="f8i_base").set_value(226.0).run(); tick(at, 1)
    at.number_input(key="f8i_cols").set_value(7).run(); at.number_input(key="f8i_rows").set_value(5).run(); at.number_input(key="f8i_sq").set_value(25.0).run()
    assert not sess.guide.confirmed[2]                                                  # editing a measurement withdraws that step's confirmation
    tick(at, 2); tick(at, 3); upload_both(env, sess); at.run()
    assert sess.guide.settings == dict(cols=7, rows=5, square=25.0, baseline=226.0)    # one value per setting, on the session
    cells = " ".join(str(c.value) for c in at.table[0].cells) if False else str(at.table[0].value)
    assert "7 × 5" in cells and "25 mm" in cells and "226 mm" in cells
    assert {n.key for n in at.number_input} == {"f8i_base", "f8i_cols", "f8i_rows", "f8i_sq"}      # nothing editable in step 5
    at.checkbox(key="f8i_sure").check().run(); assert not button(at, "Start calibration").disabled
    sess.step = "home"; at.run(); sess.step = "setup"; at.run()                         # Home and back
    assert sess.guide.confirmed == {1: True, 2: True, 3: True} and at.number_input(key="f8i_cols").value == 7 and "226 mm" in str(at.table[0].value)


def test_editing_an_earlier_measurement_updates_step_5_and_invalidates_dependents(env):
    at, sess = setup_app(env)
    for n in (1, 2, 3):
        tick(at, n)
    upload_both(env, sess); at.run(); at.checkbox(key="f8i_sure").check().run()
    assert "Not measured" in str(at.table[0].value)
    at.number_input(key="f8i_base").set_value(230.0).run(); assert not at.exception
    assert sess.guide.confirmed == {1: False, 2: False, 3: False}                       # step 1 and everything after it must be confirmed again
    assert not [b for b in at.button if "Start calibration" in b.label] and "Locked. Upload both videos" in texts(at)
    for n in (1, 2, 3):
        tick(at, n)
    at.run(); assert "230 mm" in str(at.table[0].value) and not at.checkbox(key="f8i_sure").value      # summary follows at once; the long-run acknowledgement is asked again


def test_invalid_measurements_cannot_be_confirmed(env):
    at, sess = setup_app(env); tick(at, 1)
    at.number_input(key="f8i_cols").set_value(6).run(); at.number_input(key="f8i_rows").set_value(6).run()
    assert "different number of columns and rows" in texts(at) and at.checkbox(key="f8i_gc_2").disabled
    at.number_input(key="f8i_rows").set_value(5).run(); assert not at.checkbox(key="f8i_gc_2").disabled


def test_baseline_out_of_range_blocks_step_1(env):
    at, sess = setup_app(env); at.number_input(key="f8i_base").set_value(3.0).run()
    assert "baseline must be between" in texts(at) and at.checkbox(key="f8i_gc_1").disabled


def test_measurement_inputs_are_frozen_while_a_job_exists(env):
    sess = S.VisitorSession(); sess.guide.confirmed.update({1: True, 2: True, 3: True}); run_job(env, sess); sess.step = "setup"
    at = page_app(env, sess); at.run(); assert all(n.disabled for n in at.number_input)


def test_disabled_processing_keeps_start_disabled_and_step_6_locked(env):
    off = S.VisitorCalibrationService(env.store, env.results, CR.RunnerLimits(allow_full=False), runner_cls=FakeRunner)
    sess = S.VisitorSession(); sess.step = "setup"; upload_both(off, sess)
    at = page_app(off, sess); at.run()
    for n in (1, 2, 3):
        tick(at, n)
    at.checkbox(key="f8i_sure").check().run()
    assert button(at, "Start calibration").disabled and "switched off on this server" in texts(at)
    assert "Locked. Available once your calibration" in texts(at) and FakeRunner.instances == [] and sess.runner is None and S.ROOM_PROCESSING_ENABLED is False


def test_page_refresh_never_unlocks_anything(env):
    at, sess = setup_app(env)
    for n in (1, 2, 3):
        tick(at, n)
    at2 = page_app(env); at2.session_state["f8i_session"] = S.VisitorSession(); at2.session_state["f8i_session"].step = "setup"; at2.run()      # a refresh = a NEW session
    assert labels(at2) == [GS.CONFIRM_LABELS[1]] and len(at2.number_input) == 1


def test_real_result_unlocks_step_6_then_review_and_accept_use_the_existing_workflow(env):
    sess = S.VisitorSession(); sess.guide.confirmed.update({1: True, 2: True, 3: True}); run_job(env, sess); sess.step = "setup"
    at = page_app(env, sess); at.run(); assert not at.exception
    assert "The calibration has finished" in texts(at) and not button(at, "Review the calibration").disabled
    assert all(c.disabled for c in at.checkbox if c.label in GS.CONFIRM_LABELS.values())                  # confirmations frozen while the job exists
    button(at, "Review the calibration").click().run(); assert sess.step == "review" and "Good" in texts(at)
    button(at, "Accept this calibration").click().run(); assert sess.accepted is not None


def test_failed_calibration_does_not_unlock_step_6(env):
    sess = S.VisitorSession(); sess.guide.confirmed.update({1: True, 2: True, 3: True}); run_job(env, sess, "FAILED", outcome="failed"); sess.step = "setup"
    at = page_app(env, sess); at.run(); assert not at.exception
    assert not [b for b in at.button if b.label == "Review the calibration"] and "Locked. Available once your calibration" in texts(at)


def test_retry_after_finish_returns_to_the_settings(env):
    sess = S.VisitorSession(); sess.guide.confirmed.update({1: True, 2: True, 3: True}); run_job(env, sess, kept=True); sess.step = "setup"
    at = page_app(env, sess); at.run(); button(at, "Retry").click().run()
    assert sess.runner is None and button(at, "Start calibration").disabled and len(at.number_input) == 4


def test_recovery_panel_stays_available_without_any_confirmation(env):
    at, sess = setup_app(env)
    assert any("recovery code" in e.label for e in at.expander) and at.text_input(key="f8i_rec_code")


def test_saved_calibration_import_still_works(env):
    sess = S.VisitorSession(); env.import_saved(sess, J(make_doc("GOOD"))); at = page_app(env, sess); at.run()
    assert "Good" in texts(at); button(at, "Accept this calibration").click().run(); assert sess.accepted is not None and not sess.guide.confirmed[1]


# ------------------------------------------------------------------ F8.31D: Upload Saved Calibration pages have no Start over / Lock buttons
def _no_reset_buttons(at):
    assert not any("Start over" in b.label or "Lock this session" in b.label for b in at.button), [b.label for b in at.button]


def test_import_review_and_accepted_pages_have_no_start_over_or_lock(env):
    sess = S.VisitorSession(); sess.step = "import"
    at = page_app(env, sess); at.run(); assert not at.exception
    _no_reset_buttons(at)
    assert [e.label for e in at.expander] == ["View Instructions"]
    assert "How this works" not in " ".join(e.label for e in at.expander)
    env.import_saved(sess, J(make_doc("GOOD"))); at.run(); _no_reset_buttons(at); assert sess.step == "review" and button(at, "Accept this calibration")
    button(at, "Accept this calibration").click().run(); assert sess.step == "review" or sess.accepted is not None
    sess.step = "ready"; at.run(); _no_reset_buttons(at); assert "Calibration accepted" in texts(at)


def test_instructions_text_is_unchanged(env):
    sess = S.VisitorSession(); sess.step = "import"; at = page_app(env, sess); at.run()
    t = texts(at)
    for frag in ("Supported files.", "Quality review.", "Acceptance.", "Nothing is used until you review it"):
        assert frag in t


def test_reset_and_lock_remain_on_the_calibrate_and_room_pages(env):
    at, sess = setup_app(env)
    assert any("Start over" in b.label for b in at.button) and any("Lock this session" in b.label for b in at.button)
    sess2 = S.VisitorSession(); run_job(env, sess2); env.finished_review(sess2); sess2.step = "review"; at2 = page_app(env, sess2); at2.run()           # review of a calibration made in this app
    assert any("Start over" in b.label for b in at2.button) and any("Lock this session" in b.label for b in at2.button)
    button(at2, "Accept this calibration").click().run(); sess2.step = "ready"; at2.run()
    assert any("Start over" in b.label for b in at2.button)
    sess2.step = "room"; at2.run(); assert not any(("Lock this session" in b.label or "Start over" in b.label) for b in at2.button)       # F8.31F: the room-upload page shows only the heading, the buttons and the navigation


def test_the_reset_and_lock_logic_still_works(env):
    sess = S.VisitorSession(); env.import_saved(sess, J(make_doc("GOOD"))); env.accept(sess); owner = sess.owner
    sess.reset(env.store, env.room_store); assert sess.accepted is None and sess.step == "home" and sess.owner != owner

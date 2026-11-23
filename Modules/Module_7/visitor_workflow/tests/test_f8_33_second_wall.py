"""Second wall (room length): two independent wall measurements from the one calibration.  SYNTHETIC fixtures and INJECTED fake results only: no worker, no recording, no algorithm touched.

Wall A and Wall B each run the ONE existing measurement path (``MeasurementService.start`` -> ``run_visitor_reconstruction``); the room page only reads the two results."""
import json
import threading

import pytest

import f8_measure_page as MP
import f8_measurement as M
import f8_result_view as RVW
import f8_visitor_service as S
import f8_walls as W
import room_reconstruction as RR
import room_videos as RV
from test_f8_31a_room_videos import put
from test_f8_31b_frames import annotate_all, choose_and_confirm, svc, accepted, clips, cstore, rstore   # noqa: F401
from test_f8_31b_pages import button, has_button, page_app, texts
from test_f8_31c2a_measurement_interface import PAIRS5, ANY_SETS, failed_result, ok_result, use_measurement, wait, run_to_completion


@pytest.fixture
def two_walls(svc, clips):
    s = accepted(svc)
    out = []
    for w in ("A", "B"):
        s.switch_wall(w)
        for slot, p in clips[0].items():                                  # the same synthetic clips for both walls (each wall's uploads are owned separately)
            put(svc.room_store, s.room_owner, slot, p.read_bytes(), name=p.name)
        st = svc.frame_state(s); assert st["state"] == "ok", st
        wf = st["workflow"]; choose_and_confirm(wf, sets=5, pairs=PAIRS5); annotate_all(wf, sets=5)
        assert wf.readiness(deep=True).ready
        out.append(wf)
    s.switch_wall("A")
    return s, out[0], out[1]


def run_wall(svc, s, wall, result):
    s.switch_wall(wall); use_measurement(svc, runner=lambda ctx: result); run_to_completion(svc.measurement, s)


def two_results(svc, s, a=None, b=None):
    run_wall(svc, s, "B", b or ok_result(width=3400.0, height=2400.0)); run_wall(svc, s, "A", a or ok_result(width=2850.0, height=2360.0))


# ------------------------------------------------------------------ isolation
def test_wall_b_has_its_own_uploads_frames_and_owner(svc, two_walls):
    s, wa, wb = two_walls
    assert wa is not wb and s.room_owner == s.owner
    ua = {k: v["upload_id"] for k, v in svc.room_store.job_status(s.room_owner)["slots"].items()}
    s.switch_wall("B"); assert s.room_owner != s.owner and s.frames is wb
    ub = {k: v["upload_id"] for k, v in svc.room_store.job_status(s.room_owner)["slots"].items()}
    assert len(ua) == len(ub) == 4 and not set(ua.values()) & set(ub.values())
    svc.remove_room_video(s, "p1_left")                                   # removing a Wall B video ...
    s.switch_wall("A")
    assert len(svc.room_store.job_status(s.room_owner)["slots"]) == 4 and s.frames is wa and not wa.closed and wa.readiness(deep=True).ready     # ... leaves Wall A whole


def test_boundaries_and_frames_are_independent_per_wall(svc, two_walls):
    s, wa, wb = two_walls
    s.switch_wall("B"); wb.reject(0, "AL_corner"); assert not wb.readiness().ready
    s.switch_wall("A"); assert wa.readiness(deep=True).ready


def test_starting_wall_b_keeps_wall_a_result_and_calibration(svc, two_walls):
    s, wa, wb = two_walls; contract, acc = s.contract, s.accepted
    run_wall(svc, s, "A", ok_result(width=2850.0, height=2360.0))
    s.switch_wall("B"); assert s.measurement.result is None and s.contract is contract and s.accepted is acc and s.f6 is not None
    s.switch_wall("A"); assert s.measurement.result["width_mm"] == 2850.0 and svc.measurement.current(s) is s.measurement.result


def test_a_running_wall_cannot_be_left(svc, two_walls):
    s, wa, wb = two_walls; ev = threading.Event()
    ms = use_measurement(svc, runner=lambda ctx: (ev.wait(10), ok_result())[1]); ms.start(s)
    with pytest.raises(S.ServiceError, match="running"):
        s.switch_wall("B")
    ev.set(); wait(s.measurement.job); s.switch_wall("B")


def test_a_changed_calibration_clears_both_walls(svc, two_walls):
    s, wa, wb = two_walls; two_results(svc, s)
    s.forget_acceptance()
    assert s.active_wall == "A" and s.measurement.result is None and s.frames is None
    with s.on_wall("B"):
        assert s.measurement.result is None and s.frames is None


def test_new_measurement_releases_both_walls_uploads(svc, two_walls):
    s, wa, wb = two_walls; owners = s.wall_owners(); assert len(owners) == 2
    s.reset(svc.store, svc.room_store)
    assert all(not svc.room_store.job_status(o)["slots"] for o in owners) and s.active_wall == "A" and s.accepted is None


# ------------------------------------------------------------------ Method J is reached once per wall, through the existing path
def test_each_wall_runs_the_existing_reconstruction_with_its_own_inputs(svc, two_walls, monkeypatch):
    s, wa, wb = two_walls; calls = []

    def spy(wf, owner, contract, cal, sources, limits=None, **kw):
        calls.append((wf, owner, sources)); return ok_result(width=3000.0 + 100 * len(calls))
    monkeypatch.setattr(RR, "run_visitor_reconstruction", spy)
    ms = M.MeasurementService(svc, gate=lambda: True, limits=ANY_SETS); svc.measurement = ms
    run_to_completion(ms, s); s.switch_wall("B"); run_to_completion(ms, s)
    assert [c[0] for c in calls] == [wa, wb] and calls[0][1] == s.owner and calls[1][1] == s.room_owner != s.owner
    assert {x.upload_id for x in calls[0][2].values()}.isdisjoint({x.upload_id for x in calls[1][2].values()})
    s.switch_wall("A"); assert s.measurement.result["width_mm"] == 3100.0
    s.switch_wall("B"); assert s.measurement.result["width_mm"] == 3200.0


# ------------------------------------------------------------------ the room summary
def test_room_dimensions_come_from_the_right_wall_and_are_not_averaged():
    a, b = ok_result(width=2850.482, height=2358.141), ok_result(width=3412.5, height=2400.0)
    rv = RVW.room_view(a, b)
    assert rv["complete"] and [d["key"] for d in rv["dimensions"]] == ["room_width", "room_length", "room_height"]
    vals = {d["key"]: d["value"] for d in rv["dimensions"]}
    assert vals["room_width"]["mm"] == 2850.482 and vals["room_length"]["mm"] == 3412.5 and vals["room_height"]["mm"] == 2358.141      # unrounded, one source each
    assert vals["room_length"]["cm_text"] == "341 cm" and vals["room_length"]["m_text"] == "3.41 m" and vals["room_height"]["mm"] != 2400.0


def test_no_room_without_two_accepted_walls():
    a = ok_result(); f = failed_result("FEATURE_MATCHING_FAILED", cls="method_j"); f["width_mm"], f["height_mm"] = 9999.0, 9999.0
    for x, y in ((a, f), (f, a), (a, None), (None, a)):
        assert RVW.room_view(x, y)["complete"] is False and RVW.room_view(x, y)["dimensions"] == [] and RVW.room_summary_document(x, y) is None
    partial = ok_result(); partial["width_mm"] = None
    assert RVW.room_view(a, partial)["complete"] is False


def test_combined_json(svc):
    a, b = ok_result(width=2850.0, height=2360.0), ok_result(width=3400.0, height=2400.0)
    doc = RVW.room_summary_document(a, b)
    assert doc["schema"] == RVW.ROOM_SCHEMA and doc["dimensions"]["room_width"]["cm"] == 285.0 and doc["dimensions"]["room_length"]["cm"] == 340.0 and doc["dimensions"]["room_height"]["cm"] == 236.0
    assert doc["dimensions"]["room_length"]["source"].startswith("Wall B") and doc["dimensions"]["room_height"]["source"].startswith("Wall A")
    assert doc["walls"]["wall_a"]["width"]["mm"] == 2850.0 and doc["walls"]["wall_b"]["width"]["mm"] == 3400.0 and doc["walls"]["wall_b"]["height"]["mm"] == 2400.0
    assert json.loads(json.dumps(doc)) == doc and "path" not in json.dumps(doc).lower().replace("per_frame_set", "")


# ------------------------------------------------------------------ pages
def test_wall_a_page_is_step_one_of_two_and_continue_opens_wall_b(svc, two_walls):
    s, wa, wb = two_walls; run_wall(svc, s, "A", ok_result(width=2850.0, height=2360.0)); s.step = "measure"
    at = page_app(svc, s); at.run(); assert not at.exception, at.exception
    t = texts(at)
    assert "Wall A Measured — Step 1 of 2" in t and "Room Width" in t and "285 cm" in t and "Room Height" in t and "236 cm" in t and W.WALL_A_DONE_TEXT in t
    assert "Estimated Wall" not in t and "Room Length" not in t.replace("Wall B — Room Length", "").replace("Measure Wall B (Room Length)", "")
    assert "Measurement Complete" not in t.replace("Room Measurement Complete", "") and "Step 1 of 2 — Measure Wall A (Room Width)" in t
    assert 'aria-valuenow=\'100\'' not in t and [d.proto.label for d in at.get("download_button")] == ["Download Wall A Results"]      # no full bar, no final "Download Results"
    b = button(at, "Continue to Wall B — Room Length"); assert b.proto.type == "primary" and not has_button(at, "Measure Second Wall")
    b.click().run(); assert not at.exception, at.exception
    t = texts(at)
    assert s.active_wall == "B" and s.step == "room" and "Wall B — Room Length" in t and W.WALL_UPLOAD_HINT["B"] in t and "Step 2 of 2 — Measure Wall B (Room Length)" in t
    assert "Wall A — Room Width" not in t.replace("Wall A — Room Width →", "")
    s.switch_wall("A"); assert s.measurement.result["width_mm"] == 2850.0 and svc.calibration_context(s) is not None and s.frames is wa and wa.readiness(deep=True).ready


def _room_upload_page(svc, s, wall):
    s.switch_wall(wall); s.step = "room"
    for n in (1, 2):
        s.guide.room.set_confirmed(n, True)
    at = page_app(svc, s); at.run(); assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("wall,heading", [("A", "Wall A — Room Width"), ("B", "Wall B — Room Length")])
def test_upload_pages_identify_the_wall(svc, two_walls, wall, heading):
    s, wa, wb = two_walls; at = _room_upload_page(svc, s, wall); t = texts(at)
    for frag in (heading, W.WALL_UPLOAD_HINT[wall], W.WALL_STEP[wall]):
        assert frag in t, frag
    assert f"Upload Your Four Wall {wall} Videos" in [x.value for x in at.subheader]
    assert "Upload Your Four Videos" not in t and "Measure Second Wall" not in t


@pytest.mark.parametrize("step", ["frames", "measure"])
@pytest.mark.parametrize("wall", ["A", "B"])
def test_every_wall_page_names_the_active_wall(svc, two_walls, wall, step):
    s, wa, wb = two_walls; s.switch_wall(wall); s.step = step
    at = page_app(svc, s); at.run(); assert not at.exception, at.exception
    t = texts(at); assert W.WALL_STEP[wall] in t and ("Room Length" if wall == "B" else "Room Width") in t


def test_wall_b_calculation_runs_the_existing_pipeline_and_final_page_is_the_only_complete_one(svc, two_walls):
    s, wa, wb = two_walls; two_results(svc, s); s.switch_wall("B"); s.step = "measure"
    at = page_app(svc, s); at.run(); t = texts(at)
    assert "Room Measurement Complete" in t and "aria-valuenow='100'" in t
    assert "Wall B" in t and "400 cm" not in t and "240 cm" not in t          # Wall B's own height (2400 mm) is never shown as a room dimension


def test_room_page_shows_three_cards_limitations_and_download(svc, two_walls):
    s, wa, wb = two_walls; two_results(svc, s); s.step = "measure"
    at = page_app(svc, s); at.run(); assert not at.exception, at.exception
    t = texts(at)
    for frag in ("Room Measurement Complete", "Room Width", "Room Length", "Room Height", "285 cm", "2.85 m", "340 cm", "3.40 m", "236 cm", "2.36 m", "Measurement Limitations", RVW.LIMITATIONS_TEXT):
        assert frag in t, frag
    assert at.get("download_button")[0].proto.label == "Download Results" and has_button(at, "Change Wall B Boundaries") and has_button(at, "New Measurement")
    assert not has_button(at, "Continue to Wall B") and "Start over" not in t and "Ready to calculate" not in t
    assert [e.label for e in at.expander].count("Technical Details") == 1 and all(not e.proto.expanded for e in at.expander)


def test_failed_wall_b_keeps_wall_a_and_shows_the_real_reason(svc, two_walls):
    s, wa, wb = two_walls
    run_wall(svc, s, "A", ok_result(width=2850.0, height=2360.0))
    run_wall(svc, s, "B", failed_result("FEATURE_MATCHING_FAILED", cls="method_j")); s.step = "measure"
    at = page_app(svc, s); at.run(); assert not at.exception, at.exception
    t = texts(at)
    assert "Room Measurement Complete" not in t and "Room Length" not in t.replace("Calculate Room Length", "").replace("Measure Wall B (Room Length)", "") and "Room Length —" not in t and VM_text("FEATURE_MATCHING_FAILED") in t
    assert has_button(at, "Go to") and has_button(at, "View Wall A") and [d.proto.label for d in at.get("download_button")] == ["Download Wall A Results"]
    assert RVW.room_summary_document(s._walls["A"]["measurement"].result, s.measurement.result) is None
    button(at, "View Wall A").click().run(); assert s.active_wall == "A" and "Wall A Measured — Step 1 of 2" in texts(at) and "285 cm" in texts(at)


def test_wall_b_calculate_page_is_gated_and_labelled(svc, two_walls, monkeypatch):
    monkeypatch.setattr(S, "WALL_B_PROCESSING_ENABLED", False)                                          # closed Wall B gate
    s, wa, wb = two_walls; run_wall(svc, s, "A", ok_result()); s.switch_wall("B"); s.step = "measure"
    svc.measurement = M.MeasurementService(svc, runner=lambda ctx: pytest.fail("must not run"))     # the default gate: closed
    at = page_app(svc, s); at.run(); assert not at.exception, at.exception
    assert button(at, "Calculate Room Length").disabled and S.ROOM_PROCESSING_ENABLED is False and "Wall A measured" in texts(at)


def VM_text(code):
    import visitor_messages as VM
    return VM.failure_text(code).split(".")[0]

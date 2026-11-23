"""F8.31C-2A: visitor measurement interface, readiness, gated action and results.  SYNTHETIC fixtures and INJECTED fake results only.

These are INTERFACE tests.  They check readiness wiring, the closed gate, binding / invalidation, duplicate-submission and worker-state handling, and the presentation of a result
the backend (or, here, a fake runner) already produced.  No real reconstruction worker is started by any test, no original recording is read, and nothing here validates
Method J on real video (that is the backend's own test file, test_f8_31c1_reconstruction.py).
"""
import copy
import inspect
import json
import re
import threading
import time
from pathlib import Path

import pytest

import f8_measure_page as MP
import f8_measurement as M
import f8_result_view as RVW
import f8_visitor_service as S
import room_frames as RF
import room_reconstruction as RR
import room_videos as RV
import visitor_messages as VM
from test_f8_31b_frames import (annotate_all, choose_and_confirm, clips, cstore, complete, ready, rstore, svc, upload_all, accepted)    # noqa: F401  (fixtures)
from test_f8_31b_pages import button, has_button, page_app, texts
from test_f8_31a_room_videos import put
from test_f8_visitor_service import make_doc

HERE = Path(__file__).resolve().parents[1]
MODULE = HERE.parent
PAIRS5 = {1: [(10, 14), (20, 27), (5, 9), (15, 19), (25, 29)], 2: [(8, 6), (24, 30), (12, 16), (2, 4), (18, 22)]}
ANY_SETS = RR.ReconstructionLimits(required_frame_sets=None)             # the 2-set synthetic workflow; the default (exactly five) is tested separately


# ------------------------------------------------------------------ helpers
def ok_result(wf=None, *, status="SUCCESS", warnings=(), width=4123.456789, height=2718.281828, partial=False):
    n = 5
    dm = lambda v, i: None if v is None else v + i                                                    # noqa: E731
    return dict(schema=RR.SCHEMA_RESULT, status=status, width_mm=width, height_mm=height, units="mm", partial=partial, dimensions_accepted=True,
                per_frame_set=[dict(frame_set=i, frame_indices=dict(AL=1, PL=2, AR=3, PR=4), features={}, status="reconstructed", width_mm=dm(width, i), height_mm=dm(height, -i),
                                    registration=dict(inliers=120 + i, rms_px=0.8), vertical_error_deg=0.2, skew_mm=1.5) for i in range(n)],
                failures=[], failure_classes=[], failure_class=None, warnings=[dict(code=c, advisory=True) for c in warnings], notes=[],
                diagnostics=dict(pose=dict(inlier_fraction=0.7, n_inliers=900, joint=dict(converged=True)), registration=dict(rms=0.5, n_inliers=80), set_failures=[],
                                 geometry_consistency_informational=None, original_pipeline_decision=None, n_sets=n, n_sets_ok=n, calibration_use=dict(lines_retargeted=False, digest="d")),
                processing=dict(seconds=27.25, worker={}, stage_seconds=None, worker_self_peak_rss_mb=800.0),
                provenance=dict(calibration=dict(calibration_id="cal123", image_size_wh=[1920, 1080], baseline_mm=226.0, baseline_unit="mm", baseline_source="visitor measured",
                                                 camera_order=["left", "right"], distortion_model="opencv5", capture_mode=dict(left="video", right="video"), calibration_digest="x"),
                                videos={s: dict(sha256="ab" * 32, upload_id="u", size_bytes=10, frame_count=36, fps=30.0, wh=[1920, 1080]) for s in RV.ROOM_UPLOAD_SLOTS},
                                frame_indices=[dict(AL=1, PL=2, AR=3, PR=4)] * n, n_frame_sets=n, frame_fingerprints=[{}] * n, annotations=[dict(frame_set=0, line="AL_corner")] * 30,
                                coordinate_space="x", owner_digest="OWNERDIGEST123", method="Method J (original)", top_variant=0, required_lines=[], ground_truth_used=False,
                                recipe_a_used=False, historical_baseline_used=False),
                uncertainty=dict(status="not_computed"), ground_truth_used=False, note="n", visitor_notice="n")


def failed_result(code, cls="visitor_input", message="technical /secret/path detail"):
    r = RR.assemble_result(None, failures=[RR.failure(code, message, stage="test")])
    assert r["status"] == "FAILED"
    return r


def instant(result_fn):
    return lambda ctx: result_fn(ctx)


def use_measurement(svc, *, runner=None, gate=True, limits=ANY_SETS):
    svc.measurement = M.MeasurementService(svc, gate=(lambda: True) if gate else None, runner=runner, limits=limits)
    return svc.measurement


def wait(job, timeout=10.0):
    t0 = time.time()
    while not job.done and time.time() - t0 < timeout:
        time.sleep(0.01)
    assert job.done, "the job did not finish"


def run_to_completion(ms, s):
    ms.start(s); job = s.measurement.job; wait(job); return job


PAIRS = {1: [(10, 14), (20, 27)], 2: [(8, 6), (24, 30)]}


@pytest.fixture
def complete5(ready):
    s, wf = ready; choose_and_confirm(wf, sets=5, pairs=PAIRS5); annotate_all(wf, sets=5)
    assert wf.readiness(deep=True).ready, wf.readiness().to_dict()
    return s, wf


# ================================================================ 1. readiness (from the backend's input check)
def test_complete_readiness_comes_from_the_backend_input_check(svc, complete5):
    s, wf = complete5
    rd = svc.measurement.readiness(s)
    assert rd.ready and rd.input_status == "current" and all(i.ok for i in rd.items) and rd.binding is not None and rd.gate_open is False
    assert [i.key for i in rd.items] == ["calibration", "videos", "frames", "annotations"]
    sm = rd.summary
    assert sm["n_frame_sets"] == 5 and sm["annotations_accepted"] == 30 == sm["annotations_required"] and sm["baseline_mm"] == 226.0 and sm["camera_order"] == list(RF.CAMERA_ORDER)
    assert sm["calibration_size_wh"] == [1920, 1080] and set(sm["videos"]) == set(RV.ROOM_UPLOAD_SLOTS) and all(v["size_wh"] == [1920, 1080] for v in sm["videos"].values())
    prep = RR.prepare(wf, s.owner, s.contract, s.f6.job.cal, svc.room_sources(s), RR.ReconstructionLimits(verify_video_hashes=False))
    assert prep.request is not None and rd.ready is True                                  # same verdict as the backend's own check
    assert json.loads(json.dumps(rd.to_dict()))["schema"] == M.READINESS_SCHEMA


def test_readiness_is_the_backends_verdict_not_a_ui_rule(svc, complete):
    s, wf = complete                                                                      # two fully annotated sets: every visible field is filled
    assert wf.readiness(deep=True).ready                                                  # the frame workflow is happy ...
    rd = svc.measurement.readiness(s)
    assert not rd.ready and "FRAME_SET_COUNT" in rd.items[2].codes                      # ... but the backend requires exactly five sets
    assert rd.items[2].step == "frames" and any("five frame sets" in p for p in rd.items[2].problems)
    assert use_measurement(svc).readiness(s).ready                                         # with the backend's own limits relaxed (tests only) it is ready


def test_missing_calibration(svc):
    s = S.VisitorSession(); rd = svc.measurement.readiness(s)
    assert not rd.ready and rd.input_status == "not_accepted" and rd.binding is None
    assert not rd.items[0].ok and rd.items[0].step == "home" and "no accepted calibration" in rd.items[0].problems[0].lower()


def test_missing_room_video(svc, clips):
    s = accepted(svc); upload_all(svc, s, clips, only=["p1_left", "p1_right", "p2_left"])
    rd = svc.measurement.readiness(s)
    assert not rd.ready and rd.input_status == "expired" and not rd.items[1].ok and rd.items[1].step == "room"
    assert any("missing or has expired" in p for p in rd.items[1].problems) and rd.summary["videos"]["p2_right"] is None


def test_incomplete_frame_selections(svc, ready):
    s, wf = ready
    rd = svc.measurement.readiness(s)
    assert not rd.ready and not rd.items[2].ok and rd.items[2].step == "frames" and rd.input_status == "incomplete"
    ms = use_measurement(svc); choose_and_confirm(wf, sets=2); wf.pairs.pop((2, 1))                    # one pair removed
    rd = ms.readiness(s); assert not rd.ready and "FRAME_SET_INCOMPLETE" in rd.items[2].codes


def test_missing_annotations(svc, ready):
    s, wf = ready; ms = use_measurement(svc); choose_and_confirm(wf, sets=2)
    rd = ms.readiness(s)
    assert not rd.ready and not rd.items[3].ok and "ANNOTATION_MISSING" in rd.items[3].codes and rd.items[3].step == "frames" and rd.summary["annotations_accepted"] == 0


def test_stale_calibration_fingerprint(svc, complete):
    s, wf = complete; ms = use_measurement(svc); ms.readiness(s)
    before = M.binding_of(s.owner, s.contract, wf, svc.room_sources(s))
    s.review = S.review_f8_document(make_doc("GOOD")[1], measured_baseline_mm=230.0)               # a different calibration is accepted
    svc.accept(s)
    rd = ms.readiness(s)
    assert not rd.ready and rd.summary["baseline_mm"] == 230.0 and not rd.items[2].ok            # the selections made under the old calibration are gone, not reused
    assert rd.binding is None or rd.binding["calibration"] != before["calibration"]
    assert not wf.pairs                                                                                 # the workflow was resynchronised: nothing chosen under the old calibration survives


def test_replaced_video(svc, complete, clips):
    s, wf = complete; ms = use_measurement(svc); assert ms.readiness(s).ready
    put(svc.room_store, s.owner, "p1_left", clips[0]["p2_right"].read_bytes(), name="other.mp4")        # a different video in an existing slot
    rd = ms.readiness(s)
    assert not rd.ready and rd.input_status in ("replaced", "incomplete", "expired")
    assert not rd.items[1].ok or not rd.items[2].ok


def test_expired_uploads(svc, complete):
    s, wf = complete; ms = use_measurement(svc); assert ms.readiness(s).ready
    svc.room_store.release_owner(s.owner)                                                              # the uploads are gone (expiry / sweep)
    rd = ms.readiness(s)
    assert not rd.ready and rd.input_status == "expired" and not rd.items[1].ok and rd.items[1].step == "room"
    with pytest.raises(M.MeasurementError):
        ms.start(s)                                                                                    # an expired input is never valid for a new run


# ================================================================ 9. visitor isolation
def test_visitor_isolation(svc, complete, clips):
    a, wfa = complete; ms = use_measurement(svc, runner=lambda ctx: ok_result())
    run_to_completion(ms, a); assert ms.current(a) is not None
    b = accepted(svc); upload_all(svc, b, clips); svc.frame_state(b)
    assert ms.current(b) is None and b.measurement.result is None
    b.measurement.result, b.measurement.binding = a.measurement.result, a.measurement.binding            # even a transplanted record is refused
    b.frames = svc.frame_state(b)["workflow"]; choose_and_confirm(b.frames); annotate_all(b.frames)
    assert ms.current(b) is None and any("discarded" in n for n in b.measurement.notes)
    assert ms.current(a) is not None
    rd_b = ms.readiness(b); rd_a = ms.readiness(a); assert rd_a.binding["owner"] != rd_b.binding["owner"]


# ================================================================ 10-11. the gate
def test_gate_disabled_button_and_message(svc, complete5):
    s, wf = complete5; s.step = "measure"; calls = []
    svc.measurement = M.MeasurementService(svc, runner=lambda ctx: calls.append(ctx))                   # the default gate (the code constant); a runner that must never be called
    at = page_app(svc, s); at.run()
    assert not at.exception, at.exception
    t = texts(at); assert "Room measurement is being validated and is not yet available. Your calibration, frame selections, and annotations can still be reviewed." in t
    b = button(at, "Calculate Wall Dimensions"); assert b.disabled and S.ROOM_PROCESSING_ENABLED is False
    assert "Ready to calculate" in t                                                                  # a short readiness summary is still shown
    with pytest.raises(M.MeasurementError, match="not yet available"):
        svc.measurement.start(s)
    assert calls == [] and s.measurement.job is None and s.measurement.result is None


def test_no_alternate_route_around_the_gate(svc, complete5, monkeypatch):
    s, wf = complete5; s.step = "measure"; calls = []
    for k, v in (("ROOM_PROCESSING_ENABLED", "1"), ("F8_ROOM_PROCESSING_ENABLED", "true"), ("F8_ENABLE_ROOM_PROCESSING", "1")):
        monkeypatch.setenv(k, v)
    svc.measurement = M.MeasurementService(svc, runner=lambda ctx: calls.append(ctx))
    assert svc.measurement.gate_open() is False
    at = page_app(svc, s)
    for q in ("measure", "start", "run", "debug", "room_processing"):
        at.query_params[q] = "1"
    at.run(); assert not at.exception and button(at, "Calculate Wall Dimensions").disabled
    at.session_state["f8i_m_start"] = True; at.run(); assert calls == [] and s.measurement.job is None            # forging a widget key cannot start anything
    # code-level: one call site, one gate, no environment lookup
    srcs = {p.name: p.read_text() for p in HERE.glob("*.py")}
    assert [n for n, t in srcs.items() if re.search(r"measurement\.start\(", t)] == ["f8_measure_page.py"] and srcs["f8_measure_page.py"].count("measurement.start(") == 1
    code_only = lambda n: re.sub(r'""".*?"""', "", srcs[n], flags=re.S)                                  # noqa: E731  (docstrings may describe what is NOT done)
    for n in ("f8_measure_page.py", "f8_measurement.py", "f8_result_view.py"):
        assert not re.search(r"query_params|os\.environ|getenv|experimental_get_query", code_only(n)), n
    assert "ROOM_PROCESSING_ENABLED = False" in srcs["f8_visitor_service.py"] and not [n for n, t in srcs.items() if "ROOM_PROCESSING_ENABLED = True" in t]
    app = srcs["app_f8_integrated.py"]; assert not re.search(r"measurement_|runner=|gate=|ROOM_PROCESSING", app)                      # the factory passes no test seam and never touches the gate constant
    assert re.search(r"VisitorCalibrationService\(store, results, RunnerLimits\.from_env\(\), room_store=room_store\)", app)
    assert [n for n, t in srcs.items() if "room_reconstruction" in t and n not in ("room_reconstruction.py", "reconstruction_worker.py")] == ["f8_measurement.py"]
    for n in ("f8_measure_page.py", "f8_result_view.py", "f8_visitor_pages.py", "f8_frames_page.py", "f8_visitor_service.py"):
        assert "visitor_method_j_adapter" not in srcs[n] and "room_reconstruction" not in srcs[n] and "import pose" not in srcs[n], n


def test_start_takes_only_the_session_and_builds_trusted_values(svc, complete5, monkeypatch):
    s, wf = complete5; seen = []
    ms = M.MeasurementService(svc, gate=lambda: True, runner=lambda ctx: (seen.append(ctx), ok_result())[1])
    svc.measurement = ms
    assert list(inspect.signature(ms.start).parameters) == ["session"]
    import resource_budget as RB
    job = run_to_completion(ms, s); ctx = seen[0]
    assert isinstance(ctx.limits, RR.ReconstructionLimits) and ctx.limits == RR.ReconstructionLimits() and ctx.limits.min_usable_sets is None and ctx.coordinator is RB.app_coordinator()
    assert isinstance(ctx.retention, RR.StoreRetention) and ctx.retention.store is svc.room_store and ctx.retention.owner == s.owner
    assert ctx.owner == s.owner and ctx.workflow is s.frames and re.fullmatch(r"[0-9a-f]{16}", ctx.job_id) and ctx.job_id == job.job_id
    assert set(ctx.sources) == set(RV.ROOM_UPLOAD_SLOTS) and all(str(svc.room_store.root) in v.path for v in ctx.sources.values())
    assert ctx.calibration is s.f6.job.cal and ctx.contract is s.contract


def test_default_runner_calls_the_backend_only_after_the_gate_with_trusted_arguments(svc, complete5, monkeypatch):
    s, wf = complete5; got = {}

    def spy(wf_, owner, contract, cal, sources, limits=None, **kw):
        got.update(wf=wf_, owner=owner, limits=limits, kw=kw); return ok_result()
    monkeypatch.setattr(RR, "run_visitor_reconstruction", spy)                                        # a spy: no worker process is ever started
    ms = M.MeasurementService(svc); svc.measurement = ms
    with pytest.raises(M.MeasurementError):
        ms.start(s)
    assert got == {}                                                                                  # closed gate: not even a call
    ms = M.MeasurementService(svc, gate=lambda: True); svc.measurement = ms
    run_to_completion(ms, s)
    assert got["owner"] == s.owner and got["wf"] is s.frames and isinstance(got["limits"], RR.ReconstructionLimits) and set(got["kw"]) == {"cancel", "coordinator", "retention"}
    assert isinstance(got["kw"]["retention"], RR.StoreRetention) and isinstance(got["kw"]["cancel"], threading.Event)


# ================================================================ 12-14. result rendering (page level, injected results)
def finished(svc, s, result, warnings_ok=True):
    ms = use_measurement(svc, runner=lambda ctx: result); s.step = "measure"
    run_to_completion(ms, s); return ms


def test_success_result_rendering(svc, complete, ):
    s, wf = complete; finished(svc, s, ok_result())
    at = page_app(svc, s); at.run(); assert not at.exception, at.exception
    t = texts(at)
    assert "412 cm" in t and "4.12 m" in t and "272 cm" in t and "2.72 m" in t and "Wall A Measured — Step 1 of 2" in t and "Measurement Complete" not in t.replace("Room Measurement Complete", "") and "Room Width" in t and "Room Height" in t
    assert "27.2 s" in t and "no confidence interval" in t.lower() and "percentage error" in t.lower() and "%" not in re.sub(r"<[^>]+>", "", t.replace("percentage", ""))
    assert "±" not in t and "guaranteed accuracy is claimed" in t.lower()
    assert [m for m in at.markdown if "f8m-value" in m.value]                                       # the dimensions are the prominent elements
    assert "Calculate Wall Dimensions" not in t and "Measurement readiness" not in t.lower() and "SHA-256" not in " ".join(m.value for m in at.markdown)


def test_questionable_result_rendering_keeps_the_warning_visible(svc, complete):
    s, wf = complete; finished(svc, s, ok_result(status="QUESTIONABLE", warnings=["REGISTRATION_HIGH_RMS"]))
    at = page_app(svc, s); at.run(); t = texts(at)
    assert not at.exception and "412 cm" in t and "Some automatic checks raised notes" in t and "The two viewpoints joined less tightly than usual." in t     # advisory detail lives in Technical Details
    assert "Why it matters" in t and "What you can do" in t


def test_success_with_a_warning_is_not_hidden_by_success(svc, complete):
    s, wf = complete; finished(svc, s, ok_result(status="SUCCESS", warnings=["POSE_LOW_INLIER_FRACTION"]))
    at = page_app(svc, s); at.run(); t = texts(at)
    assert RVW.build_view(s.measurement.result)["state"] == RVW.DONE_WARN and "Only a small share of the image matches" in t and "Wall A Measured — Step 1 of 2" in t and "Measurement Complete" not in t.replace("Room Measurement Complete", "") and "Some automatic checks raised notes" in t


def test_failed_result_rendering(svc, complete):
    s, wf = complete; r = failed_result("FEATURE_MATCHING_FAILED", cls="method_j", message="TECHNICAL: too few matches in /tmp/secret"); r["width_mm"], r["height_mm"] = 1234.0, 5678.0    # numbers must never show
    finished(svc, s, r)
    at = page_app(svc, s); at.run(); t = texts(at)
    assert not at.exception and "The measurement could not be completed" in t and VM.failure_text("FEATURE_MATCHING_FAILED").split(".")[0] in t
    assert "123 cm" not in t and "568 cm" not in t and "Estimated Wall Width" not in t and "No dimension is reported" in t
    assert has_button(at, "Go to the room videos")                                                    # F8.31D-2 (H3): the advice is "re-record", so the correction path is the room-video step
    assert "TECHNICAL" not in " ".join(m.value for m in at.error)                                      # technical detail is not the main message
    assert "TECHNICAL" in json.dumps([str(x) for x in at.get("expander")]) or "Technical Details" in " ".join(e.label for e in at.expander)


# ================================================================ 15. every failure message
@pytest.mark.parametrize("code", sorted(VM.FAILURE_TEXT))
def test_every_defined_failure_message_is_presented_without_dimensions(code):
    r = failed_result(code); r["width_mm"] = 1000.0
    v = RVW.build_view(r)
    p = VM.failure_parts(code)
    assert v["accepted"] is False and v["width"] is None and v["height"] is None and v["per_set"] == [] and v["warnings"] == []
    assert v["failure"]["what"] == p["what"] and v["failure"]["next_step"] == p["next_step"] and v["failure"]["retry"]["label"]
    assert v["state"] in (RVW.DONE_FAILED, RVW.BUSY, RVW.NO_MEMORY, RVW.TIMEOUT, RVW.INPUTS_CHANGED) and RVW.summary_document(r)["width"] is None
    assert v["failure"]["technical"][0]["code"] == code and "technical" not in v["failure"]["what"].lower()


def test_unknown_failure_code_and_garbage_results_are_never_a_measurement():
    for bad in (None, {}, "x", dict(schema="other/1", status="SUCCESS", width_mm=1.0), dict(schema=RR.SCHEMA_RESULT, status="SUCCESS", width_mm=None, height_mm=None, dimensions_accepted=True),
                dict(schema=RR.SCHEMA_RESULT, status="SUCCESS", width_mm=5.0, height_mm=5.0, dimensions_accepted=False)):
        v = RVW.build_view(bad); assert v["accepted"] is False and v["width"] is None and v["failure"] is not None, bad
    assert RVW.build_view(failed_result("SOME_NEW_CODE"))["failure"]["what"] == VM.GENERIC_FAILURE[0]


# ================================================================ 16. the seven advisory warnings
def test_all_seven_advisory_warnings_are_presented_with_meaning_reliability_and_action():
    assert len(VM.WARNING_TEXT) == 7
    for code, (meaning, why, action, _) in VM.WARNING_TEXT.items():
        v = RVW.build_view(ok_result(warnings=[code]))
        (c,) = v["warnings"]
        assert c["code"] == code and c["meaning"] == meaning and c["reliability"] == why and c["action"] == action and v["state"] == RVW.DONE_WARN
        assert v["width"]["mm"] == 4123.456789                                                         # a warning never changes a number
    both = RVW.build_view(ok_result(warnings=list(VM.WARNING_TEXT)))
    assert [w["code"] for w in both["warnings"]] == list(VM.WARNING_TEXT)


def test_all_seven_warnings_render_on_the_page(svc, complete):
    s, wf = complete; finished(svc, s, ok_result(status="QUESTIONABLE", warnings=list(VM.WARNING_TEXT)))
    at = page_app(svc, s); at.run(); t = texts(at); assert not at.exception
    for code, (meaning, why, action, _) in VM.WARNING_TEXT.items():
        assert meaning in t and why in t and action in t and code in t


# ================================================================ 17-18. units and provenance
def test_unit_conversion_is_exact_and_rounding_is_display_only():
    d = RVW.dimension(4123.456789)
    assert d["mm"] == 4123.456789 and d["cm"] == 4123.456789 / 10.0 and d["m"] == 4123.456789 / 1000.0 and d["cm_text"] == "412 cm" and d["m_text"] == "4.12 m"
    assert RVW.dimension(0.0) is None and RVW.dimension(float("nan")) is None and RVW.dimension(-5) is None and RVW.dimension(True) is None and RVW.dimension(None) is None
    v = RVW.build_view(ok_result(width=2500.0, height=2400.5))
    assert (v["width"]["cm"], v["width"]["m"], v["height"]["cm"], v["height"]["m"]) == (250.0, 2.5, 240.05, 2.4005) and v["width"]["cm_text"] == "250 cm" and v["height"]["m_text"] == "2.40 m"
    r = ok_result(width=4123.456789); before = copy.deepcopy(r); RVW.build_view(r); RVW.summary_document(r); assert r == before        # the view never edits the result
    doc = RVW.summary_document(r); assert doc["width"] == dict(mm=4123.456789, cm=412.3456789, m=4.123456789)
    per = RVW.build_view(r)["per_set"]; assert per[2]["width"]["mm"] == 4125.456789 and per[2]["height"]["mm"] == 2716.281828
    assert RVW.fmt_cm(None) == "not available"


def test_partial_result_shows_only_the_available_dimension():
    v = RVW.build_view(ok_result(width=3000.0, height=None, partial=True, status="QUESTIONABLE", warnings=["DIMENSION_PARTIAL"]))
    assert v["width"]["cm_text"] == "300 cm" and v["height"] is None and v["partial"] and v["state"] == RVW.DONE_WARN


def test_result_provenance_and_the_downloadable_summary(svc, complete):
    s, wf = complete; r = ok_result()
    v = RVW.build_view(r, "calibrated"); rows = dict(v["provenance"])
    assert dict(RVW.build_view(r, "measured")["provenance"])["Baseline"] == "226 mm (measured by you)"          # F8.31D-2 (M7): the label follows the real provenance
    assert rows["Calibration id"] == "cal123" and rows["Baseline"] == "226 mm (estimated from the calibration)" and rows["Image size"] == "1920 × 1080" and rows["Camera order"] == "left, right"
    assert rows["Frame sets"] == "5" and rows["Accepted boundaries"] == "30" and rows["Ground truth used"] == "no" and rows["Original recipe values used"] == "no"
    assert all(RV.SLOT_INFO[slot][2] in rows for slot in RV.ROOM_UPLOAD_SLOTS) and "SHA-256" not in json.dumps(v["provenance"]) and "OWNERDIGEST123" not in json.dumps(v["provenance"])
    doc = json.dumps(RVW.summary_document(r))
    for banned in ("OWNERDIGEST123", "path", "/Users", "/tmp", "owner", "pixels", "base64"):
        assert banned not in doc, banned
    assert json.loads(doc)["uncertainty"] == "none claimed" and "percent" not in doc.lower().replace("percentage error is computed", "")
    failed_doc = RVW.summary_document(failed_result("FEATURE_MATCHING_FAILED", cls="method_j")); assert failed_doc["width"] is None and failed_doc["height"] is None and failed_doc["dimensions_accepted"] is False


# ================================================================ 19. result invalidation
def _measured(svc, s):
    ms = use_measurement(svc, runner=lambda ctx: ok_result()); run_to_completion(ms, s); assert ms.current(s) is not None; return ms


def test_a_changed_annotation_invalidates_the_result(svc, complete):
    s, wf = complete; ms = _measured(svc, s)
    wf.reject(0, "AL_corner")
    assert ms.current(s) is None and s.measurement.result is None and any("discarded" in n for n in s.measurement.notes)


def test_a_changed_frame_pair_invalidates_the_result(svc, complete):
    s, wf = complete; ms = _measured(svc, s); wf.set_pair(1, 0, 11, 14)
    assert ms.current(s) is None


def test_a_changed_calibration_invalidates_the_result(svc, complete):
    s, wf = complete; ms = _measured(svc, s)
    s.review = S.review_f8_document(make_doc("GOOD")[1], measured_baseline_mm=231.0); svc.accept(s)          # accepted directly: the binding no longer matches
    assert ms.current(s) is None and s.measurement.result is None and any("changed" in n for n in s.measurement.notes)
    s2, wf2 = s, wf; ms = _measured(svc, s) if False else ms
    # the service's own path for a new calibration attempt clears it at once
    s.measurement.result, s.measurement.binding = ok_result(), dict(key="x", owner="o"); s.forget_acceptance()
    assert s.measurement.result is None and any("calibration changed" in n for n in s.measurement.notes)


def test_a_replaced_or_expired_video_invalidates_the_result(svc, complete, clips):
    s, wf = complete; ms = _measured(svc, s)
    put(svc.room_store, s.owner, "p2_right", clips[0]["p2_right"].read_bytes(), name="replacement.mp4")      # the same bytes uploaded again are a NEW upload
    assert ms.current(s) is None


def test_expired_uploads_invalidate_the_result(svc, complete):
    s, wf = complete; ms = _measured(svc, s)
    svc.room_store.release_owner(s.owner); assert ms.current(s) is None


def test_start_over_clears_everything(svc, complete):
    s, wf = complete; ms = _measured(svc, s); s.reset(svc.store, svc.room_store)
    assert s.measurement.result is None and s.measurement.job is None and s.measurement.binding is None


def test_identical_inputs_are_not_measured_twice_but_a_change_allows_it(svc, complete):
    s, wf = complete; ms = _measured(svc, s)
    with pytest.raises(M.MeasurementError, match="already measured"):
        ms.start(s)
    wf.reject(0, "AL_corner")
    with pytest.raises(M.MeasurementError):                                                            # not ready any more: the boundary is gone
        ms.start(s)


def test_a_result_never_appears_for_a_different_binding_on_the_page(svc, complete):
    s, wf = complete; ms = _measured(svc, s); s.step = "measure"
    at = page_app(svc, s); at.run(); assert "412 cm" in texts(at)
    wf.reject(0, "AL_corner"); at.run(); t = texts(at)
    assert "412 cm" not in t and "previous result was cleared" in t and "Estimated Wall Width" not in t and has_button(at, "Recalculate Wall Dimensions")


# ================================================================ 20-24. duplicate submission, worker states, cleanup
def test_duplicate_submission_is_prevented(svc, complete):
    s, wf = complete; gate_ev, calls = threading.Event(), []

    def slow(ctx):
        calls.append(ctx.job_id); gate_ev.wait(10); return ok_result()
    ms = use_measurement(svc, runner=slow); jid = ms.start(s)
    for _ in range(3):
        with pytest.raises(M.MeasurementError, match="already running"):
            ms.start(s)
    results, errs = [], []
    def again():
        try: ms.start(s)
        except M.MeasurementError as e: errs.append(str(e))
    ts = [threading.Thread(target=again) for _ in range(4)]; [t.start() for t in ts]; [t.join() for t in ts]
    assert len(errs) == 4 and len(calls) <= 1 and s.measurement.job.job_id == jid
    gate_ev.set(); wait(s.measurement.job); assert calls == [jid]


def test_running_page_shows_activity_without_a_percentage_and_blocks_navigation(svc, complete):
    s, wf = complete; ev = threading.Event()
    ms = use_measurement(svc, runner=lambda ctx: (ev.wait(10), ok_result())[1]); s.step = "measure"; ms.start(s)
    at = page_app(svc, s); at.session_state["f8i_m_autorefresh"] = False; at.run(); t = texts(at)
    assert not at.exception and "Measuring" in t and "no percentage is shown" in t and not at.get("progress") and has_button(at, "Cancel the measurement")
    assert button(at, "Go back").disabled and not has_button(at, "Calculate Wall Dimensions")
    with pytest.raises(S.ServiceError, match="running"):
        s.go("frames")
    ev.set(); wait(s.measurement.job); s.go("frames"); assert s.step == "frames"


@pytest.mark.parametrize("code,state,fragment", [("WORKER_BUSY", RVW.BUSY, "Another measurement is running"), ("INSUFFICIENT_MEMORY", RVW.NO_MEMORY, "does not have enough free memory"),
                                                  ("WORKER_MEMORY_LIMIT", RVW.NO_MEMORY, "more memory"), ("WORKER_TIMEOUT", RVW.TIMEOUT, "took too long"),
                                                  ("UPLOAD_EXPIRED", RVW.INPUTS_CHANGED, "expired"), ("VIDEO_REPLACED", RVW.INPUTS_CHANGED, "replaced"),
                                                  ("WORKER_CRASHED", RVW.DONE_FAILED, "stopped unexpectedly")])
def test_worker_failure_states_are_shown_as_such_and_never_as_a_measurement(svc, complete, code, state, fragment):
    s, wf = complete; r = failed_result(code, cls="worker_resource" if code.startswith(("WORKER", "INSUFF")) else "visitor_input")
    finished(svc, s, r)
    assert RVW.build_view(s.measurement.result)["state"] == state
    at = page_app(svc, s); at.run(); t = texts(at)
    assert not at.exception and fragment in t and RVW.STATE_TEXT[state][0] in t and "Estimated Wall Width" not in t and "cm" not in re.sub(r"[^\n]*Technical[^\n]*", "", t.replace("cm_", ""))
    if r["failures"][0]["failure_class"] == "worker_resource":
        assert has_button(at, "Try again")                                                             # a retry path when the gate is open (tests only)
    else:
        assert has_button(at, "Go to")


def test_retry_after_a_failed_run_is_allowed(svc, complete):
    s, wf = complete; seq = iter([failed_result("WORKER_BUSY", cls="worker_resource"), ok_result()])
    ms = use_measurement(svc, runner=lambda ctx: next(seq))
    run_to_completion(ms, s); assert RVW.build_view(ms.current(s))["state"] == RVW.BUSY
    run_to_completion(ms, s); assert RVW.build_view(ms.current(s))["state"] == RVW.DONE_OK


def test_worker_cleanup_state_after_success_failure_exception_and_cancel(svc, complete):
    s, wf = complete
    ms = use_measurement(svc, runner=lambda ctx: (ctx.retention.acquire(ctx.sources, 60.0), ok_result())[1])        # the lease is really taken ...
    job = run_to_completion(ms, s)
    assert job.cleanup["retention_released"] is True and job.cleanup["thread_finished"] and not s.measurement.running                     # ... and released
    assert svc.room_store.job_status(s.owner)["slots"]["p1_left"]["leased"] is False
    # an exception inside the runner: typed failure, never a measurement
    def boom(ctx): raise RuntimeError("secret internal detail /private/path")
    s.measurement.discard(); ms = use_measurement(svc, runner=boom); job = run_to_completion(ms, s)
    r = job.result; assert r["status"] == "FAILED" and r["dimensions_accepted"] is False and r["width_mm"] is None and "/private/path" not in json.dumps(RVW.build_view(r)["failure"]["what"])
    # a garbage return value
    s.measurement.discard(); ms = use_measurement(svc, runner=lambda ctx: {"status": "SUCCESS", "width_mm": 1.0}); job = run_to_completion(ms, s)
    assert job.result["status"] == "FAILED" and not RVW.build_view(job.result)["accepted"]
    # cancel: a discarded job can never publish a result afterwards
    ev = threading.Event(); s.measurement.discard()
    ms = use_measurement(svc, runner=lambda ctx: (ev.wait(10), ok_result())[1]); ms.start(s); job = s.measurement.job
    assert ms.cancel(s) and job.cancel.is_set() and job.cancelled
    s.measurement.discard(); ev.set(); wait(job)
    assert s.measurement.result is None and s.measurement.job is None and job.cleanup["thread_finished"]
    assert ms.cancel(s) is False


def test_an_interrupted_run_never_leaves_a_completed_measurement(svc, complete):
    s, wf = complete; ev = threading.Event()
    ms = use_measurement(svc, runner=lambda ctx: (ev.wait(10), ok_result())[1]); ms.start(s)
    assert s.measurement.result is None and ms.current(s) is None and s.measurement.running                  # while running there is no result
    s.reset(svc.store, svc.room_store); ev.set()
    time.sleep(0.2); assert s.measurement.result is None


# ================================================================ 25. navigation between workflow stages
def test_navigation_from_the_measure_page(svc, complete5):
    s, wf = complete5; s.step = "measure"
    at = page_app(svc, s); at.run(); assert not at.exception
    button(at, "Go back").click().run(); assert s.step == "frames"
    s.step = "measure"
    for k in range(4): wf.reject(k, "AL_corner")                                  # F8.31D-6: one failed set is skipped; only ONE usable set left is a blocking problem
    at = page_app(svc, s); at.run()
    assert has_button(at, "Fix this: go to the frames and boundaries")
    button(at, "Fix this: go to the frames and boundaries").click().run(); assert s.step == "frames"


def test_navigation_fix_links_for_room_videos_and_calibration(svc, clips):
    s = accepted(svc); upload_all(svc, s, clips, only=["p1_left", "p1_right", "p2_left"]); s.room_ready = True; s.step = "room"
    s.step = "measure"
    at = page_app(svc, s); at.run(); assert not at.exception and has_button(at, "Fix this: go to the room videos")
    button(at, "Fix this: go to the room videos").click().run(); assert s.step == "room"
    s2 = S.VisitorSession(); s2.step = "measure"; at = page_app(svc, s2); at.run()
    assert not at.exception and has_button(at, "Go to the calibration")
    button(at, "Go to the calibration").click().run(); assert s2.step == "home"


def test_full_step_graph_is_reachable_and_preserves_the_work(svc, complete5):
    s, wf = complete5; svc.room_report(s)
    for a, b in (("home", "ready"), ("ready", "room"), ("room", "frames"), ("frames", "measure"), ("measure", "frames"), ("frames", "room"), ("room", "ready"), ("ready", "home")):
        s.go(a) if s.step != a else None; s.go(b); assert s.step == b
    s.go("measure"); assert wf.readiness().ready and s.frames is wf


# ================================================================ scope guards
def test_the_original_calibration_and_method_j_sources_are_untouched():
    import subprocess
    out = subprocess.run(["git", "status", "--porcelain", "--", *(f"room_measurement/{f}.py" for f in ("pose", "measure", "geometry", "lines", "features", "pipeline", "calibration"))],
                         cwd=MODULE, capture_output=True, text=True, timeout=60).stdout
    assert out.strip() == "", out


def test_no_ml_dl_is_introduced_by_the_new_files():
    for n in ("f8_measurement.py", "f8_result_view.py", "f8_measure_page.py"):
        t = (HERE / n).read_text()
        for banned in ("torch", "tensorflow", "onnx", "keras", "sklearn", "transformers"):
            assert not re.search(rf"^\s*(import|from)\s+{banned}", t, re.M), (n, banned)
        assert "VideoCapture" not in t and "original_recordings" not in t

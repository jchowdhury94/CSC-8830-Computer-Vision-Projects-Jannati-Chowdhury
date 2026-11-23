"""Final visitor-UI cleanup: results-only page, one Calculate action, Download Results, Change Boundaries, New Measurement, clean upload facts.
SYNTHETIC fixtures and INJECTED fake results only (same seams as the C-2A interface tests): no worker, no original recording, no algorithm is touched."""
import json

import pytest

import f8_measure_page as MP
import f8_result_view as RVW
import f8_visitor_pages as P
import room_videos as RV
from test_f8_31b_pages import button, has_button, page_app, texts
from test_f8_31c2a_measurement_interface import (complete, complete5, failed_result, finished, ok_result, ready, svc, use_measurement, wait, accepted, clips, cstore, rstore)   # noqa: F401


def measured_page(svc, s, **kw):
    finished(svc, s, ok_result(**kw)); s.step = "measure"
    at = page_app(svc, s); at.run(); assert not at.exception, at.exception
    return at


def test_results_page_is_results_only(svc, complete):
    s, wf = complete; at = measured_page(svc, s); t = texts(at)
    assert "Wall A Measured — Step 1 of 2" in t and "Measurement Complete" not in t.replace("Room Measurement Complete", "") and "Room Width" in t and "Room Height" in t and "412 cm" in t and "4.12 m" in t and "272 cm" in t and "2.72 m" in t
    assert RVW.ESTIMATE_NOTE in t
    for gone in ("Start room measurement", "Measurement readiness", "Ready to calculate", "COMPETING_EDGE_NEARBY", "SHA-256", "cal123", "OWNERDIGEST123"):
        assert gone not in t, gone
    assert not has_button(at, "Calculate") and not has_button(at, "Start over")       # locking lives in the small Session Options expander only
    assert {b.label for b in at.button} >= {"Change Boundaries", "New Measurement"}
    assert [e.label for e in at.expander if e.label == "Session Options"]


def test_technical_details_is_one_collapsed_section_and_per_frame_set_is_collapsed(svc, complete):
    s, wf = complete; at = measured_page(svc, s, status="QUESTIONABLE", warnings=["REGISTRATION_HIGH_RMS"])
    labels = [e.label for e in at.expander]
    assert labels.count("Technical Details") == 0 and "Wall A Details" in labels      # Wall A alone: per-frame-set and technical detail live in ONE collapsed "Wall A Details"
    assert all(not e.proto.expanded for e in at.expander)                                             # everything is collapsed by default
    assert "Some automatic checks raised notes" in texts(at)


def test_per_frame_set_rows_show_their_own_values_not_the_final_result(svc, complete):
    s, wf = complete; at = measured_page(svc, s, width=4123.0, height=2718.0)
    v = RVW.build_view(s.measurement.result)
    rows = MP._per_set_table(v)
    widths = [r["Wall Width"] for r in rows]; heights = [r["Wall Height"] for r in rows]
    assert len(rows) == 5 and len(set(widths)) == 5 and len(set(heights)) == 5                          # distinct per set (the fake result offsets each set)
    assert widths[0] == "412.30 cm" and widths[4] == "412.70 cm" and heights[0] == "271.80 cm" and heights[4] == "271.40 cm"
    assert v["width"]["mm"] == 4123.0                                                                  # the final value keeps full precision


def test_download_button_is_prominent_and_json_format_unchanged(svc, complete):
    s, wf = complete; at = measured_page(svc, s)
    dl = at.get("download_button")
    assert len(dl) == 1 and dl[0].proto.label == "Download Wall A Results" and dl[0].proto.icon == ":material/download:"
    doc = RVW.summary_document(s.measurement.result)
    assert doc["schema"] == RVW.SUMMARY_SCHEMA and set(doc) >= {"width", "height", "per_frame_set", "provenance"}
    assert json.loads(json.dumps(doc)) == doc and "path" not in " ".join(doc) and set(doc["provenance"]) == {"calibration", "frame_indices", "confirmed_boundaries", "n_frame_sets", "videos", "ground_truth_used", "recipe_a_used", "historical_baseline_used"}


def test_change_boundaries_keeps_uploads_and_result_until_something_changes(svc, complete):
    s, wf = complete; at = measured_page(svc, s)
    button(at, "Change Boundaries").click().run(); assert not at.exception and s.step == "frames"
    assert svc.room_report(s).ready and s.frames is wf and not s.frames.closed and s.measurement.result is not None   # uploads, frames and the result are all still there


def test_new_measurement_needs_confirmation(svc, complete):
    s, wf = complete; at = measured_page(svc, s); owner = s.owner
    button(at, "New Measurement").click().run()
    assert s.owner == owner and s.accepted is not None and s.measurement.result is not None            # nothing cleared yet
    assert has_button(at, "Yes, start a new measurement") and has_button(at, "Cancel") and "deletes" in texts(at)
    button(at, "Cancel").click().run(); assert s.accepted is not None and not has_button(at, "Yes, start a new measurement")
    button(at, "New Measurement").click().run(); button(at, "Yes, start a new measurement").click().run()
    assert not at.exception and s.accepted is None and s.measurement.result is None and s.step == "home" and s.owner != owner


def test_calculate_runs_once_per_click_then_shows_results_only(svc, complete5):
    s, wf = complete5; calls = []
    def runner(ctx):
        calls.append(ctx.job_id); return ok_result()
    use_measurement(svc, runner=runner, limits=None); s.step = "measure"
    at = page_app(svc, s); at.run(); assert not at.exception
    assert button(at, "Calculate Wall Dimensions").proto.type == "primary" and not button(at, "Calculate Wall Dimensions").disabled and "Ready to calculate" in texts(at)
    assert not at.get("download_button")
    button(at, "Calculate Wall Dimensions").click().run()
    wait(s.measurement.job); at.run(); at.run()
    t = texts(at); assert calls and len(calls) == 1 and "Wall A Measured — Step 1 of 2" in t and "Measurement Complete" not in t.replace("Room Measurement Complete", "") and not has_button(at, "Calculate Wall Dimensions")


def test_results_layout_accepts_a_further_measurement_without_a_redesign(svc, complete, monkeypatch):
    s, wf = complete
    at = measured_page(svc, s); assert [m["key"] for m in RVW.build_view(s.measurement.result)["dimensions"]] == ["wall_width", "wall_height"] and "Depth" not in texts(at) and "depth" not in texts(at).lower()
    monkeypatch.setattr(RVW, "MEASUREMENTS", RVW.MEASUREMENTS + (("room_depth", "Estimated Room Depth", "depth_mm"),))
    r = ok_result(); r["depth_mm"] = 5000.0
    v = RVW.build_view(r); assert [m["key"] for m in v["dimensions"]] == ["wall_width", "wall_height", "room_depth"] and v["dimensions"][2]["value"]["cm_text"] == "500 cm"


def test_failed_result_stays_visible_and_actionable(svc, complete):
    s, wf = complete; finished(svc, s, failed_result("FEATURE_MATCHING_FAILED", cls="method_j")); s.step = "measure"
    at = page_app(svc, s); at.run(); t = texts(at)
    assert not at.exception and "The measurement could not be completed" in t and "Estimated Wall Width" not in t and has_button(at, "Go to")
    assert [e.label for e in at.expander].count("Technical Details") == 1


def test_upload_card_facts_are_clean():
    class R:  # the same facts the room report holds
        facts = dict(width=1920, height=1080, fps=30.0, frame_count=300, duration_s=10.04)
    assert P._slot_facts(R) == "1920 × 1080 px · 10.0 s · 30 fps"

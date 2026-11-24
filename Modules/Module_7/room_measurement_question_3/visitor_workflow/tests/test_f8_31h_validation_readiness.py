"""F8.31H: boundary-selection instructions, results-page readiness, and the isolated (test-only) development reconstruction path.  SYNTHETIC fixtures only.

No original recording is opened, Method J is not edited, ROOM_PROCESSING_ENABLED stays False.  The development reconstruction entry point is the existing test-side harness
(``room_reconstruction.run_request`` with the synthetic ``assets`` fixture of test_f8_31c1_reconstruction): it lives in ``tests/``, is not imported by any live file, and
cannot enable the visitor measurement (guarded below and by the architectural tests of F8.31C-1 / C-2A).
"""
import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

import f8_frames_page as FP
import f8_measure_page as MP
import f8_result_view as RVW
import f8_ui_text as UT
import f8_visitor_service as S
import room_frames as RF
import room_reconstruction as RR
from test_f8_31b_frames import LINE_POINTS, accepted, clips, cstore, ready, rstore, svc, upload_all          # noqa: F401  (fixtures)
from test_f8_31b_pages import button, page_app, texts
from test_f8_31c1_reconstruction import assets, call_params, make_request                                      # noqa: F401  (module-scoped synthetic fixture, harness helpers)
from test_f8_31d6_simplified_frames import auto, applied, seed_all                                              # noqa: F401
from test_f8_31f_boundary_review import patch_suggestions, _app

MODULE = Path(__file__).resolve().parents[2]
VW = MODULE / "visitor_workflow"

WALL_HEADING, WALL_BODY = "Select the VERTICAL wall corner", "Click two points along the same wall corner: one near the top and one near the bottom."
CEIL_HEADING = "Select the HORIZONTAL ceiling boundary"
CEIL_BODY = "Click two points along the edge where the wall meets the ceiling: one toward the left and one toward the right."
VARIABILITY = ("Wall measurements are estimated using manually selected wall and ceiling boundaries. Small differences in point placement can affect the calculated width and height. "
               "Therefore, repeating the measurement with the same videos may produce slightly different results.")
EXPECT = {"AL_corner": "wall", "PL_corner": "wall", "PL_ceilF": "ceil", "AR_corner": "wall", "PR_corner": "wall", "PR_ceilF": "ceil"}


# ================================================================== 1-2. boundary instructions and example illustrations
def test_the_guide_data_matches_the_required_wording_and_covers_all_six_boundaries():
    assert tuple(RF.REQUIRED_LINES) == tuple(EXPECT)
    for line, kind in EXPECT.items():
        g = UT.boundary_guide(line)
        assert g["heading"] == (WALL_HEADING if kind == "wall" else CEIL_HEADING) and g["body"] == (WALL_BODY if kind == "wall" else CEIL_BODY)
        assert g["orientation"] == ("vertical" if kind == "wall" else "horizontal")
    assert UT.boundary_guide("AL_corner")["note"] is None
    assert "slightly slanted" in UT.boundary_guide("PL_ceilF")["note"] and "perspective" in UT.boundary_guide("PL_ceilF")["note"] and "forcing a perfectly horizontal line" in UT.boundary_guide("PL_ceilF")["note"]
    with pytest.raises(KeyError):
        UT.boundary_guide("XX_unknown")


def test_each_of_the_six_screens_shows_its_own_instruction_and_example(svc, ready, patch_suggestions):
    patch_suggestions(); s, wf = ready; at = _app(svc, s)
    seen = []
    for k, line in enumerate(RF.REQUIRED_LINES):
        t = texts(at); kind = EXPECT[line]
        assert f"{k + 1} of 6" in t and UT.boundary_label(line) in t
        wanted, other = ((WALL_HEADING, CEIL_HEADING) if kind == "wall" else (CEIL_HEADING, WALL_HEADING))
        assert wanted in t and other not in t, line
        assert (WALL_BODY if kind == "wall" else CEIL_BODY) in t and (CEIL_BODY if kind == "wall" else WALL_BODY) not in t
        assert ("perspective" in t) == (kind == "ceil"), line                              # the slant note belongs to ceiling screens only
        svgs = [m.value for m in at.markdown if "<svg" in m.value]
        assert len(svgs) == 1
        assert ("wall corner" in svgs[0]) == (kind == "wall") and ("ceiling / wall" in svgs[0]) == (kind == "ceil")        # the example matches the active type
        assert "x1='90' y1='45' x2='90' y2='150'" in svgs[0] if kind == "wall" else "x1='20' y1='60' x2='170' y2='60'" in svgs[0]   # vertical vs horizontal red edge
        seen.append(line)
        button(at, "Looks Correct").click().run(); assert not at.exception, at.exception
        button(at, "Finish" if k == 5 else "Next").click().run(); assert not at.exception, at.exception
    assert seen == list(EXPECT)


def test_instruction_sits_above_the_image_and_adds_no_snapping():
    src = (VW / "f8_frames_page.py").read_text()
    body = src[src.index("def _step2"):src.index("# ------------------------------------------------------------------ 3 · review")]
    assert body.index("_guide(line)") < body.index("streamlit_image_coordinates") < body.index("_toolbar(") + 10_000
    assert body.index("_guide(line)") < body.index("_toolbar(")
    for forbidden in ("snap", "horizontal_only", "vertical_only", "np.round", "x = pts[0][0]"):
        assert forbidden not in body


def test_clicks_are_stored_exactly_as_clicked_without_axis_snapping(svc, ready, patch_suggestions):
    patch_suggestions(available=False); s, wf = ready
    slanted_wall = [(1000.0, 200.0), (1013.0, 900.0)]; slanted_ceiling = [(300.0, 300.0), (1700.0, 241.0)]
    for pt in slanted_wall:
        wf.seed_click("AL_corner", *pt)
    for pt in slanted_ceiling:
        wf.seed_click("PL_ceilF", *pt)
    assert wf.seed_points("AL_corner") == [list(p) for p in slanted_wall] and wf.seed_points("PL_ceilF") == [list(p) for p in slanted_ceiling]


# ================================================================== 3, 12. results page: computed outputs, variability, annotation preservation
def good_result(assets):
    return copy.deepcopy(assets.result)


def test_view_model_reports_the_backend_numbers_not_reference_constants(assets):
    r = good_result(assets); v = RVW.build_view(r)
    assert v["accepted"] and v["width"]["mm"] == r["width_mm"] and v["height"]["mm"] == r["height_mm"]
    r2 = copy.deepcopy(r); r2["width_mm"] += 123.0; r2["height_mm"] -= 77.0
    v2 = RVW.build_view(r2)
    assert v2["width"]["mm"] == r["width_mm"] + 123.0 and v2["height"]["mm"] == r["height_mm"] - 77.0              # follows the result, so nothing is a stored reference
    for src in ("f8_result_view.py", "f8_measure_page.py"):
        t = (VW / src).read_text()
        assert not re.search(r"2879|2380|\b290\b|\b240\b|method_j_experiment_reference|wall_boundary_annotations|DEFAULT_CALIBRATION", t), src


def test_variability_notice_is_shown_only_for_accepted_results(assets):
    ok = RVW.build_view(good_result(assets))
    assert ok["variability"] == dict(title="Measurement Variability", text=VARIABILITY)
    failed = RVW.build_view(dict(schema=RVW.SCHEMA_RESULT, status="FAILED", failures=[dict(code="FLOOR_PLANE_FAILED", failure_class="method_j")], dimensions_accepted=False))
    assert failed["variability"] is None and failed["width"] is None and failed["height"] is None and failed["boundaries"] == []


class _Meas:
    running = False
    def gate_open(self): return False


class _Svc:
    measurement = _Meas()
    def calibration_context(self, sess): return dict(baseline_source="file")


def _render(result):
    from streamlit.testing.v1 import AppTest
    src = ("def run():\n    import sys as _s\n    import streamlit as st\n    for p in %r:\n        if p not in _s.path: _s.path.insert(0, p)\n"
           "    import f8_measure_page as MP\n    MP.render_result(st.session_state['_svc'], None, st.session_state['_res'])\nrun()\n") % ((str(MODULE), str(VW), str(MODULE / "f8_chunk_poc"), str(MODULE.parent.parent)),)
    at = AppTest.from_string(src, default_timeout=60); at.session_state["_svc"] = _Svc(); at.session_state["_res"] = result; at.run()
    assert not at.exception, at.exception
    return at


def test_results_page_shows_computed_dimensions_and_the_variability_notice_near_them(assets):
    r = good_result(assets); at = _render(r)
    t = texts(at); w = RVW.fmt_cm(r["width_mm"]); h = RVW.fmt_cm(r["height_mm"])
    assert "Measurement Complete" in t and w in t and h in t and "Estimated Wall Width" in t and "Estimated Wall Height" in t
    assert RVW.ESTIMATE_NOTE in t and t.index("Estimated Wall Width") < t.index(RVW.ESTIMATE_NOTE)               # the estimate note sits right below the two measurements


def test_failed_result_shows_no_dimension_no_variability_and_names_the_stage(assets):
    r = dict(schema=RVW.SCHEMA_RESULT, status="FAILED", dimensions_accepted=False, width_mm=2900.0, height_mm=2400.0,       # numbers inside a FAILED result must never be shown
             failures=[dict(code="FLOOR_PLANE_FAILED", failure_class="method_j", stage="floor_plane", message="plane fit needs 3 points")], warnings=[], provenance={})
    at = _render(r); t = texts(at)
    assert not any("Measurement Variability" in i.value for i in at.info)
    assert "2900" not in t and "290 cm" not in t and "2400" not in t and "240 cm" not in t and "No dimension is reported" in t
    assert any("floor_plane" in str(c) for tb in at.table for c in tb.value.to_dict("records")) or "FLOOR_PLANE_FAILED" in str(at.table[0].value if at.table else "")


def test_confirmed_boundary_coordinates_are_preserved_in_the_summary(assets):
    r = good_result(assets)
    r["provenance"]["annotations"] = [dict(frame_set=i, line=n, points_distorted_full_px=[[10.5 + i, 20.25], [30.0, 400.75 + i]]) for i in range(2) for n in RF.REQUIRED_LINES]
    rows = RVW.boundary_rows(r["provenance"]); assert len(rows) == 12
    assert rows[0] == dict(frame_set=1, line="AL_corner", label="Viewpoint 1 · Left camera · Wall corner", points=[[10.5, 20.25], [30.0, 400.75]])
    doc = RVW.summary_document(r)
    assert doc["provenance"]["confirmed_boundaries"] == rows and json.dumps(doc, allow_nan=False)
    assert RVW.boundary_rows({"annotations": [dict(line="x")]}) == [] and RVW.boundary_rows(None) == []                 # malformed entries are dropped, nothing invented


# ================================================================== 4-8, 10. isolated development reconstruction path (synthetic)
def test_development_path_transfers_calibration_frames_boundaries_and_units(assets):
    r = assets.result; req = assets.request; pr = r["provenance"]
    assert pr["calibration"]["calibration_digest"] == assets.ctx.digest() and pr["calibration"]["baseline_unit"] == "mm"          # 4 calibration transfer
    assert pr["frame_indices"] == [dict(t) for t in req.frame_table] or pr["frame_indices"] == [dict(t) for t in req.frame_table]  # 5 frame-index transfer
    assert [p["frame_indices"] for p in r["per_frame_set"]] == [dict(t) for t in req.frame_table]
    assert len(req.line_sets) == 5 and all(set(ls) == set(RR.REQUIRED_LINES) for ls in req.line_sets)                              # 6 boundary transfer
    assert r["width_mm"] > 100 and r["height_mm"] > 100 and abs(r["width_mm"] / 1000 - RVW.mm_to_m(r["width_mm"])) < 1e-12        # 7 millimetres in, centimetres only at display
    assert RVW.dimension(r["width_mm"])["cm"] == r["width_mm"] / 10.0


def test_identical_inputs_give_identical_results(assets, tmp_path):                                                                 # 8 determinism
    import resource_budget as RB
    again = RR.run_request(assets.request, RR.ReconstructionLimits(), workspace_root=tmp_path, coordinator=RB.ResourceCoordinator(root=tmp_path / "rb", cfg=RB.BudgetConfig(ceiling_mb=65536.0),
                           container=dict(limit_mb=None, reliable=False, reason="test isolation")))
    a, b = assets.result, again
    assert (a["width_mm"], a["height_mm"]) == (b["width_mm"], b["height_mm"])
    assert [(p["width_mm"], p["height_mm"]) for p in a["per_frame_set"]] == [(p["width_mm"], p["height_mm"]) for p in b["per_frame_set"]]


def test_invalid_or_incomplete_inputs_fail_typed_without_dimensions(assets, tmp_path):                                              # 9, 10
    bad = copy.copy(assets.request); bad_lines = tuple({k: v for k, v in ls.items() if k != "PR_ceilF"} for ls in assets.request.line_sets)
    bad = RR.ReconstructionRequest(assets.request.owner, assets.request.calibration, assets.request.frames, bad_lines, {}, (), assets.request.frame_table)
    import resource_budget as RB
    r = RR.run_request(bad, RR.ReconstructionLimits(), workspace_root=tmp_path, coordinator=RB.ResourceCoordinator(root=tmp_path / "rb", cfg=RB.BudgetConfig(ceiling_mb=65536.0),
                       container=dict(limit_mb=None, reliable=False, reason="test isolation")))
    assert r["status"] == RR.FAILED and r["failures"] and r["width_mm"] is None and r["height_mm"] is None and not r["dimensions_accepted"]
    v = RVW.build_view(r); assert v["width"] is None and v["height"] is None and v["failure"] is not None and v["variability"] is None


def test_the_development_path_is_test_side_only_and_does_not_touch_the_gate_or_originals():                                          # 13-15
    assert S.ROOM_PROCESSING_ENABLED is False and re.search(r"^ROOM_PROCESSING_ENABLED = False\b", (VW / "f8_visitor_service.py").read_text(), re.M)
    live = [p for p in list(VW.glob("*.py")) + list((MODULE / "f8_chunk_poc").glob("*.py")) + list(MODULE.glob("*.py")) if p.name not in ("room_reconstruction.py", "reconstruction_worker.py", "f8_measurement.py")]
    for p in live:
        assert "test_f8_31h" not in p.read_text(errors="ignore"), p.name
    assert "ROOM_PROCESSING_ENABLED = True" not in "".join(p.read_text(errors="ignore") for p in VW.glob("*.py"))


def test_originals_are_unmodified_in_git():
    import subprocess
    root = MODULE.parents[2]
    out = subprocess.run(["git", "diff", "--name-only", "--", "Modules/Module_7/room_measurement_question_3/room_measurement", "Modules/Module_7/room_measurement_question_3/calibration", "Modules/Module_7/room_measurement_question_3/config", "Modules/Module_7/room_measurement_question_3/outputs"],
                         cwd=root, capture_output=True, text=True).stdout.split()
    protected = [f for f in out if re.search(r"/(features|pose|measure|geometry|lines|pipeline|calibration|frames|uncertainty)\.py$|/config/|room_video_recipe|method_j_experiment_reference|frame_manifest", f)]
    assert protected == [], protected

"""Wall B manual-boundary fallback.  SYNTHETIC clips only; the refinement failure is forced by tightening ONE Stage D threshold for the target boundaries (real edge search, real result)."""
from dataclasses import replace

import numpy as np
import pytest

import f8_frames_page as FP
import f8_result_view as RVW
import f8_visitor_service as S
import room_frames as RF
import wall_b_fallback as FB
from room_measurement import geometry as G
from room_measurement import visitor_annotations as V
from test_f8_31b_frames import LINE_POINTS, choose_and_confirm, svc, accepted, clips, cstore, rstore, ready   # noqa: F401
from test_f8_31b_pages import page_app, texts

TARGETS = ("AR_corner", "PR_corner")


@pytest.fixture
def forced_failure(monkeypatch):
    orig = V.process_annotation
    def forced(raw, ctx, cal, limits=V.DEFAULT_LIMITS, **kw):
        if V.line_name(raw["placement"], raw["camera"], raw["boundary_type"]) in TARGETS:
            limits = replace(limits, min_edge_points_fail=10 ** 6)
        return orig(raw, ctx, cal, limits, **kw)
    monkeypatch.setattr(V, "process_annotation", forced)


def applied(wf, sets=2):
    choose_and_confirm(wf, sets=sets)
    for n in RF.REQUIRED_LINES:
        wf.seed_set(n, LINE_POINTS[n.split("_")[1]])
    return wf.apply_seeds()


def test_forced_failure_is_a_refinement_quality_failure(ready, forced_failure):
    s, wf = ready; out = applied(wf)
    assert all(out[(i, n)] == "FAIL" for i in range(2) for n in TARGETS) and all(out[(i, n)] != "FAIL" for i in range(2) for n in RF.REQUIRED_LINES if n not in TARGETS)
    assert not wf.readiness().ready and all(FB.eligible(wf.ann[(i, n)].result) for i in range(2) for n in TARGETS)


def test_fallback_uses_exactly_the_visitors_line_and_keeps_refined_boundaries(ready, forced_failure):
    s, wf = ready; applied(wf)
    before = {(i, n): wf.ann[(i, n)].result for i in range(2) for n in RF.REQUIRED_LINES if n not in TARGETS}
    done = FB.apply_manual_fallback(wf)
    assert sorted(done) == sorted((i, n) for i in range(2) for n in TARGETS)
    assert all(wf.ann[k].result is r and wf.ann[k].accepted for k, r in before.items())                  # refined boundaries untouched
    assert FB.apply_manual_fallback(wf) == []                                                              # idempotent
    rd = wf.readiness(deep=True); assert rd.ready, rd.to_dict()
    for i in range(2):
        for n in TARGETS:
            st = wf.ann[(i, n)]; res = st.result; line = res.downstream_line
            exp = V.line_from_points(*np.asarray(res.transforms["points_undistorted_full"], float))
            assert line["kind"] == exp["kind"] and line["a"] == pytest.approx(exp["a"]) and line["b"] == pytest.approx(exp["b"]) and tuple(line["span"]) == pytest.approx(exp["span"])      # the clicked line, nothing else
            P = np.asarray(line["pts"]); w = (line["a"] * (P[:, 1] if line["kind"] == "v" else P[:, 0]) + line["b"]); assert len(P) >= 8 and np.allclose(w, P[:, 0] if line["kind"] == "v" else P[:, 1], atol=1e-6)
            assert res.status is V.Status.WARNING and FB.FALLBACK_CODE in res.warning_codes and res.diagnostics["manual_fallback"] is True and res.diagnostics["automatic_refinement_failure"]
            assert st.acknowledged_codes and FB.FALLBACK_CODE in st.acknowledged_codes
    ad = wf.lines_for_adapter()                                                                            # the form Method J reads (measure_wall -> line_to_h)
    assert all(np.isfinite(G.line_to_h(ad[i][n])).all() for i in range(2) for n in RF.REQUIRED_LINES)
    assert FB.fallback_report(wf) == [dict(line=n, sets=[1, 2]) for n in TARGETS]


def test_invalid_input_is_never_bypassed(ready):
    s, wf = ready; choose_and_confirm(wf, sets=2)
    for n in RF.REQUIRED_LINES:
        wf.seed_set(n, LINE_POINTS[n.split("_")[1]])
    wf.seed_set("AR_corner", LINE_POINTS["ceilF"])                                                         # a horizontal line for a wall corner: the clicked line itself is invalid
    wf.apply_seeds(); res = wf.ann[(0, "AR_corner")].result
    assert res.status is V.Status.FAIL and "UNREASONABLE_ORIENTATION" in res.failure_codes and not FB.eligible(res)
    assert FB.apply_manual_fallback(wf) == [] and not wf.ann[(0, "AR_corner")].accepted and not wf.readiness().ready


def test_successful_refinement_is_never_replaced(ready):
    s, wf = ready; applied(wf)
    assert FB.apply_manual_fallback(wf) == [] and wf.readiness(deep=True).ready and not FB.fallback_report(wf)


def test_page_applies_it_to_wall_b_only(svc, clips, forced_failure, monkeypatch):
    s = accepted(svc); calls = []
    real = FB.apply_manual_fallback
    monkeypatch.setattr(FB, "apply_manual_fallback", lambda wf: (calls.append(s.active_wall), real(wf))[1])
    from test_f8_31a_room_videos import put
    for wall in ("A", "B"):
        s.switch_wall(wall)
        for slot, p in clips[0].items():
            put(svc.room_store, s.room_owner, slot, p.read_bytes(), name=p.name)
        wf = svc.frame_state(s)["workflow"]; assert svc.frame_state(s)["state"] == "ok"; applied(wf)
        assert not wf.readiness().ready
    s.switch_wall("A"); s.step = "frames"; at = page_app(svc, s); at.run(); assert not at.exception, at.exception
    t = texts(at); assert "Boundary refinement failed" in t and calls == [] and not s.frames.ann[(0, "AR_corner")].accepted            # Wall A: unchanged behaviour
    s.switch_wall("B"); s.step = "frames"; at = page_app(svc, s); at.run(); assert not at.exception, at.exception
    t = texts(at); assert calls == ["B"] and s.frames.ann[(0, "AR_corner")].accepted and "the line you marked is used there" in t and "Boundary refinement failed" not in t
    assert s.frames.readiness(deep=True).ready
    s.switch_wall("A"); assert not s.frames.ann[(0, "AR_corner")].accepted                                  # Wall A still failing, independent


def test_technical_details_identify_fallback_boundaries():
    prov = dict(annotations=[dict(frame_set=0, line="AR_corner", accepted_warning_codes=["MANUAL_BOUNDARY_FALLBACK"]), dict(frame_set=1, line="AR_corner", accepted_warning_codes=["MANUAL_BOUNDARY_FALLBACK"]),
                             dict(frame_set=0, line="AL_corner", accepted_warning_codes=[])])
    rows = dict(RVW.provenance_rows(prov)); key = [k for k in rows if "manual line" in k]
    assert len(key) == 1 and "AR_corner (frame sets 1, 2)" in rows[key[0]] and "AL_corner" not in rows[key[0]]
    assert not [k for k, _ in RVW.provenance_rows(dict(annotations=[dict(frame_set=0, line="AL_corner", accepted_warning_codes=[])])) if "manual line" in k]


def test_wall_b_reconstruction_still_calls_the_original_method_j():
    import inspect
    from room_measurement import visitor_method_j_adapter as A
    src = inspect.getsource(A.measure_sets_original); assert "M.measure_wall(" in src and "wall_b_fallback" not in src
    for mod in ("visitor_method_j_adapter", "measure", "geometry", "lines", "visitor_annotations"):
        import importlib; assert "wall_b_fallback" not in inspect.getsource(importlib.import_module("room_measurement." + mod))


def test_reconstruction_preparation_accepts_the_fallback_and_records_it(svc, ready, forced_failure):
    import room_reconstruction as RR
    s, wf = ready; applied(wf); FB.apply_manual_fallback(wf)
    prep = RR.prepare(wf, s.owner, s.contract, s.f6.job.cal, svc.room_sources(s), RR.ReconstructionLimits(required_frame_sets=None))
    assert prep.request is not None, prep.failures
    flagged = {(a["frame_set"], a["line"]) for a in prep.request.annotations if "MANUAL_BOUNDARY_FALLBACK" in a["accepted_warning_codes"]}
    assert flagged == {(i, n) for i in range(2) for n in TARGETS}                                     # only the failed boundaries, identified in the provenance

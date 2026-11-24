"""Wall B only: use the visitor's own two-point boundary line when automatic edge refinement cannot fit it (for example a wall corner partly hidden by furniture).

What Method J consumes: ``measure.measure_wall`` reads each boundary only through ``geometry.line_to_h`` (``kind``, ``a``, ``b``).  The fitted line's ``pts`` are used only if the lines are
re-expressed under a different calibration (``lines.retarget_lines`` refits them).  A refined line and the visitor's clicked line therefore have the SAME form: the clicked line is what
``visitor_annotations.process_annotation`` already computes as ``initial_line`` (the two clicks mapped to the full-resolution frame, undistorted with the accepted calibration, then
``line_from_points``).  This module builds the fitted-line dict from that line and nothing else: no coordinate is invented, estimated or tuned.

Rules (all enforced here):
  * Only a (frame set, boundary) whose automatic refinement FAILED, and failed ONLY for refinement-quality reasons (no usable edge, too few edge points, low coverage, ...), is touched.
    A failure about the visitor's own input (points too close, wrong orientation, outside the image, wrong frame, size mismatch ...) is never bypassed: the line itself is not valid.
  * A boundary that was refined successfully (PASS or WARNING) is never replaced.
  * The substituted line is recorded with the warning code ``MANUAL_BOUNDARY_FALLBACK`` (it reaches the result's provenance as an accepted warning code) and ``diagnostics['manual_fallback']``.
  * The same seed is applied to every frame set of the wall, exactly as the original experiment and ``RoomFrameWorkflow.apply_seeds`` do (static scene, rigid rig); the frame sets whose
    refinement succeeded keep their own refined line.
  * Method J itself is not changed or bypassed: it still validates the geometry (floor plane, corner/floor intersection, registration) and fails with a typed code when the lines are not usable.
The caller (the frames page) applies this to Wall B only.  Wall A never calls it.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from room_measurement import visitor_annotations as V

FALLBACK_CODE = "MANUAL_BOUNDARY_FALLBACK"
# Refinement-quality failures only (the clicked line itself passed every input check before the edge search ran).
ELIGIBLE_FAILURES = frozenset({V.Code.REFINE_FAILED, V.Code.FEW_POINTS, V.Code.LOW_COVERAGE, V.Code.LOW_INLIERS, V.Code.HIGH_RMS, V.Code.WEAK_EDGE, V.Code.AT_LIMIT, V.Code.ANGLE_CHANGE})
SAMPLE_STEP_PX = 6.0


def eligible(res) -> bool:
    """True when ``res`` is a FAILED automatic refinement of a valid clicked line that failed only for refinement-quality reasons."""
    if res is None or res.status is not V.Status.FAIL or res.initial_line is None:
        return False
    codes = res.failure_codes
    return bool(codes) and all(c in ELIGIBLE_FAILURES for c in codes)


def manual_line(initial_line: dict) -> dict:
    """The fitted-line dict (keys kind, a, b, n, N, rms, span, pts) for the visitor's clicked line.  ``pts`` are samples ON that line (needed only if lines are retargeted); ``rms`` is 0
    because nothing was fitted to image data; ``n`` / ``N`` count those samples."""
    line = dict(kind=initial_line["kind"], a=float(initial_line["a"]), b=float(initial_line["b"]), span=(float(initial_line["span"][0]), float(initial_line["span"][1])))
    u0, u1 = line["span"]
    u = np.arange(u0, u1 + 1e-6, SAMPLE_STEP_PX)
    if len(u) < 8:
        u = np.linspace(u0, u1, 12)
    pts = np.asarray(V._points_on(line, u), float)
    if not np.isfinite(pts).all():
        raise ValueError("the clicked line is not finite")
    line.update(n=int(len(pts)), N=int(len(pts)), rms=0.0, pts=pts)
    return line


def apply_manual_fallback(wf) -> list:
    """Accept the visitor's clicked line for every eligible failed (frame set, boundary) of ``wf`` (a Wall B workflow).  Idempotent.  Returns ``[(frame_set_index, line_name), ...]`` applied now."""
    import room_frames as RF
    if not (wf.seeds_complete() and wf.seeds_applied()):
        return []
    done = []
    for i in range(wf.n_sets):
        for name in RF.REQUIRED_LINES:
            st = wf.ann.get((i, name))
            if st is None or st.accepted or st.error or not eligible(st.result):
                continue
            res = st.result
            try:
                line = manual_line(res.initial_line)
            except (ValueError, KeyError, TypeError):
                continue
            original = [x for x in res.issues if x.status is V.Status.FAIL]
            note = V.Issue(V.Status.WARNING, FALLBACK_CODE,
                           "Automatic edge refinement could not fit this boundary (" + ", ".join(x.code for x in original) + "), so the visitor's own two-point line is used unrefined.")
            issues = [x for x in res.issues if x.status is not V.Status.FAIL] + [note]
            diag = dict(res.diagnostics or {}, manual_fallback=True, automatic_refinement_failure=[x.to_dict() for x in original])
            new = replace(res, status=V.Status.WARNING, issues=issues, diagnostics=diag, refined_line=V._public_line(line), downstream_line=line, downstream_permitted=True)
            st.result, st.accepted, st.auto, st.acknowledged_codes = new, True, True, tuple(new.warning_codes)
            done.append((i, name))
    return done


def fallback_report(wf) -> list:
    """``[dict(line, sets)]``: the boundaries of ``wf`` whose accepted line is the visitor's manual line (1-based frame sets), for the visitor notice and the technical details."""
    import room_frames as RF
    out = []
    for name in RF.REQUIRED_LINES:
        sets = [i + 1 for i in range(wf.n_sets) if (st := wf.ann.get((i, name))) is not None and st.accepted and st.result is not None and (st.result.diagnostics or {}).get("manual_fallback")]
        if sets:
            out.append(dict(line=name, sets=sets))
    return out

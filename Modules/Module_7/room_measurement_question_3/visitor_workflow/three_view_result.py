"""Three-viewpoint Wall B: the visitor RESULT and its view model (pure Python: no Streamlit, no OpenCV, no backend import).

The result uses the SAME envelope as the two-viewpoint result (``schema f8.visitor_reconstruction/1``, ``status``, ``dimensions_accepted``, ``width_mm``, ``failures``, ``warnings``, ``provenance``), so
the measurement service, the saved-result binding and the final room page work unchanged: Wall B's horizontal measurement is stored as ``width_mm`` (the room LENGTH), exactly like a two-viewpoint
Wall B.  It adds ``method = "three_viewpoint_wall_b"`` and a ``three_view`` block with everything specific to this method: per-frame classifications, hard failures, warnings and quality indicators.

Rules kept here:
* the aggregate length exists only when at least two frame sets were geometrically usable (``three_view_quality``); otherwise the result is FAILED with
  ``THREE_VIEW_INSUFFICIENT_EVIDENCE`` and NO length, not even a single frame's;
* warnings are returned and displayed WITH the number; a result with any warning is QUESTIONABLE ("approximate");
* no height is produced (Wall A supplies the room height), no ground truth and no percentage error is ever used or shown;
* the wording of the limitations is the owner's fixed statement.
"""
from __future__ import annotations

import json
import math

import visitor_messages as VM

SCHEMA_RESULT = "f8.visitor_reconstruction/1"           # the shared envelope (duplicated on purpose: this module must not import the backend)
METHOD = "three_viewpoint_wall_b"
SUCCESS, QUESTIONABLE, FAILED = "SUCCESS", "QUESTIONABLE", "FAILED"
MEASURED, MEASURED_WITH_WARNINGS, INSUFFICIENT_EVIDENCE = "MEASURED", "MEASURED_WITH_WARNINGS", "INSUFFICIENT_EVIDENCE"
LIMITATIONS = ("Room dimensions are approximate estimates obtained from stereo videos. Accuracy may be affected by camera calibration, camera positioning, video quality, partially hidden corners, "
               "and manual boundary selection. The method assumes an approximately rectangular room with perpendicular walls. Results are intended for demonstration purposes and may differ from actual measurements.")
METHOD_TEXT = "Three-viewpoint stereo reconstruction (left end, middle and right end of the wall are registered into one frame; the length is the distance between the two bottom wall corners)"
QUALITY_NOTE = "The quality checks are provisional engineering checks, not validated accuracy guarantees. A measurement that raises no warning is still an approximate estimate."
INSUFFICIENT_CODE = "THREE_VIEW_INSUFFICIENT_EVIDENCE"
EXCLUDED_CODE = "THREE_VIEW_SETS_EXCLUDED"
# module ThreeViewError code -> the existing visitor failure code (class)
_STRUCTURAL = {"RESOLUTION_MISMATCH": ("CALIBRATION_VIDEO_SIZE_MISMATCH", "visitor_input"), "CALIBRATION_INVALID": ("CALIBRATION_INVALID", "visitor_input"),
               "ANNOTATION_MISSING": ("ANNOTATION_MISSING", "visitor_input"), "ANNOTATION_INVALID": ("ANNOTATION_INVALID", "visitor_input"), "ANNOTATION_SCHEMA": ("ANNOTATION_INVALID", "visitor_input"),
               "ANNOTATION_MALFORMED": ("ANNOTATION_INVALID", "visitor_input"), "ANNOTATION_OUTSIDE_FRAME": ("ANNOTATION_INVALID", "visitor_input"), "ANNOTATION_DUPLICATE": ("ANNOTATION_INVALID", "visitor_input"),
               "FRAME_STALE": ("FRAME_STALE", "visitor_input"), "FRAME_SIZE_MISMATCH": ("FRAME_SIZE_MISMATCH", "visitor_input"), "FRAME_MISSING": ("FRAMES_NOT_EXTRACTED", "visitor_input"),
               "VIDEO_MISSING": ("UPLOAD_EXPIRED", "visitor_input"), "INTERNAL_ERROR": ("INTERNAL_ERROR", "worker_resource")}


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _typed(f: dict) -> dict:
    """A worker failure -> the typed failure record of the shared envelope."""
    code, cls = _STRUCTURAL.get(str(f.get("code")), (str(f.get("code")), "method_j"))
    return dict(code=code, failure_class=cls, stage=f.get("stage", "worker"), message=str(f.get("message", ""))[:400], visitor_message=VM.failure_text(code))


def indicators_of(m: dict | None) -> dict | None:
    """The few numbers a visitor can read from one frame's metrics (largest residual, fewest inliers, ...)."""
    if not m:
        return None
    pose = [d for d in (m.get("pose") or {}).values() if d]; regs = [m["reg_v1_v2"], m["reg_v3_v2"]]; fc = m["floor_consistency"]; c = m["corner"]
    return dict(pose_scale_px_max=max((d["scale_px"] for d in pose), default=None), registration_inliers_min=min(r["n_inliers"] for r in regs), registration_rms_mm_max=max(r["rms_mm"] for r in regs),
                floor_angle_deg_max=max(fc["ang_v1_v2_deg"], fc["ang_v3_v2_deg"], fc["ang_v1_v3_deg"]), corner_direction_deg_max=max(c["vertical_edge_vs_floor_normal_deg"].values()),
                corner_off_floor_mm_max=max(abs(c[k]) for k in ("b1_off_v2_floor_mm", "b3_off_v2_floor_mm", "b1_off_v3_floor_mm", "b3_off_v1_floor_mm")))


def _finding(f: dict) -> dict:
    p = VM.warning_parts(f["code"]) if f.get("severity") == "warning" else VM.failure_parts(f["code"])
    return dict(code=f["code"], severity=f["severity"], scope=f.get("scope"), value=f.get("value"), limit=f.get("limit"), message=f.get("message"), action=f.get("action"),
                meaning=p.get("meaning") or p.get("what"), next_step=p.get("action") or p.get("next_step"))


def assemble(meta: dict, *, worker_out: dict | None = None, failures=None, process_info=None, seconds: float = 0.0) -> dict:
    """The result document.  ``meta``: dict(calibration (id, baseline_mm, image_size_wh), videos {slot: {sha256, frame_count}}, frame_sets {key: {slot: frame number}}, excluded_sets [keys],
    source_sets [keys]).  ``failures`` are failures of the calling layer (input checks, worker supervision); ``worker_out`` is the worker's document."""
    failures = [dict(f) for f in (failures or [])]
    res = None
    if worker_out is not None and not failures:
        failures += [_typed(f) for f in (worker_out.get("failures") or [])]
        res = worker_out.get("result") if not failures else None
    status, length, warnings, per_set, tv = FAILED, None, [], [], None
    meta = meta or {}
    if res is not None:
        q = res["quality"]; frames = []
        for r in res["per_frame"]:
            fq = r["quality"]; used = fq["status"] != "FAILED"; key = int(r["frame"])
            frames.append(dict(frame_set=key, frame_indices=dict((meta.get("frame_sets") or {}).get(str(key)) or (meta.get("frame_sets") or {}).get(key) or {}), status=fq["status"], used=used,
                               length_mm=r.get("length_mm") if used else None, hard_failures=[_finding(f) for f in fq["hard_failures"]], warnings=[_finding(f) for f in fq["warnings"]],
                               indicators=indicators_of(r.get("metrics"))))
        excluded = [int(k) for k in (meta.get("excluded_sets") or [])]
        ok_agg = q["status"] != INSUFFICIENT_EVIDENCE
        if not ok_agg:
            failures.append(dict(code=INSUFFICIENT_CODE, failure_class="method_j", stage="quality", message=str(q.get("message", "")), visitor_message=VM.failure_text(INSUFFICIENT_CODE)))
        else:
            length = q["length_mm"]; status = SUCCESS if q["status"] == MEASURED and not excluded else QUESTIONABLE
        seen = {}
        for f in q.get("frame_warnings", []) + q.get("aggregate_warnings", []):
            seen.setdefault(f["code"], dict(code=f["code"], advisory=True, frames=[], visitor_explanation=VM.warning_parts(f["code"])))["frames"].append(f.get("frame"))
        if q.get("frames_failed") and ok_agg:
            seen.setdefault("SOME_FRAME_SETS_FAILED", dict(code="SOME_FRAME_SETS_FAILED", advisory=True, frames=list(q["frames_failed"]), visitor_explanation=VM.warning_parts("SOME_FRAME_SETS_FAILED")))
        if excluded and ok_agg:
            seen[EXCLUDED_CODE] = dict(code=EXCLUDED_CODE, advisory=True, frames=excluded, visitor_explanation=VM.warning_parts(EXCLUDED_CODE))
        warnings = list(seen.values()) if ok_agg else []
        per_set = [dict(frame_set=f["frame_set"], status="reconstructed" if (f["used"] and ok_agg) else "failed", width_mm=f["length_mm"] if ok_agg else None, height_mm=None,
                        frame_indices=f["frame_indices"], failure_code=(f["hard_failures"][0]["code"] if f["hard_failures"] else None), quality_status=f["status"]) for f in frames]
        tv = dict(method=METHOD, n_viewpoints=3, matcher=worker_out.get("matcher"), status=q["status"], frames_total=q["frames_total"], frames_usable=q["frames_usable"], frames_failed=list(q["frames_failed"]),
                  frame_sets_excluded=excluded, frames=frames, aggregate_warnings=[_finding(f) for f in q.get("aggregate_warnings", [])], message=q.get("message"),
                  length_spread_mm=q.get("length_spread_mm") if ok_agg else None, baseline_mm=(meta.get("calibration") or {}).get("baseline_mm"), image_size_wh=(meta.get("calibration") or {}).get("image_size_wh"),
                  thresholds_note=res.get("thresholds_note") or QUALITY_NOTE)
        if not ok_agg:
            for f in tv["frames"]:
                f["length_mm"] = None                                            # no single-frame number is offered when the aggregate is refused
    ok = status != FAILED and not failures
    if not ok:
        status, length, warnings = FAILED, None, ([] if res is None else warnings)
        per_set = [dict(p, width_mm=None) for p in per_set]
    classes = sorted({f.get("failure_class") for f in failures if f.get("failure_class")})
    out = dict(
        schema=SCHEMA_RESULT, method=METHOD, n_viewpoints=3, status=status, width_mm=length if ok else None, height_mm=None, length_mm=length if ok else None, units="mm", partial=False,
        dimensions_accepted=ok, per_frame_set=per_set, failures=failures, failure_classes=classes, failure_class=classes[0] if len(classes) == 1 else ("mixed" if classes else None),
        warnings=warnings if ok else [], notes=[], diagnostics=dict(pose=None, registration=None, set_failures=[], n_sets=(tv or {}).get("frames_total"), n_sets_ok=(tv or {}).get("frames_usable")),
        processing=dict(seconds=round(float(seconds), 3), worker=dict(**(process_info or {})), stage_seconds=(worker_out or {}).get("timings"), worker_self_peak_rss_mb=(worker_out or {}).get("peak_rss_mb")),
        provenance=dict(calibration=meta.get("calibration"), frame_indices=meta.get("frame_sets"), n_frame_sets=len(meta.get("source_sets") or []), videos=meta.get("videos") or {}, annotations=[],
                        method=METHOD, matcher=(worker_out or {}).get("matcher")),
        three_view=tv if ok or tv is not None else None,
        uncertainty=dict(status="not_computed", note="no confidence interval or calibrated uncertainty is claimed; no percentage error is computed"), ground_truth_used=False,
        note="three-viewpoint reconstruction: an approximate estimate, not a validated accuracy guarantee" if ok else "no valid reconstruction; no length is reported",
        visitor_notice=VM.QUESTIONABLE_SUMMARY if status == QUESTIONABLE else (None if ok else "No length is reported."))
    return json.loads(json.dumps(out, allow_nan=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def failed(meta: dict | None, failures: list, *, process_info=None, seconds=0.0) -> dict:
    """A FAILED result for failures of the calling layer (no worker output)."""
    return assemble(meta or {}, failures=failures, process_info=process_info, seconds=seconds)


def is_three_view(result) -> bool:
    return isinstance(result, dict) and result.get("method") == METHOD


# ------------------------------------------------------------------ view model
def _dim(mm):
    return None if not (_finite(mm) and mm > 0) else dict(mm=float(mm), cm=mm / 10.0, m=mm / 1000.0, cm_text=f"{mm / 10.0:.0f} cm", m_text=f"{mm / 1000.0:.2f} m", cm_text_precise=f"{mm / 10.0:.2f} cm")


def view_of(result) -> dict | None:
    """What the page draws for a three-viewpoint result (also for a FAILED one: its frame-level failures are shown, never hidden)."""
    if not is_three_view(result):
        return None
    tv = result.get("three_view") or {}; ok = bool(result.get("dimensions_accepted")) and result.get("status") in (SUCCESS, QUESTIONABLE)
    n_warn = len(result.get("warnings") or []); approx = ok
    frames = []
    for f in tv.get("frames", []):
        frames.append(dict(frame_set=int(f["frame_set"]) + 1, status=f["status"], used=f["used"], length=_dim(f.get("length_mm")) if ok else None, warnings=list(f.get("warnings") or []),
                           hard_failures=list(f.get("hard_failures") or []), indicators=f.get("indicators"), frame_indices=f.get("frame_indices") or {}))
    ind = [f["indicators"] for f in frames if f["used"] and f["indicators"]]
    rows = []
    if ok and ind:
        rows = [("Camera-pair fit (largest residual)", f"{max(i['pose_scale_px_max'] for i in ind):.2f} px"), ("Viewpoint joins (fewest shared points)", f"{min(i['registration_inliers_min'] for i in ind)}"),
                ("Viewpoint joins (largest RMS)", f"{max(i['registration_rms_mm_max'] for i in ind):.1f} mm"), ("Floor agreement between viewpoints (largest)", f"{max(i['floor_angle_deg_max'] for i in ind):.2f}°"),
                ("Corner direction error (largest)", f"{max(i['corner_direction_deg_max'] for i in ind):.2f}°")]
        sp = tv.get("length_spread_mm")
        if sp:
            rows.append(("Frame-to-frame spread", f"{sp['range']:.0f} mm"))
    headline = ("Approximate length" + (f" — {n_warn} quality warning" + ("s" if n_warn != 1 else "") if n_warn else "")) if ok else "No length could be computed"
    return dict(ok=ok, approximate=approx, headline=headline, length=_dim(result.get("length_mm")) if ok else None, n_viewpoints=3, method=METHOD_TEXT, matcher=tv.get("matcher"),
                n_frame_sets=tv.get("frames_total"), frames_usable=tv.get("frames_usable"), frames_failed=[i + 1 for i in tv.get("frames_failed", [])], excluded=[i + 1 for i in tv.get("frame_sets_excluded", [])],
                frames=frames, aggregate_warnings=list(tv.get("aggregate_warnings") or []), indicators=rows, baseline_mm=tv.get("baseline_mm"), image_size_wh=tv.get("image_size_wh"),
                quality_note=tv.get("thresholds_note") or QUALITY_NOTE, limitations=LIMITATIONS, message=tv.get("message"))

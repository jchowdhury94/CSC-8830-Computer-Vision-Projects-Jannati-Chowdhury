"""F8.31C-2A: the visitor-facing VIEW MODEL of one ``f8.visitor_reconstruction/1`` result (pure Python: no Streamlit, no OpenCV, no backend import).

It only PRESENTS a result the backend already decided.  It never changes a number, a status or a success / failure decision:

* millimetres -> centimetres / metres is one division on the unrounded backend value; rounding happens only in the display strings (whole centimetres, centimetre-resolution
  metres: the method's own accuracy is not claimed to be better);
* a FAILED result (or one whose ``dimensions_accepted`` is not true) never yields a dimension, whatever numbers the dict carries;
* wording comes from ``visitor_messages`` (failure codes and the seven advisory flags); nothing here invents a threshold;
* no confidence interval, no guaranteed accuracy, no percentage error (no ground truth is supplied in this phase).
"""
from __future__ import annotations

import json
import math

import f8_ui_text as UT
import three_view_result as TVR                    # pure Python (no backend import): the three-viewpoint Wall B result and its view model
import visitor_messages as VM

SCHEMA_RESULT = "f8.visitor_reconstruction/1"           # the backend's schema string (duplicated on purpose: this module must not import the backend)
SUMMARY_SCHEMA = "f8.visitor_result_summary/1"
SUCCESS, QUESTIONABLE, FAILED = "SUCCESS", "QUESTIONABLE", "FAILED"

# display states (the interface vocabulary of section 6 of the task)
WAITING, VALIDATING, PROCESSING = "waiting", "validating", "processing"
DONE_OK, DONE_WARN, DONE_FAILED = "success", "success_with_warnings", "failed"
BUSY, NO_MEMORY, TIMEOUT, INPUTS_CHANGED = "busy", "insufficient_memory", "timeout", "inputs_changed"
STATES = (WAITING, VALIDATING, PROCESSING, DONE_OK, DONE_WARN, DONE_FAILED, BUSY, NO_MEMORY, TIMEOUT, INPUTS_CHANGED)

STATE_TEXT = {      # state -> (label, tone, one sentence)
    WAITING: ("Ready to start", "neutral", "Nothing has been started yet."),
    VALIDATING: ("Checking your inputs", "info", "Your inputs are being checked. No dimensions exist yet."),
    PROCESSING: ("Measuring", "info", "The measurement is running on the server. This can take a while; no percentage is shown because the backend reports none."),
    DONE_OK: ("Measurement complete", "good", "A measurement was computed and no advisory flags were raised."),
    DONE_WARN: ("Measurement complete, with advisory warnings", "warn", "A measurement was computed, but the checks raised advisory warnings. Read them before relying on the numbers."),
    DONE_FAILED: ("The measurement could not be completed", "bad", "No dimension is reported."),
    BUSY: ("The server is busy", "warn", "Another measurement is running. Nothing was concluded about your room."),
    NO_MEMORY: ("Not enough memory", "warn", "The server could not provide enough memory. Nothing was concluded about your room."),
    TIMEOUT: ("The measurement took too long", "warn", "It was stopped. Nothing was concluded about your room."),
    INPUTS_CHANGED: ("Your inputs expired or were replaced", "warn", "A video, the calibration, a frame or a boundary is no longer what was chosen. Nothing was concluded about your room."),
}
GATE_MESSAGE = ("Room measurement is being validated and is not yet available. Your calibration, frame selections, and annotations can still be reviewed.")
VARIABILITY_TITLE = "Measurement Variability"
VARIABILITY_TEXT = ("Wall measurements are estimated using manually selected wall and ceiling boundaries. Small differences in point placement can affect the calculated width and height. "
                    "Therefore, repeating the measurement with the same videos may produce slightly different results.")
RESULT_TITLE = "Measurement Complete"
ESTIMATE_NOTE = "Measurements are estimates. Different manual selections of wall and ceiling boundaries may produce slightly different results."
# One entry per reported measurement: (key, visitor label, result field).  The results page draws whatever this yields, so a further measurement is one more entry here.
MEASUREMENTS = (("wall_width", "Estimated Wall Width", "width_mm"), ("wall_height", "Estimated Wall Height", "height_mm"))
DISCLAIMER = "Estimates only: no confidence interval and no guaranteed accuracy is claimed, and no percentage error is computed."

_INPUTS_CHANGED_CODES = {"UPLOAD_EXPIRED", "VIDEO_REPLACED", "VIDEO_HASH_MISMATCH", "CALIBRATION_CHANGED", "FRAME_STALE", "ANNOTATION_STALE", "OWNERSHIP_MISMATCH"}
_MEMORY_CODES = {"INSUFFICIENT_MEMORY", "WORKER_MEMORY_LIMIT", "WORKER_MEMORY_UNMEASURABLE"}
_CALIBRATION_CODES = {"CALIBRATION_NOT_ACCEPTED", "CALIBRATION_INVALID", "CALIBRATION_INCONSISTENT", "BASELINE_INVALID", "OWNERSHIP_MISMATCH", "INPUT_INVALID_CALIBRATION",
                      "INPUT_IMAGE_SIZE_MISMATCH", "BASELINE_CHANGED_UNLABELLED", "LINE_RETARGET_FAILED"}
_VIDEO_CODES = {"UPLOAD_EXPIRED", "VIDEO_REPLACED", "VIDEO_HASH_MISMATCH", "CALIBRATION_VIDEO_SIZE_MISMATCH", "JOB_TOO_LARGE"}


def step_for_code(code) -> str | None:
    """The workflow step ('home' = calibration, 'room' = videos, 'frames' = frames and boundaries) where the visitor can fix the cause; None when nothing the visitor changes would help."""
    code = str(code)
    if code in _CALIBRATION_CODES:
        return "home"
    if code in _VIDEO_CODES:
        return "room"
    if code.startswith(("FRAME", "ANNOTATION", "INPUT_", "CALIBRATION_")) or code in ("REQUEST_INVALID", "FRAMES_NOT_EXTRACTED"):
        return "frames"
    return None


STEP_LABEL = {"home": "Go to the calibration", "room": "Go to the room videos", "frames": "Go to frame selection and boundaries"}


# ------------------------------------------------------------------ units (one division, no intermediate rounding)
def mm_to_cm(mm):
    return None if mm is None else float(mm) / 10.0


def mm_to_m(mm):
    return None if mm is None else float(mm) / 1000.0


def fmt_cm(mm) -> str:
    return "not available" if mm is None else f"{mm_to_cm(mm):.0f} cm"


def fmt_m(mm) -> str:
    return "not available" if mm is None else f"{mm_to_m(mm):.2f} m"


def _positive_finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def dimension(mm) -> dict | None:
    """The exact backend value (mm) with its two conversions and their display strings; None when there is no valid number."""
    if not _positive_finite(mm):
        return None
    return dict(mm=float(mm), cm=mm_to_cm(mm), m=mm_to_m(mm), cm_text=fmt_cm(mm), m_text=fmt_m(mm), cm_text_precise=f"{mm_to_cm(mm):.2f} cm")


# ------------------------------------------------------------------ diagnostics and provenance tables (read-only copies)
def _scalar_text(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def scalar_rows(d, prefix: str = "", *, max_rows: int = 40) -> list:
    """Flatten one nested diagnostics dict into [(label, text)] rows of scalars (lists longer than six items and anything deeper than two levels are skipped)."""
    rows = []

    def walk(o, key, depth):
        if len(rows) >= max_rows:
            return
        if isinstance(o, dict) and depth < 2:
            for k, v in o.items():
                walk(v, f"{key} · {k}" if key else str(k), depth + 1)
        elif isinstance(o, (list, tuple)) and 0 < len(o) <= 6 and all(isinstance(x, (int, float, str, bool)) for x in o):
            rows.append((key.replace("_", " "), ", ".join(_scalar_text(x) for x in o)))
        elif isinstance(o, (int, float, str, bool)):
            rows.append((key.replace("_", " "), _scalar_text(o)))
    walk(d or {}, prefix, 0)
    return rows


RV_LABEL = {"p1_left": "Viewpoint 1 · Left camera video", "p1_right": "Viewpoint 1 · Right camera video", "p2_left": "Viewpoint 2 · Left camera video", "p2_right": "Viewpoint 2 · Right camera video"}


def technical_rows(prov: dict | None) -> list:
    """[(label, text)] for developers only: the raw checksums of the videos the result was computed from."""
    return [(f"Video {slot} · SHA-256", str(v.get("sha256", ""))) for slot, v in sorted(((prov or {}).get("videos") or {}).items())]


def provenance_rows(prov: dict | None, baseline_source=None) -> list:
    """[(label, text)] of what this result was computed from.  Owner digest, paths, pixels and raw hashes are never part of it (hashes: ``technical_rows``).
    ``baseline_source`` is the ACCEPTED calibration's real provenance (measured / calibrated / file / entered); the label never claims a measurement that did not happen (F8.31D-2, M7)."""
    p = prov or {}
    c = p.get("calibration") or {}
    rows = []
    if c:
        rows += [("Calibration id", str(c.get("calibration_id"))), ("Baseline", f"{c.get('baseline_mm'):g} mm ({UT.baseline_phrase(baseline_source)})" if _positive_finite(c.get("baseline_mm")) else "—"),
                 ("Image size", " × ".join(str(v) for v in c.get("image_size_wh", []))), ("Camera order", ", ".join(str(v) for v in c.get("camera_order", []))),
                 ("Distortion model", str(c.get("distortion_model")))]
    for slot, v in sorted((p.get("videos") or {}).items()):
        rows.append((RV_LABEL.get(slot, f"Video {slot}"), f"{v.get('frame_count')} frames"))
    if p.get("n_frame_sets") is not None:
        rows.append(("Frame sets", str(p["n_frame_sets"])))
    if p.get("annotations") is not None:
        rows.append(("Accepted boundaries", str(len(p["annotations"]))))
        manual = {}
        for a in p["annotations"]:
            if isinstance(a, dict) and "MANUAL_BOUNDARY_FALLBACK" in (a.get("accepted_warning_codes") or []):
                manual.setdefault(str(a.get("line")), []).append(int(a.get("frame_set", 0)) + 1)
        if manual:                                                       # Wall B only: boundaries whose automatic refinement failed and whose unrefined two-point line was used
            rows.append(("Boundaries from the visitor's manual line (automatic refinement failed)", "; ".join(f"{k} (frame sets {', '.join(map(str, v))})" for k, v in sorted(manual.items()))))
    rows += [("Method", str(p.get("method", "—"))), ("Original recipe values used", "no" if not p.get("recipe_a_used") else "yes"),
             ("Historical baseline used", "no" if not p.get("historical_baseline_used") else "yes"), ("Ground truth used", "no" if not p.get("ground_truth_used") else "yes")]
    return rows


# ------------------------------------------------------------------ state of a result
def _valid_schema(r) -> bool:
    return isinstance(r, dict) and r.get("schema") == SCHEMA_RESULT and r.get("status") in (SUCCESS, QUESTIONABLE, FAILED)


def _first_failure_code(r) -> str | None:
    for f in (r.get("failures") or []):
        if isinstance(f, dict) and f.get("code"):
            return str(f["code"])
    return None


def result_state(result) -> str:
    """The display state of a finished result (never a success state for anything that is not a valid, accepted measurement)."""
    if not _valid_schema(result):
        return DONE_FAILED
    if result["status"] == FAILED or not result.get("dimensions_accepted") or result.get("failures"):
        code = _first_failure_code(result)
        if code in _MEMORY_CODES:
            return NO_MEMORY
        if code == "WORKER_BUSY":
            return BUSY
        if code == "WORKER_TIMEOUT":
            return TIMEOUT
        if code in _INPUTS_CHANGED_CODES:
            return INPUTS_CHANGED
        return DONE_FAILED
    if dimension(result.get("width_mm")) is None and dimension(result.get("height_mm")) is None:
        return DONE_FAILED                                   # a "success" without any valid number is not a measurement
    if result["status"] == QUESTIONABLE or result.get("warnings"):
        return DONE_WARN
    return DONE_OK


def warning_cards(result) -> list:
    """One card per advisory warning that is PRESENT in the result (explanation, reliability implication, suggested action), in the backend's order, from ``visitor_messages``."""
    cards = []
    for w in (result.get("warnings") or []) if isinstance(result, dict) else []:
        code = w.get("code") if isinstance(w, dict) else str(w)
        p = VM.warning_parts(code)
        cards.append(dict(code=p["code"], known=p["known"], meaning=p["meaning"], reliability=p["reliability"], action=p["action"], visitor_can_act=p["visitor_can_act"]))
    return cards


def failure_card(result) -> dict | None:
    """The visitor-friendly failure explanation (separate from the technical details)."""
    result = result if isinstance(result, dict) else {}
    fl = [f for f in (result.get("failures") or []) if isinstance(f, dict)]
    if not fl:
        fl = [dict(code="INTERNAL_ERROR")]               # a result that is neither an accepted measurement nor carries a typed failure is an internal error, never a success
    p = VM.failure_parts(fl[0].get("code"))
    code = p["code"]
    cls = fl[0].get("failure_class")
    step = step_for_code(code)
    steps = UT.recovery_steps(code, step)                  # F8.31D-2: typed routing; codes whose advice means "record again" lead to the room videos
    if steps:
        step = steps[0]
    if step is not None:
        retry = dict(kind="fix", step=step, label=STEP_LABEL[step])
    elif code == "WORKER_CANCELLED" or cls == "worker_resource":
        retry = dict(kind="again", step=None, label="Try again")
    else:
        retry = dict(kind="fix", step="frames", label=STEP_LABEL["frames"])          # a Method J rejection: different frames are the practical correction
    actions = [dict(kind="fix", step=x, label=STEP_LABEL[x]) for x in steps] if steps else ([retry] if retry["kind"] == "fix" else [])
    import quality_feedback as QF
    return dict(code=code, what=p["what"], next_step=p["next_step"], retry=retry, actions=actions, failure_class=cls, feedback=QF.failure_item(code).to_dict(),
                technical=[dict(code=str(f.get("code")), failure_class=f.get("failure_class"), stage=f.get("stage"), message=str(f.get("message", ""))[:400]) for f in fl][:8])


def build_view(result, baseline_source=None) -> dict:
    """Everything the results page draws, from ONE result.  Dimensions only for an accepted SUCCESS / QUESTIONABLE result."""
    state = result_state(result)
    label, tone, sentence = STATE_TEXT[state]
    ok = state in (DONE_OK, DONE_WARN)
    r = result if isinstance(result, dict) else {}
    tv = TVR.view_of(result) if _valid_schema(result) else None                       # three-viewpoint Wall B only; None for every two-viewpoint result
    measurements = ((("room_length", "Estimated Room Length", "width_mm"),) if tv is not None else MEASUREMENTS)
    width = dimension(r.get("width_mm")) if ok else None
    height = dimension(r.get("height_mm")) if ok else None
    per_set = []
    if ok:
        for s in r.get("per_frame_set") or []:
            if not isinstance(s, dict):
                continue
            rec = s.get("registration")
            done = s.get("status") == "reconstructed"
            row = dict(frame_set=int(s.get("frame_set", len(per_set))) + 1, status=s.get("status"), width=dimension(s.get("width_mm")) if done else None,
                       height=dimension(s.get("height_mm")) if done else None,
                       dims={k: (dimension(s.get(f)) if done else None) for k, _, f in measurements}, frame_indices=dict(s.get("frame_indices") or {}),
                       registration=scalar_rows(rec) if isinstance(rec, dict) else [], vertical_error_deg=s.get("vertical_error_deg"), skew_mm=s.get("skew_mm"))
            if s.get("status") == "failed":
                row["failure"] = VM.failure_parts(s.get("failure_code"))["what"]
            per_set.append(row)
    diag = r.get("diagnostics") or {}
    proc = r.get("processing") or {}
    dims = [dict(key=k, label=lab, value=dimension(r.get(f))) for k, lab, f in measurements] if ok else []
    return dict(
        dimensions=dims, title=RESULT_TITLE, estimate_note=ESTIMATE_NOTE,
        state=state, label=label, tone=tone, sentence=sentence, accepted=ok, status=r.get("status") if _valid_schema(result) else FAILED,
        width=width, height=height, partial=bool(ok and r.get("partial")), per_set=per_set, warnings=warning_cards(r) if ok else [],
        failure=None if ok else failure_card(result if _valid_schema(result) else dict(status=FAILED, failures=[dict(code="INTERNAL_ERROR")])),
        notice=VM.QUESTIONABLE_SUMMARY if ok and r.get("status") == QUESTIONABLE else None, seconds=proc.get("seconds") if isinstance(proc.get("seconds"), (int, float)) and not isinstance(proc.get("seconds"), bool) and math.isfinite(proc.get("seconds")) else None,
        pose=scalar_rows(diag.get("pose")) if ok else [], registration=scalar_rows(diag.get("registration")) if ok else [],
        provenance=provenance_rows(r.get("provenance"), baseline_source) if _valid_schema(result) else [],
        technical=technical_rows(r.get("provenance")) if _valid_schema(result) else [], disclaimer=DISCLAIMER,
        variability=dict(title=VARIABILITY_TITLE, text=VARIABILITY_TEXT) if ok else None, boundaries=boundary_rows(r.get("provenance")) if ok else [], three_view=tv)


def boundary_rows(prov: dict | None) -> list:
    """The visitor's confirmed boundary points as stored in the result's provenance (so repeated runs can be compared): [dict(frame_set, line, label, points)]. Nothing is invented."""
    out = []
    for a in (prov or {}).get("annotations") or []:
        try:
            out.append(dict(frame_set=int(a["frame_set"]) + 1, line=str(a["line"]), label=UT.boundary_label(str(a["line"])),
                            points=[[float(x), float(y)] for x, y in a["points_distorted_full_px"]]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def summary_document(result, baseline_source=None) -> dict | None:
    """A small JSON document the visitor may download.  Numbers, status, warnings and provenance only: no video, frame, path or owner identifier; a FAILED result carries no dimension."""
    if not _valid_schema(result):
        return None
    v = build_view(result)
    prov = json.loads(json.dumps(result.get("provenance") or {}))
    prov.pop("owner_digest", None)
    if baseline_source and isinstance(prov.get("calibration"), dict):
        prov["calibration"]["baseline_source"] = baseline_source        # the real provenance of the accepted calibration, not the backend's fixed label
    doc = dict(schema=SUMMARY_SCHEMA, status=v["status"], state=v["state"], dimensions_accepted=bool(v["accepted"]), processing_seconds=v["seconds"],
               width=None if v["width"] is None else {k: v["width"][k] for k in ("mm", "cm", "m")}, height=None if v["height"] is None else {k: v["height"][k] for k in ("mm", "cm", "m")},
               warnings=[w["code"] for w in v["warnings"]], failure=None if v["failure"] is None else {k: v["failure"][k] for k in ("code", "what", "next_step")},
               per_frame_set=[dict(frame_set=s["frame_set"], status=s["status"], width_mm=None if s["width"] is None else s["width"]["mm"], height_mm=None if s["height"] is None else s["height"]["mm"])
                              for s in v["per_set"]],
               provenance=dict(calibration=prov.get("calibration"), frame_indices=prov.get("frame_indices"), confirmed_boundaries=v["boundaries"], n_frame_sets=prov.get("n_frame_sets"),
                               videos={s: dict(sha256=x.get("sha256"), frame_count=x.get("frame_count")) for s, x in (prov.get("videos") or {}).items()},
                               ground_truth_used=False, recipe_a_used=False, historical_baseline_used=False),
               uncertainty="none claimed", note=DISCLAIMER)
    if v.get("three_view") is not None:                       # three-viewpoint Wall B: the method, the quality classification and the limitations travel with the number
        t = v["three_view"]
        doc["method"] = TVR.METHOD; doc["n_viewpoints"] = 3; doc["approximate"] = bool(t["approximate"]); doc["limitations"] = TVR.LIMITATIONS
        doc["three_view"] = dict(frames_usable=t["frames_usable"], n_frame_sets=t["n_frame_sets"], frames_failed=t["frames_failed"], frame_sets_excluded=t["excluded"], quality_note=t["quality_note"],
                                 per_frame=[dict(frame_set=f["frame_set"], status=f["status"], length_mm=None if f["length"] is None else f["length"]["mm"], warnings=[w["code"] for w in f["warnings"]],
                                                 hard_failures=[w["code"] for w in f["hard_failures"]]) for f in t["frames"]],
                                 aggregate_warnings=[w["code"] for w in t["aggregate_warnings"]], length_spread_mm=((result.get("three_view") or {}).get("length_spread_mm")))
        doc["note"] = TVR.LIMITATIONS
    return json.loads(json.dumps(doc, allow_nan=False))


# ------------------------------------------------------------------ the whole room: three dimensions from two independent wall measurements
ROOM_SCHEMA = "f8.visitor_room_summary/1"
ROOM_TITLE = "Room Measurement Complete"
LIMITATIONS_TITLE = "Measurement Limitations"
LIMITATIONS_TEXT = ("Room dimensions are approximate estimates obtained from stereo videos of two walls. Accuracy may be affected by camera calibration, camera positioning, video quality, "
                    "partially hidden corners, and manual boundary selection. The method assumes an approximately rectangular room with perpendicular walls. Results are intended for "
                    "demonstration purposes and may differ from actual measurements.")
# (key, visitor label, wall that supplies it, result field of that wall, source description).  Each dimension comes from exactly ONE wall's own result; nothing is averaged or combined.
ROOM_MEASUREMENTS = (("room_width", "Room Width", "A", "width_mm", "Wall A · horizontal width"),
                     ("room_length", "Room Length", "B", "width_mm", "Wall B · horizontal width"),
                     ("room_height", "Room Height", "A", "height_mm", "Wall A · height"))


def room_view(result_a, result_b, baseline_source=None) -> dict:
    """The final room page from the two walls' results.  ``complete`` only when BOTH are accepted measurements and all three values exist; otherwise no dimension is returned."""
    va, vb = build_view(result_a, baseline_source), build_view(result_b, baseline_source)
    src = {"A": result_a if isinstance(result_a, dict) else {}, "B": result_b if isinstance(result_b, dict) else {}}
    ok = bool(va["accepted"] and vb["accepted"])
    b3 = vb.get("three_view")                                    # Wall B measured with three viewpoints: its length is still Wall B's ``width_mm``; only the description and the statement change
    dims = [dict(key=k, label=lab, wall=w, source=("Wall B · three-viewpoint length" if (b3 is not None and w == "B") else desc), value=dimension(src[w].get(f)) if ok else None)
            for k, lab, w, f, desc in ROOM_MEASUREMENTS]
    complete = ok and all(d["value"] is not None for d in dims)
    return dict(complete=complete, title=ROOM_TITLE, dimensions=dims if complete else [], wall_a=va, wall_b=vb, limitations_title=LIMITATIONS_TITLE, limitations=TVR.LIMITATIONS if b3 is not None else LIMITATIONS_TEXT,
                wall_b_three_view=b3)


def room_summary_document(result_a, result_b, baseline_source=None) -> dict | None:
    """The combined JSON the visitor may download: the three room dimensions (with the wall each one came from) plus each wall's own summary document.  No video, frame or path."""
    rv = room_view(result_a, result_b, baseline_source)
    if not rv["complete"]:
        return None
    doc = dict(schema=ROOM_SCHEMA,
               dimensions={d["key"]: dict({k: d["value"][k] for k in ("mm", "cm", "m")}, source=d["source"]) for d in rv["dimensions"]},
               walls=dict(wall_a=summary_document(result_a, baseline_source), wall_b=summary_document(result_b, baseline_source)),
               note=rv["limitations"])
    return json.loads(json.dumps(doc, allow_nan=False))

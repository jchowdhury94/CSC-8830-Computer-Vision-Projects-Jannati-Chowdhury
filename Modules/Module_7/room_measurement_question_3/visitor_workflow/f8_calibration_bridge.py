"""Stage F8.15: hand-off of an F8 Cloud calibration RESULT to the F6 room-measurement workflow.  Presentation-independent (NO Streamlit import), standard library only.

What it does
    F8 writes one JSON document per calibration job (``f8.calibration_result/1`` or ``/2``, see ``f8_chunk_poc/result_store.build_result_document``).
    This module validates that document, applies the visitor's explicit-acceptance rules and returns the Stage A input contract
    (K1, D1, K2, D2, image_size_wh, capture_mode, baseline_mm) that ``F6Session.import_calibration`` consumes.

What it does NOT do
    * No calibration or geometry mathematics (the contract was produced by the unchanged F7 ``contract_from_model``); nothing is recomputed or re-estimated.
    * It never invents values: a missing or malformed field is an error, never a default.  Capture-mode provenance is carried over, never guessed.
    * It does not accept a bare contract / Phase 1 file (F6's own importer, ``f6_core.parse_calibration_json``, already handles those).

Baseline rule (same semantics as F7 ``contract_from_model`` / ``assess``)
    The visitor-MEASURED baseline wins when one is supplied here, or was entered for the F8 job; otherwise the calibrated |T| stored in the contract is used.
    The other value is kept for display; a >5 % disagreement is reported as a concern.  F6 scales by the baseline it is given (the model's own T is not used).

Acceptance rule (same as ``CalibrationWizard.accept``)
    GOOD -> may be accepted.  QUESTIONABLE -> only with ``acknowledged=True``.  FAILED, an unfinished job, a benchmark document, or anything malformed -> never.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from numbers import Real
from typing import Mapping

SCHEMA = "f8.calibration_result/1"                 # kept: the original schema constant (callers compare against it)
SCHEMA_V2 = "f8.calibration_result/2"              # F8.29: adds ``calibration_observations``
SUPPORTED_SCHEMAS = (SCHEMA, SCHEMA_V2)


def _observation_module():
    """F8's standard-library-only observation validator, loaded by path (no sys.path change, no Streamlit)."""
    p = Path(__file__).resolve().parent.parent / "f8_chunk_poc" / "calibration_observations.py"
    spec = importlib.util.spec_from_file_location("f8_calibration_observations", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
GOOD, QUESTIONABLE, FAILED = "GOOD", "QUESTIONABLE", "FAILED"
BASELINE_DISAGREE_WARN = 0.05                      # same threshold as f7_calibration.BASELINE_DISAGREE_WARN
BASELINE_RANGE_MM = (10.0, 2000.0)                 # same range the F7 wizard accepts for a measured baseline
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024               # F8's own result cap (ResultStore max_bytes)
CONTRACT_FIELDS = ("K1", "D1", "K2", "D2", "image_size_wh", "capture_mode", "baseline_mm")


class HandoffError(ValueError):
    """The F8 result cannot be handed to F6.  ``problems`` is a list of dict(code, message); ``str()`` joins the messages."""

    def __init__(self, problems):
        self.problems = [dict(code=c, message=m) for c, m in problems]
        super().__init__("; ".join(p["message"] for p in self.problems))

    @property
    def codes(self) -> list:
        return [p["code"] for p in self.problems]


def _num(v) -> bool:
    return isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v)


def _matrix(v, shape) -> bool:
    try:
        if len(shape) == 2:
            return len(v) == shape[0] and all(len(r) == shape[1] and all(_num(x) for x in r) for r in v)
        return len(v) == shape[0] and all(_num(x) for x in v)
    except TypeError:
        return False


@dataclass(frozen=True)
class Handoff:
    """A validated F8 result, ready for review.  ``accept`` is the only way to obtain the contract for F6."""
    level: str
    headline: str | None
    concerns: tuple
    stereo_rms_px: float | None
    calibrated_baseline_mm: float | None
    baseline_used_mm: float
    baseline_source: str                            # "measured" | "calibrated"
    image_size_wh: tuple
    job_id: str | None
    _contract: dict = field(repr=False)
    observations_available: bool = False            # F8.29: True only for a /2 result whose retained corner observations passed validation
    uncertainty_note: str = "calibration uncertainty unavailable"
    _observations: dict | None = field(default=None, repr=False)

    def observations(self) -> dict | None:
        """The validated corner observations (a copy), or None. Their presence is NOT an uncertainty estimate."""
        return None if self._observations is None else json.loads(json.dumps(self._observations))

    @property
    def requires_acknowledgement(self) -> bool:
        return self.level == QUESTIONABLE

    def can_accept(self, acknowledged: bool = False) -> bool:
        return self.level == GOOD or (self.level == QUESTIONABLE and bool(acknowledged))

    def accept(self, acknowledged: bool = False) -> dict:
        """The visitor's explicit acceptance.  Returns a fresh copy of the Stage A contract for ``F6Session.import_calibration``."""
        if self.level == FAILED:
            raise HandoffError([("FAILED_CALIBRATION", "A failed calibration cannot be used.")])
        if self.level == QUESTIONABLE and not acknowledged:
            raise HandoffError([("ACKNOWLEDGEMENT_REQUIRED", "Acknowledge the calibration's concerns before using it.")])
        if self.level not in (GOOD, QUESTIONABLE):
            raise HandoffError([("UNKNOWN_QUALITY", "The calibration has no recognised quality level.")])
        return json.loads(json.dumps(self._contract))

    def provenance(self) -> dict:
        """Small record for ``F6Session.calibration_provenance`` (numbers and labels only)."""
        return dict(source="f8_cloud_calibration", job_id=self.job_id, level=self.level, acknowledged_concerns=list(self.concerns) if self.level == QUESTIONABLE else [],
                    stereo_rms_px=self.stereo_rms_px, calibrated_baseline_mm=self.calibrated_baseline_mm, baseline_used_mm=self.baseline_used_mm, baseline_source=self.baseline_source)


def _load(doc, problems):
    if isinstance(doc, (bytes, bytearray, str)):
        if len(doc) > MAX_DOCUMENT_BYTES:
            problems.append(("DOCUMENT_TOO_LARGE", "The calibration file is too large to be an F8 result."))
            return None
        try:
            doc = json.loads(doc)
        except (ValueError, RecursionError):
            problems.append(("NOT_JSON", "The file is not valid JSON."))
            return None
    if not isinstance(doc, Mapping):
        problems.append(("NOT_AN_OBJECT", "The calibration file must be a JSON object."))
        return None
    return doc


def parse_f8_result(doc, *, measured_baseline_mm=None, expected_image_size_wh=None) -> Handoff:
    """Validate an F8 result document (dict, JSON text or bytes) and return a ``Handoff``.  Raises ``HandoffError`` listing every problem found.

    ``measured_baseline_mm``: the visitor's measured lens distance; overrides the baseline stored in the contract.
    ``expected_image_size_wh``: [w, h] of the room videos, when known; a different calibration size is an error (F6 requires COMPATIBLE_METADATA)."""
    problems: list = []
    d = _load(doc, problems)
    if d is None:
        raise HandoffError(problems)
    if d.get("schema") not in SUPPORTED_SCHEMAS:
        raise HandoffError([("UNRECOGNISED_SCHEMA", f"This is not an F8 calibration result (expected schema '{SCHEMA}' or '{SCHEMA_V2}'). A bare calibration file can be imported with the existing import option instead.")])
    if d.get("mode") != "calibrate":
        problems.append(("NOT_A_CALIBRATION", "This F8 result is not a full calibration (it is a benchmark or another kind of job)."))
    if d.get("outcome") != "completed" or d.get("ok") is not True:
        problems.append(("JOB_NOT_COMPLETED", f"The F8 job did not complete successfully (outcome: {d.get('outcome')})."))
    q = d.get("quality") if isinstance(d.get("quality"), Mapping) else {}
    level = q.get("level")
    if level == FAILED:
        problems.append(("FAILED_CALIBRATION", "The calibration was assessed as FAILED and cannot be used."))
    elif level not in (GOOD, QUESTIONABLE):
        problems.append(("UNKNOWN_QUALITY", "The result has no recognised quality assessment (GOOD or QUESTIONABLE)."))
    if q.get("problems"):
        problems.append(("ASSESSMENT_PROBLEMS", "The calibration's own validity checks reported problems: " + "; ".join(str(p) for p in q["problems"][:3])))

    c = d.get("contract")
    if not isinstance(c, Mapping):
        problems.append(("CONTRACT_MISSING", "The result has no calibration contract (a failed or unfinished job does not produce one)."))
        raise HandoffError(problems)
    missing = [k for k in CONTRACT_FIELDS if c.get(k) is None]
    if missing:
        problems.append(("CONTRACT_MISSING_FIELD", f"The calibration contract is missing: {', '.join(missing)}."))
        raise HandoffError(problems)
    for k in ("K1", "K2"):
        if not _matrix(c[k], (3, 3)):
            problems.append(("INVALID_MATRIX", f"{k} must be a finite 3x3 matrix."))
    for k in ("D1", "D2"):
        if not _matrix(c[k], (5,)):
            problems.append(("INVALID_DISTORTION", f"{k} must be five finite numbers."))
    sz = c["image_size_wh"]
    if not (_matrix(sz, (2,)) and all(float(v).is_integer() and v > 0 for v in sz)):
        problems.append(("INVALID_IMAGE_SIZE", "image_size_wh must be two positive whole numbers [width, height]."))
        sz = None
    cm = c["capture_mode"]
    if not (isinstance(cm, Mapping) and all(isinstance(cm.get(s), str) and cm[s].strip() for s in ("left", "right"))):
        problems.append(("CAPTURE_MODE_PROVENANCE_REQUIRED", "capture_mode must declare both cameras (for example 'video'); it is never guessed."))
    if not _num(c["baseline_mm"]) or c["baseline_mm"] <= 0:
        problems.append(("INVALID_BASELINE", "The baseline stored in the result must be a positive number of millimetres."))

    cams = d.get("cameras") if isinstance(d.get("cameras"), Mapping) else {}                  # the result's own copies must agree with the contract
    for side, kk in (("left", "K1"), ("right", "K2")):
        cam = cams.get(side) if isinstance(cams.get(side), Mapping) else None
        if cam is not None and cam.get("K") is not None and _matrix(c[kk], (3, 3)) and _matrix(cam["K"], (3, 3)) and any(
                abs(a - b) > 1e-9 * max(1.0, abs(a)) for ra, rb in zip(c[kk], cam["K"]) for a, b in zip(ra, rb)):
            problems.append(("INCONSISTENT_RESULT", f"The {side} camera matrix differs between the contract and the result's camera section; the file may have been edited."))
    if sz is not None:
        for side in ("left", "right"):
            cs = (cams.get(side) or {}).get("image_size_wh") if isinstance(cams.get(side), Mapping) else None
            if cs is not None and list(cs) != list(sz):
                problems.append(("IMAGE_SIZE_MISMATCH", f"The {side} camera was calibrated at {list(cs)} but the contract says {list(sz)}."))
        if expected_image_size_wh is not None:
            try:
                want = [int(expected_image_size_wh[0]), int(expected_image_size_wh[1])]
            except (TypeError, ValueError, IndexError):
                want = None
            if want is None:
                problems.append(("INVALID_EXPECTED_SIZE", "The expected room-video size must be [width, height]."))
            elif want != [int(sz[0]), int(sz[1])]:
                problems.append(("CALIBRATION_VIDEO_INCOMPATIBLE", f"The calibration was made at {int(sz[0])}x{int(sz[1])} px but the room videos are {want[0]}x{want[1]} px; both must match, in the same orientation."))

    obs = None
    if d.get("schema") == SCHEMA_V2 and d.get("calibration_observations") is not None:      # a /2 file's observations must match its own calibration, or the file is rejected
        try:
            obs = _observation_module().validate_document_observations(dict(d))
        except Exception as e:
            problems.append(("INVALID_OBSERVATIONS", f"The calibration observations stored in this file are not valid ({str(e)[:160]}); the file may be corrupted or edited."))
    elif d.get("schema") == SCHEMA and d.get("calibration_observations") is not None:
        problems.append(("INVALID_OBSERVATIONS", "A version 1 result cannot contain calibration observations."))
    if measured_baseline_mm is not None:
        b = measured_baseline_mm
        if not (_num(b) and BASELINE_RANGE_MM[0] <= b <= BASELINE_RANGE_MM[1]):
            problems.append(("INVALID_MEASURED_BASELINE", f"The measured baseline must be a number between {BASELINE_RANGE_MM[0]:g} and {BASELINE_RANGE_MM[1]:g} mm."))
    if problems:
        raise HandoffError(problems)

    ext = d.get("extrinsics") if isinstance(d.get("extrinsics"), Mapping) else {}
    cal_b = ext.get("baseline_mm") if _num(ext.get("baseline_mm")) and ext.get("baseline_mm") > 0 else None
    stored_measured = (d.get("settings") or {}).get("measured_baseline_mm") if isinstance(d.get("settings"), Mapping) else None
    if measured_baseline_mm is not None:
        used, source = float(measured_baseline_mm), "measured"
    else:
        used = float(c["baseline_mm"])
        source = "measured" if _num(stored_measured) and stored_measured > 0 else "calibrated"
    concerns = [str(x) for x in (q.get("concerns") or [])]
    if cal_b is not None and source == "measured" and abs(used - cal_b) / used > BASELINE_DISAGREE_WARN:
        concerns.append(f"The baseline you measured ({used:.1f} mm) differs from the calibrated one ({cal_b:.1f} mm) by more than {int(BASELINE_DISAGREE_WARN * 100)}%. The measured value will be used.")
    if level == GOOD and concerns:                 # a GOOD result with concerns is shown as questionable: acceptance then needs the acknowledgement
        level_out = QUESTIONABLE
    else:
        level_out = level
    contract = dict(K1=[[float(x) for x in r] for r in c["K1"]], D1=[float(x) for x in c["D1"]], K2=[[float(x) for x in r] for r in c["K2"]], D2=[float(x) for x in c["D2"]],
                    image_size_wh=[int(sz[0]), int(sz[1])], capture_mode=dict(left=cm["left"].strip(), right=cm["right"].strip()), baseline_mm=used)
    rms = q.get("stereo_rms_px")
    return Handoff(level=level_out, headline=q.get("headline"), concerns=tuple(concerns), stereo_rms_px=float(rms) if _num(rms) else None, calibrated_baseline_mm=cal_b,
                   baseline_used_mm=used, baseline_source=source, image_size_wh=(int(sz[0]), int(sz[1])), job_id=d.get("job_id") if isinstance(d.get("job_id"), str) else None, _contract=contract,
                   observations_available=obs is not None, uncertainty_note=("corner observations retained; no calibration uncertainty has been computed" if obs is not None else "calibration uncertainty unavailable"),
                   _observations=obs)

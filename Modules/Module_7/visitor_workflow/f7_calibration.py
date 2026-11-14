"""Stage F7 (Module 7 Part 3, Phase 2B): guided visitor CALIBRATION wizard -- presentation-independent core (NO Streamlit import).

Checkerboard calibration from two videos (left / right camera), classical computer vision only:

    sequential decode  ->  cv2 checkerboard detection (``user_calibration.detect_board``: findChessboardCorners + cornerSubPix)
    -> left/right time offset from the two videos' frame-difference motion signals (cross-correlation)
    -> board-still frames only (so a small sync error does not matter), diverse board poses
    -> corner-ordering verification by epipolar geometry (``user_calibration.resolve_pair_orderings``)
    -> ``user_calibration.calibrate_stereo`` (cv2.calibrateCamera per camera + cv2.stereoCalibrateExtended)
    -> quality assessment (GOOD / QUESTIONABLE / FAILED) -> the Stage A input contract (K1, D1, K2, D2, image_size_wh, capture_mode, baseline_mm).

Nothing here is a new calibration algorithm: the solve, the corner detection and the ordering check are the existing Module 7 visitor-calibration
functions.  The new parts are video frame extraction, offset estimation, still/diverse pair selection and the wizard state machine.

Wizard rules (enforced here, not only in the page)
  * six steps, one at a time; ``next()`` refuses until the current step's requirements are met; entered values survive ``back()``;
  * a calibration is never accepted automatically: ``accept()`` needs a finished, current, non-FAILED assessment; QUESTIONABLE also needs
    ``acknowledged=True``; FAILED can never be accepted;
  * changing the checkerboard settings or a video invalidates any earlier result.
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Mapping

import cv2
import numpy as np

MODULE = Path(__file__).resolve().parents[1]
import sys                                                                                    # noqa: E402
if str(MODULE) not in sys.path: sys.path.insert(0, str(MODULE))

from room_measurement import visitor_calibration as VC, visitor_video_validation as VV       # noqa: E402
from stereo_measure import calibration_wizard as wiz, user_calibration as uc                 # noqa: E402

STEPS = ("Instructions", "Cameras & Checkerboard", "Upload Calibration Videos", "Process Calibration", "Quality Assessment", "Accept or Discard")
GOOD, QUESTIONABLE, FAILED = "GOOD", "QUESTIONABLE", "FAILED"
LIMITS = VV.VideoLimits(min_duration_s=10.0)          # Stage B limits (200 MB, 120 s maximum) plus a 10 s minimum: a pose sweep needs time
STRIDE = 5                                            # a detection attempt every 5th frame (~0.17 s at 30 fps)
MAX_PAIRS = 30
MIN_PAIRS = uc.MIN_VALID_PAIRS
OFFSET_SEARCH_S = 3.0
BASELINE_DISAGREE_WARN = 0.05                          # measured vs calibrated baseline
INSTRUCTIONS = (
    "Use the SAME two phones, mounted rigidly side by side, that you will use to film the room. Do not change zoom, lens or video mode afterwards.",
    "Print a checkerboard on a flat, rigid surface. Measure one square with a ruler.",
    "Film the board with BOTH phones at the same time (start both, then move the board).",
    "Hold the board still for about 2 seconds in each pose, then move to the next. Aim for 20 or more different poses.",
    "Vary the board's distance, tilt it, and place it near the edges and corners of the picture. Keep it fully visible to both cameras.",
    "Keep each video between 10 and 120 seconds and under 200 MB. Both videos must have the same resolution and orientation as your room videos.",
)


class WizardError(RuntimeError):
    pass


def _clean_name(s) -> str: return " ".join(str(s or "").split())[:60]


def _quick_has_board(gray, pattern) -> bool:
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    try: return bool(cv2.findChessboardCornersSB(small, (int(pattern[0]), int(pattern[1])), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)[0])
    except cv2.error: return False


# ------------------------------------------------------------------ detection from a video (sequential decode, no seeking)
def extract_detections(path, pattern, stride: int = STRIDE, progress: Callable[[float], None] | None = None, max_frames: int | None = None) -> dict:
    """Decode ``path`` sequentially; every ``stride``-th frame run board detection.  Returns dict(size, fps, n_frames, motion (per-frame mean abs diff of a
    small gray copy), detections [dict(frame, corners, square_px, sharp)]).  Frames are not kept."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise WizardError(f"cannot open {Path(path).name}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1; fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    motion, dets, prev, i, size = [], [], None, 0, None
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames is not None and i >= max_frames): break
        size = (fr.shape[1], fr.shape[0]); sm = cv2.resize(fr, (64, max(8, int(64 * fr.shape[0] / fr.shape[1]))), interpolation=cv2.INTER_AREA)
        small = (cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY) if sm.ndim == 3 else sm).astype(np.float32)
        motion.append(0.0 if prev is None or prev.shape != small.shape else float(np.abs(small - prev).mean())); prev = small
        g = None
        if i % stride == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
        if g is not None and _quick_has_board(g, pattern):                    # cheap quarter-scale test first: full detection on board-free frames is slow
            d = uc.detect_board(g, pattern)
            if d["status"] == "ok":
                c = d["corners"]; x0, y0 = np.floor(c.min(0)).astype(int); x1, y1 = np.ceil(c.max(0)).astype(int); pad = int(0.1 * max(x1 - x0, y1 - y0))
                crop = g[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]
                dets.append(dict(frame=i, corners=np.asarray(c, np.float64), square_px=float(d["square_px"]), sharp=float(cv2.Laplacian(crop, cv2.CV_32F).var()) if crop.size else 0.0))
        i += 1
        if progress and i % 30 == 0: progress(min(i / n_total, 1.0))
    cap.release()
    if progress: progress(1.0)
    return dict(size=size, fps=fps, n_frames=i, motion=np.asarray(motion, np.float64), detections=dets)


# ------------------------------------------------------------------ synchronisation / pairing / selection
def estimate_offset(motion_l, motion_r, fps: float, search_s: float = OFFSET_SEARCH_S) -> dict:
    """Frame offset ``d`` such that right frame ``i + d`` shows the same instant as left frame ``i`` (cross-correlation of the two motion signals)."""
    a, b = np.asarray(motion_l, float), np.asarray(motion_r, float); n = min(len(a), len(b)); L = int(min(round(search_s * fps), n // 8))
    if L < 5 or a.std() < 1e-9 or b.std() < 1e-9: return dict(offset_frames=0, peak=None, second=None, confident=False, reason="not enough motion to measure the offset")
    a, b = (a[:n] - a[:n].mean()) / a[:n].std(), (b[:n] - b[:n].mean()) / b[:n].std(); lags = np.arange(-L, L + 1); sc = []
    for d in lags:
        x, y = (a[:n - d], b[d:]) if d >= 0 else (a[-d:], b[:n + d]); sc.append(float((x * y).mean()))
    sc = np.asarray(sc); k = int(sc.argmax()); mask = np.abs(lags - lags[k]) > 5; second = float(sc[mask].max()) if mask.any() else 0.0
    conf = bool(sc[k] >= 0.3 and sc[k] - second >= 0.1)
    return dict(offset_frames=int(lags[k]) if conf else 0, peak=float(sc[k]), second=second, confident=conf, reason=None if conf else "the two motion signals do not agree clearly; pairing assumes no time offset and relies on board-still frames")


def still_flags(dets, stride: int = STRIDE, motion=None) -> list:
    """A detection is 'still' when the board is stable over at least two consecutive samples (displacement to one neighbouring sample below a tolerance) and the
    frame-difference motion around the frame is low.  Holding for ~2 s satisfies this easily; the first/last sample of a hold is accepted only if the motion check agrees."""
    by = {d["frame"]: d for d in dets}; out = []; mot = None if motion is None or len(motion) < 5 else np.asarray(motion, float)
    thr = None if mot is None else max(1.0, 4.0 * float(np.percentile(mot, 30)))
    for d in dets:
        tol = max(1.5, 0.08 * d["square_px"]); disp = []
        for nb in (d["frame"] - stride, d["frame"] + stride):
            if nb in by: disp.append(float(np.linalg.norm(by[nb]["corners"] - d["corners"], axis=1).mean()))
        ok = bool(disp) and min(disp) < tol
        if ok and thr is not None:
            f = d["frame"]; ok = float(mot[max(f - 2, 0):f + 3].max()) < thr
        out.append(ok)
    return out


def _features(c, size):
    g = c.reshape(-1, 2); w, h = size; cols_est = None
    top, bot = np.linalg.norm(g[1] - g[0]), np.linalg.norm(g[-1] - g[-2])
    return np.array([g[:, 0].mean() / w / 0.08, g[:, 1].mean() / h / 0.08, math.log(max(np.sqrt(np.ptp(g[:, 0]) * np.ptp(g[:, 1])), 1.0)) / 0.15, (top / max(bot, 1e-6) - 1) / 0.08])


def pair_detections(left: dict, right: dict, offset: int, stride: int = STRIDE, max_pairs: int = MAX_PAIRS) -> dict:
    """Still left detections matched to the nearest still right detection (after ``offset``), then reduced to diverse, sharp poses."""
    sl, sr = still_flags(left["detections"], stride, left.get("motion")), still_flags(right["detections"], stride, right.get("motion"))
    L = [d for d, s in zip(left["detections"], sl) if s]; R = [d for d, s in zip(right["detections"], sr) if s]
    if not L or not R: return dict(pairs=[], n_still_left=len(L), n_still_right=len(R), n_matched=0)
    rf = np.array([d["frame"] for d in R]); matched = []
    for d in L:
        j = int(np.abs(rf - (d["frame"] + offset)).argmin())
        if abs(int(rf[j]) - (d["frame"] + offset)) <= stride: matched.append((d, R[j]))
    chosen, feats = [], []
    for dl, dr in sorted(matched, key=lambda p: -(min(p[0]["sharp"], p[1]["sharp"]))):
        f = _features(dl["corners"], left["size"])
        if all(np.linalg.norm(f - g) > 1.0 for g in feats): chosen.append((dl, dr)); feats.append(f)
        if len(chosen) >= max_pairs: break
    chosen.sort(key=lambda p: p[0]["frame"])
    return dict(pairs=chosen, n_still_left=len(L), n_still_right=len(R), n_matched=len(matched))


# ------------------------------------------------------------------ the calibration run
def calibrate_from_detections(left: dict, right: dict, settings: Mapping, progress=None) -> dict:
    """Pure function of the two extraction dicts: sync -> pairs -> ordering check -> stereo solve.  Never raises for 'not enough data': returns ok=False + reason."""
    pattern = (int(settings["cols"]), int(settings["rows"])); sq = float(settings["square_mm"])
    out = dict(ok=False, error=None, model=None, pattern=pattern, square_mm=sq, size_left=left["size"], size_right=right["size"], fps=(left["fps"], right["fps"]),
               detections=dict(left=len(left["detections"]), right=len(right["detections"])))
    if left["size"] != right["size"]: out["error"] = f"The two videos have different resolutions ({left['size']} vs {right['size']})."; return out
    sync = estimate_offset(left["motion"], right["motion"], left["fps"]); out["sync"] = sync
    pr = pair_detections(left, right, sync["offset_frames"]); out["pairing"] = dict(n_still_left=pr["n_still_left"], n_still_right=pr["n_still_right"], n_matched=pr["n_matched"], n_selected=len(pr["pairs"]))
    if len(pr["pairs"]) < uc.MIN_PAIRS_TO_VERIFY: out["error"] = (f"Only {len(pr['pairs'])} usable board poses were visible to both cameras while the board was still "
                                                                 f"(found {len(left['detections'])} left and {len(right['detections'])} right detections). Need at least {MIN_PAIRS}.")
    if out["error"]: return out
    pairs = [(a["corners"], b["corners"]) for a, b in pr["pairs"]]; res = uc.resolve_pair_orderings(pairs, pattern, max(left["size"]))
    verdict = [p["status"] for p in res["pairs"]]; out["ordering"] = dict(counts={s: verdict.count(s) for s in set(verdict)}, messages=[p["message"] for p in res["pairs"] if p["status"] != "ok"][:5])
    keep = [i for i, p in enumerate(res["pairs"]) if p["status"] == "ok"]; out["pairs_used"] = len(keep); out["pairs_discarded"] = len(pairs) - len(keep)
    out["pair_frames"] = [(pr["pairs"][i][0]["frame"], pr["pairs"][i][1]["frame"]) for i in keep]
    if len(keep) < MIN_PAIRS: out["error"] = f"Only {len(keep)} board poses could be matched reliably between the two cameras; at least {MIN_PAIRS} are needed."; return out
    try:
        model = uc.calibrate_stereo([pairs[i][0] for i in keep], [res["pairs"][i]["right_corners"] for i in keep], pattern, sq, left["size"], right["size"], pair_ids=[i + 1 for i in keep])
    except (uc.CalibrationFailed, ValueError) as e: out["error"] = f"The calibration solver failed: {e}"; return out
    out.update(ok=True, model=model); return out


def contract_from_model(model: Mapping, baseline_mm: float | None = None) -> dict:
    """Stage A input contract from a calibration model.  The baseline is the visitor's measured one if given, else the calibrated |T| (metric: the board's square size is known)."""
    def d5(D):
        D = np.asarray(D, float).reshape(-1); return (D.tolist() + [0.0] * 5)[:5]
    w, h = model["image_size_left"]
    return dict(K1=np.asarray(model["K1"], float).tolist(), D1=d5(model["D1"]), K2=np.asarray(model["K2"], float).tolist(), D2=d5(model["D2"]), image_size_wh=[int(w), int(h)],
                capture_mode=dict(left="video", right="video"), baseline_mm=float(baseline_mm if baseline_mm else model["baseline_mm"]))


def assess(result: Mapping, measured_baseline_mm: float | None = None) -> dict:
    """GOOD / QUESTIONABLE / FAILED with plain-language reasons.  FAILED can never be accepted; QUESTIONABLE needs an explicit acknowledgement."""
    if not result.get("ok"):
        return dict(level=FAILED, headline="Calibration failed", problems=[result.get("error") or "Calibration could not be computed."], concerns=[], metrics=None, advice="Discard this attempt and film a new calibration pair.")
    m = result["model"]; problems = list(m.get("errors") or []); concerns = []
    try: contract = contract_from_model(m, measured_baseline_mm); adapted = VC.adapt_visitor_calibration(dict(K1=contract["K1"], D1=contract["D1"], K2=contract["K2"], D2=contract["D2"], image_size_left=contract["image_size_wh"], image_size_right=contract["image_size_wh"],
                                                                                                         camera=dict(left=dict(capture_mode="video"), right=dict(capture_mode="video"))), contract["baseline_mm"], name="visitor_video_calibration")
    except Exception as e: contract, adapted = None, None; problems.append(f"The calibration could not be converted for measurement ({type(e).__name__}).")
    if adapted is not None and not adapted.ok: problems += [i.message for i in adapted.errors]
    if adapted is not None: concerns += [i.message for i in adapted.warnings]
    concerns += [w["message"] for w in (m.get("warnings") or [])]
    sync = result.get("sync") or {}
    if not sync.get("confident"): concerns.append(f"Left/right timing could not be verified ({sync.get('reason')}).")
    disc = result.get("pairs_discarded", 0); tot = disc + result.get("pairs_used", 0)
    if tot and disc / tot > 0.3: concerns.append(f"{disc} of {tot} board poses were discarded because the left/right corner correspondence could not be verified.")
    if measured_baseline_mm and abs(measured_baseline_mm - m["baseline_mm"]) / measured_baseline_mm > BASELINE_DISAGREE_WARN:
        concerns.append(f"The baseline you measured ({measured_baseline_mm:.1f} mm) differs from the calibrated one ({m['baseline_mm']:.1f} mm) by more than {int(BASELINE_DISAGREE_WARN * 100)}%. The measured value will be used.")
    mt = m["metrics"]; metrics = dict(pairs_used=result.get("pairs_used"), pairs_discarded=disc, rms_left_px=mt["rms_left"], rms_right_px=mt["rms_right"], rms_stereo_px=mt["rms_stereo"],
                                      epipolar_median_px=(mt.get("epipolar_px") or {}).get("median"), calibrated_baseline_mm=m["baseline_mm"], measured_baseline_mm=measured_baseline_mm,
                                      baseline_used_mm=None if contract is None else contract["baseline_mm"], board_square_scale=(mt.get("board_square_scale") or {}).get("median"),
                                      distortion_left=m.get("distortion_left"), distortion_right=m.get("distortion_right"), time_offset_frames=sync.get("offset_frames"), image_size_wh=list(result["size_left"]),
                                      coverage_cells=(min(m["board"]["coverage_left"]["cells_used"], m["board"]["coverage_right"]["cells_used"]) if m.get("board") else None))
    if problems: return dict(level=FAILED, headline="Calibration failed its validity checks", problems=problems, concerns=concerns, metrics=metrics, advice="Discard this attempt and film a new calibration pair.")
    if concerns: return dict(level=QUESTIONABLE, headline="Usable, but with concerns", problems=[], concerns=concerns, metrics=metrics, advice="You may continue only if you accept that these issues can reduce measurement accuracy. Re-filming is safer.")
    return dict(level=GOOD, headline="Calibration looks good", problems=[], concerns=[], metrics=metrics, advice="These checks passed. They do not prove measurement accuracy.")


# ------------------------------------------------------------------ the wizard
class CalibrationWizard:
    """One visitor's calibration state.  Owns a temporary workspace for the two staged videos (``close()`` removes only that directory)."""

    def __init__(self, limits: VV.VideoLimits = LIMITS):
        self.limits = limits; self.step = 0; self.info = dict(left_camera="", right_camera="", cols=None, rows=None, square_mm=None, baseline_mm=None)
        self.ws = VV.TemporaryVideoWorkspace(prefix="f7_cal_"); self.videos = dict(left=None, right=None); self.result = None; self._assessment = None; self._used_settings = None
        self.acknowledged = False; self.decision = None; self.contract = None; self._thread = None; self._prog = dict(left=0.0, right=0.0); self._status = dict(state="idle", message="", error=None); self._lock = threading.Lock(); self.closed = False

    # ---- lifecycle
    def close(self):
        if not self.closed: self.closed = True; self.ws.close()

    # ---- step 2: information
    def set_info(self, **kw):
        for k, v in kw.items():
            if k not in self.info: raise KeyError(k)
            self.info[k] = v
        if self.result is not None and self._used_settings != self._settings_tuple(): self._invalidate_result()

    def _settings_tuple(self):
        s, _ = wiz.validate_settings(self.info["cols"], self.info["rows"], self.info["square_mm"]); return None if s is None else (s["cols"], s["rows"], s["square_mm"])

    def info_problems(self) -> list:
        p = []
        if not _clean_name(self.info["left_camera"]): p.append("Enter a name for the LEFT camera (for example 'iPhone 15 Pro').")
        if not _clean_name(self.info["right_camera"]): p.append("Enter a name for the RIGHT camera.")
        s, errs = wiz.validate_settings(self.info["cols"], self.info["rows"], self.info["square_mm"]); p += errs
        b = self.info["baseline_mm"]
        if b not in (None, "", 0):
            try: ok = isinstance(b, (int, float)) and not isinstance(b, bool) and math.isfinite(b) and 10 <= b <= 2000
            except Exception: ok = False
            if not ok: p.append("The measured distance between the cameras must be between 10 and 2000 mm (or leave it empty).")
        return p

    @property
    def settings(self):
        s, _ = wiz.validate_settings(self.info["cols"], self.info["rows"], self.info["square_mm"]); return s

    @property
    def baseline_mm(self): b = self.info["baseline_mm"]; return float(b) if isinstance(b, (int, float)) and not isinstance(b, bool) and b else None

    # ---- step 3: videos
    def set_video(self, side: str, data: bytes, name: str) -> dict:
        if side not in ("left", "right"): raise KeyError(side)
        old = self.videos[side]
        if old and Path(old["path"]).exists() and Path(self.ws.path) in Path(old["path"]).parents: Path(old["path"]).unlink()
        self._invalidate_result(); path = self.ws.write(bytes(data), suffix=Path(name or "v.mp4").suffix or ".mp4")
        rep = VV.inspect_video(path, f"{side}_calibration_video", self.limits); self.videos[side] = dict(path=str(path), name=Path(name or "").name[:80], report=rep, size_mb=len(data) / 1048576)
        return self.videos[side]

    def remove_video(self, side):
        v = self.videos[side]
        if v and Path(v["path"]).exists() and Path(self.ws.path) in Path(v["path"]).parents: Path(v["path"]).unlink()
        self.videos[side] = None; self._invalidate_result()

    def video_problems(self) -> list:
        p = []
        for side in ("left", "right"):
            v = self.videos[side]
            if v is None: p.append(f"Upload the {side} camera's calibration video."); continue
            p += [i.message for i in v["report"].errors]
        L, R = self.videos["left"], self.videos["right"]
        if L and R and not p:
            a, b = L["report"], R["report"]
            if (a.width, a.height) != (b.width, b.height): p.append(f"The two videos have different resolutions ({a.width}×{a.height} vs {b.width}×{b.height}). Both cameras must record at the same resolution.")
        return p

    def video_warnings(self) -> list:
        w = []
        for side in ("left", "right"):
            v = self.videos[side]
            if v: w += [i.message for i in v["report"].warnings]
        L, R = self.videos["left"], self.videos["right"]
        if L and R and L["report"].fps and R["report"].fps and abs(L["report"].fps - R["report"].fps) > 1.0: w.append("The two videos have different frame rates; timing will be harder to match.")
        return w

    # ---- navigation (the single source of truth for what 'Continue' may do)
    def blockers(self, step: int | None = None) -> list:
        s = self.step if step is None else step
        if s == 0: return []
        if s == 1: return self.info_problems()
        if s == 2: return self.info_problems() or self.video_problems()
        if s == 3:
            if self._status["state"] == "running": return ["Calibration is still running."]
            return [] if self.result is not None else ["Start the calibration first."]
        if s == 4: a = self.assessment(); return [] if a and a["level"] != FAILED else ["A failed calibration cannot be used. Discard it and recalibrate."]
        return ["This is the last step."]

    def can_continue(self) -> bool: return not self.blockers()

    def next(self) -> bool:
        if self.step >= len(STEPS) - 1 or self.blockers(): return False
        self.step += 1; return True

    def back(self) -> bool:
        if self.step == 0: return False
        self.step -= 1; return True

    def go_to_step(self, s: int):
        """Only to a step that is already reachable (never skips ahead past a blocker)."""
        if not 0 <= s < len(STEPS): raise ValueError("no such step")
        if s > self.step: raise WizardError("future steps stay hidden until the current one is complete")
        self.step = s

    # ---- step 4: processing
    def _invalidate_result(self):
        self.result = None; self._assessment = None; self.acknowledged = False; self.decision = None; self.contract = None; self._used_settings = None
        with self._lock: self._status = dict(state="idle", message="", error=None); self._prog = dict(left=0.0, right=0.0)

    def start_processing(self):
        if self.info_problems() or self.video_problems(): raise WizardError("complete the earlier steps first")
        if self._status["state"] == "running": return
        self._invalidate_result(); settings = self.settings; self._used_settings = self._settings_tuple()
        with self._lock: self._status = dict(state="running", message="Finding the checkerboard in both videos…", error=None)
        self._thread = threading.Thread(target=self._run, args=(settings,), daemon=True); self._thread.start()

    def _run(self, settings):
        pattern = (settings["cols"], settings["rows"]); t0 = time.perf_counter()
        def cb(side): return lambda f: self._prog.__setitem__(side, f)
        try:
            with ThreadPoolExecutor(2) as ex:
                fl = ex.submit(extract_detections, self.videos["left"]["path"], pattern, STRIDE, cb("left")); fr = ex.submit(extract_detections, self.videos["right"]["path"], pattern, STRIDE, cb("right"))
                left, right = fl.result(), fr.result()
            with self._lock: self._status["message"] = "Matching the two cameras and solving the calibration…"
            res = calibrate_from_detections(left, right, settings); res["runtime_s"] = time.perf_counter() - t0
            res["videos"] = {k: dict(name=v["name"], size_mb=round(v["size_mb"], 1), width=v["report"].width, height=v["report"].height, fps=v["report"].fps, duration_s=v["report"].duration_s) for k, v in self.videos.items()}
            self.result = res; self._assessment = None
            with self._lock: self._status = dict(state="done", message="Calibration finished.", error=None)
        except Exception as e:                                                                       # never leave the page spinning
            self.result = dict(ok=False, error=f"The calibration could not be computed ({type(e).__name__}: {e}).", model=None)
            with self._lock: self._status = dict(state="done", message="Calibration finished.", error=str(e))

    def process_blocking(self):
        self.start_processing(); self._thread.join(); return self.result

    def processing_status(self) -> dict:
        with self._lock: s = dict(self._status)
        s["progress"] = 0.5 * (self._prog["left"] + self._prog["right"]) * 0.9 + (0.1 if s["state"] == "done" else 0.0); return s

    # ---- step 5/6: assessment and decision
    def assessment(self) -> dict | None:
        if self.result is None: return None
        if self._assessment is None: self._assessment = assess(self.result, self.baseline_mm)
        return self._assessment

    def acknowledge(self, value: bool): self.acknowledged = bool(value)

    def can_accept(self) -> bool:
        a = self.assessment()
        if not a or self.decision is not None: return False
        return a["level"] == GOOD or (a["level"] == QUESTIONABLE and self.acknowledged)

    def accept(self) -> dict:
        """The visitor's explicit acceptance.  Returns the Stage A input contract.  FAILED -> never; QUESTIONABLE -> needs acknowledgement."""
        a = self.assessment()
        if a is None: raise WizardError("there is no calibration to accept")
        if a["level"] == FAILED: raise WizardError("a failed calibration cannot be accepted")
        if a["level"] == QUESTIONABLE and not self.acknowledged: raise WizardError("acknowledge the concerns before accepting this calibration")
        if self._used_settings != self._settings_tuple(): raise WizardError("the checkerboard settings changed after this calibration was computed")
        self.contract = contract_from_model(self.result["model"], self.baseline_mm); self.decision = "accepted"; return self.contract

    def discard(self):
        """Discard & Recalibrate: drops the result and any decision; keeps the entered information and the videos, and returns to the upload step."""
        self._invalidate_result(); self.step = 2

    def provenance(self) -> dict | None:
        if self.decision != "accepted": return None
        a = self.assessment(); return dict(source="visitor_video_calibration", cameras=dict(left=_clean_name(self.info["left_camera"]), right=_clean_name(self.info["right_camera"])), checkerboard=dict(pattern=self._used_settings[:2], square_mm=self._used_settings[2]),
                                           level=a["level"], acknowledged_concerns=a["concerns"] if a["level"] == QUESTIONABLE else [], metrics=a["metrics"], baseline_source="measured" if self.baseline_mm else "calibrated")

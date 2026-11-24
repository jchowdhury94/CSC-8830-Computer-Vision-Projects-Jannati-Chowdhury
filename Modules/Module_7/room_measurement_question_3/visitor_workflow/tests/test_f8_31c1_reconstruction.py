"""F8.31C-1: visitor Method J backend.  SYNTHETIC fixtures only; no original recording, no Recipe A value, no historical file.

Two kinds of test are kept apart on purpose:

* REAL feature extraction: the ``assets`` fixture renders photometric stereo frames of a made-up room and the sandboxed worker runs the original
  ``features.match_set`` (SIFT) on them.  Tests marked "REAL SIFT" use that run.
* MOCKED correspondences: tests marked "MOCKED" call ``reconstruction_worker.run_job`` in-process with a ``match_set`` stand-in that returns cached (real) or
  fabricated matches, to reach situations that cannot be produced naturally (sparse sets, garbage matches).  They prove wiring and classification, not that
  such matches occur.

Nothing here is real-video validation.  The known wall size (3000 x 2400 mm) is a sanity reference in assertions only; it is never an input.
"""
import copy
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import f8_visitor_service as FS
import resource_budget as RB
import reconstruction_worker as RW
import room_frames as RF
import room_reconstruction as RR
import room_videos as RV
import synthetic_room_render as SR
from room_measurement import features, geometry as G, lines as L, measure as M, pose as P, visitor_annotations as V, visitor_measurement as VM
from room_measurement import visitor_method_j_adapter as A
from test_f8_31a_room_videos import put
from test_f8_31b_frames import (PAIRS, W, H, accepted, annotate_all, choose_and_confirm, clips, complete, cstore, ready, rstore, sequential_gray, svc, upload_all)   # noqa: F401  (fixtures)
from test_f8_visitor_service import make_doc

HERE = Path(__file__).resolve().parents[1]
MODULE = HERE.parent
N_SETS = 5
TW, TH = SR.WALL_WIDTH_MM, SR.WALL_HEIGHT_MM
LIM = RR.ReconstructionLimits(required_frame_sets=None)             # the workflow fixtures use two frame sets; the default (five) is tested separately
FAKE_PY = lambda code: (lambda job: [sys.executable, "-c", code])   # noqa: E731


# ================================================================ synthetic assets (module scope: rendered once, real SIFT once, direct Method J once)
def call_params(ctx):
    return dict(K_left=ctx.K_left.tolist(), D_left=ctx.D_left.tolist(), K_right=ctx.K_right.tolist(), D_right=ctx.D_right.tolist(), image_size_wh=list(ctx.image_size_wh), baseline_mm=ctx.baseline_mm,
                baseline_unit="mm", camera_order=["left", "right"], distortion_model=RF.DISTORTION_MODEL, capture_mode={"left": "video", "right": "video"}, calibration_id="synthetic-calibration")


def make_request(tmp, ctx, frame_arrays, lines, *, owner="owner-digest", indices=None):
    rows, table, ls = [], [], []
    for i, fr in enumerate(frame_arrays):
        row, tb = {}, {}
        for k, g in fr.items():
            p = Path(tmp) / f"s{i}_{k}.png"; cv2.imwrite(str(p), g); idx = (indices or {}).get((i, k), 10 * i + len(k) + ord(k[0]))
            row[k] = dict(path=str(p), index=idx, fingerprint=V.frame_fingerprint(g)); tb[k] = idx
        rows.append(row); table.append(tb); ls.append({n: RR.serialize_line(l) for n, l in lines[i].items() if n in RR.REQUIRED_LINES})
    return RR.ReconstructionRequest(owner, call_params(ctx), tuple(rows), tuple(ls), {}, (), tuple(table))


def original_replay(ctx, match_sets, line_sets, y_min):
    """Direct execution of the ORIGINAL Method J functions in the original order (``pipeline.run_estimation`` -> ``measure.measure_all_sets``), with the one input-scale
    adaptation the adapter documents: the floor-row cut scaled to the frame height.  Returns (accepted, pose params, rows)."""
    cal = ctx.to_stereo_calibration(); D = P.build_dataset(cal, list(match_sets)); pJ, pA, prep, _ = P.estimate_pose(D)
    if prep["method_J"]["flags"]["degenerate"]:
        return False, np.asarray(pJ, float), None, prep
    R, T = G.pose_from_params(pJ, cal.baseline_mm); rows = []
    for i, (F, Ls) in enumerate(zip(match_sets, line_sets)):
        try:
            X1, a1, i1 = M.view_points(cal, R, T, F["A1"], F["B1"], F["M1"]); X2, a2, i2 = M.view_points(cal, R, T, F["A2"], F["B2"], F["M2"])
            f1, f2 = G.fit_floor_plane(X1, a1, y_min=y_min), G.fit_floor_plane(X2, a2, y_min=y_min)
            R0, t0, rms, ni, nt = M.register_views(X1, i1[:, 0], X2, i2[:, 0], F["MX"], np.random.default_rng(3 + i))
            g = dict(f1=f1, f2=f2, R0=R0, t0=t0, registration=dict(rms_mm=rms, n_inliers=ni, n_pairs=nt), n_points=(len(X1), len(X2)))
            o = M.measure_wall(cal, R, T, g, Ls, 0); rows.append(dict(ok=True, width_mm=o["width_mm"], height_mm=o["height_mm"]))
        except Exception:
            rows.append(dict(ok=False, width_mm=float("nan"), height_mm=float("nan")))
    return bool([r for r in rows if r["ok"]]), np.asarray(pJ, float), rows, prep


class Assets:
    pass


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    a = Assets(); a.tmp = tmp_path_factory.mktemp("recon"); a.ctx = SR.make_context(); room = SR.Room()
    a.frames = [SR.render_set(room, a.ctx, i) for i in range(N_SETS)]
    a.true_lines = [SR.true_lines(a.ctx, i) for i in range(N_SETS)]
    a.indices = {(i, k): idx for i in range(N_SETS) for k, idx in zip(RF.VIEW_KEYS, (40 + 7 * i, 85 + 11 * i, 52 + 5 * i, 130 - 9 * i))}      # independent per video, offsets differ per set
    a.request = make_request(a.tmp, a.ctx, a.frames, a.true_lines, indices=a.indices)
    a.y_min = VM.floor_row_threshold(a.ctx.image_size_wh[1])
    t = time.perf_counter(); a.matches = [features.match_set(f) for f in a.frames]; a.t_direct_match = time.perf_counter() - t       # the original matcher, called directly
    a.lines = [{n: RR.deserialize_line(d) for n, d in ls.items()} for ls in a.request.line_sets]
    t = time.perf_counter(); a.direct = original_replay(a.ctx, a.matches, a.lines, a.y_min); a.t_direct_geo = time.perf_counter() - t
    t = time.perf_counter(); a.result = RR.run_request(a.request, RR.ReconstructionLimits(), workspace_root=a.tmp, coordinator=RB.ResourceCoordinator(root=a.tmp / 'rb', cfg=RB.BudgetConfig(ceiling_mb=65536.0), container=dict(limit_mb=None, reliable=False, reason='test'))); a.t_backend = time.perf_counter() - t   # the real sandboxed worker, real SIFT
    return a


def cached_matcher(assets, subset=None):
    """MOCKED: stands in for ``match_set`` and returns the cached (real SIFT) matches in call order."""
    it = iter(assets.matches if subset is None else subset)
    return lambda frames: next(it)


def stage_and_run(tmp, request, **kw):
    import tempfile
    job = Path(tempfile.mkdtemp(prefix="job_", dir=str(tmp))); RR._stage(request, job, RR.ReconstructionLimits()); out = RW.run_job(job, **kw)
    out = json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o))); return out


# ================================================================ 1-2. visitor K, D and baseline  (REAL SIFT)
def test_real_backend_run_succeeds_on_synthetic_stereo_frames(assets):
    r = assets.result
    assert r["status"] == RR.SUCCESS and not r["failures"] and r["dimensions_accepted"], r["failures"]
    assert abs(r["width_mm"] - TW) < 15 and abs(r["height_mm"] - TH) < 15            # sanity reference of the synthetic scene ONLY; not evidence of real-world accuracy
    assert [p["status"] for p in r["per_frame_set"]] == ["reconstructed"] * N_SETS
    assert all(p["features"]["n_stereo_matches_view1"] > 1000 and p["features"]["n_four_view_tracks"] > 500 for p in r["per_frame_set"])    # SIFT really ran


def test_visitor_calibration_is_used_exactly_and_nothing_else(assets):
    c = assets.result["provenance"]["calibration"]; ctx = assets.ctx
    assert c["calibration_digest"] == ctx.digest() and c["baseline_mm"] == ctx.baseline_mm and c["baseline_unit"] == "mm" and c["image_size_wh"] == list(ctx.image_size_wh)
    assert c["camera_order"] == ["left", "right"] and "plumb-bob" in c["distortion_model"] and c["capture_mode"] == {"left": "video", "right": "video"}
    assert assets.result["diagnostics"]["calibration_use"]["digest"] == ctx.digest() and assets.result["diagnostics"]["calibration_use"]["lines_retargeted"] is False
    assert RR.context_from_params(assets.request.calibration).digest() == ctx.digest()                         # K, D of both cameras survive the JSON round trip bit for bit
    pr = assets.result["provenance"]; assert pr["recipe_a_used"] is False and pr["historical_baseline_used"] is False and pr["ground_truth_used"] is False


def test_the_visitor_baseline_scales_the_result_and_is_preserved_in_mm(assets, tmp_path):
    """MOCKED correspondences (cached real matches): only the baseline differs, and the output scales with it (the baseline is the metric scale).  Not bit-exact: the 40 mm registration
    threshold and the 800-20000 mm depth gate of Method J are absolute, so a rescaled scene differs by a few parts per million (measured: 4e-6)."""
    p = dict(assets.request.calibration, baseline_mm=assets.ctx.baseline_mm * 1.5)
    req = RR.ReconstructionRequest(*(p if f == "calibration" else getattr(assets.request, f) for f in assets.request.__dataclass_fields__))
    out = stage_and_run(tmp_path, req, match_set=cached_matcher(assets)); res = RR.assemble_result(req, worker_out=out)
    assert res["status"] == RR.SUCCESS and res["provenance"]["calibration"]["baseline_mm"] == pytest.approx(assets.ctx.baseline_mm * 1.5)
    assert res["width_mm"] == pytest.approx(assets.result["width_mm"] * 1.5, rel=1e-4) and res["height_mm"] == pytest.approx(assets.result["height_mm"] * 1.5, rel=1e-4)


def test_no_default_calibration_or_historical_input_exists_in_the_backend():
    src = {n: (HERE / n).read_text() for n in ("room_reconstruction.py", "reconstruction_worker.py")}
    for name, t in src.items():
        for bad in ("room_video_recipe_a", "DEFAULT_CALIBRATION_PATH", "DEFAULT_VIDEO_DIR", "DEFAULT_MANIFEST", "DEFAULT_ANNOTATIONS", "frame_manifest", "wall_boundary_annotations", "load_calibration",
                    "load_annotations", "load_frame_sets", "verify_videos", "225", "Recipe-A", "ground_truth", "validation"):
            assert bad not in re.sub(r"ground_truth_used", "", t), (name, bad)
        assert "import streamlit" not in t and "session_state" not in t                                     # no Streamlit dependency


# ================================================================ 3-4. slots, frames and independent indices  (workflow + real extracted frames)
def test_four_video_slot_mapping_and_independent_indices(svc, ready, clips, tmp_path):
    s, wf = ready; choose_and_confirm(wf); annotate_all(wf)
    prep = RR.prepare(wf, s.owner, s.contract, s.f6.job.cal, svc.room_sources(s), LIM); assert prep.request is not None, prep.failures
    req = prep.request; slot_of = RF.SLOT_OF_VIEW
    assert slot_of == {"AL": "p1_left", "PL": "p1_right", "AR": "p2_left", "PR": "p2_right"}
    for i in range(2):
        for vk in RF.VIEW_KEYS:
            slot = slot_of[vk]; vp = int(slot[1]); side = slot.split("_")[1]
            idx = PAIRS[vp][i][0 if side == "left" else 1]
            assert req.frame_table[i][vk] == req.frames[i][vk]["index"] == idx                                  # the visitor's own, independent index for this video
            assert np.array_equal(cv2.imread(req.frames[i][vk]["path"], cv2.IMREAD_GRAYSCALE), sequential_gray(clips[0][slot], idx))   # frame content untouched, from the right slot
    offsets = {(vp, i): PAIRS[vp][i][1] - PAIRS[vp][i][0] for vp in (1, 2) for i in range(2)}
    assert len(set(offsets.values())) > 1                                                                       # the left and right videos are NOT assumed to be synchronised
    assert [{k: v for k, v in t.items()} for t in req.frame_table] == wf.frame_table()
    assert set(req.videos) == set(RV.ROOM_UPLOAD_SLOTS) and all(len(v["sha256"]) == 64 for v in req.videos.values())


def test_boundary_annotations_map_to_exactly_the_six_adapter_lines(svc, complete):
    s, wf = complete
    req = RR.prepare(wf, s.owner, s.contract, s.f6.job.cal, svc.room_sources(s), LIM).request
    adapter_lines = wf.lines_for_adapter()
    assert RR.REQUIRED_LINES == A.REQUIRED_LINES == ("AL_corner", "PL_corner", "PL_ceilF", "AR_corner", "PR_corner", "PR_ceilF")
    for i in range(2):
        assert set(req.line_sets[i]) == set(RR.REQUIRED_LINES) and not {"PL_ceilS", "PR_ceilS"} & set(req.line_sets[i])         # no extra boundary is introduced
        for n in RR.REQUIRED_LINES:
            back, ref = RR.deserialize_line(req.line_sets[i][n]), adapter_lines[i][n]
            assert back["a"] == ref["a"] and back["b"] == ref["b"] and back["kind"] == ref["kind"] and back["n"] == ref["n"] and back["span"] == tuple(ref["span"]) and np.array_equal(back["pts"], ref["pts"])
    prov = {(a["frame_set"], a["line"]): a for a in req.annotations}; assert len(prov) == 12
    for (i, n), a in prov.items():
        assert a["points_distorted_full_px"] == [list(map(float, p)) for p in wf.ann[(i, n)].points] and a["status"] == wf.ann[(i, n)].result.status.value
        assert a["frame_fingerprint"] == req.frames[i][n[:2]]["fingerprint"] and a["frame_index"] == req.frame_table[i][n[:2]]


def test_workflow_backed_run_on_untextured_frames_is_a_typed_method_j_side_failure_without_dimensions(svc, complete, tmp_path):
    """REAL SIFT on drawn (untextured) frames: the original pipeline cannot match enough features, and the answer is typed, FAILED and carries no number."""
    s, wf = complete; t = time.perf_counter()
    r = RR.run_visitor_reconstruction(wf, s.owner, s.contract, s.f6.job.cal, svc.room_sources(s), LIM, workspace_root=tmp_path)
    assert r["status"] == "FAILED" and r["width_mm"] is None and r["height_mm"] is None and not r["dimensions_accepted"]
    assert r["failures"] and r["failure_class"] in ("method_j", "visitor_input") and all(f["code"] and f["visitor_message"] for f in r["failures"])
    assert all(p["width_mm"] is None and p["height_mm"] is None for p in r["per_frame_set"]) and "per_set" not in r["diagnostics"]
    json.dumps(r, allow_nan=False); assert not list(tmp_path.glob("f8_recon_*"))                                # the job directory is gone


# ================================================================ 5-6. the original feature matching and four-view tracks
def test_the_original_matcher_is_called_once_per_frame_set_with_the_exact_frames(assets, tmp_path):
    """MOCKED matches, REAL call chain: the spy records what the worker hands to ``match_set``."""
    seen = []
    def spy(frames):
        seen.append({k: np.array(v, copy=True) for k, v in frames.items()}); return assets.matches[len(seen) - 1]
    stage_and_run(tmp_path, assets.request, match_set=spy)
    assert len(seen) == N_SETS
    for i, fr in enumerate(seen):
        assert list(fr) == list(RF.VIEW_KEYS) and all(fr[k].dtype == np.uint8 and fr[k].shape == (SR.SIZE[1], SR.SIZE[0]) for k in fr)
        assert all(np.array_equal(fr[k], assets.frames[i][k]) for k in fr)                                         # unmodified: not resized, cropped, rotated or resampled


def test_the_worker_uses_the_original_match_set_with_its_original_parameters(monkeypatch, assets, tmp_path):
    import inspect
    d = inspect.signature(features.match_set).parameters
    assert (d["ratio"].default, d["ratio_cross"].default, d["nfeatures"].default, d["contrast"].default, d["seed"].default) == (0.75, 0.85, 12000, 0.04, 0)
    calls = []; real = features.match_set
    monkeypatch.setattr(features, "match_set", lambda *a, **k: (calls.append((a, k)), assets.matches[len(calls) - 1])[1])
    stage_and_run(tmp_path, assets.request)                                                                        # default matcher = ``features.match_set`` looked up at call time
    assert len(calls) == N_SETS and all(len(a) == 1 and not k for a, k in calls)                                   # no parameter override
    assert "match_set(frames)" in (HERE / "reconstruction_worker.py").read_text()


def test_real_sift_counts_and_four_view_tracks_equal_the_direct_original_functions(assets):
    for i, row in enumerate(assets.result["per_frame_set"]):
        F = assets.matches[i]; f = row["features"]
        assert f["n_stereo_matches_view1"] == len(F["A1"]) and f["n_stereo_matches_view2"] == len(F["A2"]) and f["n_cross_view_matches"] == len(F["MX"])
        assert f["n_four_view_tracks"] == len(features.build_tracks(F))                                          # original ``build_tracks``


# ================================================================ 7-9. adapter inputs and numerical parity with direct Method J
def test_adapter_receives_the_visitor_context_matches_and_lines(assets, tmp_path):
    """MOCKED matches; spy on the adapter entry point."""
    got = {}
    def spy(inputs, ctx, **k):
        got.update(inputs=inputs, ctx=ctx, kw=k); return A.evaluate(inputs, ctx, **k)
    stage_and_run(tmp_path, assets.request, match_set=cached_matcher(assets), evaluate=spy)
    inp, ctx = got["inputs"], got["ctx"]
    assert isinstance(inp, A.ReconstructionInputs) and isinstance(ctx, A.VisitorCalibrationContext) and not got["kw"]
    assert len(inp.match_sets) == len(inp.line_sets) == N_SETS and inp.match_sets[0] is assets.matches[0]
    assert ctx.digest() == assets.ctx.digest() and ctx.baseline_mm == assets.ctx.baseline_mm and ctx.same_intrinsics(inp.lines_calibration) and inp.problems(ctx) == []
    assert all(set(ls) == set(A.REQUIRED_LINES) for ls in inp.line_sets)


def test_backend_equals_direct_original_method_j_bit_for_bit(assets):
    """REAL SIFT in the worker versus the original functions called directly on the same frames (floor-row cut scaled to the frame height, as the adapter documents)."""
    accepted, pJ, rows, prep = assets.direct; r = assets.result
    assert accepted and all(x["ok"] for x in rows)
    assert r["diagnostics"]["pose"]["params"] == pJ.tolist()                                                       # shared pose identical
    for row, ref in zip(r["per_frame_set"], rows):
        assert row["width_mm"] == ref["width_mm"] and row["height_mm"] == ref["height_mm"]
    assert r["width_mm"] == float(np.mean([x["width_mm"] for x in rows])) and r["height_mm"] == float(np.mean([x["height_mm"] for x in rows]))
    fl = prep["method_J"]["flags"]; pose = r["diagnostics"]["pose"]
    assert (pose["robust_scale_px"], pose["inlier_fraction"], pose["converged"]) == (fl["scale_px"], fl["inlier_fraction"], fl["converged"])


# ================================================================ 10-11. original pose-failure behaviour; sparse-frame acceptance parity  (MOCKED matches)
def garbage_matches(n=400, seed=0):
    rng = np.random.default_rng(seed); W_, H_ = SR.SIZE; q = lambda: rng.uniform([0, 0], [W_, H_], (n, 2)).astype(np.float32); pair = np.c_[np.arange(n), np.arange(n)]      # noqa: E731
    return dict(A1=q(), B1=q(), M1=pair, A2=q(), B2=q(), M2=pair, MX=pair)


def test_original_pose_rejection_is_reproduced_as_a_typed_failure_without_dimensions(assets, tmp_path):
    """MOCKED: fabricated random matches.  The original rejects the pose; so does the backend, with typed codes, no dimensions, no per-set numbers."""
    garbage = [garbage_matches(seed=i) for i in range(3)]
    req = make_request(tmp_path, assets.ctx, assets.frames[:3], assets.true_lines[:3])
    out = stage_and_run(tmp_path, req, match_set=cached_matcher(assets, garbage))
    res = RR.assemble_result(req, worker_out=out)
    ok, _, _, prep = original_replay(assets.ctx, garbage, [{n: RR.deserialize_line(d) for n, d in ls.items()} for ls in req.line_sets], assets.y_min)
    assert not ok                                                                                                    # the original raises / rejects
    assert res["status"] == "FAILED" and res["width_mm"] is None and res["height_mm"] is None and not res["dimensions_accepted"]
    codes = {f["code"] for f in res["failures"]}; assert codes & {"POSE_DEGENERATE_UNRECOVERABLE", "POSE_JOINT_NOT_CONVERGED", "POSE_JOINT_SCALE_ABOVE_LIMIT", "NO_USABLE_FRAME_SETS", "DATASET_BUILD_FAILED"}, codes
    assert res["failure_class"] == "method_j" and res["diagnostics"]["original_pipeline_decision"]["rejected"] in (True, None)
    assert all(p["width_mm"] is None for p in res["per_frame_set"])


def test_wrong_calibration_focal_length_original_and_backend_agree_on_accept_or_reject(assets, tmp_path):
    """MOCKED matches (cached real SIFT) with the lens model of the request wrong by 40 %: the backend's decision equals a replay of the original pipeline's decision."""
    p = copy.deepcopy(assets.request.calibration)
    for k in ("K_left", "K_right"):
        p[k][0][0] *= 0.6; p[k][1][1] *= 0.6
    req = RR.ReconstructionRequest(*(p if f == "calibration" else getattr(assets.request, f) for f in assets.request.__dataclass_fields__))
    sub = assets.matches[:3]; req = RR.ReconstructionRequest(req.owner, req.calibration, req.frames[:3], req.line_sets[:3], req.videos, req.annotations, req.frame_table[:3])
    out = stage_and_run(tmp_path, req, match_set=cached_matcher(assets, sub)); res = RR.assemble_result(req, worker_out=out)
    ctx = RR.context_from_params(p); ok, pJ, rows, prep = original_replay(ctx, sub, [{n: RR.deserialize_line(d) for n, d in ls.items()} for ls in req.line_sets], assets.y_min)
    assert (res["status"] != "FAILED") == ok
    if not ok:
        assert res["width_mm"] is None and res["failure_class"] == "method_j"
    else:
        assert res["width_mm"] == float(np.mean([x["width_mm"] for x in rows if x["ok"]]))


def test_sparse_frame_set_acceptance_equals_the_original_and_the_old_minimum_is_not_reintroduced(assets, tmp_path):
    """MOCKED: set 0 is cut to 8 floor-point matches (8 four-view tracks) per view.  The original accepts every set; so does the backend, with the original's numbers.  The visitor mirror
    ``visitor_measurement._measure_sets`` (minimum of 10 gated points) would have rejected the sparse set; the backend does not use it."""
    def sparse(F, n_tracks):
        """Keep ``n_tracks`` four-view tracks whose left-image point is a floor point (row > the scaled floor cut): a few matches per view, tracks intact."""
        d1 = {int(q): i for i, q in enumerate(F["M1"][:, 0])}; d2 = {int(q): i for i, q in enumerate(F["M2"][:, 0])}
        keep = [(q, v) for q, v in F["MX"] if int(q) in d1 and int(v) in d2 and F["A1"][d1[int(q)]][1] > assets.y_min + 20 and F["A2"][d2[int(v)]][1] > assets.y_min + 20][:n_tracks]
        i1 = np.array([d1[int(q)] for q, _ in keep]); i2 = np.array([d2[int(v)] for _, v in keep]); out = dict(F)
        out["A1"], out["B1"], out["M1"] = F["A1"][i1], F["B1"][i1], F["M1"][i1]; out["A2"], out["B2"], out["M2"] = F["A2"][i2], F["B2"][i2], F["M2"][i2]
        out["MX"] = np.array(keep).reshape(-1, 2); return out
    sets = [sparse(assets.matches[0], 8)] + assets.matches[1:3]
    req = RR.ReconstructionRequest(assets.request.owner, assets.request.calibration, assets.request.frames[:3], assets.request.line_sets[:3], {}, (), assets.request.frame_table[:3])
    out = stage_and_run(tmp_path, req, match_set=cached_matcher(assets, sets)); res = RR.assemble_result(req, worker_out=out)
    lines = [{n: RR.deserialize_line(d) for n, d in ls.items()} for ls in req.line_sets]; ok, pJ, rows, _ = original_replay(assets.ctx, sets, lines, assets.y_min)
    assert ok and len(rows) == 3
    for row, ref in zip(res["per_frame_set"], rows):
        assert row["status"] == ("reconstructed" if ref["ok"] else "failed")
        if ref["ok"]:
            assert row["width_mm"] == ref["width_mm"] and row["height_mm"] == ref["height_mm"]
    mirror = VM._measure_sets(assets.ctx.to_stereo_calibration(), pJ, sets, lines, assets.ctx.image_size_wh)
    assert not mirror[0]["ok"]                                                                                      # the old 10-point rule WOULD have rejected set 0
    body = (HERE / "room_reconstruction.py").read_text() + (HERE / "reconstruction_worker.py").read_text()
    assert not re.search(r"\._measure_sets|\._reconstruct_set|import[^\n]*(_measure_sets|_reconstruct_set)", body)                    # the 10-point mirror is neither called nor imported
    assert [l.strip() for l in body.splitlines() if re.search(r"^(from|import) .*visitor_measurement", l)] == [l.strip() for l in body.splitlines() if l.startswith("from room_measurement.visitor_measurement import JobLimits")]


# ================================================================ 12-15. missing annotations, invalid calibration, expiry, replacement  (workflow)
def prepare(svc, s, wf, **kw):
    return RR.prepare(wf, kw.pop("owner", s.owner), kw.pop("contract", s.contract), kw.pop("cal", s.f6.job.cal), kw.pop("sources", svc.room_sources(s)), kw.pop("limits", LIM))


def codes(prep): return [f["code"] for f in prep.failures]


def test_a_complete_workflow_prepares_and_incomplete_ones_do_not(svc, ready, complete):
    s, wf = complete; p = prepare(svc, s, wf); assert p.request is not None and not p.failures
    assert RR.prepare(wf, s.owner, s.contract, s.f6.job.cal, svc.room_sources(s)).failures[0]["code"] == "FRAME_SET_COUNT"          # the default limits require the original five sets


def test_missing_annotations_are_typed_and_nothing_is_fabricated(svc, complete):
    s, wf = complete; wf.reject(0, "PR_ceilF"); wf.ann.pop((1, "AL_corner"))
    p = prepare(svc, s, wf); assert p.request is None and codes(p) == ["ANNOTATION_MISSING"] * 2
    assert {(f["frame_set"], f["line"]) for f in p.failures} == {(0, "PR_ceilF"), (1, "AL_corner")}
    r = RR.run_visitor_reconstruction(wf, s.owner, s.contract, s.f6.job.cal, svc.room_sources(s), LIM)
    assert r["status"] == "FAILED" and r["failure_class"] == "visitor_input" and r["width_mm"] is None and r["per_frame_set"] == []


def test_a_stale_annotation_frame_and_an_unextracted_frame_are_typed(svc, complete):
    s, wf = complete; wf.ann[(0, "AL_corner")].frame_fingerprint = "0" * 64; wf.full.pop((1, "PR"))
    got = codes(prepare(svc, s, wf)); assert "ANNOTATION_STALE" in got and "FRAMES_NOT_EXTRACTED" in got


def test_invalid_calibration_is_a_typed_failure_never_silently_corrected(svc, complete):
    s, wf = complete; cal = s.f6.job.cal
    assert codes(prepare(svc, s, wf, contract=None)) == ["CALIBRATION_NOT_ACCEPTED"]
    bad = copy.deepcopy(s.contract); bad["K1"][0][0] = float("nan"); assert codes(prepare(svc, s, wf, contract=bad)) == ["CALIBRATION_INVALID"]
    bad = copy.deepcopy(s.contract); bad["D2"] = [0.0] * 4; assert codes(prepare(svc, s, wf, contract=bad)) == ["CALIBRATION_INVALID"]
    bad = copy.deepcopy(s.contract); bad["baseline_mm"] = 0.0; assert "CALIBRATION_INVALID" in codes(prepare(svc, s, wf, contract=bad))        # the calibration object rejects a non-positive baseline
    bad = copy.deepcopy(s.contract); bad["baseline_mm"] = 231.0
    got = codes(prepare(svc, s, wf, contract=bad)); assert "CALIBRATION_CHANGED" in got and "CALIBRATION_INCONSISTENT" in got                    # a different baseline is a different calibration
    sources = {k: (None if v is None else RF.SourceVideo(v.slot, v.path, v.upload_id, v.sha256, v.size_bytes, v.frame_count, v.fps, (v.wh[1], v.wh[0]))) for k, v in svc.room_sources(s).items()}
    assert set(codes(prepare(svc, s, wf, sources=sources))) == {"CALIBRATION_VIDEO_SIZE_MISMATCH"}                                                  # orientation / size is compared, never rescaled
    assert prepare(svc, s, wf).request is not None                                                                                                 # the calibration itself was never modified by any of this


def test_expired_uploads_are_an_actionable_failure(svc, complete):
    s, wf = complete; src = svc.room_sources(s)
    gone = dict(src, p2_left=None); p = prepare(svc, s, wf, sources=gone); assert codes(p) == ["UPLOAD_EXPIRED"] and p.failures[0]["slot"] == "p2_left"
    assert "Upload the four room videos again" in p.failures[0]["visitor_message"]
    os.unlink(src["p1_right"].path); p = prepare(svc, s, wf); assert codes(p) == ["UPLOAD_EXPIRED"] and p.failures[0]["slot"] == "p1_right"
    r = RR.run_visitor_reconstruction(wf, s.owner, s.contract, s.f6.job.cal, svc.room_sources(s), LIM)
    assert r["status"] == "FAILED" and r["failure_class"] == "visitor_input" and r["failures"][0]["code"] == "UPLOAD_EXPIRED"


def test_a_corrupted_stored_video_fails_the_hash_check(svc, complete):
    s, wf = complete; path = Path(svc.room_sources(s)["p2_right"].path); b = bytearray(path.read_bytes()); b[len(b) // 2] ^= 0xFF; path.write_bytes(bytes(b))
    assert codes(prepare(svc, s, wf)) == ["VIDEO_HASH_MISMATCH"]
    assert prepare(svc, s, wf, limits=RR.ReconstructionLimits(required_frame_sets=None, verify_video_hashes=False)).request is not None        # the check is the only thing that catches it


def test_a_replaced_video_is_detected_and_its_frames_and_annotations_are_not_reused(svc, complete, clips):
    s, wf = complete; p = clips[0]["p1_left"]; data = bytearray(p.read_bytes()); data[-1] ^= 0xFF; data[-2] ^= 0x55
    put(svc.room_store, s.owner, "p1_left", bytes(data), name="replacement.mp4")
    svc._inspector = lambda path, t=svc._inspector: (t(path) if Path(path).stat().st_size in clips[1] else dict(clips[1][p.stat().st_size], size=Path(path).stat().st_size))
    got = codes(prepare(svc, s, wf)); assert got == ["VIDEO_REPLACED"], got                                              # caught BEFORE any synchronisation
    svc.frame_state(s)                                                                                                 # the workflow synchronises: viewpoint 1 is discarded
    got = set(codes(prepare(svc, s, wf))); assert "FRAME_SET_INCOMPLETE" in got and "ANNOTATION_MISSING" in got and not any(k[1] in ("AL", "PL") for k in wf.full)
    assert RR.run_visitor_reconstruction(wf, s.owner, s.contract, s.f6.job.cal, svc.room_sources(s), LIM)["status"] == "FAILED"


def test_a_new_calibration_invalidates_room_processing_readiness(svc, complete):
    s, wf = complete; c2 = copy.deepcopy(s.contract); c2["baseline_mm"] = 230.0
    cal2 = type("C", (), dict(image_size=(W, H), K_left=wf.cal.K_left, D_left=wf.cal.D_left, K_right=wf.cal.K_right, D_right=wf.cal.D_right, baseline_mm=230.0))()
    p = RR.prepare(wf, s.owner, c2, cal2, svc.room_sources(s), LIM); assert "CALIBRATION_CHANGED" in codes(p) and p.request is None
    wf.sync(s.owner, c2, cal2, svc.room_sources(s)); assert set(codes(RR.prepare(wf, s.owner, c2, cal2, svc.room_sources(s), LIM))) >= {"FRAME_SET_INCOMPLETE", "ANNOTATION_MISSING"}


# ================================================================ 16. visitor isolation
def test_another_visitor_cannot_use_this_workflow_or_its_frames(svc, complete, clips, tmp_path):
    s, wf = complete; other = accepted(svc, make_doc("GOOD")[1])
    p = RR.prepare(wf, other.owner, s.contract, s.f6.job.cal, svc.room_sources(s), LIM); assert codes(p) == ["OWNERSHIP_MISMATCH"] and p.request is None
    r = RR.run_visitor_reconstruction(wf, other.owner, s.contract, s.f6.job.cal, svc.room_sources(s), LIM)
    assert r["status"] == "FAILED" and r["failures"][0]["code"] == "OWNERSHIP_MISMATCH" and r["provenance"]["ground_truth_used"] is False
    upload_all(svc, other, clips)                                                                                      # the other visitor's own, byte-identical uploads
    assert set(codes(RR.prepare(wf, s.owner, s.contract, s.f6.job.cal, svc.room_sources(other), LIM))) == {"VIDEO_REPLACED"}      # their files are never mistaken for this visitor's
    outside = tmp_path / "foreign.png"; cv2.imwrite(str(outside), wf.frame_gray(0, "AL")); wf.full[(0, "AL")]["path"] = str(outside)
    assert "FRAME_STALE" in codes(prepare(svc, s, wf))                                                                  # a frame outside the visitor's own workspace is refused
    a, b = RR.ReconstructionWorkspace(tmp_path), RR.ReconstructionWorkspace(tmp_path)
    try:
        assert a.path != b.path and oct(a.path.stat().st_mode & 0o777) == "0o700" and RF.owner_digest(s.owner) != RF.owner_digest(other.owner)
    finally:
        a.close(); b.close()


def test_the_request_is_owned_and_the_worker_sees_only_its_job_directory(assets, tmp_path):
    seen = {}
    def spy(job):
        seen["files"] = sorted(str(p.relative_to(job)) for p in job.rglob("*") if p.is_file()); return [sys.executable, "-c", "import sys; sys.exit(5)"]
    r = RR.run_request(assets.request, RR.ReconstructionLimits(), worker_cmd=spy, workspace_root=tmp_path)
    assert sorted(f for f in seen["files"] if f.startswith("frames/")) == sorted(f"frames/s{i}_{k}.png" for i in range(N_SETS) for k in RF.VIEW_KEYS) and "request.json" in seen["files"]
    assert set(seen["files"]) == {"request.json"} | {f for f in seen["files"] if f.startswith("frames/")}            # nothing else is staged: no video, no other visitor's file
    assert not any(".." in f or f.startswith("/") for f in seen["files"]) and r["failures"][0]["code"] == "WORKER_CRASHED"
    env = RR._scrubbed_env(); assert not any(k.startswith(("F8_", "STREAMLIT", "AWS", "SECRET")) for k in env)


# ================================================================ 17-18. worker timeout, memory limits, cleanup
def alive(pid):
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wait_dead(pid, s=5.0):
    t = time.time()
    while time.time() - t < s and alive(pid):
        time.sleep(0.05)
    return not alive(pid)


def test_timeout_kills_the_whole_process_tree_and_leaves_nothing_behind(assets, tmp_path):
    pidfile = tmp_path / "child.pid"; root = tmp_path / "ws"; root.mkdir()
    cmd = lambda job: ["sh", "-c", f"sleep 300 & echo $! > {pidfile}; wait"]                                           # noqa: E731  (a worker that has started a child of its own)
    t = time.time(); r = RR.run_request(assets.request, RR.ReconstructionLimits(timeout_s=1.5, poll_s=0.1, kill_grace_s=1.0), worker_cmd=cmd, workspace_root=root)
    assert time.time() - t < 15 and r["status"] == "FAILED" and r["failures"][0]["code"] == "WORKER_TIMEOUT" and r["failure_class"] == "worker_resource"
    assert r["width_mm"] is None and "no valid reconstruction" in r["note"]
    pid = int(pidfile.read_text()); assert wait_dead(pid), "the worker's child survived"
    assert not list(root.iterdir())                                                                                    # the private workspace was removed
    assert RB.app_coordinator().state()["reservations"] == []                                                              # F8.31C-1C: the shared reservation was released


def test_memory_limit_kills_a_runaway_worker(assets, tmp_path):
    code = "import time\nx = bytearray(700 * 1024 * 1024)\nfor i in range(0, len(x), 4096): x[i] = 1\ntime.sleep(60)"
    root = tmp_path / "ws"; root.mkdir(); t = time.time()
    r = RR.run_request(assets.request, RR.ReconstructionLimits(max_tree_rss_mb=300, poll_s=0.1, timeout_s=60, kill_grace_s=1.0), worker_cmd=FAKE_PY(code), workspace_root=root)
    f = r["failures"][0]
    assert time.time() - t < 40 and f["code"] == "WORKER_MEMORY_LIMIT" and f["failure_class"] == "worker_resource" and r["status"] == "FAILED"
    assert r["processing"]["worker"]["tree_peak_rss_mb"] > 300 and not list(root.iterdir())


def test_an_unmeasurable_memory_limit_fails_closed(assets, tmp_path, monkeypatch):
    monkeypatch.setattr(RR, "snapshot_processes", lambda *a, **k: {})
    r = RR.run_request(assets.request, RR.ReconstructionLimits(poll_s=0.05, blind_polls_max=3, timeout_s=60, kill_grace_s=1.0), worker_cmd=FAKE_PY("import time; time.sleep(60)"), workspace_root=tmp_path)
    assert r["failures"][0]["code"] == "WORKER_MEMORY_UNMEASURABLE" and r["status"] == "FAILED"


def test_crash_garbage_and_oversized_output_are_typed_and_clean(assets, tmp_path):
    root = tmp_path / "ws"; root.mkdir()
    run = lambda code, **lim: RR.run_request(assets.request, RR.ReconstructionLimits(poll_s=0.05, **lim), worker_cmd=FAKE_PY(code), workspace_root=root)       # noqa: E731
    assert run("import os; os._exit(7)")["failures"][0]["code"] == "WORKER_CRASHED"
    cmd = lambda body: (lambda job: [sys.executable, "-c", f"import pathlib\npathlib.Path({str(job)!r}, 'result.json').write_text({body!r})"])                    # noqa: E731
    for body, code in (("not json", "WORKER_OUTPUT_INVALID"), ('{"schema": "wrong"}', "WORKER_OUTPUT_INVALID"), ('{"schema": "f8.reconstruction_worker/1", "adapter": {"outcome": "SUCCESS", "failures": [], "warnings": [], "width_mm": -5, "height_mm": 2}}', "WORKER_OUTPUT_INVALID")):
        r = RR.run_request(assets.request, RR.ReconstructionLimits(poll_s=0.05), worker_cmd=cmd(body), workspace_root=root)
        assert r["failures"][0]["code"] == code and r["status"] == "FAILED" and r["width_mm"] is None, body
    r = RR.run_request(assets.request, RR.ReconstructionLimits(poll_s=0.05, max_result_bytes=10), worker_cmd=cmd("x" * 100), workspace_root=root); assert r["failures"][0]["code"] == "WORKER_OUTPUT_INVALID"
    assert not list(root.iterdir())


def test_job_too_large_and_busy_slot_are_typed_and_clean(assets, tmp_path):
    root = tmp_path / "ws"; root.mkdir()
    r = RR.run_request(assets.request, RR.ReconstructionLimits(max_job_bytes=1000), worker_cmd=FAKE_PY("pass"), workspace_root=root)
    assert r["failures"][0]["code"] == "JOB_TOO_LARGE" and not list(root.iterdir())
    cancel = threading.Event(); out = {}
    th = threading.Thread(target=lambda: out.update(r=RR.run_request(assets.request, RR.ReconstructionLimits(poll_s=0.05, timeout_s=60), worker_cmd=FAKE_PY("import time; time.sleep(60)"), workspace_root=root, cancel=cancel)))
    th.start()
    try:
        t = time.time()
        while not RB.app_coordinator().state()["reservations"] and time.time() - t < 10:
            time.sleep(0.02)
        busy = RR.run_request(assets.request, RR.ReconstructionLimits(), worker_cmd=FAKE_PY("pass"), workspace_root=root)                      # one reconstruction at a time
        assert busy["failures"][0]["code"] == "WORKER_BUSY" and busy["failure_class"] == "worker_resource"
    finally:
        cancel.set(); th.join(30)
    assert out["r"]["failures"][0]["code"] == "WORKER_CANCELLED" and not list(root.iterdir())


def test_a_worker_failure_before_method_j_leaves_no_dimension(assets, tmp_path):
    """The REAL worker on frames whose recorded fingerprint no longer matches: typed, visitor-input class, no number."""
    bad = RR.ReconstructionRequest(assets.request.owner, assets.request.calibration, tuple({k: dict(v, fingerprint="0" * 64) for k, v in fs.items()} for fs in assets.request.frames[:2]),
                                   assets.request.line_sets[:2], {}, (), assets.request.frame_table[:2])
    root = tmp_path / "ws"; root.mkdir(); r = RR.run_request(bad, RR.ReconstructionLimits(), workspace_root=root)
    assert r["status"] == "FAILED" and r["failures"][0]["code"] == "FRAME_STALE" and r["failure_class"] == "visitor_input" and r["width_mm"] is None and not list(root.iterdir())


# ================================================================ failure-class separation and the result contract
def test_failure_families_and_advisory_warnings_stay_apart():
    for c in ("UPLOAD_EXPIRED", "VIDEO_REPLACED", "ANNOTATION_MISSING", "CALIBRATION_CHANGED", "INPUT_LINES_MISSING", "INPUT_MALFORMED_FRAME_SET", "FRAME_STALE"):
        assert RR.failure_class_of(c) == "visitor_input", c
    for c in ("POSE_DEGENERATE_UNRECOVERABLE", "POSE_JOINT_SCALE_ABOVE_LIMIT", "POSE_JOINT_NOT_CONVERGED", "NO_USABLE_FRAME_SETS", "FEATURE_MATCHING_FAILED", "FLOOR_PLANE_FAILED", "REGISTRATION_FAILED", "DATASET_BUILD_FAILED"):
        assert RR.failure_class_of(c) == "method_j", c
    for c in ("WORKER_TIMEOUT", "WORKER_MEMORY_LIMIT", "WORKER_CRASHED", "WORKER_BUSY", "JOB_TOO_LARGE", "INTERNAL_ERROR"):
        assert RR.failure_class_of(c) == "worker_resource", c
    assert not {f.value for f in A.QualityFlag} & {c.value for c in RR.Code}                                            # advisory flags are not failure codes
    assert all(c.value in RR.VISITOR_MESSAGES for c in RR.Code)                                                         # every code here has a visitor-readable message
    for fr in A.FailureReason:
        assert RR.failure_class_of(fr.value) in {"visitor_input", "method_j", "worker_resource"}


def test_advisory_warnings_never_change_a_number_or_become_failures(assets, tmp_path):
    """MOCKED: the adapter is told to use limits that make every advisory flag fire.  QUESTIONABLE, warnings present and advisory, numbers bit-identical."""
    lim = VM.QualityLimits(pose_inlier_fraction_warn=1.01, registration_min_inliers_warn=10 ** 9, registration_min_inlier_fraction_warn=1.01, registration_rms_mm_warn=0.0)
    out = stage_and_run(tmp_path, assets.request, match_set=cached_matcher(assets), evaluate=lambda i, c: A.evaluate(i, c, limits=lim)); res = RR.assemble_result(assets.request, worker_out=out)
    assert res["status"] == "QUESTIONABLE" and not res["failures"] and res["dimensions_accepted"] and res["warnings"] and all(w["advisory"] for w in res["warnings"])
    assert res["width_mm"] == assets.result["width_mm"] and res["height_mm"] == assets.result["height_mm"] and res["failure_class"] is None


def test_result_schema_is_json_safe_complete_and_makes_no_statistical_claim(assets):
    r = assets.result
    assert json.loads(json.dumps(r, allow_nan=False)) == r
    for k in ("schema", "status", "width_mm", "height_mm", "units", "per_frame_set", "failures", "failure_class", "warnings", "diagnostics", "processing", "provenance", "uncertainty", "dimensions_accepted"):
        assert k in r, k
    assert r["schema"] == "f8.visitor_reconstruction/1" and r["units"] == "mm" and r["uncertainty"]["status"] == "not_computed" and r["ground_truth_used"] is False
    blob = json.dumps(r).lower()
    assert "confidence_interval" not in blob and "percent_error" not in blob and "error_pct" not in blob and "gt_" not in blob
    pose = r["diagnostics"]["pose"]; assert {"params", "robust_scale_px", "inlier_fraction", "converged", "n_tracks", "epipolar_guard"} <= set(pose)
    assert r["diagnostics"]["registration"]["min_inliers"] > 0 and r["processing"]["seconds"] > 0 and r["processing"]["worker_self_peak_rss_mb"] > 0 and r["processing"]["stage_seconds"]["method_j"] > 0
    pr = r["provenance"]; assert pr["n_frame_sets"] == N_SETS and pr["frame_indices"][0] == dict(zip(RF.VIEW_KEYS, (40, 85, 52, 130))) and len(pr["frame_fingerprints"][0]) == 4


def test_a_failed_result_exposes_no_number_even_if_the_worker_document_has_some():
    """MOCKED worker document with dimensions inside a FAILED adapter result: nothing leaks."""
    req = RR.ReconstructionRequest("o", call_params(SR.make_context()), ({k: dict(path="p", index=1, fingerprint="f") for k in RF.VIEW_KEYS},), ({},), {}, (), ({k: 1 for k in RF.VIEW_KEYS},))
    ev = dict(outcome="FAILED", width_mm=2999.0, height_mm=float("nan"), failures=[dict(code="POSE_JOINT_SCALE_ABOVE_LIMIT", message="m", stage="pose")], warnings=[], notes=[], diagnostic_notes=[],
              diagnostics=dict(pose=dict(params=[0.0]), per_set=[dict(frame_set=0, width_mm=2999.0, height_mm=2400.0)], set_failures=[]))
    res = RR.assemble_result(req, worker_out=dict(schema="f8.reconstruction_worker/1", adapter=ev, sets=[], timings={}, peak_rss_mb=1.0))
    assert res["status"] == "FAILED" and res["width_mm"] is None and res["height_mm"] is None and "per_set" not in res["diagnostics"] and res["per_frame_set"][0]["width_mm"] is None
    assert res["failure_class"] == "method_j" and "2999" not in json.dumps(res["per_frame_set"]) and json.dumps(res, allow_nan=False)
    assert RR.json_safe(dict(a=float("inf"), b=np.float32(1.5), c=(1, np.int64(2)), d=np.array([np.nan, 1.0]))) == dict(a=None, b=1.5, c=[1, 2], d=[None, 1.0])


def test_serialised_lines_round_trip_exactly(assets):
    for ls_src, ls in zip(assets.true_lines, assets.request.line_sets):
        for n, d in ls.items():
            back = RR.deserialize_line(d); ref = ls_src[n]
            assert back["a"] == ref["a"] and back["b"] == ref["b"] and np.array_equal(back["pts"], ref["pts"]) and back["pts"].dtype == ref["pts"].dtype and back["span"] == tuple(ref["span"])


# ================================================================ 20. the public gate stays closed and nothing live uses the backend; originals are untouched
def test_public_reconstruction_gate_remains_disabled_and_the_backend_is_wired_to_nothing():
    assert FS.ROOM_PROCESSING_ENABLED is False
    src = (HERE / "f8_visitor_service.py").read_text()
    assert re.search(r"^ROOM_PROCESSING_ENABLED = False\b", src, re.M) and "environ" not in src.split("ROOM_PROCESSING_ENABLED = False")[0].splitlines()[-1]
    live = [p for p in list(HERE.glob("*.py")) + list((MODULE / "f8_chunk_poc").glob("*.py")) + list(MODULE.glob("*.py")) if p.name not in ("room_reconstruction.py", "reconstruction_worker.py", "f8_measurement.py")]     # F8.31C-2A: f8_measurement.py is the one trusted, gated service boundary
    for p in live:
        t = p.read_text(errors="ignore"); assert "room_reconstruction" not in t and "reconstruction_worker" not in t and "ROOM_PROCESSING_ENABLED = True" not in t, p.name
    assert "ROOM_PROCESSING_ENABLED" not in (HERE / "room_reconstruction.py").read_text().replace("``ROOM_PROCESSING_ENABLED`` is not consulted", "")
    assert not hasattr(FS.VisitorCalibrationService, "measure") and not hasattr(FS.VisitorCalibrationService, "run_reconstruction")


def test_original_method_j_and_calibration_sources_are_unmodified():
    names = ["pose.py", "measure.py", "geometry.py", "lines.py", "features.py", "pipeline.py", "calibration.py", "uncertainty.py"]
    r = subprocess.run(["git", "diff", "--name-only", "HEAD", "--"] + [f"Modules/Module_7/room_measurement_question_3/room_measurement/{n}" for n in names], cwd=MODULE.parents[2], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "", r.stdout
    body = (HERE / "room_reconstruction.py").read_text() + (HERE / "reconstruction_worker.py").read_text()
    for forbidden in ("def match_set", "def build_tracks", "def estimate_pose", "def build_dataset", "def triangulate", "def fit_floor_plane", "def measure_wall", "SIFT_create", "cv2.findEssentialMat", "cv2.triangulatePoints"):
        assert forbidden not in body                                                                                  # no mathematics, no alternative stereo algorithm

"""F8.18: local visitor calibration integration (service layer, pages, entry app).  SYNTHETIC DATA ONLY.

No original recording, derived original-room frame, Phase 1 calibration file or regression output is read: calibration numbers are made up, videos are a few random
bytes, the calibration worker is replaced by a fake runner, and the F6 session only validates numbers (``import_calibration``).  Room processing is asserted to stay off."""
import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7"
WF = MODULE / "visitor_workflow"
F8 = MODULE / "f8_chunk_poc"
for p in (str(MODULE), str(WF), str(F8), str(ROOT)):
    if p not in sys.path: sys.path.insert(0, p)

import calibration_runner as CR  # noqa: E402
import f6_core as F  # noqa: E402
import f7_calibration as K7  # noqa: E402
import f8_calibration_bridge as B  # noqa: E402
import f8_visitor_service as S  # noqa: E402
from chunk_store import ChunkStore, Limits  # noqa: E402
from result_store import ResultStore, build_result_document, new_code  # noqa: E402

SIZE = [2160, 3840]
K1 = [[3050.0, 0.0, 1075.0], [0.0, 3052.0, 1925.0], [0.0, 0.0, 1.0]]
K2 = [[3061.0, 0.0, 1081.0], [0.0, 3060.0, 1918.0], [0.0, 0.0, 1.0]]
D1 = [0.012, -0.045, 0.0004, -0.0002, 0.03]
D2 = [0.010, -0.040, -0.0003, 0.0001, 0.02]
CAL_BASELINE = 226.826
CS = 64 * 1024
APP = WF / "app_f8_integrated.py"


def make_doc(level="GOOD", measured=226.0, concerns=(), outcome="completed", mode="calibrate", code="synthetic-recovery-code-0123456789"):
    model = dict(K1=K1, D1=D1, K2=K2, D2=D2, image_size_left=SIZE, image_size_right=SIZE, baseline_mm=CAL_BASELINE, stereo_rms=0.7357, rms_left=0.6, rms_right=0.65,
                 R=np.eye(3).tolist(), T=[-CAL_BASELINE, 0.0, 0.0], distortion_left=None, distortion_right=None)
    contract = K7.contract_from_model(model, measured) if level != "FAILED" else None
    metrics = dict(pairs_used=15, pairs_discarded=1, rms_stereo_px=0.7357, calibrated_baseline_mm=CAL_BASELINE, measured_baseline_mm=measured, baseline_used_mm=measured or CAL_BASELINE)
    res = dict(mode=mode, outcome=outcome, ok=level != "FAILED", model=model, contract=contract, calibration=dict(pairs_used=15, pairs_discarded=1),
               assessment=dict(level=level, headline="synthetic headline", problems=[], concerns=list(concerns), advice="", metrics=metrics),
               validation=dict(videos={s: dict(width=2160, height=3840, fps=30.0) for s in ("left", "right")}))
    return build_result_document(res, code=code, settings=dict(cols=8, rows=6, square_mm=20.0, baseline_mm=measured))


def phase1_like(baseline=225.0):
    return dict(left=dict(K=K1, distortion_k1_k2_p1_p2_k3=D1), right=dict(K=K2, distortion_k1_k2_p1_p2_k3=D2), image_size_wh=SIZE, physical_baseline_mm=baseline)


def bare_contract(baseline=226.0):
    return dict(K1=K1, D1=D1, K2=K2, D2=D2, image_size_wh=SIZE, capture_mode=dict(left="video", right="video"), baseline_mm=baseline)


def J(o): return json.dumps(o).encode()


class FakeRunner:
    """Stands in for CalibrationJobRunner: same constructor / status / cancel surface, no worker process, no video."""
    instances = []
    finish_on_start = None                       # a document: the job finishes at once (the real page polls a running job forever, which AppTest cannot wait for)

    def __init__(self, store, owner, settings, limits=None, mode="calibrate", results=None, job_code=None, **kw):
        self.store, self.owner, self.settings, self.mode, self.results, self.job_code = store, owner, dict(settings), mode, results, job_code
        self.state, self.result, self.cancelled, self.started = "idle", None, False, False
        FakeRunner.instances.append(self)

    def start(self):
        self.started = True; self.state = "running"; CR._ACTIVE[self.job_code] = self
        if FakeRunner.finish_on_start is not None: self.finish(FakeRunner.finish_on_start)

    def status(self):
        return dict(state=self.state, stage="detect_left", fraction=0.4, elapsed_s=3.0, outcome=None, monitor=None, message="", started=1.0)

    @property
    def done(self): return self.state == "done"

    def cancel(self): self.cancelled = True

    def finish(self, doc, kept=False):
        self.results.save_result(self.job_code, doc)
        self.result = dict(outcome=doc["outcome"], cleanup=dict(videos_kept_for_next_step=kept, kept_for_retry_s=1800 if kept else None))
        self.state = "done"; CR._ACTIVE.pop(self.job_code, None)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("F8_RESULTS_DIR", str(tmp_path / "f8_results"))
    monkeypatch.delenv("F8_PASSCODE", raising=False)
    FakeRunner.instances.clear(); FakeRunner.finish_on_start = None
    yield
    for c in [c for c, r in CR._ACTIVE.items() if isinstance(r, FakeRunner)]: CR._ACTIVE.pop(c, None)


@pytest.fixture
def env(tmp_path):
    store = ChunkStore(Limits(min_chunk_bytes=1024, min_free_bytes=1, min_free_fraction=0.0, max_file_bytes=64 * 1024 * 1024), root=tmp_path / "chunks")
    results = ResultStore(tmp_path / "results")
    limits = CR.RunnerLimits(allow_full=True)
    svc = S.VisitorCalibrationService(store, results, limits, runner_cls=FakeRunner)
    yield svc
    store.close()


def upload_both(svc, sess):
    for slot in ("left", "right"):
        data = os.urandom(CS * 2 + 7)
        uid = svc.store.begin(sess.owner, "clip.mov", len(data), CS, slot, slot=slot)["upload_id"]
        for i in range(0, len(data), CS):
            c = data[i:i + CS]; svc.store.put_chunk(sess.owner, uid, i // CS, c, hashlib.sha256(c).hexdigest())
        svc.store.finish(sess.owner, uid, hashlib.sha256(data).hexdigest())


def run_job(svc, sess, doc_level="GOOD", measured=226.0, concerns=(), kept=False, outcome="completed"):
    upload_both(svc, sess)
    code = svc.start_calibration(sess, S.parse_setup(8, 6, 20.0, measured))
    sess.runner.finish(make_doc(doc_level, measured, concerns, outcome, code=code), kept=kept)
    return code


# ===================================================================== setup form
class TestSetupForm:
    def test_valid(self):
        s = S.parse_setup(8, 6, 20.0, 225)
        assert (s.cols, s.rows, s.square_mm, s.baseline_mm) == (8, 6, 20.0, 225.0) and S.parse_setup(8, 6, 20.0, 0).baseline_mm is None

    @pytest.mark.parametrize("args", [(1, 6, 20.0, 0), (8, 41, 20.0, 0), (8, 8, 20.0, 0), (8, 6, 0.5, 0), (8, 6, 20.0, 5), (8, 6, 20.0, 5000), (8.5, 6, 20.0, 0), (True, 6, 20.0, 0),
                                      (8, 6, float("nan"), 0), (8, 6, 20.0, float("nan"))])
    def test_invalid(self, args):
        with pytest.raises(S.ServiceError): S.parse_setup(*args)


# ===================================================================== Option A
class TestOptionA:
    def test_full_path_good(self, env):
        sess = S.VisitorSession(); code = run_job(env, sess)
        rev = env.finished_review(sess)
        assert rev.level == "GOOD" and rev.quality_assessed_by_f8 and rev.baseline_used_mm == 226.0 and rev.baseline_source == "measured" and rev.calibrated_baseline_mm == pytest.approx(CAL_BASELINE)
        assert sess.f6 is None                                         # nothing imported before acceptance
        a = env.accept(sess)
        assert a.ok and a.can_proceed and a.baseline_used_mm == 226.0 and sess.step == "ready" and tuple(a.image_size_wh) == (2160, 3840)
        assert sess.f6.calibration_provenance["source"] == "f8_cloud_calibration" and sess.f6.calibration_provenance["baseline_used_mm"] == 226.0
        assert code not in json.dumps(sess.f6.calibration_provenance) and code not in json.dumps(a.provenance)       # the secret code is not part of provenance

    def test_calibrated_baseline_is_used_when_none_measured(self, env):
        sess = S.VisitorSession(); run_job(env, sess, measured=None)
        rev = env.finished_review(sess)
        assert (rev.baseline_source, rev.baseline_used_mm) == ("calibrated", pytest.approx(CAL_BASELINE))
        assert env.accept(sess).baseline_used_mm == pytest.approx(CAL_BASELINE)

    def test_questionable_needs_acknowledgement(self, env):
        sess = S.VisitorSession(); run_job(env, sess, "QUESTIONABLE", concerns=["Fewer poses than recommended."])
        rev = env.finished_review(sess)
        assert rev.level == "QUESTIONABLE" and rev.requires_acknowledgement and not rev.can_accept(False) and rev.can_accept(True)
        with pytest.raises(S.ServiceError): env.accept(sess, acknowledged=False)
        assert sess.accepted is None and sess.f6 is None and sess.step != "ready"
        assert env.accept(sess, acknowledged=True).ok

    def test_failed_can_never_be_accepted(self, env):
        sess = S.VisitorSession(); run_job(env, sess, "FAILED")
        rev = env.finished_review(sess)
        assert rev.level == "FAILED" and not rev.usable and not rev.can_accept(True) and rev.problems
        with pytest.raises(S.ServiceError): env.accept(sess, acknowledged=True)
        assert sess.f6 is None and sess.accepted is None

    def test_unfinished_job_has_no_review(self, env):
        sess = S.VisitorSession(); upload_both(env, sess); env.start_calibration(sess, S.parse_setup(8, 6, 20.0))
        with pytest.raises(S.ServiceError): env.finished_review(sess)
        assert sess.busy

    def test_job_that_did_not_complete_is_failed(self, env):
        sess = S.VisitorSession(); run_job(env, sess, "GOOD", outcome="timeout")
        rev = env.finished_review(sess)
        assert rev.level == "FAILED" and not rev.can_accept(True)

    def test_start_requires_both_videos_and_full_calibration_and_is_one_shot(self, env):
        sess = S.VisitorSession(); setup = S.parse_setup(8, 6, 20.0)
        with pytest.raises(S.ServiceError, match="LEFT"): env.start_calibration(sess, setup)
        upload_both(env, sess)
        off = S.VisitorCalibrationService(env.store, env.results, CR.RunnerLimits(allow_full=False), runner_cls=FakeRunner)
        with pytest.raises(S.ServiceError, match="not enabled"): off.start_calibration(sess, setup)
        assert FakeRunner.instances == []
        env.start_calibration(sess, setup)
        with pytest.raises(S.ServiceError): env.start_calibration(sess, setup)
        assert len(FakeRunner.instances) == 1 and FakeRunner.instances[0].mode == "calibrate"

    def test_cancel(self, env):
        sess = S.VisitorSession(); upload_both(env, sess); env.start_calibration(sess, S.parse_setup(8, 6, 20.0))
        assert env.cancel(sess) and sess.runner.cancelled

    def test_cannot_leave_while_running(self, env):
        sess = S.VisitorSession(); upload_both(env, sess); env.start_calibration(sess, S.parse_setup(8, 6, 20.0))
        for step in ("home", "import", "review"):
            with pytest.raises(S.ServiceError): sess.go(step)
        sess.go("setup")

    def test_retry_keeps_videos_only_when_f8_kept_them(self, env):
        sess = S.VisitorSession(); run_job(env, sess, "FAILED", kept=True)
        assert env.retry_possible(sess); epoch = sess.epoch
        env.clear_result(sess)
        assert sess.runner is None and sess.epoch == epoch and env.slots_ready(sess)
        sess2 = S.VisitorSession(); run_job(env, sess2, "FAILED", kept=False); epoch2 = sess2.epoch
        env.clear_result(sess2); assert sess2.epoch == epoch2 + 1

    def test_result_download_bytes_are_the_saved_document(self, env):
        sess = S.VisitorSession(); code = run_job(env, sess)
        data = env.result_bytes(sess)
        assert json.loads(data)["schema"] == B.SCHEMA_V2 and B.parse_f8_result(data).accept()["baseline_mm"] == 226.0     # F8.29: new documents are /2 (legacy /1 is covered below)

    def test_new_documents_are_schema_2_and_legacy_schema_1_still_imports(self, env):
        """F8.31A-B: the service classifier must treat /1 and /2 alike (it once recognised only /1, so every /2 file was rejected as a bare Phase 1 file)."""
        doc2 = make_doc("GOOD"); assert doc2["schema"] == B.SCHEMA_V2
        legacy = {k: v for k, v in doc2.items() if not k.startswith("calibration_")}; legacy["schema"] = B.SCHEMA        # a genuine /1 document has none of the F8.29 keys
        for label, doc in (("/2", doc2), ("/1", legacy)):
            sess = S.VisitorSession(); rev = env.import_saved(sess, J(doc))
            assert rev.kind == "f8_file" and rev.level == "GOOD", label
            a = env.accept(sess); assert a.ok and a.baseline_used_mm == 226.0, label
        bad = dict(doc2, schema="f8.calibration_result/9")                                                                # unknown schemas are still not F8 results
        assert env.import_saved(S.VisitorSession(), J(bad)).kind != "f8_file"


# ===================================================================== Option B
class TestOptionB:
    def test_f8_result_file(self, env):
        sess = S.VisitorSession(); rev = env.import_saved(sess, J(make_doc("GOOD")))
        assert rev.kind == "f8_file" and rev.level == "GOOD" and rev.quality_assessed_by_f8 and sess.step == "review" and sess.f6 is None
        a = env.accept(sess); assert a.ok and a.baseline_used_mm == 226.0 and sess.f6.calibration_provenance["quality_assessed_by_f8"] is True

    def test_f8_result_questionable_and_failed(self, env):
        sess = S.VisitorSession()
        q = env.import_saved(sess, J(make_doc("QUESTIONABLE", concerns=["x"]))); assert q.level == "QUESTIONABLE" and not q.can_accept(False) and q.can_accept(True)
        f = env.import_saved(sess, J(make_doc("FAILED"))); assert f.level == "FAILED" and not f.can_accept(True)
        with pytest.raises(S.ServiceError): env.accept(sess, True)

    def test_f8_baseline_override(self, env):
        rev = S.review_saved_file(J(make_doc("GOOD", measured=None)), baseline_mm=226)
        assert (rev.baseline_source, rev.baseline_used_mm) == ("measured", 226.0) and rev.calibrated_baseline_mm == pytest.approx(CAL_BASELINE)
        big = S.review_saved_file(J(make_doc("GOOD")), baseline_mm=300)
        assert big.level == "QUESTIONABLE" and any("differs" in c for c in big.concerns)

    def test_bare_contract_is_not_assessed_and_needs_acknowledgement(self, env):
        sess = S.VisitorSession(); rev = env.import_saved(sess, J(bare_contract()))
        assert rev.kind == "contract_file" and rev.level == "NOT_ASSESSED" and not rev.quality_assessed_by_f8 and rev.baseline_used_mm == 226.0 and rev.level != "GOOD"
        assert not rev.can_accept(False) and rev.can_accept(True)
        with pytest.raises(S.ServiceError): env.accept(sess, False)
        a = env.accept(sess, True)
        assert a.ok and a.baseline_used_mm == 226.0 and sess.f6.calibration_provenance["quality_assessed_by_f8"] is False and sess.f6.calibration_provenance["source"] == "imported_calibration_file"

    def test_phase1_style_file_needs_declared_capture_mode_and_never_guesses(self, env):
        none = S.review_saved_file(J(phase1_like()))
        assert none.level == "INVALID" and none.problems                                          # F6's validation: capture mode is required, not guessed
        ok = S.review_saved_file(J(phase1_like()), capture_mode="video")
        assert ok.kind == "phase1_file" and ok.level == "NOT_ASSESSED" and ok.baseline_used_mm == 225.0 and ok.baseline_source == "file"
        entered = S.review_saved_file(J(phase1_like(baseline=None)), capture_mode="video", baseline_mm=224.0)
        assert entered.usable and entered.baseline_used_mm == 224.0 and entered.baseline_source == "entered"

    def test_bare_contract_missing_baseline_is_invalid_until_entered(self):
        c = bare_contract(); c.pop("baseline_mm")
        assert S.review_saved_file(J(c)).level == "INVALID"
        assert S.review_saved_file(J(c), baseline_mm=226.0).usable

    @pytest.mark.parametrize("payload", [b"not json", b"[1,2]", b"null", J({}), J({"schema": B.SCHEMA}), J({"left": {}, "right": {}}),
                                         J({"K1": [[1, 2], [3, 4]], "D1": D1, "K2": K2, "D2": D2, "image_size_wh": SIZE, "baseline_mm": 100}),
                                         J(make_doc("GOOD", mode="benchmark")), b"\xff\xfe", b"{" * 5000])
    def test_invalid_or_incompatible_files(self, payload):
        rev = S.review_saved_file(payload)
        assert rev.level in ("INVALID", "FAILED") and not rev.usable and not rev.can_accept(True)
        with pytest.raises(S.ServiceError): rev.contract_for_accept(True)

    def test_oversized_file_is_refused(self):
        assert S.review_saved_file(b" " * (S.MAX_FILE_BYTES + 1)).level == "INVALID"

    def test_tampered_f8_result(self):
        d = make_doc("GOOD"); d["cameras"]["left"]["K"][0][0] += 5
        assert S.review_saved_file(J(d)).level == "INVALID"
        d = make_doc("GOOD"); d["contract"]["K1"][0][0] = float("nan")
        assert S.review_saved_file(json.dumps(d).encode()).level == "INVALID"

    def test_a_calibration_f6_rejects_never_reaches_the_session(self, env):
        c = bare_contract(); c["image_size_wh"] = [100, 100]
        rev = S.review_saved_file(J(c))
        if rev.usable:                                   # if F6 tolerates the size, acceptance must still go through F6 and report the truth
            sess = S.VisitorSession(); sess.review = rev; a = env.accept(sess, True); assert a.ok == a.can_proceed
        else:
            assert rev.level == "INVALID"

    def test_import_then_export_roundtrip(self, env):
        sess = S.VisitorSession(); env.import_saved(sess, J(make_doc("GOOD"))); env.accept(sess)
        again = S.review_saved_file(env.contract_json(sess))
        assert again.level == "NOT_ASSESSED" and again.baseline_used_mm == 226.0


# ===================================================================== F6 hand-off
class TestF6Handoff:
    def test_exactly_the_validated_contract_reaches_f6_and_nothing_else(self, env):
        sess = S.VisitorSession(); env.import_saved(sess, J(make_doc("GOOD"))); env.accept(sess)
        job = sess.f6.job
        assert job.stage == "calibrated" or "calib" in job.stage                                  # the F6 job stops at the calibration stage
        assert job.baseline_mm == 226.0 and not sess.f6.reviews and not sess.f6.detections and sess.f6.measurement is None and job.video_validation is None
        assert sess.contract["baseline_mm"] == 226.0 and set(sess.contract) == set(B.CONTRACT_FIELDS)

    def test_f6_rejection_is_reported_and_leaves_no_provenance(self, env):
        sess = S.VisitorSession(); rev = S.review_saved_file(J(bare_contract()))
        bad = copy.copy(rev); object.__setattr__(bad, "_contract", dict(rev._contract, baseline_mm=-1.0)); sess.review = bad
        a = env.accept(sess, True)
        assert a.ok is False and a.problems and sess.accepted is None and sess.f6.calibration_provenance is None and sess.step != "ready"

    def test_level_and_provenance_for_questionable(self, env):
        sess = S.VisitorSession(); env.import_saved(sess, J(make_doc("QUESTIONABLE", concerns=["n"]))); a = env.accept(sess, True)
        assert a.level == "QUESTIONABLE" and sess.f6.calibration_provenance["acknowledged_concerns"] == ["n"]


# ===================================================================== recovery, refresh, session loss
class TestRecovery:
    def test_new_session_finds_a_saved_result_by_code(self, env):
        code = new_code(); env.results.save_result(code, make_doc("GOOD", code=code))
        fresh = S.VisitorSession()                                   # a refreshed / new browser session
        info = env.recover(code); assert info["state"] == "result" and info["review"].level == "GOOD"
        rev = env.use_recovered(fresh, code); assert rev.kind == "f8_recovered" and fresh.step == "review" and env.accept(fresh).ok
        assert env.result_bytes(fresh)                                # download still works for the recovered job

    def test_old_session_loss_does_not_lose_the_result(self, env):
        a = S.VisitorSession(); code = run_job(env, a)
        a.reset(env.store)                                           # the first session is gone (start over / disconnect)
        b = S.VisitorSession(); assert env.use_recovered(b, code).level == "GOOD"

    def test_running_job_is_visible_and_cancellable_from_another_session(self, env):
        a = S.VisitorSession(); upload_both(env, a); code = env.start_calibration(a, S.parse_setup(8, 6, 20.0))
        info = env.recover(code); assert info["state"] == "running" and 0 <= info["fraction"] <= 1
        assert env.cancel_recovered(code) and a.runner.cancelled
        with pytest.raises(S.ServiceError): env.use_recovered(S.VisitorSession(), code)

    @pytest.mark.parametrize("code", ["", "   ", None, "short", "x" * 100, "../../etc/passwd", 123])
    def test_malformed_codes(self, env, code):
        assert env.recover(code)["state"] == "invalid" and not env.cancel_recovered(code)

    def test_unknown_code_and_benchmark_code(self, env):
        assert env.recover(new_code())["state"] == "not_found"
        c = new_code(); env.results.save_result(c, make_doc("GOOD", mode="benchmark", code=c)); assert env.recover(c)["state"] == "not_found"

    def test_interrupted_job(self, env):
        c = new_code(); env.results.write_status(c, dict(state="running", stage="solve"))
        assert env.recover(c)["state"] == "interrupted"

    def test_failed_saved_result_is_shown_as_failed(self, env):
        c = new_code(); env.results.save_result(c, make_doc("FAILED", code=c))
        assert env.recover(c)["review"].level == "FAILED"

    def test_reclaim_videos_moves_ownership_after_session_loss(self, env):
        a = S.VisitorSession(); upload_both(env, a); code = new_code(); env.store.set_recovery(a.owner, code)
        b = S.VisitorSession(); assert env.reclaim_videos(b, code) == 2 and env.slots_ready(b) and not env.slots_ready(a)
        with pytest.raises(S.ServiceError): env.reclaim_videos(b, new_code())

    def test_other_sessions_cannot_use_my_uploads(self, env):
        a, b = S.VisitorSession(), S.VisitorSession(); upload_both(env, a)
        assert env.slots_ready(a) and not env.slots_ready(b)
        with pytest.raises(S.ServiceError): env.start_calibration(b, S.parse_setup(8, 6, 20.0))


# ===================================================================== navigation and start-over cleanup
class TestNavigationAndCleanup:
    def test_navigation_rules(self, env):
        sess = S.VisitorSession()
        with pytest.raises(S.ServiceError): sess.go("ready")
        with pytest.raises(S.ServiceError): sess.go("nowhere")
        env.import_saved(sess, J(make_doc("GOOD"))); assert sess.step == "review" and sess.review is not None
        sess.go("import"); assert sess.review is None
        sess.go("home"); assert sess.step == "home"

    def test_start_over_cleans_everything(self, env):
        sess = S.VisitorSession(); run_job(env, sess); env.finished_review(sess); env.accept(sess)
        ws = sess.f6.workspace_path(); owner, epoch = sess.owner, sess.epoch
        assert ws is not None and Path(ws).exists() and env.slots_ready(sess)
        sess.reset(env.store)
        assert (sess.f6, sess.review, sess.runner, sess.job_code, sess.accepted, sess.contract, sess.setup) == (None,) * 7
        assert sess.step == "home" and sess.epoch == epoch + 1 and sess.owner != owner and sess.protos == {"left": {}, "right": {}}
        assert not Path(ws).exists()                                  # F6 workspace removed
        assert env.store.job_status(owner)["slots"] == {}             # uploads deleted
        with pytest.raises(S.ServiceError): sess.go("ready")

    def test_start_over_cancels_a_running_job(self, env):
        sess = S.VisitorSession(); upload_both(env, sess); env.start_calibration(sess, S.parse_setup(8, 6, 20.0)); r = sess.runner
        sess.reset(env.store); assert r.cancelled and sess.runner is None

    def test_a_second_session_is_unaffected_by_the_first_start_over(self, env):
        a, b = S.VisitorSession(), S.VisitorSession(); upload_both(env, b); env.import_saved(b, J(make_doc("GOOD"))); env.accept(b)
        a.reset(env.store); assert env.slots_ready(b) and b.accepted is not None and not b.f6.closed

    def test_new_import_replaces_the_previous_acceptance_candidate_only_at_review(self, env):
        sess = S.VisitorSession(); env.import_saved(sess, J(make_doc("GOOD"))); env.accept(sess)
        env.import_saved(sess, J(make_doc("FAILED")))
        with pytest.raises(S.ServiceError): env.accept(sess, True)


# ===================================================================== pages and entry point (Streamlit AppTest; no browser, no server)
streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def page_app(svc, sess=None):
    from streamlit.testing.v1 import AppTest
    src = ("def run():\n    import sys as _s\n    import streamlit as st\n    for p in %r:\n        if p not in _s.path: _s.path.insert(0, p)\n"
           "    import f8_visitor_pages as P\n    P.render(st.session_state['_svc'], P.session_of(st.session_state), st.session_state)\nrun()\n") % ((str(MODULE), str(WF), str(F8), str(ROOT)),)
    at = AppTest.from_string(src, default_timeout=60)
    at.session_state["_svc"] = svc
    if sess is not None: at.session_state["f8i_session"] = sess
    return at


def texts(at):
    return " ".join([m.value for m in at.markdown] + [e.value for e in at.error] + [e.value for e in at.warning] + [e.value for e in at.success] + [e.value for e in at.info] + [c.value for c in at.caption])


def button(at, label_part):
    for b in at.button:
        if label_part in b.label: return b
    raise AssertionError(f"no button containing {label_part!r}: {[b.label for b in at.button]}")


class TestPages:
    def test_home_offers_both_paths_and_starts_nothing(self, env):
        at = page_app(env); at.run()
        assert not at.exception and button(at, "Calibrate My Cameras") is not None and button(at, "Upload Saved Calibration") is not None       # the home page now offers the two paths as buttons
        for _ in range(3): at.run()
        assert FakeRunner.instances == [] and "f8i_session" in at.session_state

    def test_namespace_is_isolated(self, env):
        at = page_app(env); at.run()
        own = {k for k in list(at.session_state) if not str(k).startswith("_")}
        assert own and all(str(k).startswith("f8i_") for k in own), own

    def test_rerun_with_ready_videos_does_not_start_a_job(self, env):
        sess = S.VisitorSession(); upload_both(env, sess); sess.step = "setup"; sess.guide.confirmed.update({1: True, 2: True, 3: True})   # F8.31D-3B: steps 1-3 confirmed
        at = page_app(env, sess); at.run(); at.run(); at.run()
        assert FakeRunner.instances == [] and not at.exception
        assert button(at, "Start calibration").disabled                              # a click is the only trigger

    def test_start_button_is_disabled_until_acknowledged_and_starts_exactly_once(self, env):
        sess = S.VisitorSession(); upload_both(env, sess); sess.step = "setup"; sess.guide.confirmed.update({1: True, 2: True, 3: True})   # F8.31D-3B: steps 1-3 confirmed
        at = page_app(env, sess); at.run()
        assert button(at, "Start calibration").disabled
        at.checkbox(key="f8i_sure").check().run()
        assert not button(at, "Start calibration").disabled and FakeRunner.instances == []
        FakeRunner.finish_on_start = make_doc("GOOD")
        button(at, "Start calibration").click().run()
        assert len(FakeRunner.instances) == 1 and FakeRunner.instances[0].started
        at.run(); at.run(); assert len(FakeRunner.instances) == 1                  # reruns after the start never start a second job

    def test_full_calibration_disabled_server_blocks_start(self, env):
        sess = S.VisitorSession(); upload_both(env, sess); sess.step = "setup"; sess.guide.confirmed.update({1: True, 2: True, 3: True})   # F8.31D-3B: steps 1-3 confirmed
        off = S.VisitorCalibrationService(env.store, env.results, CR.RunnerLimits(allow_full=False), runner_cls=FakeRunner)
        at = page_app(off, sess); at.run(); at.checkbox(key="f8i_sure").check().run()
        assert button(at, "Start calibration").disabled and "switched off on this server" in texts(at)

    def test_review_good_accept_ready_confirmation_and_room_disabled(self, env):
        sess = S.VisitorSession(); env.import_saved(sess, J(make_doc("GOOD")))
        at = page_app(env, sess); at.run()
        assert "Good" in texts(at) and "226 mm (measured by you)" in " ".join(str(t.value) for t in at.table)
        button(at, "Accept this calibration").click().run()
        assert sess.accepted is not None and sess.accepted.can_proceed
        t = texts(at); assert "ready for room measurement" in t
        assert S.ROOM_PROCESSING_ENABLED is False
        room = button(at, "Continue to Room Measurement"); assert not room.disabled        # F8.31A: the step now exists, but it only uploads and checks video metadata
        assert sess.f6.measurement is None and not sess.f6.reviews

    def test_review_questionable_checkbox_gate(self, env):
        sess = S.VisitorSession(); env.import_saved(sess, J(make_doc("QUESTIONABLE", concerns=["Only 9 poses."])))
        at = page_app(env, sess); at.run()
        assert "Only 9 poses." in texts(at) and button(at, "Accept this calibration").disabled
        at.checkbox[0].check().run(); assert not button(at, "Accept this calibration").disabled
        button(at, "Accept this calibration").click().run(); assert sess.accepted is not None

    def test_review_failed_has_no_way_to_accept(self, env):
        sess = S.VisitorSession(); env.import_saved(sess, J(make_doc("FAILED")))
        at = page_app(env, sess); at.run()
        assert "cannot be accepted" in texts(at) and button(at, "Accept this calibration").disabled and not at.checkbox
        assert sess.accepted is None

    def test_not_assessed_file_is_labelled_and_needs_acknowledgement(self, env):
        sess = S.VisitorSession(); env.import_saved(sess, J(bare_contract()))
        at = page_app(env, sess); at.run()
        assert "no quality information" in texts(at) and "Quality not assessed" in texts(at) and button(at, "Accept this calibration").disabled
        at.checkbox[0].check().run(); button(at, "Accept this calibration").click().run(); assert sess.accepted is not None

    def test_invalid_file_shows_reasons(self, env):
        sess = S.VisitorSession(); env.import_saved(sess, b"not json")
        at = page_app(env, sess); at.run()
        assert "not valid JSON" in texts(at) and button(at, "Accept this calibration").disabled

    def test_start_over_button_clears_the_session(self, env):
        sess = S.VisitorSession(); env.import_saved(sess, J(make_doc("GOOD"))); env.accept(sess)
        sess.step = "setup"                                                # Start over / Lock sit on the calibration workflow pages, not on the home page
        at = page_app(env, sess); at.run(); button(at, "Start over").click().run()
        assert sess.accepted is None and sess.f6 is None and sess.step == "home" and button(at, "Calibrate My Cameras") is not None

    def test_lock_clears_the_whole_namespace(self, env):
        sess = S.VisitorSession(); env.import_saved(sess, J(make_doc("GOOD"))); env.accept(sess); sess.step = "setup"                              # the Lock button is on the calibration workflow pages (and in the measurement page expander), not on the home page
        at = page_app(env, sess); at.run(); button(at, "Lock this session").click().run()
        fresh = at.session_state["f8i_session"]                      # the harness has no gate, so the page re-creates an EMPTY session; the entry-script tests cover the gate
        assert fresh is not sess and fresh.accepted is None and fresh.f6 is None and fresh.step == "home" and fresh.owner != sess.owner
        assert at.session_state["f8i_authed"] is False and sess.f6 is None and sess.accepted is None and sess.contract is None

    def test_recovery_panel_reviews_a_saved_result(self, env):
        code = new_code(); env.results.save_result(code, make_doc("GOOD", code=code))
        sess = S.VisitorSession(); sess.step = "setup"                      # the recovery-code box lives on the calibration setup page
        at = page_app(env, sess); at.run(); at.text_input(key="f8i_rec_code").input(code).run()
        assert "Saved result found" in texts(at)
        button(at, "Review this result").click().run()
        assert at.session_state["f8i_session"].step == "review" and "Good" in texts(at)

    def test_setup_page_shows_finished_job_review_button(self, env):
        sess = S.VisitorSession(); run_job(env, sess); sess.step = "setup"
        at = page_app(env, sess); at.run()
        assert "has finished" in texts(at) and not button(at, "Review the calibration").disabled
        assert sess.job_code in " ".join(c.value for c in at.code)       # shown as page text to this session only
        button(at, "Review the calibration").click().run(); assert sess.step == "review"


# ===================================================================== unauthorized access (the real entry script)
@pytest.fixture
def entry(tmp_path, monkeypatch):
    import streamlit as st
    import chunk_store
    st.cache_resource.clear()
    made = []

    def factory(limits=None, **kw):
        s = ChunkStore(Limits(min_chunk_bytes=1024, min_free_bytes=1, min_free_fraction=0.0), root=tmp_path / f"c{len(made)}"); made.append(s); return s
    monkeypatch.setattr(chunk_store, "ChunkStore", factory)
    yield made
    for s in made: s.close()
    st.cache_resource.clear()


def app_at():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(APP), default_timeout=60)


class TestUnauthorized:
    def test_no_passcode_configured_fails_closed(self, entry):
        at = app_at(); at.run()
        assert any("locked" in e.value for e in at.error) and not at.text_input and not entry
        assert "f8i_session" not in at.session_state

    def test_short_passcode_fails_closed(self, entry, monkeypatch):
        monkeypatch.setenv("F8_PASSCODE", "short")
        at = app_at(); at.run(); assert any("locked" in e.value for e in at.error) and not entry

    def test_gate_shown_and_nothing_behind_it(self, entry, monkeypatch):
        monkeypatch.setenv("F8_PASSCODE", "correct-horse-battery")
        at = app_at(); at.run()
        assert len(at.text_input) == 1 and at.text_input[0].label == "Passcode" and not at.exception
        assert "f8i_session" not in at.session_state and not entry and not at.button[1:] and "Calibrate now" not in texts(at)

    def test_wrong_passcode_is_refused_and_rate_limited(self, entry, monkeypatch):
        monkeypatch.setenv("F8_PASSCODE", "correct-horse-battery")
        at = app_at(); at.run()
        for _ in range(6):
            at.text_input[0].input("wrong-guess").run(); at.button[0].click().run()
        msgs = " ".join(e.value for e in at.error)
        assert "Incorrect passcode" in msgs or "Too many" in msgs
        assert "f8i_session" not in at.session_state and not entry and not (at.session_state["f8i_authed"] if "f8i_authed" in at.session_state else False)
        assert "wrong-guess" not in texts(at) and "correct-horse-battery" not in texts(at)

    def test_correct_passcode_unlocks_the_home_page_and_room_processing_stays_off(self, entry, monkeypatch, tmp_path):
        monkeypatch.setenv("F8_PASSCODE", "correct-horse-battery")
        at = app_at(); at.run()
        at.text_input[0].input("correct-horse-battery").run(); at.button[0].click().run()
        assert not at.exception and at.session_state["f8i_authed"] is True and button(at, "Calibrate My Cameras") is not None
        assert "correct-horse-battery" not in texts(at)
        s = at.session_state["f8i_session"]; assert s.f6 is None and s.accepted is None and S.ROOM_PROCESSING_ENABLED is False and len(entry) == 2       # F8.31A: calibration store + separate room-video store
        assert not any(b.label == "Select room videos" and not b.disabled for b in at.button)

    def test_lock_returns_to_the_gate(self, entry, monkeypatch):
        monkeypatch.setenv("F8_PASSCODE", "correct-horse-battery")
        at = app_at(); at.run(); at.text_input[0].input("correct-horse-battery").run(); at.button[0].click().run()
        at.session_state["f8i_session"].step = "setup"; at.run()           # Lock is on the calibration workflow pages, not on the home page
        button(at, "Lock this session").click().run()
        assert len(at.text_input) == 1 and at.text_input[0].label == "Passcode" and not any(b.label == "Calibrate My Cameras" for b in at.button)

    def test_session_refresh_requires_the_passcode_again(self, entry, monkeypatch):
        monkeypatch.setenv("F8_PASSCODE", "correct-horse-battery")
        a = app_at(); a.run(); a.text_input[0].input("correct-horse-battery").run(); a.button[0].click().run()
        b = app_at(); b.run()                                         # a refresh is a NEW Streamlit session: it must not inherit the authentication
        assert len(b.text_input) == 1 and "f8i_session" not in b.session_state


# ===================================================================== room processing stays disabled; structural guarantees
FORBIDDEN_CALLS = {"submit_videos", "select_frames", "measure", "detect_boundaries", "select_candidate", "confirm", "use_manual", "export_for_measurement", "load_gray", "VideoCapture",
                   "validate_videos", "inspect_video"}


class TestRoomProcessingDisabled:
    @pytest.mark.parametrize("name", ["f8_visitor_service.py", "f8_visitor_pages.py", "app_f8_integrated.py"])
    def test_no_room_step_or_video_decoding_is_called(self, name):
        tree = ast.parse((WF / name).read_text())
        called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "") for n in ast.walk(tree) if isinstance(n, ast.Call)}
        allowed = {"inspect_video"} if name == "f8_visitor_service.py" else set()     # F8.31A: ONLY the sandboxed metadata inspector (separate process), in the service only
        assert not (called & FORBIDDEN_CALLS) - allowed, (called & FORBIDDEN_CALLS) - allowed
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not any(m and ("cv2" in m or ("video_check" in m and name != "f8_visitor_service.py") or "visitor_measurement" in m or "visitor_frame_selection" in m or m.startswith("stage_")) for m in imported), imported

    @pytest.mark.parametrize("name", ["f8_visitor_service.py", "f8_visitor_pages.py", "app_f8_integrated.py"])
    def test_no_original_recording_or_fixture_is_referenced(self, name):
        src = (WF / name).read_text()
        for bad in ("calibration_videos", "DEFAULT_CALIBRATION_PATH", ".MOV", "FIX_VIDEO", "F6_SHOW_FIXTURES", "_fixtures_visible", "data/room_measurement", "outputs/"):
            assert bad not in src, bad

    def test_room_processing_cannot_be_enabled_from_the_environment(self, env, monkeypatch):
        for k in ("F6_ROOM_PROCESSING", "F8_ROOM_PROCESSING", "ROOM_PROCESSING_ENABLED"): monkeypatch.setenv(k, "1")
        assert S.ROOM_PROCESSING_ENABLED is False and env.room_processing_enabled() is False

    def test_no_file_uploader_for_room_videos(self):
        src = (WF / "f8_visitor_pages.py").read_text()
        assert 'type=["json"]' in src and src.count("st.file_uploader(") == 1        # F8.31A: room videos use the chunked uploader (bounded, on disk), never st.file_uploader

    def test_existing_sources_are_unchanged(self):
        out = subprocess.run(["git", "status", "--porcelain", "--", "Modules", "Pages", "home.py", "home_navigation.py", "tests", "requirements.txt"], cwd=ROOT, capture_output=True, text=True).stdout
        changed = [l for l in out.splitlines() if not l.startswith("??")]
        assert changed == [], changed

    def test_original_f8_dual_page_and_app_f6_are_not_imported_by_the_integration(self):
        for name in ("f8_visitor_service.py", "f8_visitor_pages.py", "app_f8_integrated.py"):
            src = (WF / name).read_text()
            assert "import dual_page" not in src and "import app_f6" not in src and "from app_f6" not in src and "page_nav" not in src

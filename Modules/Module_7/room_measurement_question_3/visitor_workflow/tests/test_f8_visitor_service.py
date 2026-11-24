"""F8.20a regression tests for the integrated visitor service (Option A / Option B -> review -> accept -> F6 hand-off).

SYNTHETIC only: hand-built result documents and a stub runner. No video is opened, no original/derived frames or regression datasets are read, and no calibration is run."""
import ast
import re
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import f8_visitor_service as S
from calibration_runner import RunnerLimits
from chunk_store import ChunkStore, Limits, MiB
from result_store import ResultStore, job_id, new_code

MODULE = Path(__file__).resolve().parents[2]
K = [[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]]


def make_doc(level="GOOD", concerns=(), code=None):
    code = code or new_code()
    d = dict(schema="f8.calibration_result/1", job_id=job_id(code), mode="calibrate", outcome="completed", ok=True, settings=dict(cols=8, rows=6, square_mm=20, measured_baseline_mm=None),
             quality=dict(level=level, headline="synthetic", problems=[], concerns=list(concerns), stereo_rms_px=0.7),
             cameras=dict(left=dict(K=copy.deepcopy(K), image_size_wh=[1920, 1080]), right=dict(K=copy.deepcopy(K), image_size_wh=[1920, 1080])),
             extrinsics=dict(baseline_mm=226.0),
             contract=dict(K1=copy.deepcopy(K), D1=[0.0] * 5, K2=copy.deepcopy(K), D2=[0.0] * 5, image_size_wh=[1920, 1080], capture_mode=dict(left="video", right="video"), baseline_mm=226.0))
    return code, d


@pytest.fixture
def store(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB), root=tmp_path / "chunks"); yield s; s.close()


@pytest.fixture
def results(tmp_path): return ResultStore(tmp_path / "results")


@pytest.fixture
def svc(store, results): return S.VisitorCalibrationService(store, results, RunnerLimits())


class Stub:
    """Stands in for CalibrationJobRunner: never starts a process."""
    instances = []

    def __init__(self, *a, **k): self.done = False; self.cancelled = False; self.args = (a, k); Stub.instances.append(self)
    def start(self): pass
    def cancel(self): self.cancelled = True; self.done = True
    def status(self): return dict(stage="x", fraction=0.1, elapsed_s=1.0)


@pytest.fixture
def svc_on(store, results):
    Stub.instances = []
    s = S.VisitorCalibrationService(store, results, RunnerLimits(allow_full=True), runner_cls=Stub)
    return s


def put(store, owner, slot, data=None):
    data = data or (slot.encode() * 1500)                                     # the two slots must differ
    h = hashlib.sha256(data).hexdigest()
    uid = store.begin(owner, "c.mov", len(data), MiB, slot + h[:6], slot=slot)["upload_id"]
    store.put_chunk(owner, uid, 0, data, h); store.finish(owner, uid, h)


SETUP = S.parse_setup(8, 6, 20, 226)


# ---------------------------------------------------------------- start restrictions
@pytest.mark.parametrize("args", [(8, 8, 20, 0), (8, 6, 0.5, 0), (8, 6, 600, 0), (8, 6, 20, 5), (8, 6, 20, 5000), (1, 6, 20, 0), (8, 41, 20, 0), (8.5, 6, 20, 0), (True, 6, 20, 0), ("8", 6, 20, 0)])
def test_parse_setup_rejects_bad_input(args):
    with pytest.raises(S.ServiceError):
        S.parse_setup(*args)


def test_parse_setup_baseline_optional():
    assert S.parse_setup(8, 6, 20, None).baseline_mm is None and S.parse_setup(8, 6, 20, 0).baseline_mm is None and S.parse_setup(8, 6, 20, 226).baseline_mm == 226.0


def test_start_refused_while_full_calibration_is_off(svc):
    s = S.VisitorSession(); put(svc.store, s.owner, "left"); put(svc.store, s.owner, "right")
    assert svc.full_calibration_available is False
    with pytest.raises(S.ServiceError, match="not enabled"):
        svc.start_calibration(s, SETUP)
    assert s.runner is None


def test_default_limits_and_env_keep_full_calibration_off():
    assert RunnerLimits().allow_full is False and RunnerLimits.from_env({}).allow_full is False
    assert RunnerLimits.from_env({"F8_ENABLE_FULL_CALIBRATION": "true"}).allow_full is False      # only the exact value '1'


def test_start_refused_until_both_videos_are_uploaded(svc_on):
    s = S.VisitorSession()
    with pytest.raises(S.ServiceError, match="LEFT and the RIGHT"):
        svc_on.start_calibration(s, SETUP)
    put(svc_on.store, s.owner, "left")
    with pytest.raises(S.ServiceError, match="LEFT and the RIGHT"):
        svc_on.start_calibration(s, SETUP)
    assert Stub.instances == []


def test_start_once_per_session_sets_code_first_and_blocks_navigation(svc_on):
    s = S.VisitorSession(); put(svc_on.store, s.owner, "left"); put(svc_on.store, s.owner, "right")
    code = svc_on.start_calibration(s, SETUP)
    assert len(Stub.instances) == 1 and Stub.instances[0].args[1]["job_code"] == code and Stub.instances[0].args[1]["mode"] == "calibrate"
    with pytest.raises(S.ServiceError, match="already been started"):
        svc_on.start_calibration(s, SETUP)
    with pytest.raises(S.ServiceError, match="running"):
        s.go("home")
    assert len(Stub.instances) == 1


def test_cancel_stops_the_runner(svc_on):
    s = S.VisitorSession(); put(svc_on.store, s.owner, "left"); put(svc_on.store, s.owner, "right")
    svc_on.start_calibration(s, SETUP)
    assert svc_on.cancel(s) is True and Stub.instances[0].cancelled and svc_on.cancel(s) is False


# ---------------------------------------------------------------- GOOD / QUESTIONABLE / FAILED and the F6 hand-off
def test_good_is_accepted_and_f6_validates_it(svc):
    code, d = make_doc("GOOD"); s = S.VisitorSession(); s.review = S.review_f8_document(d)
    assert s.review.level == S.GOOD and not s.review.requires_acknowledgement and s.review.can_accept(False)
    r = svc.accept(s)
    assert r.ok and r.can_proceed and r.baseline_used_mm == 226.0 and r.image_size_wh == (1920, 1080) and s.step == "ready"
    assert s.f6.calibration_provenance["source"] == "f8_cloud_calibration" and s.f6.calibration_provenance["baseline_used_mm"] == 226.0
    assert json.loads(svc.contract_json(s))["baseline_mm"] == 226.0


def test_questionable_needs_acknowledgement(svc):
    _, d = make_doc("QUESTIONABLE", concerns=["synthetic concern"]); s = S.VisitorSession(); s.review = S.review_f8_document(d)
    assert s.review.requires_acknowledgement and not s.review.can_accept(False) and s.review.can_accept(True)
    with pytest.raises(S.ServiceError):
        svc.accept(s, acknowledged=False)
    assert s.accepted is None and s.step == "home"
    assert svc.accept(s, acknowledged=True).ok


def test_good_with_concerns_is_shown_as_questionable(svc):
    _, d = make_doc("GOOD", concerns=["x"]); assert S.review_f8_document(d).level == S.QUESTIONABLE


def test_failed_is_never_accepted(svc):
    _, d = make_doc("FAILED"); s = S.VisitorSession(); s.review = S.review_f8_document(d)
    assert s.review.level == S.FAILED and not s.review.usable and not s.review.can_accept(True)
    with pytest.raises(S.ServiceError):
        svc.accept(s, acknowledged=True)
    assert s.accepted is None and (s.f6 is None or s.f6.calibration_provenance is None)


def test_accept_without_review_is_refused(svc):
    with pytest.raises(S.ServiceError, match="no calibration"):
        svc.accept(S.VisitorSession())


@pytest.mark.parametrize("mutate", [lambda d: d.update(outcome="timeout", ok=False), lambda d: d.update(mode="benchmark"), lambda d: d.update(schema="other/1"),
                                    lambda d: d["quality"].update(level="EXCELLENT"), lambda d: d["contract"].pop("capture_mode"), lambda d: d["contract"].update(baseline_mm=-1),
                                    lambda d: d["contract"]["K1"].pop()])
def test_unusable_documents_are_never_acceptable(svc, mutate):
    _, d = make_doc("GOOD"); mutate(d); rev = S.review_f8_document(d)
    assert not rev.usable and not rev.can_accept(True)


def test_tampering_with_one_copy_of_the_camera_matrix_is_detected():
    _, d = make_doc("GOOD")
    assert d["cameras"]["left"]["K"] is not d["contract"]["K1"]               # two independent copies
    d["cameras"]["left"]["K"][0][0] = 1234.0                                   # modify ONLY the result's camera-section copy
    rev = S.review_f8_document(d)
    assert rev.level == S.INVALID and any("differs" in p for p in rev.problems)
    _, d2 = make_doc("GOOD"); d2["contract"]["K1"][0][0] = 1234.0              # and ONLY the contract copy
    assert S.review_f8_document(d2).level == S.INVALID


def test_measured_baseline_override_and_disagreement_concern():
    _, d = make_doc("GOOD")
    rev = S.review_f8_document(d, measured_baseline_mm=300.0)
    assert rev.baseline_used_mm == 300.0 and rev.baseline_source == "measured" and rev.level == S.QUESTIONABLE and any("differs" in c for c in rev.concerns)


def test_option_b_file_is_never_labelled_good(svc):
    code, d = make_doc("GOOD"); s = S.VisitorSession(); s.review = S.review_f8_document(d); svc.accept(s)
    rev = svc.import_saved(S.VisitorSession(), svc.contract_json(s))
    assert rev.level == S.NOT_ASSESSED and rev.requires_acknowledgement and not rev.quality_assessed_by_f8
    assert svc.import_saved(S.VisitorSession(), b"not json").level == S.INVALID and svc.import_saved(S.VisitorSession(), b"[1]").level == S.INVALID


# ---------------------------------------------------------------- room processing stays disabled
def test_room_processing_is_disabled(svc):
    assert S.ROOM_PROCESSING_ENABLED is False and svc.room_processing_enabled() is False


@pytest.mark.parametrize("fname", ["f8_visitor_service.py", "f8_visitor_pages.py", "app_f8_integrated.py"])
def test_no_room_step_is_ever_called(fname):
    tree = ast.parse((MODULE / "visitor_workflow" / fname).read_text())
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "") for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not called & {"submit_videos", "select_frames", "measure", "triangulate", "detect_boundaries"}, called


def test_no_measure_button_exists_in_the_page_source():
    """F8.31A replaced the disabled placeholder with the room-video upload; there must still be no way to start a measurement."""
    src = (MODULE / "visitor_workflow" / "f8_visitor_pages.py").read_text()
    assert "f8i_room_disabled" not in src and 'key="f8i_to_frames"' in src
    assert not re.search(r"st\.button\([^)]*Measure Room", src, re.I)


# ---------------------------------------------------------------- recovery and reset
def test_recovery_states(svc, results):
    assert svc.recover("abc")["state"] == "invalid" and svc.recover(None)["state"] == "invalid"
    assert svc.recover(new_code())["state"] == "not_found"
    code, d = make_doc("GOOD"); results.save_result(code, d)
    info = svc.recover(code); assert info["state"] == "result" and info["review"].level == S.GOOD
    c2 = new_code(); results.write_status(c2, dict(state="running")); assert svc.recover(c2)["state"] == "interrupted"
    c3, d3 = make_doc("GOOD"); d3["mode"] = "benchmark"; results.save_result(c3, d3); assert svc.recover(c3)["state"] == "not_found"


def test_recovered_result_can_be_used_and_accepted(svc, results):
    code, d = make_doc("GOOD"); results.save_result(code, d); s = S.VisitorSession()
    rev = svc.use_recovered(s, code)
    assert s.step == "review" and rev.kind == "f8_recovered" and svc.result_bytes(s) is not None
    assert svc.accept(s).ok
    with pytest.raises(S.ServiceError):
        svc.use_recovered(S.VisitorSession(), new_code())


def test_reset_deletes_uploads_cancels_job_and_forgets_everything(svc_on):
    s = S.VisitorSession(); put(svc_on.store, s.owner, "left"); put(svc_on.store, s.owner, "right")
    svc_on.start_calibration(s, SETUP); old_owner, ep = s.owner, s.epoch
    s.f6_session()
    s.reset(svc_on.store)
    assert Stub.instances[0].cancelled and s.runner is None and s.job_code is None and s.review is None and s.accepted is None and s.f6 is None and s.step == "home"
    assert s.owner != old_owner and s.epoch == ep + 1
    assert svc_on.store.job_status(old_owner) == dict(slots={}, lease=None) and list(Path(svc_on.store.root).rglob("*.part")) == []


# ---------------------------------------------------------------- launcher (never starts a server: refusals happen before streamlit is spawned)
def run_launcher(*args):
    return subprocess.run([sys.executable, "-B", str(MODULE / "run_f8_integrated_app.py"), "8691", *args], capture_output=True, text=True, timeout=30, env={"PATH": "/usr/bin:/bin"})


def test_launcher_requires_an_explicit_memory_limit_for_full_calibration():
    r = run_launcher("--enable-full-calibration")
    assert r.returncode != 0 and "requires an explicit memory limit" in (r.stderr + r.stdout)


@pytest.mark.parametrize("val", ["abc", "100", "999999", "nan"])
def test_launcher_rejects_bad_limits(val):
    r = run_launcher("--enable-full-calibration", f"--max-tree-rss-mb={val}")
    assert r.returncode != 0 and "--max-tree-rss-mb" in (r.stderr + r.stdout)


def test_launcher_limit_parser(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_f8_integrated_app.py"]); monkeypatch.syspath_prepend(str(MODULE))
    import run_f8_integrated_app as L
    assert L.parse_rss_limit(["--max-tree-rss-mb=3000"]) == (3000.0, None) and L.parse_rss_limit([]) == (None, None)
    assert L.parse_rss_limit(["--max-tree-rss-mb=1"])[1] and L.parse_rss_limit(["--max-tree-rss-mb=x"])[1]

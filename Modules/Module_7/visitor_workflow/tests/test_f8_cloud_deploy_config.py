"""Isolated Cloud test deployment: the lean requirements file, the entry point and the committed files must be enough for the visitor app.
Static checks only (AST of the source, the committed calibration file); nothing is decoded, nothing is processed, the gate stays closed."""
import ast
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
MODULE = HERE.parent
ROOT = MODULE.parents[1]
ENTRY = HERE / "app_f8_integrated.py"
REQ = HERE / "requirements.txt"
DIST = {"PIL": "pillow", "cv2": "opencv-python-headless", "streamlit": "streamlit", "numpy": "numpy", "streamlit_image_coordinates": "streamlit-image-coordinates"}
FORBIDDEN = ("torch", "torchvision", "tensorflow", "keras", "sam-2", "sam2", "onnx", "ultralytics", "transformers", "hydra", "omegaconf", "iopath")
LEGACY = {"app_f6", "f7_pages", "page_nav", "browser_walkthrough_f6", "browser_walkthrough_f7", "dual_page"}


def _pins(path):
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, _, ver = line.partition("==")
            out[name.strip().lower()] = ver.strip()
    return out


def _local_modules():
    mods = {}
    for d in ("", "visitor_workflow", "room_measurement", "f8_chunk_poc", "stereo_measure", "annotation_ui"):
        for p in (MODULE / d).glob("*.py"):
            mods.setdefault(p.stem, []).append(p)
        for p in (MODULE / d).glob("*/__init__.py"):
            mods.setdefault(p.parent.name, []).append(p)
    mods.setdefault("home_navigation", []).append(ROOT / "home_navigation.py")
    mods["annotation_ui"] = [MODULE / "annotation_ui" / "core.py"]                              # a namespace package: only core.py is imported
    return mods


def _closure():
    """Every first-party file the visitor app can import (plus the worker scripts it launches by path), and every third-party top-level import found in them."""
    mods, seen, third = _local_modules(), set(), {}
    queue = [ENTRY] + [HERE / n for n in ("calibration_worker.py",)] + [MODULE / "f8_chunk_poc" / n for n in ("calibration_worker.py",)] \
        + [HERE / n for n in ("reconstruction_worker.py", "three_view_worker.py", "frame_worker.py", "f7_calibration.py")]
    while queue:
        p = queue.pop()
        if p in seen or not p.exists():
            continue
        seen.add(p)
        for n in ast.walk(ast.parse(p.read_text())):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else []
            for full in names:
                top = full.split(".")[0]
                if top in mods:
                    queue += [x for x in mods[top] if x not in seen]
                elif top not in sys.stdlib_module_names:
                    third.setdefault(top, set()).add(p.name)
    return seen, third


def test_requirements_are_pinned_lean_and_match_the_other_requirement_files():
    pins = _pins(REQ)
    assert set(DIST.values()) <= set(pins) and set(pins) - set(DIST.values()) <= {"pyarrow", "pandas"}      # the five imported packages, plus Streamlit's pinned data libraries
    assert all(re.fullmatch(r"\d+(\.\d+)*", v) for v in pins.values())
    text = REQ.read_text().lower()
    for bad in FORBIDDEN:
        assert not re.search(rf"^\s*{re.escape(bad)}\b", text, re.M), bad                         # classical CV only
    for other in (ROOT / "requirements.txt", MODULE / "f8_chunk_poc" / "requirements.txt"):
        theirs = _pins(other)
        for name, ver in pins.items():
            assert theirs.get(name, ver) == ver, (other.name, name)                               # the same pins that are already used elsewhere


def test_every_third_party_import_of_the_visitor_app_is_in_the_requirements():
    seen, third = _closure()
    assert not set(third) - set(DIST), {k: v for k, v in third.items() if k not in DIST}          # an unlisted package would crash the app on Cloud
    assert set(third) <= set(DIST) and {"streamlit", "cv2", "numpy"} <= set(third)
    assert not [p for p in seen if p.stem in LEGACY], [p.name for p in seen if p.stem in LEGACY]  # the F6/F7 legacy pages are not part of the visitor app
    assert not [p for p in seen if {"outputs", "data", "tests"} & set(p.parts)]                   # no generated output, recording folder or test file is needed at run time


def test_entry_point_is_the_gated_app_and_the_gate_is_closed():
    src = ENTRY.read_text()
    assert "load_passcode" in src and "st.stop()" in src and "F8_PASSCODE" in src                 # fail closed without the secret
    assert re.search(r"^ROOM_PROCESSING_ENABLED = False\b", (HERE / "f8_visitor_service.py").read_text(), re.M)
    assert ENTRY.relative_to(ROOT).as_posix() == "Modules/Module_7/visitor_workflow/app_f8_integrated.py"
    assert (ROOT / "Modules/Module_7/f8_chunk_poc/frontend/index.html").exists()                  # the chunk-upload component's page


def test_the_validated_calibration_file_can_be_imported_with_capture_mode_video():
    import f8_visitor_service as S
    raw = (MODULE / "calibration/room_video/room_video_recipe_a_calibration.json").read_bytes()
    rev = S.review_saved_file(raw, capture_mode="video")
    assert rev.level == S.NOT_ASSESSED and not rev.problems and rev.requires_acknowledgement      # usable, but F8 does not claim a quality verdict
    assert S.review_saved_file(raw).level == S.INVALID                                            # the capture mode is never guessed


def test_the_theme_is_configured_per_app_next_to_the_entry_script_and_matches_the_local_launcher():
    import tomllib
    cfg = HERE / ".streamlit" / "config.toml"; assert cfg.parent.parent == ENTRY.parent               # Streamlit reads <main script dir>/.streamlit/config.toml: the root and the main app are untouched
    d = tomllib.loads(cfg.read_text())
    assert d["theme"] == {"base": "dark", "primaryColor": "#C9363E"} and d["browser"]["gatherUsageStats"] is False
    launcher = (MODULE / "run_f8_integrated_app.py").read_text()
    assert '"--theme.base", "dark", "--theme.primaryColor", "#C9363E"' in launcher                    # the same values the local launcher passes
    import streamlit.config as sc                                                                       # and Streamlit itself resolves them for this main script
    old = sc._main_script_path
    try:
        sc._main_script_path = str(ENTRY); sc.get_config_options(force_reparse=True)
        assert sc.get_option("theme.base") == "dark" and sc.get_option("theme.primaryColor") == "#C9363E"
    finally:
        sc._main_script_path = old; sc.get_config_options(force_reparse=True)
    assert "secrets" not in {p.name.split(".")[0] for p in cfg.parent.iterdir() if p.is_file() and p.suffix == ".toml"}   # no secrets file is part of the checkout


def test_the_upload_limit_text_says_the_storage_is_one_pool_shared_by_both_walls():
    import f8_ui_text as UT
    import room_videos as RV
    from chunk_store import MiB
    lim = RV.room_limits()
    for n, vps in ((4, [(1, None), (2, None)]), (6, [(1, "left end"), (2, "middle"), (3, "right end")])):
        t = "\n".join(UT.upload_requirements(n_videos=n, viewpoints=vps, per_file_mb=lim.max_file_bytes / MiB, store_total_mb=lim.max_disk_bytes / MiB,
                                             min_duration_s=RV.F6_LIMITS.min_duration_s, max_duration_s=RV.F6_LIMITS.max_duration_s, min_fps=RV.F6_LIMITS.min_fps, max_fps=RV.F6_LIMITS.max_fps,
                                             fps_tolerance=RV.FPS_MATCH_TOLERANCE, validated_wh=RV.VALIDATED_FRAME_SIZES[0]))
        assert "one pool for both walls" in t and "Wall A and of Wall B are stored together" in t and "not allocated separately to each wall" in t
        assert "1,024 MB" in t and "4,096 MB" in t                                                       # the enforced numbers are unchanged
    assert lim.max_file_bytes == 1024 * MiB and lim.max_disk_bytes == 4096 * MiB

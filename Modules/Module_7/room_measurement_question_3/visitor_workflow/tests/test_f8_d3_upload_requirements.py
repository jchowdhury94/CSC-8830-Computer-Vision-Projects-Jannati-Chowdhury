"""Phase D.3: the recording / upload requirements shown beside the room-video uploaders must be exactly what the server enforces.
Synthetic stores, fake inspections and generated headers only.  Nothing is decoded, nothing is processed, the gate stays closed."""
import re
from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import f8_ui_text as UT
import f8_viewpoints as VPC
import room_videos as RV
from chunk_store import ChunkStore, UploadError, MiB
from test_f8_31a_room_videos import SLOTS, _paths, good_info                                       # noqa: F401  (autouse fixture and helper)
from video_check import sniff_container

HERE = Path(__file__).resolve().parents[1]
MODULE = HERE.parent
CHUNK = 64 * 1024


def requirements(n_viewpoints, lim=None):
    """Exactly what the page passes: every number comes from the objects that enforce it."""
    lim = lim or RV.room_limits(); nvp = n_viewpoints; nv = len(RV.slots_for(nvp))
    return "\n".join(UT.upload_requirements(
        n_videos=nv, viewpoints=[(vp, VPC.ROLE[vp] if nvp == 3 else None) for vp in range(1, nvp + 1)], per_file_mb=lim.max_file_bytes / MiB, store_total_mb=lim.max_disk_bytes / MiB,
        min_duration_s=RV.F6_LIMITS.min_duration_s, max_duration_s=RV.F6_LIMITS.max_duration_s, min_fps=RV.F6_LIMITS.min_fps, max_fps=RV.F6_LIMITS.max_fps,
        fps_tolerance=RV.FPS_MATCH_TOLERANCE, validated_wh=RV.VALIDATED_FRAME_SIZES[0]))


# ================================================================ the text states the verified limits, for six and for four videos
def test_the_six_video_requirements_state_every_verified_limit():
    t = requirements(3)
    assert t.startswith("**Recording and upload requirements**") and "Six in total" in t
    for v, role in ((1, "left end"), (2, "middle"), (3, "right end")):
        assert f"**V{v}** ({role})" in t                                                             # grouped by V1, V2, V3
    assert "one video from the **left** camera and one from the **right** camera" in t
    assert "1,024 MB" in t and "4,096 MB" in t and "shared by all visitors" in t and "no separate per-visitor limit" in t and "One upload runs at a time" in t
    assert "1–120 seconds" in t and "1–480 frames per second" in t and "more than 1 fps" in t
    assert "2160 × 3840" in t and "exactly the frame size and orientation of your accepted calibration" in t and "never resized or rotated" in t
    assert "Not yet verified" in t and "H.265/HEVC" in t and "not a promise of processing time" in t   # what is NOT proven is labelled as such
    assert "683 MB" in t                                                                              # 4096 / 6: only arithmetic on the real budget, not a new limit


def test_the_four_video_requirements_name_four_videos_and_two_viewpoints():
    t = requirements(2)
    assert "Four in total" in t and "**V1**; **V2**" in t and "V3" not in t
    assert "4 videos averaging up to about 1,024 MB each fit" in t                                       # 4096 / 4


def test_every_number_follows_the_enforcing_limits_and_is_never_hard_coded():
    lim = replace(RV.room_limits(), max_file_bytes=300 * MiB, max_disk_bytes=1000 * MiB)
    t = requirements(3, lim)
    assert "300 MB" in t and "1,000 MB" in t and "1,024 MB" not in t and "4,096 MB" not in t
    assert "167 MB each" in t                                                                          # 1000 / 6


def test_the_aggregate_cap_is_not_raised_to_fit_six_maximum_size_files():
    lim = RV.room_limits()
    assert lim.max_disk_bytes == 4 * lim.max_file_bytes < 6 * lim.max_file_bytes                         # unchanged: four files' worth
    assert "averaging up to about 683 MB each" in requirements(3)                                        # and the text is honest that six maximum-size files will not all fit


# ================================================================ the server enforces what is displayed
def store(tmp_path, **kw):
    s = ChunkStore(replace(RV.room_limits(), max_active_uploads=kw.pop("active", 1), **kw), root=tmp_path / "rooms", slots=RV.ALL_ROOM_SLOTS, root_prefix="f8_room_videos_"); return s


def test_the_per_file_limit_is_refused_with_the_displayed_number(tmp_path):
    s = store(tmp_path, max_file_bytes=5 * MiB, max_disk_bytes=20 * MiB)
    try:
        with pytest.raises(UploadError) as e:
            s.begin("visitor-a", "a.mp4", 6 * MiB, CHUNK, slot="p1_left")
        assert e.value.code == "file_too_large" and "limit is 5 MB" in e.value.message
        assert "5 MB" in requirements(3, s.limits)                                                      # the page shows the very number the server refuses at
    finally:
        s.close()


def test_the_total_budget_is_shared_by_all_visitors_and_one_upload_runs_at_a_time(tmp_path):
    s = store(tmp_path, max_file_bytes=6 * MiB, max_disk_bytes=10 * MiB, active=1, reserve_sibling=False)
    try:
        s.begin("visitor-a", "a.mp4", 5 * MiB, CHUNK, slot="p1_left")
        with pytest.raises(UploadError) as busy:
            s.begin("visitor-b", "b.mp4", 4 * MiB, CHUNK, slot="p1_left")                              # a DIFFERENT visitor while one upload is in progress
        assert busy.value.code == "busy" and "Another upload is in progress" in busy.value.message
    finally:
        s.close()
    s = store(tmp_path / "second", max_file_bytes=6 * MiB, max_disk_bytes=10 * MiB, active=4, reserve_sibling=False)
    try:
        s.begin("visitor-a", "a.mp4", 5 * MiB, CHUNK, slot="p1_left")
        s.begin("visitor-a", "b.mp4", 4 * MiB, CHUNK, slot="p1_right")                                   # 9 of 10 MB reserved by one visitor
        with pytest.raises(UploadError) as full:
            s.begin("visitor-b", "c.mp4", 2 * MiB, CHUNK, slot="p1_left")                                # another visitor: the SERVER-WIDE budget is what ran out
        assert full.value.code == "disk_limit" and "temporary storage limit" in full.value.message
        t = requirements(3, s.limits)
        assert "10 MB" in t and "shared by all visitors" in t and "no separate per-visitor limit" in t     # the text says exactly this
    finally:
        s.close()


CAL = dict(image_size_wh=[160, 96], capture_mode=dict(left="video", right="video"))


def uploads(**override):
    out = {}
    for i, slot in enumerate(RV.ROOM_UPLOAD_SLOTS):
        info = override.get(slot, good_info())
        out[slot] = dict(name=f"{slot}.mp4", size_bytes=1000 + i, sha256=f"{i:064x}", state="complete", inspection=info)
    return out


def errors_of(slot, **override):
    rep = RV.assess_room_set(uploads(**override), CAL); return rep.slots[slot].errors


def test_duration_frame_rate_resolution_and_size_are_enforced_with_the_displayed_limits():
    t = requirements(2)
    assert any("1–120 s" in m for m in errors_of("p1_left", p1_left=good_info(n=3631, fps=30.0)))      # 121 s is refused with the very range shown
    assert any("1–480" in m for m in errors_of("p1_left", p1_left=good_info(fps=500.0)))
    assert any("160 × 96" in m and "320 × 192" in m for m in errors_of("p1_left", p1_left=good_info(w=320, h=192)))        # resolution must equal the calibration
    assert any("portrait" in m or "landscape" in m for m in errors_of("p1_left", p1_left=good_info(w=96, h=160)))        # and so must the orientation
    big = uploads(); big["p1_left"]["size_bytes"] = RV.room_file_limit_bytes() + 1
    assert any("limit for a room video is 1024 MB" in m for m in RV.assess_room_set(big, CAL).slots["p1_left"].errors)
    assert "1–120 seconds" in t and "1–480 frames per second" in t and "1,024 MB" in t


def test_left_right_and_cross_video_requirements_are_enforced():
    rep = RV.assess_room_set(uploads(p1_right=good_info(w=320, h=192)), CAL)
    assert any("different frame sizes" in m for m in rep.set_errors)                                      # "all videos must have the same frame size"
    dup = uploads(); dup["p1_right"]["sha256"] = dup["p1_left"]["sha256"]
    assert any("same file" in m for m in RV.assess_room_set(dup, CAL).slots["p1_left"].errors)            # "every video must be a different file"
    rep = RV.assess_room_set(uploads(p1_right=good_info(fps=25.0)), CAL)
    assert any("different frame rates" in m for m in rep.set_warnings)                                    # a left/right difference of more than 1 fps is flagged


# ================================================================ the accepted formats are the ones the file check accepts (and the picker offers no other)
HEADS = {
    "MP4 / M4V (ISO)": b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00", "M4V": b"\x00\x00\x00\x18ftypM4V \x00\x00\x00\x00", "MOV (QuickTime)": b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00",
    "Matroska/WebM": b"\x1aE\xdf\xa3" + b"\x00" * 12, "AVI": b"RIFF\x00\x00\x00\x00AVI LIST",
}


@pytest.mark.parametrize("name", list(HEADS))
def test_each_listed_container_passes_the_signature_check(name):
    assert sniff_container(HEADS[name])[0] in ("mp4", "quicktime", "matroska", "avi")


def test_the_listed_formats_match_the_signature_check_and_other_types_are_refused():
    assert {sniff_container(h)[0] for h in HEADS.values()} == {"mp4", "quicktime", "matroska", "avi"}
    for word in ("MP4", "M4V", "MOV", "Matroska", "WebM", "AVI"):
        assert word in UT.ACCEPTED_FORMATS
    for bad in (b"\x00\x00\x00\x01\x40\x01\x0c\x01\xff\xff\x01\x60\x00\x00\x03\x00\x90",       # a raw H.265 stream: no container, no signature
                b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, b"GIF89a" + b"\x00" * 8, b"%PDF-1.7" + b"\x00" * 8, b"PK\x03\x04" + b"\x00" * 12):
        assert sniff_container(bad)[0] is None


def test_the_file_picker_offers_only_what_the_server_accepts():
    html = (MODULE / "f8_chunk_poc" / "frontend" / "index.html").read_text()
    accept = re.search(r'accept="([^"]*)"', html).group(1).split(",")
    assert ".hevc" not in accept                                                                          # a raw .hevc stream is refused by the signature check
    assert {"video/*", ".mov", ".mp4", ".m4v"} <= set(accept)


# ================================================================ the real page shows the block beside the uploaders, from the real limits
def _app(nvp, per_file_mb, total_mb):
    import sys
    for p in (HERE_S, HERE_S + "/..", HERE_S + "/../f8_chunk_poc"):
        sys.path.insert(0, p)
    import streamlit as st
    from dataclasses import replace as rep
    import f8_visitor_pages as P
    import f8_visitor_service as S
    import room_videos as RV
    from calibration_runner import RunnerLimits
    from chunk_store import ChunkStore, Limits, MiB
    from result_store import ResultStore
    import tempfile
    if "svc" not in st.session_state:
        d = tempfile.mkdtemp(); lim = rep(RV.room_limits(), max_file_bytes=int(per_file_mb * MiB), max_disk_bytes=int(total_mb * MiB))
        st.session_state["svc"] = S.VisitorCalibrationService(ChunkStore(Limits(), root=d + "/c"), ResultStore(d + "/r"), RunnerLimits(), room_store=ChunkStore(lim, root=d + "/v", slots=RV.ALL_ROOM_SLOTS))
    svc = st.session_state["svc"]; sess = P.session_of(st.session_state)
    if sess.accepted is None and "tested" not in st.session_state:
        from test_f8_visitor_service import make_doc
        st.session_state["tested"] = True
        sess.review = S.review_f8_document(make_doc("GOOD")[1]); svc.accept(sess)
        if nvp == 3:
            svc.set_viewpoint_count(sess, "B", 3); sess.switch_wall("B")
    P.render(svc, sess, st.session_state)


def open_room_page(nvp, per_file_mb, total_mb):
    at = AppTest.from_function(_app, args=(nvp, per_file_mb, total_mb), default_timeout=30)
    at.run(); assert not at.exception
    sess = at.session_state["f8i_session"]; sess.go("room"); sess.guide.room.set_confirmed(1, True); sess.guide.room.set_confirmed(2, True); at.run()
    assert not at.exception
    return at


def test_the_two_viewpoint_page_shows_the_block_with_the_stores_own_limits():
    at = open_room_page(2, 300, 1000)
    text = "\n".join(m.value for m in at.markdown)
    assert "**Recording and upload requirements**" in text and "Four in total" in text
    assert "**300 MB**" in text and "**1,000 MB**" in text and "shared by all visitors" in text                  # the numbers are the store's, not constants
    assert "1–120 seconds" in text and "2160 × 3840" in text and "Not yet verified" in text


def test_the_three_viewpoint_page_names_six_videos_and_the_three_viewpoints():
    at = open_room_page(3, 1024, 4096)
    text = "\n".join(m.value for m in at.markdown)
    assert "Six in total" in text and "**V1** (left end); **V2** (middle); **V3** (right end)" in text and "**4,096 MB**" in text
    assert "Viewpoint 3 · Right camera video" in text                                                          # the uploaders are there too, below the block


def test_wall_a_three_viewpoints_stays_disabled_and_labelled_not_yet_validated():
    at = open_room_page(2, 1024, 4096)                                                                         # Wall A page
    text = "\n".join(c.value for c in at.caption) + "\n".join(m.value for m in at.markdown)
    assert VPC.NOT_VALIDATED in text and "3 viewpoints" in text


def test_the_production_gate_stays_closed():
    import f8_visitor_service as S
    assert S.ROOM_PROCESSING_ENABLED is False

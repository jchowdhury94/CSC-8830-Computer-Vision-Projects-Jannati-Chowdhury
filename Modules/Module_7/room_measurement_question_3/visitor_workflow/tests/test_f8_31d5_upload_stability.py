"""F8.31D-5: the uploader iframes must never be re-created while an upload may be in flight.

Found in the real app: the bordered step container was keyed with its state (``..._active`` / ``..._done``).  When the state changed (all four videos accepted, or one of them
replaced afterwards) Streamlit re-created every uploader iframe inside it; the upload in flight was abandoned and its card stayed at "Uploading / checking" forever.
The box now has a STABLE key and only a small heading wrapper carries the state.  SYNTHETIC fixtures only."""
import re
from pathlib import Path

import f8_visitor_pages as P
from test_f8_31b_frames import clips, cstore, rstore, svc, upload_all, accepted         # noqa: F401  (fixtures)
from test_f8_31b_pages import page_app


def test_no_state_dependent_key_on_a_container_that_holds_uploaders():
    src = Path(P.__file__).read_text()
    for m in re.finditer(r'st\.container\(([^)]*)\)', src):
        args = m.group(1)
        if "border=True" in args and re.search(r'key=f"f8i_step\w*\{state\}', args):
            raise AssertionError("a bordered step box is keyed with its state: " + args)
    assert 'key=f"f8i_stepbox_r{n}{sfx}"' in src and 'key=f"f8i_stepbox_{n}"' in src          # stable boxes
    assert 'key=f"f8i_stmark_r{n}{sfx}_{state}"' in src and 'key=f"f8i_stmark_{n}_{state}"' in src   # the state lives only in the heading wrapper


def test_room_page_renders_in_the_active_and_the_done_state(svc, clips):
    s = accepted(svc); s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True); s.step = "room"
    upload_all(svc, s, clips, only=["p1_left"])
    at = page_app(svc, s); at.run(); assert not at.exception
    upload_all(svc, s, clips)                                                           # all four: step 3 flips to "done"
    at = page_app(svc, s); at.run(); assert not at.exception
    assert any("4 of 4" in str(getattr(p, "text", "")) or "4 of 4" in str(p) for p in at.get("progress")) or s.room_ready

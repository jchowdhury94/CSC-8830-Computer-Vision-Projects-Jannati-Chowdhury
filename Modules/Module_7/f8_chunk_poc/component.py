"""Stage F8 PoC: the chunk-upload component (static frontend, no build step; same approach as Module 7's corner_drag component)."""
from pathlib import Path

import streamlit.components.v1 as components

_component = components.declare_component("f8_chunk_uploader", path=str(Path(__file__).parent / "frontend"))


def chunk_uploader(chunk_size: int, max_file_bytes: int, ack, epoch: int, key: str, slot: str | None = None, label: str | None = None, theme: str | None = None, hide_idle_hint: bool = False):
    """Renders the file picker + progress UI. Returns the browser's latest message (dict) or None.
    ``ack`` is the server's answer to the previous message; ``epoch`` changes to reset the widget.
    ``theme`` ('dark' | 'light' | None): None follows the OS preference (the F8 proof of concept); the always-dark visitor app passes 'dark'.
    ``hide_idle_hint`` (default False) blanks the initial technical hint text for visitor-facing pages; nothing else changes.
    ``slot`` ('left' | 'right' | None) is sent with the init message: the SERVER decides what the slot means, never the file name."""
    return _component(chunk_size=int(chunk_size), max_file_bytes=int(max_file_bytes), ack=ack, epoch=int(epoch), slot=slot, label=label, theme=theme, hide_idle_hint=bool(hide_idle_hint), key=key, default=None)

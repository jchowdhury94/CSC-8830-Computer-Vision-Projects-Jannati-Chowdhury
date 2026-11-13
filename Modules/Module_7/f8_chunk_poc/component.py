"""Stage F8 PoC: the chunk-upload component (static frontend, no build step; same approach as Module 7's corner_drag component)."""
from pathlib import Path

import streamlit.components.v1 as components

_component = components.declare_component("f8_chunk_uploader", path=str(Path(__file__).parent / "frontend"))


def chunk_uploader(chunk_size: int, max_file_bytes: int, ack, epoch: int, key: str):
    """Renders the file picker + progress UI. Returns the browser's latest message (dict) or None.
    ``ack`` is the server's answer to the previous message; ``epoch`` changes to reset the widget."""
    return _component(chunk_size=int(chunk_size), max_file_bytes=int(max_file_bytes), ack=ack, epoch=int(epoch), key=key, default=None)

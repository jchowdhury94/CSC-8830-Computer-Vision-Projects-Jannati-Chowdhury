"""Stage F8 PoC: message handling between the browser component and the ChunkStore (NO Streamlit import).

The browser sends one JSON message at a time through ``setComponentValue``; Python answers with an ack that is passed back to the component as
an argument. Every message carries a ``msg_id``. Streamlit re-delivers the component's last value on every rerun, so a message is processed
once and its ack is cached (``state``); a re-sent message with the same id gets the cached ack and cannot write a chunk twice.
"""
from __future__ import annotations

import base64
import binascii
import time

from chunk_store import ChunkStore, UploadError

MSG_TYPES = ("init", "chunk", "finish", "cancel")


def _err(msg_id, e: UploadError) -> dict:
    d = e.to_dict(); d["msg_id"] = msg_id
    return d


def handle(store: ChunkStore, owner: str, state: dict, msg) -> dict | None:
    """Process one component value. ``state`` ({'last_id', 'last_ack', 'processed'}) lives in the session; it holds no file data.
    Returns the ack for ``msg`` (new or cached), or the last ack if there is no message."""
    if not isinstance(msg, dict):
        return state.get("last_ack")
    msg_id = msg.get("msg_id")
    if not isinstance(msg_id, str) or not 0 < len(msg_id) <= 64:
        return state.get("last_ack")
    if msg_id == state.get("last_id"):
        return state.get("last_ack")                                   # rerun / retransmission: no side effects
    t0 = time.perf_counter()
    try:
        ack = _dispatch(store, owner, msg)
    except UploadError as e:
        ack = _err(msg_id, e)
    except Exception:                                                  # never leak internals or leave the page spinning
        ack = dict(ok=False, error="internal", message="Unexpected server error.")
    ack["msg_id"] = msg_id
    ack["handler_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    state["last_id"], state["last_ack"] = msg_id, ack
    state["processed"] = state.get("processed", 0) + 1
    return ack


def _dispatch(store: ChunkStore, owner: str, msg: dict) -> dict:
    kind = msg.get("type")
    if kind not in MSG_TYPES:
        raise UploadError("bad_request", "Unknown message type.")
    if kind == "init":
        return store.begin(owner, msg.get("name"), msg.get("size"), msg.get("chunk_size"), msg.get("fingerprint", ""))
    uid = msg.get("upload_id")
    if kind == "chunk":
        b64 = msg.get("data")
        if not isinstance(b64, str):
            raise UploadError("bad_request", "Missing chunk data.")
        if len(b64) > (store.limits.max_chunk_bytes * 4) // 3 + 16:    # reject BEFORE decoding
            raise UploadError("bad_length", "Chunk is larger than the maximum chunk size.")
        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            raise UploadError("corrupt_chunk", "Chunk data is not valid base64; it can be sent again.")
        return store.put_chunk(owner, uid, msg.get("index"), data, msg.get("sha256"))
    if kind == "finish":
        return store.finish(owner, uid, msg.get("sha256"), msg.get("client_stats"))
    return store.cancel(owner, uid)

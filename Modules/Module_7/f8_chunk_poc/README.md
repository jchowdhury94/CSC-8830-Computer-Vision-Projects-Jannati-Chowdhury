# Stage F8 / F8.1 — chunked-upload proof of concept (isolated, hardened)

An experiment, **not part of Module 7**: nothing in `Pages/`, `home.py`, `requirements.txt` or the existing deployment imports or links to it.
Classical computer vision only (OpenCV decode, no calibration, no ML).

## What it does
1. A passcode gate (F8.1) stands in front of everything. Without the correct passcode no store, upload component or message handler exists.
2. A custom Streamlit component (`frontend/index.html`, static, no build step) slices the chosen file with `File.slice()` into ~2 MB chunks.
3. Chunks go **one at a time** through `Streamlit.setComponentValue` (no custom HTTP endpoint) with `{upload_id, index, sha256, base64 data}`.
4. Python (`protocol.py` → `chunk_store.py`) verifies index, length and SHA-256 of each chunk, appends it to a private temp file and acknowledges it through a component argument.
5. The browser then sends the SHA-256 of the whole file; the server compares it with its running hash **and** a fresh re-read of the file from disk.
6. `video_check.py` inspects the assembled file **safely** (below). Nothing is decoded in the Streamlit process.

## Security (F8.1)
| Protection | Where |
|---|---|
| Passcode from Streamlit secrets or env `F8_PASSCODE` (≥ 8 chars); **fail closed** if missing/short/not text; constant-time compare; per-session and global lock-out; never logged | `auth.py`, top of `app.py` |
| Signature check (MP4/MOV/M4V, Matroska/WebM, AVI); extension never used; image/audio ISO brands (HEIC, M4A…) refused | `video_check.sniff_container` |
| MP4/MOV header parsed in pure Python with bounded reads; declared size checked **before any decoder runs** (max side 8192 px, max 40 MP; larger of stsd/tkhd wins) | `video_check.parse_iso_video_header` |
| OpenCV runs in a **separate process**: isolated interpreter (`-I`), scrubbed environment (no passcode), own process group, 20 s wall-clock timeout (whole group killed), CPU limit, Linux address-space cap; it re-checks reported and decoded sizes | `video_check._worker_main`, `_run_worker` |
| Real free-space check at `begin` (free − bytes promised to running uploads − margin of max(512 MB, 5 % of disk)), re-checked before every chunk (64 MB floor), ENOSPC handled; decided under one lock so concurrent sessions cannot bypass it; unreadable disk ⇒ refuse | `chunk_store.begin/put_chunk/capacity` |
| One active upload; max file 1 GiB; max reserved 3 GiB; chunk 64 KB–8 MB; idle 300 s; max 3600 s; stalled (>20 s) upload may be taken over; files kept ≤ 900 s | `chunk_store.Limits` |
| Private files (0700/0600), random upload ids, owner token per session, no file names in paths or logs | `chunk_store.py` |

## Files
`app.py` page · `auth.py` passcode gate · `chunk_store.py` sessions/limits/disk/assembly · `protocol.py` message handling · `component.py` + `frontend/index.html` browser component ·
`video_check.py` safe inspection · `perf.py` RSS helpers · `make_synthetic.py` test inputs · `run_e2e_local.py`, `run_e2e_gate.py` real-browser tests ·
`tests/` (pytest, synthetic data only) · `requirements.txt` (for a separate Cloud deployment) · `.gitignore` · `.streamlit/secrets.toml.example` (placeholder only) · `outputs/` (local results, git-ignored).

## Test locally
```bash
cd "<repo>/Modules/Module_7/f8_chunk_poc"
python -m pytest tests -q                                  # 117 tests (~5 s; one test deliberately waits ~2 min only if the timeout protection is broken)
python -c 'import secrets; print(secrets.token_urlsafe(18))'          # generate a passcode; do not commit it
export F8_PASSCODE='<paste-the-generated-passcode-here>'
../../../.venv/bin/streamlit run app.py --server.port 8741 --server.address 127.0.0.1    # http://127.0.0.1:8741  (log in with the passcode)
# optional real-browser tests (Playwright from the framework Python)
F8_E2E_PASSCODE="$F8_PASSCODE" python3 run_e2e_gate.py http://127.0.0.1:8741
F8_E2E_PASSCODE="$F8_PASSCODE" python3 run_e2e_local.py <file> <out.json> http://127.0.0.1:8741 <server_pid>
```
Without `F8_PASSCODE` the app shows "locked" and nothing else (try it: start without the variable).

## Separate Cloud test deployment (NOT done — needs approval)
1. Approvals first: creating the branch, pushing it, creating a **new** Community Cloud app (never replace the existing one), and the choice of account.
2. On the branch commit only this folder's source (see `.gitignore`); leave the root `requirements.txt` and any root `.streamlit/` untouched.
3. share.streamlit.io → *Create app* → main file `Modules/Module_7/f8_chunk_poc/app.py`, Python **3.12** (numpy 2.5.3 needs ≥ 3.12; the version cannot be changed later).
4. Advanced settings → Secrets: paste `F8_PASSCODE = "<your long random passcode>"` (never in the repo). The app stays locked until this exists.
5. Run the Cloud test plan from small to large; delete the app and the branch afterwards.

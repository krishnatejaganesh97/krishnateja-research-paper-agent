#!/bin/bash
# Run the Teja Research Paper Agent locally (demo mode, no API key needed).
# Fixed: always invoke pip/uvicorn through the venv python
# (`python -m pip`) instead of the `pip` wrapper script — that wrapper's
# shebang goes stale if the project folder is ever renamed, which broke
# `pip install` with "externally-managed-environment" / file-not-found.
# If the venv itself is missing, it is created first.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install -q -r requirements.txt
PORT="${PORT:-8002}"
.venv/bin/python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port "$PORT"

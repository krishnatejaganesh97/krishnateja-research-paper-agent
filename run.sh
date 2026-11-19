#!/bin/bash
# Run the Teja Research Paper Agent locally (demo mode, no API key needed).
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -q -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8002

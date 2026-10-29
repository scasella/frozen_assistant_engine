#!/usr/bin/env bash
# Launch the Frozen Assistant Studio on http://127.0.0.1:8000 (pass --instruct swap|cached on low-memory machines).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || PY=python
exec "$PY" -m demo.server "$@"

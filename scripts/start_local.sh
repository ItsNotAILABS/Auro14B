#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8090}

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "error: Python 3.10+ is required" >&2
  exit 127
fi

export HOST PORT
exec "$PYTHON" -m auro_native_llm.server.app

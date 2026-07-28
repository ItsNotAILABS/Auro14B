#!/usr/bin/env sh
set -eu

echo "notice: Replit deployment is retired; forwarding to scripts/start_local.sh" >&2
exec "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/start_local.sh" "$@"

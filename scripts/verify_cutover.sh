#!/usr/bin/env sh
set -eu

fail() {
  printf '%s\n' "cutover gate failed: $1" >&2
  exit 1
}

[ ! -e .replit ] || fail '.replit must be removed'
[ -f scripts/start_local.sh ] || fail 'scripts/start_local.sh is required'
[ -f native_llm/fabric/package.json ] || fail 'portable fabric package is required'
[ -f cloudflare-platform/wrangler.jsonc ] || fail 'Cloudflare production envelope is required'

find . \
  -path './.git' -prune -o \
  -path './node_modules' -prune -o \
  -path './REPLIT_RETIREMENT.md' -prune -o \
  -path './scripts/replit_start.sh' -prune -o \
  -type f \( \
    -name 'package.json' -o \
    -name 'package-lock.json' -o \
    -name '*.js' -o \
    -name '*.mjs' -o \
    -name '*.cjs' -o \
    -name '*.ts' -o \
    -name '*.tsx' -o \
    -name '*.jsx' \
  \) -print0 | xargs -0 grep -nH '@replit/' > /tmp/replit-packages.txt 2>/dev/null || true

if [ -s /tmp/replit-packages.txt ]; then
  cat /tmp/replit-packages.txt >&2
  fail 'active @replit package references remain'
fi

node --check native_llm/fabric/src/index.mjs
node --check native_llm/fabric/src/platform-client.mjs
node --check native_llm/fabric/src/scheduler.mjs
node --check native_llm/fabric/src/receipt-chain.mjs
node --check native_llm/fabric/src/model-family.mjs

printf '%s\n' 'cutover policy check passed'

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

if grep -RIn --exclude-dir=.git --exclude='REPLIT_RETIREMENT.md' --exclude='replit_start.sh' '@replit/' . >/tmp/replit-packages.txt 2>/dev/null; then
  cat /tmp/replit-packages.txt >&2
  fail 'active @replit package references remain'
fi

node --check native_llm/fabric/src/index.mjs
node --check native_llm/fabric/src/platform-client.mjs
node --check native_llm/fabric/src/scheduler.mjs
node --check native_llm/fabric/src/receipt-chain.mjs
node --check native_llm/fabric/src/model-family.mjs

printf '%s\n' 'cutover policy check passed'

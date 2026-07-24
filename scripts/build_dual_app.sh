#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PUBLIC="$ROOT/cloudflare-platform/public"
WEB3_DIST="$ROOT/him-web3/client/dist"

rm -rf "$PUBLIC"
mkdir -p "$PUBLIC/web3" "$PUBLIC/foundry"

npm ci --prefix "$ROOT/him-web3/client"
npm run build --prefix "$ROOT/him-web3/client"
cp -R "$WEB3_DIST"/. "$PUBLIC/web3/"
cp "$ROOT/auro_foundry/web/index.html" "$PUBLIC/foundry/index.html"

cat > "$PUBLIC/index.html" <<'EOF'
<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HIM · Auro Applications</title><style>:root{color-scheme:dark;font-family:Inter,system-ui;background:#070b12;color:#eef4ff}body{margin:0;min-height:100vh;display:grid;place-items:center;background:radial-gradient(circle at top,#1e3154,#070b12 58%)}main{width:min(920px,92vw)}h1{font-size:clamp(2rem,6vw,4.5rem);margin:.2em 0}p{color:#a8b6cc}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px;margin-top:30px}a{display:block;text-decoration:none;color:inherit;padding:24px;border:1px solid #ffffff22;border-radius:20px;background:#111a2acc;box-shadow:0 20px 60px #0007}a:hover{border-color:#76a8ff}small{color:#7f91ad}</style><main><small>ONE RUNTIME · TWO APPLICATIONS</small><h1>HIM / Auro</h1><p>The Web3 operations surface and the Foundry conversation surface share one governed Cloudflare backend and receipt plane.</p><div class="grid"><a href="/web3/"><h2>HIM Web3</h2><p>Chain reads, balances, blocks, capabilities, and governed operations.</p></a><a href="/foundry/"><h2>Auro Foundry</h2><p>Conversation and model-facing interface through the shared Worker API.</p></a><a href="/operator/"><h2>THESIS Operator</h2><p>Inspect, plan, authorize, execute, and verify Cloudflare operations.</p></a></div></main></html>
EOF

printf '%s\n' "dual app assets built at $PUBLIC"

# THESIS Polyglot Builder Platform

A deployable builder envelope that routes HIM browser-local intelligence, NIO/Python, MATHESIS/Julia, Node/WASM, Cloudflare, Solidity, and Monad through one governed THESIS surface. Cloudflare supplies the edge control plane; it is not the boundary of the product. Reads and mutations have separate authorization. The API token never enters model context or browser JavaScript.

## Deploy

```bash
npm install
npx wrangler secret put CLOUDFLARE_API_TOKEN
npx wrangler secret put OPERATOR_TOKEN
npx wrangler secret put EXECUTION_SECRET
npx wrangler deploy
```

Use a least-privilege Cloudflare API token. Set `CLOUDFLARE_ACCOUNT_ID` with `wrangler secret put` when an operation needs it. `npm run deploy` is the repeat-deploy command after initial secrets are configured.

## Security model

- Every control-plane route requires the independently configured operator token; only health and the static login shell are public.
- The chat model produces a proposal, never an execution claim.
- `GET`, `HEAD`, and `OPTIONS` may execute immediately.
- Mutations require a proposal-bound, HMAC-signed, short-lived grant.
- Arbitrary origins, traversal paths, token management, membership changes, and cache purges are rejected.
- Every plan and API operation advances a hash-linked Durable Object receipt chain.
- Cloudflare credentials remain in the Worker host.

The default model is Workers AI and is explicit—not a silent fallback. For the repository-native HIM checkpoint, replace `plan()` with a Cloudflare Service Binding to an approved HIM inference deployment; do not label Workers AI weights as HIM weights.

## Next promotion lanes

The Cloudflare API MCP Code Mode server, Worker Loader/Dynamic Workers, Browser Rendering, and Sandbox SDK should be added behind the same host authorization callback. Code Mode is experimental, so it is intentionally not granted unrestricted mutation authority in this baseline.


## User setup choices

Wrangler is only the operator/CI lane. In the web app, Settings accepts an optional Cloudflare API token for the current browser session. The token remains in session storage, is sent only to the authenticated same-origin Worker, is never put into model context, and is not persisted. A managed Worker secret remains available for teams that want durable hosted configuration.

Browser-only HIM/WASM/Monad-read functionality can operate without a Cloudflare management token. Python and Julia appear in the capability registry and activate when their governed service bindings are configured. Monad transactions are prepared unsigned and returned to the browser wallet; the Worker has no signing key and no broadcast method.

See docs/THESIS_POLYGLOT_ENVELOPE.md for runtime ownership and the complete mission lifecycle.

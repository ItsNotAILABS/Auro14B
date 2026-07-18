# HIM Cloudflare Platform

A deployable, Cloudflare-native operator surface: chat planning through Workers AI, Cloudflare API execution, Durable Object receipts, and a responsive web UI. Reads and mutations have separate authorization. The API token never enters model context or browser JavaScript.

## Deploy

```bash
npm install
npx wrangler secret put CLOUDFLARE_API_TOKEN
npx wrangler secret put EXECUTION_SECRET
npx wrangler deploy
```

Use a least-privilege Cloudflare API token. Set `CLOUDFLARE_ACCOUNT_ID` with `wrangler secret put` when an operation needs it. `npm run deploy` is the repeat-deploy command after initial secrets are configured.

## Security model

- The chat model produces a proposal, never an execution claim.
- `GET`, `HEAD`, and `OPTIONS` may execute immediately.
- Mutations require a proposal-bound, HMAC-signed, short-lived grant.
- Arbitrary origins, traversal paths, token management, membership changes, and cache purges are rejected.
- Every plan and API operation advances a hash-linked Durable Object receipt chain.
- Cloudflare credentials remain in the Worker host.

The default model is Workers AI and is explicit—not a silent fallback. For the repository-native HIM checkpoint, replace `plan()` with a Cloudflare Service Binding to an approved HIM inference deployment; do not label Workers AI weights as HIM weights.

## Next promotion lanes

The Cloudflare API MCP Code Mode server, Worker Loader/Dynamic Workers, Browser Rendering, and Sandbox SDK should be added behind the same host authorization callback. Code Mode is experimental, so it is intentionally not granted unrestricted mutation authority in this baseline.

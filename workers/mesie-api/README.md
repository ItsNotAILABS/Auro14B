# MESIE API — Cloudflare Worker

Edge HTTP API for MESIE spectral validation and matching. Runs on **Cloudflare Workers** (V8); complements the Python `mesie` package for low-latency, global endpoints.

## Prerequisites

- [Cloudflare account](https://dash.cloudflare.com/) with Workers enabled
- Node.js 18+
- `npm install` in this directory

## Platform note (Windows ARM64)

`workerd` (Wrangler local runtime) does not support **win32 arm64**. On this machine:

- **Deploy** still works: `npx wrangler deploy` (build runs on Cloudflare).
- **Local dev** use WSL2 x64, a GitHub Action, or another x64 machine.

## Quick start (local — x64 / WSL / macOS / Linux)

```bash
cd workers/mesie-api
npm install
npm run dev
```

Test:

```bash
curl http://localhost:8787/health
curl -X POST http://localhost:8787/v1/validate \
  -H "Content-Type: application/json" \
  -d "{\"record_id\":\"t1\",\"components\":[{\"name\":\"a\",\"frequency\":[1,2,3],\"amplitude\":[0.1,0.5,0.2]}]}"
```

## Deploy to Cloudflare

1. Log in:

```bash
npx wrangler login
```

2. Deploy:

```bash
npm run deploy
```

3. Optional API key:

```bash
npx wrangler secret put MESIE_API_KEY
# paste a strong random key
```

Clients send `Authorization: Bearer <key>` or `X-MESIE-Key: <key>`.

4. Public dev mode (no key):

```bash
npx wrangler secret delete MESIE_API_KEY
# or set in wrangler.toml: MESIE_PUBLIC = "true"
```

## API routes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status |
| GET | `/v1` | API index |
| GET | `/v1/datasets` | Bundled dataset catalog (metadata) |
| POST | `/v1/validate` | Structural + spectral validation |
| POST | `/v1/match` | Cosine + RMSE composite match |

### Match example

```bash
curl -X POST https://mesie-api.<your-subdomain>.workers.dev/v1/match \
  -H "Content-Type: application/json" \
  -d "{\"reference\":{\"record_id\":\"r1\",\"components\":[{\"name\":\"a\",\"frequency\":[1,2,3,4],\"amplitude\":[0.2,0.5,0.3,0.1]}]},\"candidate\":{\"record_id\":\"r2\",\"components\":[{\"name\":\"a\",\"frequency\":[1,2,3,4],\"amplitude\":[0.22,0.48,0.31,0.12]}]}}"
```

## Expansion roadmap

| Phase | Feature |
|-------|---------|
| **Now** | Validate + match at the edge |
| **Next** | R2 bucket `mesie-datasets` for reference JSON; `GET /v1/reference/:name` |
| **Next** | Workers AI binding for embedding summaries |
| **Later** | Queue consumer calling Python MESIE on GPU host for full `mesie.ai` pipelines |

## Python parity

For full MESIE (6-level validation, RotDnn, cognitive adapters, 656 tests), use:

```bash
pip install mesie[full]
```

This worker implements a **subset** aligned with public JSON record shape.
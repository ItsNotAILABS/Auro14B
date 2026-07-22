# NEXUS Relay

**The self-hosted public-web context API for AI agents.**

NEXUS Relay runs on your Cloudflare account and converts public web pages, RSS/Atom feeds, and JSON endpoints into normalized, provenance-bearing objects. It exposes the same capability through REST and MCP, so Claude, Codex, ChatGPT-compatible clients, AURO, NOVA, and custom agents can use one endpoint without a mandatory scraping-platform subscription.

## What it replaces

For the supported public-web use case, Relay can replace recurring subscriptions used only to fetch and clean public URLs or feeds. It is not a promise to bypass authentication, CAPTCHAs, private profiles, platform access controls, or website terms.

## Surfaces

- `GET /health`
- `GET /v1/read?url=https://example.com`
- `POST /v1/read`
- `POST /mcp`

Every successful read includes the final URL, fetch timestamp, HTTP status, byte count, latency, canonical URL when available, and a SHA-256 hash of normalized content.

## Deploy

```bash
npm install -g wrangler
wrangler login
wrangler deploy
```

Optional KV cache:

```bash
wrangler kv namespace create CACHE
# Add the returned binding to wrangler.toml
```

## REST example

```bash
curl -X POST https://YOUR-WORKER.workers.dev/v1/read \
  -H 'content-type: application/json' \
  -d '{"url":"https://example.com/feed.xml","mode":"auto"}'
```

## MCP configuration

```json
{
  "mcpServers": {
    "nexus-relay": {
      "url": "https://YOUR-WORKER.workers.dev/mcp"
    }
  }
}
```

## Product position

Relay is a substrate, not a marketplace. It does not resell third-party Actors. Operators own the Worker, cache, policies, domains, and bills. Future adapters can add Browser Run, signed webhooks, scheduled watches, D1 history, R2 archives, and paid provider fallbacks behind the same schema.

## Security boundary

- Public HTTP(S) only.
- Loopback and common private-network targets are blocked.
- Credentials embedded in URLs are stripped.
- Response size is bounded.
- No browser automation or anti-bot bypass is enabled in v0.1.
- Operators remain responsible for source terms, robots directives, privacy law, retention, and commercial use.

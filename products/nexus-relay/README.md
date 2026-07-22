# NEXUS Relay

**The hosted public-web context API for AI agents.**

NEXUS Relay runs on the operator's Cloudflare account. Customers receive an API key and call one hosted REST or MCP endpoint. They do not deploy infrastructure.

## Customer flow

1. Customer subscribes.
2. Operator issues an `nr_live_...` API key.
3. Customer downloads `SKILL.md` from the hosted service.
4. Their agent uses the skill plus the API key.
5. Relay authenticates, enforces quota and rate limits, records usage, fetches the public source, and returns normalized content with a receipt.

## Surfaces

- `GET /health` — public service status and skill URL
- `GET /SKILL.md` — downloadable agent integration skill
- `GET /v1/usage` — authenticated usage and remaining allowance
- `GET /v1/read?url=https://example.com` — authenticated read
- `POST /v1/read` — authenticated read
- `POST /mcp` — authenticated MCP server

## Customer request

```bash
curl -X POST https://YOUR-DOMAIN/v1/read \
  -H "Authorization: Bearer $NEXUS_RELAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/feed.xml","mode":"auto"}'
```

## MCP configuration

```json
{
  "mcpServers": {
    "nexus-relay": {
      "url": "https://YOUR-DOMAIN/mcp",
      "headers": {
        "Authorization": "Bearer ${NEXUS_RELAY_API_KEY}"
      }
    }
  }
}
```

## Operator deployment

```bash
cd products/nexus-relay
npm install -g wrangler
wrangler login
wrangler d1 create nexus-relay
# Put the returned database id into wrangler.toml
wrangler d1 execute nexus-relay --remote --file=schema.sql
wrangler deploy
```

Issue a customer key:

```bash
node scripts/issue-key.mjs customer@example.com developer 1000 30
```

The script prints the plaintext key once and emits SQL containing only its SHA-256 hash. Deliver the plaintext key securely and never store it in source control.

## Metering model

D1 stores customers, hashed API keys, monthly limits, per-minute limits, and append-only usage events. Every billable `relay_read` returns a billing event ID. Stripe or another billing system can consume aggregated usage later without changing the customer-facing API.

## Current supported source types

- Public HTML pages
- RSS and Atom feeds
- Public JSON APIs

Markdown, plain text, CSV, and sitemap modes are included in the downloadable skill contract as the next adapter lane and must not be marketed as live until their parsers are merged and tested.

## Security boundary

- Public HTTP(S) only.
- Loopback and common private-network targets are blocked.
- Credentials embedded in URLs are stripped.
- Response size is bounded.
- Customer keys are stored as SHA-256 hashes.
- Reads require an active key and are metered.
- No login bypass, CAPTCHA evasion, private-profile access, or anti-bot circumvention is enabled.
- The operator remains responsible for source terms, robots directives, privacy law, retention, pricing, and commercial use.

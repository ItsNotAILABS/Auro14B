# NEXUS Relay Public Release v1

NEXUS Relay is ready for public release as a hosted, metered public-web context API for AI agents.

## Shipped product

- Hosted REST and MCP endpoint
- Downloadable `SKILL.md`
- API-key authentication
- Customer and subscription state in Cloudflare D1
- Atomic monthly quota and per-minute rate limiting
- Stripe Checkout, portal, webhook reconciliation, invoice state, and usage-meter outbox
- Public HTML, RSS/Atom, JSON, Markdown, plain text, CSV, and XML sitemap normalization
- Provenance receipts with final URL, redirect chain, latency, byte count, fetch time, and SHA-256 content hash
- Redirect validation and optional pinned egress service
- SignalLens fail-closed client

## Release boundary

The repository contains production code, tests, migrations, deployment configuration, and operator documentation. A live hosted deployment still requires operator-owned external values that must never be fabricated or committed:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`
- Cloudflare D1 database ID
- Stripe secret key
- Stripe webhook secret
- Real Stripe price IDs and entitlement mapping
- Billing success/cancel/portal URLs
- Optional egress URL and shared secret
- Optional production domain and allowed browser origins

## Go-live commands

```bash
cd products/nexus-relay
npm install
npm test
npm run check

wrangler d1 create nexus-relay
# Replace the D1 placeholder in wrangler.toml with the returned database ID.
wrangler d1 execute nexus-relay --remote --file=schema.sql

wrangler secret put STRIPE_SECRET_KEY
wrangler secret put STRIPE_WEBHOOK_SECRET
wrangler secret put BILLING_CRON_SECRET

# Optional hardened egress
wrangler secret put RELAY_EGRESS_SECRET

wrangler deploy
```

After deployment, configure Stripe to send the documented events to:

```text
https://YOUR-DOMAIN/v1/billing/webhook
```

Then issue the first customer and API key using `scripts/issue-key.mjs`.

## Public launch claim

> NEXUS Relay gives AI agents one hosted REST and MCP endpoint for reading public web sources, returning clean structured context with authentication, usage controls, billing, and cryptographic provenance receipts.

## Claims not made

NEXUS Relay does not claim login bypass, CAPTCHA evasion, private-profile access, anti-bot circumvention, residential proxy coverage, or universal website compatibility.

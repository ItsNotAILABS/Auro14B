# NEXUS Relay

**The hosted public-web context API for AI agents.**

NEXUS Relay runs on the operator's infrastructure. Customers receive an API key, download `SKILL.md`, and call one hosted REST or MCP endpoint. They do not deploy Cloudflare or manage the retrieval stack.

## Customer flow

1. Customer selects an operator-approved Stripe price.
2. Relay creates a hosted Checkout Session.
3. Signed Stripe webhooks synchronize subscription entitlements into D1.
4. Customer receives an `nr_live_...` API key and downloads `SKILL.md`.
5. Their agent uses Relay through REST or MCP.
6. Relay authenticates the key and parent customer, reserves quota atomically, retrieves the public source, completes usage only after success, and returns a provenance receipt.
7. Failed payment suspends the customer and disables all customer keys.

## Surfaces

- `GET /health` — public status, billing configuration and egress mode
- `GET /SKILL.md` — downloadable agent integration skill
- `GET /v1/usage` — authenticated quota usage
- `GET /v1/billing` — authenticated subscription and invoice summary
- `POST /v1/billing/checkout` — authenticated Stripe Checkout creation
- `POST /v1/billing/portal` — authenticated Stripe customer portal creation
- `POST /v1/billing/webhook` — signed, idempotent Stripe reconciliation
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

## Billing configuration

The price catalog is an explicit entitlement map. Unknown Stripe prices do not activate Relay keys.

```toml
RELAY_PRICE_CATALOG = '{
  "price_developer": {"plan":"developer","monthly_limit":1000,"rate_limit_per_minute":30},
  "price_team": {"plan":"team","monthly_limit":25000,"rate_limit_per_minute":120}
}'
```

Set secrets:

```bash
wrangler secret put STRIPE_SECRET_KEY
wrangler secret put STRIPE_WEBHOOK_SECRET
```

Configure the Stripe webhook endpoint as:

```text
https://YOUR-DOMAIN/v1/billing/webhook
```

Required events:

- `checkout.session.completed`
- `customer.subscription.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`
- `invoice.paid`
- `invoice.payment_failed`

Webhook event IDs are reserved atomically with `INSERT OR IGNORE`. Failed processing removes the reservation so Stripe retries can reconcile it. Completed events retain a SHA-256 digest and result receipt.

## Pinned outbound egress

Worker DNS preflight alone cannot pin the exact destination IP used by a later HTTPS fetch. For the strongest SSRF boundary, deploy `egress/server.mjs` as a small Node service:

```bash
docker build -t nexus-relay-egress ./egress
docker run --rm -p 8788:8788 \
  -e RELAY_EGRESS_SECRET='replace-me' \
  nexus-relay-egress
```

Then configure the Worker:

```bash
wrangler secret put RELAY_EGRESS_SECRET
# Set RELAY_EGRESS_URL to the private egress /fetch endpoint.
```

The egress service resolves once, rejects private/reserved A and AAAA results, pins the selected IP through Node's `lookup` callback, preserves TLS SNI and Host verification, validates every redirect independently, bounds response bytes, and authenticates Worker requests with HMAC-SHA256.

## SignalLens canonical perception

`clients/signallens.mjs` is the fail-closed SignalLens adapter. It calls only Relay, requires a Relay receipt and metering event, preserves `content_sha256`, and never silently falls back to an ungoverned direct fetch.

```js
import { signalLensFromEnv } from "./clients/signallens.mjs";
const lens = signalLensFromEnv();
const perception = await lens.read("https://example.com/feed.xml", "auto");
```

## Operator deployment

```bash
cd products/nexus-relay
wrangler d1 create nexus-relay
# Put the database ID into wrangler.toml.
wrangler d1 execute nexus-relay --remote --file=schema.sql
wrangler deploy
```

For an existing database:

```bash
wrangler d1 execute nexus-relay --remote --file=migrations/0002_security_and_atomic_metering.sql
wrangler d1 execute nexus-relay --remote --file=migrations/0003_billing_control_plane.sql
```

## Supported source types

Public HTML, RSS/Atom, JSON, Markdown, plain text, CSV, XML sitemaps and sitemap indexes.

## Security boundary

- Customer API keys and parent customer status are both enforced.
- Quota reservations are atomic and failed reads do not consume completed usage.
- Every redirect destination is validated.
- Pinned egress closes the Worker DNS time-of-check/time-of-use gap when configured.
- Authenticated routes do not use wildcard CORS.
- No login bypass, CAPTCHA evasion, private-profile access or anti-bot circumvention is provided.
- The operator remains responsible for source terms, privacy, retention, taxes, refunds, pricing and Stripe account configuration.

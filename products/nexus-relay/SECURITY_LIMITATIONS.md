# NEXUS Relay Security and Commercial Boundaries

## Implemented controls

- Validate the initial URL and every redirect target.
- Resolve A and AAAA records before every outbound hop and reject private, loopback, link-local, multicast, documentation and reserved address ranges.
- Limit redirect count and response size.
- Authenticate API keys against both the key record and parent customer status.
- Reserve quota atomically with one conditional SQLite statement.
- Remove failed reservations so upstream and normalization failures do not consume completed-request quota.
- Restrict authenticated CORS to explicitly configured origins.
- Verify Stripe webhook timestamps and HMAC signatures.
- Reserve Stripe event IDs atomically and retain reconciliation receipts.
- Suspend customers and keys after failed payment or invalid subscription entitlement.
- Reject Stripe prices that are not present in the operator-controlled entitlement catalog.

## DNS pinning modes

### Pinned egress mode

When `RELAY_EGRESS_URL` and `RELAY_EGRESS_SECRET` are configured, Relay sends authenticated retrieval requests to `egress/server.mjs`. The egress service resolves the hostname once, rejects blocked A and AAAA results, pins the selected address through Node's `lookup` callback, preserves TLS SNI and Host validation, and re-runs the same policy for every redirect destination.

This is the required production mode for strong DNS-rebinding resistance.

### Worker-preflight fallback

Without pinned egress, the Worker still performs DNS-over-HTTPS preflight and manual redirect validation. Cloudflare Workers do not expose the exact destination IP selected by the following HTTPS fetch, so a DNS time-of-check/time-of-use window remains in fallback mode. `/health` reports the active egress mode.

## Billing boundary

Relay implements the application billing control plane: Checkout creation, customer portal creation, signed and idempotent webhook reconciliation, subscription and invoice persistence, entitlement synchronization, failed-payment suspension and operator-defined price enforcement.

Stripe remains the payment processor and invoice authority. The operator must configure Stripe products, prices, tax settings, refund policy, webhook delivery and account compliance. Relay does not replace Stripe's ledger or tax services.

## Canonical perception boundary

`clients/signallens.mjs` is a fail-closed canonical perception adapter. It accepts only Relay responses containing provenance and metering receipts and never silently falls back to direct retrieval. Repositories using an older independent SignalLens implementation must replace their direct-fetch boundary with this client before claiming cross-repository canonical perception.

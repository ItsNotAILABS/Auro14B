# NEXUS Relay Security and Commercial Boundaries

## Implemented controls

- Validate the initial URL and every redirect target.
- Resolve A and AAAA records before every outbound hop and reject private, loopback, link-local, multicast, documentation, and reserved address ranges.
- Limit redirect count and response size.
- Authenticate API keys against both the key record and the parent customer status.
- Reserve quota atomically with one conditional SQLite statement.
- Remove failed reservations so upstream and normalization failures do not consume completed-request quota.
- Restrict authenticated CORS to explicitly configured origins.

## Residual DNS rebinding limitation

Cloudflare Workers `fetch` does not expose or pin the exact destination IP selected for an HTTPS request. DNS-over-HTTPS preflight materially reduces obvious DNS rebinding and private-address resolution attacks, but a time-of-check/time-of-use race remains possible if authoritative DNS changes between preflight resolution and the Worker fetch.

A stronger deployment must route outbound retrieval through a controlled egress service that resolves once, pins the destination IP, preserves TLS SNI and Host verification, and rejects redirects independently. NEXUS Relay must not be described as fully DNS-rebinding-proof until that egress lane is implemented and tested.

## Metering versus billing

The current service provides authentication, quota enforcement, rate limiting, and usage-event records. It is metering infrastructure. It does not yet implement subscriptions, payment collection, pricing enforcement, invoices, refunds, taxes, dunning, or Stripe reconciliation.

## Canonical perception integration

NEXUS Relay is intended to become the canonical public-web perception layer for SignalLens and other Medina agents. Until those repositories call Relay rather than fetching sources independently, cross-repository canonical perception is not complete.

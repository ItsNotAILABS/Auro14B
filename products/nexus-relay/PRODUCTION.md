# Production Contract

NEXUS Relay v0.1 provides a stable normalized-document contract for public web content.

## Guarantees

1. One REST and one MCP entrypoint.
2. Deterministic schemas for HTML, feed, and JSON responses.
3. Provenance and content hash on every completed read.
4. SSRF controls for loopback and common private address ranges.
5. Bounded response bodies.
6. Optional operator-owned KV caching.
7. No silent paid-provider fallback.

## Not yet implemented

Browser Run rendering, scheduled watches, webhook delivery, D1 history, R2 archives, API keys, billing, domain allow/deny policy, robots parser, rate limiting, and source-specific adapters are roadmap items—not shipped claims.

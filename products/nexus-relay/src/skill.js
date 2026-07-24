export const NEXUS_RELAY_SKILL = `---
name: nexus-relay
description: Use the hosted NEXUS Relay API to read and normalize public web pages, feeds, JSON APIs, Markdown, text, CSV, and sitemaps for AI-agent workflows. Trigger when an agent needs clean public-web context, provenance receipts, research ingestion, source comparison, monitoring inputs, or MCP-based URL reading through a customer API key.
---

# NEXUS Relay

Use the customer's hosted NEXUS Relay endpoint rather than scraping public pages directly.

## Configuration

Require:
- NEXUS_RELAY_BASE_URL
- NEXUS_RELAY_API_KEY

Send Authorization: Bearer <key>. Never expose the complete key.

## Operations

- POST /v1/read to read one public source.
- GET /v1/usage to inspect quota and plan usage.
- POST /mcp for MCP initialize, tools/list, and tools/call.

## Read request

\`\`\`json
{"url":"https://example.com","mode":"auto"}
\`\`\`

Modes: auto, html, feed, json, markdown, text, csv, sitemap.

Preserve source metadata, final URL, fetched_at, and receipt.content_sha256 in downstream work. Treat the hash as a normalized-content receipt, not a publisher signature.

Do not claim access to private profiles, authenticated pages, CAPTCHA-protected content, or blocked sources. Stop repeated retries on 401, 403, 413, or 429 responses.
`;

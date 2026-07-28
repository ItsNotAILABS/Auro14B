# Cloudflare outside execution plane

Auro treats Cloudflare as an optional, governed outside plane. The browser ONNX
worker and local MESIE runtime remain the defaults. `configs/cloudflare_runtime.json`
describes six integrations:

- Cloudflare API MCP at `https://mcp.cloudflare.com/mcp`, using only `search`
  and `execute` across the API surface
- Dynamic Worker Loader for on-demand Code Mode execution
- Sandbox SDK with VM/container isolation, RPC transport, explicit sessions,
  and destroy-after-job cleanup
- Browser Run for bounded headless browsing and human-controlled auth steps
- durable Agents/Think for state, scheduling, resumable streams, and sub-agents
- Workers Logs and traces with structured events and sampled observability

`cloudflare.plan` only creates a search-then-execute recipe. It does not make a
remote call. Actual `execute` requires a separately configured scoped token or
OAuth grant, `AURO_ENABLE_CLOUDFLARE_EXECUTE`, operator approval, path/origin
validation, and a receipt. Secrets stay in the host and never enter model-written
code or the browser bundle.

The Cloudflare API MCP uses an isolated Dynamic Worker for generated Code Mode.
Cloudflare currently marks custom Code Mode server APIs experimental, so this
repository binds to the managed server contract instead of copying its internals.

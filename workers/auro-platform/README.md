# Auro Sovereign Cloudflare Platform

A deployable Cloudflare-native operator combining a polished web chat, Workers
AI, durable Think/Agents state, Cloudflare API MCP, Dynamic Worker Code Mode,
Browser Run, workspace files, extensions, and Workers Observability.

## Deploy

```bash
npm install
npx wrangler login
npm run deploy
```

That is the deployment command after the one-time `wrangler login`. CI can use
`CLOUDFLARE_API_TOKEN` and run the same `npm run deploy` command.

For automated Cloudflare API MCP access, provide a narrowly scoped token:

```bash
npx wrangler secret put CLOUDFLARE_API_TOKEN
```

The default is inspection and planning only. Mutations require both a deliberate
configuration change and per-turn approval:

```bash
npx wrangler versions secret put CLOUDFLARE_API_TOKEN
npx wrangler deploy --var ALLOW_CLOUDFLARE_MUTATIONS:true
```

Even then, the agent system prompt requires `OPERATOR_APPROVED` on the exact
turn and stops for destructive, billing, authentication, domain-transfer, and
Zero Trust policy changes. Use scoped tokens as the enforceable permission
boundary; prompt instructions are defense in depth, not an authorization system.

## What is included

- Think durable chat, SQLite session state, compaction, workspace and streaming
- Workers AI model selected by `WORKERS_AI_MODEL`
- managed Cloudflare API MCP at `https://mcp.cloudflare.com/mcp`
- unified Code Mode `execute` tool backed by the `LOADER` Dynamic Worker binding
- Browser Run through `BROWSER`, including human Live View support
- dynamic extension loader
- structured Workers Logs and 5% trace sampling
- health and machine-readable platform-discovery endpoints

The separate Cloudflare Sandbox SDK container binding is not enabled in the
single-command base deployment because it requires a paid-plan container image
and lifecycle configuration. Code Mode still executes generated JavaScript in
isolated Dynamic Workers. Add Sandbox as a second execution backend only after
account-level container limits and cleanup policy are chosen.

## Validation notes

`npm run build` performs the production Vite bundle and a strict TypeScript
check. `npm run check` additionally runs Wrangler's deployment dry-run. Current
Cloudflare `workerd` releases do not provide a Windows ARM64 executable, so run
the Wrangler check on Linux, macOS, Windows x64, or CI. This host limitation
does not affect the deployed Worker runtime.

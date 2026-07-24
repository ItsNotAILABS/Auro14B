# Cloudflare Production Envelope

Cloudflare remains the production deployment target for Auro14B.

## Runtime boundaries

- `cloudflare-platform/` owns the edge-facing production Worker and Durable Object envelope.
- `workers/mesie-api/` remains a separately deployable Worker surface where required.
- `native_llm/fabric/` is a portable Node.js control-plane companion. It must communicate with Cloudflare through an explicit URL and token boundary; it must not impersonate a trained model or silently bypass governed execution.

## Required production secrets

Secrets must be configured in Cloudflare or GitHub Actions, never committed:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`
- platform authentication token used by the portable fabric
- Monad/Ethereum gateway credentials when those routes are enabled
- OAuth client secrets when OAuth is enabled

## Deployment gate

A production deployment is complete only when all configured surfaces pass:

1. Worker deployment succeeds from a clean checkout.
2. `/health` and required THESIS/API routes return expected contracts.
3. Durable Object state survives a new request/session.
4. Governed execution denial tests reject unauthorized calls.
5. Wallet and chain gateways pass non-destructive smoke checks.
6. OAuth callback and session persistence pass in the deployed environment.
7. Synthetic users complete the production smoke suite.
8. Evidence artifacts and rollback identifiers are retained.

Repository CI proves code-level readiness only. Live secrets, DNS, OAuth providers, chain gateways, and Cloudflare account resources require deployment-environment evidence before the release can be declared production-complete.

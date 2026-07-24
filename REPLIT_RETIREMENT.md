# Replit Retirement

Replit is not a deployment target for this repository.

## Source of truth

- GitHub is the source-of-truth repository.
- Cloudflare is the production envelope.
- `native_llm/fabric` is an additional portable Node.js runtime and does not replace Cloudflare.

## Compatibility

`scripts/replit_start.sh` is retained only as a temporary compatibility shim. It delegates to `scripts/start_local.sh` and must not contain Replit-specific configuration, packages, secrets, or runtime assumptions.

## Prohibited production dependencies

- `.replit`
- `@replit/*` runtime packages
- Replit OAuth assumptions
- Replit-hosted persistence
- Replit deployment secrets

The automated cutover gate in `scripts/verify_cutover.sh` enforces the repository-level portions of this policy.

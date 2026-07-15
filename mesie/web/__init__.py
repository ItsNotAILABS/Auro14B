"""MESIE Web — Deploy the Spectral Intelligence Engine to raw HTTP.

Exposes the full internal engine bus (validate, match, generate,
embed, reason, fingerprint, etc.) as a zero-dependency Python HTTP
service that runs anywhere: bare metal, containers, serverless.

Quick start:
    mesie-web                    # starts on 0.0.0.0:8080
    mesie-web --port 9000       # custom port
    python -m mesie.web         # module invocation

Endpoints:
    GET  /                      — service metadata
    GET  /health                — health check
    GET  /v1/engines            — list registered engines
    POST /v1/engine/{name}/{action} — call any engine action
    POST /v1/intelligence/reason    — shortcut: reason over a record
    POST /v1/match              — spectral matching
    POST /v1/validate           — record validation
    POST /v1/embed              — embedding generation
    POST /v1/tokenomics/score   — token value scoring
"""

from mesie.web.app import MESIEWebApp
from mesie.web.server import serve

__all__ = ["MESIEWebApp", "serve"]

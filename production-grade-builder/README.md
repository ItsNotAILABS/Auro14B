# Production-Grade Builder

Reusable scaffolding engine that generates production-grade deliverable packets for every surface type in the MESIE ecosystem.

## Supported Deliverable Types

| Kind | Description |
|------|-------------|
| `static-app` | Static HTML/JS/CSS applications |
| `python-service` | Python microservices and libraries |
| `node-service` | Node.js services and packages |
| `benchmark-engine` | Performance and correctness benchmark suites |
| `local-api` | Local REST/gRPC API scaffolds |
| `ci-workflow` | CI/CD pipeline configurations |
| `dataset` | Structured dataset packages |
| `manifest` | System manifests and inventories |
| `proof-pack` | Verification and proof artifacts |
| `research-packet` | Research papers and technical reports |
| `dashboard` | Monitoring and analytics dashboards |
| `docs` | Documentation sites and packages |
| `deploy-scaffold` | Deployment and infrastructure templates |
| `repo-surface` | Repository-level configuration surfaces |

## Usage

```bash
# Scaffold a new deliverable
python production-grade-builder/tools/scaffold.py --kind python-service --name my-service --output ./out

# Run quality gate on any packet
python production-grade-builder/tools/quality_gate.py ./out/my-service
```

## Structure

```
production-grade-builder/
├── README.md
├── PACKET_POLICY.md          # Root production-grade law
├── profiles/                 # Deliverable type profiles (JSON)
├── tools/
│   ├── scaffold.py           # Zero-dependency scaffolding CLI
│   └── quality_gate.py       # Universal quality gate checker
├── registry/
│   └── schema.json           # Registry schema for all deliverables
└── examples/                 # Verified example packets
```

## Production-Grade Law

Every deliverable **must** contain:

1. `README.md` — purpose, usage, verification steps
2. `PACKET_POLICY.md` — applicable standards
3. `manifest.json` — machine-readable metadata
4. `release_manifest.json` — version and checksum
5. `reports/build_report.json` — build/generation report
6. `scores/quality_score.json` — quality metrics
7. `tools/quality_gate.py` — self-contained verification

A deliverable is production-grade when its own `quality_gate.py` exits 0.

# Production-Grade Packet Policy

**Version:** 1.0.0
**Applies to:** All deliverables in the MESIE ecosystem

## Law

Every deliverable, regardless of surface type, must satisfy the following before it can be considered production-grade:

### Required Artifacts

| Artifact | Purpose |
|----------|---------|
| `README.md` | Human-readable description, usage, and verification |
| `PACKET_POLICY.md` | Standards this deliverable adheres to |
| `manifest.json` | Machine-readable metadata (name, version, kind, author, created) |
| `release_manifest.json` | Release info (version, checksum, timestamp) |
| `reports/build_report.json` | Build provenance and generation details |
| `scores/quality_score.json` | Quantified quality metrics |
| `tools/quality_gate.py` | Self-contained verification script (zero external deps) |

### Quality Gate Rules

1. All required artifacts must exist.
2. `manifest.json` must be valid JSON with keys: `name`, `version`, `kind`, `author`, `created`.
3. `quality_score.json` must report `"pass": true`.
4. `tools/quality_gate.py` must exit 0 when run from the packet root.
5. No secrets, credentials, or tokens in any file.
6. README must contain at least one verification command.

### Surface-Specific Extensions

Each deliverable kind may require additional artifacts (e.g., `Dockerfile` for services, `index.html` for static apps). These are defined in the corresponding profile under `profiles/`.

### Enforcement

- The builder scaffolds all required artifacts automatically.
- CI pipelines must run `quality_gate.py` before merge.
- Manual deliverables must pass the same gate before export.

### Versioning

This policy follows semver. Breaking changes increment the major version.

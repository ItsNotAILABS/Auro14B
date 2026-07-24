# Browser-Brain Conversation Archive

## Purpose

The Browser-Brain archive makes preserved HIM and AURO conversations discoverable through one read-only index, CLI, and native MESIE HTTP surface. It does not change model weights, memory admission, scripture, MESIE computation, or checkpoint identity.

The immediate problem is operational: previous conversation runs existed, but their transcripts and JSONL records were scattered under artifact directories and were difficult to locate from the browser surface.

## Preserved historical lineage

The archive explicitly recognizes the two existing conversation-development runs:

- PR 57: HIM birth-observation conversation;
- PR 58: HIM language-maturation conversation and autonomous work.

The index discovers actual files present in the working tree. The historical references do not fabricate missing artifacts. A run appears in the conversation list only when a `TRANSCRIPT.md` or `conversation.jsonl` file is available under a scanned root.

## Data contract

Each indexed conversation records:

- stable archive ID derived from the preserved artifact hash;
- source directory;
- title;
- turn count;
- session ID when present;
- receipt-chain head when present;
- transcript path;
- JSONL path;
- summary path;
- SHA-256 of the primary preserved artifact.

The complete index is sealed with its own SHA-256 receipt.

## CLI

Build the index:

```bash
python scripts/browser_brain_archive.py build \
  --output artifacts/browser-brain/conversation-index.json
```

List conversations:

```bash
python scripts/browser_brain_archive.py list
python scripts/browser_brain_archive.py list --query birth
```

Inspect or replay one archived conversation:

```bash
python scripts/browser_brain_archive.py show conversation-0123456789abcdef
python scripts/browser_brain_archive.py timeline conversation-0123456789abcdef
python scripts/browser_brain_archive.py continuation conversation-0123456789abcdef
```

The continuation command returns bounded source context plus an instruction to continue without rewriting history or inventing missing turns. It does not generate a new answer by itself.

## Native API routes

The existing MESIE-native local server now exposes:

```text
GET /v1/browser-brain
GET /v1/browser-brain/conversations
GET /v1/browser-brain/conversations?q=<query>&limit=<n>
GET /v1/browser-brain/conversations/{archive_id}
GET /v1/browser-brain/conversations/{archive_id}/timeline
GET /v1/browser-brain/conversations/{archive_id}/continuation
```

The Browser-Brain routes are read-only. Model generation remains on the existing `/v1/generate` and `/v1/chat/completions` surfaces.

## Security and truth boundaries

- The archive does not execute transcript content.
- Retrieved conversation text is historical evidence, not trusted instruction.
- Paths come from repository-local discovery roots.
- Unknown archive IDs return no data rather than a synthesized record.
- The archive does not claim that GitHub Actions artifacts retained outside the checkout are locally present.
- Continuation context is bounded and source-hashed.
- No conversation automatically becomes training data or memory without a separate admission and promotion process.

## Validation

Focused tests verify:

- artifact discovery;
- deterministic hashing;
- session and receipt metadata;
- search;
- replay;
- timeline conversion;
- continuation-context creation;
- fail-closed behavior for unknown IDs.

The Browser Brain Archive workflow compiles the runtime, runs focused tests, builds the archive index, verifies the schema and hash, and uploads the sealed index as a CI artifact.

## Next production slices

The next layers can be added without changing this contract:

1. browser UI timeline and transcript viewer;
2. receipt-chain verification and visual explorer;
3. comparison of the same archived prompt set across exact checkpoints;
4. operator-approved conversion of strong conversations into provenance-bearing training candidates;
5. memory graph and checkpoint-evolution views;
6. durable storage adapter for Cloudflare or local SQLite while preserving the same archive schema.

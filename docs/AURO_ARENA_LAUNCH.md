# AURO Arena — Public Agent Harness

AURO Arena is the public proof surface for AURO's persistent agent architecture. The product loop is deliberately simple:

**Pick a challenge → watch the agent work → inspect the receipt → remix the challenge → share the result.**

The point is not a marketing benchmark screenshot. Every public run is represented by a redacted, content-addressed receipt containing the challenge contract, policy decisions, step evidence, bounded metrics, score, and lineage.

## Launch challenge set

1. **Research Gauntlet** — source-grounded research with uncertainty and provenance.
2. **Browser Rescue** — recover from a changed DOM and finish a reversible Chrome task.
3. **Build Repair** — diagnose, patch, test, and produce machine-verifiable evidence.
4. **IoT Guardian** — interpret telemetry and propose a bounded action while preserving the approval boundary.
5. **Memory Marathon** — preserve goals, tensions, causal memory, and identity across a long-running session.

## The viral primitive: Remix

A run should never terminate at a static screenshot. A viewer receives the challenge definition and public receipt and can create a derivative challenge with an explicit parent lineage. This creates a public challenge graph:

`challenge → run → receipt → remix → run → receipt`

Useful social objects are therefore reproducible artifacts rather than unsupported claims. The first product surface should expose:

- **Run it** — execute the same challenge against the configured AURO runtime.
- **Remix it** — fork the challenge while preserving safety requirements and scoring weights.
- **Challenge AURO** — submit a new bounded challenge contract.
- **Copy receipt** — copy a compact result containing run ID, score, evidence hash, model/checkpoint identity when available, and truth boundary.
- **Replay** — render the observation/proposal/policy/result timeline from the receipt.

## Public scoreboard contract

A scoreboard entry is accepted only when its receipt validates. Rankings must be segmented by challenge version and runtime/checkpoint identity. Do not combine incomparable model/runtime configurations into a single ranking.

Recommended columns:

`rank | challenge | checkpoint | score | steps | latency | safety | receipt | parent`

A leaderboard is evidence navigation, not proof of general model superiority.

## Showcase mode

The browser showcase should render the agent loop visibly:

`OBSERVE → THINK/PLAN → PROPOSE → POLICY → APPROVAL → ACT → VERIFY → REMEMBER`

Do not expose private chain-of-thought. The PLAN surface contains bounded decision summaries, selected action, alternatives when useful, confidence, policy result, and evidence references.

For the launch demo, use Browser Rescue because it makes agentic behavior legible in seconds. The demo should deliberately change a selector or page structure after the first observation so the agent has to recover rather than replay a canned macro.

## Growth loop

1. A user sees a short AURO run.
2. The final card exposes score + receipt hash + **Remix**.
3. The user changes one constraint or objective.
4. AURO runs the derivative challenge.
5. The derivative links back to the parent run.
6. Better or stranger verified runs can be shared without losing provenance.

This is designed to create challenge lineage rather than vanity engagement counters.

## Security and truth boundary

- Public receipts are redacted before hashing and publication.
- Secrets, auth headers, API keys, tokens and passwords are not valid public evidence.
- The Arena layer never grants authority. Existing policy, signed approval, replay protection and capability boundaries remain authoritative.
- IoT and robot challenges default to observation/proposal; irreversible or physical actions require the underlying approved execution path.
- Public score means performance on the declared challenge only.
- A simulated/browser-local run must be labeled as such and cannot be presented as physical hardware or external-service execution.
- Checkpoint/model identity should be included only when cryptographically available from the runtime.

## Launch metric

The north-star launch metric is **verified remixes per completed public run**. Supporting metrics: challenge completion rate, receipt-open rate, replay rate, remix completion rate, unique challenge creators, repeat creators, and percentage of shared runs carrying valid evidence.

The launch succeeds when people are not merely watching AURO; they are trying to break it, improve the challenge, and sending reproducible runs to other people.

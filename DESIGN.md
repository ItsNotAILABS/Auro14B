# Auro Operator Console — Design System

## Register

Product UI, not a marketing page. The visual voice is **instrumental, sovereign, warm, and restrained**. It should feel like a well-made scientific instrument: quiet until state changes, precise under pressure.

## Anti-references

No purple gradients, glassmorphism, floating decoration, gradient text, excessive cards, pill-shaped containers, fake metrics, generic “AI-powered” copy, or motion without state meaning.

## Tokens

- Canvas: `#10110f`; surface: `#171915`; raised: `#20231d`
- Ink: `#f1eee4`; muted: `#aaa99e`; quiet: `#777b70`
- Signal: `#d9a441`; success: `#6fb58a`; warning: `#e0a35b`; danger: `#d76d61`
- Rule: `#34382f`; focus: `#f2c66d`
- Display: Georgia / Charter / serif; interface: system sans; evidence: system monospace
- Spacing scale: 4, 8, 12, 16, 24, 32, 48, 64
- Radius: 2 px controls, 6 px panels; never use fully rounded cards

## Hierarchy

One serif page title, one compact runtime status rail, and a two-column workbench. Use rules, alignment, spacing, and typography before adding containers. Capability rows behave like a registry, not collectible cards.

## Components

- `masthead`: identity, canonical lineage, connection state
- `status rail`: verified runtime facts only
- `workbench`: prompt, execution intent, response, council details
- `registry`: filterable capability rows with organ/mode/approval labels
- `receipt room`: chain verification and recent receipt hashes
- `evidence drawer`: structured JSON in monospace with copy control

## Interaction

- Enter inserts a newline; Ctrl/Cmd+Enter submits.
- Execution is off by default. Enabling it reveals the in-memory operator-token input.
- Mutating capability calls require a second explicit approval in the request form.
- Loading, success, denial, and failure must be distinguishable without color alone.
- Honor `prefers-reduced-motion`; any transition must complete within 180 ms.

## Copy

Use direct nouns and verbs: “Ask Auro,” “Verify chain,” “Approval required.” Avoid hype, anthropomorphic certainty, and claims such as “fully autonomous,” “bank-grade,” or “production ready” unless the current evidence record supports them.


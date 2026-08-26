import assert from 'node:assert/strict';
import { GovernedChromeAgentLoop } from '../src/agent-loop.js';

const memoryEvents = [];
const memory = {
  async snapshot() { return { continuity: 'ok', events: memoryEvents.length }; },
  async append(event) { memoryEvents.push(event); },
};

const denied = new GovernedChromeAgentLoop({
  infer: async () => ({ intent: 'book', target: 'ride' }),
  observe: async () => ({ url: 'https://example.invalid', dom: 'ride estimate' }),
  proposeAction: async () => ({ kind: 'book', target: 'ride' }),
  executeAction: async () => ({ ok: true, done: true }),
  memory,
  policy: { evaluate: async () => ({ allowed: true, approvalRequired: true, approved: false }) },
});

const receipt = await denied.step('Get a ride estimate and stop before booking');
assert.equal(receipt.executed, false);
assert.equal(memoryEvents.at(-1).type, 'approval_required');

const executed = new GovernedChromeAgentLoop({
  infer: async () => ({ intent: 'navigate' }),
  observe: async () => ({ url: 'about:blank', dom: '' }),
  proposeAction: async () => ({ kind: 'navigate', url: 'https://example.invalid' }),
  executeAction: async () => ({ ok: true, done: true, observedUrl: 'https://example.invalid' }),
  memory,
  policy: { evaluate: async () => ({ allowed: true, approvalRequired: false, approved: false }) },
});

const result = await executed.run('Navigate to research target');
assert.equal(result.receipts[0].executed, true);
assert.equal(result.receipts[0].result.ok, true);
console.log(JSON.stringify({ ok: true, receipts: result.receipts.length, memoryEvents: memoryEvents.length }));

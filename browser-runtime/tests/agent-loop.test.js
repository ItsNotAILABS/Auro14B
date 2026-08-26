import test from 'node:test';
import assert from 'node:assert/strict';
import { GovernedChromeAgentLoop } from '../src/agent-loop.js';

function makeLoop(decision) {
  const events = [];
  return {
    events,
    loop: new GovernedChromeAgentLoop({
      infer: async () => ({ intent: 'click' }),
      observe: async () => ({ url: 'https://example.com', dom: '<button>Continue</button>' }),
      proposeAction: async () => ({ type: 'click', selector: 'button' }),
      executeAction: async () => ({ done: true }),
      policy: { evaluate: async () => decision },
      memory: {
        snapshot: async () => ({ recent: [] }),
        append: async (event) => events.push(event),
      },
    }),
  };
}

test('blocks execution when policy denies', async () => {
  const { loop, events } = makeLoop({ allowed: false, approvalRequired: false, approved: false });
  const receipt = await loop.step('navigate');
  assert.equal(receipt.executed, false);
  assert.equal(events[0].type, 'policy_denial');
});

test('stops for explicit approval', async () => {
  const { loop, events } = makeLoop({ allowed: true, approvalRequired: true, approved: false });
  const receipt = await loop.step('book ride');
  assert.equal(receipt.executed, false);
  assert.equal(events[0].type, 'approval_required');
});

test('executes allowed approved action', async () => {
  const { loop, events } = makeLoop({ allowed: true, approvalRequired: true, approved: true });
  const receipt = await loop.step('continue');
  assert.equal(receipt.executed, true);
  assert.equal(receipt.result.done, true);
  assert.equal(events[0].type, 'execution');
});

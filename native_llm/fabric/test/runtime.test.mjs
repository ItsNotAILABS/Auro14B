import assert from 'node:assert/strict';
import test from 'node:test';
import { once } from 'node:events';
import { createServer } from '../src/index.mjs';
import { createReceiptChain } from '../src/receipt-chain.mjs';

async function withServer(run) {
  const server = createServer();
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  try {
    await run(`http://127.0.0.1:${address.port}`);
  } finally {
    server.close();
    await once(server, 'close');
  }
}

test('receipt chain verifies after append', () => {
  const chain = createReceiptChain();
  chain.append('test', { ok: true });
  assert.equal(chain.verify(), true);
  assert.equal(chain.history().receipts.length, 1);
});

test('health endpoint reports a valid chain', async () => {
  await withServer(async (baseUrl) => {
    const response = await fetch(`${baseUrl}/health`);
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.ok, true);
    assert.equal(body.receipt_chain_valid, true);
  });
});

test('model family endpoint does not claim trained weights', async () => {
  await withServer(async (baseUrl) => {
    const response = await fetch(`${baseUrl}/v1/model-family`);
    const body = await response.json();
    assert.equal(body.trained_weights_present, false);
    assert.equal(body.status, 'orchestration-only');
  });
});

test('job submission is idempotent', async () => {
  await withServer(async (baseUrl) => {
    const headers = {
      'content-type': 'application/json',
      'idempotency-key': 'same-job',
    };
    const first = await fetch(`${baseUrl}/v1/jobs`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ kind: 'synthetic.smoke' }),
    }).then((response) => response.json());
    const second = await fetch(`${baseUrl}/v1/jobs`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ kind: 'synthetic.smoke' }),
    }).then((response) => response.json());
    assert.equal(first.job.id, second.job.id);
  });
});

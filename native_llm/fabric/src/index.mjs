import http from 'node:http';
import { randomUUID } from 'node:crypto';
import { createPlatformClient } from './platform-client.mjs';
import { createReceiptChain } from './receipt-chain.mjs';
import { createScheduler } from './scheduler.mjs';
import { describeModelFamily } from './model-family.mjs';

const port = Number.parseInt(process.env.PORT ?? '8787', 10);
const host = process.env.HOST ?? '0.0.0.0';
const serviceName = process.env.SERVICE_NAME ?? 'medina-node-fabric';
const startedAt = new Date().toISOString();
const receipts = createReceiptChain();
const platform = createPlatformClient();
const scheduler = createScheduler({ receipts });

function json(response, status, body) {
  const payload = JSON.stringify(body);
  response.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(payload),
    'cache-control': 'no-store',
  });
  response.end(payload);
}

async function readJson(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  if (chunks.length === 0) return {};
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

export function createServer() {
  return http.createServer(async (request, response) => {
    const requestId = request.headers['x-request-id'] ?? randomUUID();
    const url = new URL(request.url ?? '/', `http://${request.headers.host ?? 'localhost'}`);

    try {
      if (request.method === 'GET' && url.pathname === '/health') {
        return json(response, 200, {
          ok: true,
          service: serviceName,
          started_at: startedAt,
          request_id: requestId,
          receipt_chain_valid: receipts.verify(),
          platform: platform.describe(),
        });
      }

      if (request.method === 'GET' && url.pathname === '/v1/model-family') {
        return json(response, 200, describeModelFamily());
      }

      if (request.method === 'GET' && url.pathname === '/v1/receipts') {
        return json(response, 200, receipts.history());
      }

      if (request.method === 'GET' && url.pathname === '/v1/jobs') {
        return json(response, 200, { jobs: scheduler.list() });
      }

      if (request.method === 'POST' && url.pathname === '/v1/jobs') {
        const body = await readJson(request);
        const job = scheduler.submit({
          kind: body.kind,
          payload: body.payload,
          idempotencyKey: request.headers['idempotency-key'],
        });
        return json(response, 202, { job, request_id: requestId });
      }

      if (request.method === 'POST' && url.pathname === '/v1/platform/request') {
        const body = await readJson(request);
        const result = await platform.request(body.path, {
          method: body.method,
          body: body.body,
          headers: body.headers,
        });
        const receipt = receipts.append('platform.request', {
          request_id: requestId,
          path: body.path,
          status: result.status,
        });
        return json(response, result.status, { ...result, receipt });
      }

      return json(response, 404, {
        error: 'not_found',
        request_id: requestId,
      });
    } catch (error) {
      const receipt = receipts.append('request.error', {
        request_id: requestId,
        message: error instanceof Error ? error.message : String(error),
      });
      return json(response, 500, {
        error: 'internal_error',
        request_id: requestId,
        receipt_id: receipt.id,
      });
    }
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const server = createServer();
  server.listen(port, host, () => {
    const receipt = receipts.append('service.started', { host, port, service: serviceName });
    console.log(JSON.stringify({ event: 'service.started', host, port, receipt }));
  });

  const shutdown = (signal) => {
    receipts.append('service.stopping', { signal });
    server.close((error) => {
      if (error) {
        console.error(error);
        process.exitCode = 1;
      }
    });
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

import { parentPort } from 'node:worker_threads';
import { createHash } from 'node:crypto';

parentPort.on('message', async ({ id, task, payload = {} }) => {
  try {
    let result;
    if (task === 'process') {
      const text = String(payload.text || '');
      result = { chars: text.length, words: text.trim() ? text.trim().split(/\s+/).length : 0, sha256: createHash('sha256').update(text).digest('hex') };
    } else if (task === 'crawl') {
      const response = await fetch(String(payload.url), { redirect: 'follow' });
      const text = (await response.text()).slice(0, Number(payload.maxChars || 200000));
      result = { url: response.url, status: response.status, text, sha256: createHash('sha256').update(text).digest('hex') };
    } else throw new Error(`Unknown outside-worker task: ${task}`);
    parentPort.postMessage({ id, ok: true, task, result });
  } catch (error) { parentPort.postMessage({ id, ok: false, task, error: String(error?.message || error) }); }
});

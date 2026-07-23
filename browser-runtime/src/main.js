import './style.css';
import { BrowserWorkerPool } from './worker-pool.js';

const worker = new Worker(new URL('./model-worker.js', import.meta.url), { type: 'module' });
const pending = new Map();
const output = document.querySelector('#output');
const taskWorkers = new BrowserWorkerPool();

worker.onmessage = ({ data }) => {
  const resolve = pending.get(data.id);
  if (resolve) {
    pending.delete(data.id);
    resolve(data);
  }
};

function infer(model, prompt) {
  const id = crypto.randomUUID();
  return new Promise((resolve) => {
    pending.set(id, resolve);
    worker.postMessage({ id, model, prompt });
  });
}

document.querySelector('#inference').addEventListener('submit', async (event) => {
  event.preventDefault();
  output.textContent = 'Loading local ONNX model...';
  const response = await infer(
    document.querySelector('#model').value.trim(),
    document.querySelector('#prompt').value.trim(),
  );
  output.textContent = JSON.stringify(response, null, 2);
});

output.textContent = 'Ready. Models load only from /models/ and WASM only from /wasm/.';
taskWorkers.run('process', { text: 'Auro MESIE Sovereign worker pool ready' }).then((receipt) => {
  output.textContent += `\nTask workers ready: ${receipt.result.sha256.slice(0, 12)}`;
});

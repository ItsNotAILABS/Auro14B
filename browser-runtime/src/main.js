import './style.css';

const worker = new Worker(new URL('./model-worker.js', import.meta.url), { type: 'module' });
const pending = new Map();
const output = document.querySelector('#output');

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

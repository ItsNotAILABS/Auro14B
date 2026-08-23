import './style.css';
import { BrowserWorkerPool } from './worker-pool.js';
import { SHOWCASE_SCENARIOS, buildShowcasePrompt } from './showcase.js';

const worker = new Worker(new URL('./model-worker.js', import.meta.url), { type: 'module' });
const pending = new Map();
const output = document.querySelector('#output');
const telemetry = document.querySelector('#telemetry');
const taskWorkers = new BrowserWorkerPool();
const scenarioSelect = document.querySelector('#scenario');

for (const scenario of SHOWCASE_SCENARIOS) {
  const option = document.createElement('option');
  option.value = scenario.id;
  option.textContent = scenario.title;
  scenarioSelect.appendChild(option);
}

worker.onmessage = ({ data }) => {
  const entry = pending.get(data.id);
  if (entry) {
    pending.delete(data.id);
    entry.resolve(data);
  }
};

function infer(model, prompt, maxNewTokens = 256) {
  const id = crypto.randomUUID();
  const started = performance.now();
  return new Promise((resolve) => {
    pending.set(id, {
      resolve: (data) => resolve({ ...data, clientLatencyMs: Math.round((performance.now() - started) * 1000) / 1000 }),
    });
    worker.postMessage({ id, model, prompt, maxNewTokens });
  });
}

function selectedScenario() {
  return SHOWCASE_SCENARIOS.find((item) => item.id === scenarioSelect.value) || SHOWCASE_SCENARIOS[0];
}

scenarioSelect.addEventListener('change', () => {
  const scenario = selectedScenario();
  document.querySelector('#prompt').value = scenario.prompt;
  document.querySelector('#capability').textContent = scenario.capability;
});

document.querySelector('#inference').addEventListener('submit', async (event) => {
  event.preventDefault();
  const scenario = selectedScenario();
  const model = document.querySelector('#model').value.trim();
  const prompt = buildShowcasePrompt(document.querySelector('#prompt').value.trim(), scenario);
  output.textContent = 'Loading local model and running inference...';
  telemetry.textContent = 'Running';
  const response = await infer(model, prompt, Number(document.querySelector('#tokens').value || 256));
  output.textContent = JSON.stringify(response, null, 2);
  telemetry.textContent = response.ok
    ? `local=${model} · latency=${response.clientLatencyMs}ms · remoteModels=false`
    : `failed · latency=${response.clientLatencyMs}ms`;
});

const first = SHOWCASE_SCENARIOS[0];
document.querySelector('#prompt').value = first.prompt;
document.querySelector('#capability').textContent = first.capability;
output.textContent = 'Ready. Models load only from /models/; remote model loading is disabled.';
telemetry.textContent = 'Local-only browser inference boundary active';

taskWorkers.run('process', { text: 'Auro MESIE Sovereign worker pool ready' }).then((receipt) => {
  document.querySelector('#worker-status').textContent = `Task workers ready · ${receipt.result.sha256.slice(0, 12)}`;
});

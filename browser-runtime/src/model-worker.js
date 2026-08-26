import { env, pipeline } from '@huggingface/transformers';

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = '/models/';
env.backends.onnx.wasm.wasmPaths = '/wasm/';

const pipelines = new Map();

async function getGenerator(model) {
  let entry = pipelines.get(model);
  if (entry) return entry;
  const device = globalThis.navigator?.gpu ? 'webgpu' : 'wasm';
  const started = performance.now();
  const generator = await pipeline('text-generation', model, {
    local_files_only: true,
    device,
    dtype: 'q8',
  });
  entry = { generator, device, loadMs: Math.round((performance.now() - started) * 1000) / 1000 };
  pipelines.set(model, entry);
  return entry;
}

self.onmessage = async ({ data }) => {
  const { id, model, prompt, maxNewTokens = 96 } = data;
  try {
    if (!model || !prompt) throw new Error('model and prompt are required');
    const entry = await getGenerator(model);
    const started = performance.now();
    const result = await entry.generator(prompt, {
      max_new_tokens: Math.max(1, Math.min(Number(maxNewTokens) || 96, 1024)),
      do_sample: false,
    });
    const inferenceMs = Math.round((performance.now() - started) * 1000) / 1000;
    self.postMessage({
      id,
      ok: true,
      result,
      telemetry: {
        engine: 'transformers.js',
        device: entry.device,
        dtype: 'q8',
        loadMs: entry.loadMs,
        inferenceMs,
        localModelsOnly: true,
        remoteModelsAllowed: false,
      },
    });
  } catch (error) {
    self.postMessage({
      id,
      ok: false,
      error: String(error?.message || error),
      telemetry: { localModelsOnly: true, remoteModelsAllowed: false },
    });
  }
};

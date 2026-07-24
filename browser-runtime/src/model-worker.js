import { env, pipeline } from '@huggingface/transformers';

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = '/models/';
env.backends.onnx.wasm.wasmPaths = '/wasm/';

const pipelines = new Map();

self.onmessage = async ({ data }) => {
  const { id, model, prompt, maxNewTokens = 96 } = data;
  try {
    let generator = pipelines.get(model);
    if (!generator) {
      generator = await pipeline('text-generation', model, { local_files_only: true });
      pipelines.set(model, generator);
    }
    const result = await generator(prompt, { max_new_tokens: maxNewTokens, do_sample: false });
    self.postMessage({ id, ok: true, result });
  } catch (error) {
    self.postMessage({ id, ok: false, error: String(error?.message || error) });
  }
};

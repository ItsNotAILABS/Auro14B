# Auro Sovereign Browser Runtime

This workspace runs compatible Auro ONNX exports in a Web Worker with
Transformers.js. It explicitly enables local models, disables Hugging Face Hub
model downloads, loads models from `/models/`, and loads ONNX Runtime WASM from
`/wasm/`.

```bash
npm install
npm run build
```

Place each Transformers.js-compatible export under
`public/models/<model-id>/` and the ONNX Runtime Web binaries under
`public/wasm/`. Model directories normally include tokenizer/config files and
an `onnx/` directory. The runtime never silently falls back to a hosted model.

For a Hugging Face-compatible local PyTorch checkpoint, install Optimum ONNX
and export it before packaging:

```bash
python -m pip install "optimum[onnx]"
optimum-cli export onnx --model ./checkpoints/auro-hf --task text-generation-with-past --dynamo ./browser-runtime/public/models/auro-14b-browser
python scripts/prepare_browser_model.py --model-dir browser-runtime/public/models/auro-14b-browser
```

The current MESIE SpectralGPT checkpoint format is not automatically a Hugging
Face Transformers architecture. It requires a compatible export adapter before
the Optimum command can consume it; the preparation script deliberately rejects
directories without real ONNX model files.

The companion Auro API reports embedded, local, and explicitly configured cloud
compute planes through `compute.engines`. Configure cloud endpoints with
`AURO_CLOUD_ENGINES_JSON`; API keys are referenced by environment-variable name
and are never embedded in the browser bundle. Embedded browser inference is the
default and there is no automatic remote fallback.

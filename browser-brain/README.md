# HIM Embedded Browser Brain

This package is the browser-local envelope for HIM: local Transformers.js inference, IndexedDB memory, security monitoring, a knowledge graph, autonomous research organs, governed training-data nomination, and document generation.

It also includes a decentralized same-origin compute mesh across extension contexts/tabs, WebGPU-aware task selection, durable dependency workflows, and an optional governed bridge to HIM's Python runtime. The mesh has no central server and uses `BroadcastChannel`; it does not claim internet-scale distributed training.

## Sovereignty rules

- Transformers.js is configured with `env.allowRemoteModels = false`.
- Model files must exist under `/models/ItsNotAILABS/HIM-native-onnx/`.
- WASM runtime assets must be self-hosted under `/vendor/transformers/`.
- Browser research updates memory and the graph immediately. Base-weight training candidates require human approval before export.
- No secret is intentionally stored. IndexedDB is origin-local storage, not an encrypted vault.
- Local Python access is an optional Chrome permission limited to `127.0.0.1:8090`; execution still requires the separate execution token.

## Use

```js
import { EmbeddedBrain } from './src/brain.js';

const him = new EmbeddedBrain();
await him.awaken();
const response = await him.think('Map NOVA to HIM with evidence.');
const research = await him.research.run('Study local model evaluation failures.');
await him.research.nominateForTraining(research, { approved: true });
const trainingRows = await him.research.exportApprovedTraining();
```

## Model boundary

Transformers.js loads local ONNX artifacts first. Because `HIM-native-v0` uses the repository's auditable NumPy context-MLP format, the extension includes a local JavaScript float32 inference fallback for that checkpoint. It does not label the fallback as Transformers.js. The next checkpoint promotion must export a supported causal architecture to ONNX, place all artifacts under the local model directory, run a browser generation test with networking disabled, and hash the resulting bundle.

Build the unpacked Chrome extension with `npm run build`, then load the generated `dist/` directory from `chrome://extensions` using **Load unpacked**. No host permissions are requested.

## Verify

```bash
npm install
npm test
npm run check
```

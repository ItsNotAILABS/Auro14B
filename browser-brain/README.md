# HIM Embedded Browser Brain

This package is the browser-local envelope for HIM: local Transformers.js inference, IndexedDB memory, security monitoring, a knowledge graph, autonomous research organs, governed training-data nomination, and document generation.

## Sovereignty rules

- Transformers.js is configured with `env.allowRemoteModels = false`.
- Model files must exist under `/models/ItsNotAILABS/HIM-native-onnx/`.
- WASM runtime assets must be self-hosted under `/vendor/transformers/`.
- Browser research updates memory and the graph immediately. Base-weight training candidates require human approval before export.
- No secret is intentionally stored. IndexedDB is origin-local storage, not an encrypted vault.

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

Transformers.js loads ONNX artifacts. `HIM-native-v0` currently uses the repository's auditable NumPy context-MLP format; it is not falsely labeled as Transformers.js compatible. The next checkpoint promotion must export a supported causal architecture to ONNX, place all artifacts under the local model directory, run a browser generation test with networking disabled, and hash the resulting bundle.

## Verify

```bash
npm install
npm test
npm run check
```

# Feline-Inspired Neuromorphic Architecture

## Status

This is a **runtime cognitive-control architecture** for HIM/NOVA. It is not a biological cat-brain simulation, does not change language-model checkpoint weights, and does not establish consciousness or neuroscience equivalence.

The implementation lives in:

- `auro_native_llm/brain/feline_neuromorphic.py`
- `auro_native_llm/brain/neuromorphic_brain.py`
- canonical export: `auro_native_llm.brain.HIMBrain`

## Why feline-inspired

The design borrows computational motifs historically studied in feline sensory and orienting systems: hierarchical visual processing, recurrent context, feature-selective sparse responses, inhibition, rapid orienting pathways, and adaptive neural response thresholds. The software uses those motifs as engineering priors rather than attempting cell-by-cell biological replication.

## Architecture

```text
observation
   |
   v
44-region HIM controller
   |  salience / anomaly / region activation
   v
feline-inspired neuromorphic substrate
   |-- leaky membrane integration
   |-- sparse spike thresholding
   |-- refractory windows
   |-- recurrent synaptic traces
   |-- population inhibitory tone
   |-- adaptive thresholds
   |-- bounded local plasticity
   |-- orienting burst path
   |-- compute-energy accounting
   v
neuromorphic cycle receipt
   |
   +--> HIM snapshot / analytics
   +--> energy pressure may downgrade execute -> deliberate
   X--> never authorizes execution
```

## Orienting pathway

The runtime's fast orienting path is expressed through the existing region namespace:

`SC -> THL_L / THL_R -> LC -> V1`

This approximates an engineering pattern of fast sensory orientation before slower deliberative integration. It is not claimed to be an anatomically complete feline pathway.

## Spiking state

Each region carries:

- membrane potential
- adaptive threshold
- short synaptic trace
- refractory counter
- bounded local synaptic gain

A spike is emitted only when integrated membrane state exceeds the current dynamic threshold. Recent population activity increases inhibitory tone, which suppresses runaway dense firing.

## Energy model

Energy is tracked in **CEU: normalized compute-energy units**.

CEU is deliberately not reported as joules. Physical energy claims require a calibrated hardware backend measuring wall power or accelerator telemetry. Current CEU accounting charges separately for:

- idle region maintenance
- membrane integration
- spike emission
- plasticity updates

Energy pressure reduces available integration gain and raises effective thresholds. It therefore provides a homeostatic compute budget rather than an ornamental metric.

## Plasticity

The runtime uses a bounded local Hebbian-style rule: salient coincident drive can increase a region's synaptic gain slightly; inactive gains decay toward neutral. The rule modifies only transient runtime state. It does not silently modify AURO checkpoint tensors.

## Authority boundary

Neuromorphic state is advisory. Spikes, salience, or energy cannot grant tool authority. In the canonical `HIMBrain`, excessive energy pressure is allowed to downgrade `execute` to `deliberate`, but it can never upgrade `answer` or `deliberate` into `execute`.

Server-authoritative approval remains the only mutating execution boundary.

## Evidence gates

`tests/test_feline_neuromorphic.py` verifies:

- sparse spike and energy accounting
- refractory suppression
- adaptive thresholds
- inhibitory feedback after dense activity
- explicit CEU/non-joule claim boundary
- canonical 44-region integration
- no execution-authority escalation

`.github/workflows/neuromorphic-brain-gate.yml` compiles and runs the neuromorphic contract on pull requests and `main`.

## Next measurable upgrades

Future iterations should be promoted only with receipts for measurable effects:

1. per-region spike-rate stability across long runs
2. energy-per-cycle comparison against non-spiking control logic
3. latency impact on production responses
4. task-regression checks with neuromorphic control enabled/disabled
5. per-layer/model coupling experiments without altering checkpoint claims
6. hardware-calibrated energy telemetry before any joule or efficiency claim
7. STDP-style timing experiments only when they outperform the current bounded plasticity rule under controlled tests

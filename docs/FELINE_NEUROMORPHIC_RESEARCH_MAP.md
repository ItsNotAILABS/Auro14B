# Feline Neuromorphic Research-to-Code Map

This note records the scientific inspiration behind the AURO/HIM neuromorphic control layer and keeps that inspiration separate from implementation claims.

## 1. Hierarchical visual feature processing

**Research basis:** Hubel & Wiesel, *Receptive fields, binocular interaction and functional architecture in the cat's visual cortex*, Journal of Physiology 160 (1962), 106-154. DOI: `10.1113/jphysiol.1962.sp006837`.

The classic cat visual-cortex work established structured receptive-field organization and progressively more complex cortical visual responses.

**Engineering mapping:**

```text
V1 -> V2V3 -> ITG_L / FFA -> PPC -> DLPFC
```

Implemented as weighted excitatory `Synapse` edges in `feline_neuromorphic.py`.

**Not claimed:** the software graph is not a faithful anatomical reconstruction and the named HIM regions are engineering abstractions.

## 2. Superior-colliculus orienting path

**Research basis:** cat superior-colliculus studies show an important role for SC in visual detection/orienting responses. Fitzmaurice et al., *Visual detection deficits following inactivation of the superior colliculus in the cat*, Visual Neuroscience (2003/2004 publication record), DOI `10.1017/S095252380320609X`.

Additional cat SC studies report orientation/direction-selective visual responses and multisensory/orienting roles.

**Engineering mapping:**

```text
SC -> THL_L / THL_R -> V1
SC -> LC -> V1 / DLPFC_R
```

A novelty/salience threshold can create an `orienting_burst`, temporarily increasing input drive on the fast orienting regions.

**Not claimed:** this is not a complete tectothalamic/cortical pathway model.

## 3. Excitation / inhibition balance

**Research basis:** Populin, *Anesthetics Change the Excitation/Inhibition Balance That Governs Sensory Processing in the Cat Superior Colliculus*, Journal of Neuroscience 25(25), 5903-5914 (2005), DOI `10.1523/JNEUROSCI.1147-05.2005`.

**Engineering mapping:**

- global population inhibitory tone rises with recent firing density;
- explicit inhibitory synapses provide targeted braking;
- current graph includes conflict/action/threat braking edges;
- inhibitory current directly reduces membrane integration.

**Not claimed:** numerical weights are engineering hyperparameters, not measured feline synaptic strengths.

## 4. Sparse event-driven computation

Sparse/event-driven activation is used here as an engineering strategy compatible with neuromorphic/spiking computation: inactive regions incur only a small idle CEU charge while spike and synaptic events incur additional CEU cost.

**Engineering mapping:**

- leaky membrane integration;
- thresholded spikes;
- refractory periods;
- adaptive thresholds;
- event-gated synaptic transmission;
- event-gated plasticity;
- measured spike rate and sparsity per cycle.

**Evidence boundary:** CEU is a normalized software cost model. Hardware energy efficiency is unverified until CEU is calibrated against measured accelerator/CPU wall-power telemetry.

## 5. Synaptic plasticity and recurrence

The runtime uses bounded local and edge-level adaptation to preserve recurrence without silently training the language model.

**Engineering mapping:**

- short region traces model recent activity;
- excitatory source/target coincidence can raise edge gain;
- unused gains decay toward `1.0`;
- gains are capped;
- state is atomically persisted and hash sealed across restarts.

**Not claimed:** this is not biological STDP and does not modify AURO checkpoint tensors.

## 6. Model integration

`PersonaRuntime` wraps the model orchestrator with `NeuromorphicAwareGenerator`.

Each model call receives a compact telemetry block containing:

- spike rate
- sparsity
- inhibitory tone
- synaptic-event count
- CEU pressure
- orienting-burst state
- active region IDs

The original persona/system instruction remains first. The telemetry block is explicitly marked `telemetry_only` and `can_authorize_execution=false`.

## 7. Governance invariant

Neuromorphic dynamics can influence attention, pacing, model selection context, or downgrade an overloaded `execute` route to `deliberate`.

They **cannot**:

- issue tool approval;
- bypass server-signed execution grants;
- modify checkpoint weights;
- establish consciousness;
- establish biological equivalence;
- justify physical energy-efficiency claims without hardware telemetry.

## 8. Evidence artifacts

The neuromorphic gate is designed to produce:

- `evidence/neuromorphic-benchmark.json`
- `evidence/neuromorphic-brain.json`

The benchmark includes quiet, visual-orienting, executive, and sustained-overload scenarios with spike rate, sparsity, synaptic-event density, CEU mean/p95, energy pressure, orienting bursts, and a normalized dense-control reference.

# Changelog

## [Unreleased] — Auro model family spine

### Added
- **Scriptural Systems Architecture** — full libraries in `auro_native_llm/scripture/` (canon, gates, governance, executor, memory, substrate, train hooks)
- **Executable canon** — `native_llm/scripture/AURO_CANON.v1.json` (fail-closed doctrine for generate/train/dispatch/claim)
- **Scriptural memory** — doctrine-tagged embeddings injected into LM context; persisted under `deliverables/auro_scripture/`
- **Inner AI governance** — refuses denied intents / false weight claims / cloud primary before LM runs
- **Docs** — `docs/SCRIPTURAL_SYSTEMS_ARCHITECTURE.md`
- **CLI** — `auro-scripture` / `python -m auro_native_llm.scripture.cli`
- **Auro native LLM family** — 2B / 4B / 8B / 14B / 100B lanes (`native_llm/configs/auro_family.json` + `family/*.json`)
- **MESIE compute plane** — all native inference via `auro_native_llm/mesie_compute.py` (torch spectral → foundation → NeuroCore → FFT)
- **AuroLanguageModel** — first-class text LM on MESIE SpectralGPT + MoE + meaning + spectral fusion
- **AuroNativeModel / AuroNativeRuntime** — executable native lanes + multi-embedded dispatch that *runs* on MESIE
- **Live serve** — `python -m auro_native_llm.serve.local --live` (stdlib HTTP, no cloud LLM)
- **Multi-embedded sub-agents** — larger lanes host smaller lanes by role (`auro_native_llm/subagents.py`)
- **Polyglot family types** — Python `auro_native_llm/types.py`, Julia `bindings/julia/AuroFamily/`, Haskell `bindings/haskell/AuroFamily.hs`
- **CLI** — `auro-family`, `auro-lm`, `auro-scripture`
- **Tests** — `tests/test_auro_model_family.py`, `tests/test_auro_native_mesie.py`, `tests/test_auro_language_model.py`, `tests/test_auro_scripture.py`

## [0.4.0] - 2026-06-07

### Added
- **Foundation Pretraining Suite** — Masked Spectral Modeling, InfoNCE Contrastive Learning, Temporal Prediction, and unified loss orchestration
- **3D Connectome Brain Environment** — 44 real brain regions with MNI coordinates, 68 white-matter tract connections, and neural signal propagation simulation
- **Miniverse Nesting** — Recursive containment with scale-bridging and downward attention via SpectralAttentionAdapter
- **MAESI SDK v1.1** — Technical library, research knowledge catalog, FastSpectralCompute, fingerprint pipeline
- **Edge API** — Cloudflare Worker API for spectral validation and matching at the edge
- **Desktop Application** — Cross-platform Electron app with builds for Windows, macOS, and Linux
- **Polyglot Bindings** — Rust, Julia, TypeScript, and Motoko (Internet Computer) integrations
- **Intelligence AI Protocols** — Multi-level autonomous reasoning from passive observation to fully autonomous decision-making
- **Spectral Transformer Pipeline** — End-to-end transformer encoder with configurable tokenization and multi-head spectral attention
- **Observation Encoder** — Raw world → spectra → MESIE embedding → agent observation vector pipeline
- **Digital Twin Simulation** — Physics-based environments with RL reward signals
- **Spectral Memory Store** — k-NN retrieval over spectral embeddings with event/time filtering
- **Reasoning Datasets** and **Training Recipe** modules for foundation model development
- Zenodo DOI: [10.5281/zenodo.20598320](https://doi.org/10.5281/zenodo.20598320)

### Changed
- Bumped package version from 0.3.0 to 0.4.0
- All transformer and intelligence components implemented in pure NumPy for portability
- Optional PyTorch/HuggingFace Transformers integration via `[intelligence]` or `[full]` install extras

## [0.2.2] - 2026-06-05

### Added
- MAESI SDK v1.1: technical library, research knowledge catalog, `FastSpectralCompute`, `MAESIClient`
- Fingerprint pipeline (TF, salient, LSH, ANN) and `scripts/run_maesi_sdk.py`

## [0.2.1] - 2026-06-04

### Added
- Internal API bus (`mesie/internal_api`) for cross-engine communication
- Nine processing engines: embedding, matching, generation, validation, intelligence, control, movement, workflow, logic
- Octopus engineering controller with eight arms (sense, embed, match, move, control, workflow, logic, memory)
- User spectral library loader wired to EMBED arm (`mesie/library/user_corpus.py`)
- Cloudflare Worker API (`workers/mesie-api`), bundled data package, laptop/octopus docs and scripts

### Changed
- `embed_my_library.py` saves index and optional `--octopus` demo cycle
- Reference datasets clipped/validated for level-6 compliance

All notable changes to MESIE will be documented in this file.

## [0.2.2] - 2026-06-05

### Added
- MAESI SDK v1.1: technical library, research knowledge catalog, `FastSpectralCompute`, `MAESIClient`
- Fingerprint pipeline (TF, salient, LSH, ANN) and `scripts/run_maesi_sdk.py`

## [Unreleased]

### Added
- **Foundation Pretraining Suite** (`mesie/pretraining/foundation_objectives.py`)
  - Masked Spectral Modeling with three masking strategies (random, contiguous, band)
  - InfoNCE Contrastive Learning with full augmentation pipeline (Gaussian noise, frequency masking, amplitude scaling, circular shifts)
  - Temporal Prediction with configurable context aggregation (weighted, mean, last, concatenated)
  - Unified `FoundationObjectiveSuite` orchestrating all losses with configurable weights
- **Observation Encoder** (`mesie/pretraining/observation_encoder.py`)
  - Encodes raw world → spectra → MESIE embedding → agent observation vector
  - Multi-modality support (spectral, state, semantic) with configurable normalization and weighting
- **Digital Twin Simulation** (`mesie/pretraining/digital_twin.py`)
  - Physics-based environments (rotating machinery, structural elements, power systems, robotic joints, fluid systems)
  - RL reward signals tied to resonance avoidance, drift minimization, coherence maintenance, anomaly detection
- **Spectral Memory Store** (`mesie/pretraining/spectral_memory.py`)
  - k-NN retrieval over spectral embeddings
  - Event/time filtering and lineage reconstruction
  - Importance-weighted memory consolidation
- **3D Connectome Brain Environment** (`mesie/connectome/`)
  - 44 real brain regions with MNI 3D coordinates across 10 functional systems
  - 68 biologically-inspired white-matter tract connections
  - 3D neural simulation engine with signal propagation using ~6 mm/ms conduction velocity
  - Global coherence metrics and system-level activation tracking
  - Full 3D state export for visualization
- **Miniverse Nesting** (`mesie/cognitive/miniverse.py`)
  - Recursive containment: outer memory objects contain inner MESIE engines that re-activate on query
  - Scale-bridging protocol: MatchResult → MemoryEntry promotion with resonance-based importance
  - Downward attention: outer layer selects which inner micro-patterns to amplify via SpectralAttentionAdapter
- Example script: `examples/08_3d_connectome_brain.py`
- Test suites: `tests/test_foundation_objectives.py`, `tests/test_pretraining.py`, `tests/test_connectome.py`, `tests/test_miniverse.py`

## [0.2.0] - 2026-06-03

### Added
- **Intelligence AI Protocols** (`mesie.ai.intelligence_protocols`)
  - `IntelligenceProtocol` — orchestrator for autonomous spectral reasoning with configurable intelligence levels (passive, reactive, adaptive, predictive, autonomous)
  - `IntelligenceConfig` — configuration for reasoning behavior, memory, and attention settings
  - `ReasoningResult` — structured output with conclusions, confidence, evidence, and recommended actions
  - `SpectralMemoryBuffer` — episodic memory with importance-weighted retention and similarity-based retrieval
  - `AttentionFocusModule` — multi-head attention mechanism that learns informative frequency regions
  - `ReasoningStrategy` enum — statistical, pattern matching, anomaly detection, causal inference, ensemble

- **Spectral Transformer Pipeline** (`mesie.ai.transformer_pipeline`)
  - `SpectralTransformerPipeline` — end-to-end transformer encoder for spectral sequences
  - `TransformerConfig` — configurable architecture (d_model, n_heads, n_layers, feedforward dim, pooling)
  - `SpectralTokenizer` — converts continuous spectra to discrete tokens (frequency bins, wavelets, patches)
  - `PositionalEncoder` — sinusoidal and learnable positional encodings for spectral sequences
  - `MultiHeadSpectralAttention` — scaled dot-product multi-head attention optimized for frequency data
  - `TransformerEncoderLayer` — full encoder block with attention, feed-forward, residual connections, and layer norm
  - `TransformerOutput` — structured output with embeddings, pooled vectors, and attention maps

- New install extras: `[intelligence]` for full AI protocol + transformer stack
- Added `transformers>=4.30` and `torch>=2.0` as optional dependencies
- Tests for intelligence protocols and transformer pipeline

### Changed
- Bumped version from 0.1.0 to 0.2.0
- Updated `.zenodo.json` with full Zenodo release metadata including references, subjects, and related identifiers
- Updated `CITATION.cff` with new version and expanded keywords
- Updated `pyproject.toml` description and keywords to reflect transformer and intelligence capabilities
- Extended `[full]` and `[ai]` optional dependency groups to include transformers and torch

## [0.2.1] - 2026-06-04

### Fixed
- Bundled reference PSD/FAS JSON: clip negative amplitudes; set component units; all references reach validation level 6
- `data` package included correctly in PyPI wheel (`data/__init__.py` + JSON)

### Added
- `scripts/orbital_edge_50d_analysis.py` — 50d backward + 50d forward orbital-edge matching demo
- `scripts/determinism_benchmark.py` — timing and seed reproducibility proof
- `tests/test_bundled_data.py`, `tests/test_bundled_training_smoke.py`
- Cloudflare Worker scaffold `workers/mesie-api/`

## [0.1.0] - 2024-01-01

### Added
- Initial public research release
- Core spectral record data model (MultiElementRecord, SpectralComponent, SpectralMetadata)
- Spectral validation with multi-level checks
- Normalization, interpolation, and smoothing
- Spectral matching engine with composite scoring
- PSD, FAS, and RotDnn generation
- Electro-spectral feature extraction
- Node topology mapping and lineage
- Spectral embedding vectorizers
- Cognitive architecture adapters
- Example scripts and test suite
- Documentation skeleton
- Zenodo metadata and CITATION.cff

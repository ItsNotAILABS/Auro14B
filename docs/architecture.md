# MESIE Architecture

## System Overview

MESIE is organized as a modular Python package with clear separation of concerns:

```
mesie/
├── core/          — Data structures and configuration
├── io/            — Loading and exporting records
├── processing/    — Normalization, interpolation, smoothing
├── matching/      — Spectral comparison and scoring
├── generation/    — Synthetic spectrum generation
├── features/      — Feature extraction (electro-spectral, resonance, coherence)
├── topology/      — Node mapping and lineage tracking
├── embeddings/    — Vector representations for AI
├── cognitive/     — Adapters for cognitive architectures
├── validation/    — Multi-level validation
└── visualization/ — Plotting and diagrams
```

## Data Flow

1. **Input** → Records loaded from JSON, CSV, arrays, or DataFrames
2. **Validation** → Multi-level checks (file, spectral, component, format, embedding-readiness)
3. **Processing** → Normalization, interpolation, smoothing
4. **Features** → Electro-spectral signatures, band energy, resonance, coherence
5. **Matching** → Multi-metric comparison with composite scoring
6. **Generation** → PSD/FAS/RotDnn synthetic output
7. **Embeddings** → Fixed-size vectors for ML/AI pipelines
8. **Cognitive** → Memory, attention, anomaly, and state adapters

## Core Design Principles

- Spectra are structured objects, not flat arrays
- Every operation preserves lineage and metadata
- Optional dependencies (scipy, pandas, networkx) degrade gracefully
- Public API is minimal and consistent
- Internal modules are testable in isolation

## Key Abstractions

### MultiElementRecord
The primary data object containing components, metadata, and topology.

### SpectralComponent
A single frequency-amplitude pair with metadata and node linkage.

### MatchResult
Comprehensive matching output with composite score and metric breakdown.

### GenerationConfig
Declarative configuration for spectral generation.

### ElectroSpectralSignature
Computed feature signature for spectral analysis.

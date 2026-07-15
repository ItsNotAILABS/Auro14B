# API Reference

## Top-Level Functions

### `load_record(source, record_id=None)`
Load a spectral record from various formats (dict, array, file path, DataFrame).

### `validate_record(record)`
Validate a record through 6 levels of checks. Returns `ValidationReport`.

### `normalize_record(record, method="max")`
Normalize all component amplitudes. Methods: 'max', 'l2', 'zscore'.

### `match_records(reference, candidate, matcher=None)`
Match two records. Returns `MatchResult` with composite score.

### `generate_psd(config)`
Generate PSD-compatible spectral record.

### `generate_fas(config)`
Generate FAS-compatible spectral record.

### `generate_rotdnn(config)`
Generate RotDnn multi-component spectral record.

## Core Classes

### `MultiElementRecord`
Multi-component spectral record with metadata and lineage.

### `SpectralComponent`
Single frequency-amplitude component with phase and node linkage.

### `GenerationConfig`
Configuration for spectral generation (shape, seed, blending, constraints).

### `SpectralMatcher`
Matching engine with fit/score/match/rank_matches methods.

### `MatchResult`
Match output with `.score`, `.composite_score`, `.metrics`, `.metric_breakdown`.

### `ValidationReport`
Validation output with `.is_valid`, `.errors`, `.warnings`, `.level`.

### `SpectralVectorizer`
Convert records to fixed-size embedding vectors.

### `SpectralRetriever`
Index records and perform nearest-neighbor retrieval.

## Cognitive Adapters

### `SpectralMemoryAdapter`
Convert records to memory objects for cognitive systems.

### `SpectralAttentionAdapter`
Compute attention weights from spectral features.

### `AgentStateSpectralAdapter`
Convert records to agent state vectors.

### `SpectralAnomalyAdapter`
Detect spectral anomalies relative to a baseline.

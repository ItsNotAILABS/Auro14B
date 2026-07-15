# MESIE Documentation

**Multi-Element Spectral Intelligence Engine**

## Overview

MESIE is an open-source Python framework for multi-component spectral matching, signal generation, resonance-aware embeddings, and AI-native spectral representation.

## Contents

- [Research Program](research_program.md)
- [Architecture](architecture.md)
- [API Reference](api_reference.md)
- [Bundled Datasets](data.md)
- [Cloudflare Worker API](cloudflare.md)
- [Zenodo Release Notes](zenodo_release_notes.md)

## Quick Start

```python
from mesie import load_record, validate_record, match_records

record = load_record({"record_id": "r1", "components": [...]})
report = validate_record(record)
result = match_records(reference, candidate)
```

## Installation

```bash
pip install mesie
```

For full functionality:

```bash
pip install mesie[full]
```

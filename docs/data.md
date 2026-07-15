# Bundled Datasets (v0.2.0)

See `data/` (~2.4 MB JSON). Load via:

```python
from data import load_reference_record, load_benchmark, list_references

record = load_reference_record("earthquake_psd_reference")
meta = load_benchmark("embedding_training_data")
```

`load_reference_record` normalizes `frequencies`/`amplitudes` keys to MESIE loader format.

Reference files were repaired (negative PSD amplitudes clipped to `1e-12`); all four pass validation level 6.

## Orbital-edge workflow

```bash
python scripts/orbital_edge_50d_analysis.py
```

Writes `scripts/orbital_edge_50d_report.json` — 50 days history + 50 days forecast vs earthquake anchor.
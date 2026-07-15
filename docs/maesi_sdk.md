# MAESI SDK v1.1

Unified client over MESIE + knowledge libraries + fast compute.

## Knowledge loaded

| Library | Count | Module |
|---------|-------|--------|
| Physical laws | 33+ | `mesie.sdk.physical_laws` |
| Chemical elements | 38 | `mesie.sdk.chemical_elements` |
| Biological systems | 15 | `mesie.sdk.biological_systems` |
| **Technical** | 20 | `mesie.sdk.technical_library` |
| **Research** | 24 | `mesie.sdk.research_knowledge` |

## Fast compute

`FastSpectralCompute` uses:

- Shared `SpectralVectorizer` singleton
- `batch_transform` for corpus embed
- Matrix cosine search (`embeddings @ query`) instead of per-pair `match_records` loops

```python
from mesie.sdk import MAESIClient
from data import load_reference_record, list_references

client = MAESIClient(fast=True)
refs = [load_reference_record(n) for n in list_references()]
report = client.run_full(refs, benchmark=True)
print(report.plain_summary)
```

## Run

```bash
python scripts/run_maesi_sdk.py
```

Output: `deliverables/MAESI_SDK_Run_Report.json`
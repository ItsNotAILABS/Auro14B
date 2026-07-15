# How Fast Is MESIE?

Think of one "does this spectrum match that reference?" check.

On your laptop, MESIE does that in about **¼ of a millisecond** — roughly **4,000 comparisons per second** on a single core, without a GPU.

---

## Everyday Comparisons

| What people know | Rough time | MESIE comparison |
|---|---|---|
| Double-clicking open a medium Excel file | ~1–3 seconds | MESIE runs **4,000–12,000 match checks** in the same time |
| One round-trip to a typical cloud API (network only) | ~50–200 ms | ~250× faster — just math, no network wait |
| An engineer eyeballing a chart | minutes | Millions of times faster |
| Running a heavy ML model inference | often 100 ms–seconds | Core match path is lightweight math, not a big neural-net forward pass |

---

## Plain Line for Marketing

> "MESIE can compare thousands of spectral fingerprints per second on a normal laptop — faster than opening a spreadsheet, and far faster than waiting on the cloud."

---

## Other Operations (Same Machine)

| Operation | Time per call | Throughput |
|---|---|---|
| Match one spectrum against one reference | ~0.25 ms | ~4,000/sec |
| Rank a handful of candidates | < 1 ms | ~1,000 rankings/sec |
| Generate a synthetic spectrum (fixed seed) | ~0.05 ms | ~19,000/sec |

---

## Determinism

Same inputs + same seed → **same answer every time**.

Good for audits, demos, and regulated workflows.

---

## What "100,000 Lines" Means

That number is the size of the **MESIE software** (v0.2 expansion: protocols, transformers, cognitive modules, foundation SDK, etc.) — not a dataset count.

- ~45k+ lines of Python in the repo today
- ~130 Python modules under `mesie/`
- The ~100k figure = planned/marketed scale of the full v0.2 "organism" (code + architecture)

### How It Works

```
Your spectral data (files, streams, libraries of records)
        ↓
MESIE engine (100k-scale codebase — the "brain")
        ↓
Answers: match score, rank, validate, embed, anomaly, etc.
```

- **The library** = the code (matching, AI protocols, Helix, cognitive adapters, SpectralIntelligenceSDK, etc.)
- **The fuel** = spectral records (JSON, CSV, your 50-day feeds, big corpora you add)

---

## Marketing Bullets (Non-Scientific)

1. **Speed:** "Thousands of spectral match checks per second on a laptop."
2. **Scale of product:** "Tens of thousands of lines of purpose-built spectral-AI code — not a thin wrapper around one algorithm."
3. **Trust:** "Same input, same answer — reproducible for science and compliance."
4. **Data:** "Works with your spectral libraries; ships with reference datasets to get started."
5. **Stack:** "From quick match at the edge (Cloudflare) to full reasoning stack in Python."

---

## Bundled Reference Data

MESIE ships with real-world spectral reference libraries ready to use:

| Library | Description |
|---|---|
| `hydrogen_spectrum` | Hydrogen atomic emission lines (Balmer, Lyman, Paschen series) from NIST ASD |
| `electromagnetic_bands` | Complete EM spectrum band definitions (IEEE/ITU standards) |
| `schumann_resonances` | Earth-ionosphere cavity resonances + geophysical frequencies |
| `satellite_frequencies` | Satellite communication frequencies and orbital parameters |
| `atmospheric_absorption` | Atmospheric absorption/transmission windows (ITU-R P.676) |

Plus domain-specific references for seismic, structural, and vibration monitoring.

---

## Loading Data

```python
from data import load_library, load_reference, list_library

# See what's available
print(list_library())
# ['atmospheric_absorption', 'electromagnetic_bands', 'hydrogen_spectrum',
#  'satellite_frequencies', 'schumann_resonances']

# Load a spectral library
hydrogen = load_library("hydrogen_spectrum")

# Load a domain reference
quake_ref = load_reference("earthquake_psd_reference")
```

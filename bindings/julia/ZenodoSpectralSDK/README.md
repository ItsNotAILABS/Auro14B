# ZenodoSpectralSDK

Julia SDK for accessing Zenodo datasets with MESIE transformer-based spectral intelligence.

## 🚀 Full Installation (Recommended)

The `install.sh` script handles **everything** — installs bash, git, Julia, and all SDK dependencies:

```bash
git clone https://github.com/FreddyCreates/Multi-Element-Spectral-Intelligence-Engine-MESIE-.git
cd Multi-Element-Spectral-Intelligence-Engine-MESIE-/bindings/julia/ZenodoSpectralSDK
chmod +x install.sh
./install.sh
```

This will:
1. ✅ Install/verify **bash** (auto-detects apt, dnf, brew, apk, pacman)
2. ✅ Install/verify **git**
3. ✅ Install/verify **Julia** (downloads official binary if missing)
4. ✅ Install all Julia package dependencies (HTTP, JSON, Downloads)
5. ✅ Precompile packages for fast startup
6. ✅ Run a health check to verify the SDK works

### After Installation — Launch the Research OS Terminal

```bash
julia launch.jl
```

The launcher verifies system dependencies (bash, git) are present, installs Julia packages if needed, then launches the interactive Research OS terminal.

### Quick Start (If You Already Have Julia + Git + Bash)

```bash
cd Multi-Element-Spectral-Intelligence-Engine-MESIE-/bindings/julia/ZenodoSpectralSDK
julia launch.jl
```

Or from a Julia REPL if you already have the repo:

```julia
include("bindings/julia/ZenodoSpectralSDK/launch.jl")
```

The launcher will:
1. ✅ Verify bash and git are available (offers to install if missing)
2. ✅ Auto-install all Julia dependencies (HTTP, JSON, etc.)
3. ✅ Load the full ZenodoSpectralSDK
4. ✅ Launch an interactive Research OS terminal in your Julia REPL

The Research OS provides:
- 🌐 **Zenodo Browser** — Search, discover, and download scientific datasets
- 🔬 **Spectral Analysis Lab** — Anomaly detection, resonance, classification
- 🧠 **Transformer Lab** — Embeddings, matching, fingerprinting
- 📂 **Research Workspaces** — Organize datasets, analyses, and notes
- ⭐ **Curated Datasets** — High-impact spectral/signal datasets
- ⚡ **Quick Analysis** — Instant analysis from file or manual input

## Overview

ZenodoSpectralSDK connects to [Zenodo](https://zenodo.org)'s public repository to discover, fetch, and analyze popular scientific datasets using MESIE's spectral transformer architecture. It focuses on the most-viewed/downloaded datasets relevant to spectral analysis and signal processing.

## Features

### Zenodo Client
- **Dataset Search**: Query Zenodo's REST API for datasets by keyword, community, or type
- **Popularity Ranking**: Sort datasets by views/downloads to find the most-used records
- **Record Fetching**: Retrieve full metadata and usage statistics for any Zenodo record
- **File Download**: Download dataset files directly from Zenodo

### Spectral Transformer Model
- **Spectral Tokenization**: Frequency-band windowing with 8 statistical features per token
- **Positional Encoding**: Standard sinusoidal encoding from "Attention Is All You Need"
- **Multi-Head Attention**: Configurable attention heads for spectral pattern recognition
- **Feed-Forward Network**: Position-wise FFN with ReLU activation and layer normalization
- **Embedding Generation**: L2-normalized spectral embeddings via global average pooling

### Intelligence Protocols
- **Anomaly Detection**: Combined Z-score and local deviation analysis with severity levels
- **Resonance Analysis**: Peak detection, Q-factor estimation, and target frequency alignment
- **Spectral Matching**: Transformer-enhanced similarity with composite scoring
- **Fingerprinting**: Binary spectral fingerprints for fast approximate search
- **Pattern Classification**: Rule-based spectral type classification (narrowband, harmonic, noise)

## Installation

```julia
using Pkg
Pkg.activate("bindings/julia/ZenodoSpectralSDK")
Pkg.instantiate()
```

## Quick Start

### Search Zenodo for Popular Datasets

```julia
using ZenodoSpectralSDK

client = ZenodoClient()

# Search for spectral datasets
results = search_datasets(client; query="spectral time series", size=10)

# Get curated list of popular spectral datasets
popular = POPULAR_SPECTRAL_DATASETS
for ds in popular
    println("$(ds["title"]) — Views: $(ds["estimated_views"])")
end
```

### Transformer-Based Spectral Embedding

```julia
# Create a spectral record
record = Dict(
    "frequency" => collect(0.1:0.1:100.0),
    "amplitude" => rand(1000) .* sin.(collect(0.1:0.1:100.0))
)

# Generate transformer embedding
model = SpectralTransformer(d_model=64, n_heads=4, n_layers=2)
embedding = transformer_embed(model, record["amplitude"], record["frequency"])
println("Embedding dimension: $(length(embedding))")
```

### Intelligence Protocol Analysis

```julia
intel = SpectralIntelligence(protocol_level=3)

# Full spectral analysis
result = intelligence_protocol(intel, record; action="analyze")
println("Anomalies found: $(result["anomaly_detection"]["n_anomalies"])")
println("Dominant frequency: $(result["resonance_analysis"]["dominant_frequency"]) Hz")

# Pattern classification
classification = intelligence_protocol(intel, record; action="classify")
println("Pattern type: $(classification["class"])")
```

### Spectral Matching with Transformer Embeddings

```julia
record_a = Dict("frequency" => collect(1.0:100.0), "amplitude" => sin.(collect(1.0:100.0)))
record_b = Dict("frequency" => collect(1.0:100.0), "amplitude" => sin.(collect(1.0:100.0) .+ 0.1))

match_result = spectral_match(intel, record_a, record_b)
println("Similarity: $(match_result["composite_score"])")
println("Confidence: $(match_result["confidence"])")
```

## CLI Usage

```bash
# Health check
julia cli.jl health

# List popular spectral datasets from Zenodo
julia cli.jl popular

# Search Zenodo
julia cli.jl search --query="earthquake spectra" --size=5

# Generate embedding from stdin JSON
echo '{"record":{"frequency":[1,2,3,4,5],"amplitude":[0.5,0.8,1.2,0.6,0.3]}}' | julia cli.jl embed

# Run intelligence protocol
echo '{"record":{"frequency":[1,2,3,4,5],"amplitude":[0.5,0.8,1.2,0.6,0.3]},"sub_action":"analyze"}' | julia cli.jl protocol
```

## JSON IPC Dispatch

All operations are accessible via JSON message passing for integration with Python/MESIE:

```json
{
  "action": "transformer_embed",
  "record": {
    "frequency": [1.0, 2.0, 3.0, 4.0, 5.0],
    "amplitude": [0.5, 0.8, 1.2, 0.6, 0.3]
  }
}
```

Supported actions:
- `health` — SDK status
- `search_datasets` — Search Zenodo
- `popular_datasets` — Rank by popularity
- `curated_popular` — Curated popular datasets
- `fetch_record` — Fetch record metadata
- `transformer_embed` — Generate transformer embedding
- `match` — Spectral similarity matching
- `anomaly_detect` — Anomaly detection
- `resonance` — Resonance analysis
- `fingerprint` — Spectral fingerprint
- `protocol` — Intelligence protocol execution

## Popular Zenodo Datasets (Curated)

| Dataset | Record ID | Use Case |
|---------|-----------|----------|
| IEEEPPG | 3902710 | PPG signal processing, biomedical spectral analysis |
| PSML Power Grid | 5130612 | Power spectral density, grid frequency monitoring |
| CESNET-TimeSeries24 | 13382427 | Network spectral anomaly detection |
| Strong Motion Seismic | 1161203 | Response spectra, PSD analysis |
| EEG Motor Imagery | 7714714 | Brain spectral mapping, coherence analysis |
| Sentinel-2 Multispectral | 6513152 | Multi-band spectral classification |
| CORD-19 | 4737174 | Spectral text embeddings |
| Zenodo Analytics | 3553943 | Usage pattern analysis |

## Architecture

```
ZenodoSpectralSDK/
├── Project.toml           # Julia package manifest
├── launch.jl              # One-shot launcher (auto-installs + starts REPL)
├── cli.jl                 # CLI entry point
├── README.md              # This file
├── test/                  # Test suite
│   └── runtests.jl
└── src/
    ├── ZenodoSpectralSDK.jl  # Module definition
    ├── zenodo_client.jl      # Zenodo REST API client
    ├── transformer.jl        # Spectral transformer model
    ├── intelligence.jl       # Intelligence protocols + dispatch
    └── repl.jl               # Interactive Research OS terminal
```

## Integration with MESIE

This SDK integrates with the broader MESIE ecosystem:
- **Python core**: Use via JSON IPC or `juliacall` for direct function calls
- **MESIEPolyglot**: Compatible data formats (same record dict structure)
- **Edge API**: Transformer embeddings can be cached and served via Cloudflare Workers
- **Desktop App**: Results can be visualized in the MESIE Electron application

## License

Apache-2.0 — Same as the MESIE project.

"""
ZenodoSpectralSDK — Julia SDK for Zenodo dataset access with MESIE spectral intelligence.

Provides:
- Zenodo REST API client (search, fetch, rank by popularity)
- Spectral transformer model pipeline (tokenize → multi-head attention → embed)
- Spectral intelligence protocols (matching, fingerprinting, anomaly detection)
- Integration with MESIEPolyglot for core spectral computations
"""
module ZenodoSpectralSDK

using HTTP
using JSON
using LinearAlgebra
using Statistics
using Random
using Downloads

export ZenodoClient, search_datasets, fetch_record, list_popular_datasets
export download_dataset_file, get_record_stats, POPULAR_SPECTRAL_DATASETS
export SpectralTransformer, tokenize_spectrum, transformer_encode, spectral_attention
export transformer_embed, transformer_pipeline
export SpectralIntelligence, detect_anomalies, compute_resonance, spectral_match
export spectral_fingerprint, intelligence_protocol
export dispatch_action, health_check

include("zenodo_client.jl")
include("transformer.jl")
include("intelligence.jl")

end  # module

"""
Test suite for ZenodoSpectralSDK.

Tests cover:
- Zenodo client configuration and record parsing
- Spectral tokenization and transformer pipeline
- Intelligence protocols (anomaly, resonance, matching, fingerprint)
- JSON dispatch routing
"""

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "ZenodoSpectralSDK.jl"))
using .ZenodoSpectralSDK
using .ZenodoSpectralSDK: POPULAR_SPECTRAL_DATASETS, positional_encoding, _softmax_rows
using Test
using LinearAlgebra
using Statistics

@testset "ZenodoSpectralSDK" begin

    @testset "ZenodoClient" begin
        client = ZenodoClient()
        @test client.base_url == "https://zenodo.org/api"
        @test client.access_token == ""
        @test client.timeout == 30

        client_auth = ZenodoClient(access_token="test_token", timeout=60)
        @test client_auth.timeout == 60
    end

    @testset "Popular Datasets Curated" begin
        @test length(POPULAR_SPECTRAL_DATASETS) >= 5
        for ds in POPULAR_SPECTRAL_DATASETS
            @test haskey(ds, "record_id")
            @test haskey(ds, "title")
            @test haskey(ds, "estimated_views")
            @test ds["estimated_views"] > 0
        end
    end

    @testset "Spectral Tokenization" begin
        freq = collect(1.0:0.1:100.0)
        amp = sin.(freq) .+ 0.5

        tokens = tokenize_spectrum(amp, freq; n_bands=16, d_model=64)
        @test size(tokens) == (16, 64)

        # Tokens should be L2 normalized (either ~1.0 or 0.0 for empty bands)
        for i in 1:16
            nrm = norm(tokens[i, :])
            @test (isapprox(nrm, 1.0; atol=0.01) || isapprox(nrm, 0.0; atol=0.01))
        end

        # Empty input
        empty_tokens = tokenize_spectrum(Float64[], Float64[]; n_bands=8, d_model=32)
        @test size(empty_tokens) == (8, 32)
        @test all(empty_tokens .== 0.0)
    end

    @testset "Positional Encoding" begin
        pe = ZenodoSpectralSDK.positional_encoding(10, 64)
        @test size(pe) == (10, 64)
        # Values should be bounded [-1, 1]
        @test all(-1.0 .<= pe .<= 1.0)
    end

    @testset "Spectral Attention" begin
        Q = randn(8, 16)
        K = randn(8, 16)
        V = randn(8, 16)

        output = spectral_attention(Q, K, V)
        @test size(output) == (8, 16)
        # Output should be finite
        @test all(isfinite.(output))
    end

    @testset "SpectralTransformer" begin
        model = SpectralTransformer(d_model=32, n_heads=2, n_layers=1, n_bands=8)
        @test model.d_model == 32
        @test model.n_heads == 2
        @test model.n_layers == 1
        @test model.n_bands == 8
    end

    @testset "Transformer Encode" begin
        model = SpectralTransformer(d_model=32, n_heads=2, n_layers=1, n_bands=8)
        tokens = randn(8, 32)

        encoded = transformer_encode(model, tokens)
        @test size(encoded) == (8, 32)
        @test all(isfinite.(encoded))
    end

    @testset "Transformer Embed" begin
        model = SpectralTransformer(d_model=32, n_heads=2, n_layers=1, n_bands=8)
        freq = collect(1.0:100.0)
        amp = sin.(freq .* 0.1) .+ rand(100) .* 0.1

        embedding = transformer_embed(model, amp, freq)
        @test length(embedding) == 32
        @test norm(embedding) ≈ 1.0 atol=0.01  # L2 normalized
        @test all(isfinite.(embedding))
    end

    @testset "Transformer Pipeline (Dict)" begin
        record = Dict(
            "frequency" => collect(1.0:50.0),
            "amplitude" => rand(50)
        )

        result = transformer_pipeline(record)
        @test result["runtime"] == "julia"
        @test result["model"] == "SpectralTransformer"
        @test length(result["embedding"]) == 64  # default d_model
        @test result["n_points"] == 50

        # Empty record
        empty_result = transformer_pipeline(Dict())
        @test length(empty_result["embedding"]) == 64
        @test empty_result["n_tokens"] == 0
    end

    @testset "Anomaly Detection" begin
        intel = SpectralIntelligence()

        # Normal signal with injected spike
        amp = ones(100)
        amp[50] = 20.0  # Inject anomaly

        result = detect_anomalies(intel, amp)
        @test result["n_anomalies"] > 0
        @test 50 in result["anomalies"]
        @test result["runtime"] == "julia"

        # Clean signal (very few or no anomalies)
        clean_amp = ones(100) .+ randn(100) .* 0.01
        clean_result = detect_anomalies(intel, clean_amp)
        @test clean_result["n_anomalies"] <= 3  # Allow a few false positives from random noise
    end

    @testset "Resonance Analysis" begin
        intel = SpectralIntelligence()

        # Create signal with clear peak
        freq = collect(1.0:0.1:100.0)
        amp = zeros(length(freq))
        # Add Gaussian peak at 50 Hz
        for (i, f) in enumerate(freq)
            amp[i] = exp(-((f - 50.0)^2) / 5.0)
        end

        result = compute_resonance(intel, freq, amp)
        @test result["n_peaks"] > 0
        @test result["runtime"] == "julia"
        # Dominant frequency should be near 50
        @test abs(result["dominant_frequency"] - 50.0) < 2.0

        # Target alignment
        result_target = compute_resonance(intel, freq, amp;
            target_frequencies=[50.0, 25.0])
        @test length(result_target["target_alignment"]) == 2
        # Alignment with 50 Hz should be high
        @test result_target["target_alignment"][1] > 0.5
    end

    @testset "Spectral Match" begin
        intel = SpectralIntelligence()

        record_a = Dict("frequency" => collect(1.0:100.0),
                        "amplitude" => sin.(collect(1.0:100.0)))
        record_b = Dict("frequency" => collect(1.0:100.0),
                        "amplitude" => sin.(collect(1.0:100.0)))  # Same signal

        result = spectral_match(intel, record_a, record_b)
        @test result["composite_score"] > 0.5
        @test result["runtime"] == "julia"
        @test haskey(result, "embedding_similarity")
        @test haskey(result, "confidence")

        # Different signals should have lower score
        record_c = Dict("frequency" => collect(1.0:100.0),
                        "amplitude" => rand(100))
        result_diff = spectral_match(intel, record_a, record_c)
        @test result_diff["composite_score"] < result["composite_score"]
    end

    @testset "Spectral Fingerprint" begin
        intel = SpectralIntelligence()

        record = Dict("frequency" => collect(1.0:50.0),
                      "amplitude" => rand(50))

        result = spectral_fingerprint(intel, record)
        @test haskey(result, "fingerprint")
        @test haskey(result, "hash")
        @test result["method"] == "transformer_enhanced"
        @test all(fp -> fp == 0 || fp == 1, result["fingerprint"])
    end

    @testset "Intelligence Protocol" begin
        intel = SpectralIntelligence(protocol_level=3)

        record = Dict("frequency" => collect(1.0:100.0),
                      "amplitude" => sin.(collect(1.0:100.0)) .+ 0.5)

        # Analyze
        result = intelligence_protocol(intel, record; action="analyze")
        @test haskey(result, "anomaly_detection")
        @test haskey(result, "resonance_analysis")
        @test haskey(result, "embedding")
        @test result["protocol_level"] == 3

        # Classify
        classify_result = intelligence_protocol(intel, record; action="classify")
        @test haskey(classify_result, "class")
        @test classify_result["class"] in ["narrowband", "harmonic", "colored_noise", "white_noise"]

        # Summarize
        summary = intelligence_protocol(intel, record; action="summarize")
        @test haskey(summary, "frequency_range")
        @test haskey(summary, "total_energy")
    end

    @testset "Dispatch Action" begin
        # Health check
        result = dispatch_action(Dict("action" => "health"))
        @test result["ok"] == true
        @test result["data"]["sdk"] == "ZenodoSpectralSDK"

        # Curated popular
        result = dispatch_action(Dict("action" => "curated_popular"))
        @test result["ok"] == true
        @test haskey(result["data"], "datasets")

        # Transformer embed
        result = dispatch_action(Dict(
            "action" => "transformer_embed",
            "record" => Dict("frequency" => collect(1.0:20.0), "amplitude" => rand(20))
        ))
        @test result["ok"] == true
        @test result["data"]["runtime"] == "julia"

        # Unknown action
        result = dispatch_action(Dict("action" => "unknown_xyz"))
        @test result["ok"] == false
    end

end

println("\n✅ All ZenodoSpectralSDK tests passed!")

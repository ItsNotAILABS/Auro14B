"""
Spectral Intelligence Protocols — Autonomous reasoning and analysis for spectral data.

Implements MESIE's intelligence protocol levels adapted for Julia:
- Level 1: Passive observation (validation, basic stats)
- Level 2: Reactive analysis (anomaly detection, threshold alerts)
- Level 3: Deliberative reasoning (pattern matching, resonance analysis)
- Level 4: Autonomous intelligence (self-directed analysis, fingerprinting)

Integrates with the SpectralTransformer for embedding-based intelligence.
"""

# --- Configuration ---

"""
    SpectralIntelligence

Configuration for the spectral intelligence system.

# Fields
- `transformer::SpectralTransformer`: Underlying transformer model
- `anomaly_threshold::Float64`: Z-score threshold for anomaly detection
- `resonance_tolerance::Float64`: Frequency matching tolerance (Hz)
- `protocol_level::Int`: Intelligence protocol level (1-4)
"""
struct SpectralIntelligence
    transformer::SpectralTransformer
    anomaly_threshold::Float64
    resonance_tolerance::Float64
    protocol_level::Int

    function SpectralIntelligence(;
        transformer::Union{Nothing,SpectralTransformer}=nothing,
        anomaly_threshold::Float64=2.5,
        resonance_tolerance::Float64=0.05,
        protocol_level::Int=3
    )
        if transformer === nothing
            transformer = SpectralTransformer()
        end
        new(transformer, anomaly_threshold, resonance_tolerance, protocol_level)
    end
end


"""
    detect_anomalies(intel::SpectralIntelligence, amplitude::Vector{Float64};
                     window_size::Int=32) -> Dict

Detect spectral anomalies using statistical and transformer-based methods.

# Methods
1. Z-score detection: Points exceeding the anomaly threshold
2. Moving average deviation: Points deviating from local trend
3. Spectral energy spikes: Sudden energy concentration changes

# Returns
Dict with anomaly indices, scores, severity levels, and summary statistics.
"""
function detect_anomalies(intel::SpectralIntelligence, amplitude::Vector{Float64};
    window_size::Int=32)

    n = length(amplitude)
    if n < 4
        return Dict(
            "anomalies" => Int[],
            "scores" => Float64[],
            "severity" => String[],
            "n_anomalies" => 0,
            "method" => "combined",
            "runtime" => "julia",
        )
    end

    μ = mean(amplitude)
    σ = std(amplitude) + 1e-12

    anomaly_indices = Int[]
    anomaly_scores = Float64[]
    severities = String[]

    for i in 1:n
        # Z-score
        z = abs(amplitude[i] - μ) / σ

        # Local deviation (moving window)
        win_start = max(1, i - div(window_size, 2))
        win_end = min(n, i + div(window_size, 2))
        local_μ = mean(amplitude[win_start:win_end])
        local_dev = abs(amplitude[i] - local_μ) / (std(amplitude[win_start:win_end]) + 1e-12)

        # Combined anomaly score
        score = 0.6 * z + 0.4 * local_dev

        if score > intel.anomaly_threshold
            push!(anomaly_indices, i)
            push!(anomaly_scores, score)

            severity = if score > intel.anomaly_threshold * 2.0
                "critical"
            elseif score > intel.anomaly_threshold * 1.5
                "high"
            else
                "moderate"
            end
            push!(severities, severity)
        end
    end

    return Dict(
        "anomalies" => anomaly_indices,
        "scores" => anomaly_scores,
        "severity" => severities,
        "n_anomalies" => length(anomaly_indices),
        "threshold" => intel.anomaly_threshold,
        "method" => "combined_zscore_localdev",
        "runtime" => "julia",
    )
end


"""
    compute_resonance(intel::SpectralIntelligence, frequency::Vector{Float64},
                      amplitude::Vector{Float64};
                      target_frequencies::Vector{Float64}=Float64[]) -> Dict

Compute resonance analysis for spectral data.

Identifies natural frequencies (peaks), computes Q-factors, and optionally
checks alignment with target frequencies.

# Returns
Dict with resonant peaks, Q-factors, damping ratios, and target alignment.
"""
function compute_resonance(intel::SpectralIntelligence, frequency::Vector{Float64},
    amplitude::Vector{Float64};
    target_frequencies::Vector{Float64}=Float64[])

    n = length(amplitude)
    if n < 3
        return Dict(
            "peaks" => Dict[],
            "n_peaks" => 0,
            "target_alignment" => Float64[],
            "runtime" => "julia",
        )
    end

    # Find peaks (local maxima)
    peaks = Dict[]
    for i in 2:(n-1)
        if amplitude[i] > amplitude[i-1] && amplitude[i] > amplitude[i+1]
            # Estimate Q-factor (quality factor)
            half_power = amplitude[i] / sqrt(2.0)

            # Find half-power bandwidth
            left_idx = i
            right_idx = i
            for j in (i-1):-1:1
                if amplitude[j] <= half_power
                    left_idx = j
                    break
                end
            end
            for j in (i+1):n
                if amplitude[j] <= half_power
                    right_idx = j
                    break
                end
            end

            f_center = frequency[i]
            bandwidth = frequency[min(right_idx, n)] - frequency[max(left_idx, 1)]
            q_factor = bandwidth > 0 ? f_center / bandwidth : 0.0
            damping_ratio = q_factor > 0 ? 1.0 / (2.0 * q_factor) : 1.0

            push!(peaks, Dict(
                "index" => i,
                "frequency" => f_center,
                "amplitude" => amplitude[i],
                "q_factor" => q_factor,
                "damping_ratio" => damping_ratio,
                "bandwidth" => bandwidth,
            ))
        end
    end

    # Sort peaks by amplitude (descending)
    sort!(peaks; by=p -> p["amplitude"], rev=true)

    # Check target frequency alignment
    target_alignment = Float64[]
    if !isempty(target_frequencies) && !isempty(peaks)
        for tf in target_frequencies
            # Find closest peak
            min_dist = Inf
            for p in peaks
                dist = abs(p["frequency"] - tf)
                if dist < min_dist
                    min_dist = dist
                end
            end
            # Alignment score: 1.0 = perfect match, 0.0 = far
            alignment = exp(-min_dist / (intel.resonance_tolerance + 1e-12))
            push!(target_alignment, clamp(alignment, 0.0, 1.0))
        end
    end

    return Dict(
        "peaks" => peaks[1:min(10, length(peaks))],  # Top 10 peaks
        "n_peaks" => length(peaks),
        "target_alignment" => target_alignment,
        "dominant_frequency" => isempty(peaks) ? 0.0 : peaks[1]["frequency"],
        "mean_q_factor" => isempty(peaks) ? 0.0 : mean(p["q_factor"] for p in peaks),
        "runtime" => "julia",
    )
end


"""
    spectral_match(intel::SpectralIntelligence, record_a::Dict, record_b::Dict) -> Dict

Advanced spectral matching using transformer embeddings and traditional metrics.

Computes:
1. Cosine similarity between transformer embeddings
2. Traditional RMSE and Pearson correlation
3. Resonance alignment score
4. Composite intelligence score

# Returns
Dict with composite score, individual metrics, and confidence level.
"""
function spectral_match(intel::SpectralIntelligence, record_a::Dict, record_b::Dict)
    # Get transformer embeddings
    result_a = transformer_pipeline(record_a; model=intel.transformer)
    result_b = transformer_pipeline(record_b; model=intel.transformer)

    emb_a = result_a["embedding"]
    emb_b = result_b["embedding"]

    # Cosine similarity of embeddings
    cosine_emb = dot(emb_a, emb_b) / (norm(emb_a) * norm(emb_b) + 1e-12)

    # Extract raw amplitudes for traditional metrics
    amp_a = _extract_amplitude(record_a)
    amp_b = _extract_amplitude(record_b)

    n = min(length(amp_a), length(amp_b))

    if n == 0
        return Dict(
            "composite_score" => 0.0,
            "embedding_similarity" => cosine_emb,
            "metrics" => Dict("cosine" => 0.0, "rmse" => 1.0, "pearson" => 0.0),
            "confidence" => "low",
            "runtime" => "julia",
        )
    end

    va, vb = amp_a[1:n], amp_b[1:n]

    # Traditional metrics
    cosine_raw = dot(va, vb) / (norm(va) * norm(vb) + 1e-12)
    rmse = sqrt(mean((va .- vb) .^ 2))
    pearson = std(va) > 0 && std(vb) > 0 ? cor(va, vb) : 0.0

    # Composite score (weighted combination)
    # Embedding similarity gets higher weight as it captures semantic features
    rmse_score = 1.0 / (1.0 + rmse)
    pearson_score = (pearson + 1.0) / 2.0

    composite = clamp(
        0.35 * cosine_emb +
        0.25 * cosine_raw +
        0.20 * rmse_score +
        0.20 * pearson_score,
        0.0, 1.0
    )

    # Confidence based on data quality
    confidence = if n > 100 && composite > 0.7
        "high"
    elseif n > 50
        "medium"
    else
        "low"
    end

    return Dict(
        "composite_score" => composite,
        "embedding_similarity" => cosine_emb,
        "metrics" => Dict(
            "cosine_embedding" => cosine_emb,
            "cosine_raw" => cosine_raw,
            "rmse" => rmse,
            "pearson" => pearson,
        ),
        "n_compared" => n,
        "confidence" => confidence,
        "protocol_level" => intel.protocol_level,
        "runtime" => "julia",
        "model" => "SpectralTransformer+Intelligence",
    )
end


"""
    spectral_fingerprint(intel::SpectralIntelligence, record::Dict;
                         resolution::Int=32) -> Dict

Generate a spectral fingerprint using transformer-enhanced encoding.

Combines traditional binning with transformer attention patterns
to create a compact but semantically-rich fingerprint.

# Returns
Dict with binary fingerprint, hash, and metadata.
"""
function spectral_fingerprint(intel::SpectralIntelligence, record::Dict;
    resolution::Int=32)

    # Get transformer embedding
    result = transformer_pipeline(record; model=intel.transformer)
    embedding = result["embedding"]

    # Quantize embedding to binary fingerprint
    med = median(embedding)
    fingerprint = UInt8[e > med ? 1 : 0 for e in embedding]

    # Also compute traditional fingerprint from raw amplitude
    amp = _extract_amplitude(record)
    if !isempty(amp)
        n = length(amp)
        bin_size = max(1, div(n, resolution))
        bins = Float64[]
        for i in 1:resolution
            start_idx = (i - 1) * bin_size + 1
            end_idx = min(i * bin_size, n)
            if start_idx > n
                push!(bins, 0.0)
            else
                push!(bins, mean(amp[start_idx:end_idx]))
            end
        end
        trad_med = median(bins)
        traditional_fp = UInt8[b > trad_med ? 1 : 0 for b in bins]
    else
        traditional_fp = zeros(UInt8, resolution)
    end

    # Compute hash from fingerprint
    fp_hash = _simple_hash(fingerprint)

    return Dict(
        "fingerprint" => fingerprint,
        "traditional_fingerprint" => traditional_fp,
        "hash" => fp_hash,
        "resolution" => length(fingerprint),
        "method" => "transformer_enhanced",
        "runtime" => "julia",
    )
end


"""
    intelligence_protocol(intel::SpectralIntelligence, record::Dict;
                          action::String="analyze") -> Dict

Execute a spectral intelligence protocol on a record.

# Actions
- "analyze": Full analysis (validation + anomaly detection + resonance + embedding)
- "classify": Classify the spectral pattern type
- "summarize": Generate a compact intelligence summary

# Returns
Dict with protocol results based on the chosen action.
"""
function intelligence_protocol(intel::SpectralIntelligence, record::Dict;
    action::String="analyze")

    freq = _extract_frequency(record)
    amp = _extract_amplitude(record)

    if action == "analyze"
        # Full analysis pipeline
        anomalies = detect_anomalies(intel, amp)
        resonance = compute_resonance(intel, freq, amp)
        embedding_result = transformer_pipeline(record; model=intel.transformer)

        return Dict(
            "protocol_level" => intel.protocol_level,
            "action" => "analyze",
            "anomaly_detection" => anomalies,
            "resonance_analysis" => resonance,
            "embedding" => embedding_result["embedding"],
            "n_points" => length(amp),
            "spectral_energy" => sum(amp .^ 2),
            "peak_frequency" => isempty(freq) ? 0.0 : freq[argmax(amp)],
            "runtime" => "julia",
        )

    elseif action == "classify"
        # Pattern classification based on spectral features
        if isempty(amp)
            return Dict("class" => "empty", "confidence" => 0.0, "runtime" => "julia")
        end

        energy = sum(amp .^ 2)
        peak_ratio = maximum(amp) / (mean(amp) + 1e-12)
        spectral_flatness = exp(mean(log.(abs.(amp) .+ 1e-12))) / (mean(abs.(amp)) + 1e-12)

        # Simple rule-based classification
        class = if peak_ratio > 10.0
            "narrowband"
        elseif spectral_flatness > 0.8
            "white_noise"
        elseif spectral_flatness > 0.4
            "colored_noise"
        else
            "harmonic"
        end

        return Dict(
            "class" => class,
            "confidence" => clamp(1.0 - spectral_flatness, 0.0, 1.0),
            "features" => Dict(
                "energy" => energy,
                "peak_ratio" => peak_ratio,
                "spectral_flatness" => spectral_flatness,
            ),
            "runtime" => "julia",
        )

    elseif action == "summarize"
        # Compact intelligence summary
        embedding_result = transformer_pipeline(record; model=intel.transformer)

        return Dict(
            "n_points" => length(amp),
            "frequency_range" => isempty(freq) ? [0.0, 0.0] : [freq[1], freq[end]],
            "amplitude_range" => isempty(amp) ? [0.0, 0.0] : [minimum(amp), maximum(amp)],
            "mean_amplitude" => isempty(amp) ? 0.0 : mean(amp),
            "total_energy" => sum(amp .^ 2),
            "embedding_norm" => norm(embedding_result["embedding"]),
            "runtime" => "julia",
        )
    else
        return Dict("error" => "unknown action: $action", "runtime" => "julia")
    end
end


# --- Dispatch for JSON IPC ---

"""
    dispatch_action(request::Dict) -> Dict

Route a JSON request to the appropriate function based on the "action" field.

Supports all ZenodoSpectralSDK operations via JSON message passing.
"""
function dispatch_action(request::Dict)
    action = get(request, "action", "health")

    data = if action == "health"
        health_check()
    elseif action == "search_datasets"
        client = ZenodoClient(; access_token=get(request, "token", ""))
        search_datasets(client;
            query=get(request, "query", ""),
            size=get(request, "size", 10),
            sort=get(request, "sort", "mostrecent"),
        )
    elseif action == "popular_datasets"
        client = ZenodoClient(; access_token=get(request, "token", ""))
        query = get(request, "query", "")
        top_n = get(request, "top_n", 10)
        Dict("datasets" => list_popular_datasets(client; query=query, top_n=top_n))
    elseif action == "curated_popular"
        Dict("datasets" => POPULAR_SPECTRAL_DATASETS, "source" => "curated")
    elseif action == "fetch_record"
        client = ZenodoClient(; access_token=get(request, "token", ""))
        fetch_record(client, get(request, "record_id", 0))
    elseif action == "transformer_embed"
        record = get(request, "record", Dict())
        transformer_pipeline(record)
    elseif action == "match"
        intel = SpectralIntelligence()
        a = get(request, "record_a", get(request, "record", Dict()))
        b = get(request, "record_b", Dict())
        spectral_match(intel, a, b)
    elseif action == "anomaly_detect"
        intel = SpectralIntelligence()
        amp = Float64.(get(request, "amplitude", Float64[]))
        detect_anomalies(intel, amp)
    elseif action == "resonance"
        intel = SpectralIntelligence()
        freq = Float64.(get(request, "frequency", Float64[]))
        amp = Float64.(get(request, "amplitude", Float64[]))
        targets = Float64.(get(request, "target_frequencies", Float64[]))
        compute_resonance(intel, freq, amp; target_frequencies=targets)
    elseif action == "fingerprint"
        intel = SpectralIntelligence()
        record = get(request, "record", Dict())
        spectral_fingerprint(intel, record)
    elseif action == "protocol"
        intel = SpectralIntelligence(; protocol_level=get(request, "level", 3))
        record = get(request, "record", Dict())
        sub_action = get(request, "sub_action", "analyze")
        intelligence_protocol(intel, record; action=sub_action)
    else
        Dict("error" => "unsupported action: $action")
    end

    return Dict("ok" => !haskey(data, "error"), "data" => data)
end


"""
    health_check() -> Dict

Return runtime health status for the ZenodoSpectralSDK.
"""
function health_check()
    return Dict(
        "status" => "ok",
        "runtime" => "julia",
        "sdk" => "ZenodoSpectralSDK",
        "version" => "0.1.0",
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(),
        "capabilities" => [
            "zenodo_search",
            "zenodo_fetch",
            "popularity_ranking",
            "transformer_embed",
            "spectral_match",
            "anomaly_detection",
            "resonance_analysis",
            "spectral_fingerprint",
            "intelligence_protocol",
        ],
    )
end


# --- Internal helpers ---

"""
    _extract_amplitude(record::Dict) -> Vector{Float64}

Extract amplitude array from a spectral record dict.
"""
function _extract_amplitude(record::Dict)
    comps = get(record, "components", nothing)
    if comps !== nothing && length(comps) > 0
        return Float64.(get(comps[1], "amplitude", Float64[]))
    end
    return Float64.(get(record, "amplitude", Float64[]))
end


"""
    _extract_frequency(record::Dict) -> Vector{Float64}

Extract frequency array from a spectral record dict.
"""
function _extract_frequency(record::Dict)
    comps = get(record, "components", nothing)
    if comps !== nothing && length(comps) > 0
        return Float64.(get(comps[1], "frequency", Float64[]))
    end
    return Float64.(get(record, "frequency", Float64[]))
end


"""
    _simple_hash(fingerprint::Vector{UInt8}) -> String

Compute a simple hash string from a binary fingerprint.
"""
function _simple_hash(fingerprint::Vector{UInt8})
    # Convert binary array to hex string
    n = length(fingerprint)
    bytes = UInt8[]
    for i in 1:8:n
        byte = UInt8(0)
        for j in 0:min(7, n - i)
            if fingerprint[i+j] == 1
                byte |= UInt8(1) << j
            end
        end
        push!(bytes, byte)
    end
    return bytes2hex(bytes)
end
